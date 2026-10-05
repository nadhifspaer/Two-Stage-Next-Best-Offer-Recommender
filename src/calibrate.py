# isotonic calibration and calibration metrics 
import math
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

# fixed so the customer split is reproducible run to run, matching
# LGBM_SEED in src/rank.py and NEGATIVE_SAMPLING_SEED in src/dataset.py
CALIBRATION_SPLIT_SEED = 42

# customer_id hash bucket ranges, out of 100: two disjoint 20% subsets, the
# remaining 60% unused for calibration in this step
FIT_BUCKET_LOWER = 0
FIT_BUCKET_UPPER = 20
EVAL_BUCKET_LOWER = 20
EVAL_BUCKET_UPPER = 40

N_QUANTILE_BINS = 10


def customer_bucket(customer_ids: pl.Series, seed: int = CALIBRATION_SPLIT_SEED) -> pl.Series:
    """Deterministic hash of customer_id into [0, 100), splitting validation customers into disjoint fit/eval subsets;
    a customer's whole candidate set lands in exactly one bucket."""
    return (customer_ids.hash(seed=seed) % 100).cast(pl.Int32).alias("bucket")


def fit_bucket_mask(bucket: pl.Series) -> pl.Series:
    return (bucket >= FIT_BUCKET_LOWER) & (bucket < FIT_BUCKET_UPPER)


def eval_bucket_mask(bucket: pl.Series) -> pl.Series:
    return (bucket >= EVAL_BUCKET_LOWER) & (bucket < EVAL_BUCKET_UPPER)


def fit_calibrator(y_true: np.ndarray, raw_prob: np.ndarray) -> IsotonicRegression:
    """Fits isotonic regression mapping the classifier's raw output to a calibrated probability, on unsampled candidates only.
    The classifier saw a 9.10% positive rate, the real pool is about 0.02%, so a sampled fit inflates p_purchase by a factor in the hundreds."""
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(raw_prob, y_true)
    return calibrator


def apply_calibration(calibrator: IsotonicRegression, raw_prob: np.ndarray) -> np.ndarray:
    return calibrator.predict(raw_prob)


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    # float64 and fsum: exact, independent of row order
    squared_error = (y_prob.astype(np.float64) - y_true.astype(np.float64)) ** 2
    return math.fsum(squared_error) / len(squared_error)


def brier_skill_score(brier: float, base_rate: float) -> float:
    """Skill relative to a constant predictor at the observed base rate: 1.0 perfect, 0.0 no better than the constant, negative worse.
    Raw Brier is uninformative at a 0.02% base rate, since a constant predictor already scores near zero."""
    reference_brier = base_rate * (1 - base_rate)
    if reference_brier == 0:
        return float("nan")
    return 1 - brier / reference_brier


def _lexical_rank(series: pl.Series) -> np.ndarray:
    # rank of each value by its string, not by categorical physical code
    uniq = series.unique()
    order = uniq.cast(pl.Utf8).arg_sort().to_numpy()
    rank_of = np.empty(len(uniq), dtype=np.int64)
    rank_of[order] = np.arange(len(uniq))
    lookup = pl.DataFrame({"k": uniq, "r": rank_of})
    return series.to_frame("k").join(lookup, on="k", how="left", maintain_order="left")["r"].to_numpy()


def quantile_sort_order(y_prob: np.ndarray, keys: pl.DataFrame) -> np.ndarray:
    """Row order by (y_prob, customer_id, article_id), all ascending; the id tiebreak makes the order independent of input row order
    across tied probabilities, which isotonic output has in long runs."""
    return np.lexsort((_lexical_rank(keys["article_id"]), _lexical_rank(keys["customer_id"]), y_prob))


def quantile_bin_index(y_prob: np.ndarray, keys: pl.DataFrame, n_bins: int = N_QUANTILE_BINS) -> np.ndarray:
    n = len(y_prob)
    bin_idx = np.empty(n, dtype=np.int32)
    bin_idx[quantile_sort_order(y_prob, keys)] = np.arange(n) * n_bins // n
    return bin_idx


def quantile_reliability_table(
    y_true: np.ndarray, y_prob: np.ndarray, keys: pl.DataFrame, n_bins: int = N_QUANTILE_BINS
) -> pl.DataFrame:
    """Quantile bins of predicted probability with equal row count per bin, since the distribution is heavily skewed toward zero;
    reports predicted mean, observed rate and row count per bin. keys holds customer_id and article_id for the tie order."""
    n = len(y_prob)
    order = quantile_sort_order(y_prob, keys)
    # rows aggregated in sorted order, so float sums do not depend on input row order
    df = pl.DataFrame(
        {
            "y_true": y_true[order],
            "y_prob": y_prob[order],
            "bin": (np.arange(n) * n_bins // n).astype(np.int32),
        }
    )
    return (
        df.group_by("bin", maintain_order=True)
        .agg(
            pl.len().alias("n"),
            pl.col("y_prob").mean().alias("predicted_mean"),
            pl.col("y_true").mean().alias("observed_rate"),
        )
        .sort("bin")
    )


def save_calibrator(calibrator: IsotonicRegression, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrator, path)


def load_calibrator(path: Path) -> IsotonicRegression:
    return joblib.load(path)
