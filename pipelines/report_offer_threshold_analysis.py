# Reporting only: compares calibrated eval-subset probabilities to the expected_value > 0 break-even over C_VALUES.
import src.env_config  # noqa: F401 - must stay the first import: sets OPENBLAS_NUM_THREADS before any numeric library loads

import json
from pathlib import Path

import polars as pl

import src.calibrate as calibrate
import src.dataset as dataset

DATA_DIR = Path("data/processed")
SCORED_DIR = DATA_DIR / "calibration" / "scored"
CALIBRATOR_PATH = DATA_DIR / "models" / "calibrator.joblib"
OUTPUT_PATH = DATA_DIR / "calibration" / "_offer_threshold_report.json"

QUANTILES = [0.5, 0.9, 0.99, 0.999, 0.9999]
C_VALUES = [0.02, 0.01, 0.005, 0.001, 0.0005, 0.0001]
MARGIN_RATE = 0.5  # stated assumption


def _quantile_summary(values: pl.Series) -> dict:
    summary = {f"p{q * 100:g}": values.quantile(q, interpolation="linear") for q in QUANTILES}
    summary["max"] = values.max()
    return summary


def trailing_article_price(transactions: pl.LazyFrame, trailing_start, window_start) -> pl.DataFrame:
    # per-article mean price in [trailing_start, window_start), default engine only (no float reduction under streaming)
    return (
        transactions.filter((pl.col("t_dat") >= trailing_start) & (pl.col("t_dat") < window_start))
        .group_by("article_id")
        .agg(pl.col("price").mean().alias("trailing_4w_price"))
        .sort("article_id")
        .collect()
    )


def run() -> dict:
    pl.enable_string_cache()

    calibrator = calibrate.load_calibrator(CALIBRATOR_PATH)

    scored = pl.concat([pl.read_parquet(p) for p in sorted(SCORED_DIR.glob("part-*.parquet"))])
    eval_df = scored.filter(calibrate.eval_bucket_mask(scored["bucket"]))
    calibrated = calibrate.apply_calibration(calibrator, eval_df["raw_prob"].to_numpy())
    eval_df = eval_df.with_columns(pl.Series("calibrated_prob", calibrated))

    validation_window = next(w for w in dataset.label_windows() if w["split"] == "validation")
    window_start = validation_window["window_start"]
    trailing_start = window_start - dataset.timedelta(weeks=4)

    transactions = pl.scan_parquet(DATA_DIR / "transactions_train.parquet").with_columns(
        pl.col("customer_id", "article_id").cast(pl.Categorical)
    )
    trailing_price = trailing_article_price(transactions, trailing_start, window_start)
    median_price = float(trailing_price["trailing_4w_price"].median())

    eval_df = eval_df.join(trailing_price, on="article_id", how="left")

    candidate_prob_quantiles = _quantile_summary(eval_df["calibrated_prob"])

    per_customer_max = eval_df.group_by("customer_id").agg(
        pl.col("calibrated_prob").max().alias("max_calibrated_prob")
    )
    customer_max_quantiles = _quantile_summary(per_customer_max["max_calibrated_prob"])

    total_customers = eval_df["customer_id"].n_unique()
    share_by_c = {}
    for c in C_VALUES:
        threshold = c * median_price
        flagged = eval_df.with_columns(
            (MARGIN_RATE * pl.col("trailing_4w_price") * pl.col("calibrated_prob") > threshold).alias("flag")
        )
        customers_with_hit = (
            flagged.group_by("customer_id")
            .agg(pl.col("flag").fill_null(False).any().alias("any_flag"))
            .filter(pl.col("any_flag"))
            .height
        )
        share_by_c[str(c)] = customers_with_hit / total_customers

    result = {
        "eval_rows": eval_df.height,
        "eval_customers": total_customers,
        "median_trailing_4w_price": median_price,
        "articles_with_trailing_4w_price": trailing_price.height,
        "candidate_calibrated_prob_quantiles": candidate_prob_quantiles,
        "customer_max_calibrated_prob_quantiles": customer_max_quantiles,
        "share_of_customers_with_positive_expected_value_candidate": share_by_c,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
