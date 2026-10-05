# LGBMRanker and LGBMClassifier training
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

EXCLUDED_COLUMNS = ["customer_id", "article_id", "label", "window_start"]

# every column in the assembled matrix except the join/group keys, the label and window_start; the only feature source
# either model reads (exclusion check in the rank tests)
FEATURE_COLUMNS = [
    "prior_purchase_count_this_article",
    "days_since_last_purchase_this_article",
    "prior_purchase_count_same_product_code",
    "prior_purchase_count_same_colour",
    "prior_purchase_count_same_department",
    "price_ratio_to_customer_mean",
    "text_cosine_similarity",
    "n_strategies",
    "best_rank",
    "rank_repurchase",
    "rank_colour_variant",
    "rank_recent_popularity",
    "rank_segment_popularity",
    "rank_als",
    "from_repurchase",
    "from_colour_variant",
    "from_recent_popularity",
    "from_segment_popularity",
    "from_als",
    "club_member_status",
    "fashion_news_frequency",
    "age_bucket",
    "txn_count_total",
    "distinct_article_count_total",
    "mean_purchase_price",
    "std_purchase_price",
    "share_sales_channel_1",
    "share_sales_channel_2",
    "txn_count_1w",
    "distinct_article_count_1w",
    "txn_count_4w",
    "distinct_article_count_4w",
    "txn_count_12w",
    "distinct_article_count_12w",
    "days_since_last_purchase",
    "product_type_name",
    "department_name",
    "index_group_name",
    "garment_group_name",
    "article_mean_price",
    "distinct_buyers_total",
    "purchase_count_1w",
    "purchase_count_4w",
    "purchase_count_12w",
    "purchase_count_wow_trend",
    "days_since_first_sale",
    "days_since_last_sale",
]

# the seven string-typed columns in FEATURE_COLUMNS; every other column is
# already numeric or boolean
CATEGORICAL_COLUMNS = [
    "club_member_status",
    "fashion_news_frequency",
    "age_bucket",
    "product_type_name",
    "department_name",
    "index_group_name",
    "garment_group_name",
]

# LightGBM treats a negative categorical code as missing, so reusing that convention for an unseen category
# avoids inventing a code that could collide with a real one
UNKNOWN_CATEGORY_CODE = -1
_NULL_SENTINEL = "__NULL__"

EARLY_STOPPING_TRAIN_WINDOWS = ["2020-08-19", "2020-08-26", "2020-09-02"]
EARLY_STOPPING_VALIDATION_WINDOW = "2020-09-09"
EARLY_STOPPING_ROUNDS = 50

# fixed seed, matching ALS_RANDOM_STATE and NEGATIVE_SAMPLING_SEED; deterministic=True with force_row_wise=True is required
# for bit-exact reproducibility, deterministic=True alone does not hold under multi-threaded histogram building
LGBM_SEED = 42
LGBM_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "random_state": LGBM_SEED,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}


def fit_categorical_mapping(
    df: pl.DataFrame, columns: list[str] = CATEGORICAL_COLUMNS
) -> dict[str, dict[str, int]]:
    """Fits one category -> integer code mapping per column; nulls are their own category.
    A value absent from this fit has no entry and must map to UNKNOWN_CATEGORY_CODE at apply time."""
    mapping = {}
    for col in columns:
        categories = (
            df.select(pl.col(col).fill_null(_NULL_SENTINEL))
            .to_series()
            .unique()
            .sort()
            .to_list()
        )
        mapping[col] = {category: code for code, category in enumerate(categories)}
    return mapping


def apply_categorical_mapping(
    df: pl.DataFrame,
    mapping: dict[str, dict[str, int]],
    columns: list[str] = CATEGORICAL_COLUMNS,
) -> pl.DataFrame:
    """Applies a mapping fit elsewhere, the same persisted mapping at training and every later inference, never re-derived from this data.
    A value with no entry, including a category unseen at training, becomes UNKNOWN_CATEGORY_CODE."""
    exprs = [
        pl.col(col)
        .fill_null(_NULL_SENTINEL)
        .replace_strict(mapping[col], default=UNKNOWN_CATEGORY_CODE, return_dtype=pl.Int32)
        .alias(col)
        for col in columns
    ]
    return df.with_columns(exprs)


def save_categorical_mapping(mapping: dict[str, dict[str, int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)


def load_categorical_mapping(path: Path) -> dict[str, dict[str, int]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_matrix(df: pl.DataFrame, mapping: dict[str, dict[str, int]]) -> pl.DataFrame:
    # public: encode categoricals with the persisted mapping and sort rows by (window_start, customer_id, article_id),
    # the contiguous per-group order the ranker requires
    return apply_categorical_mapping(df, mapping).sort(["window_start", "customer_id", "article_id"])


def _group_sizes(df: pl.DataFrame) -> list[int]:
    # df must already be sorted by (window_start, customer_id, ...); group
    # sizes are read off in that same row order via maintain_order
    return (
        df.group_by(["window_start", "customer_id"], maintain_order=True)
        .agg(pl.len().alias("n"))["n"]
        .to_list()
    )


def to_pandas_X(df: pl.DataFrame):
    # public: the FEATURE_COLUMNS allowlist as a pandas frame
    return df.select(FEATURE_COLUMNS).to_pandas()


def train_ranker(
    train_df: pl.DataFrame, categorical_mapping: dict[str, dict[str, int]]
) -> tuple[lgb.LGBMRanker, dict]:
    """Two-phase training: early stopping on the fourth window picks the iteration count, then a refit on all four windows with it fixed; the validation week is never in the data.
    Output is a relative ordering score with no probabilistic meaning: never an input to expected value, a probability threshold, or a purchase likelihood."""
    fit_df = prepare_matrix(train_df, categorical_mapping)

    early_train = fit_df.filter(pl.col("window_start").is_in(EARLY_STOPPING_TRAIN_WINDOWS))
    early_val = fit_df.filter(pl.col("window_start") == EARLY_STOPPING_VALIDATION_WINDOW)

    probe = lgb.LGBMRanker(objective="lambdarank", **LGBM_PARAMS)
    probe.fit(
        to_pandas_X(early_train),
        early_train["label"].to_numpy(),
        group=_group_sizes(early_train),
        eval_set=[(to_pandas_X(early_val), early_val["label"].to_numpy())],
        eval_group=[_group_sizes(early_val)],
        eval_at=[12],
        categorical_feature=CATEGORICAL_COLUMNS,
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    best_iteration = probe.best_iteration_

    final_params = {**LGBM_PARAMS, "n_estimators": best_iteration}
    final_model = lgb.LGBMRanker(objective="lambdarank", **final_params)
    final_model.fit(
        to_pandas_X(fit_df),
        fit_df["label"].to_numpy(),
        group=_group_sizes(fit_df),
        categorical_feature=CATEGORICAL_COLUMNS,
    )

    return final_model, {"best_iteration": best_iteration}


def train_classifier(
    train_df: pl.DataFrame, categorical_mapping: dict[str, dict[str, int]]
) -> tuple[lgb.LGBMClassifier, dict]:
    """Same two-phase split as train_ranker. Output is calibrated only to the sampled 9.10% positive rate (10 negatives per positive), not the real rate,
    and is unusable by the decision layer until isotonic calibration is applied on the validation week."""
    fit_df = prepare_matrix(train_df, categorical_mapping)

    early_train = fit_df.filter(pl.col("window_start").is_in(EARLY_STOPPING_TRAIN_WINDOWS))
    early_val = fit_df.filter(pl.col("window_start") == EARLY_STOPPING_VALIDATION_WINDOW)

    probe = lgb.LGBMClassifier(**LGBM_PARAMS)
    probe.fit(
        to_pandas_X(early_train),
        early_train["label"].to_numpy(),
        eval_set=[(to_pandas_X(early_val), early_val["label"].to_numpy())],
        eval_metric="binary_logloss",
        categorical_feature=CATEGORICAL_COLUMNS,
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    best_iteration = probe.best_iteration_

    final_params = {**LGBM_PARAMS, "n_estimators": best_iteration}
    final_model = lgb.LGBMClassifier(**final_params)
    final_model.fit(
        to_pandas_X(fit_df),
        fit_df["label"].to_numpy(),
        categorical_feature=CATEGORICAL_COLUMNS,
    )

    return final_model, {"best_iteration": best_iteration}


def score_distributions(
    ranker, classifier, df: pl.DataFrame, categorical_mapping: dict[str, dict[str, int]],
    n_bins: int = 80, batch_rows: int = 250_000,
) -> dict:
    """Histogram of ranker scores and uncalibrated classifier probabilities by label, shared bin edges per model.
    Rows are scored in batches; only the score vectors are held in full."""
    scored = prepare_matrix(df, categorical_mapping)
    labels = scored["label"].to_numpy()
    ranker_scores = np.empty(scored.height, dtype=np.float64)
    classifier_probs = np.empty(scored.height, dtype=np.float64)
    for start in range(0, scored.height, batch_rows):
        X = to_pandas_X(scored.slice(start, batch_rows))
        stop = start + len(X)
        ranker_scores[start:stop] = ranker.predict(X)
        classifier_probs[start:stop] = classifier.predict_proba(X)[:, 1]
        del X

    def summarize(values: np.ndarray) -> dict:
        edges = np.histogram_bin_edges(values, bins=n_bins)
        return {
            "bin_edges": edges.tolist(),
            "negative_counts": np.histogram(values[labels == 0], bins=edges)[0].tolist(),
            "positive_counts": np.histogram(values[labels == 1], bins=edges)[0].tolist(),
            "mean": float(values.mean()),
            "p01": float(np.quantile(values, 0.01)),
            "p50": float(np.quantile(values, 0.50)),
            "p99": float(np.quantile(values, 0.99)),
        }

    return {
        "rows": int(scored.height),
        "positives": int(labels.sum()),
        "positive_rate": float(labels.mean()),
        "ranker": summarize(ranker_scores),
        "classifier": summarize(classifier_probs),
    }
