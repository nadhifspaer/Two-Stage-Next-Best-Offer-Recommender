# Parquet conversion, weekly label windows, date split
import os
import shutil
import tempfile
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from src.schema import (
    ArticlesSchema,
    CustomersSchema,
    SampleSubmissionSchema,
    TransactionsSchema,
)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# leading-zero identifier and code columns; must be read as strings, never inferred as int
TRANSACTIONS_DTYPES = {
    "t_dat": pl.Date,
    "customer_id": pl.Utf8,
    "article_id": pl.Utf8,
    "price": pl.Float64,
    "sales_channel_id": pl.Int64,
}
ARTICLES_DTYPES = {
    "article_id": pl.Utf8,
    "product_code": pl.Utf8,
    "colour_group_code": pl.Utf8,
}
CUSTOMERS_DTYPES = {
    "customer_id": pl.Utf8,
    "postal_code": pl.Utf8,
}
SAMPLE_SUBMISSION_DTYPES = {
    "customer_id": pl.Utf8,
}

CONVERSIONS = [
    ("transactions_train", TransactionsSchema, TRANSACTIONS_DTYPES),
    ("articles", ArticlesSchema, ARTICLES_DTYPES),
    ("customers", CustomersSchema, CUSTOMERS_DTYPES),
    ("sample_submission", SampleSubmissionSchema, SAMPLE_SUBMISSION_DTYPES),
]


def convert_raw_to_parquet() -> dict[str, dict[str, int]]:
    """Writes to a temporary directory, then renames atomically into PROCESSED_DIR only after all four tables pass their Pandera schema,
    so a failed run never leaves PROCESSED_DIR half converted."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(dir=PROCESSED_DIR.parent, prefix=".raw_to_parquet_tmp_"))
    try:
        report = {}
        tmp_paths = {}
        for name, schema, dtypes in CONVERSIONS:
            raw_path = RAW_DIR / f"{name}.csv"
            tmp_path = tmp_dir / f"{name}.parquet"
            df = pl.read_csv(raw_path, schema_overrides=dtypes)
            df = schema.validate(df)
            df.write_parquet(tmp_path)
            tmp_paths[name] = tmp_path
            report[name] = {
                "raw_bytes": raw_path.stat().st_size,
                "parquet_bytes": tmp_path.stat().st_size,
            }

        # all four validated: publish each with an atomic rename
        for name, tmp_path in tmp_paths.items():
            os.replace(tmp_path, PROCESSED_DIR / f"{name}.parquet")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return report


def validate_processed_tables() -> dict:
    """Validates the four already-converted tables in PROCESSED_DIR against the Pandera schemas, before any transform.
    Raises pandera.errors.SchemaError on the first table that fails."""
    validated = []
    for name, schema, _dtypes in CONVERSIONS:
        df = pl.read_parquet(PROCESSED_DIR / f"{name}.parquet")
        schema.validate(df)
        validated.append(name)
    return {"tables_validated": validated}


LABEL_WINDOW_DAYS = 7
TRAIN_WINDOW_START = date(2020, 8, 19)
VALIDATION_WINDOW_START = date(2020, 9, 16)


def label_windows() -> list[dict]:
    # four training label weeks plus the validation label week
    windows = [
        {
            "split": "train",
            "window_start": TRAIN_WINDOW_START + timedelta(days=7 * i),
            "window_end": TRAIN_WINDOW_START + timedelta(days=7 * i + LABEL_WINDOW_DAYS - 1),
        }
        for i in range(4)
    ]
    windows.append(
        {
            "split": "validation",
            "window_start": VALIDATION_WINDOW_START,
            "window_end": VALIDATION_WINDOW_START + timedelta(days=LABEL_WINDOW_DAYS - 1),
        }
    )
    return windows


def transactions_before(transactions: pl.DataFrame, window_start: date) -> pl.DataFrame:
    # strictly before window_start, date cutoff only, no shuffled splitting
    return transactions.filter(pl.col("t_dat") < window_start)


NEGATIVES_PER_POSITIVE = 10
# fixed so negative sampling is reproducible run to run, matching
# ALS_RANDOM_STATE in src/candidates.py
NEGATIVE_SAMPLING_SEED = 42


def assemble_training_labels(
    candidates: pl.DataFrame | pl.LazyFrame,
    transactions: pl.DataFrame | pl.LazyFrame,
    window_start: date,
    window_end: date,
    negatives_per_positive: int = NEGATIVES_PER_POSITIVE,
    seed: int = NEGATIVE_SAMPLING_SEED,
) -> pl.DataFrame:
    """Labels one window's pooled candidates (positive if the pair is in the window's transactions) and samples negatives per positive per customer
    by a seeded hash with an article_id tiebreak, reproducible regardless of thread order; a customer with no positive adds no rows (no lambdarank gradient)."""
    actual = (
        transactions.lazy()
        .filter((pl.col("t_dat") >= window_start) & (pl.col("t_dat") <= window_end))
        .select("customer_id", "article_id")
        .unique()
        .with_columns(pl.lit(1).cast(pl.Int8).alias("label"))
    )

    labeled = (
        candidates.lazy()
        .join(actual, on=["customer_id", "article_id"], how="left")
        .with_columns(pl.col("label").fill_null(0).cast(pl.Int8))
    )

    positives = labeled.filter(pl.col("label") == 1)
    n_positives = positives.group_by("customer_id").agg(pl.len().alias("n_positives"))

    negatives = (
        labeled.filter(pl.col("label") == 0)
        .join(n_positives, on="customer_id", how="inner")  # drops customers with no positive
        .with_columns(pl.struct(["customer_id", "article_id"]).hash(seed=seed).alias("_sample_key"))
        .sort(["customer_id", "_sample_key", "article_id"])
        .with_columns((pl.int_range(pl.len()).over("customer_id") + 1).alias("_neg_rank"))
        .filter(pl.col("_neg_rank") <= negatives_per_positive * pl.col("n_positives"))
        .drop("_sample_key", "_neg_rank", "n_positives")
    )

    return pl.concat([positives, negatives], how="vertical").collect(engine="streaming")
