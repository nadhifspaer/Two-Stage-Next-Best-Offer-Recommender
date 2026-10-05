# Recall@100, MAP@12, Brier, baseline comparison
from datetime import date

import polars as pl

# bucket edges match the EDA (notebooks/01_eda.ipynb); "never" is a
# separate label for customers with no purchase before the anchor date
RECENCY_BUCKET_EDGES_DAYS = [7, 28, 84, 364]
RECENCY_BUCKET_LABELS = [
    "last 1 week",
    "last 4 weeks",
    "last 12 weeks",
    "last 52 weeks",
    "more than 52 weeks ago",
]


def recall_hits(
    candidates: pl.DataFrame | pl.LazyFrame,
    actual_purchases: pl.DataFrame | pl.LazyFrame,
    k: int,
) -> tuple[int, int]:
    candidates = candidates.lazy().select("customer_id", "article_id").unique()
    actual = (
        actual_purchases.lazy()
        .select("customer_id", "article_id")
        .unique()
        .collect(engine="streaming")
    )

    # candidates must already be capped at k per customer by the caller; this
    # function measures overlap, it does not itself truncate to top-k
    over_cap = (
        candidates.group_by("customer_id")
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") > k)
        .collect(engine="streaming")
    )
    if over_cap.height > 0:
        raise ValueError(f"{over_cap.height} customers have more than {k} candidates")

    hits = (
        actual.lazy()
        .join(candidates, on=["customer_id", "article_id"], how="semi")
        .collect(engine="streaming")
    )
    return hits.height, actual.height


def recall_at_k(
    candidates: pl.DataFrame | pl.LazyFrame,
    actual_purchases: pl.DataFrame | pl.LazyFrame,
    k: int,
) -> float:
    hits, total = recall_hits(candidates, actual_purchases, k)
    return hits / total


MAP_K = 12


def ap_at_12_per_customer(
    predictions: pl.DataFrame | pl.LazyFrame,
    actual_purchases: pl.DataFrame | pl.LazyFrame,
) -> pl.DataFrame:
    """predictions: customer_id, article_id, rank_position (1..12, 1 is top); actual_purchases: customer_id, article_id.
    Returns customer_id, m (true positive count), ap for each customer with a purchase in the window (the scoring population)."""
    predictions = predictions.lazy().select("customer_id", "article_id", "rank_position")
    actual = actual_purchases.lazy().select("customer_id", "article_id").unique()

    over_cap = (
        predictions.group_by("customer_id")
        .agg(pl.col("rank_position").max().alias("max_rank"))
        .filter(pl.col("max_rank") > MAP_K)
        .collect(engine="streaming")
    )
    if over_cap.height > 0:
        raise ValueError(f"{over_cap.height} customers have a rank_position beyond {MAP_K}")

    purchase_counts = actual.group_by("customer_id").agg(pl.len().alias("m"))

    ap_numerator = (
        predictions.join(
            actual.with_columns(pl.lit(True).alias("hit")), on=["customer_id", "article_id"], how="left"
        )
        .with_columns(pl.col("hit").fill_null(False))
        .sort(["customer_id", "rank_position"])
        .with_columns(pl.col("hit").cast(pl.Int32).cum_sum().over("customer_id").alias("cum_hits"))
        .with_columns((pl.col("cum_hits") / pl.col("rank_position")).alias("precision_at_k"))
        .with_columns((pl.col("precision_at_k") * pl.col("hit").cast(pl.Int32)).alias("contribution"))
        .group_by("customer_id")
        .agg(pl.col("contribution").sum().alias("ap_numerator"))
    )

    # default engine, not streaming: ap_numerator is a float sum and streaming combines partial sums in a run-dependent order;
    # sorted by customer_id on return because the downstream .mean() in map_at_12 is order-sensitive too
    return (
        purchase_counts.join(ap_numerator, on="customer_id", how="left")
        .with_columns(pl.col("ap_numerator").fill_null(0.0))
        .with_columns((pl.col("ap_numerator") / pl.min_horizontal(pl.col("m"), pl.lit(MAP_K))).alias("ap"))
        .select("customer_id", "m", "ap")
        .sort("customer_id")
        .collect()
    )


def map_at_12(
    predictions: pl.DataFrame | pl.LazyFrame,
    actual_purchases: pl.DataFrame | pl.LazyFrame,
) -> tuple[float, int]:
    """Competition rule: a customer with no purchase in the window is excluded, so the average is over customers with a purchase.
    Returns (MAP@12, number of customers scored)."""
    scored = ap_at_12_per_customer(predictions, actual_purchases)
    return float(scored["ap"].mean()), scored.height


def oracle_ap_at_12_per_customer(
    m_found: pl.DataFrame | pl.LazyFrame,
    actual_purchases: pl.DataFrame | pl.LazyFrame,
) -> pl.DataFrame:
    """m_found: customer_id, m_found, the count of a customer's true purchases anywhere in their candidate pool. Best any ordering can do:
    AP@12 = min(m_found, 12) / min(m, 12), the MAP@12 ceiling Recall@100 allows. Returns customer_id, m, m_found, ap."""
    purchase_counts = (
        actual_purchases.lazy()
        .select("customer_id", "article_id")
        .unique()
        .group_by("customer_id")
        .agg(pl.len().alias("m"))
    )
    # engine="streaming" is fine here: ap is min_horizontal(int, int) / min_horizontal(int, int), no float reduction;
    # still sorted by customer_id so oracle_map_at_12's downstream .mean() sums in the same order every run
    return (
        purchase_counts.join(m_found.lazy(), on="customer_id", how="left")
        .with_columns(pl.col("m_found").fill_null(0))
        .with_columns(
            (
                pl.min_horizontal(pl.col("m_found"), pl.lit(MAP_K))
                / pl.min_horizontal(pl.col("m"), pl.lit(MAP_K))
            ).alias("ap")
        )
        .select("customer_id", "m", "m_found", "ap")
        .sort("customer_id")
        .collect(engine="streaming")
    )


def oracle_map_at_12(
    m_found: pl.DataFrame | pl.LazyFrame,
    actual_purchases: pl.DataFrame | pl.LazyFrame,
) -> tuple[float, int]:
    """Same exclusion rule as map_at_12. Returns (MAP@12, number of
    customers scored)."""
    scored = oracle_ap_at_12_per_customer(m_found, actual_purchases)
    return float(scored["ap"].mean()), scored.height


def customer_recency_buckets(
    transactions: pl.DataFrame | pl.LazyFrame,
    customers: pl.DataFrame | pl.LazyFrame,
    anchor: date,
) -> pl.DataFrame:
    # days since last purchase strictly before anchor; anchored at the label
    # window start (not the corpus end date) so this stays leakage-safe
    last_purchase = (
        transactions.lazy()
        .filter(pl.col("t_dat") < anchor)
        .group_by("customer_id")
        .agg(pl.col("t_dat").max().alias("last_t_dat"))
    )
    recency = (
        customers.lazy()
        .select("customer_id")
        .join(last_purchase, on="customer_id", how="left")
        .with_columns(
            (pl.lit(anchor) - pl.col("last_t_dat")).dt.total_days().alias("days_since_last")
        )
    )
    e = RECENCY_BUCKET_EDGES_DAYS
    labels = RECENCY_BUCKET_LABELS
    bucket_expr = (
        pl.when(pl.col("days_since_last").is_null())
        .then(pl.lit("never"))
        .when(pl.col("days_since_last") <= e[0])
        .then(pl.lit(labels[0]))
        .when(pl.col("days_since_last") <= e[1])
        .then(pl.lit(labels[1]))
        .when(pl.col("days_since_last") <= e[2])
        .then(pl.lit(labels[2]))
        .when(pl.col("days_since_last") <= e[3])
        .then(pl.lit(labels[3]))
        .otherwise(pl.lit(labels[4]))
    )
    return (
        recency.with_columns(bucket_expr.alias("recency_bucket"))
        .select("customer_id", "recency_bucket")
        .collect(engine="streaming")
    )
