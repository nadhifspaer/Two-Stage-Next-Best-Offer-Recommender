# customer, article, pair, and provenance feature blocks
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from src.candidates import _age_bucket_expr
from src.text_embed import REDUCED_DIM

WINDOW_1W = timedelta(days=7)
WINDOW_4W = timedelta(weeks=4)
WINDOW_12W = timedelta(weeks=12)

RECENT_TEXT_N = 10
RECENT_TEXT_VECTOR_COLUMNS = [f"recent_text_vec_{i}" for i in range(REDUCED_DIM)]

# rows per join batch, so no join touches a full candidate partition (two partition-sized frames measured a 2+ GB spike);
# 600,000, reduced from 1,000,000 with CUSTOMER_CHUNK_SIZE for margin under the 5 GB budget
PARTITION_BATCH_SIZE = 600_000

# customers per pass in customer_features and customer_recent_text_mean; each pass filters the scan to its own customers
# so the 31.8M-row aggregation never runs whole
CUSTOMER_BLOCK_CHUNK_SIZE = 100_000

PROVENANCE_STRATEGIES = [
    "repurchase",
    "colour_variant",
    "recent_popularity",
    "segment_popularity",
    "als",
]

_COUNT_FILL_ZERO = [
    "txn_count_total",
    "distinct_article_count_total",
    "txn_count_1w",
    "distinct_article_count_1w",
    "txn_count_4w",
    "distinct_article_count_4w",
    "txn_count_12w",
    "distinct_article_count_12w",
]


def customer_features(
    transactions: pl.DataFrame | pl.LazyFrame,
    customers: pl.DataFrame | pl.LazyFrame,
    window_start: date,
    chunk_size: int = CUSTOMER_BLOCK_CHUNK_SIZE,
) -> pl.DataFrame:
    # backward-only aggregates, one customer chunk per pass
    base = (
        customers.lazy()
        .select("customer_id", "age", "club_member_status", "fashion_news_frequency")
        .with_columns(_age_bucket_expr())
        .collect()
    )
    parts = [
        _customer_features_chunk(transactions, base.slice(offset, chunk_size), window_start)
        for offset in range(0, max(base.height, 1), chunk_size)
    ]
    return pl.concat(parts, how="vertical")


def _customer_features_chunk(
    transactions: pl.DataFrame | pl.LazyFrame,
    chunk: pl.DataFrame,
    window_start: date,
) -> pl.DataFrame:
    before = (
        transactions.lazy()
        .filter(pl.col("t_dat") < window_start)
        .filter(pl.col("customer_id").is_in(chunk["customer_id"]))
    )

    # default engine, not streaming: mean_purchase_price, std_purchase_price and share_sales_channel_* are float reductions
    # and the streaming engine combines partial sums in a run-dependent order
    totals = before.group_by("customer_id").agg(
        pl.len().alias("txn_count_total"),
        pl.col("article_id").n_unique().alias("distinct_article_count_total"),
        pl.col("price").mean().alias("mean_purchase_price"),
        pl.col("price").std().alias("std_purchase_price"),
        pl.col("t_dat").max().alias("last_t_dat"),
        (pl.col("sales_channel_id") == 1).mean().alias("share_sales_channel_1"),
        (pl.col("sales_channel_id") == 2).mean().alias("share_sales_channel_2"),
    ).collect()

    def windowed(horizon: timedelta, suffix: str) -> pl.DataFrame:
        return (
            before.filter(pl.col("t_dat") >= window_start - horizon)
            .group_by("customer_id")
            .agg(
                pl.len().alias(f"txn_count_{suffix}"),
                pl.col("article_id").n_unique().alias(f"distinct_article_count_{suffix}"),
            )
            .collect(engine="streaming")
        )

    w1 = windowed(WINDOW_1W, "1w")
    w4 = windowed(WINDOW_4W, "4w")
    w12 = windowed(WINDOW_12W, "12w")

    return (
        chunk.lazy()
        .join(totals.lazy(), on="customer_id", how="left")
        .join(w1.lazy(), on="customer_id", how="left")
        .join(w4.lazy(), on="customer_id", how="left")
        .join(w12.lazy(), on="customer_id", how="left")
        .with_columns(
            (pl.lit(window_start) - pl.col("last_t_dat"))
            .dt.total_days()
            .alias("days_since_last_purchase")
        )
        .with_columns([pl.col(c).fill_null(0) for c in _COUNT_FILL_ZERO])
        .drop("last_t_dat", "age")
        .collect()
    )


def article_features(
    transactions: pl.DataFrame | pl.LazyFrame,
    articles: pl.DataFrame | pl.LazyFrame,
    window_start: date,
) -> pl.DataFrame:
    before = transactions.lazy().filter(pl.col("t_dat") < window_start)

    # each aggregation is collected on its own so Polars releases that scan's buffers before the next starts;
    # default engine, not streaming: article_mean_price is a float mean (see customer_features)
    totals = before.group_by("article_id").agg(
        pl.col("t_dat").min().alias("first_t_dat"),
        pl.col("t_dat").max().alias("last_t_dat"),
        pl.col("price").mean().alias("article_mean_price"),
        pl.col("customer_id").n_unique().alias("distinct_buyers_total"),
    ).collect()

    def range_count(lo: date, hi: date, suffix: str) -> pl.DataFrame:
        return (
            before.filter((pl.col("t_dat") >= lo) & (pl.col("t_dat") < hi))
            .group_by("article_id")
            .agg(pl.len().alias(f"purchase_count_{suffix}"))
            .collect(engine="streaming")
        )

    c1 = range_count(window_start - WINDOW_1W, window_start, "1w")
    c4 = range_count(window_start - WINDOW_4W, window_start, "4w")
    c12 = range_count(window_start - WINDOW_12W, window_start, "12w")
    # trailing week immediately before the last 7-day window, for week-over-week trend only
    c_prior1 = range_count(window_start - 2 * WINDOW_1W, window_start - WINDOW_1W, "prior_1w")

    categoricals = articles.lazy().select(
        "article_id",
        "product_type_name",
        "department_name",
        "index_group_name",
        "garment_group_name",
    )

    result = (
        categoricals.join(totals.lazy(), on="article_id", how="left")
        .join(c1.lazy(), on="article_id", how="left")
        .join(c4.lazy(), on="article_id", how="left")
        .join(c12.lazy(), on="article_id", how="left")
        .join(c_prior1.lazy(), on="article_id", how="left")
        .with_columns(
            [
                pl.col(c).fill_null(0)
                for c in [
                    "distinct_buyers_total",
                    "purchase_count_1w",
                    "purchase_count_4w",
                    "purchase_count_12w",
                    "purchase_count_prior_1w",
                ]
            ]
        )
        .with_columns(
            # purchase_count_* are unsigned; cast to signed before subtracting or a
            # declining article's negative trend underflows to a huge positive value
            (
                pl.col("purchase_count_1w").cast(pl.Int64)
                - pl.col("purchase_count_prior_1w").cast(pl.Int64)
            ).alias("purchase_count_wow_trend"),
            (pl.lit(window_start) - pl.col("first_t_dat")).dt.total_days().alias("days_since_first_sale"),
            (pl.lit(window_start) - pl.col("last_t_dat")).dt.total_days().alias("days_since_last_sale"),
        )
        .drop("first_t_dat", "last_t_dat", "purchase_count_prior_1w")
        .collect(engine="streaming")
    )
    return result


def customer_recent_text_mean(
    transactions: pl.DataFrame | pl.LazyFrame,
    text_vectors: pl.DataFrame | pl.LazyFrame,
    window_start: date,
    n_recent: int = RECENT_TEXT_N,
    customer_ids: pl.Series | None = None,
    chunk_size: int = CUSTOMER_BLOCK_CHUNK_SIZE,
) -> pl.DataFrame:
    # mean of the last n_recent article text vectors per customer, one customer chunk per pass;
    # rank-and-cap site with article_id tiebreak after t_dat, same rule as src/candidates.py
    before = transactions.lazy().filter(pl.col("t_dat") < window_start)
    if customer_ids is None:
        customer_ids = before.select("customer_id").unique().collect()["customer_id"]
    customer_ids = customer_ids.sort()
    vec_cols = [f"text_vec_{i}" for i in range(REDUCED_DIM)]
    vectors = text_vectors.lazy().select("article_id", *vec_cols).collect()

    parts = []
    for offset in range(0, max(len(customer_ids), 1), chunk_size):
        chunk_ids = customer_ids[offset : offset + chunk_size]
        ranked = (
            before.filter(pl.col("customer_id").is_in(chunk_ids))
            .sort(["customer_id", "t_dat", "article_id"], descending=[False, True, False])
            .with_columns((pl.int_range(pl.len()).over("customer_id") + 1).alias("recency_rank"))
            .filter(pl.col("recency_rank") <= n_recent)
        )
        # not engine="streaming": all 32 columns are float mean reductions
        # feeding text_cosine_similarity; see customer_features.
        parts.append(
            ranked.join(vectors.lazy(), on="article_id", how="inner")
            .group_by("customer_id")
            .agg([pl.col(c).mean().alias(RECENT_TEXT_VECTOR_COLUMNS[i]) for i, c in enumerate(vec_cols)])
            .collect()
        )
    return pl.concat(parts, how="vertical").sort("customer_id")


def customer_article_pair_features(
    history: pl.DataFrame,
    articles: pl.DataFrame | pl.LazyFrame,
    candidates: pl.DataFrame,
    customer_block: pl.DataFrame,
    article_block: pl.DataFrame,
    text_vectors: pl.DataFrame,
    customer_recent_text_mean_block: pl.DataFrame,
    window_start: date,
    batch_size: int = PARTITION_BATCH_SIZE,
) -> pl.DataFrame:
    """history: this partition's customers only, t_dat < window_start, columns customer_id/article_id/t_dat, pre-filtered by the caller
    so the full transactions table is never re-scanned per partition (6+ GB peak RSS across 14 partitions)."""
    attrs = articles.lazy().select("article_id", "product_code", "colour_group_name", "department_name")
    history = history.lazy().join(attrs, on="article_id", how="left").collect(engine="streaming")

    exact = history.lazy().group_by(["customer_id", "article_id"]).agg(
        pl.len().alias("prior_purchase_count_this_article"),
        pl.col("t_dat").max().alias("last_t_dat_this_article"),
    )
    by_product_code = (
        history.lazy()
        .group_by(["customer_id", "product_code"])
        .agg(pl.len().alias("prior_purchase_count_same_product_code"))
    )
    by_colour = (
        history.lazy()
        .group_by(["customer_id", "colour_group_name"])
        .agg(pl.len().alias("prior_purchase_count_same_colour"))
    )
    by_department = (
        history.lazy()
        .group_by(["customer_id", "department_name"])
        .agg(pl.len().alias("prior_purchase_count_same_department"))
    )

    # exact/by_product_code/by_colour/by_department are small (grouped down from history, a few million rows at most);
    # collecting each once is cheap, and history is no longer needed past this point
    exact = exact.collect(engine="streaming")
    by_product_code = by_product_code.collect(engine="streaming")
    by_colour = by_colour.collect(engine="streaming")
    by_department = by_department.collect(engine="streaming")
    del history

    return _pair_features_batched(
        candidates,
        attrs,
        exact,
        by_product_code,
        by_colour,
        by_department,
        customer_block,
        article_block,
        text_vectors,
        customer_recent_text_mean_block,
        window_start,
        batch_size=batch_size,
    )


def _pair_features_batched(
    candidates: pl.DataFrame,
    attrs: pl.LazyFrame,
    exact: pl.DataFrame,
    by_product_code: pl.DataFrame,
    by_colour: pl.DataFrame,
    by_department: pl.DataFrame,
    customer_block: pl.DataFrame,
    article_block: pl.DataFrame,
    text_vectors: pl.DataFrame,
    customer_recent_text_mean_block: pl.DataFrame,
    window_start: date,
    batch_size: int = PARTITION_BATCH_SIZE,
) -> pl.DataFrame:
    # every join below runs inside one pass per batch, so none materializes a full-partition (7M+ row) result;
    # joining two full-partition frames at the end measured a 2+ GB spike
    art_vec_cols = [f"text_vec_{i}" for i in range(REDUCED_DIM)]
    dot = sum(
        pl.col(art_vec_cols[i]).fill_null(0.0) * pl.col(RECENT_TEXT_VECTOR_COLUMNS[i]).fill_null(0.0)
        for i in range(REDUCED_DIM)
    )
    art_norm = sum(pl.col(c).fill_null(0.0) ** 2 for c in art_vec_cols).sqrt()
    cust_norm = sum(pl.col(c).fill_null(0.0) ** 2 for c in RECENT_TEXT_VECTOR_COLUMNS).sqrt()

    text_vectors_lazy = text_vectors.lazy().select("article_id", *art_vec_cols)
    recent_text_mean_lazy = customer_recent_text_mean_block.lazy()
    customer_price_lazy = customer_block.lazy().select("customer_id", "mean_purchase_price")
    article_price_lazy = article_block.lazy().select("article_id", "article_mean_price")

    batches = []
    for offset in range(0, candidates.height, batch_size):
        batch = (
            candidates.slice(offset, batch_size)
            .lazy()
            .select("customer_id", "article_id")
            .join(attrs, on="article_id", how="left")
            .join(exact.lazy(), on=["customer_id", "article_id"], how="left")
            .join(by_product_code.lazy(), on=["customer_id", "product_code"], how="left")
            .join(by_colour.lazy(), on=["customer_id", "colour_group_name"], how="left")
            .join(by_department.lazy(), on=["customer_id", "department_name"], how="left")
            .join(customer_price_lazy, on="customer_id", how="left")
            .join(article_price_lazy, on="article_id", how="left")
            .join(text_vectors_lazy, on="article_id", how="left")
            .join(recent_text_mean_lazy, on="customer_id", how="left")
            .with_columns(
                pl.col("prior_purchase_count_this_article").fill_null(0),
                pl.col("prior_purchase_count_same_product_code").fill_null(0),
                pl.col("prior_purchase_count_same_colour").fill_null(0),
                pl.col("prior_purchase_count_same_department").fill_null(0),
                (pl.lit(window_start) - pl.col("last_t_dat_this_article"))
                .dt.total_days()
                .alias("days_since_last_purchase_this_article"),
                dot.alias("_dot"),
                art_norm.alias("_art_norm"),
                cust_norm.alias("_cust_norm"),
            )
            .with_columns(
                pl.when(
                    pl.col("mean_purchase_price").is_null() | (pl.col("mean_purchase_price") == 0)
                )
                .then(None)
                .otherwise(pl.col("article_mean_price") / pl.col("mean_purchase_price"))
                .alias("price_ratio_to_customer_mean"),
                pl.when((pl.col("_art_norm") == 0) | (pl.col("_cust_norm") == 0))
                .then(0.0)
                .otherwise(pl.col("_dot") / (pl.col("_art_norm") * pl.col("_cust_norm")))
                .alias("text_cosine_similarity"),
            )
            .select(
                "customer_id",
                "article_id",
                "prior_purchase_count_this_article",
                "days_since_last_purchase_this_article",
                "prior_purchase_count_same_product_code",
                "prior_purchase_count_same_colour",
                "prior_purchase_count_same_department",
                "price_ratio_to_customer_mean",
                "text_cosine_similarity",
            )
            .collect(engine="streaming")
        )
        batches.append(batch)

    return pl.concat(batches, how="vertical")


def candidate_provenance_features(candidates: pl.DataFrame) -> pl.DataFrame:
    # rank_<strategy> columns come from pool_candidates: one rank per strategy, null where that strategy did not produce the candidate;
    # best_rank (the minimum across producing strategies) stays alongside since pool_candidates caps on it
    rank_cols = [f"rank_{strategy}" for strategy in PROVENANCE_STRATEGIES]
    flags = [
        pl.col("strategies").list.contains(strategy).alias(f"from_{strategy}")
        for strategy in PROVENANCE_STRATEGIES
    ]
    return candidates.select(
        "customer_id", "article_id", "n_strategies", "best_rank", *rank_cols, *flags
    )


def assemble_partition_features(
    pair: pl.DataFrame,
    provenance: pl.DataFrame,
    customer_block: pl.DataFrame,
    article_block: pl.DataFrame,
    batch_size: int = PARTITION_BATCH_SIZE,
    write_dir=None,
    file_stem: str | None = None,
) -> pl.DataFrame | int:
    """Joins the pair block to provenance, customer and article blocks in row batches; with write_dir each batch is written to its own
    Parquet file `{file_stem}-batch-NNNN.parquet` and the row count is returned, without it a DataFrame is returned."""
    provenance_lazy = provenance.lazy()
    customer_lazy = customer_block.lazy()
    article_lazy = article_block.lazy()

    def _batch(offset: int) -> pl.DataFrame:
        return (
            pair.slice(offset, batch_size)
            .lazy()
            .join(provenance_lazy, on=["customer_id", "article_id"], how="inner")
            .join(customer_lazy, on="customer_id", how="left")
            .join(article_lazy, on="article_id", how="left")
            .collect(engine="streaming")
        )

    if write_dir is None:
        return pl.concat([_batch(o) for o in range(0, pair.height, batch_size)], how="vertical")

    write_dir = Path(write_dir)
    write_dir.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    for batch_idx, offset in enumerate(range(0, pair.height, batch_size)):
        batch = _batch(offset)
        batch.write_parquet(write_dir / f"{file_stem}-batch-{batch_idx:04d}.parquet")
        rows_written += batch.height
    return rows_written
