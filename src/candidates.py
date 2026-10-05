# the five candidate strategies, ALS fit, pooling, chunked orchestration
import src.env_config  # noqa: F401 - must stay the first import: sets OPENBLAS_NUM_THREADS before any numeric library loads

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares

import src.publish as publish

REPURCHASE_HORIZON = timedelta(weeks=52)
POPULARITY_WINDOW = timedelta(days=7)
ALS_PADDING_SCORE = -1e30
ALS_RECOMMEND_BATCH_SIZE = 5000
# fixed so candidate generation is reproducible run to run; without it Recall@100,
# provenance features, and experiment comparisons all shift between runs
ALS_RANDOM_STATE = 42

# per-strategy caps applied before pooling, so the cap bounds memory rather than
# only bounding the final pool
REPURCHASE_CAP = 30
COLOUR_VARIANT_CAP = 30
RECENT_POPULARITY_CAP = 20
SEGMENT_POPULARITY_CAP = 20
ALS_CAP = 30
POOL_CAP = 100

# customers per chunk in generate_candidates_chunked; peak memory depends on this, not on the customer base
# 60,000/40,000/30,000 measured 5.34/4.30/4.11 GB peak RSS on window 2020-08-19; 40,000 chosen for 2.57 GB headroom, content unchanged
CUSTOMER_CHUNK_SIZE = 40_000

# age bucket edges in years; null age falls into "unknown", never dropped
AGE_BUCKET_EDGES = [25, 35, 45, 55, 65]
AGE_BUCKET_LABELS = ["<25", "25-34", "35-44", "45-54", "55-64", "65+"]


def _age_bucket_expr() -> pl.Expr:
    return (
        pl.col("age")
        .cut(AGE_BUCKET_EDGES, labels=AGE_BUCKET_LABELS)
        .cast(pl.Utf8)
        .fill_null("unknown")
        .alias("age_bucket")
    )


def repurchase_candidates(
    transactions: pl.DataFrame | pl.LazyFrame,
    window_start: date,
    customer_ids: pl.Series | None = None,
    cap: int = REPURCHASE_CAP,
) -> pl.DataFrame:
    horizon_start = window_start - REPURCHASE_HORIZON
    prior = transactions.lazy().filter(
        (pl.col("t_dat") >= horizon_start) & (pl.col("t_dat") < window_start)
    )
    if customer_ids is not None:
        prior = prior.filter(pl.col("customer_id").is_in(customer_ids))

    return (
        prior.sort(["t_dat", "article_id"], descending=[True, False])
        .unique(subset=["customer_id", "article_id"], keep="first")
        # article_id as a tiebreak: Polars sort is not stable across runs for
        # ties on t_dat alone, which made rank_in_strategy non-deterministic
        .sort(["customer_id", "t_dat", "article_id"], descending=[False, True, False])
        .with_columns(
            (pl.int_range(pl.len()).over("customer_id") + 1).alias("rank_in_strategy")
        )
        .filter(pl.col("rank_in_strategy") <= cap)
        .with_columns(pl.lit("repurchase").alias("strategy"))
        .select(["customer_id", "article_id", "strategy", "rank_in_strategy"])
        .collect(engine="streaming")
    )


def colour_variant_candidates(
    transactions: pl.DataFrame | pl.LazyFrame,
    articles: pl.DataFrame | pl.LazyFrame,
    window_start: date,
    customer_ids: pl.Series | None = None,
    cap: int = COLOUR_VARIANT_CAP,
) -> pl.DataFrame:
    horizon_start = window_start - REPURCHASE_HORIZON
    articles = articles.lazy()

    # restrict to the chunk's customers before the variant join, not just before
    # the final cap, so the join fans out over a bounded row count
    prior = transactions.lazy().filter(
        (pl.col("t_dat") >= horizon_start) & (pl.col("t_dat") < window_start)
    )
    if customer_ids is not None:
        prior = prior.filter(pl.col("customer_id").is_in(customer_ids))
    prior = prior.group_by(["customer_id", "article_id"]).agg(pl.col("t_dat").max())

    purchased = prior.join(
        articles.select(["article_id", "product_code"]), on="article_id", how="inner"
    )
    variants = purchased.join(
        articles.select(pl.col("article_id").alias("variant_article_id"), "product_code"),
        on="product_code",
        how="inner",
    ).filter(pl.col("variant_article_id") != pl.col("article_id"))

    return (
        variants.sort(["t_dat", "variant_article_id"], descending=[True, False])
        .unique(subset=["customer_id", "variant_article_id"], keep="first")
        # variant_article_id as a tiebreak: Polars sort is not stable across runs
        # for ties on t_dat alone, which made rank_in_strategy non-deterministic
        .sort(
            ["customer_id", "t_dat", "variant_article_id"], descending=[False, True, False]
        )
        .with_columns(
            (pl.int_range(pl.len()).over("customer_id") + 1).alias("rank_in_strategy")
        )
        .filter(pl.col("rank_in_strategy") <= cap)
        .with_columns(pl.lit("colour_variant").alias("strategy"))
        .select(
            "customer_id",
            pl.col("variant_article_id").alias("article_id"),
            "strategy",
            "rank_in_strategy",
        )
        .collect(engine="streaming")
    )


def recent_popularity_table(
    transactions: pl.DataFrame | pl.LazyFrame, window_start: date, cap: int = RECENT_POPULARITY_CAP
) -> pl.DataFrame:
    recent_start = window_start - POPULARITY_WINDOW
    return (
        transactions.lazy()
        .filter((pl.col("t_dat") >= recent_start) & (pl.col("t_dat") < window_start))
        .group_by("article_id")
        .agg(pl.len().alias("purchase_count"))
        # article_id as a tiebreak: Polars sort is not stable across runs for
        # ties on purchase_count alone, which made rank_in_strategy non-deterministic
        .sort(["purchase_count", "article_id"], descending=[True, False])
        .head(cap)
        .with_row_index("rank_in_strategy", offset=1)
        .select(["article_id", "rank_in_strategy"])
        .collect(engine="streaming")
    )


def recent_popularity_candidates(
    transactions: pl.DataFrame | pl.LazyFrame,
    customers: pl.DataFrame | pl.LazyFrame,
    window_start: date,
    cap: int = RECENT_POPULARITY_CAP,
    customer_ids: pl.Series | None = None,
    top_articles: pl.DataFrame | None = None,
) -> pl.DataFrame:
    if top_articles is None:
        top_articles = recent_popularity_table(transactions, window_start, cap)
    universe = customers.lazy().select("customer_id").unique()
    if customer_ids is not None:
        universe = universe.filter(pl.col("customer_id").is_in(customer_ids))
    return (
        universe.join(top_articles.lazy(), how="cross")
        .with_columns(pl.lit("recent_popularity").alias("strategy"))
        .select(["customer_id", "article_id", "strategy", "rank_in_strategy"])
        .collect(engine="streaming")
    )


def segment_popularity_table(
    transactions: pl.DataFrame | pl.LazyFrame,
    customers: pl.DataFrame | pl.LazyFrame,
    window_start: date,
    cap: int = SEGMENT_POPULARITY_CAP,
) -> pl.DataFrame:
    recent_start = window_start - POPULARITY_WINDOW
    customer_buckets = customers.lazy().select(["customer_id", _age_bucket_expr()])
    recent = transactions.lazy().filter(
        (pl.col("t_dat") >= recent_start) & (pl.col("t_dat") < window_start)
    )
    recent_with_bucket = recent.join(customer_buckets, on="customer_id", how="inner")

    return (
        recent_with_bucket.group_by(["age_bucket", "article_id"])
        .agg(pl.len().alias("purchase_count"))
        # article_id as a tiebreak: Polars sort is not stable across runs for
        # ties on purchase_count alone, which made rank_in_strategy non-deterministic
        .sort(["age_bucket", "purchase_count", "article_id"], descending=[False, True, False])
        .with_columns(
            (pl.int_range(pl.len()).over("age_bucket") + 1).alias("rank_in_strategy")
        )
        .filter(pl.col("rank_in_strategy") <= cap)
        .select(["age_bucket", "article_id", "rank_in_strategy"])
        .collect(engine="streaming")
    )


def segment_popularity_candidates(
    transactions: pl.DataFrame | pl.LazyFrame,
    customers: pl.DataFrame | pl.LazyFrame,
    window_start: date,
    cap: int = SEGMENT_POPULARITY_CAP,
    customer_ids: pl.Series | None = None,
    top_by_bucket: pl.DataFrame | None = None,
) -> pl.DataFrame:
    customer_buckets = customers.lazy().select(["customer_id", _age_bucket_expr()])
    if customer_ids is not None:
        customer_buckets = customer_buckets.filter(pl.col("customer_id").is_in(customer_ids))
    if top_by_bucket is None:
        top_by_bucket = segment_popularity_table(transactions, customers, window_start, cap)

    return (
        customer_buckets.join(top_by_bucket.lazy(), on="age_bucket", how="inner")
        .with_columns(pl.lit("segment_popularity").alias("strategy"))
        .select(["customer_id", "article_id", "strategy", "rank_in_strategy"])
        .collect(engine="streaming")
    )


def fit_als(
    transactions: pl.DataFrame | pl.LazyFrame,
    window_start: date,
    factors: int = 64,
    iterations: int = 15,
    random_state: int = ALS_RANDOM_STATE,
) -> tuple[AlternatingLeastSquares, pl.DataFrame, pl.DataFrame, sp.csr_matrix]:
    interactions = (
        transactions.lazy()
        .filter(pl.col("t_dat") < window_start)
        .group_by(["customer_id", "article_id"])
        .agg(pl.len().alias("weight"))
        .collect(engine="streaming")
    )

    customer_codes = (
        interactions.select("customer_id")
        .unique()
        .sort("customer_id")
        .with_row_index("customer_code")
    )
    article_codes = (
        interactions.select("article_id")
        .unique()
        .sort("article_id")
        .with_row_index("article_code")
    )
    interactions = interactions.join(customer_codes, on="customer_id").join(
        article_codes, on="article_id"
    )

    n_customers = customer_codes.height
    n_articles = article_codes.height
    user_items = sp.csr_matrix(
        (
            interactions["weight"].to_numpy(),
            (
                interactions["customer_code"].to_numpy(),
                interactions["article_code"].to_numpy(),
            ),
        ),
        shape=(n_customers, n_articles),
    )
    del interactions

    model = AlternatingLeastSquares(factors=factors, iterations=iterations, random_state=random_state)
    model.fit(user_items)

    return model, customer_codes, article_codes, user_items


def als_candidates(
    model: AlternatingLeastSquares,
    customer_codes: pl.DataFrame,
    article_codes: pl.DataFrame,
    user_items: sp.csr_matrix,
    cap: int = ALS_CAP,
    customer_ids: pl.Series | None = None,
    recommend_batch_size: int = ALS_RECOMMEND_BATCH_SIZE,
) -> pl.DataFrame:
    codes = customer_codes
    if customer_ids is not None:
        codes = codes.filter(pl.col("customer_id").is_in(customer_ids))
    target_codes = codes["customer_code"].to_numpy()

    if len(target_codes) == 0:
        return pl.DataFrame(
            schema={
                "customer_id": customer_codes["customer_id"].dtype,
                "article_id": article_codes["article_id"].dtype,
                "strategy": pl.Utf8,
                "rank_in_strategy": pl.Int64,
            }
        )

    # N > n_articles leaves implicit's padding sentinel incomplete (trailing slots read as a real score of 0.0),
    # so uncapped padding rows would silently pass the score filter below
    n_articles = user_items.shape[1]
    effective_cap = min(cap, n_articles)

    # top-N scoring needs a dense (batch_size, n_articles) score matrix; batching
    # bounds peak memory instead of scoring all requested customers in one dense pass
    batches = []
    for batch_start in range(0, len(target_codes), recommend_batch_size):
        batch_ids = target_codes[batch_start : batch_start + recommend_batch_size]
        item_ids, scores = model.recommend(
            batch_ids, user_items[batch_ids], N=effective_cap, filter_already_liked_items=True
        )
        batches.append(
            pl.DataFrame(
                {
                    "customer_code": np.repeat(batch_ids, item_ids.shape[1]),
                    "article_code": item_ids.reshape(-1),
                    "score": scores.reshape(-1),
                    "rank_in_strategy": np.tile(
                        np.arange(1, item_ids.shape[1] + 1), len(batch_ids)
                    ),
                }
            )
        )
    result = pl.concat(batches, how="vertical")
    # implicit pads short recommendation lists with a sentinel score, not a sentinel id
    result = result.filter(pl.col("score") > ALS_PADDING_SCORE)

    return (
        result.join(customer_codes, on="customer_code")
        .join(article_codes, on="article_code")
        .with_columns(pl.lit("als").alias("strategy"))
        .select(["customer_id", "article_id", "strategy", "rank_in_strategy"])
    )


STRATEGY_NAMES = [
    "repurchase",
    "colour_variant",
    "recent_popularity",
    "segment_popularity",
    "als",
]


def pool_candidates(
    strategy_frames: list[pl.DataFrame], customers: pl.DataFrame | pl.LazyFrame, cap: int = POOL_CAP
) -> pl.DataFrame:
    # rank_in_strategy dtype varies by strategy (row-index vs. rank vs. numpy); normalize before concat
    lazy_frames = [
        frame.lazy().with_columns(pl.col("rank_in_strategy").cast(pl.Int64))
        for frame in strategy_frames
    ]
    combined = pl.concat(lazy_frames, how="vertical")
    grouped = combined.group_by(["customer_id", "article_id"]).agg(
        # sorted: group_by/unique element order is not guaranteed stable across runs
        pl.col("strategy").unique().sort().alias("strategies"),
        pl.col("strategy").n_unique().alias("n_strategies"),
        pl.col("rank_in_strategy").min().alias("best_rank"),
        # one rank column per strategy, null where that strategy did not produce the candidate;
        # best_rank alone cannot tell a most-recent repurchase from a top ALS score
        *[
            pl.col("rank_in_strategy")
            .filter(pl.col("strategy") == strategy)
            .min()
            .alias(f"rank_{strategy}")
            for strategy in STRATEGY_NAMES
        ],
    )

    # article_id as a final tiebreak: Polars sort is not stable across runs for
    # ties on (n_strategies, best_rank) alone, which made the cap boundary non-deterministic
    ordered = grouped.sort(
        ["customer_id", "n_strategies", "best_rank", "article_id"],
        descending=[False, True, False, False],
    ).with_columns((pl.int_range(pl.len()).over("customer_id") + 1).alias("candidate_rank"))
    pooled = (
        ordered.filter(pl.col("candidate_rank") <= cap)
        .drop("candidate_rank")
        .collect(engine="streaming")
    )

    universe = customers.lazy().select("customer_id").unique().collect(engine="streaming")
    covered = pooled.select("customer_id").unique()
    missing = universe.join(covered, on="customer_id", how="anti")
    if missing.height > 0:
        raise ValueError(f"{missing.height} customers received zero candidates after pooling")

    return pooled


def generate_candidates_chunked(
    transactions: pl.DataFrame | pl.LazyFrame,
    articles: pl.DataFrame | pl.LazyFrame,
    customers: pl.DataFrame | pl.LazyFrame,
    window_start: date,
    output_dir: Path,
    chunk_size: int = CUSTOMER_CHUNK_SIZE,
    pool_cap: int = POOL_CAP,
    max_chunks: int | None = None,
    repurchase_cap: int = REPURCHASE_CAP,
    colour_variant_cap: int = COLOUR_VARIANT_CAP,
    recent_popularity_cap: int = RECENT_POPULARITY_CAP,
    segment_popularity_cap: int = SEGMENT_POPULARITY_CAP,
    als_cap: int = ALS_CAP,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    # stale partitions from an earlier chunk size are read by every part-*.parquet glob
    publish.clear_globbed(output_dir, "part-*.parquet")

    # global artifacts, fit once: both are small (a few hundred rows and one
    # factor model), unlike the per-customer candidate expansion below
    top_articles = recent_popularity_table(transactions, window_start, cap=recent_popularity_cap)
    top_by_bucket = segment_popularity_table(
        transactions, customers, window_start, cap=segment_popularity_cap
    )
    model, customer_codes, article_codes, user_items = fit_als(transactions, window_start)

    all_customer_ids = (
        customers.lazy().select("customer_id").unique().sort("customer_id").collect()["customer_id"]
    )
    n_customers = len(all_customer_ids)

    chunk_reports = []
    for chunk_idx, start in enumerate(range(0, n_customers, chunk_size)):
        if max_chunks is not None and chunk_idx >= max_chunks:
            break

        chunk_ids = all_customer_ids[start : start + chunk_size]
        chunk_customers = pl.DataFrame({"customer_id": chunk_ids})

        rep = repurchase_candidates(
            transactions, window_start, customer_ids=chunk_ids, cap=repurchase_cap
        )
        cv = colour_variant_candidates(
            transactions, articles, window_start, customer_ids=chunk_ids, cap=colour_variant_cap
        )
        rp = recent_popularity_candidates(
            transactions, customers, window_start, customer_ids=chunk_ids, top_articles=top_articles
        )
        segp = segment_popularity_candidates(
            transactions, customers, window_start, customer_ids=chunk_ids, top_by_bucket=top_by_bucket
        )
        als = als_candidates(
            model, customer_codes, article_codes, user_items, customer_ids=chunk_ids, cap=als_cap
        )

        pooled = pool_candidates([rep, cv, rp, segp, als], chunk_customers, cap=pool_cap)
        pooled.write_parquet(output_dir / f"part-{chunk_idx:04d}.parquet")

        chunk_reports.append(
            {
                "chunk_idx": chunk_idx,
                "n_customers": len(chunk_ids),
                "rows_repurchase": rep.height,
                "rows_colour_variant": cv.height,
                "rows_recent_popularity": rp.height,
                "rows_segment_popularity": segp.height,
                "rows_als": als.height,
                "rows_pooled": pooled.height,
            }
        )

    return chunk_reports
