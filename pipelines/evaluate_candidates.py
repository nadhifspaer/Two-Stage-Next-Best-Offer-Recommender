# Recall@100 overall, per strategy, marginal per strategy and by recency bucket, one pass per partition.
import src.env_config  # noqa: F401 - must stay the first import: sets OPENBLAS_NUM_THREADS before any numeric library loads

import json
import os
import threading
import time
from pathlib import Path

import polars as pl
import psutil

import src.dataset as dataset
import src.evaluate as evaluate

DATA_DIR = Path("data/processed")
CANDIDATES_DIR = DATA_DIR / "candidates" / "validation"
RECALL_MANIFEST_PATH = DATA_DIR / "candidates" / "_recall_manifest.json"

STRATEGIES = ["repurchase", "colour_variant", "recent_popularity", "segment_popularity", "als"]
POOL_CAP = 100


def _strategy_subset(part: pl.DataFrame, strategy: str) -> pl.DataFrame:
    return part.filter(pl.col("strategies").list.contains(strategy)).select(
        "customer_id", "article_id"
    )


def _pool_without_strategy(part: pl.DataFrame, strategy: str) -> pl.DataFrame:
    # a candidate drops out only if this strategy was its sole contributor;
    # candidates other strategies also found stay in the pool
    return part.filter(
        ~((pl.col("n_strategies") == 1) & pl.col("strategies").list.contains(strategy))
    ).select("customer_id", "article_id")


def _peak_rss_sampler(peak: dict, stop_flag: threading.Event, interval_s: float = 0.3) -> None:
    proc = psutil.Process()
    while not stop_flag.is_set():
        rss = proc.memory_info().rss
        if rss > peak["bytes"]:
            peak["bytes"] = rss
        time.sleep(interval_s)


def publishes_manifest(candidates_dir: Path, window: dict | None) -> bool:
    # only the committed validation run publishes; sweeps and training windows do not
    return window is None and Path(candidates_dir) == CANDIDATES_DIR


def write_recall_manifest(result: dict, path: Path = RECALL_MANIFEST_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def run(candidates_dir: Path = CANDIDATES_DIR, window: dict | None = None) -> dict:
    publish = publishes_manifest(candidates_dir, window)
    pl.enable_string_cache()

    peak_rss = {"bytes": 0}
    stop_flag = threading.Event()
    sampler = threading.Thread(target=_peak_rss_sampler, args=(peak_rss, stop_flag), daemon=True)
    sampler.start()
    t0 = time.time()

    transactions = pl.scan_parquet(DATA_DIR / "transactions_train.parquet").with_columns(
        pl.col("customer_id", "article_id").cast(pl.Categorical)
    )
    customers = pl.read_parquet(DATA_DIR / "customers.parquet").with_columns(
        pl.col("customer_id").cast(pl.Categorical)
    )

    if window is None:
        window = next(w for w in dataset.label_windows() if w["split"] == "validation")
    window_start = window["window_start"]
    window_end = window["window_end"]

    actual_purchases = (
        transactions.filter((pl.col("t_dat") >= window_start) & (pl.col("t_dat") <= window_end))
        .select("customer_id", "article_id")
        .unique()
        .collect(engine="streaming")
    )
    total_true_pairs = actual_purchases.height

    # small (customer_id, recency_bucket) table, held in memory across partitions
    recency = evaluate.customer_recency_buckets(transactions, customers, window_start)
    bucket_labels = evaluate.RECENCY_BUCKET_LABELS + ["never"]

    pool_hits = 0
    strategy_hits = {s: 0 for s in STRATEGIES}
    without_hits = {s: 0 for s in STRATEGIES}
    bucket_hits = {b: 0 for b in bucket_labels}
    bucket_true_pairs = {b: 0 for b in bucket_labels}
    seen_true_pairs = 0

    partition_paths = sorted(Path(candidates_dir).glob("part-*.parquet"))
    if not partition_paths:
        raise FileNotFoundError(f"no part-*.parquet files under {candidates_dir}")

    for path in partition_paths:
        part = pl.read_parquet(path)

        over_cap = (
            part.group_by("customer_id").agg(pl.len().alias("n")).filter(pl.col("n") > POOL_CAP)
        )
        if over_cap.height > 0:
            raise ValueError(f"{over_cap.height} customers have more than {POOL_CAP} candidates in {path}")

        part_customers = part.select("customer_id").unique()
        # every customer belongs to exactly one partition, so this partition's
        # true purchases never overlap another partition's
        part_true = actual_purchases.join(part_customers, on="customer_id", how="semi")
        seen_true_pairs += part_true.height

        part_candidates = part.select("customer_id", "article_id").unique()
        pool_hits += part_true.join(part_candidates, on=["customer_id", "article_id"], how="semi").height

        for strategy in STRATEGIES:
            strategy_hits[strategy] += part_true.join(
                _strategy_subset(part, strategy), on=["customer_id", "article_id"], how="semi"
            ).height
            without_hits[strategy] += part_true.join(
                _pool_without_strategy(part, strategy), on=["customer_id", "article_id"], how="semi"
            ).height

        part_true_bucketed = part_true.join(recency, on="customer_id", how="left")
        for bucket in bucket_labels:
            bucket_part_true = part_true_bucketed.filter(pl.col("recency_bucket") == bucket)
            bucket_true_pairs[bucket] += bucket_part_true.height
            bucket_hits[bucket] += bucket_part_true.join(
                part_candidates, on=["customer_id", "article_id"], how="semi"
            ).height

        del part, part_candidates, part_true, part_true_bucketed

    if seen_true_pairs != total_true_pairs:
        raise ValueError(
            f"partition true-pair counts summed to {seen_true_pairs}, expected {total_true_pairs}: "
            "a customer appears in more than one partition"
        )
    if sum(bucket_true_pairs.values()) != total_true_pairs:
        raise ValueError("recency-bucket true-pair counts do not sum to the total")

    pool_recall = pool_hits / total_true_pairs
    per_strategy = []
    for strategy in STRATEGIES:
        per_strategy.append(
            {
                "strategy": strategy,
                "recall_at_100": strategy_hits[strategy] / total_true_pairs,
                "marginal_contribution": pool_recall - without_hits[strategy] / total_true_pairs,
            }
        )

    per_recency_bucket = []
    for bucket in bucket_labels:
        if bucket_true_pairs[bucket] == 0:
            per_recency_bucket.append({"recency_bucket": bucket, "true_pairs": 0, "recall_at_100": None})
        else:
            per_recency_bucket.append(
                {
                    "recency_bucket": bucket,
                    "true_pairs": bucket_true_pairs[bucket],
                    "recall_at_100": bucket_hits[bucket] / bucket_true_pairs[bucket],
                }
            )

    wall_time_s = time.time() - t0
    stop_flag.set()
    sampler.join(timeout=2)

    result = {
        "window_start": str(window_start),
        "window_end": str(window_end),
        "partitions": len(partition_paths),
        "true_pairs": total_true_pairs,
        "pool_hits": pool_hits,
        "pool_recall_at_100": pool_recall,
        "per_strategy": per_strategy,
        "per_recency_bucket": per_recency_bucket,
        "wall_time_s": round(wall_time_s, 1),
        "peak_rss_gb": round(peak_rss["bytes"] / 1e9, 2),
    }
    if publish:
        write_recall_manifest(result)
    return result


def _print_report(result: dict) -> None:
    print(f"wall_time_s={result['wall_time_s']} peak_rss_gb={result['peak_rss_gb']}")
    print(f"validation window {result['window_start']} to {result['window_end']}")
    print(f"true (customer, article) purchase pairs: {result['true_pairs']}")
    print(f"pooled Recall@100: {result['pool_recall_at_100']:.4f}")
    print()
    print("per-strategy recall and marginal contribution to the pool:")
    for row in result["per_strategy"]:
        print(
            f"  {row['strategy']:<20} recall@100={row['recall_at_100']:.4f}"
            f"  marginal={row['marginal_contribution']:+.4f}"
        )
    print()
    print("pooled recall by customer recency bucket (as of window start):")
    for row in result["per_recency_bucket"]:
        if row["recall_at_100"] is None:
            print(f"  {row['recency_bucket']:<25} true_pairs=0")
        else:
            print(
                f"  {row['recency_bucket']:<25} true_pairs={row['true_pairs']:<8}"
                f" recall@100={row['recall_at_100']:.4f}"
            )


if __name__ == "__main__":
    result = run()
    _print_report(result)
