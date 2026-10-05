# MAP@12 on the 2020-09-09 backtest window for the four orderings, using the backtest ranker's scores.
import src.env_config  # noqa: F401 - must stay the first import: sets OPENBLAS_NUM_THREADS before any numeric library loads

import json
import threading
import time
from pathlib import Path

import polars as pl
import psutil

import src.candidates as candidates
import src.dataset as dataset
import src.evaluate as evaluate

DATA_DIR = Path("data/processed")
CANDIDATES_DIR = DATA_DIR / "candidates" / "train" / "2020-09-09"
RANKER_TOP12_DIR = DATA_DIR / "evaluation" / "backtest_ranker_top12"
MANIFEST_PATH = DATA_DIR / "evaluation" / "_backtest_map12_manifest.json"

TOP_K = 12
BACKTEST_WINDOW_START = dataset.date(2020, 9, 9)


def _peak_rss_sampler(peak: dict, stop_flag: threading.Event, interval_s: float = 0.3) -> None:
    proc = psutil.Process()
    while not stop_flag.is_set():
        rss = proc.memory_info().rss
        if rss > peak["bytes"]:
            peak["bytes"] = rss
        time.sleep(interval_s)


def run() -> dict:
    pl.enable_string_cache()
    peak = {"bytes": 0}
    stop_flag = threading.Event()
    sampler = threading.Thread(target=_peak_rss_sampler, args=(peak, stop_flag), daemon=True)
    sampler.start()
    t0 = time.time()

    backtest_window = next(
        w for w in dataset.label_windows() if w["window_start"] == BACKTEST_WINDOW_START
    )
    window_start = backtest_window["window_start"]
    window_end = backtest_window["window_end"]

    transactions = pl.scan_parquet(DATA_DIR / "transactions_train.parquet").with_columns(
        pl.col("customer_id", "article_id").cast(pl.Categorical)
    )
    actual_purchases = (
        transactions.filter((pl.col("t_dat") >= window_start) & (pl.col("t_dat") <= window_end))
        .select("customer_id", "article_id")
        .unique()
        .collect(engine="streaming")
    )
    purchasing_customers = actual_purchases.select("customer_id").unique()

    # ordering #1: ranker top-12, already scored per partition
    ranker_top12 = pl.concat(
        [pl.read_parquet(p) for p in sorted(RANKER_TOP12_DIR.glob("part-*.parquet"))]
    ).select("customer_id", "article_id", "rank_position")

    # ordering #2: recent popularity top-12, anchored at 2020-09-09, identical for every customer
    top12_popularity = candidates.recent_popularity_table(transactions, window_start, cap=TOP_K).rename(
        {"rank_in_strategy": "rank_position"}
    )
    popularity_top12 = (
        purchasing_customers.join(top12_popularity, how="cross")
        .select("customer_id", "article_id", "rank_position")
    )

    # ordering #3 (best_rank, no ranker) and oracle's m_found: single pass per
    # candidate partition, narrow columns only, same pattern as evaluate_map12.py
    partition_paths = sorted(CANDIDATES_DIR.glob("part-*.parquet"))
    if not partition_paths:
        raise FileNotFoundError(f"no part-*.parquet files under {CANDIDATES_DIR}")

    best_rank_frames = []
    m_found_frames = []
    for path in partition_paths:
        part = pl.read_parquet(path, columns=["customer_id", "article_id", "best_rank"])

        part_top12 = (
            part.sort(["customer_id", "best_rank", "article_id"], descending=[False, False, False])
            .with_columns((pl.int_range(pl.len()).over("customer_id") + 1).alias("rank_position"))
            .filter(pl.col("rank_position") <= TOP_K)
            .select("customer_id", "article_id", "rank_position")
        )
        best_rank_frames.append(part_top12)

        part_m_found = (
            part.join(actual_purchases, on=["customer_id", "article_id"], how="semi")
            .group_by("customer_id")
            .agg(pl.len().alias("m_found"))
        )
        m_found_frames.append(part_m_found)
        del part, part_top12, part_m_found

    best_rank_top12 = pl.concat(best_rank_frames, how="vertical")
    m_found = pl.concat(m_found_frames, how="vertical")

    # MAP@12 for the four orderings
    map_ranker, n_ranker = evaluate.map_at_12(ranker_top12, actual_purchases)
    map_popularity, n_popularity = evaluate.map_at_12(popularity_top12, actual_purchases)
    map_pool_order, n_pool_order = evaluate.map_at_12(best_rank_top12, actual_purchases)
    map_oracle, n_oracle = evaluate.oracle_map_at_12(m_found, actual_purchases)

    wall_time_s = time.time() - t0
    stop_flag.set()
    sampler.join(timeout=2)

    manifest = {
        "window_start": str(window_start),
        "window_end": str(window_end),
        "note": "2020-09-09 reused here as a backtest evaluation week; ranker trained "
        "only on 2020-08-19, 2020-08-26, 2020-09-02, fixed at 212 iterations, the count "
        "early stopping selected evaluating on this same window",
        "orderings": {
            "ranker": {"map_at_12": map_ranker, "customers_scored": n_ranker},
            "recent_popularity": {"map_at_12": map_popularity, "customers_scored": n_popularity},
            "candidate_pool_order": {"map_at_12": map_pool_order, "customers_scored": n_pool_order},
            "oracle": {"map_at_12": map_oracle, "customers_scored": n_oracle},
        },
        "ranker_ratio_vs": {
            "recent_popularity": map_ranker / map_popularity,
            "candidate_pool_order": map_ranker / map_pool_order,
            "oracle": map_ranker / map_oracle,
        },
        "wall_time_s": round(wall_time_s, 1),
        "peak_rss_gb": round(peak["bytes"] / 1e9, 2),
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
