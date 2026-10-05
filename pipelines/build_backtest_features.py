# Full-pool feature assembly for the 2020-09-09 backtest window, same design as build_features.py.
import src.env_config  # noqa: F401 - must stay the first import: sets OPENBLAS_NUM_THREADS before any numeric library loads

import argparse
import gc
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import polars as pl
import psutil

import src.dataset as dataset
import src.features as features
import src.phase_peaks as phase_peaks
import src.publish as publish
import src.text_embed as text_embed

DATA_DIR = Path("data/processed")
BACKTEST_WINDOW_START = dataset.date(2020, 9, 9)
CANDIDATES_DIR = DATA_DIR / "candidates" / "train" / "2020-09-09"
OUTPUT_DIR = DATA_DIR / "features" / "backtest_2020-09-09"
HISTORY_DIR = OUTPUT_DIR / "_history"
MANIFEST_PATH = OUTPUT_DIR / "_manifest.json"
CUSTOMER_BLOCK_PATH = OUTPUT_DIR / "_customer_block.parquet"
ARTICLE_BLOCK_PATH = OUTPUT_DIR / "_article_block.parquet"
RECENT_TEXT_BLOCK_PATH = OUTPUT_DIR / "_recent_text_block.parquet"


def _peak_memory_sampler(peak: dict, stop_flag: threading.Event, interval_s: float = 0.3) -> None:
    proc = psutil.Process()
    while not stop_flag.is_set():
        full = proc.memory_full_info()
        if full.rss > peak["rss_bytes"]:
            peak["rss_bytes"] = full.rss
        if full.uss > peak["uss_bytes"]:
            peak["uss_bytes"] = full.uss
        time.sleep(interval_s)


def _start_sampler() -> tuple[dict, threading.Event, threading.Thread]:
    peak = {"rss_bytes": 0, "uss_bytes": 0}
    stop_flag = threading.Event()
    sampler = threading.Thread(target=_peak_memory_sampler, args=(peak, stop_flag), daemon=True)
    sampler.start()
    return peak, stop_flag, sampler


def _stop_sampler(stop_flag: threading.Event, sampler: threading.Thread) -> None:
    stop_flag.set()
    sampler.join(timeout=2)


def _load_global_blocks() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    return (
        pl.read_parquet(CUSTOMER_BLOCK_PATH).with_columns(pl.col("customer_id").cast(pl.Categorical)),
        pl.read_parquet(ARTICLE_BLOCK_PATH).with_columns(pl.col("article_id").cast(pl.Categorical)),
        pl.read_parquet(RECENT_TEXT_BLOCK_PATH).with_columns(
            pl.col("customer_id").cast(pl.Categorical)
        ),
    )


def _write_history_partitions(transactions: pl.LazyFrame, partition_paths: list[Path]) -> None:
    chunk_maps = []
    for chunk_idx, path in enumerate(partition_paths):
        ids = pl.read_parquet(path, columns=["customer_id"]).unique()
        chunk_maps.append(ids.with_columns(pl.lit(chunk_idx).alias("chunk_idx")))
    chunk_map = pl.concat(chunk_maps, how="vertical")

    before = (
        transactions.filter(pl.col("t_dat") < BACKTEST_WINDOW_START)
        .select("customer_id", "article_id", "t_dat")
        .join(chunk_map.lazy(), on="customer_id", how="inner")
        .collect(engine="streaming")
    )

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    for chunk_idx, path in enumerate(partition_paths):
        before.filter(pl.col("chunk_idx") == chunk_idx).drop("chunk_idx").write_parquet(
            HISTORY_DIR / path.name
        )
    del before, chunk_map, chunk_maps
    gc.collect()


def process_partition(path: Path) -> dict:
    """Runs in its own subprocess: one candidate partition in, one feature partition out."""
    pl.enable_string_cache()

    peak, stop_flag, sampler = _start_sampler()
    t0 = time.time()

    articles = pl.read_parquet(DATA_DIR / "articles.parquet").with_columns(
        pl.col("article_id", "product_code").cast(pl.Categorical)
    )
    text_vectors = text_embed.load_article_text_vectors().with_columns(
        pl.col("article_id").cast(pl.Categorical)
    )
    customer_block, article_block, recent_text_block = _load_global_blocks()
    history = pl.read_parquet(HISTORY_DIR / path.name).with_columns(
        pl.col("customer_id", "article_id").cast(pl.Categorical)
    )

    candidates = pl.read_parquet(path)
    pair = features.customer_article_pair_features(
        history,
        articles,
        candidates,
        customer_block,
        article_block,
        text_vectors,
        recent_text_block,
        BACKTEST_WINDOW_START,
    )
    provenance = features.candidate_provenance_features(candidates)
    rows_written = features.assemble_partition_features(
        pair,
        provenance,
        customer_block,
        article_block,
        write_dir=OUTPUT_DIR,
        file_stem=path.stem,
    )
    del candidates, pair, provenance, history
    gc.collect()

    wall_time_s = time.time() - t0
    _stop_sampler(stop_flag, sampler)

    return {
        "partition": path.name,
        "rows_written": rows_written,
        "wall_time_s": round(wall_time_s, 1),
        "peak_rss_gb": round(peak["rss_bytes"] / 1e9, 2),
        "peak_uss_gb": round(peak["uss_bytes"] / 1e9, 2),
    }


def run() -> dict:
    pl.enable_string_cache()
    t0 = time.time()
    peak, stop_flag, sampler = _start_sampler()
    phases = phase_peaks.PhasePeaks()

    transactions = pl.scan_parquet(DATA_DIR / "transactions_train.parquet").with_columns(
        pl.col("customer_id", "article_id").cast(pl.Categorical)
    )
    articles = pl.read_parquet(DATA_DIR / "articles.parquet").with_columns(
        pl.col("article_id", "product_code").cast(pl.Categorical)
    )
    customers = pl.read_parquet(DATA_DIR / "customers.parquet").with_columns(
        pl.col("customer_id").cast(pl.Categorical)
    )
    text_vectors = text_embed.load_article_text_vectors().with_columns(
        pl.col("article_id").cast(pl.Categorical)
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # stale batch files from an earlier layout are read by every scorer's glob
    publish.clear_globbed(OUTPUT_DIR, "part-*-batch-*.parquet")
    partition_paths = sorted(CANDIDATES_DIR.glob("part-*.parquet"))
    if not partition_paths:
        raise FileNotFoundError(f"no part-*.parquet files under {CANDIDATES_DIR}")

    # pre-window history materialized once (about 31.5M rows, five columns) so the
    # chunked block passes filter in memory instead of re-scanning parquet
    with phases.phase("materialize_history"):
        transactions = (
            transactions.filter(pl.col("t_dat") < BACKTEST_WINDOW_START)
            .select("customer_id", "article_id", "t_dat", "price", "sales_channel_id")
            .collect()
        )
    with phases.phase("customer_features"):
        customer_block = features.customer_features(transactions, customers, BACKTEST_WINDOW_START)
    with phases.phase("article_features"):
        article_block = features.article_features(transactions, articles, BACKTEST_WINDOW_START)
    with phases.phase("customer_recent_text_mean"):
        recent_text_block = features.customer_recent_text_mean(
            transactions, text_vectors, BACKTEST_WINDOW_START
        )
    customer_block.write_parquet(CUSTOMER_BLOCK_PATH)
    article_block.write_parquet(ARTICLE_BLOCK_PATH)
    recent_text_block.write_parquet(RECENT_TEXT_BLOCK_PATH)
    del customer_block, article_block, recent_text_block
    gc.collect()

    with phases.phase("history_split"):
        _write_history_partitions(transactions.lazy(), partition_paths)
    del transactions, articles, customers, text_vectors
    gc.collect()

    _stop_sampler(stop_flag, sampler)
    phases.stop()
    parent_phase_report = phases.report()
    parent_peak_rss_gb = max(round(peak["rss_bytes"] / 1e9, 2), phases.peak_rss_gb)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())

    rows_written = 0
    peak_rss_gb = round(peak["rss_bytes"] / 1e9, 2)
    peak_uss_gb = round(peak["uss_bytes"] / 1e9, 2)
    partition_reports = []
    for path in partition_paths:
        proc = subprocess.run(
            [sys.executable, __file__, "--partition", str(path)],
            cwd=Path.cwd(),
            env=env,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"partition {path.name} failed:\n{proc.stderr}")
        report = json.loads(proc.stdout.strip().splitlines()[-1])
        partition_reports.append(report)
        rows_written += report["rows_written"]
        peak_rss_gb = max(peak_rss_gb, report["peak_rss_gb"])
        peak_uss_gb = max(peak_uss_gb, report["peak_uss_gb"])

    wall_time_s = time.time() - t0

    manifest = {
        "window_start": str(BACKTEST_WINDOW_START),
        "partitions": len(partition_paths),
        "rows_written": rows_written,
        "wall_time_s": round(wall_time_s, 1),
        "peak_rss_gb": peak_rss_gb,
        "peak_uss_gb": peak_uss_gb,
        "parent_peak_rss_gb": parent_peak_rss_gb,
        "partition_peak_rss_gb": max(r["peak_rss_gb"] for r in partition_reports),
        "parent_phases": parent_phase_report,
        "partition_reports": partition_reports,
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", type=Path, default=None)
    args = parser.parse_args()

    if args.partition is not None:
        print(json.dumps(process_partition(args.partition)))
    else:
        result = run()
        print(json.dumps(result, indent=2))
