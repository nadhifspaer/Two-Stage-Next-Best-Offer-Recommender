# Pooled candidates per training window, each with its own ALS fit, one subprocess per window.
import src.env_config  # noqa: F401 - must stay the first import: sets OPENBLAS_NUM_THREADS before any numeric library loads

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import polars as pl
import psutil

import src.candidates as candidates
import src.dataset as dataset

DATA_DIR = Path("data/processed")
CANDIDATES_ROOT = DATA_DIR / "candidates" / "train"


def _peak_rss_sampler(peak: dict, stop_flag: threading.Event, interval_s: float = 0.3) -> None:
    proc = psutil.Process()
    while not stop_flag.is_set():
        rss = proc.memory_info().rss
        if rss > peak["bytes"]:
            peak["bytes"] = rss
        time.sleep(interval_s)


def generate_for_window(window_start_iso: str) -> dict:
    """Runs in its own subprocess: one training label window in, its pooled candidate partitions out."""
    pl.enable_string_cache()
    peak = {"bytes": 0}
    stop_flag = threading.Event()
    sampler = threading.Thread(target=_peak_rss_sampler, args=(peak, stop_flag), daemon=True)
    sampler.start()
    t0 = time.time()

    window_start = dataset.date.fromisoformat(window_start_iso)
    output_dir = CANDIDATES_ROOT / window_start_iso

    transactions = pl.scan_parquet(DATA_DIR / "transactions_train.parquet").with_columns(
        pl.col("customer_id", "article_id").cast(pl.Categorical)
    )
    articles = pl.read_parquet(DATA_DIR / "articles.parquet").with_columns(
        pl.col("article_id", "product_code").cast(pl.Categorical)
    )
    customers = pl.read_parquet(DATA_DIR / "customers.parquet").with_columns(
        pl.col("customer_id").cast(pl.Categorical)
    )

    chunk_reports = candidates.generate_candidates_chunked(
        transactions, articles, customers, window_start, output_dir
    )

    wall_time_s = time.time() - t0
    stop_flag.set()
    sampler.join(timeout=2)

    totals = {
        "repurchase": sum(r["rows_repurchase"] for r in chunk_reports),
        "colour_variant": sum(r["rows_colour_variant"] for r in chunk_reports),
        "recent_popularity": sum(r["rows_recent_popularity"] for r in chunk_reports),
        "segment_popularity": sum(r["rows_segment_popularity"] for r in chunk_reports),
        "als": sum(r["rows_als"] for r in chunk_reports),
        "pooled": sum(r["rows_pooled"] for r in chunk_reports),
    }
    manifest = {
        "window_start": window_start_iso,
        "n_chunks": len(chunk_reports),
        "wall_time_s": round(wall_time_s, 1),
        "peak_rss_gb": round(peak["bytes"] / 1e9, 2),
        "totals": totals,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def run() -> dict:
    train_windows = [w for w in dataset.label_windows() if w["split"] == "train"]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())

    reports = []
    for w in train_windows:
        window_start_iso = str(w["window_start"])
        proc = subprocess.run(
            [sys.executable, __file__, "--window", window_start_iso],
            cwd=Path.cwd(),
            env=env,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"training window {window_start_iso} failed:\n{proc.stderr}")
        reports.append(json.loads(proc.stdout.strip().splitlines()[-1]))

    return {"windows": reports}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=str, default=None)
    args = parser.parse_args()

    if args.window is not None:
        print(json.dumps(generate_for_window(args.window)))
    else:
        result = run()
        print(json.dumps(result, indent=2))
