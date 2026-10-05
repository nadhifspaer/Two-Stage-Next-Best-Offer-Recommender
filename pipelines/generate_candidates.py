# Standalone candidate generation: writes Parquet partitions and a manifest the candidates notebook reads.
import src.env_config  # noqa: F401 - must stay the first import: sets OPENBLAS_NUM_THREADS before any numeric library loads

import json
import threading
import time
from pathlib import Path

import polars as pl
import psutil

import src.candidates as candidates
import src.dataset as dataset

DATA_DIR = Path("data/processed")
OUTPUT_DIR = DATA_DIR / "candidates" / "validation"
MANIFEST_PATH = OUTPUT_DIR / "_manifest.json"


def _peak_rss_sampler(peak: dict, stop_flag: threading.Event, interval_s: float = 0.3) -> None:
    proc = psutil.Process()
    while not stop_flag.is_set():
        rss = proc.memory_info().rss
        if rss > peak["bytes"]:
            peak["bytes"] = rss
        time.sleep(interval_s)


def run() -> dict:
    pl.enable_string_cache()

    peak_rss = {"bytes": 0}
    stop_flag = threading.Event()
    sampler = threading.Thread(
        target=_peak_rss_sampler, args=(peak_rss, stop_flag), daemon=True
    )
    sampler.start()
    t0 = time.time()

    transactions = pl.scan_parquet(DATA_DIR / "transactions_train.parquet").with_columns(
        pl.col("customer_id", "article_id").cast(pl.Categorical)
    )
    articles = pl.read_parquet(DATA_DIR / "articles.parquet").with_columns(
        pl.col("article_id", "product_code").cast(pl.Categorical)
    )
    customers = pl.read_parquet(DATA_DIR / "customers.parquet").with_columns(
        pl.col("customer_id").cast(pl.Categorical)
    )

    validation_window = next(w for w in dataset.label_windows() if w["split"] == "validation")
    window_start = validation_window["window_start"]

    chunk_reports = candidates.generate_candidates_chunked(
        transactions, articles, customers, window_start, OUTPUT_DIR
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
        "window_start": str(window_start),
        "chunk_size": candidates.CUSTOMER_CHUNK_SIZE,
        "als_random_state": candidates.ALS_RANDOM_STATE,
        "n_chunks": len(chunk_reports),
        "wall_time_s": round(wall_time_s, 1),
        "peak_rss_gb": round(peak_rss["bytes"] / 1e9, 2),
        "totals": totals,
        "chunk_reports": chunk_reports,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
