# Scores every 2020-09-09 candidate with the backtest ranker, keeps the top 12 per customer, batched per partition.
import src.env_config  # noqa: F401 - must stay the first import: sets OPENBLAS_NUM_THREADS before any numeric library loads

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import mlflow.artifacts
import mlflow.lightgbm
import polars as pl
import psutil

import src.publish as publish
import src.rank as rank
import src.tracking as tracking

DATA_DIR = Path("data/processed")
FEATURES_DIR = DATA_DIR / "features" / "backtest_2020-09-09"
BACKTEST_RUN_MANIFEST = DATA_DIR / "evaluation" / "_backtest_train_manifest.json"
OUTPUT_DIR = DATA_DIR / "evaluation" / "backtest_ranker_top12"
MANIFEST_PATH = DATA_DIR / "evaluation" / "_backtest_ranker_score_manifest.json"

BATCH_SIZE = 1_000_000
TOP_K = 12


def _peak_rss_sampler(peak: dict, stop_flag: threading.Event, interval_s: float = 0.3) -> None:
    proc = psutil.Process()
    while not stop_flag.is_set():
        rss = proc.memory_info().rss
        if rss > peak["bytes"]:
            peak["bytes"] = rss
        time.sleep(interval_s)


def _start_sampler():
    peak = {"bytes": 0}
    stop_flag = threading.Event()
    sampler = threading.Thread(target=_peak_rss_sampler, args=(peak, stop_flag), daemon=True)
    sampler.start()
    return peak, stop_flag, sampler


def _load_backtest_ranker_and_config():
    run_manifest = json.loads(BACKTEST_RUN_MANIFEST.read_text(encoding="utf-8"))
    tracking.configure_mlflow()
    run_id = run_manifest["mlflow_run_id"]
    ranker = mlflow.lightgbm.load_model(f"runs:/{run_id}/ranker")
    feature_columns = mlflow.artifacts.load_dict(f"runs:/{run_id}/feature_columns.json")
    categorical_mapping = mlflow.artifacts.load_dict(f"runs:/{run_id}/categorical_mapping.json")
    assert feature_columns == rank.FEATURE_COLUMNS
    return ranker, categorical_mapping


def score_partition(partition_prefix: str) -> dict:
    """Runs in its own subprocess: one candidate partition's batch files in,
    that partition's top-12-by-ranker-score-per-customer out."""
    pl.enable_string_cache()
    peak, stop_flag, sampler = _start_sampler()
    t0 = time.time()

    ranker, categorical_mapping = _load_backtest_ranker_and_config()

    batch_paths = sorted(FEATURES_DIR.glob(f"{partition_prefix}-batch-*.parquet"))
    score_frames = []
    rows_scored = 0
    for batch_path in batch_paths:
        part = pl.read_parquet(batch_path, columns=["customer_id", "article_id", *rank.FEATURE_COLUMNS])
        for offset in range(0, part.height, BATCH_SIZE):
            sub = part.slice(offset, BATCH_SIZE)
            encoded = rank.apply_categorical_mapping(sub, categorical_mapping)
            X = rank.to_pandas_X(encoded)
            scores = ranker.predict(X).astype("float32")
            score_frames.append(
                pl.DataFrame(
                    {
                        "customer_id": sub["customer_id"],
                        "article_id": sub["article_id"],
                        "ranker_score": scores,
                    }
                )
            )
            rows_scored += sub.height
            del sub, encoded, X, scores
        del part

    all_scores = pl.concat(score_frames, how="vertical")
    top12 = (
        all_scores.sort(["customer_id", "ranker_score", "article_id"], descending=[False, True, False])
        .with_columns((pl.int_range(pl.len()).over("customer_id") + 1).alias("rank_position"))
        .filter(pl.col("rank_position") <= TOP_K)
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    top12.write_parquet(OUTPUT_DIR / f"{partition_prefix}.parquet")

    wall_time_s = time.time() - t0
    stop_flag.set()
    sampler.join(timeout=2)

    return {
        "partition": partition_prefix,
        "rows_scored": rows_scored,
        "customers": top12["customer_id"].n_unique(),
        "wall_time_s": round(wall_time_s, 1),
        "peak_rss_gb": round(peak["bytes"] / 1e9, 2),
    }


def run() -> dict:
    t0 = time.time()
    publish.clear_globbed(OUTPUT_DIR, "part-*.parquet")
    publish.assert_rows_match_manifest(
        FEATURES_DIR, "part-*-batch-*.parquet", FEATURES_DIR / "_manifest.json"
    )
    partition_prefixes = sorted(
        {p.name.split("-batch-")[0] for p in FEATURES_DIR.glob("part-*-batch-*.parquet")}
    )
    if not partition_prefixes:
        raise FileNotFoundError(f"no part-*-batch-*.parquet files under {FEATURES_DIR}")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())

    reports = []
    for prefix in partition_prefixes:
        proc = subprocess.run(
            [sys.executable, __file__, "--partition", prefix],
            cwd=Path.cwd(),
            env=env,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"partition {prefix} failed:\n{proc.stderr}")
        reports.append(json.loads(proc.stdout.strip().splitlines()[-1]))

    wall_time_s = time.time() - t0
    manifest = {
        "total_rows_scored": sum(r["rows_scored"] for r in reports),
        "total_customers": sum(r["customers"] for r in reports),
        "wall_time_s": round(wall_time_s, 1),
        "peak_rss_gb": max(r["peak_rss_gb"] for r in reports),
        "partition_reports": reports,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", type=str, default=None)
    args = parser.parse_args()

    if args.partition is not None:
        print(json.dumps(score_partition(args.partition)))
    else:
        result = run()
        print(json.dumps(result, indent=2))
