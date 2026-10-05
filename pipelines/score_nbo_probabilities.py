# Calibrated purchase probability for every ranker top-12 row, reusing persisted classifier scores where available.
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

import pipelines.score_validation_candidates as score_validation_candidates
import src.calibrate as calibrate
import src.publish as publish
import src.rank as rank

DATA_DIR = Path("data/processed")
RANKER_TOP12_DIR = DATA_DIR / "evaluation" / "ranker_top12"
CALIBRATION_SCORED_DIR = DATA_DIR / "calibration" / "scored"
FEATURES_DIR = DATA_DIR / "features" / "validation"
MODELS_DIR = DATA_DIR / "models"
CALIBRATOR_PATH = MODELS_DIR / "calibrator.joblib"
OUTPUT_DIR = DATA_DIR / "nbo" / "scored_top12"
MANIFEST_PATH = DATA_DIR / "nbo" / "_score_manifest.json"


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


def score_partition(partition_prefix: str) -> dict:
    """Runs in its own subprocess: one partition's ranker top-12 rows in, a
    (customer_id, article_id, ranker_score, rank_position, raw_prob, bucket, p_purchase) file out."""
    pl.enable_string_cache()
    peak, stop_flag, sampler = _start_sampler()
    t0 = time.time()

    top12 = pl.read_parquet(RANKER_TOP12_DIR / f"{partition_prefix}.parquet")
    persisted = pl.read_parquet(
        CALIBRATION_SCORED_DIR / f"{partition_prefix}.parquet",
        columns=["customer_id", "article_id", "raw_prob", "bucket"],
    )
    joined = top12.join(persisted, on=["customer_id", "article_id"], how="left")

    reused = joined.filter(pl.col("raw_prob").is_not_null())
    missing_keys = joined.filter(pl.col("raw_prob").is_null()).select("customer_id", "article_id").unique()

    if missing_keys.height > 0:
        classifier = score_validation_candidates._load_classifier()
        categorical_mapping = rank.load_categorical_mapping(MODELS_DIR / "categorical_mapping.json")

        batch_paths = sorted(FEATURES_DIR.glob(f"{partition_prefix}-batch-*.parquet"))
        fresh_frames = []
        for batch_path in batch_paths:
            part = pl.read_parquet(batch_path, columns=["customer_id", "article_id", *rank.FEATURE_COLUMNS])
            matched = part.join(missing_keys, on=["customer_id", "article_id"], how="semi")
            if matched.height > 0:
                encoded = rank.apply_categorical_mapping(matched, categorical_mapping)
                X = rank.to_pandas_X(encoded)
                raw_prob = classifier.predict_proba(X)[:, 1].astype("float32")
                fresh_frames.append(
                    pl.DataFrame(
                        {
                            "customer_id": matched["customer_id"],
                            "article_id": matched["article_id"],
                            "raw_prob": raw_prob,
                        }
                    )
                )
            del part, matched

        fresh = pl.concat(fresh_frames, how="vertical")
        if fresh.height != missing_keys.height:
            raise RuntimeError(
                f"{partition_prefix}: fresh-scored {fresh.height} rows, expected {missing_keys.height}"
            )

        missing_scored = (
            joined.filter(pl.col("raw_prob").is_null())
            .drop("raw_prob")
            .join(fresh, on=["customer_id", "article_id"], how="left")
            .select(reused.columns)
        )
        full = pl.concat([reused, missing_scored], how="vertical")
    else:
        full = reused

    calibrator = calibrate.load_calibrator(CALIBRATOR_PATH)
    calibrated_prob = calibrate.apply_calibration(calibrator, full["raw_prob"].to_numpy())
    full = full.with_columns(pl.Series("p_purchase", calibrated_prob))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    full.write_parquet(OUTPUT_DIR / f"{partition_prefix}.parquet")

    wall_time_s = time.time() - t0
    stop_flag.set()
    sampler.join(timeout=2)

    return {
        "partition": partition_prefix,
        "rows": full.height,
        "reused_rows": reused.height,
        "freshly_scored_rows": full.height - reused.height,
        "wall_time_s": round(wall_time_s, 1),
        "peak_rss_gb": round(peak["bytes"] / 1e9, 2),
    }


def run() -> dict:
    t0 = time.time()
    publish.clear_globbed(OUTPUT_DIR, "part-*.parquet")
    publish.assert_rows_match_manifest(
        FEATURES_DIR, "part-*-batch-*.parquet", FEATURES_DIR / "_manifest.json"
    )
    n_customers = publish.customer_count(DATA_DIR / "customers.parquet")
    publish.assert_top12_exact(RANKER_TOP12_DIR, n_customers)
    partition_prefixes = sorted(p.stem for p in RANKER_TOP12_DIR.glob("part-*.parquet"))
    if not partition_prefixes:
        raise FileNotFoundError(f"no part-*.parquet files under {RANKER_TOP12_DIR}")

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

    publish.assert_top12_exact(OUTPUT_DIR, n_customers)

    wall_time_s = time.time() - t0
    manifest = {
        "total_rows": sum(r["rows"] for r in reports),
        "reused_rows": sum(r["reused_rows"] for r in reports),
        "freshly_scored_rows": sum(r["freshly_scored_rows"] for r in reports),
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
