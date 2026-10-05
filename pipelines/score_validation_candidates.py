# Scores the classifier over the full validation candidate pool, keeps only the columns calibration needs.
import src.env_config  # noqa: F401 - must stay the first import: sets OPENBLAS_NUM_THREADS before any numeric library loads

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import mlflow.lightgbm
import polars as pl
import psutil

import src.calibrate as calibrate
import src.dataset as dataset
import src.production_models as production_models
import src.publish as publish
import src.rank as rank
import src.tracking as tracking

DATA_DIR = Path("data/processed")
FEATURES_DIR = DATA_DIR / "features" / "validation"
MODELS_DIR = DATA_DIR / "models"
OUTPUT_DIR = DATA_DIR / "calibration" / "scored"
MANIFEST_PATH = DATA_DIR / "calibration" / "_score_manifest.json"


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


def _load_classifier():
    tracking.configure_mlflow()
    production = production_models.load_production_models()
    return mlflow.lightgbm.load_model(f"runs:/{production.classifier_run_id}/classifier")


def score_partition(partition_prefix: str) -> dict:
    """Runs in its own subprocess: one candidate partition's batch files in,
    a small (customer_id, article_id, label, raw_prob, bucket) file out."""
    pl.enable_string_cache()
    peak, stop_flag, sampler = _start_sampler()
    t0 = time.time()

    classifier = _load_classifier()
    categorical_mapping = rank.load_categorical_mapping(MODELS_DIR / "categorical_mapping.json")

    validation_window = next(w for w in dataset.label_windows() if w["split"] == "validation")
    transactions = pl.scan_parquet(DATA_DIR / "transactions_train.parquet").with_columns(
        pl.col("customer_id", "article_id").cast(pl.Categorical)
    )
    actual = (
        transactions.filter(
            (pl.col("t_dat") >= validation_window["window_start"])
            & (pl.col("t_dat") <= validation_window["window_end"])
        )
        .select("customer_id", "article_id")
        .unique()
        .with_columns(pl.lit(1).cast(pl.Int8).alias("label"))
        .collect(engine="streaming")
    )

    # one ~1M-row batch file at a time, never the whole ~7.5M-row partition: concat, join, encode and pandas
    # conversion each copy it, and stacking those copies measured 9+ GB peak RSS
    batch_paths = sorted(FEATURES_DIR.glob(f"{partition_prefix}-batch-*.parquet"))

    sum_prob = 0.0
    sum_label = 0
    count = 0
    keep_frames = []

    for batch_path in batch_paths:
        part = pl.read_parquet(batch_path).with_columns(
            pl.col("customer_id", "article_id").cast(pl.Categorical)
        )
        labeled = (
            part.lazy()
            .join(actual.lazy(), on=["customer_id", "article_id"], how="left")
            .with_columns(pl.col("label").fill_null(0).cast(pl.Int8))
            .collect(engine="streaming")
        )
        encoded = rank.apply_categorical_mapping(labeled, categorical_mapping)
        X = rank.to_pandas_X(encoded)
        raw_prob = classifier.predict_proba(X)[:, 1].astype("float32")
        bucket = calibrate.customer_bucket(labeled["customer_id"])

        result = pl.DataFrame(
            {
                "customer_id": labeled["customer_id"],
                "article_id": labeled["article_id"],
                "label": labeled["label"],
                "raw_prob": raw_prob,
                "bucket": bucket,
            }
        )

        # fsum over float64: independent of row order within the batch
        sum_prob += math.fsum(raw_prob.astype("float64"))
        sum_label += int(result["label"].sum())
        count += result.height

        keep_frames.append(
            result.filter(calibrate.fit_bucket_mask(result["bucket"]) | calibrate.eval_bucket_mask(result["bucket"]))
        )
        del part, labeled, encoded, X, raw_prob, result

    keep = pl.concat(keep_frames, how="vertical")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    keep.write_parquet(OUTPUT_DIR / f"{partition_prefix}.parquet")

    wall_time_s = time.time() - t0
    stop_flag.set()
    sampler.join(timeout=2)

    return {
        "partition": partition_prefix,
        "rows": count,
        "kept_rows": keep.height,
        "sum_prob": sum_prob,
        "sum_label": sum_label,
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

    total_rows = sum(r["rows"] for r in reports)
    builder_rows = json.loads((FEATURES_DIR / "_manifest.json").read_text(encoding="utf-8"))["rows_written"]
    if total_rows != builder_rows:
        raise RuntimeError(f"scored {total_rows} rows, builder manifest rows_written={builder_rows}")
    total_kept = sum(r["kept_rows"] for r in reports)
    sum_prob = sum(r["sum_prob"] for r in reports)
    sum_label = sum(r["sum_label"] for r in reports)

    wall_time_s = time.time() - t0
    manifest = {
        "total_rows": total_rows,
        "kept_rows": total_kept,
        "mean_raw_probability": sum_prob / total_rows,
        "observed_positive_rate": sum_label / total_rows,
        "positives": sum_label,
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
