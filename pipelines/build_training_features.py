# Labels, then features, for the four training windows, one subprocess per step.
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

import src.dataset as dataset
import src.features as features
import src.publish as publish
import src.text_embed as text_embed

DATA_DIR = Path("data/processed")
CANDIDATES_ROOT = DATA_DIR / "candidates" / "train"
LABELS_ROOT = DATA_DIR / "features" / "train" / "_labels"
OUTPUT_DIR = DATA_DIR / "features" / "train"
MANIFEST_PATH = OUTPUT_DIR / "_manifest.json"


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


def _lookup_train_window(window_start_iso: str) -> dict:
    return next(
        w
        for w in dataset.label_windows()
        if w["split"] == "train" and str(w["window_start"]) == window_start_iso
    )


def label_partition(window_start_iso: str, partition_path: Path) -> dict:
    """Subprocess worker: labels one candidate partition of one training window."""
    pl.enable_string_cache()
    peak, stop_flag, sampler = _start_sampler()
    t0 = time.time()

    window = _lookup_train_window(window_start_iso)
    transactions = pl.scan_parquet(DATA_DIR / "transactions_train.parquet").with_columns(
        pl.col("customer_id", "article_id").cast(pl.Categorical)
    )
    part_candidates = pl.read_parquet(partition_path)
    labeled = dataset.assemble_training_labels(
        part_candidates, transactions, window["window_start"], window["window_end"]
    )

    out_dir = LABELS_ROOT / window_start_iso
    out_dir.mkdir(parents=True, exist_ok=True)
    labeled.write_parquet(out_dir / Path(partition_path).name)

    wall_time_s = time.time() - t0
    stop_flag.set()
    sampler.join(timeout=2)

    return {
        "window_start": window_start_iso,
        "partition": Path(partition_path).name,
        "rows": labeled.height,
        "wall_time_s": round(wall_time_s, 1),
        "peak_rss_gb": round(peak["bytes"] / 1e9, 2),
    }


def build_features_for_window(window_start_iso: str) -> dict:
    """Subprocess worker: builds features for one window's already-labeled rows."""
    pl.enable_string_cache()
    peak, stop_flag, sampler = _start_sampler()
    t0 = time.time()

    window = _lookup_train_window(window_start_iso)
    window_start = window["window_start"]

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

    labels_dir = LABELS_ROOT / window_start_iso
    labeled = pl.concat([pl.read_parquet(p) for p in sorted(labels_dir.glob("*.parquet"))])

    relevant_customer_ids = labeled["customer_id"].unique()
    relevant_article_ids = labeled["article_id"].unique()
    customers_scoped = customers.filter(pl.col("customer_id").is_in(relevant_customer_ids))
    articles_scoped = articles.filter(pl.col("article_id").is_in(relevant_article_ids))

    customer_block = features.customer_features(transactions, customers_scoped, window_start)
    article_block = features.article_features(transactions, articles_scoped, window_start)
    recent_text_block = features.customer_recent_text_mean(
        transactions, text_vectors, window_start, customer_ids=relevant_customer_ids
    )
    history = (
        transactions.filter(
            (pl.col("t_dat") < window_start) & pl.col("customer_id").is_in(relevant_customer_ids)
        )
        .select("customer_id", "article_id", "t_dat")
        .collect(engine="streaming")
    )

    pair = features.customer_article_pair_features(
        history,
        articles,
        labeled,
        customer_block,
        article_block,
        text_vectors,
        recent_text_block,
        window_start,
    )
    provenance = features.candidate_provenance_features(labeled)

    assembled = (
        pair.lazy()
        .join(provenance.lazy(), on=["customer_id", "article_id"], how="inner")
        .join(customer_block.lazy(), on="customer_id", how="left")
        .join(article_block.lazy(), on="article_id", how="left")
        .join(
            labeled.lazy().select("customer_id", "article_id", "label"),
            on=["customer_id", "article_id"],
            how="inner",
        )
        .with_columns(pl.lit(window_start_iso).alias("window_start"))
        .collect(engine="streaming")
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    assembled.write_parquet(OUTPUT_DIR / f"window-{window_start_iso}.parquet")

    wall_time_s = time.time() - t0
    stop_flag.set()
    sampler.join(timeout=2)

    return {
        "window_start": window_start_iso,
        "rows": assembled.height,
        "positive_rate": round(assembled["label"].mean(), 6),
        "distinct_customers": assembled["customer_id"].n_unique(),
        "feature_column_count": len(assembled.columns),
        "wall_time_s": round(wall_time_s, 1),
        "peak_rss_gb": round(peak["bytes"] / 1e9, 2),
    }


def _run_subprocess(args: list[str], env: dict) -> dict:
    proc = subprocess.run(
        [sys.executable, __file__, *args], cwd=Path.cwd(), env=env, capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(f"subprocess {args} failed:\n{proc.stderr}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def run() -> dict:
    t0 = time.time()
    publish.clear_globbed(OUTPUT_DIR, "window-*.parquet")
    train_windows = [w for w in dataset.label_windows() if w["split"] == "train"]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())

    label_reports = []
    for window in train_windows:
        window_start_iso = str(window["window_start"])
        candidates_dir = CANDIDATES_ROOT / window_start_iso
        publish.clear_globbed(LABELS_ROOT / window_start_iso, "*.parquet")
        for path in sorted(candidates_dir.glob("part-*.parquet")):
            label_reports.append(
                _run_subprocess(["--label-partition", window_start_iso, str(path)], env)
            )

    feature_reports = []
    for window in train_windows:
        window_start_iso = str(window["window_start"])
        feature_reports.append(_run_subprocess(["--build-window", window_start_iso], env))

    # small per-window files, concatenated for exact cross-window stats
    # (a customer can appear in several windows, so distinct counts cannot be summed)
    full = pl.concat(
        [pl.read_parquet(OUTPUT_DIR / f"window-{r['window_start']}.parquet") for r in feature_reports]
    )

    wall_time_s = time.time() - t0
    manifest = {
        "windows": [r["window_start"] for r in feature_reports],
        "row_count": full.height,
        "positive_rate": round(full["label"].mean(), 6),
        "distinct_customers": full["customer_id"].n_unique(),
        "feature_column_count": len(full.columns),
        "wall_time_s": round(wall_time_s, 1),
        "peak_rss_gb": max(
            [r["peak_rss_gb"] for r in label_reports] + [r["peak_rss_gb"] for r in feature_reports]
        ),
        "label_reports": label_reports,
        "feature_reports": feature_reports,
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-partition", nargs=2, metavar=("WINDOW", "PARTITION_PATH"), default=None)
    parser.add_argument("--build-window", type=str, default=None)
    args = parser.parse_args()

    if args.label_partition is not None:
        window_start_iso, partition_path = args.label_partition
        print(json.dumps(label_partition(window_start_iso, Path(partition_path))))
    elif args.build_window is not None:
        print(json.dumps(build_features_for_window(args.build_window)))
    else:
        result = run()
        print(json.dumps(result, indent=2))
