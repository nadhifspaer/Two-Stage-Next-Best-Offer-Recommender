# Cloud sample: build_offer_store.run() on a customer subset, same code path as the full store.
import src.env_config  # noqa: F401 - must stay the first import: sets OPENBLAS_NUM_THREADS before any numeric library loads

import hashlib
import json
import os
import shutil
import sqlite3
import time
from pathlib import Path

import polars as pl

import pipelines.build_offer_store as build_offer_store
import src.evaluate as evaluate
import src.nbo as nbo
import src.production_models as production_models
import src.score_pipeline as score_pipeline

DATA_DIR = Path("data/processed")
SAMPLE_DIR = Path("dashboard/sample")
SAMPLE_DB_PATH = SAMPLE_DIR / "offer_store_sample.db"
SAMPLE_MANIFEST_PATH = SAMPLE_DIR / "offer_store_sample_manifest.json"
STAGING_DIR = DATA_DIR / "nbo" / "_sample_staging"
STAGING_BUILD_DIR = DATA_DIR / "nbo" / "_sample_build_tmp"
NBO_MANIFEST_PATH = DATA_DIR / "nbo" / "_manifest.json"

CUSTOMERS_PER_BUCKET = 300
SELECTION_SEED = "nbo-sample-2020-09-16"
MAX_SAMPLE_BYTES = 100_000_000
REGENERATE_COMMAND = "python -m pipelines.run_guarded_stage build_offer_store_sample"


def select_customers(buckets: pl.DataFrame, per_bucket: int = CUSTOMERS_PER_BUCKET, seed: str = SELECTION_SEED) -> pl.DataFrame:
    # per recency bucket, the per_bucket customers with the smallest seeded sha256 of customer_id
    ids = buckets.with_columns(pl.col("customer_id").cast(pl.String)).sort("customer_id")
    keys = [hashlib.sha256(f"{seed}:{c}".encode()).hexdigest() for c in ids["customer_id"].to_list()]
    ranked = ids.with_columns(pl.Series("sort_key", keys)).with_columns(
        pl.col("sort_key").rank("ordinal").over("recency_bucket").alias("rank_in_bucket")
    )
    return (
        ranked.filter(pl.col("rank_in_bucket") <= per_bucket)
        .select("customer_id", "recency_bucket")
        .sort("recency_bucket", "customer_id")
    )


def _add_sample_customers(db_path: Path, selected: pl.DataFrame) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE sample_customers (customer_id TEXT NOT NULL PRIMARY KEY, recency_bucket TEXT NOT NULL)")
    conn.executemany("INSERT INTO sample_customers VALUES (?, ?)", selected.rows())
    # pin recorded at build time, checked by cloud mode through score_pipeline.load_store(verify_embedded_pin=True)
    pinned = production_models.load_production_models()
    conn.execute("CREATE TABLE pinned_models (ranker_run_id TEXT NOT NULL, classifier_run_id TEXT NOT NULL)")
    conn.execute("INSERT INTO pinned_models VALUES (?, ?)", (pinned.ranker_run_id, pinned.classifier_run_id))
    conn.commit()
    conn.execute("VACUUM")
    conn.close()


def _verify_against_full_store(sample_db_path: Path, selected: pl.DataFrame) -> int:
    # every sampled customer's served response must equal the full store's response
    full = score_pipeline.load_store()
    sample = score_pipeline.load_store(sample_db_path, verify_production_pin=False, verify_embedded_pin=True)
    checked = 0
    for customer_id in selected["customer_id"].to_list():
        if score_pipeline.recommend(customer_id, store=sample) != score_pipeline.recommend(customer_id, store=full):
            raise RuntimeError(f"sample response differs from the full store for customer {customer_id}")
        checked += 1
    score_pipeline.close_store(sample)
    score_pipeline.close_store(full)
    return checked


def _atomic_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dest)


def run() -> dict:
    t0 = time.time()
    buckets = evaluate.customer_recency_buckets(
        pl.scan_parquet(DATA_DIR / "transactions_train.parquet").select("customer_id", "t_dat"),
        pl.scan_parquet(DATA_DIR / "customers.parquet").select("customer_id"),
        nbo.SCORING_DATE,
    )
    selected = select_customers(buckets)

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    staged_db = STAGING_DIR / "offer_store_sample.db"
    store_report = build_offer_store.run(
        customer_ids=selected,
        output_db_path=staged_db,
        build_dir=STAGING_BUILD_DIR,
        manifest_path=STAGING_DIR / "_build_manifest.json",
    )
    _add_sample_customers(staged_db, selected)
    verified = _verify_against_full_store(staged_db, selected)

    size = staged_db.stat().st_size
    if size >= MAX_SAMPLE_BYTES:
        raise RuntimeError(f"sample is {size} bytes, limit {MAX_SAMPLE_BYTES}")

    nbo_manifest = json.loads(NBO_MANIFEST_PATH.read_text(encoding="utf-8"))
    conn = sqlite3.connect(f"file:{staged_db.resolve().as_posix()}?mode=ro", uri=True)
    rows = conn.execute("SELECT COUNT(*) FROM served_top12").fetchone()[0]
    selected_offers = conn.execute("SELECT COUNT(DISTINCT customer_id) FROM served_top12 WHERE selected = 1").fetchone()[0]
    scoring_date, ranker_run_id, classifier_run_id = conn.execute(
        "SELECT scoring_date, ranker_run_id, classifier_run_id FROM metadata"
    ).fetchone()
    conn.close()

    manifest = {
        "scoring_date": scoring_date,
        "ranker_run_id": ranker_run_id,
        "classifier_run_id": classifier_run_id,
        "rows": rows,
        "customers": selected.height,
        "customers_with_selected_offer": selected_offers,
        "customers_per_bucket": {r[0]: r[1] for r in selected.group_by("recency_bucket").len().sort("recency_bucket").rows()},
        "file_size_bytes": size,
        "file_size_limit_bytes": MAX_SAMPLE_BYTES,
        "verified_equal_to_full_store_customers": verified,
        "selection": {"per_bucket": CUSTOMERS_PER_BUCKET, "seed": SELECTION_SEED, "anchor_date": str(nbo.SCORING_DATE)},
        "assumptions": {
            "margin_rate": nbo_manifest["margin_rate"],
            "contact_cost_multiplier": nbo_manifest["contact_cost_multiplier"],
            "median_trailing_4w_price": nbo_manifest["median_trailing_4w_price"],
            "contact_cost": nbo_manifest["contact_cost"],
        },
        "regenerate_command": REGENERATE_COMMAND,
        "full_store_rows": json.loads(build_offer_store.MANIFEST_PATH.read_text(encoding="utf-8"))["rows"],
    }

    _atomic_copy(staged_db, SAMPLE_DB_PATH)
    manifest_tmp = SAMPLE_MANIFEST_PATH.with_name(SAMPLE_MANIFEST_PATH.name + ".tmp")
    manifest_tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(manifest_tmp, SAMPLE_MANIFEST_PATH)

    staged_db.unlink()
    (STAGING_DIR / "_build_manifest.json").unlink()
    STAGING_DIR.rmdir()

    manifest["wall_time_s"] = round(time.time() - t0, 1)
    manifest["peak_rss_gb"] = store_report["peak_rss_gb"]
    return manifest


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
