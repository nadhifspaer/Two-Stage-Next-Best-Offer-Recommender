# Builds the SQLite offer store that score_pipeline serves from; the last batch step.
import src.env_config  # noqa: F401 - must stay the first import: sets OPENBLAS_NUM_THREADS before any numeric library loads

import json
import os
import sqlite3
import threading
import time
from pathlib import Path

import polars as pl
import psutil
import pyarrow.parquet as pq

import src.nbo as nbo
import src.phase_peaks as phase_peaks
import src.publish as publish
import src.production_models as production_models

DATA_DIR = Path("data/processed")
SCORED_TOP12_DIR = DATA_DIR / "nbo" / "scored_top12"
CANDIDATES_DIR = DATA_DIR / "candidates" / "validation"
TRANSACTIONS_PATH = DATA_DIR / "transactions_train.parquet"
OFFERS_PATH = DATA_DIR / "nbo" / f"offers_{nbo.SCORING_DATE}.parquet"

OUTPUT_DB_PATH = DATA_DIR / "nbo" / "offer_store.db"
BUILD_DIR = DATA_DIR / "nbo" / "_build_tmp"
MANIFEST_PATH = DATA_DIR / "nbo" / "_offer_store_manifest.json"

INSERT_BATCH_SIZE = 1_000_000


def _peak_rss_sampler(peak: dict, stop_flag: threading.Event, interval_s: float = 0.3) -> None:
    proc = psutil.Process()
    while not stop_flag.is_set():
        rss = proc.memory_info().rss
        if rss > peak["bytes"]:
            peak["bytes"] = rss
        time.sleep(interval_s)


def _lookup_run_ids() -> tuple[str, str]:
    production = production_models.load_production_models()
    return production.ranker_run_id, production.classifier_run_id


def run(
    customer_ids: pl.DataFrame | None = None,
    output_db_path: Path = OUTPUT_DB_PATH,
    build_dir: Path = BUILD_DIR,
    manifest_path: Path = MANIFEST_PATH,
) -> dict:
    # customer_ids (one customer_id column) restricts the build to that subset, same code path; None builds every customer
    BUILD_TMP_DB_PATH = build_dir / "offer_store.db.tmp"
    TOP12_TMP_PATH = build_dir / "top12_with_price_eligibility.parquet"
    STRATEGIES_TMP_DIR = build_dir / "strategies"
    ENRICHED_TMP_PATH = build_dir / "top12_with_strategies.parquet"
    FINAL_TMP_PATH = build_dir / "top12_final.parquet"

    pl.enable_string_cache()
    if customer_ids is None:
        publish.assert_top12_exact(SCORED_TOP12_DIR, publish.customer_count(DATA_DIR / "customers.parquet"))
    else:
        subset_ids = customer_ids.select(pl.col("customer_id").cast(pl.String)).unique()
    phases = phase_peaks.PhasePeaks(track_uss=False)
    start_rss_bytes = psutil.Process().memory_info().rss
    peak = {"bytes": 0}
    stop_flag = threading.Event()
    sampler = threading.Thread(target=_peak_rss_sampler, args=(peak, stop_flag), daemon=True)
    sampler.start()
    t0 = time.time()

    ranker_run_id, classifier_run_id = _lookup_run_ids()

    with phases.phase("load_transactions"):
        transactions = (
            pl.scan_parquet(TRANSACTIONS_PATH)
            .with_columns(pl.col("customer_id", "article_id").cast(pl.Categorical))
            .collect(engine="streaming")
        )
    with phases.phase("transaction_lookups"):
        recently_sold = nbo.recently_sold_articles(transactions, nbo.SCORING_DATE)
        recent_purchases = nbo.recent_customer_purchases(transactions, nbo.SCORING_DATE)
        price = nbo.trailing_price(transactions, nbo.SCORING_DATE)
        del transactions

    with phases.phase("top12_eligibility"):
        top12_parts = []
        for p in sorted(SCORED_TOP12_DIR.glob("part-*.parquet")):
            part = pl.read_parquet(p).drop("bucket")
            if customer_ids is not None:
                part = part.filter(pl.col("customer_id").cast(pl.String).is_in(subset_ids["customer_id"].implode()))
            top12_parts.append(part)
        top12 = pl.concat(top12_parts)
        del top12_parts
        if customer_ids is not None and top12.select(pl.col("customer_id").n_unique()).item() != subset_ids.height:
            raise RuntimeError("customer subset has customers missing from scored_top12")
        top12 = top12.join(price, on="article_id", how="left")
        top12 = nbo.attach_eligibility(top12, recently_sold, recent_purchases)

        TOP12_TMP_PATH.parent.mkdir(parents=True, exist_ok=True)
        top12.write_parquet(TOP12_TMP_PATH)
        total_rows = top12.height
        del top12

    # strategy provenance: one candidate partition at a time, shrinking the unmatched key set;
    # matches go to small per-partition files
    with phases.phase("strategy_provenance"):
        STRATEGIES_TMP_DIR.mkdir(parents=True, exist_ok=True)
        publish.clear_globbed(STRATEGIES_TMP_DIR, "part-*.parquet")
        remaining_keys = pl.read_parquet(TOP12_TMP_PATH, columns=["customer_id", "article_id"]).unique()
        matched_total = 0
        for i, path in enumerate(sorted(CANDIDATES_DIR.glob("part-*.parquet"))):
            if remaining_keys.height == 0:
                break
            part = pl.read_parquet(path, columns=["customer_id", "article_id", "strategies"])
            matched = part.join(remaining_keys, on=["customer_id", "article_id"], how="inner")
            if matched.height > 0:
                matched.write_parquet(STRATEGIES_TMP_DIR / f"part-{i:04d}.parquet")
                matched_total += matched.height
                remaining_keys = remaining_keys.join(
                    matched.select("customer_id", "article_id"), on=["customer_id", "article_id"], how="anti"
                )
            del part, matched

        if remaining_keys.height > 0:
            raise RuntimeError(f"{remaining_keys.height} top-12 rows found no strategy provenance")
        if matched_total != total_rows:
            raise RuntimeError(f"matched {matched_total} strategy rows, expected {total_rows}")

    with phases.phase("enrich_strategies"):
        (
            pl.scan_parquet(TOP12_TMP_PATH)
            .join(pl.scan_parquet(STRATEGIES_TMP_DIR / "part-*.parquet"), on=["customer_id", "article_id"], how="left")
            .collect(engine="streaming")
            .write_parquet(ENRICHED_TMP_PATH)
        )

    # selected-offer flag: left join against the small offers table, never
    # the other way around
    with phases.phase("selected_flag"):
        (
            pl.scan_parquet(ENRICHED_TMP_PATH)
            .join(
                pl.scan_parquet(OFFERS_PATH).select("customer_id", "article_id", "expected_value"),
                on=["customer_id", "article_id"],
                how="left",
            )
            .with_columns(pl.col("expected_value").is_not_null().alias("selected"))
            .collect(engine="streaming")
            .write_parquet(FINAL_TMP_PATH)
        )

    # atomic publish: build at a temporary path, os.replace into place only once verified;
    # a running API process must be restarted to pick up the new store
    with phases.phase("sqlite_insert"):
        BUILD_TMP_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        BUILD_TMP_DB_PATH.unlink(missing_ok=True)
        conn = sqlite3.connect(str(BUILD_TMP_DB_PATH))
        # fresh, rebuildable batch artifact: durability is not needed mid-build
        conn.execute("PRAGMA synchronous = OFF")
        conn.execute("PRAGMA journal_mode = MEMORY")
        # WITHOUT ROWID keeps one customer's rows contiguous, so a lookup is a single index seek
        conn.execute(
            """
            CREATE TABLE served_top12 (
                customer_id TEXT NOT NULL,
                rank INTEGER NOT NULL,
                article_id TEXT NOT NULL,
                ranker_score REAL NOT NULL,
                p_purchase REAL NOT NULL,
                eligible INTEGER NOT NULL,
                strategies TEXT NOT NULL,
                price REAL,
                selected INTEGER NOT NULL,
                expected_value REAL,
                PRIMARY KEY (customer_id, rank)
            ) WITHOUT ROWID
            """
        )
        conn.execute(
            "CREATE TABLE metadata (scoring_date TEXT NOT NULL, ranker_run_id TEXT NOT NULL, classifier_run_id TEXT NOT NULL)"
        )

        insert_sql = (
            "INSERT INTO served_top12 "
            "(customer_id, rank, article_id, ranker_score, p_purchase, eligible, strategies, price, selected, expected_value) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        inserted = 0
        pf = pq.ParquetFile(FINAL_TMP_PATH)
        for batch in pf.iter_batches(batch_size=INSERT_BATCH_SIZE):
            cols = batch.to_pydict()
            rows = zip(
                cols["customer_id"],
                cols["rank_position"],
                cols["article_id"],
                cols["ranker_score"],
                cols["p_purchase"],
                (1 if v else 0 for v in cols["eligible"]),
                (json.dumps(list(v)) for v in cols["strategies"]),
                cols["price"],
                (1 if v else 0 for v in cols["selected"]),
                cols["expected_value"],
            )
            conn.executemany(insert_sql, rows)
            inserted += len(cols["customer_id"])
        pf.close()
        del pf, batch, cols, rows
        conn.commit()

    with phases.phase("sqlite_finalize"):
        row_count = conn.execute("SELECT COUNT(*) FROM served_top12").fetchone()[0]
        if row_count != total_rows:
            raise RuntimeError(f"inserted {row_count} rows, expected {total_rows}")

        conn.execute(
            "INSERT INTO metadata (scoring_date, ranker_run_id, classifier_run_id) VALUES (?, ?, ?)",
            (str(nbo.SCORING_DATE), ranker_run_id, classifier_run_id),
        )
        conn.commit()
        conn.execute("PRAGMA journal_mode = DELETE")
        conn.execute("ANALYZE")
        conn.commit()
        conn.close()

        # publish: single atomic rename, the prior store is untouched until this point
        output_db_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(BUILD_TMP_DB_PATH, output_db_path)

        TOP12_TMP_PATH.unlink()
        ENRICHED_TMP_PATH.unlink()
        FINAL_TMP_PATH.unlink()
        for p in STRATEGIES_TMP_DIR.glob("part-*.parquet"):
            p.unlink()
        STRATEGIES_TMP_DIR.rmdir()
        build_dir.rmdir()

    wall_time_s = time.time() - t0
    stop_flag.set()
    sampler.join(timeout=2)
    phases.stop()

    manifest = {
        "rows": row_count,
        "db_file_size_bytes": output_db_path.stat().st_size,
        "wall_time_s": round(wall_time_s, 1),
        "peak_rss_gb": max(round(peak["bytes"] / 1e9, 2), phases.peak_rss_gb),
        "start_rss_gb": round(start_rss_bytes / 1e9, 2),
        "phases": phases.report(),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
