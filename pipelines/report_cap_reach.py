# Per-strategy cap-reach for repurchase and colour_variant on the validation window, raw counts per customer.
import src.env_config  # noqa: F401 - must stay the first import: sets OPENBLAS_NUM_THREADS before any numeric library loads

import json
import time
from pathlib import Path

import polars as pl

import src.candidates as candidates
import src.dataset as dataset
import src.phase_peaks as phase_peaks

DATA_DIR = Path("data/processed")
MANIFEST_PATH = DATA_DIR / "candidates" / "_cap_reach.json"
UNCAPPED = 10_000


def _tally(raw_counts: pl.DataFrame, cap: int) -> dict:
    # integer reductions only, exact regardless of combination order
    return {
        "customers_with_candidates": raw_counts.height,
        "customers_reaching_cap": int((raw_counts["n"] >= cap).sum()),
        "raw_rows": int(raw_counts["n"].sum()),
    }


def run() -> dict:
    t0 = time.time()
    peaks = phase_peaks.PhasePeaks()

    window_start = next(w for w in dataset.label_windows() if w["split"] == "validation")["window_start"]
    horizon_start = window_start - candidates.REPURCHASE_HORIZON

    prior = (
        pl.scan_parquet(DATA_DIR / "transactions_train.parquet")
        .filter((pl.col("t_dat") >= horizon_start) & (pl.col("t_dat") < window_start))
        .select("customer_id", "article_id", "t_dat")
        .collect()
    )
    articles = pl.read_parquet(DATA_DIR / "articles.parquet").select("article_id", "product_code")
    all_ids = pl.read_parquet(DATA_DIR / "customers.parquet")["customer_id"].unique().sort()
    n_customers = len(all_ids)

    totals = {
        "repurchase": {"cap": candidates.REPURCHASE_CAP, "customers_with_candidates": 0, "customers_reaching_cap": 0, "raw_rows": 0},
        "colour_variant": {"cap": candidates.COLOUR_VARIANT_CAP, "customers_with_candidates": 0, "customers_reaching_cap": 0, "raw_rows": 0},
    }
    chunk_size = candidates.CUSTOMER_CHUNK_SIZE
    for start in range(0, n_customers, chunk_size):
        chunk_ids = all_ids[start : start + chunk_size]
        rep = candidates.repurchase_candidates(prior, window_start, customer_ids=chunk_ids, cap=UNCAPPED)
        cv = candidates.colour_variant_candidates(prior, articles, window_start, customer_ids=chunk_ids, cap=UNCAPPED)
        for name, frame in (("repurchase", rep), ("colour_variant", cv)):
            tally = _tally(frame.group_by("customer_id").agg(pl.len().alias("n")), totals[name]["cap"])
            for key, value in tally.items():
                totals[name][key] += value
        del rep, cv

    peaks.stop()
    result = {"window_start": str(window_start), "total_customers": n_customers, "uncapped_cap_argument": UNCAPPED}
    for name, t in totals.items():
        result[name] = {
            **t,
            "share_reaching_cap": t["customers_reaching_cap"] / n_customers,
            "mean_raw_per_customer_with_candidates": t["raw_rows"] / t["customers_with_candidates"],
        }
    result["wall_time_s"] = round(time.time() - t0, 1)
    result["peak_rss_gb"] = peaks.peak_rss_gb
    MANIFEST_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
