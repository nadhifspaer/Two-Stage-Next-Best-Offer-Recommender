# Cap allocation sweep: only the five per-strategy caps vary, B-E go to scratch directories and are discarded.
import src.env_config  # noqa: F401 - must stay the first import: sets OPENBLAS_NUM_THREADS before any numeric library loads

import os
import shutil
from pathlib import Path

import polars as pl

import pipelines.evaluate_candidates as evaluate_candidates
import src.candidates as candidates
import src.dataset as dataset

DATA_DIR = Path("data/processed")
PRODUCTION_CANDIDATES_DIR = DATA_DIR / "candidates" / "validation"
SWEEP_DIR = DATA_DIR / "candidates_sweep"
REPORT_PATH = Path("logs/cap_allocation_sweep.md")

STRATEGIES = ["repurchase", "colour_variant", "recent_popularity", "segment_popularity", "als"]

# repurchase, colour_variant, recent_popularity, segment_popularity, als
ALLOCATIONS = {
    "A": (30, 30, 20, 20, 30),  # current baseline
    "B": (40, 30, 10, 10, 40),
    "C": (40, 20, 10, 10, 50),
    "D": (50, 30, 10, 10, 30),
    "E": (30, 30, 20, 0, 40),  # segment_popularity dropped entirely
}


def _generate(label: str, caps: tuple[int, int, int, int, int]) -> Path:
    repurchase_cap, colour_variant_cap, recent_popularity_cap, segment_popularity_cap, als_cap = caps

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

    output_dir = SWEEP_DIR / label
    if output_dir.exists():
        shutil.rmtree(output_dir)

    candidates.generate_candidates_chunked(
        transactions,
        articles,
        customers,
        window_start,
        output_dir,
        repurchase_cap=repurchase_cap,
        colour_variant_cap=colour_variant_cap,
        recent_popularity_cap=recent_popularity_cap,
        segment_popularity_cap=segment_popularity_cap,
        als_cap=als_cap,
    )
    return output_dir


def run() -> dict:
    pl.enable_string_cache()

    results = {}
    for label, caps in ALLOCATIONS.items():
        if label == "A":
            candidates_dir = PRODUCTION_CANDIDATES_DIR
        else:
            print(f"generating allocation {label} {caps} ...", flush=True)
            candidates_dir = _generate(label, caps)

        print(f"evaluating allocation {label} ...", flush=True)
        eval_result = evaluate_candidates.run(candidates_dir=candidates_dir)
        results[label] = {"caps": caps, "eval": eval_result}

        if label != "A":
            shutil.rmtree(candidates_dir)

    if SWEEP_DIR.exists() and not any(SWEEP_DIR.iterdir()):
        SWEEP_DIR.rmdir()

    return results


def _print_report(results: dict) -> None:
    for label, row in results.items():
        r = row["caps"]
        e = row["eval"]
        print(
            f"{label}  caps={r}  pool_recall@100={e['pool_recall_at_100']:.4f}"
            f"  wall_time_s={e['wall_time_s']}  peak_rss_gb={e['peak_rss_gb']}"
        )
        for s in row["eval"]["per_strategy"]:
            print(f"    {s['strategy']:<20} marginal={s['marginal_contribution']:+.4f}")


def _write_report(results: dict) -> None:
    header = (
        "| Allocation | repurchase | colour_variant | recent_popularity | "
        "segment_popularity | als | Pool Recall@100 |"
    )
    sep = "|---|---|---|---|---|---|---|"
    rows = [header, sep]
    for label, row in results.items():
        r = row["caps"]
        e = row["eval"]
        rows.append(
            f"| {label} | {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {e['pool_recall_at_100']:.4f} |"
        )
    caps_table = "\n".join(rows)

    marginal_header = "| Allocation | " + " | ".join(STRATEGIES) + " |"
    marginal_sep = "|---|" + "---|" * len(STRATEGIES)
    marginal_rows = [marginal_header, marginal_sep]
    for label, row in results.items():
        by_strategy = {s["strategy"]: s["marginal_contribution"] for s in row["eval"]["per_strategy"]}
        marginal_rows.append(
            "| "
            + label
            + " | "
            + " | ".join(f"{by_strategy[s]:+.4f}" for s in STRATEGIES)
            + " |"
        )
    marginal_table = "\n".join(marginal_rows)

    section = f"""### Cap allocation sweep

Pool cap held at 100; `CUSTOMER_CHUNK_SIZE`, `ALS_RANDOM_STATE`, the 52-week repurchase/colour-variant horizon, and the deterministic tiebreaking are all held fixed. Only the five per-strategy caps vary. Allocation A is the deployed baseline; B through E were generated into scratch directories, evaluated, and discarded, they are not the deployed candidate set.

Pooled Recall@100 per allocation:

{caps_table}

Marginal contribution per strategy per allocation:

{marginal_table}

No allocation is selected here.
"""

    # new file, never the report: the section is pasted into the report by hand
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = REPORT_PATH.with_suffix(".md.tmp")
    tmp.write_text(section, encoding="utf-8")
    os.replace(tmp, REPORT_PATH)


if __name__ == "__main__":
    results = run()
    _print_report(results)
    _write_report(results)
