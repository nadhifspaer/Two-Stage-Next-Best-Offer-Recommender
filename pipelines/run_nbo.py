# Decision layer: eligibility, expected value and selection at the scoring date, plus descriptive reports.
import src.env_config  # noqa: F401 - must stay the first import: sets OPENBLAS_NUM_THREADS before any numeric library loads

import json
import threading
import time
from fractions import Fraction
from pathlib import Path

import polars as pl
import psutil

import src.calibrate as calibrate
import src.candidates as candidates
import src.nbo as nbo
import src.publish as publish

DATA_DIR = Path("data/processed")
SCORED_TOP12_DIR = DATA_DIR / "nbo" / "scored_top12"
CANDIDATES_DIR = DATA_DIR / "candidates" / "validation"
CUSTOMERS_PATH = DATA_DIR / "customers.parquet"
TRANSACTIONS_PATH = DATA_DIR / "transactions_train.parquet"
OFFERS_PATH = DATA_DIR / "nbo" / f"offers_{nbo.SCORING_DATE}.parquet"
MANIFEST_PATH = DATA_DIR / "nbo" / "_manifest.json"

MARGIN_RATE_GRID = [0.3, 0.4, 0.5, 0.6]
CONTACT_COST_MULTIPLIER_GRID = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]


def _peak_rss_sampler(peak: dict, stop_flag: threading.Event, interval_s: float = 0.3) -> None:
    proc = psutil.Process()
    while not stop_flag.is_set():
        rss = proc.memory_info().rss
        if rss > peak["bytes"]:
            peak["bytes"] = rss
        time.sleep(interval_s)


def run() -> dict:
    pl.enable_string_cache()
    peak = {"bytes": 0}
    stop_flag = threading.Event()
    sampler = threading.Thread(target=_peak_rss_sampler, args=(peak, stop_flag), daemon=True)
    sampler.start()
    t0 = time.time()

    customers = pl.read_parquet(CUSTOMERS_PATH).with_columns(pl.col("customer_id").cast(pl.Categorical))
    total_customers = customers.height
    del customers
    publish.assert_top12_exact(SCORED_TOP12_DIR, total_customers)

    # transactions-derived lookups first, then the transaction table is dropped before scored (16.46M rows) loads
    transactions = (
        pl.scan_parquet(TRANSACTIONS_PATH)
        .with_columns(pl.col("customer_id", "article_id").cast(pl.Categorical))
        .collect(engine="streaming")
    )
    price = nbo.trailing_price(transactions, nbo.SCORING_DATE)
    median_price = float(price["price"].median())
    contact_cost = nbo.contact_cost_from_median_price(median_price, nbo.CONTACT_COST_MULTIPLIER)
    recently_sold = nbo.recently_sold_articles(transactions, nbo.SCORING_DATE)
    recent_purchases = nbo.recent_customer_purchases(transactions, nbo.SCORING_DATE)
    del transactions

    scored = (
        pl.scan_parquet(str(SCORED_TOP12_DIR / "part-*.parquet"))
        .drop("bucket")
        .collect(engine="streaming")
    )
    # customer_bucket is a pure hash of customer_id, recomputed because the joined column is null for freshly scored rows
    scored = scored.with_columns(calibrate.customer_bucket(scored["customer_id"]).alias("bucket"))
    scored = scored.join(price, on="article_id", how="left")
    scored = nbo.attach_eligibility(scored, recently_sold, recent_purchases)

    selected = nbo.select_offers(scored, margin_rate=nbo.MARGIN_RATE, contact_cost=contact_cost)

    OFFERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    selected.select(
        "customer_id",
        "article_id",
        "expected_value",
        "ranker_score",
        "rank_position",
        "p_purchase",
        "price",
        "bucket",
    ).write_parquet(OFFERS_PATH)

    coverage = nbo.coverage_summary(selected, total_customers)
    in_sample_covered = selected.filter(calibrate.fit_bucket_mask(selected["bucket"])).height
    coverage["in_sample_covered_customers"] = in_sample_covered
    coverage["in_sample_share_of_covered"] = (
        in_sample_covered / coverage["covered_customers"] if coverage["covered_customers"] else 0.0
    )

    concentration = nbo.top_article_concentration(selected)

    # strategy provenance for selected offers only: one partition at a time, semi-joined down to the selected subset
    selected_keys = selected.select("customer_id", "article_id")
    strategy_cols = [f"rank_{s}" for s in candidates.STRATEGY_NAMES]
    provenance_frames = []
    for path in sorted(CANDIDATES_DIR.glob("part-*.parquet")):
        part = pl.read_parquet(path, columns=["customer_id", "article_id", *strategy_cols])
        matched = part.join(selected_keys, on=["customer_id", "article_id"], how="inner")
        if matched.height > 0:
            provenance_frames.append(matched)
        del part, matched
    candidate_provenance = pl.concat(provenance_frames, how="vertical")
    mix = nbo.strategy_mix(selected, candidate_provenance, candidates.STRATEGY_NAMES)

    ranker_top1 = scored.filter(pl.col("rank_position") == 1).select("customer_id", "article_id")
    divergence = nbo.ranker_top1_divergence(selected, ranker_top1)

    top1_priced = ranker_top1.join(price, on="article_id", how="left")
    price_dist = nbo.price_distribution_summary(selected["price"], top1_priced["price"])

    # sensitivity grid: cells grouped by exact contact_cost/margin_rate ratio (Fraction), selected once per ratio and mapped to every cell sharing it
    # the check below verifies the grouping and mapping, not the argmax invariance itself
    ratio_groups: dict[Fraction, list[tuple[float, float]]] = {}
    for margin_rate in MARGIN_RATE_GRID:
        for multiplier in CONTACT_COST_MULTIPLIER_GRID:
            ratio = Fraction(multiplier).limit_denominator(10**9) / Fraction(margin_rate).limit_denominator(10**9)
            ratio_groups.setdefault(ratio, []).append((margin_rate, multiplier))

    cells = []
    for ratio, members in ratio_groups.items():
        rep_margin_rate, rep_multiplier = members[0]
        cc = nbo.contact_cost_from_median_price(median_price, rep_multiplier)
        sel = nbo.select_offers(scored, margin_rate=rep_margin_rate, contact_cost=cc)
        ratio_coverage = sel["customer_id"].n_unique() / total_customers
        del sel

        for margin_rate, multiplier in members:
            cells.append(
                {
                    "margin_rate": margin_rate,
                    "contact_cost_multiplier": multiplier,
                    "contact_cost": nbo.contact_cost_from_median_price(median_price, multiplier),
                    "ratio": float(ratio),
                    "coverage": ratio_coverage,
                }
            )

    ratio_mismatches = []
    cells_by_ratio: dict[float, list[dict]] = {}
    for cell in cells:
        own_ratio = float(
            Fraction(cell["contact_cost_multiplier"]).limit_denominator(10**9)
            / Fraction(cell["margin_rate"]).limit_denominator(10**9)
        )
        if own_ratio != cell["ratio"]:
            ratio_mismatches.append({"cell": cell, "recomputed_ratio": own_ratio})
        cells_by_ratio.setdefault(cell["ratio"], []).append(cell)
    for ratio, group in cells_by_ratio.items():
        base_coverage = group[0]["coverage"]
        for other in group[1:]:
            if other["coverage"] != base_coverage:
                ratio_mismatches.append({"ratio": ratio, "cell_a": group[0], "cell_b": other})
    if ratio_mismatches:
        raise AssertionError(f"sensitivity grid ratio mapping is inconsistent: {ratio_mismatches}")

    equal_ratio_groups_checked = sum(1 for members in ratio_groups.values() if len(members) >= 2)

    wall_time_s = time.time() - t0
    stop_flag.set()
    sampler.join(timeout=2)

    manifest = {
        "scoring_date": str(nbo.SCORING_DATE),
        "margin_rate": nbo.MARGIN_RATE,
        "contact_cost_multiplier": nbo.CONTACT_COST_MULTIPLIER,
        "median_trailing_4w_price": median_price,
        "contact_cost": contact_cost,
        "coverage": coverage,
        "top_article_concentration": concentration,
        "strategy_mix": mix,
        "ranker_top1_divergence": divergence,
        "price_distribution_vs_ranker_top1": price_dist,
        "sensitivity_grid": cells,
        "sensitivity_equal_ratio_groups_checked": equal_ratio_groups_checked,
        "sensitivity_equal_ratio_assertion": "passed, every cell's own contact_cost/margin_rate ratio recomputed to its group's ratio and every group's mapped coverage was identical across members; the underlying argmax-invariance fact is proven separately on synthetic data by tests/test_nbo.py, not re-derived here",
        "wall_time_s": round(wall_time_s, 1),
        "peak_rss_gb": round(peak["bytes"] / 1e9, 2),
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
