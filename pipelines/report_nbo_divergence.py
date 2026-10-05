# Splits covered customers whose selected offer differs from the ranker top-1 into three exclusive causes.
import src.env_config  # noqa: F401 - must stay the first import: sets OPENBLAS_NUM_THREADS before any numeric library loads

import json
from pathlib import Path

import polars as pl

import src.nbo as nbo

DATA_DIR = Path("data/processed")
SCORED_TOP12_DIR = DATA_DIR / "nbo" / "scored_top12"
OFFERS_PATH = DATA_DIR / "nbo" / f"offers_{nbo.SCORING_DATE}.parquet"
TRANSACTIONS_PATH = DATA_DIR / "transactions_train.parquet"
OUTPUT_PATH = DATA_DIR / "nbo" / "_divergence_manifest.json"


def run() -> dict:
    pl.enable_string_cache()

    selected = pl.read_parquet(OFFERS_PATH)

    transactions = (
        pl.scan_parquet(TRANSACTIONS_PATH)
        .with_columns(pl.col("customer_id", "article_id").cast(pl.Categorical))
        .collect(engine="streaming")
    )
    recently_sold = nbo.recently_sold_articles(transactions, nbo.SCORING_DATE)
    recent_purchases = nbo.recent_customer_purchases(transactions, nbo.SCORING_DATE)
    price = nbo.trailing_price(transactions, nbo.SCORING_DATE)
    del transactions

    scored = pl.concat(
        [pl.read_parquet(p) for p in sorted(SCORED_TOP12_DIR.glob("part-*.parquet"))]
    ).drop("bucket")

    top1 = (
        scored.filter(pl.col("rank_position") == 1)
        .select("customer_id", "article_id", "p_purchase")
        .with_columns(pl.col("article_id").is_in(recently_sold).alias("top1_sold_recently"))
        .join(
            recent_purchases.with_columns(pl.lit(True).alias("top1_purchased_recently")),
            on=["customer_id", "article_id"],
            how="left",
        )
        .with_columns(pl.col("top1_purchased_recently").fill_null(False))
    )
    del scored

    top1 = (
        top1.join(price, on="article_id", how="left")
        .with_columns((pl.col("top1_sold_recently") & ~pl.col("top1_purchased_recently")).alias("top1_eligible"))
        .with_columns((pl.col("price") * pl.col("p_purchase")).alias("top1_price_x_p"))
        .rename({"article_id": "top1_article_id", "price": "top1_price"})
        .select(
            "customer_id",
            "top1_article_id",
            "top1_sold_recently",
            "top1_purchased_recently",
            "top1_eligible",
            "top1_price",
            "top1_price_x_p",
        )
    )

    joined = (
        selected.select("customer_id", "article_id", "price", "p_purchase")
        .with_columns((pl.col("price") * pl.col("p_purchase")).alias("selected_price_x_p"))
        .join(top1, on="customer_id", how="left")
    )

    covered_customers = joined.height
    differs = joined.filter(pl.col("article_id") != pl.col("top1_article_id"))

    case_a = differs.filter(pl.col("top1_purchased_recently"))
    case_b = differs.filter(~pl.col("top1_purchased_recently") & ~pl.col("top1_sold_recently"))
    case_c = differs.filter(pl.col("top1_eligible"))

    if case_a.height + case_b.height + case_c.height != differs.height:
        raise AssertionError(
            f"decomposition does not partition the diverging set: "
            f"a={case_a.height} b={case_b.height} c={case_c.height} total={differs.height}"
        )

    both_reasons_ineligible = differs.filter(
        pl.col("top1_purchased_recently") & ~pl.col("top1_sold_recently")
    ).height

    case_c_ratio = case_c.with_columns((pl.col("price") / pl.col("top1_price")).alias("price_ratio"))
    median_price_ratio_case_c = float(case_c_ratio["price_ratio"].median())

    manifest = {
        "covered_customers": covered_customers,
        "differs_from_top1": differs.height,
        "differs_share_of_covered": differs.height / covered_customers,
        "cases": {
            "a_top1_purchased_recently": {
                "count": case_a.height,
                "share_of_differs": case_a.height / differs.height,
            },
            "b_top1_not_sold_recently": {
                "count": case_b.height,
                "share_of_differs": case_b.height / differs.height,
            },
            "c_top1_eligible_but_beaten_on_price_x_p": {
                "count": case_c.height,
                "share_of_differs": case_c.height / differs.height,
                "median_selected_to_top1_price_ratio": median_price_ratio_case_c,
            },
        },
        "both_ineligibility_reasons_note": (
            f"{both_reasons_ineligible} of the diverging customers had a top-1 that was both "
            "purchased recently and not sold recently; counted under (a)"
        ),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
