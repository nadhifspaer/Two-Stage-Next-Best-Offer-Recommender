# Offer decision layer, scoring date 2020-09-16: a simulation, validated against nothing.
# p_purchase is always the calibrated classifier probability, never the ranker score, which has no probabilistic meaning.
from datetime import date, timedelta

import polars as pl

import src.dataset as dataset

MARGIN_RATE = 0.5
CONTACT_COST_MULTIPLIER = 0.001
RECENT_SALE_WINDOW_DAYS = 7
RECENT_PURCHASE_WINDOW_DAYS = 14
TRAILING_PRICE_WINDOW_WEEKS = 4

SCORING_DATE: date = dataset.VALIDATION_WINDOW_START


def recently_sold_articles(transactions: pl.DataFrame, scoring_date: date) -> pl.Series:
    """Article IDs with at least one sale in the 7 days before scoring_date."""
    window_start = scoring_date - timedelta(days=RECENT_SALE_WINDOW_DAYS)
    return (
        transactions.filter((pl.col("t_dat") >= window_start) & (pl.col("t_dat") < scoring_date))
        .select("article_id")
        .unique()
        .to_series()
    )


def recent_customer_purchases(transactions: pl.DataFrame, scoring_date: date) -> pl.DataFrame:
    """(customer_id, article_id) pairs purchased in the 14 days before scoring_date."""
    window_start = scoring_date - timedelta(days=RECENT_PURCHASE_WINDOW_DAYS)
    return (
        transactions.filter((pl.col("t_dat") >= window_start) & (pl.col("t_dat") < scoring_date))
        .select("customer_id", "article_id")
        .unique()
    )


def trailing_price(transactions: pl.DataFrame, scoring_date: date) -> pl.DataFrame:
    """(article_id, price) trailing 4-week mean price as of scoring_date,
    articles with at least one sale in that window only. No currency unit."""
    window_start = scoring_date - timedelta(weeks=TRAILING_PRICE_WINDOW_WEEKS)
    return (
        transactions.filter((pl.col("t_dat") >= window_start) & (pl.col("t_dat") < scoring_date))
        .group_by("article_id")
        .agg(pl.col("price").mean().alias("price"))
    )


def contact_cost_from_median_price(median_price: float, multiplier: float = CONTACT_COST_MULTIPLIER) -> float:
    return multiplier * median_price


def attach_eligibility(
    scored: pl.DataFrame, recently_sold: pl.Series, recent_purchases: pl.DataFrame
) -> pl.DataFrame:
    """Adds a boolean eligible column: sold in the last 7 days, not
    purchased by this customer in the last 14 days."""
    return (
        scored.with_columns(pl.col("article_id").is_in(recently_sold).alias("sold_recently"))
        .join(
            recent_purchases.with_columns(pl.lit(True).alias("purchased_recently")),
            on=["customer_id", "article_id"],
            how="left",
        )
        .with_columns(pl.col("purchased_recently").fill_null(False))
        .with_columns((pl.col("sold_recently") & ~pl.col("purchased_recently")).alias("eligible"))
        .drop("sold_recently", "purchased_recently")
    )


def compute_expected_value(scored: pl.DataFrame, margin_rate: float, contact_cost: float) -> pl.DataFrame:
    return scored.with_columns(
        (margin_rate * pl.col("price") * pl.col("p_purchase") - contact_cost).alias("expected_value")
    )


def select_offers(scored: pl.DataFrame, margin_rate: float, contact_cost: float) -> pl.DataFrame:
    """scored: one row per (customer_id, article_id) in a customer's ranker top 12, with price, p_purchase, ranker_score and eligible.
    Returns one row per covered customer: the eligible positive-expected_value candidate with the highest expected_value, ties on ranker_score then article_id."""
    candidates = compute_expected_value(scored.filter(pl.col("eligible")), margin_rate, contact_cost)
    positive = candidates.filter(pl.col("expected_value") > 0)
    return (
        positive.sort(
            ["customer_id", "expected_value", "ranker_score", "article_id"],
            descending=[False, True, True, False],
        )
        .with_columns((pl.int_range(pl.len()).over("customer_id") + 1).alias("_offer_rank"))
        .filter(pl.col("_offer_rank") == 1)
        .drop("_offer_rank")
    )


def coverage_summary(selected: pl.DataFrame, total_customers: int) -> dict:
    covered = selected["customer_id"].n_unique()
    return {
        "covered_customers": covered,
        "total_customers": total_customers,
        "coverage": covered / total_customers,
    }


def top_article_concentration(selected: pl.DataFrame, top_n: int = 10) -> dict:
    counts = selected.group_by("article_id").agg(pl.len().alias("n")).sort("n", descending=True)
    total = counts["n"].sum()
    top = counts.head(top_n)["n"].sum()
    return {
        "distinct_selected_articles": counts.height,
        "total_selections": total,
        "share_of_selections_in_top_10": top / total if total else 0.0,
    }


def strategy_mix(selected: pl.DataFrame, candidate_pool: pl.DataFrame, strategy_names: list[str]) -> dict:
    """Share of selected offers each strategy contributed to, by membership in the article's originating strategies (rank_<strategy> not null);
    a selection found by several strategies counts toward each, so shares need not sum to 1."""
    joined = selected.join(
        candidate_pool.select("customer_id", "article_id", *[f"rank_{s}" for s in strategy_names]),
        on=["customer_id", "article_id"],
        how="left",
    )
    n = joined.height
    if n == 0:
        return {s: 0.0 for s in strategy_names}
    return {s: joined[f"rank_{s}"].is_not_null().sum() / n for s in strategy_names}


def ranker_top1_divergence(selected: pl.DataFrame, ranker_top1: pl.DataFrame) -> dict:
    """Share of covered customers whose selected offer differs from the
    ranker's top-ranked (rank_position == 1) article."""
    joined = selected.select("customer_id", "article_id").join(
        ranker_top1.select("customer_id", pl.col("article_id").alias("top1_article_id")),
        on="customer_id",
        how="left",
    )
    differs = (joined["article_id"] != joined["top1_article_id"]).sum()
    return {
        "covered_customers": joined.height,
        "share_differing_from_ranker_top1": differs / joined.height if joined.height else 0.0,
    }


def price_distribution_summary(selected_prices: pl.Series, top1_prices: pl.Series) -> dict:
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

    def _summary(values: pl.Series) -> dict:
        summary = {f"p{int(q * 100)}": values.quantile(q, interpolation="linear") for q in quantiles}
        summary["mean"] = values.mean()
        return summary

    return {"selected_offers": _summary(selected_prices), "ranker_top1": _summary(top1_prices)}
