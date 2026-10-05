# offer decision layer tests for src/nbo.py
from datetime import date

import polars as pl

import src.nbo as nbo


def test_eligibility_requires_recent_sale_and_excludes_recent_purchase():
    scored = pl.DataFrame({"customer_id": ["c1", "c1", "c1"], "article_id": ["a1", "a2", "a3"]})
    recently_sold = pl.Series(["a1", "a2"])  # a3 did not sell recently
    recent_purchases = pl.DataFrame({"customer_id": ["c1"], "article_id": ["a2"]})  # a2 bought recently

    result = nbo.attach_eligibility(scored, recently_sold, recent_purchases)

    eligible = dict(zip(result["article_id"], result["eligible"]))
    assert eligible == {"a1": True, "a2": False, "a3": False}


def test_expected_value_formula():
    scored = pl.DataFrame({"price": [10.0], "p_purchase": [0.1]})
    result = nbo.compute_expected_value(scored, margin_rate=0.5, contact_cost=0.2)
    assert result["expected_value"][0] == 0.5 * 10.0 * 0.1 - 0.2


def test_select_offers_picks_highest_expected_value_and_excludes_nonpositive():
    scored = pl.DataFrame(
        {
            "customer_id": ["c1", "c1", "c1", "c2"],
            "article_id": ["a1", "a2", "a3", "a1"],
            "ranker_score": [1.0, 2.0, 3.0, 1.0],
            "price": [10.0, 10.0, 10.0, 10.0],
            "p_purchase": [0.05, 0.09, 0.001, 0.001],
            "eligible": [True, True, True, True],
        }
    )
    # margin_rate 0.5, contact_cost 0.2: c1/a1 ev=0.05, c1/a2 ev=0.25, c1/a3 ev=-0.185, c2/a1 ev=-0.175
    selected = nbo.select_offers(scored, margin_rate=0.5, contact_cost=0.2)

    assert selected["customer_id"].to_list() == ["c1"]
    assert selected["article_id"].to_list() == ["a2"]


def test_select_offers_tiebreak_ranker_score_then_article_id():
    scored = pl.DataFrame(
        {
            "customer_id": ["c1", "c1"],
            "article_id": ["a2", "a1"],
            "ranker_score": [1.0, 2.0],
            "price": [10.0, 10.0],
            "p_purchase": [0.1, 0.1],  # identical expected_value for both
            "eligible": [True, True],
        }
    )
    selected = nbo.select_offers(scored, margin_rate=0.5, contact_cost=0.2)
    assert selected["article_id"].to_list() == ["a1"]  # higher ranker_score wins


def test_select_offers_tiebreak_falls_to_article_id():
    scored = pl.DataFrame(
        {
            "customer_id": ["c1", "c1"],
            "article_id": ["a2", "a1"],
            "ranker_score": [1.0, 1.0],  # tied ranker_score too
            "price": [10.0, 10.0],
            "p_purchase": [0.1, 0.1],
            "eligible": [True, True],
        }
    )
    selected = nbo.select_offers(scored, margin_rate=0.5, contact_cost=0.2)
    assert selected["article_id"].to_list() == ["a1"]  # lower article_id wins


def _synthetic_scored(seed: int = 0, n_customers: int = 40) -> pl.DataFrame:
    import numpy as np

    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_customers):
        customer_id = f"c{i}"
        for j in range(12):
            rows.append(
                {
                    "customer_id": customer_id,
                    "article_id": f"a{i}_{j}",
                    "ranker_score": float(rng.uniform(0, 1)),
                    "price": float(rng.uniform(1, 50)),
                    "p_purchase": float(rng.uniform(0, 0.05)),
                    "eligible": bool(rng.random() > 0.2),
                }
            )
    return pl.DataFrame(rows)


def test_sensitivity_equal_ratio_produces_identical_coverage_and_selections():
    scored = _synthetic_scored()
    median_price = 20.0

    ratio_cells = {
        (0.5, 0.001): 0.001 * median_price / 0.5,
        (1.0, 0.002): 0.002 * median_price / 1.0,  # same ratio as (0.5, 0.001)
        (0.5, 0.02): 0.02 * median_price / 0.5,  # 20x higher ratio, must select fewer
    }
    assert ratio_cells[(0.5, 0.001)] == ratio_cells[(1.0, 0.002)]

    selections = {}
    for (margin_rate, multiplier), _ratio in ratio_cells.items():
        contact_cost = nbo.contact_cost_from_median_price(median_price, multiplier)
        selected = nbo.select_offers(scored, margin_rate=margin_rate, contact_cost=contact_cost)
        key = (margin_rate, multiplier)
        selections[key] = set(zip(selected["customer_id"].to_list(), selected["article_id"].to_list()))

    assert selections[(0.5, 0.001)] == selections[(1.0, 0.002)]
    assert len(selections[(0.5, 0.02)]) < len(selections[(0.5, 0.001)])


def test_scoring_date_is_validation_window_start():
    assert nbo.SCORING_DATE == date(2020, 9, 16)
