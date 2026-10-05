# MAP@12 tests for src/evaluate.py
import numpy as np
import polars as pl
import pytest

import src.evaluate as evaluate


def _content_checksum(df: pl.DataFrame, sort_key) -> int:
    return int(df.sort(sort_key).select(pl.struct(pl.all()).hash().sum()).item())


def test_ap_at_12_per_customer_is_deterministic_across_two_runs():
    # regression guard: a float sum under engine="streaming" varied run to run; fixed by the default engine and a customer_id sort;
    # generated data sized well above the 50-row floor that reproduced it
    rng = np.random.default_rng(42)
    n_customers = 2_000
    ranks_per_customer = 4
    customer_ids = np.repeat(np.arange(n_customers), ranks_per_customer)
    article_ids = rng.integers(0, 50_000, size=n_customers * ranks_per_customer)
    rank_positions = np.tile(np.arange(1, ranks_per_customer + 1), n_customers)
    predictions = pl.DataFrame(
        {
            "customer_id": [f"c{i}" for i in customer_ids],
            "article_id": [f"a{i}" for i in article_ids],
            "rank_position": rank_positions,
        }
    )
    # every customer has 1-3 actual purchases, some overlapping predictions
    # (hits) and some not (purchases outside the predicted top-k)
    n_actual = n_customers * 2
    actual_customer_ids = rng.integers(0, n_customers, size=n_actual)
    actual_article_ids = rng.integers(0, 60_000, size=n_actual)
    actual = pl.DataFrame(
        {
            "customer_id": [f"c{i}" for i in actual_customer_ids],
            "article_id": [f"a{i}" for i in actual_article_ids],
        }
    ).unique()

    r1 = evaluate.ap_at_12_per_customer(predictions, actual)
    r2 = evaluate.ap_at_12_per_customer(predictions, actual)
    assert _content_checksum(r1, "customer_id") == _content_checksum(r2, "customer_id")

    map1, n1 = evaluate.map_at_12(predictions, actual)
    map2, n2 = evaluate.map_at_12(predictions, actual)
    assert n1 == n2
    assert map1 == map2


def test_map_at_12_worked_example():
    # c1: hits at ranks 1 and 3, miss at 2, m=3, AP = (1/1 + 2/3) / 3; c2: miss at 1, hit at 2, m=1, AP = (1/2) / 1;
    # c3 has no purchase in the window and is excluded from scoring
    predictions = pl.DataFrame(
        {
            "customer_id": ["c1", "c1", "c1", "c2", "c2"],
            "article_id": ["a1", "a4", "a2", "a9", "a5"],
            "rank_position": [1, 2, 3, 1, 2],
        }
    )
    actual = pl.DataFrame(
        {
            "customer_id": ["c1", "c1", "c1", "c2"],
            "article_id": ["a1", "a2", "a3", "a5"],
        }
    )

    map12, n_scored = evaluate.map_at_12(predictions, actual)

    assert n_scored == 2
    expected = ((1 / 1 + 2 / 3) / 3 + (1 / 2) / 1) / 2
    assert abs(map12 - expected) < 1e-9


def test_map_at_12_purchaser_with_no_predictions_scores_zero():
    predictions = pl.DataFrame(
        {"customer_id": ["c1"], "article_id": ["a1"], "rank_position": [1]}
    )
    actual = pl.DataFrame(
        {
            "customer_id": ["c1", "c2"],
            "article_id": ["a1", "a5"],
        }
    )
    map12, n_scored = evaluate.map_at_12(predictions, actual)
    assert n_scored == 2
    # c1: AP = 1.0 (hit at rank 1, m=1); c2: no predictions at all, AP = 0
    assert abs(map12 - 0.5) < 1e-9


def test_map_at_12_rejects_rank_position_beyond_k():
    predictions = pl.DataFrame(
        {"customer_id": ["c1"], "article_id": ["a1"], "rank_position": [13]}
    )
    actual = pl.DataFrame({"customer_id": ["c1"], "article_id": ["a1"]})
    with pytest.raises(ValueError):
        evaluate.map_at_12(predictions, actual)


def test_oracle_map_at_12_worked_example():
    # c1: m=3 true positives, 2 found in candidates -> AP = min(2,12)/min(3,12) = 2/3
    # c2: m=1, 1 found -> AP = 1/1 = 1.0
    m_found = pl.DataFrame({"customer_id": ["c1", "c2"], "m_found": [2, 1]})
    actual = pl.DataFrame(
        {
            "customer_id": ["c1", "c1", "c1", "c2"],
            "article_id": ["a1", "a2", "a3", "a5"],
        }
    )
    map12, n_scored = evaluate.oracle_map_at_12(m_found, actual)
    assert n_scored == 2
    expected = (2 / 3 + 1 / 1) / 2
    assert abs(map12 - expected) < 1e-9


def test_oracle_map_at_12_customer_missing_from_m_found_scores_zero():
    m_found = pl.DataFrame({"customer_id": ["c1"], "m_found": [1]})
    actual = pl.DataFrame({"customer_id": ["c1", "c2"], "article_id": ["a1", "a5"]})
    map12, n_scored = evaluate.oracle_map_at_12(m_found, actual)
    assert n_scored == 2
    # c1: AP = 1/1 = 1.0; c2: absent from m_found -> treated as 0 found -> AP = 0
    assert abs(map12 - 0.5) < 1e-9
