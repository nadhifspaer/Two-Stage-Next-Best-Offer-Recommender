# scoring interface tests for src/score_pipeline.py
import json
import sqlite3

import pytest

import src.score_pipeline as score_pipeline

ACTIVE_CUSTOMER = "active_customer"
NO_HISTORY_CUSTOMER = "no_history_customer"
UNKNOWN_CUSTOMER = "unknown_customer"


def _build_store_db(db_path, rows, scoring_date="2020-09-16", ranker_run_id="fake-ranker-run", classifier_run_id="fake-classifier-run"):
    conn = sqlite3.connect(str(db_path))
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
    conn.executemany(
        "INSERT INTO served_top12 "
        "(customer_id, rank, article_id, ranker_score, p_purchase, eligible, strategies, price, selected, expected_value) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.execute(
        "INSERT INTO metadata (scoring_date, ranker_run_id, classifier_run_id) VALUES (?, ?, ?)",
        (scoring_date, ranker_run_id, classifier_run_id),
    )
    conn.commit()
    conn.close()


def _rows():
    rows = []
    for rank in range(1, 13):
        # the selected offer sits at rank 6, deliberately not rank 1, so a
        # k < 6 request still exercises the "selected independent of k" path
        rows.append(
            (
                ACTIVE_CUSTOMER,
                rank,
                f"a{rank}",
                float(13 - rank),
                0.01 * rank,
                1 if rank % 2 == 0 else 0,
                json.dumps(["repurchase", "als"] if rank % 3 == 0 else ["repurchase"]),
                10.0,
                1 if rank == 6 else 0,
                0.01 if rank == 6 else None,
            )
        )
    for rank in range(1, 13):
        rows.append(
            (
                NO_HISTORY_CUSTOMER,
                rank,
                f"p{rank}",
                float(13 - rank),
                0.001 * rank,
                0,
                json.dumps(["recent_popularity"] if rank % 2 == 0 else ["segment_popularity"]),
                5.0,
                0,
                None,
            )
        )
    return rows


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "offer_store.db"
    _build_store_db(db_path, _rows())
    return score_pipeline.load_store(db_path=db_path, verify_production_pin=False)


def test_active_customer_returns_12_ranked_articles(store):
    result = score_pipeline.recommend(ACTIVE_CUSTOMER, k=12, store=store)
    assert len(result["ranked_articles"]) == 12
    assert result["ranked_articles"][0]["article_id"] == "a1"
    assert result["selected_offer"]["article_id"] == "a6"


def test_no_history_customer_returns_12_articles_and_null_offer(store):
    result = score_pipeline.recommend(NO_HISTORY_CUSTOMER, k=12, store=store)
    assert len(result["ranked_articles"]) == 12
    assert result["selected_offer"] is None


def test_unknown_customer_returns_none(store):
    assert score_pipeline.recommend(UNKNOWN_CUSTOMER, k=12, store=store) is None


def test_k_is_respected(store):
    result = score_pipeline.recommend(ACTIVE_CUSTOMER, k=5, store=store)
    assert len(result["ranked_articles"]) == 5
    assert result["k"] == 5


def test_selected_offer_independent_of_k(store):
    # selected offer is at rank 6; k=3 must still surface it
    result = score_pipeline.recommend(ACTIVE_CUSTOMER, k=3, store=store)
    assert len(result["ranked_articles"]) == 3
    assert result["selected_offer"]["article_id"] == "a6"


def test_k_out_of_range_raises(store):
    with pytest.raises(ValueError):
        score_pipeline.recommend(ACTIVE_CUSTOMER, k=13, store=store)
    with pytest.raises(ValueError):
        score_pipeline.recommend(ACTIVE_CUSTOMER, k=0, store=store)


def test_ranker_score_never_labelled_a_probability(store):
    result = score_pipeline.recommend(ACTIVE_CUSTOMER, k=12, store=store)
    assert "not a purchase probability" in result["notes"]["ranker_score"]


def test_price_note_present(store):
    result = score_pipeline.recommend(ACTIVE_CUSTOMER, k=12, store=store)
    assert "no currency" in result["notes"]["price_unit"]


def test_response_carries_no_purchase_ground_truth(store):
    result = score_pipeline.recommend(ACTIVE_CUSTOMER, k=12, store=store)
    flat_keys = set(result.keys()) | {k for row in result["ranked_articles"] for k in row.keys()}
    forbidden = {"label", "purchased", "actual_purchase", "ground_truth", "target"}
    assert flat_keys.isdisjoint(forbidden)
