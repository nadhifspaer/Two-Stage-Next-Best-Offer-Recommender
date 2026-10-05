# API tests for api/main.py, synthetic SQLite store, no real data needed
import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

import src.score_pipeline as score_pipeline

ACTIVE_CUSTOMER = "active_customer"
NO_HISTORY_CUSTOMER = "no_history_customer"


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
        rows.append(
            (
                ACTIVE_CUSTOMER,
                rank,
                f"a{rank}",
                float(13 - rank),
                0.01 * rank,
                1 if rank % 2 == 0 else 0,
                json.dumps(["repurchase"]),
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
                json.dumps(["recent_popularity"]),
                5.0,
                0,
                None,
            )
        )
    return rows


@pytest.fixture
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "offer_store.db"
    _build_store_db(db_path, _rows())
    test_store = score_pipeline.load_store(db_path=db_path, verify_production_pin=False)
    monkeypatch.setattr(score_pipeline, "_default_store", test_store)

    import api.main as api_main

    with TestClient(api_main.app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_recommend_active_customer(client):
    response = client.get(f"/recommend/{ACTIVE_CUSTOMER}")
    assert response.status_code == 200
    body = response.json()
    assert len(body["ranked_articles"]) == 12
    assert body["selected_offer"]["article_id"] == "a6"


def test_recommend_states_price_is_in_scaled_units(client):
    for customer_id in (ACTIVE_CUSTOMER, NO_HISTORY_CUSTOMER):
        note = client.get(f"/recommend/{customer_id}").json()["notes"]["price_unit"]
        assert isinstance(note, str) and note.strip()
        assert "scaled" in note.lower()


def test_recommend_states_ranker_score_is_not_a_probability(client):
    for customer_id in (ACTIVE_CUSTOMER, NO_HISTORY_CUSTOMER):
        note = client.get(f"/recommend/{customer_id}").json()["notes"]["ranker_score"]
        assert isinstance(note, str) and note.strip()
        assert "not" in note.lower() and "probab" in note.lower()


def test_recommend_no_history_customer_null_offer(client):
    response = client.get(f"/recommend/{NO_HISTORY_CUSTOMER}")
    assert response.status_code == 200
    body = response.json()
    assert len(body["ranked_articles"]) == 12
    assert body["selected_offer"] is None


def test_recommend_unknown_customer_404(client):
    response = client.get("/recommend/nobody")
    assert response.status_code == 404


def test_recommend_k_respected(client):
    response = client.get(f"/recommend/{ACTIVE_CUSTOMER}", params={"k": 3})
    assert response.status_code == 200
    assert len(response.json()["ranked_articles"]) == 3


def test_recommend_k_out_of_range_rejected(client):
    response = client.get(f"/recommend/{ACTIVE_CUSTOMER}", params={"k": 13})
    assert response.status_code == 422


def test_metrics_counters_increment(client):
    before = client.get("/metrics").text
    client.get(f"/recommend/{ACTIVE_CUSTOMER}")
    after = client.get("/metrics").text

    def _count_for(text, endpoint):
        for line in text.splitlines():
            if line.startswith("api_requests_total") and f'endpoint="{endpoint}"' in line:
                return float(line.rsplit(" ", 1)[1])
        return 0.0

    assert _count_for(after, "/recommend/{customer_id}") > _count_for(before, "/recommend/{customer_id}")


def test_metrics_error_counter_increments_on_404(client):
    before = client.get("/metrics").text
    client.get("/recommend/nobody")
    after = client.get("/metrics").text

    def _count_for(text, endpoint):
        for line in text.splitlines():
            if line.startswith("api_errors_total") and f'endpoint="{endpoint}"' in line:
                return float(line.rsplit(" ", 1)[1])
        return 0.0

    assert _count_for(after, "/recommend/{customer_id}") > _count_for(before, "/recommend/{customer_id}")


def test_error_series_exist_at_zero_before_any_error():
    from prometheus_client import generate_latest

    import api.main as api_main

    text = generate_latest().decode()
    for endpoint in ("/health", "/recommend/{customer_id}"):
        assert f'api_errors_total{{endpoint="{endpoint}"}}' in text
        assert f'api_requests_total{{endpoint="{endpoint}"}}' in text
    assert api_main.ERROR_COUNT is not None
