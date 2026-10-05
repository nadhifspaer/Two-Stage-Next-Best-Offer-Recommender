# local-mode lookup failure paths: unknown customer, API down, timeout, HTTP error, success
import sqlite3
import sys
from pathlib import Path

import pytest
import requests
from streamlit.testing.v1 import AppTest

import dashboard.local_mode as local_mode
import src.score_pipeline as score_pipeline

APP_PATH = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")
CUSTOMER = "a" * 64

ASSUMPTIONS = {
    "margin_rate": 0.5,
    "contact_cost_multiplier": 0.001,
    "median_trailing_4w_price": 0.022,
    "contact_cost": 0.000022,
}
SUMMARY = {
    "scoring_date": "2020-09-16",
    "assumptions": ASSUMPTIONS,
    "coverage": {"covered_customers": 10, "total_customers": 20, "coverage": 0.5},
    "top_article_concentration": {"distinct_selected_articles": 3, "share_of_selections_in_top_10": 1.0},
    "strategy_mix": {"als": 0.5, "repurchase": 0.25},
    "top_selected_articles": [{"article_id": "a1", "selections": 5, "price": 0.03}],
}
RECOMMENDATION = {
    "customer_id": CUSTOMER,
    "scoring_date": "2020-09-16",
    "k": 12,
    "ranked_articles": [
        {"article_id": f"a{r}", "rank_position": r, "ranker_score": 1.0, "p_purchase": 0.01, "eligible": True, "strategies": ["als"]}
        for r in range(1, 13)
    ],
    "selected_offer": {"article_id": "a2", "expected_value": 0.0005, "price": 0.03, "p_purchase": 0.01},
}


def _response(status: int, payload: dict | None = None) -> requests.Response:
    r = requests.Response()
    r.status_code = status
    r.url = "http://api.test/recommend/x"
    r._content = (b"{}" if payload is None else __import__("json").dumps(payload).encode())
    return r


@pytest.fixture
def local_app(tmp_path, monkeypatch):
    db = tmp_path / "store.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE served_top12 (customer_id TEXT NOT NULL, rank INTEGER NOT NULL, article_id TEXT NOT NULL, "
        "ranker_score REAL NOT NULL, p_purchase REAL NOT NULL, eligible INTEGER NOT NULL, strategies TEXT NOT NULL, "
        "price REAL, selected INTEGER NOT NULL, expected_value REAL, PRIMARY KEY (customer_id, rank)) WITHOUT ROWID"
    )
    conn.execute("CREATE TABLE metadata (scoring_date TEXT NOT NULL, ranker_run_id TEXT NOT NULL, classifier_run_id TEXT NOT NULL)")
    conn.execute("INSERT INTO metadata VALUES ('2020-09-16', 'ranker-run', 'classifier-run')")
    conn.execute("INSERT INTO served_top12 VALUES (?, 1, 'a1', 1.0, 0.01, 1, '[\"als\"]', 0.03, 0, NULL)", (CUSTOMER,))
    conn.commit()
    conn.close()
    store = score_pipeline.load_store(db, verify_production_pin=False)
    monkeypatch.setenv("APP_MODE", "local")
    monkeypatch.setitem(sys.modules, "dashboard.local_mode", local_mode)
    monkeypatch.setattr(local_mode, "_load_store", lambda: store)
    monkeypatch.setattr(local_mode, "_load_summary", lambda: SUMMARY)
    yield
    score_pipeline.close_store(store)


def _run(monkeypatch, get) -> AppTest:
    monkeypatch.setattr(local_mode.requests, "get", get)
    return AppTest.from_file(APP_PATH, default_timeout=60).run()


def test_success_renders_selected_offer_and_ranked_table(local_app, monkeypatch):
    at = _run(monkeypatch, lambda *a, **k: _response(200, RECOMMENDATION))
    assert not at.exception and len(at.error) == 0
    assert len(at.dataframe[0].value) == 12
    assert any(m.label == "Expected value (scaled units)" for m in at.metric)


def test_unknown_customer_404_shows_warning_not_error(local_app, monkeypatch):
    at = _run(monkeypatch, lambda *a, **k: _response(404))
    assert not at.exception
    assert len(at.warning) == 1 and "not found" in at.warning[0].value
    assert len(at.error) == 0


@pytest.mark.parametrize(
    "failure",
    [
        requests.ConnectionError("connection refused"),
        requests.Timeout("read timed out"),
    ],
)
def test_api_unreachable_shows_error_and_summary_tab_still_renders(local_app, monkeypatch, failure):
    def get(*a, **k):
        raise failure

    at = _run(monkeypatch, get)
    assert not at.exception
    assert len(at.error) == 1 and "did not answer" in at.error[0].value and "reload" in at.error[0].value
    assert any(m.label == "Offer coverage" for m in at.metric)


def test_http_500_shows_error_not_exception(local_app, monkeypatch):
    at = _run(monkeypatch, lambda *a, **k: _response(500))
    assert not at.exception
    assert len(at.error) == 1 and "did not answer" in at.error[0].value


def test_typed_unknown_id_after_success_shows_warning(local_app, monkeypatch):
    def get(url, **k):
        return _response(200, RECOMMENDATION) if CUSTOMER in url else _response(404)

    at = _run(monkeypatch, get)
    at.text_input[0].set_value("no-such-customer").run()
    assert len(at.warning) == 1 and not at.exception


def test_fetch_passes_timeout_and_k(monkeypatch):
    seen = {}

    def get(url, params, timeout):
        seen.update(url=url, params=params, timeout=timeout)
        return _response(200, RECOMMENDATION)

    monkeypatch.setattr(local_mode.requests, "get", get)
    assert local_mode._fetch("abc", 5) == RECOMMENDATION
    assert seen["params"] == {"k": 5} and seen["timeout"] == local_mode.REQUEST_TIMEOUT_S and seen["url"].endswith("/recommend/abc")
