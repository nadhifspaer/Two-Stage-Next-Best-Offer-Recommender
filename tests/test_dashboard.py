# Dashboard tests: sample selection, helper queries, cloud isolation, app render
import random
import sqlite3
import sys
from pathlib import Path

import polars as pl
import pytest
from streamlit.testing.v1 import AppTest

import pipelines.build_offer_store_sample as sample_builder
import src.score_pipeline as score_pipeline

APP_PATH = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")
CLOUD_SOURCE = (Path(__file__).resolve().parent.parent / "dashboard" / "cloud_mode.py").read_text(encoding="utf-8")


def _buckets() -> pl.DataFrame:
    rows = [(f"c{i:03d}", "last 1 week") for i in range(50)] + [(f"n{i:03d}", "never") for i in range(5)]
    return pl.DataFrame(rows, schema=["customer_id", "recency_bucket"], orient="row")


def test_select_customers_caps_each_bucket_and_keeps_small_buckets_whole():
    selected = sample_builder.select_customers(_buckets(), per_bucket=10, seed="s")
    counts = dict(selected.group_by("recency_bucket").len().rows())
    assert counts == {"last 1 week": 10, "never": 5}


def test_select_customers_is_deterministic_and_seed_dependent():
    a = sample_builder.select_customers(_buckets(), per_bucket=10, seed="s")
    b = sample_builder.select_customers(_buckets().sample(fraction=1.0, shuffle=True, seed=3), per_bucket=10, seed="s")
    c = sample_builder.select_customers(_buckets(), per_bucket=10, seed="other")
    assert a.equals(b)
    assert not a.equals(c)


def _build_db(path: Path, customers: list[str]) -> None:
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE served_top12 (customer_id TEXT NOT NULL, rank INTEGER NOT NULL, article_id TEXT NOT NULL, "
        "ranker_score REAL NOT NULL, p_purchase REAL NOT NULL, eligible INTEGER NOT NULL, strategies TEXT NOT NULL, "
        "price REAL, selected INTEGER NOT NULL, expected_value REAL, PRIMARY KEY (customer_id, rank)) WITHOUT ROWID"
    )
    conn.execute("CREATE TABLE metadata (scoring_date TEXT NOT NULL, ranker_run_id TEXT NOT NULL, classifier_run_id TEXT NOT NULL)")
    conn.execute("INSERT INTO metadata VALUES ('2020-09-16', 'r', 'c')")
    for customer in customers:
        for rank in range(1, 13):
            conn.execute(
                "INSERT INTO served_top12 VALUES (?, ?, ?, 1.0, 0.01, 1, '[\"als\"]', 0.03, ?, ?)",
                (customer, rank, f"a{rank}", int(rank == 2), 0.001 if rank == 2 else None),
            )
    conn.commit()
    conn.close()


@pytest.fixture
def store(tmp_path):
    ids = [f"{i:x}".ljust(64, "0") for i in range(16)]
    db = tmp_path / "store.db"
    _build_db(db, ids)
    loaded = score_pipeline.load_store(db, verify_production_pin=False)
    yield loaded
    score_pipeline.close_store(loaded)


def test_random_customer_ids_returns_distinct_existing_ids(store):
    ids = score_pipeline.random_customer_ids(5, store, rng=random.Random(1))
    assert len(set(ids)) == 5
    assert all(score_pipeline.recommend(i, store=store) is not None for i in ids)


def test_sample_customers_empty_without_table(store):
    assert score_pipeline.sample_customers(store) == []


def test_cloud_mode_source_has_no_fastapi_or_full_store_reference():
    for banned in ("requests", "httpx", "API_URL", "8000", "offer_store.db", "DEFAULT_DB_PATH", "local_mode"):
        assert banned not in CLOUD_SOURCE


def test_cloud_app_renders_sample_and_never_imports_local_mode(monkeypatch):
    monkeypatch.setenv("APP_MODE", "cloud")
    monkeypatch.delitem(sys.modules, "dashboard.local_mode", raising=False)
    sample_db = Path(APP_PATH).parent / "sample" / "offer_store_sample.db"
    conn = sqlite3.connect(f"file:{sample_db.as_posix()}?mode=ro", uri=True)
    with_offer, bucket = conn.execute(
        "SELECT t.customer_id, s.recency_bucket FROM served_top12 t JOIN sample_customers s USING (customer_id) "
        "WHERE t.selected = 1 ORDER BY t.customer_id LIMIT 1"
    ).fetchone()
    without_offer = conn.execute(
        "SELECT customer_id FROM sample_customers WHERE recency_bucket = ? AND customer_id NOT IN "
        "(SELECT customer_id FROM served_top12 WHERE selected = 1) ORDER BY customer_id LIMIT 1",
        (bucket,),
    ).fetchone()[0]
    conn.close()

    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    assert not at.exception
    assert "dashboard.local_mode" not in sys.modules
    at.sidebar.selectbox[0].select(bucket).run()
    at.sidebar.selectbox[1].select(with_offer).run()
    assert not at.exception
    assert len(at.dataframe[0].value) == 12
    assert len(at.metric) == 4
    at.sidebar.selectbox[1].select(without_offer).run()
    assert len(at.metric) == 0 and len(at.info) >= 1
    page_text = " ".join(c.value for c in at.caption)
    assert "scaled price units" in page_text and "not a currency" in page_text.lower()


def test_cloud_app_popularity_only_bucket_uses_popularity_strategies(monkeypatch):
    monkeypatch.setenv("APP_MODE", "cloud")
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    at.sidebar.selectbox[0].select("never").run()
    assert not at.exception
    strategies = set(at.dataframe[0].value["Candidate strategies"].str.split(", ").explode())
    assert strategies <= {"recent_popularity", "segment_popularity"}


def _render_components_page(with_offer: bool) -> None:
    import dashboard.components as components

    ranked = [
        {"rank_position": 1, "article_id": "a1", "strategies": ["als"], "p_purchase": 0.01, "eligible": True, "ranker_score": 1.5}
    ]
    selected = {"article_id": "a1", "p_purchase": 0.01, "price": 0.03, "expected_value": 0.001} if with_offer else None
    assumptions = {
        "margin_rate": 0.5,
        "contact_cost_multiplier": 0.001,
        "median_trailing_4w_price": 0.022,
        "contact_cost": 0.000022,
    }
    components.render_recommendation({"selected_offer": selected, "ranked_articles": ranked}, assumptions)


@pytest.mark.parametrize("with_offer", [True, False])
def test_components_state_price_is_scaled_and_ranker_score_is_not_a_probability(with_offer):
    at = AppTest.from_function(_render_components_page, kwargs={"with_offer": with_offer}, default_timeout=60).run()
    assert not at.exception
    text = " ".join(c.value for c in at.caption).lower()
    assert "scaled" in text and "not a currency" in text
    assert "ranker score" in text and "not a probability" in text


def test_invalid_app_mode_shows_error(monkeypatch):
    monkeypatch.setenv("APP_MODE", "staging")
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    assert len(at.error) == 1


def _add_pin(db: Path, ranker: str, classifier: str) -> None:
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE pinned_models (ranker_run_id TEXT NOT NULL, classifier_run_id TEXT NOT NULL)")
    conn.execute("INSERT INTO pinned_models VALUES (?, ?)", (ranker, classifier))
    conn.commit()
    conn.close()


def test_embedded_pin_accepts_matching_ids(tmp_path):
    db = tmp_path / "s.db"
    _build_db(db, ["a" * 64])
    _add_pin(db, "r", "c")
    loaded = score_pipeline.load_store(db, verify_production_pin=False, verify_embedded_pin=True)
    score_pipeline.close_store(loaded)


def test_embedded_pin_refuses_mismatch(tmp_path):
    db = tmp_path / "s.db"
    _build_db(db, ["a" * 64])
    _add_pin(db, "r", "other-classifier")
    with pytest.raises(RuntimeError, match="differ from its embedded pin"):
        score_pipeline.load_store(db, verify_production_pin=False, verify_embedded_pin=True)
    score_pipeline.close_store(score_pipeline.Store(db, "", "", ""))


def test_embedded_pin_refuses_store_without_pin_table(tmp_path):
    db = tmp_path / "s.db"
    _build_db(db, ["a" * 64])
    with pytest.raises(RuntimeError, match="no pinned_models table"):
        score_pipeline.load_store(db, verify_production_pin=False, verify_embedded_pin=True)
    score_pipeline.close_store(score_pipeline.Store(db, "", "", ""))


def test_bundled_sample_carries_a_pin_equal_to_its_run_ids():
    sample_db = Path(APP_PATH).parent / "sample" / "offer_store_sample.db"
    loaded = score_pipeline.load_store(sample_db, verify_production_pin=False, verify_embedded_pin=True)
    score_pipeline.close_store(loaded)
