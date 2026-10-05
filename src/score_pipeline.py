# Scoring interface: serves the committed offer store read-only and never recomputes per request. ranker_score is a relative
# ordering score, never a probability; price and expected_value are scaled price units, never a currency.
import json
import random
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

import src.production_models as production_models

DATA_DIR = Path("data/processed")
DEFAULT_DB_PATH = DATA_DIR / "nbo" / "offer_store.db"

PRICE_UNIT_NOTE = "scaled price units, no currency"
RANKER_SCORE_NOTE = "relative lambdarank ordering score, not a purchase probability"

# one connection per thread: SQLite connections are not safe to share across worker threads
_thread_local = threading.local()


def _get_connection(db_path: Path) -> sqlite3.Connection:
    if not hasattr(_thread_local, "connections"):
        _thread_local.connections = {}
    key = str(db_path)
    if key not in _thread_local.connections:
        uri = f"file:{Path(db_path).resolve().as_posix()}?mode=ro"
        _thread_local.connections[key] = sqlite3.connect(uri, uri=True, check_same_thread=True)
    return _thread_local.connections[key]


@dataclass
class Store:
    db_path: Path
    scoring_date: str
    ranker_run_id: str
    classifier_run_id: str


def _verify_embedded_pin(conn: sqlite3.Connection, db_path: Path, ranker_run_id: str, classifier_run_id: str) -> None:
    # sample stores carry the pin recorded at build time in pinned_models; no external manifest needed
    has_table = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'pinned_models'").fetchone()
    if has_table is None:
        raise RuntimeError(f"store at {db_path} has no pinned_models table, refusing to serve it unverified")
    pinned_ranker, pinned_classifier = conn.execute("SELECT ranker_run_id, classifier_run_id FROM pinned_models").fetchone()
    if ranker_run_id != pinned_ranker or classifier_run_id != pinned_classifier:
        raise RuntimeError(
            f"store at {db_path} has model run ids ranker={ranker_run_id!r}, classifier={classifier_run_id!r} "
            f"that differ from its embedded pin ranker={pinned_ranker!r}, classifier={pinned_classifier!r}. Rebuild the sample."
        )


def load_store(
    db_path: Path = DEFAULT_DB_PATH, verify_production_pin: bool = True, verify_embedded_pin: bool = False
) -> Store:
    """verify_production_pin cross-checks the store's run ids against the pinned production config and refuses a mismatch;
    tests pass False because their fixtures use fake run ids."""
    conn = _get_connection(db_path)
    scoring_date, ranker_run_id, classifier_run_id = conn.execute(
        "SELECT scoring_date, ranker_run_id, classifier_run_id FROM metadata"
    ).fetchone()

    if verify_production_pin:
        pinned = production_models.load_production_models()
        if ranker_run_id != pinned.ranker_run_id or classifier_run_id != pinned.classifier_run_id:
            raise RuntimeError(
                f"offer store at {db_path} was built from ranker_run_id={ranker_run_id!r}, "
                f"classifier_run_id={classifier_run_id!r}, which does not match the pinned "
                f"production models ranker_run_id={pinned.ranker_run_id!r}, "
                f"classifier_run_id={pinned.classifier_run_id!r} in "
                f"{production_models.MODELS_MANIFEST_PATH}. Rebuild the offer store."
            )

    if verify_embedded_pin:
        _verify_embedded_pin(conn, db_path, ranker_run_id, classifier_run_id)

    return Store(
        db_path=Path(db_path),
        scoring_date=scoring_date,
        ranker_run_id=ranker_run_id,
        classifier_run_id=classifier_run_id,
    )


_default_store: Store | None = None


def get_default_store() -> Store:
    global _default_store
    if _default_store is None:
        _default_store = load_store()
    return _default_store


def recommend(customer_id: str, k: int = 12, store: Store | None = None) -> dict | None:
    """None means customer_id is not in the offer store. k is the number of ranked articles returned, 1 to 12;
    selected_offer is looked up over the full top-12 row set, independent of k."""
    if not 1 <= k <= 12:
        raise ValueError("k must be between 1 and 12")

    store = store or get_default_store()
    conn = _get_connection(store.db_path)
    rows = conn.execute(
        "SELECT article_id, rank, ranker_score, p_purchase, eligible, strategies, price, selected, expected_value "
        "FROM served_top12 WHERE customer_id = ? ORDER BY rank",
        (customer_id,),
    ).fetchall()
    if not rows:
        return None

    ranked_articles = []
    selected_offer = None
    for i, (article_id, rank, ranker_score, p_purchase, eligible, strategies, price, selected, expected_value) in enumerate(rows):
        if i < k:
            ranked_articles.append(
                {
                    "article_id": article_id,
                    "rank_position": rank,
                    "ranker_score": ranker_score,
                    "p_purchase": p_purchase,
                    "eligible": bool(eligible),
                    "strategies": json.loads(strategies),
                }
            )
        if selected:
            selected_offer = {
                "article_id": article_id,
                "expected_value": expected_value,
                "price": price,
                "p_purchase": p_purchase,
            }

    return {
        "customer_id": customer_id,
        "scoring_date": store.scoring_date,
        "k": k,
        "notes": {"price_unit": PRICE_UNIT_NOTE, "ranker_score": RANKER_SCORE_NOTE},
        "model_run_ids": {"ranker": store.ranker_run_id, "classifier": store.classifier_run_id},
        "ranked_articles": ranked_articles,
        "selected_offer": selected_offer,
    }


def close_store(store: Store) -> None:
    # closes this thread's cached connection to the store's database
    connections = getattr(_thread_local, "connections", {})
    conn = connections.pop(str(store.db_path), None)
    if conn is not None:
        conn.close()


def sample_customers(store: Store) -> list[tuple[str, str]]:
    """(customer_id, recency_bucket) rows of a sample store's sample_customers
    table, empty for a store without one."""
    conn = _get_connection(store.db_path)
    has_table = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'sample_customers'").fetchone()
    if has_table is None:
        return []
    return conn.execute("SELECT customer_id, recency_bucket FROM sample_customers ORDER BY recency_bucket, customer_id").fetchall()


def random_customer_ids(n: int, store: Store | None = None, rng: random.Random | None = None) -> list[str]:
    """n customer_ids found by index seeks from random hex probes, so the
    draw is approximately uniform and never scans the table."""
    store = store or get_default_store()
    conn = _get_connection(store.db_path)
    rng = rng or random.Random()
    found: list[str] = []
    for _ in range(50 * n):
        if len(found) == n:
            break
        probe = f"{rng.getrandbits(64):016x}"
        row = conn.execute("SELECT customer_id FROM served_top12 WHERE customer_id >= ? LIMIT 1", (probe,)).fetchone()
        if row is None:
            row = conn.execute("SELECT customer_id FROM served_top12 LIMIT 1").fetchone()
        if row[0] not in found:
            found.append(row[0])
    if len(found) < n:
        raise RuntimeError(f"found {len(found)} of {n} distinct customer_ids after {50 * n} probes")
    return found
