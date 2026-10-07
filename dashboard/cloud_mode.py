# cloud mode: bundled sample store read in process through src/score_pipeline.py, no other service
import json
from pathlib import Path

import streamlit as st

import dashboard.components as components
import src.score_pipeline as score_pipeline

SAMPLE_DIR = Path(__file__).resolve().parent / "sample"
SAMPLE_DB_PATH = SAMPLE_DIR / "offer_store_sample.db"
SAMPLE_MANIFEST_PATH = SAMPLE_DIR / "offer_store_sample_manifest.json"

POPULARITY_ONLY_BUCKET = "never"


@st.cache_resource
def _load_store() -> score_pipeline.Store:
    return score_pipeline.load_store(SAMPLE_DB_PATH, verify_production_pin=False, verify_embedded_pin=True)


@st.cache_data
def _load_manifest() -> dict:
    return json.loads(SAMPLE_MANIFEST_PATH.read_text(encoding="utf-8"))


def _bucket_label(bucket: str) -> str:
    return "never purchased (popularity strategies only)" if bucket == POPULARITY_ONLY_BUCKET else bucket


def run() -> None:
    store = _load_store()
    manifest = _load_manifest()
    st.title("Two-Stage Next Best Offer Recommender (H&M Personalized Fashion Recommendation)")
    st.caption(
        f"Sample of the offer store: {manifest['customers']:,} customers, {manifest['rows']:,} rows, "
        f"{manifest['customers_per_bucket'][POPULARITY_ONLY_BUCKET]} of them with no purchase before the scoring date."
    )

    customers = score_pipeline.sample_customers(store)
    by_bucket: dict[str, list[str]] = {}
    for customer_id, bucket in customers:
        by_bucket.setdefault(bucket, []).append(customer_id)

    st.sidebar.header("Sample customer")
    bucket = st.sidebar.selectbox("Days since last purchase", sorted(by_bucket), format_func=_bucket_label)
    ids = by_bucket[bucket]
    customer_id = st.sidebar.selectbox(
        "Customer", ids, format_func=lambda c: f"{ids.index(c) + 1:03d}  {c[:12]}"
    )
    k = st.sidebar.slider("Articles shown", 1, 12, 12)
    if bucket == POPULARITY_ONLY_BUCKET:
        st.sidebar.info("This customer has no purchase before the scoring date. Candidates come from the two popularity strategies only.")

    result = score_pipeline.recommend(customer_id, k=k, store=store)
    components.render_recommendation(result, manifest["assumptions"])
