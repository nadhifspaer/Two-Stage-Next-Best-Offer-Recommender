# local mode: full offer store through src/score_pipeline.py, on-demand lookups through FastAPI /recommend
import os

import pandas as pd
import requests
import streamlit as st

import dashboard.components as components
import src.offer_summary as offer_summary
import src.score_pipeline as score_pipeline

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT_S = 10


@st.cache_resource
def _load_store() -> score_pipeline.Store:
    return score_pipeline.load_store()


@st.cache_data
def _load_summary() -> dict:
    return offer_summary.load_offer_summary()


def _fetch(customer_id: str, k: int) -> dict | None:
    response = requests.get(f"{API_URL}/recommend/{customer_id}", params={"k": k}, timeout=REQUEST_TIMEOUT_S)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()


def _render_lookup(assumptions: dict, store: score_pipeline.Store) -> None:
    if "customer_id" not in st.session_state:
        st.session_state["customer_id"] = score_pipeline.random_customer_ids(1, store)[0]
    left, right = st.columns([4, 1])
    typed = left.text_input("Customer id", value=st.session_state["customer_id"])
    right.write("")
    right.write("")
    if right.button("Random customer"):
        st.session_state["customer_id"] = score_pipeline.random_customer_ids(1, store)[0]
        st.rerun()
    st.session_state["customer_id"] = typed.strip()
    k = st.slider("Articles shown", 1, 12, 12)
    try:
        result = _fetch(st.session_state["customer_id"], k)
    except requests.RequestException as exc:
        st.error(f"FastAPI at {API_URL} did not answer: {exc}. Press Random customer or reload once the API is up.")
        return
    if result is None:
        st.warning("Customer id not found in the offer store.")
        return
    components.render_recommendation(result, assumptions)


def _render_summary(summary: dict) -> None:
    coverage = summary["coverage"]
    concentration = summary["top_article_concentration"]
    a, b, c = st.columns(3)
    a.metric("Customers with a selected offer", f"{coverage['covered_customers']:,}")
    b.metric("Offer coverage", f"{coverage['coverage'] * 100:.1f}% of {coverage['total_customers']:,}")
    c.metric("Distinct selected articles", f"{concentration['distinct_selected_articles']:,}")
    st.caption(f"Top 10 articles hold {concentration['share_of_selections_in_top_10'] * 100:.1f}% of selections.")

    st.subheader("Top selected articles")
    top = pd.DataFrame(summary["top_selected_articles"]).rename(
        columns={"article_id": "Article", "selections": "Selections", "price": "Price (scaled units)"}
    )
    st.dataframe(top, hide_index=True, width="stretch")
    st.caption(components.PRICE_UNIT_NOTE)

    st.subheader("Strategy mix of selected offers")
    mix = pd.Series(summary["strategy_mix"], name="Share of selected offers").sort_values(ascending=False)
    st.bar_chart(mix)
    st.caption(
        "Share of selected offers whose article was proposed by each strategy. One article can carry several "
        "strategies, so the shares add to more than 100%."
    )
    components.render_assumptions(summary["assumptions"])


def run() -> None:
    store = _load_store()
    summary = _load_summary()
    components.render_header("local, full offer store and FastAPI", store.scoring_date, store.ranker_run_id, store.classifier_run_id)
    st.sidebar.caption(f"FastAPI: {API_URL}")
    lookup_tab, summary_tab = st.tabs(["Customer lookup", "Offer summary"])
    with lookup_tab:
        _render_lookup(summary["assumptions"], store)
    with summary_tab:
        _render_summary(summary)
