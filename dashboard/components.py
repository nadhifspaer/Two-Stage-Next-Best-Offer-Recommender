# layout and result-display components shared by cloud and local mode
import pandas as pd
import streamlit as st

PRICE_UNIT_NOTE = "Price-derived values are in the dataset's scaled price units. They are not a currency."
SIMULATION_NOTE = (
    "The offer layer is a simulation over two stated assumptions. It is validated against nothing, "
    "because the dataset has no record of offers sent."
)


def format_probability(p: float) -> str:
    return f"{p * 100:.3f}%"


def render_header(mode: str, scoring_date: str, ranker_run_id: str, classifier_run_id: str) -> None:
    st.title("Two-Stage Next Best Offer Recommender (H&M Personalized Fashion Recommendation)")
    st.caption(
        f"Mode: {mode}. Scoring date: {scoring_date}. "
        f"Ranker run {ranker_run_id[:8]}, classifier run {classifier_run_id[:8]}."
    )


def render_assumptions(assumptions: dict) -> None:
    items = [
        f"margin_rate {assumptions['margin_rate']}",
        f"contact_cost {assumptions['contact_cost_multiplier']} of the median trailing 4-week article price",
        f"median trailing 4-week article price {assumptions['median_trailing_4w_price']:.6f}, "
        f"giving contact_cost {assumptions['contact_cost']:.6f}",
        PRICE_UNIT_NOTE,
        SIMULATION_NOTE,
    ]
    st.caption("\n".join(f"- {item}" for item in items))


def _strategy_text(strategies: list[str]) -> str:
    return ", ".join(strategies)


def render_recommendation(result: dict, assumptions: dict) -> None:
    selected = result["selected_offer"]
    st.subheader("Selected offer")
    if selected is None:
        st.info("No offer is selected for this customer: no eligible article has an expected value above zero.")
    else:
        a, b, c, d = st.columns(4)
        a.metric("Article", selected["article_id"])
        b.metric("Calibrated probability", format_probability(selected["p_purchase"]))
        c.metric("Price (scaled units)", f"{selected['price']:.4f}")
        d.metric("Expected value (scaled units)", f"{selected['expected_value']:.6f}")
    render_assumptions(assumptions)

    st.subheader("Ranked articles")
    selected_id = selected["article_id"] if selected else None
    table = pd.DataFrame(
        [
            {
                "Rank": r["rank_position"],
                "Article": r["article_id"],
                "Candidate strategies": _strategy_text(r["strategies"]),
                "Calibrated probability": format_probability(r["p_purchase"]),
                "Eligible for offer": r["eligible"],
                "Selected offer": r["article_id"] == selected_id,
                "Ranker score (relative)": round(r["ranker_score"], 4),
            }
            for r in result["ranked_articles"]
        ]
    )
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption("The ranker score is a relative lambdarank ordering score, not a probability.")
