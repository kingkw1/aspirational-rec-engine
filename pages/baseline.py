"""
pages/baseline.py — Baseline (content-similarity) feed for reference comparison.

Shows what a standard engagement-driven recommender would produce — no aspirational
signal, no regression penalty. Useful as a reference point for how the Aspirational
Engine improves upon a purely similarity-based approach.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from recommender import ScoredPost, recommend_baseline

# ── Constants ────────────────────────────────────────────────────────────────

N_TOTAL = 20   # always score this many
PAGE_SIZE = 5  # show this many at a time

# ── Read shared state ────────────────────────────────────────────────────────

if "data" not in st.session_state:
    st.switch_page("app.py")

data = st.session_state.data
extraction = st.session_state.extraction
post_lookup = st.session_state.post_lookup
selected_user = st.session_state.selected_user

# ── Layout: reserve visual slots for header and metrics ──────────────────────

header_area = st.container()
metrics_area = st.container()

# ── Epsilon Slider ───────────────────────────────────────────────────────────

epsilon = st.slider(
    "Exploration rate (ε)",
    min_value=0.0,
    max_value=1.0,
    value=st.session_state.get("epsilon", 0.1),
    step=0.05,
    help="Higher ε → more random exploration. Lower ε → stricter exploitation of top scores.",
)
st.session_state.epsilon = epsilon

# ── Generate Baseline Recommendations ────────────────────────────────────────

base_result = recommend_baseline(
    user_id=selected_user,
    data=data,
    extraction=extraction,
    n=N_TOTAL,
    epsilon=epsilon,
    seed=42,
)

# ── Header (fills reserved slot above slider) ────────────────────────────────

with header_area:
    st.title("📊 Baseline Feed (Content-Similarity)")
    st.markdown(
        "This is what a **standard engagement-driven recommender** would produce. "
        "It builds a preference centroid from the user's engagement history, then ranks "
        "*unseen* posts by cosine similarity to that centroid — with the same ε-greedy "
        "exploration as the aspirational engine. "
        "**No aspirational shaping, no regression penalty** — just \"more of what you liked.\""
    )

# ── Metrics (fills reserved slot above slider) ──────────────────────────────

m = base_result.metrics

with metrics_area:
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(
            "Aspirational Alignment",
            f"{m.aspirational_alignment_rate:.0%}",
            help="% of posts whose topic happens to align with aspirational state. This engine doesn't optimize for it — any alignment is incidental.",
        )
    with col2:
        st.metric(
            "Regression Rate",
            f"{m.regression_rate:.0%}",
            help="% of posts that reinforce the user's current negative state. No penalty applied — regressions pass through unchecked.",
        )
    with col3:
        st.metric(
            "Topic Diversity",
            f"{m.topic_entropy:.2f}",
            help="Shannon entropy of aspirational topic labels. Measures topic variety.",
        )
    with col4:
        st.metric(
            "Avg Relevance",
            f"{m.avg_relevance:.2f}",
            help="Mean content-similarity to engagement centroid. This is the only signal the baseline uses for ranking.",
        )
    with col5:
        st.metric(
            "Avg Score",
            f"{m.avg_score:.2f}",
            help="Mean total score (historical + aspirational + regression). Aspirational/regression are measured but were NOT used for ranking.",
        )

# ── Caption & Score Decomposition Chart ─────────────────────────────────────

st.caption(
    f"Content-similarity engine  ·  ε = {epsilon:.2f}  ·  "
    f"metrics computed on top {N_TOTAL} recommendations  ·  🎲 = exploration pick"
)

_chart_recs = base_result.recommendations[:N_TOTAL]
if _chart_recs:
    _chart_data = pd.DataFrame(
        [
            {
                "Post": f"#{i:02d} · {r.post_id}" + (" 🎲" if r.is_exploration else ""),
                "Historical": r.breakdown.historical,
                "Aspirational": r.breakdown.aspirational,
                "Regression": r.breakdown.regression,
            }
            for i, r in enumerate(_chart_recs, 1)
        ]
    )
    with st.expander("📊 Score Decomposition Chart", expanded=False):
        st.caption(
            "Stacked bar chart showing how each post's total score decomposes. "
            "The baseline engine ranks by historical relevance only — aspirational "
            "and regression components are measured retroactively for comparison."
        )
        st.bar_chart(
            _chart_data.set_index("Post"),
            color=["#6baed6", "#74c476", "#e6550d"],
            horizontal=True,
            height=max(250, len(_chart_recs) * 28),
        )

st.markdown("---")

# ── Paginated Feed ───────────────────────────────────────────────────────────

if "base_visible" not in st.session_state:
    st.session_state.base_visible = PAGE_SIZE


def render_baseline_card(rec: ScoredPost, rank: int) -> None:
    """Render a baseline recommendation card with content-similarity score."""
    post = post_lookup.get(rec.post_id)
    snippet_text = post.snippet if post else "(post text unavailable)"
    title = f"#{rank} — Post {rec.post_id}"
    if rec.is_exploration:
        title += "  🎲"

    with st.expander(f"{title}  —  Relevance: **{rec.score:.2f}**", expanded=(rank <= 3)):
        if rec.is_exploration:
            st.caption("🎲 Exploration pick — randomly selected to introduce diversity (ε-greedy)")
        if rec.breakdown.historical > 0:
            st.markdown(
                f"📖 **Similarity to engagement history:** {rec.breakdown.historical:.2f}  \n"
                "*Ranked by how similar this post is to content the user previously engaged with.*"
            )
        else:
            st.markdown("*No engagement history to compare against.*")
        st.markdown(f"> {snippet_text}")

        if post and post.topic_keywords:
            kw_display = ", ".join(post.topic_keywords[:8])
            st.caption(f"🏷️ Topics: {kw_display}")


visible = st.session_state.base_visible
for i, rec in enumerate(base_result.recommendations[:visible], 1):
    render_baseline_card(rec, i)

# Show more / Show less buttons
col_less, col_more = st.columns(2)
with col_less:
    if visible > PAGE_SIZE:
        if st.button("Show less", key="base_less"):
            st.session_state.base_visible = max(PAGE_SIZE, visible - PAGE_SIZE)
            st.rerun()
with col_more:
    if visible < len(base_result.recommendations):
        if st.button("Show more", key="base_more"):
            st.session_state.base_visible = min(len(base_result.recommendations), visible + PAGE_SIZE)
            st.rerun()
