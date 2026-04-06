"""
pages/recommendations.py — Aspirational Recommendation Engine (hero page).

Showcases the aspirational contextual bandit with transparent score breakdowns,
metrics delta vs. baseline, and exploration pick labeling.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from recommender import (
    RecommendationResult,
    ScoredPost,
    recommend_aspirational,
    recommend_baseline,
)

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
# The epsilon slider renders below the metrics but its code runs first so
# the returned value is available for recommendation generation.

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

# ── Generate Recommendations ─────────────────────────────────────────────────

asp_result: RecommendationResult = recommend_aspirational(
    user_id=selected_user,
    data=data,
    extraction=extraction,
    n=N_TOTAL,
    epsilon=epsilon,
    seed=42,
)

base_result: RecommendationResult = recommend_baseline(
    user_id=selected_user,
    data=data,
    extraction=extraction,
    n=N_TOTAL,
    epsilon=epsilon,
    seed=42,
)

# ── Header (fills reserved slot above slider) ────────────────────────────────

with header_area:
    st.title("🌱 Aspirational Recommendation Engine")
    st.markdown(
        "This feed is powered by a **contextual bandit** that balances historical engagement "
        "with *aspirational alignment* — recommending content that nudges the user toward "
        "who they **want to become**, not just what they've consumed before."
    )

# ── Metrics Row (fills reserved slot above slider) ───────────────────────────

m_asp = asp_result.metrics
m_base = base_result.metrics

with metrics_area:
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)

    with col_m1:
        st.metric(
            "Aspirational Alignment",
            f"{m_asp.aspirational_alignment_rate:.0%}",
            delta=f"{m_asp.aspirational_alignment_rate - m_base.aspirational_alignment_rate:+.0%} vs baseline",
            help="% of recommended posts whose topic aligns with the user's aspirational state (score above threshold). Higher is better.",
        )
    with col_m2:
        st.metric(
            "Regression Rate",
            f"{m_asp.regression_rate:.0%}",
            delta=f"{m_asp.regression_rate - m_base.regression_rate:+.0%} vs baseline",
            delta_color="inverse",
            help="% of recommended posts that reinforce the user's current negative state (regression penalty fired). Lower is better.",
        )
    with col_m3:
        st.metric(
            "Topic Diversity",
            f"{m_asp.topic_entropy:.2f}",
            delta=f"{m_asp.topic_entropy - m_base.topic_entropy:+.2f} vs baseline",
            help="Shannon entropy of aspirational topic labels across the batch. Higher = more diverse topic mix.",
        )
    with col_m4:
        st.metric(
            "Avg Relevance",
            f"{m_asp.avg_relevance:.2f}",
            delta=f"{m_asp.avg_relevance - m_base.avg_relevance:+.2f} vs baseline",
            help="Mean content-similarity to engagement centroid. Measures how relevant recommendations are to the user's history.",
        )
    with col_m5:
        st.metric(
            "Avg Score",
            f"{m_asp.avg_score:.2f}",
            delta=f"{m_asp.avg_score - m_base.avg_score:+.2f} vs baseline",
            help="Mean total score (historical + aspirational + regression) across all recommendations.",
        )

# ── Caption & Score Decomposition Chart ─────────────────────────────────────

st.caption(
    f"ε-greedy contextual bandit  ·  ε = {epsilon:.2f}  ·  "
    f"metrics computed on top {N_TOTAL} recommendations  ·  🎲 = exploration pick"
)

_chart_recs = asp_result.recommendations[:N_TOTAL]
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
            "Stacked bar chart showing how each post's total score decomposes into "
            "historical relevance, aspirational alignment, and regression penalty."
        )
        st.bar_chart(
            _chart_data.set_index("Post"),
            color=["#6baed6", "#74c476", "#e6550d"],
            horizontal=True,
            height=max(250, len(_chart_recs) * 28),
        )

st.markdown("---")

# ── Paginated Aspirational Feed ──────────────────────────────────────────────

# Track how many recs to show
if "asp_visible" not in st.session_state:
    st.session_state.asp_visible = PAGE_SIZE


def render_aspirational_card(rec: ScoredPost, rank: int) -> None:
    """Render an aspirational recommendation card with full score decomposition."""
    post = post_lookup.get(rec.post_id)
    snippet_text = post.snippet if post else "(post text unavailable)"
    title = f"#{rank} — Post {rec.post_id}"
    if rec.is_exploration:
        title += "  🎲"

    # Build the score badge line
    score_parts = []
    if rec.breakdown.historical > 0:
        score_parts.append(f"📖 Relevance: **{rec.breakdown.historical:.2f}**")
    if rec.breakdown.aspirational > 0:
        score_parts.append(f"🌱 Aspirational: **+{rec.breakdown.aspirational:.2f}**")
    if rec.breakdown.regression < 0:
        score_parts.append(f"⚠️ Regression: **{rec.breakdown.regression:.2f}**")

    score_line = " &nbsp;|&nbsp; ".join(score_parts) if score_parts else "*No signal — no engagement or state alignment detected*"

    with st.expander(f"{title}  —  Total: **{rec.score:.2f}**", expanded=(rank <= 3)):
        if rec.is_exploration:
            st.caption("🎲 Exploration pick — randomly selected to introduce diversity (ε-greedy)")
        st.markdown(f"**Score Breakdown:** {score_line}")
        st.markdown(f"> {snippet_text}")

        # Topic keywords from spaCy
        if post and post.topic_keywords:
            kw_display = ", ".join(post.topic_keywords[:8])
            st.caption(f"🏷️ Topics: {kw_display}")

        # Post profile: top aspirational & current tags
        if rec.post_profile:
            asp_tag = rec.post_profile.top_aspirational_topic
            cur_tag = rec.post_profile.top_current_topic
            st.caption(
                f"Post aspirational topic: **{asp_tag.label.replace('_', ' ')}** ({asp_tag.score:.2f}) · "
                f"Post current topic: **{cur_tag.label.replace('_', ' ')}** ({cur_tag.score:.2f})"
            )


visible = st.session_state.asp_visible
for i, rec in enumerate(asp_result.recommendations[:visible], 1):
    render_aspirational_card(rec, i)

# Show more / Show less buttons
col_less, col_more = st.columns(2)
with col_less:
    if visible > PAGE_SIZE:
        if st.button("Show less", key="asp_less"):
            st.session_state.asp_visible = max(PAGE_SIZE, visible - PAGE_SIZE)
            st.rerun()
with col_more:
    if visible < len(asp_result.recommendations):
        if st.button("Show more", key="asp_more"):
            st.session_state.asp_visible = min(len(asp_result.recommendations), visible + PAGE_SIZE)
            st.rerun()
