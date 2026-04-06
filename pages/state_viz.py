"""
pages/state_viz.py — User State Visualization page.

Two interactive Plotly visualizations:
1. Butterfly Chart: normalized profile showing current struggles (left) vs
   aspirational goals (right), with low-signal dimensions grayed out.
2. Recommendation Alignment Radar: how well each engine's recs match the
   user's aspirational profile.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from recommender import recommend_aspirational, recommend_baseline
from state_extractor import ASPIRATIONAL_STATE_TAXONOMY, CURRENT_STATE_TAXONOMY

# ── Constants ────────────────────────────────────────────────────────────────

N_RECS = 20  # match the recommendation pages

# Natural pairings: current_state → aspirational_state (index-aligned)
# Each tuple: (current_label, aspirational_label, display_name)
STATE_PAIRS = [
    ("anxiety_and_overwhelm", "emotionally_resilient", "Anxiety → Resilience"),
    ("social_isolation", "socially_connected", "Isolation → Connection"),
    ("physical_health_decline", "physically_active", "Health Decline → Active"),
    ("technology_frustration", "independent_and_capable", "Tech Frustration → Capable"),
    ("grief_and_loss", "purposeful_and_engaged", "Grief → Purpose"),
    ("work_life_imbalance", "healthy_and_balanced", "Imbalance → Balance"),
    ("low_motivation", "adventurous_and_growing", "Low Motivation → Growth"),
    ("fear_and_uncertainty", "creative_and_curious", "Fear → Curiosity"),
]

# Default labels (taxonomy insertion order)
ASPIRATIONAL_LABELS = [
    label.replace("_", " ").title()
    for label, _ in ASPIRATIONAL_STATE_TAXONOMY
]

# Semantically grouped order for radar axes.
# Four thematic clusters are arranged consecutively so that related states are
# adjacent on the chart and produce interpretable polygon shapes:
#   Physical health → Agency / capability → Exploration / growth →
#   Social / emotional → (wraps back to Physical)
RADAR_AXIS_ORDER = [
    "physically_active",       # ── Physical health cluster
    "healthy_and_balanced",    #    lifestyle & recovery (adjacent to Active)
    "independent_and_capable", # ── Agency cluster
    "adventurous_and_growing", #    expanding capacity & new challenges
    "creative_and_curious",    # ── Exploration cluster
    "purposeful_and_engaged",  #    meaning-making (adjacent to Creative)
    "socially_connected",      # ── Social / emotional cluster
    "emotionally_resilient",   #    bridges Social back to Physical
]

RADAR_DISPLAY_LABELS = [
    label.replace("_", " ").title()
    for label in RADAR_AXIS_ORDER
]

# ── Read shared state ────────────────────────────────────────────────────────

if "data" not in st.session_state:
    st.switch_page("app.py")

data = st.session_state.data
extraction = st.session_state.extraction
selected_user = st.session_state.selected_user
epsilon = st.session_state.get("epsilon", 0.1)

user_profile = extraction.user_profiles[selected_user]

# ── Header ───────────────────────────────────────────────────────────────────

st.title("🧭 User State Visualization")
st.markdown(
    "Interactive visualizations of the selected user's **inferred state profile** — "
    "where they are now, where they want to go, and how well each recommendation "
    "engine aligns with their aspirational direction."
)

# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: State Profile (Butterfly Chart)
# ═══════════════════════════════════════════════════════════════════════════════

st.header("1. The Aspirational Gap")
st.markdown(
    "A diverging bar chart showing *what proportion* of the user's profile "
    "each dimension accounts for. **Current struggles** extend left (red); "
    "**aspirational goals** extend right (green). Dimensions grayed out have "
    "insufficient signal (below the noise floor)."
)

# Build score lookups from the user's full state profiles
current_scores = {s.label: s.score for s in user_profile.current_states}
aspirational_scores = {s.label: s.score for s in user_profile.aspirational_states}

# Normalize each taxonomy into a proportion (sum-to-1)
cur_raw = np.array([current_scores.get(c, 0.0) for c, _, _ in STATE_PAIRS])
asp_raw = np.array([aspirational_scores.get(a, 0.0) for _, a, _ in STATE_PAIRS])

cur_total = cur_raw.sum() or 1.0
asp_total = asp_raw.sum() or 1.0
cur_pct = cur_raw / cur_total * 100  # percentages
asp_pct = asp_raw / asp_total * 100

# Significance threshold: if the raw cosine similarity is below this, the
# dimension is indistinguishable from noise for MiniLM-L6-v2.
SIG_THRESHOLD = 0.22

# Sort rows by largest aspirational share (top) to smallest
order = np.argsort(asp_pct)  # ascending; we'll reverse for display
# Separate left (current) and right (aspirational) labels
labels_current = [STATE_PAIRS[i][2].split(" → ")[0] for i in order]
labels_aspirational = [STATE_PAIRS[i][2].split(" → ")[1] for i in order]
cur_sorted = cur_pct[order]
asp_sorted = asp_pct[order]
cur_raw_sorted = cur_raw[order]
asp_raw_sorted = asp_raw[order]

# Determine which rows have meaningful signal
cur_sig = cur_raw_sorted >= SIG_THRESHOLD
asp_sig = asp_raw_sorted >= SIG_THRESHOLD

# Colors: full color if significant, light gray if not
RED = "#e74c3c"
GREEN = "#2ecc71"
GRAY = "#d5d5d5"
cur_colors = [RED if s else GRAY for s in cur_sig]
asp_colors = [GREEN if s else GRAY for s in asp_sig]

fig_butterfly = go.Figure()

# Current struggles (left-extending = negative x) — plotted on primary y-axis
fig_butterfly.add_trace(go.Bar(
    y=labels_current,
    x=[-v for v in cur_sorted],
    orientation="h",
    name="Current Struggles",
    marker_color=cur_colors,
    hovertemplate=[
        f"{labels_current[i]}<br>Current: {cur_sorted[i]:.1f}% (raw: {cur_raw_sorted[i]:.3f})"
        + ("<br>⚠️ below noise floor" if not cur_sig[i] else "")
        + "<extra></extra>"
        for i in range(len(labels_current))
    ],
))

# Aspirational goals (right-extending = positive x) — plotted on secondary y-axis
fig_butterfly.add_trace(go.Bar(
    y=labels_aspirational,
    x=list(asp_sorted),
    orientation="h",
    name="Aspirational Goals",
    marker_color=asp_colors,
    yaxis="y2",
    hovertemplate=[
        f"{labels_aspirational[i]}<br>Aspirational: {asp_sorted[i]:.1f}% (raw: {asp_raw_sorted[i]:.3f})"
        + ("<br>⚠️ below noise floor" if not asp_sig[i] else "")
        + "<extra></extra>"
        for i in range(len(labels_aspirational))
    ],
))

max_pct = max(cur_pct.max(), asp_pct.max()) * 1.15

fig_butterfly.update_layout(
    xaxis=dict(
        title="← Current Struggles (%)  |  Aspirational Goals (%) →",
        range=[-max_pct, max_pct],
        tickvals=[-30, -20, -10, 0, 10, 20, 30],
        ticktext=["30%", "20%", "10%", "0", "10%", "20%", "30%"],
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor="#333",
    ),
    yaxis=dict(title="", side="left"),
    yaxis2=dict(title="", side="right", overlaying="y"),
    barmode="overlay",
    height=420,
    margin=dict(l=160, r=160, t=20, b=50),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    template="plotly_white",
)

st.plotly_chart(fig_butterfly, use_container_width=True)

st.caption(
    "Raw cosine similarity scores are normalized into percentages within each "
    "taxonomy so the bars represent *share of profile*, not absolute similarity. "
    "Gray bars indicate dimensions below the noise floor "
    f"(cosine similarity < {SIG_THRESHOLD}) — the model has insufficient evidence "
    "to distinguish them from background."
)

# Highlight dominant dimensions
top_cur_idx = int(np.argmax(cur_pct))
top_asp_idx = int(np.argmax(asp_pct))
st.info(
    f"**Dominant struggle:** {STATE_PAIRS[top_cur_idx][2].split(' → ')[0]} "
    f"({cur_pct[top_cur_idx]:.0f}% of current profile)  \n"
    f"**Dominant aspiration:** {STATE_PAIRS[top_asp_idx][2].split(' → ')[1]} "
    f"({asp_pct[top_asp_idx]:.0f}% of aspirational profile)",
    icon="🎯",
)

# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: Recommendation Alignment Radar
# ═══════════════════════════════════════════════════════════════════════════════

st.header("2. Recommendation Alignment")
st.markdown(
    "The radar chart shows the user's **aspirational profile** (what they want to "
    "become) overlaid with the **topic footprint** of recommendations from each "
    "engine. A well-aligned engine's shape should track the user's aspirational shape."
)

# Generate recommendations for both engines
asp_result = recommend_aspirational(
    user_id=selected_user, data=data, extraction=extraction,
    n=N_RECS, epsilon=epsilon, seed=42,
)
base_result = recommend_baseline(
    user_id=selected_user, data=data, extraction=extraction,
    n=N_RECS, epsilon=epsilon, seed=42,
)

# Build the user's aspirational profile vector in semantic radar order
user_asp_vector = [aspirational_scores.get(label, 0.0) for label in RADAR_AXIS_ORDER]

# Compute mean aspirational topic scores across recommended posts for each engine
def _rec_footprint(recs):
    """Compute mean aspirational state scores across recommended posts."""
    if not recs:
        return [0.0] * len(RADAR_AXIS_ORDER)
    scores_per_dim = {label: [] for label in RADAR_AXIS_ORDER}
    for rec in recs:
        if rec.post_profile and rec.post_profile.aspirational_state_scores:
            for ss in rec.post_profile.aspirational_state_scores:
                if ss.label in scores_per_dim:
                    scores_per_dim[ss.label].append(ss.score)
    return [
        float(np.mean(scores_per_dim[label])) if scores_per_dim[label] else 0.0
        for label in RADAR_AXIS_ORDER
    ]


asp_footprint = _rec_footprint(asp_result.recommendations)
base_footprint = _rec_footprint(base_result.recommendations)

# Close the radar polygon by repeating the first value
radar_labels = RADAR_DISPLAY_LABELS + [RADAR_DISPLAY_LABELS[0]]
user_radar = user_asp_vector + [user_asp_vector[0]]
asp_radar = asp_footprint + [asp_footprint[0]]
base_radar = base_footprint + [base_footprint[0]]

fig_radar = go.Figure()

# User's aspirational profile (dashed green)
fig_radar.add_trace(go.Scatterpolar(
    r=user_radar,
    theta=radar_labels,
    name="User Aspirational Profile",
    line=dict(color="#2ecc71", width=2, dash="dash"),
    fill="none",
    hovertemplate="%{theta}<br>User: %{r:.3f}<extra></extra>",
))

# Aspirational engine footprint (solid blue)
fig_radar.add_trace(go.Scatterpolar(
    r=asp_radar,
    theta=radar_labels,
    name="Aspirational Engine Recs",
    line=dict(color="#3498db", width=2),
    fill="toself",
    opacity=0.3,
    hovertemplate="%{theta}<br>Asp Engine: %{r:.3f}<extra></extra>",
))

# Baseline engine footprint (dark orange)
fig_radar.add_trace(go.Scatterpolar(
    r=base_radar,
    theta=radar_labels,
    name="Baseline Recs",
    line=dict(color="#e67e22", width=2),
    fill="toself",
    opacity=0.25,
    hovertemplate="%{theta}<br>Baseline: %{r:.3f}<extra></extra>",
))

fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(visible=True, range=[0, max(max(user_radar), max(asp_radar), max(base_radar)) * 1.1]),
    ),
    height=500,
    margin=dict(l=80, r=80, t=40, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
    template="plotly_white",
)

st.plotly_chart(fig_radar, use_container_width=True)

st.caption(
    f"Radar shows mean aspirational topic scores across {N_RECS} recommended posts "
    f"(ε = {epsilon:.2f}). The green dashed line is the user's aspirational profile. "
    f"The blue fill is the aspirational engine's recommendation footprint; the gray "
    f"fill is the baseline's."
)

# Compute alignment score (cosine similarity between user aspirational vector and rec footprint)
def _cosine(a, b):
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


asp_alignment = _cosine(user_asp_vector, asp_footprint)
base_alignment = _cosine(user_asp_vector, base_footprint)

col1, col2 = st.columns(2)
with col1:
    st.metric(
        "Aspirational Engine Alignment",
        f"{asp_alignment:.3f}",
        help="Cosine similarity between the user's aspirational profile and the engine's recommendation topic footprint.",
    )
with col2:
    st.metric(
        "Baseline Alignment",
        f"{base_alignment:.3f}",
        delta=f"{base_alignment - asp_alignment:+.3f} vs aspirational",
        delta_color="inverse",
        help="Cosine similarity between the user's aspirational profile and the baseline's recommendation topic footprint.",
    )
