"""
app.py — Entry point for the Aspirational Recommendation System.

Sets up shared data loading, sidebar controls, and multi-page navigation.

Run with: streamlit run app.py
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

# ── Environment Fixes for Deployment ──────────────────────────────────────────

# Fix for "TypeError: Descriptors cannot be created directly" (protobuf mismatch)
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Mock torchvision to satisfy transformers discovery without installing 500MB+
# This prevents the "ModuleNotFoundError" and potential import deadlocks.
if "torchvision" not in sys.modules:
    import types
    from importlib.machinery import ModuleSpec
    
    mock_tv = types.ModuleType("torchvision")
    mock_tv.__path__ = []
    mock_tv.__spec__ = ModuleSpec("torchvision", None, is_package=True)
    sys.modules["torchvision"] = mock_tv
    
    # Submodules
    for sub in ["transforms", "transforms.v2", "ops", "io", "ops.boxes"]:
        full_name = f"torchvision.{sub}"
        m = MagicMock(name=full_name)
        m.__spec__ = ModuleSpec(full_name, None)
        sys.modules[full_name] = m
        # Ensure 'transforms' attribute exists on the main mock_tv
        if "." not in sub:
            setattr(mock_tv, sub, m)

import streamlit as st

from data_loader import load_all
from state_extractor import UserProfile, extract_all

# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Aspirational Recommendation System",
    page_icon="🌱",
    layout="wide",
)

# ── Cached Data Loading ──────────────────────────────────────────────────────


@st.cache_resource(show_spinner="Loading data & extracting user states…")
def load_pipeline():
    """Load data and run extraction once, cached across reruns."""
    data = load_all()
    extraction = extract_all(data)
    return data, extraction


data, extraction = load_pipeline()

# Store in session state for pages to access
st.session_state.data = data
st.session_state.extraction = extraction
st.session_state.post_lookup = {p.post_id: p for p in data.posts}

# Initialize default epsilon (pages own the slider)
if "epsilon" not in st.session_state:
    st.session_state.epsilon = 0.1

# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title("🌱 Aspirational Rec Engine")

# Navigation (page links render here)
pg = st.navigation([
    st.Page("pages/recommendations.py", title="Recommendations", icon=":material/spa:", default=True),
    st.Page("pages/baseline.py", title="Baseline Feed", icon=":material/bar_chart:"),
    st.Page("pages/state_viz.py", title="User States", icon=":material/psychology:"),
    st.Page("pages/conversation.py", title="Conversation", icon=":material/chat:"),
    st.Page("pages/evaluation.py", title="Evaluation", icon=":material/monitoring:"),
    st.Page("pages/explanation.py", title="How It Works", icon=":material/school:"),
])

# ── User Selection & Inferred Profile ────────────────────────────────────────

st.sidebar.markdown("---")

profiled_ids = sorted(extraction.user_profiles.keys())
selected_user = st.sidebar.selectbox(
    "Select a user",
    profiled_ids,
    format_func=lambda uid: f"User {uid}",
)

user_profile: UserProfile = extraction.user_profiles[selected_user]

st.sidebar.subheader("Inferred User Profile")

st.sidebar.markdown("**Current State** (struggling with)")
for s in user_profile.current_states[:3]:
    label_clean = s.label.replace("_", " ").title()
    st.sidebar.markdown(
        f"- {label_clean} <span style='color:#e74c3c'>{s.score:.2f}</span>",
        unsafe_allow_html=True,
    )

st.sidebar.markdown("**Aspirational State** (wants to become)")
for s in user_profile.aspirational_states[:3]:
    label_clean = s.label.replace("_", " ").title()
    st.sidebar.markdown(
        f"- {label_clean} <span style='color:#2ecc71; font-weight:600'>{s.score:.2f}</span>",
        unsafe_allow_html=True,
    )

# ── Engagement History ───────────────────────────────────────────────────────

activity = data.user_activity_index.get(selected_user)
if activity:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Engagement History**")
    st.sidebar.markdown(
        f"- Posts read: **{len(activity.read_posts)}**\n"
        f"- Posts commented: **{len(activity.commented_posts)}**\n"
        f"- Posts created: **{len(activity.created_posts)}**"
    )

# ── Credit ───────────────────────────────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.caption("Built by Kevin King")

# Store selection for pages
st.session_state.selected_user = selected_user

pg.run()
