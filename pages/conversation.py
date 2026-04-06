"""
pages/conversation.py — AI Health Coach Conversation Viewer page.

Displays the raw conversation used to infer the user's current and aspirational states,
with chat-style message bubbles.
"""

from __future__ import annotations

import streamlit as st

# ── Read shared state ────────────────────────────────────────────────────────

if "data" not in st.session_state:
    st.switch_page("app.py")

data = st.session_state.data
selected_user = st.session_state.selected_user

# ── Header ───────────────────────────────────────────────────────────────────

st.title("💬 AI Health Coach Conversation")
st.markdown(
    "The raw conversation used to infer this user's current and aspirational states. "
    "The system uses **only the user's messages** (not the coach's) to compute embeddings "
    "and match against the state taxonomy."
)

# ── Conversation Messages ────────────────────────────────────────────────────

user_conv = next((c for c in data.conversations if c.ref_user_id == selected_user), None)

if user_conv:
    st.caption(f"{len(user_conv.messages)} messages · {len(user_conv.user_messages)} from user")
    for msg in user_conv.messages:
        is_bot = msg.ref_user_id == 1
        role = "assistant" if is_bot else "user"
        with st.chat_message(role):
            st.markdown(msg.text)
else:
    st.info("No AI Health Coach conversation found for this user.")
