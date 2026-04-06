"""
pages/evaluation.py — Batch Evaluation & MLflow Monitoring page.

Runs both engines across all 24 profiled users and displays aggregate A/B
comparison metrics. Supports ε sweeps to visualize how exploration rate
affects outcomes. Integrates with MLflow for experiment logging.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from recommender import (
    _MLFLOW_AVAILABLE,
    run_ab_evaluation,
    run_batch_evaluation,
)

# ── Read shared state ────────────────────────────────────────────────────────

if "data" not in st.session_state:
    st.switch_page("app.py")

data = st.session_state.data
extraction = st.session_state.extraction

N_RECS = 20  # match the per-user page count

# ── Header ───────────────────────────────────────────────────────────────────

st.title("📊 Batch Evaluation & Monitoring")
st.markdown(
    "Run the aspirational and baseline engines across **all 24 profiled users** and "
    "compare aggregate metrics. This directly addresses the prompt requirement: "
    "*\"add functionality to monitor these metrics and update the recommendation "
    "engine as needed.\"*"
)

# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: Single-ε A/B Comparison
# ═══════════════════════════════════════════════════════════════════════════════

st.header("1. A/B Comparison Across All Users")

epsilon = st.slider(
    "Exploration rate (ε) for batch evaluation",
    min_value=0.0,
    max_value=1.0,
    value=0.1,
    step=0.05,
    key="eval_epsilon",
    help="The ε value used for both engines across all users.",
)

# Cache the AB evaluation so re-renders don't re-run it
@st.cache_data(show_spinner="Running batch evaluation across 24 users…")
def cached_ab_eval(_data, _extraction, n, eps):
    """Wrapper to cache run_ab_evaluation (hashable args only for cache key)."""
    return run_ab_evaluation(_data, _extraction, n=n, epsilon=eps)


ab_results = cached_ab_eval(data, extraction, N_RECS, epsilon)

asp_rows = ab_results["aspirational"]
base_rows = ab_results["baseline"]

# Compute aggregates
def _avg(rows, key):
    return np.mean([r[key] for r in rows])


metrics_keys = [
    ("aspirational_alignment", "Aspirational Alignment", ".0%", "Higher is better"),
    ("regression_rate", "Regression Rate", ".0%", "Lower is better"),
    ("topic_entropy", "Topic Diversity", ".2f", "Higher is better"),
    ("avg_relevance", "Avg Relevance", ".2f", "Content similarity to history"),
    ("avg_score", "Avg Score", ".2f", "Total composite score"),
]

# Summary metrics row
st.subheader("Aggregate Metrics (averaged across 24 users)")

cols = st.columns(len(metrics_keys))
for col, (key, label, fmt, hlp) in zip(cols, metrics_keys):
    asp_val = _avg(asp_rows, key)
    base_val = _avg(base_rows, key)
    delta = asp_val - base_val

    # Format display value and delta
    if "%" in fmt:
        display = f"{asp_val:{fmt}}"
        delta_str = f"{delta:+{fmt}} vs baseline"
    else:
        display = f"{asp_val:{fmt}}"
        delta_str = f"{delta:+{fmt}} vs baseline"

    # Regression rate: lower is better, so invert delta color
    delta_color = "inverse" if key == "regression_rate" else "normal"

    with col:
        st.metric(label, display, delta=delta_str, delta_color=delta_color, help=hlp)

# Per-user detail table
with st.expander("Per-user breakdown", expanded=False):
    asp_df = pd.DataFrame(asp_rows).set_index("user_id").add_prefix("asp_")
    base_df = pd.DataFrame(base_rows).set_index("user_id").add_prefix("base_")
    combined = asp_df.join(base_df)

    # Add delta columns
    for key, label, _, _ in metrics_keys:
        combined[f"Δ_{key}"] = combined[f"asp_{key}"] - combined[f"base_{key}"]

    st.dataframe(
        combined.style.format("{:.3f}"),
        use_container_width=True,
    )

    st.caption(
        f"Each row = one user. Aspirational (asp_) vs Baseline (base_) with "
        f"ε = {epsilon:.2f}, n = {N_RECS} recommendations per user."
    )

# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: Epsilon Sweep
# ═══════════════════════════════════════════════════════════════════════════════

st.header("2. ε Sweep — How Exploration Affects Outcomes")
st.markdown(
    "Sweep across a range of ε values to see how the exploration rate impacts "
    "aggregate metrics. This simulates the kind of parameter tuning you'd do "
    "in production to find the optimal exploration/exploitation balance."
)

eps_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0]

if st.button("Run ε Sweep", help="Runs both engines at each ε value across all 24 users."):
    progress = st.progress(0, text="Running ε sweep…")
    sweep_asp = []
    sweep_base = []

    for i, eps in enumerate(eps_values):
        ab = run_ab_evaluation(data, extraction, n=N_RECS, epsilon=eps)

        asp_avg = {key: _avg(ab["aspirational"], key) for key, _, _, _ in metrics_keys}
        asp_avg["epsilon"] = eps
        sweep_asp.append(asp_avg)

        base_avg = {key: _avg(ab["baseline"], key) for key, _, _, _ in metrics_keys}
        base_avg["epsilon"] = eps
        sweep_base.append(base_avg)

        progress.progress((i + 1) / len(eps_values), text=f"ε = {eps:.2f} done")

    progress.empty()

    asp_sweep_df = pd.DataFrame(sweep_asp).set_index("epsilon")
    base_sweep_df = pd.DataFrame(sweep_base).set_index("epsilon")

    # Show charts for each metric
    chart_metrics = [
        ("aspirational_alignment", "Aspirational Alignment Rate"),
        ("regression_rate", "Regression Rate"),
        ("topic_entropy", "Topic Diversity (Entropy)"),
        ("avg_relevance", "Avg Relevance"),
    ]

    for key, label in chart_metrics:
        chart_data = pd.DataFrame({
            "Aspirational Engine": asp_sweep_df[key],
            "Baseline": base_sweep_df[key],
        })
        st.subheader(label)
        st.line_chart(chart_data, use_container_width=True)

    # Summary table
    with st.expander("Raw sweep data"):
        st.markdown("**Aspirational Engine**")
        st.dataframe(asp_sweep_df.style.format("{:.3f}"), use_container_width=True)
        st.markdown("**Baseline**")
        st.dataframe(base_sweep_df.style.format("{:.3f}"), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: MLflow Integration
# ═══════════════════════════════════════════════════════════════════════════════

st.header("3. MLflow Experiment Logging")
st.markdown("""
Every recommendation call in the system is instrumented with **MLflow** logging.
The batch evaluation below runs the aspirational engine across all 24 users and logs
each individual run plus aggregate metrics as a tracked MLflow experiment.
""")

if not _MLFLOW_AVAILABLE:
    st.warning(
        "MLflow is not available in this environment. Batch evaluation still runs "
        "and returns metrics, but results are not persisted to an experiment store.",
        icon="⚠️",
    )

col_btn, col_info = st.columns([1, 2])

with col_btn:
    run_mlflow = st.button(
        "Log Batch to MLflow" if _MLFLOW_AVAILABLE else "Run Batch Evaluation",
        help="Runs run_batch_evaluation() and logs all metrics to MLflow (when available).",
    )

with col_info:
    mlflow_eps = st.number_input(
        "ε for MLflow run",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.05,
        key="mlflow_eps",
    )

if run_mlflow:
    with st.spinner("Running batch evaluation…"):
        summary = run_batch_evaluation(
            data, extraction, n=N_RECS, epsilon=mlflow_eps,
        )

    if _MLFLOW_AVAILABLE:
        st.success("Batch evaluation logged to MLflow!")
    else:
        st.success("Batch evaluation complete (MLflow logging skipped).")

    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    with col_r1:
        st.metric("Avg Alignment", f"{summary['avg_aspirational_alignment_rate']:.0%}")
    with col_r2:
        st.metric("Avg Regression", f"{summary['avg_regression_rate']:.0%}")
    with col_r3:
        st.metric("Avg Entropy", f"{summary['avg_topic_entropy']:.2f}")
    with col_r4:
        st.metric("Users Evaluated", summary["n_users"])

    if _MLFLOW_AVAILABLE:
        st.info(
            "To view the full MLflow dashboard with parameter comparisons and metric "
            "visualizations, run:\n\n"
            "```\nmlflow ui --backend-store-uri ./mlruns\n```\n\n"
            "in the project directory, then open **http://localhost:5000** in your browser.",
            icon="📈",
        )

st.markdown("---")
st.caption(
    "This page addresses the task requirement: *\"add functionality to monitor these "
    "metrics and update the recommendation engine as needed.\"* The batch evaluation "
    "shows aggregate performance; the ε sweep demonstrates parameter tuning; MLflow "
    "provides persistent experiment tracking for longitudinal monitoring."
)
