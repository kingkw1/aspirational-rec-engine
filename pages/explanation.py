"""
pages/explanation.py — "How It Works" explanatory page.

A narrative walkthrough of the system architecture, scoring formula, design
decisions, metrics, and monitoring strategy — aimed at both technical and
non-technical interview audiences.
"""

from __future__ import annotations

import streamlit as st

# ── Header ───────────────────────────────────────────────────────────────────

st.title("📖 How It Works")
st.markdown(
    "This page walks through the design of the **Aspirational Recommendation Engine** — "
    "how data flows through the system, how posts are scored, what trade-offs were made, "
    "and how we'd monitor and improve the engine in production."
)

# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: The Problem
# ═══════════════════════════════════════════════════════════════════════════════

st.header("1. The Problem")
st.markdown("""
Traditional recommendation engines optimize for **immediate engagement** — clicks, likes,
time-on-page. This creates *filter bubbles* that reinforce a user's **current state**, even
when that state is one they're trying to move past (anxiety, isolation, grief).

**The goal:** Recommend community discussion posts that align with **who a user wants to
become** (their *aspirational state*), not just what is entertaining or familiar right now —
while still keeping recommendations relevant enough to engage with.
""")

# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: Data Flow
# ═══════════════════════════════════════════════════════════════════════════════

st.header("2. Data Flow")

st.markdown("""
The system ingests three data sources and transforms them into recommendation-ready
signals through a two-stage pipeline:
""")

st.code("""
┌────────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                                │
├────────────────────┬─────────────────────┬─────────────────────────┤
│ conversations.json │ discussions.json    │ activity.json           │
│ 24 users × Coach │ 56 posts + comments │ 2,549 interaction logs  │
│ (personal context) │ (community content) │ (read/comment/create)   │
└────────┬───────────┴───────┬─────────────┴─────────────┬───────────┘
         │                   │                           │
         ▼                   ▼                           ▼
┌─────────────────┐ ┌─────────────────┐  ┌───────────────────────────┐
│ State Extractor │ │ Topic Extractor │  │  Engagement Index         │
│ (NLP Embeddings)│ │ (NLP Embeddings)│  │  (Activity per user)      │
│                 │ │ + spaCy NER     │  │                           │
└────────┬────────┘ └────────┬────────┘  └───────────────┬───────────┘
         │                   │                           │
         ▼                   ▼                           ▼
   User Profile         Post Profile           Engagement Centroid
   • current_state      • topic scores         • mean embedding of
   • aspirational_state • topic keywords         engaged posts
   • embedding          • embedding
         │                   │                           │
         └───────────────────┼───────────────────────────┘
                             ▼
                   ┌──────────────────┐
                   │ Contextual Bandit│
                   │   (ε-greedy)     │
                   │                  │
                   │  Score each post │
                   │  Select top-N    │
                   └────────┬─────────┘
                            ▼
                  Ranked Recommendations
                  (with score breakdowns)
""", language=None)

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown("**Stage 1: State Extraction**")
    st.markdown(
        "Each user's AI Health Coach conversation is embedded using a sentence-transformer "
        "model (`all-MiniLM-L6-v2`) and compared against a curated **state taxonomy** "
        "of 8 current states and 8 aspirational states. Scores for all 8 states per "
        "dimension are stored; the top-3 are shown in the sidebar profile and the "
        "full distribution powers the User States visualizations."
    )
with col_b:
    st.markdown("**Stage 2: Post Profiling**")
    st.markdown(
        "Each discussion post's text (+ comments) is embedded with the same model and "
        "scored against the same taxonomy. Additionally, **spaCy** extracts noun-phrase "
        "and entity keywords for interpretable topic labels."
    )
with col_c:
    st.markdown("**Stage 3: Engagement Signal**")
    st.markdown(
        "A user's read/comment/create history is aggregated. An **engagement centroid** "
        "(mean embedding of all previously engaged posts) captures their content "
        "preference as a single vector for efficient similarity scoring."
    )

# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: The State Taxonomy
# ═══════════════════════════════════════════════════════════════════════════════

st.header("3. The State Taxonomy")
st.markdown("""
The system matches users and posts against a hand-curated taxonomy of **16 states** —
8 representing what a user may be *struggling with right now* and 8 representing what they
*aspire to become*. The taxonomy was derived inductively by reading all 24 AI Health Coach
conversations and cataloguing recurring themes, then validated against two established
frameworks:

- **Self-Determination Theory (Deci & Ryan, 1985):** psychological wellbeing requires
  three core needs — *relatedness*, *competence*, and *autonomy*. The current states are
  thwarted forms of those needs; the aspirational states are their fulfilled expressions.
- **PERMA (Seligman, 2011):** flourishing consists of Positive emotion, Engagement,
  Relationships, Meaning, and Achievement. Each aspirational state maps to at least one
  PERMA pillar.

Each current state has a **natural aspirational counterpart** — the direction of change the
user is seeking. These pairs are grouped into four thematic clusters that also structure the
radar chart on the User States page.
""")

# Four clusters: each current state points toward its paired aspirational state
CLUSTERS = [
    {
        "name": "Social & Emotional",
        "sdt": "SDT: Relatedness need",
        "color": "#8e44ad",
        "pairs": [
            ("Social Isolation", "loneliness, disconnected from others",
             "Socially Connected", "community groups, friendships, belonging"),
            ("Anxiety & Overwhelm", "stress, unable to cope, fear of daily life",
             "Emotionally Resilient", "mindfulness, stress management, inner calm"),
        ],
    },
    {
        "name": "Physical Health",
        "sdt": "PERMA: Positive emotion / Vitality",
        "color": "#27ae60",
        "pairs": [
            ("Physical Health Decline", "chronic pain, mobility issues, falls",
             "Physically Active", "exercise, yoga, strength, fitness"),
            ("Work-Life Imbalance", "burnout, juggling career and family",
             "Healthy & Balanced", "recovery, eating well, sustainable pace"),
        ],
    },
    {
        "name": "Agency & Capability",
        "sdt": "SDT: Competence + Autonomy needs",
        "color": "#e67e22",
        "pairs": [
            ("Technology Frustration", "helpless with apps/devices, excluded",
             "Independent & Capable", "mastering skills, self-sufficient, confident"),
            ("Low Motivation", "stuck, lacking energy, can't start habits",
             "Adventurous & Growing", "new experiences, hiking, pushing boundaries"),
        ],
    },
    {
        "name": "Exploration & Meaning",
        "sdt": "PERMA: Meaning + Engagement",
        "color": "#2980b9",
        "pairs": [
            ("Fear & Uncertainty", "scared of health outcomes, fearful future",
             "Creative & Curious", "hobbies, pottery, cooking, lifelong learning"),
            ("Grief & Loss", "grieving a loved one, processing bereavement",
             "Purposeful & Engaged", "volunteering, mentoring, feeling valued"),
        ],
    },
]

# Build an HTML table: cluster column (rowspan=2) | current state | → | aspirational state
rows_html = ""
for cluster in CLUSTERS:
    color = cluster["color"]
    for i, (cur_label, cur_desc, asp_label, asp_desc) in enumerate(cluster["pairs"]):
        top_border = "border-top:2px solid #333" if i == 0 else "border-top:1px solid #2a2a2a"
        cluster_cell = (
            f'<td rowspan="2" style="border-left:5px solid {color};'
            f'padding:12px 16px;vertical-align:middle;min-width:148px;{top_border}">'
            f'<strong>{cluster["name"]}</strong><br>'
            f'<span style="font-size:0.75rem;color:#888">{cluster["sdt"]}</span></td>'
            if i == 0 else ""
        )
        rows_html += (
            f'<tr style="{top_border}">'
            f"{cluster_cell}"
            f'<td style="padding:10px 14px;vertical-align:top">'
            f'<span style="color:#e74c3c;font-weight:600">{cur_label}</span><br>'
            f'<span style="font-size:0.8rem;color:#888">{cur_desc}</span></td>'
            f'<td style="padding:10px 10px;text-align:center;vertical-align:middle;'
            f'font-size:1.3rem;color:#555">→</td>'
            f'<td style="padding:10px 14px;vertical-align:top">'
            f'<span style="color:#2ecc71;font-weight:600">{asp_label}</span><br>'
            f'<span style="font-size:0.8rem;color:#888">{asp_desc}</span></td>'
            f"</tr>"
        )

table_html = (
    '<table style="width:100%;border-collapse:collapse;margin-bottom:0.5rem">'
    "<thead><tr>"
    '<th style="padding:8px 16px;text-align:left;color:#888;font-weight:500;'
    'border-bottom:2px solid #444;min-width:148px">Cluster</th>'
    '<th style="padding:8px 14px;text-align:left;color:#e74c3c;font-weight:500;'
    'border-bottom:2px solid #444">Current State '
    '<em style="font-weight:400;color:#888">(struggling with)</em></th>'
    '<th style="border-bottom:2px solid #444"></th>'
    '<th style="padding:8px 14px;text-align:left;color:#2ecc71;font-weight:500;'
    'border-bottom:2px solid #444">Aspirational State '
    '<em style="font-weight:400;color:#888">(wants to become)</em></th>'
    "</tr></thead>"
    f"<tbody>{rows_html}</tbody>"
    "</table>"
)

st.markdown(table_html, unsafe_allow_html=True)

st.caption(
    "All 8 cosine-similarity scores per dimension are computed and stored. "
    "The top-3 are shown in the sidebar profile; the full distribution drives "
    "the butterfly chart and recommendation alignment radar on the User States page."
)

# ═══════════════════════════════════════════════════════════════════════════════
# Section 4: The Reward Function
# ═══════════════════════════════════════════════════════════════════════════════

st.header("4. The Reward Function")
st.markdown("""
For each candidate post, the engine computes a **three-component score** that balances
relevance, aspiration, and safety:
""")

st.latex(r"""
\text{Score}(u, p) = \underbrace{1.0 \times \cos(\mathbf{p},\, \mathbf{c}_u)}_{\text{Historical Relevance}}
\;+\; \underbrace{3.0 \times s_{\text{asp}}}_{\text{Aspirational Alignment}}
\;+\; \underbrace{(-2.0) \times s_{\text{reg}}}_{\text{Regression Penalty}}
""")

st.markdown("Where:")
col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    st.markdown(
        "**Historical Relevance (×1.0)**\n\n"
        "Cosine similarity between the post embedding and the user's "
        "**engagement centroid** (mean of all previously engaged posts). "
        "Keeps recommendations relevant to demonstrated interests.\n\n"
        "*Typical range: 0.85 – 0.95*"
    )

with col_f2:
    st.markdown(
        "**Aspirational Alignment (×3.0)**\n\n"
        "Fires when the post's topic score on the user's **top aspirational state** "
        "exceeds a threshold of **0.20**. The 3× multiplier ensures aspirational "
        "content can outrank merely familiar content.\n\n"
        "*Threshold: 0.20 · Weight: 3.0*"
    )

with col_f3:
    st.markdown(
        "**Regression Penalty (×-2.0)**\n\n"
        "Fires when the post's topic score on the user's **top current (negative) state** "
        "exceeds a threshold of **0.35**. Penalizes content that reinforces the "
        "filter bubble. Higher threshold = only strong matches penalized.\n\n"
        "*Threshold: 0.35 · Weight: -2.0*"
    )

st.info(
    "**Why these weights?** The 3:1:2 ratio is an empirical starting point. "
    "The aspirational weight (3.0) must be large enough to overcome the historical "
    "relevance baseline (~0.9), so aspirational posts can rank first. The regression "
    "penalty (2.0) is strong but has a higher threshold (0.35 vs 0.20) so only "
    "strongly negative matches are suppressed — we don't want to over-censor. "
    "In production, these weights would be tuned via A/B testing on engagement + "
    "state-drift outcomes.",
    icon="⚖️",
)

# ═══════════════════════════════════════════════════════════════════════════════
# Section 5: Exploration vs Exploitation
# ═══════════════════════════════════════════════════════════════════════════════

st.header("5. Exploration vs. Exploitation (ε-Greedy)")
st.markdown("""
The engine uses an **ε-greedy contextual bandit** strategy:

- With probability **(1 - ε)**: pick the highest-scoring post (**exploit** — show the best recommendation).
- With probability **ε**: pick a random unseen post (**explore** — introduce diversity and discover new alignment signals).

Exploration posts are marked with 🎲 in the feed. Both the aspirational engine and the
baseline use **identical ε-greedy mechanics**, so when comparing the two feeds, the only
variable is the aspirational/regression signal — not the presence or absence of exploration.

The ε slider on each recommendation page lets you adjust this live. At ε = 0, you see pure
exploitation (strict ranking). At ε = 1, the feed is fully randomized.
""")

# ═══════════════════════════════════════════════════════════════════════════════
# Section 6: Baseline Comparison
# ═══════════════════════════════════════════════════════════════════════════════

st.header("6. Baseline Comparison (A/B Design)")
st.markdown("""
The **Baseline Feed** uses a standard content-similarity recommender:

| | Aspirational Engine | Baseline |
|---|---|---|
| **Historical Relevance** | ✅ Cosine sim to engagement centroid | ✅ Same |
| **Aspirational Alignment** | ✅ +3.0 × topic alignment | ❌ Not used |
| **Regression Penalty** | ✅ -2.0 × negative state match | ❌ Not used |
| **ε-Greedy Exploration** | ✅ Same ε | ✅ Same ε |
| **Excludes Consumed Posts** | ✅ | ✅ |

The baseline ranks posts **solely by similarity to the engagement centroid** — "you liked X,
here's more like X." Aspirational and regression metrics are computed *retroactively* on
baseline results so we can compare the two engines on a level playing field.

This isolates the aspirational signal: any improvement in aspirational alignment or
reduction in regression rate on the Recommendations page is directly attributable to the
aspirational reward shaping.
""")

# ═══════════════════════════════════════════════════════════════════════════════
# Section 7: Metrics
# ═══════════════════════════════════════════════════════════════════════════════

st.header("7. Key Metrics")
st.markdown("The system tracks five metrics — three are computed live in the prototype, two are production targets:")

st.subheader("Computed in Prototype")
col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.markdown(
        "**Aspirational Alignment Rate**\n\n"
        "% of recommended posts with a positive aspirational score "
        "(topic matches the user's aspirational state above the 0.20 threshold).\n\n"
        "*Target: > 40%*"
    )
with col_m2:
    st.markdown(
        "**Regression Rate**\n\n"
        "% of recommended posts that fire the regression penalty "
        "(reinforce the user's current negative state).\n\n"
        "*Target: < 10%*"
    )
with col_m3:
    st.markdown(
        "**Topic Diversity (Shannon Entropy)**\n\n"
        "Entropy of aspirational topic labels across the batch. "
        "Higher = more diverse topic mix, less chance of a new filter bubble.\n\n"
        "*Target: Higher than baseline*"
    )

st.subheader("Production Metrics (require live user interaction data)")
col_m4, col_m5 = st.columns(2)
with col_m4:
    st.markdown(
        "**Engagement Retention**\n\n"
        "Click-through rate and read-through rate compared to the engagement baseline. "
        "If we're nudging users toward aspirational content but they stop clicking, "
        "the nudge is too aggressive. We need to stay within ~15% of baseline engagement "
        "to confirm the aspirational content is still *relevant*, not just *virtuous*.\n\n"
        "*Target: Within 15% of baseline CTR*"
    )
with col_m5:
    st.markdown(
        "**State Drift Velocity**\n\n"
        "Over time, does the user's inferred `current_state` move toward their "
        "`aspirational_state`? We'd re-run state extraction periodically (e.g., weekly) "
        "and measure the cosine distance between the two embeddings. A shrinking gap "
        "means the user is *actually becoming who they want to be* — the ultimate "
        "success metric for this system.\n\n"
        "*Target: Positive trend (gap decreasing over time)*"
    )

# ═══════════════════════════════════════════════════════════════════════════════
# Section 8: Monitoring & Updating the Engine
# ═══════════════════════════════════════════════════════════════════════════════

st.header("8. Monitoring & Updating the Engine")
st.markdown("""
The task prompt asks: *"add some functionality to monitor these metrics and update the
recommendation engine as needed."* Here's how the system addresses this at each level:
""")

st.subheader("What's Built Now")
st.markdown("""
- **MLflow logging:** Every recommendation call logs parameters (`user_id`, `ε`, `engine_type`)
  and metrics (alignment rate, regression rate, entropy, avg score) to MLflow.
  Run `mlflow ui` to inspect experiments.
- **Batch evaluation:** `run_batch_evaluation()` runs the engine across all 24 profiled users
  and logs aggregate metrics as a single MLflow experiment. This lets you compare
  different ε values or weight configurations across the full user population.
- **Live A/B comparison:** The Recommendations vs. Baseline pages show metric deltas
  in real time — if you change ε, you immediately see how alignment and regression shift.
""")

st.subheader("The Cybernetic Feedback Loop (Production Architecture)")
st.markdown("""
In production, the system would operate as a closed-loop feedback cycle:
""")

st.code("""
┌────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐    ┌────────────────┐
│ Recommend  │───→│ User Engages │───→│ New Activity│───→│ Re-extract   │───→│ Update Centroid│
│ (ε-greedy) │    │ (or ignores) │    │ Data        │    │ User State   │    │ + Detect Drift │
└──────┬─────┘    └──────────────┘    └─────────────┘    │ (did they    │    │ + Tune Weights │
       ▲                                                 │  move?)      │    └───────┬────────┘
       │                                                 └──────────────┘            │
       │                                                                             │
       └─────────────────────────────────────────────────────────────────────────────┘
""", language=None)

st.markdown("""
**Step 1 — Recommend:** The engine scores unseen posts and presents a feed.

**Step 2 — Observe:** The user reads, comments, or ignores the recommendations.
New interaction data flows into the activity log.

**Step 3 — Re-extract State:** Periodically (daily/weekly), the state extractor re-runs
on updated conversations. The user's `current_state` and `aspirational_state` vectors
may shift — this is the **state drift** signal.

**Step 4 — Update the Engine:**
- The **engagement centroid** automatically shifts as new activity data arrives (no retraining needed).
- **Weight tuning:** If batch evaluation shows alignment is too low or regression too high,
  the reward weights (3.0 / -2.0) and thresholds (0.20 / 0.35) can be adjusted. In production,
  this could be done via Bayesian optimization or a bandit over weight configurations.
- **ε decay:** As confidence grows in a user's profile, ε can be decreased (less exploration needed).
  New users or users with shifting states would get higher ε.

This is the **cybernetic loop** — the system doesn't just recommend, it *observes the effect
of its recommendations and adapts*. The same architecture used by contextual bandit
nudge services.
""")

# ═══════════════════════════════════════════════════════════════════════════════
# Section 9: Design Decisions & Trade-offs
# ═══════════════════════════════════════════════════════════════════════════════

st.header("9. Design Decisions & Trade-offs")

decisions = [
    (
        "Why ε-greedy over Thompson Sampling or UCB?",
        "ε-greedy is the simplest contextual bandit strategy — highly interpretable, "
        "easy to explain, and appropriate for a cold-start prototype with 56 posts. "
        "Thompson Sampling or UCB would be natural upgrades as the action space grows, "
        "but they require maintaining posterior distributions per arm, which adds "
        "complexity without proportional value at this scale."
    ),
    (
        "Why a curated taxonomy instead of unsupervised clustering?",
        "With only 24 users and 56 posts, unsupervised methods (topic modeling, clustering) "
        "would produce noisy, uninterpretable results. A hand-curated taxonomy gives us "
        "human-readable labels and domain-aligned categories that make the system's "
        "reasoning transparent — critical for healthcare where decisions must be explainable."
    ),
    (
        "Why sentence-transformers instead of a fine-tuned model?",
        "all-MiniLM-L6-v2 is a general-purpose embedding model. It works well enough "
        "for semantic similarity on health/wellness text. Fine-tuning on domain-specific conversation "
        "data would improve accuracy but requires labeled data (which state is 'correct'?) "
        "and more compute time than a rapid prototype allows. The Strategy pattern in "
        "state_extractor.py makes this a swap-in upgrade."
    ),
    (
        "Why exclude already-consumed posts?",
        "In a community with only 56 posts, re-recommending content the user already read "
        "wastes the limited recommendation surface. This forces the engine to find the "
        "best *unseen* aspirational content. In a larger corpus, we might allow re-surfacing "
        "at a discount."
    ),
    (
        "Why 24-of-186 user coverage?",
        "Only 24 users have AI Health Coach conversations — the sole source for inferring "
        "aspirational state. The remaining 162 users fall back to engagement-only "
        "scoring (the centroid-similarity baseline). This is a deliberate cold-start "
        "trade-off: as more users chat with the AI Health Coach, coverage expands automatically."
    ),
]

for question, answer in decisions:
    with st.expander(question):
        st.markdown(answer)

# ═══════════════════════════════════════════════════════════════════════════════
# Section 10: Technology Stack
# ═══════════════════════════════════════════════════════════════════════════════

st.header("10. Technology Stack")

st.markdown("""
| Layer | Technology | Why |
|-------|-----------|-----|
| **Embeddings** | `sentence-transformers` (`all-MiniLM-L6-v2`) via HuggingFace/PyTorch | Fast, local, no API keys. 384-dim embeddings. |
| **NLP / NER** | spaCy (`en_core_web_sm`) | Noun-phrase and entity extraction for interpretable topic keywords. |
| **Similarity** | scikit-learn (`cosine_similarity`) | Standard, vectorized cosine similarity computation. |
| **Monitoring** | MLflow | Parameter + metric logging per recommendation call and batch evaluation. Gracefully disabled on Streamlit Cloud. |
| **Visualization** | Plotly | Butterfly chart (state distributions) and radar chart (recommendation alignment) on the User States page. |
| **UI** | Streamlit | Rapid interactive prototyping with multi-page layout and reactive widgets. Deployed on Streamlit Community Cloud. |
| **Testing** | pytest (23 tests) | Unit tests on reward math + integration test on full pipeline. |
| **Architecture** | Strategy pattern (extractor), dataclasses, functional scoring | Clean separation of concerns; extractor backend is swappable. |
""")