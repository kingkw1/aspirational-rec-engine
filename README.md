# Aspirational Recommendation Engine for Digital Health (Prototype)

**Live demo:** [recsys-prototype.streamlit.app](https://lore-recsys-prototype-niqqq7ndkhzgrers28qwjs.streamlit.app)

## Overview
This repository contains a functional prototype of an Aspirational Recommendation System designed for digital health platforms.

Traditional recommendation engines optimize purely for immediate engagement (clicks, likes, time-on-page), which inevitably creates "filter bubbles" that reinforce a user's current state. This system takes a different approach. It leverages a lightweight **Contextual Bandit** architecture to balance short-term historical engagement with long-term alignment to a user's *aspirational state* — inferred directly from their conversations with the AI Health Coach.

The goal is to gently nudge users toward who they want to become, without sacrificing the relevance required to keep them engaged.

## The Architecture

```
conversations.json ──→ [State Extractor] ──→ user current_state + aspirational_state
                         (NLP/Embeddings)

discussions.json ───→ [Topic Extractor] ──→ post topic vectors
                       (Embeddings)

activity.json ──────→ [Engagement Signal] ──→ historical preference vectors

                    ┌──────────────────┐
 user state ───────→│                  │
 post vectors ─────→│ Contextual Bandit│──→ ranked recommendations
 engagement ───────→│  (ε-greedy)      │
                    └──────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │  Streamlit App     │
                    │  (A/B comparison)  │
                    └───────────────────┘
```

The prototype is built as an interactive multi-page **Streamlit** app, with the Aspirational Engine as the hero and a baseline content-similarity feed for reference comparison.

### Core Components
1.  **Data Ingestion (`data_loader.py`):**
    - Parses the provided sample data (`conversations.json`, `discussions.json`, `activity.json`).
    - Normalizes records and filters flagged content (`reported_or_removed`).
    - Uses **spaCy** (`en_core_web_sm`) for entity and noun-phrase extraction on discussion posts, producing interpretable `topic_keywords` per post.

2.  **State Extraction (`state_extractor.py`):**
    - **The core ML step.** Infers each user's `current_state` and `aspirational_state` from their AI Health Coach conversations using sentence embeddings and a curated state taxonomy.
    - Built on a **Strategy pattern** with a pluggable extractor interface: the default `EmbeddingExtractor` runs a local **HuggingFace** sentence-transformer model (`all-MiniLM-L6-v2`) via **PyTorch**; a stubbed `LLMExtractor` demonstrates the production swap path to a fine-tuned or API-based model.
    - Derives topic vectors for each discussion post, merging embedding similarity scores with spaCy-extracted keywords for interpretable topic labels.
    - Only 24 of 186 users have conversation data; the remaining users fall back to engagement-only recommendations (a deliberate cold-start trade-off).
    - **Note on Data:** The conversation and discussion datasets included with this prototype are synthetically generated to demonstrate the system's architecture and are not actual user data.

3.  **The Contextual Bandit (`recommender.py`):**
    - Implements an $\epsilon$-greedy approach to balance exploitation (showing content they already like) with exploration (introducing novel, aspirational content).
    - Both the aspirational and baseline engines **exclude already-consumed posts** — only unseen content is recommended.
    - **Engagement Centroid:** Computes a mean embedding ("preference centroid") from all posts the user previously engaged with. Historical relevance is scored as the cosine similarity between each candidate post and this centroid, producing a continuous signal (typically 0.85–0.95) rather than a binary indicator.
    - **The Reward Function:**
      - **Historical Relevance (×1.0):** Cosine similarity of the post to the user's engagement centroid.
      - **Aspirational Alignment (×3.0):** The post topic aligns with the user's inferred `aspirational_state`.
      - **Regression Penalty (×-2.0):** The post reinforces negative aspects of the user's `current_state` (e.g., feeding anxiety-related content to an anxious user).
    - **Baseline Recommender:** A content-similarity engine that ranks unseen posts solely by similarity to the engagement centroid — no aspirational or regression signals. Serves as a reference for how a standard recommender would behave.
    - **MLflow Integration:** Recommendation metrics (aspirational alignment rate, regression rate, topic entropy) are logged per call, enabling the monitoring functionality requested in the problem statement.

4.  **The Presentation Layer (`app.py` + multi-page layout):**
    - A multi-page Streamlit app. The sidebar provides user selection, inferred user profile (current & aspirational states), and engagement history. Each recommendation page has its own ε-greedy exploration rate slider.
    - **Recommendations page (hero):** The Aspirational Engine feed with full score breakdowns, score decomposition charts, exploration pick labels (🎲), and metric deltas vs. baseline. Paginated (20 scored, 5 shown at a time).
    - **Conversation page:** The AI Health Coach conversation used to infer the user's states, with chat-style message bubbles.
    - **Baseline Feed page:** The content-similarity recommender for reference comparison — shows what a standard engagement-driven feed would produce, with identical pagination and exploration controls.
    - **User States page:** Per-user butterfly chart showing normalized current-struggle vs. aspirational-goal distributions across all 8 taxonomy states, and a radar chart comparing the aspirational engine's recommendation footprint against the baseline.
    - **Evaluation page:** Batch A/B comparison across all 24 users, ε sweep charts, and MLflow experiment logging.
    - **How It Works page:** Interactive explainer covering the data flow, scoring formula, state taxonomy, design decisions, and technology stack — tailored for both technical and non-technical interviewers.

## Setup & Installation

This project requires **Python 3.11** (3.11.x recommended). The spaCy model and all other dependencies are installed automatically via `requirements.txt`.

```bash
# 1. Clone the repository
git clone https://github.com/kingkw1/health-recsys-prototype.git
cd health-recsys-prototype

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# 3. Install all dependencies (includes spaCy model wheel)
pip install -r requirements.txt

# 4. Run the Streamlit application
streamlit run app.py
```

## Running Tests

To validate the mathematical behavior of the reward function and the bandit logic:

```bash
pytest test_recommender.py -v
```

## Monitoring

Recommendation metrics are logged via **MLflow** to a file-based tracking store (`./mlruns`). To view tracked experiments after running locally:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Then open `http://localhost:5000` in your browser to inspect logged runs, metrics, and parameters.

You can also trigger a batch evaluation from the **Evaluation** page in the Streamlit app, which logs per-user and aggregate metrics as MLflow runs.

> **Note:** MLflow filesystem logging is enabled when running locally. On Streamlit Community Cloud (read-only filesystem), MLflow logging is gracefully disabled and the app falls back to displaying metrics directly in the UI.

## Key Metrics to Monitor

In production, the following metrics would be tracked to evaluate whether the aspirational engine is performing as intended:

| Metric | What it measures | Target |
|---|---|---|
| **Aspirational Alignment Rate** | % of recommended posts whose topic vector is closer to aspirational_state than current_state | > 40% |
| **Engagement Retention** | CTR / read-through rate compared to a pure engagement baseline | Within 15% of baseline |
| **State Drift Velocity** | Rate at which a user's inferred current_state moves toward their aspirational_state over time | Positive trend |
| **Filter Bubble Diversity** | Topic entropy of a user's recommendation feed | Higher than baseline |
| **Regression Rate** | % of shown posts scoring a regression penalty | < 10% |

## Trade-offs & Future Architecture

Given the rapid prototyping constraint of this exercise, several architectural trade-offs were made. If this were to be scaled to production, I would recommend the following pivots:

* **From Batch to Streaming:** Currently, the system loads static JSON state. In a production environment with high-velocity conversational data, this should transition to a streaming architecture (e.g., Kafka/Flink) where the user's `current_state` vector is continuously updated in near real-time via a cybernetic feedback loop.
* **Advanced RL Models:** The $\epsilon$-greedy bandit is highly interpretable and great for a cold-start prototype. However, as the dataset scales, transitioning to a more complex Hierarchical Reinforcement Learning architecture would allow the system to optimize for delayed, multi-step sequential rewards rather than immediate single-post interactions.
* **Embeddings over Tags:** The current feature extraction uses sentence-transformer embeddings matched against a curated taxonomy. At scale, fine-tuned embeddings stored in a vector database (like Pinecone) would allow for much deeper, semantic matching without a predefined taxonomy.
* **Graph-Based Cohort Discovery:** User-post interactions naturally form a bipartite graph. Running graph embedding techniques (e.g., Node2Vec) over this structure would enable discovery of *aspirational cohorts* — clusters of users with similar growth trajectories whose successful state transitions could inform recommendations for newer users. This network-based approach enables discovery of effective health journeys.
* **Cold-Start Resolution:** The 24-of-186 user coverage limitation would be resolved as more users engage with the AI Health Coach over time, or by incorporating discussion participation text as a secondary signal for state inference.
