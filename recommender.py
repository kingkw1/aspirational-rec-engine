"""
recommender.py — Contextual Bandit recommendation engine for the Aspirational System.

Implements:
  - A 3-component reward function: historical relevance (content similarity to engagement centroid),
    aspirational alignment (+3×sim), regression penalty (-2×sim).
  - Epsilon-greedy action selection for explore/exploit balance.
  - A baseline (content-similarity) recommender for A/B comparison.
  - Already-consumed content is excluded from recommendations.
  - MLflow metric logging for monitoring.
"""

from __future__ import annotations

import pathlib
import random
from dataclasses import dataclass, field
from math import log2
from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from data_loader import AppData, UserActivityIndex
from state_extractor import ExtractionResult, PostProfile, UserProfile

# MLflow is optional — it requires a writable filesystem which isn't available
# on Streamlit Community Cloud.  All MLflow-dependent functions degrade gracefully.
try:
    import mlflow

    _MLFLOW_AVAILABLE = True
except Exception:
    # Catching Exception instead of just ImportError because dependency issues
    # (like protobuf version mismatch) can raise TypeError/AttributeError during import.
    _MLFLOW_AVAILABLE = False

# ── Reward Weights ───────────────────────────────────────────────────────────

HISTORICAL_ENGAGEMENT_REWARD = 1.0
ASPIRATIONAL_ALIGNMENT_REWARD = 3.0
REGRESSION_PENALTY = -2.0

# Cosine similarity thresholds for triggering aspirational/regression scores.
# These control how aggressively we reward alignment or penalize regression.
# Calibrated against actual post score distributions: most posts cluster
# 0.15-0.45 on aspirational topics, so 0.20 captures meaningful alignment
# while 0.35 restricts regression to strong negative matches.
ASPIRATIONAL_THRESHOLD = 0.20
REGRESSION_THRESHOLD = 0.35


# ── Dataclasses ──────────────────────────────────────────────────────────────


@dataclass
class ScoreBreakdown:
    """Transparent decomposition of a recommendation score."""
    historical: float = 0.0
    aspirational: float = 0.0
    regression: float = 0.0

    @property
    def total(self) -> float:
        return self.historical + self.aspirational + self.regression


@dataclass
class ScoredPost:
    """A discussion post with its recommendation score and breakdown."""
    post_id: int
    score: float
    breakdown: ScoreBreakdown
    post_profile: Optional[PostProfile] = None
    is_exploration: bool = False

    @property
    def snippet(self) -> str:
        """Short preview — delegated to PostProfile if available."""
        return ""


@dataclass
class RecommendationResult:
    """Output of a recommendation call."""
    user_id: int
    recommendations: list[ScoredPost]
    engine_type: str  # "aspirational" or "baseline"
    epsilon: float = 0.0

    @property
    def metrics(self) -> RecommendationMetrics:
        return compute_metrics(self)


@dataclass
class RecommendationMetrics:
    """Aggregate metrics for a recommendation batch."""
    aspirational_alignment_rate: float
    regression_rate: float
    topic_entropy: float
    avg_score: float
    avg_relevance: float
    n_recommendations: int


# ── Engagement Centroid ──────────────────────────────────────────────────────


def compute_engagement_centroid(
    engaged_posts: set[int],
    post_profiles: dict[int, PostProfile],
) -> Optional[np.ndarray]:
    """Compute mean embedding of posts the user engaged with.

    Returns None if the user has no engagement or no matching post profiles.
    """
    embeddings = [
        post_profiles[pid].embedding
        for pid in engaged_posts
        if pid in post_profiles
    ]
    if not embeddings:
        return None
    return np.mean(embeddings, axis=0).astype(np.float32)


# ── Scoring Function ─────────────────────────────────────────────────────────


def score_post(
    user_profile: Optional[UserProfile],
    post_profile: PostProfile,
    user_activity: Optional[UserActivityIndex],
    engagement_centroid: Optional[np.ndarray] = None,
    historical_weight: float = HISTORICAL_ENGAGEMENT_REWARD,
    aspirational_weight: float = ASPIRATIONAL_ALIGNMENT_REWARD,
    regression_weight: float = REGRESSION_PENALTY,
    aspirational_threshold: float = ASPIRATIONAL_THRESHOLD,
    regression_threshold: float = REGRESSION_THRESHOLD,
) -> ScoreBreakdown:
    """Score a single (user, post) pair across three reward components.

    Args:
        user_profile: Inferred user state (None for activity-only users).
        post_profile: Inferred post topic profile.
        user_activity: User's historical engagement records.
        engagement_centroid: Mean embedding of user's engaged posts. When provided,
            historical score is continuous (similarity to centroid) rather than binary.
        *_weight: Reward weights for each component.
        *_threshold: Cosine similarity thresholds.

    Returns:
        ScoreBreakdown with historical, aspirational, and regression components.
    """
    breakdown = ScoreBreakdown()

    # ── Component 1: Historical Relevance ────────────────────────────────
    # When a centroid is provided, score = weight × cosine similarity to
    # the engagement centroid.  Otherwise fall back to binary membership
    # (backward compat for unit tests that call score_post directly).
    if engagement_centroid is not None:
        sim = cosine_similarity(
            post_profile.embedding.reshape(1, -1),
            engagement_centroid.reshape(1, -1),
        )[0][0]
        if sim > 0:
            breakdown.historical = historical_weight * float(sim)
    elif user_activity and post_profile.post_id in user_activity.all_engaged_posts:
        breakdown.historical = historical_weight

    # ── Components 2 & 3 require a user profile (conversation data) ──────
    if user_profile is not None and user_profile.embedding is not None:
        user_emb = user_profile.embedding.reshape(1, -1)
        post_emb = post_profile.embedding.reshape(1, -1)

        # Aspirational: does this post align with who the user wants to become?
        # Compare post embedding against the user's aspirational state embedding direction.
        if user_profile.aspirational_states:
            # Use the top aspirational state's score as reference,
            # but the real signal is: how similar is this post to aspirational content?
            top_aspirational_label = user_profile.top_aspirational.label
            for asp_score in post_profile.aspirational_state_scores:
                if asp_score.label == top_aspirational_label and asp_score.score >= aspirational_threshold:
                    breakdown.aspirational = aspirational_weight * asp_score.score
                    break

        # Regression: does this post reinforce the user's current negative state?
        if user_profile.current_states:
            top_current_label = user_profile.top_current.label
            for cur_score in post_profile.current_state_scores:
                if cur_score.label == top_current_label and cur_score.score >= regression_threshold:
                    breakdown.regression = regression_weight * cur_score.score
                    break

    return breakdown


# ── Recommender Engines ──────────────────────────────────────────────────────


def recommend_aspirational(
    user_id: int,
    data: AppData,
    extraction: ExtractionResult,
    n: int = 10,
    epsilon: float = 0.1,
    seed: Optional[int] = None,
) -> RecommendationResult:
    """Aspirational recommendation engine (epsilon-greedy contextual bandit).

    Args:
        user_id: Target user.
        data: Loaded data container.
        extraction: Extraction result with user/post profiles.
        n: Number of posts to recommend.
        epsilon: Exploration rate (0 = pure exploitation, 1 = random).
        seed: Random seed for reproducibility.
    """
    rng = random.Random(seed)

    user_profile = extraction.user_profiles.get(user_id)
    user_activity = data.user_activity_index.get(user_id)
    engaged_posts = user_activity.all_engaged_posts if user_activity else set()
    centroid = compute_engagement_centroid(engaged_posts, extraction.post_profiles)

    # Score unseen posts only — already-consumed content is excluded
    scored: list[ScoredPost] = []
    for post_id, post_profile in extraction.post_profiles.items():
        if post_id in engaged_posts:
            continue
        breakdown = score_post(
            user_profile, post_profile, user_activity,
            engagement_centroid=centroid,
        )
        scored.append(ScoredPost(
            post_id=post_id,
            score=breakdown.total,
            breakdown=breakdown,
            post_profile=post_profile,
        ))

    # Epsilon-greedy selection
    scored_sorted = sorted(scored, key=lambda s: s.score, reverse=True)
    selected: list[ScoredPost] = []
    remaining = list(scored_sorted)

    for _ in range(min(n, len(remaining))):
        if not remaining:
            break
        if rng.random() < epsilon:
            # Explore: pick a random post from remaining
            choice = rng.choice(remaining)
            choice.is_exploration = True
        else:
            # Exploit: pick the highest-scoring remaining post
            choice = remaining[0]
        selected.append(choice)
        remaining.remove(choice)
        # Re-sort remaining (only needed if we removed from middle)
        remaining.sort(key=lambda s: s.score, reverse=True)

    return RecommendationResult(
        user_id=user_id,
        recommendations=selected,
        engine_type="aspirational",
        epsilon=epsilon,
    )


def recommend_baseline(
    user_id: int,
    data: AppData,
    extraction: ExtractionResult,
    n: int = 10,
    epsilon: float = 0.1,
    seed: Optional[int] = None,
) -> RecommendationResult:
    """Baseline content-similarity recommender (no aspirational signal).

    Computes a preference centroid from posts the user previously engaged with,
    then ranks *unseen* posts by cosine similarity to that centroid.  This models
    a standard engagement-driven recommender: "you liked X, here's similar Y."

    Uses the same epsilon-greedy exploration as the aspirational engine so the
    only variable in an A/B comparison is the aspirational/regression signal,
    not the presence or absence of exploration.
    """
    rng = random.Random(seed)

    user_activity = data.user_activity_index.get(user_id)
    engaged_posts = user_activity.all_engaged_posts if user_activity else set()
    centroid = compute_engagement_centroid(engaged_posts, extraction.post_profiles)

    scored: list[ScoredPost] = []
    for post_id, post_profile in extraction.post_profiles.items():
        if post_id in engaged_posts:
            continue
        # user_profile=None → no aspirational/regression components
        breakdown = score_post(None, post_profile, None, engagement_centroid=centroid)
        scored.append(ScoredPost(
            post_id=post_id,
            score=breakdown.total,
            breakdown=breakdown,
            post_profile=post_profile,
        ))

    # Epsilon-greedy selection (same mechanism as aspirational engine)
    scored_sorted = sorted(scored, key=lambda s: s.score, reverse=True)
    selected: list[ScoredPost] = []
    remaining = list(scored_sorted)

    for _ in range(min(n, len(remaining))):
        if not remaining:
            break
        if rng.random() < epsilon:
            choice = rng.choice(remaining)
            choice.is_exploration = True
        else:
            choice = remaining[0]
        selected.append(choice)
        remaining.remove(choice)
        remaining.sort(key=lambda s: s.score, reverse=True)

    # Retroactively compute full breakdowns (with aspirational/regression) so
    # that metrics reflect the actual characteristics of the selected posts.
    # The *ranking* was based on content-similarity only; these components are
    # added purely for measurement and comparison purposes.
    user_profile = extraction.user_profiles.get(user_id)
    if user_profile is not None:
        for sp in selected:
            full = score_post(
                user_profile, sp.post_profile, user_activity,
                engagement_centroid=centroid,
            )
            sp.breakdown = full
            # Keep sp.score as original content-similarity ranking score

    return RecommendationResult(
        user_id=user_id,
        recommendations=selected,
        engine_type="baseline",
        epsilon=epsilon,
    )


# ── Metrics ──────────────────────────────────────────────────────────────────


def compute_metrics(result: RecommendationResult) -> RecommendationMetrics:
    """Compute aggregate metrics for a set of recommendations."""
    recs = result.recommendations
    if not recs:
        return RecommendationMetrics(
            aspirational_alignment_rate=0.0,
            regression_rate=0.0,
            topic_entropy=0.0,
            avg_score=0.0,
            avg_relevance=0.0,
            n_recommendations=0,
        )

    n = len(recs)

    # Aspirational alignment rate: % of recs with positive aspirational score
    aspirational_count = sum(1 for r in recs if r.breakdown.aspirational > 0)
    aspirational_rate = aspirational_count / n

    # Regression rate: % of recs with non-zero regression penalty
    regression_count = sum(1 for r in recs if r.breakdown.regression < 0)
    regression_rate = regression_count / n

    # Topic entropy: diversity of top aspirational topics in the batch
    topic_counts: dict[str, int] = {}
    for r in recs:
        if r.post_profile and r.post_profile.aspirational_state_scores:
            label = r.post_profile.top_aspirational_topic.label
            topic_counts[label] = topic_counts.get(label, 0) + 1
    if topic_counts:
        total = sum(topic_counts.values())
        probs = [c / total for c in topic_counts.values()]
        topic_entropy = -sum(p * log2(p) for p in probs if p > 0)
    else:
        topic_entropy = 0.0

    avg_score = sum(r.score for r in recs) / n
    avg_relevance = sum(r.breakdown.historical for r in recs) / n

    return RecommendationMetrics(
        aspirational_alignment_rate=aspirational_rate,
        regression_rate=regression_rate,
        topic_entropy=topic_entropy,
        avg_score=avg_score,
        avg_relevance=avg_relevance,
        n_recommendations=n,
    )


# ── MLflow Logging ───────────────────────────────────────────────────────────

# Resolve the tracking URI once so all MLflow calls point at the same store
# that `mlflow ui` reads (the file-based ./mlruns directory).

_MLRUNS_DIR = pathlib.Path(__file__).resolve().parent / "mlruns"
_TRACKING_URI = _MLRUNS_DIR.as_uri()  # file:///...absolute.../mlruns


def _ensure_mlflow_tracking() -> None:
    """Set the MLflow tracking URI to the project-local mlruns/ directory.

    No-op when MLflow is not available or the filesystem is read-only.
    """
    if not _MLFLOW_AVAILABLE:
        return
    try:
        mlflow.set_tracking_uri(_TRACKING_URI)
    except Exception:
        pass


def log_recommendation_run(
    result: RecommendationResult,
    experiment_name: str = "health-recsys",
) -> None:
    """Log a recommendation result to MLflow.

    Logs parameters (user_id, epsilon, engine_type) and metrics
    (aspirational alignment rate, regression rate, topic entropy, avg score).
    Silently skipped when MLflow is unavailable or the filesystem is read-only.
    """
    if not _MLFLOW_AVAILABLE:
        return
    try:
        _ensure_mlflow_tracking()
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(nested=True):
            mlflow.log_param("user_id", result.user_id)
            mlflow.log_param("engine_type", result.engine_type)
            mlflow.log_param("epsilon", result.epsilon)
            mlflow.log_param("n_recommendations", len(result.recommendations))

            metrics = result.metrics
            mlflow.log_metric("aspirational_alignment_rate", metrics.aspirational_alignment_rate)
            mlflow.log_metric("regression_rate", metrics.regression_rate)
            mlflow.log_metric("topic_entropy", metrics.topic_entropy)
            mlflow.log_metric("avg_score", metrics.avg_score)
    except Exception:
        pass  # read-only filesystem or other cloud restriction


def run_batch_evaluation(
    data: AppData,
    extraction: ExtractionResult,
    n: int = 10,
    epsilon: float = 0.1,
    experiment_name: str = "health-recsys-batch",
) -> dict[str, float]:
    """Run the recommender across all profiled users and log aggregate metrics.

    Returns a summary dict of averaged metrics across all users.
    MLflow logging is best-effort; the summary is always returned.
    """
    all_metrics: list[RecommendationMetrics] = []

    # Attempt MLflow-tracked run; fall back to plain computation
    _mlflow_ok = _MLFLOW_AVAILABLE
    if _mlflow_ok:
        try:
            _ensure_mlflow_tracking()
            mlflow.set_experiment(experiment_name)
            parent_run = mlflow.start_run(run_name="batch_evaluation")
            mlflow.log_param("epsilon", epsilon)
            mlflow.log_param("n_per_user", n)
            mlflow.log_param("n_profiled_users", len(extraction.user_profiles))
        except Exception:
            _mlflow_ok = False

    for user_id in extraction.user_profiles:
        asp_result = recommend_aspirational(
            user_id, data, extraction, n=n, epsilon=epsilon
        )
        all_metrics.append(asp_result.metrics)
        if _mlflow_ok:
            log_recommendation_run(asp_result, experiment_name)

    avg_alignment = np.mean([m.aspirational_alignment_rate for m in all_metrics])
    avg_regression = np.mean([m.regression_rate for m in all_metrics])
    avg_entropy = np.mean([m.topic_entropy for m in all_metrics])
    avg_score = np.mean([m.avg_score for m in all_metrics])

    if _mlflow_ok:
        try:
            mlflow.log_metric("avg_aspirational_alignment_rate", float(avg_alignment))
            mlflow.log_metric("avg_regression_rate", float(avg_regression))
            mlflow.log_metric("avg_topic_entropy", float(avg_entropy))
            mlflow.log_metric("avg_score", float(avg_score))
            mlflow.end_run()
        except Exception:
            pass

    summary = {
        "avg_aspirational_alignment_rate": float(avg_alignment),
        "avg_regression_rate": float(avg_regression),
        "avg_topic_entropy": float(avg_entropy),
        "avg_score": float(avg_score),
        "n_users": len(all_metrics),
    }
    return summary


def run_ab_evaluation(
    data: AppData,
    extraction: ExtractionResult,
    n: int = 20,
    epsilon: float = 0.1,
) -> dict[str, list[dict[str, float]]]:
    """Run both engines across all profiled users and return per-user metrics.

    Returns a dict with keys "aspirational" and "baseline", each containing
    a list of per-user metric dicts (with a "user_id" key).
    """
    asp_rows: list[dict[str, float]] = []
    base_rows: list[dict[str, float]] = []

    for user_id in sorted(extraction.user_profiles.keys()):
        asp_result = recommend_aspirational(
            user_id, data, extraction, n=n, epsilon=epsilon, seed=42,
        )
        base_result = recommend_baseline(
            user_id, data, extraction, n=n, epsilon=epsilon, seed=42,
        )
        m_asp = asp_result.metrics
        m_base = base_result.metrics

        asp_rows.append({
            "user_id": user_id,
            "aspirational_alignment": m_asp.aspirational_alignment_rate,
            "regression_rate": m_asp.regression_rate,
            "topic_entropy": m_asp.topic_entropy,
            "avg_relevance": m_asp.avg_relevance,
            "avg_score": m_asp.avg_score,
        })
        base_rows.append({
            "user_id": user_id,
            "aspirational_alignment": m_base.aspirational_alignment_rate,
            "regression_rate": m_base.regression_rate,
            "topic_entropy": m_base.topic_entropy,
            "avg_relevance": m_base.avg_relevance,
            "avg_score": m_base.avg_score,
        })

    return {"aspirational": asp_rows, "baseline": base_rows}


# ── CLI Smoke Test ───────────────────────────────────────────────────────────


if __name__ == "__main__":
    from data_loader import load_all
    from state_extractor import extract_all

    print("Loading data...")
    data = load_all()

    print("Extracting states...")
    extraction = extract_all(data)

    print(f"\n{'='*60}")
    print("RECOMMENDATION ENGINE SMOKE TEST")
    print(f"{'='*60}")

    # Pick a user with rich data
    test_user = 221

    print(f"\n--- User {test_user} ---")
    profile = extraction.user_profiles[test_user]
    print(f"Current:      {profile.top_current.label} ({profile.top_current.score:.3f})")
    print(f"Aspirational: {profile.top_aspirational.label} ({profile.top_aspirational.score:.3f})")

    # Aspirational engine
    asp_result = recommend_aspirational(test_user, data, extraction, n=5, epsilon=0.0, seed=42)
    print(f"\n  Aspirational Engine (ε=0.0, top 5):")
    for r in asp_result.recommendations:
        b = r.breakdown
        print(f"    Post {r.post_id:5d} | total={r.score:+.3f} | hist={b.historical:+.1f} asp={b.aspirational:+.3f} reg={b.regression:+.3f}")

    metrics = asp_result.metrics
    print(f"\n  Metrics: align_rate={metrics.aspirational_alignment_rate:.2f}, "
          f"reg_rate={metrics.regression_rate:.2f}, "
          f"entropy={metrics.topic_entropy:.3f}, "
          f"avg_score={metrics.avg_score:.3f}")

    # Baseline engine
    base_result = recommend_baseline(test_user, data, extraction, n=5)
    print(f"\n  Baseline Engine (top 5):")
    for r in base_result.recommendations:
        b = r.breakdown
        print(f"    Post {r.post_id:5d} | total={r.score:+.3f} | hist={b.historical:+.1f} asp={b.aspirational:+.3f} reg={b.regression:+.3f}")

    base_metrics = base_result.metrics
    print(f"\n  Metrics: align_rate={base_metrics.aspirational_alignment_rate:.2f}, "
          f"reg_rate={base_metrics.regression_rate:.2f}, "
          f"entropy={base_metrics.topic_entropy:.3f}, "
          f"avg_score={base_metrics.avg_score:.3f}")

    # A/B comparison
    print(f"\n--- A/B Comparison ---")
    print(f"  Aspirational alignment: {metrics.aspirational_alignment_rate:.0%} vs {base_metrics.aspirational_alignment_rate:.0%} (baseline)")
    print(f"  Regression rate:        {metrics.regression_rate:.0%} vs {base_metrics.regression_rate:.0%} (baseline)")

    print(f"\n{'='*60}")
    print("Phase 3 complete. Recommender engine operational.")
    print(f"{'='*60}")
