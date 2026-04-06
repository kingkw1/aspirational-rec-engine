"""
test_recommender.py — Unit and integration tests for the Aspirational Recommendation Engine.

Unit tests use synthetic fixtures to mathematically prove the reward function behaves correctly.
Integration test runs the full pipeline end-to-end against real data.

Run with: pytest test_recommender.py -v
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from data_loader import UserActivityIndex
from recommender import (
    ASPIRATIONAL_ALIGNMENT_REWARD,
    ASPIRATIONAL_THRESHOLD,
    HISTORICAL_ENGAGEMENT_REWARD,
    REGRESSION_PENALTY,
    REGRESSION_THRESHOLD,
    ScoreBreakdown,
    compute_engagement_centroid,
    compute_metrics,
    recommend_aspirational,
    recommend_baseline,
    score_post,
)
from state_extractor import PostProfile, StateScore, UserProfile


# ── Helpers: Synthetic Fixtures ──────────────────────────────────────────────


def make_user_profile(
    user_id: int = 1,
    current_label: str = "anxiety_and_overwhelm",
    aspirational_label: str = "emotionally_resilient",
    current_score: float = 0.4,
    aspirational_score: float = 0.4,
) -> UserProfile:
    """Create a synthetic UserProfile for testing."""
    return UserProfile(
        ref_user_id=user_id,
        current_states=[
            StateScore(label=current_label, description="test current", score=current_score),
        ],
        aspirational_states=[
            StateScore(label=aspirational_label, description="test aspirational", score=aspirational_score),
        ],
        embedding=np.random.randn(384).astype(np.float32),
    )


def make_post_profile(
    post_id: int = 100,
    current_label: str = "anxiety_and_overwhelm",
    aspirational_label: str = "emotionally_resilient",
    current_score: float = 0.5,
    aspirational_score: float = 0.5,
) -> PostProfile:
    """Create a synthetic PostProfile for testing."""
    return PostProfile(
        post_id=post_id,
        current_state_scores=[
            StateScore(label=current_label, description="test current topic", score=current_score),
        ],
        aspirational_state_scores=[
            StateScore(label=aspirational_label, description="test aspirational topic", score=aspirational_score),
        ],
        topic_keywords=["stress", "wellness"],
        embedding=np.random.randn(384).astype(np.float32),
    )


def make_activity(user_id: int, engaged_posts: set[int]) -> UserActivityIndex:
    """Create a synthetic UserActivityIndex."""
    idx = UserActivityIndex(ref_user_id=user_id)
    idx.read_posts = engaged_posts
    return idx


# ── Unit Tests: Reward Function Math ─────────────────────────────────────────


class TestScoreBreakdown:
    """Test the ScoreBreakdown total computation."""

    def test_total_sums_components(self):
        b = ScoreBreakdown(historical=1.0, aspirational=3.0, regression=-2.0)
        assert b.total == pytest.approx(2.0)

    def test_total_defaults_to_zero(self):
        b = ScoreBreakdown()
        assert b.total == 0.0


class TestScorePost:
    """Test the core score_post function against synthetic fixtures."""

    def test_aspirational_post_scores_higher_than_historical_only(self):
        """A post matching aspirational state scores higher than one with only historical engagement."""
        user = make_user_profile()
        activity = make_activity(user.ref_user_id, {100, 200})

        # Post that matches aspirational state AND has historical engagement
        aspirational_post = make_post_profile(
            post_id=100,
            aspirational_label="emotionally_resilient",
            aspirational_score=0.45,
            current_label="social_isolation",       # not user's current state
            current_score=0.1,
        )

        # Post that only has historical engagement (no aspirational match)
        historical_post = make_post_profile(
            post_id=200,
            aspirational_label="creative_and_curious",  # not user's aspirational state
            aspirational_score=0.45,
            current_label="social_isolation",
            current_score=0.1,
        )

        score_asp = score_post(user, aspirational_post, activity)
        score_hist = score_post(user, historical_post, activity)

        assert score_asp.aspirational > 0, "Aspirational post should have positive aspirational component"
        assert score_hist.aspirational == 0, "Non-matching post should have zero aspirational component"
        assert score_asp.total > score_hist.total, "Aspirational-aligned post must score higher"

    def test_regression_penalty_lowers_score(self):
        """A post triggering regression penalty scores lower than a neutral post."""
        user = make_user_profile(current_label="anxiety_and_overwhelm")
        activity = make_activity(user.ref_user_id, set())

        # Post that reinforces user's current negative state
        regressing_post = make_post_profile(
            post_id=100,
            current_label="anxiety_and_overwhelm",
            current_score=0.5,              # above regression threshold
            aspirational_label="other",
            aspirational_score=0.05,
        )

        # Neutral post — doesn't match any user state
        neutral_post = make_post_profile(
            post_id=200,
            current_label="grief_and_loss",     # not user's current
            current_score=0.5,
            aspirational_label="other",
            aspirational_score=0.05,
        )

        score_reg = score_post(user, regressing_post, activity)
        score_neu = score_post(user, neutral_post, activity)

        assert score_reg.regression < 0, "Regressing post should have negative regression component"
        assert score_neu.regression == 0, "Neutral post should have zero regression"
        assert score_reg.total < score_neu.total, "Regressing post must score lower than neutral"

    def test_aspirational_plus_historical_scores_highest(self):
        """A post matching both aspirational state AND historical engagement scores highest."""
        user = make_user_profile(aspirational_label="physically_active")
        activity = make_activity(user.ref_user_id, {100})

        # Post with both signals
        both_post = make_post_profile(
            post_id=100,
            aspirational_label="physically_active",
            aspirational_score=0.4,
            current_label="other",
            current_score=0.1,
        )

        # Post with only aspirational (no historical)
        asp_only_post = make_post_profile(
            post_id=200,
            aspirational_label="physically_active",
            aspirational_score=0.4,
            current_label="other",
            current_score=0.1,
        )

        # Post with only historical (no aspirational)
        hist_only_post = make_post_profile(
            post_id=100,
            aspirational_label="creative_and_curious",  # not user's asp state
            aspirational_score=0.4,
            current_label="other",
            current_score=0.1,
        )

        score_both = score_post(user, both_post, activity)
        score_asp = score_post(user, asp_only_post, activity)
        score_hist = score_post(user, hist_only_post, activity)

        assert score_both.total > score_asp.total, "Both signals > aspirational only"
        assert score_both.total > score_hist.total, "Both signals > historical only"
        assert score_both.historical > 0 and score_both.aspirational > 0, "Both components should be positive"

    def test_no_profile_falls_back_to_engagement_only(self):
        """Users without conversation data get engagement-only scoring."""
        activity = make_activity(user_id=999, engaged_posts={100, 200})

        engaged_post = make_post_profile(post_id=100)
        unengaged_post = make_post_profile(post_id=300)

        # user_profile=None simulates activity-only user
        score_eng = score_post(None, engaged_post, activity)
        score_no = score_post(None, unengaged_post, activity)

        assert score_eng.historical > 0, "Engaged post should have positive historical score"
        assert score_eng.aspirational == 0, "No profile means no aspirational score"
        assert score_eng.regression == 0, "No profile means no regression score"
        assert score_no.total == 0, "Unengaged post with no profile should score zero"

    def test_aspirational_below_threshold_gives_zero(self):
        """Aspirational score below threshold should not contribute."""
        user = make_user_profile(aspirational_label="physically_active")

        post = make_post_profile(
            post_id=100,
            aspirational_label="physically_active",
            aspirational_score=ASPIRATIONAL_THRESHOLD - 0.05,  # just below
            current_label="other",
            current_score=0.1,
        )

        result = score_post(user, post, None)
        assert result.aspirational == 0, "Below-threshold aspirational should be zero"

    def test_regression_below_threshold_gives_zero(self):
        """Regression score below threshold should not penalize."""
        user = make_user_profile(current_label="anxiety_and_overwhelm")

        post = make_post_profile(
            post_id=100,
            current_label="anxiety_and_overwhelm",
            current_score=REGRESSION_THRESHOLD - 0.05,  # just below
            aspirational_label="other",
            aspirational_score=0.05,
        )

        result = score_post(user, post, None)
        assert result.regression == 0, "Below-threshold regression should be zero"

    def test_reward_weights_are_correct(self):
        """Verify the reward weights produce the expected magnitudes."""
        user = make_user_profile(
            current_label="anxiety_and_overwhelm",
            aspirational_label="emotionally_resilient",
        )
        activity = make_activity(user.ref_user_id, {100})

        post = make_post_profile(
            post_id=100,
            aspirational_label="emotionally_resilient",
            aspirational_score=0.5,
            current_label="anxiety_and_overwhelm",
            current_score=0.5,
        )

        result = score_post(user, post, activity)

        assert result.historical == pytest.approx(HISTORICAL_ENGAGEMENT_REWARD)
        assert result.aspirational == pytest.approx(ASPIRATIONAL_ALIGNMENT_REWARD * 0.5)
        assert result.regression == pytest.approx(REGRESSION_PENALTY * 0.5)

    def test_centroid_historical_gives_continuous_score(self):
        """With engagement centroid, historical should be continuous, not binary."""
        user = make_user_profile()

        post = make_post_profile(post_id=300)
        # Centroid identical to post embedding → cosine similarity = 1.0
        centroid = post.embedding.copy()

        result = score_post(user, post, None, engagement_centroid=centroid)

        assert result.historical == pytest.approx(HISTORICAL_ENGAGEMENT_REWARD * 1.0, abs=0.01)

    def test_centroid_similar_vs_dissimilar(self):
        """Posts more similar to the engagement centroid should score higher."""
        np.random.seed(42)
        centroid = np.random.randn(384).astype(np.float32)

        # Similar post: centroid + small noise
        similar_post = make_post_profile(post_id=1)
        similar_post.embedding = (centroid + np.random.randn(384).astype(np.float32) * 0.1).astype(np.float32)

        # Dissimilar post: orthogonal direction
        np.random.seed(99)
        dissimilar_post = make_post_profile(post_id=2)
        dissimilar_post.embedding = np.random.randn(384).astype(np.float32)

        score_sim = score_post(None, similar_post, None, engagement_centroid=centroid)
        score_dis = score_post(None, dissimilar_post, None, engagement_centroid=centroid)

        assert score_sim.historical > score_dis.historical, "Similar post should score higher"


class TestEngagementCentroid:
    """Test the engagement centroid computation."""

    def test_centroid_computes_mean(self):
        """Centroid should be the mean of engaged post embeddings."""
        emb1 = np.ones(384, dtype=np.float32)
        emb2 = np.ones(384, dtype=np.float32) * 3

        p1 = make_post_profile(post_id=1)
        p1.embedding = emb1
        p2 = make_post_profile(post_id=2)
        p2.embedding = emb2

        profiles = {1: p1, 2: p2}
        centroid = compute_engagement_centroid({1, 2}, profiles)

        assert centroid is not None
        np.testing.assert_array_almost_equal(centroid, np.ones(384) * 2)

    def test_centroid_returns_none_for_no_engagement(self):
        """No engagement should return None."""
        centroid = compute_engagement_centroid(set(), {})
        assert centroid is None


class TestEpsilonGreedy:
    """Test the epsilon-greedy selection behavior."""

    def test_pure_exploitation_returns_top_scored(self):
        """With ε=0, recommendations should be strictly sorted by score."""
        from data_loader import load_all
        from state_extractor import extract_all

        data = load_all()
        extraction = extract_all(data)

        result = recommend_aspirational(
            user_id=221, data=data, extraction=extraction,
            n=10, epsilon=0.0, seed=42,
        )

        scores = [r.score for r in result.recommendations]
        assert scores == sorted(scores, reverse=True), "ε=0 should return posts in strict score order"

    def test_exploration_rate_converges(self):
        """Over many runs, the fraction of non-greedy picks should converge to ε."""
        from data_loader import load_all
        from state_extractor import extract_all

        data = load_all()
        extraction = extract_all(data)

        epsilon = 0.3
        n_trials = 500
        n_recs = 5
        exploration_count = 0
        total_selections = 0

        # Get the greedy ordering once (ε=0)
        greedy_result = recommend_aspirational(
            user_id=221, data=data, extraction=extraction,
            n=n_recs, epsilon=0.0, seed=0,
        )
        greedy_top = greedy_result.recommendations[0].post_id

        for trial in range(n_trials):
            result = recommend_aspirational(
                user_id=221, data=data, extraction=extraction,
                n=1, epsilon=epsilon, seed=trial,
            )
            if result.recommendations[0].post_id != greedy_top:
                exploration_count += 1
            total_selections += 1

        observed_rate = exploration_count / total_selections
        # Allow reasonable tolerance — with 500 trials and ε=0.3,
        # the observed exploration rate should be within [0.15, 0.45]
        assert 0.10 < observed_rate < 0.50, (
            f"Exploration rate {observed_rate:.2f} deviates too far from ε={epsilon}"
        )


class TestBaseline:
    """Test the baseline engagement-only recommender."""

    def test_baseline_measures_aspirational_and_regression(self):
        """Baseline should have retroactively computed aspirational/regression for metrics."""
        from data_loader import load_all
        from state_extractor import extract_all

        data = load_all()
        extraction = extract_all(data)

        result = recommend_baseline(user_id=221, data=data, extraction=extraction, n=10, epsilon=0.0)

        # The baseline now retroactively computes full breakdowns so metrics
        # reflect the actual aspirational/regression characteristics.
        # At least some posts should have non-zero aspirational or regression.
        has_asp = any(r.breakdown.aspirational > 0 for r in result.recommendations)
        has_reg = any(r.breakdown.regression < 0 for r in result.recommendations)
        assert has_asp or has_reg, (
            "Baseline should have measured aspirational/regression on at least some posts"
        )

        # But score should still be based on historical only (content-similarity)
        for r in result.recommendations:
            assert r.score == pytest.approx(r.breakdown.historical, abs=0.01), (
                "Baseline ranking score should match historical component only"
            )

    def test_baseline_excludes_engaged_and_ranks_by_similarity(self):
        """Baseline should exclude already-engaged posts and rank unseen by content similarity."""
        from data_loader import load_all
        from state_extractor import extract_all

        data = load_all()
        extraction = extract_all(data)

        user_id = 221
        activity = data.user_activity_index.get(user_id)
        assert activity is not None, "Test user should have activity data"

        engaged = activity.all_engaged_posts
        # Use epsilon=0 for deterministic ordering in this test
        result = recommend_baseline(user_id=user_id, data=data, extraction=extraction, n=10, epsilon=0.0)

        # No recommended post should be one the user already engaged with
        for r in result.recommendations:
            assert r.post_id not in engaged, "Baseline should not recommend already-engaged posts"

        # Scores should be continuous (not binary 0/1)
        scores = [r.score for r in result.recommendations]
        unique_scores = set(round(s, 4) for s in scores if s > 0)
        assert len(unique_scores) > 1, "Baseline should have varied (non-binary) scores"

        # Should be in descending order (pure exploitation)
        assert scores == sorted(scores, reverse=True), "Baseline should be sorted by score"

    def test_baseline_exploration_flag(self):
        """Baseline with epsilon > 0 should flag exploration picks."""
        from data_loader import load_all
        from state_extractor import extract_all

        data = load_all()
        extraction = extract_all(data)

        result_explore = recommend_baseline(
            user_id=221, data=data, extraction=extraction,
            n=10, epsilon=1.0, seed=42,
        )
        # With epsilon=1.0, all picks are exploration
        for r in result_explore.recommendations:
            assert r.is_exploration, "All picks should be exploration at ε=1.0"

        result_greedy = recommend_baseline(
            user_id=221, data=data, extraction=extraction,
            n=10, epsilon=0.0, seed=42,
        )
        # With epsilon=0.0, no picks are exploration
        for r in result_greedy.recommendations:
            assert not r.is_exploration, "No picks should be exploration at ε=0.0"


class TestMetrics:
    """Test the metrics computation."""

    def test_metrics_aspirational_alignment_rate(self):
        """Verify aspirational alignment rate counts correctly."""
        from recommender import RecommendationResult, ScoredPost

        recs = [
            ScoredPost(post_id=1, score=2.0, breakdown=ScoreBreakdown(aspirational=1.0)),
            ScoredPost(post_id=2, score=1.0, breakdown=ScoreBreakdown(aspirational=0.0)),
            ScoredPost(post_id=3, score=1.5, breakdown=ScoreBreakdown(aspirational=0.5)),
            ScoredPost(post_id=4, score=0.5, breakdown=ScoreBreakdown(aspirational=0.0)),
        ]
        result = RecommendationResult(user_id=1, recommendations=recs, engine_type="test")
        metrics = compute_metrics(result)

        assert metrics.aspirational_alignment_rate == pytest.approx(0.5)  # 2 of 4
        assert metrics.n_recommendations == 4

    def test_metrics_regression_rate(self):
        """Verify regression rate counts correctly."""
        from recommender import RecommendationResult, ScoredPost

        recs = [
            ScoredPost(post_id=1, score=0.0, breakdown=ScoreBreakdown(regression=-0.5)),
            ScoredPost(post_id=2, score=1.0, breakdown=ScoreBreakdown(regression=0.0)),
            ScoredPost(post_id=3, score=0.0, breakdown=ScoreBreakdown(regression=-1.0)),
        ]
        result = RecommendationResult(user_id=1, recommendations=recs, engine_type="test")
        metrics = compute_metrics(result)

        assert metrics.regression_rate == pytest.approx(2 / 3)

    def test_empty_recommendations_returns_zeros(self):
        """Empty recommendation list should return zeroed metrics."""
        from recommender import RecommendationResult

        result = RecommendationResult(user_id=1, recommendations=[], engine_type="test")
        metrics = compute_metrics(result)

        assert metrics.aspirational_alignment_rate == 0.0
        assert metrics.regression_rate == 0.0
        assert metrics.topic_entropy == 0.0
        assert metrics.n_recommendations == 0


# ── Integration Test: Full Pipeline ──────────────────────────────────────────


class TestIntegration:
    """End-to-end integration test: load data → extract → recommend → verify."""

    def test_full_pipeline_produces_valid_output(self):
        """Load real data, extract states, generate recommendations, and verify output shape."""
        from data_loader import load_all
        from state_extractor import extract_all

        data = load_all()
        extraction = extract_all(data)

        # Verify extraction produced the expected structure
        assert len(extraction.user_profiles) == 24
        assert len(extraction.post_profiles) == 56

        # Run aspirational recommender for a profiled user
        asp_result = recommend_aspirational(
            user_id=221, data=data, extraction=extraction,
            n=10, epsilon=0.1, seed=42,
        )
        assert len(asp_result.recommendations) == 10
        assert asp_result.engine_type == "aspirational"

        # Each recommendation has a valid breakdown
        for r in asp_result.recommendations:
            assert r.post_id in extraction.post_profiles
            assert isinstance(r.breakdown, ScoreBreakdown)
            assert r.score == pytest.approx(r.breakdown.total)

        # Metrics are computable
        metrics = asp_result.metrics
        assert 0 <= metrics.aspirational_alignment_rate <= 1
        assert 0 <= metrics.regression_rate <= 1
        assert metrics.topic_entropy >= 0

        # Run baseline for comparison
        base_result = recommend_baseline(
            user_id=221, data=data, extraction=extraction, n=10,
        )
        assert len(base_result.recommendations) == 10
        assert base_result.engine_type == "baseline"

    def test_aspirational_engine_outperforms_baseline_on_alignment(self):
        """The aspirational engine should have a higher alignment rate than baseline across users."""
        from data_loader import load_all
        from state_extractor import extract_all

        data = load_all()
        extraction = extract_all(data)

        asp_alignment_rates = []
        base_alignment_rates = []

        for user_id in extraction.user_profiles:
            asp = recommend_aspirational(user_id, data, extraction, n=10, epsilon=0.0, seed=42)
            base = recommend_baseline(user_id, data, extraction, n=10)
            asp_alignment_rates.append(asp.metrics.aspirational_alignment_rate)
            base_alignment_rates.append(base.metrics.aspirational_alignment_rate)

        avg_asp = sum(asp_alignment_rates) / len(asp_alignment_rates)
        avg_base = sum(base_alignment_rates) / len(base_alignment_rates)

        assert avg_asp > avg_base, (
            f"Aspirational engine avg alignment ({avg_asp:.2f}) should exceed baseline ({avg_base:.2f})"
        )
