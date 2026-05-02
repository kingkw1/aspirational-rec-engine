"""
state_extractor.py — Infer user states and post topics via sentence embeddings.

Uses a curated state taxonomy and sentence-transformers to:
  1. Infer current_state and aspirational_state per user from AI Health Coach conversations.
  2. Derive topic vectors and tags for each discussion post.

Built on a Strategy pattern: BaseExtractor ABC allows swapping the embedding backend.
Default EmbeddingExtractor runs a local HuggingFace model (all-MiniLM-L6-v2) via PyTorch.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from data_loader import Conversation, DiscussionPost, AppData


# ── State Taxonomy ───────────────────────────────────────────────────────────

# Current states: things the user is struggling with right now.
# Each entry is (label, descriptive sentence for embedding).
CURRENT_STATE_TAXONOMY: list[tuple[str, str]] = [
    (
        "anxiety_and_overwhelm",
        "feeling anxious, overwhelmed, stressed out, and unable to cope with daily life",
    ),
    (
        "social_isolation",
        "experiencing loneliness, social isolation, missing family and friends, feeling disconnected",
    ),
    (
        "physical_health_decline",
        "declining physical health, mobility issues, chronic pain, knee problems, falls, seizures",
    ),
    (
        "technology_frustration",
        "struggling with technology, frustrated with apps and devices, feeling helpless with computers",
    ),
    (
        "grief_and_loss",
        "grieving the loss of a loved one, processing death and bereavement, feeling sad about loss",
    ),
    (
        "work_life_imbalance",
        "overwhelmed by work deadlines, juggling family and career responsibilities, burnout and exhaustion",
    ),
    (
        "low_motivation",
        "lacking motivation and energy, feeling stuck, unable to start or maintain healthy habits",
    ),
    (
        "fear_and_uncertainty",
        "feeling scared about health, fearful of the future, anxiety about medical tests and diagnoses",
    ),
]

# Aspirational states: who the user wants to become.
ASPIRATIONAL_STATE_TAXONOMY: list[tuple[str, str]] = [
    (
        "physically_active",
        "becoming physically active, exercising regularly, running, doing yoga, building strength and fitness",
    ),
    (
        "socially_connected",
        "building social connections, joining community groups, making friends, feeling part of a community",
    ),
    (
        "emotionally_resilient",
        "developing emotional resilience, managing stress effectively, practicing mindfulness and calm",
    ),
    (
        "independent_and_capable",
        "becoming independent, mastering new skills, feeling capable and self-sufficient with technology",
    ),
    (
        "creative_and_curious",
        "exploring new hobbies, being creative, trying pottery or cooking or art, lifelong learning",
    ),
    (
        "healthy_and_balanced",
        "achieving work-life balance, maintaining good health, eating well, sleeping better, recovery",
    ),
    (
        "purposeful_and_engaged",
        "finding purpose and meaning in life, volunteering, mentoring, feeling valued and contributing",
    ),
    (
        "adventurous_and_growing",
        "seeking new experiences, hiking, traveling, pushing personal boundaries, setting and achieving goals",
    ),
]


# ── Dataclasses ──────────────────────────────────────────────────────────────


@dataclass
class StateScore:
    """A single state label with its similarity score."""
    label: str
    description: str
    score: float


@dataclass
class UserProfile:
    """Inferred profile for a user with conversation data."""
    ref_user_id: int
    current_states: list[StateScore]
    aspirational_states: list[StateScore]
    embedding: NDArray[np.float32]

    @property
    def top_current(self) -> StateScore:
        return self.current_states[0]

    @property
    def top_aspirational(self) -> StateScore:
        return self.aspirational_states[0]

    @property
    def current_labels(self) -> list[str]:
        return [s.label for s in self.current_states]

    @property
    def aspirational_labels(self) -> list[str]:
        return [s.label for s in self.aspirational_states]


@dataclass
class PostProfile:
    """Inferred topic profile for a discussion post."""
    post_id: int
    current_state_scores: list[StateScore]
    aspirational_state_scores: list[StateScore]
    topic_keywords: list[str]
    embedding: NDArray[np.float32]

    @property
    def top_current_topic(self) -> StateScore:
        return self.current_state_scores[0]

    @property
    def top_aspirational_topic(self) -> StateScore:
        return self.aspirational_state_scores[0]


@dataclass
class ExtractionResult:
    """Container for all extraction outputs."""
    user_profiles: dict[int, UserProfile]
    post_profiles: dict[int, PostProfile]


# ── Strategy Pattern: Extractor Interface ────────────────────────────────────


class BaseExtractor(ABC):
    """Abstract base for state extraction backends."""

    @abstractmethod
    def extract(self, data: AppData) -> ExtractionResult:
        """Run extraction over loaded data, returning user and post profiles."""
        ...


class EmbeddingExtractor(BaseExtractor):
    """Default extractor using local HuggingFace sentence-transformers.

    Loads all-MiniLM-L6-v2 via PyTorch, embeds user text and post text,
    and scores against the curated state taxonomy via cosine similarity.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        top_k_states: int = 8,
    ):
        self.model_name = model_name
        self.top_k = top_k_states
        self._model: SentenceTransformer | None = None
        self._taxonomy_embeddings: dict[str, NDArray] | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _embed(self, texts: list[str]) -> NDArray[np.float32]:
        """Embed a batch of texts, returning (n, dim) array."""
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def _get_taxonomy_embeddings(self) -> tuple[NDArray, NDArray]:
        """Embed the state taxonomy (cached after first call)."""
        if self._taxonomy_embeddings is None:
            current_texts = [desc for _, desc in CURRENT_STATE_TAXONOMY]
            aspirational_texts = [desc for _, desc in ASPIRATIONAL_STATE_TAXONOMY]
            self._taxonomy_embeddings = {
                "current": self._embed(current_texts),
                "aspirational": self._embed(aspirational_texts),
            }
        return self._taxonomy_embeddings["current"], self._taxonomy_embeddings["aspirational"]

    def _score_against_taxonomy(
        self,
        embedding: NDArray,
        taxonomy: list[tuple[str, str]],
        taxonomy_embeddings: NDArray,
        top_k: int | None = None,
    ) -> list[StateScore]:
        """Score a single embedding against a taxonomy, returning sorted StateScores."""
        top_k = top_k or self.top_k
        # embedding shape: (dim,) -> (1, dim)
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(embedding, taxonomy_embeddings)[0]
        ranked_idx = np.argsort(sims)[::-1][:top_k]
        return [
            StateScore(
                label=taxonomy[i][0],
                description=taxonomy[i][1],
                score=float(sims[i]),
            )
            for i in ranked_idx
        ]

    def _build_user_profile(
        self,
        conversation: Conversation,
        current_emb: NDArray,
        aspirational_emb: NDArray,
    ) -> UserProfile:
        """Build a UserProfile from a single conversation."""
        user_text = conversation.user_text
        if not user_text.strip():
            # Edge case: no user messages — return empty profile
            dummy = np.zeros(self.model.get_sentence_embedding_dimension(), dtype=np.float32)
            return UserProfile(
                ref_user_id=conversation.ref_user_id,
                current_states=[],
                aspirational_states=[],
                embedding=dummy,
            )

        user_embedding = self._embed([user_text])[0]

        current_scores = self._score_against_taxonomy(
            user_embedding, CURRENT_STATE_TAXONOMY, current_emb
        )
        aspirational_scores = self._score_against_taxonomy(
            user_embedding, ASPIRATIONAL_STATE_TAXONOMY, aspirational_emb
        )

        return UserProfile(
            ref_user_id=conversation.ref_user_id,
            current_states=current_scores,
            aspirational_states=aspirational_scores,
            embedding=user_embedding,
        )

    def _build_post_profile(
        self,
        post: DiscussionPost,
        current_emb: NDArray,
        aspirational_emb: NDArray,
    ) -> PostProfile:
        """Build a PostProfile from a discussion post."""
        text = post.clean_text
        if not text.strip():
            dummy = np.zeros(self.model.get_sentence_embedding_dimension(), dtype=np.float32)
            return PostProfile(
                post_id=post.post_id,
                current_state_scores=[],
                aspirational_state_scores=[],
                topic_keywords=post.topic_keywords,
                embedding=dummy,
            )

        post_embedding = self._embed([text])[0]

        # Score against both taxonomies so the recommender can check alignment in both directions
        current_scores = self._score_against_taxonomy(
            post_embedding, CURRENT_STATE_TAXONOMY, current_emb, top_k=len(CURRENT_STATE_TAXONOMY)
        )
        aspirational_scores = self._score_against_taxonomy(
            post_embedding, ASPIRATIONAL_STATE_TAXONOMY, aspirational_emb, top_k=len(ASPIRATIONAL_STATE_TAXONOMY)
        )

        return PostProfile(
            post_id=post.post_id,
            current_state_scores=current_scores,
            aspirational_state_scores=aspirational_scores,
            topic_keywords=post.topic_keywords,
            embedding=post_embedding,
        )

    def extract(self, data: AppData) -> ExtractionResult:
        """Run full extraction: user profiles + post profiles."""
        current_emb, aspirational_emb = self._get_taxonomy_embeddings()

        user_profiles = {}
        for conv in data.conversations:
            profile = self._build_user_profile(conv, current_emb, aspirational_emb)
            user_profiles[conv.ref_user_id] = profile

        post_profiles = {}
        for post in data.posts:
            profile = self._build_post_profile(post, current_emb, aspirational_emb)
            post_profiles[post.post_id] = profile

        return ExtractionResult(
            user_profiles=user_profiles,
            post_profiles=post_profiles,
        )


class LLMExtractor(BaseExtractor):
    """Stub: production extractor using a fine-tuned or API-based LLM.

    In production, this would call a hosted model (e.g., a fine-tuned
    HuggingFace model on AWS SageMaker, or an API like OpenAI) to generate
    richer, free-text state descriptions per user. The interface is identical
    to EmbeddingExtractor so it can be swapped without changing downstream code.

    Not implemented for this prototype — included to demonstrate the
    extensible architecture.
    """

    def extract(self, data: AppData) -> ExtractionResult:
        raise NotImplementedError(
            "LLMExtractor is a production stub. Use EmbeddingExtractor for this prototype."
        )


# ── Convenience Function ─────────────────────────────────────────────────────


def extract_all(
    data: AppData,
    extractor: BaseExtractor | None = None,
) -> ExtractionResult:
    """Run state extraction using the given extractor (default: EmbeddingExtractor)."""
    if extractor is None:
        extractor = EmbeddingExtractor()
    return extractor.extract(data)


# ── CLI Smoke Test ───────────────────────────────────────────────────────────


if __name__ == "__main__":
    from data_loader import load_all

    print("Loading data...")
    data = load_all()

    print("Running state extraction (this may download the model on first run)...")
    result = extract_all(data)

    print(f"\n{'='*60}")
    print("EXTRACTION SUMMARY")
    print(f"{'='*60}")

    print(f"\nUser Profiles: {len(result.user_profiles)}")
    for uid, profile in sorted(result.user_profiles.items())[:5]:
        print(f"\n  User {uid}:")
        print(f"    Current:      {profile.top_current.label} ({profile.top_current.score:.3f})")
        if len(profile.current_states) > 1:
            print(f"                  {profile.current_states[1].label} ({profile.current_states[1].score:.3f})")
        print(f"    Aspirational: {profile.top_aspirational.label} ({profile.top_aspirational.score:.3f})")
        if len(profile.aspirational_states) > 1:
            print(f"                  {profile.aspirational_states[1].label} ({profile.aspirational_states[1].score:.3f})")

    print(f"\nPost Profiles: {len(result.post_profiles)}")
    for pid, profile in list(sorted(result.post_profiles.items()))[:5]:
        print(f"\n  Post {pid}:")
        print(f"    Top current topic:      {profile.top_current_topic.label} ({profile.top_current_topic.score:.3f})")
        print(f"    Top aspirational topic: {profile.top_aspirational_topic.label} ({profile.top_aspirational_topic.score:.3f})")
        print(f"    Keywords: {profile.topic_keywords[:5]}")

    print(f"\n{'='*60}")
    print("Phase 2 complete. State extraction successful.")
    print(f"{'='*60}")
