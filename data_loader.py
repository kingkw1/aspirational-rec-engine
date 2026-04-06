"""
data_loader.py — Data ingestion and preprocessing for the Aspirational Recommendation Engine.

Loads conversations.json, discussions.json, and activity.json.
Normalizes into clean dataclasses, filters flagged content, builds indexes,
and extracts interpretable topic keywords via spaCy.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import spacy

DATA_DIR = Path(__file__).parent / "data"

COACH_USER_ID = 1


# ── Dataclasses ──────────────────────────────────────────────────────────────


@dataclass
class Message:
    """A single message in a chatbot conversation."""
    ref_conversation_id: int
    ref_user_id: int
    timestamp: str
    screen_name: str
    text: str


@dataclass
class Conversation:
    """A full conversation between a user and the AI Health Coach."""
    ref_conversation_id: int
    ref_user_id: int
    messages: list[Message]

    @property
    def user_messages(self) -> list[Message]:
        """Return only the user's messages (exclude the coach)."""
        return [m for m in self.messages if m.ref_user_id != COACH_USER_ID]

    @property
    def user_text(self) -> str:
        """Concatenated text of all user messages."""
        return " ".join(m.text for m in self.user_messages)


@dataclass
class Comment:
    """A single comment (or opening post) in a discussion thread."""
    post_id: int
    comment_id: Optional[int]  # None for the opening post
    text: str
    author_ref_user_id: int
    reported_or_removed: bool

    @property
    def is_opening_post(self) -> bool:
        return self.comment_id is None


@dataclass
class DiscussionPost:
    """A full discussion thread: opening post + comments."""
    post_id: int
    comments: list[Comment]
    topic_keywords: list[str] = field(default_factory=list)

    @property
    def opening_post(self) -> Optional[Comment]:
        for c in self.comments:
            if c.is_opening_post:
                return c
        return None

    @property
    def clean_comments(self) -> list[Comment]:
        """Comments excluding reported/removed content."""
        return [c for c in self.comments if not c.reported_or_removed]

    @property
    def clean_text(self) -> str:
        """Concatenated text of all non-removed messages."""
        return " ".join(c.text for c in self.clean_comments)

    @property
    def snippet(self) -> str:
        """Short preview from the opening post."""
        op = self.opening_post
        if op:
            return op.text[:200] + ("..." if len(op.text) > 200 else "")
        return ""


@dataclass
class ActivityRecord:
    """A single user activity event."""
    ref_user_id: int
    activity_type: str  # "read", "commented", "created"
    post_id: int


@dataclass
class UserActivityIndex:
    """Aggregated activity for a single user, bucketed by type."""
    ref_user_id: int
    read_posts: set[int] = field(default_factory=set)
    commented_posts: set[int] = field(default_factory=set)
    created_posts: set[int] = field(default_factory=set)

    @property
    def all_engaged_posts(self) -> set[int]:
        """All posts the user has interacted with, in any way."""
        return self.read_posts | self.commented_posts | self.created_posts


# ── Loading Functions ────────────────────────────────────────────────────────


def load_conversations(path: Optional[Path] = None) -> list[Conversation]:
    """Load and parse conversations.json."""
    path = path or DATA_DIR / "conversations.json"
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    conversations = []
    for entry in raw:
        messages = [
            Message(
                ref_conversation_id=m["ref_conversation_id"],
                ref_user_id=m["ref_user_id"],
                timestamp=m["transaction_datetime_utc"],
                screen_name=m["screen_name"],
                text=m["message"],
            )
            for m in entry["messages_list"]
        ]
        conversations.append(
            Conversation(
                ref_conversation_id=entry["ref_conversation_id"],
                ref_user_id=entry["ref_user_id"],
                messages=messages,
            )
        )
    return conversations


def load_discussions(path: Optional[Path] = None) -> list[DiscussionPost]:
    """Load and parse discussions.json."""
    path = path or DATA_DIR / "discussions.json"
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    posts = []
    for entry in raw:
        comments = [
            Comment(
                post_id=m["post_id"],
                comment_id=m["comment_id"],
                text=m["text"],
                author_ref_user_id=m["author_ref_user_id"],
                reported_or_removed=m["reported_or_removed"],
            )
            for m in entry["messages_list"]
        ]
        posts.append(DiscussionPost(post_id=entry["post_id"], comments=comments))
    return posts


def load_activity(path: Optional[Path] = None) -> list[ActivityRecord]:
    """Load and parse activity.json."""
    path = path or DATA_DIR / "activity.json"
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    return [
        ActivityRecord(
            ref_user_id=r["ref_user_id"],
            activity_type=r["activity_type"],
            post_id=r["post_id"],
        )
        for r in raw
    ]


# ── Index Builders ───────────────────────────────────────────────────────────


def build_user_activity_index(
    activity: list[ActivityRecord],
) -> dict[int, UserActivityIndex]:
    """Build a per-user activity index bucketed by interaction type."""
    index: dict[int, UserActivityIndex] = {}
    for rec in activity:
        if rec.ref_user_id not in index:
            index[rec.ref_user_id] = UserActivityIndex(ref_user_id=rec.ref_user_id)
        entry = index[rec.ref_user_id]
        if rec.activity_type == "read":
            entry.read_posts.add(rec.post_id)
        elif rec.activity_type == "commented":
            entry.commented_posts.add(rec.post_id)
        elif rec.activity_type == "created":
            entry.created_posts.add(rec.post_id)
    return index


def build_post_content_index(posts: list[DiscussionPost]) -> dict[int, str]:
    """Build a mapping of post_id -> concatenated clean text."""
    return {post.post_id: post.clean_text for post in posts}


# ── spaCy Topic Keyword Extraction ──────────────────────────────────────────


def extract_topic_keywords(
    posts: list[DiscussionPost], nlp: Optional[spacy.language.Language] = None
) -> None:
    """Extract noun phrases and named entities from each post using spaCy.

    Mutates each DiscussionPost in-place, populating `topic_keywords`.
    """
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")

    for post in posts:
        doc = nlp(post.clean_text)

        keywords: set[str] = set()

        # Named entities (ORG, PERSON excluded — focus on topical ones)
        for ent in doc.ents:
            if ent.label_ not in ("PERSON", "ORDINAL", "CARDINAL", "DATE", "TIME"):
                keywords.add(ent.text.lower().strip())

        # Noun chunks (filter short/stop-word-only chunks)
        for chunk in doc.noun_chunks:
            # Keep chunks with at least one meaningful token
            meaningful = [t for t in chunk if not t.is_stop and not t.is_punct and len(t.text) > 2]
            if meaningful:
                keywords.add(chunk.lemma_.lower().strip())

        post.topic_keywords = sorted(keywords)


# ── Top-Level Loader ─────────────────────────────────────────────────────────


@dataclass
class AppData:
    """Container for all loaded and indexed data."""
    conversations: list[Conversation]
    posts: list[DiscussionPost]
    activity: list[ActivityRecord]
    user_activity_index: dict[int, UserActivityIndex]
    post_content_index: dict[int, str]

    @property
    def conversation_user_ids(self) -> set[int]:
        return {c.ref_user_id for c in self.conversations}

    @property
    def activity_user_ids(self) -> set[int]:
        return set(self.user_activity_index.keys())


def load_all(data_dir: Optional[Path] = None, extract_keywords: bool = True) -> AppData:
    """Load all data, build indexes, and extract topic keywords.

    Args:
        data_dir: Override the default data directory.
        extract_keywords: If True, run spaCy keyword extraction on posts.
    """
    if data_dir is not None:
        conv_path = data_dir / "conversations.json"
        disc_path = data_dir / "discussions.json"
        act_path = data_dir / "activity.json"
    else:
        conv_path = disc_path = act_path = None

    conversations = load_conversations(conv_path)
    posts = load_discussions(disc_path)
    activity = load_activity(act_path)

    user_activity_index = build_user_activity_index(activity)
    post_content_index = build_post_content_index(posts)

    if extract_keywords:
        extract_topic_keywords(posts)

    return AppData(
        conversations=conversations,
        posts=posts,
        activity=activity,
        user_activity_index=user_activity_index,
        post_content_index=post_content_index,
    )


# ── CLI Smoke Test ───────────────────────────────────────────────────────────


if __name__ == "__main__":
    print("Loading all data...")
    data = load_all()

    print(f"\n{'='*60}")
    print("DATA SUMMARY")
    print(f"{'='*60}")

    print(f"\nConversations: {len(data.conversations)}")
    print(f"  Users with conversations: {len(data.conversation_user_ids)}")
    msg_counts = [len(c.messages) for c in data.conversations]
    print(f"  Messages per conversation: min={min(msg_counts)}, max={max(msg_counts)}, avg={sum(msg_counts)/len(msg_counts):.1f}")

    print(f"\nDiscussion Posts: {len(data.posts)}")
    total_comments = sum(len(p.comments) for p in data.posts)
    removed = sum(1 for p in data.posts for c in p.comments if c.reported_or_removed)
    print(f"  Total comments (incl. opening posts): {total_comments}")
    print(f"  Reported/removed: {removed}")

    print(f"\nActivity Records: {len(data.activity)}")
    print(f"  Unique users: {len(data.activity_user_ids)}")
    from collections import Counter
    type_counts = Counter(r.activity_type for r in data.activity)
    for atype, count in type_counts.most_common():
        print(f"    {atype}: {count}")

    # User overlap
    overlap = data.conversation_user_ids & data.activity_user_ids
    conv_only = data.conversation_user_ids - data.activity_user_ids
    act_only = data.activity_user_ids - data.conversation_user_ids
    print(f"\nUser Overlap:")
    print(f"  In both conversations + activity: {len(overlap)}")
    print(f"  Conversation-only: {len(conv_only)}")
    print(f"  Activity-only: {len(act_only)}")

    # spaCy keyword samples
    print(f"\nSample Topic Keywords (first 3 posts):")
    for post in data.posts[:3]:
        print(f"  Post {post.post_id}: {post.topic_keywords[:8]}")

    print(f"\n{'='*60}")
    print("Phase 1 complete. All data loaded successfully.")
    print(f"{'='*60}")
