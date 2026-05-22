"""Convert clusters into PM-readable insights."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from config import SETTINGS
from utils.logger import get_logger


logger = get_logger(__name__)
MAX_FEATURES = 1000
NGRAM_RANGE = (1, 2)
MIN_KEYWORD_LENGTH = 2
MAX_KEYWORDS = 5
MAX_POSITIVE_SCORE = 3
MAX_NEGATIVE_SCORE = 3

POSITIVE_WORDS = {
    "love",
    "great",
    "good",
    "excellent",
    "fast",
    "helpful",
    "best",
    "awesome",
    "clean",
    "improving",
    "nice",
    "smooth",
    "saved",
    "happy",
}
NEGATIVE_WORDS = {
    "refund",
    "expensive",
    "broken",
    "bug",
    "bugs",
    "crash",
    "crashes",
    "error",
    "errors",
    "fail",
    "failed",
    "slow",
    "freeze",
    "frozen",
    "charged",
    "cancel",
    "cancelled",
    "issue",
    "issues",
    "problem",
    "problems",
}
TOPIC_RULES: Dict[str, set[str]] = {
    "Pricing & Value Concerns": {"price", "pricing", "cost", "costs", "expensive", "budget", "value", "plan", "roi"},
    "Refund & Billing Issues": {"refund", "billing", "charged", "invoice", "subscription", "payment", "cancel", "cancelled", "credit"},
    "Bug Reports & Reliability": {"bug", "bugs", "error", "errors", "crash", "crashes", "freeze", "frozen", "sync", "failure"},
    "Feature Requests": {"feature", "request", "requests", "dark mode", "permissions", "audit logs", "roadmap", "workflow"},
    "Integration & API Questions": {"integration", "integrate", "api", "webhook", "slack", "docs", "documentation", "sso", "scim"},
    "Positive Feedback": {"love", "great", "excellent", "awesome", "best", "helpful", "fast", "clean", "saved"},
    "Account & Access Issues": {"login", "account", "access", "permission", "permissions", "subscription", "locked"},
}


@dataclass(slots=True)
class ClusterSummary:
    """Human-readable description of a cluster and its metric signal."""

    cluster_id: int
    topic: str
    size: int
    percentage: float
    sentiment: str
    key_phrases: List[str]
    sample_conversations: List[str]
    insight_text: str
    metric_type: str
    metric_value: float
    confidence: float


def extract_top_keywords(texts: Sequence[str], max_keywords: int = MAX_KEYWORDS) -> List[str]:
    """Return the strongest tokens or phrases that describe a cluster.

    Args:
        texts: Conversations assigned to one cluster.
        max_keywords: Number of keywords to keep.

    Returns:
        Ordered keyword phrases ranked by frequency.
    """

    if not texts:
        return []

    vectorizer = CountVectorizer(stop_words="english", ngram_range=NGRAM_RANGE, max_features=MAX_FEATURES)
    matrix = vectorizer.fit_transform(texts)
    if matrix.shape[1] == 0:
        return []

    feature_names = np.asarray(vectorizer.get_feature_names_out())
    counts = np.asarray(matrix.sum(axis=0)).ravel()
    ranked = np.argsort(counts)[::-1]

    keywords: List[str] = []
    for index in ranked:
        token = str(feature_names[index]).strip()
        if len(token) < MIN_KEYWORD_LENGTH:
            continue
        keywords.append(token)
        if len(keywords) >= max_keywords:
            break
    return keywords


def score_sentiment(text: str) -> float:
    """Return a simple sentiment score in the range ``[-1, 1]``.

    Args:
        text: Conversation text.

    Returns:
        A normalized sentiment score where negative values indicate friction.
    """

    tokens = {token.strip(".,!?;:\"'()[]{}-").lower() for token in text.split()}
    if not tokens:
        return 0.0

    positive_hits = len(tokens & POSITIVE_WORDS)
    negative_hits = len(tokens & NEGATIVE_WORDS)
    raw_score = positive_hits - negative_hits
    if raw_score == 0:
        return 0.0
    denominator = max(MAX_POSITIVE_SCORE, MAX_NEGATIVE_SCORE, len(tokens))
    return float(max(-1.0, min(1.0, raw_score / denominator)))


def dominant_sentiment(texts: Sequence[str]) -> str:
    """Summarize the cluster sentiment using the lexicon score.

    Args:
        texts: Cluster conversations.

    Returns:
        ``positive``, ``negative``, or ``neutral``.
    """

    if not texts:
        return "neutral"
    average = float(np.mean([score_sentiment(text) for text in texts]))
    if average > 0.1:
        return "positive"
    if average < -0.1:
        return "negative"
    return "neutral"


def infer_topic_label(key_phrases: Sequence[str], sentiment: str) -> str:
    """Infer a product-facing topic label from cluster keywords.

    Args:
        key_phrases: Ranked cluster keywords.
        sentiment: Dominant sentiment for the cluster.

    Returns:
        A readable topic label suitable for PM reporting.
    """

    normalized_phrases = [phrase.lower() for phrase in key_phrases]
    for topic, keywords in TOPIC_RULES.items():
        if any(keyword in phrase for phrase in normalized_phrases for keyword in keywords):
            return topic

    if key_phrases:
        primary = key_phrases[0].replace("_", " ").title()
        return f"{primary} ({sentiment})" if sentiment != "neutral" else primary
    return "General Feedback"


def build_cluster_summaries(conversations: Sequence[str], labels: Sequence[int], silhouette_score: float = 0.0) -> List[ClusterSummary]:
    """Create structured summaries for each cluster.

    Args:
        conversations: Cleaned conversations in the same order as ``labels``.
        labels: Cluster IDs assigned by K-means.
        silhouette_score: Quality signal from the clustering step.

    Returns:
        Cluster summaries ready for storage and API responses.
    """

    total = len(conversations)
    if total == 0:
        return []
    if len(conversations) != len(labels):
        raise ValueError("Conversation and label counts must match.")

    summaries: List[ClusterSummary] = []
    unique_labels = sorted({int(label) for label in labels})

    for cluster_id in unique_labels:
        cluster_texts = [text for text, label in zip(conversations, labels) if int(label) == cluster_id]
        if not cluster_texts:
            continue

        key_phrases = extract_top_keywords(cluster_texts)
        sentiment = dominant_sentiment(cluster_texts)
        topic = infer_topic_label(key_phrases, sentiment)
        percentage = round((len(cluster_texts) / total) * 100.0, 1)
        primary_phrase = key_phrases[0] if key_phrases else topic.lower()
        insight_text = f"{percentage:.1f}% of conversations mention {primary_phrase}."
        confidence = round(min(0.99, max(0.4, 0.55 + (percentage / 200.0) + abs(float(silhouette_score)) / 4.0)), 2)
        summary = ClusterSummary(
            cluster_id=cluster_id,
            topic=topic,
            size=len(cluster_texts),
            percentage=percentage,
            sentiment=sentiment,
            key_phrases=key_phrases,
            sample_conversations=cluster_texts[: SETTINGS.sample_conversation_limit],
            insight_text=insight_text,
            metric_type="cluster_share",
            metric_value=percentage,
            confidence=confidence,
        )
        logger.debug("Built summary for cluster %s with topic %s", cluster_id, topic)
        summaries.append(summary)

    return summaries


def cluster_summary_to_dict(summary: ClusterSummary) -> dict:
    """Serialize a cluster summary for JSON responses."""

    return asdict(summary)
