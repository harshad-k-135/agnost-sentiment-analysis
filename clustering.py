"""Embedding, clustering, and insight extraction utilities.

This module keeps the analytics pipeline intentionally simple and explainable:
normalize the input, embed it, cluster it, then turn each cluster into a short
sentence that a PM can read without decoding model internals.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from functools import lru_cache
from typing import List, Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer


# Config constants
MODEL_NAME = "all-MiniLM-L6-v2"
RANDOM_STATE = 42
DEFAULT_CLUSTER_COUNT = 5
MIN_CLUSTER_COUNT = 2
MIN_SAMPLES_FOR_SILHOUETTE = 3
MAX_CLUSTERS_CAP = 8
MAX_KEYWORDS_PER_CLUSTER = 3
MAX_CLUSTER_SAMPLE_CONVERSATIONS = 3
MIN_KEYWORD_LENGTH = 2
NGRAM_MIN = 1
NGRAM_MAX = 2
TOP_FEATURE_COUNT = 1000


@dataclass
class ClusterResult:
    """Structured summary for one conversation cluster."""

    cluster_id: int
    size: int
    percentage: float
    top_keywords: List[str]
    insight_text: str
    sample_conversations: List[str]


def normalize_conversations(conversations: Sequence[str]) -> List[str]:
    """Normalize and trim whitespace, and drop empty rows.

    Args:
        conversations: Raw user conversation strings.

    Returns:
        A cleaned list of non-empty strings suitable for embedding.
    """

    out: List[str] = []
    for c in conversations:
        if not c:
            continue
        cleaned = " ".join(c.split()).strip()
        if cleaned:
            out.append(cleaned)
    return out


@lru_cache(maxsize=1)
def load_embedding_model() -> SentenceTransformer:
    """Load and cache the sentence-transformer model used for embeddings."""

    return SentenceTransformer(MODEL_NAME)


def embed_conversations(conversations: Sequence[str]) -> np.ndarray:
    """Convert conversations into normalized semantic embeddings."""

    model = load_embedding_model()
    embeddings = model.encode(list(conversations), show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(embeddings)


def choose_cluster_count(embeddings: np.ndarray) -> int:
    """Choose the K-means cluster count using silhouette score when possible."""

    n = embeddings.shape[0]
    if n < MIN_CLUSTER_COUNT:
        return 1

    upper = min(DEFAULT_CLUSTER_COUNT, MAX_CLUSTERS_CAP, n - 1)
    if n < MIN_SAMPLES_FOR_SILHOUETTE or upper < MIN_CLUSTER_COUNT:
        return min(DEFAULT_CLUSTER_COUNT, n)

    best_k = MIN_CLUSTER_COUNT
    best_score = float("-inf")
    for k in range(MIN_CLUSTER_COUNT, upper + 1):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(embeddings)
        if len(set(labels)) < MIN_CLUSTER_COUNT:
            continue
        score = silhouette_score(embeddings, labels)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k


def cluster_conversations(conversations: Sequence[str]):
    """Cluster conversations and return labels with their embeddings.

    Args:
        conversations: Raw or normalized conversation strings.

    Returns:
        A tuple of ``(labels, embeddings)``.
    """

    normalized = normalize_conversations(conversations)
    if not normalized:
        raise ValueError("At least one non-empty conversation is required.")

    embeddings = embed_conversations(normalized)
    k = choose_cluster_count(embeddings)
    if k <= 1:
        labels = np.zeros(len(normalized), dtype=int)
        return labels, embeddings

    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(embeddings)
    return labels, embeddings


def extract_top_keywords(texts: Sequence[str]) -> List[str]:
    """Return the top tokens or phrases that characterize a cluster."""

    if not texts:
        return []
    vec = CountVectorizer(stop_words="english", ngram_range=(NGRAM_MIN, NGRAM_MAX), max_features=TOP_FEATURE_COUNT)
    X = vec.fit_transform(texts)
    if X.shape[1] == 0:
        return []

    sums = np.asarray(X.sum(axis=0)).ravel()
    features = np.asarray(vec.get_feature_names_out())
    ranked = np.argsort(sums)[::-1]

    keywords: List[str] = []
    for idx in ranked:
        token = features[idx].strip()
        if len(token) < MIN_KEYWORD_LENGTH:
            continue
        keywords.append(token)
        if len(keywords) >= MAX_KEYWORDS_PER_CLUSTER:
            break
    return keywords


def keyword_coverage(texts: Sequence[str], keyword: str) -> float:
    """Calculate the percentage of texts that mention a keyword."""

    if not texts or not keyword:
        return 0.0
    key = keyword.lower()
    matches = sum(1 for t in texts if key in t.lower())
    return (matches / len(texts)) * 100.0


def build_cluster_results(conversations: Sequence[str], labels: Sequence[int]) -> List[ClusterResult]:
    """Build human-friendly cluster summaries for each cluster ID in ``labels``."""

    normalized = normalize_conversations(conversations)
    if len(normalized) != len(labels):
        raise ValueError("Conversation and label counts must match.")

    total = len(normalized)
    results: List[ClusterResult] = []
    unique_clusters = sorted(set(int(l) for l in labels))

    for cid in unique_clusters:
        cluster_texts = [normalized[i] for i, l in enumerate(labels) if int(l) == cid]
        if not cluster_texts:
            continue
        top_keywords = extract_top_keywords(cluster_texts)
        primary = top_keywords[0] if top_keywords else "cluster theme"
        coverage = keyword_coverage(cluster_texts, primary)
        pct = round((len(cluster_texts) / total) * 100.0, 2)
        insight = f"{coverage:.0f}% of users in cluster_{cid} mention '{primary}'"
        samples = cluster_texts[:MAX_CLUSTER_SAMPLE_CONVERSATIONS]

        results.append(ClusterResult(cluster_id=cid, size=len(cluster_texts), percentage=pct, top_keywords=top_keywords, insight_text=insight, sample_conversations=samples))

    return results


def analyze_conversations(conversations: Sequence[str]):
    """Run the full analysis pipeline and return labels plus cluster summaries."""

    normalized = normalize_conversations(conversations)
    labels, _ = cluster_conversations(normalized)
    cluster_results = build_cluster_results(normalized, labels)
    return {"labels": labels.tolist(), "cluster_results": cluster_results}


def cluster_result_to_dict(result: ClusterResult) -> dict:
    """Convert a cluster result dataclass to a JSON-friendly dictionary."""

    return asdict(result)
