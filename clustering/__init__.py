"""Public clustering package API for embedding, topic discovery, and insight generation."""

from __future__ import annotations

from dataclasses import asdict
from typing import List, Sequence, Tuple

import numpy as np

from .embedder import encode_conversations as _encode_conversations
from .embedder import load_embedding_model
from .clusterer import cluster_embeddings, get_optimal_k
from .insight_extractor import ClusterSummary, build_cluster_summaries, cluster_summary_to_dict, dominant_sentiment, extract_top_keywords, infer_topic_label, score_sentiment
from utils.logger import get_logger


logger = get_logger(__name__)


def normalize_conversations(conversations: Sequence[str]) -> List[str]:
    """Normalize, trim, and drop empty conversation strings."""

    cleaned: List[str] = []
    for conversation in conversations:
        if not conversation:
            continue
        text = " ".join(str(conversation).split()).strip()
        if text:
            cleaned.append(text)
    return cleaned


def embed_conversations(conversations: Sequence[str]) -> np.ndarray:
    """Embed conversations using the configured sentence-transformer model."""

    return _encode_conversations(conversations, model=load_embedding_model())


def cluster_conversations(conversations: Sequence[str], num_clusters: int | None = None) -> Tuple[np.ndarray, np.ndarray, int, float]:
    """Cluster cleaned conversations and return labels plus embeddings."""

    normalized = normalize_conversations(conversations)
    if not normalized:
        raise ValueError("At least one non-empty conversation is required.")

    embeddings = embed_conversations(normalized)
    labels, optimal_k, silhouette = cluster_embeddings(embeddings, num_clusters=num_clusters)
    return labels, embeddings, optimal_k, silhouette


def build_cluster_results(conversations: Sequence[str], labels: Sequence[int], silhouette_score: float = 0.0) -> List[ClusterSummary]:
    """Build structured summaries for each cluster.

    This helper keeps the package-level API compatible with the previous repo
    while returning richer, PM-friendly cluster metadata.
    """

    normalized = normalize_conversations(conversations)
    return build_cluster_summaries(normalized, labels, silhouette_score=silhouette_score)


def analyze_conversations(conversations: Sequence[str], num_clusters: int | None = None) -> dict:
    """Run the full clustering and insight pipeline.

    Args:
        conversations: Raw conversation strings.
        num_clusters: Optional manual cluster count.

    Returns:
        A structured payload containing cluster labels, summaries, and metadata.
    """

    normalized = normalize_conversations(conversations)
    if not normalized:
        raise ValueError("At least one non-empty conversation is required.")

    labels, _embeddings, optimal_k, silhouette_score = cluster_conversations(normalized, num_clusters=num_clusters)
    cluster_results = build_cluster_summaries(normalized, labels, silhouette_score=silhouette_score)
    payload = {
        "labels": labels.tolist(),
        "cluster_results": [cluster_summary_to_dict(result) for result in cluster_results],
        "optimal_k": optimal_k,
        "silhouette_score": float(silhouette_score),
        "total_conversations": len(normalized),
    }
    logger.info("Analyzed %s conversations into %s clusters", len(normalized), len(cluster_results))
    return payload


__all__ = [
    "ClusterSummary",
    "analyze_conversations",
    "build_cluster_results",
    "cluster_conversations",
    "cluster_embeddings",
    "dominant_sentiment",
    "embed_conversations",
    "extract_top_keywords",
    "get_optimal_k",
    "infer_topic_label",
    "load_embedding_model",
    "normalize_conversations",
    "score_sentiment",
]
