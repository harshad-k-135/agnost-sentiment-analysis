"""K-means clustering helpers with automatic k selection."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from config import SETTINGS
from utils.logger import get_logger


logger = get_logger(__name__)
RANDOM_STATE = 42
MIN_CLUSTERS = 2
N_INIT = 10


def get_optimal_k(embeddings: np.ndarray, max_k: int = 10) -> Tuple[int, float]:
    """Detect the best cluster count using silhouette score.

    Args:
        embeddings: Normalized embedding matrix shaped ``[n_samples, n_features]``.
        max_k: Upper bound for the search space.

    Returns:
        A tuple of ``(optimal_k, best_silhouette_score)``.

    Raises:
        ValueError: If the embeddings array is invalid or too small.
    """

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array.")

    sample_count = embeddings.shape[0]
    if sample_count < MIN_CLUSTERS:
        raise ValueError("Need at least 2 samples to estimate k.")

    upper_bound = min(max_k, SETTINGS.default_max_clusters, sample_count - 1)
    if upper_bound < MIN_CLUSTERS:
        return 1, 0.0

    best_k = MIN_CLUSTERS
    best_score = float("-inf")

    for k in range(MIN_CLUSTERS, upper_bound + 1):
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
        labels = model.fit_predict(embeddings)
        if len(set(labels.tolist())) < MIN_CLUSTERS:
            continue
        score = silhouette_score(embeddings, labels)
        logger.debug("Evaluated k=%s with silhouette=%.4f", k, score)
        if score > best_score:
            best_score = score
            best_k = k

    if best_score == float("-inf"):
        return 1, 0.0
    return best_k, float(best_score)


def cluster_embeddings(embeddings: np.ndarray, num_clusters: int | None = None) -> Tuple[np.ndarray, int, float]:
    """Cluster embeddings with K-means.

    Args:
        embeddings: Normalized embedding matrix.
        num_clusters: Optional explicit cluster count.

    Returns:
        ``(labels, optimal_k, silhouette_score)``.

    Raises:
        ValueError: If the inputs are invalid.
    """

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array.")

    sample_count = embeddings.shape[0]
    if sample_count < 2:
        return np.zeros(sample_count, dtype=int), 1, 0.0

    if num_clusters is not None:
        if num_clusters < 1:
            raise ValueError("num_clusters must be at least 1.")
        optimal_k = min(num_clusters, sample_count)
        score = 0.0
    else:
        optimal_k, score = get_optimal_k(embeddings)

    if optimal_k <= 1:
        labels = np.zeros(sample_count, dtype=int)
        return labels, 1, score

    model = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, n_init=N_INIT)
    labels = model.fit_predict(embeddings)
    logger.info("Clustered %s conversations into %s groups", sample_count, optimal_k)
    return labels, optimal_k, score
