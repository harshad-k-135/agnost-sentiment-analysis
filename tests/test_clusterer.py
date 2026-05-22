"""Unit tests for clustering and insight generation."""

from __future__ import annotations

import numpy as np

import clustering


class DummyModel:
    """Deterministic embedding model used to keep tests offline."""

    def encode(self, texts, show_progress_bar=False, normalize_embeddings=True):  # noqa: D401
        return np.asarray([[float(index)] * 4 for index, _ in enumerate(texts)], dtype=float)


def test_get_optimal_k_returns_reasonable_value() -> None:
    """The silhouette search should choose a valid k for clustered points."""

    embeddings = np.asarray([[0.0, 0.0], [0.1, 0.1], [1.0, 1.0], [1.1, 1.1]], dtype=float)
    optimal_k, score = clustering.get_optimal_k(embeddings, max_k=4)
    assert optimal_k in {2, 3}
    assert -1.0 <= score <= 1.0


def test_cluster_conversations_and_summaries(monkeypatch) -> None:
    """A small batch should produce aligned labels and readable summaries."""

    monkeypatch.setattr(clustering, "load_embedding_model", lambda: DummyModel())
    texts = ["refund please", "need refund", "love the product", "great support"]

    labels, embeddings, optimal_k, silhouette_score = clustering.cluster_conversations(texts)
    assert len(labels) == len(texts)
    assert embeddings.shape[0] == len(texts)
    assert optimal_k >= 1

    summaries = clustering.build_cluster_results(texts, labels, silhouette_score=silhouette_score)
    assert summaries
    assert all(summary.topic for summary in summaries)


def test_analyze_conversations_returns_metadata(monkeypatch) -> None:
    """The orchestration helper should return the public payload shape."""

    monkeypatch.setattr(clustering, "load_embedding_model", lambda: DummyModel())
    result = clustering.analyze_conversations(["pricing is too high", "love the tool", "bug on login"])
    assert result["total_conversations"] == 3
    assert "clusters" not in result
    assert result["cluster_results"]
