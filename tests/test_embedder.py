"""Unit tests for the public embedding helpers."""

from __future__ import annotations

import numpy as np

import clustering


class DummyModel:
    """Deterministic embedding model used to keep tests offline."""

    def encode(self, texts, show_progress_bar=False, normalize_embeddings=True):  # noqa: D401
        return np.asarray([[float(index)] * 4 for index, _ in enumerate(texts)], dtype=float)


def test_normalize_conversations() -> None:
    """Whitespace-only rows should be removed and content normalized."""

    raw = ["  hello  ", "", "world\n\n", "  "]
    assert clustering.normalize_conversations(raw) == ["hello", "world"]


def test_embed_conversations_uses_injected_model(monkeypatch) -> None:
    """The embedding helper should work with a monkeypatched model loader."""

    monkeypatch.setattr(clustering, "load_embedding_model", lambda: DummyModel())
    embeddings = clustering.embed_conversations(["alpha", "beta", "gamma"])
    assert embeddings.shape == (3, 4)
    assert embeddings[1, 0] == 1.0
