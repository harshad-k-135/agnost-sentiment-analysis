"""Integration tests for the FastAPI contract."""

from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

import clustering
import main


class DummyModel:
    """Deterministic embedding model used to keep API tests offline."""

    def encode(self, texts, show_progress_bar=False, normalize_embeddings=True):  # noqa: D401
        return np.asarray([[float(index)] * 4 for index, _ in enumerate(texts)], dtype=float)


def _client(monkeypatch) -> TestClient:
    monkeypatch.setattr(clustering, "load_embedding_model", lambda: DummyModel())
    return TestClient(main.app)


def test_analyze_endpoint(monkeypatch) -> None:
    """The analyze endpoint should return the new structured response."""

    client = _client(monkeypatch)
    response = client.post(
        "/api/v1/analyze",
        json={
            "conversations": [
                "User: pricing is too high",
                "User: love the new dashboard",
                "User: bug when logging in",
            ],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["data"]["total_conversations"] == 3
    assert payload["data"]["clusters"]
    assert payload["data"]["metadata"]["clustering_algo"] == "kmeans"


def test_storage_and_cached_insights(monkeypatch) -> None:
    """Batch storage and cached insight retrieval should both succeed."""

    client = _client(monkeypatch)
    storage = client.post(
        "/api/v1/conversations/batch",
        json={"conversations": ["User: need a refund", "User: please add Slack integration"], "source": "support_chat"},
    )
    assert storage.status_code == 201
    assert storage.json()["stored_count"] == 2

    insights = client.get("/api/v1/insights", params={"limit": 10, "offset": 0})
    assert insights.status_code == 200
    body = insights.json()
    assert "insights" in body
    assert "pagination" in body
    assert body["cached"] is True
