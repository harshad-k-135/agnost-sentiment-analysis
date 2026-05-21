import json

from fastapi.testclient import TestClient

import main
import clustering


def dummy_embed(texts):
    # simple deterministic embeddings
    return [[float(i)] * 4 for i in range(len(texts))]


def test_analyze_endpoint(monkeypatch):
    monkeypatch.setattr(clustering, "embed_conversations", lambda x: __import__("numpy").array(dummy_embed(x)))
    client = TestClient(main.app)

    payload = {"conversations": ["refund please", "I want a refund", "feature request: dark mode"]}
    resp = client.post("/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "clusters" in data and data["total_conversations"] == 3


def test_store_conversations_and_get_insights(monkeypatch):
    # Monkeypatch embeddings so /analyze doesn't attempt model download during other tests
    monkeypatch.setattr(clustering, "embed_conversations", lambda x: __import__("numpy").array(dummy_embed(x)))
    client = TestClient(main.app)

    payload = {"conversations": ["a test conversation"]}
    resp = client.post("/conversations", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_conversations"] == 1

    # insights endpoint should return a list (may be empty)
    resp2 = client.get("/insights")
    assert resp2.status_code == 200
    assert isinstance(resp2.json().get("insights"), list)
