import numpy as np
import clustering


def dummy_model():
    class M:
        def encode(self, texts, show_progress_bar=False, normalize_embeddings=True):
            # deterministic embeddings: each text becomes vector [i, i, i]
            return np.array([[float(i)] * 3 for i in range(len(texts))], dtype=float)

    return M()


def test_normalize_conversations():
    raw = ["  hello  ", "", "world\n\n", "  "]
    out = clustering.normalize_conversations(raw)
    assert out == ["hello", "world"]


def test_embed_and_cluster_monkeypatch(monkeypatch):
    texts = ["a", "b", "c", "d"]
    # monkeypatch model loader so tests don't download models
    monkeypatch.setattr(clustering, "load_embedding_model", lambda: dummy_model())
    labels, embeddings, optimal_k, silhouette_score = clustering.cluster_conversations(texts)
    assert len(labels) == len(texts)
    assert embeddings.shape[0] == len(texts)
    assert optimal_k >= 1
    assert silhouette_score >= 0.0


def test_extract_top_keywords():
    texts = ["refund please", "need refund", "refund now", "feature request"]
    keywords = clustering.extract_top_keywords(texts)
    assert any("refund" in kw for kw in keywords)


def test_build_cluster_results_monkeypatch(monkeypatch):
    texts = ["refund 1", "refund 2", "feature x", "feature y"]
    # simple embeddings: two related groups, but with unique rows so clustering stays warning-free
    monkeypatch.setattr(
        clustering,
        "embed_conversations",
        lambda x: np.array([[0.0, 0.0], [0.2, 0.2], [1.0, 1.0], [1.2, 1.2]]),
    )
    labels, _, _, _ = clustering.cluster_conversations(texts)
    results = clustering.build_cluster_results(texts, labels)
    assert isinstance(results, list)
    assert len(results) >= 1