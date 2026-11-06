"""Shared test setup: swap the real ML models for tiny fakes and use a temporary store folder.

Tests therefore need no network, no API key and no model download.
"""

import numpy as np
import pytest

from core import config, retrieval, vector_store

FAKE_DIMENSION = 8


def fake_embed(texts: list[str]) -> np.ndarray:
    """Deterministic 'embedding' built from the first 8 characters of each text, normalized."""
    vectors = np.zeros((len(texts), FAKE_DIMENSION), dtype=np.float32)
    for i, text in enumerate(texts):
        for j, ch in enumerate(text[:FAKE_DIMENSION]):
            vectors[i, j] = (ord(ch) % 13) / 13.0
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def fake_rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """Reverses the FAISS order, so tests can tell that re-ranking actually ran."""
    reranked = list(reversed(candidates))
    return [{**c, "rerank_score": float(len(reranked) - i)} for i, c in enumerate(reranked)][:top_k]


def make_chunks(source: str, n: int = 3) -> list[dict]:
    return [{"text": f"chunk {i} from {source}", "page": i + 1, "source": source} for i in range(n)]


@pytest.fixture
def fake_store(tmp_path, monkeypatch):
    """An empty vector store in a temp folder, using the fake embedder and reranker."""
    monkeypatch.setattr(vector_store, "embed", fake_embed)
    monkeypatch.setattr(vector_store, "embedding_dimension", lambda: FAKE_DIMENSION)
    monkeypatch.setattr(config, "EMBEDDING_MODEL_NAME", "fake-embedder")
    monkeypatch.setattr(retrieval, "rerank", fake_rerank)
    vector_store.load_store(tmp_path)
    return tmp_path
