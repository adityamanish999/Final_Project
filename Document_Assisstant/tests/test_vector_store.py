import numpy as np
import pytest

from core.vector_store import EmbeddingModelMismatchError, VectorStore


class FakeEmbedder:
    """Deterministic hash-based embedder so tests don't need to download a real model."""

    model_name = "fake-embedder"
    dimension = 8

    def encode_normalized(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for i, text in enumerate(texts):
            for j, ch in enumerate(text[: self.dimension]):
                vectors[i, j] = (ord(ch) % 13) / 13.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


def make_chunks(source: str, n: int = 3) -> list[dict]:
    return [
        {"text": f"chunk {i} from {source}", "page": i + 1, "source": source, "chunk_size": 100, "overlap": 10}
        for i in range(n)
    ]


def test_add_document_and_search_roundtrip(tmp_path):
    store = VectorStore(store_dir=tmp_path, embedder=FakeEmbedder())
    chunks = make_chunks("doc1.pdf")

    added = store.add_document(chunks, file_hash="hash1", filename="doc1.pdf")
    assert added == 3
    assert store.is_ingested("hash1")

    query_vec = FakeEmbedder().encode_normalized(["chunk 0 from doc1.pdf"])[0]
    results = store.search(query_vec, top_k=2)
    assert len(results) == 2
    assert results[0]["source"] == "doc1.pdf"


def test_add_document_is_idempotent_for_same_hash(tmp_path):
    store = VectorStore(store_dir=tmp_path, embedder=FakeEmbedder())
    chunks = make_chunks("doc1.pdf")

    store.add_document(chunks, file_hash="hash1", filename="doc1.pdf")
    added_again = store.add_document(chunks, file_hash="hash1", filename="doc1.pdf")

    assert added_again == 0
    assert store.index.ntotal == 3


def test_save_and_load_persists_across_instances(tmp_path):
    embedder = FakeEmbedder()
    store = VectorStore(store_dir=tmp_path, embedder=embedder)
    store.add_document(make_chunks("doc1.pdf"), file_hash="hash1", filename="doc1.pdf")
    store.save()

    reloaded = VectorStore(store_dir=tmp_path, embedder=embedder)
    assert reloaded.index.ntotal == 3
    assert reloaded.is_ingested("hash1")
    assert reloaded.list_documents()[0]["filename"] == "doc1.pdf"


def test_embedding_model_mismatch_raises(tmp_path):
    store = VectorStore(store_dir=tmp_path, embedder=FakeEmbedder())
    store.add_document(make_chunks("doc1.pdf"), file_hash="hash1", filename="doc1.pdf")
    store.save()

    class OtherEmbedder(FakeEmbedder):
        model_name = "different-model"
        dimension = 16

    with pytest.raises(EmbeddingModelMismatchError):
        VectorStore(store_dir=tmp_path, embedder=OtherEmbedder())


def test_clear_resets_store(tmp_path):
    store = VectorStore(store_dir=tmp_path, embedder=FakeEmbedder())
    store.add_document(make_chunks("doc1.pdf"), file_hash="hash1", filename="doc1.pdf")
    store.save()

    store.clear()
    assert store.index.ntotal == 0
    assert store.list_documents() == []
    assert not store.is_ingested("hash1")
