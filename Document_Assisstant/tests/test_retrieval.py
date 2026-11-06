from core import retrieval, vector_store
from tests.conftest import make_chunks


def test_retrieve_returns_reranked_top_k(fake_store):
    vector_store.add_document(make_chunks("doc1.pdf", n=5), file_hash="hash1", filename="doc1.pdf")

    results = retrieval.retrieve("chunk 0 from doc1.pdf", top_k=2, fetch_k=5)

    assert len(results) == 2
    assert all("rerank_score" in r for r in results)


def test_retrieve_empty_store_returns_empty(fake_store):
    assert retrieval.retrieve("anything") == []
