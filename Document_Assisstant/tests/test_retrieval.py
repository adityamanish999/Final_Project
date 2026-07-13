from core.retrieval import retrieve_and_rank
from core.vector_store import VectorStore
from tests.test_vector_store import FakeEmbedder, make_chunks


class FakeReranker:
    """Reranks by reversing FAISS order, so tests can tell rerank actually ran."""

    def rerank(self, query, candidates, top_n):
        reranked = list(reversed(candidates))
        for i, c in enumerate(reranked):
            c["rerank_score"] = float(len(reranked) - i)
        return reranked[:top_n]


def test_retrieve_and_rank_returns_reranked_top_k(tmp_path):
    embedder = FakeEmbedder()
    store = VectorStore(store_dir=tmp_path, embedder=embedder)
    store.add_document(make_chunks("doc1.pdf", n=5), file_hash="hash1", filename="doc1.pdf")

    results = retrieve_and_rank(
        "chunk 0 from doc1.pdf",
        store,
        FakeReranker(),
        embedder=embedder,
        fetch_k=5,
        top_k=2,
    )

    assert len(results) == 2
    assert all("rerank_score" in r for r in results)


def test_retrieve_and_rank_empty_store_returns_empty(tmp_path):
    embedder = FakeEmbedder()
    store = VectorStore(store_dir=tmp_path, embedder=embedder)

    results = retrieve_and_rank("anything", store, FakeReranker(), embedder=embedder)
    assert results == []
