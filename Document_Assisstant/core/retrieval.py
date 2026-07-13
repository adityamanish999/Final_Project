"""Phase 4: retrieval orchestration — FAISS search followed by cross-encoder re-ranking."""

from core import config
from core.embeddings import Embedder
from core.reranker import Reranker
from core.vector_store import VectorStore


def retrieve_and_rank(
    query: str,
    store: VectorStore,
    reranker: Reranker,
    embedder: Embedder | None = None,
    fetch_k: int = config.DEFAULT_FETCH_K,
    top_k: int = config.DEFAULT_TOP_K,
) -> list[dict]:
    """Over-fetches candidates from FAISS, then re-ranks with a cross-encoder.

    Returns a list of {"text","page","source","score","rerank_score","id"} dicts,
    ordered by rerank_score descending, truncated to top_k.
    """
    embedder = embedder or store.embedder
    query_vector = embedder.encode_normalized([query])
    if len(query_vector) == 0:
        return []

    candidates = store.search(query_vector[0], top_k=fetch_k)
    if not candidates:
        return []

    return reranker.rerank(query, candidates, top_n=top_k)
