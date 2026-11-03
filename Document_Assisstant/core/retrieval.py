"""Phase 4: retrieval and ranking — FAISS search followed by cross-encoder re-ranking."""

from sentence_transformers import CrossEncoder

from core import config
from core.vector_store import Embedder, VectorStore


class Reranker:
    def __init__(self, model_name: str = config.RERANKER_MODEL_NAME):
        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[dict], top_n: int) -> list[dict]:
        """Scores each (query, candidate.text) pair and returns the top_n by rerank score."""
        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]
        scores = self._model.predict(pairs)

        ranked = sorted(
            (
                {**candidate, "rerank_score": float(score)}
                for candidate, score in zip(candidates, scores)
            ),
            key=lambda c: c["rerank_score"],
            reverse=True,
        )
        return ranked[:top_n]


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
