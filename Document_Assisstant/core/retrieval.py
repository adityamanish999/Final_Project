"""Step 4: find the best chunks for a question.

Two stages:
1. FAISS search: fast but rough. We grab more candidates than we need (`fetch_k`, e.g. 20).
2. Re-ranking: a cross-encoder model reads the question and each candidate together and gives
   a more accurate relevance score. We keep the best `top_k` (e.g. 5).
"""

from sentence_transformers import CrossEncoder

from core import config, vector_store

_reranker_model = None  # loaded on first use, then reused


def get_reranker_model() -> CrossEncoder:
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = CrossEncoder(config.RERANKER_MODEL_NAME)
    return _reranker_model


def rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """Scores every (query, chunk) pair and returns the `top_k` best, each with a "rerank_score"."""
    if not candidates:
        return []

    pairs = [(query, c["text"]) for c in candidates]
    scores = get_reranker_model().predict(pairs)

    scored = [{**c, "rerank_score": float(s)} for c, s in zip(candidates, scores)]
    scored.sort(key=lambda c: c["rerank_score"], reverse=True)
    return scored[:top_k]


def retrieve(query: str, top_k: int = config.DEFAULT_TOP_K, fetch_k: int = config.DEFAULT_FETCH_K) -> list[dict]:
    """FAISS search for `fetch_k` candidates, then re-rank them down to the best `top_k`."""
    candidates = vector_store.search(query, top_k=fetch_k)
    return rerank(query, candidates, top_k)
