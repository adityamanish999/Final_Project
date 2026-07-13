"""Phase 4: re-ranking retrieved candidates with a cross-encoder."""

from sentence_transformers import CrossEncoder

from core import config


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
