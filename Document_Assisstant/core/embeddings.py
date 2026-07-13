"""Phase 3: embedding generation."""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from core import config


class Embedder:
    def __init__(self, model_name: str = config.EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        if hasattr(self._model, "get_embedding_dimension"):
            return self._model.get_embedding_dimension()
        return self._model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        return self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def encode_normalized(self, texts: list[str]) -> np.ndarray:
        """L2-normalized embeddings so FAISS inner product == cosine similarity."""
        vectors = self.encode(texts).astype(np.float32)
        if len(vectors):
            faiss.normalize_L2(vectors)
        return vectors
