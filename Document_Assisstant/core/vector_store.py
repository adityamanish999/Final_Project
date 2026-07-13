"""Phase 3: persistent FAISS vector store with dedup and incremental add."""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

from core import config
from core.embeddings import Embedder


class EmbeddingModelMismatchError(Exception):
    """Raised when the persisted store was built with a different embedding model."""


class VectorStore:
    def __init__(self, store_dir: Path = config.VECTOR_STORE_DIR, embedder: Embedder | None = None):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or Embedder()

        self._index_path = self.store_dir / "index.faiss"
        self._metadata_path = self.store_dir / "metadata.json"
        self._manifest_path = self.store_dir / "manifest.json"

        self.index: faiss.Index
        self.metadata: list[dict] = []
        self.manifest: dict = {}

        self.load()

    def load(self) -> None:
        if self._index_path.exists() and self._metadata_path.exists() and self._manifest_path.exists():
            manifest = json.loads(self._manifest_path.read_text())
            if manifest.get("embedding_model") != self.embedder.model_name or manifest.get("dimension") != self.embedder.dimension:
                raise EmbeddingModelMismatchError(
                    f"Persisted store was built with model={manifest.get('embedding_model')!r} "
                    f"dim={manifest.get('dimension')}, but the current embedder is "
                    f"model={self.embedder.model_name!r} dim={self.embedder.dimension}. "
                    "Either revert to the original embedding model, or clear the vector store "
                    "(core.vector_store.VectorStore.clear) and re-ingest your documents."
                )
            self.index = faiss.read_index(str(self._index_path))
            self.metadata = json.loads(self._metadata_path.read_text())
            self.manifest = manifest
        else:
            self.index = faiss.IndexFlatIP(self.embedder.dimension)
            self.metadata = []
            self.manifest = {
                "embedding_model": self.embedder.model_name,
                "dimension": self.embedder.dimension,
                "ingested_files": {},
            }

    def is_ingested(self, hash_: str) -> bool:
        return hash_ in self.manifest.get("ingested_files", {})

    def add_document(self, chunks: list[dict], file_hash: str, filename: str) -> int:
        """Embeds and appends chunks. No-ops if file_hash is already ingested. Returns count added."""
        if self.is_ingested(file_hash):
            return 0
        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        vectors = self.embedder.encode_normalized(texts)
        start_id = len(self.metadata)

        self.index.add(vectors)
        for i, chunk in enumerate(chunks):
            self.metadata.append({
                "id": start_id + i,
                "text": chunk["text"],
                "page": chunk["page"],
                "source": chunk["source"],
                "chunk_size": chunk.get("chunk_size"),
                "overlap": chunk.get("overlap"),
            })

        self.manifest.setdefault("ingested_files", {})[file_hash] = {
            "filename": filename,
            "num_chunks": len(chunks),
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        }
        return len(chunks)

    def save(self) -> None:
        self._atomic_write_index()
        self._atomic_write_json(self._metadata_path, self.metadata)
        self._atomic_write_json(self._manifest_path, self.manifest)

    def _atomic_write_index(self) -> None:
        fd, tmp_path = tempfile.mkstemp(dir=self.store_dir, suffix=".faiss.tmp")
        os.close(fd)
        faiss.write_index(self.index, tmp_path)
        os.replace(tmp_path, self._index_path)

    def _atomic_write_json(self, path: Path, data) -> None:
        fd, tmp_path = tempfile.mkstemp(dir=self.store_dir, suffix=".json.tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[dict]:
        if self.index.ntotal == 0:
            return []
        query_vector = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        scores, idxs = self.index.search(query_vector, min(top_k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append({**self.metadata[idx], "score": float(score)})
        return results

    def list_documents(self) -> list[dict]:
        return [
            {"file_hash": h, **info}
            for h, info in self.manifest.get("ingested_files", {}).items()
        ]

    def clear(self) -> None:
        self.index = faiss.IndexFlatIP(self.embedder.dimension)
        self.metadata = []
        self.manifest = {
            "embedding_model": self.embedder.model_name,
            "dimension": self.embedder.dimension,
            "ingested_files": {},
        }
        self.save()
