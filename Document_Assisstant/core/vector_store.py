"""Step 3: turn text into vectors (embeddings) and keep them in a FAISS index saved on disk.

How it works:
- `embed()` uses a small local model (all-MiniLM-L6-v2) to turn each text into 384 numbers.
  The vectors are normalized to length 1, so FAISS "inner product" equals cosine similarity.
- `store` is a plain dict holding everything in memory: the FAISS index, the list of chunks
  (chunk number i in the list == vector number i in the index), and the files ingested so far.
- `save_store()` / `load_store()` write and read three files in data/vector_store/:
    index.faiss    the vectors
    metadata.json  the chunks (text, page, source)
    manifest.json  which embedding model was used + which files were ingested (by hash)
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from core import config

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

_embedding_model = None  # loaded on first use, then reused


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    return _embedding_model


def embedding_dimension() -> int:
    """How many numbers each vector has (384 for all-MiniLM-L6-v2)."""
    model = get_embedding_model()
    # Newer sentence-transformers versions renamed this method, so support both names.
    if hasattr(model, "get_embedding_dimension"):
        return model.get_embedding_dimension()
    return model.get_sentence_embedding_dimension()


def embed(texts: list[str]) -> np.ndarray:
    """Turns a list of texts into a (len(texts), dimension) array of normalized vectors."""
    if not texts:
        return np.empty((0, embedding_dimension()), dtype=np.float32)
    vectors = get_embedding_model().encode(texts, convert_to_numpy=True, show_progress_bar=False)
    vectors = vectors.astype(np.float32)
    faiss.normalize_L2(vectors)
    return vectors


# ---------------------------------------------------------------------------
# The vector store
# ---------------------------------------------------------------------------

# Everything the store knows lives in this one dict. It is filled by load_store().
store = {
    "dir": None,      # folder the files are saved in
    "index": None,    # the FAISS index (None until loaded)
    "chunks": [],     # chunk dicts, in the same order as the vectors in the index
    "files": {},      # {file_hash: {"filename", "num_chunks", "ingested_at"}}
}


def _file_paths(store_dir: Path) -> tuple[Path, Path, Path]:
    return store_dir / "index.faiss", store_dir / "metadata.json", store_dir / "manifest.json"


def load_store(store_dir=config.VECTOR_STORE_DIR) -> None:
    """Loads the store from disk, or starts an empty one if nothing has been saved yet."""
    store_dir = Path(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)
    index_path, metadata_path, manifest_path = _file_paths(store_dir)
    store["dir"] = store_dir

    if not (index_path.exists() and metadata_path.exists() and manifest_path.exists()):
        store["index"] = faiss.IndexFlatIP(embedding_dimension())
        store["chunks"] = []
        store["files"] = {}
        return

    manifest = json.loads(manifest_path.read_text())

    # Vectors from different embedding models can't be compared with each other, so refuse to
    # load a store that was built with another model.
    saved_model = manifest.get("embedding_model")
    if saved_model != config.EMBEDDING_MODEL_NAME:
        raise ValueError(
            f"The vector store in {store_dir} was built with the embedding model "
            f"{saved_model!r}, but config.EMBEDDING_MODEL_NAME is "
            f"{config.EMBEDDING_MODEL_NAME!r}. Change the model back, or clear the store and "
            "re-ingest your documents."
        )

    store["index"] = faiss.read_index(str(index_path))
    store["chunks"] = json.loads(metadata_path.read_text())
    store["files"] = manifest.get("ingested_files", {})


def _make_sure_loaded() -> None:
    if store["index"] is None:
        load_store()


def save_store() -> None:
    """Writes the index, chunks and manifest to disk."""
    _make_sure_loaded()
    index_path, metadata_path, manifest_path = _file_paths(store["dir"])
    manifest = {
        "embedding_model": config.EMBEDDING_MODEL_NAME,
        "dimension": store["index"].d,
        "ingested_files": store["files"],
    }
    faiss.write_index(store["index"], str(index_path))
    metadata_path.write_text(json.dumps(store["chunks"], indent=2))
    manifest_path.write_text(json.dumps(manifest, indent=2))


def is_ingested(file_hash: str) -> bool:
    _make_sure_loaded()
    return file_hash in store["files"]


def add_document(chunks: list[dict], file_hash: str, filename: str) -> int:
    """Embeds the chunks and adds them to the index. Returns how many chunks were added.

    Does nothing (returns 0) if this file was already ingested. Call save_store() afterwards
    to write the change to disk.
    """
    _make_sure_loaded()
    if is_ingested(file_hash) or not chunks:
        return 0

    vectors = embed([c["text"] for c in chunks])
    store["index"].add(vectors)

    first_id = len(store["chunks"])
    for i, chunk in enumerate(chunks):
        store["chunks"].append({
            "id": first_id + i,
            "text": chunk["text"],
            "page": chunk["page"],
            "source": chunk["source"],
        })

    store["files"][file_hash] = {
        "filename": filename,
        "num_chunks": len(chunks),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }
    return len(chunks)


def search(query: str, top_k: int) -> list[dict]:
    """Finds the `top_k` chunks most similar to the query.

    Returns chunk dicts with an added "score" (cosine similarity, higher is more similar).
    """
    _make_sure_loaded()
    index = store["index"]
    if index.ntotal == 0:
        return []

    query_vector = embed([query])
    scores, ids = index.search(query_vector, min(top_k, index.ntotal))

    results = []
    for score, chunk_id in zip(scores[0], ids[0]):
        if chunk_id == -1:  # FAISS uses -1 for "no result"
            continue
        results.append({**store["chunks"][chunk_id], "score": float(score)})
    return results


def list_documents() -> list[dict]:
    """All ingested files, as [{"file_hash", "filename", "num_chunks", "ingested_at"}, ...]."""
    _make_sure_loaded()
    return [{"file_hash": h, **info} for h, info in store["files"].items()]


def clear_store() -> None:
    """Deletes every document from the store (in memory and on disk)."""
    _make_sure_loaded()
    store["index"] = faiss.IndexFlatIP(embedding_dimension())
    store["chunks"] = []
    store["files"] = {}
    save_store()
