"""Ingestion: the full path from a PDF file to searchable chunks in the vector store.

    read + clean (documents) -> chunk (documents) -> embed + store (vector_store) -> save to disk
"""

from pathlib import Path

from core import config, vector_store
from core.documents import chunk_document, load_document


def ingest_pdf(
    pdf_path,
    display_name: str | None = None,
    chunk_size: int = config.DEFAULT_CHUNK_SIZE,
    overlap: int = config.DEFAULT_CHUNK_OVERLAP,
) -> dict:
    """Adds one PDF to the vector store. Skips it if the same file was ingested before.

    `display_name` is the filename to show in citations. Use it when `pdf_path` is a temporary
    file (e.g. a Streamlit upload) so answers cite the original name.

    Returns {"filename": str, "added_chunks": int, "skipped": bool}.
    """
    pages, file_hash = load_document(pdf_path)
    filename = display_name or Path(pdf_path).name

    if vector_store.is_ingested(file_hash):
        return {"filename": filename, "added_chunks": 0, "skipped": True}

    chunks = chunk_document(pages, chunk_size, overlap)
    for chunk in chunks:
        chunk["source"] = filename

    added = vector_store.add_document(chunks, file_hash, filename)
    vector_store.save_store()
    return {"filename": filename, "added_chunks": added, "skipped": False}
