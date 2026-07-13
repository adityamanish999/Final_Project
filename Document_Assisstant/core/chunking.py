"""Phase 2: document chunking."""

from dataclasses import dataclass

from core import config


@dataclass(frozen=True)
class ChunkConfig:
    chunk_size: int = config.DEFAULT_CHUNK_SIZE
    overlap: int = config.DEFAULT_CHUNK_OVERLAP
    strategy: str = "fixed"


def chunk_text(text: str, page: int, source: str, chunk_config: ChunkConfig) -> list[dict]:
    """Fixed-size character windowing with overlap. Chunks never cross page boundaries."""
    if chunk_config.strategy != "fixed":
        raise ValueError(f"Unsupported chunking strategy: {chunk_config.strategy}")

    chunks = []
    start = 0
    step = chunk_config.chunk_size - chunk_config.overlap
    while start < len(text):
        piece = text[start:start + chunk_config.chunk_size]
        if piece:
            chunks.append({
                "text": piece,
                "page": page,
                "source": source,
                "chunk_size": chunk_config.chunk_size,
                "overlap": chunk_config.overlap,
            })
        start += step
    return chunks


def chunk_document(pages: list[dict], chunk_config: ChunkConfig | None = None) -> list[dict]:
    """Applies chunk_text per page and concatenates. `pages` is [{"page","text","source"}]."""
    chunk_config = chunk_config or ChunkConfig()
    chunks = []
    for page in pages:
        chunks.extend(
            chunk_text(page["text"], page["page"], page["source"], chunk_config)
        )
    return chunks
