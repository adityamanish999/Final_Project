"""Phases 1-2: PDF text extraction, cleaning, and chunking.

Phase 1 (ingestion): pypdf extracts per-page text, repeated headers/footers and page-number-only
lines are removed, whitespace is normalized, and a sha256 file hash is computed for dedup.

Phase 2 (chunking): fixed-size character windows with overlap, applied per page so chunks never
cross page boundaries and every chunk keeps an exact page number for citations.
"""

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from core import config

# ---------------------------------------------------------------------------
# Phase 1: ingestion
# ---------------------------------------------------------------------------

_PAGE_NUMBER_LINE = re.compile(r"^\d{1,4}$")
_WHITESPACE = re.compile(r"\s+")


def file_hash(pdf_path: str | Path) -> str:
    """sha256 of file contents, used to detect already-ingested documents."""
    digest = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_pages(pdf_path: str | Path) -> list[dict]:
    """Raw per-page text via pypdf. Returns [{"page": int, "text": str}]."""
    reader = PdfReader(str(pdf_path))
    pages = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append({"page": page_num, "text": text})
    return pages


def _strip_repeated_boilerplate(pages: list[dict]) -> list[dict]:
    """Drops lines that repeat identically across most pages (running headers/footers)."""
    if len(pages) < 3:
        return pages

    line_counts = Counter()
    per_page_lines = []
    for page in pages:
        lines = [line.strip() for line in page["text"].splitlines() if line.strip()]
        per_page_lines.append(lines)
        line_counts.update(set(lines))

    threshold = max(3, int(len(pages) * 0.5))
    boilerplate = {line for line, count in line_counts.items() if count >= threshold}

    cleaned = []
    for page, lines in zip(pages, per_page_lines):
        kept = [line for line in lines if line not in boilerplate]
        cleaned.append({"page": page["page"], "text": "\n".join(kept)})
    return cleaned


def clean_text(text: str) -> str:
    """Whitespace normalization plus dropping page-number-only lines."""
    lines = [line for line in text.splitlines() if not _PAGE_NUMBER_LINE.match(line.strip())]
    joined = "\n".join(lines)
    return _WHITESPACE.sub(" ", joined).strip()


def load_document(pdf_path: str | Path) -> tuple[list[dict], str]:
    """Extracts, de-boilerplates, and cleans a PDF's pages.

    Returns (pages, hash) where pages is [{"page", "text", "source"}] with cleaned text.
    """
    source = Path(pdf_path).name
    raw_pages = extract_pages(pdf_path)
    deboilerplated = _strip_repeated_boilerplate(raw_pages)
    pages = [
        {"page": p["page"], "text": clean_text(p["text"]), "source": source}
        for p in deboilerplated
    ]
    return pages, file_hash(pdf_path)


# ---------------------------------------------------------------------------
# Phase 2: chunking
# ---------------------------------------------------------------------------


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
