"""Steps 1-2: read a PDF, clean its text, and cut it into chunks.

Step 1 (reading): pypdf gives us the text of each page. We remove lines that repeat on most
pages (running headers/footers) and lines that are only a page number, then squash extra
whitespace. We also compute a hash of the file so we can tell if it was already ingested.

Step 2 (chunking): each page's text is cut into fixed-size pieces that overlap a little.
Chunks never cross a page boundary, so every chunk knows its exact page number for citations.
"""

import hashlib
import re
from collections import Counter
from pathlib import Path

from pypdf import PdfReader

from core import config

# ---------------------------------------------------------------------------
# Step 1: read and clean a PDF
# ---------------------------------------------------------------------------


def file_hash(pdf_path) -> str:
    """A fingerprint of the file's contents. Same file -> same hash, even if renamed."""
    return hashlib.sha256(Path(pdf_path).read_bytes()).hexdigest()


def extract_pages(pdf_path) -> list[dict]:
    """Raw text of every page, as [{"page": 1, "text": "..."}, ...]."""
    reader = PdfReader(str(pdf_path))
    pages = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append({"page": page_num, "text": text})
    return pages


def remove_repeated_lines(pages: list[dict]) -> list[dict]:
    """Removes running headers/footers.

    A line that appears on at least half of the pages (and on at least 3 pages) is almost
    certainly a header or footer like "Berkshire Hathaway Inc." rather than real content.
    """
    if len(pages) < 3:
        return pages

    # Count on how many pages each line appears (set() so a line counts once per page).
    line_counts = Counter()
    for page in pages:
        lines = {line.strip() for line in page["text"].splitlines() if line.strip()}
        line_counts.update(lines)

    min_pages = max(3, len(pages) // 2)
    repeated = {line for line, count in line_counts.items() if count >= min_pages}

    cleaned = []
    for page in pages:
        kept = [
            line.strip() for line in page["text"].splitlines()
            if line.strip() and line.strip() not in repeated
        ]
        cleaned.append({"page": page["page"], "text": "\n".join(kept)})
    return cleaned


def clean_text(text: str) -> str:
    """Drops lines that are only a page number, then collapses all whitespace to single spaces."""
    lines = [line for line in text.splitlines() if not re.fullmatch(r"\d{1,4}", line.strip())]
    return re.sub(r"\s+", " ", "\n".join(lines)).strip()


def load_document(pdf_path) -> tuple[list[dict], str]:
    """Reads and cleans a PDF.

    Returns (pages, hash) where pages is [{"page": 1, "text": "...", "source": "file.pdf"}, ...].
    """
    source = Path(pdf_path).name
    pages = remove_repeated_lines(extract_pages(pdf_path))
    pages = [
        {"page": p["page"], "text": clean_text(p["text"]), "source": source}
        for p in pages
    ]
    return pages, file_hash(pdf_path)


# ---------------------------------------------------------------------------
# Step 2: chunking
# ---------------------------------------------------------------------------


def chunk_text(
    text: str,
    page: int,
    source: str,
    chunk_size: int = config.DEFAULT_CHUNK_SIZE,
    overlap: int = config.DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    """Cuts one page of text into pieces of `chunk_size` characters.

    Each new chunk starts `chunk_size - overlap` characters after the previous one, so
    neighbouring chunks share `overlap` characters and a sentence cut in half still appears
    whole in one of them.
    """
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(text), step):
        chunks.append({
            "text": text[start:start + chunk_size],
            "page": page,
            "source": source,
        })
    return chunks


def chunk_document(
    pages: list[dict],
    chunk_size: int = config.DEFAULT_CHUNK_SIZE,
    overlap: int = config.DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    """Chunks every page separately and returns all the chunks in one list."""
    chunks = []
    for page in pages:
        chunks.extend(chunk_text(page["text"], page["page"], page["source"], chunk_size, overlap))
    return chunks
