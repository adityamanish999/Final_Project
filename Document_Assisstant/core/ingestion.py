"""Phase 1: PDF text extraction and cleaning."""

import hashlib
import re
from collections import Counter
from pathlib import Path

from pypdf import PdfReader

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
