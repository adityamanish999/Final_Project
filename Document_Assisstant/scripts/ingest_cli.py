"""CLI to ingest one or more PDFs into the persistent vector store.

Usage:
    python scripts/ingest_cli.py path/to/one.pdf path/to/another.pdf
    python scripts/ingest_cli.py data/sample_pdfs/*.pdf
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.pipeline import DocumentAssistant  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDFs into the persistent vector store.")
    parser.add_argument("pdf_paths", nargs="+", help="Path(s) to PDF file(s) to ingest.")
    args = parser.parse_args()

    assistant = DocumentAssistant()
    for pdf_path in args.pdf_paths:
        result = assistant.ingest_pdf(pdf_path)
        if result["skipped"]:
            print(f"skipped (already ingested): {result['filename']}")
        else:
            print(f"ingested: {result['filename']} ({result['added_chunks']} chunks)")


if __name__ == "__main__":
    main()
