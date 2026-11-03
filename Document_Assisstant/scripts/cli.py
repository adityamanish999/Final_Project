"""Command-line helpers: download the sample PDFs, and ingest PDFs into the vector store.

Usage:
    python scripts/cli.py download
    python scripts/cli.py ingest data/sample_pdfs/*.pdf
    python scripts/cli.py ingest path/to/one.pdf path/to/another.pdf

Note on the sample data: SEC EDGAR's raw 10-K filings are almost all served as HTML, not PDF,
so instead of scraping EDGAR's filing index `download` pulls from a small curated list of
stable, directly downloadable PDF annual reports / shareholder letters. This still satisfies
the "Public Annual Reports Dataset" spirit of the assignment (real, public, company-published
annual-report documents) while guaranteeing the files are actually PDFs pypdf can parse.
"""

import argparse
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import config  # noqa: E402

SAMPLE_DOCS = {
    "berkshire_2023_annual_report.pdf": "https://www.berkshirehathaway.com/2023ar/2023ar.pdf",
    "berkshire_2023_shareholder_letter.pdf": "https://www.berkshirehathaway.com/letters/2023ltr.pdf",
    "berkshire_2022_shareholder_letter.pdf": "https://www.berkshirehathaway.com/letters/2022ltr.pdf",
}


def download_sample_pdfs(dest_dir: Path = config.SAMPLE_PDFS_DIR) -> list[Path]:
    """Downloads SAMPLE_DOCS into dest_dir, skipping files already present."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for filename, url in SAMPLE_DOCS.items():
        dest_path = dest_dir / filename
        if dest_path.exists():
            print(f"skip (already present): {filename}")
            downloaded.append(dest_path)
            continue
        print(f"downloading: {filename} <- {url}")
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        response.raise_for_status()
        dest_path.write_bytes(response.content)
        downloaded.append(dest_path)
    return downloaded


def ingest_pdfs(pdf_paths: list[str]) -> None:
    """Ingests each PDF into the persistent vector store (idempotent per file content)."""
    from core.pipeline import DocumentAssistant  # loads the models, so only import when needed

    assistant = DocumentAssistant()
    for pdf_path in pdf_paths:
        result = assistant.ingest_pdf(pdf_path)
        if result["skipped"]:
            print(f"skipped (already ingested): {result['filename']}")
        else:
            print(f"ingested: {result['filename']} ({result['added_chunks']} chunks)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Document Assistant command-line helpers.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download", help="Download the sample Berkshire Hathaway PDFs.")
    ingest = commands.add_parser("ingest", help="Ingest PDFs into the persistent vector store.")
    ingest.add_argument("pdf_paths", nargs="+", help="Path(s) to PDF file(s) to ingest.")
    args = parser.parse_args()

    if args.command == "download":
        paths = download_sample_pdfs()
        print(f"\n{len(paths)} sample PDFs available in {config.SAMPLE_PDFS_DIR}")
    else:
        ingest_pdfs(args.pdf_paths)


if __name__ == "__main__":
    main()
