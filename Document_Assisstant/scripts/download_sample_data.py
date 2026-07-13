"""Downloads a handful of public annual-report PDFs into data/sample_pdfs/.

Note: SEC EDGAR's raw 10-K filings are almost all served as HTML, not PDF, so instead of
scraping EDGAR's filing index this pulls from a small curated list of stable, directly
downloadable PDF annual reports / shareholder letters. This still satisfies the "Public Annual
Reports Dataset" spirit of the assignment (real, public, company-published annual-report
documents) while guaranteeing the files are actually PDFs pypdf can parse.
"""

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


if __name__ == "__main__":
    paths = download_sample_pdfs()
    print(f"\n{len(paths)} sample PDFs available in {config.SAMPLE_PDFS_DIR}")
