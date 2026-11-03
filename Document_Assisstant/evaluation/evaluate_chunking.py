"""Compares retrieval quality across multiple chunk_size/overlap configurations.

Builds an isolated, in-memory FAISS index per ChunkConfig over the given PDFs (the persistent
app vector store in data/vector_store/ is left untouched), runs the fixed query set
EVAL_QUERIES below, and returns a tidy DataFrame — one row per (config, query) — plus a
config-level summary. This is real, reusable module code: notebooks/demo.ipynb only calls into
it and displays the results, it does not reimplement any of this logic.
"""

import time
from statistics import mean

import faiss
import pandas as pd

from core.documents import ChunkConfig, chunk_document, load_document
from core.retrieval import Reranker
from core.vector_store import Embedder

# Fixed evaluation query set, tailored to the sample Berkshire Hathaway annual report /
# shareholder letter documents downloaded by `python scripts/cli.py download`.
# `expected_keywords` is an OPTIONAL list of any-of terms used only as a cheap relevance proxy
# (substring hit-rate) in the absence of a labeled ground-truth QA dataset — not an exhaustive
# or authoritative answer key.
EVAL_QUERIES = [
    {
        "query": "What does Warren Buffett say about Berkshire Hathaway's float from insurance operations?",
        "expected_keywords": ["float", "insurance"],
    },
    {
        "query": "How does Berkshire Hathaway think about capital allocation and acquisitions?",
        "expected_keywords": ["capital", "acquisition"],
    },
    {
        "query": "What does the letter say about Charlie Munger?",
        "expected_keywords": ["Munger"],
    },
    {
        "query": "What is Berkshire Hathaway's view on stock buybacks / share repurchases?",
        "expected_keywords": ["repurchase", "buyback", "shares"],
    },
    {
        "query": "How is Berkshire's investment in Apple discussed?",
        "expected_keywords": ["Apple"],
    },
    {
        "query": "What does the report say about BNSF railroad?",
        "expected_keywords": ["BNSF", "railroad"],
    },
    {
        "query": "What does the report say about Berkshire Hathaway Energy?",
        "expected_keywords": ["Energy"],
    },
    {
        "query": "What is Berkshire Hathaway's book value per share?",
        "expected_keywords": ["book value", "per share"],
    },
    {
        "query": "What risks or uncertainties does the report describe for the insurance business?",
        "expected_keywords": ["risk", "insurance"],
    },
    {
        "query": "What does the letter say about long-term shareholders and Berkshire's investment philosophy?",
        "expected_keywords": ["long-term", "shareholders"],
    },
]

DEFAULT_CONFIGS = [
    ChunkConfig(chunk_size=400, overlap=50),
    ChunkConfig(chunk_size=800, overlap=150),
    ChunkConfig(chunk_size=1200, overlap=200),
]


def _config_label(chunk_config: ChunkConfig) -> str:
    return f"{chunk_config.chunk_size}/{chunk_config.overlap}"


def build_in_memory_index(pdf_paths: list[str], chunk_config: ChunkConfig, embedder: Embedder):
    """Chunks + embeds `pdf_paths` under one config into a standalone FAISS index (not persisted)."""
    all_chunks = []
    for pdf_path in pdf_paths:
        pages, _ = load_document(pdf_path)
        all_chunks.extend(chunk_document(pages, chunk_config))

    texts = [c["text"] for c in all_chunks]
    vectors = embedder.encode_normalized(texts)
    index = faiss.IndexFlatIP(embedder.dimension)
    if len(vectors):
        index.add(vectors)
    return index, all_chunks


def _search(index, chunks: list[dict], query_vector, top_k: int) -> list[dict]:
    if index.ntotal == 0:
        return []
    scores, idxs = index.search(query_vector.reshape(1, -1), min(top_k, index.ntotal))
    return [
        {**chunks[i], "score": float(s)} for s, i in zip(scores[0], idxs[0]) if i != -1
    ]


def run_comparison(
    pdf_paths: list[str],
    configs: list[ChunkConfig] | None = None,
    queries: list[dict] | None = None,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    fetch_k: int = 20,
    top_k: int = 5,
) -> pd.DataFrame:
    """Runs the chunk-size comparison experiment. Returns one row per (config, query)."""
    configs = configs or DEFAULT_CONFIGS
    queries = queries if queries is not None else EVAL_QUERIES
    embedder = embedder or Embedder()
    reranker = reranker or Reranker()

    rows = []
    for chunk_config in configs:
        label = _config_label(chunk_config)

        start = time.time()
        index, chunks = build_in_memory_index(pdf_paths, chunk_config, embedder)
        build_seconds = time.time() - start

        avg_chunk_len = mean(len(c["text"]) for c in chunks) if chunks else 0
        num_chunks = len(chunks)
        index_bytes_est = index.ntotal * embedder.dimension * 4  # float32

        for item in queries:
            query = item["query"]
            expected_keywords = item.get("expected_keywords", [])

            query_vector = embedder.encode_normalized([query])[0]
            candidates = _search(index, chunks, query_vector, fetch_k)
            ranked = reranker.rerank(query, candidates, top_k) if candidates else []

            mean_faiss_score = mean(c["score"] for c in candidates[:top_k]) if candidates else 0.0
            top1_faiss_score = candidates[0]["score"] if candidates else 0.0
            mean_rerank_score = mean(r["rerank_score"] for r in ranked) if ranked else 0.0

            hit = bool(expected_keywords) and any(
                kw.lower() in c["text"].lower() for c in ranked for kw in expected_keywords
            )

            rows.append({
                "config": label,
                "chunk_size": chunk_config.chunk_size,
                "overlap": chunk_config.overlap,
                "num_chunks": num_chunks,
                "avg_chunk_len": round(avg_chunk_len, 1),
                "build_seconds": round(build_seconds, 3),
                "index_bytes_est": index_bytes_est,
                "query": query,
                "top1_faiss_score": round(top1_faiss_score, 4),
                "mean_faiss_score": round(mean_faiss_score, 4),
                "mean_rerank_score": round(mean_rerank_score, 4),
                "keyword_hit": hit,
            })

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Config-level aggregation: mean scores and keyword hit-rate per chunk_size/overlap config."""
    return (
        df.groupby(["config", "chunk_size", "overlap", "num_chunks", "avg_chunk_len", "build_seconds"])
        .agg(
            mean_top1_faiss_score=("top1_faiss_score", "mean"),
            mean_faiss_score=("mean_faiss_score", "mean"),
            mean_rerank_score=("mean_rerank_score", "mean"),
            keyword_hit_rate=("keyword_hit", "mean"),
        )
        .reset_index()
        .sort_values("chunk_size")
    )


if __name__ == "__main__":
    import glob
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from core import config as app_config

    pdfs = sorted(glob.glob(str(app_config.SAMPLE_PDFS_DIR / "*.pdf")))
    if not pdfs:
        raise SystemExit(
            f"No sample PDFs found in {app_config.SAMPLE_PDFS_DIR}. "
            "Run `python scripts/cli.py download` first."
        )

    results = run_comparison(pdfs)
    print(summarize(results).to_string(index=False))
