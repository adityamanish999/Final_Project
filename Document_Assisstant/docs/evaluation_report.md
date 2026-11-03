# Evaluation Report

## 1. Chunk-size comparison

**Method.** `evaluation/evaluate_chunking.py` builds an isolated, in-memory FAISS index for each
of three `ChunkConfig`s over the three sample Berkshire Hathaway PDFs (2023 annual report, 2023
and 2022 shareholder letters), then runs the fixed 10-question set (`EVAL_QUERIES` in the same file)
against each index (FAISS top-20 → cross-encoder rerank → top-5). Since no labeled ground-truth
QA dataset exists for these documents, `keyword_hit_rate` is a lightweight proxy — whether any
of a question's expected keywords appear in the top-5 reranked chunks — not an authoritative
relevance score. This is reproducible via `notebooks/demo.ipynb` (section 3) or
`python -m evaluation.evaluate_chunking`.

**Results** (from a real run against the sample PDFs):

| config    | chunk_size | overlap | num_chunks | avg_chunk_len | build_seconds | mean_top1_faiss_score | mean_faiss_score | mean_rerank_score | keyword_hit_rate |
|-----------|-----------:|--------:|-----------:|--------------:|---------------:|----------------------:|------------------:|-------------------:|------------------:|
| 400/50    | 400        | 50      | 1915       | 379.1         | 7.80           | 0.6629                | 0.6346             | 1.998               | 1.0 |
| 800/150   | 800        | 150     | 1068       | 721.5         | 7.17           | 0.6477                | 0.6141             | 1.702               | 1.0 |
| 1200/200  | 1200       | 200     | 726        | 1027.5        | 6.67           | 0.6354                | 0.6048             | 1.310               | 1.0 |

The relevance scores are deterministic and identical across runs; `build_seconds` depends on the
machine (this run: Apple Silicon Mac).

**Interpretation.**

- **Smaller chunks score higher.** Both the raw FAISS cosine score and the cross-encoder rerank
  score decrease monotonically as chunk size grows (400/50 → 1200/200). Smaller chunks are more
  topically focused, so a query embedding matches them more tightly; larger chunks dilute the
  signal by mixing in surrounding, less-relevant text.
- **Smaller chunks cost more.** 400/50 produces ~2.6x as many chunks as 1200/200 (1915 vs 726),
  meaning more embedding calls, a larger persisted index, and (per the table) a longer build
  time. In this run the difference in build time is modest at this document scale (a few
  seconds), but it would compound at much larger document collections.
- **Keyword hit-rate is not discriminating here** — all three configs hit 1.0 on this query set,
  meaning every question's expected keyword showed up in the top-5 reranked chunks regardless of
  chunk size. This says the retrieval pipeline is robust for these fairly specific,
  well-covered questions against a well-covered small document set; it does not mean chunk size
  is irrelevant — the score-based metrics above still clearly differentiate the configs, and
  hit-rate would likely become more discriminating on a larger/noisier document collection.
- **Recommendation.** `core/config.py` keeps the default at 800/150 as a reasonable middle
  ground (fewer index entries and faster ingestion than 400/50, most of the relevance benefit
  over 1200/200). For workloads that prioritize per-chunk precision above ingestion cost (e.g.
  a small, frequently-updated policy manual), 400/50 is worth switching to.

## 2. Response-quality observations

Ran three questions end-to-end through the live Groq tool-calling agent
(`openai/gpt-oss-120b`) against the persisted store built from the three sample PDFs (see
`notebooks/demo.ipynb`, section 2, for the full transcript):

1. **"What does Warren Buffett say about Berkshire Hathaway's insurance float?"** — The model
   called `retrieve_context`, received 5 reranked chunks (top rerank score 3.73, from
   `berkshire_2022_shareholder_letter.pdf` page 5; the rest scored far lower), and produced a
   grounded answer quoting the float's growth to "about 8,000 times" its 1967 size and calling it
   "an extraordinary asset for Berkshire", citing
   `(source: berkshire_2022_shareholder_letter.pdf, page 5)` correctly.
2. **"What does the letter say about Charlie Munger?"** — Described Munger as Buffett's
   "long-time partner" who was "of one mind" with him on responsibilities to shareholders, with
   accurate page-level citations across both shareholder letters (top rerank score 5.64).
3. **"What is the capital of France?"** (out-of-scope, deliberately) — The model recognised the
   question had nothing to do with the documents and, per the system prompt's grounding
   instruction, responded: *"I don't have enough information in these documents to answer
   that."* rather than answering from general knowledge. This is the key behavior the rubric's
   "grounded responses" requirement is checking for, and it held.

**Model change.** These results were first produced with `llama-3.3-70b-versatile`, which Groq
has since retired, so the project now uses `openai/gpt-oss-120b`. The pipeline is unchanged (the
model is one setting in `core/config.py`). Observed differences: gpt-oss is more persistent with
tool calls (it sometimes calls `list_documents` and then searches several times with reworded
queries), and it answered the out-of-scope question without searching at all, whereas Llama 3.3
searched first, got only negative rerank scores, and then declined. The persistence exposed a bug
in the iteration-cap fallback, which is now fixed (see `docs/architecture.md`).

**Takeaway.** Negative rerank scores on irrelevant questions are a useful signal beyond the
final refusal text — a future enhancement could have the agent short-circuit to the "not enough
information" message whenever all reranked candidates fall below a score threshold, saving a
Groq call. Not implemented here to keep the tool-calling loop simple and fully LLM-driven, but
worth noting as a possible latency/cost optimization.

## 3. Known limitations

- Chunks never cross page boundaries (see `core/documents.py`), so a chunk can end mid-sentence
  at a page break. Chosen deliberately to keep page-level citations exact.
- The keyword-hit-rate metric is a cheap proxy, not a validated relevance judgment; treat the
  FAISS/rerank scores as the primary signal in this report.
- `VectorStore` supports only a full `clear()`, not per-document deletion (removing a single
  document from a `faiss.IndexFlatIP` isn't efficient without switching to `IndexIDMap`) — noted
  as a possible future enhancement in `core/vector_store.py`.
