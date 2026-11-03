# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

RAG "Document Assistant": ingest PDFs, ask questions, get answers grounded in the documents with source/page citations. Deliberately **vanilla Python — no LangChain/LangGraph/CrewAI**; the LLM tool-calling loop is hand-rolled against the Groq SDK. Keep it that way.

This directory is one subproject of a monorepo (git root is the parent `Final_Project/`, sibling project `FinsightsAI/`). Run all commands from this directory.

## Commands

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python scripts/cli.py download                        # fetch Berkshire Hathaway sample PDFs -> data/sample_pdfs/
python scripts/cli.py ingest data/sample_pdfs/*.pdf   # ingest into persistent store (idempotent)
streamlit run app.py                                  # UI (needs GROQ_API_KEY)
python evaluation/evaluate_chunking.py                # chunk-size comparison over sample PDFs
pytest tests/                                         # all tests
pytest tests/test_llm_agent.py::test_agent_stops_at_max_iterations_and_falls_back   # single test
```

- `GROQ_API_KEY` is the only required variable; copy `.env.example` to `.env` in this directory (loaded once by `core/config.py`). For notebook execution: `jupyter nbconvert --to notebook --execute --inplace notebooks/demo.ipynb` (needs `nbconvert` + `ipykernel`, not in requirements.txt).
- Tests need no network, API key, or model download: they use `FakeEmbedder`/`make_chunks` (`tests/test_vector_store.py`), `FakeReranker` (`tests/test_retrieval.py`), and `FakeGroqClient` (`tests/test_llm_agent.py`). Other test modules import these via `from tests.test_... import ...`, so run pytest from this directory. Reuse these fakes for new tests.

## Architecture

`core/` has one module per assignment phase. Pipeline: `documents` (phases 1-2: pypdf per-page text + header/footer cleanup + sha256 file hash, then fixed-size char-window chunking with overlap, **never crossing page boundaries** so citations stay page-accurate) → `vector_store` (phase 3: `Embedder` with local `all-MiniLM-L6-v2`, L2-normalized so FAISS inner product = cosine, and `VectorStore` over FAISS `IndexFlatIP`) → `retrieval` (phase 4: over-fetch `DEFAULT_FETCH_K=20` from FAISS, then cross-encoder `Reranker` down to `top_k`) → `llm_agent` (phase 5: tools + agent loop on Groq `openai/gpt-oss-120b`; `llama-3.3-70b-versatile` was retired by Groq). `prompts.py` holds all prompt templates. All defaults/model names live in `core/config.py`.

Key design points that span files:

- **`core/pipeline.py` `DocumentAssistant` is the facade.** `app.py`, `scripts/cli.py`, and `notebooks/demo.ipynb` call only it, not lower-level `core` modules. The Groq `Agent` is constructed lazily so ingestion-only flows don't need an API key.
- **Tool-calling loop (`core/llm_agent.py`)**: the model decides when to call tools (not always-retrieve). Tools are defined as JSON schemas in `TOOLS` plus a `TOOL_DISPATCH` name→handler map at the top of the same file; handlers have signature `(args: dict, agent) -> (content_str, sources_list)`. Adding a tool means updating both. The loop is capped at `MAX_TOOL_ITERATIONS=4`, then a no-tools completion forces an answer; that fallback request is rebuilt from `FALLBACK_*` prompts plus the retrieved context, **without** tool-call messages (gpt-oss tries to call tools again if it sees them, and Groq returns a 400). gpt-oss often uses several tool rounds per question, so this path is exercised in practice. Only clean `{"role","content"}` user/assistant turns are appended to the caller's `history` (mutated in place); tool-call scaffolding stays local to the turn. `sources` returned are those of the last `retrieve_context` call.
- **Grounding/citation behavior is driven by the prompt** in `core/prompts.py` (`SYSTEM_PROMPT`, `CONTEXT_BLOCK_TEMPLATE`, `NO_RESULTS_MESSAGE`): answer only from retrieved context, decline otherwise, cite source/page.
- **Persistent store (`data/vector_store/`)**: `index.faiss` + `metadata.json` (chunk records, position-aligned with FAISS ids — never reorder/delete individual entries) + `manifest.json` (embedding model name/dimension + ingested file hashes). Writes are atomic (temp file + `os.replace`). Dedup is by file content hash; new files are appended incrementally. Changing `EMBEDDING_MODEL_NAME` makes loading raise `EmbeddingModelMismatchError` — the store must be cleared and re-ingested. The committed store under `data/vector_store/` contains the ingested sample PDFs; tests use `tmp_path` stores and must not touch it.
- **Evaluation (`evaluation/`)** builds isolated in-memory FAISS indexes per `ChunkConfig` (does not touch the persistent store) and scores them against the fixed `EVAL_QUERIES` set defined in `evaluation/evaluate_chunking.py`. The notebook only displays results; logic belongs in `evaluation/`/`core/`, not notebooks.

See `docs/architecture.md` (diagrams) and `docs/evaluation_report.md` (chunk-size findings) for detail.
