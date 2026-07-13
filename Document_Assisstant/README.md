# Document Assistant (RAG)

An AI-powered Document Assistant: upload PDFs, ask natural-language questions, get grounded
answers with source/page citations. Built with vanilla Python (no LangChain/LangGraph/CrewAI) —
Groq for the LLM, `sentence-transformers` for embeddings/reranking, and a persistent FAISS
vector store.

See `docs/architecture.md` for the full pipeline diagram and `docs/evaluation_report.md` for the
chunk-size comparison and response-quality findings.

## Project structure

```
RAG/
├── app.py                  # Streamlit application
├── core/                   # all pipeline logic (ingestion, chunking, embeddings, FAISS store,
│                            # reranking, retrieval, prompts, Groq tool-calling agent)
├── scripts/                 # download_sample_data.py, ingest_cli.py
├── evaluation/               # chunk-size comparison experiment (evaluate_chunking.py)
├── notebooks/demo.ipynb      # thin demo/reporting notebook (imports core/evaluation, no logic)
├── data/                      # sample_pdfs/ (downloaded), vector_store/ (persisted FAISS index)
├── docs/                       # architecture.md, evaluation_report.md
└── tests/                       # pytest unit tests
```

## Setup

1. **Python**: 3.11+ recommended (tested on 3.14).
2. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Get a free Groq API key at https://console.groq.com and add it to a `.env` file:
   ```bash
   cp .env.example .env
   # edit .env: GROQ_API_KEY=your-actual-key
   ```

## Usage

**1. Download sample data** (a few public Berkshire Hathaway annual-report / shareholder-letter
PDFs, used for testing and the evaluation notebook):
```bash
python scripts/download_sample_data.py
```

**2. Ingest PDFs into the persistent vector store** (idempotent — re-running skips already-
ingested files):
```bash
python scripts/ingest_cli.py data/sample_pdfs/*.pdf
# or any of your own PDFs:
python scripts/ingest_cli.py path/to/your.pdf
```

**3. Run the Streamlit app:**
```bash
streamlit run app.py
```
Upload additional PDFs from the sidebar, or just start asking questions if you've already
ingested documents via the CLI — the vector store persists across restarts.

**4. Run the demo/evaluation notebook:**
```bash
jupyter notebook notebooks/demo.ipynb
```
Demonstrates ingestion, a few live Q&A turns (including a deliberately out-of-scope question to
show the assistant declining rather than hallucinating), and the chunk-size comparison
experiment with charts.

**5. Run tests:**
```bash
pytest tests/
```
Tests use fake embedder/reranker/Groq-client doubles, so they run fast and don't require
network access or an API key.

## How it works (short version)

1. **Ingestion** (`core/ingestion.py`): `pypdf` extracts per-page text; whitespace and repeated
   headers/footers are cleaned up.
2. **Chunking** (`core/chunking.py`): fixed-size character windows with overlap, configurable via
   `ChunkConfig`, never crossing page boundaries.
3. **Embeddings** (`core/embeddings.py`): `sentence-transformers/all-MiniLM-L6-v2`, local, no API
   calls.
4. **Persistent vector store** (`core/vector_store.py`): FAISS `IndexFlatIP` + JSON metadata/
   manifest sidecar files under `data/vector_store/`, with dedup by file hash and incremental
   add for new documents.
5. **Retrieval + ranking** (`core/retrieval.py`, `core/reranker.py`): FAISS over-fetches
   candidates, a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) reranks them.
6. **Answer generation** (`core/llm_agent.py`, `core/tools.py`, `core/prompts.py`): Groq
   (`llama-3.3-70b-versatile`) with native tool-calling — the model decides when to call
   `retrieve_context`/`list_documents`; results are dispatched and fed back as tool messages
   until it returns a grounded, cited answer.
7. **Streamlit app** (`app.py`): upload/ingest PDFs, chat with history, view sources per answer,
   clear the knowledge base.
