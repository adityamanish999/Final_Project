"""High-level facade tying ingestion, chunking, and the agent together.

This is what app.py, scripts/ingest_cli.py, and notebooks/demo.ipynb call — none of them
touch core.ingestion/chunking/vector_store directly.
"""

from pathlib import Path

from core.chunking import ChunkConfig, chunk_document
from core.embeddings import Embedder
from core.ingestion import load_document
from core.llm_agent import Agent
from core.reranker import Reranker
from core.vector_store import VectorStore


class DocumentAssistant:
    def __init__(self, store: VectorStore | None = None, reranker: Reranker | None = None):
        self.embedder = Embedder()
        self.store = store or VectorStore(embedder=self.embedder)
        self.reranker = reranker or Reranker()
        self._agent: Agent | None = None

    @property
    def agent(self) -> Agent:
        """Lazy — constructing it requires a Groq API key, which ingestion-only flows don't need."""
        if self._agent is None:
            self._agent = Agent(store=self.store, reranker=self.reranker, embedder=self.embedder)
        return self._agent

    def ingest_pdf(
        self,
        pdf_path: str | Path,
        chunk_config: ChunkConfig | None = None,
        display_name: str | None = None,
    ) -> dict:
        """Extracts, cleans, chunks, embeds, and persists a PDF. Idempotent per file content.

        `display_name` overrides the stored filename — useful when `pdf_path` is a temp file
        (e.g. a Streamlit upload) and the original filename should be shown/cited instead.
        """
        pages, file_hash = load_document(pdf_path)
        filename = display_name or Path(pdf_path).name

        if self.store.is_ingested(file_hash):
            return {"filename": filename, "added_chunks": 0, "skipped": True}

        chunks = chunk_document(pages, chunk_config)
        for chunk in chunks:
            chunk["source"] = filename
        added = self.store.add_document(chunks, file_hash, filename)
        self.store.save()
        return {"filename": filename, "added_chunks": added, "skipped": False}

    def ask(self, query: str, history: list[dict] | None = None) -> dict:
        return self.agent.ask(query, history)

    def list_documents(self) -> list[dict]:
        return self.store.list_documents()

    def clear(self) -> None:
        self.store.clear()
