"""Streamlit Document Assistant — upload PDFs, ask questions, get grounded answers with citations."""

import os
import tempfile
from pathlib import Path

import streamlit as st

from core import config
from core.pipeline import DocumentAssistant
from core.vector_store import EmbeddingModelMismatchError

st.set_page_config(page_title="Document Assistant", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading models (embeddings + re-ranker)...")
def get_assistant() -> DocumentAssistant:
    return DocumentAssistant()


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander("Sources"):
        for src in sources:
            score = src.get("rerank_score", src.get("score", 0))
            st.markdown(f"**{src['source']}** (page {src['page']}) — score {score:.3f}")
            preview = src["text"][:300] + ("..." if len(src["text"]) > 300 else "")
            st.caption(preview)


def main() -> None:
    st.title("📄 Document Assistant")
    st.caption("Upload PDFs and ask natural-language questions grounded in their content.")

    if not os.environ.get(config.GROQ_API_KEY_ENV_VAR):
        st.error(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your Groq API key, "
            "then restart the app."
        )
        st.stop()

    try:
        assistant = get_assistant()
    except EmbeddingModelMismatchError as exc:
        st.error(str(exc))
        st.stop()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # [{"role","content"} ...] — sent to the LLM
    if "display_history" not in st.session_state:
        st.session_state.display_history = []  # [{"role","content","sources"} ...] — rendered in UI

    with st.sidebar:
        st.header("Knowledge base")

        uploaded_files = st.file_uploader(
            "Upload PDF documents", type=["pdf"], accept_multiple_files=True
        )
        if uploaded_files and st.button("Ingest uploaded files", type="primary"):
            config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(
                    dir=config.UPLOADS_DIR, suffix=".pdf", delete=False
                ) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = Path(tmp.name)

                with st.spinner(f"Ingesting {uploaded_file.name}..."):
                    result = assistant.ingest_pdf(tmp_path, display_name=uploaded_file.name)

                if result["skipped"]:
                    st.info(f"Already indexed, skipped: {result['filename']}")
                else:
                    st.success(f"Indexed {result['filename']}: {result['added_chunks']} chunks")

        st.divider()
        docs = assistant.list_documents()
        st.subheader(f"Documents in store ({len(docs)})")
        if docs:
            for doc in docs:
                st.write(f"- **{doc['filename']}** ({doc['num_chunks']} chunks)")
        else:
            st.caption("No documents ingested yet.")

        st.divider()
        if st.checkbox("Confirm clear knowledge base"):
            if st.button("Clear knowledge base", type="secondary"):
                assistant.clear()
                st.session_state.chat_history = []
                st.session_state.display_history = []
                st.rerun()

    for turn in st.session_state.display_history:
        with st.chat_message(turn["role"]):
            st.write(turn["content"])
            render_sources(turn.get("sources", []))

    if query := st.chat_input("Ask a question about your documents..."):
        st.session_state.display_history.append({"role": "user", "content": query, "sources": []})
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = assistant.ask(query, st.session_state.chat_history)
            st.write(result["answer"])
            render_sources(result["sources"])

        st.session_state.display_history.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        })


if __name__ == "__main__":
    main()
