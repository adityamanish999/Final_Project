"""Groq tool schemas and dispatch table for the Document Assistant agent."""

import json

from core import config
from core.prompts import CONTEXT_BLOCK_TEMPLATE, NO_RESULTS_MESSAGE
from core.retrieval import retrieve_and_rank

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_context",
            "description": (
                "Search the ingested documents for chunks relevant to a query. Use this "
                "whenever answering a question that may depend on document content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query, rephrased for retrieval if needed.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of chunks to return.",
                        "default": config.DEFAULT_TOP_K,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": "List the documents currently available in the knowledge base, with chunk counts.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def _run_retrieve_context(args: dict, agent) -> tuple[str, list[dict]]:
    query = args["query"]
    top_k = int(args.get("top_k") or config.DEFAULT_TOP_K)
    chunks = retrieve_and_rank(
        query, agent.store, agent.reranker, embedder=agent.embedder, top_k=top_k
    )
    if not chunks:
        return NO_RESULTS_MESSAGE, []

    content = "\n\n".join(
        CONTEXT_BLOCK_TEMPLATE.format(source=c["source"], page=c["page"], text=c["text"])
        for c in chunks
    )
    return content, chunks


def _run_list_documents(args: dict, agent) -> tuple[str, list[dict]]:
    docs = agent.store.list_documents()
    if not docs:
        return "No documents have been ingested yet.", []
    return json.dumps(
        [{"filename": d["filename"], "num_chunks": d["num_chunks"]} for d in docs]
    ), []


TOOL_DISPATCH = {
    "retrieve_context": _run_retrieve_context,
    "list_documents": _run_list_documents,
}
