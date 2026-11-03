"""Phase 5: Groq-powered answer generation with native tool-calling.

Vanilla Python state/tool-call management (no LangChain/LangGraph/CrewAI): we send the
conversation plus tool schemas (`TOOLS`) to Groq, parse any `tool_calls` on the response,
dispatch them to real Python functions (`TOOL_DISPATCH`), feed the results back as
`role: "tool"` messages, and loop until the model returns a plain-text answer.
"""

import json

from groq import Groq

from core import config
from core.prompts import (
    CONTEXT_BLOCK_TEMPLATE,
    FALLBACK_SYSTEM_PROMPT,
    FALLBACK_USER_TEMPLATE,
    NO_RESULTS_MESSAGE,
    SYSTEM_PROMPT,
)
from core.retrieval import Reranker, retrieve_and_rank
from core.vector_store import Embedder, VectorStore

# ---------------------------------------------------------------------------
# Tools: JSON schemas sent to Groq + dispatch table to real Python functions
# ---------------------------------------------------------------------------

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


def _run_retrieve_context(args: dict, agent: "Agent") -> tuple[str, list[dict]]:
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


def _run_list_documents(args: dict, agent: "Agent") -> tuple[str, list[dict]]:
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


# ---------------------------------------------------------------------------
# Agent: the tool-calling loop
# ---------------------------------------------------------------------------


class Agent:
    def __init__(
        self,
        store: VectorStore,
        reranker: Reranker,
        embedder: Embedder | None = None,
        client: Groq | None = None,
        model: str = config.GROQ_MODEL_NAME,
    ):
        self.store = store
        self.reranker = reranker
        self.embedder = embedder or store.embedder
        self.client = client or Groq()
        self.model = model

    def ask(self, user_query: str, history: list[dict] | None = None) -> dict:
        """Runs the tool-calling loop for one user turn.

        `history` is a plain list of {"role","content"} dicts from prior turns (mutated
        in-place with this turn's user/assistant messages, and also returned for convenience).
        Returns {"answer": str, "sources": list[dict], "history": list[dict]}.
        """
        history = history if history is not None else []
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_query})

        last_sources: list[dict] = []

        for _ in range(config.MAX_TOOL_ITERATIONS):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOLS,
                temperature=0.1,
            )
            message = response.choices[0].message

            if not message.tool_calls:
                answer = message.content or ""
                history.append({"role": "user", "content": user_query})
                history.append({"role": "assistant", "content": answer})
                return {"answer": answer, "sources": last_sources, "history": history}

            messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in message.tool_calls
                ],
            })

            for tool_call in message.tool_calls:
                name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                handler = TOOL_DISPATCH.get(name)
                if handler is None:
                    content, sources = f"Unknown tool: {name}", []
                else:
                    content, sources = handler(args, self)

                if name == "retrieve_context" and sources:
                    last_sources = sources

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": name,
                    "content": content,
                })

        # Iteration cap hit: answer from the context already retrieved, with no tools.
        # The request is rebuilt without the tool-call messages: some models (e.g. gpt-oss)
        # try to call a tool again when they see them, and Groq rejects that with a 400.
        retrieved = [
            m["content"] for m in messages
            if m["role"] == "tool" and m["name"] == "retrieve_context"
        ]
        fallback_messages = [{"role": "system", "content": FALLBACK_SYSTEM_PROMPT}]
        fallback_messages.extend(history)
        fallback_messages.append({
            "role": "user",
            "content": FALLBACK_USER_TEMPLATE.format(
                context="\n\n".join(retrieved) or NO_RESULTS_MESSAGE, question=user_query
            ),
        })
        fallback = self.client.chat.completions.create(
            model=self.model, messages=fallback_messages, temperature=0.1
        )
        answer = fallback.choices[0].message.content or ""
        history.append({"role": "user", "content": user_query})
        history.append({"role": "assistant", "content": answer})
        return {"answer": answer, "sources": last_sources, "history": history}
