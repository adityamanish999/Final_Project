"""Step 5: answer a question with the Groq LLM, letting it call tools to search the documents.

How one question is answered (`ask`):
1. Send the conversation to Groq along with a description of our tools (`TOOLS`).
2. If the model replies with plain text, that's the answer. Done.
3. If the model asks to call a tool, we run the matching Python function (`run_tool`), send
   the result back, and go to step 1 again.
4. If the model is still calling tools after MAX_TOOL_ITERATIONS rounds, we stop and ask it
   to answer from whatever it has retrieved so far, with no tools allowed.

No LangChain or similar: it's just a loop over a list of message dicts.
"""

import json

from groq import Groq

from core import config, retrieval, vector_store
from core.prompts import (
    CONTEXT_BLOCK_TEMPLATE,
    FALLBACK_SYSTEM_PROMPT,
    FALLBACK_USER_TEMPLATE,
    NO_RESULTS_MESSAGE,
    SYSTEM_PROMPT,
)

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

# This is what Groq sees: the name, purpose and arguments of each tool, as JSON schema.
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


# The real Python code behind each tool. Each returns (text for the LLM, source chunks for the UI).

def tool_retrieve_context(query: str, top_k: int = config.DEFAULT_TOP_K) -> tuple[str, list[dict]]:
    chunks = retrieval.retrieve(query, top_k=top_k)
    if not chunks:
        return NO_RESULTS_MESSAGE, []

    text = "\n\n".join(
        CONTEXT_BLOCK_TEMPLATE.format(source=c["source"], page=c["page"], text=c["text"])
        for c in chunks
    )
    return text, chunks


def tool_list_documents() -> tuple[str, list[dict]]:
    docs = vector_store.list_documents()
    if not docs:
        return "No documents have been ingested yet.", []
    return json.dumps([{"filename": d["filename"], "num_chunks": d["num_chunks"]} for d in docs]), []


def run_tool(name: str, args: dict) -> tuple[str, list[dict]]:
    """Runs the tool the model asked for."""
    if name == "retrieve_context":
        return tool_retrieve_context(args["query"], int(args.get("top_k") or config.DEFAULT_TOP_K))
    if name == "list_documents":
        return tool_list_documents()
    return f"Unknown tool: {name}", []


# ---------------------------------------------------------------------------
# Groq client
# ---------------------------------------------------------------------------

_client = None  # created on first use, so ingesting PDFs doesn't need an API key


def get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq()  # reads GROQ_API_KEY from the environment
    return _client


# ---------------------------------------------------------------------------
# The question-answering loop
# ---------------------------------------------------------------------------


def ask(question: str, history: list[dict] | None = None) -> dict:
    """Answers one question.

    `history` is the list of earlier {"role", "content"} messages in this conversation. This
    question and its answer are appended to it.

    Returns {"answer": str, "sources": list of chunk dicts, "history": the updated history}.
    """
    if history is None:
        history = []

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": question})

    sources = []          # chunks from the latest retrieve_context call, shown in the UI
    retrieved_texts = []  # everything retrieve_context returned, used by the fallback

    for _ in range(config.MAX_TOOL_ITERATIONS):
        response = get_client().chat.completions.create(
            model=config.GROQ_MODEL_NAME,
            messages=messages,
            tools=TOOLS,
            temperature=0.1,
        )
        message = response.choices[0].message

        # No tool calls -> the model has written its final answer.
        if not message.tool_calls:
            answer = message.content or ""
            return _finish(question, answer, sources, history)

        # The model wants to call one or more tools. Record its request, run each tool,
        # and add the results to the conversation so the model can read them next round.
        messages.append(_assistant_message_with_tool_calls(message))

        for tool_call in message.tool_calls:
            name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            text, tool_sources = run_tool(name, args)

            if name == "retrieve_context":
                retrieved_texts.append(text)
                if tool_sources:
                    sources = tool_sources

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": name,
                "content": text,
            })

    # Still calling tools after MAX_TOOL_ITERATIONS rounds: force an answer.
    answer = _answer_without_tools(question, history, retrieved_texts)
    return _finish(question, answer, sources, history)


def _assistant_message_with_tool_calls(message) -> dict:
    """Converts Groq's tool-call reply into a plain dict we can send back next round."""
    return {
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
    }


def _answer_without_tools(question: str, history: list[dict], retrieved_texts: list[str]) -> str:
    """Asks the model to answer using only the text retrieved so far.

    This request is built from scratch, without any of the tool-call messages: some models
    (e.g. gpt-oss) try to call a tool again when they see them, and Groq rejects that with a 400.
    """
    context = "\n\n".join(retrieved_texts) or NO_RESULTS_MESSAGE
    messages = [{"role": "system", "content": FALLBACK_SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({
        "role": "user",
        "content": FALLBACK_USER_TEMPLATE.format(context=context, question=question),
    })
    response = get_client().chat.completions.create(
        model=config.GROQ_MODEL_NAME, messages=messages, temperature=0.1
    )
    return response.choices[0].message.content or ""


def _finish(question: str, answer: str, sources: list[dict], history: list[dict]) -> dict:
    """Saves this turn to the history (only the plain question and answer, no tool messages)."""
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    return {"answer": answer, "sources": sources, "history": history}
