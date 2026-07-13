"""Phase 5: Groq-powered answer generation with native tool-calling.

Vanilla Python state/tool-call management (no LangChain/LangGraph/CrewAI): we send the
conversation plus tool schemas to Groq, parse any `tool_calls` on the response, dispatch them
to real Python functions, feed the results back as `role: "tool"` messages, and loop until the
model returns a plain-text answer.
"""

import json

from groq import Groq

from core import config
from core.embeddings import Embedder
from core.prompts import SYSTEM_PROMPT
from core.reranker import Reranker
from core.tools import TOOL_DISPATCH, TOOLS
from core.vector_store import VectorStore


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

        # Iteration cap hit: force a plain (no-tools) completion so we still answer.
        fallback = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=0.1
        )
        answer = fallback.choices[0].message.content or ""
        history.append({"role": "user", "content": user_query})
        history.append({"role": "assistant", "content": answer})
        return {"answer": answer, "sources": last_sources, "history": history}
