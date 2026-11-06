import json
from types import SimpleNamespace

import pytest

from core import llm_agent, vector_store
from tests.conftest import make_chunks


def _tool_call(call_id, name, arguments):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _completion(content=None, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeGroqClient:
    """Returns the given responses in order and records every request it receives."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


@pytest.fixture
def use_fake_client(fake_store, monkeypatch):
    """Puts one document in the fake store; returns a function that installs a FakeGroqClient."""
    vector_store.add_document(make_chunks("doc1.pdf", n=3), file_hash="hash1", filename="doc1.pdf")

    def install(responses):
        client = FakeGroqClient(responses)
        monkeypatch.setattr(llm_agent, "get_client", lambda: client)
        return client

    return install


def test_agent_dispatches_tool_call_and_returns_final_answer(use_fake_client):
    client = use_fake_client([
        _completion(
            content=None,
            tool_calls=[_tool_call("call_1", "retrieve_context", {"query": "chunk 0", "top_k": 2})],
        ),
        _completion(content="Here is the answer, grounded in the retrieved chunks."),
    ])

    result = llm_agent.ask("What is chunk 0 about?", history=[])

    assert result["answer"] == "Here is the answer, grounded in the retrieved chunks."
    assert len(result["sources"]) == 2
    assert result["history"][-2] == {"role": "user", "content": "What is chunk 0 about?"}
    assert result["history"][-1]["role"] == "assistant"

    # second call's messages must include the tool result message we appended
    second_call_messages = client.calls[1]["messages"]
    tool_messages = [m for m in second_call_messages if m["role"] == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "call_1"


def test_agent_answers_directly_without_tool_calls(use_fake_client):
    client = use_fake_client([_completion(content="No tools needed for this one.")])

    result = llm_agent.ask("Hello", history=[])

    assert result["answer"] == "No tools needed for this one."
    assert result["sources"] == []
    assert len(client.calls) == 1


def test_agent_stops_at_max_iterations_and_falls_back(use_fake_client):
    looping_tool_call = [_tool_call("call_x", "retrieve_context", {"query": "chunk"})]
    responses = [_completion(content=None, tool_calls=looping_tool_call) for _ in range(4)]
    responses.append(_completion(content="Fallback answer after hitting the iteration cap."))
    client = use_fake_client(responses)

    result = llm_agent.ask("Keep looping", history=[])

    assert result["answer"] == "Fallback answer after hitting the iteration cap."
    assert len(client.calls) == 5  # 4 tool-calling rounds (config.MAX_TOOL_ITERATIONS) + 1 fallback

    # The fallback call must offer no tools and contain no tool-call scaffolding (models such as
    # gpt-oss try to call a tool again if they see it), but must still carry the retrieved context.
    fallback_call = client.calls[-1]
    assert "tools" not in fallback_call
    assert all(m["role"] in ("system", "user", "assistant") for m in fallback_call["messages"])
    assert not any("tool_calls" in m for m in fallback_call["messages"])
    assert "chunk" in fallback_call["messages"][-1]["content"]
    assert "Keep looping" in fallback_call["messages"][-1]["content"]
