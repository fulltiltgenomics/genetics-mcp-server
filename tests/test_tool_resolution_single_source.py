"""The prompt's tool set and the model's tool set come from ONE derivation.

genetics-results-suite-4h6.69 gated the system prompt on the tool list. As shipped it
held only because `chat_api` happened to pass the same profile to two independent
resolutions — `resolve_local_tool_names` for the prompt, `get_anthropic_tools` inside
`_stream_anthropic` for the model. A change to one and not the other would have told the
model about tools it did not have, silently, with the whole suite green
(genetics-results-suite-4h6.77).

These tests pin both ends of the single derivation: the endpoint hands its resolved set
down, and the streaming path consumes it instead of deriving its own.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from genetics_mcp_server import chat_api, llm_service
from genetics_mcp_server.llm_service import LLMService, ResolvedLocalTools


class _FakeStream:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def __aiter__(self):
        return
        yield  # pragma: no cover - an empty async iterator

    async def get_final_message(self):
        return SimpleNamespace(
            content=[],
            stop_reason="end_turn",
            usage=SimpleNamespace(
                input_tokens=1,
                output_tokens=1,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            ),
        )


class _FakeMessages:
    def __init__(self):
        self.calls = []

    def stream(self, **params):
        self.calls.append(params)
        return _FakeStream()


def _service():
    svc = LLMService.__new__(LLMService)
    svc.openai_client = None
    svc.executor = None
    svc.subagent_service = None
    svc.anthropic_client = SimpleNamespace(messages=_FakeMessages())
    return svc


def _tool(name):
    return {"name": name, "description": name, "input_schema": {"type": "object"}}


@pytest.mark.asyncio
async def test_the_streaming_path_derives_no_tool_list_of_its_own(monkeypatch):
    """Given a caller's resolution, `_stream_anthropic` must not resolve a second one.

    `get_anthropic_tools` is made to EXPLODE rather than merely counted: the point is not
    that two derivations currently agree, it is that the second derivation is gone. Before
    genetics-results-suite-4h6.77 this call is exactly what built the model's tool list, so
    this test fails there on the re-derivation itself, not on a mismatch.
    """

    def _forbidden(*args, **kwargs):
        raise AssertionError(
            "_stream_anthropic re-derived the local tool set; the prompt was built from a "
            "different derivation and the two can now drift (4h6.77)"
        )

    monkeypatch.setattr(llm_service, "get_anthropic_tools", _forbidden)
    monkeypatch.setattr(llm_service, "get_external_anthropic_tools", lambda: [_tool("ext")])
    monkeypatch.setattr(llm_service, "get_rag_anthropic_tools", lambda: [_tool("rag")])

    resolved = ResolvedLocalTools([_tool("only_tool_the_prompt_named")])
    svc = _service()

    async for _ in svc._stream_anthropic(
        messages=[{"role": "user", "content": "hi"}],
        model="claude-opus-5",
        system_prompt="prompt built from resolved.names",
        enable_tools=True,
        local_tools=resolved,
    ):
        pass

    sent = svc.anthropic_client.messages.calls[0]["tools"]
    assert [t["name"] for t in sent] == ["only_tool_the_prompt_named", "ext", "rag"]
    # the external/RAG append must not leak back into the set the prompt was built from
    assert resolved.names == {"only_tool_the_prompt_named"}


@pytest.mark.asyncio
async def test_resolved_names_are_projected_off_the_definitions_not_stored():
    """There is no way to hold a resolution whose names describe another tool set."""
    resolved = ResolvedLocalTools([_tool("a")])
    assert resolved.names == {"a"}
    resolved.definitions.append(_tool("b"))
    assert resolved.names == {"a", "b"}


class _CapturingService:
    """Records what the endpoint hands `stream_chat`, resolving tools for real."""

    def __init__(self):
        self.anthropic_client = object()
        self.openai_client = object()
        self.subagent_service = object()
        self.kwargs = None

    def resolve_local_tools(
        self, tool_profile=None, enable_tools=True, custom_tool_descriptions=None
    ):
        self._disabled_tools = lambda: LLMService._disabled_tools(self)
        return LLMService.resolve_local_tools(
            self, tool_profile, enable_tools, custom_tool_descriptions
        )

    def stream_chat(self, **kwargs):
        self.kwargs = kwargs

        async def _stream():
            return
            yield  # pragma: no cover

        return _stream()


def test_the_endpoint_resolves_once_and_hands_that_same_resolution_down(test_client):
    """One derivation per request, and the model gets the object the prompt was built from.

    The stub `get_anthropic_tools` returns a DIFFERENT set on every call, so a second
    derivation anywhere on the request path cannot coincidentally match the first. Before
    genetics-results-suite-4h6.77 the endpoint passed only `tool_profile` onward and there
    was no resolution to hand down at all, so `local_tools` is absent and this fails.
    """
    calls = []

    def _counting(custom_descriptions=None, tool_profile=None, disabled_tools=None):
        calls.append(tool_profile)
        return [_tool(f"tool_from_derivation_{len(calls)}")]

    recorded = {}

    def _recording_prompt(app_name, tool_names=None, **kwargs):
        recorded["tool_names"] = tool_names
        return "SYSTEM"

    service = _CapturingService()
    with (
        patch.object(llm_service, "get_anthropic_tools", _counting),
        patch.object(chat_api, "default_system_prompt", _recording_prompt),
        patch.object(chat_api, "get_llm_service", return_value=service),
        patch.object(chat_api, "get_llm_config_db", return_value=MagicMock()),
    ):
        response = test_client.post(
            "/chat/v1/chat",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "tool_profile": "bigquery",
            },
            headers={"X-Goog-Authenticated-User-Email": "accounts.google.com:a@finngen.fi"},
        )
    assert response.status_code == 200

    assert calls == ["bigquery"], f"expected ONE local tool derivation, got {len(calls)}"
    assert recorded["tool_names"] == {"tool_from_derivation_1"}
    assert service.kwargs["local_tools"].names == recorded["tool_names"]
