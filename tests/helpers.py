"""Stand-ins and builders that more than one test module needs.

Fixtures live in conftest.py; this holds what a test imports by name.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from genetics_mcp_server.llm_service import LLMService
from genetics_mcp_server.tools.definitions import get_anthropic_tools, resolve_tools
from genetics_mcp_server.tools.executor import ToolExecutor


class ExecutorCase:
    """Base for a test class that drives a real executor, closed after each test."""

    executor_class = ToolExecutor

    @pytest.fixture(autouse=True)
    async def setup_executor(self):
        self.executor = self.executor_class()
        yield
        await self.executor.close()

    def _patch_get(self, resolver):
        """Patch the shared client's GET with a URL router; returns the patch and call log."""
        calls: list[str] = []

        def handler(url, *args, **kwargs):
            text = str(url)
            calls.append(text)
            resp = resolver(text)
            assert resp is not None, f"unexpected request: {text}"
            return resp

        patcher = patch.object(
            self.executor.external_client, "get", new_callable=AsyncMock, side_effect=handler
        )
        return patcher, calls


class RecordingExecutor:
    """Stands in for ToolExecutor: any method reached for is recorded and answers `payload`.

    __getattr__ rather than named methods because a caller dispatches one function across
    several executor methods, and which one is reached is for the test to read off `calls`.
    """

    def __init__(self, payload=None):
        self.payload = payload
        self.calls: list[tuple[str, tuple, dict]] = []

    def __getattr__(self, name):
        async def call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return self.payload

        return call


def mock_response(status_code: int = 200, json_data: dict | None = None, text: str = ""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.json = MagicMock(return_value=json_data or {})
    return resp


def code_surface_tool_names() -> set[str]:
    """The tools offered alongside code execution: the SDK-replaceable ones are left out."""
    return {t["name"] for t in get_anthropic_tools(code_execution=True)}


def tool_param(tools, name, param):
    (tool,) = [t for t in tools if t["name"] == name]
    return tool["input_schema"]["properties"][param]["description"]


def tool_description(tools, name):
    (tool,) = [t for t in tools if t["name"] == name]
    return tool["description"]


def assert_module_definitions_lack(*needles):
    """Rendering the chat tools must leave the module-level definitions without the
    deployment's hints: they are added to copies."""
    get_anthropic_tools()
    for tool in resolve_tools(False):
        blob = repr(tool)
        assert not any(needle in blob for needle in needles), tool["name"]


def seed_session(db, user, title, message="What about APOE?", project_id=None):
    """A titled session holding one user message."""
    session = db.create_session(user, project_id=project_id)
    db.update_session(session.id, user, title=title)
    db.add_message(session.id, f"m-{session.id}", "user", message)
    return session


def set_updated_at(db, session_id: str, value: str) -> None:
    """Force a deterministic chat_sessions.updated_at (CURRENT_TIMESTAMP has 1s resolution)."""
    db._conn.execute("UPDATE chat_sessions SET updated_at = ? WHERE id = ?", (value, session_id))
    db._conn.commit()


def sandbox_result_body(execution_id, **overrides):
    """A supervisor result for a script that ran and exited 0."""
    body = {
        "execution_id": execution_id,
        "status": "ok",
        "exit_code": 0,
        "signal": None,
        "duration_ms": 12,
        "output": "hello\n",
        "output_bytes": 6,
        "output_truncated": False,
        "error": None,
        "artifacts": [],
        "artifacts_omitted": 0,
    }
    body.update(overrides)
    return body


def bare_llm_service(messages, executor=None):
    """An LLMService that skipped __init__, streaming from `messages` instead of Anthropic."""
    svc = LLMService.__new__(LLMService)
    svc.openai_client = None
    svc.executor = executor
    svc.subagent_service = None
    svc.anthropic_client = SimpleNamespace(messages=messages)
    return svc


class CapturingService:
    """A stand-in LLM service that records the kwargs the endpoint hands `stream_chat`,
    then streams `chunks`."""

    def __init__(self, chunks=()):
        self.anthropic_client = object()
        self.openai_client = object()
        # deliberately LIVE: with it None, `Settings.enable_subagents` defaulting false and
        # the service-liveness check would both be hiding launch_subagents at once, and no
        # test could tell which one did it
        self.subagent_service = object()
        self.kwargs = None
        self._chunks = chunks

    def resolve_local_tools(
        self, *, code_execution=False, enable_tools=True, custom_tool_descriptions=None
    ):
        """The real resolution, not a stub: the endpoint assembles the system prompt from
        it, so a stub here would stop a test from seeing the prompt the endpoint sends."""
        self._disabled_tools = lambda: LLMService._disabled_tools(self)
        return LLMService.resolve_local_tools(
            self,
            code_execution=code_execution,
            enable_tools=enable_tools,
            custom_tool_descriptions=custom_tool_descriptions,
        )

    def stream_chat(self, **kwargs):
        self.kwargs = kwargs

        async def _stream():
            for chunk in self._chunks:
                yield chunk

        return _stream()
