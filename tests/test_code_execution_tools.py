"""Tests for the code-execution tools: list_capabilities, read_artifact, run_analysis.

genetics-results-suite-4h6.15 and -4h6.48. The sandbox is not deployed, so these exercise
the tool layer only: the SDK catalogue rendering, the artifact read together with the path
validation it borrows from skills/sandbox_tools.py, and run_analysis against a stubbed
transport — no live sandbox, no credentials.
"""

import asyncio
import base64
import inspect
import json

import pytest
from conftest import settings_env
from fastapi import Request

from genetics_mcp_server.config.settings import Settings
from genetics_mcp_server.llm_service import _script_result_payload, _truncation_notice
from genetics_mcp_server.sandbox_client import ArtifactResult
from genetics_mcp_server.tools import ToolExecutor
from genetics_mcp_server.tools import executor as executor_module
from genetics_mcp_server.tools.definitions import TOOL_DEFINITIONS, get_anthropic_tools

PNG_HEADER = b"\x89PNG\r\n\x1a\n" + bytes(range(256))


@pytest.fixture
def executor():
    return ToolExecutor()


@pytest.fixture(autouse=True)
def _clean_manifest_registry():
    """The sid -> execution map is process-wide state; no test may see another's rows."""
    executor_module._ARTIFACT_MANIFESTS.clear()
    yield
    executor_module._ARTIFACT_MANIFESTS.clear()


class TestToolDefinitions:
    def test_both_tools_defined(self):
        names = {t["name"] for t in TOOL_DEFINITIONS}
        assert {"list_capabilities", "read_artifact"} <= names

    def test_category_is_orchestration(self):
        """They hand work to another runtime rather than fetching data.

        The category does NOT by itself keep them out of subagents — TOOL_PROFILES
        includes orchestration in the api and bigquery profiles — so subagent.py names
        them; tests/test_subagent.py pins that.
        """
        by_name = {t["name"]: t for t in TOOL_DEFINITIONS}
        assert by_name["list_capabilities"]["category"] == "orchestration"
        assert by_name["read_artifact"]["category"] == "orchestration"

    def test_read_artifact_takes_a_name_not_a_path(self):
        """The parameter name is load-bearing: an execution id or path is never accepted."""
        by_name = {t["name"]: t for t in TOOL_DEFINITIONS}
        params = by_name["read_artifact"]["parameters"]
        assert set(params) == {"name"}
        assert params["name"]["required"] is True

    def test_anthropic_schema_shape(self):
        tools = {t["name"]: t for t in get_anthropic_tools()}
        assert tools["read_artifact"]["input_schema"]["required"] == ["name"]
        capabilities = tools["list_capabilities"]["input_schema"]
        assert capabilities["required"] == []
        assert capabilities["properties"]["module"]["enum"] == ["genetics", "client", "errors"]


class TestListCapabilities:
    async def test_index_lists_every_module(self, executor):
        result = await executor.list_capabilities()
        assert result["success"] is True
        assert [m["module"] for m in result["modules"]] == ["genetics", "client", "errors"]
        assert all(m["summary"] for m in result["modules"])

    async def test_index_covers_the_whole_sdk_export_list(self, executor):
        """The catalogue is derived from the SDK, so a new dataset function shows up free."""
        from genetics_mcp_server import sdk

        result = await executor.list_capabilities()
        genetics = next(m for m in result["modules"] if m["module"] == "genetics")
        assert set(sdk._FUNCTIONS) <= set(genetics["names"])

    async def test_genetics_module_renders_sync_signatures(self, executor):
        result = await executor.list_capabilities(module="genetics")
        assert result["success"] is True
        signatures = result["signatures"]
        assert "def credible_sets(" in signatures
        assert "async def" not in signatures
        # `self` belongs to the client form, not to what a script calls
        assert "(self" not in signatures
        # the evaluated annotation is rewritten to the form a script actually writes
        assert "-> pl.DataFrame" in signatures
        assert "polars.dataframe" not in signatures
        # docstrings are what make the catalogue worth fetching
        assert "Fine-mapped credible sets" in signatures

    async def test_client_module_renders_awaitables(self, executor):
        result = await executor.list_capabilities(module="client")
        assert "async def credible_sets(" in result["signatures"]
        assert "(self" not in result["signatures"]

    async def test_errors_module_renders_classes(self, executor):
        result = await executor.list_capabilities(module="errors")
        assert "class GeneticsError(RuntimeError):" in result["signatures"]
        assert "class GeneticsUsageError(GeneticsError):" in result["signatures"]

    async def test_unknown_module_is_refused_with_the_valid_names(self, executor):
        result = await executor.list_capabilities(module="genetics_mcp_server")
        assert result["success"] is False
        assert "genetics, client, errors" in result["error"]

    async def test_discloses_no_credentials(self, executor, monkeypatch):
        monkeypatch.setenv("INTERNAL_API_SECRET", "super-secret-value")
        for module in (None, "genetics", "client", "errors"):
            result = await executor.list_capabilities(module=module)
            assert "super-secret-value" not in str(result)

    async def test_module_docstrings_are_not_rendered(self, executor):
        """Module docs describe the deployment, not the SDK's shape.

        list_capabilities is reachable from MCP by design, so what it returns is
        per-function signatures and docstrings only. sdk.__doc__ and friends name the
        credential and endpoint env vars, the internal services, the per-execution quotas
        and the sandbox itself — none of which a script needs to write a call.
        """
        # everything below reached the output ONLY through a module docstring, so stripping
        # those removes it. It is not the whole of the internal vocabulary in the output:
        # *function* docstrings are written to describe the SDK, so they still disclose SDK
        # internals by category — the settings mechanism (`_URL_SETTINGS`, and that URLs
        # come from the environment), internal service names (db-api, the FinnGen LD
        # server, the sandbox), the server-side execution model behind `limit=`, and the
        # row/byte limit values — and those are what the catalogue exists to render. The
        # examples are illustrative, not an enumeration; enumerating them precisely has
        # been wrong twice. Removing them means rewriting SDK docstrings, which is a
        # separate decision and would also drift the generated sandbox stubs — so the
        # justification must not claim they are gone.
        forbidden = (
            "INTERNAL_API_SECRET",
            "GENETICS_API_URL",
            "BIGQUERY_API_URL",
            "results-api",
            "genetics-results-suite-6uk",
        )
        for module in (None, "genetics", "client", "errors"):
            result = await executor.list_capabilities(module=module)
            assert "doc" not in result
            rendered = str(result)
            for token in forbidden:
                assert token not in rendered, f"{token!r} leaked for module {module!r}"

    @pytest.mark.parametrize("module", [None, "genetics", "client", "errors"])
    async def test_every_response_says_how_to_import_the_sdk(self, executor, module):
        """genetics-results-suite-706: the catalogue is the only reachable place that can.

        `sdk.__doc__` carries the import line and is deliberately stripped (the test above),
        so a script's author had no way to learn it — every session opened with
        `import genetics` -> ModuleNotFoundError and several `pkgutil` probes. The line used
        to be on the index only, which is the response a model that already knows its module
        never asks for.
        """
        result = await executor.list_capabilities(module=module)
        assert "import genetics" in result["usage"]

    async def test_index_summaries_do_not_come_from_module_docstrings(self, executor):
        from genetics_mcp_server import sdk

        result = await executor.list_capabilities()
        summaries = {m["module"]: m["summary"] for m in result["modules"]}
        assert summaries["genetics"] not in (sdk.__doc__ or "")


EXEC_A = "8f14e45f-ceea-467a-a3d3-6f1b1b1b1b1b"
EXEC_B = "1c1c1c1c-2d2d-4e4e-8f8f-9a9a9a9a9a9a"


class _ArtifactSandbox:
    """Stands in for SandboxClient's artifact half. Replays one outcome per (id, name)."""

    def __init__(self, answers=None):
        # (execution_id, name) -> ArtifactResult, or name -> ArtifactResult
        self.answers = answers or {}
        self.asked = []

    async def get_artifact(self, execution_id, name):
        self.asked.append((execution_id, name))
        answer = self.answers.get((execution_id, name), self.answers.get(name))
        if answer is None:
            return ArtifactResult(ok=False, name=name, status_code=404, error_type="NotFound")
        return answer

    async def fetch_artifact(self, execution_id, name):
        result = await self.get_artifact(execution_id, name)
        if not result.ok or result.data is None:
            return None
        return {
            "name": name,
            "content_type": result.content_type,
            "content_base64": result.content_base64,
            "size": len(result.data),
        }


def _served(name, data, content_type=None):
    return ArtifactResult(
        ok=True,
        name=name,
        status_code=200,
        content_type=content_type,
        data=data,
        content_base64=base64.b64encode(data).decode("ascii"),
    )


def _record(session_id, execution_id, *names):
    executor_module._ARTIFACT_MANIFESTS.record(session_id, execution_id, names)


class TestReadArtifactProxiesOverHTTP:
    """genetics-results-suite-4h6.52. The read leaves this process; nothing local is opened."""

    async def test_reads_a_text_artifact_of_this_session(self, executor):
        _record("conv-9", EXEC_A, "hits.tsv")
        sandbox = _ArtifactSandbox({"hits.tsv": _served("hits.tsv", b"rsid\tpval\n")})
        executor._sandbox = sandbox

        result = await executor.read_artifact(name="hits.tsv", session_id="conv-9")
        assert result["success"] is True
        assert result["encoding"] == "utf-8"
        assert result["content"] == "rsid\tpval\n"
        assert result["size"] == 10
        assert result["truncated"] is False
        assert sandbox.asked == [(EXEC_A, "hits.tsv")]

    async def test_binary_comes_back_base64(self, executor):
        _record("conv-9", EXEC_A, "manhattan.png")
        executor._sandbox = _ArtifactSandbox(
            {"manhattan.png": _served("manhattan.png", PNG_HEADER, "image/png")}
        )
        result = await executor.read_artifact(name="manhattan.png", session_id="conv-9")
        assert result["encoding"] == "base64"
        assert base64.b64decode(result["content"]) == PNG_HEADER
        assert result["content_type"] == "image/png"

    async def test_the_model_never_supplies_an_execution_id(self):
        """The declared schema is one parameter. The id is resolved server-side or nowhere."""
        by_name = {t["name"]: t for t in TOOL_DEFINITIONS}
        params = by_name["read_artifact"]["parameters"]
        assert set(params) == {"name"}
        signature = inspect.signature(ToolExecutor.read_artifact)
        assert signature.parameters["session_id"].kind is inspect.Parameter.KEYWORD_ONLY

    async def test_no_local_filesystem_read_happens(self, executor, tmp_path, monkeypatch):
        """The whole point of the change: chat-backend's own /data holds chat_history.db.

        Pointing the old environment variable at a directory that HAS the name asked for must
        produce nothing, because there is no reader left to point.
        """
        planted = tmp_path / "artifacts"
        planted.mkdir()
        (planted / "hits.tsv").write_text("crown jewels")
        monkeypatch.setenv("SANDBOX_ARTIFACTS_DIR", str(planted))
        monkeypatch.setenv("SUBAGENT_ALLOWED_PATHS", str(tmp_path))
        executor._sandbox = _ArtifactSandbox()

        result = await executor.read_artifact(name="hits.tsv", session_id="conv-9")
        assert result["success"] is False
        assert "crown jewels" not in str(result)
        # nothing was even asked of the sandbox: the name resolves to no execution
        assert executor._sandbox.asked == []

    @pytest.mark.parametrize(
        "name",
        [
            "../secret.txt",
            "/etc/passwd",
            "sub/dir.txt",
            "back\\slash.txt",
            ".",
            "..",
            "  ",
            "",  # the empty name, refused before the manifest lookup
            "hits.tsv\x00.png",  # NUL truncation: the C-level name would be "hits.tsv"
            "../../chat_history.db",  # the crown jewels, by the traversal that names them
        ],
    )
    async def test_rejects_anything_that_is_not_a_bare_name(self, executor, name):
        _record("conv-9", EXEC_A, name)
        executor._sandbox = _ArtifactSandbox({name: _served(name, b"x")})
        result = await executor.read_artifact(name=name, session_id="conv-9")
        assert result["success"] is False
        assert executor._sandbox.asked == []


class TestReadArtifactSessionScoping:
    async def test_another_sessions_artifact_is_indistinguishable_from_a_missing_one(
        self, executor
    ):
        """Byte-for-byte the same answer, because "that name exists somewhere" is itself a
        cross-session fact.
        """
        _record("other-conv", EXEC_A, "secrets.tsv")
        executor._sandbox = _ArtifactSandbox({"secrets.tsv": _served("secrets.tsv", b"data")})

        theirs = await executor.read_artifact(name="secrets.tsv", session_id="conv-9")
        never = await executor.read_artifact(name="no-such-file.tsv", session_id="conv-9")
        assert theirs["success"] is False
        assert theirs["error"].replace("secrets.tsv", "X") == never["error"].replace(
            "no-such-file.tsv", "X"
        )
        assert theirs["error_type"] == never["error_type"] == "ArtifactNotFound"
        assert executor._sandbox.asked == []

    async def test_a_name_collision_resolves_to_the_most_recent_execution(self, executor):
        _record("conv-9", EXEC_A, "manhattan.png")
        _record("conv-9", EXEC_B, "manhattan.png")
        executor._sandbox = _ArtifactSandbox(
            {
                (EXEC_A, "manhattan.png"): _served("manhattan.png", b"old"),
                (EXEC_B, "manhattan.png"): _served("manhattan.png", b"new"),
            }
        )
        result = await executor.read_artifact(name="manhattan.png", session_id="conv-9")
        assert result["content"] == "new"
        assert executor._sandbox.asked == [(EXEC_B, "manhattan.png")]

    async def test_an_older_execution_still_answers_for_a_name_the_newer_one_lacks(
        self, executor
    ):
        _record("conv-9", EXEC_A, "hits.tsv")
        _record("conv-9", EXEC_B, "manhattan.png")
        executor._sandbox = _ArtifactSandbox({"hits.tsv": _served("hits.tsv", b"rows")})
        result = await executor.read_artifact(name="hits.tsv", session_id="conv-9")
        assert result["success"] is True
        assert executor._sandbox.asked == [(EXEC_A, "hits.tsv")]

    async def test_a_call_with_no_session_reads_nothing(self, executor):
        _record("conv-9", EXEC_A, "hits.tsv")
        executor._sandbox = _ArtifactSandbox({"hits.tsv": _served("hits.tsv", b"rows")})
        result = await executor.read_artifact(name="hits.tsv", session_id=None)
        assert result["success"] is False
        assert executor._sandbox.asked == []

    async def test_run_analysis_is_what_records_the_mapping(self, executor):
        body = _result_body(
            artifacts=[{"name": "hits.tsv", "size": 4, "content_type": "text/tab-separated-values"}]
        )
        sandbox = _ArtifactSandbox({"hits.tsv": _served("hits.tsv", b"rows")})
        sandbox.result = body
        sandbox.calls = []

        async def execute(**kwargs):
            return body

        sandbox.execute = execute
        executor._sandbox = sandbox
        await executor.run_analysis(code="x", user="u@finngen.fi", session_id="conv-9")

        result = await executor.read_artifact(name="hits.tsv", session_id="conv-9")
        assert result["success"] is True
        # ...and only for that session
        assert (await executor.read_artifact(name="hits.tsv", session_id="conv-8"))[
            "success"
        ] is False


class TestReadArtifactLifetime:
    """The map must not outlive what it points at (RETENTION_S in sandbox/supervisor.py)."""

    def test_the_ttl_matches_the_supervisors_retention(self):
        assert executor_module.ARTIFACT_RETENTION_S == 300
        assert executor_module._ArtifactManifests()._ttl_s == 300

    async def test_a_row_expires_with_the_artifact_it_points_at(self, executor, monkeypatch):
        registry = executor_module._ArtifactManifests(ttl_s=60)
        monkeypatch.setattr(executor_module, "_ARTIFACT_MANIFESTS", registry)
        clock = {"now": 1000.0}
        monkeypatch.setattr(executor_module.time, "monotonic", lambda: clock["now"])

        registry.record("conv-9", EXEC_A, ["hits.tsv"])
        executor._sandbox = _ArtifactSandbox({"hits.tsv": _served("hits.tsv", b"rows")})
        assert (await executor.read_artifact(name="hits.tsv", session_id="conv-9"))["success"]

        clock["now"] += 61
        expired = await executor.read_artifact(name="hits.tsv", session_id="conv-9")
        assert expired["success"] is False
        assert expired["error_type"] == "ArtifactNotFound"

    def test_the_map_is_bounded_in_both_dimensions(self):
        registry = executor_module._ArtifactManifests()
        for i in range(registry._MAX_SESSIONS + 50):
            registry.record(f"conv-{i}", EXEC_A, ["a.txt"])
        assert len(registry._sessions) == registry._MAX_SESSIONS
        # the oldest sessions were dropped, the newest kept
        assert "conv-0" not in registry._sessions
        assert f"conv-{registry._MAX_SESSIONS + 49}" in registry._sessions

        for i in range(registry._MAX_EXECUTIONS + 20):
            registry.record("conv-x", f"{i:08d}-0000-4000-8000-000000000000", ["a.txt"])
        assert len(registry._sessions["conv-x"]) == registry._MAX_EXECUTIONS

    def test_nothing_persists_the_map(self):
        """A persisted row would outlive both the artifacts and the supervisor-memory key
        that decrypts them, promising reads that can only 404 or 409.
        """
        source = inspect.getsource(executor_module._ArtifactManifests)
        assert "sqlite" not in source.lower() and "open(" not in source


class TestReadArtifactCaps:
    def test_the_byte_cap_is_the_transports(self):
        """4 MiB -> 512 KiB, a deliberate REDUCTION: the supervisor 413s above its own cap,
        so a larger number here would only promise bytes the sandbox refuses.
        """
        from genetics_mcp_server.sandbox_client import ARTIFACT_READ_MAX_BYTES

        assert ToolExecutor._MAX_ARTIFACT_BYTES == ARTIFACT_READ_MAX_BYTES == 512 * 1024
        assert executor_module.ARTIFACT_READ_MAX_BYTES == ARTIFACT_READ_MAX_BYTES

    async def test_long_text_is_still_truncated_at_the_character_cap(self, executor):
        """Survives the move because it bounds the MODEL'S CONTEXT, which no transport cap
        does: 512 KiB of TSV is well over 100k tokens in one tool result.
        """
        _record("conv-9", EXEC_A, "big.txt")
        payload = ("x" * (ToolExecutor._MAX_ARTIFACT_TEXT_CHARS + 500)).encode()
        executor._sandbox = _ArtifactSandbox({"big.txt": _served("big.txt", payload)})
        result = await executor.read_artifact(name="big.txt", session_id="conv-9")
        assert result["truncated"] is True
        assert len(result["content"]) == ToolExecutor._MAX_ARTIFACT_TEXT_CHARS
        assert result["size"] == len(payload)

    async def test_a_413_tells_the_model_to_write_a_smaller_file(self, executor):
        _record("conv-9", EXEC_A, "huge.bin")
        executor._sandbox = _ArtifactSandbox(
            {
                "huge.bin": ArtifactResult(
                    ok=False, name="huge.bin", status_code=413, error_type="ArtifactTooLarge"
                )
            }
        )
        result = await executor.read_artifact(name="huge.bin", session_id="conv-9")
        assert result["error_type"] == "ArtifactTooLarge"
        assert result["retryable"] is False
        assert str(ToolExecutor._MAX_ARTIFACT_BYTES) in result["error"]


class TestReadArtifactErrorMapping:
    """409 and 413 reach this tool for the first time; neither may read as a generic failure."""

    async def test_a_409_is_reported_as_a_modified_artifact_and_is_not_retryable(self, executor):
        _record("conv-9", EXEC_A, "hits.tsv")
        executor._sandbox = _ArtifactSandbox(
            {
                "hits.tsv": ArtifactResult(
                    ok=False, name="hits.tsv", status_code=409, error_type="ArtifactModified"
                )
            }
        )
        result = await executor.read_artifact(name="hits.tsv", session_id="conv-9")
        assert result["success"] is False
        assert result["error_type"] == "ArtifactModified"
        assert result["retryable"] is False
        assert "re-run" in result["error"].lower()
        assert "not found" not in result["error"].lower()

    async def test_a_404_from_the_sandbox_says_the_window_may_have_passed(self, executor):
        _record("conv-9", EXEC_A, "gone.tsv")
        executor._sandbox = _ArtifactSandbox()
        result = await executor.read_artifact(name="gone.tsv", session_id="conv-9")
        assert result["error_type"] == "ArtifactNotFound"
        assert result["retryable"] is False

    async def test_an_unreachable_sandbox_is_retryable_and_not_a_missing_name(self, executor):
        _record("conv-9", EXEC_A, "hits.tsv")
        executor._sandbox = _ArtifactSandbox(
            {
                "hits.tsv": ArtifactResult(
                    ok=False, name="hits.tsv", error_type="SandboxUnreachable"
                )
            }
        )
        result = await executor.read_artifact(name="hits.tsv", session_id="conv-9")
        assert result["error_type"] == "ArtifactUnavailable"
        assert result["retryable"] is True

    async def test_a_malformed_200_body_is_not_retryable(self):
        """A 200 the client could not parse is DETERMINISTIC: the same id and name produce the
        same body, so re-asking spends a model roundtrip on something that cannot succeed. This
        tool has no total budget above it the way run_analysis has its 300s wait_for.
        """
        from genetics_mcp_server.sandbox_client import ERROR_MALFORMED_RESPONSE

        executor = ToolExecutor()
        result = executor._artifact_error(
            "hits.tsv",
            ArtifactResult(
                ok=False, name="hits.tsv", status_code=200,
                error_type=ERROR_MALFORMED_RESPONSE,
            ),
        )
        assert result["error_type"] == "ArtifactUnavailable"
        assert result["retryable"] is False
        assert "Do not retry" in result["error"]

    async def test_a_locally_rejected_execution_id_is_not_retryable(self):
        """The client's own pre-flight refused the id and issued NO request, so a retry
        re-rejects the identical id. It is our bug, not the model's, so it gets the
        indistinguishable not-found wording rather than a description of our internals.
        """
        from genetics_mcp_server.sandbox_client import ERROR_BAD_EXECUTION_ID

        executor = ToolExecutor()
        result = executor._artifact_error(
            "hits.tsv",
            ArtifactResult(ok=False, name="hits.tsv", error_type=ERROR_BAD_EXECUTION_ID),
        )
        assert result["error_type"] == "ArtifactNotFound"
        assert result["retryable"] is False


class _StubSandbox:
    """Stands in for SandboxClient. Records what it was asked and replays one outcome."""

    def __init__(self, result=None, raises=None, artifacts=None):
        self.result = result
        self.raises = raises
        self.calls = []
        # name -> what fetch_artifact answers. The default is None for every name, which is
        # the real client's answer for a reaped, evicted, oversize or unservable artifact.
        self.artifacts = artifacts or {}
        self.fetched = []

    async def execute(self, *, code, user, session_id, timeout_s=None, execution_id=None):
        self.calls.append(
            {
                "code": code,
                "user": user,
                "session_id": session_id,
                "timeout_s": timeout_s,
                "execution_id": execution_id,
            }
        )
        if self.raises is not None:
            raise self.raises
        return self.result

    async def fetch_artifact(self, execution_id, name):
        self.fetched.append((execution_id, name))
        return self.artifacts.get(name)


def _result_body(**overrides):
    body = {
        "execution_id": "8f14e45f-ceea-467a-a3d3-6f1b1b1b1b1b",
        "status": "ok",
        "exit_code": 0,
        "signal": None,
        "duration_ms": 1234,
        "output": "rows: 12\n",
        "output_bytes": 9,
        "output_truncated": False,
        "error": None,
        "artifacts": [],
        "artifacts_omitted": 0,
    }
    body.update(overrides)
    return body


async def _run(executor, sandbox, **kwargs):
    executor._sandbox = sandbox
    kwargs.setdefault("code", "print('hi')")
    kwargs.setdefault("user", "u@finngen.fi")
    kwargs.setdefault("session_id", "conv-9")
    return await executor.run_analysis(**kwargs)


class TestRunAnalysisDefinition:
    def test_defined_as_an_orchestration_tool(self):
        by_name = {t["name"]: t for t in TOOL_DEFINITIONS}
        assert by_name["run_analysis"]["category"] == "orchestration"

    def test_takes_code_and_a_timeout_and_no_identity(self):
        """The identity is the caller's, never the model's: it is not an argument."""
        by_name = {t["name"]: t for t in TOOL_DEFINITIONS}
        params = by_name["run_analysis"]["parameters"]
        assert params["code"]["required"] is True
        assert params["timeout_s"]["type"] == "integer"
        assert "user" not in params and "session_id" not in params
        # nor the provenance of the identity, for the same reason and one more: a schema key
        # is an invitation to emit it, and `_execute_tool` strips it precisely so a model
        # cannot vouch for its own caller
        assert "gateway_asserted" not in params

    def test_points_at_the_retrieval_path_that_now_exists(self):
        """genetics-results-suite-4h6.52 built it. The description said the opposite while
        the tool read a local directory that was never the sandbox's /scratch; saying so now
        would cost the model the retrieval it is entitled to.
        """
        by_name = {t["name"]: t for t in TOOL_DEFINITIONS}
        description = by_name["run_analysis"]["description"]
        assert "read_artifact" in description
        assert "CANNOT BE RETRIEVED" not in description

    def test_reaches_the_chat_tool_list_when_the_sandbox_is_enabled(self):
        """Registered unconditionally; it is `disabled_tools` that withholds it, so the
        unfiltered registry is NOT what chat resolves (see TestRunAnalysisSandboxFlag).
        """
        assert "run_analysis" in {t["name"] for t in get_anthropic_tools()}
        enabled = Settings(sandbox_enabled=True).disabled_tools
        names = {t["name"] for t in get_anthropic_tools(disabled_tools=enabled)}
        assert "run_analysis" in names


class TestRunAnalysisSandboxFlag:
    """genetics-results-suite-4h6.56: no sandbox deployed, no tool offered.

    The failure this prevents is not "a tool errors". An unreachable sandbox classifies as
    SandboxUnavailable with `retryable: True`, and the prompt tells the model to PREFER
    run_analysis — so before the flag, every chat turn on a sandbox-less deployment was
    steered into a tool that always failed and asked to be retried.
    """

    def test_the_flag_is_off_unless_the_environment_says_otherwise(self, monkeypatch):
        monkeypatch.delenv("SANDBOX_ENABLED", raising=False)
        assert Settings().sandbox_enabled is False
        monkeypatch.setenv("SANDBOX_ENABLED", "true")
        assert Settings().sandbox_enabled is True

    def test_the_default_deployment_withholds_the_tool(self, monkeypatch):
        monkeypatch.delenv("SANDBOX_ENABLED", raising=False)
        assert "run_analysis" in Settings().disabled_tools

    def test_no_profile_can_put_it_back(self):
        """disabled_tools is applied BEFORE the profile filter, including for the
        name-listed `code` profile, so the flag is not something a request can route
        around by asking for a different surface.
        """
        disabled = Settings(sandbox_enabled=False).disabled_tools
        for profile in (None, "api", "bigquery", "rag", "code"):
            names = {
                t["name"]
                for t in get_anthropic_tools(tool_profile=profile, disabled_tools=disabled)
            }
            assert "run_analysis" not in names, profile

    def test_the_other_two_code_tools_are_not_gated(self):
        """They are inert without a sandbox rather than broken by it — list_capabilities
        renders from the local SDK and read_artifact reads this process's own directory —
        and neither is a tool the prompt steers toward, which is what made run_analysis
        expensive to ship early.
        """
        disabled = Settings(sandbox_enabled=False).disabled_tools
        assert "list_capabilities" not in disabled
        assert "read_artifact" not in disabled

    def test_the_flag_reaches_the_service_that_advertises_tools(self, monkeypatch):
        """settings.disabled_tools is what llm_service resolves against, so the gate has
        to hold through that path and not only in the settings object.
        """
        from genetics_mcp_server.config import settings as settings_module
        from genetics_mcp_server.llm_service import LLMService

        service = LLMService.__new__(LLMService)
        service.subagent_service = None
        monkeypatch.delenv("SANDBOX_ENABLED", raising=False)
        settings_module.get_settings.cache_clear()
        try:
            assert "run_analysis" in service._disabled_tools()
        finally:
            settings_module.get_settings.cache_clear()

    async def test_a_model_emitted_call_is_refused_without_dispatching(self, monkeypatch):
        """Withholding the tool from the list is not enough on its own.

        _execute_tool resolves the handler by getattr on the executor, so a tool_use the
        model emits anyway reaches the sandbox client regardless of what was advertised —
        and the model does not have to invent the name: a client-supplied history with a
        paired run_analysis tool_use/tool_result survives _sanitize_tool_blocks verbatim
        and primes it. In a deployment with no sandbox that lands on SandboxUnavailable
        with retryable: True, which is the exact failure 4h6.56 exists to prevent.
        """
        from genetics_mcp_server.config import settings as settings_module
        from genetics_mcp_server.llm_service import LLMService

        dispatched = []

        class _Recorder:
            async def run_analysis(self, **kwargs):
                dispatched.append(kwargs)
                return {"success": True}

        service = LLMService.__new__(LLMService)
        service.executor = _Recorder()
        service.subagent_service = None

        monkeypatch.delenv("SANDBOX_ENABLED", raising=False)
        settings_module.get_settings.cache_clear()
        try:
            result = await service._execute_tool(
                "run_analysis", {"code": "print(1)"}, None, "real@finngen.fi", "conv-7"
            )
        finally:
            settings_module.get_settings.cache_clear()

        assert dispatched == []
        assert result["success"] is False
        # the refusal must not read as transient, or the model spends the turn retrying
        assert result["retryable"] is False
        assert result["error_type"] == "SandboxNotConfigured"
        assert _script_result_payload(1, result)["status"] == "SandboxNotConfigured"


class TestRunAnalysisFailsClosed:
    """The one hazard that is neither a script failure nor a security hole.

    An unset SANDBOX_TOKEN_SIGNING_KEY makes every execution impossible. No credential is
    ever sent — the client raises before the request — so this cannot leak anything. What
    it must not do is arrive at the model as an ordinary "tool failed", because the model
    then retries, repeatedly, against a sandbox that can never work.
    """

    async def test_an_unset_signing_key_is_an_operator_error_and_sends_nothing(
        self, executor, monkeypatch
    ):
        import httpx

        from genetics_mcp_server.config import settings as settings_module
        from genetics_mcp_server.sandbox_client import SandboxClient

        settings_module.get_settings.cache_clear()
        monkeypatch.delenv("SANDBOX_TOKEN_SIGNING_KEY", raising=False)
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(200, json=_result_body())

        try:
            executor._sandbox = SandboxClient(
                "http://sandbox:8080", transport=httpx.MockTransport(handler)
            )
            result = await executor.run_analysis(
                code="print(1)", user="u@finngen.fi", session_id="conv-9"
            )
        finally:
            settings_module.get_settings.cache_clear()

        assert requests == [], "fail-closed must mean no request at all"
        assert result["success"] is False
        assert result["error_type"] == "SandboxNotConfigured"
        assert result["retryable"] is False

    async def test_an_unset_sandbox_url_is_an_operator_error_and_sends_nothing(
        self, executor, monkeypatch
    ):
        """The sibling of the signing-key case (genetics-results-suite-6um).

        SANDBOX_URL has no default any more, so an unconfigured deployment cannot build a
        client at all. SandboxNotConfigured is a SandboxError subclass and the family's
        fallback clause answers `retryable: True`; this pins the non-retryable answer, so
        an edit that let the generic clause take this case fails here instead of silently
        flipping the payload the model reads.
        """
        from genetics_mcp_server.config import settings as settings_module

        monkeypatch.delenv("SANDBOX_URL", raising=False)
        settings_module.get_settings.cache_clear()
        try:
            result = await executor.run_analysis(
                code="print(1)", user="u@finngen.fi", session_id="conv-9"
            )
        finally:
            settings_module.get_settings.cache_clear()

        assert result["success"] is False
        assert result["error_type"] == "SandboxNotConfigured"
        assert result["retryable"] is False

    async def test_read_artifact_answers_the_same_way_rather_than_raising(
        self, executor, monkeypatch
    ):
        """The second toucher of `_sandbox`, and it has no try/except of its own.

        The manifest lookup succeeds before the transport is ever needed, so without the
        shared guard the cached_property's constructor raises straight out of the tool
        call — a bare exception where every other failure of this tool is a dict.
        """
        from genetics_mcp_server.config import settings as settings_module

        _record("conv-9", EXEC_A, "hits.tsv")
        monkeypatch.delenv("SANDBOX_URL", raising=False)
        settings_module.get_settings.cache_clear()
        try:
            result = await executor.read_artifact(name="hits.tsv", session_id="conv-9")
        finally:
            settings_module.get_settings.cache_clear()

        assert result["success"] is False
        assert result["error_type"] == "SandboxNotConfigured"
        assert result["retryable"] is False

    async def test_the_house_error_style_does_not_swallow_it(self, executor):
        """~40 handlers in this module wrap their body in `except Exception`, and
        SandboxTokenUnavailable is a plain RuntimeError. Written that way, this handler
        would report the generic internal-error message and the model would retry it.
        """
        from genetics_mcp_server.sandbox_token import SandboxTokenUnavailable

        sandbox = _StubSandbox(raises=SandboxTokenUnavailable("no signing key"))
        result = await _run(executor, sandbox)

        assert result["error"] != executor_module.INTERNAL_ERROR_MSG
        assert result["error_type"] == "SandboxNotConfigured"
        assert result["retryable"] is False

    def test_the_handler_has_no_blanket_exception_clause(self):
        """Structural, because ordering is what would break: a later edit that adds
        `except Exception` above the named clause reintroduces the swallow silently, and
        no behavioural test of the current ordering would notice.
        """
        import ast
        import textwrap

        tree = ast.parse(textwrap.dedent(inspect.getsource(ToolExecutor.run_analysis)))
        tries = [node for node in ast.walk(tree) if isinstance(node, ast.Try)]
        assert len(tries) == 1
        caught = [
            ast.unparse(handler.type) if handler.type else "BARE"
            for handler in tries[0].handlers
        ]
        assert "Exception" not in caught and "BARE" not in caught
        assert caught[0] == "SandboxTokenUnavailable", (
            "the fail-closed exception must be caught before anything broader"
        )

    async def test_a_missing_identity_is_also_an_operator_error(self, executor):
        """Nothing to mint against. A wiring fault, not something the model can fix."""
        sandbox = _StubSandbox(result=_result_body())
        result = await _run(executor, sandbox, user=None)
        assert result["error_type"] == "SandboxNotConfigured"
        assert result["retryable"] is False
        assert sandbox.calls == []


class TestRunAnalysisTurnBudget:
    """4h6.47 bounds each ATTEMPT and deliberately offers no total; this layer owns it."""

    def test_the_cap_admits_one_full_attempt_and_bounds_the_retry_chain(self):
        from genetics_mcp_server import sandbox_client

        worst_single_attempt = (
            sandbox_client.CONNECT_TIMEOUT_S
            + sandbox_client.BODY_WRITE_DEADLINE_S
            + sandbox_client.client_deadline_s(sandbox_client.MAX_TIMEOUT_S)
        )
        worst_uncapped = (
            worst_single_attempt + sandbox_client.MAX_RETRY_WAIT_S + worst_single_attempt
        )
        deadline = ToolExecutor._RUN_ANALYSIS_DEADLINE_S

        assert deadline >= worst_single_attempt, (
            "a cap below one attempt's worst case would abandon executions the supervisor "
            "is about to answer"
        )
        assert deadline < worst_uncapped, "the whole point is that the attempts do not sum"

    async def test_exceeding_it_is_not_reported_as_a_script_failure(self, executor, monkeypatch):
        monkeypatch.setattr(ToolExecutor, "_RUN_ANALYSIS_DEADLINE_S", 0.05)

        class _Hangs:
            async def execute(self, **kwargs):
                await asyncio.sleep(10)

        executor._sandbox = _Hangs()
        result = await executor.run_analysis(
            code="print(1)", user="u@finngen.fi", session_id="conv-9"
        )
        assert result["success"] is False
        assert result["error_type"] == "TurnBudgetExceeded"
        assert result["retryable"] is True
        assert "may still be running" in result["error"]


class TestRunAnalysisTransportFailures:
    async def test_an_unavailable_sandbox_does_not_read_as_a_broken_script(self, executor):
        """strategy Recreate + terminationGracePeriodSeconds 130 leaves no sandbox for
        ~130s when a deploy lands on an in-flight execution. That is a wait-and-retry
        condition, not a defect in the code the model wrote.
        """
        from genetics_mcp_server.sandbox_client import SandboxUnavailable

        result = await _run(executor, _StubSandbox(raises=SandboxUnavailable("no route")))
        assert result["error_type"] == "SandboxUnavailable"
        assert result["retryable"] is True
        assert "not a problem with the script" in result["error"]

    async def test_busy_carries_the_retry_after(self, executor):
        from genetics_mcp_server.sandbox_client import SandboxBusy

        result = await _run(executor, _StubSandbox(raises=SandboxBusy("full", retry_after=42)))
        assert result["error_type"] == "SandboxBusy"
        assert result["retry_after_s"] == 42

    async def test_a_rejection_is_not_retryable(self, executor):
        from genetics_mcp_server.sandbox_client import SandboxRejected

        result = await _run(
            executor, _StubSandbox(raises=SandboxRejected("timeout_s must be 1-120"))
        )
        assert result["error_type"] == "SandboxRejected"
        assert result["retryable"] is False

    async def test_an_unrecognised_member_of_the_family_is_still_reported(self, executor):
        """SandboxError subclasses are not a closed set either."""
        from genetics_mcp_server.sandbox_client import SandboxError

        class _NewFailure(SandboxError):
            pass

        result = await _run(executor, _StubSandbox(raises=_NewFailure("something new")))
        assert result["success"] is False
        assert result["error_type"] == "_NewFailure"


class TestRunAnalysisRendering:
    async def test_a_successful_run_returns_what_the_script_printed(self, executor):
        result = await _run(executor, _StubSandbox(result=_result_body()))
        assert result["success"] is True
        assert result["status"] == "ok"
        assert result["output"] == "rows: 12\n"
        assert result["duration_ms"] == 1234

    async def test_the_execution_id_never_reaches_the_model(self, executor):
        """It is the audit join key and the manifest's key. A model-visible id invites a
        model-SUPPLIED one back in, which is what artifact resolution rules out.
        """
        result = await _run(executor, _StubSandbox(result=_result_body()))
        assert "8f14e45f" not in str(result)

    async def test_the_manifest_is_name_size_content_type_and_nothing_else(self, executor):
        body = _result_body(
            artifacts=[
                {
                    "name": "manhattan.png",
                    "size": 2048,
                    "content_type": "image/png",
                    "path": "/scratch/8f14e45f/artifacts/manhattan.png",
                    "url": "http://sandbox/artifacts/8f14e45f/manhattan.png",
                }
            ]
        )
        result = await _run(executor, _StubSandbox(result=body))
        assert result["artifacts"] == [
            {"name": "manhattan.png", "size": 2048, "content_type": "image/png"}
        ]
        assert "/scratch/" not in str(result)

    async def test_the_manifest_says_how_to_read_the_contents(self, executor):
        body = _result_body(
            artifacts=[{"name": "plot.png", "size": 10, "content_type": "image/png"}]
        )
        result = await _run(executor, _StubSandbox(result=body))
        assert "read_artifact" in result["artifacts_note"]
        assert "5 minutes" in result["artifacts_note"]


def _image(name="plot.png", content_type="image/png", data="aW1hZ2UtYnl0ZXM="):
    return {"name": name, "content_type": content_type, "content_base64": data, "size": 11}


class TestRunAnalysisImages:
    """Image artifacts come back automatically; nothing else does.

    Automatic because the picture is for the USER: routing it through a tool the model calls
    spends a roundtrip fetching something the model cannot look at. The base64 rides on the
    result under `images` and llm_service strips it before the tool_result is serialised —
    tested there, since this layer is what produces it.
    """

    async def test_an_image_artifact_is_fetched_and_attached(self, executor):
        body = _result_body(
            artifacts=[{"name": "plot.png", "size": 11, "content_type": "image/png"}]
        )
        sandbox = _StubSandbox(result=body, artifacts={"plot.png": _image()})
        result = await _run(executor, sandbox)

        assert [i["name"] for i in result["images"]] == ["plot.png"]
        assert sandbox.fetched == [("8f14e45f-ceea-467a-a3d3-6f1b1b1b1b1b", "plot.png")]
        assert "displayed to the user" in result["artifacts_note"]

    async def test_only_image_content_types_are_fetched(self, executor):
        body = _result_body(
            artifacts=[
                {"name": "table.csv", "size": 9, "content_type": "text/csv"},
                {"name": "plot.png", "size": 11, "content_type": "image/png"},
                {"name": "notes.txt", "size": 4, "content_type": None},
            ]
        )
        sandbox = _StubSandbox(result=body, artifacts={"plot.png": _image()})
        result = await _run(executor, sandbox)

        assert [name for _, name in sandbox.fetched] == ["plot.png"]
        # the note has to say BOTH things: the plot is shown, the csv is read by name
        assert "displayed to the user" in result["artifacts_note"]
        assert "read_artifact" in result["artifacts_note"]

    async def test_an_oversize_image_is_not_even_requested(self, executor):
        from genetics_mcp_server.sandbox_client import ARTIFACT_READ_MAX_BYTES

        body = _result_body(
            artifacts=[
                {
                    "name": "huge.png",
                    "size": ARTIFACT_READ_MAX_BYTES + 1,
                    "content_type": "image/png",
                }
            ]
        )
        sandbox = _StubSandbox(result=body)
        result = await _run(executor, sandbox)
        assert sandbox.fetched == []
        assert "images" not in result

    async def test_the_number_of_images_is_bounded(self, executor):
        names = [f"p{i}.png" for i in range(10)]
        body = _result_body(
            artifacts=[{"name": n, "size": 11, "content_type": "image/png"} for n in names]
        )
        sandbox = _StubSandbox(result=body, artifacts={n: _image(name=n) for n in names})
        result = await _run(executor, sandbox)
        assert len(result["images"]) == executor._MAX_ANALYSIS_IMAGES
        assert len(sandbox.fetched) == executor._MAX_ANALYSIS_IMAGES

    async def test_an_artifact_the_sandbox_will_not_serve_costs_only_the_picture(self, executor):
        """The reaper, the retained-size ceiling and a restart all answer None here."""
        body = _result_body(
            artifacts=[{"name": "plot.png", "size": 11, "content_type": "image/png"}]
        )
        result = await _run(executor, _StubSandbox(result=body, artifacts={}))
        assert result["success"] is True
        assert result["output"] == "rows: 12\n"
        assert "images" not in result
        # the note still points at read_artifact: this failure was the automatic IMAGE fetch,
        # and the by-name read may well succeed on a later attempt
        assert "read_artifact" in result["artifacts_note"]

    async def test_a_failed_run_fetches_nothing(self, executor):
        """A killed script's artifacts/ has been trimmed and the run has no picture to show;
        asking anyway spends a round trip per artifact on the path that is already slowest."""
        body = _result_body(
            status="error",
            error={"type": "ValueError", "message": "boom", "traceback": None, "limit": None},
            artifacts=[{"name": "plot.png", "size": 11, "content_type": "image/png"}],
        )
        sandbox = _StubSandbox(result=body, artifacts={"plot.png": _image()})
        result = await _run(executor, sandbox)
        assert sandbox.fetched == []
        assert "images" not in result

    async def test_the_execution_id_still_never_reaches_the_model(self, executor):
        """It is used to fetch and is not part of what is rendered."""
        body = _result_body(
            artifacts=[{"name": "plot.png", "size": 11, "content_type": "image/png"}]
        )
        result = await _run(
            executor, _StubSandbox(result=body, artifacts={"plot.png": _image()})
        )
        assert "8f14e45f" not in str(result)

    async def test_a_malformed_manifest_entry_is_dropped_not_fatal(self, executor):
        body = _result_body(artifacts=["plot.png", {"size": 3}, {"name": "ok.tsv"}])
        result = await _run(executor, _StubSandbox(result=body))
        assert [a["name"] for a in result["artifacts"]] == ["ok.tsv"]

    async def test_a_failing_script_carries_the_type_and_the_traceback(self, executor):
        body = _result_body(
            status="error",
            exit_code=1,
            output="Traceback...\n",
            error={
                "type": "ValueError",
                "message": "no such phenotype",
                "traceback": '  File "<script>", line 3\nValueError: no such phenotype',
                "limit": None,
            },
        )
        result = await _run(executor, _StubSandbox(result=body))
        assert result["success"] is False
        assert result["status"] == "error"
        assert result["error_type"] == "ValueError"
        assert "line 3" in result["traceback"]
        assert result["error"] == "no such phenotype"

    async def test_an_sdk_misuse_is_pointed_at_the_catalogue(self, executor):
        body = _result_body(
            status="error",
            error={"type": "TypeError", "message": "unexpected keyword 'pheno'"},
        )
        result = await _run(executor, _StubSandbox(result=body))
        assert "list_capabilities" in result["hint"]

    async def test_a_limit_says_which_limit_fired(self, executor):
        body = _result_body(
            status="limit",
            error={"type": "OutputLimit", "message": "output cap", "limit": "OutputLimit"},
        )
        result = await _run(executor, _StubSandbox(result=body))
        assert result["limit_exceeded"] == "OutputLimit"
        assert "summary" in result["hint"]

    async def test_an_unknown_error_type_is_rendered_not_switched_on(self, executor):
        """`error.type` is an OPEN string — half its range is the child's own exception
        class name — so an unrecognised value is a label to display, never a crash.
        """
        body = _result_body(
            status="limit",
            error={"type": "SomethingNew", "message": "?", "limit": "SomethingNew"},
        )
        result = await _run(executor, _StubSandbox(result=body))
        assert result["error_type"] == "SomethingNew"
        assert result["limit_exceeded"] == "SomethingNew"
        assert result["hint"]

    async def test_an_unknown_status_is_not_a_success(self, executor):
        result = await _run(executor, _StubSandbox(result=_result_body(status="quarantined")))
        assert result["success"] is False
        assert result["status"] == "quarantined"

    async def test_a_body_missing_optional_fields_does_not_crash(self, executor):
        result = await _run(executor, _StubSandbox(result={"status": "ok"}))
        assert result["success"] is True
        assert result["output"] == ""
        assert result["artifacts"] == []

    async def test_unknown_top_level_fields_are_ignored(self, executor):
        result = await _run(
            executor, _StubSandbox(result=_result_body(sandbox_host="sandbox-0", cost_usd=1))
        )
        assert result["success"] is True
        assert "sandbox-0" not in str(result)


class TestRunAnalysisRequiresARealUser:
    """genetics-results-suite-4h6.27: the MCP exclusion boundary is one hop long.

    The NetworkPolicy closes mcp-server -> sandbox, but mcp-server holds
    INTERNAL_API_SECRET and is admitted to chat-backend:8000, and chat-backend is the pod
    admitted to the sandbox. A valid marker with no identity header resolves to exactly
    `mcp-tool` (genetics-results-suite-th2), so "the caller authenticated" is not the
    property this dispatch needs — a real person is.
    """

    async def test_the_service_identity_cannot_execute_code(self, executor):
        from genetics_mcp_server.auth.core import SERVICE_IDENTITY

        sandbox = _StubSandbox(result=_result_body())
        result = await _run(executor, sandbox, user=SERVICE_IDENTITY)

        assert result["success"] is False
        assert result["error_type"] == "SandboxNotConfigured"
        assert result["retryable"] is False
        assert sandbox.calls == [], "nothing may be dispatched, and nothing minted"

    async def test_the_refusal_is_the_one_auth_required_actually_returns(self):
        """Pinned against the resolver rather than a literal: if case 3 ever returns a
        different string, this test fails instead of the guard silently missing it.
        """
        from genetics_mcp_server.auth import dependencies

        source = inspect.getsource(dependencies.auth_required)
        assert "return SERVICE_IDENTITY  # case 3" in source

    async def test_a_real_user_the_gateway_asserted_still_runs(self, executor):
        """The guard must be a rejection of one identity, not of service-shaped ones.

        QUALIFIED for genetics-results-suite-4h6.84, and the qualification is the point.
        This used to assert only `user="real@finngen.fi"` and could therefore not tell a
        browser session from an address any marker holder typed into the identity header —
        `auth_required` case 1 turns both into exactly this call. `gateway_asserted` is now
        the thing that separates them, so a positive test has to state it; the negative twin
        is TestSandboxDispatchRequiresTheGatewaySecret below.
        """
        sandbox = _StubSandbox(result=_result_body())
        with settings_env(REQUIRE_AUTH="true"):
            result = await _run(
                executor, sandbox, user="real@finngen.fi", gateway_asserted=True
            )
        assert result["success"] is True
        assert len(sandbox.calls) == 1

    def test_the_guard_runs_before_anything_is_minted(self):
        """Structural: the check sits above the try block, so no reordering can leave a
        credential minted for a subject that was then refused.
        """
        import ast
        import textwrap

        tree = ast.parse(textwrap.dedent(inspect.getsource(ToolExecutor.run_analysis)))
        body = tree.body[0].body
        guard_index = next(
            i
            for i, node in enumerate(body)
            if isinstance(node, ast.If) and "SERVICE_IDENTITY" in ast.unparse(node.test)
        )
        try_index = next(i for i, node in enumerate(body) if isinstance(node, ast.Try))
        assert guard_index < try_index


_INTERNAL_SECRET = "test-internal-secret"
# a DIFFERENT value, and the difference is the entire mechanism under test: mcp-server and
# results-api hold the first by design and neither holds the second
_GATEWAY_SECRET = "test-gateway-identity-secret"


def _marked_request(headers: dict[str, str]) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/chat/v1/chat",
            "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        }
    )


class TestSandboxDispatchRequiresTheGatewaySecret:
    """genetics-results-suite-4h6.84: the residual TestRunAnalysisRequiresARealUser leaves.

    That class refuses `mcp-tool` — `auth_required` case 3, the marker with no identity
    header. Case 1 beats case 3: marker PLUS an allow-listed identity header resolves to
    that address, and never to SERVICE_IDENTITY, so the guard above evaluates False. Every
    holder of INTERNAL_API_SECRET can produce that request, mcp-server included, and the
    NetworkPolicy admits it to chat-backend:8000. The `sub` of both per-execution JWTs, the
    artifact scope and the audit trail would then name someone who never asked.

    THE FIRST FIX FOR THAT WAS WRONG AND THESE TESTS ARE SHAPED BY HOW IT FAILED. It keyed
    on the marker's TRANSPORT — `X-Internal-Auth` (auth-gateway's) versus
    `Authorization: Bearer` (mcp-server's and results-api's) — which measured, end to end,
    as: bearer refused, and the same caller admitted the moment it copied its own secret
    into `X-Internal-Auth`. A header name is not a secret, and both of those services hold
    INTERNAL_API_SECRET by design.

    What separates the callers now is a second secret, GATEWAY_IDENTITY_SECRET, mounted only
    into auth-gateway and chat-backend. Each test below therefore states, independently and
    unconditionally, that a caller holding INTERNAL_API_SECRET is refused however it presents
    it — including both transports at once, which is the exact request that defeated the
    previous fix.
    """

    @staticmethod
    async def _dispatch(executor, sandbox, headers: dict[str, str], gateway_secret=_GATEWAY_SECRET):
        """The full production chain: headers -> dependency -> _execute_tool -> run_analysis."""
        from genetics_mcp_server.auth.dependencies import gateway_asserted_identity
        from genetics_mcp_server.llm_service import LLMService

        executor._sandbox = sandbox
        service = LLMService.__new__(LLMService)
        service.executor = executor
        service.subagent_service = None

        with settings_env(
            REQUIRE_AUTH="true",
            INTERNAL_API_SECRET=_INTERNAL_SECRET,
            GATEWAY_IDENTITY_SECRET=gateway_secret,
            ALLOWED_EMAIL_DOMAINS="finngen.fi",
            SANDBOX_ENABLED="true",
        ):
            request = _marked_request(
                {
                    **headers,
                    "X-Goog-Authenticated-User-Email": "anyone@finngen.fi",
                }
            )
            asserted = await gateway_asserted_identity(request)
            result = await service._execute_tool(
                "run_analysis",
                {"code": "print(1)"},
                None,
                "anyone@finngen.fi",
                "anything",
                asserted,
            )
        return asserted, result

    @staticmethod
    def _assert_refused(asserted, result, sandbox):
        assert asserted is False
        assert sandbox.calls == [], "nothing may be dispatched, and nothing minted"
        assert result["success"] is False
        assert result["error_type"] == "SandboxNotConfigured"
        assert result["retryable"] is False

    async def test_a_bearer_marker_asserting_an_identity_is_refused(self, executor):
        """The bead's exact request, in the transport mcp-server and results-api use."""
        sandbox = _StubSandbox(result=_result_body())
        asserted, result = await self._dispatch(
            executor, sandbox, {"Authorization": f"Bearer {_INTERNAL_SECRET}"}
        )
        self._assert_refused(asserted, result, sandbox)

    async def test_the_internal_marker_header_asserting_an_identity_is_refused(self, executor):
        """PROBE-B: the same caller, having simply renamed its own header.

        This is what the transport-based fix admitted. Nothing stops mcp-server from writing
        the secret it already holds into auth-gateway's header name.
        """
        sandbox = _StubSandbox(result=_result_body())
        asserted, result = await self._dispatch(
            executor, sandbox, {"X-Internal-Auth": _INTERNAL_SECRET}
        )
        self._assert_refused(asserted, result, sandbox)

    async def test_both_internal_transports_at_once_are_refused(self, executor):
        """PROBE-D, the request that broke the previous fix: a bearer caller that also adds
        `X-Internal-Auth`. Under the transport check this dispatched.
        """
        sandbox = _StubSandbox(result=_result_body())
        asserted, result = await self._dispatch(
            executor,
            sandbox,
            {
                "Authorization": f"Bearer {_INTERNAL_SECRET}",
                "X-Internal-Auth": _INTERNAL_SECRET,
            },
        )
        self._assert_refused(asserted, result, sandbox)

    async def test_the_internal_secret_in_the_gateway_header_is_refused(self, executor):
        """And renaming the header the other way does not help either: the value is compared,
        so a caller holding only INTERNAL_API_SECRET fails even in auth-gateway's own header.
        """
        sandbox = _StubSandbox(result=_result_body())
        asserted, result = await self._dispatch(
            executor, sandbox, {"X-Gateway-Auth": _INTERNAL_SECRET}
        )
        self._assert_refused(asserted, result, sandbox)

    async def test_the_gateway_secret_is_what_dispatches(self, executor):
        """The control. Identical request, carrying the secret only auth-gateway holds.

        If this fails the refusals above prove nothing, because something other than the
        gateway secret would be stopping all of them.
        """
        sandbox = _StubSandbox(result=_result_body())
        asserted, result = await self._dispatch(
            executor, sandbox, {"X-Gateway-Auth": _GATEWAY_SECRET}
        )

        assert asserted is True
        assert len(sandbox.calls) == 1
        assert sandbox.calls[0]["user"] == "anyone@finngen.fi"
        assert result["success"] is True

    async def test_an_unprovisioned_gateway_secret_refuses_rather_than_admits(self, executor):
        """Fail closed on misconfiguration: with GATEWAY_IDENTITY_SECRET unset under
        REQUIRE_AUTH, even a request carrying the header is refused. The alternative — an
        unset secret degrading to "everyone is the gateway" — is the failure this whole bead
        exists to prevent, and it would arrive silently on a deploy that forgot the key.
        """
        sandbox = _StubSandbox(result=_result_body())
        asserted, result = await self._dispatch(
            executor, sandbox, {"X-Gateway-Auth": _GATEWAY_SECRET}, gateway_secret=""
        )
        self._assert_refused(asserted, result, sandbox)

    async def test_the_dispatch_refuses_on_its_own_without_the_dependency(self, executor):
        """The guard is at the waist, not at the route: calling run_analysis directly with
        no stated provenance is refused too, so a future caller that forgets to plumb the
        flag loses code execution rather than inheriting trust.
        """
        sandbox = _StubSandbox(result=_result_body())
        with settings_env(REQUIRE_AUTH="true"):
            result = await _run(executor, sandbox, user="real@finngen.fi")

        assert sandbox.calls == []
        assert result["success"] is False
        assert result["error_type"] == "SandboxNotConfigured"

    async def test_the_chat_route_supplies_the_provenance(self):
        """Structural, and it earned its place: the first draft of this change declared the
        dependency on the route and never forwarded it to `stream_chat`, so every execution
        would have run on the default `False` and code execution would have been dead in
        production while the two tests above stayed green. Both halves are asserted —
        resolving it, and passing it on.
        """
        from genetics_mcp_server import chat_api
        from genetics_mcp_server.auth.dependencies import gateway_asserted_identity

        parameter = inspect.signature(chat_api.stream_chat).parameters["gateway_asserted"]
        assert parameter.default.dependency is gateway_asserted_identity
        source = inspect.getsource(chat_api.stream_chat)
        assert "gateway_asserted=gateway_asserted" in source

    async def test_the_gateway_secret_alone_does_not_assert_an_identity(self):
        """Case 3 is `mcp-tool`, which names nobody: the gateway secret by itself is not an
        assertion about a person, so it must not satisfy this dependency either.
        """
        from genetics_mcp_server.auth.dependencies import gateway_asserted_identity

        with settings_env(
            REQUIRE_AUTH="true",
            INTERNAL_API_SECRET=_INTERNAL_SECRET,
            GATEWAY_IDENTITY_SECRET=_GATEWAY_SECRET,
        ):
            asserted = await gateway_asserted_identity(
                _marked_request({"X-Gateway-Auth": _GATEWAY_SECRET})
            )
        assert asserted is False


class TestRunAnalysisIdentity:
    async def test_the_caller_identity_is_what_reaches_the_client(self, executor):
        sandbox = _StubSandbox(result=_result_body())
        await _run(executor, sandbox, user="real@finngen.fi", session_id="conv-7")
        assert sandbox.calls[0]["user"] == "real@finngen.fi"
        assert sandbox.calls[0]["session_id"] == "conv-7"

    async def test_llm_service_strips_a_model_supplied_identity(self, monkeypatch):
        """tool_input is splatted verbatim into the handler, and the model can emit keys
        the schema does not declare. Same shape as the literature `backend` strip.
        """
        from genetics_mcp_server.config import settings as settings_module
        from genetics_mcp_server.llm_service import LLMService

        seen = {}

        class _Recorder:
            async def run_analysis(self, **kwargs):
                seen.update(kwargs)
                return {"success": True}

        service = LLMService.__new__(LLMService)
        service.executor = _Recorder()
        service.subagent_service = None

        # the dispatch allowlist refuses a withheld tool before any of this, so the strip
        # is only observable on a deployment that has the sandbox
        monkeypatch.setenv("SANDBOX_ENABLED", "true")
        settings_module.get_settings.cache_clear()
        try:
            await service._execute_tool(
                "run_analysis",
                {"code": "print(1)", "user": "attacker@evil.example", "session_id": "other"},
                None,
                "real@finngen.fi",
                "conv-7",
            )
        finally:
            settings_module.get_settings.cache_clear()
        assert seen["user"] == "real@finngen.fi"
        assert seen["session_id"] == "conv-7"


# ------------------------------------------------- the script_result SSE payload (4h6.71)


def test_script_result_payload_marks_a_successful_run_as_not_a_failure(executor):
    payload = _script_result_payload(
        3, executor._render_analysis({"status": "ok", "duration_ms": 812}), tool_use_id="t7"
    )
    assert payload == {
        "iteration": 3,
        "tool_use_id": "t7",
        "ran": True,
        "ok": True,
        "status": "ok",
        "timed_out": False,
        "exception": None,
        "limit": None,
        "duration_ms": 812,
    }


def test_script_result_payload_reports_a_script_exception(executor):
    rendered = executor._render_analysis(
        {
            "status": "error",
            "error": {"type": "ValueError", "message": "bad", "traceback": "..."},
            "duration_ms": 40,
        }
    )
    payload = _script_result_payload(1, rendered)
    assert payload["ran"] is True
    assert payload["ok"] is False
    assert payload["exception"] == "ValueError"
    assert payload["timed_out"] is False


def test_script_result_payload_distinguishes_timeout_from_a_limit(executor):
    timed_out = _script_result_payload(1, executor._render_analysis({"status": "timeout"}))
    assert timed_out["timed_out"] is True and timed_out["ok"] is False

    limited = executor._render_analysis(
        {"status": "limit", "error": {"type": "OutputLimit", "limit": "OutputLimit"}}
    )
    payload = _script_result_payload(1, limited)
    assert payload["timed_out"] is False
    assert payload["limit"] == "OutputLimit"
    assert payload["status"] == "limit"


async def test_a_blank_script_is_reported_as_its_own_shape_not_as_unknown(executor):
    """`ran: False` is not a verdict on whose fault it was, so the shape has to be readable.

    Without an error_type this arrives on the wire as `status: "unknown"` — the same string
    a genuinely unrecognised transport fault produces — and a benchmark reading it books the
    model emitting no code as the sandbox being flaky.
    """
    for blank in ("", "   \n\t", None, 42):
        result = await executor.run_analysis(code=blank, user="u@finngen.fi", session_id="c1")
        assert result["success"] is False
        assert result["error_type"] == "EmptyScript", blank
        assert _script_result_payload(1, result)["status"] == "EmptyScript"


def test_script_result_payload_marks_a_sandbox_fault_as_not_run(executor):
    """A restarting sandbox is not a failed script and must not enter the failure rate."""
    for shape in (
        {
            "success": False,
            "error": "unavailable",
            "error_type": "SandboxUnavailable",
            "retryable": True,
        },
        executor._sandbox_operator_error("not configured"),
    ):
        payload = _script_result_payload(1, shape)
        assert payload["ran"] is False, shape
        assert payload["ok"] is False
        assert payload["timed_out"] is False

    # ...whereas anything the supervisor itself answered carries a status, so it did run
    assert _script_result_payload(1, executor._render_analysis({"status": "error"}))["ran"] is True


class TestReadArtifactSessionInjection:
    """The session is the authorization, so it is injected exactly the way run_analysis's is."""

    @staticmethod
    async def _dispatch(executor, tool_input, session_id="conv-9"):
        from genetics_mcp_server.llm_service import LLMService

        service = LLMService.__new__(LLMService)
        service.executor = executor
        service.subagent_service = None
        return await service._execute_tool(
            "read_artifact", tool_input, None, "u@finngen.fi", session_id, False
        )

    async def test_the_authenticated_session_is_injected(self, executor):
        _record("conv-9", EXEC_A, "hits.tsv")
        executor._sandbox = _ArtifactSandbox({"hits.tsv": _served("hits.tsv", b"rows")})
        result = await self._dispatch(executor, {"name": "hits.tsv"})
        assert result["success"] is True

    async def test_a_model_supplied_session_cannot_reach_another_conversation(self, executor):
        """tool_input is splatted verbatim, so an unstripped key would be a cross-session read
        of somebody else's artifacts — the same forgery the run_analysis strip exists for.
        """
        _record("other-conv", EXEC_A, "secrets.tsv")
        executor._sandbox = _ArtifactSandbox({"secrets.tsv": _served("secrets.tsv", b"data")})
        result = await self._dispatch(
            executor, {"name": "secrets.tsv", "session_id": "other-conv", "user": "x@y.z"}
        )
        assert result["success"] is False
        assert result["error_type"] == "ArtifactNotFound"
        assert executor._sandbox.asked == []


class TestArtifactsRetainedInClear:
    """genetics-results-suite-4h6.97: the supervisor's exposure signal must reach the model.

    Rendered only when true, matching artifacts_omitted — a field that is false on every run
    is noise the model learns to skip past, and this one is adversarial-only in practice.
    """

    async def test_absent_on_an_ordinary_run(self, executor):
        result = await _run(executor, _StubSandbox(result=_result_body()))
        assert "artifacts_retained_in_clear" not in result
        assert "artifacts_retained_in_clear_note" not in result

    async def test_absent_when_the_supervisor_says_false(self, executor):
        body = _result_body(artifacts_retained_in_clear=False)
        result = await _run(executor, _StubSandbox(result=body))
        assert "artifacts_retained_in_clear" not in result

    async def test_rendered_with_actionable_wording_when_true(self, executor):
        body = _result_body(artifacts_retained_in_clear=True)
        result = await _run(executor, _StubSandbox(result=body))
        assert result["artifacts_retained_in_clear"] is True
        note = result["artifacts_retained_in_clear_note"]
        # the honest framing: the exposure is to other code in the sandbox, not to this user,
        # so the model is told to report a cleanup failure rather than to alarm the user about
        # their own data
        assert "other code" in note
        assert "mention to the user" in note
        assert result["success"] is True

    async def test_it_is_not_folded_into_artifacts_omitted(self, executor):
        """Omission is recoverable and says nothing about exposure; conflating them is the
        whole reason the supervisor keeps them separate.
        """
        body = _result_body(artifacts_retained_in_clear=True, artifacts_omitted=0)
        result = await _run(executor, _StubSandbox(result=body))
        assert "artifacts_omitted" not in result
        assert result["artifacts_retained_in_clear"] is True

    async def test_a_non_boolean_is_not_treated_as_the_signal(self, executor):
        body = _result_body(artifacts_retained_in_clear="yes")
        result = await _run(executor, _StubSandbox(result=body))
        assert "artifacts_retained_in_clear" not in result

    async def test_a_large_output_cannot_truncate_the_signal_away(self, executor):
        """genetics-results-suite-4h6.97: the ANALYSED SCRIPT must not be able to delete it.

        llm_service cuts a serialised tool result over `mcp_max_result_size` (50,000) to a
        PREFIX, and `output` is script-controlled up to the supervisor's 64 KiB cap. Measured
        before the fix, with the flag rendered seventh and after `output`: a 66,569-byte result
        truncated at 50,000 contained NEITHER the flag NOR the note — so a script that both
        provoked the condition (chmod artifacts/) and printed ~50 KB suppressed its own
        warning, behind a generic "[TRUNCATED: ...]" that says nothing about a dropped field.

        Both defences are asserted here because they are independent: the field is now ordered
        AHEAD of `output` (survives the prefix), and `_truncation_notice` re-states the note
        from the whole pre-truncation dict (survives any ordering).
        """
        body = _result_body(
            artifacts_retained_in_clear=True,
            output="x" * 64 * 1024,
            output_truncated=True,
        )
        result = await _run(executor, _StubSandbox(result=body))
        result_json = json.dumps(result)
        limit = Settings().mcp_max_result_size
        assert len(result_json) > limit

        # exactly llm_service._execute_tool's truncation
        truncated = result_json[: limit - 1000] + _truncation_notice(result)

        assert "artifacts_retained_in_clear" in truncated
        assert "other code running in the sandbox" in truncated
        # the notice half on its own, so removing the re-attachment fails this even while the
        # ordering half still carries the note inside the prefix
        assert "other code running in the sandbox" in _truncation_notice(result)
        # and the ordering half specifically: the flag is inside the surviving PREFIX, not only
        # in the appended notice
        prefix = result_json[: limit - 1000]
        assert "artifacts_retained_in_clear" in prefix
        assert prefix.index("artifacts_retained_in_clear") < prefix.index('"output"')
