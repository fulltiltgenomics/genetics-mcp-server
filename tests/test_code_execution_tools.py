"""Tests for the code-execution tools: list_capabilities, read_artifact, run_analysis.

genetics-results-suite-4h6.15 and -4h6.48. The sandbox is not deployed, so these exercise
the tool layer only: the SDK catalogue rendering, the artifact read together with the path
validation it borrows from skills/sandbox_tools.py, and run_analysis against a stubbed
transport — no live sandbox, no credentials.
"""

import asyncio
import base64
import inspect
import os
import signal

import pytest

from genetics_mcp_server.tools import ToolExecutor
from genetics_mcp_server.tools import executor as executor_module
from genetics_mcp_server.tools.definitions import TOOL_DEFINITIONS, get_anthropic_tools

PNG_HEADER = b"\x89PNG\r\n\x1a\n" + bytes(range(256))


@pytest.fixture
def executor():
    return ToolExecutor()


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    """A tmp stand-in for the hardcoded /scratch prefix an artifacts dir must sit under."""
    root = os.path.realpath(tmp_path / "scratch")
    os.makedirs(root, exist_ok=True)
    monkeypatch.setattr(executor_module, "_ARTIFACTS_DIR_PREFIX", root + "/")
    return root


@pytest.fixture
def artifacts(tmp_path, scratch, monkeypatch):
    """A configured artifacts directory, plus a sibling nobody may reach from it."""
    artifacts_dir = tmp_path / "scratch" / "exec-1" / "artifacts"
    artifacts_dir.mkdir(parents=True)
    (tmp_path / "scratch" / "exec-1" / "secret.txt").write_text("working files")
    (tmp_path / "chat_history.db").write_text("crown jewels")
    monkeypatch.setenv("SANDBOX_ARTIFACTS_DIR", str(artifacts_dir))
    return artifacts_dir


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

    async def test_index_summaries_do_not_come_from_module_docstrings(self, executor):
        from genetics_mcp_server import sdk

        result = await executor.list_capabilities()
        summaries = {m["module"]: m["summary"] for m in result["modules"]}
        assert summaries["genetics"] not in (sdk.__doc__ or "")


class TestReadArtifact:
    async def test_refuses_when_no_artifacts_directory_is_configured(
        self, executor, monkeypatch
    ):
        """chat-backend's own environment: nothing to read, and no local read attempted."""
        monkeypatch.delenv("SANDBOX_ARTIFACTS_DIR", raising=False)
        result = await executor.read_artifact(name="manhattan.png")
        assert result["success"] is False
        assert "not enabled" in result["error"]

    async def test_reads_a_text_artifact(self, executor, artifacts):
        (artifacts / "hits.tsv").write_text("gene\tpip\nIL7R\t0.9\n")
        result = await executor.read_artifact(name="hits.tsv")
        assert result["success"] is True
        assert result["content"] == "gene\tpip\nIL7R\t0.9\n"
        assert result["encoding"] == "utf-8"
        assert result["truncated"] is False
        assert result["content_type"] == "text/tab-separated-values"

    async def test_reads_a_binary_artifact_as_base64(self, executor, artifacts):
        (artifacts / "manhattan.png").write_bytes(PNG_HEADER)
        result = await executor.read_artifact(name="manhattan.png")
        assert result["success"] is True
        assert result["encoding"] == "base64"
        assert result["content_type"] == "image/png"
        assert base64.b64decode(result["content"]) == PNG_HEADER

    async def test_truncates_a_long_text_artifact_and_says_so(self, executor, artifacts):
        (artifacts / "big.txt").write_text("x" * (ToolExecutor._MAX_ARTIFACT_TEXT_CHARS + 10))
        result = await executor.read_artifact(name="big.txt")
        assert result["truncated"] is True
        assert len(result["content"]) == ToolExecutor._MAX_ARTIFACT_TEXT_CHARS

    async def test_refuses_an_oversized_artifact_rather_than_cutting_it(
        self, executor, artifacts
    ):
        (artifacts / "huge.png").write_bytes(b"\x00" * (ToolExecutor._MAX_ARTIFACT_BYTES + 1))
        result = await executor.read_artifact(name="huge.png")
        assert result["success"] is False
        assert "read limit" in result["error"]

    async def test_the_oversize_refusal_is_not_a_size_oracle(self, executor, artifacts):
        """The refusal states the limit, never the file's own size."""
        size = ToolExecutor._MAX_ARTIFACT_BYTES + 12345
        (artifacts / "huge2.png").write_bytes(b"\x00" * size)
        result = await executor.read_artifact(name="huge2.png")
        assert result["success"] is False
        assert str(size) not in result["error"]

    async def test_missing_artifact(self, executor, artifacts):
        result = await executor.read_artifact(name="nope.tsv")
        assert result["success"] is False
        assert result["error"] == "Artifact not found: nope.tsv"

    async def test_directory_is_not_an_artifact(self, executor, artifacts):
        (artifacts / "plots").mkdir()
        result = await executor.read_artifact(name="plots")
        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.parametrize(
        "name",
        [
            "../secret.txt",
            "../../chat_history.db",
            "..",
            ".",
            "sub/hits.tsv",
            "/etc/passwd",
            "..\\secret.txt",
            "hits.tsv\x00.png",
            "",
            "   ",
        ],
    )
    async def test_rejects_anything_that_is_not_a_bare_name(self, executor, artifacts, name):
        result = await executor.read_artifact(name=name)
        assert result["success"] is False
        assert result["error"].startswith(("Invalid artifact name", "An artifact name"))

    async def test_traversal_cannot_reach_the_execution_scratch_dir(
        self, executor, artifacts
    ):
        """The sibling file exists and is readable by the process; the tool still refuses."""
        assert (artifacts.parent / "secret.txt").is_file()
        result = await executor.read_artifact(name="../secret.txt")
        assert result["success"] is False
        assert "working files" not in str(result)

    async def test_symlink_out_of_the_artifacts_dir_is_refused(self, executor, artifacts):
        """The name is bare and the file exists; only _validate_path's resolve catches it.

        A script can write whatever it likes into its own artifacts directory, symlinks
        included, so the name check alone is not enough.
        """
        (artifacts / "leak.txt").symlink_to(artifacts.parent.parent.parent / "chat_history.db")
        result = await executor.read_artifact(name="leak.txt")
        assert result["success"] is False
        assert result["error"] == "Artifact not found: leak.txt"
        assert "crown jewels" not in str(result)

    async def test_does_not_read_subagent_allowed_paths(
        self, executor, tmp_path, monkeypatch
    ):
        """SUBAGENT_ALLOWED_PATHS is /data in the deployment; it must gain no new reader."""
        monkeypatch.delenv("SANDBOX_ARTIFACTS_DIR", raising=False)
        monkeypatch.setenv("SUBAGENT_ALLOWED_PATHS", str(tmp_path))
        (tmp_path / "chat_history.db").write_text("crown jewels")
        result = await executor.read_artifact(name="chat_history.db")
        assert result["success"] is False
        assert "crown jewels" not in str(result)

    async def test_hardlink_to_an_outside_file_is_refused(self, executor, artifacts, tmp_path):
        """A hardlink has nothing to resolve, so both path layers see an in-tree path.

        The name is bare and the resolved path is inside the allow-list; only the link
        count taken off the open fd tells the inode apart from a genuine artifact.
        """
        os.link(tmp_path / "chat_history.db", artifacts / "hard.db")
        result = await executor.read_artifact(name="hard.db")
        assert result["success"] is False
        assert result["error"] == "Artifact not found: hard.db"
        assert "crown jewels" not in str(result)

    async def test_a_hardlinked_artifact_is_refused_even_inside_the_dir(
        self, executor, artifacts
    ):
        """st_nlink != 1 is stated here rather than inherited from fs.protected_hardlinks."""
        (artifacts / "hits.tsv").write_text("gene\tpip\n")
        os.link(artifacts / "hits.tsv", artifacts / "alias.tsv")
        result = await executor.read_artifact(name="alias.tsv")
        assert result["success"] is False

    async def test_the_read_opens_once_with_o_nofollow(self, executor, artifacts, monkeypatch):
        """Every decision is taken off one fd; nothing is re-opened by path."""
        (artifacts / "hits.tsv").write_text("gene\tpip\n")
        calls = []
        real_open = os.open

        def recording_open(path, flags, *args, **kwargs):
            calls.append((str(path), flags))
            return real_open(path, flags, *args, **kwargs)

        monkeypatch.setattr(os, "open", recording_open)
        result = await executor.read_artifact(name="hits.tsv")
        assert result["success"] is True
        artifact_opens = [c for c in calls if c[0].endswith("hits.tsv")]
        assert len(artifact_opens) == 1
        assert artifact_opens[0][1] & os.O_NOFOLLOW

    async def test_a_swap_after_the_open_cannot_leak(
        self, executor, artifacts, tmp_path, monkeypatch
    ):
        """TOCTOU: the script owns the directory and can swap the name mid-read.

        The swap is performed inside os.open, i.e. after the path-based layers have
        resolved the name and before any byte is read. Whatever the outcome, the bytes
        can only come from the fd that was opened, never from the symlink's target.
        """
        (artifacts / "hits.tsv").write_text("gene\tpip\n")
        real_open = os.open
        swapped = []

        def swapping_open(path, flags, *args, **kwargs):
            fd = real_open(path, flags, *args, **kwargs)
            if str(path).endswith("hits.tsv") and not swapped:
                swapped.append(True)
                decoy = artifacts / "decoy"
                decoy.symlink_to(tmp_path / "chat_history.db")
                os.replace(decoy, artifacts / "hits.tsv")
            return fd

        monkeypatch.setattr(os, "open", swapping_open)
        result = await executor.read_artifact(name="hits.tsv")
        assert swapped
        assert "crown jewels" not in str(result)

    async def test_the_file_is_opened_relative_to_the_directory_fd(
        self, executor, artifacts, monkeypatch
    ):
        """Nothing is addressed by path after the directory check: the name is opened
        against the verified dirfd, so an intermediate component cannot be swapped."""
        (artifacts / "hits.tsv").write_text("gene\tpip\n")
        calls = []
        real_open = os.open

        def recording_open(path, flags, *args, **kwargs):
            calls.append((str(path), flags, kwargs.get("dir_fd")))
            return real_open(path, flags, *args, **kwargs)

        monkeypatch.setattr(os, "open", recording_open)
        result = await executor.read_artifact(name="hits.tsv")
        assert result["success"] is True
        dir_opens = [c for c in calls if c[1] & os.O_DIRECTORY]
        assert len(dir_opens) == 1
        assert dir_opens[0][1] & os.O_NOFOLLOW
        # the bare name, resolved against a dir_fd — not an absolute path
        artifact_opens = [c for c in calls if c[0] == "hits.tsv"]
        assert len(artifact_opens) == 1
        assert isinstance(artifact_opens[0][2], int)
        assert artifact_opens[0][1] & os.O_NOFOLLOW
        assert artifact_opens[0][1] & os.O_NONBLOCK

    async def test_a_directory_swap_after_the_check_cannot_leak(
        self, executor, artifacts, tmp_path, monkeypatch
    ):
        """TOCTOU one level up: the swapped component is the artifacts DIRECTORY.

        O_NOFOLLOW guards only the final component and `_validate_path` resolves both
        sides through the same swapped link, so both agree. Only holding the directory
        open and resolving the name from that fd refuses this.
        """
        real_open = os.open
        swapped = []

        def swapping_open(path, flags, *args, **kwargs):
            fd = real_open(path, flags, *args, **kwargs)
            if flags & os.O_DIRECTORY and not swapped:
                swapped.append(True)
                os.rename(artifacts, artifacts.parent / "artifacts.orig")
                os.symlink(tmp_path, artifacts)
            return fd

        monkeypatch.setattr(os, "open", swapping_open)
        result = await executor.read_artifact(name="chat_history.db")
        assert swapped
        assert result["success"] is False
        assert "crown jewels" not in str(result)

    async def test_a_fifo_is_refused_without_blocking(self, executor, artifacts):
        """O_RDONLY on a writerless FIFO blocks in the kernel before S_ISREG is reached.

        A script can mkfifo any name in its own artifacts directory, so without O_NONBLOCK
        one read_artifact call hangs the calling coroutine — and the chat backend with it —
        forever. SIGALRM rather than asyncio.wait_for on purpose: the open blocks inside
        the coroutine's own thread, so only a signal can break a regression out of it.
        """
        os.mkfifo(artifacts / "results.tsv")

        def _blew_the_deadline(signum, frame):
            raise TimeoutError("read_artifact blocked on a FIFO")

        previous = signal.signal(signal.SIGALRM, _blew_the_deadline)
        signal.setitimer(signal.ITIMER_REAL, 5.0)
        try:
            result = await executor.read_artifact(name="results.tsv")
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)

        assert result["success"] is False
        assert result["error"] == "Artifact not found: results.tsv"

    async def test_the_reported_size_is_the_payload_not_the_stat(
        self, executor, artifacts, monkeypatch
    ):
        """A file that grows between the fstat and the read must not report st_size."""
        target = artifacts / "hits.tsv"
        target.write_text("gene\tpip\n")
        real_fstat = os.fstat

        def growing_fstat(fd, *args, **kwargs):
            st = real_fstat(fd, *args, **kwargs)
            return os.stat_result(
                (st.st_mode, st.st_ino, st.st_dev, st.st_nlink, st.st_uid, st.st_gid,
                 st.st_size + 1_000_000, st.st_atime, st.st_mtime, st.st_ctime)
            )

        monkeypatch.setattr(os, "fstat", growing_fstat)
        result = await executor.read_artifact(name="hits.tsv")
        assert result["success"] is True
        assert result["size"] == len(result["content"])


class TestArtifactsDirectoryChecks:
    """`_artifacts_dir` fails closed before a name is looked at."""

    def test_the_required_prefix_is_the_scratch_mount(self):
        """Pinned so it cannot be widened into a path chat-backend actually has."""
        assert executor_module._ARTIFACTS_DIR_PREFIX == "/scratch/"

    async def test_a_symlinked_artifacts_dir_is_refused(
        self, executor, scratch, tmp_path, monkeypatch
    ):
        """_validate_path resolves both sides, so a symlinked root validates everything.

        The child uid owns /scratch/<id>, so it can rmdir its artifacts directory and
        relink it at another execution's retained artifacts.
        """
        other = tmp_path / "scratch" / "exec-2" / "artifacts"
        other.mkdir(parents=True)
        (other / "chat_history.db").write_text("crown jewels")
        linked = tmp_path / "scratch" / "exec-1-artifacts"
        linked.symlink_to(other)
        monkeypatch.setenv("SANDBOX_ARTIFACTS_DIR", str(linked))

        result = await executor.read_artifact(name="chat_history.db")
        assert result["success"] is False
        assert "not enabled" in result["error"]
        assert "crown jewels" not in str(result)

    async def test_an_artifacts_dir_outside_the_prefix_is_refused(
        self, executor, scratch, tmp_path, monkeypatch
    ):
        """chat-backend has no /scratch volume, so SANDBOX_ARTIFACTS_DIR=/data is inert."""
        data = tmp_path / "data"
        data.mkdir()
        (data / "chat_history.db").write_text("crown jewels")
        monkeypatch.setenv("SANDBOX_ARTIFACTS_DIR", str(data))

        result = await executor.read_artifact(name="chat_history.db")
        assert result["success"] is False
        assert "not enabled" in result["error"]
        assert "crown jewels" not in str(result)

    async def test_a_prefix_lookalike_directory_is_refused(
        self, executor, scratch, tmp_path, monkeypatch
    ):
        """/scratchpad must not pass a check meant for /scratch/."""
        lookalike = tmp_path / "scratchpad"
        lookalike.mkdir()
        (lookalike / "chat_history.db").write_text("crown jewels")
        monkeypatch.setenv("SANDBOX_ARTIFACTS_DIR", str(lookalike))

        result = await executor.read_artifact(name="chat_history.db")
        assert result["success"] is False
        assert "not enabled" in result["error"]


# --------------------------------------------------------------------------- run_analysis
# genetics-results-suite-4h6.48. The tool layer over sandbox_client (4h6.47). Everything
# below runs with NO sandbox and NO credentials: the transport is either a stub `execute`
# or httpx.MockTransport, and the one test that needs a signing key is the one asserting
# what happens when there isn't one.


class _StubSandbox:
    """Stands in for SandboxClient. Records what it was asked and replays one outcome."""

    def __init__(self, result=None, raises=None):
        self.result = result
        self.raises = raises
        self.calls = []

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

    def test_does_not_promise_artifact_contents(self):
        """The HTTP retrieval path is 4h6.52 and does not exist; read_artifact in this
        process reads a local directory that is not the sandbox's /scratch. Telling the
        model it can fetch a plot costs a roundtrip to find out it cannot.
        """
        by_name = {t["name"]: t for t in TOOL_DEFINITIONS}
        description = by_name["run_analysis"]["description"]
        assert "CANNOT BE RETRIEVED" in description
        assert "read_artifact" not in description

    def test_reaches_the_chat_tool_list(self):
        names = {t["name"] for t in get_anthropic_tools()}
        assert "run_analysis" in names


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

    async def test_the_manifest_says_the_contents_cannot_be_fetched(self, executor):
        body = _result_body(
            artifacts=[{"name": "plot.png", "size": 10, "content_type": "image/png"}]
        )
        result = await _run(executor, _StubSandbox(result=body))
        assert "cannot be retrieved" in result["artifacts_note"]

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


class TestRunAnalysisIdentity:
    async def test_the_caller_identity_is_what_reaches_the_client(self, executor):
        sandbox = _StubSandbox(result=_result_body())
        await _run(executor, sandbox, user="real@finngen.fi", session_id="conv-7")
        assert sandbox.calls[0]["user"] == "real@finngen.fi"
        assert sandbox.calls[0]["session_id"] == "conv-7"

    async def test_llm_service_strips_a_model_supplied_identity(self):
        """tool_input is splatted verbatim into the handler, and the model can emit keys
        the schema does not declare. Same shape as the literature `backend` strip.
        """
        from genetics_mcp_server.llm_service import LLMService

        seen = {}

        class _Recorder:
            async def run_analysis(self, **kwargs):
                seen.update(kwargs)
                return {"success": True}

        service = LLMService.__new__(LLMService)
        service.executor = _Recorder()
        service.subagent_service = None

        await service._execute_tool(
            "run_analysis",
            {"code": "print(1)", "user": "attacker@evil.example", "session_id": "other"},
            None,
            "real@finngen.fi",
            "conv-7",
        )
        assert seen["user"] == "real@finngen.fi"
        assert seen["session_id"] == "conv-7"
