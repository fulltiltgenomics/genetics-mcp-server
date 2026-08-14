"""Tests for the code-execution tool halves: list_capabilities and read_artifact.

genetics-results-suite-4h6.15. The sandbox is not deployed, so these exercise the tool
layer only: the SDK catalogue rendering, and the artifact read together with the path
validation it borrows from skills/sandbox_tools.py.
"""

import base64
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
