"""The SDK audit trail (genetics-results-suite-4h6.12).

Once data access moves inside a sandboxed script, the `Executing tool:` line that
chat-backend logs per data access stops describing it: the whole script is one
`run_analysis` tool call. These tests pin the replacement — one line per SDK function call,
in the same shape as the tool line, carrying who, which function, a summary of the arguments
and how many rows came back.

The disclosure rule is under test as much as the format is. An SDK argument is
script-authored and unbounded, so anything that is not identifier-shaped must be reduced to
a type and a length: a raw value would let an injected script write chosen text, including
forged newline-separated log lines, into the operator's log pipeline.
"""

import logging
import re

import polars as pl
import pytest

from genetics_mcp_server.sdk import client as sdk_client
from genetics_mcp_server.sdk.client import GeneticsClient
from genetics_mcp_server.sdk.errors import GeneticsUsageError

AUDIT_LOGGER = "genetics_mcp_server.sdk.audit"

# the parser in scripts/analyze_conversations.py must agree with this; that agreement is
# asserted in test_analyze_conversations_sdk_stats.py rather than restated here
LINE_RE = re.compile(
    r"^\[user=(?P<user>[^\]]*)\] \[session=(?P<session>[^\]]*)\] \[execution=(?P<execution>[^\]]*)\] "
    r"Executing SDK function: (?P<function>\S+) with input: (?P<arguments>.*) "
    r"rows: (?P<rows>\d+)(?: error: (?P<error>\S+))?$"
)


class _StubExecutor:
    """Stands in for ToolExecutor: any method the client reaches for returns the payload.

    __getattr__ rather than named methods because the SDK dispatches one function across
    several executor methods depending on which argument was supplied, and which one is
    reached is not what these tests are about.
    """

    def __init__(self, payload=None):
        self.payload = payload or {"success": True, "results": [], "columns": ["gene", "pip"]}
        self.calls = []

    async def close(self):
        pass

    async def get_available_resources(self):
        return {"success": True, "resources": {"a": 1, "b": 2, "c": 3}}

    def __getattr__(self, name):
        async def call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return self.payload

        return call


def _client(payload=None) -> GeneticsClient:
    return GeneticsClient(executor=_StubExecutor(payload))


@pytest.fixture
def audit_lines(caplog):
    caplog.set_level(logging.INFO, logger=AUDIT_LOGGER)
    return caplog


def _lines(caplog) -> list[str]:
    return [r.message for r in caplog.records if r.name == AUDIT_LOGGER]


class TestOneLinePerCall:
    @pytest.mark.asyncio
    async def test_call_emits_exactly_one_line_with_every_field(self, audit_lines):
        client = _client({
            "success": True,
            "results": [["IL7R", 0.9], ["IL7R", 0.5]],
            "columns": ["gene", "pip"],
        })
        await client.credible_sets(gene="IL7R")

        lines = _lines(audit_lines)
        assert len(lines) == 1, f"expected one audit line, got {lines}"
        match = LINE_RE.match(lines[0])
        assert match, f"line does not match the audit format: {lines[0]!r}"
        assert match.group("function") == "credible_sets"
        assert match.group("arguments") == "{'gene': 'IL7R'}"
        assert match.group("rows") == "2"
        assert match.group("error") is None

    @pytest.mark.asyncio
    async def test_line_mirrors_the_tool_line_prefix(self, audit_lines):
        """llm_service logs `[user=…] [session=…] Executing tool: …`; an operator's query
        for either must key on the same prefix shape."""
        await _client().credible_sets(gene="IL7R")
        line = _lines(audit_lines)[0]
        assert line.startswith("[user=unknown] [session=unknown] [execution=unknown] ")
        assert " with input: " in line

    @pytest.mark.asyncio
    async def test_does_not_collide_with_the_tool_marker(self, audit_lines):
        """An existing query for `Executing tool:` must not start matching SDK calls."""
        await _client().credible_sets(gene="IL7R")
        assert "Executing tool:" not in _lines(audit_lines)[0]

    @pytest.mark.asyncio
    async def test_sync_surface_does_not_double_count(self, audit_lines, monkeypatch):
        """`genetics.credible_sets()` delegates to the client method; only one line."""
        import genetics_mcp_server.sdk as genetics

        stub = _client()
        monkeypatch.setattr(genetics, "_client", stub)
        genetics.credible_sets(gene="IL7R")
        assert len(_lines(audit_lines)) == 1

    @pytest.mark.asyncio
    async def test_identity_comes_from_the_environment(self, audit_lines, monkeypatch):
        monkeypatch.setenv("SANDBOX_USER", "someone@example.org")
        monkeypatch.setenv("SANDBOX_SESSION_ID", "sess-1")
        monkeypatch.setenv("SANDBOX_EXECUTION_ID", "exec-1")
        await _client().credible_sets(gene="IL7R")
        match = LINE_RE.match(_lines(audit_lines)[0])
        assert match.group("user") == "someone@example.org"
        assert match.group("session") == "sess-1"
        assert match.group("execution") == "exec-1"


class TestRowCount:
    @pytest.mark.asyncio
    async def test_empty_result_reports_zero_rows(self, audit_lines):
        await _client({"success": True, "results": [], "columns": ["gene", "pip"]}).credible_sets(
            gene="IL7R"
        )
        assert LINE_RE.match(_lines(audit_lines)[0]).group("rows") == "0"

    @pytest.mark.asyncio
    async def test_non_empty_result_reports_the_frame_height(self, audit_lines):
        rows = [["IL7R", i / 10] for i in range(7)]
        client = _client({"success": True, "results": rows, "columns": ["gene", "pip"]})
        frame = await client.credible_sets(gene="IL7R")
        assert frame.height == 7
        assert LINE_RE.match(_lines(audit_lines)[0]).group("rows") == "7"

    @pytest.mark.asyncio
    async def test_dict_returning_function_reports_top_level_entries(self, audit_lines):
        await _client().resources()
        line = _lines(audit_lines)[0]
        assert LINE_RE.match(line).group("rows") == "3"

    @pytest.mark.asyncio
    async def test_upstream_failure_reports_rows_zero_and_only_the_exception_type(
        self, audit_lines
    ):
        client = _client({"success": False, "error": "db-api said no"})
        with pytest.raises(Exception):
            await client.credible_sets(gene="IL7R")

        lines = _lines(audit_lines)
        assert len(lines) == 1
        match = LINE_RE.match(lines[0])
        assert match.group("rows") == "0"
        assert match.group("error") == "GeneticsError"
        # upstream error text is not ours to copy into the operator's log
        assert "db-api said no" not in lines[0]


class TestCallsThatNeverReachedTheExecutor:
    """A local argument-validation failure read nothing.

    Recording it in the READ shape both inflated the flood (the refusal path is the cheap
    one — no network, no upstream) and made "what did that script read?" unanswerable,
    because the answer was full of calls that read nothing.
    """

    @pytest.mark.asyncio
    async def test_a_refused_call_is_not_recorded_as_a_read(self, audit_lines):
        client = _client()
        with pytest.raises(GeneticsUsageError):
            await client.credible_sets()  # no selector: exactly-one-of fails

        lines = _lines(audit_lines)
        assert len(lines) == 1
        assert "Executing SDK function:" not in lines[0]
        assert lines[0].startswith("[user=unknown] [session=unknown] [execution=unknown] ")
        assert "Rejected SDK function: credible_sets with input: {}" in lines[0]
        assert " rows: " not in lines[0], "a refusal must carry no row count at all"
        assert lines[0].endswith(" error: GeneticsUsageError")
        # the message quotes the argument names back; only the type may be logged
        assert "provide exactly one of" not in lines[0]
        assert not LINE_RE.match(lines[0]), "a refusal must not parse as a data access"

    @pytest.mark.asyncio
    async def test_a_refusal_inside_a_reached_branch_is_still_a_refusal(self, audit_lines):
        """`_reject` fires after the branch is chosen but still before any executor call."""
        client = _client()
        with pytest.raises(GeneticsUsageError):
            await client.credible_sets(variant="1:1:A:G", window=1000)
        assert "Rejected SDK function: credible_sets" in _lines(audit_lines)[0]


@pytest.fixture
def fresh_budget(monkeypatch):
    """The budgets are process globals on purpose (nothing outside the module may reset them),
    so a test that exercises one has to start it from a known value."""
    monkeypatch.setattr(sdk_client, "_audit_refusals", 0)
    monkeypatch.setattr(sdk_client, "_audit_meta_records", 0)


def _meta(caplog, contains="") -> list[str]:
    return [
        r.message
        for r in caplog.records
        if r.name == f"{AUDIT_LOGGER}.meta" and contains in r.message
    ]


class TestFloodBound:
    """Only the CHEAP side is bounded.

    A refusal costs the script nothing — no socket, no upstream — so it is the flooding
    primitive and is capped. A call that reached the executor paid an HTTP round-trip and is
    charged against the sandbox's byte and row quotas, so it cannot be driven at flood rates;
    capping it would not bound a flood, it would sell a script silence for 1000 cheap refusals.
    """

    def test_the_refusal_ceiling_is_a_deliberate_bound(self):
        assert 100 <= sdk_client._AUDIT_MAX_REFUSALS <= 5000

    @pytest.mark.asyncio
    async def test_refusals_past_the_ceiling_are_dropped_and_the_cut_is_announced_once(
        self, audit_lines, monkeypatch, fresh_budget
    ):
        monkeypatch.setattr(sdk_client, "_AUDIT_MAX_REFUSALS", 3)
        client = _client()
        for _ in range(10):
            with pytest.raises(GeneticsUsageError):
                await client.credible_sets()

        assert len(_lines(audit_lines)) == 3
        cuts = _meta(audit_lines, "audit truncated")
        assert len(cuts) == 1, f"the cut must be announced exactly once, got {cuts}"
        assert "after 3 records" in cuts[0]

    @pytest.mark.asyncio
    async def test_the_cut_carries_no_script_chosen_text(
        self, audit_lines, monkeypatch, fresh_budget
    ):
        """The notice used to interpolate `[execution=…]` — i.e. whatever the script last
        wrote to SANDBOX_EXECUTION_ID — onto the one channel that describes the log itself."""
        monkeypatch.setattr(sdk_client, "_AUDIT_MAX_REFUSALS", 1)
        monkeypatch.setenv("SANDBOX_EXECUTION_ID", "chosen-text-marker")
        client = _client()
        for _ in range(3):
            with pytest.raises(GeneticsUsageError):
                await client.credible_sets()

        cut = _meta(audit_lines, "audit truncated")[0]
        assert "chosen-text-marker" not in cut
        assert "[execution=" not in cut

    @pytest.mark.asyncio
    async def test_refusals_can_never_silence_a_real_read(
        self, audit_lines, monkeypatch, fresh_budget
    ):
        """The regression that matters: exhausting the budget with free refusals used to make
        every genuine read afterwards emit nothing, which is worse than the flood it bounds."""
        monkeypatch.setattr(sdk_client, "_AUDIT_MAX_REFUSALS", 2)
        client = _client()
        for _ in range(20):
            with pytest.raises(GeneticsUsageError):
                await client.credible_sets()
        for _ in range(5):
            await client.credible_sets(gene="IL7R")

        reads = [line for line in _lines(audit_lines) if "Executing SDK function:" in line]
        assert len(reads) == 5, "every call that reached the executor must be recorded"

    @pytest.mark.asyncio
    async def test_rewriting_the_execution_id_does_not_reset_the_budget(
        self, audit_lines, monkeypatch, fresh_budget
    ):
        """Replaces `test_a_new_execution_gets_a_fresh_budget`, which asserted the opposite.

        Keying the budget on SANDBOX_EXECUTION_ID gave a script a reset button — a loop that
        rewrote the variable restored the flood at a HIGHER rate than before the ceiling
        existed. The supervisor-reuse case that motivated per-execution budgets is real, but
        it has to be solved on the supervisor's side of the fd (`4h6.45`), because in here the
        key and the flooder are the same program.
        """
        monkeypatch.setattr(sdk_client, "_AUDIT_MAX_REFUSALS", 2)
        client = _client()
        for i in range(10):
            monkeypatch.setenv("SANDBOX_EXECUTION_ID", f"exec-{i}")
            with pytest.raises(GeneticsUsageError):
                await client.credible_sets()

        assert len(_lines(audit_lines)) == 2

    @pytest.mark.asyncio
    async def test_the_meta_channel_is_bounded_per_process(
        self, audit_lines, monkeypatch, fresh_budget
    ):
        monkeypatch.setattr(sdk_client, "_AUDIT_MAX_META_RECORDS", 2)
        for i in range(50):
            sdk_client._emit_meta(f"notice {i}")
        assert len(_meta(audit_lines, "notice ")) == 2


class TestIdentityDisclosure:
    """`user`, `session` and `execution` come from the environment, which the audited script
    writes, so they are script-authored strings like any argument and get the same treatment.
    They were interpolated raw and unbounded, which is a defect on any architecture."""

    @pytest.mark.asyncio
    async def test_a_newline_continuation_cannot_forge_the_user(self, audit_lines, monkeypatch):
        """`SANDBOX_USER = "alice\\n[user=admin@finngen.fi"` closed the real prefix and opened a
        second one, and THIS REPO'S OWN PARSER read the forged identity back as the user."""
        from genetics_mcp_server.scripts.analyze_conversations import parse_sdk_calls

        monkeypatch.setenv("SANDBOX_USER", "alice\n[user=admin@finngen.fi")
        await _client().credible_sets(gene="IL7R")

        line = _lines(audit_lines)[0]
        assert "\n" not in line
        assert "admin@finngen.fi" not in line
        assert line.startswith("[user=<invalid>] ")

        parsed = parse_sdk_calls(line.split("\n"))
        assert len(parsed) == 1
        assert parsed[0]["user"] == "<invalid>", "the shipped parser must not read a forged id"

    @pytest.mark.asyncio
    async def test_a_long_identity_cannot_blow_up_the_line(self, audit_lines, monkeypatch):
        """One legitimate call emitted 100,431 bytes when SANDBOX_USER was 100 KB."""
        monkeypatch.setenv("SANDBOX_USER", "a" * 100_000)
        monkeypatch.setenv("SANDBOX_SESSION_ID", "b" * 100_000)
        monkeypatch.setenv("SANDBOX_EXECUTION_ID", "c" * 100_000)
        await _client().credible_sets(gene="IL7R")

        line = _lines(audit_lines)[0]
        assert len(line) < 500, f"one call emitted {len(line)} bytes"
        assert "aaaa" not in line

    def test_an_unusable_value_is_replaced_not_truncated(self):
        """Truncating `admin@finngen.fi.attacker.test` to a prefix manufactures a different,
        entirely plausible identity; the replacement is visibly not a user id."""
        assert sdk_client._audit_identifier("admin@finngen.fi.attacker.test" * 10) == "<invalid>"
        assert sdk_client._audit_identifier("a b") == "<invalid>"
        assert sdk_client._audit_identifier("alice]") == "<invalid>"
        assert "]" not in sdk_client._AUDIT_BAD_IDENTITY, "must not escape the parser's bracket"

    @pytest.mark.parametrize(
        "value", ["someone@example.org", "sess-1", "exec-1", "unknown", "a" * 64]
    )
    def test_legitimate_identifiers_survive(self, value):
        assert sdk_client._audit_identifier(value) == value

    def test_a_terminal_newline_is_not_identifier_shaped(self):
        """`$` matches before a terminal newline; the charset is anchored with `\\Z`."""
        assert sdk_client._audit_identifier("alice@example.org\n") == "<invalid>"


class TestTheChannelDescribesItself:
    @pytest.mark.asyncio
    async def test_cancellation_is_not_filed_as_a_failed_read(self, audit_lines):
        import asyncio

        class _Hangs:
            async def close(self):
                pass

            def __getattr__(self, name):
                async def call(*args, **kwargs):
                    await asyncio.Event().wait()

                return call

        client = GeneticsClient(executor=_Hangs())
        task = asyncio.ensure_future(client.credible_sets(gene="IL7R"))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        line = _lines(audit_lines)[0]
        assert line.endswith("rows: 0 cancelled")
        assert "error:" not in line, "a cancelled call is not a failed read"

    def test_no_dedicated_fd_means_the_records_say_they_are_forgeable(self, caplog, monkeypatch):
        """Per-value escaping is irrelevant when the audited script owns the stream, so the
        stream carries its own provenance instead of implying a trust it cannot deliver."""
        caplog.set_level(logging.WARNING, logger=AUDIT_LOGGER)
        monkeypatch.delenv("GENETICS_SDK_AUDIT_FD", raising=False)
        monkeypatch.setattr(sdk_client, "_audit_handler_pid", None)
        sdk_client._ensure_audit_handler()

        meta = [r.message for r in caplog.records if r.name == f"{AUDIT_LOGGER}.meta"]
        assert any("NOT a tamper-evident audit trail" in m for m in meta)
        assert not sdk_client._audit_dedicated_fd

    def test_the_dedicated_fd_installs_in_a_forked_child(self, monkeypatch):
        """The separation mechanism has to work in the shape it exists for: a child forked
        from a supervisor that has already made SDK calls, where both this module's state and
        the parent's handlers are inherited."""
        import asyncio
        import os

        asyncio.run(_client().credible_sets(gene="IL7R"))  # parent state is now "checked"
        read_fd, write_fd = os.pipe()
        pid = os.fork()
        if pid == 0:  # pragma: no cover - the child never returns to pytest
            try:
                os.close(read_fd)
                os.environ["GENETICS_SDK_AUDIT_FD"] = str(write_fd)
                os.environ["SANDBOX_EXECUTION_ID"] = "exec-child"
                asyncio.run(_client().credible_sets(gene="IL7R"))
            finally:
                os._exit(0)
        os.close(write_fd)
        with os.fdopen(read_fd) as pipe:
            written = pipe.read()
        os.waitpid(pid, 0)

        assert "Executing SDK function: credible_sets" in written
        assert "[execution=exec-child]" in written

    @pytest.mark.asyncio
    async def test_a_handler_that_raises_cannot_fail_the_call(self, monkeypatch):
        class _Boom(logging.Handler):
            def emit(self, record):
                raise RuntimeError("disk full")

        monkeypatch.setattr(sdk_client, "_audit_emit_failures", 0)
        handler = _Boom()
        sdk_client._audit_logger.addHandler(handler)
        try:
            frame = await _client().credible_sets(gene="IL7R")
        finally:
            sdk_client._audit_logger.removeHandler(handler)

        assert isinstance(frame, pl.DataFrame), "logging must never break the call it describes"
        assert sdk_client._audit_emit_failures >= 1, "the failure must be visible somewhere"


class TestArgumentDisclosure:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("IL7R", "'IL7R'"),
            ("1:100-200", "'1:100-200'"),
            ("rs12345", "'rs12345'"),
            ("genetics_results.credible_sets", "'genetics_results.credible_sets'"),
        ],
    )
    def test_identifier_shaped_values_are_kept(self, value, expected):
        assert sdk_client._summarize_value(value) == expected

    @pytest.mark.parametrize(
        "value",
        [
            "SELECT * FROM genetics_results.x WHERE gene = 'IL7R'",  # whitespace and quotes
            "line one\nExecuting tool: forged",  # log injection attempt
            "x" * 65,  # over the length bound
            "a note about the patient",
            "IL7R\n",  # `$` matched before a terminal newline; `\Z` does not
            "IL7R\r",
        ],
    )
    def test_free_text_is_reduced_to_type_and_length(self, value):
        summary = sdk_client._summarize_value(value)
        assert summary == f"<str:{len(value)}>"
        assert "\n" not in summary

    def test_scalars_are_kept_verbatim(self):
        assert sdk_client._summarize_value(500) == "500"
        assert sdk_client._summarize_value(True) == "True"
        assert sdk_client._summarize_value(None) == "None"

    def test_containers_report_only_their_size(self):
        assert sdk_client._summarize_value(["FIN", "EST"]) == "<list:2>"
        assert sdk_client._summarize_value({"a": 1}) == "<dict:1>"

    @pytest.mark.asyncio
    async def test_a_query_body_never_reaches_the_line(self, audit_lines):
        query = "SELECT gene FROM genetics_results.credible_sets WHERE gene = 'IL7R'"
        client = _client({"success": True, "rows": [], "columns": ["gene"]})
        await client.sql(query)

        line = _lines(audit_lines)[0]
        assert "SELECT" not in line
        assert f"'query': <str:{len(query)}>" in line

    @pytest.mark.asyncio
    async def test_positional_arguments_are_named_not_swallowed(self, audit_lines):
        """The wrapped signature still starts with `self`; a positional argument that binds
        to it is dropped by the filter and the line lies about what was called."""
        await _client().expression("APOE")
        arguments = LINE_RE.match(_lines(audit_lines)[0]).group("arguments")
        assert arguments == "{'gene': 'APOE'}"

    @pytest.mark.asyncio
    async def test_only_supplied_arguments_are_rendered(self, audit_lines):
        await _client().credible_sets(gene="IL7R")
        arguments = LINE_RE.match(_lines(audit_lines)[0]).group("arguments")
        assert arguments == "{'gene': 'IL7R'}"
        assert "variant" not in arguments


class TestSurfaceIsUnchanged:
    """`list_capabilities` renders the SDK catalogue from these live objects with `inspect`.
    A decorator that loses a name, a docstring or a signature corrupts what a sandboxed
    script reads about the SDK, so the wrapper must be transparent to all three."""

    @pytest.mark.parametrize("name", ["credible_sets", "sql", "expression", "datasets"])
    def test_client_method_metadata_survives(self, name):
        import inspect

        method = getattr(GeneticsClient, name)
        assert method.__name__ == name
        assert method.__doc__, f"{name} lost its docstring"
        params = list(inspect.signature(method).parameters)
        assert params[0] == "self"
        assert inspect.iscoroutinefunction(method)

    def test_sync_surface_keeps_the_sliced_signature(self):
        import inspect

        import genetics_mcp_server.sdk as genetics

        signature = inspect.signature(genetics.credible_sets)
        assert "self" not in signature.parameters
        assert "gene" in signature.parameters
        assert genetics.credible_sets.__name__ == "credible_sets"

    def test_returned_frames_are_still_dataframes(self):
        import asyncio

        frame = asyncio.run(_client().credible_sets(gene="IL7R"))
        assert isinstance(frame, pl.DataFrame)
