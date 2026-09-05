"""Tests for message-history helpers in llm_service.

These cover the invariants the Anthropic API requires when replaying persisted
conversations: every tool_use must be paired with a matching tool_result, and
orphans must be stripped. This is the load-bearing logic for both tool_result
persistence (resumed conversations carry the data) and backward compatibility
(old conversations without persisted results behave exactly as before).

The second half covers the per-turn metrics row written where the "Chat complete"
line is logged, driving the same `_stream_anthropic` harness test_stream_truncation
uses.
"""

import json
import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from test_stream_truncation import _Block, _delta_event, _FakeMessage, _service

from genetics_mcp_server.cost import estimate_cost
from genetics_mcp_server.llm_service import (
    LLMService,
    _count_result_items,
    _mark_history_cache_breakpoint,
    _sanitize_tool_blocks,
    _strip_tool_use_markers,
    _truncation_notice,
)


def _tool_use(tid):
    return {"type": "tool_use", "id": tid, "name": "x", "input": {}}


def _tool_result(tid):
    return {"type": "tool_result", "tool_use_id": tid, "content": "ok"}


class TestSanitizeToolBlocks:
    def test_matched_pair_kept(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "text", "text": "calling"}, _tool_use("a")]},
            {"role": "user", "content": [_tool_result("a")]},
            {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
        ]
        out = _sanitize_tool_blocks(messages)
        # the tool_use and its tool_result both survive
        assert any(b.get("type") == "tool_use" for b in out[1]["content"])
        assert any(b.get("type") == "tool_result" for b in out[2]["content"])

    def test_interleaved_multiple_tool_uses_kept(self):
        """A consolidated assistant turn with interleaved text + several tool_use
        blocks, answered by one user message with all tool_results, is preserved.
        This is exactly the shape produced when replaying a persisted turn."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "t1"},
                    _tool_use("a"),
                    {"type": "text", "text": "t2"},
                    _tool_use("b"),
                    {"type": "text", "text": "t3"},
                ],
            },
            {"role": "user", "content": [_tool_result("a"), _tool_result("b")]},
        ]
        out = _sanitize_tool_blocks(messages)
        kept_uses = {b["id"] for b in out[0]["content"] if b.get("type") == "tool_use"}
        kept_results = {b["tool_use_id"] for b in out[1]["content"] if b.get("type") == "tool_result"}
        assert kept_uses == {"a", "b"}
        assert kept_results == {"a", "b"}

    def test_orphan_tool_use_stripped(self):
        """Backward compat: an assistant tool_use with no following tool_result
        message (old conversation, results not persisted) is stripped."""
        messages = [
            {"role": "assistant", "content": [{"type": "text", "text": "answer"}, _tool_use("a")]},
            {"role": "user", "content": "next question"},
        ]
        out = _sanitize_tool_blocks(messages)
        assert all(b.get("type") != "tool_use" for b in out[0]["content"])
        # the text block survives
        assert any(b.get("type") == "text" for b in out[0]["content"])

    def test_orphan_tool_result_stripped(self):
        messages = [
            {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
            {"role": "user", "content": [_tool_result("a"), {"type": "text", "text": "q"}]},
        ]
        out = _sanitize_tool_blocks(messages)
        assert all(b.get("type") != "tool_result" for b in out[1]["content"])

    def test_partial_match_only_unmatched_stripped(self):
        messages = [
            {"role": "assistant", "content": [_tool_use("a"), _tool_use("b")]},
            {"role": "user", "content": [_tool_result("a")]},
        ]
        out = _sanitize_tool_blocks(messages)
        kept_uses = {b["id"] for b in out[0]["content"] if b.get("type") == "tool_use"}
        assert kept_uses == {"a"}


class TestStripToolUseMarkers:
    """The '*[Using tool: ...]*' markers are display-only. They must be removed from
    replayed assistant content so the model never learns to imitate them as prose
    instead of emitting real tool_use blocks (the fabrication failure mode)."""

    def test_strips_marker_from_string_content(self):
        messages = [
            {"role": "assistant", "content": "*[Using tool: get_variant_annotations; variant: 8:1:C:T]*\n\nThe answer is 42."},
        ]
        out = _strip_tool_use_markers(messages)
        assert "Using tool" not in out[0]["content"]
        assert "The answer is 42." in out[0]["content"]

    def test_strips_multiline_marker_with_sql(self):
        """Marker params can span lines (SQL) — DOTALL + non-greedy must still match."""
        text = (
            "*[Using tool: query_database; sql: SELECT a, b\nFROM t\nWHERE x = '1:2:C:T'\nLIMIT 200]*"
            "\n\n## Result\nrows: 0"
        )
        out = _strip_tool_use_markers([{"role": "assistant", "content": text}])
        assert "Using tool" not in out[0]["content"]
        assert "## Result" in out[0]["content"]

    def test_strips_marker_from_text_block(self):
        messages = [{
            "role": "assistant",
            "content": [
                {"type": "text", "text": "*[Using tool: x; a: b]*\n\nreal prose"},
                {"type": "tool_use", "id": "a", "name": "x", "input": {}},
            ],
        }]
        out = _strip_tool_use_markers(messages)
        text_blocks = [b for b in out[0]["content"] if b.get("type") == "text"]
        assert text_blocks[0]["text"] == "real prose"
        # real tool_use is untouched
        assert any(b.get("type") == "tool_use" for b in out[0]["content"])

    def test_marker_only_text_block_dropped(self):
        messages = [{
            "role": "assistant",
            "content": [
                {"type": "text", "text": "*[Using tool: x; a: b]*"},
                {"type": "tool_use", "id": "a", "name": "x", "input": {}},
            ],
        }]
        out = _strip_tool_use_markers(messages)
        assert all(b.get("type") != "text" for b in out[0]["content"])
        assert any(b.get("type") == "tool_use" for b in out[0]["content"])

    def test_marker_only_string_falls_back_to_original(self):
        """Never emit empty content — a turn that was nothing but a marker keeps
        its original content rather than becoming an empty (API-invalid) message."""
        messages = [{"role": "assistant", "content": "*[Using tool: x; a: b]*"}]
        out = _strip_tool_use_markers(messages)
        assert out[0]["content"] != ""

    def test_user_and_string_messages_untouched(self):
        messages = [
            {"role": "user", "content": "*[Using tool: x]* (user typed this, leave it)"},
            {"role": "assistant", "content": "no markers here"},
        ]
        out = _strip_tool_use_markers(messages)
        assert out[0]["content"] == "*[Using tool: x]* (user typed this, leave it)"
        assert out[1]["content"] == "no markers here"


class TestMarkHistoryCacheBreakpoint:
    def test_marks_last_block_of_last_message(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
        _mark_history_cache_breakpoint(messages)
        assert messages[-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}

    def test_normalizes_string_content(self):
        messages = [{"role": "user", "content": "hello"}]
        _mark_history_cache_breakpoint(messages)
        last = messages[-1]["content"]
        assert isinstance(last, list)
        assert last[-1]["type"] == "text"
        assert last[-1]["cache_control"] == {"type": "ephemeral"}

    def test_empty_messages_noop(self):
        messages = []
        _mark_history_cache_breakpoint(messages)
        assert messages == []


class TestTruncationNotice:
    """The notice appended when a tool result exceeds mcp_max_result_size.

    Truncation keeps an ordered prefix, so the notice has to say the invisible part is
    not a random sample and block absence/count conclusions drawn from what survives.
    """

    def test_counts_summarized_credible_sets(self):
        """The summarized shape has n_cs/cs, not results; its count used to be lost."""
        result = {"n_cs": 159, "cs": {"pQTL": [{}, {}], "caQTL": [{}]}}
        assert _count_result_items(result) == 159
        assert "159 total items" in _truncation_notice(result)

    def test_counts_cs_groups_without_n_cs(self):
        assert _count_result_items({"cs": {"pQTL": [{}, {}], "caQTL": [{}]}}) == 3

    def test_counts_variant_level_and_query_shapes(self):
        assert _count_result_items({"results": [1, 2, 3]}) == 3
        assert _count_result_items({"rows": [1, 2], "total_rows": 500}) == 500
        assert _count_result_items({"rows": [1, 2]}) == 2

    def test_unknown_shape_degrades_without_a_count(self):
        assert _count_result_items({"foo": 1}) is None
        assert _count_result_items("not a dict") is None
        notice = _truncation_notice({"foo": 1})
        assert "a larger result" in notice
        assert "total items" not in notice

    def test_notice_warns_against_absence_and_count_conclusions(self):
        notice = _truncation_notice({"results": [1]})
        assert "TRUNCATED" in notice
        assert "ORDERED" in notice
        assert "absent" in notice
        assert "summarize=true" in notice


MODEL = "claude-opus-5"


def _usage(message, *, input_tokens, output_tokens, cache_read=0, cache_create=0):
    message.usage = SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_create,
    )
    return message


def _answer_turn(text="done", **usage):
    message = _FakeMessage([_Block("text", text=text)], "end_turn")
    return ([_delta_event("text_delta", text)], _usage(message, **usage))


def _tool_turn(*tool_ids, **usage):
    blocks = [_Block("tool_use", id=tid, name="lookup", input={}) for tid in tool_ids]
    return ([], _usage(_FakeMessage(blocks, "tool_use"), **usage))


async def _run(svc, **kwargs):
    params = dict(
        messages=[{"role": "user", "content": "hi"}],
        model=MODEL,
        system_prompt=None,
        enable_tools=False,
        session_id="sess1",
        message_id="msg1",
    )
    params.update(kwargs)
    return [chunk async for chunk in svc._stream_anthropic(**params)]


def _with_tools(svc):
    """Make the loop execute tool_use blocks: it needs a truthy executor, and the
    executor itself is irrelevant to what is being counted."""
    svc.executor = object()

    async def _execute(
        name,
        tool_input,
        literature_backend=None,
        user=None,
        session_id=None,
        gateway_asserted=False,
    ):
        # the signature stays exact: the loop calls this positionally, so a widened
        # *args stub would silently accept a call shape production could never make.
        # `gateway_asserted` is the sixth positional since genetics-results-suite-4h6.84;
        # it arrives here because the loop forwards it to every tool, and only
        # run_analysis reads it
        return {"success": True, "rows": []}

    svc._execute_tool = _execute
    return svc


class TestTurnMetrics:
    """The chat_turn_metrics row written where the 'Chat complete' line is emitted."""

    def _patch_db(self, db):
        return patch(
            "genetics_mcp_server.db.chat_history_db.get_chat_history_db",
            return_value=db,
        )

    @pytest.mark.asyncio
    async def test_records_one_row_with_turn_totals(self, chat_history_db):
        turns = [
            _tool_turn("t1", input_tokens=100, output_tokens=50, cache_read=900, cache_create=40),
            _answer_turn(input_tokens=200, output_tokens=80, cache_read=1100, cache_create=0),
        ]
        svc = _with_tools(_service(turns))

        with self._patch_db(chat_history_db):
            await _run(svc, tool_profile="bigquery")

        rows = chat_history_db.get_turn_metrics("sess1")
        assert len(rows) == 1
        row = rows[0]
        assert row["message_id"] == "msg1"
        assert row["iterations"] == 2
        assert row["tool_call_count"] == 1
        assert row["input_tokens"] == 300
        assert row["output_tokens"] == 130
        assert row["cache_read_tokens"] == 2000
        assert row["cache_create_tokens"] == 40
        assert row["tool_profile"] == "bigquery"
        assert row["model"] == MODEL
        assert row["wall_ms"] >= 0
        expected_cost = estimate_cost(MODEL, 100, 50, 900, 40) + estimate_cost(MODEL, 200, 80, 1100, 0)
        assert row["cost_usd"] == pytest.approx(expected_cost)

    @pytest.mark.asyncio
    async def test_tool_call_count_spans_parallel_and_sequential_calls(self, chat_history_db):
        """The count content_json cannot reconstruct: two parallel calls in one roundtrip
        plus one in the next are three calls over three iterations."""
        turns = [
            _tool_turn("a", "b", input_tokens=1, output_tokens=1),
            _tool_turn("c", input_tokens=1, output_tokens=1),
            _answer_turn(input_tokens=1, output_tokens=1),
        ]
        svc = _with_tools(_service(turns))

        with self._patch_db(chat_history_db):
            await _run(svc)

        row = chat_history_db.get_turn_metrics("sess1")[0]
        assert row["iterations"] == 3
        assert row["tool_call_count"] == 3

    @pytest.mark.asyncio
    async def test_tool_uses_are_not_counted_without_an_executor(self, chat_history_db):
        """The loop breaks before executing them, so nothing was actually called."""
        turns = [_tool_turn("a", input_tokens=1, output_tokens=1)]
        svc = _service(turns)

        with self._patch_db(chat_history_db):
            await _run(svc)

        assert chat_history_db.get_turn_metrics("sess1")[0]["tool_call_count"] == 0

    @pytest.mark.asyncio
    async def test_secret_chat_records_nothing(self, chat_history_db):
        """Secret chat is promised to leave no trace. Counts and costs are not content,
        but a row still says a conversation happened and how expensive it was."""
        turns = [_answer_turn(input_tokens=100, output_tokens=50)]
        svc = _service(turns)

        with self._patch_db(chat_history_db) as get_db:
            chunks = await _run(svc, secret=True)

        assert get_db.call_count == 0
        count = chat_history_db._conn.execute(
            "SELECT COUNT(*) FROM chat_turn_metrics"
        ).fetchone()[0]
        assert count == 0
        # the answer itself is unaffected
        assert [c.type for c in chunks].count("done") == 1

    @pytest.mark.asyncio
    async def test_a_db_failure_does_not_truncate_the_answer(self, chat_history_db):
        turns = [_answer_turn(input_tokens=100, output_tokens=50)]
        svc = _service(turns)

        boom = patch.object(
            type(chat_history_db),
            "record_turn_metrics",
            side_effect=RuntimeError("database is locked"),
        )
        with self._patch_db(chat_history_db), boom:
            chunks = await _run(svc)

        assert [c.type for c in chunks].count("done") == 1
        assert "".join(c.content for c in chunks if c.type == "text") == "done"

    @pytest.mark.asyncio
    async def test_records_a_turn_that_hit_the_iteration_cap(self, chat_history_db):
        """The expensive tail the benchmark exists to measure must not be the one case
        that goes unrecorded."""
        turns = [_tool_turn(f"t{i}", input_tokens=1, output_tokens=1) for i in range(3)]
        svc = _with_tools(_service(turns))

        with self._patch_db(chat_history_db), patch(
            "genetics_mcp_server.llm_service.get_settings"
        ) as get_settings:
            get_settings.return_value = SimpleNamespace(
                default_provider="anthropic",
                default_model=MODEL,
                max_tokens=1000,
                temperature=None,
                mcp_enabled=False,
                mcp_max_iterations=3,
                mcp_max_result_size=100_000,
                max_continuations=1,
                disabled_tools=[],
            )
            chunks = await _run(svc)

        row = chat_history_db.get_turn_metrics("sess1")[0]
        assert row["iterations"] == 3
        assert row["tool_call_count"] == 3
        assert "Max tool iterations reached" in "".join(
            c.content for c in chunks if c.type == "text"
        )

    @pytest.mark.asyncio
    async def test_done_is_emitted_before_the_metrics_write(self, chat_history_db):
        """The done chunk carries the message_content the client persists. An await ahead
        of it can block on SQLite's busy timeout while the nightly job holds the write lock,
        and is a cancellation point a client disconnect can land on."""
        turns = [_answer_turn(input_tokens=100, output_tokens=50)]
        svc = _service(turns)

        events = []
        real = type(chat_history_db).record_turn_metrics

        def _spy(db_self, **kwargs):
            events.append("write")
            return real(db_self, **kwargs)

        with self._patch_db(chat_history_db), patch.object(
            type(chat_history_db), "record_turn_metrics", _spy
        ):
            async for chunk in svc._stream_anthropic(
                messages=[{"role": "user", "content": "hi"}],
                model=MODEL,
                system_prompt=None,
                enable_tools=False,
                session_id="sess1",
                message_id="msg1",
                user="user@example.com",
            ):
                events.append(chunk.type)

        assert "write" in events
        assert events.index("done") < events.index("write")

    @pytest.mark.asyncio
    async def test_records_the_authenticated_user(self, chat_history_db):
        """Without user_id the upsert cannot tell whose row a client message_id targets,
        and per-user cost analysis is impossible."""
        turns = [_answer_turn(input_tokens=100, output_tokens=50)]
        svc = _service(turns)

        with self._patch_db(chat_history_db):
            await _run(svc, user="user@example.com")

        assert chat_history_db.get_turn_metrics("sess1")[0]["user_id"] == "user@example.com"

    @pytest.mark.asyncio
    async def test_missing_client_ids_still_record(self, chat_history_db):
        """The browser creates the session only after the first exchange, so a
        conversation's opening turn streams with no session id at all."""
        turns = [_answer_turn(input_tokens=100, output_tokens=50)]
        svc = _service(turns)

        with self._patch_db(chat_history_db):
            await _run(svc, session_id=None, message_id=None)

        row = chat_history_db._conn.execute(
            "SELECT session_id, message_id, iterations FROM chat_turn_metrics"
        ).fetchone()
        assert row["session_id"] is None
        assert row["message_id"] is None
        assert row["iterations"] == 1


class TestUsageChunk:
    """The per-iteration `usage` chunk the replay benchmark reads cost off.

    A benchmark run is secret=true, which writes no chat_turn_metrics row, so the
    stream is the only place its cache split can come from. Cache reads and cache
    creations are priced far apart, so folding them into one number makes an exact cost
    underivable — they have to arrive as separate fields.
    """

    def _patch_db(self, db):
        return patch(
            "genetics_mcp_server.db.chat_history_db.get_chat_history_db",
            return_value=db,
        )

    @staticmethod
    def _payloads(chunks):
        return [json.loads(c.content) for c in chunks if c.type == "usage"]

    @staticmethod
    def _two_turns():
        return [
            _tool_turn("t1", input_tokens=100, output_tokens=50, cache_read=900, cache_create=40),
            _answer_turn(input_tokens=200, output_tokens=80, cache_read=1100, cache_create=7),
        ]

    @pytest.mark.asyncio
    async def test_reports_the_cache_split_per_iteration(self, chat_history_db):
        svc = _with_tools(_service(self._two_turns()))

        with self._patch_db(chat_history_db):
            chunks = await _run(svc, tool_profile="bigquery")

        payloads = self._payloads(chunks)
        assert len(payloads) == 2
        first, second = payloads

        assert first["iteration"] == 1
        assert first["cache_read"] == 900
        assert first["cache_create"] == 40
        assert first["output_tokens"] == 50
        assert first["total_input_tokens"] == 100
        # deliberately unchanged: input_tokens is the whole context, what the browser's
        # context meter renders against context_window
        assert first["input_tokens"] == 100 + 900 + 40

        assert second["iteration"] == 2
        assert second["cache_read"] == 1100
        assert second["cache_create"] == 7
        assert second["total_input_tokens"] == 300
        assert second["input_tokens"] == 200 + 1100 + 7

    @pytest.mark.asyncio
    async def test_the_stream_alone_reproduces_the_recorded_cost(self, chat_history_db):
        """The point of the split: cost computed from the chunks must equal the cost the
        metrics row records, exactly, not an interval bracketing it.

        This pins the payload arithmetic, not the completeness of the accounting. Both
        sides share the same blind spots — subagent calls and retried attempts are absent
        from the stream and from `total_cost` alike — so equality here says the three
        token components round-trip, not that either number is the turn's true spend.
        """
        svc = _with_tools(_service(self._two_turns()))

        with self._patch_db(chat_history_db):
            chunks = await _run(svc, tool_profile="bigquery")

        from_stream = sum(
            estimate_cost(
                MODEL,
                p["input_tokens"] - p["cache_read"] - p["cache_create"],
                p["output_tokens"],
                p["cache_read"],
                p["cache_create"],
            )
            for p in self._payloads(chunks)
        )
        recorded = chat_history_db.get_turn_metrics("sess1")[0]["cost_usd"]
        assert from_stream == pytest.approx(recorded)


class TestRunAnalysisDisplayInput:
    """The tool-use indicator and its log line mirror the execution-side identity strip.

    _execute_tool discards a model-supplied `user`/`session_id` before calling
    run_analysis, but the indicator is rendered from the raw tool_use input — so a forged
    identity would still be logged and streamed as if it were a real argument, which is
    exactly what a log join reads as identity.
    """

    def _patch_db(self, db):
        return patch(
            "genetics_mcp_server.db.chat_history_db.get_chat_history_db",
            return_value=db,
        )

    def _turns(self, tool_input):
        blocks = [_Block("tool_use", id="t1", name="run_analysis", input=tool_input)]
        return [
            ([], _usage(_FakeMessage(blocks, "tool_use"), input_tokens=1, output_tokens=1)),
            _answer_turn(input_tokens=1, output_tokens=1),
        ]

    def _streamed(self, chunks):
        """Everything that reaches the client, whatever chunk type carried it.

        Deliberately not just the text chunks: the identity strip below is a claim about
        what the USER can see, so narrowing it to one chunk type would let a forged
        `user` pass simply by moving to another one.
        """
        return "".join(c.content for c in chunks if c.type in ("text", "tool_use"))

    def _tool_uses(self, chunks):
        return [json.loads(c.content) for c in chunks if c.type == "tool_use"]

    @pytest.mark.asyncio
    async def test_forged_identity_is_neither_streamed_nor_logged(self, chat_history_db, caplog):
        turns = self._turns({
            "code": "print(1)",
            "user": "attacker@evil.example",
            "session_id": "other-sid",
        })
        svc = _with_tools(_service(turns))

        with self._patch_db(chat_history_db), caplog.at_level(
            logging.INFO, logger="genetics_mcp_server.llm_service"
        ):
            chunks = await _run(svc, enable_tools=True)

        streamed = self._streamed(chunks)
        assert "attacker@evil.example" not in streamed
        assert "other-sid" not in streamed
        assert "attacker@evil.example" not in caplog.text

        (tool_use,) = self._tool_uses(chunks)
        assert tool_use["name"] == "run_analysis"
        assert tool_use["id"] == "t1"
        assert "user" not in tool_use["input"]
        assert "session_id" not in tool_use["input"]

    @pytest.mark.asyncio
    async def test_the_whole_script_is_streamed_while_the_log_line_stays_capped(
        self, chat_history_db, caplog
    ):
        """The cap is a log concern, not a display one (genetics-results-suite-inp).

        It used to bound both, which meant the one field the user most needed to read was
        the one field guaranteed to be cut off. The client renders this collapsed, so
        size is not a reason to withhold it.
        """
        code = "x = 1  # padding\n" * 4000
        svc = _with_tools(_service(self._turns({"code": code})))

        with self._patch_db(chat_history_db), caplog.at_level(
            logging.INFO, logger="genetics_mcp_server.llm_service"
        ):
            chunks = await _run(svc, enable_tools=True)

        (tool_use,) = self._tool_uses(chunks)
        assert tool_use["input"]["code"] == code
        assert "chars total" not in tool_use["input"]["code"]

        assert len(caplog.text) < len(code)
        assert f"{len(code)} chars total" in caplog.text


class TestResolveLocalToolNames:
    """The system prompt is assembled from this, so it must agree with the tool list.

    Both failure modes below used to survive the whole suite: every test disabled
    subagents through `Settings.enable_subagents` while ALSO leaving
    `subagent_service = None`, so the flag and the liveness check were indistinguishable,
    and nothing exercised `mcp_enabled` here at all.
    """

    @staticmethod
    def _svc(*, subagent_service):
        svc = LLMService.__new__(LLMService)
        svc.subagent_service = subagent_service
        return svc

    @staticmethod
    def _patch_settings(monkeypatch, **overrides):
        from genetics_mcp_server.config.settings import Settings

        settings = Settings(**overrides)
        monkeypatch.setattr("genetics_mcp_server.llm_service.get_settings", lambda: settings)

    def test_mcp_disabled_advertises_nothing(self, monkeypatch):
        """MCP_ENABLED=false hands the model no tools, so the prompt must name none."""
        self._patch_settings(monkeypatch, mcp_enabled=False)
        assert self._svc(subagent_service=object()).resolve_local_tool_names() == set()

    def test_mcp_enabled_advertises_tools(self, monkeypatch):
        self._patch_settings(monkeypatch, mcp_enabled=True)
        assert len(self._svc(subagent_service=object()).resolve_local_tool_names()) > 20

    def test_enable_tools_false_advertises_nothing(self, monkeypatch):
        self._patch_settings(monkeypatch, mcp_enabled=True)
        svc = self._svc(subagent_service=object())
        assert svc.resolve_local_tool_names(enable_tools=False) == set()

    def test_flag_on_but_service_dead_still_hides_launch_subagents(self, monkeypatch):
        """ENABLE_SUBAGENTS=true with no live subagent service.

        `Settings.disabled_tools` gates on the flag alone, so only the liveness check
        removes the tool here. This is the exact drift 4h6.69 exists to prevent: the
        prompt would otherwise carry the whole Subagent Orchestration section for a tool
        the model was never handed.
        """
        self._patch_settings(monkeypatch, enable_subagents=True)
        dead = self._svc(subagent_service=None).resolve_local_tool_names()
        live = self._svc(subagent_service=object()).resolve_local_tool_names()
        assert "launch_subagents" not in dead
        assert "launch_subagents" in live

    def test_prompt_matches_the_resolution_when_the_service_is_dead(self, monkeypatch):
        from genetics_mcp_server.config.defaults import default_system_prompt

        self._patch_settings(monkeypatch, enable_subagents=True)
        names = self._svc(subagent_service=None).resolve_local_tool_names()
        prompt = default_system_prompt("FinnGenie", tool_names=names)
        assert "launch_subagents" not in prompt
        assert "Subagent Orchestration" not in prompt
