"""analyze_conversations counting SDK function calls alongside tool calls (4h6.12).

The two counts must stay separate and both must survive: a tool call is one model decision,
an SDK call is one line of a script the model wrote, and a `run_analysis` session would
otherwise report a single tool call for an unbounded number of queries.

The regex under test parses the lines `genetics_mcp_server/sdk/client.py` emits. The first
test here consumes a line produced by the SDK itself rather than a hand-written one, so the
two cannot drift apart in a way that leaves the analyzer silently counting nothing.
"""

import logging

import polars as pl
import pytest

from genetics_mcp_server.scripts.analyze_conversations import (
    _SDK_SEQUENCE_MAX,
    SDK_SHARED_STREAM_MARKER,
    ConversationMetrics,
    build_session_sdk_stats,
    build_session_tool_stats,
    compute_all_metrics,
    generate_report,
    parse_sdk_calls,
    scan_sdk_notices,
)
from genetics_mcp_server.sdk.client import SHARED_STREAM_WARNING, GeneticsClient
from genetics_mcp_server.sdk.errors import GeneticsUsageError


class _StubExecutor:
    async def close(self):
        pass

    def __getattr__(self, name):
        async def call(*args, **kwargs):
            return {"success": True, "results": [["IL7R", 0.9]], "columns": ["gene", "pip"]}

        return call


@pytest.mark.asyncio
async def test_parser_consumes_a_line_the_sdk_actually_emitted(caplog, monkeypatch):
    monkeypatch.setenv("SANDBOX_SESSION_ID", "sess-a")
    monkeypatch.setenv("SANDBOX_USER", "someone@example.org")
    monkeypatch.setenv("SANDBOX_EXECUTION_ID", "exec-a")
    caplog.set_level(logging.INFO, logger="genetics_mcp_server.sdk.audit")

    await GeneticsClient(executor=_StubExecutor()).credible_sets(gene="IL7R")
    emitted = [r.message for r in caplog.records if r.name == "genetics_mcp_server.sdk.audit"]

    calls = parse_sdk_calls(emitted)
    assert len(calls) == 1
    assert calls[0] == {
        "session_id": "sess-a",
        "user": "someone@example.org",
        "execution_id": "exec-a",
        "sdk_function": "credible_sets",
        "sdk_rows": 1,
        "sdk_error": "",
    }


def test_parser_skips_unrelated_lines_and_reads_formatted_ones():
    lines = [
        "2026-08-14 10:00:00,000 - genetics_mcp_server.sdk.audit - INFO - "
        "[user=u@x] [session=s1] [execution=e1] Executing SDK function: expression "
        "with input: {'gene': 'IL7R'} rows: 12",
        "2026-08-14 10:00:01,000 - llm_service - INFO - [user=u@x] [session=s1] "
        "Executing tool: get_expression with input: {'gene': 'IL7R'}",
        "some unrelated line",
        "[user=u@x] [session=s1] [execution=e1] Executing SDK function: sql "
        "with input: {'query': <str:120>} rows: 0 error: GeneticsError",
    ]
    calls = parse_sdk_calls(lines)
    assert [c["sdk_function"] for c in calls] == ["expression", "sql"]
    assert [c["sdk_rows"] for c in calls] == [12, 0]
    assert calls[1]["sdk_error"] == "GeneticsError"


def test_session_stats_aggregate_calls_rows_and_executions():
    calls = [
        {"session_id": "s1", "user": "u", "execution_id": "e1",
         "sdk_function": "expression", "sdk_rows": 10, "sdk_error": ""},
        {"session_id": "s1", "user": "u", "execution_id": "e1",
         "sdk_function": "sql", "sdk_rows": 5, "sdk_error": ""},
        {"session_id": "s1", "user": "u", "execution_id": "e2",
         "sdk_function": "expression", "sdk_rows": 0, "sdk_error": ""},
        {"session_id": "s2", "user": "u", "execution_id": "e3",
         "sdk_function": "ld", "sdk_rows": 3, "sdk_error": ""},
    ]
    stats = build_session_sdk_stats(calls).sort("session_id")
    assert stats["total_sdk_calls"].to_list() == [3, 1]
    assert stats["sdk_rows"].to_list() == [15, 3]
    assert stats["sdk_executions"].to_list() == [2, 1]
    assert stats["unique_sdk_functions"].to_list() == [2, 1]


def test_empty_input_gives_the_joinable_empty_frame():
    stats = build_session_sdk_stats([])
    assert stats.height == 0
    assert "total_sdk_calls" in stats.columns


def _sessions_and_messages():
    sessions = pl.DataFrame({
        "id": ["s1"], "user_id": ["u@x"], "created_at": ["2026-08-01"], "rating": [None],
    })
    messages = pl.DataFrame({
        "session_id": ["s1", "s1"],
        "role": ["user", "assistant"],
        "content": ["hi", "*[Using tool: run_analysis; script: ...]*"],
        "content_json": [
            None,
            '[{"type": "tool_use", "name": "run_analysis", "input": {}}]',
        ],
        "created_at": ["2026-08-01", "2026-08-01"],
        "thumbs_up": [None, None],
        "tool_profile": [None, None],
        "_rowid": [1, 2],
    })
    return sessions, messages


def test_sdk_calls_are_counted_alongside_tool_calls():
    sessions, messages = _sessions_and_messages()
    sdk_stats = build_session_sdk_stats([
        {"session_id": "s1", "user": "u@x", "execution_id": "e1",
         "sdk_function": "expression", "sdk_rows": 40, "sdk_error": ""},
        {"session_id": "s1", "user": "u@x", "execution_id": "e1",
         "sdk_function": "sql", "sdk_rows": 2, "sdk_error": ""},
    ])
    metrics = compute_all_metrics(
        sessions, messages, build_session_tool_stats(messages), {}, sdk_stats=sdk_stats
    )
    assert len(metrics) == 1
    m = metrics[0]
    # the one run_analysis tool call is still one tool call, and stays one
    assert m.total_tool_calls == 1
    assert m.total_sdk_calls == 2
    assert m.sdk_rows == 42
    assert m.unique_sdk_functions == 2
    assert m.sdk_sequence == "expression -> sql"


def test_without_an_sdk_log_the_counts_are_zero_and_tool_counts_are_untouched():
    sessions, messages = _sessions_and_messages()
    metrics = compute_all_metrics(sessions, messages, build_session_tool_stats(messages), {})
    assert metrics[0].total_tool_calls == 1
    assert metrics[0].total_sdk_calls == 0
    assert metrics[0].sdk_sequence == ""


def test_report_states_the_absence_of_a_log_rather_than_reporting_zero():
    sessions, messages = _sessions_and_messages()
    metrics = compute_all_metrics(sessions, messages, build_session_tool_stats(messages), {})
    report = generate_report(metrics, sessions, messages, build_session_tool_stats(messages))
    assert "SDK function calls (sandboxed scripts)" in report
    assert "No SDK audit log supplied" in report


def test_report_lists_sdk_functions_when_a_log_is_supplied():
    sessions, messages = _sessions_and_messages()
    sdk_stats = build_session_sdk_stats([
        {"session_id": "s1", "user": "u@x", "execution_id": "e1",
         "sdk_function": "expression", "sdk_rows": 40, "sdk_error": ""},
    ])
    metrics = compute_all_metrics(
        sessions, messages, build_session_tool_stats(messages), {}, sdk_stats=sdk_stats
    )
    report = generate_report(
        metrics, sessions, messages, build_session_tool_stats(messages), sdk_stats=sdk_stats
    )
    assert "| SDK function | Calls |" in report
    assert "| expression | 1 |" in report
    assert "Total rows returned to scripts: 40" in report


def test_calls_matching_no_session_are_reported_not_dropped():
    """Until 4h6.43/4h6.44 deliver the token, every line says session=unknown."""
    sessions, messages = _sessions_and_messages()
    sdk_stats = build_session_sdk_stats([
        {"session_id": "unknown", "user": "unknown", "execution_id": "unknown",
         "sdk_function": "sql", "sdk_rows": 9, "sdk_error": ""},
    ])
    metrics = compute_all_metrics(
        sessions, messages, build_session_tool_stats(messages), {}, sdk_stats=sdk_stats
    )
    assert metrics[0].total_sdk_calls == 0
    report = generate_report(
        metrics, sessions, messages, build_session_tool_stats(messages), sdk_stats=sdk_stats
    )
    assert "Total SDK function calls: 1" in report
    assert "match no session in this DB" in report


def test_metrics_dataclass_carries_the_sdk_fields():
    m = ConversationMetrics(session_id="s1")
    assert m.total_sdk_calls == 0
    assert m.sdk_rows == 0
    assert m.sdk_sequence == ""


@pytest.mark.asyncio
async def test_a_refused_call_is_not_parsed_as_a_data_access(caplog, monkeypatch):
    """The SDK gives a call that never reached the executor a different marker; the analyzer
    must count it as a refusal and never as a read."""
    monkeypatch.setenv("SANDBOX_EXECUTION_ID", "exec-r")
    caplog.set_level(logging.INFO, logger="genetics_mcp_server.sdk.audit")
    with pytest.raises(GeneticsUsageError):
        await GeneticsClient(executor=_StubExecutor()).credible_sets()
    emitted = [r.message for r in caplog.records if r.name == "genetics_mcp_server.sdk.audit"]

    assert parse_sdk_calls(emitted) == []
    assert scan_sdk_notices(emitted)["rejected"] == 1


def test_a_cancelled_call_is_distinguishable_from_a_failed_read():
    lines = [
        "[user=u] [session=s1] [execution=e1] Executing SDK function: sql "
        "with input: {'query': <str:80>} rows: 0 cancelled",
        "[user=u] [session=s1] [execution=e1] Executing SDK function: sql "
        "with input: {'query': <str:80>} rows: 0 error: GeneticsError",
    ]
    assert [c["sdk_error"] for c in parse_sdk_calls(lines)] == ["cancelled", "GeneticsError"]


def test_notices_report_a_truncated_and_shared_stream():
    lines = [
        f"WARNING - {SHARED_STREAM_WARNING}",
        "[execution=e1] SDK audit truncated after 1000 records; further SDK calls from this "
        "execution are NOT recorded.",
    ]
    notices = scan_sdk_notices(lines)
    assert notices["shared_stream"] is True
    assert notices["truncated"] == 1
    assert notices["truncated_at"] == 1000


def test_the_marker_the_analyzer_looks_for_is_the_one_the_sdk_writes():
    """Otherwise a report silently claims nothing about a forgeable log."""
    assert SDK_SHARED_STREAM_MARKER in SHARED_STREAM_WARNING


def test_report_says_the_trail_is_forgeable_when_the_log_came_from_a_shared_stream():
    sessions, messages = _sessions_and_messages()
    sdk_stats = build_session_sdk_stats([
        {"session_id": "s1", "user": "u@x", "execution_id": "e1",
         "sdk_function": "expression", "sdk_rows": 4, "sdk_error": ""},
    ])
    metrics = compute_all_metrics(
        sessions, messages, build_session_tool_stats(messages), {}, sdk_stats=sdk_stats
    )
    report = generate_report(
        metrics, sessions, messages, build_session_tool_stats(messages), sdk_stats=sdk_stats,
        sdk_notices={"shared_stream": True, "truncated": 1, "truncated_at": 1000, "rejected": 7},
    )
    assert "not a tamper-evident audit trail" in report
    assert "hit the SDK audit ceiling of 1000 records" in report
    assert "Calls refused before any data access: 7" in report


def test_the_sequence_is_bounded_while_the_per_function_totals_stay_exact():
    """A script can make thousands of SDK calls from one `run_analysis`; the joined sequence
    would otherwise be a ~150 KB cell in the metrics row and in every CSV."""
    calls = [
        {"session_id": "s1", "user": "u@x", "execution_id": "e1",
         "sdk_function": "expression", "sdk_rows": 1, "sdk_error": ""}
        for _ in range(_SDK_SEQUENCE_MAX + 70)
    ]
    stats = build_session_sdk_stats(calls)
    sequence = stats["sdk_sequence"].to_list()[0]
    assert sequence.count("expression") == _SDK_SEQUENCE_MAX
    assert sequence.endswith("(+70 more)")
    assert stats["sdk_function_counts"].to_list()[0] == f"expression:{_SDK_SEQUENCE_MAX + 70}"

    sessions, messages = _sessions_and_messages()
    metrics = compute_all_metrics(
        sessions, messages, build_session_tool_stats(messages), {}, sdk_stats=stats
    )
    report = generate_report(
        metrics, sessions, messages, build_session_tool_stats(messages), sdk_stats=stats
    )
    assert f"| expression | {_SDK_SEQUENCE_MAX + 70} |" in report
