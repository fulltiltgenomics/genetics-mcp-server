"""Tests for the paired A/B replay benchmark harness.

Everything that touches the network runs against a real local SSE server (uvicorn on
an ephemeral port, using the same sse-starlette EventSourceResponse production uses),
so the harness's stream parsing is exercised over real chunked HTTP rather than a mock.
"""

import asyncio
import json
import socket
from pathlib import Path

import pytest
import uvicorn
from fastapi import FastAPI, Request
from sse_starlette.sse import EventSourceResponse

from genetics_mcp_server.llm_service import _sanitize_tool_blocks
from genetics_mcp_server.scripts.replay_benchmark import (
    ALL_TOOLS_ARM,
    MIN_N_FOR_PERCENTILE,
    _iteration_timing_lines,
    _script_lines,
    build_parser,
    count_tool_calls,
    distribution,
    format_summary,
    load_cases,
    percentile,
    run_benchmark,
)


class StubChatServer:
    """A programmable stand-in for /chat/v1/chat.

    `plan` maps a tool_profile value (None for all-tools) to a list of SSE payload
    dicts; the entry may instead be a callable taking the request body so a test can
    vary the reply per turn.
    """

    def __init__(self):
        self.plan: dict[object, object] = {}
        self.requests: list[dict] = []
        self.headers: list[dict] = []
        self.delay = 0.0
        self.in_flight = 0
        self.max_in_flight = 0
        self.app = FastAPI()

        @self.app.post("/chat/v1/chat")
        async def chat(request: Request):
            body = await request.json()
            self.requests.append(body)
            self.headers.append(dict(request.headers))
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            try:
                if self.delay:
                    await asyncio.sleep(self.delay)
            finally:
                self.in_flight -= 1
            entry = self.plan.get(body.get("tool_profile"), [])
            payloads = entry(body) if callable(entry) else entry

            async def gen():
                for payload in payloads:
                    yield {
                        "event": "error" if payload.get("type") == "error" else "message",
                        "data": json.dumps(payload),
                    }

            return EventSourceResponse(gen())


def _usage(iteration, input_tokens, output_tokens, total_in, total_out):
    return {
        "type": "usage",
        "iteration": iteration,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "context_window": 1_000_000,
        "context_percent": round(input_tokens / 1_000_000 * 100, 1),
    }


def _done(blocks=None, tool_results=None):
    return {
        "type": "done",
        "message_content": blocks or [{"type": "text", "text": "ok"}],
        "tool_results": tool_results,
    }


def _tool_turn(tool_use_id="t1"):
    """A done chunk shaped like a real tool-using turn: tool_use block + its result."""
    return _done(
        blocks=[
            {"type": "tool_use", "id": tool_use_id, "name": "get_variants", "input": {"q": "x"}},
            {"type": "text", "text": "answer"},
        ],
        tool_results=[
            {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": '{"results": ["a lot of tokens"]}',
            }
        ],
    )


@pytest.fixture
async def stub_server():
    stub = StubChatServer()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    config = uvicorn.Config(stub.app, log_level="error", lifespan="off")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve(sockets=[sock]))
    for _ in range(200):
        if server.started:
            break
        await asyncio.sleep(0.02)
    assert server.started, "stub SSE server did not start"
    stub.base_url = f"http://127.0.0.1:{port}"
    try:
        yield stub
    finally:
        server.should_exit = True
        await task


def write_dataset(tmp_path: Path, cases) -> Path:
    path = tmp_path / "eval_dataset.json"
    path.write_text(json.dumps(cases))
    return path


def make_case(session_id, turns=1, options=None):
    return {
        "session_id": session_id,
        "topic": "variants",
        "user_turns": [
            {
                "index": i,
                "message_id": f"{session_id}-m{i}",
                "content": f"question {i}",
                "options": options
                or {
                    "verbosity": "brief",
                    "tool_profile": "api",
                    "instruction_set_id": None,
                    "literature_backend": "europepmc",
                },
            }
            for i in range(turns)
        ],
    }


# ---------------------------------------------------------------- pure functions


def test_percentile_interpolates_like_numpy():
    values = [1.0, 2.0, 3.0, 4.0]
    assert percentile(values, 50) == 2.5
    assert percentile(values, 0) == 1.0
    assert percentile(values, 100) == 4.0
    assert percentile(values, 25) == pytest.approx(1.75)


def test_percentile_single_observation():
    assert percentile([7.0], 95) == 7.0


def test_percentile_rejects_empty_sample():
    with pytest.raises(ValueError):
        percentile([], 50)


def test_distribution_flags_percentiles_the_sample_cannot_support():
    d = distribution([1, 2, 3, 4, 5])
    assert d["n"] == 5
    assert d["p50"] == 3.0
    # n=5 has no observation above p90 or p95, so both are just the maximum
    assert set(d["unreliable_percentiles"]) == {"p90", "p95"}
    assert "p25" not in d["unreliable_percentiles"]

    big = distribution(list(range(100)))
    assert big["unreliable_percentiles"] == []


def test_distribution_ignores_none_and_reports_empty_sample():
    assert distribution([None, None])["n"] == 0
    d = distribution([None, 5, None, 15])
    assert d["n"] == 2 and d["mean"] == 10.0


def test_count_tool_calls_uses_blocks_not_display_markers():
    blocks = [
        {"type": "text", "text": "*[Using tool: fake_tool; a: 1]*"},
        {"type": "tool_use", "id": "t1", "name": "real_tool", "input": {}},
        {"type": "tool_use", "id": "t2", "name": "real_tool", "input": {}},
    ]
    assert count_tool_calls(blocks) == 2
    assert count_tool_calls(None) == 0
    assert count_tool_calls([]) == 0


def test_cli_defaults_to_localhost_not_production():
    args = build_parser().parse_args([])
    assert "localhost" in args.base_url
    assert args.arm_a == ALL_TOOLS_ARM
    assert args.model is None


def test_load_cases_is_deterministic_and_drops_unreplayable(tmp_path):
    dataset = write_dataset(
        tmp_path,
        [make_case("ccc"), {"session_id": "bbb", "user_turns": []}, make_case("aaa")],
    )
    assert [c["session_id"] for c in load_cases(dataset, None)] == ["aaa", "ccc"]
    assert [c["session_id"] for c in load_cases(dataset, 1)] == ["aaa"]


# ------------------------------------------------------------------- end to end


async def test_usage_chunk_parsing_and_cost_bounds(stub_server, tmp_path):
    stub_server.plan = {
        None: [
            _usage(1, 40_000, 500, 10_000, 500),
            _usage(2, 60_000, 300, 12_000, 800),
            _done(
                [
                    {"type": "tool_use", "id": "t1", "name": "x", "input": {}},
                    {"type": "text", "text": "answer"},
                ]
            ),
        ],
        "bigquery": [_usage(1, 1_000, 10, 1_000, 10), _done()],
    }
    dataset = write_dataset(tmp_path, [make_case("s1")])

    report = await run_benchmark(
        dataset=dataset,
        base_url=stub_server.base_url,
        arms=(ALL_TOOLS_ARM, "bigquery"),
        limit=None,
        concurrency=1,
        model="claude-opus-4-5",
        timeout=30.0,
        max_turns=None,
        auth_token=None,
    )

    turn = next(t for t in report["turns"] if t["arm"] == ALL_TOOLS_ARM)
    assert turn["status"] == "ok"
    assert turn["iterations"] == 2
    assert turn["tool_calls"] == 1
    assert turn["input_tokens"] == 12_000
    assert turn["output_tokens"] == 800
    # per-iteration input_tokens is the whole context; the remainder after the billed
    # uncached input is cache_read + cache_creation, which the stream cannot separate
    assert turn["cached_input_tokens"] == 40_000 + 60_000 - 12_000
    assert turn["context_tokens_first_iteration"] == 40_000
    assert turn["context_tokens_last_iteration"] == 60_000
    assert turn["context_percent_last_iteration"] == 6.0
    assert turn["ms_to_done"] is not None
    assert turn["hit_max_iterations"] is False
    assert len(turn["iterations_detail"]) == 2

    # cost is an interval, never a single fabricated number
    assert 0 < turn["cost_usd_min"] < turn["cost_usd_max"]


async def test_cost_is_not_priced_without_a_model(stub_server, tmp_path):
    stub_server.plan = {None: [_usage(1, 100, 10, 100, 10), _done()], "bigquery": [_done()]}
    dataset = write_dataset(tmp_path, [make_case("s1")])

    report = await run_benchmark(
        dataset=dataset,
        base_url=stub_server.base_url,
        arms=(ALL_TOOLS_ARM, "bigquery"),
        limit=None,
        concurrency=1,
        model=None,
        timeout=30.0,
        max_turns=None,
        auth_token=None,
    )
    turn = next(t for t in report["turns"] if t["arm"] == ALL_TOOLS_ARM)
    assert turn["cost_usd_min"] is None and turn["cost_usd_max"] is None
    assert report["per_arm"][ALL_TOOLS_ARM]["distributions"]["cost_usd_min"]["n"] == 0


async def test_max_iterations_turn_is_flagged(stub_server, tmp_path):
    stub_server.plan = {
        None: [
            _usage(25, 100, 10, 100, 10),
            _done([{"type": "text", "text": "\n\n*[Max tool iterations reached]*\n"}]),
        ],
        "bigquery": [_usage(1, 100, 10, 100, 10), _done()],
    }
    dataset = write_dataset(tmp_path, [make_case("s1")])
    report = await run_benchmark(
        dataset=dataset,
        base_url=stub_server.base_url,
        arms=(ALL_TOOLS_ARM, "bigquery"),
        limit=None,
        concurrency=1,
        model=None,
        timeout=30.0,
        max_turns=None,
        auth_token=None,
    )
    assert report["per_arm"][ALL_TOOLS_ARM]["turns_hitting_max_iterations"] == 1
    assert report["per_arm"]["bigquery"]["turns_hitting_max_iterations"] == 0


async def test_request_shape_overrides_arm_and_always_sends_secret(stub_server, tmp_path):
    stub_server.plan = {None: [_done()], "bigquery": [_done()]}
    dataset = write_dataset(tmp_path, [make_case("s1")])

    await run_benchmark(
        dataset=dataset,
        base_url=stub_server.base_url,
        arms=(ALL_TOOLS_ARM, "bigquery"),
        limit=None,
        concurrency=1,
        model="claude-opus-4-5",
        timeout=30.0,
        max_turns=None,
        auth_token=None,
    )

    assert len(stub_server.requests) == 2
    for body in stub_server.requests:
        assert body["secret"] is True
        # the three non-arm options are replayed exactly as recorded
        assert body["verbosity"] == "brief"
        assert body["literature_backend"] == "europepmc"
        assert body["instruction_set_id"] is None
    # the recorded tool_profile ("api") is deliberately discarded; the arm wins
    assert {b["tool_profile"] for b in stub_server.requests} == {None, "bigquery"}


async def test_paired_execution_keeps_arms_adjacent_and_alternates_order(
    stub_server, tmp_path
):
    ok_turn = [_usage(1, 100, 10, 100, 10), _done()]
    stub_server.plan = {None: list(ok_turn), "bigquery": list(ok_turn)}
    cases = [make_case(sid) for sid in ("aaa", "bbb", "ccc", "ddd")]
    dataset = write_dataset(tmp_path, cases)

    report = await run_benchmark(
        dataset=dataset,
        base_url=stub_server.base_url,
        arms=(ALL_TOOLS_ARM, "bigquery"),
        limit=None,
        concurrency=1,
        model=None,
        timeout=30.0,
        max_turns=None,
        auth_token=None,
    )

    order = report["arm_order_per_case"]
    assert order["aaa"] == [ALL_TOOLS_ARM, "bigquery"]
    assert order["bbb"] == ["bigquery", ALL_TOOLS_ARM]
    assert order["ccc"] == [ALL_TOOLS_ARM, "bigquery"]
    assert order["ddd"] == ["bigquery", ALL_TOOLS_ARM]

    # every case contributes exactly one turn to each arm
    for arm in (ALL_TOOLS_ARM, "bigquery"):
        assert report["per_arm"][arm]["turns_ok"] == 4

    # ...and on the wire the two arms of a case are adjacent, so drift hits both equally
    profiles = [b["tool_profile"] for b in stub_server.requests]
    assert profiles == [None, "bigquery", "bigquery", None, None, "bigquery", "bigquery", None]


async def test_concurrency_parallelises_cases_never_the_arms_within_one(
    stub_server, tmp_path
):
    """Concurrency is over cases only; a case's two arms are always sequential.

    Global wire adjacency cannot hold once cases overlap, so the invariant checked here
    is the one that actually protects the pairing: no more than `concurrency` requests
    are ever in flight, which is only possible if the two arms of a case never overlap.
    """
    stub_server.plan = {None: [_done()], "bigquery": [_done()]}
    stub_server.delay = 0.05
    cases = [make_case(f"s{i:02d}") for i in range(6)]
    dataset = write_dataset(tmp_path, cases)

    report = await run_benchmark(
        dataset=dataset,
        base_url=stub_server.base_url,
        arms=(ALL_TOOLS_ARM, "bigquery"),
        limit=None,
        concurrency=3,
        model=None,
        timeout=30.0,
        max_turns=None,
        auth_token=None,
    )

    assert stub_server.max_in_flight <= 3
    assert stub_server.max_in_flight > 1, "the run never actually overlapped; test is vacuous"

    # each case issued exactly its two arms, in the order the report recorded
    by_case: dict[str, list[object]] = {}
    for body in stub_server.requests:
        # session ids are "replay-<run>-<case>-<arm>"; the case segment pairs the two arms
        by_case.setdefault(body["session_id"].rsplit("-", 1)[0], []).append(
            body["tool_profile"]
        )
    assert len(by_case) == 6
    for case_id, order in report["arm_order_per_case"].items():
        wire = next(v for k, v in by_case.items() if k.endswith(case_id[:8]))
        assert wire == [None if a == ALL_TOOLS_ARM else a for a in order]


async def test_erroring_turn_is_counted_and_does_not_abort_the_run(stub_server, tmp_path):
    def failing(body):
        if body["messages"][-1]["content"] == "question 1":
            return [{"type": "error", "error": "upstream exploded"}]
        return [_usage(1, 100, 10, 100, 10), _done()]

    stub_server.plan = {
        None: failing,
        "bigquery": [_usage(1, 100, 10, 100, 10), _done()],
    }
    dataset = write_dataset(tmp_path, [make_case("s1", turns=3), make_case("s2", turns=1)])

    report = await run_benchmark(
        dataset=dataset,
        base_url=stub_server.base_url,
        arms=(ALL_TOOLS_ARM, "bigquery"),
        limit=None,
        concurrency=1,
        model=None,
        timeout=30.0,
        max_turns=None,
        auth_token=None,
    )

    arm = report["per_arm"][ALL_TOOLS_ARM]
    assert arm["turns_by_status"]["error"] == 1
    # the remaining turns of that case/arm are recorded as not_attempted, never dropped
    assert arm["turns_by_status"]["not_attempted"] == 1
    assert arm["turns_by_status"]["ok"] == 2  # s1 turn 0, plus s2
    # the other arm is unaffected and the run completed
    assert report["per_arm"]["bigquery"]["turns_ok"] == 4
    # every turn of every case is accounted for on both arms
    assert len(report["turns"]) == 8

    failed = next(t for t in report["turns"] if t["status"] == "error")
    assert "upstream exploded" in failed["error"]


async def test_unreachable_target_is_recorded_not_raised(tmp_path):
    dataset = write_dataset(tmp_path, [make_case("s1")])
    # port 1 on loopback: connection refused, the harness must survive it
    report = await run_benchmark(
        dataset=dataset,
        base_url="http://127.0.0.1:1",
        arms=(ALL_TOOLS_ARM, "bigquery"),
        limit=None,
        concurrency=1,
        model=None,
        timeout=5.0,
        max_turns=None,
        auth_token=None,
    )
    assert {t["status"] for t in report["turns"]} == {"error"}
    assert all(t["error"] for t in report["turns"])


async def test_multi_turn_replay_accumulates_history(stub_server, tmp_path):
    stub_server.plan = {
        None: [_done([{"type": "text", "text": "reply"}])],
        "bigquery": [_done([{"type": "text", "text": "reply"}])],
    }
    dataset = write_dataset(tmp_path, [make_case("s1", turns=3)])

    await run_benchmark(
        dataset=dataset,
        base_url=stub_server.base_url,
        arms=(ALL_TOOLS_ARM, "bigquery"),
        limit=None,
        concurrency=1,
        model=None,
        timeout=30.0,
        max_turns=None,
        auth_token=None,
    )

    first_arm = [b for b in stub_server.requests if b["tool_profile"] is None]
    assert [len(b["messages"]) for b in first_arm] == [1, 3, 5]
    assert first_arm[-1]["messages"][1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "reply"}],
    }


async def test_script_metrics_are_unmeasured_not_zero(stub_server, tmp_path):
    stub_server.plan = {None: [_usage(1, 100, 10, 100, 10), _done()], "bigquery": [_done()]}
    dataset = write_dataset(tmp_path, [make_case("s1")])

    report = await run_benchmark(
        dataset=dataset,
        base_url=stub_server.base_url,
        arms=(ALL_TOOLS_ARM, "bigquery"),
        limit=None,
        concurrency=1,
        model=None,
        timeout=30.0,
        max_turns=None,
        auth_token=None,
    )

    arm = report["per_arm"][ALL_TOOLS_ARM]
    assert arm["script_failure_rate"] is None
    assert arm["script_failures"] is None
    assert arm["retry_loops"] is None
    assert "script_failures" in arm["unmeasured"]
    assert "NOT MEASURED" in format_summary(report)


async def test_script_metrics_populate_when_the_stream_emits_them(stub_server, tmp_path):
    """The chunk type does not exist yet; this pins the contract the sandbox arm must meet."""
    stub_server.plan = {
        None: [
            _usage(1, 100, 10, 100, 10),
            {"type": "script_result", "exit_code": 1, "timed_out": False},
            _usage(2, 200, 10, 200, 20),
            {"type": "script_result", "exit_code": 0, "timed_out": False},
            _usage(3, 300, 10, 300, 30),
            _done(),
        ],
        "bigquery": [
            _usage(1, 100, 10, 100, 10),
            {"type": "script_result", "exit_code": 0, "timed_out": False},
            _done(),
        ],
    }
    dataset = write_dataset(tmp_path, [make_case("s1")])

    report = await run_benchmark(
        dataset=dataset,
        base_url=stub_server.base_url,
        arms=(ALL_TOOLS_ARM, "bigquery"),
        limit=None,
        concurrency=1,
        model=None,
        timeout=30.0,
        max_turns=None,
        auth_token=None,
    )

    arm = report["per_arm"][ALL_TOOLS_ARM]
    assert arm["script_runs"] == 2
    assert arm["script_failures"] == 1
    assert arm["script_failure_rate"] == 0.5
    # the failed script was followed by another model roundtrip: one wasted iteration
    assert arm["retry_loops"] == 1
    assert "unmeasured" not in arm

    # an arm that ran a script and it succeeded reports a real 0, distinct from unmeasured
    other = report["per_arm"]["bigquery"]
    assert other["script_runs"] == 1
    assert other["script_failures"] == 0
    assert other["script_failure_rate"] == 0.0


async def test_summary_states_n_and_warns_on_a_small_sample(stub_server, tmp_path):
    stub_server.plan = {None: [_usage(1, 100, 10, 100, 10), _done()], "bigquery": [_done()]}
    dataset = write_dataset(tmp_path, [make_case("s1"), make_case("s2")])

    report = await run_benchmark(
        dataset=dataset,
        base_url=stub_server.base_url,
        arms=(ALL_TOOLS_ARM, "bigquery"),
        limit=None,
        concurrency=1,
        model=None,
        timeout=30.0,
        max_turns=None,
        auth_token=None,
    )

    text = format_summary(report)
    assert "cases (N)   : 2" in text
    assert "WARNING" in text
    assert "unreliable at n=" in text


async def test_auth_token_is_sent_when_configured(stub_server, tmp_path):
    stub_server.plan = {None: [_done()], "bigquery": [_done()]}
    dataset = write_dataset(tmp_path, [make_case("s1")])

    await run_benchmark(
        dataset=dataset,
        base_url=stub_server.base_url,
        arms=(ALL_TOOLS_ARM, "bigquery"),
        limit=None,
        concurrency=1,
        model=None,
        timeout=30.0,
        max_turns=None,
        auth_token="s3cret",
    )
    assert stub_server.headers
    assert all(h.get("authorization") == "Bearer s3cret" for h in stub_server.headers)


async def test_replayed_history_carries_tool_results_so_tool_use_survives_sanitising(
    stub_server, tmp_path
):
    """Turn 2's history must contain the tool results, exactly as the browser replays them.

    Without them llm_service's _sanitize_tool_blocks strips every tool_use block of the
    replayed assistant turn as orphaned, deleting the tool output that is the bulk of the
    context growth this harness exists to measure — and deleting more of it from whichever
    arm calls more tools, which compresses the measured gap toward zero.
    """
    plan = [_usage(1, 100, 10, 100, 10), _tool_turn("tu-42")]
    stub_server.plan = {None: list(plan), "bigquery": list(plan)}
    dataset = write_dataset(tmp_path, [make_case("s1", turns=2)])

    await run_benchmark(
        dataset=dataset,
        base_url=stub_server.base_url,
        arms=(ALL_TOOLS_ARM, "bigquery"),
        limit=None,
        concurrency=1,
        model=None,
        timeout=30.0,
        max_turns=None,
        auth_token=None,
    )

    second_turn = [b for b in stub_server.requests if b["tool_profile"] is None][1]
    messages = second_turn["messages"]
    # user, assistant(with tool_use), user(tool_result), user
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "user"]
    assert messages[2]["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": "tu-42",
            "content": '{"results": ["a lot of tokens"]}',
        }
    ]

    # and the pairing is the one the server accepts: the tool_use block survives
    sanitized = _sanitize_tool_blocks(messages)
    assert any(
        b.get("type") == "tool_use" and b.get("id") == "tu-42"
        for b in sanitized[1]["content"]
    ), "tool_use was stripped as orphaned; the replayed context is not production-shaped"
    assert sanitized[2]["content"] == messages[2]["content"]


async def test_matched_comparison_does_not_flatter_an_arm_that_fails_late(
    stub_server, tmp_path
):
    """An arm that dies on the expensive late turns must not win on its cheap survivors."""

    def cheap_then_explode(body):
        if body["messages"][-1]["content"] == "question 0":
            return [_usage(1, 1_000, 10, 1_000, 10), _done()]
        return [{"type": "error", "error": "died on the hard turn"}]

    def cheap_then_expensive(body):
        if body["messages"][-1]["content"] == "question 0":
            return [_usage(1, 1_000, 10, 1_000, 10), _done()]
        return [_usage(1, 90_000, 10, 90_000, 10), _usage(2, 95_000, 10, 95_000, 20), _done()]

    stub_server.plan = {None: cheap_then_explode, "bigquery": cheap_then_expensive}
    dataset = write_dataset(tmp_path, [make_case("s1", turns=3)])

    report = await run_benchmark(
        dataset=dataset,
        base_url=stub_server.base_url,
        arms=(ALL_TOOLS_ARM, "bigquery"),
        limit=None,
        concurrency=1,
        model=None,
        timeout=30.0,
        max_turns=None,
        auth_token=None,
    )

    # only turn 0 succeeded on both arms
    assert report["matched"]["pairs"] == 1
    assert report["matched"]["dropped_ok_turns_per_arm"]["bigquery"] == 2
    assert report["matched"]["dropped_ok_turns_per_arm"][ALL_TOOLS_ARM] == 0

    matched_failing = report["matched"]["per_arm"][ALL_TOOLS_ARM]["distributions"]
    matched_healthy = report["matched"]["per_arm"]["bigquery"]["distributions"]
    assert matched_failing["input_tokens"]["n"] == matched_healthy["input_tokens"]["n"] == 1
    # on the matched turn the arms are identical, so the failing arm shows no advantage
    assert matched_failing["input_tokens"]["p50"] == matched_healthy["input_tokens"]["p50"]

    # the unmatched marginals are exactly the trap: the failing arm looks 45x cheaper
    marginal_failing = report["per_arm"][ALL_TOOLS_ARM]["distributions"]["input_tokens"]
    marginal_healthy = report["per_arm"]["bigquery"]["distributions"]["input_tokens"]
    assert marginal_failing["p50"] < marginal_healthy["p50"]

    text = format_summary(report)
    assert "MATCHED PAIRS: 1" in text
    assert "the arms' ok-sets differ" in text
    assert "bigquery: 2" in text
    assert "UNMATCHED MARGINALS" in text


async def test_measured_zero_script_runs_is_not_printed_as_not_measured(
    stub_server, tmp_path
):
    """script_runs==0 with a script chunk seen is measured; only script_runs is None is not."""
    stub_server.plan = {
        None: [
            _usage(1, 100, 10, 100, 10),
            {"type": "script_result", "exit_code": 0, "timed_out": False},
            _done(),
        ],
        "bigquery": [_usage(1, 100, 10, 100, 10), _done()],
    }
    dataset = write_dataset(tmp_path, [make_case("s1")])

    report = await run_benchmark(
        dataset=dataset,
        base_url=stub_server.base_url,
        arms=(ALL_TOOLS_ARM, "bigquery"),
        limit=None,
        concurrency=1,
        model=None,
        timeout=30.0,
        max_turns=None,
        auth_token=None,
    )

    # a measured arm whose scripts all ran and all succeeded
    measured = report["per_arm"][ALL_TOOLS_ARM]
    assert measured["script_runs"] == 1 and measured["script_failure_rate"] == 0.0

    # the same arm restricted to zero runs must still read as measured, not NOT MEASURED
    zero_runs = dict(
        measured,
        script_runs=0,
        script_attempts=0,
        script_failures=0,
        script_failure_rate=None,
    )
    lines = "\n".join(_script_lines(zero_runs))
    assert "NOT MEASURED" not in lines
    assert "script attempts=0 (executed=0)" in lines
    assert "NOT MEASURED" in "\n".join(
        _script_lines(report["per_arm"]["bigquery"])
    ), "an arm with no script chunk at all must still read as unmeasured"


async def test_turn_without_usage_chunks_is_not_counted_as_a_zero_tool_call_turn(
    stub_server, tmp_path
):
    """The OpenAI path emits a done with no usage chunk and no tool_use blocks."""
    stub_server.plan = {
        None: [_done([{"type": "text", "text": "synthetic openai answer"}])],
        "bigquery": [_usage(1, 100, 10, 100, 10), _done()],
    }
    dataset = write_dataset(tmp_path, [make_case("s1", turns=2)])

    report = await run_benchmark(
        dataset=dataset,
        base_url=stub_server.base_url,
        arms=(ALL_TOOLS_ARM, "bigquery"),
        limit=None,
        concurrency=1,
        model=None,
        timeout=30.0,
        max_turns=None,
        auth_token=None,
        provider="anthropic",
    )

    unmeasurable = [t for t in report["turns"] if t["arm"] == ALL_TOOLS_ARM]
    assert {t["status"] for t in unmeasurable} == {"no_usage_chunks"}
    # no fake zero enters the tool-call distribution
    assert all(t["tool_calls"] is None for t in unmeasurable)
    arm = report["per_arm"][ALL_TOOLS_ARM]
    assert arm["distributions"]["tool_calls"]["n"] == 0
    assert arm["distributions"]["iterations"]["n"] == 0
    # the history is intact, so the rest of the case still ran
    assert arm["turns_by_status"]["no_usage_chunks"] == 2
    assert "not_attempted" not in arm["turns_by_status"]

    # the provider is pinned on the wire and recorded in the config
    assert all(b["provider"] == "anthropic" for b in stub_server.requests)
    assert report["config"]["provider"] == "anthropic"


def test_provider_is_omitted_when_not_pinned():
    assert build_parser().parse_args([]).provider is None


async def test_whole_case_exception_records_every_turn_of_both_arms(
    stub_server, tmp_path, monkeypatch
):
    """A case that blows up must not collapse 2*N turns into two turn_index=0 records."""
    import genetics_mcp_server.scripts.replay_benchmark as rb

    async def boom(**kwargs):
        raise RuntimeError("worker exploded")

    monkeypatch.setattr(rb, "replay_case", boom)
    # case index 1 gets the reversed arm order; the failure path must record that order
    dataset = write_dataset(
        tmp_path, [make_case("aaa", turns=2), make_case("bbb", turns=4)]
    )

    report = await run_benchmark(
        dataset=dataset,
        base_url=stub_server.base_url,
        arms=(ALL_TOOLS_ARM, "bigquery"),
        limit=None,
        concurrency=1,
        model=None,
        timeout=30.0,
        max_turns=None,
        auth_token=None,
    )

    assert len(report["turns"]) == (2 + 4) * 2
    for arm in (ALL_TOOLS_ARM, "bigquery"):
        s = report["per_arm"][arm]
        assert s["turns_attempted"] == 6
        assert s["turns_by_status"]["error"] == 6
    bbb = [t for t in report["turns"] if t["case_id"] == "bbb" and t["arm"] == ALL_TOOLS_ARM]
    assert sorted(t["turn_index"] for t in bbb) == [0, 1, 2, 3]
    # the alternated order is preserved, not silently reset to (a, b)
    assert report["arm_order_per_case"]["bbb"] == ["bigquery", ALL_TOOLS_ARM]
    assert {t["arm_position"] for t in bbb} == {1}


def test_min_n_for_p90_is_ten_not_eleven():
    """1 - 90/100 is 0.09999999999999998 in binary floating point; 100/(100-90) is not."""
    assert MIN_N_FOR_PERCENTILE == {25: 2, 50: 2, 75: 4, 90: 10, 95: 20}
    assert distribution(list(range(10)))["unreliable_percentiles"] == ["p95"]


async def test_unpriceable_model_reports_no_cost_rather_than_sonnet_prices(
    stub_server, tmp_path
):
    stub_server.plan = {
        None: [_usage(1, 10_000, 500, 10_000, 500), _done()],
        "bigquery": [_usage(1, 10_000, 500, 10_000, 500), _done()],
    }
    dataset = write_dataset(tmp_path, [make_case("s1")])

    report = await run_benchmark(
        dataset=dataset,
        base_url=stub_server.base_url,
        arms=(ALL_TOOLS_ARM, "bigquery"),
        limit=None,
        concurrency=1,
        model="gpt-4o",
        timeout=30.0,
        max_turns=None,
        auth_token=None,
    )
    assert all(t["cost_usd_min"] is None and t["cost_usd_max"] is None for t in report["turns"])
    assert report["per_arm"][ALL_TOOLS_ARM]["distributions"]["cost_usd_min"]["n"] == 0


# ------------------------- exact cost, per-iteration timing and real script_result chunks


def _usage_v2(
    iteration,
    input_tokens,
    output_tokens,
    total_in,
    total_out,
    *,
    cache_read,
    cache_create,
    turn_elapsed_ms,
    model_ms,
):
    """A usage chunk as the current llm_service emits it: cache split plus timing."""
    return {
        **_usage(iteration, input_tokens, output_tokens, total_in, total_out),
        "cache_read": cache_read,
        "cache_create": cache_create,
        "turn_elapsed_ms": turn_elapsed_ms,
        "model_ms": model_ms,
    }


# two iterations whose uncached input (context - cache_read - cache_create) is exactly the
# increment of total_input_tokens, which is the invariant llm_service's comment states
_SPLIT_TURN = [
    _usage_v2(
        1, 40_000, 500, 10_000, 500,
        cache_read=25_000, cache_create=5_000, turn_elapsed_ms=1_200, model_ms=1_000,
    ),
    _usage_v2(
        2, 60_000, 300, 22_000, 800,
        cache_read=45_000, cache_create=3_000, turn_elapsed_ms=9_000, model_ms=2_000,
    ),
]


async def _one_case(stub_server, tmp_path, plan, model="claude-opus-4-5"):
    stub_server.plan = plan
    return await run_benchmark(
        dataset=write_dataset(tmp_path, [make_case("s1")]),
        base_url=stub_server.base_url,
        arms=(ALL_TOOLS_ARM, "bigquery"),
        limit=None,
        concurrency=1,
        model=model,
        timeout=30.0,
        max_turns=None,
        auth_token=None,
    )


async def test_cache_split_prices_the_turn_exactly(stub_server, tmp_path):
    report = await _one_case(
        stub_server,
        tmp_path,
        {
            None: [*_SPLIT_TURN, _done()],
            "bigquery": [_usage(1, 1_000, 10, 1_000, 10), _done()],
        },
    )
    turn = next(t for t in report["turns"] if t["arm"] == ALL_TOOLS_ARM)

    assert turn["cache_read_tokens"] == 70_000
    assert turn["cache_create_tokens"] == 8_000
    # the split must reconcile with the figure derived by subtraction, or one of the two
    # is measuring something other than what its name says
    assert turn["cached_input_tokens"] == 70_000 + 8_000

    # opus pricing per Mtok: 5 in / 25 out / 0.5 cache read / 6.25 cache creation
    expected = (22_000 * 5 + 800 * 25 + 70_000 * 0.5 + 8_000 * 6.25) / 1e6
    assert turn["cost_usd"] == pytest.approx(expected)
    assert turn["cost_basis"] == "exact"
    # the exact figure sits inside the bracket the harness used to report on its own
    assert turn["cost_usd_min"] < turn["cost_usd"] < turn["cost_usd_max"]
    # ...and the bracket really is wide enough that the difference matters
    assert turn["cost_usd_max"] > 3 * turn["cost_usd_min"]

    arm = report["per_arm"][ALL_TOOLS_ARM]
    assert arm["cost_basis"] == {"exact": 1, "interval": 0, "unpriced": 0}
    assert arm["distributions"]["cost_usd"]["n"] == 1


async def test_a_stream_without_the_cache_split_falls_back_to_the_interval_and_says_so(
    stub_server, tmp_path
):
    """An older server emits no cache_read/cache_create. Silence would price it as if
    nothing were cached, which is the confidently-wrong number the interval exists to avoid.
    """
    report = await _one_case(
        stub_server,
        tmp_path,
        {
            None: [_usage(1, 40_000, 500, 10_000, 500), _done()],
            "bigquery": [*_SPLIT_TURN, _done()],
        },
    )
    interval_turn = next(t for t in report["turns"] if t["arm"] == ALL_TOOLS_ARM)
    assert interval_turn["cost_usd"] is None
    assert interval_turn["cost_basis"] == "interval"
    assert interval_turn["cache_read_tokens"] is None
    assert interval_turn["cost_usd_min"] < interval_turn["cost_usd_max"]

    arm = report["per_arm"][ALL_TOOLS_ARM]
    assert arm["cost_basis"] == {"exact": 0, "interval": 1, "unpriced": 0}
    # an interval turn contributes nothing to the exact distribution rather than a wrong 0
    assert arm["distributions"]["cost_usd"]["n"] == 0

    text = format_summary(report)
    assert "cost_usd is UNAVAILABLE for 1 turn(s)" in text
    assert "cost_usd is EXACT for 1 turn(s)" in text


async def test_one_iteration_carrying_no_split_demotes_the_whole_turn(
    stub_server, tmp_path
):
    """Pricing part of a turn exactly and the rest by assumption is not an exact turn."""
    report = await _one_case(
        stub_server,
        tmp_path,
        {
            None: [_SPLIT_TURN[0], _usage(2, 60_000, 300, 22_000, 800), _done()],
            "bigquery": [_usage(1, 1_000, 10, 1_000, 10), _done()],
        },
    )
    turn = next(t for t in report["turns"] if t["arm"] == ALL_TOOLS_ARM)
    assert turn["cost_basis"] == "interval"
    assert turn["cost_usd"] is None


async def test_per_iteration_timeline_separates_model_time_from_the_tool_phase(
    stub_server, tmp_path
):
    """turn_elapsed_ms is cumulative, so the SEGMENT between chunks is its difference, and
    the part of each difference that was not the model call is the PRECEDING tool phase.

    `iteration_ms` must therefore not be that difference. Ground truth here: iteration 1's
    model call took 1000ms and its tools 5800ms, iteration 2's model call 2000ms and it
    answered. Iteration 2's own work is 2000ms; the 5800 belongs to iteration 1 and is
    already printed on iteration 1's row. Deriving iteration_ms from the segment named
    iteration 2 as the slowest at 7800 — the localisation pointed one roundtrip too late.
    """
    report = await _one_case(
        stub_server,
        tmp_path,
        {
            None: [*_SPLIT_TURN, _done()],
            "bigquery": [_usage(1, 1_000, 10, 1_000, 10), _done()],
        },
    )
    turn = next(t for t in report["turns"] if t["arm"] == ALL_TOOLS_ARM)
    first, second = turn["iterations_detail"]

    assert first["turn_elapsed_ms"] == 1_200 and second["turn_elapsed_ms"] == 9_000
    assert first["segment_ms"] == 1_200
    assert second["segment_ms"] == 7_800  # 9000 - 1200, NOT 9000
    assert first["pre_model_ms"] == 200  # turn setup before the first model call
    assert second["pre_model_ms"] == 5_800  # 7800 - 2000: iteration 1's tools
    # the same 5800 attributed to the iteration whose tool calls it was
    assert first["tool_phase_ms"] == 5_800
    assert second["tool_phase_ms"] is None  # the last iteration answered; it ran no tools

    # the row sums: model + that iteration's OWN tools
    assert first["iteration_ms"] == 1_000 + 5_800
    # this turn ended because the model stopped calling tools, so iteration 2's tool phase
    # is a measured ZERO rather than an unknown, and its roundtrip is its model call
    assert second["iteration_ms"] == 2_000

    assert turn["model_ms_total"] == 3_000
    assert turn["slowest_iteration"] == 1
    assert turn["slowest_iteration_ms"] == 6_800

    timing = report["per_arm"][ALL_TOOLS_ARM]["iteration_timing"]
    assert timing["iterations"] == 2
    assert [row["iteration"] for row in timing["by_iteration_index"]] == [1, 2]
    assert timing["by_iteration_index"][0]["model_ms"]["p50"] == 1_000
    # ...but the tools COLUMN still records no observation: a 0 there would drag every
    # by-index median toward zero for a reason unrelated to how long tools take
    assert timing["by_iteration_index"][1]["tool_phase_ms"]["n"] == 0
    assert timing["by_iteration_index"][1]["iteration_ms"]["n"] == 1
    assert "per-iteration timeline" in format_summary(report)


def test_the_harness_and_llm_service_agree_on_the_max_iterations_marker():
    """The marker decides whether the final iteration's tool phase is 0 or unmeasured.

    The harness keeps its own literal because it parses a remote server's stream, so a
    rename in llm_service would not fail an import — it would silently start imputing 0 to
    turns whose tools really did run past the last usage chunk. This is the pin that makes
    that a red test instead.
    """
    from genetics_mcp_server.llm_service import MAX_ITERATIONS_NOTICE
    from genetics_mcp_server.scripts.replay_benchmark import MAX_ITERATIONS_MARKER

    assert MAX_ITERATIONS_MARKER in MAX_ITERATIONS_NOTICE


async def test_a_turn_stopped_at_the_iteration_ceiling_keeps_an_unmeasured_final_phase(
    stub_server, tmp_path
):
    """The other half of the final-iteration discrimination.

    Here tools DID run on the last iteration and no following model call ever closed the
    span, so 2000 would be a fabricated roundtrip. The same two usage chunks as the test
    above; only the marker differs, which is exactly the signal being relied on.
    """
    report = await _one_case(
        stub_server,
        tmp_path,
        {
            None: [
                *_SPLIT_TURN,
                _done([{"type": "text", "text": "\n\n*[Max tool iterations reached]*\n"}]),
            ],
            "bigquery": [_usage(1, 1_000, 10, 1_000, 10), _done()],
        },
    )
    turn = next(t for t in report["turns"] if t["arm"] == ALL_TOOLS_ARM)
    assert turn["hit_max_iterations"] is True
    assert turn["iterations_detail"][1]["iteration_ms"] is None
    # iteration 1 is unaffected: its tool phase was measured by iteration 2's usage chunk
    assert turn["iterations_detail"][0]["iteration_ms"] == 6_800
    assert turn["slowest_iteration"] == 1


async def test_a_single_iteration_turn_still_reports_a_roundtrip(stub_server, tmp_path):
    """~36% of production turns are single-iteration. Reporting None for all of them would
    drop them out of slowest_iteration_ms entirely and make a slow final roundtrip — the one
    answering against the largest context — structurally invisible.
    """
    report = await _one_case(
        stub_server,
        tmp_path,
        {
            None: [
                _usage_v2(
                    1, 40_000, 500, 10_000, 500,
                    cache_read=0, cache_create=0, turn_elapsed_ms=3_100, model_ms=3_000,
                ),
                _done(),
            ],
            "bigquery": [
                _usage_v2(
                    1, 1_000, 10, 1_000, 10,
                    cache_read=0, cache_create=0, turn_elapsed_ms=900, model_ms=800,
                ),
                _done(),
            ],
        },
    )
    turn = next(t for t in report["turns"] if t["arm"] == ALL_TOOLS_ARM)
    assert turn["iterations_detail"][0]["iteration_ms"] == 3_000
    assert turn["slowest_iteration"] == 1 and turn["slowest_iteration_ms"] == 3_000
    assert report["per_arm"][ALL_TOOLS_ARM]["distributions"]["slowest_iteration_ms"]["n"] == 1


async def test_the_timeline_prints_each_column_its_own_n_and_flags_thin_percentiles(
    stub_server, tmp_path
):
    """One n per row would quote the tools p90 as if it rested on the row's iteration count.

    At index 1 there are 2 model observations and 1 tool-phase observation, and a p90 over
    n=1 IS the maximum. distribution() already computes unreliable_percentiles; the timeline
    is the only table in the report that used to drop it on the floor.
    """
    report = await _one_case(
        stub_server,
        tmp_path,
        {
            None: [*_SPLIT_TURN, _done()],
            "bigquery": [
                _usage_v2(
                    1, 1_000, 10, 1_000, 10,
                    cache_read=0, cache_create=0, turn_elapsed_ms=500, model_ms=400,
                ),
                _done(),
            ],
        },
    )
    arm = report["per_arm"][ALL_TOOLS_ARM]
    index_1 = arm["iteration_timing"]["by_iteration_index"][0]
    assert index_1["model_ms"]["n"] == 1 and index_1["tool_phase_ms"]["n"] == 1

    lines = _iteration_timing_lines(arm)
    header = lines[1]
    assert "model n" in header and "tools n" in header and "iter n" in header
    # the flag is present because p50/p90 over n=1 cannot be told from the max
    assert any("*" in line for line in lines)
    assert "not distinguishable from the column's maximum" in "\n".join(lines)
    assert "model_attempts" in "\n".join(lines)


async def test_an_untimed_chunk_in_the_middle_does_not_merge_two_iterations(
    stub_server, tmp_path
):
    """A gap must break the timeline, not be papered over.

    Advancing the baseline only on timed chunks would make iteration 3's segment span
    iterations 2 AND 3 (9000 - 1000 = 8000) and hand slowest_iteration the wrong index. The
    cost path demotes a whole turn when one chunk lacks the cache split; the timeline has to
    apply the same standard.
    """
    middle_untimed = [
        _usage_v2(
            1, 10_000, 100, 1_000, 100,
            cache_read=0, cache_create=0, turn_elapsed_ms=1_000, model_ms=800,
        ),
        _usage(2, 20_000, 100, 2_000, 200),  # no turn_elapsed_ms, no model_ms
        _usage_v2(
            3, 30_000, 100, 3_000, 300,
            cache_read=0, cache_create=0, turn_elapsed_ms=9_000, model_ms=500,
        ),
    ]
    report = await _one_case(
        stub_server,
        tmp_path,
        {
            None: [*middle_untimed, _done()],
            "bigquery": [_usage(1, 1_000, 10, 1_000, 10), _done()],
        },
    )
    turn = next(t for t in report["turns"] if t["arm"] == ALL_TOOLS_ARM)
    rows = turn["iterations_detail"]

    assert rows[0]["segment_ms"] == 1_000
    assert rows[1]["segment_ms"] is None  # the chunk itself carried no clock
    assert rows[2]["segment_ms"] is None, "8000 here would span iterations 2 and 3"
    assert rows[2]["pre_model_ms"] is None
    # ...so neither iteration on the far side of the gap can claim a roundtrip
    assert rows[0]["iteration_ms"] is None  # its tool phase would have to come from row 1
    assert rows[1]["iteration_ms"] is None  # the chunk carried no model_ms either
    # iteration 3 is still measurable on its own terms: it ended the turn without tools, so
    # its tool phase is a measured zero and its model call is its whole roundtrip. The gap
    # cost the SEGMENT, not this.
    assert rows[2]["iteration_ms"] == 500
    assert turn["slowest_iteration"] == 3 and turn["slowest_iteration_ms"] == 500


async def test_an_untimed_usage_chunk_leaves_the_timeline_empty_not_zero(
    stub_server, tmp_path
):
    report = await _one_case(
        stub_server,
        tmp_path,
        {
            None: [_usage(1, 100, 10, 100, 10), _done()],
            "bigquery": [_usage(1, 100, 10, 100, 10), _done()],
        },
    )
    turn = next(t for t in report["turns"] if t["arm"] == ALL_TOOLS_ARM)
    assert turn["iterations_detail"][0]["iteration_ms"] is None
    assert turn["model_ms_total"] is None
    assert report["per_arm"][ALL_TOOLS_ARM]["iteration_timing"]["distributions"][
        "iteration_ms"
    ]["n"] == 0


def _script(iteration, *, ok=True, ran=True, exception=None, status=None, timed_out=False):
    """A script_result chunk in the shape llm_service._script_result_payload emits."""
    return {
        "type": "script_result",
        "iteration": iteration,
        "ran": ran,
        "ok": ok,
        "status": status or ("ok" if ok else "error"),
        "timed_out": timed_out,
        "exception": exception,
        "limit": None,
        "duration_ms": 42,
    }


def test_the_harness_reads_the_chunk_the_chat_backend_actually_emits():
    """Pins the two ends of the wire together.

    The harness's predicates and llm_service's payload are in different repositories' worth
    of distance from each other; a fabricated chunk in a test proves only that the harness
    reads what the TEST writes. This drives the real producer.
    """
    from genetics_mcp_server.llm_service import _script_result_payload
    from genetics_mcp_server.scripts.replay_benchmark import (
        OUTCOME_EXECUTED_FAILED,
        OUTCOME_EXECUTED_OK,
        OUTCOME_INFRA,
        _script_ok,
        _script_outcome,
        _script_ran,
    )
    from genetics_mcp_server.tools import ToolExecutor

    render = ToolExecutor()._render_analysis
    ok = _script_result_payload(1, render({"status": "ok", "duration_ms": 5}))
    failed = _script_result_payload(
        1, render({"status": "error", "error": {"type": "ValueError", "message": "x"}})
    )
    infra = _script_result_payload(
        1, {"success": False, "error": "gone", "error_type": "SandboxUnavailable"}
    )

    assert _script_ran(ok) and _script_ok(ok)
    assert _script_ran(failed) and not _script_ok(failed)
    assert not _script_ran(infra) and not _script_ok(infra)

    assert _script_outcome(ok) == (OUTCOME_EXECUTED_OK, "ok")
    assert _script_outcome(failed) == (OUTCOME_EXECUTED_FAILED, "error")
    assert _script_outcome(infra) == (OUTCOME_INFRA, "SandboxUnavailable")


async def test_the_model_caused_non_run_shapes_the_real_executor_emits_are_not_infra():
    """Drives the REAL run_analysis for the two shapes a fabricated chunk cannot vouch for.

    Both reach the wire with `ran: False`. Classifying on that alone books the model
    emitting no code, and the model choosing timeout_s=300, as sandbox flakiness. The
    shapes are read out of the executor rather than written into the test, so a rename
    there fails here instead of quietly re-inflating the infra bucket.
    """
    import httpx

    from genetics_mcp_server.llm_service import _script_result_payload
    from genetics_mcp_server.sandbox_client import SandboxClient
    from genetics_mcp_server.scripts.replay_benchmark import (
        OUTCOME_MODEL_REJECTED,
        _script_outcome,
    )
    from genetics_mcp_server.tools import ToolExecutor

    executor = ToolExecutor()
    # a transport that would fail the test if it were ever reached: neither shape below is
    # allowed to leave this process, which is exactly why the sandbox cannot be blamed
    executor._sandbox = SandboxClient(
        "http://sandbox.invalid",
        transport=httpx.MockTransport(
            lambda request: pytest.fail("no request may be sent for a caller-side rejection")
        ),
    )

    blank = _script_result_payload(
        1, await executor.run_analysis(code="   ", user="u@finngen.fi", session_id="c1")
    )
    oversize_timeout = _script_result_payload(
        2,
        await executor.run_analysis(
            code="print(1)", user="u@finngen.fi", session_id="c1", timeout_s=300
        ),
    )

    assert blank["ran"] is False and oversize_timeout["ran"] is False
    # the blank script used to arrive as a bare "unknown", indistinguishable in the JSON
    # from a transport fault, so even a careful reader could not separate them
    assert blank["status"] == "EmptyScript"
    assert oversize_timeout["status"] == "SandboxRejected"
    assert _script_outcome(blank) == (OUTCOME_MODEL_REJECTED, "EmptyScript")
    assert _script_outcome(oversize_timeout) == (OUTCOME_MODEL_REJECTED, "SandboxRejected")


async def test_a_successful_script_is_not_counted_as_a_failure(stub_server, tmp_path):
    report = await _one_case(
        stub_server,
        tmp_path,
        {
            None: [
                _usage(1, 100, 10, 100, 10),
                _script(1, ok=True),
                _usage(2, 200, 10, 200, 20),
                _done(),
            ],
            "bigquery": [_usage(1, 100, 10, 100, 10), _done()],
        },
    )
    arm = report["per_arm"][ALL_TOOLS_ARM]
    assert arm["script_runs"] == 1
    assert arm["script_failures"] == 0
    assert arm["script_failure_rate"] == 0.0
    # a further roundtrip after a SUCCESSFUL script is the model reading the result, not a
    # retry; counting it would make every code-arm turn look like a retry loop
    assert arm["retry_loops"] == 0
    assert arm["script_infra_errors"] == 0


async def test_a_failed_then_retried_script_is_exactly_one_retry_loop(stub_server, tmp_path):
    """Two failures inside one iteration still cost one extra roundtrip, not two."""
    report = await _one_case(
        stub_server,
        tmp_path,
        {
            None: [
                _usage(1, 100, 10, 100, 10),
                _script(1, ok=False, exception="ValueError"),
                _script(1, ok=False, exception="KeyError"),
                _usage(2, 200, 10, 200, 20),
                _script(2, ok=True),
                _usage(3, 300, 10, 300, 30),
                _done(),
            ],
            "bigquery": [_usage(1, 100, 10, 100, 10), _done()],
        },
    )
    arm = report["per_arm"][ALL_TOOLS_ARM]
    assert arm["script_runs"] == 3
    assert arm["script_failures"] == 2
    assert arm["retry_loops"] == 1


async def test_a_failure_on_the_last_iteration_is_not_a_retry_loop(stub_server, tmp_path):
    report = await _one_case(
        stub_server,
        tmp_path,
        {
            None: [
                _usage(1, 100, 10, 100, 10),
                _script(1, ok=False, exception="ValueError"),
                _done(),
            ],
            "bigquery": [_usage(1, 100, 10, 100, 10), _done()],
        },
    )
    arm = report["per_arm"][ALL_TOOLS_ARM]
    assert arm["script_failures"] == 1
    assert arm["retry_loops"] == 0


async def test_a_sandbox_fault_is_not_charged_to_the_scripts_failure_rate(
    stub_server, tmp_path
):
    """A restarting sandbox says nothing about the model's script. Counting it would let a
    deploy landing mid-run decide the rollout.
    """
    report = await _one_case(
        stub_server,
        tmp_path,
        {
            None: [
                _usage(1, 100, 10, 100, 10),
                _script(1, ok=False, ran=False, status="SandboxUnavailable"),
                _usage(2, 200, 10, 200, 20),
                _script(2, ok=True),
                _usage(3, 300, 10, 300, 30),
                _done(),
            ],
            "bigquery": [_usage(1, 100, 10, 100, 10), _done()],
        },
    )
    arm = report["per_arm"][ALL_TOOLS_ARM]
    assert arm["script_runs"] == 1  # the fault never ran, so it is not in the denominator
    assert arm["script_attempts"] == 1
    assert arm["script_failures"] == 0
    assert arm["script_failure_rate"] == 0.0
    assert arm["script_infra_errors"] == 1
    # it still burned a roundtrip, which is a latency and cost fact even if not a script fact
    assert arm["retry_loops"] == 1
    # ...but it is not the same fact as a retry after the model's own script failed, so the
    # printed number says which it was
    assert arm["retry_loops_infra"] == 1 and arm["retry_loops_script"] == 0
    text = "\n".join(_script_lines(arm))
    assert "sandbox faults (not script failures, not in the rate)=1" in text
    assert "after a sandbox fault=1" in text
    assert "SandboxUnavailable=1" in text


async def test_a_model_caused_rejection_is_a_script_failure_not_a_sandbox_fault(
    stub_server, tmp_path
):
    """`ran: False` is not the same question as "whose fault".

    A model asking for timeout_s=300 gets SandboxRejected, and a blank `code` gets
    EmptyScript — neither reaches the sandbox, so both used to be booked as infrastructure.
    That reports failures=0 rate=0.000 beside a pile of "sandbox faults" and tells a reader
    the scripts never fail and the platform is flaky, which is the exact inversion of what
    happened, on the metric that exists to price the code arm's own risk.
    """
    report = await _one_case(
        stub_server,
        tmp_path,
        {
            None: [
                _usage(1, 100, 10, 100, 10),
                _script(1, ok=False, ran=False, status="SandboxRejected"),
                _usage(2, 200, 10, 200, 20),
                _script(2, ok=False, ran=False, status="EmptyScript"),
                _usage(3, 300, 10, 300, 30),
                _script(3, ok=True),
                _usage(4, 400, 10, 400, 40),
                _done(),
            ],
            "bigquery": [_usage(1, 100, 10, 100, 10), _done()],
        },
    )
    arm = report["per_arm"][ALL_TOOLS_ARM]
    assert arm["script_infra_errors"] == 0, "neither shape is the platform's doing"
    assert arm["script_runs"] == 1  # only the third reached the sandbox
    # both rejections are in the numerator AND the denominator, so the rate cannot exceed 1
    assert arm["script_attempts"] == 3
    assert arm["script_failures"] == 2
    assert arm["script_failure_rate"] == pytest.approx(2 / 3)
    assert arm["retry_loops_script"] == 2 and arm["retry_loops_infra"] == 0

    text = "\n".join(_script_lines(arm))
    assert "SandboxRejected=1" in text and "EmptyScript=1" in text and "ok=1" in text
    assert "(executed_failed + model_rejected)" in text


async def test_the_disputed_deadline_shape_is_bucketed_with_neither_side(
    stub_server, tmp_path
):
    """TurnBudgetExceeded is classified differently by different readers, so it is reported
    on its own line and left out of both the numerator and the denominator. Whichever way a
    reader classifies it, the arithmetic is available on the page.
    """
    report = await _one_case(
        stub_server,
        tmp_path,
        {
            None: [
                _usage(1, 100, 10, 100, 10),
                _script(1, ok=False, ran=False, status="TurnBudgetExceeded"),
                _usage(2, 200, 10, 200, 20),
                _script(2, ok=True),
                _usage(3, 300, 10, 300, 30),
                _done(),
            ],
            "bigquery": [_usage(1, 100, 10, 100, 10), _done()],
        },
    )
    arm = report["per_arm"][ALL_TOOLS_ARM]
    assert arm["script_budget_exceeded"] == 1
    assert arm["script_infra_errors"] == 0 and arm["script_failures"] == 0
    assert arm["script_attempts"] == 1 and arm["script_failure_rate"] == 0.0
    assert arm["script_outcomes"] == {"TurnBudgetExceeded": 1, "ok": 1}
    # the wasted roundtrip is attributed to it specifically, not merged into either side
    assert arm["retry_loops"] == 1 and arm["retry_loops_disputed"] == 1
    assert arm["retry_loops_script"] == 0 and arm["retry_loops_infra"] == 0

    text = "\n".join(_script_lines(arm))
    assert "TurnBudgetExceeded (classified by neither, not in the rate)=1" in text


async def test_a_call_is_attributed_to_its_iteration_over_the_real_wire(stub_server, tmp_path):
    """The done chunk cannot answer this and the stream's ORDERING is the only thing that can.

    `message_content` flattens every iteration's blocks into one list, so three calls made in
    one parallel roundtrip and three roundtrips of one call each are indistinguishable in it
    — and that is precisely the difference between an arm that is wide and one that is slow.
    llm_service emits each `tool_use` chunk after its iteration's `usage` chunk and before any
    of that iteration's tools run, so the iteration in force when the chunk arrives is the
    one the call belongs to.
    """
    stub_server.plan = {
        None: [
            _usage(1, 100, 10, 100, 10),
            {"type": "tool_use", "id": "a", "name": "search_genes", "input": {"query": "FOXP3"}},
            {"type": "tool_use", "id": "b", "name": "get_gene_to_peaks", "input": {"gene": "F"}},
            _usage(2, 200, 10, 200, 20),
            {"type": "tool_use", "id": "c", "name": "run_analysis", "input": {"code": "x"}},
            {
                "type": "script_result",
                "tool_use_id": "c",
                "iteration": 2,
                "ran": True,
                "ok": True,
                "status": "ok",
                "duration_ms": 1420,
            },
            _usage(3, 300, 10, 300, 30),
            _done(
                blocks=[
                    {"type": "tool_use", "id": "a", "name": "search_genes", "input": {"query": "FOXP3"}},
                    {"type": "tool_use", "id": "b", "name": "get_gene_to_peaks", "input": {"gene": "F"}},
                    {"type": "tool_use", "id": "c", "name": "run_analysis", "input": {"code": "x"}},
                    {"type": "text", "text": "answer"},
                ]
            ),
        ],
        "bigquery": [_usage(1, 100, 10, 100, 10), _done()],
    }
    dataset = write_dataset(tmp_path, [make_case("s1")])

    report = await run_benchmark(
        dataset=dataset,
        base_url=stub_server.base_url,
        arms=(ALL_TOOLS_ARM, "bigquery"),
        limit=None,
        concurrency=1,
        model=None,
        timeout=30.0,
        max_turns=None,
        auth_token=None,
    )

    calls = [t for t in report["turns"] if t["arm"] == ALL_TOOLS_ARM][0]["tool_calls_detail"]
    assert [c["iteration"] for c in calls] == [1, 1, 2], "two calls shared a roundtrip; one did not"
    # only run_analysis reports a duration on the wire, and it is the SANDBOX's own clock
    assert calls[2]["script_duration_ms"] == 1420 and calls[2]["script_status"] == "ok"
    assert "script_duration_ms" not in calls[0], "no other tool is timed; absent must stay absent"


async def test_the_arguments_come_from_the_done_chunk_not_the_streamed_copy(stub_server, tmp_path):
    """llm_service REWRITES the input it streams; the model's own is what explains a failure.

    It substitutes `search_scientific_literature`'s backend and strips `run_analysis`'s
    model-invented `user`/`session_id` before emitting the `tool_use` chunk, so the streamed
    copy is what the server ran. Taking arguments from there would silently answer a
    different question than "what did the model ask for" — which is the question a losing
    case needs answered.
    """
    stub_server.plan = {
        None: [
            _usage(1, 100, 10, 100, 10),
            {"type": "tool_use", "id": "z", "name": "run_analysis", "input": {"code": "SERVER"}},
            _usage(2, 200, 10, 200, 20),
            _done(
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "z",
                        "name": "run_analysis",
                        "input": {"code": "MODEL", "user": "forged"},
                    }
                ]
            ),
        ],
        "bigquery": [_usage(1, 100, 10, 100, 10), _done()],
    }
    dataset = write_dataset(tmp_path, [make_case("s1")])

    report = await run_benchmark(
        dataset=dataset,
        base_url=stub_server.base_url,
        arms=(ALL_TOOLS_ARM, "bigquery"),
        limit=None,
        concurrency=1,
        model=None,
        timeout=30.0,
        max_turns=None,
        auth_token=None,
    )

    call = [t for t in report["turns"] if t["arm"] == ALL_TOOLS_ARM][0]["tool_calls_detail"][0]
    assert call["input"] == {"code": "MODEL", "user": "forged"}
    assert call["iteration"] == 1, "the id still correlates the iteration"


def test_the_harness_reads_the_tool_use_chunk_the_chat_backend_actually_emits():
    """Pins the attribution to the real producer, as the script_result contract test does.

    A fabricated chunk in a test proves the harness reads what the TEST writes. chat_api
    builds this chunk by splatting llm_service's JSON over `{"type": "tool_use"}`, so the
    key the harness correlates on is `id` — if either end renames it, every call silently
    loses its iteration and the transcript quietly stops showing roundtrips.
    """
    import inspect

    from genetics_mcp_server import chat_api, llm_service

    producer = inspect.getsource(llm_service.LLMService)
    assert '"id": tool_use.id' in producer, "llm_service must still key the chunk on `id`"
    assert '{"type": "tool_use", **json.loads(chunk.content)}' in inspect.getsource(chat_api), (
        "chat_api must still forward the tool_use chunk's fields verbatim"
    )
