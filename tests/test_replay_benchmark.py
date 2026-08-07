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
    zero_runs = dict(measured, script_runs=0, script_failures=0, script_failure_rate=None)
    lines = "\n".join(_script_lines(zero_runs))
    assert "NOT MEASURED" not in lines
    assert "script runs=0" in lines
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
