"""Paired A/B replay benchmark for /chat/v1/chat.

Replays recorded multi-turn conversations (from analyze_conversations'
eval_dataset.json) through the chat endpoint under two `tool_profile` arms and
reports per-arm distributions of iterations, tool calls, latency, tokens and cost.

Two properties make the comparison trustworthy:

* PAIRED: both arms of a case run back-to-back inside one worker, so a model
  swap, an API slowdown or a cache warm-up mid-run hits both arms equally.
  Concurrency parallelises over cases, never within a case.
* ALTERNATED ARM ORDER: case i runs (a, b) when i is even and (b, a) when odd,
  so the second-position warm-cache advantage does not accrue to one arm. The
  order actually used is recorded per case.
* MATCHED ANALYSIS: the headline distributions cover only the (case, turn) pairs
  that succeeded on BOTH arms. Comparing each arm's own ok turns would reward an
  arm for failing on the hard, late, expensive ones.

Replayed history carries the tool_result blocks from the `done` chunk back as a
synthetic user turn, exactly as the browser does (LLMChat.tsx). Dropping them makes
llm_service strip the assistant turn's tool_use blocks as orphaned, which deletes
most of the context growth this harness exists to measure — and asymmetrically, in
favour of whichever arm calls more tools.

Metrics are read off the SSE stream, NOT out of chat_history.db. Every request
is sent with secret=true, and secret chat deliberately writes no chat_turn_metrics
row (genetics-results-suite-4h6.1) — a benchmark must never leave a trace in a
user's history, so the stream is the only honest source. It also means the harness
works against any deployment without database access.

RUNNING THIS COSTS REAL MONEY: production averaged $2.01 per turn. The target URL
defaults to localhost and there is a --dry-run that resolves the whole plan without
issuing a single request.
"""

import argparse
import asyncio
import json
import logging
import math
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

import httpx

from genetics_mcp_server.cost import estimate_cost, has_pricing

logger = logging.getLogger(__name__)

CHAT_PATH = "/chat/v1/chat"

# `tool_profile: null` means "all tools" (definitions.py), and null is unspellable on
# the command line, so this literal is how a caller asks for it.
ALL_TOOLS_ARM = "all"

REPORTED_PERCENTILES = (25, 50, 75, 90, 95)

# a percentile is only distinguishable from the maximum once the sample has at least
# one observation above it: n * (1 - p) >= 1. Below that, p95 IS the max and quoting
# it as a percentile is a lie about the sample.
# integer arithmetic on purpose: 1 - 90/100 is 0.09999999999999998 in binary floating
# point, which rounds p90's threshold up to 11 and silently contradicts the documented 10
MIN_N_FOR_PERCENTILE = {p: math.ceil(100 / (100 - p)) for p in REPORTED_PERCENTILES}

# nothing on the chat stream emits script-execution results today. The code-execution
# arm (the sandbox track of this epic) is expected to add a chunk of this type; until
# some chunk of this type is seen, script metrics stay None — "not measured" — rather
# than 0, which would read as "measured, no failures" and hand the code arm a free win.
SCRIPT_RESULT_CHUNK_TYPE = "script_result"

UNMEASURED_REASONS = {
    "script_failures": (
        f"no chat/v1/chat SSE chunk of type '{SCRIPT_RESULT_CHUNK_TYPE}' was seen in this "
        "run. The code-execution arm must emit one per script with exit_code / timed_out / "
        "exception for this to populate; until then the value is unmeasured, not zero."
    ),
    "retry_loops": (
        f"derived from '{SCRIPT_RESULT_CHUNK_TYPE}' chunks, none of which were seen in this "
        "run. A retry loop is a failed script followed by a further model iteration."
    ),
}


@dataclass
class IterationUsage:
    """One `usage` SSE chunk, as emitted by llm_service per model roundtrip."""

    iteration: int
    # NOTE: llm_service sets this to input_tok + cache_read + cache_create, i.e. the whole
    # context sent that iteration, not the billed uncached input. Named as the wire names it.
    input_tokens: int
    output_tokens: int
    total_input_tokens: int
    total_output_tokens: int
    context_window: int
    context_percent: float


@dataclass
class TurnRecord:
    """One replayed user turn under one arm."""

    case_id: str
    arm: str
    arm_position: int  # 0 = ran first for this case, 1 = ran second
    turn_index: int
    status: str  # ok | error | timeout | incomplete | not_attempted
    error: str | None = None

    iterations: int | None = None
    tool_calls: int | None = None
    hit_max_iterations: bool = False

    ms_to_first_token: float | None = None
    ms_to_done: float | None = None

    # billed uncached input; llm_service's total_input_tokens accumulates raw input_tokens
    input_tokens: int | None = None
    output_tokens: int | None = None
    # cache_read and cache_creation are NOT separable from the stream: the usage chunk
    # carries their sum folded into per-iteration input_tokens. See cost_usd_min/max.
    cached_input_tokens: int | None = None

    context_tokens_first_iteration: int | None = None
    context_tokens_last_iteration: int | None = None
    context_percent_last_iteration: float | None = None

    cost_usd_min: float | None = None
    cost_usd_max: float | None = None

    # populated only once something emits SCRIPT_RESULT_CHUNK_TYPE; None means unmeasured
    script_runs: int | None = None
    script_failures: int | None = None
    retry_loops: int | None = None

    iterations_detail: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CaseResult:
    case_id: str
    topic: str | None
    arm_order: list[str]
    turns: list[TurnRecord] = field(default_factory=list)


def _sse_events(lines: AsyncIterator[str]) -> Any:
    """Parse an SSE byte stream into (event, data-dict) pairs.

    sse-starlette emits `event: <name>` / `data: <json>` pairs terminated by a blank
    line, and keepalive comments beginning with ':'. Multi-line data is rejoined with
    newlines per the SSE spec.
    """

    async def gen():
        event = "message"
        data_lines: list[str] = []
        async for raw in lines:
            line = raw.rstrip("\r")
            if line == "":
                if data_lines:
                    payload = "\n".join(data_lines)
                    try:
                        yield event, json.loads(payload)
                    except json.JSONDecodeError:
                        logger.warning("unparseable SSE data, skipped: %.200s", payload)
                event = "message"
                data_lines = []
                continue
            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        if data_lines:
            payload = "\n".join(data_lines)
            try:
                yield event, json.loads(payload)
            except json.JSONDecodeError:
                logger.warning("unparseable trailing SSE data, skipped: %.200s", payload)

    return gen()


def count_tool_calls(message_content: list[dict[str, Any]] | None) -> int:
    """Count real tool_use blocks in a done chunk's message_content.

    Deliberately not the '*[Using tool: X]*' text markers: those are display prose the
    model has been observed to imitate (analyze_conversations, genetics-results-suite-4h6.2).
    message_content carries the actual blocks from every iteration of the turn.
    """
    if not message_content:
        return 0
    return sum(1 for b in message_content if isinstance(b, dict) and b.get("type") == "tool_use")


def _cost_bounds(
    model: str | None, input_tokens: int, output_tokens: int, cached_tokens: int
) -> tuple[float | None, float | None]:
    """Bound the turn's USD cost.

    The stream cannot separate cache reads from cache creations (llm_service folds both
    into the usage chunk's per-iteration input_tokens), and they differ by more than 12x
    in price. So the honest answer is an interval: everything-cache-read is the floor,
    everything-cache-creation the ceiling. In steady state the truth sits near the floor,
    but the harness does not pretend to know that.
    """
    if not model:
        return None, None
    if not has_pricing(model):
        # cost.py prices anything unrecognised as Sonnet. A confident wrong USD number in
        # the report that gates the ship decision is worse than no number at all.
        logger.warning(
            "model %r has no pricing entry (cost.py knows claude-opus / claude-sonnet / "
            "claude-haiku); USD is reported as not-priced rather than guessed",
            model,
        )
        return None, None
    low = estimate_cost(model, input_tokens, output_tokens, cache_read_tokens=cached_tokens)
    high = estimate_cost(model, input_tokens, output_tokens, cache_creation_tokens=cached_tokens)
    return low, high


async def replay_turn(
    client: httpx.AsyncClient,
    base_url: str,
    messages: list[dict[str, Any]],
    options: dict[str, Any],
    arm: str,
    case_id: str,
    arm_position: int,
    turn_index: int,
    session_id: str,
    model: str | None,
    provider: str | None,
    timeout: float,
) -> tuple[TurnRecord, list[dict[str, Any]] | None, list[dict[str, Any]]]:
    """Send one turn and read its metrics off the stream.

    Returns the record, the assistant message_content to append to the history for the
    next turn (None if the turn did not complete), and the tool_result blocks that answer
    that message's tool_use blocks (empty when the turn used no tools).
    """
    record = TurnRecord(
        case_id=case_id,
        arm=arm,
        arm_position=arm_position,
        turn_index=turn_index,
        status="incomplete",
    )

    body: dict[str, Any] = {
        "messages": messages,
        # never negotiable: a benchmark must not write into anyone's history
        "secret": True,
        "session_id": session_id,
        "enable_tools": True,
        # the arm IS the tool_profile; the recorded value is deliberately discarded
        "tool_profile": None if arm == ALL_TOOLS_ARM else arm,
        "verbosity": options.get("verbosity"),
        "instruction_set_id": options.get("instruction_set_id"),
        "literature_backend": options.get("literature_backend"),
    }
    if model:
        body["model"] = model
    if provider:
        # unset means the deployment's default_provider decides, and the two providers do
        # not stream the same chunks (the OpenAI path emits no usage chunk at all)
        body["provider"] = provider

    usages: list[IterationUsage] = []
    script_runs = 0
    script_failures = 0
    retry_loops = 0
    saw_script_chunk = False
    pending_script_failure = False
    message_content: list[dict[str, Any]] | None = None
    tool_results: list[dict[str, Any]] = []
    started = time.perf_counter()

    try:
        # httpx's timeout is per READ, and sse-starlette sends a keepalive comment every
        # 15s that _sse_events drops silently — each one resets that read timeout, so a
        # wedged generator could stream keepalives forever without ever tripping it. This
        # is the wall-clock deadline the --timeout flag actually promises.
        async with asyncio.timeout(timeout):
            async with client.stream(
                "POST", f"{base_url}{CHAT_PATH}", json=body, timeout=timeout
            ) as response:
                if response.status_code != 200:
                    await response.aread()
                    record.status = "error"
                    record.error = f"HTTP {response.status_code}: {response.text[:300]}"
                    return record, None, []

                async for event, data in _sse_events(response.aiter_lines()):
                    dtype = data.get("type")
                    if dtype == "content":
                        if record.ms_to_first_token is None and data.get("content"):
                            record.ms_to_first_token = (time.perf_counter() - started) * 1000
                    elif dtype == "usage":
                        usages.append(
                            IterationUsage(
                                iteration=int(data.get("iteration", len(usages) + 1)),
                                input_tokens=int(data.get("input_tokens", 0)),
                                output_tokens=int(data.get("output_tokens", 0)),
                                total_input_tokens=int(data.get("total_input_tokens", 0)),
                                total_output_tokens=int(data.get("total_output_tokens", 0)),
                                context_window=int(data.get("context_window", 0)),
                                context_percent=float(data.get("context_percent", 0.0)),
                            )
                        )
                        if pending_script_failure:
                            # a failed script that is followed by another model roundtrip
                            # is exactly the wasted iteration this counter exists to expose
                            retry_loops += 1
                            pending_script_failure = False
                    elif dtype == SCRIPT_RESULT_CHUNK_TYPE:
                        saw_script_chunk = True
                        script_runs += 1
                        failed = bool(
                            data.get("exit_code")
                            or data.get("timed_out")
                            or data.get("exception")
                        )
                        if failed:
                            script_failures += 1
                            pending_script_failure = True
                    elif dtype == "error" or event == "error":
                        record.status = "error"
                        record.error = str(data.get("error", data))[:500]
                        break
                    elif dtype == "done":
                        message_content = data.get("message_content") or []
                        tool_results = data.get("tool_results") or []
                        record.ms_to_done = (time.perf_counter() - started) * 1000
                        record.status = "ok"
                        break
    except (httpx.TimeoutException, asyncio.TimeoutError) as e:
        record.status = "timeout"
        record.error = f"{type(e).__name__}: {e}"[:500]
    except Exception as e:  # a broken turn must not abort the run
        record.status = "error"
        record.error = f"{type(e).__name__}: {e}"[:500]

    if usages:
        last = usages[-1]
        record.iterations = last.iteration
        record.input_tokens = last.total_input_tokens
        record.output_tokens = last.total_output_tokens
        # per-iteration input_tokens is the whole context; subtracting the accumulated
        # uncached input leaves cache_read + cache_creation, which the stream cannot split
        record.cached_input_tokens = max(
            0, sum(u.input_tokens for u in usages) - last.total_input_tokens
        )
        record.context_tokens_first_iteration = usages[0].input_tokens
        record.context_tokens_last_iteration = last.input_tokens
        record.context_percent_last_iteration = last.context_percent
        record.cost_usd_min, record.cost_usd_max = _cost_bounds(
            model, record.input_tokens, record.output_tokens, record.cached_input_tokens
        )
        record.iterations_detail = [asdict(u) for u in usages]

    if record.status == "ok" and not usages:
        # a turn that reached `done` without a single usage chunk is not measurable: the
        # OpenAI path in llm_service emits one synthetic text block and no usage chunk at
        # all, so counting its (necessarily zero) tool_use blocks would push a fake 0 into
        # the tool-call distribution while contributing nothing to any other — the two
        # samples' n would diverge and the tool-call median would be dragged down.
        record.status = "no_usage_chunks"
        record.error = (
            "the turn reached `done` but emitted no `usage` chunk, so iterations, tokens, "
            "cost and tool calls are all unmeasurable for it (the OpenAI provider path "
            "behaves this way). Pin --provider anthropic."
        )

    if record.status == "ok":
        record.tool_calls = count_tool_calls(message_content)
        record.hit_max_iterations = any(
            isinstance(b, dict)
            and b.get("type") == "text"
            and "Max tool iterations reached" in (b.get("text") or "")
            for b in (message_content or [])
        )

    if saw_script_chunk:
        record.script_runs = script_runs
        record.script_failures = script_failures
        record.retry_loops = retry_loops

    # an unmeasurable turn still leaves the history intact, so the case continues; it is
    # the status, not an abort, that keeps it out of the comparison
    completed = record.status in ("ok", "no_usage_chunks")
    return record, (message_content if completed else None), (tool_results if completed else [])


async def replay_case_arm(
    client: httpx.AsyncClient,
    base_url: str,
    case: dict[str, Any],
    arm: str,
    arm_position: int,
    run_id: str,
    model: str | None,
    provider: str | None,
    timeout: float,
    max_turns: int | None,
) -> list[TurnRecord]:
    """Replay one case's whole user-turn sequence under one arm."""
    case_id = str(case.get("session_id") or case.get("id") or "unknown")
    user_turns = case.get("user_turns") or []
    if max_turns is not None:
        user_turns = user_turns[:max_turns]

    session_id = f"replay-{run_id}-{case_id[:8]}-{arm}"
    history: list[dict[str, Any]] = []
    records: list[TurnRecord] = []
    aborted = False

    for turn_index, turn in enumerate(user_turns):
        if aborted:
            # counted, never dropped: the history is broken after a failed turn, so the
            # rest of the sequence cannot be replayed under comparable conditions
            records.append(
                TurnRecord(
                    case_id=case_id,
                    arm=arm,
                    arm_position=arm_position,
                    turn_index=turn_index,
                    status="not_attempted",
                    error="an earlier turn of this case/arm failed",
                )
            )
            continue

        history.append({"role": "user", "content": turn["content"]})
        record, message_content, tool_results = await replay_turn(
            client=client,
            base_url=base_url,
            messages=history,
            options=turn.get("options") or {},
            arm=arm,
            case_id=case_id,
            arm_position=arm_position,
            turn_index=turn_index,
            session_id=session_id,
            model=model,
            provider=provider,
            timeout=timeout,
        )
        records.append(record)
        if message_content is None:
            aborted = True
        else:
            history.append({"role": "assistant", "content": message_content})
            if tool_results:
                # exactly what the browser replays (LLMChat.tsx: entries.push({role:"user",
                # content: toolResults})). Without it llm_service's _sanitize_tool_blocks
                # strips every tool_use block from this turn as orphaned, so from turn 2 on
                # the model would see an assistant answer with all its tool output deleted —
                # and tool results are the bulk of the context growth this harness measures.
                history.append({"role": "user", "content": tool_results})

    return records


def arm_order_for_case(arms: tuple[str, str], case_index: int) -> list[str]:
    """The alternating arm order for case i: (a, b) when even, (b, a) when odd."""
    return list(arms) if case_index % 2 == 0 else list(reversed(arms))


async def replay_case(
    client: httpx.AsyncClient,
    base_url: str,
    case: dict[str, Any],
    case_index: int,
    arms: tuple[str, str],
    run_id: str,
    model: str | None,
    provider: str | None,
    timeout: float,
    max_turns: int | None,
) -> CaseResult:
    """Run both arms of one case back to back, in an alternating order."""
    order = arm_order_for_case(arms, case_index)
    result = CaseResult(
        case_id=str(case.get("session_id") or "unknown"),
        topic=case.get("topic"),
        arm_order=order,
    )
    for position, arm in enumerate(order):
        result.turns.extend(
            await replay_case_arm(
                client=client,
                base_url=base_url,
                case=case,
                arm=arm,
                arm_position=position,
                run_id=run_id,
                model=model,
                provider=provider,
                timeout=timeout,
                max_turns=max_turns,
            )
        )
    return result


def percentile(sorted_values: list[float], p: float) -> float:
    """Linear-interpolated percentile over pre-sorted values."""
    if not sorted_values:
        raise ValueError("percentile of an empty sample")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (p / 100) * (len(sorted_values) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(sorted_values[low])
    return float(sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (rank - low))


def distribution(values: list[float | int | None]) -> dict[str, Any]:
    """Percentile summary, with every percentile the sample is too small to support flagged."""
    clean = sorted(float(v) for v in values if v is not None)
    n = len(clean)
    if n == 0:
        return {"n": 0, "unreliable_percentiles": [], "note": "no observations"}
    out: dict[str, Any] = {
        "n": n,
        "mean": sum(clean) / n,
        "min": clean[0],
        "max": clean[-1],
    }
    unreliable = []
    for p in REPORTED_PERCENTILES:
        out[f"p{p}"] = percentile(clean, p)
        if n < MIN_N_FOR_PERCENTILE[p]:
            unreliable.append(f"p{p}")
    out["unreliable_percentiles"] = unreliable
    return out


_TURN_DISTRIBUTIONS = (
    ("iterations", "iterations"),
    ("tool_calls", "tool_calls"),
    ("ms_to_first_token", "ms_to_first_token"),
    ("ms_to_done", "ms_to_done"),
    ("input_tokens", "input_tokens"),
    ("output_tokens", "output_tokens"),
    ("cached_input_tokens", "cached_input_tokens"),
    ("context_tokens_last_iteration", "context_tokens_last_iteration"),
    ("cost_usd_min", "cost_usd_min"),
    ("cost_usd_max", "cost_usd_max"),
)


def summarize_arm(records: list[TurnRecord]) -> dict[str, Any]:
    ok = [r for r in records if r.status == "ok"]
    status_counts: dict[str, int] = {}
    for r in records:
        status_counts[r.status] = status_counts.get(r.status, 0) + 1

    summary: dict[str, Any] = {
        "turns_attempted": sum(1 for r in records if r.status != "not_attempted"),
        "turns_ok": len(ok),
        "turns_by_status": status_counts,
        "turns_hitting_max_iterations": sum(1 for r in ok if r.hit_max_iterations),
        "distributions": {
            name: distribution([getattr(r, attr) for r in ok])
            for name, attr in _TURN_DISTRIBUTIONS
        },
    }

    measured_scripts = [r for r in records if r.script_runs is not None]
    if measured_scripts:
        runs = sum(r.script_runs or 0 for r in measured_scripts)
        failures = sum(r.script_failures or 0 for r in measured_scripts)
        summary["script_runs"] = runs
        summary["script_failures"] = failures
        summary["script_failure_rate"] = (failures / runs) if runs else None
        summary["retry_loops"] = sum(r.retry_loops or 0 for r in measured_scripts)
    else:
        summary["script_runs"] = None
        summary["script_failures"] = None
        summary["script_failure_rate"] = None
        summary["retry_loops"] = None
        summary["unmeasured"] = dict(UNMEASURED_REASONS)

    return summary


def matched_pairs(
    all_turns: list[TurnRecord], arms: tuple[str, str]
) -> tuple[set[tuple[str, int]], dict[str, int]]:
    """The `(case_id, turn_index)` keys that succeeded on BOTH arms, plus per-arm dropout.

    The design is paired, so the comparison has to be paired too. An arm that fails on the
    hard, late, expensive turns keeps only its cheap early turns in an unmatched
    distribution, while the healthy arm carries the full tail — its medians would drop
    BECAUSE it failed. Restricting to keys ok on both arms removes that reward.
    """
    ok_by_arm = {
        arm: {(t.case_id, t.turn_index) for t in all_turns if t.arm == arm and t.status == "ok"}
        for arm in arms
    }
    matched: set[tuple[str, int]] = set.intersection(*ok_by_arm.values()) if ok_by_arm else set()
    dropped = {arm: len(keys - matched) for arm, keys in ok_by_arm.items()}
    return matched, dropped


def build_report(
    cases: list[CaseResult], arms: tuple[str, str], config: dict[str, Any]
) -> dict[str, Any]:
    all_turns = [t for c in cases for t in c.turns]
    matched, dropped = matched_pairs(all_turns, arms)
    return {
        "config": config,
        "cases": len(cases),
        "arms": list(arms),
        "arm_order_per_case": {c.case_id: c.arm_order for c in cases},
        # headline: the paired comparison, over turns that succeeded on both arms
        "matched": {
            "pairs": len(matched),
            "dropped_ok_turns_per_arm": dropped,
            "per_arm": {
                arm: summarize_arm(
                    [
                        t
                        for t in all_turns
                        if t.arm == arm and (t.case_id, t.turn_index) in matched
                    ]
                )
                for arm in arms
            },
        },
        # secondary: each arm's own ok turns, which differential dropout can bias
        "per_arm": {arm: summarize_arm([t for t in all_turns if t.arm == arm]) for arm in arms},
        "turns": [asdict(t) for t in all_turns],
    }


def _fmt(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:,.{digits}f}"


def _distribution_lines(summary: dict[str, Any]) -> list[str]:
    lines = [
        f"  {'metric':<32}{'n':>5}{'mean':>12}{'p25':>10}{'p50':>10}"
        f"{'p75':>10}{'p90':>10}{'p95':>10}{'max':>12}"
    ]
    for name, _attr in _TURN_DISTRIBUTIONS:
        d = summary["distributions"][name]
        if d["n"] == 0:
            lines.append(f"  {name:<32}{0:>5}   (no observations)")
            continue
        digits = 4 if name.startswith("cost_") else 1
        row = (
            f"  {name:<32}{d['n']:>5}{_fmt(d['mean'], digits):>12}"
            f"{_fmt(d['p25'], digits):>10}{_fmt(d['p50'], digits):>10}"
            f"{_fmt(d['p75'], digits):>10}{_fmt(d['p90'], digits):>10}"
            f"{_fmt(d['p95'], digits):>10}{_fmt(d['max'], digits):>12}"
        )
        if d["unreliable_percentiles"]:
            row += f"   [{','.join(d['unreliable_percentiles'])} unreliable at n={d['n']}]"
        lines.append(row)
    return lines


def _script_lines(summary: dict[str, Any]) -> list[str]:
    # branch on script_runs, not the rate: an arm that WAS measured and ran zero scripts
    # has a rate of None too, and printing NOT MEASURED for it defeats the whole point
    # of the field, which is telling "unmeasured" apart from "measured, no failures"
    if summary["script_runs"] is None:
        lines = [
            "  script failure rate           : NOT MEASURED",
            "  retry-loop count              : NOT MEASURED",
        ]
        lines.extend(f"      {key}: {reason}" for key, reason in summary.get("unmeasured", {}).items())
        return lines
    rate = summary["script_failure_rate"]
    rate_text = "n/a (0 scripts run)" if rate is None else f"{rate:.3f}"
    return [
        f"  script runs={summary['script_runs']} failures={summary['script_failures']} "
        f"rate={rate_text} retry loops={summary['retry_loops']}"
    ]


def format_summary(report: dict[str, Any]) -> str:
    lines: list[str] = []
    cfg = report["config"]
    lines.append("=" * 78)
    lines.append("PAIRED REPLAY BENCHMARK")
    lines.append("=" * 78)
    lines.append(f"target      : {cfg['base_url']}")
    lines.append(f"dataset     : {cfg['dataset']}")
    lines.append(f"model       : {cfg['model'] or 'server default (cost NOT priced)'}")
    lines.append(f"provider    : {cfg.get('provider') or 'server default'}")
    lines.append(f"cases (N)   : {report['cases']}   arms: {' vs '.join(report['arms'])}")
    lines.append("design      : paired (both arms per case, alternating arm order), secret=true")
    if report["cases"] < 10:
        lines.append(
            f"WARNING     : N={report['cases']} cases. Tail percentiles are not stable at "
            "this size; treat everything below as indicative only."
        )
    lines.append("")

    matched = report["matched"]
    dropped = matched["dropped_ok_turns_per_arm"]
    lines.append("=" * 78)
    lines.append(
        f"HEADLINE — MATCHED PAIRS: {matched['pairs']} turns ok on BOTH arms"
    )
    lines.append("=" * 78)
    if any(dropped.values()):
        detail = ", ".join(f"{arm}: {n}" for arm, n in dropped.items())
        lines.append(
            f"NOTE        : the arms' ok-sets differ. Turns ok on one arm but not the "
            f"other, excluded here ({detail})."
        )
        lines.append(
            "              An arm that fails on the hard, late turns would otherwise look "
            "cheaper for having failed; the matched comparison removes that reward."
        )
    if matched["pairs"] == 0:
        lines.append("NOTE        : no turn succeeded on both arms — there is no valid comparison.")
    lines.append("")

    for arm in report["arms"]:
        s = matched["per_arm"][arm]
        lines.append("-" * 78)
        lines.append(f"ARM (matched): {arm}   turns={s['turns_ok']}")
        lines.extend(_distribution_lines(s))
        lines.extend(_script_lines(s))
        lines.append("")

    lines.append("=" * 78)
    lines.append("SECONDARY — UNMATCHED MARGINALS (each arm's own ok turns)")
    lines.append(
        "these are NOT comparable across arms when the ok-sets differ; they are here to "
        "show each arm's dropout, not to be diffed."
    )
    lines.append("=" * 78)
    for arm in report["arms"]:
        s = report["per_arm"][arm]
        lines.append("-" * 78)
        lines.append(f"ARM (unmatched marginal): {arm}")
        lines.append(
            f"  turns attempted={s['turns_attempted']} ok={s['turns_ok']} "
            f"by status={s['turns_by_status']} "
            f"hit max-iterations={s['turns_hitting_max_iterations']}"
        )
        lines.extend(_distribution_lines(s))
        lines.extend(_script_lines(s))
        lines.append("")

    lines.append("-" * 78)
    lines.append(
        "cost is reported as an interval: the usage stream folds cache reads and cache "
        "creations together, and they differ >12x in price. The true cost lies between "
        "cost_usd_min and cost_usd_max, near the minimum in steady state."
    )
    return "\n".join(lines)


def load_cases(dataset: Path, limit: int | None) -> list[dict[str, Any]]:
    """Load eval cases in a deterministic order, keeping only replayable ones."""
    with open(dataset) as f:
        data = json.load(f)
    cases = [c for c in data if c.get("user_turns")]
    cases.sort(key=lambda c: str(c.get("session_id") or ""))
    if limit is not None:
        cases = cases[:limit]
    return cases


async def run_benchmark(
    dataset: Path,
    base_url: str,
    arms: tuple[str, str],
    limit: int | None,
    concurrency: int,
    model: str | None,
    timeout: float,
    max_turns: int | None,
    auth_token: str | None,
    provider: str | None = None,
) -> dict[str, Any]:
    cases = load_cases(dataset, limit)
    run_id = uuid.uuid4().hex[:8]
    headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else {}
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async with httpx.AsyncClient(headers=headers, timeout=timeout) as client:

        async def guarded(index: int, case: dict[str, Any]) -> CaseResult:
            # the semaphore is held for the WHOLE case so both arms stay adjacent in time
            async with semaphore:
                return await replay_case(
                    client=client,
                    base_url=base_url,
                    case=case,
                    case_index=index,
                    arms=arms,
                    run_id=run_id,
                    model=model,
                    provider=provider,
                    timeout=timeout,
                    max_turns=max_turns,
                )

        results = await asyncio.gather(
            *(guarded(i, c) for i, c in enumerate(cases)), return_exceptions=True
        )

    cases_out: list[CaseResult] = []
    for index, (case, result) in enumerate(zip(cases, results)):
        if isinstance(result, BaseException):
            logger.error("case %s failed entirely: %s", case.get("session_id"), result)
            case_id = str(case.get("session_id") or "unknown")
            order = arm_order_for_case(arms, index)
            planned_turns = (case.get("user_turns") or [])[:max_turns]
            cases_out.append(
                CaseResult(
                    case_id=case_id,
                    topic=case.get("topic"),
                    arm_order=order,
                    # one record per (arm, turn): collapsing a 6-turn case into two
                    # turn_index=0 records would under-report turns_attempted by 10 and
                    # hide the loss from the per-status table
                    turns=[
                        TurnRecord(
                            case_id=case_id,
                            arm=arm,
                            arm_position=pos,
                            turn_index=turn_index,
                            status="error",
                            error=f"{type(result).__name__}: {result}"[:500],
                        )
                        for pos, arm in enumerate(order)
                        for turn_index in range(len(planned_turns))
                    ],
                )
            )
        else:
            cases_out.append(result)

    return build_report(
        cases_out,
        arms,
        {
            "dataset": str(dataset),
            "base_url": base_url,
            "run_id": run_id,
            "model": model,
            "provider": provider,
            "limit": limit,
            "concurrency": concurrency,
            "max_turns": max_turns,
            "timeout_s": timeout,
            "secret": True,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paired A/B replay benchmark for /chat/v1/chat. Costs real money.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("conversation_analysis/eval_dataset.json"),
        help="eval_dataset.json from analyze_conversations --export-eval",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="chat service base URL. Deliberately localhost by default: pointing this at "
        "production spends real money (measured mean $2.01/turn).",
    )
    parser.add_argument("--arm-a", default=ALL_TOOLS_ARM, help=f"tool_profile for arm A ('{ALL_TOOLS_ARM}' = all tools)")
    parser.add_argument("--arm-b", default="bigquery", help="tool_profile for arm B")
    parser.add_argument("--limit", type=int, default=None, help="max cases to replay")
    parser.add_argument("--max-turns", type=int, default=None, help="max user turns per case")
    parser.add_argument("--concurrency", type=int, default=1, help="cases in flight (arms within a case are always sequential)")
    parser.add_argument(
        "--model",
        default=None,
        help="pin the model. Strongly recommended: it removes model drift AND is the only "
        "way cost can be priced — without it USD is reported as not-priced, not as zero.",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="pin the LLM provider ('anthropic' or 'openai'). Unset inherits the "
        "deployment's default_provider, and the OpenAI path emits no usage chunk at all, "
        "so every turn would come back unmeasurable.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=900.0,
        help="per-turn wall-clock deadline, seconds (not a per-read timeout: SSE "
        "keepalives would reset that indefinitely)",
    )
    parser.add_argument("--output", type=Path, default=None, help="write the JSON report here")
    parser.add_argument("--dry-run", action="store_true", help="resolve the plan, issue no requests")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    if args.arm_a == args.arm_b:
        print("arm-a and arm-b must differ", file=sys.stderr)
        return 2
    if not args.dataset.exists():
        print(f"dataset not found: {args.dataset}", file=sys.stderr)
        return 2

    arms = (args.arm_a, args.arm_b)

    if args.model and not has_pricing(args.model):
        print(
            f"WARNING: no pricing entry for model {args.model!r}; USD will be reported as "
            "not-priced rather than guessed at Sonnet rates.",
            file=sys.stderr,
        )

    if args.dry_run:
        cases = load_cases(args.dataset, args.limit)
        turns = sum(
            len((c.get("user_turns") or [])[: args.max_turns]) for c in cases
        )
        print(f"{len(cases)} cases, {turns} turns per arm, {turns * 2} model turns total")
        print(f"arms: {arms[0]} vs {arms[1]}   target: {args.base_url}")
        for i, c in enumerate(cases):
            order = arm_order_for_case(arms, i)
            print(f"  {i:>3} {c.get('session_id')} order={order[0]},{order[1]}")
        return 0

    report = asyncio.run(
        run_benchmark(
            dataset=args.dataset,
            base_url=args.base_url.rstrip("/"),
            arms=arms,
            limit=args.limit,
            concurrency=args.concurrency,
            model=args.model,
            timeout=args.timeout,
            max_turns=args.max_turns,
            auth_token=os.environ.get("REPLAY_AUTH_TOKEN"),
            provider=args.provider,
        )
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("wrote %s", args.output)

    print(format_summary(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
