"""Paired A/B replay benchmark for /chat/v1/chat.

Replays recorded multi-turn conversations (from analyze_conversations'
eval_dataset.json) through the chat endpoint under two `tool_profile` arms and
reports per-arm distributions of iterations, tool calls, latency, tokens and cost,
plus a per-iteration timeline that separates model time from the tool phase and the
script-failure / retry-loop counters the code-execution arm is judged on.

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

Quality is a SEPARATE, OPTIONAL pass: `--judge` hands the matched pairs to
`pairwise_judge`, which compares the two arms' FINAL ANSWERS blind, in both presentation
orders, and reports wins/losses/ties. It is off by default — cost and latency are measured
with no judge call at all — and a saved report can be judged later without replaying
anything.

RUNNING THIS COSTS REAL MONEY: production averaged $2.01 per turn. The target URL
defaults to localhost and there is a --dry-run that resolves the whole plan without
issuing a single request. `--judge` is Opus-5 spend ON TOP of that, doubled by the second
presentation order, and is priced before the first call.
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

# llm_service emits one chunk of this type per completed `run_analysis`, carrying `ran` /
# `ok` / `status` / `timed_out` / `exception` (genetics-results-suite-4h6.71). An arm that
# never calls run_analysis therefore emits none, and its script metrics stay None — "not
# measured" — rather than 0, which would read as "measured, no failures". The distinction
# still matters in the other direction too: an arm that DID run scripts and had none fail
# reports a real 0.
SCRIPT_RESULT_CHUNK_TYPE = "script_result"

# llm_service appends this when the agentic loop stops at its iteration ceiling. Kept as the
# harness's OWN literal rather than imported, because the harness reads a REMOTE server's
# stream and that server may not be this build — but a test pins it against
# `llm_service.MAX_ITERATIONS_NOTICE`, so a rename here turns into a red test rather than
# into a silently mis-timed final iteration (the flag decides whether that iteration's tool
# phase is a measured zero or an unmeasured span).
MAX_ITERATIONS_MARKER = "Max tool iterations reached"

# `ran: False` on that chunk says only that the SANDBOX did not execute the script. It does
# NOT say whose fault that was, and two of the shapes carrying it are the model's doing:
# `EmptyScript` (blank or non-string `code`) and `SandboxRejected` (a `timeout_s` outside
# 1..120, or oversize code — the executor's own comment calls it "a caller bug ...
# Actionable, because the model chose the value"). Booking those as infrastructure hands the
# code arm a free 0 on the very metric that exists to price its characteristic risk: a model
# asking for timeout_s=300 twice per case would otherwise report failures=0 rate=0.000
# beside 2N sandbox faults, and a reader would conclude the sandbox is flaky.
SCRIPT_SHAPES_MODEL_CAUSED = frozenset({"EmptyScript", "SandboxRejected"})

# `TurnBudgetExceeded` — run_analysis's own ~300s turn deadline — is the one shape two
# reviewers classified differently: one calls it the code arm's characteristic risk (scripts
# blowing the deadline), the other calls it infrastructure. The cap is ~300s against the
# sandbox's own 120s per-execution MAX_TIMEOUT_S, so a single script cannot trigger it.
# Rather than pick, it gets its OWN bucket, in neither the failure numerator nor the
# denominator, and every shape's count is reported individually — a reader who disagrees
# moves it by hand from numbers that are on the page instead of arguing with a bucketing
# decision baked invisibly into a rate.
SCRIPT_SHAPES_DISPUTED = frozenset({"TurnBudgetExceeded"})

# outcome categories, disjoint and exhaustive over script_result chunks
OUTCOME_EXECUTED_OK = "executed_ok"
OUTCOME_EXECUTED_FAILED = "executed_failed"
OUTCOME_MODEL_REJECTED = "model_rejected"  # never reached the sandbox, and that is the model's doing
OUTCOME_DISPUTED = "disputed"
OUTCOME_INFRA = "infra"

# numerator and denominator, spelled out next to the rate wherever it is printed, so the
# rate cannot be read as covering something it does not
SCRIPT_FAILURE_RATE_DEFINITION = (
    "script_failure_rate = (executed_failed + model_rejected) / (executed_ok + "
    "executed_failed + model_rejected). Sandbox faults are in NEITHER; TurnBudgetExceeded "
    "is in neither and is counted on its own line."
)

UNMEASURED_REASONS = {
    "script_failures": (
        f"no chat/v1/chat SSE chunk of type '{SCRIPT_RESULT_CHUNK_TYPE}' was seen in this "
        "run, i.e. no run_analysis call completed on this arm. That is expected for an arm "
        "whose profile has no code-execution tool; the value is unmeasured, not zero."
    ),
    "retry_loops": (
        f"derived from '{SCRIPT_RESULT_CHUNK_TYPE}' chunks, none of which were seen in this "
        "run. A retry loop is any non-successful script outcome followed by a further model "
        "iteration; it is reported split by what armed it, since the causes are not the same "
        "kind of fact."
    ),
}

# how a turn's USD figure was arrived at. Recorded per turn and counted per arm, because
# a mixed run must not present one number as if it were the same kind of number throughout.
COST_BASIS_EXACT = "exact"  # the usage chunks carried cache_read/cache_create; priced exactly
COST_BASIS_INTERVAL = "interval"  # they did not; only cost_usd_min/max are defensible
COST_BASIS_UNPRICED = "unpriced"  # no model pinned, or no pricing entry for it


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

    # the two cached components of input_tokens, priced >12x apart. Present since
    # genetics-results-suite-n3p; None when the server predates it or the provider path
    # does not report them, which is what forces the turn back onto the cost interval.
    cache_read: int | None = None
    cache_create: int | None = None

    # SINCE THE TURN STARTED, not since the previous iteration, and sampled when this
    # iteration's model response completed. Cumulative on purpose: the per-iteration
    # duration is the difference between consecutive values (see _derive_iteration_timing),
    # so both readings are available and neither has to be guessed at.
    turn_elapsed_ms: float | None = None
    # NOT model latency, despite the name it is given on the wire: the span llm_service
    # times encloses the whole transient-error retry loop, so a single retry buries its
    # 1/2/4s backoff sleep in here, and since the producer is an async generator yielding
    # per delta it also carries downstream SSE serialisation and socket backpressure.
    # Attribution to this iteration is right; the figure is "model call and everything the
    # server did to deliver it". `model_attempts` is what makes a retry-inflated reading
    # identifiable.
    model_ms: float | None = None
    # attempts of the streaming call this iteration; 1 = no retry, so no backoff inside
    # model_ms. None when the server predates the field.
    model_attempts: int | None = None


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
    # the calls behind that count: name, arguments and order, one entry per tool_use block.
    # `tool_calls` is len() of this, so a count that disagrees with the listing is
    # impossible. See `extract_tool_calls` for what is and is not kept.
    tool_calls_detail: list[dict[str, Any]] = field(default_factory=list)
    hit_max_iterations: bool = False

    ms_to_first_token: float | None = None
    ms_to_done: float | None = None

    # billed uncached input; llm_service's total_input_tokens accumulates raw input_tokens
    input_tokens: int | None = None
    output_tokens: int | None = None
    # cache_read + cache_creation, derived by subtraction. Kept because it is computable
    # from any usage stream; it is what forces the interval when the split is missing.
    cached_input_tokens: int | None = None
    # the split, when the stream reported it. None means this turn is interval-priced.
    cache_read_tokens: int | None = None
    cache_create_tokens: int | None = None

    context_tokens_first_iteration: int | None = None
    context_tokens_last_iteration: int | None = None
    context_percent_last_iteration: float | None = None

    # the exact figure, available only when every usage chunk carried the cache split
    cost_usd: float | None = None
    cost_basis: str | None = None
    # the fallback bracket, always computed when the model is priced: min prices all cached
    # tokens as cache reads, max as cache creations. When cost_usd is set it lies inside.
    cost_usd_min: float | None = None
    cost_usd_max: float | None = None

    # summed per-iteration model_ms; ms_to_done minus this is roughly everything that was not
    # the model call — tool execution, prompt assembly, serialisation and HTTP time — and it
    # inherits model_ms's caveat: retry backoff sits inside the model side of that split
    model_ms_total: float | None = None
    # the slowest ROUNDTRIP: model call plus the tool phase that followed it, both belonging
    # to the same iteration index. None for an iteration whose tool phase is unmeasured.
    slowest_iteration: int | None = None
    slowest_iteration_ms: float | None = None
    # iterations that needed more than one attempt at the streaming call, i.e. whose model_ms
    # contains backoff. None when the stream never reported model_attempts.
    iterations_with_model_retries: int | None = None

    # populated only once something emits SCRIPT_RESULT_CHUNK_TYPE; None means unmeasured
    script_runs: int | None = None  # the sandbox executed it
    # runs + model_rejected: everything attempted that the model is answerable for. This,
    # not script_runs, is script_failure_rate's denominator.
    script_attempts: int | None = None
    script_failures: int | None = None
    retry_loops: int | None = None
    # retry_loops split by what armed them; the three sum to retry_loops. A roundtrip
    # following a mix is attributed by precedence script > disputed > infra, so a wasted
    # roundtrip is never credited to the platform when the model's own script also failed.
    retry_loops_script: int | None = None
    retry_loops_disputed: int | None = None
    retry_loops_infra: int | None = None
    # sandbox faults (restart, full queue, unminted token). NOT script failures: they say
    # nothing about the model's script, so they are counted apart from script_runs.
    script_infra_errors: int | None = None
    # TurnBudgetExceeded, in its own bucket — see SCRIPT_SHAPES_DISPUTED
    script_budget_exceeded: int | None = None
    # every distinct status/error_type seen, verbatim, with its count
    script_outcomes: dict[str, int] = field(default_factory=dict)

    iterations_detail: list[dict[str, Any]] = field(default_factory=list)

    # the paired judge's input (genetics-results-suite-4h6.72), recorded here so a saved
    # report can be judged later without replaying anything. `final_answer` is the text
    # AFTER the last tool_use block — what the user is left with, and deliberately not the
    # tool trace, which would name the arm to a supposedly blind judge; see
    # `pairwise_judge.final_answer_split`. `user_question` comes from the replayed dataset
    # and is therefore identical on both arms, which is what makes the pair a pair.
    #
    # `final_answer_dropped_chars` is how much text that rule THREW AWAY — the prose written
    # before the last tool call. It is recorded because the rule is not neutral between the
    # arms: one early tool call keeps almost everything, one late tool call discards whatever
    # was written between calls. The judge's report prints the per-arm median so a verdict
    # produced by the slicing rule cannot be mistaken for one about quality.
    user_question: str | None = None
    final_answer: str | None = None
    final_answer_dropped_chars: int = 0
    # ...and the discarded text ITSELF, with the call it followed, so a transcript can put it
    # back where the model wrote it. Same boundary as the count above (one definition, in
    # `pairwise_judge.dropped_prose_blocks`), and kept for the READER only: the judge is
    # still shown `final_answer` alone, since this half names tools and scripts and would
    # identify the arm on sight. Always captured — it is the model's own visible output, it
    # is small next to the tool arguments already in the report, and its absence made the
    # transcript claim a fragment was the whole reply.
    final_answer_dropped_prose: list[dict[str, Any]] = field(default_factory=list)

    # one entry per iteration that reasoned, {"iteration", "text"}, and EMPTY unless the run
    # asked for it with --capture-thinking. Populated from the `thinking_summary` chunks,
    # which the server emits only on request: the text exists nowhere else, since llm_service
    # keeps thinking blocks out of `message_content` and the ordinary `thinking` chunk is a
    # contentless keepalive. It is the model's SUMMARIZED reasoning — no model exposes the
    # raw chain of thought — and it is never shown to the judge, which would see tool and
    # script names in it and stop being blind.
    thinking_detail: list[dict[str, Any]] = field(default_factory=list)


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


def extract_tool_calls(message_content: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Every tool_use block in a done chunk, in the order the model emitted them.

    Deliberately not the '*[Using tool: X]*' text markers: those are display prose the
    model has been observed to imitate (analyze_conversations, genetics-results-suite-4h6.2).
    message_content carries the actual blocks from every iteration of the turn, so this is
    the whole turn's call sequence, in order, with the arguments each was given.

    THE ARGUMENTS ARE KEPT VERBATIM AND UNTRUNCATED, including `run_analysis`'s entire
    script. That is the point: a tool-call count says the code arm made one call where the
    baseline made six, and cannot say whether the one call asked for the right thing. Only
    the arguments distinguish "one good script" from "one wrong script the model never
    recovered from", and that distinction is most of what a losing case needs explaining.

    Results are NOT recorded, only calls. `tool_results` can carry thousands of data rows
    per call, which would make a report unreadable and unopenable, and the question this
    answers is what the model ASKED for.

    `secret=true` does not redact these: llm_service omits tool input from its LOG line, not
    from the `done` chunk (`block.model_dump(exclude_none=True)`). Verified 2026-08-19. A
    benchmark dataset is replayed questions, not user data — but note that anything in a
    replayed question's arguments does land in the report.
    """
    if not message_content:
        return []
    calls = []
    for block in message_content:
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            continue
        calls.append(
            {
                "seq": len(calls),
                "name": block.get("name"),
                "id": block.get("id"),
                "input": block.get("input"),
            }
        )
    return calls


def count_tool_calls(message_content: list[dict[str, Any]] | None) -> int:
    """Count real tool_use blocks. Defined via the extractor so the two cannot disagree."""
    return len(extract_tool_calls(message_content))


def attach_call_metadata(
    calls: list[dict[str, Any]],
    call_iteration: dict[str, int],
    script_by_call: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join the STREAM's per-call metadata onto calls extracted from the `done` chunk.

    Two things the done chunk structurally cannot answer, and one it must keep answering:

      * WHICH ITERATION A CALL BELONGS TO. `message_content` is every iteration's blocks
        flattened into one list with no boundary between them, so a reader cannot tell six
        calls in one parallel iteration from six iterations of one call each — which is the
        whole difference between a wide arm and a slow one. llm_service emits its `tool_use`
        chunks after the iteration's `usage` chunk and before any of that iteration's tools
        run, so the count of usage chunks seen at that moment IS the iteration.
      * HOW LONG A `run_analysis` CALL TOOK. Its `script_result` chunk carries the sandbox's
        own wall clock keyed by `tool_use_id`. No other tool reports a duration on the wire.

    ARGUMENTS ARE NOT TAKEN FROM THE STREAM, only correlated by id. llm_service rewrites the
    copy it streams before emitting it — it substitutes `search_scientific_literature`'s
    backend and strips `run_analysis`'s model-invented `user`/`session_id` — so the streamed
    input is what the SERVER ran, not what the MODEL asked for, and the second is what a
    losing case needs explaining.

    Absent stays absent. Against a server that emits no `tool_use` chunks the key is simply
    missing rather than guessed, and benchmark_scorecard says the report predates it instead
    of printing an iteration nobody measured.
    """
    for call in calls:
        call_id = call.get("id")
        if not isinstance(call_id, str):
            continue
        if call_id in call_iteration:
            call["iteration"] = call_iteration[call_id]
        script = script_by_call.get(call_id)
        if script is None:
            continue
        # the SCRIPT's own wall clock inside the sandbox (_script_result_payload), which is
        # NOT the iteration's tool phase: that also covers dispatch, the other tools of the
        # same iteration running alongside it, and rendering the result.
        call["script_duration_ms"] = script.get("duration_ms")
        call["script_status"] = script.get("status")
        call["script_ok"] = script.get("ok")
    return calls


def _opt_int(data: dict[str, Any], key: str) -> int | None:
    """The field as an int, or None when the chunk did not carry it.

    Absent is not zero. A server that predates the field must leave the harness knowing it
    does not know, so that the cost falls back to the interval instead of being priced as
    if nothing were cached.
    """
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _opt_float(data: dict[str, Any], key: str) -> float | None:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _script_ran(data: dict[str, Any]) -> bool:
    """Did the sandbox actually execute the script, as opposed to refusing to take it?

    `ran: False` marks an infrastructure fault — a restarting sandbox, a full queue, an
    unminted token, this process's own turn budget. Counting those as script failures would
    charge the code arm for the platform's availability and inflate the one number the
    rollout decision turns on. A chunk without the field is assumed to have run, which is
    what the pre-4h6.71 exit_code-shaped contract implied.
    """
    value = data.get("ran")
    return value if isinstance(value, bool) else True


def _script_ok(data: dict[str, Any]) -> bool:
    """Did the script succeed?

    `ok` is authoritative. The exit_code / timed_out / exception fallback is what the
    harness looked for before anything emitted this chunk; the sandbox has no exit code
    (its supervisor answers with a status string), so `ok` is what a real chunk carries.
    """
    value = data.get("ok")
    if isinstance(value, bool):
        return value
    return not (data.get("exit_code") or data.get("timed_out") or data.get("exception"))


def _script_outcome(data: dict[str, Any]) -> tuple[str, str]:
    """Classify one script_result chunk as (category, shape).

    The shape is the `status` string the chunk already carries — the supervisor's status for
    a script that ran, the executor's `error_type` for one that did not. It is reported
    verbatim and counted per shape, so no bucketing decision made here can hide anything: a
    reader who classifies a shape differently can recompute from the per-shape counts.
    """
    shape = data.get("status")
    shape = shape if isinstance(shape, str) and shape else "unknown"
    if _script_ran(data):
        return (OUTCOME_EXECUTED_OK if _script_ok(data) else OUTCOME_EXECUTED_FAILED), shape
    if shape in SCRIPT_SHAPES_MODEL_CAUSED:
        return OUTCOME_MODEL_REJECTED, shape
    if shape in SCRIPT_SHAPES_DISPUTED:
        return OUTCOME_DISPUTED, shape
    return OUTCOME_INFRA, shape


def _cost_bounds(
    model: str | None, input_tokens: int, output_tokens: int, cached_tokens: int
) -> tuple[float | None, float | None]:
    """Bound the turn's USD cost when the cache split is NOT available.

    Cache reads and cache creations differ by more than 12x in price, so a stream that
    reports only their sum can only be bracketed: everything-cache-read is the floor,
    everything-cache-creation the ceiling. In steady state the truth sits near the floor,
    but the harness does not pretend to know that. `_exact_cost` supersedes this whenever
    the usage chunks carry the split; the bracket is still computed, both as the fallback
    and as a sanity range the exact figure must sit inside.
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


def _exact_cost(
    model: str | None,
    input_tokens: int,
    output_tokens: int,
    cache_read: int,
    cache_create: int,
) -> float | None:
    """Price the turn exactly from the reported cache split.

    `input_tokens` here is llm_service's `total_input_tokens`, which accumulates Anthropic's
    `usage.input_tokens` — the BILLED UNCACHED input, cached tokens already excluded. So the
    three token classes partition the turn's input and this is a priced sum, not an estimate.
    """
    if not model or not has_pricing(model):
        return None
    return estimate_cost(
        model,
        input_tokens,
        output_tokens,
        cache_read_tokens=cache_read,
        cache_creation_tokens=cache_create,
    )


def _derive_iteration_timing(
    usages: list[IterationUsage], *, final_iteration_ran_no_tools: bool = False
) -> list[dict[str, Any]]:
    """Turn the cumulative `turn_elapsed_ms` readings into a per-iteration timeline.

    Per iteration:
      * `segment_ms` — the span between usage chunks, turn_elapsed_ms[i] -
        turn_elapsed_ms[i-1] with turn_elapsed_ms[0] taken as 0. NAMED FOR ITS EPOCH, not
        for an iteration, because it is not one: it is model[i] + the tool phase of i-1.
      * `pre_model_ms` — the part of that segment that was not this iteration's model call.
        For i = 1 that is the turn's setup (history sanitising, tool-schema assembly); for
        i > 1 it is iteration i-1's TOOL PHASE, because the usage chunk is emitted before
        any tool of its own iteration runs.
      * `tool_phase_ms` — the same quantity attributed to the iteration whose tools it was,
        i.e. iteration i's tool phase is iteration i+1's `pre_model_ms`. None on the last
        iteration, which by construction ran no tools (it answered). One caveat: on the
        `max_tokens` CONTINUATION path llm_service resumes the same turn without running
        any tool, so that iteration's `tool_phase_ms` is continuation bookkeeping rather
        than tool time. It is small and it is not restructured here; read it as "what
        happened between the two model calls".
      * `iteration_ms` — the ROUNDTRIP that belongs to iteration i: model_ms[i] +
        tool_phase_ms[i]. The row therefore sums, and `slowest_iteration` names the
        iteration whose own work was slowest. Deriving it from `segment_ms` instead — as
        this did before genetics-results-suite-4h6.73 — pointed one roundtrip too late: it
        charged iteration i with iteration i-1's tools, which are already printed on row
        i-1, and named the wrong bottleneck to the reader who came here to localise one.
        None whenever either half is unmeasured — but the LAST iteration is two cases, not
        one, and collapsing them throws away real observations. When the turn ended because
        the model stopped calling tools, that iteration's tool phase is not unknown, it is
        genuinely ZERO, and `iteration_ms` is `model_ms`. When the turn stopped at the
        iteration ceiling, tools DID run and no following model call ever closed the span,
        so it stays None. `final_iteration_ran_no_tools` carries that discrimination in;
        see `replay_turn` for why the caller can make it and when it declines to.

        `tool_phase_ms` stays None in BOTH cases even though the first one's is zero. The
        column means "a tool phase that was measured", and seeding it with one 0 per turn
        would drag every by-index median toward zero for a reason that has nothing to do
        with how long tools take.

    Anything the wire did not carry stays None rather than being imputed as 0: a missing
    timing must not read as an instantaneous iteration. That extends to GAPS — an untimed
    chunk in the middle resets the baseline, so the following iteration reports None rather
    than a segment silently spanning two iterations. The cost path already demotes a whole
    turn when one chunk lacks the cache split; patching over the hole here instead would
    apply the harness's own standard inconsistently.
    """
    rows = [asdict(u) for u in usages]
    previous_elapsed: float | None = 0.0
    for row, usage in zip(rows, usages):
        row["segment_ms"] = None
        row["pre_model_ms"] = None
        if usage.turn_elapsed_ms is None:
            previous_elapsed = None
            continue
        if previous_elapsed is not None:
            row["segment_ms"] = usage.turn_elapsed_ms - previous_elapsed
            row["pre_model_ms"] = (
                None if usage.model_ms is None else row["segment_ms"] - usage.model_ms
            )
        previous_elapsed = usage.turn_elapsed_ms
    for index, row in enumerate(rows):
        is_final = index + 1 == len(rows)
        row["tool_phase_ms"] = None if is_final else rows[index + 1]["pre_model_ms"]
        tool_phase = (
            0.0 if is_final and final_iteration_ran_no_tools else row["tool_phase_ms"]
        )
        row["iteration_ms"] = (
            None
            if row["model_ms"] is None or tool_phase is None
            else row["model_ms"] + tool_phase
        )
    return rows


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
    capture_thinking: bool = False,
) -> tuple[TurnRecord, list[dict[str, Any]] | None, list[dict[str, Any]]]:
    """Send one turn and read its metrics off the stream.

    Returns the record, the assistant message_content to append to the history for the
    next turn (None if the turn did not complete), and the tool_result blocks that answer
    that message's tool_use blocks (empty when the turn used no tools).
    """
    # imported here, not at module scope: pairwise_judge imports `matched_pairs` from this
    # module (the pairing rule has one definition), so a module-level import back would be a
    # cycle. Same lazy-import shape analyze_conversations uses for its own heavy deps.
    from genetics_mcp_server.scripts.pairwise_judge import (
        dropped_prose_blocks,
        final_answer_split,
        user_question_text,
    )

    record = TurnRecord(
        case_id=case_id,
        arm=arm,
        arm_position=arm_position,
        turn_index=turn_index,
        status="incomplete",
        user_question=user_question_text(messages[-1].get("content") if messages else None),
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
        # a run-level choice, not a per-turn option: it changes what the stream carries, not
        # what the model is asked. A server that predates the field ignores it and the run
        # simply records no reasoning, which the scorecard then says outright.
        "capture_thinking": capture_thinking,
    }
    if model:
        body["model"] = model
    if provider:
        # unset means the deployment's default_provider decides, and the two providers do
        # not stream the same chunks (the OpenAI path emits no usage chunk at all)
        body["provider"] = provider

    usages: list[IterationUsage] = []
    script_outcomes: dict[str, int] = {}
    script_categories: dict[str, int] = {}
    retry_loops_by_cause: dict[str, int] = {}
    saw_script_chunk = False
    # the outcome categories seen since the last usage chunk, i.e. within the current
    # iteration. Non-empty at the next usage chunk means that roundtrip was spent on them.
    pending_causes: set[str] = set()
    # per-call metadata that only exists in the STREAM's ordering; see attach_call_metadata
    call_iteration: dict[str, int] = {}
    script_by_call: dict[str, dict[str, Any]] = {}
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
                    if response.status_code == 429:
                        # NOT a per-turn failure. The server is refusing everything now, so
                        # every remaining turn will fail the same way and the cases already
                        # replayed become uncomparable as their later turns cascade to
                        # not_attempted. Measured 2026-08-19: a 20-turn run against the
                        # default RATE_LIMIT_PER_HOUR=20 produced 8 ok / 16 error /
                        # 29 not_attempted per arm and a report that still looked plausible
                        # -- 20 cases, both arms, correct arm_tools -- while carrying 8 of
                        # 53 matched pairs. Raising it to the whole run is the only useful
                        # response, so stop here rather than spending the rest of the plan
                        # discovering it 45 more times.
                        raise RateLimitedError(record.error)
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
                                # presence, not truthiness: 0 cache reads is a legitimate
                                # measurement and must not degrade the turn to the interval
                                cache_read=_opt_int(data, "cache_read"),
                                cache_create=_opt_int(data, "cache_create"),
                                turn_elapsed_ms=_opt_float(data, "turn_elapsed_ms"),
                                model_ms=_opt_float(data, "model_ms"),
                                model_attempts=_opt_int(data, "model_attempts"),
                            )
                        )
                        if pending_causes:
                            # a script outcome followed by another model roundtrip is exactly
                            # the wasted iteration this counter exists to expose. One per
                            # iteration, not per outcome: two scripts failing in the same
                            # iteration still cost exactly one extra roundtrip.
                            #
                            # It over-counts in one known way, in the safe direction: if one
                            # of two PARALLEL scripts fails while the other succeeds, the
                            # next roundtrip is booked as a retry loop even though the model
                            # may simply be consuming the successful result. That penalises
                            # the code arm, so it is left rather than guessed at.
                            cause = (
                                OUTCOME_EXECUTED_FAILED
                                if pending_causes & {OUTCOME_EXECUTED_FAILED, OUTCOME_MODEL_REJECTED}
                                else OUTCOME_DISPUTED
                                if OUTCOME_DISPUTED in pending_causes
                                else OUTCOME_INFRA
                            )
                            retry_loops_by_cause[cause] = retry_loops_by_cause.get(cause, 0) + 1
                            pending_causes.clear()
                    elif dtype == "tool_use":
                        # emitted after this iteration's usage chunk and before any of its
                        # tools run, so the iteration in force is the one that call belongs
                        # to. Read off the usage chunk rather than counted, so a server that
                        # renumbers iterations stays authoritative over the harness.
                        tool_use_id = data.get("id")
                        if isinstance(tool_use_id, str) and usages:
                            call_iteration[tool_use_id] = usages[-1].iteration
                    elif dtype == "thinking_summary":
                        # emitted per iteration as its blocks are collected, i.e. after that
                        # iteration's usage chunk, so the server's own `iteration` is
                        # authoritative and the usage count is only the fallback
                        text = data.get("text")
                        if isinstance(text, str) and text:
                            record.thinking_detail.append(
                                {
                                    "iteration": data.get("iteration")
                                    or (usages[-1].iteration if usages else None),
                                    "text": text,
                                }
                            )
                    elif dtype == SCRIPT_RESULT_CHUNK_TYPE:
                        saw_script_chunk = True
                        script_tool_use_id = data.get("tool_use_id")
                        if isinstance(script_tool_use_id, str):
                            script_by_call[script_tool_use_id] = data
                        category, shape = _script_outcome(data)
                        script_categories[category] = script_categories.get(category, 0) + 1
                        script_outcomes[shape] = script_outcomes.get(shape, 0) + 1
                        if category != OUTCOME_EXECUTED_OK:
                            # every non-success burns another iteration if the model tries
                            # again, whoever's fault it was, so all of them arm the counter
                            pending_causes.add(category)
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
    except RateLimitedError:
        # the ONE exception to "a broken turn must not abort the run" below: this is not a
        # broken turn, it is the server refusing every turn from here on
        raise
    except (httpx.TimeoutException, asyncio.TimeoutError) as e:
        record.status = "timeout"
        record.error = f"{type(e).__name__}: {e}"[:500]
    except Exception as e:  # a broken turn must not abort the run
        record.status = "error"
        record.error = f"{type(e).__name__}: {e}"[:500]

    # computed BEFORE the timing derivation, which needs it: see the comment there
    record.hit_max_iterations = any(
        isinstance(b, dict)
        and b.get("type") == "text"
        and MAX_ITERATIONS_MARKER in (b.get("text") or "")
        for b in (message_content or [])
    )

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

        # ALL iterations must carry the split, not merely one: a turn priced exactly on some
        # iterations and by assumption on the rest is a mixed number wearing an exact label.
        split_reported = all(
            u.cache_read is not None and u.cache_create is not None for u in usages
        )
        if split_reported:
            record.cache_read_tokens = sum(u.cache_read or 0 for u in usages)
            record.cache_create_tokens = sum(u.cache_create or 0 for u in usages)
            record.cost_usd = _exact_cost(
                model,
                record.input_tokens,
                record.output_tokens,
                record.cache_read_tokens,
                record.cache_create_tokens,
            )
        if record.cost_usd_min is None:
            record.cost_basis = COST_BASIS_UNPRICED
        elif record.cost_usd is not None:
            record.cost_basis = COST_BASIS_EXACT
        else:
            record.cost_basis = COST_BASIS_INTERVAL

        # THE FINAL ITERATION IS TWO CASES AND ONLY ONE OF THEM IS AN ABSENCE.
        # llm_service leaves the loop after the last usage chunk in exactly four ways: no
        # tool_use blocks, the max_tokens continuation budget exhausted, the unfilled-result
        # continuation budget exhausted, or the iteration ceiling. The first three ran NO
        # tools, so that iteration's tool phase is a measured zero, not an unknown. Only the
        # ceiling ran tools whose phase no following model call ever closed — and reaching
        # the ceiling is precisely what appends MAX_ITERATIONS_MARKER, so tools-after-the-
        # last-usage-chunk implies the marker. There is no false negative from the server.
        #
        # This matters rather than being tidy: ~36% of production turns are single-iteration,
        # and the final iteration answers against the largest context (median 39k -> 117k
        # tokens by iteration 7+). Reporting None for all of them would drop every
        # single-iteration turn out of slowest_iteration_ms and make a slow final roundtrip
        # structurally invisible — the same class of defect as naming the wrong iteration.
        #
        # Restricted to `ok` turns because the other endings genuinely are unmeasured: an
        # error or timeout mid-turn can land after tools ran, with no marker and no `done`.
        # The marker's other failure mode is a false POSITIVE (a model quoting the text
        # back, or the loop reaching the ceiling on a turn that then answered without
        # tools), which yields None — an honest absence, the safe direction.
        record.iterations_detail = _derive_iteration_timing(
            usages,
            final_iteration_ran_no_tools=(
                record.status == "ok" and not record.hit_max_iterations
            ),
        )
        model_times = [u.model_ms for u in usages if u.model_ms is not None]
        if model_times:
            record.model_ms_total = sum(model_times)
        attempts = [u.model_attempts for u in usages if u.model_attempts is not None]
        if attempts:
            record.iterations_with_model_retries = sum(1 for a in attempts if a > 1)
        timed = [
            (row["iteration_ms"], row["iteration"])
            for row in record.iterations_detail
            if row.get("iteration_ms") is not None
        ]
        if timed:
            record.slowest_iteration_ms, record.slowest_iteration = max(timed)

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
        record.tool_calls_detail = attach_call_metadata(
            extract_tool_calls(message_content), call_iteration, script_by_call
        )
        record.tool_calls = len(record.tool_calls_detail)
        record.final_answer, record.final_answer_dropped_chars = final_answer_split(
            message_content
        )
        record.final_answer_dropped_prose = dropped_prose_blocks(message_content)
    else:
        # the flag is only meaningful for a turn that finished; an aborted one never got
        # far enough for the absence of the marker to mean anything
        record.hit_max_iterations = False

    if saw_script_chunk:
        executed_ok = script_categories.get(OUTCOME_EXECUTED_OK, 0)
        executed_failed = script_categories.get(OUTCOME_EXECUTED_FAILED, 0)
        model_rejected = script_categories.get(OUTCOME_MODEL_REJECTED, 0)
        record.script_runs = executed_ok + executed_failed
        record.script_attempts = record.script_runs + model_rejected
        record.script_failures = executed_failed + model_rejected
        record.script_infra_errors = script_categories.get(OUTCOME_INFRA, 0)
        record.script_budget_exceeded = script_categories.get(OUTCOME_DISPUTED, 0)
        record.script_outcomes = script_outcomes
        record.retry_loops_script = retry_loops_by_cause.get(OUTCOME_EXECUTED_FAILED, 0)
        record.retry_loops_disputed = retry_loops_by_cause.get(OUTCOME_DISPUTED, 0)
        record.retry_loops_infra = retry_loops_by_cause.get(OUTCOME_INFRA, 0)
        record.retry_loops = (
            record.retry_loops_script + record.retry_loops_disputed + record.retry_loops_infra
        )

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
    capture_thinking: bool = False,
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
            capture_thinking=capture_thinking,
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
    capture_thinking: bool = False,
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
                capture_thinking=capture_thinking,
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
    ("model_ms_total", "model_ms_total"),
    ("slowest_iteration_ms", "slowest_iteration_ms"),
    ("input_tokens", "input_tokens"),
    ("output_tokens", "output_tokens"),
    ("cached_input_tokens", "cached_input_tokens"),
    ("context_tokens_last_iteration", "context_tokens_last_iteration"),
    ("cost_usd", "cost_usd"),
    ("cost_usd_min", "cost_usd_min"),
    ("cost_usd_max", "cost_usd_max"),
)

# per-iteration rather than per-turn: n here is iterations, not turns. Every one of these
# has its OWN n — tool_phase_ms is None for every iteration that ended the turn, so its
# sample is strictly smaller than model_ms's at the same index, and printing one n for the
# row would quote a p90 over a single observation as if it stood on the row's count.
_ITERATION_DISTRIBUTIONS = (
    "iteration_ms",
    "model_ms",
    "pre_model_ms",
    "tool_phase_ms",
    "segment_ms",
)


def _iteration_timing(records: list[TurnRecord]) -> dict[str, Any]:
    """Distributions over ITERATIONS, plus the same broken out by iteration index.

    The by-index table is the point of the whole exercise: a turn total cannot say whether
    the cost sits in the first roundtrip or the seventh, and the epic's own measurement says
    context roughly triples between them. Iteration indices are pooled across turns, so a
    row's n falls away as the tail thins — which is why each column carries its own n and
    its own unreliable-percentile flags into the printed table.
    """
    rows = [row for r in records for row in r.iterations_detail]
    by_index: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_index.setdefault(int(row.get("iteration") or 0), []).append(row)
    reported_attempts = [row for row in rows if row.get("model_attempts") is not None]
    return {
        "iterations": len(rows),
        # how many iterations' model_ms is retry-inflated, and over how many the question
        # could even be asked. Both, because "0 retries" and "the server never said" are
        # different statements about the same column.
        "iterations_reporting_model_attempts": len(reported_attempts),
        "iterations_with_model_retries": sum(
            1 for row in reported_attempts if (row["model_attempts"] or 1) > 1
        ),
        "distributions": {
            name: distribution([row.get(name) for row in rows])
            for name in _ITERATION_DISTRIBUTIONS
        },
        "by_iteration_index": [
            {
                "iteration": index,
                "turns": len(group),
                **{
                    name: distribution([row.get(name) for row in group])
                    for name in _ITERATION_DISTRIBUTIONS
                },
            }
            for index, group in sorted(by_index.items())
        ],
    }


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
        "iteration_timing": _iteration_timing(ok),
        # how many of this arm's ok turns got an exact USD figure and how many only a
        # bracket. A run that mixes the two must say so rather than average across them.
        "cost_basis": {
            basis: sum(1 for r in ok if r.cost_basis == basis)
            for basis in (COST_BASIS_EXACT, COST_BASIS_INTERVAL, COST_BASIS_UNPRICED)
        },
    }

    measured_scripts = [r for r in records if r.script_runs is not None]
    if measured_scripts:
        attempts = sum(r.script_attempts or 0 for r in measured_scripts)
        failures = sum(r.script_failures or 0 for r in measured_scripts)
        outcomes: dict[str, int] = {}
        for r in measured_scripts:
            for shape, count in (r.script_outcomes or {}).items():
                outcomes[shape] = outcomes.get(shape, 0) + count
        summary["script_runs"] = sum(r.script_runs or 0 for r in measured_scripts)
        summary["script_attempts"] = attempts
        summary["script_failures"] = failures
        summary["script_failure_rate"] = (failures / attempts) if attempts else None
        summary["script_failure_rate_definition"] = SCRIPT_FAILURE_RATE_DEFINITION
        summary["retry_loops"] = sum(r.retry_loops or 0 for r in measured_scripts)
        summary["retry_loops_script"] = sum(r.retry_loops_script or 0 for r in measured_scripts)
        summary["retry_loops_disputed"] = sum(
            r.retry_loops_disputed or 0 for r in measured_scripts
        )
        summary["retry_loops_infra"] = sum(r.retry_loops_infra or 0 for r in measured_scripts)
        summary["script_infra_errors"] = sum(
            r.script_infra_errors or 0 for r in measured_scripts
        )
        summary["script_budget_exceeded"] = sum(
            r.script_budget_exceeded or 0 for r in measured_scripts
        )
        summary["script_outcomes"] = dict(sorted(outcomes.items()))
    else:
        for key in (
            "script_runs",
            "script_attempts",
            "script_failures",
            "script_failure_rate",
            "retry_loops",
            "retry_loops_script",
            "retry_loops_disputed",
            "retry_loops_infra",
            "script_infra_errors",
            "script_budget_exceeded",
        ):
            summary[key] = None
        summary["script_outcomes"] = {}
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


RESOLVED_TOOLS_PATH = "/chat/v1/tools/resolved"


class ArmResolutionError(RuntimeError):
    """An arm does not name a profile the target server knows."""


class RateLimitedError(RuntimeError):
    """The chat service refused a turn with 429. Run-invalidating, not per-turn."""


async def resolve_arm_tools(
    client: httpx.AsyncClient, base_url: str, arms: tuple[str, str]
) -> dict[str, Any]:
    """Ask the SERVER what each arm actually resolves to, before spending anything.

    Two things this prevents, both of which produce a run that looks fine and means nothing:

    1. A MISSPELLED ARM. `get_anthropic_tools` degrades an unrecognised profile to
       general-only rather than raising — deliberately, because the value comes back from
       rows written by older clients — so `--arm-a nocod` silently yields an 18-tool
       baseline and reports plausible numbers against it. `known_profile: false` is fatal
       here: refusing to start costs nothing, and the alternative is discovering it after
       the spend.
    2. A SERVER RUNNING OLDER CODE. The profile is resolved in the chat service's process,
       which imports the definitions at startup. A profile added on disk but not yet loaded
       by a running server resolves through the same silent fallback, and locally that is
       one forgotten restart away.

    The resolved counts and names are recorded in the report so a saved run PROVES what each
    arm was given, rather than leaving it to be re-derived later from a tree that has since
    moved. `count` is local tools only; external and RAG surfaces resolve separately.

    A server without the endpoint (older build) is a WARNING, not a failure — the run is
    still valid, it just cannot carry the proof. Silence would be the wrong trade in the
    other direction.
    """
    out: dict[str, Any] = {}
    unknown: list[str] = []
    for arm in arms:
        params = {} if arm == ALL_TOOLS_ARM else {"tool_profile": arm}
        try:
            resp = await client.get(f"{base_url}{RESOLVED_TOOLS_PATH}", params=params)
        except httpx.HTTPError as exc:
            logger.warning("could not resolve arm %r against %s: %s", arm, base_url, exc)
            out[arm] = {"error": str(exc)}
            continue
        if resp.status_code == 404:
            logger.warning(
                "%s has no %s endpoint, so this report cannot record what each arm was "
                "given. The run is still valid; verify the arms by hand.",
                base_url,
                RESOLVED_TOOLS_PATH,
            )
            return {"unavailable": f"{base_url} has no {RESOLVED_TOOLS_PATH}"}
        if resp.status_code != 200:
            logger.warning("resolving arm %r returned HTTP %s", arm, resp.status_code)
            out[arm] = {"error": f"HTTP {resp.status_code}"}
            continue
        data = resp.json()
        out[arm] = {
            "count": data.get("count"),
            "known_profile": data.get("known_profile"),
            "names": data.get("names"),
        }
        if data.get("known_profile") is False:
            unknown.append(arm)
        else:
            logger.info("arm %r resolves to %s local tools", arm, data.get("count"))
    if unknown:
        raise ArmResolutionError(
            f"{base_url} does not recognise these arm profiles: {', '.join(unknown)}. "
            "An unrecognised profile silently degrades to general-only (18 tools), so this "
            "run would have measured a surface you did not intend. Check the spelling, and "
            "check the server has been restarted since the profile was added."
        )
    return out


async def _dry_run_resolve(base_url: str, arms: tuple[str, str]) -> dict[str, Any]:
    """resolve_arm_tools with its own short-lived client, for the --dry-run path."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        return await resolve_arm_tools(client, base_url, arms)


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


def _iteration_timing_lines(summary: dict[str, Any]) -> list[str]:
    timing = summary.get("iteration_timing") or {}
    by_index = timing.get("by_iteration_index") or []
    if not by_index:
        return []
    columns = (("model", "model_ms"), ("tools", "tool_phase_ms"), ("iter", "iteration_ms"))
    header = f"    {'iter':>4}"
    for label, _metric in columns:
        header += f"{label + ' n':>9}{label + ' p50':>12}{label + ' p90':>12}"
    lines = [
        f"  per-iteration timeline ({timing['iterations']} iterations across "
        f"{summary['turns_ok']} turns; ms)",
        header,
    ]
    flagged = False
    for row in by_index:
        # each column carries its own n: tool_phase_ms is None for every turn that ENDED at
        # this index, so quoting the row's iteration count beside it would present a p90
        # over one observation as if it rested on all of them
        cells = f"    {row['iteration']:>4}"
        for _label, metric in columns:
            d = row[metric]
            if d["n"] == 0:
                cells += f"{0:>9}{'n/a':>12}{'n/a':>12}"
                continue
            cells += f"{d['n']:>9}"
            for p in ("p50", "p90"):
                text = _fmt(d[p], 1)
                if p in d["unreliable_percentiles"]:
                    # distribution() already knows this percentile is indistinguishable from
                    # the maximum at this n; every other table in this report says so
                    text += "*"
                    flagged = True
                cells += f"{text:>12}"
        lines.append(cells)
    lines.append(
        "    'tools' is the tool phase that FOLLOWED that iteration's model call, so the "
        "last iteration of a turn has no observation there — for a turn that ended by "
        "answering that phase is a measured ZERO, which 'iter' uses but this column "
        "deliberately does not record;"
    )
    lines.append(
        "    'iter' is that iteration's own roundtrip, model + its own tools — NOT the gap "
        "between usage chunks (that is segment_ms in the JSON, which carries the PRECEDING "
        "iteration's tool phase)."
    )
    lines.append(
        "    'model' is the server's whole model call, so it includes transient-error retry "
        "backoff and SSE delivery, not provider latency alone: "
        f"{timing.get('iterations_with_model_retries')} of "
        f"{timing.get('iterations_reporting_model_attempts')} iterations reporting "
        "model_attempts needed more than one attempt."
    )
    if flagged:
        lines.append(
            "    * that percentile is not distinguishable from the column's maximum at that "
            "column's n; see unreliable_percentiles in the JSON."
        )
    return lines


def _cost_basis_line(summary: dict[str, Any]) -> str:
    basis = summary.get("cost_basis") or {}
    parts = ", ".join(f"{name}={basis.get(name, 0)}" for name in basis)
    if basis.get(COST_BASIS_INTERVAL):
        parts += "  <- interval turns are NOT priced exactly; read cost_usd_min/max for them"
    return f"  cost basis (turns)            : {parts}"


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
    rate_text = "n/a (0 script attempts)" if rate is None else f"{rate:.3f}"
    outcomes = summary.get("script_outcomes") or {}
    lines = [
        f"  script attempts={summary['script_attempts']} "
        f"(executed={summary['script_runs']}) failures={summary['script_failures']} "
        f"rate={rate_text}",
        f"      rate is {SCRIPT_FAILURE_RATE_DEFINITION}",
        "      by shape (verbatim status/error_type): "
        + (", ".join(f"{shape}={count}" for shape, count in outcomes.items()) or "none"),
        f"      sandbox faults (not script failures, not in the rate)"
        f"={summary.get('script_infra_errors')}   "
        f"TurnBudgetExceeded (classified by neither, not in the rate)"
        f"={summary.get('script_budget_exceeded')}",
        f"  retry loops={summary['retry_loops']} "
        f"(after a script failure={summary.get('retry_loops_script')}, "
        f"after TurnBudgetExceeded={summary.get('retry_loops_disputed')}, "
        f"after a sandbox fault={summary.get('retry_loops_infra')})",
    ]
    return lines


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
        lines.append(_cost_basis_line(s))
        lines.extend(_script_lines(s))
        lines.extend(_iteration_timing_lines(s))
        lines.append("")

    if report.get("judging"):
        # imported here for the same cycle reason as in replay_turn
        from genetics_mcp_server.scripts.pairwise_judge import format_judging

        lines.extend(format_judging(report["judging"]))
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
        lines.append(_cost_basis_line(s))
        lines.extend(_script_lines(s))
        lines.extend(_iteration_timing_lines(s))
        lines.append("")

    lines.append("-" * 78)
    interval_turns = sum(
        (report["per_arm"][arm].get("cost_basis") or {}).get(COST_BASIS_INTERVAL, 0)
        for arm in report["arms"]
    )
    exact_turns = sum(
        (report["per_arm"][arm].get("cost_basis") or {}).get(COST_BASIS_EXACT, 0)
        for arm in report["arms"]
    )
    lines.append(
        f"cost_usd is EXACT for {exact_turns} turn(s): the usage stream reported cache "
        "reads and cache creations separately, and they are priced separately."
    )
    if interval_turns:
        lines.append(
            f"cost_usd is UNAVAILABLE for {interval_turns} turn(s) whose usage chunks "
            "carried no cache_read/cache_create split (an older server, or a provider path "
            "that does not report it). Those turns have only the interval cost_usd_min .. "
            "cost_usd_max: the min prices every cached token as a cache read and the max as "
            "a cache creation, and those two differ >12x in price. The BRACKET is narrower "
            "than that — output tokens and uncached input are in both endpoints — but it is "
            "wide enough to matter. They are NOT in the cost_usd distribution — do not read "
            "a mixed run's cost_usd as covering every turn."
        )
    judging = report.get("judging")
    if judging:
        actual = judging.get("cost_actual") or {}
        usd = actual.get("usd")
        lines.append(
            "judging cost is a SEPARATE line item and is NOT in any cost_usd figure above: "
            f"{judging['judge_calls']} judge call(s), {actual.get('input_tokens', 0):,} in + "
            f"{actual.get('output_tokens', 0):,} out tokens = "
            + ("NOT PRICED" if usd is None else f"${usd:,.2f}")
            + ". The arms' USD above is what the ANSWERS cost; this is what grading them cost."
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
    capture_thinking: bool = False,
) -> dict[str, Any]:
    cases = load_cases(dataset, limit)
    run_id = uuid.uuid4().hex[:8]
    headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else {}
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async with httpx.AsyncClient(headers=headers, timeout=timeout) as client:
        arm_tools = await resolve_arm_tools(client, base_url, arms)

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
                    capture_thinking=capture_thinking,
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
            "capture_thinking": capture_thinking,
            # what the SERVER said each arm resolves to, asked before the run rather than
            # re-derived afterwards from a tree that may have moved
            "arm_tools": arm_tools,
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
    parser.add_argument(
        "--capture-thinking",
        action="store_true",
        help="ask the server to stream each iteration's summarized reasoning and record it "
        "per turn, so benchmark_scorecard --markdown can show the thinking behind each "
        "tool call. Off by default: it multiplies the report's size and is not needed for "
        "any metric — thinking tokens are already inside output_tokens either way",
    )
    parser.add_argument("--output", type=Path, default=None, help="write the JSON report here")
    parser.add_argument("--dry-run", action="store_true", help="resolve the plan, issue no requests")
    parser.add_argument(
        "--judge",
        action="store_true",
        help="after the benchmark, judge the matched pairs blind and pairwise "
        "(genetics-results-suite-4h6.72). OFF by default and independently skippable: cost "
        "and latency are measured with no judge call at all. This is Opus-5 spend ON TOP OF "
        "the benchmark's, doubled by judging every pair in both presentation orders; the "
        "estimate is printed before the first call. A saved report can also be judged later "
        "with `python -m genetics_mcp_server.scripts.pairwise_judge --report <file>`.",
    )
    parser.add_argument("--judge-model", default=None, help="judge model (default claude-opus-5)")
    parser.add_argument("--judge-concurrency", type=int, default=4, help="judge calls in flight")
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

    from genetics_mcp_server.scripts import pairwise_judge

    judge_model = args.judge_model or pairwise_judge.DEFAULT_JUDGE_MODEL

    if args.dry_run:
        cases = load_cases(args.dataset, args.limit)
        turns = sum(
            len((c.get("user_turns") or [])[: args.max_turns]) for c in cases
        )
        print(f"{len(cases)} cases, {turns} turns per arm, {turns * 2} model turns total")
        print(f"arms: {arms[0]} vs {arms[1]}   target: {args.base_url}")
        # the cheapest place a misspelled arm can possibly be caught, so catch it here too
        # rather than only in the paid path. Reaches the server but spends nothing.
        try:
            resolved = asyncio.run(_dry_run_resolve(args.base_url.rstrip("/"), arms))
        except ArmResolutionError as exc:
            print(f"\nERROR: {exc}", file=sys.stderr)
            return 2
        for arm in arms:
            info = resolved.get(arm) or {}
            if "count" in info:
                print(f"  {arm:>8} resolves to {info['count']} local tools")
        for i, c in enumerate(cases):
            order = arm_order_for_case(arms, i)
            print(f"  {i:>3} {c.get('session_id')} order={order[0]},{order[1]}")
        if args.judge:
            # the ceiling: every turn matching on both arms. Turns that fail on one arm are
            # not judged, so the real pair count can only be lower — an over-estimate is the
            # safe direction for a number a reader uses to decide whether to spend.
            for line in pairwise_judge.estimate_lines(
                pairwise_judge.estimate_judging_cost(None, judge_model, pair_count=turns)
            ):
                print(line)
        return 0

    try:
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
                capture_thinking=args.capture_thinking,
            )
        )
    except RateLimitedError as exc:
        turns_needed = sum(
            len((c.get("user_turns") or [])[: args.max_turns])
            for c in load_cases(args.dataset, args.limit)
        ) * 2
        print(
            f"\nABORTED: the chat service is rate-limiting this run.\n  {exc}\n\n"
            f"This plan needs {turns_needed} requests; the service's default is "
            "RATE_LIMIT_PER_HOUR=20 and RATE_LIMIT_PER_DAY=100, both counted per user. A "
            "run that hits this does not fail cleanly — the turns already replayed keep "
            "their cost while their later turns cascade to not_attempted, so the report "
            "looks complete and carries almost no matched pairs.\n\n"
            "Raise the limits above the whole plan and restart the chat service:\n"
            f"  RATE_LIMIT_PER_HOUR={max(2000, turns_needed * 2)} "
            f"RATE_LIMIT_PER_DAY={max(10000, turns_needed * 10)} "
            "scripts/dev-stack.sh up chat-api\n",
            file=sys.stderr,
        )
        return 2

    if args.judge:
        # after the benchmark and before the JSON is written, so the saved report carries the
        # judging block; a failure here still leaves the cost/latency report printed below
        try:
            asyncio.run(
                pairwise_judge.judge_report(
                    report, model=judge_model, concurrency=args.judge_concurrency
                )
            )
        except Exception as e:
            logger.error("judging failed (%s: %s); the benchmark report is unaffected", type(e).__name__, e)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("wrote %s", args.output)

    print(format_summary(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
