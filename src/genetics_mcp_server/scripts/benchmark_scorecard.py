"""Per-question scorecard for a saved replay_benchmark report.

`replay_benchmark` reports DISTRIBUTIONS, which answer "which arm is cheaper" and cannot
answer "on which questions". This renders the second view: one row per replayed case, both
arms side by side, over the four numbers a rollout decision actually turns on —

    wall clock to done · USD · tool calls · the judge's verdict

Nothing here re-measures or re-judges anything. It reads the JSON written by
`replay_benchmark --output` (optionally already judged, in place, by `pairwise_judge`), so
it is free to run, re-run, and run against an old report.

WHY A CASE CAN BE UNCOMPARABLE, and why that is shown rather than smoothed. A case whose
turns did not all succeed on BOTH arms is NOT summed into a comparable row: an arm that
aborted on turn 2 of 3 spent less time, less money and fewer tool calls than one that
finished, and presenting those totals side by side would score failure as efficiency. Such
rows are marked and excluded from the TOTALS line, and the reason is printed. This mirrors
`replay_benchmark`'s own matched analysis, which exists for the same reason.

The judge column is a PAIRWISE verdict, not a score. `pairwise_judge` compares the two arms'
final answers blind and in both presentation orders and reports a winner or a tie; there is
no absolute per-arm quality number to put in a column, and inventing one by counting wins as
points would imply a scale the judge never produced. A case with several turns therefore
shows a tally across its turns.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# a turn only contributes to a comparable row if it actually completed. Everything else --
# error, timeout, incomplete, not_attempted -- makes its whole case uncomparable.
OK = "ok"


def _turns_by_case(report: dict[str, Any], arms: list[str]) -> dict[str, dict[str, list[dict]]]:
    """case_id -> arm -> its turns, in turn order."""
    out: dict[str, dict[str, list[dict]]] = {}
    for t in report.get("turns", []):
        if t.get("arm") not in arms:
            continue
        out.setdefault(t["case_id"], {a: [] for a in arms})[t["arm"]].append(t)
    for per_arm in out.values():
        for turns in per_arm.values():
            turns.sort(key=lambda t: t.get("turn_index", 0))
    return out


def _case_totals(turns: list[dict]) -> dict[str, Any]:
    """Sum one arm's turns for one case. `exact_cost` is False if anything was bracketed."""
    seconds = sum((t.get("ms_to_done") or 0.0) for t in turns) / 1000.0
    tool_calls = sum((t.get("tool_calls") or 0) for t in turns)
    cost = 0.0
    exact = True
    priced = True
    for t in turns:
        if t.get("cost_usd") is not None:
            cost += t["cost_usd"]
            continue
        lo, hi = t.get("cost_usd_min"), t.get("cost_usd_max")
        if lo is None or hi is None:
            # cost.py refuses to price an unrecognised model rather than guessing it as
            # Sonnet, so this is "not priced", never "free"
            priced = False
            continue
        cost += (lo + hi) / 2.0
        exact = False
    return {
        "seconds": seconds,
        "tool_calls": tool_calls,
        "cost": cost if priced else None,
        "exact_cost": exact,
        "turns": len(turns),
    }


def _blockers(per_arm: dict[str, list[dict]], arms: list[str]) -> list[str]:
    """Why this case is not comparable. Empty means it is."""
    reasons = []
    counts = {a: len(per_arm[a]) for a in arms}
    if len(set(counts.values())) != 1:
        reasons.append("turn counts differ (" + ", ".join(f"{a}={counts[a]}" for a in arms) + ")")
    for a in arms:
        bad = [t for t in per_arm[a] if t.get("status") != OK]
        if bad:
            shapes = sorted({str(t.get("status")) for t in bad})
            reasons.append(f"{a}: {len(bad)} turn(s) {'/'.join(shapes)}")
    return reasons


def _judge_by_case(report: dict[str, Any]) -> tuple[dict[str, list[dict]], bool]:
    """case_id -> its per-turn verdicts. Second value is whether the report was judged."""
    judging = report.get("judging") or {}
    pairs = judging.get("pairs")
    if pairs is None:
        # either never judged, or judged by a build that did not persist per-pair rows
        return {}, bool(judging)
    by_case: dict[str, list[dict]] = {}
    for p in pairs:
        by_case.setdefault(p["case_id"], []).append(p)
    return by_case, True


def _judge_cell(verdicts: list[dict], arms: list[str]) -> str:
    if not verdicts:
        return "-"
    tally: dict[str, int] = {}
    for v in verdicts:
        key = v["winner"] if v.get("winner") else "tie"
        tally[key] = tally.get(key, 0) + 1
    order = [a for a in arms if a in tally] + [k for k in tally if k not in arms]
    cell = " / ".join(f"{k} {tally[k]}" for k in order)
    if any(v.get("arm_identifiable") for v in verdicts):
        cell += " !"
    return cell


def _fmt_cost(total: dict[str, Any]) -> str:
    if total["cost"] is None:
        return "n/p"
    return f"{total['cost']:.2f}" + ("" if total["exact_cost"] else "~")


def render(report: dict[str, Any], csv: bool = False) -> str:
    arms = list(report.get("arms") or [])
    if len(arms) != 2:
        return f"expected 2 arms, report has {arms!r}"
    a, b = arms
    by_case = _turns_by_case(report, arms)
    judge, judged = _judge_by_case(report)

    rows = []
    for case_id in sorted(by_case):
        per_arm = by_case[case_id]
        blockers = _blockers(per_arm, arms)
        ta, tb = _case_totals(per_arm[a]), _case_totals(per_arm[b])
        rows.append(
            {
                "case": case_id,
                "a": ta,
                "b": tb,
                "judge": _judge_cell(judge.get(case_id, []), arms),
                "blockers": blockers,
            }
        )

    if csv:
        out = [f"case,{a}_seconds,{b}_seconds,{a}_usd,{b}_usd,{a}_tools,{b}_tools,judge,comparable"]
        for r in rows:
            out.append(
                ",".join(
                    [
                        r["case"],
                        f"{r['a']['seconds']:.1f}",
                        f"{r['b']['seconds']:.1f}",
                        _fmt_cost(r["a"]),
                        _fmt_cost(r["b"]),
                        str(r["a"]["tool_calls"]),
                        str(r["b"]["tool_calls"]),
                        '"' + r["judge"] + '"',
                        "no" if r["blockers"] else "yes",
                    ]
                )
            )
        return "\n".join(out)

    w = max([len(r["case"]) for r in rows] + [4])
    lines = [
        f"{'':{w}}   {'seconds':^15}  {'USD':^15}  {'tool calls':^13}",
        f"{'case':{w}}   {a:>7} {b:>7}  {a:>7} {b:>7}  {a:>6} {b:>6}   judge",
        "-" * (w + 3 + 15 + 2 + 15 + 2 + 13 + 3 + 20),
    ]
    for r in rows:
        mark = " *" if r["blockers"] else "  "
        lines.append(
            f"{r['case']:{w}}{mark} {r['a']['seconds']:>7.1f} {r['b']['seconds']:>7.1f}  "
            f"{_fmt_cost(r['a']):>7} {_fmt_cost(r['b']):>7}  "
            f"{r['a']['tool_calls']:>6} {r['b']['tool_calls']:>6}   {r['judge']}"
        )

    clean = [r for r in rows if not r["blockers"]]
    if clean:
        sa = sum(r["a"]["seconds"] for r in clean)
        sb = sum(r["b"]["seconds"] for r in clean)
        pa = [r["a"] for r in clean if r["a"]["cost"] is not None]
        pb = [r["b"] for r in clean if r["b"]["cost"] is not None]
        ca = sum(x["cost"] for x in pa) if pa else None
        cb = sum(x["cost"] for x in pb) if pb else None
        na = sum(r["a"]["tool_calls"] for r in clean)
        nb = sum(r["b"]["tool_calls"] for r in clean)
        lines.append("-" * (w + 3 + 15 + 2 + 15 + 2 + 13 + 3 + 20))
        lines.append(
            f"{'TOTAL (' + str(len(clean)) + ' comparable)':{w}}   {sa:>7.1f} {sb:>7.1f}  "
            f"{(f'{ca:.2f}' if ca is not None else 'n/p'):>7} "
            f"{(f'{cb:.2f}' if cb is not None else 'n/p'):>7}  {na:>6} {nb:>6}"
        )

    notes = []
    # a run the server refused partway through is not a slightly incomplete run: the cases
    # replayed before the refusal keep their cost while everything after cascades, so the
    # report looks complete and compares almost nothing. Say so before any number is read.
    limited = [
        t for t in report.get("turns", [])
        if t.get("status") == "error" and "429" in str(t.get("error") or "")
    ]
    if limited:
        comparable = len([r for r in rows if not r["blockers"]])
        notes.append(
            f"!! THIS RUN WAS RATE-LIMITED: {len(limited)} turn(s) came back HTTP 429, so "
            f"only {comparable} of {len(rows)} cases are comparable and the numbers below "
            "are not a benchmark. Raise RATE_LIMIT_PER_HOUR / RATE_LIMIT_PER_DAY on the "
            "chat service above the whole plan, restart it, and re-run."
        )
    skipped = [r for r in rows if r["blockers"]]
    if skipped:
        notes.append(
            f"* {len(skipped)} case(s) NOT comparable and excluded from TOTAL — an arm that "
            "did not finish spent less of everything, and summing it beside one that did "
            "would score failure as efficiency:"
        )
        notes += [f"    {r['case']}: {'; '.join(r['blockers'])}" for r in skipped]
    if any(not r[k]["exact_cost"] for r in rows for k in ("a", "b")):
        notes.append(
            "~ interval-priced: the usage stream carried no cache split for some turns, so "
            "that figure is the midpoint of the min/max bracket, not a measurement."
        )
    if any("n/p" == _fmt_cost(r[k]) for r in rows for k in ("a", "b")):
        notes.append("n/p = not priced (unrecognised model); cost.py refuses to guess.")
    if not judged:
        notes.append(
            "judge column empty: this report has not been judged. Run "
            "`python -m genetics_mcp_server.scripts.pairwise_judge --report <file>`."
        )
    elif not judge:
        notes.append(
            "judge column empty: this report was judged by a build that did not persist "
            "per-pair verdicts. Re-judge it to populate them."
        )
    if any(" !" in r["judge"] for r in rows):
        notes.append(
            "! the judge could identify an arm from the answer text on at least one turn of "
            "that case, so the blinding did not hold there."
        )
    notes.append(
        "judge cells are PAIRWISE verdicts tallied across the case's turns, not scores: "
        "the judge picks a winner or a tie per turn, it does not rate an arm on a scale."
    )
    return "\n".join(lines + [""] + notes)


def _arg_preview(value: Any, width: int) -> str:
    """One line of an argument, marked when shortened so nothing reads as complete."""
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def render_tool_calls(report: dict[str, Any], width: int = 100, case: str | None = None) -> str:
    """The ordered call sequence per case and arm, for reading a result by hand.

    Previews are ELIDED, and say so with a `…`. The untruncated arguments — including
    `run_analysis`'s whole script — are in the report:

        jq '.turns[] | select(.case_id=="<case>") | {arm, turn_index, tool_calls_detail}' report.json
    """
    arms = list(report.get("arms") or [])
    by_case = _turns_by_case(report, arms)
    out: list[str] = []
    for case_id in sorted(by_case):
        if case and case_id != case:
            continue
        out.append(f"\n=== {case_id}")
        for arm in arms:
            for t in by_case[case_id][arm]:
                calls = t.get("tool_calls_detail")
                turn = f"  [{arm}] turn {t.get('turn_index')}"
                if t.get("status") != OK:
                    out.append(f"{turn}: {t.get('status')} — {t.get('error') or 'no detail'}")
                    continue
                if calls is None:
                    out.append(f"{turn}: no tool_calls_detail (report predates it)")
                    continue
                if not calls:
                    out.append(f"{turn}: answered with no tool calls")
                    continue
                out.append(f"{turn}: {len(calls)} call(s)")
                for c in calls:
                    args = c.get("input") or {}
                    if isinstance(args, dict) and args:
                        for k, v in args.items():
                            out.append(f"      {c['seq']}. {c.get('name')}  {k}={_arg_preview(v, width)}")
                            break
                        for k, v in list(args.items())[1:]:
                            out.append(f"         {' ' * len(str(c['seq']))}   {k}={_arg_preview(v, width)}")
                    else:
                        out.append(f"      {c['seq']}. {c.get('name')}  (no arguments)")
    if not out:
        return "no matching cases"
    out.append(
        "\nArguments above are elided to fit; `…` marks it. Full, untruncated arguments:\n"
        "  jq '.turns[] | select(.case_id==\"<case>\") | {arm, turn_index, tool_calls_detail}' <report>"
    )
    return "\n".join(out)


def _secs(ms: Any) -> str:
    """Milliseconds as seconds, or `?` when the wire never carried the measurement."""
    if not isinstance(ms, (int, float)) or isinstance(ms, bool):
        return "?"
    return f"{ms / 1000.0:.1f}s"


def _tool_phase_total(turn: dict) -> tuple[float | None, bool]:
    """Summed per-iteration tool phase, and whether any iteration's was unmeasured.

    NOT the sum of per-call durations, which nothing measures. An iteration's tools are
    dispatched with `asyncio.gather`, so its phase is roughly the SLOWEST call plus dispatch
    and result rendering, not the total work done in it. Two arms with the same tool-phase
    total can therefore have made very different numbers of calls, which is exactly the
    comparison this view exists to let a reader make by eye.
    """
    rows = turn.get("iterations_detail") or []
    measured = [r.get("tool_phase_ms") for r in rows if r.get("tool_phase_ms") is not None]
    # the final iteration's phase is None by construction (it answered rather than calling
    # tools), so its absence is not a gap; any OTHER None is
    gaps = any(r.get("tool_phase_ms") is None for r in rows[:-1])
    return (sum(measured) if measured else None), gaps


def _turn_summary(turn: dict) -> list[str]:
    """The numbers that explain a turn's wall clock, above its call list."""
    if turn.get("status") != OK:
        return [f"!! {turn.get('status')}: {str(turn.get('error') or 'no detail')[:200]}"]
    phase, gaps = _tool_phase_total(turn)
    lines = [
        f"{_secs(turn.get('ms_to_done'))} wall  ·  {turn.get('iterations')} iters  ·  "
        f"{turn.get('tool_calls')} calls",
        f"  model {_secs(turn.get('model_ms_total'))}  ·  tool phases "
        f"{_secs(phase)}{'+' if gaps else ''}"
        + (
            f"  ·  slowest iter {turn.get('slowest_iteration')} "
            f"({_secs(turn.get('slowest_iteration_ms'))})"
            if turn.get("slowest_iteration")
            else ""
        ),
    ]
    if turn.get("script_attempts"):
        outcomes = turn.get("script_outcomes") or {}
        shapes = ", ".join(f"{k} {v}" for k, v in sorted(outcomes.items()))
        lines.append(
            f"  scripts: {turn.get('script_attempts')} attempted, "
            f"{turn.get('script_failures') or 0} failed"
            + (f"  [{shapes}]" if shapes else "")
        )
    if turn.get("retry_loops"):
        # the iterations bought by a failed script rather than by the question: this is the
        # first thing to look at when the code arm is slower than the arm without it
        lines.append(f"  retry loops: {turn.get('retry_loops')} (extra roundtrips after a script failed)")
    if turn.get("hit_max_iterations"):
        lines.append("  !! hit the iteration ceiling — the answer is whatever it had by then")
    return lines


def _call_lines(call: dict, width: int, arg_lines: int) -> list[str]:
    """One call: its position, name, what it cost if that is knowable, then its arguments."""
    head = f"{call.get('seq')}. {call.get('name')}"
    if call.get("iteration") is not None:
        head = f"[i{call['iteration']}] " + head
    if call.get("script_duration_ms") is not None:
        status = call.get("script_status") or ("ok" if call.get("script_ok") else "?")
        head += f"  (sandbox {_secs(call['script_duration_ms'])}, {status})"
    elif call.get("script_status") is not None:
        head += f"  (sandbox {call['script_status']})"
    lines = [head]
    args = call.get("input")
    if not isinstance(args, dict) or not args:
        lines.append("     (no arguments)")
        return lines
    for key, value in args.items():
        text = value if isinstance(value, str) else json.dumps(value, default=str)
        # newlines are KEPT for script arguments: a script's shape is most of what makes it
        # readable, and rewrapping it into a paragraph destroys exactly that
        body = f"{key}={text}".replace("\t", "    ")
        wrapped: list[str] = []
        for para in body.split("\n"):
            para = para.rstrip()
            if not para:
                wrapped.append("")
                continue
            while para:
                wrapped.append(para[: width - 5])
                para = para[width - 5 :]
        if len(wrapped) > arg_lines:
            wrapped = wrapped[:arg_lines]
            wrapped[-1] = wrapped[-1][: width - 6] + "…"
        lines += [f"     {line}" for line in wrapped]
    return lines


def _two_column(left: list[str], right: list[str], colw: int) -> list[str]:
    out = []
    for i in range(max(len(left), len(right))):
        lhs = (left[i] if i < len(left) else "")[:colw]
        rhs = (right[i] if i < len(right) else "")[:colw]
        out.append(f"{lhs:<{colw}} | {rhs}".rstrip())
    return out


def render_transcript(
    report: dict[str, Any],
    width: int = 200,
    case: str | None = None,
    arg_lines: int = 6,
) -> str:
    """Both arms' call sequences for one case, aligned turn by turn in two columns.

    The scorecard says the code arm made more calls and took longer. It cannot say WHY, and
    the distributions cannot either — the answer is always in one case's sequence: a script
    that failed and was rewritten, a tool called twice with the same arguments, a wide
    parallel fan-out the baseline did serially. This is that sequence, with the per-turn
    timing beside it so a slow turn can be attributed to model time or to tool time.

    WHAT IS AND IS NOT MEASURED per call. `run_analysis` reports the sandbox's own wall
    clock, so those calls carry a duration. NOTHING ELSE ON THE WIRE DOES: every other call
    shows only its position and arguments, and the time it took is inside its iteration's
    tool phase together with every other call of that iteration, which ran in parallel with
    it. `[iN]` is the iteration a call belongs to, recorded from the stream's ordering; a
    report replayed before that was captured shows no `[iN]` and says so at the end.
    """
    arms = list(report.get("arms") or [])
    if len(arms) != 2:
        return f"expected 2 arms, report has {arms!r}"
    a, b = arms
    colw = max(28, (width - 3) // 2)
    by_case = _turns_by_case(report, arms)
    selected = [c for c in sorted(by_case) if not case or c == case]
    if not selected:
        return f"no case matching {case!r}; report has: " + ", ".join(sorted(by_case))

    out: list[str] = []
    saw_iteration = False
    saw_script_timing = False
    for case_id in selected:
        per_arm = by_case[case_id]
        out.append("")
        out.append("=" * (colw * 2 + 3))
        out.append(f"CASE {case_id}")
        blockers = _blockers(per_arm, arms)
        if blockers:
            out.append(f"  NOT COMPARABLE: {'; '.join(blockers)}")
        out.append(f"{a:^{colw}} | {b:^{colw}}")
        out.append("-" * (colw * 2 + 3))
        for index in range(max(len(per_arm[a]), len(per_arm[b]))):
            turns = {
                arm: (per_arm[arm][index] if index < len(per_arm[arm]) else None) for arm in arms
            }
            question = next(
                (t.get("user_question") for t in turns.values() if t and t.get("user_question")),
                "",
            )
            out.append("")
            out.append(f"--- turn {index}: {' '.join(str(question).split())[: width - 16]}")
            sides = []
            for arm in arms:
                turn = turns[arm]
                if turn is None:
                    sides.append(["(this arm has no turn here)"])
                    continue
                lines = _turn_summary(turn)
                calls = turn.get("tool_calls_detail")
                if calls is None:
                    lines.append("  (no tool_calls_detail — report predates it)")
                elif not calls:
                    lines.append("  answered with no tool calls")
                else:
                    lines.append("")
                    for call in calls:
                        saw_iteration = saw_iteration or call.get("iteration") is not None
                        saw_script_timing = (
                            saw_script_timing or call.get("script_duration_ms") is not None
                        )
                        lines += _call_lines(call, colw, arg_lines)
                sides.append(lines)
            out += _two_column(sides[0], sides[1], colw)

    notes = [
        "",
        "-" * (colw * 2 + 3),
        "Per-call durations are NOT measured for ordinary tools — an iteration's calls are "
        "dispatched in parallel and only the whole phase is timed, so `tool phases` is the "
        "sum of those phases, not of the calls.",
    ]
    if not saw_iteration:
        notes.append(
            "No call carries `[iN]`: this report predates the iteration attribution, so the "
            "call list is in order but does not show where each roundtrip began. Re-run to "
            "capture it."
        )
    if not saw_script_timing:
        notes.append(
            "No call carries a sandbox duration. `run_analysis` is the one tool that reports "
            "its own wall clock; either this arm ran none, or the report predates capturing it."
        )
    notes.append(
        f"Arguments are wrapped to {arg_lines} line(s) and elided with `…`. Whole values, "
        "including entire scripts:\n"
        "  jq '.turns[] | select(.case_id==\"<case>\") | {arm, turn_index, tool_calls_detail}' <report>"
    )
    return "\n".join(out + notes)


def _fence(text: str, lang: str = "") -> list[str]:
    """A fenced block whose fence is longer than any backtick run inside it.

    Arguments are reproduced VERBATIM here — this renderer's whole reason to exist is that
    the terminal views elide, and an elided script cannot be read. A value containing
    ```` ``` ```` would otherwise close the block early and silently reflow the rest of the
    document as prose, so the fence grows instead.
    """
    longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
    bar = "`" * max(3, longest + 1)
    return [f"{bar}{lang}", text.rstrip("\n"), bar]


def _md_arg(key: str, value: Any) -> list[str]:
    """One tool argument, whole. Long or multi-line values become blocks, short ones inline."""
    if isinstance(value, str):
        text, lang = value, "python" if key in ("script", "code") else ""
    else:
        text, lang = json.dumps(value, indent=2, default=str), "json"
    # a backtick in a short value would end the inline span mid-argument, so it takes the
    # block path too rather than being rendered as half a value plus stray markup
    if "\n" in text or len(text) > 160 or "`" in text:
        return [f"- `{key}`:"] + _fence(text, lang)
    return [f"- `{key}`: `{text}`"]


def _md_call(call: dict) -> list[str]:
    head = f"**{call.get('seq')}. `{call.get('name')}`**"
    if call.get("iteration") is not None:
        head = f"**[iteration {call['iteration']}] {call.get('seq')}. `{call.get('name')}`**"
    if call.get("script_duration_ms") is not None:
        status = call.get("script_status") or ("ok" if call.get("script_ok") else "?")
        head += f" — sandbox {_secs(call['script_duration_ms'])}, `{status}`"
    elif call.get("script_status") is not None:
        head += f" — sandbox `{call['script_status']}`"
    lines = [head, ""]
    args = call.get("input")
    if not isinstance(args, dict) or not args:
        return lines + ["- (no arguments)", ""]
    for key, value in args.items():
        lines += _md_arg(key, value)
    return lines + [""]


def _thinking_iterations(turn: dict) -> list[Any]:
    """The iterations this turn recorded reasoning for, in order, each once.

    `_md_thinking` renders every row of the iteration it is given, so visiting an iteration
    twice would print its reasoning twice.
    """
    seen: list[Any] = []
    for row in turn.get("thinking_detail") or []:
        if row.get("iteration") not in seen:
            seen.append(row.get("iteration"))
    return seen


def _md_thinking(turn: dict, iteration: Any) -> list[str]:
    """The reasoning recorded for one iteration, if the run captured any.

    Interleaved with the calls rather than collected at the top of the turn, because the
    point of having it is to read the reasoning immediately before the calls it produced.
    `iteration is None` selects the entries whose iteration the stream never carried, so a
    partially-attributed report still shows its text instead of dropping it.
    """
    rows = [
        r for r in (turn.get("thinking_detail") or []) if r.get("iteration") == iteration
    ]
    if not rows:
        return []
    label = f"thinking · iteration {iteration}" if iteration is not None else "thinking"
    lines = [f"<details><summary>{label}</summary>", ""]
    for row in rows:
        lines += [(row.get("text") or "").strip(), ""]
    return lines + ["</details>", ""]


def _md_turn_arm(turn: dict | None, arm: str) -> list[str]:
    lines = [f"#### arm `{arm}`", ""]
    if turn is None:
        return lines + ["*this arm has no turn here.*", ""]
    lines += [f"> {line}" for line in _turn_summary(turn)] + [""]
    if turn.get("status") != OK:
        return lines
    calls = turn.get("tool_calls_detail")
    if calls is None:
        lines += ["*no `tool_calls_detail` — this report predates it.*", ""]
    elif not calls:
        lines += ["*answered with no tool calls.*", ""]
        for iteration in _thinking_iterations(turn):
            lines += _md_thinking(turn, iteration)
    else:
        lines += ["<details><summary>" f"{len(calls)} tool call(s)" "</summary>", ""]
        seen_iterations: list[Any] = []
        for call in calls:
            iteration = call.get("iteration")
            if iteration not in seen_iterations:
                seen_iterations.append(iteration)
                lines += _md_thinking(turn, iteration)
            lines += _md_call(call)
        lines += ["</details>", ""]
        # the reasoning of iterations that called nothing — including the final one, which
        # answered — belongs to the turn just as much, and is where a turn that went wrong
        # without ever calling a tool explains itself
        for iteration in _thinking_iterations(turn):
            if iteration not in seen_iterations:
                seen_iterations.append(iteration)
                lines += _md_thinking(turn, iteration)
    dropped = turn.get("final_answer_dropped_chars") or 0
    if dropped:
        # the discarded text is not recoverable from the report — only its length was kept,
        # so saying "answer" without saying this would present a fragment as the whole reply
        lines += [
            f"*{dropped} character(s) of assistant prose written before the last tool call "
            "were discarded at capture (see `pairwise_judge.final_answer_split`) and are "
            "not in the report.*",
            "",
        ]
    lines += ["**Answer**", ""]
    answer = (turn.get("final_answer") or "").strip()
    lines += ([answer] if answer else ["*(empty)*"]) + [""]
    return lines


def _md_judge(pairs: list[dict], turn_index: int) -> list[str]:
    """The pairwise verdict for one turn, with both presentation orders' reasoning."""
    rows = [p for p in pairs if p.get("turn_index") == turn_index]
    if not rows:
        return []
    lines = ["#### judge", ""]
    for row in rows:
        winner = row.get("winner") or "tie"
        margin = f", margin {row['margin']}" if row.get("margin") else ""
        lines.append(f"- **{winner}**{margin} (`{row.get('outcome')}`)")
        for p in row.get("passes") or []:
            order = " vs ".join(p.get("order") or [])
            reason = " ".join(str(p.get("reason") or "").split())
            lines.append(f"    - shown *{order}* → **{p.get('verdict')}**: {reason}")
        if row.get("arm_identifiable"):
            lines.append("    - ⚠ the judge could name an arm from the answer text: blinding failed here.")
    return lines + [""]


def render_markdown(
    report: dict[str, Any], case: str | None = None, only_arm: str | None = None
) -> str:
    """Every case's conversation, both arms, as markdown — nothing elided.

    The terminal views (`--tools`, `--transcript`) are shaped by a column width and
    therefore truncate: a `run_analysis` script, the one argument most worth reading when an
    arm loses, is exactly the value that never fits. This view has no width, so the
    arguments are whole and the answers are verbatim, and the file can be read, diffed or
    handed to someone who was not at the terminal.

    WHAT A REPORT CANNOT GIVE THIS, stated here rather than discovered halfway down the
    file. `replay_benchmark` persists the user question, the assistant's tool_use blocks and
    the final answer; it does NOT persist **tool results**, and it keeps only the LENGTH of
    any assistant prose written before the last tool call. So this is the full conversation
    as recorded — question, calls, answer — not a wire log: what a tool returned is absent,
    and a turn that wrote a table before its last call shows the character count it lost.

    `only_arm` renders ONE arm's side of the same run: the questions in the same order, that
    arm's calls and answers, and the pairwise verdict that still names the other arm because
    there is no per-arm quality number to put in its place. Comparability is still stated on
    every case — the property belongs to the PAIR, and a one-arm file that dropped it would
    read as a clean run of an arm whose partner fell over.
    """
    arms = list(report.get("arms") or [])
    if len(arms) != 2:
        return f"expected 2 arms, report has {arms!r}"
    if only_arm is not None and only_arm not in arms:
        return f"no arm {only_arm!r} in this report; it has: " + ", ".join(arms)
    shown = [only_arm] if only_arm else arms
    cfg = report.get("config") or {}
    by_case = _turns_by_case(report, arms)
    judge, judged = _judge_by_case(report)
    selected = [c for c in sorted(by_case) if not case or c == case]
    if not selected:
        return f"no case matching {case!r}; report has: " + ", ".join(sorted(by_case))

    title = f"# Benchmark transcripts — run `{cfg.get('run_id', '?')}`"
    if only_arm:
        title += f" — arm `{only_arm}` only"
    out = [
        title,
        "",
        (f"- arm: `{only_arm}` (of `{arms[0]}` vs `{arms[1]}`)" if only_arm
         else f"- arms: `{arms[0]}` vs `{arms[1]}`")
        + "".join(
            f" · `{a}` = {(cfg.get('arm_tools') or {}).get(a, {}).get('count', '?')} tools"
            for a in arms
        ),
        f"- model: `{cfg.get('model') or 'deployment default'}`"
        f" · provider: `{cfg.get('provider') or 'deployment default'}`",
        f"- dataset: `{cfg.get('dataset', '?')}`",
        f"- cases: {len(selected)}"
        + ("" if judged else " · **not judged** — the judge sections are absent"),
        "",
        "Tool **arguments and answers are verbatim**. Tool *results* are not in the report, "
        "and assistant prose written before a turn's last tool call was discarded at "
        "capture with only its length kept — where that happened the turn says so.",
        "",
    ]
    if len(selected) > 1:
        out += ["## Contents", ""]
        out += [f"- [{c}](#{c.lower().replace(' ', '-')})" for c in selected] + [""]

    for case_id in selected:
        per_arm = by_case[case_id]
        out += ["---", "", f"## {case_id}", ""]
        blockers = _blockers(per_arm, arms)
        if blockers:
            out += [f"> **NOT COMPARABLE**: {'; '.join(blockers)}", ""]
        for index in range(max(len(per_arm[a]) for a in arms)):
            turns = {
                arm: (per_arm[arm][index] if index < len(per_arm[arm]) else None)
                for arm in arms
            }
            # the question comes from EITHER arm's record even in a one-arm file: both
            # replayed the same dataset turn, and the arm being rendered may be the one that
            # never got far enough to have recorded it
            question = next(
                (t.get("user_question") for t in turns.values() if t and t.get("user_question")),
                "",
            )
            out += [f"### Turn {index}", "", "**User**", ""]
            out += [f"> {line}" for line in str(question).splitlines() or [""]] + [""]
            for arm in shown:
                out += _md_turn_arm(turns[arm], arm)
            out += _md_judge(judge.get(case_id, []), index)

    return "\n".join(out).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Per-question side-by-side scorecard from a saved replay_benchmark report."
    )
    p.add_argument("report", type=Path, help="JSON written by replay_benchmark --output")
    p.add_argument("--csv", action="store_true", help="emit CSV instead of a table")
    p.add_argument(
        "--tools",
        action="store_true",
        help="print each case's ordered tool-call sequence with arguments, instead of the table",
    )
    p.add_argument(
        "--transcript",
        action="store_true",
        help="print both arms' call sequences side by side, per case and turn, with per-turn timing",
    )
    p.add_argument(
        "--markdown",
        type=Path,
        metavar="FILE",
        default=None,
        help="write every case's full conversation (both arms, nothing elided) to FILE as "
        "markdown, plus one file per arm beside it (FILE.<arm>.md); `-` writes the paired "
        "document to stdout and no per-arm files",
    )
    p.add_argument(
        "--case", default=None, help="with --tools/--transcript/--markdown, restrict to one case_id"
    )
    p.add_argument("--arg-width", type=int, default=100, help="argument preview width for --tools")
    p.add_argument("--width", type=int, default=200, help="total width for --transcript")
    p.add_argument(
        "--arg-lines", type=int, default=6, help="lines per argument value for --transcript"
    )
    args = p.parse_args(argv)
    try:
        report = json.loads(args.report.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"cannot read {args.report}: {exc}", file=sys.stderr)
        return 1
    if args.markdown:
        text = render_markdown(report, case=args.case)
        if not text.startswith("#"):
            # render_markdown returned a refusal (wrong arm count, no such case) rather than
            # a document; writing that into the requested file would leave a plausible-looking
            # artefact whose one line says nothing was rendered
            print(text, file=sys.stderr)
            return 1
        if str(args.markdown) == "-":
            # the per-arm files are files by definition; a single stream cannot be three of
            # them, so stdout gets the side-by-side document alone rather than a silent
            # concatenation that would read as one transcript
            print(text)
            print(
                "(per-arm files are written only when --markdown names a path)",
                file=sys.stderr,
            )
            return 0
        # one file per arm beside the paired one: the paired document answers "why did this
        # case go differently", and a single arm's file is what gets read on its own or
        # diffed against the same arm from another run, where the other arm's calls are noise
        written = [(args.markdown, text)] + [
            (
                args.markdown.with_name(
                    f"{args.markdown.stem}.{arm}{args.markdown.suffix or '.md'}"
                ),
                render_markdown(report, case=args.case, only_arm=arm),
            )
            for arm in (report.get("arms") or [])
        ]
        for path, body in written:
            try:
                path.write_text(body)
            except OSError as exc:
                print(f"cannot write {path}: {exc}", file=sys.stderr)
                return 1
            print(f"wrote {path} ({len(body):,} chars)")
        return 0
    if args.transcript:
        print(
            render_transcript(
                report, width=args.width, case=args.case, arg_lines=args.arg_lines
            )
        )
        return 0
    if args.tools:
        print(render_tool_calls(report, width=args.arg_width, case=args.case))
        return 0
    print(render(report, csv=args.csv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
