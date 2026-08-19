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
    p.add_argument("--case", default=None, help="with --tools, restrict to one case_id")
    p.add_argument("--arg-width", type=int, default=100, help="argument preview width for --tools")
    args = p.parse_args(argv)
    try:
        report = json.loads(args.report.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"cannot read {args.report}: {exc}", file=sys.stderr)
        return 1
    if args.tools:
        print(render_tool_calls(report, width=args.arg_width, case=args.case))
        return 0
    print(render(report, csv=args.csv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
