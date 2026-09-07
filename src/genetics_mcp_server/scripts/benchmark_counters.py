"""Per-arm mechanics of a benchmark run: what the model DID, not how good the answer was.

    python -m genetics_mcp_server.scripts.benchmark_counters report.json
    python -m genetics_mcp_server.scripts.benchmark_counters report.json --arm code
    python -m genetics_mcp_server.scripts.benchmark_counters before.json --baseline after.json

`benchmark_scorecard` is a side-by-side instrument and refuses a report with one arm, which
is correct — a rollout decision needs the comparison. This is the other half: numbers that
are meaningful for ONE arm on its own, so a single-arm run (`--arm-b none`) is readable and
so a prompt or image change can be measured against a previous run's recorded figures.

It exists because these counters were computed ad hoc while diagnosing run 9c6595ac, and a
measurement that is re-derived each time is not the same measurement twice. The baselines
below are that run, arm `code`, so a later run can be read against them without re-deriving
anything.

WHAT IS AND IS NOT MEASURED HERE. Everything here is mechanical: iterations, calls, what the
scripts contained. Answer QUALITY is not in it, and a single-arm run cannot supply it —
`pairwise_judge` needs an opposite arm. A run that improves every counter below and degrades
the answers looks like an improvement to this tool.
"""

import argparse
import collections
import json
import re
import sys
from pathlib import Path
from typing import Any

# arm `code` of run 9c6595ac (2026-08-27), the state before the cgr9 work. Recorded so a
# later run is read against a fixed point rather than against whatever is remembered.
#
# THESE ARE WHAT THE CODE IN THIS FILE COMPUTES ON THAT REPORT, and that is the only way they
# are allowed to be produced. Two of them differ from the figures quoted in the epic that
# commissioned this tool, which were computed by ad-hoc regexes during the diagnosis:
# `scripts_probing_schema` was 93 there and is 76 here (that pass also counted `LIMIT 1`,
# `LIMIT 5` and `.head(` as probing), and `scripts_reexecuting` was 28 there and is 31 here
# (that pass reset its seen-set per TURN; this one scopes it to the CASE, which is the
# question actually being asked). The other six are identical.
#
# The discrepancy is the reason this file exists. A counter re-derived at each reading is not
# the same counter twice, and a "delta" between two different definitions is fiction. Change a
# regex above and these numbers are stale: re-run this tool against the 9c6595ac report and
# replace them in the same commit.
# run 9c6595ac was serial. Everything above `wall_s` is concurrency-invariant — a script
# either probed the schema or did not, whatever else was in flight — but `wall_s` sums
# per-turn latency, and a concurrent run contends for the same local stack, so its per-turn
# figures are inflated by queueing that the baseline never paid. render() warns when the two
# runs disagree about this rather than leaving the reader to notice.
BASELINE_9C6595AC_CONCURRENCY = 1

BASELINE_9C6595AC = {
    "turns": 50,
    "turns_with_calls": 46,
    "discovery_first_turns": 22,
    "scripts": 172,
    "scripts_probing_schema": 76,
    "scripts_setting_pl_config": 78,
    "scripts_reexecuting": 31,
    "iterations": 238,
    "wall_s": 3664,
}

# a script whose only purpose is finding out what the data looks like
_DISCOVERY = re.compile(
    r"genetics\.schema\(|genetics\.datasets\(|genetics\.resources\(|"
    r"\.columns\b|\.dtypes\b|SELECT DISTINCT|\bdescribe\(|\bglimpse\("
)
_PL_CONFIG = re.compile(r"pl\.Config|POLARS_FMT|set_tbl_|set_fmt_")
# the retrieval operations a script performs, for the re-execution counter
_SQL_LITERAL = re.compile(r'"""(.*?)"""|\'\'\'(.*?)\'\'\'', re.S)
_SDK_CALL = re.compile(r"genetics\.([a-z_]+)\s*\(([^()]{0,200})\)")
_NON_RETRIEVAL = {"schema", "datasets", "resources", "sql", "show", "close", "configure"}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _retrievals(code: str) -> list[tuple[str, str]]:
    """(kind, key) for each retrieval the script performs. Approximate but consistent."""
    out: list[tuple[str, str]] = []
    for match in _SQL_LITERAL.finditer(code):
        body = match.group(1) or match.group(2)
        if re.search(r"\bselect\b", body, re.I):
            out.append(("sql", _norm(body)))
    for match in _SDK_CALL.finditer(code):
        name = match.group(1)
        if name not in _NON_RETRIEVAL:
            out.append(("sdk", _norm(f"{name}({match.group(2)})")))
    return out


_ROWS = [
    ("turns", "turns"),
    ("turns_with_calls", "turns that called a tool"),
    ("discovery_first_turns", "turns OPENING with a discovery-only script"),
    ("scripts", "run_analysis scripts"),
    ("scripts_probing_schema", "  ...that probe schema or return shape"),
    ("scripts_setting_pl_config", "  ...that set polars display config"),
    ("scripts_reexecuting", "  ...that re-run a retrieval already done this case"),
    ("iterations", "model iterations"),
    ("wall_s", "wall clock (s)"),
]


def counters(report: dict[str, Any], arm: str) -> dict[str, Any]:
    turns = [t for t in report.get("turns") or [] if t.get("arm") == arm]
    per_case: dict[str, list[str]] = collections.defaultdict(list)
    out = collections.Counter()
    wall = 0.0
    for turn in sorted(turns, key=lambda t: (t.get("case_id") or "", t.get("turn_index") or 0)):
        out["turns"] += 1
        out["iterations"] += turn.get("iterations") or 0
        wall += (turn.get("ms_to_done") or 0) / 1000
        scripts = [
            c["input"].get("code", "")
            for c in (turn.get("tool_calls_detail") or [])
            if c.get("name") == "run_analysis"
        ]
        if turn.get("tool_calls"):
            out["turns_with_calls"] += 1
        # a turn OPENING on a discovery-only script is the expensive shape: the model pays a
        # whole round trip before the work starts
        if scripts and _DISCOVERY.search(scripts[0]) and len(scripts[0]) < 500:
            out["discovery_first_turns"] += 1
        for code in scripts:
            out["scripts"] += 1
            if _DISCOVERY.search(code):
                out["scripts_probing_schema"] += 1
            if _PL_CONFIG.search(code):
                out["scripts_setting_pl_config"] += 1
            seen = per_case[turn.get("case_id") or ""]
            ops = _retrievals(code)
            if any(op in seen for op in ops):
                out["scripts_reexecuting"] += 1
            seen.extend(ops)
    # defaulted rather than dict(out): Counter omits keys that never incremented, so an
    # arm that did none of something would be MISSING the key rather than reporting 0 —
    # indistinguishable from an older report that did not measure it
    result = {key: out.get(key, 0) for key, _ in _ROWS if key != "wall_s"}
    result["wall_s"] = round(wall)
    return result




def render(
    now: dict[str, Any],
    base: dict[str, Any] | None,
    arm: str,
    concurrency: int | None = None,
    base_concurrency: int | None = None,
) -> str:
    lines = [f"arm `{arm}` — mechanics only; answer quality is NOT measured here", ""]
    width = max(len(label) for _, label in _ROWS)
    head = f"  {'':<{width}}  {'now':>7}"
    if base:
        head += f"  {'was':>7}  {'delta':>8}"
    lines += [head, "  " + "-" * (width + 26 if base else width + 9)]
    for key, label in _ROWS:
        value = now.get(key, 0)
        row = f"  {label:<{width}}  {value:>7}"
        if base:
            prior = base.get(key, 0)
            delta = value - prior
            pct = f"{100*delta/prior:+.0f}%" if prior else "  n/a"
            row += f"  {prior:>7}  {pct:>8}"
        lines.append(row)
    if base and concurrency is not None and base_concurrency is not None and concurrency != base_concurrency:
        lines += [
            "",
            f"  NOTE: this run used --concurrency {concurrency}, the baseline {base_concurrency}.",
            "  Every counter above wall clock is concurrency-invariant and comparable; WALL CLOCK",
            "  IS NOT — it sums per-turn latency, and a concurrent run queues against the same",
            "  local stack. Re-run serially before reading a wall-clock delta as a result.",
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("report", type=Path)
    parser.add_argument("--arm", default=None, help="arm to count (default: every arm present)")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="an earlier report to compare against; omit to use run 9c6595ac's recorded figures",
    )
    parser.add_argument("--json", action="store_true", help="emit the counters as JSON")
    args = parser.parse_args(argv)

    try:
        report = json.loads(args.report.read_text())
    except Exception as exc:
        print(f"cannot read {args.report}: {exc}", file=sys.stderr)
        return 2

    arms = [args.arm] if args.arm else list(report.get("arms") or [])
    if not arms:
        print("report names no arms", file=sys.stderr)
        return 2

    baseline_report = None
    if args.baseline:
        try:
            baseline_report = json.loads(args.baseline.read_text())
        except Exception as exc:
            print(f"cannot read {args.baseline}: {exc}", file=sys.stderr)
            return 2

    payload = {}
    for arm in arms:
        now = counters(report, arm)
        payload[arm] = now
        if args.json:
            continue
        if baseline_report is not None:
            base = counters(baseline_report, arm)
            source = str(args.baseline)
        elif arm == "code":
            base, source = BASELINE_9C6595AC, "run 9c6595ac (recorded)"
        else:
            base, source = None, None
        print(
            render(
                now,
                base,
                arm,
                concurrency=(report.get("config") or {}).get("concurrency"),
                base_concurrency=(
                    (baseline_report.get("config") or {}).get("concurrency")
                    if baseline_report is not None
                    else BASELINE_9C6595AC_CONCURRENCY
                ),
            )
        )
        if base:
            print(f"\n  baseline: {source}")
        print()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
