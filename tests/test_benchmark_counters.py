"""The counters must be reproducible, or a before/after delta is fiction.

The point of the module under test is that a measurement re-derived at each reading is not
the same measurement twice — so the thing to pin is that the recorded baseline is exactly
what the current code computes on the run it was recorded from.
"""

import json

import pytest

from genetics_mcp_server.scripts import benchmark_counters as bc

REPORT_9C6595AC = "/home/jkarjala/benchmarks/benchmark1.json"


def _turn(case, index, arm, scripts, iterations=1, ms=1000.0):
    return {
        "case_id": case,
        "turn_index": index,
        "arm": arm,
        "iterations": iterations,
        "ms_to_done": ms,
        "tool_calls": len(scripts),
        "tool_calls_detail": [
            {"seq": i, "name": "run_analysis", "input": {"code": s}}
            for i, s in enumerate(scripts)
        ],
    }


def test_discovery_first_counts_the_opening_script_only():
    """A discovery script mid-turn is cheap; one that OPENS the turn costs a round trip
    before any work starts, which is the shape worth counting."""
    report = {
        "arms": ["code"],
        "turns": [
            _turn("a", 0, "code", ["import genetics\ns = genetics.schema()\nprint(s)"]),
            _turn("b", 0, "code", ["genetics.sql('SELECT 1')", "genetics.schema()"]),
        ],
    }
    out = bc.counters(report, "code")
    assert out["discovery_first_turns"] == 1
    assert out["scripts_probing_schema"] == 2


def test_reexecution_is_scoped_to_the_case_not_the_turn():
    """A follow-up turn re-fetching what the previous turn already fetched is the whole
    phenomenon; a per-turn scope would score it zero."""
    call = "genetics.credible_sets(variant='4:102267552:C:T')"
    report = {
        "arms": ["code"],
        "turns": [_turn("a", 0, "code", [call]), _turn("a", 1, "code", [call])],
    }
    assert bc.counters(report, "code")["scripts_reexecuting"] == 1
    # a different case is not a repeat
    other = {
        "arms": ["code"],
        "turns": [_turn("a", 0, "code", [call]), _turn("b", 0, "code", [call])],
    }
    assert bc.counters(other, "code")["scripts_reexecuting"] == 0


def test_arms_are_counted_independently():
    report = {
        "arms": ["code", "nocode"],
        "turns": [
            _turn("a", 0, "code", ["genetics.schema()"]),
            _turn("a", 0, "nocode", []),
        ],
    }
    assert bc.counters(report, "code")["scripts"] == 1
    assert bc.counters(report, "nocode")["scripts"] == 0


@pytest.mark.skipif(
    not __import__("pathlib").Path(REPORT_9C6595AC).exists(),
    reason="the 9c6595ac report is not on this machine",
)
def test_recorded_baseline_is_what_this_code_computes():
    """If this fails, either a regex changed or the baseline was hand-edited. Re-run the
    tool against the report and replace BASELINE_9C6595AC in the same commit — do not
    adjust the expectation to match a remembered number."""
    with open(REPORT_9C6595AC) as fh:
        report = json.load(fh)
    assert bc.counters(report, "code") == bc.BASELINE_9C6595AC
