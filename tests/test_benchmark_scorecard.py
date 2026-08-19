"""The scorecard's job is to not lie about a case where an arm fell over.

Everything else it does is arithmetic on a saved report; the property worth pinning is that
an arm which did not finish never appears cheaper or faster than one that did.
"""

from genetics_mcp_server.scripts.benchmark_scorecard import render


def _turn(case, arm, i, status="ok", ms=1000.0, tools=2, cost=0.5, bracket=None):
    t = {
        "case_id": case,
        "arm": arm,
        "turn_index": i,
        "status": status,
        "ms_to_done": ms,
        "tool_calls": tools,
        "cost_usd": cost,
    }
    if bracket:
        t["cost_usd"] = None
        t["cost_usd_min"], t["cost_usd_max"] = bracket
    return t


def _report(turns, judging=None):
    r = {"arms": ["nocode", "code"], "turns": turns}
    if judging is not None:
        r["judging"] = judging
    return r


def test_a_case_with_a_failed_turn_is_excluded_from_the_total():
    # `code` aborts on turn 1 having spent almost nothing. Summed naively beside a `nocode`
    # that finished, that reads as a 30x efficiency win.
    turns = [
        _turn("clean", "nocode", 0, ms=10_000, tools=4, cost=1.0),
        _turn("clean", "code", 0, ms=10_000, tools=1, cost=1.0),
        _turn("broken", "nocode", 0, ms=30_000, tools=9, cost=3.0),
        _turn("broken", "code", 0, status="timeout", ms=100, tools=0, cost=0.01),
    ]
    out = render(_report(turns))

    total = [ln for ln in out.splitlines() if ln.startswith("TOTAL")]
    assert total, out
    assert "1 comparable" in total[0]
    # the total is the clean case alone: 10.0s and $1.00 per arm, not 40.0s vs 10.1s
    assert "10.0" in total[0] and "30" not in total[0]
    assert "1.00" in total[0] and "3.0" not in total[0]

    broken = [ln for ln in out.splitlines() if ln.startswith("broken")][0]
    assert "*" in broken, "an uncomparable case must be marked in its own row"
    assert "code: 1 turn(s) timeout" in out, "and the reason must be stated"


def test_differing_turn_counts_make_a_case_uncomparable():
    turns = [
        _turn("lopsided", "nocode", 0),
        _turn("lopsided", "nocode", 1),
        _turn("lopsided", "code", 0),
    ]
    out = render(_report(turns))
    assert "turn counts differ" in out
    # the footnote mentions the word TOTAL, so match the ROW, not the substring
    assert not [ln for ln in out.splitlines() if ln.startswith("TOTAL")], (
        "no case is comparable, so there is nothing to total"
    )


def test_interval_priced_cost_is_marked_not_presented_as_measured():
    turns = [
        _turn("bracketed", "nocode", 0, bracket=(1.0, 2.0)),
        _turn("bracketed", "code", 0, cost=0.5),
    ]
    out = render(_report(turns))
    assert "1.50~" in out, "the midpoint is shown"
    assert "interval-priced" in out, "and flagged as not a measurement"


def test_unpriced_model_is_not_reported_as_free():
    turns = [
        _turn("unpriced", "nocode", 0, cost=None),
        _turn("unpriced", "code", 0, cost=None),
    ]
    for t in turns:
        t.pop("cost_usd_min", None)
        t.pop("cost_usd_max", None)
    out = render(_report(turns))
    assert "n/p" in out and "0.00" not in out


def test_judge_cell_tallies_per_turn_verdicts_and_flags_broken_blinding():
    turns = [_turn("c", "nocode", i) for i in range(2)] + [_turn("c", "code", i) for i in range(2)]
    judging = {
        "arms": ["nocode", "code"],
        "pairs": [
            {"case_id": "c", "turn_index": 0, "outcome": "win", "winner": "code",
             "margin": "clear", "arm_identifiable": False},
            {"case_id": "c", "turn_index": 1, "outcome": "tie_both_good", "winner": None,
             "margin": None, "arm_identifiable": True},
        ],
    }
    out = render(_report(turns, judging))
    assert "code 1" in out and "tie 1" in out
    assert "!" in out and "blinding did not hold" in out


def test_an_unjudged_report_says_so_rather_than_showing_an_empty_column():
    turns = [_turn("c", "nocode", 0), _turn("c", "code", 0)]
    assert "has not been judged" in render(_report(turns))


def test_a_rate_limited_run_is_called_out_before_any_number_is_read():
    # the shape measured on 2026-08-19: the server refuses partway, the cases already
    # replayed keep their cost, everything after cascades, and the report still looks whole
    turns = [
        _turn("done", "nocode", 0),
        _turn("done", "code", 0),
        _turn("refused", "nocode", 0, status="error"),
        _turn("refused", "code", 0, status="error"),
    ]
    for t in turns[2:]:
        t["error"] = 'HTTP 429: {"detail":"Rate limit exceeded (hourly limit 20/hour)."}'
    out = render(_report(turns))
    assert "RATE-LIMITED" in out
    assert "1 of 2 cases are comparable" in out
    assert "RATE_LIMIT_PER_HOUR" in out, "the note must name the knob that fixes it"


def test_a_report_judged_without_per_pair_rows_is_distinguished_from_an_unjudged_one():
    turns = [_turn("c", "nocode", 0), _turn("c", "code", 0)]
    out = render(_report(turns, judging={"arms": ["nocode", "code"], "wins": {}}))
    assert "did not persist" in out
