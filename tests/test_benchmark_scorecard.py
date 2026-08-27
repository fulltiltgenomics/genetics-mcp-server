"""The scorecard's job is to not lie about a case where an arm fell over.

Everything else it does is arithmetic on a saved report; the property worth pinning is that
an arm which did not finish never appears cheaper or faster than one that did.
"""

import json

from genetics_mcp_server.scripts.benchmark_scorecard import (
    main,
    render,
    render_markdown,
)


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


def test_markdown_reproduces_a_script_whole_where_the_terminal_views_elide_it():
    # the reason this renderer exists: `--tools`/`--transcript` are width-bound, and the one
    # argument worth reading when the code arm loses is exactly the one that never fits
    script = "\n".join(f"print({i})  # {'x' * 300}" for i in range(40))
    turns = [
        _turn("c", "nocode", 0),
        _turn("c", "code", 0),
    ]
    turns[1]["tool_calls_detail"] = [
        {"seq": 0, "name": "run_analysis", "iteration": 2, "input": {"code": script},
         "script_duration_ms": 1200.0, "script_status": "ok"}
    ]
    turns[1]["final_answer"] = "42"
    out = render_markdown(_report(turns))
    assert script in out, "the script must appear verbatim, not elided"
    assert "…" not in out
    assert "```python" in out
    assert "sandbox 1.2s, `ok`" in out


def test_markdown_fence_survives_an_argument_that_contains_a_fence():
    turns = [_turn("c", "nocode", 0), _turn("c", "code", 0)]
    turns[1]["tool_calls_detail"] = [
        {"seq": 0, "name": "run_analysis", "input": {"code": "x = '''```'''\ny = 1"}}
    ]
    out = render_markdown(_report(turns))
    body = out.split("- `code`:")[1]
    fence = body.strip().splitlines()[0]
    assert fence.startswith("````"), f"fence must outgrow the value's own backticks: {fence!r}"
    assert body.count(fence.rstrip("python")) >= 2


def test_markdown_says_what_the_report_could_not_keep():
    # a turn whose prose was thrown away at capture must not present its remainder as the
    # whole reply, and no report holds tool RESULTS at all
    turns = [_turn("c", "nocode", 0), _turn("c", "code", 0)]
    turns[0]["final_answer"] = "In summary, yes."
    turns[0]["final_answer_dropped_chars"] = 900
    out = render_markdown(_report(turns))
    assert "900 character(s)" in out
    assert "Tool *results* are not in the report" in out


def test_markdown_marks_an_uncomparable_case_and_shows_the_failed_arm() -> None:
    turns = [
        _turn("broken", "nocode", 0),
        _turn("broken", "code", 0, status="timeout"),
    ]
    turns[1]["error"] = "wall-clock deadline"
    out = render_markdown(_report(turns))
    assert "NOT COMPARABLE" in out
    assert "code: 1 turn(s) timeout" in out
    assert "wall-clock deadline" in out, "the failed arm must show why, not an empty section"


def test_markdown_refuses_a_case_that_is_not_in_the_report():
    turns = [_turn("c", "nocode", 0), _turn("c", "code", 0)]
    out = render_markdown(_report(turns), case="other")
    assert out.startswith("no case matching"), "main() keys the non-zero exit off this shape"


def test_a_one_arm_markdown_holds_only_that_arm_but_still_states_comparability():
    # dropping the pairing from a one-arm file would let an arm whose partner fell over read
    # as a clean run — the property belongs to the pair, not to the arm
    turns = [
        _turn("broken", "nocode", 0),
        _turn("broken", "code", 0, status="timeout"),
    ]
    turns[0]["final_answer"] = "the nocode answer"
    out = render_markdown(_report(turns), only_arm="nocode")
    assert "arm `code`" not in out, "the other arm's turns must not be in a one-arm file"
    assert "the nocode answer" in out
    assert "NOT COMPARABLE" in out and "code: 1 turn(s) timeout" in out


def test_a_one_arm_markdown_keeps_the_question_even_when_only_the_other_arm_recorded_it():
    turns = [_turn("c", "nocode", 0), _turn("c", "code", 0, status="error")]
    turns[0]["user_question"] = "what chromosome is PCSK9 on?"
    out = render_markdown(_report(turns), only_arm="code")
    assert "what chromosome is PCSK9 on?" in out


def test_markdown_refuses_an_arm_that_is_not_in_the_report():
    turns = [_turn("c", "nocode", 0), _turn("c", "code", 0)]
    assert render_markdown(_report(turns), only_arm="rag").startswith("no arm 'rag'")


def test_markdown_writes_one_file_per_arm_beside_the_paired_one(tmp_path):
    report = _report([_turn("c", "nocode", 0), _turn("c", "code", 0)])
    report["config"] = {"run_id": "abc123"}
    src = tmp_path / "report.json"
    src.write_text(json.dumps(report))
    out = tmp_path / "transcripts.md"

    assert main([str(src), "--markdown", str(out)]) == 0

    assert out.exists()
    for arm in ("nocode", "code"):
        per_arm = tmp_path / f"transcripts.{arm}.md"
        assert per_arm.exists(), f"{arm} got no file of its own"
        assert f"arm `{arm}` only" in per_arm.read_text()


def test_markdown_interleaves_thinking_with_the_calls_it_produced():
    # the value of having the reasoning is reading it immediately BEFORE the calls it
    # explains; collected at the top of the turn it answers nothing
    turns = [_turn("c", "nocode", 0), _turn("c", "code", 0)]
    turns[1]["tool_calls_detail"] = [
        {"seq": 0, "name": "search_genes", "iteration": 1, "input": {"query": "LDLR"}},
        {"seq": 1, "name": "run_analysis", "iteration": 2, "input": {"code": "print(1)"}},
    ]
    turns[1]["thinking_detail"] = [
        {"iteration": 1, "text": "resolve the symbol first"},
        {"iteration": 2, "text": "now aggregate the burden rows"},
    ]
    out = render_markdown(_report(turns))

    first = out.index("resolve the symbol first")
    call_one = out.index("search_genes")
    second = out.index("now aggregate the burden rows")
    call_two = out.index("run_analysis")
    assert first < call_one < second < call_two


def test_markdown_shows_the_reasoning_of_iterations_that_called_nothing():
    # the final iteration answers rather than calling tools, and a turn that went wrong
    # without calling anything explains itself only here
    turns = [_turn("c", "nocode", 0), _turn("c", "code", 0)]
    turns[1]["tool_calls_detail"] = [
        {"seq": 0, "name": "search_genes", "iteration": 1, "input": {"query": "LDLR"}}
    ]
    turns[1]["thinking_detail"] = [
        {"iteration": 1, "text": "look it up"},
        {"iteration": 2, "text": "that is enough to answer"},
    ]
    turns[0]["tool_calls_detail"] = []
    turns[0]["thinking_detail"] = [{"iteration": 1, "text": "I already know this one"}]
    out = render_markdown(_report(turns))

    assert "that is enough to answer" in out
    assert "I already know this one" in out


def test_markdown_prints_each_iteration_of_reasoning_once():
    turns = [_turn("c", "nocode", 0), _turn("c", "code", 0)]
    turns[1]["tool_calls_detail"] = [
        {"seq": 0, "name": "a", "iteration": 1, "input": {}},
        {"seq": 1, "name": "b", "iteration": 1, "input": {}},
    ]
    turns[1]["thinking_detail"] = [{"iteration": 1, "text": "fan these out together"}]
    assert render_markdown(_report(turns)).count("fan these out together") == 1


def test_markdown_of_a_report_without_captured_thinking_is_unchanged():
    turns = [_turn("c", "nocode", 0), _turn("c", "code", 0)]
    turns[1]["tool_calls_detail"] = [{"seq": 0, "name": "a", "iteration": 1, "input": {}}]
    assert "thinking" not in render_markdown(_report(turns))


def test_markdown_shows_the_discarded_prose_where_it_was_written():
    # the table an arm laid out before one last call is not commentary; presenting the
    # remainder as "the answer" without it presents a fragment as the whole reply
    turns = [_turn("c", "nocode", 0), _turn("c", "code", 0)]
    turns[0]["tool_calls_detail"] = [
        {"seq": 0, "name": "search_genes", "input": {"query": "LDLR"}},
        {"seq": 1, "name": "get_burden", "input": {}},
    ]
    turns[0]["final_answer"] = "In summary, yes."
    turns[0]["final_answer_dropped_chars"] = 260
    turns[0]["final_answer_dropped_prose"] = [
        {"after_call": None, "text": "Let me look that up."},
        {"after_call": 0, "text": "| gene | p |"},
    ]
    out = render_markdown(_report(turns))

    assert "Let me look that up." in out
    assert "| gene | p |" in out
    # position is most of the meaning: the second block sits between the two calls
    assert out.index("search_genes") < out.index("| gene | p |") < out.index("get_burden")
    assert out.index("Let me look that up.") < out.index("search_genes")
    assert "260 character(s)" not in out, "the text is here, so the apology must be gone"
    assert "2 prose block(s)" in out


def test_markdown_still_declares_prose_a_report_kept_only_the_length_of():
    turns = [_turn("c", "nocode", 0), _turn("c", "code", 0)]
    turns[0]["tool_calls_detail"] = [{"seq": 0, "name": "a", "input": {}}]
    turns[0]["final_answer_dropped_chars"] = 260
    out = render_markdown(_report(turns))
    assert "260 character(s)" in out
    assert "predates their capture" in out
