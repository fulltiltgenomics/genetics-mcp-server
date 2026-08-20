"""A tool-call COUNT says the code arm was worse. Only the sequence says why.

The scorecard's aggregate can report that one arm made more calls and took longer, and that
is where it stops. The explanation is always inside one case — a script that failed and was
rewritten, five roundtrips spent probing a schema, a tool called twice with the same
arguments — so what is pinned here is that the side-by-side view carries the things that
distinguish those causes, and never invents a measurement it does not have.
"""

import re

from genetics_mcp_server.scripts.benchmark_scorecard import _tool_phase_total, render_transcript
from genetics_mcp_server.scripts.replay_benchmark import attach_call_metadata, extract_tool_calls


def _call_lines_of(out):
    """The rendered CALL rows only. The footnotes discuss the same markers in prose, and a
    substring search over the whole output therefore matches the explanation of an absence."""
    return [ln for ln in out.splitlines() if re.match(r"\s*\d+\.\s", ln) or "|" in ln and re.search(r"\|\s*\d+\.\s", ln)]


def _call(seq, name, inp, id_):
    return {"seq": seq, "name": name, "id": id_, "input": inp}


def _turn(case, arm, i, calls, **kw):
    t = {
        "case_id": case,
        "arm": arm,
        "turn_index": i,
        "status": "ok",
        "ms_to_done": 10_000.0,
        "model_ms_total": 8_000.0,
        "iterations": len(calls) + 1,
        "tool_calls": len(calls),
        "cost_usd": 0.5,
        "tool_calls_detail": calls,
        "user_question": "how many caQTLs does FOXP3 have?",
        "iterations_detail": [{"iteration": n + 1, "tool_phase_ms": 500.0} for n in range(len(calls))]
        + [{"iteration": len(calls) + 1, "tool_phase_ms": None}],
    }
    t.update(kw)
    return t


def _report(nocode_turns, code_turns):
    return {"arms": ["nocode", "code"], "turns": nocode_turns + code_turns}


# --- what the stream carries and the done chunk cannot -------------------------------------


def test_a_call_is_attributed_to_the_iteration_the_stream_saw_it_in():
    # `message_content` flattens every iteration's blocks into one list with no boundary, so
    # six calls in one parallel iteration and six iterations of one call each look identical
    calls = extract_tool_calls(
        [
            {"type": "tool_use", "id": "a", "name": "search_genes", "input": {"query": "FOXP3"}},
            {"type": "tool_use", "id": "b", "name": "run_analysis", "input": {"code": "x"}},
        ]
    )
    attach_call_metadata(calls, {"a": 1, "b": 4}, {})
    assert [c["iteration"] for c in calls] == [1, 4]


def test_an_unattributed_call_has_no_iteration_rather_than_a_guessed_one():
    calls = extract_tool_calls([{"type": "tool_use", "id": "a", "name": "t", "input": {}}])
    attach_call_metadata(calls, {}, {})
    assert "iteration" not in calls[0], "absent must stay absent, never imputed"


def test_run_analysis_carries_the_sandbox_wall_clock_correlated_by_tool_use_id():
    calls = extract_tool_calls(
        [{"type": "tool_use", "id": "tu9", "name": "run_analysis", "input": {"code": "print(1)"}}]
    )
    attach_call_metadata(
        calls, {}, {"tu9": {"duration_ms": 1420, "status": "ok", "ok": True, "iteration": 3}}
    )
    assert calls[0]["script_duration_ms"] == 1420
    assert calls[0]["script_status"] == "ok"


def test_metadata_never_overwrites_the_arguments_the_model_actually_sent():
    # llm_service rewrites the copy it STREAMS (it strips run_analysis's invented user/
    # session_id), so only the id may be taken from there -- the done chunk keeps the truth
    calls = extract_tool_calls(
        [
            {
                "type": "tool_use",
                "id": "tu1",
                "name": "run_analysis",
                "input": {"code": "print(1)", "user": "forged"},
            }
        ]
    )
    attach_call_metadata(
        calls, {"tu1": 2}, {"tu1": {"duration_ms": 5, "status": "ok", "input": {"code": "OTHER"}}}
    )
    assert calls[0]["input"] == {"code": "print(1)", "user": "forged"}


# --- the tool phase is not the sum of the calls in it --------------------------------------


def test_the_final_iterations_absent_tool_phase_is_not_reported_as_a_gap():
    # the last iteration answered instead of calling tools, so its None is by construction
    total, gaps = _tool_phase_total(
        {"iterations_detail": [{"tool_phase_ms": 300.0}, {"tool_phase_ms": None}]}
    )
    assert total == 300.0 and gaps is False


def test_an_unmeasured_middle_iteration_is_flagged_so_the_total_is_not_read_as_whole():
    total, gaps = _tool_phase_total(
        {"iterations_detail": [{"tool_phase_ms": None}, {"tool_phase_ms": 300.0}, {"tool_phase_ms": None}]}
    )
    assert total == 300.0 and gaps is True
    assert "+" in render_transcript(
        _report(
            [_turn("c", "nocode", 0, [_call(0, "t", {}, "a")])],
            [
                _turn(
                    "c",
                    "code",
                    0,
                    [_call(0, "t", {}, "b")],
                    iterations_detail=[{"tool_phase_ms": None}, {"tool_phase_ms": 300.0}, {"tool_phase_ms": None}],
                )
            ],
        )
    ), "a partly-unmeasured total must be marked"


# --- the side-by-side view -----------------------------------------------------------------


def test_both_arms_appear_on_the_same_line_so_the_sequences_can_be_read_against_each_other():
    out = render_transcript(
        _report(
            [_turn("c", "nocode", 0, [_call(0, "search_genes", {"query": "FOXP3"}, "a")])],
            [_turn("c", "code", 0, [_call(0, "run_analysis", {"code": "import genetics"}, "b")])],
        ),
        width=160,
    )
    paired = [ln for ln in out.splitlines() if "search_genes" in ln and "run_analysis" in ln]
    assert paired, "the two arms' calls must be aligned, not listed one after the other"


def test_the_retry_loops_that_explain_a_slow_code_arm_are_shown_not_left_to_arithmetic():
    out = render_transcript(
        _report(
            [_turn("c", "nocode", 0, [_call(0, "get_x", {}, "a")])],
            [
                _turn(
                    "c",
                    "code",
                    0,
                    [_call(0, "run_analysis", {"code": "x"}, "b")],
                    retry_loops=2,
                    script_attempts=10,
                    script_failures=2,
                    script_outcomes={"ok": 8, "error": 2},
                )
            ],
        ),
        width=200,
    )
    assert "retry loops: 2" in out
    assert "10 attempted, 2 failed" in out and "error 2" in out


def test_a_report_without_iteration_attribution_says_so_instead_of_showing_nothing():
    out = render_transcript(
        _report(
            [_turn("c", "nocode", 0, [_call(0, "t", {}, "a")])],
            [_turn("c", "code", 0, [_call(0, "t", {}, "b")])],
        )
    )
    assert "predates the iteration attribution" in out
    # the explanatory note itself spells `[iN]`, so match the CALL lines, not the whole output
    assert not [ln for ln in _call_lines_of(out) if "[i" in ln]


def test_an_attributed_report_shows_the_roundtrip_each_call_belongs_to():
    calls = [_call(0, "t", {}, "a") | {"iteration": 3}]
    out = render_transcript(_report([_turn("c", "nocode", 0, calls)], [_turn("c", "code", 0, [])]))
    assert "[i3]" in out
    assert "predates the iteration attribution" not in out


def test_a_failed_turn_shows_its_status_beside_the_arm_that_finished():
    out = render_transcript(
        _report(
            [_turn("c", "nocode", 0, [_call(0, "t", {}, "a")])],
            [_turn("c", "code", 0, [], status="timeout", error="TimeoutException: 900s")],
        ),
        width=160,
    )
    assert "timeout" in out and "TimeoutException" in out
    assert "NOT COMPARABLE" in out, "and the case must be marked, as in the table"


def test_an_arm_missing_a_turn_entirely_does_not_drop_the_other_arms_turn():
    out = render_transcript(
        _report(
            [_turn("c", "nocode", 0, [_call(0, "kept", {}, "a")]),
             _turn("c", "nocode", 1, [_call(0, "also_kept", {}, "b")])],
            [_turn("c", "code", 0, [])],
        ),
        width=160,
    )
    assert "also_kept" in out
    assert "no turn here" in out


def test_a_script_is_shown_with_its_line_structure_and_elided_visibly():
    script = "import genetics\n" + "\n".join(f"row_{n} = {n}" for n in range(50))
    out = render_transcript(
        _report(
            [_turn("c", "nocode", 0, [])],
            [_turn("c", "code", 0, [_call(0, "run_analysis", {"code": script}, "b")])],
        ),
        width=160,
        arg_lines=4,
    )
    assert "row_1 = 1" in out, "a script must keep its lines, not be rewrapped into a paragraph"
    assert "row_49 = 49" not in out
    assert "…" in out and "tool_calls_detail" in out, "elision must be marked and the whole value located"


def test_asking_for_a_case_that_is_not_in_the_report_lists_what_is():
    out = render_transcript(_report([_turn("real", "nocode", 0, [])], []), case="typo")
    assert "no case matching" in out and "real" in out
