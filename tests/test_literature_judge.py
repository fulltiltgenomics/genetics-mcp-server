"""Tests for the literature-evidence judge harness: no network, no API calls."""

import json

import pytest

from genetics_mcp_server.scripts import literature_judge as lj

REVIEW = """\
# Review

## Findings

- **[F1] session=aaaa1111 msg=m1-full-uuid time=2026-07-01 10:00:00 user=bneale backend=perplexity**
  - Question: why?
  - Claim (verbatim): "GPR17 drives demyelination in schizophrenia patients strongly"
  - Problem: 1, 7 — relayed with no n.
  - Severity: high

- **[F2] session=bbbb2222 msg=m2abcdef time=2026-07-02 10:00:00 user=mjdaly backend=none**
- Problem: 7/3 — no search.
- Severity: low.

## Counter-examples

- **[G1] session=aaaa1111 msg=m1-full time=2026-07-01 10:00:00 backend=perplexity** — "Very small, candidate-gene design with seventy six cases" and nothing else.
- **[G2] session=bbbb2222 msg=m2abcdef** — literature reconciled with the loaded data (category-6 handled correctly).

## Patterns
- **not a label** — prose
"""


def _tool_use(tid, name, inp=None):
    return {"type": "tool_use", "id": tid, "name": name, "input": inp or {}}


def _row(mid, role, content, *, session="s1", t="2026-07-01 10:00:00", uses=(), results=()):
    return {
        "id": mid, "session_id": session, "role": role, "content": content, "created_at": t,
        "content_json": json.dumps(list(uses)) if uses else None,
        "tool_results_json": json.dumps(list(results)) if results else None,
        "user_id": "bneale@example.org",
    }


def test_parse_labels_reads_both_kinds_and_their_fields():
    labels = lj.parse_labels(REVIEW, "review-x")
    assert [lab.key for lab in labels] == ["review-x:F1", "review-x:F2", "review-x:G1", "review-x:G2"]
    f1, f2, g1, g2 = labels
    assert (f1.session, f1.msg, f1.categories, f1.severity) == ("aaaa1111", "m1-full-uuid", {1, 7}, "high")
    assert (f2.categories, f2.severity) == ({3, 7}, "low")
    assert g1.categories == frozenset() and g1.quotes == ["Very small, candidate-gene design with seventy six cases"]
    assert g2.categories == {6}


def test_resolve_labels_matches_short_and_full_ids_and_reports_the_rest():
    labels = lj.parse_labels(REVIEW, "r")
    turns = [
        lj.Turn(id="m1-full-uuid", question="", answer="", literature_results=[], session_id="aaaa1111-x"),
        lj.Turn(id="m2abcdef-0000", question="", answer="", literature_results=[], session_id="bbbb2222-y"),
    ]
    resolved, unresolved = lj.resolve_labels(labels, turns)
    assert {k: [lab.key for lab in v] for k, v in resolved.items()} == {
        "m1-full-uuid": ["r:F1", "r:G1"], "m2abcdef-0000": ["r:F2", "r:G2"],
    }
    assert unresolved == []
    # an ambiguous prefix resolves to nothing rather than to an arbitrary turn
    turns.append(lj.Turn(id="m1-full-other", question="", answer="", literature_results=[], session_id="aaaa1111-z"))
    _, unresolved = lj.resolve_labels(labels, turns)
    assert [lab.key for lab in unresolved] == ["r:G1"]


@pytest.mark.parametrize("uses, expected", [
    ([_tool_use("t1", "search_scientific_literature")], True),
    ([_tool_use("t1", "launch_subagents", {"tasks": [{"skill": "literature_review", "query": "q"}]})], True),
    ([_tool_use("t1", "launch_subagents", {"tasks": [{"skill": "data_analysis", "query": "q"}]})], False),
    ([_tool_use("t1", "web_search")], False),
    ([], False),
])
def test_literature_bearing_predicate(uses, expected):
    assert lj.is_literature_bearing(_row("a", "assistant", "x", uses=uses)) is expected


def test_user_rows_are_never_literature_bearing():
    assert not lj.is_literature_bearing(_row("u", "user", "x", uses=[_tool_use("t1", "search_scientific_literature")]))


def test_build_turns_matches_results_by_id_and_carries_earlier_results():
    rows = [
        _row("u1", "user", "first question", t="2026-07-01 10:00:00"),
        _row("a1", "assistant", "answer one", t="2026-07-01 10:01:00",
             uses=[_tool_use("t1", "search_scientific_literature", {"query": "q"}), _tool_use("t2", "query_bigquery")],
             results=[{"type": "tool_result", "tool_use_id": "t2", "content": "rows"},
                      {"type": "tool_result", "tool_use_id": "t1", "content": "{\"summary\": \"paper\"}"}]),
        _row("u2", "user", "second question", t="2026-07-01 10:05:00"),
        _row("a2", "assistant", "answer two", t="2026-07-01 10:06:00"),
    ]
    t1, t2 = lj.build_turns(rows)
    assert (t1.question, t1.persisted, t1.literature_bearing) == ("first question", True, True)
    assert t1.literature_results == [{"name": "search_scientific_literature", "input": {"query": "q"},
                                      "content": "{\"summary\": \"paper\"}"}]
    assert (t2.question, t2.persisted, t2.literature_bearing) == ("second question", False, False)
    assert t2.earlier_results == t1.literature_results


def test_agreement_arithmetic():
    labels = lj.parse_labels(REVIEW, "r")
    by_turn = {"m1": [labels[0], labels[2]], "m2": [labels[1], labels[3]]}
    judgements = {
        # reports category 1 but not 7; its finding does not quote G1's passage
        "m1": {"findings": [{"claim": "GPR17 drives demyelination", "categories": [1]}]},
        # reports 3 and 7, and a category-6 finding that violates G2's named category
        "m2": {"findings": [{"claim": "x", "categories": [3, 7]}, {"claim": "y", "categories": ["6"]}]},
    }
    agg = lj.score_agreement(by_turn, judgements)
    assert (agg.finding_hits, agg.finding_pairs) == (3, 4)
    assert (agg.counter_kept, agg.counter_examples) == (1, 2)
    assert agg.combined == pytest.approx(4 / 6)
    assert agg.missed == [("m1", 7)]
    assert [v[1] for v in agg.violated] == ["r:G2"]


def test_counter_example_is_violated_by_a_finding_quoting_its_passage():
    labels = lj.parse_labels(REVIEW, "r")
    judged = {"m1": {"findings": [{"claim": "the study was very small, candidate-gene design with seventy six cases",
                                   "categories": [5]}]}}
    agg = lj.score_agreement({"m1": [labels[2]]}, judged)
    assert (agg.counter_kept, agg.counter_examples) == (0, 1)


def test_unjudged_turns_are_left_out_of_agreement():
    labels = lj.parse_labels(REVIEW, "r")
    agg = lj.score_agreement({"m1": [labels[0]]}, {})
    assert agg.turns == 0 and agg.combined is None


def test_half_split_is_deterministic_and_uses_both_halves():
    ids = [f"msg-{i}" for i in range(40)]
    halves = [lj.half_of(i) for i in ids]
    assert halves == [lj.half_of(i) for i in ids]
    assert set(halves) == {"dev", "heldout"}


def test_category_mix():
    mix = lj.category_mix([{1}, {3}, {1, 7}, {6}, {2, 5}])
    assert mix == {"record_only": 2, "claim_only": 1, "both": 1, "other": 1}


def test_load_replay_accepts_both_result_shapes(tmp_path):
    path = tmp_path / "replay.json"
    path.write_text(json.dumps({"turns": [
        {"id": "q1", "question": "Q", "answer": "A",
         "literature_results": ["raw result", {"name": "launch_subagents", "input": {"tasks": []}, "content": "digest"}]},
        {"question": "Q2", "answer": "A2", "literature_results": []},
    ]}))
    t1, t2 = lj.load_replay(path)
    assert t1.id == "q1" and [r["content"] for r in t1.literature_results] == ["raw result", "digest"]
    assert t1.literature_results[0]["name"] == "search_scientific_literature"
    assert t2.id == "1" and not t2.literature_bearing


def test_load_replay_accepts_the_harness_report_shape(tmp_path):
    """replay_benchmark.py's own --output JSON, read with no conversion step."""
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"turns": [
        {"case_id": "c1", "arm": "code", "turn_index": 0, "status": "ok",
         "user_question": "Q", "final_answer": "A",
         "literature_results": [{"name": "search_scientific_literature", "input": {"q": "x"},
                                  "content": "paper text"}]},
        {"case_id": "c1", "arm": "code", "turn_index": 1, "status": "error",
         "user_question": "Q2", "final_answer": None, "literature_results": []},
    ]}))
    turns = lj.load_replay(path)
    assert len(turns) == 1  # the "error" turn has no answer to judge and is skipped
    t = turns[0]
    assert t.id == "c1:code:0" and t.question == "Q" and t.answer == "A"
    assert t.literature_bearing and t.literature_results[0]["content"] == "paper text"


def test_load_replay_fills_earlier_results_for_the_harness_shape(tmp_path):
    """The harness's own --output report carries no earlier_results field per turn (unlike
    a prod row), so the loader has to rebuild the same per-(case, arm) accumulation
    build_turns does for prod rows, in turn_index order."""
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"turns": [
        {"case_id": "c1", "arm": "code", "turn_index": 0, "status": "ok",
         "user_question": "Q0", "final_answer": "A0",
         "literature_results": [{"name": "search_scientific_literature", "input": {},
                                  "content": "paper one"}]},
        {"case_id": "c1", "arm": "code", "turn_index": 1, "status": "ok",
         "user_question": "Q1", "final_answer": "A1",
         "literature_results": [{"name": "search_scientific_literature", "input": {},
                                  "content": "paper two"}]},
        # a different arm of the same case starts its own accumulation from empty
        {"case_id": "c1", "arm": "nocode", "turn_index": 0, "status": "ok",
         "user_question": "Q0", "final_answer": "A0'", "literature_results": []},
    ]}))
    t0, t1, t0_other_arm = lj.load_replay(path)
    assert t0.earlier_results == []
    assert [r["content"] for r in t1.earlier_results] == ["paper one"]
    assert t0_other_arm.earlier_results == []


@pytest.mark.parametrize("payload", [
    {"not_turns": []},
    [{"question": "Q"}],
    [{"question": "Q", "answer": "A", "literature_results": [{"no_content": 1}]}],
])
def test_load_replay_rejects_malformed_input(tmp_path, payload):
    path = tmp_path / "replay.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        lj.load_replay(path)


def test_render_prompt_caps_each_result_and_names_missing_ones():
    turn = lj.Turn(id="t", question="Q", answer="A",
                   literature_results=[{"name": "search_scientific_literature", "input": {}, "content": "x" * (lj.RESULT_CAP + 50)},
                                       {"name": "search_scientific_literature", "input": {}, "content": ""}])
    prompt = lj.render_prompt(turn, "2026-09-27")
    assert "[... 50 chars cut by the judge harness ...]" in prompt
    assert "(no result persisted)" in prompt


def test_uncategorised_findings_are_neither_violations_nor_recall_hits():
    labels = lj.parse_labels(REVIEW, "r")
    judged = {"m1": {"findings": [
        # quotes G1's praised passage but names no rubric category: not a rubric finding
        {"claim": "Counter-example candidate: very small, candidate-gene design with seventy six cases"},
        {"claim": "listed here only as reference", "categories": []},
        {"claim": "GPR17 drives demyelination", "categories": [1]},
    ]}}
    agg = lj.score_agreement({"m1": [labels[0], labels[2]]}, judged)
    assert (agg.counter_kept, agg.counter_examples) == (1, 1)
    assert (agg.finding_hits, agg.finding_pairs) == (1, 2)
    assert agg.uncategorised == 2
    s = lj.summarize(judged)
    assert (s["findings"], s["uncategorised_dropped"], s["turns_with_finding"]) == (1, 2, 1)


def test_void_labels_are_skipped_and_reported(tmp_path):
    voided = REVIEW.replace(
        "and nothing else.\n",
        "and nothing else.\n  - VOID 2026-09-27: the full record contradicts the praised sentence.\n")
    (tmp_path / "review-x.md").write_text(voided)
    live, void = lj.load_labels(tmp_path)
    assert [lab.key for lab in void] == ["review-x:G1"]
    assert [lab.key for lab in live] == ["review-x:F1", "review-x:F2", "review-x:G2"]


def test_rows_cache_depends_on_users_and_context(tmp_path):
    a = lj.rows_cache_path("ctx", ["mjdaly", "bneale"], tmp_path)
    assert a == lj.rows_cache_path("ctx", ["bneale", "mjdaly"], tmp_path)
    assert a != lj.rows_cache_path("ctx", ["bneale"], tmp_path)
    assert a != lj.rows_cache_path("other", ["mjdaly", "bneale"], tmp_path)
