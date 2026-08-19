"""The call listing has to be complete and in order, or it cannot be used to check a result.

A tool-call COUNT says the code arm made one call where the baseline made six. It cannot say
whether that one call asked for the right thing — only the arguments distinguish a good
script from a wrong one the model never recovered from.
"""

from genetics_mcp_server.scripts.benchmark_scorecard import render_tool_calls
from genetics_mcp_server.scripts.replay_benchmark import count_tool_calls, extract_tool_calls


def _tool_use(name, inp, id_="tu_1"):
    return {"type": "tool_use", "id": id_, "name": name, "input": inp}


def test_calls_are_returned_in_emission_order_with_their_arguments():
    content = [
        {"type": "text", "text": "let me look"},
        _tool_use("search_genes", {"query": "PCSK9"}, "a"),
        {"type": "text", "text": "and now"},
        _tool_use("get_gene_annotations", {"symbol": "PCSK9"}, "b"),
    ]
    calls = extract_tool_calls(content)
    assert [c["name"] for c in calls] == ["search_genes", "get_gene_annotations"]
    assert [c["seq"] for c in calls] == [0, 1]
    assert calls[0]["input"] == {"query": "PCSK9"}
    assert calls[1]["id"] == "b"


def test_a_long_script_argument_is_stored_whole():
    script = "from genetics import sql\n" * 500
    calls = extract_tool_calls([_tool_use("run_analysis", {"script": script})])
    assert calls[0]["input"]["script"] == script, "the script must not be truncated at capture"


def test_the_count_cannot_disagree_with_the_listing():
    content = [_tool_use("a", {}), {"type": "text", "text": "x"}, _tool_use("b", {})]
    assert count_tool_calls(content) == len(extract_tool_calls(content)) == 2


def test_display_prose_imitating_a_tool_marker_is_not_counted_as_a_call():
    # the model has been observed writing these; only real tool_use blocks count
    content = [{"type": "text", "text": "*[Using tool: search_genes]*"}]
    assert extract_tool_calls(content) == []


def test_empty_and_missing_content_are_both_no_calls():
    assert extract_tool_calls(None) == []
    assert extract_tool_calls([]) == []


def _report(detail, status="ok"):
    return {
        "arms": ["nocode", "code"],
        "turns": [
            {"case_id": "c", "arm": "nocode", "turn_index": 0, "status": status,
             "ms_to_done": 1.0, "tool_calls": len(detail), "cost_usd": 0.1,
             "tool_calls_detail": detail},
            {"case_id": "c", "arm": "code", "turn_index": 0, "status": "ok",
             "ms_to_done": 1.0, "tool_calls": 0, "cost_usd": 0.1, "tool_calls_detail": []},
        ],
    }


def test_the_tools_view_elides_visibly_and_says_where_the_whole_value_is():
    long_script = "x" * 5000
    out = render_tool_calls(_report([{"seq": 0, "name": "run_analysis", "id": "t",
                                      "input": {"script": long_script}}]), width=60)
    assert "…" in out, "a shortened argument must be marked"
    assert "x" * 5000 not in out
    assert "tool_calls_detail" in out, "and the report path for the full value must be given"


def test_a_turn_with_no_calls_is_distinguished_from_one_that_was_never_recorded():
    assert "answered with no tool calls" in render_tool_calls(_report([]))
    missing = _report([])
    for t in missing["turns"]:
        t.pop("tool_calls_detail")
    assert "predates it" in render_tool_calls(missing)


def test_a_failed_turn_shows_its_status_rather_than_an_empty_call_list():
    out = render_tool_calls(_report([], status="timeout"))
    assert "timeout" in out
