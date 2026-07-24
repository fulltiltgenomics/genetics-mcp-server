"""Tests for message-history helpers in llm_service.

These cover the invariants the Anthropic API requires when replaying persisted
conversations: every tool_use must be paired with a matching tool_result, and
orphans must be stripped. This is the load-bearing logic for both tool_result
persistence (resumed conversations carry the data) and backward compatibility
(old conversations without persisted results behave exactly as before).
"""

from genetics_mcp_server.llm_service import (
    _mark_history_cache_breakpoint,
    _sanitize_tool_blocks,
    _strip_tool_use_markers,
)


def _tool_use(tid):
    return {"type": "tool_use", "id": tid, "name": "x", "input": {}}


def _tool_result(tid):
    return {"type": "tool_result", "tool_use_id": tid, "content": "ok"}


class TestSanitizeToolBlocks:
    def test_matched_pair_kept(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "text", "text": "calling"}, _tool_use("a")]},
            {"role": "user", "content": [_tool_result("a")]},
            {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
        ]
        out = _sanitize_tool_blocks(messages)
        # the tool_use and its tool_result both survive
        assert any(b.get("type") == "tool_use" for b in out[1]["content"])
        assert any(b.get("type") == "tool_result" for b in out[2]["content"])

    def test_interleaved_multiple_tool_uses_kept(self):
        """A consolidated assistant turn with interleaved text + several tool_use
        blocks, answered by one user message with all tool_results, is preserved.
        This is exactly the shape produced when replaying a persisted turn."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "t1"},
                    _tool_use("a"),
                    {"type": "text", "text": "t2"},
                    _tool_use("b"),
                    {"type": "text", "text": "t3"},
                ],
            },
            {"role": "user", "content": [_tool_result("a"), _tool_result("b")]},
        ]
        out = _sanitize_tool_blocks(messages)
        kept_uses = {b["id"] for b in out[0]["content"] if b.get("type") == "tool_use"}
        kept_results = {b["tool_use_id"] for b in out[1]["content"] if b.get("type") == "tool_result"}
        assert kept_uses == {"a", "b"}
        assert kept_results == {"a", "b"}

    def test_orphan_tool_use_stripped(self):
        """Backward compat: an assistant tool_use with no following tool_result
        message (old conversation, results not persisted) is stripped."""
        messages = [
            {"role": "assistant", "content": [{"type": "text", "text": "answer"}, _tool_use("a")]},
            {"role": "user", "content": "next question"},
        ]
        out = _sanitize_tool_blocks(messages)
        assert all(b.get("type") != "tool_use" for b in out[0]["content"])
        # the text block survives
        assert any(b.get("type") == "text" for b in out[0]["content"])

    def test_orphan_tool_result_stripped(self):
        messages = [
            {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
            {"role": "user", "content": [_tool_result("a"), {"type": "text", "text": "q"}]},
        ]
        out = _sanitize_tool_blocks(messages)
        assert all(b.get("type") != "tool_result" for b in out[1]["content"])

    def test_partial_match_only_unmatched_stripped(self):
        messages = [
            {"role": "assistant", "content": [_tool_use("a"), _tool_use("b")]},
            {"role": "user", "content": [_tool_result("a")]},
        ]
        out = _sanitize_tool_blocks(messages)
        kept_uses = {b["id"] for b in out[0]["content"] if b.get("type") == "tool_use"}
        assert kept_uses == {"a"}


class TestStripToolUseMarkers:
    """The '*[Using tool: ...]*' markers are display-only. They must be removed from
    replayed assistant content so the model never learns to imitate them as prose
    instead of emitting real tool_use blocks (the fabrication failure mode)."""

    def test_strips_marker_from_string_content(self):
        messages = [
            {"role": "assistant", "content": "*[Using tool: get_variant_annotations; variant: 8:1:C:T]*\n\nThe answer is 42."},
        ]
        out = _strip_tool_use_markers(messages)
        assert "Using tool" not in out[0]["content"]
        assert "The answer is 42." in out[0]["content"]

    def test_strips_multiline_marker_with_sql(self):
        """Marker params can span lines (SQL) — DOTALL + non-greedy must still match."""
        text = (
            "*[Using tool: query_database; sql: SELECT a, b\nFROM t\nWHERE x = '1:2:C:T'\nLIMIT 200]*"
            "\n\n## Result\nrows: 0"
        )
        out = _strip_tool_use_markers([{"role": "assistant", "content": text}])
        assert "Using tool" not in out[0]["content"]
        assert "## Result" in out[0]["content"]

    def test_strips_marker_from_text_block(self):
        messages = [{
            "role": "assistant",
            "content": [
                {"type": "text", "text": "*[Using tool: x; a: b]*\n\nreal prose"},
                {"type": "tool_use", "id": "a", "name": "x", "input": {}},
            ],
        }]
        out = _strip_tool_use_markers(messages)
        text_blocks = [b for b in out[0]["content"] if b.get("type") == "text"]
        assert text_blocks[0]["text"] == "real prose"
        # real tool_use is untouched
        assert any(b.get("type") == "tool_use" for b in out[0]["content"])

    def test_marker_only_text_block_dropped(self):
        messages = [{
            "role": "assistant",
            "content": [
                {"type": "text", "text": "*[Using tool: x; a: b]*"},
                {"type": "tool_use", "id": "a", "name": "x", "input": {}},
            ],
        }]
        out = _strip_tool_use_markers(messages)
        assert all(b.get("type") != "text" for b in out[0]["content"])
        assert any(b.get("type") == "tool_use" for b in out[0]["content"])

    def test_marker_only_string_falls_back_to_original(self):
        """Never emit empty content — a turn that was nothing but a marker keeps
        its original content rather than becoming an empty (API-invalid) message."""
        messages = [{"role": "assistant", "content": "*[Using tool: x; a: b]*"}]
        out = _strip_tool_use_markers(messages)
        assert out[0]["content"] != ""

    def test_user_and_string_messages_untouched(self):
        messages = [
            {"role": "user", "content": "*[Using tool: x]* (user typed this, leave it)"},
            {"role": "assistant", "content": "no markers here"},
        ]
        out = _strip_tool_use_markers(messages)
        assert out[0]["content"] == "*[Using tool: x]* (user typed this, leave it)"
        assert out[1]["content"] == "no markers here"


class TestMarkHistoryCacheBreakpoint:
    def test_marks_last_block_of_last_message(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
        _mark_history_cache_breakpoint(messages)
        assert messages[-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}

    def test_normalizes_string_content(self):
        messages = [{"role": "user", "content": "hello"}]
        _mark_history_cache_breakpoint(messages)
        last = messages[-1]["content"]
        assert isinstance(last, list)
        assert last[-1]["type"] == "text"
        assert last[-1]["cache_control"] == {"type": "ephemeral"}

    def test_empty_messages_noop(self):
        messages = []
        _mark_history_cache_breakpoint(messages)
        assert messages == []
