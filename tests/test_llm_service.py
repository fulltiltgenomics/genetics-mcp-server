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
