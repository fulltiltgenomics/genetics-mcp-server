"""Tests for the Anthropic streaming loop: thinking keepalives, max_tokens continuation,
and resuming a turn that presented unfilled results without calling a tool.

These cover `LLMService._stream_anthropic`, which the rest of the suite bypasses by
mocking `stream_chat` wholesale.
"""

import asyncio
import json
from types import SimpleNamespace

import pytest

from genetics_mcp_server.llm_service import LLMService, _has_unfilled_output


def _delta_event(delta_type, value):
    delta = SimpleNamespace(type=delta_type)
    setattr(delta, "text" if delta_type == "text_delta" else "thinking", value)
    return SimpleNamespace(type="content_block_delta", delta=delta)


class _Block:
    def __init__(self, block_type, **fields):
        self.type = block_type
        self._fields = fields
        # the SDK exposes block fields as attributes (block.text, block.input, ...)
        self.__dict__.update(fields)

    def model_dump(self, exclude_none=False):
        return {"type": self.type, **self._fields}


class _FakeMessage:
    def __init__(self, content, stop_reason):
        self.content = content
        self.stop_reason = stop_reason
        self.usage = SimpleNamespace(
            input_tokens=10,
            output_tokens=20,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        )


class _FakeStream:
    """One `messages.stream(...)` call: async context manager + async iterator."""

    def __init__(self, events, message):
        self._events = events
        self._message = message

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def __aiter__(self):
        for event in self._events:
            yield event

    async def get_final_message(self):
        return self._message


class _FakeMessages:
    def __init__(self, turns):
        self._turns = list(turns)
        self.calls = []

    def stream(self, **params):
        self.calls.append(params)
        events, message = self._turns.pop(0)
        return _FakeStream(events, message)


def _service(turns, executor=None):
    svc = LLMService.__new__(LLMService)
    svc.openai_client = None
    svc.executor = executor
    svc.subagent_service = None
    svc.anthropic_client = SimpleNamespace(messages=_FakeMessages(turns))
    return svc


def _text_turn(text, stop_reason="end_turn"):
    return ([_delta_event("text_delta", text)], _FakeMessage([_Block("text", text=text)], stop_reason))


async def _collect(svc):
    chunks = []
    async for chunk in svc._stream_anthropic(
        messages=[{"role": "user", "content": "hi"}],
        model="claude-opus-5",
        system_prompt=None,
        enable_tools=False,
    ):
        chunks.append(chunk)
    return chunks


@pytest.mark.asyncio
async def test_text_deltas_stream_through():
    turns = [(
        [_delta_event("text_delta", "Hello "), _delta_event("text_delta", "world")],
        _FakeMessage([_Block("text", text="Hello world")], "end_turn"),
    )]
    chunks = await _collect(_service(turns))
    text = "".join(c.content for c in chunks if c.type == "text")
    assert text == "Hello world"
    assert [c.type for c in chunks].count("done") == 1


@pytest.mark.asyncio
async def test_thinking_deltas_emit_throttled_keepalive():
    """Two rapid thinking deltas produce one keepalive, and no reasoning text leaks."""
    turns = [(
        [
            _delta_event("thinking_delta", "step one"),
            _delta_event("thinking_delta", "step two"),
            _delta_event("text_delta", "answer"),
        ],
        _FakeMessage([_Block("text", text="answer")], "end_turn"),
    )]
    chunks = await _collect(_service(turns))
    keepalives = [c for c in chunks if c.type == "thinking"]
    assert len(keepalives) == 1
    assert keepalives[0].content == ""
    assert "step one" not in "".join(c.content for c in chunks if c.type == "text")


@pytest.mark.asyncio
async def test_thinking_blocks_are_not_persisted():
    turns = [(
        [_delta_event("text_delta", "answer")],
        _FakeMessage(
            [_Block("thinking", thinking="secret", signature="sig"), _Block("text", text="answer")],
            "end_turn",
        ),
    )]
    chunks = await _collect(_service(turns))
    done = next(c for c in chunks if c.type == "done")
    assert [b["type"] for b in done.message_content] == ["text"]


@pytest.mark.asyncio
async def test_max_tokens_turn_is_continued():
    """A truncated turn is resumed, and the resume request ends on a user turn."""
    turns = [
        (
            [_delta_event("text_delta", "first half")],
            _FakeMessage([_Block("text", text="first half")], "max_tokens"),
        ),
        (
            [_delta_event("text_delta", " second half")],
            _FakeMessage([_Block("text", text=" second half")], "end_turn"),
        ),
    ]
    svc = _service(turns)
    chunks = await _collect(svc)

    text = "".join(c.content for c in chunks if c.type == "text")
    assert text == "first half second half"
    # no truncation notice: the continuation completed the turn
    assert "cut short" not in text

    # a trailing assistant message would be a prefill, which Opus 4.6+ rejects
    resume_messages = svc.anthropic_client.messages.calls[1]["messages"]
    assert resume_messages[-1]["role"] == "user"
    assert resume_messages[-2]["role"] == "assistant"


@pytest.mark.asyncio
async def test_continuations_are_bounded_and_reported(monkeypatch):
    """When the cap keeps being hit, stop and tell the user instead of looping."""
    from dataclasses import replace

    from genetics_mcp_server.config import get_settings

    capped = replace(get_settings(), max_continuations=2)
    monkeypatch.setattr(
        "genetics_mcp_server.llm_service.get_settings", lambda: capped
    )

    turns = [
        (
            [_delta_event("text_delta", f"part{i} ")],
            _FakeMessage([_Block("text", text=f"part{i} ")], "max_tokens"),
        )
        for i in range(3)
    ]
    svc = _service(turns)
    chunks = await _collect(svc)

    text = "".join(c.content for c in chunks if c.type == "text")
    assert "cut short by the output token limit" in text
    # initial turn + 2 continuations, then it gives up
    assert len(svc.anthropic_client.messages.calls) == 3
    done = next(c for c in chunks if c.type == "done")
    assert any("cut short" in b.get("text", "") for b in done.message_content)


# shapes taken from the 2026-07-20 session that motivated the guard, and from the
# turns in the same history that must NOT trigger it
PLACEHOLDER_CELLS = """Here's what I have:

| Gene | Coloc? | Direction | Endpoints |
|---|---|---|---|
| **MTNR1B** | *[from query]* | | |
| **LGR4** | Yes (from earlier) | | I9_HYPTENS(ESS) |

I haven't surfaced the actual query output — let me pull the concrete rows."""

HEADER_ONLY = """## Unique genes by data source

| Data source | Trait type | Unique genes (P<1e-4) | Unique genes (P<1e-6) |
|---|---|---:|---:|

The table is empty because I need to actually run the query. Let me pull the counts."""

FILLED = """| Gene | beta | p-value |
|---|---:|---:|
| CHRM4 | 2.053 | 1.4e-6 |
| ADGRL1 | 1.598 | 1.0e-7 |"""

CITATIONS = """| Finding | Source |
|---|---|
| IRF7 hypomethylation in SLE renal involvement | [PMC5819620](https://pmc.ncbi.nlm.nih.gov/) |"""

LABELLED_VALUE = """| Result | **0 rows** |
|---|---|

No credible sets passed the PIP threshold."""

QUESTION_ENDING = (
    "Want me to (a) pull the SCHEMA effect sizes as a clean table, or "
    "(b) check whether these three genes share a pathway? Just say the word and I'll run it."
)


@pytest.mark.parametrize(
    "text,expected",
    [
        (PLACEHOLDER_CELLS, True),
        (HEADER_ONLY, True),
        (FILLED, False),
        (CITATIONS, False),  # markdown links are data, not placeholders
        (LABELLED_VALUE, False),  # two-column header carrying the value itself
        (QUESTION_ENDING, False),  # a promise/question with no table is a legitimate stop
        ("", False),
    ],
)
def test_unfilled_output_detection(text, expected):
    assert _has_unfilled_output(text) is expected


@pytest.mark.asyncio
async def test_unfilled_results_turn_is_continued():
    """A turn that tables up placeholders without calling a tool gets resumed."""
    from genetics_mcp_server.config.defaults import CONTINUE_UNFILLED_PROMPT

    svc = _service([_text_turn(PLACEHOLDER_CELLS), _text_turn(FILLED)], executor=object())
    chunks = await _collect(svc)

    text = "".join(c.content for c in chunks if c.type == "text")
    assert "CHRM4 | 2.053" in text
    assert len(svc.anthropic_client.messages.calls) == 2

    resume_messages = svc.anthropic_client.messages.calls[1]["messages"]
    assert resume_messages[-1] == {"role": "user", "content": CONTINUE_UNFILLED_PROMPT}
    assert resume_messages[-2]["role"] == "assistant"


@pytest.mark.asyncio
async def test_filled_results_turn_is_not_continued():
    svc = _service([_text_turn(FILLED)], executor=object())
    await _collect(svc)
    assert len(svc.anthropic_client.messages.calls) == 1


@pytest.mark.asyncio
async def test_turn_ending_in_an_offer_is_not_continued():
    """The failure mode this guard replaced: resuming on "let me pull ..." phrasing
    would answer over the top of a turn that is correctly waiting for the user."""
    svc = _service([_text_turn(QUESTION_ENDING)], executor=object())
    await _collect(svc)
    assert len(svc.anthropic_client.messages.calls) == 1


@pytest.mark.asyncio
async def test_unfilled_turn_is_not_continued_without_tools():
    """With no executor there is nothing to resume with, so the turn stands."""
    svc = _service([_text_turn(PLACEHOLDER_CELLS)], executor=None)
    await _collect(svc)
    assert len(svc.anthropic_client.messages.calls) == 1


@pytest.mark.asyncio
async def test_unfilled_continuations_are_bounded_and_reported(monkeypatch):
    """A model that keeps emitting placeholders is stopped and the user is told."""
    from dataclasses import replace

    from genetics_mcp_server.config import get_settings

    capped = replace(get_settings(), max_continuations=1)
    monkeypatch.setattr("genetics_mcp_server.llm_service.get_settings", lambda: capped)

    svc = _service([_text_turn(PLACEHOLDER_CELLS) for _ in range(3)], executor=object())
    chunks = await _collect(svc)

    # initial turn + 1 continuation, then it gives up
    assert len(svc.anthropic_client.messages.calls) == 2
    text = "".join(c.content for c in chunks if c.type == "text")
    assert "left unfilled" in text
    done = next(c for c in chunks if c.type == "done")
    assert any("left unfilled" in b.get("text", "") for b in done.message_content)


@pytest.mark.asyncio
async def test_adaptive_thinking_requested_for_supporting_model():
    turns = [(
        [_delta_event("text_delta", "hi")],
        _FakeMessage([_Block("text", text="hi")], "end_turn"),
    )]
    svc = _service(turns)
    await _collect(svc)
    assert svc.anthropic_client.messages.calls[0]["thinking"] == {
        "type": "adaptive",
        "display": "summarized",
    }


@pytest.mark.asyncio
async def test_adaptive_thinking_omitted_for_older_model():
    turns = [(
        [_delta_event("text_delta", "hi")],
        _FakeMessage([_Block("text", text="hi")], "end_turn"),
    )]
    svc = _service(turns)
    async for _ in svc._stream_anthropic(
        messages=[{"role": "user", "content": "hi"}],
        model="claude-haiku-4-5",
        system_prompt=None,
        enable_tools=False,
    ):
        pass
    assert "thinking" not in svc.anthropic_client.messages.calls[0]


# ------------------------------- per-iteration timing and the script_result chunk (4h6.73, 4h6.71)


_TOOL_PHASE_S = 0.12


def _run_analysis_turn(tool_use_id="ra-1"):
    block = _Block("tool_use", id=tool_use_id, name="run_analysis", input={"code": "print(1)"})
    return ([], _FakeMessage([block], "tool_use"))


async def _collect_with_tool(svc):
    chunks = []
    async for chunk in svc._stream_anthropic(
        messages=[{"role": "user", "content": "hi"}],
        model="claude-opus-5",
        system_prompt=None,
        enable_tools=False,
    ):
        chunks.append(chunk)
    return chunks


def _tooled_service(turns, tool_result):
    """A service whose only tool call is `run_analysis`, answered by `tool_result`.

    `_execute_tool` is replaced rather than the executor stubbed: the point of these tests
    is the streaming loop's chunk emission and its clock, not tool dispatch. The sleep makes
    the tool phase measurable, which is what separates `model_ms` from `turn_elapsed_ms`.
    """
    svc = _service(turns, executor=SimpleNamespace())

    async def _execute_tool(name, tool_input, *args, **kwargs):
        await asyncio.sleep(_TOOL_PHASE_S)
        return tool_result

    svc._execute_tool = _execute_tool
    return svc


@pytest.mark.asyncio
async def test_usage_chunks_carry_turn_elapsed_and_model_time_separately():
    """`turn_elapsed_ms` is cumulative from the turn's start; `model_ms` is this call alone.

    The tool phase sits BETWEEN iteration 1's usage chunk and iteration 2's model call, so
    it must appear in the growth of turn_elapsed_ms and must NOT appear in either model_ms.
    That is the property that tells the two epochs apart: were turn_elapsed_ms a
    per-iteration delta, or model_ms measured across the tool phase, this would fail.
    """
    turns = [_run_analysis_turn(), _text_turn("answer")]
    svc = _tooled_service(turns, {"success": True, "status": "ok", "output": "1"})
    chunks = await _collect_with_tool(svc)

    usages = [json.loads(c.content) for c in chunks if c.type == "usage"]
    assert len(usages) == 2
    first, second = usages

    assert first["turn_elapsed_ms"] < second["turn_elapsed_ms"]
    # each model call is bounded by the elapsed reading taken right after it
    assert first["model_ms"] <= first["turn_elapsed_ms"]
    assert second["model_ms"] <= second["turn_elapsed_ms"] - first["turn_elapsed_ms"] + 1

    tool_phase_ms = second["turn_elapsed_ms"] - first["turn_elapsed_ms"] - second["model_ms"]
    assert tool_phase_ms >= _TOOL_PHASE_S * 1000 * 0.9
    # the sleep is in the tool phase, not in either model call
    assert second["model_ms"] < _TOOL_PHASE_S * 1000


@pytest.mark.asyncio
async def test_model_attempts_makes_a_retry_inflated_model_ms_identifiable():
    """`model_ms` is not model latency: the span encloses the retry loop's backoff sleep.

    One transient failure costs a 1s `asyncio.sleep` that lands inside the figure, so a
    reader diffing "model time" across arms is diffing something that includes it. The
    attempt count is on the wire beside it precisely so that reading is identifiable rather
    than merely disclaimed. Both halves are asserted: the inflation is real, and the field
    reports it.
    """
    import httpx
    from anthropic import APIConnectionError

    turns = [_text_turn("answer")]
    svc = _service(turns)
    inner = svc.anthropic_client.messages
    failures = {"left": 1}

    def stream(**params):
        if failures["left"]:
            failures["left"] -= 1
            raise APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com"))
        return _FakeMessages.stream(inner, **params)

    svc.anthropic_client = SimpleNamespace(messages=SimpleNamespace(stream=stream))

    usage = json.loads(next(c.content for c in await _collect(svc) if c.type == "usage"))
    assert usage["model_attempts"] == 2
    # the first backoff is 2**0 = 1s, and it is inside model_ms — which is the whole point
    assert usage["model_ms"] >= 900


@pytest.mark.asyncio
async def test_model_attempts_is_one_when_the_call_succeeds_first_time():
    usage = json.loads(
        next(c.content for c in await _collect(_service([_text_turn("hi")])) if c.type == "usage")
    )
    assert usage["model_attempts"] == 1


@pytest.mark.asyncio
async def test_run_analysis_emits_one_script_result_chunk_before_the_next_usage():
    """Ordering is what the retry-loop counter reads: failure, then a further roundtrip."""
    turns = [_run_analysis_turn(), _text_turn("sorry")]
    failed = {
        "success": False,
        "status": "error",
        "output": "",
        "error": "boom",
        "error_type": "ValueError",
        "duration_ms": 7,
    }
    svc = _tooled_service(turns, failed)
    chunks = await _collect_with_tool(svc)

    ordered = [c.type for c in chunks if c.type in ("usage", "script_result", "done")]
    assert ordered == ["usage", "script_result", "usage", "done"]

    payload = json.loads(next(c.content for c in chunks if c.type == "script_result"))
    assert payload["iteration"] == 1
    assert payload["ran"] is True and payload["ok"] is False
    assert payload["exception"] == "ValueError"
    assert payload["duration_ms"] == 7


@pytest.mark.asyncio
async def test_a_tool_that_is_not_run_analysis_emits_no_script_result_chunk():
    block = _Block("tool_use", id="t1", name="get_variants", input={})
    turns = [([], _FakeMessage([block], "tool_use")), _text_turn("answer")]
    svc = _tooled_service(turns, {"success": True, "results": []})
    chunks = await _collect_with_tool(svc)
    assert not [c for c in chunks if c.type == "script_result"]
