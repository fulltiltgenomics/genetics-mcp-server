"""Tests for the Anthropic streaming loop: thinking keepalives, max_tokens continuation,
and resuming a turn that presented unfilled results without calling a tool.

These cover `LLMService._stream_anthropic`, which the rest of the suite bypasses by
mocking `stream_chat` wholesale.
"""

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
