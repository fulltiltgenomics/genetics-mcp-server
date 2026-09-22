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


async def _collect(svc, **kwargs):
    chunks = []
    async for chunk in svc._stream_anthropic(
        messages=[{"role": "user", "content": "hi"}],
        model="claude-opus-5",
        system_prompt=None,
        enable_tools=False,
        code_execution=False,
        **kwargs,
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
async def test_reasoning_text_needs_an_explicit_opt_in():
    """The UI path: without capture_thinking the summary leaves the process nowhere."""
    turns = [(
        [_delta_event("text_delta", "answer")],
        _FakeMessage(
            [_Block("thinking", thinking="secret", signature="sig"), _Block("text", text="answer")],
            "end_turn",
        ),
    )]
    chunks = await _collect(_service(turns))
    assert not [c for c in chunks if c.type == "thinking_summary"]
    assert "secret" not in "".join(c.content for c in chunks)


@pytest.mark.asyncio
async def test_capture_thinking_emits_the_summary_and_still_never_persists_it():
    """The benchmark's opt-in changes what is EMITTED, never what is stored: a caller that
    asks for reasoning must not be able to write it into a conversation or replay it."""
    turns = [(
        [_delta_event("text_delta", "answer")],
        _FakeMessage(
            [_Block("thinking", thinking="secret", signature="sig"), _Block("text", text="answer")],
            "end_turn",
        ),
    )]
    chunks = await _collect(_service(turns), capture_thinking=True)

    summaries = [c for c in chunks if c.type == "thinking_summary"]
    assert len(summaries) == 1
    payload = json.loads(summaries[0].content)
    assert payload["text"] == "secret"
    assert payload["iteration"] == 1

    done = next(c for c in chunks if c.type == "done")
    assert [b["type"] for b in done.message_content] == ["text"], (
        "thinking must stay out of message_content even when it is being streamed"
    )


@pytest.mark.asyncio
async def test_redacted_thinking_is_not_emitted_as_an_empty_summary():
    """Its payload is encrypted, so there is no text to show and no chunk to send."""
    turns = [(
        [_delta_event("text_delta", "answer")],
        _FakeMessage(
            [_Block("redacted_thinking", data="ENCRYPTED"), _Block("text", text="answer")],
            "end_turn",
        ),
    )]
    chunks = await _collect(_service(turns), capture_thinking=True)
    assert not [c for c in chunks if c.type == "thinking_summary"]


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
        code_execution=False,
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
        code_execution=False,
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
async def test_the_script_result_names_the_tool_use_it_belongs_to():
    """An iteration can hold more than one run_analysis, so the iteration number does not
    identify one. The client attaches the outcome to a specific collapsed tool call."""
    turns = [_run_analysis_turn(tool_use_id="ra-7"), _text_turn("done")]
    svc = _tooled_service(turns, {"success": True, "status": "ok", "output": "1"})
    chunks = await _collect_with_tool(svc)

    payload = json.loads(next(c.content for c in chunks if c.type == "script_result"))
    tool_use = json.loads(next(c.content for c in chunks if c.type == "tool_use"))
    assert payload["tool_use_id"] == "ra-7" == tool_use["id"]


@pytest.mark.asyncio
async def test_image_artifacts_are_streamed_and_never_reach_the_tool_result():
    """The base64 is for the browser. In the tool_result it is tokens the model pays for
    and cannot see — the same reason `image_base64` is stripped on the single-plot path."""
    data = "aW1hZ2UtYnl0ZXM" + "A" * 200
    turns = [_run_analysis_turn(), _text_turn("the plot shows a peak")]
    svc = _tooled_service(
        turns,
        {
            "success": True,
            "status": "ok",
            "output": "done\n",
            "images": [
                {"name": "locus.png", "content_type": "image/png", "content_base64": data},
                {"name": "qq.svg", "content_type": "image/svg+xml", "content_base64": data},
            ],
        },
    )
    chunks = await _collect_with_tool(svc)

    images = [c for c in chunks if c.type == "image"]
    assert [c.image_alt for c in images] == ["locus.png", "qq.svg"]
    assert [c.image_format for c in images] == ["png", "svg+xml"]
    assert all(c.content == data for c in images)

    done = next(c for c in chunks if c.type == "done")
    serialised = json.dumps(done.tool_results)
    assert data not in serialised
    assert "images" not in serialised
    assert "displayed to the user" in serialised


@pytest.mark.asyncio
async def test_a_malformed_image_entry_is_skipped_without_losing_the_result():
    turns = [_run_analysis_turn(), _text_turn("done")]
    svc = _tooled_service(
        turns,
        {
            "success": True,
            "status": "ok",
            "output": "done\n",
            "images": ["not a dict", {"name": "x.png"}, {"content_base64": "tiny"}],
        },
    )
    chunks = await _collect_with_tool(svc)

    assert not [c for c in chunks if c.type == "image"]
    done = next(c for c in chunks if c.type == "done")
    (result,) = done.tool_results
    assert json.loads(result["content"])["output"] == "done\n"


@pytest.mark.asyncio
async def test_a_tool_that_is_not_run_analysis_emits_no_script_result_chunk():
    block = _Block("tool_use", id="t1", name="get_variants", input={})
    turns = [([], _FakeMessage([block], "tool_use")), _text_turn("answer")]
    svc = _tooled_service(turns, {"success": True, "results": []})
    chunks = await _collect_with_tool(svc)
    assert not [c for c in chunks if c.type == "script_result"]


@pytest.mark.asyncio
async def test_a_refused_turn_ends_with_a_notice_and_is_not_resumed():
    """`stop_reason: refusal` with empty content is a declined request, not an answer."""
    message = _FakeMessage([], "refusal")
    message.stop_details = SimpleNamespace(category="bio")
    svc = _service([([], message)])
    chunks = await _collect(svc)

    text = "".join(c.content for c in chunks if c.type == "text")
    assert "declined this request (bio)" in text
    assert len(svc.anthropic_client.messages.calls) == 1
    assert [c.type for c in chunks].count("done") == 1


@pytest.mark.asyncio
async def test_a_mid_stream_refusal_keeps_the_partial_text_and_runs_no_tool():
    block = _Block("tool_use", id="t1", name="get_variants", input={})
    message = _FakeMessage([_Block("text", text="Partial"), block], "refusal")
    svc = _tooled_service([([_delta_event("text_delta", "Partial")], message)], {"success": True})
    chunks = await _collect_with_tool(svc)

    text = "".join(c.content for c in chunks if c.type == "text")
    assert text.startswith("Partial")
    assert "declined this request" in text
    assert not [c for c in chunks if c.type == "tool_use"]
    assert len(svc.anthropic_client.messages.calls) == 1


# --- server-side refusal fallback -------------------------------------------------------

from genetics_mcp_server.llm_service import _refusal_fallback_params, _replayable_content


def _fallback_block(declined="claude-fable-5-1", served="claude-opus-5"):
    return _Block("fallback", **{"from": {"model": declined}, "to": {"model": served}})


def _fallback_start_event(block):
    return SimpleNamespace(type="content_block_start", content_block=block)


class TestRefusalFallbackParams:
    def test_default_mode_carries_its_own_beta_header(self):
        params = _refusal_fallback_params("claude-fable-5-1", "default")
        assert params["extra_body"] == {"fallbacks": "default"}
        assert params["extra_headers"] == {"anthropic-beta": "server-side-fallback-2026-07-01"}

    def test_a_pinned_model_uses_the_array_form_and_header(self):
        params = _refusal_fallback_params("claude-opus-5", "claude-opus-4-8")
        assert params["extra_body"] == {"fallbacks": [{"model": "claude-opus-4-8"}]}
        assert params["extra_headers"] == {"anthropic-beta": "server-side-fallback-2026-06-01"}

    @pytest.mark.parametrize("model", ["claude-haiku-4-5", "claude-opus-4-8", "claude-sonnet-5"])
    def test_models_without_classifiers_get_no_fallback(self, model):
        assert _refusal_fallback_params(model, "default") == {}

    def test_empty_setting_turns_it_off(self):
        assert _refusal_fallback_params("claude-fable-5-1", "") == {}


@pytest.mark.asyncio
async def test_the_chat_request_opts_into_the_fallback_by_default():
    svc = _service([_text_turn("ok")])
    await _collect(svc)
    call = svc.anthropic_client.messages.calls[0]
    assert call["extra_body"] == {"fallbacks": "default"}
    assert "server-side-fallback" in call["extra_headers"]["anthropic-beta"]


@pytest.mark.asyncio
async def test_a_pre_output_fallback_is_announced_before_the_answer_and_priced_as_served():
    marker = _fallback_block()
    message = _FakeMessage([marker, _Block("text", text="Answer")], "end_turn")
    message.model = "claude-opus-5"
    turns = [([_fallback_start_event(marker), _delta_event("text_delta", "Answer")], message)]
    svc = _service(turns)
    chunks = await _collect(svc)

    text = "".join(c.content for c in chunks if c.type == "text")
    assert text.index("Claude Fable 5.1 declined this request; Claude Opus 5 answered instead") < text.index("Answer")
    done = next(c for c in chunks if c.type == "done")
    persisted = [b["text"] for b in done.message_content if b["type"] == "text"]
    assert "declined this request" in persisted[0]
    assert persisted[1] == "Answer"
    assert not [b for b in done.message_content if b["type"] == "fallback"]
    usage = json.loads(next(c for c in chunks if c.type == "usage").content)
    assert usage["context_window"] == 1_000_000


@pytest.mark.asyncio
async def test_a_mid_output_fallback_drops_the_declining_models_tool_call():
    """Blocks before the marker were the declined model's; its tool call never ran."""
    stale = _Block("tool_use", id="t0", name="get_variants", input={})
    live = _Block("tool_use", id="t1", name="run_analysis", input={"code": "print(1)"})
    marker = _fallback_block()
    turn1 = _FakeMessage([_Block("text", text="Start"), stale, marker, live], "tool_use")
    turn1.model = "claude-opus-5"
    svc = _tooled_service([([], turn1), _text_turn("done")], {"success": True, "output": "1"})
    chunks = await _collect_with_tool(svc)

    emitted = [json.loads(c.content)["id"] for c in chunks if c.type == "tool_use"]
    assert emitted == ["t1"]
    replayed = svc.anthropic_client.messages.calls[1]["messages"][-2]["content"]
    assert [b["type"] for b in replayed] == ["text", "tool_use"]
    assert replayed[1]["id"] == "t1"
    done = next(c for c in chunks if c.type == "done")
    assert [b.get("id") for b in done.message_content if b["type"] == "tool_use"] == ["t1"]


def test_replayable_content_keeps_everything_when_nothing_fell_back():
    content = [_Block("thinking", thinking="", signature="s"), _Block("text", text="a")]
    assert [b["type"] for b in _replayable_content(content)] == ["thinking", "text"]


@pytest.mark.asyncio
async def test_a_sticky_fallback_turn_is_announced_after_the_answer():
    message = _FakeMessage([_Block("text", text="Answer")], "end_turn")
    message.model = "claude-opus-5"
    svc = _service([([_delta_event("text_delta", "Answer")], message)])
    chunks = []
    async for chunk in svc._stream_anthropic(
        messages=[{"role": "user", "content": "hi"}],
        model="claude-fable-5-1",
        system_prompt=None,
        enable_tools=False,
        code_execution=False,
    ):
        chunks.append(chunk)

    text = "".join(c.content for c in chunks if c.type == "text")
    assert text.startswith("Answer")
    assert "[Answered by Claude Opus 5]" in text


# The output cap can land inside a tool call's streamed arguments, not only after the
# text. The turn then carries a tool_use whose input never finished arriving — commonly
# {} — and the earlier guard, which resumed only when NO tool_use was present, let that
# call through to dispatch. Staging session ff82f2a0 (2026-09-10) ran that loop seven
# times at the full output cap: each dispatch raised "missing 1 required positional
# argument: 'code'", which reads as a server fault, so the model reissued the same
# oversized call. $12.73 of a $19.68 turn bought nothing.


def _truncated_tool_call_turn(text="Here is the run properly", tool_input=None):
    """A max_tokens turn whose tool_use arguments were cut off mid-stream."""
    blocks = [
        _Block("text", text=text),
        _Block("tool_use", id="ra-cut", name="run_analysis", input=tool_input or {}),
    ]
    return ([_delta_event("text_delta", text)], _FakeMessage(blocks, "max_tokens"))


@pytest.mark.asyncio
async def test_tool_call_truncated_by_max_tokens_is_never_dispatched():
    """The half-written call is dropped, not run with whatever arguments arrived."""
    turns = [_truncated_tool_call_turn(), _text_turn("answer")]
    svc = _service(turns, executor=SimpleNamespace())

    dispatched = []

    async def _execute_tool(name, tool_input, *args, **kwargs):
        dispatched.append((name, tool_input))
        return {"success": True, "status": "ok", "output": "1"}

    svc._execute_tool = _execute_tool
    chunks = await _collect(svc)

    assert dispatched == [], "a truncated tool call must not reach the executor"
    text = "".join(c.content for c in chunks if c.type == "text")
    assert text.endswith("answer")


@pytest.mark.asyncio
async def test_truncated_tool_call_is_dropped_from_the_replay():
    """A replayed tool_use with no matching tool_result is rejected by the API."""
    turns = [_truncated_tool_call_turn(), _text_turn("answer")]
    svc = _service(turns, executor=SimpleNamespace())
    svc._execute_tool = lambda *a, **k: None
    await _collect(svc)

    resume = svc.anthropic_client.messages.calls[1]["messages"]
    assert resume[-1]["role"] == "user"
    assistant = resume[-2]
    assert assistant["role"] == "assistant"
    assert all(b["type"] != "tool_use" for b in assistant["content"])


@pytest.mark.asyncio
async def test_truncated_tool_call_prompt_names_truncation_not_a_missing_argument():
    """The message the model gets is the whole reason the loop broke or repeated."""
    turns = [_truncated_tool_call_turn(), _text_turn("answer")]
    svc = _service(turns, executor=SimpleNamespace())
    svc._execute_tool = lambda *a, **k: None
    await _collect(svc)

    sent = svc.anthropic_client.messages.calls[1]["messages"][-1]["content"]
    assert "output token limit" in sent
    assert "was not run" in sent
    assert "smaller" in sent


@pytest.mark.asyncio
async def test_truncated_tool_call_with_no_other_content_still_replays():
    """A turn that spent its whole budget inside the arguments leaves nothing to echo."""
    blocks = [_Block("tool_use", id="ra-cut", name="run_analysis", input={})]
    turns = [([], _FakeMessage(blocks, "max_tokens")), _text_turn("answer")]
    svc = _service(turns, executor=SimpleNamespace())
    svc._execute_tool = lambda *a, **k: None
    await _collect(svc)

    assistant = svc.anthropic_client.messages.calls[1]["messages"][-2]
    assert assistant["content"], "an empty assistant message is rejected by the API"


def _budgets(monkeypatch, finish, hard):
    """Pin both turn budgets; estimate_cost of the fake usage is tiny but positive, so a
    budget of 1e-9 trips on iteration 1 and 0 disables it."""
    from dataclasses import replace

    from genetics_mcp_server.config import get_settings

    capped = replace(get_settings(), max_turn_cost_usd=finish, max_turn_cost_hard_usd=hard)
    monkeypatch.setattr("genetics_mcp_server.llm_service.get_settings", lambda: capped)


@pytest.mark.asyncio
async def test_turn_stops_when_it_reaches_the_hard_cap(monkeypatch):
    """Bounds the bill for failure shapes no specific guard anticipated."""
    _budgets(monkeypatch, finish=0.0, hard=1e-9)

    turns = [_run_analysis_turn(), _text_turn("never reached")]
    svc = _service(turns, executor=SimpleNamespace())
    svc._execute_tool = lambda *a, **k: None
    chunks = await _collect(svc)

    assert len(svc.anthropic_client.messages.calls) == 1, "must not call the model again"
    text = "".join(c.content for c in chunks if c.type == "text")
    assert "cost limit" in text
    done = next(c for c in chunks if c.type == "done")
    assert any("cost limit" in b.get("text", "") for b in done.message_content)


@pytest.mark.asyncio
async def test_hard_cap_is_checked_before_the_finish_budget(monkeypatch):
    """A hard cap at or below the finish budget is a stop, not a finish."""
    _budgets(monkeypatch, finish=1e-9, hard=1e-9)

    turns = [_run_analysis_turn(), _text_turn("never reached")]
    svc = _service(turns, executor=SimpleNamespace())
    svc._execute_tool = lambda *a, **k: None
    chunks = await _collect(svc)

    assert len(svc.anthropic_client.messages.calls) == 1
    text = "".join(c.content for c in chunks if c.type == "text")
    assert "cost limit" in text and "cost budget" not in text


@pytest.mark.asyncio
async def test_hard_cap_crossed_by_the_final_call_reports_nothing(monkeypatch):
    """The call that crossed the cap ended the turn on its own, so nothing was cut."""
    _budgets(monkeypatch, finish=0.0, hard=1e-9)

    chunks = await _collect(_service([_text_turn("answer")]))
    text = "".join(c.content for c in chunks if c.type == "text")
    assert text == "answer"


@pytest.mark.asyncio
async def test_finish_budget_runs_the_pending_tools_then_asks_for_the_answer(monkeypatch):
    """Crossing the finish budget does not drop the tools the model just asked for: their
    results are what the answer is written from. The NEXT call carries them with
    tool calling off and the finish instruction, and the notice comes after the answer."""
    from genetics_mcp_server.config.defaults import FINISH_TURN_PROMPT

    _budgets(monkeypatch, finish=1e-9, hard=0.0)

    turns = [_run_analysis_turn(), _text_turn("final answer")]
    svc = _tooled_service(turns, {"success": True, "status": "ok", "output": "1"})
    chunks = await _collect_with_tool(svc)

    calls = svc.anthropic_client.messages.calls
    assert len(calls) == 2
    assert "tool_choice" not in calls[0]
    assert calls[1]["tool_choice"] == {"type": "none"}
    last_user = calls[1]["messages"][-1]
    assert last_user["role"] == "user"
    assert last_user["content"][0]["type"] == "tool_result"
    assert last_user["content"][-1] == {"type": "text", "text": FINISH_TURN_PROMPT}

    text = "".join(c.content for c in chunks if c.type == "text")
    assert text.startswith("final answer")
    assert "cost budget" in text and "cost limit" not in text
    done = next(c for c in chunks if c.type == "done")
    assert any("cost budget" in b.get("text", "") for b in done.message_content)


@pytest.mark.asyncio
async def test_finish_budget_crossed_by_the_final_call_reports_nothing(monkeypatch):
    """A turn that finished on its own is not told it was finished for it."""
    _budgets(monkeypatch, finish=1e-9, hard=0.0)

    chunks = await _collect(_service([_text_turn("answer")]))
    text = "".join(c.content for c in chunks if c.type == "text")
    assert text == "answer"


@pytest.mark.asyncio
async def test_cost_budgets_of_zero_are_disabled(monkeypatch):
    """0 means no cap, so a normal turn is untouched by either."""
    _budgets(monkeypatch, finish=0.0, hard=0.0)

    turns = [_run_analysis_turn(), _text_turn("answer")]
    svc = _tooled_service(turns, {"success": True, "status": "ok", "output": "1"})
    chunks = await _collect_with_tool(svc)
    assert len(svc.anthropic_client.messages.calls) == 2
    assert "tool_choice" not in svc.anthropic_client.messages.calls[1]
    text = "".join(c.content for c in chunks if c.type == "text")
    assert "cost" not in text


# --- client-side refusal retry with fallback credit --------------------------------------
#
# The server-side fallback leaves two refusals standing: a category with no recommended
# fallback, and a streaming decline inside an open tool-use block. Prod 2026-09-22 hit the
# second four turns out of four; each reached the user as "declined this request" with the
# category logged as None, because the SDK's stream accumulator drops `stop_details`.

from genetics_mcp_server.llm_service import (
    _refusal_echo,
    _refusal_retry_ladder,
    _refusal_retry_params,
)


def _refusal_delta(category="bio", explanation=None, token="tok-1", claim=True):
    details = SimpleNamespace(
        type="refusal",
        category=category,
        explanation=explanation,
        fallback_credit_token=token,
        fallback_has_prefill_claim=claim,
    )
    return SimpleNamespace(type="message_delta", delta=SimpleNamespace(stop_details=details))


def _refused_message(content, model="claude-fable-5-1"):
    message = _FakeMessage(content, "refusal")
    message.model = model
    return message


def _bad_request(text):
    import anthropic
    import httpx

    return anthropic.BadRequestError(
        text,
        response=httpx.Response(400, request=httpx.Request("POST", "https://api/v1/messages")),
        body={"error": {"type": "invalid_request_error", "message": text}},
    )


class _RejectingStream(_FakeStream):
    def __init__(self, error):
        self._error = error

    async def __aenter__(self):
        raise self._error


class _FakeMessagesWithRejections(_FakeMessages):
    """A turn entry that is an exception is raised when that call's stream opens."""

    def stream(self, **params):
        self.calls.append(params)
        turn = self._turns.pop(0)
        if isinstance(turn, Exception):
            return _RejectingStream(turn)
        return _FakeStream(*turn)


def _fable_service(turns, executor=None):
    svc = _service(turns, executor=executor)
    svc.anthropic_client = SimpleNamespace(messages=_FakeMessagesWithRejections(turns))
    return svc


async def _collect_fable(svc):
    chunks = []
    async for chunk in svc._stream_anthropic(
        messages=[{"role": "user", "content": "hi"}],
        model="claude-fable-5-1",
        system_prompt=None,
        enable_tools=False,
        code_execution=False,
    ):
        chunks.append(chunk)
    return chunks


def _opus_answer(text="Answer", stop_reason="end_turn", content=None):
    message = _FakeMessage(content or [_Block("text", text=text)], stop_reason)
    message.model = "claude-opus-5"
    return ([_delta_event("text_delta", text)], message)


class TestRefusalCreditParams:
    def test_the_request_opts_into_the_credit_beta_alongside_the_fallback(self):
        params = _refusal_fallback_params("claude-fable-5-1", "default", "claude-opus-5")
        assert params["extra_body"] == {"fallbacks": "default"}
        assert params["extra_headers"]["anthropic-beta"] == (
            "server-side-fallback-2026-07-01,fallback-credit-2026-07-01"
        )

    def test_the_credit_beta_is_sent_even_with_the_server_side_fallback_off(self):
        params = _refusal_fallback_params("claude-fable-5-1", "", "claude-opus-5")
        assert "extra_body" not in params
        assert params["extra_headers"] == {"anthropic-beta": "fallback-credit-2026-07-01"}

    def test_the_echo_drops_tool_calls_and_trailing_whitespace(self):
        content = [
            _Block("thinking", thinking="", signature="s"),
            _Block("tool_use", id="t1", name="run_analysis", input={"code": "x"}),
            _Block("text", text="Partial  \n"),
        ]
        assert _refusal_echo(content) == [
            {"type": "thinking", "thinking": "", "signature": "s"},
            {"type": "text", "text": "Partial"},
        ]

    def test_a_whitespace_only_tail_is_dropped_from_the_echo(self):
        assert _refusal_echo([_Block("text", text="  \n")]) == []

    def test_the_ladder_continues_first_unless_the_claim_is_false(self):
        echo = [{"type": "text", "text": "Partial"}]
        assert _refusal_retry_ladder(True, echo) == ["continue", "exact", "plain"]
        assert _refusal_retry_ladder(None, echo) == ["continue", "exact", "plain"]
        assert _refusal_retry_ladder(False, echo) == ["exact", "plain"]
        assert _refusal_retry_ladder(True, []) == ["exact", "plain"]

    def test_the_retry_keeps_the_body_and_swaps_the_fallback_for_the_credit(self):
        request = {
            "model": "claude-fable-5-1",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10,
            "tools": [{"name": "t"}],
            "extra_body": {"fallbacks": "default"},
            "extra_headers": {"anthropic-beta": "server-side-fallback-2026-07-01,fallback-credit-2026-07-01"},
        }
        echo = [{"type": "text", "text": "Partial"}]
        cont = _refusal_retry_params(request, "claude-opus-5", "tok", "continue", echo)
        assert cont["model"] == "claude-opus-5"
        assert cont["tools"] == request["tools"] and cont["max_tokens"] == 10
        assert cont["extra_body"] == {"fallback_credit_token": "tok"}
        assert cont["extra_headers"] == {"anthropic-beta": "fallback-credit-2026-07-01"}
        assert cont["messages"] == [*request["messages"], {"role": "assistant", "content": echo}]
        exact = _refusal_retry_params(request, "claude-opus-5", "tok", "exact", echo)
        assert exact["messages"] == request["messages"]
        assert exact["extra_body"] == {"fallback_credit_token": "tok"}
        plain = _refusal_retry_params(request, "claude-opus-5", "tok", "plain", echo)
        assert "extra_body" not in plain
        # the refused request is untouched: every rung is built from the same body
        assert request["extra_body"] == {"fallbacks": "default"}
        assert request["messages"] == [{"role": "user", "content": "hi"}]


@pytest.mark.asyncio
async def test_the_refusal_category_is_read_off_the_delta_not_the_final_message():
    """The accumulator never carries `stop_details`; the delta does. No token, no retry."""
    turns = [([_refusal_delta(category="general_harms", explanation="policy", token=None)],
              _refused_message([]))]
    svc = _fable_service(turns)
    chunks = await _collect_fable(svc)

    text = "".join(c.content for c in chunks if c.type == "text")
    assert "declined this request (general_harms): policy" in text
    assert len(svc.anthropic_client.messages.calls) == 1


@pytest.mark.asyncio
async def test_a_standing_refusal_is_retried_on_opus_continuing_the_partial_output():
    stale = _Block("tool_use", id="t0", name="run_analysis", input={"code": "x"})
    refused = _refused_message([_Block("text", text="Partial "), stale])
    turns = [
        ([_delta_event("text_delta", "Partial "), _refusal_delta()], refused),
        _opus_answer("Answer"),
    ]
    svc = _fable_service(turns)
    chunks = await _collect_fable(svc)

    text = "".join(c.content for c in chunks if c.type == "text")
    partial, notice, answer = (
        text.index("Partial"),
        text.index("Claude Fable 5.1 declined this request; Claude Opus 5 answered instead"),
        text.index("Answer"),
    )
    assert partial < notice < answer
    assert "Try rephrasing" not in text
    assert "[Answered by" not in text

    calls = svc.anthropic_client.messages.calls
    assert len(calls) == 2
    retry = calls[1]
    assert retry["model"] == "claude-opus-5"
    assert retry["extra_body"] == {"fallback_credit_token": "tok-1"}
    assert retry["extra_headers"] == {"anthropic-beta": "fallback-credit-2026-07-01"}
    assert retry["messages"][-1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "Partial"}],
    }
    assert retry["messages"][:-1] == calls[0]["messages"]

    done = next(c for c in chunks if c.type == "done")
    persisted = [b for b in done.message_content if b["type"] == "text"]
    assert [b["text"].strip() for b in persisted][0] == "Partial"
    assert "declined this request" in persisted[1]["text"]
    assert persisted[2]["text"] == "Answer"
    assert not [b for b in done.message_content if b["type"] == "tool_use"]
    # both attempts are billed: the refused partial output and the answer
    usage = json.loads(next(c for c in chunks if c.type == "usage").content)
    assert usage["context_window"] == 1_000_000


@pytest.mark.asyncio
async def test_a_refusal_before_any_output_is_retried_from_scratch():
    turns = [([_refusal_delta(claim=False)], _refused_message([])), _opus_answer("Answer")]
    svc = _fable_service(turns)
    chunks = await _collect_fable(svc)

    text = "".join(c.content for c in chunks if c.type == "text")
    assert text.index("declined this request; Claude Opus 5 answered") < text.index("Answer")
    retry = svc.anthropic_client.messages.calls[1]
    assert retry["messages"] == svc.anthropic_client.messages.calls[0]["messages"]
    assert retry["extra_body"] == {"fallback_credit_token": "tok-1"}


@pytest.mark.asyncio
async def test_a_rejected_continuation_falls_down_the_ladder_before_forfeiting_the_credit():
    refused = _refused_message([_Block("text", text="Partial")])
    turns = [
        ([_refusal_delta()], refused),
        _bad_request("request body does not match the refused request"),
        _bad_request("fallback_credit_token is invalid"),
        _opus_answer("Answer"),
    ]
    svc = _fable_service(turns)
    chunks = await _collect_fable(svc)

    text = "".join(c.content for c in chunks if c.type == "text")
    assert text.count("declined this request; Claude Opus 5 answered") == 1
    assert text.endswith("Answer") or "Answer" in text
    calls = svc.anthropic_client.messages.calls
    assert [c["model"] for c in calls] == ["claude-fable-5-1"] + ["claude-opus-5"] * 3
    assert calls[1]["messages"][-1]["role"] == "assistant"
    assert calls[2]["messages"] == calls[0]["messages"]
    assert calls[2]["extra_body"] == {"fallback_credit_token": "tok-1"}
    assert "extra_body" not in calls[3]


@pytest.mark.asyncio
async def test_a_transient_redemption_failure_leaves_the_refusal_standing():
    turns = [
        ([_refusal_delta()], _refused_message([_Block("text", text="Partial")])),
        _bad_request("credit redemption temporarily unavailable"),
    ]
    svc = _fable_service(turns)
    chunks = await _collect_fable(svc)

    text = "".join(c.content for c in chunks if c.type == "text")
    assert "declined this request (bio). Try rephrasing" in text
    assert len(svc.anthropic_client.messages.calls) == 2


@pytest.mark.asyncio
async def test_a_non_400_failure_on_the_retry_propagates():
    import anthropic
    import httpx

    error = anthropic.PermissionDeniedError(
        "no",
        response=httpx.Response(403, request=httpx.Request("POST", "https://api/v1/messages")),
        body={"error": {"type": "permission_error", "message": "no"}},
    )
    turns = [([_refusal_delta()], _refused_message([])), error]
    with pytest.raises(anthropic.PermissionDeniedError):
        await _collect_fable(_fable_service(turns))


@pytest.mark.asyncio
async def test_the_retry_model_refusing_too_reaches_the_user_as_a_notice():
    opus = _refused_message([], model="claude-opus-5")
    turns = [
        ([_refusal_delta()], _refused_message([])),
        ([_refusal_delta(category="cyber", token=None)], opus),
    ]
    svc = _fable_service(turns)
    chunks = await _collect_fable(svc)

    text = "".join(c.content for c in chunks if c.type == "text")
    assert "declined this request (cyber). Try rephrasing" in text
    assert len(svc.anthropic_client.messages.calls) == 2


@pytest.mark.asyncio
async def test_after_a_retry_the_rest_of_the_turn_stays_on_the_retry_model():
    live = _Block("tool_use", id="t1", name="run_analysis", input={"code": "print(1)"})
    refused = _refused_message([_Block("text", text="Start")])
    first_opus = _FakeMessage([_Block("text", text="Running"), live], "tool_use")
    first_opus.model = "claude-opus-5"
    turns = [([_refusal_delta()], refused), ([], first_opus), _opus_answer("done")]
    svc = _fable_service(turns, executor=SimpleNamespace())

    async def _execute_tool(name, tool_input, *args, **kwargs):
        return {"success": True, "output": "1"}

    svc._execute_tool = _execute_tool
    chunks = await _collect_fable(svc)

    calls = svc.anthropic_client.messages.calls
    assert [c["model"] for c in calls] == ["claude-fable-5-1", "claude-opus-5", "claude-opus-5"]
    assert "extra_body" not in calls[2]
    assert calls[2]["extra_headers"] == {"anthropic-beta": "fallback-credit-2026-07-01"}
    # the tool the retry model asked for ran, and the replay carries the declined text,
    # the notice and the retry's own blocks as one assistant turn
    assert [json.loads(c.content)["id"] for c in chunks if c.type == "tool_use"] == ["t1"]
    replayed = calls[2]["messages"][-2]["content"]
    assert [b["type"] for b in replayed] == ["text", "text", "text", "tool_use"]
    assert replayed[0]["text"] == "Start"
    assert "declined this request" in replayed[1]["text"]
    text = "".join(c.content for c in chunks if c.type == "text")
    assert "[Answered by" not in text


@pytest.mark.asyncio
async def test_an_empty_retry_model_turns_the_client_side_retry_off(monkeypatch):
    from genetics_mcp_server import llm_service

    real = llm_service.get_settings

    def _settings():
        s = real()
        s.refusal_retry_model = ""
        return s

    monkeypatch.setattr(llm_service, "get_settings", _settings)
    turns = [([_refusal_delta()], _refused_message([]))]
    svc = _fable_service(turns)
    chunks = await _collect_fable(svc)
    text = "".join(c.content for c in chunks if c.type == "text")
    assert "Try rephrasing" in text
    assert len(svc.anthropic_client.messages.calls) == 1
