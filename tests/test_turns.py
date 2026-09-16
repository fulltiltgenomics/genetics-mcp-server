"""The turn registry: a run that outlives the response that started it."""

import asyncio
import base64
import json
import re
import tempfile
import threading
import time
from unittest.mock import patch

import pytest
from conftest import close_and_unlink
from fastapi.testclient import TestClient

from genetics_mcp_server import llm_service as llm_service_module
from genetics_mcp_server import turns
from genetics_mcp_server.auth import auth_required
from genetics_mcp_server.chat_api import app
from genetics_mcp_server.db.chat_history_db import ChatHistoryDB
from genetics_mcp_server.db.singleton import Singleton
from genetics_mcp_server.llm_service import StreamChunk
from genetics_mcp_server.turns import (
    Turn,
    TurnAlreadyRunning,
    TurnRegistry,
    render_transcript,
)


async def _source(payloads, *, gate=None, closed=None):
    """An event source that can be held at `gate` and reports when it is closed."""
    try:
        for i, p in enumerate(payloads):
            if gate is not None and i == len(payloads) // 2:
                await gate.wait()
            yield "message", p
    finally:
        if closed is not None:
            closed.set()


def _content(text):
    return {"type": "content", "content": text}


DONE = {"type": "done", "message_content": [{"type": "text", "text": "hi"}], "tool_results": None}


class TestRegistry:
    @pytest.mark.asyncio
    async def test_a_cancelled_subscriber_does_not_stop_the_turn(self):
        registry = TurnRegistry(retain_seconds=60)
        gate = asyncio.Event()
        finished = []

        async def on_finish(turn):
            finished.append(turn)

        turn = Turn("t1", session_id="s1", user="u", secret=False)
        registry.start(turn, _source([_content("a"), _content("b"), DONE], gate=gate), on_finish)

        async def subscriber():
            async for _ in turn.subscribe(0):
                pass

        sub = asyncio.create_task(subscriber())
        await asyncio.sleep(0)
        sub.cancel()
        with pytest.raises(asyncio.CancelledError):
            await sub

        gate.set()
        await turn.task
        assert [e.payload for e in turn.events] == [_content("a"), _content("b"), DONE]
        assert turn.outcome == "complete"
        assert finished == [turn]
        assert turn.settled

    @pytest.mark.asyncio
    async def test_a_late_subscriber_replays_from_its_sequence_number(self):
        registry = TurnRegistry(retain_seconds=60)
        gate = asyncio.Event()
        turn = Turn("t2", session_id="s1", user="u", secret=False)
        registry.start(turn, _source([_content("a"), _content("b"), _content("c"), DONE], gate=gate), None)
        await asyncio.sleep(0)
        assert turn.next_seq == 2

        async def collect(from_seq):
            return [e.seq for e in [ev async for ev in turn.subscribe(from_seq)]]

        tail = asyncio.create_task(collect(from_seq=2))
        whole = asyncio.create_task(collect(from_seq=0))
        await asyncio.sleep(0)
        gate.set()
        assert await tail == [2, 3]
        assert await whole == [0, 1, 2, 3]

    @pytest.mark.asyncio
    async def test_cancel_closes_the_source_and_still_runs_the_finish_hook(self):
        registry = TurnRegistry(retain_seconds=60)
        gate = asyncio.Event()
        closed = asyncio.Event()
        finished = []

        async def on_finish(turn):
            finished.append(turn.outcome)

        turn = Turn("t3", session_id="s1", user="u", secret=False)
        registry.start(turn, _source([_content("a"), _content("b")], gate=gate, closed=closed), on_finish)
        await asyncio.sleep(0)
        assert turn.cancel()
        await turn.task
        assert closed.is_set()
        assert turn.outcome == "cancelled"
        assert finished == ["cancelled"]
        assert [e.payload for e in turn.events] == [_content("a"), {"type": "cancelled"}]
        assert not turn.cancel()

    @pytest.mark.asyncio
    async def test_a_turn_stays_active_until_persisted_then_is_evicted(self):
        registry = TurnRegistry(retain_seconds=0.05)
        release = asyncio.Event()
        hook_entered = asyncio.Event()

        async def slow_persist(turn):
            hook_entered.set()
            await release.wait()

        turn = Turn("t4", session_id="s1", user="u", secret=False)
        registry.start(turn, _source([_content("a"), DONE]), slow_persist)
        await hook_entered.wait()
        # streaming is over but the row has not landed: the session must still point here
        assert turn.closed and not turn.settled
        assert registry.active_for_session("s1") is turn
        release.set()
        await turn.task
        assert registry.active_for_session("s1") is None
        assert registry.get("t4") is turn
        await asyncio.sleep(0.1)
        assert registry.get("t4") is None

    @pytest.mark.asyncio
    async def test_the_same_turn_id_cannot_start_twice(self):
        registry = TurnRegistry(retain_seconds=60)
        turn = Turn("t5", session_id="s1", user="u", secret=False)
        registry.start(turn, _source([DONE]), None)
        with pytest.raises(TurnAlreadyRunning):
            registry.start(Turn("t5", session_id="s1", user="u", secret=False), _source([DONE]), None)
        await turn.task

    @pytest.mark.asyncio
    async def test_a_failing_source_leaves_a_terminator(self):
        async def broken():
            yield "message", _content("a")
            raise RuntimeError("boom")

        registry = TurnRegistry(retain_seconds=60)
        turn = Turn("t6", session_id="s1", user="u", secret=False)
        registry.start(turn, broken(), None)
        await turn.task
        assert turn.outcome == "error"
        assert turn.events[-1].event == "error"

    @pytest.mark.asyncio
    async def test_drain_waits_for_running_turns(self):
        registry = TurnRegistry(retain_seconds=60)
        gate = asyncio.Event()
        turn = Turn("t7", session_id="s1", user="u", secret=False)
        registry.start(turn, _source([_content("a"), DONE], gate=gate), None)
        await asyncio.sleep(0)
        asyncio.get_running_loop().call_later(0.05, gate.set)
        started = time.monotonic()
        assert await registry.drain(timeout=5) == 0
        assert time.monotonic() - started >= 0.04
        assert turn.settled

    @pytest.mark.asyncio
    async def test_drain_cancels_what_outlives_the_deadline(self):
        registry = TurnRegistry(retain_seconds=60)
        gate = asyncio.Event()
        turn = Turn("t8", session_id="s1", user="u", secret=False)
        registry.start(turn, _source([_content("a"), DONE], gate=gate), None)
        await asyncio.sleep(0)
        assert await registry.drain(timeout=0.05) == 1
        await turn.task
        assert turn.outcome == "cancelled"


# the marker grammars from the browser's imageMarker.ts, fileMarker.ts and toolCallMarker.ts
IMAGE_RE = re.compile(r"\[IMAGE:([^:\]]+):([^:\]]+):([^\]]+)\]")
FILE_RE = re.compile(r"\[FILE:([^:\]]+):([^:\]]+):([^\]]+)\]")
TOOL_RE = re.compile(r"\[TOOLUSE:([A-Za-z0-9+/=]*)\]")


def _ev(seq, payload, event="message"):
    return turns.TurnEvent(seq, event, payload)


class TestRenderTranscript:
    def test_prose_is_concatenated_and_other_events_ignored(self):
        events = [
            _ev(0, {"type": "memory", "sessions": 2}),
            _ev(1, {"type": "thinking"}),
            _ev(2, _content("Hello ")),
            _ev(3, {"type": "usage", "input_tokens": 5}),
            _ev(4, _content("world")),
            _ev(5, DONE),
            _ev(6, {"type": "error", "error": "x"}, event="error"),
            _ev(7, {"type": "cancelled"}),
        ]
        assert render_transcript(events) == "Hello world"

    def test_an_image_becomes_a_marker_the_browser_can_parse(self):
        events = [_ev(0, {"type": "image", "image_data": "AAAA", "image_format": "png", "image_alt": "a:b[c]"})]
        m = IMAGE_RE.search(render_transcript(events))
        assert m and m.groups() == ("png", "a b c ", "AAAA")

    def test_a_file_marker_survives_a_name_with_delimiters(self):
        events = [_ev(0, {"type": "file", "file_data": "QUJD", "file_mime": "text/csv", "file_name": "a:b].csv"})]
        m = FILE_RE.search(render_transcript(events))
        assert m and m.group(1) == "text/csv" and m.group(3) == "QUJD"
        assert m.group(2) == "a%3Ab%5D.csv"

    def test_an_empty_file_payload_is_not_written(self):
        events = [_ev(0, {"type": "file", "file_data": "", "file_mime": "text/csv", "file_name": "x"})]
        assert render_transcript(events) == ""

    def test_a_tool_call_and_its_outcome_round_trip(self):
        events = [
            _ev(0, {"type": "tool_use", "id": "tu_1", "name": "run_analysis", "input": {"code": "print(1)\n"}}),
            _ev(1, {"type": "tool_use", "id": "tu_2", "name": "search", "input": {"q": "ä"}}),
            _ev(2, {"type": "script_result", "tool_use_id": "tu_1", "ran": True, "ok": False,
                    "status": "error", "duration_ms": 12, "exception": "ValueError"}),
        ]
        markers = TOOL_RE.findall(render_transcript(events))
        assert len(markers) == 2
        first, second = (json.loads(base64.b64decode(m)) for m in markers)
        assert first == {
            "id": "tu_1", "name": "run_analysis", "input": {"code": "print(1)\n"},
            "outcome": {"ran": True, "ok": False, "status": "error", "durationMs": 12, "exception": "ValueError"},
        }
        assert second == {"id": "tu_2", "name": "search", "input": {"q": "ä"}}


@pytest.fixture
def history_db():
    if ChatHistoryDB in Singleton._instances:
        del Singleton._instances[ChatHistoryDB]
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    db = ChatHistoryDB(db_path)
    yield db
    if ChatHistoryDB in Singleton._instances:
        del Singleton._instances[ChatHistoryDB]
    close_and_unlink(db, db_path)


@pytest.fixture
def client(history_db):
    async def mock_auth():
        return "test@example.com"

    app.dependency_overrides[auth_required] = mock_auth
    turns._registry = TurnRegistry(retain_seconds=60)
    with (
        patch("genetics_mcp_server.routers.chat_history.get_chat_history_db", return_value=history_db),
        patch("genetics_mcp_server.chat_api.get_chat_history_db", return_value=history_db),
        patch.object(llm_service_module.get_llm_service(), "anthropic_client", True),
        TestClient(app) as c,
    ):
        yield c
    app.dependency_overrides.clear()
    turns._registry = None


def _stream(chunks, *, hold_before=None, release=None):
    """A stream_chat stand-in. With `hold_before`, it waits before yielding that chunk until
    the test sets `release` — a threading.Event, because the test drives it from outside
    the TestClient's event loop."""
    async def stream_chat(**kwargs):
        for i, chunk in enumerate(chunks):
            if hold_before is not None and i == hold_before:
                while not release.is_set():
                    await asyncio.sleep(0.01)
            yield chunk
    return stream_chat


def _post_in_background(client, body):
    result = {}

    def run():
        result["response"] = client.post("/chat/v1/chat", json=body)

    thread = threading.Thread(target=run)
    thread.start()
    return thread, result


def _wait_until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def _events(response):
    out = []
    current = {}
    for line in response.text.splitlines():
        if not line:
            if current:
                out.append(current)
                current = {}
            continue
        key, _, value = line.partition(":")
        current[key] = value.strip()
    if current:
        out.append(current)
    return out


def _wait_for_messages(db, session_id, n, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        msgs = db.get_messages(session_id)
        if len(msgs) >= n:
            return msgs
        time.sleep(0.02)
    return db.get_messages(session_id)


class TestChatTurnEndpoints:
    def test_the_answer_is_persisted_by_the_server_with_the_client_id(self, client, history_db):
        session = history_db.create_session("test@example.com")
        chunks = [
            StreamChunk(type="text", content="Hel"),
            StreamChunk(type="text", content="lo"),
            StreamChunk(type="image", content="AAAA", image_format="png", image_alt="plot"),
            StreamChunk(type="done", message_content=[{"type": "text", "text": "Hello"}],
                        tool_results=[{"type": "tool_result", "tool_use_id": "x", "content": "y"}]),
        ]
        with patch.object(llm_service_module.get_llm_service(), "stream_chat", _stream(chunks)):
            r = client.post("/chat/v1/chat", json={
                "messages": [{"role": "user", "content": "hi"}],
                "session_id": session.id, "message_id": "msg-1", "verbosity": "brief",
                "tool_profile": "code", "instruction_set_id": "ins-1",
            })
        assert r.status_code == 200
        events = _events(r)
        assert [e["id"] for e in events] == ["0", "1", "2", "3"]

        msgs = _wait_for_messages(history_db, session.id, 1)
        assert [m.id for m in msgs] == ["msg-1"]
        assert msgs[0].role == "assistant"
        assert msgs[0].content.startswith("Hello\n\n[IMAGE:png:plot:AAAA]")
        assert json.loads(msgs[0].content_json) == [{"type": "text", "text": "Hello"}]
        assert json.loads(msgs[0].tool_results_json)[0]["tool_use_id"] == "x"
        assert (msgs[0].verbosity, msgs[0].tool_profile, msgs[0].instruction_set_id) == ("brief", "code", "ins-1")

    def test_a_secret_turn_writes_nothing(self, client, history_db):
        session = history_db.create_session("test@example.com")
        chunks = [StreamChunk(type="text", content="x"), StreamChunk(type="done", message_content=[])]
        with patch.object(llm_service_module.get_llm_service(), "stream_chat", _stream(chunks)):
            r = client.post("/chat/v1/chat", json={
                "messages": [{"role": "user", "content": "hi"}],
                "session_id": session.id, "message_id": "msg-s", "secret": True,
            })
        assert r.status_code == 200
        time.sleep(0.1)
        assert history_db.get_messages(session.id) == []

    def test_a_turn_with_no_content_writes_nothing(self, client, history_db):
        session = history_db.create_session("test@example.com")
        chunks = [StreamChunk(type="done", message_content=[])]
        with patch.object(llm_service_module.get_llm_service(), "stream_chat", _stream(chunks)):
            client.post("/chat/v1/chat", json={
                "messages": [{"role": "user", "content": "hi"}],
                "session_id": session.id, "message_id": "msg-e",
            })
        time.sleep(0.1)
        assert history_db.get_messages(session.id) == []

    def test_reattaching_replays_the_buffer_and_the_session_names_the_turn(self, client, history_db):
        session = history_db.create_session("test@example.com")
        chunks = [
            StreamChunk(type="text", content="a"),
            StreamChunk(type="text", content="b"),
            StreamChunk(type="done", message_content=[{"type": "text", "text": "ab"}]),
        ]
        with patch.object(llm_service_module.get_llm_service(), "stream_chat", _stream(chunks)):
            r = client.post("/chat/v1/chat", json={
                "messages": [{"role": "user", "content": "hi"}],
                "session_id": session.id, "message_id": "msg-r",
            })
            assert len(_events(r)) == 3
            tail = client.get("/chat/v1/chat/turns/msg-r/events", params={"from_seq": 1})
        assert tail.status_code == 200
        assert [json.loads(e["data"])["type"] for e in _events(tail)] == ["content", "done"]
        _wait_for_messages(history_db, session.id, 1)
        detail = client.get(f"/chat/v1/chat/sessions/{session.id}").json()
        assert detail["active_turn"] is None
        assert [m["id"] for m in detail["messages"]] == ["msg-r"]

    def test_a_running_turn_is_reported_on_its_session(self, client, history_db):
        session = history_db.create_session("test@example.com")
        release = threading.Event()
        chunks = [
            StreamChunk(type="text", content="a"),
            StreamChunk(type="done", message_content=[{"type": "text", "text": "a"}]),
        ]
        registry = turns.get_turn_registry()
        with patch.object(llm_service_module.get_llm_service(), "stream_chat",
                          _stream(chunks, hold_before=1, release=release)):
            thread, result = _post_in_background(client, {
                "messages": [{"role": "user", "content": "hi"}],
                "session_id": session.id, "message_id": "msg-live",
            })
            assert _wait_until(lambda: (t := registry.get("msg-live")) is not None and t.next_seq == 1)
            detail = client.get(f"/chat/v1/chat/sessions/{session.id}").json()
            assert detail["active_turn"] == {"message_id": "msg-live"}
            assert detail["messages"] == []
            release.set()
            thread.join(timeout=5)
        assert result["response"].status_code == 200
        assert _wait_until(lambda: registry.get("msg-live").settled)
        detail = client.get(f"/chat/v1/chat/sessions/{session.id}").json()
        assert detail["active_turn"] is None
        assert [m["id"] for m in detail["messages"]] == ["msg-live"]

    def test_cancel_stops_the_run_and_keeps_the_partial(self, client, history_db):
        session = history_db.create_session("test@example.com")
        release = threading.Event()
        chunks = [
            StreamChunk(type="text", content="partial"),
            StreamChunk(type="text", content=" never"),
            StreamChunk(type="done", message_content=[{"type": "text", "text": "partial never"}]),
        ]
        registry = turns.get_turn_registry()
        with patch.object(llm_service_module.get_llm_service(), "stream_chat",
                          _stream(chunks, hold_before=1, release=release)):
            thread, result = _post_in_background(client, {
                "messages": [{"role": "user", "content": "hi"}],
                "session_id": session.id, "message_id": "msg-c",
            })
            assert _wait_until(lambda: (t := registry.get("msg-c")) is not None and t.next_seq == 1)
            assert client.post("/chat/v1/chat/turns/msg-c/cancel").json() == {"cancelled": True}
            thread.join(timeout=5)
        assert result["response"].status_code == 200
        assert [json.loads(e["data"])["type"] for e in _events(result["response"])] == ["content", "cancelled"]
        msgs = _wait_for_messages(history_db, session.id, 1)
        assert [m.content for m in msgs] == ["partial"]
        assert msgs[0].content_json is None
        assert registry.get("msg-c").outcome == "cancelled"
        # a second cancel, and one during the write, are no-ops rather than lost rows
        assert client.post("/chat/v1/chat/turns/msg-c/cancel").json() == {"cancelled": False}

    def test_another_user_sees_no_turn(self, client, history_db):
        session = history_db.create_session("test@example.com")
        chunks = [StreamChunk(type="text", content="a"), StreamChunk(type="done", message_content=[])]
        with patch.object(llm_service_module.get_llm_service(), "stream_chat", _stream(chunks)):
            client.post("/chat/v1/chat", json={
                "messages": [{"role": "user", "content": "hi"}],
                "session_id": session.id, "message_id": "msg-o",
            })
        turn = turns.get_turn_registry().get("msg-o")
        turn.user = "someone-else@example.com"
        assert client.get("/chat/v1/chat/turns/msg-o/events").status_code == 404
        assert client.post("/chat/v1/chat/turns/msg-o/cancel").status_code == 404
        assert client.get("/chat/v1/chat/turns/nope/events").status_code == 404

    def test_a_duplicate_message_id_is_refused(self, client, history_db):
        chunks = [StreamChunk(type="text", content="a"), StreamChunk(type="done", message_content=[])]
        body = {"messages": [{"role": "user", "content": "hi"}], "message_id": "msg-dup"}
        with patch.object(llm_service_module.get_llm_service(), "stream_chat", _stream(chunks)):
            assert client.post("/chat/v1/chat", json=body).status_code == 200
            assert client.post("/chat/v1/chat", json=body).status_code == 409

    def test_shutdown_drains_a_turn_nobody_is_watching(self, history_db):
        """The lifespan waits for a detached turn, so its row lands before the app stops."""
        async def mock_auth():
            return "test@example.com"

        app.dependency_overrides[auth_required] = mock_auth
        registry = turns._registry = TurnRegistry(retain_seconds=60)
        session = history_db.create_session("test@example.com")
        release = threading.Event()
        chunks = [
            StreamChunk(type="text", content="slow"),
            StreamChunk(type="done", message_content=[{"type": "text", "text": "slow"}]),
        ]
        try:
            with (
                patch("genetics_mcp_server.chat_api.get_chat_history_db", return_value=history_db),
                patch.object(llm_service_module.get_llm_service(), "anthropic_client", True),
                patch.object(llm_service_module.get_llm_service(), "stream_chat",
                             _stream(chunks, hold_before=1, release=release)),
            ):
                with TestClient(app) as c:
                    thread, result = _post_in_background(c, {
                        "messages": [{"role": "user", "content": "hi"}],
                        "session_id": session.id, "message_id": "msg-drain",
                    })
                    assert _wait_until(lambda: (t := registry.get("msg-drain")) is not None and t.next_seq == 1)
                    # the pod is told to stop while the turn is mid-answer; the answer arrives
                    # a little later and shutdown has to still be waiting for it
                    threading.Timer(0.3, release.set).start()
                thread.join(timeout=5)
            assert registry.get("msg-drain").settled
            assert history_db.get_messages(session.id)[0].content == "slow"
        finally:
            app.dependency_overrides.clear()
            turns._registry = None
