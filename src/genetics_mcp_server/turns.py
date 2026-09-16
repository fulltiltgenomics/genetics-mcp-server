"""Server-owned chat turns.

A turn used to be the lifetime of one HTTP response: the browser opened `POST /chat/v1/chat`,
the model ran inside that response's generator, and the browser was the only thing that
ever wrote the answer to chat history. A laptop going to sleep mid-answer cancelled the
generator, and the tokens already paid for were gone with nothing persisted — not even the
user's own message.

Here the run is a task the registry owns, and every HTTP response is a *subscriber* that
replays the turn's buffered events from a sequence number and then tails it. A subscriber
being cancelled (client gone) leaves the task running; the task's finish persists the
assistant message itself, from the same events the browser renders, so the transcript on
reload is what the screen would have shown.

The buffer is in-process. chat-backend runs one replica with `strategy: Recreate` for
reasons stated on its Deployment, so there is no second pod a reconnect could land on;
a pod restart loses the buffer, and the shutdown drain exists so it does not lose the
persisted row too.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

logger = logging.getLogger(__name__)

# how long a settled turn stays replayable. A browser that woke up and reloaded its session
# in this window gets the buffered tail; after it, the persisted row is the only copy
RETAIN_SECONDS = 600


class TurnAlreadyRunning(Exception):
    """A second start for a turn id that is still in the registry."""


@dataclass
class TurnEvent:
    seq: int
    event: str
    payload: dict[str, Any]
    _data: str | None = field(default=None, repr=False)

    @property
    def data(self) -> str:
        """The payload serialised once, however many subscribers send it."""
        if self._data is None:
            self._data = json.dumps(self.payload)
        return self._data


class Turn:
    """One turn's event buffer and lifecycle flags.

    `closed` means no more events will be appended and subscribers reach end-of-stream.
    `settled` means the finish hook (persistence) has also run; a session reports a turn
    as active until then, so a client that loads the session between the last event and
    the row landing still attaches here rather than concluding the turn produced nothing.
    """

    def __init__(
        self, turn_id: str, *, session_id: str | None, user: str | None, secret: bool
    ) -> None:
        self.id = turn_id
        self.session_id = session_id
        self.user = user
        self.secret = secret
        self.events: list[TurnEvent] = []
        self.closed = False
        self.settled = False
        self.saw_done = False
        self.outcome: str | None = None
        self.started_at = time.monotonic()
        self.task: asyncio.Task | None = None
        self._changed = asyncio.Event()

    def append(self, event: str, payload: dict[str, Any]) -> None:
        if self.closed:
            raise RuntimeError("append on a closed turn")
        if event == "message" and payload.get("type") == "done":
            self.saw_done = True
        self.events.append(TurnEvent(len(self.events), event, payload))
        self._notify()

    def close(self) -> None:
        self.closed = True
        self._notify()

    def _notify(self) -> None:
        # each waiter holds the Event that was current when it went to sleep; swapping in a
        # fresh one before setting the old means a waiter can never miss a notification
        # that arrived between its emptiness check and its wait
        changed, self._changed = self._changed, asyncio.Event()
        changed.set()

    @property
    def next_seq(self) -> int:
        return len(self.events)

    def cancel(self) -> bool:
        """Stop the run. False once it has already stopped, including while its finish
        hook is still writing: cancelling that would lose the row the run produced."""
        if self.task is None or self.task.done() or self.closed:
            return False
        return self.task.cancel()

    async def subscribe(self, from_seq: int = 0) -> AsyncIterator[TurnEvent]:
        """Yield events with seq >= `from_seq`, then tail until the turn closes."""
        i = max(from_seq, 0)
        while True:
            while i < len(self.events):
                yield self.events[i]
                i += 1
            if self.closed:
                return
            # no await between the checks above and taking this reference (see _notify)
            await self._changed.wait()


EventSource = AsyncIterator[tuple[str, dict[str, Any]]]
FinishHook = Callable[["Turn"], Awaitable[None]]


class TurnRegistry:
    def __init__(self, *, retain_seconds: float = RETAIN_SECONDS) -> None:
        self._turns: dict[str, Turn] = {}
        self._retain_seconds = retain_seconds
        self._eviction_handles: dict[str, asyncio.TimerHandle] = {}

    def get(self, turn_id: str) -> Turn | None:
        return self._turns.get(turn_id)

    def active_for_session(self, session_id: str) -> Turn | None:
        """The newest unsettled turn of a session, or None."""
        candidates = [
            t for t in self._turns.values() if t.session_id == session_id and not t.settled
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda t: t.started_at)

    def running(self) -> list[Turn]:
        return [t for t in self._turns.values() if not t.settled]

    def start(self, turn: Turn, source: EventSource, on_finish: FinishHook | None) -> Turn:
        existing = self._turns.get(turn.id)
        if existing is not None:
            raise TurnAlreadyRunning(turn.id)
        self._turns[turn.id] = turn
        turn.task = asyncio.create_task(
            self._run(turn, source, on_finish), name=f"chat-turn-{turn.id}"
        )
        return turn

    async def _run(self, turn: Turn, source: EventSource, on_finish: FinishHook | None) -> None:
        try:
            async for event, payload in source:
                turn.append(event, payload)
            turn.outcome = "complete" if turn.saw_done else "error"
        except asyncio.CancelledError:
            turn.outcome = "cancelled"
            # the cancellation has done its work — the source generator is closed and the
            # model call inside it was cancelled — so this task finishes normally and
            # persists what it has, rather than propagating and skipping the finish hook
            if turn.task is not None:
                turn.task.uncancel()
            # a terminator, so a subscriber can tell a stopped turn from a lost connection:
            # without it, end-of-stream with no `done` looks exactly like the network going
            # away, and the client would keep reattaching to a turn that is over
            turn.append("message", {"type": "cancelled"})
        except Exception as e:
            # chat_api's generator classifies and yields its own errors; this is for the
            # ones that escape it, so a subscriber still sees a terminator
            logger.error(f"[turn={turn.id}] unhandled error in turn source: {e}", exc_info=True)
            turn.outcome = "error"
            turn.append("error", {"type": "error", "error": "A server error occurred"})
        finally:
            turn.close()
            try:
                if on_finish is not None:
                    await on_finish(turn)
            except Exception as e:
                logger.error(f"[turn={turn.id}] turn finish hook failed: {e}", exc_info=True)
            finally:
                turn.settled = True
                self._schedule_eviction(turn.id)

    def _schedule_eviction(self, turn_id: str) -> None:
        loop = asyncio.get_running_loop()
        self._eviction_handles[turn_id] = loop.call_later(
            self._retain_seconds, self._evict, turn_id
        )

    def _evict(self, turn_id: str) -> None:
        self._turns.pop(turn_id, None)
        self._eviction_handles.pop(turn_id, None)

    async def drain(self, timeout: float) -> int:
        """Wait for unsettled turns to finish; the number still running at the deadline."""
        tasks = [t.task for t in self.running() if t.task is not None]
        if not tasks:
            return 0
        logger.info(f"draining {len(tasks)} chat turn(s), up to {timeout:.0f}s")
        _done, pending = await asyncio.wait(tasks, timeout=timeout)
        if pending:
            logger.warning(f"{len(pending)} chat turn(s) still running at shutdown; cancelling")
            for task in pending:
                task.cancel()
        return len(pending)


_registry: TurnRegistry | None = None


def get_turn_registry() -> TurnRegistry:
    global _registry
    if _registry is None:
        _registry = TurnRegistry()
    return _registry


# --- transcript rendering -----------------------------------------------------------------
#
# The browser writes a turn into one `content` string: prose, plus a marker per image, file
# and tool call, which is the only form anything displaying a stored message ever reads.
# The server has to produce the same string from the same events, or a turn persisted here
# would render differently from one the browser saved. The marker grammars below are the
# ones in genetics-results-browser's imageMarker.ts, fileMarker.ts and toolCallMarker.ts;
# the two ends cannot import one module, so the shape is stated in both.

_TOOL_CALL_MARKER_RE = re.compile(r"\[TOOLUSE:([A-Za-z0-9+/=]*)\]")


def _encode_tool_call(record: dict[str, Any]) -> str:
    encoded = base64.b64encode(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode()
    ).decode()
    return f"[TOOLUSE:{encoded}]"


def _decode_tool_call(encoded: str) -> dict[str, Any] | None:
    try:
        record = json.loads(base64.b64decode(encoded))
    except Exception:
        return None
    if isinstance(record, dict) and isinstance(record.get("name"), str):
        return record
    return None


def _with_tool_call_outcome(content: str, tool_use_id: str, outcome: dict[str, Any]) -> str:
    def rewrite(match: re.Match) -> str:
        record = _decode_tool_call(match.group(1))
        if not record or record.get("id") != tool_use_id:
            return match.group(0)
        return _encode_tool_call({**record, "outcome": outcome})

    return _TOOL_CALL_MARKER_RE.sub(rewrite, content)


def render_transcript(events: list[TurnEvent]) -> str:
    """The `content` string the browser would have accumulated from these events."""
    content = ""
    for ev in events:
        if ev.event != "message":
            continue
        p = ev.payload
        kind = p.get("type")
        if kind == "content" and p.get("content"):
            content += p["content"]
        elif kind == "image":
            image_format = re.sub(r"[^\w+.-]", "", p.get("image_format") or "png", flags=re.ASCII)
            image_alt = re.sub(r"[:\[\]]", " ", p.get("image_alt") or "Generated image")
            content += f"\n\n[IMAGE:{image_format}:{image_alt}:{p.get('image_data') or ''}]\n\n"
        elif kind == "file" and p.get("file_data"):
            mime = re.sub(r"[^\w+./-]", "", p.get("file_mime") or "", flags=re.ASCII)
            mime = mime or "application/octet-stream"
            data = re.sub(r"[^A-Za-z0-9+/=]", "", p["file_data"])
            # encodeURIComponent's unreserved set
            name = quote(p.get("file_name") or "artifact", safe="-_.!~*'()")
            content += f"\n\n[FILE:{mime}:{name}:{data}]\n\n"
        elif kind == "tool_use" and p.get("name"):
            record = {"id": p.get("id") or "", "name": p["name"], "input": p.get("input") or {}}
            content += f"\n\n{_encode_tool_call(record)}\n\n"
        elif kind == "script_result" and p.get("tool_use_id"):
            duration = p.get("duration_ms")
            outcome = {
                "ran": bool(p.get("ran")),
                "ok": bool(p.get("ok")),
                "status": p["status"] if isinstance(p.get("status"), str) else "unknown",
                "durationMs": duration if isinstance(duration, (int, float)) else None,
                "exception": p["exception"] if isinstance(p.get("exception"), str) else None,
            }
            content = _with_tool_call_outcome(content, p["tool_use_id"], outcome)
    return content


def done_payload(events: list[TurnEvent]) -> dict[str, Any] | None:
    for ev in reversed(events):
        if ev.event == "message" and ev.payload.get("type") == "done":
            return ev.payload
    return None
