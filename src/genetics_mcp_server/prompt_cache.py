"""Keep Anthropic prompt-cache entries alive between real requests.

A request with `max_tokens: 0` produces no output and is billed for its input only, which
for a prompt already in the cache is a cache read; the read restarts the entry's timer.
Two things are kept alive that way:

- a conversation, for a while after each turn, so it can be written with the 5-minute TTL
  (1.25x input per token) rather than the 1-hour one (2x) and still be read when the user
  replies a quarter of an hour later;
- the prefix every conversation shares (tool definitions and the first system block), so
  one that starts after a quiet hour does not write it again.

Everything here is best effort: a failed re-send costs the next real request a cache
write, never an answer.
"""

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# a 5-minute entry's timer runs from the START of the request that last touched it, so
# the margin has to cover scheduling jitter only
KEEPALIVE_INTERVAL_S = 270.0
# under the 1-hour TTL with room for a slow tick
PREWARM_INTERVAL_S = 3000.0
PREWARM_TICK_S = 60.0
# one re-read of the shared prefix costs about 1/80 of writing it, so keeping an unused
# prefix warm for longer than this costs more than the write it saves
PREWARM_IDLE_DROP_S = 72 * 3600.0
# a prefix differs by model, tool surface and response-length setting; a deployment
# serves a handful, and each one kept costs a read every PREWARM_INTERVAL_S
MAX_PREFIXES = 4
# each holds one request body (up to a few MB) until its window closes
MAX_KEEPALIVE_SESSIONS = 200

_PREFIX_FIELDS = ("model", "thinking", "tools", "extra_body", "extra_headers")


def cache_usage(usage: Any) -> tuple[int, int, int, int]:
    """(uncached input, cache read, cache written, of which written with the 1-hour TTL).

    A response with no per-TTL breakdown is priced as all 1-hour, the dearer rate.
    """
    read = getattr(usage, "cache_read_input_tokens", 0) or 0
    written = getattr(usage, "cache_creation_input_tokens", 0) or 0
    one_hour = getattr(getattr(usage, "cache_creation", None), "ephemeral_1h_input_tokens", None)
    if not isinstance(one_hour, int):
        one_hour = written
    return usage.input_tokens, read, written, one_hour


@dataclass
class _Prefix:
    params: dict[str, Any]
    touched_at: float
    used_at: float
    writes_in_a_row: int = 0


class PromptCacheKeeper:
    """Schedules the zero-output re-sends. One per process.

    `send` makes the request and returns the response message; `on_spend` is told about
    every response so the caller can price and record it.
    """

    def __init__(
        self,
        send: Callable[[dict[str, Any]], Awaitable[Any]],
        on_spend: Callable[..., Awaitable[None]],
        *,
        keepalive_s: float,
        prewarm: bool,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        self._send = send
        self._on_spend = on_spend
        self._keepalive_s = keepalive_s
        self._prewarm = prewarm
        self._clock = clock
        self._sleep = sleep
        self._sessions: dict[str, asyncio.Task] = {}
        self._prefixes: dict[str, _Prefix] = {}
        self._prewarm_task: asyncio.Task | None = None

    # -- a conversation, between turns ------------------------------------------------

    def begin_turn(self, session_id: str | None) -> None:
        """Stop re-sending for a conversation whose next turn has arrived."""
        task = self._sessions.pop(session_id or "", None)
        if task:
            task.cancel()

    def end_turn(
        self,
        session_id: str | None,
        request_params: dict[str, Any],
        sent_at: float,
        *,
        user: str | None,
        secret: bool,
    ) -> None:
        """Keep the turn's last request readable for the keep-alive window.

        `sent_at` is when that request was sent, on this keeper's clock: the entry it
        wrote lapses five minutes from then, not from when the turn finished.
        """
        if self._keepalive_s <= 0 or not session_id:
            return
        self.begin_turn(session_id)
        if len(self._sessions) >= MAX_KEEPALIVE_SESSIONS:
            logger.warning("Cache keep-alive skipped: too many conversations being kept")
            return
        task = asyncio.get_running_loop().create_task(
            self._keep_alive(session_id, request_params, sent_at, user, secret)
        )
        self._sessions[session_id] = task
        task.add_done_callback(
            lambda t: (
                self._sessions.pop(session_id, None)
                if self._sessions.get(session_id) is t
                else None
            )
        )

    async def _keep_alive(
        self,
        session_id: str,
        request_params: dict[str, Any],
        sent_at: float,
        user: str | None,
        secret: bool,
    ) -> None:
        ping = {**request_params, "max_tokens": 0}
        until = self._clock() + self._keepalive_s
        next_at = sent_at + KEEPALIVE_INTERVAL_S
        sent: list[Any] = []
        try:
            while next_at < until:
                await self._sleep(max(0.0, next_at - self._clock()))
                next_at = self._clock() + KEEPALIVE_INTERVAL_S
                message = await self._send(ping)
                sent.append(message)
                _, read, written, _ = cache_usage(message.usage)
                # the entry had already lapsed, or this body is not the one the turn
                # sent: either way the next re-send would pay for a write again
                if written > read:
                    logger.warning(
                        f"[session={session_id}] Cache keep-alive stopped: re-send wrote "
                        f"{written} tokens and read {read}"
                    )
                    break
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"[session={session_id}] Cache keep-alive stopped: {e}")
        finally:
            if sent:
                # shielded: the next turn cancels this task, and what was sent is spent
                await asyncio.shield(
                    self._on_spend(
                        "keep-alive",
                        sent,
                        model=request_params.get("model"),
                        session_id=session_id,
                        user=user,
                        secret=secret,
                    )
                )

    # -- the prefix every conversation shares ------------------------------------------

    def note_prefix(self, request_params: dict[str, Any]) -> None:
        """Register the shared prefix of a request that is about to be sent.

        The registry lives in this process, so after a restart a prefix is re-read again
        only once a request has carried it.
        """
        if not self._prewarm:
            return
        system = request_params.get("system") or []
        if not system or "cache_control" not in system[0]:
            return
        params = {k: request_params[k] for k in _PREFIX_FIELDS if k in request_params}
        params["system"] = system[:1]
        key = hashlib.sha256(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()
        now = self._clock()
        known = self._prefixes.get(key)
        if known:
            known.used_at = now
        else:
            if len(self._prefixes) >= MAX_PREFIXES:
                oldest = min(self._prefixes, key=lambda k: self._prefixes[k].used_at)
                del self._prefixes[oldest]
            self._prefixes[key] = _Prefix(params, touched_at=now, used_at=now)
        if self._prewarm_task is None or self._prewarm_task.done():
            self._prewarm_task = asyncio.get_running_loop().create_task(self._prewarm_loop())

    async def _prewarm_loop(self) -> None:
        while True:
            await self._sleep(PREWARM_TICK_S)
            for key, prefix in list(self._prefixes.items()):
                now = self._clock()
                if now - prefix.used_at > PREWARM_IDLE_DROP_S:
                    del self._prefixes[key]
                elif now - prefix.touched_at >= PREWARM_INTERVAL_S:
                    await self._rewarm(key, prefix)

    async def _rewarm(self, key: str, prefix: _Prefix) -> None:
        # the breakpoint that matters is the one on the system block; the placeholder
        # turn after it is read and never answered
        ping = {
            **prefix.params,
            "max_tokens": 0,
            "messages": [{"role": "user", "content": "warmup"}],
        }
        prefix.touched_at = self._clock()
        try:
            message = await self._send(ping)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"Cache pre-warm failed: {e}")
            return
        _, read, written, _ = cache_usage(message.usage)
        # an entry touched under an hour ago is still there, so a write means this body
        # is not what real requests send, and re-sending it buys nothing
        prefix.writes_in_a_row = prefix.writes_in_a_row + 1 if written > read else 0
        if prefix.writes_in_a_row >= 2:
            logger.error(
                f"Cache pre-warm dropped a prefix: re-sends keep writing it "
                f"({written} tokens written, {read} read)"
            )
            self._prefixes.pop(key, None)
        await self._on_spend(
            "pre-warm",
            [message],
            model=prefix.params.get("model"),
            session_id=None,
            user=None,
            secret=False,
        )

    async def close(self) -> None:
        tasks = [*self._sessions.values()]
        if self._prewarm_task:
            tasks.append(self._prewarm_task)
        self._sessions.clear()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
