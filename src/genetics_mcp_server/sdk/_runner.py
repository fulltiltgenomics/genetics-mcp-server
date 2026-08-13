"""Runs the SDK's async client from synchronous script code.

The executor is async all the way down (httpx.AsyncClient, concurrent fan-out), but the
sandbox runs plain scripts: `df = genetics.credible_sets(gene="IL7R")`. A dedicated event
loop on a background thread bridges the two while keeping the HTTP connection pool and the
executor's per-process caches alive across calls — `asyncio.run()` per call would discard
both and re-handshake TLS every time.

Using a separate loop (rather than the caller's, if any) means the sync API also works from
inside an already-running loop, where asyncio.run() would raise.
"""

import asyncio
import atexit
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


class LoopRunner:
    """Lazily-started background event loop."""

    def __init__(self, name: str = "genetics-sdk") -> None:
        self._name = name
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is None:
                loop = asyncio.new_event_loop()
                thread = threading.Thread(
                    target=loop.run_forever, name=self._name, daemon=True
                )
                thread.start()
                self._loop, self._thread = loop, thread
            return self._loop

    def run(self, coro: Coroutine[Any, Any, T]) -> T:
        return asyncio.run_coroutine_threadsafe(coro, self._ensure_loop()).result()

    def shutdown(self) -> None:
        with self._lock:
            loop, thread = self._loop, self._thread
            self._loop = self._thread = None
        if loop is None:
            return
        loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=5)
        loop.close()


_runner = LoopRunner()
atexit.register(_runner.shutdown)


def run(coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine on the shared SDK loop and return its result."""
    return _runner.run(coro)


def shutdown() -> None:
    _runner.shutdown()
