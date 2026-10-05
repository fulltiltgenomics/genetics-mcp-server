"""The shared SDK client keeps its connection pool on one event loop, whoever awaits it.

A script that fans out takes `genetics.get_client()` and runs `asyncio.run(main())`, which
is a second loop next to the one the synchronous functions use. These run against a real
socket server, because the failure is in the connection pool: an httpx connection belongs
to the loop that opened it, and a mock transport has no connections to get wrong.
"""

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import polars as pl
import pytest

from genetics_mcp_server import sdk
from genetics_mcp_server.sdk.client import GeneticsClient

DELAY_S = 0.05


class _Api(BaseHTTPRequestHandler):
    # keep-alive, or nothing is ever left in the pool for the other loop to pick up
    protocol_version = "HTTP/1.1"
    lock = threading.Lock()
    in_flight = 0
    most_in_flight = 0

    def _answer(self):
        length = int(self.headers.get("content-length") or 0)
        if length:
            self.rfile.read(length)
        with _Api.lock:
            _Api.in_flight += 1
            _Api.most_in_flight = max(_Api.most_in_flight, _Api.in_flight)
        time.sleep(DELAY_S)
        with _Api.lock:
            _Api.in_flight -= 1
        body = json.dumps([{"phenotype": "X", "beta": 0.1, "mlog10p": 1.0}]).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = do_POST = _answer

    def log_message(self, *args):
        pass


@pytest.fixture
def shared_client(monkeypatch):
    """`sdk.get_client()` against a local server, with the process-wide client put back."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Api)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("GENETICS_API_URL", f"http://127.0.0.1:{server.server_port}/api")
    monkeypatch.setattr(sdk, "_client", None)
    _Api.in_flight = _Api.most_in_flight = 0
    try:
        yield sdk.get_client()
    finally:
        sdk.close()
        server.shutdown()
        server.server_close()


def _one():
    return sdk.summary_stats(phenotypes=["A"], variants=["1:1:A:T"])


async def _fan(client, n=16):
    got = await asyncio.gather(
        *(client.summary_stats(phenotypes=["A"], variants=["1:1:A:T"]) for _ in range(n)),
        return_exceptions=True,
    )
    return [g for g in got if not isinstance(g, pl.DataFrame)]


def test_a_fan_out_after_synchronous_calls_loses_no_request(shared_client):
    # the synchronous calls leave connections in the pool; reused from the script's loop
    # they failed "bound to a different event loop", one request per pooled connection
    assert _one().height == 1 and _one().height == 1
    assert asyncio.run(_fan(shared_client)) == []


def test_synchronous_calls_still_work_after_the_scripts_loop_has_closed(shared_client):
    # every later call used to fail "Event loop is closed", on connections whose loop was
    # the one asyncio.run() had just torn down
    assert asyncio.run(_fan(shared_client)) == []
    assert _one().height == 1


def test_a_script_may_run_its_own_loop_more_than_once(shared_client):
    assert asyncio.run(_fan(shared_client)) == []
    assert asyncio.run(_fan(shared_client)) == []


def test_awaited_calls_are_still_concurrent(shared_client):
    assert asyncio.run(_fan(shared_client, n=12)) == []
    # counted at the server rather than timed: moved onto one loop, the calls must still
    # overlap there and not queue up behind each other
    assert _Api.most_in_flight >= 4


def test_cancelling_the_awaiting_task_cancels_the_request(shared_client, monkeypatch):
    seen = {}

    async def hangs(*args, **kwargs):
        seen["started"] = True
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            seen["cancelled"] = True
            raise

    monkeypatch.setattr(shared_client._executor, "get_summary_stats", hangs)

    async def main():
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                shared_client.summary_stats(phenotypes=["A"], variants=["1:1:A:T"]), 0.2
            )
        # the cancellation crosses to the SDK's loop, which needs a moment to deliver it
        for _ in range(50):
            if seen.get("cancelled"):
                break
            await asyncio.sleep(0.02)

    asyncio.run(main())
    assert seen == {"started": True, "cancelled": True}


async def test_a_client_built_directly_stays_on_its_callers_loop():
    # only the process-wide client is shared with the synchronous functions; one a caller
    # built around its own executor may be the running service's, bound to that loop
    ran_on = []

    class _Executor:
        async def get_summary_stats(self, *args, **kwargs):
            ran_on.append(asyncio.get_running_loop())
            return {"success": True, "results": []}

    client = GeneticsClient(executor=_Executor())
    assert client._home is None
    await client.summary_stats(phenotypes=["A"], variants=["1:1:A:T"])
    assert ran_on == [asyncio.get_running_loop()]
