"""Unit tests for upstream-unreachable handling in the tool executor.

Self-contained: they point the executor at a closed port so the connection is
refused, exercising the real _ResilientAsyncClient path (no running API needed).
"""

import httpx

from genetics_mcp_server.tools import ToolExecutor
from genetics_mcp_server.tools.executor import (
    UPSTREAM_UNREACHABLE_MSG,
    _ResilientAsyncClient,
    _UNREACHABLE_HEADER,
)

# port 1 is privileged/unused — connection is refused immediately
UNREACHABLE_URL = "http://127.0.0.1:1"


async def test_resilient_client_returns_synthetic_503_on_connect_error():
    client = _ResilientAsyncClient(timeout=2.0)
    try:
        resp = await client.get(f"{UNREACHABLE_URL}/anything")
        assert resp.status_code == 503
        assert resp.headers.get(_UNREACHABLE_HEADER) == "1"
        assert resp.text == UPSTREAM_UNREACHABLE_MSG
    finally:
        await client.aclose()


async def test_resilient_client_passes_through_real_responses():
    # a genuine 4xx/5xx from upstream must NOT be tagged as unreachable
    transport = httpx.MockTransport(lambda req: httpx.Response(404, text="nope"))
    client = _ResilientAsyncClient(timeout=2.0, transport=transport)
    try:
        resp = await client.get("http://example.test/x")
        assert resp.status_code == 404
        assert resp.headers.get(_UNREACHABLE_HEADER) is None
    finally:
        await client.aclose()


async def test_get_bigquery_schema_flags_unreachable():
    executor = ToolExecutor(bigquery_api_url=UNREACHABLE_URL)
    try:
        result = await executor.get_bigquery_schema()
        assert result["success"] is False
        assert result.get("unreachable") is True
        assert result["error"] == UPSTREAM_UNREACHABLE_MSG
    finally:
        await executor.close()
