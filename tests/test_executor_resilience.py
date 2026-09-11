"""Unit tests for upstream-unreachable handling in the tool executor.

Self-contained: they point the executor at a closed port so the connection is
refused, exercising the real _ResilientAsyncClient path (no running API needed).
"""

import httpx

from genetics_mcp_server.tools import ToolExecutor
from genetics_mcp_server.tools.executor import (
    _UNREACHABLE_HEADER,
    INTERNAL_ERROR_MSG,
    MOUSEMINE_UNAVAILABLE_MSG,
    UPSTREAM_UNREACHABLE_MSG,
    _ResilientAsyncClient,
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


async def test_get_database_schema_flags_unreachable():
    executor = ToolExecutor(bigquery_api_url=UNREACHABLE_URL)
    try:
        result = await executor.get_database_schema()
        assert result["success"] is False
        assert result.get("unreachable") is True
        assert result["error"] == UPSTREAM_UNREACHABLE_MSG
    finally:
        await executor.close()


async def test_search_mgi_reports_mousemine_read_timeout_as_unavailable():
    """MouseMine's characteristic failure is to accept the connection and then never
    answer. _ResilientAsyncClient deliberately rewrites only connect-level failures, so
    the read timeout has to be caught in _mousemine_query — otherwise it reaches
    search_mgi's generic handler as an opaque internal error plus a logged traceback."""

    def hang(request):
        raise httpx.ReadTimeout("timed out", request=request)

    executor = ToolExecutor()
    try:
        await executor.external_client.aclose()
        executor.external_client = _ResilientAsyncClient(
            timeout=2.0, transport=httpx.MockTransport(hang)
        )
        for query_type in ("gene_phenotypes", "phenotype_genes", "allele", "ortholog"):
            result = await executor.search_mgi("Trim28", query_type=query_type)
            assert result["success"] is False, query_type
            assert result["error"] == MOUSEMINE_UNAVAILABLE_MSG, query_type
            assert result["error"] != INTERNAL_ERROR_MSG, query_type
    finally:
        await executor.close()
