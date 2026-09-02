"""LD is fetched through results-api, not straight from the public internet.

This is the property the whole proxy exists for, and it is exactly the kind a later refactor
undoes without noticing: both call sites used to name `https://api.finngen.fi/api/ld` on
`external_client`, which works everywhere EXCEPT the one place the LD is wanted. The sandbox
has no DNS and no internet egress by design, so a `run_analysis` script's `genetics.ld(...)`
resolved nothing and every locuszoom came back uncoloured.

So these assert the destination and the credential, not just that a value comes back.
"""

import contextlib
from unittest.mock import patch
from urllib.parse import urlsplit

import pytest

from genetics_mcp_server.tools.executor import ToolExecutor

BASE = "http://api.internal:2000/api"

UPSTREAM_ENTRIES = [
    {"variation1": "6:44693011:A:G", "variation2": "6:44682355:C:G", "r2": 0.81, "d_prime": 0.94},
    {"variation1": "6:44693011:A:G", "variation2": "6:44690000:T:C", "r2": 0.42, "d_prime": 0.7},
]


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {"ld": UPSTREAM_ENTRIES}

    def json(self):
        return self._payload


@contextlib.asynccontextmanager
async def executor_with(response=None, capture=None):
    executor = ToolExecutor(api_base_url=BASE)
    response = response or FakeResponse()

    async def fake_get(url, **kwargs):
        if capture is not None:
            capture["url"] = url
            capture["kwargs"] = kwargs
        return response

    async def refuse(url, **kwargs):  # pragma: no cover - only runs on a regression
        raise AssertionError(f"LD reached the public internet directly: {url}")

    try:
        with patch.object(executor.client, "get", side_effect=fake_get):
            with patch.object(executor.external_client, "get", side_effect=refuse):
                yield executor
    finally:
        await executor.close()


async def test_variants_in_ld_asks_results_api_and_not_the_ld_server():
    capture = {}
    async with executor_with(capture=capture) as executor:
        result = await executor.get_variants_in_ld("6:44693011:A:G", window=250000)

    assert result["success"] is True
    url = capture["url"]
    assert url.startswith(BASE), f"LD went to {url}"
    assert urlsplit(url).path == "/api/v1/ld/6%3A44693011%3AA%3AG"
    assert "finngen.fi" not in url


async def test_the_pair_lookup_uses_the_same_path():
    capture = {}
    async with executor_with(capture=capture) as executor:
        result = await executor.get_ld_between_variants("6:44693011:A:G", "6:44682355:C:G")

    assert result["success"] is True
    assert result["in_ld"] is True
    assert result["r2"] == 0.81
    assert urlsplit(capture["url"]).path.startswith("/api/v1/ld/")


async def test_the_proxys_parameter_names_are_sent_not_the_upstreams():
    """results-api takes r2_threshold and translates; sending the upstream's `r2_thresh`
    here would be silently dropped as an unknown query parameter — and results-api rejects
    unknown ones, so it would 4xx rather than quietly widening the threshold."""
    capture = {}
    async with executor_with(capture=capture) as executor:
        await executor.get_variants_in_ld("6:44693011:A:G", window=1000, r2_threshold=0.25)

    params = capture["kwargs"]["params"]
    assert params == {"window": 1000, "r2_threshold": 0.25, "panel": "sisu42"}
    assert "r2_thresh" not in params


async def test_the_upstream_entries_are_still_parsed_the_way_they_always_were():
    """The proxy passes the upstream's own field names through, so the 'other variant'
    extraction and the r² sort stay here rather than moving into results-api."""
    async with executor_with() as executor:
        result = await executor.get_variants_in_ld("6:44693011:A:G")

    assert [v["variant"] for v in result["variants"]] == ["6:44682355:C:G", "6:44690000:T:C"]
    assert [v["r2"] for v in result["variants"]] == [0.81, 0.42]


async def test_an_unreachable_ld_server_reads_as_unavailable_rather_than_as_a_bad_request():
    async with executor_with(FakeResponse(status_code=502)) as executor:
        result = await executor.get_variants_in_ld("6:44693011:A:G")

    assert result["success"] is False
    assert "unavailable" in result["error"]


async def test_a_refused_request_says_what_to_change():
    async with executor_with(
        FakeResponse(status_code=422, payload={"detail": "window 99999999 is outside 1..11000000"})
    ) as executor:
        result = await executor.get_variants_in_ld("6:44693011:A:G")

    assert result["success"] is False
    assert "refused" in result["error"]
    assert "11000000" in result["error"]


@pytest.mark.parametrize("status", [500, 503])
async def test_any_other_status_still_fails_closed(status):
    async with executor_with(FakeResponse(status_code=status)) as executor:
        result = await executor.get_variants_in_ld("6:44693011:A:G")
    assert result["success"] is False
    assert str(status) in result["error"]
