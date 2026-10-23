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


async def test_the_pair_lookup_uses_results_apis_pair_route():
    """Not a window query filtered here: results-api's pair route is what knows, per panel,
    whether an empty answer means below the floor or possibly absent, and how far a pair
    may be apart."""
    capture = {}
    pair = {
        "variation1": "6:44693011:A:G",
        "variation2": "6:44682355:C:G",
        "r2": 0.81,
        "d_prime": 0.94,
    }
    async with executor_with(FakeResponse(payload={"ld": [pair]}), capture=capture) as executor:
        result = await executor.get_ld_between_variants("6:44693011:A:G", "6:44682355:C:G")

    assert result["success"] is True
    assert result["in_ld"] is True
    assert result["r2"] == 0.81
    assert result["d_prime"] == 0.94
    assert "r" not in result
    assert urlsplit(capture["url"]).path == "/api/v1/ld/6%3A44693011%3AA%3AG/6%3A44682355%3AC%3AG"
    assert capture["kwargs"]["params"] == {"r2_threshold": 0.1, "panel": "sisu42"}


async def test_a_pair_too_far_apart_is_results_apis_refusal_relayed():
    """The distance a pair may span is the panel's (2 Mb on the LD server, 3 Mb on the
    suite's own matrices), so it is not second-guessed here with one number."""
    async with executor_with(
        FakeResponse(
            status_code=422,
            payload={
                "detail": "6:10000000:A:G and 6:13000001:C:G are 3,000,001 bp apart; LD panel finngen_r12 holds pairs up to 3,000,000 bp apart"
            },
        )
    ) as executor:
        refused = await executor.get_ld_between_variants("6:10000000:A:G", "6:13000001:C:G")

    assert refused["success"] is False
    assert "3,000,000 bp apart" in refused["error"]


async def test_a_pair_on_two_chromosomes_is_refused_without_a_request():
    capture = {}
    async with executor_with(capture=capture) as executor:
        refused = await executor.get_ld_between_variants("6:44693011:A:G", "7:44682355:C:G")
    assert refused["success"] is False
    assert "same chromosome" in refused["error"]
    assert capture == {}


async def test_signed_r_comes_back_when_the_panel_carries_it():
    """The suite's own panels carry r with its sign, and the sign is the whole point: an
    estimator over correlated instruments cannot recover it from r2."""
    entries = [
        {"variation1": "6:44693011:A:G", "variation2": "6:44682355:C:G", "r": -0.9, "r2": 0.81},
        {"variation1": "6:44693011:A:G", "variation2": "6:44690000:T:C", "r": 0.6481, "r2": 0.42},
    ]
    payload = {"ld": entries, "fields": ["r", "r2"], "r2_floor": 0.01, "max_distance": 3_000_000}
    capture = {}
    async with executor_with(FakeResponse(payload=payload), capture=capture) as executor:
        result = await executor.get_variants_in_ld("6:44693011:A:G", panel="finngen_r12")

    assert capture["kwargs"]["params"]["panel"] == "finngen_r12"
    assert result["variants"] == [
        {"variant": "6:44682355:C:G", "r": -0.9, "r2": 0.81},
        {"variant": "6:44690000:T:C", "r": 0.6481, "r2": 0.42},
    ]
    assert (result["fields"], result["r2_floor"], result["max_distance"]) == (
        ["r", "r2"],
        0.01,
        3_000_000,
    )


async def test_fields_are_passed_through_and_only_those_come_back():
    entries = [
        {"variation1": "6:44693011:A:G", "variation2": "6:44690000:T:C", "r": 0.6481},
        {"variation1": "6:44693011:A:G", "variation2": "6:44682355:C:G", "r": -0.9},
    ]
    payload = {"ld": entries, "fields": ["r"], "r2_floor": 0.01, "max_distance": 3_000_000}
    capture = {}
    async with executor_with(FakeResponse(payload=payload), capture=capture) as executor:
        result = await executor.get_variants_in_ld(
            "6:44693011:A:G", panel="finngen_r12", fields=["r"]
        )
        params = dict(capture["kwargs"]["params"])
        await executor.get_ld_between_variants(
            "6:44693011:A:G", "6:44682355:C:G", panel="finngen_r12", fields="r, r2"
        )
        pair_params = dict(capture["kwargs"]["params"])

    assert params["fields"] == "r"
    assert pair_params["fields"] == "r,r2"
    # strongest first even with no r2 on the entry: |r| orders them
    assert [v["variant"] for v in result["variants"]] == ["6:44682355:C:G", "6:44690000:T:C"]
    assert all(set(v) == {"variant", "r"} for v in result["variants"])


async def test_an_empty_pair_on_an_own_panel_says_independent_not_unknown():
    """results-api reads both variants' blocks on its own panels, so an empty answer with a
    floor stated means both were found and the pair is below it."""
    own = {"ld": [], "fields": ["r", "r2"], "r2_floor": 0.01, "max_distance": 3_000_000}
    async with executor_with(FakeResponse(payload=own)) as executor:
        result = await executor.get_ld_between_variants(
            "6:44693011:A:G", "6:44682355:C:G", r2_threshold=0.0, panel="finngen_r12"
        )
    assert result["in_ld"] is False
    assert "independent" in result["message"]
    assert "0.01" in result["message"]
    assert result["r2_floor"] == 0.01

    upstream = {"ld": [], "fields": ["r2", "d_prime"], "r2_floor": None, "max_distance": None}
    async with executor_with(FakeResponse(payload=upstream)) as executor:
        result = await executor.get_ld_between_variants("6:44693011:A:G", "6:44682355:C:G")
    assert result["in_ld"] is False
    assert "not in reference panel" in result["message"]


async def test_a_statistic_the_panel_lacks_is_refused_by_name():
    async with executor_with(
        FakeResponse(
            status_code=422,
            payload={"detail": "LD panel sisu42 does not carry r; it carries r2, d_prime"},
        )
    ) as executor:
        result = await executor.get_variants_in_ld("6:44693011:A:G", fields="r")
    assert result["success"] is False
    assert "does not carry r" in result["error"]


async def test_the_panel_listing_is_relayed():
    capture = {}
    payload = {
        "panels": [
            {
                "panel": "finngen_r12",
                "fields": ["r", "r2"],
                "r2_floor": 0.01,
                "max_distance": 3_000_000,
                "n_samples": 520_210,
                "description": "...",
            }
        ],
        "default_panel": "sisu42",
        "upstream_fields": ["r2", "d_prime"],
    }
    async with executor_with(FakeResponse(payload=payload), capture=capture) as executor:
        result = await executor.get_ld_panels()
    assert urlsplit(capture["url"]).path == "/api/v1/ld/panels"
    assert result["success"] is True
    assert result["panels"][0]["panel"] == "finngen_r12"
    assert result["default_panel"] == "sisu42"


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


async def test_a_variant_the_panel_does_not_carry_is_not_reported_as_an_outage():
    async with executor_with(
        FakeResponse(
            status_code=404, payload={"detail": "6:44693011:A:G is not in LD panel sisu42"}
        )
    ) as executor:
        result = await executor.get_variants_in_ld("6:44693011:A:G")

    assert result["success"] is False
    assert "not in the LD panel" in result["error"]
    assert "unavailable" not in result["error"]
    assert result["error_code"] == "ld_variant_not_in_panel"


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
