"""Proves, as a committed test, what genetics-results-suite-4h6.11's review checked by
hand: percent-encoding a caller-controlled value with `_seg()` and putting it into a URL
*path segment* survives a real ASGI request byte-identically.

tests/test_bigquery_gene_tools.py's `test_url_path_segments_cannot_escape_their_endpoint`
already proves containment (the encoded segment cannot escape its endpoint prefix), but
nothing asserted the positive case: that the exact original value is what a real server
receives as the path parameter. A future change to `_seg()`, to a route signature, or to
how the identifier is unquoted could silently break every by-variant/by-region/by-gene
tool while that containment check stays green.

Routes below mirror real call sites in tools/executor.py (`_seg(...)` is the same helper
those call sites use to build the path segment):
  - credible_sets_by_variant/{variant}, credible_sets_by_region/{region}
  - exome_results_by_gene/{gene}
  - credible_sets/{resource_or_dataset}/stats  (dotted dataset ids, e.g. Open_Targets_26.06)

genetics-results-suite-25p.
"""

import httpx
import pytest
from fastapi import FastAPI

from genetics_mcp_server.tools.executor import _seg

_app = FastAPI()


@_app.get("/v1/credible_sets_by_variant/{variant}")
async def _echo_variant(variant: str):
    return {"received": variant}


@_app.get("/v1/credible_sets_by_region/{region}")
async def _echo_region(region: str):
    return {"received": region}


@_app.get("/v1/exome_results_by_gene/{gene}")
async def _echo_gene(gene: str):
    return {"received": gene}


@_app.get("/v1/credible_sets/{resource_or_dataset}/stats")
async def _echo_dataset(resource_or_dataset: str):
    return {"received": resource_or_dataset}


@pytest.fixture
async def client():
    """A real ASGI round trip (httpx's ASGITransport drives the FastAPI/Starlette
    routing stack exactly as uvicorn would, including unquoting the request path before
    matching), not a direct call into the route function."""
    transport = httpx.ASGITransport(app=_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


# the four CPRA separators this suite documents as accepted, e.g. tools/definitions.py:
# "Single variant in chr:pos:ref:alt format ... Any separator (: - _ |) accepted."
_VARIANT_SEPARATOR_CASES = [
    "19:44908684:T:C",
    "19-44908684-T-C",
    "19_44908684_T_C",
    "19|44908684|T|C",
]


@pytest.mark.parametrize("variant", _VARIANT_SEPARATOR_CASES)
async def test_variant_round_trips_for_every_documented_separator(client, variant):
    resp = await client.get(f"/v1/credible_sets_by_variant/{_seg(variant)}")

    assert resp.status_code == 200
    assert resp.json()["received"] == variant


async def test_region_round_trips(client):
    region = "1:13668-14506"

    resp = await client.get(f"/v1/credible_sets_by_region/{_seg(region)}")

    assert resp.status_code == 200
    assert resp.json()["received"] == region


async def test_gene_symbol_with_punctuation_round_trips(client):
    gene = "HLA-DRB1"

    resp = await client.get(f"/v1/exome_results_by_gene/{_seg(gene)}")

    assert resp.status_code == 200
    assert resp.json()["received"] == gene


async def test_dotted_dataset_id_round_trips(client):
    dataset_id = "Open_Targets_26.06"

    resp = await client.get(f"/v1/credible_sets/{_seg(dataset_id)}/stats")

    assert resp.status_code == 200
    assert resp.json()["received"] == dataset_id


async def test_literal_percent_encoded_value_round_trips(client):
    """The other cases above (`: - _ | . @`) are all legal unencoded in a path segment,
    so an identity `_seg` would pass them too — they don't discriminate. A literal '%'
    followed by hex digits does: left unencoded, it already looks like a valid escape, so
    httpx does not re-encode it, but the server DOES decode it before path matching. If
    `_seg` did not encode '%' to '%25', a caller-supplied "%3A" would silently arrive on
    the other side as ":" instead of round-tripping as the literal text "%3A"."""
    gene = "BRCA1%3Atest"

    resp = await client.get(f"/v1/exome_results_by_gene/{_seg(gene)}")

    assert resp.status_code == 200
    assert resp.json()["received"] == gene
