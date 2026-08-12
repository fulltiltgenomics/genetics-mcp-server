"""The empty-result contract for the results-api-backed half of the SDK.

genetics-results-suite-6uk: `hla(phenotype=...)` on a no-hit phenotype raised
ColumnNotFoundError, because results-api's JSON range responses are bare arrays and an
empty one carries no schema. The BigQuery-backed branches never had this problem — db-api
derives `columns` from the job schema, which exists for a zero-row result — so only the
results-api branches are covered here; the db-api ones are pinned in test_sdk.py.

results-api now advertises the file's own header in an `X-Columns` response header
(app/core/responses.py::columns_header), the executor lifts it into `column_names` when
its owner asked for it, and the SDK builds a named empty frame from that.

These go through a fake httpx transport rather than a fake executor, because the wiring
that was missing spans all three layers and a fake executor would pin only the last one.
"""

import httpx
import polars as pl
import pytest

from genetics_mcp_server.sdk.client import GeneticsClient, _frame
from genetics_mcp_server.tools.executor import ToolExecutor

HLA_COLUMNS = [
    "resource", "version", "phenotype", "chr", "pos", "gene", "allele", "pval",
    "mlog10p", "beta", "se", "af", "af_cases", "af_controls", "info",
]


def _client(body, headers=None) -> GeneticsClient:
    """A real ToolExecutor whose HTTP layer is a stub, wired the way the SDK wires it."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body, headers=headers or {})

    client = GeneticsClient()
    client.executor.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


async def test_the_original_defect_no_hit_phenotype_then_filter():
    """The exact reproduction from the bead: empty result, then a filter on a column."""
    client = _client([], {"X-Columns": ",".join(HLA_COLUMNS)})
    frame = await client.hla(phenotype="NO_SUCH_TRAIT")
    assert frame.columns == HLA_COLUMNS
    assert frame.height == 0
    # this is what raised ColumnNotFoundError before
    assert frame.filter(pl.col("mlog10p") > 7.3).height == 0


@pytest.mark.parametrize(
    ("call", "kwargs"),
    [
        ("credible_sets", {"gene": "IL7R"}),
        ("credible_sets", {"variant": "5:35874575:T:C"}),
        ("credible_sets", {"phenotype": "K11_COELIAC"}),
        ("colocalization", {"variant": "5:35874575:T:C"}),
        ("exome", {"gene": "IL7R"}),
        ("exome", {"variant": "5:35874575:T:C"}),
        ("hla", {"phenotype": "K11_COELIAC"}),
        ("open_chromatin", {"variant": "5:35874575:T:C"}),
        ("open_chromatin", {"peak": "chr5:35874000-35875000"}),
        ("mpra", {"variant": "5:35874575:T:C"}),
        ("variant_effect", {"variant": "5:35874575:T:C"}),
        ("asm_qtl", {"variant": "5:35874575:T:C"}),
        ("peak_to_gene", {"peak": "chr5:35874000-35875000"}),
        ("variant_annotation", {"variant": "5:35874575:T:C"}),
        ("expression", {}),
        ("summary_stats", {"phenotypes": ["K11_COELIAC"], "variants": ["5:35874575:T:C"]}),
    ],
)
async def test_every_wired_results_api_branch_keeps_its_schema_when_empty(call, kwargs):
    """Scoped as the class, not as hla() alone — the bead's explicit instruction."""
    client = _client([], {"X-Columns": "phenotype,mlog10p"})
    args = ("IL7R",) if call == "expression" else ()
    frame = await getattr(client, call)(*args, **kwargs)
    assert frame.columns == ["phenotype", "mlog10p"], call
    assert frame.height == 0, call
    assert frame.filter(pl.col("mlog10p") > 7.3).height == 0, call


async def test_a_non_empty_result_is_unaffected_by_the_header():
    """`column_names` must not reroute named dict rows through the positional path."""
    rows = [{"phenotype": "K11_COELIAC", "mlog10p": 1596.65}]
    client = _client(rows, {"X-Columns": "phenotype,mlog10p"})
    frame = await client.hla(phenotype="K11_COELIAC")
    assert frame.columns == ["phenotype", "mlog10p"]
    assert frame["mlog10p"].to_list() == [1596.65]
    assert frame.schema["mlog10p"] == pl.Float64


async def test_an_endpoint_that_does_not_advertise_degrades_to_the_old_behaviour():
    """No header (an endpoint outside the range_response family) must not break."""
    client = _client([])
    frame = await client.search(query="coeliac")
    assert frame.height == 0


# ------------------------------------------------------------------ MCP output


def test_a_default_executor_adds_nothing_to_its_result_dicts():
    """The MCP tool payload IS the executor dict, and this epic forbids changing it."""
    resp = httpx.Response(200, headers={"X-Columns": "a,b"}, request=httpx.Request("GET", "http://t"))
    assert ToolExecutor()._columns_meta(resp) == {}
    assert ToolExecutor(row_limit=None)._columns_meta(resp) == {}
    assert ToolExecutor(expose_columns=True)._columns_meta(resp) == {"column_names": ["a", "b"]}
    # an endpoint that does not advertise stays byte-identical even when asked
    bare = httpx.Response(200, request=httpx.Request("GET", "http://t"))
    assert ToolExecutor(expose_columns=True)._columns_meta(bare) == {}


def test_the_sdk_builds_an_executor_that_exposes_columns():
    assert GeneticsClient().executor._expose_columns is True
    assert GeneticsClient(executor=ToolExecutor()).executor._expose_columns is False


# ------------------------------------------------------------------ _frame


def test_db_api_columns_still_win_over_the_results_api_list():
    """`columns` is REQUIRED to read db-api's positional rows; `empty_columns` is advisory."""
    frame = _frame([[1, 2]], columns=["a", "b"], empty_columns=["wrong"])
    assert frame.columns == ["a", "b"]
    assert frame.row(0) == (1, 2)


def test_empty_columns_alone_names_an_empty_frame():
    assert _frame([], empty_columns=["a", "b"]).columns == ["a", "b"]
    assert _frame([]).columns == []


def test_mixed_type_columns_still_take_the_strict_false_fallback():
    """The reason `empty_columns` is not merged into `columns`: dicts keep from_dicts."""
    rows = [{"x": 1}, {"x": "two"}]
    assert _frame(rows, empty_columns=["x"]).height == 2
