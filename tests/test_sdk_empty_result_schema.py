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


# The results-api endpoints that ACTUALLY advertise `X-Columns`, re-derived from that
# repo's routers: `range_response`'s JSON branch (directly or through a module's
# `_range_stream_response` helper), plus the endpoints that declare their columns for
# genetics-results-suite-8a1. This list is the claim this file makes about the other repo,
# and `_api_client` below honours it per request path.
#
# WHY IT EXISTS: a transport that attaches the header to every response passes against a
# server that attaches it to none. That is how `credible_sets(phenotype=...)` and
# `exome(phenotype=...)` sat in the parametrize below as covered while their endpoints —
# `json_phenotype` callers, not `range_response` callers — sent no header at all.
_ADVERTISING_PATHS = (
    "/v1/credible_sets_by_gene/",
    "/v1/credible_sets_by_variant",
    "/v1/credible_sets_by_region/",
    "/v1/credible_sets_by_qtl_gene/",
    "/v1/credible_sets_by_phenotype/",
    "/v1/credible_sets_by_phenotype_leads/",
    "/v1/colocalization_by_variant/",
    "/v1/colocalization_by_credible_set_id/",
    "/v1/exome_results_by_gene/",
    "/v1/exome_results_by_variant/",
    "/v1/exome_results_by_region/",
    "/v1/exome_results_by_phenotype/",
    "/v1/expression_by_gene/",
    "/v1/gene_based_results_by_phenotype/",
    "/v1/gene_disease/",
    "/v1/gene_group/members",
    "/v1/gene_to_peaks/",
    "/v1/genes_in_region/",
    "/v1/hla/",
    "/v1/mpra/",
    "/v1/nearest_genes/",
    "/v1/open_chromatin/",
    "/v1/peak_to_genes/",
    "/v1/search",
    "/v1/summary_stats/",
    "/v1/summary_stats_by_region/",
    "/v1/variant_annotation/",
    "/v1/variant_effect/",
)
# Deliberately absent: /v1/rsid/variants and the LD server (still uncovered), and
# /v1/gene_based/{gene}, which serves TSV — its schema is the header line in the body.


def _api_client(body, columns=("phenotype", "mlog10p"), status=200) -> GeneticsClient:
    """A fake results-api that advertises only where the real one does."""
    def handler(request: httpx.Request) -> httpx.Response:
        headers = (
            {"X-Columns": ",".join(columns)}
            if any(p in request.url.path for p in _ADVERTISING_PATHS)
            else {}
        )
        return httpx.Response(status, json=body, headers=headers)

    client = GeneticsClient()
    client._executor.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


def _client(body, headers=None) -> GeneticsClient:
    """A real ToolExecutor whose HTTP layer is a stub, wired the way the SDK wires it."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body, headers=headers or {})

    client = GeneticsClient()
    client._executor.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
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
        ("credible_sets", {"phenotype": "K11_COELIAC", "leads_only": True}),
        ("exome", {"phenotype": "K11_COELIAC"}),
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
    """Scoped as the class, not as hla() alone — the bead's explicit instruction.

    Through `_api_client`, so a branch whose endpoint does not advertise fails here.
    """
    client = _api_client([])
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
    """An endpoint this file does not claim advertises must degrade, not break."""
    client = _api_client([], columns=["variant", "rsid"])
    frame = await client.search(rsids="rs123")  # /v1/rsid/variants, still uncovered
    assert frame.columns == []
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
    assert GeneticsClient()._executor._expose_columns is True
    assert GeneticsClient(executor=ToolExecutor())._executor._expose_columns is False


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


# ------------------------------------ the four that compute their JSON (suite-8a1)
#
# These do not go through results-api's `range_response`, so there is no file header to
# advertise: each endpoint DECLARES its columns instead (genetics-results-api
# app/core/responses.py::verified_columns_header, which refuses when a declaration and a
# returned row disagree). From here they look identical to the streaming ones — the same
# header, lifted by the same `_columns_meta` — which is the point of checking them here.

GENE_COLUMNS = [
    "gene_name", "chrom", "gene_start", "gene_end", "gene_strand", "gene_type",
    "hgnc_symbol", "hgnc_name", "hgnc_alias_symbol", "hgnc_prev_symbol",
]
DISEASE_COLUMNS = [
    "resource", "uuid", "gene_symbol", "disease_curie", "disease_title",
    "classification", "mode_of_inheritance", "submitter",
]


def _client_from(handler) -> GeneticsClient:
    client = GeneticsClient()
    client._executor.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


async def test_empty_gene_annotations_by_region_keeps_its_columns():
    client = _api_client([], GENE_COLUMNS)
    frame = await client.gene_annotations(region="1:55000000-55100000")
    assert frame.columns == GENE_COLUMNS
    assert frame.height == 0
    assert frame.filter(pl.col("gene_start") > 0).height == 0


async def test_empty_gene_annotations_by_proximity_keeps_its_columns():
    columns = GENE_COLUMNS[:6] + ["distance"] + GENE_COLUMNS[6:]
    client = _api_client([], columns)
    frame = await client.gene_annotations(nearest_to="5:35874575:T:C")
    assert frame.columns == columns


async def test_empty_gene_group_membership_keeps_its_columns():
    members = ["hgnc_id", "symbol", "ensembl_id", "chr", "gene_start", "gene_end"]
    client = _api_client(
        {"group_id": 110, "group_name": "GPCRs", "count": 0, "members": []}, members
    )
    frame = await client.gene_annotations(group=110)
    assert frame.columns == members
    assert frame.height == 0


async def test_empty_gene_disease_keeps_its_columns_across_the_404():
    """This endpoint expresses "no associations" as a 404 that the client reads as empty."""
    def handler(request):
        return httpx.Response(
            404,
            json={"detail": "No disease associations found for gene NOSUCHGENE"},
            headers={"X-Columns": ",".join(DISEASE_COLUMNS)},
        )

    frame = await _client_from(handler).gene_disease("NOSUCHGENE")
    assert frame.columns == DISEASE_COLUMNS
    assert frame.height == 0
    assert frame.filter(pl.col("classification") == "Strong").height == 0


@pytest.mark.parametrize("kind", ["genes", "phenotypes"])
async def test_empty_search_keeps_its_columns(kind):
    columns = ["type", "symbol", "name", "match_type", "match_score"]
    client = _api_client([], columns)
    frame = await client.search("nosuchthing", kind=kind)
    assert frame.columns == columns
    assert frame.height == 0


async def test_empty_gene_burden_by_gene_keeps_the_columns_of_its_tsv_header():
    """A TSV body: `tabix -h` prints the header even with no matching row."""
    header = "dataset\ttrait\tgene\tmlog10p_burden"

    def handler(request):
        return httpx.Response(200, text=header + "\n")

    frame = await _client_from(handler).gene_burden(gene="NOSUCHGENE")
    assert frame.columns == ["dataset", "trait", "gene", "mlog10p_burden"]
    assert frame.height == 0
    assert frame.filter(pl.col("mlog10p_burden") > 4).height == 0


async def test_empty_gene_burden_by_phenotype_keeps_its_columns():
    columns = ["dataset", "trait", "gene", "mlog10p_burden"]
    client = _api_client([], columns)
    frame = await client.gene_burden(phenotype="NOSUCHTRAIT")
    assert frame.columns == columns


async def test_a_default_executor_still_adds_nothing_for_a_tsv_response():
    """The MCP payload invariant, for the header-line source as well as the header."""
    assert ToolExecutor()._columns_meta_from(["a", "b"]) == {}
    assert ToolExecutor(expose_columns=True)._columns_meta_from(["a", "b"]) == {
        "column_names": ["a", "b"]
    }
    assert ToolExecutor(expose_columns=True)._columns_meta_from(None) == {}
