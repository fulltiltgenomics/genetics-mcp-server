"""Unit tests for the genetics SDK.

The SDK collapses the by_gene/by_variant/by_region/by_phenotype tool grid into one
function per data product, so the thing worth testing is the dispatch: which executor
method a given argument shape reaches, and that ambiguous shapes are refused rather than
silently resolved. No network is involved — the executor is faked.
"""

import subprocess
import sys
from unittest.mock import AsyncMock

import polars as pl
import pytest

import genetics_mcp_server.sdk as genetics
from genetics_mcp_server.sdk.client import GeneticsClient, parse_region
from genetics_mcp_server.sdk.errors import GeneticsError, GeneticsUsageError
from genetics_mcp_server.tools.executor import ToolExecutor


class FakeExecutor:
    """Records which executor method the SDK dispatched to."""

    def __init__(self, payload=None):
        self.calls: list[tuple[str, tuple, dict]] = []
        self._row_limit = 500
        self._payload = payload or {
            "success": True,
            "results": [{"variant": "1:1:A:G", "pip": 0.9}],
            "variants": [{"variant": "1:1:A:G", "r2": 0.8}],
            "genes": [{"symbol": "IL7R"}],
            "members": [{"symbol": "IL7R"}],
        }

    def __getattr__(self, name):
        async def call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return self._payload

        return call

    @property
    def last(self):
        return self.calls[-1]


def make_client(payload=None) -> tuple[GeneticsClient, FakeExecutor]:
    executor = FakeExecutor(payload)
    return GeneticsClient(executor=executor), executor


# --------------------------------------------------------------------- dispatch


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"gene": "IL7R"}, "get_credible_sets_by_gene"),
        ({"qtl_gene": "IL7R"}, "get_credible_sets_by_qtl_gene"),
        ({"variant": "5:35874575:T:C"}, "get_credible_sets_by_variant"),
        ({"region": "5:35800000-35900000"}, "get_credible_sets_by_region"),
        ({"phenotype": "T1D"}, "get_credible_sets_by_phenotype"),
        ({"phenotype": "T1D", "leads_only": True}, "get_credible_set_leads_by_phenotype"),
        (
            {"phenotype": "T1D", "credible_set_id": "cs1"},
            "get_credible_set_by_id",
        ),
    ],
)
async def test_credible_sets_dispatches_on_argument_shape(kwargs, expected):
    client, executor = make_client()
    await client.credible_sets(**kwargs)
    assert executor.last[0] == expected


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"gene": "APOE"}, "get_mpra_by_gene"),
        ({"variant": "19:44908822:T:C"}, "get_mpra_by_variant"),
        ({"region": "19:44900000-44910000"}, "get_mpra_by_region"),
    ],
)
async def test_mpra_dispatches_on_argument_shape(kwargs, expected):
    client, executor = make_client()
    await client.mpra(**kwargs)
    assert executor.last[0] == expected


@pytest.mark.parametrize(
    ("method", "kwargs", "expected"),
    [
        ("exome", {"gene": "APOE"}, "get_exome_results_by_gene"),
        ("exome", {"variant": "19:44908822:T:C"}, "get_exome_results_by_variant"),
        ("exome", {"region": "19:1-2"}, "get_exome_results_by_region"),
        ("exome", {"phenotype": "T1D"}, "get_exome_results_by_phenotype"),
        ("gene_burden", {"gene": "APOE"}, "get_gene_based_results"),
        ("gene_burden", {"phenotype": "T1D"}, "get_gene_based_results_by_phenotype"),
        ("hla", {"phenotype": "K11_COELIAC"}, "get_hla_by_phenotype"),
        ("hla", {"allele": "B*27:05"}, "get_hla_by_allele"),
        ("asm_qtl", {"gene": "APOE"}, "get_asm_qtl_by_gene"),
        ("asm_qtl", {"variant": "19:44908822:T:C"}, "get_asm_qtl_by_variant"),
        ("variant_effect", {"gene": "APOE"}, "get_variant_effect_by_gene"),
        ("variant_effect", {"variant": "19:44908822:T:C"}, "get_variant_effect_by_variant"),
        ("open_chromatin", {"gene": "APOE"}, "get_open_chromatin_by_gene"),
        ("open_chromatin", {"peak": "peak1"}, "get_open_chromatin_by_peak"),
        ("open_chromatin", {"region": "19:1-2"}, "get_open_chromatin_by_region"),
        ("open_chromatin", {"variant": "19:44908822:T:C"}, "get_open_chromatin_by_variant"),
        ("peak_to_gene", {"peak": "peak1"}, "get_peak_to_genes"),
        ("peak_to_gene", {"gene": "APOE"}, "get_gene_to_peaks"),
        ("colocalization", {"variant": "19:44908822:T:C"}, "get_colocalization"),
        (
            "colocalization",
            {"phenotype": "T1D", "credible_set_id": "cs1"},
            "get_colocalization_by_credible_set",
        ),
        ("gene_annotations", {"region": "19:1-2"}, "get_genes_in_region"),
        ("gene_annotations", {"nearest_to": "19:44908822:T:C"}, "get_nearest_genes"),
        ("gene_annotations", {"group": 1234}, "get_gene_group_members"),
    ],
)
async def test_product_functions_dispatch_on_argument_shape(method, kwargs, expected):
    client, executor = make_client()
    await getattr(client, method)(**kwargs)
    assert executor.last[0] == expected


async def test_summary_stats_dispatches_variants_vs_region():
    client, executor = make_client()
    await client.summary_stats(variants=["1:1:A:G"], phenotypes=["T1D"])
    assert executor.last[0] == "get_summary_stats"
    await client.summary_stats(region="1:1-2", phenotypes="T1D")
    assert executor.last[0] == "get_summary_stats_by_region"


async def test_summary_stats_accepts_a_single_phenotype_string():
    client, executor = make_client()
    await client.summary_stats(region="1:1-2", phenotypes="T1D")
    assert executor.last[1][1] == ["T1D"]


async def test_ld_dispatches_pair_vs_neighbourhood_with_shape_specific_defaults():
    client, executor = make_client(
        {"success": True, "variant1": "1:1:A:G", "variant2": "1:2:C:T", "in_ld": True, "r2": 0.9}
    )
    await client.ld("1:1:A:G", "1:2:C:T")
    assert executor.last[0] == "get_ld_between_variants"
    assert executor.last[2]["r2_threshold"] == 0.1

    client, executor = make_client()
    await client.ld("1:1:A:G")
    assert executor.last[0] == "get_variants_in_ld"
    assert executor.last[2]["r2_threshold"] == 0.6


async def test_search_dispatches_between_index_and_rsid_lookup():
    client, executor = make_client()
    await client.search("diabetes")
    assert executor.last[0] == "search_phenotypes"
    await client.search("PCSK9", kind="genes")
    assert executor.last[0] == "search_genes"
    await client.search(rsids=["rs1234567", "rs7654321"])
    assert executor.last[0] == "lookup_variants_by_rsid"
    assert executor.last[1][0] == "rs1234567,rs7654321"


# --------------------------------------------------------------------- argument errors


@pytest.mark.parametrize("kwargs", [{}, {"gene": "IL7R", "variant": "1:1:A:G"}])
async def test_ambiguous_or_empty_argument_shape_is_refused(kwargs):
    client, _ = make_client()
    with pytest.raises(GeneticsUsageError):
        await client.credible_sets(**kwargs)


async def test_credible_set_id_requires_phenotype():
    client, _ = make_client()
    with pytest.raises(GeneticsUsageError):
        await client.credible_sets(credible_set_id="cs1")


@pytest.mark.parametrize("region", ["not-a-region", "19:abc-def", "19-1-2-3"])
def test_parse_region_rejects_malformed_input(region):
    with pytest.raises(GeneticsUsageError):
        parse_region(region)


@pytest.mark.parametrize(
    ("region", "expected"),
    [
        ("chr19:44900000-44910000", ("19", 44900000, 44910000)),
        ("19:44,900,000-44,910,000", ("19", 44900000, 44910000)),
        ("X:1-2", ("X", 1, 2)),
    ],
)
def test_parse_region_accepts_common_spellings(region, expected):
    assert parse_region(region) == expected


# --------------------------------------------------------------------- return contract


async def test_rows_come_back_as_a_polars_frame():
    client, _ = make_client()
    frame = await client.credible_sets(gene="IL7R")
    assert isinstance(frame, pl.DataFrame)
    assert frame["pip"].to_list() == [0.9]


async def test_empty_result_is_an_empty_frame_not_an_error():
    client, _ = make_client({"success": True, "results": []})
    frame = await client.credible_sets(gene="IL7R")
    assert isinstance(frame, pl.DataFrame)
    assert frame.height == 0


async def test_failure_raises_instead_of_returning_a_flag():
    client, _ = make_client({"success": False, "error": "HTTP 500: boom"})
    with pytest.raises(GeneticsError, match="boom"):
        await client.credible_sets(gene="IL7R")


async def test_sql_returns_a_frame_with_the_query_columns():
    client, _ = make_client(
        {"success": True, "columns": ["gene", "pip"], "rows": [["IL7R", 0.9], ["APOE", 0.5]]}
    )
    frame = await client.sql("SELECT gene, pip FROM credible_sets_v")
    assert frame.columns == ["gene", "pip"]
    assert frame.height == 2


async def test_sql_with_no_rows_keeps_the_column_names():
    client, _ = make_client({"success": True, "columns": ["gene"], "rows": []})
    frame = await client.sql("SELECT gene FROM credible_sets_v LIMIT 0")
    assert frame.columns == ["gene"]
    assert frame.height == 0


async def test_client_lifts_the_context_row_cap_only_on_executors_it_builds():
    """The 500-row inline cap exists to protect the model's context; a script filters rows
    itself and must not receive a positional prefix.

    An injected executor is left alone: it may be the running service's shared one, and
    lifting its cap in place would flood the model's context at every MCP call site.
    """
    own = GeneticsClient()
    try:
        assert own._executor._row_limit is None
    finally:
        await own.close()

    injected = ToolExecutor()
    try:
        assert injected._row_limit == ToolExecutor._REGION_ROW_LIMIT
        GeneticsClient(executor=injected)
        assert injected._row_limit == ToolExecutor._REGION_ROW_LIMIT
    finally:
        await injected.close()


async def test_bigquery_rows_are_named_by_the_query_columns():
    """db-api returns rows positionally. Without `columns` polars transposes them into
    column_0/column_1 with every value stringified, and nothing raises."""
    client, _ = make_client(
        {
            "success": True,
            "results": [["19:44908822:T:C", 0.87], ["19:44905910:G:A", 0.12]],
            "columns": ["variant", "log2Skew"],
        }
    )
    frame = await client.mpra(gene="APOE")
    assert frame.columns == ["variant", "log2Skew"]
    assert frame["log2Skew"].to_list() == [0.87, 0.12]


async def test_bigquery_empty_result_keeps_its_schema():
    client, _ = make_client(
        {"success": True, "results": [], "columns": ["variant", "pip"]}
    )
    frame = await client.credible_sets(gene="IL7R")
    assert frame.columns == ["variant", "pip"]
    assert frame.height == 0


async def test_truncated_result_raises_instead_of_returning_a_prefix():
    """Silent truncation is the one failure a script cannot detect: the frame is
    well-formed and merely missing rows."""
    client, _ = make_client({"success": True, "results": [[1]], "columns": ["x"], "truncated": True})
    with pytest.raises(GeneticsError, match="truncated"):
        await client.mpra(gene="APOE")


async def test_sql_truncation_names_the_cap_db_api_actually_applied():
    """The old message named no number at all, and the only string that did named 100,000 —
    db-api's relaxed transfer cap, four times what a sandbox execution runs under. The number
    is db-api's constant, so it is quoted from the response, never hardcoded here."""
    client, _ = make_client({
        "success": True,
        "rows": [[1]],
        "columns": ["x"],
        "truncated": True,
        "capped_by_server": True,
        "server_row_cap": 25_000,
    })
    with pytest.raises(GeneticsError, match=r"row cap of 25,000 rows"):
        await client.sql("SELECT 1")


async def test_sql_truncation_degrades_when_the_cap_is_not_reported():
    """An older db-api sends no `max_rows_applied`; the error must still fire and still say
    which ceiling was hit, just without a number."""
    client, _ = make_client({
        "success": True,
        "rows": [[1]],
        "columns": ["x"],
        "truncated": True,
        "capped_by_server": True,
    })
    with pytest.raises(GeneticsError, match=r"db-api's row cap, so rows are missing"):
        await client.sql("SELECT 1")


async def test_a_truncation_the_server_did_not_cause_keeps_the_generic_advice():
    """`truncated` also covers the executor's own LLM-facing slice and the typed endpoints,
    where raising `limit` IS the remedy. Only the server-capped case says otherwise."""
    client, _ = make_client({
        "success": True,
        "rows": [[1]],
        "columns": ["x"],
        "truncated": True,
        "capped_by_server": False,
    })
    with pytest.raises(GeneticsError, match=r"raise `limit`"):
        await client.sql("SELECT 1")


class _FakeDbApiResponse:
    """One db-api /query response, as httpx would hand it back."""

    def __init__(self, payload):
        self.status_code = 200
        self._payload = payload

    def json(self):
        return self._payload


def _client_over_real_executor(db_api_payload) -> tuple[GeneticsClient, ToolExecutor]:
    """A GeneticsClient over a REAL ToolExecutor with only the HTTP call faked.

    The two tests above and their counterparts in test_bigquery_gene_tools.py each cover one
    half — one asserts on `query_database`'s own dict, the other hand-feeds `_check_truncation`
    a dict that already carries the cap keys — so neither crosses `_query_metadata`, which is
    exactly where the keys were being dropped for every typed method.
    """
    executor = ToolExecutor(bigquery_api_url="http://unused.test")
    executor.client.post = AsyncMock(return_value=_FakeDbApiResponse(db_api_payload))
    return GeneticsClient(executor=executor), executor


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("mpra", {"gene": "APOE"}),  # via _bq_gene_payload
        ("hla", {"allele": "DRB1*15:01"}),  # via the HLA branch's own payload
    ],
)
async def test_typed_methods_name_the_cap_end_to_end(method, kwargs):
    """db-api -> query_database -> _query_metadata -> _check_truncation, unfaked in between.

    A typed method reaches db-api through the same `query_database` as `sql()`, so it must get
    the same cap-naming message. It did not: `_query_metadata` copied only `columns` and
    `truncated`, so the cap keys never reached the SDK and the caller was told to "raise
    `limit`" for a ceiling `limit` cannot move.
    """
    client, executor = _client_over_real_executor({
        "columns": ["x"],
        "rows": [[1]],
        "total_rows": 90_000,
        "bytes_processed": 10,
        "truncated": True,
        "max_rows_applied": 25_000,
    })
    try:
        with pytest.raises(GeneticsError, match=r"row cap of 25,000 rows"):
            await getattr(client, method)(**kwargs)
    finally:
        await executor.close()


async def test_typed_methods_drop_the_number_when_db_api_does_not_report_a_cap():
    """A db-api predating `max_rows_applied` reports the cut and no cap. The typed path must
    still raise and still say which ceiling was hit — just without inventing a number."""
    client, executor = _client_over_real_executor({
        "columns": ["x"],
        "rows": [[1]],
        "total_rows": 90_000,
        "truncated": True,
    })
    try:
        with pytest.raises(GeneticsError, match=r"db-api's row cap, so rows are missing"):
            await client.mpra(gene="APOE")
    finally:
        await executor.close()


async def test_bigquery_limit_defaults_to_the_row_ceiling_not_500():
    client, executor = make_client()
    await client.mpra(gene="APOE")
    assert executor.last[2]["limit"] == 100_000
    assert executor.last[2]["with_metadata"] is True


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("credible_sets", {"phenotype": "T1D", "credible_set_id": "cs1", "gene": "IL7R"}),
        ("colocalization", {"phenotype": "T1D", "credible_set_id": "cs1", "variant": "1:1:A:G"}),
        ("exome", {"gene": "IL7R", "resources": ["genebass"]}),
        # results-api serves whole per-phenotype files: there is nothing to threshold on
        ("hla", {"phenotype": "K11_COELIAC", "min_mlogp": 7.3}),
        ("hla", {"phenotype": "K11_COELIAC", "min_info": 0.5}),
        ("hla", {"phenotype": "K11_COELIAC", "limit": 10}),
        # the BigQuery view is queried for one allele, so a gene filter is meaningless
        ("hla", {"allele": "B*27:05", "genes": "HLA-B"}),
        ("gene_annotations", {"region": "1:1-2", "n": 5}),
        ("gene_annotations", {"nearest_to": "1:1:A:G", "exclude_olfactory": True}),
    ],
)
async def test_arguments_the_branch_cannot_honour_are_refused(method, kwargs):
    client, _ = make_client()
    with pytest.raises(GeneticsUsageError):
        await getattr(client, method)(**kwargs)


# ------------------------------------------------------------------------- HLA


async def test_hla_by_phenotype_accepts_one_trait_or_many_and_joins_the_gene_filter():
    client, executor = make_client()

    await client.hla(phenotype="K11_COELIAC", genes=["HLA-B", "HLA-DQB1"])
    assert executor.last[1] == (["K11_COELIAC"],)
    assert executor.last[2]["genes"] == "HLA-B,HLA-DQB1"

    await client.hla(phenotype=["K11_COELIAC", "T1D"], resource="finngen")
    assert executor.last[1] == (["K11_COELIAC", "T1D"],)


async def test_hla_by_allele_uses_the_bigquery_defaults_and_the_row_ceiling():
    """The allele shape goes to hla_associations_v, so it needs `columns` back to label
    the rows and `truncated` to refuse a prefix — both only arrive with_metadata."""
    client, executor = make_client()
    await client.hla(allele="HLA-B*27:05")
    kwargs = executor.last[2]
    assert kwargs["min_mlogp"] == 7.3
    assert kwargs["min_info"] == 0.5
    assert kwargs["max_rows"] == 100_000
    assert kwargs["with_metadata"] is True


async def test_hla_min_info_zero_is_forwarded_not_replaced_by_the_default():
    """0 disables the imputation-quality filter; a falsy-not-None default would eat it."""
    client, executor = make_client()
    await client.hla(allele="B*27:05", min_info=0, min_mlogp=0)
    assert executor.last[2]["min_info"] == 0
    assert executor.last[2]["min_mlogp"] == 0


async def test_hla_by_allele_rows_are_named_by_the_query_columns():
    client, _ = make_client(
        {
            "success": True,
            "results": [["K11_COELIAC", "HLA-DQB1", "DQB1*02:01", 1596.65]],
            "columns": ["phenotype", "gene", "allele", "mlog10p"],
        }
    )
    frame = await client.hla(allele="DQB1*02:01")
    assert frame.columns == ["phenotype", "gene", "allele", "mlog10p"]
    assert frame["mlog10p"].to_list() == [1596.65]


async def test_hla_by_allele_empty_result_keeps_its_schema():
    """A script filtering a no-hit allele must get an empty frame, not ColumnNotFound."""
    client, _ = make_client(
        {"success": True, "results": [], "columns": ["phenotype", "mlog10p"]}
    )
    frame = await client.hla(allele="B*27:05")
    assert frame.columns == ["phenotype", "mlog10p"]
    assert frame.height == 0


async def test_hla_truncated_result_raises_instead_of_returning_a_prefix():
    client, _ = make_client(
        {"success": True, "results": [["T1D"]], "columns": ["phenotype"], "truncated": True}
    )
    with pytest.raises(GeneticsError, match="truncated"):
        await client.hla(allele="B*27:05")


async def test_hla_requires_exactly_one_selector():
    client, _ = make_client()
    for kwargs in ({}, {"phenotype": "T1D", "allele": "B*27:05"}):
        with pytest.raises(GeneticsUsageError):
            await client.hla(**kwargs)


async def test_phenotype_codes_resolve_to_names():
    client, _ = make_client({"success": True, "names": {"I9_CHD": "Coronary heart disease"}})
    frame = await client.lookup_phenotype_names(["I9_CHD"])
    assert frame.columns == ["phenotype", "name"]
    assert frame["name"].to_list() == ["Coronary heart disease"]


async def test_normalize_gene_symbols_puts_unresolved_inputs_in_the_frame():
    client, _ = make_client(
        {
            "success": True,
            "mappings": [{"input": "p53", "approved": "TP53", "matched_on": "alias"}],
            "unresolved": ["NOTAGENE"],
        }
    )
    frame = await client.normalize_gene_symbols(["p53", "NOTAGENE"])
    assert frame["symbol"].to_list() == ["TP53", None]
    assert frame.filter(pl.col("symbol").is_null())["input"].to_list() == ["NOTAGENE"]


def test_configure_cannot_redirect_the_authenticated_client():
    """The client attaches INTERNAL_API_SECRET to every request, so a caller-supplied base
    URL is a one-line credential exfiltration."""
    with pytest.raises(GeneticsUsageError, match="endpoint URLs"):
        genetics.configure(api_base_url="http://attacker.example/api")
    with pytest.raises(TypeError):
        GeneticsClient(api_base_url="http://attacker.example/api")


async def test_resources_are_normalized_to_the_comma_string_the_executor_expects():
    client, executor = make_client()
    await client.mpra(gene="APOE", resources=["a", "b"])
    assert executor.last[2]["resources"] == "a,b"


# --------------------------------------------------------------------- packaging


def test_sync_facade_exposes_every_client_function(monkeypatch):
    for name in genetics._FUNCTIONS:
        assert callable(getattr(genetics, name))
        assert hasattr(GeneticsClient, name)


def test_sync_facade_runs_the_coroutine_and_returns_the_frame(monkeypatch):
    client, executor = make_client()
    monkeypatch.setattr(genetics, "_client", client)
    frame = genetics.credible_sets(gene="IL7R")
    assert isinstance(frame, pl.DataFrame)
    assert executor.last[0] == "get_credible_sets_by_gene"


def test_sdk_imports_without_the_chat_backend():
    """The SDK has to run inside a separate sandbox process, so importing it must not drag
    in the FastAPI app, the LLM service, the MCP server or the SQLite databases."""
    forbidden = [
        "genetics_mcp_server.chat_api",
        "genetics_mcp_server.llm_service",
        "genetics_mcp_server.mcp_server",
        "genetics_mcp_server.db.chat_history_db",
        "fastapi",
        "mcp",
        "anthropic",
    ]
    script = (
        "import sys; import genetics_mcp_server.sdk as genetics; "
        f"leaked=[m for m in {forbidden!r} if m in sys.modules]; "
        "print(','.join(leaked)); "
        "assert genetics.credible_sets is not None"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "", f"SDK import pulled in: {proc.stdout.strip()}"
