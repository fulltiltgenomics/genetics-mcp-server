"""Unit tests for the classical HLA allele association tools.

Self-contained: the results-api call and query_database are stubbed, so no running
services are needed. The point of interest is that the two tools have different
backends for a reason — get_hla_by_phenotype reads per-phenotype files through
results-api, while get_hla_by_allele must go to BigQuery because no single file
spans phenotypes.
"""

from unittest.mock import AsyncMock, Mock

import pytest

from genetics_mcp_server.tools import ToolExecutor
from genetics_mcp_server.tools.definitions import TOOL_DEFINITIONS, get_anthropic_tools


def _api_response(payload, status_code=200):
    resp = Mock()
    resp.status_code = status_code
    resp.json = Mock(return_value=payload)
    resp.text = ""
    return resp


HLA_ROWS = [
    {
        "resource": "finngen",
        "phenotype": "K11_COELIAC",
        "gene": "HLA-DQB1",
        "allele": "DQB1*02:01",
        "mlog10p": 1596.65,
        "beta": 1.6397,
        "info": 0.96906,
    }
]

# get_hla_by_allele goes through db-api's /query, which serializes rows POSITIONALLY
# (`[_serialize_value(v) for v in row.values()]`) with the names in a separate key —
# unlike the results-api rows above, which arrive as dicts.
ALLELE_COLUMNS = [
    "phenotype", "gene", "allele", "mlog10p", "pval", "beta", "se",
    "af", "af_cases", "af_controls", "info",
]
ALLELE_ROWS = [
    ["K11_COELIAC", "HLA-DQB1", "DQB1*02:01", 1596.65, 0.0, 1.6397, 0.0421, 0.30, 0.51, 0.28, 0.96906]
]


class TestGetHlaByPhenotype:
    async def test_surfaces_rows_and_download_url(self):
        executor = ToolExecutor(api_base_url="http://unused.test/api")
        try:
            executor.client.get = AsyncMock(return_value=_api_response(HLA_ROWS))

            result = await executor.get_hla_by_phenotype(["K11_COELIAC"])

            assert result["success"] is True
            assert result["results"] == HLA_ROWS
            assert result["total_count"] == 1
            assert result["truncated"] is False
            assert "phenotypes=K11_COELIAC" in result["_download_url"]
            # the download link must not pin the JSON format the LLM asked for
            assert "format=tsv" in result["_download_url"]

            called = executor.client.get.await_args
            assert called.args[0].endswith("/v1/hla/finngen")
            assert called.kwargs["params"]["phenotypes"] == "K11_COELIAC"
            assert "genes" not in called.kwargs["params"]
        finally:
            await executor.close()

    async def test_genes_filter_is_forwarded(self):
        executor = ToolExecutor(api_base_url="http://unused.test/api")
        try:
            executor.client.get = AsyncMock(return_value=_api_response(HLA_ROWS))

            await executor.get_hla_by_phenotype(["K11_COELIAC"], genes="HLA-B, HLA-DQB1")

            params = executor.client.get.await_args.kwargs["params"]
            assert params["genes"] == "HLA-B,HLA-DQB1"
        finally:
            await executor.close()

    async def test_no_phenotypes_fails_without_calling_the_api(self):
        executor = ToolExecutor(api_base_url="http://unused.test/api")
        try:
            executor.client.get = AsyncMock()

            for empty in ([], ["", "  "]):
                result = await executor.get_hla_by_phenotype(empty)
                assert result["success"] is False
                assert "phenotypes" in result["error"].lower()
            executor.client.get.assert_not_awaited()
        finally:
            await executor.close()

    async def test_resource_is_percent_encoded_into_the_path(self):
        """`resource` reaches the path of a request carrying the internal bearer token,
        and httpx normalises `..`, so an unencoded segment resolves to another endpoint."""
        executor = ToolExecutor(api_base_url="http://unused.test/api")
        try:
            executor.client.get = AsyncMock(return_value=_api_response(HLA_ROWS))

            await executor.get_hla_by_phenotype(["K11_COELIAC"], resource="../../admin/users")

            # the separators are encoded, so the `..` never become path segments httpx
            # could normalise away — the whole thing stays one opaque resource name
            url = executor.client.get.await_args.args[0]
            assert url.endswith("/v1/hla/..%2F..%2Fadmin%2Fusers")
            assert "/admin/users" not in url
        finally:
            await executor.close()

    async def test_api_error_is_reported(self):
        executor = ToolExecutor(api_base_url="http://unused.test/api")
        try:
            executor.client.get = AsyncMock(return_value=_api_response(None, status_code=404))

            result = await executor.get_hla_by_phenotype(["NOT_A_PHENOTYPE"])

            assert result["success"] is False
            assert "404" in result["error"]
        finally:
            await executor.close()


class TestGetHlaByAllele:
    async def test_surfaces_named_rows(self):
        """db-api returns rows POSITIONALLY with the names in a separate `columns` key.
        The model cannot tell mlog10p from beta from af in a bare list, so `results`
        carries dicts built by zipping `columns`."""
        executor = ToolExecutor(bigquery_api_url="http://unused.test")
        try:
            executor.query_database = AsyncMock(
                return_value={
                    "success": True,
                    "rows": ALLELE_ROWS,
                    "columns": ALLELE_COLUMNS,
                }
            )

            result = await executor.get_hla_by_allele("B*27:05")

            assert result["success"] is True
            assert result["allele"] == "B*27:05"
            assert result["count"] == 1
            assert result["results"] == [dict(zip(ALLELE_COLUMNS, ALLELE_ROWS[0]))]
            assert result["results"][0]["mlog10p"] == 1596.65
            assert result["results"][0]["beta"] == 1.6397
            # the download keeps the positional form _convert_to_tsv already handles
            assert result["_download_data"]["columns"] == ALLELE_COLUMNS
            assert result["_download_data"]["rows"] == ALLELE_ROWS
            # '*' and ':' are not safe in a filename
            assert result["_download_data"]["filename"] == "B_27_05_hla.tsv"
        finally:
            await executor.close()

    async def test_download_data_converts_to_tsv_with_a_header(self):
        """The download used to carry a list of lists under `results`; _convert_to_tsv's
        `results` branch does `results[0].keys()`, so it raised AttributeError inside
        _process_download_hints' except and the user silently got no download link."""
        from genetics_mcp_server.llm_service import _convert_to_tsv

        executor = ToolExecutor(bigquery_api_url="http://unused.test")
        try:
            executor.query_database = AsyncMock(
                return_value={
                    "success": True,
                    "columns": ["phenotype", "allele", "mlog10p"],
                    "rows": [["K11_COELIAC", "DQB1*02:01", 1596.65], ["M13_ANKYLOSPON", "B*27:05", 88.2]],
                }
            )

            result = await executor.get_hla_by_allele("B*27:05")

            tsv = _convert_to_tsv(result["_download_data"]).decode()
            assert tsv.splitlines() == [
                "phenotype\tallele\tmlog10p",
                "K11_COELIAC\tDQB1*02:01\t1596.65",
                "M13_ANKYLOSPON\tB*27:05\t88.2",
            ]
        finally:
            await executor.close()

    @pytest.mark.parametrize(
        "bad_result",
        [
            {"success": True, "columns": ["phenotype", "allele"], "rows": [["K11", "B*27:05", 1.0]]},
            {"success": True, "columns": ["phenotype", "allele", "mlog10p"], "rows": [["K11", "B*27:05"]]},
            {"success": True, "columns": [], "rows": [["K11", "B*27:05"]]},
            {"success": True, "columns": ["phenotype", "allele"], "rows": [{"phenotype": "K11", "allele": "B*27:05"}]},
        ],
        ids=["row-too-long", "row-too-short", "no-column-names", "row-is-a-dict"],
    )
    async def test_fails_loudly_on_column_row_mismatch(self, bad_result):
        """Zipping unequal-length sequences truncates silently, which would label genomic
        values with the wrong column names. A dict row is the same class of failure: the
        names would zip against dict KEYS. Refusing is the only safe outcome."""
        executor = ToolExecutor(bigquery_api_url="http://unused.test")
        try:
            executor.query_database = AsyncMock(return_value=bad_result)

            result = await executor.get_hla_by_allele("B*27:05")

            assert result["success"] is False
            assert "column" in result["error"] or "positional" in result["error"]
        finally:
            await executor.close()

    async def test_metadata_is_off_for_the_model_and_on_for_the_sdk(self):
        """`results` now carries the names on every row, but an EMPTY result has no row
        to carry them, so the SDK still needs `columns` to keep a schema — and needs
        `truncated` to refuse a silent prefix. Both stay opt-in so the model's payload is
        not padded with them."""
        executor = ToolExecutor(bigquery_api_url="http://unused.test")
        try:
            query_result = {
                "success": True,
                "rows": [["K11_COELIAC", "DQB1*02:01", 1596.65]],
                "columns": ["phenotype", "allele", "mlog10p"],
                "truncated": True,
            }
            executor.query_database = AsyncMock(return_value=query_result)

            plain = await executor.get_hla_by_allele("B*27:05")
            assert "columns" not in plain
            assert "truncated" not in plain
            # the names reach the model on the rows themselves instead
            assert plain["results"] == [
                {"phenotype": "K11_COELIAC", "allele": "DQB1*02:01", "mlog10p": 1596.65}
            ]

            annotated = await executor.get_hla_by_allele("B*27:05", with_metadata=True)
            assert annotated["columns"] == ["phenotype", "allele", "mlog10p"]
            assert annotated["truncated"] is True
        finally:
            await executor.close()

    async def test_written_gene_prefix_is_normalized_away(self):
        """Users write 'HLA-B*27:05'; the data stores 'B*27:05'."""
        executor = ToolExecutor(bigquery_api_url="http://unused.test")
        try:
            executor.query_database = AsyncMock(
                return_value={"success": True, "rows": [], "columns": []}
            )

            result = await executor.get_hla_by_allele("HLA-B*27:05")

            assert result["success"] is True
            assert result["allele"] == "B*27:05"
            assert "allele = 'B*27:05'" in executor.query_database.await_args.args[0]
        finally:
            await executor.close()

    @pytest.mark.parametrize(
        "bad", ["B27", "'; DROP TABLE hla_associations--", "B*27", "", "* OR 1=1"]
    )
    async def test_malformed_allele_never_reaches_sql(self, bad):
        executor = ToolExecutor(bigquery_api_url="http://unused.test")
        try:
            executor.query_database = AsyncMock()

            result = await executor.get_hla_by_allele(bad)

            assert result["success"] is False
            assert "not an HLA allele name" in result["error"]
            executor.query_database.assert_not_awaited()
        finally:
            await executor.close()

    @pytest.mark.parametrize(
        "bad", ["finngen' OR 1=1--", "finngen; DROP TABLE hla_associations", "fin ngen", 7]
    )
    async def test_unsafe_resource_never_reaches_sql(self, bad):
        """`resource` is a caller-controlled literal, so it goes through the same
        allow-list as every other resource filter rather than being interpolated raw."""
        executor = ToolExecutor(bigquery_api_url="http://unused.test")
        try:
            executor.query_database = AsyncMock()

            result = await executor.get_hla_by_allele("B*27:05", resource=bad)

            assert result["success"] is False
            assert "resource" in result["error"]
            executor.query_database.assert_not_awaited()
        finally:
            await executor.close()

    async def test_legitimate_resource_is_quoted_into_the_filter(self):
        executor = ToolExecutor(bigquery_api_url="http://unused.test")
        try:
            executor.query_database = AsyncMock(
                return_value={"success": True, "rows": [], "columns": []}
            )

            await executor.get_hla_by_allele("B*27:05", resource="finngen_mvp_ukbb")

            assert "resource = 'finngen_mvp_ukbb'" in executor.query_database.await_args.args[0]
        finally:
            await executor.close()

    async def test_max_rows_is_range_checked_before_it_reaches_limit(self):
        """query_database strips the trailing LIMIT today, which makes an unchecked
        `LIMIT {int(max_rows)}` only accidentally safe — it is bounded at the source."""
        executor = ToolExecutor(bigquery_api_url="http://unused.test")
        try:
            executor.query_database = AsyncMock(
                return_value={"success": True, "rows": [], "columns": []}
            )

            await executor.get_hla_by_allele("B*27:05", max_rows=200)
            assert executor.query_database.await_args.args[0].endswith("LIMIT 200")

            for bad in (0, -1, ToolExecutor._MAX_SQL_LIMIT + 1, 1.5, "200"):
                result = await executor.get_hla_by_allele("B*27:05", max_rows=bad)
                assert result["success"] is False
                assert "max_rows" in result["error"]
        finally:
            await executor.close()

    async def test_info_filter_is_applied_by_default_and_can_be_disabled(self):
        executor = ToolExecutor(bigquery_api_url="http://unused.test")
        try:
            executor.query_database = AsyncMock(
                return_value={"success": True, "rows": [], "columns": []}
            )

            await executor.get_hla_by_allele("B*27:05")
            assert "info >= 0.5" in executor.query_database.await_args.args[0]

            await executor.get_hla_by_allele("B*27:05", min_info=0)
            assert "info >=" not in executor.query_database.await_args.args[0]
        finally:
            await executor.close()


# the house spelling both HLA backends must use. results-api's _HLA_HEADER_SCHEMA emits
# these directly; hla_associations_v reaches them by renaming FinnGen's native
# mlogp/sebeta/af_alt/af_alt_cases/af_alt_controls in the view definition
# (genetics-results-db/schemas/hla_associations_v.sql). genetics-results-suite-5wm was a
# divergence nothing tested. What lives in THIS repo can only pin the allele branch, whose
# SQL is written here; the phenotype branch's names are chosen in genetics-results-api and
# are pinned there by tests/test_hla_column_names.py.
HLA_STAT_COLUMNS = ("mlog10p", "se", "af", "af_cases", "af_controls")
HLA_LEGACY_STAT_COLUMNS = ("mlogp", "sebeta", "af_alt", "af_alt_cases", "af_alt_controls")


class TestHlaColumnNamesAgreeAcrossBranches:
    async def test_phenotype_branch_passes_through_the_house_names(self):
        """The executor forwards results-api rows verbatim, so that repo's header schema is
        this tool's output shape unaltered. This pins the pass-through only — the row here
        is authored in this file, so it cannot detect a rename in results-api;
        genetics-results-api/tests/test_hla_column_names.py is what pins those names."""
        executor = ToolExecutor(api_base_url="http://unused.test/api")
        try:
            row = {
                "phenotype": "K11_COELIAC",
                "gene": "HLA-DQB1",
                "allele": "DQB1*02:01",
                "pval": 0.0,
                "mlog10p": 1596.65,
                "beta": 1.6397,
                "se": 0.0421,
                "af": 0.30,
                "af_cases": 0.51,
                "af_controls": 0.28,
                "info": 0.96906,
            }
            executor.client.get = AsyncMock(return_value=_api_response([row]))

            result = await executor.get_hla_by_phenotype(["K11_COELIAC"])

            keys = set(result["results"][0])
            assert set(HLA_STAT_COLUMNS) <= keys
            assert keys.isdisjoint(HLA_LEGACY_STAT_COLUMNS)
        finally:
            await executor.close()

    async def test_allele_branch_selects_the_same_names_from_the_view(self):
        """The view renames on the way out, so the SQL must ask for the renamed columns —
        asking for the table's native names is the 'code old, view new' failure."""
        executor = ToolExecutor(bigquery_api_url="http://unused.test")
        try:
            executor.query_database = AsyncMock(
                return_value={"success": True, "rows": [], "columns": []}
            )

            await executor.get_hla_by_allele("B*27:05")
            sql = executor.query_database.await_args.args[0]

            for column in HLA_STAT_COLUMNS:
                assert f" {column}," in sql or f" {column} " in sql
            assert "ORDER BY mlog10p DESC" in sql
            for legacy in HLA_LEGACY_STAT_COLUMNS:
                assert legacy not in sql
        finally:
            await executor.close()


class TestHlaToolDefinitions:
    def test_both_tools_are_registered(self):
        names = {t["name"] for t in get_anthropic_tools()}
        assert {"get_hla_by_phenotype", "get_hla_by_allele"} <= names

    def test_tool_names_match_executor_methods(self):
        """Dispatch is getattr(executor, tool_name), so a mismatch is a silent 'unknown tool'."""
        for name in ("get_hla_by_phenotype", "get_hla_by_allele"):
            assert callable(getattr(ToolExecutor, name))

    def test_required_parameters(self):
        by_name = {t["name"]: t for t in TOOL_DEFINITIONS}
        assert by_name["get_hla_by_phenotype"]["parameters"]["phenotypes"]["required"]
        assert by_name["get_hla_by_allele"]["parameters"]["allele"]["required"]
        # both are api-profile tools
        assert by_name["get_hla_by_phenotype"]["category"] == "api"
        assert by_name["get_hla_by_allele"]["category"] == "api"
