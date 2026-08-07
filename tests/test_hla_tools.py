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
    async def test_surfaces_rows(self):
        executor = ToolExecutor(bigquery_api_url="http://unused.test")
        try:
            executor.query_database = AsyncMock(
                return_value={"success": True, "rows": HLA_ROWS, "columns": []}
            )

            result = await executor.get_hla_by_allele("B*27:05")

            assert result["success"] is True
            assert result["allele"] == "B*27:05"
            assert result["results"] == HLA_ROWS
            # '*' and ':' are not safe in a filename
            assert result["_download_data"]["filename"] == "B_27_05_hla.tsv"
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
