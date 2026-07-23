"""Unit tests for BigQuery-backed by-gene tools.

Self-contained: query_bigquery is stubbed so no running BigQuery proxy is needed.
Guards against reading the query result under the wrong key (query_bigquery
returns rows under 'rows', not 'results').
"""

from unittest.mock import AsyncMock

from genetics_mcp_server.tools import ToolExecutor


async def test_get_asm_qtl_by_gene_surfaces_rows():
    executor = ToolExecutor(bigquery_api_url="http://unused.test")
    try:
        fake_rows = [{"chr": "19", "pos": 44908822, "mlog10p": 12.3}]
        executor.query_bigquery = AsyncMock(
            return_value={"success": True, "rows": fake_rows, "columns": ["chr", "pos", "mlog10p"]}
        )

        result = await executor.get_asm_qtl_by_gene("APOE")

        assert result["success"] is True
        assert result["gene"] == "APOE"
        assert result["results"] == fake_rows
        assert result["_download_data"]["results"] == fake_rows
    finally:
        await executor.close()
