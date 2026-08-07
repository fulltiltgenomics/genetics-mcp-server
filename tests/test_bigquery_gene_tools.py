"""Unit tests for BigQuery-backed by-gene tools.

Self-contained: query_database is stubbed so no running BigQuery proxy is needed.
Guards against reading the query result under the wrong key (query_database
returns rows under 'rows', not 'results').
"""

import contextlib
from unittest.mock import AsyncMock
from urllib.parse import urlsplit

import pytest

from genetics_mcp_server.tools import ToolExecutor


async def test_get_asm_qtl_by_gene_surfaces_rows():
    executor = ToolExecutor(bigquery_api_url="http://unused.test")
    try:
        # db-api returns rows POSITIONALLY — a list per row, names in `columns`
        # (genetics-results-db api/main.py builds `[_serialize_value(v) for v in row.values()]`).
        # Mocking dicts here hid a real defect: the SDK fed these to pl.from_dicts, which
        # does not raise on lists but silently transposes and stringifies them.
        fake_columns = ["chr", "pos", "mlog10p"]
        fake_rows = [["19", 44908822, 12.3]]
        executor.query_database = AsyncMock(
            return_value={"success": True, "rows": fake_rows, "columns": fake_columns}
        )

        result = await executor.get_asm_qtl_by_gene("APOE")

        assert result["success"] is True
        assert result["gene"] == "APOE"
        assert result["results"] == fake_rows
        assert result["_download_data"]["results"] == fake_rows
        # column names reach the SDK only when asked for, so the MCP payload is unchanged
        assert "columns" not in result
        with_meta = await executor.get_asm_qtl_by_gene("APOE", with_metadata=True)
        assert with_meta["columns"] == fake_columns
        assert with_meta["truncated"] is False
    finally:
        await executor.close()


# These five methods build SQL by interpolating caller-supplied values, and the db-api's
# /query endpoint takes SQL as a string with no parameter-binding channel. Under the MCP
# tool surface the values arrived through a typed schema; under the SDK they are composed
# by a script, so the validation below is the only thing between a caller string and
# BigQuery.

_SQL_BUILDING_METHODS = [
    "get_asm_qtl_by_gene",
    "get_open_chromatin_by_gene",
    "get_variant_effect_by_gene",
    "get_mpra_by_gene",
    "get_mpra_pip_concordance_by_gene",
]

_RESOURCE_FILTER_METHODS = [
    "get_asm_qtl_by_gene",
    "get_open_chromatin_by_gene",
    "get_variant_effect_by_gene",
    "get_mpra_by_gene",
]

_INJECTION_PAYLOADS = [
    "APOE' OR '1'='1",
    "APOE'); DROP TABLE x; --",
    "APOE' UNION ALL SELECT * FROM `genetics_results.credible_sets_v` --",
    "APOE' AND FALSE --",
]


@pytest.mark.parametrize(
    ("method", "args", "prefix"),
    [
        ("get_mpra_by_variant", ("../../admin/users",), "/api/v1/mpra/variant/"),
        ("get_open_chromatin_by_variant", ("x?token=leak#frag",), "/api/v1/open_chromatin/variant/"),
        ("get_variant_effect_by_variant", ("../..",), "/api/v1/variant_effect/variant/"),
        ("get_exome_results_by_gene", ("../v1/resources",), "/api/v1/exome_results_by_gene/"),
        ("get_gene_expression", ("a/b",), "/api/v1/expression_by_gene/"),
    ],
)
async def test_url_path_segments_cannot_escape_their_endpoint(method, args, prefix):
    """Caller-controlled values go into results-api paths on a client that carries the
    internal bearer token, and httpx normalises `..`. Unencoded, `mpra(variant=
    '../../admin/users')` reaches an arbitrary endpoint with the secret attached."""
    from unittest.mock import patch

    executor = ToolExecutor(api_base_url="http://api.internal:2000/api")
    try:
        captured = {}

        async def fake_get(url, **kwargs):
            captured["url"] = url
            raise RuntimeError("stop")

        with patch.object(executor.client, "get", side_effect=fake_get):
            with contextlib.suppress(RuntimeError):
                await getattr(executor, method)(*args)

        path = urlsplit(captured["url"]).path
        assert path.startswith(prefix)
        assert "?" not in captured["url"] and "#" not in captured["url"]
        # the whole value must survive as ONE segment, and never as a relative one
        tail = path[len(prefix) :]
        assert "/" not in tail
        assert tail not in (".", "..")
    finally:
        await executor.close()


@contextlib.asynccontextmanager
async def _stubbed_executor():
    executor = ToolExecutor(bigquery_api_url="http://unused.test")
    executor.query_database = AsyncMock(
        return_value={"success": True, "rows": [], "columns": []}
    )
    try:
        yield executor
    finally:
        await executor.close()


@pytest.mark.parametrize("method", _SQL_BUILDING_METHODS)
@pytest.mark.parametrize("payload", _INJECTION_PAYLOADS)
async def test_sql_builders_reject_injected_gene(method, payload):
    async with _stubbed_executor() as executor:
        result = await getattr(executor, method)(payload)

        assert result["success"] is False
        assert "invalid gene" in result["error"]
        executor.query_database.assert_not_awaited()


@pytest.mark.parametrize("method", _RESOURCE_FILTER_METHODS)
async def test_sql_builders_reject_injected_resources(method):
    async with _stubbed_executor() as executor:
        result = await getattr(executor, method)("APOE", resources="ok,bad' OR TRUE --")

        assert result["success"] is False
        assert "invalid resources" in result["error"]
        executor.query_database.assert_not_awaited()


async def test_pip_concordance_rejects_injected_resource_and_out_of_range_pip():
    async with _stubbed_executor() as executor:
        bad_resource = await executor.get_mpra_pip_concordance_by_gene(
            "APOE", resource="finngen' OR TRUE --"
        )
        assert bad_resource["success"] is False
        assert "invalid resource" in bad_resource["error"]

        bad_pip = await executor.get_mpra_pip_concordance_by_gene("APOE", min_pip=float("nan"))
        assert bad_pip["success"] is False

        executor.query_database.assert_not_awaited()


@pytest.mark.parametrize("method", _SQL_BUILDING_METHODS)
async def test_sql_builders_reject_non_numeric_window(method):
    async with _stubbed_executor() as executor:
        result = await getattr(executor, method)("APOE", window="500000 OR TRUE")

        assert result["success"] is False
        executor.query_database.assert_not_awaited()


@pytest.mark.parametrize("method", _SQL_BUILDING_METHODS)
async def test_sql_builders_quote_a_legitimate_gene_symbol(method):
    async with _stubbed_executor() as executor:
        await getattr(executor, method)("HLA-DRB1")

        sql = executor.query_database.await_args.args[0]
        assert "symbol = 'HLA-DRB1'" in sql


async def test_sql_builders_render_resources_as_a_quoted_in_list():
    async with _stubbed_executor() as executor:
        await executor.get_mpra_by_gene("APOE", resources="mpra_a,mpra_b")

        sql = executor.query_database.await_args.args[0]
        assert "a.resource IN ('mpra_a', 'mpra_b')" in sql


async def test_limit_is_honoured_by_the_statement_and_the_row_cap():
    async with _stubbed_executor() as executor:
        await executor.get_mpra_by_gene("APOE", limit=5000)

        sql = executor.query_database.await_args.args[0]
        assert sql.endswith("LIMIT 5000")
        assert executor.query_database.await_args.kwargs["max_rows"] == 5000
