"""Unit tests for BigQuery-backed by-gene tools.

Self-contained: query_database is stubbed so no running BigQuery proxy is needed.
Guards against reading the query result under the wrong key (query_database
returns rows under 'rows', not 'results').
"""

import contextlib
import re
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
        # the model gets NAMED rows: bare positional lists are uninterpretable to it
        assert result["results"] == [{"chr": "19", "pos": 44908822, "mlog10p": 12.3}]
        # the download keeps the positional form _convert_to_tsv already handles
        assert result["_download_data"]["columns"] == fake_columns
        assert result["_download_data"]["rows"] == fake_rows
        # `columns` at the top level is SDK-only, so the model's payload is not padded
        assert "columns" not in result
        with_meta = await executor.get_asm_qtl_by_gene("APOE", with_metadata=True)
        assert with_meta["columns"] == fake_columns
        assert with_meta["truncated"] is False
    finally:
        await executor.close()


_BY_GENE_FILENAMES = {
    "get_asm_qtl_by_gene": "APOE_asm_qtl.tsv",
    "get_open_chromatin_by_gene": "APOE_open_chromatin.tsv",
    "get_variant_effect_by_gene": "APOE_variant_effect.tsv",
    "get_mpra_by_gene": "APOE_mpra.tsv",
    "get_mpra_pip_concordance_by_gene": "APOE_mpra_pip_concordance.tsv",
}


@pytest.mark.parametrize("method", sorted(_BY_GENE_FILENAMES))
async def test_by_gene_download_data_converts_to_tsv_with_a_header(method):
    """The download used to carry a list of lists under `results`; _convert_to_tsv's
    `results` branch does `results[0].keys()`, so it raised AttributeError inside
    _process_download_hints' except and the user silently got no download link."""
    from genetics_mcp_server.llm_service import _convert_to_tsv

    executor = ToolExecutor(bigquery_api_url="http://unused.test")
    try:
        executor.query_database = AsyncMock(
            return_value={
                "success": True,
                "columns": ["chr", "pos", "mlog10p"],
                "rows": [["19", 44908822, 12.3], ["19", 44909000, 8.1]],
            }
        )
        result = await getattr(executor, method)("APOE")

        download = result["_download_data"]
        assert download["filename"] == _BY_GENE_FILENAMES[method]
        tsv = _convert_to_tsv(download).decode()
        assert tsv.splitlines() == [
            "chr\tpos\tmlog10p",
            "19\t44908822\t12.3",
            "19\t44909000\t8.1",
        ]
    finally:
        await executor.close()


@pytest.mark.parametrize("method", sorted(_BY_GENE_FILENAMES))
async def test_by_gene_empty_result_is_sane(method):
    async with _stubbed_executor() as executor:
        result = await getattr(executor, method)("APOE")

        assert result["success"] is True
        assert result["results"] == []
        assert result["_download_data"]["rows"] == []
        assert result["_download_data"]["columns"] == []


@pytest.mark.parametrize(
    "bad_result",
    [
        {"success": True, "columns": ["chr", "pos"], "rows": [["19", 1, 2.0]]},
        {"success": True, "columns": ["chr", "pos", "mlog10p"], "rows": [["19", 1]]},
        {"success": True, "columns": [], "rows": [["19", 1]]},
        {"success": True, "columns": ["chr", "pos"], "rows": [{"chr": "19", "pos": 1}]},
    ],
    ids=["row-too-long", "row-too-short", "no-column-names", "row-is-a-dict"],
)
async def test_by_gene_fails_loudly_on_column_row_mismatch(bad_result):
    """Zipping unequal-length sequences truncates silently, which would label genomic
    values with the wrong column names. Refusing is the only safe outcome."""
    executor = ToolExecutor(bigquery_api_url="http://unused.test")
    try:
        executor.query_database = AsyncMock(return_value=bad_result)
        result = await executor.get_mpra_by_gene("APOE")

        assert result["success"] is False
        assert "column" in result["error"] or "positional" in result["error"]
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


# db-api owns the dataset identity: it rewrites bare view names to
# `{PROJECT_ID}.{DATASET_ID}.{view}` before the allow-list check, so one binary serves
# dev and production. Its rewrite regex is `\b(FROM|JOIN)(\s+)<name>\b`, which does NOT
# match a backtick after the whitespace — so a name that is unqualified but backticked
# sails past the rewrite untouched and then fails at BigQuery as a missing table. The two
# assertions below are one guard: qualifying is wrong, and so is backticking.
_TABLE_REF = re.compile(r"\b(?:FROM|JOIN)\s+(\S+)", re.IGNORECASE)

# a CTE is referenced by FROM/JOIN exactly like a view, so it has to be excluded from the
# check — but excluding it by hard-coded name means a second CTE either fails the check
# spuriously or, once added to the skip list, silently stops it noticing anything.
_CTE_DECL = re.compile(r"[()]|[A-Za-z_]\w*\s+AS\s*\(", re.IGNORECASE)


def _cte_names(sql: str) -> set[str]:
    """Names declared by the SQL's WITH clause, if it has one."""
    if not re.match(r"\s*WITH\b", sql, re.IGNORECASE):
        return set()
    names: set[str] = set()
    depth = 0
    # only a declaration at paren depth 0 is a CTE; `<name> AS (` deeper in is a subquery
    for match in _CTE_DECL.finditer(sql):
        text = match.group()
        if text == "(":
            depth += 1
        elif text == ")":
            depth -= 1
        else:
            if depth == 0:
                names.add(text.split()[0])
            depth += 1  # the declaration match consumed its own opening paren
    return names


def _emits_via_query(method: str, *args, **kwargs):
    """Adapter for the methods that reach BigQuery: returns the SQL they sent."""

    async def emit(executor):
        await getattr(executor, method)(*args, **kwargs)
        assert executor.query_database.await_args is not None, (
            f"{method} rejected its arguments before building SQL"
        )
        return executor.query_database.await_args.args[0]

    return emit


async def _emits_gene_window_cte(executor):
    return executor._gene_window_cte("'APOE'")


# every method that puts a view name into SQL, each carrying its own arguments — the
# earlier positional `(gene)` call shape could not reach get_hla_by_allele, which takes
# an allele, and so left one of the rewritten `FROM` sites uncovered
_SQL_EMITTERS = [
    ("_gene_window_cte", _emits_gene_window_cte),
    ("get_asm_qtl_by_gene", _emits_via_query("get_asm_qtl_by_gene", "APOE")),
    ("get_open_chromatin_by_gene", _emits_via_query("get_open_chromatin_by_gene", "APOE")),
    ("get_variant_effect_by_gene", _emits_via_query("get_variant_effect_by_gene", "APOE")),
    ("get_mpra_by_gene", _emits_via_query("get_mpra_by_gene", "APOE")),
    (
        "get_mpra_pip_concordance_by_gene",
        _emits_via_query("get_mpra_pip_concordance_by_gene", "APOE"),
    ),
    ("get_hla_by_allele", _emits_via_query("get_hla_by_allele", "DRB1*15:01")),
]


@pytest.mark.parametrize(
    "method,emit", _SQL_EMITTERS, ids=[name for name, _ in _SQL_EMITTERS]
)
async def test_generated_sql_names_views_bare_and_unbackticked(method, emit):
    async with _stubbed_executor() as executor:
        sql = await emit(executor)

        ctes = _cte_names(sql)
        refs = [r for r in _TABLE_REF.findall(sql) if r not in ctes]
        assert refs, f"{method} generated no table reference to check: {sql}"
        for ref in refs:
            assert "`" not in ref, (
                f"{method} backticked {ref!r}; db-api's rewrite skips it and BigQuery "
                f"then cannot resolve the view"
            )
            assert "." not in ref, (
                f"{method} qualified {ref!r}; the dataset name must come from db-api's "
                f"DATASET_ID, not from the emitted SQL"
            )
