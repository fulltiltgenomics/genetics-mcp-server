"""Unit tests for the rare-CNV tools (Collins et al. 2022 dosage sensitivity map).

Self-contained: query_database is stubbed, so no running services are needed. Both tools
build their own SQL, so the SQL TEXT is the contract under test — the executor is the only
place the paper's significance rule and the "tested, no estimate" exclusion are written
down, and a silent change to either returns a different answer with no failure anywhere.
"""

from unittest.mock import AsyncMock

import pytest

from genetics_mcp_server.tools import ToolExecutor
from genetics_mcp_server.tools.definitions import TOOL_DEFINITIONS, get_anthropic_tools

# db-api serializes rows POSITIONALLY with the names in a separate `columns` key.
DOSAGE_COLUMNS = [
    "symbol", "symbol_gencode_v19", "ensembl_gene_id",
    "phaplo", "ptriplo", "haploinsufficient", "triplosensitive",
]
DOSAGE_ROWS = [["SHANK3", "SHANK3", "ENSG00000251322", 0.9917, 0.6021, True, False]]

RCNV_COLUMNS = [
    "phenotype", "trait_name", "cnv_type", "symbol", "ensembl_gene_id",
    "n_nominal_cohorts", "beta", "mlog10p", "mlog10_fdr_q",
]
RCNV_ROWS = [["HP0012759", "Neurodevelopmental abnormality", "DEL", "SHANK3",
              "ENSG00000251322", 4, 3.699, 12.4, 9.1]]


def _executor(result=None):
    executor = ToolExecutor(bigquery_api_url="http://unused.test")
    executor.query_database = AsyncMock(
        return_value=result or {"success": True, "rows": [], "columns": []}
    )
    return executor


def _sql(executor):
    return executor.query_database.await_args.args[0]


class TestGetDosageSensitivity:
    async def test_surfaces_named_rows_and_a_download(self):
        executor = _executor({"success": True, "rows": DOSAGE_ROWS, "columns": DOSAGE_COLUMNS})
        try:
            result = await executor.get_dosage_sensitivity(["SHANK3"])

            assert result["success"] is True
            assert result["genes"] == ["SHANK3"]
            assert result["count"] == 1
            assert result["results"] == [dict(zip(DOSAGE_COLUMNS, DOSAGE_ROWS[0]))]
            assert result["results"][0]["phaplo"] == 0.9917
            # the download keeps the positional form _convert_to_tsv handles
            assert result["_download_data"]["columns"] == DOSAGE_COLUMNS
            assert result["_download_data"]["rows"] == DOSAGE_ROWS
        finally:
            await executor.close()

    async def test_matches_all_three_identifier_columns_case_insensitively(self):
        """The scores were published on GENCODE v19, so a gene renamed since is reachable
        only under its old spelling; the caller should not have to know which it holds."""
        executor = _executor()
        try:
            await executor.get_dosage_sensitivity(["shank3", "ENSG00000251322"])
            sql = _sql(executor)

            assert "UPPER(symbol) IN ('SHANK3', 'ENSG00000251322')" in sql
            assert "UPPER(symbol_gencode_v19) IN ('SHANK3', 'ENSG00000251322')" in sql
            assert "UPPER(ensembl_gene_id) IN ('SHANK3', 'ENSG00000251322')" in sql
            assert "FROM dosage_sensitivity_v" in sql
            assert sql.endswith("ORDER BY phaplo DESC LIMIT 1000")
        finally:
            await executor.close()

    @pytest.mark.parametrize(
        "genes",
        [[], ["", "  "], "SHANK3"],
        ids=["empty-list", "blank-strings", "bare-string-not-a-list"],
    )
    async def test_no_usable_genes_never_reaches_sql(self, genes):
        executor = _executor()
        try:
            result = await executor.get_dosage_sensitivity(genes)

            assert result["success"] is False
            executor.query_database.assert_not_awaited()
        finally:
            await executor.close()

    async def test_non_string_gene_is_rejected_not_silently_dropped(self):
        """A non-string member used to be filtered out by the `isinstance` guard on the
        list comprehension, so ["APOE", 123] silently became ["APOE"] and ["SHANK3", None]
        with no valid entries reported the generic 'No genes provided' — hiding which
        argument was actually wrong."""
        executor = _executor()
        try:
            result = await executor.get_dosage_sensitivity(["SHANK3", 123])

            assert result["success"] is False
            assert "123" in result["error"]
            executor.query_database.assert_not_awaited()
        finally:
            await executor.close()

    @pytest.mark.parametrize(
        "bad", ["SHANK3'; DROP TABLE dosage_sensitivity--", "SHANK3 OR 1=1", "a" * 129]
    )
    async def test_unsafe_gene_never_reaches_sql(self, bad):
        executor = _executor()
        try:
            result = await executor.get_dosage_sensitivity([bad])

            assert result["success"] is False
            assert "genes" in result["error"]
            executor.query_database.assert_not_awaited()
        finally:
            await executor.close()

    async def test_max_rows_is_range_checked_before_it_reaches_limit(self):
        executor = _executor()
        try:
            await executor.get_dosage_sensitivity(["SHANK3"], max_rows=25)
            assert _sql(executor).endswith("LIMIT 25")

            for bad in (0, -1, ToolExecutor._MAX_SQL_LIMIT + 1, 1.5, "25"):
                result = await executor.get_dosage_sensitivity(["SHANK3"], max_rows=bad)
                assert result["success"] is False
                assert "max_rows" in result["error"]
        finally:
            await executor.close()

    async def test_metadata_is_off_by_default_and_carries_the_empty_schema(self):
        """An empty result has no row to carry the column names, and the SDK builds its
        empty frame from `columns`."""
        executor = _executor({"success": True, "rows": [], "columns": DOSAGE_COLUMNS})
        try:
            plain = await executor.get_dosage_sensitivity(["NOT_A_GENE"])
            assert "columns" not in plain

            annotated = await executor.get_dosage_sensitivity(["NOT_A_GENE"], with_metadata=True)
            assert annotated["columns"] == DOSAGE_COLUMNS
            assert annotated["truncated"] is False
        finally:
            await executor.close()


class TestGetRcnvAssociations:
    async def test_surfaces_named_rows_and_echoes_the_filters(self):
        executor = _executor({"success": True, "rows": RCNV_ROWS, "columns": RCNV_COLUMNS})
        try:
            result = await executor.get_rcnv_associations(gene="SHANK3", cnv_type="DEL")

            assert result["success"] is True
            assert result["gene"] == "SHANK3"
            assert result["cnv_type"] == "DEL"
            assert result["count"] == 1
            assert result["results"] == [dict(zip(RCNV_COLUMNS, RCNV_ROWS[0]))]
            assert result["_download_data"]["filename"] == "SHANK3_rcnv.tsv"
        finally:
            await executor.close()

    async def test_at_least_one_selector_is_required(self):
        executor = _executor()
        try:
            result = await executor.get_rcnv_associations()

            assert result["success"] is False
            assert "gene=" in result["error"] and "phenotype=" in result["error"]
            executor.query_database.assert_not_awaited()
        finally:
            await executor.close()

    async def test_gene_only_query_is_the_phewas_shape(self):
        executor = _executor()
        try:
            await executor.get_rcnv_associations(gene="nrxn1")
            sql = _sql(executor)

            assert (
                "WHERE (UPPER(r.symbol) = 'NRXN1' "
                "OR UPPER(r.symbol_gencode_v19) = 'NRXN1' "
                "OR UPPER(r.ensembl_gene_id) = 'NRXN1') "
                "AND r.beta IS NOT NULL "
                "ORDER BY r.mlog10p DESC LIMIT 200"
            ) in sql
            assert "FROM rcnv_gene_associations_v r" in sql
        finally:
            await executor.close()

    async def test_the_phenotype_name_is_joined_on_and_the_join_is_left(self):
        """A phenotype group with no phenotypes_v row must still return its associations —
        an inner join would silently drop them and look like 'no data for this gene'."""
        executor = _executor()
        try:
            await executor.get_rcnv_associations(gene="SHANK3")
            sql = _sql(executor)

            assert (
                "LEFT JOIN phenotypes_v p "
                "ON p.dataset = r.dataset AND p.trait_original = r.phenotype"
            ) in sql
            assert "p.trait_name" in sql
        finally:
            await executor.close()

    @pytest.mark.parametrize(
        ("given", "expected"),
        [
            ("HP:0012759", "HP0012759"),
            ("HP0012759", "HP0012759"),
            ("hp:0000118", "HP0000118"),
            ("  HP:0001249  ", "HP0001249"),
        ],
    )
    async def test_hpo_ids_are_accepted_in_either_spelling(self, given, expected):
        """The data drops the colon; both spellings are in circulation."""
        executor = _executor()
        try:
            await executor.get_rcnv_associations(phenotype=given)
            assert f"r.phenotype = '{expected}'" in _sql(executor)
            assert "LIKE" not in _sql(executor)
        finally:
            await executor.close()

    async def test_unknown_is_an_exact_group_not_a_name_search(self):
        executor = _executor()
        try:
            await executor.get_rcnv_associations(phenotype="unknown")
            assert "r.phenotype = 'UNKNOWN'" in _sql(executor)
        finally:
            await executor.close()

    async def test_a_non_hpo_phenotype_becomes_a_name_substring_match(self):
        """search_phenotypes reads the results-api index, which does not cover this
        BigQuery-only dataset, so the name has to be resolved in this query."""
        executor = _executor()
        try:
            await executor.get_rcnv_associations(phenotype="Intellectual disability")
            assert "UPPER(p.trait_name) LIKE '%INTELLECTUAL DISABILITY%'" in _sql(executor)
        finally:
            await executor.close()

    @pytest.mark.parametrize(
        "bad",
        ["intellectual'; DROP TABLE rcnv_gene_associations--", "100% penetrant", "", "  ", 7],
        ids=["quote-break-out", "percent-wildcard", "empty", "blank", "not-a-string"],
    )
    async def test_unsafe_phenotype_never_reaches_sql(self, bad):
        executor = _executor()
        try:
            result = await executor.get_rcnv_associations(phenotype=bad)

            assert result["success"] is False
            assert "phenotype" in result["error"]
            executor.query_database.assert_not_awaited()
        finally:
            await executor.close()

    @pytest.mark.parametrize("bad", ["NRXN1'; DROP TABLE x--", "NRXN1 OR 1=1", 7])
    async def test_unsafe_gene_never_reaches_sql(self, bad):
        executor = _executor()
        try:
            result = await executor.get_rcnv_associations(gene=bad)

            assert result["success"] is False
            assert "gene" in result["error"]
            executor.query_database.assert_not_awaited()
        finally:
            await executor.close()

    @pytest.mark.parametrize(("given", "expected"), [("DEL", "DEL"), ("dup", "DUP")])
    async def test_cnv_type_is_normalized_to_the_two_stored_values(self, given, expected):
        executor = _executor()
        try:
            await executor.get_rcnv_associations(gene="SHANK3", cnv_type=given)
            assert f"r.cnv_type = '{expected}'" in _sql(executor)
        finally:
            await executor.close()

    @pytest.mark.parametrize("bad", ["INV", "DELETION", "'; DROP TABLE x--", 7])
    async def test_a_cnv_type_outside_the_pair_is_refused(self, bad):
        executor = _executor()
        try:
            result = await executor.get_rcnv_associations(gene="SHANK3", cnv_type=bad)

            assert result["success"] is False
            assert "cnv_type" in result["error"]
            executor.query_database.assert_not_awaited()
        finally:
            await executor.close()

    async def test_null_beta_rows_are_excluded_by_default_and_can_be_kept(self):
        """65% of the view is 'tested, no estimate' — every gene appears for every
        phenotype and CNV type whether or not a qualifying CNV was ever observed."""
        executor = _executor()
        try:
            await executor.get_rcnv_associations(gene="SHANK3")
            assert "r.beta IS NOT NULL" in _sql(executor)

            await executor.get_rcnv_associations(gene="SHANK3", include_no_estimate=True)
            where = _sql(executor).split(" WHERE ", 1)[1].split(" ORDER BY ", 1)[0]
            assert "IS NOT NULL" not in where
            # n_nominal_cohorts is NOT a proxy for the same filter (the null rows are not
            # the n_nominal_cohorts = 0 rows), so it must not appear as one
            assert "n_nominal_cohorts" not in where
        finally:
            await executor.close()

    async def test_significant_only_is_the_papers_full_rule(self):
        """Both tiers, each gated by the secondary-evidence requirement. Measured on dev:
        the FDR tier alone is 6,450 associations, exome-wide alone is 3,864 (a strict subset
        of the FDR tier); gating on FDR AND n_nominal_cohorts>=2 — an earlier draft's rule —
        gives 5,644, still short of the paper's 5,680 over 739 genes, because it drops the
        mlog10p_secondary alternative this query's OR also accepts."""
        executor = _executor()
        try:
            await executor.get_rcnv_associations(gene="SHANK3")
            assert "LOG10" not in _sql(executor)

            await executor.get_rcnv_associations(phenotype="HP0001249", significant_only=True)
            assert (
                "((r.mlog10_fdr_q > -LOG10(0.01) OR r.mlog10p > -LOG10(2.90e-6)) "
                "AND (r.n_nominal_cohorts >= 2 OR r.mlog10p_secondary > -LOG10(0.05)))"
            ) in _sql(executor)
        finally:
            await executor.close()

    async def test_max_fdr_q_is_translated_to_the_stored_minus_log10_scale(self):
        """The column stores -log10(q), so a MAXIMUM q is a MINIMUM on the column; getting
        the direction wrong returns exactly the rows the caller asked to exclude."""
        executor = _executor()
        try:
            await executor.get_rcnv_associations(gene="SHANK3", max_fdr_q=0.01)
            assert "r.mlog10_fdr_q >= -LOG10(0.01)" in _sql(executor)
        finally:
            await executor.close()

    @pytest.mark.parametrize("bad", [0, -0.1, 1.5, "0.01"])
    async def test_an_out_of_range_max_fdr_q_never_reaches_sql(self, bad):
        executor = _executor()
        try:
            result = await executor.get_rcnv_associations(gene="SHANK3", max_fdr_q=bad)

            assert result["success"] is False
            assert "max_fdr_q" in result["error"]
            executor.query_database.assert_not_awaited()
        finally:
            await executor.close()

    async def test_min_mlog10p_is_a_range_checked_float(self):
        executor = _executor()
        try:
            await executor.get_rcnv_associations(gene="SHANK3", min_mlog10p=5.3)
            assert "r.mlog10p >= 5.3" in _sql(executor)

            for bad in (-1, "5.3", float("inf")):
                result = await executor.get_rcnv_associations(gene="SHANK3", min_mlog10p=bad)
                assert result["success"] is False
                assert "min_mlog10p" in result["error"]
        finally:
            await executor.close()

    async def test_limit_is_range_checked_before_it_reaches_limit(self):
        executor = _executor()
        try:
            await executor.get_rcnv_associations(gene="SHANK3", limit=50)
            assert _sql(executor).endswith("LIMIT 50")

            for bad in (0, -1, ToolExecutor._MAX_SQL_LIMIT + 1, 1.5, "50"):
                result = await executor.get_rcnv_associations(gene="SHANK3", limit=bad)
                assert result["success"] is False
                assert "limit" in result["error"]
        finally:
            await executor.close()

    async def test_every_filter_at_once_composes_with_and(self):
        executor = _executor()
        try:
            await executor.get_rcnv_associations(
                gene="SHANK3",
                phenotype="HP:0012759",
                cnv_type="DEL",
                min_mlog10p=3,
                max_fdr_q=0.05,
                significant_only=True,
                limit=10,
            )
            sql = _sql(executor)

            where = sql.split(" WHERE ", 1)[1].split(" ORDER BY ", 1)[0]
            assert where == (
                "(UPPER(r.symbol) = 'SHANK3' OR UPPER(r.symbol_gencode_v19) = 'SHANK3' "
                "OR UPPER(r.ensembl_gene_id) = 'SHANK3') "
                "AND r.phenotype = 'HP0012759' "
                "AND r.cnv_type = 'DEL' "
                "AND r.beta IS NOT NULL "
                "AND r.mlog10p >= 3.0 "
                "AND r.mlog10_fdr_q >= -LOG10(0.05) "
                "AND ((r.mlog10_fdr_q > -LOG10(0.01) OR r.mlog10p > -LOG10(2.90e-6)) "
                "AND (r.n_nominal_cohorts >= 2 OR r.mlog10p_secondary > -LOG10(0.05)))"
            )
        finally:
            await executor.close()

    async def test_a_phenotype_only_query_names_the_result_file_generically(self):
        executor = _executor({"success": True, "rows": [], "columns": RCNV_COLUMNS})
        try:
            result = await executor.get_rcnv_associations(phenotype="HP0012759")
            assert result["_download_data"]["filename"] == "rcnv_associations.tsv"
        finally:
            await executor.close()

    async def test_a_column_row_mismatch_fails_loudly(self):
        """Zipping unequal-length sequences truncates silently, which would label an odds
        ratio with a p-value's name."""
        executor = _executor(
            {"success": True, "columns": ["phenotype", "beta"], "rows": [["HP0012759"]]}
        )
        try:
            result = await executor.get_rcnv_associations(gene="SHANK3")
            assert result["success"] is False
            assert "column names" in result["error"]
        finally:
            await executor.close()


class TestRcnvToolDefinitions:
    def test_both_tools_are_registered(self):
        names = {t["name"] for t in get_anthropic_tools()}
        assert {"get_dosage_sensitivity", "get_rcnv_associations"} <= names

    def test_tool_names_match_executor_methods(self):
        """Dispatch is getattr(executor, tool_name), so a mismatch is a silent 'unknown tool'."""
        for name in ("get_dosage_sensitivity", "get_rcnv_associations"):
            assert callable(getattr(ToolExecutor, name))

    def test_parameter_shape(self):
        by_name = {t["name"]: t for t in TOOL_DEFINITIONS}

        dosage = by_name["get_dosage_sensitivity"]
        assert dosage["category"] == "api"
        assert dosage["sdk_replaceable"] is True
        assert dosage["parameters"]["genes"]["required"]
        assert dosage["parameters"]["genes"]["type"] == "array"

        rcnv = by_name["get_rcnv_associations"]
        assert rcnv["category"] == "api"
        assert rcnv["sdk_replaceable"] is True
        # neither selector is `required`: the tool takes either one, and the pair is
        # enforced in the executor because a schema cannot express "at least one of"
        assert not any(p.get("required") for p in rcnv["parameters"].values())
        assert set(rcnv["parameters"]) == {
            "gene", "phenotype", "cnv_type", "min_mlog10p", "max_fdr_q",
            "significant_only", "include_no_estimate", "limit",
        }

    def test_the_no_estimate_rows_are_described_where_the_model_reads(self):
        """The 65% NULL-beta share is the single most misleading thing about this view, and
        the default exclusion is invisible unless the description says so."""
        by_name = {t["name"]: t for t in TOOL_DEFINITIONS}
        description = by_name["get_rcnv_associations"]["description"]
        assert "TESTED BUT NO ESTIMATE" in description
        assert "include_no_estimate" in description

    def test_both_tools_are_in_the_genetics_api_skill_toolset(self):
        from genetics_mcp_server.skills.definitions import _GENETICS_API_TOOLS

        assert {"get_dosage_sensitivity", "get_rcnv_associations"} <= _GENETICS_API_TOOLS
