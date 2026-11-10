"""Unit tests for get_gene_based_results_by_phenotype's bounded result shapes.

Self-contained: the results-api call is stubbed with a table the size of a real exome
study's, since the defect these guard against was the whole table going inline and the
MCP client dropping the connection on the oversized message.
"""

import json
from unittest.mock import AsyncMock, Mock

from genetics_mcp_server.sdk.client import GeneticsClient
from genetics_mcp_server.tools import ToolExecutor
from genetics_mcp_server.tools.definitions import TOOL_DEFINITIONS

ANNOTATIONS = ("PTV", "PTV_missense")
N_GENES = 17_178
MAX_INLINE_BYTES = 1_000_000


def _row(gene: str, gene_id: str, annotation: str, mlog10p: float | None) -> dict:
    return {
        "dataset": "SCHEMA2",
        "trait": "Schizophrenia",
        "gene": gene,
        "gene_id": gene_id,
        "gene_chr": 16,
        "gene_start_pos": 30957753,
        "gene_end_pos": 30984664,
        "annotation": annotation,
        "mlog10p_burden": mlog10p,
        "beta": 0.5,
        "se": 0.1,
        "total_variants": 100,
        "total_variants_pheno": 100,
        "n_cases": 28898,
        "n_controls": 103041,
        "trait_original": "SCZ",
        "flags": None,
    }


def _table() -> list[dict]:
    """Every gene x annotation in file order, which is not significance order."""
    rows = []
    for i in range(N_GENES - 1):
        for j, annotation in enumerate(ANNOTATIONS):
            # weak everywhere except GENE7, the one exome-wide hit besides SETD1A's PTV
            mlog10p = 9.0 if i == 7 and j == 0 else (i % 40) / 10
            rows.append(_row(f"GENE{i}", f"ENSG{i:011d}", annotation, mlog10p))
    rows.append(_row("SETD1A", "ENSG00000099381", "PTV", 11.2))
    rows.append(_row("SETD1A", "ENSG00000099381", "PTV_missense", None))
    return rows


def _executor(rows: list[dict], **kwargs) -> ToolExecutor:
    executor = ToolExecutor(api_base_url="http://unused.test/api", **kwargs)
    resp = Mock()
    resp.status_code = 200
    resp.json = Mock(return_value=rows)
    resp.text = ""
    resp.headers = {}
    executor.client.get = AsyncMock(return_value=resp)
    return executor


def _size(result: dict) -> int:
    return len(json.dumps(result).encode())


class TestUnfilteredCallIsBounded:
    async def test_default_is_a_summary_under_the_size_limit(self):
        executor = _executor(_table())
        try:
            result = await executor.get_gene_based_results_by_phenotype("schema2", "SCZ")
        finally:
            await executor.close()

        assert result["success"] is True
        assert _size(result) < MAX_INLINE_BYTES
        assert "results" not in result
        assert result["total_count"] == 2 * N_GENES
        assert result["n_genes"] == N_GENES
        assert result["n_significant_genes"] == 2
        assert result["by_annotation"] == {
            "PTV": {"n_rows": N_GENES, "n_significant": 2},
            "PTV_missense": {"n_rows": N_GENES, "n_significant": 0},
        }
        assert [r["gene"] for r in result["top_results"][:2]] == ["SETD1A", "GENE7"]
        assert "gene_based_results_by_phenotype/schema2/SCZ" in result["_download_url"]

    async def test_rows_are_capped_strongest_first_and_say_so(self):
        executor = _executor(_table())
        try:
            result = await executor.get_gene_based_results_by_phenotype(
                "schema2", "SCZ", summarize=False
            )
        finally:
            await executor.close()

        assert _size(result) < MAX_INLINE_BYTES
        assert result["truncated"] is True
        assert result["total_count"] == 2 * N_GENES
        assert len(result["results"]) == ToolExecutor._REGION_ROW_LIMIT
        values = [r["mlog10p_burden"] for r in result["results"]]
        assert values == sorted(values, reverse=True)
        assert values[0] == 11.2


class TestFilters:
    async def test_genes_returns_exactly_the_named_genes_rows(self):
        executor = _executor(_table())
        try:
            result = await executor.get_gene_based_results_by_phenotype(
                "schema2", "SCZ", genes=["SETD1A"]
            )
        finally:
            await executor.close()

        # the null PTV_missense row is the point: tested-and-null must come back
        assert [(r["gene"], r["annotation"]) for r in result["results"]] == [
            ("SETD1A", "PTV"),
            ("SETD1A", "PTV_missense"),
        ]
        assert result["total_count"] == 2
        assert result["truncated"] is False
        assert result["genes_not_found"] == []

    async def test_genes_accepts_a_comma_string_ids_and_any_case(self):
        executor = _executor(_table())
        try:
            result = await executor.get_gene_based_results_by_phenotype(
                "schema2", "SCZ", genes="setd1a, ENSG00000000003, NOTAGENE"
            )
        finally:
            await executor.close()

        assert {r["gene"] for r in result["results"]} == {"SETD1A", "GENE3"}
        assert result["genes_not_found"] == ["NOTAGENE"]

    async def test_annotation_and_min_mlog10p_narrow_the_summary(self):
        executor = _executor(_table())
        try:
            result = await executor.get_gene_based_results_by_phenotype(
                "schema2", "SCZ", annotation="PTV", min_mlog10p=6
            )
        finally:
            await executor.close()

        assert result["total_count"] == 2
        assert list(result["by_annotation"]) == ["PTV"]


class TestOtherSurfaces:
    async def test_sdk_receives_every_row(self):
        rows = _table()
        executor = _executor(rows, row_limit=None)
        try:
            client = GeneticsClient(executor)
            frame = await client.gene_burden(phenotype="SCZ", resource="schema2")
        finally:
            await executor.close()

        assert frame.height == len(rows)

    def test_tool_definition_exposes_the_parameters(self):
        (definition,) = [
            t for t in TOOL_DEFINITIONS if t["name"] == "get_gene_based_results_by_phenotype"
        ]
        assert {"genes", "annotation", "min_mlog10p", "summarize"} <= set(definition["parameters"])
