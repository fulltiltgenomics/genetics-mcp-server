"""Tests for the trait filter of get_credible_sets_stats.

A stats row names its study twice: `trait_original` is the code, `trait` is what results
display, and the two are equal only for resources whose display form is the code itself.
The filter has to find a study by either, since a caller may hold the accession from a
search or the display value from an earlier result.
"""

import httpx

from genetics_mcp_server.tools import ToolExecutor


def _stat(trait, trait_original, n_risk_cs):
    return {
        "trait": trait,
        "trait_original": trait_original,
        "dataset": "Open_Targets",
        "data_type": "GWAS",
        "n_risk_cs": n_risk_cs,
    }


CODE_AS_TRAIT = [_stat("GCST1", "GCST1", 3), _stat("GCST2", "GCST2", 5)]
NAME_AS_TRAIT = [
    _stat("Type_2_diabetes_(GCST1)", "GCST1", 3),
    _stat("Type_2_diabetes_(GCST2)", "GCST2", 5),
]


async def _stats(rows, trait):
    executor = ToolExecutor(api_base_url="http://api.test/api")
    executor.client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json=rows))
    )
    try:
        return await executor.get_credible_sets_stats("open_targets", trait=trait)
    finally:
        await executor.close()


class TestTraitFilter:
    async def test_accession_matches_through_trait_original(self):
        result = await _stats(NAME_AS_TRAIT, "GCST2")
        assert [row["trait"] for row in result["traits"]] == ["Type_2_diabetes_(GCST2)"]
        assert result["n_traits"] == 1
        assert result["totals"]["n_risk_cs"] == 5

    async def test_display_trait_matches(self):
        result = await _stats(NAME_AS_TRAIT, "Type_2_diabetes_(GCST1)")
        assert [row["trait_original"] for row in result["traits"]] == ["GCST1"]
        assert result["totals"]["n_risk_cs"] == 3

    async def test_accession_matches_when_trait_is_the_accession(self):
        result = await _stats(CODE_AS_TRAIT, "GCST1")
        assert [row["trait"] for row in result["traits"]] == ["GCST1"]
        assert result["totals"]["n_risk_cs"] == 3

    async def test_unknown_trait_matches_nothing(self):
        result = await _stats(NAME_AS_TRAIT, "Type_2_diabetes")
        assert result["success"] is True
        assert result["traits"] == []

    async def test_rows_without_trait_original_still_filter_on_trait(self):
        rows = [{"trait": "T2D", "n_risk_cs": 2}, {"trait": "HEIGHT", "n_risk_cs": 7}]
        result = await _stats(rows, "T2D")
        assert [row["trait"] for row in result["traits"]] == ["T2D"]
