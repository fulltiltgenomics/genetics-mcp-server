"""Integration tests for tool executor (requires running genetics API).

Run with: pytest -m integration
"""

import pytest

from genetics_mcp_server.tools import ServerToolExecutor, ToolExecutor
from genetics_mcp_server.tools.definitions import (
    BIGQUERY_TOOL_DEFINITIONS,
    SUBAGENT_TOOL_DEFINITIONS,
    TOOL_DEFINITIONS,
    TOOL_PROFILE_TOOLS,
    TOOL_PROFILES,
    get_anthropic_tools,
)


@pytest.mark.integration
class TestSearchTools:
    """Tests for search tools."""

    @pytest.fixture(autouse=True)
    async def setup_executor(self):
        """Create and cleanup executor for each test."""
        self.executor = ToolExecutor()
        yield
        await self.executor.close()

    async def test_search_phenotypes(self):
        """Test searching phenotypes by query."""
        result = await self.executor.search_phenotypes("diabetes", limit=5)

        assert result["success"] is True
        assert "results" in result
        assert len(result["results"]) <= 5

    async def test_search_phenotypes_empty_query(self):
        """Test searching phenotypes with a query that may return few results."""
        result = await self.executor.search_phenotypes("xyznonexistent123")

        assert result["success"] is True
        assert "results" in result

    async def test_search_genes(self):
        """Test searching genes by query."""
        result = await self.executor.search_genes("APOE", limit=3)

        assert result["success"] is True
        assert "results" in result
        assert len(result["results"]) <= 3

    async def test_search_genes_by_name(self):
        """Test searching genes by full name."""
        result = await self.executor.search_genes("apolipoprotein", limit=5)

        assert result["success"] is True
        assert "results" in result

    async def test_lookup_variants_by_rsid(self):
        """Test converting rsIDs to variant IDs."""
        result = await self.executor.lookup_variants_by_rsid("rs429358")

        assert result["success"] is True
        assert "variants" in result

    async def test_lookup_variants_by_rsid_multiple(self):
        """Test batch conversion of multiple rsIDs."""
        result = await self.executor.lookup_variants_by_rsid("rs429358,rs7412")

        assert result["success"] is True
        assert "variants" in result

    async def test_lookup_variants_by_rsid_empty(self):
        """Test lookup with empty rsID string."""
        result = await self.executor.lookup_variants_by_rsid("")

        assert result["success"] is False
        assert "error" in result

    async def test_lookup_phenotype_names(self):
        """Test batch lookup of phenotype codes to names."""
        result = await self.executor.lookup_phenotype_names(["T2D", "CAD"])

        assert result["success"] is True
        assert "names" in result
        assert "T2D" in result["names"]
        assert "CAD" in result["names"]

    async def test_lookup_phenotype_names_empty(self):
        """Test lookup with empty codes list."""
        result = await self.executor.lookup_phenotype_names([])

        assert result["success"] is False
        assert "error" in result


@pytest.mark.integration
class TestCredibleSetTools:
    """Tests for credible set tools."""

    @pytest.fixture(autouse=True)
    async def setup_executor(self):
        """Create and cleanup executor for each test."""
        self.executor = ToolExecutor()
        yield
        await self.executor.close()

    async def test_get_credible_sets_by_gene(self):
        """Test getting credible sets for a gene."""
        result = await self.executor.get_credible_sets_by_gene("APOE")

        assert result["success"] is True
        assert result["gene"] == "APOE"
        assert "n_cs" in result

    async def test_get_credible_sets_by_gene_with_window(self):
        """Test getting credible sets with custom window size."""
        result = await self.executor.get_credible_sets_by_gene(
            "APOE", window=50000
        )

        assert result["success"] is True

    async def test_get_credible_sets_by_gene_with_data_types(self):
        """Test filtering credible sets by data type."""
        result = await self.executor.get_credible_sets_by_gene(
            "APOE", data_types="GWAS"
        )

        assert result["success"] is True

    async def test_get_credible_sets_by_gene_no_summarize(self):
        """Test getting raw credible sets without summarization."""
        result = await self.executor.get_credible_sets_by_gene(
            "APOE", summarize=False
        )

        assert result["success"] is True
        assert "results" in result
        assert "total_count" in result

    async def test_get_credible_sets_by_variant(self):
        """Test getting credible sets containing a variant."""
        result = await self.executor.get_credible_sets_by_variant(
            "19:44908684:T:C"
        )

        assert result["success"] is True
        assert result["variant"] == "19:44908684:T:C"

    async def test_get_credible_sets_by_phenotype(self):
        """Test getting credible sets for a phenotype."""
        result = await self.executor.get_credible_sets_by_phenotype(
            "T2D", resource="finngen"
        )

        assert result["success"] is True
        assert result["phenotype"] == "T2D"

    async def test_get_credible_sets_by_qtl_gene(self):
        """Test getting QTL credible sets for a gene."""
        result = await self.executor.get_credible_sets_by_qtl_gene("APOE")

        assert result["success"] is True
        assert result["gene"] == "APOE"


@pytest.mark.integration
class TestGeneDataTools:
    """Tests for gene data tools."""

    @pytest.fixture(autouse=True)
    async def setup_executor(self):
        """Create and cleanup executor for each test."""
        self.executor = ToolExecutor()
        yield
        await self.executor.close()

    async def test_get_gene_expression(self):
        """Test getting tissue expression for a gene."""
        result = await self.executor.get_gene_expression("APOE")

        assert result["success"] is True
        assert result["gene"] == "APOE"
        assert "results" in result

    async def test_get_gene_disease_associations(self):
        """Test getting gene-disease associations."""
        result = await self.executor.get_gene_disease_associations("BRCA1")

        assert result["success"] is True
        assert result["gene"] == "BRCA1"
        assert "results" in result

    async def test_get_gene_disease_associations_none_found(self):
        """Test gene with no Mendelian disease associations."""
        result = await self.executor.get_gene_disease_associations("APOE")

        assert result["success"] is True

    async def test_get_exome_results_by_gene(self):
        """Test getting exome sequencing results."""
        result = await self.executor.get_exome_results_by_gene("APOE")

        assert result["success"] is True
        assert result["gene"] == "APOE"

    async def test_get_gene_based_results(self):
        """Test getting gene-level burden test results."""
        result = await self.executor.get_gene_based_results("APOE")

        assert result["success"] is True
        assert result["gene"] == "APOE"
        assert "count" in result
        assert "results" in result
        assert isinstance(result["results"], list)

    async def test_get_gene_based_results_multiple_genes(self):
        """Test getting gene-based results for multiple genes."""
        result = await self.executor.get_gene_based_results("APOE,BRCA1")

        assert result["success"] is True
        assert result["gene"] == "APOE,BRCA1"
        assert result["count"] > 0

    async def test_get_gene_based_results_invalid_gene(self):
        """Test gene-based results for a non-existent gene."""
        result = await self.executor.get_gene_based_results("FAKEGENE12345")

        assert result["success"] is False


@pytest.mark.integration
class TestLDTools:
    """Tests for FinnGen LD server tools."""

    @pytest.fixture(autouse=True)
    async def setup_executor(self):
        """Create and cleanup executor for each test."""
        self.executor = ToolExecutor()
        yield
        await self.executor.close()

    async def test_get_ld_between_variants_found(self):
        """Test getting LD between two variants that are in LD."""
        result = await self.executor.get_ld_between_variants(
            "6:44693011:A:G", "6:44682355:C:G"
        )
        assert result["success"] is True
        assert result["variant1"] == "6:44693011:A:G"
        assert result["variant2"] == "6:44682355:C:G"
        # either found in LD or not, but should succeed
        assert "in_ld" in result

    async def test_get_ld_between_variants_different_chromosomes(self):
        """Test error when variants are on different chromosomes."""
        result = await self.executor.get_ld_between_variants(
            "6:44693011:A:G", "7:12345678:C:T"
        )
        assert result["success"] is False
        assert "same chromosome" in result["error"].lower()

    async def test_get_ld_between_variants_too_far_apart(self):
        """Test error when variants are more than 5 Mb apart."""
        result = await self.executor.get_ld_between_variants(
            "6:10000000:A:G", "6:20000000:C:T"
        )
        assert result["success"] is False
        assert "too far apart" in result["error"].lower()

    async def test_get_ld_between_variants_invalid_format(self):
        """Test error with invalid variant format."""
        result = await self.executor.get_ld_between_variants(
            "invalid", "6:44682355:C:G"
        )
        assert result["success"] is False
        assert "invalid" in result["error"].lower()

    async def test_get_ld_between_variants_custom_panel(self):
        """Test LD lookup with different reference panel."""
        result = await self.executor.get_ld_between_variants(
            "6:44693011:A:G", "6:44682355:C:G", panel="sisu4"
        )
        assert result["success"] is True

    async def test_get_ld_between_variants_custom_threshold(self):
        """Test LD lookup with custom r2 threshold."""
        result = await self.executor.get_ld_between_variants(
            "6:44693011:A:G", "6:44682355:C:G", r2_threshold=0.8
        )
        assert result["success"] is True

    async def test_get_variants_in_ld(self):
        """Test getting variants in LD with a query variant."""
        result = await self.executor.get_variants_in_ld("6:44693011:A:G")
        assert result["success"] is True
        assert "n_variants" in result
        assert "variants" in result
        assert result["query_variant"] == "6:44693011:A:G"

    async def test_get_variants_in_ld_with_window(self):
        """Test LD query with custom window size."""
        result = await self.executor.get_variants_in_ld(
            "6:44693011:A:G", window=100000
        )
        assert result["success"] is True

    async def test_get_variants_in_ld_high_threshold(self):
        """Test LD query with high r2 threshold."""
        result = await self.executor.get_variants_in_ld(
            "6:44693011:A:G", r2_threshold=0.9
        )
        assert result["success"] is True
        # with high threshold, all returned variants should meet threshold
        for v in result.get("variants", []):
            assert v["r2"] >= 0.9

    async def test_get_variants_in_ld_invalid_variant(self):
        """Test error with invalid variant format."""
        result = await self.executor.get_variants_in_ld("invalid_format")
        assert result["success"] is False

    async def test_get_variants_in_ld_sorted_by_r2(self):
        """Test that results are sorted by r2 descending."""
        result = await self.executor.get_variants_in_ld(
            "6:44693011:A:G", r2_threshold=0.5
        )
        if result["success"] and len(result["variants"]) > 1:
            r2_values = [v["r2"] for v in result["variants"]]
            assert r2_values == sorted(r2_values, reverse=True)


@pytest.mark.integration
class TestColocalizationTools:
    """Tests for colocalization and report tools."""

    @pytest.fixture(autouse=True)
    async def setup_executor(self):
        """Create and cleanup executor for each test."""
        self.executor = ToolExecutor()
        yield
        await self.executor.close()

    async def test_get_colocalization(self):
        """Test getting colocalization results."""
        result = await self.executor.get_colocalization("19:44908684:T:C")

        assert result["success"] is True
        assert result["variant"] == "19:44908684:T:C"

    async def test_get_phenotype_report(self):
        """Test getting phenotype markdown report."""
        result = await self.executor.get_phenotype_report("finngen", "T2D")

        assert result["success"] is True
        assert result["phenotype_code"] == "T2D"

    async def test_list_datasets(self):
        """Test listing datasets."""
        result = await self.executor.list_datasets()

        assert result["success"] is True
        assert "datasets" in result


@pytest.mark.integration
class TestRegionTools:
    """Tests for region/variant location tools."""

    @pytest.fixture(autouse=True)
    async def setup_executor(self):
        """Create and cleanup executor for each test."""
        self.executor = ToolExecutor()
        yield
        await self.executor.close()

    async def test_get_nearest_genes(self):
        """Test getting genes nearest to a variant."""
        result = await self.executor.get_nearest_genes("19:44908684:T:C")

        assert result["success"] is True
        assert result["variant"] == "19:44908684:T:C"
        assert "genes" in result

    async def test_get_nearest_genes_with_params(self):
        """Test getting nearest genes with custom parameters."""
        result = await self.executor.get_nearest_genes(
            "19:44908684:T:C",
            gene_type="protein_coding",
            n=5,
            max_distance=500000,
        )

        assert result["success"] is True

    async def test_get_genes_in_region(self):
        """Test getting genes in a genomic region."""
        result = await self.executor.get_genes_in_region(
            chr="19", start=44905000, end=44910000
        )

        assert result["success"] is True
        assert "19:44905000-44910000" in result["region"]
        assert "genes" in result


@pytest.mark.integration
class TestLiteratureSearch:
    """Tests for literature search tools."""

    @pytest.fixture(autouse=True)
    async def setup_executor(self):
        """Create and cleanup executor for each test."""
        self.executor = ServerToolExecutor()
        yield
        await self.executor.close()

    async def test_search_scientific_literature(self):
        """Test searching scientific literature via Europe PMC."""
        result = await self.executor.search_scientific_literature(
            "APOE Alzheimer", max_results=5, backend="europepmc"
        )

        assert result["success"] is True
        assert result["query"] == "APOE Alzheimer"
        assert "results" in result
        assert "total_found" in result

    async def test_search_scientific_literature_exclude_preprints(self):
        """Test searching literature excluding preprints."""
        result = await self.executor.search_scientific_literature(
            "BRCA1 breast cancer",
            max_results=5,
            include_preprints=False,
            backend="europepmc",
        )

        assert result["success"] is True

    async def test_search_scientific_literature_with_date_range(self):
        """Test searching literature with date filter."""
        result = await self.executor.search_scientific_literature(
            "type 2 diabetes genetics",
            max_results=5,
            date_range="last_year",
            backend="europepmc",
        )

        assert result["success"] is True

    async def test_search_scientific_literature_explicit_europepmc(self):
        """Test explicitly selecting Europe PMC backend."""
        result = await self.executor.search_scientific_literature(
            "PCSK9 cholesterol",
            max_results=3,
            backend="europepmc",
        )

        assert result["success"] is True
        assert result.get("source") == "europepmc"

    async def test_search_scientific_literature_perplexity_unavailable_without_key(
        self, monkeypatch
    ):
        """Test error when Perplexity requested but key not set."""
        monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)

        result = await self.executor.search_scientific_literature(
            "IL6 inflammation",
            max_results=3,
            backend="perplexity",
        )

        # should return error, not fallback
        assert result["success"] is False
        assert "unavailable" in result["error"].lower()


@pytest.mark.integration
class TestLiteratureSearchPerplexity:
    """Tests for Perplexity-based literature search (requires API key)."""

    @pytest.fixture(autouse=True)
    async def setup_executor(self):
        """Create and cleanup executor for each test."""
        self.executor = ServerToolExecutor()
        yield
        await self.executor.close()

    @pytest.mark.skipif(
        not __import__("os").environ.get("PERPLEXITY_API_KEY"),
        reason="PERPLEXITY_API_KEY not set",
    )
    async def test_search_perplexity_literature(self):
        """Test searching literature via Perplexity."""
        result = await self.executor.search_scientific_literature(
            "APOE Alzheimer genetics",
            max_results=5,
            backend="perplexity",
        )

        assert result["success"] is True
        assert result.get("source") == "perplexity"
        assert "summary" in result
        assert "results" in result

    @pytest.mark.skipif(
        not __import__("os").environ.get("PERPLEXITY_API_KEY"),
        reason="PERPLEXITY_API_KEY not set",
    )
    async def test_search_perplexity_literature_exclude_preprints(self):
        """Test Perplexity search excluding preprints."""
        result = await self.executor.search_scientific_literature(
            "BRCA1 breast cancer",
            max_results=5,
            backend="perplexity",
            include_preprints=False,
        )

        assert result["success"] is True
        assert result.get("source") == "perplexity"


@pytest.mark.integration
class TestCredibleSetStatsTools:
    """Tests for credible set statistics tools."""

    @pytest.fixture(autouse=True)
    async def setup_executor(self):
        self.executor = ToolExecutor()
        yield
        await self.executor.close()

    async def test_get_credible_sets_stats(self):
        """Test getting credible sets statistics."""
        result = await self.executor.get_credible_sets_stats("finngen")

        assert result["success"] is True
        assert result["resource_or_dataset"] == "finngen"
        assert "n_traits" in result
        assert "totals" in result
        assert "n_protective_cs" in result["totals"]
        assert "n_risk_cs" in result["totals"]
        assert "_download_url" in result
        assert "finngen/stats" in result["_download_url"]

    async def test_get_credible_sets_stats_with_dataset_id(self):
        """Test getting stats with specific dataset ID."""
        result = await self.executor.get_credible_sets_stats("finngen_gwas")

        assert result["success"] is True

    async def test_get_credible_sets_stats_with_trait(self):
        """Test getting stats filtered by trait."""
        result = await self.executor.get_credible_sets_stats(
            "finngen", trait="T2D"
        )

        assert result["success"] is True
        if result["n_traits"] > 0:
            assert len(result["traits"]) > 0


@pytest.mark.integration
class TestSummaryStatsTools:
    """Tests for summary statistics tools."""

    @pytest.fixture(autouse=True)
    async def setup_executor(self):
        """Create and cleanup executor for each test."""
        self.executor = ToolExecutor()
        yield
        await self.executor.close()

    async def test_get_summary_stats(self):
        """Test fetching summary stats for a variant-phenotype pair."""
        result = await self.executor.get_summary_stats(
            variants=["19:44908684:T:C"],
            phenotypes=["T2D"],
            resource="finngen",
            data_type="gwas",
        )

        assert result["success"] is True
        assert result["resource"] == "finngen"
        assert result["data_type"] == "gwas"
        assert "results" in result
        assert result["count"] > 0
        row = result["results"][0]
        assert "pval" in row
        assert "beta" in row
        assert "se" in row

    async def test_get_summary_stats_multiple_variants(self):
        """Test fetching summary stats for multiple variants."""
        result = await self.executor.get_summary_stats(
            variants=["19:44908684:T:C", "19:44908822:C:T"],
            phenotypes=["T2D"],
            resource="finngen",
        )

        assert result["success"] is True
        assert result["count"] > 0

    async def test_get_summary_stats_multiple_phenotypes(self):
        """Test fetching summary stats for multiple phenotypes."""
        result = await self.executor.get_summary_stats(
            variants=["19:44908684:T:C"],
            phenotypes=["T2D", "I9_CHD"],
        )

        assert result["success"] is True
        assert result["count"] >= 2

    async def test_get_summary_stats_colon_separator(self):
        """Test that colon-separated variants are normalized correctly."""
        result = await self.executor.get_summary_stats(
            variants=["19:44908684:T:C"],
            phenotypes=["T2D"],
        )

        assert result["success"] is True

    async def test_get_summary_stats_empty_variants(self):
        """Test with empty variant list."""
        result = await self.executor.get_summary_stats(
            variants=[],
            phenotypes=["T2D"],
        )

        assert result["success"] is False
        assert "error" in result

    async def test_get_summary_stats_empty_phenotypes(self):
        """Test with empty phenotype list."""
        result = await self.executor.get_summary_stats(
            variants=["19:44908684:T:C"],
            phenotypes=[],
        )

        assert result["success"] is False
        assert "error" in result

    async def test_get_summary_stats_invalid_phenotype(self):
        """Test with nonexistent phenotype."""
        result = await self.executor.get_summary_stats(
            variants=["19:44908684:T:C"],
            phenotypes=["NONEXISTENT_PHENO_XYZ"],
        )

        assert result["success"] is False

    async def test_get_summary_stats_meta_analysis(self):
        """Test fetching from meta-analysis resource."""
        result = await self.executor.get_summary_stats(
            variants=["19:44908684:T:C"],
            phenotypes=["T2D"],
            resource="finngen_mvp_ukbb",
            data_type="gwas",
        )

        assert result["success"] is True


@pytest.mark.integration
class TestVariantAnnotationTools:
    """Tests for variant annotation tools."""

    @pytest.fixture(autouse=True)
    async def setup_executor(self):
        """Create and cleanup executor for each test."""
        self.executor = ToolExecutor()
        yield
        await self.executor.close()

    async def test_get_variant_annotations_by_gene(self):
        """Test fetching variant annotations for a gene."""
        result = await self.executor.get_variant_annotations(gene="PCSK9")

        assert result["success"] is True
        assert result["source"] == "finngen"
        assert result["count"] > 0
        assert "results" in result
        row = result["results"][0]
        assert "most_severe" in row
        assert "gene_most_severe" in row

    async def test_get_variant_annotations_by_variant(self):
        """Test fetching annotation for a single variant."""
        result = await self.executor.get_variant_annotations(variant="1:13668:G:A")

        assert result["success"] is True
        assert "results" in result

    async def test_get_variant_annotations_by_region(self):
        """Test fetching variant annotations for a region."""
        result = await self.executor.get_variant_annotations(region="1:13668-14506")

        assert result["success"] is True
        assert "results" in result

    async def test_get_variant_annotations_batch(self):
        """Test batch variant annotation lookup via POST."""
        result = await self.executor.get_variant_annotations(
            variants=["1:13668:G:A", "1:14506:G:A"]
        )

        assert result["success"] is True
        assert "results" in result

    async def test_get_variant_annotations_no_query(self):
        """Test that missing query parameter returns error."""
        result = await self.executor.get_variant_annotations()

        assert result["success"] is False
        assert "error" in result

    async def test_get_variant_annotations_multiple_query_params(self):
        """Test that providing multiple query params returns error."""
        result = await self.executor.get_variant_annotations(
            variant="1:13668:G:A", gene="PCSK9"
        )

        assert result["success"] is False
        assert "error" in result

    async def test_get_variant_annotations_unknown_gene(self):
        """Test that unknown gene returns not found error."""
        result = await self.executor.get_variant_annotations(gene="NONEXISTENTGENE123")

        assert result["success"] is False

    async def test_get_variant_annotations_download_url(self):
        """Test that GET queries include download URL."""
        result = await self.executor.get_variant_annotations(gene="PCSK9")

        assert result["success"] is True
        if result["count"] > 0:
            assert "_download_url" in result

    async def test_get_variant_annotations_batch_download_data(self):
        """Test that POST queries include download data."""
        result = await self.executor.get_variant_annotations(
            variants=["1:13668:G:A"]
        )

        assert result["success"] is True
        if result["count"] > 0:
            assert "_download_data" in result


# Frozen expected name sets for the category-based profiles, recorded 2026-08-18.
#
# These are literals on purpose. Deriving them from TOOL_DEFINITIONS at test time makes
# both sides of the comparison move together, so recategorising a tool - the very change
# that was ruled out when the code profile was made an explicit allow-list - could not
# fail the test. A literal fails on any substitution, not only on ones that change a count.
# Update them only alongside a deliberate, reviewed change to a tool's category.
_PROFILE_NONE_NAMES = {
    "analyze_variant_list",
    "get_asm_qtl_by_gene",
    "get_asm_qtl_by_variant",
    "get_colocalization",
    "get_colocalization_by_credible_set",
    "get_credible_set_by_id",
    "get_credible_set_leads_by_phenotype",
    "get_credible_sets_by_gene",
    "get_credible_sets_by_phenotype",
    "get_credible_sets_by_qtl_gene",
    "get_credible_sets_by_region",
    "get_credible_sets_by_variant",
    "get_credible_sets_stats",
    "get_database_schema",
    "get_dataset_display_names",
    "get_exome_results_by_gene",
    "get_exome_results_by_phenotype",
    "get_exome_results_by_region",
    "get_exome_results_by_variant",
    "get_gene_based_results",
    "get_gene_based_results_by_phenotype",
    "get_gene_disease_associations",
    "get_gene_expression",
    "get_gene_group_members",
    "get_gene_to_peaks",
    "get_genes_in_region",
    "get_hla_by_allele",
    "get_hla_by_phenotype",
    "get_ld_between_variants",
    "get_mpra_by_gene",
    "get_mpra_by_region",
    "get_mpra_by_variant",
    "get_mpra_pip_concordance_by_gene",
    "get_myvariant_annotations",
    "get_nearest_genes",
    "get_open_chromatin_by_gene",
    "get_open_chromatin_by_peak",
    "get_open_chromatin_by_region",
    "get_open_chromatin_by_variant",
    "get_peak_to_genes",
    "get_phenotype_report",
    "get_protein_annotations",
    "get_resource_metadata",
    "get_summary_stats",
    "get_summary_stats_by_region",
    "get_variant_annotations",
    "get_variant_effect_by_gene",
    "get_variant_effect_by_variant",
    "get_variant_protein_effect",
    "get_variants_in_ld",
    "launch_subagents",
    "list_capabilities",
    "list_datasets",
    "lookup_phenotype_names",
    "lookup_variants_by_rsid",
    "map_protein_variants",
    "normalize_gene_symbols",
    "query_database",
    "read_artifact",
    "run_analysis",
    "search_cbioportal",
    "search_genes",
    "search_mgi",
    "search_phenotypes",
    "search_scientific_literature",
    "search_uniprot",
    "web_search",
}

_PROFILE_API_NAMES = {
    "analyze_variant_list",
    "get_asm_qtl_by_gene",
    "get_asm_qtl_by_variant",
    "get_colocalization",
    "get_colocalization_by_credible_set",
    "get_credible_set_by_id",
    "get_credible_set_leads_by_phenotype",
    "get_credible_sets_by_gene",
    "get_credible_sets_by_phenotype",
    "get_credible_sets_by_qtl_gene",
    "get_credible_sets_by_region",
    "get_credible_sets_by_variant",
    "get_credible_sets_stats",
    "get_dataset_display_names",
    "get_exome_results_by_gene",
    "get_exome_results_by_phenotype",
    "get_exome_results_by_region",
    "get_exome_results_by_variant",
    "get_gene_based_results",
    "get_gene_based_results_by_phenotype",
    "get_gene_disease_associations",
    "get_gene_expression",
    "get_gene_group_members",
    "get_gene_to_peaks",
    "get_genes_in_region",
    "get_hla_by_allele",
    "get_hla_by_phenotype",
    "get_ld_between_variants",
    "get_mpra_by_gene",
    "get_mpra_by_region",
    "get_mpra_by_variant",
    "get_mpra_pip_concordance_by_gene",
    "get_myvariant_annotations",
    "get_nearest_genes",
    "get_open_chromatin_by_gene",
    "get_open_chromatin_by_peak",
    "get_open_chromatin_by_region",
    "get_open_chromatin_by_variant",
    "get_peak_to_genes",
    "get_phenotype_report",
    "get_protein_annotations",
    "get_resource_metadata",
    "get_summary_stats",
    "get_summary_stats_by_region",
    "get_variant_annotations",
    "get_variant_effect_by_gene",
    "get_variant_effect_by_variant",
    "get_variant_protein_effect",
    "get_variants_in_ld",
    "launch_subagents",
    "list_capabilities",
    "list_datasets",
    "lookup_phenotype_names",
    "lookup_variants_by_rsid",
    "map_protein_variants",
    "normalize_gene_symbols",
    "read_artifact",
    "run_analysis",
    "search_cbioportal",
    "search_genes",
    "search_mgi",
    "search_phenotypes",
    "search_scientific_literature",
    "search_uniprot",
    "web_search",
}

_PROFILE_BIGQUERY_NAMES = {
    "get_database_schema",
    "get_dataset_display_names",
    "get_gene_group_members",
    "get_protein_annotations",
    "get_resource_metadata",
    "get_variant_protein_effect",
    "launch_subagents",
    "list_capabilities",
    "list_datasets",
    "lookup_phenotype_names",
    "lookup_variants_by_rsid",
    "map_protein_variants",
    "normalize_gene_symbols",
    "query_database",
    "read_artifact",
    "run_analysis",
    "search_cbioportal",
    "search_genes",
    "search_mgi",
    "search_phenotypes",
    "search_scientific_literature",
    "search_uniprot",
    "web_search",
}

# also the resolved set for any unrecognised profile string, which degrades to general-only
_PROFILE_RAG_NAMES = {
    "get_dataset_display_names",
    "get_gene_group_members",
    "get_protein_annotations",
    "get_resource_metadata",
    "get_variant_protein_effect",
    "list_datasets",
    "lookup_phenotype_names",
    "lookup_variants_by_rsid",
    "map_protein_variants",
    "normalize_gene_symbols",
    "search_cbioportal",
    "search_genes",
    "search_mgi",
    "search_phenotypes",
    "search_scientific_literature",
    "search_uniprot",
    "web_search",
}


class TestToolDefinitions:
    """Tests for tool definitions and profile filtering."""

    def test_all_tools_have_category(self):
        """Every tool definition must have a category field."""
        for tool in TOOL_DEFINITIONS:
            assert "category" in tool, f"Tool {tool['name']} missing category"
        for tool in BIGQUERY_TOOL_DEFINITIONS:
            assert "category" in tool, f"Tool {tool['name']} missing category"

    def test_valid_categories(self):
        """Tool categories must be one of the known values."""
        valid = {"general", "api", "bigquery", "orchestration"}
        for tool in TOOL_DEFINITIONS + BIGQUERY_TOOL_DEFINITIONS + SUBAGENT_TOOL_DEFINITIONS:
            assert tool["category"] in valid, (
                f"Tool {tool['name']} has invalid category {tool['category']}"
            )

    def test_get_anthropic_tools_no_profile_returns_all(self):
        """No profile returns all tools (general + api + bigquery)."""
        tools = get_anthropic_tools()
        names = {t["name"] for t in tools}

        assert "search_phenotypes" in names  # general
        assert "get_credible_sets_by_gene" in names  # api
        assert "query_database" in names  # bigquery

        total = len(TOOL_DEFINITIONS) + len(BIGQUERY_TOOL_DEFINITIONS) + len(SUBAGENT_TOOL_DEFINITIONS)
        assert len(tools) == total

    def test_get_anthropic_tools_api_profile(self):
        """API profile returns general + api tools only."""
        tools = get_anthropic_tools(tool_profile="api")
        names = {t["name"] for t in tools}

        assert "search_phenotypes" in names  # general
        assert "get_credible_sets_by_gene" in names  # api
        assert "query_database" not in names  # bigquery excluded

    def test_get_anthropic_tools_bigquery_profile(self):
        """BigQuery profile returns general + bigquery tools only."""
        tools = get_anthropic_tools(tool_profile="bigquery")
        names = {t["name"] for t in tools}

        assert "search_phenotypes" in names  # general
        assert "query_database" in names  # bigquery
        assert "get_database_schema" in names  # bigquery
        assert "get_credible_sets_by_gene" not in names  # api excluded

    def test_get_anthropic_tools_rag_profile(self):
        """RAG profile returns general tools only (no api, no bigquery)."""
        tools = get_anthropic_tools(tool_profile="rag")
        names = {t["name"] for t in tools}

        assert "search_phenotypes" in names  # general
        assert "web_search" in names  # general
        assert "get_credible_sets_by_gene" not in names  # api excluded
        assert "query_database" not in names  # bigquery excluded

    def test_get_anthropic_tools_unknown_profile_returns_general_only(self):
        """Unknown profile falls back to general tools only."""
        tools = get_anthropic_tools(tool_profile="unknown")
        names = {t["name"] for t in tools}

        assert "search_phenotypes" in names  # general
        assert "get_credible_sets_by_gene" not in names
        assert "query_database" not in names

    def test_code_profile_resolves_to_exactly_its_seven_tools(self):
        """The code profile is an explicit allow-list, not a category union.

        Asserting the SET rather than the count so a silent substitution — one tool
        renamed or swapped for another — fails instead of passing on arithmetic.
        """
        names = {t["name"] for t in get_anthropic_tools(tool_profile="code")}

        assert names == {
            "run_analysis",
            "list_capabilities",
            "read_artifact",
            "search_genes",
            "search_phenotypes",
            "search_scientific_literature",
            "lookup_variants_by_rsid",
        }

    def test_code_profile_excludes_launch_subagents(self):
        """launch_subagents shares the orchestration category with run_analysis but is
        deliberately not in the code profile: this profile measures one agent with a
        sandbox, not a fan-out. It is the case a category-based profile could not express.
        """
        names = {t["name"] for t in get_anthropic_tools(tool_profile="code")}

        assert "launch_subagents" not in names
        assert "run_analysis" in names  # same category, still included

    def test_explicit_profile_respects_disabled_tools(self):
        """disabled_tools is applied before the profile filter, so a deployment flag or
        the env-driven disable list still removes a tool the profile names."""
        names = {
            t["name"]
            for t in get_anthropic_tools(tool_profile="code", disabled_tools={"read_artifact"})
        }

        assert "read_artifact" not in names
        assert "run_analysis" in names

    def test_existing_profiles_unchanged_by_the_code_profile(self):
        """Adding an explicit-allow-list profile must not move any existing profile.

        Compares whole NAME SETS against frozen literals rather than counts: swapping two
        tools between categories keeps every count identical while changing what each
        profile actually sends to the model, and that is exactly the accident this test
        exists to catch.
        """
        assert {t["name"] for t in get_anthropic_tools()} == _PROFILE_NONE_NAMES
        assert {
            t["name"] for t in get_anthropic_tools(tool_profile="api")
        } == _PROFILE_API_NAMES
        assert {
            t["name"] for t in get_anthropic_tools(tool_profile="bigquery")
        } == _PROFILE_BIGQUERY_NAMES
        assert {
            t["name"] for t in get_anthropic_tools(tool_profile="rag")
        } == _PROFILE_RAG_NAMES

    def test_unknown_profile_still_degrades_silently_to_general(self):
        """Pinned deliberately: an unrecognised profile name does not raise.

        The value is persisted per message in chat_messages.tool_profile and read back
        from rows written by older clients, so raising would turn a stale row into a 500.
        The cost is that a typo silently drops 50 tools — documented, not accidental.

        Compared against the frozen literal rather than a set recomputed from
        TOOL_DEFINITIONS, so a recategorisation cannot move both sides at once.
        """
        assert {
            t["name"] for t in get_anthropic_tools(tool_profile="cdoe")
        } == _PROFILE_RAG_NAMES
        assert {t["name"] for t in get_anthropic_tools(tool_profile="")} == _PROFILE_RAG_NAMES

    def test_explicit_and_category_profiles_have_disjoint_names(self):
        """No profile name may appear in both dicts.

        get_anthropic_tools checks TOOL_PROFILE_TOOLS first and returns on a hit, so a
        name present in both silently redefines the category-based profile: the entry in
        TOOL_PROFILES becomes dead code and the profile starts sending a different tool
        set with nothing else failing.
        """
        overlap = set(TOOL_PROFILE_TOOLS) & set(TOOL_PROFILES)

        assert not overlap, (
            f"profile name(s) {sorted(overlap)} defined in both TOOL_PROFILE_TOOLS and "
            "TOOL_PROFILES; the explicit allow-list wins and silently overrides the "
            "category-based definition"
        )

    def test_explicit_profiles_only_name_tools_that_exist(self):
        """Every name in every explicit allow-list must resolve to a real tool.

        A typo or a renamed tool leaves the profile quietly one tool short, and the filter
        cannot detect it: it intersects, so a name matching nothing simply drops out. This
        covers all explicit profiles, not just the one that has its own resolution test.
        """
        defined = {
            t["name"]
            for t in TOOL_DEFINITIONS + BIGQUERY_TOOL_DEFINITIONS + SUBAGENT_TOOL_DEFINITIONS
        }

        for profile, names in TOOL_PROFILE_TOOLS.items():
            assert names, f"explicit profile {profile} names no tools"
            missing = set(names) - defined
            assert not missing, (
                f"profile {profile} names non-existent tool(s) {sorted(missing)}"
            )
            resolved = {t["name"] for t in get_anthropic_tools(tool_profile=profile)}
            assert resolved == set(names), (
                f"profile {profile} resolved to {sorted(resolved)}, expected {sorted(names)}"
            )

    def test_general_tools_present_in_all_profiles(self):
        """General tools should appear in every category-based profile.

        TOOL_PROFILE_TOOLS profiles are excluded by construction: they name their tools
        explicitly and the code profile takes only 4 of the 18 general tools.
        """
        general_tools = {t["name"] for t in TOOL_DEFINITIONS if t["category"] == "general"}
        assert len(general_tools) > 0

        for profile in TOOL_PROFILES:
            tools = get_anthropic_tools(tool_profile=profile)
            names = {t["name"] for t in tools}
            for gen_tool in general_tools:
                assert gen_tool in names, (
                    f"General tool {gen_tool} missing from profile {profile}"
                )


# MouseMine PathQuery returns row arrays whose column order is dictated by the
# 'view=' attribute on the inline XML in executor._mgi_* helpers. The mocks
# below mirror those exact column orders so the parsing logic is exercised.
@pytest.mark.asyncio
class TestSearchMGI:
    """Mocked HTTP tests for ToolExecutor.search_mgi against MouseMine."""

    @pytest.fixture(autouse=True)
    async def setup_executor(self):
        from genetics_mcp_server.tools.executor import ToolExecutor

        self.executor = ToolExecutor()
        yield
        await self.executor.close()

    @staticmethod
    def _mock_response(status_code: int = 200, json_data: dict | None = None, text: str = ""):
        from unittest.mock import MagicMock

        resp = MagicMock()
        resp.status_code = status_code
        resp.text = text
        resp.json = MagicMock(return_value=json_data or {})
        return resp

    async def test_gene_phenotypes_success(self):
        from unittest.mock import AsyncMock, patch

        # columns per executor._mgi_gene_phenotypes view:
        # [mgi_id, symbol, name, mp_id, mp_term, allele_id, allele_symbol, allele_name]
        json_data = {
            "results": [
                [
                    "MGI:97874",
                    "Trp53",
                    "transformation related protein 53",
                    "MP:0001262",
                    "decreased body weight",
                    "MGI:1857436",
                    "Trp53<tm1Tyj>",
                    "targeted mutation 1, Tyler Jacks",
                ],
                # second MP term on the same allele — collapsed into one gene entry
                [
                    "MGI:97874",
                    "Trp53",
                    "transformation related protein 53",
                    "MP:0002169",
                    "no abnormal phenotype detected",
                    "MGI:1857436",
                    "Trp53<tm1Tyj>",
                    "targeted mutation 1, Tyler Jacks",
                ],
            ]
        }
        mock = self._mock_response(json_data=json_data)
        with patch.object(self.executor.external_client, "get", new_callable=AsyncMock, return_value=mock):
            result = await self.executor.search_mgi(query="Trp53", query_type="gene_phenotypes")

        assert result["success"] is True
        assert result["query"] == "Trp53"
        assert result["query_type"] == "gene_phenotypes"
        assert result["source"] == "mgi"
        assert result["returned"] == 1
        gene = result["results"][0]
        assert gene["mgi_id"] == "MGI:97874"
        assert gene["symbol"] == "Trp53"
        assert gene["url"] == "https://www.informatics.jax.org/marker/MGI:97874"
        mp_ids = {p["mp_id"] for p in gene["phenotype_terms"]}
        assert mp_ids == {"MP:0001262", "MP:0002169"}
        assert len(gene["alleles"]) == 1
        assert gene["alleles"][0]["mgi_id"] == "MGI:1857436"
        assert gene["alleles"][0]["url"] == "https://www.informatics.jax.org/allele/MGI:1857436"

    async def test_phenotype_genes_success(self):
        from unittest.mock import AsyncMock, patch

        # columns per executor._mgi_phenotype_genes view:
        # [mp_id, mp_term, gene_mgi_id, gene_symbol, gene_name]
        json_data = {
            "results": [
                ["MP:0001262", "decreased body weight", "MGI:97874", "Trp53", "transformation related protein 53"],
                ["MP:0001262", "decreased body weight", "MGI:88336", "Brca1", "breast cancer 1, early onset"],
                # row with missing gene id is dropped
                ["MP:0001262", "decreased body weight", None, None, None],
            ]
        }
        mock = self._mock_response(json_data=json_data)
        with patch.object(self.executor.external_client, "get", new_callable=AsyncMock, return_value=mock):
            result = await self.executor.search_mgi(
                query="MP:0001262", query_type="phenotype_genes"
            )

        assert result["success"] is True
        assert result["returned"] == 2
        symbols = {g["symbol"] for g in result["results"]}
        assert symbols == {"Trp53", "Brca1"}
        for gene in result["results"]:
            assert gene["phenotype_terms"][0]["mp_id"] == "MP:0001262"
            assert gene["url"].startswith("https://www.informatics.jax.org/marker/")

    async def test_allele_success(self):
        from unittest.mock import AsyncMock, patch

        # columns per executor._mgi_allele view:
        # [allele_id, allele_symbol, allele_name, allele_type,
        #  gene_id, gene_symbol, gene_name, mp_id, mp_term]
        json_data = {
            "results": [
                [
                    "MGI:1857436",
                    "Trp53<tm1Tyj>",
                    "targeted mutation 1, Tyler Jacks",
                    "Targeted (knock-out)",
                    "MGI:97874",
                    "Trp53",
                    "transformation related protein 53",
                    "MP:0001262",
                    "decreased body weight",
                ],
            ]
        }
        mock = self._mock_response(json_data=json_data)
        with patch.object(self.executor.external_client, "get", new_callable=AsyncMock, return_value=mock):
            result = await self.executor.search_mgi(
                query="MGI:1857436", query_type="allele"
            )

        assert result["success"] is True
        assert result["source"] == "mgi"
        assert result["returned"] == 1
        entry = result["results"][0]
        assert entry["mgi_id"] == "MGI:97874"
        assert entry["symbol"] == "Trp53"
        assert entry["alleles"][0]["mgi_id"] == "MGI:1857436"
        assert entry["alleles"][0]["allele_type"] == "Targeted (knock-out)"
        assert entry["phenotype_terms"][0]["mp_id"] == "MP:0001262"

    async def test_ortholog_success(self):
        from unittest.mock import AsyncMock, patch

        # columns per executor._mgi_ortholog view:
        # [mgi_id, symbol, name,
        #  ortho_mgi_id, ortho_symbol, ortho_name, ortho_organism]
        json_data = {
            "results": [
                [
                    "MGI:97874",
                    "Trp53",
                    "transformation related protein 53",
                    "HGNC:11998",
                    "TP53",
                    "tumor protein p53",
                    "H. sapiens",
                ],
            ]
        }
        mock = self._mock_response(json_data=json_data)
        with patch.object(self.executor.external_client, "get", new_callable=AsyncMock, return_value=mock):
            result = await self.executor.search_mgi(
                query="Trp53", query_type="ortholog", species="mouse"
            )

        assert result["success"] is True
        assert result["returned"] == 1
        entry = result["results"][0]
        assert entry["symbol"] == "Trp53"
        assert entry["orthologs"][0]["mgi_id"] == "HGNC:11998"
        assert entry["orthologs"][0]["symbol"] == "TP53"
        assert entry["orthologs"][0]["organism"] == "H. sapiens"
        assert entry["url"] == "https://www.informatics.jax.org/marker/MGI:97874"

    async def test_empty_results(self):
        from unittest.mock import AsyncMock, patch

        mock = self._mock_response(json_data={"results": []})
        with patch.object(self.executor.external_client, "get", new_callable=AsyncMock, return_value=mock):
            result = await self.executor.search_mgi(query="Xyz123", query_type="gene_phenotypes")

        assert result["success"] is True
        assert result["returned"] == 0
        assert result["results"] == []
        assert result["source"] == "mgi"

    async def test_http_failure_returns_error_not_raises(self):
        from unittest.mock import AsyncMock, patch

        mock = self._mock_response(status_code=500, text="MouseMine is down")
        with patch.object(self.executor.external_client, "get", new_callable=AsyncMock, return_value=mock):
            result = await self.executor.search_mgi(query="Trp53")

        assert result["success"] is False
        assert "error" in result
        assert "500" in result["error"]

    async def test_network_exception_returns_error(self):
        # network-level failure (e.g. timeout) must be caught by search_mgi
        # and surfaced as {success: False, error: ...} rather than propagating
        from unittest.mock import AsyncMock, patch

        import httpx

        with patch.object(
            self.executor.external_client,
            "get",
            new_callable=AsyncMock,
            side_effect=httpx.TimeoutException("timed out"),
        ):
            result = await self.executor.search_mgi(query="Trp53")

        assert result["success"] is False
        assert "error" in result

    async def test_unknown_query_type(self):
        result = await self.executor.search_mgi(query="x", query_type="nonsense")
        assert result["success"] is False
        assert "Unknown query_type" in result["error"]


# cBioPortal responses are mocked by (method, path) because a single query type
# fans out to several endpoints; the fixtures below mirror the real payload
# shapes verified against https://www.cbioportal.org/api.
@pytest.mark.asyncio
class TestSearchCBioPortal:
    """Mocked HTTP tests for ToolExecutor.search_cbioportal."""

    @pytest.fixture(autouse=True)
    async def setup_executor(self):
        from genetics_mcp_server.tools.executor import ToolExecutor

        self.executor = ToolExecutor()
        yield
        await self.executor.close()

    STUDIES = [
        {"studyId": "luad_tcga_pan_can_atlas_2018", "name": "Lung Adenocarcinoma (TCGA)",
         "cancerTypeId": "luad", "allSampleCount": 566, "referenceGenome": "hg19",
         "citation": "TCGA, Cell 2018", "pmid": "29625048", "description": "lung"},
        {"studyId": "msk_impact_2017", "name": "MSK-IMPACT Clinical Sequencing Cohort",
         "cancerTypeId": "mixed", "allSampleCount": 10945, "referenceGenome": "hg38",
         "citation": "Zehir et al. Nat Med 2017", "pmid": "28481359", "description": "pan-cancer"},
    ]

    PROFILES = [
        {"molecularProfileId": "luad_tcga_pan_can_atlas_2018_mutations",
         "studyId": "luad_tcga_pan_can_atlas_2018", "molecularAlterationType": "MUTATION_EXTENDED"},
        {"molecularProfileId": "msk_impact_2017_mutations",
         "studyId": "msk_impact_2017", "molecularAlterationType": "MUTATION_EXTENDED"},
        {"molecularProfileId": "msk_impact_2017_structural_variants",
         "studyId": "msk_impact_2017", "molecularAlterationType": "STRUCTURAL_VARIANT"},
    ]

    @staticmethod
    def _resp(json_data, status_code: int = 200, headers: dict | None = None, text: str = ""):
        from unittest.mock import MagicMock

        resp = MagicMock()
        resp.status_code = status_code
        resp.text = text
        resp.headers = headers or {}
        resp.json = MagicMock(return_value=json_data)
        return resp

    def _patch(self, gets: dict, posts: dict):
        """Route mocked GET/POST by URL suffix, failing loudly on an unmapped path."""
        from unittest.mock import AsyncMock, patch

        def pick(table, url):
            for suffix, payload in table.items():
                if url.endswith(suffix) or suffix in url:
                    return payload
            raise AssertionError(f"unmocked cBioPortal path: {url}")

        async def do_get(url, **kwargs):
            return pick(gets, url)

        async def do_post(url, **kwargs):
            return pick(posts, url)

        return (
            patch.object(self.executor.external_client, "get", new_callable=AsyncMock, side_effect=do_get),
            patch.object(self.executor.external_client, "post", new_callable=AsyncMock, side_effect=do_post),
        )

    async def test_gene_summary(self):
        gets = {
            "/genes/EGFR": self._resp({"entrezGeneId": 1956, "hugoGeneSymbol": "EGFR", "type": "protein-coding"}),
            "/studies": self._resp(self.STUDIES),
        }
        posts = {
            "/mutation-data-counts/fetch": self._resp([{"hugoGeneSymbol": "EGFR", "counts": [
                {"value": "MUTATED", "count": 100},
                {"value": "NOT_MUTATED", "count": 900},
                {"value": "NOT_PROFILED", "count": 50},
            ]}]),
            "/genomic-data-counts/fetch": self._resp([{"hugoGeneSymbol": "EGFR", "counts": [
                {"value": "NA", "count": 400},
                {"value": "2", "count": 30},
                {"value": "0", "count": 560},
                {"value": "-2", "count": 10},
            ]}]),
        }
        g, p = self._patch(gets, posts)
        with g, p:
            result = await self.executor.search_cbioportal(query="EGFR", query_type="gene_summary")

        assert result["success"] is True
        assert result["source"] == "cbioportal"
        mut = result["pan_cancer"]["mutation"]
        assert mut["altered_samples"] == 100
        # denominator excludes the 50 samples never profiled for this gene
        assert mut["profiled_samples"] == 1000
        assert mut["frequency"] == 0.1
        cna = result["pan_cancer"]["copy_number"]
        # 'NA' means no CNA call for this gene and must not inflate the denominator
        assert cna["profiled_samples"] == 600
        assert cna["levels"]["amplification"]["samples"] == 30
        assert cna["levels"]["deep_deletion"]["frequency"] == round(10 / 600, 5)
        # diploid is not an alteration and is dropped from the reported levels
        assert "diploid" not in cna["levels"]
        assert result["cohort"]["studies_by_reference_genome"] == {"hg19": 1, "hg38": 1}

    async def test_gene_by_cancer_type_merges_spellings_and_uses_profiled_denominator(self):
        gets = {
            "/genes/EGFR": self._resp({"entrezGeneId": 1956, "hugoGeneSymbol": "EGFR", "type": "protein-coding"}),
            "/studies": self._resp(self.STUDIES),
            "/molecular-profiles": self._resp(self.PROFILES),
        }
        calls = []

        from unittest.mock import AsyncMock, patch

        async def do_post(url, **kwargs):
            body = kwargs.get("json")
            calls.append(body)
            # the denominator call carries genomicProfiles, the numerator geneFilters
            if "genomicProfiles" in body.get("studyViewFilter", {}):
                return self._resp([{"attributeId": "CANCER_TYPE", "counts": [
                    {"value": "Non-Small Cell Lung Cancer", "count": 600},
                    {"value": "Non Small Cell Lung Cancer", "count": 400},
                    {"value": "Tiny Cohort", "count": 10},
                ]}])
            return self._resp([{"attributeId": "CANCER_TYPE", "counts": [
                {"value": "Non-Small Cell Lung Cancer", "count": 150},
                {"value": "Non Small Cell Lung Cancer", "count": 50},
                {"value": "Tiny Cohort", "count": 9},
            ]}])

        async def do_get(url, **kwargs):
            for suffix, payload in gets.items():
                if suffix in url:
                    return payload
            raise AssertionError(f"unmocked path: {url}")

        with patch.object(self.executor.external_client, "get", new_callable=AsyncMock, side_effect=do_get), \
             patch.object(self.executor.external_client, "post", new_callable=AsyncMock, side_effect=do_post):
            result = await self.executor.search_cbioportal(
                query="EGFR", query_type="gene_by_cancer_type"
            )

        assert result["success"] is True
        # the two spellings fold into one row: 200/1000, not two rows of 150/600 and 50/400
        assert len(result["results"]) == 1
        row = result["results"][0]
        assert row["altered_samples"] == 200
        assert row["profiled_samples"] == 1000
        assert row["frequency"] == 0.2
        # a 9/10 cohort would otherwise top the ranking at 90%
        assert result["cancer_types_below_min_cohort"] == 1
        # denominator query must restrict to samples that have mutation data
        den_call = next(c for c in calls if "genomicProfiles" in c["studyViewFilter"])
        assert den_call["studyViewFilter"]["genomicProfiles"] == [["mutations"]]

    async def test_gene_by_cancer_type_filters_to_requested_types(self):
        gets = {
            "/genes/EGFR": self._resp({"entrezGeneId": 1956, "hugoGeneSymbol": "EGFR", "type": "protein-coding"}),
            "/studies": self._resp(self.STUDIES),
            "/molecular-profiles": self._resp(self.PROFILES),
        }
        from unittest.mock import AsyncMock, patch

        async def do_post(url, **kwargs):
            counts = ([{"value": "Glioma", "count": 500}, {"value": "Melanoma", "count": 500}]
                      if "genomicProfiles" in kwargs["json"].get("studyViewFilter", {})
                      else [{"value": "Glioma", "count": 50}, {"value": "Melanoma", "count": 100}])
            return self._resp([{"attributeId": "CANCER_TYPE", "counts": counts}])

        async def do_get(url, **kwargs):
            for suffix, payload in gets.items():
                if suffix in url:
                    return payload
            raise AssertionError(f"unmocked path: {url}")

        with patch.object(self.executor.external_client, "get", new_callable=AsyncMock, side_effect=do_get), \
             patch.object(self.executor.external_client, "post", new_callable=AsyncMock, side_effect=do_post):
            result = await self.executor.search_cbioportal(
                query="EGFR", query_type="gene_by_cancer_type", cancer_types=["glioma"]
            )

        assert [r["cancer_type"] for r in result["results"]] == ["Glioma"]

    async def test_gene_mutations_keeps_builds_separate(self):
        gets = {
            "/genes/TP53": self._resp({"entrezGeneId": 7157, "hugoGeneSymbol": "TP53", "type": "protein-coding"}),
            "/molecular-profiles": self._resp(self.PROFILES),
        }
        records = [
            {"uniqueSampleKey": "s1", "proteinChange": "R175H", "proteinPosStart": 175,
             "mutationType": "Missense_Mutation", "ncbiBuild": "GRCh37", "chr": "17",
             "startPosition": 7578406, "referenceAllele": "C", "variantAllele": "T", "studyId": "a"},
            {"uniqueSampleKey": "s2", "proteinChange": "R175H", "proteinPosStart": 175,
             "mutationType": "Missense_Mutation", "ncbiBuild": "GRCh38", "chr": "17",
             "startPosition": 7675088, "referenceAllele": "C", "variantAllele": "T", "studyId": "b"},
            # same sample seen twice must not double-count
            {"uniqueSampleKey": "s1", "proteinChange": "R175H", "proteinPosStart": 175,
             "mutationType": "Missense_Mutation", "ncbiBuild": "GRCh37", "chr": "17",
             "startPosition": 7578406, "referenceAllele": "C", "variantAllele": "T", "studyId": "a"},
            {"uniqueSampleKey": "s3", "proteinChange": "R248Q", "proteinPosStart": 248,
             "mutationType": "Missense_Mutation", "ncbiBuild": "GRCh37", "chr": "17",
             "startPosition": 7577538, "referenceAllele": "C", "variantAllele": "T", "studyId": "a"},
        ]
        from unittest.mock import AsyncMock, patch

        async def do_post(url, **kwargs):
            if kwargs.get("params", {}).get("projection") == "META":
                return self._resp(None, headers={"total-count": "3"})
            return self._resp(records)

        async def do_get(url, **kwargs):
            for suffix, payload in gets.items():
                if suffix in url:
                    return payload
            raise AssertionError(f"unmocked path: {url}")

        with patch.object(self.executor.external_client, "get", new_callable=AsyncMock, side_effect=do_get), \
             patch.object(self.executor.external_client, "post", new_callable=AsyncMock, side_effect=do_post):
            result = await self.executor.search_cbioportal(query="TP53", query_type="gene_mutations")

        assert result["success"] is True
        assert result["scope"] == "all_studies"
        top = result["results"][0]
        assert top["protein_change"] == "R175H"
        assert top["sample_count"] == 2
        # the same amino-acid change sits at different positions per build: never merged
        assert top["coordinates_by_build"]["GRCh37"]["start_position"] == 7578406
        assert top["coordinates_by_build"]["GRCh38"]["start_position"] == 7675088
        assert result["genome_builds"] == {"GRCh37": 2, "GRCh38": 1}
        assert "GRCh38" in result["genome_build_note"]

    async def test_gene_mutations_falls_back_to_curated_cohort_when_too_large(self):
        gets = {
            "/genes/TP53": self._resp({"entrezGeneId": 7157, "hugoGeneSymbol": "TP53", "type": "protein-coding"}),
            "/molecular-profiles": self._resp(self.PROFILES),
            "/studies": self._resp(self.STUDIES),
        }
        seen_bodies = []
        from unittest.mock import AsyncMock, patch

        async def do_post(url, **kwargs):
            if kwargs.get("params", {}).get("projection") == "META":
                return self._resp(None, headers={"total-count": "141838"})
            seen_bodies.append(kwargs["json"])
            return self._resp([])

        async def do_get(url, **kwargs):
            for suffix, payload in gets.items():
                if suffix in url:
                    return payload
            raise AssertionError(f"unmocked path: {url}")

        with patch.object(self.executor.external_client, "get", new_callable=AsyncMock, side_effect=do_get), \
             patch.object(self.executor.external_client, "post", new_callable=AsyncMock, side_effect=do_post):
            result = await self.executor.search_cbioportal(query="TP53", query_type="gene_mutations")

        assert result["scope"] == "curated_cohort"
        assert result["total_mutation_records"] == 141838
        # the narrowed fetch must actually use the curated profiles, not the full set
        assert set(seen_bodies[-1]["molecularProfileIds"]) == {
            "luad_tcga_pan_can_atlas_2018_mutations", "msk_impact_2017_mutations"
        }

    async def test_gene_fusions_reports_partner_gene(self):
        gets = {
            "/genes/ALK": self._resp({"entrezGeneId": 238, "hugoGeneSymbol": "ALK", "type": "protein-coding"}),
            "/molecular-profiles": self._resp(self.PROFILES),
        }
        posts = {
            "/structural-variant/fetch": self._resp([
                {"site1HugoSymbol": "EML4", "site2HugoSymbol": "ALK", "studyId": "a", "variantClass": "DELETION"},
                {"site1HugoSymbol": "ALK", "site2HugoSymbol": "EML4", "studyId": "b", "variantClass": "DELETION"},
                {"site1HugoSymbol": "", "site2HugoSymbol": "ALK", "studyId": "b", "variantClass": "NA"},
            ]),
        }
        g, p = self._patch(gets, posts)
        with g, p:
            result = await self.executor.search_cbioportal(query="ALK", query_type="gene_fusions")

        assert result["success"] is True
        top = result["results"][0]
        # partner is whichever side is not the queried gene, regardless of orientation
        assert top["partner_gene"] == "EML4"
        assert top["sample_count"] == 2
        assert top["study_count"] == 2
        assert result["results"][1]["partner_gene"] == "(intergenic)"

    async def test_variant_hotspot_accepts_change_or_bare_position(self):
        gets = {"/genes/TP53": self._resp({"entrezGeneId": 7157, "hugoGeneSymbol": "TP53", "type": "protein-coding"})}
        posts = {"/mutation-counts-by-position/fetch": self._resp(
            [{"entrezGeneId": 7157, "proteinPosStart": 175, "proteinPosEnd": 175, "count": 6757}]
        )}
        for query in ("TP53 R175H", "TP53 175", "TP53:R175"):
            g, p = self._patch(gets, posts)
            with g, p:
                result = await self.executor.search_cbioportal(query=query, query_type="variant_hotspot")
            assert result["success"] is True, query
            assert result["protein_position"] == 175, query
            assert result["sample_count"] == 6757, query

    async def test_variant_hotspot_rejects_missing_residue(self):
        result = await self.executor.search_cbioportal(query="TP53", query_type="variant_hotspot")
        assert result["success"] is False
        assert "residue" in result["error"]

    async def test_study_search_ranks_by_size(self):
        gets = {"/studies": self._resp(self.STUDIES)}
        g, p = self._patch(gets, {})
        with g, p:
            result = await self.executor.search_cbioportal(query="lung", query_type="study_search")

        assert result["success"] is True
        assert result["matched"] == 1
        entry = result["results"][0]
        assert entry["study_id"] == "luad_tcga_pan_can_atlas_2018"
        assert entry["reference_genome"] == "hg19"
        assert entry["pmid"] == "29625048"

    async def test_unknown_gene_returns_error(self):
        gets = {"/genes/NOPE": self._resp(None, status_code=404, text="Gene not found")}
        g, p = self._patch(gets, {})
        with g, p:
            result = await self.executor.search_cbioportal(query="NOPE", query_type="gene_summary")

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    async def test_http_failure_returns_error_not_raises(self):
        gets = {"/genes/TP53": self._resp(None, status_code=500, text="cBioPortal is down")}
        g, p = self._patch(gets, {})
        with g, p:
            result = await self.executor.search_cbioportal(query="TP53", query_type="gene_summary")

        assert result["success"] is False
        assert "500" in result["error"]

    async def test_network_exception_returns_error(self):
        from unittest.mock import AsyncMock, patch

        import httpx

        with patch.object(
            self.executor.external_client,
            "get",
            new_callable=AsyncMock,
            side_effect=httpx.TimeoutException("timed out"),
        ):
            result = await self.executor.search_cbioportal(query="TP53", query_type="gene_summary")

        assert result["success"] is False
        assert "error" in result

    async def test_unknown_query_type(self):
        result = await self.executor.search_cbioportal(query="TP53", query_type="nonsense")
        assert result["success"] is False
        assert "Unknown query_type" in result["error"]
