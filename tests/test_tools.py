"""Integration tests for tool executor (requires running genetics API)."""

import pytest

from genetics_mcp_server.tools import ToolExecutor
from genetics_mcp_server.tools.definitions import (
    BIGQUERY_TOOL_DEFINITIONS,
    SUBAGENT_TOOL_DEFINITIONS,
    TOOL_DEFINITIONS,
    TOOL_PROFILES,
    get_anthropic_tools,
)


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


class TestLiteratureSearch:
    """Tests for literature search tools."""

    @pytest.fixture(autouse=True)
    async def setup_executor(self):
        """Create and cleanup executor for each test."""
        self.executor = ToolExecutor()
        yield
        await self.executor.close()

    async def test_search_scientific_literature(self):
        """Test searching scientific literature via Europe PMC."""
        result = await self.executor.search_scientific_literature(
            "APOE Alzheimer", max_results=5
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
        )

        assert result["success"] is True

    async def test_search_scientific_literature_with_date_range(self):
        """Test searching literature with date filter."""
        result = await self.executor.search_scientific_literature(
            "type 2 diabetes genetics",
            max_results=5,
            date_range="last_year",
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


class TestLiteratureSearchPerplexity:
    """Tests for Perplexity-based literature search (requires API key)."""

    @pytest.fixture(autouse=True)
    async def setup_executor(self):
        """Create and cleanup executor for each test."""
        self.executor = ToolExecutor()
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
        assert "INCLUDE_IN_RESPONSE" in result
        assert "finngen/stats" in result["INCLUDE_IN_RESPONSE"]

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


class TestVisualizationTools:
    """Tests for visualization tools."""

    @pytest.fixture(autouse=True)
    async def setup_executor(self):
        """Create and cleanup executor for each test."""
        self.executor = ToolExecutor()
        yield
        await self.executor.close()

    async def test_create_phewas_plot(self):
        """Test creating a PheWAS plot for a variant."""
        result = await self.executor.create_phewas_plot("19:44908684:T:C")

        assert result["success"] is True
        assert result["variant"] == "19:44908684:T:C"
        assert "n_associations" in result
        assert "n_significant" in result
        assert "categories" in result
        assert "image_base64" in result
        assert result["image_format"] == "png"
        # verify base64 is valid PNG (starts with PNG magic bytes when decoded)
        import base64
        decoded = base64.b64decode(result["image_base64"])
        assert decoded[:8] == b"\x89PNG\r\n\x1a\n"

    async def test_create_phewas_plot_with_resource(self):
        """Test creating PheWAS plot filtered by resource."""
        result = await self.executor.create_phewas_plot(
            "19:44908684:T:C",
            resource="finngen",
        )

        assert result["success"] is True
        assert "image_base64" in result

    async def test_create_phewas_plot_with_thresholds(self):
        """Test creating PheWAS plot with custom thresholds."""
        result = await self.executor.create_phewas_plot(
            "19:44908684:T:C",
            significance_threshold=5.0,
            min_mlog10p=1.0,
        )

        assert result["success"] is True
        assert "image_base64" in result

    async def test_create_phewas_plot_no_associations(self):
        """Test error when variant has no GWAS associations."""
        result = await self.executor.create_phewas_plot(
            "1:1:A:T",  # unlikely to have associations
            min_mlog10p=100.0,  # very high threshold
        )

        assert result["success"] is False
        assert "error" in result


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
        valid = {"general", "api", "bigquery"}
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
        assert "query_bigquery" in names  # bigquery

        total = len(TOOL_DEFINITIONS) + len(BIGQUERY_TOOL_DEFINITIONS) + len(SUBAGENT_TOOL_DEFINITIONS)
        assert len(tools) == total

    def test_get_anthropic_tools_api_profile(self):
        """API profile returns general + api tools only."""
        tools = get_anthropic_tools(tool_profile="api")
        names = {t["name"] for t in tools}

        assert "search_phenotypes" in names  # general
        assert "get_credible_sets_by_gene" in names  # api
        assert "query_bigquery" not in names  # bigquery excluded

    def test_get_anthropic_tools_bigquery_profile(self):
        """BigQuery profile returns general + bigquery tools only."""
        tools = get_anthropic_tools(tool_profile="bigquery")
        names = {t["name"] for t in tools}

        assert "search_phenotypes" in names  # general
        assert "query_bigquery" in names  # bigquery
        assert "get_bigquery_schema" in names  # bigquery
        assert "get_credible_sets_by_gene" not in names  # api excluded

    def test_get_anthropic_tools_rag_profile(self):
        """RAG profile returns general tools only (no api, no bigquery)."""
        tools = get_anthropic_tools(tool_profile="rag")
        names = {t["name"] for t in tools}

        assert "search_phenotypes" in names  # general
        assert "web_search" in names  # general
        assert "get_credible_sets_by_gene" not in names  # api excluded
        assert "query_bigquery" not in names  # bigquery excluded

    def test_get_anthropic_tools_unknown_profile_returns_general_only(self):
        """Unknown profile falls back to general tools only."""
        tools = get_anthropic_tools(tool_profile="unknown")
        names = {t["name"] for t in tools}

        assert "search_phenotypes" in names  # general
        assert "get_credible_sets_by_gene" not in names
        assert "query_bigquery" not in names

    def test_general_tools_present_in_all_profiles(self):
        """General tools should appear in every profile."""
        general_tools = {t["name"] for t in TOOL_DEFINITIONS if t["category"] == "general"}
        assert len(general_tools) > 0

        for profile in TOOL_PROFILES:
            tools = get_anthropic_tools(tool_profile=profile)
            names = {t["name"] for t in tools}
            for gen_tool in general_tools:
                assert gen_tool in names, (
                    f"General tool {gen_tool} missing from profile {profile}"
                )
