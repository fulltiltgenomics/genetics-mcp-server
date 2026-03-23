"""Tests for variant list analysis tool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from genetics_mcp_server.tools.executor import ToolExecutor


class TestParseVariantList:
    """Tests for variant list input parsing."""

    def test_simple_variant_list(self):
        text = "1:154453788:C:T\n19:44908684:T:C"
        result = ToolExecutor._parse_variant_list(text)
        assert len(result) == 2
        assert result[0]["variant"] == "1:154453788:C:T"
        assert result[1]["variant"] == "19:44908684:T:C"
        assert result[0]["beta"] is None

    def test_with_header_and_stats(self):
        text = "variant\tbeta\tse\tpvalue\n1:154453788:C:T\t0.15\t0.02\t1e-12\n19:44908684:T:C\t-0.08\t0.01\t5e-8"
        result = ToolExecutor._parse_variant_list(text)
        assert len(result) == 2
        assert result[0]["beta"] == 0.15
        assert result[0]["se"] == 0.02
        assert result[0]["pvalue"] == 1e-12
        assert result[1]["beta"] == -0.08

    def test_positional_stats_without_header(self):
        text = "1:154453788:C:T\t0.15\t0.02\t1e-12"
        result = ToolExecutor._parse_variant_list(text)
        assert len(result) == 1
        assert result[0]["beta"] == 0.15
        assert result[0]["se"] == 0.02

    def test_chr_prefix_stripped(self):
        text = "chr1:154453788:C:T\nchr19:44908684:T:C"
        result = ToolExecutor._parse_variant_list(text)
        assert result[0]["variant"] == "1:154453788:C:T"
        assert result[1]["variant"] == "19:44908684:T:C"

    def test_dash_separated_format(self):
        text = "1-154453788-C-T\n19-44908684-T-C"
        result = ToolExecutor._parse_variant_list(text)
        assert len(result) == 2
        assert result[0]["variant"] == "1:154453788:C:T"

    def test_comma_separated(self):
        text = "variant,beta,se,pvalue\n1:154453788:C:T,0.15,0.02,1e-12"
        result = ToolExecutor._parse_variant_list(text)
        assert len(result) == 1
        assert result[0]["beta"] == 0.15

    def test_empty_input(self):
        assert ToolExecutor._parse_variant_list("") == []
        assert ToolExecutor._parse_variant_list("  \n  ") == []

    def test_invalid_variants_skipped(self):
        text = "1:154453788:C:T\nnot_a_variant\n19:44908684:T:C"
        result = ToolExecutor._parse_variant_list(text)
        assert len(result) == 2

    def test_x_chromosome(self):
        text = "X:12345678:A:G"
        result = ToolExecutor._parse_variant_list(text)
        assert len(result) == 1
        assert result[0]["variant"] == "X:12345678:A:G"

    def test_case_insensitive_header(self):
        text = "VARIANT\tBETA\tSE\tPVALUE\n1:154453788:C:T\t0.1\t0.02\t1e-8"
        result = ToolExecutor._parse_variant_list(text)
        assert result[0]["beta"] == 0.1

    def test_alternative_header_names(self):
        text = "snp\teffect\tstderr\tp\n1:154453788:C:T\t0.1\t0.02\t1e-8"
        result = ToolExecutor._parse_variant_list(text)
        assert result[0]["beta"] == 0.1
        assert result[0]["se"] == 0.02
        assert result[0]["pvalue"] == 1e-8


class TestAnalyzeVariantList:
    """Tests for the analyze_variant_list aggregation logic."""

    def _mock_cs_response(self, variant, results):
        """Build a mock credible set API response."""
        return {"variant": variant, "results": results}

    @pytest.mark.asyncio
    async def test_basic_aggregation(self):
        """Test GWAS phenotype counting."""
        executor = ToolExecutor(api_base_url="http://fake")

        # mock the httpx client
        mock_client = AsyncMock()
        executor.client = mock_client

        v1 = "1:100:A:G"
        v2 = "2:200:C:T"

        # credible set responses
        cs_responses = {
            v1: [
                {"data_type": "GWAS", "trait": "T2D", "beta": 0.1, "gene_most_severe": "G1", "cell_type": None},
                {"data_type": "eQTL", "trait": "expr", "beta": 0.2, "gene_most_severe": "SORT1", "cell_type": "liver"},
            ],
            v2: [
                {"data_type": "GWAS", "trait": "T2D", "beta": -0.05, "gene_most_severe": "G2", "cell_type": None},
                {"data_type": "pQTL", "trait": "prot", "beta": 0.3, "gene_most_severe": "PCSK9", "cell_type": "plasma"},
            ],
        }

        gene_responses = {
            v1: [{"name": "IL6R", "distance": 0}],
            v2: [{"name": "APOE", "distance": 5000}],
        }

        async def mock_get(url, **kwargs):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            if "credible_sets_by_variant" in url:
                vid = url.split("/")[-1]
                mock_resp.json.return_value = cs_responses.get(vid, [])
            elif "nearest_genes" in url:
                vid = url.split("/")[-1].split("?")[0]
                mock_resp.json.return_value = gene_responses.get(vid, [])
            return mock_resp

        mock_client.get = mock_get

        # mock lookup_phenotype_names
        with patch.object(
            executor, "lookup_phenotype_names",
            new_callable=AsyncMock,
            return_value={"success": True, "names": {"T2D": "Type 2 diabetes"}},
        ):
            result = await executor.analyze_variant_list(f"{v1}\n{v2}")

        assert result["success"] is True
        assert result["n_variants"] == 2
        assert result["n_variants_with_cs"] == 2
        assert result["input_has_betas"] is False

        # T2D should have 2 variants
        gwas = {p["trait"]: p for p in result["gwas_phenotypes"]}
        assert gwas["T2D"]["n_variants"] == 2
        assert gwas["T2D"]["name"] == "Type 2 diabetes"

        # PCSK9 pQTL from one variant
        pqtl = {g["gene"]: g for g in result["pqtl_genes"]}
        assert pqtl["PCSK9"]["n_variants"] == 1

        # eQTL: SORT1/liver from one variant
        assert len(result["eqtl_genes"]) == 1
        assert result["eqtl_genes"][0]["gene"] == "SORT1"
        assert result["eqtl_genes"][0]["tissue"] == "liver"

        # tissue enrichment
        tissues = {t["tissue"]: t for t in result["tissue_enrichment"]}
        assert tissues["liver"]["n_eqtl_variants"] == 1

        # nearest genes
        genes = {g["variant"]: g for g in result["variant_genes"]}
        assert genes[v1]["nearest_gene"] == "IL6R"
        assert genes[v2]["nearest_gene"] == "APOE"

    @pytest.mark.asyncio
    async def test_direction_consistency(self):
        """Test that direction consistency is computed when betas provided."""
        executor = ToolExecutor(api_base_url="http://fake")
        mock_client = AsyncMock()
        executor.client = mock_client

        v1 = "1:100:A:G"

        async def mock_get(url, **kwargs):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            if "credible_sets_by_variant" in url:
                mock_resp.json.return_value = [
                    {"data_type": "GWAS", "trait": "T2D", "beta": 0.1, "gene_most_severe": "G1", "cell_type": None},
                ]
            elif "nearest_genes" in url:
                mock_resp.json.return_value = [{"name": "G1", "distance": 0}]
            return mock_resp

        mock_client.get = mock_get

        with patch.object(
            executor, "lookup_phenotype_names",
            new_callable=AsyncMock,
            return_value={"success": True, "names": {"T2D": "Type 2 diabetes"}},
        ):
            # input beta is positive, CS beta is positive → consistent
            result = await executor.analyze_variant_list(f"variant\tbeta\n{v1}\t0.05")

        assert result["success"] is True
        assert result["input_has_betas"] is True
        gwas = result["gwas_phenotypes"][0]
        assert gwas["n_consistent"] == 1
        assert gwas["n_inconsistent"] == 0

    @pytest.mark.asyncio
    async def test_empty_input(self):
        executor = ToolExecutor(api_base_url="http://fake")
        result = await executor.analyze_variant_list("")
        assert result["success"] is False
        assert "No valid variants" in result["error"]

    @pytest.mark.asyncio
    async def test_api_errors_handled(self):
        """Variants that fail API calls should not crash the analysis."""
        executor = ToolExecutor(api_base_url="http://fake")
        mock_client = AsyncMock()
        executor.client = mock_client

        async def mock_get(url, **kwargs):
            mock_resp = MagicMock()
            mock_resp.status_code = 500
            mock_resp.json.return_value = []
            return mock_resp

        mock_client.get = mock_get

        with patch.object(
            executor, "lookup_phenotype_names",
            new_callable=AsyncMock,
            return_value={"success": True, "names": {}},
        ):
            result = await executor.analyze_variant_list("1:100:A:G")

        assert result["success"] is True
        assert result["n_variants"] == 1
        assert result["n_variants_with_cs"] == 0
