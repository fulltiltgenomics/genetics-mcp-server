"""Tests for myvariant.info integration (HGVS conversion, API calls, response flattening)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from genetics_mcp_server.tools.executor import ToolExecutor


class TestVariantToHgvs:
    """Unit tests for _variant_to_hgvs conversion."""

    def test_snv(self):
        assert ToolExecutor._variant_to_hgvs("1:55051215:G:A") == "chr1:g.55051215G>A"

    def test_snv_with_chr_prefix(self):
        assert ToolExecutor._variant_to_hgvs("chr7:117559590:A:G") == "chr7:g.117559590A>G"

    def test_snv_chrx(self):
        assert ToolExecutor._variant_to_hgvs("X:12345:C:T") == "chrX:g.12345C>T"

    def test_deletion_single_base(self):
        # ref=AT, alt=A → deletion of T at pos+1
        assert ToolExecutor._variant_to_hgvs("1:100:AT:A") == "chr1:g.101del"

    def test_deletion_multiple_bases(self):
        # ref=ATCG, alt=A → deletion of TCG at pos+1 to pos+3
        assert ToolExecutor._variant_to_hgvs("1:100:ATCG:A") == "chr1:g.101_103del"

    def test_insertion_single_base(self):
        # ref=A, alt=AT → insertion of T after pos
        assert ToolExecutor._variant_to_hgvs("1:100:A:AT") == "chr1:g.100_101insT"

    def test_insertion_multiple_bases(self):
        # ref=A, alt=ATCG → insertion of TCG after pos
        assert ToolExecutor._variant_to_hgvs("1:100:A:ATCG") == "chr1:g.100_101insTCG"

    def test_mnv_delins(self):
        # ref=AT, alt=GC → complex substitution
        assert ToolExecutor._variant_to_hgvs("1:100:AT:GC") == "chr1:g.100_101delinsGC"

    def test_single_base_delins(self):
        # ref=A, alt=TG → single base replaced by two
        assert ToolExecutor._variant_to_hgvs("1:100:A:TG") == "chr1:g.100delinsTG"

    def test_invalid_format(self):
        with pytest.raises(ValueError, match="Invalid variant format"):
            ToolExecutor._variant_to_hgvs("invalid")

    def test_various_separators(self):
        # the regex splits on : | - _
        assert ToolExecutor._variant_to_hgvs("1:100:A:G") == "chr1:g.100A>G"


class TestFlattenMyvariantResult:
    """Tests for _flatten_myvariant_result response flattening."""

    def test_clinvar_single_rcv(self):
        data = {
            "clinvar": {
                "rcv": {
                    "clinical_significance": "Pathogenic",
                    "preferred_name": "BRCA1 c.5266dupC",
                },
                "review": {"review_status": "criteria provided, multiple submitters, no conflicts"},
                "variant_id": "12345",
            }
        }
        result = ToolExecutor._flatten_myvariant_result(data)
        assert result["clinvar"]["clinical_significance"] == ["Pathogenic"]
        assert result["clinvar"]["conditions"] == ["BRCA1 c.5266dupC"]
        assert "criteria provided" in result["clinvar"]["review_status"]
        assert result["clinvar"]["variant_id"] == "12345"

    def test_clinvar_multiple_rcv(self):
        data = {
            "clinvar": {
                "rcv": [
                    {"clinical_significance": "Pathogenic", "preferred_name": "Breast cancer"},
                    {
                        "clinical_significance": "Likely pathogenic",
                        "preferred_name": "Ovarian cancer",
                    },
                ],
                "review": {"review_status": "reviewed by expert panel"},
            }
        }
        result = ToolExecutor._flatten_myvariant_result(data)
        assert set(result["clinvar"]["clinical_significance"]) == {
            "Pathogenic",
            "Likely pathogenic",
        }
        assert set(result["clinvar"]["conditions"]) == {"Breast cancer", "Ovarian cancer"}

    def test_cadd_comes_from_dbnsfp_on_hg38(self):
        data = {"dbnsfp": {"cadd": {"phred": 23.0, "raw_score": 3.6, "raw_rankscore": 0.5}}}
        result = ToolExecutor._flatten_myvariant_result(data)
        assert result["cadd"] == {"phred": 23.0, "raw_score": 3.6}

    def test_top_level_cadd_wins_when_an_index_has_it(self):
        data = {
            "cadd": {"phred": 25.3, "rawscore": 4.5, "consequence": "NON_SYNONYMOUS"},
            "dbnsfp": {"cadd": {"phred": 1.0}},
        }
        result = ToolExecutor._flatten_myvariant_result(data)
        assert result["cadd"] == {"phred": 25.3, "raw_score": 4.5, "consequence": "NON_SYNONYMOUS"}

    def test_dbnsfp_predictions(self):
        # shapes as the hg38 index answers them: polyphen2 nested per model, metalr with
        # `rankscore`, per-transcript lists
        data = {
            "dbnsfp": {
                "sift": {"score": 0.01, "pred": "D", "converted_rankscore": 0.9},
                "polyphen2": {
                    "hdiv": {"score": 0.971, "pred": "D", "rankscore": 0.57},
                    "hvar": {"score": 0.943, "pred": "D", "rankscore": 0.68},
                },
                "metalr": {"score": 0.0855, "pred": "T", "rankscore": 0.33},
                "revel": {"score": 0.931, "rankscore": 0.98},
                "alphamissense": {"score": [0.98, 0.99], "pred": ["P", "P"], "rankscore": 0.97},
                "mutationtaster": {"score": [1.0, 1.0], "pred": ["N", "N"]},
                "genename": ["PCSK9", "PCSK9"],
            }
        }
        preds = ToolExecutor._flatten_myvariant_result(data)["functional_predictions"]
        assert preds["sift"] == {"score": 0.01, "prediction": "D", "rankscore": 0.9}
        assert preds["polyphen2_hdiv"] == {"score": 0.971, "prediction": "D", "rankscore": 0.57}
        assert preds["polyphen2_hvar"]["score"] == 0.943
        assert preds["metalr"]["rankscore"] == 0.33
        assert preds["revel"] == {"score": 0.931, "rankscore": 0.98}
        assert preds["alphamissense"] == {
            "score": [0.98, 0.99],
            "prediction": "P",
            "rankscore": 0.97,
        }
        assert preds["mutationtaster"] == {"score": 1.0, "prediction": "N"}
        assert ToolExecutor._flatten_myvariant_result(data)["gene"] == "PCSK9"

    def test_a_variant_dbnsfp_does_not_cover_has_no_scores(self):
        result = ToolExecutor._flatten_myvariant_result({"dbsnp": {"rsid": "rs12740374"}})
        assert result == {"rsid": "rs12740374"}

    def test_dbsnp_rsid(self):
        data = {"dbsnp": {"rsid": "rs429358"}}
        result = ToolExecutor._flatten_myvariant_result(data)
        assert result["rsid"] == "rs429358"

    def test_empty_data(self):
        result = ToolExecutor._flatten_myvariant_result({})
        assert result == {}


@pytest.mark.asyncio
class TestGetMyvariantAnnotations:
    """Mocked HTTP tests for get_myvariant_annotations."""

    @pytest.fixture(autouse=True)
    async def setup_executor(self):
        self.executor = ToolExecutor()
        yield
        await self.executor.close()

    async def test_single_variant_success(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "clinvar": {
                "rcv": {"clinical_significance": "Benign", "preferred_name": "not specified"},
                "review": {"review_status": "no assertion criteria provided"},
            },
            "cadd": {"phred": 12.5, "rawscore": 2.1},
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            self.executor.external_client, "get", new_callable=AsyncMock, return_value=mock_response
        ):
            result = await self.executor.get_myvariant_annotations(variant="1:55051215:G:A")

        assert result["success"] is True
        assert result["variant"] == "1:55051215:G:A"
        assert result["found"] is True
        assert result["annotations"]["clinvar"]["clinical_significance"] == ["Benign"]
        assert result["annotations"]["cadd"]["phred"] == 12.5

    async def test_single_variant_not_found(self):
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch.object(
            self.executor.external_client, "get", new_callable=AsyncMock, return_value=mock_response
        ):
            result = await self.executor.get_myvariant_annotations(variant="1:99999999:A:G")

        assert result["success"] is True
        assert result["found"] is False
        assert result["annotations"] == {}

    async def test_rate_limit(self):
        mock_response = MagicMock()
        mock_response.status_code = 429

        with patch.object(
            self.executor.external_client, "get", new_callable=AsyncMock, return_value=mock_response
        ):
            result = await self.executor.get_myvariant_annotations(variant="1:55051215:G:A")

        assert result["success"] is False
        assert "rate limit" in result["error"]

    async def test_timeout(self):
        import httpx

        with patch.object(
            self.executor.external_client,
            "get",
            new_callable=AsyncMock,
            side_effect=httpx.TimeoutException("timed out"),
        ):
            result = await self.executor.get_myvariant_annotations(variant="1:55051215:G:A")

        assert result["success"] is False
        assert "timed out" in result["error"]

    async def test_batch_success(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "_id": "chr1:g.55051215G>A",
                "clinvar": {
                    "rcv": {"clinical_significance": "Pathogenic", "preferred_name": "test"}
                },
            },
            {"_id": "chr7:g.117559590A>G", "notfound": True},
        ]
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            self.executor.external_client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await self.executor.get_myvariant_annotations(
                variants=["1:55051215:G:A", "7:117559590:A:G"]
            )

        assert result["success"] is True
        assert result["total_queried"] == 2
        assert result["total_found"] == 1
        assert result["annotations"]["1:55051215:G:A"]["found"] is True
        assert result["annotations"]["7:117559590:A:G"]["found"] is False

    async def test_batch_too_many_variants(self):
        result = await self.executor.get_myvariant_annotations(variants=["1:100:A:G"] * 1001)
        assert result["success"] is False
        assert "1000" in result["error"]

    async def test_no_params(self):
        result = await self.executor.get_myvariant_annotations()
        assert result["success"] is False
        assert "variant" in result["error"]

    async def test_invalid_variant_format(self):
        result = await self.executor.get_myvariant_annotations(variant="invalid")
        assert result["success"] is False
        assert "Invalid variant format" in result["error"]
