"""Mocked HTTP tests for the HGNC gene-group executor tools.

Mirror the offline mocking style of tests/test_tools.py::TestSearchMGI: build a
ToolExecutor, patch self.executor.client.get with an AsyncMock returning a
MagicMock response (.status_code/.json()/.text), and assert no real network is
touched. The unresolved/mapping shapes here match the genetics-results-api
/v1/gene_group/members and /v1/gene/normalize endpoints.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _mock_response(status_code: int = 200, json_data: dict | None = None, text: str = ""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.json = MagicMock(return_value=json_data or {})
    return resp


@pytest.mark.asyncio
class TestGetGeneGroupMembers:
    @pytest.fixture(autouse=True)
    async def setup_executor(self):
        from genetics_mcp_server.tools.executor import ToolExecutor

        self.executor = ToolExecutor()
        yield
        await self.executor.close()

    async def test_success_by_group_id(self):
        json_data = {
            "group_id": 588,
            "group_name": "Solute carriers",
            "count": 2,
            "members": [
                {"hgnc_id": "HGNC:10921", "symbol": "SLC2A1"},
                {"hgnc_id": "HGNC:11005", "symbol": "SLC2A4"},
            ],
        }
        mock = _mock_response(json_data=json_data)
        with patch.object(
            self.executor.client, "get", new_callable=AsyncMock, return_value=mock
        ) as mock_get:
            result = await self.executor.get_gene_group_members(group_id=588)

        assert result["success"] is True
        assert result["group_id"] == 588
        assert result["group_name"] == "Solute carriers"
        assert result["count"] == 2
        assert result["members"] == json_data["members"]
        # message only present when there are no members
        assert "message" not in result
        # request used the group_id param, not group_name; olfactory excluded by default
        _, kwargs = mock_get.call_args
        assert kwargs["params"] == {"group_id": 588, "exclude_olfactory": True}

    async def test_success_by_group_name(self):
        json_data = {
            "group_id": 588,
            "group_name": "Solute carriers",
            "count": 1,
            "members": [{"hgnc_id": "HGNC:10921", "symbol": "SLC2A1"}],
        }
        mock = _mock_response(json_data=json_data)
        with patch.object(
            self.executor.client, "get", new_callable=AsyncMock, return_value=mock
        ) as mock_get:
            result = await self.executor.get_gene_group_members(
                group_name="Solute carriers"
            )

        assert result["success"] is True
        assert result["count"] == 1
        _, kwargs = mock_get.call_args
        assert kwargs["params"] == {
            "group_name": "Solute carriers",
            "exclude_olfactory": True,
        }

    async def test_exclude_olfactory_false_passed_through(self):
        json_data = {"group_id": 139, "group_name": "GPCRs", "count": 0, "members": []}
        mock = _mock_response(json_data=json_data)
        with patch.object(
            self.executor.client, "get", new_callable=AsyncMock, return_value=mock
        ) as mock_get:
            await self.executor.get_gene_group_members(
                group_id=139, exclude_olfactory=False
            )
        _, kwargs = mock_get.call_args
        assert kwargs["params"] == {"group_id": 139, "exclude_olfactory": False}

    async def test_neither_arg_makes_no_api_call(self):
        with patch.object(
            self.executor.client, "get", new_callable=AsyncMock
        ) as mock_get:
            result = await self.executor.get_gene_group_members()

        assert result["success"] is False
        assert "exactly one" in result["error"]
        mock_get.assert_not_called()

    async def test_both_args_makes_no_api_call(self):
        with patch.object(
            self.executor.client, "get", new_callable=AsyncMock
        ) as mock_get:
            result = await self.executor.get_gene_group_members(
                group_id=588, group_name="Solute carriers"
            )

        assert result["success"] is False
        assert "exactly one" in result["error"]
        mock_get.assert_not_called()

    async def test_unknown_group_404(self):
        mock = _mock_response(status_code=404, text="not found")
        with patch.object(
            self.executor.client, "get", new_callable=AsyncMock, return_value=mock
        ):
            result = await self.executor.get_gene_group_members(group_id=999999)

        assert result["success"] is False
        assert "Unknown gene group" in result["error"]

    async def test_zero_count_returns_message(self):
        json_data = {
            "group_id": 588,
            "group_name": "Solute carriers",
            "count": 0,
            "members": [],
        }
        mock = _mock_response(json_data=json_data)
        with patch.object(
            self.executor.client, "get", new_callable=AsyncMock, return_value=mock
        ):
            result = await self.executor.get_gene_group_members(group_id=588)

        assert result["success"] is True
        assert result["count"] == 0
        assert result["members"] == []
        assert "message" in result


@pytest.mark.asyncio
class TestNormalizeGeneSymbols:
    @pytest.fixture(autouse=True)
    async def setup_executor(self):
        from genetics_mcp_server.tools.executor import ToolExecutor

        self.executor = ToolExecutor()
        yield
        await self.executor.close()

    async def test_success_passthrough(self):
        json_data = {
            "mappings": [
                {"input": "PTPN9", "approved": "PTPN9", "matched_on": "approved"},
                {"input": "p53", "approved": "TP53", "matched_on": "alias"},
            ],
            "unresolved": ["NOTAGENE"],
        }
        mock = _mock_response(json_data=json_data)
        with patch.object(
            self.executor.client, "get", new_callable=AsyncMock, return_value=mock
        ):
            result = await self.executor.normalize_gene_symbols(
                ["PTPN9", "p53", "NOTAGENE"]
            )

        assert result["success"] is True
        assert result["mappings"] == json_data["mappings"]
        assert result["unresolved"] == ["NOTAGENE"]

    async def test_empty_list_makes_no_api_call(self):
        with patch.object(
            self.executor.client, "get", new_callable=AsyncMock
        ) as mock_get:
            result = await self.executor.normalize_gene_symbols([])

        assert result["success"] is False
        mock_get.assert_not_called()

    async def test_whitespace_only_list_makes_no_api_call(self):
        with patch.object(
            self.executor.client, "get", new_callable=AsyncMock
        ) as mock_get:
            result = await self.executor.normalize_gene_symbols(["  ", "", "\t"])

        assert result["success"] is False
        mock_get.assert_not_called()

    async def test_symbols_are_cleaned_before_request(self):
        json_data = {"mappings": [], "unresolved": []}
        mock = _mock_response(json_data=json_data)
        with patch.object(
            self.executor.client, "get", new_callable=AsyncMock, return_value=mock
        ) as mock_get:
            result = await self.executor.normalize_gene_symbols(
                ["  TP53 ", "", "  ", "BRCA1\t", None]
            )

        assert result["success"] is True
        # empty/whitespace entries stripped, surviving symbols trimmed and joined
        _, kwargs = mock_get.call_args
        assert kwargs["params"] == {"symbols": "TP53,BRCA1"}

    async def test_non_200_returns_failure(self):
        mock = _mock_response(status_code=500, text="server error")
        with patch.object(
            self.executor.client, "get", new_callable=AsyncMock, return_value=mock
        ):
            result = await self.executor.normalize_gene_symbols(["TP53"])

        assert result["success"] is False
        assert "500" in result["error"]
