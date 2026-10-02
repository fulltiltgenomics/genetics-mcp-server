"""get_variant_annotations names the release its rows come from.

Self-contained: the results-api response is stubbed, so no running API is needed.
"""

from unittest.mock import AsyncMock

import httpx
import pytest

from genetics_mcp_server.tools import ToolExecutor

ROW = {"variant": "19:55025227:G:A", "AF": "0.87897", "AC_Het": "110368", "AC_Hom": "803532"}


def _response(headers: dict[str, str]) -> httpx.Response:
    return httpx.Response(
        200, json=[ROW], headers=headers, request=httpx.Request("GET", "http://unused.test")
    )


@pytest.mark.parametrize("batch", [False, True], ids=["get", "post"])
async def test_version_is_relayed_from_the_response_header(batch):
    executor = ToolExecutor(api_base_url="http://unused.test")
    stub = AsyncMock(return_value=_response({"X-Dataset-Version": "R14"}))
    executor.client.get = executor.client.post = stub
    try:
        if batch:
            result = await executor.get_variant_annotations(variants=[ROW["variant"]])
        else:
            result = await executor.get_variant_annotations(variant=ROW["variant"])

        assert result["success"] is True
        assert result["version"] == "R14"
        assert result["source"] == "finngen"
    finally:
        await executor.close()


async def test_no_version_is_invented_for_a_results_api_that_sends_none():
    executor = ToolExecutor(api_base_url="http://unused.test")
    executor.client.get = AsyncMock(return_value=_response({}))
    try:
        result = await executor.get_variant_annotations(variant=ROW["variant"])

        assert result["success"] is True
        assert "version" not in result
    finally:
        await executor.close()
