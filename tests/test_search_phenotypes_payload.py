"""What a search_phenotypes hit carries into the conversation.

The index returns ranking diagnostics on every hit — the strings it matched on and three
scores — that the model never acts on and that doubled the size of the largest replayed
tool output in production. They are dropped at the executor, so every consumer of the
tool (chat, MCP, the subagent) gets the same shape.
"""

import httpx
import pytest

from genetics_mcp_server.tools import ToolExecutor

HIT = {
    "type": "phenotype",
    "code": "AB1_WHOOPCOUGH",
    "name": "Whooping cough",
    "resource": "finngen",
    "data_type": "gwas",
    "sample_size": 430255,
    "n_cases": 242,
    "n_controls": 430013,
    "has_summary_stats": True,
    "has_credible_sets": True,
    "search_strings": ["ab1_whoopcough", "whooping cough"],
    "match_type": "exact",
    "match_score": 92.85714285714286,
    "rank_score": 1142.857142857143,
    "matched_key": "Whooping cough",
}


@pytest.mark.asyncio
async def test_ranking_diagnostics_are_dropped_from_every_hit():
    def serve(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/v1/search")
        return httpx.Response(200, json=[HIT, {**HIT, "code": "OTHER"}])

    executor = ToolExecutor()
    try:
        await executor.client.aclose()
        executor.client = httpx.AsyncClient(transport=httpx.MockTransport(serve))
        result = await executor.search_phenotypes("whooping cough,pertussis")
    finally:
        await executor.close()

    assert result["success"] is True
    assert [hit["code"] for hit in result["results"]] == ["AB1_WHOOPCOUGH", "OTHER"]
    for hit in result["results"]:
        assert not {"search_strings", "match_score", "rank_score", "matched_key"} & hit.keys()
        # what the model does act on stays: the identity, the counts, what data exists
        assert hit["match_type"] == "exact"
        assert hit["n_cases"] == 242 and hit["has_credible_sets"] is True
