"""Unit tests for literature-search backend selection and result metadata.

Self-contained: the Perplexity and Europe PMC calls are served by an httpx
MockTransport, so no API keys or network access are needed.
"""

import httpx
import pytest

from genetics_mcp_server.llm_service import LLMService
from genetics_mcp_server.tools import ToolExecutor
from genetics_mcp_server.tools.definitions import TOOL_DEFINITIONS
from genetics_mcp_server.tools.executor import _ResilientAsyncClient

PERPLEXITY_RESPONSE = {
    "choices": [{"message": {"content": "Two papers describe the locus."}}],
    "citations": [
        "https://pubmed.ncbi.nlm.nih.gov/34580418/",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC2974578/",
    ],
    "search_results": [
        {
            "title": "Genetic variants associated with platelet count are ...",
            "url": "https://pubmed.ncbi.nlm.nih.gov/34580418/",
            "date": "2021-09-27",
            "snippet": "By analogy with ABCG4, they could function in platelet count regulation.",
        },
        {
            "title": "A second paper",
            "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC2974578/",
            "date": "2010-01-15",
            "snippet": "Snippet text.",
        },
    ],
}

EPMC_RESPONSE = {
    "hitCount": 2,
    "resultList": {
        "result": [
            {
                "pmid": "34580418",
                "doi": "10.1038/s42003-021-02642-9",
                "title": "Genetic variants associated with platelet count",
                "authorString": "Astle WJ, Elding H, Jiang T.",
                "journalTitle": "Commun Biol",
                "pubYear": "2021",
                "abstractText": "Full abstract text.",
                "source": "MED",
            },
            {
                "pmcid": "PMC2974578",
                "title": "A second paper",
                "authorString": "Smith A, Jones B.",
                "journalTitle": "J Test",
                "pubYear": "2010",
                "abstractText": "Second abstract.",
                "source": "PMC",
            },
        ]
    },
}


def _executor_with_transport(handler) -> ToolExecutor:
    """Build an executor whose external calls are served by `handler`."""
    executor = ToolExecutor()
    executor.external_client = _ResilientAsyncClient(
        timeout=5.0, transport=httpx.MockTransport(handler)
    )
    return executor


def _handler(epmc_status: int = 200, perplexity_payload: dict | None = None):
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.perplexity.ai":
            return httpx.Response(
                200, json=perplexity_payload if perplexity_payload is not None else PERPLEXITY_RESPONSE
            )
        if request.url.host == "www.ebi.ac.uk":
            if epmc_status != 200:
                return httpx.Response(epmc_status, text="upstream error")
            return httpx.Response(200, json=EPMC_RESPONSE)
        raise AssertionError(f"unexpected host: {request.url.host}")

    return handle


class TestBackendSelection:
    async def test_defaults_to_perplexity(self, monkeypatch):
        """With nothing configured the perplexity backend is used, not europepmc."""
        monkeypatch.delenv("LITERATURE_SEARCH_BACKEND", raising=False)
        monkeypatch.setenv("PERPLEXITY_API_KEY", "test-key")

        executor = _executor_with_transport(_handler())
        try:
            result = await executor.search_scientific_literature("platelet count", max_results=5)
        finally:
            await executor.close()

        assert result["success"] is True
        assert result["backend"] == "perplexity"
        assert result["source"] == "perplexity"

    async def test_env_var_selects_europepmc(self, monkeypatch):
        monkeypatch.setenv("LITERATURE_SEARCH_BACKEND", "europepmc")

        executor = _executor_with_transport(_handler())
        try:
            result = await executor.search_scientific_literature("platelet count", max_results=5)
        finally:
            await executor.close()

        assert result["success"] is True
        assert result["backend"] == "europepmc"

    async def test_argument_overrides_env_var(self, monkeypatch):
        monkeypatch.setenv("LITERATURE_SEARCH_BACKEND", "perplexity")

        executor = _executor_with_transport(_handler())
        try:
            result = await executor.search_scientific_literature(
                "platelet count", max_results=5, backend="europepmc"
            )
        finally:
            await executor.close()

        assert result["backend"] == "europepmc"

    async def test_missing_key_reports_backend(self, monkeypatch):
        monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
        monkeypatch.delenv("LITERATURE_SEARCH_BACKEND", raising=False)

        executor = _executor_with_transport(_handler())
        try:
            result = await executor.search_scientific_literature("platelet count")
        finally:
            await executor.close()

        assert result["success"] is False
        assert result["backend"] == "perplexity"


class TestPerplexityMetadata:
    async def test_hits_carry_bibliographic_metadata(self, monkeypatch):
        """Titles come from search_results; authors/journal are hydrated from Europe PMC."""
        monkeypatch.setenv("PERPLEXITY_API_KEY", "test-key")

        executor = _executor_with_transport(_handler())
        try:
            result = await executor.search_scientific_literature(
                "platelet count", max_results=5, backend="perplexity"
            )
        finally:
            await executor.close()

        by_pmid = result["results"][0]
        assert by_pmid["title"] == "Genetic variants associated with platelet count"
        assert by_pmid["authors"] == "Astle WJ, Elding H, Jiang T."
        assert by_pmid["journal"] == "Commun Biol"
        assert by_pmid["year"] == "2021"
        assert by_pmid["doi"] == "10.1038/s42003-021-02642-9"
        assert by_pmid["source"] == "perplexity"
        assert by_pmid["metadata_source"] == "europepmc"

        # the PMC-only hit is matched on its PMCID, which has no PMID in the URL
        by_pmcid = result["results"][1]
        assert by_pmcid["authors"] == "Smith A, Jones B."
        assert by_pmcid["journal"] == "J Test"

    async def test_hydration_failure_leaves_perplexity_metadata(self, monkeypatch):
        """A Europe PMC outage must not fail the search — titles/years still come through."""
        monkeypatch.setenv("PERPLEXITY_API_KEY", "test-key")

        executor = _executor_with_transport(_handler(epmc_status=503))
        try:
            result = await executor.search_scientific_literature(
                "platelet count", max_results=5, backend="perplexity"
            )
        finally:
            await executor.close()

        assert result["success"] is True
        record = result["results"][0]
        assert record["title"] == "Genetic variants associated with platelet count are ..."
        assert record["year"] == "2021"
        assert record["abstract"].startswith("By analogy")
        assert record["authors"] == ""
        assert record["metadata_source"] == "perplexity"

    async def test_falls_back_to_citations_without_search_results(self, monkeypatch):
        """Older Perplexity responses carry only a URL list."""
        monkeypatch.setenv("PERPLEXITY_API_KEY", "test-key")
        payload = {k: v for k, v in PERPLEXITY_RESPONSE.items() if k != "search_results"}

        executor = _executor_with_transport(_handler(perplexity_payload=payload))
        try:
            result = await executor.search_scientific_literature(
                "platelet count", max_results=5, backend="perplexity"
            )
        finally:
            await executor.close()

        assert result["returned"] == 2
        record = result["results"][0]
        assert record["pmid"] == "34580418"
        # no title in the response, so it is hydrated from Europe PMC
        assert record["title"] == "Genetic variants associated with platelet count"


class TestBackendIsCallerControlled:
    """The backend is the user's setting; the model cannot select or influence it."""

    @pytest.fixture
    def service(self, monkeypatch):
        service = LLMService.__new__(LLMService)
        service.executor = _executor_with_transport(_handler())
        service.subagent_service = None
        monkeypatch.setenv("PERPLEXITY_API_KEY", "test-key")
        return service

    def test_tool_exposes_no_backend_parameter(self):
        """A backend argument in the schema is what let the model ask for europepmc."""
        tool = next(
            t for t in TOOL_DEFINITIONS if t["name"] == "search_scientific_literature"
        )
        assert "backend" not in tool["parameters"]

    async def test_model_supplied_backend_is_discarded(self, service):
        """A hallucinated backend argument must not reach the executor."""
        result = await service._execute_tool(
            "search_scientific_literature",
            {"query": "platelet count", "backend": "europepmc"},
            literature_backend="perplexity",
        )
        await service.executor.close()

        assert result["backend"] == "perplexity"

    async def test_user_choice_selects_europepmc(self, service):
        result = await service._execute_tool(
            "search_scientific_literature",
            {"query": "platelet count"},
            literature_backend="europepmc",
        )
        await service.executor.close()

        assert result["backend"] == "europepmc"

    async def test_falls_back_to_perplexity_without_user_choice(self, service, monkeypatch):
        monkeypatch.delenv("LITERATURE_SEARCH_BACKEND", raising=False)

        result = await service._execute_tool(
            "search_scientific_literature",
            {"query": "platelet count", "backend": "europepmc"},
            literature_backend=None,
        )
        await service.executor.close()

        assert result["backend"] == "perplexity"
