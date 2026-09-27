"""Unit tests for literature-search backend selection and result metadata.

Self-contained: the Perplexity and Europe PMC calls are served by an httpx
MockTransport, so no API keys or network access are needed.
"""

import asyncio
import json
import time
from pathlib import Path

import httpx
import pytest

from genetics_mcp_server.llm_service import LLMService
from genetics_mcp_server.tools import ServerToolExecutor, orchestration
from genetics_mcp_server.tools.definitions import TOOL_DEFINITIONS
from genetics_mcp_server.tools.executor import _ResilientAsyncClient
from genetics_mcp_server.tools.orchestration import (
    _EPMC_SUBJECTS_CAP,
    _HYDRATION_LOOKUP_LIMIT,
    _literature_ids_from_url,
)

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
                # core fields as Europe PMC returned them for this PMID on 2026-09-27
                "pubTypeList": {"pubType": ["Meta-Analysis", "research-article", "Journal Article"]},
                "meshHeadingList": {
                    "meshHeading": [
                        {"majorTopic_YN": "N", "descriptorName": name}
                        for name in (
                            "Humans", "Platelet Count", "Phenotype", "Quantitative Trait Loci",
                            "Female", "Male", "Genetic Variation", "Biomarkers",
                        )
                    ]
                },
                "citedByCount": 16,
                "publicationStatus": "epublish",
            },
            {
                "pmcid": "PMC2974578",
                "title": "A second paper",
                "authorString": "Smith A, Jones B.",
                "journalTitle": "J Test",
                "pubYear": "2010",
                "abstractText": "Second abstract.",
                "source": "PMC",
                # core fields as Europe PMC returned them for PMC2974578 on 2026-09-27: 14
                # descriptors, so the subjects cap applies
                "pubTypeList": {
                    "pubType": [
                        "Research Support, Non-U.S. Gov't", "review-article", "Review",
                        "Journal Article", "Research Support, N.I.H., Extramural",
                    ]
                },
                "meshHeadingList": {
                    "meshHeading": [
                        {"majorTopic_YN": "N", "descriptorName": name}
                        for name in (
                            "Blood Platelets", "Chromosomes, Human, Pair 12", "Humans",
                            "Platelet Count", "Platelet Function Tests", "Risk Factors",
                            "Genomics", "Thrombopoiesis", "Cell Size", "Platelet Aggregation",
                            "Polymorphism, Single Nucleotide", "Quantitative Trait Loci",
                            "Coronary Artery Disease", "Genome-Wide Association Study",
                        )
                    ]
                },
                "citedByCount": 57,
                "publicationStatus": "ppublish",
            },
        ]
    },
}


def _executor_with_transport(handler) -> ServerToolExecutor:
    """Build an executor whose external calls are served by `handler`."""
    executor = ServerToolExecutor()
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


class TestCoreFields:
    """Europe PMC core fields a reader can weigh a hit by, on both backends."""

    # a live PPR record (2026-09-27): no MeSH, no publicationStatus
    PREPRINT = {
        "id": "PPR822204",
        "source": "PPR",
        "doi": "10.1101/2024.03.14.584883",
        "title": "Transcriptomic and epigenomic consequences of heterozygous loss",
        "pubTypeList": {"pubType": ["Preprint"]},
        "citedByCount": 1,
    }

    async def test_europepmc_record_keeps_core_fields(self, monkeypatch):
        monkeypatch.setenv("LITERATURE_SEARCH_BACKEND", "europepmc")
        executor = _executor_with_transport(_handler())
        try:
            result = await executor.search_scientific_literature("platelet count", max_results=5)
        finally:
            await executor.close()

        record = result["results"][0]
        assert record["pub_types"] == ["Meta-Analysis", "research-article", "Journal Article"]
        assert record["subjects"][:2] == ["Humans", "Platelet Count"]
        assert record["cited_by"] == 16
        assert record["publication_status"] == "epublish"
        assert record["is_preprint"] is False

    def test_subjects_are_capped_in_indexer_order(self):
        record = ServerToolExecutor()._format_literature_results(
            [EPMC_RESPONSE["resultList"]["result"][1]]
        )[0]
        assert len(record["subjects"]) == _EPMC_SUBJECTS_CAP < 14
        assert record["subjects"][:3] == ["Blood Platelets", "Chromosomes, Human, Pair 12", "Humans"]

    def test_missing_fields_are_empty_not_errors(self):
        preprint, bare = ServerToolExecutor()._format_literature_results(
            [self.PREPRINT, {"title": "t", "citedByCount": "n/a", "meshHeadingList": None}]
        )
        assert preprint["is_preprint"] is True
        assert preprint["pub_types"] == ["Preprint"]
        assert preprint["subjects"] == []
        assert preprint["cited_by"] == 1
        assert preprint["publication_status"] is None
        assert bare["pub_types"] == [] and bare["subjects"] == []
        assert bare["cited_by"] is None

    async def test_hydrated_perplexity_hit_carries_core_fields(self, monkeypatch):
        monkeypatch.setenv("PERPLEXITY_API_KEY", "test-key")
        executor = _executor_with_transport(_handler())
        try:
            result = await executor.search_scientific_literature(
                "platelet count", max_results=5, backend="perplexity"
            )
        finally:
            await executor.close()

        by_pmid, by_pmcid = result["results"]
        assert by_pmid["pub_types"] == ["Meta-Analysis", "research-article", "Journal Article"]
        assert by_pmid["cited_by"] == 16
        assert by_pmid["publication_status"] == "epublish"
        assert by_pmcid["cited_by"] == 57
        assert len(by_pmcid["subjects"]) == _EPMC_SUBJECTS_CAP

    async def test_unhydrated_perplexity_hit_has_no_core_fields(self, monkeypatch):
        """Absent means unknown; an empty list would read as Europe PMC saying so."""
        monkeypatch.setenv("PERPLEXITY_API_KEY", "test-key")
        executor = _executor_with_transport(_handler(epmc_status=503))
        try:
            result = await executor.search_scientific_literature(
                "platelet count", max_results=5, backend="perplexity"
            )
        finally:
            await executor.close()

        for field in ("pub_types", "subjects", "cited_by", "publication_status"):
            assert field not in result["results"][0]

    async def test_biorxiv_hit_matched_to_non_preprint_record_stays_preprint(self):
        """is_preprint is set from the URL and a match may only raise it, never lower it."""
        doi = "10.1101/2024.03.14.584883"

        def handle(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json=_title_page([_epmc_record("MED", "t", "2024", doi, "Author A.")]),
            )

        hit = {"url": f"https://www.biorxiv.org/content/{doi}v1.full", "title": "t"}
        record = (await _hydrate_with(handle, [hit]))[0]

        assert record["metadata_source"] == "europepmc"
        assert record["doi"] == doi
        assert record["is_preprint"] is True


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
            advertised_tools={"search_scientific_literature"},
        )
        await service.executor.close()

        assert result["backend"] == "perplexity"

    async def test_user_choice_selects_europepmc(self, service):
        result = await service._execute_tool(
            "search_scientific_literature",
            {"query": "platelet count"},
            literature_backend="europepmc",
            advertised_tools={"search_scientific_literature"},
        )
        await service.executor.close()

        assert result["backend"] == "europepmc"

    async def test_falls_back_to_perplexity_without_user_choice(self, service, monkeypatch):
        monkeypatch.delenv("LITERATURE_SEARCH_BACKEND", raising=False)

        result = await service._execute_tool(
            "search_scientific_literature",
            {"query": "platelet count", "backend": "europepmc"},
            literature_backend=None,
            advertised_tools={"search_scientific_literature"},
        )
        await service.executor.close()

        assert result["backend"] == "perplexity"


# live Europe PMC responses captured for these search_results; the batch response is the
# real one, so it omits the PMC-hosted preprint that only a single-id query returns
HYDRATION_FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "literature" / "perplexity_hydration.json").read_text()
)


def _fixture_handler(sent: list[str]):
    by_query = HYDRATION_FIXTURE["europepmc_by_query"]

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "www.ebi.ac.uk", request.url.host
        query = request.url.params["query"]
        sent.append(query)
        return httpx.Response(
            200, json=by_query.get(query, {"hitCount": 0, "resultList": {"result": []}})
        )

    return handle


async def _hydrate(search_results: list[dict], sent: list[str]) -> list[dict]:
    executor = _executor_with_transport(_fixture_handler(sent))
    try:
        data = {"choices": [{"message": {"content": ""}}], "search_results": search_results}
        results = executor._format_perplexity_literature_results(data, "q", 100)["results"]
        await executor._hydrate_literature_metadata(results)
    finally:
        await executor.close()
    return results


class TestLiteratureIdsFromUrl:
    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("https://pubmed.ncbi.nlm.nih.gov/34580418/", ("34580418", None, None)),
            (
                "https://pmc.ncbi.nlm.nih.gov/articles/PMC11601760/figure/F4/",
                (None, None, "PMC11601760"),
            ),
            (
                "https://www.nature.com/articles/s41467-018-05379-y",
                (None, "10.1038/s41467-018-05379-y", None),
            ),
            ("https://www.nature.com/articles/nn.2410.pdf", (None, "10.1038/nn.2410", None)),
            (
                "https://www.biorxiv.org/content/10.1101/427484v1.full.pdf",
                (None, "10.1101/427484", None),
            ),
            (
                "https://www.medrxiv.org/content/10.1101/2024.11.20.24317557v2.full-text",
                (None, "10.1101/2024.11.20.24317557", None),
            ),
            (
                "https://www.pnas.org/doi/abs/10.1073/pnas.2022580118",
                (None, "10.1073/pnas.2022580118", None),
            ),
            (
                "https://doi.org/10.1126/science.abf8683?via=x",
                (None, "10.1126/science.abf8683", None),
            ),
            (
                "https://pubmed.ncbi.nlm.nih.gov/?linkname=pubmed_pubmed_citedin&from_uid=21041656",
                (None, None, None),
            ),
            ("https://connect.biorxiv.org/relate/feed/214", (None, None, None)),
        ],
    )
    def test_ids(self, url, expected):
        assert _literature_ids_from_url(url) == expected


class TestHydrationResolution:
    async def test_fixture_hits_resolve(self):
        sent: list[str] = []
        results = await _hydrate(HYDRATION_FIXTURE["search_results"], sent)
        by_url = {r["url"]: r for r in results}

        resolved = {
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC11410376/": "10.1093/bib/bbae449",
            # the batch OR query never returns this one; the single PMCID query does
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC11870466/": "10.1101/2025.02.18.638922",
            "https://www.nature.com/articles/nn.2410": "10.1038/nn.2410",
            "https://www.biorxiv.org/content/10.1101/2024.03.14.584883v1.full": "10.1101/2024.03.14.584883",
            "https://www.pnas.org/doi/10.1073/pnas.0809885106": "10.1073/pnas.0809885106",
            # no id in the URL: matched on the exact title, and on a long truncated prefix
            "https://www.cell.com/cell-genomics/fulltext/S2666-979X(23)00218-5": "10.1016/j.xgen.2023.100404",
            "https://www.cell.com/cell-stem-cell/pdf/S1934-5909(20)30004-7.pdf": "10.1016/j.stem.2020.01.004",
        }
        for url, doi in resolved.items():
            assert by_url[url]["metadata_source"] == "europepmc", url
            assert by_url[url]["doi"] == doi
            assert by_url[url]["authors"]

        # a truncated prefix this short is not searched, and a database page never is
        for url in (
            "https://www.cell.com/cell-reports/fulltext/S2211-1247(25)00037-3",
            "https://www.ncbi.nlm.nih.gov/gene/2840",
        ):
            assert by_url[url]["metadata_source"] == "perplexity"
        assert not any("Oligodendrocytes drive" in q or "GPR17 G protein" in q for q in sent)
        assert sent[0].count(" OR ") == 4
        assert sum(q.startswith("TITLE:") for q in sent) == 2
        # the word the ellipsis follows may be cut short, so it is not searched
        assert 'TITLE:"DUX-miR-344-ZMYM2-Mediated Activation of"' in sent

    async def test_title_match_must_be_exact(self):
        sent: list[str] = []
        title = "Systematic investigation of allelic regulatory activity"
        results = await _hydrate([{"url": "https://example.org/a", "title": title}], sent)
        assert sent == [f'TITLE:"{title}"']
        assert results[0]["metadata_source"] == "perplexity"

    async def test_quote_in_doi_is_escaped(self):
        """An unbalanced quote makes Europe PMC return zero hits for the whole OR query."""
        sent: list[str] = []
        await _hydrate(
            [
                {"url": 'https://doi.org/10.1000/a"b', "title": ""},
                {"url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11410376/", "title": ""},
            ],
            sent,
        )
        assert sent[0] == 'DOI:"10.1000/a\\"b" OR PMCID:PMC11410376'

    async def test_follow_up_queries_are_capped(self):
        sent: list[str] = []
        hits = [
            {"url": f"https://pubmed.ncbi.nlm.nih.gov/{n}/", "title": ""}
            for n in range(1, _HYDRATION_LOOKUP_LIMIT + 6)
        ] + [{"url": "https://example.org/x", "title": "A title long enough to be searched on"}]
        await _hydrate(hits, sent)
        assert len(sent) == 1 + _HYDRATION_LOOKUP_LIMIT
        assert all(q.endswith(" AND SRC:MED") for q in sent[1:])

    async def test_no_follow_ups_when_europepmc_fails(self, monkeypatch):
        calls = []

        def handle(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(400, text="bad request")

        executor = _executor_with_transport(handle)
        try:
            data = {
                "choices": [{"message": {"content": ""}}],
                "search_results": HYDRATION_FIXTURE["search_results"],
            }
            results = executor._format_perplexity_literature_results(data, "q", 100)["results"]
            await executor._hydrate_literature_metadata(results)
        finally:
            await executor.close()
        assert len(calls) == 1
        assert {r["metadata_source"] for r in results} == {"perplexity"}


def _title_page(records: list[dict], hit_count: int | None = None) -> dict:
    return {
        "hitCount": len(records) if hit_count is None else hit_count,
        "resultList": {"result": records},
    }


def _epmc_record(source: str, title: str, year: str, doi: str, authors: str) -> dict:
    return {
        "source": source,
        "title": title,
        "pubYear": year,
        "doi": doi,
        "authorString": authors,
        "journalTitle": "J",
    }


async def _hydrate_with(handler, search_results: list[dict]) -> list[dict]:
    executor = _executor_with_transport(handler)
    try:
        data = {"choices": [{"message": {"content": ""}}], "search_results": search_results}
        results = executor._format_perplexity_literature_results(data, "q", 100)["results"]
        await executor._hydrate_literature_metadata(results)
    finally:
        await executor.close()
    return results


class TestTitleMatchIsOneWork:
    TITLE = "Genetics of type 2 diabetes in two unrelated cohorts"

    async def _resolve(self, page: dict, url: str = "https://example.org/a", date: str = ""):
        def handle(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=page)

        hit = {"url": url, "title": self.TITLE, "date": date}
        return (await _hydrate_with(handle, [hit]))[0]

    async def test_two_different_same_title_records_are_refused(self):
        record = await self._resolve(
            _title_page([
                _epmc_record("MED", self.TITLE, "2010", "10.1/a", "Smith A, Jones B."),
                _epmc_record("MED", self.TITLE, "2019", "10.1/b", "Brown C."),
            ])
        )
        assert record["metadata_source"] == "perplexity"
        assert record["doi"] is None

    async def test_preprint_and_journal_version_follow_the_hit_url(self):
        page = _title_page([
            _epmc_record("MED", self.TITLE, "2023", "10.1/journal", "McAfee JC, Lee S."),
            _epmc_record("PPR", self.TITLE, "2022", "10.1101/preprint", "McAfee JC, Lee S."),
        ])
        journal = await self._resolve(page)
        preprint = await self._resolve(page, url="https://www.biorxiv.org/search/diabetes")
        assert journal["doi"] == "10.1/journal"
        assert preprint["doi"] == "10.1101/preprint"

    async def test_year_must_be_close_to_the_perplexity_date(self):
        record = await self._resolve(
            _title_page([_epmc_record("MED", self.TITLE, "2010", "10.1/a", "Smith A.")]),
            date="2021-03-01",
        )
        assert record["metadata_source"] == "perplexity"

    async def test_partial_page_is_refused(self):
        """Uniqueness cannot be judged when Europe PMC has more hits than the page shows."""
        record = await self._resolve(
            _title_page(
                [_epmc_record("MED", self.TITLE, "2010", "10.1/a", "Smith A.")], hit_count=74
            )
        )
        assert record["metadata_source"] == "perplexity"


class TestHydrationDegrades:
    async def test_unknown_derived_doi_falls_back_to_title(self):
        """Old nature.com slugs drop the DOI's dots, so the derived DOI is unknown."""
        sent: list[str] = []
        results = await _hydrate(
            [
                {
                    "url": "https://www.nature.com/articles/mp201577",
                    "title": "CRMPs: critical molecules for neurite morphogenesis and "
                    "neuropsychiatric diseases - Molecular Psychiatry",
                }
            ],
            sent,
        )
        assert sent[0] == 'DOI:"10.1038/mp201577"'
        assert sent[-1].startswith('TITLE:"CRMPs')
        assert results[0]["metadata_source"] == "europepmc"
        assert results[0]["doi"] == "10.1038/mp.2015.77"

    async def test_duplicate_ids_spend_one_follow_up(self):
        sent: list[str] = []
        hits = [
            {"url": "https://pubmed.ncbi.nlm.nih.gov/111/", "title": ""},
            {"url": "https://pubmed.ncbi.nlm.nih.gov/111/?from=x", "title": ""},
        ]
        await _hydrate(hits, sent)
        assert sent[1:] == ["EXT_ID:111 AND SRC:MED"]

    async def test_follow_up_that_raises_leaves_the_rest(self):
        batch_query = next(iter(HYDRATION_FIXTURE["europepmc_by_query"]))

        def handle(request: httpx.Request) -> httpx.Response:
            query = request.url.params["query"]
            if query != batch_query:
                raise httpx.ReadError("connection reset", request=request)
            return httpx.Response(200, json=HYDRATION_FIXTURE["europepmc_by_query"][query])

        results = await _hydrate_with(handle, HYDRATION_FIXTURE["search_results"])
        by_url = {r["url"]: r for r in results}
        assert by_url["https://www.nature.com/articles/nn.2410"]["metadata_source"] == "europepmc"
        pmc_preprint = by_url["https://pmc.ncbi.nlm.nih.gov/articles/PMC11870466/"]
        assert pmc_preprint["metadata_source"] == "perplexity"

    async def test_hung_europepmc_is_bounded_by_the_budget(self, monkeypatch):
        """An id-less page skips the batch, so only the total budget bounds the follow-ups."""
        monkeypatch.setattr(orchestration, "_HYDRATION_FOLLOW_UP_BUDGET_S", 0.3)

        async def handle(request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(30)
            raise AssertionError("unreachable")

        hits = [
            {"url": f"https://example.org/{n}", "title": f"A long enough title number {n}"}
            for n in range(6)
        ]
        started = time.monotonic()
        results = await _hydrate_with(handle, hits)
        assert time.monotonic() - started < 3
        assert {r["metadata_source"] for r in results} == {"perplexity"}

    async def test_hydration_exception_returns_unhydrated_hits(self, monkeypatch):
        """A malformed Europe PMC entry must not fail the whole search."""
        monkeypatch.setenv("PERPLEXITY_API_KEY", "test-key")

        def handle(request: httpx.Request) -> httpx.Response:
            if request.url.host == "api.perplexity.ai":
                return httpx.Response(200, json=PERPLEXITY_RESPONSE)
            return httpx.Response(200, json=_title_page(["not a record"]))

        executor = _executor_with_transport(handle)
        try:
            result = await executor.search_scientific_literature(
                "platelet count", max_results=5, backend="perplexity"
            )
        finally:
            await executor.close()

        assert result["success"] is True
        assert result["results"][0]["title"] == "Genetic variants associated with platelet count are ..."
        assert {r["metadata_source"] for r in result["results"]} == {"perplexity"}


def _hits(n: int) -> list[dict]:
    return [
        {"title": f"Paper {i}", "url": f"https://example.org/paper-{i}", "snippet": f"s{i}"}
        for i in range(1, n + 1)
    ]


def _format(search_results: list[dict], summary: str, max_results: int) -> dict:
    data = {"choices": [{"message": {"content": summary}}], "search_results": search_results}
    return ServerToolExecutor()._format_perplexity_literature_results(data, "q", max_results)


class TestRecordKind:
    @pytest.mark.parametrize(
        "url",
        [
            "https://www.ncbi.nlm.nih.gov/gene/7157",
            "https://www.ncbi.nlm.nih.gov/clinvar/variation/12345/",
            "https://www.ncbi.nlm.nih.gov/books/NBK1116/",
            "https://pubmed.ncbi.nlm.nih.gov/?term=TP53",
            "https://www.genecards.org/cgi-bin/carddisp.pl?gene=TP53",
            "https://omim.org/entry/191170",
            "https://www.uniprot.org/uniprotkb/P04637/entry",
        ],
    )
    def test_database_pages(self, url):
        result = _format([{"title": "TP53", "url": url}], "", 10)
        assert result["results"][0]["record_kind"] == "database_page"

    @pytest.mark.parametrize(
        "url",
        [
            "https://pubmed.ncbi.nlm.nih.gov/34580418/",
            "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2974578/",
            "https://www.ncbi.nlm.nih.gov/pubmed/34580418",
            "https://www.nature.com/articles/s41586-022-05473-8",
            "https://example.org/press-release",
        ],
    )
    def test_unmatched_non_database_hits_are_snippets(self, url):
        result = _format([{"title": "T", "url": url}], "", 10)
        assert result["results"][0]["record_kind"] == "perplexity_snippet"

    async def test_hydration_sets_europepmc_and_leaves_the_rest(self, monkeypatch):
        monkeypatch.setenv("PERPLEXITY_API_KEY", "test-key")
        payload = {
            **PERPLEXITY_RESPONSE,
            "search_results": [
                *PERPLEXITY_RESPONSE["search_results"],
                {"title": "TP53 gene", "url": "https://www.ncbi.nlm.nih.gov/gene/7157"},
            ],
        }
        executor = _executor_with_transport(_handler(perplexity_payload=payload))
        try:
            result = await executor.search_scientific_literature(
                "platelet count", max_results=5, backend="perplexity"
            )
        finally:
            await executor.close()

        assert [r["record_kind"] for r in result["results"]] == [
            "europepmc", "europepmc", "database_page",
        ]
        assert all(r["metadata_source"] == "europepmc" for r in result["results"][:2])

    async def test_unhydrated_hits_keep_their_kind(self, monkeypatch):
        monkeypatch.setenv("PERPLEXITY_API_KEY", "test-key")
        executor = _executor_with_transport(_handler(epmc_status=503))
        try:
            result = await executor.search_scientific_literature(
                "platelet count", max_results=5, backend="perplexity"
            )
        finally:
            await executor.close()

        assert {r["record_kind"] for r in result["results"]} == {"perplexity_snippet"}


class TestSummaryCitations:
    def test_marker_past_max_results_gets_a_cited_only_record(self):
        result = _format(_hits(6), "A [1] and B [5][2]. C [5].", max_results=3)

        assert result["returned"] == 3
        ranked, cited = result["results"][:3], result["results"][3:]
        assert [r["title"] for r in ranked] == ["Paper 1", "Paper 2", "Paper 3"]
        assert cited == [
            {"title": "Paper 5", "url": "https://example.org/paper-5", "record_kind": "cited_only"}
        ]

    def test_citations_resolve_against_every_hit_with_null_past_the_list(self):
        result = _format(_hits(4), "A [2]. B [4]. C [9]. D [0].", max_results=2)

        assert result["summary_citations"] == {
            0: None,
            2: {
                "title": "Paper 2",
                "url": "https://example.org/paper-2",
                "record_kind": "perplexity_snippet",
            },
            4: {"title": "Paper 4", "url": "https://example.org/paper-4", "record_kind": "cited_only"},
            9: None,
        }

    def test_summary_is_labelled_as_perplexity_prose(self):
        result = _format(_hits(1), "Claim [1].", max_results=1)

        assert result["summary"] == "Claim [1]."
        assert "AI-generated" in result["summary_note"]
        assert "not to any paper" in result["summary_note"]

    async def test_cited_only_records_are_not_hydrated(self, monkeypatch):
        monkeypatch.setenv("PERPLEXITY_API_KEY", "test-key")
        payload = {
            "choices": [{"message": {"content": "Platelets [1]; second [2]."}}],
            "search_results": PERPLEXITY_RESPONSE["search_results"],
        }
        queries: list[str] = []

        def handle(request: httpx.Request) -> httpx.Response:
            if request.url.host == "www.ebi.ac.uk":
                queries.append(request.url.params["query"])
            return _handler(perplexity_payload=payload)(request)

        executor = _executor_with_transport(handle)
        try:
            result = await executor.search_scientific_literature(
                "platelet count", max_results=1, backend="perplexity"
            )
        finally:
            await executor.close()

        assert result["returned"] == 1
        assert [r["record_kind"] for r in result["results"]] == ["europepmc", "cited_only"]
        assert queries and all("PMC2974578" not in q for q in queries)
        assert result["results"][1] == {
            "title": "A second paper",
            "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC2974578/",
            "record_kind": "cited_only",
        }
        # rebuilt after hydration, so a cited hit's corrected title and kind reach the map
        assert result["summary_citations"][1] == {
            "title": "Genetic variants associated with platelet count",
            "url": "https://pubmed.ncbi.nlm.nih.gov/34580418/",
            "record_kind": "europepmc",
        }
        assert result["summary_citations"][2]["record_kind"] == "cited_only"
