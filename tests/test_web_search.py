"""Domain restriction on the DuckDuckGo path of `web_search`.

Self-contained: `ddgs` is replaced by a fake that answers from a table keyed on the query
text, so what reaches the backend is asserted as well as what comes back.
"""

import asyncio
import sys
import types

import pytest

from genetics_mcp_server.tools import ServerToolExecutor
from genetics_mcp_server.tools.orchestration import (
    _DDGS_NO_RESULTS,
    _search_domains,
    _url_in_domains,
)


def _hit(url: str) -> dict:
    return {"title": f"title of {url}", "href": url, "body": f"body of {url}"}


@pytest.fixture
def backend(monkeypatch):
    """Install a fake `ddgs`; returns (answers, calls). An answer may be an exception."""
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    answers: dict[str, object] = {}
    calls: list[tuple[str, int]] = []

    class DDGS:
        def text(self, query, max_results=None):
            calls.append((query, max_results))
            answer = answers.get(query, Exception(_DDGS_NO_RESULTS))
            if isinstance(answer, Exception):
                raise answer
            return answer[:max_results]

    monkeypatch.setitem(sys.modules, "ddgs", types.SimpleNamespace(DDGS=DDGS))
    return answers, calls


def _search(**kwargs) -> dict:
    return asyncio.run(ServerToolExecutor().web_search(**kwargs))


def test_include_domains_become_one_site_query_each(backend):
    answers, calls = backend
    answers["rules site:docs.finngen.fi"] = [_hit("https://docs.finngen.fi/a")]
    answers["rules site:risteys.finngen.fi"] = [_hit("https://risteys.finngen.fi/b")]

    result = _search(
        query="rules", include_domains=["docs.finngen.fi", "https://www.risteys.finngen.fi/"]
    )

    assert sorted(q for q, _ in calls) == [
        "rules site:docs.finngen.fi",
        "rules site:risteys.finngen.fi",
    ]
    assert result["success"] is True
    assert [r["url"] for r in result["results"]] == [
        "https://docs.finngen.fi/a",
        "https://risteys.finngen.fi/b",
    ]
    assert "note" not in result


def test_results_are_interleaved_and_capped(backend):
    answers, _ = backend
    answers["q site:a.org"] = [_hit(f"https://a.org/{i}") for i in range(5)]
    answers["q site:b.org"] = [_hit(f"https://sub.b.org/{i}") for i in range(5)]

    result = _search(query="q", max_results=4, include_domains=["a.org", "b.org"])

    assert [r["url"] for r in result["results"]] == [
        "https://a.org/0",
        "https://sub.b.org/0",
        "https://a.org/1",
        "https://sub.b.org/1",
    ]


def test_a_hit_outside_the_included_domain_is_dropped(backend):
    """The `site:` operator is the backend's to honour; the filter is what guarantees it."""
    answers, _ = backend
    answers["q site:a.org"] = [_hit("https://nota.org/x"), _hit("https://a.org/y")]

    result = _search(query="q", include_domains=["a.org"])

    assert [r["url"] for r in result["results"]] == ["https://a.org/y"]


def test_no_hit_on_an_included_domain_is_an_empty_success(backend):
    """A restricted search with nothing to find used to read as a broken search tool."""
    result = _search(query="q", include_domains=["a.org", "b.org"])

    assert result["success"] is True
    assert result["results"] == []
    assert "a.org, b.org" in result["note"]


def test_a_failed_domain_is_named_when_another_answers(backend):
    answers, _ = backend
    answers["q site:a.org"] = [_hit("https://a.org/y")]
    answers["q site:b.org"] = Exception("202 Ratelimit")

    result = _search(query="q", include_domains=["a.org", "b.org"])

    assert result["success"] is True
    assert [r["url"] for r in result["results"]] == ["https://a.org/y"]
    assert "b.org" in result["note"] and "a.org" not in result["note"]


def test_every_domain_failing_is_a_failure(backend):
    answers, _ = backend
    answers["q site:a.org"] = Exception("202 Ratelimit")

    result = _search(query="q", include_domains=["a.org"])

    assert result["success"] is False
    assert "a.org: 202 Ratelimit" in result["error"]


def test_exclude_domains_are_filtered_from_an_overfetch(backend):
    answers, calls = backend
    answers["q"] = [
        _hit("https://www.bad.com/1"),
        _hit("https://good.org/1"),
        _hit("https://m.bad.com/2"),
        _hit("https://good.org/2"),
    ]

    result = _search(query="q", max_results=2, exclude_domains=["bad.com"])

    assert calls == [("q", 4)]
    assert [r["url"] for r in result["results"]] == ["https://good.org/1", "https://good.org/2"]


def test_unrestricted_search_is_one_plain_query(backend):
    answers, calls = backend
    answers["q"] = [_hit("https://a.org/1")]

    result = _search(query="q", max_results=3)

    assert calls == [("q", 3)]
    assert result == {
        "success": True,
        "query": "q",
        "source": "duckduckgo",
        "results": [
            {
                "title": "title of https://a.org/1",
                "url": "https://a.org/1",
                "content": "body of https://a.org/1",
            }
        ],
    }


def test_unrestricted_search_with_no_hits_still_reports_the_backend_error(backend):
    result = _search(query="q")

    assert result == {"success": False, "error": f"Web search failed: {_DDGS_NO_RESULTS}"}


def test_domain_spellings_reduce_to_one_hostname():
    assert _search_domains(
        [
            "Docs.FinnGen.fi",
            "https://docs.finngen.fi/page",
            "www.docs.finngen.fi",
            "*.who.int",
            ".thl.fi",
            " ",
        ]
    ) == ["docs.finngen.fi", "who.int", "thl.fi"]
    assert _search_domains(None) == []


def test_domain_match_is_on_label_boundaries():
    assert _url_in_domains("https://r12.risteys.finngen.fi/x", ["finngen.fi"])
    assert _url_in_domains("https://finngen.fi", ["finngen.fi"])
    assert not _url_in_domains("https://notfinngen.fi/x", ["finngen.fi"])
    assert not _url_in_domains("not a url", ["finngen.fi"])
