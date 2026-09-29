"""The custom GWAS prompt section and tool hints exist only where results-api serves an
on-request dataset — a daly deployment never hears the words.

Both are rendered from one probe (config.prompt_blocks.on_request_datasets); these tests
install its answer and read what the prompt and the tool definitions then say.
"""

import httpx
import pytest

from genetics_mcp_server.config import prompt_blocks
from genetics_mcp_server.config.defaults import default_system_prompt
from genetics_mcp_server.tools.definitions import (
    all_anthropic_tools,
    get_anthropic_tools,
    resolve_tools,
)
from genetics_mcp_server.tools.executor import ToolExecutor

HEADING = "### Users' own custom GWAS (sandbox userresults)"
_ALL_VARIANTS = (None, "condensed")


@pytest.fixture
def deployment_answers(monkeypatch):
    def install(answer):
        monkeypatch.setattr(prompt_blocks, "_probe_on_request_datasets", lambda: answer)
        monkeypatch.setattr(prompt_blocks, "_on_request_datasets", None)

    return install


def _param(tools, name, param):
    (tool,) = [t for t in tools if t["name"] == name]
    return tool["input_schema"]["properties"][param]["description"]


def _description(tools, name):
    (tool,) = [t for t in tools if t["name"] == name]
    return tool["description"]


class TestPrompt:
    @pytest.mark.parametrize("variant", _ALL_VARIANTS)
    def test_a_finngen_deployment_names_its_releases(self, variant):
        prompt = default_system_prompt("FinnGenie", variant=variant)
        assert HEADING in prompt
        assert "`finngen_custom_r12`, `finngen_custom_r13`, `finngen_custom_r14`" in prompt
        assert "{resources}" not in prompt

    @pytest.mark.parametrize("variant", _ALL_VARIANTS)
    @pytest.mark.parametrize("answer", [(), None], ids=["none served", "api silent"])
    def test_a_deployment_without_one_says_nothing(self, deployment_answers, variant, answer):
        deployment_answers(answer)
        prompt = default_system_prompt("FinnGenie", variant=variant)
        assert HEADING not in prompt
        assert "custom GWAS" not in prompt and "userresults" not in prompt
        assert "finngen_custom" not in prompt

    def test_the_names_are_the_deployments_own(self, deployment_answers):
        deployment_answers((("other_custom_r1", "gwas"),))
        prompt = default_system_prompt("FinnGenie")
        assert "`other_custom_r1`" in prompt and "finngen_custom" not in prompt

    def test_a_probe_that_raises_does_not_reach_the_chat_turn(self, monkeypatch):
        def boom():
            raise RuntimeError("results-api blew up")

        monkeypatch.setattr(prompt_blocks, "_probe_on_request_datasets", boom)
        monkeypatch.setattr(prompt_blocks, "_on_request_datasets", None)
        assert HEADING not in default_system_prompt("FinnGenie")

    def test_silence_is_retried_and_an_answer_kept(self, monkeypatch):
        now = [1000.0]
        probes = []
        answers = iter([None, None, (("finngen_custom_r14", "gwas"),), (("x", "gwas"),)])

        def probe():
            probes.append(now[0])
            return next(answers)

        monkeypatch.setattr(prompt_blocks.time, "monotonic", lambda: now[0])
        monkeypatch.setattr(prompt_blocks, "_probe_on_request_datasets", probe)
        monkeypatch.setattr(prompt_blocks, "_on_request_datasets", None)
        assert prompt_blocks.custom_gwas_resources() == ()
        assert prompt_blocks.custom_gwas_resources() == ()  # within the minute: not asked again
        now[0] += 61
        assert prompt_blocks.custom_gwas_resources() == ()  # asked, still silent
        now[0] += 61
        assert prompt_blocks.custom_gwas_resources() == ("finngen_custom_r14",)
        now[0] += 3600
        assert prompt_blocks.custom_gwas_resources() == ("finngen_custom_r14",)  # kept
        assert len(probes) == 3


class TestToolHints:
    def test_a_finngen_deployment_hints_every_route_to_a_run(self):
        tools = get_anthropic_tools()
        names = "'finngen_custom_r12', 'finngen_custom_r13', 'finngen_custom_r14'"
        assert names in _description(tools, "search_phenotypes")
        for tool in (
            "search_phenotypes",
            "get_credible_sets_by_phenotype",
            "get_credible_set_leads_by_phenotype",
            "get_credible_set_by_id",
            "list_datasets",
            "get_summary_stats",
            "get_summary_stats_by_region",
        ):
            assert names in _param(tools, tool, "resource"), tool
        hla = _param(tools, "get_hla_by_phenotype", "resource")
        assert "'finngen_custom_r14'" in hla and "r13" not in hla

    @pytest.mark.parametrize("answer", [(), None], ids=["none served", "api silent"])
    def test_a_deployment_without_one_never_mentions_it(self, deployment_answers, answer):
        deployment_answers(answer)
        for tools in (get_anthropic_tools(), get_anthropic_tools(code_execution=True), all_anthropic_tools()):
            for tool in tools:
                blob = repr(tool)
                assert "custom GWAS" not in blob and "finngen_custom" not in blob, tool["name"]

    def test_the_hla_hint_needs_an_hla_release(self, deployment_answers):
        deployment_answers((("finngen_custom_r13", "gwas"),))
        tools = get_anthropic_tools()
        assert "'finngen_custom_r13'" in _param(tools, "get_summary_stats", "resource")
        assert "custom" not in _param(tools, "get_hla_by_phenotype", "resource")

    def test_the_module_definitions_are_not_rewritten(self):
        get_anthropic_tools()
        for tool in resolve_tools(False):
            assert "finngen_custom" not in repr(tool), tool["name"]


class TestProbe:
    async def test_reads_the_route_and_pairs_resource_with_data_type(self):
        seen = {}

        def handler(request):
            seen["path"] = request.url.path
            seen["params"] = dict(request.url.params)
            return httpx.Response(200, json=[
                {"dataset_id": "a", "resource": "finngen_custom_r14", "data_type": "gwas"},
                {"dataset_id": "b", "resource": "finngen_custom_r14", "data_type": "hla"},
                {"dataset_id": "c", "resource": "finngen_custom_r13", "data_type": "gwas"},
            ])

        executor = ToolExecutor(api_base_url="http://api.test/api")
        executor.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        assert await executor.on_request_datasets() == (
            ("finngen_custom_r13", "gwas"),
            ("finngen_custom_r14", "gwas"),
            ("finngen_custom_r14", "hla"),
        )
        assert seen["path"] == "/api/v1/datasets/on_request"
        assert seen["params"] == {"include_stats": "false"}

    async def test_a_results_api_without_the_route_is_unknown_not_a_catalogue(self):
        executor = ToolExecutor(api_base_url="http://api.test/api")
        executor.client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda req: httpx.Response(404, text="no"))
        )
        assert await executor.on_request_datasets() is None


class TestOneRunsMetadata:
    """A release holds hundreds of runs and get_resource_metadata caps rows, so a run's
    date and sizes are read by name rather than by paging the release's table."""

    async def test_named_phenotypes_reach_the_endpoint_as_a_filter(self):
        seen = {}

        def handler(request):
            seen["params"] = dict(request.url.params)
            return httpx.Response(200, json=[{"phenotype_code": "G6_MS", "date": "2026-09-02"}])

        executor = ToolExecutor(api_base_url="http://api.test/api")
        executor.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        result = await executor.get_resource_metadata("finngen_custom_r14", ["G6_MS", " AFB "])
        assert seen["params"] == {"format": "json", "phenotypes": "G6_MS,AFB"}
        assert result["success"] and result["metadata"][0]["date"] == "2026-09-02"
        assert not result["truncated"]

    async def test_an_unknown_run_names_the_run_in_the_error(self):
        executor = ToolExecutor(api_base_url="http://api.test/api")
        executor.client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda req: httpx.Response(404, text="no"))
        )
        result = await executor.get_resource_metadata("finngen_custom_r14", ["NOPE"])
        assert result["success"] is False and "NOPE" in result["error"]

    def test_the_tool_offers_the_filter(self):
        (tool,) = [t for t in get_anthropic_tools() if t["name"] == "get_resource_metadata"]
        assert "phenotypes" in tool["input_schema"]["properties"]
        assert "phenotypes" not in tool["input_schema"].get("required", [])
