"""The FinnGen R12 sex-specific and recessive GWAS are on-request datasets of a second
kind: the prompt section and the tool hints that name them exist only where results-api
serves them, and they are told apart from the sandbox custom GWAS releases by data type —
the one thing results-api's answer carries besides the resource.
"""

import pytest

from genetics_mcp_server.config import prompt_blocks
from genetics_mcp_server.config.defaults import default_system_prompt
from genetics_mcp_server.tools.definitions import (
    all_anthropic_tools,
    get_anthropic_tools,
    resolve_tools,
)

HEADING = "### FinnGen R12 sex-specific and recessive GWAS"
CUSTOM_HEADING = "### Users' own custom GWAS (sandbox userresults)"
_ALL_VARIANTS = (None, "condensed")
MODELS = (("finngen_recessive", "gwas_recessive"), ("finngen_sex_specific", "gwas_sex_specific"))


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


class TestKinds:
    def test_the_data_type_tells_the_kinds_apart(self, deployment_answers):
        deployment_answers((("finngen_custom_r14", "gwas"), ("finngen_custom_r14", "hla")) + MODELS)
        assert prompt_blocks.custom_gwas_resources() == ("finngen_custom_r14",)
        assert prompt_blocks.r12_model_datasets() == MODELS

    def test_one_kind_alone_renders_only_its_own_section(self, deployment_answers):
        deployment_answers(MODELS)
        prompt = default_system_prompt("FinnGenie")
        assert HEADING in prompt and CUSTOM_HEADING not in prompt
        deployment_answers((("finngen_custom_r14", "gwas"),))
        prompt = default_system_prompt("FinnGenie")
        assert CUSTOM_HEADING in prompt and HEADING not in prompt


class TestPrompt:
    @pytest.mark.parametrize("variant", _ALL_VARIANTS)
    def test_a_finngen_deployment_names_both_releases(self, variant):
        prompt = default_system_prompt("FinnGenie", variant=variant)
        assert HEADING in prompt
        section = prompt.split(HEADING, 1)[1].split("\n### ", 1)[0]
        assert "`finngen_sex_specific` (data_type `gwas_sex_specific`)" in section
        assert "`finngen_recessive` (data_type `gwas_recessive`)" in section
        assert "{models}" not in section and "{resource}" not in section
        assert "Only on request" in section

    @pytest.mark.parametrize("variant", _ALL_VARIANTS)
    @pytest.mark.parametrize("answer", [(), None], ids=["none served", "api silent"])
    def test_a_deployment_without_them_says_nothing(self, deployment_answers, variant, answer):
        deployment_answers(answer)
        prompt = default_system_prompt("FinnGenie", variant=variant)
        assert HEADING not in prompt
        assert "gwas_sex_specific" not in prompt and "gwas_recessive" not in prompt
        assert "finngen_sex_specific" not in prompt and "finngen_recessive" not in prompt

    @pytest.mark.parametrize("variant", _ALL_VARIANTS)
    def test_only_the_served_model_is_described(self, deployment_answers, variant):
        deployment_answers((("other_sex", "gwas_sex_specific"),))
        prompt = default_system_prompt("FinnGenie", variant=variant)
        assert "`other_sex` (data_type `gwas_sex_specific`)" in prompt
        assert "gwas_recessive" not in prompt and "finngen_sex_specific" not in prompt


class TestToolHints:
    def test_a_finngen_deployment_hints_the_routes(self):
        tools = get_anthropic_tools()
        names = "'finngen_recessive', 'finngen_sex_specific'"
        assert names in _description(tools, "search_phenotypes")
        assert names in _param(tools, "search_phenotypes", "resource")
        assert names in _param(tools, "list_datasets", "resource")
        for tool in ("get_summary_stats", "get_summary_stats_by_region"):
            resource = _param(tools, tool, "resource")
            assert "'finngen_sex_specific' with data_type 'gwas_sex_specific'" in resource
            assert "'finngen_recessive' with data_type 'gwas_recessive'" in resource
            assert "ON REQUEST ONLY" in resource
            data_type = _param(tools, tool, "data_type")
            assert "'gwas_recessive' or 'gwas_sex_specific'" in data_type
        # the custom GWAS hints are still there beside them
        assert "'finngen_custom_r14'" in _param(tools, "get_summary_stats", "resource")

    @pytest.mark.parametrize("answer", [(), None], ids=["none served", "api silent"])
    def test_a_deployment_without_them_never_mentions_them(self, deployment_answers, answer):
        deployment_answers(answer)
        for tools in (
            get_anthropic_tools(),
            get_anthropic_tools(code_execution=True),
            all_anthropic_tools(),
        ):
            for tool in tools:
                blob = repr(tool)
                assert "sex_specific" not in blob and "recessive" not in blob, tool["name"]

    def test_the_module_definitions_are_not_rewritten(self):
        get_anthropic_tools()
        for tool in resolve_tools(False):
            assert "sex_specific" not in repr(tool) and "finngen_recessive" not in repr(tool), tool[
                "name"
            ]
