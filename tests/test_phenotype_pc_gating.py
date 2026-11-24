"""The phenotype-PC GWAS is an on-request dataset of a third kind: its prompt section and
tool hints exist only where results-api serves it, and its data type is what keeps it from
being described as a sandbox custom GWAS.
"""

import pytest
from helpers import assert_module_definitions_lack
from helpers import tool_description as _description
from helpers import tool_param as _param

from genetics_mcp_server.config import prompt_blocks
from genetics_mcp_server.config.defaults import default_system_prompt
from genetics_mcp_server.tools.definitions import (
    all_anthropic_tools,
    get_anthropic_tools,
)

HEADING = "### FinnGen phenotype principal components (PC GWAS)"
CUSTOM_HEADING = "### Users' own custom GWAS (sandbox userresults)"
_ALL_VARIANTS = (None, "condensed")
PCS = (("finngen_phenotype_pca", "gwas_phenotype_pc"),)


class TestKinds:
    def test_the_data_type_keeps_it_out_of_the_custom_gwas(self, deployment_answers):
        deployment_answers((("finngen_custom_r14", "gwas"),) + PCS)
        assert prompt_blocks.custom_gwas_resources() == ("finngen_custom_r14",)
        assert prompt_blocks.phenotype_pc_datasets() == PCS
        assert prompt_blocks.r12_model_datasets() == ()

    def test_a_deployment_serving_it_alone_renders_only_its_section(self, deployment_answers):
        deployment_answers(PCS)
        prompt = default_system_prompt("FinnGenie")
        assert HEADING in prompt and CUSTOM_HEADING not in prompt
        assert "'finngen_custom" not in repr(get_anthropic_tools())


class TestPrompt:
    @pytest.mark.parametrize("variant", _ALL_VARIANTS)
    def test_the_section_names_the_served_resource(self, variant):
        prompt = default_system_prompt("FinnGenie", variant=variant)
        section = prompt.split(HEADING, 1)[1].split("\n### ", 1)[0]
        assert "`finngen_phenotype_pca` (data_type `gwas_phenotype_pc`)" in section
        assert "{pcs}" not in section
        assert "Only on request" in section and "SIGN IS ARBITRARY" in section

    @pytest.mark.parametrize("variant", _ALL_VARIANTS)
    @pytest.mark.parametrize("answer", [(), None], ids=["none served", "api silent"])
    def test_a_deployment_without_it_says_nothing(self, deployment_answers, variant, answer):
        deployment_answers(answer)
        prompt = default_system_prompt("FinnGenie", variant=variant)
        assert HEADING not in prompt and "phenotype_pc" not in prompt


class TestToolHints:
    def test_the_routes_are_hinted(self):
        tools = get_anthropic_tools()
        name = "'finngen_phenotype_pca'"
        assert name in _description(tools, "search_phenotypes")
        assert name in _param(tools, "list_datasets", "resource")
        for tool in (
            "get_credible_sets_by_variant",
            "get_credible_sets_by_gene",
            "get_credible_sets_by_region",
        ):
            assert "NEVER included when this is omitted" in _param(tools, tool, "resource")
        for tool in ("get_credible_sets_by_phenotype", "get_credible_set_by_id"):
            assert name in _param(tools, tool, "resource")
        for tool in ("get_summary_stats", "get_summary_stats_by_region"):
            assert f"{name} with data_type 'gwas_phenotype_pc'" in _param(tools, tool, "resource")
            assert "'gwas_phenotype_pc'" in _param(tools, tool, "data_type")
        # the other kinds' hints are still there beside it
        assert "'finngen_custom_r14'" in _param(tools, "get_summary_stats", "resource")
        assert "'finngen_recessive'" in _param(tools, "get_summary_stats", "resource")

    @pytest.mark.parametrize("answer", [(), None], ids=["none served", "api silent"])
    def test_a_deployment_without_it_never_mentions_it(self, deployment_answers, answer):
        deployment_answers(answer)
        for tools in (
            get_anthropic_tools(),
            get_anthropic_tools(code_execution=True),
            all_anthropic_tools(),
        ):
            for tool in tools:
                assert "phenotype_pc" not in repr(tool) and "phenotype-PC" not in repr(tool), tool[
                    "name"
                ]

    def test_the_module_definitions_are_not_rewritten(self):
        assert_module_definitions_lack("phenotype_pc", "phenotype-PC")
