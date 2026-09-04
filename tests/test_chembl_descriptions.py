"""Every backticked name in the ChEMBL tool descriptions and prompt blocks must be
something the code actually produces.

Three review cycles each found a description or prompt sentence naming a key the executor
never emits (`resolution.ambiguous`, `other_targets` on the bioactivity tool, a bare
`other_candidates`). The tests pinned headings and bounds, so nothing compared the prose
against the result shape. This does: the tools are run against the mocked happy paths of
test_chembl.py, and each backticked token has to resolve to a parameter, a tool name, or a
key that reaches the model in the result.
"""

import re

import pytest
from test_chembl import (
    _PPARG_TARGET_DETAIL,
    _PPARG_TARGETS,
    _ROSIGLITAZONE,
    _STATUS,
    _TROGLITAZONE,
    _activity,
    _ChEMBLToolCase,
    _mechanism,
    _molecule_pages,
    _page,
    _pages,
)

from genetics_mcp_server.config.defaults import _PROMPT_BLOCKS
from genetics_mcp_server.tools.definitions import TOOL_DEFINITIONS

_CHEMBL_TOOLS = ("get_drug_targets_for_gene", "get_drug_profile", "get_target_bioactivity")

# literals that are neither a parameter, a tool name, nor a result key. Each is quoted
# because the model has to type it or recognise it verbatim.
_LITERALS = {
    "include_indications=True": "an example argument in a worked call",
    "CHEMBL<number>": "the id shape the query parameter accepts, not a key",
    "chembl_id": "a value of resolution.kind, not a key",
    "synonym": "a value of resolution.kind, not a key",
}

_BACKTICKED = re.compile(r"`([^`\n]+)`")


def _tokens(text: str) -> list[str]:
    """Backticked tokens worth checking: an identifier-shaped name, possibly dotted.

    Prose quoted for the model to reproduce verbatim — the `### Drug and Target Evidence
    (ChEMBL)` heading — carries spaces and is not a name to resolve.
    """
    return [t for t in _BACKTICKED.findall(text) if not re.search(r"\s", t)]


def _visible_keys(node, prefix: str = "", depth: int = 2):
    """Key names the model can see in a result, bare and dotted, two levels down.

    A list stands for its first element, so `rows[0].atc_codes` is named `atc_codes`
    exactly as the description writes it. Bare names are yielded at every depth, which is
    what makes an unqualified mention of a nested key legal — so this catches an INVENTED
    name, not one written at the wrong level.
    """
    if isinstance(node, list):
        node = node[0] if node and isinstance(node[0], dict) else None
    if not isinstance(node, dict) or depth <= 0:
        return
    for key, value in node.items():
        yield key
        if prefix:
            yield f"{prefix}.{key}"
        yield from _visible_keys(value, f"{prefix}.{key}" if prefix else key, depth - 1)


def _unresolved(text: str, results: dict[str, dict], parameters: set[str]) -> list[str]:
    """The backticked tokens in `text` that name nothing the code produces."""
    tool_names = {tool["name"] for tool in TOOL_DEFINITIONS}
    keys = {key for result in results.values() for key in _visible_keys(result)}
    known = keys | tool_names | parameters | set(_LITERALS)
    return [t for t in _tokens(text) if t not in known]


def _parameters(tool: dict) -> set[str]:
    return set(tool.get("parameters", {}))


def _tool(name: str) -> dict:
    return next(tool for tool in TOOL_DEFINITIONS if tool["name"] == name)


def _description(tool: dict) -> str:
    """The description plus its parameter descriptions — one definition, one contract."""
    return "\n".join(
        [tool["description"]]
        + [p.get("description", "") for p in tool.get("parameters", {}).values()]
    )


def _chembl_blocks() -> list:
    """Prompt blocks that are about the ChEMBL tools, derived rather than listed.

    The caveats block names no tool in its text — that is deliberate, so disabling one
    tool does not delete the max_phase semantics — so its gating conditions count too.
    """
    return [
        block
        for block in _PROMPT_BLOCKS
        if any(
            name in block.text
            or name in block.requires_any
            or name in block.requires_all
            for name in _CHEMBL_TOOLS
        )
    ]


@pytest.mark.asyncio
class TestChEMBLDescriptionsNameOnlyRealKeys(_ChEMBLToolCase):
    async def _results(self) -> dict[str, dict]:
        """One happy-path result per tool, from the same mocks as test_chembl.py."""
        indications = [
            {
                "molecule_chembl_id": "CHEMBL121",
                "efo_id": "EFO:0001360",
                "efo_term": "type II diabetes mellitus",
                "mesh_heading": "Diabetes Mellitus, Type 2",
                "max_phase_for_ind": 4,
            }
        ]
        drug_targets = _pages(
            status=[_STATUS],
            target=[_page("targets", _PPARG_TARGETS)],
            mechanism=[_page("mechanisms", [_mechanism("CHEMBL121")])],
            molecule=_molecule_pages([_ROSIGLITAZONE, _TROGLITAZONE]),
            drug_indication=[_page("drug_indications", indications)],
        )
        patcher, _calls = self._patch_get(drug_targets)
        with self._stub_resolver(), patcher:
            targets_result = await self.executor.chembl.get_drug_targets_for_gene(
                "PPARG", include_indications=True
            )

        profile = _pages(
            status=[_STATUS],
            molecule=[_page("molecules", [_ROSIGLITAZONE])],
            mechanism=[
                _page(
                    "mechanisms",
                    [
                        {
                            "target_chembl_id": "CHEMBL235",
                            "mechanism_of_action": "PPAR gamma agonist",
                            "action_type": "AGONIST",
                            "max_phase": 4,
                        }
                    ],
                )
            ],
            target=[_page("targets", [_PPARG_TARGET_DETAIL])],
            drug_indication=[_page("drug_indications", indications)],
        )
        patcher, _calls = self._patch_get(profile)
        with patcher:
            profile_result = await self.executor.chembl.get_drug_profile("rosiglitazone")

        bioactivity = _pages(
            status=[_STATUS],
            target=[_page("targets", _PPARG_TARGETS)],
            activity=[
                _page(
                    "activities",
                    [_activity("CHEMBL121", "7.2"), _activity("CHEMBL595", "9.0")],
                    total=4210,
                )
            ],
            molecule=[_page("molecules", [_ROSIGLITAZONE, _TROGLITAZONE])],
        )
        patcher, _calls = self._patch_get(bioactivity)
        with self._stub_resolver(), patcher:
            bioactivity_result = await self.executor.chembl.get_target_bioactivity("PPARG")

        for result in (targets_result, profile_result, bioactivity_result):
            assert result["success"] is True
        return {
            "get_drug_targets_for_gene": targets_result,
            "get_drug_profile": profile_result,
            "get_target_bioactivity": bioactivity_result,
        }

    async def test_a_tool_description_names_only_parameters_tools_and_result_keys(self):
        results = await self._results()
        offenders = {}
        for name in _CHEMBL_TOOLS:
            tool = _tool(name)
            description = _description(tool)
            assert _tokens(description), f"{name} names nothing in backticks — check the regex"
            unresolved = _unresolved(description, {name: results[name]}, _parameters(tool))
            if unresolved:
                offenders[name] = unresolved
        assert not offenders

    async def test_the_prompt_blocks_name_only_parameters_tools_and_result_keys(self):
        results = await self._results()
        parameters = {p for name in _CHEMBL_TOOLS for p in _parameters(_tool(name))}
        blocks = _chembl_blocks()
        assert blocks, "no ChEMBL prompt block found — the derivation above is broken"
        assert any(_tokens(block.text) for block in blocks)
        offenders = {
            block.text[:60]: unresolved
            for block in blocks
            if (unresolved := _unresolved(block.text, results, parameters))
        }
        assert not offenders

    async def test_a_key_the_code_does_not_produce_is_caught(self):
        results = await self._results()
        tool = _tool("get_target_bioactivity")
        doctored = tool["description"] + "\n`other_targets` and `resolution.ambiguous`.\n"
        # other_targets IS on this result; only the invented one is reported
        assert _unresolved(doctored, {"x": results["get_target_bioactivity"]}, _parameters(tool)) == [
            "resolution.ambiguous"
        ]
