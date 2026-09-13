"""Frozen baseline of every resolvable tool surface, across the profile collapse.

The replacement of the six tool profiles by a single `code_execution` boolean had to be
provably non-destructive for `nocode`, and the golden is what proved it: that set is
byte-identical either side of the change. The four legacy values and `None` now resolve to
that same set, and `code` to the surface the boolean's other half selects. Nothing else in the suite compares a whole resolved
tool list against a recorded one — the existing tests assert membership of individual names —
so a profile that silently gained or lost tools during the refactor would pass every one of
them. This file records the lists themselves.

REGENERATE DELIBERATELY, never to make a red test go green:

    UPDATE_TOOL_SURFACE_GOLDEN=1 uv run pytest tests/test_tool_surface_golden.py

which rewrites tests/golden/tool_surface.json in place, so an intended change to the surface
arrives as a reviewable diff in the golden rather than as an edit to an assertion.

The surfaces are resolved under the flag values the manifests set in
genetics-results-suite/k8s/deployments/{chat-backend,mcp-server}.yaml. A flag the manifest
leaves unset is *unset here too* rather than pinned to its current default, so a change to a
default in settings.py moves this baseline the same way it would move production.

The chat half is resolved through the two functions the collapse edits — `LLMService.
resolve_local_tools` and `resolve_proxied_tools` — rather than through `get_anthropic_tools`
underneath them, so a collapse wired correctly in one and wrongly in the other is invisible
to a baseline taken below them. The profile names in `PROFILES` reach the local half only
through `code_execution_requested`, the edge every chat request goes through; the proxied
half takes no surface argument at all, so it is recorded once for the whole chat backend
rather than per profile — six copies of one answer would be six things to keep agreeing.
"""

from __future__ import annotations

import ast
import json
import os
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

import genetics_mcp_server
from genetics_mcp_server import llm_service as llm_service_module
from genetics_mcp_server.config import get_settings
from genetics_mcp_server.llm_service import LLMService, resolve_proxied_tools
from genetics_mcp_server.tools.definitions import (
    all_local_tool_definitions,
    code_execution_requested,
    register_mcp_tools,
)
from genetics_mcp_server.tools.executor import ToolExecutor

GOLDEN_PATH = Path(__file__).parent / "golden" / "tool_surface.json"
UPDATE_ENV_VAR = "UPDATE_TOOL_SURFACE_GOLDEN"

# every environment variable that reaches Settings.disabled_tools. Listed so a surface can be
# resolved from a known-empty starting point: a value inherited from the developer's shell or
# from .env would make the recorded baseline that machine's, not the deployment's.
_DISABLED_TOOLS_ENV_VARS = (
    "ENABLE_CREDIBLE_SETS_STATS",
    "ENABLE_PHENOTYPE_REPORT",
    "ENABLE_SUBAGENTS",
    "ENABLE_LITERATURE_SEARCH",
    "SANDBOX_ENABLED",
    "ALPHAGENOME_ENABLED",
    "ALPHAGENOME_API_KEY",
)

# read off k8s/deployments/chat-backend.yaml and mcp-server.yaml in genetics-results-suite.
# "unset" is the operative half: those manifests name neither ENABLE_CREDIBLE_SETS_STATS,
# ENABLE_PHENOTYPE_REPORT nor ENABLE_LITERATURE_SEARCH, and mcp-server does not name
# SANDBOX_ENABLED either, so each takes its settings.py default in the cluster.
# ALPHAGENOME_ENABLED and ALPHAGENOME_API_KEY together gate the AlphaGenome tools:
# ALPHAGENOME_ENABLED is the deployment's own switch (envsubst'd from terraform at deploy
# time), and ALPHAGENOME_API_KEY is an OPTIONAL secret key on top of it. The baseline
# records the deployment with both set, since the withdrawn direction is the trivial one;
# mcp-server is never given either, and the tool is withheld from /mcp by `_mcp_disabled`
# besides.
DEPLOYED_FLAGS: dict[str, dict[str, str]] = {
    "chat_backend": {
        "SANDBOX_ENABLED": "true",
        "ENABLE_SUBAGENTS": "false",
        "ALPHAGENOME_ENABLED": "true",
        "ALPHAGENOME_API_KEY": "configured",
    },
    "mcp_server": {"ENABLE_SUBAGENTS": "false"},
}

# the profile values a chat request can carry. None is not a profile name, so it is keyed as
# the string "null" for JSON's sake. Every one of them is still recorded even though only two
# distinct surfaces remain: the collapse is a claim about what each stored value resolves to,
# and dropping the four that coincide would delete the evidence for it.
PROFILES: list[tuple[str, str | None]] = [
    ("null", None),
    ("api", "api"),
    ("bigquery", "bigquery"),
    ("rag", "rag"),
    ("nocode", "nocode"),
    ("code", "code"),
]


@contextmanager
def _deployed_env(surface: str):
    """The deployed flags for `surface` in force, yielding the Settings snapshot they build.

    Settings reads os.environ in its field factories, and the production resolvers reach it
    through the `get_settings` lru_cache rather than through a Settings a caller passes in —
    so the environment has to be in place *before* the snapshot is built, and the cache has to
    be dropped on both edges or this records whichever environment happened to fill it first.
    """
    saved = {var: os.environ.get(var) for var in _DISABLED_TOOLS_ENV_VARS}
    try:
        for var in _DISABLED_TOOLS_ENV_VARS:
            os.environ.pop(var, None)
        os.environ.update(DEPLOYED_FLAGS[surface])
        get_settings.cache_clear()
        yield get_settings()
    finally:
        for var, value in saved.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value
        get_settings.cache_clear()


class _ResolverService:
    """Just enough of LLMService to run its real resolver.

    `LLMService.__init__` builds provider clients and dials the external MCP servers, none of
    which the resolution depends on; the two attributes it does read are bound here. The
    functions themselves are the production ones, so the profile gate this baseline exists to
    freeze is the one that ships. `subagent_service` is None to match the deployment, whose
    ENABLE_SUBAGENTS=false leaves the live service unbuilt.
    """

    subagent_service = None

    _disabled_tools = LLMService._disabled_tools
    resolve_local_tools = LLMService.resolve_local_tools


_PROXY_SENTINEL = [{"name": "_present"}]


def _proxied_inclusion() -> dict[str, bool]:
    """Which proxied groups `resolve_proxied_tools` hands a request — any request.

    The proxy registries are empty in a test process, so both groups come back empty whatever
    the rule decided and the decision would be unrecordable. Sentinels make it observable
    while leaving the rule under test the production one.
    """
    with (
        patch.object(
            llm_service_module, "get_external_anthropic_tools", lambda: _PROXY_SENTINEL
        ),
        patch.object(
            llm_service_module, "get_rag_anthropic_tools", lambda: _PROXY_SENTINEL
        ),
    ):
        external, rag = resolve_proxied_tools()
    return {"external_tools": bool(external), "rag_tools": bool(rag)}


def _mcp_hardcoded_exclusions() -> set[str]:
    """The literal half of mcp_server._mcp_disabled, read from the source rather than run.

    The module composes `_settings.disabled_tools | {…}` at import time, so the attribute
    holds the two halves already fused under whatever environment imported it first. Only the
    literal is a property of the code — the other half is the deployment's — and the literal
    is the layer-1 code-execution control that this baseline exists to pin.
    """
    source = (Path(genetics_mcp_server.__file__).parent / "mcp_server.py").read_text()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(t, ast.Name) and t.id == "_mcp_disabled" for t in node.targets
        ):
            continue
        value = node.value
        if not isinstance(value, ast.BinOp) or not isinstance(value.op, ast.BitOr):
            break
        if not isinstance(value.right, ast.Set):
            break
        return {
            elt.value
            for elt in value.right.elts
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
        }
    pytest.fail(
        "mcp_server._mcp_disabled is no longer `<settings half> | {literal}`; this baseline "
        "cannot read the hardcoded exclusions from it. Update the reader, then regenerate."
    )


def _named(names) -> dict[str, object]:
    ordered = sorted(names)
    return {"count": len(ordered), "tools": ordered}


def build_surface() -> dict:
    """Resolve every surface from the code as it stands right now."""
    by_category: dict[str, list[str]] = {}
    for definition in all_local_tool_definitions():
        by_category.setdefault(definition["category"], []).append(definition["name"])

    with _deployed_env("chat_backend"):
        service = _ResolverService()
        # the service's gate, not settings' alone: it also withholds launch_subagents when the
        # subagent service is unbuilt, and that is the set the model is filtered against
        chat_disabled = sorted(service._disabled_tools())
        profiles = {
            key: {
                "local": _named(
                    t["name"]
                    for t in service.resolve_local_tools(
                        code_execution=code_execution_requested(profile)
                    ).definitions
                ),
            }
            for key, profile in PROFILES
        }
        proxied = _proxied_inclusion()

    with _deployed_env("mcp_server") as mcp_settings:
        settings_disabled = sorted(mcp_settings.disabled_tools)
        hardcoded = _mcp_hardcoded_exclusions()
        mcp_disabled = mcp_settings.disabled_tools | hardcoded

    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("Tool surface baseline")
    register_mcp_tools(mcp, ToolExecutor(), disabled_tools=mcp_disabled)
    if not hasattr(mcp, "_tool_manager"):
        # fail rather than skip: this runs inside the fixture every test here depends on, so a
        # skip would take the frozen profile sets down with it and leave the run green
        pytest.fail(
            "FastMCP exposes no _tool_manager; cannot enumerate /mcp tools. Update the "
            "enumeration, then regenerate."
        )
    registered = set(mcp._tool_manager._tools.keys())

    return {
        "local_tool_definitions": {
            "count": sum(len(v) for v in by_category.values()),
            "by_category": {k: _named(v) for k, v in sorted(by_category.items())},
        },
        "chat_backend": {
            "deployed_flags_set": dict(sorted(DEPLOYED_FLAGS["chat_backend"].items())),
            "deployed_flags_unset": sorted(
                set(_DISABLED_TOOLS_ENV_VARS) - set(DEPLOYED_FLAGS["chat_backend"])
            ),
            "disabled_tools": chat_disabled,
            "proxied": proxied,
            "profiles": profiles,
        },
        "mcp_server": {
            "deployed_flags_set": dict(sorted(DEPLOYED_FLAGS["mcp_server"].items())),
            "deployed_flags_unset": sorted(
                set(_DISABLED_TOOLS_ENV_VARS) - set(DEPLOYED_FLAGS["mcp_server"])
            ),
            "settings_disabled_tools": settings_disabled,
            "hardcoded_exclusions": sorted(hardcoded),
            "registered_tools": _named(registered),
        },
    }


@pytest.fixture(scope="module")
def resolved() -> dict:
    surface = build_surface()
    if os.environ.get(UPDATE_ENV_VAR, "").lower() in ("1", "true", "yes"):
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(json.dumps(surface, indent=2, sort_keys=True) + "\n")
    return surface


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(GOLDEN_PATH.read_text())


def _diff(actual: dict, expected: dict) -> str:
    added = sorted(set(actual["tools"]) - set(expected["tools"]))
    removed = sorted(set(expected["tools"]) - set(actual["tools"]))
    return (
        f"count {expected['count']} -> {actual['count']}; "
        f"added {added or 'none'}; removed {removed or 'none'}"
    )


def _profile_diff(actual: dict, expected: dict) -> str:
    return _diff(actual["local"], expected["local"])


class TestResolvedProfiles:
    def test_nocode_matches_the_frozen_set(self, resolved, golden):
        """The one the collapse to a boolean must not move.

        `code_execution=False` is the surface the collapse maps `nocode` onto, so drift here
        is the collapse being destructive rather than a rename — whoever is on the profile.
        """
        actual = resolved["chat_backend"]["profiles"]["nocode"]
        expected = golden["chat_backend"]["profiles"]["nocode"]
        assert actual == expected, "nocode surface moved: " + _profile_diff(actual, expected)

    @pytest.mark.parametrize("key", [key for key, _ in PROFILES])
    def test_every_profile_matches_the_frozen_set(self, key, resolved, golden):
        actual = resolved["chat_backend"]["profiles"][key]
        expected = golden["chat_backend"]["profiles"][key]
        assert actual == expected, f"profile {key} moved: " + _profile_diff(actual, expected)

    def test_the_proxied_groups_are_frozen_and_shared_by_both_surfaces(self, resolved, golden):
        """The second layer the collapse has to get right, and it is now one answer.

        The externals have no route into the sandbox, so withholding them from the code
        surface would leave it with no way to reach them; RAG follows RAG_MCP_SERVER alone.
        Recorded outside `profiles` because the resolver takes no surface argument — a
        per-profile record here would be the shape a per-profile rule grows back into.
        """
        actual = resolved["chat_backend"]["proxied"]
        expected = golden["chat_backend"]["proxied"]
        assert actual == expected, f"proxied groups moved: {expected} -> {actual}"

    @pytest.mark.parametrize("key", ["null", "api", "bigquery", "rag"])
    def test_the_legacy_names_collapsed_onto_nocode(self, key, resolved, golden):
        """The collapse itself, recorded rather than left as a coincidence in the golden.

        Before this change each of these resolved to a different local set; now the shim
        maps every value except "code" onto the no-code surface. The golden would show a
        legacy name drifting away from `nocode` only as an unexplained diff, so it is
        asserted here too. Local tools are all a profile name reaches: the proxied half is
        the same for every request.
        """
        assert (
            resolved["chat_backend"]["profiles"][key]["local"]
            == resolved["chat_backend"]["profiles"]["nocode"]["local"]
        )

    def test_code_is_the_only_value_that_resolves_anywhere_else(self, resolved):
        code = resolved["chat_backend"]["profiles"]["code"]["local"]
        nocode = resolved["chat_backend"]["profiles"]["nocode"]["local"]
        assert code != nocode
        assert "run_analysis" in code["tools"]
        assert "run_analysis" not in nocode["tools"]

    def test_the_profile_set_itself_has_not_changed(self, resolved, golden):
        assert set(resolved["chat_backend"]["profiles"]) == set(
            golden["chat_backend"]["profiles"]
        )

    def test_deployed_disabled_tools_match(self, resolved, golden):
        assert resolved["chat_backend"]["disabled_tools"] == golden["chat_backend"][
            "disabled_tools"
        ]


class TestMCPToolList:
    def test_registered_tools_match_the_frozen_list(self, resolved, golden):
        """Enumerated from the FastMCP instance, not from the exclusion constant.

        A test against `_mcp_disabled` only proves someone typed a name; this proves what the
        registration path actually produced.
        """
        actual = resolved["mcp_server"]["registered_tools"]
        expected = golden["mcp_server"]["registered_tools"]
        assert actual == expected, "/mcp tool list moved: " + _diff(actual, expected)

    def test_hardcoded_exclusions_match(self, resolved, golden):
        assert (
            resolved["mcp_server"]["hardcoded_exclusions"]
            == golden["mcp_server"]["hardcoded_exclusions"]
        )

    def test_every_hardcoded_exclusion_actually_excludes(self, resolved):
        """An excluded name that still registers is an inert control.

        Every handler now consults `disabled_tools` through one gate, so this holds by
        construction rather than by each site having remembered a guard — which is exactly
        why it is still asserted: a site that reached for `mcp.tool()` directly would make a
        name in the literal inert again, and the frozen list above would show it only as a
        regeneration diff nobody has to explain.
        """
        hardcoded = set(resolved["mcp_server"]["hardcoded_exclusions"])
        registered = set(resolved["mcp_server"]["registered_tools"]["tools"])
        assert hardcoded & registered == set()

    def test_code_execution_is_absent(self, resolved):
        registered = set(resolved["mcp_server"]["registered_tools"]["tools"])
        assert "run_analysis" not in registered
        assert "read_artifact" not in registered


class TestLocalDefinitions:
    def test_the_whole_surface_matches(self, resolved, golden):
        """Deep equality, so anything the narrower tests above do not name still fails here."""
        assert resolved == golden

    def test_categories_match(self, resolved, golden):
        assert resolved["local_tool_definitions"] == golden["local_tool_definitions"]
