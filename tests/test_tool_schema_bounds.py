"""The bounds a tool schema declares are claims about enforcement (4h6.70).

`get_anthropic_tools` used to emit only type/description/default/items/enum, so every
numeric range a description stated in prose was advisory. These tests pin the three
keywords it now forwards, tie each declared bound back to the constant the server
enforces it with, and hold the line that a parameter which declares nothing still emits
exactly what it emitted before.
"""

import pytest
from pydantic import ValidationError

from genetics_mcp_server import sandbox_client
from genetics_mcp_server.tools.definitions import (
    BIGQUERY_TOOL_DEFINITIONS,
    SUBAGENT_TOOL_DEFINITIONS,
    TOOL_DEFINITIONS,
    get_anthropic_tools,
)
from genetics_mcp_server.tools.executor import ToolExecutor

# tool, parameter, expected {keyword: value}. Every row is a bound the server applies
# today; the file:line of the enforcing code is named in definitions.py beside each one.
EXPECTED_BOUNDS = [
    ("get_asm_qtl_by_gene", "window", {"minimum": 0, "maximum": 10_000_000}),
    ("get_open_chromatin_by_gene", "window", {"minimum": 0, "maximum": 10_000_000}),
    ("get_variant_effect_by_gene", "window", {"minimum": 0, "maximum": 10_000_000}),
    ("get_mpra_by_gene", "window", {"minimum": 0, "maximum": 10_000_000}),
    ("get_mpra_pip_concordance_by_gene", "window", {"minimum": 0, "maximum": 10_000_000}),
    ("get_mpra_pip_concordance_by_gene", "min_pip", {"minimum": 0.0, "maximum": 1.0}),
    ("get_hla_by_allele", "max_rows", {"minimum": 1, "maximum": 100_000}),
    ("run_analysis", "timeout_s", {"minimum": 1, "maximum": 120}),
    ("web_search", "max_results", {"maximum": 10}),
    ("search_mgi", "max_results", {"minimum": 1, "maximum": 100}),
    ("search_cbioportal", "max_results", {"minimum": 1, "maximum": 100}),
    ("search_uniprot", "size", {"maximum": 500}),
    ("get_drug_targets_for_gene", "min_phase", {"minimum": 0, "maximum": 4}),
    ("get_drug_targets_for_gene", "max_results", {"minimum": 1, "maximum": 100}),
    ("get_target_bioactivity", "pchembl_min", {"minimum": 0, "maximum": 14}),
    ("get_target_bioactivity", "max_results", {"minimum": 1, "maximum": 100}),
]


def _schemas():
    return {t["name"]: t["input_schema"] for t in get_anthropic_tools()}


@pytest.mark.parametrize("tool_name,param,expected", EXPECTED_BOUNDS)
def test_declared_bounds_reach_the_emitted_schema(tool_name, param, expected):
    prop = _schemas()[tool_name]["properties"][param]
    for keyword, value in expected.items():
        assert prop[keyword] == value, f"{tool_name}.{param} {keyword}"


def test_bounds_track_the_constants_that_enforce_them():
    """A literal in definitions.py and the constant it mirrors must not drift apart."""
    props = _schemas()
    assert (
        props["get_asm_qtl_by_gene"]["properties"]["window"]["maximum"]
        == ToolExecutor._MAX_SQL_WINDOW
    )
    assert (
        props["get_hla_by_allele"]["properties"]["max_rows"]["maximum"]
        == ToolExecutor._MAX_SQL_LIMIT
    )
    assert (
        props["run_analysis"]["properties"]["timeout_s"]["maximum"]
        == sandbox_client.MAX_TIMEOUT_S
    )


def test_no_declared_bound_contradicts_its_own_default():
    for tool in TOOL_DEFINITIONS + BIGQUERY_TOOL_DEFINITIONS + SUBAGENT_TOOL_DEFINITIONS:
        for name, info in tool.get("parameters", {}).items():
            if "default" not in info or info["default"] is None:
                continue
            where = f"{tool['name']}.{name}"
            if "minimum" in info:
                assert info["default"] >= info["minimum"], where
            if "maximum" in info:
                assert info["default"] <= info["maximum"], where


def test_a_parameter_without_bounds_emits_an_unchanged_schema():
    """The builder must add nothing to a parameter that declares nothing.

    `query_database.max_rows` is the deliberate abstainer: its cap lives downstream in
    db-api, not here, so it carries no bound and its schema is exactly the four keys the
    pre-4h6.70 builder produced.
    """
    prop = _schemas()["query_database"]["properties"]["max_rows"]
    assert set(prop) == {"type", "description", "default"}
    assert prop["type"] == "integer"
    assert prop["default"] == 1000


def test_search_scientific_literature_max_results_stays_unbounded():
    """Its "max 25" holds only on the europepmc path; perplexity is the default backend."""
    prop = _schemas()["search_scientific_literature"]["properties"]["max_results"]
    assert "maximum" not in prop
    assert "minimum" not in prop


def test_pattern_is_forwarded_when_a_parameter_declares_one(monkeypatch):
    """No shipped parameter declares `pattern`, so the keyword is exercised synthetically."""
    fake = [
        {
            "name": "synthetic_tool",
            "category": "general",
            "description": "synthetic",
            "parameters": {
                "code": {
                    "type": "string",
                    "description": "synthetic",
                    "pattern": "^[A-Z]{2}$",
                    "required": True,
                },
                "plain": {"type": "string", "description": "synthetic"},
            },
        }
    ]
    monkeypatch.setattr("genetics_mcp_server.tools.definitions.TOOL_DEFINITIONS", fake)
    monkeypatch.setattr("genetics_mcp_server.tools.definitions.BIGQUERY_TOOL_DEFINITIONS", [])
    monkeypatch.setattr("genetics_mcp_server.tools.definitions.SUBAGENT_TOOL_DEFINITIONS", [])
    props = get_anthropic_tools()[0]["input_schema"]["properties"]
    assert props["code"]["pattern"] == "^[A-Z]{2}$"
    assert set(props["plain"]) == {"type", "description"}


def test_zero_valued_bounds_survive_the_builder(monkeypatch):
    """`0` and `0.0` are legitimate bounds — a truthiness test would silently drop them."""
    fake = [
        {
            "name": "synthetic_tool",
            "category": "general",
            "description": "synthetic",
            "parameters": {
                "n": {"type": "integer", "minimum": 0, "maximum": 0},
                "x": {"type": "number", "minimum": 0.0},
            },
        }
    ]
    monkeypatch.setattr("genetics_mcp_server.tools.definitions.TOOL_DEFINITIONS", fake)
    monkeypatch.setattr("genetics_mcp_server.tools.definitions.BIGQUERY_TOOL_DEFINITIONS", [])
    monkeypatch.setattr("genetics_mcp_server.tools.definitions.SUBAGENT_TOOL_DEFINITIONS", [])
    props = get_anthropic_tools()[0]["input_schema"]["properties"]
    assert props["n"]["minimum"] == 0 and props["n"]["maximum"] == 0
    assert props["x"]["minimum"] == 0.0


class TestTheMcpSurface:
    """FastMCP builds its schemas from the signatures, so it is a second, validating path."""

    @staticmethod
    def _mcp_schemas():
        import asyncio

        from mcp.server.fastmcp import FastMCP

        from genetics_mcp_server.tools.definitions import register_mcp_tools

        class _StubExecutor:
            def __getattr__(self, _name):
                async def _call(*_args, **_kwargs):
                    return {}

                return _call

        mcp = FastMCP("bounds-test")
        register_mcp_tools(mcp, _StubExecutor())
        return {t.name: t.inputSchema for t in asyncio.run(mcp.list_tools())}

    def test_reject_already_parameters_declare_their_bounds(self):
        schemas = self._mcp_schemas()
        window = schemas["get_asm_qtl_by_gene"]["properties"]["window"]
        assert window["minimum"] == 0
        assert window["maximum"] == ToolExecutor._MAX_SQL_WINDOW
        min_pip = schemas["get_mpra_pip_concordance_by_gene"]["properties"]["min_pip"]
        assert (min_pip["minimum"], min_pip["maximum"]) == (0.0, 1.0)
        max_rows = schemas["get_hla_by_allele"]["properties"]["max_rows"]
        assert max_rows["minimum"] == 1
        assert max_rows["maximum"] == ToolExecutor._MAX_SQL_LIMIT

    def test_clamped_parameters_stay_bare_here(self):
        """Declaring a clamp on this surface would turn a working call into an error."""
        schemas = self._mcp_schemas()
        for tool, param in (
            ("web_search", "max_results"),
            ("search_mgi", "max_results"),
            ("search_cbioportal", "max_results"),
            ("search_uniprot", "size"),
            ("get_drug_targets_for_gene", "min_phase"),
            ("get_drug_targets_for_gene", "max_results"),
            ("get_target_bioactivity", "pchembl_min"),
            ("get_target_bioactivity", "max_results"),
        ):
            prop = schemas[tool]["properties"][param]
            assert "maximum" not in prop, f"{tool}.{param}"
            assert "minimum" not in prop, f"{tool}.{param}"


def test_the_sql_sites_really_do_reject_out_of_range_values():
    """The premise behind the pydantic annotations: these are rejections, not clamps."""
    from genetics_mcp_server.tools.sql_safety import SqlValueError, sql_float, sql_int

    with pytest.raises(SqlValueError):
        sql_int(10_000_001, name="window", minimum=0, maximum=ToolExecutor._MAX_SQL_WINDOW)
    with pytest.raises(SqlValueError):
        sql_int(0, name="max_rows", minimum=1, maximum=ToolExecutor._MAX_SQL_LIMIT)
    with pytest.raises(SqlValueError):
        sql_float(1.5, name="min_pip", minimum=0.0, maximum=1.0)


def test_sandbox_client_rejects_timeouts_outside_the_declared_range():
    """The other premise: run_analysis's 1-120 is enforcement, not prose."""
    assert sandbox_client.MAX_TIMEOUT_S == 120
    with pytest.raises((sandbox_client.SandboxRejected, ValidationError, ValueError)):
        sandbox_client.SandboxClient._validate(
            object.__new__(sandbox_client.SandboxClient),
            "print(1)",
            100_000,
            "u",
            "s",
            "e",
        )
