"""get_cell_type_expression: the tool, its SDK function, and where it is offered.

results-api is faked at the transport, so the URL, query and status handling are what is
exercised; the endpoint itself is genetics-results-api's `/v1/cell_type_expression/{gene}`.
"""

from urllib.parse import unquote

import httpx
import pytest

from genetics_mcp_server.config.settings import Settings
from genetics_mcp_server.sdk.client import GeneticsClient
from genetics_mcp_server.sdk.errors import GeneticsError
from genetics_mcp_server.tools.definitions import TOOL_DEFINITIONS, get_anthropic_tools
from genetics_mcp_server.tools.executor import ToolExecutor

TOOL = "get_cell_type_expression"

SUMMARY_ROW = {
    "gene_id": "ENSG00000169174.11",
    "gene_name": "PCSK9",
    "cell_type_rollup": "hepatocyte",
    "tissue_general": "liver",
    "n_cells": 5000,
    "n_donors": 12,
    "n_datasets": 3,
    "mean_log1p_cp10k": 1.21,
    "frac_expressing": 0.43,
    "donor_pseudobulk_mean_cp10k": 2.9,
    "donor_pseudobulk_n_donors": 12,
    "tau": 0.91,
}
DATASET_ROWS = [
    {
        "gene_name": "PCSK9",
        "dataset_id": "d1",
        "cell_type": "hepatocyte",
        "mean_log1p_cp10k": 1.3,
        "frac_expressing": 0.5,
        "status": "present",
    },
    {
        "gene_name": "PCSK9",
        "dataset_id": "d2",
        "cell_type": None,
        "mean_log1p_cp10k": 0,
        "frac_expressing": 0,
        "status": "measured_no_row",
    },
]


def _executor(status=200, body=None, text=None, headers=None, seen=None) -> ToolExecutor:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        if text is not None:
            return httpx.Response(status, text=text, headers=headers or {})
        return httpx.Response(status, json=body, headers=headers or {})

    executor = ToolExecutor(api_base_url="http://api.test/api", expose_columns=True)
    executor.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return executor


@pytest.mark.parametrize(("level", "body"), [("summary", [SUMMARY_ROW]), ("dataset", DATASET_ROWS)])
async def test_both_levels_reach_the_endpoint_and_return_rows(level, body):
    seen: list[httpx.Request] = []
    executor = _executor(body=body, headers={"X-Dataset-Version": "2026-10-01"}, seen=seen)
    try:
        result = await executor.get_cell_type_expression("PCSK9", level=level)
    finally:
        await executor.close()
    assert result["success"] is True
    assert result["results"] == body
    assert result["level"] == level
    assert result["release"] == "2026-10-01"
    (request,) = seen
    assert request.url.path == "/api/v1/cell_type_expression/PCSK9"
    assert request.url.params["level"] == level
    assert request.url.params["format"] == "json"


async def test_measured_no_row_is_passed_through_untouched():
    executor = _executor(body=DATASET_ROWS)
    try:
        result = await executor.get_cell_type_expression("PCSK9", level="dataset")
    finally:
        await executor.close()
    assert [r["status"] for r in result["results"]] == ["present", "measured_no_row"]


@pytest.mark.parametrize(
    ("given", "sent"),
    [
        ("ENSG00000169174.11", "ENSG00000169174"),
        ("ensg00000169174.2", "ENSG00000169174"),
        ("ENSG00000169174", "ENSG00000169174"),
        ("PCSK9", "PCSK9"),
        # a symbol with a dot-number is not an Ensembl id and is left alone
        ("HLA-DRB1.1", "HLA-DRB1.1"),
        ("PCSK9, ENSG00000130164.15", "PCSK9,ENSG00000130164"),
    ],
)
async def test_ensembl_version_is_stripped_before_the_call(given, sent):
    seen: list[httpx.Request] = []
    executor = _executor(body=[], seen=seen)
    try:
        await executor.get_cell_type_expression(given)
    finally:
        await executor.close()
    raw_path = seen[0].url.raw_path.decode().split("?")[0]
    assert unquote(raw_path) == f"/api/v1/cell_type_expression/{sent}"


async def test_not_configured_404_says_not_available_on_this_deployment():
    executor = _executor(
        status=404,
        text='{"detail":"Cell-type expression data is not configured for this deployment"}',
    )
    try:
        result = await executor.get_cell_type_expression("PCSK9")
    finally:
        await executor.close()
    assert result["success"] is False
    assert result["error_code"] == "not_available"
    assert "not available on this deployment" in result["error"]


async def test_unknown_gene_404_is_not_reported_as_unavailable():
    executor = _executor(status=404, text='{"detail":"Gene NOPE not found"}')
    try:
        result = await executor.get_cell_type_expression("NOPE")
    finally:
        await executor.close()
    assert result["success"] is False
    assert "error_code" not in result
    assert "NOPE not found" in result["error"]


async def test_503_before_the_first_release_loads():
    executor = _executor(status=503, text='{"detail":"no cell-type expression release is loaded"}')
    try:
        result = await executor.get_cell_type_expression("PCSK9")
    finally:
        await executor.close()
    assert result["success"] is False
    assert result["error_code"] == "release_not_loaded"
    assert "temporarily unavailable" in result["error"]


async def test_bad_level_is_refused_without_a_request():
    seen: list[httpx.Request] = []
    executor = _executor(body=[], seen=seen)
    try:
        result = await executor.get_cell_type_expression("PCSK9", level="cell")
    finally:
        await executor.close()
    assert result["success"] is False
    assert seen == []


def _sdk(status=200, body=None, text=None, headers=None) -> GeneticsClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if text is not None:
            return httpx.Response(status, text=text)
        return httpx.Response(status, json=body, headers=headers or {})

    client = GeneticsClient()
    client._executor.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


async def test_sdk_function_returns_a_frame():
    frame = await _sdk(body=DATASET_ROWS).cell_type_expression(
        "ENSG00000169174.11", level="dataset"
    )
    assert frame.height == 2
    assert frame["status"].to_list() == ["present", "measured_no_row"]


async def test_sdk_function_raises_on_a_deployment_without_the_data():
    client = _sdk(status=404, text='{"detail":"... not configured for this deployment"}')
    with pytest.raises(GeneticsError, match="not available on this deployment") as info:
        await client.cell_type_expression("PCSK9")
    assert info.value.code == "not_available"


def test_sdk_exports_it_under_a_name_that_is_not_a_view_stem():
    import genetics_mcp_server.sdk as sdk

    assert "cell_type_expression" in sdk.__all__
    assert callable(sdk.cell_type_expression)
    assert not any(name.lower().startswith("cellxgene") for name in sdk.__all__)


def test_tool_text_names_no_gated_view():
    tool = next(t for t in TOOL_DEFINITIONS if t["name"] == TOOL)
    text = str(tool).lower()
    assert "cellxgene_" not in text
    assert "tpm" in text and "log1p cp10k" in text and "measured_no_row" in text
    gene_expr = next(t for t in TOOL_DEFINITIONS if t["name"] == "get_gene_expression")
    assert TOOL in gene_expr["description"]


@pytest.mark.parametrize(
    ("profile", "enabled", "offered"),
    [
        ("daly", True, True),
        ("daly", False, False),
        ("finngen", True, False),
        ("finngen", False, False),
        (None, True, False),
        ("unknown", True, False),
    ],
)
def test_offered_only_where_enabled_on_a_profile_that_serves_the_data(profile, enabled, offered):
    disabled = Settings(config_profile=profile, cell_type_expression_enabled=enabled).disabled_tools
    names = {t["name"] for t in get_anthropic_tools(code_execution=False, disabled_tools=disabled)}
    assert (TOOL in names) is offered


def test_sdk_replaceable_so_the_code_surface_uses_the_sdk():
    names = {t["name"] for t in get_anthropic_tools(code_execution=True)}
    assert TOOL not in names


def test_reads_config_profile_and_enabled_flag_from_the_environment(monkeypatch):
    monkeypatch.setenv("CONFIG_PROFILE", "daly")
    monkeypatch.setenv("CELL_TYPE_EXPRESSION_ENABLED", "true")
    assert TOOL not in Settings().disabled_tools
    monkeypatch.setenv("CELL_TYPE_EXPRESSION_ENABLED", "false")
    assert TOOL in Settings().disabled_tools
    monkeypatch.delenv("CELL_TYPE_EXPRESSION_ENABLED")
    assert TOOL in Settings().disabled_tools
    monkeypatch.setenv("CELL_TYPE_EXPRESSION_ENABLED", "true")
    monkeypatch.setenv("CONFIG_PROFILE", "finngen")
    assert TOOL in Settings().disabled_tools
    monkeypatch.delenv("CONFIG_PROFILE")
    assert TOOL in Settings().disabled_tools
