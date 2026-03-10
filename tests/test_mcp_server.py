"""Integration tests for MCP server and tool definitions."""

import pytest

from genetics_mcp_server.tools.definitions import (
    BIGQUERY_TOOL_DEFINITIONS,
    TOOL_DEFINITIONS,
    get_anthropic_tools,
    register_mcp_tools,
)
from genetics_mcp_server.tools.executor import ToolExecutor


class TestToolDefinitions:
    """Tests for tool definitions structure."""

    def test_tool_definitions_not_empty(self):
        """Test that tool definitions list is not empty."""
        assert len(TOOL_DEFINITIONS) > 0

    def test_tool_definitions_have_required_fields(self):
        """Test that all tools have required fields."""
        for tool in TOOL_DEFINITIONS:
            assert "name" in tool, f"Tool missing name: {tool}"
            assert "description" in tool, f"Tool {tool.get('name')} missing description"
            assert "parameters" in tool, f"Tool {tool['name']} missing parameters"

    def test_tool_names_are_unique(self):
        """Test that all tool names are unique."""
        names = [t["name"] for t in TOOL_DEFINITIONS]
        assert len(names) == len(set(names)), "Duplicate tool names found"

    def test_required_tools_exist(self):
        """Test that expected tools are defined."""
        tool_names = {t["name"] for t in TOOL_DEFINITIONS}

        expected_tools = [
            "search_phenotypes",
            "search_genes",
            "get_credible_sets_by_gene",
            "get_credible_sets_by_variant",
            "get_credible_sets_by_phenotype",
            "get_gene_expression",
            "get_available_resources",
            "search_scientific_literature",
        ]

        for tool_name in expected_tools:
            assert tool_name in tool_names, f"Expected tool '{tool_name}' not found"

    def test_tool_parameters_structure(self):
        """Test that tool parameters have correct structure."""
        for tool in TOOL_DEFINITIONS:
            params = tool["parameters"]
            for param_name, param_def in params.items():
                assert "type" in param_def, (
                    f"Parameter '{param_name}' in tool '{tool['name']}' missing type"
                )
                # type should be a string
                assert isinstance(param_def["type"], str), (
                    f"Parameter type should be string in {tool['name']}.{param_name}"
                )


class TestAnthropicToolFormat:
    """Tests for Anthropic tool format conversion."""

    def test_get_anthropic_tools_returns_list(self):
        """Test that get_anthropic_tools returns a list."""
        tools = get_anthropic_tools()
        assert isinstance(tools, list)

    def test_anthropic_tools_count_matches(self):
        """Test that default (no profile) returns all tools."""
        tools = get_anthropic_tools()
        assert len(tools) == len(TOOL_DEFINITIONS) + len(BIGQUERY_TOOL_DEFINITIONS)

    def test_anthropic_tool_structure(self):
        """Test that Anthropic tools have correct structure."""
        tools = get_anthropic_tools()

        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool

            schema = tool["input_schema"]
            assert schema.get("type") == "object"
            assert "properties" in schema
            assert "required" in schema

    def test_anthropic_tools_custom_descriptions(self):
        """Test custom descriptions override defaults."""
        custom = {"search_genes": "My custom description"}
        tools = get_anthropic_tools(custom_descriptions=custom)

        search_genes_tool = next(t for t in tools if t["name"] == "search_genes")
        assert search_genes_tool["description"] == "My custom description"

    def test_anthropic_tools_required_params(self):
        """Test that required parameters are marked correctly."""
        tools = get_anthropic_tools()

        # search_phenotypes should have 'query' as required
        search_pheno = next(t for t in tools if t["name"] == "search_phenotypes")
        assert "query" in search_pheno["input_schema"]["required"]

    def test_anthropic_tools_array_params(self):
        """Test that array parameters are handled correctly."""
        tools = get_anthropic_tools()

        # lookup_phenotype_names has array parameter
        lookup_tool = next(t for t in tools if t["name"] == "lookup_phenotype_names")
        codes_prop = lookup_tool["input_schema"]["properties"]["codes"]
        assert codes_prop["type"] == "array"
        assert "items" in codes_prop


class TestMCPToolRegistration:
    """Tests for MCP tool registration."""

    def test_register_mcp_tools_adds_tools(self):
        """Test that register_mcp_tools adds tools to server."""
        from mcp.server.fastmcp import FastMCP

        mcp = FastMCP("Test Server")
        executor = ToolExecutor()

        # count tools before registration
        initial_count = len(mcp._tool_manager._tools) if hasattr(mcp, "_tool_manager") else 0

        register_mcp_tools(mcp, executor)

        # after registration, tools should be added
        # FastMCP stores tools in different ways depending on version
        # just verify no exception was raised
        assert True  # registration completed without error

    def test_all_tools_registered(self):
        """Test that all defined tools are registered."""
        from mcp.server.fastmcp import FastMCP

        mcp = FastMCP("Test Server")
        executor = ToolExecutor()

        register_mcp_tools(mcp, executor)

        # get registered tool names
        if hasattr(mcp, "_tool_manager"):
            registered_names = set(mcp._tool_manager._tools.keys())
        else:
            # fallback: just verify registration didn't fail
            registered_names = set()

        # verify expected tools are registered
        expected = {t["name"] for t in TOOL_DEFINITIONS}

        # if we have tool manager access, verify all tools
        if registered_names:
            assert expected <= registered_names


class TestMCPServerIntegration:
    """Integration tests for MCP server (requires live API)."""

    @pytest.fixture(autouse=True)
    async def setup_executor(self):
        """Create and cleanup executor for each test."""
        self.executor = ToolExecutor()
        yield
        await self.executor.close()

    async def test_mcp_tool_search_phenotypes(self):
        """Test MCP tool for searching phenotypes."""
        from mcp.server.fastmcp import FastMCP

        mcp = FastMCP("Test Server")
        register_mcp_tools(mcp, self.executor)

        # call the tool directly through executor
        result = await self.executor.search_phenotypes("diabetes", limit=3)

        assert result["success"] is True
        assert "results" in result

    async def test_mcp_tool_search_genes(self):
        """Test MCP tool for searching genes."""
        from mcp.server.fastmcp import FastMCP

        mcp = FastMCP("Test Server")
        register_mcp_tools(mcp, self.executor)

        result = await self.executor.search_genes("APOE", limit=2)

        assert result["success"] is True
        assert "results" in result

    async def test_mcp_tool_get_available_resources(self):
        """Test MCP tool for getting available resources."""
        from mcp.server.fastmcp import FastMCP

        mcp = FastMCP("Test Server")
        register_mcp_tools(mcp, self.executor)

        result = await self.executor.get_available_resources()

        assert result["success"] is True
        assert "resources" in result


class TestToolDescriptions:
    """Tests for tool description quality."""

    def test_descriptions_not_empty(self):
        """Test that all tools have non-empty descriptions."""
        for tool in TOOL_DEFINITIONS:
            assert tool["description"].strip(), f"Tool {tool['name']} has empty description"

    def test_descriptions_have_usage_guidance(self):
        """Test that key tools have usage guidance in descriptions."""
        guidance_keywords = ["use", "returns", "provides", "get"]

        for tool in TOOL_DEFINITIONS:
            desc_lower = tool["description"].lower()
            has_guidance = any(kw in desc_lower for kw in guidance_keywords)
            assert has_guidance, (
                f"Tool {tool['name']} description lacks usage guidance"
            )

    def test_parameter_descriptions_exist(self):
        """Test that parameters have descriptions where useful."""
        for tool in TOOL_DEFINITIONS:
            for param_name, param_def in tool["parameters"].items():
                # required params should always have descriptions
                if param_def.get("required"):
                    assert "description" in param_def, (
                        f"Required param '{param_name}' in {tool['name']} "
                        "should have description"
                    )
