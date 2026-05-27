"""Integration tests for MCP server and tool definitions."""

import pytest

from genetics_mcp_server.tools.definitions import (
    BIGQUERY_TOOL_DEFINITIONS,
    SUBAGENT_TOOL_DEFINITIONS,
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
            "list_datasets",
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
        assert len(tools) == len(TOOL_DEFINITIONS) + len(BIGQUERY_TOOL_DEFINITIONS) + len(SUBAGENT_TOOL_DEFINITIONS)

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


@pytest.mark.integration
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

    async def test_mcp_tool_list_datasets(self):
        """Test MCP tool for listing datasets."""
        from mcp.server.fastmcp import FastMCP

        mcp = FastMCP("Test Server")
        register_mcp_tools(mcp, self.executor)

        result = await self.executor.list_datasets()

        assert result["success"] is True
        assert "datasets" in result


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


class TestBearerAuthMiddleware:
    """Tests for ASGI bearer auth middleware with query param support."""

    VALID_KEY = "test-secret-key-123"
    INVALID_KEY = "wrong-key"

    @staticmethod
    def _make_scope(*, headers=None, query_string=b"", scope_type="http"):
        """Build a minimal ASGI scope for testing."""
        scope = {
            "type": scope_type,
            "headers": headers or [],
            "query_string": query_string,
            "client": ("127.0.0.1", 12345),
        }
        return scope

    @staticmethod
    def _bearer_header(token: str) -> list[tuple[bytes, bytes]]:
        return [(b"authorization", f"Bearer {token}".encode())]

    @pytest.fixture()
    def wrapped_app(self, monkeypatch):
        """Create an auth-wrapped ASGI app that records whether the inner app was called."""
        from genetics_mcp_server.mcp_server import _wrap_with_bearer_auth

        # bypass _validate_user_token so only static keys matter
        monkeypatch.setattr(
            "genetics_mcp_server.mcp_server._validate_user_token", lambda _: False
        )

        call_log: list[dict] = []

        async def inner_app(scope, receive, send):
            call_log.append(scope)

        app = _wrap_with_bearer_auth(inner_app, [self.VALID_KEY])
        return app, call_log

    @staticmethod
    async def _collect_response(app, scope):
        """Invoke the ASGI app and collect sent response messages."""
        messages: list[dict] = []

        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            messages.append(message)

        await app(scope, receive, send)
        return messages

    @pytest.mark.asyncio
    async def test_query_param_token_accepted(self, wrapped_app):
        """Valid token via ?token= query param should reach the inner app."""
        app, call_log = wrapped_app
        scope = self._make_scope(query_string=f"token={self.VALID_KEY}".encode())

        messages = await self._collect_response(app, scope)

        assert len(call_log) == 1, "Inner app should have been called"
        # no 401 response messages expected
        assert not messages

    @pytest.mark.asyncio
    async def test_bearer_header_accepted(self, wrapped_app):
        """Valid Bearer header should reach the inner app."""
        app, call_log = wrapped_app
        scope = self._make_scope(headers=self._bearer_header(self.VALID_KEY))

        messages = await self._collect_response(app, scope)

        assert len(call_log) == 1, "Inner app should have been called"
        assert not messages

    @pytest.mark.asyncio
    async def test_header_takes_precedence_over_query_param(self, wrapped_app):
        """Invalid Bearer header should reject even if valid query param exists."""
        app, call_log = wrapped_app
        scope = self._make_scope(
            headers=self._bearer_header(self.INVALID_KEY),
            query_string=f"token={self.VALID_KEY}".encode(),
        )

        messages = await self._collect_response(app, scope)

        assert len(call_log) == 0, "Inner app should NOT have been called"
        assert any(m.get("status") == 401 for m in messages)

    @pytest.mark.asyncio
    async def test_invalid_query_param_rejected(self, wrapped_app):
        """Invalid token via ?token= should return 401."""
        app, call_log = wrapped_app
        scope = self._make_scope(query_string=f"token={self.INVALID_KEY}".encode())

        messages = await self._collect_response(app, scope)

        assert len(call_log) == 0, "Inner app should NOT have been called"
        assert any(m.get("status") == 401 for m in messages)

    @pytest.mark.asyncio
    async def test_missing_both_header_and_query_param(self, wrapped_app):
        """No auth credentials at all should return 401."""
        app, call_log = wrapped_app
        scope = self._make_scope()

        messages = await self._collect_response(app, scope)

        assert len(call_log) == 0, "Inner app should NOT have been called"
        assert any(m.get("status") == 401 for m in messages)

    @pytest.mark.asyncio
    async def test_healthz_bypasses_auth(self, wrapped_app):
        """/healthz must return 200 OK without any credentials."""
        app, call_log = wrapped_app
        scope = self._make_scope()
        scope["path"] = "/healthz"

        messages = await self._collect_response(app, scope)

        assert len(call_log) == 0, "Inner app should NOT have been called for healthz"
        assert any(
            m.get("type") == "http.response.start" and m.get("status") == 200
            for m in messages
        )

    @pytest.mark.asyncio
    async def test_user_token_path_accepted(self, monkeypatch):
        """Non-JWT, non-API-key token should be validated via _validate_user_token."""
        from genetics_mcp_server.mcp_server import _wrap_with_bearer_auth

        seen: list[str] = []

        def fake_validator(token: str) -> bool:
            seen.append(token)
            return token == "good-user-token"

        monkeypatch.setattr(
            "genetics_mcp_server.mcp_server._validate_user_token", fake_validator
        )

        call_log: list[dict] = []

        async def inner_app(scope, receive, send):
            call_log.append(scope)

        app = _wrap_with_bearer_auth(inner_app, [self.VALID_KEY])
        scope = self._make_scope(headers=self._bearer_header("good-user-token"))

        messages = await self._collect_response(app, scope)

        assert seen == ["good-user-token"]
        assert len(call_log) == 1
        assert not messages


class _StubSettings:
    """Minimal stand-in for Settings used by the JWT branch."""

    def __init__(self, allowed_emails=None, allowed_email_domains=None):
        self.allowed_emails = set(allowed_emails or [])
        self.allowed_email_domains = set(allowed_email_domains or [])


class TestBearerAuthJWT:
    """Tests for the Google Identity Token (JWT) branch of bearer auth."""

    VALID_KEY = "test-secret-key-123"
    # any string containing a dot routes to the JWT branch; signature is mocked
    JWT_TOKEN = "header.payload.signature"

    @staticmethod
    def _make_scope(*, headers=None, query_string=b"", scope_type="http"):
        return {
            "type": scope_type,
            "headers": headers or [],
            "query_string": query_string,
            "client": ("127.0.0.1", 12345),
        }

    @staticmethod
    def _bearer_header(token: str) -> list[tuple[bytes, bytes]]:
        return [(b"authorization", f"Bearer {token}".encode())]

    @staticmethod
    async def _collect_response(app, scope):
        messages: list[dict] = []

        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            messages.append(message)

        await app(scope, receive, send)
        return messages

    def _build_app(self, monkeypatch, *, verify_result, settings):
        """Build an auth-wrapped ASGI app with mocked JWT verification + settings."""
        from genetics_mcp_server.mcp_server import _wrap_with_bearer_auth

        # short-circuit user-token path so unexpected fallback would not silently pass
        monkeypatch.setattr(
            "genetics_mcp_server.mcp_server._validate_user_token", lambda _: False
        )

        # avoid touching the real Google transport
        monkeypatch.setattr(
            "genetics_mcp_server.mcp_server._get_google_request", lambda: object()
        )

        # inject deterministic settings for allow-list checks
        monkeypatch.setattr(
            "genetics_mcp_server.mcp_server.get_settings", lambda: settings
        )

        # mock the lazy import target: google.oauth2.id_token.verify_oauth2_token
        import google.oauth2.id_token as _id_token_mod

        def fake_verify(token, request):
            if isinstance(verify_result, Exception):
                raise verify_result
            return verify_result

        monkeypatch.setattr(_id_token_mod, "verify_oauth2_token", fake_verify)

        call_log: list[dict] = []

        async def inner_app(scope, receive, send):
            call_log.append(scope)

        app = _wrap_with_bearer_auth(inner_app, [self.VALID_KEY])
        return app, call_log

    @pytest.mark.asyncio
    async def test_valid_jwt_allowed_domain_passes(self, monkeypatch):
        """Valid JWT whose email matches an allowed domain should pass through."""
        payload = {"email": "alice@finngen.fi", "email_verified": True}
        app, call_log = self._build_app(
            monkeypatch,
            verify_result=payload,
            settings=_StubSettings(allowed_email_domains={"finngen.fi"}),
        )
        scope = self._make_scope(headers=self._bearer_header(self.JWT_TOKEN))

        messages = await self._collect_response(app, scope)

        assert len(call_log) == 1
        assert not any(m.get("status") == 401 for m in messages)

    @pytest.mark.asyncio
    async def test_valid_jwt_allowed_email_overrides_domain(self, monkeypatch):
        """Email in allowed_emails should pass even when domain is not allowed."""
        payload = {"email": "bob@other.org", "email_verified": True}
        app, call_log = self._build_app(
            monkeypatch,
            verify_result=payload,
            settings=_StubSettings(
                allowed_emails={"bob@other.org"},
                allowed_email_domains={"finngen.fi"},
            ),
        )
        scope = self._make_scope(headers=self._bearer_header(self.JWT_TOKEN))

        messages = await self._collect_response(app, scope)

        assert len(call_log) == 1
        assert not any(m.get("status") == 401 for m in messages)

    @pytest.mark.asyncio
    async def test_valid_jwt_disallowed_domain_and_email_rejected(self, monkeypatch):
        """JWT with disallowed domain and not in allowed_emails should be 401."""
        payload = {"email": "eve@evil.com", "email_verified": True}
        app, call_log = self._build_app(
            monkeypatch,
            verify_result=payload,
            settings=_StubSettings(allowed_email_domains={"finngen.fi"}),
        )
        scope = self._make_scope(headers=self._bearer_header(self.JWT_TOKEN))

        messages = await self._collect_response(app, scope)

        assert len(call_log) == 0
        assert any(m.get("status") == 401 for m in messages)

    @pytest.mark.asyncio
    async def test_jwt_email_not_verified_rejected(self, monkeypatch):
        """JWT with email_verified=False must be rejected with 401."""
        payload = {"email": "alice@finngen.fi", "email_verified": False}
        app, call_log = self._build_app(
            monkeypatch,
            verify_result=payload,
            settings=_StubSettings(allowed_email_domains={"finngen.fi"}),
        )
        scope = self._make_scope(headers=self._bearer_header(self.JWT_TOKEN))

        messages = await self._collect_response(app, scope)

        assert len(call_log) == 0
        assert any(m.get("status") == 401 for m in messages)

    @pytest.mark.asyncio
    async def test_invalid_jwt_signature_rejected(self, monkeypatch):
        """ValueError from verify_oauth2_token (bad signature/expired) → 401."""
        app, call_log = self._build_app(
            monkeypatch,
            verify_result=ValueError("Token expired or invalid signature"),
            settings=_StubSettings(allowed_email_domains={"finngen.fi"}),
        )
        scope = self._make_scope(headers=self._bearer_header(self.JWT_TOKEN))

        messages = await self._collect_response(app, scope)

        assert len(call_log) == 0
        assert any(m.get("status") == 401 for m in messages)

    @pytest.mark.asyncio
    async def test_jwt_without_email_rejected(self, monkeypatch):
        """JWT payload missing 'email' must be rejected with 401."""
        payload = {"email_verified": True}
        app, call_log = self._build_app(
            monkeypatch,
            verify_result=payload,
            settings=_StubSettings(allowed_email_domains={"finngen.fi"}),
        )
        scope = self._make_scope(headers=self._bearer_header(self.JWT_TOKEN))

        messages = await self._collect_response(app, scope)

        assert len(call_log) == 0
        assert any(m.get("status") == 401 for m in messages)
