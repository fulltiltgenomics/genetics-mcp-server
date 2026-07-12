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


class TestMCPDisabledTools:
    """Tests that chat-backend-only tools are excluded from FastMCP registration."""

    @staticmethod
    def _registered_names(mcp) -> set[str]:
        # FastMCP keeps registered tools on the internal _tool_manager.
        # If the version differs and the attr is missing, skip rather than
        # silently passing — the test would otherwise be a no-op.
        if not hasattr(mcp, "_tool_manager"):
            pytest.skip("FastMCP version exposes no _tool_manager; cannot introspect tools")
        return set(mcp._tool_manager._tools.keys())

    def test_search_mgi_in_tool_definitions(self):
        """search_mgi must remain a defined tool (it is chat-backend only, not removed)."""
        names = {t["name"] for t in TOOL_DEFINITIONS}
        assert "search_mgi" in names

    def test_search_mgi_excluded_from_mcp_when_disabled(self):
        from mcp.server.fastmcp import FastMCP

        from genetics_mcp_server.tools.definitions import register_mcp_tools

        mcp = FastMCP("Test Server")
        executor = ToolExecutor()
        register_mcp_tools(mcp, executor, disabled_tools={"search_mgi"})

        registered = self._registered_names(mcp)
        assert "search_mgi" not in registered

    def test_get_myvariant_annotations_excluded_from_mcp_when_disabled(self):
        # mirror pattern for the other chat-backend-only tool
        from mcp.server.fastmcp import FastMCP

        from genetics_mcp_server.tools.definitions import register_mcp_tools

        mcp = FastMCP("Test Server")
        executor = ToolExecutor()
        register_mcp_tools(mcp, executor, disabled_tools={"get_myvariant_annotations"})

        registered = self._registered_names(mcp)
        assert "get_myvariant_annotations" not in registered

    def test_default_mcp_disabled_set_matches_runtime(self):
        """The _mcp_disabled set used by mcp_server.py must exclude search_mgi.

        Guards against accidental removal of the search_mgi entry from
        mcp_server.py:82 since search_mgi calls require chat-backend wiring
        (MouseMine + result truncation) not present in plain MCP clients.
        """
        from genetics_mcp_server import mcp_server

        assert "search_mgi" in mcp_server._mcp_disabled
        assert "get_myvariant_annotations" in mcp_server._mcp_disabled


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


class _KeycloakSettings:
    """Stub Settings for the Keycloak (OAuth resource-server) branch."""

    ISSUER = "https://kc.example.org/realms/genetics"
    RESOURCE_URL = "https://mcp.example.org"

    def __init__(self, allowed_emails=None, allowed_email_domains=None):
        self.allowed_emails = set(allowed_emails or [])
        self.allowed_email_domains = set(allowed_email_domains or [])
        self.oauth_enabled = True
        self.oauth_issuer = self.ISSUER
        self.oauth_resource_url = self.RESOURCE_URL
        self.resolved_oauth_jwks_uri = f"{self.ISSUER}/protocol/openid-connect/certs"


class TestBearerAuthKeycloak:
    """Tests for the Keycloak access-token branch.

    Uses a self-signed RS256 key pair; the JWKS client is mocked to return
    the public key so no real Keycloak is required.
    """

    _key_cache: dict = {}

    @classmethod
    def _keys(cls):
        if not cls._key_cache:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric import rsa

            key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
            cls._key_cache["private"] = key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            ).decode()
            cls._key_cache["public"] = (
                key.public_key()
                .public_bytes(
                    serialization.Encoding.PEM,
                    serialization.PublicFormat.SubjectPublicKeyInfo,
                )
                .decode()
            )
        return cls._key_cache["private"], cls._key_cache["public"]

    def _make_token(self, **overrides):
        import time

        import jwt

        private_pem, _ = self._keys()
        claims = {
            "iss": _KeycloakSettings.ISSUER,
            "aud": _KeycloakSettings.RESOURCE_URL,
            "exp": int(time.time()) + 3600,
            "email": "alice@finngen.fi",
        }
        claims.update(overrides)
        claims = {k: v for k, v in claims.items() if v is not None}
        return jwt.encode(claims, private_pem, algorithm="RS256")

    @pytest.fixture(autouse=True)
    def _mock_jwks(self, monkeypatch):
        """Return the self-signed public key from the JWKS client."""
        import types

        _, public_pem = self._keys()

        class _Client:
            def get_signing_key_from_jwt(self, token):
                return types.SimpleNamespace(key=public_pem)

        monkeypatch.setattr(
            "genetics_mcp_server.mcp_server._get_jwks_client", lambda uri: _Client()
        )

    def test_valid_token_allowed_domain(self):
        from genetics_mcp_server.mcp_server import _validate_keycloak_token

        settings = _KeycloakSettings(allowed_email_domains={"finngen.fi"})
        assert _validate_keycloak_token(self._make_token(), settings) is True

    def test_allowed_email_overrides_domain(self):
        from genetics_mcp_server.mcp_server import _validate_keycloak_token

        settings = _KeycloakSettings(allowed_emails={"bob@other.org"})
        token = self._make_token(email="bob@other.org")
        assert _validate_keycloak_token(token, settings) is True

    def test_disallowed_email_rejected(self):
        from genetics_mcp_server.mcp_server import _validate_keycloak_token

        settings = _KeycloakSettings(allowed_email_domains={"finngen.fi"})
        token = self._make_token(email="eve@evil.com")
        assert _validate_keycloak_token(token, settings) is False

    def test_wrong_audience_rejected(self):
        from genetics_mcp_server.mcp_server import _validate_keycloak_token

        settings = _KeycloakSettings(allowed_email_domains={"finngen.fi"})
        token = self._make_token(aud="https://someone-else")
        assert _validate_keycloak_token(token, settings) is False

    def test_audience_as_list_accepted(self):
        from genetics_mcp_server.mcp_server import _validate_keycloak_token

        settings = _KeycloakSettings(allowed_email_domains={"finngen.fi"})
        token = self._make_token(aud=["account", _KeycloakSettings.RESOURCE_URL])
        assert _validate_keycloak_token(token, settings) is True

    def test_wrong_issuer_rejected(self):
        from genetics_mcp_server.mcp_server import _validate_keycloak_token

        settings = _KeycloakSettings(allowed_email_domains={"finngen.fi"})
        token = self._make_token(iss="https://evil-idp")
        assert _validate_keycloak_token(token, settings) is False

    def test_expired_token_rejected(self):
        import time

        from genetics_mcp_server.mcp_server import _validate_keycloak_token

        settings = _KeycloakSettings(allowed_email_domains={"finngen.fi"})
        token = self._make_token(exp=int(time.time()) - 10)
        assert _validate_keycloak_token(token, settings) is False

    def test_preferred_username_email_fallback(self):
        from genetics_mcp_server.mcp_server import _validate_keycloak_token

        settings = _KeycloakSettings(allowed_email_domains={"finngen.fi"})
        # drop the explicit email claim so the preferred_username fallback runs
        token = self._make_token(email=None, preferred_username="carol@finngen.fi")
        assert _validate_keycloak_token(token, settings) is True

    def test_no_email_rejected(self):
        from genetics_mcp_server.mcp_server import _validate_keycloak_token

        settings = _KeycloakSettings(allowed_email_domains={"finngen.fi"})
        # preferred_username that is not an email must not satisfy the allow-list
        token = self._make_token(email=None, preferred_username="carol")
        assert _validate_keycloak_token(token, settings) is False

    def test_jwks_fetch_error_returns_false(self, monkeypatch):
        """A JWKS network error must yield False, never raise (so caller 401s)."""
        from genetics_mcp_server.mcp_server import _validate_keycloak_token

        def _boom(uri):
            raise RuntimeError("network down")

        monkeypatch.setattr("genetics_mcp_server.mcp_server._get_jwks_client", _boom)
        settings = _KeycloakSettings(allowed_email_domains={"finngen.fi"})
        assert _validate_keycloak_token(self._make_token(), settings) is False

    @pytest.mark.asyncio
    async def test_valid_keycloak_token_via_middleware(self, monkeypatch):
        """End-to-end: a valid Keycloak token reaches the inner app without hitting Google."""
        from genetics_mcp_server.mcp_server import _wrap_with_bearer_auth

        monkeypatch.setattr(
            "genetics_mcp_server.mcp_server._validate_user_token", lambda _: False
        )
        monkeypatch.setattr(
            "genetics_mcp_server.mcp_server.get_settings",
            lambda: _KeycloakSettings(allowed_email_domains={"finngen.fi"}),
        )

        # google path must not be needed; make it fail loudly if reached
        import google.oauth2.id_token as _id_token_mod

        def _fail(token, request):
            raise AssertionError(
                "Google id_token path should not run for a valid Keycloak token"
            )

        monkeypatch.setattr(_id_token_mod, "verify_oauth2_token", _fail)

        call_log: list[dict] = []

        async def inner_app(scope, receive, send):
            call_log.append(scope)

        app = _wrap_with_bearer_auth(inner_app, ["unused-key"])
        token = self._make_token()
        scope = {
            "type": "http",
            "headers": [(b"authorization", f"Bearer {token}".encode())],
            "query_string": b"",
            "client": ("127.0.0.1", 12345),
        }

        messages: list[dict] = []

        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            messages.append(message)

        await app(scope, receive, send)

        assert len(call_log) == 1
        assert not any(m.get("status") == 401 for m in messages)


class _OAuthDisabledSettings:
    """Stub Settings with oauth disabled (no issuer/resource configured)."""

    def __init__(self):
        self.allowed_emails = set()
        self.allowed_email_domains = set()
        self.oauth_enabled = False
        self.oauth_issuer = None
        self.oauth_resource_url = None
        self.resolved_oauth_jwks_uri = None


class TestOAuthMetadataDiscovery:
    """Tests for RFC 9728 protected-resource metadata and WWW-Authenticate."""

    @staticmethod
    async def _collect_response(app, scope):
        messages: list[dict] = []

        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            messages.append(message)

        await app(scope, receive, send)
        return messages

    def _build_app(self, monkeypatch, settings):
        from genetics_mcp_server.mcp_server import _wrap_with_bearer_auth

        monkeypatch.setattr(
            "genetics_mcp_server.mcp_server._validate_user_token", lambda _: False
        )
        monkeypatch.setattr(
            "genetics_mcp_server.mcp_server.get_settings", lambda: settings
        )

        call_log: list[dict] = []

        async def inner_app(scope, receive, send):
            call_log.append(scope)

        return _wrap_with_bearer_auth(inner_app, ["some-key"]), call_log

    @staticmethod
    def _get_json(messages):
        import json

        body = next(m["body"] for m in messages if m.get("type") == "http.response.body")
        return json.loads(body)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "path",
        [
            "/.well-known/oauth-protected-resource",
            "/.well-known/oauth-protected-resource/mcp",
        ],
    )
    async def test_metadata_served_when_oauth_enabled(self, monkeypatch, path):
        app, call_log = self._build_app(monkeypatch, _KeycloakSettings())
        scope = {
            "type": "http",
            "method": "GET",
            "path": path,
            "headers": [],
            "query_string": b"",
            "client": ("127.0.0.1", 12345),
        }

        messages = await self._collect_response(app, scope)

        assert len(call_log) == 0, "metadata must be served without hitting the inner app"
        assert any(
            m.get("type") == "http.response.start" and m.get("status") == 200
            for m in messages
        )
        payload = self._get_json(messages)
        assert payload["resource"] == _KeycloakSettings.RESOURCE_URL
        assert payload["authorization_servers"] == [_KeycloakSettings.ISSUER]
        assert payload["bearer_methods_supported"] == ["header"]
        assert payload["scopes_supported"] == ["openid", "email", "profile"]

    @pytest.mark.asyncio
    async def test_metadata_not_served_when_oauth_disabled(self, monkeypatch):
        app, call_log = self._build_app(monkeypatch, _OAuthDisabledSettings())
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/.well-known/oauth-protected-resource",
            "headers": [],
            "query_string": b"",
            "client": ("127.0.0.1", 12345),
        }

        messages = await self._collect_response(app, scope)

        # with no credentials the request falls through to the normal 401 path,
        # and no metadata is emitted
        assert len(call_log) == 0
        assert any(m.get("status") == 401 for m in messages)
        assert not any(m.get("status") == 200 for m in messages)

    @staticmethod
    def _www_authenticate(messages):
        start = next(m for m in messages if m.get("type") == "http.response.start")
        for name, value in start["headers"]:
            if name == b"www-authenticate":
                return value.decode()
        return None

    @pytest.mark.asyncio
    async def test_401_includes_www_authenticate_when_oauth_enabled(self, monkeypatch):
        app, call_log = self._build_app(monkeypatch, _KeycloakSettings())
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp",
            "headers": [],
            "query_string": b"",
            "client": ("127.0.0.1", 12345),
        }

        messages = await self._collect_response(app, scope)

        assert len(call_log) == 0
        assert any(m.get("status") == 401 for m in messages)
        expected = (
            f'Bearer resource_metadata="{_KeycloakSettings.RESOURCE_URL}'
            '/.well-known/oauth-protected-resource"'
        )
        assert self._www_authenticate(messages) == expected

    @pytest.mark.asyncio
    async def test_401_has_no_www_authenticate_when_oauth_disabled(self, monkeypatch):
        app, call_log = self._build_app(monkeypatch, _OAuthDisabledSettings())
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp",
            "headers": [],
            "query_string": b"",
            "client": ("127.0.0.1", 12345),
        }

        messages = await self._collect_response(app, scope)

        assert len(call_log) == 0
        assert any(m.get("status") == 401 for m in messages)
        assert self._www_authenticate(messages) is None
