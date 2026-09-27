"""The external MCP proxy: entry syntax, endpoint path, and the OAuth refresh-token source.

Everything runs against an in-process httpx.MockTransport standing in for both the MCP
server and the OAuth token endpoint; nothing opens a socket.
"""

import json
import time

import httpx
import pytest

from genetics_mcp_server import mcp_proxy
from genetics_mcp_server.mcp_proxy import (
    MCPProxyClient,
    OAuthRefreshTokenSource,
    ServerConfig,
    build_proxy_client,
)

TOKEN_ENDPOINT = "https://issuer.invalid/oauth2/token"


class _Server:
    """A scripted MCP server plus token endpoint; records what it was sent."""

    def __init__(self, tools=("query_kb", "run_pipeline")):
        self.requests: list[httpx.Request] = []
        self.tools = [{"name": t, "description": "d", "inputSchema": {"type": "object"}} for t in tools]
        self.refresh_count = 0
        self.rotate = True
        self.reject_bearer: str | None = None
        self.sse_notification_first = False

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if str(request.url) == TOKEN_ENDPOINT:
            form = dict(p.split("=", 1) for p in request.content.decode().split("&"))
            self.refresh_count += 1
            body = {"access_token": f"at{self.refresh_count}", "expires_in": 3600, "token_type": "Bearer"}
            if self.rotate:
                body["refresh_token"] = f"rt{self.refresh_count}"
            assert form["grant_type"] == "refresh_token"
            return httpx.Response(200, json=body)
        if self.reject_bearer and request.headers.get("authorization") == f"Bearer {self.reject_bearer}":
            return httpx.Response(401)
        payload = json.loads(request.content)
        if "id" not in payload:
            return httpx.Response(202)
        method = payload["method"]
        if method == "initialize":
            result = {"protocolVersion": "2025-06-18", "serverInfo": {"name": "fake"}}
        elif method == "tools/list":
            result = {"tools": self.tools}
        else:
            result = {"content": [{"type": "text", "text": json.dumps({"ok": method})}]}
        reply = {"jsonrpc": "2.0", "id": payload["id"], "result": result}
        if self.sse_notification_first and method == "tools/call":
            note = {"jsonrpc": "2.0", "method": "notifications/message", "params": {"level": "info", "data": "KB query finished"}}
            body = f"event: message\ndata: {json.dumps(note)}\n\nevent: message\ndata: {json.dumps(reply)}\n\n"
            return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=reply)

    def mcp_methods(self) -> list[str]:
        return [json.loads(r.content)["method"] for r in self.requests if str(r.url) != TOKEN_ENDPOINT]


@pytest.fixture
def server(monkeypatch):
    fake = _Server()
    real_client, real_async = httpx.Client, httpx.AsyncClient
    monkeypatch.setattr(
        mcp_proxy.httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(fake), **kw)
    )
    monkeypatch.setattr(
        mcp_proxy.httpx,
        "AsyncClient",
        lambda **kw: real_async(transport=httpx.MockTransport(fake), **kw),
    )
    return fake


def _seed(refresh_token="seed"):
    return {"token_endpoint": TOKEN_ENDPOINT, "client_id": "cid", "refresh_token": refresh_token}


class TestServerConfig:
    def test_a_bare_url_gets_the_defaults(self):
        cfg = ServerConfig.parse("https://mcp.platform.opentargets.org", default_timeout=60.0)
        assert cfg == ServerConfig(url="https://mcp.platform.opentargets.org", timeout=60.0)

    def test_the_legacy_pipe_token_form_still_parses_even_with_an_equals_sign_in_it(self):
        cfg = ServerConfig.parse("https://x.invalid|abc==")
        assert cfg.auth_token == "abc=="
        assert cfg.oauth_env is None

    def test_every_option(self):
        cfg = ServerConfig.parse(
            "https://mcp.c3po.bio|path=/|timeout=120|oauth=C3PO_MCP_OAUTH|prefix=c3po|tools=query_kb+get_kb_query_result"
        )
        assert cfg.endpoint_path == "/"
        assert cfg.timeout == 120.0
        assert cfg.oauth_env == "C3PO_MCP_OAUTH"
        assert cfg.prefix == "c3po"
        assert cfg.tools == frozenset({"query_kb", "get_kb_query_result"})
        assert cfg.admits("query_kb") and not cfg.admits("run_pipeline")

    def test_no_allow_list_admits_everything(self):
        assert ServerConfig.parse("https://x.invalid").admits("anything")


class TestSseParsing:
    async def test_a_notification_ahead_of_the_response_on_the_stream_is_not_mistaken_for_it(self, server):
        server.sse_notification_first = True
        client = MCPProxyClient("https://x.invalid")
        assert await client.call_tool("query_kb", {"query": "q"}) == {"ok": "tools/call"}

    def test_the_response_is_picked_by_request_id_and_a_bare_notification_yields_none(self):
        client = MCPProxyClient("https://x.invalid")
        stream = (
            'data: {"jsonrpc":"2.0","method":"notifications/message","params":{}}\n\n'
            'data: {"jsonrpc":"2.0","id":7,"result":{"a":1}}\n\n'
            'data: {"jsonrpc":"2.0","id":8,"result":{"a":2}}\n\n'
        )
        assert client._parse_sse_response(stream, 7) == {"jsonrpc": "2.0", "id": 7, "result": {"a": 1}}
        assert client._parse_sse_response(stream, 9)["id"] == 8
        assert client._parse_sse_response('data: {"jsonrpc":"2.0","method":"notifications/message"}\n', 1) is None


class TestEndpointPath:
    def test_the_default_appends_mcp(self, server):
        client = MCPProxyClient("https://x.invalid/")
        assert client.list_tools_sync()
        assert str(server.requests[0].url) == "https://x.invalid/mcp"

    def test_a_root_path_posts_to_the_bare_url(self, server):
        client = MCPProxyClient("https://mcp.c3po.bio", endpoint_path="/")
        assert client.list_tools_sync()
        assert str(server.requests[0].url) == "https://mcp.c3po.bio"

    def test_initialized_is_notified_before_the_first_request(self, server):
        MCPProxyClient("https://x.invalid").list_tools_sync()
        assert server.mcp_methods() == ["initialize", "notifications/initialized", "tools/list"]


class TestOAuthRefreshTokenSource:
    def test_a_missing_field_is_refused_up_front(self):
        with pytest.raises(ValueError, match="refresh_token"):
            OAuthRefreshTokenSource({"token_endpoint": TOKEN_ENDPOINT, "client_id": "c"})

    def test_the_first_call_refreshes_and_the_bearer_reaches_the_server(self, server):
        source = OAuthRefreshTokenSource(_seed())
        client = MCPProxyClient("https://x.invalid", token_source=source)
        assert client.list_tools_sync()
        assert server.refresh_count == 1
        mcp_requests = [r for r in server.requests if str(r.url) != TOKEN_ENDPOINT]
        assert {r.headers["authorization"] for r in mcp_requests} == {"Bearer at1"}

    def test_a_valid_token_is_reused_and_an_expiring_one_is_not(self, server):
        source = OAuthRefreshTokenSource(_seed())
        assert source.access_token() == source.access_token() == "at1"
        source._expires_at = time.monotonic() + 30  # inside the refresh margin
        assert source.access_token() == "at2"

    def test_a_rotated_refresh_token_is_persisted_and_read_back_ahead_of_the_seed(self, server, tmp_path):
        path = tmp_path / "C3PO.json"
        OAuthRefreshTokenSource(_seed(), state_path=path).access_token()
        assert json.loads(path.read_text())["refresh_token"] == "rt1"
        assert path.stat().st_mode & 0o777 == 0o600

        again = OAuthRefreshTokenSource(_seed(), state_path=path)
        again.access_token()
        sent = server.requests[-1].content.decode()
        assert "refresh_token=rt1" in sent

    def test_a_fresh_seed_wins_over_the_stored_token(self, server, tmp_path):
        path = tmp_path / "C3PO.json"
        OAuthRefreshTokenSource(_seed("old"), state_path=path).access_token()
        OAuthRefreshTokenSource(_seed("new-login"), state_path=path).access_token()
        assert "refresh_token=new-login" in server.requests[-1].content.decode()

    def test_an_unrotated_token_is_kept(self, server, tmp_path):
        server.rotate = False
        path = tmp_path / "C3PO.json"
        source = OAuthRefreshTokenSource(_seed(), state_path=path)
        source.access_token()
        assert not path.exists()
        source.invalidate()
        source.access_token()
        assert "refresh_token=seed" in server.requests[-1].content.decode()

    def test_a_refusal_names_the_oauth_error(self, monkeypatch):
        def handler(request):
            return httpx.Response(400, json={"error": "invalid_grant"})

        real_client = httpx.Client
        monkeypatch.setattr(
            mcp_proxy.httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw)
        )
        with pytest.raises(RuntimeError, match="invalid_grant"):
            OAuthRefreshTokenSource(_seed()).access_token()

    async def test_a_401_on_a_call_forces_a_refresh_on_the_next_one(self, server):
        source = OAuthRefreshTokenSource(_seed())
        client = MCPProxyClient("https://x.invalid", token_source=source)
        assert client.list_tools_sync()
        server.reject_bearer = "at1"
        result = await client.call_tool("query_kb", {})
        assert result["success"] is False and "401" in result["error"]
        server.reject_bearer = None
        result = await client.call_tool("query_kb", {})
        assert result == {"ok": "tools/call"}
        assert server.refresh_count == 2


class TestBuildProxyClient:
    def test_an_oauth_entry_reads_its_env_var_and_the_state_dir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("C3PO_MCP_OAUTH", json.dumps(_seed()))
        monkeypatch.setenv("EXTERNAL_MCP_STATE_DIR", str(tmp_path))
        client, config = build_proxy_client("https://mcp.c3po.bio|path=/|oauth=C3PO_MCP_OAUTH", 60.0)
        assert client.url == "https://mcp.c3po.bio"
        assert client.token_source.state_path == tmp_path / "C3PO_MCP_OAUTH.json"
        assert config.oauth_env == "C3PO_MCP_OAUTH"

    def test_an_unset_oauth_env_var_is_an_error_not_a_silent_unauthenticated_client(self, monkeypatch):
        monkeypatch.delenv("NOPE_OAUTH", raising=False)
        with pytest.raises(ValueError, match="NOPE_OAUTH"):
            build_proxy_client("https://x.invalid|oauth=NOPE_OAUTH", 60.0)


class TestRegistration:
    def test_the_allow_list_and_the_exclude_list_both_subtract(self, server, monkeypatch):
        monkeypatch.setenv(
            "EXTERNAL_MCP_SERVERS",
            "https://a.invalid|tools=query_kb+run_pipeline,https://b.invalid",
        )
        monkeypatch.setenv("EXTERNAL_MCP_EXCLUDE_TOOLS", "run_pipeline")
        monkeypatch.delenv("RAG_MCP_SERVER", raising=False)
        monkeypatch.setattr(mcp_proxy, "_proxy_clients", {})
        monkeypatch.setattr(mcp_proxy, "_rag_proxy_clients", {})
        assert mcp_proxy.initialize_external_servers() == 2
        assert set(mcp_proxy._proxy_clients) == {"query_kb"}

    async def test_a_prefix_namespaces_the_registered_names_and_dispatch_strips_it(self, server, monkeypatch):
        monkeypatch.setenv("EXTERNAL_MCP_SERVERS", "https://a.invalid|prefix=c3po|tools=query_kb")
        monkeypatch.delenv("EXTERNAL_MCP_EXCLUDE_TOOLS", raising=False)
        monkeypatch.delenv("RAG_MCP_SERVER", raising=False)
        monkeypatch.setattr(mcp_proxy, "_proxy_clients", {})
        monkeypatch.setattr(mcp_proxy, "_rag_proxy_clients", {})
        assert mcp_proxy.initialize_external_servers() == 1
        assert [t["name"] for t in mcp_proxy.get_external_anthropic_tools()] == ["c3po_query_kb"]
        assert await mcp_proxy.execute_external_tool("c3po_query_kb", {}) == {"ok": "tools/call"}
        assert json.loads(server.requests[-1].content)["params"]["name"] == "query_kb"

    def test_a_server_whose_oauth_env_is_unset_is_skipped_and_the_rest_still_register(self, server, monkeypatch):
        monkeypatch.setenv("EXTERNAL_MCP_SERVERS", "https://a.invalid|oauth=UNSET_OAUTH,https://b.invalid")
        monkeypatch.delenv("UNSET_OAUTH", raising=False)
        monkeypatch.delenv("EXTERNAL_MCP_EXCLUDE_TOOLS", raising=False)
        monkeypatch.delenv("RAG_MCP_SERVER", raising=False)
        monkeypatch.setattr(mcp_proxy, "_proxy_clients", {})
        monkeypatch.setattr(mcp_proxy, "_rag_proxy_clients", {})
        assert mcp_proxy.initialize_external_servers() == 2
        assert {c.base_url for c in mcp_proxy._proxy_clients.values()} == {"https://b.invalid"}
