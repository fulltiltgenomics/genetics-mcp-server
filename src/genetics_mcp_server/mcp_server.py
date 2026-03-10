"""
Standalone MCP server for genetics data tools.

This server can be used with Claude Desktop, Cursor, or any MCP-compatible client.
It provides tools for querying genetics data from a REST API.

Environment Variables:
    GENETICS_API_URL: URL of the genetics API (default: http://0.0.0.0:2000/api)
    MCP_DISABLE_TRANSPORT_SECURITY: Disable transport security for remote access
    EXTERNAL_MCP_SERVERS: Comma-separated list of external MCP server URLs to proxy
    EXTERNAL_MCP_EXCLUDE_TOOLS: Comma-separated list of tool names to exclude from proxying
"""

import argparse
import logging
import os
import sys

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import TransportSecuritySettings

from genetics_mcp_server.tools.definitions import register_mcp_tools
from genetics_mcp_server.tools.executor import ToolExecutor

# load environment variables
load_dotenv()

from genetics_mcp_server.logging_config import setup_logging

# for stdio transport, MCP uses stdout for communication so we must not log there;
# for SSE/HTTP transport, stdout logging is fine and used for GCP Cloud Logging
_transport_arg = None
for i, arg in enumerate(sys.argv):
    if arg == "--transport" and i + 1 < len(sys.argv):
        _transport_arg = sys.argv[i + 1]

if _transport_arg in ("sse", "streamable-http"):
    setup_logging(os.environ.get("LOG_LEVEL", "INFO"))
else:
    # stdio: keep logging to stderr
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )

logger = logging.getLogger(__name__)

# check if we should disable transport security (for remote access)
disable_security = os.environ.get("MCP_DISABLE_TRANSPORT_SECURITY", "").lower() in (
    "1",
    "true",
    "yes",
)

# create MCP server instance
if disable_security:
    logger.warning("Transport security DISABLED - allowing all hosts and origins")
    mcp = FastMCP(
        "Genetics Data Tools",
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=False,
            allowed_hosts=["*"],
            allowed_origins=["*"],
        ),
    )
else:
    mcp = FastMCP("Genetics Data Tools")

# create executor and register tools
api_url = os.environ.get("GENETICS_API_URL", "http://0.0.0.0:2000/api")
executor = ToolExecutor(api_base_url=api_url)
register_mcp_tools(mcp, executor)

logger.info(f"Registered MCP tools. API URL: {api_url}")

# register tools from external MCP servers
external_servers = os.environ.get("EXTERNAL_MCP_SERVERS", "")
exclude_tools_str = os.environ.get("EXTERNAL_MCP_EXCLUDE_TOOLS", "")
exclude_tools = set(t.strip() for t in exclude_tools_str.split(",") if t.strip())

if external_servers:
    from genetics_mcp_server.mcp_proxy import MCPProxyClient, register_proxy_tools

    for server_url in external_servers.split(","):
        server_url = server_url.strip()
        if not server_url:
            continue

        logger.info(f"Connecting to external MCP server: {server_url}")
        try:
            proxy_client = MCPProxyClient(base_url=server_url, timeout=60.0)
            register_proxy_tools(mcp, proxy_client, exclude_tools=exclude_tools)
        except Exception as e:
            logger.error(f"Failed to connect to external MCP server {server_url}: {e}")


def _validate_user_token(token: str) -> bool:
    """Validate a user API token via chat backend or local DB."""
    # try local DB first (works when both services share the same filesystem)
    try:
        from genetics_mcp_server.db import get_llm_config_db
        db = get_llm_config_db()
        if db.validate_api_token(token) is not None:
            return True
    except Exception:
        pass

    # fall back to chat backend HTTP call (works across pods in k8s)
    chat_url = os.environ.get("CHAT_BACKEND_URL", "")
    if not chat_url:
        return False
    try:
        import httpx
        internal_secret = os.environ.get("INTERNAL_API_SECRET", "")
        headers = {"Authorization": f"Bearer {internal_secret}"} if internal_secret else {}
        resp = httpx.post(
            f"{chat_url}/v1/tokens/validate",
            json={"token": token},
            headers=headers,
            timeout=5.0,
        )
        if resp.status_code == 200 and resp.json().get("valid"):
            return True
    except Exception:
        logger.exception("Error validating user API token via chat backend")
    return False


def _wrap_with_bearer_auth(app, api_keys: list[str]):
    """Wrap an ASGI app with bearer token authentication middleware."""
    import hmac

    async def auth_middleware(scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            headers = dict(scope.get("headers", []))
            auth_header = headers.get(b"authorization", b"").decode()

            if not auth_header.startswith("Bearer "):
                if scope["type"] == "http":
                    await _send_401(send)
                    return
            else:
                token = auth_header[7:]
                # check static keys first
                if not any(hmac.compare_digest(token, key) for key in api_keys):
                    # fall back to per-user tokens
                    if not _validate_user_token(token):
                        if scope["type"] == "http":
                            await _send_401(send)
                            return

        await app(scope, receive, send)

    async def _send_401(send):
        await send({
            "type": "http.response.start",
            "status": 401,
            "headers": [(b"content-type", b"application/json")],
        })
        await send({
            "type": "http.response.body",
            "body": b'{"error":"Invalid or missing bearer token"}',
        })

    return auth_middleware


def main():
    """Main entry point for the MCP server."""
    parser = argparse.ArgumentParser(
        description="Genetics MCP Server - Provides genetics data tools for AI assistants"
    )
    parser.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio", "sse", "streamable-http"],
        help="Transport protocol to use (default: stdio)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for SSE/HTTP transport (default: 8080)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host for SSE/HTTP transport (default: 0.0.0.0)",
    )
    args = parser.parse_args()

    logger.info(f"Starting MCP server with transport: {args.transport}")

    if args.transport == "stdio":
        mcp.run()
    elif args.transport in ("sse", "streamable-http"):
        import uvicorn

        # get the ASGI app from FastMCP
        if args.transport == "sse":
            app = mcp.sse_app()
        else:
            app = mcp.streamable_http_app()

        # require MCP_API_KEY for remote transports (comma-separated for multiple keys)
        api_key_env = os.environ.get("MCP_API_KEY", "").strip()
        if not api_key_env:
            logger.error("MCP_API_KEY is required for remote transports — refusing to start without authentication")
            sys.exit(1)
        api_keys = [k.strip() for k in api_key_env.split(",") if k.strip()]
        app = _wrap_with_bearer_auth(app, api_keys)
        logger.info(f"Bearer token authentication enabled ({len(api_keys)} key(s))")

        logger.info(f"Starting {args.transport} server on {args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
