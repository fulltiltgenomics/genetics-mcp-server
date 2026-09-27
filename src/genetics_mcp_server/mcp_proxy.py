"""MCP proxy for integrating remote MCP servers.

This module provides functionality to connect to remote MCP servers and proxy their
tools through the local server. It fetches tool definitions from the remote server
and creates wrapper functions that forward calls.
"""

import asyncio
import hashlib
import json
import keyword
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

# ensure env is loaded before configuring logging
load_dotenv()

# configure logging: set our package to LOG_LEVEL, keep root at INFO to avoid
# noise from matplotlib, httpcore, etc.
_log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )
# set our package logger to the requested level
logging.getLogger("genetics_mcp_server").setLevel(getattr(logging, _log_level, logging.INFO))

logger = logging.getLogger(__name__)

# global registry of proxy clients for use by LLM service
_proxy_clients: dict[str, "MCPProxyClient"] = {}
# separate registry for RAG server tools
_rag_proxy_clients: dict[str, "MCPProxyClient"] = {}


# refresh this many seconds before the access token's stated expiry, so a call that starts
# just under the wire does not arrive with a token the server already considers dead
_TOKEN_REFRESH_MARGIN = 60.0


class OAuthRefreshTokenSource:
    """Bearer tokens for a server behind OAuth 2.1, minted from a stored refresh token.

    Servers such as C3PO accept no client-credentials grant, so a service process cannot log
    itself in: an operator runs the device-code login once (`scripts/mcp_oauth_login.py`) and
    the refresh token it yields is what this process holds. Access tokens are short-lived and
    are re-minted here on demand.

    Refresh tokens rotate at some issuers — using one retires it and the response carries its
    successor — so the newest one is written to `state_path` after every refresh and read back
    ahead of the seed on the next start. The seed still wins when it is a different token
    from the one the file grew out of: that is a fresh login, and the file is the stale side.
    Without a state path a rotated token dies with the process, and the next start refreshes
    from a retired seed and fails; `from_env` warns about that at startup rather than at the
    first 401.
    """

    def __init__(self, seed: dict[str, Any], state_path: Path | None = None, name: str = ""):
        for key in ("token_endpoint", "client_id", "refresh_token"):
            if not seed.get(key):
                raise ValueError(f"OAuth credentials for {name or 'external MCP server'} lack {key!r}")
        self.name = name
        self.token_endpoint: str = seed["token_endpoint"]
        self.client_id: str = seed["client_id"]
        self.client_secret: str | None = seed.get("client_secret") or None
        self.state_path = state_path
        self._seed_fingerprint = self._fingerprint(seed["refresh_token"])
        self._refresh_token: str = seed["refresh_token"]
        self._access_token: str | None = None
        self._expires_at = 0.0
        self._lock = threading.Lock()
        self._load_state()

    @staticmethod
    def _fingerprint(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()[:16]

    def _load_state(self) -> None:
        if not self.state_path or not self.state_path.exists():
            return
        try:
            state = json.loads(self.state_path.read_text())
        except (OSError, ValueError) as e:
            logger.warning(f"Ignoring unreadable OAuth state {self.state_path}: {e}")
            return
        if state.get("seed_fingerprint") != self._seed_fingerprint:
            logger.info(f"OAuth seed for {self.name} changed; ignoring the stored refresh token")
            return
        if state.get("refresh_token"):
            self._refresh_token = state["refresh_token"]

    def _save_state(self) -> None:
        if not self.state_path:
            return
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(
                    {"seed_fingerprint": self._seed_fingerprint, "refresh_token": self._refresh_token}
                )
            )
            tmp.chmod(0o600)
            tmp.replace(self.state_path)
        except OSError as e:
            logger.error(f"Could not persist rotated OAuth refresh token for {self.name}: {e}")

    def _refresh(self) -> None:
        data = {
            "grant_type": "refresh_token",
            "refresh_token": self._refresh_token,
            "client_id": self.client_id,
        }
        if self.client_secret:
            data["client_secret"] = self.client_secret
        with httpx.Client(timeout=30.0) as client:
            response = client.post(self.token_endpoint, data=data)
        if response.status_code != 200:
            # the body names the OAuth error (invalid_grant = the refresh token is dead and an
            # operator has to log in again); it never carries a token, so it is safe to log
            raise RuntimeError(
                f"OAuth token refresh for {self.name} failed: HTTP {response.status_code} "
                f"{response.text[:300]}"
            )
        body = response.json()
        self._access_token = body["access_token"]
        self._expires_at = time.monotonic() + float(body.get("expires_in", 300))
        rotated = body.get("refresh_token")
        if rotated and rotated != self._refresh_token:
            self._refresh_token = rotated
            self._save_state()
        logger.info(f"Refreshed OAuth access token for {self.name}")

    def access_token(self) -> str:
        """A currently valid access token, refreshing when the held one is near expiry."""
        with self._lock:
            if not self._access_token or time.monotonic() >= self._expires_at - _TOKEN_REFRESH_MARGIN:
                self._refresh()
            return self._access_token  # type: ignore[return-value]

    def invalidate(self) -> None:
        """Forget the access token, so the next call refreshes; used after a 401."""
        with self._lock:
            self._access_token = None
            self._expires_at = 0.0

    @classmethod
    def from_env(cls, env_var: str, state_dir: str | None) -> "OAuthRefreshTokenSource":
        raw = os.environ.get(env_var, "")
        if not raw:
            raise ValueError(f"{env_var} is not set; run the device-code login and store its output there")
        seed = json.loads(raw)
        state_path = Path(state_dir) / f"{env_var}.json" if state_dir else None
        if state_path is None:
            logger.warning(
                f"EXTERNAL_MCP_STATE_DIR is unset: a refresh token rotated for {env_var} "
                "will not survive a restart"
            )
        return cls(seed, state_path=state_path, name=env_var)


@dataclass(frozen=True)
class ServerConfig:
    """One EXTERNAL_MCP_SERVERS entry, parsed.

    Entry syntax is `URL[|option]...`, an option being `key=value` for a known key, else a
    bare static bearer token (the original `URL|TOKEN` form; a token may itself contain
    `=`). Keys: `path` (the JSON-RPC endpoint under the URL, default `/mcp`; `/` for a
    server that answers at its root), `timeout` (seconds), `oauth` (the env var holding
    the operator login's JSON), `token` (an explicit static bearer token), `tools` (a
    `+`-separated allow-list of upstream names; `,` already separates entries), `prefix`
    (registers each tool as `<prefix>_<name>`, keeping a server's names out of the flat
    space the local tools share — C3PO's `list_datasets` would otherwise shadow ours).
    """

    url: str
    endpoint_path: str = "/mcp"
    timeout: float = 60.0
    auth_token: str | None = None
    oauth_env: str | None = None
    tools: frozenset[str] | None = None
    prefix: str = ""

    KNOWN_OPTIONS = ("path", "timeout", "oauth", "token", "tools", "prefix")

    @classmethod
    def parse(cls, entry: str, default_timeout: float = 60.0) -> "ServerConfig":
        parts = [p.strip() for p in entry.split("|")]
        fields: dict[str, Any] = {"url": parts[0], "timeout": default_timeout}
        for opt in parts[1:]:
            if not opt:
                continue
            key, sep, value = opt.partition("=")
            if sep and key in cls.KNOWN_OPTIONS:
                if key == "path":
                    fields["endpoint_path"] = value
                elif key == "timeout":
                    fields["timeout"] = float(value)
                elif key == "oauth":
                    fields["oauth_env"] = value
                elif key == "token":
                    fields["auth_token"] = value
                elif key == "tools":
                    fields["tools"] = frozenset(t for t in value.split("+") if t)
                elif key == "prefix":
                    fields["prefix"] = value
            else:
                fields["auth_token"] = opt
        return cls(**fields)

    def admits(self, tool_name: str) -> bool:
        return self.tools is None or tool_name in self.tools


class MCPProxyClient:
    """Client for proxying tools from a remote MCP server."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        prefix: str = "",
        auth_token: str | None = None,
        endpoint_path: str = "/mcp",
        token_source: OAuthRefreshTokenSource | None = None,
    ):
        """
        Initialize the proxy client.

        Args:
            base_url: Base URL of the remote MCP server
            timeout: Request timeout in seconds
            prefix: Optional prefix to add to tool names to avoid conflicts
            auth_token: Optional static Bearer token for the Authorization header
            endpoint_path: Path of the JSON-RPC endpoint under base_url; "/" for a server
                that answers at its root
            token_source: Bearer tokens minted per call, for a server behind OAuth; wins
                over auth_token
        """
        self.base_url = base_url.rstrip("/")
        path = endpoint_path.strip("/")
        self.url = f"{self.base_url}/{path}" if path else self.base_url
        self.timeout = timeout
        self.prefix = prefix
        self.auth_token = auth_token
        self.token_source = token_source
        self.session_id: str | None = None
        self._request_id = 0
        self._tools: list[dict] = []
        self._initialized = False

    def _headers(self) -> dict[str, str]:
        """Request headers; with a token source this may block on a token refresh."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.token_source:
            headers["Authorization"] = f"Bearer {self.token_source.access_token()}"
        elif self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        return headers

    def _has_auth(self) -> bool:
        return bool(self.token_source or self.auth_token)

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _jsonrpc_request(self, method: str, params: dict | None = None) -> dict:
        return {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": method,
            "params": params or {},
        }

    def _parse_sse_response(self, text: str) -> dict | None:
        """Parse SSE response to extract JSON-RPC result."""
        for line in text.strip().split("\n"):
            if line.startswith("data: "):
                data = line[6:]
                try:
                    return json.loads(data)
                except json.JSONDecodeError:
                    pass
        return None

    def _notify_sync(self, method: str) -> None:
        """Send a JSON-RPC notification; the server answers with no body, and a refusal is
        logged rather than raised since the notification carries nothing we need back."""
        payload = {"jsonrpc": "2.0", "method": method}
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(self.url, json=payload, headers=self._headers())
            if response.status_code >= 400:
                logger.debug(f"{method} to {self.url} answered HTTP {response.status_code}")
        except Exception as e:
            logger.debug(f"{method} to {self.url} failed: {e}")

    def _post_sync(self, payload: dict) -> dict:
        """Synchronous POST to the server's JSON-RPC endpoint."""
        url = self.url
        headers = self._headers()

        logger.debug(f"POST {url} method={payload.get('method')} auth={'yes' if self._has_auth() else 'no'}")

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, json=payload, headers=headers)
            logger.debug(f"Response status={response.status_code} content-type={response.headers.get('content-type')}")
            logger.debug(f"Response body (first 500 chars): {response.text[:500]}")
            response.raise_for_status()

            if "mcp-session-id" in response.headers:
                self.session_id = response.headers["mcp-session-id"]

            content_type = response.headers.get("content-type", "")
            if "text/event-stream" in content_type:
                result = self._parse_sse_response(response.text)
                if result:
                    return result
                raise RuntimeError(f"Failed to parse SSE response: {response.text}")

            return response.json()

    async def _post_async(self, payload: dict) -> dict:
        """Async POST to the server's JSON-RPC endpoint."""
        url = self.url
        # a token refresh is a blocking HTTP round trip; keep it off the event loop
        headers = await asyncio.get_running_loop().run_in_executor(None, self._headers)

        logger.debug(f"POST {url} method={payload.get('method')} auth={'yes' if self._has_auth() else 'no'}")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload, headers=headers)
            logger.debug(f"Response status={response.status_code} content-type={response.headers.get('content-type')}")
            logger.debug(f"Response body (first 500 chars): {response.text[:500]}")
            response.raise_for_status()

            if "mcp-session-id" in response.headers:
                self.session_id = response.headers["mcp-session-id"]

            content_type = response.headers.get("content-type", "")
            if "text/event-stream" in content_type:
                result = self._parse_sse_response(response.text)
                if result:
                    return result
                raise RuntimeError(f"Failed to parse SSE response: {response.text}")

            return response.json()

    def initialize_sync(self) -> bool:
        """Initialize connection to the remote MCP server synchronously."""
        logger.debug(f"Initializing MCP connection to {self.base_url}")
        try:
            payload = self._jsonrpc_request(
                "initialize",
                {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "genetics-mcp-proxy", "version": "1.0.0"},
                },
            )
            result = self._post_sync(payload)
            logger.debug(f"Initialize result: {result}")

            if "error" in result:
                logger.error(f"Failed to initialize remote MCP: {result['error']}")
                return False

            self._initialized = True
            # the spec has the client confirm before its first request; servers holding a
            # session refuse tools/list until they see it, stateless ones ignore it
            self._notify_sync("notifications/initialized")
            logger.info(
                f"Connected to remote MCP server: {result.get('result', {}).get('serverInfo', {})}"
            )
            return True

        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error connecting to {self.base_url}: {e.response.status_code} - {e.response.text[:200]}"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to connect to remote MCP server at {self.base_url}: {e}")
            return False

    def list_tools_sync(self) -> list[dict]:
        """List tools from the remote MCP server synchronously."""
        if not self._initialized:
            logger.debug(f"Not initialized, calling initialize_sync for {self.base_url}")
            if not self.initialize_sync():
                logger.warning(f"Failed to initialize {self.base_url}, returning empty tool list")
                return []

        try:
            payload = self._jsonrpc_request("tools/list")
            result = self._post_sync(payload)
            logger.debug(f"tools/list result: {result}")

            if "result" in result and "tools" in result["result"]:
                self._tools = result["result"]["tools"]
                logger.info(f"Found {len(self._tools)} tools from {self.base_url}: {[t.get('name') for t in self._tools]}")
                return self._tools

            logger.warning(f"Unexpected tools/list response format from {self.base_url}: {result}")
            return result.get("result", [])

        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error listing tools from {self.base_url}: {e.response.status_code} - {e.response.text[:200]}"
            )
            return []
        except Exception as e:
            logger.error(f"Failed to list tools from {self.base_url}: {e}")
            return []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool on the remote MCP server."""
        # reinitialize if session expired
        if not self._initialized:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.initialize_sync)

        try:
            payload = self._jsonrpc_request(
                "tools/call",
                {"name": name, "arguments": arguments},
            )
            result = await self._post_async(payload)
            result_str = json.dumps(result)
            logger.info(
                f"External tool {name} response ({len(result_str)} chars): "
                f"{result_str[:500]}{'...[truncated]' if len(result_str) > 500 else ''}"
            )

            if "error" in result:
                error_msg = result["error"]
                if isinstance(error_msg, dict):
                    error_msg = error_msg.get("message", str(error_msg))
                return {"success": False, "error": f"Remote MCP error: {error_msg}"}

            if "result" in result:
                mcp_result = result["result"]
                # extract text content from MCP response
                if isinstance(mcp_result, dict) and "content" in mcp_result:
                    content = mcp_result["content"]
                    if isinstance(content, list):
                        texts = []
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                texts.append(item.get("text", ""))
                        if texts:
                            combined = "\n".join(texts)
                            # try to parse as JSON
                            try:
                                return json.loads(combined)
                            except json.JSONDecodeError:
                                return {"success": True, "result": combined}
                return {"success": True, "result": mcp_result}

            return {"success": True, "result": result}

        except httpx.HTTPStatusError as e:
            # session may have expired, try to reinitialize
            if e.response.status_code in (400, 401, 403):
                self._initialized = False
                self.session_id = None
                if self.token_source:
                    self.token_source.invalidate()
            return {"success": False, "error": f"HTTP {e.response.status_code}: {e.response.text}"}

        except Exception as e:
            logger.error(f"Error calling remote tool {name}: {e}")
            return {"success": False, "error": str(e)}

    def get_prefixed_name(self, original_name: str) -> str:
        """Get the prefixed tool name."""
        if self.prefix:
            return f"{self.prefix}_{original_name}"
        return original_name


def _json_type_to_python(json_type: str, is_required: bool = True) -> str:
    """Convert JSON schema type to Python type annotation string."""
    type_map = {
        "string": "str",
        "integer": "int",
        "number": "float",
        "boolean": "bool",
        "array": "list",
        "object": "dict",
    }
    py_type = type_map.get(json_type, "Any")
    if not is_required:
        return f"{py_type} | None"
    return py_type


def _is_safe_identifier(name: str) -> bool:
    """A name is only safe to interpolate into generated source if it is a bare identifier."""
    return bool(name) and name.isidentifier() and not keyword.iskeyword(name)


def _build_function_signature(input_schema: dict) -> tuple[list[str], list[str], list[Any]]:
    """
    Build function parameter strings from JSON schema.

    Everything interpolated into the generated source is either a validated identifier or a
    fixed type name from _json_type_to_python's table. Default *values* are never rendered as
    source — they are passed through the `_defaults` list and referenced by index — because a
    remote server controls them and `{default}` would otherwise be executed as Python.

    Returns:
        Tuple of (required_params, optional_params, defaults) — defaults is the list the
        generated source indexes into, and must be bound as `_defaults` in the exec namespace.

    Raises:
        ValueError: if the schema declares a property name that is not a bare identifier.
    """
    properties = input_schema.get("properties", {})
    required = set(input_schema.get("required", []))

    required_params = []
    optional_params = []
    defaults: list[Any] = []

    for param_name, param_info in properties.items():
        if not _is_safe_identifier(param_name):
            raise ValueError(f"unsafe parameter name from remote server: {param_name!r}")

        param_type = param_info.get("type", "string")
        py_type = _json_type_to_python(param_type, param_name in required)

        if param_name in required:
            required_params.append(f"{param_name}: {py_type}")
        else:
            defaults.append(param_info.get("default"))
            optional_params.append(f"{param_name}: {py_type} = _defaults[{len(defaults) - 1}]")

    return required_params, optional_params, defaults


def register_proxy_tools(
    mcp,
    proxy_client: MCPProxyClient,
    exclude_tools: set[str] | None = None,
    allow_tools: frozenset[str] | None = None,
):
    """
    Register proxy tools from a remote MCP server with a FastMCP instance.

    Args:
        mcp: FastMCP server instance
        proxy_client: Initialized MCPProxyClient
        exclude_tools: Set of tool names to exclude (without prefix)
        allow_tools: the entry's `tools=` allow-list; None admits everything
    """
    exclude_tools = exclude_tools or set()
    tools = proxy_client.list_tools_sync()

    if not tools:
        logger.warning(f"No tools found on remote server {proxy_client.base_url}")
        return

    registered_count = 0
    for tool in tools:
        original_name = tool.get("name", "")
        if original_name in exclude_tools or (allow_tools is not None and original_name not in allow_tools):
            logger.debug(f"Skipping excluded tool: {original_name}")
            continue

        prefixed_name = proxy_client.get_prefixed_name(original_name)
        description = tool.get("description", f"Proxy tool: {original_name}")
        input_schema = tool.get("inputSchema", {})

        # the tool name, its parameter names, its description and its defaults all come from
        # the remote server. Only validated identifiers may reach the generated source; the
        # description and the upstream tool name are bound into the namespace and attached
        # afterwards, so no remote string can close the docstring or the string literal and
        # have the remainder executed as Python.
        if not _is_safe_identifier(prefixed_name):
            logger.error(f"Skipping proxy tool with unsafe name: {prefixed_name!r}")
            continue

        try:
            required_params, optional_params, defaults = _build_function_signature(input_schema)
        except ValueError as e:
            logger.error(f"Skipping proxy tool {prefixed_name}: {e}")
            continue

        all_params = required_params + optional_params
        params_str = ", ".join(all_params) if all_params else ""

        func_code = f'''
async def {prefixed_name}({params_str}) -> dict:
    kwargs = {{k: v for k, v in locals().items() if v is not None}}
    return await _proxy_client.call_tool(_original_name, kwargs)
'''

        namespace = {
            "_proxy_client": proxy_client,
            "_original_name": original_name,
            "_defaults": defaults,
            "Any": Any,
        }
        try:
            exec(func_code, namespace)
            proxy_func = namespace[prefixed_name]
            proxy_func.__doc__ = description

            # register using decorator
            mcp.tool()(proxy_func)
            registered_count += 1
            logger.debug(f"Registered proxy tool: {prefixed_name}")

        except Exception as e:
            logger.error(f"Failed to register proxy tool {prefixed_name}: {e}")

    # register proxy client globally for LLM service use.
    # this loop does NOT re-apply `exclude_tools`, so a tool skipped above still gets a
    # _proxy_clients entry. Harmless only because nothing in the mcp-server process
    # dispatches through that registry — llm_service, which does, populates it from
    # initialize_external_servers(), where the exclusion is applied. A dispatch path in
    # THIS process reading _proxy_clients would make an excluded tool callable again.
    for tool in tools:
        tool_name = proxy_client.get_prefixed_name(tool.get("name", ""))
        _proxy_clients[tool_name] = proxy_client

    logger.info(
        f"Registered {registered_count} proxy tools from {proxy_client.base_url}"
        f"{f' with prefix {proxy_client.prefix!r}' if proxy_client.prefix else ''}"
    )


def get_external_anthropic_tools() -> list[dict[str, Any]]:
    """
    Get tool definitions from all registered external MCP servers in Anthropic format.

    Returns:
        List of tool definitions in Anthropic's tool format
    """
    # collect unique tools (avoid duplicates if same tool registered multiple times)
    seen_tools: set[str] = set()
    anthropic_tools: list[dict[str, Any]] = []

    for tool_name, proxy_client in _proxy_clients.items():
        if tool_name in seen_tools:
            continue

        # find the tool definition
        for tool in proxy_client._tools:
            prefixed_name = proxy_client.get_prefixed_name(tool.get("name", ""))
            if prefixed_name == tool_name:
                seen_tools.add(tool_name)
                input_schema = tool.get("inputSchema", {})

                anthropic_tools.append({
                    "name": prefixed_name,
                    "description": tool.get("description", f"External tool: {tool_name}"),
                    "input_schema": input_schema,
                })
                break

    return anthropic_tools


def get_rag_anthropic_tools() -> list[dict[str, Any]]:
    """
    Get tool definitions from the RAG MCP server in Anthropic format.

    Returns:
        List of tool definitions in Anthropic's tool format
    """
    seen_tools: set[str] = set()
    anthropic_tools: list[dict[str, Any]] = []

    for tool_name, proxy_client in _rag_proxy_clients.items():
        if tool_name in seen_tools:
            continue

        for tool in proxy_client._tools:
            prefixed_name = proxy_client.get_prefixed_name(tool.get("name", ""))
            if prefixed_name == tool_name:
                seen_tools.add(tool_name)
                anthropic_tools.append({
                    "name": prefixed_name,
                    "description": tool.get("description", f"RAG tool: {tool_name}"),
                    "input_schema": tool.get("inputSchema", {}),
                })
                break

    return anthropic_tools


async def execute_external_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """
    Execute a tool on an external MCP server.

    Args:
        tool_name: Name of the tool (may be prefixed)
        arguments: Tool arguments

    Returns:
        Tool execution result
    """
    proxy_client = _proxy_clients.get(tool_name) or _rag_proxy_clients.get(tool_name)
    if not proxy_client:
        return {"success": False, "error": f"No proxy client found for tool: {tool_name}"}

    # find the original tool name (without prefix)
    original_name = tool_name
    if proxy_client.prefix and tool_name.startswith(f"{proxy_client.prefix}_"):
        original_name = tool_name[len(proxy_client.prefix) + 1:]

    return await proxy_client.call_tool(original_name, arguments)


def build_proxy_client(entry: str, default_timeout: float) -> tuple[MCPProxyClient, ServerConfig]:
    """A client for one EXTERNAL_MCP_SERVERS entry (see ServerConfig for the syntax).

    Raises ValueError when the entry names an `oauth=` env var that is unset or malformed,
    so the caller can skip the server with the reason in the log rather than register it
    and fail on the first call.
    """
    config = ServerConfig.parse(entry, default_timeout)
    token_source = None
    if config.oauth_env:
        token_source = OAuthRefreshTokenSource.from_env(
            config.oauth_env, os.environ.get("EXTERNAL_MCP_STATE_DIR") or None
        )
    client = MCPProxyClient(
        base_url=config.url,
        timeout=config.timeout,
        prefix=config.prefix,
        auth_token=config.auth_token,
        endpoint_path=config.endpoint_path,
        token_source=token_source,
    )
    return client, config


def _initialize_rag_server(exclude_tools: set[str] | None = None) -> int:
    """
    Initialize connection to the RAG MCP server from RAG_MCP_SERVER env var.

    Args:
        exclude_tools: Tool names EXTERNAL_MCP_EXCLUDE_TOOLS withholds. The RAG registry
            honours the same list as the external one, so an excluded name enters neither
            registry and no surface can advertise it.

    Returns:
        Number of tools registered from the RAG server
    """
    exclude_tools = exclude_tools or set()
    rag_server = os.environ.get("RAG_MCP_SERVER", "")
    if not rag_server:
        logger.debug("RAG_MCP_SERVER not set, skipping RAG server initialization")
        return 0

    try:
        proxy_client, config = build_proxy_client(rag_server.strip(), default_timeout=180.0)
        server_url = config.url
        logger.info(f"Connecting to RAG MCP server: {server_url}")
        tools = proxy_client.list_tools_sync()

        if not tools:
            logger.warning(f"No tools returned from RAG server {server_url}")
            return 0

        registered = 0
        for tool in tools:
            original_name = tool.get("name", "")
            if original_name in exclude_tools or not config.admits(original_name):
                logger.debug(f"Skipping excluded RAG tool: {original_name}")
                continue
            _rag_proxy_clients[proxy_client.get_prefixed_name(original_name)] = proxy_client
            registered += 1

        logger.info(
            f"Registered {registered} RAG tools from {server_url} "
            f"(excluded {len(tools) - registered})"
        )
        return registered

    except Exception as e:
        logger.error(f"Failed to connect to RAG MCP server {rag_server}: {e}", exc_info=True)
        return 0


def initialize_external_servers() -> int:
    """
    Initialize connections to external MCP servers from environment config.

    Reads EXTERNAL_MCP_SERVERS env var (always-on servers like gnomAD, Open Targets)
    and RAG_MCP_SERVER env var (RAG server, registered into its own registry).

    Returns:
        Number of tools registered from all external servers
    """
    total_tools = 0

    exclude_tools_str = os.environ.get("EXTERNAL_MCP_EXCLUDE_TOOLS", "")
    exclude_tools = set(t.strip() for t in exclude_tools_str.split(",") if t.strip())
    if exclude_tools:
        logger.info(f"Excluding tools from external servers: {exclude_tools}")

    external_servers = os.environ.get("EXTERNAL_MCP_SERVERS", "")
    if external_servers:
        logger.info(f"Initializing external MCP servers from config (found {len(external_servers.split(','))} entries)")

        for server_entry in external_servers.split(","):
            server_entry = server_entry.strip()
            if not server_entry:
                continue

            server_url = server_entry.split("|", 1)[0].strip()
            try:
                proxy_client, config = build_proxy_client(server_entry, default_timeout=60.0)
                logger.info(
                    f"Connecting to external MCP server: {server_url} "
                    f"(auth={'oauth' if config.oauth_env else 'token' if config.auth_token else 'none'}, "
                    f"endpoint={proxy_client.url}, timeout={config.timeout}s)"
                )
                tools = proxy_client.list_tools_sync()

                if not tools:
                    logger.warning(f"No tools returned from {server_url} - server may not be MCP-compatible")
                    continue

                registered = 0
                for tool in tools:
                    original_name = tool.get("name", "")
                    if original_name in exclude_tools or not config.admits(original_name):
                        logger.debug(f"Skipping excluded tool: {original_name}")
                        continue
                    tool_name = proxy_client.get_prefixed_name(original_name)
                    _proxy_clients[tool_name] = proxy_client
                    total_tools += 1
                    registered += 1

                logger.info(f"Registered {registered} tools from {server_url} (excluded {len(tools) - registered})")

            except Exception as e:
                logger.error(f"Failed to connect to external MCP server {server_url}: {e}", exc_info=True)
    else:
        logger.debug("EXTERNAL_MCP_SERVERS not set, skipping external server initialization")

    # initialize RAG server separately
    total_tools += _initialize_rag_server(exclude_tools)

    logger.info(
        f"External MCP initialization complete: {total_tools} total tools "
        f"({len(_proxy_clients)} always-on, {len(_rag_proxy_clients)} RAG)"
    )
    return total_tools


def is_external_tool(tool_name: str) -> bool:
    """Check if a tool is from an external MCP server (including RAG)."""
    return tool_name in _proxy_clients or tool_name in _rag_proxy_clients
