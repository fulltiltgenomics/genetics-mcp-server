"""MCP proxy for integrating remote MCP servers.

This module provides functionality to connect to remote MCP servers and proxy their
tools through the local server. It fetches tool definitions from the remote server
and creates wrapper functions that forward calls.
"""

import asyncio
import json
import logging
import os
import sys
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


class MCPProxyClient:
    """Client for proxying tools from a remote MCP server."""

    def __init__(
        self, base_url: str, timeout: float = 30.0, prefix: str = "", auth_token: str | None = None
    ):
        """
        Initialize the proxy client.

        Args:
            base_url: Base URL of the remote MCP server (without /mcp suffix)
            timeout: Request timeout in seconds
            prefix: Optional prefix to add to tool names to avoid conflicts
            auth_token: Optional Bearer token for Authorization header
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.prefix = prefix
        self.auth_token = auth_token
        self.session_id: str | None = None
        self._request_id = 0
        self._tools: list[dict] = []
        self._initialized = False

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

    def _post_sync(self, payload: dict) -> dict:
        """Synchronous POST to /mcp endpoint."""
        url = f"{self.base_url}/mcp"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id

        logger.debug(f"POST {url} method={payload.get('method')} auth={'yes' if self.auth_token else 'no'}")

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
        """Async POST to /mcp endpoint."""
        url = f"{self.base_url}/mcp"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id

        logger.debug(f"POST {url} method={payload.get('method')} auth={'yes' if self.auth_token else 'no'}")

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
                    "protocolVersion": "2024-11-05",
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


def _build_function_signature(input_schema: dict) -> tuple[list[str], list[str]]:
    """
    Build function parameter strings from JSON schema.

    Returns:
        Tuple of (required_params, optional_params) as strings
    """
    properties = input_schema.get("properties", {})
    required = set(input_schema.get("required", []))

    required_params = []
    optional_params = []

    for param_name, param_info in properties.items():
        param_type = param_info.get("type", "string")
        py_type = _json_type_to_python(param_type, param_name in required)

        if param_name in required:
            required_params.append(f"{param_name}: {py_type}")
        else:
            default = param_info.get("default")
            if default is None:
                optional_params.append(f"{param_name}: {py_type} = None")
            elif isinstance(default, str):
                optional_params.append(f"{param_name}: {py_type} = {default!r}")
            else:
                optional_params.append(f"{param_name}: {py_type} = {default}")

    return required_params, optional_params


def register_proxy_tools(mcp, proxy_client: MCPProxyClient, exclude_tools: set[str] | None = None):
    """
    Register proxy tools from a remote MCP server with a FastMCP instance.

    Args:
        mcp: FastMCP server instance
        proxy_client: Initialized MCPProxyClient
        exclude_tools: Set of tool names to exclude (without prefix)
    """
    exclude_tools = exclude_tools or set()
    tools = proxy_client.list_tools_sync()

    if not tools:
        logger.warning(f"No tools found on remote server {proxy_client.base_url}")
        return

    registered_count = 0
    for tool in tools:
        original_name = tool.get("name", "")
        if original_name in exclude_tools:
            logger.debug(f"Skipping excluded tool: {original_name}")
            continue

        prefixed_name = proxy_client.get_prefixed_name(original_name)
        description = tool.get("description", f"Proxy tool: {original_name}")
        input_schema = tool.get("inputSchema", {})

        # build function with proper type hints using exec
        required_params, optional_params = _build_function_signature(input_schema)
        all_params = required_params + optional_params
        params_str = ", ".join(all_params) if all_params else ""

        # create the function code
        func_code = f'''
async def {prefixed_name}({params_str}) -> dict:
    """{description}"""
    kwargs = {{k: v for k, v in locals().items() if v is not None}}
    return await _proxy_client.call_tool("{original_name}", kwargs)
'''

        # execute in a namespace with the proxy_client
        namespace = {"_proxy_client": proxy_client, "Any": Any}
        try:
            exec(func_code, namespace)
            proxy_func = namespace[prefixed_name]

            # register using decorator
            mcp.tool()(proxy_func)
            registered_count += 1
            logger.debug(f"Registered proxy tool: {prefixed_name}")

        except Exception as e:
            logger.error(f"Failed to register proxy tool {prefixed_name}: {e}")

    # register proxy client globally for LLM service use
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


def get_proxy_client_for_tool(tool_name: str) -> MCPProxyClient | None:
    """
    Get the proxy client that can execute the given tool.

    Args:
        tool_name: Name of the tool

    Returns:
        MCPProxyClient if tool is external, None if local
    """
    return _proxy_clients.get(tool_name) or _rag_proxy_clients.get(tool_name)


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


def _parse_server_config(server_entry: str) -> tuple[str, str | None]:
    """
    Parse server entry to extract URL and optional auth token.

    Format: URL or URL|AUTH_TOKEN

    Returns:
        Tuple of (url, auth_token)
    """
    if "|" in server_entry:
        parts = server_entry.split("|", 1)
        return parts[0].strip(), parts[1].strip()
    return server_entry, None


def _initialize_rag_server() -> int:
    """
    Initialize connection to the RAG MCP server from RAG_MCP_SERVER env var.

    Returns:
        Number of tools registered from the RAG server
    """
    rag_server = os.environ.get("RAG_MCP_SERVER", "")
    if not rag_server:
        logger.debug("RAG_MCP_SERVER not set, skipping RAG server initialization")
        return 0

    server_url, auth_token = _parse_server_config(rag_server.strip())
    logger.info(f"Connecting to RAG MCP server: {server_url}")
    try:
        proxy_client = MCPProxyClient(base_url=server_url, timeout=180.0, auth_token=auth_token)
        tools = proxy_client.list_tools_sync()

        if not tools:
            logger.warning(f"No tools returned from RAG server {server_url}")
            return 0

        for tool in tools:
            tool_name = proxy_client.get_prefixed_name(tool.get("name", ""))
            _rag_proxy_clients[tool_name] = proxy_client

        logger.info(f"Registered {len(tools)} RAG tools from {server_url}")
        return len(tools)

    except Exception as e:
        logger.error(f"Failed to connect to RAG MCP server {server_url}: {e}", exc_info=True)
        return 0


def initialize_external_servers() -> int:
    """
    Initialize connections to external MCP servers from environment config.

    Reads EXTERNAL_MCP_SERVERS env var (always-on servers like gnomAD, Open Targets)
    and RAG_MCP_SERVER env var (RAG server, only included in 'rag' tool profile).

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

            server_url, auth_token = _parse_server_config(server_entry)
            logger.info(f"Connecting to external MCP server: {server_url} (auth={'configured' if auth_token else 'none'})")
            try:
                proxy_client = MCPProxyClient(base_url=server_url, timeout=60.0, auth_token=auth_token)
                tools = proxy_client.list_tools_sync()

                if not tools:
                    logger.warning(f"No tools returned from {server_url} - server may not be MCP-compatible")
                    continue

                registered = 0
                for tool in tools:
                    original_name = tool.get("name", "")
                    if original_name in exclude_tools:
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
    total_tools += _initialize_rag_server()

    logger.info(
        f"External MCP initialization complete: {total_tools} total tools "
        f"({len(_proxy_clients)} always-on, {len(_rag_proxy_clients)} RAG)"
    )
    return total_tools


def is_external_tool(tool_name: str) -> bool:
    """Check if a tool is from an external MCP server (including RAG)."""
    return tool_name in _proxy_clients or tool_name in _rag_proxy_clients
