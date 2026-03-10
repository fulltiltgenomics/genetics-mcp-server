"""
Simple MCP client for calling tools on a running MCP server.

Supports both streamable-http and SSE transports.

Usage:
    # List available tools (streamable-http, default)
    python -m genetics_mcp_server.mcp_client --url http://0.0.0.0:8080 list

    # List tools via SSE transport
    python -m genetics_mcp_server.mcp_client --url http://0.0.0.0:8080/sse --transport sse list

    # Call a tool
    python -m genetics_mcp_server.mcp_client --url http://0.0.0.0:8080 call get_credible_sets_by_gene gene=APOE

    # Call with JSON arguments
    python -m genetics_mcp_server.mcp_client --url http://0.0.0.0:8080 call search_phenotypes --json '{"query": "diabetes", "limit": 10}'
"""

import argparse
import asyncio
import json
import sys
from typing import Any

import httpx


class StreamableHttpClient:
    """Client for communicating with MCP servers via streamable-http."""

    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session_id: str | None = None
        self._request_id = 0

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
                data = line[6:]  # strip "data: " prefix
                try:
                    return json.loads(data)
                except json.JSONDecodeError:
                    pass
        return None

    def _post(self, payload: dict) -> dict:
        """Post to /mcp endpoint and parse SSE response."""
        url = f"{self.base_url}/mcp"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()

            # capture session ID from response
            if "mcp-session-id" in response.headers:
                self.session_id = response.headers["mcp-session-id"]

            # response is SSE format
            content_type = response.headers.get("content-type", "")
            if "text/event-stream" in content_type:
                result = self._parse_sse_response(response.text)
                if result:
                    return result
                raise RuntimeError(f"Failed to parse SSE response: {response.text}")

            # plain JSON response
            return response.json()

    def initialize(self) -> dict:
        """Initialize the MCP connection."""
        payload = self._jsonrpc_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "mcp-client", "version": "1.0.0"},
            },
        )
        return self._post(payload)

    def list_tools(self) -> list[dict]:
        """List available tools."""
        payload = self._jsonrpc_request("tools/list")
        result = self._post(payload)

        if "result" in result and "tools" in result["result"]:
            return result["result"]["tools"]
        return result.get("result", [])

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool by name with arguments."""
        payload = self._jsonrpc_request(
            "tools/call",
            {"name": name, "arguments": arguments},
        )
        result = self._post(payload)

        if "result" in result:
            return result["result"]
        if "error" in result:
            raise RuntimeError(f"MCP error: {result['error']}")
        return result


class SSEClient:
    """Async client for communicating with MCP servers via SSE transport."""

    def __init__(self, url: str, timeout: float = 30.0):
        self.url = url
        self.timeout = timeout

    async def run(
        self, command: str, tool_name: str | None = None, tool_args: dict | None = None
    ):
        """Run a command against the SSE server."""
        from mcp import ClientSession
        from mcp.client.sse import sse_client

        try:
            async with sse_client(
                self.url, timeout=self.timeout, sse_read_timeout=self.timeout
            ) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    if command == "list":
                        result = await session.list_tools()
                        tools = result.tools
                        print(f"Available tools ({len(tools)}):\n")
                        for tool in tools:
                            print(f"  {tool.name}")
                            if tool.description:
                                desc = tool.description
                                if len(desc) > 80:
                                    desc = desc[:80] + "..."
                                print(f"    {desc}")
                            print()

                    elif command == "call":
                        result = await session.call_tool(tool_name, tool_args or {})
                        # print result content
                        for item in result.content:
                            if hasattr(item, "text"):
                                print(item.text)
                            else:
                                print(item)
        except BaseExceptionGroup as eg:
            # filter out cancellation errors which are expected during cleanup
            real_errors = [
                e
                for e in eg.exceptions
                if not isinstance(e, (asyncio.CancelledError, GeneratorExit))
            ]
            if real_errors:
                raise ExceptionGroup("SSE client errors", real_errors) from None


def parse_args_to_dict(args: list[str]) -> dict[str, Any]:
    """Parse key=value arguments into a dictionary."""
    result = {}
    for arg in args:
        if "=" not in arg:
            raise ValueError(f"Invalid argument format: {arg} (expected key=value)")
        key, value = arg.split("=", 1)
        # try to parse as JSON for complex types
        try:
            result[key] = json.loads(value)
        except json.JSONDecodeError:
            result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser(description="MCP client for calling tools")
    parser.add_argument(
        "--url",
        default="http://0.0.0.0:8080",
        help="MCP server URL (default: http://0.0.0.0:8080)",
    )
    parser.add_argument(
        "--transport",
        choices=["streamable-http", "sse"],
        default="streamable-http",
        help="Transport protocol (default: streamable-http)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Request timeout in seconds (default: 30)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # list command
    subparsers.add_parser("list", help="List available tools")

    # call command
    call_parser = subparsers.add_parser("call", help="Call a tool")
    call_parser.add_argument("tool_name", help="Name of the tool to call")
    call_parser.add_argument(
        "args",
        nargs="*",
        help="Tool arguments as key=value pairs",
    )
    call_parser.add_argument(
        "--json",
        dest="json_args",
        help="Tool arguments as JSON object",
    )

    args = parser.parse_args()

    # parse tool arguments for call command
    tool_args = None
    if args.command == "call":
        if hasattr(args, "json_args") and args.json_args:
            tool_args = json.loads(args.json_args)
        elif hasattr(args, "args") and args.args:
            tool_args = parse_args_to_dict(args.args)
        else:
            tool_args = {}

    try:
        if args.transport == "sse":
            # use async SSE client
            client = SSEClient(args.url, timeout=args.timeout)
            tool_name = args.tool_name if args.command == "call" else None
            asyncio.run(client.run(args.command, tool_name, tool_args))
        else:
            # use sync streamable-http client
            client = StreamableHttpClient(args.url, timeout=args.timeout)

            # initialize connection
            init_result = client.initialize()
            if "error" in init_result:
                print(f"Failed to initialize: {init_result['error']}", file=sys.stderr)
                sys.exit(1)

            if args.command == "list":
                tools = client.list_tools()
                print(f"Available tools ({len(tools)}):\n")
                for tool in tools:
                    print(f"  {tool['name']}")
                    if "description" in tool:
                        desc = tool["description"]
                        if len(desc) > 80:
                            desc = desc[:80] + "..."
                        print(f"    {desc}")
                    print()

            elif args.command == "call":
                result = client.call_tool(args.tool_name, tool_args)

                # pretty print result
                if isinstance(result, dict) and "content" in result:
                    # MCP tool results have content array
                    for item in result["content"]:
                        if item.get("type") == "text":
                            print(item["text"])
                else:
                    print(json.dumps(result, indent=2))

    except httpx.HTTPStatusError as e:
        print(
            f"HTTP error: {e.response.status_code} - {e.response.text}", file=sys.stderr
        )
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
