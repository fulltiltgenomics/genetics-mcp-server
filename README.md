# Genetics MCP Server

A Model Context Protocol (MCP) server and LLM chat service for genetics data tools.

This repository provides:
1. **Standalone MCP Server** - Connect any MCP client to genetics data tools
2. **LLM Chat API** - Chat service with Anthropic integration

This is deployed as part of FinnGenie AI assistant (see [https://github.com/fulltiltgenomics/genetics-results-suite](https://github.com/fulltiltgenomics/genetics-results-suite)).

## Quick Start

### Install requirements

```
uv pip install --system .
```

### Environment Variables

All environment variables are optional but needed for each type of functionality.

| Variable | Description | Default |
|----------|-------------|---------|
| `GENETICS_API_URL` | Base URL for [genetics-results-api](https://github.com/fulltiltgenomics/genetics-results-api) server | `http://0.0.0.0:2000/api` |
| `BIGQUERY_API_URL` | Base URL for [genetics-results-db](https://github.com/fulltiltgenomics/genetics-results-db) server | - |
| `ANTHROPIC_API_KEY` | Anthropic API key (for chat) | - |
| `OPENAI_API_KEY` | OpenAI API key (for chat) | - |
| `TAVILY_API_KEY` | Tavily API key (for web search) | - |
| `EXTERNAL_MCP_SERVERS` | Comma-separated URLs of external MCP servers to proxy (e.g. gnomAD, Open Targets) | - |
| `RAG_MCP_SERVER` | URL of [genetics-rag-service](https://github.com/ykjain/genetics-rag-service) server | - |


### MCP Server

```bash
cd src
export GENETICS_API_URL=https://.../api
# stdio transport
python -m genetics_mcp_server.mcp_server
# streamable http
python -m genetics_mcp_server.mcp_server --transport streamable-http --port 8080 --host 0.0.0.0
```

### Chat API Server

```bash
cd src
export ANTHROPIC_API_KEY=sk-ant-...
export GENETICS_API_URL=https://.../api
uvicorn genetics_mcp_server.chat_api:app --port 8000
```

## Development

```bash
git clone https://github.com/fulltiltgenomics/genetics-mcp-server.git
cd genetics-mcp-server
uv sync --extra dev

# tests
uv run pytest

# lint
uv run ruff check src/
```

## License

MIT
