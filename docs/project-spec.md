# genetics-mcp-server - Project specification

## Introduction

genetics-mcp-server is a Model Context Protocol (MCP) server and LLM chat service that provides AI assistants and agents with tools to access human genetics data. The server also acts as a bridge between LLMs and a genetics results REST API, translating tool calls into API requests and formatting responses for AI consumption. External MCP servers such as those from gnomAD or Open Targets can also be added in the set of tools available.

## Purpose and Goals

- Provide an MCP server that exposes genetics data tools to AI assistants
- Enable agentic LLM interactions with genetics data through a FastAPI chat service
- Support multiple transport protocols (stdio, SSE, streamable HTTP) for flexible deployment
- Proxy tools from external MCP servers to aggregate genetics data sources
- Maintain clean separation between tool definitions, execution, and LLM integration

## Key Features

- **Standalone MCP Server**: Connects to Claude Desktop, Cursor, or any MCP client via stdio, SSE or streamable HTTP
- **LLM Chat API**: FastAPI service with streaming responses, supporting Anthropic and OpenAI providers
- **Genetics data tools**: Comprehensive access to GWAS, QTL, colocalization, expression, Mendelian disease data, LD, visualizations, and BigQuery for advanced queries
- **Literature and web search**: Integration with Europe PMC, Perplexity, Tavily, and DuckDuckGo
- **External MCP server proxying**: Aggregate tools from remote MCP servers (e.g., gnomAD, Open Targets Platform)
- **Optional IAP/oauth2-proxy authentication**: Protect the chat API via `X-Goog-Authenticated-User-Email` header
- **Per-user API tokens**: Users can create personal bearer tokens for MCP server access, with create/list/revoke management via the chat API
- **Per-user rate limiting**: Sliding window rate limit on chat requests, keyed by user email
- **Cost logging**: Estimated USD cost logged for every Anthropic API call based on token usage and model pricing
- **Chat history persistence**: SQLite-based storage of conversation threads
- **Configurable prompts**: Per-user LLM configuration stored in database

## Technical implementation considerations

- polars should be used to process tabular data from the genetics API
- matplotlib is used for generating scientific visualizations (PheWAS plots, etc.)
- Asynchronous code execution using async/await with httpx for HTTP calls
- The MCP server uses FastMCP from the mcp library for tool registration
- Tool definitions are shared between MCP server and LLM service via a common module
- Error responses from tools should include `success: false` and an `error` field
- Large tool results are truncated to prevent context overflow in LLM conversations
- External tool errors should not cause problems for the server or clients; log and return error response

## Available tools

### Search tools

| Tool | Description |
|------|-------------|
| `search_phenotypes` | Look up phenotype codes by disease/trait name |
| `search_genes` | Look up gene symbols and genomic positions |
| `lookup_variants_by_rsid` | Convert rsIDs to variant IDs (chr:pos:ref:alt format) |
| `lookup_phenotype_names` | Batch translate phenotype codes to human-readable names |

### Credible set tools (fine-mapped GWAS results)

| Tool | Description |
|------|-------------|
| `get_credible_sets_by_gene` | Get credible sets for variants near a gene |
| `get_credible_sets_by_variant` | Find associations containing a specific variant |
| `get_credible_sets_by_phenotype` | Get all GWAS associations for a phenotype |
| `get_credible_set_by_id` | Get all variants in a specific credible set |
| `get_credible_sets_by_qtl_gene` | Get QTL associations where a gene is the molecular trait |
| `get_credible_sets_stats` | Get summary statistics of credible sets for a dataset |

### Gene data tools

| Tool | Description |
|------|-------------|
| `get_gene_expression` | Get tissue-specific gene expression levels |
| `get_gene_disease_associations` | Get Mendelian disease relationships from ClinGen/GENCC |
| `get_exome_results_by_gene` | Get rare variant burden test results |
| `get_gene_based_results` | Get gene-level burden test results from genebass and SCHEMA |
| `get_nearest_genes` | Get genes nearest to a variant position |
| `get_genes_in_region` | Get all genes in a genomic region |

### Other genetics tools

| Tool | Description |
|------|-------------|
| `get_colocalization` | Find traits sharing causal signals at a variant |
| `get_phenotype_report` | Get detailed markdown report for a phenotype |
| `list_datasets` | List all datasets with descriptions, provenance, sample-size stats, and supported products |
| `get_summary_stats` | Get summary statistics (p-value, beta, SE, allele frequencies) for specific variant-phenotype pairs |
| `get_variant_annotations` | Get variant annotations (consequence, allele frequency, rsID, enrichment) by variant, region, gene, or batch variants |

### LD tools (FinnGen LD Server)

| Tool | Description |
|------|-------------|
| `get_ld_between_variants` | Get LD (r2, D') between two specific variants using FinnGen reference panel |
| `get_variants_in_ld` | Get all variants in LD with a query variant within a specified window |

### Visualization tools

| Tool | Description |
|------|-------------|
| `create_phewas_plot` | Create a PheWAS plot showing phenotype associations for a variant (returns base64 PNG) |
| `analyze_variant_list` | Analyze a list of variants for shared phenotype associations, QTL patterns, tissue enrichment, and nearest genes |

### BigQuery tools (fallback for complex queries)

| Tool | Description |
|------|-------------|
| `query_bigquery` | Execute custom SQL against genetics views (fallback for queries specialized tools cannot handle) |
| `get_bigquery_schema` | Get schema for BigQuery views before writing queries |

BigQuery contains multiple tables beyond just credible sets — including exome/burden test results, colocalization, and more. The `get_bigquery_schema` tool discovers all available tables. Views include a derived `resource` column that maps dataset names to resource identifiers (e.g., `FinnGen_R13` → `finngen`, `UKB_PPP` → `ukbb`, `Open_Targets_25.12` → `open_targets`). This allows filtering by `WHERE resource = 'finngen'` instead of matching dataset names directly.

### External search tools

| Tool | Description |
|------|-------------|
| `search_scientific_literature` | Search PubMed/bioRxiv via Europe PMC or Perplexity |
| `web_search` | General web search via Tavily or DuckDuckGo |

### External MCP server tools (proxied)

#### gnomAD MCP

Provides variant population frequency and gene constraint data from gnomAD.

#### Open Targets Platform MCP

| Tool | Description |
|------|-------------|
| `get_open_targets_graphql_schema` | Retrieve the Open Targets Platform GraphQL schema for query construction |
| `search_entities` | Search for targets, diseases, drugs, variants, and studies by name |
| `query_open_targets_graphql` | Execute GraphQL queries against the Open Targets Platform API |
| `batch_query_open_targets_graphql` | Execute the same GraphQL query with multiple variable sets |

## Tool Profiles

The chat API supports a `tool_profile` parameter that controls which tool categories are available per request. This enables A/B testing of different tool strategies (API vs BigQuery vs RAG) by sending identical prompts with different profiles.

### Tool categories

Each tool has a `category` field in its definition:

| Category | Description |
|----------|-------------|
| `general` | Always available: search_phenotypes, search_genes, lookup_phenotype_names, list_datasets, search_scientific_literature, web_search, create_phewas_plot |
| `api` | Local genetics API tools: credible sets, gene data, colocalization, phenotype report, variant annotations, etc. |
| `bigquery` | BigQuery SQL tools: query_bigquery, get_bigquery_schema |
| `orchestration` | Main-agent-only tools: launch_subagents. Excluded from subagent tool sets to prevent recursive launches. |

### Profile behavior

| `tool_profile` value | Local tools | External tools |
|----------------------|-------------|----------------|
| `null` (default) | general + api + bigquery + orchestration | always-on (gnomAD, OT) + RAG |
| `"api"` | general + api + orchestration | always-on only |
| `"bigquery"` | general + bigquery + orchestration | always-on only |
| `"rag"` | general only | RAG only |

Always-on external servers (gnomAD, Open Targets from `EXTERNAL_MCP_SERVERS`) are included in every profile except `"rag"`. The RAG server (`RAG_MCP_SERVER`) is only included when `tool_profile` is `"rag"` or unset.

## Architecture

### Module structure

```
src/genetics_mcp_server/
├── __init__.py
├── mcp_server.py        # standalone MCP server entry point
├── mcp_client.py        # MCP client for testing
├── mcp_proxy.py         # proxy for external MCP servers
├── chat_api.py          # FastAPI chat service
├── llm_service.py       # LLM provider integration
├── logging_config.py    # GCP Cloud Logging JSON formatter
├── rate_limit.py        # per-user sliding window rate limiter
├── cost.py              # Anthropic API cost estimation
├── download_store.py    # disk-persisted download storage for TSV files
├── config/
│   ├── __init__.py
│   ├── settings.py      # configuration dataclass
│   └── defaults.py      # default prompts and values
├── tools/
│   ├── __init__.py
│   ├── definitions.py   # tool definitions (shared)
│   ├── executor.py      # tool execution via HTTP
│   └── phewas_categories.py  # PheWAS plot category mappings
├── subagent.py             # parallel subagent service
├── scripts/
│   └── analyze_variants.py # standalone variant list analysis CLI
├── skills/
│   ├── __init__.py
│   ├── definitions.py      # skill definitions and registry
│   ├── sandbox_tools.py    # file read and script execution tools
│   └── instructions/       # markdown instruction files per skill
├── auth/
│   ├── __init__.py
│   ├── core.py          # IAP/oauth2-proxy header extraction
│   └── dependencies.py  # FastAPI auth dependencies
├── db/
│   ├── __init__.py
│   ├── singleton.py     # async DB singleton base
│   ├── llm_config_db.py    # user LLM config storage
│   └── chat_history_db.py  # conversation persistence
└── routers/
    ├── __init__.py
    ├── api_tokens.py    # per-user API token management
    ├── llm_config.py    # LLM config API endpoints
    └── chat_history.py  # chat history API endpoints
```

### Data flow

1. **MCP Server mode**: Client → FastMCP → ToolExecutor → Genetics API
2. **Chat API mode**: HTTP → FastAPI → LLMService → Anthropic/OpenAI → ToolExecutor → Genetics API
3. **Subagent mode**: Main Agent → `launch_subagents` tool → SubagentService → parallel Claude API calls → ToolExecutor/External Tools → results aggregated back to main agent

### Tool execution

Tools are defined once in `tools/definitions.py` with:
- Name and description
- Parameter schemas (type, required, defaults)
- Registration helpers for both FastMCP and Anthropic formats

The `ToolExecutor` class implements each tool as an async method that:
1. Builds the request to the genetics API
2. Handles errors and 404 responses gracefully
3. Optionally summarizes large result sets
4. Returns structured JSON with `success` flag

### External MCP proxying

The `mcp_proxy.py` module allows connecting to remote MCP servers:
1. Fetches tool definitions via JSON-RPC initialize/tools/list
2. Dynamically creates wrapper functions using exec()
3. Forwards tool calls to the remote server
4. Parses SSE responses and extracts JSON-RPC results

### Subagent system

The subagent system enables the main agent to launch parallel specialized agents for complex queries. When enabled, the main agent has access to a `launch_subagents` tool that dispatches tasks to specialized subagents.

**Skills** define subagent capabilities:
- `genetics_data_extraction` — API tools for GWAS, QTL, credible sets, etc.
- `literature_review` — scientific literature and web search
- `bigquery_analysis` — complex SQL queries against the genetics database
- `variant_list_analysis` — analyze multiple variants for shared patterns
- `data_analysis` — Python script execution for custom analysis and visualizations

Each skill has:
- A markdown instruction file (system prompt) in `skills/instructions/`
- Tool categories controlling which tools the subagent can use
- Configurable model, max iterations, and timeout
- Optional sandbox tools (file read, script execution)
- `include_external` flag — when `True`, external MCP server tools (e.g. gnomAD, Open Targets) are appended to the subagent's tool set via `get_external_anthropic_tools()`. Currently enabled for `genetics_data_extraction`.

**Recursive launch prevention**: The `launch_subagents` tool has category `orchestration`, which is included only for the main agent. Subagent tool sets explicitly exclude `launch_subagents` to prevent recursive launches.

**Cost and token tracking**: `SubagentResult` accumulates `input_tokens` and `output_tokens` across all iterations of a subagent's agentic loop. After `launch_subagents` completes, `llm_service.py` sums tokens across all subagent results and logs an aggregated cost estimate using the same `estimate_cost()` function as the main agent.

**Subagent IDs**: Each subagent receives a unique ID (`sa-1`, `sa-2`, ...) assigned sequentially when `run_subagents()` launches them. The ID appears in all log messages and progress callbacks, e.g. `Subagent 'literature_review' [sa-2] calling search_scientific_literature(query='PCSK9')`.

**Progress streaming**: Subagent progress is streamed to the user in real time via an `asyncio.Queue` bridge:
1. `SubagentService.run_subagents()` accepts an optional `progress_callback` invoked at subagent start, each tool call, completion, and failure
2. Progress messages include the subagent ID and tool call parameters formatted by `_format_tool_params()` — a helper that produces compact `(key='value', ...)` strings, truncating long values and representing complex types as `<list>`/`<dict>`
3. In `llm_service.py`, the callback puts messages onto an `asyncio.Queue`
4. The main streaming loop drains the queue, yielding each message as an SSE `StreamChunk` (displayed as italicized status text)
5. A sentinel `None` signals all subagents have finished, ending the drain loop
6. Regular tools and subagents run concurrently — regular tool tasks are gathered alongside the subagent task

**System prompt orchestration guidance**: The default system prompt (`config/defaults.py`) includes a "Subagent Orchestration" section that tells the LLM:
- When to use subagents vs direct tool calls (parallel independent tasks vs simple lookups)
- Available skills and their best use cases
- How to structure subagent tasks (self-contained questions, pass context explicitly, split by skill not entity)

**Skill instructions**: Each skill has a markdown instruction file in `skills/instructions/` that serves as the subagent's system prompt. Instructions include:
- Guidelines for structured output format (exact numbers, systematic organization)
- Error handling rules (report missing data explicitly, handle tool failures gracefully)
- Data source mapping guidance (use `list_datasets` to discover datasets, case-sensitive data types)
- Scope constraints (e.g., data extraction skills should not interpret, just organize)

**Execution flow**:
1. Main agent calls `launch_subagents` with a list of skill+query tasks
2. `SubagentService` validates skills and launches subagents in parallel via `asyncio.gather()`
3. Each subagent runs its own agentic loop (non-streaming) with Claude API
4. Progress callbacks stream status to the user via `asyncio.Queue`
5. Results (including per-subagent token counts) are collected and returned to the main agent
6. Main agent synthesizes subagent outputs into its response

**Security**:
- File access restricted to configured `SUBAGENT_ALLOWED_PATHS` directories
- Script execution gated behind `ENABLE_SCRIPT_EXECUTION` flag
- Interpreter whitelist: `python3`, `Rscript`, `bash`
- Sensitive environment variables stripped from script processes
- Per-subagent and per-script timeouts

## Configuration

All configuration is via environment variables (`.env` file supported):

### Required

| Variable | Description |
|----------|-------------|
| `GENETICS_API_URL` | Base URL of the genetics REST API |
| `BIGQUERY_API_URL` | Base URL of the BigQuery proxy API |

### LLM providers (for chat API)

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude |
| `OPENAI_API_KEY` | OpenAI API key |

### Search tools (optional)

| Variable | Description |
|----------|-------------|
| `TAVILY_API_KEY` | Tavily API key for web search |
| `PERPLEXITY_API_KEY` | Perplexity API key for literature search |
| `LITERATURE_SEARCH_BACKEND` | Backend: `europepmc` (default) or `perplexity` |

### Database and storage

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_CONFIG_DB` | Path to LLM config SQLite DB | `/path/to/llm_config.db` |
| `CHAT_HISTORY_DB` | Path to chat history SQLite DB | `/path/to/chat_history.db` |
| `ATTACHMENT_STORAGE_PATH` | Path for file attachment storage | `/path/to/attachments` |
| `MAX_ATTACHMENT_SIZE` | Max attachment size in bytes | `52428800` (50MB) |
| `DOWNLOAD_STORAGE_PATH` | Path for tool result download files | `/mnt/disks/data/downloads` |
| `DOWNLOAD_TTL_SECONDS` | TTL for download files in seconds | `2592000` (30 days) |

### Authentication (optional)

| Variable | Description |
|----------|-------------|
| `REQUIRE_AUTH` | Require `X-Goog-Authenticated-User-Email` header (`true`/`false`) |
| `MCP_API_KEY` | Comma-separated bearer tokens for MCP server SSE/HTTP transport auth |

Per-user API tokens are also supported: users create tokens via the chat API (`POST /chat/v1/tokens`), which are validated alongside `MCP_API_KEY` in the MCP server's bearer auth middleware. Tokens are stored as SHA-256 hashes in the LLM config SQLite DB.

### Rate limiting

| Variable | Description | Default |
|----------|-------------|---------|
| `RATE_LIMIT_PER_HOUR` | Max chat messages per user per hour | `20` |
| `RATE_LIMIT_PER_DAY` | Max chat messages per user per day | `100` |

Rate limiting is per user email (from `X-Goog-Authenticated-User-Email` header) and applies to `POST /chat/v1/chat`. Both limits use sliding windows. Returns HTTP 429 with the specific limit hit when exceeded.

### MCP server options

| Variable | Description |
|----------|-------------|
| `LOG_LEVEL` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` (default `INFO`) |
| `MCP_DISABLE_TRANSPORT_SECURITY` | Allow all hosts/origins (dev only) |
| `EXTERNAL_MCP_SERVERS` | Comma-separated URLs of always-on external MCP servers (gnomAD, Open Targets) |
| `EXTERNAL_MCP_EXCLUDE_TOOLS` | Tool names to exclude from proxying |
| `ENABLE_CREDIBLE_SETS_STATS` | Enable `get_credible_sets_stats` tool (default `false`) |
| `RAG_MCP_SERVER` | URL of the RAG MCP server (only included when `tool_profile` is `"rag"` or unset) |

Default external servers:
- gnomAD: `https://gnomad-mcp-dpsnoyqx6q-uc.a.run.app`
- Open Targets: `https://mcp.platform.opentargets.org`

### Subagent options

| Variable | Description | Default |
|----------|-------------|---------|
| `ENABLE_SUBAGENTS` | Enable the `launch_subagents` tool | `false` |
| `SUBAGENT_MODEL` | Model for subagents (falls back to `fast_model`) | `""` |
| `SUBAGENT_TIMEOUT` | Seconds per subagent execution | `120` |
| `SUBAGENT_ALLOWED_PATHS` | Comma-separated directories for file access | `""` |
| `ENABLE_SCRIPT_EXECUTION` | Allow subagents to execute scripts | `false` |
| `SUBAGENT_SCRIPT_TIMEOUT` | Seconds per script execution | `30` |

## Logging

The application uses structured JSON logging for GCP Cloud Logging via `logging_config.py`:

- **GCPJsonFormatter**: Outputs JSON with `timestamp`, `severity`, `logger`, `message`, and optional `exception` fields. On GKE, stdout is automatically captured by fluentbit and sent to Cloud Logging.
- **MCP Server (stdio)**: Uses standard Python logging to stderr (stdout reserved for MCP protocol)
- **MCP Server (SSE/HTTP)** and **Chat API**: Use GCP JSON logging to stdout
- **Log level**: Controlled by `LOG_LEVEL` env var (default `INFO`)
- **Noisy loggers suppressed**: `uvicorn.access`, `httpx`, `httpcore`, `urllib3`, `asyncio` are set to WARNING

**Cost logging**: Every Anthropic API call logs estimated cost based on model pricing and token usage (input, output, cache read, cache creation). A summary line is logged when the chat completes with total tokens and cost. Cost is logged even for secret chats. User email is included in all log lines. Subagent API calls also track token usage: `SubagentResult` includes `input_tokens` and `output_tokens` accumulated across all iterations, and an aggregated cost log line is emitted after `launch_subagents` completes.

Log levels:
- **INFO**: Server startup, tool registration, external server connections, API call cost
- **WARNING**: Fallback scenarios (e.g., Tavily→DuckDuckGo)
- **ERROR**: API failures, tool execution errors (with tracebacks)
- **DEBUG**: HTTP call details, SSE parsing

## Testing

Tests are in `tests/` using pytest with pytest-asyncio:

| Test file | Coverage |
|-----------|----------|
| `test_mcp_server.py` | MCP server initialization and tool registration |
| `test_chat_api.py` | FastAPI endpoints (status, tools, chat) |
| `test_tools.py` | Tool executor methods |
| `test_db.py` | Database operations |
| `test_chat_history_router.py` | Chat history API |
| `test_llm_config_router.py` | LLM config API |
| `test_phewas_categories.py` | PheWAS category mappings |
| `test_subagent.py` | Subagent service, skills, sandbox tools |
| `test_variant_analysis.py` | Variant list analysis tool |
| `test_downloads.py` | Download store, TSV conversion, download endpoint |

Run tests:
```bash
pytest
pytest --cov=src/genetics_mcp_server  # with coverage
```

## Development Workflow

- **Issue tracking**: beads (`bd`) tracks epics and tasks in `.beads/`, synced with git
- **Feature planning**: new features go through architecture exploration (`.claude/agents/architecture-explorer.md`) which proposes 3 alternatives, then the selected approach is broken into ultrafocused subtasks in beads
- **Task execution**: work through subtasks via `bd ready`, updating status as you go

## Documentation

- `README.md`: Installation, quick start, tool reference
- `.env.example`: All configuration variables
- This document: Architecture and implementation details

## Architecture decisions

1. **Shared tool definitions**: Single source of truth in `definitions.py` prevents drift between MCP and LLM service
2. **Async throughout**: All I/O uses async/await for concurrent tool execution
3. **Graceful degradation**: External service failures don't crash the server; fallbacks are used where available
4. **Streaming responses**: Chat API streams tokens via SSE for responsive UX
5. **Agentic loop**: LLM service supports multi-turn tool use with configurable iteration limit
6. **Retry on transient errors**: Anthropic API calls are retried up to 2 times with exponential backoff (1s, 2s) for transient errors (HTTP 500, 502, 503, 529). If text was already streamed before the error, the user is notified with a "[Connection interrupted, retrying...]" message. Non-retryable errors (auth, bad request, rate limit) propagate immediately.
7. **Result truncation**: Large responses are truncated with warnings to prevent context overflow
8. **Downloadable results**: Tools returning tabular data include `INCLUDE_IN_RESPONSE` download links. Direct API URLs are used for genetics API tools that support TSV format; other tools (BigQuery, LD, summary stats) have their results converted to TSV and stored on disk, served via `/chat/v1/downloads/{id}`. The `_download_url` and `_download_data` hints in tool results are processed by `_process_download_hints()` in `llm_service.py` before being sent to the LLM. All download links use relative URLs (e.g., `/api/v1/...` or `/chat/v1/downloads/...`) so they work correctly regardless of deployment domain. `INCLUDE_IN_RESPONSE` is placed at the front of the result dict so it survives JSON truncation for large results. For BigQuery, trailing SQL `LIMIT` clauses are stripped and `max_rows` is set to 100,000 so the download contains the full result set even when the LLM only displays a subset. The BigQuery proxy (`genetics-results-db`) enforces `MAX_ROWS=100000` as a hard cap.

## Future considerations

1. Add caching for genetics API responses
2. Support additional LLM providers (Google, local models)
3. Implement tool-level access control
4. Add WebSocket transport for bidirectional streaming
