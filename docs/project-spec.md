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
- **Per-message size limits**: `_validate_latest_message` (in `chat_api.py`) caps the newest user message's typed-text length (`MAX_MESSAGE_CHARS`, default 50K) and attachment count (`MAX_ATTACHMENTS_PER_MESSAGE`, default 10), rejecting with HTTP 413 before any model call. Attachments are excluded from the text cap: images arrive as `image` blocks and data files (TSV/CSV/Excel) are inlined by the frontend as text blocks prefixed `[File: <name>]` — both are counted toward the attachment limit, not the character limit. The frontend (`LLMChat.tsx`) mirrors these limits for immediate feedback. Bulk data should be attached as a file rather than pasted
- **File attachments**: Upload/download/delete endpoints in `routers/chat_history.py` store files on disk (`ATTACHMENT_STORAGE_PATH`) with metadata in the `chat_attachments` table. Files are classified as `image`, `tsv`, or `excel`. Excel is a binary format, so `.xlsx`/`.xls` uploads are parsed to TSV at upload time via `excel_to_tsv()` (polars `read_excel`, calamine/`fastexcel` engine; all sheets, each prefixed `# Sheet: <name>` when multiple) and the parsed text is stored as a `.tsv` sidecar (`text_path` column); a file that fails to parse is rejected with HTTP 400 and nothing is written. The download endpoint serves the original bytes by default, or the model-ready text via `?as=text` (parsed TSV for excel, original for tsv/csv). The live frontend send path does not round-trip through these endpoints — it parses Excel→TSV client-side with SheetJS (`excelToTsv.ts`) before inlining, since sessions are created lazily after the first exchange and no `session_id` exists at first send. The server-side parse is therefore defense-in-depth: it covers direct API consumers and guarantees stored bytes are never surfaced as binary; `?as=text` is available for any client that prefers a backend round-trip
- **Cost logging**: Estimated USD cost logged for every Anthropic API call based on token usage and model pricing
- **Context usage tracking**: `get_context_window()` in `cost.py` maps model name prefixes to context window sizes (tokens). During streaming, `usage` SSE events are emitted after each agentic loop iteration, enabling the frontend to display a live context usage progress bar
- **Chat history persistence**: SQLite-based storage of conversation threads. Assistant turns persist both their content blocks (`content_json`: text + `tool_use`) and the tool outputs (`tool_results_json`: the `tool_result` blocks). Persisting tool results means a **resumed** conversation replays the actual data the model saw, not just its prose summary — preventing factual drift across turns/sessions (see "Tool result persistence" under Architecture decisions)
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
| `get_gene_group_members` | Enumerate member genes of an HGNC gene group / family by `group_id` or `group_name`, returning symbols + genomic coordinates. Olfactory receptors are **excluded by default** (`exclude_olfactory=true`) since they are GPCRs that dominate large families by count; pass `exclude_olfactory=false` for full membership. Calls the API (`GET /api/v1/gene_group/members`), not BigQuery. For whole-group BigQuery joins (e.g. cis-pQTL colocalizations for all GPCRs), prefer filtering `gene_annotations_v` on `gene_group_ids`/`gene_group_names` directly |
| `normalize_gene_symbols` | Resolve gene symbols / aliases / previous symbols to current approved HGNC symbols (exact, not fuzzy); returns mappings + unresolved inputs. Calls the API (`GET /api/v1/gene/normalize`), not BigQuery |

### Credible set tools (fine-mapped GWAS results)

| Tool | Description |
|------|-------------|
| `get_credible_sets_by_gene` | Get credible sets for variants near a gene (gene body ± `window`, default **500 kb**). The wide default is deliberate: the strongest signal attributed to a gene can sit several hundred kb away (e.g. a long-range regulatory variant), so a narrow window can silently drop the top hit |
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
| `get_exome_results_by_gene` | Get rare variant burden test results (genebass filtered to p < 1e-4, IBD exome-wide significant only) |
| `get_exome_results_by_phenotype` | Get exome variant results for a specific phenotype across all genes (genebass and IBD) |
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
| `get_myvariant_annotations` | Get clinical/functional annotations from myvariant.info (ClinVar, CADD, functional predictions, cancer data). Chat-backend only — excluded from MCP server |

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
| `get_bigquery_schema` | Get schema for BigQuery views before writing queries. Accepts optional `table` parameter to get just one table's schema. Returns resource metadata with aliases, column descriptions, allowed filter values, and example SQL queries |

BigQuery contains multiple tables beyond just credible sets — including exome/burden test results, colocalization, and more. The `get_bigquery_schema` tool discovers all available tables.

A `gene_annotations` BigQuery table/view (built in `genetics-results-db`) is also exposed via `get_bigquery_schema` and is the `query_bigquery` surface for cis/trans QTL filtering — JOIN its gene coordinates to `colocalization_v` instead of hand-typing coordinate literals — and for any-group gene enumeration. This stands apart from the two specialized gene tools (`get_gene_group_members`, `normalize_gene_symbols`), which call the API and do **not** read this table. **Coordinate windows, not `gene_most_severe`, for "near a gene" queries**: when finding signals (GWAS or QTL) physically near a gene in BigQuery, JOIN `gene_annotations_v` for the gene body and filter on a coordinate window (≈ 500 kb), rather than filtering by `gene_most_severe`. The latter is per-variant most-severe-consequence attribution — unreliable for regulatory variants and prone to both missing nearby signals and mis-attributing distant ones. `get_asm_qtl_by_gene` now selects by this coordinate window, and the system prompt instructs the LLM to do the same for ad-hoc SQL. (`get_credible_sets_by_qtl_gene` is the exception: it finds QTLs where the gene is the *molecular trait*, correctly keyed by gene name.) Single source of truth split: the specialized tools resolve gene groups/symbols via the API; BigQuery's `gene_annotations` stands alone as the surface for SQL JOINs and ad-hoc enumeration. Views include a derived `resource` column that maps dataset names to resource identifiers (e.g., `FinnGen_R13` → `finngen`, `UKB_PPP` → `ukbb`, `Open_Targets_25.12` → `open_targets`). This allows filtering by `WHERE resource = 'finngen'` instead of matching dataset names directly. The schema response includes resource metadata with human-readable labels and aliases to help agents map user intent to correct filter values (e.g., "bipex" → `resource = 'bipex2'`). Collection resources like eQTL Catalogue are collapsed into summaries rather than listing hundreds of individual IDs.

### External search tools

The external search tools split into two conceptually distinct families:

- **Literature backends** (`search_scientific_literature`): query a paper-indexing API — either `europepmc` (covers PubMed, Europe PMC, bioRxiv, medRxiv) or `perplexity` (broader scientific web). Exactly one backend is queried per call; "backend" is the API hit, not the content source indexed.
- **Structured curated databases** (`search_mgi`): query a curated biological database that returns structured records (genes, phenotypes, alleles, orthologs) rather than papers. Complements — does not replace — the literature backends.

| Tool | Description |
|------|-------------|
| `search_scientific_literature` | Search PubMed/bioRxiv via Europe PMC or Perplexity |
| `web_search` | General web search via Tavily or DuckDuckGo |
| `search_mgi` | Search Jackson Lab Mouse Genome Informatics for curated mouse gene→phenotype annotations, knockout/transgenic allele phenotypes, and human-mouse ortholog mappings. Chat-backend only — excluded from MCP server |

#### MGI (native tool, chat-backend only)

The `search_mgi` tool queries Jackson Lab's MouseMine (InterMine REST endpoint) for curated mouse data: gene → MP-ontology phenotype terms, knockout/transgenic allele phenotypes, and mouse-human ortholog mappings. Unlike Europe PMC and Perplexity which return papers, MGI returns structured curated records — so it complements rather than substitutes for literature search. Excluded from the MCP server (mirroring `get_myvariant_annotations`); only available via the chat API.

### External MCP server tools (proxied)

#### gnomAD MCP

Provides variant annotations, population frequencies, gene constraint/expression data, and pathogenicity interpretation from gnomAD. Server name: `gmd-agent`. Tools are registered without prefix. Five All of Us (AoU) tools are excluded via `EXTERNAL_MCP_EXCLUDE_TOOLS`.

Available tools (after exclusions):

| Tool | Description |
|------|-------------|
| `get_variant_details` | Variant details/summary |
| `get_variant_frequencies` | Population allele frequencies |
| `get_variant_summary` | Variant summary |
| `get_multiple_variant_details` | Batch variant details |
| `interpret_variant_pathogenicity` | Interpret variant pathogenicity |
| `analyze_variant_cooccurrence` | Phase relationship (cis vs trans) between variants |
| `analyze_variant_pext` | Proportion expressed across transcripts (pext) score |
| `get_gene_summary` | Gene summary including constraint scores |
| `get_gene_expression_summary` | Gene expression summary |
| `get_gene_variants` | Variants for a gene |
| `get_mendelian_gene_summary` | Mendelian disease gene summary |
| `get_region_variants` | Variants in a genomic region |
| `get_transcript_details` | Transcript details |
| `list_gene_transcripts` | List transcripts for a gene |
| `get_agent_info` | Agent information |

#### myvariant.info (native tool, chat-backend only)

The `get_myvariant_annotations` tool queries myvariant.info for clinical and functional variant annotations not available from gnomAD MCP or the local API. Provides ClinVar clinical significance, CADD deleteriousness scores, functional predictions (SIFT, PolyPhen2, MutationTaster), cancer annotations (COSMIC, CIViC), and dbSNP rsIDs. Excluded from the MCP server — only available via the chat API. gnomAD population frequency fields are excluded by default to avoid overlap with gnomAD MCP.

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
| `general` | Always available: search_phenotypes, search_genes, lookup_phenotype_names, list_datasets, search_scientific_literature, web_search, create_phewas_plot, get_gene_group_members, normalize_gene_symbols |
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
│   ├── analyze_variants.py # standalone variant list analysis CLI
│   ├── analyze_conversations.py # conversation history analysis and eval extraction
│   ├── plot_conversation_scores.py # time-series plots of quality over time (from metrics.json)
│   └── conversation_prompts.py  # LLM prompt templates for topic categorization
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
    ├── admin.py         # admin page: all conversations, analytics
    ├── api_tokens.py    # per-user API token management
    ├── llm_config.py    # LLM config API endpoints
    └── chat_history.py  # chat history API endpoints
```

### Data flow

1. **MCP Server mode**: Client → FastMCP → ToolExecutor → Genetics API
2. **Chat API mode**: HTTP → FastAPI → LLMService → Anthropic/OpenAI → ToolExecutor → Genetics API
3. **Subagent mode**: Main Agent → `launch_subagents` tool → SubagentService → parallel Claude API calls → ToolExecutor/External Tools → results aggregated back to main agent

### SSE event types

The chat API streams responses as Server-Sent Events (SSE). Each event is a JSON object with a `type` field:

| Event type | Description | Key payload fields |
|------------|-------------|--------------------|
| `content` | Streamed text token from the LLM response | `content` (string) |
| `usage` | Context usage snapshot after each agentic loop iteration | `iteration`, `input_tokens`, `output_tokens`, `total_input_tokens`, `total_output_tokens`, `context_window`, `context_percent` |
| `image` | Base64-encoded image (e.g., PheWAS plot) | `content` (base64 string) |
| `error` | Error message from the backend | `content` (error string) |
| `done` | Signals the stream is complete | `message_content` (assistant text + `tool_use` blocks for persistence), `tool_results` (the `tool_result` blocks for this turn, for persistence) |

The `usage` event is emitted by `_stream_anthropic()` in `llm_service.py` after token accounting in each iteration of the agentic loop. It is yielded as a `StreamChunk(type="usage")` with a JSON-serialized payload. The `event_generator()` in `chat_api.py` forwards it as an SSE event, spreading the usage fields into the top-level payload alongside `"type": "usage"`.

Payload fields for `usage`:
- `iteration` — current agentic loop iteration number
- `input_tokens` — input tokens consumed in the current API call
- `output_tokens` — output tokens generated in the current API call
- `total_input_tokens` — cumulative input tokens across all iterations
- `total_output_tokens` — cumulative output tokens across all iterations
- `context_window` — total context window size for the model (from `get_context_window()`)
- `context_percent` — percentage of context window consumed (`total_input_tokens / context_window * 100`)

The frontend uses `usage` events to render a live progress bar showing how much of the model's context window has been consumed during the conversation.

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

**Advertisement gated on availability**: `launch_subagents` is advertised to the LLM only when the subagent service actually initialized (`self.subagent_service is not None`), not merely when `ENABLE_SUBAGENTS` is set. The service requires a live Anthropic client + executor in addition to the flag, so `_stream_anthropic()` adds `launch_subagents` to the effective `disabled_tools` whenever the service is absent. This single source of truth prevents the LLM from seeing a tool that would return "subagent service isn't available" on call.

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

### myvariant.info (optional, chat-backend only)

| Variable | Description | Default |
|----------|-------------|---------|
| `MYVARIANT_API_URL` | myvariant.info API base URL | `https://myvariant.info/v1` |

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
| `MAX_MESSAGE_CHARS` | Max typed-text characters in a single user message (excludes attachments) | `50000` |
| `MAX_ATTACHMENTS_PER_MESSAGE` | Max attachment blocks (image/document/inlined data file) per message | `10` |
| `DOWNLOAD_STORAGE_PATH` | Path for tool result download files | `/mnt/disks/data/downloads` |
| `DOWNLOAD_TTL_SECONDS` | TTL for download files in seconds | `2592000` (30 days) |

### Authentication (optional)

| Variable | Description |
|----------|-------------|
| `REQUIRE_AUTH` | Require `X-Goog-Authenticated-User-Email` header (`true`/`false`) |
| `MCP_API_KEY` | Comma-separated bearer tokens for MCP server SSE/HTTP transport auth |
| `ALLOWED_EMAILS` | Comma-separated email allow-list for Google Identity Token (JWT) bearer auth |
| `ALLOWED_EMAIL_DOMAINS` | Comma-separated email-domain allow-list for Google Identity Token (JWT) bearer auth (default: `finngen.fi`) |

Tokens can be supplied either as an `Authorization: Bearer XXX` header or as a `?token=XXX` query parameter. The query parameter is useful for clients that don't support custom headers (e.g., claude.ai). When both are present, the Bearer header takes precedence.

The bearer auth middleware (`_wrap_with_bearer_auth` in `mcp_server.py`) routes each presented token through three branches in order, mirroring the results-api implementation:

1. **`MCP_API_KEY` shared secret** — constant-time compare against each configured value
2. **Google Identity Token (JWT)** — if the token contains `.` it is treated as a JWT and validated via `google.oauth2.id_token.verify_oauth2_token` using a lazily-initialized singleton `google.auth.transport.requests.Request` (for JWKS caching). The payload must have `email_verified == True`; the email must be in `ALLOWED_EMAILS` or its domain in `ALLOWED_EMAIL_DOMAINS` (otherwise 401/403). Identity is set to the verified email.
3. **Per-user API token** — fall back to validating against the local LLM config DB (SHA-256 hashed) or via the chat-backend `/v1/tokens/validate` endpoint. Users create tokens via the chat API (`POST /chat/v1/tokens`).

In deployment, `ALLOWED_EMAILS` and `ALLOWED_EMAIL_DOMAINS` are sourced from the shared `bearer-auth-allowed` Kubernetes ConfigMap (defined in `genetics-results-suite/k8s/configs/`), which is also consumed by results-api so both services share an identical allow-list.

### Admin page

| Variable | Description | Default |
|----------|-------------|---------|
| `ENABLE_ADMIN_PAGE` | Enable admin page and API endpoints | `false` |
| `ADMIN_USERS` | Comma-separated admin email addresses | `""` |

When `ENABLE_ADMIN_PAGE=true`, admin endpoints are available at `/chat/v1/admin/`. Access control depends on `REQUIRE_AUTH`:
- `REQUIRE_AUTH=false` (dev mode): any user can access admin endpoints
- `REQUIRE_AUTH=true`: only users listed in `ADMIN_USERS` can access admin endpoints

Admin endpoints:
- `GET /chat/v1/admin/sessions` — list all sessions with filters (user, date range, session ID) and pagination
- `GET /chat/v1/admin/sessions/{id}` — session detail with all messages
- `GET /chat/v1/admin/analytics/usage?period=week|month|year` — daily usage stats (unique users, conversations)
- `GET /chat/v1/admin/feedback` — unified, paginated feed of all user feedback sorted by `created_at` DESC. Merges two sources: standalone feedback from the `user_comments` table (submitted via the Feedback dialog) and per-session comments from `chat_sessions.comment`. Response includes `items` (each with `user`, `comment`, `preview`, `created_at`, `source`, and optional `session_id`), `total` count, `latest_at` timestamp, and pagination parameters (`offset`, `limit`)

The `/chat/v1/auth` endpoint includes an `is_admin` boolean in its response, used by the frontend to show/hide the admin menu.

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
| `test_analyze_conversations.py` | Conversation analysis: parsing, categorization, metrics, eval export |
| `test_admin_router.py` | Admin router endpoints, auth guards, DB methods |
| `test_cost.py` | Cost estimation and context window lookup |

Run tests:
```bash
pytest
pytest --cov=src/genetics_mcp_server  # with coverage
```

## Conversation Analysis

`scripts/analyze_conversations.py` is an offline tool that reads the chat-history
SQLite DB, persists per-conversation analysis results back into that DB (the
`conversation_analysis` / `conversation_issue` tables), and produces a markdown
report (`report.md`) plus an eval dataset. With `--output-dir` it also writes a
local-dev `metrics.json` (consumed by `plot_conversation_scores.py` for
quality-over-time plots).

- **Models** (env-overridable): topic classification uses Haiku
  (`$ANALYZE_TOPIC_MODEL`), the quality judge uses Sonnet (`$ANALYZE_QUALITY_MODEL`).
  CLI flags `--topic-model` / `--quality-model` override the env defaults.
- **LLM-as-judge** evaluates each conversation. The judge is given today's date and
  is told it cannot see raw tool output, so it must not flag real (precise, recent)
  data as fabricated. Attachments (stored only in a message's `content_json`) are
  surfaced to the judge so file-based questions aren't mistaken for fabrication.
- **Disposition** classifies each conversation's outcome: `good_answer`,
  `agent_failure`, `technical_failure`, `out_of_scope`, `unfinished`,
  `weird_or_unclear`. Only `good_answer`/`agent_failure` count toward the
  **agent-quality** metric (successful/neutral/unsuccessful). `technical_failure`
  keeps a low score but buckets separately (infra ≠ agent); out-of-scope / unfinished
  / weird requests are not penalized. This keeps the quality trend measuring only
  conversations the agent could have done well at. Conversations the judge skipped
  (no quality score) and with no user rating are labelled `unknown` rather than given
  a heuristic label, so they stay out of the quality metric.
- **Issue categorization**: the judge's detailed per-conversation issues are mapped
  onto a fixed taxonomy (`conversation_prompts.py:ISSUE_CATEGORIES`) via a cheap Haiku
  pass so the report surfaces recurring problems instead of count-1 unique strings.
- **Caching**: per-session topic + quality + derived results are persisted to the
  `conversation_analysis` / `conversation_issue` SQLite tables (with the full
  `ConversationMetrics` blob in `metrics_json`) and read back via `get_analysis_map`
  so already-analyzed sessions skip the LLM. A session is treated as cached only if
  its row's `analyzer_version` equals the module-level `ANALYZER_VERSION` (bumping
  that constant invalidates every cached analysis). `source_updated_at` is stored as
  the raw `chat_sessions.updated_at` string so staleness comparisons stay consistent.
- **Staleness-based selection**: the nightly run does minimal LLM work by asking the DB
  (`get_stale_or_missing_session_ids`) which in-range sessions actually need (re)analysis —
  ones with no row, a continued conversation (`chat_sessions.updated_at` advanced past
  `conversation_analysis.analyzed_at`), or an `analyzer_version` mismatch. Only those are
  evicted from the reconstructed topic/quality cache and sent to the LLM; unchanged,
  current-version sessions are skipped but still flow into the report (the report always
  aggregates the full in-range set, cached + freshly judged). `--start-from` / `--until`
  intersect with the stale set as an additional date filter. Because `upsert_analysis`
  writes `analyzed_at = CURRENT_TIMESTAMP`, a re-judged continued conversation is no
  longer stale on the next run (a future `updated_at` would correctly stay stale).
  After changing the judge prompt or scoring, re-run with `--refresh-quality`
  (re-judge, keep topic cache), `--no-cache` (recompute all), or `--force` (reanalyze
  every conversation from scratch — a superset of `--no-cache` for the selected range;
  `--force` wins over `--refresh-quality` since it recomputes topics too). The issue text →
  taxonomy-category map remains a small flat sidecar at `<output-dir>/.cache/issue_categories.json`.

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
4. **Streaming responses**: Chat API streams tokens via SSE for responsive UX. Multiple event types (`content`, `usage`, `image`, `error`, `done`) provide real-time feedback. Context usage tracking via `usage` events enables the frontend to show a live progress bar of context window consumption (see SSE event types section).
5. **Agentic loop**: LLM service supports multi-turn tool use with configurable iteration limit
6. **Retry on transient errors**: Anthropic API calls are retried up to 3 times with exponential backoff (1s, 2s, 4s) for transient errors. Retryability is detected two ways because of a streaming quirk: connection errors and `APIStatusError` with HTTP status 500/502/503/529, **and** by the error type carried in the body (`overloaded_error`, `api_error`, `internal_server_error`). The latter is essential — errors that arrive mid-stream (after the SSE connection returns HTTP 200) surface as a base `APIStatusError` with `status_code=200`, so status-code matching alone misses them (`anthropic_error_type()` in `llm_service.py` reads the real type from the body). If text was already streamed before the error, the user is notified with a "[Connection interrupted, retrying...]" message. When retries are exhausted, `_classify_error` (in `chat_api.py`) maps the error to a user-facing message keyed on the same body type: overload → "Claude is temporarily overloaded… please wait a moment and resend"; internal/upstream → "Claude had a temporary upstream error." Non-retryable errors (auth, bad request, rate limit) propagate immediately.
7. **Result truncation**: Large responses are truncated with warnings to prevent context overflow
8. **Downloadable results**: Tools returning tabular data include `INCLUDE_IN_RESPONSE` download links. Direct API URLs are used for genetics API tools that support TSV format; other tools (BigQuery, LD, summary stats) have their results converted to TSV and stored on disk, served via `/chat/v1/downloads/{id}`. The `_download_url` and `_download_data` hints in tool results are processed by `_process_download_hints()` in `llm_service.py` before being sent to the LLM. All download links use relative URLs (e.g., `/api/v1/...` or `/chat/v1/downloads/...`) so they work correctly regardless of deployment domain. `INCLUDE_IN_RESPONSE` is placed at the front of the result dict so it survives JSON truncation for large results. For BigQuery, trailing SQL `LIMIT` clauses are stripped and `max_rows` is set to 100,000 so the download contains the full result set even when the LLM only displays a subset. The BigQuery proxy (`genetics-results-db`) enforces `MAX_ROWS=100000` as a hard cap.
9. **Tool result persistence (resumed conversations carry the data substrate)**: The chat API is stateless per request — the frontend replays the full conversation each turn. Tool `tool_result` blocks are persisted (`chat_messages.tool_results_json`, added via the standard PRAGMA/ALTER migration) so a resumed conversation replays the actual tool outputs the model saw, not just its prose summary. `_stream_anthropic` collects `all_tool_results` across agentic-loop iterations and emits them in the `done` SSE event; the frontend stores them and, on resume, rebuilds the `assistant(tool_use) → user(tool_result)` pairing (its history builder splits each persisted assistant turn into the assistant message plus a synthetic user message of `tool_result` blocks). The already-truncated, image-base64-stripped result content is stored as-is. **Backward compatible**: conversations saved before this feature have `tool_results_json = NULL`; on resume they emit only the assistant message and `_sanitize_tool_blocks` (in `llm_service.py`) strips the now-orphaned `tool_use` blocks — exactly the prior behavior. **Marker-strip safeguard**: the `*[Using tool: …]*` annotations injected during streaming are display-only, but they are persisted into the assistant text. Before history reaches the model, `_strip_tool_use_markers` (in `llm_service.py`, run just before `_sanitize_tool_blocks`) removes them from replayed assistant content (both string and text-block forms). Without this, a long/repetitive conversation could teach the model to imitate the notation — writing `*[Using tool: X]*` as prose instead of emitting a real `tool_use` block, then fabricating the result (observed in a real session whose tool-less turns predated the persistence fix). Real `tool_use` blocks are left untouched. To offset the larger replayed payload, `_mark_history_cache_breakpoint` adds a `cache_control: ephemeral` breakpoint on the last replayed message (the 3rd of Anthropic's 4 breakpoints, alongside the system prompt and tool definitions). System-prompt guardrails (`config/defaults.py`) additionally instruct the model to treat credible-set membership as distinct from LD and to re-query authoritative tools for count/membership/lead questions rather than relying on earlier summaries.

## Future considerations

1. Add caching for genetics API responses
2. Support additional LLM providers (Google, local models)
3. Implement tool-level access control
4. Add WebSocket transport for bidirectional streaming
