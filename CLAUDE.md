====


QUALITY CODING RULES


# Code changes

1. If you find errors or suggestions in code which are not DIRECTLY related to user's current request, never change it without asking first.
2. Before suggesting changes to files, always assume user might have changed the file since your last read and consider reading the file again.


# Security

1. Never commit sensitive files
2. Use environment variables for API keys and credentials
3. Keep API keys and credentials out of logs and output


# Project Specifications

1. Project documentation is maintained in files in `docs/` folder.
2. `docs/project-spec.md` is an overview of project purpose, structure and logic.
3. Create other files under `docs/` if necessary.
4. Maintain `docs/project-spec.md` and any other generated files to be up to date with the project.
5. Reread `docs/project-spec.md` often and whenever you need to refresh your context with what the project is about and implementation logic.
6. This should often be your first step in understanding a task.


# Documentation ownership

Changing a path on the left makes the doc on the right wrong until it is updated in
the same commit. `scripts/check-doc-drift.sh` warns (never blocks) on commits that
violate this; it runs from the `pre-commit` hook.

| changed path | doc to update | what to check |
|---|---|---|
| `src/genetics_mcp_server/tools/definitions.py` | `docs/project-spec.md` | the "Available tools" tables, tool profile categories |
| `src/genetics_mcp_server/tools/executor.py` | `docs/project-spec.md`, `docs/variant-list-analysis.md` | tool execution, truncation and download handling; the variant-list doc when `analyze_variant_list` changes |
| `src/genetics_mcp_server/routers/**` | `docs/project-spec.md` | admin endpoint list, API token endpoints, admin access rules |
| `src/genetics_mcp_server/config/**` | `docs/project-spec.md`, `.env.example` | env-var tables, default prompts; every new variable in `.env.example` |
| `src/genetics_mcp_server/auth/**`, `mcp_server.py`, `mcp_proxy.py` | `docs/project-spec.md`, `README.md` | bearer auth branches, `MCP_API_KEY` being mandatory on remote transports, `MCP_ALLOW_QUERY_TOKEN` gating, external MCP proxying |
| `src/genetics_mcp_server/scripts/analyze_variants.py` | `docs/variant-list-analysis.md` | CLI flags, input format, output JSON shape |

A doc is stale the moment it *enumerates* something the code no longer matches.
Counts and lists rot silently — tool tables, endpoint lists, env-var tables — so
re-derive them from the code rather than trusting them.


# Cross-repo documentation

`genetics-results-suite` is the spec of record for the suite as a whole; this repo
documents only itself. Adding or changing an **MCP tool** here therefore also
requires updating that repo's `docs/project-spec.md`, not just the docs here.


# Software Development Behavior Guidelines

1. Don't guess and do things which you are not certain about. Ask the user instead.
2. Don't add or modify code unrelated to the specific request and context at the moment.
3. In interactive mode: only use git when asked, stage changes and propose a commit message for user review. In autonomous/orchestrator mode (e.g. ralph wiggum): commit after each completed task with a descriptive message.
4. **Always** prior to finishing a task and considering it completed, revise all the changes and update Project Specification files.
5. **Always** prior to finishing a task and considering it completed, make sure all tests run successfully.
6. When trying to fix any bug or compiler error **ALWAYS** think carefully and analyze in detail what happened and WHY? Explain and confirm with user.


# Code Conventions

1. Project structure contains `docs/`, `src/` and `tests/` folders at the root
2. Code should be self-descriptive
   - Only add comments for tricky or complex parts of the code (explaining WHY something is done)
   - NO redundant and trivial comments that simply restate what the code does
3. This project uses async/await throughout for I/O operations
   - All HTTP calls use httpx.AsyncClient
   - Tool executor methods are async
   - MCP and LLM service handlers are async
4. Private fields and methods should be prefixed with underscore
5. Code should pass linting at all times (`ruff check src/`)


# Project-specific conventions

1. Tool definitions live in `src/genetics_mcp_server/tools/definitions.py`
   - Both MCP server and LLM service use these definitions
   - Keep them in sync when adding or modifying tools
2. Tool execution logic lives in `src/genetics_mcp_server/tools/executor.py`
   - Each tool is an async method on the ToolExecutor class
   - Return `{"success": True, ...}` on success, `{"success": False, "error": "..."}` on failure
3. Configuration uses environment variables loaded via python-dotenv
   - All settings defined in `src/genetics_mcp_server/config/settings.py`
   - Document new variables in `.env.example`
4. Tests use pytest with pytest-asyncio
   - Run with `pytest` or `pytest --cov` for coverage
   - **In a git worktree, run `uv sync --extra dev` first.** Without it `uv run pytest` falls
     through to the pyenv shim, whose interpreter has the MAIN checkout installed editable,
     and the worktree's tests then exercise the main checkout's source and report green
     (genetics-results-suite-6o3). `tests/conftest.py` now aborts the run in
     `pytest_configure` when `genetics_mcp_server` resolves outside the pytest rootdir, so
     this fails at startup with the fix instructions instead of passing silently.


====

**Don't forget any of the 'QUALITY CODING RULES' above!!!**
