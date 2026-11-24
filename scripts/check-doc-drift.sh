#!/usr/bin/env sh
# Warns when a commit changes something the docs describe but leaves the doc
# untouched. Mappings mirror the "Documentation ownership" table in CLAUDE.md.
#
# The rules are this repo's own; `check` and the rest of the mechanism live in
# scripts/doc-drift-lib.sh, which is the same file in every repo of the suite.

. "$(dirname "$0")/doc-drift-lib.sh"

DOCS_SPEC='^(docs/project-spec\.md|README\.md)$'

check '^src/genetics_mcp_server/tools/definitions\.py$' \
    '^docs/project-spec\.md$' \
    'tools/definitions.py -> docs/project-spec.md (the Available tools tables, Tool Profiles categories)'

check '^src/genetics_mcp_server/tools/executor\.py$' \
    '^docs/(project-spec|variant-list-analysis)\.md$' \
    'tools/executor.py -> docs/project-spec.md (tool execution, truncation, downloads); docs/variant-list-analysis.md when analyze_variant_list changes'

check '^src/genetics_mcp_server/tools/orchestration\.py$' \
    '^docs/project-spec\.md$' \
    'tools/orchestration.py -> docs/project-spec.md (the code-execution tool layer: the run_analysis gateway and its identity refusal, artifact scoping, literature and web search) — and the suite repo owns what the sandbox image ships'

check '^src/genetics_mcp_server/routers/' '^docs/project-spec\.md$' \
    'routers/ -> docs/project-spec.md (admin endpoint list, API token endpoints, admin access rules)'

check '^src/genetics_mcp_server/skills/' '^docs/project-spec\.md$' \
    'skills/ -> docs/project-spec.md (which subagent skills exist, the core tool list, which skills get run_analysis)'

# the owning doc lives in genetics-results-suite, so no staged file here can satisfy this; it
# fires as a reminder, and the suite's gen-doc-blocks.py --check is the gate that bites
check '^src/genetics_mcp_server/(mcp_server|llm_service|subagent)\.py$' '^$' \
    'mcp_server.py, llm_service.py, subagent.py -> genetics-results-suite/docs/chat-tool-reference.md (the three exclusion layers, resolved per-surface tool counts) - update it in the suite repo'

check '^src/genetics_mcp_server/config/' '^(docs/project-spec\.md|\.env\.example)$' \
    'config/ -> docs/project-spec.md (env var tables) + .env.example (every new variable)'

check '^src/genetics_mcp_server/(auth/|mcp_server\.py$|mcp_proxy\.py$)' "$DOCS_SPEC" \
    'auth/, mcp_server.py, mcp_proxy.py -> docs/project-spec.md + README.md (bearer auth branches, MCP_API_KEY being mandatory, MCP_ALLOW_QUERY_TOKEN, external MCP proxying)'

check '^src/genetics_mcp_server/scripts/analyze_variants\.py$' \
    '^docs/variant-list-analysis\.md$' \
    'scripts/analyze_variants.py -> docs/variant-list-analysis.md (CLI flags, input format, output JSON shape)'

check '^src/genetics_mcp_server/scripts/(replay_benchmark|pairwise_judge)\.py$' \
    '^docs/project-spec\.md$' \
    'scripts/replay_benchmark.py, pairwise_judge.py -> docs/project-spec.md (Replay Benchmark + Paired Quality Judging: what the report enumerates, MIN_DECISIVE_PAIRS=6, the 12,000-char elision limit, the sha256 seed, the test-file table row)'

check '^(scripts/(lint-staged|install-git-hooks)\.sh|pyproject\.toml)$' "$DOCS_SPEC" \
    'lint gate (scripts/lint-staged.sh, install-git-hooks.sh, pyproject.toml) -> README.md + docs/project-spec.md (which commits it blocks, the ruff rule set and pinned version, how ruff is resolved in a worktree)'

doc_drift_finish
