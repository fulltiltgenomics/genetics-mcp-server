#!/usr/bin/env sh
# Warns when a commit changes something the docs describe but leaves the doc
# untouched. Mappings mirror the "Documentation ownership" table in CLAUDE.md.
#
# This never blocks. A warning that is occasionally ignored beats a gate that
# gets bypassed with --no-verify, because a bypassed gate is both absent and
# assumed present.

set -u

root=$(git rev-parse --show-toplevel 2>/dev/null) || exit 0
cd "$root" || exit 0

staged=$(git diff --cached --name-only --diff-filter=ACMRD)
[ -n "$staged" ] || exit 0

hit() {
    printf '%s\n' "$staged" | grep -qE "$1"
}

found=0
check() {
    if hit "$1" && ! hit "$2"; then
        if [ "$found" -eq 0 ]; then
            printf '\ndoc-drift warning — this commit changes code the docs describe:\n\n' >&2
            found=1
        fi
        printf '  %s\n' "$3" >&2
    fi
}

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

if [ "$found" -eq 1 ]; then
    printf '\n  Update the doc in this commit, or note why it does not apply.\n' >&2
    printf '  Not blocking. Mappings live in CLAUDE.md > Documentation ownership.\n\n' >&2
fi

exit 0
