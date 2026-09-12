#!/usr/bin/env sh
# Lints the Python files in the index and FAILS the commit when ruff reports anything.
#
# Deliberately blocking, unlike the doc-drift check next to it in the hook: a gate that
# only warns is one nobody reads. `git commit --no-verify` is the escape hatch.
#
# Two things this does NOT do, both on purpose:
#
#   * It lints the WORKING TREE copies of the staged paths, not the staged blobs. Doing
#     that properly means materialising the index somewhere, or stashing, and a hook
#     that stashes can lose work if it is interrupted. The gap only bites when a file is
#     partially staged, so that case is detected and reported rather than hidden.
#   * It does not lint the whole repo, so a finding in a file you did not touch never
#     blocks you. The cost is that pre-existing findings are only cleared by touching
#     the file; `scripts/lint-staged.sh --all` runs the repo-wide check.
#
# Usage:
#   scripts/lint-staged.sh         lint the staged Python files (what the hook runs)
#   scripts/lint-staged.sh --all   lint the whole repo

set -eu

RUFF_PIN=0.15.12

root=$(git rev-parse --show-toplevel) || exit 2
cd "$root"

# resolution order matters for worktrees: one under .claude/worktrees has no .venv of
# its own, so fall back to the main checkout's before reaching for the network
resolve_ruff() {
    if [ -x "$root/.venv/bin/ruff" ]; then echo "$root/.venv/bin/ruff"; return 0; fi
    common=$(git rev-parse --git-common-dir 2>/dev/null) || common=""
    if [ -n "$common" ]; then
        common=$(cd "$common" && pwd)
        main_venv=$(dirname "$common")/.venv/bin/ruff
        if [ -x "$main_venv" ]; then echo "$main_venv"; return 0; fi
    fi
    if command -v ruff >/dev/null 2>&1; then command -v ruff; return 0; fi
    if command -v uvx >/dev/null 2>&1; then echo "uvx ruff@$RUFF_PIN"; return 0; fi
    return 1
}

if ! ruff=$(resolve_ruff); then
    printf '\nlint-staged: no ruff, and no uvx to fetch one.\n\n' >&2
    printf '  Fix: uv pip install -e ".[dev]"   (or install uv)\n\n' >&2
    printf '  Refusing to pass the commit unchecked; --no-verify bypasses deliberately.\n\n' >&2
    exit 1
fi

# the fallbacks above can land on an older ruff than this repo pins — the main
# checkout's .venv, or a system install — and the same file then passes here and fails
# in CI, or the reverse. Warn rather than block: an out-of-date venv is not a reason to
# refuse the commit, but it is a reason not to trust a clean result.
have=$($ruff --version 2>/dev/null | awk '{print $2}')
if [ -n "${have:-}" ] && [ "$have" != "$RUFF_PIN" ]; then
    printf 'lint-staged: using ruff %s, but this repo pins %s — results can differ.\n' \
        "$have" "$RUFF_PIN" >&2
    printf '            refresh it with: uv pip install -e ".[dev]"\n' >&2
fi

if [ "${1:-}" = "--all" ]; then
    # word splitting is wanted: $ruff may be `uvx ruff@<pin>`
    # shellcheck disable=SC2086
    exec $ruff check .
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

git diff --cached --name-only --diff-filter=ACMR -z -- '*.py' '*.pyi' > "$tmp/staged"
[ -s "$tmp/staged" ] || exit 0

# a path that is both staged and dirty gets linted in its working-tree state, which is
# not what is about to be committed; say so rather than let the difference pass unseen
git diff --name-only -z -- '*.py' '*.pyi' > "$tmp/dirty"
tr '\0' '\n' < "$tmp/staged" | sort > "$tmp/staged.lines"
tr '\0' '\n' < "$tmp/dirty" | sort > "$tmp/dirty.lines"
if [ -s "$tmp/dirty.lines" ] && overlap=$(comm -12 "$tmp/staged.lines" "$tmp/dirty.lines") && [ -n "$overlap" ]; then
    printf '\nlint-staged: these paths have unstaged edits, so the lint below reflects the\n' >&2
    printf 'working tree rather than what is being committed:\n' >&2
    printf '%s\n' "$overlap" | sed 's/^/  /' >&2
    printf '\n' >&2
fi

# --force-exclude applies the config's `exclude` to explicitly-passed paths too, which
# is what keeps a staged .claude/worktrees/** file out of the check
# shellcheck disable=SC2086
if ! xargs -0 $ruff check --force-exclude -- < "$tmp/staged"; then
    printf '\nlint-staged: ruff rejected the staged Python files (above).\n\n' >&2
    printf '  Auto-fixable?  %s check --fix <file>\n' "$ruff" >&2
    printf '  Deliberate?    add a targeted `# noqa: <CODE>` with a reason\n' >&2
    printf '  Bypass:        git commit --no-verify\n\n' >&2
    exit 1
fi
