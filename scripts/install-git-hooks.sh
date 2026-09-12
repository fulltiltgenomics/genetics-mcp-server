#!/usr/bin/env sh
# Wires this checkout up to the tracked hooks in .beads/hooks.
#
# Two things have to be true for the hooks below to run on a commit, and only one of
# them survives `git clone`:
#
#   1. .beads/hooks/pre-commit exists and carries our blocks. The file is tracked, so a
#      clone gets it. Beads owns the top of that file (the "BEADS INTEGRATION" markers)
#      and patches between the markers rather than rewriting the file — measured against
#      bd 1.0.3 with `bd hooks install`, `--force`, and a forced version-marker bump — so
#      appended blocks do survive a beads upgrade. This script re-appends them anyway,
#      because that guarantee is beads' implementation detail and not ours to rely on.
#   2. core.hooksPath points at .beads/hooks. That is LOCAL git config, stored in the git
#      dir, and is not tracked by anything. A fresh clone has no hooks at all until beads
#      or this script sets it, and the resulting silence looks exactly like a clean
#      commit. core.hooksPath is shared across worktrees, so running this once in the
#      main checkout also wires every worktree, existing and future.
#
# Usage:
#   scripts/install-git-hooks.sh          repair both, print what changed
#   scripts/install-git-hooks.sh --check  report only, exit 1 if anything is missing

set -u

mode=install
case "${1:-}" in
    --check) mode=check ;;
    "") ;;
    *) echo "usage: $0 [--check]" >&2; exit 2 ;;
esac

# hooks live in the MAIN checkout even when this runs from a worktree, and
# core.hooksPath is shared across worktrees, so resolve via the common git dir
common=$(git rev-parse --git-common-dir 2>/dev/null) || {
    echo "install-git-hooks: not a git repository" >&2
    exit 2
}
common=$(cd "$common" && pwd) || exit 2
# assumes the git dir is named .git directly under the worktree root; a
# --separate-git-dir checkout or a bare-main layout would resolve this wrongly
main_root=$(dirname "$common")
hooks_dir="$main_root/.beads/hooks"
pre_commit="$hooks_dir/pre-commit"

# The blocks this script manages, in the order they are appended. Each is emitted from
# exactly one place, so the copy in .beads/hooks/pre-commit and the copy this script
# would re-append cannot drift apart — an earlier version kept the text in both and
# relied on a comment asking people to keep them byte-identical.
BLOCKS='doc-drift lint'

marker_for() {
    case "$1" in
        doc-drift) echo '# --- doc-drift warning (not managed by beads) ---' ;;
        lint)      echo '# --- lint gate (not managed by beads) ---' ;;
        *) echo "install-git-hooks: unknown block '$1'" >&2; exit 2 ;;
    esac
}

emit_block() {
    printf '\n%s\n' "$(marker_for "$1")"
    case "$1" in
        doc-drift)
            cat <<'EOF'
# never blocks: || true keeps a broken check from stopping a commit
if [ -x ./scripts/check-doc-drift.sh ]; then
  ./scripts/check-doc-drift.sh || true
fi
EOF
            ;;
        lint)
            cat <<'EOF'
# blocks, unlike the doc-drift check above: a gate that only warns is one nobody reads.
# `git commit --no-verify` is the deliberate bypass.
if [ -x ./scripts/lint-staged.sh ]; then
  ./scripts/lint-staged.sh || exit 1
fi
EOF
            ;;
    esac
}

# Exact consecutive-line match of the text on stdin inside file $1. The marker alone
# only answers "is a block there"; this answers "is it the text this script installs",
# which is the failure the previous version could not see — a hand-edited block in
# .beads/hooks/pre-commit became a second source of truth, and the next repair silently
# restored the old text over it.
contains_block() {
    awk -v want="$2" '
    BEGIN { n = 0; while ((getline line < want) > 0) { w[++n] = line } }
    { buf[NR] = $0 }
    END {
        for (i = 1; i + n - 1 <= NR; i++) {
            ok = 1
            for (j = 1; j <= n; j++) if (buf[i + j - 1] != w[j]) { ok = 0; break }
            if (ok) exit 0
        }
        exit 1
    }' "$1"
}

problems=0
messages=""
note() {
    problems=$((problems + 1))
    messages="${messages}  $1
"
}

current=$(git config --get core.hooksPath 2>/dev/null || true)
path_ok=0
[ "$current" = "$hooks_dir" ] && path_ok=1

block_present() {
    [ -f "$pre_commit" ] && grep -qF "$(marker_for "$1")" "$pre_commit"
}

if [ "$mode" = check ]; then
    [ "$path_ok" -eq 1 ] || note "core.hooksPath is '${current:-unset}', expected '$hooks_dir' — no git hooks run in this checkout"
    tmp=$(mktemp -d)
    trap 'rm -rf "$tmp"' EXIT
    for b in $BLOCKS; do
        if ! block_present "$b"; then
            note "$pre_commit is missing the '$b' block"
            continue
        fi
        emit_block "$b" > "$tmp/want"
        contains_block "$pre_commit" "$tmp/want" || \
            note "$pre_commit has a '$b' block that differs from the one this script installs — reconcile them by hand, or delete the block and re-run without --check"
    done
    if [ "$problems" -gt 0 ]; then
        printf '\ngit hooks are not wired up in this checkout:\n\n' >&2
        printf '%s' "$messages" >&2
        printf '\n  Fix: %s\n\n' "scripts/install-git-hooks.sh" >&2
        exit 1
    fi
    exit 0
fi

changed=0

if [ "$path_ok" -eq 0 ]; then
    git config core.hooksPath "$hooks_dir" || exit 2
    echo "install-git-hooks: core.hooksPath -> $hooks_dir"
    changed=1
fi

for b in $BLOCKS; do
    block_present "$b" && continue
    mkdir -p "$hooks_dir"
    if [ ! -f "$pre_commit" ]; then
        printf '#!/usr/bin/env sh\n' > "$pre_commit"
    fi
    emit_block "$b" >> "$pre_commit"
    chmod +x "$pre_commit"
    echo "install-git-hooks: re-appended the '$b' block to $pre_commit"
    changed=1
done

[ "$changed" -eq 0 ] && echo "install-git-hooks: already wired up ($hooks_dir)"
exit 0
