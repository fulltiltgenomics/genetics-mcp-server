# shellcheck shell=sh
# The mechanism behind every repo's scripts/check-doc-drift.sh: which paths are staged, the
# `check` rule and the closing note. Sourced, never run. Each repo's check-doc-drift.sh
# holds only its own rules, because those mirror that repo's own CLAUDE.md table.
#
# One file for every repo of the suite: genetics-results-suite's copy is the source and
# scripts/vendor.py writes the others, so edit it there.
#
# This never blocks. A warning that is occasionally ignored beats a gate that
# gets bypassed with --no-verify, because a bypassed gate is both absent and
# assumed present.

set -u

root=$(git rev-parse --show-toplevel 2>/dev/null) || exit 0
cd "$root" || exit 0

# only the *staged* tree is examined, because that is what the pre-commit hook is about
# to turn into a commit. Run by hand with nothing staged this checks nothing, so say so
# rather than exiting silently — silence here reads as a pass.
staged=$(git diff --cached --name-only --diff-filter=ACMRD)
if [ -z "$staged" ]; then
    printf 'doc-drift: nothing staged, nothing checked.\n' >&2
    exit 0
fi

hit() {
    printf '%s\n' "$staged" | grep -qE "$1"
}

found=0
# check <code paths> <doc paths> <message> [paths to ignore inside <code paths>]
#
# The 4th argument exists because a rule whose path pattern is broader than the doc
# concern it names fires where it can never apply, and a warning that fires when it
# cannot apply is how a warn-only check becomes wallpaper.
# Use it only where the named doc demonstrably does not describe the excluded files, and
# say which property of them makes that true — "generated" is not by itself such a
# property, since a doc can and does reason about generated content.
#
# It is matched case-insensitively, and only it: an exclusion is a list of file
# extensions, and `logo.PNG` is the same kind of file as `logo.png`, so a
# case-sensitive exclusion warns about the one and not the other. The inclusion
# pattern stays case-sensitive, so a file reaches this filter only after matching
# its rule literally — widening the exclusion can therefore only drop warnings the
# rule's own doc concern does not cover.
check() {
    matched=$(printf '%s\n' "$staged" | grep -E "$1")
    [ -n "$matched" ] || return 0
    if [ -n "${4:-}" ]; then
        matched=$(printf '%s\n' "$matched" | grep -viE "$4")
        [ -n "$matched" ] || return 0
    fi
    if ! hit "$2"; then
        if [ "$found" -eq 0 ]; then
            printf '\ndoc-drift warning — this commit changes code the docs describe:\n\n' >&2
            found=1
        fi
        printf '  %s\n' "$3" >&2
    fi
}

# call last, after every rule
doc_drift_finish() {
    if [ "$found" -eq 1 ]; then
        printf '\n  Update the doc in this commit, or note why it does not apply.\n' >&2
        printf '  Not blocking. Mappings live in CLAUDE.md > Documentation ownership.\n\n' >&2
    fi
    exit 0
}
