"""The system-prompt block type, shared by every prompt variant.

It lives apart from `defaults` so a variant module can import it without importing the
module that holds the registry those variants are registered in — the cycle that would
otherwise force the registration to happen at the bottom of a file, where it breaks
depending on which module the process imports first.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class _Block:
    r"""One fragment of the system prompt, with the conditions for emitting it.

    `text` owns its surrounding newlines so that concatenating the emitted blocks
    reproduces the document structure with no re-joining.

    Three rules for writing the text, none of which the gate can enforce for you:

    1. NAME TOOLS EXACTLY. The gate matches `(?<!\w)NAME\b` (see `tools_named_in`), so a
       plural or suffixed mention — `get_hla_by_alleles` where the tool is
       `get_hla_by_allele` — is not seen as a mention at all, and the block is then
       emitted on surfaces that do not have the tool. No test catches that:
       tests/test_system_prompt.py's scan is independent of the gate on the ALGORITHM
       (tokenise-then-intersect vs per-name regex) but shares its NORMALISATION, so the
       two agree with each other while both being wrong. There is no live instance today
       (genetics-results-suite-4h6.78); keep it that way by naming tools verbatim and
       rephrasing the sentence around the exact name.

    2. A PROHIBITION GATES POSITIVELY. `_assemble` asks only WHICH names appear in the
       text, never with what polarity, so a block written to warn AGAINST a tool requires
       that tool to be available. Remove the tool from a surface and the warning
       disappears — along with everything else that block carries. The
       genetics-results-suite-4h6.17/.69 cycle fixed the live instance (the HLA block,
       where `get_summary_stats` appeared only inside a negation); the property remains.
       If a rule has to outlive the tool it warns about, put the warning in its own block.

    3. GATE ON WHAT A RULE NEEDS, NOT ON WHAT IT IS ABOUT. Science and grounding belong in
       blocks that name no tool; only the "which tool" clause is gated. The line between
       the two is whether the rule can be OBEYED without rows:
       - An obligation that attaches when the model PRESENTS data holds on every surface,
         because a surface with no data path can still present data it retrieved from a
         document. So the pseudo-credible-set labelling duty ("not statistically
         fine-mapped", "always tell the user explicitly") and the construction facts
         needed to read such a result (the r² membership criteria, the PIP caution) are
         ungated, and reach `rag`.
       - A rule that can only be carried out by FETCHING rows — membership is whatever
         `credible_sets_v` returns, re-query rather than answer from memory — is gated on
         having a path to those rows: on a surface without one it names an action the
         model cannot take, and "verify it" with nothing to verify against is worse than
         silence (genetics-results-suite-4h6.79).
    """

    text: str
    # emitted only if at least one of these is available; use for a section whose own text
    # names no tool but which presupposes a capability (e.g. SQL guidance, reachable either
    # through query_database or through the SDK's sql() inside run_analysis)
    requires_any: frozenset[str] = field(default_factory=frozenset)
    # suppressed if any of these is available; use to pick between mutually exclusive
    # wordings of the same guidance for different tool surfaces
    excludes: frozenset[str] = field(default_factory=frozenset)
    # emitted only if ALL of these are available. The text-derived name gate is itself an
    # implicit requires_all, so guidance whose emission is a real precondition on a tool
    # used to be expressed by happening to name that tool — which made it hostage to every
    # OTHER name in the same text, including illustrative "e.g." asides. State the
    # precondition here and keep the asides in their own blocks instead.
    requires_all: frozenset[str] = field(default_factory=frozenset)


def _fs(*names: str) -> frozenset[str]:
    return frozenset(names)
