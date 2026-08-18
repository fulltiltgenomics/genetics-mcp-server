"""The system prompt must never name a tool the model was not given.

genetics-results-suite-4h6.69: the prompt and the tool list used to be assembled
independently and nothing checked them against each other, so the prompt documented
`launch_subagents` at length while ENABLE_SUBAGENTS defaults false and removes it from the
tool list, described `get_phenotype_report` behind another flag defaulting false, and never
mentioned `run_analysis` at all.

The scan below is deliberately NOT `defaults.tools_named_in` — that function is what
DECIDES which blocks are emitted, so using it here would make the test assert that the
gate agrees with itself. This tokenises the emitted text instead and intersects with the
tool registry, so a bug in either implementation shows up as a disagreement.
"""

import itertools
import re

import pytest

from genetics_mcp_server.config.defaults import _assemble, _Block, default_system_prompt
from genetics_mcp_server.config.settings import Settings
from genetics_mcp_server.tools.definitions import (
    BIGQUERY_TOOL_DEFINITIONS,
    SUBAGENT_TOOL_DEFINITIONS,
    TOOL_DEFINITIONS,
    get_anthropic_tools,
)

ALL_TOOL_NAMES = frozenset(
    t["name"] for t in (*TOOL_DEFINITIONS, *BIGQUERY_TOOL_DEFINITIONS, *SUBAGENT_TOOL_DEFINITIONS)
)

# every profile the chat surface can be asked for. `None` is the deployed default (no
# filtering at all); "code" is the A/B arm from genetics-results-suite-4h6.16.
PROFILES = [None, "api", "bigquery", "rag", "code"]


def tool_names_mentioned(text: str) -> set[str]:
    """Tool names appearing in prompt text, found independently of the gate."""
    tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text))
    return tokens & ALL_TOOL_NAMES


def flag_disabled_tools(*, subagents: bool) -> set[str]:
    """The disabled set the deployment flags actually produce.

    Derived from Settings rather than hard-coded, so a flag added in front of another
    tool (genetics-results-suite-4h6.56 proposes one for run_analysis) is picked up here
    without editing this file.
    """
    return Settings(enable_subagents=subagents).disabled_tools


def resolve(profile: str | None, *, subagents: bool) -> set[str]:
    return {
        t["name"]
        for t in get_anthropic_tools(
            tool_profile=profile, disabled_tools=flag_disabled_tools(subagents=subagents)
        )
    }


class TestPromptNamesOnlyAvailableTools:
    @pytest.mark.parametrize("subagents", [True, False], ids=["subagents_on", "subagents_off"])
    @pytest.mark.parametrize("profile", PROFILES, ids=[str(p) for p in PROFILES])
    def test_every_tool_named_in_the_prompt_is_in_the_tool_list(self, profile, subagents):
        available = resolve(profile, subagents=subagents)
        prompt = default_system_prompt("FinnGenie", tool_names=available)
        assert tool_names_mentioned(prompt) - available == set()

    def test_the_scan_finds_tool_names_at_all(self):
        """Guards the test above from passing because it detects nothing."""
        available = resolve(None, subagents=True)
        mentioned = tool_names_mentioned(default_system_prompt("FinnGenie", tool_names=available))
        assert len(mentioned) > 20
        assert "get_credible_sets_by_gene" in mentioned
        assert "launch_subagents" in mentioned

    def test_subagent_guidance_disappears_with_the_flag(self):
        on = default_system_prompt("FinnGenie", tool_names=resolve(None, subagents=True))
        off = default_system_prompt("FinnGenie", tool_names=resolve(None, subagents=False))
        assert "launch_subagents" in on
        assert "Subagent Orchestration" in on
        assert "launch_subagents" not in off
        assert "Subagent Orchestration" not in off
        # the skill list is only reachable through launch_subagents, and a skill name is
        # not a tool name, so it cannot be gated by the text scan alone
        assert "variant_list_analysis" in on
        assert "variant_list_analysis" not in off

    def test_run_analysis_guidance_follows_the_tool(self):
        """The gate is what makes a future run_analysis flag (4h6.56) free."""
        with_tool = default_system_prompt("FinnGenie", tool_names=resolve(None, subagents=False))
        assert "run_analysis" in with_tool
        without = default_system_prompt(
            "FinnGenie", tool_names=resolve(None, subagents=False) - {"run_analysis"}
        )
        assert "run_analysis" not in without
        assert "list_capabilities" not in without
        # and the surviving routing guidance still describes the paths that remain
        assert "Prefer the dedicated API tools over the database" in without

    def test_unfiltered_assembly_still_available(self):
        """tool_names=None keeps the pre-4h6.69 behaviour for callers with no tool list."""
        full = default_system_prompt("FinnGenie")
        assert "launch_subagents" in full
        assert "get_phenotype_report" in full
        assert "run_analysis" in full


class TestRoutingArbitrationHasOneHomePerSurface:
    """Exactly one data-routing section, worded for the surface in force."""

    @pytest.mark.parametrize("profile", PROFILES, ids=[str(p) for p in PROFILES])
    def test_no_arm_is_told_to_prefer_a_path_it_does_not_have(self, profile):
        available = resolve(profile, subagents=False)
        prompt = default_system_prompt("FinnGenie", tool_names=available)
        if "query_database" not in available:
            assert "Fall back to the database" not in prompt
        if not (available & {"get_credible_sets_by_gene", "query_database", "run_analysis"}):
            assert "## Choosing How to Get Data" not in prompt

    def test_code_arm_is_not_told_to_prefer_api_tools(self):
        prompt = default_system_prompt("FinnGenie", tool_names=resolve("code", subagents=False))
        assert "Prefer the dedicated API tools" not in prompt
        assert "run_analysis" in prompt

    def test_baseline_arm_keeps_its_api_preference(self):
        prompt = default_system_prompt("FinnGenie", tool_names=resolve(None, subagents=False))
        assert "Prefer the dedicated API tools over the database" in prompt

    def test_bigquery_arm_gets_the_database_wording(self):
        prompt = default_system_prompt("FinnGenie", tool_names=resolve("bigquery", subagents=False))
        assert "The database is the data path here" in prompt
        assert "Prefer the dedicated API tools" not in prompt


class TestDomainScienceSurvives:
    """Routing guidance is per-surface; the science is not (4h6.17).

    The MPRA / caQTL / variant-effect / open-chromatin distinction is a statement about
    what the assays measure. It is NOT in the SDK docstrings the code arm can reach
    through list_capabilities, so it has to stay in the prompt on every surface.
    """

    @pytest.mark.parametrize("profile", PROFILES, ids=[str(p) for p in PROFILES])
    def test_regulatory_readout_distinction_present_on_every_surface(self, profile):
        prompt = default_system_prompt("FinnGenie", tool_names=resolve(profile, subagents=False))
        assert "Functional / Regulatory Readouts" in prompt
        for term in ("MPRA", "caQTL", "variant effect", "open chromatin"):
            assert term in prompt
        assert "in-silico" in prompt

    @pytest.mark.parametrize("profile", PROFILES, ids=[str(p) for p in PROFILES])
    def test_grounding_and_terminology_survive_on_every_surface(self, profile):
        prompt = default_system_prompt("FinnGenie", tool_names=resolve(profile, subagents=False))
        for section in ("## Core Principles", "## Prohibited", "## Terminology"):
            assert section in prompt


class TestAssemblyMechanism:
    """The gate itself, independent of today's prompt text."""

    def test_block_naming_an_absent_tool_is_dropped(self):
        blocks = (
            _Block("plain\n"),
            _Block("call get_credible_sets_by_gene\n"),
        )
        assert _assemble({"get_credible_sets_by_gene"}, blocks=blocks) == (
            "plain\ncall get_credible_sets_by_gene\n"
        )
        assert _assemble(set(), blocks=blocks) == "plain\n"

    def test_excludes_suppresses_the_alternate_wording(self):
        blocks = (_Block("fallback\n", excludes=frozenset({"query_database"})),)
        assert _assemble(set(), blocks=blocks) == "fallback\n"
        assert _assemble({"query_database"}, blocks=blocks) == ""

    def test_requires_any_gates_text_that_names_no_tool(self):
        blocks = (_Block("sql talk\n", requires_any=frozenset({"query_database", "run_analysis"})),)
        assert _assemble({"run_analysis"}, blocks=blocks) == "sql talk\n"
        assert _assemble({"search_genes"}, blocks=blocks) == ""

    def test_a_word_boundary_prefix_is_not_a_mention(self):
        blocks = (_Block("get_credible_sets_by_gene_extra is not a tool\n"),)
        assert _assemble(set(), blocks=blocks) != ""

    def test_requires_all_needs_every_named_tool(self):
        blocks = (_Block("routed\n", requires_all=frozenset({"query_database", "run_analysis"})),)
        assert _assemble({"query_database", "run_analysis"}, blocks=blocks) == "routed\n"
        assert _assemble({"query_database"}, blocks=blocks) == ""


_ARM_ROUTING_SENTENCES = (
    "**Prefer the dedicated API tools over the database.**",
    "**The API tools are the data path here.**",
    "**The database is the data path here.**",
    "Scripts are the only data path on this surface",
)
# the three ways a surface can reach data: the per-entity API tools (get_credible_sets_by_gene
# is the sentinel the routing variants already branch on), the database tool, or a script
_DATA_PATH_TOOLS = frozenset({"get_credible_sets_by_gene", "query_database", "run_analysis"})

# tool groups a deployment flag plausibly removes as a unit — 4h6.56 proposes one in front of
# run_analysis. The name-derived families are read off the registry so a tool added to one
# joins the removal set without editing this list.
_TOOL_FAMILIES = {
    "database": {"query_database", "get_database_schema"} & ALL_TOOL_NAMES,
    "code_execution": {"run_analysis", "list_capabilities"} & ALL_TOOL_NAMES,
    "credible_sets": {n for n in ALL_TOOL_NAMES if "credible_set" in n},
    "gene_keyed": {n for n in ALL_TOOL_NAMES if n.endswith("_by_gene") or "gene_based" in n},
    "subagents": {"launch_subagents"} & ALL_TOOL_NAMES,
}


def _tool_sets_to_probe() -> dict[str, set[str]]:
    """Tool sets that mostly correspond to no profile the chat surface can be asked for.

    Parametrising over today's five profiles is exactly what missed the defect this guards:
    every one of them happens to carry all three tools the API-preference bullet cited as
    examples, so the bullet's dependence on them was invisible. Driving from the full tool
    set with single tools and plausible flag-shaped groups removed exposes it.
    """
    full = resolve(None, subagents=True)
    sets = {"full": set(full)}
    for name in sorted(full):
        sets[f"-{name}"] = full - {name}
    for name, family in _TOOL_FAMILIES.items():
        sets[f"-{name}"] = full - family
    for a, b in itertools.combinations(sorted(_TOOL_FAMILIES), 2):
        sets[f"-{a}-{b}"] = full - _TOOL_FAMILIES[a] - _TOOL_FAMILIES[b]
    sets["no_data_path"] = (
        full
        - _TOOL_FAMILIES["database"]
        - _TOOL_FAMILIES["code_execution"]
        - _TOOL_FAMILIES["credible_sets"]
        - _TOOL_FAMILIES["gene_keyed"]
    )
    return sets


class TestEverySurfaceWithADataPathIsRouted:
    """One arm-routing sentence, never zero and never two, on any tool set.

    The API-vs-database arbitration is the counterweight to the run_analysis bullet. If it
    goes missing the prompt silently becomes pro-script with nothing opposing it — the worst
    direction for a benchmark whose whole point is comparing a script arm against a tool arm,
    and one that no heading pin, no `_REQUIRED_*` string and no profile parametrisation can
    see, because the heading survives and every pinned string is elsewhere.
    """

    def test_exactly_one_arm_routing_sentence_per_surface(self):
        wrong = {}
        for label, tools in _tool_sets_to_probe().items():
            prompt = default_system_prompt("FinnGenie", tool_names=tools)
            hits = [s for s in _ARM_ROUTING_SENTENCES if s in prompt]
            if len(hits) != (1 if tools & _DATA_PATH_TOOLS else 0):
                wrong[label] = hits
        assert wrong == {}

    def test_the_probe_reaches_beyond_the_current_profiles(self):
        """Guards the test above from passing because it probes nothing interesting."""
        sets = _tool_sets_to_probe()
        assert "-get_gene_based_results" in sets
        assert "-get_exome_results_by_gene" in sets
        assert len(sets) > 50
        profiles = [frozenset(resolve(p, subagents=s)) for p in PROFILES for s in (True, False)]
        assert sum(frozenset(t) not in profiles for t in sets.values()) > 40
        # and each arm-routing variant is actually exercised somewhere in the probe
        emitted = {
            s
            for tools in sets.values()
            for s in _ARM_ROUTING_SENTENCES
            if s in default_system_prompt("FinnGenie", tool_names=tools)
        }
        assert emitted == set(_ARM_ROUTING_SENTENCES)


def _headings(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("#")]


def _heading_of_each_line(text: str) -> dict[str, set[str | None]]:
    """Every body line mapped to the heading(s) in force above it."""
    current: str | None = None
    out: dict[str, set[str | None]] = {}
    for line in text.splitlines():
        if line.startswith("#"):
            current = line
        elif line.strip():
            out.setdefault(line, set()).add(current)
    return out


class TestSectionStructureSurvivesFiltering:
    """A dropped heading must never reparent the body that followed it.

    `## Data Sources and Resource Names` was gated on `list_datasets` while its body —
    the products/`data_type` paragraph, the aggregate-counts and sample-size-provenance
    paragraphs, and the whole database section — was deliberately ungated. On the `code`
    profile the heading vanished and ~4 KB of body landed under
    `### Functional / Regulatory Readouts`, an H3 about MPRA and caQTL assays.
    """

    @pytest.mark.parametrize("profile", PROFILES, ids=[str(p) for p in PROFILES])
    @pytest.mark.parametrize("subagents", [True, False], ids=["subagents_on", "subagents_off"])
    def test_no_body_line_lands_under_a_different_heading(self, profile, subagents):
        full = _heading_of_each_line(default_system_prompt("FinnGenie"))
        filtered = _heading_of_each_line(
            default_system_prompt("FinnGenie", tool_names=resolve(profile, subagents=subagents))
        )
        reparented = {
            line: (heads, full[line])
            for line, heads in filtered.items()
            if line in full and not heads <= full[line]
        }
        assert reparented == {}

    @pytest.mark.parametrize("profile", PROFILES, ids=[str(p) for p in PROFILES])
    def test_no_heading_is_emitted_empty(self, profile):
        """A heading with no body under it is guidance that lost its content."""
        prompt = default_system_prompt("FinnGenie", tool_names=resolve(profile, subagents=False))
        lines = prompt.splitlines()
        for i, line in enumerate(lines):
            if not line.startswith("#"):
                continue
            rest = lines[i + 1 :]
            body = [x for x in rest[: next((j for j, y in enumerate(rest) if y.startswith("#")), len(rest))] if x.strip()]
            assert body, f"{profile}: heading {line!r} has no body"


# Emitted section headings, pinned per profile. A profile losing a section is a real
# decision, so it has to be made here rather than discovered later from a benchmark:
# absence-only assertions let a `requires_any` change silently delete 3.5 KB from the
# `api` prompt and 3.0 KB from `code` with every other test still green.
_SHARED_TAIL = [
    "## Response Style",
    "## Handling Uncertainty",
    "## Out of Scope and Limitations",
    "## Contextualizing Findings Against Prior Knowledge",
    "## Prohibited",
    "## Terminology",
]
_EXPECTED_HEADINGS = {
    None: [
        "## Core Principles",
        "## Analyzing data",
        "## Tool Usage Guidelines",
        "### Mouse Model Evidence (search_mgi)",
        "## Variant Annotation Sources",
        "### Functional / Regulatory Readouts",
        "### HLA / the MHC region",
        "### Protein Annotation (UniProt)",
        "## Data Sources and Resource Names",
        "### Pseudo Credible Sets",
        "## Choosing How to Get Data",
        *_SHARED_TAIL,
    ],
    "api": [
        "## Core Principles",
        "## Analyzing data",
        "## Tool Usage Guidelines",
        "### Mouse Model Evidence (search_mgi)",
        "## Variant Annotation Sources",
        "### Functional / Regulatory Readouts",
        "### HLA / the MHC region",
        "### Protein Annotation (UniProt)",
        "## Data Sources and Resource Names",
        "### Pseudo Credible Sets",
        "## Choosing How to Get Data",
        *_SHARED_TAIL,
    ],
    "bigquery": [
        "## Core Principles",
        "## Analyzing data",
        "## Tool Usage Guidelines",
        "### Mouse Model Evidence (search_mgi)",
        "### Functional / Regulatory Readouts",
        "### HLA / the MHC region",
        "### Protein Annotation (UniProt)",
        "## Data Sources and Resource Names",
        "### Pseudo Credible Sets",
        "## Choosing How to Get Data",
        *_SHARED_TAIL,
    ],
    "rag": [
        "## Core Principles",
        "## Analyzing data",
        "## Tool Usage Guidelines",
        "### Mouse Model Evidence (search_mgi)",
        "### Functional / Regulatory Readouts",
        "### Protein Annotation (UniProt)",
        "## Data Sources and Resource Names",
        "### Pseudo Credible Sets",
        *_SHARED_TAIL,
    ],
    "code": [
        "## Core Principles",
        "## Analyzing data",
        "## Tool Usage Guidelines",
        "### Functional / Regulatory Readouts",
        "### HLA / the MHC region",
        "## Data Sources and Resource Names",
        "### Pseudo Credible Sets",
        "## Choosing How to Get Data",
        *_SHARED_TAIL,
    ],
}

# Domain science and grounding rules, with the surfaces that must carry each. These were
# all deleted from at least one arm by a tool mention that was parenthetical, illustrative
# or negated rather than instructional — the gate suppresses a block for ANY unavailable
# name in it, so an aside about where a flag is visible took the whole section with it.
_REQUIRED_EVERYWHERE = [
    "Data types are case-sensitive",
    "`GWAS`, `eQTL`, `pQTL`, `sQTL`, `caQTL`, `asmQTL`",
    "not statistically fine-mapped credible sets",
    "Always tell the user explicitly when presenting pseudo credible set data",
    "r² > 0.95 to the lead",
    "PIPs from pseudo credible sets should be interpreted with more caution",
]
# these presuppose a path to credible-set / MHC rows, which `rag` does not have
_REQUIRED_WITH_A_DATA_PATH = [
    "**Membership is NOT the same as LD.**",
    "in partial LD with the lead",
    "**Re-query; do not answer from memory.**",
    "### HLA / the MHC region",
    "LD across the MHC is so extensive",
    "`pval` underflows to 0",
    "low **`info`** (imputation quality)",
]
_WITH_A_DATA_PATH = [None, "api", "bigquery", "code"]


class TestLoadBearingTextIsPresent:
    """Absence-only assertions cannot see text going missing.

    A `requires_any` mutated from any-match to all-match silently removed 3.5 KB from the
    `api` prompt and 3.0 KB from `code` while every profile-level test passed.
    """

    @pytest.mark.parametrize("profile", PROFILES, ids=[str(p) for p in PROFILES])
    def test_emitted_headings_are_pinned(self, profile):
        prompt = default_system_prompt("FinnGenie", tool_names=resolve(profile, subagents=False))
        assert _headings(prompt) == _EXPECTED_HEADINGS[profile]

    @pytest.mark.parametrize("profile", PROFILES, ids=[str(p) for p in PROFILES])
    def test_science_and_grounding_rules_survive_on_every_surface(self, profile):
        prompt = default_system_prompt("FinnGenie", tool_names=resolve(profile, subagents=False))
        for text in _REQUIRED_EVERYWHERE:
            assert text in prompt, f"{profile} lost: {text!r}"

    @pytest.mark.parametrize("profile", _WITH_A_DATA_PATH, ids=[str(p) for p in _WITH_A_DATA_PATH])
    def test_credible_set_grounding_survives_wherever_the_rows_are_reachable(self, profile):
        prompt = default_system_prompt("FinnGenie", tool_names=resolve(profile, subagents=False))
        for text in _REQUIRED_WITH_A_DATA_PATH:
            assert text in prompt, f"{profile} lost: {text!r}"

    @pytest.mark.parametrize("profile", ["api", "code"], ids=["api", "code"])
    def test_sql_surfaces_without_a_database_tool_get_a_schema_route(self, profile):
        """These have `run_analysis` (whose SDK exposes `sql()`) but neither
        `query_database` nor `get_database_schema`, so they read all the SQL guidance with
        no way to discover a column name unless the prompt gives them one."""
        available = resolve(profile, subagents=False)
        assert "query_database" not in available
        assert "get_database_schema" not in available
        prompt = default_system_prompt("FinnGenie", tool_names=available)
        assert "genetics.schema()" in prompt
        assert "genetics.sql(...)` inside a script is the only route" in prompt

    def test_database_tool_surfaces_do_not_get_the_sdk_route(self):
        prompt = default_system_prompt("FinnGenie", tool_names=resolve("bigquery", subagents=False))
        assert "genetics.schema()" not in prompt


_RUN_ANALYSIS_BULLET_START = "- **Write one script with run_analysis"
_RUN_ANALYSIS_BULLET_END = "A script is not cheaper than one call."


def _run_analysis_bullet(prompt: str) -> str:
    start = prompt.index(_RUN_ANALYSIS_BULLET_START)
    end = prompt.index(_RUN_ANALYSIS_BULLET_END, start) + len(_RUN_ANALYSIS_BULLET_END)
    return prompt[start:end]


class TestRunAnalysisWordingIsArmNeutral:
    """The A/B in genetics-results-suite-4h6.23 compares arms that differ in TOOLS.

    If the bullet describing run_analysis were worded more (or less) encouragingly on the
    `code` arm than on the baseline, the comparison would be measuring the prompt instead.
    Nothing asserted this before, so an edit could tilt the benchmark with tests green.
    """

    def test_the_bullet_is_byte_identical_across_the_arms_that_carry_it(self):
        carriers = [p for p in PROFILES if "run_analysis" in resolve(p, subagents=False)]
        assert carriers == [None, "api", "bigquery", "code"]
        bullets = {
            p: _run_analysis_bullet(
                default_system_prompt("FinnGenie", tool_names=resolve(p, subagents=False))
            )
            for p in carriers
        }
        assert len(set(bullets.values())) == 1, "run_analysis wording differs between arms"

    def test_the_bullet_is_absent_where_the_tool_is(self):
        prompt = default_system_prompt("FinnGenie", tool_names=resolve("rag", subagents=False))
        assert _RUN_ANALYSIS_BULLET_START not in prompt
