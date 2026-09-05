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

from genetics_mcp_server.config.defaults import (
    _SUMMARIZE_PARAM_TOOLS,
    _assemble,
    _Block,
    default_system_prompt,
)
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
# filtering at all); "code" is the A/B arm from genetics-results-suite-4h6.16 and "nocode"
# is the arm it is measured against, so both have to be exercised here.
PROFILES = [None, "api", "bigquery", "rag", "code", "nocode"]


def tool_names_mentioned(text: str) -> set[str]:
    """Tool names appearing in prompt text, found independently of the gate."""
    tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text))
    return tokens & ALL_TOOL_NAMES


def flag_disabled_tools(*, subagents: bool, sandbox: bool = True) -> set[str]:
    """The disabled set the deployment flags actually produce.

    Derived from Settings rather than hard-coded, so a flag added in front of another
    tool is picked up here without editing this file. `sandbox` defaults to True — the
    opposite of the deployed default — because everything below is about what the prompt
    says when a tool IS present; the flag-off direction is asserted explicitly instead
    (genetics-results-suite-4h6.56).
    """
    return Settings(enable_subagents=subagents, sandbox_enabled=sandbox).disabled_tools


def resolve(profile: str | None, *, subagents: bool, sandbox: bool = True) -> set[str]:
    return {
        t["name"]
        for t in get_anthropic_tools(
            tool_profile=profile,
            disabled_tools=flag_disabled_tools(subagents=subagents, sandbox=sandbox),
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

    def test_the_sandbox_flag_takes_the_guidance_with_the_tool(self):
        """4h6.56 cashed in: SANDBOX_ENABLED=false must silence the prompt too.

        Asserted through Settings and the real resolution rather than by subtracting the
        name by hand, because the whole point of the flag is that ONE value moves both the
        tool list and the prompt. A prompt that kept steering toward run_analysis while the
        tool was gone would be the worst of the three states.
        """
        off = default_system_prompt("FinnGenie", tool_names=resolve(None, subagents=False, sandbox=False))
        assert "run_analysis" not in off
        assert "Choosing How to Get Data" in off, "the other paths still need routing"
        # the default profile is not the only surface: the code arm resolves to nothing it
        # can steer toward either
        code_off = resolve("code", subagents=False, sandbox=False)
        assert "run_analysis" not in code_off
        assert "run_analysis" not in default_system_prompt("FinnGenie", tool_names=code_off)

    def test_run_analysis_guidance_follows_the_tool(self):
        """The gate is what makes the run_analysis flag (4h6.56) free."""
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
        "### Drug and Target Evidence (ChEMBL)",
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
        "### Drug and Target Evidence (ChEMBL)",
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
        "### Drug and Target Evidence (ChEMBL)",
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
        "### Drug and Target Evidence (ChEMBL)",
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
    # the `code` arm's comparator: everything except the code trio, so it currently emits
    # the same sections as the unfiltered prompt. Pinned separately anyway — the point of
    # this dict is that a section disappearing from one arm is a decision, and the two
    # lists coinciding today is a fact about the tool split, not something to rely on.
    "nocode": [
        "## Core Principles",
        "## Analyzing data",
        "## Tool Usage Guidelines",
        "### Mouse Model Evidence (search_mgi)",
        "## Variant Annotation Sources",
        "### Functional / Regulatory Readouts",
        "### HLA / the MHC region",
        "### Protein Annotation (UniProt)",
        "### Drug and Target Evidence (ChEMBL)",
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
# derived from tool resolution, not prompt text, so a profile gaining or losing a data-path
# tool is picked up here without editing this list by hand — that hand-maintenance is what
# let `nocode` (routed through both get_credible_sets_by_gene and query_database) go unwritten
_WITH_A_DATA_PATH = [p for p in PROFILES if _DATA_PATH_TOOLS & resolve(p, subagents=False)]


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


def _tools_with_parameter(param: str) -> set[str]:
    """Tool names whose schema actually declares `param`, read from the definitions.

    Derived rather than listed so that a parameter added to (or dropped from) a tool moves
    this set, and the gate keyed on it is checked against reality instead of a copy.
    """
    return {
        t["name"]
        for t in get_anthropic_tools()
        if param in t.get("input_schema", {}).get("properties", {})
    }


_TRUNCATION_RULE = "is a PREFIX of an ordered result"
_SUMMARIZE_REMEDY = "or with `summarize=true` until the result is complete"
_GENERIC_REMEDY = "Narrow the request until the result is complete."
_DB_COUNT_REMEDY = "Query the database for the count directly"
_SDK_COUNT_REMEDY = "Count the rows in a script with `genetics.sql(...)`"
_PRODUCTS_IMPERATIVE = "always check each dataset's `products` field"
_PRODUCTS_KNOWLEDGE = "but its `products` field determines what you can actually *query*"
_SDK_CATALOG_ROUTE = "`genetics.datasets(resource=..., include_stats=True)` inside a script"


class TestGuidanceKeyedOnAParameterOrAFieldIsGated:
    """genetics-results-suite-4h6.75.

    The gate matches TOOL NAMES, so a block whose actionability rests on a PARAMETER or an
    OUTPUT FIELD names no tool and was emitted unconditionally — telling `code` to pass
    `summarize=true` to tools it does not have, and `rag` to query a database it cannot
    reach. Everything below reads the RENDERED PROMPT per profile: asserting on `_Block`
    metadata would only restate the constant that was changed.
    """

    def test_the_pinned_summarize_tool_set_still_matches_the_real_schemas(self):
        assert _SUMMARIZE_PARAM_TOOLS == _tools_with_parameter("summarize")

    @pytest.mark.parametrize("profile", PROFILES, ids=[str(p) for p in PROFILES])
    def test_the_summarize_remedy_reaches_exactly_the_surfaces_with_the_parameter(self, profile):
        available = resolve(profile, subagents=False)
        prompt = default_system_prompt("FinnGenie", tool_names=available)
        reachable = bool(_tools_with_parameter("summarize") & available)
        assert (_SUMMARIZE_REMEDY in prompt) is reachable
        # and the surfaces without it are still told to narrow, rather than left with the
        # prohibition and no way out
        assert (_GENERIC_REMEDY in prompt) is not reachable
        assert _TRUNCATION_RULE in prompt

    @pytest.mark.parametrize("profile", PROFILES, ids=[str(p) for p in PROFILES])
    def test_the_count_remedy_names_the_database_route_the_surface_actually_has(self, profile):
        available = resolve(profile, subagents=False)
        prompt = default_system_prompt("FinnGenie", tool_names=available)
        assert (_DB_COUNT_REMEDY in prompt) is ("query_database" in available)
        assert (_SDK_COUNT_REMEDY in prompt) is (
            "run_analysis" in available and "query_database" not in available
        )

    def test_rag_is_told_to_count_by_neither_route_because_it_has_neither(self):
        available = resolve("rag", subagents=False)
        assert not {"query_database", "run_analysis"} & available
        prompt = default_system_prompt("FinnGenie", tool_names=available)
        assert _TRUNCATION_RULE in prompt
        assert _DB_COUNT_REMEDY not in prompt
        assert _SDK_COUNT_REMEDY not in prompt
        assert "query the database" not in prompt

    @pytest.mark.parametrize("sandbox", [True, False], ids=["sandbox_on", "sandbox_off"])
    @pytest.mark.parametrize("profile", PROFILES, ids=[str(p) for p in PROFILES])
    def test_the_products_imperative_reaches_only_surfaces_that_can_read_the_field(
        self, profile, sandbox
    ):
        """Two routes read `products`, not one: `list_datasets`, and the SDK's
        `genetics.datasets()` — which delegates to the same executor method and the same
        `/v1/datasets` response, whose per-dataset payload carries `products`. Gating on
        `list_datasets` alone dropped the guidance from the sandboxed arm."""
        available = resolve(profile, subagents=False, sandbox=sandbox)
        prompt = default_system_prompt("FinnGenie", tool_names=available)
        reachable = bool({"list_datasets", "run_analysis"} & available)
        assert (_PRODUCTS_IMPERATIVE in prompt) is reachable
        assert ("always mention which products each dataset supports" in prompt) is reachable
        # and a surface reaching the catalog only through the SDK is told which call that is
        assert (_SDK_CATALOG_ROUTE in prompt) is (
            "run_analysis" in available and "list_datasets" not in available
        )
        # the products-vs-data_type distinction is knowledge about the data and must not
        # have gone with the imperative
        assert _PRODUCTS_KNOWLEDGE in prompt


_ANNOTATION_PROHIBITION = "you must NEVER query the database for them"
_ANNOTATION_TOOL_ROUTE = "Those per-variant annotations come from `get_variant_annotations`"
# the SDK remedy has two wordings: one for a surface that ALSO has the protein-effect tool
# and one for a surface that has only the SDK. The second's "clinical significance ... is
# not in the database either" is true only where nothing returns it.
_ANNOTATION_SDK_AND_PROTEIN = "Fetch consequence, allele frequency and gene in a script instead"
_ANNOTATION_SDK_ONLY = "Fetch them in a script instead: `genetics.variant_annotation("
_ANNOTATION_DB_PROTEIN_ROUTE = "The database is not an alternative route to them. For a coding SNV"
_ANNOTATION_NO_ROUTE = "there is no variant-annotation tool on this surface"
_ANNOTATION_ROUTES = (
    _ANNOTATION_TOOL_ROUTE,
    _ANNOTATION_SDK_AND_PROTEIN,
    _ANNOTATION_SDK_ONLY,
    _ANNOTATION_DB_PROTEIN_ROUTE,
    _ANNOTATION_NO_ROUTE,
)

# A refusal sentence paired with the LOCAL tool whose presence in the SAME rendered prompt
# would make it false. This is the defect class of genetics-results-suite-4h6.76's second
# half: a prompt that tells the model to say something is unavailable while itself
# documenting the tool that returns it. Nothing pinned it before — the no-route variant
# shipped on `bigquery`, which has `get_variant_protein_effect`.
_REFUSAL_CONTRADICTED_BY = {
    _ANNOTATION_NO_ROUTE: "get_variant_protein_effect",
    "What it does not cover — clinical significance, pathogenicity scores, multi-population"
    " frequencies — is not in the database either": "get_variant_protein_effect",
}


class TestTheAnnotationProhibitionAlwaysCarriesARoute:
    """genetics-results-suite-4h6.76.

    The prohibition ("NEVER query the database for consequence / AF / rsID") is gated on
    `query_database` or `run_analysis`; its remedy NAMES the two annotation tools, so the
    text gate dropped the remedy on every surface without them — measured on `bigquery` and
    on `code` — leaving a dead end. Each surface must now get the route it has, or be told
    plainly that it has none.
    """

    @pytest.mark.parametrize("sandbox", [True, False], ids=["sandbox_on", "sandbox_off"])
    @pytest.mark.parametrize("profile", PROFILES, ids=[str(p) for p in PROFILES])
    def test_no_surface_gets_the_prohibition_without_a_route(self, profile, sandbox):
        available = resolve(profile, subagents=False, sandbox=sandbox)
        prompt = default_system_prompt("FinnGenie", tool_names=available)
        emitted = sum(r in prompt for r in _ANNOTATION_ROUTES)
        if _ANNOTATION_PROHIBITION not in prompt:
            assert emitted == 0, f"{profile}/{sandbox}: a route with no prohibition to answer"
            return
        assert emitted == 1, f"{profile}/{sandbox}: not exactly one route"

    @pytest.mark.parametrize("sandbox", [True, False], ids=["sandbox_on", "sandbox_off"])
    @pytest.mark.parametrize("profile", PROFILES, ids=[str(p) for p in PROFILES])
    def test_no_prompt_refuses_what_the_same_prompt_explains_how_to_get(self, profile, sandbox):
        """The defect class, pinned directly: a refusal sentence and the tool that
        contradicts it must never appear in the same rendered prompt."""
        available = resolve(profile, subagents=False, sandbox=sandbox)
        prompt = default_system_prompt("FinnGenie", tool_names=available)
        for refusal, tool in _REFUSAL_CONTRADICTED_BY.items():
            assert not (refusal in prompt and tool in prompt), (
                f"{profile}/{sandbox}: prompt refuses what {tool} returns"
            )

    @pytest.mark.parametrize("profile", [None, "api"], ids=["None", "api"])
    def test_surfaces_with_the_annotation_tools_are_pointed_at_them(self, profile):
        available = resolve(profile, subagents=False)
        assert "get_variant_annotations" in available
        prompt = default_system_prompt("FinnGenie", tool_names=available)
        assert _ANNOTATION_TOOL_ROUTE in prompt

    def test_the_sandbox_surface_with_the_protein_tool_gets_both_halves(self):
        """`bigquery` + sandbox: the SDK covers consequence/AF/gene, and
        `get_variant_protein_effect` covers a coding SNV's ClinVar significance and rsID.
        The earlier wording asserted clinical significance was unavailable here."""
        available = resolve("bigquery", subagents=False)
        assert not {"get_variant_annotations", "get_myvariant_annotations"} & available
        assert {"run_analysis", "get_variant_protein_effect"} <= available
        prompt = default_system_prompt("FinnGenie", tool_names=available)
        assert _ANNOTATION_PROHIBITION in prompt
        assert _ANNOTATION_SDK_AND_PROTEIN in prompt
        assert "get_variant_protein_effect` adds the amino-acid change" in prompt
        assert _ANNOTATION_SDK_ONLY not in prompt
        assert _ANNOTATION_TOOL_ROUTE not in prompt

    def test_the_sandbox_surface_without_the_protein_tool_gets_the_sdk_alone(self):
        """`code`: seven tools, no annotation tool of any kind, so the SDK is the whole
        route and "clinical significance is not available" is true as written."""
        available = resolve("code", subagents=False)
        assert not {
            "get_variant_annotations",
            "get_myvariant_annotations",
            "get_variant_protein_effect",
        } & available
        assert "run_analysis" in available
        prompt = default_system_prompt("FinnGenie", tool_names=available)
        assert _ANNOTATION_PROHIBITION in prompt
        assert _ANNOTATION_SDK_ONLY in prompt
        assert _ANNOTATION_SDK_AND_PROTEIN not in prompt

    def test_the_database_only_surface_is_pointed_at_the_protein_tool_it_has(self):
        """`bigquery` with the sandbox flag off — what chat-backend.yaml declares today.
        `query_database` keeps the prohibition alive while `run_analysis` is gone, but
        `get_variant_protein_effect` is still there, so telling the model to say a coding
        SNV's consequence is unavailable would be false on the deployed surface."""
        available = resolve("bigquery", subagents=False, sandbox=False)
        assert "run_analysis" not in available
        assert {"query_database", "get_variant_protein_effect"} <= available
        prompt = default_system_prompt("FinnGenie", tool_names=available)
        assert _ANNOTATION_PROHIBITION in prompt
        assert _ANNOTATION_DB_PROTEIN_ROUTE in prompt
        assert _ANNOTATION_NO_ROUTE not in prompt
        assert _ANNOTATION_SDK_ONLY not in prompt

    def test_a_surface_with_no_route_at_all_is_told_so(self):
        """The blanket refusal is not dead: it is what a database-only surface WITHOUT
        the protein-effect tool gets. No shipped profile is that shape today, so this
        drives the assembly directly rather than through a profile."""
        available = resolve("bigquery", subagents=False, sandbox=False) - {
            "get_variant_protein_effect"
        }
        prompt = default_system_prompt("FinnGenie", tool_names=available)
        assert _ANNOTATION_PROHIBITION in prompt
        assert _ANNOTATION_NO_ROUTE in prompt
        assert _ANNOTATION_DB_PROTEIN_ROUTE not in prompt
        assert _ANNOTATION_SDK_ONLY not in prompt
