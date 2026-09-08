"""Skill definitions for subagents.

Each skill defines a specialized subagent configuration: which tools it can use,
its system prompt (loaded from a markdown file), and execution constraints.
"""

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

INSTRUCTIONS_DIR = Path(__file__).parent / "instructions"

# Every skill names the tools it gets. Deriving them from a tool `category` or a tool_profile
# made a subagent's surface move whenever either was retuned for the main agent, which is the
# wrong coupling: the profiles shape what a user's chat advertises, while a skill's list
# bounds what a task-driven agent with no session of its own can reach.

# lookup, search and metadata tools every skill gets
_CORE_TOOLS = frozenset({
    "get_dataset_display_names",
    "get_drug_profile",
    "get_drug_targets_for_gene",
    "get_gene_group_members",
    "get_protein_annotations",
    "get_resource_metadata",
    "get_target_bioactivity",
    "get_variant_protein_effect",
    "list_datasets",
    "lookup_phenotype_names",
    "lookup_variants_by_rsid",
    "map_protein_variants",
    "normalize_gene_symbols",
    "search_cbioportal",
    "search_genes",
    "search_mgi",
    "search_phenotypes",
    "search_scientific_literature",
    "search_uniprot",
    "web_search",
})

# the results-API data tools
_GENETICS_API_TOOLS = frozenset({
    "analyze_variant_list",
    "get_asm_qtl_by_gene",
    "get_asm_qtl_by_variant",
    "get_colocalization",
    "get_colocalization_by_credible_set",
    "get_credible_set_by_id",
    "get_credible_set_leads_by_phenotype",
    "get_credible_sets_by_gene",
    "get_credible_sets_by_phenotype",
    "get_credible_sets_by_qtl_gene",
    "get_credible_sets_by_region",
    "get_credible_sets_by_variant",
    "get_credible_sets_stats",
    "get_exome_results_by_gene",
    "get_exome_results_by_phenotype",
    "get_exome_results_by_region",
    "get_exome_results_by_variant",
    "get_gene_based_results",
    "get_gene_based_results_by_phenotype",
    "get_gene_disease_associations",
    "get_gene_expression",
    "get_gene_to_peaks",
    "get_genes_in_region",
    "get_hla_by_allele",
    "get_hla_by_phenotype",
    "get_ld_between_variants",
    "get_mpra_by_gene",
    "get_mpra_by_region",
    "get_mpra_by_variant",
    "get_mpra_pip_concordance_by_gene",
    "get_myvariant_annotations",
    "get_nearest_genes",
    "get_open_chromatin_by_gene",
    "get_open_chromatin_by_peak",
    "get_open_chromatin_by_region",
    "get_open_chromatin_by_variant",
    "get_peak_to_genes",
    "get_phenotype_report",
    "get_summary_stats",
    "get_summary_stats_by_region",
    "get_variant_annotations",
    "get_variant_effect_by_gene",
    "get_variant_effect_by_variant",
    "get_variants_in_ld",
})

# direct SQL against the genetics database
_DATABASE_TOOLS = frozenset({
    "get_database_schema",
    "query_database",
})


@dataclass
class SkillDefinition:
    """A skill that can be assigned to a subagent."""

    name: str
    description: str
    instruction_file: str
    tools: frozenset[str]
    model: str | None = None
    # covers thinking as well as the report text, so leave room for both
    max_tokens: int = 8192
    max_iterations: int = 10
    allow_file_read: bool = False
    allowed_paths: list[str] = field(default_factory=list)
    include_external: bool = False


SKILL_REGISTRY: dict[str, SkillDefinition] = {
    "genetics_data_extraction": SkillDefinition(
        name="genetics_data_extraction",
        description=(
            "Extract genetics data for genes, variants, or phenotypes using the API tools. "
            "Use for GWAS associations, credible sets, QTL data, gene expression, "
            "colocalization, LD, and exome/burden test results."
        ),
        instruction_file="genetics_data_extraction.md",
        tools=_CORE_TOOLS | _GENETICS_API_TOOLS,
        include_external=True,
    ),
    "literature_review": SkillDefinition(
        name="literature_review",
        description=(
            "Search scientific literature and the web for information about genes, "
            "variants, phenotypes, or biological mechanisms. Returns summaries of "
            "relevant papers and web sources."
        ),
        instruction_file="literature_review.md",
        tools=_CORE_TOOLS,
    ),
    "database_analysis": SkillDefinition(
        name="database_analysis",
        description=(
            "Run complex SQL queries against the genetics database. "
            "Use for cross-dataset comparisons, aggregations, or queries that "
            "specialized API tools cannot handle."
        ),
        instruction_file="database_analysis.md",
        tools=_CORE_TOOLS | _DATABASE_TOOLS,
    ),
    "variant_list_analysis": SkillDefinition(
        name="variant_list_analysis",
        description=(
            "Analyze a list of variants (e.g., lead variants from a GWAS) for shared "
            "phenotype associations, QTL patterns, tissue enrichment, and nearest genes. "
            "Use when a user provides 3 or more variants in any format (one per line, "
            "space-separated, tab-separated, etc.). ALWAYS prefer this over fetching "
            "individual variant details when multiple variants are given."
        ),
        instruction_file="variant_list_analysis.md",
        tools=_CORE_TOOLS | _GENETICS_API_TOOLS,
    ),
    "data_analysis": SkillDefinition(
        name="data_analysis",
        description=(
            "Write and RUN a Python script for statistical analysis or data processing "
            "(polars/numpy/scipy). This subagent writes the script, runs it in the sandbox "
            "with `run_analysis`, iterates on failures, and reports the printed output. "
            "Figures it produces are NOT displayed to the user, so plot on the main path "
            "by calling `run_analysis` directly instead."
        ),
        instruction_file="data_analysis.md",
        # the one skill that names run_analysis: it runs under the caller's authenticated
        # identity, threaded into run_subagents from the request context
        tools=_CORE_TOOLS | {"run_analysis"},
        allow_file_read=True,
    ),
}


@lru_cache(maxsize=None)
def _load_instruction(filename: str) -> str:
    """Load and cache a skill instruction markdown file."""
    path = INSTRUCTIONS_DIR / filename
    if not path.exists():
        logger.warning(f"Skill instruction file not found: {path}")
        return ""
    return path.read_text()


def get_skill(name: str) -> SkillDefinition | None:
    """Get a skill definition by name."""
    return SKILL_REGISTRY.get(name)


def get_skill_instructions(skill: SkillDefinition) -> str:
    """Get the system prompt instructions for a skill."""
    return _load_instruction(skill.instruction_file)


def get_skill_descriptions() -> str:
    """Get a formatted string of all available skills for the main agent."""
    lines = []
    for skill in SKILL_REGISTRY.values():
        lines.append(f"- **{skill.name}**: {skill.description}")
    return "\n".join(lines)
