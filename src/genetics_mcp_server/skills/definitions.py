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


@dataclass
class SkillDefinition:
    """A skill that can be assigned to a subagent."""

    name: str
    description: str
    instruction_file: str
    tool_categories: set[str]
    extra_tools: list[str] = field(default_factory=list)
    model: str | None = None
    max_tokens: int = 4096
    max_iterations: int = 10
    allow_file_read: bool = False
    allow_script_exec: bool = False
    allowed_paths: list[str] = field(default_factory=list)


SKILL_REGISTRY: dict[str, SkillDefinition] = {
    "genetics_data_extraction": SkillDefinition(
        name="genetics_data_extraction",
        description=(
            "Extract genetics data for genes, variants, or phenotypes using the API tools. "
            "Use for GWAS associations, credible sets, QTL data, gene expression, "
            "colocalization, LD, and exome/burden test results."
        ),
        instruction_file="genetics_data_extraction.md",
        tool_categories={"general", "api"},
    ),
    "literature_review": SkillDefinition(
        name="literature_review",
        description=(
            "Search scientific literature and the web for information about genes, "
            "variants, phenotypes, or biological mechanisms. Returns summaries of "
            "relevant papers and web sources."
        ),
        instruction_file="literature_review.md",
        tool_categories={"general"},
        extra_tools=["search_scientific_literature", "web_search"],
    ),
    "bigquery_analysis": SkillDefinition(
        name="bigquery_analysis",
        description=(
            "Run complex SQL queries against the genetics BigQuery database. "
            "Use for cross-dataset comparisons, aggregations, or queries that "
            "specialized API tools cannot handle."
        ),
        instruction_file="bigquery_analysis.md",
        tool_categories={"general", "bigquery"},
    ),
    "variant_list_analysis": SkillDefinition(
        name="variant_list_analysis",
        description=(
            "Analyze a list of variants (e.g., lead variants from a GWAS) for shared "
            "phenotype associations, QTL patterns, tissue enrichment, and nearest genes. "
            "Use when a user pastes or attaches a list of variants."
        ),
        instruction_file="variant_list_analysis.md",
        tool_categories={"general", "api"},
    ),
    "data_analysis": SkillDefinition(
        name="data_analysis",
        description=(
            "Execute Python scripts for statistical analysis, data processing, "
            "or custom visualizations (matplotlib/polars/scipy). Use when the user "
            "needs computations or plots beyond what built-in tools provide."
        ),
        instruction_file="data_analysis.md",
        tool_categories={"general"},
        allow_file_read=True,
        allow_script_exec=True,
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
