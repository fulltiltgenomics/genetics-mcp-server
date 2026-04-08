"""Tool definitions for genetics data access.

This module provides tool definitions in two formats:
1. FastMCP registration (for standalone MCP server)
2. Anthropic tool format (for LLM service)
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from genetics_mcp_server.tools.executor import ToolExecutor

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "search_phenotypes",
        "category": "general",
        "description": "Look up phenotypes. Use when you need to find if there is a phenotype for a disease/trait name or the exact phenotype code for a disease/trait name. Do NOT use this to find disease associations - use get_credible_sets_by_gene instead.",
        "parameters": {
            "query": {
                "type": "string",
                "description": "Disease or trait name(s) to look up. Supports comma-separated values for batch lookup (e.g., 'diabetes,obesity,hypertension')",
                "required": True,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum results (default 100)",
                "default": 100,
            },
        },
    },
    {
        "name": "search_genes",
        "category": "general",
        "description": "Look up gene symbols and positions. Use ONLY when you need to verify a gene symbol or find its genomic coordinates. Do NOT use this to find gene associations.",
        "parameters": {
            "query": {
                "type": "string",
                "description": "Gene name(s) or symbol(s) to look up. Supports comma-separated values for batch lookup (e.g., 'BRCA1,TP53,EGFR')",
                "required": True,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum results (default 10)",
                "default": 10,
            },
        },
    },
    {
        "name": "lookup_variants_by_rsid",
        "category": "general",
        "description": "Convert rsIDs to variant IDs (chr:pos:ref:alt format). Use this when you have rsIDs and need to convert them to variant format for use with other tools.",
        "parameters": {
            "rsids": {
                "type": "string",
                "description": "rsID or comma-separated list of rsIDs (e.g., 'rs1234567' or 'rs1234567,rs9876543')",
                "required": True,
            },
        },
    },
    {
        "name": "get_credible_sets_by_gene",
        "category": "api",
        "description": "Get credible sets for variants near a gene. Returns fine-mapped variants with phenotype codes, p-values, effect sizes, and PIPs. **IMPORTANT**: Always use the data_types parameter to filter results ('GWAS', 'eQTL', 'pQTL', 'sQTL', 'caQTL'). Without filtering, results may be truncated.",
        "parameters": {
            "gene": {
                "type": "string",
                "description": "Gene symbol or comma-separated list of gene symbols (e.g., 'APOE', 'IL23R', 'PCSK9')",
                "required": True,
            },
            "window": {
                "type": "integer",
                "description": "Window size in bp around gene (default 100000)",
                "default": 100000,
            },
            "resource": {
                "type": "string",
                "description": "Data resource: e.g. 'finngen', 'ukbb', or omit to search all.",
            },
            "data_types": {
                "type": "string",
                "description": "Comma-separated data types: 'GWAS' (disease), 'eQTL' (expression), 'pQTL' (protein), 'sQTL' (splicing), 'caQTL' (chromatin).",
            },
            "summarize": {
                "type": "boolean",
                "description": "If true, return credible set-level summary instead of variant-level data.",
                "default": True,
            },
        },
    },
    {
        "name": "get_credible_sets_by_variant",
        "category": "api",
        "description": "Get credible sets containing a specific variant. Returns fine-mapped associations where this variant is part of a credible set. Use this to find which phenotypes/traits a variant is associated with and its causal probability (PIP). NOTE: For 3+ variants, use analyze_variant_list instead — it is much faster and provides aggregated pattern analysis.",
        "parameters": {
            "variant": {
                "type": "string",
                "description": "Variant ID in format chr:pos:ref:alt (e.g., '19:44908684:T:C')",
                "required": True,
            },
            "resource": {
                "type": "string",
                "description": "Data resource: e.g. 'finngen', 'ukbb', or omit to search all.",
            },
            "data_types": {
                "type": "string",
                "description": "Comma-separated data types: 'GWAS', 'eQTL', 'pQTL', 'sQTL', 'caQTL'.",
            },
            "summarize": {
                "type": "boolean",
                "description": "If true, return credible set-level summary instead of variant-level data.",
                "default": True,
            },
        },
    },
    {
        "name": "get_credible_sets_by_phenotype",
        "category": "api",
        "description": "**PRIMARY TOOL for phenotype-to-gene queries.** Get ALL genes/variants associated with a phenotype from GWAS fine-mapping. Returns genome-wide significant loci with causal variant candidates ranked by PIP.",
        "parameters": {
            "phenotype": {
                "type": "string",
                "description": "Phenotype code (e.g., 'I9_CHD', 'T2D', 'K11_CROHN')",
                "required": True,
            },
            "resource": {
                "type": "string",
                "description": "Data resource: 'finngen' or 'ukbb' (default 'finngen')",
                "default": "finngen",
            },
            "summarize": {
                "type": "boolean",
                "description": "If true, return credible set-level summary. Default is true.",
                "default": True,
            },
        },
    },
    {
        "name": "get_credible_set_by_id",
        "category": "api",
        "description": "Get all variants in a specific credible set. Use this to investigate a credible set in detail - see all variants, their consequences, PIPs, and count how many variants are in the set.",
        "parameters": {
            "resource": {
                "type": "string",
                "description": "Data resource (e.g., 'finngen', 'ukbb')",
                "required": True,
            },
            "phenotype": {
                "type": "string",
                "description": "Phenotype code (e.g., 'K11_IBD_STRICT')",
                "required": True,
            },
            "credible_set_id": {
                "type": "string",
                "description": "Credible set ID (e.g., 'chr1:6535440-9535440_1')",
                "required": True,
            },
        },
    },
    {
        "name": "get_credible_sets_by_qtl_gene",
        "category": "api",
        "description": "Get QTL associations where a gene is the molecular trait (target). Returns variants ANYWHERE in the genome that affect expression/splicing/protein levels of the gene. Different from get_credible_sets_by_gene which finds variants NEAR a gene.",
        "parameters": {
            "gene": {
                "type": "string",
                "description": "Gene symbol or comma-separated list of gene symbols (e.g., 'APOE', 'IL23R', 'PCSK9')",
                "required": True,
            },
            "data_types": {
                "type": "string",
                "description": "Comma-separated QTL types: 'eQTL', 'pQTL', 'sQTL', 'caQTL'. Default returns all.",
            },
            "resource": {
                "type": "string",
                "description": "Data resource (default uses all available)",
            },
            "summarize": {
                "type": "boolean",
                "description": "If true, return credible set-level summary.",
                "default": False,
            },
        },
    },
    {
        "name": "get_gene_expression",
        "category": "api",
        "description": "Get tissue-specific gene expression levels. Returns expression data across tissues/cell types. Use this to understand where a gene is expressed.",
        "parameters": {
            "gene": {"type": "string", "description": "Gene symbol or comma-separated list of gene symbols", "required": True},
        },
    },
    {
        "name": "get_gene_disease_associations",
        "category": "api",
        "description": "Get Mendelian/rare disease gene-disease relationships from ClinGen/GENCC. Use ONLY for rare disease genetics questions, NOT for GWAS/common variant associations.",
        "parameters": {
            "gene": {"type": "string", "description": "Gene symbol or comma-separated list of gene symbols", "required": True},
        },
    },
    {
        "name": "get_colocalization",
        "category": "api",
        "description": "Get colocalization results for a variant. Returns trait pairs that share the same causal signal at this locus. Use this to find traits that may share biological mechanisms.",
        "parameters": {
            "variant": {
                "type": "string",
                "description": "Variant ID (e.g., '1:123456:A:G' or 'rs12345')",
                "required": True,
            },
        },
    },
    {
        "name": "get_exome_results_by_gene",
        "category": "api",
        "description": "Get rare variant burden test results for a gene. Returns gene-level association statistics from exome sequencing across available resources (FinnGen, UKBB/Genebass, etc.). Use this for single-gene burden queries. For batch queries across many genes, use BigQuery instead (call get_bigquery_schema to find the exome results table).",
        "parameters": {
            "gene": {"type": "string", "description": "Gene symbol or comma-separated list of gene symbols", "required": True},
        },
    },
    {
        "name": "get_phenotype_report", # TODO WHEN DISCUSSING SAMPLE SIZE, INCLUDE NUMBERS OF CASES AND CONTROLS
        "category": "api",
        "description": "Get a detailed markdown report for a phenotype. Returns a markdown report with credible sets and gene evidence summaries in those credible sets. This is the first line of phenotype-based inquiry and should be called first before calling other tools.",
        "parameters": {
            "resource": {
                "type": "string",
                "description": "Data resource: 'finngen', 'ukbb', 'open_targets' (default 'finngen')",
                "default": "finngen",
            },
            "phenotype_code": {
                "type": "string",
                "description": "Phenotype code (e.g., 'I9_CHD', 'T2D')",
                "required": True,
            },
        },
    },
    {
        "name": "lookup_phenotype_names",
        "category": "general",
        "description": "**Use this to translate phenotype codes to human-readable names.** Takes a list of phenotype codes and returns their names. Call this ONCE with ALL codes you need.",
        "parameters": {
            "codes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of phenotype codes to look up",
                "required": True,
            },
        },
    },
    {
        "name": "get_available_resources",
        "category": "general",
        "description": "Get a catalog of all available data resources.",
        "parameters": {},
    },
    {
        "name": "get_credible_sets_stats",
        "category": "api",
        "description": "Get summary statistics of credible sets (fine-mapped associations) for a dataset. Returns counts of risk and protective credible sets, including those with coding/LoF variants. Use this to answer questions like 'how many protective associations in FinnGen Kanta?' CRITICAL: Your response MUST include the INCLUDE_IN_RESPONSE field value verbatim - it contains a download link the user needs.",
        "parameters": {
            "resource_or_dataset": {
                "type": "string",
                "description": "Resource ID (e.g., 'finngen') or dataset ID (e.g., 'finngen_kanta', 'finngen_gwas'). Use get_available_resources to see available options.",
                "required": True,
            },
            "trait": {
                "type": "string",
                "description": "Optional: filter to specific trait/phenotype code",
            },
        },
    },
    {
        "name": "get_nearest_genes",
        "category": "api",
        "description": "Get genes nearest to a variant. Returns genes sorted by distance, with distance=0 for variants inside a gene. By default, only protein-coding genes are returned. Includes gene coordinates, strand, type, and HGNC annotations.",
        "parameters": {
            "variant": {
                "type": "string",
                "description": "Variant ID in format chr:pos:ref:alt (e.g., '5:56444534:A:T')",
                "required": True,
            },
            "gene_type": {
                "type": "string",
                "description": "Type of genes: 'protein_coding' or 'all' (default 'protein_coding')",
                "default": "protein_coding",
            },
            "n": {
                "type": "integer",
                "description": "Maximum number of genes to return (default 3, max 20)",
                "default": 3,
            },
            "max_distance": {
                "type": "integer",
                "description": "Maximum distance in bp from variant (default 1000000)",
                "default": 1000000,
            },
            "gencode_version": {
                "type": "string",
                "description": "Gencode version to use (optional)",
            },
            "return_hgnc_symbol_if_only_ensg": {
                "type": "boolean",
                "description": "Return HGNC symbol if gencode has only ENSG id (default false)",
                "default": False,
            },
        },
    },
    {
        "name": "get_genes_in_region",
        "category": "api",
        "description": "Get all genes in a genomic region. Returns genes overlapping the specified coordinates with gene name, position, strand, type, and HGNC annotations.",
        "parameters": {
            "chr": {
                "type": "string",
                "description": "Chromosome (e.g., '1', '22', 'X')",
                "required": True,
            },
            "start": {
                "type": "integer",
                "description": "Start position (bp)",
                "required": True,
            },
            "end": {
                "type": "integer",
                "description": "End position (bp)",
                "required": True,
            },
            "gene_type": {
                "type": "string",
                "description": "Type of genes: 'protein_coding' or 'all' (default 'protein_coding')",
                "default": "protein_coding",
            },
            "gencode_version": {
                "type": "string",
                "description": "Gencode version to use (optional)",
            },
        },
    },
    {
        "name": "search_scientific_literature",
        "category": "general",
        "description": "Search scientific literature (PubMed, bioRxiv, medRxiv preprints). Use for finding research papers about genes, variants, diseases, or biological mechanisms.",
        "parameters": {
            "query": {
                "type": "string",
                "description": "Search query - can include gene names, disease names, variant IDs, or biological concepts.",
                "required": True,
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum papers to return (default 10, max 25)",
                "default": 10,
            },
            "include_preprints": {
                "type": "boolean",
                "description": "Include bioRxiv/medRxiv preprints (default true)",
                "default": True,
            },
            "date_range": {
                "type": "string",
                "description": "Optional date filter: 'last_year', 'last_5_years', or 'YYYY-YYYY' range",
            },
            "backend": {
                "type": "string",
                "description": "Search backend: 'europepmc' (structured results) or 'perplexity' (AI-enhanced with summary). Defaults to server configuration.",
                "enum": ["europepmc", "perplexity"],
            },
        },
    },
    {
        "name": "web_search",
        "category": "general",
        "description": "Search the web for general information. Use for finding drug information, clinical guidelines, news, or explanations of concepts. Use search_scientific_literature for research papers instead.",
        "parameters": {
            "query": {
                "type": "string",
                "description": "Search query",
                "required": True,
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum results (default 5, max 10)",
                "default": 5,
            },
            "include_domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional: only search these domains",
            },
            "exclude_domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional: exclude these domains",
            },
        },
    },
    {
        "name": "create_phewas_plot",
        "category": "general",
        "description": "Create a PheWAS (Phenome-Wide Association Study) plot showing all phenotype associations for a variant. Returns a base64-encoded PNG image with phenotypes grouped by category on the X-axis and -log10(p-value) on the Y-axis.",
        "parameters": {
            "variant": {
                "type": "string",
                "description": "Variant ID (chr:pos:ref:alt, e.g., '19:44908684:T:C')",
                "required": True,
            },
            "resource": {
                "type": "string",
                "description": "Data resource: 'finngen', 'ukbb', or omit for all sources",
            },
            "significance_threshold": {
                "type": "number",
                "description": "Show significance line at this -log10(p) value (default 7.3, genome-wide significance)",
                "default": 7.3,
            },
            "min_mlog10p": {
                "type": "number",
                "description": "Only show associations with -log10(p) above this value (default 2.0)",
                "default": 2.0,
            },
        },
    },
    {
        "name": "get_ld_between_variants",
        "category": "api",
        "description": "Get linkage disequilibrium (LD) statistics between two specific variants. Returns r2 and D' values from the FinnGen reference panel. Both variants must be on the same chromosome and within 5 Mb of each other.",
        "parameters": {
            "variant1": {
                "type": "string",
                "description": "First variant ID in format chr:pos:ref:alt (e.g., '6:44693011:A:G')",
                "required": True,
            },
            "variant2": {
                "type": "string",
                "description": "Second variant ID in format chr:pos:ref:alt (e.g., '6:44682355:C:G')",
                "required": True,
            },
            "r2_threshold": {
                "type": "number",
                "description": "Minimum r2 threshold to consider variants in LD (default 0.1)",
                "default": 0.1,
            },
            "panel": {
                "type": "string",
                "description": "LD reference panel: 'sisu42' (latest, freeze 10+), 'sisu4', or 'sisu3'",
                "default": "sisu42",
                "enum": ["sisu3", "sisu4", "sisu42"],
            },
        },
    },
    {
        "name": "get_variants_in_ld",
        "category": "api",
        "description": "Get all variants in linkage disequilibrium (LD) with a given variant. Returns variants within the specified window that exceed the r2 threshold, useful for finding proxy variants or understanding LD structure.",
        "parameters": {
            "variant": {
                "type": "string",
                "description": "Variant ID in format chr:pos:ref:alt (e.g., '6:44693011:A:G')",
                "required": True,
            },
            "window": {
                "type": "integer",
                "description": "Window size in base pairs around the variant (default 1500000)",
                "default": 1500000,
            },
            "r2_threshold": {
                "type": "number",
                "description": "Minimum r2 threshold to return variants (default 0.6)",
                "default": 0.6,
            },
            "panel": {
                "type": "string",
                "description": "LD reference panel: 'sisu42' (latest, freeze 10+), 'sisu4', or 'sisu3'",
                "default": "sisu42",
                "enum": ["sisu3", "sisu4", "sisu42"],
            },
        },
    },
    {
        "name": "get_summary_stats",
        "category": "api",
        "description": """Get summary statistics (p-value, beta, standard error, allele frequencies) for specific variant-phenotype pairs from a resource.

Use this tool when:
- The user asks about a variant's association with a specific phenotype (e.g., "what is the p-value of rs429358 for Alzheimer's in FinnGen?")
- A result seems suspiciously missing — e.g., a variant is in a credible set for a FinnGen phenotype but not in the corresponding meta-analysis credible set
- You need the actual effect size or p-value for a variant-phenotype combination, not just whether it's in a credible set
- You want to compare association statistics across resources for the same variant-phenotype pair

Do NOT use this as a discovery tool — use credible set tools or PheWAS for that. This tool is for targeted lookups when you already know which variant(s) and phenotype(s) to query.""",
        "parameters": {
            "variants": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of variant IDs in chr:pos:ref:alt format (e.g., ['19:44908684:T:C', '1:154453788:C:T']). Separator can be : - _ or |",
                "required": True,
            },
            "phenotypes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of phenotype codes (e.g., ['T2D', 'I9_CHD'])",
                "required": True,
            },
            "resource": {
                "type": "string",
                "description": "Data resource: 'finngen' or 'finngen_mvp_ukbb'",
                "default": "finngen",
            },
            "data_type": {
                "type": "string",
                "description": "Analysis data type: 'gwas' or 'eqtl'",
                "default": "gwas",
            },
        },
    },
    {
        "name": "analyze_variant_list",
        "category": "api",
        "description": """Analyze a list of variants for shared phenotype associations, QTL patterns, and tissue enrichment.

Use this when a user provides a list of variants (e.g., lead variants from a GWAS) and wants to know:
- Which phenotypes are associated with multiple variants (pleiotropy)
- Which pQTL and eQTL genes are shared across variants
- Which tissues show eQTL enrichment
- What the nearest gene is for each variant

Input: variants separated by newlines or spaces (chr:pos:ref:alt format, any separator like : - _ | / accepted, chr prefix optional, 23 treated as X).
Optionally include beta/se/pvalue columns (tab, comma, or space separated).
If betas are provided, direction consistency is reported (whether the variant's effect and the association effect are in the same direction).

IMPORTANT: When a user provides multiple variants (3+), ALWAYS use this tool instead of fetching individual variant details one by one.

Returns aggregated counts sorted by frequency. The response already includes nearest genes for every variant in the variant_genes array — do NOT call get_nearest_genes separately after using this tool.""",
        "parameters": {
            "variants": {
                "type": "string",
                "description": "Variant list: one per line or space-separated. Format: chr:pos:ref:alt (any CPRA separator accepted: : - _ | / \\). Optionally include tab/comma/space-separated beta, se, pvalue columns. A header row is auto-detected.",
                "required": True,
            },
            "resource": {
                "type": "string",
                "description": "Filter to a specific data resource (e.g., 'finngen', 'ukbb'). Omit to search all.",
            },
        },
    },
]

# BigQuery tools for advanced queries
BIGQUERY_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "query_bigquery",
        "category": "bigquery",
        "description": """Execute a SQL query against the genetics BigQuery database.

For simple single-gene or single-variant lookups, prefer specialized tools (get_credible_sets_by_gene, get_credible_sets_by_variant, etc.).

**USE BIGQUERY when the question involves:**
- Aggregations across many phenotypes, genes, or variants
- Complex filtering (e.g., "LoF variants with PIP > 0.05 AND MAF < 0.05 across all traits")
- Cross-referencing between data types (e.g., fine-mapping results vs. burden test results)
- Batch queries over many genes/variants that would require many individual API calls
- Custom statistical summaries or counts

**IMPORTANT: Always call get_bigquery_schema FIRST** to discover all available tables and their columns. The database contains more tables than just credible sets — including exome/burden test results and other data types.

Use fully qualified view names (e.g., `genetics_results.credible_sets_v`).
Views include a `resource` column (finngen, ukbb, open_targets, etc.) for filtering by data source.
Always include a LIMIT clause in your SQL to control how many rows are shown to the user.
The download file automatically includes all matching rows (up to 100,000) regardless of the SQL LIMIT.
If the download hits the 100,000-row cap, tell the user to add filters to narrow the results.""",
        "parameters": {
            "sql": {
                "type": "string",
                "description": "SQL query to execute. Use fully qualified view names (e.g., genetics_results.credible_sets_v). Call get_bigquery_schema first to discover available tables. Always include LIMIT clause.",
                "required": True,
            },
            "max_rows": {
                "type": "integer",
                "description": "Maximum rows to return to the LLM (default 1000). The download file is not affected by this limit.",
                "default": 1000,
            },
            "dry_run": {
                "type": "boolean",
                "description": "If true, estimate cost without executing",
                "default": False,
            },
        },
    },
    {
        "name": "get_bigquery_schema",
        "category": "bigquery",
        "description": "Get schema for ALL available BigQuery tables and views. **Always call this before query_bigquery** to discover what data is available — the database contains tables for credible sets, colocalization, exome/burden test results, and more. Returns column names, types, and descriptions for each table.",
        "parameters": {},
    },
]

SUBAGENT_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "launch_subagents",
        "category": "general",
        "description": """Launch one or more specialized subagents in parallel to handle complex queries.
Each subagent has its own skill (instructions + tools) and runs independently.
Use this when the question requires multiple independent data gathering or analysis tasks that can run simultaneously.

Available skills:
- **genetics_data_extraction**: Extract genetics data (GWAS, QTL, credible sets, gene expression, LD, etc.)
- **literature_review**: Search scientific literature and web for relevant publications
- **bigquery_analysis**: Run complex SQL queries against the genetics database
- **data_analysis**: Execute Python scripts for statistical analysis or custom visualizations
- **variant_list_analysis**: Analyze a list of variants for phenotype, QTL, and tissue patterns""",
        "parameters": {
            "tasks": {
                "type": "array",
                "description": "List of subagent tasks to run in parallel",
                "required": True,
                "items": {
                    "type": "object",
                    "properties": {
                        "skill": {
                            "type": "string",
                            "description": "Skill name (genetics_data_extraction, literature_review, bigquery_analysis, data_analysis, variant_list_analysis)",
                        },
                        "query": {
                            "type": "string",
                            "description": "Specific question or task for this subagent",
                        },
                        "context": {
                            "type": "string",
                            "description": "Additional context from the conversation to pass to the subagent",
                        },
                    },
                    "required": ["skill", "query"],
                },
            },
        },
    },
]

# valid tool profiles and which categories each profile includes
TOOL_PROFILES: dict[str, set[str]] = {
    "api": {"general", "api"},
    "bigquery": {"general", "bigquery"},
    "rag": {"general"},
}


def get_anthropic_tools(
    custom_descriptions: dict[str, str] | None = None,
    tool_profile: str | None = None,
    disabled_tools: set[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Return tool definitions in Anthropic's format, filtered by tool profile.

    Args:
        custom_descriptions: Optional dict mapping tool names to custom descriptions
        tool_profile: Profile controlling which tool categories to include.
            None = all tools, "api" = general+api, "bigquery" = general+bigquery,
            "rag" = general only (RAG tools are external, handled separately).
        disabled_tools: Optional set of tool names to exclude.
    """
    anthropic_tools = []

    all_tools = list(TOOL_DEFINITIONS) + list(BIGQUERY_TOOL_DEFINITIONS) + list(SUBAGENT_TOOL_DEFINITIONS)

    if disabled_tools:
        all_tools = [t for t in all_tools if t["name"] not in disabled_tools]

    if tool_profile is not None:
        allowed_categories = TOOL_PROFILES.get(tool_profile, {"general"})
        all_tools = [t for t in all_tools if t.get("category") in allowed_categories]

    for tool_def in all_tools:
        # build input_schema from parameters
        properties = {}
        required = []

        for param_name, param_info in tool_def.get("parameters", {}).items():
            prop = {"type": param_info["type"]}
            if "description" in param_info:
                prop["description"] = param_info["description"]
            if "default" in param_info:
                prop["default"] = param_info["default"]
            if param_info.get("items"):
                prop["items"] = param_info["items"]
            if param_info.get("enum"):
                prop["enum"] = param_info["enum"]
            properties[param_name] = prop

            if param_info.get("required"):
                required.append(param_name)

        description = tool_def["description"]
        if custom_descriptions and tool_def["name"] in custom_descriptions:
            description = custom_descriptions[tool_def["name"]]

        anthropic_tools.append(
            {
                "name": tool_def["name"],
                "description": description,
                "input_schema": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            }
        )

    return anthropic_tools


def register_mcp_tools(
    mcp: "FastMCP",
    executor: "ToolExecutor",
    disabled_tools: set[str] | None = None,
) -> None:
    """
    Register all tools with a FastMCP server instance.

    Args:
        mcp: FastMCP server instance
        executor: ToolExecutor instance for making API calls
        disabled_tools: Optional set of tool names to skip registration.
    """
    _disabled = disabled_tools or set()

    @mcp.tool()
    async def search_phenotypes(query: str, limit: int = 100) -> dict:
        """Look up phenotypes by disease/trait name. Supports comma-separated values for batch lookup."""
        return await executor.search_phenotypes(query, limit)

    @mcp.tool()
    async def search_genes(query: str, limit: int = 10) -> dict:
        """Look up gene symbols and positions. Supports comma-separated values for batch lookup."""
        return await executor.search_genes(query, limit)

    @mcp.tool()
    async def lookup_variants_by_rsid(rsids: str) -> dict:
        """Convert rsIDs to variant IDs (chr:pos:ref:alt format)."""
        return await executor.lookup_variants_by_rsid(rsids)

    @mcp.tool()
    async def get_credible_sets_by_gene(
        gene: str,
        window: int = 100000,
        resource: str | None = None,
        data_types: str | None = None,
        summarize: bool = True,
    ) -> dict:
        """Get credible sets for variants near a gene."""
        return await executor.get_credible_sets_by_gene(
            gene, window, resource, data_types, summarize
        )

    @mcp.tool()
    async def get_credible_sets_by_variant(
        variant: str,
        resource: str | None = None,
        data_types: str | None = None,
        summarize: bool = True,
    ) -> dict:
        """Get credible sets containing a specific variant."""
        return await executor.get_credible_sets_by_variant(
            variant, resource, data_types, summarize
        )

    @mcp.tool()
    async def get_credible_sets_by_phenotype(
        phenotype: str,
        resource: str = "finngen",
        summarize: bool = True,
    ) -> dict:
        """Get all genes/variants associated with a phenotype from GWAS fine-mapping."""
        return await executor.get_credible_sets_by_phenotype(
            phenotype, resource, summarize
        )

    @mcp.tool()
    async def get_credible_set_by_id(
        resource: str,
        phenotype: str,
        credible_set_id: str,
    ) -> dict:
        """Get all variants in a specific credible set."""
        return await executor.get_credible_set_by_id(resource, phenotype, credible_set_id)

    @mcp.tool()
    async def get_credible_sets_by_qtl_gene(
        gene: str,
        data_types: str | None = None,
        resource: str | None = None,
        summarize: bool = False,
    ) -> dict:
        """Get QTL associations where a gene is the molecular trait."""
        return await executor.get_credible_sets_by_qtl_gene(
            gene, data_types, resource, summarize
        )

    @mcp.tool()
    async def get_gene_expression(gene: str) -> dict:
        """Get tissue-specific gene expression levels."""
        return await executor.get_gene_expression(gene)

    @mcp.tool()
    async def get_gene_disease_associations(gene: str) -> dict:
        """Get Mendelian/rare disease gene-disease relationships."""
        return await executor.get_gene_disease_associations(gene)

    @mcp.tool()
    async def get_colocalization(variant: str) -> dict:
        """Get colocalization results for a variant."""
        return await executor.get_colocalization(variant)

    @mcp.tool()
    async def get_exome_results_by_gene(gene: str) -> dict:
        """Get rare variant burden test results for a gene."""
        return await executor.get_exome_results_by_gene(gene)

    if "get_phenotype_report" not in _disabled:

        @mcp.tool()
        async def get_phenotype_report(resource: str, phenotype_code: str) -> dict:
            """Get a detailed markdown report for a phenotype."""
            return await executor.get_phenotype_report(resource, phenotype_code)

    @mcp.tool()
    async def lookup_phenotype_names(codes: list[str]) -> dict:
        """Translate phenotype codes to human-readable names."""
        return await executor.lookup_phenotype_names(codes)

    @mcp.tool()
    async def get_available_resources() -> dict:
        """Get a catalog of all available data resources."""
        return await executor.get_available_resources()

    if "get_credible_sets_stats" not in _disabled:

        @mcp.tool()
        async def get_credible_sets_stats(
            resource_or_dataset: str,
            trait: str | None = None,
        ) -> dict:
            """Get credible sets stats. CRITICAL: Include the INCLUDE_IN_RESPONSE field value verbatim in your response."""
            return await executor.get_credible_sets_stats(resource_or_dataset, trait)

    @mcp.tool()
    async def get_nearest_genes(
        variant: str,
        gene_type: str = "protein_coding",
        n: int = 3,
        max_distance: int = 1000000,
        gencode_version: str | None = None,
        return_hgnc_symbol_if_only_ensg: bool = False,
    ) -> dict:
        """Get genes nearest to a variant."""
        return await executor.get_nearest_genes(
            variant,
            gene_type,
            n,
            max_distance,
            gencode_version,
            return_hgnc_symbol_if_only_ensg,
        )

    @mcp.tool()
    async def get_genes_in_region(
        chr: str,
        start: int,
        end: int,
        gene_type: str = "protein_coding",
        gencode_version: str | None = None,
    ) -> dict:
        """Get all genes in a genomic region."""
        return await executor.get_genes_in_region(
            chr, start, end, gene_type, gencode_version
        )

    @mcp.tool()
    async def search_scientific_literature(
        query: str,
        max_results: int = 10,
        include_preprints: bool = True,
        date_range: str | None = None,
        backend: str | None = None,
    ) -> dict:
        """Search scientific literature via Europe PMC or Perplexity."""
        return await executor.search_scientific_literature(
            query, max_results, include_preprints, date_range, backend
        )

    @mcp.tool()
    async def web_search(
        query: str,
        max_results: int = 5,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
    ) -> dict:
        """Search the web for general information."""
        return await executor.web_search(
            query, max_results, include_domains, exclude_domains
        )

    @mcp.tool()
    async def create_phewas_plot(
        variant: str,
        resource: str | None = None,
        significance_threshold: float = 7.3,
        min_mlog10p: float = 2.0,
    ) -> dict:
        """Create a PheWAS plot showing phenotype associations for a variant."""
        return await executor.create_phewas_plot(
            variant, resource, significance_threshold, min_mlog10p
        )

    @mcp.tool()
    async def get_ld_between_variants(
        variant1: str,
        variant2: str,
        r2_threshold: float = 0.1,
        panel: str = "sisu42",
    ) -> dict:
        """Get LD statistics between two specific variants from FinnGen reference panel."""
        return await executor.get_ld_between_variants(
            variant1, variant2, r2_threshold, panel
        )

    @mcp.tool()
    async def get_variants_in_ld(
        variant: str,
        window: int = 1500000,
        r2_threshold: float = 0.6,
        panel: str = "sisu42",
    ) -> dict:
        """Get all variants in LD with a given variant from FinnGen reference panel."""
        return await executor.get_variants_in_ld(variant, window, r2_threshold, panel)

    @mcp.tool()
    async def analyze_variant_list(
        variants: str,
        resource: str | None = None,
    ) -> dict:
        """Analyze a list of variants for phenotype, QTL, and tissue patterns."""
        return await executor.analyze_variant_list(variants, resource)

    @mcp.tool()
    async def get_summary_stats(
        variants: list[str],
        phenotypes: list[str],
        resource: str = "finngen",
        data_type: str = "gwas",
    ) -> dict:
        """Get summary statistics for specific variant-phenotype pairs."""
        return await executor.get_summary_stats(variants, phenotypes, resource, data_type)

    # BigQuery tools - available via MCP server for direct SQL queries
    @mcp.tool()
    async def query_bigquery(
        sql: str,
        max_rows: int = 1000,
        dry_run: bool = False,
    ) -> dict:
        """Execute SQL against genetics BigQuery. Call get_bigquery_schema first to discover available tables."""
        return await executor.query_bigquery(sql, max_rows, dry_run)

    @mcp.tool()
    async def get_bigquery_schema() -> dict:
        """Get schema for ALL available BigQuery tables. Always call this before writing queries."""
        return await executor.get_bigquery_schema()
