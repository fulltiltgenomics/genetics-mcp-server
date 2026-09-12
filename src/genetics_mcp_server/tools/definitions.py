"""Tool definitions for genetics data access.

This module provides tool definitions in two formats:
1. FastMCP registration (for standalone MCP server)
2. Anthropic tool format (for LLM service)
"""

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import Field

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from genetics_mcp_server.tools.orchestration import ServerToolExecutor

logger = logging.getLogger(__name__)

# WHEN A PARAMETER MAY DECLARE `minimum`/`maximum`/`pattern` (genetics-results-suite-4h6.70).
# get_anthropic_tools copies these three keywords straight into the emitted input_schema, so
# a bound here is a claim about the SERVER, not a wish. The rule is: declare a bound only
# where enforcing code already applies it to every path that parameter can take, and derive
# the number from that code rather than from the description — where the two disagree the
# enforcement wins.
#
# Two different kinds of bound live on this surface, and this comment must not blur them:
#   - REJECTED: the `sql_int`/`sql_float` sites (the four `window` params, `min_pip`,
#     `get_hla_by_allele.max_rows`) and the sandbox timeout raise on an out-of-range value.
#     These are also mirrored onto the MCP surface as pydantic `Field(ge=..., le=...)` —
#     see the docstring on register_mcp_tools below — because there the identical
#     rejection just moves earlier.
#   - CLAMPED: `web_search.max_results`, `search_mgi.max_results`,
#     `search_cbioportal.max_results`, `search_uniprot.size`,
#     `get_drug_targets_for_gene.min_phase`, `get_drug_targets_for_gene.max_results`,
#     `get_target_bioactivity.pchembl_min`, `get_target_bioactivity.max_results` never
#     reject a numeric out-of-range value; it is silently coerced into range and the call
#     still succeeds (a NON-numeric value is a different matter: the ChEMBL pair returns
#     a `stage: input` failure rather than clamping, which no bound here can express).
#     These are advisory-only on this surface (they steer the model, nothing enforces them here
#     directly — the enforcement is downstream) and deliberately NOT mirrored onto the MCP
#     surface, because rejecting there would break a live client that today sends an
#     out-of-range value and gets a clamped, successful result back.
# A CLAMPED `minimum`/`maximum` is only honest if the clamp truly applies to every value
# that reaches it — see the note beside `search_uniprot.size` below for a case where it
# does not, and the bound is omitted rather than declared wrong.
#
# Each declaration below names its enforcing site; tests/test_tool_schema_bounds.py
# ties the ones that mirror a named constant back to the constant so the pair cannot drift.
#
# Deliberately NOT declared, so the next reader does not "complete" the set:
#   - search_scientific_literature.max_results — the "max 25" clamp exists only on the
#     europepmc path (executor.py `_search_europepmc_literature`); the DEFAULT backend is
#     perplexity, which slices to max_results uncapped.
#   - query_database.max_rows — capped downstream by db-api on bytes, not here.
#   - `pattern` on any parameter. The plausible candidates (get_hla_by_allele.allele,
#     read_artifact.name) are validated AFTER a normalization step that widens what is
#     accepted, so a regex matching the validator would reject inputs the server handles.

# THE OPT-IN, WRITTEN ONCE AND SHARED BY BOTH AlphaGenome TOOLS. The user ruled that this
# wording plus the system-prompt block is the ENTIRE enforcement -- there is no setting and
# no gate behind it -- so two capabilities under two names must not carry two drifting
# copies of it. Sharing the literal is what makes "covered identically" checkable rather
# than a promise; tests/test_alphagenome_comparison.py asserts both descriptions contain it.
_ALPHAGENOME_OPT_IN = """CALL THIS ONLY WHEN THE USER HAS ASKED FOR IT. Exactly three things count as asking:
1. the user names AlphaGenome;
2. the user asks for a model prediction of a variant's regulatory effect;
3. the user asks how a measured value in this suite compares with what a model predicts for the same variant — that comparison is a first-class use of this tool, not a workaround.

Nothing else is. In particular:
- Do NOT call it as background enrichment, and do not add a prediction to an answer nobody asked one for.
- "What does this variant do?", "tell me about rs...", "is this variant causal?", "why is this locus associated?" are NOT requests for AlphaGenome. Answer them from this suite's own measured and fine-mapped data.
- This suite having nothing to say about a variant is NOT a reason to call it. Say the data is silent; you may OFFER a prediction in one line and then wait to be asked.
- It is an ADDITIONAL source of evidence, not a fallback for gaps — and having it available is not a reason to use it. It is a rate-limited external model under a non-commercial licence."""

# The reading rules for the per-modality `validation` block, likewise shared: it is the same
# block on both tools' results, and a rule stated on one surface only is a rule the model
# will follow on one surface only.
_ALPHAGENOME_VALIDATION_RULES = """READ THE `validation` BLOCK BEFORE QUOTING A NUMBER. Every modality in the result carries its own — `tier`, `status`, `quantity`, `calibrated_against`, `population_rho`, `rho_scope`:
- `tier` is how deeply the MODALITY was calibrated here — 1-3 against this suite's own measurements, 4 against nothing. It is a property of the modality and says nothing about how good this variant's prediction is; it is not a score, a rank or a confidence.
- `quantity: "signed"` — the sign is meaningful (negative is a predicted decrease). `quantity: "magnitude"` — the direction is NOT reported and you must not state or infer one.
- `population_rho` with `rho_scope: "population"` is a cohort-level Spearman correlation between this MODALITY and `calibrated_against`, across many variants. It is a property of the modality. It is NOT a confidence for the variant in hand and must never be quoted as one.
- `status: "unvalidated"` (no `population_rho`) means the modality was never checked against anything measured in this suite. Say so whenever you report one.
- `quantile` ranks the score against a genome-wide background and usually says more than the raw value."""

_ALPHAGENOME_SIDE_BY_SIDE = """SIDE BY SIDE WITH MEASURED DATA the labelling matters MORE, not less: label every number from this tool as predicted, name the source of every measured number, never merge or average the two into one figure, and where they disagree say that they disagree."""


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    # The three entity lookups that open this list — search_phenotypes, search_genes,
    # lookup_variants_by_rsid — are the exception to what `sdk_replaceable` otherwise means.
    # The SDK does cover them (`genetics.search`), so True would be the mechanical answer;
    # they are False because resolving a symbol or a phenotype name to an id is what the
    # model does BEFORE it writes a script, and the code surface would otherwise force a
    # sandbox round-trip for it. This is the same allow-list the pre-collapse `code` profile
    # carried, plus the catalogue pair (list_datasets, get_resource_metadata) further down,
    # which is False for the same reason. Flip one to True and the code surface loses it.
    {
        "name": "search_phenotypes",
        "category": "general",
        "sdk_replaceable": False,
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
        "sdk_replaceable": False,
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
        "sdk_replaceable": False,
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
        "sdk_replaceable": True,
        "description": "Get credible sets for variants near a gene. Returns fine-mapped variants with phenotype codes, p-values, effect sizes, and PIPs. **IMPORTANT**: Always use the data_types parameter to filter results ('GWAS', 'eQTL', 'pQTL', 'sQTL', 'caQTL'). Without filtering, results may be truncated.",
        "parameters": {
            "gene": {
                "type": "string",
                "description": "Gene symbol or comma-separated list of gene symbols (e.g., 'APOE', 'IL23R', 'PCSK9')",
                "required": True,
            },
            "window": {
                "type": "integer",
                "description": "Flank in bp added on each side of the gene body (default 500000). A wide window is used because the strongest signal attributed to a gene can sit far from its body — e.g. a long-range regulatory variant several hundred kb upstream. Narrow it only when you specifically want signals inside or immediately around the gene.",
                "default": 500000,
            },
            "resource": {
                "type": "string",
                "description": "Data resource: e.g. 'finngen', 'ukbb', or omit to search all.",
            },
            "data_types": {
                "type": "string",
                "description": "Comma-separated data types: 'GWAS' (disease), 'eQTL' (expression), 'pQTL' (protein), 'sQTL' (splicing), 'caQTL' (chromatin accessibility).",
            },
            "summarize": {
                "type": "boolean",
                "description": (
                    "If true, return credible set-level summary instead of variant-level "
                    "data. The summary carries a `counts` block with the per-data-type "
                    "totals (credible sets, associations, variants, traits, cell types, and "
                    "peaks for caQTL) — read those for any 'how many' question rather than "
                    "counting the listed credible sets, which may be truncated."
                ),
                "default": True,
            },
        },
    },
    {
        "name": "get_credible_sets_by_variant",
        "category": "api",
        "sdk_replaceable": True,
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
                "description": (
                    "If true, return credible set-level summary instead of variant-level "
                    "data. The summary carries a `counts` block with the per-data-type "
                    "totals (credible sets, associations, variants, traits, cell types, and "
                    "peaks for caQTL) — read those for any 'how many' question rather than "
                    "counting the listed credible sets, which may be truncated."
                ),
                "default": True,
            },
        },
    },
    {
        "name": "get_credible_sets_by_region",
        "category": "api",
        "sdk_replaceable": True,
        "description": "Get credible sets overlapping a genomic region across all resources. Use this when the locus is defined by coordinates rather than a gene or a variant — e.g. a GWAS peak boundary, a fine-mapping window from a paper, or 'what else is fine-mapped in this interval'. For a gene use get_credible_sets_by_gene (it applies the window for you) and for a single variant use get_credible_sets_by_variant.",
        "parameters": {
            "region": {
                "type": "string",
                "description": "Region as chr:start-end (e.g. '1:1000000-1500000'; X is accepted). Max 10Mb.",
                "required": True,
            },
            "resource": {
                "type": "string",
                "description": "Comma-separated resources, e.g. 'finngen' or 'finngen,eqtl_catalogue'. Omit to search all.",
            },
            "coding_only": {
                "type": "boolean",
                "description": "If true, return only coding variants (by their most_severe consequence).",
                "default": False,
            },
            "summarize": {
                "type": "boolean",
                "description": (
                    "If true, return a credible set-level summary instead of variant-level rows. "
                    "The summary carries a `counts` block with per-data-type totals — read those "
                    "for any 'how many' question. If false, variant rows are capped at 500 and "
                    "`truncated` says whether more exist; the full set is at `_download_url`."
                ),
                "default": True,
            },
        },
    },
    {
        "name": "get_credible_sets_by_phenotype",
        "category": "api",
        "sdk_replaceable": True,
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
        "name": "get_credible_set_leads_by_phenotype",
        "category": "api",
        "sdk_replaceable": True,
        "description": "Get ONE row per credible set for a phenotype: the lead variant of each set (the flagged lead, else highest PIP with ties broken by p-value). Use this to enumerate a trait's independent signals — 'how many loci does this trait have', 'list the lead variants' — without pulling every member variant. get_credible_sets_by_phenotype returns all member variants of all sets, which is far larger; use that only when you need the members.",
        "parameters": {
            "phenotype": {
                "type": "string",
                "description": "Phenotype code (e.g., 'I9_CHD', 'T2D', 'K11_CROHN')",
                "required": True,
            },
            "resource": {
                "type": "string",
                "description": "Data resource (default 'finngen')",
                "default": "finngen",
            },
        },
    },
    {
        "name": "get_credible_set_by_id",
        "category": "api",
        "sdk_replaceable": True,
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
        "sdk_replaceable": True,
        "description": (
            "Get QTL associations where a gene is the molecular trait (target). Returns variants "
            "ANYWHERE in the genome that affect expression/splicing/protein levels of the gene. "
            "Different from get_credible_sets_by_gene which finds variants NEAR a gene. "
            "**This is also the correct tool for gene-based caQTL questions.** A caQTL trait is a "
            "chromatin ACCESSIBILITY PEAK, not a gene, so 'caQTL for gene X' means variants "
            "affecting peaks LINKED to X. This tool already resolves that link (Open4Gene "
            "peak-to-gene, cell-type-matched): for caQTL rows `trait` is the linked gene symbol "
            "and `trait_original` / `cs_id` hold the peak id (chr-start-end). Do NOT fall back to "
            "matching peak coordinates against the gene's position — linked peaks sit up to ~1 Mb "
            "away and most peaks near a gene are not linked to it."
        ),
        "parameters": {
            "gene": {
                "type": "string",
                "description": "Gene symbol or comma-separated list of gene symbols (e.g., 'APOE', 'IL23R', 'PCSK9')",
                "required": True,
            },
            "data_types": {
                "type": "string",
                "description": (
                    "Comma-separated QTL types: 'eQTL', 'pQTL', 'sQTL', 'caQTL'. Case-insensitive. "
                    "Default returns all, which for a well-studied gene can be thousands of rows "
                    "that get truncated before you see them — always set this when you only care "
                    "about one type."
                ),
            },
            "resource": {
                "type": "string",
                "description": "Data resource (default uses all available)",
            },
            "summarize": {
                "type": "boolean",
                "description": (
                    "If true (the default), return credible set-level summary instead of "
                    "variant-level data. Keep it true for counting questions: the variant-level "
                    "result for a well-studied gene runs to millions of characters and is cut off "
                    "before you see all of it. The summary carries a `counts` block with the "
                    "per-data-type totals (credible sets, associations, variants, traits, cell "
                    "types, and peaks for caQTL) — read those for any 'how many' question rather "
                    "than counting the listed credible sets, which may be truncated."
                ),
                "default": True,
            },
        },
    },
    {
        "name": "get_gene_expression",
        "category": "api",
        "sdk_replaceable": True,
        "description": "Get tissue-specific gene expression levels. Returns expression data across tissues/cell types. Use this to understand where a gene is expressed.",
        "parameters": {
            "gene": {"type": "string", "description": "Gene symbol or comma-separated list of gene symbols", "required": True},
        },
    },
    {
        "name": "get_asm_qtl_by_variant",
        "category": "api",
        "sdk_replaceable": True,
        "description": "Get allele-specific methylation QTL (ASM-QTL) data for a variant. Returns associations between a sequence variant and CpG/MDS methylation rates, including effect sizes, methylation rates on reference and alternative haplotypes, and variant rank (primary/secondary).",
        "parameters": {
            "variant": {
                "type": "string",
                "description": "Variant ID in format chr:pos:ref:alt (e.g., '1:808040:G:A')",
                "required": True,
            },
            "resources": {
                "type": "string",
                "description": "Comma-separated resources: 'decode_cpg' (CpG methylation), 'decode_mds' (MDS methylation). Omit to search all.",
            },
        },
    },
    {
        "name": "get_asm_qtl_by_gene",
        "category": "api",
        "sdk_replaceable": True,
        "description": "Get allele-specific methylation QTL (ASM-QTL) data for variants near a gene. Returns associations between sequence variants and CpG/MDS methylation rates for variants within the gene body ± window, selected by genomic coordinates (not by most-severe-consequence attribution, which misses nearby regulatory variants).",
        "parameters": {
            "gene": {
                "type": "string",
                "description": "Gene symbol or comma-separated list of gene symbols (e.g., 'PCSK9')",
                "required": True,
            },
            "resources": {
                "type": "string",
                "description": "Comma-separated resources: 'decode_cpg' (CpG methylation), 'decode_mds' (MDS methylation). Omit to search all.",
            },
            "window": {
                "type": "integer",
                "description": "Flank in bp added on each side of the gene body (default 500000).",
                "default": 500000,
                # executor.py sql_int(window, minimum=0, maximum=ToolExecutor._MAX_SQL_WINDOW)
                "minimum": 0,
                "maximum": 10_000_000,
            },
        },
    },
    {
        "name": "get_open_chromatin_by_variant",
        "category": "api",
        "sdk_replaceable": True,
        "description": "Get open-chromatin (scATAC/snATAC/bulk-ATAC/chromHMM) atlas peaks overlapping a variant's position. Answers 'in which cell types/tissues/conditions is this variant's region of open/accessible chromatin?'. Returns overlapping accessible regions labeled by cell_type, tissue, life_stage and condition (resting/stimulated/AD/control) so cell-type specificity can be reported. This is a peak ATLAS (measured accessibility across brain, heart, immune and body-wide contexts) — distinct from caqtl (accessibility QTL) and chromatin_peaks (peak-to-gene links).",
        "parameters": {
            "variant": {
                "type": "string",
                "description": "Variant as chr:pos:ref:alt or chr:pos (e.g., '1:1000500:A:G' or '1:1000500'); only chromosome and position are used for overlap",
                "required": True,
            },
            "resources": {
                "type": "string",
                "description": "Comma-separated resources: 'marderstein' (fetal+adult brain/heart scATAC), 'li_brain_atac' (adult brain), 'catlas' (body-wide adult), 'epimap' (bulk chromHMM regulatory states), 'calderon_immune' (stimulation-responsive immune), 'rosmap_brain' (aged/AD brain). Omit to search all.",
            },
        },
    },
    {
        "name": "get_open_chromatin_by_region",
        "category": "api",
        "sdk_replaceable": True,
        "description": "Get open-chromatin (scATAC/snATAC/bulk-ATAC/chromHMM) atlas peaks overlapping a genomic region. Answers 'in which cell types/tissues/conditions is this region of open/accessible chromatin?'. Returns overlapping accessible regions labeled by cell_type, tissue, life_stage and condition. This is a peak ATLAS of measured accessibility — distinct from caqtl (accessibility QTL) and chromatin_peaks (peak-to-gene links).",
        "parameters": {
            "chrom": {
                "type": "string",
                "description": "Chromosome (e.g., '1', 'chr1', 'X')",
                "required": True,
            },
            "start": {
                "type": "integer",
                "description": "Region start position (1-based, inclusive)",
                "required": True,
            },
            "end": {
                "type": "integer",
                "description": "Region end position (1-based, inclusive)",
                "required": True,
            },
            "resources": {
                "type": "string",
                "description": "Comma-separated resources: 'marderstein', 'li_brain_atac', 'catlas', 'epimap', 'calderon_immune', 'rosmap_brain'. Omit to search all.",
            },
        },
    },
    {
        "name": "get_open_chromatin_by_peak",
        "category": "api",
        "sdk_replaceable": True,
        "description": "Get one open-chromatin atlas peak by its peak id, returning every cell_type/tissue/condition row recorded for it. Use this to follow up a peak id returned by get_open_chromatin_by_variant/_by_region when you want that peak's full annotation rather than everything overlapping a position. Atlas peak ids are a SEPARATE id space from caQTL/Open4Gene peak ids (credible_sets trait, get_peak_to_genes): those will not be found here, so reach the atlas from a caQTL peak by region overlap (get_open_chromatin_by_region) instead.",
        "parameters": {
            "peak_id": {
                "type": "string",
                "description": "Atlas peak ID as chr-start-end with a bare numeric chromosome (e.g. '22-20750312-20751112'; X=23). This endpoint also tolerates a 'chr' prefix, but the open_chromatin_v BigQuery view does not — SQL must use the bare form.",
                "required": True,
            },
            "resources": {
                "type": "string",
                "description": "Comma-separated resources: 'marderstein', 'li_brain_atac', 'catlas', 'epimap', 'calderon_immune', 'rosmap_brain'. Omit to search all.",
            },
        },
    },
    {
        "name": "get_peak_to_genes",
        "category": "api",
        "sdk_replaceable": True,
        "description": "Get the GENES an Open4Gene chromatin peak is linked to, with the cell type each link was significant in. This is the peak-to-gene LINK table (which gene a regulatory region acts on) — distinct from get_open_chromatin_by_peak, which returns measured accessibility of the peak itself. Use this to interpret a caQTL signal: caQTL credible sets are keyed by peak, and this is what turns a peak id into candidate target genes.",
        "parameters": {
            "peak_id": {
                "type": "string",
                "description": "Peak ID as chr-start-end (e.g. 'chr5-35482826-35484273')",
                "required": True,
            },
            "resources": {
                "type": "string",
                "description": "Comma-separated resources. Omit to use all.",
            },
            "gencode_version": {
                "type": "string",
                "description": "GENCODE version for the returned gene coordinates. Omit for the latest available.",
            },
        },
    },
    {
        "name": "get_gene_to_peaks",
        "category": "api",
        "sdk_replaceable": True,
        "description": "Get the Open4Gene chromatin PEAKS linked to a gene, per cell type — the inverse of get_peak_to_genes. Answers 'which regulatory regions act on this gene, and in which cell types'. Distinct from get_open_chromatin_by_gene, which returns measured accessibility near the gene by coordinate overlap with no link evidence. Rows are capped at 500 inline; `truncated` says whether more exist.",
        "parameters": {
            "gene": {
                "type": "string",
                "description": "Gene symbol or ENSG ID (e.g. 'PCSK9', 'ENSG00000169174')",
                "required": True,
            },
            "resources": {
                "type": "string",
                "description": "Comma-separated resources. Omit to use all.",
            },
            "gencode_version": {
                "type": "string",
                "description": "GENCODE version for the gene's coordinates. Omit for the latest available.",
            },
        },
    },
    {
        "name": "get_open_chromatin_by_gene",
        "category": "api",
        "sdk_replaceable": True,
        "description": "Get open-chromatin (scATAC/snATAC/bulk-ATAC/chromHMM) atlas peaks near a gene, selected by genomic coordinates (gene body ± window, not most-severe-consequence attribution which misses nearby regulatory/enhancer peaks). Answers 'in which cell types/tissues/conditions is the chromatin around this gene open/accessible?'. Returns accessible regions labeled by cell_type, tissue, life_stage and condition. This is a peak ATLAS of measured accessibility — distinct from caqtl (accessibility QTL) and chromatin_peaks (peak-to-gene links).",
        "parameters": {
            "gene": {
                "type": "string",
                "description": "Gene symbol (e.g., 'PCSK9')",
                "required": True,
            },
            "resources": {
                "type": "string",
                "description": "Comma-separated resources: 'marderstein', 'li_brain_atac', 'catlas', 'epimap', 'calderon_immune', 'rosmap_brain'. Omit to search all.",
            },
            "window": {
                "type": "integer",
                "description": "Flank in bp added on each side of the gene body (default 500000).",
                "default": 500000,
                # executor.py sql_int(window, minimum=0, maximum=ToolExecutor._MAX_SQL_WINDOW)
                "minimum": 0,
                "maximum": 10_000_000,
            },
        },
    },
    {
        "name": "get_variant_effect_by_variant",
        "category": "api",
        "sdk_replaceable": True,
        "description": "Get in-silico PREDICTED variant effect on chromatin accessibility for a variant. Answers 'is this variant predicted to disrupt chromatin accessibility, how strongly, and in which cell types?'. Returns per-model, per-cell-type predicted scores: ChromBPNet (model=chrombpnet) gives the predicted accessibility effect (score/mlog10p/quantile_rank/is_significant) in specific cell_type/tissue contexts; FLARE (model=flare) gives a pan-context regulatory score (cell_type/tissue may be null). These are MODEL PREDICTIONS — distinct from measured caqtl (accessibility QTL) and open_chromatin (measured accessibility atlas).",
        "parameters": {
            "variant": {
                "type": "string",
                "description": "Variant as chr:pos:ref:alt or chr:pos (e.g., '1:1000500:A:G' or '1:1000500')",
                "required": True,
            },
            "resources": {
                "type": "string",
                "description": "Comma-separated resources: 'marderstein' (Marderstein/Kundaje 2026 ChromBPNet + FLARE predictions). Omit to search all.",
            },
        },
    },
    {
        "name": "get_variant_effect_by_gene",
        "category": "api",
        "sdk_replaceable": True,
        "description": "Get in-silico PREDICTED variant effects on chromatin accessibility for variants near a gene, selected by genomic coordinates (gene body ± window, not most-severe-consequence attribution which misses nearby regulatory variants). Answers 'how strongly and in which cell types are this gene's variants predicted to affect chromatin accessibility?'. Returns per-model, per-cell-type predicted-effect rows: ChromBPNet (model=chrombpnet) predicted accessibility effect in specific cell_type/tissue contexts; FLARE (model=flare) pan-context regulatory score (cell_type/tissue may be null). These are MODEL PREDICTIONS — distinct from measured caqtl (accessibility QTL) and open_chromatin (measured accessibility atlas).",
        "parameters": {
            "gene": {
                "type": "string",
                "description": "Gene symbol (e.g., 'PCSK9')",
                "required": True,
            },
            "resources": {
                "type": "string",
                "description": "Comma-separated resources: 'marderstein' (Marderstein/Kundaje 2026 ChromBPNet + FLARE predictions). Omit to search all.",
            },
            "window": {
                "type": "integer",
                "description": "Flank in bp added on each side of the gene body (default 500000).",
                "default": 500000,
                # executor.py sql_int(window, minimum=0, maximum=ToolExecutor._MAX_SQL_WINDOW)
                "minimum": 0,
                "maximum": 10_000_000,
            },
        },
    },
    {
        "name": "get_mpra_by_variant",
        "category": "api",
        "sdk_replaceable": True,
        "description": "Get MEASURED cis-regulatory allelic activity for a variant from a massively parallel reporter assay (MPRA; Siraj et al. 2026). Answers 'does this variant's allele actually change reporter/enhancer activity, and in which cell lines?'. Returns one LONG row per cell_line: cell_line is 'meta' (cross-cell-line meta-analysis summary) or one of K562/HEPG2/SKNSH/HCT116/A549. Key calls per row: emVar (allele modulates reporter expression — allelic skew significant), active (element drives reporter above background); plus log2Skew (signed allelic effect log2(alt/ref), positive = alt drives higher expression), log2FC (element activity), log2Skew_mlog10p/log2FC_mlog10p (significance), mean_RNA_ref/alt (per-line reporter levels). MPRA MEASURES intrinsic cis-regulatory allelic activity — distinct from in-silico variant_effect (ChromBPNet/FLARE) PREDICTIONS and from endogenous eQTL/caQTL. emVar rate and allelic-effect concordance scale with FinnGen fine-mapping PIP, so this corroborates that a fine-mapped/credible-set variant is functionally active. Coverage is partial (fine-mapped GTEx/UKBB/BBJ + control common variants; absence != no effect).",
        "parameters": {
            "variant": {
                "type": "string",
                "description": "Variant as chr:pos:ref:alt or chr:pos (e.g., '1:1000500:A:G' or '1:1000500')",
                "required": True,
            },
            "resources": {
                "type": "string",
                "description": "Comma-separated resources: 'siraj_mpra' (Siraj et al. 2026 MPRA of 221K fine-mapped + 86K control variants in 5 cell lines). Omit to search all.",
            },
        },
    },
    {
        "name": "get_mpra_by_region",
        "category": "api",
        "sdk_replaceable": True,
        "description": "Get MEASURED cis-regulatory allelic MPRA activity (Siraj et al. 2026) for variants overlapping a genomic region. Answers 'which variants in this region have allele-modulating (emVar) or active regulatory elements, and in which cell lines?'. Returns LONG rows (one per variant per cell_line): cell_line is 'meta' (cross-cell-line summary) or one of K562/HEPG2/SKNSH/HCT116/A549; emVar (allelic skew significant — the key call), active (element drives reporter above background), log2Skew (signed allelic effect log2(alt/ref)), log2FC (element activity), *_mlog10p significance, mean_RNA_ref/alt. MPRA MEASURES intrinsic cis-regulatory allelic activity — distinct from in-silico variant_effect (ChromBPNet/FLARE) PREDICTIONS and from endogenous eQTL/caQTL; emVar rate/effect concordance scale with FinnGen fine-mapping PIP. Coverage is partial (fine-mapped GTEx/UKBB/BBJ + control common variants; absence != no effect).",
        "parameters": {
            "chrom": {
                "type": "string",
                "description": "Chromosome (e.g., '1', 'chr1', 'X')",
                "required": True,
            },
            "start": {
                "type": "integer",
                "description": "Region start position (1-based, inclusive)",
                "required": True,
            },
            "end": {
                "type": "integer",
                "description": "Region end position (1-based, inclusive)",
                "required": True,
            },
            "resources": {
                "type": "string",
                "description": "Comma-separated resources: 'siraj_mpra'. Omit to search all.",
            },
        },
    },
    {
        "name": "get_mpra_by_gene",
        "category": "api",
        "sdk_replaceable": True,
        "description": "Get MEASURED cis-regulatory allelic MPRA activity (Siraj et al. 2026) for variants near a gene, selected by genomic coordinates (gene body ± window, not most-severe-consequence attribution which misses nearby regulatory variants). Answers 'which of this gene's variants actually modulate reporter/enhancer activity (emVar), how strongly, and in which cell lines?'. Returns LONG rows (one per variant per cell_line): cell_line is 'meta' (cross-cell-line summary) or one of K562/HEPG2/SKNSH/HCT116/A549; emVar (allelic skew significant — the key call), active (element drives reporter above background), log2Skew (signed allelic effect log2(alt/ref)), log2FC (element activity), *_mlog10p significance, mean_RNA_ref/alt. MPRA MEASURES intrinsic cis-regulatory allelic activity — distinct from in-silico variant_effect (ChromBPNet/FLARE) PREDICTIONS and from endogenous eQTL/caQTL; emVar rate/effect concordance scale with FinnGen fine-mapping PIP, so this corroborates functionally active fine-mapped variants. Coverage is partial (fine-mapped GTEx/UKBB/BBJ + control common variants; absence != no effect).",
        "parameters": {
            "gene": {
                "type": "string",
                "description": "Gene symbol (e.g., 'PCSK9')",
                "required": True,
            },
            "resources": {
                "type": "string",
                "description": "Comma-separated resources: 'siraj_mpra'. Omit to search all.",
            },
            "window": {
                "type": "integer",
                "description": "Flank in bp added on each side of the gene body (default 500000).",
                "default": 500000,
                # executor.py sql_int(window, minimum=0, maximum=ToolExecutor._MAX_SQL_WINDOW)
                "minimum": 0,
                "maximum": 10_000_000,
            },
        },
    },
    {
        "name": "get_mpra_pip_concordance_by_gene",
        "category": "api",
        "sdk_replaceable": True,
        "description": "Cross-reference FinnGen fine-mapped credible-set PIP against MEASURED MPRA emVar calls for variants near a gene — the core regulatory-buffering check (Kanai et al.): do high-PIP (credibly causal) fine-mapped variants actually show measured cis-regulatory allelic activity (emVar) in MPRA? Joins credible_sets_v (FinnGen fine-mapped, filtered to resource + pip>=min_pip) to the MPRA cross-cell-line meta row (mpra_v.cell_line='meta') on the shared chr:pos:ref:alt variant key. Per matched variant returns: FinnGen PIP, cs_id, trait, data_type, GWAS mlog10p/beta, and the meta MPRA call — emVar (allele modulates reporter expression), active (element drives reporter above background), log2Skew (signed allelic effect log2(alt/ref)), log2Skew_mlog10p (skew significance), log2FC (element activity), cohort. Ordered emVar then PIP. This corroborates whether fine-mapped variants are FUNCTIONALLY active in a reporter assay — MPRA measures intrinsic cis-regulatory allelic activity, distinct from in-silico variant_effect predictions and endogenous eQTL/caQTL. Distinct from get_mpra_by_gene, which returns MPRA rows WITHOUT the PIP cross-reference. FinnGen-credible-set-based and meta-row-based by default; MPRA coverage is partial (fine-mapped GTEx/UKBB/BBJ + control common variants).",
        "parameters": {
            "gene": {
                "type": "string",
                "description": "Gene symbol (e.g., 'PCSK9')",
                "required": True,
            },
            "window": {
                "type": "integer",
                "description": "Flank in bp added on each side of the gene body (default 500000).",
                "default": 500000,
                # executor.py sql_int(window, minimum=0, maximum=ToolExecutor._MAX_SQL_WINDOW)
                "minimum": 0,
                "maximum": 10_000_000,
            },
            "resource": {
                "type": "string",
                "description": "Fine-mapping resource in credible_sets_v to cross-reference (default 'finngen').",
                "default": "finngen",
            },
            "min_pip": {
                "type": "number",
                "description": "Minimum posterior inclusion probability (PIP) to include, so results focus on credibly causal variants (default 0.1).",
                "default": 0.1,
                # executor.py sql_float(min_pip, minimum=0.0, maximum=1.0)
                "minimum": 0.0,
                "maximum": 1.0,
            },
        },
    },
    {
        "name": "get_gene_disease_associations",
        "category": "api",
        "sdk_replaceable": True,
        "description": "Get Mendelian/rare disease gene-disease relationships from GenCC curation submissions (ClinGen, Genomics England PanelApp, Orphanet and other panels) and the Monarch Initiative knowledge graph (OMIM, Orphanet, ClinGen). One row per source assertion, so a gene carries several rows per disease and they need not agree. 'classification' is GenCC's validity term (Definitive, Strong, Moderate, Limited, Disputed Evidence, Refuted Evidence, Supportive, No Known Disease Relationship) on gencc rows and the Biolink predicate (causes, gene_associated_with_condition, contributes_to, associated_with_increased_likelihood_of) on monarch rows, so weigh the two vocabularies separately; 'mode_of_inheritance' is GenCC-only. Use ONLY for rare disease genetics questions, NOT for GWAS/common variant associations.",
        "parameters": {
            "gene": {"type": "string", "description": "Gene symbol or comma-separated list of gene symbols", "required": True},
        },
    },
    {
        "name": "get_colocalization",
        "category": "api",
        "sdk_replaceable": True,
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
        "name": "get_colocalization_by_credible_set",
        "category": "api",
        "sdk_replaceable": True,
        "description": "Get the credible sets that colocalize with ONE specific credible set, identified by resource + phenotype + cs_id. Use this after get_credible_sets_by_gene/_by_variant/_by_region has given you a cs_id and you want that signal's colocalizations specifically — get_colocalization takes a variant and returns everything colocalizing at the position, which mixes in other signals at the same locus.",
        "parameters": {
            "resource": {
                "type": "string",
                "description": "Data resource of the credible set (e.g. 'finngen')",
                "required": True,
            },
            "phenotype": {
                "type": "string",
                "description": "Phenotype or study code of the credible set (e.g. 'K11_IBD_STRICT')",
                "required": True,
            },
            "credible_set_id": {
                "type": "string",
                "description": "Credible set ID (e.g. 'chr1:65744548-68744548_3')",
                "required": True,
            },
            "dual_format": {
                "type": "boolean",
                "description": "If true, return columns for both traits of each colocalizing pair instead of the compact single-trait view.",
                "default": False,
            },
        },
    },
    {
        "name": "get_exome_results_by_gene",
        "category": "api",
        "sdk_replaceable": True,
        "description": "Get rare variant burden test results for a gene. Returns individual variant-level association statistics from exome sequencing across available resources (genebass/UKBB filtered to p<1e-4, IBD exome containing only exome-wide significant variants). Use this for single-gene queries. For batch queries across many genes, use the database instead (call get_database_schema to find the exome results table). For full individual-trait results, use get_exome_results_by_phenotype.",
        "parameters": {
            "gene": {"type": "string", "description": "Gene symbol or comma-separated list of gene symbols", "required": True},
        },
    },
    {
        "name": "get_exome_results_by_variant",
        "category": "api",
        "sdk_replaceable": True,
        "description": "Get rare-variant exome association results for one specific variant across exome resources (genebass/UKBB filtered to p<1e-4, IBD exome exome-wide significant). Use this to check whether a named coding variant has a rare-variant association, as the counterpart to get_credible_sets_by_variant for GWAS. For a gene use get_exome_results_by_gene.",
        "parameters": {
            "variant": {
                "type": "string",
                "description": "Variant ID as chr:pos:ref:alt (e.g. '19:44908684:T:C')",
                "required": True,
            },
            "resources": {
                "type": "string",
                "description": "Comma-separated exome resources (e.g. 'genebass', 'ibd_exome_2026'). Omit to search all.",
            },
        },
    },
    {
        "name": "get_exome_results_by_region",
        "category": "api",
        "sdk_replaceable": True,
        "description": "Get rare-variant exome association results overlapping a genomic region across exome resources. Use this when the locus is coordinates rather than a gene — e.g. checking whether a GWAS interval also carries rare-variant signal. For a single gene use get_exome_results_by_gene. Rows are capped at 500 inline; `truncated` says whether more exist and the full result is at `_download_url`.",
        "parameters": {
            "region": {
                "type": "string",
                "description": "Region as chr:start-end (e.g. '1:1000000-1500000'). Max 10Mb.",
                "required": True,
            },
            "resources": {
                "type": "string",
                "description": "Comma-separated exome resources (e.g. 'genebass', 'ibd_exome_2026'). Omit to search all.",
            },
        },
    },
    {
        "name": "get_exome_results_by_phenotype",
        "category": "api",
        "sdk_replaceable": True,
        "description": "Get individual variant exome results for a specific phenotype within an exome dataset. Returns the full set of variant-level results for one trait from a given resource (e.g. genebass, ibd_exome_2026). Use this when you need all exome variants for a particular phenotype rather than a gene-centric view.",
        "parameters": {
            "resource": {
                "type": "string",
                "description": "Exome data resource (e.g. 'genebass', 'ibd_exome_2026')",
                "required": True,
            },
            "phenotype": {
                "type": "string",
                "description": "Phenotype or study code (e.g. 'categorical_41210_both_sexes_S068_', 'IBD')",
                "required": True,
            },
        },
    },
    {
        "name": "get_gene_based_results",
        "category": "api",
        "sdk_replaceable": True,
        "description": "Get gene-level burden test results from genebass, IBD, BipEx2, and SCHEMA datasets. Returns gene-based association statistics aggregated at the gene level. Different from get_exome_results_by_gene which returns individual variant-level exome results. genebass rows here are limited to p<1e-4; for a gene's result in a specific trait regardless of significance use get_gene_based_results_by_phenotype, or the gene_burden_results table in the database (unfiltered) for batch queries across many genes or traits.",
        "parameters": {
            "gene": {
                "type": "string",
                "description": "Gene symbol or comma-separated list of gene symbols (e.g., 'APOE', 'BRCA1,TP53')",
                "required": True,
            },
        },
    },
    {
        "name": "get_gene_based_results_by_phenotype",
        "category": "api",
        "sdk_replaceable": True,
        "description": "Get the complete, unfiltered gene burden test results for one phenotype: every gene and annotation class tested in that trait, with no p-value cutoff. Use this to check whether a gene was tested in a trait and what the result was even when it is not significant, or to rank all genes within one trait. For a gene across many traits use get_gene_based_results instead.",
        "parameters": {
            "resource": {
                "type": "string",
                "description": "Gene-based data resource ('genebass', 'schema', 'bipex', 'ibd')",
                "required": True,
            },
            "phenotype": {
                "type": "string",
                "description": "Phenotype or study code (e.g. 'categorical_41210_both_sexes_S068_', 'schizophrenia', 'bipolar_disorder', 'inflammatory_bowel_disease'). These are trait_original values from the burden results, which for IBD spell the disease out rather than using the IBD/UC/CD codes the exome variant results use",
                "required": True,
            },
        },
    },
    {
        "name": "get_phenotype_report", # TODO WHEN DISCUSSING SAMPLE SIZE, INCLUDE NUMBERS OF CASES AND CONTROLS
        "category": "api",
        "sdk_replaceable": True,
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
        "sdk_replaceable": True,
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
        "name": "list_datasets",
        "category": "general",
        # False for the reason the entity lookups are: a catalogue question is answered
        # before any script is worth writing. With this True, "what X data do we have?"
        # cost the code surface five SQL scripts surveying views one by one (100 s against
        # 50 s on the no-code surface, benchmark2-20260908) because the model did not
        # reach for `genetics.datasets()`.
        "sdk_replaceable": False,
        "description": (
            "List all datasets available in the API with descriptions, provenance "
            "(author, version, publication date), sample-size statistics (number of "
            "phenotypes, median sample size, case/control ranges), and which products "
            "(credible sets / summary stats / colocalization) each dataset supports. "
            "ALWAYS call this FIRST when the user asks about data availability, sample "
            "sizes, number of endpoints/phenotypes, dataset metadata, or mentions a "
            "data source by name. The returned `dataset_id` and `resource` are what "
            "you pass to downstream tools. For datasets marked `collection: true` "
            "(e.g. eQTL Catalogue), sub-studies are enumerated in "
            "/resource_metadata/{resource} (link in `metadata_endpoint`)."
        ),
        "parameters": {
            "resource": {
                "type": "string",
                "description": "Optional: filter to a specific resource (e.g. 'finngen', 'eqtl_catalogue'). Omit to list all.",
            },
            "include_stats": {
                "type": "boolean",
                "description": "Include aggregate sample-size stats. Default true.",
            },
        },
    },
    {
        "name": "get_resource_metadata",
        "category": "general",
        # the per-trait half of the catalogue; see list_datasets
        "sdk_replaceable": False,
        "description": "Get the harmonized per-trait metadata of one resource: every phenotype/study it serves with its trait name, sample sizes and (for collections like eQTL Catalogue) the sub-studies. Use this after list_datasets when the question is about a resource's contents — which traits exist, how many, what a trait code means, or how large a study is. list_datasets gives dataset-level aggregates; this gives the per-trait rows behind them.",
        "parameters": {
            "resource": {
                "type": "string",
                "description": "Resource name (e.g. 'finngen', 'eqtl_catalogue')",
                "required": True,
            },
        },
    },
    {
        "name": "get_dataset_display_names",
        "category": "general",
        "sdk_replaceable": True,
        "description": "Get the display-name overrides for raw `dataset` column values. Use this when a `dataset` value in a result (e.g. 'FinnGen_R13') needs to be rendered as its human-readable name in an answer, table or figure.",
        "parameters": {},
    },
    {
        "name": "get_credible_sets_stats",
        "category": "api",
        "sdk_replaceable": True,
        "description": "Get summary statistics of credible sets (fine-mapped associations) for a dataset. Returns counts of risk and protective credible sets, including those with coding/LoF variants. Use this to answer questions like 'how many protective associations in FinnGen Kanta?' CRITICAL: Your response MUST include the INCLUDE_IN_RESPONSE field value verbatim - it contains a download link the user needs.",
        "parameters": {
            "resource_or_dataset": {
                "type": "string",
                "description": "Resource name or dataset_id. Call list_datasets to see available dataset_ids and their resources.",
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
        "sdk_replaceable": True,
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
        "sdk_replaceable": True,
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
        "sdk_replaceable": False,
        "description": (
            "Search scientific literature for research papers about genes, variants, diseases, or biological mechanisms. "
            "Each call queries exactly ONE backend API: either 'europepmc' OR 'perplexity' — never both. "
            "You do NOT choose the backend and there is no parameter for it: the backend is set by the user's own setting "
            "(defaulting to 'perplexity'), and the user can change it if they want a different one. "
            "These two backends are distinct APIs, not interchangeable labels for the same source:\n"
            "- 'europepmc' backend: queries the Europe PMC API, which indexes PubMed, Europe PMC, bioRxiv, and medRxiv. Returns structured paper records.\n"
            "- 'perplexity' backend: queries the Perplexity AI API, which searches a broader configured set of scientific web domains and returns an AI-generated summary with citations.\n"
            "When reporting results to the user, name the backend that was actually queried: the 'backend' field in the response, which is authoritative. "
            "Do NOT invent hybrid labels like 'PubMed/Europe PMC' or 'Perplexity/PubMed' — PubMed etc. are content indexed by the europepmc backend, not separate backends. "
            "Perplexity hits carry bibliographic metadata (authors, journal) looked up in Europe PMC where a PMID/DOI/PMCID was available; that is recorded per record in 'metadata_source' and does not change which backend was searched."
        ),
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
                "description": "Include bioRxiv/medRxiv preprints (default true). Only affects the 'europepmc' backend.",
                "default": True,
            },
            "date_range": {
                "type": "string",
                "description": "Optional date filter: 'last_year', 'last_5_years', or 'YYYY-YYYY' range",
            },
        },
    },
    {
        "name": "web_search",
        "category": "general",
        "sdk_replaceable": False,
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
                # executor.web_search caps every backend with min(max_results, 10); no floor is applied
                "maximum": 10,
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
        "name": "search_mgi",
        "category": "general",
        "sdk_replaceable": False,
        "description": "Search Jackson Lab Mouse Genome Informatics (MGI) for curated mouse gene → phenotype annotations (MP ontology), knockout/transgenic allele phenotypes, and human-mouse ortholog mappings. Returns structured records (not papers). Complements search_scientific_literature — use it for mouse KO / phenotype / MP-ontology / ortholog questions.",
        "parameters": {
            "query": {
                "type": "string",
                "description": "Gene symbol (human or mouse), phenotype term, or MGI ID, depending on query_type.",
                "required": True,
            },
            "query_type": {
                "type": "string",
                "description": "What to look up: 'gene_phenotypes' (gene → MP phenotype terms + alleles), 'phenotype_genes' (MP term → genes), 'allele' (allele details), or 'ortholog' (mouse-human ortholog mapping).",
                "enum": ["gene_phenotypes", "phenotype_genes", "allele", "ortholog"],
                "default": "gene_phenotypes",
            },
            "species": {
                "type": "string",
                "description": "Species of the input query: 'mouse' or 'human' (used to set ortholog lookup direction). Default 'mouse'.",
                "enum": ["mouse", "human"],
                "default": "mouse",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum records to return (default 25, max 100).",
                "default": 25,
                # executor.search_mgi: size = max(1, min(max_results, 100))
                "minimum": 1,
                "maximum": 100,
            },
        },
    },
    {
        "name": "search_cbioportal",
        "category": "general",
        "sdk_replaceable": False,
        "description": """Query cBioPortal for how often a gene is somatically altered in cancer: pan-cancer mutation and copy-number frequency, the breakdown by cancer type, recurrent protein changes (hotspots), and fusion partners. Covers ~540 studies and ~400,000 tumour samples. Returns structured counts, not papers.

This is somatic tumour data. It says nothing about germline association — do not read a high mutation frequency here as evidence for a GWAS or disease-association claim, and do not read the absence of a gene as evidence against one.

GENOME BUILD — read before quoting any coordinate. cBioPortal reports each record on its source study's build, which is GRCh37 for most studies, and does not lift over. This suite is GRCh38. Never compare a coordinate from this tool against a GRCh38 position. Gene symbols and protein changes ARE build-independent, so match on those. Coordinates are returned grouped under the build they came from and are never merged across builds. To start from a GRCh38 variant, call get_variant_protein_effect first to get its protein change, then query here by protein change or residue.

Examples:
- How often is a gene mutated in cancer at all: search_cbioportal(query='PCSK9', query_type='gene_summary')
- Which cancers it is mutated in: search_cbioportal(query='EGFR', query_type='gene_by_cancer_type')
- Just lung and glioma: search_cbioportal(query='EGFR', query_type='gene_by_cancer_type', cancer_types=['Non-Small Cell Lung Cancer', 'Glioma'])
- Hotspot residues: search_cbioportal(query='TP53', query_type='gene_mutations')
- Recurrence at one residue: search_cbioportal(query='TP53 R175H', query_type='variant_hotspot')
- Fusion partners: search_cbioportal(query='ALK', query_type='gene_fusions')

Frequencies from gene_by_cancer_type are lower bounds: their denominator counts every sample with mutation data, including samples sequenced on gene panels that omit this gene. gene_summary reports the panel-aware profiled count and a not_profiled_samples figure — check it before treating a per-cancer-type frequency as exact.""",
        "parameters": {
            "query": {
                "type": "string",
                "description": "A gene symbol for the gene_* query types; 'GENE RESIDUE' (e.g. 'TP53 R175H' or 'TP53 175') for variant_hotspot; a free-text term for study_search.",
                "required": True,
            },
            "query_type": {
                "type": "string",
                "description": "What to look up: 'gene_summary' (pan-cancer mutation + copy-number frequency), 'gene_by_cancer_type' (frequency per cancer type), 'gene_mutations' (recurrent protein changes / hotspots), 'gene_fusions' (structural-variant partners), 'variant_hotspot' (sample count at one residue), or 'study_search' (find studies).",
                "enum": [
                    "gene_summary",
                    "gene_by_cancer_type",
                    "gene_mutations",
                    "gene_fusions",
                    "variant_hotspot",
                    "study_search",
                ],
                "default": "gene_summary",
            },
            "cancer_types": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional, gene_by_cancer_type only: restrict to these cancer types by name (matched case- and punctuation-insensitively, e.g. 'Non-Small Cell Lung Cancer'). Omit to rank all of them.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum records to return (default 25, max 100).",
                "default": 25,
                # executor.search_cbioportal: size = max(1, min(max_results, 100))
                "minimum": 1,
                "maximum": 100,
            },
        },
    },
    {
        "name": "get_protein_annotations",
        "category": "general",
        "sdk_replaceable": False,
        "description": """Get curated protein annotations from UniProt: residue-level features (active sites, binding sites, domains, disulfide bonds, signal peptides, PTMs), function and subcellular location comments, cross-references, and optionally the amino-acid sequence.

ALWAYS prefer a gene symbol over an accession. Do NOT pass an accession you remember — remembered accessions are frequently wrong and will silently annotate the wrong protein. Pass query='PRSS55', not query='Q7Z5A4'. Only pass an accession the user supplied or that a previous tool result returned.

Every result carries a resolution block naming the protein that was actually annotated (accession, entry name, protein name, gene names, organism, reviewed status, whether the match was ambiguous). Read it before citing anything: if it names a protein other than the one you meant, the annotations are not about your protein.

Examples:
- Catalytic triad of a serine protease: get_protein_annotations(query='PRSS55', feature_types=['ACT_SITE', 'BINDING'])
- Domain layout of a huge protein: get_protein_annotations(query='TTN', include=['features'], feature_types=['DOMAIN'])
- Function plus sequence: get_protein_annotations(query='TPO', include=['function', 'sequence'])
- Just the features in one region: get_protein_annotations(query='TTN', feature_types=['DOMAIN'], residue_range='1-2000')

Do NOT use this tool for protein-position → genomic-coordinate mapping — use map_protein_variants. Do NOT use it to find which proteins share a property — use search_uniprot.""",
        "parameters": {
            "query": {
                "type": ["string", "array"],
                "items": {"type": "string"},
                "description": "Gene symbol (strongly preferred, e.g. 'TPO', 'PRSS55'), UniProt entry name, or accession. Never supply an accession recalled from memory when a gene symbol is available. PASS A LIST to annotate many proteins in one call — up to 100 — rather than calling once per protein. A list answers with a flat `results` row per input, each row carrying its own identity and match_basis so a row can never be attributed to the wrong protein.",
                "required": True,
            },
            "organism_id": {
                "type": "integer",
                "description": "NCBI taxon ID to restrict symbol resolution to (default 9606, human). Use 10090 for mouse. Pass null to search all organisms.",
                "default": 9606,
            },
            "include": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Annotation sections to return (default ['features', 'function']). 'sequence' returns the full amino-acid sequence and can be very large for proteins like TTN.",
                "enum": ["features", "function", "sequence", "xrefs"],
                "default": ["features", "function"],
            },
            "feature_types": {
                "type": "array",
                "items": {"type": "string"},
                "description": "UniProt feature-type keys to keep, e.g. ['ACT_SITE', 'BINDING', 'DOMAIN', 'DISULFID', 'SIGNAL', 'MOD_RES', 'VARIANT']. Omit for all feature types. Essential for large proteins.",
            },
            "residue_range": {
                "type": "string",
                "description": "Restrict features to a residue window of the canonical sequence, as 'start-end' in 1-based protein coordinates (e.g. '1-2000').",
            },
        },
    },
    {
        "name": "map_protein_variants",
        "category": "general",
        "sdk_replaceable": False,
        "description": """Map protein-level variants (amino-acid substitutions such as 'P70A') onto genomic coordinates, using UniProt's curated genomic coordinate mapping. Returns, per variant, the genome position, reference and alternate alleles, the codon, the transcript/exon context, and any matching curated UniProt VARIANT annotation (including disease association and dbSNP rsID when UniProt records one).

This is the tool for "what is the rs ID / genomic position of this amino-acid change?". Do NOT guess candidate genomic coordinates and test them one at a time — that approach has failed here before. Do NOT use get_variant_annotations or get_myvariant_annotations first: they take genomic coordinates, which is exactly what this tool produces. Feed the coordinates or rsIDs it returns into those tools afterwards for allele frequencies and clinical significance.

Canonical example — four thyroid peroxidase substitutions in one call:
  map_protein_variants(variants=['P70A', 'G393A', 'R438H', 'W873C'], query='TPO')

Pass the gene symbol, not an accession you remember. A wrong accession maps every variant against the wrong sequence and produces confidently wrong coordinates. Accepted variant notations: 'P70A', 'Pro70Ala', 'p.Pro70Ala'. The position is a 1-based residue index into the canonical UniProt sequence.

Every result carries a resolution block naming the protein the variants were mapped against, plus a per-variant check that the reference amino acid matches that sequence. A reference mismatch means the variant is not on this isoform (or not on this protein) — do not report its coordinates as if it were.""",
        "parameters": {
            "variants": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Amino-acid substitutions, e.g. ['P70A', 'G393A', 'R438H', 'W873C']. One-letter ('P70A'), three-letter ('Pro70Ala') and HGVS protein ('p.Pro70Ala') notation all accepted. Batch them in a single call rather than one call per variant.",
                "required": True,
            },
            "query": {
                "type": "string",
                "description": "Gene symbol of the protein the variants belong to (strongly preferred, e.g. 'TPO'), or a UniProt accession the user supplied. Never an accession recalled from memory.",
                "required": True,
            },
            "organism_id": {
                "type": "integer",
                "description": "NCBI taxon ID for symbol resolution (default 9606, human). Genomic coordinate mapping is only available for organisms UniProt maps to a reference genome.",
                "default": 9606,
            },
        },
    },
    {
        "name": "get_variant_protein_effect",
        "category": "general",
        "sdk_replaceable": False,
        "description": """Map genomic coding variants onto their curated UniProt protein consequence. This is the genomic→protein direction: feed a `chr:pos:ref:alt` variant and get back the amino-acid change plus UniProt's curated annotation for it — disease association, clinical significance, population frequency and dbSNP/ClinVar cross-references.

This is the tool for "what does this coding variant do to the protein, and what is known about it?". Use it instead of asserting an amino-acid change (e.g. G2019S) from memory: the residue change, disease link and clinical significance all come from UniProt/ClinVar, not from the reference sequence or recall.

Canonical example:
  get_variant_protein_effect(variants=['12:40340400:G:A'])  → LRRK2 p.Gly2019Ser, missense, ClinVar Pathogenic, Parkinson disease 8 (PARK8), gnomAD AF.

Batch variants in one call. Assembly is GRCh38 (variant ids are matched against the GRCh38 RefSeq chromosomes). Only reviewed (Swiss-Prot) entries and their isoforms are reported; canonical first.

Scope and limits:
- Single-nucleotide substitutions only. An indel or MNV comes back with a note that it is unsupported here — do not read that as "no effect". For those, use map_protein_variants (protein→genomic) or get_myvariant_annotations.
- A variant with no coding consequence (intronic, intergenic, or simply not annotated on a reviewed entry) returns an explicit note, not an error.
- Already have an amino-acid change and want its genomic coordinate/rsID instead? That is the opposite direction — use map_protein_variants.""",
        "parameters": {
            "variants": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Genomic SNVs as 'chr:pos:ref:alt' on GRCh38, e.g. ['12:40340400:G:A', '19:55014977:T:G']. A leading 'chr' is accepted. Batch them in a single call.",
                "required": True,
            },
        },
    },
    {
        "name": "search_uniprot",
        "category": "general",
        "sdk_replaceable": False,
        "description": """Search UniProtKB with its native query syntax to find the set of proteins matching a property — a keyword, a family, a subcellular location, a function. Returns one summary row per entry (accession, entry name, protein name, gene names, organism, reviewed status) plus whatever extra fields you request.

Use this when the question is "which proteins ...?" rather than "what about this protein?" (that is get_protein_annotations).

Examples:
- Count reviewed human proteins with a keyword: search_uniprot(keyword='KW-0865', count_only=True)
- Enumerate them with lengths: search_uniprot(keyword='KW-0865', fields='accession,id,gene_names,length', size=100)
- Free-text plus a structured clause: search_uniprot(query='thyroid peroxidase AND family:peroxidase')
- Non-human: search_uniprot(query='gene:Tpo', organism_id=10090)

`query` is passed to UniProt as-is, so field clauses (gene:, family:, cc_scl_term:, ec:, length:[100 TO 200]) and boolean operators work. organism_id and reviewed_only are added as separate clauses — do not also write them into `query`.

Do NOT use this to look up a protein you can already name; resolving a gene symbol is what get_protein_annotations and map_protein_variants do for you. Never cite a UniProt accession from memory — if you need one, get it from this tool's output.""",
        "parameters": {
            "query": {
                "type": "string",
                "description": "UniProtKB query string, free text or native field syntax (e.g. 'family:peroxidase', 'cc_scl_term:SL-0173 AND length:[500 TO *]'). Provide query or keyword or both.",
            },
            "keyword": {
                "type": "string",
                "description": "UniProt keyword ID (e.g. 'KW-0865') or keyword name, added as a keyword: clause. Provide query or keyword or both.",
            },
            "organism_id": {
                "type": "integer",
                "description": "NCBI taxon ID restricting the search (default 9606, human). Pass null to search all organisms.",
                "default": 9606,
            },
            "reviewed_only": {
                "type": "boolean",
                "description": "Restrict to reviewed Swiss-Prot entries (default true). Set false to include unreviewed TrEMBL entries, which are far more numerous and not manually curated.",
                "default": True,
            },
            "fields": {
                "type": "string",
                "description": "Comma-separated UniProt return fields (default 'accession,id,protein_name,gene_names,organism_name'). Add e.g. 'length,cc_function,ft_act_site' for more per-entry detail.",
                "default": "accession,id,protein_name,gene_names,organism_name",
            },
            "size": {
                "type": "integer",
                "description": "Maximum entries to return (default 25, max 500). Use count_only first when the set may be large.",
                "default": 25,
                # uniprot.py: size = max(1, min(int(size or _DEFAULT_SEARCH_SIZE), 500)) —
                # no `minimum` here: `size=0` is falsy, so `size or _DEFAULT_SEARCH_SIZE`
                # substitutes the default (25) before the floor ever runs, so size=0
                # returns 25 rows, not 1. A `minimum: 1` declaration would describe a
                # clamp this code does not actually apply.
                "maximum": 500,
            },
            "count_only": {
                "type": "boolean",
                "description": "Return only the total number of matching entries, no rows. Cheap way to size a query before enumerating it.",
                "default": False,
            },
        },
    },
    {
        "name": "get_drug_targets_for_gene",
        "category": "general",
        "sdk_replaceable": False,
        "description": """List the drugs and clinical candidates ChEMBL records as acting on a gene's protein target, with each drug's mechanism of action, action type (INHIBITOR, AGONIST, ANTAGONIST, ...), highest clinical phase reached, first approval year, withdrawal flag, ATC codes, and — only with `include_indications=True` — the indications they are developed for, at most 10 per drug with `n_indications` giving the true total.

Use this before calling any gene a promising or novel drug target, and whenever the user asks about drugs, druggability, inhibitors, agonists, repurposing, or clinical phase for a gene. If approved drugs or clinical candidates already exist, say so and frame the finding as supporting a known mechanism rather than as a new opportunity.

`max_phase` is ChEMBL's highest phase reached ANYWHERE, by any regulator, for any indication: 4 means approved somewhere in the world, NOT "FDA-approved" — never write "FDA-approved" on the strength of this field. 0 to 3 are preclinical and clinical stages — 0 is a phase ChEMBL records, distinct from None, which means no phase recorded: unknown rather than zero. `mechanism_max_phase` is the phase of that specific mechanism annotation when it differs from the molecule's.

`query` is a gene, never a drug name: a gene symbol (preferred), a UniProt accession, or a `CHEMBL<number>` target id. A symbol or accession is resolved through UniProt, then to the human ChEMBL target sharing that accession; the SINGLE PROTEIN target is chosen where one exists. Check which target answered before quoting the result — `target_chembl_id`, `target_pref_name` and `target_type` name it, `other_targets` lists any others sharing the accession, and `resolution` carries the `accession`, `n_targets` and a `note`. A gene with no ChEMBL target is a normal result with `count` 0, not an error.

Examples:
- Does anything drug this gene: get_drug_targets_for_gene(query='PCSK9')
- Approved drugs only, with what they treat: get_drug_targets_for_gene(query='IL6R', min_phase=4, include_indications=True)

NEVER cite a ChEMBL id, max_phase, mechanism or indication from memory — they must come from a tool result in this conversation. Every successful result carries an `attribution` line; include it when citing ChEMBL content.

For one named drug (its targets, ATC class and indications) use get_drug_profile. For how much medicinal chemistry exists against the target — potency measurements rather than drugs — use get_target_bioactivity.""",
        "parameters": {
            "query": {
                "type": ["string", "array"],
                "items": {"type": "string"},
                "description": "Gene symbol (preferred, e.g. 'PCSK9'), UniProt accession, or ChEMBL target id ('CHEMBL235'). Never an accession or ChEMBL id recalled from memory. PASS A LIST TO ASK ABOUT MANY AT ONCE — up to 50 — and do so whenever you have more than one: calling once per gene is the single most expensive mistake on this tool. A list answers in ONE call, with a flat `drugs` table whose rows each name their `query`, each gene's own resolution block under `per_query`, and `batch.no_rows_for` / `batch.failed` naming the inputs that returned nothing and the ones that failed.",
                "required": True,
            },
            "min_phase": {
                "type": "number",
                "description": "Keep only drugs whose max_phase is at least this (0 keeps everything including unknown-phase rows, 4 keeps only drugs approved somewhere). Default 0.",
                "default": 0,
                # chembl.get_drug_targets_for_gene: floor = min(4.0, max(0.0, float(min_phase)))
                "minimum": 0,
                "maximum": 4,
            },
            "include_indications": {
                "type": "boolean",
                "description": "Also fetch what each drug is developed or approved for (EFO/MeSH terms with a per-indication max phase), at most 10 per drug. Costs an extra request; default false.",
                "default": False,
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum drug rows to return, highest phase first (default 25, max 100). `n_matching` reports how many passed the phase filter before this cap.",
                "default": 25,
                # chembl.get_drug_targets_for_gene: result_cap = max(1, min(int(max_results), 100))
                "minimum": 1,
                "maximum": 100,
            },
        },
    },
    {
        "name": "get_drug_profile",
        "category": "general",
        "sdk_replaceable": False,
        "description": """Get what ChEMBL holds about one drug or compound: its preferred name and ChEMBL id, highest clinical phase, first approval year, withdrawal flag, ATC classification, the targets it acts on with mechanism of action and action type, and the indications it is developed or approved for (EFO and MeSH terms, each with its own max phase), at most 50 of them with `n_indications` giving the true total.

Use this when the user names a drug — "what does metformin target?", "what is CHEMBL1431 approved for?", "is this compound withdrawn?".

`max_phase` is the highest phase reached ANYWHERE, by any regulator, for any indication: 4 means approved somewhere in the world, NOT "FDA-approved". None means ChEMBL records no phase — unknown, not zero.

`query` is a drug, never a gene symbol: a drug name, synonym or trade name, or a `CHEMBL<number>` molecule id. Check which molecule answered before quoting the result: `resolution.kind` says how the name matched (`chembl_id`, `pref_name` or `synonym`), `drug.molecule_chembl_id` says which molecule was chosen, `resolution.n_candidates` how many matched, and `resolution.other_candidates` lists the rest. A name with no ChEMBL molecule returns `drug` None with a note, not an error.

NEVER cite a ChEMBL id, max_phase, mechanism or indication from memory — they must come from a tool result in this conversation. Every successful result carries an `attribution` line; include it when citing ChEMBL content.

Start from a gene rather than a drug — "what drugs hit this gene?" — with get_drug_targets_for_gene. For the potency measurements recorded against a target, use get_target_bioactivity.""",
        "parameters": {
            "query": {
                "type": ["string", "array"],
                "items": {"type": "string"},
                "description": "Drug name, synonym or trade name (e.g. 'metformin', 'evolocumab'), or a ChEMBL molecule id ('CHEMBL1431'). Never a ChEMBL id recalled from memory. PASS A LIST TO ASK ABOUT MANY AT ONCE — up to 50 — and do so whenever you have more than one: calling once per drug is the single most expensive mistake on this tool. A list answers in ONE call, with a flat `indications` table whose rows each name their `query`, each drug's own resolution block under `per_query`, and `batch.no_rows_for` / `batch.failed` naming the inputs that returned nothing and the ones that failed.",
                "required": True,
            },
        },
    },
    {
        "name": "get_target_bioactivity",
        "category": "general",
        "sdk_replaceable": False,
        "description": """Summarise the medicinal chemistry recorded against a gene's protein target: how many potency measurements exist at or above a pChEMBL threshold, how many distinct compounds they cover, the breakdown by assay type (IC50, Ki, EC50, ...), and the most potent compounds with their best pChEMBL value and clinical phase.

Use this for "how tractable / how well explored is this target?" — whether a chemical series exists at all, and how potent the best compounds are. pChEMBL is -log10 of the molar activity value, so 6 is 1 µM, 7 is 100 nM, 9 is 1 nM; 6 is the usual "active" cut-off.

This is a count of assay measurements, not evidence of clinical use. A target with thousands of activities may have no drug in humans, and a drugged target may have few measurements. For drugs and clinical candidates, and their phases, call get_drug_targets_for_gene; for one named drug, call get_drug_profile.

`query` is a gene, never a drug name: a gene symbol (preferred), a UniProt accession, or a `CHEMBL<number>` target id, resolved the same way as get_drug_targets_for_gene. Check which target answered before quoting the result — `target_chembl_id`, `target_pref_name` and `target_type` name it, `other_targets` lists any others sharing the accession, and `resolution` carries the `accession`, `n_targets` and a `note`. The activity walk is capped, so read `truncated` and `total_count`: when `truncated` is true, `n_activities`, `n_distinct_molecules` and `by_standard_type` count only the rows read, while `total_count` stays ChEMBL's count for the whole filter, so you can say how much was left behind.

NEVER cite a ChEMBL id, pChEMBL value or activity count from memory — they must come from a tool result in this conversation. Every successful result carries an `attribution` line; include it when citing ChEMBL content.""",
        "parameters": {
            "query": {
                "type": ["string", "array"],
                "items": {"type": "string"},
                "description": "Gene symbol (preferred, e.g. 'PPARG'), UniProt accession, or ChEMBL target id ('CHEMBL235'). Never an accession or ChEMBL id recalled from memory. PASS A LIST TO ASK ABOUT MANY AT ONCE — up to 50 — and do so whenever you have more than one: calling once per target is the single most expensive mistake on this tool. A list answers in ONE call, with a flat `top_compounds` table whose rows each name their `query`, each target's own resolution block under `per_query`, and `batch.no_rows_for` / `batch.failed` naming the inputs that returned nothing and the ones that failed.",
                "required": True,
            },
            "pchembl_min": {
                "type": "number",
                "description": "Minimum pChEMBL value to count (default 6.0, i.e. 1 µM). Raise to 7 or 8 to look only at potent compounds.",
                "default": 6.0,
                # chembl.get_target_bioactivity: threshold = min(14.0, max(0.0, float(pchembl_min)))
                "minimum": 0,
                "maximum": 14,
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum top compounds to return, best pChEMBL first (default 25, max 100). The counts and the assay-type breakdown cover every activity read, not just these.",
                "default": 25,
                # chembl.get_target_bioactivity: result_cap = max(1, min(int(max_results), 100))
                "minimum": 1,
                "maximum": 100,
            },
        },
    },
    {
        "name": "get_alphagenome_variant_predictions",
        "category": "general",
        # False for the same reason search_uniprot is: this is an OUTSIDE resource, not
        # internal genetics data the SDK can fetch, so it belongs on both surfaces by
        # `resolve_tools`' own rule. The consequence is deliberate for this phase: the
        # tool is advertised to the code-execution surface, but a SCRIPT cannot call it —
        # the sandbox egress allow-list names db-api and results-api only, and the image
        # has no `alphagenome`. The model calls the tool; the script does not.
        "sdk_replaceable": False,
        "description": f"""MODEL PREDICTIONS from AlphaGenome (Google DeepMind) — what a deep-learning model predicts one variant does to regulatory activity: chromatin accessibility, histone and TF binding, transcription, splicing, optionally in a named cell type or tissue. NOTHING HERE WAS MEASURED IN ANYONE. It is not a FinnGen result and not an assay; never present a number from this tool as either.

{_ALPHAGENOME_OPT_IN}

{_ALPHAGENOME_VALIDATION_RULES}

{_ALPHAGENOME_SIDE_BY_SIDE}""",
        "parameters": {
            "variants": {
                "type": ["string", "array"],
                "items": {"type": "string"},
                "description": "GRCh38 variants as chr:pos:ref:alt, e.g. ['19:44908684:T:C']. A leading 'chr' is accepted and X may be spelled 23. Pass a list and batch them: at most 25 per call, and one call per variant is the expensive mistake here. A variant the model cannot score comes back as its own failed row, leaving the rest of the batch intact.",
                "required": True,
            },
            "cell_type": {
                "type": "string",
                "description": "Cell type or tissue to score in, matched against AlphaGenome's own biosample names (e.g. 'liver', 'K562'). Omit to take the strongest effect across all tracks. A request that matches nothing falls back to all tracks and says so in `cell_type_match`.",
            },
            "modalities": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "DNASE",
                        "ATAC",
                        "CHIP_HISTONE",
                        "CHIP_TF",
                        "CAGE",
                        "PROCAP",
                        "RNA_SEQ",
                        "SPLICE_SITES",
                        "SPLICE_SITE_USAGE",
                        "SPLICE_JUNCTIONS",
                        "POLYADENYLATION",
                        "CONTACT_MAPS",
                    ],
                },
                "description": "Modalities to score. Omit for the default set, which is exactly the modalities calibrated against this suite's own measurements. Any modality NOT in that default is uncalibrated and has to be asked for by name; its result says so in `validation`.",
            },
        },
    },
    {
        "name": "compare_alphagenome_with_measured",
        "category": "general",
        # False for the same reason the prediction tool is: it reaches an outside model. It
        # also reads db-api, which a script CAN reach, but the AlphaGenome half it exists to
        # pair that with is unreachable from the sandbox, so half a comparison is the only
        # thing a script could build.
        "sdk_replaceable": False,
        "description": f"""MEASURED RESULTS FROM THIS SUITE PLACED BESIDE ALPHAGENOME'S PREDICTION for the same variant, per modality, with their concordance. The measured side is this suite's own data — caQTL, eQTL and sQTL effect sizes from `credible_sets_v`, MPRA allelic skew from `mpra_v` — and the predicted side is the same model output `get_alphagenome_variant_predictions` returns. Use this when someone wants to know how a prediction stands up against what was actually measured.

This is NOT for variants the suite is silent about: a comparison needs both halves, and it is worth most exactly where the measured data already exists.

{_ALPHAGENOME_OPT_IN}

HOW TO READ THE RESULT. It carries BOTH kinds of number, so the envelope has no single `measured` flag — every value inside carries its own:
- A `measured: true` value names its `source`: the view, the column, the assay, the resource, and the gene or accessibility peak that was measured. A `measured: false` value names AlphaGenome. Never merge, average or reconcile the two into one number, and never report a predicted value as a result from this suite.
- `concordance.direction` is `"agrees"` or `"disagrees"` — the two signs match, or they do not. That is the whole claim. Do NOT compute a correlation, an error or an agreement score: with one variant there is nothing to correlate, and any such number would be fiction.
- A modality whose `quantity` is `"magnitude"` has NO `direction` key at all, and both sides are reported unsigned. The absence IS the statement: a measured sQTL beta orients to a leafcutter intron cluster and the predicted splice delta has no corresponding orientation, so no direction agreement exists to report. Do not infer one, and do not describe such a pair as consistent or inconsistent in direction.
- `measured_substrates[].population_rho`, with `rho_scope: "population"`, is how well that MODALITY tracked that substrate across a cohort of variants. It is a property of the pairing and NEVER this variant's confidence.
- `context_match: "cross_tissue"` means the measurement is in a different cell type or tissue than the prediction was asked for. Say so — cell-type-matched comparisons are the stronger evidence.
- The prediction's `cell_type_match` carries two independent flags: `matched` says whether the requested cell type resolved to tracks, and `resolution_failed` says the lookup itself failed. A failed lookup is "could not be checked", not "no match" — report them differently.
- Empty `measurements` means one of three different things and the `note` says which: this suite has measured nothing for this variant; the modality has no measured substrate here at all (the unvalidated tier-4 modalities); or the measured lookup FAILED, so nothing is known either way. Never invent a comparison for the second — "nothing measured to compare against" is the answer — and never render the third as "nothing measured": it is "could not be checked", and `measured_substrates[].lookup: "failed"` names which substrate.

{_ALPHAGENOME_VALIDATION_RULES}""",
        "parameters": {
            "variants": {
                "type": ["string", "array"],
                "items": {"type": "string"},
                "description": "GRCh38 variants as chr:pos:ref:alt, e.g. ['19:44908684:T:C']. A leading 'chr' is accepted and X may be spelled 23. Pass a list and batch them: at most 25 per call.",
                "required": True,
            },
            "cell_type": {
                "type": "string",
                "description": "Cell type or tissue to compare in. It matches BOTH sides — AlphaGenome's biosample names and the measured assay's cell type or MPRA cell line (K562, HEPG2, SKNSH, HCT116, A549) — so passing it is what makes a matched comparison possible. Omit and every measurement comes back as `context_match: \"not_requested\"`.",
            },
            "modalities": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "DNASE",
                        "ATAC",
                        "CHIP_HISTONE",
                        "CHIP_TF",
                        "CAGE",
                        "PROCAP",
                        "RNA_SEQ",
                        "SPLICE_SITES",
                        "SPLICE_SITE_USAGE",
                        "SPLICE_JUNCTIONS",
                        "POLYADENYLATION",
                        "CONTACT_MAPS",
                    ],
                },
                "description": "Modalities to compare. Omit for the default set, which is exactly the modalities that HAVE a measured substrate here. A modality outside it has nothing to compare against and comes back saying so.",
            },
        },
    },
    {
        "name": "get_ld_between_variants",
        "category": "api",
        "sdk_replaceable": True,
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
        "sdk_replaceable": True,
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
        "sdk_replaceable": True,
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
                "description": "Data resource — use list_datasets to find available resources. Common values: 'finngen', 'finngen_mvp_ukbb', 'finngen_ukbb', 'pgc'",
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
        "name": "get_hla_by_phenotype",
        "category": "api",
        "sdk_replaceable": True,
        "description": """Get the classical HLA allele associations for one or more phenotypes — every imputed HLA allele (187 alleles across HLA-A, -B, -C, -DPB1, -DQA1, -DQB1, -DRB1, -DRB3, -DRB4, -DRB5) tested against the trait in FinnGen R14.

Use this whenever a question touches the MHC/HLA region:
- "Which HLA allele drives coeliac disease / T1D / ankylosing spondylitis?"
- A credible set or a strong signal lands on chr6:29-33Mb — SNP summary stats there are hard to interpret because of the extreme LD, and the allele-level result is the interpretable answer
- The user asks about HLA typing, haplotypes, or a named allele for a specific disease

The unit is an ALLELE, not a variant: there is no chr:pos:ref:alt to look up, so get_summary_stats cannot answer this. Every allele of a gene shares that gene's anchor position.

Read `mlog10p`, NOT `pval` — pval underflows to 0 for the strongest HLA signals (coeliac DQB1*02:01 is mlog10p 1596). Always check `info`: a rare allele imputed at info < 0.5 produces a huge unstable beta that is an imputation artifact, not an association.

For the reverse question — which traits an allele is associated with — use get_hla_by_allele.""",
        "parameters": {
            "phenotypes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of FinnGen endpoint codes (e.g. ['K11_COELIAC', 'T1D'])",
                "required": True,
            },
            "genes": {
                "type": "string",
                "description": "Optional comma-separated HLA gene filter, e.g. 'HLA-B,HLA-DQB1'. Omit for all 10 genes. HLA-DRB3/DRB4/DRB5 share one anchor position and always return together",
            },
            "resource": {
                "type": "string",
                "description": "Data resource carrying HLA results",
                "default": "finngen",
            },
        },
    },
    {
        "name": "get_hla_by_allele",
        "category": "api",
        "sdk_replaceable": True,
        "description": """Get every phenotype a classical HLA allele is associated with — the PheWAS view of one HLA allele across all 2,712 FinnGen R14 endpoints.

Use this when the user names an allele:
- "What is HLA-B*27:05 associated with?" / "What diseases does DQB1*02:01 predispose to?"
- You found a lead allele with get_hla_by_phenotype and want to know what else it drives (pleiotropy across autoimmune traits is the norm in the MHC)

Pass the allele gene-stripped and two-field, exactly as it appears in the data: 'B*27:05', 'DQB1*02:01', 'DRB1*15:01' — NOT 'HLA-B*27:05'.

Results are filtered to `min_info` (default 0.5) because rare badly-imputed alleles produce enormous unstable betas that look like spectacular associations; pass min_info=0 to see them. Ranked by `mlog10p`.""",
        "parameters": {
            "allele": {
                "type": "string",
                "description": "Gene-stripped two-field HLA allele name, e.g. 'B*27:05' or 'DQB1*02:01'",
                "required": True,
            },
            "min_mlogp": {
                "type": "number",
                "description": "Minimum -log10 p-value (7.3 = genome-wide significance)",
                "default": 7.3,
            },
            "min_info": {
                "type": "number",
                "description": "Minimum imputation INFO for the allele; 0 disables the filter",
                "default": 0.5,
            },
            "resource": {
                "type": "string",
                "description": "Data resource carrying HLA results",
                "default": "finngen",
            },
            "max_rows": {
                "type": "integer",
                "description": "Maximum phenotypes to return",
                "default": 200,
                # executor.py sql_int(max_rows, minimum=1, maximum=ToolExecutor._MAX_SQL_LIMIT)
                "minimum": 1,
                "maximum": 100_000,
            },
        },
    },
    {
        "name": "get_dosage_sensitivity",
        "category": "api",
        "sdk_replaceable": True,
        "description": """Get the rare-CNV dosage sensitivity scores (pHaplo, pTriplo) for one or more genes — Collins et al. 2022, the reference dosage-sensitivity map of the human genome (18,641 autosomal protein-coding genes, learned from rare CNVs in 950,278 individuals).

Use this whenever the question is about gene dosage rather than about a variant:
- "Is GENE haploinsufficient?" / "Would a deletion of GENE matter?" / "Is a third copy harmful?"
- You have a list of candidate genes and want to rank them by how badly they tolerate a copy-number change

pHaplo is the probability that ONE functional copy is not enough; pTriplo the probability that a THIRD copy is harmful. The paper's own cutoffs are pHaplo >= 0.86 (`haploinsufficient`) and pTriplo >= 0.94 (`triplosensitive`), returned as columns so you need not restate them — but rank on the probabilities, which are the continuous evidence.

This is a general, per-gene score, not a per-disease result. For "which phenotype is a deletion of this gene associated with" use get_rcnv_associations.""",
        "parameters": {
            "genes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Gene symbols or Ensembl gene IDs, e.g. ['SHANK3', 'NRXN1', 'ENSG00000251322']. Matched case-insensitively against the current symbol, the GENCODE v19 symbol the paper published, and the Ensembl ID, so an outdated gene name still resolves",
                "required": True,
            },
        },
    },
    {
        "name": "get_rcnv_associations",
        "category": "api",
        "sdk_replaceable": True,
        "description": """Get rare-CNV gene association statistics from Collins et al. 2022 — which HPO phenotype group a DELETION or DUPLICATION of a gene is associated with, across 54 phenotype groups x {DEL, DUP} x 17,263 genes.

Use this for the per-phenotype dosage question:
- "What is a deletion of NRXN1 associated with?" (pass gene=)
- "Which genes are associated with intellectual disability when duplicated?" (pass phenotype= and cnv_type='DUP')
- You found a dosage-sensitive gene with get_dosage_sensitivity and want the disease it points at

At least one of `gene` or `phenotype` is required. `phenotype` takes an HPO id in either spelling ('HP:0012759' or 'HP0012759'), the literal 'UNKNOWN', or a substring of the phenotype's name ('intellectual disability') matched case-insensitively — search_phenotypes does NOT cover this dataset, so do not try to resolve the code with it first. 'HP0000118' is every case pooled, not a peer of the other 53 groups.

`beta` is ln(odds ratio), so OR = EXP(beta). Every gene appears for every phenotype and CNV type, including the 65% of rows where the gene was TESTED BUT NO ESTIMATE was produced because no qualifying CNV was seen; those carry NULL from `beta` onward and are excluded unless you set include_no_estimate. Rank on `mlog10p`; for the paper's own gene lists set significant_only, which applies both significance tiers (FDR < 1% or P <= 2.90e-6) together with the secondary-evidence requirement (>= 2 nominal cohorts, or the leave-top-cohort-out p-value still nominally significant) — a bare threshold on mlog10p or mlog10_fdr_q does not reproduce the published results.""",
        "parameters": {
            "gene": {
                "type": "string",
                "description": "Gene symbol or Ensembl gene ID. Matched case-insensitively against the current symbol, the GENCODE v19 symbol and the Ensembl ID",
            },
            "phenotype": {
                "type": "string",
                "description": "HPO id in either spelling ('HP:0012759' or 'HP0012759'), 'UNKNOWN', or a case-insensitive substring of the phenotype name ('intellectual disability')",
            },
            "cnv_type": {
                "type": "string",
                "description": "Restrict to one CNV class: 'DEL' or 'DUP'. Omit for both",
            },
            "min_mlog10p": {
                "type": "number",
                "description": "Minimum -log10 p-value of the meta-analysis",
            },
            "max_fdr_q": {
                "type": "number",
                "description": "Maximum FDR q-value, e.g. 0.01 for the paper's FDR tier. Applied as mlog10_fdr_q >= -LOG10(max_fdr_q)",
            },
            "significant_only": {
                "type": "boolean",
                "description": "Apply the paper's full significance rule: (FDR < 1% OR P <= 2.90e-6) AND (>= 2 nominal cohorts OR secondary P < 0.05)",
                "default": False,
            },
            "include_no_estimate": {
                "type": "boolean",
                "description": "Keep the 'tested, no estimate' rows (NULL beta onward, 65% of the view). Off by default",
                "default": False,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum rows to return, ranked by mlog10p",
                "default": 200,
            },
        },
    },
    {
        "name": "get_summary_stats_by_region",
        "category": "api",
        "sdk_replaceable": True,
        "description": """Get summary statistics for EVERY variant in a genomic region for one or more phenotypes — the full association profile of a locus, not just fine-mapped or significant variants.

Use this when:
- You need all associations across an interval for a trait (e.g. to describe a locus, or to see the shape of a signal around a lead variant)
- You want to check a region for sub-threshold signal that credible sets would not include

Phenotypes are REQUIRED: summary stats are stored per phenotype, so there is no region query across all traits. For specific known variants use get_summary_stats instead — it is much cheaper. Region size is capped (5Mb here); rows are capped at 500 inline with `truncated` set, and the full result is at `_download_url`.""",
        "parameters": {
            "region": {
                "type": "string",
                "description": "Region as chr:start-end (e.g. '1:1000000-1100000'; X is accepted)",
                "required": True,
            },
            "phenotypes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of phenotype codes (e.g. ['T2D', 'I9_CHD'])",
                "required": True,
            },
            "resource": {
                "type": "string",
                "description": "Data resource — use list_datasets to find available ones. Common: 'finngen', 'finngen_mvp_ukbb', 'finngen_ukbb'",
                "default": "finngen",
            },
            "data_type": {
                "type": "string",
                "description": "Analysis data type: 'gwas', 'pqtl' or 'eqtl'",
                "default": "gwas",
            },
        },
    },
    {
        "name": "analyze_variant_list",
        "category": "api",
        "sdk_replaceable": True,
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
    {
        "name": "get_variant_annotations",
        "category": "api",
        "sdk_replaceable": True,
        "description": """Get variant annotations including allele frequency, consequence, gene, rsID, and enrichment data.

Use this tool when:
- The user asks about a variant's functional annotation (e.g., "what is the consequence of rs429358?")
- The user wants to see all variants in a gene with their annotations (e.g., "list variants in PCSK9")
- The user wants variant annotations for a genomic region
- The user needs allele frequencies, consequence types, or enrichment values for variants

Query by exactly ONE of: a single variant, a genomic region, or a gene name.
For batch lookups of multiple specific variants, use the 'variants' parameter instead.

Returns: variant ID, chromosome, position, ref/alt alleles, allele frequency (AF), heterozygous/homozygous counts, most severe consequence, gene for most severe consequence, rsID, and exome/genome enrichment values.""",
        "parameters": {
            "variant": {
                "type": "string",
                "description": "Single variant in chr:pos:ref:alt format (e.g., '1:13668:G:A'). Any separator (: - _ |) accepted.",
            },
            "region": {
                "type": "string",
                "description": "Genomic region in chr:start-end format (e.g., '1:13668-14506'). 1-based, inclusive.",
            },
            "gene": {
                "type": "string",
                "description": "Gene name (e.g., 'PCSK9', 'BRCA2'). Case-insensitive, supports HGNC aliases and ENSG IDs.",
            },
            "variants": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of variant IDs for batch lookup (e.g., ['1:13668:G:A', '1:14506:G:A']). Max 2000.",
            },
            "source": {
                "type": "string",
                "description": "Annotation source (default 'finngen')",
                "default": "finngen",
            },
        },
    },
    {
        "name": "get_myvariant_annotations",
        "category": "api",
        "sdk_replaceable": False,
        "description": """Get clinical and functional variant annotations from myvariant.info.

Use this tool when:
- The user asks about clinical significance or pathogenicity of a variant (ClinVar data)
- The user wants deleteriousness or pathogenicity scores (CADD scores)
- The user wants functional impact predictions (SIFT, PolyPhen2, MutationTaster, etc.)
- The user asks about cancer relevance of a variant (COSMIC, CIViC data)
- The user asks "is this variant pathogenic?" or "what is the clinical interpretation?"

Do NOT use this tool for:
- Population allele frequencies → use gnomAD MCP tools instead
- Gene constraint scores (pLI, LOEUF) → use gnomAD MCP get_gene instead
- FinnGen-specific annotations (AF, consequence, enrichment) → use get_variant_annotations instead

Returns: ClinVar clinical significance and conditions, CADD phred score, functional predictions (SIFT, PolyPhen2, MutationTaster, etc.), COSMIC cancer data, CIViC clinical evidence, and rsID.""",
        "parameters": {
            "variant": {
                "type": "string",
                "description": "Single variant in chr:pos:ref:alt format (e.g., '1:55051215:G:A'). Any separator (: - _ |) accepted.",
            },
            "variants": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of variant IDs for batch lookup (e.g., ['1:55051215:G:A', '7:117559590:ATCT:A']). Max 1000.",
            },
            "fields": {
                "type": "string",
                "description": "Comma-separated annotation sources to query (default: clinvar,cadd,dbnsfp,cosmic,civic,dbsnp). Do not include gnomad_genome or gnomad_exome.",
                "default": "clinvar,cadd,dbnsfp,cosmic,civic,dbsnp",
            },
        },
    },
    {
        "name": "get_gene_group_members",
        "category": "general",
        "sdk_replaceable": True,
        "description": (
            "Enumerate the member genes of an HGNC gene group / family (e.g. all GPCRs), "
            "returning gene symbols together with their genomic coordinates. "
            "Identify the group by exactly ONE of group_id (HGNC gene-group ID) or "
            "group_name (HGNC gene-group name); provide one, not both. "
            "By default olfactory receptors are EXCLUDED (exclude_olfactory=true): they are "
            "GPCRs that dominate large families like GPCRs by sheer count and are rarely the "
            "analysis target. Set exclude_olfactory=false to get the full membership. "
            "Results come from HGNC gene-group data served by the API. "
            "TIP: for database analyses joining a whole gene group (e.g. cis-pQTL "
            "colocalizations for all GPCRs), prefer filtering gene_annotations_v directly "
            "on gene_group_ids/gene_group_names rather than enumerating members here — see "
            "the get_database_schema example for gene_annotations_v."
        ),
        "parameters": {
            "group_id": {
                "type": "integer",
                "description": "HGNC gene-group ID. Provide exactly one of group_id or group_name.",
            },
            "group_name": {
                "type": "string",
                "description": "HGNC gene-group / family name (e.g. 'G protein-coupled receptors'). Provide exactly one of group_id or group_name.",
            },
            "exclude_olfactory": {
                "type": "boolean",
                "description": (
                    "Exclude olfactory receptors (default true). They are GPCRs that dominate "
                    "large families by count; set false to include them in the full membership."
                ),
                "default": True,
            },
        },
    },
    {
        "name": "normalize_gene_symbols",
        "category": "general",
        "sdk_replaceable": True,
        "description": (
            "Resolve input gene symbols / aliases / previous symbols to their current "
            "approved HGNC symbol (exact match, not fuzzy). Useful to clean up a gene "
            "list before querying. Returns mappings + any unresolved inputs. "
            "Served by the API."
        ),
        "parameters": {
            "symbols": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Gene symbols, aliases, or previous symbols to resolve to current approved HGNC symbols.",
                "required": True,
            },
        },
    },
]

# The tools code execution IS, rather than tools it can replace: the gateway to the
# sandbox, the SDK index a script is written against, and the reader for what a run
# left behind. They are a list of their own because `resolve_tools` includes them on
# exactly one surface — nothing here is a datum a script could fetch instead, so the
# sdk_replaceable question does not arise for them.
#
# subagent.py names all three in its `disabled` set except where a skill is granted
# run_analysis explicitly; run_analysis and read_artifact are additionally in
# mcp_server.py's _mcp_disabled, and run_analysis has no register_mcp_tools block at
# all, so no disabled_tools set can register it (see the comment there).
CODE_EXECUTION_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "list_capabilities",
        "category": "orchestration",
        "sdk_replaceable": False,
        "description": (
            "List the `genetics` SDK surface available to analysis scripts, one module at a "
            "time. Returns signatures with their docstrings, and the `usage` line saying "
            "exactly how to import it. Call this before writing a "
            "script instead of guessing function names. Modules: 'genetics' (the sync functions a "
            "script calls), 'client' (the awaitable GeneticsClient form), 'errors' (what a "
            "script catches). Omit `module` for a cheap index of module names and the "
            "functions each exports."
        ),
        "parameters": {
            "module": {
                "type": "string",
                "description": "SDK module to describe. Omit for the index.",
                "enum": ["genetics", "client", "errors", "plots"],
            },
        },
    },
    {
        "name": "run_analysis",
        "category": "orchestration",
        "sdk_replaceable": False,
        "description": (
            # the "use this INSTEAD OF chaining data-access tools" arbitration that used to
            # live here moved into the system prompt's "Choosing How to Get Data" section
            # (genetics-results-suite-4h6.69): a preference BETWEEN tools cannot be stated
            # inside one tool's description, where it is invisible to the prompt and
            # contradicted whatever the prompt said about the other path. Nothing is lost
            # for MCP clients, which never see this tool at all — run_analysis has no
            # register_mcp_tools block.
            "Run a Python script against the genetics data in a sandbox and get back what it "
            "printed. One script can query, join, filter and summarise in a single call.\n\n"
            "Keep a script to ONE chain of work. A run that overruns `timeout_s` returns "
            "nothing at all, so bundling independent analyses into one script risks losing "
            "every one of them to the slowest — split independent work across separate "
            "calls.\n\n"
            "Write the script against the `genetics` SDK — `import genetics` — and call "
            "list_capabilities first for the exact signatures rather than guessing. PRINT "
            "EVERYTHING YOU WANT TO SEE: only the "
            "script's output comes back (stdout and stderr interleaved, capped at 64 KiB with "
            "the middle elided). The value of the last expression is not returned.\n\n"
            "SAVE FILES INTO THE ARTIFACTS DIRECTORY, NOT THE WORKING DIRECTORY. The script's "
            "cwd is a scratch directory that is DISCARDED — a relative `savefig(\"x.png\")` or "
            "`write_csv(\"x.csv\")` is thrown away and reported as no artifact at all. Write to "
            "`os.path.join(os.environ[\"SANDBOX_ARTIFACTS_DIR\"], name)`, or use a "
            "`genetics.plots` helper, which resolves a relative path there for you.\n\n"
            "Files in the artifacts directory are reported as a manifest of names and sizes. An "
            "IMAGE artifact is fetched and shown to the user automatically — save a figure and "
            "it appears, so do not also render the plot as text or emit a "
            "markdown image placeholder for it. Non-image artifacts are offered to the user as "
            "DOWNLOAD LINKS automatically (the first few, smallest first): name them in your "
            "answer, but never paste their contents and never invent a URL. Any artifact can "
            "also be read back with "
            "read_artifact by name, for about 5 minutes after the run; printing what you need "
            "is still cheaper than reading a file back, so print anything small.\n\n"
            "CONCURRENCY: at most 4 data requests may be in flight at once from one script. "
            "Going over answers 429 and, under `asyncio.gather`, loses the results of the "
            "requests that did succeed — batch at 4 and pass `return_exceptions=True`.\n\n"
            "Standard figures are already written: `genetics.plots` has the conventional ones "
            "— a locuszoom and an upset among them — so a request for one is a call, not a "
            "plot to compose from scratch. list_capabilities(module=\"plots\") lists what is "
            "there. Every "
            "figure is styled by the sandbox itself; a script neither needs nor should add a "
            "style, and one that sets its own is overriding a deliberate default.\n\n"
            "Each run is independent: no variables, files or imports survive from one call to "
            "the next, so a follow-up script must redo the work it needs."
        ),
        "parameters": {
            "code": {
                "type": "string",
                "description": "Python source to run. Print the results you want to see.",
                "required": True,
            },
            "timeout_s": {
                "type": "integer",
                "description": (
                    "Wall-clock seconds allowed for the script, 1-120 (default 60). Raise it "
                    "only for a script you expect to be slow; a larger value does not make a "
                    "queued run start sooner."
                ),
                "default": 60,
                # sandbox_client._validate rejects outside 1..MAX_TIMEOUT_S
                "minimum": 1,
                "maximum": 120,
            },
        },
    },
    {
        "name": "read_artifact",
        "category": "orchestration",
        "sdk_replaceable": False,
        "description": (
            "Read a file that a run_analysis script in THIS conversation wrote to its "
            "artifacts directory. Takes the artifact NAME exactly as reported in that run's "
            "manifest — never a path and never an execution id. Returns text inline "
            "(truncated if very long) and binary content base64-encoded with its content "
            "type. Artifacts are readable for about 5 minutes after the run finishes and "
            "only from the conversation that produced them; anything else is 'not found'. "
            "Image artifacts are already shown to the user automatically, so read one only "
            "if you need its bytes. For a couple of numbers, printing them from the script "
            "is cheaper than reading the file back."
        ),
        "parameters": {
            "name": {
                "type": "string",
                "description": "Artifact file name from the run's manifest, e.g. 'manhattan.png'.",
                "required": True,
            },
        },
    },
]

# BigQuery tools for advanced queries
BIGQUERY_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "query_database",
        "category": "bigquery",
        "sdk_replaceable": True,
        "description": """Execute a SQL query against the genetics database.

For simple single-gene or single-variant lookups, prefer specialized tools (get_credible_sets_by_gene, get_credible_sets_by_variant, etc.).

**USE the database when the question involves:**
- Aggregations across many phenotypes, genes, or variants
- Complex filtering (e.g., "LoF variants with PIP > 0.05 AND MAF < 0.05 across all traits")
- Cross-referencing between data types (e.g., fine-mapping results vs. burden test results)
- Batch queries over many genes/variants that would require many individual API calls
- Custom statistical summaries or counts

**IMPORTANT: Always call get_database_schema FIRST** to discover all available tables and their columns. The database contains more tables than just credible sets — including exome/burden test results and other data types.

Refer to views by their bare name (e.g., `credible_sets_v`) — do NOT prefix them with a project or dataset.
Views include a `resource` column (finngen, ukbb, open_targets, etc.) for filtering by data source.
Always include a LIMIT clause in your SQL to control how many rows are shown to the user.
The download file automatically includes all matching rows (up to 100,000) regardless of the SQL LIMIT.
If the download hits the 100,000-row cap, tell the user to add filters to narrow the results.""",
        "parameters": {
            "sql": {
                "type": "string",
                "description": "SQL query to execute. Refer to views by their bare name (e.g., credible_sets_v) — do not prefix them with a project or dataset. Call get_database_schema first to discover available tables. Always include LIMIT clause.",
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
        "name": "get_database_schema",
        "category": "bigquery",
        "sdk_replaceable": True,
        "description": "Get schema for database tables. **Always call this before query_database** to discover available data. Returns resource descriptions with aliases, table/column metadata with allowed filter values, and example SQL queries. Optionally pass a table name to get schema for just that table.",
        "parameters": {
            "table": {
                "type": "string",
                "description": "Optional: return schema for just this table (e.g. 'gene_burden_results_v'). Omit for all tables. Available: credible_sets_v, colocalization_v, coloc_credsets_v, exome_variant_results_v, gene_burden_results_v",
            },
        },
    },
]

SUBAGENT_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "launch_subagents",
        "category": "orchestration",
        "sdk_replaceable": False,
        "description": """Launch one or more specialized subagents in parallel to handle complex queries.
Each subagent has its own skill (instructions + tools) and runs independently.
Use this when the question requires multiple independent data gathering or analysis tasks that can run simultaneously.

Available skills:
- **genetics_data_extraction**: Extract genetics data (GWAS, QTL, credible sets, gene expression, LD, etc.)
- **literature_review**: Search scientific literature and web for relevant publications
- **database_analysis**: Run complex SQL queries against the genetics database
- **data_analysis**: Write and run a Python script for statistical analysis or data processing — the subagent writes the script, runs it in the sandbox itself, iterates on failures, and reports the printed output. Figures it produces are NOT displayed to the user, so call `run_analysis` yourself when the answer is a plot
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
                            "description": "Skill name (genetics_data_extraction, literature_review, database_analysis, data_analysis, variant_list_analysis)",
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

# the values a client may still put on the wire in `tool_profile`. NOT a set of surfaces:
# `code_execution_requested` below maps every one of them onto the single boolean the
# surface is resolved from. It survives because two callers have to tell a value this
# server recognises from one it does not — the admin's DEFAULT_TOOL_PROFILE validation
# (routers/llm_config.py) and `known_profile` on /chat/v1/tools/resolved.
KNOWN_TOOL_PROFILES: frozenset[str] = frozenset(
    {"api", "bigquery", "rag", "nocode", "code"}
)

# unknown profile values already warned about. The coercion stays silent to the caller on
# purpose (see `code_execution_requested`), but an operator has to be able to see the
# drift, and the value arrives on EVERY turn of a session that stored it — a per-request
# warning would bury itself and stop being read. Bounded so a client that invents a new
# value per request floods neither the log nor this set.
_WARNED_UNKNOWN_PROFILES: set[str] = set()
_MAX_WARNED_UNKNOWN_PROFILES = 64


def _warn_unknown_profile(tool_profile: str) -> None:
    if tool_profile in _WARNED_UNKNOWN_PROFILES:
        return
    if len(_WARNED_UNKNOWN_PROFILES) >= _MAX_WARNED_UNKNOWN_PROFILES:
        return
    _WARNED_UNKNOWN_PROFILES.add(tool_profile)
    logger.warning(
        "Unrecognised tool_profile %r - resolved to the no-code surface, so this request "
        "gets the data tools and no code execution. Only %r selects code execution; %s are "
        "the values this server recognises. A client offering anything else has drifted "
        "from it; GET /chat/v1/tools/resolved?tool_profile=<value> reports the same thing "
        "per request.",
        tool_profile,
        "code",
        ", ".join(sorted(KNOWN_TOOL_PROFILES)),
    )


def code_execution_requested(tool_profile: str | None) -> bool:
    """Coerce the wire `tool_profile` to the boolean the surface is resolved from.

    THE EDGE, and the only place a profile name means anything to the surface: `"code"` is
    code execution and EVERYTHING else — `None`, the legacy `api`/`bigquery`/`rag`,
    `"nocode"`, and a value this server has never heard of — is the no-code surface.

    The fallback is universal on purpose. The value is read back from `chat_messages` rows
    and from `user_settings.chat_tool_profile` written by older clients, and no-code is the
    direction where a stale row loses code execution rather than acquiring it. Nothing
    rewrites the stored string: history and the `tool_profile IS NULL` analysis still read
    what the client sent, only its resolution is decided here. An unrecognised value logs a
    WARNING once per distinct value and is otherwise silent to the model and the caller.
    """
    if tool_profile is not None and tool_profile not in KNOWN_TOOL_PROFILES:
        _warn_unknown_profile(tool_profile)
    return tool_profile == "code"


def all_local_tool_definitions() -> list[dict[str, Any]]:
    """Every locally-defined tool, before any surface or disabled filter.

    The four lists are one surface everywhere they are used together; naming that here
    keeps `register_mcp_tools` and `tool_category` registering and labelling the same set.
    This is NOT what a chat request is handed — `resolve_tools` is — but it is where a
    caller that narrows by explicit tool name (subagent skills) starts.
    """
    return (
        list(TOOL_DEFINITIONS)
        + list(CODE_EXECUTION_TOOL_DEFINITIONS)
        + list(BIGQUERY_TOOL_DEFINITIONS)
        + list(SUBAGENT_TOOL_DEFINITIONS)
    )


def resolve_tools(
    code_execution: bool, disabled: set[str] | None = None
) -> list[dict[str, Any]]:
    """The local tool definitions one surface is handed.

    There are exactly two surfaces and one boolean chooses between them:

      code_execution=True  — the sandbox's own tools, plus every tool the SDK inside the
        sandbox cannot stand in for. That is not "everything minus run_analysis": the line
        is internal genetics data (a script fetches it through the SDK) against outside
        resources (the sandbox egress allow-list names db-api and results-api only, so no
        script reaches them), with the entity lookups kept because resolving a symbol or a
        phenotype name to an id is what the model does *before* it writes a script.
      code_execution=False — every data tool, and none of the sandbox's.

    Membership is `sdk_replaceable` on the definition itself, so adding a tool is one
    decision taken where the tool is defined rather than an edit to a table elsewhere.
    `launch_subagents` reaches neither surface: which one should carry it is an open
    question, and ENABLE_SUBAGENTS=false keeps it out of every deployment meanwhile.
    """
    data_tools = list(TOOL_DEFINITIONS) + list(BIGQUERY_TOOL_DEFINITIONS)
    if code_execution:
        tools = list(CODE_EXECUTION_TOOL_DEFINITIONS) + [
            t for t in data_tools if not t["sdk_replaceable"]
        ]
    else:
        tools = data_tools
    if disabled:
        tools = [t for t in tools if t["name"] not in disabled]
    return tools


_LOCAL_TOOL_CATEGORIES: dict[str, str] = {
    t["name"]: t["category"] for t in all_local_tool_definitions()
}


def tool_category(name: str) -> str | None:
    """A local tool's display label, or None for a name defined nowhere local.

    No surface decision reads `category` — `resolve_tools` goes by `sdk_replaceable`.
    `get_anthropic_tools` drops it on the way to Anthropic's format, which has no field for
    it; a caller that has to label a resolved tool (the tools panel groups by it) reads it
    back through here.
    """
    return _LOCAL_TOOL_CATEGORIES.get(name)


def get_anthropic_tools(
    custom_descriptions: dict[str, str] | None = None,
    # keyword-only: every profile string is truthy, so an old-style positional call
    # `get_anthropic_tools(None, "nocode")` would resolve to the CODE surface
    *,
    code_execution: bool = False,
    disabled_tools: set[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Return one surface's tool definitions in Anthropic's format.

    `resolve_tools` in Anthropic clothing, and nothing more: no profile name reaches here,
    because a request's name was coerced to this boolean by `code_execution_requested` at
    the edge it arrived on.

    Args:
        custom_descriptions: Optional dict mapping tool names to custom descriptions
        code_execution: Which of the two surfaces; see `resolve_tools`.
        disabled_tools: Optional set of tool names to exclude, applied after the surface.
    """
    return _to_anthropic_format(
        resolve_tools(code_execution, disabled_tools), custom_descriptions
    )


def all_anthropic_tools(
    custom_descriptions: dict[str, str] | None = None,
    disabled_tools: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Every local tool in Anthropic's format, narrowed only by `disabled_tools`.

    For a caller that then narrows by explicit tool name and so must not be handed a
    surface first: a subagent skill names both data tools and `run_analysis`, which no
    single surface carries.
    """
    definitions = all_local_tool_definitions()
    if disabled_tools:
        definitions = [t for t in definitions if t["name"] not in disabled_tools]
    return _to_anthropic_format(definitions, custom_descriptions)


def _to_anthropic_format(
    tool_definitions: list[dict[str, Any]],
    custom_descriptions: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    anthropic_tools = []

    for tool_def in tool_definitions:
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
            # bounds are emitted only when a parameter declares them, and a parameter
            # declares one only where the server already enforces it — see the block
            # comment above TOOL_DEFINITIONS. `0`/`0.0` are legitimate bounds, so these
            # test for presence rather than truthiness.
            for keyword in ("minimum", "maximum", "pattern"):
                if keyword in param_info:
                    prop[keyword] = param_info[keyword]
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


def _gate(
    mcp: "FastMCP",
    disabled: set[str],
    code_execution: bool | None,
) -> Callable[[], Callable[[Callable], Callable]]:
    """The one admission rule every `register_mcp_tools` handler goes through.

    Handlers used to divide into two classes — a few wrapped in `if "x" not in disabled:`
    and the rest registered whatever the caller asked for — so naming a tool in
    `disabled_tools` was an inert control over most of the surface, and no caller could
    subtract a data tool from /mcp at all. Routing every site through one decorator makes
    the withheld set the only thing a caller has to get right.

    The decision is taken on the handler's own `__name__`, which is the registered tool
    name, so a site cannot drift from the name it is gated under. A withheld handler is
    returned undecorated: the function exists in the enclosing scope, FastMCP never learns
    of it.

    `code_execution` is None where the caller has chosen no surface — nothing beyond
    `disabled` is withheld, which is what the deployed server does. Given a boolean it
    subtracts by the same rule `resolve_tools` applies to a chat request, so the two cannot
    disagree about what belongs to a surface.

    This can only ever SUBTRACT. A handler that does not exist is registered by no argument
    to this function (`run_analysis` has none, deliberately — see the comment at its place
    in the ordering below), and neither is a name `resolve_tools` does not return.
    """
    surface = (
        None
        if code_execution is None
        else {tool["name"] for tool in resolve_tools(code_execution)}
    )

    def tool() -> Callable[[Callable], Callable]:
        def register(handler: Callable) -> Callable:
            name = handler.__name__
            if name in disabled or (surface is not None and name not in surface):
                return handler
            return mcp.tool()(handler)

        return register

    return tool


def register_mcp_tools(
    mcp: "FastMCP",
    executor: "ServerToolExecutor",
    disabled_tools: set[str] | None = None,
    code_execution: bool | None = None,
) -> None:
    """
    Register all tools with a FastMCP server instance.

    Args:
        mcp: FastMCP server instance
        executor: ServerToolExecutor instance for making API calls
        disabled_tools: Optional set of tool names to skip registration.
        code_execution: Optional surface to register, by the same rule `resolve_tools`
            uses — True for the code surface, False for the no-code one. None registers
            everything `disabled_tools` leaves, which is what the deployed server does;
            the startup setting that would pass a boolean does not exist yet.

    EVERY handler below registers through `_gate`, never through `mcp.tool()` directly. A
    site that reaches for the raw decorator is unreachable by both filters and silently
    re-opens whatever the caller was trying to withhold.

    BOUNDS ON THIS SURFACE ARE NOT THE SAME DECISION AS ON THE ANTHROPIC ONE
    (genetics-results-suite-4h6.70). FastMCP derives each schema from the signature below,
    so an `Annotated[..., Field(ge=..., le=...)]` here does not merely advertise a bound —
    pydantic REJECTS an out-of-range value before the executor sees it. That is why only
    the parameters the executor ALREADY rejects carry one (the `sql_int`/`sql_float` sites:
    the four `window` arguments, `min_pip`, `get_hla_by_allele.max_rows`); there the
    declaration moves the identical rejection earlier, so it is not a breaking change
    (neither path ever returns success) — but it is not unobservable either: previously
    an out-of-range call got a normal tool result (`{"success": false, "error": "window
    must be <= 10000000, got ..."}`), and now pydantic rejects it before the executor
    runs, surfacing as FastMCP's own `ToolError`/`isError` response instead. A client
    branching on `result["success"]` sees a different envelope for the same rejection.
    The CLAMPED parameters — `web_search.max_results`, `search_mgi.max_results`,
    `search_cbioportal.max_results`, `search_uniprot.size`,
    `get_drug_targets_for_gene.min_phase` / `.max_results` and
    `get_target_bioactivity.pchembl_min` / `.max_results` — are deliberately left bare:
    the server accepts an over-large value today and silently returns the capped count, so
    declaring the cap here would turn a working MCP call into a validation error. Their
    bounds are declared only in `parameters`, where they steer the model without rejecting.
    """
    if code_execution is not None and not isinstance(code_execution, bool):
        # every non-empty env string is truthy, so an unconverted "false" would select the
        # code surface; refuse it here rather than register the wrong one
        raise TypeError(f"code_execution must be a bool or None, got {type(code_execution).__name__}")

    _tool = _gate(mcp, disabled_tools or set(), code_execution)

    @_tool()
    async def search_phenotypes(query: str, limit: int = 100) -> dict:
        """Look up phenotypes by disease/trait name. Supports comma-separated values for batch lookup."""
        return await executor.search_phenotypes(query, limit)

    @_tool()
    async def search_genes(query: str, limit: int = 10) -> dict:
        """Look up gene symbols and positions. Supports comma-separated values for batch lookup."""
        return await executor.search_genes(query, limit)

    @_tool()
    async def lookup_variants_by_rsid(rsids: str) -> dict:
        """Convert rsIDs to variant IDs (chr:pos:ref:alt format)."""
        return await executor.lookup_variants_by_rsid(rsids)

    @_tool()
    async def get_credible_sets_by_gene(
        gene: str,
        window: int = 500000,
        resource: str | None = None,
        data_types: str | None = None,
        summarize: bool = True,
    ) -> dict:
        """Get credible sets for variants near a gene."""
        return await executor.get_credible_sets_by_gene(
            gene, window, resource, data_types, summarize
        )

    @_tool()
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

    @_tool()
    async def get_credible_sets_by_region(
        region: str,
        resource: str | None = None,
        coding_only: bool = False,
        summarize: bool = True,
    ) -> dict:
        """Get credible sets overlapping a genomic region (chr:start-end)."""
        return await executor.get_credible_sets_by_region(
            region, resource, coding_only, summarize
        )

    @_tool()
    async def get_credible_sets_by_phenotype(
        phenotype: str,
        resource: str = "finngen",
        summarize: bool = True,
    ) -> dict:
        """Get all genes/variants associated with a phenotype from GWAS fine-mapping."""
        return await executor.get_credible_sets_by_phenotype(
            phenotype, resource, summarize
        )

    @_tool()
    async def get_credible_set_leads_by_phenotype(
        phenotype: str, resource: str = "finngen"
    ) -> dict:
        """Get one lead variant per credible set for a phenotype."""
        return await executor.get_credible_set_leads_by_phenotype(phenotype, resource)

    @_tool()
    async def get_credible_set_by_id(
        resource: str,
        phenotype: str,
        credible_set_id: str,
    ) -> dict:
        """Get all variants in a specific credible set."""
        return await executor.get_credible_set_by_id(resource, phenotype, credible_set_id)

    @_tool()
    async def get_credible_sets_by_qtl_gene(
        gene: str,
        data_types: str | None = None,
        resource: str | None = None,
        summarize: bool = True,
    ) -> dict:
        """Get QTL associations where a gene is the molecular trait.

        Also answers gene-based caQTL questions: a caQTL trait is a chromatin peak, and this
        resolves the Open4Gene peak-to-gene link (cell-type-matched), returning the linked gene
        symbol in `trait` and the peak id in `trait_original` / `cs_id`.
        """
        return await executor.get_credible_sets_by_qtl_gene(
            gene, data_types, resource, summarize
        )

    @_tool()
    async def get_gene_expression(gene: str) -> dict:
        """Get tissue-specific gene expression levels."""
        return await executor.get_gene_expression(gene)

    @_tool()
    async def get_asm_qtl_by_variant(
        variant: str,
        resources: str | None = None,
    ) -> dict:
        """Get ASM-QTL data for a variant."""
        return await executor.get_asm_qtl_by_variant(variant, resources)

    @_tool()
    async def get_asm_qtl_by_gene(
        gene: str,
        resources: str | None = None,
        window: Annotated[int, Field(ge=0, le=10_000_000)] = 500000,
    ) -> dict:
        """Get ASM-QTL data for variants near a gene."""
        return await executor.get_asm_qtl_by_gene(gene, resources, window)

    @_tool()
    async def get_open_chromatin_by_variant(
        variant: str,
        resources: str | None = None,
    ) -> dict:
        """Get open-chromatin atlas peaks overlapping a variant's position."""
        return await executor.get_open_chromatin_by_variant(variant, resources)

    @_tool()
    async def get_open_chromatin_by_region(
        chrom: str,
        start: int,
        end: int,
        resources: str | None = None,
    ) -> dict:
        """Get open-chromatin atlas peaks overlapping a genomic region."""
        return await executor.get_open_chromatin_by_region(chrom, start, end, resources)

    @_tool()
    async def get_open_chromatin_by_peak(
        peak_id: str,
        resources: str | None = None,
    ) -> dict:
        """Get one open-chromatin atlas peak by its peak id."""
        return await executor.get_open_chromatin_by_peak(peak_id, resources)

    @_tool()
    async def get_open_chromatin_by_gene(
        gene: str,
        resources: str | None = None,
        window: Annotated[int, Field(ge=0, le=10_000_000)] = 500000,
    ) -> dict:
        """Get open-chromatin atlas peaks near a gene."""
        return await executor.get_open_chromatin_by_gene(gene, resources, window)

    @_tool()
    async def get_peak_to_genes(
        peak_id: str,
        resources: str | None = None,
        gencode_version: str | None = None,
    ) -> dict:
        """Get the genes an Open4Gene chromatin peak is linked to, per cell type."""
        return await executor.get_peak_to_genes(peak_id, resources, gencode_version)

    @_tool()
    async def get_gene_to_peaks(
        gene: str,
        resources: str | None = None,
        gencode_version: str | None = None,
    ) -> dict:
        """Get the Open4Gene chromatin peaks linked to a gene, per cell type."""
        return await executor.get_gene_to_peaks(gene, resources, gencode_version)

    @_tool()
    async def get_variant_effect_by_variant(
        variant: str,
        resources: str | None = None,
    ) -> dict:
        """Get in-silico predicted variant effect on chromatin accessibility for a variant."""
        return await executor.get_variant_effect_by_variant(variant, resources)

    @_tool()
    async def get_variant_effect_by_gene(
        gene: str,
        resources: str | None = None,
        window: Annotated[int, Field(ge=0, le=10_000_000)] = 500000,
    ) -> dict:
        """Get in-silico predicted variant effects on chromatin accessibility near a gene."""
        return await executor.get_variant_effect_by_gene(gene, resources, window)

    @_tool()
    async def get_mpra_by_variant(
        variant: str,
        resources: str | None = None,
    ) -> dict:
        """Get measured MPRA cis-regulatory allelic activity (emVar/active/log2Skew) for a variant."""
        return await executor.get_mpra_by_variant(variant, resources)

    @_tool()
    async def get_mpra_by_region(
        chrom: str,
        start: int,
        end: int,
        resources: str | None = None,
    ) -> dict:
        """Get measured MPRA cis-regulatory allelic activity for variants overlapping a region."""
        return await executor.get_mpra_by_region(chrom, start, end, resources)

    @_tool()
    async def get_mpra_by_gene(
        gene: str,
        resources: str | None = None,
        window: Annotated[int, Field(ge=0, le=10_000_000)] = 500000,
    ) -> dict:
        """Get measured MPRA cis-regulatory allelic activity for variants near a gene."""
        return await executor.get_mpra_by_gene(gene, resources, window)

    @_tool()
    async def get_mpra_pip_concordance_by_gene(
        gene: str,
        window: Annotated[int, Field(ge=0, le=10_000_000)] = 500000,
        resource: str = "finngen",
        min_pip: Annotated[float, Field(ge=0.0, le=1.0)] = 0.1,
    ) -> dict:
        """Cross-reference FinnGen fine-mapped credible-set PIP against measured MPRA emVar calls near a gene."""
        return await executor.get_mpra_pip_concordance_by_gene(gene, window, resource, min_pip)

    @_tool()
    async def get_gene_disease_associations(gene: str) -> dict:
        """Get Mendelian/rare disease gene-disease relationships."""
        return await executor.get_gene_disease_associations(gene)

    @_tool()
    async def get_colocalization(variant: str) -> dict:
        """Get colocalization results for a variant."""
        return await executor.get_colocalization(variant)

    @_tool()
    async def get_colocalization_by_credible_set(
        resource: str,
        phenotype: str,
        credible_set_id: str,
        dual_format: bool = False,
    ) -> dict:
        """Get the credible sets that colocalize with one specific credible set."""
        return await executor.get_colocalization_by_credible_set(
            resource, phenotype, credible_set_id, dual_format
        )

    @_tool()
    async def get_exome_results_by_gene(gene: str) -> dict:
        """Get rare variant burden test results for a gene."""
        return await executor.get_exome_results_by_gene(gene)

    @_tool()
    async def get_exome_results_by_variant(
        variant: str, resources: str | None = None
    ) -> dict:
        """Get rare-variant exome association results for one variant."""
        return await executor.get_exome_results_by_variant(variant, resources)

    @_tool()
    async def get_exome_results_by_region(
        region: str, resources: str | None = None
    ) -> dict:
        """Get rare-variant exome association results overlapping a genomic region."""
        return await executor.get_exome_results_by_region(region, resources)

    @_tool()
    async def get_exome_results_by_phenotype(resource: str, phenotype: str) -> dict:
        """Get individual variant exome results for a specific phenotype within an exome dataset."""
        return await executor.get_exome_results_by_phenotype(resource, phenotype)

    @_tool()
    async def get_gene_based_results(gene: str) -> dict:
        """Get gene-level burden test results from genebass, IBD, BipEx2, and SCHEMA."""
        return await executor.get_gene_based_results(gene)

    @_tool()
    async def get_gene_based_results_by_phenotype(resource: str, phenotype: str) -> dict:
        """Get the complete unfiltered gene burden results for one phenotype."""
        return await executor.get_gene_based_results_by_phenotype(resource, phenotype)

    @_tool()
    async def get_phenotype_report(resource: str, phenotype_code: str) -> dict:
        """Get a detailed markdown report for a phenotype."""
        return await executor.get_phenotype_report(resource, phenotype_code)

    @_tool()
    async def lookup_phenotype_names(codes: list[str]) -> dict:
        """Translate phenotype codes to human-readable names."""
        return await executor.lookup_phenotype_names(codes)

    @_tool()
    async def list_datasets(
        resource: str | None = None, include_stats: bool = True
    ) -> dict:
        """List all datasets with descriptions, products, and sample sizes."""
        return await executor.list_datasets(resource, include_stats)

    @_tool()
    async def get_resource_metadata(resource: str) -> dict:
        """Get the harmonized per-trait metadata of one resource."""
        return await executor.get_resource_metadata(resource)

    @_tool()
    async def get_dataset_display_names() -> dict:
        """Get display-name overrides keyed by the raw dataset column value."""
        return await executor.get_dataset_display_names()

    @_tool()
    async def get_credible_sets_stats(
        resource_or_dataset: str,
        trait: str | None = None,
    ) -> dict:
        """Get credible sets stats. CRITICAL: Include the INCLUDE_IN_RESPONSE field value verbatim in your response."""
        return await executor.get_credible_sets_stats(resource_or_dataset, trait)

    @_tool()
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

    @_tool()
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

    @_tool()
    async def get_gene_group_members(
        group_id: int | None = None,
        group_name: str | None = None,
        exclude_olfactory: bool = True,
    ) -> dict:
        """Enumerate member genes of an HGNC gene group/family (e.g. all GPCRs) with their coordinates. Provide exactly one of group_id or group_name. Olfactory receptors are excluded by default (exclude_olfactory=true)."""
        return await executor.get_gene_group_members(
            group_id, group_name, exclude_olfactory
        )

    @_tool()
    async def normalize_gene_symbols(symbols: list[str]) -> dict:
        """Resolve gene symbols/aliases/previous symbols to current approved HGNC symbols (exact match). Returns mappings plus any unresolved inputs."""
        return await executor.normalize_gene_symbols(symbols)

    @_tool()
    async def search_scientific_literature(
        query: str,
        max_results: int = 10,
        include_preprints: bool = True,
        date_range: str | None = None,
    ) -> dict:
        """Search scientific literature via Europe PMC or Perplexity. The backend is set by configuration, not by the caller."""
        return await executor.search_scientific_literature(
            query, max_results, include_preprints, date_range
        )

    @_tool()
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

    @_tool()
    async def search_mgi(
        query: str,
        query_type: str = "gene_phenotypes",
        species: str = "mouse",
        max_results: int = 25,
    ) -> dict:
        """Search Jackson Lab MGI for curated mouse phenotypes, alleles, and orthologs."""
        return await executor.search_mgi(
            query, query_type, species, max_results
        )

    @_tool()
    async def search_cbioportal(
        query: str,
        query_type: str = "gene_summary",
        cancer_types: list[str] | None = None,
        max_results: int = 25,
    ) -> dict:
        """Search cBioPortal for somatic alteration frequency in cancer cohorts. Coordinates are mostly GRCh37 — match on gene symbol and protein change, not position."""
        return await executor.search_cbioportal(
            query, query_type, cancer_types, max_results
        )

    @_tool()
    async def get_protein_annotations(
        query: str,
        organism_id: int | None = 9606,
        include: list[str] | None = None,
        feature_types: list[str] | None = None,
        residue_range: str | None = None,
    ) -> dict:
        """Get UniProt protein annotations (residue features, function, sequence). Pass a gene symbol, not a remembered accession."""
        return await executor.get_protein_annotations(
            query, organism_id, include, feature_types, residue_range
        )

    @_tool()
    async def map_protein_variants(
        variants: list[str],
        query: str,
        organism_id: int | None = 9606,
    ) -> dict:
        """Map amino-acid substitutions (e.g. ['P70A','R438H'] in TPO) to genomic coordinates and rsIDs via UniProt."""
        return await executor.map_protein_variants(variants, query, organism_id)

    @_tool()
    async def get_variant_protein_effect(variants: list[str]) -> dict:
        """Map genomic coding SNVs (e.g. ['12:40340400:G:A'], GRCh38) to the amino-acid change and curated UniProt/ClinVar annotation."""
        return await executor.get_variant_protein_effect(variants)

    @_tool()
    async def search_uniprot(
        query: str | None = None,
        keyword: str | None = None,
        organism_id: int | None = 9606,
        reviewed_only: bool = True,
        fields: str = "accession,id,protein_name,gene_names,organism_name",
        size: int = 25,
        count_only: bool = False,
    ) -> dict:
        """Search UniProtKB for the set of proteins matching a keyword, family, location or free-text query."""
        return await executor.search_uniprot(
            query, keyword, organism_id, reviewed_only, fields, size, count_only
        )

    @_tool()
    async def get_alphagenome_variant_predictions(
        variants: list[str],
        cell_type: str | None = None,
        modalities: list[str] | None = None,
    ) -> dict:
        """AlphaGenome's PREDICTED regulatory effect of one or more variants — a model's output, never a measurement."""
        return await executor.get_alphagenome_variant_predictions(variants, cell_type, modalities)

    @_tool()
    async def compare_alphagenome_with_measured(
        variants: list[str],
        cell_type: str | None = None,
        modalities: list[str] | None = None,
    ) -> dict:
        """This suite's MEASURED effect sizes beside AlphaGenome's PREDICTION for the same variant."""
        return await executor.compare_alphagenome_with_measured(variants, cell_type, modalities)

    @_tool()
    async def get_drug_targets_for_gene(
        query: str,
        min_phase: float = 0,
        include_indications: bool = False,
        max_results: int = 25,
    ) -> dict:
        """List the drugs and clinical candidates ChEMBL records against a gene's target, with mechanism, action type and highest clinical phase."""
        return await executor.get_drug_targets_for_gene(
            query, min_phase, include_indications, max_results
        )

    @_tool()
    async def get_drug_profile(query: str) -> dict:
        """Get ChEMBL's profile for one drug: highest clinical phase, approval and withdrawal, ATC class, targets and indications."""
        return await executor.get_drug_profile(query)

    @_tool()
    async def get_target_bioactivity(
        query: str,
        pchembl_min: float = 6.0,
        max_results: int = 25,
    ) -> dict:
        """Summarise ChEMBL potency measurements against a gene's target: activity counts, assay-type breakdown and the most potent compounds."""
        return await executor.get_target_bioactivity(query, pchembl_min, max_results)

    @_tool()
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

    @_tool()
    async def get_variants_in_ld(
        variant: str,
        window: int = 1500000,
        r2_threshold: float = 0.6,
        panel: str = "sisu42",
    ) -> dict:
        """Get all variants in LD with a given variant from FinnGen reference panel."""
        return await executor.get_variants_in_ld(variant, window, r2_threshold, panel)

    @_tool()
    async def analyze_variant_list(
        variants: str,
        resource: str | None = None,
    ) -> dict:
        """Analyze a list of variants for phenotype, QTL, and tissue patterns."""
        return await executor.analyze_variant_list(variants, resource)

    @_tool()
    async def get_summary_stats(
        variants: list[str],
        phenotypes: list[str],
        resource: str = "finngen",
        data_type: str = "gwas",
    ) -> dict:
        """Get summary statistics for specific variant-phenotype pairs."""
        return await executor.get_summary_stats(variants, phenotypes, resource, data_type)

    @_tool()
    async def get_summary_stats_by_region(
        region: str,
        phenotypes: list[str],
        resource: str = "finngen",
        data_type: str = "gwas",
    ) -> dict:
        """Get summary statistics for every variant in a region for one or more phenotypes."""
        return await executor.get_summary_stats_by_region(
            region, phenotypes, resource, data_type
        )

    @_tool()
    async def get_hla_by_phenotype(
        phenotypes: list[str],
        genes: str | None = None,
        resource: str = "finngen",
    ) -> dict:
        """Get classical HLA allele associations for one or more phenotypes."""
        return await executor.get_hla_by_phenotype(phenotypes, genes, resource)

    @_tool()
    async def get_hla_by_allele(
        allele: str,
        min_mlogp: float = 7.3,
        min_info: float = 0.5,
        resource: str = "finngen",
        max_rows: Annotated[int, Field(ge=1, le=100_000)] = 200,
    ) -> dict:
        """Get every phenotype a classical HLA allele is associated with."""
        return await executor.get_hla_by_allele(
            allele, min_mlogp, min_info, resource, max_rows
        )

    @_tool()
    async def get_dosage_sensitivity(genes: list[str]) -> dict:
        """Get rare-CNV dosage sensitivity scores (pHaplo, pTriplo) for genes."""
        return await executor.get_dosage_sensitivity(genes)

    @_tool()
    async def get_rcnv_associations(
        gene: str | None = None,
        phenotype: str | None = None,
        cnv_type: str | None = None,
        min_mlog10p: float | None = None,
        max_fdr_q: float | None = None,
        significant_only: bool = False,
        include_no_estimate: bool = False,
        limit: int = 200,
    ) -> dict:
        """Get rare-CNV gene associations (DEL/DUP) for a gene or an HPO phenotype group."""
        return await executor.get_rcnv_associations(
            gene=gene,
            phenotype=phenotype,
            cnv_type=cnv_type,
            min_mlog10p=min_mlog10p,
            max_fdr_q=max_fdr_q,
            significant_only=significant_only,
            include_no_estimate=include_no_estimate,
            limit=limit,
        )

    @_tool()
    async def get_variant_annotations(
        variant: str | None = None,
        region: str | None = None,
        gene: str | None = None,
        variants: list[str] | None = None,
        source: str = "finngen",
    ) -> dict:
        """Get variant annotations (consequence, allele frequency, rsID, enrichment)."""
        return await executor.get_variant_annotations(
            variant=variant, region=region, gene=gene, variants=variants, source=source
        )

    @_tool()
    async def get_myvariant_annotations(
        variant: str | None = None,
        variants: list[str] | None = None,
        fields: str = "clinvar,cadd,dbnsfp,cosmic,civic,dbsnp",
    ) -> dict:
        """Get clinical/functional variant annotations from myvariant.info (ClinVar, CADD, functional predictions, cancer data)."""
        return await executor.get_myvariant_annotations(
            variant=variant, variants=variants, fields=fields
        )

    @_tool()
    async def list_capabilities(module: str | None = None) -> dict:
        """List the `genetics` SDK surface for one module ('genetics', 'client', 'errors', 'plots') as signatures with docstrings. Omit module for the index."""
        return await executor.list_capabilities(module=module)

    # run_analysis has NO block here, deliberately, and the omission is the point.
    # docs/code-execution-security.md §5 layer 1 names two registration-layer controls for
    # run_analysis — membership of mcp_server.py's _mcp_disabled, and the absent handler
    # here — and then says the layer as a whole is assumed defeatable. The missing block is
    # the half no set passed to this function can undo: `disabled_tools` can only subtract. It also matches what
    # the tool needs — the handler is given the authenticated user and the chat session id
    # by the caller, and an MCP session has neither, so a registered wrapper could only
    # ever pass identity it does not have. Keep _mcp_disabled's entry as well: it is the
    # named control the security doc and the tests reason about, and it is what catches a
    # future block added here without this comment being read.

    @_tool()
    async def read_artifact(name: str) -> dict:
        """Read a named file an analysis script wrote to its artifacts directory."""
        return await executor.read_artifact(name=name)

    # BigQuery tools - available via MCP server for direct SQL queries
    @_tool()
    async def query_database(
        sql: str,
        max_rows: int = 1000,
        dry_run: bool = False,
    ) -> dict:
        """Execute SQL against the genetics database. Call get_database_schema first to discover available tables."""
        return await executor.query_database(sql, max_rows, dry_run)

    @_tool()
    async def get_database_schema(table: str | None = None) -> dict:
        """Get schema for database tables. Always call this before writing queries. Pass a table name to get just that table's schema."""
        return await executor.get_database_schema(table)
