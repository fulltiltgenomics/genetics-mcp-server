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
        "name": "get_asm_qtl_by_variant",
        "category": "api",
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
            },
        },
    },
    {
        "name": "get_open_chromatin_by_variant",
        "category": "api",
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
        "name": "get_open_chromatin_by_gene",
        "category": "api",
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
            },
        },
    },
    {
        "name": "get_variant_effect_by_variant",
        "category": "api",
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
            },
        },
    },
    {
        "name": "get_mpra_by_variant",
        "category": "api",
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
            },
        },
    },
    {
        "name": "get_mpra_pip_concordance_by_gene",
        "category": "api",
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
            },
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
        "description": "Get rare variant burden test results for a gene. Returns individual variant-level association statistics from exome sequencing across available resources (genebass/UKBB filtered to p<1e-4, IBD exome containing only exome-wide significant variants). Use this for single-gene queries. For batch queries across many genes, use the database instead (call get_database_schema to find the exome results table). For full individual-trait results, use get_exome_results_by_phenotype.",
        "parameters": {
            "gene": {"type": "string", "description": "Gene symbol or comma-separated list of gene symbols", "required": True},
        },
    },
    {
        "name": "get_exome_results_by_phenotype",
        "category": "api",
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
        "description": "Get gene-level burden test results from genebass, IBD, BipEx2, and SCHEMA datasets. Returns gene-based association statistics aggregated at the gene level. Different from get_exome_results_by_gene which returns individual variant-level exome results.",
        "parameters": {
            "gene": {
                "type": "string",
                "description": "Gene symbol or comma-separated list of gene symbols (e.g., 'APOE', 'BRCA1,TP53')",
                "required": True,
            },
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
        "name": "list_datasets",
        "category": "general",
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
        "name": "get_credible_sets_stats",
        "category": "api",
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
        "description": (
            "Search scientific literature for research papers about genes, variants, diseases, or biological mechanisms. "
            "Each call queries exactly ONE backend API (the 'backend' parameter): either 'europepmc' OR 'perplexity' — never both. "
            "These two backends are distinct APIs, not interchangeable labels for the same source:\n"
            "- 'europepmc' backend: queries the Europe PMC API, which indexes PubMed, Europe PMC, bioRxiv, and medRxiv. Returns structured paper records.\n"
            "- 'perplexity' backend: queries the Perplexity AI API, which searches a broader configured set of scientific web domains and returns an AI-generated summary with citations.\n"
            "When reporting results to the user, name the backend that was actually queried (the 'source' field in the response: 'europepmc' or 'perplexity'). "
            "Do NOT invent hybrid labels like 'PubMed/Europe PMC' or 'Perplexity/PubMed' — PubMed etc. are content indexed by the europepmc backend, not separate backends."
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
            "backend": {
                "type": "string",
                "description": (
                    "Which API to query for this call (NOT the underlying content source). "
                    "'europepmc' = call the Europe PMC API (which indexes PubMed/Europe PMC/bioRxiv/medRxiv); returns structured paper records. "
                    "'perplexity' = call the Perplexity AI API (broader scientific web); returns AI-generated summary with citations. "
                    "Exactly one backend is queried per call. Defaults to server configuration."
                ),
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
        "name": "search_mgi",
        "category": "general",
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
            },
        },
    },
    {
        "name": "get_protein_annotations",
        "category": "general",
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
                "type": "string",
                "description": "Gene symbol (strongly preferred, e.g. 'TPO', 'PRSS55'), UniProt entry name, or accession. Never supply an accession recalled from memory when a gene symbol is available.",
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
            },
            "count_only": {
                "type": "boolean",
                "description": "Return only the total number of matching entries, no rows. Cheap way to size a query before enumerating it.",
                "default": False,
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
    {
        "name": "get_variant_annotations",
        "category": "api",
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

# BigQuery tools for advanced queries
BIGQUERY_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "query_database",
        "category": "bigquery",
        "description": """Execute a SQL query against the genetics database.

For simple single-gene or single-variant lookups, prefer specialized tools (get_credible_sets_by_gene, get_credible_sets_by_variant, etc.).

**USE the database when the question involves:**
- Aggregations across many phenotypes, genes, or variants
- Complex filtering (e.g., "LoF variants with PIP > 0.05 AND MAF < 0.05 across all traits")
- Cross-referencing between data types (e.g., fine-mapping results vs. burden test results)
- Batch queries over many genes/variants that would require many individual API calls
- Custom statistical summaries or counts

**IMPORTANT: Always call get_database_schema FIRST** to discover all available tables and their columns. The database contains more tables than just credible sets — including exome/burden test results and other data types.

Use fully qualified view names (e.g., `genetics_results.credible_sets_v`).
Views include a `resource` column (finngen, ukbb, open_targets, etc.) for filtering by data source.
Always include a LIMIT clause in your SQL to control how many rows are shown to the user.
The download file automatically includes all matching rows (up to 100,000) regardless of the SQL LIMIT.
If the download hits the 100,000-row cap, tell the user to add filters to narrow the results.""",
        "parameters": {
            "sql": {
                "type": "string",
                "description": "SQL query to execute. Use fully qualified view names (e.g., genetics_results.credible_sets_v). Call get_database_schema first to discover available tables. Always include LIMIT clause.",
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
        "description": """Launch one or more specialized subagents in parallel to handle complex queries.
Each subagent has its own skill (instructions + tools) and runs independently.
Use this when the question requires multiple independent data gathering or analysis tasks that can run simultaneously.

Available skills:
- **genetics_data_extraction**: Extract genetics data (GWAS, QTL, credible sets, gene expression, LD, etc.)
- **literature_review**: Search scientific literature and web for relevant publications
- **database_analysis**: Run complex SQL queries against the genetics database
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

# valid tool profiles and which categories each profile includes
TOOL_PROFILES: dict[str, set[str]] = {
    "api": {"general", "api", "orchestration"},
    "bigquery": {"general", "bigquery", "orchestration"},
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
        window: int = 500000,
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
    async def get_asm_qtl_by_variant(
        variant: str,
        resources: str | None = None,
    ) -> dict:
        """Get ASM-QTL data for a variant."""
        return await executor.get_asm_qtl_by_variant(variant, resources)

    @mcp.tool()
    async def get_asm_qtl_by_gene(
        gene: str,
        resources: str | None = None,
        window: int = 500000,
    ) -> dict:
        """Get ASM-QTL data for variants near a gene."""
        return await executor.get_asm_qtl_by_gene(gene, resources, window)

    @mcp.tool()
    async def get_open_chromatin_by_variant(
        variant: str,
        resources: str | None = None,
    ) -> dict:
        """Get open-chromatin atlas peaks overlapping a variant's position."""
        return await executor.get_open_chromatin_by_variant(variant, resources)

    @mcp.tool()
    async def get_open_chromatin_by_region(
        chrom: str,
        start: int,
        end: int,
        resources: str | None = None,
    ) -> dict:
        """Get open-chromatin atlas peaks overlapping a genomic region."""
        return await executor.get_open_chromatin_by_region(chrom, start, end, resources)

    @mcp.tool()
    async def get_open_chromatin_by_gene(
        gene: str,
        resources: str | None = None,
        window: int = 500000,
    ) -> dict:
        """Get open-chromatin atlas peaks near a gene."""
        return await executor.get_open_chromatin_by_gene(gene, resources, window)

    @mcp.tool()
    async def get_variant_effect_by_variant(
        variant: str,
        resources: str | None = None,
    ) -> dict:
        """Get in-silico predicted variant effect on chromatin accessibility for a variant."""
        return await executor.get_variant_effect_by_variant(variant, resources)

    @mcp.tool()
    async def get_variant_effect_by_gene(
        gene: str,
        resources: str | None = None,
        window: int = 500000,
    ) -> dict:
        """Get in-silico predicted variant effects on chromatin accessibility near a gene."""
        return await executor.get_variant_effect_by_gene(gene, resources, window)

    @mcp.tool()
    async def get_mpra_by_variant(
        variant: str,
        resources: str | None = None,
    ) -> dict:
        """Get measured MPRA cis-regulatory allelic activity (emVar/active/log2Skew) for a variant."""
        return await executor.get_mpra_by_variant(variant, resources)

    @mcp.tool()
    async def get_mpra_by_region(
        chrom: str,
        start: int,
        end: int,
        resources: str | None = None,
    ) -> dict:
        """Get measured MPRA cis-regulatory allelic activity for variants overlapping a region."""
        return await executor.get_mpra_by_region(chrom, start, end, resources)

    @mcp.tool()
    async def get_mpra_by_gene(
        gene: str,
        resources: str | None = None,
        window: int = 500000,
    ) -> dict:
        """Get measured MPRA cis-regulatory allelic activity for variants near a gene."""
        return await executor.get_mpra_by_gene(gene, resources, window)

    @mcp.tool()
    async def get_mpra_pip_concordance_by_gene(
        gene: str,
        window: int = 500000,
        resource: str = "finngen",
        min_pip: float = 0.1,
    ) -> dict:
        """Cross-reference FinnGen fine-mapped credible-set PIP against measured MPRA emVar calls near a gene."""
        return await executor.get_mpra_pip_concordance_by_gene(gene, window, resource, min_pip)

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

    @mcp.tool()
    async def get_exome_results_by_phenotype(resource: str, phenotype: str) -> dict:
        """Get individual variant exome results for a specific phenotype within an exome dataset."""
        return await executor.get_exome_results_by_phenotype(resource, phenotype)

    @mcp.tool()
    async def get_gene_based_results(gene: str) -> dict:
        """Get gene-level burden test results from genebass, IBD, BipEx2, and SCHEMA."""
        return await executor.get_gene_based_results(gene)

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
    async def list_datasets(
        resource: str | None = None, include_stats: bool = True
    ) -> dict:
        """List all datasets with descriptions, products, and sample sizes."""
        return await executor.list_datasets(resource, include_stats)

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
    async def get_gene_group_members(
        group_id: int | None = None,
        group_name: str | None = None,
        exclude_olfactory: bool = True,
    ) -> dict:
        """Enumerate member genes of an HGNC gene group/family (e.g. all GPCRs) with their coordinates. Provide exactly one of group_id or group_name. Olfactory receptors are excluded by default (exclude_olfactory=true)."""
        return await executor.get_gene_group_members(
            group_id, group_name, exclude_olfactory
        )

    @mcp.tool()
    async def normalize_gene_symbols(symbols: list[str]) -> dict:
        """Resolve gene symbols/aliases/previous symbols to current approved HGNC symbols (exact match). Returns mappings plus any unresolved inputs."""
        return await executor.normalize_gene_symbols(symbols)

    if "search_scientific_literature" not in _disabled:

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

    if "web_search" not in _disabled:

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

    if "search_mgi" not in _disabled:

        @mcp.tool()
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

    if "get_protein_annotations" not in _disabled:

        @mcp.tool()
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

    if "map_protein_variants" not in _disabled:

        @mcp.tool()
        async def map_protein_variants(
            variants: list[str],
            query: str,
            organism_id: int | None = 9606,
        ) -> dict:
            """Map amino-acid substitutions (e.g. ['P70A','R438H'] in TPO) to genomic coordinates and rsIDs via UniProt."""
            return await executor.map_protein_variants(variants, query, organism_id)

    if "get_variant_protein_effect" not in _disabled:

        @mcp.tool()
        async def get_variant_protein_effect(variants: list[str]) -> dict:
            """Map genomic coding SNVs (e.g. ['12:40340400:G:A'], GRCh38) to the amino-acid change and curated UniProt/ClinVar annotation."""
            return await executor.get_variant_protein_effect(variants)

    if "search_uniprot" not in _disabled:

        @mcp.tool()
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

    @mcp.tool()
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

    if "get_myvariant_annotations" not in _disabled:

        @mcp.tool()
        async def get_myvariant_annotations(
            variant: str | None = None,
            variants: list[str] | None = None,
            fields: str = "clinvar,cadd,dbnsfp,cosmic,civic,dbsnp",
        ) -> dict:
            """Get clinical/functional variant annotations from myvariant.info (ClinVar, CADD, functional predictions, cancer data)."""
            return await executor.get_myvariant_annotations(
                variant=variant, variants=variants, fields=fields
            )

    # BigQuery tools - available via MCP server for direct SQL queries
    @mcp.tool()
    async def query_database(
        sql: str,
        max_rows: int = 1000,
        dry_run: bool = False,
    ) -> dict:
        """Execute SQL against the genetics database. Call get_database_schema first to discover available tables."""
        return await executor.query_database(sql, max_rows, dry_run)

    @mcp.tool()
    async def get_database_schema(table: str | None = None) -> dict:
        """Get schema for database tables. Always call this before writing queries. Pass a table name to get just that table's schema."""
        return await executor.get_database_schema(table)
