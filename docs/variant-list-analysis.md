# Variant List Analysis

Analyzes a list of variants (e.g., lead variants from a GWAS) for shared phenotype associations, QTL patterns, tissue enrichment, and nearest genes.

## Input format

One variant per line in `chr:pos:ref:alt` format. Optionally include tab or comma-separated columns for beta, se, and pvalue. A header row is auto-detected.

### Variants only

```
1:154453788:C:T
19:44908684:T:C
2:21263900:A:G
```

### With effect sizes

```
variant	beta	se	pvalue
1:154453788:C:T	0.15	0.02	1e-12
19:44908684:T:C	-0.08	0.01	5e-8
2:21263900:A:G	0.12	0.03	2e-6
```

Any CPRA separator is accepted (`:` `-` `_` `|` `/` `\`, so `1-154453788-C-T` works), as is a `chr` prefix (`chr1:154453788:C:T`). Chromosome `23` is treated as `X`. A single line of space-separated variants is split into one variant per line.

## What it does

For each input variant, the tool:

1. **Fetches all credible sets** (GWAS, eQTL, pQTL, caQTL) from the Genetics API
2. **Fetches the nearest gene** from the Genetics API
3. **Aggregates** the results:

| Aggregation | Description |
|---|---|
| **GWAS phenotypes** | For each phenotype, how many input variants associate with it. If betas provided, splits by direction consistency. |
| **pQTL genes** | For each gene with pQTL credible sets, how many input variants. With direction consistency. |
| **eQTL genes** | For each (gene, tissue) pair from eQTL credible sets, how many input variants. With direction consistency. |
| **caQTL tissues** | For each tissue from caQTL credible sets, how many input variants. |
| **Tissue enrichment** | For each tissue, how many input variants have any eQTL in that tissue. |
| **pQTL summary** | Total number of input variants with any pQTL. |
| **Variant-gene mapping** | Nearest protein-coding gene for each input variant. |

The GWAS, pQTL, eQTL and caQTL rows also carry the `resource` and `dataset` values they were seen in (comma-joined when more than one). Those four are the only data types aggregated: an sQTL credible set still counts its variant in `n_variants_with_cs`, but produces no aggregation row of its own.

## Usage

### Chat interface

Say: **"analyze this list of variants"** followed by the variant list (pasted or attached).

Optional context: mention the GWAS or locus the variants come from, and the resource to filter to (FinnGen, UK Biobank, etc.).

### Standalone CLI

```bash
# from stdin
echo "1:154453788:C:T
19:44908684:T:C" | python -m genetics_mcp_server.scripts.analyze_variants

# from file
python -m genetics_mcp_server.scripts.analyze_variants variants.txt --pretty

# filter to FinnGen
python -m genetics_mcp_server.scripts.analyze_variants variants.txt --resource finngen
```

Requires `GENETICS_API_URL` environment variable (defaults to `http://0.0.0.0:2000/api`), and `INTERNAL_API_SECRET` if the API requires it.

### MCP tool

The tool `analyze_variant_list` is available via the MCP server and the chat API.

## Output

JSON with the following structure:

```json
{
  "success": true,
  "n_variants": 10,
  "n_variants_with_cs": 8,
  "input_has_betas": true,
  "gwas_phenotypes": [
    {"trait": "T2D", "name": "Type 2 diabetes", "resource": "finngen", "dataset": "FinnGen_R13", "n_variants": 5, "n_consistent": 4, "n_inconsistent": 1}
  ],
  "pqtl_genes": [
    {"gene": "PCSK9", "resource": "ukbb", "dataset": "UKB_PPP", "n_variants": 3, "n_consistent": 2, "n_inconsistent": 1}
  ],
  "eqtl_genes": [
    {"gene": "SORT1", "tissue": "liver", "resource": "...", "dataset": "...", "n_variants": 4, "n_consistent": 3, "n_inconsistent": 1}
  ],
  "caqtl_tissues": [
    {"tissue": "liver", "resource": "...", "dataset": "...", "n_variants": 2}
  ],
  "tissue_enrichment": [
    {"tissue": "liver", "n_eqtl_variants": 6}
  ],
  "pqtl_summary": {"n_variants_with_pqtl": 4},
  "variant_genes": [
    {"variant": "1:154453788:C:T", "nearest_gene": "IL6R", "distance": 0}
  ]
}
```

Direction consistency fields (`n_consistent`, `n_inconsistent`) are only present when input betas are provided. `resource` and `dataset` are comma-joined lists of every value seen for that row, so they are empty strings when the upstream rows carry neither. `distance` is `-1` when no nearest gene was returned for a variant.
