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
- **Per-message size limits**: `_validate_latest_message` (in `chat_api.py`) caps the newest user message's typed-text length (`MAX_MESSAGE_CHARS`, default 50K) and attachment count (`MAX_ATTACHMENTS_PER_MESSAGE`, default 10), rejecting with HTTP 413 before any model call. Attachments are excluded from the text cap: images arrive as `image` blocks and data files (TSV/CSV/Excel) are inlined by the frontend as text blocks prefixed `[File: <name>]` — both are counted toward the attachment limit, not the character limit. The frontend (`LLMChat.tsx`) mirrors these limits for immediate feedback. Bulk data should be attached as a file rather than pasted
- **File attachments**: Upload/download/delete endpoints in `routers/chat_history.py` store files on disk (`ATTACHMENT_STORAGE_PATH`) with metadata in the `chat_attachments` table. Files are classified as `image`, `tsv`, or `excel`. Excel is a binary format, so `.xlsx`/`.xls` uploads are parsed to TSV at upload time via `excel_to_tsv()` (polars `read_excel`, calamine/`fastexcel` engine; all sheets, each prefixed `# Sheet: <name>` when multiple) and the parsed text is stored as a `.tsv` sidecar (`text_path` column); a file that fails to parse is rejected with HTTP 400 and nothing is written. The download endpoint serves the original bytes by default, or the model-ready text via `?as=text` (parsed TSV for excel, original for tsv/csv). The live frontend send path does not round-trip through these endpoints — it parses Excel→TSV client-side with SheetJS (`excelToTsv.ts`) before inlining, since sessions are created lazily after the first exchange and no `session_id` exists at first send. The server-side parse is therefore defense-in-depth: it covers direct API consumers and guarantees stored bytes are never surfaced as binary; `?as=text` is available for any client that prefers a backend round-trip
- **Cost logging**: Estimated USD cost logged for every Anthropic API call based on token usage and model pricing
- **Context usage tracking**: `get_context_window()` in `cost.py` maps model name prefixes to context window sizes (tokens). During streaming, `usage` SSE events are emitted after each agentic loop iteration, enabling the frontend to display a live context usage progress bar
- **Chat history persistence**: SQLite-based storage of conversation threads. Assistant turns persist both their content blocks (`content_json`: text + `tool_use`) and the tool outputs (`tool_results_json`: the `tool_result` blocks). Persisting tool results means a **resumed** conversation replays the actual data the model saw, not just its prose summary — preventing factual drift across turns/sessions (see "Tool result persistence" under Architecture decisions)
- **Configurable prompts**: Per-user LLM configuration stored in database

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
| `get_credible_sets_by_phenotype` | Get all GWAS associations for a phenotype |
| `get_credible_set_by_id` | Get all variants in a specific credible set |
| `get_credible_sets_by_qtl_gene` | Get QTL associations where a gene is the molecular trait. `summarize` defaults to **true** (credible set-level), like the sibling credible-set tools — see Architecture decision 7. Also the correct tool for **gene-based caQTL** questions: a caQTL trait is a chromatin peak, and the underlying `all_cs_qtl_file` resolves the Open4Gene peak-to-gene link (cell-type-matched), so `trait` holds the linked gene symbol while `trait_original`/`cs_id` keep the peak id. Peak-vs-gene coordinate matching is NOT a substitute — linked peaks sit up to ~1 Mb away and most peaks near a gene are not linked to it |
| `get_credible_sets_stats` | Get summary statistics of credible sets for a dataset |

### Gene data tools

| Tool | Description |
|------|-------------|
| `get_gene_expression` | Get tissue-specific gene expression levels |
| `get_gene_disease_associations` | Get Mendelian disease relationships from ClinGen/GENCC |
| `get_exome_results_by_gene` | Get rare variant burden test results (genebass filtered to p < 1e-4, IBD exome-wide significant only) |
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
| `get_open_chromatin_by_variant`, `get_open_chromatin_by_region`, `get_open_chromatin_by_gene` | Measured open-chromatin **atlas** peaks (scATAC/snATAC/bulk-ATAC/chromHMM) overlapping a variant position, a region, or a gene's window, labelled by `cell_type`, `tissue`, `life_stage` and `condition` so cell-type specificity can be reported. `resources`: `marderstein`, `li_brain_atac`, `catlas`, `epimap`, `calderon_immune`, `rosmap_brain` |
| `get_variant_effect_by_variant`, `get_variant_effect_by_gene` | In-silico **predicted** variant effect on chromatin accessibility: ChromBPNet (`model=chrombpnet`) per cell type/tissue with `score`/`mlog10p`/`quantile_rank`/`is_significant`, FLARE (`model=flare`) as a pan-context score with null cell type. `resources`: `marderstein` |
| `get_mpra_by_variant`, `get_mpra_by_region`, `get_mpra_by_gene` | **Measured** cis-regulatory allelic activity from a massively parallel reporter assay (Siraj et al. 2026). One long row per `cell_line` — `meta` (cross-cell-line meta-analysis) or K562/HEPG2/SKNSH/HCT116/A549 — carrying `emVar` (allele modulates reporter expression), `active`, `log2Skew` (signed allelic effect), `log2FC`, and their `*_mlog10p`. Coverage is partial (fine-mapped GTEx/UKBB/BBJ plus control common variants), so absence ≠ no effect. `resources`: `siraj_mpra` |
| `get_mpra_pip_concordance_by_gene` | Joins FinnGen fine-mapped credible sets (`credible_sets_v`, filtered to `resource` + `pip >= min_pip`, default 0.1) to the MPRA cross-cell-line meta row on the shared variant key, for variants near a gene — the regulatory-buffering check of whether credibly causal variants are measurably active. Ordered `emVar` then PIP. Distinct from `get_mpra_by_gene`, which returns MPRA rows without the PIP cross-reference |

### Other genetics tools

| Tool | Description |
|------|-------------|
| `get_colocalization` | Find traits sharing causal signals at a variant |
| `get_phenotype_report` | Get detailed markdown report for a phenotype. Disabled by default — enable with `ENABLE_PHENOTYPE_REPORT` |
| `list_datasets` | List all datasets with descriptions, provenance, sample-size stats, and supported products |
| `get_summary_stats` | Get summary statistics (p-value, beta, SE, allele frequencies) for specific variant-phenotype pairs |
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
| `general` | Always available: search_phenotypes, search_genes, lookup_variants_by_rsid, lookup_phenotype_names, list_datasets, search_scientific_literature, web_search, search_mgi, search_cbioportal, get_protein_annotations, map_protein_variants, get_variant_protein_effect, search_uniprot, create_phewas_plot, get_gene_group_members, normalize_gene_symbols |
| `api` | Local genetics API tools: credible sets, gene data, colocalization, phenotype report, variant annotations, etc. |
| `bigquery` | BigQuery SQL tools: query_database, get_database_schema |
| `orchestration` | Main-agent-only tools: launch_subagents. Excluded from subagent tool sets to prevent recursive launches. |

### Profile behavior

| `tool_profile` value | Local tools | External tools |
|----------------------|-------------|----------------|
| `null` (default) | general + api + bigquery + orchestration | always-on (gnomAD, OT) + RAG |
| `"api"` | general + api + orchestration | always-on only |
| `"bigquery"` | general + bigquery + orchestration | always-on only |
| `"rag"` | general only | RAG only |

Always-on external servers (gnomAD, Open Targets from `EXTERNAL_MCP_SERVERS`) are included in every profile except `"rag"`. The RAG server (`RAG_MCP_SERVER`) is only included when `tool_profile` is `"rag"` or unset.

## Response Length

The chat API takes a `verbosity` parameter (`"brief"` — the default — or `"detailed"`), surfaced in the web UI as the **Answer** radio group beside the literature-backend and tool-profile selectors. `chat_api.stream_chat` appends the matching fragment from `_VERBOSITY_PROMPTS` (`config/defaults.py`, via `verbosity_prompt()`) to the end of the system prompt.

| `verbosity` value | Effect on the write-up |
|-------------------|------------------------|
| `"brief"` (default, and the fallback for null/unrecognized values) | Report the three-pass analysis as its conclusions: the answer, the rows that carry it, and interpretation-changing caveats. Retrieved-but-unused data is left to the `INCLUDE_IN_RESPONSE` download links, with a one-line note of what was held back |
| `"detailed"` | The full pass-by-pass write-up — complete data extraction, then literature, then analysis, with the per-source inventory |

**The setting scopes presentation, never method or rigor.** The three-pass approach under "Analyzing data" and every grounding rule apply identically at both settings; only the volume of what gets printed changes. An unrecognized value falls back to `"brief"` rather than raising, since a presentation preference must not fail a chat turn.

Both fragments sit inside the cached system block, so each setting keeps its own prompt-cache entry instead of invalidating the other's. The setting is per-request and is **not** persisted per message the way `literature_backend` and `tool_profile` are (`chat_messages` has no `verbosity` column) — a resumed conversation uses whatever the selector currently shows.

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
│   ├── uniprot.py       # UniProtKB / EBI Proteins API client (TTL cache, accession/symbol resolution)
│   └── phewas_categories.py  # PheWAS plot category mappings
├── subagent.py             # parallel subagent service
├── scripts/
│   ├── analyze_variants.py # standalone variant list analysis CLI
│   ├── analyze_conversations.py # conversation history analysis and eval extraction
│   ├── analysis_timeseries.py  # shared rolling-window aggregation used by both renderers
│   ├── plot_conversation_scores.py # time-series plots of quality over time (from metrics.json)
│   ├── backfill_metrics_dates.py # one-off: join session created_at into an older metrics.json
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
| `usage` | Context usage snapshot after each agentic loop iteration | `iteration`, `input_tokens`, `output_tokens`, `total_input_tokens`, `total_output_tokens`, `context_window`, `context_percent` |
| `image` | Base64-encoded image (e.g., PheWAS plot) | `content` (base64 string) |
| `error` | Error message from the backend | `content` (error string) |
| `done` | Signals the stream is complete | `message_content` (assistant text + `tool_use` blocks for persistence), `tool_results` (the `tool_result` blocks for this turn, for persistence) |

The `usage` event is emitted by `_stream_anthropic()` in `llm_service.py` after token accounting in each iteration of the agentic loop. It is yielded as a `StreamChunk(type="usage")` with a JSON-serialized payload. The `event_generator()` in `chat_api.py` forwards it as an SSE event, spreading the usage fields into the top-level payload alongside `"type": "usage"`.

Payload fields for `usage`:
- `iteration` — current agentic loop iteration number
- `input_tokens` — input tokens consumed in the current API call
- `output_tokens` — output tokens generated in the current API call
- `total_input_tokens` — cumulative input tokens across all iterations
- `total_output_tokens` — cumulative output tokens across all iterations
- `context_window` — total context window size for the model (from `get_context_window()`)
- `context_percent` — percentage of context window consumed (`total_input_tokens / context_window * 100`)

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
| `INTERNAL_API_SECRET` | Shared secret sent as `Authorization: Bearer` on every call to results-api and the BigQuery proxy, for deployments where those services require internal auth. Only attached to `ToolExecutor.client` — the separate `external_client` carries no default auth, so the secret can never leak to a third-party API such as MouseMine or myvariant.info | - |
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
| `MAX_ATTACHMENTS_PER_MESSAGE` | Max attachment blocks (image/document/inlined data file) per message | `10` |
| `DOWNLOAD_STORAGE_PATH` | Path for tool result download files | `/mnt/disks/data/downloads` |
| `DOWNLOAD_TTL_SECONDS` | TTL for download files in seconds | `2592000` (30 days) |

**Both SQLite databases run in WAL mode**, set by `_create_connection` in `chat_history_db.py`
and `llm_config_db.py`. Under the default rollback journal a reader is refused while a write is
being applied (commit takes an EXCLUSIVE lock), and a writer's commit is refused while any
reader still holds a read transaction; WAL removes both directions. For `llm_config.db` the
mixed read+write hot path is API-token validation: `validate_api_token` runs a SELECT plus a
bookkeeping UPDATE and COMMIT on every MCP request (`mcp_server.py:_validate_user_token`, and
`routers/api_tokens.py` for the cross-pod HTTP fallback). The per-request settings and
tool-description reads and writes in `routers/llm_config.py` hit the same file from the same
process. Journal mode is a property of the file, not of the connection: the pragma converts an
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
string-matches the description in force. The `user_comments` accessors are deliberately left out
of this: their `created_at` is still naive, naive UTC out of the reads and naive *local* out of
`add_user_comment`. That is a scope decision, not a technical obstacle: making them aware would
leave the feedback feed's merge key total and in the same order, because both stamps render as a
fixed 19-character `YYYY-MM-DDTHH:MM:SS` that decides every comparison except an exact tie, where
the added `+00:00` would only flip which of the two sources wins — arbitrary either way. The
change is owned by `genetics-results-suite-ni9`.

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
| `GOOGLE_TOKEN_AUDIENCE` | Comma-separated OAuth client ids a Google Identity Token's `aud` must be one of. Unset means the audience is not checked at all |
| `ALLOWED_EMAILS` | Comma-separated email allow-list shared by all JWT bearer paths (Google Identity Token and Keycloak) |
| `ALLOWED_EMAIL_DOMAINS` | Comma-separated email-domain allow-list shared by all JWT bearer paths (default: `finngen.fi`) |
| `OAUTH_ISSUER` | Keycloak realm issuer URL; enables the OAuth resource-server bearer path when set together with `OAUTH_RESOURCE_URL` |
| `OAUTH_RESOURCE_URL` | Expected `aud` claim (this server's canonical URL) for Keycloak access tokens |
| `OAUTH_JWKS_URI` | Override for the JWKS endpoint; defaults to `<OAUTH_ISSUER>/protocol/openid-connect/certs` |

Tokens are supplied as an `Authorization: Bearer XXX` header. A `?token=XXX` query parameter is also accepted, but only when `MCP_ALLOW_QUERY_TOKEN` is set — off by default, because a credential in a URL is captured where request headers are not: the GKE load balancer's request logs (upstream of the auth-gateway's `token=` redaction), browser history, and the `Referer` header of any outbound link. Enable it only for a client that genuinely cannot set the header (e.g. claude.ai). The query parameter is consulted only when no Bearer header is present, so the header always takes precedence, and a request with neither gets 401.

The bearer auth middleware (`_wrap_with_bearer_auth` in `mcp_server.py`) routes each presented token through four branches in order, mirroring the results-api implementation:

1. **`MCP_API_KEY` shared secret** — constant-time compare against each configured value
2. **Keycloak OAuth access token (JWT)** — only attempted when `OAUTH_ISSUER` and `OAUTH_RESOURCE_URL` are both set (`settings.oauth_enabled`). If the token contains `.` it is verified with PyJWT against Keycloak's JWKS (fetched and cached via a per-URI singleton `jwt.PyJWKClient`): RS256 signature, `iss == OAUTH_ISSUER`, `aud` includes `OAUTH_RESOURCE_URL` (string or list), and `exp` not expired. The email is taken from the `email` claim (falling back to `preferred_username` only when it is itself an email) and checked against the same `ALLOWED_EMAILS` / `ALLOWED_EMAIL_DOMAINS` allow-list. Any failure (wrong iss/aud/signature, expired, or a JWKS network error) is non-fatal and **falls through** to branch 3 rather than 500-ing.
3. **Google Identity Token (JWT)** — if the token contains `.` it is validated via `google.oauth2.id_token.verify_oauth2_token` using a lazily-initialized singleton `google.auth.transport.requests.Request` (for JWKS caching). The payload must have `email_verified == True`; the email must satisfy the same allow-list (otherwise 401/403). Identity is set to the verified email. `verify_oauth2_token` **skips the `aud` claim when no audience is passed**, so `_audience_allowed` additionally requires `aud ∈ GOOGLE_TOKEN_AUDIENCE` — without it a token minted for an unrelated application would be accepted as long as its email is allow-listed. The check is inert (with a warning logged per token) while `GOOGLE_TOKEN_AUDIENCE` is unset; the deployment sets it to the gcloud CLI client id, which is what `gcloud auth print-identity-token` issues.
4. **Per-user API token** — fall back to validating against the local LLM config DB (SHA-256 hashed) or via the chat-backend `/v1/tokens/validate` endpoint. Users create tokens via the chat API (`POST /chat/v1/tokens`).

**Token expiry is idle-based, not absolute.** `user_api_tokens.expires_at` is a *rolling* deadline: `validate_api_token` rejects a token once the deadline has passed, and otherwise pushes it forward to `now + API_TOKEN_TTL_DAYS` (default 90; `0` disables expiry) in the same statement that updates `last_used_at`. A token in regular use therefore never expires, while an abandoned or quietly-leaked one stops working on its own. Rows predating the column have `expires_at IS NULL` and are judged on `COALESCE(last_used_at, created_at) + TTL`, so an actively-used legacy token is not killed the first time it is presented. Timestamps are written in SQLite's own `%Y-%m-%d %H:%M:%S` UTC form so the column stays lexicographically sortable alongside `CURRENT_TIMESTAMP` values; `_as_utc` accepts both that and ISO-8601 when reading. Pushing the deadline is bookkeeping, not part of the decision: it is the one write in `LLMConfigDB` whose failure is logged and swallowed rather than raised, so a locked or full database cannot reject a token the SELECT has already accepted. Every other write accessor rolls back and re-raises, and each of them (reads included) discards a transaction found open on the thread's cached connection, so no failure can hold the write lock against the other writers of `llm_config.db`.

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
| `test_db.py` | Database operations, LLM-config write transaction safety, LLM-config journal mode (WAL, and the reader/writer concurrency it buys), same-second tiebreak in the tool-description, user-setting and user-comment accessors (`changed_at`/`created_at` have one-second resolution, so the later `id` wins; both row orders, blank timestamps, several keys tied at once), and malformed-stamp reads (a NULL or unparseable `changed_at` degrades to the epoch in the singular and plural accessors alike rather than raising or dropping the key, and a group holding both a NULL and a sentinel stamp, `''` or `0` — the one shape that separates the `IS` join from a coalescing one — resolves to the same row in both), and the zone the write path returns (the saves and the version history hand back aware UTC, as the reads do) |
| `test_chat_history_router.py` | Chat history API |
| `test_llm_config_router.py` | LLM config API |
| `test_llm_config_db_migration.py` | One-shot import of legacy per-user instructions into instruction sets |
| `test_instruction_sets_db.py` | Instruction-set accessors: per-user scoping, write-time caps (including a concurrent-create race), over-cap rows reported not truncated, history, archiving, ordering, timestamp degradation, transaction safety (rollback on failure or on a failed commit, update racing an archive, update's read-modify-write under the write lock, reads never returning uncommitted rows) |
| `test_llm_service.py` | Replayed-history helpers: `tool_use`/`tool_result` pairing, marker stripping, cache breakpoint, truncation item counting |
| `test_phewas_categories.py` | PheWAS category mappings |
| `test_subagent.py` | Subagent service, skills, sandbox tools |
| `test_variant_analysis.py` | Variant list analysis tool |
| `test_downloads.py` | Download store, TSV conversion, download endpoint |
| `test_bigquery_gene_tools.py` | BigQuery-backed gene tools |
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

Run tests:
```bash
pytest
pytest --cov=src/genetics_mcp_server  # with coverage
```

## Conversation Analysis

`scripts/analyze_conversations.py` is an offline tool that reads the chat-history
SQLite DB, persists per-conversation analysis results back into that DB (the
`conversation_analysis` / `conversation_issue` tables), and produces a markdown
report (`report.md`) plus an eval dataset. With `--output-dir` it also writes a
local-dev `metrics.json` (consumed by `plot_conversation_scores.py` for
quality-over-time plots).

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
8. **Downloadable results**: Tools returning tabular data include `INCLUDE_IN_RESPONSE` download links. Direct API URLs are used for genetics API tools that support TSV format; other tools (BigQuery, LD, summary stats) have their results converted to TSV and stored on disk, served via `/chat/v1/downloads/{id}`. The `_download_url` and `_download_data` hints in tool results are processed by `_process_download_hints()` in `llm_service.py` before being sent to the LLM. All download links use relative URLs (e.g., `/api/v1/...` or `/chat/v1/downloads/...`) so they work correctly regardless of deployment domain. `INCLUDE_IN_RESPONSE` is placed at the front of the result dict so it survives JSON truncation for large results. For BigQuery, trailing SQL `LIMIT` clauses are stripped and `max_rows` is set to 100,000 so the download contains the full result set even when the LLM only displays a subset. The BigQuery proxy (`genetics-results-db`) enforces `MAX_ROWS=100000` as a hard cap.
9. **Tool result persistence (resumed conversations carry the data substrate)**: The chat API is stateless per request — the frontend replays the full conversation each turn. Tool `tool_result` blocks are persisted (`chat_messages.tool_results_json`, added via the standard PRAGMA/ALTER migration) so a resumed conversation replays the actual tool outputs the model saw, not just its prose summary. `_stream_anthropic` collects `all_tool_results` across agentic-loop iterations and emits them in the `done` SSE event; the frontend stores them and, on resume, rebuilds the `assistant(tool_use) → user(tool_result)` pairing (its history builder splits each persisted assistant turn into the assistant message plus a synthetic user message of `tool_result` blocks). The already-truncated, image-base64-stripped result content is stored as-is. **Backward compatible**: conversations saved before this feature have `tool_results_json = NULL`; on resume they emit only the assistant message and `_sanitize_tool_blocks` (in `llm_service.py`) strips the now-orphaned `tool_use` blocks — exactly the prior behavior. **Marker-strip safeguard**: the `*[Using tool: …]*` annotations injected during streaming are display-only, but they are persisted into the assistant text. Before history reaches the model, `_strip_tool_use_markers` (in `llm_service.py`, run just before `_sanitize_tool_blocks`) removes them from replayed assistant content (both string and text-block forms). Without this, a long/repetitive conversation could teach the model to imitate the notation — writing `*[Using tool: X]*` as prose instead of emitting a real `tool_use` block, then fabricating the result (observed in a real session whose tool-less turns predated the persistence fix). Real `tool_use` blocks are left untouched. To offset the larger replayed payload, `_mark_history_cache_breakpoint` adds a `cache_control: ephemeral` breakpoint on the last replayed message (the 3rd of Anthropic's 4 breakpoints, alongside the system prompt and tool definitions). System-prompt guardrails (`config/defaults.py`) additionally instruct the model to treat credible-set membership as distinct from LD and to re-query authoritative tools for count/membership/lead questions rather than relying on earlier summaries.

## Future considerations

1. Add caching for genetics API responses
2. Support additional LLM providers (Google, local models)
3. Implement tool-level access control
4. Add WebSocket transport for bidirectional streaming
