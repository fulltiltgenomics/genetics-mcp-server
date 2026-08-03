# cBioPortal integration — API verification record

Status: **implemented** as `search_cbioportal` (2026-08-03). The design and its
rationale live in `docs/project-spec.md`; this file is the measurement record
behind it — what was probed against the live API, what it cost, and which numbers
the implementation was validated against.

Everything below was measured against `https://www.cbioportal.org/api` on
2026-08-03, not read off the documentation.

## Access

| Check | Result |
|---|---|
| Authentication | None for public data. Bearer tokens exist only for private institutional instances. |
| Rate limiting | 8 rapid sequential calls: all HTTP 200, ~350 ms each, no throttling, no quota headers. |
| Corpus | 539 studies, 399,909 samples, 2,534 molecular profiles (533 mutation, 449 CNA, 215 structural variant). |
| Reference genome | 467 studies hg19 / 72 hg38; 375,384 vs 24,525 samples. Records carry a per-record `ncbiBuild`, mixed within one gene's results (TP53: 134,010 GRCh37, 7,824 GRCh38, 3 "37", 1 "NA"). No liftover. |
| Caching headers | `no-cache, no-store` — cache in-process, not via HTTP. |
| Licence | ODC Open Database License. Reuse with attribution to cBioPortal and the originating studies. `/api/studies` carries `citation` and `pmid` per study. |
| Spec | `https://www.cbioportal.org/api/v3/api-docs`, Swagger UI at `/api/swagger-ui/index.html`. |

## Endpoint costs

| Endpoint | Measured |
|---|---|
| `GET /genes/{hugoSymbol}` | 0.35 s |
| `POST /mutation-data-counts/fetch` | 1.2 s / **300 bytes**, panel-aware per-gene counts |
| `POST /genomic-data-counts/fetch` (CNA) | ~1 s, discrete CNA levels per gene |
| `POST /clinical-data-counts/fetch` | 1.3–1.4 s / 6–11 KB, counts by `CANCER_TYPE` |
| `POST /mutation-counts-by-position/fetch` | instant |
| `POST /structural-variant/fetch` | ALK pan-cancer: 1.3 s / 2.2 MB / 1,665 records |
| `POST /mutations/fetch` `projection=META` | 0.4 s, `total-count` header only |
| `POST /mutations/fetch` `projection=SUMMARY` | PROC 645 records: 1.4 s / 452 KB. **TP53 141,838 records: 12.3 s / 102 MB** |
| `POST /mutations/fetch` on curated cohort | TP53: 23,055 records / 16.7 MB / 2.7 s |
| `POST /mutated-genes/fetch` | all 539 studies: 5.2 s / 6.7 MB / 23,813 genes |
| `POST /cna-genes/fetch` | all studies: 7.4 s / 24 MB |
| `POST /structuralvariant-genes/fetch` | all studies: 2.9 s / 5.4 MB |
| `GET /studies`, `/cancer-types`, `/molecular-profiles` | small, cached |

The `*-genes/fetch` family returns every gene regardless of filter, which is why
the implementation uses the per-gene count endpoints instead — same numbers, three
orders of magnitude less transfer.

## Validation

Each number the implementation produces was checked against an independent
cBioPortal figure:

| Claim | Cross-check |
|---|---|
| EGFR pan-cancer mutation | `mutation-data-counts` 17,483/366,522 vs authoritative `numberOfProfiledCases` 366,522 — exact |
| EGFR altered-sample count | `clinical-data-counts` numerator sums to 17,474 vs `mutated-genes/fetch` `numberOfAlteredCases` 17,474 — exact |
| Per-study aggregation | distinct EGFR-altered samples in 37 lung studies = 3,373 vs scoped `mutated-genes/fetch` 3,373 — exact |
| Mutation-profiled denominator | `genomicProfiles: [["mutations"]]` sums to 367,336 = the `mutations` profile sample count |
| Biology | EGFR 21.6% in NSCLC; TP53 hotspots R175H > R248Q > R273C; EML4 the top ALK fusion partner |
| Build labelling | TP53 R175H returned at chr17:7,578,406 under `GRCh37` (the GRCh38 position is 7,675,088) |

## Traps found, and what was done about them

1. **Unrestricted `clinical-data-counts` denominators count unsequenced samples**
   (400,081 vs 367,336). Fixed with `genomicProfiles: [["mutations"]]`. Without
   this every per-cancer-type frequency is silently ~9% low, and much worse in
   poorly sequenced cancer types.
2. **`genomicDataFilters` with categorical values (`MUTATED`) silently returns
   zeros** rather than erroring — it is for continuous data only. The panel-aware
   per-cancer-type denominator is therefore not available; `gene_by_cancer_type`
   documents its frequencies as lower bounds instead of pretending otherwise.
3. **Grouping by study cancer type loses the pan-cancer cohorts.** MSK-IMPACT
   (~10k samples) has `cancerTypeId: mixed`. Sample-level `CANCER_TYPE` is used.
4. **`mutation-data-counts/fetch` ignores `profileType`** and always answers for
   mutations — passing `cna` returns mutation counts, not an error. CNA goes
   through `genomic-data-counts/fetch`.
5. **Free-text cancer-type labels split real cohorts** ("Non Small Cell Lung
   Cancer" vs "Non-Small Cell Lung Cancer"). Folded on a normalized key.
6. **`no-cache` headers plus 100 MB worst-case responses.** META pre-flight and
   the curated-cohort fallback bound it; metadata is cached in-process.

## Testing note

The worktree's `.venv` has an editable install pointing at the parent checkout,
so `uv run pytest` inside a worktree imports the parent's sources and the new
tests fail with `AttributeError`. Run them as:

    PYTHONPATH=$PWD/src uv run pytest -q
