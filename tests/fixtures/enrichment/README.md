# gene set enrichment reference outputs

R reference outputs that `genetics.enrichment` is to be held to. Inputs are synthetic and
small; the committed files are the fixture, `make_inputs.R` only records how they were made.

Image: `rocker/r-ver:4.4.3` (Ubuntu noble, CRAN pinned to the p3m snapshot of 2025-04-10),
Bioconductor 3.20, fgsea 1.32.4, BiocParallel 1.40.2. `reference.log` ends with the full `sessionInfo()`.

## Inputs

- `gene_sets.gmt` — 30 sets drawn from `G0001..G0800` and `X001..X100`.
- `universe.txt` — `G0001..G0900`. `G0801..G0900` are in no set; `X*` genes are in sets but
  outside the universe.
- `gene_list.txt` — the ORA query, enriched for `SET10` and `SET20`, with five `X*` genes.
- `ranked_stats.tsv` — one statistic per universe gene, `SET16` shifted up, `SET17` down.
- `gene_stats.tsv` — gene `z` and a size-like covariate `log_len`, `SET15` shifted up.

Every analysis uses `minSize = 5`, `maxSize = 300`, so `SET01` (3 genes) and `SET02` (400)
are dropped.

## Universe convention

ORA restricts each set and the gene list to the universe `U`, keeps the sets whose size
**after** that restriction is within `[5, 300]`, takes
`P(X >= k)` with `X ~ Hypergeom(|U|, |set ∩ U|, |list ∩ U|)`, and applies BH over the sets
that passed the filter. The tools disagree about `U`:

- `*_given` columns in `ora_phyper.tsv`: `U` is the universe as given. This is what
  `fgsea::fora` computes (`ora_fora.tsv`), and the two agree to machine precision.
- `*_annotated` columns: `U` is the given universe intersected with genes annotated to any
  set in the collection (before the size filter), which is what `clusterProfiler::enricher`
  does by default. clusterProfiler is not installed; this is the same `phyper` call with the
  smaller `U`.

Set sizes and overlaps are identical under both, since every set is a subset of the
annotated genes; only `|U|` and `|list ∩ U|` change, and the p-values differ by far more
than 1e-6.

## Outputs

- `ora_phyper.tsv`, `ora_fora.tsv` — ORA as above (deterministic).
- `gsea_fgseaSimple_nperm10000_seed1.tsv` — `fgseaSimple`, `nperm = 10000`, `gseaParam = 1`,
  `set.seed(1)`, serial `BiocParallel`. `ES` and `leadingEdge` are deterministic. `NES` is **not**:
  it divides `ES` by the mean of the same-signed permutation ESs, so it carries Monte Carlo
  noise like `pval`.
- `gsea_fgseaMultilevel_eps0_seed1.tsv` — `fgseaMultilevel`, `eps = 0`, `set.seed(1)`: a
  more accurate p-value for small p, to compare permutation p-values against within noise.
- `competitive_lm.tsv` — per set, `lm(z ~ in_set + log_len)` over every gene in
  `gene_stats.tsv`, with the one-sided p `pt(t, df, lower.tail = FALSE)` for the membership
  coefficient (MAGMA's competitive gene-set test; deterministic).

## Regenerating

```
docker build -f Dockerfile.r-reference -t enrichment-ref .
docker run --rm -v "$PWD:/out" enrichment-ref Rscript /out/make_inputs.R
docker run --rm -v "$PWD:/out" enrichment-ref Rscript /out/reference.R > reference.log
```

`make_inputs.R` overwrites the inputs; run only the last line to regenerate outputs from the
committed inputs.
