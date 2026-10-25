# MR replication fixtures

Published Mendelian-randomization tables the `genetics.mr` work is measured against, as TSV.
Derived, never hand-edited: regenerate with

```bash
cd genetics-mcp-server
uv run python -I tests/fixtures/mr/convert_fixtures.py <workbook-dir>
```

where `<workbook-dir>` holds the downloads below in their per-paper subdirectories (on the daly
dev VM: `~/mr_fixtures_untrusted/`, whose `MANIFEST.md` records how they were fetched). The
workbooks are untrusted downloads: the script reads their sheet XML with the standard library
under `python -I`, evaluates nothing (a formula cell yields only its cached value), refuses any
workbook whose sha256 differs from the one pinned in it, and matches every header by content. Output is deterministic; `SHA256SUMS` here is written by the script and is what
`genetics-results-suite/scripts/mr-proving-ground.py --groups ground` checks.

How a cell is written follows the type the workbook stores for it:

- a number with an integral value is written as an integer (positions, sample sizes, counts);
  any other number as Python's `repr(float)`, the shortest text that parses back to the exact
  double the workbook stores. Nothing is rounded: a `0` is a stored zero.
- a text cell is written verbatim, including text that looks numeric, such as the
  `OR(95%CI)` strings and the ORs parsed out of them.
- an Excel error cell is written as its literal, `#N/A` in these sheets (Zheng TableS1 has
  them throughout its validation columns), and a blank cell as the empty string, so the two
  never read alike.

Free-text cells that wrap inside the workbook have the wrap folded to a space (one cell, a
protein name in Zheng TableS1).

## Sources

URL pattern: `https://static-content.springer.com/esm/art%3A10.1038%2F<doi-suffix>/MediaObjects/<file>`

| paper | doi | file | sha256 |
|---|---|---|---|
| Zhang X et al. 2025, Nat Commun 16:2813 | 10.1038/s41467-025-57879-3 | `zhang2025_natcommun/41467_2025_57879_MOESM3_ESM.xlsx` | `78f66fa2c9e01cffaf2f0dad5a47fe27418397a915d125e6e5708debc7c60449` |
| Zhou S et al. 2021, Nat Med 27:659 | 10.1038/s41591-021-01281-1 | `zhou2021_natmed/41591_2021_1281_MOESM2_ESM.xlsx` | `dff7cbfbf5f01988032c87a2698498bcef35d92294280542ffd806a7ee030e6c` |
| Zheng J et al. 2020, Nat Genet 52:1122 | 10.1038/s41588-020-0682-6 | `zheng2020_natgenet/41588_2020_682_MOESM3_ESM.xlsx` | `0812c04ca38150e67d6eae1bde3863331ffd61fc0f7634e1120780edd8328a14` |

## Sheets and columns

**`zhang2025_data8_instruments.tsv`** — Zhang Supplementary Data 8, the UKB-PPP cis-pQTL
instruments. One row per protein-SNP; the trailing footnote row is dropped, and the Olink,
UniProt, panel, R² and F-statistic columns are not carried.

| TSV | workbook |
|---|---|
| protein | Protein |
| rsid | SNP |
| chr | chr.exposure |
| pos_grch38 | pos.exposure (GRCh38: instruments resolve in the finngen_r12 panel at the workbook position, e.g. 19:58347310 A/G, 7:77284278 T/C, 11:116778822 A/G, and the reversed allele order is refused) |
| ea, oa | effect_allele.exposure, other_allele.exposure |
| eaf | eaf.exposure |
| beta, se, p | beta.exposure, se.exposure, pval.exposure |
| n | samplesize.exposure |

**`zhang2025_data9_mr.tsv`** — Zhang Supplementary Data 9, the kill-criterion comparison. The
sheet has three side-by-side blocks, IBD / CD / UC, each repeating the protein; the script
asserts the three name the same protein on every row. Per block `<o>` in `ibd`, `cd`, `uc`:
`<o>_observational` ("Significance in the observational study"), then for each study `<s>` in
`iibdgc`, `finngen`, `combined`: `<o>_<s>_or_ci` (the "OR(95%CI)" string verbatim),
`<o>_<s>_or`, `<o>_<s>_ci_low`, `<o>_<s>_ci_high` (parsed out of that string, as written), and
`<o>_<s>_p` ("P-value"); then `<o>_fdr` ("FDR", which the sheet gives for Combined only). A
protein marked `No` keeps its row with every MR field empty.

**`zhang2025_data13_decode_mr.tsv`** — Zhang Supplementary Data 13, the replication with
deCODE (SomaScan) cis-pQTLs. Like Data 9 the sheet has three side-by-side blocks, IBD / CD /
UC, but each block lists its own proteins in its own order (274 / 146 / 63), so the blocks are
stacked long: one row per `outcome` (`ibd`, `cd`, `uc`, from the block's header) and
`protein` ("Proteins"). For each study `<s>` in `iibdgc`, `finngen`, `combined`: `<s>_or_ci`
(the "OR(95%CI)" string verbatim), `<s>_or`, `<s>_ci_low`, `<s>_ci_high` (parsed out of it),
and `<s>_p` ("P-value"). The sheet names no instrument SNP and has no FDR or observational
column. The IBD block's trailing footnote is dropped; past a block's first break the script
accepts only rows with text in the first column alone (a footnote), and it refuses a block
naming a protein twice. A protein with no IIBDGC estimate keeps its
row with those fields empty.

**`zhou2021_st2_somamer_a2.tsv`** — Zhou Supplementary Table 2, INTERVAL SOMAmer MR on very
severe COVID-19 (A2). The sheet names no SNP; `n_snp` is the number of independent cis-SNPs
in the test. Columns: somamer (SOMAmers), method (Method), n_snp (#SNP), beta (Beta), se
(Se), p (P value), p_fdr (P FDR), or (OR), ci_low / ci_high (95% CI lower / upper bound). The
script refuses a sheet in which OAS1.10361.25.3 is not the single row with OR 0.544.

**`zheng2020_s1_instruments.tsv`** — Zheng TableS1.Instruments, every column with the
workbook's own header names. The sheet has `Platform_id` twice; the second is
`Platform_id_2`.

**`zheng2020_s9_multicis.tsv`** — Zheng TableS9.multi-cis-top, both of its tables stacked with a
leading `table` column (`9A`: multi-cis agrees with single-cis; `9B`: multi-cis gains power),
every other column with the workbook's header names.

The R reference outputs the estimators are tested against, and the inputs built from these
fixtures for them, are in `r_reference/` (its own README).
