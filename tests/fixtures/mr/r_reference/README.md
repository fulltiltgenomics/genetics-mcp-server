# MR reference outputs (MendelianRandomization + TwoSampleMR)

Reference numbers `genetics_mcp_server.sdk.mr` is held to. Per decision D4 of the MR epic the
estimates follow MendelianRandomization and Steiger, leave-one-out and single-SNP follow
TwoSampleMR; both packages run on every table so the two conventions can be compared.

## Image

`Dockerfile.r-reference`: `rocker/r-ver:4.5.1` (R 4.5.1, Ubuntu 24.04, OpenBLAS 0.3.26). CRAN
packages come from the Posit Package Manager snapshot **2026-10-01**
(`https://p3m.dev/cran/__linux__/noble/2026-10-01`, binaries), written into `Rprofile.site`.
TwoSampleMR is not on CRAN and r-universe cannot be pinned by date, so it and its non-CRAN
imports are installed from GitHub at the commits mrcieu.r-universe.dev had built on 2026-10-10:

| package | version | commit |
|---|---|---|
| MendelianRandomization (CRAN snapshot) | 0.10.0 | |
| TwoSampleMR | 0.7.12 | MRCIEU/TwoSampleMR@fe10242 |
| ieugwasr (CRAN's 1.1.0 is below TwoSampleMR's >= 1.2.0) | 1.2.0.9000 | MRCIEU/ieugwasr@90cafe1 |
| MRMix | 0.1.0 | gqi/MRMix@56afdb2 |
| MRPRESSO | 1.0 | rondolab/MR-PRESSO@3e3c92d |
| RadialMR | 1.2.5 | WSpiller/RadialMR@7ca35c0 |

The versions of ieugwasr, MRMix, MRPRESSO and RadialMR cannot be checked against
`sessionInfo.txt`: they are never loaded at run time. They were read from `installed.packages()`
(`RemoteSha`); the Dockerfile pins the commits.

The build asserts the two package versions. `sessionInfo.txt` is `sessionInfo()` from the run
that wrote these outputs (every loaded package and its version). The build took 124 s on a
4-core host; a run of `reference.R` takes about 3 minutes (the mode bootstraps dominate).

## Inputs

- `input_cis_ibd.csv`: one row per instrument, 38 rows for 10 proteins (IL12B 7, CD6 5,
  ANGPTL3 4, CD55 4, FAS 4, CD274 3, CD40 3, CXCL10 3, IL18 3, RSPO3 2). Exposure: every
  instrument for those proteins in `../zhang2025_data8_instruments.tsv` (Zhang 2025 Data 8,
  UKB-PPP, n about 34,000), none dropped. MXRA8, CXCL9 and IFNG have one Data 8 instrument
  each, so they are not here. Outcome: IIBDGC inflammatory bowel disease, resource `ibd_gwas`,
  phenotype `IBD` (the 2026 release, up to 125,992 cases), fetched on 2026-10-10 from the local
  dev stack (results-api :2000, db-api :8085) by `build_inputs.py`. rsIDs are resolved with
  `genetics.search(rsids=)` to the id carrying the instrument's alleles; the effect allele
  equal to the outcome row's alt keeps beta, equal to its ref flips beta and eaf, anything else
  would drop the instrument (none did). Columns: `protein`, `variant` (outcome row's
  chr:pos:ref:alt, GRCh38), `rsid`, `beta_x`/`se_x` (exposure, per alt allele, SD units),
  `beta_y`/`se_y` (outcome log-OR per alt allele), `af_x` (Zhang's eaf on alt), `af_y`
  (outcome row's af), `n_x` (Zhang's n), `n_y` = `ncase_y` + `ncontrol_y` (the outcome row's
  own counts, which vary per variant: 21,961 to 125,992 cases).
- `ld_<PROTEIN>.csv`: the signed r among that protein's instruments, finngen_r12, from
  `genetics.ld(v1, v2, panel="finngen_r12", fields="r", r2_threshold=0)`, one call per pair
  (r as served, 4 decimals). Square matrix; the header row and first column give the
  variant order. A pair below the panel's r2 0.01 floor is 0 — most pairs are, since Data 8
  instruments were clumped. Membership in the panel is probed with a self-pair call; a
  variant the panel lacks is left out of the matrix and so out of the correlated
  estimators only: **CXCL10 4:76027659:C:A (rs369112747)**, so CXCL10's correlated rows use 2
  of its 3 instruments. 8 of the 60 pairs are non-zero, across 5 proteins:
  ANGPTL3 1:62417463:T:C / 1:62360176:T:C 0.1374; CD274 9:5448218:T:C / 9:5505508:A:C -0.1106
  and 9:5448218:T:C / 9:5384069:A:T -0.1348; FAS 10:89023165:T:A / 10:88999045:C:T -0.1207;
  IL12B 5:159397166:C:G / 5:159432597:G:A -0.1705 and 5:159397166:C:G / 5:158318033:T:A 0.1673;
  IL18 11:112209661:T:C / 11:112406352:C:T 0.154 and 11:112209661:T:C / 11:112099396:C:A -0.1077.
- 8 of the 38 cis instruments are palindromic (A/T or C/G); FAS 10:89023165:T:A (eaf 0.585) and
  CD6 11:60449088:C:G (0.639) have mid-range frequencies, so strand cannot be inferred from
  them. `af_x` agrees with `af_y` for all 8 (same side of 0.5, within 0.025). The tables are a
  numeric yardstick (R and Python receive the same inputs), not a biological claim.
- `input_synthetic.csv`: from `make_synthetic.py` (standard library only, seed 20261010): 50
  variants, causal effect 0.3, 10 pleiotropic variants (column `pleiotropic`, true direct
  effect `alpha_true` ~ N(0.02, 0.01), independent of the exposure effect), n_x 30,000,
  n_y 50,000, both traits in SD units. Same columns as the cis table without
  `protein`/`rsid`/counts. No r matrix: the correlated estimators do not run on it.

## Outputs

Two tidy long CSVs; every number is `sprintf("%.17g")`, `NA` where the package returns NA.

- `out_estimates.csv`: `table, package, method, options, quantity, value`, one value per row.
- `out_per_variant.csv`: the same plus `variant`.

`table` is the protein or `synthetic`. `options` spells every non-default argument, seed and
iteration count. A call the package refuses is one row with `quantity = "error"` and its
message (Egger and median need 3 variants: RSPO3 and CXCL10's 2-variant correlated table).

| method | package | what |
|---|---|---|
| `wald` | base (per variant) | by/bx, first-order SE se_y/abs(bx); written out in R, no package function |
| `mr_ivw` `model=default/fixed/random` | MendelianRandomization | slots Model, Estimate, StdError, CI, Pvalue, Q and Q_pval (from `Heter.Stat`), RSE; `I2_from_Q` = max(0, (Q - (K-1))/Q) computed in `reference.R` from mr_ivw's Q, not a package slot. `default` is fixed for K <= 3, random for K >= 4 (the `Model` row says which) |
| `mr_ivw` `correl=TRUE;model=fixed/random` | MendelianRandomization | the same with `mr_input(correlation=)` |
| `mr_egger` `default`, `correl=TRUE` | MendelianRandomization | slope and intercept with SEs, CIs, p; Q; `I.sq` is I2_GX |
| `mr_median` weighted/simple | MendelianRandomization | iterations 10000, seed 314159265 |
| `mr_mbe` weighted/unweighted | MendelianRandomization | weighted and simple mode, stderror delta, phi 1, iterations 10000, seed 314159265 |
| `mr:<method>` | TwoSampleMR | `mr()` with its default method list, `set.seed(20261010)` first, nboot 1000, phi 1 |
| `mr_heterogeneity:*`, `mr_pleiotropy_test` | TwoSampleMR | Q for IVW and Egger, Egger intercept |
| `mr_singlesnp`, `mr_leaveoneout` (per variant) | TwoSampleMR | defaults (Wald ratio per SNP; IVW leaving each out, K > 2 only) |
| `get_r_from_bsen:*`, `get_r_from_lor:outcome`, `input` (per variant) | TwoSampleMR | the r each convention gives, and the ncase/ncontrol used |
| `steiger_filtering` (per variant), `directionality_test`, `mr_steiger` | TwoSampleMR | exposure `units = "SD"` (r from beta, se, n); the IBD outcome `units = "log odds"` with the row's ncase/ncontrol and `prevalence` 0.1 (TwoSampleMR's assumed default) and 0.005 (closer to IBD), both written; synthetic outcome in SD units |

p-values handed to TwoSampleMR are 2 * pnorm(-|beta/se|) computed in `reference.R`.

## Tolerances for the Python tests

- Deterministic and reproduced byte for byte by a second container run: everything below
  that is not a bootstrap SE. Closed-form estimators (Wald, IVW in every form, Egger, Q,
  I2, leave-one-out, single-SNP, the r conventions, Steiger) can be matched to ~1e-10.
- Weighted/simple median and mode **point estimates** are deterministic; their **SEs, CIs and
  p-values** come from seeded bootstraps (R's RNG), so a port matches them only within Monte
  Carlo error (a few percent at 10,000 iterations, more at TwoSampleMR's 1,000).
- The two packages disagree on the weighted median by design: MendelianRandomization weights
  by first-order inverse ratio variance, TwoSampleMR by second-order (synthetic: 0.33080 vs
  0.31802, both reproduced by hand). The mode estimates agree between the packages.
- `mr_mbe`'s bootstrap emits "number of items to replace is not a multiple of replacement
  length" (about 5,400 times per run): a resample with tied modes returns several values and
  R keeps the first. The mode SEs therefore depend on that tie-breaking, which is a package
  implementation detail, not a formula; treat them as version-dependent.
- Steiger on a binary outcome depends on the assumed prevalence; match the `prevalence=` row.

## Regenerate

From this directory. Inputs (needs the dev stack; the secret is read from its file, never
printed):

```
cd ../../../.. && GENETICS_API_URL=http://localhost:2000/api BIGQUERY_API_URL=http://localhost:8085 \
  INTERNAL_API_SECRET=$(cat ~/.cache/genetics-dev-stack/internal-api-secret) PYTHONPATH=src \
  uv run python tests/fixtures/mr/r_reference/build_inputs.py; cd -
python3 -I make_synthetic.py
```

Outputs:

```
docker build --memory=6g -f Dockerfile.r-reference -t mr-r-reference .
docker run --rm --network none --memory=4g -u "$(id -u):$(id -g)" -v "$PWD:/out" mr-r-reference Rscript /out/reference.R
sha256sum *.csv sessionInfo.txt > SHA256SUMS
```
