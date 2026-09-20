# linemodels reference outputs

Inputs and outputs of Matti Pirinen's `linemodels` R package (github.com/mjpirinen/linemodels,
commit d66e07b, version 0.5.0), which `tests/test_sdk_linemodels.py` holds
`genetics_mcp_server.sdk.linemodels` to.

- `linemodels_ex1.csv`, `linemodels_ex2.csv` — the package's bundled data sets, written out
  with `write.csv` at full precision.
- `covid_hgi_v6_B2_C2_common.tsv` — the COVID-19 HGI release 6 table the package's Example 2
  and the paper's worked example use, from github.com/mjpirinen/covid19-hgi_subtypes.
- `*_line_models.csv` — `line.models` outputs (deterministic).
- `covid_proportions_*.csv` — `line.models.with.proportions` with `set.seed(91)`, 10,000
  sweeps after 50 (stochastic; the tests compare within Monte Carlo noise).
- `ex1.R`, `ex2_ex4.R` — the scripts that produced them; `ex2_ex4.log` is the second script's
  console output, which holds the EM optima the tests quote.

To regenerate, clone the package beside `Dockerfile.r-reference` as `linemodels/`, build the
image and run each script with the fixture directory mounted at `/out`:

```
git clone https://github.com/mjpirinen/linemodels
docker build -f Dockerfile.r-reference -t linemodels-ref .
docker run --rm -v "$PWD:/out" linemodels-ref Rscript /out/ex1.R
docker run --rm -v "$PWD:/out" linemodels-ref Rscript /out/ex2_ex4.R > ex2_ex4.log
```
