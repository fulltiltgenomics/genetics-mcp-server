"""The MR estimators reproduce MendelianRandomization 0.10.0 and TwoSampleMR 0.7.12.

The yardstick is tests/fixtures/mr/r_reference (its README gives the image, the inputs and the
tolerances): ten cis-pQTL -> IBD tables and one synthetic table, every package number in two
long CSVs. Closed-form numbers are held to floating point. Bootstrap SEs cannot be: R's RNG
is not reproduced in numpy, so a seeded numpy bootstrap is held to a Monte Carlo band. The
mode SEs additionally depend on how mr_mbe breaks ties between bootstrap modes, so their band
is wider.
"""

import math
from functools import cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from genetics_mcp_server.sdk import _mr_estimators as mr
from genetics_mcp_server.sdk.errors import GeneticsUsageError

REF = Path(__file__).parent / "fixtures" / "mr" / "r_reference"

# floating-point agreement; the worst observed is ~4e-14 on Q of a 7-instrument table.
# ABS is only for references that are exactly 0 (I2_from_Q rows); every other quantity,
# p-values included, is held to REL alone because small p-values have no absolute floor
REL, ABS = 1e-9, 1e-12
EXACT_ZERO_QUANTITIES = {"I2_from_Q", "I.sq"}
# the package reports no I^2_GX for a correlated Egger and no I^2 for a correlated IVW; this
# port computes neither there
NOT_REPORTED = {("mr_egger", "I.sq"), ("mr_ivw", "I2_from_Q")}


def abs_for(quantity: str) -> float:
    return ABS if quantity in EXACT_ZERO_QUANTITIES else 0.0
# two independent 10,000-draw bootstraps of the same SD differ by ~1%; worst observed 1.5%
MEDIAN_SE_BAND = 0.05
# the README calls mode SEs version-dependent (mr_mbe keeps the first of tied modes);
# worst observed 2.9%
MODE_SE_BAND = 0.15

ESTIMATES = pl.read_csv(REF / "out_estimates.csv", infer_schema_length=0)
PER_VARIANT = pl.read_csv(REF / "out_per_variant.csv", infer_schema_length=0)
CIS = pl.read_csv(REF / "input_cis_ibd.csv").rename(
    {"ncase_y": "n_cases", "ncontrol_y": "n_controls"}
)
TABLES = {p: CIS.filter(pl.col("protein") == p) for p in sorted(CIS["protein"].unique())}
TABLES["synthetic"] = pl.read_csv(REF / "input_synthetic.csv")
PROTEINS = [t for t in TABLES if t != "synthetic"]


def ld(protein: str) -> pl.DataFrame:
    return pl.read_csv(REF / f"ld_{protein}.csv")


def close(got: float, want: float, rel: float = REL, abs_: float = ABS) -> bool:
    return math.isclose(got, want, rel_tol=rel, abs_tol=abs_)


def groups(frame: pl.DataFrame, package: str, prefix: str = "") -> list[tuple[str, str, str]]:
    rows = frame.filter(
        (pl.col("package") == package) & pl.col("method").str.starts_with(prefix)
    ).select("table", "method", "options").unique(maintain_order=True)
    return [tuple(r) for r in rows.iter_rows()]


def reference(table: str, method: str, options: str, frame=ESTIMATES) -> dict[str, str]:
    rows = frame.filter(
        (pl.col("table") == table) & (pl.col("method") == method) & (pl.col("options") == options)
    )
    if "variant" in rows.columns:
        return {(v, q): x for v, q, x in rows.select("variant", "quantity", "value").iter_rows()}
    return dict(rows.select("quantity", "value").iter_rows())


def gid(case) -> str:
    return "-".join(str(c) for c in case).replace(" ", "_")


# --------------------------------------------------------------------------- MendelianRandomization

IVW_KEYS = {"Estimate": "estimate", "StdError": "se", "CILower": "ci_low", "CIUpper": "ci_high",
            "Pvalue": "p", "Q": "Q", "Q_pval": "Q_pval", "RSE": "RSE", "I2_from_Q": "I2",
            "SNPs": "n_instruments"}
EGGER_KEYS = {"Estimate": "estimate", "StdError.Est": "se", "CILower.Est": "ci_low",
              "CIUpper.Est": "ci_high", "Pvalue.Est": "p", "Intercept": "intercept",
              "StdError.Int": "intercept_se", "CILower.Int": "intercept_ci_low",
              "CIUpper.Int": "intercept_ci_high", "Pvalue.Int": "intercept_p", "Q": "Q",
              "Q_pval": "Q_pval", "I.sq": "I2_GX", "RSE": "RSE", "SNPs": "n_instruments"}
SE_KEYS = {"Estimate": "estimate", "StdError": "se", "SNPs": "n_instruments"}


def parse_options(options: str) -> dict[str, str]:
    return dict(kv.split("=") for kv in options.split(";") if "=" in kv)


@cache
def run_mr_package(table: str, method: str, options: str):
    o = parse_options(options)
    data = TABLES[table]
    correl = ld(table) if o.get("correl") == "TRUE" else None
    if method == "mr_ivw":
        return mr.ivw(data, model=o["model"], correl=correl), IVW_KEYS
    if method == "mr_egger":
        return mr.egger(data, correl=correl), EGGER_KEYS
    if method == "mr_median":
        fn = mr.weighted_median if o["weighting"] == "weighted" else mr.simple_median
        return fn(data, seed=int(o["seed"]), iterations=int(o["iterations"])), SE_KEYS
    if method == "mr_mbe":
        fn = mr.weighted_mode if o["weighting"] == "weighted" else mr.simple_mode
        return fn(data, phi=float(o["phi"]), seed=int(o["seed"]),
                  iterations=int(o["iterations"])), SE_KEYS
    raise AssertionError(method)


MR_CASES = groups(ESTIMATES, "MendelianRandomization")
MR_ERRORS = [c for c in MR_CASES if "error" in reference(*c)]


@pytest.mark.parametrize("case", [c for c in MR_CASES if c not in MR_ERRORS], ids=gid)
def test_mendelianrandomization_numbers(case):
    table, method, options = case
    ref = reference(*case)
    res, keys = run_mr_package(*case)
    if "Model" in ref:
        model = res.get("model", res["options"].get("model"))
        assert model == ref["Model"]
    for quantity, key in keys.items():
        if "correl=TRUE" in options and (method, quantity) in NOT_REPORTED:
            continue
        assert quantity in ref, f"reference lacks {quantity}"
        want = ref[quantity]
        assert want != "NA", f"reference has NA for {quantity}"
        got = res[key]
        if quantity != "StdError":
            if method == "mr_mbe" and table == "RSPO3" and "unweighted" in options:
                continue  # checked in test_two_point_simple_mode_is_a_grid_tie
            assert close(got, float(want), abs_=abs_for(quantity)), (quantity, got, want)
        else:
            band = MEDIAN_SE_BAND if method == "mr_median" else MODE_SE_BAND
            assert abs(got / float(want) - 1) < band, (quantity, got, want)


@pytest.mark.parametrize("case", MR_ERRORS, ids=gid)
def test_refuses_where_the_package_refuses(case):
    with pytest.raises(GeneticsUsageError, match="at least 3 instruments"):
        run_mr_package(*case)


def test_two_point_simple_mode_is_a_grid_tie():
    """Two equally weighted ratios peak half-way between two density grid points; R's pick
    is FP noise, so the port is held to one grid step (README: RSPO3, K = 2)."""
    opt = "weighting=unweighted;stderror=delta;phi=1;iterations=10000;seed=314159265"
    want = float(reference("RSPO3", "mr_mbe", opt)["Estimate"])
    ratio = (TABLES["RSPO3"]["beta_y"] / TABLES["RSPO3"]["beta_x"]).to_numpy()
    bw = 0.9 * min(np.std(ratio, ddof=1), mr._mad(ratio[None, :])[0]) / 2 ** 0.2
    step = (np.ptp(ratio) + 6 * bw) / 511
    got = mr.simple_mode(TABLES["RSPO3"], iterations=200)["estimate"]
    assert abs(got - want) <= step * (1 + 1e-9)
    assert abs(got - ratio.mean()) == pytest.approx(step / 2, rel=1e-6)


def test_default_ivw_model_switches_at_four_instruments():
    k = {t: f.height for t, f in TABLES.items()}
    for table, frame in TABLES.items():
        assert mr.ivw(frame)["model"] == ("fixed" if k[table] <= 3 else "random")


# --------------------------------------------------------------------------- correlated IVW


def test_correlated_ivw_drops_the_instrument_the_panel_lacks():
    res = mr.ivw(TABLES["CXCL10"], model="fixed", correl=ld("CXCL10"))
    assert res["options"]["dropped_no_ld"] == ["4:76027659:C:A"]
    assert res["n_instruments"] == 2
    ref = reference("CXCL10", "mr_ivw", "correl=TRUE;model=fixed")
    assert close(res["estimate"], float(ref["Estimate"]))
    with pytest.raises(GeneticsUsageError):
        mr.egger(TABLES["CXCL10"], correl=ld("CXCL10"))


@pytest.mark.parametrize("protein", ["ANGPTL3", "CD274", "FAS", "IL12B", "IL18"])
def test_correlation_moves_the_estimate_where_ld_is_nonzero(protein):
    plain = mr.ivw(TABLES[protein], model="fixed")
    correl = mr.ivw(TABLES[protein], model="fixed", correl=ld(protein))
    assert correl["options"]["repaired"] is False
    assert abs(correl["estimate"] - plain["estimate"]) > 1e-6


def test_numpy_correlation_in_frame_order_matches_the_frame_form():
    frame = TABLES["IL12B"]
    m = ld("IL12B")
    order = [m["variant"].to_list().index(v) for v in frame["variant"]]
    arr = m.drop("variant").to_numpy()[np.ix_(order, order)]
    assert mr.ivw(frame, correl=arr)["estimate"] == pytest.approx(
        mr.ivw(frame, correl=m)["estimate"], rel=1e-14
    )


def test_indefinite_ld_is_repaired_and_says_so():
    frame = TABLES["CD274"]
    r = np.array([[1, 0.9, 0.9], [0.9, 1, -0.9], [0.9, -0.9, 1]])
    assert np.linalg.eigvalsh(r).min() < 0
    res = mr.ivw(frame, model="fixed", correl=r)
    assert res["options"]["repaired"] is True
    assert res["options"]["min_eigenvalue"] < 0
    assert math.isfinite(res["estimate"]) and res["se"] > 0
    fixed, info = mr._repair(r)
    assert np.allclose(np.diag(fixed), 1)
    assert np.linalg.eigvalsh(fixed).min() > 0
    assert info["eigenvalue_floor"] == mr._PD_FLOOR


# --------------------------------------------------------------------------- TwoSampleMR

# TwoSampleMR's mr() rows reproduced by a function here, point estimates only; mr() writes
# only its IVW row below 3 instruments
TSMR_POINT = {
    "mr:Inverse variance weighted": lambda f: mr.ivw(f, model="random"),
    "mr:MR Egger": mr.egger,
    "mr:Weighted median": lambda f: mr.weighted_median(f, weights="second_order", iterations=10),
    "mr:Weighted mode": lambda f: mr.weighted_mode(f, iterations=10),
    "mr:Simple mode": lambda f: mr.simple_mode(f, iterations=10),
}


@pytest.mark.parametrize(
    "case", [c for c in groups(ESTIMATES, "TwoSampleMR", "mr:") if c[1] in TSMR_POINT], ids=gid
)
def test_twosamplemr_point_estimates(case):
    table, method, _ = case
    ref = reference(*case)
    res = TSMR_POINT[method](TABLES[table])
    assert close(res["estimate"], float(ref["b"]))
    if method in ("mr:Inverse variance weighted", "mr:MR Egger"):
        assert close(res["se"], float(ref["se"]))


@pytest.mark.parametrize("table", list(TABLES))
def test_heterogeneity_and_pleiotropy(table):
    het = mr.heterogeneity(TABLES[table])
    ref = reference(table, "mr_heterogeneity:Inverse variance weighted", "defaults")
    for q in ("Q", "Q_df", "Q_pval"):
        assert close(het[q], float(ref[q]), abs_=0), q
    if TABLES[table].height < 3:
        return
    eg = mr.egger(TABLES[table])
    ref = reference(table, "mr_heterogeneity:MR Egger", "defaults")
    for q in ("Q", "Q_df", "Q_pval"):
        assert close(eg[q], float(ref[q]), abs_=0), q
    ref = reference(table, "mr_pleiotropy_test", "defaults")
    assert close(eg["intercept"], float(ref["egger_intercept"]))
    assert close(eg["intercept_se"], float(ref["se"]))


def per_variant_check(got: pl.DataFrame, ref: dict, keys: dict[str, str]) -> None:
    assert set(got["variant"]) == {v for v, _ in ref}
    for row in got.iter_rows(named=True):
        for quantity, key in keys.items():
            want = ref[(row["variant"], quantity)]
            if want == "NA":
                assert math.isnan(row[key])
            else:
                assert close(row[key], float(want), abs_=0), (row["variant"], quantity)


@pytest.mark.parametrize("table", list(TABLES))
def test_wald_single_snp_and_leave_one_out(table):
    frame = TABLES[table]
    per_variant_check(mr.wald(frame), reference(table, "wald", "first_order", PER_VARIANT),
                      {"estimate": "estimate", "se": "se"})
    keys = {"b": "estimate", "se": "se", "p": "p"}
    per_variant_check(mr.single_snp(frame),
                      reference(table, "mr_singlesnp", "defaults", PER_VARIANT), keys)
    loo = reference(table, "mr_leaveoneout", "defaults", PER_VARIANT)
    if frame.height < 3:
        assert {v for v, _ in loo} == {"All"}
        with pytest.raises(GeneticsUsageError):
            mr.leave_one_out(frame)
    else:
        per_variant_check(mr.leave_one_out(frame), loo, keys)


STEIGER = [(t, o) for t, _, o in groups(ESTIMATES, "TwoSampleMR", "mr_steiger")]


@pytest.mark.parametrize("case", STEIGER, ids=gid)
def test_steiger_matches_mr_steiger(case):
    table, options = case
    o = parse_options(options)
    prevalence = float(o["prevalence"]) if "prevalence" in o else None
    res = mr.steiger(TABLES[table], prevalence=prevalence)
    ref = reference(table, "mr_steiger", options)
    assert close(res["r2_exposure"], float(ref["r2_exp"]))
    assert close(res["r2_outcome"], float(ref["r2_out"]))
    assert str(res["correct_causal_direction"]).upper() == ref["correct_causal_direction"]
    # an r.test p below the double range is 0 in R too
    assert close(res["steiger_pval"], float(ref["steiger_test"]), abs_=0)
    pv = reference(table, "get_r_from_bsen:exposure", options, PER_VARIANT)
    out_method = "get_r_from_lor:outcome" if prevalence else "get_r_from_bsen:outcome"
    pv_out = reference(table, out_method, options, PER_VARIANT)
    for row in res["per_variant"].iter_rows(named=True):
        assert close(row["r2_exposure"], float(pv[(row["variant"], "r")]) ** 2)
        assert close(row["r2_outcome"], float(pv_out[(row["variant"], "r")]) ** 2)
    if prevalence:
        # steiger_filtering's outcome r2 takes the same get_r_from_lor route
        sf = reference(table, "steiger_filtering", options, PER_VARIANT)
        for row in res["per_variant"].iter_rows(named=True):
            assert close(row["r2_outcome"], float(sf[(row["variant"], "rsq.outcome")]))


def test_steiger_on_a_binary_outcome_requires_prevalence():
    with pytest.raises(GeneticsUsageError, match="prevalence.*run\\(\\) supplies"):
        mr.steiger(TABLES["CD274"])
    opts = mr.steiger(TABLES["CD274"], prevalence=0.005)["options"]
    assert opts["prevalence"] == 0.005 and opts["prevalence_source"] == "given"


def test_binary_outcome_without_allele_frequency_is_a_usage_error():
    frame = TABLES["CD274"]
    for col in ("af_y", "af_x"):
        if col in frame.columns:
            frame = frame.drop(col)
    with pytest.raises(GeneticsUsageError, match="af_y"):
        mr.steiger(frame, prevalence=0.1)


# --------------------------------------------------------------------------- contract


def test_result_keys_checks_reads():
    frame = TABLES["synthetic"]
    for res in (mr.ivw(frame), mr.heterogeneity(frame), mr.egger(frame)):
        assert {"Q", "Q_df", "Q_pval"} <= res.keys()
    assert {"correct_causal_direction", "steiger_pval"} <= mr.steiger(frame).keys()
    for fn in (mr.ivw, mr.egger, mr.weighted_median, mr.simple_median, mr.weighted_mode,
               mr.simple_mode):
        res = fn(frame) if fn in (mr.ivw, mr.egger) else fn(frame, iterations=50)
        assert {"method", "estimate", "se", "p", "ci_low", "ci_high", "n_instruments",
                "options"} <= res.keys()


def test_accepts_harmonised_and_single_instrument():
    frame = TABLES["CD274"]
    h = SimpleNamespace(frame=frame, units={"exposure": "SD (quantitative)",
                                            "outcome": "log OR (binary)"})
    assert mr.ivw(h)["estimate"] == mr.ivw(frame)["estimate"]
    assert mr.steiger(h, prevalence=0.1)["steiger_pval"] == mr.steiger(frame, prevalence=0.1)[
        "steiger_pval"]
    one = mr.ivw(frame.head(1))
    w = mr.wald(frame.head(1)).row(0, named=True)
    assert one["method"] == "wald" and one["estimate"] == w["estimate"] and one["se"] == w["se"]


def test_bootstrap_is_repeatable_by_seed():
    frame = TABLES["synthetic"]
    a = mr.weighted_median(frame, seed=7, iterations=500)
    assert a == mr.weighted_median(frame, seed=7, iterations=500)
    assert a["se"] != mr.weighted_median(frame, seed=8, iterations=500)["se"]


def test_bad_inputs_are_usage_errors():
    frame = TABLES["synthetic"]
    with pytest.raises(GeneticsUsageError, match="lacks"):
        mr.ivw(frame.drop("se_y"))
    with pytest.raises(GeneticsUsageError, match="model"):
        mr.ivw(frame, model="mixed")
    with pytest.raises(GeneticsUsageError, match="beta_x = 0"):
        mr.ivw(frame.with_columns(beta_x=pl.lit(0.0)))
    with pytest.raises(GeneticsUsageError, match="must be positive"):
        mr.ivw(frame.with_columns(se_x=pl.lit(0.0)))
    for col in ("beta_y", "se_y"):
        with pytest.raises(GeneticsUsageError, match="missing or infinite"):
            mr.ivw(frame.with_columns(pl.when(pl.int_range(pl.len()) == 0)
                                      .then(math.inf).otherwise(pl.col(col)).alias(col)))
    with pytest.raises(GeneticsUsageError, match="expected"):
        mr.ivw({"beta_x": [1]})


def test_i2_gx_is_nan_where_q_is_not_finite():
    one = np.ones(3)
    with np.errstate(invalid="ignore"):
        assert math.isnan(mr._i2_gx(np.array([1.0, 2.0, np.inf]), one, one))
