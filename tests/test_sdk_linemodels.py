"""The line-models port reproduces Pirinen's R package on the package's own examples.

The fixtures under tests/fixtures/linemodels/ are the R package's two bundled data sets
(`linemodels.ex1`, `linemodels.ex2`) and the COVID-19 HGI release 6 table its paper analyses,
with the outputs of linemodels 0.5.0 written by the scripts beside them (ex1.R, ex2_ex4.R) in
a container with R 4.4.3. Fixed-parameter classification is deterministic and is held to
floating-point agreement. The Gibbs sampler and the EM optimiser are held to the tolerances
their own randomness and optimiser differences leave: the sampler to Monte Carlo noise and
the paper's published counts, the optimiser to the optimum R found from the same start.
"""

import math
import os
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from genetics_mcp_server.sdk import linemodels as lm
from genetics_mcp_server.sdk.errors import GeneticsUsageError

FIXTURES = Path(__file__).parent / "fixtures" / "linemodels"


def fixture(name: str) -> pl.DataFrame:
    return pl.read_csv(FIXTURES / name)


@pytest.fixture(scope="module")
def ex1():
    return fixture("linemodels_ex1.csv")


@pytest.fixture(scope="module")
def covid():
    return fixture("covid_line_models.csv")


SLOPE_BOTH = math.tan(math.atan(0.2) + (math.atan(1) - math.atan(0.2)) / 2)
COVID_MODELS = dict(
    slopes=[0.2, 1, SLOPE_BOTH],
    scales=0.15,
    cors=0.999,
    names=["SEVER.", "SUSCEP.", "BOTH"],
    r_lkhood=0.4539485,
)


# --------------------------------------------------------------------------- classify


def test_classify_matches_r_on_the_package_example():
    ref = fixture("ex1_line_models.csv")
    res = lm.classify(
        ref.select("beta1", "beta2"), ref.select("se1", "se2"),
        slopes=[0, 0.5, 1], scales=0.2, cors=0.995, names=["M0", "M.5", "M1"],
    )
    got = res["groups"].select("M0", "M.5", "M1").to_numpy()
    assert np.abs(got - ref.select("M0", "M.5", "M1").to_numpy()).max() < 1e-12
    assert res["groups"]["model"][0] == "M0"
    assert res["scale_source"] == "given"
    assert res["models"]["prior"].to_list() == pytest.approx([1 / 3] * 3)


def test_classify_matches_r_with_correlated_estimators(covid):
    """The paper's COVID example: r_lkhood = 0.454 exercises the off-diagonal path."""
    res = lm.classify(
        covid.select("B2_beta", "C2_beta"), covid.select("B2_sebeta", "C2_sebeta"), **COVID_MODELS
    )
    got = res["groups"].select("SEVER.", "SUSCEP.", "BOTH").to_numpy()
    assert np.abs(got - covid.select("SEVER.", "SUSCEP.", "BOTH").to_numpy()).max() < 1e-12


def test_classify_matches_r_on_the_annotation_example():
    data = fixture("linemodels_ex2.csv")
    ref = fixture("ex2_line_models.csv")
    res = lm.classify(
        data.select("beta1", "beta2"), data.select("se1", "se2"),
        slopes=[0, 3], scales=0.1, cors=0.999,
    )
    got = res["groups"].select("M1", "M2").to_numpy()
    assert np.abs(got - ref.to_numpy()).max() < 1e-12
    # the outlier with a large SE sits between the lines and is undetermined
    assert 0.4 < res["groups"]["max_prob"][100] < 0.6


def test_classify_accepts_arrays_and_a_full_correlation_matrix(covid):
    x = covid.select("B2_beta", "C2_beta").to_numpy()
    se = covid.select("B2_sebeta", "C2_sebeta").to_numpy()
    r = 0.4539485
    a = lm.classify(x, se, **{**COVID_MODELS, "r_lkhood": [[1, r], [r, 1]]})
    b = lm.classify(x, se, **{**COVID_MODELS, "r_lkhood": [r]})
    assert a["groups"].equals(b["groups"])
    assert a["models"]["model"].to_list() == ["SEVER.", "SUSCEP.", "BOTH"]


def test_default_models_and_the_derived_scale(ex1):
    x = ex1.select("beta1", "beta2")
    res = lm.classify(x, ex1.select("se1", "se2"))
    models = res["models"]
    assert models["model"].to_list() == ["beta1_only", "shared", "beta2_only"]
    assert models["slope"].to_list() == [0.0, 1.0, math.inf]
    assert models["cor"].to_list() == [0.995] * 3
    expected_scale = float(np.quantile(np.abs(x.to_numpy()), 0.95, axis=0).max()) / 2
    assert models["scale"].to_list() == pytest.approx([expected_scale] * 3)
    assert res["scale_source"] == "derived"
    assert res["groups"].columns == ["beta1_only", "shared", "beta2_only", "model", "max_prob"]
    assert res["groups"].height == ex1.height
    # the first variants of ex1 come from the slope-0 model
    assert res["groups"]["model"][0] == "beta1_only"


def test_maf_scaling_is_the_sqrt_heritability_transform(ex1):
    factor = np.sqrt(2 * ex1["maf"].to_numpy() * (1 - ex1["maf"].to_numpy()))[:, None]
    x = ex1.select("beta1", "beta2").to_numpy()
    se = ex1.select("se1", "se2").to_numpy()
    scaled = lm.classify(x, se, maf=ex1["maf"], slopes=[0, 1], scales=0.15)
    manual = lm.classify(x * factor, se * factor, slopes=[0, 1], scales=0.15)
    assert scaled["scaled"] is True
    assert scaled["groups"].equals(manual["groups"])


def test_the_null_model_is_scale_zero(ex1):
    x = ex1.select("beta1", "beta2").to_numpy()
    se = ex1.select("se1", "se2").to_numpy()
    # a variant with no effect beyond its noise, appended to the simulated ones
    x = np.vstack([x, [[0.002, -0.001]]])
    se = np.vstack([se, [[0.01, 0.01]]])
    res = lm.classify(
        x, se, slopes=[0, 0, 1], scales=[0.0, 0.2, 0.2], names=["null", "x_only", "shared"]
    )
    models = res["groups"]["model"].to_list()
    assert models[-1] == "null"
    assert res["groups"]["max_prob"][-1] > 0.9
    # ex1's 20 slope-0 and 40 slope-1 variants have effects far above their SEs
    assert models.count("x_only") >= 18 and models.count("shared") >= 35


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(slopes=[0, 1], cors=[1.5, 0.9]), "cor must lie in"),
        (dict(slopes=[0, 1], scales=[-1, 0.2]), "non-negative"),
        (dict(slopes=[0, 1], names=["a"]), "one entry per model"),
        (dict(slopes=[0, 1], names=["a", "a"]), "unique"),
        (dict(slopes=[0, 1], priors=[0, 0]), "priors"),
        (dict(slopes=[0, 1], r_lkhood=[0.1, 0.2]), "upper-triangle"),
        (dict(slopes=[0, 1], r_lkhood=1.5), "[-1, 1]"),
        (dict(slopes=[[0, 1], [1, 1]]), "column(s)"),
    ],
)
def test_bad_model_specifications_are_usage_errors(ex1, kwargs, message):
    with pytest.raises(GeneticsUsageError) as err:
        lm.classify(ex1.select("beta1", "beta2"), ex1.select("se1", "se2"), **kwargs)
    assert message in str(err.value)


def test_bad_data_is_a_usage_error(ex1):
    x = ex1.select("beta1", "beta2")
    with pytest.raises(GeneticsUsageError, match="shape"):
        lm.classify(x, ex1.select("se1", "se2").head(50))
    with pytest.raises(GeneticsUsageError, match="positive"):
        lm.classify(x, ex1.select("se1", "se2").with_columns(pl.lit(0.0).alias("se1")))
    with pytest.raises(GeneticsUsageError, match="null, NaN"):
        lm.classify(x.with_columns(pl.lit(None, dtype=pl.Float64).alias("beta2")), ex1.select("se1", "se2"))
    with pytest.raises(GeneticsUsageError, match="at least two columns"):
        lm.classify(ex1.select("beta1"), ex1.select("se1"))
    with pytest.raises(GeneticsUsageError, match="slopes is required"):
        lm.classify(ex1.select("beta1", "beta2", "beta3"), ex1.select("se1", "se2", "se3"))
    with pytest.raises(GeneticsUsageError, match="maf"):
        lm.classify(x, ex1.select("se1", "se2"), maf=[0.1, 0.2])


# --------------------------------------------------------------------------- proportions


def test_proportions_reproduces_the_paper(covid):
    """Pirinen 2023: 64% (42–82) severity, 25% (9–45) susceptibility, 11% (0–31) both; at
    0.95, 12 severity-only, 5 susceptibility-only, 0 both. R's own run of the same
    settings (seed 91, 10,000 sweeps) is in covid_proportions_params.csv."""
    res = lm.proportions(
        covid.select("B2_beta", "C2_beta"), covid.select("B2_sebeta", "C2_sebeta"),
        **COVID_MODELS, n_iter=10000, n_burnin=50, seed=91,
    )
    params = res["params"]
    assert params["model"].to_list() == ["SEVER.", "SUSCEP.", "BOTH"]
    r_params = fixture("covid_proportions_params.csv")
    assert params["mean"].to_list() == pytest.approx(r_params["mean"].to_list(), abs=0.02)
    assert params["low95"].to_list() == pytest.approx(r_params["95%low"].to_list(), abs=0.03)
    assert params["up95"].to_list() == pytest.approx(r_params["95%up"].to_list(), abs=0.03)
    probs = res["groups"].select("SEVER.", "SUSCEP.", "BOTH").to_numpy()
    assert (probs > 0.95).sum(axis=0).tolist() == [12, 5, 0]
    r_groups = fixture("covid_proportions_groups.csv").select("SEVER.", "SUSCEP.", "BOTH").to_numpy()
    assert np.abs(probs - r_groups).max() < 0.05
    assert res["n_iter"] == 10000 and res["n_burnin"] == 50


def test_proportions_is_repeatable_by_seed_and_matches_r_on_ex1(ex1):
    x, se = ex1.select("beta1", "beta2"), ex1.select("se1", "se2")
    kwargs = dict(slopes=[0, 0.5, 1], scales=0.2, cors=0.995, n_iter=2000, n_burnin=200)
    a = lm.proportions(x, se, seed=1, **kwargs)
    b = lm.proportions(x, se, seed=1, **kwargs)
    assert a["params"].equals(b["params"]) and a["groups"].equals(b["groups"])
    # R with set.seed(1): 0.1888 0.3711 0.4401; the data were simulated as 20/40/40
    assert a["params"]["mean"].to_list() == pytest.approx([0.189, 0.371, 0.440], abs=0.02)
    assert a["params"].columns == ["model", "mean", "low95", "up95", "sd"]
    assert a["models"].columns == ["model", "scale", "slope", "cor"]


def test_proportions_argument_checks(ex1):
    x, se = ex1.select("beta1", "beta2"), ex1.select("se1", "se2")
    with pytest.raises(GeneticsUsageError, match="n_iter"):
        lm.proportions(x, se, n_iter=0)
    with pytest.raises(GeneticsUsageError, match="n_burnin"):
        lm.proportions(x, se, n_burnin=-1)
    with pytest.raises(GeneticsUsageError, match="diri_prior"):
        lm.proportions(x, se, diri_prior=[0, 1, 1])


# --------------------------------------------------------------------------- optimize


def test_optimize_finds_the_optimum_r_found(ex1):
    """linemodels_examples.R, Example 1: from a perturbed start the middle model reaches
    scale 0.23198, slope 0.50755, cor 0.99775 (R's BFGS)."""
    res = lm.optimize(
        ex1.select("beta1", "beta2"), ex1.select("se1", "se2"),
        slopes=[0, 0.2, 1], scales=[0.2, 0.05, 0.2], cors=[0.995, 0.1, 0.995],
        names=["M0", "M.5", "M1"], fit=[[False] * 3, [True] * 3, [False] * 3], tol_loglik=1e-2,
    )
    models = res["models"]
    assert models["scale"].to_list() == pytest.approx([0.2, 0.23198327, 0.2], abs=1e-3)
    assert models["slope"].to_list() == pytest.approx([0.0, 0.50755371, 1.0], abs=1e-3)
    assert models["cor"].to_list() == pytest.approx([0.995, 0.99774733, 0.995], abs=1e-3)
    assert models["weight"].to_list() == pytest.approx([0.197, 0.347, 0.456], abs=5e-3)
    assert res["converged"] and res["criterion"] == "loglik"
    assert res["groups"].height == ex1.height


def test_optimize_single_parameter_uses_the_bounded_path(ex1):
    """The vignette's one-slope fit: slope 0.5098276, log-likelihood 171.47986, weights
    0.1893305 / 0.3718141 / 0.4388554."""
    res = lm.optimize(
        ex1.select("beta1", "beta2"), ex1.select("se1", "se2"),
        slopes=[0, 0.3, 1], scales=0.2, cors=0.995, fit={"slopes": [False, True, False]},
    )
    assert res["models"]["slope"][1] == pytest.approx(0.5098276, abs=1e-3)
    assert res["loglik"] == pytest.approx(171.47986, abs=1e-3)
    assert res["weights"] == pytest.approx(
        {"M1": 0.1893305, "M2": 0.3718141, "M3": 0.4388554}, abs=1e-3
    )


def test_optimize_scales_on_the_heritability_scale_with_constant_se(ex1):
    """linemodels_examples.R: after maf scaling the scales come out 0.1515844, 0.1418315,
    0.1350342 with assume.constant.SE = TRUE."""
    res = lm.optimize(
        ex1.select("beta1", "beta2"), ex1.select("se1", "se2"), maf=ex1["maf"],
        slopes=[0, 0.2, 1], scales=1.0, cors=[0.995, 0.1, 0.995],
        fit=[[True, False, False], [True, True, True], [True, False, False]],
        tol_loglik=1e-2, constant_se=True,
    )
    assert res["models"]["scale"].to_list() == pytest.approx([0.1515844, 0.1418315, 0.1350342], abs=1e-3)
    assert res["models"]["slope"][1] == pytest.approx(0.50838806, abs=2e-3)
    assert res["scaled"] is True


def test_optimize_three_dimensions_with_one_shared_scale(ex1):
    """Example 3: the slopes recover the simulation's (0,0), (1,1), (0.5,0.2); R's shared
    scale is 0.13801688."""
    res = lm.optimize(
        ex1.select("beta1", "beta2", "beta3"), ex1.select("se1", "se2", "se3"), maf=ex1["maf"],
        slopes=[[0, 0], [1, 1], [0.5, 0.5]], scales=0.15, cors=0.995, r_lkhood=[0, 0, 0],
        fit={"scales": True, "slopes": True}, force_same_scales=True,
        tol_loglik=1e-2, constant_se=True,
    )
    models = res["models"]
    assert models.columns == ["model", "scale", "slope_beta2", "slope_beta3", "cor", "weight"]
    assert len(set(models["scale"].to_list())) == 1
    assert models["scale"][0] == pytest.approx(0.13801688, abs=2e-3)
    assert models["slope_beta2"].to_list() == pytest.approx([0.0156, 0.9954, 0.5089], abs=5e-3)
    assert models["slope_beta3"].to_list() == pytest.approx([-0.0031, 1.0142, 0.1864], abs=5e-3)
    assert models["weight"].to_list() == pytest.approx([0.192, 0.423, 0.385], abs=5e-3)


def test_optimize_proportions_only_and_argument_checks(ex1):
    x, se = ex1.select("beta1", "beta2"), ex1.select("se1", "se2")
    # R with par.include = NULL: weights 0.22084734 0.35970368 0.41944898
    fixed = lm.optimize(x, se, slopes=[0, 0.5, 1], scales=0.2, fit="proportions")
    assert fixed["models"]["slope"].to_list() == [0.0, 0.5, 1.0]
    assert fixed["weights"] == pytest.approx(
        {"M1": 0.22084734, "M2": 0.35970368, "M3": 0.41944898}, abs=1e-4
    )
    with pytest.raises(GeneticsUsageError, match="start finite"):
        lm.optimize(x, se, slopes=[0, math.inf], fit="slopes")
    with pytest.raises(GeneticsUsageError, match="unknown"):
        lm.optimize(x, se, slopes=[0, 1], fit="slope")
    with pytest.raises(GeneticsUsageError, match="method"):
        lm.optimize(x, se, slopes=[0, 1], method="CG")
    with pytest.raises(GeneticsUsageError, match="tol_loglik"):
        lm.optimize(x, se, slopes=[0, 1], tol_loglik=0)


# --------------------------------------------------------------------------- helpers


def test_estimator_correlation_is_bhattacharjee():
    """beta.cor.case.control(1000, 1000, 1000, 2000, r1r2 = 1000) in R."""
    assert lm.estimator_correlation(1000, 1000, 1000, 2000, shared_controls=1000) == pytest.approx(
        0.2886751345948129
    )
    assert lm.estimator_correlation(1000, 1000, 1000, 2000) == 0.0
    with pytest.raises(GeneticsUsageError):
        lm.estimator_correlation(0, 1, 1, 1)


def test_prior_covariance_matches_r():
    """prior.V values from the R package."""
    v = lm._prior_covariance(0.15, np.array([0.2]), 0.999)
    assert v == pytest.approx(np.array([[0.0225, 0.0044976589], [0.0044976589, 0.0009112374]]), abs=1e-9)
    v = lm._prior_covariance(0.2, np.array([0.5, 0.2]), 0.995)
    assert v[0] == pytest.approx([0.04, 0.0199568771, 0.0079827508], abs=1e-9)
    v = lm._prior_covariance(0.1, np.array([math.inf]), 0.99)
    assert v == pytest.approx(np.array([[5.0251256281e-05, 0.0], [0.0, 0.01]]), abs=1e-12)
    assert (lm._prior_covariance(0.0, np.array([1.0]), 0.5) == 0).all()


# --------------------------------------------------------------------------- surfaces


def test_the_sdk_resolves_the_module_lazily():
    import genetics_mcp_server.sdk as sdk

    assert "linemodels" in dir(sdk)
    assert sdk.linemodels is lm
    assert sdk.linemodels.classify is lm.classify


def test_list_capabilities_renders_the_module():
    from genetics_mcp_server.tools.executor import _sdk_capabilities

    index = _sdk_capabilities(None)
    entry = next(m for m in index["modules"] if m["module"] == "linemodels")
    assert entry["names"] == list(lm.__all__)
    detail = _sdk_capabilities("linemodels")
    assert "genetics.linemodels.classify" in detail["usage"]
    assert "def classify(" in detail["signatures"]
    assert "r_lkhood" in detail["signatures"]


def test_the_figure_draws_and_counts(ex1, tmp_path, monkeypatch):
    from genetics_mcp_server.sdk import plots

    monkeypatch.setenv("SANDBOX_ARTIFACTS_DIR", str(tmp_path))
    x, se = ex1.select("beta1", "beta2"), ex1.select("se1", "se2")
    res = lm.classify(x, se, slopes=[0, 0.5, 1], scales=0.2, names=["M0", "M.5", "M1"])
    out = plots.linemodels(res, X=x, SE=se, threshold=0.95, title="ex1")
    assert out["path"] == os.path.join(str(tmp_path), "linemodels.png")
    assert os.path.getsize(out["path"]) > 0
    assert out["models"] == ["M0", "M.5", "M1"]
    assert out["n_points"] == 100
    confident = int((res["groups"]["max_prob"] >= 0.95).sum())
    assert sum(out["n_assigned"].values()) == confident
    assert out["n_undetermined"] == 100 - confident

    # models alone, with an infinite slope and a null model, draw without data
    only = plots.linemodels(
        pl.DataFrame({"model": ["null", "y"], "scale": [0.0, 0.2], "slope": [1.0, math.inf], "cor": [0.995, 0.995]}),
        path="m.png",
    )
    assert only["n_points"] == 0 and os.path.exists(os.path.join(str(tmp_path), "m.png"))
    with pytest.raises(GeneticsUsageError, match="two effect variables"):
        plots.linemodels(res, X=ex1.select("beta1", "beta2", "beta3"))
    with pytest.raises(GeneticsUsageError, match="models frame"):
        plots.linemodels(pl.DataFrame({"model": ["a"]}))
