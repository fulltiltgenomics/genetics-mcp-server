"""Line models: cluster effect estimates by the linear relationship between them.

    import genetics
    res = genetics.linemodels.classify(df.select("beta_a", "beta_b"), df.select("se_a", "se_b"))
    res["groups"]      # one row per variant: P(model) per model, the best model, its probability
    res["models"]      # the models that were used, with every default that was derived

A port of Matti Pirinen's `linemodels` R package (github.com/mjpirinen/linemodels; Pirinen M,
Bioinformatics 2023, doi:10.1093/bioinformatics/btad115), validated against that package's own
example data and the paper's COVID-19 HGI example. The R names are given beside each function.

WHAT THE MODEL IS. Each variant has M effect estimates (typically M = 2: the same variant in
two GWAS — two phenotypes, two cohorts, two sexes) with standard errors. A line model says the
true effects are scattered around a line through the origin: `slope` gives its direction
(β2 = slope · β1), `scale` the standard deviation of the larger effect (so 95% of effects lie
within 2·scale), and `cor` how tightly the effects hug the line (1 = exactly on it, 0.99–0.999
= close). Given K such models and their prior probabilities, each variant gets a posterior
probability of belonging to each model, because its likelihood under model k is
N(β̂; 0, Θ_k + Σ̂) with Θ_k the model's covariance and Σ̂ the estimate's — so the same point is
classified differently with a small SE and a large one.

THE PARAMETERS NOBODY CAN READ OFF THE DATA, and what this module does about each:

- `slopes` are hypotheses, not estimates. "Effect only in the first GWAS" is slope 0, "same
  effect in both" is slope 1, "only in the second" is slope inf, "opposite" is -1, and a
  model half-way between two lines is `tan((atan(b1) + atan(b2)) / 2)`. Without `slopes`
  the two-dimensional default is the three canonical models (0, 1, inf). When the question
  is "what IS the relationship", `optimize` fits a slope by maximum likelihood, but the
  package's own advice is not to optimise a line that already states the hypothesis.
- `scales` default to half the 95th percentile of |effect| over the data (the package's
  rule, and what published analyses used); the derived value is reported in `models`.
  Effects on the log-odds scale in published GWAS work fall between 0.15 and 0.25.
- `cors` default to 0.995. Below ~0.99 the lines stop being distinguishable.
- `r_lkhood`, the correlation between the two estimators, is 0 for disjoint samples and is
  NOT 0 for two FinnGen endpoints (they share controls) or two traits measured on one
  cohort. `estimator_correlation` computes it for case-control overlap; for two continuous
  traits on the same samples it is roughly the phenotypic correlation.
- `maf`: Pirinen recommends fitting on the √heritability scale — multiplying effects and SEs
  by √(2·maf·(1−maf)) — so rare variants with large SEs do not dominate; pass the allele
  frequencies and the transform is applied before fitting.

Every function returns a dict of polars DataFrames rather than the R matrices, and each
returned `models` frame records the parameter values actually used, so a defaulted analysis
can be restated exactly.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl

from genetics_mcp_server.sdk.errors import GeneticsUsageError

__all__ = ["classify", "proportions", "optimize", "estimator_correlation"]

# the package's stated defaults and conventions; kept as literals so the sandbox stub can
# render them
_DEFAULT_COR = 0.995
_SCALE_QUANTILE = 0.95

# below this a scale is treated as the null model: prior.V would divide by max(S) = 0
_NULL_SCALE = 1e-16


# --------------------------------------------------------------------------- the model


def _rotation_to_line(slopes: np.ndarray) -> np.ndarray:
    """Orthogonal Q taking the diagonal direction (1,…,1) onto (1, slope_1, …); R's
    rotate.diagonal.to.line."""
    m = slopes.size + 1
    x = np.ones(m) / math.sqrt(m)
    y = np.concatenate([[1.0], slopes])
    if np.isinf(slopes).any():
        # an infinite slope means the effect is entirely in that dimension; every finite
        # coordinate (the first included) collapses to zero
        finite = np.isfinite(y)
        y = np.where(finite, 0.0, np.sign(y))
    y = y / math.sqrt(float(y @ y))
    v = y - float(x @ y) * x
    v = v / math.sqrt(float(v @ v))
    cost = min(1.0, max(-1.0, float(x @ y)))
    sint = math.sqrt(1.0 - cost * cost)
    rot = np.array([[cost, -sint], [sint, cost]])
    a = np.column_stack([x, v])
    return np.eye(m) - np.outer(x, x) - np.outer(v, v) + a @ rot @ a.T


def _prior_covariance(scale: float, slopes: np.ndarray, cor: float) -> np.ndarray:
    """The model's covariance of the true effects, Θ_k; R's prior.V.

    The distribution with pairwise correlation `cor` around the diagonal is rotated onto the
    model's line and rescaled so the largest variance is scale². scale 0 is the null model.
    """
    m = slopes.size + 1
    if scale < _NULL_SCALE:
        return np.zeros((m, m))
    q = _rotation_to_line(slopes)
    c = np.full((m, m), cor)
    np.fill_diagonal(c, 1.0)
    s = q @ c @ q.T
    s = 0.5 * (s + s.T)
    return scale * scale * s / s.max()


def _log_densities(
    x: np.ndarray, se: np.ndarray, r: np.ndarray, spec: _Models, constant_se: bool
) -> np.ndarray:
    """log N(x_i; 0, Σ̂_i + Θ_k) for every row i and model k, as an (n, K) array.

    Batched over rows with one Cholesky per model rather than R's row loop: the only thing
    that varies per row is diag(SE_i) R diag(SE_i), and numpy factorises a stack at once.
    """
    n, m = x.shape
    if constant_se:
        med = np.median(se, axis=0)
        v = (med[:, None] * r * med[None, :])[None, :, :]
    else:
        v = se[:, :, None] * r[None, :, :] * se[:, None, :]
    out = np.empty((n, spec.k))
    rhs = x[:, :, None]
    for k in range(spec.k):
        try:
            chol = np.linalg.cholesky(v + spec.covariances[k][None, :, :])
        except np.linalg.LinAlgError as exc:
            raise GeneticsUsageError(
                f"the covariance of model {spec.names[k]!r} plus the standard errors is not "
                "positive definite; every SE must be positive and r_lkhood a valid "
                "correlation"
            ) from exc
        a = np.linalg.solve(chol, rhs)[:, :, 0]
        logdet = 2.0 * np.log(np.diagonal(chol, axis1=-2, axis2=-1)).sum(axis=-1)
        out[:, k] = -0.5 * m * math.log(2.0 * math.pi) - 0.5 * (logdet + (a * a).sum(axis=1))
    return out


def _posteriors(logd: np.ndarray, log_pis: np.ndarray) -> tuple[np.ndarray, float]:
    """Membership probabilities per row and the mixture log-likelihood of the whole data."""
    t = logd + log_pis[None, :]
    top = t.max(axis=1, keepdims=True)
    log_rowsum = np.log(np.exp(t - top).sum(axis=1, keepdims=True)) + top
    return np.exp(t - log_rowsum), float(log_rowsum.sum())


# --------------------------------------------------------------------------- inputs


class _Models:
    """The K models as arrays, plus the names of the effect dimensions."""

    def __init__(
        self,
        names: list[str],
        scales: np.ndarray,
        slopes: np.ndarray,
        cors: np.ndarray,
        priors: np.ndarray,
        dims: list[str],
        scale_source: str,
    ) -> None:
        self.names = names
        self.scales = scales
        self.slopes = slopes
        self.cors = cors
        self.priors = priors
        self.dims = dims
        self.scale_source = scale_source

    @property
    def k(self) -> int:
        return len(self.names)

    @property
    def covariances(self) -> list[np.ndarray]:
        return [
            _prior_covariance(float(self.scales[i]), self.slopes[i], float(self.cors[i]))
            for i in range(self.k)
        ]

    def frame(self, weights: np.ndarray | None = None, weight_name: str = "prior") -> pl.DataFrame:
        """One row per model with the parameter values that were used."""
        m = self.slopes.shape[1] + 1
        cols: dict[str, Any] = {"model": self.names, "scale": self.scales.tolist()}
        if m == 2:
            cols["slope"] = self.slopes[:, 0].tolist()
        else:
            for j in range(m - 1):
                cols[f"slope_{self.dims[j + 1]}"] = self.slopes[:, j].tolist()
        cols["cor"] = self.cors.tolist()
        if weights is not None:
            cols[weight_name] = weights.tolist()
        return pl.DataFrame(cols)


def _matrix(value: Any, what: str) -> tuple[np.ndarray, list[str] | None]:
    if isinstance(value, pl.DataFrame):
        names = list(value.columns)
        try:
            arr = value.select(pl.all().cast(pl.Float64)).to_numpy()
        except Exception as exc:
            raise GeneticsUsageError(f"{what} has a column that is not numeric: {exc}") from exc
    else:
        names = None
        arr = np.asarray(value, dtype=float)
    if arr.ndim != 2:
        raise GeneticsUsageError(
            f"{what} must be two-dimensional: one row per variant, one column per effect "
            f"variable (got shape {arr.shape})"
        )
    if arr.shape[1] < 2:
        raise GeneticsUsageError(f"{what} needs at least two columns; got {arr.shape[1]}")
    if arr.shape[0] == 0:
        raise GeneticsUsageError(f"{what} is empty")
    if not np.isfinite(arr).all():
        raise GeneticsUsageError(
            f"{what} contains null, NaN or infinite values; drop those rows before fitting"
        )
    return arr, names


def _data(x: Any, se: Any, maf: Any) -> tuple[np.ndarray, np.ndarray, list[str], bool]:
    xa, xnames = _matrix(x, "X")
    sa, snames = _matrix(se, "SE")
    if sa.shape != xa.shape:
        raise GeneticsUsageError(f"X has shape {xa.shape} but SE has shape {sa.shape}")
    if (sa <= 0).any():
        raise GeneticsUsageError("every standard error in SE must be positive")
    dims = xnames or snames or [f"effect{j + 1}" for j in range(xa.shape[1])]
    scaled = False
    if maf is not None:
        f = np.asarray(maf.to_numpy() if isinstance(maf, pl.Series) else maf, dtype=float)
        if f.shape != (xa.shape[0],):
            raise GeneticsUsageError(
                f"maf must have one value per row of X ({xa.shape[0]}); got shape {f.shape}"
            )
        if not np.isfinite(f).all() or (f <= 0).any() or (f >= 1).any():
            raise GeneticsUsageError("every maf must lie strictly between 0 and 1")
        factor = np.sqrt(2.0 * f * (1.0 - f))[:, None]
        xa = xa * factor
        sa = sa * factor
        scaled = True
    return xa, sa, dims, scaled


def _correlation_matrix(r_lkhood: Any, m: int) -> np.ndarray:
    r = np.asarray(r_lkhood, dtype=float)
    if r.ndim == 0:
        out = np.full((m, m), float(r))
        np.fill_diagonal(out, 1.0)
    elif r.ndim == 1:
        pairs = m * (m - 1) // 2
        if r.size != pairs:
            raise GeneticsUsageError(
                f"r_lkhood as a vector must give the {pairs} upper-triangle correlations "
                f"in row-major order for {m} effect variables; got {r.size} values"
            )
        out = np.eye(m)
        iu = np.triu_indices(m, 1)
        out[iu] = r
        out = out + out.T - np.eye(m)
    elif r.shape == (m, m):
        out = r.copy()
    else:
        raise GeneticsUsageError(
            f"r_lkhood must be a number, a vector of the upper triangle or an {m}x{m} "
            f"matrix; got shape {r.shape}"
        )
    if not np.allclose(out, out.T, atol=1e-10) or not np.allclose(np.diag(out), 1.0, atol=1e-10):
        raise GeneticsUsageError("r_lkhood must be a symmetric correlation matrix with unit diagonal")
    if (out < -1).any() or (out > 1).any():
        raise GeneticsUsageError("every value of r_lkhood must lie in [-1, 1]")
    return out


def _per_model(value: Any, k: int, what: str, default: float | None = None) -> np.ndarray:
    if value is None:
        if default is None:
            raise GeneticsUsageError(f"{what} is required")
        return np.full(k, float(default))
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(k, float(arr))
    if arr.shape != (k,):
        raise GeneticsUsageError(f"{what} must have one value per model ({k}); got shape {arr.shape}")
    return arr


def _default_scale(x: np.ndarray) -> float:
    """Half the 95th percentile of |effect|, the largest over the dimensions: the package's
    rule that 95% of effects lie within 2·scale, applied to the observed effects."""
    q = np.quantile(np.abs(x), _SCALE_QUANTILE, axis=0)
    return float(q.max()) / 2.0


def _models(
    x: np.ndarray,
    dims: list[str],
    *,
    slopes: Any,
    scales: Any,
    cors: Any,
    names: Any,
    priors: Any,
) -> _Models:
    m = x.shape[1]
    if slopes is None:
        if m != 2:
            raise GeneticsUsageError(
                "slopes is required when X has more than two columns: give one row per "
                "model with a slope for each effect variable after the first"
            )
        slope_arr = np.array([[0.0], [1.0], [math.inf]])
        default_names = [f"{dims[0]}_only", "shared", f"{dims[1]}_only"]
    else:
        slope_arr = np.asarray(slopes, dtype=float)
        if slope_arr.ndim == 1:
            if m != 2:
                raise GeneticsUsageError(
                    f"with {m} effect variables slopes must be a matrix with {m - 1} columns, "
                    "one row per model"
                )
            slope_arr = slope_arr[:, None]
        if slope_arr.ndim != 2 or slope_arr.shape[1] != m - 1 or slope_arr.shape[0] == 0:
            raise GeneticsUsageError(
                f"slopes must have one row per model and {m - 1} column(s); got shape "
                f"{slope_arr.shape}"
            )
        if np.isnan(slope_arr).any():
            raise GeneticsUsageError("slopes must not contain NaN")
        default_names = [f"M{i + 1}" for i in range(slope_arr.shape[0])]
    k = slope_arr.shape[0]

    if scales is None:
        scale_arr = np.full(k, _default_scale(x))
        scale_source = "derived"
    else:
        scale_arr = _per_model(scales, k, "scales")
        scale_source = "given"
    if (scale_arr < 0).any():
        raise GeneticsUsageError("every scale must be non-negative (0 is the null model)")

    cor_arr = _per_model(cors, k, "cors", _DEFAULT_COR)
    if (cor_arr < 0).any() or (cor_arr > 1).any():
        raise GeneticsUsageError(
            "every cor must lie in [0, 1]; a negative relationship is a negative slope"
        )

    if names is None:
        name_list = default_names
    else:
        name_list = [str(n) for n in names]
        if len(name_list) != k:
            raise GeneticsUsageError(f"names must have one entry per model ({k}); got {len(name_list)}")
        if len(set(name_list)) != k:
            raise GeneticsUsageError("model names must be unique")

    prior_arr = _per_model(priors, k, "priors", 1.0)
    if (prior_arr < 0).any() or prior_arr.sum() <= 0:
        raise GeneticsUsageError("priors must be non-negative and not all zero")
    prior_arr = prior_arr / prior_arr.sum()

    return _Models(name_list, scale_arr, slope_arr, cor_arr, prior_arr, dims, scale_source)


def _groups_frame(post: np.ndarray, spec: _Models) -> pl.DataFrame:
    best = post.argmax(axis=1)
    return pl.DataFrame({
        **{name: post[:, i].tolist() for i, name in enumerate(spec.names)},
        "model": [spec.names[i] for i in best],
        "max_prob": post[np.arange(post.shape[0]), best].tolist(),
    })


# --------------------------------------------------------------------------- functions


def classify(
    X: Any,
    SE: Any,
    *,
    slopes: Any = None,
    scales: Any = None,
    cors: Any = None,
    names: Any = None,
    priors: Any = None,
    r_lkhood: Any = 0.0,
    maf: Any = None,
) -> dict[str, Any]:
    """Posterior probability of each line model for each variant, with fixed priors.

    R: `line.models`. `X` holds the effect estimates and `SE` their standard errors, one row
    per variant and one column per effect variable, as polars DataFrames (column names become
    the variable names) or 2-D arrays. Rows with a null in either are refused, so filter first.

    `slopes` — one per model (two effect variables), or one row per model with a slope for
    each variable after the first (more). Slope b means β_j = b · β_1: 0 is "effect only in
    the first variable", 1 "the same effect in both", `float("inf")` "only in the second",
    -1 "opposite". Default, for two variables: those first three, named `<first>_only`,
    `shared`, `<second>_only`.
    `scales` — SD of the larger true effect under each model; one value applies to all.
    Default: half the 95th percentile of |effect| in the data (reported in `models`).
    `cors` — how tightly effects hug the line; default 0.995. Use 0.99–0.999.
    `priors` — prior weight per model, normalised; default equal. To let the data set them,
    use `proportions`.
    `r_lkhood` — correlation between the estimators of the effect variables: 0 for disjoint
    samples, `estimator_correlation(...)` for overlapping case-control GWAS (two FinnGen
    endpoints share controls), about the phenotypic correlation for two traits measured on
    the same people. A number for two variables, or the upper triangle or full matrix.
    `maf` — allele frequency per row; when given, effects and SEs are multiplied by
    √(2·maf·(1−maf)) before fitting, the scale Pirinen recommends fitting on.

    Returns `{"groups", "models", "loglik", "scaled"}`. `groups` has one row per input row:
    a probability column per model, `model` (the most probable) and `max_prob`; join it back
    with `pl.concat([df, res["groups"]], how="horizontal")`. `models` lists each model's
    scale, slope, cor and prior as used. A variant is usually called assigned at
    `max_prob >= 0.95`; points near the origin stay undetermined by design.
    """
    x, se, dims, scaled = _data(X, SE, maf)
    spec = _models(x, dims, slopes=slopes, scales=scales, cors=cors, names=names, priors=priors)
    r = _correlation_matrix(r_lkhood, x.shape[1])
    post, loglik = _posteriors(_log_densities(x, se, r, spec, False), np.log(spec.priors))
    return {
        "groups": _groups_frame(post, spec),
        "models": spec.frame(spec.priors, "prior"),
        "loglik": loglik,
        "scale_source": spec.scale_source,
        "scaled": scaled,
    }


def proportions(
    X: Any,
    SE: Any,
    *,
    slopes: Any = None,
    scales: Any = None,
    cors: Any = None,
    names: Any = None,
    r_lkhood: Any = 0.0,
    maf: Any = None,
    n_iter: int = 2000,
    n_burnin: int = 200,
    diri_prior: Any = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Membership probabilities with the model proportions estimated from the data.

    R: `line.models.with.proportions`. The same models as `classify`, but instead of fixed
    priors the proportion of variants under each model gets a Dirichlet(`diri_prior`) prior
    (default 1/K each) and a Gibbs sampler estimates it jointly with the memberships. Use
    this when the question includes "what fraction of these variants are shared". Arguments
    are as in `classify`; `n_iter` post-burn-in sweeps set the resolution of the returned
    probabilities (1/n_iter), and `seed` makes a run repeatable.

    Returns `{"groups", "models", "params", "n_iter", "n_burnin"}`. `params` has one row per
    model: posterior `mean`, `low95`, `up95` and `sd` of its proportion. `groups` is as in
    `classify`. The R package's default is 200 sweeps after 20; the paper's example used
    10,000, and the default here is 2,000 after 200, which the numbers are stable at.
    """
    x, se, dims, scaled = _data(X, SE, maf)
    spec = _models(x, dims, slopes=slopes, scales=scales, cors=cors, names=names, priors=None)
    r = _correlation_matrix(r_lkhood, x.shape[1])
    if n_iter < 1:
        raise GeneticsUsageError("n_iter must be at least 1")
    if n_burnin < 0:
        raise GeneticsUsageError("n_burnin must be non-negative")
    alpha = _per_model(diri_prior, spec.k, "diri_prior", 1.0 / spec.k)
    if (alpha <= 0).any():
        raise GeneticsUsageError("every diri_prior value must be positive")

    logd = _log_densities(x, se, r, spec, False)
    n, k = logd.shape
    rng = np.random.default_rng(seed)
    counts = np.zeros((n, k))
    draws = np.empty((n_iter, k))
    groups = rng.integers(0, k, size=n)
    for it in range(n_burnin + n_iter):
        tally = np.bincount(groups, minlength=k)
        pis = rng.dirichlet(alpha + tally)
        t = logd + np.log(pis)[None, :]
        p = np.exp(t - t.max(axis=1, keepdims=True))
        cum = p.cumsum(axis=1)
        u = rng.random(n) * cum[:, -1]
        groups = (cum < u[:, None]).sum(axis=1)
        if it >= n_burnin:
            counts[np.arange(n), groups] += 1.0
            draws[it - n_burnin] = pis
    post = counts / n_iter
    lo, hi = np.quantile(draws, [0.025, 0.975], axis=0)
    params = pl.DataFrame({
        "model": spec.names,
        "mean": draws.mean(axis=0).tolist(),
        "low95": lo.tolist(),
        "up95": hi.tolist(),
        "sd": draws.std(axis=0, ddof=1).tolist() if n_iter > 1 else [0.0] * k,
    })
    return {
        "groups": _groups_frame(post, spec),
        "models": spec.frame(),
        "params": params,
        "n_iter": n_iter,
        "n_burnin": n_burnin,
        "scale_source": spec.scale_source,
        "scaled": scaled,
    }


# which parameters `optimize` fits, as (K, 3) booleans over scale / slope row / cor
_FIT_KEYS = ("scales", "slopes", "cors")


def _fit_mask(fit: Any, k: int, m: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scales = np.zeros(k, bool)
    slopes = np.zeros((k, m - 1), bool)
    cors = np.zeros(k, bool)
    if fit is None or fit == "proportions":
        return scales, slopes, cors
    if isinstance(fit, str):
        fit = (fit,)
    if isinstance(fit, dict):
        unknown = sorted(set(fit) - set(_FIT_KEYS))
        if unknown:
            raise GeneticsUsageError(f"fit has unknown keys {unknown}; expected {_FIT_KEYS}")
        if "scales" in fit:
            scales = np.broadcast_to(np.asarray(fit["scales"], bool), (k,)).copy()
        if "slopes" in fit:
            s = np.asarray(fit["slopes"], bool)
            if s.ndim == 1 and m == 2:
                s = s[:, None]
            slopes = np.broadcast_to(s, (k, m - 1)).copy()
        if "cors" in fit:
            cors = np.broadcast_to(np.asarray(fit["cors"], bool), (k,)).copy()
        return scales, slopes, cors
    if isinstance(fit, (list, tuple)) and all(isinstance(f, str) for f in fit):
        keys = set(fit)
        if "all" in keys:
            keys = set(_FIT_KEYS)
        unknown = sorted(keys - set(_FIT_KEYS))
        if unknown:
            raise GeneticsUsageError(
                f"fit names unknown parameters {unknown}; expected any of {_FIT_KEYS}, 'all' "
                "or 'proportions'"
            )
        return (
            np.full(k, "scales" in keys),
            np.full((k, m - 1), "slopes" in keys),
            np.full(k, "cors" in keys),
        )
    mat = np.asarray(fit, bool)
    if mat.shape != (k, 3):
        raise GeneticsUsageError(
            f"fit must be a parameter name, a list of them, a dict, or a {k}x3 boolean matrix "
            "(scale, slope, cor per model)"
        )
    return mat[:, 0].copy(), np.repeat(mat[:, 1:2], m - 1, axis=1), mat[:, 2].copy()


def optimize(
    X: Any,
    SE: Any,
    *,
    slopes: Any,
    scales: Any = None,
    cors: Any = None,
    names: Any = None,
    priors: Any = None,
    r_lkhood: Any = 0.0,
    maf: Any = None,
    fit: Any = "slopes",
    force_same_scales: bool = False,
    tol_loglik: float = 1e-3,
    tol_par: float = 0.0,
    method: str = "BFGS",
    constant_se: bool = False,
    max_iter: int = 100,
) -> dict[str, Any]:
    """Fit chosen line-model parameters by maximum likelihood (EM), starting from `slopes`.

    R: `line.models.optimize`. `slopes`, `scales` and `cors` are the initial values (the same
    forms and defaults as in `classify`); `fit` says which of them move: `"slopes"` (default),
    `"scales"`, `"cors"`, `"all"`, a list of those, `"proportions"` for none, a dict such as
    `{"slopes": [False, True, False]}` naming them per model, or a K×3 boolean matrix. The
    mixture proportions are always fitted. `force_same_scales` fits one scale shared by
    every model whose scale is being fitted, which stabilises small data sets.

    Use it when the data show a relationship whose slope is unknown — "the shared variants
    follow some line, which?" — and keep a line fixed when it states the hypothesis (slope 0
    or 1). Fitting `cors` weakens what a line means; published analyses fix them. EM finds a
    local optimum: rerun from other starting slopes on real data. Do not fit a slope that
    starts at infinity. `constant_se` replaces each column's SEs by their median, which is
    appropriate after `maf` scaling and much faster.

    Returns `{"models", "weights", "loglik", "groups", "n_iter", "converged", "criterion"}`:
    `models` carries the fitted scale, slope and cor per model and its `weight`; `groups`
    is the classification at the optimum, as in `classify`.
    """
    from scipy import optimize as sp_opt

    x, se, dims, scaled = _data(X, SE, maf)
    if slopes is None:
        raise GeneticsUsageError("optimize needs initial slopes: one per model")
    spec = _models(x, dims, slopes=slopes, scales=scales, cors=cors, names=names, priors=priors)
    r = _correlation_matrix(r_lkhood, x.shape[1])
    if tol_loglik <= 0:
        raise GeneticsUsageError("tol_loglik must be positive")
    if method not in ("BFGS", "Nelder-Mead"):
        raise GeneticsUsageError("method must be 'BFGS' or 'Nelder-Mead'")
    k, m = spec.k, x.shape[1]
    fit_scales, fit_slopes, fit_cors = _fit_mask(fit, k, m)
    if fit_slopes.any() and np.isinf(spec.slopes[fit_slopes]).any():
        raise GeneticsUsageError("a slope being fitted must start finite")
    same = bool(force_same_scales and fit_scales.any())
    if same:
        spec.scales[fit_scales] = spec.scales[fit_scales].mean()
    n_optim = int(fit_scales.sum() + fit_slopes.sum() + fit_cors.sum())
    if same:
        n_optim -= int(fit_scales.sum()) - 1

    def pack(sc: np.ndarray, sl: np.ndarray, co: np.ndarray) -> np.ndarray:
        # the transforms R uses: log scale, atan slope, logit cor (clamped at ±40)
        parts = []
        if fit_scales.any():
            logs = np.minimum(np.log(sc[fit_scales]), math.log(1e300))
            parts.append(logs[:1] if same else logs)
        if fit_slopes.any():
            parts.append(np.arctan(sl[fit_slopes]))
        if fit_cors.any():
            c = co[fit_cors]
            logit = np.log(c) - np.log1p(-c)
            parts.append(np.clip(logit, -40.0, 40.0))
        return np.concatenate(parts) if parts else np.empty(0)

    def unpack(par: np.ndarray, sc: np.ndarray, sl: np.ndarray, co: np.ndarray):
        sc, sl, co = sc.copy(), sl.copy(), co.copy()
        i = 0
        if fit_scales.any():
            n_sc = 1 if same else int(fit_scales.sum())
            sc[fit_scales] = np.exp(par[i : i + n_sc]) if not same else math.exp(par[i])
            i += n_sc
        if fit_slopes.any():
            n_sl = int(fit_slopes.sum())
            sl[fit_slopes] = np.tan(par[i : i + n_sl])
            i += n_sl
        if fit_cors.any():
            co[fit_cors] = 1.0 / (1.0 + np.exp(-par[i:]))
        return sc, sl, co

    def loglik_of(sc, sl, co, w):
        trial = _Models(spec.names, sc, sl, co, w, dims, spec.scale_source)
        return _posteriors(_log_densities(x, se, r, trial, constant_se), np.log(w))

    cur_sc, cur_sl, cur_co = spec.scales.copy(), spec.slopes.copy(), spec.cors.copy()
    w = spec.priors.copy()
    post, loglik = loglik_of(cur_sc, cur_sl, cur_co, w)

    # R uses Brent over a bounded range when exactly one parameter moves
    bounds = None
    if n_optim == 1:
        if fit_scales.any():
            bounds = (math.log(1e-6), math.log(1e6))
        elif fit_slopes.any():
            bounds = (-math.pi / 2, math.pi / 2)
        else:
            bounds = (math.log(1e-6 / (1 - 1e-6)), math.log((1 - 1e-6) / 1e-6))

    n_iter = 0
    converged = False
    criterion = None
    while not converged and n_iter < max_iter:
        n_iter += 1
        prev_loglik = loglik
        prev = (cur_sc.copy(), cur_sl.copy(), cur_co.copy())
        new_w = post.mean(axis=0)
        if n_optim > 0:
            def negative_expected(par):
                sc, sl, co = unpack(par, cur_sc, cur_sl, cur_co)
                trial = _Models(spec.names, sc, sl, co, w, dims, spec.scale_source)
                return -float((post * _log_densities(x, se, r, trial, constant_se)).sum())

            start = pack(cur_sc, cur_sl, cur_co)
            if bounds is not None:
                res = sp_opt.minimize_scalar(
                    lambda v: negative_expected(np.array([v])), bounds=bounds, method="bounded"
                )
                new_par = np.array([res.x])
            else:
                res = sp_opt.minimize(negative_expected, start, method=method)
                new_par = res.x
            new_sc, new_sl, new_co = unpack(new_par, cur_sc, cur_sl, cur_co)
        else:
            new_sc, new_sl, new_co = cur_sc, cur_sl, cur_co
        new_post, new_loglik = loglik_of(new_sc, new_sl, new_co, new_w)
        if new_loglik > loglik:
            loglik, w, post = new_loglik, new_w, new_post
            cur_sc, cur_sl, cur_co = new_sc, new_sl, new_co
        # a step that fails to raise the likelihood is discarded and both criteria then hold
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.concatenate([
                np.abs((cur_sc - prev[0]) / prev[0]),
                np.abs((cur_sl - prev[1]) / prev[1]).ravel(),
                np.abs((cur_co - prev[2]) / prev[2]),
            ])
        rel = rel[~np.isnan(rel)]
        par_ok = bool((rel <= tol_par).all())
        loglik_ok = (loglik - prev_loglik) < tol_loglik
        converged = par_ok or loglik_ok
        if converged:
            criterion = "loglik" if loglik_ok else "parameters"

    fitted = _Models(spec.names, cur_sc, cur_sl, cur_co, w, dims, spec.scale_source)
    return {
        "models": fitted.frame(w, "weight"),
        "weights": dict(zip(spec.names, w.tolist())),
        "loglik": loglik,
        "groups": _groups_frame(post, fitted),
        "n_iter": n_iter,
        "converged": converged,
        "criterion": criterion,
        "scaled": scaled,
    }


def estimator_correlation(
    cases1: float,
    controls1: float,
    cases2: float,
    controls2: float,
    *,
    shared_cases: float = 0.0,
    shared_controls: float = 0.0,
    cases1_controls2: float = 0.0,
    controls1_cases2: float = 0.0,
) -> float:
    """Correlation between two case-control GWAS effect estimators from their sample overlap.

    R: `beta.cor.case.control`; the formula of Bhattacharjee et al. (2012, AJHG 90:821). The
    value is what `r_lkhood` should be when the two GWAS share samples: `shared_cases` are
    people who are cases in both, `shared_controls` controls in both, and the two mixed
    counts people who are a case in one study and a control in the other. Two endpoints from
    the same biobank typically share most of their controls, so for those pass roughly
    `shared_controls=min(controls1, controls2)`; cases shared depends on the comorbidity and
    is 0 when unknown. Disjoint studies give 0.
    """
    for name, value in (("cases1", cases1), ("controls1", controls1), ("cases2", cases2), ("controls2", controls2)):
        if value <= 0:
            raise GeneticsUsageError(f"{name} must be positive")
    return (
        math.sqrt(cases1 * controls1 / (cases1 + controls1))
        * math.sqrt(cases2 * controls2 / (cases2 + controls2))
        * (
            shared_cases / cases1 / cases2
            - cases1_controls2 / cases1 / controls2
            - controls1_cases2 / controls1 / cases2
            + shared_controls / controls1 / controls2
        )
    )
