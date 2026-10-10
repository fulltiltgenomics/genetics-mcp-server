"""Two-sample MR estimators over a harmonised instrument frame.

    import genetics
    h = genetics.mr.harmonise(exposure, outcome)
    genetics.mr.ivw(h)                  # dict: method, estimate, se, p, ci_low, ci_high, ...
    genetics.mr.leave_one_out(h)        # polars frame, one row per left-out variant + "All"

Every function takes a polars DataFrame with `variant`, `beta_x`, `se_x`, `beta_y`, `se_y`
(and, where a function needs them, `n_x`, `n_y`, `n_cases`, `n_controls`, `af_y`) or the
`Harmonised` object `harmonise` returns. A scalar result is a dict carrying `method`,
`estimate`, `se`, `p`, `ci_low`, `ci_high`, `n_instruments` and `options` — the defaults the
call actually used, so a defaulted analysis can be restated exactly.

WHICH PACKAGE EACH FOLLOWS. The estimates are MendelianRandomization 0.10.0's (`mr_ivw`,
`mr_egger`, `mr_median`, `mr_mbe`); Steiger, leave-one-out and single-SNP are TwoSampleMR
0.7.12's. Both are held to tests/fixtures/mr/r_reference. Where the two packages disagree the
difference is stated at the function:

- p-values and CIs are normal-theory for every MendelianRandomization estimate (its default
  `distribution = "normal"`). TwoSampleMR uses a t distribution for MR Egger (K - 2 df) and
  for the modes (K - 1 df); only `single_snp`'s Egger row here follows that.
- the weighted median weights are first-order (`se_y / |beta_x|`) in MendelianRandomization
  and second-order in TwoSampleMR; `weights="second_order"` gives the latter.
- `ivw`'s `model="default"` is fixed-effect for K <= 3 instruments and multiplicative random
  effects for K >= 4 (MendelianRandomization's rule). TwoSampleMR's IVW, used in
  `leave_one_out` and `single_snp`, is always multiplicative random effects.
- multiplicative random effects scale the fixed-effect SE by max(1, RSE), RSE being the
  residual standard error sqrt(Q / df) of the weighted regression.
- MR Egger orients every variant to a positive `beta_x` before regressing, flipping
  `beta_y` with it; the intercept is on that orientation.

Bootstrap SEs (median, mode) use numpy's generator with the given seed. R's RNG is not
reproduced, so they agree with the packages only within Monte Carlo error; the point
estimates are deterministic and match to floating point.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl

from genetics_mcp_server.sdk.errors import GeneticsUsageError

__all__ = [
    "wald",
    "ivw",
    "egger",
    "weighted_median",
    "simple_median",
    "weighted_mode",
    "simple_mode",
    "heterogeneity",
    "leave_one_out",
    "single_snp",
    "steiger",
]

_Z975 = 1.959963984540054
# MendelianRandomization's own default seed for mr_median and mr_mbe
_MR_SEED = 314159265
_ITERATIONS = 10000
# floor applied to the eigenvalues before rescaling to a unit diagonal (the rescale can take
# one slightly below it; the matrix stays positive definite); the 0.01 r2 floor of the LD
# panels zeroes small r, which can leave the served matrix indefinite
_PD_FLOOR = 1e-6


# --------------------------------------------------------------------------- inputs


def _frame_of(data: Any) -> pl.DataFrame:
    frame = getattr(data, "frame", data)
    if not isinstance(frame, pl.DataFrame):
        raise GeneticsUsageError(
            "expected a polars DataFrame of harmonised instruments or harmonise()'s result"
        )
    missing = [c for c in ("variant", "beta_x", "se_x", "beta_y", "se_y") if c not in frame.columns]
    if missing:
        raise GeneticsUsageError(f"the instrument frame lacks {', '.join(missing)}")
    return frame


def _col(frame: pl.DataFrame, name: str) -> np.ndarray:
    v = frame[name].cast(pl.Float64).to_numpy()
    if not np.isfinite(v).all():
        raise GeneticsUsageError(
            f"`{name}` has missing or infinite values; drop those instruments first"
        )
    return v


class _Inst:
    """The four effect arrays, checked once."""

    def __init__(self, data: Any, min_k: int = 1, method: str = "this estimator"):
        self.frame = _frame_of(data)
        self.variant = self.frame["variant"].cast(pl.String).to_list()
        self.bx = _col(self.frame, "beta_x")
        self.sx = _col(self.frame, "se_x")
        self.by = _col(self.frame, "beta_y")
        self.sy = _col(self.frame, "se_y")
        self.k = self.bx.size
        if self.k < min_k:
            raise GeneticsUsageError(
                f"{method} needs at least {min_k} instruments, got {self.k}"
            )
        if (self.bx == 0).any():
            raise GeneticsUsageError("an instrument has beta_x = 0; its ratio is undefined")
        if (self.sy <= 0).any() or (self.sx <= 0).any():
            raise GeneticsUsageError("standard errors must be positive")


def _p_norm(z: Any) -> Any:
    from scipy.stats import norm

    return 2 * norm.sf(np.abs(z))


def _p_t(z: Any, df: float) -> Any:
    from scipy.stats import t

    return 2 * t.sf(np.abs(z), df)


def _p_chi2(q: float, df: float) -> float:
    from scipy.stats import chi2

    return float(chi2.sf(q, df)) if df > 0 else math.nan


def _result(method: str, est: float, se: float, k: int, options: dict, **extra: Any) -> dict:
    est, se = float(est), float(se)
    out = {
        "method": method,
        "estimate": est,
        "se": se,
        "p": float(_p_norm(est / se)) if se > 0 else math.nan,
        "ci_low": est - _Z975 * se,
        "ci_high": est + _Z975 * se,
        "n_instruments": k,
    }
    out.update(extra)
    out["options"] = options
    return out


# --------------------------------------------------------------------------- Wald


def wald(data: Any) -> pl.DataFrame:
    """Per-variant Wald ratio beta_y / beta_x with first-order SE se_y / |beta_x| and a
    two-sided normal p. One instrument is this table's single row; `ivw` on one instrument
    returns that row as its result."""
    d = _Inst(data)
    est = d.by / d.bx
    se = d.sy / np.abs(d.bx)
    return pl.DataFrame({
        "variant": d.variant,
        "estimate": est,
        "se": se,
        "p": _p_norm(est / se),
        "ci_low": est - _Z975 * se,
        "ci_high": est + _Z975 * se,
    })


# --------------------------------------------------------------------------- IVW


def _correlation(correl: Any, variants: list[str]) -> tuple[np.ndarray, list[str]]:
    """The signed r matrix in frame order, and the frame variants it does not cover.

    A polars frame is read as the LD files are written: a `variant` column naming the rows
    and one column per variant. A numpy array must already be in frame order.
    """
    if isinstance(correl, pl.DataFrame):
        if "variant" not in correl.columns:
            raise GeneticsUsageError("a correlation frame needs a `variant` column naming its rows")
        rows = correl["variant"].cast(pl.String).to_list()
        missing_cols = [v for v in rows if v not in correl.columns]
        if missing_cols:
            raise GeneticsUsageError(f"correlation frame lacks columns for {missing_cols}")
        keep = [v for v in variants if v in rows]
        dropped = [v for v in variants if v not in rows]
        idx = [rows.index(v) for v in keep]
        r = correl.select(keep).cast(pl.Float64).to_numpy()[idx, :]
        return r, dropped
    r = np.asarray(correl, dtype=float)
    if r.shape != (len(variants), len(variants)):
        raise GeneticsUsageError(
            f"correlation matrix is {r.shape}, expected {len(variants)} x {len(variants)}"
        )
    return r, []


def _repair(r: np.ndarray) -> tuple[np.ndarray, dict]:
    """Clip a symmetric r matrix's eigenvalues to a positive floor and rescale it back to a
    unit diagonal, reporting what was done."""
    r = (r + r.T) / 2
    vals, vecs = np.linalg.eigh(r)
    info = {"min_eigenvalue": float(vals.min()), "repaired": False}
    if vals.min() > _PD_FLOOR:
        return r, info
    fixed = (vecs * np.maximum(vals, _PD_FLOOR)) @ vecs.T
    scale = 1 / np.sqrt(np.diag(fixed))
    info.update(repaired=True, eigenvalue_floor=_PD_FLOOR)
    return fixed * np.outer(scale, scale), info


def _subset(data: Any, keep: list[str]) -> pl.DataFrame:
    frame = _frame_of(data)
    return frame.filter(pl.col("variant").cast(pl.String).is_in(keep))


def _gls(x: np.ndarray, y: np.ndarray, omega_inv: np.ndarray):
    """Generalised least squares: coefficients, unscaled covariance, residual quadratic form."""
    xtw = x.T @ omega_inv
    cov = np.linalg.inv(xtw @ x)
    beta = cov @ xtw @ y
    resid = y - x @ beta
    return beta, cov, float(resid @ omega_inv @ resid)


def ivw(data: Any, *, model: str = "default", correl: Any = None) -> dict:
    """Inverse-variance weighted estimate; MendelianRandomization::mr_ivw.

    Weighted regression of beta_y on beta_x through the origin with weights 1/se_y^2
    (first-order weights). `model`: "fixed", "random" (multiplicative: SE times max(1, RSE))
    or "default" — fixed for K <= 3, random for K >= 4, the package's rule. `Q`, `Q_df`,
    `Q_pval` and `I2` (max(0, (Q - df)/Q)) are Cochran's heterogeneity on the same fit.

    `correl` is the signed LD r among the instruments (a numpy array in frame order, or a
    polars frame with a `variant` column, as genetics.ld matrices are written). Instruments
    the matrix lacks are left out and listed in `options["dropped_no_ld"]`. The fit becomes
    generalised least squares with Omega = se_y se_y' * r; an r matrix that is not positive
    definite is repaired by clipping its eigenvalues to a small floor, and `options` says so.
    """
    if model not in ("default", "fixed", "random"):
        raise GeneticsUsageError("model must be 'default', 'fixed' or 'random'")
    frame = _frame_of(data)
    options: dict[str, Any] = {"model": model, "weights": "first_order", "correl": correl is not None}
    r = None
    if correl is not None:
        r, dropped = _correlation(correl, frame["variant"].cast(pl.String).to_list())
        if dropped:
            frame = _subset(frame, [v for v in frame["variant"].to_list() if v not in dropped])
            options["dropped_no_ld"] = dropped
    d = _Inst(frame)
    if d.k == 1:
        row = wald(frame).row(0, named=True)
        options["single_instrument"] = "the IVW estimate of one instrument is its Wald ratio"
        return _result("wald", row["estimate"], row["se"], 1, options,
                       Q=math.nan, Q_df=0, Q_pval=math.nan, I2=math.nan, RSE=math.nan,
                       model="fixed")
    used = model if model != "default" else ("fixed" if d.k <= 3 else "random")
    if r is None:
        omega_inv = np.diag(d.sy ** -2)
    else:
        r, info = _repair(r)
        options.update(info)
        omega_inv = np.linalg.inv(np.outer(d.sy, d.sy) * r)
    beta, cov, q = _gls(d.bx[:, None], d.by, omega_inv)
    df = d.k - 1
    rse = math.sqrt(q / df)
    se = math.sqrt(cov[0, 0]) * (max(1.0, rse) if used == "random" else 1.0)
    return _result(
        "ivw", beta[0], se, d.k, options, model=used,
        Q=q, Q_df=df, Q_pval=_p_chi2(q, df), I2=max(0.0, (q - df) / q) if q > 0 else 0.0,
        RSE=rse,
    )


def heterogeneity(data: Any) -> dict:
    """Cochran's Q of the fixed-effect IVW fit (weights 1/se_y^2), its df K - 1, chi-square
    p, and I2 = max(0, (Q - df)/Q); the key names TwoSampleMR::mr_heterogeneity uses."""
    d = _Inst(data, 2, "heterogeneity")
    res = ivw(d.frame, model="fixed")
    return {
        "method": "heterogeneity",
        "Q": res["Q"], "Q_df": res["Q_df"], "Q_pval": res["Q_pval"], "I2": res["I2"],
        "n_instruments": d.k,
        "options": {"weights": "first_order", "fit": "ivw fixed"},
    }


# --------------------------------------------------------------------------- Egger


def _i2_gx(bx: np.ndarray, sx: np.ndarray, sy: np.ndarray) -> float:
    """MendelianRandomization's I^2_GX: I2 of |beta_x| / se_y around their weighted mean,
    weights (se_x / se_y)^-2 — the exposure effects on the scale the Egger fit weights them."""
    bx, sx = np.abs(bx) / sy, sx / sy
    w = sx ** -2.0
    mu = np.sum(w * bx) / np.sum(w)
    q = float(np.sum(w * (bx - mu) ** 2))
    return max(0.0, (q - (bx.size - 1)) / q) if math.isfinite(q) else math.nan


def egger(data: Any, *, correl: Any = None) -> dict:
    """MR Egger; MendelianRandomization::mr_egger.

    Every variant is oriented to positive beta_x (beta_y flipped with it) and beta_y is
    regressed on beta_x with an intercept, weights 1/se_y^2. SEs are multiplicative random
    effects (times max(1, RSE)), p and CIs normal. `estimate` is the slope; the intercept
    (directional pleiotropy) is `intercept` with its own se, p and CI. `Q` has K - 2 df.
    `I2_GX` is the package's I^2 of the oriented beta_x — the regression-dilution check, low
    values (< 0.9) bias the slope towards zero. Needs 3 instruments.
    """
    frame = _frame_of(data)
    options: dict[str, Any] = {"orientation": "beta_x > 0", "weights": "first_order",
                               "model": "random", "correl": correl is not None}
    r = None
    if correl is not None:
        r, dropped = _correlation(correl, frame["variant"].cast(pl.String).to_list())
        if dropped:
            frame = _subset(frame, [v for v in frame["variant"].to_list() if v not in dropped])
            options["dropped_no_ld"] = dropped
    d = _Inst(frame, 3, "MR Egger")
    sign = np.where(d.bx < 0, -1.0, 1.0)
    bx, by = np.abs(d.bx), d.by * sign
    if r is None:
        omega_inv = np.diag(d.sy ** -2)
    else:
        r, info = _repair(r)
        options.update(info)
        omega_inv = np.linalg.inv(np.outer(d.sy, d.sy) * r * np.outer(sign, sign))
    x = np.column_stack([np.ones(d.k), bx])
    beta, cov, q = _gls(x, by, omega_inv)
    df = d.k - 2
    rse = math.sqrt(q / df)
    ses = np.sqrt(np.diag(cov)) * max(1.0, rse)
    out = _result(
        "egger", beta[1], ses[1], d.k, options,
        intercept=float(beta[0]), intercept_se=float(ses[0]),
        intercept_p=float(_p_norm(beta[0] / ses[0])),
        intercept_ci_low=float(beta[0] - _Z975 * ses[0]),
        intercept_ci_high=float(beta[0] + _Z975 * ses[0]),
        Q=q, Q_df=df, Q_pval=_p_chi2(q, df), RSE=rse,
        I2_GX=_i2_gx(bx, d.sx, d.sy),
    )
    return out


# --------------------------------------------------------------------------- median


def _weighted_median(b: np.ndarray, w: np.ndarray) -> np.ndarray:
    """MendelianRandomization's interpolated weighted median, row-wise over the last axis."""
    w = np.broadcast_to(w, b.shape)
    order = np.argsort(b, axis=-1, kind="stable")
    bs = np.take_along_axis(b, order, -1)
    ws = np.take_along_axis(w, order, -1)
    cs = np.cumsum(ws, -1) - 0.5 * ws
    cs = cs / ws.sum(-1, keepdims=True)
    below = (cs < 0.5).sum(-1, keepdims=True) - 1
    b0 = np.take_along_axis(bs, below, -1)
    b1 = np.take_along_axis(bs, below + 1, -1)
    c0 = np.take_along_axis(cs, below, -1)
    c1 = np.take_along_axis(cs, below + 1, -1)
    return (b0 + (b1 - b0) * (0.5 - c0) / (c1 - c0))[..., 0]


def _median(data: Any, weighting: str, weights: str, seed: int | None, iterations: int) -> dict:
    d = _Inst(data, 3, f"the {weighting} median")
    ratio = d.by / d.bx
    if weighting == "simple":
        w = np.ones(d.k)
    elif weights == "first_order":
        w = (d.sy / d.bx) ** -2
    elif weights == "second_order":
        w = 1 / (d.sy ** 2 / d.bx ** 2 + d.by ** 2 * d.sx ** 2 / d.bx ** 4)
    else:
        raise GeneticsUsageError("weights must be 'first_order' or 'second_order'")
    est = float(_weighted_median(ratio, w))
    rng = np.random.default_rng(seed)
    boot = rng.normal(d.by, d.sy, (iterations, d.k)) / rng.normal(d.bx, d.sx, (iterations, d.k))
    se = float(np.std(_weighted_median(boot, w), ddof=1))
    options = {"weighting": weighting, "iterations": iterations, "seed": seed,
               "bootstrap": "parametric, weights held at their observed values"}
    if weighting == "weighted":
        options["weights"] = weights
    return _result(f"{weighting}_median", est, se, d.k, options)


def weighted_median(
    data: Any, *, seed: int | None = _MR_SEED, iterations: int = _ITERATIONS,
    weights: str = "first_order",
) -> dict:
    """Weighted median of the Wald ratios; MendelianRandomization::mr_median.

    `weights="first_order"` (the package's) weights each ratio by beta_x^2 / se_y^2;
    `"second_order"` adds the exposure's error, 1/(se_y^2/bx^2 + by^2 se_x^2/bx^4), as
    TwoSampleMR's `mr_weighted_median` does. The SE is the SD of a seeded parametric
    bootstrap (beta_x and beta_y resampled from their normal errors, weights held fixed).
    Needs 3 instruments.
    """
    return _median(data, "weighted", weights, seed, iterations)


def simple_median(data: Any, *, seed: int | None = _MR_SEED, iterations: int = _ITERATIONS) -> dict:
    """Unweighted median of the Wald ratios; mr_median(weighting="simple"). Needs 3."""
    return _median(data, "simple", "first_order", seed, iterations)


# --------------------------------------------------------------------------- mode

_DENSITY_N = 512


def _mad(x: np.ndarray) -> np.ndarray:
    med = np.median(x, -1, keepdims=True)
    return 1.4826 * np.median(np.abs(x - med), -1)


def _density_mode(x: np.ndarray, w: np.ndarray, bw: np.ndarray) -> np.ndarray:
    """The grid point where R's `density(x, weights=w, bw=bw)` peaks, row-wise.

    A transcription of R's density.default (gaussian kernel, n = 512, cut = 3): linear
    binning onto 2n points over [from - 4 bw, to + 4 bw], FFT convolution with the kernel,
    linear interpolation onto the 512-point output grid, first maximum. The output grid is
    what makes the mode a grid value, so matching R's estimate needs this grid, not an exact
    kernel-density maximiser. Kernel ordinates are spaced at the binning step (R 4.4's
    corrected coordinates); the reference fixtures pick the same grid point under either
    spacing.

    Two equally weighted ratios put the true maximum exactly half-way between two grid
    points, so which of the two R reports is floating-point noise; this port may return
    the other one, one grid step away.
    """
    n = _DENSITY_N
    rows = x.shape[0]
    frm = x.min(1) - 3 * bw
    to = x.max(1) + 3 * bw
    lo = frm - 4 * bw
    up = to + 4 * bw
    delta = (up - lo) / (n - 1)
    xpos = (x - lo[:, None]) / delta[:, None]
    ix = np.floor(xpos).astype(np.int64)
    fx = xpos - ix
    w = np.broadcast_to(w, x.shape)
    y = np.zeros((rows, 2 * n))
    base = (np.arange(rows) * 2 * n)[:, None]
    np.add.at(y.reshape(-1), (base + ix).ravel(), (w * (1 - fx)).ravel())
    np.add.at(y.reshape(-1), (base + ix + 1).ravel(), (w * fx).ravel())
    kords = np.arange(2 * n)[None, :] * delta[:, None]
    kords[:, n + 1:] = -kords[:, n - 1:0:-1]
    kern = np.exp(-0.5 * (kords / bw[:, None]) ** 2) / (bw[:, None] * math.sqrt(2 * math.pi))
    dens = np.fft.ifft(np.fft.fft(y) * np.conj(np.fft.fft(kern))).real[:, :n]
    dens = np.maximum(dens, 0)
    grid = frm[:, None] + np.arange(n)[None, :] * ((to - frm) / (n - 1))[:, None]
    pos = (grid - lo[:, None]) / delta[:, None]
    j = np.clip(np.floor(pos).astype(np.int64), 0, n - 2)
    t = pos - j
    yi = np.take_along_axis(dens, j, 1) * (1 - t) + np.take_along_axis(dens, j + 1, 1) * t
    return np.take_along_axis(grid, np.argmax(yi, 1)[:, None], 1)[:, 0]


def _mbe(ratio: np.ndarray, se_ratio: np.ndarray, phi: float) -> np.ndarray:
    """mr_mbe's estimator, row-wise: bandwidth 0.9 min(sd, mad) / K^(1/5) times phi."""
    k = ratio.shape[-1]
    s = 0.9 * np.minimum(np.std(ratio, -1, ddof=1), _mad(ratio)) / k ** 0.2
    h = np.maximum(1e-8, s * phi)
    w = se_ratio ** -2.0
    return _density_mode(ratio, w / w.sum(), h)


def _mode(data: Any, weighting: str, phi: float, seed: int | None, iterations: int) -> dict:
    d = _Inst(data, 2, f"the {weighting} mode")
    ratio = d.by / d.bx
    # delta-method (second-order) SE of each ratio, mr_mbe's stderror = "delta"
    se_ratio = np.sqrt(d.sy ** 2 / d.bx ** 2 + d.by ** 2 * d.sx ** 2 / d.bx ** 4)
    wse = se_ratio if weighting == "weighted" else np.ones(d.k)
    est = float(_mbe(ratio[None, :], wse, phi)[0])
    rng = np.random.default_rng(seed)
    boots = []
    for start in range(0, iterations, 1000):
        m = min(1000, iterations - start)
        boots.append(_mbe(rng.normal(ratio, se_ratio, (m, d.k)), wse, phi))
    se = float(_mad(np.concatenate(boots)))
    options = {"weighting": weighting, "stderror": "delta", "phi": phi,
               "iterations": iterations, "seed": seed,
               "bootstrap": "parametric on the ratios; SE is the MAD of the bootstrap modes"}
    return _result(f"{weighting}_mode", est, se, d.k, options)


def weighted_mode(
    data: Any, *, phi: float = 1, seed: int | None = _MR_SEED, iterations: int = _ITERATIONS,
) -> dict:
    """Weighted mode-based estimate; MendelianRandomization::mr_mbe(weighting="weighted",
    stderror="delta").

    The mode of a Gaussian kernel density of the Wald ratios, each weighted by its inverse
    delta-method variance, bandwidth phi * 0.9 min(sd, mad) / K^(1/5), located on R's
    512-point density grid. SE is the MAD of a seeded parametric bootstrap of the ratios.
    The package's own bootstrap keeps the first of tied modes, so its SEs are
    implementation-dependent; this one's agree only loosely.
    """
    return _mode(data, "weighted", phi, seed, iterations)


def simple_mode(
    data: Any, *, phi: float = 1, seed: int | None = _MR_SEED, iterations: int = _ITERATIONS,
) -> dict:
    """Unweighted mode-based estimate; mr_mbe(weighting="unweighted")."""
    return _mode(data, "simple", phi, seed, iterations)


# --------------------------------------------------------------------------- TwoSampleMR


def _tsmr_ivw(bx: np.ndarray, by: np.ndarray, sy: np.ndarray) -> tuple[float, float, float]:
    """TwoSampleMR::mr_ivw: always multiplicative random effects."""
    if bx.size < 2:
        return math.nan, math.nan, math.nan
    w = sy ** -2
    sxx = np.sum(w * bx * bx)
    b = float(np.sum(w * bx * by) / sxx)
    sigma = math.sqrt(float(np.sum(w * (by - b * bx) ** 2)) / (bx.size - 1))
    se = max(1.0, sigma) / math.sqrt(sxx)
    return b, se, float(_p_norm(b / se))


def leave_one_out(data: Any) -> pl.DataFrame:
    """IVW leaving out each variant in turn, then the "All" row; TwoSampleMR::mr_leaveoneout.

    TwoSampleMR's IVW is multiplicative random effects at every K, so the "All" row equals
    `ivw(model="random")`, not `ivw()`'s K <= 3 fixed default. Needs 3 instruments, as the
    package's per-variant rows do; p is normal.
    """
    d = _Inst(data, 3, "leave-one-out")
    rows = []
    for i, v in enumerate(d.variant):
        keep = np.arange(d.k) != i
        rows.append((v, *_tsmr_ivw(d.bx[keep], d.by[keep], d.sy[keep]), d.k - 1))
    rows.append(("All", *_tsmr_ivw(d.bx, d.by, d.sy), d.k))
    return pl.DataFrame(rows, schema=["variant", "estimate", "se", "p", "n_instruments"],
                        orient="row")


def single_snp(data: Any) -> pl.DataFrame:
    """Per-variant Wald ratios, then "All - Inverse variance weighted" and "All - MR Egger";
    TwoSampleMR::mr_singlesnp.

    The IVW row is multiplicative random effects with a normal p; the Egger row follows
    TwoSampleMR (t distribution on K - 2 df for p, missing below 3 instruments), unlike
    `egger()`'s normal p.
    """
    d = _Inst(data)
    w = wald(d.frame).select("variant", "estimate", "se", "p")
    b, se, p = _tsmr_ivw(d.bx, d.by, d.sy)
    eb = ese = ep = math.nan
    if d.k >= 3:
        e = egger(d.frame)
        eb, ese = e["estimate"], e["se"]
        ep = float(_p_t(eb / ese, d.k - 2))
    tail = pl.DataFrame({
        "variant": ["All - Inverse variance weighted", "All - MR Egger"],
        "estimate": [b, eb], "se": [se, ese], "p": [p, ep],
    })
    return pl.concat([w, tail])


def _r_from_bsen(b: np.ndarray, se: np.ndarray, n: np.ndarray) -> np.ndarray:
    """TwoSampleMR::get_r_from_bsen: F = (b/se)^2, r^2 = F / (n - 2 + F), signed by b."""
    f = (b / se) ** 2
    return np.sqrt(f / (n - 2 + f)) * np.sign(b)


def _zapsmall(m: np.ndarray, digits: int = 7) -> np.ndarray:
    mx = np.max(np.abs(m))
    if mx <= 0:
        return m
    return np.round(m, max(0, digits - int(math.ceil(math.log10(mx)))))


def _population_af(af: float, prop: float, odds_ratio: float, prevalence: float) -> float:
    """TwoSampleMR::get_population_allele_frequency via its `contingency` table. The root is
    chosen on the zapsmall-rounded table, as the package does; the frequencies are read from
    the unrounded one, which is what reproduces its numbers."""
    a = odds_ratio - 1
    b = (af + prop) * (1 - odds_ratio) - 1
    c = odds_ratio * af * prop
    if abs(a) < 1e-15:
        roots = [-c / b]
    else:
        disc = b * b - 4 * a * c
        s = [0.0] if disc < 1e-30 else [-1.0, 1.0]
        roots = [(-b + si * math.sqrt(max(0.0, disc))) / (2 * a) for si in s]
    for z in roots:
        # R's matrix(c(z, prop - z, af - z, 1 + z - af - prop), 2, 2), column-major
        tab = np.array([[z, af - z], [prop - z, 1 + z - af - prop]])
        if (_zapsmall(tab) >= 0).all():
            af_controls = tab[0, 1] / (tab[0, 1] + tab[1, 1])
            af_cases = tab[0, 0] / (tab[0, 0] + tab[1, 0])
            return af_controls * (1 - prevalence) + af_cases * prevalence
    raise GeneticsUsageError("no valid case/control allele table for this log-OR and frequency")


def _r_from_lor(lor, af, ncase, ncontrol, prevalence) -> np.ndarray:
    """TwoSampleMR::get_r_from_lor (logit model, no correction): liability-scale r from the
    log-OR, the population allele frequency implied by the case fraction and prevalence,
    and residual variance pi^2/3."""
    ve = math.pi ** 2 / 3
    out = np.empty(len(lor))
    for i, (b, f, nca, nco) in enumerate(zip(lor, af, ncase, ncontrol)):
        popaf = _population_af(f, nca / (nca + nco), math.exp(b), prevalence)
        vg = b * b * popaf * (1 - popaf)
        out[i] = math.sqrt(vg / (vg + ve)) * np.sign(b)
    return out


def _r_test(r1, r2, n1, n2):
    """psych::r.test for two independent correlations (Fisher z), two-sided normal p."""
    z = (np.arctanh(r1) - np.arctanh(r2)) / np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    return _p_norm(z)


def _binary_outcome(data: Any, frame: pl.DataFrame) -> bool:
    units = getattr(data, "units", None)
    if isinstance(units, dict) and "outcome" in units:
        return "binary" in str(units["outcome"])
    return "n_cases" in frame.columns and frame["n_cases"].drop_nulls().len() > 0


def steiger(data: Any, *, prevalence: float | None = None, exposure_r2: Any = None) -> dict:
    """Steiger directionality test; TwoSampleMR::mr_steiger with its r conventions.

    Exposure r per variant from beta/se/n (get_r_from_bsen: F = (b/se)^2,
    r^2 = F / (n_x - 2 + F)) unless `exposure_r2` gives per-variant r^2. Outcome r the same
    way for a continuous outcome; for a binary one (n_cases present, or harmonise() units
    saying so) get_r_from_lor's liability-scale r from the log-OR, `af_y` (else `af_x`), the
    row's n_cases/n_controls and `prevalence`, which is required: the outcome's prevalence
    is not in the data and a silent default would move the liability scale. The test compares sqrt(sum r_x^2) with
    sqrt(sum r_y^2) by Fisher z at the mean sample sizes; `correct_causal_direction` is
    r2_exposure > r2_outcome.

    TwoSampleMR's `directionality_test` and `steiger_filtering` reach r through the
    p-value instead (get_r_from_pn), which loses the strongest instruments where p
    underflows; this follows mr_steiger's direct route. `per_variant` carries each
    variant's r^2 and its own Fisher-z p.
    """
    d = _Inst(data)
    frame = d.frame
    binary = _binary_outcome(data, frame)
    for need in ("n_x", "n_cases", "n_controls") if binary else ("n_x", "n_y"):
        if need not in frame.columns:
            raise GeneticsUsageError(f"Steiger needs `{need}`, the sample size per instrument")
    options: dict[str, Any] = {"exposure_r": "get_r_from_bsen" if exposure_r2 is None else "given"}
    n_x = _col(frame, "n_x")
    if exposure_r2 is None:
        r_x = np.abs(_r_from_bsen(d.bx, d.sx, n_x))
    else:
        r_x = np.sqrt(np.broadcast_to(np.asarray(exposure_r2, dtype=float), (d.k,)))
    if binary:
        ncase, nctrl = _col(frame, "n_cases"), _col(frame, "n_controls")
        if prevalence is None:
            raise GeneticsUsageError(
                "Steiger on a binary outcome needs the outcome's `prevalence` for the log-OR "
                "route; run() supplies the sample prevalence n_cases/(n_cases+n_controls), "
                "with a note, when the caller gives none"
            )
        if "af_y" in frame.columns and frame["af_y"].null_count() == 0:
            af_name = "af_y"
        elif "af_x" in frame.columns:
            af_name = "af_x"
        else:
            raise GeneticsUsageError("Steiger on a binary outcome needs `af_y` or `af_x`")
        r_y = np.abs(_r_from_lor(d.by, _col(frame, af_name), ncase, nctrl, prevalence))
        n_y = (_col(frame, "n_y") if "n_y" in frame.columns and frame["n_y"].null_count() == 0
               else ncase + nctrl)
        options.update(outcome_r="get_r_from_lor (logit)", prevalence=prevalence,
                       outcome_af=af_name)
        options["prevalence_source"] = "given"
    else:
        n_y = _col(frame, "n_y")
        r_y = np.abs(_r_from_bsen(d.by, d.sy, n_y))
        options["outcome_r"] = "get_r_from_bsen"
    r2x, r2y = float(np.sum(r_x ** 2)), float(np.sum(r_y ** 2))
    p = float(_r_test(math.sqrt(r2x), math.sqrt(r2y), np.mean(n_x), np.mean(n_y)))
    per_variant = pl.DataFrame({
        "variant": d.variant,
        "r2_exposure": r_x ** 2,
        "r2_outcome": r_y ** 2,
        "correct_causal_direction": r_x ** 2 > r_y ** 2,
        "steiger_pval": _r_test(r_x, r_y, n_x, n_y),
    })
    return {
        "method": "steiger",
        "correct_causal_direction": r2x > r2y,
        "steiger_pval": p,
        "r2_exposure": r2x,
        "r2_outcome": r2y,
        "n_exposure": float(np.mean(n_x)),
        "n_outcome": float(np.mean(n_y)),
        "n_instruments": d.k,
        "per_variant": per_variant,
        "options": options,
    }
