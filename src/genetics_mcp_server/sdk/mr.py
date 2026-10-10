"""Mendelian randomization: instruments, harmonisation, estimates and checks in one call.

    import genetics
    res = genetics.mr.run({"dataset": "UKB_PPP", "trait": "IL12B"}, ("ibd_gwas", "IBD"))
    res.summary()      # one row: headline method, estimate, se, p, CI, K, overlap, failing checks
    res.estimates      # method -> estimator dict; res.sensitivity, res.checks, res.parameters

    batch = genetics.mr.run(exposure, outcomes=[("finngen", "K11_IBD_STRICT"), ...])
    batch.summary()    # one row per outcome; batch.failed names the outcomes that could not run

The pieces `run()` composes are public too:

    h = genetics.mr.harmonise(exposure_rows, outcome_rows,
                              exposure_dataset=ukb_ppp, outcome_dataset=iibdgc_ibd)
    h.frame        # one row per instrument, both sides' effects and the per-instrument checks
    h.units        # {"exposure": "SD (quantitative)", "outcome": "log OR (binary)"}
    genetics.mr.checks(h, steiger=..., heterogeneity=...)   # check, status, detail

Pure functions over polars frames: the caller (or `instruments()` / `run()`) fetches the
rows, these functions join and judge them. The only I/O is the allele-frequency reference
lookup in `harmonise`, and it goes through an injectable `annotation` callable.

ALLELES. Both sides are GRCh38 `chr:pos:ref:alt` with beta on alt, so the join is exact and
there is no palindrome inference. An outcome row found only under the swapped alleles is
never flipped: with both sides reference-normalised it is either a different variant
(1:100:A:AT is an insertion, 1:100:AT:A a deletion) or a store that was not normalised, and
flipping would guess which allele its beta is on. Such an instrument is `outcome_match`
"swapped_not_used", `in_outcome` False with null outcome columns, so a proxy step treats it
as absent and the count stays visible.

THE ALLELE-FREQUENCY CHECK (the one test that catches a mismatched effect allele) compares
the exposure's own frequency with a reference picked in the ruled order, and
`af_checked_against` names the reference:

    exposure AF   outcome row af   reference                        af_checked_against
    alt AF        present          the outcome row's af             outcome
    MAF           present          MAF of the outcome row's af      outcome
    any           absent           FinnGen annotation AF            finngen_annotation
    any           absent           gnomAD AF_nfe (AF_fin if NA),    gnomad_nfe / gnomad_fin
                                   non-Finnish exposure only, when
                                   the FinnGen annotation lacks it
    none          any              nothing to compare               none

gnomAD is never consulted for a Finnish exposure (one whose dataset lists the `finngen`
cohort): the FinnGen annotation is its reference or there is none.

The exposure frequency's scale (`af_scale`) comes from what is known about its source, never
from its values: a `maf` column is a minor-allele frequency, an `af`/`aaf` column the alt
allele's, except for a dataset in `_MAF_UNDER_ALT_NAME`. On the MAF scale the reference is
folded to a MAF too, so the comparison catches a wrong variant but cannot see a swapped
allele at all, and `checks()` never passes it for orientation.

`orientation_unverifiable` marks every instrument whose MAF exceeds (1 - af_tolerance) / 2:
its two alleles' frequencies differ by less than the tolerance, so a mislabelled allele
passes the frequency check. With no frequency on either side it falls back to
`palindromic`; otherwise `palindromic` is information only.

INSTRUMENTS. `instruments(exposure, ...)` picks the leads harmonise() takes, as a function of
(credible-set rows, LD accessor) so a batch job could reuse it with a file-backed accessor. The
LD scans run at the panel's floor rather than at `r2_prune`: the kept leads are by definition
below `r2_prune` with each other, so a scan at that threshold could only ever return an
identity r matrix. At instruments()' default `r2_prune` (equal to finngen_r12's floor of
0.01) the matrix is the identity anyway; it carries information only with a looser prune,
which is why run()'s cis default is 0.1. Pseudo credible sets
are refused as exposures by default: their satellite and singleton sets put leads tens of kb
from the real signal (deCODE against UKB_PPP, 2026-10-10).

RUN. `run()` is instruments() -> one batched summary_stats per chunk of outcomes (leads and
their r2 >= 0.8 proxies together) -> harmonise() -> estimates -> sensitivity -> checks(). One
instrument gives the Wald ratio. In cis mode with more than one, the headline is the IVW with
the instruments' signed LD r (`ivw_correlated`), the independent IVW beside it, MR Egger from 3;
cis prunes at r2 0.1 by default so that r carries information. Genome-wide prunes at r2 0.01
and reports the full method set (IVW, Egger, weighted and simple median and mode) with the IVW
as headline. Leave-one-out and single-SNP follow TwoSampleMR and so ignore the LD r.

WHAT `run()` DOES NOT DO, and why:
- multivariable MR: it needs instruments for several exposures harmonised jointly, and the
  credible-set lead rule picks instruments per trait;
- CAUSE, MR-APSS and other genome-wide-summary methods: they read every variant's summary
  statistics, beyond the request budget of one execution;
- colocalization on demand: it needs full regional summary statistics on both sides; only the
  precomputed colocalization rows are attached, when the two datasets are coloc partners;
- many-exposure (proteome-wide) screens: a scope decision for v1, not a budget limit, so a
  list of exposures is refused. The lead-only Wald form of a screen fits one execution; the
  proving-ground harness carries it.

SAMPLE OVERLAP. A dataset's `cohorts` token names a sample collection, not a sample set. A
shared token says only "same collection(s), overlap fraction not recorded"; it never
establishes one-sample MR, and disjoint lists never establish independent samples, because
consortium tokens may contain FinnGen or UK Biobank individuals. The statement is a warning
at most, never a failure.
"""

from __future__ import annotations

import asyncio
import inspect
import math
import re
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import polars as pl

# every estimator is re-exported as genetics.mr.<name>
from genetics_mcp_server.sdk._mr_estimators import (
    __all__ as _ESTIMATORS,
)
from genetics_mcp_server.sdk._mr_estimators import (
    egger,
    heterogeneity,
    ivw,
    leave_one_out,
    simple_median,
    simple_mode,
    single_snp,
    steiger,
    weighted_median,
    weighted_mode,
)
from genetics_mcp_server.sdk._mr_estimators import (
    wald as wald,
)
from genetics_mcp_server.sdk.errors import GeneticsError, GeneticsUsageError
from genetics_mcp_server.sdk.plots import _CODING_CONSEQUENCES

# run first: the catalogue's call-path example and the stub both follow this order
__all__ = ["run", "instruments", "harmonise", "checks", "sample_overlap_statement",
           "MRResult", "MRBatch", "Harmonised", "Instruments", *_ESTIMATORS]

AF_TOLERANCE = 0.2
F_WEAK = 10.0

# exposure datasets (the `dataset` value) whose loaded credible sets carry a minor-allele
# frequency under the alt-allele name: UKB_PPP's loaded `aaf` is a MAF. The dataset
# registry is the better home for this fact; it lives here until the registry records it.
_MAF_UNDER_ALT_NAME = frozenset({"UKB_PPP"})

_MOLECULAR_QTL = frozenset({"pqtl", "eqtl", "sqtl", "metaboqtl", "caqtl", "asmqtl"})

AnnotationFn = Callable[..., pl.DataFrame]


@dataclass
class Harmonised:
    """`harmonise`'s result: the instrument frame plus what a frame cannot carry.

    `units` names the effect scale of each side; `dropped` lists exposure rows left out and
    why; `notes` states every assumption the call made (Finnish-ness of the exposure, the
    MAF reading of its frequency column).
    """

    frame: pl.DataFrame
    units: dict[str, str]
    dropped: pl.DataFrame
    notes: list[str] = field(default_factory=list)
    exposure_dataset: dict[str, Any] | None = None
    outcome_dataset: dict[str, Any] | None = None


# --------------------------------------------------------------------------- inputs


def _num(frame: pl.DataFrame, name: str) -> pl.Expr:
    """A float column; results-api serves every value as a string with `NA` for missing."""
    if name not in frame.columns:
        return pl.lit(None, dtype=pl.Float64)
    col = pl.col(name)
    if frame.schema[name] == pl.String:
        col = pl.when(col.is_in(["NA", "nan", ""])).then(None).otherwise(col)
    return col.cast(pl.Float64, strict=False)


def _first(frame: pl.DataFrame, *names: str) -> str | None:
    return next((n for n in names if n in frame.columns), None)


def _first_populated(frame: pl.DataFrame, *names: str) -> str | None:
    """The first of `names` holding at least one value: an all-NA `aaf` must not hide `maf`."""
    return next(
        (n for n in names
         if n in frame.columns and frame.select(_num(frame, n)).to_series().drop_nulls().len()),
        None,
    )


def _variant_key(frame: pl.DataFrame) -> pl.Expr:
    """`chr:pos:ref:alt` with the chromosome bare and X as 23, as the results views hold it."""
    if all(c in frame.columns for c in ("chr", "pos", "ref", "alt")):
        parts = [pl.col("chr").cast(pl.String), pl.col("pos").cast(pl.String),
                 pl.col("ref").cast(pl.String), pl.col("alt").cast(pl.String)]
        raw = pl.concat_str(parts, separator=":")
    elif "variant" in frame.columns:
        raw = pl.col("variant").cast(pl.String).str.replace_all("_", ":")
    else:
        raise GeneticsUsageError(
            "a frame needs a `variant` column or `chr`, `pos`, `ref` and `alt` columns"
        )
    return (
        raw.str.replace(r"^(?i:chr)", "")
        .str.replace(r"^X:", "23:")
        .str.replace(r"^Y:", "24:")
    )


def _n_expr(frame: pl.DataFrame) -> pl.Expr:
    name = _first(frame, "n", "n_samples")
    if name:
        return _num(frame, name)
    return _num(frame, "n_cases") + _num(frame, "n_controls")


def _is_finnish(dataset: dict[str, Any] | None) -> bool:
    return bool(dataset) and "finngen" in (dataset.get("cohorts") or [])


def _unit(dataset: dict[str, Any] | None, binary_rows: bool) -> str:
    if not dataset:
        return "unknown"
    data_type = (dataset.get("data_type") or "").lower()
    trait_type = (dataset.get("trait_type") or "").lower()
    if data_type in _MOLECULAR_QTL:
        return "SD (normalised molecular level)"
    if trait_type == "binary" or (trait_type == "mixed" and binary_rows):
        return "log OR (binary)"
    if trait_type == "quantitative":
        return "SD (quantitative)"
    return "unknown"


def _default_annotation(variants: list[str], source: str) -> pl.DataFrame:
    import genetics_mcp_server.sdk as sdk

    return sdk.variant_annotation(variants=variants, source=source)


def _annotation_af(
    annotation: AnnotationFn, variants: list[str], source: str, columns: tuple[str, ...]
) -> dict[str, tuple[float, str]]:
    """variant -> (frequency, column) for the first of `columns` that is not NA."""
    if not variants:
        return {}
    rows = annotation(variants=variants, source=source)
    if rows is None or rows.is_empty():
        return {}
    rows = rows.with_columns(_variant_key(rows).alias("_key"))
    out: dict[str, tuple[float, str]] = {}
    for col in columns:
        if col not in rows.columns:
            continue
        for key, value in rows.select("_key", _num(rows, col).alias("_v")).iter_rows():
            if value is not None and not math.isnan(value) and key not in out:
                out[key] = (value, col)
    return out


# --------------------------------------------------------------------------- harmonise


def harmonise(
    exposure: pl.DataFrame,
    outcome: pl.DataFrame,
    *,
    exposure_dataset: dict[str, Any] | None = None,
    outcome_dataset: dict[str, Any] | None = None,
    annotation: AnnotationFn | None = None,
    af_tolerance: float = AF_TOLERANCE,
) -> Harmonised:
    """Join exposure instruments to one outcome's rows and check each instrument's alleles.

    `exposure` is one row per instrument (credible-set leads, or a frame of the caller's):
    `beta`, `se` and a variant key are required; `af`/`aaf`/`maf`, `pip`, `cs_id`,
    `most_severe` and `n`/`n_samples`/`n_cases`+`n_controls` are used when present.
    `outcome` is one phenotype's summary-statistic rows (`beta`, `se`, `af`, `n_cases`,
    `n_controls`, ...). `*_dataset` are rows of `genetics.datasets()`; they give the units,
    the exposure's Finnish-ness for the AF reference, and the overlap statement in `checks`.
    `annotation(variants=, source=)` defaults to `genetics.variant_annotation` and is called
    only for instruments whose outcome row has no frequency.

    Returns a `Harmonised`; `.frame` has one row per kept instrument, instruments absent
    from the outcome included (`in_outcome` False, outcome columns null) for a proxy step.
    """
    annotation = annotation or _default_annotation
    notes: list[str] = []
    if exposure_dataset is None:
        notes.append("exposure dataset not given: treated as non-Finnish for the AF reference")
    elif "cohorts" not in exposure_dataset:
        notes.append("exposure dataset records no cohorts: treated as non-Finnish")
    finnish = _is_finnish(exposure_dataset)

    if "phenotype" in outcome.columns and outcome["phenotype"].n_unique() > 1:
        raise GeneticsUsageError(
            "harmonise() takes one outcome phenotype; got "
            f"{outcome['phenotype'].n_unique()} — call it once per phenotype"
        )

    af_col = _first_populated(exposure, "af", "aaf", "maf")
    x = exposure.select(
        _variant_key(exposure).alias("variant"),
        _num(exposure, "beta").alias("beta_x"),
        _num(exposure, "se").alias("se_x"),
        (_num(exposure, af_col) if af_col else pl.lit(None, dtype=pl.Float64)).alias("af_x"),
        _n_expr(exposure).alias("n_x"),
        _num(exposure, "pip").alias("pip"),
        (pl.col("cs_id").cast(pl.String) if "cs_id" in exposure.columns
         else pl.lit(None, dtype=pl.String)).alias("cs_id"),
        (pl.col("most_severe").cast(pl.String) if "most_severe" in exposure.columns
         else pl.lit(None, dtype=pl.String)).alias("most_severe"),
    )
    if exposure_dataset and exposure_dataset.get("n_samples") is not None:
        x = x.with_columns(pl.col("n_x").fill_null(float(exposure_dataset["n_samples"])))
    duplicated = x.filter(pl.col("variant").is_duplicated())["variant"].unique().to_list()
    if duplicated:
        raise GeneticsUsageError(
            f"exposure lists {len(duplicated)} variant(s) more than once ({duplicated[:3]}); "
            "pick one row per instrument before harmonising"
        )

    unusable = pl.col("beta_x").is_null() | pl.col("se_x").is_null() | (pl.col("se_x") <= 0)
    dropped = x.filter(unusable).select(
        "variant", pl.lit("exposure beta or se missing or se <= 0").alias("reason")
    )
    x = x.filter(~unusable)

    af_scale = "alt"
    if af_col == "maf":
        af_scale = "maf"
    elif af_col and (exposure_dataset or {}).get("dataset") in _MAF_UNDER_ALT_NAME:
        af_scale = "maf"
        notes.append(
            f"exposure `{af_col}` of {exposure_dataset['dataset']} is a minor-allele frequency: "
            "compared on the MAF scale, which cannot see a swapped allele"
        )

    parts = x.select(
        "variant",
        *(pl.col("variant").str.split(":").list.get(i).alias(n)
          for i, n in enumerate(("chr", "pos", "ref", "alt"))),
    ).with_columns(pl.col("chr").cast(pl.Int64, strict=False), pl.col("pos").cast(pl.Int64))
    x = x.join(parts, on="variant", how="left")
    x = x.with_columns(
        pl.concat_str("ref", "alt").is_in(["AT", "TA", "CG", "GC"]).alias("palindromic"),
        (pl.col("chr").cast(pl.String) + ":" + pl.col("pos").cast(pl.String) + ":"
         + pl.col("alt") + ":" + pl.col("ref")).alias("_swapped"),
    )

    y = outcome.select(
        _variant_key(outcome).alias("_key"),
        _num(outcome, "beta").alias("beta_y"),
        _num(outcome, "se").alias("se_y"),
        _num(outcome, _first(outcome, "af", "af_alt") or "af").alias("af_y"),
        _n_expr(outcome).alias("n_y"),
        _num(outcome, "n_cases").alias("n_cases"),
        _num(outcome, "n_controls").alias("n_controls"),
    )
    if outcome_dataset and outcome_dataset.get("n_samples") is not None:
        y = y.with_columns(pl.col("n_y").fill_null(float(outcome_dataset["n_samples"])))
    if y["_key"].is_duplicated().any():
        raise GeneticsUsageError("outcome lists a variant more than once; pass one phenotype")
    y = y.filter(pl.col("beta_y").is_not_null() & pl.col("se_y").is_not_null())

    merged = x.join(y, left_on="variant", right_on="_key", how="left")
    merged = merged.with_columns(
        pl.when(pl.col("beta_y").is_not_null()).then(pl.lit("exact"))
        .when(pl.col("_swapped").is_in(y["_key"].implode())).then(pl.lit("swapped_not_used"))
        .otherwise(pl.lit("absent")).alias("outcome_match"),
    ).drop("_swapped")
    merged = merged.with_columns((pl.col("outcome_match") == "exact").alias("in_outcome"))

    merged = _af_check(merged, annotation, finnish, af_scale, af_tolerance)
    maf_ref = pl.coalesce("af_y", "af_reference", "af_x")
    merged = merged.with_columns(
        pl.lit(af_scale).alias("af_scale"),
        (pl.min_horizontal(maf_ref, 1 - maf_ref) > (1 - af_tolerance) / 2)
        .fill_null(pl.col("palindromic")).alias("orientation_unverifiable"),
        pl.when(pl.col("most_severe").is_null()).then(None)
        .otherwise(pl.col("most_severe").is_in(list(_CODING_CONSEQUENCES))).alias("coding"),
    )
    columns = [
        "variant", "chr", "pos", "beta_x", "se_x", "af_x", "n_x", "pip", "cs_id",
        "beta_y", "se_y", "af_y", "n_y", "n_cases", "n_controls", "most_severe",
        "in_outcome", "outcome_match", "af_checked_against", "af_reference", "af_scale",
        "af_diff", "af_flag", "palindromic", "orientation_unverifiable", "coding",
    ]
    binary_rows = merged["n_cases"].drop_nulls().len() > 0
    units = {
        "exposure": _unit(exposure_dataset, False),
        "outcome": _unit(outcome_dataset, binary_rows),
    }
    return Harmonised(
        frame=merged.select(columns),
        units=units,
        dropped=dropped,
        notes=notes,
        exposure_dataset=exposure_dataset,
        outcome_dataset=outcome_dataset,
    )


def _af_check(
    frame: pl.DataFrame,
    annotation: AnnotationFn,
    finnish: bool,
    af_scale: str,
    tolerance: float,
) -> pl.DataFrame:
    has_x = frame["af_x"].is_not_null().to_list()
    has_y = frame["af_y"].is_not_null().to_list()
    variants = frame["variant"].to_list()
    need = [v for v, hx, hy in zip(variants, has_x, has_y) if hx and not hy]

    fg = _annotation_af(annotation, need, "finngen", ("AF",))
    gn: dict[str, tuple[float, str]] = {}
    if not finnish:
        gn = _annotation_af(
            annotation, [v for v in need if v not in fg], "gnomad", ("AF_nfe", "AF_fin")
        )

    labels, refs = [], []
    for v, hx, hy, af_y in zip(variants, has_x, has_y, frame["af_y"].to_list()):
        if not hx:
            labels.append("none")
            refs.append(None)
        elif hy:
            labels.append("outcome")
            refs.append(af_y)
        elif v in fg:
            labels.append("finngen_annotation")
            refs.append(fg[v][0])
        elif v in gn:
            labels.append("gnomad_nfe" if gn[v][1] == "AF_nfe" else "gnomad_fin")
            refs.append(gn[v][0])
        else:
            labels.append("none")
            refs.append(None)

    frame = frame.with_columns(
        pl.Series("af_checked_against", labels, dtype=pl.String),
        pl.Series("af_reference", refs, dtype=pl.Float64),
    )
    checked = pl.col("af_checked_against") != "none"
    if af_scale == "maf":
        ref = pl.min_horizontal(pl.col("af_reference"), 1 - pl.col("af_reference"))
        diff = pl.col("af_x") - ref
    else:
        diff = pl.col("af_x") - pl.col("af_reference")
    return frame.with_columns(
        pl.when(checked).then(diff).otherwise(None).alias("af_diff")
    ).with_columns(
        pl.when(checked).then(pl.col("af_diff").abs() > tolerance).otherwise(None)
        .alias("af_flag")
    )


# --------------------------------------------------------------------------- checks


def _row(check: str, status: str, detail: str) -> dict[str, str]:
    return {"check": check, "status": status, "detail": detail}


def _strength(frame: pl.DataFrame) -> dict[str, str]:
    f = frame.select(((pl.col("beta_x") / pl.col("se_x")) ** 2).alias("f"))["f"].drop_nulls()
    if f.is_empty():
        return _row("instrument_strength", "not_computable", "no instrument with beta_x and se_x")
    weak = int((f < F_WEAK).sum())
    mean = float(f.mean())
    status = "fail" if mean < F_WEAK else "warn" if weak else "pass"
    return _row("instrument_strength", status,
                f"mean F {mean:.1f} over {f.len()} instruments; {weak} below {F_WEAK:g}")


def _af_concordance(frame: pl.DataFrame) -> dict[str, str]:
    if frame.is_empty():
        return _row("af_concordance", "not_computable", "no instruments")
    flagged = int(frame["af_flag"].fill_null(False).sum())
    unverifiable = int(frame["orientation_unverifiable"].fill_null(False).sum())
    unchecked = int((frame["af_checked_against"] == "none").sum())
    maf_scale = bool((frame["af_scale"] == "maf").any())
    status = "fail" if flagged else "warn" if unverifiable or unchecked or maf_scale else "pass"
    detail = (f"{flagged} flagged, {unverifiable} orientation unverifiable, "
              f"{unchecked} unchecked of {frame.height}")
    if maf_scale:
        detail += ("; the exposure frequency is a MAF, so it can only catch a wrong variant, "
                   "not a swapped allele")
    return _row("af_concordance", status, detail)


def _pleiotropy(frame: pl.DataFrame) -> dict[str, str]:
    known = frame["coding"].drop_nulls()
    if known.is_empty():
        return _row("pleiotropy_proxy", "not_computable", "no most_severe annotation")
    coding = int(known.sum())
    status = "warn" if coding else "pass"
    return _row("pleiotropy_proxy", status,
                f"{coding} of {known.len()} annotated instruments are coding "
                f"({coding / known.len():.0%}); a non-coding-only run is the sensitivity check")


def _steiger(result: dict[str, Any] | None) -> dict[str, str]:
    if result is None:
        return _row("steiger", "not_computable", "estimator result not supplied")
    direction = result.get("correct_causal_direction")
    p = result.get("steiger_pval")
    if direction is None:
        return _row("steiger", "not_computable", "estimator result has no direction")
    p_text = f", p {p:.2g}" if p is not None else ""
    if not direction:
        return _row("steiger", "fail", f"variance explained in the outcome exceeds the exposure{p_text}")
    status = "pass" if p is not None and p < 0.05 else "warn"
    return _row("steiger", status, f"exposure-to-outcome direction{p_text}")


def _heterogeneity(result: dict[str, Any] | None) -> dict[str, str]:
    if result is None:
        return _row("heterogeneity", "not_computable", "estimator result not supplied")
    q, df, p = result.get("Q"), result.get("Q_df"), result.get("Q_pval")
    if q is None or p is None:
        return _row("heterogeneity", "not_computable", "estimator result has no Q or Q_pval")
    df_text = f" on {df} df" if df is not None else ""
    return _row("heterogeneity", "warn" if p < 0.05 else "pass",
                f"Cochran's Q {q:.2f}{df_text}, p {p:.2g}")


def sample_overlap_statement(
    exposure_dataset: dict[str, Any] | None, outcome_dataset: dict[str, Any] | None
) -> tuple[str, str]:
    """(status, detail) for the two datasets' `cohorts` lists, under the ruled wording."""
    x = (exposure_dataset or {}).get("cohorts")
    y = (outcome_dataset or {}).get("cohorts")
    if not x or not y:
        side = "exposure" if not x else "outcome"
        return "not_computable", f"{side} dataset's cohorts are not recorded"
    shared = sorted(set(x) & set(y))
    if shared:
        return "warn", (
            f"same collection(s) {', '.join(shared)}: overlap fraction not recorded, "
            "treat as overlapping"
        )
    return "pass", (
        f"no collection in common ({', '.join(x)} vs {', '.join(y)}); disjoint lists do not "
        "establish independent samples"
    )


def checks(
    frame: Harmonised | pl.DataFrame,
    *,
    exposure_dataset: dict[str, Any] | None = None,
    outcome_dataset: dict[str, Any] | None = None,
    steiger: dict[str, Any] | None = None,
    heterogeneity: dict[str, Any] | None = None,
) -> pl.DataFrame:
    """The STROBE-MR checks as a frame of check, status (pass/warn/fail/not_computable), detail.

    Takes `harmonise`'s result (its datasets are used unless given here) or its `.frame`.
    `steiger` and `heterogeneity` are the estimator module's result dicts, read by the
    TwoSampleMR names (`correct_causal_direction`, `steiger_pval`; `Q`, `Q_df`, `Q_pval`).
    Strength, pleiotropy and AF concordance count instruments present in the outcome only.
    """
    if isinstance(frame, Harmonised):
        exposure_dataset = exposure_dataset or frame.exposure_dataset
        outcome_dataset = outcome_dataset or frame.outcome_dataset
        frame = frame.frame
    used = frame.filter(pl.col("in_outcome"))
    overlap = sample_overlap_statement(exposure_dataset, outcome_dataset)
    rows = [
        _strength(used),
        _af_concordance(used),
        _pleiotropy(used),
        _steiger(steiger),
        _heterogeneity(heterogeneity),
        _row("sample_overlap", *overlap),
    ]
    return pl.DataFrame(rows, schema={"check": pl.String, "status": pl.String, "detail": pl.String})


# --------------------------------------------------------------------------- instruments

MHC = (6, 25_000_000, 34_000_000)
LD_SCAN_CAP = 3_000_000
PROXY_R2 = 0.8
# results-api refuses a larger variant_annotation batch with a 422
_ANNOTATION_BATCH = 2000
_CS_ID_VARIANT = re.compile(r"^chr([0-9XY]+)_([0-9]+)_([A-Za-z]+)_([A-Za-z]+)$")
_SET_KEY = ("dataset", "trait", "trait_original", "cell_type", "cs_id")
_LEAD_COLUMNS = [
    "variant", "chr", "pos", "ref", "alt", "beta", "se", "pip", "cs_id", "mlog10p", "maf",
    "af_source", "trait", "trait_original", "cell_type", "dataset",
]
_CS_SQL_COLUMNS = (
    "dataset, resource, trait, trait_original, cell_type, cs_id, variant, chr, pos, ref, alt, "
    "beta, se, pip, mlog10p, maf, aaf"
)
_CS_MAX_ROWS = 100_000


@dataclass
class Instruments:
    """`instruments`' result.

    `frame`: one row per kept instrument. A proxied row describes the proxy's own alleles:
    its variant, chr, pos, ref and alt, beta = sign(proxy_r) * the lead's beta (se, pip and
    mlog10p are the lead's), maf null with af_source "proxy: no frequency", and the lead's
    values in `lead_beta`, `lead_maf`, `proxy_for`, `proxy_r`, `proxy_r2`; `in_outcome` is
    null when no outcome was given. `dropped`: variant, cs_id,
    reason for every set or lead left out. `r`: the signed-r matrix of the kept leads (by
    lead id, which is `proxy_for` on a proxied row), a `variant` column plus one column per
    lead in the same order, diagonal 1, pairs below the panel floor 0. `panel`: name,
    r2_floor, max_distance, the scan window and threshold used. `requests`: calls per kind
    and their total. `notes`: every assumption the call made.
    """

    frame: pl.DataFrame
    dropped: pl.DataFrame
    r: pl.DataFrame
    panel: dict[str, Any]
    requests: dict[str, int]
    notes: list[str] = field(default_factory=list)
    exposure_dataset: dict[str, Any] | None = None
    cis_region: dict[str, Any] | None = None


class _Counted:
    def __init__(self, fn: Callable[..., Any], counts: dict[str, int], kind: str) -> None:
        self.fn, self.counts, self.kind = fn, counts, kind
        counts.setdefault(kind, 0)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.counts[self.kind] += 1
        return self.fn(*args, **kwargs)


def _sdk(name: str) -> Callable[..., Any]:
    """The SDK function `name`, looked up at call time so a monkeypatch reaches it."""
    def call(*args: Any, **kwargs: Any) -> Any:
        import genetics_mcp_server.sdk as sdk

        return getattr(sdk, name)(*args, **kwargs)
    return call


def _quote(value: str) -> str:
    return "'" + str(value).replace("\\", "\\\\").replace("'", "\\'") + "'"


def _dataset_row(rows: list[dict[str, Any]], dataset: str | None,
                 resource: str | None) -> dict[str, Any] | None:
    def names(row: dict[str, Any]) -> list[str]:
        value = row.get("dataset")
        return value if isinstance(value, list) else [value]
    if dataset is None:
        matches = [r for r in rows if resource and r.get("resource") == resource]
    else:
        matches = [r for r in rows if dataset in names(r)]
    if resource and dataset is not None:
        matches = [r for r in matches if r.get("resource") == resource] or matches
    if not matches:
        return None
    merged = dict(matches[0])
    # a dataset under two registry entries is pseudo if either says so: refusing is the
    # cheap mistake, a satellite-set lead is the expensive one
    merged["pseudo_credible_sets"] = any(m.get("pseudo_credible_sets") for m in matches)
    return merged


def _normalise_sets(frame: pl.DataFrame) -> pl.DataFrame:
    key = _variant_key(frame)
    parts = key.str.split(":")
    text = {c: (pl.col(c).cast(pl.String) if c in frame.columns else pl.lit(None, pl.String))
            for c in ("dataset", "resource", "trait", "trait_original", "cell_type", "cs_id")}
    return frame.select(
        key.alias("variant"),
        parts.list.get(0).cast(pl.Int64, strict=False).alias("chr"),
        parts.list.get(1).cast(pl.Int64, strict=False).alias("pos"),
        parts.list.get(2).alias("ref"),
        parts.list.get(3).alias("alt"),
        *(_num(frame, c).alias(c) for c in ("beta", "se", "pip", "mlog10p", "maf", "aaf")),
        *(e.alias(c) for c, e in text.items()),
    )


def _with_maf(rows: pl.DataFrame, annotation: AnnotationFn, finnish: bool) -> pl.DataFrame:
    """`maf` and `af_source`: the set's maf, else its aaf folded, else the annotation AF."""
    rows = rows.with_columns(
        pl.when(pl.col("maf").is_not_null()).then(pl.lit("credible_set maf"))
        .when(pl.col("aaf").is_not_null()).then(pl.lit("credible_set aaf"))
        .otherwise(None).alias("af_source"),
        pl.coalesce("maf", pl.min_horizontal("aaf", 1 - pl.col("aaf"))).alias("maf"),
    )
    need = rows.filter(pl.col("maf").is_null())["variant"].unique().sort().to_list()
    if not need:
        return rows

    def lookup(variants: list[str], source: str, columns: tuple[str, ...]) -> dict:
        out: dict[str, tuple[float, str]] = {}
        for i in range(0, len(variants), _ANNOTATION_BATCH):
            out.update(_annotation_af(annotation, variants[i:i + _ANNOTATION_BATCH], source,
                                      columns))
        return out

    found = {v: (af, "finngen_annotation")
             for v, (af, _) in lookup(need, "finngen", ("AF",)).items()}
    if not finnish:
        rest = [v for v in need if v not in found]
        found.update({v: (af, "gnomad_nfe" if col == "AF_nfe" else "gnomad_fin")
                      for v, (af, col) in lookup(rest, "gnomad", ("AF_nfe", "AF_fin")).items()})
    ann = pl.DataFrame(
        {"variant": list(found), "_af": [a for a, _ in found.values()],
         "_src": [s for _, s in found.values()]},
        schema={"variant": pl.String, "_af": pl.Float64, "_src": pl.String},
    )
    return rows.join(ann, on="variant", how="left").with_columns(
        pl.coalesce("maf", pl.min_horizontal("_af", 1 - pl.col("_af"))).alias("maf"),
        pl.coalesce("af_source", "_src").alias("af_source"),
    ).drop("_af", "_src")


def _pick_lead(members: list[dict[str, Any]], floor: float, by_cs_id: bool) -> dict | None:
    eligible = [m for m in members if m["maf"] is not None and m["maf"] >= floor]
    if by_cs_id:
        named = _CS_ID_VARIANT.match(members[0]["cs_id"] or "")
        if named:
            wanted = ":".join(named.groups()).upper().replace("X:", "23:", 1)
            hit = next((m for m in eligible if m["variant"].upper() == wanted), None)
            if hit:
                return hit
    eligible.sort(key=lambda m: (-(m["pip"] or 0), -(m["mlog10p"] or 0), m["variant"]))
    return eligible[0] if eligible else None


def _in_mhc(lead: dict[str, Any]) -> bool:
    return lead["chr"] == MHC[0] and MHC[1] <= (lead["pos"] or 0) <= MHC[2]


def instruments(
    exposure: pl.DataFrame | dict[str, Any],
    *,
    mode: str = "cis",
    cis_window: int = 1_000_000,
    p_threshold: float = 5e-8,
    maf_floor: float = 0.01,
    r2_prune: float = 0.01,
    exclude_mhc: bool = True,
    panel: str = "finngen_r12",
    allow_pseudo_credible_sets: bool = False,
    exposure_dataset: dict[str, Any] | None = None,
    outcome: dict[str, Any] | None = None,
    ld: Callable[..., pl.DataFrame] | None = None,
    annotation: AnnotationFn | None = None,
    outcome_lookup: Callable[..., pl.DataFrame] | None = None,
    sql: Callable[..., pl.DataFrame] | None = None,
    ld_panels: Callable[[], pl.DataFrame] | None = None,
) -> Instruments:
    """Pick independent credible-set leads of one exposure trait as MR instruments.

    `exposure` is a credible_sets_v-shaped frame of ONE (dataset, trait) — every member row
    of its sets, not only leads — or a dict {"dataset" and/or "resource", "trait",
    ["trait_original"]} that this fetches through `sql`. LEAD RULE: per credible set the
    max-pip variant whose MAF reaches `maf_floor` (ties by mlog10p, then id); a set with
    none contributes nothing. MAF is the set's `maf`, else its `aaf` folded, else the
    annotation AF (FinnGen first; gnomAD AF_nfe only for a non-Finnish exposure), with the
    source in `af_source`. A lead must reach `p_threshold` and, with `exclude_mhc`, lie
    outside chr6:25-34 Mb. `mode="cis"` keeps rows within `cis_window` of the gene whose
    HGNC symbol equals the trait, and REFUSES when no cis set survives (no fallback).
    `cis_window` is 1 Mb either side of the gene, not the investigation doc's 500 kb: 1 Mb
    is what Zhang 2025 and the kill measurement used.

    COST: a cis dict exposure costs two `sql` calls, the gene lookup then the fetch (a
    literal chr and pos range, which prunes the partition). `mode="genome_wide"` on a dict
    skips the gene lookup and fetches every credible-set row of the trait, which is allowed
    but costs a whole-table scan and an `ld` scan per independent lead (live, UKB_PPP IL12B:
    34 kept, 42 requests).

    The exposure's registry row decides pseudo-ness and Finnish-ness. It is looked up by the
    rows' `dataset` label (by `resource` when the rows carry none); rows holding more than one
    dataset are refused, and a dataset with no registry row is treated as non-Finnish and not
    pseudo, with a note.

    Pseudo-credible-set datasets are refused unless `allow_pseudo_credible_sets`; then the
    lead is the variant `cs_id` names when it passes the floor, else the max-pip one.

    PRUNING: greedy by mlog10p, one `ld` neighbourhood scan per lead considered; a lead
    with r2 >= `r2_prune` to a kept one is dropped, and one the panel lacks is dropped as
    "LD unknown", never kept as independent. The kept leads' scans give the signed-r matrix;
    a pair whose two scans disagree in sign or by more than 1e-6 raises GeneticsError.

    PROXIES (when `outcome` {"phenotype", "resource"[, "data_type"]} or `outcome_lookup`
    is given): a kept lead absent from the outcome is replaced by its highest-r2 neighbour
    at r2 >= 0.8 that the outcome has, from the same scans, in one batched lookup; its beta
    is sign(proxy_r) times the lead's, so the row is on the proxy's alt allele.

    Injectables, each defaulting to the SDK function looked up at call time:
    `ld(variant, *, window, r2_threshold, panel, fields)`, `annotation(variants=, source=)`,
    `outcome_lookup(variants)` -> rows with a variant key, `sql(query, max_rows=)`,
    `ld_panels()`. `exposure_dataset` is a `genetics.datasets()` row; looked up when absent.
    """
    if mode not in ("cis", "genome_wide"):
        raise GeneticsUsageError(f"mode must be 'cis' or 'genome_wide', got {mode!r}")
    counts: dict[str, int] = {}
    ld = _Counted(ld or _sdk("ld"), counts, "ld")
    annotation = _Counted(annotation or _sdk("variant_annotation"), counts, "annotation")
    sql = _Counted(sql or _sdk("sql"), counts, "sql")
    ld_panels = _Counted(ld_panels or _sdk("ld_panels"), counts, "ld_panels")
    notes: list[str] = []
    dropped: list[dict[str, Any]] = []
    threshold = -math.log10(p_threshold)

    if isinstance(exposure, dict):
        spec = dict(exposure)
        trait = spec.get("trait")
        dataset, resource = spec.get("dataset"), spec.get("resource")
        if not trait or not (dataset or resource):
            raise GeneticsUsageError(
                "an exposure dict needs 'trait' and 'dataset' and/or 'resource'")
        frame = None
    else:
        frame = exposure
        for column in ("trait", "dataset"):
            if column in frame.columns and frame[column].n_unique() > 1:
                raise GeneticsUsageError(
                    f"instruments() takes one exposure: the frame has {frame[column].n_unique()} "
                    f"values of `{column}`; filter it to one first")
        trait = frame["trait"][0] if "trait" in frame.columns and frame.height else None
        dataset = frame["dataset"][0] if "dataset" in frame.columns and frame.height else None
        resource = frame["resource"][0] if "resource" in frame.columns and frame.height else None
        if mode == "cis" and not trait:
            raise GeneticsUsageError("cis mode needs the frame's `trait` column (the gene symbol)")

    def settle() -> None:
        pseudo = bool((exposure_dataset or {}).get("pseudo_credible_sets"))
        if pseudo and not allow_pseudo_credible_sets:
            raise GeneticsUsageError(
                f"{dataset or resource} has pseudo credible sets (LD-clumped, not fine-mapped) "
                "and is refused as an MR exposure: they emit satellite and singleton sets whose "
                "leads sit tens of kb from the real signal (measured on deCODE against "
                "UKB_PPP). Pass allow_pseudo_credible_sets=True to use the variant each cs_id "
                "names instead")

    def registry_row(name: str | None) -> dict[str, Any] | None:
        datasets = _Counted(_sdk("datasets"), counts, "datasets")
        return _dataset_row(datasets(include_stats=False), name, resource)

    resolved = exposure_dataset is not None
    if not resolved and frame is None and dataset:
        # refused before any SQL: a dict names its dataset, so the registry row is final here
        exposure_dataset = registry_row(dataset)
        if exposure_dataset is None:
            raise GeneticsUsageError(
                f"dataset {dataset!r} is not in genetics.datasets(); a dict exposure names "
                "the `dataset` value the results views hold (e.g. 'UKB_PPP')")
        resolved = True
    if resolved:
        settle()

    region = None
    if mode == "cis":
        genes = sql(
            "SELECT symbol, chr, gene_start, gene_end FROM gene_annotations_v "
            f"WHERE symbol = {_quote(trait)}", max_rows=100)
        if genes.height != 1:
            raise GeneticsUsageError(
                f"cis mode reads the trait {trait!r} as an HGNC symbol and gene_annotations_v "
                f"has {genes.height} rows for it; use mode='genome_wide' or a symbol trait")
        g = genes.to_dicts()[0]
        region = {"gene": trait, "chr": int(g["chr"]),
                  "start": max(0, int(g["gene_start"]) - cis_window),
                  "end": int(g["gene_end"]) + cis_window, "cis_window": cis_window}
        notes.append(f"cis gene: the trait {trait} is read as its HGNC symbol (no "
                     "trait_original or aptamer mapping)")

    if frame is None:
        where = [f"trait = {_quote(trait)}"]
        where += [f"{c} = {_quote(spec[c])}" for c in ("dataset", "resource", "trait_original")
                  if spec.get(c)]
        if region:
            # a literal chr prunes the partition, and joining gene_annotations_v in SQL
            # instead would cost a scan of both
            where += [f"chr = {region['chr']}",
                      f"pos BETWEEN {region['start']} AND {region['end']}"]
        # sql() raises rather than truncate at max_rows
        frame = sql(f"SELECT {_CS_SQL_COLUMNS} FROM credible_sets_v WHERE "
                    + " AND ".join(where), max_rows=_CS_MAX_ROWS)
        fetched_only_window = region is not None
    else:
        fetched_only_window = False

    labels = (sorted(frame["dataset"].drop_nulls().cast(pl.String).unique().to_list())
              if "dataset" in frame.columns else [])
    if len(labels) > 1:
        raise GeneticsUsageError(
            f"the credible-set rows of {trait} hold {len(labels)} datasets {labels}; "
            "instruments() takes one exposure: name its `dataset`")
    if labels:
        dataset = labels[0]
    if not resolved:
        if dataset or resource:
            exposure_dataset = registry_row(dataset)
        if exposure_dataset is None:
            notes.append(f"exposure {dataset or resource or '(unnamed)'} has no "
                         "genetics.datasets() row: treated as non-Finnish and not pseudo for "
                         "want of one")
        settle()
    pseudo = bool((exposure_dataset or {}).get("pseudo_credible_sets"))
    if pseudo:
        notes.append(f"{dataset} has pseudo credible sets: lead = the variant cs_id names when "
                     f"it reaches MAF {maf_floor}, else the max-pip variant that does")
    if exposure_dataset is not None and not exposure_dataset.get("cohorts"):
        notes.append(f"{dataset} records no cohorts: treated as non-Finnish for the AF fallback")
    finnish = _is_finnish(exposure_dataset)

    rows = _normalise_sets(frame) if frame.height else _normalise_sets(
        pl.DataFrame(schema={"variant": pl.String}))
    n_outside = 0
    if region:
        inside = (pl.col("chr") == region["chr"]) & pl.col("pos").is_between(
            region["start"], region["end"])
        n_outside = rows.filter(~inside).height
        rows = rows.filter(inside)
        if n_outside:
            notes.append(f"{n_outside} credible-set row(s) outside the cis window "
                         f"chr{region['chr']}:{region['start']}-{region['end']} were left out")
    rows = _with_maf(rows, annotation, finnish)

    sets: dict[tuple, list[dict[str, Any]]] = {}
    for r in rows.to_dicts():
        sets.setdefault(tuple(r[c] for c in _SET_KEY), []).append(r)
    candidates, n_floor, n_p, n_mhc = [], 0, 0, 0
    for members in sets.values():
        lead = _pick_lead(members, maf_floor, by_cs_id=pseudo)
        cs_id = members[0]["cs_id"]
        if lead is None:
            n_floor += 1
            dropped.append({"variant": None, "cs_id": cs_id,
                            "reason": f"no variant with MAF >= {maf_floor:g}"})
        elif (lead["mlog10p"] or 0) < threshold:
            n_p += 1
            dropped.append({"variant": lead["variant"], "cs_id": cs_id,
                            "reason": f"lead p above {p_threshold:g}"})
        elif exclude_mhc and _in_mhc(lead):
            n_mhc += 1
            dropped.append({"variant": lead["variant"], "cs_id": cs_id, "reason": "lead in MHC"})
        else:
            candidates.append(lead)

    if not candidates:
        where = (f"cis window chr{region['chr']}:{region['start']}-{region['end']} "
                 f"({trait} ± {cis_window:,})" if region else "genome-wide")
        if not sets:
            if fetched_only_window:
                reason = "no credible set fine-mapped within the window"
            elif n_outside:
                reason = f"its {n_outside} credible-set rows all lie outside it (trans sets only)"
            else:
                reason = "no credible-set rows"
        else:
            reason = (f"no credible set with a lead at MAF >= {maf_floor:g} and p <= "
                      f"{p_threshold:g}{' outside the MHC' if exclude_mhc else ''} "
                      f"({len(sets)} sets: {n_floor} below the MAF floor, {n_p} above the p "
                      f"threshold, {n_mhc} in the MHC)")
        raise GeneticsUsageError(f"no instrument for {trait} in {dataset}, {where}: {reason}")

    candidates.sort(key=lambda m: (-(m["mlog10p"] or 0), -(m["pip"] or 0), m["variant"]))
    seen: dict[str, dict] = {}
    unique = []
    for lead in candidates:
        if lead["variant"] in seen:
            dropped.append({"variant": lead["variant"], "cs_id": lead["cs_id"],
                            "reason": f"same lead as set {seen[lead['variant']]['cs_id']}"})
        else:
            seen[lead["variant"]] = lead
            unique.append(lead)

    panel_info = _panel_info(ld_panels(), panel)
    window = min(LD_SCAN_CAP, panel_info["max_distance"])
    scan_r2 = panel_info["r2_floor"]
    panel_info.update(window=window, scan_r2_threshold=scan_r2)
    if r2_prune < scan_r2:
        notes.append(f"r2_prune {r2_prune:g} is below the panel floor {scan_r2:g}: pairs "
                     "between the two are not served and count as independent")

    kept: list[dict] = []
    scans: dict[str, dict[str, tuple[float, float]]] = {}
    for lead in unique:
        v = lead["variant"]
        hit = _ld_hit(v, kept, scans, r2_prune)
        if hit is None:
            try:
                scan = ld(v, window=window, r2_threshold=scan_r2, panel=panel,
                          fields=["r", "r2"])
            except GeneticsError as exc:
                if exc.code != "ld_variant_not_in_panel":
                    raise
                # results-api's 404 also covers a swapped spelling and a variant with no
                # partner above the floor; its text says which
                dropped.append({"variant": v, "cs_id": lead["cs_id"],
                                "reason": f"LD unknown: not in panel {panel}: {exc}"})
                continue
            scans[v] = {} if scan.is_empty() else {
                k: (r, r2) for k, r, r2 in scan.select(
                    _variant_key(scan), _num(scan, "r"), _num(scan, "r2")).iter_rows()
                if k != v}
            hit = _ld_reverse_hit(v, kept, scans, r2_prune)
        if hit is not None:
            dropped.append({"variant": v, "cs_id": lead["cs_id"],
                            "reason": f"r2 {hit[1]:.3f} with stronger lead {hit[0]}"})
            continue
        kept.append(lead)

    order = [k["variant"] for k in kept]
    r_matrix, asym = _r_matrix(order, scans)
    if asym:
        notes.append(f"{asym} lead pair(s) differed between the two scan orderings by at most "
                     "1e-6 or appeared in one only; the matrix uses their mean")

    out = pl.DataFrame(kept, infer_schema_length=None).select(_LEAD_COLUMNS).with_columns(
        pl.lit(False).alias("is_proxy"), pl.lit(None, pl.String).alias("proxy_for"),
        pl.lit(None, pl.Float64).alias("proxy_r"), pl.lit(None, pl.Float64).alias("proxy_r2"),
        pl.lit(None, pl.Boolean).alias("in_outcome"),
        pl.lit(None, pl.Float64).alias("lead_beta"), pl.lit(None, pl.Float64).alias("lead_maf"),
    )
    if outcome_lookup is None and outcome is not None:
        outcome_lookup = _default_outcome_lookup(outcome)
    if outcome_lookup is not None:
        out = _proxies(out, scans, _Counted(outcome_lookup, counts, "outcome_lookup"), notes)

    counts["total"] = sum(counts.values())
    return Instruments(
        frame=out,
        dropped=pl.DataFrame(dropped, schema={"variant": pl.String, "cs_id": pl.String,
                                              "reason": pl.String}),
        r=r_matrix,
        panel=panel_info,
        requests=counts,
        notes=notes,
        exposure_dataset=exposure_dataset,
        cis_region=region,
    )


def _panel_info(panels: pl.DataFrame, panel: str) -> dict[str, Any]:
    rows = [] if panels.is_empty() else panels.filter(pl.col("panel") == panel).to_dicts()
    if not rows:
        served = [] if panels.is_empty() else panels["panel"].to_list()
        raise GeneticsUsageError(
            f"instruments() needs signed r, which only an own panel serves; {panel!r} is not "
            f"one of ld_panels() {served}")
    row = rows[0]
    return {"name": panel, "r2_floor": float(row["r2_floor"]),
            "max_distance": int(row["max_distance"])}


def _ld_hit(v: str, kept: list[dict], scans: dict, r2_prune: float) -> tuple | None:
    """(kept lead, r2) for the first kept lead whose scan holds `v` at r2 >= r2_prune."""
    for k in kept:
        pair = scans[k["variant"]].get(v)
        if pair and pair[1] is not None and pair[1] >= r2_prune:
            return k["variant"], pair[1]
    return None


def _ld_reverse_hit(v: str, kept: list[dict], scans: dict, r2_prune: float) -> tuple | None:
    for k in kept:
        pair = scans[v].get(k["variant"])
        if pair and pair[1] is not None and pair[1] >= r2_prune:
            return k["variant"], pair[1]
    return None


def _r_matrix(order: list[str], scans: dict) -> tuple[pl.DataFrame, int]:
    asym = 0
    values = [[0.0] * len(order) for _ in order]
    for i, a in enumerate(order):
        values[i][i] = 1.0
        for j in range(i + 1, len(order)):
            b = order[j]
            ab = (scans[a].get(b) or (None,))[0]
            ba = (scans[b].get(a) or (None,))[0]
            if ab is None and ba is None:
                continue
            if ab is not None and ba is not None and (ab * ba < 0 or abs(ab - ba) > 1e-6):
                raise GeneticsError(
                    f"the two LD scans disagree on r({a}, {b}): {ab} from {a}'s scan, {ba} "
                    f"from {b}'s; the panel's r is not usable for this pair")
            if ab is None or ba is None or ab != ba:
                asym += 1
            r = (ab if ba is None else ba if ab is None else (ab + ba) / 2)
            values[i][j] = values[j][i] = r
    frame = pl.DataFrame({"variant": order, **{v: [row[i] for row in values]
                                               for i, v in enumerate(order)}},
                         schema={"variant": pl.String, **{v: pl.Float64 for v in order}})
    return frame, asym


def _default_outcome_lookup(outcome: dict[str, Any]) -> Callable[[list[str]], pl.DataFrame]:
    if not outcome.get("phenotype") or not outcome.get("resource"):
        raise GeneticsUsageError("outcome needs 'phenotype' and 'resource'")

    def lookup(variants: list[str]) -> pl.DataFrame:
        import genetics_mcp_server.sdk as sdk

        return sdk.summary_stats(phenotypes=[outcome["phenotype"]], variants=variants,
                                 resource=outcome["resource"],
                                 data_type=outcome.get("data_type", "gwas"))
    return lookup


def _proxies(out: pl.DataFrame, scans: dict, lookup: Callable[..., pl.DataFrame],
             notes: list[str]) -> pl.DataFrame:
    """Replace each lead the outcome lacks by its best r2 >= PROXY_R2 neighbour it has."""
    leads = out["variant"].to_list()
    options = {v: sorted(((k, r, r2) for k, (r, r2) in scans[v].items()
                          if r2 is not None and r2 >= PROXY_R2),
                         key=lambda o: (-o[2], o[0]))
               for v in leads}
    wanted = sorted(set(leads) | {k for opts in options.values() for k, _, _ in opts})
    rows = lookup(wanted)
    present: set[str] = set()
    if rows is not None and not rows.is_empty():
        if "beta" in rows.columns:
            rows = rows.filter(_num(rows, "beta").is_not_null())
        present = set(rows.select(_variant_key(rows)).to_series().to_list())

    records = []
    n_proxied = 0
    for rec in out.to_dicts():
        lead = rec["variant"]
        if lead in present:
            records.append({**rec, "in_outcome": True})
            continue
        best = next((o for o in options[lead] if o[0] in present), None)
        if best is None:
            records.append({**rec, "in_outcome": False})
            continue
        n_proxied += 1
        c, p, ref, alt = best[0].split(":")
        sign = 1.0 if best[1] > 0 else -1.0
        records.append({**rec, "variant": best[0], "chr": int(c), "pos": int(p), "ref": ref,
                        "alt": alt, "beta": None if rec["beta"] is None else sign * rec["beta"],
                        "maf": None, "af_source": "proxy: no frequency",
                        "lead_beta": rec["beta"], "lead_maf": rec["maf"], "is_proxy": True,
                        "proxy_for": lead, "proxy_r": best[1], "proxy_r2": best[2],
                        "in_outcome": True})
    if n_proxied:
        notes.append(
            f"{n_proxied} lead(s) replaced by an outcome-present proxy at r2 >= {PROXY_R2:g}: "
            "the row carries the proxy's own alleles and beta = sign(proxy_r) * the lead's "
            "beta (applied here; the lead's is in lead_beta), so the outcome beta read at the "
            "proxy aligns as is; maf is null, the scan carries no frequency")
    return pl.DataFrame(records, schema=out.schema)


# --------------------------------------------------------------------------- run

# results-api's SANDBOX_MAX_REQUESTS_PER_EXECUTION default; the server refuses the next request
REQUEST_CAP = 1000
# the server's per-execution concurrency cap: a 5th concurrent request is refused, not queued
MAX_IN_FLIGHT = 4
# phenotypes per summary_stats call when fanning out over many outcomes
OUTCOME_CHUNK = 10
_R2_PRUNE = {"cis": 0.1, "genome_wide": 0.01}
_INJECTABLES = frozenset({"ld", "annotation", "sql", "ld_panels", "datasets", "summary_stats",
                          "phenotypes", "exposure_dataset"})
_COLOC_MAX_ROWS = 10_000
# `cohorts` tokens of case-control consortia, whose sample prevalence says nothing about the
# population's; a token not listed here gets the generic Steiger prevalence note
_CASE_CONTROL_CONSORTIA = frozenset({"iibdgc", "pgc", "covid_hgi", "gp2"})
_select_instruments = instruments


@dataclass
class MRResult:
    """One exposure-outcome analysis from `run()`.

    `harmonised` (harmonise()'s result) and `instruments` (instruments()' result, its frame
    after proxies) are the inputs; `estimates` maps method -> estimator dict (`wald`, `ivw`,
    `ivw_correlated`, `egger`, `weighted_median`, `simple_median`, `weighted_mode`,
    `simple_mode`, whichever ran), `headline` names the one `summary()` reports.
    `sensitivity` holds `leave_one_out` and `single_snp` frames and the `steiger` and
    `heterogeneity` dicts (None where too few instruments, with a note); `checks` is checks()'
    frame; `parameters` every default used; `overlap` (status, detail); `versions` the
    datasets() version of each side and the outcome rows' own `version`; `coloc` the
    precomputed colocalization rows for this outcome or None; `requests` calls per kind and
    their total (for a batch, the whole batch's).
    """

    exposure: str
    outcome: str
    headline: str
    harmonised: Harmonised
    instruments: Instruments
    estimates: dict[str, dict[str, Any]]
    sensitivity: dict[str, Any]
    checks: pl.DataFrame
    parameters: dict[str, Any]
    overlap: tuple[str, str]
    versions: dict[str, Any]
    coloc: pl.DataFrame | None
    notes: list[str]
    requests: dict[str, int]

    def summary(self) -> pl.DataFrame:
        e = self.estimates[self.headline]
        failing = 0 if self.checks.is_empty() else int((self.checks["status"] == "fail").sum())
        return pl.DataFrame([{
            "exposure": self.exposure, "outcome": self.outcome, "method": self.headline,
            "estimate": e["estimate"], "se": e["se"], "p": e["p"], "ci_low": e["ci_low"],
            "ci_high": e["ci_high"], "n_instruments": e["n_instruments"],
            "overlap": self.overlap[0], "checks_failing": failing,
        }])

    def __repr__(self) -> str:
        e = self.estimates[self.headline]
        return (f"MRResult({self.exposure} -> {self.outcome}: {self.headline} "
                f"{e['estimate']:.4g} (se {e['se']:.3g}, p {e['p']:.3g}), "
                f"K={e['n_instruments']}, overlap {self.overlap[0]})")


@dataclass
class MRBatch:
    """`run(..., outcomes=[...])`: one MRResult per outcome that ran, and why the rest did not."""

    results: list[MRResult]
    failed: pl.DataFrame
    instruments: Instruments
    requests: dict[str, int]
    notes: list[str] = field(default_factory=list)

    def summary(self) -> pl.DataFrame:
        frames = [r.summary() for r in self.results]
        return pl.concat(frames) if frames else pl.DataFrame()

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    def __getitem__(self, i: int) -> MRResult:
        return self.results[i]

    def __repr__(self) -> str:
        return (f"MRBatch({len(self.results)} ran, {self.failed.height} failed, "
                f"{self.requests.get('total', 0)} requests)")


def _outcome_spec(spec: Any) -> dict[str, Any]:
    if isinstance(spec, tuple) and len(spec) == 2:
        return {"resource": spec[0], "phenotype": spec[1], "data_type": "gwas"}
    if isinstance(spec, dict) and spec.get("phenotype") and spec.get("resource"):
        return {"data_type": "gwas", **spec}
    raise GeneticsUsageError(
        f"an outcome is {{'phenotype', 'resource'}} or a (resource, phenotype) tuple, got {spec!r}")


def _exposure_names(exposure: Any) -> tuple[str | None, str | None, str | None]:
    """(dataset, resource, trait) of a dict or a credible-set frame."""
    if isinstance(exposure, dict):
        return exposure.get("dataset"), exposure.get("resource"), exposure.get("trait")
    if isinstance(exposure, pl.DataFrame):
        def first(c: str) -> str | None:
            return str(exposure[c][0]) if c in exposure.columns and exposure.height else None
        return first("dataset"), first("resource"), first("trait")
    raise GeneticsUsageError(
        "exposure is a dict {dataset or resource, trait} or one trait's credible-set frame")


def _scan_recorder(ld: Callable[..., pl.DataFrame], scans: dict) -> Callable[..., pl.DataFrame]:
    """Keeps each neighbourhood scan as instruments() reads it, so proxies can be chosen per
    outcome after one instrument selection."""
    def call(variant: str, **kwargs: Any) -> pl.DataFrame:
        scan = ld(variant, **kwargs)
        scans[variant] = {} if scan.is_empty() else {
            k: (r, r2) for k, r, r2 in scan.select(
                _variant_key(scan), _num(scan, "r"), _num(scan, "r2")).iter_rows()
            if k != variant}
        return scan
    return call


def _fan_out(fetch: Callable[..., Any], jobs: list[dict[str, Any]],
             max_in_flight: int) -> list[Any]:
    """The SDK's documented fan-out: batches of `max_in_flight`, return_exceptions=True."""
    async def one(kwargs: dict[str, Any]) -> Any:
        out = fetch(**kwargs)
        return await out if inspect.isawaitable(out) else out

    async def main() -> list[Any]:
        out: list[Any] = []
        for i in range(0, len(jobs), max_in_flight):
            out += await asyncio.gather(*(one(j) for j in jobs[i:i + max_in_flight]),
                                        return_exceptions=True)
        return out
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(main())
    from genetics_mcp_server.sdk import _runner

    # inside a running loop asyncio.run raises; the SDK's own loop runs it instead
    return _runner.run(main())


def _default_summary_stats(**kwargs: Any) -> Any:
    import genetics_mcp_server.sdk as sdk

    return sdk.get_client().summary_stats(**kwargs)


def _resolve_outcome(datasets: list[dict], spec: dict, meta: pl.DataFrame | None,
                     notes: list[str]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """(registry row, phenotypes() row) of one outcome. A code is often listed under several
    datasets of one resource (FinnGen R12, R14 and HLA); the one serving summary statistics
    of the outcome's data_type is the one summary_stats() reads."""
    def serves(row: dict | None) -> bool:
        return bool(row) and row.get("data_type") == spec["data_type"] and bool(
            (row.get("products") or {}).get("summary_stats"))

    if spec.get("dataset"):
        row = _dataset_row(datasets, spec["dataset"], spec["resource"])
        metas = [] if meta is None or meta.is_empty() or "trait_original" not in meta.columns \
            else meta.filter(pl.col("trait_original") == spec["phenotype"]).to_dicts()
        names = row.get("dataset") if row else None
        names = names if isinstance(names, list) else [names]
        return row, next((m for m in metas if m.get("dataset") in names), None)
    metas = [] if meta is None or meta.is_empty() or "trait_original" not in meta.columns \
        else meta.filter(pl.col("trait_original") == spec["phenotype"]).to_dicts()
    listed = [(_dataset_row(datasets, m.get("dataset"), spec["resource"]), m) for m in metas]
    for row, m in listed:
        if serves(row):
            return row, m
    if listed:
        # listed, but under no dataset that serves summary statistics: refused by the caller
        return listed[0]
    candidates = [d for d in datasets if d.get("resource") == spec["resource"] and serves(d)]
    if len(candidates) > 1:
        notes.append(f"outcome dataset of {spec['resource']} {spec['phenotype']} is ambiguous "
                     f"({[c['dataset_id'] for c in candidates]}); using "
                     f"{candidates[0]['dataset_id']}; name it with outcome['dataset']")
    return (candidates[0] if candidates else None), None


def _r_for(frame: pl.DataFrame, r: pl.DataFrame | None, scans: dict) -> pl.DataFrame | None:
    """The instruments' signed r, labelled by the rows' variants, for the correlated IVW.

    `frame` carries `proxy_for` and `proxy_r` (joined from the instruments frame; harmonise()
    does not keep them). A lead row reads `r`. A proxied row has no LD scan of its own, so its
    r with row b is the chain r(lead_a, lead_b) * r(proxy_a, lead_a) * r(proxy_b, lead_b),
    each proxy_r read from its lead's scan and a factor of 1 for a side that is not proxied.
    When b is a lead whose own scan holds a's proxy, that exact r is used instead. The chain
    is exact only as the proxy r2 approaches 1. The diagonal is 1. None when a row's lead is
    missing from `r` (a caller's r that does not cover an instrument).
    """
    if r is None or "variant" not in r.columns:
        return None
    leads = r["variant"].to_list()
    rows = frame.select(
        "variant",
        (pl.col("proxy_for") if "proxy_for" in frame.columns else pl.lit(None, pl.String))
        .fill_null(pl.col("variant")).alias("_lead"),
        (pl.col("proxy_r") if "proxy_r" in frame.columns else pl.lit(None, pl.Float64))
        .alias("_proxy_r"),
    ).to_dicts()
    if any(x["_lead"] not in leads for x in rows):
        return None
    idx = {v: i for i, v in enumerate(leads)}
    m = r.select(leads).to_numpy()

    def pair(a: dict, b: dict) -> float:
        if a["variant"] == b["variant"]:
            return 1.0
        if a["_proxy_r"] is not None and b["_proxy_r"] is None:
            exact = (scans.get(b["variant"], {}).get(a["variant"]) or (None,))[0]
            if exact is not None:
                return float(exact)
        if b["_proxy_r"] is not None and a["_proxy_r"] is None:
            exact = (scans.get(a["variant"], {}).get(b["variant"]) or (None,))[0]
            if exact is not None:
                return float(exact)
        return (float(m[idx[a["_lead"]], idx[b["_lead"]]])
                * (1.0 if a["_proxy_r"] is None else a["_proxy_r"])
                * (1.0 if b["_proxy_r"] is None else b["_proxy_r"]))

    out = {"variant": [x["variant"] for x in rows]}
    for b in rows:
        out[b["variant"]] = [pair(a, b) for a in rows]
    return pl.DataFrame(out)


def _try(notes: list[str], label: str, fn: Callable[[], Any],
         errors: type[Exception] = GeneticsUsageError) -> Any:
    """fn(), or None with a note: an estimator's refusal (too few instruments, no sample size)
    or, with errors=GeneticsError, a lookup whose failure only loses decoration."""
    try:
        return fn()
    except errors as exc:
        notes.append(f"{label} not computed: {exc}")
        return None


def _estimate(used: pl.DataFrame, mode: str, r: pl.DataFrame | None,
              notes: list[str]) -> tuple[dict[str, dict], str]:
    k = used.height
    if k == 1:
        return {"wald": ivw(used)}, "wald"
    est: dict[str, dict] = {}
    if mode == "cis" and r is not None:
        corr = ivw(used, correl=r)
        corr["method"] = "ivw_correlated"
        est["ivw_correlated"] = corr
        est["ivw"] = ivw(used)
        if k >= 3:
            est["egger"] = egger(used, correl=r)
        else:
            notes.append("MR Egger not computed: it needs 3 instruments")
        return est, "ivw_correlated"
    est["ivw"] = ivw(used)
    methods: dict[str, Callable[[], dict]] = {"egger": lambda: egger(used)}
    if mode != "cis":
        methods.update(weighted_median=lambda: weighted_median(used),
                       simple_median=lambda: simple_median(used),
                       weighted_mode=lambda: weighted_mode(used),
                       simple_mode=lambda: simple_mode(used))
    for name, fn in methods.items():
        value = _try(notes, name, fn)
        if value is not None:
            est[name] = value
    return est, "ivw"


def run(
    exposure: pl.DataFrame | dict[str, Any],
    outcome: dict[str, Any] | tuple[str, str] | None = None,
    *,
    outcomes: list[dict[str, Any] | tuple[str, str]] | None = None,
    instruments: str | Instruments | pl.DataFrame = "cis",
    prevalence: float | None = None,
    r2_prune: float | None = None,
    cis_window: int = 1_000_000,
    p_threshold: float = 5e-8,
    maf_floor: float = 0.01,
    panel: str = "finngen_r12",
    allow_pseudo_credible_sets: bool = False,
    attach_coloc: bool = True,
    max_in_flight: int = MAX_IN_FLIGHT,
    **injectables: Any,
) -> MRResult | MRBatch:
    """Two-sample MR of one exposure on one outcome (an MRResult) or on many (an MRBatch).

    `exposure`: a dict {"dataset" and/or "resource", "trait"[, "trait_original"]} or one
    trait's credible-set frame, as instruments() takes it. A list of exposures is refused.
    `outcome`: {"phenotype", "resource"[, "data_type", "dataset"]} or (resource, phenotype);
    `outcomes`: a list of them, analysed against ONE instrument selection.

    `instruments`: "cis" (default; r2_prune 0.1) or "genome_wide" (r2_prune 0.01) runs
    instruments() with these parameters; an Instruments or a frame of instrument rows
    (`variant`, `beta`, `se`, optional `maf`, `cs_id`) is used as given, with no proxies, and
    estimated as cis when it carries a cis region and an r matrix.

    `prevalence`: the binary outcome's population prevalence for Steiger. Without it the
    sample prevalence n_cases / (n_cases + n_controls) is used and noted: close to the
    population's for a population cohort such as FinnGen, not for a case-control consortium.

    REFUSED: the same dataset and trait on both sides; an outcome dataset without summary
    statistics; an outcome where no instrument (lead or r2 >= 0.8 proxy) is found; more
    than `MAX_IN_FLIGHT` in flight; and a batch whose estimated requests exceed
    `REQUEST_CAP` (1,000 requests per execution: instrument selection + phenotypes + summary_stats chunks + coloc, the
    annotation lookups for rows without a frequency not included). instruments() adds the
    no-cis-set (D6) and pseudo-credible-set refusals. In a batch, an outcome that fails is
    listed in `.failed` with the reason instead of stopping the others.

    Injectables (each defaults to the SDK function): ld, annotation, sql, ld_panels,
    datasets, phenotypes, colocalization, summary_stats (sync or async, called with
    phenotypes=, variants=, resource=, data_type=), and exposure_dataset (a datasets() row).
    """
    unknown = set(injectables) - _INJECTABLES
    if unknown:
        raise GeneticsUsageError(f"run() got unexpected keyword(s) {sorted(unknown)}")
    if isinstance(exposure, (list, tuple)):
        raise GeneticsUsageError(
            "run() takes one exposure. Many-exposure (proteome-wide) screens are out of scope "
            "for v1: the lead-only Wald form of a screen (one sql of the lead rows and a few "
            "summary_stats calls) fits one execution, but this module does not provide it")
    if (outcome is None) == (outcomes is None):
        raise GeneticsUsageError("give exactly one of `outcome` or `outcomes`")
    if not 1 <= max_in_flight <= MAX_IN_FLIGHT:
        raise GeneticsUsageError(
            f"max_in_flight must be 1-{MAX_IN_FLIGHT}: the server refuses a 5th concurrent "
            "request rather than queueing it")
    batch = outcomes is not None
    specs = [_outcome_spec(o) for o in (outcomes if batch else [outcome])]
    if not specs:
        raise GeneticsUsageError("`outcomes` is empty")

    counts: dict[str, int] = {}
    datasets_fn = _Counted(injectables.get("datasets") or _sdk("datasets"), counts, "datasets")
    phenotypes_fn = _Counted(injectables.get("phenotypes") or _sdk("phenotypes"), counts,
                             "phenotypes")
    coloc_sql = _Counted(injectables.get("sql") or _sdk("sql"), counts, "colocalization.sql")
    fetch = _Counted(injectables.get("summary_stats") or _default_summary_stats, counts,
                     "summary_stats")
    annotation = injectables.get("annotation")
    notes: list[str] = []

    if isinstance(instruments, str):
        if instruments not in _R2_PRUNE:
            raise GeneticsUsageError(
                f"instruments is 'cis', 'genome_wide' or an Instruments/frame, got {instruments!r}")
        mode = instruments
        given = None
    else:
        given = instruments
        mode = "given"
    if r2_prune is None:
        r2_prune = _R2_PRUNE.get(mode)

    groups: dict[tuple[str, str], list[dict]] = {}
    for s in specs:
        groups.setdefault((s["resource"], s["data_type"]), []).append(s)
    n_fetch = sum(math.ceil(len(v) / OUTCOME_CHUNK) for v in groups.values())
    floor = 1 + len(groups) + n_fetch + (5 if mode != "given" else 0)
    if floor > REQUEST_CAP:
        raise GeneticsUsageError(
            f"{len(specs)} outcomes need at least {floor} requests before counting LD scans "
            f"and colocalization, over the {REQUEST_CAP:,}-request cap of one execution; "
            "split the outcomes across executions")

    datasets = datasets_fn(include_stats=False)
    x_dataset_name, x_resource, trait = _exposure_names(exposure)
    exposure_dataset = injectables.get("exposure_dataset")
    if exposure_dataset is None and (x_dataset_name or x_resource):
        exposure_dataset = _dataset_row(datasets, x_dataset_name, x_resource)

    meta_by_resource: dict[str, pl.DataFrame | None] = {}
    for resource in sorted({s["resource"] for s in specs}):
        codes = sorted({s["phenotype"] for s in specs if s["resource"] == resource})
        meta_by_resource[resource] = _try(
            notes, f"phenotypes() metadata for {resource}",
            lambda r=resource, c=codes: phenotypes_fn(codes=c, resource=r), GeneticsError)

    failed: list[dict[str, str]] = []
    resolved: list[tuple[dict, dict | None, dict | None]] = []
    for s in specs:
        y_dataset, meta = _resolve_outcome(datasets, s, meta_by_resource.get(s["resource"]),
                                           notes)
        try:
            _refuse_pair(exposure_dataset, trait, exposure, s, y_dataset)
        except GeneticsUsageError as exc:
            if not batch:
                raise
            failed.append({"resource": s["resource"], "phenotype": s["phenotype"],
                           "reason": str(exc)})
            continue
        resolved.append((s, y_dataset, meta))

    scans: dict[str, dict] = {}
    if given is None:
        ld = _scan_recorder(injectables.get("ld") or _sdk("ld"), scans)
        inst = _select_instruments(
            exposure, mode=mode, cis_window=cis_window, p_threshold=p_threshold,
            maf_floor=maf_floor, r2_prune=r2_prune, panel=panel,
            allow_pseudo_credible_sets=allow_pseudo_credible_sets,
            exposure_dataset=exposure_dataset, ld=ld, annotation=annotation,
            sql=injectables.get("sql"), ld_panels=injectables.get("ld_panels"))
        exposure_dataset = inst.exposure_dataset or exposure_dataset
    else:
        inst = given if isinstance(given, Instruments) else Instruments(
            frame=given, dropped=pl.DataFrame(schema={"variant": pl.String, "cs_id": pl.String,
                                                      "reason": pl.String}),
            r=None, panel={}, requests={"total": 0}, cis_region=None)
        mode = "cis" if inst.cis_region and inst.r is not None else "genome_wide"
        exposure_dataset = exposure_dataset or inst.exposure_dataset
        notes.append("instruments given by the caller: used as given, no proxy lookup")
    for kind, n in inst.requests.items():
        if kind != "total":
            counts[f"instruments.{kind}"] = n

    base = inst.frame
    leads = base["variant"].to_list()
    wanted = sorted(set(leads) | {k for v in leads for k, (_, r2) in scans.get(v, {}).items()
                                  if r2 is not None and r2 >= PROXY_R2})
    x_resource = x_resource or (exposure_dataset or {}).get("resource")
    partnered = [(s, y, m) for s, y, m in resolved if _coloc_partnered(exposure_dataset, y)]
    cs_ids = (sorted(base["cs_id"].drop_nulls().cast(pl.String).unique().to_list())
              if "cs_id" in base.columns else [])
    coloc_wanted = attach_coloc and bool(partnered) and bool(cs_ids) and bool(trait)
    spent = sum(counts.values())
    estimate = spent + n_fetch + int(coloc_wanted)
    if estimate > REQUEST_CAP:
        raise GeneticsUsageError(
            f"this run needs about {estimate} requests ({spent} spent on instruments and "
            f"metadata, {n_fetch} summary_stats chunk(s) of up to {OUTCOME_CHUNK} outcomes, "
            f"{int(coloc_wanted)} colocalization), over the {REQUEST_CAP:,}-request cap of one "
            "execution; split the outcomes across executions")

    jobs, job_specs = [], []
    for (resource, data_type), members in groups.items():
        live = [s for s, _, _ in resolved if s in members]
        for i in range(0, len(live), OUTCOME_CHUNK):
            chunk = live[i:i + OUTCOME_CHUNK]
            jobs.append({"phenotypes": [s["phenotype"] for s in chunk], "variants": wanted,
                         "resource": resource, "data_type": data_type})
            job_specs.append(chunk)
    rows_by: dict[tuple[str, str, str], pl.DataFrame | BaseException] = {}
    for chunk, result in zip(job_specs, _fan_out(fetch, jobs, max_in_flight)):
        for s in chunk:
            key = (s["resource"], s["data_type"], s["phenotype"])
            if isinstance(result, BaseException):
                rows_by[key] = result
            elif result is None or result.is_empty() or "phenotype" not in result.columns:
                rows_by[key] = pl.DataFrame()
            else:
                rows_by[key] = result.filter(pl.col("phenotype") == s["phenotype"])

    coloc_all = None
    if coloc_wanted:
        coloc_all = _try(notes, "colocalization lookup", lambda: _coloc_query(
            coloc_sql, exposure_dataset, trait, cs_ids, partnered, _chromosomes(base)),
            GeneticsError)
    selection = {"cis_window": cis_window, "p_threshold": p_threshold, "maf_floor": maf_floor,
                 "panel": panel, "r2_prune": r2_prune}
    if given is not None:
        selection = dict.fromkeys(selection, "not used (instruments given)")
    parameters = {"mode": mode, **selection,
                  "allow_pseudo_credible_sets": allow_pseudo_credible_sets,
                  "af_tolerance": AF_TOLERANCE, "proxy_r2": PROXY_R2,
                  "attach_coloc": attach_coloc, "max_in_flight": max_in_flight,
                  "outcome_chunk": OUTCOME_CHUNK}
    results: list[MRResult] = []
    for s, y_dataset, meta in resolved:
        rows = rows_by[(s["resource"], s["data_type"], s["phenotype"])]
        if isinstance(rows, BaseException):
            if not batch:
                raise rows
            # any exception of one chunk, not only a GeneticsError, fails its outcomes only
            failed.append({"resource": s["resource"], "phenotype": s["phenotype"],
                           "reason": str(rows) or type(rows).__name__})
            continue
        try:
            results.append(_one_outcome(
                exposure_label=f"{x_dataset_name or x_resource} {trait}", spec=s,
                inst=inst, scans=scans, rows=rows, meta=meta, exposure_dataset=exposure_dataset,
                y_dataset=y_dataset, mode=mode, prevalence=prevalence, annotation=annotation,
                coloc=coloc_all, x_trait=trait, partnered=_coloc_partnered(exposure_dataset, y_dataset),
                attach_coloc=attach_coloc,
                parameters=parameters,
                base_notes=notes, counts=counts))
        except GeneticsError as exc:
            if not batch:
                raise
            failed.append({"resource": s["resource"], "phenotype": s["phenotype"],
                           "reason": str(exc)})
    counts["total"] = sum(v for k, v in counts.items() if k != "total")
    if not batch:
        return results[0]
    return MRBatch(results=results,
                   failed=pl.DataFrame(failed, schema={"resource": pl.String,
                                                       "phenotype": pl.String,
                                                       "reason": pl.String}),
                   instruments=inst, requests=counts, notes=notes)


def _refuse_pair(exposure_dataset: dict | None, trait: str | None, exposure: Any, spec: dict,
                 y_dataset: dict | None) -> None:
    if y_dataset is not None and not (y_dataset.get("products") or {}).get("summary_stats"):
        has = sorted(k for k, v in (y_dataset.get("products") or {}).items() if v)
        raise GeneticsUsageError(
            f"outcome {spec['resource']} {spec['phenotype']}: dataset {y_dataset['dataset_id']} "
            f"serves no summary statistics (it has {', '.join(has) or 'nothing'}), and an MR "
            "outcome needs each instrument's effect, not only fine-mapped sets")
    x_names = {trait} | ({exposure.get("trait_original")} if isinstance(exposure, dict) else set())
    if (exposure_dataset and y_dataset
            and exposure_dataset.get("dataset_id") == y_dataset.get("dataset_id")
            and spec["phenotype"] in x_names):
        raise GeneticsUsageError(
            f"exposure and outcome are the same trait {spec['phenotype']} in the same dataset "
            f"{y_dataset['dataset_id']}: that regresses a trait on itself")


def _coloc_partnered(x: dict | None, y: dict | None) -> bool:
    """Whether colocalization_v pairs the two datasets, by the registry's partner lists."""
    def partners(d: dict | None) -> set[str]:
        coloc = ((d or {}).get("products") or {}).get("colocalization")
        return set(coloc.get("partners") or []) if isinstance(coloc, dict) else set()
    return bool(x and y) and (y.get("dataset_id") in partners(x)
                              or x.get("dataset_id") in partners(y))


def _labels(d: dict | None) -> list[str]:
    value = (d or {}).get("dataset")
    return [v for v in (value if isinstance(value, list) else [value]) if v]


def _chromosomes(frame: pl.DataFrame) -> list[int]:
    """The instruments' chromosomes as colocalization_v's integer `chr` (X is 23)."""
    out = set()
    for v in frame["variant"].to_list():
        c = v.split(":")[0].removeprefix("chr")
        if c.isdigit():
            out.add(int(c))
        elif c.upper() == "X":
            out.add(23)
    return sorted(out)


def _coloc_query(sql: Callable[..., pl.DataFrame], x: dict | None, trait: str,
                 cs_ids: list[str], partnered: list[tuple],
                 chromosomes: list[int] | None = None) -> pl.DataFrame:
    """One query for every instrument's set against every partnered outcome, either side.

    `genetics.colocalization(credible_set_id=)` answers from one side of the pair only, and
    UKB_PPP sets sit on side 2 of colocalization_v, so it returns nothing for them. The view
    is partitioned by chr and prunes only on a literal chr predicate."""
    def in_list(values: list[str]) -> str:
        return "(" + ", ".join(_quote(v) for v in values) + ")"
    xl = in_list(_labels(x))
    yl = in_list(sorted({v for _, y, m in partnered
                         for v in ([m["dataset"]] if m and m.get("dataset") else _labels(y))}))
    ph = in_list(sorted({s["phenotype"] for s, _, _ in partnered}))
    cs = in_list(cs_ids)
    sides = []
    for a, b in (("1", "2"), ("2", "1")):
        sides.append(f"(dataset{a} IN {xl} AND trait{a} = {_quote(trait)} AND cs{a}_id IN {cs} "
                     f"AND dataset{b} IN {yl} AND trait{b} IN {ph})")
    chr_filter = (f"chr IN ({', '.join(str(c) for c in chromosomes)}) AND "
                  if chromosomes else "")
    return sql("SELECT * FROM colocalization_v WHERE " + chr_filter
               + "(" + " OR ".join(sides) + ")", max_rows=_COLOC_MAX_ROWS)


def _one_outcome(*, exposure_label: str, spec: dict, inst: Instruments, scans: dict,
                 rows: pl.DataFrame, meta: dict | None, exposure_dataset: dict | None,
                 y_dataset: dict | None, mode: str, prevalence: float | None,
                 annotation: AnnotationFn | None, coloc: pl.DataFrame | None, partnered: bool,
                 x_trait: str | None,
                 attach_coloc: bool,
                 parameters: dict, base_notes: list[str], counts: dict) -> MRResult:
    label = f"{spec['resource']} {spec['phenotype']}"
    notes = list(base_notes) + list(inst.notes)
    if meta:
        missing = [c for c in ("n_cases", "n_controls")
                   if (c not in rows.columns or rows.is_empty()
                       or rows[c].cast(pl.String).is_in(["NA", ""]).fill_null(True).all())
                   and meta.get(c) is not None]
        if missing and not rows.is_empty():
            rows = rows.with_columns(*(pl.lit(float(meta[c])).alias(c) for c in missing))
            notes.append(f"{label}: {', '.join(missing)} taken from phenotypes() (the rows carry "
                         "none)")

    frame = inst.frame
    if scans:
        frame = _proxies(frame, scans, lambda _variants: rows, notes)
        dup = frame["variant"].is_duplicated()
        if dup.any():
            notes.append(f"{label}: {int(dup.sum())} rows proxied to an already used variant; "
                         "the stronger lead's row is kept")
            frame = frame.unique(subset="variant", keep="first", maintain_order=True)
    if rows.is_empty():
        raise GeneticsUsageError(
            f"summary_stats returned no rows of {label} for the {frame.height} instrument(s) "
            "and their proxies; check the phenotype code (phenotypes(codes=[...], resource=...) "
            "lists the codes a resource serves)")
    h = harmonise(frame, rows, exposure_dataset=exposure_dataset, outcome_dataset=y_dataset,
                  **({"annotation": annotation} if annotation else {}))
    used = h.frame.filter(pl.col("in_outcome"))
    if used.is_empty():
        match = h.frame["outcome_match"].value_counts().sort("outcome_match").to_dicts()
        detail = ", ".join(f"{m['count']} {m['outcome_match']}" for m in match) or "none kept"
        raise GeneticsUsageError(
            f"no instrument of {exposure_label} is in {label} after the r2 >= {PROXY_R2:g} proxy "
            f"lookup: {h.frame.height} instrument(s), {detail}; {h.dropped.height} exposure "
            "row(s) unusable")
    notes += h.notes
    r = None
    if mode == "cis":
        proxy_cols = [c for c in ("proxy_for", "proxy_r", "is_proxy") if c in frame.columns]
        tagged = used.join(frame.select("variant", *proxy_cols), on="variant", how="left") \
            if proxy_cols else used
        r = _r_for(tagged, inst.r, scans)
        proxied = (tagged.filter(pl.col("proxy_for").is_not_null())["variant"].to_list()
                   if "proxy_for" in tagged.columns else [])
        if r is not None and proxied and used.height > 1:
            notes.append(
                f"LD r of the proxied instrument(s) {', '.join(proxied)} is approximated by the "
                "chain r(lead_a, lead_b) * r(proxy, its lead) (exact only as the proxy r2 "
                "approaches 1); a lead's own scan value is used where it holds the proxy")
    if mode == "cis" and r is None and used.height > 1:
        notes.append("no LD r for these instruments: the IVW is independent")
    estimates, headline = _estimate(used, mode, r, notes)
    if r is not None and used.height > 1:
        notes.append("leave-one-out, single-SNP, heterogeneity and Steiger treat the "
                     "instruments as independent (the LD r enters only the headline IVW and "
                     "Egger)")

    binary = "binary" in h.units.get("outcome", "")
    prevalence_source = "given" if prevalence is not None else None
    if binary and prevalence is None:
        nc, nn = _num(used, "n_cases"), _num(used, "n_controls")
        share = used.select((nc / (nc + nn)).mean()).item()
        if share is not None and math.isfinite(share):
            prevalence = float(share)
            prevalence_source = "sample"
            consortia = sorted(set((y_dataset or {}).get("cohorts") or [])
                               & _CASE_CONTROL_CONSORTIA)
            notes.append(
                f"Steiger prevalence {prevalence:.4g} is the sample's n_cases/(n_cases"
                "+n_controls), not the population prevalence: "
                + (f"{label} comes from a case-control consortium collection "
                   f"({', '.join(consortia)}) whose sample prevalence is far from the "
                   "population's; pass prevalence= for a meaningful Steiger" if consortia else
                   "close for a population cohort such as FinnGen, not for a case-control "
                   "consortium"))
    sensitivity = {
        "heterogeneity": _try(notes, "heterogeneity", lambda: heterogeneity(used))
        if used.height >= 2 else None,
        "leave_one_out": _try(notes, "leave-one-out", lambda: leave_one_out(used))
        if used.height >= 3 else None,
        "single_snp": _try(notes, "single-SNP", lambda: single_snp(used)),
        "steiger": _try(notes, "Steiger", lambda: steiger(
            replace(h, frame=used) if binary else used, prevalence=prevalence)),
    }
    if used.height < 3:
        notes.append(f"{used.height} instrument(s): leave-one-out needs 3"
                     + ("" if used.height >= 2 else ", heterogeneity 2"))
    checks_frame = checks(h, steiger=sensitivity["steiger"],
                          heterogeneity=sensitivity["heterogeneity"])
    overlap = sample_overlap_statement(exposure_dataset, y_dataset)

    coloc_frame = None
    if attach_coloc and not partnered:
        notes.append(f"colocalization: {label} is not a colocalization partner of the exposure "
                     "dataset in the registry; none attached")
    elif attach_coloc and coloc is not None and not coloc.is_empty():
        lead_of = dict(used.select("cs_id", "variant").drop_nulls("cs_id").iter_rows())
        x_first = pl.col("dataset1").is_in(_labels(exposure_dataset)) & (
            pl.col("trait1") == x_trait)
        y_side = pl.when(x_first).then(pl.col("trait2")).otherwise(pl.col("trait1"))
        y_set = pl.when(x_first).then(pl.col("dataset2")).otherwise(pl.col("dataset1"))
        y_labels = [meta["dataset"]] if meta and meta.get("dataset") else _labels(y_dataset)
        x_cs = pl.when(x_first).then(pl.col("cs1_id")).otherwise(pl.col("cs2_id"))
        got = coloc.filter((y_side == spec["phenotype"]) & y_set.is_in(y_labels)).with_columns(
            x_cs.replace_strict(lead_of, default=None).alias("instrument"))
        coloc_frame = got.filter(pl.col("instrument").is_not_null()) if got.height else None
        if coloc_frame is not None and coloc_frame.is_empty():
            coloc_frame = None
    if attach_coloc and partnered and coloc_frame is None:
        notes.append(f"colocalization: no precomputed row pairs an instrument's credible set "
                     f"with {label}")
    versions = {
        "exposure": {k: (exposure_dataset or {}).get(k) for k in ("dataset_id", "version")},
        "outcome": {k: (y_dataset or {}).get(k) for k in ("dataset_id", "version")},
        "outcome_rows": (sorted(rows["version"].drop_nulls().cast(pl.String).unique().to_list())
                         if "version" in rows.columns else None),
        "source": "genetics.datasets() version fields and the summary_stats rows' version "
                  "column; summary_stats does not surface results-api's X-Dataset-Version",
    }
    head = estimates[headline]
    params = {**parameters, "mode": mode, "prevalence": prevalence,
              "prevalence_source": prevalence_source,
              "model": head.get("model"), "headline": headline,
              "n_instruments": used.height}
    return MRResult(
        exposure=exposure_label, outcome=label, headline=headline, harmonised=h,
        instruments=replace(inst, frame=frame, notes=list(inst.notes)), estimates=estimates,
        sensitivity=sensitivity, checks=checks_frame, parameters=params, overlap=overlap,
        versions=versions, coloc=coloc_frame, notes=list(dict.fromkeys(notes)),
        requests=counts,
    )
