"""Mendelian randomization: harmonise an exposure and an outcome, and state the checks.

    import genetics
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

SAMPLE OVERLAP. A dataset's `cohorts` token names a sample collection, not a sample set. A
shared token says only "same collection(s), overlap fraction not recorded"; it never
establishes one-sample MR, and disjoint lists never establish independent samples, because
consortium tokens may contain FinnGen or UK Biobank individuals. The statement is a warning
at most, never a failure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import polars as pl

from genetics_mcp_server.sdk.errors import GeneticsUsageError
from genetics_mcp_server.sdk.plots import _CODING_CONSEQUENCES

__all__ = ["Harmonised", "harmonise", "checks", "sample_overlap_statement"]

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
