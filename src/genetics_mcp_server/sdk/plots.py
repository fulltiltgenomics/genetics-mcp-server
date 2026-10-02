"""Standard plots, drawn the same way every time.

    import genetics
    genetics.plots.locuszoom(phenotype="H8_HEARINGLOSS", variant="12:49578357:C:T")
    genetics.plots.phewas(variant="19:44908684:T:C")
    genetics.plots.upset(sets={"Crohn": cd_ids, "UC": uc_ids})
    genetics.plots.forest(frame, label="cohort", scale="log_ratio", pool="fixed")

WHY THESE ARE FUNCTIONS AND NOT INSTRUCTIONS. A locuszoom has conventions a script rederives
badly under time pressure: which axis is -log10 p, that the LD ramp is binned rather than
continuous, that the lead variant is a diamond, that genes belong under the association panel
and not beside it. Written out per request, each of those is a coin flip. Written here once,
they are the same in every conversation and a defect is fixed in one place.

WHY NOT A TOOL. A tool is a round trip with a fixed argument list; this is a Python function,
so a script can take the axes back and add to them, or call it for each of several phenotypes
in one execution and lay the results out itself. Every function here takes `ax=` for that
reason and returns what it drew rather than only a path.

STYLE IS MOSTLY NOT SET HERE. These draw under whatever rcParams are in force — matplotlib's
own defaults plus the render density the sandbox bakes (genetics-results-suite
sandbox/gen_mplrc.py) — so a caller who prefers another style sets it and these follow. Two
things are set anyway. The LD colour ramp and the two marker shapes, because both are
semantic: a reader decodes r² from the colours and consequence from the shapes, so neither
may follow a style's prop_cycle. And the type sizes and rule widths, because matplotlib's
defaults are sized for a figure twice as wide as the one these draw on and a caller who has
set no style should not have to correct for that.

ADDING A PLOT. Write the function, export it in `__all__`, and give it a docstring whose first
line reads as a description: `list_capabilities(module="plots")` and the generated
`sandbox/stubs/plots.pyi` both derive from this module, so nothing else has to be updated for
a script's author to discover it.
"""

from __future__ import annotations

import itertools
import math
import os
import re
import statistics
import textwrap
from typing import Any

import numpy as np
import polars as pl

from genetics_mcp_server.sdk.errors import GeneticsError, GeneticsUsageError

__all__ = ["locuszoom", "phewas", "upset", "linemodels", "forest"]

# The LocusZoom convention, and deliberately not the house style's prop_cycle: a reader decodes
# r² from these, so they are data encoding rather than decoration. Ordered high to low; the
# first threshold a variant's r² meets or exceeds wins.
_LD_BINS: tuple[tuple[float, str, str], ...] = (
    (0.8, "#D43F3A", "0.8–1.0"),
    (0.6, "#EEA236", "0.6–0.8"),
    (0.4, "#5CB85C", "0.4–0.6"),
    (0.2, "#46B8DA", "0.2–0.4"),
    (0.0, "#357EBD", "< 0.2"),
)
_LD_UNKNOWN = "#BBBBBB"
_LEAD_COLOUR = "#7D26CD"

# Shape carries consequence, colour carries LD: two channels, so a coding variant in high LD
# reads as both at once. The lead follows the same rule as everything else — it is marked out
# by size and colour, not by a third shape nothing else uses.
_MARKER_CODING = "s"
_MARKER_OTHER = "o"

# VEP terms that change the protein a transcript codes for. This is the suite's shared
# definition of "coding" and it is duplicated rather than shared because the other four copies
# are in another repo or another language: genetics-results-api app/config/common.py
# `coding_set`, genetics-results-browser src/utils/coding.ts and bff/coding.ts, and the chat
# prompt's Terminology block in config/defaults.py. Changing one means changing all five.
#
# TWO DELIBERATE EXCLUSIONS, both of which look like omissions. `synonymous_variant` sits in a
# coding sequence and leaves the protein identical, so a square would claim a protein effect
# the term denies; the same reasoning drops `start_retained_variant` and `stop_retained_variant`.
# `splice_region_variant` is excluded because VEP assigns it up to 8 bp into an intron, so a
# square there would claim a coding position the term does not establish — the two splice-site
# terms that abolish a site ARE here.
_CODING_CONSEQUENCES = frozenset({
    "frameshift_variant",
    "inframe_deletion",
    "inframe_insertion",
    "incomplete_terminal_codon_variant",
    "missense_variant",
    "protein_altering_variant",
    "splice_acceptor_variant",
    "splice_donor_variant",
    "start_lost",
    "stop_gained",
    "stop_lost",
    "transcript_ablation",
})

# the significance line and its label are scaffolding, not data: grey keeps them behind the
# points instead of competing with the LD ramp for the reader's attention
_SIGNIFICANCE_GREY = "#888888"
_WARNING_COLOUR = "#AA0000"

# Type sizes and rule widths, in points on the 6.5 in figure these draw by default. Set here
# rather than inherited: matplotlib's defaults are sized for a figure roughly twice this
# wide, and at this one they crowd a panel whose own annotations are 5-7 pt. Everything else
# still follows the caller's rcParams.
_TITLE_SIZE = 6
_LABEL_SIZE = 6
_TICK_SIZE = 6
_AXIS_LINEWIDTH = 0.5
_TICK_LENGTH = 2.0

# Asked of the LD server rather than 0.0. At r²≥0 it answers with every variant in the panel,
# which costs twice: the informative points disappear into a navy cloud of r²≈0, and the answer
# is truncated positionally. Measured at 12:49272869:C:T — a ±250 kb request came back with
# 3000 entries spanning 49,023,161–49,503,366 while the panel holds 3097 in that window, so the
# right-hand edge of the plot went grey with nothing to say why. At this floor the same locus
# returns 17 entries across ±500 kb: every point that carries colour, and no truncation.
# A variant below it is grey — not measured-and-low, but "not among the ones worth colouring",
# which is what grey already means for a variant the panel does not carry at all.
_LD_MIN_R2 = 0.05

# LD is asked for over this multiple of the plotted span so a correlated partner just outside
# the window can be named rather than silently omitted. Measured at the same locus: the
# strongest variant there (r²=0.78, and more significant than the lead) sits 292 kb away, so a
# ±250 kb plot drops the one point showing the signal is not a singleton.
_LD_SEARCH_SPAN_MULTIPLE = 2

# The LD server's own ceiling on `window`: above it the answer is HTTP 400 "window must be
# between 100000 and 5000000" and the whole figure goes grey. results-api holds the same
# number in app/config/ld.py; an upstream that changes its bounds falsifies both.
_LD_MAX_WINDOW = 5_000_000
_LD_MIN_WINDOW = 200_000

# Below the significance line, the strongest variant in a window is usually a rare one with
# an unstable estimate — measured: AF 5e-5 with beta 40 at CACNA1C, AF 0.00025 with beta 2.1
# at RBFOX1, both at p ~ 1e-4. Marked as the lead it is a caption for noise, and it is often
# absent from the LD panel, which costs the colours too. So a lead nobody chose has to earn
# it one of two ways: reach the line, or be at least this common.
_LEAD_MIN_MAF = 0.01

# how many `label_r2` neighbours are named. More than this and the labels are the figure.
_MAX_R2_LABELS = 8

# r² at which a partner outside the window is worth reporting. The note asks the reader to
# redraw at a wider window, so it sits where that is worth doing — a partner the ramp would
# have shown as strongly correlated — rather than at every partner the search span reaches.
_LD_NOTABLE_R2 = 0.6


def _artifacts_dir() -> str:
    """Where a sandbox execution's files are collected. Outside the sandbox, the cwd."""
    return os.environ.get("SANDBOX_ARTIFACTS_DIR") or "."


def _resolve_path(path: str | None, default: str) -> str:
    """Where the figure is written. A relative name lands in the artifacts directory.

    Not `path or <default>`: a relative `path=` used to be written to the process cwd, which
    the sandbox does not collect, so a correct call drew the figure and returned nothing. The
    caller's only clue was an empty artifact list, and the docstring promised the opposite.
    An absolute path is honoured as given.
    """
    if path is None:
        return os.path.join(_artifacts_dir(), default)
    if os.path.isabs(path):
        return path
    return os.path.join(_artifacts_dir(), path)


def _norm_chrom(value: Any) -> str:
    return str(value).strip().lower().removeprefix("chr")


def _variant_id(chrom: Any, pos: Any, ref: Any, alt: Any) -> str:
    return f"{_norm_chrom(chrom)}:{pos}:{str(ref).upper()}:{str(alt).upper()}"


def _norm_variant_id(value: Any) -> str:
    """chr:pos:ref:alt in one spelling, so ids from three sources can be compared.

    Not _norm_chrom applied to the whole string: that lowercases the ALLELES too, which turns
    every LD join into a miss and every plot grey. Only the chromosome field is case- and
    prefix-normalised; the alleles go upper, which is how both the sumstats files and the LD
    server write them.
    """
    parts = str(value).strip().split(":")
    if len(parts) != 4:
        return str(value).strip()
    chrom, pos, ref, alt = parts
    return _variant_id(chrom, pos, ref, alt)


def _format_p(mlog10p: float | None) -> str:
    """A p-value as a decimal string, taken apart from -log10(p) rather than computed.

    `10 ** -400` is 0.0 in a float, so a p-value that far down cannot be computed and then
    formatted — the lead of a strong locus would print as `0.0e+00`. Splitting the exponent
    from the mantissa keeps the digits: -log10(p) = 400.5 renders 3.2e-401 with no arithmetic
    that can underflow.
    """
    if mlog10p is None or not math.isfinite(mlog10p) or mlog10p <= 0:
        return "1"
    exponent = math.floor(mlog10p)
    mantissa = 10 ** (1 - (mlog10p - exponent))
    exponent += 1
    if mantissa >= 10:  # an integral -log10(p), e.g. 13.0 -> 10.0e-14
        mantissa /= 10
        exponent -= 1
    return f"{mantissa:.2f}e-{exponent}"


def _pretty_consequence(term: str | None) -> str | None:
    """`missense_variant` -> `missense`: a VEP term as a caption writes it.

    Display only. `lead_consequence` in the returned dict keeps the term verbatim, because
    that is the value a follow-up query filters on and a shortened one would not match.
    """
    if not term:
        return term
    return str(term).removesuffix("_variant").replace("_", " ") or str(term)


def _variant_head(variant_id: str, consequence: str | None, gene: str | None) -> str:
    """`19:44908684:T:C  APOE missense`, or the bare id when nothing is annotated."""
    term = _pretty_consequence(consequence)
    if not term:
        return variant_id
    return f"{variant_id}  {gene} {term}" if gene else f"{variant_id}  {term}"


def _af_columns(columns: list[str]) -> list[str]:
    """`af` first, then the per-cohort ones a meta-analysis carries instead (`fg_af`, ...)."""
    return [c for c in columns if c == "af"] + sorted(c for c in columns if c.endswith("_af"))


def _allele_frequency(row: dict[str, Any]) -> tuple[float, str] | None:
    """The row's allele frequency and the column it came from, or None when it has none."""
    for column in _af_columns(list(row)):
        value = row.get(column)
        if value is None:
            continue
        try:
            return float(value), column
        except (TypeError, ValueError):
            continue
    return None


def _default_lead(frame: pl.DataFrame, significance: float | None) -> dict[str, Any]:
    """The row a locuszoom leads with when the caller named none.

    The strongest variant, unless it is both below the significance line and rarer than
    _LEAD_MIN_MAF, in which case the strongest variant that is not rare. A variant with no
    frequency is not called rare, so a frame without any AF column behaves as it always did.
    """
    ranked = frame.sort("_y", descending=True)
    top = ranked.row(0, named=True)
    line = -math.log10(significance or 5e-8)
    if top["_y"] >= line:
        return top
    for row in ranked.iter_rows(named=True):
        found = _allele_frequency(row)
        if found is None or min(found[0], 1 - found[0]) >= _LEAD_MIN_MAF:
            return row
    return top


def _lead_label(
    variant_id: str,
    row: dict[str, Any],
    mlog10p: float,
    consequence: str | None = None,
    gene: str | None = None,
) -> str:
    """The id and its consequence, then whichever of p, beta and AF the frame carries.

    Built from what is present rather than from a fixed list, because `data=` may be any frame
    a caller assembled and a KeyError there would lose the whole figure for a caption.
    """
    head = _variant_head(variant_id, consequence, gene)
    parts = [f"p {_format_p(mlog10p)}"]
    beta = row.get("beta")
    if beta is not None:
        parts.append(f"beta {float(beta):.3g}")
    found = _allele_frequency(row)
    if found is not None:
        af, column = found
        # a meta-analysis row has one frequency per cohort and no pooled one, so the label
        # says whose it is
        source = "" if column == "af" else f" ({column.removesuffix('_af')})"
        parts.append(f"AF {af:.4g}{source}")
    return f"{head}\n" + "  ".join(parts)


# resolved once per process rather than per figure: a script that draws a panel per phenotype
# would otherwise refetch the whole schema for each one, and labels do not change under a
# running sandbox. A failed fetch is NOT cached — it falls back to the raw token for that call
# and tries again on the next.
_RESOURCE_LABELS: dict[str, str] | None = None


def _resource_label(resource: str) -> str:
    """The display name `configs/datasets.yaml` gives this resource, or the resource itself.

    Taken from the live schema rather than a map here: a table of resource names in this file
    is one more list to go stale the next time a resource is added.
    """
    global _RESOURCE_LABELS
    from genetics_mcp_server import sdk

    if _RESOURCE_LABELS is None:
        try:
            resources = sdk.schema().get("resources") or {}
        except Exception:
            return resource
        _RESOURCE_LABELS = {
            key: str(value.get("label") or key)
            for key, value in resources.items()
            if isinstance(value, dict)
        }
    return _RESOURCE_LABELS.get(resource, resource)


def _phenotype_name(phenotype: str) -> str | None:
    """The trait's human-readable name, or None when it cannot be resolved.

    None and the code itself are the same answer here — both mean "nothing to add to the
    title" — so an unresolved code degrades to the title this replaced rather than to a
    caption saying `Unknown: H8_HL_IDIOP`, which is what the upstream returns for one.
    """
    from genetics_mcp_server import sdk

    try:
        frame = sdk.lookup_phenotype_names(phenotype)
    except Exception:
        return None
    if frame.is_empty() or "name" not in frame.columns:
        return None
    name = frame["name"][0]
    if not name or str(name).startswith("Unknown:") or str(name) == phenotype:
        return None
    return str(name)


def _default_title(phenotype: str, region: str, frame: pl.DataFrame, resource: str) -> str:
    """`Sudden idiopathic hearing loss (H8_HL_IDIOP, FinnGen R14) — 12:49022869-49522869`.

    The code stays in the title even when the name resolves, because it is what a follow-up
    query takes and the name is not. Release and resource are read off the frame rather than
    off the arguments, so a caller who passed `data=` gets the label of the data they actually
    plotted instead of this function's `resource` default.

    Every part is optional and the title degrades one piece at a time: no name leaves the code
    in front, no release leaves the resource alone, and neither leaves `CODE — region`.
    """
    def first(column: str) -> str | None:
        if column not in frame.columns or frame.height == 0:
            return None
        value = frame[column][0]
        return str(value) if value else None

    name = _phenotype_name(phenotype)
    source = " ".join(
        part for part in (_resource_label(first("resource") or resource), first("version"))
        if part
    )
    inside = ", ".join(part for part in (phenotype if name else None, source) if part)
    head = name or phenotype
    return f"{head} ({inside}) — {region}" if inside else f"{head} — {region}"


def _consequences(region: str) -> dict[str, tuple[str | None, str | None]] | None:
    """variant id -> (most_severe, gene_most_severe), or None when the lookup did not answer.

    The consequence is not in the summary statistics, so it is a second fetch. Failure is
    tolerated the same way the LD fetch is — the figure is still the right picture of the
    locus without it. `None` and `{}` are deliberately different answers: an empty mapping is
    a region with no annotation, `None` is a lookup that failed, and only the first may be
    reported to the caller as "nothing here is coding".
    """
    from genetics_mcp_server import sdk

    try:
        annotations = sdk.variant_annotation(region=region)
    except Exception:
        return None
    if annotations.is_empty() or not {"variant", "most_severe"} <= set(annotations.columns):
        return None
    return {
        _norm_variant_id(row["variant"]): (
            row.get("most_severe"), row.get("gene_most_severe")
        )
        for row in annotations.iter_rows(named=True)
    }


def _variant_pos(value: Any) -> int | None:
    parts = str(value).split(":")
    if len(parts) < 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def _partners_outside(
    ld_frame: pl.DataFrame | None, lo: int, hi: int
) -> list[dict[str, Any]]:
    """LD partners the plotted window excludes, strongest first.

    A locuszoom is read as "this is the locus", so a correlated variant just past the edge is
    not a missing detail — it is the difference between a lone signal and a supported one, and
    the window is a default nobody chose per locus.
    """
    if ld_frame is None or ld_frame.is_empty() or "variant" not in ld_frame.columns:
        return []
    found = []
    for row in ld_frame.iter_rows(named=True):
        variant, r2 = row.get("variant"), row.get("r2")
        if variant is None or r2 is None or float(r2) < _LD_NOTABLE_R2:
            continue
        pos = _variant_pos(variant)
        if pos is None or lo <= pos <= hi:
            continue
        found.append({"variant": _norm_variant_id(variant), "pos": pos, "r2": float(r2)})
    found.sort(key=lambda partner: partner["r2"], reverse=True)
    return found


def _region_from_variant(variant: str, flank: int) -> tuple[str, str]:
    parts = str(variant).split(":")
    if len(parts) != 4:
        raise GeneticsUsageError(
            f"variant {variant!r} is not chr:pos:ref:alt, so no region can be centred on it"
        )
    chrom, pos = parts[0], parts[1]
    try:
        centre = int(pos)
    except ValueError:
        raise GeneticsUsageError(f"variant {variant!r} has a non-integer position")
    start = max(1, centre - flank)
    return f"{_norm_chrom(chrom)}:{start}-{centre + flank}", _norm_chrom(chrom)


def _mlog10p(frame: pl.DataFrame) -> pl.Series:
    """-log10(p) from whichever of the two columns the resource actually returned.

    `mlog10p` is preferred where present and not merely as a shortcut: a p-value that has
    underflowed to 0.0 in the file still has a finite mlog10p, and -log10(0) is inf, which
    matplotlib drops from the axis limits and draws off the top of the panel.
    """
    if "mlog10p" in frame.columns:
        series = frame["mlog10p"]
        if series.null_count() < frame.height:
            return series.fill_null(0.0)
    if "pval" not in frame.columns:
        raise GeneticsUsageError(
            "summary statistics carry neither `mlog10p` nor `pval`, so there is nothing to "
            f"plot on the y axis; columns are {frame.columns}"
        )
    return (
        frame["pval"]
        .cast(pl.Float64, strict=False)
        # 0.0 would give inf; the smallest positive double is the honest floor
        .map_elements(
            lambda p: None if p is None else -math.log10(max(p, 5e-324)),
            return_dtype=pl.Float64,
        )
        .fill_null(0.0)
    )


def _ld_colours(frame: pl.DataFrame, lead_id: str, ld_frame: pl.DataFrame | None):
    """One colour per row, plus the r² used, plus whether any LD was actually joined."""
    r2_by_id: dict[str, float] = {}
    if ld_frame is not None and not ld_frame.is_empty() and "variant" in ld_frame.columns:
        for row in ld_frame.iter_rows(named=True):
            variant = row.get("variant")
            r2 = row.get("r2")
            if variant is None or r2 is None:
                continue
            r2_by_id[_norm_variant_id(variant)] = float(r2)

    colours, values = [], []
    for vid in frame["_variant_id"]:
        if vid == lead_id:
            colours.append(_LEAD_COLOUR)
            values.append(1.0)
            continue
        r2 = r2_by_id.get(vid)
        if r2 is None:
            colours.append(_LD_UNKNOWN)
            values.append(None)
            continue
        for threshold, colour, _label in _LD_BINS:
            if r2 >= threshold:
                colours.append(colour)
                break
        else:  # pragma: no cover - _LD_BINS ends at 0.0, so this cannot be reached
            colours.append(_LD_UNKNOWN)
        values.append(r2)
    return colours, values, bool(r2_by_id)


# A gene model reads by thickness: the body is a hairline, an exon is a bar, and the
# translated part of that exon is thicker still. Widths are in points rather than data
# units, so the bars keep their proportions whatever span the window covers.
_GENE_BODY_WIDTH = 0.7
_GENE_EXON_WIDTH = 2.6
_GENE_CDS_WIDTH = 4.4
_GENE_COLOUR = "#4A4A4A"


def _exon_spans(gene: dict[str, Any]) -> list[tuple[Any, Any, Any, Any]]:
    """(exon_start, exon_end, cds_start, cds_end) per exon, empty when there is no structure.

    The API's four arrays are positional and equal length, so an exon's coding bounds sit at
    the same index as the exon; a GENCODE release with no exon file returns all four empty,
    and a results-api too old to serve them returns none of the columns at all. Both arrive
    here as no exons, which draws the bare body rather than failing.
    """
    starts, ends = gene.get("exon_starts"), gene.get("exon_ends")
    if not starts or not ends or len(starts) != len(ends):
        return []
    cds_starts = gene.get("cds_starts") or [None] * len(starts)
    cds_ends = gene.get("cds_ends") or [None] * len(starts)
    if len(cds_starts) != len(starts) or len(cds_ends) != len(starts):
        cds_starts = cds_ends = [None] * len(starts)
    return list(zip(starts, ends, cds_starts, cds_ends))


def _bar(ax, lo, hi, row: int, width: float, start: int, end: int) -> None:
    """One segment of a gene model, clipped to the window and skipped when outside it."""
    if lo is None or hi is None:
        return
    lo, hi = max(lo, start), min(hi, end)
    if lo > hi:
        return
    ax.plot([lo, hi], [-row, -row], linewidth=width, solid_capstyle="butt",
            color=_GENE_COLOUR, zorder=2)


def _gene_label(gene: dict[str, Any]) -> str | None:
    """The symbol to draw, or None for a gene the track should leave out.

    GENCODE names some protein-coding genes by their ENSG alone. An ENSG on a locus plot is
    a row of characters nobody can look up, and it crowds a track that is already packing
    genes into four rows, so those are dropped rather than drawn nameless. A few of them do
    carry an HGNC symbol, which is why that is consulted rather than assumed absent.
    """
    for candidate in (gene.get("gene_name"), gene.get("hgnc_symbol")):
        if candidate and not str(candidate).startswith("ENSG"):
            return str(candidate)
    return None


def _drawable(gene: dict[str, Any]) -> tuple[str, int, int, list, str] | None:
    """One gene reduced to what the track draws, or None if it cannot be drawn.

    THE SPAN IS THE TRANSCRIPT'S, NOT THE GENE RECORD'S, and that is the whole point of this
    function. The exons belong to one transcript while a GENCODE gene record spans every
    transcript it has, and the two disagree badly: measured on v49, the canonical transcript
    covers under a tenth of the gene record for 185 protein-coding genes and under a quarter
    for 641 — TUBA1C's record runs 86 kb while its MANE transcript is 9.5 kb. Drawing the
    record's span put four exons in the right-hand tenth of a long bare line, which reads as
    exons in the wrong place. A gene with no exon structure has nothing but the record to
    draw, so it keeps it.
    """
    name = _gene_label(gene)
    if name is None:
        return None
    spans = _exon_spans(gene)
    if spans:
        g_start = min(exon_start for exon_start, _e, _cs, _ce in spans)
        g_end = max(exon_end for _s, exon_end, _cs, _ce in spans)
    else:
        g_start, g_end = gene.get("gene_start"), gene.get("gene_end")
    if g_start is None or g_end is None:
        return None
    return name, g_start, g_end, spans, (gene.get("gene_strand") or "").strip()


def _draw_genes(
    ax, genes: pl.DataFrame, start: int, end: int, max_rows: int = 4
) -> tuple[int, int]:
    """Gene models on a small track, packed into non-overlapping rows.

    Returns (genes drawn, exons drawn). The second is 0 when the API served no exon
    structure, which is what tells the caller the track is bodies only. Genes GENCODE names
    only by an ENSG are left out entirely, so the first can be short of the number of genes
    in the window.
    """
    # no frame and no scale: the position axis is the association panel's, drawn directly
    # above, and a box around the models reads as a second plot rather than as a strip of one
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(bottom=False, labelbottom=False)
    ax.set_yticks([])
    if genes.is_empty() or "gene_start" not in genes.columns:
        return 0, 0
    # sorted by what is actually drawn, so the row packing below sees the same spans the
    # reader does
    # A gene whose drawn span misses the window is left out: the API returns a gene whose
    # RECORD overlaps the region, and its canonical transcript can lie wholly outside, where
    # its label would float past the axes with no model under it.
    drawable = sorted(
        (
            d for d in (_drawable(g) for g in genes.iter_rows(named=True))
            if d and d[2] >= start and d[1] <= end
        ),
        key=lambda d: d[1],
    )
    row_ends: list[int] = []
    drawn = 0
    exons_drawn = 0
    span = max(end - start, 1)
    for name, g_start, g_end, spans, strand in drawable:
        # pack: first row whose last gene ends before this one starts, with a gap for the label
        gap = span * 0.06
        for row, occupied_to in enumerate(row_ends):
            if g_start > occupied_to + gap:
                break
        else:
            row = len(row_ends)
            if row >= max_rows:
                continue
            row_ends.append(0)
        row_ends[row] = g_end
        # clipped to the window: a gene overlapping the boundary is returned whole, and on a
        # shared x axis its far end drags the association panel's limits out with it
        left, right = max(g_start, start), min(g_end, end)
        _bar(ax, left, right, row, _GENE_BODY_WIDTH, start, end)
        for exon_start, exon_end, cds_start, cds_end in spans:
            _bar(ax, exon_start, exon_end, row, _GENE_EXON_WIDTH, start, end)
            _bar(ax, cds_start, cds_end, row, _GENE_CDS_WIDTH, start, end)
            exons_drawn += 1
        label = f"{name}{'→' if strand == '+' else '←' if strand == '-' else ''}"
        ax.text((left + right) / 2, -row + 0.22, label, ha="center", va="bottom",
                fontsize=5, color="#222222")
        drawn += 1
    ax.set_ylim(-max(len(row_ends), 1) + 0.2, 0.9)
    return drawn, exons_drawn

def _significance_line(ax, significance: float, *, label_right: bool = False) -> float:
    """The significance line and its label; returns the line's y.

    The label sits on the line rather than in a legend box: grey scaffolding named where it
    is, so the figure carries no legend unless something else needs one. Returns 0.0 and
    draws nothing when `significance` is falsy.

    `label_right` is for a panel with a legend in its upper left corner, which covers a
    left-hand label whenever nothing in the window clears the line. There the label is also
    backed: on a tall axis the line runs through the cloud at the foot of the panel, and a
    bare label is lost in it.
    """
    if not significance:
        return 0.0
    line_y = -math.log10(significance)
    ax.axhline(line_y, color=_SIGNIFICANCE_GREY, linewidth=0.6, linestyle="--", zorder=1)
    backing = (
        {"bbox": {"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.6},
         "zorder": 4}
        if label_right else {}
    )
    # `:g` renders 5e-8 as "5e-08"; the padded exponent is not how anyone writes it
    ax.text(0.994 if label_right else 0.006, line_y,
            f"p {significance:g}".replace("e-0", "e-"),
            transform=ax.get_yaxis_transform(), ha="right" if label_right else "left",
            va="bottom", fontsize=6, color=_SIGNIFICANCE_GREY, **backing)
    return line_y


def _size(ax) -> None:
    """The tick type size and the rule widths every one of these plots shares."""
    ax.tick_params(labelsize=_TICK_SIZE, width=_AXIS_LINEWIDTH, length=_TICK_LENGTH)
    for spine in ax.spines.values():
        spine.set_linewidth(_AXIS_LINEWIDTH)


def _dress(ax, title: str) -> None:
    """The y label, the title and the type sizes an association panel shares."""
    ax.set_ylabel(r"$-\log_{10}(p)$", fontsize=_LABEL_SIZE)
    ax.set_title(title, fontsize=_TITLE_SIZE)
    _size(ax)


def _mb_axis(ax) -> None:
    """Positions in Mb, with as many decimals as the tick step needs and no more.

    The default formatter prints base pairs against an offset (`1.5915 ... 1e8`), which a
    reader has to add up to find a position.
    """
    from matplotlib.ticker import FuncFormatter

    def fmt(value: float, _pos: Any) -> str:
        locs = sorted(ax.xaxis.get_majorticklocs())
        step = min((b - a for a, b in zip(locs, locs[1:])), default=1e6) / 1e6
        decimals = next(
            (d for d in range(7) if abs(round(step, d) - step) < 1e-9), 6
        )
        return f"{value / 1e6:.{decimals}f}"

    ax.xaxis.set_major_formatter(FuncFormatter(fmt))


def _track_symbol(symbol: str | None, genes: pl.DataFrame | None) -> str | None:
    """The name the gene track draws for a gene the annotation calls `symbol`.

    The two come from different releases: the consequence annotation still says MINOS1 where
    GENCODE says MICOS10, and a label in the annotation's spelling names a gene the track
    under it does not have.
    Only a previous symbol of exactly one gene in the window is rewritten; an alias is not
    consulted, because aliases are shared between genes.
    """
    if not symbol or genes is None or genes.is_empty() or "gene_name" not in genes.columns:
        return symbol
    names = set(genes["gene_name"].to_list())
    if symbol in names or "hgnc_prev_symbol" not in genes.columns:
        return symbol
    renamed = [
        gene for gene in genes.iter_rows(named=True)
        if symbol in str(gene.get("hgnc_prev_symbol") or "").split("|")
    ]
    if len(renamed) != 1:
        return symbol
    return _gene_label(renamed[0]) or symbol


def _renderer(figure):
    """The figure's renderer after a layout pass, or None where the backend has none.

    Text extents and the data transform are only final once the layout has run, and both
    are what the title fit and the label placement measure against.
    """
    try:
        figure.draw_without_rendering()
        return figure.canvas.get_renderer()
    except Exception:
        return None


def _fit_title(ax, figure, own_figure: bool) -> None:
    """Break a title that is wider than what it sits over, rather than let it be clipped.

    A long phenotype name makes the default title wider than the figure. The break goes
    at the dash before the region where that leaves two lines that fit, else wherever
    textwrap puts it.
    """
    renderer = _renderer(figure)
    title = ax.title.get_text()
    if renderer is None or not title:
        return
    available = (figure.bbox if own_figure else ax.bbox).width
    width = ax.title.get_window_extent(renderer).width
    if width <= available:
        return
    per_line = max(int(len(title) * available / width) - 2, 20)
    head, dash, tail = title.rpartition(" \u2014 ")
    if dash and max(len(head), len(tail)) <= per_line:
        ax.set_title(f"{head}\n{tail}", fontsize=_TITLE_SIZE)
    else:
        ax.set_title(textwrap.fill(title, per_line), fontsize=_TITLE_SIZE)


def _label_spots() -> list[tuple[float, float, str, str]]:
    """Where a label may sit relative to its point, in points, nearest first: (dx, dy, ha, va).

    Below comes first because the strongest association is at the top of the panel, where a
    label above it runs into the frame. The rings further out are for a crowded peak, where
    several labelled points sit within a few points of each other and each needs a leader
    line to somewhere clear.
    """
    spots: list[tuple[float, float, str, str]] = [
        (0, -9, "center", "top"), (9, 0, "left", "center"), (-9, 0, "right", "center"),
        (9, -9, "left", "top"), (-9, -9, "right", "top"), (0, 9, "center", "bottom"),
        (9, 9, "left", "bottom"), (-9, 9, "right", "bottom"),
    ]
    for radius in (28, 50, 75, 105):
        for degrees in (-30, -150, 0, 180, -60, -120, 30, 150, 60, 120):
            dx = radius * math.cos(math.radians(degrees))
            dy = radius * math.sin(math.radians(degrees))
            va = "center" if degrees in (0, 180) else "top" if dy < 0 else "bottom"
            spots.append((dx, dy, "left" if dx > 0 else "right", va))
    return spots


_LABEL_SPOTS = _label_spots()
_LEADER_FROM = 10  # points of offset beyond which a label gets a line back to its point


def _place_labels(ax, renderer, labels: list[tuple[str, float, float, float]],
                  points: np.ndarray, taken: list) -> None:
    """Annotate each (text, x, y, fontsize) where it covers the fewest points.

    At a fixed offset the lead's caption lands on its own LD partners at a dense locus, and
    across the y axis when the lead sits at the window's edge. Each label tries the spots in
    _LABEL_SPOTS and keeps the first that is inside the axes, clear of the legend, the corner
    notes and the labels already placed, and over no points; failing that, the least bad.
    Without a renderer nothing can be measured and every label goes below its point.
    """
    from matplotlib.text import Text

    frame = ax.bbox
    for text, x, y, fontsize in labels:
        note = ax.annotate(
            text, (x, y), textcoords="offset points", xytext=_LABEL_SPOTS[0][:2],
            ha=_LABEL_SPOTS[0][2], va=_LABEL_SPOTS[0][3], fontsize=fontsize, zorder=5,
            arrowprops={"arrowstyle": "-", "linewidth": 0.4, "color": "#555555",
                        "shrinkA": 0, "shrinkB": 3},
        )
        note.arrow_patch.set_visible(False)
        if renderer is None:
            continue
        best = None
        for index, (dx, dy, ha, va) in enumerate(_LABEL_SPOTS):
            note.xyann = (dx, dy)
            note.set_ha(ha)
            note.set_va(va)
            # the text alone: an annotation's own extent takes in its leader line, which
            # reaches back to the point and so always "covers" it
            note.update_positions(renderer)
            box = Text.get_window_extent(note, renderer)
            inside = (box.x0 >= frame.x0 and box.x1 <= frame.x1
                      and box.y0 >= frame.y0 and box.y1 <= frame.y1)
            covered = int((
                (points[:, 0] >= box.x0 - 2) & (points[:, 0] <= box.x1 + 2)
                & (points[:, 1] >= box.y0 - 2) & (points[:, 1] <= box.y1 + 2)
            ).sum()) if len(points) else 0
            clashes = sum(1 for other in taken if box.overlaps(other))
            cost = (0 if inside else 10_000) + 500 * clashes + covered
            if best is None or cost < best[0]:
                best = (cost, index, box)
            if cost == 0:
                break
        _cost, index, box = best
        dx, dy, ha, va = _LABEL_SPOTS[index]
        note.xyann = (dx, dy)
        note.set_ha(ha)
        note.set_va(va)
        note.arrow_patch.set_visible(math.hypot(dx, dy) > _LEADER_FROM)
        taken.append(box)


def locuszoom(
    *,
    phenotype: str,
    region: str | None = None,
    variant: str | None = None,
    flank: int = 250_000,
    resource: str = "finngen",
    data_type: str = "gwas",
    lead: str | None = None,
    ld: bool = True,
    ld_panel: str = "sisu42",
    genes: bool = True,
    coding: bool = True,
    highlight: str | list[str] | None = None,
    label_r2: float | None = None,
    data: pl.DataFrame | None = None,
    path: str | None = None,
    title: str | None = None,
    # genome-wide significance, the line every GWAS figure carries. A literal rather than a
    # module constant because the generated stub renders the signature verbatim, and a name
    # it cannot resolve is worse for the reader than the number.
    significance: float = 5e-8,
    ax: Any = None,
) -> dict[str, Any]:
    """Regional association plot: -log10 p against position, coloured by LD with the lead.

    Give either `region` ("12:49400000-49800000") or `variant` ("12:49578357:C:T"), which is
    centred with `flank` either side. LD is taken against `lead` from the FinnGen LD server.

    `lead` defaults to the strongest association in the window, with one exception: when
    nothing reaches `significance`, a variant rarer than 1% is passed over for the strongest
    one that is not, because the top of a flat window is usually a rare variant with an
    unstable estimate. `strongest` in the returned dict is the plain maximum either way, so
    the two differ exactly when that happened. A lead below the line is the top of the
    noise, not a hit, and should be described that way.

    `highlight` rings and names further variants (ids as chr:pos:ref:alt), and `label_r2`
    names the lead's neighbours at or above that r² — `label_r2=0.8` is "label the top
    variant and everything in tight LD with it". With `variant=`, the centred variant is
    ringed and named on its own whenever it is not the lead. Labels use the rsID where the
    summary statistics carry one. `highlighted` in the returned dict lists what was named,
    as {variant, mlog10p, r2}.

    THE DEFAULT WINDOW IS THE RIGHT ONE UNLESS THE QUESTION IS ABOUT THE WINDOW. `flank` is
    250 kb either side, i.e. a 500 kb plot; pass `variant=` and leave it alone. Widening it as
    a matter of course spreads the signal over a picture that is mostly empty, and a locus
    that genuinely needs more says so — `ld_partners_outside_window` names the correlated
    variants the window excluded, and that is the signal to redraw wider.

    Colour is LD and shape is consequence, so a coding variant in high LD reads as both at
    once: a square sits in a coding sequence or changes the protein, a circle does not, and
    the lead takes whichever shape its own consequence gives it. Grey means "no r² to show" —
    either the LD panel does not carry the variant or its r² is below the floor these plots
    colour at. Only the correlated variants are coloured, so the ramp reads at a glance
    instead of painting the whole cloud navy.

    The gene track draws one model per gene: a hairline over the transcript, a bar per exon
    of it, and a thicker bar over the part of each exon that is translated, so an
    untranslated leading or trailing exon reads as such. The transcript is GENCODE's
    Ensembl-canonical one, and the hairline spans IT rather than the gene record, which can
    be many times longer where a gene has transcripts the canonical one does not reach.
    Genes GENCODE names only by an ENSG are left out. `n_exons` in the returned dict is 0
    when the API served no exon structure, in which case the track is gene bodies only.

    Returns a dict describing what was drawn: `path`, `lead`, `lead_mlog10p`, `region`,
    `phenotype`, `n_variants`, `n_genes`, `n_exons`, plus the ones below, worth reading every
    time.

    `ld_joined` is False when there is no r² to colour by — a plot with grey points rather
    than an error, because a locuszoom without LD is still the right picture of the locus.
    `ld_status` says why, and the figure carries the same reason. Report the one that
    applies rather than guessing at an outage:
    "joined" — coloured as asked;
    "partial" — the plot is wider than the LD server's 5 Mb window, so points more than
    2.5 Mb from the lead are grey because they were never asked about;
    "no_partners" — the panel carries the lead and nothing is correlated with it;
    "lead_not_in_panel" — the panel does not carry the lead, which retrying will not change
    (pass a `lead=` the panel has, or another `ld_panel`);
    "unavailable" — the LD server failed, the one case worth retrying later;
    "off" — `ld=False`.

    `ld_partners_outside_window` lists the variants correlated with the lead that fall
    outside the window, strongest first, as {variant, pos, r2}. It is non-empty when the
    window is too narrow for the locus — the signal has support the plot does not show — and
    the fix is to redraw with a larger `flank` or an explicit `region`. The figure carries
    the same warning so a reader who never sees this dict is not misled.

    `coding_marked` is False when the consequence lookup did not answer, in which case every
    point is a circle and shape means nothing; set `coding=False` to skip that fetch outright.
    The same lookup fills `lead_consequence` and `lead_gene`, which the lead's label also
    carries, so the lead names what it does and where before anyone asks. `lead_gene` is the
    annotation's symbol, the one a follow-up query filters on; `lead_gene_label` is what the
    figure prints, which differs where GENCODE has since renamed the gene.

    `path` may be relative, in which case it is written inside the execution's artifacts
    directory and returned to the user automatically; that is also where the default goes.
    Pass `ax` to draw into an existing axis instead, in which case no gene track is added and
    nothing is saved.
    """
    import matplotlib.pyplot as plt

    from genetics_mcp_server import sdk

    if not region and not variant:
        # required even with data=: the window is what the title, the x axis and the returned
        # dict all name, and a frame does not carry the window it was drawn from
        raise GeneticsUsageError("locuszoom needs region= or variant=")
    if region and variant:
        raise GeneticsUsageError(
            "give region= or variant=, not both; a variant is only a way to centre a region"
        )

    chrom = None
    if variant and not region:
        region, chrom = _region_from_variant(variant, flank)

    frame = data if data is not None else sdk.summary_stats(
        phenotypes=phenotype, region=region, resource=resource, data_type=data_type
    )
    if frame.is_empty():
        raise GeneticsUsageError(
            f"no summary statistics for {phenotype!r} in {region} "
            f"(resource={resource!r}, data_type={data_type!r}) — nothing to plot"
        )
    for needed in ("pos",):
        if needed not in frame.columns:
            raise GeneticsUsageError(
                f"summary statistics have no {needed!r} column; got {frame.columns}"
            )

    frame = frame.with_columns(_mlog10p(frame).alias("_y"))
    if {"chr", "ref", "alt"} <= set(frame.columns):
        frame = frame.with_columns(
            pl.struct(["chr", "pos", "ref", "alt"])
            .map_elements(
                lambda r: _variant_id(r["chr"], r["pos"], r["ref"], r["alt"]),
                return_dtype=pl.Utf8,
            )
            .alias("_variant_id")
        )
    else:
        frame = frame.with_columns(pl.lit(None, dtype=pl.Utf8).alias("_variant_id"))

    def row_id(row: dict[str, Any]) -> str:
        return row["_variant_id"] or f"{row.get('chr', chrom)}:{row['pos']}"

    strongest = row_id(frame.sort("_y", descending=True).row(0, named=True))
    if lead is None:
        lead_row = _default_lead(frame, significance)
        lead = row_id(lead_row)
    else:
        match = frame.filter(pl.col("_variant_id") == _norm_variant_id(lead))
        if match.is_empty():
            raise GeneticsUsageError(f"lead {lead!r} is not among the variants in {region}")
        lead_row = match.row(0, named=True)
    lead_pos, lead_y = lead_row["pos"], lead_row["_y"]
    lead_id = _norm_variant_id(lead)

    if label_r2 is not None and not 0 < label_r2 <= 1:
        raise GeneticsUsageError(f"label_r2 is an r² threshold in (0, 1], got {label_r2!r}")
    known_ids = set(frame["_variant_id"].drop_nulls().to_list())
    marked: list[str] = []
    centre = _norm_variant_id(variant) if variant else None
    if centre and centre != lead_id and centre in known_ids:
        marked.append(centre)
    for wanted in [highlight] if isinstance(highlight, str) else list(highlight or []):
        wanted_id = _norm_variant_id(wanted)
        if wanted_id not in known_ids:
            raise GeneticsUsageError(
                f"highlight {wanted!r} is not among the variants in {region}"
            )
        if wanted_id != lead_id and wanted_id not in marked:
            marked.append(wanted_id)

    # the window the DATA covers. Computed here rather than at plotting time because the LD
    # request is sized from it: `flank` is meaningless when the caller gave region=, and the
    # old `2 * flank` asked for the default width regardless of what was actually drawn.
    span_lo, span_hi = int(frame["pos"].min()), int(frame["pos"].max())

    ld_frame = None
    ld_failure = None
    outside: list[dict[str, Any]] = []
    # the server's `window` is the TOTAL span it centres on the lead, so this is
    # _LD_SEARCH_SPAN_MULTIPLE times the plotted width — wide enough to cover the window from
    # a lead anywhere inside it, and to see just past both edges — up to the server's ceiling
    ld_window = min(
        max(_LD_SEARCH_SPAN_MULTIPLE * max(span_hi - span_lo, 1), _LD_MIN_WINDOW),
        _LD_MAX_WINDOW,
    )
    if ld:
        try:
            ld_frame = sdk.ld(
                lead_id, window=ld_window, r2_threshold=_LD_MIN_R2, panel=ld_panel
            )
        except GeneticsError as exc:
            # neither loses the figure, but they are different answers: a lead the panel does
            # not carry stays that way, and only a failed server is worth a retry
            ld_failure = (
                "lead_not_in_panel" if exc.code == "ld_variant_not_in_panel" else "unavailable"
            )
        except Exception:
            ld_failure = "unavailable"
        else:
            outside = _partners_outside(ld_frame, span_lo, span_hi)
    colours, r2_values, ld_joined = _ld_colours(frame, lead_id, ld_frame)
    ld_reach = ld_window // 2
    if not ld:
        ld_status = "off"
    elif ld_failure:
        ld_status = ld_failure
    elif not ld_joined:
        ld_status = "no_partners"
    elif lead_pos - ld_reach > span_lo or lead_pos + ld_reach < span_hi:
        ld_status = "partial"
    else:
        ld_status = "joined"
    frame = frame.with_columns(pl.Series("_r2", r2_values, dtype=pl.Float64))

    consequences = _consequences(region) if coding else None
    coding_marked = consequences is not None
    coding_flags = [
        consequences.get(vid, (None, None))[0] in _CODING_CONSEQUENCES if consequences else False
        for vid in frame["_variant_id"]
    ]
    lead_consequence, lead_gene = (consequences or {}).get(lead_id, (None, None))

    named = list(marked)
    if label_r2 is not None:
        partners = frame.filter(
            (pl.col("_r2") >= label_r2) & (pl.col("_variant_id") != lead_id)
        ).sort("_y", descending=True)
        named += [
            vid for vid in partners["_variant_id"].head(_MAX_R2_LABELS) if vid not in named
        ]

    own_figure = ax is None
    gene_frame = None
    if own_figure:
        want_genes = genes
        if want_genes:
            try:
                gene_frame = sdk.gene_annotations(region=region)
            except Exception:
                gene_frame = None
            want_genes = gene_frame is not None and not gene_frame.is_empty()
        if want_genes:
            figure, (ax, gene_ax) = plt.subplots(
                2, 1, sharex=True, height_ratios=[4, 1],
                figsize=(6.5, 4.2), constrained_layout=True,
            )
        else:
            figure, ax = plt.subplots(figsize=(6.5, 3.4), constrained_layout=True)
            gene_ax = None
    else:
        figure, gene_ax = ax.get_figure(), None

    # one scatter per shape: matplotlib takes a single marker per call, and the shape is what
    # separates a coding variant from the rest
    positions, ys = list(frame["pos"]), list(frame["_y"])
    for marker, wanted in ((_MARKER_OTHER, False), (_MARKER_CODING, True)):
        rows = [i for i, is_coding in enumerate(coding_flags) if is_coding is wanted]
        if not rows:
            continue
        ax.scatter([positions[i] for i in rows], [ys[i] for i in rows],
                   c=[colours[i] for i in rows], marker=marker, s=9, linewidths=0.2,
                   edgecolors="#33333355", zorder=2)

    index_of = {vid: i for i, vid in enumerate(frame["_variant_id"]) if vid is not None}
    lead_index = index_of.get(lead_id)
    lead_coding = bool(lead_index is not None and coding_flags[lead_index])
    ax.scatter([lead_pos], [lead_y], marker=_MARKER_CODING if lead_coding else _MARKER_OTHER,
               s=40, c=_LEAD_COLOUR, edgecolors="black", linewidths=0.4, zorder=3)
    for vid in marked:
        # a ring, so the point keeps the colour and shape it already carries
        i = index_of[vid]
        ax.scatter([positions[i]], [ys[i]], s=42, facecolors="none", edgecolors="black",
                   linewidths=0.7, zorder=3,
                   marker=_MARKER_CODING if coding_flags[i] else _MARKER_OTHER)
    line_y = _significance_line(ax, significance, label_right=True)

    ld_note = {
        "lead_not_in_panel": f"no LD: {lead_id} is not in the {ld_panel} panel",
        "unavailable": "no LD: the LD server did not answer",
        "no_partners": rf"nothing at r$^2\geq${_LD_MIN_R2:g} with {lead_id} in {ld_panel}",
    }.get(ld_status)
    if ld_note:
        # where the legend would have been, so grey points say why they are grey
        ax.text(0.006, 0.99, ld_note, transform=ax.transAxes, ha="left", va="top",
                fontsize=5, color=_SIGNIFICANCE_GREY)

    if outside:
        # on the figure and not only in the returned dict: the figure is what reaches the
        # reader, and a plot that silently omits the locus's best-correlated variant is wrong
        # in the one way a reader cannot detect
        nearest = min(
            outside, key=lambda p: min(abs(p["pos"] - span_lo), abs(p["pos"] - span_hi))
        )
        gap = min(abs(nearest["pos"] - span_lo), abs(nearest["pos"] - span_hi))
        ax.text(
            0.995, 0.99,
            rf"{len(outside)} r$^2\geq${_LD_NOTABLE_R2:g} partner"
            f"{'' if len(outside) == 1 else 's'} outside window; "
            f"nearest {gap / 1000:.0f} kb out, r$^2$ {nearest['r2']:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7,
            color=_WARNING_COLOUR,
        )

    _dress(ax, title if title is not None else _default_title(phenotype, region, frame, resource))
    ax.margins(x=0.02)
    if ld_joined:
        handles = [
            plt.Line2D([], [], marker="o", linestyle="", markersize=4, color=colour,
                       label=label)
            for _threshold, colour, label in _LD_BINS
        ]
        # the panel is named, not just the quantity: r² is to the lead and from one LD panel,
        # and a reader comparing two figures cannot tell either from a bare "r²". The panel
        # is whatever the call asked for, so a different one relabels itself.
        legend_title = f"LD $r^2$ to {lead_id} ({ld_panel})"
        if ld_status == "partial":
            legend_title += f"\nonly within {ld_reach / 1e6:g} Mb of the lead"
        ax.legend(handles=handles, title=legend_title,
                  fontsize=5, title_fontsize=5, loc="upper left", ncol=1)

    # pinned before the gene track can widen it through sharex
    pad = max((span_hi - span_lo) * 0.02, 1)
    ax.set_xlim(span_lo - pad, span_hi + pad)

    # headroom, so the corner notes have somewhere to sit: autoscaling leaves the strongest
    # association at the top of the panel, which is exactly where the legend and the
    # outside-window warning are drawn
    y_top = max(float(frame["_y"].max()), line_y if significance else 0.0)
    ax.set_ylim(top=y_top + max(y_top * (0.22 if outside else 0.08), 0.5))

    n_genes, n_exons = 0, 0
    if gene_ax is not None:
        n_genes, n_exons = _draw_genes(gene_ax, gene_frame, span_lo, span_hi)
        # the scale belongs to the panel it is read against: sharex hides the upper axis's
        # tick labels by default, which put the position axis under the gene track and the
        # models between the points and their own scale
        ax.tick_params(labelbottom=True)
    chrom_label = frame["chr"][0] if "chr" in frame.columns else ""
    ax.set_xlabel(
        f"position on chromosome {chrom_label}".rstrip() + " (Mb)", fontsize=_LABEL_SIZE
    )
    _mb_axis(ax)

    # last, because where a label fits depends on everything else already being in place
    _fit_title(ax, figure, own_figure)
    renderer = _renderer(figure)
    lead_gene_label = _track_symbol(lead_gene, gene_frame)
    labels = [(
        _lead_label(lead_id, lead_row, lead_y, lead_consequence, lead_gene_label),
        lead_pos, lead_y, 6,
    )]
    highlighted = []
    for vid in named:
        row = frame.row(index_of[vid], named=True)
        rsid = str(row.get("rsid") or "")
        text = rsid if rsid.startswith("rs") else vid
        if row["_r2"] is not None:
            text += rf"  r$^2$ {row['_r2']:.2f}"
        labels.append((text, row["pos"], row["_y"], 5.5))
        highlighted.append({"variant": vid, "mlog10p": float(row["_y"]), "r2": row["_r2"]})
    taken = [] if renderer is None else [
        artist.get_window_extent(renderer)
        for artist in [*ax.texts, ax.get_legend()] if artist is not None
    ]
    _place_labels(
        ax, renderer, labels,
        ax.transData.transform(np.column_stack([positions, ys])), taken,
    )

    written = None
    if own_figure:
        written = _resolve_path(path, "locuszoom.png")
        figure.savefig(written)
        plt.close(figure)

    return {
        "path": written,
        "lead": lead_id,
        "lead_mlog10p": float(lead_y),
        "region": region,
        "phenotype": phenotype,
        "n_variants": frame.height,
        "n_genes": n_genes,
        "n_exons": n_exons,
        "strongest": strongest,
        "ld_joined": ld_joined,
        "ld_status": ld_status,
        "ld_partners_outside_window": outside,
        "coding_marked": coding_marked,
        "lead_consequence": lead_consequence,
        "lead_gene": lead_gene,
        "lead_gene_label": lead_gene_label,
        "highlighted": highlighted,
    }


# One slot per association along x, and this many empty slots between one category and the
# next: the gap is what makes the groups read as groups, since nothing else separates them.
_PHEWAS_CATEGORY_GAP = 2

# Room between the strongest association and the top of the panel, in -log10 p. A fixed
# amount rather than a fraction alone, because the point labels are drawn upward and at a
# modest -log10 p a fraction leaves nowhere for the top one to go.
_PHEWAS_HEADROOM = 2.0

# how many of the strongest significant associations are named on the figure, and at what
# length: past this the labels overprint each other and none can be read
_PHEWAS_LABELS = 10
_PHEWAS_LABEL_CHARS = 30

# a FinnGen ICD chapter is `VIII Diseases of the ear and mastoid process (H8_)`: too long for
# one line under a category that may hold a single point, so a tick label is wrapped to this
# width and cut after this many lines
_PHEWAS_TICK_WIDTH = 26
_PHEWAS_TICK_LINES = 2

# where a phenotype has no `phenotypes_v` row — a QTL trait, a dataset whose codes are
# already readable, a lookup that failed — its point still needs a group
_PHEWAS_UNCATEGORISED = "Other"

# past this many categories the axis is a wall of chapter names under groups of one or two
# points — measured on the APOE missense variant, where every ICD chapter and every Open
# Targets project answers — so the plot falls back to grouping by resource
_PHEWAS_MAX_CATEGORIES = 10

_ROMAN = re.compile(r"^([IVXLC]+)\b")
_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


def _roman(numeral: str) -> int:
    total = 0
    for this, following in zip(numeral, numeral[1:] + " "):
        value = _ROMAN_VALUES[this]
        total += -value if _ROMAN_VALUES.get(following, 0) > value else value
    return total


def _category_order(category: str) -> tuple[int, int, str]:
    """`Other` last; ICD chapters in chapter order; everything else alphabetically after.

    The chapters are the one source grouping with an order of its own, and it is not the
    alphabet's: sorted as strings, `IX Diseases of the circulatory system` lands between
    `II Neoplasms` and `V Mental disorders`.
    """
    if category == _PHEWAS_UNCATEGORISED:
        return (2, 0, "")
    match = _ROMAN.match(category)
    if match:
        return (0, _roman(match.group(1)), category)
    return (1, 0, category)


def _tick_label(category: str) -> str:
    lines = textwrap.wrap(category, _PHEWAS_TICK_WIDTH)
    if len(lines) > _PHEWAS_TICK_LINES:
        lines = lines[:_PHEWAS_TICK_LINES]
        lines[-1] += "…"
    return "\n".join(lines)


def _phenotype_metadata(frame: pl.DataFrame) -> dict[tuple[str | None, str], tuple[str | None, str | None]]:
    """(dataset, code) -> (name, category) from `phenotypes_v`, for the frame's traits.

    The category is the source's own grouping — a FinnGen ICD chapter, an Open Targets
    project id — rather than one harmonised here, so a phewas across resources groups each
    resource's traits the way that resource does. Keyed on both dataset and code because a
    code recurs across datasets with different names; a frame that carries no `dataset` is
    matched on the code alone. A failed fetch is an empty mapping, tolerated the way the LD
    and consequence lookups are: every point then falls into one group, and the figure is
    still the right picture of the variant.
    """
    from genetics_mcp_server import sdk

    codes = sorted({str(code) for code in frame["_code"]})
    try:
        rows = sdk.phenotypes(codes=codes)
    except Exception:
        return {}
    if rows.is_empty() or not {"dataset", "trait_original"} <= set(rows.columns):
        return {}
    found: dict[tuple[str | None, str], tuple[str | None, str | None]] = {}
    for row in rows.iter_rows(named=True):
        key = (row["dataset"], row["trait_original"])
        found[key] = (row.get("trait_name"), row.get("category"))
        found.setdefault((None, row["trait_original"]), found[key])
    return found


def _first(frame: pl.DataFrame, column: str) -> str | None:
    """The first non-null value of a column the frame may not have."""
    if column not in frame.columns:
        return None
    values = frame[column].drop_nulls()
    return str(values[0]) if len(values) else None


def _phewas_title(variant_id: str, frame: pl.DataFrame, resource: str | None) -> str:
    """`19:44908684:T:C  APOE missense — FinnGen, UK Biobank`.

    The consequence comes off the rows themselves: a credible-set row carries the variant's
    `most_severe` and `gene_most_severe`, so unlike the locuszoom this needs no second
    lookup. The resources are the ones the data actually came from, so a caller who passed
    `data=` gets the label of what they plotted, and past three they are counted rather
    than listed.
    """
    head = _variant_head(
        variant_id, _first(frame, "most_severe"), _first(frame, "gene_most_severe")
    )
    if "resource" in frame.columns:
        resources = sorted(set(str(r) for r in frame["resource"].drop_nulls()))
    else:
        resources = [resource] if resource else []
    if len(resources) > 3:
        source = f"{len(resources)} resources"
    else:
        source = ", ".join(_resource_label(r) for r in resources)
    return f"{head} — {source}" if source else head


def phewas(
    *,
    variant: str,
    resource: str | None = None,
    min_mlog10p: float = 2.0,
    data: pl.DataFrame | None = None,
    path: str | None = None,
    title: str | None = None,
    significance: float = 5e-8,
    ax: Any = None,
) -> dict[str, Any]:
    """Phenome-wide association plot: -log10 p of every GWAS association of one variant.

    The associations are the fine-mapped credible sets the variant belongs to, across every
    resource unless `resource=` names one, kept where -log10(p) is at least `min_mlog10p`.
    Each is one point, grouped along the x axis by the category `phenotypes_v` gives its
    phenotype — the source's own grouping, so a FinnGen endpoint sits in its ICD chapter and
    an Open Targets study under its project, and a phewas across resources groups each
    resource's traits the way that resource does. ICD chapters keep chapter order; a
    phenotype with no metadata row goes to `Other`, last. A variant that answers in more
    than ten categories — a pleiotropic one, where the axis would be a wall of chapter
    names over groups of a point or two — is grouped by resource instead. The strongest
    associations above the significance line are named on the figure, each phenotype once,
    by the name `phenotypes_v` carries or else by the `trait` column read as words.

    The title names the variant with its gene and consequence, taken from the rows
    themselves, and the resources the associations came from.

    Returns a dict describing what was drawn: `path`, `variant`, `n_associations`,
    `n_significant`, `grouped_by` ("category" or "resource") with the `groups` in plotting
    order, `strongest` and `strongest_name` (the
    phenotype code and name of the top association) with `strongest_mlog10p`, and
    `variant_consequence` and `variant_gene` as the title shows them.

    `path` may be relative, in which case it is written inside the execution's artifacts
    directory and returned to the user automatically; that is also where the default goes.
    Pass `ax` to draw into an existing axis instead, in which case nothing is saved. Pass
    `data=` to plot a frame already in hand — credible-set rows, or any frame with a `trait`
    column and `mlog10p` or `pval`; `trait_original` and `dataset` are what the names and
    categories are looked up by, and without them every point is `Other`.
    """
    import matplotlib.pyplot as plt

    from genetics_mcp_server import sdk

    variant_id = _norm_variant_id(variant)
    frame = data if data is not None else sdk.credible_sets(
        variant=variant_id, resource=resource, data_types="GWAS"
    )
    if "trait" not in frame.columns:
        raise GeneticsUsageError(
            f"the associations have no 'trait' column, so there is nothing to place on the "
            f"x axis; columns are {frame.columns}"
        )
    if "data_type" in frame.columns:
        # a caller's `data=` may carry the QTL rows too; a phewas is the GWAS ones
        frame = frame.filter(pl.col("data_type").cast(pl.Utf8).str.to_uppercase() == "GWAS")
    if not frame.is_empty():
        frame = frame.with_columns(_mlog10p(frame).alias("_y")).filter(
            pl.col("_y") >= min_mlog10p
        )
    if frame.is_empty():
        raise GeneticsUsageError(
            f"no GWAS associations for {variant_id} at -log10(p) >= {min_mlog10p:g} "
            f"(resource={resource!r}) — nothing to plot"
        )

    # `trait_original` is the code the metadata is keyed on; `trait` is a display form of it
    # (`Height,_inverse-rank_normalized` for `HEIGHT_IRN`) that resolves nothing, and is the
    # label of last resort, read as words
    code_column = "trait_original" if "trait_original" in frame.columns else "trait"
    frame = frame.with_columns(
        pl.col(code_column).cast(pl.Utf8).alias("_code"),
        pl.col("trait").cast(pl.Utf8).str.replace_all("_", " ").alias("_display"),
    )
    metadata = _phenotype_metadata(frame)
    datasets = frame["dataset"] if "dataset" in frame.columns else [None] * frame.height
    names, categories_of = [], []
    for dataset, code, display in zip(datasets, frame["_code"], frame["_display"]):
        name, category = metadata.get(
            (dataset, code), metadata.get((None, code), (None, None))
        )
        names.append(name or display)
        categories_of.append(category or _PHEWAS_UNCATEGORISED)
    grouped_by = "category"
    if len(set(categories_of)) > _PHEWAS_MAX_CATEGORIES:
        grouped_by = "resource"
        resources = frame["resource"] if "resource" in frame.columns else [None] * frame.height
        categories_of = [
            _resource_label(str(r)) if r else _PHEWAS_UNCATEGORISED for r in resources
        ]
    frame = frame.with_columns(
        pl.Series("_name", names, dtype=pl.Utf8),
        pl.Series("_category", categories_of, dtype=pl.Utf8),
    )
    # within a group the strongest association comes first
    order = {c: i for i, c in enumerate(sorted(set(categories_of), key=_category_order))}
    frame = frame.with_columns(
        pl.col("_category").replace_strict(order, return_dtype=pl.Int64).alias("_order")
    ).sort(["_order", "_y"], descending=[False, True])
    codes = list(frame["_code"])

    categories: list[str] = []
    spans: dict[str, list[int]] = {}
    xs: list[int] = []
    x = 0
    for category in frame["_category"]:
        if categories and category != categories[-1]:
            x += _PHEWAS_CATEGORY_GAP
        if category not in spans:
            categories.append(category)
            spans[category] = [x, x]
        spans[category][1] = x
        xs.append(x)
        x += 1
    ys = [float(y) for y in frame["_y"]]
    labels = list(frame["_name"])

    own_figure = ax is None
    if own_figure:
        figure, ax = plt.subplots(figsize=(6.5, 3.4), constrained_layout=True)
    else:
        figure = ax.get_figure()

    # colour separates neighbouring groups and encodes nothing a reader decodes, so unlike
    # the LD ramp it follows whatever prop_cycle the caller's style set
    palette = dict(zip(categories, itertools.cycle(
        plt.rcParams["axes.prop_cycle"].by_key().get("color") or ["#333333"]
    )))
    ax.scatter(xs, ys, c=[palette[c] for c in frame["_category"]],
               marker=_MARKER_OTHER, s=9, linewidths=0.2, edgecolors="#33333355", zorder=2)
    line_y = _significance_line(ax, significance)

    # each phenotype is named once, at its strongest point: the same trait from two resources
    # is two points a slot apart, and two copies of one label overprint into neither
    named: list[int] = []
    seen: set[str] = set()
    for i in sorted(range(len(ys)), key=lambda i: ys[i], reverse=True):
        if significance and ys[i] < line_y:
            break
        if codes[i] in seen:
            continue
        seen.add(codes[i])
        named.append(i)
        if len(named) == _PHEWAS_LABELS:
            break
    for i in named:
        text = labels[i]
        if len(text) > _PHEWAS_LABEL_CHARS:
            text = text[:_PHEWAS_LABEL_CHARS] + "…"
        ax.annotate(text, (xs[i], ys[i]), textcoords="offset points", xytext=(2, 2),
                    ha="left", va="bottom", fontsize=5)

    _dress(ax, title if title is not None else _phewas_title(variant_id, frame, resource))
    # the categories ARE the x scale: one label under the middle of each group, and no tick
    # marks, since a mark would point at one association among several
    ax.set_xticks([(lo + hi) / 2 for lo, hi in (spans[c] for c in categories)])
    ax.set_xticklabels([_tick_label(c) for c in categories], rotation=45, ha="right",
                       rotation_mode="anchor")
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-1, x)
    y_top = max(max(ys), line_y)
    ax.set_ylim(0, y_top + max(_PHEWAS_HEADROOM, y_top * 0.08))

    written = None
    if own_figure:
        written = _resolve_path(path, "phewas.png")
        figure.savefig(written)
        plt.close(figure)

    strongest = max(range(len(ys)), key=lambda i: ys[i])
    row = frame.row(strongest, named=True)
    return {
        "path": written,
        "variant": variant_id,
        "n_associations": frame.height,
        "n_significant": sum(y >= line_y for y in ys) if significance else 0,
        "grouped_by": grouped_by,
        "groups": categories,
        "strongest": row["_code"],
        "strongest_name": row["_name"],
        "strongest_mlog10p": ys[strongest],
        "variant_consequence": _first(frame, "most_severe"),
        "variant_gene": _first(frame, "gene_most_severe"),
    }


# ----------------------------------------------------------------------------------- upset

# Everything on an upset is one grey or another. The bars encode a count and the dots a
# membership; colour would encode nothing a reader decodes, and in the scripts this replaced
# it was spent on one hue per bar, which reads as a categorical the figure never explains.
_UPSET_INK = "#333333"
_UPSET_SET_BAR = "#8A8A8A"
_UPSET_OFF = "#DDDDDD"
_UPSET_BAND = "#F2F2F2"

# past this many intersections the columns are thinner than their count labels; the largest
# are kept and the rest reported, not drawn
_UPSET_MAX_INTERSECTIONS = 30

# inches: the intersection panel widens per column and the matrix deepens per set, so a plot
# of 3 sets and 7 columns and one of 8 sets and 30 columns are both readable at 6 pt
_UPSET_COLUMN_IN = 0.22
_UPSET_ROW_IN = 0.24
_UPSET_BARS_IN = 2.2
_UPSET_SET_PANEL_IN = 1.4
_UPSET_MAX_WIDTH_IN = 12.0

# the count labels sit outside the bar they count — above an intersection bar, beyond the
# tip of a set bar — so each scale is extended by this much past its longest bar to hold them
_UPSET_BAR_HEADROOM = 0.14
_UPSET_SET_HEADROOM = 0.4

_SORT_BY = ("size", "degree")


def _upset_key(key: Any) -> frozenset[str]:
    if isinstance(key, str):
        return frozenset([key])
    names = frozenset(str(k) for k in key)
    if not names:
        raise GeneticsUsageError("an intersection key names no set")
    return names


def _upset_from_sets(sets: Any) -> tuple[list[str], dict[frozenset[str], int]]:
    """Exclusive intersection sizes from the sets' members: each element counted once, in
    the intersection of exactly the sets that hold it."""
    members = {str(name): set(values) for name, values in dict(sets).items()}
    where: dict[Any, set[str]] = {}
    for name, values in members.items():
        for element in values:
            where.setdefault(element, set()).add(name)
    counts: dict[frozenset[str], int] = {}
    for names in where.values():
        counts[frozenset(names)] = counts.get(frozenset(names), 0) + 1
    return list(members), counts


def _upset_from_data(
    data: pl.DataFrame, columns: Any, count: str | None
) -> tuple[list[str], dict[frozenset[str], int]]:
    """Exclusive intersection sizes from a frame with one membership column per set, one row
    per element — or, with `count=`, one row per combination already tallied."""
    names = [str(c) for c in columns] if columns else [
        c for c in data.columns if data.schema[c] == pl.Boolean
    ]
    if not names:
        raise GeneticsUsageError(
            f"no membership columns: pass columns=[...] naming them, or give the frame "
            f"boolean columns; columns are {data.columns}"
        )
    missing = [c for c in names if c not in data.columns]
    if missing or (count and count not in data.columns):
        raise GeneticsUsageError(
            f"columns {missing + ([count] if count and count not in data.columns else [])} "
            f"are not in the frame; columns are {data.columns}"
        )
    flags = data.select(
        [pl.col(c).cast(pl.Boolean, strict=False).fill_null(False) for c in names]
        + ([pl.col(count).cast(pl.Int64).fill_null(0).alias("_n")] if count else [pl.lit(1).alias("_n")])
    )
    counts: dict[frozenset[str], int] = {}
    for row in flags.iter_rows():
        names_in = frozenset(n for n, on in zip(names, row) if on)
        if names_in:
            counts[names_in] = counts.get(names_in, 0) + int(row[-1])
    return names, counts


def _upset_from_counts(counts: Any) -> tuple[list[str], dict[frozenset[str], int]]:
    """Exclusive intersection sizes as given; the sets are the names the keys mention, in
    order of first mention."""
    tallies: dict[frozenset[str], int] = {}
    names: list[str] = []
    for key, value in dict(counts).items():
        names_in = _upset_key(key)
        for name in (key,) if isinstance(key, str) else key:
            if str(name) not in names:
                names.append(str(name))
        tallies[names_in] = tallies.get(names_in, 0) + int(value)
    return names, tallies


def upset(
    *,
    sets: Any = None,
    data: pl.DataFrame | None = None,
    columns: Any = None,
    count: str | None = None,
    counts: Any = None,
    sort_by: str = "size",
    min_count: int = 1,
    max_intersections: int = _UPSET_MAX_INTERSECTIONS,
    ylabel: str = "Intersection size",
    path: str | None = None,
    title: str | None = None,
    ax: Any = None,
) -> dict[str, Any]:
    """UpSet plot: how many elements fall in each exclusive intersection of some sets.

    Three ways to say what the sets are, for three shapes the data comes in:

    - `sets={"Crohn": ids, "UC": ids}` — each set's members, any hashable values; every
      element is counted once, in the intersection of exactly the sets that hold it.
    - `data=frame` with one boolean (or 0/1) column per set and one row per element;
      `columns=[...]` names the membership columns, else every boolean column is one. With
      `count="n"` each row is a combination already tallied — the shape a `GROUP BY` over
      indicator columns returns — and the column is summed rather than the rows counted.
    - `counts={("CD",): 9, ("UC",): 48, ("CD", "UC"): 4}` — exclusive intersection sizes
      already known, keyed by the names of the sets in each; a plain string keys one set.

    Sets are rows, largest at the top, with their total size barred to the left; the
    intersections are columns, largest first (`sort_by="degree"` orders them by how many sets
    they span instead), each with its count above the bar and the sets it spans marked in the
    matrix below. Only intersections of at least `min_count` are drawn, and no more than
    `max_intersections` of them. Nothing is coloured: the bars carry a count and the dots a
    membership, and a colour would encode nothing a reader decodes.

    Returns a dict describing what was drawn: `path`, `sets` in row order top to bottom with
    `set_sizes`, `intersections` in column order as `{"sets": [...], "count": n}`,
    `n_intersections` (non-empty, before the cut) and `n_elements` (their sum).

    `path` may be relative, in which case it is written inside the execution's artifacts
    directory and returned to the user automatically; that is also where the default goes.
    Pass `ax` to draw in its place in an existing figure — the axis is replaced by the three
    panels — in which case nothing is saved.
    """
    import matplotlib.pyplot as plt

    given = [name for name, value in (("sets", sets), ("data", data), ("counts", counts))
             if value is not None]
    if len(given) != 1:
        raise GeneticsUsageError(
            f"pass exactly one of sets=, data= or counts=; got {given or 'none'}"
        )
    if sort_by not in _SORT_BY:
        raise GeneticsUsageError(f"sort_by must be one of {_SORT_BY}, not {sort_by!r}")
    if sets is not None:
        names, tallies = _upset_from_sets(sets)
    elif data is not None:
        names, tallies = _upset_from_data(data, columns, count)
    else:
        names, tallies = _upset_from_counts(counts)
    if len(names) < 2:
        raise GeneticsUsageError(f"an upset needs at least two sets; got {names}")
    unknown = sorted(set().union(*tallies) - set(names)) if tallies else []
    if unknown:
        raise GeneticsUsageError(f"intersections name sets that were not given: {unknown}")

    set_sizes = {n: sum(v for k, v in tallies.items() if n in k) for n in names}
    rows = sorted(names, key=lambda n: (-set_sizes[n], names.index(n)))
    nonempty = {k: v for k, v in tallies.items() if v >= max(min_count, 1)}
    if not nonempty:
        raise GeneticsUsageError(
            f"no intersection holds {max(min_count, 1)} or more elements — nothing to plot"
        )
    if sort_by == "degree":
        order = sorted(nonempty, key=lambda k: (len(k), -nonempty[k], sorted(k)))
    else:
        order = sorted(nonempty, key=lambda k: (-nonempty[k], len(k), sorted(k)))
    shown = order[:max_intersections]
    n_columns = len(shown)

    matrix_in = max(3.2, _UPSET_COLUMN_IN * n_columns)
    rows_in = max(0.6, _UPSET_ROW_IN * len(rows))
    own_figure = ax is None
    if own_figure:
        figure = plt.figure(
            figsize=(min(_UPSET_MAX_WIDTH_IN, _UPSET_SET_PANEL_IN + matrix_in + 1.9),
                     _UPSET_BARS_IN + rows_in + 0.8),
            constrained_layout=True,
        )
        grid = figure.add_gridspec
    else:
        figure = ax.get_figure()
        grid = ax.get_subplotspec().subgridspec
        ax.remove()
    spec = grid(2, 2, width_ratios=[_UPSET_SET_PANEL_IN, matrix_in],
                height_ratios=[_UPSET_BARS_IN, rows_in], wspace=0.04, hspace=0.04)
    ax_bars = figure.add_subplot(spec[0, 1])
    ax_matrix = figure.add_subplot(spec[1, 1])
    ax_sets = figure.add_subplot(spec[1, 0])
    figure.add_subplot(spec[0, 0]).set_axis_off()

    xs = list(range(n_columns))
    heights = [nonempty[k] for k in shown]
    ax_bars.bar(xs, heights, width=0.6, color=_UPSET_INK)
    for x, h in zip(xs, heights):
        ax_bars.annotate(f"{h:,}", (x, h), textcoords="offset points", xytext=(0, 1.5),
                         ha="center", va="bottom", fontsize=5)
    ax_bars.set_ylim(0, max(heights) * (1 + _UPSET_BAR_HEADROOM))
    ax_bars.set_xlim(-0.6, n_columns - 0.4)
    ax_bars.set_xticks([])
    ax_bars.set_ylabel(ylabel, fontsize=_LABEL_SIZE)
    ax_bars.set_title(title or "", fontsize=_TITLE_SIZE)
    for side in ("top", "right"):
        ax_bars.spines[side].set_visible(False)

    # y of a set is its row, top row first; the matrix and the set bars share the scale
    y_of = {n: len(rows) - 1 - i for i, n in enumerate(rows)}
    for i, n in enumerate(rows):
        if i % 2:
            for panel in (ax_matrix, ax_sets):
                panel.axhspan(y_of[n] - 0.5, y_of[n] + 0.5, color=_UPSET_BAND, zorder=0)
    for x, key in zip(xs, shown):
        ys_on = [y_of[n] for n in rows if n in key]
        ax_matrix.scatter([x] * len(rows), [y_of[n] for n in rows], s=36,
                          color=[_UPSET_INK if n in key else _UPSET_OFF for n in rows], zorder=2)
        if len(ys_on) > 1:
            ax_matrix.plot([x, x], [min(ys_on), max(ys_on)], color=_UPSET_INK,
                           linewidth=1.2, zorder=1)
    ax_matrix.set_xlim(-0.6, n_columns - 0.4)
    ax_matrix.set_ylim(-0.5, len(rows) - 0.5)
    ax_matrix.set_xticks([])
    ax_matrix.set_yticks([])
    ax_matrix.set_axis_off()

    sizes = [set_sizes[n] for n in rows]
    ax_sets.barh([y_of[n] for n in rows], sizes, height=0.5, color=_UPSET_SET_BAR, zorder=2)
    for n in rows:
        ax_sets.annotate(f"{set_sizes[n]:,}", (set_sizes[n], y_of[n]), textcoords="offset points",
                         xytext=(-2, 0), ha="right", va="center", fontsize=5)
    ax_sets.set_xlim(max(sizes) * (1 + _UPSET_SET_HEADROOM), 0)
    ax_sets.set_ylim(-0.5, len(rows) - 0.5)
    ax_sets.set_yticks([y_of[n] for n in rows])
    ax_sets.set_yticklabels([_tick_label(n) for n in rows])
    ax_sets.tick_params(axis="y", length=0)
    ax_sets.set_xlabel("Set size", fontsize=_LABEL_SIZE)
    for side in ("top", "right", "left"):
        ax_sets.spines[side].set_visible(False)
    for panel in (ax_bars, ax_sets):
        _size(panel)

    written = None
    if own_figure:
        written = _resolve_path(path, "upset.png")
        figure.savefig(written)
        plt.close(figure)

    return {
        "path": written,
        "sets": rows,
        "set_sizes": set_sizes,
        "intersections": [{"sets": [n for n in rows if n in k], "count": nonempty[k]} for k in shown],
        "n_intersections": len(nonempty),
        "n_elements": sum(nonempty.values()),
    }


# --------------------------------------------------------------------------- line models

# Okabe–Ito, in an order that keeps the first three models far apart; the palette is data
# encoding (a reader decodes the model from the hue), so it does not follow the prop_cycle
_LINEMODEL_COLOURS = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000",
)
_LINEMODEL_UNDETERMINED = "#9A9A9A"
_LINEMODEL_ERRORBAR = "#BBBBBB"
_LINEMODEL_REGION_PROB = 0.95
_LINEMODEL_FIGURE_IN = 3.6
# how many multiples of the largest scale the axes span when no data set the extent; the
# R package's convention
_LINEMODEL_AXIS_SCALES = 3.0


def _region_boundary(covariance, prob: float):
    """The x, y of the highest-density region's boundary under N(0, covariance)."""
    from scipy.stats import chi2

    radius2 = chi2.ppf(prob, 2)
    # eigen rather than Cholesky: cor = 1 makes the covariance singular, and the region is
    # then a segment along the line rather than an error
    values, vectors = np.linalg.eigh(covariance)
    values = np.clip(values, 0.0, None)
    t = np.linspace(0.0, 2.0 * math.pi, 361)
    circle = np.vstack([np.cos(t), np.sin(t)])
    pts = vectors @ (np.sqrt(values * radius2)[:, None] * circle)
    return pts[0], pts[1]


def linemodels(
    result: Any,
    *,
    X: Any = None,
    SE: Any = None,
    groups: pl.DataFrame | None = None,
    threshold: float = 0.95,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    xlim: Any = None,
    ylim: Any = None,
    path: str | None = None,
    ax: Any = None,
) -> dict[str, Any]:
    """Line models over two effect variables, with the variants coloured by assignment.

    `result` is what `genetics.linemodels.classify`, `proportions` or `optimize` returned
    (its `models` and `groups` are used), or just a `models` frame with `model`, `scale`,
    `slope` and `cor` columns. Each model is drawn as its line through the origin and the
    dashed boundary of the region holding 95% of its effects — a slope of infinity is the
    vertical axis and a scale of 0 a point at the origin — so the figure shows what each
    hypothesis claims before any point is read against it.

    Pass the same `X` (and `SE`, for 95% error bars) the fit was given to draw the variants:
    a point is filled in its model's colour when that model's probability reaches
    `threshold`, and left grey when no model does — those sit near the origin, where the
    models are not separable, by design. `groups` overrides the result's own. Axis labels
    default to the column names of `X`.

    Returns `path`, `models` in drawing order, `n_points`, `n_assigned` per model,
    `n_undetermined` and the `threshold` used. `path` may be relative, in which case it is
    written inside the execution's artifacts directory; pass `ax` to draw into an existing
    figure instead, in which case nothing is saved.
    """
    import matplotlib.pyplot as plt

    from genetics_mcp_server.sdk import linemodels as lm

    if isinstance(result, dict):
        models = result.get("models")
        if groups is None:
            groups = result.get("groups")
    else:
        models = result
    if not isinstance(models, pl.DataFrame) or not {"model", "scale", "slope", "cor"} <= set(models.columns):
        raise GeneticsUsageError(
            "result must be a genetics.linemodels result or a models frame with model, "
            "scale, slope and cor columns; only two effect variables can be drawn"
        )
    names = models["model"].to_list()
    scales = [float(v) for v in models["scale"].to_list()]
    slopes = [float(v) for v in models["slope"].to_list()]
    cors = [float(v) for v in models["cor"].to_list()]
    if len(names) > len(_LINEMODEL_COLOURS):
        raise GeneticsUsageError(f"at most {len(_LINEMODEL_COLOURS)} models can be drawn")
    if not 0 < threshold <= 1:
        raise GeneticsUsageError("threshold must lie in (0, 1]")

    points = None
    errors = None
    dims = None
    if X is not None:
        points, dims = lm._matrix(X, "X")
        if points.shape[1] != 2:
            raise GeneticsUsageError("only two effect variables can be drawn; select two columns")
        if SE is not None:
            errors, _ = lm._matrix(SE, "SE")
            if errors.shape != points.shape:
                raise GeneticsUsageError(f"X has shape {points.shape} but SE has shape {errors.shape}")
    colour_of = dict(zip(names, _LINEMODEL_COLOURS))

    assigned: dict[str, int] = {n: 0 for n in names}
    point_colours: list[str] = []
    if points is not None and groups is not None:
        if groups.height != points.shape[0]:
            raise GeneticsUsageError(
                f"groups has {groups.height} rows but X has {points.shape[0]}"
            )
        missing = [n for n in names if n not in groups.columns]
        if missing:
            raise GeneticsUsageError(f"groups lacks a probability column for {missing}")
        probs = groups.select(names).to_numpy()
        best = probs.argmax(axis=1)
        for i in range(points.shape[0]):
            if probs[i, best[i]] >= threshold:
                name = names[best[i]]
                assigned[name] += 1
                point_colours.append(colour_of[name])
            else:
                point_colours.append(_LINEMODEL_UNDETERMINED)
    elif points is not None:
        point_colours = [_LINEMODEL_UNDETERMINED] * points.shape[0]

    own_figure = ax is None
    if own_figure:
        figure, ax = plt.subplots(figsize=(_LINEMODEL_FIGURE_IN, _LINEMODEL_FIGURE_IN),
                                  constrained_layout=True)
    else:
        figure = ax.get_figure()

    extent = _LINEMODEL_AXIS_SCALES * max(scales) if scales else 1.0
    if points is not None:
        reach = np.abs(points) + (1.96 * errors if errors is not None else 0.0)
        extent = max(extent, 1.08 * float(reach.max()))
    lo_x, hi_x = (-extent, extent) if xlim is None else (float(xlim[0]), float(xlim[1]))
    lo_y, hi_y = (-extent, extent) if ylim is None else (float(ylim[0]), float(ylim[1]))

    ax.axhline(0, color="#000000", linewidth=0.5, zorder=1)
    ax.axvline(0, color="#000000", linewidth=0.5, zorder=1)
    for name, scale, slope, cor in zip(names, scales, slopes, cors):
        colour = colour_of[name]
        if scale < lm._NULL_SCALE:
            ax.plot([0], [0], marker="o", color=colour, markersize=4, zorder=3)
            continue
        if math.isinf(slope):
            ax.plot([0, 0], [lo_y, hi_y], color=colour, linewidth=1.0, zorder=2)
        else:
            xs = np.array([lo_x, hi_x])
            ax.plot(xs, slope * xs, color=colour, linewidth=1.0, zorder=2)
        rx, ry = _region_boundary(
            lm._prior_covariance(scale, np.array([slope]), cor), _LINEMODEL_REGION_PROB
        )
        ax.plot(rx, ry, color=colour, linewidth=0.7, linestyle="--", zorder=2)

    if points is not None:
        if errors is not None:
            ax.errorbar(points[:, 0], points[:, 1], xerr=1.96 * errors[:, 0],
                        yerr=1.96 * errors[:, 1], fmt="none", ecolor=_LINEMODEL_ERRORBAR,
                        elinewidth=0.5, zorder=3)
        filled = [c != _LINEMODEL_UNDETERMINED for c in point_colours]
        colours = np.array(point_colours)
        mask = np.array(filled)
        if (~mask).any():
            ax.scatter(points[~mask, 0], points[~mask, 1], s=14, facecolors="none",
                       edgecolors=_LINEMODEL_UNDETERMINED, linewidths=0.6, zorder=4)
        if mask.any():
            ax.scatter(points[mask, 0], points[mask, 1], s=14, c=colours[mask],
                       edgecolors="#FFFFFF", linewidths=0.3, zorder=5)

    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color=colour_of[n], linewidth=1.5,
               label=f"{n} ({assigned[n]})" if points is not None and groups is not None else n)
        for n in names
    ]
    if points is not None and groups is not None:
        undetermined = sum(1 for c in point_colours if c == _LINEMODEL_UNDETERMINED)
        handles.append(Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none",
                              markeredgecolor=_LINEMODEL_UNDETERMINED,
                              label=f"< {threshold:g} ({undetermined})"))
    else:
        undetermined = 0
    ax.legend(handles=handles, fontsize=5, frameon=False, loc="best")
    ax.set_xlim(lo_x, hi_x)
    ax.set_ylim(lo_y, hi_y)
    ax.set_xlabel(xlabel or (dims[0] if dims else "effect 1"), fontsize=_LABEL_SIZE)
    ax.set_ylabel(ylabel or (dims[1] if dims else "effect 2"), fontsize=_LABEL_SIZE)
    ax.set_title(title or "", fontsize=_TITLE_SIZE)
    ax.grid(True, linewidth=0.3, color="#DDDDDD", zorder=0)
    _size(ax)

    written = None
    if own_figure:
        written = _resolve_path(path, "linemodels.png")
        figure.savefig(written)
        plt.close(figure)
    return {
        "path": written,
        "models": names,
        "n_points": 0 if points is None else int(points.shape[0]),
        "n_assigned": assigned,
        "n_undetermined": undetermined,
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------------- forest

# One ink when there is a single series: an estimate, its interval and the numbers beside it
# are one statement, and a hue there would encode nothing a reader decodes.
_FOREST_INK = "#222222"
_FOREST_MUTED = "#8A8A8A"

# Okabe–Ito without its yellow, which does not hold a hairline on white. Data encoding, as on
# the line-models figure: a reader decodes the series from the hue and, in greyscale, from
# the marker, so neither follows the prop_cycle. No diamond among the markers — that shape
# is a pooled estimate's.
_FOREST_SERIES = (
    ("#0072B2", "o"), ("#D55E00", "s"), ("#009E73", "^"), ("#CC79A7", "v"),
    ("#E69F00", "P"), ("#56B4E9", "X"), ("#000000", "h"),
)

# inches. A row holds one estimate, or one per series stacked `_FOREST_MARK_IN` apart. The
# interval panel is a fixed width and the columns either side are measured, so the figure
# is as wide as its longest label needs and as tall as its rows.
_FOREST_ROW_IN = 0.155
_FOREST_MARK_IN = 0.115
_FOREST_HEADER_IN = 0.2
_FOREST_PLOT_IN = 2.3
_FOREST_GUTTER_IN = 0.14
_FOREST_INDENT_IN = 0.09
_FOREST_SIDE_IN = 0.08
_FOREST_DIAMOND_IN = 0.045
_FOREST_LABEL_CHARS = 48
_FOREST_CELL_CHARS = 24

# A small filled circle of one size, which is what the genetics literature draws; the
# weight-scaled square is the meta-analysis convention and appears only when a weight is
# asked for. Marker AREA is what a weight scales, between these two sides in points.
_FOREST_MARKER_PT = 3.2
_FOREST_WEIGHT_PT = (1.8, 5.6)
_FOREST_INTERVAL_WIDTH = 0.75

# An interval more than this many times wider than the median one does not set the axis: it
# is drawn to the edge and ends in an arrowhead. Without it one estimate from a model that
# did not converge — a log odds ratio of -14 with a standard error of 300 is what a variant
# with no carriers among the cases returns — takes the whole axis and flattens every other
# row into a column of dots at the null.
_FOREST_CLIP = 4.0

_FOREST_SCALES = ("linear", "log_ratio", "ratio")
_FOREST_POOLS = {"fixed": "Fixed effect meta-analysis", "random": "Random effects meta-analysis"}
_FOREST_SORTS = ("estimate",)
_FOREST_EFFECT_NAMES = {"OR": "Odds ratio", "HR": "Hazard ratio", "RR": "Risk ratio"}
_FOREST_UNGROUPED = "Other"

# exp() of anything past this overflows a float, and a log ratio that large is not an
# estimate of anything
_FOREST_MAX_LOG = 700.0


def _plain(value: Any, limit: int) -> str:
    """A caller's string as text matplotlib draws literally, cut to `limit` characters.

    Two dollar signs in one string are a mathtext expression to matplotlib, so a label
    holding a pair of them fails at draw time and loses the figure.
    """
    text = " ".join(str(value).split())
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text.replace("$", r"\$")


def _text_width_in(text: str, size: float, bold: bool = False) -> float:
    """How wide a string draws, in inches, without needing a figure to draw it on."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath

    if not text:
        return 0.0
    try:
        prop = FontProperties(weight="bold" if bold else "normal")
        return float(TextPath((0, 0), text, size=size, prop=prop).get_extents().width) / 72
    except Exception:
        # a string the path renderer cannot lay out still needs a column to sit in
        return 0.6 * size * len(text) / 72


def _floats(frame: pl.DataFrame, column: str, keep_inf: bool = False) -> list[float | None]:
    """A column as floats, with what cannot be one — and NaN — as None.

    `keep_inf` is for interval bounds, where an infinite bound is a one-sided interval
    rather than a missing value.
    """
    out: list[float | None] = []
    for value in frame[column].cast(pl.Float64, strict=False).to_list():
        if value is None or math.isnan(value) or (math.isinf(value) and not keep_inf):
            out.append(None)
        else:
            out.append(float(value))
    return out


def _forest_p(mlog10p: float | None) -> str:
    """A p-value as a journal sets it: `0.034`, or `3.2×10⁻⁸` with a real superscript.

    Taken apart from -log10(p) for the reason `_format_p` gives.
    """
    if mlog10p is None or not math.isfinite(mlog10p) or mlog10p < 0:
        return ""
    if mlog10p < 3:
        return f"{10 ** -mlog10p:.2g}"
    exponent = math.floor(mlog10p)
    mantissa = 10 ** (1 - (mlog10p - exponent))
    exponent += 1
    if round(mantissa, 1) >= 10:
        mantissa /= 10
        exponent -= 1
    return rf"${mantissa:.1f}\times10^{{-{exponent}}}$"


def _forest_decimals(values: list[float]) -> int:
    """Two decimals, or as many as it takes for the typical estimate to show two digits."""
    sizes = sorted(abs(v) for v in values if v)
    if not sizes or sizes[len(sizes) // 2] >= 0.1:
        return 2
    return min(6, math.ceil(-math.log10(sizes[len(sizes) // 2])) + 1)


def _forest_number(value: float, decimals: int, ratio: bool = False) -> str:
    """One number of the estimate column, with a typographic minus.

    A ratio past three orders of magnitude either way is written as a bound: an odds ratio
    of 5×10²⁵⁷ is a model that did not converge, and its digits are not information.
    """
    size = abs(value)
    if ratio and size < 0.001:
        return "<0.001"
    if ratio and size > 1000:
        return ">1,000"
    if size >= 1e5:
        exponent = math.floor(math.log10(size))
        return rf"${value / 10 ** exponent:.1f}\times10^{{{exponent}}}$"
    if size >= 100:
        text = f"{value:.0f}"
    elif size >= 10:
        text = f"{value:.1f}"
    else:
        text = f"{value:.{3 if ratio and size < 0.01 else decimals}f}"
    if float(text) == 0:
        text = text.lstrip("-")
    return text.replace("-", "−")


def _forest_cell(value: Any) -> str:
    """One cell of a caller's extra column: counts with separators, floats to three digits."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        if value.is_integer() and abs(value) < 1e15:
            return f"{int(value):,}"
        return f"{value:.3g}".replace("-", "−")
    return _plain(value, _FOREST_CELL_CHARS)


def _forest_ratio_ticks(lo: float, hi: float) -> list[float]:
    """Tick values for a ratio axis: the numbers a reader expects, not the decades alone.

    A log axis left to itself ticks at powers of ten, which puts one tick — or none — on an
    axis running from 0.7 to 1.6, the range most odds ratios live in. A narrow axis is
    ticked in even steps and a wide one at 1-2-5, thinned to the decades when those crowd.
    """
    from matplotlib.ticker import MaxNLocator

    if hi / lo < 3:
        ticks = MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi)
        return [float(t) for t in ticks if lo <= t <= hi]
    decades = range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1)
    ticks: list[float] = []
    for mantissas in ((1, 2, 5), (1,)):
        ticks = [m * 10.0 ** e for e in decades for m in mantissas if lo <= m * 10.0 ** e <= hi]
        if len(ticks) <= 7:
            return ticks
    # every nth decade, counted from 1 so that the null keeps its tick
    step = math.ceil(len(ticks) / 7)
    return [10.0 ** e for e in decades if e % step == 0 and lo <= 10.0 ** e <= hi]


def _forest_tick_label(value: float) -> str:
    if value >= 1e4 or value < 1e-3:
        return rf"$10^{{{round(math.log10(value))}}}$"
    return f"{value:g}"


def _forest_pool(estimates: list[float], ses: list[float], method: str) -> dict[str, float]:
    """The inverse-variance pooled estimate and its heterogeneity, on the scale given.

    `random` is DerSimonian–Laird: the between-row variance is the moment estimate from
    Cochran's Q, floored at zero, where it gives the fixed-effect answer back.
    """
    from scipy.stats import chi2, norm

    b = np.asarray(estimates, dtype=float)
    variances = np.square(np.asarray(ses, dtype=float))
    w = 1.0 / variances
    fixed = float((w * b).sum() / w.sum())
    q = float((w * np.square(b - fixed)).sum())
    df = len(b) - 1
    spread = float(w.sum() - np.square(w).sum() / w.sum())
    tau2 = max(0.0, (q - df) / spread) if spread > 0 else 0.0
    if method == "random":
        w = 1.0 / (variances + tau2)
    pooled = float((w * b).sum() / w.sum())
    se = math.sqrt(1.0 / float(w.sum()))
    return {
        "a": pooled,
        "se": se,
        "mlog10p": float(-(math.log(2) + norm.logsf(abs(pooled / se))) / math.log(10)),
        "k": len(b),
        "q": q,
        "i2": max(0.0, (q - df) / q) if q > 0 else 0.0,
        "tau2": tau2,
        "p_het": float(chi2.sf(q, df)),
    }


def _forest_limits(marks: list[dict[str, Any]]) -> tuple[float, float]:
    """The axis limits on the analysis scale, where the null is 0.

    The axis covers the null, every estimate that is informative, and every interval up to
    `_FOREST_CLIP` median half-widths past those. An estimate whose own interval is wider
    than that is not informative and does not set the axis either.
    """
    halves = [
        (m["hi"] - m["lo"]) / 2 for m in marks
        if m["lo"] is not None and m["hi"] is not None
        and math.isfinite(m["lo"]) and math.isfinite(m["hi"]) and m["hi"] > m["lo"]
    ]
    reach = _FOREST_CLIP * statistics.median(halves) if halves else 0.0

    def informative(m: dict[str, Any]) -> bool:
        if m["a"] is None:
            return False
        if m["lo"] is None or m["hi"] is None or not halves:
            return True
        return (m["hi"] - m["lo"]) / 2 <= reach

    centres = [m["a"] for m in marks if informative(m)] + [0.0]
    lo, hi = min(centres), max(centres)
    lows = [max(m["lo"], lo - reach) for m in marks if m["lo"] is not None]
    highs = [min(m["hi"], hi + reach) for m in marks if m["hi"] is not None]
    lo, hi = min(lows + [lo]), max(highs + [hi])
    if hi <= lo:
        return lo - 1.0, hi + 1.0
    pad = 0.04 * (hi - lo)
    return lo - pad, hi + pad


def forest(
    data: pl.DataFrame,
    *,
    label: str,
    estimate: str = "beta",
    se: str | None = "se",
    lower: str | None = None,
    upper: str | None = None,
    scale: str = "linear",
    ci: float = 0.95,
    group: str | None = None,
    series: str | None = None,
    summary: str | None = None,
    pool: str | None = None,
    weight: str | None = None,
    pvalue: str | None = "auto",
    significance: float | None = None,
    columns: Any = None,
    table: bool = True,
    sort_by: str | None = None,
    effect: str | None = None,
    xlabel: str | None = None,
    xlim: Any = None,
    direction: Any = None,
    italic: bool = False,
    max_rows: int = 45,
    path: str | None = None,
    title: str | None = None,
    ax: Any = None,
) -> dict[str, Any]:
    """Forest plot: one estimate and its confidence interval per row, with the numbers beside it.

    `data` has one row per estimate. `label` names the column written down the left;
    `estimate` and `se` name the effect and its standard error, from which the `ci` interval
    (95%) is drawn. Pass `lower=` and `upper=` instead where the frame carries the bounds —
    they win over `se`, need not be symmetric, and an infinite one draws a one-sided
    interval. Rows are drawn top to bottom in frame order, so sort the frame first, or pass
    `sort_by="estimate"` for largest first.

    SAY WHAT THE NUMBERS ARE, with `scale`:

    - `"linear"` (default) — drawn as given, null at 0. A quantitative trait's beta.
    - `"log_ratio"` — the estimates are LOG odds/hazard/risk ratios, which is what a
      binary-trait GWAS `beta` and a burden `beta` are. They are exponentiated: the axis is
      logarithmic and labelled in ratio units, the null is 1, and the text column reads
      `1.35 (1.21–1.50)` where a linear one reads `0.12 (0.08 to 0.16)`.
    - `"ratio"` — the estimates are already ratios; same axis, nothing exponentiated. This
      one needs `lower=`/`upper=`, since a standard error is on the log scale.

    Log odds ratios and per-s.d. betas do not share an axis: draw binary and quantitative
    traits as two forests rather than one.

    STRUCTURE. `group=` puts the rows under bold section headings, in order of first
    appearance. `series=` draws several estimates on one row — one per value, each in its
    own colour and marker with a legend — for the same rows measured twice: two cohorts,
    two sexes, discovery and replication. A row is then one `label` (within its group), and
    a series a row lacks leaves its slot empty. `summary=` names a boolean column flagging
    rows that are already pooled estimates — a meta-analysis row from the data — which are
    drawn last in their group, under a hairline, as a diamond spanning the interval.

    `pool="fixed"` or `"random"` computes that diamond instead: the inverse-variance
    (DerSimonian–Laird for random) estimate over each group's rows, or over all of them
    without `group=`, and per series with `series=`. I² and the heterogeneity p are in the
    returned `pooled`, not on the figure — report them in the text. Pool only rows that
    estimate ONE quantity in INDEPENDENT samples — cohorts, ancestries, sexes. Different
    phenotypes are not replicates, and a meta-analysis row pooled with its own components
    counts those samples twice; flag such a row with `summary=`, which also keeps it out
    of the pool.

    ENCODING. Every estimate is a small filled circle unless one of these is asked for.
    `significance=` (a p threshold) fills the markers that pass it and leaves the rest
    hollow, with a legend saying so. `weight=` names a column — sample size, say — and
    draws squares whose area is scaled by it, the meta-analysis convention.

    TEXT. To the right: the estimate with its interval, the p-value when the frame has one
    (`mlog10p`, else `pval`; `pvalue=` names another column, read as -log10 p when its name
    starts with `mlog`, and `pvalue=None` drops it), and then any `columns=` — a list of
    column names, or a dict of column to heading — such as case counts. `table=False`
    leaves all of it out. `effect` is the measure's short name in the heading and the axis
    label ("β", "OR", "HR"); `direction=("lower risk", "higher risk")` writes what each
    side of the null means under the axis; `italic=True` sets the labels in italics, for
    a column of gene symbols.

    WHAT IS HANDLED RATHER THAN LEFT TO THE CALLER. A null or non-finite estimate keeps its
    row and reads `NA`. A missing or non-positive standard error draws the point alone. An
    interval far wider than the others is cut at the axis edge with an arrowhead instead of
    setting the scale, and so is anything outside an explicit `xlim=` (given in the units
    of the axis); the text column still carries the full numbers, and `n_clipped` and
    `xlim` say how many were cut and where, for the legend. Labels are cut at 48
    characters, and no more than `max_rows` rows are drawn — the first ones, with the rest
    counted on the figure and in `n_omitted`; split a longer table into several figures.

    Returns a dict describing what was drawn: `path`, `n_rows`, `n_estimates`, `n_missing`
    (estimates that were null), `n_clipped` (intervals cut at the axis), `n_omitted`,
    `scale`, `xlim` in axis units, `groups`, `series`, `rows` (the labels, top to bottom),
    `size_in` (the width and height in inches the figure needs), and `pooled`: one dict per
    computed diamond with `group`, `series`, `method`, `k`, `estimate`, `lower`, `upper` in
    axis units, `se` on the scale the estimates were given, `mlog10p`, `q`, `i2`, `tau2` and
    `p_het`.

    `path` may be relative, in which case it is written inside the execution's artifacts
    directory and returned to the user automatically; that is also where the default goes.
    Pass `ax` to draw in its place in an existing figure, in which case nothing is saved;
    give that axis about `size_in`, since the text does not shrink to fit.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.markers import CARETLEFT, CARETRIGHT
    from matplotlib.patches import Polygon
    from matplotlib.ticker import FixedFormatter, FixedLocator, MaxNLocator, NullLocator
    from matplotlib.transforms import ScaledTranslation

    if not isinstance(data, pl.DataFrame):
        raise GeneticsUsageError(f"data must be a polars DataFrame, not {type(data).__name__}")
    if scale not in _FOREST_SCALES:
        raise GeneticsUsageError(f"scale must be one of {_FOREST_SCALES}, not {scale!r}")
    if pool is not None and pool not in _FOREST_POOLS:
        raise GeneticsUsageError(
            f"pool must be one of {tuple(_FOREST_POOLS)} or None, not {pool!r}"
        )
    if sort_by is not None and sort_by not in _FOREST_SORTS:
        raise GeneticsUsageError(f"sort_by must be one of {_FOREST_SORTS} or None, not {sort_by!r}")
    if not 0 < ci < 1:
        raise GeneticsUsageError(f"ci is a coverage in (0, 1), e.g. 0.95; got {ci!r}")
    if (lower is None) != (upper is None):
        raise GeneticsUsageError("give both lower= and upper=, or neither")
    if direction is not None and (isinstance(direction, str) or len(direction) != 2):
        raise GeneticsUsageError(
            "direction is a pair: what below the null means, then what above it means"
        )
    if isinstance(columns, str):
        columns = [columns]
    extra = dict(columns) if isinstance(columns, dict) else {c: c for c in (columns or [])}
    if pvalue == "auto":
        pvalue = next((c for c in ("mlog10p", "pval") if c in data.columns), None)
    has_bounds = lower is not None
    if not has_bounds and se is not None and se not in data.columns:
        raise GeneticsUsageError(
            f"no {se!r} column to draw intervals from; name the standard error with se=, "
            f"pass lower= and upper=, or se=None for the estimates alone. "
            f"Columns are {data.columns}"
        )
    if not has_bounds and se is not None and scale == "ratio":
        raise GeneticsUsageError(
            "scale='ratio' needs lower= and upper=: a standard error is on the log scale. "
            "If the estimates are log ratios, that is scale='log_ratio'"
        )
    needed = [label, estimate, lower, upper, group, series, summary, weight, pvalue, *extra]
    missing = [c for c in needed if c is not None and c not in data.columns]
    if missing:
        raise GeneticsUsageError(f"columns {missing} are not in the frame; columns are {data.columns}")
    if data.is_empty():
        raise GeneticsUsageError("the frame has no rows — nothing to plot")
    if significance is not None and pvalue is None:
        raise GeneticsUsageError(
            "significance= needs a p-value per row and the frame has neither `mlog10p` nor "
            "`pval`; name the column with pvalue="
        )

    ratio = scale != "linear"
    z = statistics.NormalDist().inv_cdf(0.5 + ci / 2)
    n = data.height

    def analysis(value: float | None, row: int, column: str) -> float | None:
        """A value on the scale the arithmetic is done in: logged where it came as a ratio."""
        if value is None:
            return None
        if scale == "log_ratio":
            return max(min(value, _FOREST_MAX_LOG), -_FOREST_MAX_LOG)
        if scale == "linear" or value == math.inf:
            return value
        if value == 0 and column == lower:
            return -math.inf
        if value <= 0:
            raise GeneticsUsageError(
                f"scale='ratio' takes ratios, which are positive, but row {row} has "
                f"{column}={value:g}. Log odds ratios — a GWAS `beta` — are scale='log_ratio'"
            )
        return math.log(value)

    estimates = _floats(data, estimate)
    lows = _floats(data, lower, keep_inf=True) if has_bounds else [None] * n
    highs = _floats(data, upper, keep_inf=True) if has_bounds else [None] * n
    ses = _floats(data, se) if se is not None and se in data.columns and scale != "ratio" else [None] * n
    weights = _floats(data, weight) if weight else [None] * n
    if pvalue is None:
        mlog10ps: list[float | None] = [None] * n
    elif pvalue.lower().startswith("mlog"):
        mlog10ps = _floats(data, pvalue, keep_inf=True)
    else:
        mlog10ps = [
            None if p is None or not 0 <= p <= 1 else -math.log10(max(p, 5e-324))
            for p in _floats(data, pvalue)
        ]
    flagged = (
        data[summary].cast(pl.Boolean, strict=False).fill_null(False).to_list()
        if summary else [False] * n
    )
    labels = [_plain("—" if v is None else v, _FOREST_LABEL_CHARS) for v in data[label].to_list()]
    groups_of = (
        [_FOREST_UNGROUPED if v is None else _plain(v, _FOREST_LABEL_CHARS) for v in data[group].to_list()]
        if group else [None] * n
    )
    series_of = (
        ["—" if v is None else _plain(v, _FOREST_CELL_CHARS) for v in data[series].to_list()]
        if series else [None] * n
    )
    series_names = list(dict.fromkeys(series_of)) if series else []
    if len(series_names) > len(_FOREST_SERIES):
        raise GeneticsUsageError(
            f"{series!r} has {len(series_names)} values and at most {len(_FOREST_SERIES)} "
            f"series can be told apart on one row; use group= for the rest"
        )
    extra_cells = {c: [_forest_cell(v) for v in data[c].to_list()] for c in extra}
    threshold = -math.log10(significance) if significance else None

    marks: list[dict[str, Any]] = []
    for i in range(n):
        a = analysis(estimates[i], i, estimate)
        se_a = ses[i] if ses[i] is not None and ses[i] > 0 else None
        if has_bounds:
            lo, hi = analysis(lows[i], i, lower), analysis(highs[i], i, upper)
            if lo is None or hi is None:
                lo = hi = None
            elif lo > hi:
                lo, hi = hi, lo
            if se_a is None and lo is not None and math.isfinite(lo) and math.isfinite(hi) and hi > lo:
                se_a = (hi - lo) / (2 * z)
        elif a is not None and se_a is not None:
            lo, hi = a - z * se_a, a + z * se_a
        else:
            lo = hi = None
        if a is None:
            lo = hi = se_a = None
        mlog10p = mlog10ps[i]
        marks.append({
            "a": a, "lo": lo, "hi": hi, "se": se_a, "mlog10p": mlog10p,
            "weight": weights[i] if weights[i] is not None and weights[i] > 0 else None,
            "series": series_of[i], "diamond": bool(flagged[i]),
            "hollow": threshold is not None and (mlog10p is None or mlog10p < threshold),
            "extra": [extra_cells[c][i] for c in extra],
        })

    # rows, per group in order of first appearance. Without series= a row is a frame row;
    # with it a row is a label, holding one mark per series.
    by_group: dict[str | None, list[dict[str, Any]]] = {}
    index: dict[tuple[str | None, str], dict[str, Any]] = {}
    for i, mark in enumerate(marks):
        rows = by_group.setdefault(groups_of[i], [])
        row = index.get((groups_of[i], labels[i])) if series else None
        if row is None:
            row = {"label": labels[i], "marks": []}
            rows.append(row)
            index[(groups_of[i], labels[i])] = row
        elif any(m["series"] == mark["series"] for m in row["marks"]):
            raise GeneticsUsageError(
                f"two rows share label {labels[i]!r} and {series} {mark['series']!r}"
                + (f" in group {groups_of[i]!r}" if group else "")
                + "; a row is one label per series, so add what tells them apart to the label"
            )
        row["marks"].append(mark)
    for rows in by_group.values():
        for row in rows:
            row["summary"] = all(m["diamond"] for m in row["marks"])

        def order(row: dict[str, Any]) -> tuple[bool, bool, float]:
            first = next((m["a"] for m in row["marks"] if m["a"] is not None), None)
            by_estimate = sort_by == "estimate" and not row["summary"]
            return (
                row["summary"],
                by_estimate and first is None,
                -first if by_estimate and first is not None else 0.0,
            )

        rows.sort(key=order)

    pooled: list[dict[str, Any]] = []
    if pool:
        for name, rows in by_group.items():
            pooled_marks = []
            for s in series_names or [None]:
                usable = [
                    m for row in rows for m in row["marks"]
                    if m["series"] == s and not m["diamond"] and m["a"] is not None and m["se"]
                ]
                if len(usable) < 2:
                    continue
                fit = _forest_pool([m["a"] for m in usable], [m["se"] for m in usable], pool)
                pooled_marks.append({
                    "a": fit["a"], "lo": fit["a"] - z * fit["se"], "hi": fit["a"] + z * fit["se"],
                    "se": fit["se"], "mlog10p": fit["mlog10p"], "weight": None, "series": s,
                    "diamond": True, "hollow": False, "extra": [""] * len(extra),
                })
                pooled.append({"group": name, "series": s, "method": pool, **fit})
            if pooled_marks:
                rows.append({
                    "label": _FOREST_POOLS[pool], "marks": pooled_marks, "summary": True,
                    "pooled": True,
                })

    # the cut: the first max_rows rows of the frame's own estimates, a pooled row kept with
    # whatever of its group survives
    n_omitted, budget = 0, max(int(max_rows), 1)
    for name in list(by_group):
        kept = []
        for row in by_group[name]:
            if row.get("pooled"):
                if kept:
                    kept.append(row)
            elif budget > 0:
                kept.append(row)
                budget -= 1
            else:
                n_omitted += 1
        if kept:
            by_group[name] = kept
        else:
            del by_group[name]
    drawn = [m for rows in by_group.values() for row in rows for m in row["marks"]]

    # a series only takes a slot of its own when some row actually holds two estimates;
    # otherwise it is colour alone and every mark sits on its row's centre line
    dodged = any(len(row["marks"]) > 1 for rows in by_group.values() for row in rows)
    slots = len(series_names) if dodged else 1
    row_in = _FOREST_ROW_IN if slots == 1 else 0.06 + _FOREST_MARK_IN * slots
    text_size = _LABEL_SIZE if slots == 1 else 5

    def shown(a: float) -> float:
        """Analysis scale to axis units."""
        return math.exp(max(min(a, _FOREST_MAX_LOG), -_FOREST_MAX_LOG)) if ratio else a

    if xlim is not None:
        try:
            x_lo, x_hi = float(xlim[0]), float(xlim[1])
        except (TypeError, ValueError, IndexError):
            raise GeneticsUsageError(f"xlim is a (low, high) pair in the units of the axis; got {xlim!r}")
        if not x_lo < x_hi or (ratio and x_lo <= 0):
            raise GeneticsUsageError(
                f"xlim must be increasing{' and positive on a ratio axis' if ratio else ''}; got {xlim!r}"
            )
        lim_lo, lim_hi = (math.log(x_lo), math.log(x_hi)) if ratio else (x_lo, x_hi)
    else:
        lim_lo, lim_hi = _forest_limits(drawn)
        x_lo, x_hi = shown(lim_lo), shown(lim_hi)

    decimals = _forest_decimals([shown(m["a"]) for m in drawn if m["a"] is not None])

    def bound(a: float) -> str:
        if math.isinf(a):
            return ("0" if ratio else "−∞") if a < 0 else "∞"
        return _forest_number(shown(a), decimals, ratio)

    def estimate_text(m: dict[str, Any]) -> str:
        if m["a"] is None:
            return "NA"
        if m["lo"] is None:
            return bound(m["a"])
        # an en dash between two ratios; "to" where a bound can carry a minus sign of its own
        separator = "–" if ratio else " to "
        return f"{bound(m['a'])} ({bound(m['lo'])}{separator}{bound(m['hi'])})"

    if effect is None:
        effect = "OR" if ratio else "β"
    level = f"{ci * 100:g}% CI"
    headings = [f"{effect} ({level})"] + ([r"$P$"] if pvalue else []) + list(extra.values())
    numeric = [False] * (len(headings) - len(extra)) + [data.schema[c].is_numeric() for c in extra]
    if not table:
        headings, numeric = [], []
    for m in drawn:
        m["cells"] = (
            [estimate_text(m)] + ([_forest_p(m["mlog10p"])] if pvalue else []) + m["extra"]
        )[: len(headings)]

    # layout, in inches from the left edge of the label column and the top of the heading row
    indent = _FOREST_INDENT_IN if group else 0.0
    label_in = max(
        [_text_width_in(name, _LABEL_SIZE, bold=True) for name in by_group if name]
        + [
            indent + _text_width_in(row["label"], _LABEL_SIZE)
            for rows in by_group.values() for row in rows
        ]
    )
    column_in = [
        max(
            [_text_width_in(heading, _LABEL_SIZE)]
            + [_text_width_in(m["cells"][j], text_size) for m in drawn]
        )
        for j, heading in enumerate(headings)
    ]
    plot_x = label_in + _FOREST_GUTTER_IN
    column_x, cursor = [], plot_x + _FOREST_PLOT_IN + _FOREST_GUTTER_IN
    for width, right in zip(column_in, numeric):
        column_x.append(cursor + width if right else cursor)
        cursor += width + _FOREST_GUTTER_IN
    width_in = cursor - _FOREST_GUTTER_IN

    y = _FOREST_HEADER_IN
    layout: list[tuple[str, Any, float]] = []
    for name, rows in by_group.items():
        if name is not None:
            layout.append(("group", name, y + _FOREST_HEADER_IN / 2))
            y += _FOREST_HEADER_IN
        for row in rows:
            layout.append(("row", row, y + row_in / 2))
            y += row_in
    height_in = y + 0.04

    handles = []
    if series:
        present = {m["series"] for m in drawn}
        handles += [
            Line2D([0], [0], color=colour, marker=marker, markersize=_FOREST_MARKER_PT,
                   linewidth=_FOREST_INTERVAL_WIDTH, label=name)
            for name, (colour, marker) in zip(series_names, _FOREST_SERIES) if name in present
        ]
    if threshold is not None:
        # 5×10⁻⁸ as it is said, not the 5.0×10⁻⁸ a column of p-values aligns on
        p_text = _forest_p(threshold).replace(r".0\times", r"\times")
        handles += [
            Line2D([0], [0], color=_FOREST_INK, marker="o", markersize=_FOREST_MARKER_PT,
                   linestyle="none", markerfacecolor=face, markeredgewidth=0.7,
                   label=rf"$P {sign}$ {p_text}")
            for face, sign in ((_FOREST_INK, "<"), ("white", r"\geq"))
        ]
    note = f"{n_omitted:,} further row{'' if n_omitted == 1 else 's'} not drawn" if n_omitted else ""

    # points below the axis: tick labels, then the direction line, then the axis label
    below_pt = 12 + (8 if direction else 0) + 11
    top_in = 0.06 + (0.2 if title else 0.0)
    bottom_in = below_pt / 72 + (0.2 if handles or note else 0.0) + 0.05
    size_in = (width_in + 2 * _FOREST_SIDE_IN, height_in + top_in + bottom_in)

    own_figure = ax is None
    if own_figure:
        figure = plt.figure(figsize=size_in)
        canvas = figure.add_axes([
            _FOREST_SIDE_IN / size_in[0], bottom_in / size_in[1],
            width_in / size_in[0], height_in / size_in[1],
        ])
    else:
        figure, canvas = ax.get_figure(), ax
    # two axes over one area: the canvas is the table, measured in inches from its top left,
    # and the interval panel is inset into it so the two share rows by construction
    canvas.set_xlim(0, width_in)
    canvas.set_ylim(height_in, 0)
    canvas.set_axis_off()
    panel = canvas.inset_axes([plot_x / width_in, 0, _FOREST_PLOT_IN / width_in, 1])
    panel.patch.set_visible(False)
    for side in ("top", "left", "right"):
        panel.spines[side].set_visible(False)
    panel.set_yticks([])
    if ratio:
        panel.set_xscale("log")
    panel.set_xlim(x_lo, x_hi)
    panel.set_ylim(height_in, 0)

    canvas.plot([0, width_in], [_FOREST_HEADER_IN] * 2, color=_FOREST_INK,
                linewidth=_AXIS_LINEWIDTH, zorder=1)
    for heading, x, right in zip(headings, column_x, numeric):
        canvas.text(x, _FOREST_HEADER_IN / 2, heading, ha="right" if right else "left",
                    va="center", fontsize=_LABEL_SIZE, color=_FOREST_INK)
    if lim_lo < 0 < lim_hi:
        panel.plot([shown(0.0)] * 2, [_FOREST_HEADER_IN, height_in], color=_FOREST_INK,
                   linewidth=_AXIS_LINEWIDTH, linestyle=(0, (3, 2)), zorder=1)

    heaviest = max((m["weight"] for m in drawn if m["weight"]), default=None)
    style_of = dict(zip(series_names, _FOREST_SERIES))
    n_clipped, banded, ruled = 0, False, False
    for kind, item, y_row in layout:
        if kind == "group":
            banded = ruled = False
            canvas.text(0, y_row, item, ha="left", va="center", fontsize=_LABEL_SIZE,
                        fontweight="bold", color=_FOREST_INK)
            continue
        row = item
        if banded:
            canvas.axhspan(y_row - row_in / 2, y_row + row_in / 2, color=_UPSET_BAND,
                           linewidth=0, zorder=0)
        banded = not banded
        if row["summary"] and not ruled:
            # a hairline over the first summary row of a block: the diamond says what the
            # row is, the rule says where the rows it summarises stop
            canvas.plot([0, width_in], [y_row - row_in / 2] * 2, color=_FOREST_MUTED,
                        linewidth=0.25, zorder=1)
        ruled = row["summary"]
        canvas.text(indent, y_row, row["label"], ha="left", va="center", fontsize=_LABEL_SIZE,
                    fontstyle="italic" if italic and not row.get("pooled") else "normal",
                    color=_FOREST_INK)
        for m in row["marks"]:
            slot = series_names.index(m["series"]) if dodged else 0
            y_mark = y_row + (slot - (slots - 1) / 2) * _FOREST_MARK_IN
            colour, marker = style_of.get(m["series"], (_FOREST_INK, "o"))
            for cell, x, right in zip(m["cells"], column_x, numeric):
                canvas.text(x, y_mark, cell, ha="right" if right else "left", va="center",
                            fontsize=text_size,
                            color=_FOREST_MUTED if cell == "NA" else _FOREST_INK)
            if m["a"] is None:
                continue
            inside = lim_lo <= m["a"] <= lim_hi
            lo, hi = (m["a"], m["a"]) if m["lo"] is None else (m["lo"], m["hi"])
            cut_lo, cut_hi = lo < lim_lo, hi > lim_hi
            n_clipped += cut_lo or cut_hi
            seg_lo, seg_hi = max(lo, lim_lo), min(hi, lim_hi)
            if m["diamond"] and m["lo"] is not None and inside and not (cut_lo or cut_hi):
                panel.add_patch(Polygon(
                    [(shown(lo), y_mark), (shown(m["a"]), y_mark - _FOREST_DIAMOND_IN),
                     (shown(hi), y_mark), (shown(m["a"]), y_mark + _FOREST_DIAMOND_IN)],
                    closed=True, facecolor=colour, edgecolor=colour, linewidth=0.5, zorder=3,
                ))
                continue
            if seg_lo < seg_hi:
                panel.plot([shown(seg_lo), shown(seg_hi)], [y_mark] * 2, color=colour,
                           linewidth=_FOREST_INTERVAL_WIDTH, solid_capstyle="butt", zorder=2)
            # an arrowhead where the interval leaves the axis; an estimate wholly outside
            # it is an arrowhead alone, on the side it lies
            for cut, edge, head in ((cut_lo, lim_lo, CARETLEFT), (cut_hi, lim_hi, CARETRIGHT)):
                if cut:
                    panel.plot([shown(edge)], [y_mark], marker=head, markersize=3,
                               color=colour, linestyle="none", clip_on=False, zorder=4)
            if not inside:
                continue
            side_pt = _FOREST_MARKER_PT
            if heaviest and m["weight"]:
                small, large = _FOREST_WEIGHT_PT
                side_pt = math.sqrt(small ** 2 + (large ** 2 - small ** 2) * m["weight"] / heaviest)
                marker = "s"
            panel.plot([shown(m["a"])], [y_mark], marker="D" if m["diamond"] else marker,
                       markersize=side_pt, color=colour, linestyle="none",
                       markerfacecolor="white" if m["hollow"] else colour,
                       markeredgecolor=colour, markeredgewidth=0.7, zorder=3)

    if ratio:
        ticks = _forest_ratio_ticks(x_lo, x_hi)
        panel.xaxis.set_major_locator(FixedLocator(ticks))
        panel.xaxis.set_major_formatter(FixedFormatter([_forest_tick_label(t) for t in ticks]))
        panel.xaxis.set_minor_locator(NullLocator())
    else:
        panel.xaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
    _size(panel)
    if xlabel is None:
        xlabel = f"{_FOREST_EFFECT_NAMES.get(effect, effect)} ({level})"
    panel.set_xlabel(xlabel, fontsize=_LABEL_SIZE, labelpad=2 + (8 if direction else 0))
    if direction is not None:
        for text, x, ha in ((f"← {direction[0]}", 0, "left"), (f"{direction[1]} →", 1, "right")):
            panel.annotate(_plain(text, _FOREST_LABEL_CHARS), (x, 0), xycoords="axes fraction",
                           xytext=(0, -12), textcoords="offset points", ha=ha, va="top",
                           fontsize=5, color=_FOREST_MUTED)
    under = ScaledTranslation(0, -below_pt / 72, figure.dpi_scale_trans)
    if handles:
        panel.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0),
                     bbox_transform=panel.transAxes + under, ncol=min(len(handles), 4),
                     frameon=False, fontsize=5, handlelength=1.8, columnspacing=1.2,
                     borderaxespad=0.2)
    if note:
        canvas.annotate(note, (0, 0), xycoords="axes fraction", xytext=(0, -below_pt),
                        textcoords="offset points", ha="left", va="top", fontsize=5,
                        color=_FOREST_MUTED)
    if title:
        canvas.set_title(title, fontsize=_TITLE_SIZE, loc="left", pad=3)

    written = None
    if own_figure:
        written = _resolve_path(path, "forest.png")
        figure.savefig(written)
        plt.close(figure)

    for fit in pooled:
        a, se_fit = fit.pop("a"), fit["se"]
        fit.update(estimate=shown(a), lower=shown(a - z * se_fit), upper=shown(a + z * se_fit))
    return {
        "path": written,
        "n_rows": sum(len(rows) for rows in by_group.values()),
        "n_estimates": sum(m["a"] is not None for m in drawn),
        "n_missing": sum(m["a"] is None for m in drawn),
        "n_clipped": int(n_clipped),
        "n_omitted": n_omitted,
        "scale": scale,
        "xlim": (x_lo, x_hi),
        "groups": [name for name in by_group if name is not None],
        "series": series_names,
        "rows": [row["label"] for rows in by_group.values() for row in rows],
        "size_in": size_in,
        "pooled": pooled,
    }
