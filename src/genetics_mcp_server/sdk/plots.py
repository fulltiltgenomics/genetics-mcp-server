"""Standard plots, drawn the same way every time.

    import genetics
    genetics.plots.locuszoom(phenotype="H8_HEARINGLOSS", variant="12:49578357:C:T")
    genetics.plots.phewas(variant="19:44908684:T:C")

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
import textwrap
from typing import Any

import polars as pl

from genetics_mcp_server.sdk.errors import GeneticsUsageError

__all__ = ["locuszoom", "phewas"]

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
    af = row.get("af")
    if af is not None:
        parts.append(f"AF {float(af):.4g}")
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
    drawable = sorted(
        (d for d in (_drawable(g) for g in genes.iter_rows(named=True)) if d),
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

def _significance_line(ax, significance: float) -> float:
    """The significance line and its label, drawn the same way on every plot; returns its y.

    The label sits on the line rather than in a legend box: grey scaffolding named where it
    is, so the figure carries no legend unless something else needs one. Returns 0.0 and
    draws nothing when `significance` is falsy.
    """
    if not significance:
        return 0.0
    line_y = -math.log10(significance)
    ax.axhline(line_y, color=_SIGNIFICANCE_GREY, linewidth=0.6, linestyle="--", zorder=1)
    # `:g` renders 5e-8 as "5e-08"; the padded exponent is not how anyone writes it
    ax.text(0.006, line_y, f"p {significance:g}".replace("e-0", "e-"),
            transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=6,
            color=_SIGNIFICANCE_GREY)
    return line_y


def _dress(ax, title: str) -> None:
    """The y label, the title and the type sizes every one of these plots shares."""
    ax.set_ylabel(r"$-\log_{10}(p)$", fontsize=_LABEL_SIZE)
    ax.set_title(title, fontsize=_TITLE_SIZE)
    ax.tick_params(labelsize=_TICK_SIZE, width=_AXIS_LINEWIDTH, length=_TICK_LENGTH)
    for spine in ax.spines.values():
        spine.set_linewidth(_AXIS_LINEWIDTH)


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
    centred with `flank` either side. `lead` defaults to the strongest association in the
    window, and LD is taken against it from the FinnGen LD server.

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
    `phenotype`, `n_variants`, `n_genes`, `n_exons`, plus two worth reading every time.

    `ld_joined` is False when the LD server returned nothing for the lead — a plot with grey
    points rather than an error, because a locuszoom without LD is still the right picture of
    the locus. Check it rather than assuming the colours mean something: the LD server is a
    third party, reached through a proxy, so an outage there costs the colours and nothing
    else.

    `ld_partners_outside_window` lists the variants correlated with the lead that fall
    outside the window, strongest first, as {variant, pos, r2}. It is non-empty when the
    window is too narrow for the locus — the signal has support the plot does not show — and
    the fix is to redraw with a larger `flank` or an explicit `region`. The figure carries
    the same warning so a reader who never sees this dict is not misled.

    `coding_marked` is False when the consequence lookup did not answer, in which case every
    point is a circle and shape means nothing; set `coding=False` to skip that fetch outright.
    The same lookup fills `lead_consequence` and `lead_gene`, which the lead's label also
    carries, so the strongest variant names what it does and where before anyone asks.

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

    if lead is None:
        lead_row = frame.sort("_y", descending=True).row(0, named=True)
        lead = lead_row["_variant_id"] or f"{lead_row.get('chr', chrom)}:{lead_row['pos']}"
    else:
        match = frame.filter(pl.col("_variant_id") == _norm_variant_id(lead))
        if match.is_empty():
            raise GeneticsUsageError(f"lead {lead!r} is not among the variants in {region}")
        lead_row = match.row(0, named=True)
    lead_pos, lead_y = lead_row["pos"], lead_row["_y"]
    lead_id = _norm_variant_id(lead)

    # the window the DATA covers. Computed here rather than at plotting time because the LD
    # request is sized from it: `flank` is meaningless when the caller gave region=, and the
    # old `2 * flank` asked for the default width regardless of what was actually drawn.
    span_lo, span_hi = int(frame["pos"].min()), int(frame["pos"].max())

    ld_frame = None
    outside: list[dict[str, Any]] = []
    if ld:
        try:
            ld_frame = sdk.ld(
                lead_id,
                # the server's `window` is the TOTAL span it centres on the lead, so this is
                # _LD_SEARCH_SPAN_MULTIPLE times the plotted width — wide enough to cover the
                # window from a lead anywhere inside it, and to see just past both edges
                window=max(_LD_SEARCH_SPAN_MULTIPLE * max(span_hi - span_lo, 1), 200_000),
                r2_threshold=_LD_MIN_R2,
                panel=ld_panel,
            )
        except Exception:
            # the LD server is a third party and its absence must not lose the figure; the
            # returned ld_joined=False is how a caller learns the colours mean nothing
            ld_frame = None
        else:
            outside = _partners_outside(ld_frame, span_lo, span_hi)
    colours, r2_values, ld_joined = _ld_colours(frame, lead_id, ld_frame)
    frame = frame.with_columns(pl.Series("_r2", r2_values, dtype=pl.Float64))

    consequences = _consequences(region) if coding else None
    coding_marked = consequences is not None
    coding_flags = [
        consequences.get(vid, (None, None))[0] in _CODING_CONSEQUENCES if consequences else False
        for vid in frame["_variant_id"]
    ]
    lead_consequence, lead_gene = (consequences or {}).get(lead_id, (None, None))

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

    lead_index = next(
        (i for i, vid in enumerate(frame["_variant_id"]) if vid == lead_id), None
    )
    lead_coding = bool(lead_index is not None and coding_flags[lead_index])
    ax.scatter([lead_pos], [lead_y], marker=_MARKER_CODING if lead_coding else _MARKER_OTHER,
               s=40, c=_LEAD_COLOUR, edgecolors="black", linewidths=0.4, zorder=3)
    # below the point: above it, the label of a lead at the top of the panel runs into the
    # axes frame, which is where the strongest association always sits
    ax.annotate(_lead_label(lead_id, lead_row, lead_y, lead_consequence, lead_gene),
                (lead_pos, lead_y),
                textcoords="offset points", xytext=(0, -9), ha="center", va="top",
                fontsize=6)
    line_y = _significance_line(ax, significance)

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
        ax.legend(handles=handles, title=f"LD $r^2$ to {lead_id} ({ld_panel})",
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
    ax.set_xlabel(f"position on chromosome {chrom_label}".rstrip(), fontsize=_LABEL_SIZE)

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
        "ld_joined": ld_joined,
        "ld_partners_outside_window": outside,
        "coding_marked": coding_marked,
        "lead_consequence": lead_consequence,
        "lead_gene": lead_gene,
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
