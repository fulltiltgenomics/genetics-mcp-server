"""Standard plots, drawn the same way every time.

    import genetics
    genetics.plots.locuszoom(phenotype="H8_HEARINGLOSS", variant="12:49578357:C:T")

WHY THESE ARE FUNCTIONS AND NOT INSTRUCTIONS. A locuszoom has conventions a script rederives
badly under time pressure: which axis is -log10 p, that the LD ramp is binned rather than
continuous, that the lead variant is a diamond, that genes belong under the association panel
and not beside it. Written out per request, each of those is a coin flip. Written here once,
they are the same in every conversation and a defect is fixed in one place.

WHY NOT A TOOL. A tool is a round trip with a fixed argument list; this is a Python function,
so a script can take the axes back and add to them, or call it for each of several phenotypes
in one execution and lay the results out itself. Every function here takes `ax=` for that
reason and returns what it drew rather than only a path.

STYLE IS NOT SET HERE. These draw under whatever rcParams are in force — matplotlib's own
defaults plus the render density the sandbox bakes (genetics-results-suite
sandbox/gen_mplrc.py) — so a caller who prefers another style sets it and these follow. What
they DO set is the LD colour ramp and the two marker shapes, because both are semantic: a
reader decodes r² from the colours and consequence from the shapes, so neither may follow a
style's prop_cycle.

ADDING A PLOT. Write the function, export it in `__all__`, and give it a docstring whose first
line reads as a description: `list_capabilities(module="plots")` and the generated
`sandbox/stubs/plots.pyi` both derive from this module, so nothing else has to be updated for
a script's author to discover it.
"""

from __future__ import annotations

import math
import os
from typing import Any

import polars as pl

from genetics_mcp_server.sdk.errors import GeneticsUsageError

__all__ = ["locuszoom"]

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

# VEP terms that place a variant in a coding sequence or change the protein. Deliberately NOT
# including `splice_region_variant`: VEP assigns it up to 8 bp into an intron, so a square
# there would claim a coding position the term does not establish. The two splice-site terms
# ARE here — they abolish a site and are scored as loss of function.
_CODING_CONSEQUENCES = frozenset({
    "coding_sequence_variant",
    "frameshift_variant",
    "inframe_deletion",
    "inframe_insertion",
    "incomplete_terminal_codon_variant",
    "missense_variant",
    "protein_altering_variant",
    "splice_acceptor_variant",
    "splice_donor_variant",
    "start_lost",
    "start_retained_variant",
    "stop_gained",
    "stop_lost",
    "stop_retained_variant",
    "synonymous_variant",
    "transcript_ablation",
    "transcript_amplification",
})

# the significance line and its label are scaffolding, not data: grey keeps them behind the
# points instead of competing with the LD ramp for the reader's attention
_SIGNIFICANCE_GREY = "#888888"
_WARNING_COLOUR = "#AA0000"

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

# r² at which a partner outside the window is worth reporting: the first bin above "< 0.2",
# i.e. the first whose colour a reader would have noticed had it been in frame.
_LD_NOTABLE_R2 = 0.2


def _artifacts_dir() -> str:
    """Where a sandbox execution's files are collected. Outside the sandbox, the cwd."""
    return os.environ.get("SANDBOX_ARTIFACTS_DIR") or "."


def _resolve_path(path: str | None) -> str:
    """Where the figure is written. A relative name lands in the artifacts directory.

    Not `path or <default>`: a relative `path=` used to be written to the process cwd, which
    the sandbox does not collect, so a correct call drew the figure and returned nothing. The
    caller's only clue was an empty artifact list, and the docstring promised the opposite.
    An absolute path is honoured as given.
    """
    if path is None:
        return os.path.join(_artifacts_dir(), "locuszoom.png")
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


def _lead_label(variant_id: str, row: dict[str, Any], mlog10p: float) -> str:
    """What the annotation says: the id, then whichever of p, beta and AF the frame carries.

    Built from what is present rather than from a fixed list, because `data=` may be any frame
    a caller assembled and a KeyError there would lose the whole figure for a caption.
    """
    parts = [f"p={_format_p(mlog10p)}"]
    beta = row.get("beta")
    if beta is not None:
        parts.append(f"beta={float(beta):.3g}")
    af = row.get("af")
    if af is not None:
        parts.append(f"AF={float(af):.4g}")
    return f"{variant_id}\n" + "  ".join(parts)


def _coding_flags(frame: pl.DataFrame, region: str) -> tuple[list[bool], bool]:
    """One flag per row: does this variant sit in a coding sequence.

    The consequence is not in the summary statistics, so it is a second fetch. Failure is
    tolerated the same way the LD fetch is — every point falls back to a circle and the figure
    is still the right picture of the locus.
    """
    from genetics_mcp_server import sdk

    try:
        annotations = sdk.variant_annotation(region=region)
    except Exception:
        return [False] * frame.height, False
    if annotations.is_empty() or not {"variant", "most_severe"} <= set(annotations.columns):
        return [False] * frame.height, False
    coding_ids = {
        _norm_variant_id(row["variant"])
        for row in annotations.iter_rows(named=True)
        if row.get("most_severe") in _CODING_CONSEQUENCES
    }
    return [vid in coding_ids for vid in frame["_variant_id"]], True


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


def _draw_genes(ax, genes: pl.DataFrame, start: int, end: int, max_rows: int = 4) -> int:
    """Gene bodies on a small track, packed into non-overlapping rows. Returns rows used."""
    if genes.is_empty() or "gene_start" not in genes.columns:
        ax.set_yticks([])
        return 0
    ordered = genes.sort("gene_start")
    row_ends: list[int] = []
    drawn = 0
    span = max(end - start, 1)
    for gene in ordered.iter_rows(named=True):
        g_start, g_end = gene.get("gene_start"), gene.get("gene_end")
        name = gene.get("gene_name") or gene.get("hgnc_symbol") or ""
        if g_start is None or g_end is None:
            continue
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
        ax.plot([left, right], [-row, -row], linewidth=2.0, solid_capstyle="butt",
                color="#4A4A4A")
        strand = (gene.get("gene_strand") or "").strip()
        label = f"{name}{'→' if strand == '+' else '←' if strand == '-' else ''}"
        ax.text((left + right) / 2, -row + 0.22, label, ha="center", va="bottom",
                fontsize=5, color="#222222")
        drawn += 1
    ax.set_yticks([])
    ax.set_ylim(-max(len(row_ends), 1) + 0.2, 0.9)
    return drawn


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

    Colour is LD and shape is consequence, so a coding variant in high LD reads as both at
    once: a square sits in a coding sequence or changes the protein, a circle does not, and
    the lead takes whichever shape its own consequence gives it. Grey means "no r² to show" —
    either the LD panel does not carry the variant or its r² is below the floor these plots
    colour at. Only the correlated variants are coloured, so the ramp reads at a glance
    instead of painting the whole cloud navy.

    The gene track draws gene bodies only. The annotation this SDK can reach is GENCODE
    gene-level — start, end, strand, biotype — with no transcript or exon structure, so there
    are no exon boxes to draw.

    Returns a dict describing what was drawn: `path`, `lead`, `lead_mlog10p`, `region`,
    `phenotype`, `n_variants`, `n_genes`, plus two worth reading every time.

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

    coding_flags, coding_marked = (
        _coding_flags(frame, region) if coding else ([False] * frame.height, False)
    )

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
    ax.annotate(_lead_label(lead_id, lead_row, lead_y), (lead_pos, lead_y),
                textcoords="offset points", xytext=(0, -9), ha="center", va="top",
                fontsize=6)
    line_y = -math.log10(significance) if significance else 0.0
    if significance:
        ax.axhline(line_y, color=_SIGNIFICANCE_GREY, linewidth=0.6, linestyle="--", zorder=1)
        # `:g` renders 5e-8 as "5e-08"; the padded exponent is not how anyone writes it
        ax.text(0.006, line_y, f"p={significance:g}".replace("e-0", "e-"),
                transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=6,
                color=_SIGNIFICANCE_GREY)

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
            f"nearest {gap / 1000:.0f} kb out, r$^2$={nearest['r2']:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7,
            color=_WARNING_COLOUR,
        )

    ax.set_ylabel(r"$-\log_{10}(p)$")
    ax.set_title(title if title is not None else f"{phenotype} — {region}")
    ax.margins(x=0.02)
    if ld_joined:
        handles = [
            plt.Line2D([], [], marker="o", linestyle="", markersize=4, color=colour,
                       label=label)
            for _threshold, colour, label in _LD_BINS
        ]
        ax.legend(handles=handles, title=r"$r^2$", fontsize=5, title_fontsize=5,
                  loc="upper left", ncol=1)

    # pinned before the gene track can widen it through sharex
    pad = max((span_hi - span_lo) * 0.02, 1)
    ax.set_xlim(span_lo - pad, span_hi + pad)

    # headroom, so the corner notes have somewhere to sit: autoscaling leaves the strongest
    # association at the top of the panel, which is exactly where the legend and the
    # outside-window warning are drawn
    y_top = max(float(frame["_y"].max()), line_y if significance else 0.0)
    ax.set_ylim(top=y_top + max(y_top * (0.22 if outside else 0.08), 0.5))

    n_genes = 0
    if gene_ax is not None:
        n_genes = _draw_genes(gene_ax, gene_frame, span_lo, span_hi)
        gene_ax.set_xlabel(f"position on chromosome {frame['chr'][0] if 'chr' in frame.columns else ''}".rstrip())
    else:
        ax.set_xlabel("position")

    written = None
    if own_figure:
        written = _resolve_path(path)
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
        "ld_joined": ld_joined,
        "ld_partners_outside_window": outside,
        "coding_marked": coding_marked,
    }
