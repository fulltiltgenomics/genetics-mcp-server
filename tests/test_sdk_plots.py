"""The standard plots draw the right picture from the frames the SDK actually returns.

Everything here passes `data=` and turns the LD and gene fetches off, so no test reaches a
service: what is under test is the drawing and the decisions around it, not the client.

What these CANNOT check is the rcParams — render density comes from a matplotlibrc the sandbox
image bakes, so a figure drawn here is drawn under this venv's matplotlib defaults. That half
is asserted where it is true, in genetics-results-suite/sandbox/build-checks.py, which also
asserts that no STYLE is baked in: scienceplots ships in the image but a script has to ask.
"""

import math

import polars as pl
import pytest

from genetics_mcp_server.sdk import plots
from genetics_mcp_server.sdk.errors import GeneticsUsageError


def sumstats(rows):
    return pl.DataFrame(
        rows,
        schema={
            "chr": pl.Utf8,
            "pos": pl.Int64,
            "ref": pl.Utf8,
            "alt": pl.Utf8,
            "pval": pl.Float64,
            "mlog10p": pl.Float64,
        },
    )


def frame_of(n=25, with_mlog10p=True):
    rows = []
    for i in range(n):
        pval = 10 ** -(2 + (i % 7))
        rows.append(
            {
                "chr": "12",
                "pos": 49_500_000 + i * 1_000,
                "ref": "C",
                "alt": "T",
                "pval": pval,
                "mlog10p": -math.log10(pval) if with_mlog10p else None,
            }
        )
    # one unambiguous lead, so the "strongest association" choice is testable
    rows[3]["pval"] = 1e-30
    rows[3]["mlog10p"] = 30.0 if with_mlog10p else None
    return sumstats(rows)


def test_locuszoom_writes_a_figure_and_reports_what_it_drew(tmp_path):
    out = tmp_path / "lz.png"
    result = plots.locuszoom(
        phenotype="H8_HEARINGLOSS",
        region="12:49400000-49600000",
        data=frame_of(),
        ld=False,
        genes=False,
        path=str(out),
    )
    assert out.exists() and out.stat().st_size > 0
    assert result["path"] == str(out)
    assert result["n_variants"] == 25
    assert result["lead"] == "12:49503000:C:T"
    assert result["lead_mlog10p"] == pytest.approx(30.0)
    # no LD was fetched, so the caller must be able to see that the colours mean nothing
    assert result["ld_joined"] is False


def test_the_lead_comes_from_mlog10p_when_the_p_value_has_underflowed(tmp_path):
    """A p-value of 0.0 in the file is the case that decides which column to trust: -log10(0)
    is inf, and an inf silently takes the y axis with it."""
    rows = frame_of().to_dicts()
    rows[10]["pval"] = 0.0
    rows[10]["mlog10p"] = 44.0
    result = plots.locuszoom(
        phenotype="X",
        region="12:49400000-49600000",
        data=sumstats(rows),
        ld=False,
        genes=False,
        path=str(tmp_path / "lz.png"),
    )
    assert result["lead"] == "12:49510000:C:T"
    assert result["lead_mlog10p"] == pytest.approx(44.0)
    assert math.isfinite(result["lead_mlog10p"])


def test_a_missing_mlog10p_column_falls_back_to_the_p_value(tmp_path):
    result = plots.locuszoom(
        phenotype="X",
        region="12:49400000-49600000",
        data=frame_of(with_mlog10p=False).drop("mlog10p"),
        ld=False,
        genes=False,
        path=str(tmp_path / "lz.png"),
    )
    assert result["lead_mlog10p"] == pytest.approx(30.0)


def test_variant_centres_a_region_and_region_and_variant_are_exclusive(tmp_path):
    result = plots.locuszoom(
        phenotype="X",
        variant="chr12:49503000:C:T",
        flank=50_000,
        data=frame_of(),
        ld=False,
        genes=False,
        path=str(tmp_path / "lz.png"),
    )
    assert result["region"] == "12:49453000-49553000"

    with pytest.raises(GeneticsUsageError, match="not both"):
        plots.locuszoom(
            phenotype="X", region="12:1-2", variant="12:3:A:G", data=frame_of(),
            ld=False, genes=False,
        )
    with pytest.raises(GeneticsUsageError, match="region= or variant="):
        plots.locuszoom(phenotype="X", data=frame_of(), ld=False, genes=False)


def test_an_empty_region_says_so_instead_of_drawing_an_empty_panel():
    with pytest.raises(GeneticsUsageError, match="nothing to plot"):
        plots.locuszoom(
            phenotype="X",
            region="12:1-2",
            data=sumstats([]),
            ld=False,
            genes=False,
        )


def test_ld_colours_bin_r2_the_conventional_way_and_flag_the_lead():
    frame = pl.DataFrame(
        {"_variant_id": ["12:1:A:G", "12:2:A:G", "12:3:A:G", "12:4:A:G", "12:5:A:G"]}
    )
    ld = pl.DataFrame(
        {
            "variant": ["12:2:A:G", "12:3:A:G", "12:4:A:G"],
            "r2": [0.95, 0.5, 0.05],
        }
    )
    colours, values, joined = plots._ld_colours(frame, "12:1:A:G", ld)
    assert joined is True
    assert colours[0] == plots._LEAD_COLOUR
    assert colours[1] == "#D43F3A"  # >= 0.8
    assert colours[2] == "#5CB85C"  # 0.4 - 0.6
    assert colours[3] == "#357EBD"  # < 0.2
    assert colours[4] == plots._LD_UNKNOWN  # absent from the LD answer
    assert values[4] is None


def test_ld_colours_survive_an_ld_server_that_answered_with_nothing():
    frame = pl.DataFrame({"_variant_id": ["12:1:A:G", "12:2:A:G"]})
    for empty in (None, pl.DataFrame()):
        colours, _values, joined = plots._ld_colours(frame, "12:1:A:G", empty)
        assert joined is False
        assert colours[1] == plots._LD_UNKNOWN


def test_drawing_into_a_caller_supplied_axis_saves_nothing(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _figure, ax = plt.subplots()
    result = plots.locuszoom(
        phenotype="X",
        region="12:49400000-49600000",
        data=frame_of(),
        ld=False,
        genes=False,
        ax=ax,
    )
    assert result["path"] is None
    assert not list(tmp_path.iterdir())
    assert ax.collections, "nothing was drawn into the supplied axis"
    plt.close("all")


def test_the_artifacts_directory_is_the_default_destination(tmp_path, monkeypatch):
    """A figure saved anywhere else is not returned to the user, which is the whole point."""
    monkeypatch.setenv("SANDBOX_ARTIFACTS_DIR", str(tmp_path))
    result = plots.locuszoom(
        phenotype="X", region="12:49400000-49600000", data=frame_of(), ld=False, genes=False
    )
    assert result["path"] == str(tmp_path / "locuszoom.png")
    assert (tmp_path / "locuszoom.png").exists()


def test_every_exported_plot_has_a_docstring_the_catalogue_can_render():
    """list_capabilities and the generated stub both read these; an undocumented export
    reaches a script as a bare signature."""
    for name in plots.__all__:
        function = getattr(plots, name)
        assert function.__doc__ and function.__doc__.strip(), f"{name} has no docstring"


def test_a_gene_overlapping_the_boundary_does_not_widen_the_association_panel(
    tmp_path, monkeypatch
):
    """Measured on staging before this was pinned: gene_annotations returns a gene WHOLE when
    it merely overlaps the window, and on a shared x axis its far end dragged the limits out,
    squeezing the association panel into a fraction of the figure."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from genetics_mcp_server import sdk

    frame = frame_of()
    overhanging = pl.DataFrame(
        {
            "gene_name": ["TUBA1C"],
            "gene_start": [49_490_000],
            # a long way past the last variant, the way a real gene overlapping the edge is
            "gene_end": [52_000_000],
            "gene_strand": ["+"],
        }
    )
    monkeypatch.setattr(sdk, "gene_annotations", lambda **_kw: overhanging, raising=False)

    drawn = {}
    original = matplotlib.figure.Figure.savefig

    def record(self, *args, **kwargs):
        drawn["figure"] = self
        return original(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", record)

    result = plots.locuszoom(
        phenotype="X",
        region="12:49400000-49600000",
        data=frame,
        ld=False,
        genes=True,
        path=str(tmp_path / "lz.png"),
    )
    assert result["n_genes"] == 1
    _lo, hi = drawn["figure"].axes[0].get_xlim()
    assert hi < frame["pos"].max() + 10_000, (
        f"the gene track widened the shared axis to {hi}"
    )
    plt.close("all")


def test_the_x_axis_covers_the_data_and_not_much_more(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _figure, ax = plt.subplots()
    frame = frame_of()
    plots.locuszoom(
        phenotype="X", region="12:49400000-49600000", data=frame, ld=False, genes=False, ax=ax
    )
    lo, hi = ax.get_xlim()
    span = frame["pos"].max() - frame["pos"].min()
    assert lo == pytest.approx(frame["pos"].min() - span * 0.02)
    assert hi == pytest.approx(frame["pos"].max() + span * 0.02)
    plt.close("all")


def test_a_relative_path_lands_in_the_artifacts_directory(tmp_path, monkeypatch):
    """A relative name used to be written to the process cwd, which the sandbox does not
    collect: the call succeeded, the figure was drawn, and nothing came back."""
    artifacts, cwd = tmp_path / "artifacts", tmp_path / "cwd"
    artifacts.mkdir()
    cwd.mkdir()
    monkeypatch.setenv("SANDBOX_ARTIFACTS_DIR", str(artifacts))
    monkeypatch.chdir(cwd)
    result = plots.locuszoom(
        phenotype="X", region="12:49400000-49600000", data=frame_of(), ld=False,
        genes=False, path="hearing_loss.png",
    )
    assert result["path"] == str(artifacts / "hearing_loss.png")
    assert (artifacts / "hearing_loss.png").exists()
    assert not list(cwd.iterdir()), "the figure was written to the cwd and would be lost"


def test_an_absolute_path_is_honoured_as_given(tmp_path, monkeypatch):
    monkeypatch.setenv("SANDBOX_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    (tmp_path / "artifacts").mkdir()
    target = tmp_path / "somewhere_else.png"
    result = plots.locuszoom(
        phenotype="X", region="12:49400000-49600000", data=frame_of(), ld=False,
        genes=False, path=str(target),
    )
    assert result["path"] == str(target)
    assert target.exists()
    assert not list((tmp_path / "artifacts").iterdir())


def test_ld_is_asked_for_over_more_than_the_plotted_span_and_above_a_floor(
    monkeypatch, tmp_path
):
    """Both halves matter. r2>=0 makes the server answer with the whole panel, which buries
    the colour and truncates positionally; a window equal to the plot cannot see a partner
    just outside it."""
    from genetics_mcp_server import sdk

    seen = {}

    def fake_ld(variant, *, window, r2_threshold, panel):
        seen.update(variant=variant, window=window, r2_threshold=r2_threshold, panel=panel)
        return pl.DataFrame({"variant": [], "r2": []}, schema={"variant": pl.Utf8, "r2": pl.Float64})

    monkeypatch.setattr(sdk, "ld", fake_ld)
    monkeypatch.setenv("SANDBOX_ARTIFACTS_DIR", str(tmp_path))
    frame = frame_of(n=25)  # 24 kb of positions
    plots.locuszoom(
        phenotype="X", region="12:49400000-49600000", data=frame, genes=False, ld=True,
    )
    span = int(frame["pos"].max()) - int(frame["pos"].min())
    assert seen["r2_threshold"] == plots._LD_MIN_R2 > 0
    assert seen["window"] >= plots._LD_SEARCH_SPAN_MULTIPLE * span
    assert seen["panel"] == "sisu42"


def test_a_correlated_partner_outside_the_window_is_reported_and_drawn(monkeypatch, tmp_path):
    """The failure this exists for: a ±250 kb plot of 12:49272869:C:T excludes
    12:49564549:G:A, which is r²=0.78 with the lead and more significant than it."""
    from genetics_mcp_server import sdk

    def fake_ld(variant, *, window, r2_threshold, panel):
        return pl.DataFrame(
            {
                # inside the window, and one 64 kb past its right edge
                "variant": ["12:49505000:A:G", "12:49564549:G:A", "12:49570000:T:C"],
                "r2": [0.9, 0.78, 0.05],
            }
        )

    monkeypatch.setattr(sdk, "ld", fake_ld)
    monkeypatch.setenv("SANDBOX_ARTIFACTS_DIR", str(tmp_path))
    result = plots.locuszoom(
        phenotype="X", region="12:49400000-49600000", data=frame_of(), genes=False, ld=True,
    )
    outside = result["ld_partners_outside_window"]
    # 0.05 is below the reporting floor, 0.9 is inside the plotted span
    assert [p["variant"] for p in outside] == ["12:49564549:G:A"]
    assert outside[0]["pos"] == 49_564_549
    assert outside[0]["r2"] == pytest.approx(0.78)


def test_nothing_outside_the_window_reports_an_empty_list_not_a_missing_key(tmp_path, monkeypatch):
    monkeypatch.setenv("SANDBOX_ARTIFACTS_DIR", str(tmp_path))
    result = plots.locuszoom(
        phenotype="X", region="12:49400000-49600000", data=frame_of(), ld=False, genes=False
    )
    assert result["ld_partners_outside_window"] == []


def test_partners_outside_ranks_by_r2_and_ignores_what_it_cannot_place():
    ld = pl.DataFrame(
        {
            "variant": ["12:100:A:G", "12:900:A:G", "12:950:A:G", "not-a-variant", "12:500:A:G"],
            "r2": [0.3, 0.9, None, 0.99, 0.95],
        }
    )
    found = plots._partners_outside(ld, 400, 600)
    assert [p["r2"] for p in found] == [0.9, 0.3]
    assert plots._partners_outside(None, 400, 600) == []


def test_dir_of_the_sdk_package_lists_the_lazily_resolved_plots_module():
    """`dir(genetics)` is the obvious probe after a wrong guess at a name, and a module
    __getattr__ without a matching __dir__ answers that the submodule does not exist."""
    import importlib
    import sys

    sys.modules.pop("genetics_mcp_server.sdk.plots", None)
    sdk = importlib.reload(importlib.import_module("genetics_mcp_server.sdk"))
    assert "plots" in dir(sdk), "dir() hides the submodule until something touches it"
    assert not hasattr(sdk, "locuszoom"), "plots functions must not leak to the top level"


@pytest.mark.parametrize(
    "mlog10p,expected",
    [
        (12.6814, "2.08e-13"),   # 12:49272869:C:T for hearing loss, p=2.08257e-13
        (17.1694, "6.77e-18"),   # the same variant for potassium, p=6.77018e-18
        (0.282271, "5.22e-1"),   # p=0.52207
        (13.0, "1.00e-13"),      # integral: 10.0e-14 must normalise
        (400.5, "3.16e-401"),    # 10 ** -400.5 is 0.0 in a float; this is the whole point
        (0.0, "1"),
        (None, "1"),
    ],
)
def test_the_p_value_is_taken_apart_from_mlog10p_rather_than_computed(mlog10p, expected):
    assert plots._format_p(mlog10p) == expected


def test_the_lead_label_carries_p_beta_and_af_and_skips_what_is_absent():
    full = plots._lead_label("12:1:A:G", {"beta": 1.17569, "af": 0.00234211}, 12.6814)
    assert full == "12:1:A:G\np=2.08e-13  beta=1.18  AF=0.002342"
    # `data=` may be any frame a caller assembled; a missing column must not lose the figure
    assert plots._lead_label("12:1:A:G", {}, 12.6814) == "12:1:A:G\np=2.08e-13"


def test_the_lead_label_names_the_gene_and_the_consequence_when_annotated():
    labelled = plots._lead_label(
        "12:49272869:C:T", {"beta": 1.17569, "af": 0.00234211}, 12.6814,
        "missense_variant", "TUBA1C",
    )
    assert labelled.splitlines()[0] == "12:49272869:C:T  TUBA1C missense_variant"
    # an intergenic lead has a consequence and no gene; losing the consequence too would be
    # dropping the more informative half
    assert plots._lead_label("12:1:A:G", {}, 3.0, "intergenic_variant", None).splitlines()[0] \
        == "12:1:A:G  intergenic_variant"
    assert plots._lead_label("12:1:A:G", {}, 3.0, None, None).splitlines()[0] == "12:1:A:G"


def test_coding_consequences_square_and_everything_else_circles(monkeypatch, tmp_path):
    from genetics_mcp_server import sdk

    ids = [f"12:{49_500_000 + i * 1_000}:C:T" for i in range(6)]

    def fake_annotation(**kwargs):
        return pl.DataFrame(
            {
                "variant": ids,
                # two coding, then terms that must NOT square: splice_region is up to 8bp
                # into an intron, and UTR/intron/intergenic are not coding at all
                "most_severe": [
                    "missense_variant", "synonymous_variant", "splice_region_variant",
                    "3_prime_UTR_variant", "intron_variant", "intergenic_variant",
                ],
            }
        )

    monkeypatch.setattr(sdk, "variant_annotation", fake_annotation)
    monkeypatch.setenv("SANDBOX_ARTIFACTS_DIR", str(tmp_path))
    found = plots._consequences("12:1-2")
    coding = [found[vid][0] in plots._CODING_CONSEQUENCES for vid in ids]
    assert coding == [True, True, False, False, False, False]


def test_an_unavailable_consequence_lookup_leaves_every_point_a_circle(monkeypatch):
    from genetics_mcp_server import sdk

    def boom(**kwargs):
        raise RuntimeError("db-api is down")

    monkeypatch.setattr(sdk, "variant_annotation", boom)
    assert plots._consequences("12:1-2") is None, (
        "a failed lookup must be distinguishable from 'nothing here is coding'"
    )


def test_coding_false_skips_the_fetch_entirely(monkeypatch, tmp_path):
    from genetics_mcp_server import sdk

    def refuse(**kwargs):  # pragma: no cover - only runs on a regression
        raise AssertionError("variant_annotation was fetched despite coding=False")

    monkeypatch.setattr(sdk, "variant_annotation", refuse)
    monkeypatch.setenv("SANDBOX_ARTIFACTS_DIR", str(tmp_path))
    result = plots.locuszoom(
        phenotype="X", region="12:49400000-49600000", data=frame_of(), ld=False,
        genes=False, coding=False,
    )
    assert result["coding_marked"] is False
    assert result["lead_consequence"] is None and result["lead_gene"] is None


def test_the_lead_is_not_a_diamond_and_the_significance_line_is_grey(monkeypatch, tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from genetics_mcp_server import sdk

    monkeypatch.setattr(sdk, "variant_annotation", lambda **k: pl.DataFrame())
    _figure, ax = plt.subplots()
    plots.locuszoom(
        phenotype="X", region="12:49400000-49600000", data=frame_of(), ld=False,
        genes=False, ax=ax,
    )
    # reference paths built by scatter itself: a MarkerStyle path is transformed on the way
    # into a collection, so comparing against MarkerStyle("D").get_path() never matches and
    # the assertion passes whatever is drawn
    _ref_fig, ref_ax = plt.subplots()
    reference = {}
    for name in ("o", "s", "D"):
        ref_ax.scatter([0], [0], marker=name)
        reference[name] = ref_ax.collections[-1].get_paths()[0].vertices
    plt.close(_ref_fig)

    def shape_of(vertices):
        for name, ref in reference.items():
            if vertices.shape == ref.shape and (vertices == ref).all():
                return name
        return "?"

    drawn = {shape_of(c.get_paths()[0].vertices) for c in ax.collections}
    assert "D" not in drawn, "the lead is still drawn as a diamond"
    assert drawn <= {"o", "s"}, f"unexpected marker shapes: {drawn}"

    line = next(ln for ln in ax.lines if ln.get_linestyle() == "--")
    assert line.get_color() == plots._SIGNIFICANCE_GREY
    label = next(t for t in ax.texts if t.get_text().startswith("p="))
    assert label.get_text() == "p=5e-8", "the padded exponent reached the figure"
    assert label.get_color() == plots._SIGNIFICANCE_GREY
    plt.close("all")


def test_the_lead_annotation_sits_below_the_point(monkeypatch, tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from genetics_mcp_server import sdk

    monkeypatch.setattr(sdk, "variant_annotation", lambda **k: pl.DataFrame())
    _figure, ax = plt.subplots()
    plots.locuszoom(
        phenotype="X", region="12:49400000-49600000", data=frame_of(), ld=False,
        genes=False, ax=ax,
    )
    note = next(t for t in ax.texts if "p=" in t.get_text() and "\n" in t.get_text())
    assert note.get_verticalalignment() == "top"
    assert note.xyann[1] < 0, "the label is offset upwards and will run into the axes frame"
    plt.close("all")


def test_the_leads_gene_and_consequence_are_drawn_and_returned(monkeypatch, tmp_path):
    from genetics_mcp_server import sdk

    lead = "12:49503000:C:T"
    monkeypatch.setattr(
        sdk, "variant_annotation",
        lambda **k: pl.DataFrame(
            {"variant": [lead], "most_severe": ["missense_variant"],
             "gene_most_severe": ["TUBA1C"]}
        ),
    )
    monkeypatch.setenv("SANDBOX_ARTIFACTS_DIR", str(tmp_path))
    frame = sumstats([{
        "chr": "12", "pos": 49_503_000, "ref": "C", "alt": "T",
        "pval": 2.08257e-13, "mlog10p": 12.6814,
    }])
    result = plots.locuszoom(
        phenotype="X", region="12:49400000-49600000", data=frame, ld=False, genes=False,
    )
    assert result["lead_consequence"] == "missense_variant"
    assert result["lead_gene"] == "TUBA1C"


# ----------------------------------------------------------------- the gene models


def gene_axis():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt.subplots()


PCSK9 = {
    "gene_name": ["PCSK9"],
    "gene_start": [55_039_445],
    "gene_end": [55_064_852],
    "gene_strand": ["+"],
    # first exon entirely 5' UTR, second exon translated in full, third partly
    "exon_starts": [[55_039_445, 55_043_843, 55_052_000]],
    "exon_ends": [[55_039_763, 55_044_063, 55_052_400]],
    "cds_starts": [[None, 55_043_843, 55_052_000]],
    "cds_ends": [[None, 55_044_063, 55_052_200]],
}


def test_exon_spans_pairs_each_exon_with_its_own_coding_bounds():
    spans = plots._exon_spans({k: v[0] for k, v in PCSK9.items()})
    assert spans == [
        (55_039_445, 55_039_763, None, None),
        (55_043_843, 55_044_063, 55_043_843, 55_044_063),
        (55_052_000, 55_052_400, 55_052_000, 55_052_200),
    ]


@pytest.mark.parametrize(
    "gene",
    [
        {},                                                    # a results-api with no exons
        {"exon_starts": [], "exon_ends": []},                  # a release with no exon file
        {"exon_starts": [1, 2], "exon_ends": [3]},             # lists that cannot be paired
    ],
)
def test_a_gene_without_usable_exon_structure_draws_its_body_alone(gene):
    """Three different upstreams arrive as the same thing here, and none may raise: the
    track degrades to gene bodies, which is what it drew before exons existed."""
    assert plots._exon_spans(gene) == []


def test_a_coding_exon_is_drawn_thicker_than_its_utr_and_the_body_thinnest():
    """Thickness IS the encoding — a reader tells UTR from coding sequence by nothing else."""
    figure, ax = gene_axis()
    drawn, exons = plots._draw_genes(ax, pl.DataFrame(PCSK9), 55_030_000, 55_070_000)

    assert (drawn, exons) == (1, 3)
    widths = sorted({line.get_linewidth() for line in ax.lines})
    assert widths == [
        plots._GENE_BODY_WIDTH,
        plots._GENE_EXON_WIDTH,
        plots._GENE_CDS_WIDTH,
    ], "the three tiers of a gene model are not all present and distinct"

    utr_only = [
        line for line in ax.lines
        if line.get_linewidth() == plots._GENE_CDS_WIDTH
        and list(line.get_xdata())[0] == 55_039_445
    ]
    assert not utr_only, "an untranslated exon was given a coding bar"


def test_an_exon_outside_the_window_is_clipped_away_rather_than_drawn():
    """The API returns a gene whole, exons included, so a gene overhanging the window brings
    exons that belong to no visible part of it."""
    figure, ax = gene_axis()
    plots._draw_genes(ax, pl.DataFrame(PCSK9), 55_030_000, 55_040_000)

    for line in ax.lines:
        xs = list(line.get_xdata())
        assert max(xs) <= 55_040_000 and min(xs) >= 55_030_000


def test_locuszoom_reports_how_many_exons_it_drew(tmp_path, monkeypatch):
    """`n_exons` is what tells a caller whether the track has structure at all, the way
    `ld_joined` tells them whether the colours mean anything."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from genetics_mcp_server import sdk

    with_exons = pl.DataFrame(
        {**PCSK9, "gene_start": [49_490_000], "gene_end": [49_530_000],
         "exon_starts": [[49_500_000, 49_510_000]],
         "exon_ends": [[49_500_400, 49_510_600]],
         "cds_starts": [[None, 49_510_000]],
         "cds_ends": [[None, 49_510_600]]}
    )
    monkeypatch.setattr(sdk, "gene_annotations", lambda **_kw: with_exons, raising=False)
    result = plots.locuszoom(
        phenotype="X", region="12:49400000-49600000", data=frame_of(), ld=False,
        genes=True, path=str(tmp_path / "with.png"),
    )
    assert (result["n_genes"], result["n_exons"]) == (1, 2)

    bodies_only = with_exons.drop(["exon_starts", "exon_ends", "cds_starts", "cds_ends"])
    monkeypatch.setattr(sdk, "gene_annotations", lambda **_kw: bodies_only, raising=False)
    result = plots.locuszoom(
        phenotype="X", region="12:49400000-49600000", data=frame_of(), ld=False,
        genes=True, path=str(tmp_path / "without.png"),
    )
    assert (result["n_genes"], result["n_exons"]) == (1, 0)
    plt.close("all")
