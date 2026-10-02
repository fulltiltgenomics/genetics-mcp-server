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
from genetics_mcp_server.sdk.errors import GeneticsError, GeneticsUsageError


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


@pytest.fixture(autouse=True)
def _offline_lookups(monkeypatch):
    """Answer the two fetches locuszoom makes on its own, without a service.

    `ld=False, genes=False` turns off the fetches a caller asks for, but the title still
    resolves the phenotype name and the coding annotation is a second fetch, so every render
    went out over the network anyway — succeeding against a dev API on :2000 where one is up
    and timing out where it is not, at ~3s a render either way. Both answers here mean
    "nothing to add", which is the shape the drawing tests already assume; a test that cares
    what either lookup returns sets its own over this one.
    """
    from genetics_mcp_server import sdk

    monkeypatch.setattr(sdk, "lookup_phenotype_names", lambda *a, **k: pl.DataFrame())
    monkeypatch.setattr(sdk, "variant_annotation", lambda *a, **k: pl.DataFrame())


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
            "variant": ["12:100:A:G", "12:900:A:G", "12:950:A:G", "not-a-variant",
                        "12:500:A:G", "12:80:A:G"],
            "r2": [0.7, 0.9, None, 0.99, 0.95, 0.3],
        }
    )
    found = plots._partners_outside(ld, 400, 600)
    assert [p["r2"] for p in found] == [0.9, 0.7]
    assert plots._partners_outside(None, 400, 600) == []


def test_a_weakly_correlated_partner_outside_the_window_is_not_reported():
    """The note asks the reader to redraw wider, so it fires only where that is worth doing.
    A partner just over the floor is reported and one just under it is not, which is what
    makes this a threshold rather than a preference."""
    just_over = plots._LD_NOTABLE_R2
    just_under = plots._LD_NOTABLE_R2 - 0.01
    ld = pl.DataFrame(
        {"variant": ["12:100:A:G", "12:900:A:G"], "r2": [just_under, just_over]}
    )
    assert [p["r2"] for p in plots._partners_outside(ld, 400, 600)] == [just_over]


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
    assert full == "12:1:A:G\np 2.08e-13  beta 1.18  AF 0.002342"
    # `data=` may be any frame a caller assembled; a missing column must not lose the figure
    assert plots._lead_label("12:1:A:G", {}, 12.6814) == "12:1:A:G\np 2.08e-13"
    assert "=" not in full, "the caption reads as a caption, not as an assignment"


def test_the_lead_label_names_the_gene_and_the_consequence_when_annotated():
    labelled = plots._lead_label(
        "12:49272869:C:T", {"beta": 1.17569, "af": 0.00234211}, 12.6814,
        "missense_variant", "TUBA1C",
    )
    assert labelled.splitlines()[0] == "12:49272869:C:T  TUBA1C missense"
    # an intergenic lead has a consequence and no gene; losing the consequence too would be
    # dropping the more informative half
    assert plots._lead_label("12:1:A:G", {}, 3.0, "intergenic_variant", None).splitlines()[0] \
        == "12:1:A:G  intergenic"
    assert plots._lead_label("12:1:A:G", {}, 3.0, None, None).splitlines()[0] == "12:1:A:G"


def test_coding_consequences_square_and_everything_else_circles(monkeypatch, tmp_path):
    from genetics_mcp_server import sdk

    ids = [f"12:{49_500_000 + i * 1_000}:C:T" for i in range(6)]

    def fake_annotation(**kwargs):
        return pl.DataFrame(
            {
                "variant": ids,
                # one coding, then terms that must NOT square. synonymous sits in a coding
                # sequence and leaves the protein identical; splice_region is up to 8bp into
                # an intron; UTR/intron/intergenic are not coding at all
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
    assert coding == [True, False, False, False, False, False]


def test_the_coding_set_is_the_one_the_rest_of_the_suite_uses():
    """Five copies of this list exist — here, results-api's `coding_set`, the chat prompt's
    Terminology block, and the browser's two coding.ts files. They drifted four ways before
    being reconciled, so this pins the membership rather than only the behaviour.

    The terms written without a `_variant` suffix are written that way because that is the SO
    name the annotation actually holds: `transcript_ablation` occurs in the data and
    `transcript_ablation_variant` does not, so the suffixed spelling matched nothing.
    """
    assert plots._CODING_CONSEQUENCES == frozenset({
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


@pytest.mark.parametrize(
    "term",
    [
        # in a coding sequence, protein unchanged — the reason all four are excluded
        "synonymous_variant",
        "coding_sequence_variant",
        "start_retained_variant",
        "stop_retained_variant",
        # up to 8bp into an intron, so it establishes no coding position
        "splice_region_variant",
        # present in the data and not coding
        "non_coding_transcript_exon_variant",
        "3_prime_UTR_variant",
        "5_prime_UTR_variant",
        "mature_miRNA_variant",
    ],
)
def test_terms_that_must_not_square(term):
    assert term not in plots._CODING_CONSEQUENCES


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
    label = next(t for t in ax.texts if t.get_text().startswith("p "))
    assert label.get_text() == "p 5e-8", "the padded exponent reached the figure"
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
    note = next(t for t in ax.texts if "\np " in t.get_text())
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


# ------------------------------------------------- what the model spans, and what is drawn


TUBA1C = {
    # the real v49 shape: an 86 kb gene record whose MANE transcript is the rightmost 9.5 kb
    "gene_name": ["TUBA1C"],
    "gene_start": [49_188_736],
    "gene_end": [49_274_600],
    "gene_strand": ["+"],
    "exon_starts": [[49_265_082, 49_269_465, 49_269_828, 49_272_253]],
    "exon_ends": [[49_265_184, 49_269_687, 49_269_976, 49_274_600]],
    "cds_starts": [[49_265_182, 49_269_465, 49_269_828, 49_272_253]],
    "cds_ends": [[49_265_184, 49_269_687, 49_269_976, 49_273_227]],
}


def body_line(ax):
    """The hairline under one gene model — the only line drawn at the body width."""
    lines = [ln for ln in ax.lines if ln.get_linewidth() == plots._GENE_BODY_WIDTH]
    assert len(lines) == 1, f"expected one body line, got {len(lines)}"
    return list(lines[0].get_xdata())


def test_the_body_spans_the_transcript_and_not_the_gene_record():
    """A GENCODE gene record spans every transcript it has, so drawing it puts the exons in
    a corner of a long bare line and the gene reads as being somewhere it is not. TUBA1C's
    record is 86 kb; the transcript actually drawn is 9.5 kb of it."""
    figure, ax = gene_axis()
    plots._draw_genes(ax, pl.DataFrame(TUBA1C), 49_150_000, 49_300_000)

    assert body_line(ax) == [49_265_082, 49_274_600]
    assert body_line(ax)[0] != TUBA1C["gene_start"][0], "the body still spans the gene record"


def test_a_gene_with_no_exons_still_spans_its_record():
    """There is nothing else to draw it from, and it is what the track did before exons."""
    figure, ax = gene_axis()
    bodies_only = pl.DataFrame(TUBA1C).drop(
        ["exon_starts", "exon_ends", "cds_starts", "cds_ends"]
    )
    drawn, exons = plots._draw_genes(ax, bodies_only, 49_150_000, 49_300_000)

    assert (drawn, exons) == (1, 0)
    assert body_line(ax) == [49_188_736, 49_274_600]


@pytest.mark.parametrize(
    "gene,expected",
    [
        ({"gene_name": "ENSG00000258232", "hgnc_symbol": None}, None),
        ({"gene_name": "ENSG00000258232", "hgnc_symbol": ""}, None),
        # a handful of ENSG-named genes do carry an HGNC symbol, so it is consulted
        ({"gene_name": "ENSG00000123456", "hgnc_symbol": "REALGENE"}, "REALGENE"),
        ({"gene_name": "TUBA1C", "hgnc_symbol": "TUBA1C"}, "TUBA1C"),
        ({"gene_name": "TUBA1C", "hgnc_symbol": None}, "TUBA1C"),
    ],
)
def test_a_gene_named_only_by_an_ensg_has_no_label_to_draw(gene, expected):
    assert plots._gene_label(gene) == expected


def test_an_ensg_named_gene_is_left_out_of_the_track_entirely():
    """Not drawn nameless: an ENSG is a row of characters nobody can look up, and the track
    is already packing genes into four rows."""
    figure, ax = gene_axis()
    two = pl.DataFrame(
        {
            "gene_name": ["TUBA1C", "ENSG00000258232"],
            "hgnc_symbol": ["TUBA1C", None],
            "gene_start": [49_265_082, 49_265_156],
            "gene_end": [49_274_600, 49_265_198],
            "gene_strand": ["+", "-"],
            "exon_starts": [[49_265_082], [49_265_156]],
            "exon_ends": [[49_265_184], [49_265_198]],
            "cds_starts": [[None], [None]],
            "cds_ends": [[None], [None]],
        }
    )
    drawn, exons = plots._draw_genes(ax, two, 49_150_000, 49_300_000)

    assert (drawn, exons) == (1, 1), "the ENSG-named gene was drawn"
    assert [t.get_text() for t in ax.texts] == ["TUBA1C→"]


# ------------------------------------------------- the axes the reader actually reads


GENE_IN_WINDOW = {
    "gene_name": ["TUBA1C"],
    "hgnc_symbol": ["TUBA1C"],
    "gene_start": [49_500_000],
    "gene_end": [49_520_000],
    "gene_strand": ["+"],
    "exon_starts": [[49_500_000, 49_510_000]],
    "exon_ends": [[49_500_400, 49_510_600]],
    "cds_starts": [[None, 49_510_000]],
    "cds_ends": [[None, 49_510_600]],
}


def drawn_figure(monkeypatch, tmp_path, annotations=None, **over):
    """A whole locuszoom, gene track included, with the figure kept for inspection.

    `plt.close` frees the manager and leaves the Figure and its Axes intact, so capturing the
    argument is enough — nothing has to stay open. Passing `ax=` instead, as the older tests
    here do, is not an option: that path draws no gene track, which is half of what these
    assert about.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from genetics_mcp_server import sdk

    captured = {}
    real_close = plt.close

    def capture(figure=None):
        if hasattr(figure, "axes"):
            captured["figure"] = figure
        real_close(figure)

    monkeypatch.setattr(plt, "close", capture)
    frame = pl.DataFrame() if annotations is None else pl.DataFrame(annotations)
    monkeypatch.setattr(sdk, "variant_annotation", lambda **k: frame)
    monkeypatch.setattr(sdk, "gene_annotations", lambda **k: pl.DataFrame(GENE_IN_WINDOW))
    monkeypatch.setenv("SANDBOX_ARTIFACTS_DIR", str(tmp_path))

    kwargs = {
        "phenotype": "X", "region": "12:49400000-49600000", "data": frame_of(),
        "ld": False, "genes": True,
    }
    kwargs.update(over)
    result = plots.locuszoom(**kwargs)
    ax, gene_ax = captured["figure"].axes
    return result, ax, gene_ax


def visible_tick_labels(axis) -> bool:
    return any(tick.label1.get_visible() for tick in axis.xaxis.get_major_ticks())


def test_the_position_scale_sits_on_the_association_panel_above_the_gene_track(
    monkeypatch, tmp_path
):
    """sharex puts the tick labels on the bottom axis by default, which left the gene models
    between the points and the scale they are read against."""
    result, ax, gene_ax = drawn_figure(monkeypatch, tmp_path)
    assert result["n_genes"] == 1
    assert visible_tick_labels(ax), "the association panel carries no position scale"
    assert not visible_tick_labels(gene_ax), "the scale is drawn twice, or under the genes"
    assert ax.get_xlabel() == "position on chromosome 12 (Mb)"
    assert gene_ax.get_xlabel() == ""


def test_the_gene_track_is_a_strip_and_not_a_second_boxed_plot(monkeypatch, tmp_path):
    _result, ax, gene_ax = drawn_figure(monkeypatch, tmp_path)
    assert not any(spine.get_visible() for spine in gene_ax.spines.values())
    assert list(gene_ax.get_yticks()) == []
    # the association panel keeps its box; only the track loses one
    assert any(spine.get_visible() for spine in ax.spines.values())


def test_an_empty_gene_frame_still_leaves_the_track_unboxed():
    """The early return is its own path, and a bare box with nothing in it is the worst of
    both: it reads as a panel that failed rather than as a locus with no genes."""
    _figure, ax = gene_axis()
    assert plots._draw_genes(ax, pl.DataFrame(), 1, 100) == (0, 0)
    assert not any(spine.get_visible() for spine in ax.spines.values())


def test_the_legend_names_the_lead_and_the_panel_the_r2_came_from(monkeypatch, tmp_path):
    """A bare "r²" does not say what to or from where, and both change between figures."""
    from genetics_mcp_server import sdk

    monkeypatch.setattr(
        sdk, "ld",
        lambda variant, **k: pl.DataFrame({"variant": ["12:49501000:C:T"], "r2": [0.9]}),
    )
    result, ax, _gene_ax = drawn_figure(monkeypatch, tmp_path, ld=True, ld_panel="sisu99")
    title = ax.get_legend().get_title().get_text()
    assert result["lead"] in title
    assert "sisu99" in title, "the panel is hard-coded rather than taken from the call"
    assert "r^2" in title


def test_the_type_and_the_rules_are_sized_for_the_figure_these_draw(monkeypatch, tmp_path):
    """matplotlib's defaults are sized for a figure twice this wide; at 6.5 in a 12 pt title
    and 0.8 pt spines crowd a panel whose own annotations are 5-7 pt."""
    matplotlib = pytest.importorskip("matplotlib")

    _result, ax, _gene_ax = drawn_figure(monkeypatch, tmp_path)
    assert ax.title.get_fontsize() == plots._TITLE_SIZE
    assert ax.xaxis.label.get_fontsize() == plots._LABEL_SIZE
    assert ax.yaxis.label.get_fontsize() == plots._LABEL_SIZE
    assert ax.get_xticklabels()[0].get_fontsize() == plots._TICK_SIZE
    assert {spine.get_linewidth() for spine in ax.spines.values()} == {plots._AXIS_LINEWIDTH}

    # the point is that they are smaller than what would be inherited, not the numbers
    assert plots._TITLE_SIZE < matplotlib.rcParams["font.size"]
    assert plots._AXIS_LINEWIDTH < matplotlib.rcParams["axes.linewidth"]


@pytest.mark.parametrize(
    "term,expected",
    [
        ("missense_variant", "missense"),
        ("splice_acceptor_variant", "splice acceptor"),
        ("5_prime_UTR_variant", "5 prime UTR"),
        ("stop_gained", "stop gained"),
        ("intergenic_variant", "intergenic"),
        (None, None),
        ("", ""),
    ],
)
def test_a_consequence_reads_as_words_in_a_caption(term, expected):
    assert plots._pretty_consequence(term) == expected


def test_the_caption_shortens_the_term_and_the_returned_dict_keeps_it(monkeypatch, tmp_path):
    """Two audiences: a reader wants `missense`, and a follow-up query filters on the VEP
    term, which `missense` would not match."""
    lead = "12:49503000:C:T"
    frame = sumstats([{
        "chr": "12", "pos": 49_503_000, "ref": "C", "alt": "T",
        "pval": 2.08257e-13, "mlog10p": 12.6814,
    }])
    result, ax, _gene_ax = drawn_figure(
        monkeypatch, tmp_path, data=frame,
        annotations={"variant": [lead], "most_severe": ["missense_variant"],
                     "gene_most_severe": ["TUBA1C"]},
    )
    assert result["lead_consequence"] == "missense_variant"
    caption = next(t.get_text() for t in ax.texts if lead in t.get_text())
    assert "TUBA1C missense" in caption
    assert "missense_variant" not in caption


def test_no_caption_on_the_figure_reads_as_an_assignment(monkeypatch, tmp_path):
    """`p=5e-8`, `beta=1.18`, `r2=0.78` — a figure names a quantity and its value, and the
    `=` is a habit from code that reads as one on a plot."""
    from genetics_mcp_server import sdk

    monkeypatch.setattr(
        sdk, "ld",
        # one partner past the right edge, so the outside-window note is drawn too
        lambda variant, **k: pl.DataFrame(
            {"variant": ["12:49501000:C:T", "12:49900000:G:A"], "r2": [0.9, 0.78]}
        ),
    )
    _result, ax, _gene_ax = drawn_figure(
        monkeypatch, tmp_path, ld=True,
        annotations={"variant": ["12:49503000:C:T"], "most_severe": ["missense_variant"],
                     "gene_most_severe": ["TUBA1C"]},
    )
    drawn = [t.get_text() for t in ax.texts]
    assert any("kb out" in t for t in drawn), "the outside-window note was not drawn"
    assert any(t.startswith("p ") for t in drawn), "the significance line lost its label"
    for text in drawn:
        assert "=" not in text, text


# ------------------------------------------------------------------------- the title


def title_env(monkeypatch, *, name="Sudden idiopathic hearing loss",
              label="FinnGen", raises=None):
    from genetics_mcp_server import sdk

    plots._RESOURCE_LABELS = None  # the cache is process-wide; each case starts cold

    def lookup(codes):
        if raises == "name":
            raise RuntimeError("trait_name_mapping is down")
        return pl.DataFrame({"phenotype": [codes], "name": [name]})

    def schema(*_a, **_k):
        if raises == "schema":
            raise RuntimeError("db-api is down")
        return {"resources": {"finngen": {"label": label}}} if label else {"resources": {}}

    monkeypatch.setattr(sdk, "lookup_phenotype_names", lookup)
    monkeypatch.setattr(sdk, "schema", schema)


def sumstats_with_version(version="R14", resource="finngen"):
    frame = frame_of()
    columns = []
    if resource is not None:
        columns.append(pl.lit(resource).alias("resource"))
    if version is not None:
        columns.append(pl.lit(version).alias("version"))
    return frame.with_columns(columns) if columns else frame


def test_the_title_names_the_trait_the_code_and_the_release(monkeypatch):
    title_env(monkeypatch)
    assert plots._default_title(
        "H8_HL_IDIOP", "12:49022869-49522869", sumstats_with_version(), "finngen"
    ) == "Sudden idiopathic hearing loss (H8_HL_IDIOP, FinnGen R14) — 12:49022869-49522869"


def test_the_release_and_resource_come_off_the_frame_not_the_argument(monkeypatch):
    """A caller who passed `data=` gets the label of the data they actually plotted; the
    `resource` argument was never used to fetch it."""
    title_env(monkeypatch, label=None)
    got = plots._default_title(
        "X", "12:1-2", sumstats_with_version(version="R13", resource="ukbb"), "finngen"
    )
    assert "ukbb R13" in got and "finngen" not in got


@pytest.mark.parametrize(
    "kwargs,frame_kwargs,expected",
    [
        # the code stays in front when the name does not resolve
        ({"name": None}, {}, "H8_HL_IDIOP (FinnGen R14) — 12:1-2"),
        ({"name": "Unknown: H8_HL_IDIOP"}, {}, "H8_HL_IDIOP (FinnGen R14) — 12:1-2"),
        # a frame with no release still names the resource
        ({}, {"version": None},
         "Sudden idiopathic hearing loss (H8_HL_IDIOP, FinnGen) — 12:1-2"),
        # both lookups down: the resource argument is all that is left, unlabelled
        ({"raises": "name", "label": None}, {"version": None, "resource": None},
         "H8_HL_IDIOP (finngen) — 12:1-2"),
    ],
)
def test_the_title_degrades_one_piece_at_a_time(monkeypatch, kwargs, frame_kwargs, expected):
    """Every part is a third-party lookup, and none of them may cost the figure."""
    title_env(monkeypatch, **kwargs)
    frame = sumstats_with_version(**{"version": "R14", "resource": "finngen", **frame_kwargs})
    assert plots._default_title("H8_HL_IDIOP", "12:1-2", frame, "finngen") == expected


def test_nothing_at_all_to_add_leaves_the_title_this_replaced(monkeypatch):
    """The floor of the degradation, reached only when there is no resource to name either."""
    title_env(monkeypatch, raises="name", label=None)
    frame = sumstats_with_version(version=None, resource=None)
    assert plots._default_title("H8_HL_IDIOP", "12:1-2", frame, "") == "H8_HL_IDIOP — 12:1-2"


def test_a_failed_schema_fetch_is_not_cached(monkeypatch):
    """Caching the failure would leave every later figure in the execution unlabelled for a
    blip that lasted one call."""
    title_env(monkeypatch, raises="schema")
    assert plots._resource_label("finngen") == "finngen"
    assert plots._RESOURCE_LABELS is None
    title_env(monkeypatch)
    assert plots._resource_label("finngen") == "FinnGen"


def test_an_explicit_title_still_wins(monkeypatch, tmp_path):
    title_env(monkeypatch)
    _result, ax, _gene_ax = drawn_figure(monkeypatch, tmp_path, title="mine")
    assert ax.get_title() == "mine"


def test_the_drawn_title_is_the_derived_one(monkeypatch, tmp_path):
    title_env(monkeypatch)
    _result, ax, _gene_ax = drawn_figure(
        monkeypatch, tmp_path, phenotype="H8_HL_IDIOP", data=sumstats_with_version()
    )
    assert ax.get_title().startswith("Sudden idiopathic hearing loss (H8_HL_IDIOP, FinnGen R14)")


def test_the_default_window_is_250kb_either_side():
    """The house default, and the one the plots are composed for. It is stated in the
    docstring too, because that is what reaches a script's author."""
    import inspect

    assert inspect.signature(plots.locuszoom).parameters["flank"].default == 250_000
    region, _chrom = plots._region_from_variant("12:49272869:C:T", 250_000)
    assert region == "12:49022869-49522869"


# ------------------------------------------------------- which variant leads, and why


def flat_frame(top_af, *, top_mlog10p=4.2, af_column="af"):
    """A window with nothing significant, whose strongest variant has the given frequency."""
    rows = [
        {"chr": "16", "pos": 6_000_000 + i * 1_000, "ref": "A", "alt": "G",
         "mlog10p": 1.0 + (i % 5) * 0.5, af_column: 0.2}
        for i in range(20)
    ]
    rows[4].update(mlog10p=3.8, **{af_column: 0.046})   # the best common variant
    rows[9].update(mlog10p=top_mlog10p, **{af_column: top_af})
    return pl.DataFrame(rows)


def lead_of(frame, **over):
    kwargs = {"phenotype": "X", "region": "16:6000000-6020000", "data": frame,
              "ld": False, "genes": False}
    kwargs.update(over)
    return plots.locuszoom(**kwargs)


def test_a_rare_variant_below_the_line_does_not_lead_a_flat_window(monkeypatch, tmp_path):
    """Measured: the strongest variant at RBFOX1 for schizophrenia was AF 0.00025 with beta
    2.1 at p 6.5e-5, and the figure captioned it as the lead."""
    monkeypatch.setenv("SANDBOX_ARTIFACTS_DIR", str(tmp_path))
    result = lead_of(flat_frame(0.00025))
    assert result["lead"] == "16:6004000:A:G"
    assert result["strongest"] == "16:6009000:A:G", "the plain maximum is no longer reported"


def test_a_rare_variant_that_reaches_the_line_still_leads(monkeypatch, tmp_path):
    """The rule must not cost the real rare signals: TUBA1C's missense lead is AF 0.002."""
    monkeypatch.setenv("SANDBOX_ARTIFACTS_DIR", str(tmp_path))
    result = lead_of(flat_frame(0.002, top_mlog10p=12.7))
    assert result["lead"] == result["strongest"] == "16:6009000:A:G"


def test_a_frame_with_no_frequency_leads_with_its_strongest_variant(monkeypatch, tmp_path):
    monkeypatch.setenv("SANDBOX_ARTIFACTS_DIR", str(tmp_path))
    result = lead_of(flat_frame(0.00025).drop("af"))
    assert result["lead"] == "16:6009000:A:G"


def test_an_explicit_lead_is_never_overruled(monkeypatch, tmp_path):
    monkeypatch.setenv("SANDBOX_ARTIFACTS_DIR", str(tmp_path))
    result = lead_of(flat_frame(0.00025), lead="16:6009000:A:G")
    assert result["lead"] == "16:6009000:A:G"


def test_a_meta_analysis_frame_is_judged_on_its_per_cohort_frequency(monkeypatch, tmp_path):
    """finngen_ukbb has `fg_af` and `ukbb_af` and no `af`, so a rule that read `af` alone let
    a UKBB-only variant at AF 5e-5 lead with a beta of 40."""
    monkeypatch.setenv("SANDBOX_ARTIFACTS_DIR", str(tmp_path))
    result = lead_of(flat_frame(5.4e-5, af_column="ukbb_af"))
    assert result["lead"] == "16:6004000:A:G"


def test_the_lead_label_says_whose_frequency_it_prints():
    label = plots._lead_label(
        "12:2427715:G:A", {"beta": 40.0, "fg_af": None, "ukbb_af": 5.394e-05}, 3.37
    )
    assert "AF 5.394e-05 (ukbb)" in label
    assert "(ukbb)" not in plots._lead_label("1:1:A:C", {"af": 0.3}, 3.0)


# --------------------------------------------------------- why the points are grey


def ld_answer(rows):
    return pl.DataFrame(rows, schema={"variant": pl.Utf8, "r2": pl.Float64})


def wide_frame():
    """3 Mb of positions with the lead at the left edge."""
    rows = [
        {"chr": "16", "pos": 5_000_000 + i * 100_000, "ref": "A", "alt": "G",
         "pval": 0.01, "mlog10p": 2.0}
        for i in range(31)
    ]
    rows[0]["mlog10p"] = 30.0
    return sumstats(rows)


def test_ld_is_never_asked_for_over_more_than_the_server_accepts(monkeypatch, tmp_path):
    """Above 5 Mb the LD server answers 400, which cost a 2.56 Mb plot every colour it had."""
    from genetics_mcp_server import sdk

    seen = {}

    def fake_ld(variant, *, window, r2_threshold, panel):
        seen["window"] = window
        return ld_answer({"variant": ["16:5100000:A:G"], "r2": [0.9]})

    monkeypatch.setattr(sdk, "ld", fake_ld)
    monkeypatch.setenv("SANDBOX_ARTIFACTS_DIR", str(tmp_path))
    result = plots.locuszoom(
        phenotype="X", region="16:5000000-8000000", data=wide_frame(), genes=False,
    )
    assert seen["window"] == plots.LD_MAX_WINDOW
    # the lead is at the left edge, so 2.5 Mb either side of it stops short of the right one
    assert result["ld_status"] == "partial"
    assert result["ld_joined"] is True


@pytest.mark.parametrize(
    "raised,status,note",
    [
        (GeneticsError("not in panel"), "lead_not_in_panel", "is not in the sisu42 panel"),
        (GeneticsError("the LD server is unavailable"), "unavailable", "did not answer"),
        (RuntimeError("anything else"), "unavailable", "did not answer"),
    ],
)
def test_a_figure_without_ld_says_which_kind_of_without(
    monkeypatch, tmp_path, raised, status, note
):
    """Both used to come back as `ld_joined: False`, and the model told the user to wait for
    an outage when the lead was simply absent from the panel."""
    from genetics_mcp_server import sdk

    if status == "lead_not_in_panel":
        raised.code = "ld_variant_not_in_panel"

    def failing_ld(variant, **kwargs):
        raise raised

    monkeypatch.setattr(sdk, "ld", failing_ld)
    result, ax, _gene_ax = drawn_figure(monkeypatch, tmp_path, ld=True)
    assert result["ld_status"] == status
    assert result["ld_joined"] is False
    assert any(note in t.get_text() for t in ax.texts), "the figure does not say why"
    assert ax.get_legend() is None


def test_the_other_ld_statuses(monkeypatch, tmp_path):
    from genetics_mcp_server import sdk

    monkeypatch.setenv("SANDBOX_ARTIFACTS_DIR", str(tmp_path))
    call = {"phenotype": "X", "region": "12:49400000-49600000", "genes": False}
    assert plots.locuszoom(data=frame_of(), ld=False, **call)["ld_status"] == "off"

    monkeypatch.setattr(sdk, "ld", lambda variant, **k: ld_answer({"variant": [], "r2": []}))
    assert plots.locuszoom(data=frame_of(), **call)["ld_status"] == "no_partners"

    monkeypatch.setattr(
        sdk, "ld", lambda variant, **k: ld_answer({"variant": ["12:49501000:C:T"], "r2": [0.9]})
    )
    assert plots.locuszoom(data=frame_of(), **call)["ld_status"] == "joined"


def test_the_sdk_error_carries_the_tool_layers_code():
    from genetics_mcp_server.sdk.client import GeneticsClient

    with pytest.raises(GeneticsError) as caught:
        GeneticsClient._payload(
            {"success": False, "error": "nope", "error_code": "ld_variant_not_in_panel"}
        )
    assert caught.value.code == "ld_variant_not_in_panel"
    with pytest.raises(GeneticsError) as caught:
        GeneticsClient._payload({"success": False, "error": "nope"})
    assert caught.value.code is None


# ------------------------------------------------------- what the reader has to decode


def test_positions_are_printed_in_mb_and_not_against_an_offset(monkeypatch, tmp_path):
    """The default formatter drew `1.5915 ... 1.5945` under a `1e8` set in 10 pt type."""
    _result, ax, _gene_ax = drawn_figure(monkeypatch, tmp_path)
    ax.figure.canvas.draw()
    labels = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
    assert labels, "no tick labels were drawn"
    assert all(49.4 <= float(label) <= 49.7 for label in labels), labels
    assert len(set(labels)) == len(labels), f"too few decimals to tell ticks apart: {labels}"
    assert ax.xaxis.get_offset_text().get_text() == ""


def test_a_gene_whose_transcript_misses_the_window_is_not_drawn():
    """results-api returns a gene whose RECORD overlaps the region; AIM2's canonical
    transcript ends 75 kb before the FCER1A window starts, and its label floated outside the
    axes with nothing under it."""
    _figure, ax = gene_axis()
    genes = pl.DataFrame({
        "gene_name": ["AIM2", "FCER1A"],
        "gene_start": [159_056_129, 159_289_714],
        "gene_end": [159_188_222, 159_308_224],
        "gene_strand": ["-", "+"],
        "exon_starts": [[159_062_484], [159_302_336]],
        "exon_ends": [[159_076_766], [159_308_202]],
        "cds_starts": [[None], [None]],
        "cds_ends": [[None], [None]],
    })
    drawn, _exons = plots._draw_genes(ax, genes, 159_152_270, 159_452_270)
    assert drawn == 1
    assert [t.get_text() for t in ax.texts] == ["FCER1A→"]


@pytest.mark.parametrize(
    "symbol,expected",
    [
        ("MINOS1", "MICOS10"),      # renamed since the annotation was built
        ("MICOS10", "MICOS10"),
        ("NBL1", "NBL1"),
        ("C1orf151", "C1orf151"),   # a previous symbol of two genes here: left alone
        ("NOT_HERE", "NOT_HERE"),
        (None, None),
    ],
)
def test_the_leads_gene_is_named_the_way_the_track_names_it(symbol, expected):
    genes = pl.DataFrame({
        "gene_name": ["MICOS10", "NBL1", "OTHER"],
        "hgnc_prev_symbol": ["C1orf151|MINOS1", None, "C1orf151"],
    })
    assert plots._track_symbol(symbol, genes) == expected
    assert plots._track_symbol(symbol, None) == symbol


def test_a_title_wider_than_the_figure_is_broken_and_not_clipped(monkeypatch, tmp_path):
    long_title = (
        "Potassium [Moles/volume] in Serum or Plasma [mmol/l], sample-wise median "
        "(median_3023103_age_adjusted_IRN, FinnGen R14) — 12:49022869-49522869"
    )
    _result, ax, _gene_ax = drawn_figure(monkeypatch, tmp_path, title=long_title)
    figure = ax.figure
    renderer = figure.canvas.get_renderer()
    figure.draw_without_rendering()
    assert "\n" in ax.title.get_text()
    assert ax.title.get_text().replace("\n", " — ") == long_title
    assert ax.title.get_window_extent(renderer).width <= figure.bbox.width


def label_box(ax, text_fragment):
    from matplotlib.text import Text

    figure = ax.figure
    figure.draw_without_rendering()
    renderer = figure.canvas.get_renderer()
    note = next(t for t in ax.texts if text_fragment in t.get_text())
    return Text.get_window_extent(note, renderer), renderer


def test_the_lead_label_stays_inside_the_axes_and_off_the_points(monkeypatch, tmp_path):
    """Fixed below the point, the caption of a lead at the window's left edge ran across the
    y axis, and at a dense locus it sat on the lead's own LD partners."""
    rows = frame_of(n=60).to_dicts()
    for row in rows:
        row["mlog10p"], row["pval"] = 2.0, 0.01
    rows[0].update(mlog10p=30.0, pval=1e-30)
    # a column of points directly under the lead, where the label used to go
    for i in range(1, 12):
        rows[i].update(pos=rows[0]["pos"] + 10, mlog10p=30.0 - i * 0.4)
    _result, ax, _gene_ax = drawn_figure(monkeypatch, tmp_path, data=sumstats(rows))
    box, _renderer = label_box(ax, "\np ")
    frame = ax.bbox
    assert box.x0 >= frame.x0 and box.x1 <= frame.x1, "the label leaves the axes"
    points = ax.transData.transform([(r["pos"], r["mlog10p"]) for r in rows])
    covered = [
        (x, y) for x, y in points if box.x0 <= x <= box.x1 and box.y0 <= y <= box.y1
    ]
    assert not covered, "the label is drawn over data"


def test_the_significance_label_is_on_the_right_where_the_legend_is_not(
    monkeypatch, tmp_path
):
    _result, ax, _gene_ax = drawn_figure(monkeypatch, tmp_path)
    label = next(t for t in ax.texts if t.get_text() == "p 5e-8")
    assert label.get_horizontalalignment() == "right"
    assert label.get_position()[0] > 0.9


# ------------------------------------------------------------ naming more than the lead


def ld_frame_for(frame, r2s):
    ids = [f"12:{pos}:C:T" for pos in frame["pos"]]
    return ld_answer({"variant": [ids[i] for i in r2s], "r2": list(r2s.values())})


def test_the_centred_variant_is_marked_when_it_is_not_the_lead(monkeypatch, tmp_path):
    """Asked for a plot of rs2251746, the figure named only the lead 20 kb away."""
    result, ax, _gene_ax = drawn_figure(
        monkeypatch, tmp_path, region=None, variant="12:49510000:C:T", flank=100_000,
    )
    assert result["lead"] == "12:49503000:C:T"
    assert [h["variant"] for h in result["highlighted"]] == ["12:49510000:C:T"]
    assert any(t.get_text().startswith("12:49510000:C:T") for t in ax.texts)


def test_the_centred_variant_that_is_the_lead_is_not_named_twice(monkeypatch, tmp_path):
    result, _ax, _gene_ax = drawn_figure(
        monkeypatch, tmp_path, region=None, variant="12:49503000:C:T", flank=100_000,
    )
    assert result["highlighted"] == []


def test_highlight_names_what_it_is_given_and_refuses_what_is_not_there(monkeypatch, tmp_path):
    result, ax, _gene_ax = drawn_figure(
        monkeypatch, tmp_path, highlight=["chr12:49520000:c:t", "12:49503000:C:T"],
    )
    # the lead is already named; asking for it again does not label it twice
    assert [h["variant"] for h in result["highlighted"]] == ["12:49520000:C:T"]
    with pytest.raises(GeneticsUsageError, match="not among the variants"):
        drawn_figure(monkeypatch, tmp_path, highlight="12:1:A:C")


def test_label_r2_names_the_leads_neighbours_strongest_first_up_to_the_cap(
    monkeypatch, tmp_path
):
    from genetics_mcp_server import sdk

    frame = frame_of(n=25)
    # twelve partners above the threshold, one below it
    r2s = {i: 0.9 for i in range(5, 17)} | {20: 0.5}
    monkeypatch.setattr(sdk, "ld", lambda variant, **k: ld_frame_for(frame, r2s))
    result, ax, _gene_ax = drawn_figure(
        monkeypatch, tmp_path, data=frame, ld=True, label_r2=0.8,
    )
    named = result["highlighted"]
    assert len(named) == plots._MAX_R2_LABELS
    assert all(h["r2"] == pytest.approx(0.9) for h in named)
    assert [h["mlog10p"] for h in named] == sorted((h["mlog10p"] for h in named), reverse=True)
    assert sum("r$^2$ 0.90" in t.get_text() for t in ax.texts) == plots._MAX_R2_LABELS


def test_labels_use_the_rsid_where_the_frame_has_one(monkeypatch, tmp_path):
    frame = frame_of().with_columns(
        pl.when(pl.col("pos") == 49_520_000).then(pl.lit("rs356219")).otherwise(None)
        .alias("rsid")
    )
    _result, ax, _gene_ax = drawn_figure(
        monkeypatch, tmp_path, data=frame, highlight="12:49520000:C:T",
    )
    assert any(t.get_text() == "rs356219" for t in ax.texts)


@pytest.mark.parametrize("threshold", [0, -0.1, 1.5])
def test_label_r2_outside_zero_to_one_is_refused(threshold):
    with pytest.raises(GeneticsUsageError, match="label_r2"):
        plots.locuszoom(
            phenotype="X", region="12:49400000-49600000", data=frame_of(), ld=False,
            genes=False, label_r2=threshold,
        )


# ---------------------------------------------------------------------------------- phewas


def associations(rows):
    """Credible-set rows as `credible_sets(variant=...)` returns them, GWAS and QTL alike."""
    base = {
        "resource": "finngen", "dataset": "FinnGen_R14", "data_type": "GWAS", "trait": None,
        "trait_original": None, "mlog10p": None,
        "most_severe": "missense_variant", "gene_most_severe": "APOE",
    }
    out = []
    for row in rows:
        merged = {**base, **row}
        # results-api fills `trait` with a display form of the code; the code itself is
        # `trait_original`, and a test that passes only one gets the other derived
        merged.setdefault("trait_original", merged["trait"])
        merged["trait"] = merged["trait"] or merged["trait_original"]
        out.append(merged)
    return pl.DataFrame(out)


# what phenotypes_v holds for these codes: the ICD chapter, not an organ system
PHENOTYPES_V = {
    "I9_CHD": ("Coronary heart disease", "IX Diseases of the circulatory system (I9_)"),
    "T2D": ("Type 2 diabetes", "IV Endocrine, nutritional and metabolic diseases (E4_)"),
    "H8_HL_IDIOP": ("Sudden idiopathic hearing loss",
                    "VIII Diseases of the ear and mastoid process (H8_)"),
    "AD": ("Alzheimer's disease", "VI Diseases of the nervous system (G6_)"),
}


def phewas_frame():
    return associations([
        {"trait_original": "I9_CHD", "mlog10p": 12.0},
        {"trait_original": "T2D", "mlog10p": 9.5},
        {"trait_original": "AD", "mlog10p": 300.0},
        {"trait_original": "H8_HL_IDIOP", "mlog10p": 8.0},
        {"trait_original": "XYZ", "trait": "Some_odd_thing", "mlog10p": 3.0},
        {"trait_original": "WEAK", "mlog10p": 1.0},
        {"trait_original": "ENSG00000130203|ge", "trait": "APOE", "mlog10p": 40.0,
         "data_type": "eQTL"},
    ])


def fake_phenotypes(**kwargs):
    codes = kwargs.get("codes") or []
    rows = [
        {"dataset": "FinnGen_R14", "resource": "finngen", "trait_original": code,
         "trait_name": PHENOTYPES_V[code][0], "category": PHENOTYPES_V[code][1]}
        for code in codes if code in PHENOTYPES_V
    ]
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def named_phewas(monkeypatch, tmp_path, phenotypes=fake_phenotypes, **over):
    """A whole phewas drawn from `data=`, with the figure kept for inspection."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from genetics_mcp_server import sdk

    monkeypatch.setattr(sdk, "phenotypes", phenotypes, raising=False)
    # the display name comes from the live schema; the title under test is the shape, not it
    monkeypatch.setattr(plots, "_resource_label", lambda resource: resource)
    monkeypatch.setenv("SANDBOX_ARTIFACTS_DIR", str(tmp_path))
    captured = {}
    real_close = plt.close

    def capture(figure=None):
        if hasattr(figure, "axes"):
            captured["figure"] = figure
        real_close(figure)

    monkeypatch.setattr(plt, "close", capture)
    kwargs = {"variant": "19:44908684:T:C", "data": phewas_frame()}
    kwargs.update(over)
    result = plots.phewas(**kwargs)
    return result, captured["figure"].axes[0]


def test_phewas_writes_a_figure_and_reports_what_it_drew(monkeypatch, tmp_path):
    result, _ax = named_phewas(monkeypatch, tmp_path)
    assert result["path"] == str(tmp_path / "phewas.png")
    assert (tmp_path / "phewas.png").stat().st_size > 0
    assert result["variant"] == "19:44908684:T:C"
    # the QTL row and the one below min_mlog10p are not associations on a phewas
    assert result["n_associations"] == 5
    assert result["n_significant"] == 4
    assert result["strongest"] == "AD"
    assert result["strongest_name"] == "Alzheimer's disease"
    assert result["strongest_mlog10p"] == 300.0
    assert result["variant_gene"] == "APOE"
    assert result["variant_consequence"] == "missense_variant"


def test_phewas_groups_by_the_source_category_in_chapter_order_with_other_last(
    monkeypatch, tmp_path
):
    result, ax = named_phewas(monkeypatch, tmp_path)
    assert result["groups"] == [
        "IV Endocrine, nutritional and metabolic diseases (E4_)",
        "VI Diseases of the nervous system (G6_)",
        "VIII Diseases of the ear and mastoid process (H8_)",
        "IX Diseases of the circulatory system (I9_)",
        "Other",
    ]
    assert len(ax.get_xticklabels()) == len(result["groups"])
    assert not ax.get_legend(), "a phewas carries no legend box"


def test_a_long_chapter_name_is_wrapped_on_the_axis_not_truncated_to_nothing():
    label = plots._tick_label("VIII Diseases of the ear and mastoid process (H8_)")
    assert "\n" in label and label.count("\n") <= plots._PHEWAS_TICK_LINES - 1
    assert plots._tick_label("Other") == "Other"


@pytest.mark.parametrize(
    "categories,expected",
    [
        (["V Mental", "IX Circ", "II Neoplasms", "I Infectious"],
         ["I Infectious", "II Neoplasms", "V Mental", "IX Circ"]),
        (["Other", "Quantitative", "Binary", "XI Digestive"],
         ["XI Digestive", "Binary", "Quantitative", "Other"]),
    ],
)
def test_category_order_is_chapter_order_then_alphabetical_then_other(categories, expected):
    assert sorted(categories, key=plots._category_order) == expected


def test_phewas_draws_the_significance_line_the_way_the_locuszoom_does(monkeypatch, tmp_path):
    _result, ax = named_phewas(monkeypatch, tmp_path)
    line = next(ln for ln in ax.lines if ln.get_linestyle() == "--")
    assert line.get_color() == plots._SIGNIFICANCE_GREY
    assert line.get_ydata()[0] == pytest.approx(-math.log10(5e-8))
    label = next(t for t in ax.texts if t.get_text().startswith("p "))
    assert label.get_text() == "p 5e-8"
    assert label.get_color() == plots._SIGNIFICANCE_GREY


def test_phewas_leaves_at_least_two_units_above_the_strongest_association(monkeypatch, tmp_path):
    _result, ax = named_phewas(monkeypatch, tmp_path)
    assert ax.get_ylim()[1] >= 300.0 + 2.0
    # and the same when the strongest point sits below the line, so the line's own label fits
    _result, ax = named_phewas(
        monkeypatch, tmp_path, data=associations([{"trait_original": "T2D", "mlog10p": 3.0}])
    )
    assert ax.get_ylim()[1] >= -math.log10(5e-8) + 2.0
    assert ax.get_ylim()[0] == 0.0


def test_phewas_names_the_significant_associations_and_the_variant(monkeypatch, tmp_path):
    _result, ax = named_phewas(monkeypatch, tmp_path)
    drawn = {t.get_text() for t in ax.texts}
    assert {"Alzheimer's disease", "Coronary heart disease", "Type 2 diabetes",
            "Sudden idiopathic hearing loss"} <= drawn
    assert not any("odd" in t for t in drawn), "an association below the line is not named"
    assert ax.get_title() == "19:44908684:T:C  APOE missense — finngen"


def test_phewas_looks_names_up_by_the_code_and_not_the_display_form(monkeypatch, tmp_path):
    seen = {}

    def spy(**kwargs):
        seen.update(kwargs)
        return fake_phenotypes(**kwargs)

    display_form = associations([
        {"trait_original": "HEIGHT_IRN", "trait": "Height,_inverse-rank_normalized",
         "mlog10p": 30.0},
    ])
    monkeypatch.setattr(plots, "_resource_label", lambda resource: resource)
    result, ax = named_phewas(monkeypatch, tmp_path, data=display_form)
    monkeypatch.setattr("genetics_mcp_server.sdk.phenotypes", spy, raising=False)
    plots.phewas(variant="19:44908684:T:C", data=display_form, path=str(tmp_path / "x.png"))
    assert seen["codes"] == ["HEIGHT_IRN"], "the lookup was keyed on the display form"
    # no metadata row: the label is the display form read as words, the code stays verbatim
    assert result["strongest"] == "HEIGHT_IRN"
    assert result["strongest_name"] == "Height, inverse-rank normalized"
    assert result["groups"] == ["Other"]
    drawn = [t.get_text() for t in ax.texts if t.get_text().startswith("Height")]
    assert drawn and "_" not in drawn[0]


def test_phewas_with_nothing_above_the_floor_says_so():
    with pytest.raises(GeneticsUsageError, match="nothing to plot"):
        plots.phewas(
            variant="19:44908684:T:C",
            data=associations([{"trait_original": "T2D", "mlog10p": 1.0}]),
        )


def test_phewas_survives_a_metadata_lookup_that_fails(monkeypatch, tmp_path):
    def boom(**_kw):
        raise RuntimeError("db-api is down")

    monkeypatch.setattr("genetics_mcp_server.sdk.phenotypes", boom, raising=False)
    monkeypatch.setattr(plots, "_resource_label", lambda resource: resource)
    monkeypatch.setenv("SANDBOX_ARTIFACTS_DIR", str(tmp_path))
    result = plots.phewas(variant="19:44908684:T:C", data=phewas_frame())
    assert result["groups"] == ["Other"]
    assert result["strongest_name"] == "AD"


def test_phewas_into_a_caller_supplied_axis_saves_nothing(monkeypatch, tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from genetics_mcp_server import sdk

    monkeypatch.setattr(sdk, "phenotypes", fake_phenotypes, raising=False)
    monkeypatch.setenv("SANDBOX_ARTIFACTS_DIR", str(tmp_path))
    _figure, ax = plt.subplots()
    result = plots.phewas(variant="19:44908684:T:C", data=phewas_frame(), ax=ax)
    assert result["path"] is None
    assert not list(tmp_path.iterdir())
    plt.close("all")


def test_phewas_names_a_phenotype_once_however_many_resources_carry_it(monkeypatch, tmp_path):
    twice = associations([
        {"trait_original": "AD", "mlog10p": 300.0, "resource": "finngen"},
        {"trait_original": "AD", "mlog10p": 120.0, "resource": "ukbb", "dataset": "UKBB"},
        {"trait_original": "T2D", "mlog10p": 9.5},
    ])
    _result, ax = named_phewas(monkeypatch, tmp_path, data=twice)
    names = [t.get_text() for t in ax.texts if not t.get_text().startswith("p ")]
    assert names.count("Alzheimer's disease") == 1
    assert sorted(names) == ["Alzheimer's disease", "Type 2 diabetes"]


def test_a_pleiotropic_variant_is_grouped_by_resource_rather_than_by_a_wall_of_chapters(
    monkeypatch, tmp_path
):
    chapters = [f"{numeral} Chapter {i}" for i, numeral in enumerate(
        ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI"], 1
    )]
    rows, meta = [], {}
    for i, chapter in enumerate(chapters):
        code = f"CODE{i}"
        resource = "finngen" if i % 2 else "open_targets"
        rows.append({"trait_original": code, "mlog10p": 10.0 + i, "resource": resource,
                     "dataset": "FinnGen_R14" if i % 2 else "Open_Targets_26.06"})
        meta[code] = (f"Trait {i}", chapter)

    def phenotypes(**kwargs):
        return pl.DataFrame([
            {"dataset": "FinnGen_R14" if int(c[4:]) % 2 else "Open_Targets_26.06",
             "trait_original": c, "trait_name": meta[c][0], "category": meta[c][1]}
            for c in kwargs["codes"]
        ])

    result, ax = named_phewas(monkeypatch, tmp_path, phenotypes, data=associations(rows))
    assert result["grouped_by"] == "resource"
    assert result["groups"] == ["finngen", "open_targets"]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["finngen", "open_targets"]
    # one fewer category and the chapters come back
    result, _ax = named_phewas(monkeypatch, tmp_path, phenotypes, data=associations(rows[:-1]))
    assert result["grouped_by"] == "category"
    assert len(result["groups"]) == 10


# ----------------------------------------------------------------------------------- upset


CD_UC = {"UC": 48, "Crohn": 9, ("Crohn", "UC"): 4}


def drawn_upset(monkeypatch, tmp_path, **kwargs):
    """A whole upset drawn to `tmp_path`, with its four panels kept for inspection."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setenv("SANDBOX_ARTIFACTS_DIR", str(tmp_path))
    captured = {}
    real_close = plt.close

    def capture(figure=None):
        if hasattr(figure, "axes"):
            captured["figure"] = figure
        real_close(figure)

    monkeypatch.setattr(plt, "close", capture)
    result = plots.upset(**kwargs)
    bars, matrix, sets, _blank = captured["figure"].axes
    return result, bars, matrix, sets


def test_upset_writes_a_figure_and_reports_what_it_drew(monkeypatch, tmp_path):
    result, *_ = drawn_upset(monkeypatch, tmp_path, counts=CD_UC)
    assert result["path"] == str(tmp_path / "upset.png")
    assert (tmp_path / "upset.png").stat().st_size > 0
    assert result["sets"] == ["UC", "Crohn"]
    assert result["set_sizes"] == {"UC": 52, "Crohn": 13}
    assert result["intersections"] == [
        {"sets": ["UC"], "count": 48},
        {"sets": ["Crohn"], "count": 9},
        {"sets": ["UC", "Crohn"], "count": 4},
    ]
    assert result["n_intersections"] == 3
    assert result["n_elements"] == 61


def test_the_three_input_shapes_agree():
    members = {"A": range(10), "B": range(5, 20), "C": range(18, 30)}
    from_sets = plots._upset_from_sets(members)
    rows = [
        {"A": i in members["A"], "B": i in members["B"], "C": i in members["C"]}
        for i in range(30)
    ]
    from_data = plots._upset_from_data(pl.DataFrame(rows), None, None)
    tallied = (
        pl.DataFrame(rows)
        .group_by(["A", "B", "C"])
        .len()
        .with_columns(pl.col("A").cast(pl.Int64))  # 0/1 columns are memberships too
    )
    from_tally = plots._upset_from_data(tallied, ["A", "B", "C"], "len")
    from_counts = plots._upset_from_counts(
        {"A": 5, ("A", "B"): 5, ("B",): 8, frozenset({"B", "C"}): 2, "C": 10}
    )
    expected = {
        frozenset({"A"}): 5,
        frozenset({"A", "B"}): 5,
        frozenset({"B"}): 8,
        frozenset({"B", "C"}): 2,
        frozenset({"C"}): 10,
    }
    assert from_sets == (["A", "B", "C"], expected)
    assert from_data == (["A", "B", "C"], expected)
    assert from_tally == (["A", "B", "C"], expected)
    assert from_counts == (["A", "B", "C"], expected)


def test_an_element_in_no_set_and_a_null_flag_are_not_intersections():
    frame = pl.DataFrame({"A": [True, False, None], "B": [False, False, None]})
    assert plots._upset_from_data(frame, None, None) == (["A", "B"], {frozenset({"A"}): 1})


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({}, "exactly one of"),
        ({"sets": {"A": [1]}, "counts": {"A": 1}}, "exactly one of"),
        ({"sets": {"A": [1, 2]}}, "at least two sets"),
        ({"counts": {("A", "B"): 0}}, "nothing to plot"),
        ({"counts": {"A": 1, (): 2}}, "names no set"),
        ({"counts": {"A": 1}, "sort_by": "colour"}, "sort_by"),
        ({"data": pl.DataFrame({"n": [1, 2]})}, "no membership columns"),
        ({"data": pl.DataFrame({"A": [True]}), "columns": ["A", "Z"]}, "not in the frame"),
        ({"data": pl.DataFrame({"A": [True], "B": [True]}), "count": "n"}, "not in the frame"),
    ],
)
def test_upset_refuses_what_it_cannot_draw(kwargs, message):
    with pytest.raises(GeneticsUsageError, match=message):
        plots.upset(**kwargs)


def test_nothing_on_an_upset_is_coloured(monkeypatch, tmp_path):
    _result, bars, matrix, sets = drawn_upset(monkeypatch, tmp_path, counts=CD_UC)
    greys = set()
    for panel in (bars, matrix, sets):
        for patch in panel.patches:
            greys.add(patch.get_facecolor()[:3])
        for collection in panel.collections:
            for colour in collection.get_facecolors():
                greys.add(tuple(colour[:3]))
    assert greys, "nothing was drawn"
    assert all(abs(r - g) < 1e-9 and abs(g - b) < 1e-9 for r, g, b in greys), greys


def test_the_counts_sit_outside_the_bars_and_the_scales_leave_room_for_them(
    monkeypatch, tmp_path
):
    _result, bars, matrix, sets = drawn_upset(monkeypatch, tmp_path, counts=CD_UC)
    # above each intersection bar, and the scale runs past the tallest
    labels = {t.get_text(): t.xy for t in bars.texts}
    assert labels == {"48": (0, 48), "9": (1, 9), "4": (2, 4)}
    assert all(t.get_va() == "bottom" for t in bars.texts)
    assert bars.get_ylim()[1] > 48
    # beyond the tip of each set bar on an axis that runs leftwards past the longest
    labels = {t.get_text(): (t.xy, t.get_ha()) for t in sets.texts}
    assert labels == {"52": ((52, 1), "right"), "13": ((13, 0), "right")}
    assert sets.get_xlim()[0] > 52 and sets.get_xlim()[1] == 0
    # the set names are on the set-size panel only: a second copy under the matrix is what
    # the bars overprinted in the scripts this replaced
    assert [t.get_text() for t in sets.get_yticklabels()] == ["UC", "Crohn"]
    assert not matrix.get_yticklabels() and not matrix.get_xticklabels()
    # and the intersections are named by the matrix, not by a caption under it
    assert not matrix.texts


def test_the_matrix_marks_exactly_the_sets_each_column_spans(monkeypatch, tmp_path):
    _result, _bars, matrix, _sets = drawn_upset(monkeypatch, tmp_path, counts=CD_UC)
    ink = matplotlib_colour(plots._UPSET_INK)
    marked = []
    for collection in matrix.collections:
        offsets = collection.get_offsets()
        colours = collection.get_facecolors()
        marked.append({
            int(y) for (_x, y), colour in zip(offsets, colours)
            if tuple(colour[:3]) == ink
        })
    # rows: UC is the top row (y=1), Crohn below it (y=0)
    assert marked == [{1}, {0}, {0, 1}]
    # one connector, on the one column spanning two sets
    assert [tuple(line.get_xdata()) for line in matrix.lines] == [(2, 2)]


def matplotlib_colour(hex_colour):
    from matplotlib.colors import to_rgb

    return to_rgb(hex_colour)


def test_sets_are_rows_largest_first_and_degree_ordering_is_available(monkeypatch, tmp_path):
    counts = {"small": 1, "big": 30, "mid": 10, ("small", "big"): 20, ("small", "big", "mid"): 2}
    result, *_ = drawn_upset(monkeypatch, tmp_path, counts=counts, sort_by="degree")
    assert result["sets"] == ["big", "small", "mid"]
    assert [i["count"] for i in result["intersections"]] == [30, 10, 1, 20, 2]


def test_the_cut_keeps_the_largest_and_the_report_keeps_the_count(monkeypatch, tmp_path):
    counts = {"A": 5, "B": 4, "C": 3, ("A", "B"): 2, ("B", "C"): 1}
    result, bars, *_ = drawn_upset(
        monkeypatch, tmp_path, counts=counts, max_intersections=2, min_count=2
    )
    assert [i["count"] for i in result["intersections"]] == [5, 4]
    assert result["n_intersections"] == 4
    assert result["n_elements"] == 14
    assert len(bars.patches) == 2


def test_upset_into_a_caller_supplied_axis_saves_nothing(monkeypatch, tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setenv("SANDBOX_ARTIFACTS_DIR", str(tmp_path))
    figure, (left, right) = plt.subplots(1, 2)
    result = plots.upset(counts=CD_UC, ax=left)
    assert result["path"] is None
    assert not list(tmp_path.iterdir())
    # the axis was replaced by the panels, and the neighbour is untouched
    assert left not in figure.axes and right in figure.axes
    assert len(figure.axes) == 5
    plt.close("all")


def test_a_long_set_name_is_wrapped_on_the_axis(monkeypatch, tmp_path):
    name = "Set with a rather long descriptive name that keeps going"
    _result, _bars, _matrix, sets = drawn_upset(
        monkeypatch, tmp_path, counts={name: 3, "B": 2, (name, "B"): 1}
    )
    label = sets.get_yticklabels()[0].get_text()
    assert "\n" in label and label.endswith("…")


# ---------------------------------------------------------------------------------- forest


def cohort_frame(**over):
    """Six cohorts' log odds ratios for one variant, the last one too small to say anything."""
    columns = {
        "cohort": ["FinnGen", "UK Biobank", "MVP", "Estonia", "BBJ", "All of Us"],
        "beta": [0.262, 0.301, 0.224, 0.41, 0.19, 0.33],
        "se": [0.021, 0.034, 0.018, 0.09, 0.05, 0.31],
        "mlog10p": [35.2, 18.1, 34.6, 5.3, 3.9, 0.55],
        "n_cases": [48211, 31044, 91230, 6021, 40250, 812],
    }
    columns.update(over)
    return pl.DataFrame(columns)


def drawn_forest(monkeypatch, tmp_path, frame, **kwargs):
    """A whole forest drawn to `tmp_path`: the result, the table canvas and the interval panel."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setenv("SANDBOX_ARTIFACTS_DIR", str(tmp_path))
    captured = {}
    real_close = plt.close

    def capture(figure=None):
        if hasattr(figure, "axes"):
            captured["figure"] = figure
        real_close(figure)

    monkeypatch.setattr(plt, "close", capture)
    kwargs.setdefault("label", "cohort")
    result = plots.forest(frame, **kwargs)
    canvas = captured["figure"].axes[0]
    return result, canvas, canvas.child_axes[0]


def canvas_texts(canvas):
    return [t.get_text() for t in canvas.texts]


def test_forest_writes_a_figure_and_reports_what_it_drew(monkeypatch, tmp_path):
    result, canvas, _panel = drawn_forest(monkeypatch, tmp_path, cohort_frame())
    assert result["path"] == str(tmp_path / "forest.png")
    assert (tmp_path / "forest.png").stat().st_size > 0
    assert result["rows"] == ["FinnGen", "UK Biobank", "MVP", "Estonia", "BBJ", "All of Us"]
    assert result["n_rows"] == result["n_estimates"] == 6
    assert result["n_missing"] == result["n_omitted"] == 0
    assert result["scale"] == "linear" and result["pooled"] == []
    # frame order top to bottom, the numbers beside each row, and a p-value set as a
    # journal sets it rather than as Python prints it
    texts = canvas_texts(canvas)
    assert texts.index("FinnGen") < texts.index("All of Us")
    assert "0.26 (0.22 to 0.30)" in texts
    assert r"$6.3\times10^{-36}$" in texts and "0.28" in texts


def test_log_ratios_are_drawn_as_ratios_on_a_log_axis_around_one(monkeypatch, tmp_path):
    result, canvas, panel = drawn_forest(
        monkeypatch, tmp_path, cohort_frame(), scale="log_ratio"
    )
    assert panel.get_xscale() == "log"
    assert "1.30 (1.25–1.35)" in canvas_texts(canvas)
    assert "OR (95% CI)" in canvas_texts(canvas)
    assert panel.get_xlabel() == "Odds ratio (95% CI)"
    lo, hi = result["xlim"]
    assert 0 < lo < 1 < hi
    # the null line is at 1, not at 0
    assert any(tuple(line.get_xdata()) == (1.0, 1.0) for line in panel.lines)
    # and the axis is ticked where odds ratios are read, not at the decades alone
    assert len(panel.get_xticks()) >= 3


def test_ratios_given_as_ratios_need_bounds_and_must_be_positive(monkeypatch, tmp_path):
    frame = pl.DataFrame({
        "gene": ["PCSK9", "APOC3", "LPA"],
        "or": [0.53, 0.60, 1.9], "lo": [0.41, 0.0, 1.5], "hi": [0.69, 0.85, math.inf],
    })
    result, canvas, _panel = drawn_forest(
        monkeypatch, tmp_path, frame, label="gene", estimate="or", lower="lo", upper="hi",
        scale="ratio",
    )
    texts = canvas_texts(canvas)
    # a bound of zero or infinity is a one-sided interval, drawn to the edge and written so
    assert "0.60 (0–0.85)" in texts and "1.90 (1.50–∞)" in texts
    assert result["n_clipped"] == 2

    with pytest.raises(GeneticsUsageError, match="needs lower= and upper="):
        plots.forest(frame, label="gene", estimate="or", se="lo", scale="ratio")
    # the commonest mistake: a log odds ratio passed as a ratio
    logged = frame.with_columns(pl.Series("or", [-0.63, -0.51, 0.64]))
    with pytest.raises(GeneticsUsageError, match="log_ratio"):
        plots.forest(logged, label="gene", estimate="or", lower="lo", upper="hi", scale="ratio")


def test_an_estimate_that_did_not_converge_does_not_take_the_axis(monkeypatch, tmp_path):
    """A log odds ratio of -14 with a standard error of 300 is what a variant with no carriers
    among the cases returns; left to set the scale it flattens every other row."""
    frame = cohort_frame(
        beta=[0.262, 0.301, 0.224, 0.41, 0.19, -14.2],
        se=[0.021, 0.034, 0.018, 0.09, 0.05, 310.0],
    )
    result, canvas, panel = drawn_forest(monkeypatch, tmp_path, frame, scale="log_ratio")
    lo, hi = result["xlim"]
    assert 0.3 < lo and hi < 4
    assert result["n_clipped"] == 1
    # cut at both edges with an arrowhead each, and no marker for an estimate off the axis
    arrowheads = [line for line in panel.lines if not line.get_clip_on()]
    assert len(arrowheads) == 2
    # the text column still answers, without pretending the digits mean something
    assert "<0.001 (<0.001–>1,000)" in canvas_texts(canvas)


def test_a_null_estimate_keeps_its_row_and_a_missing_error_draws_the_point_alone(
    monkeypatch, tmp_path
):
    frame = cohort_frame(
        beta=[0.262, None, 0.224, float("nan"), 0.19, 0.33],
        se=[0.021, 0.034, None, 0.09, 0.0, 0.31],
    )
    result, canvas, _panel = drawn_forest(monkeypatch, tmp_path, frame)
    assert result["n_rows"] == 6
    assert result["n_missing"] == 2 and result["n_estimates"] == 4
    texts = canvas_texts(canvas)
    assert texts.count("NA") == 2
    # no interval to write for a null or zero standard error
    assert "0.22" in texts and "0.19" in texts


def test_pooling_is_inverse_variance_and_reports_heterogeneity(monkeypatch, tmp_path):
    frame = cohort_frame()
    result, canvas, panel = drawn_forest(
        monkeypatch, tmp_path, frame, scale="log_ratio", pool="fixed"
    )
    weights = [1 / s ** 2 for s in frame["se"]]
    expected = sum(w * b for w, b in zip(weights, frame["beta"])) / sum(weights)
    (fit,) = result["pooled"]
    assert fit["method"] == "fixed" and fit["k"] == 6 and fit["group"] is None
    assert fit["estimate"] == pytest.approx(math.exp(expected))
    assert fit["se"] == pytest.approx(math.sqrt(1 / sum(weights)))
    assert fit["lower"] < fit["estimate"] < fit["upper"]
    assert 0 <= fit["i2"] < 1 and 0 < fit["p_het"] <= 1
    # the pooled row is last and drawn as a diamond rather than a marker; heterogeneity is
    # returned for the text and kept off the figure
    assert result["rows"][-1] == "Fixed effect meta-analysis"
    assert len(panel.patches) == 1
    assert not any("I^2" in text for text in canvas_texts(canvas))


def test_random_effects_widen_the_interval_only_when_the_rows_disagree():
    agree = plots._forest_pool([0.2, 0.2, 0.2], [0.05, 0.04, 0.06], "random")
    assert agree["tau2"] == 0 and agree["i2"] == 0
    assert agree["se"] == pytest.approx(plots._forest_pool([0.2] * 3, [0.05, 0.04, 0.06], "fixed")["se"])
    disagree_fixed = plots._forest_pool([0.0, 0.5, 1.0], [0.05, 0.05, 0.05], "fixed")
    disagree_random = plots._forest_pool([0.0, 0.5, 1.0], [0.05, 0.05, 0.05], "random")
    assert disagree_random["tau2"] > 0 and disagree_random["i2"] > 0.9
    assert disagree_random["se"] > disagree_fixed["se"]


def test_a_row_flagged_as_a_summary_is_a_diamond_and_stays_out_of_the_pool(
    monkeypatch, tmp_path
):
    frame = cohort_frame().with_columns(pl.lit(False).alias("is_meta")).vstack(
        pl.DataFrame({
            "cohort": ["Published meta-analysis"], "beta": [0.25], "se": [0.012],
            "mlog10p": [92.0], "n_cases": [170_000], "is_meta": [True],
        })
    ).sample(fraction=1.0, shuffle=True, seed=1)
    result, _canvas, panel = drawn_forest(
        monkeypatch, tmp_path, frame, summary="is_meta", pool="fixed"
    )
    # wherever the frame had it, the summary row is drawn after the rows it summarises
    assert result["rows"][-2:] == ["Published meta-analysis", "Fixed effect meta-analysis"]
    assert result["pooled"][0]["k"] == 6
    assert len(panel.patches) == 2


def stratified_frame():
    rows = []
    for trait, effects in {
        "Height": {"Females": (0.03, 0.01), "Males": (0.02, 0.012)},
        "BMI": {"Females": (0.05, 0.02), "Males": (-0.01, 0.015)},
        "Testosterone": {"Males": (0.08, 0.03)},
    }.items():
        for sex, (beta, se) in effects.items():
            rows.append({"trait": trait, "sex": sex, "beta": beta, "se": se})
    return pl.DataFrame(rows)


def test_a_series_puts_several_estimates_on_one_row_and_names_them_in_a_legend(
    monkeypatch, tmp_path
):
    result, _canvas, panel = drawn_forest(
        monkeypatch, tmp_path, stratified_frame(), label="trait", series="sex"
    )
    assert result["rows"] == ["Height", "BMI", "Testosterone"]
    assert result["series"] == ["Females", "Males"]
    assert result["n_estimates"] == 5
    assert [t.get_text() for t in panel.get_legend().get_texts()] == ["Females", "Males"]
    # two hues, and the same series is the same hue on every row
    colours = {line.get_color() for line in panel.lines if line.get_marker() in ("o", "s")}
    assert len(colours) == 2

    doubled = stratified_frame().vstack(stratified_frame().head(1))
    with pytest.raises(GeneticsUsageError, match="share label"):
        plots.forest(doubled, label="trait", series="sex")


def test_groups_are_headed_in_order_of_first_appearance_and_pooled_separately(
    monkeypatch, tmp_path
):
    frame = cohort_frame().with_columns(
        pl.Series("ancestry", ["European", "European", "Mixed", "European", "East Asian", "Mixed"])
    )
    result, canvas, _panel = drawn_forest(
        monkeypatch, tmp_path, frame, group="ancestry", pool="random", sort_by="estimate"
    )
    assert result["groups"] == ["European", "Mixed", "East Asian"]
    # largest first within a group, the pooled row after its own group's rows, and no
    # pooled row for a group of one
    assert result["rows"] == [
        "Estonia", "UK Biobank", "FinnGen", "Random effects meta-analysis",
        "All of Us", "MVP", "Random effects meta-analysis",
        "BBJ",
    ]
    assert [fit["group"] for fit in result["pooled"]] == ["European", "Mixed"]
    headings = [t for t in canvas.texts if t.get_text() in result["groups"]]
    assert all(t.get_fontweight() == "bold" for t in headings)


def test_the_cut_keeps_the_first_rows_and_says_how_many_it_left_out(monkeypatch, tmp_path):
    frame = pl.DataFrame({
        "cohort": [f"locus {i}" for i in range(40)],
        "beta": [0.01 * i for i in range(40)],
        "se": [0.02] * 40,
    })
    result, canvas, _panel = drawn_forest(monkeypatch, tmp_path, frame, max_rows=15)
    assert result["n_rows"] == 15 and result["n_omitted"] == 25
    assert result["rows"][0] == "locus 0" and result["rows"][-1] == "locus 14"
    assert "25 further rows not drawn" in canvas_texts(canvas)


def test_significance_hollows_the_markers_that_miss_it(monkeypatch, tmp_path):
    result, _canvas, panel = drawn_forest(
        monkeypatch, tmp_path, cohort_frame(), significance=5e-8
    )
    faces = [
        line.get_markerfacecolor() for line in panel.lines
        if line.get_marker() == "o" and line.get_clip_on()
    ]
    # FinnGen, UK Biobank and MVP pass 5e-8; the other three do not
    assert faces.count("white") == 3 and len(faces) == 6
    assert len(panel.get_legend().get_texts()) == 2

    with pytest.raises(GeneticsUsageError, match="significance= needs a p-value"):
        plots.forest(cohort_frame().drop("mlog10p"), label="cohort", significance=5e-8)


def test_a_weight_draws_squares_scaled_by_area_and_nothing_else_does(monkeypatch, tmp_path):
    _result, _canvas, panel = drawn_forest(monkeypatch, tmp_path, cohort_frame())
    plain = {(line.get_marker(), line.get_markersize()) for line in panel.lines if line.get_marker() == "o"}
    assert plain == {("o", plots._FOREST_MARKER_PT)}

    _result, _canvas, panel = drawn_forest(monkeypatch, tmp_path, cohort_frame(), weight="n_cases")
    sides = [line.get_markersize() for line in panel.lines if line.get_marker() == "s"]
    assert len(sides) == 6
    # MVP has the most cases and All of Us the fewest
    assert max(sides) == sides[2] == pytest.approx(plots._FOREST_WEIGHT_PT[1])
    assert min(sides) == sides[5] and min(sides) >= plots._FOREST_WEIGHT_PT[0]


def test_extra_columns_and_the_table_switch(monkeypatch, tmp_path):
    result, canvas, _panel = drawn_forest(
        monkeypatch, tmp_path, cohort_frame(), columns={"n_cases": "Cases"}, italic=True
    )
    texts = canvas_texts(canvas)
    assert "Cases" in texts and "48,211" in texts
    assert {t.get_fontstyle() for t in canvas.texts if t.get_text() == "FinnGen"} == {"italic"}
    wide = result["size_in"][0]

    result, canvas, _panel = drawn_forest(
        monkeypatch, tmp_path, cohort_frame(), columns={"n_cases": "Cases"}, table=False
    )
    assert "Cases" not in canvas_texts(canvas) and "48,211" not in canvas_texts(canvas)
    assert result["size_in"][0] < wide


def test_a_label_is_cut_to_length_and_a_dollar_sign_is_not_mathtext(monkeypatch, tmp_path):
    frame = cohort_frame(cohort=[
        "Costs between $5 and $10 per sample", "x" * 80, "MVP", "Estonia", "BBJ", "All of Us",
    ])
    result, _canvas, _panel = drawn_forest(monkeypatch, tmp_path, frame)
    assert result["rows"][0] == r"Costs between \$5 and \$10 per sample"
    assert len(result["rows"][1]) == 48 and result["rows"][1].endswith("…")
    assert (tmp_path / "forest.png").stat().st_size > 0


def test_an_explicit_xlim_cuts_what_lies_outside_it(monkeypatch, tmp_path):
    result, _canvas, panel = drawn_forest(
        monkeypatch, tmp_path, cohort_frame(), scale="log_ratio", xlim=(1.0, 1.4)
    )
    assert result["xlim"] == (1.0, 1.4)
    assert panel.get_xlim() == pytest.approx((1.0, 1.4))
    # UK Biobank's upper bound, Estonia's estimate with it, and All of Us on both sides
    assert result["n_clipped"] == 3


def test_forest_into_a_caller_supplied_axis_saves_nothing(monkeypatch, tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setenv("SANDBOX_ARTIFACTS_DIR", str(tmp_path))
    figure, (left, right) = plt.subplots(1, 2)
    result = plots.forest(cohort_frame(), label="cohort", ax=left)
    assert result["path"] is None
    assert not list(tmp_path.iterdir())
    assert left.child_axes and not right.child_axes
    plt.close("all")


@pytest.mark.parametrize(
    "mlog10p, expected",
    [
        (0.55, "0.28"),
        (2.0, "0.01"),
        (7.3, r"$5.0\times10^{-8}$"),
        (8.0, r"$1.0\times10^{-8}$"),
        (400.5, r"$3.2\times10^{-401}$"),
        (None, ""),
        (float("inf"), ""),
    ],
)
def test_a_p_value_is_set_the_way_a_journal_sets_it(mlog10p, expected):
    assert plots._forest_p(mlog10p) == expected


@pytest.mark.parametrize(
    "lo, hi, expected",
    [
        (0.7, 1.6, [0.8, 1.0, 1.2, 1.4, 1.6]),
        (0.18, 5.6, [0.2, 0.5, 1.0, 2.0, 5.0]),
        # too many decades to tick each: every other one, counted from the null
        (1e-4, 1e4, [1e-4, 1e-2, 1.0, 100.0, 1e4]),
    ],
)
def test_a_ratio_axis_is_ticked_at_the_numbers_a_reader_expects(lo, hi, expected):
    assert plots._forest_ratio_ticks(lo, hi) == pytest.approx(expected)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"scale": "odds"}, "scale must be one of"),
        ({"pool": "bayes"}, "pool must be one of"),
        ({"sort_by": "p"}, "sort_by must be one of"),
        ({"ci": 95}, "coverage"),
        ({"lower": "beta"}, "both lower= and upper="),
        ({"se": "stderr"}, "no 'stderr' column"),
        ({"label": "study"}, "not in the frame"),
        ({"columns": ["n_controls"]}, "not in the frame"),
        ({"direction": "higher risk"}, "direction is a pair"),
        ({"xlim": (2, 1)}, "xlim must be increasing"),
        ({"scale": "log_ratio", "xlim": (-1, 2)}, "positive on a ratio axis"),
    ],
)
def test_forest_refuses_what_it_cannot_draw(kwargs, message):
    kwargs.setdefault("label", "cohort")
    with pytest.raises(GeneticsUsageError, match=message):
        plots.forest(cohort_frame(), **kwargs)


def test_an_empty_frame_is_refused_rather_than_drawn_as_an_empty_table():
    with pytest.raises(GeneticsUsageError, match="nothing to plot"):
        plots.forest(cohort_frame().head(0), label="cohort")
