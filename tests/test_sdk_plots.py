"""The standard plots draw the right picture from the frames the SDK actually returns.

Everything here passes `data=` and turns the LD and gene fetches off, so no test reaches a
service: what is under test is the drawing and the decisions around it, not the client.

What these CANNOT check is the house style — the rcParams come from a matplotlibrc the
sandbox image bakes, so a figure drawn here is drawn under this venv's matplotlib defaults.
That half is asserted where it is true, in genetics-results-suite/sandbox/build-checks.py.
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
