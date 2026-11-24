"""genetics.structure.view and .pml write what the chat's structure viewer and PyMOL can read.

The viewer's parser is genetics-results-browser `src/features/structure/viewSpec.ts`. Its
rules are ported below (`browser_parse`) so the written files are always held to them, and
when node and a checkout of that file are found the real parser is run over the same files:
set STRUCTURE_VIEW_SPEC_TS to it, or keep genetics-results-browser beside this repo.
"""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import polars as pl
import pytest

from genetics_mcp_server.sdk import _structure_contract as contract
from genetics_mcp_server.sdk import structure as st
from genetics_mcp_server.sdk.errors import GeneticsUsageError

SUFFIX = ".structure-view.json"


def _rows(windows) -> list[dict]:
    return [
        dict(
            model_file_id=f"T-F{k}",
            accession="T",
            protein_start=s,
            protein_end=e,
            n_residues=e - s + 1,
            n_windows=len(windows),
            has_pae=False,
        )
        for k, (s, e) in enumerate(windows, start=1)
    ]


# no model file is given: placing residues needs the windows alone
ONE = st.protein(_rows([(1, 3000)]), {})
TWO = st.protein(_rows([(1, 6), (4, 9)]), {})


@pytest.fixture(autouse=True)
def artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("SANDBOX_ARTIFACTS_DIR", str(tmp_path))
    return tmp_path


class SpecError(Exception):
    pass


def browser_parse(text: str) -> dict:
    """viewSpec.ts's parseViewSpecText for version 1, raising where it answers `invalid`."""

    def position(v, what):
        if isinstance(v, bool) or not isinstance(v, int) or v < 1 or v > 100_000:
            raise SpecError(what)
        return v

    def text_of(v, what):
        if v is None:
            return None
        if not isinstance(v, str):
            raise SpecError(what)
        units = v.encode("utf-16-le")
        return v if len(units) <= 160 else units[:158].decode("utf-16-le", "ignore") + "…"

    if len(text.encode()) > 1_000_000:
        raise SpecError("size")
    raw = json.loads(text)
    if not isinstance(raw, dict) or raw.get("version") != 1:
        raise SpecError("version")
    mid = raw.get("model_file_id")
    if not isinstance(mid, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", mid):
        raise SpecError("model_file_id")
    roles = contract.VIEW_ROLE_COLOURS
    by_position: dict[int, dict] = {}
    residues = raw.get("residues")
    if residues is not None and not isinstance(residues, list):
        raise SpecError("residues")
    for r in residues or []:
        if not isinstance(r, dict):
            raise SpecError("residue")
        entry = {
            "filePosition": position(r.get("file_position"), "file_position"),
            "label": text_of(r.get("label"), "label"),
        }
        role, colour = r.get("role"), r.get("colour")
        if role is not None and colour is not None:
            raise SpecError("both")
        if role is not None:
            if role not in roles:
                raise SpecError("role")
            entry["colour"] = roles[role]
        elif colour is not None:
            if not isinstance(colour, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", colour):
                raise SpecError("colour")
            entry["colour"] = colour.lower()
        else:
            entry["colour"] = roles["neutral"]
        first = by_position.setdefault(entry["filePosition"], entry)
        if not (first["label"] or "").strip() and (entry["label"] or "").strip():
            first["label"] = entry["label"]
    distinct = list(by_position.values())
    sphere = raw.get("sphere")
    if sphere is not None:
        radius = sphere.get("radius") if isinstance(sphere, dict) else None
        if isinstance(radius, bool) or not isinstance(radius, (int, float)):
            raise SpecError("radius")
        if not 0 < radius <= 100:
            raise SpecError("radius")
        sphere = {"centre": position(sphere.get("centre"), "centre"), "radius": radius}
    mode = raw.get("colour_mode") or "plddt"
    if mode not in ("plddt", "uniform"):
        raise SpecError("colour_mode")
    regions = raw.get("regions")
    if regions is not None and not isinstance(regions, list):
        raise SpecError("regions")
    spans = []
    for r in regions or []:
        if not isinstance(r, dict):
            raise SpecError("region")
        name = text_of(r.get("name"), "name")
        if not (name or "").strip():
            raise SpecError("name")
        start, end = position(r.get("start"), "start"), position(r.get("end"), "end")
        if start > end:
            raise SpecError("order")
        colour = r.get("colour")
        if colour is not None:
            if not isinstance(colour, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", colour):
                raise SpecError("colour")
            colour = colour.lower()
        spans.append({"name": name, "start": start, "end": end, "colour": colour})
    return {
        "modelFileId": mid,
        "accession": text_of(raw.get("accession"), "accession"),
        "gene": text_of(raw.get("gene"), "gene"),
        "residues": distinct[:2000],
        "sphere": sphere,
        "colourMode": mode,
        "regions": spans[:50],
        "omitted": {"residues": max(0, len(distinct) - 2000), "regions": max(0, len(spans) - 50)},
    }


def _parsed(result: dict) -> dict:
    path = Path(result["path"])
    assert path.name.lower().endswith(SUFFIX)
    assert path.stat().st_size <= 512 * 1024
    return browser_parse(path.read_text(encoding="utf-8"))


SINGLE_ARGS = dict(
    highlights=[
        {"position": 175, "label": "p.R175H", "role": "case"},
        {"position": 248, "label": "p.R248Q", "role": "control"},
        {"position": 273, "colour": "#00AA55"},
        282,
        {"position": 175, "label": "ignored", "role": "both"},
        {"position": 282, "label": "p.R282W"},
    ],
    sphere=(175, 15),
    gene="TP53",
)


class TestOneFile:
    def test_the_written_view_parses_under_the_browsers_rules(self, artifacts):
        result = st.view(ONE, **SINGLE_ARGS)
        assert result["path"] == str(artifacts / f"T-F1{SUFFIX}")
        spec = _parsed(result)
        assert (spec["modelFileId"], spec["accession"], spec["gene"]) == ("T-F1", "T", "TP53")
        assert spec["sphere"] == {"centre": 175, "radius": 15.0}
        assert spec["colourMode"] == "plddt"
        assert spec["omitted"] == {"residues": 0, "regions": 0}
        assert [(r["filePosition"], r["label"], r["colour"]) for r in spec["residues"]] == [
            (175, "p.R175H", "#d62728"),
            (248, "p.R248Q", "#1f77b4"),
            (273, None, "#00aa55"),
            (282, "p.R282W", "#7f7f7f"),
        ]
        assert result["n_residues"] == 4 and result["n_labels"] == 3
        assert result["outside_model_file"] == [] and result["other_model_files"] == []
        assert result["notes"] == [] and result["labels_not_drawn"] == []

    def test_a_repeated_residue_is_written_once_as_the_viewer_would_merge_it(self):
        written = json.loads(Path(st.view(ONE, **SINGLE_ARGS)["path"]).read_text())
        assert written["residues"] == [
            {"file_position": 175, "label": "p.R175H", "role": "case"},
            {"file_position": 248, "label": "p.R248Q", "role": "control"},
            {"file_position": 273, "colour": "#00aa55"},
            {"file_position": 282, "label": "p.R282W"},
        ]

    def test_a_model_file_id_takes_file_positions(self):
        result = st.view("AF-P04637-F1-model_v6", [5, ("AF-P04637-F1-model_v6", 7)], (5, 8.5))
        spec = _parsed(result)
        assert [r["filePosition"] for r in spec["residues"]] == [5, 7]
        assert spec["accession"] is None and spec["sphere"]["radius"] == 8.5
        with pytest.raises(GeneticsUsageError, match="another model file"):
            st.view("AF-P04637-F1-model_v6", [("T-F1", 7)])

    def test_no_highlights_is_the_bare_model(self):
        spec = _parsed(st.view(ONE, colour_mode="uniform"))
        assert spec["residues"] == [] and spec["colourMode"] == "uniform"

    def test_path_is_resolved_as_a_plot_path_and_keeps_the_ending(self, artifacts, tmp_path):
        assert st.view(ONE, path="tp53.Structure-View.JSON")["path"] == str(
            artifacts / "tp53.Structure-View.JSON"
        )
        absolute = str(tmp_path / "sub.structure-view.json")
        assert st.view(ONE, path=absolute)["path"] == absolute
        with pytest.raises(GeneticsUsageError, match="must end with"):
            st.view(ONE, path="tp53.json")


class TestSeveralFiles:
    def test_the_file_owning_the_first_highlight_is_shown_and_the_rest_reported(self):
        result = st.view(TWO, [2, {"position": 5, "label": "five"}, {"position": 8, "label": "x"}])
        spec = _parsed(result)
        assert spec["modelFileId"] == "T-F1"
        assert [r["filePosition"] for r in spec["residues"]] == [2, 5]
        assert result["outside_model_file"] == [
            {"position": 8, "label": "x", "model_file_id": "T-F2"}
        ]
        assert result["other_model_files"] == ["T-F2"]
        assert "T-F2" in result["notes"][0] and "one view per file" in result["notes"][0]

    def test_one_view_per_file_places_each_residue_in_that_files_numbering(self):
        highlights = [2, 5, 8]
        first = st.view(TWO, highlights)
        views = {first["model_file_id"]: first}
        for mid in first["other_model_files"]:
            views[mid] = st.view(TWO, highlights, model_file_id=mid)
        assert sorted(views) == ["T-F1", "T-F2"]
        # window 2 starts at protein position 4; position 5 is in both windows
        assert [r["filePosition"] for r in _parsed(views["T-F2"])["residues"]] == [2, 5]
        assert views["T-F2"]["outside_model_file"] == [
            {"position": 2, "label": None, "model_file_id": "T-F1"}
        ]
        placed = {
            (mid, r["filePosition"]) for mid, v in views.items() for r in _parsed(v)["residues"]
        }
        for mid, file_position in placed:
            s, e = TWO.windows[TWO.model_file_ids.index(mid)]
            assert s + file_position - 1 in highlights and file_position <= e - s + 1

    def test_an_address_names_its_own_file(self):
        # protein position 4 is owned by the first file; the address asks for the second
        result = st.view(TWO, [("T-F2", 1), 5])
        assert result["model_file_id"] == "T-F2"
        assert [r["filePosition"] for r in _parsed(result)["residues"]] == [1, 2]

    def test_the_sphere_centre_chooses_before_the_first_highlight(self):
        result = st.view(TWO, [2, 8], sphere=(8, 10))
        assert result["model_file_id"] == "T-F2"
        assert result["sphere"] == {"centre": 5, "radius": 10.0}
        assert [o["position"] for o in result["outside_model_file"]] == [2]

    def test_a_sphere_whose_centre_the_chosen_file_lacks_is_left_out_and_said(self):
        result = st.view(TWO, [2], sphere=(8, 10), model_file_id="T-F1")
        assert result["sphere"] is None and result["sphere_left_out"]
        assert _parsed(result)["sphere"] is None
        assert any("no sphere" in n for n in result["notes"])

    def test_nothing_to_choose_a_file_by_is_an_error(self):
        with pytest.raises(GeneticsUsageError, match="nothing was given to choose"):
            st.view(TWO)
        with pytest.raises(GeneticsUsageError, match="not a model file of T"):
            st.view(TWO, [2], model_file_id="T-F9")
        with pytest.raises(GeneticsUsageError, match="outside 1..9"):
            st.view(TWO, [10])


def _features(rows) -> pl.DataFrame:
    return pl.DataFrame(
        [
            dict(source="uniprot", feature_type=t, start=a, end=b, description=d, evidence="")
            for t, a, b, d in rows
        ]
    )


class TestRegions:
    def test_regions_are_written_in_order_and_parse_under_the_browsers_rules(self):
        regions = [
            {"name": " DNA-binding ", "start": 102, "end": 292, "colour": "#2CA02C"},
            {"name": "Tetramerisation", "start": 325, "end": 356},
            {"name": "inside the first", "start": 175, "end": 175},
        ]
        result = st.view(ONE, [175], regions=regions)
        written = json.loads(Path(result["path"]).read_text())
        assert written["version"] == 1
        assert written["regions"] == [
            {"name": "DNA-binding", "start": 102, "end": 292, "colour": "#2ca02c"},
            {"name": "Tetramerisation", "start": 325, "end": 356},
            {"name": "inside the first", "start": 175, "end": 175},
        ]
        assert result["regions"] == written["regions"] and result["n_regions"] == 3
        assert result["regions_clipped"] == [] and result["regions_outside_model_file"] == []
        assert result["notes"] == []
        assert [
            (r["name"], r["start"], r["end"], r["colour"]) for r in _parsed(result)["regions"]
        ] == [
            ("DNA-binding", 102, 292, "#2ca02c"),
            ("Tetramerisation", 325, 356, None),
            ("inside the first", 175, 175, None),
        ]

    def test_no_regions_writes_no_key(self):
        assert "regions" not in json.loads(Path(st.view(ONE, [1])["path"]).read_text())
        assert st.view(ONE, [1])["n_regions"] == 0

    def test_rows_of_the_features_frame_are_regions_as_they_stand(self):
        features = _features(
            [("DOMAIN", 5, 20, "Kinase"), ("REGION", 30, 40, None), ("DOMAIN", 50, 60, "  ")]
        )
        expected = [
            {"name": "Kinase", "start": 5, "end": 20},
            {"name": "REGION", "start": 30, "end": 40},
            {"name": "DOMAIN", "start": 50, "end": 60},
        ]
        assert st.view(ONE, regions=features)["regions"] == expected
        assert st.view(ONE, regions=features.to_dicts())["regions"] == expected
        coloured = features.with_columns(name=pl.lit("mine"), colour=pl.lit("#000000"))
        assert st.view(ONE, regions=coloured)["regions"][0] == {
            "name": "mine",
            "start": 5,
            "end": 20,
            "colour": "#000000",
        }

    def test_a_model_file_id_takes_file_positions(self):
        result = st.view("AF-X-F1-model_v6", regions=[{"name": "a", "start": 1, "end": 100_000}])
        assert result["regions"] == [{"name": "a", "start": 1, "end": 100_000}]

    def test_a_region_is_cut_to_the_model_file_and_one_outside_it_is_left_out(self):
        # windows are protein positions 1..6 and 4..9
        regions = [
            {"name": "whole", "start": 1, "end": 9, "colour": "#111111"},
            {"name": "head", "start": 1, "end": 3},
            {"name": "tail", "start": 7, "end": 9},
            {"name": "overlap", "start": 5, "end": 7},
        ]
        first = st.view(TWO, [2], regions=regions)
        assert first["model_file_id"] == "T-F1"
        assert first["regions"] == [
            {"name": "whole", "start": 1, "end": 6, "colour": "#111111"},
            {"name": "head", "start": 1, "end": 3},
            {"name": "overlap", "start": 5, "end": 6},
        ]
        assert first["regions_clipped"] == [
            {"name": "whole", "start": 1, "end": 9, "model_file_ids": ["T-F2"]},
            {"name": "overlap", "start": 5, "end": 7, "model_file_ids": ["T-F2"]},
        ]
        assert first["regions_outside_model_file"] == [
            {"name": "tail", "start": 7, "end": 9, "model_file_ids": ["T-F2"]}
        ]
        assert first["other_model_files"] == ["T-F2"]
        assert any("tail" in n and "left out" in n for n in first["notes"])
        assert any("whole, overlap" in n and "only as far" in n for n in first["notes"])
        second = st.view(TWO, regions=regions, model_file_id="T-F2")
        # window 2 starts at protein position 4, so protein 5..7 is file 2..4
        assert second["regions"] == [
            {"name": "whole", "start": 1, "end": 6, "colour": "#111111"},
            {"name": "tail", "start": 4, "end": 6},
            {"name": "overlap", "start": 2, "end": 4},
        ]
        assert [o["name"] for o in second["regions_outside_model_file"]] == ["head"]
        assert second["regions_clipped"] == [
            {"name": "whole", "start": 1, "end": 9, "model_file_ids": ["T-F1"]}
        ]
        assert _parsed(second)["regions"][1]["end"] == 6

    def test_a_clipped_region_names_the_files_holding_the_rest_of_it(self):
        result = st.view(TWO, [2], regions=[{"name": "whole", "start": 1, "end": 9}])
        assert result["model_file_id"] == "T-F1"
        assert result["regions_clipped"] == [
            {"name": "whole", "start": 1, "end": 9, "model_file_ids": ["T-F2"]}
        ]
        assert result["regions_outside_model_file"] == []
        assert result["other_model_files"] == ["T-F2"]
        assert any("only as far" in n and "regions_clipped" in n for n in result["notes"])
        # following other_model_files reaches the part cut off
        (mid,) = result["other_model_files"]
        assert st.view(TWO, regions=[{"name": "whole", "start": 1, "end": 9}], model_file_id=mid)[
            "regions"
        ] == [{"name": "whole", "start": 1, "end": 6}]
        # only a side that is cut off counts: 5..10 loses 10 from the middle file and nothing
        # before it, so the first file is not named although it holds 5..6
        three = st.protein(_rows([(1, 6), (4, 9), (7, 12)]), {})
        middle = st.view(
            three, regions=[{"name": "r", "start": 5, "end": 10}], model_file_id="T-F2"
        )
        assert middle["regions_clipped"][0]["model_file_ids"] == ["T-F3"]
        assert middle["other_model_files"] == ["T-F3"]
        both = st.view(three, regions=[{"name": "r", "start": 2, "end": 11}], model_file_id="T-F2")
        assert both["other_model_files"] == ["T-F1", "T-F3"]

    def test_the_highlight_note_names_only_the_highlights_files(self):
        three = st.protein(_rows([(1, 6), (4, 9), (7, 12)]), {})
        result = st.view(three, [2, 8], regions=[{"name": "r", "start": 1, "end": 12}])
        assert result["other_model_files"] == ["T-F2", "T-F3"]
        assert any(n.startswith("highlights") and "T-F3" not in n for n in result["notes"])

    def test_with_regions_alone_the_file_holding_most_of_the_first_region_is_shown(self):
        def chosen(start, end, protein=TWO, **kwargs):
            return st.view(protein, regions=[{"name": "t", "start": start, "end": end}], **kwargs)

        assert chosen(8, 9)["model_file_id"] == "T-F2"
        # 4..9 starts in the first file's own stretch but only the second holds it whole
        whole = chosen(4, 9)
        assert whole["model_file_id"] == "T-F2"
        assert whole["regions"] == [{"name": "t", "start": 1, "end": 6}]
        assert whole["regions_clipped"] == [] and whole["other_model_files"] == []
        # held whole by both: the earlier file
        assert chosen(4, 6)["model_file_id"] == "T-F1"
        # held whole by neither: 3..9 has four residues in the first file and six in the second
        assert chosen(3, 9)["model_file_id"] == "T-F2"
        assert chosen(1, 7)["model_file_id"] == "T-F1"
        # as much in each: the earlier file
        assert chosen(1, 9)["model_file_id"] == "T-F1"
        # only the first region chooses
        two = [{"name": "a", "start": 1, "end": 2}, {"name": "b", "start": 4, "end": 9}]
        assert st.view(TWO, regions=two)["model_file_id"] == "T-F1"
        # a highlight or a sphere still chooses before any region
        assert st.view(TWO, [2], regions=[{"name": "t", "start": 4, "end": 9}])[
            "model_file_id"
        ] == ("T-F1")
        assert st.view(TWO, [2], regions=[{"name": "t", "start": 8, "end": 9}])["n_regions"] == 0
        sphere = st.view(TWO, sphere=(2, 5), regions=[{"name": "t", "start": 4, "end": 9}])
        assert sphere["model_file_id"] == "T-F1"

    @pytest.mark.parametrize(
        "match, region",
        [
            ("start 3 is after end 2", {"name": "a", "start": 3, "end": 2}),
            ("start protein position 0 is outside 1..9", {"name": "a", "start": 0, "end": 2}),
            ("end protein position 10 is outside 1..9", {"name": "a", "start": 1, "end": 10}),
            ("must be an integer", {"name": "a", "start": 1, "end": None}),
        ],
    )
    def test_a_bad_first_region_is_refused_before_it_chooses_a_file(self, artifacts, match, region):
        with pytest.raises(GeneticsUsageError, match=match):
            st.view(TWO, regions=[region])
        assert list(artifacts.iterdir()) == []

    def test_the_cap(self, artifacts):
        many = [{"name": f"r{i}", "start": i, "end": i + 1} for i in range(1, 52)]
        assert contract.VIEW_MAX_REGIONS == 50
        assert st.view(ONE, regions=many[:50])["n_regions"] == 50
        os.remove(artifacts / f"T-F1{SUFFIX}")
        with pytest.raises(GeneticsUsageError, match="51 regions to mark.*at most 50"):
            st.view(ONE, regions=many)
        assert list(artifacts.iterdir()) == []
        # counted over what the chosen file holds, as the viewer counts
        spread = [{"name": "t", "start": 8, "end": 9}] * 30 + [
            {"name": "h", "start": 1, "end": 2}
        ] * 30
        assert st.view(TWO, regions=spread)["n_regions"] == 30

    @pytest.mark.parametrize(
        "match, region",
        [
            ("needs a name", {"start": 1, "end": 2}),
            ("needs a name", {"name": " ", "start": 1, "end": 2}),
            ("name must be text", {"name": 5, "start": 1, "end": 2}),
            ("is a dict with 'start', 'end'", {"name": "a", "start": 1}),
            ("is a dict with 'start', 'end'", (1, 2)),
            (
                "is a dict with 'start', 'end'",
                {"name": "a", "start": 1, "end": 2, "color": "#000000"},
            ),
            ("start 3 is after end 2", {"name": "a", "start": 3, "end": 2}),
            ("start protein position 0 is outside 1..3000", {"name": "a", "start": 0, "end": 2}),
            ("end protein position 3001 is outside", {"name": "a", "start": 1, "end": 3001}),
            ("must be an integer", {"name": "a", "start": 1.0, "end": 2}),
            ("must be an integer", {"name": "a", "start": 1, "end": True}),
            ("must be an integer", {"name": "a", "start": None, "end": 2}),
            ("#rrggbb", {"name": "a", "start": 1, "end": 2, "colour": "red"}),
            ("#rrggbb", {"name": "a", "start": 1, "end": 2, "colour": "#fff"}),
        ],
    )
    def test_a_bad_region_is_refused_and_nothing_written(self, artifacts, match, region):
        with pytest.raises(GeneticsUsageError, match=match):
            st.view(ONE, [1], regions=[region])
        assert list(artifacts.iterdir()) == []

    @pytest.mark.parametrize("key", ["name", "description", "feature_type"])
    def test_a_long_name_is_cut_as_the_viewer_would_cut_it(self, key):
        def named(text):
            result = st.view(ONE, regions=[{key: text, "start": 1, "end": 2}])
            assert _parsed(result)["regions"][0]["name"] == result["regions"][0]["name"]
            return result["regions"][0]["name"]

        assert named("x" * 80) == "x" * 80
        assert named("x" * 81) == "x" * 79 + "…"
        # counted in UTF-16 units, where a character outside the BMP is two
        assert named("\U0001f9ec" * 41) == "\U0001f9ec" * 39 + "…"
        # a cut that would land inside such a character leaves it out whole
        assert named("xx" + "\U0001f9ec" * 40) == "xx" + "\U0001f9ec" * 38 + "…"
        # the surrounding blanks are gone before the count
        assert named(" " + "x" * 80 + " ") == "x" * 80

    def test_what_is_not_a_list_of_regions(self, artifacts):
        for bad in ({"name": "a", "start": 1, "end": 2}, "DOMAIN", 5):
            with pytest.raises(GeneticsUsageError, match="regions is a list of dicts"):
                st.view(ONE, regions=bad)
        with pytest.raises(GeneticsUsageError, match="file position 100001 is outside"):
            st.view("AF-X-F1-model_v6", regions=[{"name": "a", "start": 1, "end": 100_001}])
        assert list(artifacts.iterdir()) == []

    def test_pml_takes_no_regions(self):
        assert "n_regions" not in st.pml(ONE, [1])


class TestLimits:
    def _refused(self, artifacts, match, *args, **kwargs):
        with pytest.raises(GeneticsUsageError, match=match):
            st.view(*args, **kwargs)
        assert list(artifacts.iterdir()) == []

    def test_residues(self, artifacts):
        assert st.view(ONE, list(range(1, 2001)))["n_residues"] == 2000
        os.remove(artifacts / f"T-F1{SUFFIX}")
        self._refused(artifacts, "at most 2000", ONE, list(range(1, 2002)))
        # counted after merging, as the viewer counts
        assert st.view(ONE, list(range(1, 2001)) * 2)["n_residues"] == 2000

    def test_labels_past_the_drawn_ones_are_kept_and_named(self):
        result = st.view(ONE, [{"position": i, "label": f"L{i}"} for i in range(1, 43)])
        assert result["n_labels"] == 42 and result["labels_not_drawn"] == ["L41", "L42"]
        assert "legend" in result["notes"][-1] and "hover" in result["notes"][-1]

    def test_text_length_is_counted_as_the_viewer_counts_it(self, artifacts):
        assert st.view(ONE, [{"position": 1, "label": "x" * 80}], gene="g" * 80)
        os.remove(artifacts / f"T-F1{SUFFIX}")
        self._refused(artifacts, "81 UTF-16 code units", ONE, [{"position": 1, "label": "x" * 81}])
        self._refused(artifacts, "gene is too long: 81", ONE, gene="g" * 81)
        # outside the BMP a character is two UTF-16 units
        self._refused(
            artifacts,
            "82 UTF-16 code units.*at most 80 ordinary",
            ONE,
            [{"position": 1, "label": "\U0001f9ec" * 41}],
        )
        self._refused(artifacts, "not valid Unicode", ONE, [{"position": 1, "label": "\ud800"}])
        self._refused(artifacts, "must be text", ONE, [{"position": 1, "label": 5}])

    def test_a_blank_label_is_no_label(self):
        result = st.view(ONE, [{"position": 1, "label": "  "}, {"position": 1, "label": " a "}])
        assert json.loads(Path(result["path"]).read_text())["residues"] == [
            {"file_position": 1, "label": "a"}
        ]

    @pytest.mark.parametrize("radius", [0, -1, 100.5, float("inf"), float("nan"), "15", True])
    def test_radius(self, artifacts, radius):
        self._refused(artifacts, "radius", ONE, sphere=(1, radius))

    def test_the_largest_radius_and_sphere_forms(self):
        assert st.view(ONE, sphere={"centre": 1, "radius": 100})["sphere"]["radius"] == 100.0
        with pytest.raises(GeneticsUsageError, match="sphere is"):
            st.view(ONE, sphere={"center": 1, "radius": 5})

    def test_positions(self, artifacts):
        mid = "AF-X-F1-model_v6"
        assert st.view(mid, [100_000])["n_residues"] == 1
        os.remove(artifacts / f"{mid}{SUFFIX}")
        self._refused(artifacts, "outside 1..100000", mid, [100_001])
        self._refused(artifacts, "outside 1..100000", mid, sphere=(100_001, 5))
        self._refused(artifacts, "must be an integer", mid, [1.5])
        self._refused(artifacts, "must be an integer", mid, [True])
        self._refused(artifacts, "outside 1..3000", ONE, [0])

    @pytest.mark.parametrize("colour", ["#fff", "red", "#12345g", "#123456\n", "#12345678", 5])
    def test_colour_is_rrggbb_only(self, artifacts, colour):
        self._refused(artifacts, "#rrggbb", ONE, [{"position": 1, "colour": colour}])

    def test_role_and_colour(self, artifacts):
        both = {"position": 1, "role": "case", "colour": "#000000"}
        self._refused(artifacts, "both role and colour", ONE, [both])
        self._refused(artifacts, "role must be one of", ONE, [{"position": 1, "role": "Case"}])
        self._refused(artifacts, "got keys", ONE, [{"position": 1, "color": "#000000"}])
        self._refused(artifacts, "got keys", ONE, [{"label": "x"}])
        self._refused(artifacts, "colour_mode", ONE, colour_mode="rainbow")
        self._refused(artifacts, "list of positions", ONE, 175)

    def test_the_file_stays_under_what_the_chat_delivers(self, artifacts):
        assert st._MAX_VIEW_BYTES == 512 * 1024 <= contract.VIEW_MAX_BYTES
        heavy = [{"position": i, "label": "→" * 80} for i in range(1, 2001)]
        self._refused(artifacts, "delivers at most 524288", ONE, heavy)
        assert _parsed(st.view(ONE, heavy[:1500]))["omitted"]["residues"] == 0

    @pytest.mark.parametrize("target", ["AF-X-F1\n", "", "a/b", "x" * 65, None, 5])
    def test_the_model_file_id_is_held_to_the_contracts_pattern(self, artifacts, target):
        self._refused(artifacts, "model file id", target, [1])

    def test_the_docstring_states_the_limits_the_contract_holds(self):
        doc = st.view.__doc__
        for said in (
            "2,000 residues",
            "50 regions",
            "80 characters",
            "100 Å",
            "512 KiB",
            "the 40 the viewer lists in its legend",
        ):
            assert said in doc
        assert (
            contract.VIEW_MAX_RESIDUES,
            contract.VIEW_MAX_TEXT_LENGTH,
            contract.VIEW_MAX_SPHERE_RADIUS,
            contract.VIEW_MAX_LABELS,
            contract.VIEW_MAX_POSITION,
            contract.VIEW_MAX_REGIONS,
        ) == (2000, 80, 100, 40, 100_000, 50)


def _real_parser() -> str | None:
    candidates = [os.environ.get("STRUCTURE_VIEW_SPEC_TS", "")] + [
        str(p / "genetics-results-browser" / "src" / "features" / "structure" / "viewSpec.ts")
        for p in Path(__file__).resolve().parents
    ]
    return next((c for c in candidates if c and Path(c).is_file()), None)


def test_the_browsers_own_parser_accepts_the_written_views(artifacts, tmp_path_factory):
    parser, node = _real_parser(), shutil.which("node")
    if parser is None or node is None:
        pytest.skip(
            "needs node and genetics-results-browser's viewSpec.ts (STRUCTURE_VIEW_SPEC_TS)"
        )
    many = [{"position": i, "label": f"L{i}", "role": "both"} for i in range(1, 2001)]
    results = [
        st.view(ONE, **SINGLE_ARGS, path="one.structure-view.json"),
        st.view(TWO, [2, 5, 8], sphere=(5, 100), path="two-f1.structure-view.json"),
        st.view(TWO, [2, 5, 8], model_file_id="T-F2", path="two-f2.structure-view.json"),
        st.view(ONE, many, colour_mode="uniform", path="many.structure-view.json"),
        st.view("AF-X-F1-model_v6", path="bare.Structure-View.JSON"),
        # a parser from before regions has to read this one too: the version did not change
        st.view(
            TWO,
            [2],
            regions=[{"name": "d", "start": 1, "end": 9, "colour": "#2ca02c"}],
            path="regions.structure-view.json",
        ),
        st.view(
            ONE,
            regions=[{"description": "d" * 200, "start": 1, "end": 9}],
            path="long-name.structure-view.json",
        ),
    ]
    paths = [r["path"] for r in results]
    script = tmp_path_factory.mktemp("node") / "parse.mjs"
    script.write_text(
        f"import * as v from {json.dumps(Path(parser).resolve().as_uri())};\n"
        'import {readFileSync} from "node:fs";\n'
        "const views = process.argv.slice(2).map((f) => ({\n"
        "  named: v.isStructureViewName(f), ...v.parseViewSpecText(readFileSync(f, 'utf8'))}));\n"
        "const pins = {VIEW_SPEC_VERSION: v.VIEW_SPEC_VERSION, VIEW_ROLES: v.VIEW_ROLES,\n"
        "  VIEW_COLOUR_MODES: v.VIEW_COLOUR_MODES, VIEW_NAME_SUFFIX: v.STRUCTURE_VIEW_SUFFIX,\n"
        "  VIEW_DEFAULT_COLOUR_MODE: v.VIEW_DEFAULT_COLOUR_MODE,\n"
        "  VIEW_ROLE_COLOURS: v.ROLE_COLOURS, VIEW_SPHERE_COLOUR: v.NEIGHBOURHOOD_COLOUR,\n"
        "  VIEW_MAX_BYTES: v.MAX_SPEC_BYTES, VIEW_MAX_RESIDUES: v.MAX_RESIDUES,\n"
        "  VIEW_MAX_LABELS: v.MAX_LABELS, VIEW_MAX_REGIONS: v.MAX_REGIONS,\n"
        "  VIEW_MAX_TEXT_LENGTH: v.MAX_TEXT_LENGTH,\n"
        "  VIEW_MAX_POSITION: v.MAX_POSITION, VIEW_MAX_SPHERE_RADIUS: v.MAX_SPHERE_RADIUS};\n"
        # a name the parser does not export is written as null, so its pin fails by name
        # where JSON.stringify would have dropped the key and the comparison with it
        "console.log(JSON.stringify({views, pins}, (k, x) => (x === undefined ? null : x)));\n"
    )
    run = subprocess.run(
        [node, "--experimental-strip-types", "--no-warnings", str(script), *paths],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert run.returncode == 0, run.stderr
    out = json.loads(run.stdout)
    for name, value in out["pins"].items():
        mine = getattr(contract, name)
        assert (list(mine) if isinstance(mine, tuple) else mine) == value, name
    for path, result, real in zip(paths, results, out["views"]):
        assert real["named"] and real["ok"], (path, real)
        spec, port = real["spec"], browser_parse(Path(path).read_text(encoding="utf-8"))
        assert spec["modelFileId"] == result["model_file_id"] == port["modelFileId"]
        assert len(spec["residues"]) == result["n_residues"]
        assert spec["omitted"] == port["omitted"]
        assert spec["omitted"] == {"residues": 0, "regions": 0}
        assert [{k: v for k, v in r.items() if v is not None} for r in spec["regions"]] == result[
            "regions"
        ]
        assert [(r["name"], r["start"], r["end"], r.get("colour")) for r in spec["regions"]] == [
            (r["name"], r["start"], r["end"], r["colour"]) for r in port["regions"]
        ]
        assert spec.get("sphere") == result["sphere"] == port["sphere"]
        # the port and the real parser agree residue by residue, so the port's verdicts on
        # the other tests' files stand for the browser's
        assert [(r["filePosition"], r.get("label"), r["colour"]) for r in spec["residues"]] == [
            (r["filePosition"], r["label"], r["colour"]) for r in port["residues"]
        ]


# --------------------------------------------------------------------------- PyMOL

HOSTILE = [
    'x"); import os; os.system("touch pwned") #',
    "'+__import__('os').system('touch pwned')+'",
    "a\npython end\n/import os\nsystem touch pwned",
    "back\\slash\\",
    '"""',
    "'''",
    "semi; colon, comma # hash",
    "p.Arg175His → case",
    "\r\x00\x1b[2J",
]


class _Cmd:
    def __init__(self):
        self.labels = []

    def label(self, selection, expression):
        # PyMOL evaluates the expression as Python for each atom of the selection
        self.labels.append((selection, eval(expression, {"__builtins__": {}})))


_PLAIN_LINE = re.compile(
    r"(# [A-Za-z0-9 ._-]+"
    r"|load [A-Za-z0-9._-]+, structure"
    r"|(hide everything|show cartoon|orient), structure|orient structure|deselect"
    r"|set_color [a-z0-9_]+, \[[01]\.\d{4}, [01]\.\d{4}, [01]\.\d{4}\]"
    r"|color [a-z0-9_]+, structure( and b > \d+)?"
    r"|(show spheres|color [a-z0-9_]+), structure and name CA and resi \d+(\+\d+)*"
    r"|select neighbourhood, byres \(\(structure and name CA\) within \d+\.\d{3} of "
    r"\(structure and name CA and resi \d+\)\))"
)


def _split(text: str) -> tuple[list[str], list[str]]:
    """the script's command lines and the lines of its python block."""
    lines = text.splitlines()
    if "python" not in lines:
        return lines, []
    start, end = lines.index("python"), lines.index("python end")
    assert lines.count("python") == 1 and lines.count("python end") == 1
    return lines[:start] + lines[end + 1 :], lines[start + 1 : end]


class TestPml:
    def test_the_script_applies_the_views_marking(self, artifacts):
        result = st.pml(ONE, **{k: v for k, v in SINGLE_ARGS.items() if k != "gene"})
        assert result["path"] == str(artifacts / "T-F1.pml")
        assert result["model_file"] == "T-F1.pdb" and result["n_residues"] == 4
        text = Path(result["path"]).read_text(encoding="ascii")
        plain, block = _split(text)
        assert "load T-F1.pdb, structure" in plain
        bands = [line for line in plain if line.startswith("color plddt_")]
        assert bands == [
            "color plddt_very_low, structure",
            "color plddt_low, structure and b > 50",
            "color plddt_confident, structure and b > 70",
            "color plddt_very_high, structure and b > 90",
        ]
        assert "set_color plddt_very_high, [0.0000, 0.3255, 0.8392]" in plain
        assert "set_color highlight_d62728, [0.8392, 0.1529, 0.1569]" in plain
        assert "show spheres, structure and name CA and resi 175" in plain
        assert "color highlight_d62728, structure and name CA and resi 175" in plain
        assert "color highlight_00aa55, structure and name CA and resi 273" in plain
        assert "color highlight_7f7f7f, structure and name CA and resi 282" in plain
        assert (
            "select neighbourhood, byres ((structure and name CA) within 15.000 of "
            "(structure and name CA and resi 175))"
        ) in plain
        assert all(_PLAIN_LINE.fullmatch(line) for line in plain), plain
        cmd = _Cmd()
        exec("\n".join(block).replace("from pymol import cmd", ""), {"cmd": cmd})
        assert cmd.labels == [
            ("structure and name CA and resi 175", "p.R175H"),
            ("structure and name CA and resi 248", "p.R248Q"),
            ("structure and name CA and resi 282", "p.R282W"),
        ]

    def test_uniform_colouring_and_long_residue_lists(self):
        result = st.pml(ONE, list(range(1, 121)), colour_mode="uniform")
        plain, block = _split(Path(result["path"]).read_text())
        assert "color grey80, structure" in plain and not block
        assert not any("plddt" in line for line in plain)
        shown = [line for line in plain if line.startswith("show spheres")]
        assert [line.count("+") + 1 for line in shown] == [50, 50, 20]

    def test_a_label_cannot_leave_its_string(self, artifacts, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = st.pml(ONE, [{"position": i, "label": t} for i, t in enumerate(HOSTILE, 1)])
        raw = Path(result["path"]).read_bytes()
        assert raw.isascii() and b"\r" not in raw and b"\x00" not in raw
        plain, block = _split(raw.decode("ascii"))
        # outside the block every line is one of the fixed command shapes; inside it every
        # line but the import is one call whose arguments are two string literals
        assert all(_PLAIN_LINE.fullmatch(line) for line in plain), plain
        assert block[0] == "from pymol import cmd" and len(block) == 1 + len(HOSTILE)
        for line in block[1:]:
            tree = compile(line, "pml", "eval", flags=0x400)  # ast.PyCF_ONLY_AST
            call = tree.body
            assert type(call).__name__ == "Call" and len(call.args) == 2 and not call.keywords
            assert all(type(a).__name__ == "Constant" for a in call.args)
        cmd = _Cmd()
        exec("\n".join(block[1:]), {"cmd": cmd, "__builtins__": {}})
        assert [text for _, text in cmd.labels] == [t.strip() for t in HOSTILE]
        assert sorted(p.name for p in artifacts.iterdir()) == ["T-F1.pml"]

    def test_several_files_and_the_file_name(self, artifacts):
        result = st.pml(TWO, [2, 8], model_file="T-F1.pdb.gz")
        assert result["model_file_id"] == "T-F1" and result["other_model_files"] == ["T-F2"]
        assert "load T-F1.pdb.gz, structure" in Path(result["path"]).read_text()
        for bad in ["a b.pdb", "x.pdb\n", "x.pdb\nquit", "../x.pdb", "x.pdb; quit", "", 5]:
            with pytest.raises(GeneticsUsageError, match="bare file name"):
                st.pml(ONE, [1], model_file=bad)
        with pytest.raises(GeneticsUsageError, match="must end with '.pml'"):
            st.pml(ONE, [1], path="x.txt")
        assert sorted(p.name for p in artifacts.iterdir()) == ["T-F1.pml"]


def test_view_and_pml_are_exported():
    assert {"view", "pml"} <= set(st.__all__)


# --------------------------------------------------------------------------- hostile ids

BAD_IDS = ["A\n/import os;os.system('id')", "T-F1\n", "../../evil", "a b", "", "x" * 65, None]


class TestIds:
    @pytest.mark.parametrize("bad", BAD_IDS)
    @pytest.mark.parametrize("column", ["model_file_id", "accession"])
    def test_protein_refuses_a_row_whose_id_is_outside_the_contract(self, bad, column):
        with pytest.raises(GeneticsUsageError, match=f"model row's {column} must be"):
            st.protein([{**_rows([(1, 5)])[0], column: bad}], {})

    @pytest.mark.parametrize("bad", BAD_IDS)
    @pytest.mark.parametrize("attribute", ["model_file_ids", "accession"])
    def test_the_writers_check_again_whatever_built_the_object(self, artifacts, bad, attribute):
        p = st.protein(_rows([(1, 5)]), {})
        setattr(p, attribute, [bad] if attribute == "model_file_ids" else bad)
        for write in (st.view, lambda *a: st.pml(*a, model_file="x.pdb")):
            with pytest.raises(GeneticsUsageError, match="must be 1 to 64 letters"):
                write(p, [1])
        assert list(artifacts.iterdir()) == []
        assert list(artifacts.parent.glob("evil*")) == []

    def test_a_script_line_outside_the_emitted_shapes_is_never_written(
        self, artifacts, monkeypatch
    ):
        monkeypatch.setattr(st, "_PML_OBJECT", "structure\n/import os")
        with pytest.raises(RuntimeError, match="internal error"):
            st.pml(ONE, [1])
        monkeypatch.undo()
        for line in (
            "/import os",
            "load x.pdb, structure; quit",
            "# caf\u00e9",
            "cmd.label('x', 'y')",
        ):
            with pytest.raises(RuntimeError, match="internal error"):
                st._check_pml(["deselect", line])
        with pytest.raises(RuntimeError, match="internal error"):
            st._check_pml(["python", "import os", "python end"])
        assert list(artifacts.iterdir()) == []


class TestBlankAsTheBrowserSeesIt:
    def test_a_label_javascript_trims_to_nothing_is_no_label(self):
        result = st.view(
            ONE,
            [
                {"position": 1, "label": "\ufeff"},
                {"position": 1, "label": "\ufeff real\u2028"},
                {"position": 2, "label": "\u00a0\u3000\ufeff"},
            ],
        )
        assert result["n_labels"] == 1
        assert json.loads(Path(result["path"]).read_text(encoding="utf-8"))["residues"] == [
            {"file_position": 1, "label": "real"},
            {"file_position": 2},
        ]

    def test_what_python_strips_and_javascript_keeps_stays_a_label(self):
        result = st.view(ONE, [{"position": 1, "label": "\x1f"}, {"position": 1, "label": "x"}])
        assert json.loads(Path(result["path"]).read_text())["residues"][0]["label"] == "\x1f"

    def test_the_trimmed_set_is_javascripts(self):
        want = {0x9, 0xA, 0xB, 0xC, 0xD, 0x20, 0xA0, 0x1680, *range(0x2000, 0x200B)}
        want |= {0x2028, 0x2029, 0x202F, 0x205F, 0x3000, 0xFEFF}
        assert {ord(c) for c in st._JS_TRIMMED} == want


def test_the_text_length_error_says_what_is_counted():
    with pytest.raises(GeneticsUsageError) as caught:
        st.view(ONE, [{"position": 1, "label": "\U0001f9ec" * 80}])
    message = str(caught.value)
    assert "160 UTF-16 code units" in message and "160 characters" not in message
    assert "at most 80 ordinary characters" in message and "is two" in message


def test_pymol_colours_are_fractions_so_a_dark_colour_stays_dark():
    text = Path(st.pml(ONE, [{"position": 1, "colour": "#010101"}])["path"]).read_text()
    assert "set_color highlight_010101, [0.0039, 0.0039, 0.0039]" in text
    assert not re.search(r"set_color \w+, \[\d+,", text)


def test_a_path_in_a_missing_directory_is_a_usage_error(artifacts):
    for write, name in ((st.view, "no/x.structure-view.json"), (st.pml, "no/x.pml")):
        with pytest.raises(GeneticsUsageError, match="directory that does not exist"):
            write(ONE, [1], path=name)
    assert list(artifacts.iterdir()) == []


def test_the_docstrings_say_what_a_run_delivers():
    for function in (st.view, st.pml):
        assert "at most four non-image files" in function.__doc__
        assert "512 KiB" in function.__doc__
    assert "25 of the 21,946 models" in st.pml.__doc__
