"""genetics.structure reproduces the lab's geometry on the lab's own reference bundle.

The fixtures under tests/fixtures/structure/ are what the lab's `utils.py` returned for a
sample of proteins (generate.py beside them, which does not import the SDK), with the inputs
in reduced form: model files cut to their CA records and PAE as the release's uint8 objects.
Neighbourhoods are held to exact agreement. Distances are held to one unit in the last place
of single precision: the port forms the squared distance by differences and the lab's
scikit-learn by a dot-product expansion, both in double precision before rounding to single,
so the two can land on adjacent single-precision values and no further apart.

Proteins whose PAE is too large to commit carry only their sampled residues' PAE rows, so
their PAE-filtered adjacency is checked row by row here and in full only when
PROTEIN_STRUCTURE_RELEASE_DIR points at a release build.
"""

import gzip
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from genetics_mcp_server.sdk import _structure_contract as contract
from genetics_mcp_server.sdk import structure as st
from genetics_mcp_server.sdk.errors import GeneticsUsageError

FIXTURES = Path(__file__).parent / "fixtures" / "structure"
META = json.loads((FIXTURES / "expected.json").read_text())
ACCESSIONS = list(META["proteins"])
SETTINGS = [(radius, cutoff or None) for radius, cutoff in META["settings"]]


def _model_rows() -> dict[str, list[dict]]:
    lines = (FIXTURES / "models.tsv").read_text().splitlines()
    header = lines[0].lstrip("#").split("\t")
    by_acc: dict[str, list[dict]] = {}
    for line in lines[1:]:
        row = dict(zip(header, line.split("\t")))
        by_acc.setdefault(row["accession"], []).append(row)
    return by_acc


ROWS = _model_rows()
MULTI = [a for a in ACCESSIONS if len(ROWS[a]) > 1]


@pytest.fixture(scope="module")
def expected():
    return np.load(FIXTURES / "expected.npz")


@pytest.fixture(scope="module")
def pae_rows():
    return np.load(FIXTURES / "pae_rows.npz")


def _structure(mid: str) -> bytes:
    return (FIXTURES / "structures" / f"{mid}.pdb.gz").read_bytes()


def _load(acc: str, pae_rows, release_dir: Path | None = None) -> tuple[st.Protein, bool]:
    """the protein, and whether every one of its PAE matrices is whole."""
    rows = ROWS[acc]
    pae, whole = {}, True
    for r in rows:
        mid = r["model_file_id"]
        if r["has_pae"] != "true":
            continue
        committed = FIXTURES / "pae" / f"{mid}.u8.gz"
        if committed.exists():
            pae[mid] = gzip.decompress(committed.read_bytes())
        elif release_dir is not None:
            pae[mid] = (release_dir / r["pae_path"]).read_bytes()
        else:
            whole = False
            pae[mid] = {
                int(p): row.tobytes()
                for p, row in zip(pae_rows[f"{mid}|file_positions"], pae_rows[f"{mid}|rows"])
            }
    return st.protein(
        rows, {r["model_file_id"]: _structure(r["model_file_id"]) for r in rows}, pae or None
    ), whole


@pytest.fixture(scope="module")
def proteins(pae_rows):
    cache: dict[str, tuple[st.Protein, bool]] = {}

    def get(acc: str) -> tuple[st.Protein, bool]:
        if acc not in cache:
            cache[acc] = _load(acc, pae_rows)
        return cache[acc]

    return get


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _adjacency_sha(adj: np.ndarray) -> str:
    return _sha(np.packbits(adj, axis=None).tobytes())


# --------------------------------------------------------------------------- the fixture set


def test_the_fixture_set_covers_what_the_port_must_handle():
    sources = {a: META["proteins"][a]["source"] for a in ACCESSIONS}
    assert len(ACCESSIONS) >= 50
    assert len(MULTI) >= 5
    assert {sources[a] for a in MULTI} == {"AF2DB", "AF3"}
    assert any(not META["proteins"][a]["has_pae"] for a in ACCESSIONS)
    assert any(META["proteins"][a]["has_pae"] for a in MULTI)


# --------------------------------------------------------------------------- the contract mirror


class TestContractMirror:
    def test_pins(self):
        assert contract.CONTRACT_VERSION == 1
        assert (contract.PAE_SCALE, contract.PAE_NO_VALUE, contract.PAE_MAX_CODE) == (8, 255, 254)
        assert contract.decode_pae(120) == 15.0
        assert contract.decode_pae(255) is None
        windows = [(1, 1400), (201, 1600), (401, 1800)]
        assert contract.owning_window(windows) == [(1, 800), (801, 1000), (1001, 1800)]
        assert contract.pair_window(windows, 10, 20) == 0
        assert contract.pair_window(windows, 700, 900) == 2
        assert contract.pair_window(windows, 250, 1500) == 1
        assert contract.pair_window(windows, 100, 1700) is None

    def test_the_module_reads_only_columns_the_model_table_has(self):
        assert set(st._MODEL_ROW_FIELDS) <= {c for c, _ in contract.MODEL_COLUMNS}

    def test_matches_the_munge_contract_when_a_checkout_is_found(self):
        candidates = [os.environ.get("PROTEIN_STRUCTURE_MUNGE_CONTRACT", "")] + [
            str(p / "genetics-results-munge" / "scripts" / "protein_structure_contract.py")
            for p in Path(__file__).resolve().parents
        ]
        found = next((c for c in candidates if c and Path(c).is_file()), None)
        if found is None:
            pytest.skip("no genetics-results-munge checkout; set PROTEIN_STRUCTURE_MUNGE_CONTRACT")
        spec = importlib.util.spec_from_file_location("_munge_protein_structure_contract", found)
        munge = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(munge)
        for name in (
            "CONTRACT_VERSION",
            "MODEL_COLUMNS",
            "PAE_SCALE",
            "PAE_NO_VALUE",
            "PAE_MAX_CODE",
        ):
            assert getattr(contract, name) == getattr(munge, name), name
        assert all(contract.decode_pae(c) == munge.decode_pae(c) for c in range(256))
        rng = np.random.default_rng(0)
        tilings = [
            [(int(r["protein_start"]), int(r["protein_end"])) for r in ROWS[a]] for a in MULTI
        ]
        tilings.append([(1, 10), (4, 12), (9, 20), (15, 23)])
        for windows in tilings:
            windows = sorted(windows)
            assert contract.owning_window(windows) == munge.owning_window(windows)
            n = max(e for _, e in windows)
            for a, b in rng.integers(1, n + 1, size=(300, 2)).tolist():
                assert contract.pair_window(windows, a, b) == munge.pair_window(windows, a, b)
        with pytest.raises(ValueError):
            contract.owning_window([(5, 9), (1, 6)])

    @pytest.mark.parametrize("acc", MULTI)
    def test_the_vectorised_pair_rule_is_the_contracts(self, acc, proteins):
        p, _ = proteins(acc)
        rng = np.random.default_rng(1)
        edges = [q for (s, e), (lo, hi) in zip(p.windows, p.owned) for q in (s, e, lo, hi)]
        positions = sorted(set(edges) | set(rng.integers(1, p.n_residues + 1, 40).tolist()))
        a, b = (np.array(x) for x in zip(*itertools.product(positions, positions)))
        want = [contract.pair_window(p.windows, i, j) for i, j in zip(a.tolist(), b.tolist())]
        assert p._pair_windows(a, b).tolist() == [-1 if w is None else w for w in want]

    @pytest.mark.parametrize("acc", MULTI)
    def test_the_release_rows_state_the_owned_ranges_the_rule_gives(self, acc, proteins):
        p, _ = proteins(acc)
        assert p.owned == [(int(r["owned_start"]), int(r["owned_end"])) for r in ROWS[acc]]


# --------------------------------------------------------------------------- parsing


class TestParseModel:
    @pytest.mark.parametrize("acc", ACCESSIONS)
    def test_residues_coordinates_and_plddt_are_what_the_labs_parser_reads(self, acc):
        for r in ROWS[acc]:
            mid = r["model_file_id"]
            df = st.parse_model(_structure(mid))
            want = META["proteins"][acc]["files"][mid]
            assert df["file_position"].to_list() == list(range(1, int(r["n_residues"]) + 1))
            assert _sha("".join(df["residue"].to_list()).encode()) == want["sequence_sha256"]
            ca = df.select("ca_x", "ca_y", "ca_z").to_numpy()
            assert ca.dtype == np.float32
            assert _sha(ca.astype("<f4").tobytes()) == want["ca_float32_sha256"]
            assert (
                _sha(df["plddt"].to_numpy().astype("<f8").tobytes()) == want["plddt_float64_sha256"]
            )

    def test_a_whole_model_file_parses_to_the_same_table_as_its_ca_records(self):
        full = META["full_model_files"]
        assert {f["source"] for f in full} == {"AF2DB", "AF3"}
        for f in full:
            mid = f["model_file_id"]
            raw = (FIXTURES / "structures_full" / f"{mid}.pdb.gz").read_bytes()
            want = st.parse_model(_structure(mid))
            assert st.parse_model(raw).equals(want)
            assert st.parse_model(gzip.decompress(raw)).equals(want)
            assert st.parse_model(gzip.decompress(raw).decode()).equals(want)

    def test_a_file_whose_numbering_is_not_1_to_n_is_refused(self):
        text = gzip.decompress(_structure(ROWS[ACCESSIONS[0]][0]["model_file_id"])).decode()
        lines = text.splitlines()
        with pytest.raises(GeneticsUsageError, match="numbered 1..n"):
            st.parse_model("\n".join(lines[:5] + lines[6:]))
        with pytest.raises(GeneticsUsageError, match="numbered 1..n"):
            st.parse_model("\n".join(lines[:5] + [lines[5][:21] + "B" + lines[5][22:]] + lines[6:]))
        with pytest.raises(GeneticsUsageError, match="no CA atoms"):
            st.parse_model("HEADER\nEND\n")

    def test_a_damaged_atom_record_is_a_usage_error(self):
        lines = _toy_pdb(0.0).splitlines()
        for damaged in (lines[2][:50], lines[2][:30] + "   x.xxx" + lines[2][38:]):
            with pytest.raises(GeneticsUsageError, match="line 3 .* not a complete PDB ATOM"):
                st.parse_model("\n".join(lines[:2] + [damaged] + lines[3:]))

    def test_alternate_locations_and_several_models_are_refused(self):
        text = _toy_pdb(0.0)
        lines = text.splitlines()
        with pytest.raises(GeneticsUsageError, match="alternate location"):
            st.parse_model("\n".join(lines[:2] + [lines[2][:16] + "A" + lines[2][17:]] + lines[3:]))
        one = "MODEL        1\n" + text + "ENDMDL\n"
        assert st.parse_model(one).equals(st.parse_model(text))
        with pytest.raises(GeneticsUsageError, match="more than one MODEL"):
            st.parse_model(one + "MODEL        2\n" + text + "ENDMDL\n")


# --------------------------------------------------------------------------- one file


class TestOneFile:
    ACC = ACCESSIONS[0]

    def test_distance_matrix_and_single_distances_agree(self):
        data = _structure(ROWS[self.ACC][0]["model_file_id"])
        model = st.parse_model(data)
        d = st.distance_matrix(model)
        assert d.dtype == np.float32 and d.shape == (model.height, model.height)
        assert (np.diagonal(d) == 0).all() and np.array_equal(d, d.T)
        assert st.ca_distance(model, 3, 40) == st.ca_distance(data, 40, 3) == float(d[2, 39])
        xyz = model.select("ca_x", "ca_y", "ca_z").to_numpy().astype(np.float64)
        assert d[2, 39] == pytest.approx(np.linalg.norm(xyz[2] - xyz[39]), rel=1e-6)
        with pytest.raises(GeneticsUsageError, match="outside 1"):
            st.ca_distance(model, 0, 5)

    def test_pae_lookup_decodes_the_release_encoding(self):
        codes = np.array([[0, 119, 255], [119, 8, 254], [255, 254, 4]], dtype=np.uint8)
        assert st.pae_value(codes.tobytes(), 3, 1, 2) == 119 / 8
        assert st.pae_value(codes, 3, 2, 3) == 31.75
        assert st.pae_value(codes, 3, 1, 3) is None
        # one row is enough for any pair that row is part of, in either order
        rows = {2: codes[1].tobytes()}
        assert st.pae_value(rows, 3, 2, 1) == st.pae_value(rows, 3, 1, 2) == 119 / 8
        with pytest.raises(GeneticsUsageError, match="neither PAE row"):
            st.pae_value(rows, 3, 1, 3)
        with pytest.raises(GeneticsUsageError, match="holds 9 values"):
            st.pae_value(codes.tobytes(), 4, 1, 2)
        with pytest.raises(GeneticsUsageError, match="uint8"):
            st.pae_value(codes.astype(np.float64), 3, 1, 2)


# --------------------------------------------------------------------------- against the lab


@pytest.mark.parametrize("acc", ACCESSIONS)
def test_distances_and_pae_at_sampled_pairs_are_the_labs(acc, proteins, expected):
    p, whole = proteins(acc)
    info = META["proteins"][acc]
    pairs = expected[f"{acc}|pairs"]
    want_d = dict(zip(map(tuple, pairs.tolist()), expected[f"{acc}|distance"]))
    want_code = (
        dict(zip(map(tuple, pairs.tolist()), expected[f"{acc}|pae_code"].tolist()))
        if info["has_pae"]
        else {}
    )
    want_raw = (
        dict(zip(map(tuple, pairs.tolist()), expected[f"{acc}|pae"].tolist()))
        if info["has_pae"]
        else {}
    )
    checked = 0
    for a in sorted({a for a, _ in want_d}):
        partners = [b for (x, b) in want_d if x == a and b != a]
        if whole:
            got = st.distances(p, [a] + partners).filter(pl.col("position_a") == a)
        else:
            # only the centres' PAE rows are committed, and partner-partner pairs have neither row
            got = pl.concat([st.distances(p, [a, b]) for b in partners])
        assert got["position_b"].to_list() == partners
        for row in got.iter_rows(named=True):
            b = row["position_b"]
            lab = want_d[(a, b)]
            window = contract.pair_window(p.windows, a, b)
            if window is None:
                # the lab leaves a pair no file contains at infinity
                assert np.isinf(lab)
                assert row["distance"] is None and row["pae"] is None and row["min_plddt"] is None
                assert row["model_file_id"] is None and row["file_position_a"] is None
                assert "no single model file" in row["reason"]
            else:
                assert abs(np.float32(row["distance"]) - lab) <= np.spacing(lab)
                assert row["reason"] is None
                assert row["model_file_id"] == p.model_file_ids[window]
                assert row["file_position_a"] == a - p.windows[window][0] + 1
                assert row["file_position_b"] == b - p.windows[window][0] + 1
                if info["has_pae"]:
                    assert row["pae"] == want_code[(a, b)] / 8
                    # the top code also stands for every larger value
                    if want_code[(a, b)] != contract.PAE_MAX_CODE:
                        assert abs(row["pae"] - want_raw[(a, b)]) <= 1 / 16 + 1e-6
                else:
                    assert row["pae"] is None
            owner = [k for k, (lo, hi) in enumerate(p.owned) for q in (a, b) if lo <= q <= hi]
            assert row["same_window"] == (owner[0] == owner[1])
            checked += 1
    assert checked == sum(1 for a, b in want_d if a != b)


@pytest.mark.parametrize("acc", ACCESSIONS)
def test_sampled_neighbourhoods_are_the_labs_adjacency_rows(acc, proteins, expected):
    p, _ = proteins(acc)
    info = META["proteins"][acc]
    centres = expected[f"{acc}|centres"].tolist()
    for i, (radius, cutoff) in enumerate(SETTINGS):
        rows = np.unpackbits(expected[f"{acc}|s{i}|centre_rows"], axis=1)[:, : info["n_residues"]]
        for centre, bits in zip(centres, rows):
            res = st.neighbourhood(p, centre, radius, cutoff)
            nb = res["neighbours"]
            assert nb["protein_position"].to_list() == (np.flatnonzero(bits) + 1).tolist()
            starts = np.array(
                [p.windows[p.model_file_ids.index(m)][0] for m in nb["model_file_id"].to_list()]
            )
            assert np.array_equal(
                nb["file_position"].to_numpy(), nb["protein_position"].to_numpy() - starts + 1
            )
            assert res["pae_filter_applied"] == (info["has_pae"] and cutoff is not None)
            if cutoff is not None and info["has_pae"]:
                assert nb["pae"].max() < cutoff
            assert nb["distance"].max() < radius


@pytest.mark.parametrize("acc", ACCESSIONS)
def test_whole_protein_adjacency_is_the_labs(acc, proteins, expected):
    p, whole = proteins(acc)
    info = META["proteins"][acc]
    for i, (radius, cutoff) in enumerate(SETTINGS):
        if info["has_pae"] and cutoff is not None and not whole:
            with pytest.raises(GeneticsUsageError, match="whole PAE matrix"):
                st.adjacency(p, radius, cutoff)
            continue
        res = st.adjacency(p, radius, cutoff)
        adj = res["adjacency"]
        assert adj.dtype == bool and adj.shape == (info["n_residues"],) * 2
        assert np.array_equal(adj.sum(axis=1), expected[f"{acc}|s{i}|row_sums"])
        assert _adjacency_sha(adj) == info["adjacency_sha256"][i]
        assert res["pae_filter_applied"] == (info["has_pae"] and cutoff is not None)
        assert np.diagonal(adj).all()


def test_whole_protein_adjacency_with_pae_for_the_large_proteins(pae_rows, expected):
    release = os.environ.get("PROTEIN_STRUCTURE_RELEASE_DIR")
    if not release:
        pytest.skip(
            "set PROTEIN_STRUCTURE_RELEASE_DIR to a release build to read whole PAE objects"
        )
    large = [
        a for a in ACCESSIONS if any(f"{r['model_file_id']}|rows" in pae_rows for r in ROWS[a])
    ]
    assert large
    for acc in large:
        p, whole = _load(acc, pae_rows, Path(release))
        assert whole
        for i, (radius, cutoff) in enumerate(SETTINGS):
            adj = st.adjacency(p, radius, cutoff)["adjacency"]
            assert np.array_equal(adj.sum(axis=1), expected[f"{acc}|s{i}|row_sums"])
            assert _adjacency_sha(adj) == META["proteins"][acc]["adjacency_sha256"][i]


# --------------------------------------------------------------------------- multi-file rules


def _toy_pdb(offset: float, n: int = 6) -> str:
    return "".join(
        f"ATOM  {i:5d}  CA  GLY A{i:4d}    {offset + 3.8 * i:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00{50.0 + i:6.2f}           C\n"
        for i in range(1, n + 1)
    )


def _toy_rows(windows, pae: bool = False) -> list[dict]:
    return [
        dict(
            model_file_id=f"T-F{k}",
            accession="T",
            protein_start=s,
            protein_end=e,
            n_residues=e - s + 1,
            n_windows=len(windows),
            has_pae=pae,
        )
        for k, (s, e) in enumerate(windows, start=1)
    ]


def _toy(pae: bool = False, f1_codes: np.ndarray | None = None) -> st.Protein:
    """two overlapping 6-residue windows on a 9-residue protein, the second frame shifted
    so far that any distance taken across files would be unmistakable."""
    codes = np.full((6, 6), 40, dtype=np.uint8)
    return st.protein(
        _toy_rows([(1, 6), (4, 9)], pae),
        {"T-F1": _toy_pdb(0.0), "T-F2": _toy_pdb(1000.0)},
        {"T-F1": codes if f1_codes is None else f1_codes, "T-F2": codes + 100} if pae else None,
    )


class TestMultiFile:
    def test_a_pair_no_file_contains_has_no_distance_and_says_why(self):
        p = _toy()
        assert p.owned == [(1, 5), (6, 9)]
        df = st.distances(p, [1, 5, 6, 9])
        by_pair = {(r["position_a"], r["position_b"]): r for r in df.iter_rows(named=True)}
        assert by_pair[(1, 5)]["distance"] == pytest.approx(15.2, rel=1e-6)
        assert by_pair[(1, 5)]["same_window"] and by_pair[(1, 5)]["model_file_id"] == "T-F1"
        # owned by different windows, measured in the last file containing both
        assert by_pair[(5, 6)]["model_file_id"] == "T-F2" and not by_pair[(5, 6)]["same_window"]
        # single precision resolves a coordinate near 1000 to about 6e-5
        assert by_pair[(5, 6)]["distance"] == pytest.approx(3.8, abs=1e-3)
        assert by_pair[(5, 6)]["min_plddt"] == 52.0
        assert by_pair[(1, 6)]["model_file_id"] == "T-F1" and not by_pair[(1, 6)]["same_window"]
        assert by_pair[(5, 9)]["model_file_id"] == "T-F2"
        row = by_pair[(1, 9)]
        assert row["distance"] is None
        assert row["reason"] == "no single model file contains both residues"
        assert not row["same_window"]
        assert df["distance"].drop_nulls().max() < 100

    def test_adjacency_and_neighbourhood_never_join_files(self):
        p = _toy()
        adj = st.adjacency(p, radius=5000.0, pae_cutoff=None)["adjacency"]
        want = np.array(
            [
                [contract.pair_window(p.windows, a, b) is not None for b in range(1, 10)]
                for a in range(1, 10)
            ]
        )
        assert np.array_equal(adj, want)
        nb = st.neighbourhood(p, 5, radius=5000.0, pae_cutoff=None)["neighbours"]
        assert not adj[:3, 6:].any() and adj[3:6].all()
        assert nb["protein_position"].to_list() == list(range(1, 10))
        assert nb["model_file_id"].to_list() == ["T-F1"] * 5 + ["T-F2"] * 4
        assert nb["file_position"].to_list() == [1, 2, 3, 4, 5, 3, 4, 5, 6]
        assert nb["distance"].max() < 100

    def test_pae_comes_from_the_file_the_distance_comes_from(self):
        p = _toy(pae=True)
        df = st.distances(p, [4, 5, 6])
        assert dict(zip(zip(df["position_a"], df["position_b"]), df["pae"])) == {
            (4, 5): 5.0,
            (4, 6): 17.5,
            (5, 6): 17.5,
        }
        assert st.distances(p, [4, 5, 6], pae=False)["pae"].null_count() == 3
        nb = st.neighbourhood(p, 5, radius=10.0, pae_cutoff=15)["neighbours"]
        assert nb["protein_position"].to_list() == [3, 4, 5]

    def test_addresses_may_be_given_as_model_file_and_file_position(self):
        p = _toy()
        by_address = st.distances(p, [("T-F2", 1), ("T-F1", 5), 4])
        assert by_address["position_a"].to_list() == [4] and by_address["position_b"].to_list() == [
            5
        ]
        with pytest.raises(GeneticsUsageError, match="not a model file"):
            st.distances(p, [("X", 1), 2])
        with pytest.raises(GeneticsUsageError, match="outside 1..9"):
            st.distances(p, [1, 10])
        with pytest.raises(GeneticsUsageError, match="at least two"):
            st.distances(p, [3, 3])

    def test_only_the_files_a_question_needs_have_to_be_given(self):
        whole = _toy()
        rows = _toy_rows(whole.windows)
        partial = st.protein(rows, {"T-F1": whole._structures["T-F1"]})
        assert (
            st.distances(partial, [1, 2])["distance"].to_list()
            == st.distances(whole, [1, 2])["distance"].to_list()
        )
        with pytest.raises(GeneticsUsageError, match="T-F2 is needed"):
            st.distances(partial, [5, 6])
        with pytest.raises(GeneticsUsageError, match="every\\s+row is needed"):
            st.protein(rows[:1], {})


class TestArguments:
    def test_model_rows_that_break_the_ownership_rule_are_refused(self):
        rows = [dict(r) for r in ROWS[MULTI[0]]]
        rows[0]["owned_end"] = str(int(rows[0]["owned_end"]) + 1)
        with pytest.raises(GeneticsUsageError, match="release contract"):
            st.protein(rows, {})

    def test_pae_cutoff_must_be_a_whole_number_in_range(self, proteins):
        p, _ = proteins(ACCESSIONS[0])
        for bad in (14.5, 0, 32, True, float("nan"), float("inf"), [15], "15"):
            with pytest.raises(GeneticsUsageError, match="pae_cutoff"):
                st.adjacency(p, 15, bad)
        assert st.adjacency(p, 15, 15.0)["pae_filter_applied"]
        with pytest.raises(GeneticsUsageError, match="radius"):
            st.adjacency(p, 0)

    def test_a_protein_with_pae_is_not_filtered_silently_when_pae_is_missing(self):
        acc = next(a for a in ACCESSIONS if META["proteins"][a]["has_pae"] and len(ROWS[a]) == 1)
        row = ROWS[acc][0]
        p = st.protein([row], {row["model_file_id"]: _structure(row["model_file_id"])})
        with pytest.raises(GeneticsUsageError, match="PAE of model file"):
            st.adjacency(p)
        with pytest.raises(GeneticsUsageError, match="PAE of model file"):
            st.distances(p, [1, 2])
        assert st.distances(p, [1, 2], pae=False)["pae"].to_list() == [None]
        assert not st.adjacency(p, pae_cutoff=None)["pae_filter_applied"]

    def test_pae_given_for_a_protein_without_it_is_refused(self):
        acc = next(a for a in ACCESSIONS if not META["proteins"][a]["has_pae"])
        with pytest.raises(GeneticsUsageError, match="has no PAE"):
            st.protein(ROWS[acc], {}, {ROWS[acc][0]["model_file_id"]: b""})


class TestModelRows:
    WINDOWS = [(1, 10), (6, 15), (11, 20)]

    def _files(self):
        return {f"T-F{k}": _toy_pdb(100.0 * k, 10) for k in (1, 2, 3)}

    def test_a_partial_table_cannot_stand_in_for_the_whole_one(self):
        rows = _toy_rows(self.WINDOWS)
        whole = st.protein(rows, self._files())
        assert whole.n_residues == 20
        assert st.distances(whole, [7, 9])["model_file_id"].to_list() == ["T-F2"]
        with pytest.raises(GeneticsUsageError, match="every\\s+row is needed"):
            st.protein(rows[:1], self._files())
        for broken in (
            {k: v for k, v in rows[0].items() if k != "n_windows"},
            dict(rows[0], n_windows=None),
        ):
            with pytest.raises(GeneticsUsageError, match="n_windows"):
                st.protein([broken], {"T-F1": _toy_pdb(0.0, 10)})

    @pytest.mark.parametrize("value", [None, "", "yes", 1, float("nan")])
    def test_has_pae_must_be_stated(self, value):
        rows = [dict(r, has_pae=value) for r in _toy_rows(self.WINDOWS)]
        with pytest.raises(GeneticsUsageError, match="has_pae must be true or false"):
            st.protein(rows, self._files())

    def test_a_missing_has_pae_column_is_refused(self):
        rows = [{k: v for k, v in r.items() if k != "has_pae"} for r in _toy_rows(self.WINDOWS)]
        with pytest.raises(GeneticsUsageError, match="has_pae"):
            st.protein(rows, self._files())

    def test_repeated_model_file_ids_are_refused(self):
        rows = _toy_rows(self.WINDOWS)
        rows[1]["model_file_id"] = rows[0]["model_file_id"]
        with pytest.raises(GeneticsUsageError, match="same model_file_id"):
            st.protein(rows, {})

    @pytest.mark.parametrize(
        "present,absent", [("owned_start", "owned_end"), ("owned_end", "owned_start")]
    )
    def test_one_owned_bound_without_the_other_is_refused(self, present, absent):
        for missing in ({}, {absent: None}, {absent: ""}):
            rows = [dict(r, **{present: 1}, **missing) for r in _toy_rows(self.WINDOWS)]
            with pytest.raises(GeneticsUsageError, match="without the other"):
                st.protein(rows, {})

    def test_a_non_integer_field_is_a_usage_error(self):
        for column, value in (("protein_start", "x"), ("n_residues", None), ("n_windows", 2.5)):
            rows = [dict(r, **{column: value}) for r in _toy_rows(self.WINDOWS)]
            with pytest.raises(GeneticsUsageError, match=f"{column} must be an integer"):
                st.protein(rows, {})


class TestNoValueCode:
    def _protein(self):
        codes = np.full((6, 6), 40, dtype=np.uint8)
        codes[3, 4] = codes[4, 3] = contract.PAE_NO_VALUE
        return _toy(pae=True, f1_codes=codes), codes

    def test_the_reserved_code_is_null_everywhere(self):
        p, codes = self._protein()
        assert st.pae_value(codes, 6, 4, 5) is None
        assert st.pae_value({5: codes[4].tobytes()}, 6, 4, 5) is None
        row = st.distances(p, [4, 5]).row(0, named=True)
        assert row["pae"] is None and row["distance"] is not None and row["model_file_id"] == "T-F1"
        nb = st.neighbourhood(p, 5, radius=10.0, pae_cutoff=None)["neighbours"]
        assert dict(zip(nb["protein_position"], nb["pae"])) == {
            3: 5.0,
            4: None,
            5: 5.0,
            6: 17.5,
            7: 17.5,
        }

    def test_the_reserved_code_passes_no_cutoff(self):
        p, _ = self._protein()
        nb = st.neighbourhood(p, 5, radius=10.0, pae_cutoff=31)["neighbours"]
        assert nb["protein_position"].to_list() == [3, 5, 6, 7]
        adj = st.adjacency(p, radius=10.0, pae_cutoff=31)["adjacency"]
        assert not adj[3, 4] and not adj[4, 3] and adj[2, 4]


class TestArithmetic:
    """the two places where a plausible rewrite gives a different last bit or a different pair."""

    def _pair(self, acc, expected):
        pairs, d = expected[f"{acc}|pairs"].tolist(), expected[f"{acc}|distance"]
        return next(
            (a, b, d[i])
            for i, (a, b) in enumerate(pairs)
            if a != b and np.isfinite(d[i]) and d[i] > 4
        )

    def test_a_composed_protein_compares_the_radius_in_double_precision(self, proteins, expected):
        acc = next(a for a in MULTI if not META["proteins"][a]["has_pae"])
        p, _ = proteins(acc)
        a, b, d = self._pair(acc, expected)
        # the next double above a real distance: above it in double, equal to it in single
        radius = float(np.nextafter(np.float64(d), np.inf))
        assert np.float32(radius) == d and radius > float(d)
        assert b in st.neighbourhood(p, a, radius, None)["neighbours"]["protein_position"].to_list()
        assert st.adjacency(p, radius, None)["adjacency"][a - 1, b - 1]
        assert (
            b
            not in st.neighbourhood(p, a, float(d), None)["neighbours"][
                "protein_position"
            ].to_list()
        )

    def test_a_one_file_protein_compares_the_radius_in_single_precision(self, proteins, expected):
        acc = next(a for a in ACCESSIONS if len(ROWS[a]) == 1)
        p, _ = proteins(acc)
        a, b, d = self._pair(acc, expected)
        radius = float(np.nextafter(np.float64(d), np.inf))
        assert (
            b
            not in st.neighbourhood(p, a, radius, None)["neighbours"]["protein_position"].to_list()
        )
        assert not st.adjacency(p, radius, None)["adjacency"][a - 1, b - 1]
        above = float(np.nextafter(d, np.float32(np.inf)))
        assert b in st.neighbourhood(p, a, above, None)["neighbours"]["protein_position"].to_list()

    def test_the_square_is_rounded_to_single_before_the_root(self, proteins, expected):
        discriminating = 0
        for acc in [a for a in ACCESSIONS if len(ROWS[a]) == 1][:10]:
            p, _ = proteins(acc)
            model = st.parse_model(_structure(ROWS[acc][0]["model_file_id"]))
            xyz = model.select("ca_x", "ca_y", "ca_z").to_numpy().astype(np.float64)
            matrix = st.distance_matrix(model)
            pairs, lab = expected[f"{acc}|pairs"], expected[f"{acc}|distance"]
            squared = ((xyz[pairs[:, 0] - 1] - xyz[pairs[:, 1] - 1]) ** 2).sum(axis=1)
            root_then_round = np.sqrt(squared).astype(np.float32)
            for i in np.flatnonzero(root_then_round != lab).tolist():
                a, b = pairs[i].tolist()
                assert np.float32(st.ca_distance(model, a, b)) == lab[i]
                assert matrix[a - 1, b - 1] == lab[i]
                assert np.float32(st.distances(p, [a, b])["distance"][0]) == lab[i]
                discriminating += 1
        # without such pairs this test could not tell the two orders apart
        assert discriminating >= 10


def test_the_module_resolves_lazily_from_the_package():
    import genetics_mcp_server.sdk as sdk

    assert sdk.structure is st
    assert "structure" in sdk._LAZY_MODULES
