"""Protein structure geometry: Cα distances, PAE and 3D neighbourhoods on predicted models.

    import genetics
    p = genetics.structure.protein(model_rows, {"AF-P04637-F1-model_v6": pdb_bytes}, pae)
    genetics.structure.distances(p, [175, 248, 273])        # one row per pair of residues
    genetics.structure.neighbourhood(p, 175)["neighbours"]  # residues within 15 Å, PAE < 15
    genetics.structure.adjacency(p)["adjacency"]            # the whole protein, n x n bool
    genetics.structure.view(p, [{"position": 175, "label": "p.R175H", "role": "case"}])

A port of the geometry in the Finucane lab's 3D rare-variant scan (`utils.py`:
`get_pairwise_distances`, `get_distance_matrix_structure`, `get_pae_matrix_structure`,
`threshold_adjacency_matrix`), held to that code's outputs on the lab's reference bundle. The
lab's names are given beside each function. Everything here computes on inputs already in
memory: the text or bytes of model files, PAE in the release's uint8 encoding, and the rows
of the release's model table.

WHAT THE NUMBERS ARE. A distance is between the Cα atoms of two residues in one predicted
model, in ångström. PAE (predicted aligned error) is the model's own estimate of how well it
knows the relative placement of two residues: a small distance with a large PAE is two
confident pieces placed against each other at a guess. pLDDT is the per-residue confidence,
the model file's B-factor column. Two residues are neighbours when their distance is below
the radius AND, where the protein has PAE, their PAE is below the cutoff; both comparisons
are strict, and a residue is its own neighbour.

LONG PROTEINS COME AS SEVERAL FILES. A protein too long for one prediction is split into
overlapping windows, each predicted on its own and each in its own coordinate frame, so a
distance exists only between residues that some single file contains. Which file gives a
pair is fixed by the release contract (`pair_window`): the window that owns both residues,
otherwise the last window in start order containing both. A pair no file contains has no
distance — it is returned as null with a reason, never as a number worked out across files.
Positions passed to and returned by the protein-level functions are protein positions; a
file position is the residue number inside one model file, and protein position =
protein_start + file position − 1.

DISTANCES ARE SINGLE PRECISION, as in the lab's code, so that a neighbourhood computed here
contains exactly the residues the lab's contains; the rounding is far below the three
decimals the coordinates carry. PAE comes back as the release stores it, in steps of 1/8:
within 1/16 of the model's value, except that the top code, 31.75, also stands for every
larger value. The models give PAE in tenths, and for values on tenths a whole-number cutoff
selects from the stored codes exactly the pairs it selects from the model's own values; a
fractional cutoff would not, which is why `pae_cutoff` must be a whole number.

SHOWING IT. `view` writes a `.structure-view.json` artifact, which the chat draws as an
interactive 3D model with the given residues marked; `pml` writes the same marking as a PyMOL
script. Both show ONE model file, so for a long protein they say which residues that file
does not contain rather than placing them in another file's coordinates.

A PROTEIN WITHOUT PAE gets distance-only neighbourhoods, as in the lab's code, and every
result says so (`pae_filter_applied`). Treat those with care in a multi-file protein: windows
are predicted independently and low-confidence stretches pack into dense masses.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import polars as pl

from genetics_mcp_server.sdk import _structure_contract as contract
from genetics_mcp_server.sdk.errors import GeneticsUsageError

__all__ = [
    "parse_model",
    "ca_distance",
    "distance_matrix",
    "pae_value",
    "protein",
    "Protein",
    "adjacency",
    "neighbourhood",
    "distances",
    "view",
    "pml",
]

_DEFAULT_RADIUS = 15.0
_DEFAULT_PAE_CUTOFF = 15

# the lab's PAE values stop below 32, so a cutoff past this filters nothing, and the
# reserved no-value code would start passing it
_MAX_PAE_CUTOFF = contract.PAE_NO_VALUE // contract.PAE_SCALE

_NO_SHARED_MODEL = "no single model file contains both residues"

# float64 elements one block of the distance computation may hold; bounds the temporary
# to tens of MiB whatever the model's length
_BLOCK_ELEMENTS = 3 * 2**20

_ONE_LETTER = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "SEC": "U",
    "PYL": "O",
}  # fmt: skip

_MODEL_ROW_FIELDS = (
    "model_file_id",
    "accession",
    "protein_start",
    "protein_end",
    "n_residues",
    "n_windows",
    "has_pae",
)


# --------------------------------------------------------------------------- one model file


class _Model:
    __slots__ = ("ca", "plddt", "residue")

    def __init__(self, ca: np.ndarray, plddt: np.ndarray, residue: np.ndarray):
        self.ca, self.plddt, self.residue = ca, plddt, residue

    @property
    def n(self) -> int:
        return self.ca.shape[0]


def _as_text(data: str | bytes) -> str:
    if isinstance(data, str):
        return data
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise GeneticsUsageError(
            "a model file is given as its text or its bytes (optionally gzipped)"
        )
    raw = bytes(data)
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return raw.decode("ascii", errors="replace")


def _parse(data: str | bytes) -> _Model:
    numbers, xyz, bfac, letters, chains = [], [], [], [], set()
    n_models = 0
    for lineno, line in enumerate(_as_text(data).splitlines(), start=1):
        if line.startswith("MODEL "):
            n_models += 1
            if n_models > 1:
                # the lab's reader concatenates the residues of every model; refusing is the
                # only reading that cannot disagree with it
                raise GeneticsUsageError(
                    "the model file holds more than one MODEL; a predicted model has one"
                )
        if not line.startswith("ATOM"):
            continue
        if line[16:17].strip():
            # the lab's reader keeps the highest-occupancy location, a choice this parser
            # does not reproduce
            raise GeneticsUsageError(
                f"line {lineno} of the model file has an alternate location; a predicted "
                "model has one position per atom"
            )
        if line[12:16] != " CA ":
            continue
        try:
            number = int(line[22:26])
            coords = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            plddt = float(line[60:66])
        except ValueError as exc:
            raise GeneticsUsageError(
                f"line {lineno} of the model file is not a complete PDB ATOM record"
            ) from exc
        chains.add(line[21])
        numbers.append(number)
        xyz.append(coords)
        bfac.append(plddt)
        letters.append(_ONE_LETTER.get(line[17:20], "X"))
    if not numbers:
        raise GeneticsUsageError("no CA atoms found: this is not a PDB-format model file")
    if len(chains) != 1 or numbers != list(range(1, len(numbers) + 1)):
        # file position IS the residue number in the release; a file that breaks that would
        # be addressed wrongly by everything downstream
        raise GeneticsUsageError(
            "the model file is not one chain numbered 1..n without gaps, so its residue "
            "numbers cannot be used as file positions"
        )
    # single precision is how the lab's parser holds coordinates, and the neighbourhoods
    # are defined on distances computed from these values
    return _Model(
        np.array(xyz, dtype=np.float32), np.array(bfac, dtype=np.float64), np.array(letters)
    )


def parse_model(data: str | bytes) -> pl.DataFrame:
    """One model file as a per-residue table.

    `data` is the text or bytes of a `.pdb` model file, gzipped or not. Returns one row per
    residue: `file_position` (the residue number, 1..n), `residue` (one-letter code),
    `ca_x`, `ca_y`, `ca_z` (the Cα atom, in Å) and `plddt` (the B-factor column).
    """
    m = _parse(data)
    return pl.DataFrame(
        {
            "file_position": np.arange(1, m.n + 1, dtype=np.int32),
            "residue": m.residue.tolist(),
            "ca_x": m.ca[:, 0],
            "ca_y": m.ca[:, 1],
            "ca_z": m.ca[:, 2],
            "plddt": m.plddt,
        }
    )


def _model_of(model: pl.DataFrame | str | bytes) -> _Model:
    if not isinstance(model, pl.DataFrame):
        return _parse(model)
    missing = [c for c in ("ca_x", "ca_y", "ca_z") if c not in model.columns]
    if missing:
        raise GeneticsUsageError(f"the model table lacks {missing}; pass what parse_model returned")
    ca = model.select("ca_x", "ca_y", "ca_z").to_numpy().astype(np.float32)
    plddt = model["plddt"].to_numpy() if "plddt" in model.columns else np.full(len(ca), np.nan)
    residue = np.array(
        model["residue"].to_list() if "residue" in model.columns else ["X"] * len(ca)
    )
    return _Model(ca, plddt.astype(np.float64), residue)


def _pair_distances(ca_a: np.ndarray, ca_b: np.ndarray) -> np.ndarray:
    """distance between matching rows of two coordinate arrays, as float32."""
    diff = ca_a.astype(np.float64) - ca_b.astype(np.float64)
    return np.sqrt(np.einsum("ij,ij->i", diff, diff).astype(np.float32))


def _distance_rows(ca: np.ndarray, rows: slice | np.ndarray) -> np.ndarray:
    """distances from the selected residues to every residue of the model, as float32.

    The squared distance is formed in double precision from the single-precision
    coordinates, rounded to single, and only then rooted — the order of the lab's
    computation (scikit-learn's euclidean distances on float32 input), which is what makes
    `distance < radius` pick the same pairs.
    """
    x = ca.astype(np.float64)
    sel = x[rows]
    out = np.empty((sel.shape[0], x.shape[0]), dtype=np.float32)
    step = max(1, _BLOCK_ELEMENTS // (3 * max(1, x.shape[0])))
    for i in range(0, sel.shape[0], step):
        diff = sel[i : i + step, None, :] - x[None, :, :]
        out[i : i + step] = np.einsum("ijk,ijk->ij", diff, diff)
    return np.sqrt(out, out=out)


def _file_index(n: int, position: int, what: str = "file position") -> int:
    if isinstance(position, bool) or not isinstance(position, (int, np.integer)):
        raise GeneticsUsageError(f"{what} must be an integer, got {position!r}")
    if not 1 <= position <= n:
        raise GeneticsUsageError(f"{what} {position} is outside 1..{n}")
    return int(position) - 1


def ca_distance(model: pl.DataFrame | str | bytes, i: int, j: int) -> float:
    """Cα–Cα distance in Å between file positions `i` and `j` of one model file.

    `model` is what `parse_model` returned, or the file itself.
    """
    m = _model_of(model)
    a, b = _file_index(m.n, i), _file_index(m.n, j)
    return float(_pair_distances(m.ca[a : a + 1], m.ca[b : b + 1])[0])


def distance_matrix(model: pl.DataFrame | str | bytes) -> np.ndarray:
    """All Cα–Cα distances of one model file: an (n, n) float32 array indexed by file
    position − 1. The lab's `get_pairwise_distances`."""
    m = _model_of(model)
    return _distance_rows(m.ca, slice(None))


# --------------------------------------------------------------------------- PAE


class _Pae:
    """One model file's PAE codes: the whole n x n matrix, or some of its rows.

    The stored matrix is symmetric, so a pair can be read from either residue's row and a
    residue's whole neighbourhood from its own row alone.
    """

    __slots__ = ("n", "matrix", "rows", "mid")

    def __init__(self, data: Any, n: int, mid: str = "the model file"):
        self.n, self.mid, self.matrix, self.rows = n, mid, None, None
        if isinstance(data, Mapping):
            self.rows = {_file_index(n, p, "PAE row"): self._row(v) for p, v in data.items()}
        else:
            arr = self._codes(data)
            if arr.size != n * n:
                raise GeneticsUsageError(
                    f"PAE of {mid} holds {arr.size} values; a model of {n} residues has {n * n}. "
                    "Pass the whole uint8 matrix, or a dict of {file position: row}"
                )
            self.matrix = arr.reshape(n, n)

    @staticmethod
    def _codes(data: Any) -> np.ndarray:
        if isinstance(data, np.ndarray):
            if data.dtype != np.uint8:
                raise GeneticsUsageError("PAE is the release's uint8 codes, not decoded values")
            return data
        if isinstance(data, (bytes, bytearray, memoryview)):
            return np.frombuffer(data, dtype=np.uint8)
        raise GeneticsUsageError("PAE is given as the bytes of the release's uint8 matrix")

    def _row(self, data: Any) -> np.ndarray:
        arr = self._codes(data).reshape(-1)
        if arr.size != self.n:
            raise GeneticsUsageError(
                f"a PAE row of {self.mid} holds {arr.size} values, not {self.n}"
            )
        return arr

    def whole(self) -> np.ndarray:
        if self.matrix is None:
            raise GeneticsUsageError(
                f"the whole PAE matrix of {self.mid} is needed here and only some rows were given"
            )
        return self.matrix

    def has_row(self, i: int) -> bool:
        return self.matrix is not None or i in self.rows

    def row(self, i: int) -> np.ndarray:
        if self.matrix is not None:
            return self.matrix[i]
        if i not in self.rows:
            raise GeneticsUsageError(f"PAE row {i + 1} of {self.mid} was not given")
        return self.rows[i]

    def codes(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self.matrix is not None:
            return self.matrix[a, b]
        out = np.empty(len(a), dtype=np.uint8)
        for k, (i, j) in enumerate(zip(a.tolist(), b.tolist())):
            if i in self.rows:
                out[k] = self.rows[i][j]
            elif j in self.rows:
                out[k] = self.rows[j][i]
            else:
                raise GeneticsUsageError(
                    f"neither PAE row {i + 1} nor {j + 1} of {self.mid} was given"
                )
        return out


def _decode(codes: np.ndarray) -> np.ndarray:
    """codes to PAE values, the reserved no-value code to NaN."""
    return np.where(codes == contract.PAE_NO_VALUE, np.nan, codes / contract.PAE_SCALE)


def pae_value(pae: Any, n_residues: int, i: int, j: int) -> float | None:
    """PAE in Å between file positions `i` and `j` of one model file, or None for no value.

    `pae` is the release's PAE object for the file — `n_residues`² uint8 codes, row-major,
    as bytes or an array — or a dict of {file position: that row's bytes} holding at least
    one of the two rows. The stored matrix is symmetric (the smaller of the model's two
    directed values, as the lab symmetrises), so the order of `i` and `j` does not matter.
    """
    p = _Pae(pae, n_residues)
    a, b = _file_index(n_residues, i), _file_index(n_residues, j)
    return contract.decode_pae(int(p.codes(np.array([a]), np.array([b]))[0]))


def _pae_threshold(pae_cutoff: int | None) -> int | None:
    """the code a PAE code must be below, or None for no PAE filter."""
    if pae_cutoff is None:
        return None
    numeric = isinstance(pae_cutoff, (int, float, np.integer, np.floating))
    if (
        not numeric
        or isinstance(pae_cutoff, (bool, np.bool_))
        or not math.isfinite(pae_cutoff)
        or float(pae_cutoff) != int(pae_cutoff)
    ):
        # `pae < c` and `code < 8c` agree for whole c provided the PAE values sit on tenths,
        # which the release's encoder enforces; for a fractional c they do not
        raise GeneticsUsageError(
            f"pae_cutoff must be a whole number of Å or None (got {pae_cutoff!r}): the stored "
            "PAE is rounded to 1/8 Å and only whole cutoffs select exactly the lab's pairs"
        )
    if not 1 <= int(pae_cutoff) <= _MAX_PAE_CUTOFF:
        raise GeneticsUsageError(
            f"pae_cutoff must be between 1 and {_MAX_PAE_CUTOFF}, or None for no PAE filter "
            "(the lab's code spells no filter as 0)"
        )
    return int(pae_cutoff) * contract.PAE_SCALE


# --------------------------------------------------------------------------- a protein


def _truthy(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    # a null read as false would switch the PAE filter off without saying so
    raise GeneticsUsageError(f"has_pae must be true or false, got {value!r}")


def _row_int(row: Mapping[str, Any], column: str) -> int:
    value = row[column]
    try:
        if isinstance(value, (bool, float)) and not (
            isinstance(value, float) and value.is_integer()
        ):
            raise ValueError
        return int(value)
    except (TypeError, ValueError):
        raise GeneticsUsageError(
            f"model row {row.get('model_file_id')!r}: {column} must be an integer, got {value!r}"
        ) from None


class Protein:
    """One protein's model files and how they tile it; build with `protein()`.

    `accession`, `n_residues` (the protein's length), `has_pae`, `model_file_ids`,
    `windows` ((protein_start, protein_end) per file) and `owned` (the protein positions
    each file owns) are in start order.
    """

    def __init__(
        self, rows: list[dict], structures: Mapping[str, Any], pae: Mapping[str, Any] | None
    ):
        self.accession: str = rows[0]["accession"]
        self.model_file_ids: list[str] = [r["model_file_id"] for r in rows]
        self.windows: list[tuple[int, int]] = [(r["protein_start"], r["protein_end"]) for r in rows]
        self.owned: list[tuple[int, int]] = contract.owning_window(self.windows)
        self.n_residues: int = max(e for _, e in self.windows)
        self.has_pae: bool = rows[0]["has_pae"]
        self._structures = dict(structures)
        self._pae_data = dict(pae or {})
        self._models: dict[int, _Model] = {}
        self._paes: dict[int, _Pae] = {}
        self._owned_hi = np.array([hi for _, hi in self.owned])

    def __repr__(self) -> str:
        return (
            f"Protein({self.accession}, {self.n_residues} residues, "
            f"{len(self.windows)} model file(s), has_pae={self.has_pae})"
        )

    def _model(self, k: int) -> _Model:
        if k not in self._models:
            mid = self.model_file_ids[k]
            if mid not in self._structures:
                raise GeneticsUsageError(f"model file {mid} is needed here and was not given")
            m = _parse(self._structures[mid])
            s, e = self.windows[k]
            if m.n != e - s + 1:
                raise GeneticsUsageError(
                    f"model file {mid} has {m.n} residues; the model table says {e - s + 1}"
                )
            self._models[k] = m
        return self._models[k]

    def _pae(self, k: int) -> _Pae:
        if k not in self._paes:
            mid = self.model_file_ids[k]
            if mid not in self._pae_data:
                raise GeneticsUsageError(
                    f"PAE of model file {mid} is needed here and was not given; pass it, or ask "
                    "for no PAE (pae_cutoff=None, or pae=False in distances)"
                )
            s, e = self.windows[k]
            self._paes[k] = _Pae(self._pae_data[mid], e - s + 1, mid)
        return self._paes[k]

    def _position(self, address: Any) -> int:
        """a protein position from a protein position or a (model_file_id, file_position)."""
        if isinstance(address, (tuple, list)) and len(address) == 2 and isinstance(address[0], str):
            mid, file_position = address
            if mid not in self.model_file_ids:
                raise GeneticsUsageError(f"{mid} is not a model file of {self.accession}")
            s, e = self.windows[self.model_file_ids.index(mid)]
            return s + _file_index(e - s + 1, file_position)
        return _file_index(self.n_residues, address, "protein position") + 1

    def _owner(self, positions: np.ndarray) -> np.ndarray:
        return np.searchsorted(self._owned_hi, positions)

    def _pair_windows(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """the contract's pair_window() over arrays of protein positions; -1 where it is None."""
        owner_a, owner_b = self._owner(a), self._owner(b)
        out = np.where(owner_a == owner_b, owner_a, -1)
        open_ = out < 0
        # ascending, so a later window overwrites: the LAST window containing both wins
        for k, (s, e) in enumerate(self.windows):
            out[open_ & (a >= s) & (a <= e) & (b >= s) & (b <= e)] = k
        return out

    def _within(self, d: np.ndarray, radius: float) -> np.ndarray:
        # the lab compares a one-file protein's float32 distances with the radius in single
        # precision and a composed protein's in double; they differ only for a distance equal
        # to a radius that single precision rounds down
        return d < (np.float32(radius) if len(self.windows) == 1 else np.float64(radius))

    def _window_adjacency(self, k: int, radius: float, threshold: int | None) -> np.ndarray:
        adj = self._within(_distance_rows(self._model(k).ca, slice(None)), radius)
        if threshold is not None:
            adj &= self._pae(k).whole() < threshold
        return adj


def _contract_id(value: Any, what: str) -> str:
    if not isinstance(value, str) or not contract.MODEL_FILE_ID_RE.match(value):
        raise GeneticsUsageError(
            f"{what} must be 1 to 64 letters, digits, '.', '_' and '-', starting with a "
            f"letter or digit; got {value!r}"
        )
    return value


def _model_rows(models: Any) -> list[dict]:
    if isinstance(models, pl.DataFrame):
        models = models.to_dicts()
    elif isinstance(models, Mapping):
        models = [models]
    rows = []
    for r in models or []:
        missing = [c for c in _MODEL_ROW_FIELDS if c not in r]
        if missing:
            raise GeneticsUsageError(f"a model row lacks {missing}; pass rows of the model table")
        row = dict(r)
        for c in ("protein_start", "protein_end", "n_residues", "n_windows"):
            row[c] = _row_int(row, c)
        # either both owned bounds or neither: one alone cannot be checked against the rule
        owned = [row.get(c) not in (None, "") for c in ("owned_start", "owned_end")]
        if owned[0] != owned[1]:
            raise GeneticsUsageError(
                f"model row {row['model_file_id']!r} gives one of owned_start and owned_end "
                "without the other"
            )
        row["owned"] = (
            (_row_int(row, "owned_start"), _row_int(row, "owned_end")) if owned[0] else None
        )
        row["has_pae"] = _truthy(row["has_pae"])
        rows.append(row)
    if not rows:
        raise GeneticsUsageError("no model rows given")
    return sorted(rows, key=lambda r: r["protein_start"])


def protein(
    models: pl.DataFrame | Sequence[Mapping[str, Any]] | Mapping[str, Any],
    structures: Mapping[str, str | bytes],
    pae: Mapping[str, Any] | None = None,
) -> Protein:
    """Put one protein together from its model-table rows, model files and PAE.

    `models` — ALL of the protein's rows of the release's model table (a DataFrame or
    dicts), one per model file: `model_file_id`, `accession`, `protein_start`,
    `protein_end`, `n_residues`, `n_windows` and `has_pae` are required and may not be
    null, and `owned_start`, `owned_end` are checked when present. Every row is needed even
    when only some files are, because which file gives a pair depends on all of the
    windows; `n_windows` is how a missing row is noticed.

    `structures` — {model_file_id: the file's text or bytes, gzipped or not}. A file is
    parsed when first needed, so a question about one stretch of a long protein needs only
    the files `pair_window` sends it to; a missing one is named in the error.

    `pae` — {model_file_id: PAE}, each the release's uint8 object for that file (n² bytes
    or an (n, n) uint8 array) or a dict of {file position: row bytes} when only some rows
    were fetched. A residue's row is enough for its `neighbourhood` and for `distances`
    involving it; `adjacency` with a PAE cutoff needs whole matrices. Leave it out for a
    protein without PAE (`has_pae` false), or when no PAE is wanted.
    """
    rows = _model_rows(models)
    for r in rows:
        # both end up in file names, request paths and the files view() and pml() write
        _contract_id(r["model_file_id"], "a model row's model_file_id")
        _contract_id(r["accession"], "a model row's accession")
    accessions = sorted({r["accession"] for r in rows})
    if len(accessions) != 1:
        raise GeneticsUsageError(f"model rows of one protein are needed, got {accessions}")
    acc = accessions[0]
    windows = [(r["protein_start"], r["protein_end"]) for r in rows]
    if len({s for s, _ in windows}) != len(windows):
        raise GeneticsUsageError(f"{acc}: two model rows start at the same position")
    ids = [r["model_file_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise GeneticsUsageError(f"{acc}: two model rows have the same model_file_id")
    declared = {r["n_windows"] for r in rows}
    if declared != {len(rows)}:
        raise GeneticsUsageError(
            f"{acc} has {sorted(declared)} model files and {len(rows)} row(s) were given; every "
            "row is needed to tell which file gives a pair of residues"
        )
    if len({r["has_pae"] for r in rows}) != 1:
        raise GeneticsUsageError(f"{acc}: has_pae differs between the protein's model files")
    owned = contract.owning_window(windows)
    for r, (s, e), (lo, hi) in zip(rows, windows, owned):
        if r["n_residues"] != e - s + 1:
            raise GeneticsUsageError(f"{r['model_file_id']}: n_residues does not match its range")
        if not s <= lo <= hi <= e or r["owned"] not in (None, (lo, hi)):
            # the rows were written under a different ownership rule than the one implemented
            # here; computing on would place pairs in the wrong file
            raise GeneticsUsageError(
                f"{r['model_file_id']}: the owned range does not follow the release contract "
                f"version {contract.CONTRACT_VERSION} this SDK implements"
            )
    known = {r["model_file_id"] for r in rows}
    for name, given_ids in (("structures", structures), ("pae", pae or {})):
        unknown = sorted(set(given_ids) - known)
        if unknown:
            raise GeneticsUsageError(f"{name} names {unknown}, not model files of {acc}")
    if pae and not rows[0]["has_pae"]:
        raise GeneticsUsageError(
            f"{acc} has no PAE in the release (has_pae is false) and PAE was given"
        )
    return Protein(rows, structures, pae)


def _filter_threshold(protein: Protein, pae_cutoff: int | None) -> int | None:
    # validated even when unused, so a bad cutoff is not accepted only on proteins without PAE
    threshold = _pae_threshold(pae_cutoff)
    return threshold if protein.has_pae else None


def _check_radius(radius: float) -> float:
    if (
        isinstance(radius, bool)
        or not isinstance(radius, (int, float, np.number))
        or not radius > 0
    ):
        raise GeneticsUsageError(f"radius must be a positive number of Å, got {radius!r}")
    return float(radius)


def adjacency(
    protein: Protein, radius: float = _DEFAULT_RADIUS, pae_cutoff: int | None = _DEFAULT_PAE_CUTOFF
) -> dict[str, Any]:
    """Which residues are 3D neighbours, for the whole protein. The lab's
    `get_adjacency_matrix`.

    Returns `adjacency`, an (n, n) bool array over protein positions (index = position − 1):
    true where distance < `radius` and, when the protein has PAE and `pae_cutoff` is not
    None, PAE < `pae_cutoff`. The diagonal is true — a residue is in its own neighbourhood.
    A pair no single model file contains is false. `pae_filter_applied` says whether PAE
    was used: it is false for a protein without PAE, whose neighbourhoods are distance-only.

    Needs every model file of the protein and, with a PAE cutoff, every whole PAE matrix.
    The matrix is dense, n² bytes. For one residue use `neighbourhood`.
    """
    radius = _check_radius(radius)
    threshold = _filter_threshold(protein, pae_cutoff)
    result = {
        "pae_filter_applied": threshold is not None,
        "radius": radius,
        "pae_cutoff": pae_cutoff,
    }
    if len(protein.windows) == 1:
        return {"adjacency": protein._window_adjacency(0, radius, threshold), **result}
    n = protein.n_residues
    out = np.zeros((n, n), dtype=bool)
    owned_blocks = []
    # the lab's composition, in its order: every window over its whole range in start
    # order, then each window's owned diagonal block on top
    for k, ((s, e), (lo, hi)) in enumerate(zip(protein.windows, protein.owned)):
        adj = protein._window_adjacency(k, radius, threshold)
        out[s - 1 : e, s - 1 : e] = adj
        owned_blocks.append(adj[lo - s : hi - s + 1, lo - s : hi - s + 1].copy())
    for (lo, hi), block in zip(protein.owned, owned_blocks):
        out[lo - 1 : hi, lo - 1 : hi] = block
    return {"adjacency": out, **result}


def neighbourhood(
    protein: Protein,
    position: int | tuple[str, int],
    radius: float = _DEFAULT_RADIUS,
    pae_cutoff: int | None = _DEFAULT_PAE_CUTOFF,
) -> dict[str, Any]:
    """The residues in 3D proximity of one residue: one row of `adjacency`, with detail.

    `position` is a protein position, or a (model_file_id, file_position) address. Returns
    `neighbours`, one row per residue with distance < `radius` and (where the protein has
    PAE and `pae_cutoff` is not None) PAE < `pae_cutoff`, the residue itself included:
    `protein_position`, `residue`, `distance`, `pae` (null without PAE), `plddt`, and the
    `model_file_id` / `file_position` the pair was measured in. `pae_filter_applied` is
    false for a protein without PAE.

    In a multi-file protein the partners outside the centre's own window come from another
    file under the pair rule, so one neighbourhood can draw on several model files; it never
    contains a residue that shares no file with the centre. Needs the model files containing
    the centre and, for the PAE filter, the centre's PAE row in each of them.
    """
    radius = _check_radius(radius)
    threshold = _filter_threshold(protein, pae_cutoff)
    centre = protein._position(position)
    partners = np.arange(1, protein.n_residues + 1)
    window = protein._pair_windows(np.full_like(partners, centre), partners)
    frames = []
    for k in np.unique(window[window >= 0]).tolist():
        s, _ = protein.windows[k]
        m = protein._model(k)
        idx = partners[window == k] - s
        d = _distance_rows(m.ca, slice(centre - s, centre - s + 1))[0, idx]
        keep = protein._within(d, radius)
        codes = None
        if threshold is not None:
            codes = protein._pae(k).row(centre - s)[idx]
            keep &= codes < threshold
        elif protein.model_file_ids[k] in protein._pae_data and protein._pae(k).has_row(centre - s):
            codes = protein._pae(k).row(centre - s)[idx]
        idx = idx[keep]
        frames.append(
            pl.DataFrame(
                {
                    "protein_position": (idx + s).astype(np.int32),
                    "residue": m.residue[idx].tolist(),
                    "distance": d[keep],
                    "pae": _decode(codes[keep]) if codes is not None else np.full(len(idx), np.nan),
                    "plddt": m.plddt[idx],
                    "model_file_id": [protein.model_file_ids[k]] * len(idx),
                    "file_position": (idx + 1).astype(np.int32),
                },
                schema_overrides={"residue": pl.String, "model_file_id": pl.String},
            )
        )
    neighbours = (
        pl.concat(frames).sort("protein_position").with_columns(pl.col("pae").fill_nan(None))
    )
    return {
        "neighbours": neighbours,
        "pae_filter_applied": threshold is not None,
        "radius": radius,
        "pae_cutoff": pae_cutoff,
    }


def distances(
    protein: Protein, positions: Sequence[int | tuple[str, int]], pae: bool = True
) -> pl.DataFrame:
    """Distance, PAE and confidence for every pair among some residues of one protein.

    `positions` are protein positions, or (model_file_id, file_position) addresses; a
    repeated residue is taken once. Returns one row per unordered pair, in the order given:

    - `position_a`, `position_b` — protein positions.
    - `distance` — Cα–Cα distance in Å in the model file the pair rule selects; null when
      no single model file contains both residues, with `reason` saying so. A distance is
      never computed across files: each is in its own coordinate frame.
    - `pae` — PAE in Å in that same file; null when the protein has no PAE, when the pair
      has no distance, or with `pae=False`.
    - `min_plddt` — the lower of the two residues' pLDDT in that file.
    - `same_window` — true exactly when one window owns both residues. False with a
      distance means the pair was measured in a file that contains both and owns only one
      of them (or neither); false without a distance means no file contains both.
    - `model_file_id`, `file_position_a`, `file_position_b` — where the pair was measured.

    Needs the model files the pairs fall in and, for PAE, at least one of each pair's two
    PAE rows in that file.
    """
    pos = list(dict.fromkeys(protein._position(p) for p in positions))
    if len(pos) < 2:
        raise GeneticsUsageError("give at least two different residues")
    ia, ib = np.triu_indices(len(pos), k=1)
    a, b = np.array(pos)[ia], np.array(pos)[ib]
    window = protein._pair_windows(a, b)
    n = len(a)
    dist = np.full(n, np.nan, dtype=np.float32)
    pae_out = np.full(n, np.nan)
    plddt = np.full(n, np.nan)
    fa, fb = np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.int32)
    for k in np.unique(window[window >= 0]).tolist():
        sel = window == k
        s, _ = protein.windows[k]
        m = protein._model(k)
        xa, xb = a[sel] - s, b[sel] - s
        dist[sel] = _pair_distances(m.ca[xa], m.ca[xb])
        plddt[sel] = np.minimum(m.plddt[xa], m.plddt[xb])
        fa[sel], fb[sel] = xa + 1, xb + 1
        if pae and protein.has_pae:
            pae_out[sel] = _decode(protein._pae(k).codes(xa, xb))
    found = window >= 0
    ids = np.array(protein.model_file_ids + [None], dtype=object)[window]
    return pl.DataFrame(
        {
            "position_a": a.astype(np.int32),
            "position_b": b.astype(np.int32),
            "distance": dist,
            "pae": pae_out,
            "min_plddt": plddt,
            "same_window": protein._owner(a) == protein._owner(b),
            "model_file_id": ids.tolist(),
            "file_position_a": fa,
            "file_position_b": fb,
            "reason": [None if f else _NO_SHARED_MODEL for f in found.tolist()],
        },
        schema_overrides={"model_file_id": pl.String, "reason": pl.String},
    ).with_columns(
        pl.col("distance", "pae", "min_plddt").fill_nan(None),
        pl.when(pl.Series(found)).then(pl.col("file_position_a", "file_position_b")),
    )


# --------------------------------------------------------------------------- showing it

# the chat delivers no artifact above this (the sandbox's ARTIFACT_READ_MAX_BYTES), which is
# below the viewer's own limit, so a larger view would be written and never drawn
_MAX_VIEW_BYTES = min(512 * 1024, contract.VIEW_MAX_BYTES)

_HIGHLIGHT_KEYS = ("position", "label", "role", "colour")

# Mol*'s pLDDT confidence theme, which the chat's viewer colours by: each band starts above
# the pLDDT given, the first covering everything below
_PLDDT_BANDS = (
    (None, "plddt_very_low", "#ff7d45"),
    (50, "plddt_low", "#ffdb13"),
    (70, "plddt_confident", "#65cbf3"),
    (90, "plddt_very_high", "#0053d6"),
)
# what JavaScript's String.prototype.trim removes (ECMAScript WhiteSpace and LineTerminator),
# which is how the viewer decides a label is blank. Python's str.strip() differs: it leaves
# U+FEFF and removes U+001C-U+001F and U+0085
_JS_TRIMMED = (
    "\u0009\u000a\u000b\u000c\u000d\u0020\u00a0\u1680"
    "\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a"
    "\u2028\u2029\u202f\u205f\u3000\ufeff"
)

_PML_FILE_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_PML_OBJECT = "structure"
_PML_RESIDUES_PER_LINE = 50


def _artifact_path(path: str | None, default: str, suffix: str) -> str:
    """the rule genetics.plots writes by: a relative name lands in the artifacts directory."""
    path = default if path is None else path
    if not isinstance(path, str) or not path.lower().endswith(suffix):
        raise GeneticsUsageError(f"path must end with {suffix!r}, got {path!r}")
    if not os.path.isabs(path):
        path = os.path.join(os.environ.get("SANDBOX_ARTIFACTS_DIR") or ".", path)
    if not os.path.isdir(os.path.dirname(path) or "."):
        raise GeneticsUsageError(
            f"path {path!r} is in a directory that does not exist; give a bare file name"
        )
    return path


def _view_text(value: Any, what: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise GeneticsUsageError(f"{what} must be text, got {value!r}")
    text = value.strip(_JS_TRIMMED)
    try:
        length = len(text.encode("utf-16-le")) // 2
    except UnicodeEncodeError:
        raise GeneticsUsageError(f"{what} is not valid Unicode text") from None
    limit = contract.VIEW_MAX_TEXT_LENGTH
    if length > limit:
        raise GeneticsUsageError(
            f"{what} is too long: {length} UTF-16 code units, and the viewer shows at most "
            f"{limit}. An ordinary character is one unit and an emoji or other character "
            f"outside the Basic Multilingual Plane is two, so use at most {limit} ordinary "
            f"characters: {text[:30]!r}..."
        )
    return text or None


def _highlight(item: Any, i: int) -> dict[str, Any]:
    what = f"highlights[{i}]"
    if not isinstance(item, Mapping):
        return {"position": item, "label": None, "role": None, "colour": None}
    unknown = sorted(set(item) - set(_HIGHLIGHT_KEYS))
    if unknown or "position" not in item:
        raise GeneticsUsageError(
            f"{what} is a position, or a dict with 'position' and optionally 'label' and "
            f"one of 'role' and 'colour'; got keys {sorted(item)}"
        )
    role, colour = item.get("role"), item.get("colour")
    if role is not None and colour is not None:
        raise GeneticsUsageError(f"{what} gives both role and colour; a residue takes one")
    if role is not None and role not in contract.VIEW_ROLES:
        raise GeneticsUsageError(
            f"{what}: role must be one of {', '.join(contract.VIEW_ROLES)}, got {role!r}"
        )
    if colour is not None:
        if not isinstance(colour, str) or not contract.VIEW_COLOUR_RE.match(colour):
            raise GeneticsUsageError(
                f"{what}: colour must be '#rrggbb', got {colour!r}; or give a role"
            )
        colour = colour.lower()
    return {
        "position": item["position"],
        "label": _view_text(item.get("label"), f"{what}: label"),
        "role": role,
        "colour": colour,
    }


def _sphere(sphere: Any) -> tuple[Any, float] | None:
    if sphere is None:
        return None
    if isinstance(sphere, Mapping) and set(sphere) == {"centre", "radius"}:
        centre, radius = sphere["centre"], sphere["radius"]
    elif isinstance(sphere, (tuple, list)) and len(sphere) == 2:
        centre, radius = sphere
    else:
        raise GeneticsUsageError(
            "sphere is (centre, radius) or {'centre': ..., 'radius': ...}, the centre a "
            "position and the radius in Å"
        )
    radius = _check_radius(radius)
    if not radius <= contract.VIEW_MAX_SPHERE_RADIUS:
        raise GeneticsUsageError(
            f"sphere radius must be at most {contract.VIEW_MAX_SPHERE_RADIUS} Å, got {radius!r}"
        )
    return centre, radius


def _is_address(position: Any) -> bool:
    return (
        isinstance(position, (tuple, list)) and len(position) == 2 and isinstance(position[0], str)
    )


def _view_position(n: int, position: Any, what: str) -> int:
    return _file_index(min(n, contract.VIEW_MAX_POSITION), position, what) + 1


class _Layout:
    """One model file and what is shown on it: what `view` and `pml` both write from."""

    def __init__(
        self,
        target: Protein | str,
        highlights: Any,
        sphere: Any,
        colour_mode: str,
        model_file_id: str | None,
    ):
        if colour_mode not in contract.VIEW_COLOUR_MODES:
            raise GeneticsUsageError(
                f"colour_mode must be one of {', '.join(contract.VIEW_COLOUR_MODES)}, "
                f"got {colour_mode!r}"
            )
        self.colour_mode = colour_mode
        if isinstance(highlights, (str, bytes, Mapping)) or not isinstance(
            highlights, (Sequence, np.ndarray, pl.Series)
        ):
            raise GeneticsUsageError("highlights is a list of positions or of dicts")
        items = [_highlight(h, i) for i, h in enumerate(list(highlights))]
        ball = _sphere(sphere)
        self.outside: list[dict[str, Any]] = []
        self.sphere: dict[str, Any] | None = None
        self.sphere_left_out = False
        placed: list[tuple[int, dict[str, Any]]] = []

        if isinstance(target, Protein):
            self.accession: str | None = target.accession
            k = self._choose(target, items, ball, model_file_id)
            # again here: the object's attributes are assignable after protein() checked them
            self.accession = _contract_id(target.accession, "the protein's accession")
            self.model_file_id = _contract_id(target.model_file_ids[k], "the model file id")
            s, e = target.windows[k]
            for item in items:
                p = target._position(item["position"])
                if s <= p <= e:
                    placed.append((_view_position(e - s + 1, p - s + 1, "file position"), item))
                else:
                    owner = target.model_file_ids[int(target._owner(np.array([p]))[0])]
                    self.outside.append(
                        {"position": p, "label": item["label"], "model_file_id": owner}
                    )
            if ball:
                centre = target._position(ball[0])
                if s <= centre <= e:
                    self.sphere = {
                        "centre": _view_position(e - s + 1, centre - s + 1, "sphere centre"),
                        "radius": ball[1],
                    }
                else:
                    self.sphere_left_out = True
        elif isinstance(target, str) and contract.MODEL_FILE_ID_RE.match(target):
            if model_file_id not in (None, target):
                raise GeneticsUsageError(f"model_file_id {model_file_id!r} is not {target!r}")
            self.accession = None
            self.model_file_id = target
            for item in items:
                placed.append((self._file_position(item["position"], "highlight"), item))
            if ball:
                self.sphere = {
                    "centre": self._file_position(ball[0], "sphere centre"),
                    "radius": ball[1],
                }
        else:
            raise GeneticsUsageError(
                "the first argument is a Protein (genetics.protein_structure(...)) or a "
                f"model file id, got {target!r}"
            )

        # the viewer's merge, done here so that what is counted and reported is what it draws
        merged: dict[int, dict[str, Any]] = {}
        for file_position, item in placed:
            first = merged.setdefault(file_position, {**item, "file_position": file_position})
            if first["label"] is None:
                first["label"] = item["label"]
        self.residues = list(merged.values())
        if len(self.residues) > contract.VIEW_MAX_RESIDUES:
            raise GeneticsUsageError(
                f"{len(self.residues)} residues to mark; the viewer draws at most "
                f"{contract.VIEW_MAX_RESIDUES}. Mark fewer"
            )
        labelled = [r for r in self.residues if r["label"]]
        self.labels_not_drawn = [r["label"] for r in labelled[contract.VIEW_MAX_LABELS :]]
        self.n_labels = len(labelled)

    def _file_position(self, position: Any, what: str) -> int:
        if _is_address(position):
            if position[0] != self.model_file_id:
                raise GeneticsUsageError(
                    f"{what} {tuple(position)!r} is in another model file than "
                    f"{self.model_file_id}; pass the Protein to place it"
                )
            position = position[1]
        return _view_position(contract.VIEW_MAX_POSITION, position, f"{what} file position")

    @staticmethod
    def _choose(
        protein: Protein, items: list[dict], ball: tuple[Any, float] | None, mid: str | None
    ) -> int:
        if mid is not None:
            if mid not in protein.model_file_ids:
                raise GeneticsUsageError(f"{mid} is not a model file of {protein.accession}")
            return protein.model_file_ids.index(mid)
        if len(protein.windows) == 1:
            return 0
        if ball is None and not items:
            raise GeneticsUsageError(
                f"{protein.accession} has {len(protein.windows)} model files and nothing was "
                f"given to choose one by; pass model_file_id, one of {protein.model_file_ids}"
            )
        anchor = ball[0] if ball else items[0]["position"]
        position = protein._position(anchor)
        if _is_address(anchor):
            return protein.model_file_ids.index(anchor[0])
        return int(protein._owner(np.array([position]))[0])

    def report(self) -> dict[str, Any]:
        others = list(dict.fromkeys(o["model_file_id"] for o in self.outside))
        notes = []
        if self.outside:
            notes.append(
                f"highlights not in {self.model_file_id} were left out "
                f"({len(self.outside)}); they are in {', '.join(others)}. A model file has its own "
                "coordinates, so write one view per file (model_file_id=...)"
            )
        if self.sphere_left_out:
            notes.append(f"the sphere's centre is not in {self.model_file_id}; no sphere")
        return {
            "model_file_id": self.model_file_id,
            "n_residues": len(self.residues),
            "n_labels": self.n_labels,
            "outside_model_file": self.outside,
            "other_model_files": others,
            "sphere": self.sphere,
            "sphere_left_out": self.sphere_left_out,
            "notes": notes,
        }


def view(
    protein: Protein | str,
    highlights: Sequence[Any] = (),
    sphere: tuple[Any, float] | Mapping[str, Any] | None = None,
    *,
    colour_mode: str = contract.VIEW_DEFAULT_COLOUR_MODE,
    gene: str | None = None,
    model_file_id: str | None = None,
    path: str | None = None,
) -> dict[str, Any]:
    """Show a predicted model in the chat as an interactive 3D view with residues marked.

    Writes `<model_file_id>.structure-view.json` into the execution's artifacts directory;
    the chat draws a file with that ending as the structure, so do not also describe a
    download. `path` renames it and must keep the `.structure-view.json` ending.

    `protein` — a `Protein` (`genetics.protein_structure(...)`; no model file or PAE needs to
    have been fetched), or a model file id, in which case every position is a file position.

    `highlights` — the residues to mark, each a protein position, a
    (model_file_id, file_position) address, or a dict `{"position": <either>, "label": ...,
    "role": ...}` with `colour` ("#rrggbb") in place of `role`. Roles are "case", "control",
    "both" and "neutral", drawn in the viewer's own colours; a residue with neither is
    neutral grey. Only a residue with a `label` is labelled. A residue given twice is one
    residue: its first role or colour and its first label stand.

    `sphere` — (centre, radius in Å), drawn as a translucent ball around that residue's Cα.
    It is a picture of a radius in ONE file's coordinates, not a computed neighbourhood: to
    show `neighbourhood()`'s residues, list them in `highlights`.

    `colour_mode` — "plddt" (the default: the cartoon in the AlphaFold confidence bands) or
    "uniform".

    ONE VIEW SHOWS ONE MODEL FILE. For a protein in several files the file is, in order:
    `model_file_id` if given; the file a (model_file_id, file_position) address names, or
    else the file that owns the residue, for the sphere's centre if there is a sphere and
    otherwise for the first highlight. Highlights the file does not contain are NOT drawn
    and come back in `outside_model_file`; write one more view per file in
    `other_model_files`, passing `model_file_id=`.

    WHAT A RUN DELIVERS. The chat gets at most four non-image files of one run, taken in
    name order, and none over 512 KiB. One view per model file plus a `.pml` plus a model
    file can be more than four: spread them over several runs.

    Returns `path`, `model_file_id`, `n_residues` (distinct residues written), `n_labels`,
    `labels_not_drawn` (labels past the 40 the viewer draws; those stay visible on hover),
    `outside_model_file` (one `{position, label, model_file_id}` per highlight left out,
    `position` the protein position and `model_file_id` the file owning it),
    `other_model_files`, `sphere` (as written, in file positions, or None),
    `sphere_left_out` (true when the chosen file does not contain the centre) and `notes`.

    Raises GeneticsUsageError, writing nothing, for what the viewer would refuse or cut:
    more than 2,000 residues, a label or gene over 80 characters (an emoji counts as two), a
    radius over 100 Å, a
    colour that is not #rrggbb, role and colour together, or a file over 512 KiB.
    """
    layout = _Layout(protein, highlights, sphere, colour_mode, model_file_id)
    spec: dict[str, Any] = {
        "version": contract.VIEW_SPEC_VERSION,
        "model_file_id": _contract_id(layout.model_file_id, "the model file id"),
    }
    if layout.accession:
        spec["accession"] = _view_text(layout.accession, "accession")
    if gene := _view_text(gene, "gene"):
        spec["gene"] = gene
    spec["residues"] = [
        {"file_position": r["file_position"]}
        | {k: r[k] for k in ("label", "role", "colour") if r[k] is not None}
        for r in layout.residues
    ]
    if layout.sphere:
        spec["sphere"] = layout.sphere
    spec["colour_mode"] = layout.colour_mode
    data = json.dumps(spec, ensure_ascii=False).encode("utf-8")
    if len(data) > _MAX_VIEW_BYTES:
        raise GeneticsUsageError(
            f"the view is {len(data)} bytes; the chat delivers at most {_MAX_VIEW_BYTES}. "
            "Mark fewer residues or shorten the labels"
        )
    written = _artifact_path(
        path, layout.model_file_id + contract.VIEW_NAME_SUFFIX, contract.VIEW_NAME_SUFFIX
    )
    with open(written, "wb") as f:
        f.write(data)
    report = layout.report()
    if layout.labels_not_drawn:
        report["notes"].append(
            f"{len(layout.labels_not_drawn)} labels are past the "
            f"{contract.VIEW_MAX_LABELS} the viewer draws; they show on hover only"
        )
    return {"path": written, "labels_not_drawn": layout.labels_not_drawn, **report}


def _rgb(colour: str) -> str:
    # as 0-1 fractions: PyMOL reads a triple with no channel above 1 as fractions whatever
    # was meant, so 0-255 integers would turn #010101 white
    return "[" + ", ".join(f"{int(colour[i : i + 2], 16) / 255:.4f}" for i in (1, 3, 5)) + "]"


_PML_ID = r"[A-Za-z0-9][A-Za-z0-9._-]*"
_PML_RESI = rf"{_PML_OBJECT} and name CA and resi \d+(\+\d+)*"
# every command shape pml() emits outside its python block; a line that is none of these is
# a bug in the writer, and the script is not written
_PML_FRACTION = r"[01]\.\d{4}"
_PML_PLAIN_LINE_RE = re.compile(
    "("
    + "|".join(
        (
            r"# [A-Za-z0-9 ._-]+",
            rf"load {_PML_ID}, {_PML_OBJECT}",
            rf"hide everything, {_PML_OBJECT}",
            rf"show cartoon, {_PML_OBJECT}",
            rf"set_color [a-z0-9_]+, \[{_PML_FRACTION}, {_PML_FRACTION}, {_PML_FRACTION}\]",
            rf"color [a-z0-9_]+, {_PML_OBJECT}( and b > \d+)?",
            rf"show spheres, {_PML_RESI}",
            rf"color [a-z0-9_]+, {_PML_RESI}",
            rf"select neighbourhood, byres \(\({_PML_OBJECT} and name CA\) within "
            rf"\d+\.\d{{3}} of \({_PML_OBJECT} and name CA and resi \d+\)\)",
            "deselect",
            rf"orient {_PML_OBJECT}",
            "python",
            "python end",
        )
    )
    + r")\Z"
)
_PML_BLOCK_LINE_RE = re.compile(
    rf"(from pymol import cmd|cmd\.label\('{_PML_OBJECT} and name CA and resi \d+', .+\))\Z"
)
_PRINTABLE_ASCII_RE = re.compile(r"[\x20-\x7e]*\Z")


def _check_pml(lines: list[str]) -> None:
    in_block = False
    for line in lines:
        if line == "python end":
            in_block = False
        shape = _PML_BLOCK_LINE_RE if in_block else _PML_PLAIN_LINE_RE
        if not _PRINTABLE_ASCII_RE.match(line) or not shape.match(line):
            raise RuntimeError(
                f"internal error: genetics.structure.pml built a line it does not emit "
                f"({line[:60]!r}); nothing was written"
            )
        if line == "python":
            in_block = True


def _resi(positions: list[int]) -> list[str]:
    return [
        "+".join(map(str, positions[i : i + _PML_RESIDUES_PER_LINE]))
        for i in range(0, len(positions), _PML_RESIDUES_PER_LINE)
    ]


def pml(
    protein: Protein | str,
    highlights: Sequence[Any] = (),
    sphere: tuple[Any, float] | Mapping[str, Any] | None = None,
    *,
    colour_mode: str = contract.VIEW_DEFAULT_COLOUR_MODE,
    model_file_id: str | None = None,
    model_file: str | None = None,
    path: str | None = None,
) -> dict[str, Any]:
    """The same marking as `view`, as a PyMOL script for someone who opens the model in
    their own PyMOL.

    Takes what `view` takes and chooses the model file the same way. Writes
    `<model_file_id>.pml` into the artifacts directory (`path` renames it; it must end
    `.pml`): the cartoon coloured by pLDDT in the four AlphaFold bands (pLDDT is the
    B-factor column) or plain grey with `colour_mode="uniform"`, a sphere on the Cα of every
    highlight in its role's colour, the labels, and for `sphere` a selection named
    `neighbourhood` of the residues whose Cα is within the radius of the centre's — by
    distance in this one file only, without PAE.

    THE SCRIPT LOADS THE MODEL FROM A FILE BESIDE IT, named `<model_file_id>.pdb` unless
    `model_file` gives another bare file name. The chat's viewer offers no download of the
    model, so write it in the same run for the user to download with the script, gzipped,
    which PyMOL reads, and pass `model_file=mid + ".pdb.gz"`:

        data = gzip.compress(genetics.protein_structure_file(mid))
        with open(os.path.join(os.environ["SANDBOX_ARTIFACTS_DIR"], mid + ".pdb.gz"), "wb") as f:
            f.write(data)

    WHAT A RUN DELIVERS. The chat gets at most four non-image files of one run, taken in
    name order, and none over 512 KiB, so several scripts and models go in several runs.
    Check `len(data)`: in the release of 2026-11-12, 25 of the 21,946 models are over 512 KiB
    even gzipped (all AlphaFold Server `AF3-` models, the largest 784 KiB). A run cannot
    hand those to the user at all; write the script anyway and say plainly that the model
    file for it is not downloadable from the chat.

    Returns `path`, `model_file` (the name the script loads) and what `view` returns except
    the label counts: PyMOL draws every label.
    """
    layout = _Layout(protein, highlights, sphere, colour_mode, model_file_id)
    mid = layout.model_file_id
    model_file = mid + ".pdb" if model_file is None else model_file
    # the name is written into a PyMOL command, so it is held to the model ids' own charset
    _contract_id(mid, "the model file id")
    if not isinstance(model_file, str) or not _PML_FILE_NAME_RE.match(model_file):
        raise GeneticsUsageError(
            f"model_file must be a bare file name of letters, digits, '.', '_' and '-', "
            f"got {model_file!r}"
        )
    obj = _PML_OBJECT
    lines = [
        f"# PyMOL script written by genetics.structure.pml for model file {mid}.",
        f"# It loads {model_file} from the directory PyMOL is run in.",
        "# Residue numbers are those of that file.",
        f"load {model_file}, {obj}",
        f"hide everything, {obj}",
        f"show cartoon, {obj}",
    ]
    if layout.colour_mode == "plddt":
        for above, name, colour in _PLDDT_BANDS:
            lines.append(f"set_color {name}, {_rgb(colour)}")
            lines.append(f"color {name}, {obj}" + (f" and b > {above}" if above else ""))
    else:
        lines.append(f"color grey80, {obj}")

    by_colour: dict[str, list[int]] = {}
    for r in layout.residues:
        colour = r["colour"] or contract.VIEW_ROLE_COLOURS[r["role"] or "neutral"]
        by_colour.setdefault(colour, []).append(r["file_position"])
    for colour, positions in by_colour.items():
        name = f"highlight_{colour[1:]}"
        lines.append(f"set_color {name}, {_rgb(colour)}")
        for resi in _resi(positions):
            lines.append(f"show spheres, {obj} and name CA and resi {resi}")
            lines.append(f"color {name}, {obj} and name CA and resi {resi}")
    if layout.sphere:
        lines.append(
            f"select neighbourhood, byres (({obj} and name CA) within "
            f"{layout.sphere['radius']:.3f} of ({obj} and name CA and resi "
            f"{layout.sphere['centre']}))"
        )
        lines.append("deselect")
    labelled = [r for r in layout.residues if r["label"]]
    if labelled:
        # labels are text a model wrote. PyMOL evaluates a label as a Python expression and
        # its command parser splits on characters a label may hold, so each goes in as a
        # Python string literal inside a python block: ascii() leaves nothing in it that can
        # end the literal, the statement or the line
        lines += ["python", "from pymol import cmd"]
        for r in labelled:
            selection = f"{obj} and name CA and resi {r['file_position']}"
            lines.append(f"cmd.label({selection!r}, {ascii(ascii(r['label']))})")
        lines.append("python end")
    lines.append(f"orient {obj}")

    _check_pml(lines)
    written = _artifact_path(path, mid + ".pml", ".pml")
    with open(written, "w", encoding="ascii", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    report = layout.report()
    del report["n_labels"]
    return {"path": written, "model_file": model_file, **report}
