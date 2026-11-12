"""The slice of the protein structure release contract that `genetics.structure` computes with.

A copy, not an import: the definition lives in genetics-results-munge
`scripts/protein_structure_contract.py`, and the SDK ships into a sandbox image that holds no
other repo's code. Only what the geometry and its fetches need is copied: the contract
version, the PAE encoding, the two window rules, `models.tsv.gz`'s column list, the
per-response byte limit, and what a view-spec for the browser's viewer may hold.
tests/test_sdk_structure.py pins these and compares them with the munge module when a
checkout of it is found.
"""

import re

CONTRACT_VERSION = 1

MODEL_FILE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")

MODEL_COLUMNS = [
    ("model_file_id", str),
    ("accession", str),
    ("uniprot_isoform", str),
    ("gene_id", str),
    ("gene_name", str),
    ("source_gene_symbol", str),
    ("source", str),
    ("source_label", str),
    ("model_version", str),
    ("licence", str),
    ("window_index", int),
    ("n_windows", int),
    ("protein_start", int),
    ("protein_end", int),
    ("n_residues", int),
    ("owned_start", int),
    ("owned_end", int),
    ("has_pae", bool),
    ("structure_path", str),
    ("structure_sha256", str),
    ("structure_bytes", int),
    ("pae_path", str),
    ("pae_sha256", str),
    ("pae_bytes", int),
    ("source_pdb_sha256", str),
    ("source_pae_sha256", str),
]

PAE_SCALE = 8
PAE_NO_VALUE = 255
PAE_MAX_CODE = 254

# SDK limit on one HTTP response
MAX_RESPONSE_BYTES = 16 * 1024 * 1024


VIEW_SPEC_VERSION = 1
VIEW_ROLES = ("case", "control", "both", "neutral")
VIEW_COLOUR_MODES = ("plddt", "uniform")
VIEW_DEFAULT_COLOUR_MODE = "plddt"
VIEW_NAME_SUFFIX = ".structure-view.json"
VIEW_ROLE_COLOURS = {
    "case": "#d62728",
    "control": "#1f77b4",
    "both": "#9467bd",
    "neutral": "#7f7f7f",
}
VIEW_SPHERE_COLOUR = "#ff7f0e"
VIEW_COLOUR_RE = re.compile(r"^#[0-9a-fA-F]{6}\Z")
VIEW_MAX_BYTES = 1_000_000
VIEW_MAX_RESIDUES = 2000
VIEW_MAX_LABELS = 40
VIEW_MAX_TEXT_LENGTH = 80
VIEW_MAX_POSITION = 100_000
VIEW_MAX_SPHERE_RADIUS = 100


def pae_rows_per_request(n: int) -> int:
    """the most rows of an n x n PAE object one request may ask for: floor(MAX_RESPONSE_BYTES / n)."""
    if n < 1:
        raise ValueError("n must be positive")
    return MAX_RESPONSE_BYTES // n


def decode_pae(code: int) -> float | None:
    return None if code == PAE_NO_VALUE else code / PAE_SCALE


def owning_window(windows: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """the protein-position range each window owns, for windows sorted by start."""
    starts = [s for s, _ in windows]
    if starts != sorted(starts) or len(set(starts)) != len(starts):
        raise ValueError("windows must be sorted by start with distinct starts")
    n = max(e for _, e in windows)
    owned, lo = [], 1
    for k, (_, end) in enumerate(windows):
        nxt = windows[k + 1][0] if k + 1 < len(windows) else n
        hi = (end + nxt) // 2
        owned.append((lo, hi))
        lo = hi + 1
    return owned


def pair_window(windows: list[tuple[int, int]], a: int, b: int) -> int | None:
    """index of the window whose model gives the pair (a, b) of protein positions, or None."""
    owned = owning_window(windows)
    for k, (lo, hi) in enumerate(owned):
        if lo <= a <= hi and lo <= b <= hi:
            return k
    for k in range(len(windows) - 1, -1, -1):
        s, e = windows[k]
        if s <= a <= e and s <= b <= e:
            return k
    return None
