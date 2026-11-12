"""Write the reference outputs tests/test_sdk_structure.py holds genetics.structure to.

Runs the lab's own geometry (utils.get_distance_and_pae_matrices, threshold_adjacency_matrix
and get_adjacency_matrix) on the lab's reference bundle and writes, per protein, what those
functions returned: distances and PAE at sampled residue pairs, per-residue neighbour counts
and a hash of the whole adjacency matrix at each setting, and the adjacency rows of sampled
centre residues. Inputs the tests need are copied beside them in reduced form: model files
cut down to their CA records, PAE as the release's uint8 objects (whole for small proteins,
the centre residues' rows for large ones).

Nothing here imports the SDK: the expected values are the lab's, produced without the port.

    python generate.py --lab-repo <clone of the lab's code> \
        --bundle <sir-reference-data> --release-dir <release build dir> --out .

needs numpy, pandas, scikit-learn, biopython and hdf5plugin (the lab's requirements).
"""

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np

# (radius in angstrom, PAE cutoff; 0 is the lab's spelling of "no PAE filter"). 8.3 is not
# representable in single precision, which is where the lab's two comparison widths differ
SETTINGS = [(15.0, 15), (15.0, 0), (8.3, 20), (30.0, 10)]

# a protein's whole PAE is shipped when its files are at most this long; above it only the
# sampled centres' rows are, which keeps the directory small
WHOLE_PAE_MAX_RESIDUES = 600

N_AF2DB_SINGLE = 32
N_AF3_SINGLE = 10
N_AF2DB_MULTI = 4
N_AF3_MULTI = 3
N_RANDOM_CENTRES = 24
N_RANDOM_PARTNERS = 6
N_FULL_MODEL_FILES = 2


def read_models(release_dir: Path) -> tuple[list[str], list[dict]]:
    with gzip.open(release_dir / "models.tsv.gz", "rt") as f:
        header = f.readline().rstrip("\n").lstrip("#").split("\t")
        return header, list(csv.DictReader(f, fieldnames=header, delimiter="\t"))


def choose_proteins(rows: list[dict], rng: np.random.Generator) -> list[str]:
    by_acc: dict[str, list[dict]] = {}
    for r in rows:
        by_acc.setdefault(r["accession"], []).append(r)

    def length(acc):
        return max(int(r["protein_end"]) for r in by_acc[acc])

    def pick(source, multi, has_pae=None, lo=0, hi=10**9):
        return sorted(
            acc
            for acc, rs in by_acc.items()
            if rs[0]["source"] == source
            and (len(rs) > 1) == multi
            and (has_pae is None or (rs[0]["has_pae"] == "true") == has_pae)
            and lo <= length(acc) <= hi
        )

    small_af2 = pick("AF2DB", False, True, 60, 420)
    small_af3 = pick("AF3", False, True, 30, WHOLE_PAE_MAX_RESIDUES)
    chosen = list(rng.choice(small_af2, N_AF2DB_SINGLE, replace=False))
    chosen += list(rng.choice(small_af3, N_AF3_SINGLE, replace=False))
    # the longest single-file model of each source
    chosen.append(max(pick("AF2DB", False, True), key=lambda a: (length(a), a)))
    chosen.append(max(pick("AF3", False, True), key=lambda a: (length(a), a)))
    af2_multi = sorted(pick("AF2DB", True), key=lambda a: (length(a), a))
    more_windows = next(a for a in af2_multi if len(by_acc[a]) > len(by_acc[af2_multi[0]]))
    chosen += af2_multi[: N_AF2DB_MULTI - 1] + [more_windows]
    chosen += sorted(pick("AF3", True, True), key=lambda a: (length(a), a))[:N_AF3_MULTI]
    assert len(set(chosen)) == len(chosen)
    return [str(a) for a in chosen]


def encode_pae(value: float) -> int:
    """the release encoding, restated so the expected codes do not come from the SDK."""
    return 255 if not math.isfinite(value) else min(254, int(math.floor(value * 8 + 0.5)))


def ca_only(pdb_text: str) -> str:
    keep = [ln for ln in pdb_text.splitlines() if ln.startswith("ATOM") and ln[12:16] == " CA "]
    return "\n".join(keep + ["END", ""])


def write_gz(path: Path, data: bytes) -> None:
    buf = io.BytesIO()
    # mtime 0 so regenerating identical content gives identical bytes
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0, compresslevel=9) as f:
        f.write(data)
    path.write_bytes(buf.getvalue())


def biopython_residues(pdb_path: Path):
    """the lab's reading of a model file: its CA coordinates, plus letters and B-factors."""
    from Bio.PDB import PDBParser
    from Bio.SeqUtils import seq1

    with gzip.open(pdb_path, "rt") as handle:
        structure = PDBParser(QUIET=True).get_structure("protein", handle)
    coords, letters, bfactors = [], [], []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    coords.append(residue["CA"].get_coord())
                    letters.append(seq1(residue.get_resname()))
                    bfactors.append(residue["CA"].get_bfactor())
    return np.array(coords), "".join(letters), np.array(bfactors, dtype=np.float64)


def sample_positions(windows, owned, n, rng):
    centres = set(
        int(p) for p in rng.choice(np.arange(1, n + 1), min(n, N_RANDOM_CENTRES), replace=False)
    )
    edges = set()
    if len(windows) > 1:
        for (s, e), (lo, hi) in zip(windows, owned):
            for p in (s, e, lo, hi):
                edges.update(q for q in (p - 1, p, p + 1) if 1 <= q <= n)
        # every edge of a few-window protein, a sample of a many-window one
        edge_list = sorted(edges)
        if len(edge_list) > 2 * N_RANDOM_CENTRES:
            edge_list = [int(p) for p in rng.choice(edge_list, 2 * N_RANDOM_CENTRES, replace=False)]
        centres.update(edge_list)
    centres = sorted(centres)
    pairs = []
    for a in centres:
        partners = set(
            int(p)
            for p in rng.choice(np.arange(1, n + 1), min(n, N_RANDOM_PARTNERS), replace=False)
        )
        partners.update(q for q in (a - 5, a - 1, a, a + 1, a + 5) if 1 <= q <= n)
        if edges:
            partners.update(
                int(p) for p in rng.choice(sorted(edges), N_RANDOM_PARTNERS, replace=False)
            )
        pairs += [(a, b) for b in sorted(partners)]
    return np.array(centres, dtype=np.int32), np.array(pairs, dtype=np.int32)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--lab-repo", required=True, type=Path)
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--release-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=20261112)
    args = ap.parse_args()

    sys.path.insert(0, str(args.lab_repo))
    import Bio
    import sklearn
    import utils as lab

    guide = str(args.bundle / "pdb_pae_file_pos_guide.tsv")
    pdb_dir, pae_dir = str(args.bundle / "pdb_files"), str(args.bundle / "pae_files")

    header, rows = read_models(args.release_dir)
    accessions = choose_proteins(rows, np.random.default_rng(args.seed))
    chosen_rows = [r for r in rows if r["accession"] in set(accessions)]

    out = args.out
    (out / "structures").mkdir(parents=True, exist_ok=True)
    (out / "structures_full").mkdir(exist_ok=True)
    (out / "pae").mkdir(exist_ok=True)
    with open(out / "models.tsv", "w") as f:
        f.write("#" + "\t".join(header) + "\n")
        for r in chosen_rows:
            f.write("\t".join(r[c] for c in header) + "\n")

    arrays: dict[str, np.ndarray] = {}
    pae_rows: dict[str, np.ndarray] = {}
    proteins: dict[str, dict] = {}
    full_files = []

    for acc in accessions:
        prows = sorted(
            (r for r in chosen_rows if r["accession"] == acc), key=lambda r: int(r["protein_start"])
        )
        windows = [(int(r["protein_start"]), int(r["protein_end"])) for r in prows]
        owned = [(int(r["owned_start"]), int(r["owned_end"])) for r in prows]
        has_pae = prows[0]["has_pae"] == "true"
        rng = np.random.default_rng([args.seed, zlib.crc32(acc.encode())])

        distance, pae = lab.get_distance_and_pae_matrices(guide, pdb_dir, pae_dir, acc, 15)
        n = distance.shape[0]
        assert n == max(e for _, e in windows), acc
        assert (pae is not None) == has_pae, acc

        centres, pairs = sample_positions(windows, owned, n, rng)
        ia, ib = pairs[:, 0] - 1, pairs[:, 1] - 1
        arrays[f"{acc}|centres"] = centres
        arrays[f"{acc}|pairs"] = pairs
        arrays[f"{acc}|distance"] = distance[ia, ib].astype(np.float32)
        assert (arrays[f"{acc}|distance"].astype(np.float64) == distance[ia, ib]).all(), acc
        if has_pae:
            raw = pae[ia, ib].astype(np.float64)
            arrays[f"{acc}|pae"] = raw.astype(np.float32)
            arrays[f"{acc}|pae_code"] = np.array([encode_pae(v) for v in raw], dtype=np.uint8)

        hashes, edges = [], []
        for i, (radius, cutoff) in enumerate(SETTINGS):
            adj = lab.threshold_adjacency_matrix(distance, pae, radius, cutoff)
            if i == 0:
                assert np.array_equal(
                    adj, lab.get_adjacency_matrix(guide, pdb_dir, pae_dir, acc, radius, cutoff)
                ), acc
            arrays[f"{acc}|s{i}|row_sums"] = adj.sum(axis=1).astype(np.int32)
            arrays[f"{acc}|s{i}|centre_rows"] = np.packbits(adj[centres - 1], axis=1)
            hashes.append(hashlib.sha256(np.packbits(adj, axis=None).tobytes()).hexdigest())
            edges.append(int(adj.sum()))
        del distance, pae

        files = {}
        for r, (s, e) in zip(prows, windows):
            mid = r["model_file_id"]
            src = args.bundle / "pdb_files" / f"{mid}.pdb.gz"
            assert hashlib.sha256(src.read_bytes()).hexdigest() == r["structure_sha256"], mid
            text = gzip.decompress(src.read_bytes()).decode()
            write_gz(out / "structures" / f"{mid}.pdb.gz", ca_only(text).encode())
            coords, letters, bfactors = biopython_residues(src)
            files[mid] = {
                "sequence_sha256": hashlib.sha256(letters.encode()).hexdigest(),
                "ca_float32_sha256": hashlib.sha256(coords.astype("<f4").tobytes()).hexdigest(),
                "plddt_float64_sha256": hashlib.sha256(
                    bfactors.astype("<f8").tobytes()
                ).hexdigest(),
            }
            if (
                len(windows) == 1
                and len(full_files) < N_FULL_MODEL_FILES
                and r["source"] not in {x["source"] for x in full_files}
            ):
                (out / "structures_full" / f"{mid}.pdb.gz").write_bytes(src.read_bytes())
                full_files.append({"model_file_id": mid, "source": r["source"]})
            if has_pae:
                nres = int(r["n_residues"])
                u8 = np.fromfile(args.release_dir / r["pae_path"], dtype=np.uint8).reshape(
                    nres, nres
                )
                if nres <= WHOLE_PAE_MAX_RESIDUES:
                    write_gz(out / "pae" / f"{mid}.u8.gz", u8.tobytes())
                else:
                    inside = centres[(centres >= s) & (centres <= e)] - s + 1
                    pae_rows[f"{mid}|file_positions"] = inside.astype(np.int32)
                    pae_rows[f"{mid}|rows"] = u8[inside - 1]
        proteins[acc] = {
            "n_residues": n,
            "source": prows[0]["source"],
            "has_pae": has_pae,
            "adjacency_sha256": hashes,
            "adjacency_edges": edges,
            "files": files,
        }
        print(f"{acc}: {n} residues, {len(windows)} file(s), {len(pairs)} pairs", flush=True)

    np.savez_compressed(out / "expected.npz", **arrays)
    np.savez_compressed(out / "pae_rows.npz", **pae_rows)
    lab_commit = subprocess.run(
        ["git", "-C", str(args.lab_repo), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    meta = {
        "settings": SETTINGS,
        "seed": args.seed,
        "release": args.release_dir.name,
        "lab_commit": lab_commit,
        "versions": {
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
            "biopython": Bio.__version__,
        },
        "full_model_files": full_files,
        "proteins": proteins,
    }
    (out / "expected.json").write_text(json.dumps(meta, indent=1) + "\n")


if __name__ == "__main__":
    main()
