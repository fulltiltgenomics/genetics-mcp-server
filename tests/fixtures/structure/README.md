# structure reference outputs

What the Finucane lab's geometry code (`utils.py`: `get_distance_and_pae_matrices`,
`threshold_adjacency_matrix`, `get_adjacency_matrix`) returns on its own reference bundle,
which `tests/test_sdk_structure.py` holds `genetics_mcp_server.sdk.structure` to. Which
proteins, which lab commit, which release and which library versions produced them is
recorded in `expected.json`; the proteins are chosen by rule in `generate.py`
(`choose_proteins`), seeded.

- `models.tsv` — the release's `models.tsv.gz` rows of the chosen proteins, unchanged.
- `structures/<model_file_id>.pdb.gz` — each model file cut down to its CA records. The lab's
  code ran on the whole files; `expected.json` holds hashes of what its parser read from them.
- `structures_full/` — whole model files, one per source, to show the cut changes nothing.
- `pae/<model_file_id>.u8.gz` — the release's PAE object, gzipped, for short model files.
- `pae_rows.npz` — for long model files, only the PAE rows of the sampled centre residues.
- `expected.npz` — per protein: the sampled `centres`, sampled `pairs` with the lab's
  `distance` (infinite where no file contains both) and `pae` (with its release `pae_code`),
  and per setting `sN` the lab's adjacency `row_sums` and the packed adjacency rows of the
  centres (`centre_rows`).
- `expected.json` — the settings (radius, PAE cutoff; 0 is the lab's "no filter"), and per
  protein the sha256 of each setting's whole packed adjacency matrix.

The lab's HDF5 variant map is not an input: it places residues at some window boundaries in
a non-owning file. The reference is the lab's geometry functions only.

To regenerate, with the lab's requirements installed (numpy, pandas, scikit-learn, biopython,
hdf5plugin) and about 3 GB of memory:

```
python tests/fixtures/structure/generate.py \
    --lab-repo <clone of the lab's code> \
    --bundle <the lab's sir-reference-data directory> \
    --release-dir <a protein-structure release build: models.tsv.gz, pae/> \
    --out tests/fixtures/structure
```

`PROTEIN_STRUCTURE_RELEASE_DIR=<release build>` makes the tests also check the PAE-filtered
whole-protein adjacency of the proteins whose PAE is only sampled here, and
`PROTEIN_STRUCTURE_MUNGE_CONTRACT=<path to protein_structure_contract.py>` compares the SDK's
copy of the release contract with the munge module when no sibling checkout is found.
