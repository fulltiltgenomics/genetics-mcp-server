"""Writes the tiny gene set bundle next to this file, in the layout the suite stages into the
sandbox image (sets/members/genes parquet plus manifest.json).

`"fixture": true` at the manifest's top level marks it as test data; the suite's image build
is to refuse a bundle carrying it. Run: python tests/fixtures/genesets/make_bundle.py
"""

import json
from pathlib import Path

import polars as pl

HERE = Path(__file__).parent
N_GENES = 32


def ens(i: int) -> str:
    return f"ENSG{i:011d}"


def main() -> None:
    symbols = [f"GENE{i}" for i in range(1, N_GENES + 1)]
    symbols[29] = "C1orf99"  # mixed case, for the case-insensitive fallback
    symbols[30], symbols[31] = "Zed1", "ZED1"  # one case-fold, two genes: must stay unresolved
    genes = pl.DataFrame(
        {
            "ensembl_gene_id": [ens(i) for i in range(1, N_GENES + 1)],
            "hgnc_id": [f"HGNC:{90000 + i}" for i in range(1, N_GENES + 1)],
            "symbol": symbols,
            # GENE29 has no Entrez id
            "entrez_id": [None if i == 29 else str(1000 + i) for i in range(1, N_GENES + 1)],
            "locus_group": ["protein-coding gene"] * N_GENES,
            "locus_type": ["gene with protein product"] * N_GENES,
            "gencode_gene_type": ["protein_coding"] * N_GENES,
            "chr": [1] * N_GENES,
            "gene_start": [1000 * i for i in range(1, N_GENES + 1)],
            "gene_end": [1000 * i + 500 for i in range(1, N_GENES + 1)],
            "strand": ["+"] * N_GENES,
        }
    )
    sets = {
        ("coll.a", "A_EARLY", "a early genes"): range(1, 9),
        ("coll.a", "A_MIDDLE", "a middle genes"): range(5, 15),
        ("coll.a", "A_LATE", "a late genes"): range(20, 27),
        ("coll.b", "B_FIRST", "b first genes"): range(1, 7),
        ("coll.b", "B_SECOND", "b second genes"): range(10, 21),
    }
    pl.DataFrame(
        {
            "set_id": [s for _, s, _ in sets],
            "collection": [c for c, _, _ in sets],
            "name": [n for _, _, n in sets],
            "url_or_stable_id": [f"https://example.org/{s}" for _, s, _ in sets],
            "n_given": [len(m) for m in sets.values()],
            "n_mapped": [len(m) for m in sets.values()],
            "n_unmapped": [0] * len(sets),
        }
    ).write_parquet(HERE / "sets.parquet")
    pl.DataFrame(
        {
            "set_id": [s for (_, s, _), m in sets.items() for _ in m],
            "ensembl_gene_id": [ens(i) for m in sets.values() for i in m],
            "symbol": [symbols[i - 1] for m in sets.values() for i in m],
        }
    ).write_parquet(HERE / "members.parquet")
    genes.write_parquet(HERE / "genes.parquet")
    manifest = {
        "fixture": True,
        "release_id": "20990101",
        "bundle": {"release_id": "20990101.fixture"},
        "collections": [
            {
                "collection": "coll.a",
                "spdx_license": "CC-BY-4.0",
                "upstream_release": "1.0",
                "go_release": "GO-basic obo file released on 2099-01-01",
                "attribution": "Fixture collection A, CC BY 4.0. GO release 2099-01-01.",
            },
            {
                "collection": "coll.b",
                "spdx_license": "LicenseRef-HPO",
                "upstream_release": "1.0",
                "hpo_release": "2099-01-02",
                "licence_url": "https://example.org/hpo-licence",
                "attribution": "Fixture collection B. HPO version 2099-01-02; cite the consortium.",
            },
        ],
        "staging": {"profile": "fixture", "dropped": []},
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    main()
