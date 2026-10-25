"""genetics.enrichment's locus-level functions: genes_for_loci, locus_enrichment and simplify.

Most tests write a small synthetic bundle whose gene coordinates are chosen so the expected
mapping can be checked by hand; the fixture bundle under tests/fixtures/genesets/ (all 32
genes on chr1 at 1000*i .. 1000*i + 500) is used where a shipped collection name matters.
"""

import json
from itertools import product
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from genetics_mcp_server.sdk import enrichment
from genetics_mcp_server.sdk.errors import GeneticsUsageError

FIXTURE = Path(__file__).parent / "fixtures" / "genesets"


def write_bundle(path: Path, genes: list[tuple], sets: dict[str, list[str]]) -> Path:
    """`genes`: (symbol, chr, start, end) or (symbol, chr, start, end, locus_group)."""
    path.mkdir(parents=True, exist_ok=True)
    rows = [g if len(g) == 5 else (*g, "protein-coding gene") for g in genes]
    ens = {r[0]: f"ENSG{i:011d}" for i, r in enumerate(rows, 1)}
    pl.DataFrame(
        {
            "ensembl_gene_id": [ens[r[0]] for r in rows],
            "hgnc_id": [f"HGNC:{i}" for i in range(len(rows))],
            "symbol": [r[0] for r in rows],
            "entrez_id": [str(1000 + i) for i in range(len(rows))],
            "locus_group": [r[4] for r in rows],
            "locus_type": ["x"] * len(rows),
            "gencode_gene_type": ["x"] * len(rows),
            "chr": [r[1] for r in rows],
            "gene_start": [r[2] for r in rows],
            "gene_end": [r[3] for r in rows],
            "strand": ["+"] * len(rows),
        },
        schema_overrides={"chr": pl.Int64, "gene_start": pl.Int64, "gene_end": pl.Int64},
    ).write_parquet(path / "genes.parquet")
    pl.DataFrame(
        {
            "set_id": list(sets),
            "collection": ["syn"] * len(sets),
            "name": [s.lower() for s in sets],
            "url_or_stable_id": list(sets),
            "n_given": [len(m) for m in sets.values()],
            "n_mapped": [len(m) for m in sets.values()],
            "n_unmapped": [0] * len(sets),
        }
    ).write_parquet(path / "sets.parquet")
    pl.DataFrame(
        {
            "set_id": [s for s, m in sets.items() for _ in m],
            "ensembl_gene_id": [ens[g] for m in sets.values() for g in m],
            "symbol": [g for m in sets.values() for g in m],
        },
        schema={"set_id": pl.Utf8, "ensembl_gene_id": pl.Utf8, "symbol": pl.Utf8},
    ).write_parquet(path / "members.parquet")
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "release_id": "syn",
                "bundle": {"release_id": "syn.1", "gencode": {"version": "99"}},
                "collections": [
                    {"collection": "syn", "spdx_license": "CC0-1.0", "attribution": "syn"}
                ],
            }
        )
    )
    return path


@pytest.fixture
def use_bundle(monkeypatch, tmp_path):
    def use(genes, sets=None):
        path = write_bundle(tmp_path / f"b{len(enrichment._BUNDLES)}", genes, sets or {})
        monkeypatch.setenv(enrichment.GENESETS_ENV, str(path))
        return path

    enrichment._BUNDLES.clear()
    yield use
    enrichment._BUNDLES.clear()


# chr1: A [1000, 2000], B [5000, 6000], C [10000, 20000] with D [12000, 13000] nested in it,
# E [30000, 31000]; chr23 (X): XG; F is a non-coding gene closer to 40000 than E
HAND = [
    ("A", 1, 1000, 2000),
    ("B", 1, 5000, 6000),
    ("C", 1, 10000, 20000),
    ("D", 1, 12000, 13000),
    ("E", 1, 30000, 31000),
    ("F", 1, 39000, 39500, "non-coding RNA"),
    ("XG", 23, 1000, 2000),
]


def nearest(loci, **kw):
    out = enrichment.genes_for_loci(loci, merge_distance=0, **kw)
    loci = out["loci"]
    first = zip(loci["lead_ids"].list.first(), loci["gene_symbols"].list.first(), strict=True)
    return dict(first), out


@pytest.mark.parametrize(
    "pos, gene, distance",
    [
        (1500, "A", 0),  # inside
        (100, "A", 900),  # before the first gene
        (3000, "A", 1000),  # outside, nearer the left
        (3500, "A", 1500),  # equidistant from A's end and B's start: the earlier gene
        (4600, "B", 400),
        (12500, "C", 0),  # inside both C and the nested D: the earlier start
        (25000, "C", 5000),  # tie between C's end and E's start
        (50000, "E", 19000),  # past the last coding gene
    ],
)
def test_nearest_gene_inside_outside_and_ties(use_bundle, pos, gene, distance):
    use_bundle(HAND)
    got, out = nearest([f"1:{pos}"])
    assert got == {f"1:{pos}": gene}
    assert out["genes"]["distance"].to_list() == [distance]


def test_nearest_matches_brute_force_with_overlapping_genes():
    rng = np.random.default_rng(3)
    start = np.sort(rng.integers(0, 100_000, 60))
    end = start + rng.integers(0, 20_000, 60)
    order = np.lexsort((end, start))
    start, end = start[order], end[order]
    ch = enrichment._Chrom(0, start, end)
    pos = rng.integers(-5_000, 130_000, 2_000)
    gene, dist = ch.nearest(pos)
    d = np.maximum(np.maximum(start[None, :] - pos[:, None], pos[:, None] - end[None, :]), 0)
    assert (dist == d.min(axis=1)).all()
    assert (gene == d.argmin(axis=1)).all(), "ties go to the earliest gene in sort order"


def test_protein_coding_restricts_the_genes(use_bundle):
    use_bundle(HAND)
    assert nearest(["1:40000"])[0] == {"1:40000": "E"}
    got, out = nearest(["1:40000"], protein_coding=False)
    assert got == {"1:40000": "F"}
    assert out["parameters"]["gene_type"] == "all genes with coordinates"


def test_window_mode_takes_every_overlapping_gene(use_bundle):
    use_bundle(HAND)
    out = enrichment.genes_for_loci(["1:3500"], mode="window", window=1600)
    assert out["loci"]["gene_symbols"].to_list() == [["A", "B"]]
    assert out["genes"]["distance"].to_list() == [1500, 1500]
    out = enrichment.genes_for_loci(["1:3500"], mode="window", window=1400)
    assert out["loci"]["gene_symbols"].to_list() == [[]]
    assert out["parameters"]["loci_without_gene"] == 1
    out = enrichment.genes_for_loci(["1:12500"], mode="window", window=0)
    assert out["loci"]["gene_symbols"].to_list() == [["C", "D"]]


def test_merging_is_single_linkage_and_zero_disables_it(use_bundle):
    use_bundle(HAND)
    leads = ["1:1000", "1:401000", "1:801000", "1:1401000", "23:1500"]
    out = enrichment.genes_for_loci(leads, merge_distance=500_000)
    loci = out["loci"]
    assert loci["n_leads"].to_list() == [3, 1, 1]
    assert loci["locus_id"].to_list() == ["1:1000-801000", "1:1401000-1401000", "X:1500-1500"]
    assert loci["lead_ids"].to_list()[0] == ["1:1000", "1:401000", "1:801000"]
    # the union over the chain's leads: E is nearest to the two far ones
    assert loci["gene_symbols"].to_list()[0] == ["A", "E"]
    assert enrichment.genes_for_loci(leads, merge_distance=0)["parameters"]["n_loci"] == 5
    assert enrichment.genes_for_loci(leads, merge_distance=399_999)["parameters"]["n_loci"] == 5


def test_identical_positions_without_merging_stay_distinct_loci(use_bundle):
    use_bundle(HAND)
    out = enrichment.genes_for_loci(["1:1500", "1:1500:A:G"], merge_distance=0)
    assert out["loci"]["locus_id"].n_unique() == 2


def test_mhc_leads_are_dropped_and_counted(use_bundle):
    use_bundle(HAND + [("H", 6, 30_000_000, 30_001_000)])
    leads = ["6:24999999", "6:25000000", "6:34000000", "6:34000001"]
    out = enrichment.genes_for_loci(leads, exclude_mhc=True, merge_distance=0)
    assert out["parameters"]["leads_dropped_mhc"] == 2
    assert out["loci"]["lead_ids"].list.first().to_list() == ["6:24999999", "6:34000001"]
    assert enrichment.genes_for_loci(leads)["parameters"]["leads_dropped_mhc"] == 0


@pytest.mark.parametrize("chrom", ["X", "chrX", "23", 23, "x"])
def test_x_is_read_in_every_spelling(use_bundle, chrom):
    use_bundle(HAND)
    out = enrichment.genes_for_loci(pl.DataFrame({"chr": [chrom], "pos": [1500]}))
    assert out["loci"]["chr"].to_list() == [23]
    assert out["loci"]["gene_symbols"].to_list() == [["XG"]]


@pytest.mark.parametrize("variant", ["chr1_1500_A_G", "1:1500:A:G", "1:1500", "chr1:1500"])
def test_variant_strings(use_bundle, variant):
    use_bundle(HAND)
    assert nearest([variant])[0] == {variant: "A"}


def test_ids_are_carried_through(use_bundle):
    use_bundle(HAND)
    df = pl.DataFrame({"chr": [1, 1], "pos": [1500, 30500], "cs_id": ["cs1", "cs2"], "x": [1, 2]})
    out = enrichment.genes_for_loci(df, merge_distance=0)
    assert out["loci"]["lead_ids"].to_list() == [["cs1"], ["cs2"]]
    out = enrichment.genes_for_loci(df, merge_distance=0, id_col="x")
    assert out["loci"]["lead_ids"].to_list() == [["1"], ["2"]]
    out = enrichment.genes_for_loci(df.drop("cs_id", "x"), merge_distance=0)
    assert out["loci"]["lead_ids"].to_list() == [["1:1500"], ["1:30500"]]


def test_parameters_name_the_coordinates(use_bundle):
    use_bundle(HAND)
    p = enrichment.genes_for_loci(["1:1500"])["parameters"]
    assert p["coordinates"] == {"release_id": "syn.1", "gencode": "99", "genome_build": "GRCh38"}
    assert p["gene_type"] == enrichment._PROTEIN_CODING


@pytest.mark.parametrize(
    "loci, kwargs, message",
    [
        ("1:1500", {}, "must be a polars DataFrame"),
        (["rs123"], {}, "cannot read 'rs123'"),
        (["27:100"], {}, "unrecognised chromosome"),
        (pl.DataFrame({"chr": ["chr27"], "pos": [1]}), {}, "unrecognised chromosome"),
        (pl.DataFrame({"chrom": [1], "pos": [1]}), {}, "needs columns 'chr' and 'pos'"),
        (pl.DataFrame({"chr": [1], "pos": [None]}), {}, "null chr or pos"),
        (pl.DataFrame({"chr": [1], "pos": [1.5]}), {}, "must be integers"),
        (pl.DataFrame({"chr": [1], "pos": [0]}), {}, ">= 1"),
        (pl.DataFrame({"chr": [1], "pos": [1]}), {"id_col": "nope"}, "not a column"),
        ([], {}, "empty"),
        (["1:1500"], {"mode": "closest"}, "mode must be"),
        (["1:1500"], {"window": -1}, "window must be"),
        (["6:30000000"], {"exclude_mhc": True}, "no leads remain"),
    ],
)
def test_genes_for_loci_usage_errors(use_bundle, loci, kwargs, message):
    use_bundle(HAND)
    with pytest.raises(GeneticsUsageError, match=message):
        enrichment.genes_for_loci(loci, **kwargs)


# --- the Poisson-binomial tail ------------------------------------------------------------


def test_poisson_binomial_tail_equals_enumeration():
    rng = np.random.default_rng(0)
    n = 9
    q = rng.uniform(0, 0.6, (n, 7))
    q[:, 6] = 1e-4  # a tiny tail
    obs = np.array([0, 1, 2, 3, 5, 9, 4])
    want = np.zeros(7)
    for draw in product((0, 1), repeat=n):
        x = np.array(draw)[:, None]
        prob = np.prod(np.where(x == 1, q, 1 - q), axis=0)
        want += np.where(x.sum() >= obs, prob, 0)
    got = enrichment._poisson_binomial_sf(q, obs)
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=0)
    assert 0 < got[6] < 1e-12


def test_poisson_binomial_tail_matches_monte_carlo():
    rng = np.random.default_rng(1)
    q = rng.uniform(0, 0.3, (40, 3))
    obs = np.array([3, 6, 10])
    draws = (rng.random((200_000, 40, 1)) < q[None]).sum(axis=1)
    mc = (draws >= obs).mean(axis=0)
    got = enrichment._poisson_binomial_sf(q, obs)
    se = np.sqrt(got * (1 - got) / 200_000)
    assert (np.abs(mc - got) < 4 * se + 1e-6).all(), (mc, got)


def test_poisson_binomial_chunks_with_different_observations_agree():
    rng = np.random.default_rng(2)
    q = rng.uniform(0, 0.5, (12, 50))
    obs = rng.integers(0, 10, 50)
    whole = enrichment._poisson_binomial_sf(q, obs)
    chunked = enrichment._poisson_binomial_sf(q, obs, chunk=3)
    np.testing.assert_allclose(chunked, whole, rtol=1e-12)


# --- locus_enrichment ----------------------------------------------------------------------


def genome(rng):
    """Three chromosomes of different gene density, ~100 genes each."""
    genes = []
    for c, gap in ((1, (20_000, 30_000)), (2, (50_000, 80_000)), (3, (20_000, 150_000))):
        pos = 100_000
        for i in range(100):
            genes.append((f"G{c}_{i}", c, pos, pos + int(rng.integers(2_000, 30_000))))
            pos += int(rng.integers(*gap))
    return genes


def grid_points(genes, step):
    out = []
    for c in sorted({g[1] for g in genes}):
        s = [g for g in genes if g[1] == c]
        lo, hi = min(g[2] for g in s), max(g[3] for g in s)
        out += [(c, p) for p in range(lo, hi + 1, step)]
    return out


def test_calibration_random_loci_do_not_pile_up_near_zero(use_bundle):
    rng = np.random.default_rng(11)
    genes = genome(rng)
    names = [g[0] for g in genes]
    sets = {
        f"S{i}": list(rng.choice(names, int(rng.integers(8, 40)), replace=False)) for i in range(40)
    }
    use_bundle(genes, sets)
    grid = grid_points(genes, 5_000)
    pvals = []
    for _ in range(25):
        pick = rng.choice(len(grid), 25, replace=False)
        df = pl.DataFrame({"chr": [grid[i][0] for i in pick], "pos": [grid[i][1] for i in pick]})
        res = enrichment.locus_enrichment(
            df,
            "syn",
            merge_distance=0,
            density_window=100_000,
            grid_step=5_000,
            density_bins=5,
            exclude_mhc=False,
        )
        pvals += res["results"]["pval"].to_list()
    pvals = np.array(pvals)
    # 1,000 p-values: a calibrated (here discrete, hence conservative) test puts at most
    # 5% below 0.05, give or take three binomial standard errors (0.021)
    assert (pvals < 0.05).mean() <= 0.05 + 0.021
    assert (pvals < 0.01).mean() <= 0.01 + 0.01
    assert np.mean(pvals) > 0.45


def planted(rng):
    genes = genome(rng)
    target = [g[0] for g in genes if g[1] == 2][10:40:3]
    sets = {"TARGET": target, "OTHER": [g[0] for g in genes if g[1] == 3][:10]}
    return genes, sets, target


def test_a_planted_enrichment_is_detected(use_bundle):
    genes, sets, target = planted(np.random.default_rng(5))
    use_bundle(genes, sets)
    where = {g[0]: g for g in genes}
    leads = [f"{where[t][1]}:{where[t][2] + 100}" for t in target[:8]]
    leads += ["1:500000", "3:2000000", "1:2000000"]
    res = enrichment.locus_enrichment(
        leads,
        "syn",
        merge_distance=0,
        density_window=100_000,
        grid_step=5_000,
        exclude_mhc=False,
    )
    top = res["results"].row(0, named=True)
    assert top["set_id"] == "TARGET"
    assert top["n_loci_hit"] == 8
    assert top["fold_enrichment"] > 5
    assert top["pval"] < 1e-6 and top["padj"] < 0.05
    assert sorted(top["hit_symbols"]) == sorted(target[:8])
    assert res["attribution"]["release_id"] == "syn.1"


def test_a_locus_with_three_genes_of_one_set_counts_once(use_bundle):
    genes = [("P", 1, 100_000, 101_000), ("Q", 1, 103_000, 104_000), ("R", 1, 106_000, 107_000)]
    genes += [(f"N{i}", 1, 200_000 + 50_000 * i, 201_000 + 50_000 * i) for i in range(40)]
    use_bundle(genes, {"PQR": ["P", "Q", "R", "N0", "N1"]})
    res = enrichment.locus_enrichment(
        ["1:103500"],
        "syn",
        mode="window",
        window=5_000,
        density_window=50_000,
        grid_step=1_000,
        exclude_mhc=False,
    )
    row = res["results"].row(0, named=True)
    assert row["n_loci_hit"] == 1
    assert sorted(row["hit_symbols"]) == ["P", "Q", "R"]
    assert row["hit_loci"] == ["1:103500-103500"]
    assert row["expected"] <= 1


def test_leads_of_one_locus_count_once(use_bundle):
    genes, sets, target = planted(np.random.default_rng(5))
    use_bundle(genes, sets)
    where = {g[0]: g for g in genes}
    a, b = where[target[0]], where[target[1]]
    leads = [f"2:{a[2]}", f"2:{b[2]}"]
    kw = {"density_window": 100_000, "grid_step": 5_000, "exclude_mhc": False}
    merged = enrichment.locus_enrichment(leads, "syn", merge_distance=1_000_000, **kw)
    apart = enrichment.locus_enrichment(leads, "syn", merge_distance=0, **kw)

    def hits(r):
        return r["results"].filter(pl.col("set_id") == "TARGET")["n_loci_hit"].item()

    assert hits(merged) == 1 and merged["parameters"]["n_loci_multi_lead"] == 1
    assert hits(apart) == 2


def density_genome():
    """chr1: a dense cluster (a gene every 5 kb over 1 Mb) and a sparse arm (every 400 kb).
    SPARSE is every gene of the sparse arm."""
    dense = [(f"D{i}", 1, 1_000_000 + 5_000 * i, 1_002_000 + 5_000 * i) for i in range(200)]
    sparse_ = [(f"S{i}", 1, 3_000_000 + 400_000 * i, 3_010_000 + 400_000 * i) for i in range(10)]
    return dense + sparse_, {"SPARSE": [g[0] for g in sparse_]}


@pytest.mark.parametrize("bins, matched", [(10, True), (1, False)])
def test_density_matching_changes_q(use_bundle, bins, matched):
    genes, sets = density_genome()
    use_bundle(genes, sets)
    kw = {
        "merge_distance": 0,
        "density_window": 300_000,
        "grid_step": 5_000,
        "density_bins": bins,
        "exclude_mhc": False,
        "min_size": 1,
    }

    def expected(lead):
        r = enrichment.locus_enrichment([lead], "syn", **kw)["results"]
        return r.filter(pl.col("set_id") == "SPARSE")["expected"].item()

    in_sparse, in_dense = expected("1:5000000"), expected("1:1500000")
    if matched:
        assert in_sparse > 0.9 and in_dense < 0.05
    else:
        assert in_sparse == pytest.approx(in_dense, abs=1e-3)


def test_parameters_describe_the_null(use_bundle):
    genes, sets = density_genome()
    use_bundle(genes, sets)
    res = enrichment.locus_enrichment(
        ["1:1500000", "6:30000000", "24:100", "23:5"], "syn", grid_step=5_000, min_size=1
    )
    p = res["parameters"]
    assert (p["leads_given"], p["leads_dropped_mhc"], p["leads_dropped_chromosome"]) == (4, 1, 2)
    assert p["null"]["grid_chromosomes"] == ["1"]
    assert sum(p["null"]["placements_per_bin"]) == p["null"]["grid_points"]
    assert p["null"]["approximations"]
    assert res["loci"].columns[-2:] == ["density", "density_bin"]


def test_locus_enrichment_takes_a_mapping_resolved_through_the_bundle(use_bundle):
    genes, sets = density_genome()
    use_bundle(genes, sets)
    res = enrichment.locus_enrichment(
        ["1:5000000"], {"mine": sets["SPARSE"][:10] + ["NOPE"]}, grid_step=5_000, min_size=1
    )
    row = res["results"].row(0, named=True)
    assert row["size"] == 10 and row["hit_symbols"] == ["S5"]
    assert res["parameters"]["id_resolution"]["gene_sets"]["unresolved"] == 1
    assert "attribution" not in res


def test_locus_enrichment_over_the_fixture_collection(monkeypatch):
    monkeypatch.setenv(enrichment.GENESETS_ENV, str(FIXTURE))
    enrichment._BUNDLES.clear()
    res = enrichment.locus_enrichment(
        ["1:2200", "1:3200", "1:7200"],
        "coll.a",
        merge_distance=0,
        grid_step=100,
        density_window=2_000,
    )
    assert set(res["results"].columns) >= {"name", "collection", "hit_symbols", "padj"}
    assert res["results"].filter(pl.col("set_id") == "A_EARLY")["n_loci_hit"].item() == 3
    assert res["attribution"]["collections"][0]["collection"] == "coll.a"
    enrichment._BUNDLES.clear()


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"grid_step": 0}, "grid_step must be"),
        ({"density_bins": 0}, "density_bins must be"),
        ({"density_window": -1}, "density_window must be"),
        ({"min_size": 0}, "min_size"),
    ],
)
def test_locus_enrichment_usage_errors(use_bundle, kwargs, message):
    genes, sets = density_genome()
    use_bundle(genes, sets)
    with pytest.raises(GeneticsUsageError, match=message):
        enrichment.locus_enrichment(["1:5000000"], "syn", **kwargs)


# --- simplify ------------------------------------------------------------------------------

MAPPING = {
    "BIG": [f"g{i}" for i in range(10)],
    "NESTED": [f"g{i}" for i in range(8)],  # Jaccard 0.8 with BIG
    "HALF": [f"g{i}" for i in range(5)],  # exactly 0.5 with BIG, 0.625 with NESTED
    "DISJOINT": [f"g{i}" for i in range(20, 30)],
    "WEAK": [f"g{i}" for i in range(30, 40)],
}


def result(pvals, padj=None):
    ids = list(pvals)
    return {
        "results": pl.DataFrame(
            {
                "set_id": ids,
                "pval": [pvals[s] for s in ids],
                "padj": [padj[s] for s in ids] if padj else [pvals[s] for s in ids],
            }
        ),
        "parameters": {"x": 1},
    }


def test_simplify_absorbs_nested_and_keeps_disjoint():
    r = result({"NESTED": 1e-5, "BIG": 1e-4, "HALF": 1e-3, "DISJOINT": 1e-2, "WEAK": 0.2})
    s = enrichment.simplify(r, gene_sets=MAPPING)
    assert s["results"]["set_id"].to_list() == ["NESTED", "DISJOINT"]
    assert s["results"]["absorbed_ids"].to_list() == [["BIG", "HALF"], []]
    assert s["results"]["n_absorbed"].to_list() == [2, 0]
    assert dict(s["clusters"].select("set_id", "representative").iter_rows()) == {
        "NESTED": "NESTED",
        "BIG": "NESTED",
        "HALF": "NESTED",
        "DISJOINT": "DISJOINT",
    }
    assert s["parameters"]["x"] == 1
    assert s["parameters"]["simplify"]["cutoff"] == 0.7
    assert s["parameters"]["simplify"]["basis_used"] == "members", "no hit column to use"
    assert r["results"].height == 5, "the input is not modified"


def test_simplify_cutoff_is_inclusive():
    r = result({"BIG": 1e-5, "HALF": 1e-4})
    kw = {"gene_sets": MAPPING, "measure": "jaccard"}
    assert enrichment.simplify(r, cutoff=0.5, **kw)["results"].height == 1
    assert enrichment.simplify(r, cutoff=0.5001, **kw)["results"].height == 2
    # HALF lies inside BIG: BIG holds all of it
    assert enrichment.simplify(r, gene_sets=MAPPING, cutoff=1.0)["results"].height == 1


def test_simplify_order_is_by_p_then_padj_then_set_id():
    r = result({"NESTED": 1e-3, "BIG": 1e-3}, padj={"NESTED": 1e-3, "BIG": 1e-3})
    assert enrichment.simplify(r, gene_sets=MAPPING)["results"]["set_id"].to_list() == ["BIG"]
    r = result({"NESTED": 1e-3, "BIG": 1e-3}, padj={"NESTED": 1e-3, "BIG": 2e-3})
    assert enrichment.simplify(r, gene_sets=MAPPING)["results"]["set_id"].to_list() == ["NESTED"]


def test_simplify_needs_members_for_a_mapping_result():
    r = result({"BIG": 1e-5})
    with pytest.raises(GeneticsUsageError, match="pass the same mapping"):
        enrichment.simplify(r)
    with pytest.raises(GeneticsUsageError, match="no members for"):
        enrichment.simplify(r, gene_sets={"X": ["g1"]})
    with pytest.raises(GeneticsUsageError, match="cutoff"):
        enrichment.simplify(r, gene_sets=MAPPING, cutoff=0)


@pytest.fixture
def fixture_bundle(monkeypatch):
    monkeypatch.setenv(enrichment.GENESETS_ENV, str(FIXTURE))
    enrichment._BUNDLES.clear()
    yield
    enrichment._BUNDLES.clear()


def test_simplify_on_ora_over_bundled_collections(fixture_bundle):
    res = enrichment.ora([f"GENE{i}" for i in range(1, 9)], ["coll.a", "coll.b"])
    s = enrichment.simplify(res, padj=1.0)
    # B_FIRST (genes 1-6) lies inside A_EARLY (1-8): Jaccard 0.75
    clusters = dict(s["clusters"].select("set_id", "representative").iter_rows())
    assert clusters["B_FIRST"] == clusters["A_EARLY"]
    rep = s["results"].filter(pl.col("set_id") == clusters["A_EARLY"]).row(0, named=True)
    assert rep["n_absorbed"] >= 1 and rep["absorbed_names"]
    assert s["parameters"]["simplify"]["genes_from"] == "the result's overlap_genes"
    assert "attribution" in s
    m = enrichment.simplify(res, padj=1.0, basis="members", measure="jaccard", cutoff=0.5)
    assert m["parameters"]["simplify"]["genes_from"] == "full bundled membership"
    # by full membership A_MIDDLE (5-14) is not A_EARLY's (1-8): Jaccard 4/14
    clusters = dict(m["clusters"].select("set_id", "representative").iter_rows())
    assert clusters["A_MIDDLE"] != clusters["A_EARLY"]


def test_simplify_on_gsea_output(fixture_bundle):
    stats = {f"GENE{i}": float(40 - i) for i in range(1, 30)}
    res = enrichment.gsea_preranked(stats, ["coll.a", "coll.b"], nperm=200, seed=1)
    s = enrichment.simplify(res, padj=1.0)
    assert s["clusters"].height == res["results"].height
    assert set(s["results"]["set_id"]) <= set(res["results"]["set_id"])
    assert s["results"]["n_absorbed"].sum() + s["results"].height == s["clusters"].height


def test_simplify_with_nothing_significant(fixture_bundle):
    res = enrichment.ora(["GENE1"], "coll.a")
    s = enrichment.simplify(res, padj=1e-12)
    assert s["results"].height == 0 and s["clusters"].height == 0
    assert "n_absorbed" in s["results"].columns


def test_simplify_overlap_absorbs_a_small_set_nested_in_a_large_one():
    mapping = {"LARGE": [f"g{i}" for i in range(40)], "SMALL": ["g1", "g2", "g3", "g4"]}
    r = result({"LARGE": 1e-5, "SMALL": 1e-4})
    assert enrichment.simplify(r, gene_sets=mapping)["results"]["set_id"].to_list() == ["LARGE"]
    jac = enrichment.simplify(r, gene_sets=mapping, measure="jaccard")
    assert jac["results"].height == 2


def test_simplify_hits_basis_compares_the_genes_driving_each_result():
    # members identical, hits disjoint: by hits they are different findings
    r = result({"A": 1e-5, "B": 1e-4})
    r["results"] = r["results"].with_columns(
        hit_genes=pl.Series([["g1", "g2"], ["g3", "g4"]], dtype=pl.List(pl.Utf8))
    )
    same = {"A": ["g1", "g2", "g3", "g4"], "B": ["g1", "g2", "g3", "g4"]}
    assert enrichment.simplify(r)["results"].height == 2
    assert enrichment.simplify(r, basis="members", gene_sets=same)["results"].height == 1


def test_simplify_on_competitive_falls_back_to_members(fixture_bundle):
    z = {f"GENE{i}": (3.0 if i <= 8 else 0.0) + 0.01 * i for i in range(1, 30)}
    res = enrichment.competitive(z, ["coll.a", "coll.b"])
    s = enrichment.simplify(res, padj=1.0)
    p = s["parameters"]["simplify"]
    assert (p["basis"], p["basis_used"], p["genes_from"]) == (
        "hits",
        "members",
        "full bundled membership",
    )


@pytest.mark.parametrize(
    "drop, kwargs, message",
    [
        ("pval", {}, r"missing \['pval'\]"),
        (None, {"basis": "genes"}, "basis must be"),
        (None, {"measure": "dice"}, "measure must be"),
        (None, {"gene_sets": ["x"]}, "must be a mapping"),
    ],
)
def test_simplify_usage_errors(drop, kwargs, message):
    r = result({"BIG": 1e-5})
    if drop:
        r["results"] = r["results"].drop(drop)
    with pytest.raises(GeneticsUsageError, match=message):
        enrichment.simplify(r, **{"gene_sets": MAPPING, **kwargs})


# --- locus_enrichment against an independent reference ---------------------------------


def mixed_genome(rng):
    """chr1 and chr2, coding genes with non-coding ones interleaved (some nearer leads)."""
    genes = []
    for c, gap in ((1, (15_000, 40_000)), (2, (30_000, 120_000))):
        pos = 100_000
        for i in range(80):
            kind = "non-coding RNA" if i % 4 == 2 else "protein-coding gene"
            genes.append((f"G{c}_{i}", c, pos, pos + int(rng.integers(1_000, 20_000)), kind))
            pos += int(rng.integers(*gap))
    return genes


def reference_expected(genes, sets, leads, step, dw, n_bins):
    """locus_enrichment's null for single-lead loci in nearest mode, written out directly:
    brute-force nearest coding gene, quantile strata of the autosomal grid, the locus's own
    position counted as one placement."""
    coding = sorted((g for g in genes if g[4] == "protein-coding gene"), key=lambda g: g[1:4])

    def on(c):
        return [g for g in coding if g[1] == c]

    def nearest(c, p):
        d = [(max(g[2] - p, p - g[3], 0), k) for k, g in enumerate(on(c))]
        return on(c)[min(d)[1]][0]

    def density(c, p):
        return sum(g[2] <= p + dw and g[3] >= p - dw for g in on(c))

    grid = []
    for c in sorted({g[1] for g in coding}):
        lo, hi = on(c)[0][2], max(g[3] for g in on(c))
        grid += [(c, p, density(c, p)) for p in range(lo, hi + 1, step) if density(c, p) > 0]
    den = np.array([d for _, _, d in grid])
    edges = np.unique(np.quantile(den, np.linspace(0, 1, n_bins + 1)[1:-1], method="lower"))
    bins = np.searchsorted(edges, den, "right")
    near = [nearest(c, p) for c, p, _ in grid]
    out, hits = {}, {}
    for s_id, members in sets.items():
        mem = set(members)
        exp, n_hit = 0.0, 0
        for c, p in leads:
            b = np.searchsorted(edges, density(c, p), "right")
            cand = [g for g, bb in zip(near, bins, strict=True) if bb == b]
            own = nearest(c, p) in mem
            n_hit += own
            exp += (sum(g in mem for g in cand) + own) / (len(cand) + 1)
        out[s_id], hits[s_id] = exp, n_hit
    return out, hits, edges, [density(c, p) for c, p in leads]


def test_locus_enrichment_matches_a_direct_reference(use_bundle):
    rng = np.random.default_rng(21)
    genes = mixed_genome(rng)
    names = [g[0] for g in genes]
    sets = {
        f"S{i}": list(rng.choice(names, int(rng.integers(10, 30)), replace=False))
        for i in range(12)
    }
    use_bundle(genes, sets)
    leads = [(1, int(p)) for p in rng.integers(150_000, 2_000_000, 8)]
    leads += [(2, int(p)) for p in rng.integers(150_000, 5_000_000, 8)]
    # inside non-coding genes, whose nearest coding gene is what must be mapped
    leads += [(g[1], g[2] + 50) for g in genes if g[4] != "protein-coding gene"][:4]
    step, dw, n_bins = 5_000, 100_000, 6
    res = enrichment.locus_enrichment(
        [f"{c}:{p}" for c, p in leads],
        "syn",
        merge_distance=0,
        density_window=dw,
        grid_step=step,
        density_bins=n_bins,
        exclude_mhc=False,
        min_size=1,
    )
    want, hits, edges, lead_density = reference_expected(genes, sets, leads, step, dw, n_bins)
    assert res["parameters"]["null"]["density_bin_edges"] == edges.tolist()
    # a locus whose density equals a bin edge belongs to the bin starting there
    assert set(lead_density) & set(edges.tolist())
    got = res["results"]
    assert dict(zip(got["set_id"], got["n_loci_hit"], strict=True)) == hits
    for s_id, e in zip(got["set_id"], got["expected"], strict=True):
        assert e == pytest.approx(want[s_id], rel=1e-12), s_id
    coding = {g[0] for g in genes if g[4] == "protein-coding gene"}
    dropped = sum(g not in coding for m in sets.values() for g in m)
    assert res["parameters"]["set_members_dropped"]["memberships"] == dropped


def test_a_multi_lead_locus_is_centred_on_its_midpoint(use_bundle):
    genes = [("A", 1, 1_000_000, 1_001_000), ("B", 1, 1_400_000, 1_401_000)]
    genes += [(f"C{k}", 1, 1_200_000 + 10_000 * k, 1_201_000 + 10_000 * k) for k in range(3)]
    genes += [(f"F{k}", 1, 2_000_000 + 30_000 * k, 2_001_000 + 30_000 * k) for k in range(30)]
    use_bundle(genes, {"S": ["A", "B", "C0", "F1", "F2"]})
    res = enrichment.locus_enrichment(
        ["1:1000000", "1:1400000"],
        "syn",
        density_window=50_000,
        grid_step=5_000,
        exclude_mhc=False,
    )
    assert res["loci"]["density"].to_list() == [3], "C0-C2 lie within 50 kb of 1,200,000"


def x_genome():
    genes = [(f"A{i}", 1, 100_000 + 30_000 * i, 110_000 + 30_000 * i) for i in range(60)]
    genes += [(f"B{i}", 2, 100_000 + 90_000 * i, 105_000 + 90_000 * i) for i in range(40)]
    genes += [(f"X{i}", 23, 100_000 + 10_000 * i, 104_000 + 10_000 * i) for i in range(50)]
    sets = {"SA": [f"A{i}" for i in range(0, 60, 3)], "SB": [f"B{i}" for i in range(0, 40, 2)]}
    return genes, sets


def test_autosomal_results_do_not_depend_on_an_x_locus(use_bundle):
    genes, sets = x_genome()
    use_bundle(genes, sets)
    auto = ["1:400000", "1:1200000", "2:900000", "2:2500000"]
    kw = {"merge_distance": 0, "density_window": 100_000, "grid_step": 5_000, "min_size": 1}
    without = enrichment.locus_enrichment(auto, "syn", **kw)
    # the X locus hits no set, so it adds nothing to any count or q
    with_x = enrichment.locus_enrichment(auto + ["X:300000"], "syn", **kw)
    cols = ["set_id", "n_loci_hit", "expected", "pval"]
    assert (
        with_x["results"]
        .select(cols)
        .sort("set_id")
        .equals(without["results"].select(cols).sort("set_id"))
    )
    assert with_x["loci"]["density_bin"].to_list()[:4] == without["loci"]["density_bin"].to_list()
    assert (
        with_x["parameters"]["null"]["placements_per_bin"]
        == (without["parameters"]["null"]["placements_per_bin"])
    )
    assert sum(with_x["parameters"]["null"]["placements_per_bin_x"]) > 0


@pytest.mark.parametrize("exclude_mhc", [False, True])
def test_multi_lead_placements_stay_on_the_gene_span_and_out_of_the_mhc(use_bundle, exclude_mhc):
    genes = [(f"G{i}", 6, 23_000_000 + 20_000 * i, 23_005_000 + 20_000 * i) for i in range(150)]
    use_bundle(genes, {"S": [g[0] for g in genes[:10]]})
    res = enrichment.locus_enrichment(
        ["6:23500000", "6:23800000"],
        "syn",
        density_window=100_000,
        grid_step=10_000,
        density_bins=1,
        exclude_mhc=exclude_mhc,
        min_size=1,
    )
    lo, hi = genes[0][2], genes[-1][3]
    mhc = (25_000_000, 34_000_000)
    grid = range(lo, hi + 1, 10_000)
    if exclude_mhc:
        grid = [p for p in grid if not mhc[0] <= p <= mhc[1]]
    dropped = 0
    for p in grid:
        at = (p - 150_000, p + 150_000)
        bad = any(not lo <= a <= hi for a in at)
        bad |= exclude_mhc and any(mhc[0] <= a <= mhc[1] for a in at)
        dropped += bad
    assert dropped > 0
    assert res["parameters"]["null"]["multi_lead_placements_dropped"] == dropped


def test_an_x_locus_in_an_empty_stratum_takes_the_nearest_filled_one(use_bundle):
    # chr1: 1 Mb segments whose density within 20 kb runs 1..20, so the autosomal edges spread
    genes, start = [], 100_000
    for k in range(1, 21):
        step = 40_000 // k
        genes += [
            (f"A{k}_{j}", 1, start + step * j, start + step * j + 500)
            for j in range(1_000_000 // step)
        ]
        start += 1_000_000
    # X: isolated genes, one dense block (densest stratum), and a 16-gene cluster between grid
    # points, so no X placement shares the lead's stratum
    genes += [(f"X{i}", 23, 100_000 + 300_000 * i, 101_000 + 300_000 * i) for i in range(15)]
    genes += [(f"XD{i}", 23, 6_000_000 + 1_500 * i, 6_000_500 + 1_500 * i) for i in range(600)]
    genes += [(f"XC{i}", 23, 5_030_000 + 2_000 * i, 5_030_500 + 2_000 * i) for i in range(16)]
    use_bundle(genes, {"XC": [f"XC{i}" for i in range(16)]})
    r = enrichment.locus_enrichment(
        ["23:5045000", "1:300000"],
        "syn",
        merge_distance=0,
        density_window=20_000,
        grid_step=100_000,
        density_bins=10,
        exclude_mhc=False,
        min_size=1,
    )
    p = r["parameters"]["null"]
    filled = np.flatnonzero(p["placements_per_bin_x"])
    x = r["loci"].filter(pl.col("chr") == 23).row(0, named=True)
    own = int(np.searchsorted(p["density_bin_edges"], x["density"], "right"))
    assert own not in filled and filled.size > 1
    nearest = int(filled[np.argmin(np.abs(filled - own))])
    assert nearest != filled[0], "the scenario must tell nearest from lowest apart"
    assert x["density_bin"] == nearest


def test_containment_does_not_absorb_a_set_into_a_few_of_its_hits():
    # SMALL (best p) has 2 hits, both inside BIG's 10: SMALL explains 2/10 of BIG
    r = result({"SMALL": 1e-6, "BIG": 1e-5})
    r["results"] = r["results"].with_columns(
        hit_genes=pl.Series([["g1", "g2"], [f"g{i}" for i in range(1, 11)]])
    )
    assert enrichment.simplify(r)["results"]["set_id"].to_list() == ["SMALL", "BIG"]
    overlap = enrichment.simplify(r, measure="overlap")
    assert overlap["results"]["set_id"].to_list() == ["SMALL"]
    # the other way round, BIG explains all of SMALL under either measure
    r = result({"SMALL": 1e-5, "BIG": 1e-6})
    r["results"] = r["results"].with_columns(
        hit_genes=pl.Series([["g1", "g2"], [f"g{i}" for i in range(1, 11)]])
    )
    assert enrichment.simplify(r)["results"]["set_id"].to_list() == ["BIG"]
