"""genetics.enrichment over a staged gene set bundle: discovery, lazy per-collection loading,
collection names as `gene_sets`, identifier resolution and attribution.

The bundle under tests/fixtures/genesets/ is written by make_bundle.py there: 30 genes
ENSG00000000001..32 (symbols GENE1..GENE29, C1orf99, Zed1 and ZED1, Entrez 1001..1032 except
GENE29; genes 31 and 32 are in no set),
collection coll.a (A_EARLY 1-8, A_MIDDLE 5-14, A_LATE 20-26) and coll.b (B_FIRST 1-6,
B_SECOND 10-20).
"""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from genetics_mcp_server.sdk import enrichment
from genetics_mcp_server.sdk.errors import GeneticsUsageError

BUNDLE = Path(__file__).parent / "fixtures" / "genesets"


def ens(i: int) -> str:
    return f"ENSG{i:011d}"


def sym(i: int) -> str:
    return "C1orf99" if i == 30 else f"GENE{i}"


@pytest.fixture(autouse=True)
def bundle(monkeypatch):
    monkeypatch.setenv(enrichment.GENESETS_ENV, str(BUNDLE))
    enrichment._BUNDLES.clear()
    yield
    enrichment._BUNDLES.clear()


def loaded() -> set[str]:
    return set(enrichment._BUNDLES[BUNDLE]._members)


def test_the_fixture_bundle_is_marked_as_one():
    assert json.loads((BUNDLE / "manifest.json").read_text())["fixture"] is True


@pytest.mark.parametrize(
    "call",
    [
        lambda: enrichment.collections(),
        lambda: enrichment.gene_sets("coll.a"),
        lambda: enrichment.ora(["GENE1"], "coll.a"),
        lambda: enrichment.gsea_preranked({"GENE1": 1.0}, "coll.a"),
        lambda: enrichment.competitive({"GENE1": 1.0}, ["coll.a"]),
    ],
)
def test_an_absent_bundle_names_the_variable(monkeypatch, tmp_path, call):
    monkeypatch.setenv(enrichment.GENESETS_ENV, str(tmp_path / "nothing-here"))
    with pytest.raises(GeneticsUsageError, match="GENETICS_GENESETS_DIR"):
        call()


def test_collections_lists_what_the_bundle_holds():
    c = enrichment.collections()
    assert c["collection"].to_list() == ["coll.a", "coll.b"]
    assert c["n_sets"].to_list() == [3, 2]
    assert c["licence"].to_list() == ["CC-BY-4.0", "LicenseRef-HPO"]
    assert c["go_release"].to_list() == ["GO-basic obo file released on 2099-01-01", None]
    assert c["hpo_release"].to_list() == [None, "2099-01-02"]
    assert c["upstream_release"].to_list() == ["1.0", "1.0"]
    assert c["release_id"].unique().to_list() == ["20990101.fixture"]
    assert c["attribution"].str.contains("HPO version 2099-01-02").to_list() == [False, True]
    assert loaded() == set(), "collections() must not read memberships"


def test_memberships_load_per_collection_on_first_use():
    enrichment.gene_sets("coll.a")
    assert loaded() == {"coll.a"}
    first = enrichment._BUNDLES[BUNDLE]._members["coll.a"]
    enrichment.ora(["GENE1", "GENE2"], "coll.a")
    assert enrichment._BUNDLES[BUNDLE]._members["coll.a"] is first, "cached, not re-read"
    assert loaded() == {"coll.a"}
    enrichment.ora(["GENE1", "GENE2"], ["coll.b"])
    assert loaded() == {"coll.a", "coll.b"}


def test_gene_sets_returns_mapping_metadata_and_attribution():
    gs = enrichment.gene_sets("coll.a")
    assert gs["gene_sets"] == {
        "A_EARLY": [ens(i) for i in range(1, 9)],
        "A_LATE": [ens(i) for i in range(20, 27)],
        "A_MIDDLE": [ens(i) for i in range(5, 15)],
    }
    assert gs["sets"]["name"].to_list() == ["a early genes", "a late genes", "a middle genes"]
    assert gs["attribution"]["release_id"] == "20990101.fixture"
    by_symbol = enrichment.gene_sets(["coll.b", "coll.a"], ids="symbol")
    assert list(by_symbol["gene_sets"])[:2] == ["B_FIRST", "B_SECOND"]
    assert by_symbol["gene_sets"]["B_FIRST"] == [sym(i) for i in range(1, 7)]
    assert [c["collection"] for c in by_symbol["attribution"]["collections"]] == [
        "coll.b", "coll.a",
    ]


@pytest.mark.parametrize(
    "value, message",
    [
        ("nope", "no gene set collection 'nope'"),
        (["coll.a", "coll.a"], "named twice"),
        ([], "collection name"),
        (42, "collection name"),
    ],
)
def test_bad_collection_names_are_refused(value, message):
    with pytest.raises(GeneticsUsageError, match=message):
        enrichment.ora(["GENE1"], value)
    with pytest.raises(GeneticsUsageError, match=message):
        enrichment.gene_sets(value)


def test_gene_sets_refuses_an_unknown_id_kind():
    with pytest.raises(GeneticsUsageError, match="ids must be"):
        enrichment.gene_sets("coll.a", ids="entrez")


STATS_COLUMNS = {
    "ora": ["set_id", "size", "overlap", "expected", "fold_enrichment", "pval", "padj",
            "overlap_genes"],
    "gsea": ["set_id", "size", "es", "nes", "pval", "padj", "n_more_extreme", "leading_edge"],
    "competitive": ["set_id", "size", "beta", "se", "t", "pval", "padj"],
}


def both_ways(kind: str, names):
    """The same analysis by collection name with mixed ids, and by mapping with Ensembl."""
    mapping = enrichment.gene_sets(names)["gene_sets"]
    if kind == "ora":
        lst = [sym(1), str(1002), ens(3) + ".7", "gene4", sym(5), sym(10), sym(11)]
        uni = [sym(i) for i in range(1, 31)]
        named = enrichment.ora(lst, names, universe=uni, universe_mode="given")
        plain = enrichment.ora(
            [ens(i) for i in (1, 2, 3, 4, 5, 10, 11)], mapping,
            universe=[ens(i) for i in range(1, 31)], universe_mode="given",
        )
    else:
        values = np.linspace(3, -2.8, 30)
        keyed = {sym(i) if i % 3 else str(1000 + i): v
                 for i, v in zip(range(1, 31), values, strict=True)}
        plain_keyed = {ens(i): v for i, v in zip(range(1, 31), values, strict=True)}
        if kind == "gsea":
            named = enrichment.gsea_preranked(keyed, names, nperm=200, seed=3)
            plain = enrichment.gsea_preranked(plain_keyed, mapping, nperm=200, seed=3)
        else:
            named = enrichment.competitive(keyed, names)
            plain = enrichment.competitive(plain_keyed, mapping)
    return named, plain


@pytest.mark.parametrize("kind", ["ora", "gsea", "competitive"])
@pytest.mark.parametrize("names", ["coll.a", ["coll.a", "coll.b"]])
def test_a_collection_name_gives_the_same_statistics_as_its_mapping(kind, names):
    named, plain = both_ways(kind, names)
    assert_frame_equal(named["results"].select(STATS_COLUMNS[kind]), plain["results"])
    assert named["parameters"]["sets_given"] == (3 if names == "coll.a" else 5)
    res = named["results"]
    assert res.columns[:3] == ["set_id", "name", "collection"]
    assert res.filter(pl.col("set_id") == "A_EARLY")["name"].to_list() == ["a early genes"]
    assert set(res["collection"]) == ({"coll.a"} if names == "coll.a" else {"coll.a", "coll.b"})
    listed = names if isinstance(names, list) else [names]
    assert named["attribution"]["release_id"] == "20990101.fixture"
    assert [c["collection"] for c in named["attribution"]["collections"]] == listed
    for c in named["attribution"]["collections"]:
        assert c["attribution"] and c["licence"] and c["upstream_release"] == "1.0"
    assert "attribution" not in plain and "id_resolution" not in plain["parameters"]


@pytest.mark.parametrize("kind, col", [("ora", "overlap_genes"), ("gsea", "leading_edge")])
def test_gene_columns_are_also_given_as_symbols(kind, col):
    res = both_ways(kind, "coll.a")[0]["results"]
    sym_col = enrichment._symbols_column(col)
    assert sym_col in res.columns
    to_symbol = {ens(i): sym(i) for i in range(1, 31)}
    for genes, symbols in res.select(col, sym_col).iter_rows():
        assert symbols == [to_symbol[g] for g in genes]
    assert res.filter(pl.col(col).list.len() > 0).height > 0


def test_mapping_results_keep_their_columns():
    _, plain = both_ways("ora", "coll.a")
    assert plain["results"].columns == STATS_COLUMNS["ora"]


def test_identifiers_resolve_and_unresolved_are_reported():
    genes = [
        ens(1) + ".12",      # versioned Ensembl
        ens(2),              # plain Ensembl
        "1003",              # Entrez
        "GENE4",             # symbol
        "gene5",             # symbol, wrong case
        "c1ORF99",           # mixed-case symbol, wrong case
        "ENSG99999999999",   # Ensembl-shaped, not in the gene table: kept
        "GENE4",             # a repeat counts once
        "NOTAGENE",          # unknown symbol
        "999999",            # unknown Entrez
    ]
    res = enrichment.ora(genes, "coll.a", universe=[ens(i) for i in range(1, 31)])
    rep = res["parameters"]["id_resolution"]["genes"]
    assert rep["given"] == 9
    assert rep["resolved"] == 7
    assert rep["unresolved"] == 2
    assert rep["unresolved_sample"] == ["NOTAGENE", "999999"]
    assert rep["resolved_by"] == {
        "ensembl": 3, "entrez": 1, "symbol": 1, "symbol_ignoring_case": 2,
    }
    # ENSG99999999999 and C1orf99 are in no coll.a set: the annotated universe drops them
    assert res["parameters"]["list_size"] == 5
    early = res["results"].filter(pl.col("set_id") == "A_EARLY")
    assert early["overlap_genes"].to_list() == [[ens(i) for i in range(1, 6)]]
    assert early["overlap_symbols"].to_list() == [[sym(i) for i in range(1, 6)]]
    assert res["parameters"]["id_resolution"]["universe"]["unresolved"] == 0


def test_the_unresolved_sample_is_capped():
    junk = [f"JUNK{i}" for i in range(25)]
    universe = [sym(i) for i in range(1, 31)] + junk
    res = enrichment.ora(["GENE1", "GENE2", *junk], "coll.a", universe=universe,
                         universe_mode="given")
    for what in ("genes", "universe"):
        rep = res["parameters"]["id_resolution"][what]
        assert rep["unresolved"] == 25
        assert rep["unresolved_sample"] == junk[: enrichment._UNRESOLVED_SAMPLE]
    assert res["parameters"]["universe_size"] == 30


def test_unresolved_stat_keys_are_dropped_and_reported():
    stats = {sym(i): float(30 - i) for i in range(1, 31)} | {"NOTAGENE": 99.0}
    res = enrichment.gsea_preranked(stats, "coll.a", nperm=100, seed=1)
    assert res["parameters"]["n_genes"] == 30
    rep = res["parameters"]["id_resolution"]["stats"]
    assert (rep["given"], rep["unresolved"], rep["unresolved_sample"]) == (31, 1, ["NOTAGENE"])


@pytest.mark.parametrize("kind", ["gsea", "competitive"])
def test_two_keys_naming_one_gene_are_refused(kind):
    stats = {sym(i): float(i) for i in range(1, 31)} | {"1001": 5.0}
    with pytest.raises(GeneticsUsageError, match="'GENE1' and '1001' are the same gene"):
        if kind == "gsea":
            enrichment.gsea_preranked(stats, "coll.a", nperm=10)
        else:
            enrichment.competitive(stats, "coll.a")


def test_stats_that_resolve_to_nothing_are_refused():
    with pytest.raises(GeneticsUsageError, match="resolves to a known gene"):
        enrichment.competitive({"NOPE": 1.0, "ALSO_NOPE": 2.0}, "coll.a")


def test_competitive_resolves_covariate_ids_too():
    values = np.linspace(2, -1, 30)
    log_len = np.log(np.arange(1, 31) * 1000.0)
    stats = {sym(i): v for i, v in zip(range(1, 31), values, strict=True)}
    covariates = pl.DataFrame({
        "gene": [str(1000 + i) if i != 29 else ens(29) + ".1" for i in range(1, 31)],
        "log_len": log_len,
    })
    named = enrichment.competitive(stats, "coll.a", covariates=covariates)
    plain = enrichment.competitive(
        {ens(i): v for i, v in zip(range(1, 31), values, strict=True)},
        enrichment.gene_sets("coll.a")["gene_sets"],
        covariates=pl.DataFrame({"gene": [ens(i) for i in range(1, 31)], "log_len": log_len}),
    )
    assert_frame_equal(named["results"].select(STATS_COLUMNS["competitive"]), plain["results"])
    assert named["parameters"]["id_resolution"]["covariates"]["resolved"] == 30


def test_a_case_fold_naming_two_genes_is_unresolved():
    res = enrichment.ora(
        ["Zed1", "ZED1", "zed1", "GENE1"], "coll.a", universe=[ens(i) for i in range(1, 33)]
    )
    rep = res["parameters"]["id_resolution"]["genes"]
    assert rep["unresolved_sample"] == ["zed1"]
    assert rep["resolved_by"]["symbol"] == 3
    assert rep["resolved_by"]["symbol_ignoring_case"] == 0


def test_a_collection_cache_entry_holds_only_its_own_memberships():
    enrichment.ora(["GENE1", "GENE2"], ["coll.a", "coll.b"])
    members = enrichment._BUNDLES[BUNDLE]._members
    for name, expected in {
        "coll.a": {"A_EARLY": 8, "A_MIDDLE": 10, "A_LATE": 7},
        "coll.b": {"B_FIRST": 6, "B_SECOND": 11},
    }.items():
        got = dict(members[name].group_by("set_id").len().iter_rows())
        assert got == expected


def test_attribution_cites_only_collections_with_a_tested_set():
    genes = [sym(i) for i in range(1, 11)]
    res = enrichment.ora(genes, ["coll.a", "coll.b"], min_size=8, max_size=10)
    assert sorted(res["results"]["collection"].unique().to_list()) == ["coll.a"]
    assert res["attribution"]["requested"] == ["coll.a", "coll.b"]
    assert [c["collection"] for c in res["attribution"]["collections"]] == ["coll.a"]
