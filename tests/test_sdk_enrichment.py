"""genetics.enrichment.ora reproduces R's hypergeometric ORA under both universe conventions.

The fixtures under tests/fixtures/enrichment/ were written by reference.R (fgsea 1.32.4, R
4.4.3; see the README there). `ora_phyper.tsv` holds `phyper` p-values and BH q-values under
the "given" and "annotated" universe conventions over gene_sets.gmt plus
gene_sets_boundary.gmt, whose sets cross the size filter only through universe restriction;
`ora_fora.tsv` is `fgsea::fora` under the "given" convention, and supplies overlap genes.
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from genetics_mcp_server.sdk import enrichment
from genetics_mcp_server.sdk.errors import GeneticsUsageError

FIXTURES = Path(__file__).parent / "fixtures" / "enrichment"
RTOL = 1e-6


def read_gmt(*names: str) -> dict[str, list[str]]:
    out = {}
    for name in names:
        for line in (FIXTURES / name).read_text().splitlines():
            set_id, _desc, *genes = line.split("\t")
            out[set_id] = genes
    return out


def read_lines(name: str) -> list[str]:
    return (FIXTURES / name).read_text().split()


@pytest.fixture(scope="module")
def inputs():
    return dict(
        genes=read_lines("gene_list.txt"),
        gene_sets=read_gmt("gene_sets.gmt", "gene_sets_boundary.gmt"),
        universe=read_lines("universe.txt"),
    )


@pytest.fixture(scope="module")
def phyper():
    return pl.read_csv(FIXTURES / "ora_phyper.tsv", separator="\t")


def assert_rel(actual, expected, what):
    actual, expected = np.asarray(actual, float), np.asarray(expected, float)
    rel = np.abs(actual - expected) / np.abs(expected)
    assert rel.max() <= RTOL, f"{what}: max relative error {rel.max():.3g}"


@pytest.mark.parametrize("mode, n_universe, n_list", [("given", 900, 58), ("annotated", 759, 52)])
def test_matches_r_phyper(inputs, phyper, mode, n_universe, n_list):
    res = enrichment.ora(**inputs, universe_mode=mode)
    ref = phyper.select(
        "pathway", *(pl.col(f"{c}_{mode}").alias(c) for c in ("size", "overlap", "pval", "padj"))
    ).sort("pathway")
    got = res["results"].sort("set_id")
    assert got["set_id"].to_list() == ref["pathway"].to_list()
    assert got["size"].to_list() == ref["size"].to_list()
    assert got["overlap"].to_list() == ref["overlap"].to_list()
    assert_rel(got["pval"], ref["pval"], f"{mode} pval")
    assert_rel(got["padj"], ref["padj"], f"{mode} padj")

    params = res["parameters"]
    assert params["universe_mode"] == mode
    assert params["universe_size"] == n_universe
    assert params["list_size"] == n_list
    assert params["list_dropped"] == len(inputs["genes"]) - n_list
    assert params["sets_given"] == len(inputs["gene_sets"])
    assert params["sets_tested"] == ref.height


def test_size_filter_is_applied_after_universe_restriction(inputs):
    """SET31..SET34 are 7, 8, 305 and 306 genes as given and 4, 5, 300, 301 inside U."""
    got = enrichment.ora(**inputs)["results"]
    tested = set(got["set_id"].to_list())
    assert {"SET32", "SET33"} <= tested
    assert not {"SET31", "SET34", "SET01", "SET02"} & tested
    sizes = dict(zip(got["set_id"].to_list(), got["size"].to_list(), strict=True))
    assert (sizes["SET32"], sizes["SET33"]) == (5, 300)


def test_overlap_genes_and_fold_enrichment_match_fora(inputs):
    fora = pl.read_csv(FIXTURES / "ora_fora.tsv", separator="\t").sort("pathway")
    got = enrichment.ora(**inputs, universe_mode="given")["results"].sort("set_id")
    assert got["set_id"].to_list() == fora["pathway"].to_list()
    ref_genes = [sorted(g.split(",")) if g else [] for g in fora["overlapGenes"].fill_null("")]
    assert got["overlap_genes"].to_list() == ref_genes
    assert_rel(got["pval"], fora["pval"], "fora pval")
    nonzero = fora["foldEnrichment"] > 0
    assert_rel(got["fold_enrichment"].filter(nonzero), fora["foldEnrichment"].filter(nonzero), "fold")


def test_results_are_ordered_by_p(inputs):
    p = enrichment.ora(**inputs)["results"]["pval"].to_numpy()
    assert (np.diff(p) >= 0).all()


def test_size_boundaries_and_bh_over_tested_sets_only():
    universe = [f"G{i}" for i in range(20)]
    sets = {
        "below": ["G0", "G1", "OUT1", "OUT2"],  # 4 given, 2 in U: dropped at min_size=3
        "at_min": ["G0", "G1", "G2", "OUT1"],  # 3 in U
        "at_max": ["G0", "G1", "G2", "G3", "G4", "OUT1"],  # 5 in U
        "above": ["G0", "G1", "G2", "G3", "G4", "G5"],  # 6 in U: dropped at max_size=5
    }
    res = enrichment.ora(
        ["G0", "G1", "G9"], sets, universe, universe_mode="given", min_size=3, max_size=5
    )
    got = res["results"].sort("set_id")
    assert got["set_id"].to_list() == ["at_max", "at_min"]
    assert got["size"].to_list() == [5, 3]
    from scipy.stats import hypergeom

    p_max, p_min = hypergeom.sf(1, 20, np.array([5, 3]), 3)
    assert p_min < p_max
    assert np.allclose(got["pval"], [p_max, p_min], rtol=1e-12)
    # BH over the two tested sets, not four
    assert np.allclose(got["padj"], [p_max, min(2 * p_min, p_max)], rtol=1e-12)
    assert res["parameters"]["sets_tested"] == 2


def test_without_universe_u_is_the_annotated_genes(inputs):
    res = enrichment.ora(inputs["genes"], inputs["gene_sets"], universe_mode="given")
    annotated = {g for genes in inputs["gene_sets"].values() for g in genes}
    assert res["parameters"]["universe_size"] == len(annotated)
    assert res["parameters"]["universe_source"].startswith("annotated")


def test_duplicates_count_once():
    sets = {"A": ["a", "a", "b", "c", "d", "e"], "B": ["f", "g", "h", "i", "j"]}
    once = enrichment.ora(["a", "b"], sets)["results"]
    twice = enrichment.ora(["a", "a", "b"], sets)["results"]
    assert once.equals(twice)
    a = once.filter(pl.col("set_id") == "A")
    assert a["size"].item() == 5
    assert a["overlap"].item() == 2
    # 5 of 10 annotated genes in the set, 2 drawn: P(X >= 2) = 2/9
    assert a["pval"].item() == pytest.approx(2 / 9, rel=1e-12)


@pytest.mark.parametrize("wrap", [list, pl.Series], ids=["list", "series"])
@pytest.mark.parametrize("where", ["genes", "universe", "gene_sets"])
def test_null_gene_ids_are_refused(inputs, wrap, where):
    bad = {
        "genes": wrap(inputs["genes"] + [None]),
        "universe": wrap(inputs["universe"] + [None]),
        "gene_sets": {**inputs["gene_sets"], "NULLSET": wrap(["G0001", None])},
    }[where]
    with pytest.raises(GeneticsUsageError, match="null"):
        enrichment.ora(**{**inputs, where: bad}, universe_mode="given")


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(universe_mode="genome"), "universe_mode"),
        (dict(min_size=10, max_size=5), "min_size"),
        (dict(genes="G0001"), "list of gene ids"),
        (dict(genes=["NOPE"]), "none of the"),
        (dict(gene_sets={"S": [1, 2, 3]}), "strings"),
        (dict(gene_sets=[["G0001"]]), "mapping"),
    ],
)
def test_usage_errors(inputs, kwargs, message):
    with pytest.raises(GeneticsUsageError, match=message):
        enrichment.ora(**{**inputs, **kwargs})
