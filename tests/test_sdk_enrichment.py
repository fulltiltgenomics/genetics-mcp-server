"""genetics.enrichment reproduces R: ORA under both universe conventions, preranked GSEA
(fgseaSimple) and the competitive gene-level test (lm).

The fixtures under tests/fixtures/enrichment/ were written by reference.R (fgsea 1.32.4, R
4.4.3; see the README there). `ora_phyper.tsv` holds `phyper` p-values and BH q-values under
the "given" and "annotated" universe conventions over gene_sets.gmt plus
gene_sets_boundary.gmt, whose sets cross the size filter only through universe restriction;
`ora_fora.tsv` is `fgsea::fora` under the "given" convention, and supplies overlap genes.
`gsea_fgseaSimple_nperm10000_seed1.tsv` is fgseaSimple on ranked_stats.tsv,
`gsea_es_gseaParam0_2.tsv` its ES under two other weights, and
`competitive_lm.tsv` is `lm(z ~ in_set + log_len)` on gene_stats.tsv.
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


# --- preranked GSEA ------------------------------------------------------------------------

NPERM = 10_000
# per-comparison two-sided tail of a 5 sigma bound is 5.7e-7; over 28 sets and two Monte
# Carlo quantities a correct implementation fails with probability ~3e-5
Z = 5.0


@pytest.fixture(scope="module")
def ranked():
    return pl.read_csv(FIXTURES / "ranked_stats.tsv", separator="\t")


@pytest.fixture(scope="module")
def gsea(ranked):
    res = enrichment.gsea_preranked(ranked, read_gmt("gene_sets.gmt"), nperm=NPERM, seed=1)
    ref = pl.read_csv(FIXTURES / "gsea_fgseaSimple_nperm10000_seed1.tsv", separator="\t")
    return res, res["results"].sort("set_id"), ref.sort("pathway")


def test_gsea_es_and_leading_edge_match_fgsea_exactly(gsea):
    """ES and the leading edge do not depend on the permutations."""
    res, got, ref = gsea
    assert got["set_id"].to_list() == ref["pathway"].to_list()
    assert got["size"].to_list() == ref["size"].to_list()
    assert np.allclose(got["es"], ref["ES"], rtol=1e-12, atol=0)
    assert got["leading_edge"].to_list() == [g.split(",") for g in ref["leadingEdge"]]
    params = res["parameters"]
    assert (params["min_size"], params["max_size"], params["nperm"], params["seed"]) == (
        5, 300, NPERM, 1,
    )
    assert params["n_genes"] == 900 and params["sets_tested"] == 28 == ref.height


def test_gsea_pval_within_monte_carlo_error_of_fgsea(gsea):
    """Each p is (X + 1) / (n0 + 1), X ~ Binomial(n0, p) over the n0 same-signed permutation
    ESs. Two independent runs therefore differ by sd sqrt(2 p (1 - p) / n0); n0 is read back
    out of R's own columns. The exact formula is held by the next test, not by this one."""
    _, got, ref = gsea
    p_r = ref["pval"].to_numpy()
    n0 = (ref["nMoreExtreme"].to_numpy() + 1) / p_r - 1
    bound = Z * np.sqrt(2 * p_r * (1 - p_r) / n0)
    diff = np.abs(got["pval"].to_numpy() - p_r)
    assert (diff <= bound).all(), dict(zip(ref["pathway"], diff / bound, strict=True))
    # the planted SET16 (up) and SET17 (down) beat every permutation, as in R
    strong = got.filter(pl.col("set_id").is_in(["SET16", "SET17"]))
    assert strong["n_more_extreme"].to_list() == [0, 0]
    assert np.allclose(got["padj"], enrichment._bh(got["pval"].to_numpy()))


def ranked_weights(ranked, weight=1.0):
    stat = ranked["stat"].to_numpy()
    return np.abs(stat[np.argsort(-stat, kind="stable")]) ** weight


def test_gsea_pval_nes_and_n_more_extreme_follow_fgsea_formulas_exactly(ranked):
    """The same null, rebuilt from the same seed, put through fgseaSimple's formulas one set
    at a time: p = min((1 + #le) / (1 + #le0), (1 + #ge) / (1 + #ge0)) and NES = ES / the
    mean of the same-signed null ESs."""
    nperm, seed = 500, 7
    got = enrichment.gsea_preranked(ranked, read_gmt("gene_sets.gmt"), nperm=nperm, seed=seed)
    got = got["results"]
    sizes = np.unique(got["size"].to_numpy())
    null = enrichment._null_es(ranked_weights(ranked), sizes, nperm, np.random.default_rng(seed))
    for row in got.iter_rows(named=True):
        col, es = null[:, np.searchsorted(sizes, row["size"])], row["es"]
        le, ge = (col <= es).sum(), (col >= es).sum()
        le0, ge0 = (col <= 0).sum(), (col >= 0).sum()
        assert row["pval"] == min((1 + le) / (1 + le0), (1 + ge) / (1 + ge0)), row["set_id"]
        assert row["n_more_extreme"] == (ge if es > 0 else le), row["set_id"]
        same = col[col >= 0] if es > 0 else col[col <= 0]
        assert row["nes"] == pytest.approx(es / abs(same.mean()), rel=1e-12), row["set_id"]


@pytest.mark.parametrize("weight", [0, 2])
def test_gsea_es_under_other_weights_matches_fgsea(ranked, weight):
    ref = pl.read_csv(FIXTURES / "gsea_es_gseaParam0_2.tsv", separator="\t")
    ref = ref.filter(pl.col("gseaParam") == weight).sort("pathway")
    got = enrichment.gsea_preranked(
        ranked, read_gmt("gene_sets.gmt"), nperm=10, seed=0, weight=weight
    )["results"].sort("set_id")
    assert got["set_id"].to_list() == ref["pathway"].to_list()
    assert np.allclose(got["es"], ref["ES"], rtol=1e-12, atol=0)


def test_gsea_nes_within_monte_carlo_error_of_fgsea(gsea, ranked):
    """NES = ES / mean of the n0 same-signed null ESs, so its relative error is that of the
    mean, CV / sqrt(n0) per run and sqrt(2) times that between two runs. The CV of each
    size's same-signed null is measured from an independent null draw."""
    _, got, ref = gsea
    w = ranked_weights(ranked)
    sizes = np.unique(got["size"].to_numpy())
    null = enrichment._null_es(w, sizes, NPERM, np.random.default_rng(2))
    es = got["es"].to_numpy()
    p_r = ref["pval"].to_numpy()
    n0 = (ref["nMoreExtreme"].to_numpy() + 1) / p_r - 1
    cv = np.empty(es.size)
    for i, (e, size) in enumerate(zip(es, got["size"].to_numpy(), strict=True)):
        col = null[:, np.searchsorted(sizes, size)]
        same = col[col >= 0] if e > 0 else col[col <= 0]
        cv[i] = same.std() / abs(same.mean())
    rel = np.abs(got["nes"].to_numpy() / ref["NES"].to_numpy() - 1)
    bound = Z * np.sqrt(2) * cv / np.sqrt(n0)
    assert (rel <= bound).all(), dict(zip(ref["pathway"], rel / bound, strict=True))


def test_gsea_null_draws_genes_without_replacement():
    """With 6 genes and a set size of 5 there are only six possible random sets; a null that
    drew with replacement would produce ESs outside their six values."""
    w = np.array([3.0, 2.5, 1.0, 0.7, 0.4, 0.1])
    subsets = np.array([[j for j in range(6) if j != i] for i in range(6)])
    possible = enrichment._running_es(w, subsets, 6)[0]
    null = enrichment._null_es(w, np.array([5]), 2000, np.random.default_rng(0))[:, 0]
    assert np.isin(null, possible).all()
    assert np.unique(null).size == 6


def test_gsea_seed_reproduces_and_a_drawn_seed_is_reported(ranked):
    sets = read_gmt("gene_sets.gmt")
    a = enrichment.gsea_preranked(ranked, sets, nperm=200)
    b = enrichment.gsea_preranked(ranked, sets, nperm=200, seed=a["parameters"]["seed"])
    assert a["results"].equals(b["results"])


def test_gsea_ties_rank_in_input_order():
    """g2, g3 and g4 tie; the set's g2 ranks third given first among them, fifth given last."""
    values = dict(zip([f"g{i}" for i in range(12)], [5.0, 4.0, 3.0, 3.0, 3.0], strict=False))
    values |= {f"g{i}": -0.1 * (i - 4) for i in range(5, 12)}
    reordered = {g: values[g] for g in ["g0", "g1", "g4", "g3", "g2", *list(values)[5:]]}
    sets = {"S": ["g0", "g2", "g9", "g10", "g11"]}
    w = np.abs(np.array(sorted(values.values(), reverse=True)))
    for stats, g2_rank in ((values, 2), (reordered, 4)):
        res = enrichment.gsea_preranked(stats, sets, nperm=50, seed=0)
        expected = enrichment._running_es(w, np.array([[0, g2_rank, 9, 10, 11]]), 12)[0]
        assert res["results"]["es"].item() == pytest.approx(expected[0], rel=1e-14)
        assert res["parameters"]["tied_stats"] == 2


@pytest.mark.parametrize(
    "stats, kwargs, message",
    [
        ({"a": 1.0, "b": None}, {}, "null"),
        ({"a": 1.0, "b": float("nan")}, {}, "NaN"),
        (pl.DataFrame({"gene": ["a", "a"], "stat": [1.0, 2.0]}), {}, "more than one"),
        (pl.DataFrame({"gene": ["a", None], "stat": [1.0, 2.0]}), {}, "null"),
        (["a", "b"], {}, "mapping"),
        ({"a": 1.0}, {"nperm": 0}, "nperm"),
        ({"a": 1.0}, {"nperm": 100.5}, "nperm"),
        ({"a": 1.0}, {"nperm": "100"}, "nperm"),
        ({"a": 1.0}, {"weight": -1}, "weight"),
        ({"a": 1.0}, {"weight": "1"}, "weight"),
        ({"a": 1.0}, {"weight": None}, "weight"),
        ({"a": 1.0}, {"seed": -1}, "seed"),
        ({"a": 1.0}, {"seed": 1.5}, "seed"),
        ({"a": 1.0}, {"seed": "1"}, "seed"),
    ],
)
def test_gsea_usage_errors(stats, kwargs, message):
    with pytest.raises(GeneticsUsageError, match=message):
        enrichment.gsea_preranked(stats, {"S": ["a", "b"]}, **kwargs)


def test_gsea_refuses_a_null_over_budget_before_doing_the_work():
    """GO BP scale fits at the default nperm; sizes up to 1,000 at the default do not."""
    assert enrichment._null_cost(np.arange(5, 301), 18_215, 10_000) < enrichment._NULL_BUDGET
    stats = {f"g{i}": float(i) for i in range(2000)}
    sets = {f"S{m}": [f"g{i}" for i in range(m)] for m in range(5, 1001)}
    with pytest.raises(GeneticsUsageError, match="lower nperm or max_size"):
        enrichment.gsea_preranked(stats, sets, max_size=1000)
    assert enrichment.gsea_preranked(stats, sets, max_size=1000, nperm=50)["results"].height


def test_gsea_with_no_set_in_range_returns_an_empty_frame(ranked):
    full = enrichment.gsea_preranked(ranked, read_gmt("gene_sets.gmt"), nperm=10, seed=0)
    res = enrichment.gsea_preranked({"a": 1.0, "b": 2.0}, {"S": ["a", "b"]}, seed=0)
    assert res["results"].height == 0
    assert res["results"].schema == full["results"].schema
    assert res["parameters"]["sets_tested"] == 0


# --- competitive gene-level test ----------------------------------------------------------


@pytest.fixture(scope="module")
def gene_stats():
    return pl.read_csv(FIXTURES / "gene_stats.tsv", separator="\t")


def test_competitive_matches_r_lm(gene_stats):
    res = enrichment.competitive(
        gene_stats.select("gene", "z"),
        read_gmt("gene_sets.gmt"),
        covariates=gene_stats.select("gene", "log_len"),
    )
    ref = pl.read_csv(FIXTURES / "competitive_lm.tsv", separator="\t").sort("pathway")
    got = res["results"].sort("set_id")
    assert got["set_id"].to_list() == ref["pathway"].to_list()
    assert got["size"].to_list() == ref["size"].to_list()
    for col in ("beta", "se", "t", "pval"):
        assert_rel(got[col], ref[col], col)
    assert np.allclose(got["padj"], enrichment._bh(got["pval"].to_numpy()))
    params = res["parameters"]
    assert params["df"] == ref["df"][0] == 897
    assert params["covariates"] == ["log_len"]
    assert (params["n_genes"], params["sets_tested"], params["sets_aliased"]) == (900, 28, 0)
    assert res["results"]["set_id"][0] == "SET15"  # the planted signal


def test_competitive_covariate_scale_does_not_matter(gene_stats):
    """Collinearity is judged on directions: a covariate in tiny units fits the same model."""
    sets = read_gmt("gene_sets.gmt")
    z = gene_stats.select("gene", "z")
    base = enrichment.competitive(z, sets, covariates=gene_stats.select("gene", "log_len"))
    tiny = enrichment.competitive(
        z, sets, covariates=gene_stats.select("gene", pl.col("log_len") * 1e-12)
    )
    for col in ("beta", "se", "pval"):
        assert np.allclose(tiny["results"][col], base["results"][col], rtol=1e-9, atol=0)


def test_competitive_refuses_a_statistic_with_no_variation(gene_stats):
    for value in (0.0, 3.7):
        stats = dict.fromkeys(gene_stats["gene"], value)
        with pytest.raises(GeneticsUsageError, match="no variation"):
            enrichment.competitive(stats, read_gmt("gene_sets.gmt"))


def test_competitive_without_covariates_matches_lstsq(gene_stats):
    stats = dict(zip(gene_stats["gene"], gene_stats["z"], strict=True))
    sets = read_gmt("gene_sets.gmt")
    got = enrichment.competitive(stats, sets)["results"].filter(pl.col("set_id") == "SET20")
    x = gene_stats["gene"].is_in(sets["SET20"]).cast(float).to_numpy()
    design = np.column_stack([np.ones(x.size), x])
    coef, rss, *_ = np.linalg.lstsq(design, gene_stats["z"].to_numpy(), rcond=None)
    se = np.sqrt(rss[0] / (x.size - 2) * np.linalg.inv(design.T @ design)[1, 1])
    assert got["beta"].item() == pytest.approx(coef[1], rel=1e-10)
    assert got["se"].item() == pytest.approx(se, rel=1e-10)


@pytest.mark.parametrize(
    "covariates, message",
    [
        (pl.DataFrame({"gene": ["G0001"], "log_len": [1.0]}), "no covariates"),
        (lambda gs: gs.select("gene", pl.lit(1.0).alias("const")), "collinear"),
        (
            lambda gs: gs.select("gene", "log_len", lin=2 * pl.col("log_len") + 3),
            "collinear",
        ),
        (lambda gs: gs.select("gene", pl.lit(0.0).alias("zero")), "collinear"),
        (
            lambda gs: gs.select(
                "gene", pl.when(pl.col("gene") == "G0001").then(None).otherwise("log_len")
            ),
            "null",
        ),
        (lambda gs: gs.select("gene"), "DataFrame"),
    ],
)
def test_competitive_usage_errors(gene_stats, covariates, message):
    if callable(covariates):
        covariates = covariates(gene_stats)
    with pytest.raises(GeneticsUsageError, match=message):
        enrichment.competitive(
            gene_stats.select("gene", "z"), read_gmt("gene_sets.gmt"), covariates=covariates
        )
