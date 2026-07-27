"""Tests for the credible-set summariser used by the by-gene/variant/QTL-gene tools.

The load-bearing invariant is that a credible set is identified by
(resource, dataset, trait, cell_type, cs_id) and NOT by cs_id alone. cs_id is unique
only within one dataset's fine-mapping run of one trait in one cell type: caQTL cs_ids
are derived from the chromatin peak and recur in every cell type the peak was tested in,
and eQTL Catalogue cs_ids like ENSG00000187608_L1 recur across QTD studies. Grouping on
cs_id alone merged those, undercounting credible sets and dropping whole cell types.
"""

from genetics_mcp_server.tools import ToolExecutor

HEADER = (
    "resource\tversion\tdataset\tdata_type\ttrait\ttrait_original\tcell_type\t"
    "chr\tpos\tref\talt\tmlog10p\tbeta\tse\tpip\tcs_id\tcs_size\tcs_min_r2\taaf\t"
    "most_severe\tgene_most_severe"
)


def _row(cs_id, cell_type, pos, pip, mlog10p, dataset="FinnGen_ATACseq",
         data_type="caQTL", trait="IL7R", resource="finngen"):
    return (
        f"{resource}\t1\t{dataset}\t{data_type}\t{trait}\tchr5-35863122-35863905\t"
        f"{cell_type}\t5\t{pos}\tA\tG\t{mlog10p}\t0.5\t0.05\t{pip}\t{cs_id}\t2\t0.9\t0.3\t"
        "intron_variant\tIL7R"
    )


def _summarize(rows):
    return ToolExecutor(api_base_url="http://unused")._summarize_credible_sets_simple(
        "\n".join([HEADER, *rows]) + "\n"
    )


class TestCredibleSetIdentity:
    def test_same_cs_id_in_different_cell_types_counts_separately(self):
        """The caQTL case: one peak-derived cs_id tested in three cell types."""
        summary = _summarize([
            _row("chr5-35863122-35863905_1", "l1.CD4_T", 35863200, 0.9, 40.0),
            _row("chr5-35863122-35863905_1", "l1.CD4_T", 35863300, 0.1, 12.0),
            _row("chr5-35863122-35863905_1", "l1.CD8_T", 35863200, 0.8, 30.0),
            _row("chr5-35863122-35863905_1", "l1.NK", 35863200, 0.7, 20.0),
        ])
        assert summary["n_cs"] == 3
        cell_types = {cs["cell_type"] for cs in summary["cs"]["caQTL"]}
        assert cell_types == {"l1.CD4_T", "l1.CD8_T", "l1.NK"}

    def test_same_cs_id_in_different_datasets_counts_separately(self):
        """The eQTL Catalogue case: ENSG-derived cs_ids recur across QTD studies."""
        summary = _summarize([
            _row("ENSG00000168685_L1", "monocyte", 35863200, 0.9, 40.0,
                 dataset="QTD000034", data_type="eQTL", resource="eqtl_catalogue"),
            _row("ENSG00000168685_L1", "T_cell", 35863200, 0.8, 30.0,
                 dataset="QTD000456", data_type="eQTL", resource="eqtl_catalogue"),
        ])
        assert summary["n_cs"] == 2

    def test_distinct_cs_ids_in_one_cell_type_still_separate(self):
        summary = _summarize([
            _row("chr5-35863122-35863905_1", "l1.CD4_T", 35863200, 0.9, 40.0),
            _row("chr5-35863122-35863905_2", "l1.CD4_T", 35863400, 0.9, 25.0),
        ])
        assert summary["n_cs"] == 2

    def test_lead_variant_is_the_top_pip_row_of_its_own_credible_set(self):
        """Each cell type keeps its own lead, rather than inheriting a merged one."""
        summary = _summarize([
            _row("chr5-35863122-35863905_1", "l1.CD4_T", 111, 0.9, 40.0),
            _row("chr5-35863122-35863905_1", "l1.CD4_T", 222, 0.1, 12.0),
            _row("chr5-35863122-35863905_1", "l1.NK", 333, 0.7, 20.0),
        ])
        leads = {cs["cell_type"]: cs["lead_variant"]["id"] for cs in summary["cs"]["caQTL"]}
        assert leads == {"l1.CD4_T": "5:111:A:G", "l1.NK": "5:333:A:G"}

    def test_null_cell_type_does_not_drop_rows(self):
        """GWAS rows have cell_type NA; a nullable group key must not lose them."""
        rows = [
            _row("chr1:1-2_1", "NA", 1000, 0.9, 40.0, dataset="FinnGen_R13",
                 data_type="GWAS", trait="K11_IBD_STRICT"),
            _row("chr1:1-2_1", "NA", 1001, 0.2, 15.0, dataset="FinnGen_R13",
                 data_type="GWAS", trait="K11_IBD_STRICT"),
        ]
        summary = _summarize(rows)
        assert summary["n_cs"] == 1
        assert summary["cs"]["GWAS"][0]["lead_variant"]["id"] == "5:1000:A:G"

    def test_empty_input(self):
        assert _summarize([]) == {"n_cs": 0, "cs": {}}


class TestSummaryCounts:
    """The counts block answers "how many X" without walking the credible-set list.

    It has to sit before `cs` in the dict, since that list is what truncation eats.
    """

    def test_counts_precede_the_credible_set_list(self):
        summary = _summarize([_row("chr5-1-2_1", "l1.CD4_T", 100, 0.9, 40.0)])
        keys = list(summary)
        assert keys.index("counts") < keys.index("cs")

    def test_caqtl_counts_peaks_from_trait_original(self):
        """trait is the linked gene on the QTL-gene endpoint; the peak is trait_original."""
        summary = _summarize([
            _row("chr5-1-2_1", "l1.CD4_T", 100, 0.9, 40.0),
            _row("chr5-1-2_1", "l1.NK", 100, 0.8, 30.0),
            _row("chr5-1-2_1", "l1.NK", 200, 0.2, 10.0),
        ])
        counts = summary["counts"]["caQTL"]
        assert counts["n_credible_sets"] == 2
        assert counts["n_associations"] == 3
        assert counts["n_variants"] == 2
        assert counts["n_cell_types"] == 2
        assert counts["n_peaks"] == 1
        assert counts["n_traits"] == 1

    def test_counts_are_per_data_type(self):
        summary = _summarize([
            _row("chr5-1-2_1", "l1.CD4_T", 100, 0.9, 40.0),
            _row("ENSG1_L1", "monocyte", 200, 0.9, 20.0,
                 dataset="QTD000034", data_type="eQTL", trait="IL7R",
                 resource="eqtl_catalogue"),
        ])
        assert summary["counts"]["caQTL"]["n_credible_sets"] == 1
        assert summary["counts"]["eQTL"]["n_credible_sets"] == 1
        assert "n_peaks" not in summary["counts"]["eQTL"]

    def test_null_cell_type_is_not_counted_as_a_cell_type(self):
        summary = _summarize([
            _row("chr1:1-2_1", "NA", 1000, 0.9, 40.0, dataset="FinnGen_R13",
                 data_type="GWAS", trait="K11_IBD_STRICT"),
        ])
        assert "n_cell_types" not in summary["counts"]["GWAS"]
