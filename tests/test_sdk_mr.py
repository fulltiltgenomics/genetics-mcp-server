"""genetics.mr harmonise() and checks(): the allele join, the AF cascade and the checks frame.

Synthetic frames only; the annotation lookup is a recording fake, so nothing reaches a network.
"""

import polars as pl
import pytest

from genetics_mcp_server.sdk import mr
from genetics_mcp_server.sdk.errors import GeneticsUsageError

UKB_PPP = {"dataset_id": "ukbb_pqtl", "dataset": "UKB_PPP", "data_type": "pqtl", "trait_type": "quantitative",
           "cohorts": ["ukb"], "n_samples": 42853}
FG_OLINK = {"dataset_id": "finngen_olink", "data_type": "pqtl", "trait_type": "quantitative",
            "cohorts": ["finngen"]}
IIBDGC = {"dataset_id": "iibdgc", "data_type": "gwas", "trait_type": "binary",
          "cohorts": ["iibdgc"]}
FG_R14 = {"dataset_id": "finngen_r14", "data_type": "gwas", "trait_type": "mixed",
          "cohorts": ["finngen"]}
UKB_GWAS = {"dataset_id": "ukbb_finucane", "data_type": "gwas", "trait_type": "quantitative",
            "cohorts": ["ukb"]}


class FakeAnnotation:
    def __init__(self, finngen=None, gnomad=None):
        self.tables = {"finngen": finngen or {}, "gnomad": gnomad or {}}
        self.calls = []

    def __call__(self, *, variants, source):
        self.calls.append((source, list(variants)))
        rows = [
            {"variant": v, **{k: str(val) for k, val in cols.items()}}
            for v, cols in self.tables[source].items() if v in variants
        ]
        return pl.DataFrame(rows) if rows else pl.DataFrame()


def exposure(rows, af="aaf"):
    return pl.DataFrame([
        {"variant": v, "beta": b, "se": s, af: f, "pip": 0.9, "cs_id": f"cs{i}",
         "most_severe": cons}
        for i, (v, b, s, f, cons) in enumerate(rows)
    ], schema_overrides={af: pl.Float64})


def outcome(rows):
    """Results-api shape: chr/pos/ref/alt and every value a string, NA for missing."""
    out = []
    for v, b, s, f in rows:
        c, p, r, a = v.split(":")
        out.append({"chr": c, "pos": p, "ref": r, "alt": a, "beta": str(b), "se": str(s),
                    "af": "NA" if f is None else str(f), "n_cases": "1000",
                    "n_controls": "9000", "phenotype": "IBD"})
    return pl.DataFrame(out)


def row(h, variant):
    return h.frame.filter(pl.col("variant") == variant).to_dicts()[0]


def no_lookup(**_):
    raise AssertionError("annotation consulted although every instrument had an outcome af")


# --------------------------------------------------------------------------- join


def test_exact_join_keeps_both_sides_and_units():
    h = mr.harmonise(
        exposure([("1:100:A:G", 0.5, 0.05, 0.7, "intron_variant")]),
        outcome([("1:100:A:G", 0.1, 0.02, 0.68)]),
        exposure_dataset=FG_OLINK, outcome_dataset=IIBDGC, annotation=no_lookup,
    )
    r = row(h, "1:100:A:G")
    assert (r["beta_x"], r["beta_y"], r["se_y"], r["af_y"]) == (0.5, 0.1, 0.02, 0.68)
    assert (r["n_cases"], r["n_controls"], r["n_y"]) == (1000, 9000, 10000)
    assert r["in_outcome"] and r["outcome_match"] == "exact"
    assert (r["chr"], r["pos"]) == (1, 100)
    assert h.units == {"exposure": "SD (normalised molecular level)", "outcome": "log OR (binary)"}


def test_units_unknown_without_datasets_and_mixed_resolved_by_case_counts():
    h = mr.harmonise(exposure([("1:100:A:G", 0.5, 0.05, 0.7, None)]),
                     outcome([("1:100:A:G", 0.1, 0.02, 0.68)]), annotation=no_lookup)
    assert h.units == {"exposure": "unknown", "outcome": "unknown"}
    assert any("non-Finnish" in n for n in h.notes)
    h = mr.harmonise(exposure([("1:100:A:G", 0.5, 0.05, 0.7, None)]),
                     outcome([("1:100:A:G", 0.1, 0.02, 0.68)]),
                     exposure_dataset=UKB_GWAS, outcome_dataset=FG_R14, annotation=no_lookup)
    assert h.units == {"exposure": "SD (quantitative)", "outcome": "log OR (binary)"}


def test_units_follow_the_rows_then_the_phenotypes_row_then_trait_type():
    x = exposure([("1:100:A:G", 0.5, 0.05, 0.7, None)])
    y = outcome([("1:100:A:G", 0.1, 0.02, 0.68)])
    # FinnGen's trait_type is binary; a quantitative endpoint's phenotypes() row has no cases
    quant = {"dataset": "FG", "trait_original": "BMI_IRN", "n_cases": None, "n_controls": None}
    h = mr.harmonise(x, y, exposure_dataset=FG_R14, outcome_dataset=FG_R14,
                     exposure_meta=quant, annotation=no_lookup)
    assert h.units == {"exposure": "SD (quantitative)", "outcome": "log OR (binary)"}
    h = mr.harmonise(x.with_columns(pl.lit("500").alias("n_cases")), y,
                     exposure_dataset=FG_R14, exposure_meta=quant, annotation=no_lookup)
    assert h.units["exposure"] == "log OR (binary)"
    h = mr.harmonise(x, y.drop("n_cases", "n_controls"), exposure_dataset=FG_R14,
                     outcome_dataset=FG_R14, outcome_meta=quant, annotation=no_lookup)
    # mixed, with neither rows nor a phenotypes() row to decide it
    assert h.units == {"exposure": "unknown", "outcome": "SD (quantitative)"}


def test_swapped_outcome_rows_are_never_used_and_stay_counted():
    # an SNV swap and an insertion-vs-deletion swap: neither is flipped
    h = mr.harmonise(
        exposure([("2:200:C:T", 0.4, 0.05, 0.3, None), ("1:100:A:AT", 0.4, 0.05, 0.3, None),
                  ("1:300:A:G", 0.4, 0.05, 0.3, None)]),
        outcome([("2:200:T:C", 0.2, 0.03, 0.7), ("1:100:AT:A", 0.2, 0.03, 0.7),
                 ("1:300:A:G", 0.2, 0.03, 0.3)]),
        exposure_dataset=FG_OLINK, annotation=FakeAnnotation(),
    )
    for v in ("2:200:C:T", "1:100:A:AT"):
        r = row(h, v)
        assert r["outcome_match"] == "swapped_not_used" and r["in_outcome"] is False
        assert r["beta_y"] is None and r["se_y"] is None and r["af_y"] is None
    assert (h.frame["outcome_match"] == "swapped_not_used").sum() == 2
    assert h.frame["in_outcome"].sum() == 1


def test_exact_match_wins_over_swapped():
    h = mr.harmonise(
        exposure([("2:200:C:T", 0.4, 0.05, 0.3, None)]),
        outcome([("2:200:C:T", 0.1, 0.03, 0.31), ("2:200:T:C", 0.9, 0.03, 0.69)]),
        annotation=no_lookup,
    )
    assert row(h, "2:200:C:T")["beta_y"] == 0.1


def test_palindromic_swap_follows_the_same_rule():
    h = mr.harmonise(
        exposure([("3:300:A:T", 0.4, 0.05, 0.2, None)]),
        outcome([("3:300:T:A", 0.2, 0.03, 0.8)]),
        annotation=FakeAnnotation(),
    )
    r = row(h, "3:300:A:T")
    assert r["outcome_match"] == "swapped_not_used"
    assert not r["in_outcome"] and r["beta_y"] is None


def test_unusable_exposure_rows_are_dropped_with_a_reason():
    h = mr.harmonise(
        exposure([("1:100:A:G", 0.5, 0.05, 0.7, None), ("1:101:A:G", None, 0.05, 0.7, None),
                  ("1:102:A:G", 0.5, 0.0, 0.7, None)]),
        outcome([("1:100:A:G", 0.1, 0.02, 0.68)]),
        annotation=no_lookup,
    )
    assert h.frame["variant"].to_list() == ["1:100:A:G"]
    assert sorted(h.dropped["variant"].to_list()) == ["1:101:A:G", "1:102:A:G"]


def test_outcome_rows_without_beta_count_as_absent():
    h = mr.harmonise(exposure([("1:100:A:G", 0.5, 0.05, 0.7, None)]),
                     outcome([("1:100:A:G", "NA", "NA", 0.68)]),
                     exposure_dataset=FG_OLINK, annotation=FakeAnnotation())
    assert row(h, "1:100:A:G")["in_outcome"] is False


def test_instrument_missing_from_outcome_is_kept_with_null_outcome_columns():
    h = mr.harmonise(
        exposure([("1:100:A:G", 0.5, 0.05, 0.7, None), ("5:500:G:C", 0.3, 0.05, 0.6, None)]),
        outcome([("1:100:A:G", 0.1, 0.02, 0.68)]),
        exposure_dataset=FG_OLINK, annotation=FakeAnnotation(),
    )
    r = row(h, "5:500:G:C")
    assert r["in_outcome"] is False and r["outcome_match"] == "absent"
    assert r["beta_y"] is None and r["se_y"] is None
    assert h.frame.height == 2


def test_variant_spellings_normalise_to_one_key():
    x = pl.DataFrame({"variant": ["chrX_1000_A_G"], "beta": [0.2], "se": [0.02]})
    h = mr.harmonise(x, outcome([("23:1000:A:G", 0.1, 0.02, None)]), annotation=FakeAnnotation())
    assert h.frame["variant"].to_list() == ["23:1000:A:G"] and h.frame["in_outcome"][0]


def test_duplicate_exposure_or_several_phenotypes_are_refused():
    x = exposure([("1:100:A:G", 0.5, 0.05, 0.7, None), ("1:100:A:G", 0.6, 0.05, 0.7, None)])
    with pytest.raises(GeneticsUsageError, match="more than once"):
        mr.harmonise(x, outcome([("1:100:A:G", 0.1, 0.02, 0.68)]), annotation=no_lookup)
    y = outcome([("1:100:A:G", 0.1, 0.02, 0.68)]).vstack(
        outcome([("1:100:A:G", 0.1, 0.02, 0.68)]).with_columns(pl.lit("CD").alias("phenotype")))
    with pytest.raises(GeneticsUsageError, match="one outcome phenotype"):
        mr.harmonise(exposure([("1:100:A:G", 0.5, 0.05, 0.7, None)]), y, annotation=no_lookup)


# --------------------------------------------------------------------------- AF cascade


def test_cascade_outcome_af_first_and_flags_a_discordant_allele():
    h = mr.harmonise(
        exposure([("1:100:A:G", 0.5, 0.05, 0.7, None), ("1:200:A:G", 0.5, 0.05, 0.7, None)]),
        outcome([("1:100:A:G", 0.1, 0.02, 0.65), ("1:200:A:G", 0.1, 0.02, 0.3)]),
        exposure_dataset=FG_OLINK, annotation=no_lookup,
    )
    ok, bad = row(h, "1:100:A:G"), row(h, "1:200:A:G")
    assert ok["af_checked_against"] == "outcome" and ok["af_reference"] == 0.65
    assert ok["af_diff"] == pytest.approx(0.05) and ok["af_flag"] is False
    assert bad["af_flag"] is True and bad["af_diff"] == pytest.approx(0.4)


def test_af_tolerance_is_a_parameter():
    h = mr.harmonise(exposure([("1:100:A:G", 0.5, 0.05, 0.7, None)]),
                     outcome([("1:100:A:G", 0.1, 0.02, 0.6)]),
                     annotation=no_lookup, af_tolerance=0.05)
    assert row(h, "1:100:A:G")["af_flag"] is True


def test_ukb_ppp_aaf_is_compared_on_the_maf_scale_and_never_passes():
    # UKB_PPP's loaded aaf is a MAF although the outcome says alt is common
    h = mr.harmonise(
        exposure([("1:100:A:G", 0.5, 0.05, 0.3, None), ("1:200:C:T", 0.5, 0.05, 0.1, None)]),
        outcome([("1:100:A:G", 0.1, 0.02, 0.72), ("1:200:C:T", 0.1, 0.02, 0.1)]),
        exposure_dataset=UKB_PPP, annotation=no_lookup,
    )
    r = row(h, "1:100:A:G")
    assert r["af_checked_against"] == "outcome" and r["af_scale"] == "maf"
    assert r["af_diff"] == pytest.approx(0.02) and r["af_flag"] is False
    assert any("minor-allele" in n for n in h.notes)
    status, detail = _checks(h)["af_concordance"]
    assert status == "warn" and "not a swapped allele" in detail


def test_alt_af_below_half_everywhere_stays_alt_and_flags_a_swap():
    # a FinnGen Olink-like frame whose instruments are all the minor allele
    h = mr.harmonise(
        exposure([("1:100:A:G", 0.5, 0.05, 0.3, None), ("1:200:C:T", 0.5, 0.05, 0.1, None)]),
        outcome([("1:100:A:G", 0.1, 0.02, 0.7), ("1:200:C:T", 0.1, 0.02, 0.1)]),
        exposure_dataset=FG_OLINK, annotation=no_lookup,
    )
    assert set(h.frame["af_scale"]) == {"alt"} and not h.notes
    r = row(h, "1:100:A:G")
    assert r["af_flag"] is True and r["af_diff"] == pytest.approx(-0.4)


def test_maf_column_is_the_maf_scale_for_any_dataset():
    h = mr.harmonise(exposure([("1:100:A:G", 0.5, 0.05, 0.3, None)], af="maf"),
                     outcome([("1:100:A:G", 0.1, 0.02, 0.7)]),
                     exposure_dataset=FG_OLINK, annotation=no_lookup)
    r = row(h, "1:100:A:G")
    assert r["af_scale"] == "maf" and r["af_checked_against"] == "outcome"
    assert r["af_flag"] is False


def test_all_null_aaf_does_not_hide_a_populated_maf():
    x = pl.DataFrame({"variant": ["1:100:A:G"], "beta": [0.5], "se": [0.05],
                      "aaf": ["NA"], "maf": [0.3]})
    h = mr.harmonise(x, outcome([("1:100:A:G", 0.1, 0.02, 0.7)]), annotation=no_lookup)
    r = row(h, "1:100:A:G")
    assert r["af_x"] == 0.3 and r["af_scale"] == "maf" and r["af_flag"] is False


def test_real_alt_af_is_not_read_as_maf():
    h = mr.harmonise(
        exposure([("1:100:A:G", 0.5, 0.05, 0.7, None), ("1:200:C:T", 0.5, 0.05, 0.1, None)]),
        outcome([("1:100:A:G", 0.1, 0.02, 0.72), ("1:200:C:T", 0.1, 0.02, 0.1)]),
        annotation=no_lookup,
    )
    assert set(h.frame["af_checked_against"]) == {"outcome"}
    assert set(h.frame["af_scale"]) == {"alt"}


def test_finngen_annotation_for_a_variant_with_no_outcome_row():
    ann = FakeAnnotation(finngen={"5:500:G:C": {"AF": 0.61}}, gnomad={"5:500:G:C": {"AF_nfe": 0.1}})
    h = mr.harmonise(exposure([("5:500:G:C", 0.3, 0.05, 0.6, None)]),
                     outcome([("1:100:A:G", 0.1, 0.02, 0.68)]),
                     exposure_dataset=UKB_GWAS, annotation=ann)
    r = row(h, "5:500:G:C")
    assert r["af_checked_against"] == "finngen_annotation" and r["af_reference"] == 0.61
    assert [c[0] for c in ann.calls] == ["finngen"]


def test_gnomad_only_for_a_non_finnish_exposure_the_finngen_side_lacks():
    ann = FakeAnnotation(gnomad={"5:500:G:C": {"AF_nfe": 0.58, "AF_fin": 0.4},
                                 "6:600:G:A": {"AF_nfe": "NA", "AF_fin": 0.2}})
    h = mr.harmonise(
        exposure([("5:500:G:C", 0.3, 0.05, 0.6, None), ("6:600:G:A", 0.3, 0.05, 0.6, None)]),
        outcome([("1:100:A:G", 0.1, 0.02, 0.68)]),
        exposure_dataset=UKB_GWAS, annotation=ann,
    )
    nfe, fin = row(h, "5:500:G:C"), row(h, "6:600:G:A")
    assert nfe["af_checked_against"] == "gnomad_nfe" and nfe["af_reference"] == 0.58
    assert fin["af_checked_against"] == "gnomad_fin" and fin["af_flag"] is True
    assert [c[0] for c in ann.calls] == ["finngen", "gnomad"]


def test_never_gnomad_for_a_finngen_exposure():
    ann = FakeAnnotation(gnomad={"5:500:G:C": {"AF_nfe": 0.58}})
    h = mr.harmonise(exposure([("5:500:G:C", 0.3, 0.05, 0.6, None)]),
                     outcome([("1:100:A:G", 0.1, 0.02, 0.68)]),
                     exposure_dataset=FG_OLINK, annotation=ann)
    assert row(h, "5:500:G:C")["af_checked_against"] == "none"
    assert [c[0] for c in ann.calls] == ["finngen"]


def test_exposure_without_af_is_unchecked():
    # deCODE / Open Targets: no frequency of their own
    x = pl.DataFrame({"variant": ["1:100:A:G"], "beta": [0.5], "se": [0.05],
                      "aaf": [None], "maf": [None]}, schema_overrides={"aaf": pl.Float64,
                                                                        "maf": pl.Float64})
    h = mr.harmonise(x, outcome([("1:100:A:G", 0.1, 0.02, 0.3)]), annotation=no_lookup)
    r = row(h, "1:100:A:G")
    assert r["af_checked_against"] == "none" and r["af_reference"] is None
    assert r["af_diff"] is None and r["af_flag"] is None


def test_default_annotation_goes_through_the_sdk(monkeypatch):
    import genetics_mcp_server.sdk as sdk

    calls = []

    def fake(*, variants, source):
        calls.append(source)
        return pl.DataFrame({"chr": ["5"], "pos": ["500"], "ref": ["G"], "alt": ["C"], "AF": ["0.6"]})

    monkeypatch.setattr(sdk, "variant_annotation", fake)
    h = mr.harmonise(exposure([("5:500:G:C", 0.3, 0.05, 0.6, None)]),
                     outcome([("1:100:A:G", 0.1, 0.02, 0.68)]), exposure_dataset=FG_OLINK)
    assert calls == ["finngen"] and row(h, "5:500:G:C")["af_checked_against"] == "finngen_annotation"


# --------------------------------------------------------------------------- flags


def test_orientation_unverifiable_above_the_tolerance_maf_for_any_variant():
    # (1 - 0.2) / 2 = 0.4: above it a mislabelled allele moves the AF by less than 0.2
    h = mr.harmonise(
        exposure([("1:100:A:T", 0.5, 0.05, 0.45, None), ("1:200:C:G", 0.5, 0.05, 0.1, None),
                  ("1:300:A:G", 0.5, 0.05, 0.45, None), ("1:400:A:G", 0.5, 0.05, 0.38, None)]),
        outcome([("1:100:A:T", 0.1, 0.02, 0.46), ("1:200:C:G", 0.1, 0.02, 0.1),
                 ("1:300:A:G", 0.1, 0.02, 0.56), ("1:400:A:G", 0.1, 0.02, 0.38)]),
        annotation=no_lookup,
    )
    flags = {r["variant"]: (r["palindromic"], r["orientation_unverifiable"]) for r in h.frame.to_dicts()}
    assert flags == {"1:100:A:T": (True, True), "1:200:C:G": (True, False),
                     "1:300:A:G": (False, True), "1:400:A:G": (False, False)}
    tight = mr.harmonise(exposure([("1:400:A:G", 0.5, 0.05, 0.38, None)]),
                         outcome([("1:400:A:G", 0.1, 0.02, 0.38)]),
                         annotation=no_lookup, af_tolerance=0.3)
    assert row(tight, "1:400:A:G")["orientation_unverifiable"] is True


def test_coding_uses_the_plots_consequence_list():
    h = mr.harmonise(
        exposure([("1:100:A:G", 0.5, 0.05, 0.7, "missense_variant"),
                  ("1:200:A:G", 0.5, 0.05, 0.7, "intron_variant"),
                  ("1:300:A:G", 0.5, 0.05, 0.7, None)]),
        outcome([("1:100:A:G", 0.1, 0.02, 0.7), ("1:200:A:G", 0.1, 0.02, 0.7),
                 ("1:300:A:G", 0.1, 0.02, 0.7)]),
        annotation=no_lookup,
    )
    assert h.frame["coding"].to_list() == [True, False, None]


# --------------------------------------------------------------------------- overlap


@pytest.mark.parametrize("x, y, status, words", [
    (["ukb"], ["ukb"], "warn", "same collection(s) ukb"),
    (["finngen", "ukb"], ["ukb", "mvp"], "warn", "same collection(s) ukb"),
    (["ukb"], ["iibdgc"], "pass", "do not establish independent"),
    (None, ["ukb"], "not_computable", "exposure"),
    (["ukb"], None, "not_computable", "outcome"),
])
def test_sample_overlap_statement(x, y, status, words):
    got_status, detail = mr.sample_overlap_statement(
        {"cohorts": x} if x else {}, {"cohorts": y} if y else None
    )
    assert got_status == status and words in detail
    assert "one-sample" not in detail


# --------------------------------------------------------------------------- checks


def _checks(h, **kw):
    return {r["check"]: (r["status"], r["detail"]) for r in mr.checks(h, **kw).to_dicts()}


def test_checks_all_pass():
    h = mr.harmonise(
        exposure([("1:100:A:G", 0.5, 0.05, 0.7, "intron_variant"),
                  ("1:200:A:G", 0.4, 0.05, 0.6, "intron_variant")]),
        outcome([("1:100:A:G", 0.1, 0.02, 0.7), ("1:200:A:G", 0.1, 0.02, 0.6)]),
        exposure_dataset=UKB_GWAS, outcome_dataset=IIBDGC, annotation=no_lookup,
    )
    got = _checks(h, steiger={"correct_causal_direction": True, "steiger_pval": 1e-10},
                  heterogeneity={"Q": 1.2, "Q_df": 1, "Q_pval": 0.27})
    assert {k: v[0] for k, v in got.items()} == {
        "instrument_strength": "pass", "af_concordance": "pass", "pleiotropy_proxy": "pass",
        "steiger": "pass", "heterogeneity": "pass", "sample_overlap": "pass",
    }
    assert "mean F 82.0" in got["instrument_strength"][1]


def test_checks_warn_and_fail_statuses():
    h = mr.harmonise(
        exposure([("1:100:A:G", 0.5, 0.05, 0.7, "missense_variant"),
                  ("1:200:A:G", 0.1, 0.05, 0.6, "intron_variant"),
                  ("1:300:A:G", 0.5, 0.05, 0.7, None)]),
        outcome([("1:100:A:G", 0.1, 0.02, 0.7), ("1:200:A:G", 0.1, 0.02, 0.2),
                 ("1:300:A:G", 0.1, 0.02, 0.7)]),
        exposure_dataset=FG_OLINK, outcome_dataset=FG_R14, annotation=no_lookup,
    )
    got = _checks(h, steiger={"correct_causal_direction": False, "steiger_pval": 0.01},
                  heterogeneity={"Q": 30.0, "Q_df": 2, "Q_pval": 1e-7})
    assert got["instrument_strength"][0] == "warn" and "1 below 10" in got["instrument_strength"][1]
    assert got["af_concordance"] == ("fail", "1 flagged, 0 orientation unverifiable, 0 unchecked of 3")
    assert got["pleiotropy_proxy"][0] == "warn" and "1 of 2" in got["pleiotropy_proxy"][1]
    assert got["steiger"][0] == "fail" and got["heterogeneity"][0] == "warn"
    assert got["sample_overlap"][0] == "warn" and "treat as overlapping" in got["sample_overlap"][1]


def test_checks_not_computable_without_estimator_results_or_datasets():
    h = mr.harmonise(exposure([("1:100:A:G", 0.5, 0.05, 0.7, None)]),
                     outcome([("1:100:A:G", 0.1, 0.02, 0.7)]), annotation=no_lookup)
    got = _checks(h.frame)
    assert got["steiger"] == ("not_computable", "estimator result not supplied")
    assert got["heterogeneity"] == ("not_computable", "estimator result not supplied")
    assert got["sample_overlap"][0] == "not_computable"
    assert got["pleiotropy_proxy"][0] == "not_computable"


def test_checks_weak_instruments_fail_and_unchecked_warn():
    x = pl.DataFrame({"variant": ["1:100:A:G"], "beta": [0.1], "se": [0.05]})
    h = mr.harmonise(x, outcome([("1:100:A:G", 0.1, 0.02, 0.7)]), annotation=no_lookup)
    got = _checks(h)
    assert got["instrument_strength"][0] == "fail"
    assert got["af_concordance"][0] == "warn" and "1 unchecked" in got["af_concordance"][1]


def test_checks_count_only_instruments_in_the_outcome():
    h = mr.harmonise(
        exposure([("1:100:A:G", 0.5, 0.05, 0.7, None), ("5:500:G:C", 0.01, 0.05, 0.6, None)]),
        outcome([("1:100:A:G", 0.1, 0.02, 0.7)]),
        exposure_dataset=FG_OLINK, annotation=FakeAnnotation(),
    )
    assert _checks(h)["instrument_strength"][0] == "pass"


def test_checks_empty_frame_is_not_computable():
    h = mr.harmonise(exposure([("5:500:G:C", 0.3, 0.05, 0.6, None)]),
                     outcome([("1:100:A:G", 0.1, 0.02, 0.7)]),
                     exposure_dataset=FG_OLINK, annotation=FakeAnnotation())
    got = _checks(h)
    assert got["instrument_strength"][0] == "not_computable"
    assert got["af_concordance"][0] == "not_computable"


# --------------------------------------------------------------------------- instruments

from genetics_mcp_server.sdk.errors import GeneticsError  # noqa: E402

GENE = {"symbol": "PROT", "chr": 1, "gene_start": 1_000_000, "gene_end": 1_010_000}
PANELS = pl.DataFrame({"panel": ["finngen_r12"], "fields": [["r", "r2"]], "r2_floor": [0.01],
                       "max_distance": [3_000_000]})
NON_FINNISH = {"dataset": "SETS", "cohorts": ["ukb"], "pseudo_credible_sets": False}
PSEUDO = {"dataset": "SETS", "cohorts": ["iceland"], "pseudo_credible_sets": True}


def cs(rows, trait="PROT", dataset="SETS"):
    """credible_sets_v rows: (variant, cs_id, pip, mlog10p, maf)."""
    return pl.DataFrame([
        {"variant": v, "cs_id": c, "pip": p, "mlog10p": m, "maf": f, "beta": 0.1 * (i + 1),
         "se": 0.01, "trait": trait, "trait_original": trait, "dataset": dataset,
         "cell_type": None}
        for i, (v, c, p, m, f) in enumerate(rows)
    ], schema_overrides={"maf": pl.Float64, "cell_type": pl.String})


class FakeLD:
    """Symmetric pair table {(a, b): r}; variants outside `panel` raise not-in-panel."""

    def __init__(self, pairs=None, panel=None, one_sided=None):
        self.r = {}
        for (a, b), r in (pairs or {}).items():
            self.r[(a, b)] = self.r[(b, a)] = r
        self.r.update(one_sided or {})
        self.panel = panel
        self.calls = []

    def __call__(self, variant, *, window, r2_threshold, panel, fields):
        self.calls.append((variant, window, r2_threshold, panel, tuple(fields)))
        if self.panel is not None and variant not in self.panel:
            exc = GeneticsError(f"{variant} not in panel")
            exc.code = "ld_variant_not_in_panel"
            raise exc
        rows = [{"variant": b, "r": r, "r2": r * r} for (a, b), r in self.r.items()
                if a == variant and r * r >= r2_threshold]
        return pl.DataFrame(rows) if rows else pl.DataFrame()


class FakeSQL:
    def __init__(self, genes=(GENE,), sets=None):
        self.genes, self.sets, self.queries = list(genes), sets, []

    def __call__(self, query, *, max_rows):
        self.queries.append(query)
        if "gene_annotations_v" in query:
            return pl.DataFrame(self.genes, schema={"symbol": pl.String, "chr": pl.Int64,
                                                   "gene_start": pl.Int64, "gene_end": pl.Int64})
        return self.sets


def run_instruments(frame, *, ld=None, sql=None, dataset=NON_FINNISH, annotation=None, **kw):
    return mr.instruments(
        frame, exposure_dataset=dataset, ld=ld or FakeLD(), sql=sql or FakeSQL(),
        annotation=annotation or FakeAnnotation(), ld_panels=lambda: PANELS, **kw)


def kept(result):
    return result.frame["variant"].to_list()


def reasons(result):
    return {(d["cs_id"], d["variant"]): d["reason"] for d in result.dropped.to_dicts()}


def test_lead_is_max_pip_above_the_maf_floor_and_a_set_without_one_contributes_nothing():
    frame = cs([
        ("1:1000100:A:G", "a", 0.9, 20.0, 0.001),   # rare pip leader of set a
        ("1:1000200:A:G", "a", 0.1, 15.0, 0.2),
        ("1:1500000:C:T", "b", 1.0, 30.0, 0.005),   # set b has nothing at the floor
    ])
    result = run_instruments(frame)
    assert kept(result) == ["1:1000200:A:G"]
    assert result.frame["af_source"].to_list() == ["credible_set maf"]
    assert reasons(result)[("b", None)] == "no variant with MAF >= 0.01"


def test_null_maf_takes_the_annotation_af_finngen_first_then_gnomad_for_non_finnish():
    frame = cs([("1:1000100:A:G", "a", 0.9, 20.0, None),
                ("1:1600000:A:G", "b", 0.9, 25.0, None)])
    ann = FakeAnnotation(finngen={"1:1000100:A:G": {"AF": 0.97}},
                         gnomad={"1:1600000:A:G": {"AF_nfe": 0.3}})
    result = run_instruments(frame, annotation=ann)
    by = {r["variant"]: r for r in result.frame.to_dicts()}
    assert by["1:1000100:A:G"]["maf"] == pytest.approx(0.03)
    assert by["1:1000100:A:G"]["af_source"] == "finngen_annotation"
    assert by["1:1600000:A:G"]["af_source"] == "gnomad_nfe"
    assert [s for s, _ in ann.calls] == ["finngen", "gnomad"]


def test_finnish_exposure_never_consults_gnomad_and_unknown_af_fails_the_floor():
    frame = cs([("1:1000100:A:G", "a", 0.9, 20.0, None), ("1:1000300:A:G", "a", 0.1, 9.0, 0.4)])
    ann = FakeAnnotation(gnomad={"1:1000100:A:G": {"AF_nfe": 0.3}})
    result = run_instruments(frame, annotation=ann,
                             dataset={"dataset": "SETS", "cohorts": ["finngen"]})
    assert kept(result) == ["1:1000300:A:G"]
    assert [s for s, _ in ann.calls] == ["finngen"]


def test_pseudo_credible_sets_are_refused_by_default_naming_dataset_and_reason():
    frame = cs([("1:1000100:A:G", "chr1_1000200_A_G", 0.9, 20.0, 0.2)])
    with pytest.raises(GeneticsUsageError, match=r"SETS has pseudo credible sets.*tens of kb"):
        run_instruments(frame, dataset=PSEUDO)


def test_pseudo_credible_sets_are_always_refused_and_no_flag_is_offered():
    frame = cs([
        ("1:1000100:A:G", "chr1_1000200_A_G", 0.9, 20.0, 0.2),
        ("1:1000200:A:G", "chr1_1000200_A_G", 0.1, 19.0, 0.2),
    ])
    with pytest.raises(GeneticsUsageError, match="pseudo credible sets") as err:
        run_instruments(frame, dataset=PSEUDO)
    assert "allow_pseudo" not in str(err.value) and "=True" not in str(err.value)
    with pytest.raises(TypeError):
        run_instruments(frame, dataset=PSEUDO, allow_pseudo_credible_sets=True)


def test_d6_refuses_a_protein_with_trans_sets_only():
    frame = cs([("7:5000000:A:G", "t", 0.9, 40.0, 0.3)])
    with pytest.raises(GeneticsUsageError) as err:
        run_instruments(frame)
    text = str(err.value)
    assert "PROT" in text and "chr1:0-2010000" in text and "trans sets only" in text


def test_d6_refuses_when_no_cis_lead_passes_floor_and_threshold():
    frame = cs([("1:1000100:A:G", "a", 0.9, 6.58, 0.3), ("1:1000500:A:G", "b", 0.9, 20.0, 0.001)])
    with pytest.raises(GeneticsUsageError, match=r"no credible set with a lead at MAF >= 0.01 "
                                                 r"and p <= 5e-08.*1 below the MAF floor, 1 above"):
        run_instruments(frame)


def test_cis_window_comes_from_the_gene_row_and_the_trait_is_its_symbol():
    sql = FakeSQL(genes=[{"symbol": "PROT", "chr": 1, "gene_start": 5_000_000,
                          "gene_end": 5_100_000}])
    frame = cs([("1:3999999:A:G", "a", 0.9, 30.0, 0.3), ("1:4000000:A:G", "b", 0.9, 20.0, 0.3),
                ("1:6100000:A:G", "c", 0.9, 20.0, 0.3), ("1:6100001:A:G", "d", 0.9, 50.0, 0.3)])
    result = run_instruments(frame, sql=sql)
    assert sorted(kept(result)) == ["1:4000000:A:G", "1:6100000:A:G"]
    assert "symbol = 'PROT'" in sql.queries[0]
    assert result.cis_region == {"gene": "PROT", "chr": 1, "start": 4_000_000,
                                 "end": 6_100_000, "cis_window": 1_000_000}
    assert any("HGNC symbol" in n for n in result.notes)


def test_cis_refuses_a_trait_that_is_not_one_gene_row():
    with pytest.raises(GeneticsUsageError, match="0 rows"):
        run_instruments(cs([("1:1000100:A:G", "a", 0.9, 20.0, 0.3)]), sql=FakeSQL(genes=[]))


def test_mhc_lead_is_excluded_unless_asked_and_genome_wide_skips_the_gene():
    frame = cs([("6:30000000:A:G", "m", 0.9, 50.0, 0.3), ("2:100:A:G", "x", 0.9, 9.0, 0.3)])
    sql = FakeSQL()
    result = run_instruments(frame, mode="genome_wide", sql=sql)
    assert kept(result) == ["2:100:A:G"]
    assert reasons(result)[("m", "6:30000000:A:G")] == "lead in MHC"
    assert sql.queries == []
    both = run_instruments(frame, mode="genome_wide", exclude_mhc=False)
    assert kept(both) == ["6:30000000:A:G", "2:100:A:G"]


def test_greedy_pruning_keeps_the_stronger_of_a_correlated_pair_and_skips_its_scan():
    a, b, c = "1:1000100:A:G", "1:1000200:A:G", "1:1300000:A:G"
    frame = cs([(a, "a", 0.9, 30.0, 0.3), (b, "b", 0.9, 20.0, 0.3), (c, "c", 0.9, 10.0, 0.3)])
    ld = FakeLD({(a, b): 0.5})
    result = run_instruments(frame, ld=ld)
    assert kept(result) == [a, c]
    assert reasons(result)[("b", b)] == f"r2 0.250 with stronger lead {a}"
    assert [call[0] for call in ld.calls] == [a, c]
    assert ld.calls[0][1:] == (3_000_000, 0.01, "finngen_r12", ("r", "r2"))


def test_a_lead_the_panel_lacks_is_listed_as_ld_unknown_never_kept():
    a, b = "1:1000100:A:G", "1:1000200:A:G"
    frame = cs([(a, "a", 0.9, 30.0, 0.3), (b, "b", 0.9, 20.0, 0.3)])
    result = run_instruments(frame, ld=FakeLD(panel={a}))
    assert kept(result) == [a]
    assert reasons(result)[("b", b)] == f"LD unknown: not in panel finngen_r12: {b} not in panel"


def test_d6_refuses_when_every_lead_is_missing_from_the_panel():
    frame = cs([("1:1000100:A:G", "a", 0.9, 30.0, 0.3), ("1:1000200:A:G", "b", 0.9, 20.0, 0.3)])
    with pytest.raises(GeneticsUsageError) as err:
        run_instruments(frame, ld=FakeLD(panel=set()))
    text = str(err.value)
    assert "PROT" in text and "chr1:0-2010000" in text
    assert "of its 2 lead(s)" in text and "(2 sets)" in text and "finngen_r12" in text
    assert "no LD scan in panel finngen_r12 for any of them" in text
    assert "spelled differently, or no partner above the r2 floor" in text
    assert text.endswith("first: LD unknown: not in panel finngen_r12: "
                         "1:1000100:A:G not in panel")


def test_cis_refuses_a_gene_row_without_a_chromosome():
    gene = {**GENE, "chr": None}
    with pytest.raises(GeneticsUsageError, match=r"PROT's position.*carries no chr\b"):
        run_instruments(cs([("1:1000100:A:G", "a", 0.9, 20.0, 0.3)]),
                        sql=FakeSQL(genes=[gene]))


def test_other_ld_errors_propagate():
    def broken(*_, **__):
        raise GeneticsError("server down")
    with pytest.raises(GeneticsError, match="server down"):
        run_instruments(cs([("1:1000100:A:G", "a", 0.9, 30.0, 0.3)]), ld=broken)


def test_r_matrix_is_square_symmetric_signed_with_zeros_below_the_floor():
    a, b, c = "1:1000100:A:G", "1:1000200:A:G", "1:1300000:A:G"
    frame = cs([(a, "a", 0.9, 30.0, 0.3), (b, "b", 0.9, 20.0, 0.3), (c, "c", 0.9, 10.0, 0.3)])
    result = run_instruments(frame, ld=FakeLD({(a, b): -0.2}), r2_prune=0.1)
    r = result.r
    assert r["variant"].to_list() == [a, b, c] == r.columns[1:]
    m = r.drop("variant").to_numpy()
    assert (m == m.T).all()
    assert m.diagonal().tolist() == [1.0, 1.0, 1.0]
    assert m[0, 1] == pytest.approx(-0.2) and m[0, 2] == 0.0 and m[1, 2] == 0.0
    assert result.panel == {"name": "finngen_r12", "r2_floor": 0.01, "max_distance": 3_000_000,
                            "window": 3_000_000, "scan_r2_threshold": 0.01}


def test_r_matrix_averages_a_rounding_difference_or_a_one_sided_pair_and_says_so():
    a, b, c = "1:1000100:A:G", "1:1000200:A:G", "1:1300000:A:G"
    frame = cs([(a, "a", 0.9, 30.0, 0.3), (b, "b", 0.9, 20.0, 0.3), (c, "c", 0.9, 10.0, 0.3)])
    ld = FakeLD(one_sided={(a, b): -0.2, (b, a): -0.2000005, (a, c): 0.3})
    result = run_instruments(frame, ld=ld, r2_prune=0.5)
    m = result.r.drop("variant").to_numpy()
    assert m[0, 1] == m[1, 0] == pytest.approx(-0.20000025)
    assert m[0, 2] == m[2, 0] == pytest.approx(0.3)
    assert any("2 lead pair(s) differed" in n for n in result.notes)


@pytest.mark.parametrize("ab, ba", [(-0.2, 0.2), (-0.2, -0.3)])
def test_r_matrix_refuses_scans_that_disagree_in_sign_or_beyond_rounding(ab, ba):
    a, b = "1:1000100:A:G", "1:1000200:A:G"
    frame = cs([(a, "a", 0.9, 30.0, 0.3), (b, "b", 0.9, 20.0, 0.3)])
    ld = FakeLD(one_sided={(a, b): ab, (b, a): ba})
    with pytest.raises(GeneticsError, match=rf"r\({a}, {b}\): {ab} from .* {ba} from"):
        run_instruments(frame, ld=ld, r2_prune=0.5)


def test_proxy_is_the_best_r2_neighbour_the_outcome_has_with_its_sign_recorded():
    a, b = "1:1000100:A:G", "1:1300000:A:G"
    best_absent, proxy, weak = "1:1000150:C:T", "1:1000160:G:A", "1:1000170:G:C"
    frame = cs([(a, "a", 0.9, 30.0, 0.3), (b, "b", 0.9, 20.0, 0.3)])
    ld = FakeLD({(a, best_absent): 0.99, (a, proxy): -0.95, (a, weak): 0.9})
    calls = []

    def lookup(variants):
        calls.append(list(variants))
        have = [v for v in variants if v in (proxy, weak, b)]
        return outcome([(v, 0.1, 0.02, 0.3) for v in have])

    result = run_instruments(frame, ld=ld, outcome_lookup=lookup)
    rows = {r["proxy_for"] or r["variant"]: r for r in result.frame.to_dicts()}
    assert rows[a]["variant"] == proxy and rows[a]["is_proxy"]
    assert rows[a]["proxy_r"] == pytest.approx(-0.95)
    assert rows[a]["proxy_r2"] == pytest.approx(0.9025)
    assert (rows[a]["chr"], rows[a]["pos"], rows[a]["ref"], rows[a]["alt"]) == (1, 1000160, "G", "A")
    # r -0.95: the proxy's alt tags the lead's ref, so its beta flips and it has no maf
    assert rows[a]["beta"] == pytest.approx(-0.1) and rows[a]["lead_beta"] == pytest.approx(0.1)
    assert rows[a]["maf"] is None and rows[a]["af_source"] == "proxy: no frequency"
    assert rows[a]["lead_maf"] == pytest.approx(0.3)
    assert rows[b]["beta"] == pytest.approx(0.2) and rows[b]["lead_beta"] is None
    assert rows[b]["is_proxy"] is False and rows[b]["in_outcome"] is True
    assert len(calls) == 1 and set(calls[0]) == {a, b, best_absent, proxy, weak}
    assert any("sign(proxy_r)" in n and "applied here" in n for n in result.notes)
    assert result.requests["outcome_lookup"] == 1


def test_lead_without_an_outcome_proxy_stays_marked_absent():
    a = "1:1000100:A:G"
    result = run_instruments(cs([(a, "a", 0.9, 30.0, 0.3)]), ld=FakeLD({(a, "1:1000150:C:T"): 0.7}),
                             outcome_lookup=lambda variants: pl.DataFrame())
    assert result.frame.to_dicts()[0]["in_outcome"] is False
    assert result.frame["is_proxy"].to_list() == [False]


def test_dict_exposure_fetches_partition_pruned_rows_and_counts_every_request():
    a = "1:1000100:A:G"
    sql = FakeSQL(sets=cs([(a, "a", 0.9, 30.0, None)]))
    ann = FakeAnnotation(finngen={a: {"AF": 0.2}})
    result = run_instruments({"dataset": "SETS", "resource": "res", "trait": "PROT"},
                             sql=sql, annotation=ann)
    assert kept(result) == [a]
    fetch = sql.queries[1]
    assert "FROM credible_sets_v" in fetch and "chr = 1" in fetch
    assert "pos BETWEEN 0 AND 2010000" in fetch and "resource = 'res'" in fetch
    assert result.requests == {"ld": 1, "annotation": 1, "sql": 2, "ld_panels": 1, "total": 5}


def test_dict_exposure_with_no_cis_rows_is_refused_without_claiming_trans():
    sql = FakeSQL(sets=cs([]).clear())
    with pytest.raises(GeneticsUsageError, match="no credible set fine-mapped within the window"):
        run_instruments({"dataset": "SETS", "trait": "PROT"}, sql=sql)


def test_exposure_dataset_is_looked_up_when_not_given(monkeypatch):
    import genetics_mcp_server.sdk as sdk

    monkeypatch.setattr(sdk, "datasets", lambda **_: [
        {"dataset": ["deCODE", "decode"], "resource": "decode", "pseudo_credible_sets": True}],
        raising=False)
    frame = cs([("1:1000100:A:G", "a", 0.9, 30.0, 0.3)], dataset="deCODE")
    with pytest.raises(GeneticsUsageError, match="deCODE has pseudo credible sets"):
        mr.instruments(frame, ld=FakeLD(), sql=FakeSQL(), ld_panels=lambda: PANELS)


def test_a_panel_without_signed_r_is_refused_and_one_exposure_per_call():
    frame = cs([("1:1000100:A:G", "a", 0.9, 30.0, 0.3)])
    with pytest.raises(GeneticsUsageError, match="'sisu42' is not one of ld_panels"):
        run_instruments(frame, panel="sisu42")
    two = pl.concat([frame, cs([("1:1000100:A:G", "a", 0.9, 30.0, 0.3)], trait="OTHER")])
    with pytest.raises(GeneticsUsageError, match="2 values of `trait`"):
        run_instruments(two)


def test_dict_exposure_naming_an_unknown_dataset_is_refused(monkeypatch):
    import genetics_mcp_server.sdk as sdk

    monkeypatch.setattr(sdk, "datasets", lambda **_: [], raising=False)
    with pytest.raises(GeneticsUsageError, match="'deCODE' is not in genetics.datasets"):
        mr.instruments({"dataset": "deCODE", "trait": "PROT"}, ld=FakeLD(), sql=FakeSQL(),
                       ld_panels=lambda: PANELS)


@pytest.mark.parametrize("with_dataset_column", [True, False])
def test_resource_only_dict_of_a_pseudo_dataset_is_refused(monkeypatch, with_dataset_column):
    import genetics_mcp_server.sdk as sdk

    monkeypatch.setattr(sdk, "datasets", lambda **_: [
        {"dataset": ["deCODE", "decode"], "resource": "decode", "pseudo_credible_sets": True}],
        raising=False)
    sets = cs([("1:1000100:A:G", "a", 0.9, 30.0, 0.3)], dataset="deCODE")
    if not with_dataset_column:
        sets = sets.drop("dataset")
    with pytest.raises(GeneticsUsageError, match="(deCODE|decode) has pseudo credible sets"):
        mr.instruments({"resource": "decode", "trait": "PROT"}, ld=FakeLD(),
                       sql=FakeSQL(sets=sets), annotation=FakeAnnotation(),
                       ld_panels=lambda: PANELS)


def test_a_fetched_frame_holding_two_datasets_is_refused():
    sets = pl.concat([cs([("1:1000100:A:G", "a", 0.9, 30.0, 0.3)], dataset="FinnGen_R13_UKBB"),
                      cs([("1:1000200:A:G", "b", 0.9, 30.0, 0.3)],
                         dataset="FinnGen_R13_UKBB_labs")])
    with pytest.raises(GeneticsUsageError, match="2 datasets"):
        run_instruments({"resource": "finngen_ukbb", "trait": "PROT"}, sql=FakeSQL(sets=sets),
                        dataset=None)


def test_a_dataset_without_a_registry_row_is_noted_as_non_finnish_and_not_pseudo(monkeypatch):
    import genetics_mcp_server.sdk as sdk

    monkeypatch.setattr(sdk, "datasets", lambda **_: [], raising=False)
    result = mr.instruments(cs([("1:1000100:A:G", "a", 0.9, 30.0, 0.3)]), ld=FakeLD(),
                            sql=FakeSQL(), annotation=FakeAnnotation(), ld_panels=lambda: PANELS)
    assert result.exposure_dataset is None
    assert any("non-Finnish and not pseudo" in n for n in result.notes)


def test_cis_mode_on_a_frame_notes_the_rows_outside_the_window():
    frame = cs([("1:1000100:A:G", "a", 0.9, 30.0, 0.3), ("7:5000000:A:G", "t", 0.9, 40.0, 0.3)])
    result = run_instruments(frame)
    assert any("1 credible-set row(s) outside the cis window chr1:0-2010000" in n
               for n in result.notes)


# --------------------------------------------------------------------------- run()

import asyncio  # noqa: E402
from pathlib import Path  # noqa: E402

from genetics_mcp_server.sdk import _mr_estimators as est  # noqa: E402

X_ROW = {"dataset_id": "ukbb_pqtl", "resource": "ukbb", "dataset": "SETS", "data_type": "pqtl",
         "trait_type": "quantitative", "version": "PPP", "n_samples": 40000, "cohorts": ["ukb"],
         "pseudo_credible_sets": False,
         "products": {"credible_sets": True, "colocalization": {"partners": ["fg_gwas"]}}}
FG_ROW = {"dataset_id": "fg_gwas", "resource": "finngen", "dataset": "FG", "data_type": "gwas",
          "trait_type": "binary", "version": "R14", "cohorts": ["finngen"],
          "products": {"credible_sets": True, "summary_stats": True}}
FG_OLD = {"dataset_id": "fg_gwas_r12", "resource": "finngen", "dataset": "FG12",
          "data_type": "gwas", "trait_type": "binary", "cohorts": ["finngen"],
          "products": {"colocalization": {"partners": []}}}
IBD_ROW = {"dataset_id": "ibd", "resource": "ibd_gwas", "dataset": "IIBDGC", "data_type": "gwas",
           "trait_type": "binary", "version": "2026", "cohorts": ["iibdgc"],
           "products": {"credible_sets": True, "summary_stats": True}}
SETS_ONLY = {"dataset_id": "sets_only", "resource": "setsres", "dataset": "SO",
             "data_type": "gwas", "trait_type": "binary", "products": {"credible_sets": True}}
REGISTRY = [X_ROW, FG_ROW, FG_OLD, IBD_ROW, SETS_ONLY]
META = {("finngen", "FG"): (1000, 9000), ("finngen", "FG12"): (900, 9000),
        ("ibd_gwas", "IIBDGC"): (2000, 8000), ("setsres", "SO"): (10, 90)}


class FakeStats:
    """summary_stats over {phenotype: {variant: (beta, se, af)}}; FinnGen-like rows carry no
    case counts unless `cases` is set. Async when asked, recording the peak concurrency."""

    def __init__(self, table, cases=None, is_async=False):
        self.table, self.cases, self.is_async = table, cases, is_async
        self.calls, self.active, self.peak = [], 0, 0

    def rows(self, phenotypes, variants, resource, data_type):
        self.calls.append((tuple(phenotypes), tuple(variants), resource, data_type))
        out = []
        for p in phenotypes:
            for v, (b, s, f) in self.table.get(p, {}).items():
                if v in variants:
                    c, pos, ref, alt = v.split(":")
                    row = {"resource": resource, "version": "R14", "phenotype": p, "chr": int(c),
                           "pos": int(pos), "ref": ref, "alt": alt, "beta": b, "se": s, "af": f}
                    if self.cases:
                        row.update(n_cases=str(self.cases[0]), n_controls=str(self.cases[1]))
                    out.append(row)
        return pl.DataFrame(out) if out else pl.DataFrame()

    def __call__(self, *, phenotypes, variants, resource, data_type):
        if not self.is_async:
            return self.rows(phenotypes, variants, resource, data_type)

        async def go():
            self.active += 1
            self.peak = max(self.peak, self.active)
            await asyncio.sleep(0.01)
            self.active -= 1
            return self.rows(phenotypes, variants, resource, data_type)
        return go()


def fake_phenotypes(*, codes, resource):
    rows = [{"dataset": ds, "trait_original": c, "n_cases": n[0], "n_controls": n[1]}
            for c in codes for (res, ds), n in META.items() if res == resource]
    return pl.DataFrame(rows)


class RunSQL(FakeSQL):
    def __init__(self, coloc=None, **kw):
        super().__init__(**kw)
        self.coloc = coloc

    def __call__(self, query, *, max_rows):
        if "colocalization_v" in query:
            self.queries.append(query)
            return self.coloc if self.coloc is not None else pl.DataFrame()
        return super().__call__(query, max_rows=max_rows)


A, B, C, D = "1:1000100:A:G", "1:1000200:C:T", "1:1000300:G:A", "1:1300000:T:C"


def three_leads():
    return cs([(A, "a", 0.9, 30.0, 0.3), (B, "b", 0.9, 20.0, 0.3), (C, "c", 0.9, 15.0, 0.3)])


def effects(*variants, slope=0.5, phenotype="P1"):
    return {phenotype: {v: (slope * 0.1 * (i + 1) + 0.002 * (-1) ** i, 0.01, 0.3)
                        for i, v in enumerate(variants)}}


def mr_run(exposure, outcome=("finngen", "P1"), *, stats=None, ld=None, sql=None, **kw):
    return mr.run(exposure, outcome, ld=ld or FakeLD(), sql=sql or RunSQL(),
                  ld_panels=lambda: PANELS, annotation=FakeAnnotation(),
                  datasets=lambda include_stats: REGISTRY, phenotypes=fake_phenotypes,
                  summary_stats=stats or FakeStats(effects(A, B, C)), **kw)


def test_estimators_are_re_exported():
    assert all(getattr(mr, name) is getattr(est, name) for name in est.__all__)
    assert set(est.__all__) <= set(mr.__all__)


def test_cis_with_three_correlated_instruments_heads_with_the_correlated_ivw():
    ld = FakeLD({(A, B): 0.25, (B, C): -0.2})
    res = mr_run(three_leads(), ld=ld)
    assert res.headline == "ivw_correlated"
    assert set(res.estimates) == {"ivw_correlated", "ivw", "egger"}
    used = res.harmonised.frame.filter(pl.col("in_outcome"))
    assert res.instruments.r.drop("variant").to_numpy()[0, 1] == pytest.approx(0.25)
    want = est.ivw(used, correl=res.instruments.r)
    assert res.estimates["ivw_correlated"]["estimate"] == pytest.approx(want["estimate"], rel=1e-12)
    assert res.estimates["ivw_correlated"]["method"] == "ivw_correlated"
    assert res.estimates["ivw"]["estimate"] == pytest.approx(est.ivw(used)["estimate"], rel=1e-12)
    assert res.estimates["ivw"]["estimate"] != pytest.approx(want["estimate"], rel=1e-9)
    assert res.parameters["r2_prune"] == 0.1 and res.parameters["mode"] == "cis"
    assert res.parameters["attach_coloc"] is True
    assert res.parameters["max_in_flight"] == mr.MAX_IN_FLIGHT
    assert res.parameters["outcome_chunk"] == mr.OUTCOME_CHUNK
    assert res.sensitivity["leave_one_out"].height == 4
    assert any("treat the instruments as independent" in n for n in res.notes)


PROXY = "1:1000150:G:C"
PROXY_STATS = {"P1": {PROXY: (-0.05, 0.01, 0.3), B: (0.1, 0.01, 0.3), C: (0.16, 0.01, 0.3)}}


def test_a_proxied_instrument_reaches_the_correlated_ivw_with_the_chain_r():
    ld = FakeLD({(A, B): 0.25, (A, C): 0.1, (A, PROXY): -0.95})
    stats = FakeStats(PROXY_STATS)
    res = mr_run(three_leads(), ld=ld, stats=stats)
    used = res.harmonised.frame.filter(pl.col("in_outcome"))
    assert PROXY in used["variant"].to_list()
    assert len(stats.calls) == 1 and PROXY in stats.calls[0][1]
    assert res.headline == "ivw_correlated"
    assert not any("no LD r" in n for n in res.notes)
    assert any(PROXY in n and "chain" in n for n in res.notes)
    r = mr._r_for(used.join(res.instruments.frame.select("variant", "proxy_for", "proxy_r"),
                            on="variant", how="left"), res.instruments.r, {})
    # r(A, B) * r(proxy, A): B's scan does not hold the proxy
    assert r.filter(pl.col("variant") == PROXY)[B].item() == pytest.approx(0.25 * -0.95)
    assert r.filter(pl.col("variant") == PROXY)[PROXY].item() == 1.0
    want = est.ivw(used, correl=r)
    assert res.estimates["ivw_correlated"]["estimate"] == pytest.approx(want["estimate"],
                                                                         rel=1e-12)


def test_a_proxied_instruments_r_prefers_the_other_leads_own_scan_value():
    ld = FakeLD({(A, B): 0.25, (A, PROXY): -0.95, (B, PROXY): -0.3})
    res = mr_run(three_leads(), ld=ld, stats=FakeStats(PROXY_STATS))
    assert res.headline == "ivw_correlated"
    used = res.harmonised.frame.filter(pl.col("in_outcome"))
    scans = {B: {PROXY: (-0.3, 0.09)}}
    r = mr._r_for(used.join(res.instruments.frame.select("variant", "proxy_for", "proxy_r"),
                            on="variant", how="left"), res.instruments.r, scans)
    assert r.filter(pl.col("variant") == PROXY)[B].item() == pytest.approx(-0.3)
    assert res.estimates["ivw_correlated"]["estimate"] == pytest.approx(
        est.ivw(used, correl=r)["estimate"], rel=1e-12)


def test_binary_steiger_runs_on_the_instruments_the_outcome_has():
    stats = FakeStats({"P1": {A: (0.05, 0.01, 0.3), B: (0.1, 0.01, 0.3)}})
    res = mr_run(three_leads(), stats=stats)
    assert res.harmonised.frame.filter(~pl.col("in_outcome")).height == 1
    assert res.sensitivity["steiger"] is not None
    assert res.sensitivity["steiger"]["per_variant"].height == 2
    assert not any("Steiger not computed" in n for n in res.notes)


def test_steiger_note_names_a_case_control_consortium():
    res = mr_run(three_leads(), ("ibd_gwas", "P1"))
    note = next(n for n in res.notes if "not the population prevalence" in n)
    assert "case-control consortium collection (iibdgc)" in note
    fg = mr_run(three_leads())
    note = next(n for n in fg.notes if "not the population prevalence" in n)
    assert "close for a population cohort" in note


def test_one_instrument_is_the_wald_ratio_with_the_sensitivity_skipped_and_said():
    stats = FakeStats({"P1": {A: (0.05, 0.01, 0.3)}})
    res = mr_run(cs([(A, "a", 0.9, 30.0, 0.3)]), stats=stats)
    assert res.headline == "wald" and set(res.estimates) == {"wald"}
    assert res.estimates["wald"]["estimate"] == pytest.approx(0.05 / 0.1)
    assert res.sensitivity["leave_one_out"] is None and res.sensitivity["heterogeneity"] is None
    assert res.sensitivity["single_snp"].height == 3
    assert any("leave-one-out needs 3" in n for n in res.notes)


def test_genome_wide_reports_the_full_method_set_at_the_tighter_prune():
    far = ["2:5000000:A:G", "3:5000000:C:T", "4:5000000:G:A", "5:5000000:T:C"]
    frame = cs([(v, f"s{i}", 0.9, 30.0 - i, 0.3) for i, v in enumerate(far)])
    sql = RunSQL()
    res = mr_run(frame, instruments="genome_wide", stats=FakeStats(effects(*far)), sql=sql)
    assert res.headline == "ivw"
    assert set(res.estimates) == {"ivw", "egger", "weighted_median", "simple_median",
                                  "weighted_mode", "simple_mode"}
    assert res.parameters["r2_prune"] == 0.01 and res.parameters["mode"] == "genome_wide"
    assert not any("gene_annotations_v" in q for q in sql.queries)


def test_a_quantitative_finngen_exposure_is_labelled_sd_from_its_phenotypes_row():
    far = ["2:5000000:A:G", "3:5000000:C:T", "4:5000000:G:A"]
    frame = cs([(v, f"s{i}", 0.9, 30.0 - i, 0.3) for i, v in enumerate(far)],
               trait="BMI_IRN", dataset="FG")
    calls = []

    def phenotypes(*, codes, resource):
        calls.append((resource, tuple(codes)))
        out = fake_phenotypes(codes=[c for c in codes if c != "BMI_IRN"], resource=resource)
        if "BMI_IRN" in codes:
            out = pl.concat([out, pl.DataFrame(
                [{"dataset": "FG", "trait_original": "BMI_IRN", "n_cases": None,
                  "n_controls": None}],
                schema={"dataset": pl.String, "trait_original": pl.String,
                        "n_cases": pl.Int64, "n_controls": pl.Int64})], how="vertical_relaxed")
        return out

    res = mr.run(frame, ("finngen", "P1"), instruments="genome_wide", ld=FakeLD(),
                 sql=RunSQL(), ld_panels=lambda: PANELS, annotation=FakeAnnotation(),
                 datasets=lambda include_stats: REGISTRY, phenotypes=phenotypes,
                 summary_stats=FakeStats(effects(*far)))
    assert res.harmonised.units == {"exposure": "SD (quantitative)", "outcome": "log OR (binary)"}
    assert calls == [("finngen", ("BMI_IRN", "P1"))]


def test_sample_prevalence_is_used_and_noted_unless_one_is_given():
    res = mr_run(three_leads())
    assert res.parameters["prevalence"] == pytest.approx(1000 / 10000)
    assert res.parameters["prevalence_source"] == "sample"
    assert any("not the population prevalence" in n for n in res.notes)
    assert any("taken from phenotypes()" in n for n in res.notes)
    assert res.sensitivity["steiger"]["options"]["prevalence"] == pytest.approx(0.1)
    given = mr_run(three_leads(), prevalence=0.01)
    assert given.parameters["prevalence_source"] == "given"
    assert given.sensitivity["steiger"]["options"]["prevalence"] == 0.01
    assert not any("not the population prevalence" in n for n in given.notes)


def test_outcome_dataset_is_the_one_serving_summary_statistics_and_the_counts_come_from_it():
    res = mr_run(three_leads())
    assert res.harmonised.outcome_dataset["dataset_id"] == "fg_gwas"
    assert res.harmonised.frame["n_cases"].unique().to_list() == [1000.0]
    assert res.versions["outcome"] == {"dataset_id": "fg_gwas", "version": "R14"}
    assert res.versions["outcome_rows"] == ["R14"]


def test_refuses_the_same_trait_in_the_same_dataset():
    with pytest.raises(GeneticsUsageError, match="same trait PROT in the same dataset fg_gwas"):
        mr_run(three_leads(), ("finngen", "PROT"), exposure_dataset=FG_ROW,
               stats=FakeStats(effects(A, B, C, phenotype="PROT")))


def test_refuses_an_outcome_dataset_without_summary_statistics_naming_what_it_has():
    with pytest.raises(GeneticsUsageError, match="sets_only serves no summary statistics "
                                                 r"\(it has credible_sets\)"):
        mr_run(three_leads(), ("setsres", "P1"))


def test_refuses_when_no_instrument_reaches_the_outcome_naming_the_counts():
    # a row whose beta is NA is no outcome effect: the instrument counts as absent
    stats = FakeStats({"P1": {A: (None, 0.01, 0.3)}})
    with pytest.raises(GeneticsUsageError, match=r"3 instrument\(s\), 3 absent"):
        mr_run(three_leads(), stats=stats)
    with pytest.raises(GeneticsUsageError, match="returned no rows .* check the phenotype code"):
        mr_run(three_leads(), stats=FakeStats({}))


def test_refuses_many_exposures_naming_the_lead_only_wald_form_and_bad_arguments():
    with pytest.raises(GeneticsUsageError, match="lead-only Wald"):
        mr.run([{"dataset": "SETS", "trait": "A"}, {"dataset": "SETS", "trait": "B"}],
               ("finngen", "P1"))
    with pytest.raises(GeneticsUsageError, match="max_in_flight must be 1-4"):
        mr_run(three_leads(), max_in_flight=5)
    with pytest.raises(GeneticsUsageError, match="exactly one of"):
        mr.run(three_leads())
    with pytest.raises(GeneticsUsageError, match="unexpected keyword"):
        mr_run(three_leads(), colour="red")


def test_d6_refusal_comes_through_from_instruments():
    trans = cs([("7:5000000:A:G", "t", 0.9, 30.0, 0.3)])
    with pytest.raises(GeneticsUsageError, match="trans sets only"):
        mr_run(trans)


def test_many_outcomes_share_one_selection_and_fetch_in_chunks_within_the_cap():
    codes = [f"P{i}" for i in range(25)]
    table = {c: effects(A, B, C, slope=0.1 * i, phenotype=c)[c] for i, c in enumerate(codes)}
    del table["P7"]
    stats = FakeStats(table, is_async=True)
    ld = FakeLD()
    batch = mr.run(three_leads(), outcomes=[("finngen", c) for c in codes], ld=ld,
                   sql=RunSQL(), ld_panels=lambda: PANELS, annotation=FakeAnnotation(),
                   datasets=lambda include_stats: REGISTRY, phenotypes=fake_phenotypes,
                   summary_stats=stats, max_in_flight=2)
    assert isinstance(batch, mr.MRBatch) and len(batch) == 24
    assert batch.failed["phenotype"].to_list() == ["P7"]
    assert [len(c[0]) for c in stats.calls] == [10, 10, 5]
    assert stats.peak == 2
    assert len(ld.calls) == 3
    assert batch.requests["summary_stats"] == 3 and batch.requests["phenotypes"] == 1
    assert batch.summary().height == 24
    assert batch.summary()["outcome"].to_list()[:2] == ["finngen P0", "finngen P1"]


def test_a_chunk_that_raises_anything_fails_only_its_own_outcomes():
    codes = [f"P{i}" for i in range(25)]
    table = {c: effects(A, B, C, slope=0.1 * i, phenotype=c)[c] for i, c in enumerate(codes)}
    inner = FakeStats(table)

    def stats(*, phenotypes, variants, resource, data_type):
        if "P10" in phenotypes:
            raise RuntimeError("connection reset")
        return inner(phenotypes=phenotypes, variants=variants, resource=resource,
                     data_type=data_type)
    batch = mr.run(three_leads(), outcomes=[("finngen", c) for c in codes], ld=FakeLD(),
                   sql=RunSQL(), ld_panels=lambda: PANELS, annotation=FakeAnnotation(),
                   datasets=lambda include_stats: REGISTRY, phenotypes=fake_phenotypes,
                   summary_stats=stats)
    assert len(batch) == 15
    assert batch.failed["phenotype"].to_list() == codes[10:20]
    assert set(batch.failed["reason"]) == {"connection reset"}


def test_instruments_given_marks_the_selection_parameters_unused():
    inst = mr.Instruments(frame=three_leads().select("variant", "beta", "se", "maf", "cs_id"),
                          dropped=pl.DataFrame(), r=None, panel={}, requests={"total": 0},
                          cis_region=None)
    res = mr_run({"dataset": "SETS", "trait": "PROT"}, instruments=inst)
    for k in ("cis_window", "p_threshold", "maf_floor", "panel", "r2_prune"):
        assert res.parameters[k] == "not used (instruments given)"


def test_budget_refusals_name_the_estimate_and_the_cap(monkeypatch):
    monkeypatch.setattr(mr, "REQUEST_CAP", 12)
    ld = FakeLD()
    with pytest.raises(GeneticsUsageError, match=r"at least \d+ requests .* 12-request cap"):
        mr.run(three_leads(), outcomes=[("finngen", f"P{i}") for i in range(60)], ld=ld,
               sql=RunSQL(), ld_panels=lambda: PANELS, datasets=lambda include_stats: REGISTRY,
               phenotypes=fake_phenotypes, summary_stats=FakeStats({}))
    assert ld.calls == []
    # datasets, phenotypes, 1 gene sql + 3 ld + ld_panels, 1 summary_stats, 1 coloc sql
    monkeypatch.setattr(mr, "REQUEST_CAP", 8)
    with pytest.raises(GeneticsUsageError, match=r"needs about 9 requests .* 8-request cap"):
        mr_run(three_leads())


def _never(*a, **k):
    pytest.fail("a fetch ran before the outcome-count refusal")


def test_a_screen_over_the_outcome_cap_is_refused_before_any_fetch(monkeypatch):
    monkeypatch.setattr(mr, "OUTCOME_CAP", 5)
    with pytest.raises(GeneticsUsageError, match=r"6 outcomes exceed the 5-outcome cap.*split"):
        mr.run({"dataset": "SETS", "trait": "PROT"},
               outcomes=[("finngen", f"P{i}") for i in range(6)], ld=_never, sql=_never,
               ld_panels=_never, datasets=_never, phenotypes=_never, summary_stats=_never)


def test_a_generator_of_outcomes_over_the_cap_gets_the_usage_refusal(monkeypatch):
    monkeypatch.setattr(mr, "OUTCOME_CAP", 5)
    with pytest.raises(GeneticsUsageError, match=r"6 outcomes exceed the 5-outcome cap"):
        mr.run({"dataset": "SETS", "trait": "PROT"},
               outcomes=(("finngen", f"P{i}") for i in range(6)), ld=_never, sql=_never,
               ld_panels=_never, datasets=_never, phenotypes=_never, summary_stats=_never)


def test_a_screen_at_the_outcome_cap_passes_that_check(monkeypatch):
    monkeypatch.setattr(mr, "OUTCOME_CAP", 5)
    monkeypatch.setattr(mr, "REQUEST_CAP", 1)
    # the request-cap refusal comes after the outcome-count check, so reaching it proves
    # exactly OUTCOME_CAP outcomes got past
    with pytest.raises(GeneticsUsageError, match="request cap|-request cap") as err:
        mr.run({"dataset": "SETS", "trait": "PROT"},
               outcomes=[("finngen", f"P{i}") for i in range(5)], ld=_never, sql=_never,
               ld_panels=_never, datasets=_never, phenotypes=_never, summary_stats=_never)
    assert "outcome cap" not in str(err.value)


def test_coloc_rows_are_attached_for_a_partner_outcome_in_one_query():
    coloc = pl.DataFrame({
        "dataset1": ["FG", "FG", "FG"], "trait1": ["P1", "OTHER", "P1"],
        "cs1_id": ["x", "y", "z"], "dataset2": ["SETS", "SETS", "SETS"],
        "trait2": ["PROT", "PROT", "PROT"], "cs2_id": ["a", "a", "c"],
        "PP_H4_abf": [0.95, 0.9, 0.2]})
    sql = RunSQL(coloc=coloc)
    res = mr_run(three_leads(), sql=sql)
    queries = [q for q in sql.queries if "colocalization_v" in q]
    assert len(queries) == 1 and "'a'" in queries[0] and "'P1'" in queries[0]
    assert "WHERE chr IN (1) AND (" in queries[0]
    assert res.coloc.select("instrument", "PP_H4_abf").rows() == [(A, 0.95), (C, 0.2)]
    assert res.requests["colocalization.sql"] == 1


def test_coloc_is_skipped_quietly_for_an_outcome_that_is_not_a_partner():
    sql = RunSQL(coloc=pl.DataFrame({"x": [1]}))
    res = mr_run(three_leads(), ("ibd_gwas", "P1"), sql=sql)
    assert res.coloc is None
    assert not any("colocalization_v" in q for q in sql.queries)
    assert any("not a colocalization partner" in n for n in res.notes)
    off = mr_run(three_leads(), sql=sql, attach_coloc=False)
    assert off.coloc is None and not any("colocalization_v" in q for q in sql.queries)


def test_summary_is_one_row_and_repr_is_short():
    res = mr_run(three_leads())
    s = res.summary()
    assert s.columns == ["exposure", "outcome", "method", "estimate", "se", "p", "ci_low",
                         "ci_high", "n_instruments", "overlap", "checks_failing"]
    assert s.height == 1 and s["method"][0] == "ivw_correlated" and s["n_instruments"][0] == 3
    assert s["exposure"][0] == "SETS PROT" and s["outcome"][0] == "finngen P1"
    assert s["overlap"][0] == "pass"
    assert repr(res).startswith("MRResult(SETS PROT -> finngen P1: ivw_correlated")
    assert "\n" not in repr(res) and len(repr(res)) < 160
    assert res.requests["total"] == sum(v for k, v in res.requests.items() if k != "total")


REF = Path(__file__).parent / "fixtures" / "mr" / "r_reference"
CIS = pl.read_csv(REF / "input_cis_ibd.csv")
PROTEINS = sorted(p for p in CIS["protein"].unique().to_list() if (REF / f"ld_{p}.csv").exists()
                  and CIS.filter(pl.col("protein") == p).height >= 2)


@pytest.mark.parametrize("protein", PROTEINS)
def test_run_headline_equals_the_estimators_on_the_reference_frames(protein):
    rows = CIS.filter(pl.col("protein") == protein)
    ld = pl.read_csv(REF / f"ld_{protein}.csv")
    inst = mr.Instruments(
        frame=rows.select("variant", pl.col("beta_x").alias("beta"), pl.col("se_x").alias("se"),
                          pl.col("af_x").alias("af"), pl.col("n_x").alias("n"),
                          pl.col("rsid").alias("cs_id")),
        dropped=pl.DataFrame(), r=ld, panel={}, requests={"total": 0},
        cis_region={"gene": protein})
    table = {"IBD": {v: (b, s, f) for v, b, s, f in
                     rows.select("variant", "beta_y", "se_y", "af_y").iter_rows()}}
    res = mr.run({"dataset": "SETS", "trait": protein}, ("ibd_gwas", "IBD"), instruments=inst,
                 annotation=FakeAnnotation(), datasets=lambda include_stats: REGISTRY,
                 phenotypes=fake_phenotypes, summary_stats=FakeStats(table, cases=(125992, 1262229)))
    frame = rows.rename({"ncase_y": "n_cases", "ncontrol_y": "n_controls"})
    if not set(rows["variant"]) <= set(ld["variant"]):
        # a caller's r that misses an instrument is not used to drop it: the IVW goes independent
        assert res.headline == "ivw" and "ivw_correlated" not in res.estimates
        assert any("no LD r for these instruments" in n for n in res.notes)
        assert res.estimates["ivw"]["estimate"] == pytest.approx(est.ivw(frame)["estimate"])
        return
    assert res.headline == "ivw_correlated"
    assert res.estimates["ivw_correlated"]["n_instruments"] == rows.height
    for key, want in [("ivw_correlated", est.ivw(frame, correl=ld)), ("ivw", est.ivw(frame))] + (
            [("egger", est.egger(frame, correl=ld))] if rows.height >= 3 else []):
        assert res.estimates[key]["estimate"] == pytest.approx(want["estimate"], rel=1e-10, abs=1e-14)
        assert res.estimates[key]["se"] == pytest.approx(want["se"], rel=1e-10)
    assert res.parameters["prevalence"] == pytest.approx(125992 / (125992 + 1262229))
