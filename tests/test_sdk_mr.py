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
