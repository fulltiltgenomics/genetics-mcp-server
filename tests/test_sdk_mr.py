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


def test_allow_pseudo_uses_the_cs_id_variant_when_it_passes_the_floor():
    frame = cs([
        ("1:1000100:A:G", "chr1_1000200_A_G", 0.9, 20.0, 0.2),
        ("1:1000200:A:G", "chr1_1000200_A_G", 0.1, 19.0, 0.2),
        ("1:1800100:A:G", "chr1_1800200_A_G", 0.9, 20.0, 0.2),
        ("1:1800200:A:G", "chr1_1800200_A_G", 0.1, 19.0, 0.001),   # named, below the floor
    ])
    result = run_instruments(frame, dataset=PSEUDO, allow_pseudo_credible_sets=True)
    assert sorted(kept(result)) == ["1:1000200:A:G", "1:1800100:A:G"]
    assert any("pseudo credible sets" in n for n in result.notes)


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
