"""Tests for the shared rolling-window aggregations in analysis_timeseries.

Pure functions, no DB: records are constructed as plain dicts. Covers windowing,
min_n line-breaking, score-share percentage sanity, issue-category dedup per
conversation, and JSON-serializability of the output (no numpy/polars leakage).
"""

import json

import pytest

from genetics_mcp_server.scripts import analysis_timeseries as ats
from genetics_mcp_server.scripts.conversation_prompts import ISSUE_CATEGORIES

ISSUE_NAMES = [c for c, _ in ISSUE_CATEGORIES]
CAT_A, CAT_B = ISSUE_NAMES[0], ISSUE_NAMES[1]


def rec(day, score=None, disposition="good_answer", success="successful", issues=None):
    return {
        "created_at": f"2026-01-{day:02d} 12:00:00",
        "llm_quality_score": score,
        "llm_disposition": disposition,
        "success_label": success,
        "llm_issue_categories": issues or [],
    }


def test_parse_date_formats():
    assert ats.parse_date("2026-01-05 12:00:00").isoformat() == "2026-01-05"
    assert ats.parse_date("2026-01-05T08:30:00").isoformat() == "2026-01-05"
    assert ats.parse_date("2026-01-05").isoformat() == "2026-01-05"
    assert ats.parse_date("") is None
    assert ats.parse_date("not-a-date") is None


def test_prepare_records_sorts_and_skips_undated():
    records = [
        rec(10, score=4),
        {"created_at": "", "llm_quality_score": 3},
        rec(2, score=5),
    ]
    prepared, skipped = ats.prepare_records(records)
    assert skipped == 1
    assert [r["_date"].day for r in prepared] == [2, 10]


def test_score_share_window_includes_neighbours():
    # two conversations one day apart; a 7-day centered window pools both,
    # so each grid day with >= min_n shows the pooled distribution.
    records = [rec(10, score=5), rec(11, score=1)]
    prepared, _ = ats.prepare_records(records)
    grid = ats.daily_grid(prepared[0]["_date"], prepared[-1]["_date"])
    out = ats.score_share_series(prepared, grid, window=7, min_n=2)
    # both grid days pool both conversations -> 50% score5, 50% score1
    assert out["series"]["5"] == pytest.approx([50.0, 50.0])
    assert out["series"]["1"] == pytest.approx([50.0, 50.0])


def test_score_share_min_n_breaks_line_with_none():
    records = [rec(10, score=5), rec(11, score=4)]
    prepared, _ = ats.prepare_records(records)
    grid = ats.daily_grid(prepared[0]["_date"], prepared[-1]["_date"])
    # window=1 isolates each day -> 1 scored each -> below min_n=2 -> all None
    out = ats.score_share_series(prepared, grid, window=1, min_n=2)
    for s in ats.SCORES:
        assert out["series"][str(s)] == [None, None]


def test_score_shares_sum_to_100_when_populated():
    records = [rec(10, score=5), rec(10, score=4), rec(10, score=1)]
    prepared, _ = ats.prepare_records(records)
    grid = ats.daily_grid(prepared[0]["_date"], prepared[-1]["_date"])
    out = ats.score_share_series(prepared, grid, window=1, min_n=1)
    for i, _ in enumerate(out["dates"]):
        total = sum(out["series"][str(s)][i] for s in ats.SCORES)
        assert total == pytest.approx(100.0)


def test_non_quality_dispositions_excluded_from_score_panels():
    # a technical_failure with a score must not count toward the score trend
    records = [
        rec(10, score=5, disposition="good_answer", success="successful"),
        rec(10, score=1, disposition="technical_failure", success="technical_failure"),
    ]
    prepared, _ = ats.prepare_records(records)
    grid = ats.daily_grid(prepared[0]["_date"], prepared[-1]["_date"])
    out = ats.score_share_series(prepared, grid, window=1, min_n=1)
    # only the good_answer counts -> 100% score5
    assert out["series"]["5"][0] == pytest.approx(100.0)
    assert out["series"]["1"][0] == pytest.approx(0.0)


def test_mean_and_volume():
    records = [rec(10, score=4), rec(10, score=2)]
    prepared, _ = ats.prepare_records(records)
    grid = ats.daily_grid(prepared[0]["_date"], prepared[-1]["_date"])
    out = ats.mean_score_and_volume_series(prepared, grid, window=1, min_n=1)
    assert out["series"]["mean"][0] == pytest.approx(3.0)
    assert out["volume"][0] == 2
    assert out["ci_low"][0] < 3.0 < out["ci_high"][0]


def test_mean_volume_present_even_below_min_n():
    records = [rec(10, score=4)]
    prepared, _ = ats.prepare_records(records)
    grid = ats.daily_grid(prepared[0]["_date"], prepared[-1]["_date"])
    out = ats.mean_score_and_volume_series(prepared, grid, window=1, min_n=5)
    assert out["volume"][0] == 1  # volume always reported
    assert out["series"]["mean"][0] is None  # mean broken below min_n


def test_disposition_mix_shares():
    records = [
        rec(10, success="successful"),
        rec(10, success="unsuccessful"),
    ]
    prepared, _ = ats.prepare_records(records)
    grid = ats.daily_grid(prepared[0]["_date"], prepared[-1]["_date"])
    out = ats.disposition_mix_series(prepared, grid, window=1, min_n=1)
    assert out["series"]["successful"][0] == pytest.approx(50.0)
    assert out["series"]["unsuccessful"][0] == pytest.approx(50.0)


def test_issue_category_dedup_per_conversation():
    # one conversation listing CAT_A twice must count CAT_A once
    records = [rec(10, issues=[CAT_A, CAT_A]), rec(10, issues=[CAT_B])]
    prepared, _ = ats.prepare_records(records)
    grid = ats.daily_grid(prepared[0]["_date"], prepared[-1]["_date"])
    out = ats.issue_category_mix_series(prepared, grid, window=1, min_n=1)
    # total instances after dedup = 2 (one A, one B) -> 50/50
    assert out["series"][CAT_A][0] == pytest.approx(50.0)
    assert out["series"][CAT_B][0] == pytest.approx(50.0)


def test_issue_category_min_n_breaks():
    records = [rec(10, issues=[CAT_A])]
    prepared, _ = ats.prepare_records(records)
    grid = ats.daily_grid(prepared[0]["_date"], prepared[-1]["_date"])
    out = ats.issue_category_mix_series(prepared, grid, window=1, min_n=3)
    assert out["series"][CAT_A] == [None]


def test_build_all_series_shape_and_meta():
    records = [rec(10, score=5, issues=[CAT_A]), rec(11, score=3)]
    out = ats.build_all_series(records, window=7, min_n=1)
    assert set(out) == {
        "score_share", "mean_and_volume", "disposition_mix",
        "issue_category_mix", "meta",
    }
    assert out["meta"]["empty"] is False
    assert out["meta"]["total"] == 2
    assert out["meta"]["scored"] == 2
    assert out["meta"]["date_min"] == "2026-01-10"
    assert out["meta"]["date_max"] == "2026-01-11"
    # parallel arrays
    for panel in ("score_share", "disposition_mix", "issue_category_mix"):
        ndates = len(out[panel]["dates"])
        for vals in out[panel]["series"].values():
            assert len(vals) == ndates


def test_build_all_series_empty_input():
    out = ats.build_all_series([{"created_at": "", "llm_quality_score": 3}])
    assert out["meta"]["empty"] is True
    assert out["meta"]["skipped_no_date"] == 1
    assert out["score_share"]["dates"] == []


def test_output_is_json_serializable():
    records = [
        rec(10, score=5, issues=[CAT_A]),
        rec(12, score=2, success="unsuccessful", issues=[CAT_B]),
        rec(14, score=4),
    ]
    out = ats.build_all_series(records, window=7, min_n=1)
    # round-trips with no numpy/polars types leaking
    dumped = json.dumps(out)
    assert json.loads(dumped) == out
