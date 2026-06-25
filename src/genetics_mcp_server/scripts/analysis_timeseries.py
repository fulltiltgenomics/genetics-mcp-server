"""Single source of truth for conversation-quality time-series aggregations.

The rolling centered-window math that both the matplotlib PNG script
(``plot_conversation_scores.py``) and the frontend Quality-plots tab render is
defined here exactly once, so the two renderers cannot drift.

The functions are pure and dependency-light: they take in-memory records (plain
dicts with at least ``created_at``/``date``, ``llm_quality_score``,
``llm_disposition``, ``success_label`` and ``llm_issue_categories``) plus the two
window params, and return JSON-serializable series (plain floats / ints / strings
/ ``None`` — never numpy/polars types). ``None`` marks a window below ``min_n``
where the renderer should break the line rather than spike.

Series shape (mirrored by the frontend):

    {
        "dates": ["YYYY-MM-DD", ...],          # one entry per day in the grid
        "series": {label: [value_or_None, ...]},  # parallel to "dates"
        ... panel-specific extra keys (e.g. "volume", "ci_low"/"ci_high") ...
    }

``build_all_series`` returns ``{"score_share", "mean_and_volume",
"disposition_mix", "issue_category_mix"}`` keyed to those four panels, plus
``meta`` (date range, counts).
"""

import math
from datetime import date, datetime, timedelta

from .conversation_prompts import ISSUE_CATEGORIES

SCORES = [1, 2, 3, 4, 5]

# every success_label bucket (quality labels + non-quality disposition buckets +
# unknown), plotted together as a share of all conversations.
DISPOSITION_LABELS = [
    "successful", "neutral", "unsuccessful",
    "technical_failure", "out_of_scope", "unfinished", "weird_or_unclear",
    "unknown",
]

# the fixed issue taxonomy, imported so it stays in sync with the analyzer.
ISSUE_CATEGORY_NAMES = [c for c, _ in ISSUE_CATEGORIES]

# dispositions that are not agent-quality failures; excluded from the score trend
# so out-of-scope / unfinished / weird / technical conversations don't skew it.
# empty disposition (pre-disposition records) is kept for backward compatibility.
_NON_QUALITY_DISPOSITIONS = {
    "technical_failure", "out_of_scope", "unfinished", "weird_or_unclear",
}


def parse_date(value) -> date | None:
    """Parse a session created_at value into a date; None if unparseable."""
    if not value:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    # accept "YYYY-MM-DD HH:MM:SS", ISO, or bare date
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text[: len(fmt) + 4], fmt).date()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def prepare_records(records: list[dict]) -> tuple[list[dict], int]:
    """Attach a parsed ``_date`` to each record that has a parseable created_at.

    Returns (sorted records with ``_date``, count skipped for no/bad date).
    ``created_at`` is preferred; ``date`` is accepted as a fallback key.
    """
    out = []
    skipped = 0
    for r in records:
        d = parse_date(r.get("created_at") or r.get("date") or "")
        if d is None:
            skipped += 1
            continue
        out.append({**r, "_date": d})
    out.sort(key=lambda r: r["_date"])
    return out, skipped


def daily_grid(min_date: date, max_date: date) -> list[date]:
    days = (max_date - min_date).days
    return [min_date + timedelta(days=i) for i in range(days + 1)]


def _rolling_windows(records: list[dict], grid: list[date], window_days: int):
    """For each day in grid, yield (day, records_in_window).

    The window is centered: [day - w//2, day + w//2] inclusive, where
    w = window_days. Sparse days borrow neighbours so lines don't go jerky.
    """
    half = window_days // 2
    origin = grid[0]
    offsets = [(r["_date"] - origin).days for r in records]
    for day in grid:
        center = (day - origin).days
        win = [
            records[i]
            for i, off in enumerate(offsets)
            if center - half <= off <= center + half
        ]
        yield day, win


def _scored(records: list[dict]) -> list[dict]:
    """Conversations that count toward the agent-quality score trend."""
    return [
        r for r in records
        if isinstance(r.get("llm_quality_score"), int)
        and r.get("llm_disposition", "") not in _NON_QUALITY_DISPOSITIONS
    ]


def _iso(grid: list[date]) -> list[str]:
    return [d.isoformat() for d in grid]


def score_share_series(
    records: list[dict], grid: list[date], window: int, min_n: int
) -> dict:
    """Rolling share (%) of scored conversations holding each score 1..5.

    Below ``min_n`` scored conversations in a window the value is ``None`` so the
    line breaks instead of spiking.
    """
    shares: dict[str, list] = {str(s): [] for s in SCORES}
    for _day, win in _rolling_windows(records, grid, window):
        scored = _scored(win)
        n = len(scored)
        if n < min_n:
            for s in SCORES:
                shares[str(s)].append(None)
            continue
        counts = {s: 0 for s in SCORES}
        for r in scored:
            sc = r["llm_quality_score"]
            if sc in counts:
                counts[sc] += 1
        for s in SCORES:
            shares[str(s)].append(100.0 * counts[s] / n)
    return {"dates": _iso(grid), "series": shares}


def mean_score_and_volume_series(
    records: list[dict], grid: list[date], window: int, min_n: int
) -> dict:
    """Rolling mean quality score with 95% CI band plus per-window volume.

    ``volume`` is the count of scored conversations in each window (always
    present, even below ``min_n``). ``mean``/``ci_low``/``ci_high`` are ``None``
    below ``min_n``.
    """
    means, los, his, vols = [], [], [], []
    for _day, win in _rolling_windows(records, grid, window):
        scored = _scored(win)
        vols.append(len(scored))
        if len(scored) < min_n:
            means.append(None)
            los.append(None)
            his.append(None)
            continue
        vals = [float(r["llm_quality_score"]) for r in scored]
        n = len(vals)
        mean = sum(vals) / n
        if n > 1:
            var = sum((v - mean) ** 2 for v in vals) / (n - 1)
            sem = math.sqrt(var) / math.sqrt(n)
        else:
            sem = 0.0
        means.append(mean)
        los.append(mean - 1.96 * sem)
        his.append(mean + 1.96 * sem)
    return {
        "dates": _iso(grid),
        "series": {"mean": means},
        "ci_low": los,
        "ci_high": his,
        "volume": vols,
    }


def disposition_mix_series(
    records: list[dict], grid: list[date], window: int, min_n: int
) -> dict:
    """Rolling share (%) of ALL conversations in each success_label bucket.

    Below ``min_n`` total conversations in a window every bucket is ``None``.
    """
    series: dict[str, list] = {lab: [] for lab in DISPOSITION_LABELS}
    for _day, win in _rolling_windows(records, grid, window):
        n = len(win)
        if n < min_n:
            for lab in DISPOSITION_LABELS:
                series[lab].append(None)
            continue
        counts = {lab: 0 for lab in DISPOSITION_LABELS}
        for r in win:
            lab = r.get("success_label")
            if lab in counts:
                counts[lab] += 1
        for lab in DISPOSITION_LABELS:
            series[lab].append(100.0 * counts[lab] / n)
    return {"dates": _iso(grid), "series": series}


def issue_category_mix_series(
    records: list[dict], grid: list[date], window: int, min_n: int
) -> dict:
    """Rolling share (%) of all issue instances per taxonomy category.

    Issues come from each conversation's ``llm_issue_categories`` (the fixed
    taxonomy, deduplicated per conversation). Below ``min_n`` total issue
    instances in a window every category is ``None``.
    """
    series: dict[str, list] = {c: [] for c in ISSUE_CATEGORY_NAMES}
    for _day, win in _rolling_windows(records, grid, window):
        # dedup per conversation so one conversation counts a category once
        instances = [
            c
            for r in win
            for c in set(r.get("llm_issue_categories") or [])
        ]
        total = len(instances)
        if total < min_n:
            for c in ISSUE_CATEGORY_NAMES:
                series[c].append(None)
            continue
        counts = {c: 0 for c in ISSUE_CATEGORY_NAMES}
        for c in instances:
            if c in counts:
                counts[c] += 1
        for c in ISSUE_CATEGORY_NAMES:
            series[c].append(100.0 * counts[c] / total)
    return {"dates": _iso(grid), "series": series}


def build_all_series(
    records: list[dict], window: int = 7, min_n: int = 3
) -> dict:
    """Compute all four panels from raw records in one call.

    Records are prepared (date-parsed, sorted) internally; the grid spans the
    full observed date range. Returns a JSON-serializable dict keyed by panel
    plus ``meta``. If no record has a parseable date, ``meta.empty`` is True and
    the four panel keys hold empty series.
    """
    prepared, skipped = prepare_records(records)
    if not prepared:
        empty = {"dates": [], "series": {}}
        return {
            "score_share": empty,
            "mean_and_volume": {**empty, "ci_low": [], "ci_high": [], "volume": []},
            "disposition_mix": empty,
            "issue_category_mix": empty,
            "meta": {
                "empty": True,
                "skipped_no_date": skipped,
                "total": 0,
                "scored": 0,
                "date_min": None,
                "date_max": None,
                "window": window,
                "min_n": min_n,
            },
        }

    dates = [r["_date"] for r in prepared]
    grid = daily_grid(dates[0], dates[-1])
    return {
        "score_share": score_share_series(prepared, grid, window, min_n),
        "mean_and_volume": mean_score_and_volume_series(prepared, grid, window, min_n),
        "disposition_mix": disposition_mix_series(prepared, grid, window, min_n),
        "issue_category_mix": issue_category_mix_series(prepared, grid, window, min_n),
        "meta": {
            "empty": False,
            "skipped_no_date": skipped,
            "total": len(prepared),
            "scored": len(_scored(prepared)),
            "date_min": dates[0].isoformat(),
            "date_max": dates[-1].isoformat(),
            "window": window,
            "min_n": min_n,
        },
    }
