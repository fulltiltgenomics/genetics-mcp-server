"""Plot evaluated conversation quality over time from analyze_conversations output.

Consumes the ``metrics.json`` produced by
``genetics_mcp_server.scripts.analyze_conversations`` (which now includes each
session's ``created_at`` and ``llm_quality_score``) and renders time-series
plots of how well the assistant is doing.

Daily conversation counts are noisy (some days have few or zero conversations),
so every series is smoothed with a centered sliding time window (``--window``
days) rather than plotted per raw day.

Usage:
    python -m genetics_mcp_server.scripts.plot_conversation_scores \
        --metrics /mnt/disks/data/eval/analysis_output/metrics.json
    python -m genetics_mcp_server.scripts.plot_conversation_scores \
        --metrics .../metrics.json --window 14 --out scores_over_time.png

Plots produced (one figure, stacked panels):
1. Per-score share over time   - one line per quality score (1..5), each the
   rolling share of conversations with that score. Directly answers
   "how is the score distribution moving".
2. Rolling mean score + volume - single trend line for average quality (1..5)
   with the daily conversation volume as faint bars for context.
3. Disposition mix             - one line per success_label bucket (successful /
   neutral / unsuccessful / technical_failure / out_of_scope / unfinished /
   weird_or_unclear / unknown) as a share of all conversations over time.
4. Issue category mix          - one line per issue taxonomy category as a share
   of all issues over time (which underlying problems dominate).

Other ways to track "how well we're doing over time" are described in the
module docstring of the report and in the project notes; this script implements
the three most useful as a starting point.
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless: write files, no display needed
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from .conversation_prompts import ISSUE_CATEGORIES  # noqa: E402

SCORES = [1, 2, 3, 4, 5]
# colourblind-friendly red->green ramp for scores 1..5
SCORE_COLORS = {
    1: "#d73027",
    2: "#fc8d59",
    3: "#fee08b",
    4: "#91cf60",
    5: "#1a9850",
}

# every success_label bucket (quality labels + non-quality disposition buckets +
# unknown), plotted together as a share of all conversations. distinct colours so
# none collide on one axis.
DISPOSITION_LABELS = [
    "successful", "neutral", "unsuccessful",
    "technical_failure", "out_of_scope", "unfinished", "weird_or_unclear",
    "unknown",
]
DISPOSITION_COLORS = {
    "successful": "#1a9850",
    "neutral": "#fee08b",
    "unsuccessful": "#d73027",
    "technical_failure": "#762a83",
    "out_of_scope": "#4575b4",
    "unfinished": "#999999",
    "weird_or_unclear": "#fc8d59",
    "unknown": "#000000",
}

# the fixed issue taxonomy (imported so it stays in sync with the analyzer),
# each category given a distinct colour from a 20-colour qualitative map
ISSUE_CATEGORY_NAMES = [c for c, _ in ISSUE_CATEGORIES]
ISSUE_CATEGORY_COLORS = {
    name: plt.cm.tab20(i % 20) for i, name in enumerate(ISSUE_CATEGORY_NAMES)
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _parse_date(value: str):
    """Parse a session created_at string into a date; None if unparseable."""
    if not value:
        return None
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


def load_metrics(path: Path) -> list[dict]:
    records = json.loads(path.read_text())
    if not isinstance(records, list):
        raise ValueError(f"{path} is not a list of metric records")
    return records


def extract_timeseries(records: list[dict]) -> tuple[list, list[dict]]:
    """Return (dates, records) for records that have a parseable date.

    Each returned record is the original dict plus a parsed ``_date`` field.
    """
    out = []
    skipped_no_date = 0
    for r in records:
        d = _parse_date(r.get("created_at", ""))
        if d is None:
            skipped_no_date += 1
            continue
        out.append({**r, "_date": d})
    if skipped_no_date:
        print(f"  Skipped {skipped_no_date} records with no/unparseable created_at "
              "(re-run analyze_conversations to populate it)", file=sys.stderr)
    out.sort(key=lambda r: r["_date"])
    return [r["_date"] for r in out], out


# ---------------------------------------------------------------------------
# Sliding-window smoothing
# ---------------------------------------------------------------------------

def daily_grid(min_date, max_date) -> list:
    days = (max_date - min_date).days
    return [min_date + timedelta(days=i) for i in range(days + 1)]


def rolling_window(
    records: list[dict], grid: list, window_days: int,
):
    """For each day in grid, yield (day, records_in_window).

    The window is centered: [day - w//2, day + w//2] inclusive, where
    w = window_days. Sparse days borrow neighbours so lines don't go jerky.
    """
    half = window_days // 2
    dates = np.array([(r["_date"] - grid[0]).days for r in records])
    for day in grid:
        center = (day - grid[0]).days
        mask = (dates >= center - half) & (dates <= center + half)
        yield day, [records[i] for i in np.nonzero(mask)[0]]


# ---------------------------------------------------------------------------
# Plot panels
# ---------------------------------------------------------------------------

# dispositions that are not agent-quality failures; excluded from the score trend
# so out-of-scope / unfinished / weird / technical conversations don't skew it.
# empty disposition (pre-disposition metrics.json) is kept for backward compatibility.
_NON_QUALITY_DISPOSITIONS = {
    "technical_failure", "out_of_scope", "unfinished", "weird_or_unclear",
}


def _scored(records: list[dict]) -> list[dict]:
    return [
        r for r in records
        if isinstance(r.get("llm_quality_score"), int)
        and r.get("llm_disposition", "") not in _NON_QUALITY_DISPOSITIONS
    ]


def panel_score_shares(ax, records, grid, window, min_n):
    """One line per score (1..5): rolling share of conversations with that score."""
    xs = []
    shares = {s: [] for s in SCORES}
    for day, win in rolling_window(records, grid, window):
        scored = _scored(win)
        n = len(scored)
        xs.append(day)
        if n < min_n:
            for s in SCORES:
                shares[s].append(np.nan)
            continue
        counts = {s: 0 for s in SCORES}
        for r in scored:
            sc = r["llm_quality_score"]
            if sc in counts:
                counts[sc] += 1
        for s in SCORES:
            shares[s].append(100.0 * counts[s] / n)

    for s in SCORES:
        ax.plot(xs, shares[s], label=f"score {s}", color=SCORE_COLORS[s], lw=2)
    ax.set_ylabel("share of conversations (%)")
    ax.set_title(f"Quality score distribution over time ({window}-day centered window)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", ncol=5, fontsize=8, frameon=False)
    ax.grid(True, alpha=0.3)


def panel_mean_and_volume(ax, records, grid, window, min_n):
    """Rolling mean quality score (left axis) + conversation volume bars (right)."""
    xs, means, los, his, vols = [], [], [], [], []
    for day, win in rolling_window(records, grid, window):
        scored = _scored(win)
        xs.append(day)
        vols.append(len(scored))
        if len(scored) < min_n:
            means.append(np.nan)
            los.append(np.nan)
            his.append(np.nan)
            continue
        vals = np.array([r["llm_quality_score"] for r in scored], dtype=float)
        mean = vals.mean()
        sem = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        means.append(mean)
        los.append(mean - 1.96 * sem)
        his.append(mean + 1.96 * sem)

    ax_v = ax.twinx()
    ax_v.bar(xs, vols, width=1.0, color="#999999", alpha=0.25, label="conversations in window")
    ax_v.set_ylabel("conversations in window", color="#666666")
    ax_v.set_ylim(bottom=0)

    ax.plot(xs, means, color="#1f78b4", lw=2.5, label="mean score")
    ax.fill_between(xs, los, his, color="#1f78b4", alpha=0.15, label="95% CI")
    ax.set_ylabel("mean quality score (1-5)", color="#1f78b4")
    ax.set_ylim(1, 5)
    ax.set_title("Rolling mean quality score with conversation volume")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, frameon=False)


def panel_disposition_shares(ax, records, grid, window, min_n):
    """One line per disposition bucket: rolling share of ALL conversations.

    Covers every success_label (successful/neutral/unsuccessful plus the
    non-quality buckets technical_failure/out_of_scope/unfinished/weird and
    unknown). Shares keep small buckets visible despite successful dominating.
    """
    xs = []
    series = {lab: [] for lab in DISPOSITION_LABELS}
    for day, win in rolling_window(records, grid, window):
        n = len(win)
        xs.append(day)
        if n < min_n:
            for lab in DISPOSITION_LABELS:
                series[lab].append(np.nan)
            continue
        counts = {lab: 0 for lab in DISPOSITION_LABELS}
        for r in win:
            lab = r.get("success_label")
            if lab in counts:
                counts[lab] += 1
        for lab in DISPOSITION_LABELS:
            series[lab].append(100.0 * counts[lab] / n)

    for lab in DISPOSITION_LABELS:
        # skip buckets with no data so the legend stays meaningful
        if all(np.isnan(v) or v == 0 for v in series[lab]):
            continue
        ax.plot(xs, series[lab], label=lab, color=DISPOSITION_COLORS[lab], lw=2)
    ax.set_ylabel("share of all conversations (%)")
    ax.set_title(f"Disposition mix over time ({window}-day centered window)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", ncol=4, fontsize=8, frameon=False)
    ax.grid(True, alpha=0.3)


def panel_issue_category_shares(ax, records, grid, window, min_n):
    """One line per issue category: rolling share of all issues in the window.

    Issues come from each conversation's llm_issue_categories (the fixed
    taxonomy, deduplicated per conversation). Shows which underlying problems
    dominate over time rather than absolute volume.
    """
    xs = []
    series = {c: [] for c in ISSUE_CATEGORY_NAMES}
    for day, win in rolling_window(records, grid, window):
        instances = [c for r in win for c in (r.get("llm_issue_categories") or [])]
        total = len(instances)
        xs.append(day)
        if total < min_n:
            for c in ISSUE_CATEGORY_NAMES:
                series[c].append(np.nan)
            continue
        counts = {c: 0 for c in ISSUE_CATEGORY_NAMES}
        for c in instances:
            if c in counts:
                counts[c] += 1
        for c in ISSUE_CATEGORY_NAMES:
            series[c].append(100.0 * counts[c] / total)

    any_data = False
    for c in ISSUE_CATEGORY_NAMES:
        if all(np.isnan(v) or v == 0 for v in series[c]):
            continue
        any_data = True
        ax.plot(xs, series[c], label=c, color=ISSUE_CATEGORY_COLORS[c], lw=2)
    ax.set_ylabel("share of issues (%)")
    ax.set_title(f"Issue category mix over time ({window}-day window, "
                 "deduped per conversation)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", ncol=3, fontsize=7, frameon=False)
    ax.grid(True, alpha=0.3)
    if not any_data:
        ax.text(0.5, 0.5, "no issues in range",
                transform=ax.transAxes, ha="center", va="center", color="#888888")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot evaluated conversation quality over time from "
                    "analyze_conversations metrics.json.",
    )
    parser.add_argument("--metrics", required=True,
                        help="Path to metrics.json from analyze_conversations")
    parser.add_argument("--out", default=None,
                        help="Output image path (default: <metrics dir>/scores_over_time.png)")
    parser.add_argument("--window", type=int, default=7,
                        help="Sliding window width in days, centered (default: 7)")
    parser.add_argument("--min-n", type=int, default=3,
                        help="Minimum conversations in a window to plot a point "
                             "(below this the line breaks instead of spiking; default: 3)")
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        print(f"Error: metrics file not found: {metrics_path}", file=sys.stderr)
        sys.exit(1)

    records = load_metrics(metrics_path)
    dates, ts = extract_timeseries(records)
    if not ts:
        print("Error: no records with a parseable created_at — re-run "
              "analyze_conversations so metrics.json includes session dates.",
              file=sys.stderr)
        sys.exit(1)

    n_scored = len(_scored(ts))
    print(f"  {len(ts)} dated conversations, {n_scored} with an LLM quality score, "
          f"{dates[0]} to {dates[-1]}", file=sys.stderr)
    if n_scored == 0:
        print("Warning: no llm_quality_score values — score panels will be empty. "
              "Run analyze_conversations without --no-llm.", file=sys.stderr)

    grid = daily_grid(dates[0], dates[-1])

    fig, axes = plt.subplots(4, 1, figsize=(13, 17), sharex=True)
    panel_score_shares(axes[0], ts, grid, args.window, args.min_n)
    panel_mean_and_volume(axes[1], ts, grid, args.window, args.min_n)
    panel_disposition_shares(axes[2], ts, grid, args.window, args.min_n)
    panel_issue_category_shares(axes[3], ts, grid, args.window, args.min_n)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    fig.suptitle(
        f"Genie LLM-as-judge quality over time  "
        f"(n={n_scored} scored, {args.window}-day window)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.99))

    out_path = Path(args.out) if args.out else metrics_path.parent / "scores_over_time.png"
    fig.savefig(out_path, dpi=130)
    print(f"  Wrote plot to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
