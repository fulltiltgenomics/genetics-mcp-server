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
3. Rolling success-rate         - share of conversations labelled successful /
   neutral / unsuccessful over time.

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

SCORES = [1, 2, 3, 4, 5]
# colourblind-friendly red->green ramp for scores 1..5
SCORE_COLORS = {
    1: "#d73027",
    2: "#fc8d59",
    3: "#fee08b",
    4: "#91cf60",
    5: "#1a9850",
}
LABEL_COLORS = {
    "successful": "#1a9850",
    "neutral": "#fee08b",
    "unsuccessful": "#d73027",
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

def _scored(records: list[dict]) -> list[dict]:
    return [r for r in records if isinstance(r.get("llm_quality_score"), int)]


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


def panel_success_rate(ax, records, grid, window, min_n):
    """Rolling share of successful / neutral / unsuccessful labels."""
    labels = ["successful", "neutral", "unsuccessful"]
    xs = []
    series = {lab: [] for lab in labels}
    for day, win in rolling_window(records, grid, window):
        # use all records that have a success_label (label exists for every record)
        labelled = [r for r in win if r.get("success_label") in labels]
        n = len(labelled)
        xs.append(day)
        if n < min_n:
            for lab in labels:
                series[lab].append(np.nan)
            continue
        for lab in labels:
            cnt = sum(1 for r in labelled if r["success_label"] == lab)
            series[lab].append(100.0 * cnt / n)

    for lab in labels:
        ax.plot(xs, series[lab], label=lab, color=LABEL_COLORS[lab], lw=2)
    ax.set_ylabel("share of conversations (%)")
    ax.set_title("Success label rate over time")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", ncol=3, fontsize=8, frameon=False)
    ax.grid(True, alpha=0.3)


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

    fig, axes = plt.subplots(3, 1, figsize=(13, 13), sharex=True)
    panel_score_shares(axes[0], ts, grid, args.window, args.min_n)
    panel_mean_and_volume(axes[1], ts, grid, args.window, args.min_n)
    panel_success_rate(axes[2], ts, grid, args.window, args.min_n)

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
