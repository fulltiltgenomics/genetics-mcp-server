"""Plot evaluated conversation quality over time from analyze_conversations output.

Reads per-conversation analysis either from the SQLite ``conversation_analysis``
/ ``conversation_issue`` tables (``--db``, the nightly-job source of truth) or
from the local-dev ``metrics.json`` produced by
``genetics_mcp_server.scripts.analyze_conversations`` (``--metrics``), and renders
time-series plots of how well the assistant is doing.

All rolling-window aggregation lives in ``analysis_timeseries`` so the PNG here
and the frontend Quality-plots tab share one source of truth and cannot drift;
this module only renders the series it returns.

Daily conversation counts are noisy (some days have few or zero conversations),
so every series is smoothed with a centered sliding time window (``--window``
days) rather than plotted per raw day.

Usage:
    python -m genetics_mcp_server.scripts.plot_conversation_scores \
        --db /mnt/disks/data/chat_history.db
    python -m genetics_mcp_server.scripts.plot_conversation_scores \
        --metrics /mnt/disks/data/eval/analysis_output/metrics.json --window 14

Plots produced (one figure, stacked panels):
1. Per-score share over time   - one line per quality score (1..5), each the
   rolling share of conversations with that score.
2. Rolling mean score + volume - single trend line for average quality (1..5)
   with the conversation volume as faint bars for context.
3. Disposition mix             - one line per success_label bucket as a share of
   all conversations over time.
4. Issue category mix          - one line per issue taxonomy category as a share
   of all issues over time (deduped per conversation).
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless: write files, no display needed
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from . import analysis_timeseries as ats  # noqa: E402
from .analysis_timeseries import (  # noqa: E402
    DISPOSITION_LABELS,
    ISSUE_CATEGORY_NAMES,
    SCORES,
)

# colourblind-friendly red->green ramp for scores 1..5
SCORE_COLORS = {
    1: "#d73027",
    2: "#fc8d59",
    3: "#fee08b",
    4: "#91cf60",
    5: "#1a9850",
}

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

# distinct colour per issue category from a 20-colour qualitative map
ISSUE_CATEGORY_COLORS = {
    name: plt.cm.tab20(i % 20) for i, name in enumerate(ISSUE_CATEGORY_NAMES)
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_metrics(path: Path) -> list[dict]:
    records = json.loads(path.read_text())
    if not isinstance(records, list):
        raise ValueError(f"{path} is not a list of metric records")
    return records


def load_from_db(db_path: Path) -> list[dict]:
    """Read analysis records from the chat_history SQLite DB.

    Delegates to ``ChatHistoryDB.list_all_analysis_rows()`` (read-only in spirit)
    so the PNG and the frontend Quality-plots tab share one authoritative DB-read
    path and cannot drift. ``created_at`` therefore comes from the native
    ``chat_sessions.created_at`` column (always present), not the metrics_json
    blob — so conversations whose stored metrics lacked a created_at are now kept
    rather than silently dropped.

    The DB returns ``issue_categories`` per conversation; ``analysis_timeseries``
    reads ``llm_issue_categories``, so we map that one key. The other keys
    (created_at, llm_quality_score, llm_disposition, success_label) already match.
    """
    from genetics_mcp_server.db.chat_history_db import ChatHistoryDB
    from genetics_mcp_server.db.singleton import Singleton

    # clear any singleton so we open this specific db_path (same dance the
    # analyzer does), not whatever instance an earlier import created
    if ChatHistoryDB in Singleton._instances:
        del Singleton._instances[ChatHistoryDB]
    db = ChatHistoryDB(str(db_path))

    return [
        {
            "created_at": row["created_at"],
            "llm_quality_score": row["llm_quality_score"],
            "success_label": row["success_label"],
            "llm_disposition": row["llm_disposition"],
            "llm_issue_categories": row["issue_categories"],
        }
        for row in db.list_all_analysis_rows()
    ]


# ---------------------------------------------------------------------------
# Plot panels (render-only; all aggregation comes from analysis_timeseries)
# ---------------------------------------------------------------------------

def _arr(values):
    """None -> nan float array so matplotlib breaks the line at empty windows
    (the series carry JSON-friendly None; matplotlib draws gaps for nan)."""
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def panel_score_shares(ax, data, window):
    xs = [ats.parse_date(d) for d in data["dates"]]
    for s in SCORES:
        ax.plot(xs, _arr(data["series"][str(s)]), label=f"score {s}",
                color=SCORE_COLORS[s], lw=2)
    ax.set_ylabel("share of conversations (%)")
    ax.set_title(f"Quality score distribution over time ({window}-day centered window)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", ncol=5, fontsize=8, frameon=False)
    ax.grid(True, alpha=0.3)


def panel_mean_and_volume(ax, data, window):
    xs = [ats.parse_date(d) for d in data["dates"]]
    means = _arr(data["series"]["mean"])
    los, his = _arr(data["ci_low"]), _arr(data["ci_high"])
    vols = data["volume"]

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


def panel_disposition_shares(ax, data, window):
    xs = [ats.parse_date(d) for d in data["dates"]]
    for lab in DISPOSITION_LABELS:
        vals = data["series"][lab]
        # skip buckets with no data so the legend stays meaningful
        if all(v is None or v == 0 for v in vals):
            continue
        ax.plot(xs, _arr(vals), label=lab, color=DISPOSITION_COLORS[lab], lw=2)
    ax.set_ylabel("share of all conversations (%)")
    ax.set_title(f"Disposition mix over time ({window}-day centered window)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", ncol=4, fontsize=8, frameon=False)
    ax.grid(True, alpha=0.3)


def panel_issue_category_shares(ax, data, window):
    xs = [ats.parse_date(d) for d in data["dates"]]
    any_data = False
    for c in ISSUE_CATEGORY_NAMES:
        vals = data["series"][c]
        if all(v is None or v == 0 for v in vals):
            continue
        any_data = True
        ax.plot(xs, _arr(vals), label=c, color=ISSUE_CATEGORY_COLORS[c], lw=2)
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
        description="Plot evaluated conversation quality over time from the "
                    "conversation_analysis DB tables or analyze_conversations "
                    "metrics.json.",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--db",
                     help="Path to chat_history.db (reads conversation_analysis "
                          "/ conversation_issue, read-only)")
    src.add_argument("--metrics",
                     help="Path to metrics.json from analyze_conversations")
    parser.add_argument("--out", default=None,
                        help="Output image path (default: <source dir>/scores_over_time.png)")
    parser.add_argument("--window", type=int, default=7,
                        help="Sliding window width in days, centered (default: 7)")
    parser.add_argument("--min-n", type=int, default=3,
                        help="Minimum conversations in a window to plot a point "
                             "(below this the line breaks instead of spiking; default: 3)")
    args = parser.parse_args()

    if args.db:
        source_path = Path(args.db)
        if not source_path.exists():
            print(f"Error: db file not found: {source_path}", file=sys.stderr)
            sys.exit(1)
        records = load_from_db(source_path)
    else:
        source_path = Path(args.metrics)
        if not source_path.exists():
            print(f"Error: metrics file not found: {source_path}", file=sys.stderr)
            sys.exit(1)
        records = load_metrics(source_path)

    series = ats.build_all_series(records, window=args.window, min_n=args.min_n)
    meta = series["meta"]

    if meta["skipped_no_date"]:
        print(f"  Skipped {meta['skipped_no_date']} records with no/unparseable "
              "created_at (re-run analyze_conversations to populate it)",
              file=sys.stderr)

    if meta["empty"]:
        print("Error: no records with a parseable created_at — re-run "
              "analyze_conversations so the analysis includes session dates.",
              file=sys.stderr)
        sys.exit(1)

    n_scored = meta["scored"]
    print(f"  {meta['total']} dated conversations, {n_scored} with an LLM quality "
          f"score, {meta['date_min']} to {meta['date_max']}", file=sys.stderr)
    if n_scored == 0:
        print("Warning: no llm_quality_score values — score panels will be empty. "
              "Run analyze_conversations without --no-llm.", file=sys.stderr)

    fig, axes = plt.subplots(4, 1, figsize=(13, 17), sharex=True)
    panel_score_shares(axes[0], series["score_share"], args.window)
    panel_mean_and_volume(axes[1], series["mean_and_volume"], args.window)
    panel_disposition_shares(axes[2], series["disposition_mix"], args.window)
    panel_issue_category_shares(axes[3], series["issue_category_mix"], args.window)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    fig.suptitle(
        f"Genie LLM-as-judge quality over time  "
        f"(n={n_scored} scored, {args.window}-day window)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.99))

    out_path = Path(args.out) if args.out else source_path.parent / "scores_over_time.png"
    fig.savefig(out_path, dpi=130)
    print(f"  Wrote plot to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
