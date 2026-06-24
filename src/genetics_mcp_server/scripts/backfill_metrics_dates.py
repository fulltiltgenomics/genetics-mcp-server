"""One-off hotfix: backfill the `created_at` field into an existing metrics.json.

`analyze_conversations` only started writing each session's `created_at` into
metrics.json recently. Metrics files produced before that change have no date
field, so plot_conversation_scores can't place conversations on a timeline.

This script reads session start times from the chat-history SQLite DB and joins
them into an existing metrics.json by `session_id`, in place (with a one-time
.bak backup). It is idempotent and safe to re-run.

Usage:
    python -m genetics_mcp_server.scripts.backfill_metrics_dates \
        --metrics /mnt/disks/data/eval/analysis_output/metrics.json \
        --db /mnt/disks/data/eval/chat_history.db

After running, plot with:
    python -m genetics_mcp_server.scripts.plot_conversation_scores \
        --metrics .../metrics.json
"""

import argparse
import json
import shutil
import sqlite3
import sys
from pathlib import Path


def load_session_dates(db_path: str) -> dict[str, str]:
    """Map session_id -> created_at (session start) from chat_sessions."""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT id, created_at FROM chat_sessions").fetchall()
    finally:
        conn.close()
    return {sid: (created_at or "") for sid, created_at in rows}


def main():
    parser = argparse.ArgumentParser(
        description="Backfill created_at into an existing metrics.json from the chat DB.",
    )
    parser.add_argument("--metrics", required=True, help="Path to metrics.json to patch")
    parser.add_argument("--db", required=True, help="Path to chat_history SQLite DB")
    parser.add_argument("--overwrite-existing", action="store_true",
                        help="Also overwrite created_at on records that already have it")
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        print(f"Error: metrics file not found: {metrics_path}", file=sys.stderr)
        sys.exit(1)
    if not Path(args.db).exists():
        print(f"Error: database not found: {args.db}", file=sys.stderr)
        sys.exit(1)

    records = json.loads(metrics_path.read_text())
    if not isinstance(records, list):
        print(f"Error: {metrics_path} is not a list of metric records", file=sys.stderr)
        sys.exit(1)

    dates = load_session_dates(args.db)
    print(f"  Loaded {len(dates)} session dates from {args.db}", file=sys.stderr)

    filled = already = unmatched = 0
    for r in records:
        sid = r.get("session_id")
        has_date = bool(r.get("created_at"))
        if has_date and not args.overwrite_existing:
            already += 1
            continue
        cd = dates.get(sid)
        if cd:
            r["created_at"] = cd
            filled += 1
        else:
            # keep whatever was there (possibly empty); session not in this DB
            r.setdefault("created_at", "")
            unmatched += 1

    # one-time backup, then write in place
    backup = metrics_path.with_suffix(metrics_path.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(metrics_path, backup)
        print(f"  Backed up original to {backup}", file=sys.stderr)
    else:
        print(f"  Backup {backup} already exists, not overwriting it", file=sys.stderr)

    metrics_path.write_text(json.dumps(records, indent=2, default=str))
    print(f"  {len(records)} records: {filled} dated, {already} already had a date, "
          f"{unmatched} unmatched (session not in DB)", file=sys.stderr)
    if unmatched:
        print("  Note: unmatched records have an empty created_at and will be "
              "skipped by the plotting script. Point --db at the DB that holds "
              "those sessions, or pass multiple DBs by re-running.", file=sys.stderr)
    print(f"  Wrote {metrics_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
