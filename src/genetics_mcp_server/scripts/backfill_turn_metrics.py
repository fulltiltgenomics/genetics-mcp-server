"""Backfill chat_turn_metrics from the chat-backend log sink, for turns that ran before the
table existed.

Reads the rows produced by scripts/turn_metrics_from_logs.sql — one per "Chat complete:"
log line — on stdin, as the JSON array `bq --format=json` prints or as JSON lines, and
hands them to ChatHistoryDB.backfill_turn_metrics_from_log, which owns the two
idempotency rules (a row id derived from the log entry's insertId; nothing at or after the
first live row). Re-running is therefore safe.

The rows carry user emails, so this prints counts only. Run inside the chat-backend pod,
where the database and CHAT_HISTORY_DB are:

    bq query --use_legacy_sql=false --format=json --max_rows=1000000 \\
        --parameter=cluster:STRING:finngenie < scripts/turn_metrics_from_logs.sql \\
      | kubectl -n genetics exec -i deploy/chat-backend -- \\
          python -m genetics_mcp_server.scripts.backfill_turn_metrics

The cluster parameter is what keeps a staging turn out of the production table: both
clusters' lines can land in the same BigQuery dataset.
"""

import argparse
import json
import os
import sys

from genetics_mcp_server.db.chat_history_db import ChatHistoryDB

REQUIRED = ("log_id", "created_at", "user_id", "iterations", "input_tokens", "output_tokens", "cost_usd")


def read_turns(stream) -> list[dict]:
    text = stream.read().strip()
    if not text:
        return []
    if text.startswith("["):
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    turns = []
    for row in rows:
        missing = [k for k in REQUIRED if row.get(k) in (None, "")]
        if missing:
            raise SystemExit(f"row {row.get('log_id')!r} is missing {missing}")
        turns.append({
            "log_id": row["log_id"],
            "created_at": row["created_at"],
            "user_id": row["user_id"],
            "session_id": row.get("session_id") or None,
            "model": row.get("model") or None,
            "iterations": int(row["iterations"]),
            "input_tokens": int(row["input_tokens"]),
            "output_tokens": int(row["output_tokens"]),
            "cost_usd": float(row["cost_usd"]),
        })
    return turns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--db",
        default=os.environ.get("CHAT_HISTORY_DB"),
        help="chat_history.db path (default: $CHAT_HISTORY_DB)",
    )
    args = parser.parse_args()
    if not args.db:
        raise SystemExit("--db or CHAT_HISTORY_DB is required")

    turns = read_turns(sys.stdin)
    result = ChatHistoryDB(args.db).backfill_turn_metrics_from_log(turns)
    print(json.dumps({"read": len(turns), **result}))


if __name__ == "__main__":
    main()
