"""Analyze conversation history to gain insights for improving the genetics AI assistant.

Usage:
    python -m genetics_mcp_server.scripts.analyze_conversations --db /path/to/chat_history.db
    python -m genetics_mcp_server.scripts.analyze_conversations --db /path/to/db --no-llm
    python -m genetics_mcp_server.scripts.analyze_conversations --db /path/to/db --start-from 2026-03-01
    python -m genetics_mcp_server.scripts.analyze_conversations --db /path/to/db --start-from 2026-03-01 --until 2026-04-01
    python -m genetics_mcp_server.scripts.analyze_conversations --db /path/to/db --refresh-quality  # re-run quality eval, keep topic cache
    python -m genetics_mcp_server.scripts.analyze_conversations --db /path/to/db --no-cache         # recompute everything
    python -m genetics_mcp_server.scripts.analyze_conversations --db /path/to/db --force            # reanalyze every conversation from scratch (e.g. after analysis-code changes)
    python -m genetics_mcp_server.scripts.analyze_conversations --db /path/to/db --output-dir ./analysis_output

Analyzes conversations for:
- Topic categorization (LLM-based or keyword fallback)
- LLM-based quality evaluation of full conversations
- Tool usage patterns and efficiency
- SDK function calls made from inside sandboxed scripts, when --sdk-log is given
- Success/failure metrics
- Eval dataset extraction
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sqlite3
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

from genetics_mcp_server.config import model_rejects_disabled_thinking

load_dotenv()

logger = logging.getLogger("analyze_conversations")

# bumping this invalidates every cached analysis: any conversation_analysis row
# whose analyzer_version differs from this value is treated as missing and gets
# recomputed by the LLM on the next run (e.g. after changing prompts or scoring).
ANALYZER_VERSION = 1


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(db_path: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load chat_sessions and chat_messages into polars DataFrames.

    Read-only: this runs nightly against the live chat_history.db on a shared RWO PVC
    while chat-backend is serving it, and nothing here writes (genetics-results-suite-4zd).
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cursor = conn.execute("SELECT * FROM chat_sessions")
        cols = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        sessions = pl.DataFrame(
            {col: [row[i] for row in rows] for i, col in enumerate(cols)},
            schema_overrides={"rating": pl.Int64},
        )

        # rowid comes along so the "last message wins" below can tiebreak on insertion order,
        # matching ChatHistoryDB.get_messages. created_at alone has one-second resolution
        cursor = conn.execute("SELECT *, rowid AS _rowid FROM chat_messages")
        cols = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        messages = pl.DataFrame(
            {col: [row[i] for row in rows] for i, col in enumerate(cols)},
        )
        # sqlite stores booleans as integers
        if "thumbs_up" in messages.columns:
            messages = messages.with_columns(
                pl.col("thumbs_up").cast(pl.Boolean, strict=False)
            )
    finally:
        conn.close()

    return sessions, messages


def resolve_llm_config_db(db: str, explicit: str | None) -> str:
    """Where to look for llm_config.db: alongside --db unless told otherwise.

    The deployed CronJob passes only --db and relies on this default, so the derivation is load
    bearing (genetics-results-suite-uvh 8). It lived inline in main() where no test reached it.
    """
    return explicit or str(Path(db).parent / "llm_config.db")


def load_instruction_set_names(db_path: str) -> dict[str, str]:
    """Map instruction set id -> name from the llm_config database.

    The sets live in a different SQLite file from the conversations, and the analyzer is
    pointed at the conversation one. Every failure to read the config file is degraded to an
    empty map rather than raised: the report groups by id instead of name, which is worse to
    read but still correct, and a nightly report is not worth failing over a sidecar file.
    """
    if not os.path.exists(db_path):
        return {}
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            rows = conn.execute(
                "SELECT id, name FROM user_instruction_sets"
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error as e:
        logger.warning(f"could not read instruction set names from {db_path}: {e}")
        return {}
    return {row[0]: row[1] for row in rows}


# ---------------------------------------------------------------------------
# Robust JSON extraction from LLM text
# ---------------------------------------------------------------------------

def extract_first_json(text: str):
    """Parse the first complete JSON value (object or array) in text.

    Models sometimes emit the JSON followed by a second object or trailing prose
    (which makes a plain json.loads raise "Extra data"). raw_decode parses one
    value from the first opening bracket and ignores anything after it.
    """
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                obj, _ = decoder.raw_decode(text, i)
                return obj
            except json.JSONDecodeError:
                continue  # this bracket didn't start valid JSON; try the next
    return None


# thinking is off on every analysis call below (see thinking_off_kwargs), but a
# response can still lead with a non-text block, so never index content[0]:
# on thinking-capable models that block is a ThinkingBlock with no .text.
def response_text(response) -> str:
    """Concatenate the text blocks of a Messages API response."""
    return "".join(b.text for b in response.content if b.type == "text")


def thinking_off_kwargs(model: str) -> dict:
    """Request kwargs that keep an analysis call thinking-free.

    These are structured-extraction and judging calls whose whole output is a
    JSON object, so reasoning tokens buy nothing but cost and latency. Opus 5
    thinks by default, so opting out has to be explicit; Fable and Mythos reject
    the opt-out entirely, hence the model check rather than a constant.
    """
    if model_rejects_disabled_thinking(model):
        return {}
    return {"thinking": {"type": "disabled"}}


# ---------------------------------------------------------------------------
# Tool usage parsing
# ---------------------------------------------------------------------------

# matches *[Using tool: tool_name; param1: val1, ...]*  or  *[Using tool: tool_name...]*
TOOL_MARKER_RE = re.compile(r"\*\[Using tool: ([^;.\]]+)[^]]*\]\*")


def parse_tool_calls(content: str) -> list[str]:
    """Extract tool names from the display markers in an assistant message's text.

    Fallback source only — see message_tool_calls. These markers are prose injected
    during streaming for the UI, and llm_service._strip_tool_use_markers removes them
    from replayed history, so they are neither authoritative nor complete.
    """
    return TOOL_MARKER_RE.findall(content)


def parse_tool_calls_from_content_json(content_json: str | None) -> list[str] | None:
    """Tool names from the tool_use blocks of a persisted assistant message.

    Returns None when the row carries no usable block list at all — absent, empty,
    unparseable, or not a JSON list — so the caller can tell "nothing was recorded
    here" apart from "recorded, and this turn called no tools" (an empty list).
    That distinction is what makes the marker fallback targeted rather than silent.

    content_json is client-supplied, so every malformed shape degrades to None
    instead of raising: one bad row must not fail a report over thousands of them.
    """
    if not content_json:
        return None
    try:
        blocks = json.loads(content_json)
    except (TypeError, ValueError):
        return None
    if not isinstance(blocks, list):
        return None

    names = []
    for block in blocks:
        # text / thinking / tool_result blocks live in here too and are not tool calls
        if isinstance(block, dict) and block.get("type") == "tool_use":
            name = block.get("name")
            if isinstance(name, str) and name:
                names.append(name)
    return names


def message_tool_calls(row: dict) -> tuple[list[str], bool]:
    """Tool names for one assistant message, and whether the count is authoritative.

    Precedence is deliberate and explicit:
      1. content_json tool_use blocks — the real record of what the model called.
      2. the *[Using tool: X]* display markers in `content` — used ONLY for rows that
         carry no block list at all (history predating the content_json migration).

    The second element is False in the fallback case: markers are display prose and
    under-count, so anything aggregated from them is a lower bound.
    """
    from_content_json = parse_tool_calls_from_content_json(row.get("content_json"))
    if from_content_json is not None:
        return from_content_json, True
    return parse_tool_calls(row.get("content") or ""), False


@dataclass
class ToolCountCoverage:
    """How much of the tool-call counting rests on authoritative content_json.

    Every tool aggregate in the report has to be read against this: a session with
    even one assistant message lacking content_json can only be counted from display
    markers, and that count is a floor, not a total.

    "Exact" is still an upper claim on quality, not a guarantee of completeness: the
    client also persists partial content_json when a stream is interrupted (LLMChat's
    partialMsg path), so a present block list may be truncated at the point the stream
    died and such a row counts as fully covered anyway.
    """
    assistant_messages: int = 0
    messages_from_content_json: int = 0
    messages_from_markers: int = 0
    sessions: int = 0
    fully_covered_sessions: int = 0
    tool_calls_from_content_json: int = 0
    tool_calls_from_markers: int = 0

    @property
    def total_tool_calls(self) -> int:
        return self.tool_calls_from_content_json + self.tool_calls_from_markers

    @property
    def is_exact(self) -> bool:
        """True only when there are assistant messages and all came from content_json.

        An empty frame is not "exact" — there is nothing to have counted exactly.
        """
        return self.assistant_messages > 0 and self.assistant_messages == self.messages_from_content_json

    def summary_lines(self) -> list[str]:
        """Markdown lines stating the counting basis and, if partial, the caveat."""
        msg_pct = (
            self.messages_from_content_json / self.assistant_messages * 100
            if self.assistant_messages else 0.0
        )
        sess_pct = (
            self.fully_covered_sessions / self.sessions * 100 if self.sessions else 0.0
        )
        lines = [
            "### Counting coverage\n",
            f"- Counted from `content_json` `tool_use` blocks: "
            f"{self.messages_from_content_json}/{self.assistant_messages} assistant messages "
            f"({msg_pct:.1f}%), {self.tool_calls_from_content_json} tool calls",
            f"- Counted from `*[Using tool: …]*` display markers (fallback, under-counts): "
            f"{self.messages_from_markers} assistant messages, "
            f"{self.tool_calls_from_markers} tool calls",
            f"- Sessions with `content_json` on **every** assistant message: "
            f"{self.fully_covered_sessions}/{self.sessions} ({sess_pct:.1f}%)",
        ]
        if self.assistant_messages == 0:
            lines.append("\nNo assistant messages: there is nothing to count, so no "
                         "exactness claim is made.")
        elif self.is_exact:
            lines.append("\nAll tool counts below are **exact**.")
        else:
            lines.append(
                f"\n**Every tool-call aggregate in this report is a LOWER BOUND** for the "
                f"{self.sessions - self.fully_covered_sessions} sessions that are not fully "
                "covered: their marker-derived messages record only the tool uses the UI "
                "happened to annotate. Do not compare them against an exactly-counted "
                "benchmark without accounting for this."
            )
        return lines


def build_tool_coverage(messages: pl.DataFrame) -> tuple[pl.DataFrame, ToolCountCoverage]:
    """Per-session tool-count provenance plus the aggregate coverage.

    Kept separate from build_session_tool_stats because coverage has to include
    sessions whose tool count is zero — a session with assistant messages that have
    neither content_json nor markers is precisely the case a coverage number exists
    to expose, and build_session_tool_stats drops it.
    """
    coverage = ToolCountCoverage()
    per_session: dict[str, dict] = {}

    assistant_msgs = messages.filter(pl.col("role") == "assistant")
    for row in assistant_msgs.iter_rows(named=True):
        tools, authoritative = message_tool_calls(row)
        sid = row["session_id"]
        entry = per_session.setdefault(
            sid, {"session_id": sid, "assistant_messages": 0, "messages_from_content_json": 0}
        )
        entry["assistant_messages"] += 1
        coverage.assistant_messages += 1
        if authoritative:
            entry["messages_from_content_json"] += 1
            coverage.messages_from_content_json += 1
            coverage.tool_calls_from_content_json += len(tools)
        else:
            coverage.messages_from_markers += 1
            coverage.tool_calls_from_markers += len(tools)

    coverage.sessions = len(per_session)
    coverage.fully_covered_sessions = sum(
        1 for e in per_session.values()
        if e["assistant_messages"] == e["messages_from_content_json"]
    )

    rows = [
        {**e, "tool_count_is_lower_bound": e["assistant_messages"] != e["messages_from_content_json"]}
        for e in per_session.values()
    ]
    if not rows:
        frame = pl.DataFrame({
            "session_id": pl.Series([], dtype=pl.Utf8),
            "assistant_messages": pl.Series([], dtype=pl.Int64),
            "messages_from_content_json": pl.Series([], dtype=pl.Int64),
            "tool_count_is_lower_bound": pl.Series([], dtype=pl.Boolean),
        })
    else:
        frame = pl.DataFrame(rows)
    return frame, coverage


def build_session_tool_stats(messages: pl.DataFrame) -> pl.DataFrame:
    """Build per-session tool usage statistics."""
    assistant_msgs = messages.filter(pl.col("role") == "assistant")

    rows = []
    for row in assistant_msgs.iter_rows(named=True):
        tools, _authoritative = message_tool_calls(row)
        if tools:
            rows.append({
                "session_id": row["session_id"],
                "tool_calls": tools,
                "tool_count": len(tools),
            })

    if not rows:
        return pl.DataFrame({
            "session_id": pl.Series([], dtype=pl.Utf8),
            "tool_calls": pl.Series([], dtype=pl.List(pl.Utf8)),
            "tool_count": pl.Series([], dtype=pl.Int64),
            "unique_tools": pl.Series([], dtype=pl.Int64),
            "tool_sequence": pl.Series([], dtype=pl.Utf8),
            "max_tools_in_message": pl.Series([], dtype=pl.Int64),
        })

    tool_df = pl.DataFrame(rows)

    # aggregate per session; max_tools_in_message flags excessive tool use within a
    # single assistant turn, which a session-wide total can't (a long conversation
    # legitimately accrues many calls spread across many messages)
    session_tools = tool_df.group_by("session_id").agg(
        pl.col("tool_calls").list.explode(keep_nulls=False, empty_as_null=False).alias("all_tools"),
        pl.col("tool_count").sum().alias("total_tool_calls"),
        pl.col("tool_count").max().alias("max_tools_in_message"),
    ).with_columns(
        pl.col("all_tools").list.n_unique().alias("unique_tools"),
        pl.col("all_tools").list.join(" -> ").alias("tool_sequence"),
    )

    return session_tools


# ---------------------------------------------------------------------------
# SDK function-call parsing (genetics-results-suite-4h6.12)
# ---------------------------------------------------------------------------

# The counterpart of the tool-call parsing above, for data access that happens INSIDE a
# sandboxed script. Such a script is a single `run_analysis` tool call no matter how many
# queries it issues, so every per-tool statistic in this report under-describes it by
# construction. The SDK emits one audit line per function call
# (`genetics_mcp_server/sdk/client.py`); this reads them back.
#
# The source is a LOG, not the chat DB — the calls happen in another process and are never
# persisted as message content, so unlike tool calls they cannot be recovered from
# chat_history.db at all. Point --sdk-log at the collected lines (a Cloud Logging export or
# a captured container log); with no --sdk-log every SDK aggregate below is zero and is
# labelled "not supplied" rather than "none happened".
# A cancelled call ends in a bare ` cancelled` rather than ` error: CancelledError`: it was
# taken away mid-flight, which is not the same event as a read that failed. It is surfaced
# here as `sdk_error="cancelled"` — distinguishable from any exception type name, and it keeps
# the parsed shape identical for every kind of line.
SDK_CALL_RE = re.compile(
    r"\[user=(?P<user>[^\]]*)\] \[session=(?P<session>[^\]]*)\] \[execution=(?P<execution>[^\]]*)\] "
    r"Executing SDK function: (?P<function>\S+) with input: (?P<arguments>.*?) "
    r"rows: (?P<rows>\d+)(?: error: (?P<error>\S+))?(?P<cancelled> cancelled)?$"
)

# A call refused by the SDK's own argument validation never reached the executor and read
# nothing, so the SDK gives it a different marker and no `rows:` field. It is counted, because
# a burst of refusals is worth seeing, but never as a data access.
SDK_REJECT_RE = re.compile(
    r"\[user=[^\]]*\] \[session=[^\]]*\] \[execution=[^\]]*\] "
    r"Rejected SDK function: (?P<function>\S+) with input: .*? error: (?P<error>\S+)$"
)

SDK_TRUNCATED_RE = re.compile(r"SDK audit truncated after (?P<limit>\d+) records")

# emitted once per process by the SDK when it has no dedicated audit fd; see
# `SHARED_STREAM_WARNING` in sdk/client.py
SDK_SHARED_STREAM_MARKER = "NOT a tamper-evident audit trail"


def parse_sdk_calls(lines) -> list[dict]:
    """Extract SDK function calls from audit log lines.

    Lines that do not match are skipped silently: the input is a raw log stream that also
    carries every other line the service logged, and one unrecognised line must not fail a
    report. A JSON-wrapped export works too — the payload is matched anywhere in the line.
    """
    calls = []
    for line in lines:
        match = SDK_CALL_RE.search(line)
        if not match:
            continue
        calls.append({
            "session_id": match.group("session"),
            "user": match.group("user"),
            "execution_id": match.group("execution"),
            "sdk_function": match.group("function"),
            "sdk_rows": int(match.group("rows")),
            "sdk_error": match.group("error") or ("cancelled" if match.group("cancelled") else ""),
        })
    return calls


def scan_sdk_notices(lines) -> dict:
    """Facts about the LOG rather than about any one call.

    `shared_stream` is the one that changes how the whole report should be read: when the SDK
    had no dedicated audit fd it says so in the stream, because the records then share a
    stream the audited script writes to itself and any of them may be forged.
    """
    notices = {"rejected": 0, "truncated": 0, "truncated_at": 0, "shared_stream": False}
    for line in lines:
        if SDK_SHARED_STREAM_MARKER in line:
            notices["shared_stream"] = True
        truncated = SDK_TRUNCATED_RE.search(line)
        if truncated:
            notices["truncated"] += 1
            notices["truncated_at"] = int(truncated.group("limit"))
        if SDK_REJECT_RE.search(line):
            notices["rejected"] += 1
    return notices


def load_sdk_log(paths: list[str] | None) -> tuple[list[dict], dict]:
    """Read each log once, returning the calls and the log-level notices together."""
    calls: list[dict] = []
    notices = {"rejected": 0, "truncated": 0, "truncated_at": 0, "shared_stream": False}
    for path in paths or []:
        # streamed, not read into memory: an SDK audit log is one line per data access from
        # every script in the range and is expected to be large
        with open(path, errors="replace") as f:
            for line in f:
                calls.extend(parse_sdk_calls([line]))
                found = scan_sdk_notices([line])
                notices["rejected"] += found["rejected"]
                notices["truncated"] += found["truncated"]
                notices["truncated_at"] = max(notices["truncated_at"], found["truncated_at"])
                notices["shared_stream"] = notices["shared_stream"] or found["shared_stream"]
    return calls, notices


def load_sdk_calls(paths: list[str] | None) -> list[dict]:
    return load_sdk_log(paths)[0]


_EMPTY_SDK_STATS = pl.DataFrame({
    "session_id": pl.Series([], dtype=pl.Utf8),
    "total_sdk_calls": pl.Series([], dtype=pl.Int64),
    "sdk_rows": pl.Series([], dtype=pl.Int64),
    "sdk_executions": pl.Series([], dtype=pl.Int64),
    "unique_sdk_functions": pl.Series([], dtype=pl.Int64),
    "sdk_sequence": pl.Series([], dtype=pl.Utf8),
    "sdk_function_counts": pl.Series([], dtype=pl.Utf8),
})

# `sdk_sequence` lands in a metrics row and in every CSV cell, and a script can make thousands
# of SDK calls from one `run_analysis`, so the join is bounded: 10k calls would otherwise be a
# ~150 KB cell. The per-function TOTALS the report needs stay exact — they come from
# `sdk_function_counts`, which is bounded by the number of SDK functions (25) instead.
_SDK_SEQUENCE_MAX = 50


def build_session_sdk_stats(calls: list[dict]) -> pl.DataFrame:
    """Per-session SDK usage, shaped like build_session_tool_stats so it joins the same way.

    A session with `unknown` for its session id is kept rather than dropped: until
    `4h6.43`/`4h6.44` deliver the sandbox token's `sid` to the SDK, that is what every line
    carries,
    and silently discarding them would report "no script access happened".
    """
    if not calls:
        return _EMPTY_SDK_STATS
    frame = pl.DataFrame(calls)
    function_counts = (
        frame.group_by(["session_id", "sdk_function"])
        .len()
        .group_by("session_id")
        .agg(
            (pl.col("sdk_function") + ":" + pl.col("len").cast(pl.Utf8))
            .str.join("|")
            .alias("sdk_function_counts")
        )
    )
    return (
        frame.group_by("session_id")
        .agg(
            pl.len().alias("total_sdk_calls"),
            pl.col("sdk_rows").sum().alias("sdk_rows"),
            pl.col("execution_id").n_unique().alias("sdk_executions"),
            pl.col("sdk_function").n_unique().alias("unique_sdk_functions"),
            pl.col("sdk_function").head(_SDK_SEQUENCE_MAX).str.join(" -> ").alias("sdk_sequence"),
            # cast first: pl.len() is UInt32 and the subtraction underflows below the bound
            (pl.len().cast(pl.Int64) - _SDK_SEQUENCE_MAX).alias("_elided"),
        )
        .with_columns(
            pl.when(pl.col("_elided") > 0)
            .then(pl.col("sdk_sequence") + pl.format(" -> ... (+{} more)", pl.col("_elided")))
            .otherwise(pl.col("sdk_sequence"))
            .alias("sdk_sequence")
        )
        .drop("_elided")
        .join(function_counts, on="session_id", how="left")
        .cast({"total_sdk_calls": pl.Int64, "sdk_rows": pl.Int64,
               "sdk_executions": pl.Int64, "unique_sdk_functions": pl.Int64})
    )


# ---------------------------------------------------------------------------
# Keyword-based topic categorization (fallback)
# ---------------------------------------------------------------------------

TOPIC_KEYWORDS = {
    "gene_lookup": [
        r"\bgene\b", r"\bgenes\b", r"\bexpression\b", r"\bexome\b",
        r"\bburden\b", r"\bconstraint\b",
    ],
    "variant_interpretation": [
        r"\bvariant\b", r"\brs\d+", r"\b\d+:\d+:[ACGT]+:[ACGT]+\b",
        r"\bmutation\b", r"\bsnp\b", r"\bsnv\b",
    ],
    "phenotype_exploration": [
        r"\bphenotype\b", r"\bpheWAS\b", r"\bloci\b", r"\blocus\b",
        r"\bgwas\b", r"\bassociat", r"\bendpoint",
    ],
    "cross_phenotype_analysis": [
        r"\bcompare\b.*\bphenotype", r"\bshared\b.*\bsignal",
        r"\bcross.?phenotype", r"\bpleiotrop",
    ],
    "colocalization_ld": [
        r"\bcolocaliz", r"\blinkage\b", r"\b[Ll][Dd]\b",
        r"\br2\b", r"\bld\b",
    ],
    "literature_search": [
        r"\bliterature\b", r"\bpaper\b", r"\bpubmed\b", r"\barticle\b",
        r"\bpublication\b", r"\bpmid\b",
    ],
    "data_source_question": [
        r"\bdata\s*source", r"\bdataset\b", r"\bmethod\b",
        r"\bhow\s+(do|does|can)\s+you", r"\bwhat\s+(are|is)\s+your\s+source",
        r"\bavailable\b.*\bdata",
    ],
    "variant_list_analysis": [
        r"\blist\s+of\s+variant", r"\bvariant\s+list\b",
        r"(?:\d+:\d+:[ACGT]+:[ACGT]+\s*\n){2,}",
    ],
    "clinical_genetics": [
        r"\bclinical\b", r"\bmendelian\b", r"\bpatient\b",
        r"\bpathogen", r"\bdiagnos", r"\bclingen\b",
        r"\bheterozygous\b", r"\bhomozygous\b", r"\bframeshift\b",
    ],
    "bigquery_advanced": [
        r"\bbigquery\b", r"\bsql\b", r"\bquery\b.*\btable",
    ],
}


def categorize_by_keywords(text: str) -> tuple[str, float]:
    """Categorize text using keyword matching. Returns (topic, confidence)."""
    text_lower = text.lower()
    scores: dict[str, int] = {}
    for topic, patterns in TOPIC_KEYWORDS.items():
        score = sum(1 for p in patterns if re.search(p, text_lower))
        if score > 0:
            scores[topic] = score

    if not scores:
        return "general_genetics", 0.3

    best_topic = max(scores, key=scores.get)  # type: ignore[arg-type]
    confidence = min(scores[best_topic] / 3.0, 1.0)
    return best_topic, confidence


# ---------------------------------------------------------------------------
# Cost tracking (real, from API token usage)
# ---------------------------------------------------------------------------

# USD per million tokens, (input, output). Keep in sync with model defaults below.
MODEL_PRICING = {
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-4-6": (3.0, 15.0),
}
# fall back to Opus-tier pricing for unknown models so cost isn't silently understated
DEFAULT_PRICING = (5.0, 25.0)


@dataclass
class CostTracker:
    """Accumulates real token usage from API responses and computes USD cost."""

    # model -> {"input": tokens, "output": tokens, "calls": n}
    usage: dict[str, dict[str, int]] = None

    def __post_init__(self):
        if self.usage is None:
            self.usage = {}

    @staticmethod
    def _as_int(value) -> int:
        # only count genuine integer token fields; ignore None and test mocks
        return value if isinstance(value, int) and not isinstance(value, bool) else 0

    def add(self, model: str, response_usage) -> None:
        inp = self._as_int(getattr(response_usage, "input_tokens", 0))
        out = self._as_int(getattr(response_usage, "output_tokens", 0))
        # cached reads/writes still cost (reduced); count them as input for a conservative estimate
        inp += self._as_int(getattr(response_usage, "cache_read_input_tokens", 0))
        inp += self._as_int(getattr(response_usage, "cache_creation_input_tokens", 0))
        if inp == 0 and out == 0:
            return
        rec = self.usage.setdefault(model, {"input": 0, "output": 0, "calls": 0})
        rec["input"] += inp
        rec["output"] += out
        rec["calls"] += 1

    def model_cost(self, model: str) -> float:
        rec = self.usage[model]
        in_price, out_price = MODEL_PRICING.get(model, DEFAULT_PRICING)
        return rec["input"] / 1e6 * in_price + rec["output"] / 1e6 * out_price

    def total_cost(self) -> float:
        return sum(self.model_cost(m) for m in self.usage)

    def summary_lines(self) -> list[str]:
        lines = []
        for model, rec in self.usage.items():
            priced = "" if model in MODEL_PRICING else " (default pricing)"
            lines.append(
                f"- {model}{priced}: {rec['calls']} calls, "
                f"{rec['input']:,} in + {rec['output']:,} out tokens "
                f"= ${self.model_cost(model):.4f}"
            )
        lines.append(f"- **Total cost**: ${self.total_cost():.4f}")
        return lines


# ---------------------------------------------------------------------------
# LLM-based topic categorization
# ---------------------------------------------------------------------------

async def categorize_with_llm(
    session_first_messages: list[dict[str, str]],
    model: str = "claude-opus-5",
    cost_tracker: "CostTracker | None" = None,
) -> dict[str, dict]:
    """Categorize conversations using Anthropic API.

    Args:
        session_first_messages: list of {"id": session_id, "text": first_user_message}
        model: Anthropic model to use

    Returns:
        dict mapping session_id to {"topic": ..., "complexity": ..., "brief_reason": ...}
    """
    import anthropic

    from .conversation_prompts import TOPIC_CLASSIFICATION_PROMPT

    client = anthropic.AsyncAnthropic()
    results = {}
    batch_size = 20

    for i in range(0, len(session_first_messages), batch_size):
        batch = session_first_messages[i:i + batch_size]
        messages_text = "\n\n".join(
            f"[ID: {m['id']}]\n{m['text'][:500]}" for m in batch
        )
        prompt = TOPIC_CLASSIFICATION_PROMPT.format(messages=messages_text)

        try:
            response = await client.messages.create(
                model=model,
                max_tokens=2000,
                **thinking_off_kwargs(model),
                messages=[{"role": "user", "content": prompt}],
            )
            if cost_tracker is not None:
                cost_tracker.add(model, response.usage)
            text = response_text(response)
            # extract JSON from response (may have markdown fences / trailing data)
            classifications = extract_first_json(text)
            if isinstance(classifications, list):
                for c in classifications:
                    results[c["id"]] = {
                        "topic": c["topic"],
                        "complexity": c.get("complexity", 2),
                        "brief_reason": c.get("brief_reason", ""),
                    }
        except Exception as e:
            logger.error(f"LLM categorization failed for batch {i // batch_size + 1}: {e}")
            # fall back to keyword for this batch
            for m in batch:
                topic, _ = categorize_by_keywords(m["text"])
                results[m["id"]] = {
                    "topic": topic,
                    "complexity": 2,
                    "brief_reason": "keyword fallback",
                }

        if i + batch_size < len(session_first_messages):
            logger.info(f"  Categorized {min(i + batch_size, len(session_first_messages))}"
                        f"/{len(session_first_messages)} sessions...")

    return results


# ---------------------------------------------------------------------------
# LLM-based issue categorization
# ---------------------------------------------------------------------------

async def categorize_issues_with_llm(
    issues: list[str],
    model: str = "claude-opus-5",
    cost_tracker: "CostTracker | None" = None,
) -> dict[str, str]:
    """Map detailed free-text quality issues onto a fixed taxonomy.

    The judge emits one detailed issue string per problem, so raw issues almost
    never recur verbatim. This groups them into recurring categories so the
    report can surface the real underlying problems.

    Args:
        issues: distinct issue strings to categorize.

    Returns:
        dict mapping each issue string to a category name (always "other" at
        worst, so every input gets a category even if the LLM call fails).
    """
    import anthropic

    from .conversation_prompts import ISSUE_CATEGORIES, ISSUE_CATEGORIZATION_PROMPT

    valid = {c for c, _ in ISSUE_CATEGORIES}
    categories_text = "\n".join(f"- {c}: {d}" for c, d in ISSUE_CATEGORIES)
    client = anthropic.AsyncAnthropic()
    results: dict[str, str] = {}
    batch_size = 40

    for i in range(0, len(issues), batch_size):
        batch = issues[i:i + batch_size]
        issues_text = "\n".join(f"[{j}] {issue[:300]}" for j, issue in enumerate(batch))
        prompt = ISSUE_CATEGORIZATION_PROMPT.format(
            categories=categories_text, issues=issues_text,
        )

        try:
            response = await client.messages.create(
                model=model,
                max_tokens=2000,
                **thinking_off_kwargs(model),
                messages=[{"role": "user", "content": prompt}],
            )
            if cost_tracker is not None:
                cost_tracker.add(model, response.usage)
            parsed = extract_first_json(response_text(response))
            if isinstance(parsed, list):
                for obj in parsed:
                    idx = obj.get("id")
                    cat = obj.get("category", "other")
                    if isinstance(idx, int) and 0 <= idx < len(batch):
                        results[batch[idx]] = cat if cat in valid else "other"
        except Exception as e:
            logger.error(f"Issue categorization failed for batch {i // batch_size + 1}: {e}")

        # anything the LLM didn't assign (or a failed batch) falls back to "other"
        for issue in batch:
            results.setdefault(issue, "other")

    return results


# ---------------------------------------------------------------------------
# LLM-based quality evaluation
# ---------------------------------------------------------------------------

def _elide_message(content: str, per_msg_max: int = 8000) -> str:
    """Elide the middle of an over-long message, keeping head and tail.

    The genetics assistant's answers are structured as: intro -> large inline
    tool-output table/data-dump -> interpretation & conclusion. Naively keeping
    only the head discards the conclusion, which is exactly what the quality
    evaluator needs to see. Keeping head + tail preserves both the question
    framing and the closing synthesis while eliding the bulky table middle.
    """
    if len(content) <= per_msg_max:
        return content
    head = per_msg_max // 2
    tail = per_msg_max - head
    elided = len(content) - per_msg_max
    return f"{content[:head]}\n[... {elided} chars elided ...]\n{content[-tail:]}"


def _attachment_note(content_json: str | None) -> str:
    """Describe any files attached to a message, from its content_json.

    Attachments (uploaded TSVs, images, etc.) are recorded only in the message's
    content_json, not in the plain `content` the judge would otherwise see. Without
    this note the evaluator can't tell a file was provided and wrongly flags the
    assistant's analysis of it as fabricated.
    """
    if not content_json:
        return ""
    try:
        data = json.loads(content_json)
    except (json.JSONDecodeError, TypeError):
        return ""
    attachments = data.get("attachments") if isinstance(data, dict) else None
    if not attachments:
        return ""
    descs = []
    for a in attachments:
        name = a.get("name", "file")
        typ = a.get("type", "")
        size = a.get("size")
        size_str = f", {size} bytes" if isinstance(size, int) else ""
        descs.append(f"{name} ({typ}{size_str})")
    return "[User attached file(s): " + "; ".join(descs) + "]"


def _format_conversation_for_eval(
    session_id: str, messages: pl.DataFrame, max_chars: int = 120000,
) -> str:
    """Format a conversation for LLM quality evaluation, eliding if needed.

    Modern Claude context windows are large, so the budget is generous; the
    goal is to never hide the assistant's final answer behind tool-output
    tables. Over-long individual messages are middle-elided (head + tail) and
    the total budget is large enough that whole later turns are rarely dropped.
    """
    session_msgs = messages.filter(
        pl.col("session_id") == session_id
    ).sort("created_at")

    parts = []
    total_len = 0
    for row in session_msgs.iter_rows(named=True):
        role = row["role"].upper()
        content = _elide_message(row["content"])
        note = _attachment_note(row.get("content_json"))
        body = f"{note}\n{content}" if note else content
        part = f"[{role}]\n{body}\n"
        total_len += len(part)
        if total_len > max_chars:
            parts.append("[... conversation truncated for length ...]")
            break
        parts.append(part)

    return "\n".join(parts)


async def evaluate_quality_with_llm(
    session_ids: list[str],
    messages: pl.DataFrame,
    model: str = "claude-opus-5",
    cost_tracker: "CostTracker | None" = None,
) -> dict[str, dict]:
    """Evaluate conversation quality using Anthropic API.

    Sends full conversations (middle-elided to ~120K chars) one at a time.

    Returns:
        dict mapping session_id to quality assessment dict
    """
    import anthropic

    from .conversation_prompts import QUALITY_ASSESSMENT_PROMPT

    # give the judge today's date so it stops treating real recent dates as "future"
    today = date.today().isoformat()
    client = anthropic.AsyncAnthropic()
    results = {}
    total = len(session_ids)

    for idx, sid in enumerate(session_ids):
        conversation_text = _format_conversation_for_eval(sid, messages)

        # skip sessions with no real content
        if len(conversation_text.strip()) < 20:
            continue

        prompt = QUALITY_ASSESSMENT_PROMPT.format(
            conversation=conversation_text, today=today,
        )

        try:
            response = await client.messages.create(
                model=model,
                max_tokens=1000,
                **thinking_off_kwargs(model),
                messages=[{"role": "user", "content": prompt}],
            )
            if cost_tracker is not None:
                cost_tracker.add(model, response.usage)
            text = response_text(response)
            assessment = extract_first_json(text)
            if isinstance(assessment, dict):
                results[sid] = assessment
        except Exception as e:
            logger.error(f"Quality eval failed for {sid}: {e}")

        if (idx + 1) % 20 == 0 or idx + 1 == total:
            logger.info(f"  Evaluated {idx + 1}/{total} conversations...")

    return results


# ---------------------------------------------------------------------------
# Success/failure metrics
# ---------------------------------------------------------------------------

@dataclass
class ConversationMetrics:
    session_id: str
    created_at: str = ""  # session start timestamp, for time-series plotting
    user_rating: int | None = None
    thumbs_up_count: int = 0
    thumbs_down_count: int = 0
    total_messages: int = 0
    user_messages: int = 0
    assistant_messages: int = 0
    total_tool_calls: int = 0
    # true when at least one assistant message in the session had no content_json, so its
    # tool calls could only be counted from display markers: total_tool_calls is a floor
    tool_count_is_lower_bound: bool = False
    max_tools_in_message: int = 0
    unique_tools: int = 0
    # data access from inside sandboxed scripts (genetics-results-suite-4h6.12). These come
    # from the SDK audit log, not from the chat DB, so they are zero unless --sdk-log was
    # given — never read a zero here as "the session ran no scripts"
    total_sdk_calls: int = 0
    sdk_rows: int = 0
    unique_sdk_functions: int = 0
    # bounded to the first _SDK_SEQUENCE_MAX calls, with the elided count appended: a script
    # can make thousands of SDK calls and the untrimmed join is a ~150 KB cell. Per-function
    # totals come from `sdk_function_counts` ("fn:n|fn:n"), which stays exact.
    sdk_sequence: str = ""
    sdk_function_counts: str = ""
    has_error_response: bool = False
    reached_conclusion: bool = True
    success_score: float = 0.0
    success_label: str = "unknown"
    topic: str = "general_genetics"
    complexity: int = 2
    topic_reason: str = ""
    tool_sequence: str = ""
    first_user_message: str = ""
    tool_profile: str = ""
    # the user-authored instruction set in force, if any. Kept alongside the id because the
    # name is what a reader recognises and the id is what survives a rename
    instruction_set_id: str = ""
    instruction_set_name: str = ""
    # LLM quality assessment fields
    llm_quality_score: int | None = None
    llm_answered: str = ""
    llm_accurate: str = ""
    llm_efficient: str = ""
    llm_concluded: str = ""
    llm_disposition: str = ""
    llm_issues: list[str] | None = None
    llm_issue_categories: list[str] | None = None


def compute_success_score(m: ConversationMetrics) -> float:
    """Compute a 0-1 success score from available signals.

    Priority: user_rating > LLM quality score > heuristics.
    """
    # direct rating is strongest signal
    if m.user_rating is not None:
        return round((m.user_rating - 1) / 4.0, 3)

    # LLM quality score is next best (1-5 scale like user rating)
    if m.llm_quality_score is not None:
        try:
            q = int(m.llm_quality_score)
        except (ValueError, TypeError):
            q = 3
        score = (q - 1) / 4.0

        # the 1-5 quality_score is the primary signal, but fold in the binary
        # verdicts so a clearly-answered conversation isn't dragged down (and a
        # clearly-unanswered one isn't propped up) by a borderline numeric score
        if m.llm_answered == "yes":
            score += 0.05
        elif m.llm_answered == "no":
            score -= 0.1
        if m.llm_concluded == "no":
            score -= 0.05

        return round(max(0.0, min(1.0, score)), 3)

    # heuristic fallback
    score = 0.5

    # thumbs signals
    total_thumbs = m.thumbs_up_count + m.thumbs_down_count
    if total_thumbs > 0:
        score += 0.2 * (m.thumbs_up_count - m.thumbs_down_count) / total_thumbs

    # tool efficiency: penalize excessive tool calls relative to message count
    if m.user_messages > 0 and m.total_tool_calls > 0:
        tools_per_msg = m.total_tool_calls / m.user_messages
        if tools_per_msg > 10:
            score -= 0.15
        elif tools_per_msg > 6:
            score -= 0.05

    # error penalty
    if m.has_error_response:
        score -= 0.15

    # very short conversations (1 user msg, no tools) may be abandoned
    if m.user_messages == 1 and m.total_tool_calls == 0 and m.assistant_messages <= 1:
        score -= 0.1

    # multi-turn engagement is a positive signal
    if m.user_messages >= 3:
        score += 0.1

    return round(max(0.0, min(1.0, score)), 3)


# labels that reflect the agent's own quality (the only ones counted in the
# quality average/trend). dispositions outside the agent's control get their own
# labels so they bucket separately and are excluded from the quality metric.
QUALITY_RELEVANT_LABELS = {"successful", "neutral", "unsuccessful"}

# dispositions that are NOT the agent failing to answer an answerable question.
# each maps to its own report bucket; none counts toward the agent-quality average.
NON_QUALITY_DISPOSITIONS = {
    "technical_failure",
    "out_of_scope",
    "unfinished",
    "weird_or_unclear",
}


def label_success(score: float) -> str:
    if score >= 0.7:
        return "successful"
    elif score >= 0.4:
        return "neutral"
    else:
        return "unsuccessful"


def label_from_disposition(disposition: str, score: float) -> str:
    """Derive the report bucket from the judge's disposition.

    good_answer/agent_failure (and the no-LLM fallback) collapse to the usual
    score-based successful/neutral/unsuccessful. The other dispositions keep
    their own label so they bucket separately and stay out of the quality metric.
    """
    if disposition in NON_QUALITY_DISPOSITIONS:
        return disposition
    return label_success(score)


def compute_all_metrics(
    sessions: pl.DataFrame,
    messages: pl.DataFrame,
    tool_stats: pl.DataFrame,
    topics: dict[str, dict],
    instruction_set_names: dict[str, str] | None = None,
    sdk_stats: pl.DataFrame | None = None,
) -> list[ConversationMetrics]:
    """Compute metrics for all sessions."""
    # pre-compute per-session message stats
    msg_stats = messages.group_by("session_id").agg(
        pl.len().alias("total_messages"),
        pl.col("role").filter(pl.col("role") == "user").count().alias("user_messages"),
        pl.col("role").filter(pl.col("role") == "assistant").count().alias("assistant_messages"),
        pl.col("thumbs_up").filter(pl.col("thumbs_up") == True).count().alias("thumbs_up_count"),  # noqa: E712
        pl.col("thumbs_up").filter(pl.col("thumbs_up") == False).count().alias("thumbs_down_count"),  # noqa: E712
    )

    # get first user message per session
    first_msgs = (
        messages.filter(pl.col("role") == "user")
        .sort("created_at")
        .group_by("session_id")
        .first()
        .select("session_id", pl.col("content").alias("first_user_message"))
    )

    # get tool_profile from first user message
    tool_profiles = (
        messages.filter(pl.col("role") == "user")
        .filter(pl.col("tool_profile").is_not_null())
        .sort("created_at")
        .group_by("session_id")
        .first()
        .select("session_id", "tool_profile")
    )

    # the last message that named an instruction set decides the session's: the selector can move
    # mid-conversation, so the newest one is what the session ended up running under.
    # Must agree with routers/admin.py's resolve_instruction_set_name, which answers the same
    # question for the admin list — so: every role, not just user (an assistant row carries the
    # same value as the turn that produced it), and tiebroken on rowid, because created_at has
    # one-second resolution and two sets recorded in the same second would otherwise be free to
    # disagree between the admin list and this report (genetics-results-suite-uvh 9).
    # The column is guarded because chat_messages predating the migration does not have it
    if "instruction_set_id" in messages.columns:
        instruction_sets = (
            messages.filter(pl.col("instruction_set_id").is_not_null())
            .sort(["created_at", "_rowid"])
            .group_by("session_id")
            .last()
            .select("session_id", "instruction_set_id")
        )
    else:
        instruction_sets = pl.DataFrame({
            "session_id": pl.Series([], dtype=pl.Utf8),
            "instruction_set_id": pl.Series([], dtype=pl.Utf8),
        })

    tool_coverage, _ = build_tool_coverage(messages)

    # check for error patterns in assistant messages
    error_sessions = set()
    for row in messages.filter(pl.col("role") == "assistant").iter_rows(named=True):
        content_lower = row["content"].lower()
        if ("error" in content_lower and "tool" in content_lower) or \
           "i apologize" in content_lower and "unable" in content_lower:
            error_sessions.add(row["session_id"])

    # join everything
    combined = sessions.join(msg_stats, left_on="id", right_on="session_id", how="left")
    combined = combined.join(tool_stats, left_on="id", right_on="session_id", how="left")
    combined = combined.join(first_msgs, left_on="id", right_on="session_id", how="left")
    combined = combined.join(tool_profiles, left_on="id", right_on="session_id", how="left")
    combined = combined.join(instruction_sets, left_on="id", right_on="session_id", how="left")
    combined = combined.join(
        tool_coverage.select("session_id", "tool_count_is_lower_bound"),
        left_on="id", right_on="session_id", how="left",
    )
    combined = combined.join(
        sdk_stats if sdk_stats is not None else _EMPTY_SDK_STATS,
        left_on="id", right_on="session_id", how="left",
    )

    names = instruction_set_names or {}
    results = []
    for row in combined.iter_rows(named=True):
        sid = row["id"]
        topic_info = topics.get(sid, {})

        m = ConversationMetrics(
            session_id=sid,
            created_at=str(row.get("created_at") or ""),
            user_rating=row.get("rating"),
            thumbs_up_count=row.get("thumbs_up_count") or 0,
            thumbs_down_count=row.get("thumbs_down_count") or 0,
            total_messages=row.get("total_messages") or 0,
            user_messages=row.get("user_messages") or 0,
            assistant_messages=row.get("assistant_messages") or 0,
            total_tool_calls=row.get("total_tool_calls") or 0,
            tool_count_is_lower_bound=bool(row.get("tool_count_is_lower_bound")),
            max_tools_in_message=row.get("max_tools_in_message") or 0,
            unique_tools=row.get("unique_tools") or 0,
            total_sdk_calls=row.get("total_sdk_calls") or 0,
            sdk_rows=row.get("sdk_rows") or 0,
            unique_sdk_functions=row.get("unique_sdk_functions") or 0,
            sdk_sequence=row.get("sdk_sequence") or "",
            sdk_function_counts=row.get("sdk_function_counts") or "",
            has_error_response=sid in error_sessions,
            topic=topic_info.get("topic", "general_genetics"),
            complexity=topic_info.get("complexity", 2),
            topic_reason=topic_info.get("brief_reason", ""),
            tool_sequence=row.get("tool_sequence") or "",
            first_user_message=row.get("first_user_message") or "",
            tool_profile=row.get("tool_profile") or "",
            instruction_set_id=row.get("instruction_set_id") or "",
            # an unresolvable id still groups, under the id itself: the sets live in a separate
            # SQLite file that this analyzer may not have been pointed at
            instruction_set_name=(
                names.get(row["instruction_set_id"], row["instruction_set_id"])
                if row.get("instruction_set_id")
                else ""
            ),
        )
        m.success_score = compute_success_score(m)
        m.success_label = label_success(m.success_score)
        results.append(m)

    return results


# ---------------------------------------------------------------------------
# DB-backed cache (conversation_analysis / conversation_issue tables)
# ---------------------------------------------------------------------------

def _is_cached(row: dict | None) -> bool:
    """A session counts as cached only if it has a current-version analysis row."""
    return bool(row) and row.get("analyzer_version") == ANALYZER_VERSION


def cached_topic_and_quality(
    analysis_map: dict[str, dict],
) -> tuple[dict[str, dict], dict[str, dict], dict[str, list[str]]]:
    """Reconstruct topic / quality / issue-category inputs from stored analysis rows.

    Each conversation_analysis row stores the full ConversationMetrics dict in
    metrics_json, so previously-analyzed sessions can be replayed through the
    same pipeline (compute_all_metrics + apply_quality_assessments) without any
    LLM calls. Only rows at the current ANALYZER_VERSION are treated as cached.
    """
    topics: dict[str, dict] = {}
    quality: dict[str, dict] = {}
    issue_cats: dict[str, list[str]] = {}

    for sid, row in analysis_map.items():
        if not _is_cached(row):
            continue
        try:
            m = json.loads(row.get("metrics_json") or "{}")
        except (json.JSONDecodeError, TypeError):
            m = {}

        topics[sid] = {
            "topic": m.get("topic", "general_genetics"),
            "complexity": m.get("complexity", 2),
            "brief_reason": m.get("topic_reason", ""),
        }
        # only replay a quality assessment if the cached run actually judged it;
        # an unjudged session has no llm_quality_score and must not get a fake one
        if m.get("llm_quality_score") is not None:
            quality[sid] = {
                "quality_score": m.get("llm_quality_score"),
                "answered": m.get("llm_answered", ""),
                "accurate": m.get("llm_accurate", ""),
                "efficient": m.get("llm_efficient", ""),
                "concluded": m.get("llm_concluded", ""),
                "disposition": m.get("llm_disposition", ""),
                "issues": m.get("llm_issues") or [],
            }
        if m.get("llm_issue_categories"):
            issue_cats[sid] = list(m["llm_issue_categories"])

    return topics, quality, issue_cats


def apply_quality_assessments(
    metrics: list[ConversationMetrics],
    assessments: dict[str, dict],
):
    """Apply LLM quality assessments to metrics and recompute scores."""
    for m in metrics:
        if m.session_id in assessments:
            qa = assessments[m.session_id]
            try:
                m.llm_quality_score = int(qa.get("quality_score", 0))
            except (ValueError, TypeError):
                m.llm_quality_score = 3
            m.llm_answered = qa.get("answered", "")
            m.llm_accurate = qa.get("accurate", "")
            m.llm_efficient = qa.get("efficient", "")
            m.llm_concluded = qa.get("concluded", "")
            m.llm_disposition = qa.get("disposition", "")
            m.llm_issues = qa.get("issues")
            # recompute score now that LLM quality is available, then let the
            # disposition decide the bucket (out_of_scope/unfinished/weird/technical
            # don't fold into the successful/neutral/unsuccessful quality labels)
            m.success_score = compute_success_score(m)
            m.success_label = label_from_disposition(m.llm_disposition, m.success_score)


def mark_unscored_unknown(metrics: list[ConversationMetrics]) -> int:
    """Label conversations with no quality judgment as 'unknown'.

    A conversation the judge skipped (no llm_quality_score) and that also has no
    user rating carries no quality signal at all — leaving it on a heuristic
    successful/neutral/unsuccessful label would silently fold it into the quality
    metric. Labelling it 'unknown' keeps it honestly separate.

    Only call this when LLM evaluation ran; in --no-llm mode the heuristic labels
    are the intended output. Returns the number of conversations relabelled.
    """
    n = 0
    for m in metrics:
        if m.llm_quality_score is None and m.user_rating is None:
            m.success_label = "unknown"
            n += 1
    return n


# ---------------------------------------------------------------------------
# Eval dataset export
# ---------------------------------------------------------------------------

# the per-turn options a replay has to restore to reproduce the original conditions.
# All four are nullable per message and a null is a real value, not a gap: the browser
# sends all four on every saveMessage, and `tool_profile IS NULL` specifically means
# "all tools" (build_anthropic_tools applies no category filter when it is None) — see
# _session_user_turns
TURN_OPTION_COLUMNS = ("verbosity", "tool_profile", "instruction_set_id", "literature_backend")


def message_sort_keys(messages: pl.DataFrame) -> list[str]:
    """Total order over messages: created_at alone has one-second resolution.

    Ties within a second are real (a user turn and its assistant reply routinely share
    one), so rowid breaks them, matching ChatHistoryDB.get_messages and routers/admin.py.
    The column is guarded because a caller may hand over a frame built without it.
    """
    return ["created_at", "_rowid"] if "_rowid" in messages.columns else ["created_at"]


def _session_user_turns(session_msgs: pl.DataFrame) -> list[dict]:
    """Every user turn of one session, in order, with the options in force at that turn.

    session_msgs must already be sorted by message_sort_keys.

    A row's own values ARE the options in force at that row, nulls included. The client
    sends all four keys on every saveMessage and the write path stores exactly what the
    request carried, so a null is the recorded choice — notably `tool_profile IS NULL`
    means "all tools". Carrying the last non-null value forward would export a user who
    switched back to all-tools as still running the previous profile.

    The one exception is a row on which ALL four columns are null. A row written by the
    current client always carries at least verbosity and literature_backend, so all-null
    means the row predates the client wiring; there the previous turn's settings are the
    best available guess and do carry forward.
    """
    in_force: dict[str, str | None] = dict.fromkeys(TURN_OPTION_COLUMNS)
    turns: list[dict] = []

    for row in session_msgs.iter_rows(named=True):
        row_options = {col: row.get(col) for col in TURN_OPTION_COLUMNS}
        if any(value is not None for value in row_options.values()):
            in_force = row_options
        if row["role"] != "user":
            continue
        turns.append({
            "index": len(turns),
            "message_id": row.get("id"),
            "created_at": str(row.get("created_at") or ""),
            # untruncated: a replay has to send exactly what the user sent
            "content": row["content"],
            "options": dict(in_force),
        })

    return turns


def export_eval_dataset(
    metrics: list[ConversationMetrics],
    messages: pl.DataFrame,
    output_dir: Path,
    max_per_topic: int = 5,
):
    """Export representative conversations as eval test cases.

    Each case carries the full ordered user-turn sequence (`user_turns`) so a replay
    harness can drive the whole conversation, not just its opening message.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # group by topic, select diverse conversations
    by_topic: dict[str, list[ConversationMetrics]] = {}
    for m in metrics:
        by_topic.setdefault(m.topic, []).append(m)

    eval_cases = []
    for topic, convs in sorted(by_topic.items()):
        # sort by success score to get a mix of good and bad
        convs.sort(key=lambda c: c.success_score, reverse=True)
        # take top, bottom, and middle
        selected = []
        if len(convs) >= max_per_topic:
            n = max_per_topic
            selected = convs[:n // 2] + convs[-(n - n // 2):]
        else:
            selected = convs

        for conv in selected:
            session_msgs = messages.filter(
                pl.col("session_id") == conv.session_id
            ).sort(message_sort_keys(messages))

            turns = []
            for msg_row in session_msgs.iter_rows(named=True):
                turns.append({
                    "role": msg_row["role"],
                    "content": msg_row["content"][:2000],
                })

            user_turns = _session_user_turns(session_msgs)

            tools_used: list[str] = []
            for msg_row in session_msgs.filter(pl.col("role") == "assistant").iter_rows(named=True):
                tools, _authoritative = message_tool_calls(msg_row)
                tools_used.extend(tools)

            eval_cases.append({
                "session_id": conv.session_id,
                "topic": conv.topic,
                "complexity": conv.complexity,
                "success_score": conv.success_score,
                "success_label": conv.success_label,
                "user_rating": conv.user_rating,
                "first_user_message": conv.first_user_message[:500],
                "tools_used": tools_used,
                "total_tool_calls": conv.total_tool_calls,
                "tool_count_is_lower_bound": conv.tool_count_is_lower_bound,
                "turn_count": len(turns),
                "turns": turns,
                "user_turn_count": len(user_turns),
                "user_turns": user_turns,
            })

    # write JSON eval file
    eval_path = output_dir / "eval_dataset.json"
    with open(eval_path, "w") as f:
        json.dump(eval_cases, f, indent=2, default=str)
    logger.info(f"  Wrote {len(eval_cases)} eval cases to {eval_path}")

    # write individual transcripts for interesting cases
    transcripts_dir = output_dir / "transcripts"
    transcripts_dir.mkdir(exist_ok=True)
    for case in eval_cases:
        transcript_path = transcripts_dir / f"{case['topic']}_{case['session_id'][:8]}.md"
        with open(transcript_path, "w") as f:
            f.write(f"# {case['topic']} | score={case['success_score']} | "
                    f"tools={case['total_tool_calls']}\n\n")
            for turn in case["turns"]:
                role_label = "**User**" if turn["role"] == "user" else "**Assistant**"
                f.write(f"### {role_label}\n\n{turn['content']}\n\n---\n\n")

    logger.info(f"  Wrote {len(eval_cases)} transcripts to {transcripts_dir}")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    metrics: list[ConversationMetrics],
    sessions: pl.DataFrame,
    messages: pl.DataFrame,
    tool_stats: pl.DataFrame,
    cost_tracker: "CostTracker | None" = None,
    issue_categories: dict[str, str] | None = None,
    sdk_stats: pl.DataFrame | None = None,
    sdk_notices: dict | None = None,
) -> str:
    """Generate a markdown analysis report."""
    lines = ["# Conversation Analysis Report\n"]
    _, tool_coverage = build_tool_coverage(messages)
    lower_bound_sessions = sum(1 for m in metrics if m.tool_count_is_lower_bound)

    def tool_count_caveat() -> str:
        """One-line provenance stamp to hang under any aggregate over tool counts."""
        if not lower_bound_sessions:
            return ("_Tool counts are exact (every assistant message has `content_json`)._")
        return (
            f"_Tool counts are a **lower bound**: {lower_bound_sessions}/{len(metrics)} sessions "
            "have assistant messages with no `content_json` and fall back to display markers "
            "(see Tool Usage Patterns → Counting coverage)._"
        )

    def tool_total(m: ConversationMetrics) -> str:
        """A session's tool count, never quoted as a total when it is only a floor."""
        return f">={m.total_tool_calls}" if m.tool_count_is_lower_bound else str(m.total_tool_calls)

    # --- overview ---
    lines.append("## Overview\n")
    lines.append(f"- **Total sessions**: {len(metrics)}")
    lines.append(f"- **Total messages**: {messages.height}")
    unique_users = sessions.select("user_id").n_unique()
    lines.append(f"- **Unique users**: {unique_users}")
    date_range = sessions.select(
        pl.col("created_at").min().alias("min"),
        pl.col("created_at").max().alias("max"),
    ).row(0)
    lines.append(f"- **Date range**: {date_range[0]} to {date_range[1]}")

    rated = sum(1 for m in metrics if m.user_rating is not None)
    lines.append(f"- **Rated sessions**: {rated}/{len(metrics)}")
    if rated:
        avg_rating = sum(m.user_rating for m in metrics if m.user_rating is not None) / rated
        lines.append(f"- **Average rating**: {avg_rating:.1f}/5")
    lines.append("")

    # --- success breakdown ---
    # the quality metric reflects only conversations the agent could have done well
    # at (good_answer/agent_failure). technical failures and out-of-scope / unfinished
    # / weird requests bucket separately and are excluded from the quality average.
    lines.append("## Success Breakdown\n")
    quality_metrics = [m for m in metrics if m.success_label in QUALITY_RELEVANT_LABELS]
    success_counts = Counter(m.success_label for m in metrics)

    lines.append(f"**Agent quality** (of {len(quality_metrics)} answerable conversations; "
                 "excludes technical failures and out-of-scope / unfinished / weird):\n")
    denom = len(quality_metrics) or 1
    for label in ["successful", "neutral", "unsuccessful"]:
        count = success_counts.get(label, 0)
        pct = count / denom * 100
        lines.append(f"- **{label}**: {count} ({pct:.1f}%)")
    if quality_metrics:
        avg_score = sum(m.success_score for m in quality_metrics) / len(quality_metrics)
        lines.append(f"\nAverage agent-quality score: {avg_score:.3f}")
    lines.append("")

    # disposition buckets that are NOT counted in the quality metric above
    lines.append("**Other outcomes** (not counted in agent quality):\n")
    other_labels = [
        ("technical_failure", "Technical failures (low score, infra not agent)"),
        ("out_of_scope", "Out of scope (we can't provide it)"),
        ("unfinished", "Unfinished (user stopped)"),
        ("weird_or_unclear", "Weird / unclear question"),
        ("unknown", "Unknown (no quality judgment — judge skipped, unrated)"),
    ]
    for label, desc in other_labels:
        count = success_counts.get(label, 0)
        if count:
            pct = count / len(metrics) * 100
            lines.append(f"- **{desc}**: {count} ({pct:.1f}%)")
    lines.append("")

    # --- topic distribution ---
    lines.append("## Topic Distribution\n")
    topic_counts = Counter(m.topic for m in metrics)
    lines.append("| Topic | Count | % | Avg Score | Avg Tools |")
    lines.append("|-------|------:|--:|----------:|----------:|")
    for topic, count in topic_counts.most_common():
        topic_metrics = [m for m in metrics if m.topic == topic]
        avg_s = sum(m.success_score for m in topic_metrics) / len(topic_metrics)
        avg_t = sum(m.total_tool_calls for m in topic_metrics) / len(topic_metrics)
        pct = count / len(metrics) * 100
        lines.append(f"| {topic} | {count} | {pct:.1f} | {avg_s:.2f} | {avg_t:.1f} |")
    lines.append("")
    lines.append(tool_count_caveat())
    lines.append("")

    # --- tool usage patterns ---
    lines.append("## Tool Usage Patterns\n")
    lines.extend(tool_coverage.summary_lines())
    lines.append("")
    all_tools: list[str] = []
    for m in metrics:
        if m.tool_sequence:
            all_tools.extend(m.tool_sequence.split(" -> "))
    tool_freq = Counter(all_tools)
    lines.append("### Most used tools\n")
    lines.append("| Tool | Count |")
    lines.append("|------|------:|")
    for tool, count in tool_freq.most_common(15):
        lines.append(f"| {tool} | {count} |")
    lines.append("")

    # tool call distribution
    tool_counts = [m.total_tool_calls for m in metrics]
    lines.append("### Tool calls per session\n")
    lines.append(f"- Min: {min(tool_counts)}, Max: {max(tool_counts)}, "
                 f"Median: {sorted(tool_counts)[len(tool_counts)//2]}, "
                 f"Mean: {sum(tool_counts)/len(tool_counts):.1f}")
    lines.append("")
    lines.append(tool_count_caveat())
    lines.append("")

    # --- SDK function calls from sandboxed scripts (genetics-results-suite-4h6.12) ---
    # Reported next to the tool counts and never folded into them: a tool call is one model
    # decision, an SDK call is one line of a script the model wrote, and adding them would
    # make "calls per session" mean two different things at once.
    lines.append("### SDK function calls (sandboxed scripts)\n")
    if sdk_stats is None:
        lines.append(
            "_No SDK audit log supplied (`--sdk-log`). Data access from inside `run_analysis` "
            "scripts is invisible to every tool count above: the whole script is ONE tool "
            "call. Absence here is absence of the log, not absence of script access._"
        )
    else:
        sdk_metrics = [m for m in metrics if m.total_sdk_calls]
        # totals come from the log, not from `metrics`: a call whose session id matches no
        # session in this DB joins to nothing and would otherwise disappear from the report
        logged_calls = int(sdk_stats.select(pl.col("total_sdk_calls").sum()).item() or 0)
        logged_rows = int(sdk_stats.select(pl.col("sdk_rows").sum()).item() or 0)
        attributed = sum(m.total_sdk_calls for m in metrics)
        lines.append(f"- Sessions with SDK calls: {len(sdk_metrics)}/{len(metrics)}")
        lines.append(f"- Total SDK function calls: {logged_calls}")
        lines.append(f"- Total rows returned to scripts: {logged_rows}")
        notices = sdk_notices or {}
        if notices.get("shared_stream"):
            lines.append("")
            lines.append(
                "> **These lines are not a tamper-evident audit trail.** The log declares "
                "(see `SHARED_STREAM_WARNING` in `sdk/client.py`) that the SDK had no "
                "dedicated audit fd, so its records shared a stream the audited script writes "
                "to itself: any line here — including its user and session — may have been "
                "written by the script rather than by the SDK, and the counts below are an "
                "upper bound on what actually happened. Delivering a dedicated fd is "
                "genetics-results-suite-4h6.45."
            )
        if notices.get("truncated"):
            lines.append("")
            lines.append(
                f"_{notices['truncated']} process(es) hit the SDK audit ceiling of "
                f"{notices['truncated_at']} records; their REFUSED calls past that point are "
                "missing from the refusal count below. The ceiling applies only to calls that "
                "never reached the executor, so no data access is missing because of it._"
            )
        if notices.get("rejected"):
            lines.append(f"- Calls refused before any data access: {notices['rejected']}")
        if sdk_metrics:
            lines.append("")
            # from the exact per-function counts, not from the (bounded) sequence
            sdk_freq = Counter()
            for m in sdk_metrics:
                for entry in filter(None, m.sdk_function_counts.split("|")):
                    name, _, count = entry.rpartition(":")
                    sdk_freq[name] += int(count)
            lines.append("| SDK function | Calls |")
            lines.append("|--------------|------:|")
            for name, count in sdk_freq.most_common(15):
                lines.append(f"| {name} | {count} |")
        if logged_calls > attributed:
            lines.append("")
            lines.append(
                f"_{logged_calls - attributed} of {logged_calls} SDK calls match no session in "
                "this DB — `session=unknown`, or a session outside the analyzed range. The "
                "sandbox does not yet receive the token whose `sid` claim identifies the "
                "session (genetics-results-suite-4h6.43 and -4h6.44), so today every line "
                "is `unknown`._"
            )
    lines.append("")

    # --- high tool-use sessions ---
    # flag excessive tool use within a single message, not session totals: a long
    # conversation with many messages can legitimately accrue many calls
    lines.append("## Sessions with Excessive Tool Use (>10 calls in a single message)\n")
    heavy = sorted([m for m in metrics if m.max_tools_in_message > 10],
                   key=lambda m: m.max_tools_in_message, reverse=True)
    if heavy:
        lines.append("| Session | Max in msg | Total tools | Topic | Score | First Message |")
        lines.append("|---------|-----------:|------------:|-------|------:|---------------|")
        for m in heavy[:20]:
            msg_preview = m.first_user_message[:80].replace("|", "/").replace("\n", " ")
            lines.append(f"| {m.session_id} | {m.max_tools_in_message} | "
                         f"{tool_total(m)} | {m.topic} | {m.success_score:.2f} | "
                         f"{msg_preview} |")
        lines.append("")
        lines.append(tool_count_caveat())
    else:
        lines.append("None found.\n")
    lines.append("")

    # --- unsuccessful conversations (genuine agent failures only) ---
    lines.append("## Unsuccessful Conversations\n")
    lines.append("Genuine agent failures only — the question was answerable but the "
                 "assistant did not answer it well. Technical failures and out-of-scope "
                 "/ unfinished / weird requests are in their own sections below.\n")
    unsuccessful = [m for m in metrics if m.success_label == "unsuccessful"]
    if unsuccessful:
        lines.append("| Session | Score | Topic | Tools | First Message |")
        lines.append("|---------|------:|-------|------:|---------------|")
        for m in sorted(unsuccessful, key=lambda m: m.success_score)[:20]:
            msg_preview = m.first_user_message[:80].replace("|", "/").replace("\n", " ")
            lines.append(f"| {m.session_id} | {m.success_score:.2f} | "
                         f"{m.topic} | {tool_total(m)} | {msg_preview} |")
        lines.append("")
        lines.append(tool_count_caveat())
    else:
        lines.append("No unsuccessful conversations found.\n")
    lines.append("")

    # --- disposition buckets (not agent-quality failures) ---
    for label, title, blurb in [
        ("technical_failure", "Technical Failures",
         "Infrastructure/backend problems (connection drops, empty tool errors), not "
         "the agent's fault. Low score, but excluded from the agent-quality metric."),
        ("out_of_scope", "Out of Scope",
         "User asked for data or actions the system genuinely cannot provide. Not "
         "penalized in the quality metric."),
        ("unfinished", "Unfinished Conversations",
         "User stopped without a failure by the assistant. Not penalized."),
        ("weird_or_unclear", "Weird / Unclear Questions",
         "Unclear or malformed request (e.g. missing attachment). Not penalized."),
    ]:
        bucket = [m for m in metrics if m.success_label == label]
        if not bucket:
            continue
        lines.append(f"## {title}\n")
        lines.append(f"{blurb}\n")
        lines.append("| Session | Topic | Tools | First Message |")
        lines.append("|---------|-------|------:|---------------|")
        for m in sorted(bucket, key=lambda m: m.session_id)[:20]:
            msg_preview = m.first_user_message[:80].replace("|", "/").replace("\n", " ")
            lines.append(f"| {m.session_id} | {m.topic} | {tool_total(m)} | "
                         f"{msg_preview} |")
        lines.append("")
        lines.append(tool_count_caveat())
        lines.append("")

    # --- tool profile analysis ---
    lines.append("## Tool Profile Usage\n")
    profile_counts = Counter(m.tool_profile for m in metrics)
    if profile_counts:
        lines.append("| Profile | Count | Avg Score |")
        lines.append("|---------|------:|----------:|")
        for profile, count in profile_counts.most_common():
            profile_label = profile or "(default)"
            profile_metrics = [m for m in metrics if m.tool_profile == profile]
            avg_s = sum(m.success_score for m in profile_metrics) / len(profile_metrics)
            lines.append(f"| {profile_label} | {count} | {avg_s:.2f} |")
    lines.append("")

    # --- instruction set analysis ---
    # without this split a user who asked for terse prose reads as a quality regression: the
    # judge sees a short answer and cannot know the user is the one who asked for it
    lines.append("## Instruction Set Usage\n")
    set_counts = Counter(m.instruction_set_name for m in metrics)
    if set_counts:
        lines.append("| Instructions | Count | Avg Score |")
        lines.append("|--------------|------:|----------:|")
        for set_name, count in set_counts.most_common():
            set_label = (set_name or "(none)").replace("|", "/")
            set_metrics = [m for m in metrics if m.instruction_set_name == set_name]
            avg_s = sum(m.success_score for m in set_metrics) / len(set_metrics)
            lines.append(f"| {set_label} | {count} | {avg_s:.2f} |")
    lines.append("")

    # --- user engagement ---
    lines.append("## User Engagement\n")
    msg_counts = [m.user_messages for m in metrics]
    lines.append(f"- Messages per session: min={min(msg_counts)}, max={max(msg_counts)}, "
                 f"median={sorted(msg_counts)[len(msg_counts)//2]}, "
                 f"mean={sum(msg_counts)/len(msg_counts):.1f}")
    multi_turn = sum(1 for m in metrics if m.user_messages >= 3)
    lines.append(f"- Multi-turn sessions (3+ user messages): {multi_turn} "
                 f"({multi_turn/len(metrics)*100:.1f}%)")

    # repeat users
    user_session_counts = sessions.group_by("user_id").len().sort("len", descending=True)
    repeat_users = user_session_counts.filter(pl.col("len") > 1).height
    total_users = user_session_counts.height
    lines.append(f"- Repeat users: {repeat_users}/{total_users} "
                 f"({repeat_users/total_users*100:.1f}%)")
    lines.append("")

    # top users
    lines.append("### Top users by session count\n")
    lines.append("| User | Sessions |")
    lines.append("|------|--------:|")
    for row in user_session_counts.head(10).iter_rows(named=True):
        lines.append(f"| {row['user_id']} | {row['len']} |")
    lines.append("")

    # --- LLM quality evaluation summary ---
    evaluated = [m for m in metrics if m.llm_quality_score is not None]
    if evaluated:
        lines.append("## LLM Quality Evaluation\n")
        lines.append(f"- **Evaluated**: {len(evaluated)}/{len(metrics)} conversations")

        # disposition breakdown (what kind of outcome each conversation was)
        disp_counts = Counter(m.llm_disposition or "unclassified" for m in evaluated)
        disp_parts = ", ".join(f"{v}: {c}" for v, c in disp_counts.most_common())
        lines.append(f"- **Disposition**: {disp_parts}")

        # headline quality score over answerable conversations only (good_answer /
        # agent_failure), so out-of-scope / unfinished / weird / technical don't skew it
        answerable = [m for m in evaluated if m.success_label in QUALITY_RELEVANT_LABELS]
        if answerable:
            avg_q = sum(m.llm_quality_score for m in answerable) / len(answerable)
            lines.append(f"- **Average quality score** (answerable only, n={len(answerable)}): "
                         f"{avg_q:.1f}/5\n")
        else:
            lines.append("")

        # breakdown by answered/accurate/efficient
        for field, label in [
            ("llm_answered", "Answered user's question"),
            ("llm_accurate", "Information accurate"),
            ("llm_efficient", "Tool calls efficient"),
            ("llm_concluded", "Reached conclusion"),
        ]:
            counts = Counter(getattr(m, field) for m in evaluated)
            parts = ", ".join(f"{v}: {c}" for v, c in counts.most_common() if v)
            lines.append(f"- **{label}**: {parts}")

        lines.append("")

        # most common issues: raw issue strings are too detailed to recur, so we
        # group them into a fixed taxonomy (issue_categories) to surface the real
        # underlying problems instead of a list of count-1 unique strings
        all_issues: list[str] = []
        for m in evaluated:
            if m.llm_issues:
                all_issues.extend(m.llm_issues)
        if all_issues:
            if issue_categories:
                cat_counter = Counter(issue_categories.get(i, "other") for i in all_issues)
                # shortest raw issue per category as a representative example
                examples: dict[str, str] = {}
                for i in all_issues:
                    c = issue_categories.get(i, "other")
                    if c not in examples or len(i) < len(examples[c]):
                        examples[c] = i
                lines.append("### Most common issue categories\n")
                lines.append("| Category | Count | % | Example |")
                lines.append("|----------|------:|--:|---------|")
                for cat, count in cat_counter.most_common():
                    ex = examples.get(cat, "")[:100].replace("|", "/").replace("\n", " ")
                    pct = count / len(all_issues) * 100
                    lines.append(f"| {cat} | {count} | {pct:.0f} | {ex} |")
            else:
                # no categorization available (e.g. --no-llm): raw frequency
                issue_freq = Counter(all_issues)
                lines.append("### Most common issues\n")
                lines.append("| Issue | Count |")
                lines.append("|-------|------:|")
                for issue, count in issue_freq.most_common(15):
                    iss = issue[:120].replace("|", "/").replace("\n", " ")
                    lines.append(f"| {iss} | {count} |")
            lines.append(f"\n*{len(all_issues)} issues across "
                         f"{len(evaluated)} evaluated conversations.*\n")

        # lowest quality conversations: answerable ones only, so genuine agent
        # weaknesses surface rather than technical/out-of-scope/unfinished cases
        low_quality = sorted(answerable, key=lambda m: m.llm_quality_score)[:10]
        lines.append("### Lowest quality conversations (answerable only)\n")
        lines.append("| Session | LLM Score | Topic | Tools | Answered | Issues |")
        lines.append("|---------|----------:|-------|------:|----------|--------|")
        for m in low_quality:
            issues_str = "; ".join(m.llm_issues) if m.llm_issues else ""
            issues_str = issues_str.replace("|", "/").replace("\n", " ")
            lines.append(f"| {m.session_id} | {m.llm_quality_score} | "
                         f"{m.topic} | {tool_total(m)} | {m.llm_answered} | "
                         f"{issues_str} |")
        lines.append("")
        lines.append(tool_count_caveat())
        lines.append("")

    # --- improvement recommendations ---
    lines.append("## Improvement Recommendations\n")

    # topics with low success (below overall average); computed over answerable
    # conversations only so out-of-scope/technical buckets don't distort topic scores
    if quality_metrics:
        overall_avg = sum(m.success_score for m in quality_metrics) / len(quality_metrics)
        for topic, count in topic_counts.most_common():
            topic_ms = [m for m in quality_metrics if m.topic == topic]
            if len(topic_ms) < 5:
                continue
            avg_s = sum(m.success_score for m in topic_ms) / len(topic_ms)
            if avg_s < overall_avg - 0.05:
                lines.append(f"- **{topic}** has below-average score ({avg_s:.2f} vs "
                             f"{overall_avg:.2f} overall) across {len(topic_ms)} "
                             "answerable sessions")

    # tools that appear in unsuccessful conversations
    unsuccessful_tools: list[str] = []
    for m in unsuccessful:
        if m.tool_sequence:
            unsuccessful_tools.extend(m.tool_sequence.split(" -> "))
    if unsuccessful_tools:
        ut_freq = Counter(unsuccessful_tools)
        lines.append(f"- Tools most common in unsuccessful conversations: "
                     f"{', '.join(t for t, _ in ut_freq.most_common(5))}")
        lines.append(f"  {tool_count_caveat()}")

    if heavy:
        avg_heavy_tools = sum(m.max_tools_in_message for m in heavy) / len(heavy)
        lines.append(f"- {len(heavy)} sessions made >10 tool calls in a single message "
                     f"(avg peak {avg_heavy_tools:.0f}) - consider optimizing tool strategies")

    lines.append("")

    # --- API cost (real, from token usage this run) ---
    if cost_tracker is not None and cost_tracker.usage:
        lines.append("## API Cost (this run)\n")
        lines.append("Real cost from token usage; cached results from prior runs are free.\n")
        lines.extend(cost_tracker.summary_lines())
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(
        description="Analyze conversation history for insights and eval extraction."
    )
    parser.add_argument("--db", required=True, help="Path to chat_history SQLite DB")
    parser.add_argument("--output-dir", default=None, help="Directory for output files")
    parser.add_argument("--llm-config-db", default=None,
                        help="Path to llm_config SQLite DB, used to name the instruction sets "
                             "the report groups by (default: llm_config.db beside --db). "
                             "If it is missing the report groups by set id instead.")
    parser.add_argument("--sdk-log", action="append", default=None, metavar="PATH",
                        help="Log file carrying the sandbox SDK's audit lines "
                             "('Executing SDK function: ...'). Repeatable. Data accessed "
                             "from inside run_analysis scripts is one tool call in the chat "
                             "DB no matter how many queries it issued, so without this the "
                             "report cannot see it at all.")
    parser.add_argument("--no-llm", action="store_true",
                        help="Use keyword categorization instead of LLM")
    parser.add_argument("--topic-model",
                        default=os.getenv("ANALYZE_TOPIC_MODEL", "claude-opus-5"),
                        help="Anthropic model for topic classification. "
                             "Defaults to $ANALYZE_TOPIC_MODEL if set.")
    parser.add_argument("--quality-model",
                        default=os.getenv("ANALYZE_QUALITY_MODEL", "claude-opus-5"),
                        help="Anthropic model for quality evaluation (judge task). "
                             "Defaults to $ANALYZE_QUALITY_MODEL if set.")
    parser.add_argument("--start-from", default=None,
                        help="Only include sessions created on or after this date (YYYY-MM-DD)")
    parser.add_argument("--until", default=None,
                        help="Only include sessions created before this date, exclusive (YYYY-MM-DD)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Ignore the DB-cached topic AND quality results; recompute "
                             "everything via the LLM (overwrites the cache). Use after "
                             "changing prompts or the truncation/eval logic.")
    parser.add_argument("--refresh-quality", action="store_true",
                        help="Ignore only the DB-cached quality assessments and re-run "
                             "them (keeps the cheap topic cache). Use to apply the "
                             "truncation fix without re-classifying topics.")
    parser.add_argument("--force", action="store_true",
                        help="Reanalyze every conversation from scratch, e.g. when the "
                             "analysis code changed. Recomputes topics AND quality for all "
                             "in-range sessions and overwrites their cached rows. --force "
                             "wins over --refresh-quality (it already recomputes everything).")
    parser.add_argument("--report-file", default=None,
                        help="Also write the stdout report (GitHub-flavored markdown) "
                             "to this file (default: <output-dir>/report.md)")
    parser.add_argument("--report-only", action="store_true",
                        help="Only print the report, skip eval export")
    args = parser.parse_args()

    # progress goes to stderr so stdout stays pipeable for the report; the level prefix
    # keeps log collectors from reading progress as failures, since GKE tags every line a
    # container writes to stderr as ERROR regardless of content
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stderr
    )

    if not os.path.exists(args.db):
        logger.error(f"database not found: {args.db}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.db).parent / "analysis_output"

    # --- load data ---
    logger.info("Loading data...")
    sessions, messages = load_data(args.db)

    if args.start_from or args.until:
        if args.start_from:
            sessions = sessions.filter(pl.col("created_at") >= args.start_from)
        if args.until:
            sessions = sessions.filter(pl.col("created_at") < args.until)
        session_ids = sessions.select("id").to_series().to_list()
        messages = messages.filter(pl.col("session_id").is_in(session_ids))
        logger.info(f"  Filtered to sessions in range "
                    f"[{args.start_from or '-inf'}, {args.until or '+inf'})")

    logger.info(f"  {sessions.height} sessions, {messages.height} messages")

    # --- parse SDK usage from the audit log ---
    sdk_stats = None
    sdk_notices = None
    if args.sdk_log:
        logger.info("Parsing SDK function calls...")
        sdk_calls, sdk_notices = load_sdk_log(args.sdk_log)
        sdk_stats = build_session_sdk_stats(sdk_calls)
        logger.info(f"  {len(sdk_calls)} SDK calls across {sdk_stats.height} sessions")
        if sdk_notices["shared_stream"]:
            logger.warning(
                "  this log came from a SHARED stream (no GENETICS_SDK_AUDIT_FD): its lines "
                "are forgeable by the audited script and are not a tamper-evident trail"
            )

    # --- parse tool usage ---
    logger.info("Parsing tool usage...")
    tool_stats = build_session_tool_stats(messages)
    sessions_with_tools = tool_stats.height
    total_tool_calls = tool_stats.select(pl.col("total_tool_calls").sum()).item() if sessions_with_tools > 0 else 0
    _, tool_coverage = build_tool_coverage(messages)
    logger.info(f"  {sessions_with_tools} sessions used tools, "
                f"{total_tool_calls} total tool calls")
    logger.info(
        f"  counting basis: {tool_coverage.messages_from_content_json}/"
        f"{tool_coverage.assistant_messages} assistant messages from content_json, "
        f"{tool_coverage.fully_covered_sessions}/{tool_coverage.sessions} sessions fully covered"
        + ("" if tool_coverage.is_exact else " — aggregates over the rest are LOWER BOUNDS")
    )

    # --- categorize ---
    logger.info("Categorizing conversations...")

    # build first-user-message list
    first_messages = (
        messages.filter(pl.col("role") == "user")
        .sort("created_at")
        .group_by("session_id")
        .first()
        .select("session_id", "content")
    )

    session_first_msgs = [
        {"id": row["session_id"], "text": row["content"]}
        for row in first_messages.iter_rows(named=True)
    ]

    # flat sidecar map (issue text -> taxonomy category); the per-session results
    # themselves now live in the SQLite conversation_analysis / conversation_issue
    # tables, not under .cache/
    cache_dir = output_dir / ".cache"
    issue_cat_cache = cache_dir / "issue_categories.json"

    # DB-backed analysis cache: read prior results to skip recompute, write new
    # results back. open the same DB we loaded the conversations from.
    from genetics_mcp_server.db.chat_history_db import ChatHistoryDB
    from genetics_mcp_server.db.singleton import Singleton

    if ChatHistoryDB in Singleton._instances:
        del Singleton._instances[ChatHistoryDB]
    analysis_db = ChatHistoryDB(args.db)
    analysis_map = analysis_db.get_analysis_map()

    # the raw stored updated_at per session, passed unchanged to upsert_analysis so
    # staleness comparisons (a string compare against analyzed_at) stay consistent
    updated_at_by_session = {
        row["id"]: row.get("updated_at")
        for row in sessions.iter_rows(named=True)
    }

    # reconstruct topic + quality inputs for already-analyzed sessions so they
    # don't hit the LLM again. --no-cache discards them entirely; --refresh-quality
    # keeps the cheap topic cache but drops cached judgments so they're re-judged.
    cached_topics_db, cached_quality_db, _ = (
        ({}, {}, {}) if args.no_cache
        else cached_topic_and_quality(analysis_map)
    )
    if args.refresh_quality:
        cached_quality_db = {}

    # staleness-based selection: ask the DB which sessions actually need
    # (re)analysis — missing rows, continued conversations (chat_sessions.updated_at
    # advanced past analyzed_at), or analyzer_version mismatches. --force selects
    # every session. We intersect with the in-range session set so --start-from /
    # --until still act as a filter on top of staleness, then drop the stale ones
    # from the cached maps so they fall through to the LLM. Non-stale sessions keep
    # their cache (no LLM work) but still flow into the report via the same maps,
    # so the report stays a full summary of all in-range conversations.
    in_range_ids = set(updated_at_by_session)
    stale_ids = set(
        analysis_db.get_stale_or_missing_session_ids(
            force=args.force, analyzer_version=ANALYZER_VERSION
        )
    ) & in_range_ids
    if stale_ids:
        logger.info(f"  {len(stale_ids)} of {len(in_range_ids)} in-range sessions are stale "
                    "(missing / continued / version-bumped) and will be reanalyzed")
    # force the stale ones to recompute by evicting their cached topic + quality.
    # --force already cleared both maps when it set no_cache-equivalent behavior
    # below, but for the non-force path this is the surgical, minimal-work eviction.
    for sid in stale_ids:
        cached_topics_db.pop(sid, None)
        cached_quality_db.pop(sid, None)

    # --force is a superset of --no-cache for the selected range: discard all cached
    # topic + quality so every in-range session is recomputed from scratch and its
    # row overwritten. This also makes --force win over --refresh-quality (topics
    # are recomputed too, not just quality).
    if args.force:
        cached_topics_db = {}
        cached_quality_db = {}

    # tracks real API cost from token usage; only reflects this run's live calls
    # (cached topic/quality results incur no new cost)
    cost_tracker = CostTracker()

    if args.no_llm:
        logger.info("  Using keyword categorization...")
        topics = {}
        for m in session_first_msgs:
            topic, confidence = categorize_by_keywords(m["text"])
            topics[m["id"]] = {
                "topic": topic,
                "complexity": 2,
                "brief_reason": f"keyword match (confidence={confidence:.1f})",
            }
    else:
        cached_topics = dict(cached_topics_db)
        if cached_topics:
            logger.info(f"  Loaded {len(cached_topics)} cached topic classifications")

        uncached_msgs = [m for m in session_first_msgs if m["id"] not in cached_topics]
        if uncached_msgs:
            logger.info(f"  Using LLM categorization for {len(uncached_msgs)} sessions "
                        f"(model={args.topic_model})...")
            new_topics = await categorize_with_llm(
                uncached_msgs, model=args.topic_model, cost_tracker=cost_tracker,
            )
            cached_topics.update(new_topics)
        topics = cached_topics

    topic_dist = Counter(v["topic"] for v in topics.values())
    logger.info(f"  Topics: {dict(topic_dist.most_common())}")

    # --- compute metrics ---
    logger.info("Computing success metrics...")
    instruction_set_names = load_instruction_set_names(
        resolve_llm_config_db(args.db, args.llm_config_db)
    )
    all_metrics = compute_all_metrics(
        sessions, messages, tool_stats, topics, instruction_set_names, sdk_stats=sdk_stats
    )

    # --- LLM quality evaluation ---
    if not args.no_llm:
        cached_quality: dict[str, dict] = dict(cached_quality_db)
        if cached_quality:
            logger.info(f"  Loaded {len(cached_quality)} cached quality assessments")

        session_ids = [m.session_id for m in all_metrics if m.session_id not in cached_quality]
        if session_ids:
            logger.info(f"Evaluating conversation quality with LLM ({len(session_ids)} conversations)...")
            new_assessments = await evaluate_quality_with_llm(
                session_ids, messages, model=args.quality_model,
                cost_tracker=cost_tracker,
            )
            cached_quality.update(new_assessments)
        else:
            logger.info("  All quality assessments cached, skipping LLM calls")

        apply_quality_assessments(all_metrics, cached_quality)
        logger.info(f"  {len(cached_quality)} conversations evaluated")

        # conversations the judge skipped (and with no user rating) have no quality
        # signal — label them 'unknown' instead of a heuristic successful/neutral
        unknown_n = mark_unscored_unknown(all_metrics)
        if unknown_n:
            logger.info(f"  {unknown_n} conversations had no quality judgment "
                        "→ labelled 'unknown'")

    # --- categorize detailed issues into recurring problem categories ---
    # raw judge issues are too specific to recur, so we map them onto a fixed
    # taxonomy with a cheap separate pass (cached by issue text, reuses the
    # quality cache above untouched)
    issue_categories: dict[str, str] = {}
    if not args.no_llm:
        distinct_issues = sorted({
            issue for m in all_metrics if m.llm_issues for issue in m.llm_issues
        })
        if distinct_issues:
            cached_cats: dict[str, str] = {}
            if issue_cat_cache.exists() and not args.no_cache:
                cached_cats = json.loads(issue_cat_cache.read_text())
                logger.info(f"  Loaded {len(cached_cats)} cached issue categories")

            uncategorized = [t for t in distinct_issues if t not in cached_cats]
            if uncategorized:
                logger.info(f"Categorizing {len(uncategorized)} distinct issues "
                            f"(model={args.topic_model})...")
                new_cats = await categorize_issues_with_llm(
                    uncategorized, model=args.topic_model, cost_tracker=cost_tracker,
                )
                cached_cats.update(new_cats)
                cache_dir.mkdir(parents=True, exist_ok=True)
                issue_cat_cache.write_text(json.dumps(cached_cats, indent=2))
            issue_categories = cached_cats

            # persist per-conversation categories for downstream use / plotting
            for m in all_metrics:
                if m.llm_issues:
                    m.llm_issue_categories = sorted({
                        issue_categories.get(i, "other") for i in m.llm_issues
                    })

    success_dist = Counter(m.success_label for m in all_metrics)
    logger.info(f"  Success: {dict(success_dist)}")

    # --- persist results to the SQLite analysis cache ---
    # one short transaction per session; source_updated_at is the raw stored
    # chat_sessions.updated_at (NOT reformatted to ISO 'T') so the staleness
    # comparison against analyzed_at stays a consistent string compare.
    for m in all_metrics:
        analysis_db.upsert_analysis(
            m,
            ANALYZER_VERSION,
            source_updated_at=updated_at_by_session.get(m.session_id),
            message_count=m.total_messages,
        )
    logger.info(f"  Persisted {len(all_metrics)} analyses to the DB cache")

    if cost_tracker.usage:
        logger.info(f"  API cost this run: ${cost_tracker.total_cost():.4f}")

    # --- generate report ---
    logger.info("Generating report...")
    report = generate_report(all_metrics, sessions, messages, tool_stats, cost_tracker,
                             issue_categories=issue_categories, sdk_stats=sdk_stats,
                             sdk_notices=sdk_notices)
    print(report)

    # also write the report (stdout content) to a markdown file
    report_path = Path(args.report_file) if args.report_file else output_dir / "report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report + "\n")
    logger.info(f"  Wrote report to {report_path}")

    # --- export eval dataset ---
    if not args.report_only:
        logger.info("Exporting eval dataset...")
        export_eval_dataset(all_metrics, messages, output_dir)

    # --- save metrics as JSON (local-dev only) ---
    # the DB cache is now the source of truth; metrics.json is a convenience
    # export for the PNG plot script, written only when --output-dir is given
    if args.output_dir:
        metrics_path = output_dir / "metrics.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump([asdict(m) for m in all_metrics], f, indent=2, default=str)
        logger.info(f"  Wrote metrics to {metrics_path}")

    logger.info("Done!")


if __name__ == "__main__":
    asyncio.run(main())
