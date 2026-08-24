"""
LLM service for handling chat interactions with multiple providers.

Supports OpenAI and Anthropic with streaming responses.
Integrates with MCP tools for agentic queries when using Anthropic.
"""

import asyncio
import csv
import io
import json
import logging
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, AsyncIterator

from genetics_mcp_server.config import (
    get_settings,
    model_rejects_temperature,
    model_supports_adaptive_thinking,
)
from genetics_mcp_server.config.defaults import (
    CONTINUE_TRUNCATED_PROMPT,
    CONTINUE_UNFILLED_PROMPT,
)
from genetics_mcp_server.cost import estimate_cost, get_context_window
from genetics_mcp_server.download_store import get_download_store
from genetics_mcp_server.mcp_proxy import (
    execute_external_tool,
    get_external_anthropic_tools,
    get_rag_anthropic_tools,
    initialize_external_servers,
    is_external_tool,
)
from genetics_mcp_server.subagent import SubagentService
from genetics_mcp_server.tools import TOOL_PROFILE_TOOLS, ToolExecutor, get_anthropic_tools
from genetics_mcp_server.tools.executor import ARTIFACTS_RETAINED_IN_CLEAR_NOTE

logger = logging.getLogger(__name__)


def anthropic_error_type(e: Exception) -> str | None:
    """Extract Anthropic's in-stream error type (e.g. 'overloaded_error', 'api_error').

    Errors that arrive mid-stream surface as a base APIStatusError carrying the
    streaming HTTP status (200, since the connection itself succeeded), so
    e.status_code is unreliable for these. The real type lives in the error body.
    """
    body = getattr(e, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and isinstance(err.get("type"), str):
            return err["type"]
    return None


# the display-only tool-use markers that end up in stored assistant `content`. Two shapes:
#
#  *[Using tool: name; k: v]*  — what this service streamed as prose until
#    genetics-results-suite-inp. Non-greedy up to the closing ']*' so an embedded ']' in
#    params (e.g. SQL) doesn't truncate the match; DOTALL because params can span lines.
#  [TOOLUSE:<base64>]          — written by the CLIENT from the structured `tool_use` chunk
#    that replaced it, so the disclosure survives a reload the same way [IMAGE:...] does.
#
# Both still occur: the first across all stored history, the second in everything new.
_TOOL_USE_MARKER_RE = re.compile(
    r"\*\[Using tool:.*?\]\*|\[TOOLUSE:[A-Za-z0-9+/=]*\]", re.DOTALL
)

# a cell the model wrote as a stand-in for data it never fetched, e.g. "*[from query]*"
_PLACEHOLDER_CELL_RE = re.compile(
    r"\[\s*(?:from (?:the )?quer(?:y|ies)|to confirm|to be confirmed|pending|tbd"
    r"|placeholder)\b[^\]]*\]",
    re.I,
)
# a markdown header separator: "|---|---:|" and friends
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|[\s:|-]+\|\s*$")


def _cell_is_data(cell: str) -> bool:
    """True if a table cell carries a real value rather than a stand-in.

    A markdown link is data — citation tables are full of them — so only blanks,
    explicit placeholders and bare dashes count as unfilled.
    """
    value = cell.strip().strip("*").strip()
    if not value or value in {"-", "--", "—", "–", "...", "…", "?", "N/A", "TBD"}:
        return False
    return not _PLACEHOLDER_CELL_RE.fullmatch(value)


def _has_unfilled_output(text: str) -> bool:
    """True if the text lays out results the model never obtained.

    Two shapes, both observed in stored conversations: placeholder cells, and a
    column-label header with no data under it. The first column is excluded from the
    row check because it holds the row label, which the model fills in from the
    question itself — "| CHRM4 | | |" is still an empty row.

    A body-less table only counts from three columns up. Two-column tables are the
    shape the model also uses for a single labelled value ("| Result | 0 rows |"),
    where the header row is the data and nothing is missing.

    This keys on the artifact rather than on "let me pull the rows" phrasing: measured
    over the stored history, the phrasing also ends many turns that are correctly
    waiting on the user ("paste your gene list and I'll run it"), where resuming would
    answer on the user's behalf.
    """
    if _PLACEHOLDER_CELL_RE.search(text):
        return True

    lines = text.split("\n")
    for i, line in enumerate(lines):
        if not _TABLE_SEPARATOR_RE.match(line):
            continue
        columns = len(line.strip().strip("|").split("|"))
        data_cells: list[str] = []
        rows = 0
        for row in lines[i + 1:]:
            if not row.strip().startswith("|"):
                break  # end of this table; a later table is judged on its own
            rows += 1
            data_cells.extend(row.strip().strip("|").split("|")[1:])
        if rows == 0:
            if columns >= 3:
                return True
            continue
        if columns >= 2 and not any(_cell_is_data(cell) for cell in data_cells):
            return True
    return False


def _strip_tool_use_markers(messages: list[dict]) -> list[dict]:
    """Remove '*[Using tool: ...]*' display markers from replayed assistant content.

    The markers are injected purely for live UI display, but get persisted into the
    stored assistant content. When fed back as plain text on replay, the model can
    imitate the notation — writing the marker as prose instead of emitting a real
    tool_use block, then fabricating the result. Stripping them before the history
    reaches the model removes that failure mode regardless of what the client sends.
    Operates only on assistant turns; real tool_use blocks are left untouched.
    """
    result = []
    for msg in messages:
        if msg.get("role") != "assistant":
            result.append(msg)
            continue
        content = msg.get("content")
        if isinstance(content, str):
            stripped = _TOOL_USE_MARKER_RE.sub("", content).strip()
            # never emit empty content; fall back to the original if the turn was
            # somehow nothing but markers
            result.append({**msg, "content": stripped or content})
        elif isinstance(content, list):
            new_blocks = []
            for b in content:
                if (
                    isinstance(b, dict)
                    and b.get("type") == "text"
                    and isinstance(b.get("text"), str)
                ):
                    text = _TOOL_USE_MARKER_RE.sub("", b["text"]).strip()
                    if text:  # drop blocks that were nothing but markers
                        new_blocks.append({**b, "text": text})
                else:
                    new_blocks.append(b)
            result.append({**msg, "content": new_blocks})
        else:
            result.append(msg)
    return result


def _sanitize_tool_blocks(messages: list[dict]) -> list[dict]:
    """Strip orphaned tool_use/tool_result blocks from conversation history.

    Persisted history may contain tool_use blocks in assistant messages without
    matching tool_result blocks in the next user message (because the agentic loop
    results were not persisted as separate messages). The Anthropic API rejects
    such sequences, so we remove orphaned blocks before sending.
    """
    result = []
    for i, msg in enumerate(messages):
        content = msg["content"]
        if not isinstance(content, list):
            result.append({"role": msg["role"], "content": content})
            continue

        if msg["role"] == "assistant":
            # collect tool_result ids from the next message (if any)
            next_tool_result_ids: set[str] = set()
            if i + 1 < len(messages):
                next_content = messages[i + 1].get("content")
                if isinstance(next_content, list):
                    next_tool_result_ids = {
                        b.get("tool_use_id") for b in next_content
                        if isinstance(b, dict) and b.get("type") == "tool_result"
                    }
            # keep tool_use blocks only if they have a matching tool_result
            content = [
                b for b in content
                if not isinstance(b, dict)
                or b.get("type") != "tool_use"
                or b.get("id") in next_tool_result_ids
            ]
        elif msg["role"] == "user":
            # collect tool_use ids from the previous message (if any)
            prev_tool_use_ids: set[str] = set()
            if i - 1 >= 0:
                prev_content = messages[i - 1].get("content")
                if isinstance(prev_content, list):
                    prev_tool_use_ids = {
                        b.get("id") for b in prev_content
                        if isinstance(b, dict) and b.get("type") == "tool_use"
                    }
            # keep tool_result blocks only if they have a matching tool_use
            content = [
                b for b in content
                if not isinstance(b, dict)
                or b.get("type") != "tool_result"
                or b.get("tool_use_id") in prev_tool_use_ids
            ]

        if content:
            result.append({"role": msg["role"], "content": content})
    return result


def _mark_history_cache_breakpoint(messages: list[dict]) -> None:
    """Add a cache_control breakpoint to the last block of the last message.

    Mutates `messages` in place. Normalizes a plain-string content into a single
    text block so the breakpoint can attach. No-op for an empty message list.
    """
    if not messages:
        return
    last = messages[-1]
    content = last.get("content")
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
        last["content"] = content
    if isinstance(content, list) and content and isinstance(content[-1], dict):
        content[-1] = {**content[-1], "cache_control": {"type": "ephemeral"}}


# comfortably under the client's 90s inactivity timeout, so several ticks are
# missed before a stalled stream is declared dead.
_THINKING_KEEPALIVE_SECONDS = 10.0

# Appended to the turn when the agentic loop stops at `mcp_max_iterations`. It is the ONLY
# signal a stream consumer has that the final iteration ran tools whose phase was never
# closed by a following model call — replay_benchmark keys its timing on it — so the text is
# a constant here rather than a literal repeated at both emission sites.
MAX_ITERATIONS_NOTICE = "\n\n*[Max tool iterations reached]*\n"

# run_analysis accepts up to 256 KiB of source; the log line exists to say what is running,
# not to carry the whole script into the log. This is a LOG cap only. It used to bound the
# streamed display too, back when the tool call reached the client as markdown prose with no
# way to expand it — which meant the one field the user most needed to read was the one field
# guaranteed to be cut off (genetics-results-suite-inp). The `tool_use` chunk carries the
# whole script and the client renders it collapsed.
_LOG_CODE_CHARS = 400


def _loggable_tool_input(tool_input: dict[str, Any]) -> dict[str, Any]:
    """Tool input with an oversize `code` capped, for the log line only."""
    code = tool_input.get("code")
    if isinstance(code, str) and len(code) > _LOG_CODE_CHARS:
        return {
            **tool_input,
            "code": f"{code[:_LOG_CODE_CHARS]}...[{len(code)} chars total]",
        }
    return tool_input


@dataclass
class StreamChunk:
    """A chunk from the LLM stream."""

    type: str  # "text", "thinking", "thinking_summary", "done", "image", "usage",
    # "script_result", "tool_use"
    content: str = ""
    # full message content blocks for persistence (only set when type="done")
    message_content: list[dict[str, Any]] | None = None
    # tool_result blocks for this turn, for persistence (only set when type="done")
    tool_results: list[dict[str, Any]] | None = None
    # image fields (only set when type="image")
    image_format: str | None = None
    image_alt: str | None = None


class DownloadShapeError(TypeError):
    """A `_download_data` payload does not match either shape `_convert_to_tsv` accepts.

    Not necessarily a local programming error: most of the ~25 producers put the sibling
    results-api's parsed response body straight into `_download_data` without shape-checking
    it (e.g. `executor._get_ld_matrix`, `_get_summary_stats`), so a shape defect can equally
    be upstream drift. That is why `_process_download_hints` surfaces it — an ERROR log plus
    a user-visible note — instead of letting it kill the chat turn. Silently swallowing it
    hid the same positional-rows bug twice (genetics-results-suite-bef, -buc).
    """


def _describe_shape(value: Any, depth: int = 0) -> str:
    """Compact type description of a download payload, for the error and the log line."""
    if isinstance(value, dict):
        if depth == 0:
            inner = ", ".join(f"{k}: {_describe_shape(v, depth + 1)}" for k, v in value.items())
            return f"{{{inner}}}"
        return f"dict[{len(value)} keys]"
    if isinstance(value, (list, tuple)):
        name = type(value).__name__
        if not value:
            return f"{name}[empty]"
        return f"{name}[{len(value)} x {_describe_shape(value[0], depth + 1)}]"
    return type(value).__name__


def _convert_to_tsv(download_info: dict) -> bytes:
    """Convert download data to TSV bytes.

    Supports two formats:
    - {"results": [list of dicts]} — keys from first dict become headers
    - {"columns": [...], "rows": [[...], ...]} — BigQuery-style columnar data

    The two shapes are deliberately NOT unified: `results` feeds the model and is capped,
    while `columns`/`rows` carries up to 100k rows positionally and must not be
    re-materialised as dicts just to be flattened again (see genetics-results-suite-bef).
    The cost of shape-per-consumer is that a producer can pass the wrong one, so every
    branch validates and raises DownloadShapeError naming expected vs observed rather than
    dying with a bare AttributeError deep inside the writer loop.
    """
    if not isinstance(download_info, dict):
        raise DownloadShapeError(
            f"_download_data must be a dict with 'results' or 'columns'+'rows'; got {_describe_shape(download_info)}"
        )

    # validated here rather than at the store: the sidecar is json.dump'd after the .tsv is
    # already on disk, so a non-str filename would raise mid-write and orphan the data file
    if "filename" in download_info and not isinstance(download_info["filename"], str):
        raise DownloadShapeError(
            "_download_data 'filename' must be a str; got "
            f"{_describe_shape(download_info['filename'])} in {_describe_shape(download_info)}"
        )

    buf = io.StringIO()
    writer = csv.writer(buf, delimiter="\t", lineterminator="\n")

    if "columns" in download_info and "rows" in download_info:
        columns, rows = download_info["columns"], download_info["rows"]
        if not isinstance(columns, (list, tuple)) or not isinstance(rows, (list, tuple)):
            raise DownloadShapeError(
                "_download_data columnar form expects {'columns': list, 'rows': list of lists}; "
                f"got {_describe_shape(download_info)}"
            )
        writer.writerow(columns)
        for row in rows:
            if not isinstance(row, (list, tuple)):
                raise DownloadShapeError(
                    "_download_data 'rows' must hold positional lists (one per row); "
                    f"got {_describe_shape(download_info)}"
                )
            writer.writerow(row)
    elif "results" in download_info:
        results = download_info["results"]
        if not isinstance(results, (list, tuple)):
            raise DownloadShapeError(
                "_download_data row form expects {'results': list of dicts}; "
                f"got {_describe_shape(download_info)}"
            )
        if not results:
            return b""
        if not all(isinstance(row, Mapping) for row in results):
            raise DownloadShapeError(
                "_download_data 'results' must hold dicts (column name -> value); positional "
                "rows belong in the {'columns': [...], 'rows': [[...]]} form instead. Got "
                f"{_describe_shape(download_info)}"
            )
        headers = list(results[0].keys())
        writer.writerow(headers)
        for row in results:
            writer.writerow([row.get(h, "") for h in headers])
    else:
        raise DownloadShapeError(
            "_download_data must carry either 'results' or both 'columns' and 'rows'; "
            f"got {_describe_shape(download_info)}"
        )

    return buf.getvalue().encode("utf-8")


def _add_include_in_response(result: dict, value: str) -> dict:
    """Add INCLUDE_IN_RESPONSE at the front of the dict so it survives JSON truncation."""
    return {"INCLUDE_IN_RESPONSE": value, **{k: v for k, v in result.items() if k != "INCLUDE_IN_RESPONSE"}}


def _count_result_items(result: Any) -> int | None:
    """Best-effort total item count for a tool result, for the truncation notice.

    Covers the shapes the data tools actually return: `results` (variant-level rows),
    `rows` with `total_rows` (query_database), and `n_cs`/`cs` (the credible-set
    summaries). Counting only `results` used to silently drop the count for summarized
    credible sets, leaving the model with no idea how much it was missing.
    """
    if not isinstance(result, dict):
        return None
    if isinstance(result.get("n_cs"), int):
        return result["n_cs"]
    if isinstance(result.get("results"), list):
        return len(result["results"])
    if isinstance(result.get("total_rows"), int):
        return result["total_rows"]
    if isinstance(result.get("rows"), list):
        return len(result["rows"])
    if isinstance(result.get("cs"), dict):
        return sum(len(v) for v in result["cs"].values() if isinstance(v, list))
    return None


def _truncation_notice(result: Any) -> str:
    """Warning appended to an over-long tool result.

    Spells out that what survives is an ordered PREFIX, not a sample: the underlying data
    is sorted (by significance, or by chromosome and position), so whatever ranks lowest is
    what got cut. Without this the model reads the visible portion as the whole answer and
    concludes that missing categories do not exist -- which is exactly how an IL7R caQTL
    query, truncated after its chromosome-1 pQTL rows, was reported as having no caQTL data
    at all when it in fact had 3,058 associations.
    """
    total = _count_result_items(result)
    scope = f"{total} total items" if total else "a larger result"
    notice = (
        f"\n\n[TRUNCATED: this is the beginning of {scope}, cut off mid-structure. "
        "The data is ORDERED, so the rows you cannot see are not a random sample -- entire "
        "categories (data types, resources, cell types, chromosomes) may be missing from the "
        "visible part. Do NOT use this result to count anything, to list what exists, or to "
        "conclude that something is absent. Re-run the tool with narrower arguments "
        "(e.g. data_types, resource) or with summarize=true, or use the download link above "
        "for the complete data.]"
    )
    if isinstance(result, dict) and result.get("artifacts_retained_in_clear") is True:
        # RE-ATTACHED, because truncation must not be able to DELETE A SECURITY FIELD
        # (genetics-results-suite-4h6.97). The prefix above is cut at a byte offset, and
        # `output` in a run_analysis result is script-controlled up to 64 KiB — so a script
        # that both provokes the retained-in-clear condition and prints ~50 KB would otherwise
        # cut its own warning out of what the model reads, behind a generic "[TRUNCATED: ...]"
        # that says nothing about a dropped field. `_render_analysis` also orders the field
        # ahead of `output` for the same reason; that defence depends on json.dumps preserving
        # insertion order, and this one does not, which is why both exist. `result` here is the
        # WHOLE pre-truncation dict, so the flag is read from what the executor produced rather
        # than from the surviving prefix.
        notice += (
            "\n\n[SECURITY NOTE PRESERVED THROUGH TRUNCATION: "
            + ARTIFACTS_RETAINED_IN_CLEAR_NOTE
            + "]"
        )
    return notice


DOWNLOAD_FAILED_NOTE = (
    "⚠️ The download file for these results could not be written, so there is no download "
    "link for this answer. The results shown are unaffected. Tell the user the download is "
    "unavailable and that the failure has been logged."
)

DOWNLOAD_SHAPE_NOTE = (
    "⚠️ The download file for these results could not be prepared because the result had an "
    "unexpected structure, so there is no download link for this answer. The results shown "
    "are unaffected. Tell the user the download is unavailable and that the problem has been "
    "logged for investigation. Do not suggest re-running the query — it would fail the same way."
)


def _process_download_hints(result: dict, owner: str | None = None, tool_name: str | None = None) -> dict:
    """Convert _download_url / _download_data hints into INCLUDE_IN_RESPONSE links.

    Uses relative URLs so links work regardless of deployment domain. `owner` binds the
    stored file to the user who ran the query so nobody else can fetch it by id.

    Nothing here is allowed to fail silently and nothing here is allowed to kill the turn.
    Storage failures and shape defects each get their own ERROR log token and their own
    user-visible INCLUDE_IN_RESPONSE note: catching everything and returning silently made
    two identical positional-rows bugs invisible in production (genetics-results-suite-bef,
    -buc), because a missing link is indistinguishable from a result that never warranted
    one. Shape defects are not fatal because most producers pass the sibling API's parsed
    body through unvalidated, so a bad shape is often upstream drift rather than a local bug
    — and losing the whole answer over a missing download link is the worse failure.
    """
    if not isinstance(result, dict) or not result.get("success"):
        return result

    if "_download_url" in result:
        url = result.pop("_download_url")
        # convert absolute URL to relative path for browser rendering
        from urllib.parse import urlparse
        parsed = urlparse(url)
        relative_url = f"{parsed.path}?{parsed.query}" if parsed.query else parsed.path
        link = f"\U0001f4e5 [Download full results as TSV]({relative_url})"
        return _add_include_in_response(result, link)

    if "_download_data" in result:
        download_info = result.pop("_download_data")
        try:
            tsv_bytes = _convert_to_tsv(download_info)
            if tsv_bytes:
                filename = download_info.get("filename", "results.tsv")
                store = get_download_store()
                download_id = store.store(tsv_bytes, filename, owner=owner)
                url = f"/chat/v1/downloads/{download_id}"
                link = f"\U0001f4e5 [Download full results as TSV]({url})"
                return _add_include_in_response(result, link)
        # a shape defect is deterministic and often upstream drift, so it is logged under
        # its own token (the suite's alerter pushes new ERROR lines to Slack) and reported
        # to the model without any "try again" advice
        except DownloadShapeError:
            logger.error(
                f"DOWNLOAD_SHAPE_DEFECT tool={tool_name or 'unknown'} "
                f"shape={_describe_shape(download_info)}",
                exc_info=True,
            )
            return _add_include_in_response(result, DOWNLOAD_SHAPE_NOTE)
        # OSError covers everything the store can fail with (ENOSPC, permissions, a
        # storage path that is not a directory); UnicodeEncodeError is reachable from real
        # upstream data, since JSON can decode lone surrogates that utf-8 cannot encode.
        except (OSError, UnicodeEncodeError) as e:
            logger.error(
                f"DOWNLOAD_FAILED tool={tool_name or 'unknown'} "
                f"shape={_describe_shape(download_info)} error={type(e).__name__}: {e}"
            )
            return _add_include_in_response(result, DOWNLOAD_FAILED_NOTE)

    return result


def _script_result_payload(
    iteration: int, result: dict[str, Any], *, tool_use_id: str | None = None
) -> dict[str, Any]:
    """Summarise one `run_analysis` outcome for the `script_result` SSE chunk.

    THE SANDBOX HAS NO EXIT CODE. The supervisor answers with a `status` of `ok`, `error`,
    `timeout` or `limit` (sandbox_client._result), so an `exit_code` here would be invented
    rather than measured, and a consumer that read `exit_code == 0` as "the script worked"
    would be reading a constant. `ok` is the field that carries the outcome.

    `ran` says exactly one thing: did the SANDBOX execute this script. `_render_analysis`
    always sets `status`, and every shape `run_analysis` returns without reaching the
    supervisor has no `status` key at all, so the absence is a reliable discriminator for
    THAT question.

    IT IS NOT A DISCRIMINATOR FOR WHOSE FAULT IT WAS, and reading it as one flatters the
    code arm. `ran: False` covers a restarting sandbox and a full queue, which say nothing
    about the model's script — but it also covers `EmptyScript` (the model emitted blank or
    non-string `code`) and `SandboxRejected` (the model chose a `timeout_s` outside 1..120,
    or oversize code), which are the model's doing entirely. `status` carries the executor's
    `error_type` for those, so a consumer classifies on the STATUS STRING, never on `ran`
    alone. Every non-run shape sets `error_type`, so the `"unknown"` fallback below means a
    genuinely unrecognised shape rather than "a blank script got here".
    """
    status = result.get("status")
    ran = isinstance(status, str)
    ok = bool(result.get("success"))
    error_type = result.get("error_type")
    error_type = error_type if isinstance(error_type, str) else None
    limit = result.get("limit_exceeded")
    duration_ms = result.get("duration_ms")
    return {
        "iteration": iteration,
        # the `tool_use` chunk this outcome belongs to. An iteration can hold more than one
        # run_analysis call, so the iteration number alone does not identify one.
        "tool_use_id": tool_use_id,
        "ran": ran,
        "ok": ok,
        "status": status if ran else (error_type or "unknown"),
        # the script's OWN wall clock, not this process's turn budget and not the client's
        # per-attempt read deadline: both of those leave the script possibly still running
        "timed_out": ran and status == "timeout",
        "exception": None if ok else error_type,
        "limit": limit if isinstance(limit, str) else None,
        "duration_ms": duration_ms if isinstance(duration_ms, int) and not isinstance(duration_ms, bool) else None,
    }


@dataclass(frozen=True)
class ResolvedLocalTools:
    """One request's local tool definitions, with the names projected off them.

    `names` is deliberately NOT stored: it is computed from `definitions` on every
    access, so there is no way to hold an object whose names describe a different tool
    set than the one it hands the model (genetics-results-suite-4h6.77). The invariant
    rests on that projection, not on `frozen=True`: freezing only prevents rebinding the
    field, and `definitions` is a plain list whose contents can still be mutated in place
    (`tests/test_tool_resolution_single_source.py` does exactly that to prove `names`
    follows).
    """

    definitions: list[dict[str, Any]]

    @property
    def names(self) -> set[str]:
        return {t["name"] for t in self.definitions}


class LLMService:
    """Service for LLM chat streaming with multi-provider support."""

    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        self.executor: ToolExecutor | None = None
        self.subagent_service: SubagentService | None = None
        self._initialize_clients()

    def _initialize_clients(self):
        """Initialize LLM provider clients based on available API keys."""
        settings = get_settings()

        if settings.openai_api_key:
            try:
                from openai import AsyncOpenAI

                self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
                logger.info("OpenAI client initialized")
            except ImportError:
                logger.warning("OpenAI package not installed")
            except Exception as e:
                logger.error(f"Error initializing OpenAI client: {e}")

        if settings.anthropic_api_key:
            try:
                from anthropic import AsyncAnthropic

                self.anthropic_client = AsyncAnthropic(api_key=settings.anthropic_api_key)
                logger.info("Anthropic client initialized")
            except ImportError:
                logger.warning("Anthropic package not installed")
            except Exception as e:
                logger.error(f"Error initializing Anthropic client: {e}")

        # initialize tool executor
        self.executor = ToolExecutor(
            api_base_url=settings.genetics_api_url,
            bigquery_api_url=settings.bigquery_api_url,
        )

        # initialize subagent service
        if self.anthropic_client and self.executor and settings.enable_subagents:
            self.subagent_service = SubagentService(self.anthropic_client, self.executor)
            logger.info("Subagent service initialized")

        # initialize external MCP servers
        external_tool_count = initialize_external_servers()
        if external_tool_count > 0:
            logger.info(f"Initialized {external_tool_count} tools from external MCP servers")

    def _disabled_tools(self) -> set[str]:
        """Tools this service must not advertise, whatever the profile.

        Single source of truth: never advertise launch_subagents unless the subagent
        service actually initialized. settings.disabled_tools gates only on the
        enable_subagents flag, but the service also requires a live anthropic client +
        executor; advertising a tool the service can't run is what produced the confusing
        "subagent service isn't available" error when a call came back.
        """
        disabled = set(get_settings().disabled_tools)
        if self.subagent_service is None:
            disabled.add("launch_subagents")
        return disabled

    def resolve_local_tools(
        self,
        tool_profile: str | None = None,
        enable_tools: bool = True,
        custom_tool_descriptions: dict[str, str] | None = None,
    ) -> ResolvedLocalTools:
        """Resolve this request's local tool definitions ONCE.

        The prompt gate (genetics-results-suite-4h6.69) requires the system prompt to be
        assembled against the tools the model is actually handed. Returning one object
        that carries the definitions and projects the names off them is what makes that
        structural rather than conventional: a caller builds the prompt from `.names` and
        hands the SAME object to `stream_chat`, so there is one derivation and no second
        one to drift from it (genetics-results-suite-4h6.77).
        """
        if not (enable_tools and get_settings().mcp_enabled):
            return ResolvedLocalTools([])
        return ResolvedLocalTools(
            get_anthropic_tools(
                custom_tool_descriptions,
                tool_profile=tool_profile,
                disabled_tools=self._disabled_tools(),
            )
        )

    def resolve_local_tool_names(
        self, tool_profile: str | None = None, enable_tools: bool = True
    ) -> set[str]:
        """Local tool names this service would advertise for such a request.

        External and RAG tools are excluded: they are proxied surfaces the system prompt
        does not name tool-by-tool.
        """
        return self.resolve_local_tools(tool_profile, enable_tools).names

    async def stream_chat(
        self,
        messages: list[dict],
        provider: str | None = None,
        model: str | None = None,
        system_prompt: str | None = None,
        enable_tools: bool = True,
        custom_tool_descriptions: dict[str, str] | None = None,
        literature_backend: str | None = None,
        tool_profile: str | None = None,
        secret: bool = False,
        user: str | None = None,
        session_id: str | None = None,
        user_instructions: str | None = None,
        message_id: str | None = None,
        capture_thinking: bool = False,
        gateway_asserted: bool = False,
        local_tools: ResolvedLocalTools | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream chat responses from LLM provider.

        Args:
            messages: Chat history in OpenAI format [{"role": "user", "content": "..."}]
            provider: "openai" or "anthropic" (defaults to config setting)
            model: Specific model to use (defaults to provider's default model)
            system_prompt: Server-assembled system prompt to prepend. Callers build it
                from default_system_prompt + verbosity_prompt; it is never client-supplied.
            enable_tools: Whether to enable MCP tools (Anthropic only)
            custom_tool_descriptions: Custom descriptions for tools
            literature_backend: Backend for literature search ('europepmc' or 'perplexity')
            tool_profile: Tool profile controlling which categories are available.
                None = all tools, "api" = general+api, "bigquery" = general+bigquery,
                "rag" = general+RAG external tools.
            secret: If True, suppress detailed logging to avoid persisting chat content.
            user: Authenticated user email for logging.
            session_id: Client conversation id, logged (id only) to count distinct conversations.
            user_instructions: Pre-wrapped envelope holding this user's stored instruction
                set. Kept separate from system_prompt rather than concatenated so it can occupy
                its own cache block on the Anthropic path; the OpenAI path has no equivalent
                breakpoint and concatenates the two into one system message.
            message_id: Client-generated id of the assistant message this turn will become,
                used only to key the recorded turn metrics to chat_messages. Optional: the
                metrics row is still written without it, it just cannot be joined.
            capture_thinking: emit each iteration's summarized reasoning as a
                `thinking_summary` chunk. OFF for every ordinary chat: the browser asks for
                the contentless `thinking` keepalive and nothing else, and this text is never
                added to `message_content`, so it is never persisted or replayed either way.
                Set by the replay benchmark, which needs the reasoning in its transcripts.
            gateway_asserted: whether `user` was asserted by auth-gateway having verified an
                oauth2-proxy session, rather than by some other holder of INTERNAL_API_SECRET.
                Carried alongside `user` — never derived from it — and consumed only by
                `run_analysis`, which will not dispatch without it
                (genetics-results-suite-4h6.84). Defaults False so a caller that does not
                state a provenance loses code execution rather than being trusted.
            local_tools: the already-resolved local tool set, from `resolve_local_tools`.
                Pass it whenever `system_prompt` was assembled against a tool list: the
                model is then handed the very definitions those names were projected off,
                so prompt and tools cannot disagree (genetics-results-suite-4h6.77).
                None resolves here instead, for callers that supply no tool-gated prompt.

        Yields:
            StreamChunk objects with text content and final message structure
        """
        settings = get_settings()
        provider = provider or settings.default_provider

        if provider == "openai":
            async for chunk in self._stream_openai(
                messages, model, system_prompt, user_instructions
            ):
                yield chunk
        elif provider == "anthropic":
            async for chunk in self._stream_anthropic(
                messages, model, system_prompt, enable_tools, custom_tool_descriptions,
                literature_backend, tool_profile, secret, user, session_id,
                user_instructions, message_id, capture_thinking,
                gateway_asserted=gateway_asserted,
                local_tools=local_tools,
            ):
                yield chunk
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    async def _stream_openai(
        self,
        messages: list[dict],
        model: str | None = None,
        system_prompt: str | None = None,
        user_instructions: str | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream chat from OpenAI."""
        if not self.openai_client:
            raise RuntimeError("OpenAI client not initialized. Check API key.")

        settings = get_settings()
        model = model or "gpt-4o"

        # the server-assembled prompt is the only system message that may ship: any
        # system-role message in `messages` is dropped, mirroring the Anthropic path.
        # `ChatMessage.role` already rejects 'system' at the API boundary, so this is
        # defence in depth for callers that reach stream_chat with raw dicts.
        messages = [msg for msg in messages if msg.get("role") != "system"]
        # concatenated rather than a second system message: the two-block split on the Anthropic
        # path exists to give each half its own prompt-cache breakpoint, which has no equivalent
        # here. Order matches that path — the envelope follows the server prompt. This used to be
        # dropped outright, so a user with a set selected saw it applied in the UI while the model
        # never received it (genetics-results-suite-b3v)
        combined = "\n\n".join(part for part in (system_prompt, user_instructions) if part)
        if combined:
            messages = [{"role": "system", "content": combined}] + messages

        try:
            logger.info(f"Streaming OpenAI chat with model {model}")
            openai_params: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "stream": True,
                "max_tokens": settings.max_tokens,
            }
            if settings.temperature is not None:
                openai_params["temperature"] = settings.temperature
            stream = await self.openai_client.chat.completions.create(**openai_params)

            accumulated_text = ""
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    accumulated_text += text
                    yield StreamChunk(type="text", content=text)

            yield StreamChunk(
                type="done", message_content=[{"type": "text", "text": accumulated_text}]
            )

        except Exception as e:
            logger.error(f"Error streaming OpenAI chat: {e}")
            raise

    async def _stream_anthropic(
        self,
        messages: list[dict],
        model: str | None = None,
        system_prompt: str | None = None,
        enable_tools: bool = True,
        custom_tool_descriptions: dict[str, str] | None = None,
        literature_backend: str | None = None,
        tool_profile: str | None = None,
        secret: bool = False,
        user: str | None = None,
        session_id: str | None = None,
        user_instructions: str | None = None,
        message_id: str | None = None,
        capture_thinking: bool = False,
        gateway_asserted: bool = False,
        local_tools: ResolvedLocalTools | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream chat from Anthropic with optional MCP tools and agentic loop."""
        if not self.anthropic_client:
            raise RuntimeError("Anthropic client not initialized. Check API key.")

        turn_started = time.monotonic()

        settings = get_settings()
        model = model or settings.default_model

        # convert messages to Anthropic format: first strip the display-only
        # '*[Using tool: ...]*' markers from replayed assistant text (so the model
        # never learns to imitate them as prose), then strip orphaned tool_use
        # blocks that were persisted without matching tool_result messages.
        # The system-role filter is kept even though `ChatMessage.role` now rejects
        # 'system' at the API boundary: this method is also callable with raw dicts,
        # and both provider paths should be safe on their own.
        anthropic_messages = _sanitize_tool_blocks(
            _strip_tool_use_markers(
                [msg for msg in messages if msg.get("role") != "system"]
            )
        )

        # cache the replayed conversation history: mark the last content block of the last
        # message with a cache_control breakpoint. This one earns its place by offsetting the
        # larger replayed payload now that tool results are persisted and replayed.
        # Anthropic allows 4 breakpoints and all 4 are spoken for — tool definitions, the shared
        # system block, this user's instruction block, and this one — so anything that wants a
        # new one has to take it from these.
        # Caveats: ephemeral cache TTL is ~5 min, and the cache lookback window means very long
        # tool-heavy single turns may not hit. Caching is most valuable for resumes and rapid
        # follow-ups.
        _mark_history_cache_breakpoint(anthropic_messages)

        # prepare request parameters
        request_params: dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": settings.max_tokens,
        }
        # temperature is off by default; only send it when explicitly
        # configured and the model supports it (Fable and Opus 4.7+ reject it)
        if settings.temperature is not None and not model_rejects_temperature(model):
            request_params["temperature"] = settings.temperature

        # be explicit rather than relying on the per-model default: Opus 5 thinks
        # when `thinking` is unset while 4.8/4.7 do not, so leaving it off makes
        # the token budget depend on which model is configured. Summarized display
        # is what lets thinking progress reach the client at all — see the
        # keepalive in the stream loop below.
        if model_supports_adaptive_thinking(model):
            request_params["thinking"] = {"type": "adaptive", "display": "summarized"}

        # the system prompt goes out as two separately cached blocks. Block 0 is identical
        # for every user (default prompt + response-length fragment), so one cache entry
        # per verbosity value serves the whole user base; block 1 carries only this user's
        # instruction envelope. Concatenating them would refragment the ~7.4K-token shared
        # block per user per event (~$0.043) rather than writing the small per-user block
        # (~$0.0025), and leaving the user block uncached costs ~$0.05 across a
        # 25-iteration turn.
        system_blocks: list[dict[str, Any]] = []
        if system_prompt:
            system_blocks.append(
                {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}
            )
        if user_instructions:
            system_blocks.append(
                {"type": "text", "text": user_instructions, "cache_control": {"type": "ephemeral"}}
            )
        if system_blocks:
            request_params["system"] = system_blocks

        # add tool definitions if enabled
        tool_definitions = None
        if enable_tools and settings.mcp_enabled:
            # the caller's resolution when it has one, so the tools the model gets are the
            # same objects the system prompt's tool names were projected off; resolving
            # here as well would be the second derivation 4h6.77 exists to delete
            # supplying `local_tools` together with `custom_tool_descriptions` or a
            # differing `enable_tools` is a CALLER ERROR: both are inputs to the
            # resolution, and a caller that already resolved has consumed them. Neither
            # diverges today, but both silently would:
            #   - `custom_tool_descriptions` is read only on this fallback, so it is
            #     dropped whenever `local_tools` is supplied (pre-4h6.77 it was always
            #     applied here). Dormant: chat_api does not pass it, and nothing loads
            #     those rows into the chat path at all (see routers/llm_config.py). If
            #     they are ever wired in, `chat_api` must pass them to
            #     `resolve_local_tools`, which already accepts them — NOT to `stream_chat`.
            #   - same shape for `enable_tools`: resolving with False and then calling
            #     `stream_chat(enable_tools=True, local_tools=...)` yields external+RAG
            #     tools and zero local ones, where pre-4h6.77 it yielded the full local
            #     set. Dormant: chat_api passes `request.enable_tools` to both.
            resolved = local_tools if local_tools is not None else self.resolve_local_tools(
                tool_profile, enable_tools, custom_tool_descriptions
            )
            tool_definitions = list(resolved.definitions)
            local_count = len(tool_definitions)

            # always-on external tools (gnomAD, Open Targets) excluded in RAG profile, and
            # in any explicit-allow-list profile: those name their surface exactly, so
            # re-adding ~20 proxied tools would defeat the surface they exist to measure
            external_tools = []
            if tool_profile != "rag" and tool_profile not in TOOL_PROFILE_TOOLS:
                external_tools = get_external_anthropic_tools()
                tool_definitions.extend(external_tools)

            # RAG tools only included when profile is None (all) or "rag"
            rag_tools = []
            if tool_profile is None or tool_profile == "rag":
                rag_tools = get_rag_anthropic_tools()
                tool_definitions.extend(rag_tools)

            # mark last tool for prompt caching so tool definitions are cached
            if tool_definitions:
                tool_definitions[-1] = {
                    **tool_definitions[-1],
                    "cache_control": {"type": "ephemeral"},
                }
            request_params["tools"] = tool_definitions
            if not secret:
                logger.info(
                    f"Including {len(tool_definitions)} MCP tools "
                    f"(profile={tool_profile or 'all'}, {local_count} local, "
                    f"{len(external_tools)} external, {len(rag_tools)} RAG)"
                )

        try:
            log_prefix = f"[user={user or 'unknown'}] [session={session_id or 'unknown'}] "
            if secret:
                logger.info(f"{log_prefix}Streaming Anthropic secret chat with model {model}")
            else:
                logger.info(f"{log_prefix}Streaming Anthropic chat with model {model}")
            max_iterations = settings.mcp_max_iterations
            iteration = 0
            total_cost = 0.0
            total_input_tokens = 0
            total_output_tokens = 0
            total_cache_read = 0
            total_cache_create = 0
            tool_call_count = 0

            # collect all content blocks for persistence
            all_content_blocks: list[dict[str, Any]] = []
            # collect tool_result blocks across iterations for persistence, so resumed
            # conversations replay the actual tool data and not just the prose summary
            all_tool_results: list[dict[str, Any]] = []
            continuations = 0
            truncated = False
            unfilled = False

            while iteration < max_iterations:
                iteration += 1

                # spans EVERY attempt of the retry loop below, backoff sleeps included, so
                # `turn_elapsed_ms` minus this is the wall time that was NOT the model call
                model_started = time.monotonic()

                # retry transient Anthropic errors with exponential backoff
                max_retries = 3
                attempt = 0
                for attempt in range(max_retries + 1):
                    text_yielded_this_attempt = False
                    try:
                        # 5 min timeout per iteration to prevent indefinite hangs
                        async with asyncio.timeout(300):
                            async with self.anthropic_client.messages.stream(**request_params) as stream:
                                last_keepalive = 0.0
                                async for event in stream:
                                    if event.type != "content_block_delta":
                                        continue
                                    delta = event.delta
                                    if delta.type == "text_delta":
                                        text_yielded_this_attempt = True
                                        yield StreamChunk(type="text", content=delta.text)
                                    elif delta.type == "thinking_delta":
                                        # thinking deltas never reach text_stream, so a long
                                        # reasoning phase reads as a dead connection to the
                                        # client's inactivity timer. Tick occasionally to keep
                                        # the stream alive; the event itself is the signal, so
                                        # the reasoning text stays out of the payload.
                                        now = time.monotonic()
                                        if now - last_keepalive >= _THINKING_KEEPALIVE_SECONDS:
                                            last_keepalive = now
                                            yield StreamChunk(type="thinking")

                                message = await stream.get_final_message()
                        break
                    except Exception as e:
                        from anthropic import APIConnectionError, APIStatusError
                        # mid-stream overload/internal errors arrive as a base
                        # APIStatusError with status_code=200, so also match on the
                        # error type carried in the body.
                        err_type = anthropic_error_type(e)
                        is_retryable = (
                            isinstance(e, APIConnectionError)
                            or (isinstance(e, APIStatusError) and e.status_code in (500, 502, 503, 529))
                            or err_type in ("overloaded_error", "api_error", "internal_server_error")
                        )
                        if not is_retryable or attempt >= max_retries:
                            raise
                        wait = 2 ** attempt
                        logger.warning(
                            f"{log_prefix}Retryable Anthropic error (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                            f"Retrying in {wait}s..."
                        )
                        if text_yielded_this_attempt:
                            yield StreamChunk(
                                type="text",
                                content="\n\n*[Connection interrupted, retrying...]*\n\n",
                            )
                        await asyncio.sleep(wait)

                model_ms = int((time.monotonic() - model_started) * 1000)
                model_attempts = attempt + 1

                # log token usage and cost for this iteration
                usage = message.usage
                input_tok = usage.input_tokens
                output_tok = usage.output_tokens
                cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
                cache_create = getattr(usage, "cache_creation_input_tokens", 0) or 0
                iter_cost = estimate_cost(model, input_tok, output_tok, cache_read, cache_create)
                total_cost += iter_cost
                total_input_tokens += input_tok
                total_output_tokens += output_tok
                total_cache_read += cache_read
                total_cache_create += cache_create
                logger.info(
                    f"{log_prefix}API call iteration={iteration} model={model} "
                    f"input_tokens={input_tok} output_tokens={output_tok} "
                    f"cache_read={cache_read} cache_create={cache_create} "
                    f"stop_reason={message.stop_reason} cost=${iter_cost:.4f}"
                )

                # actual context size includes cached tokens (Anthropic's input_tokens excludes them)
                context_tokens = input_tok + cache_read + cache_create
                context_window = get_context_window(model)
                # `input_tokens` here is the whole context, NOT the billed uncached input
                # — it is what the browser's context meter renders against context_window,
                # so its meaning is fixed. `cache_read` and `cache_create` are the two
                # cached components of it, reported separately because they differ by more
                # than 12x in price: billed uncached input is
                # `input_tokens - cache_read - cache_create`, which is exactly the
                # per-iteration increment of `total_input_tokens`. A consumer with no
                # database row (secret chat writes none) needs all three to price a turn.
                #
                # The two timing fields are deliberately named for their epoch, because
                # "elapsed" alone is ambiguous and a reader guessing wrong gets a wrong
                # bottleneck:
                #   `turn_elapsed_ms` — since the TURN started (the same monotonic zero as
                #     the `wall_ms` written to chat_turn_metrics), sampled at the moment this
                #     iteration's model response completed. Cumulative, so it only rises.
                #   `model_ms` — the wall time of THIS iteration's model call. NOT model
                #     latency: the span encloses the whole retry loop, so a transient error
                #     puts its 1/2/4s backoff sleep inside the figure, and because this is
                #     an async generator yielding per delta it also carries downstream SSE
                #     serialisation and socket backpressure. `model_attempts` is emitted
                #     beside it so a retry-inflated reading is identifiable rather than
                #     merely disclaimed: > 1 means the backoff is in there.
                #   `model_attempts` — how many times the streaming call was attempted this
                #     iteration, 1 when it succeeded first time.
                # Everything else is derivable from the pair: iteration N's segment is
                # turn_elapsed_ms[N] - turn_elapsed_ms[N-1] (with [0] = 0), and the part of
                # that segment which was not the model call is that difference minus
                # model_ms[N] — which for N > 1 is exactly iteration N-1's tool phase, since
                # this chunk is emitted before any tool of iteration N runs.
                yield StreamChunk(
                    type="usage",
                    content=json.dumps({
                        "iteration": iteration,
                        "input_tokens": context_tokens,
                        "cache_read": cache_read,
                        "cache_create": cache_create,
                        "output_tokens": output_tok,
                        "total_input_tokens": total_input_tokens,
                        "total_output_tokens": total_output_tokens,
                        "context_window": context_window,
                        "context_percent": round(context_tokens / context_window * 100, 1),
                        "turn_elapsed_ms": int((time.monotonic() - turn_started) * 1000),
                        "model_ms": model_ms,
                        "model_attempts": model_attempts,
                    }),
                )

                # add this iteration's content blocks. Thinking blocks are deliberately
                # not persisted: they are only replayable to the model that produced them,
                # and the sanitizers that rewrite stored turns don't know about them.
                #
                # `capture_thinking` changes what is EMITTED, never what is persisted: the
                # text goes out as its own chunk and still never enters all_content_blocks,
                # so a caller that asks for it cannot accidentally write reasoning into a
                # stored conversation or replay it to the model. Off by default, so the UI
                # keeps receiving only the contentless `thinking` keepalive above.
                #
                # `redacted_thinking` is skipped rather than emitted empty: its payload is
                # encrypted and carries no readable text, so there is nothing to show.
                # What DOES come through is the SUMMARY — `display: "summarized"` above —
                # since no model exposes the raw chain of thought.
                for block in message.content:
                    if block.type in ("thinking", "redacted_thinking"):
                        summary = getattr(block, "thinking", "") or ""
                        if capture_thinking and summary:
                            yield StreamChunk(
                                type="thinking_summary",
                                content=json.dumps(
                                    {"iteration": iteration, "text": summary}
                                ),
                            )
                        continue
                    all_content_blocks.append(block.model_dump(exclude_none=True))

                # check for tool use
                tool_uses = [b for b in message.content if b.type == "tool_use"]

                # a turn cut off by the output cap carries no tool_use blocks, so the loop
                # would otherwise break and report it as a completed answer. Resume it
                # instead. Guarded on tool_uses being empty: continuing a turn that holds
                # an unanswered tool_use would send an unpaired block back.
                if not tool_uses and message.stop_reason == "max_tokens":
                    if continuations >= settings.max_continuations:
                        truncated = True
                        logger.warning(
                            f"{log_prefix}Turn still truncated after "
                            f"{continuations} continuation(s); giving up"
                        )
                        break
                    continuations += 1
                    logger.info(
                        f"{log_prefix}Turn hit max_tokens; continuing "
                        f"({continuations}/{settings.max_continuations})"
                    )
                    # the partial turn has to be followed by a user turn — a trailing
                    # assistant message is a prefill, which Opus 4.6+ rejects outright.
                    request_params["messages"] = [
                        *request_params["messages"],
                        {
                            "role": "assistant",
                            "content": [b.model_dump(exclude_none=True) for b in message.content],
                        },
                        {"role": "user", "content": CONTINUE_TRUNCATED_PROMPT},
                    ]
                    continue

                # a turn that ends normally after laying out empty or placeholder
                # results, having called no tool at all, is the model announcing a
                # query it never ran. The loop would report the placeholders as the
                # answer, leaving the user to repeat the request to get the rows.
                # Resume it instead, bounded by the same continuation budget.
                if (
                    not tool_uses
                    and self.executor
                    and not all_tool_results
                    and message.stop_reason == "end_turn"
                    and _has_unfilled_output(
                        "".join(b.text for b in message.content if b.type == "text")
                    )
                ):
                    if continuations >= settings.max_continuations:
                        unfilled = True
                        logger.warning(
                            f"{log_prefix}Turn still presented unfilled results after "
                            f"{continuations} continuation(s); giving up"
                        )
                        break
                    continuations += 1
                    logger.info(
                        f"{log_prefix}Turn presented unfilled results with no tool call; "
                        f"continuing ({continuations}/{settings.max_continuations})"
                    )
                    request_params["messages"] = [
                        *request_params["messages"],
                        {
                            "role": "assistant",
                            "content": [b.model_dump(exclude_none=True) for b in message.content],
                        },
                        {"role": "user", "content": CONTINUE_UNFILLED_PROMPT},
                    ]
                    continue

                if not tool_uses or not self.executor:
                    break

                tool_call_count += len(tool_uses)

                # emit tool-use indicators to stream
                for tool_use in tool_uses:
                    effective_input = dict(tool_use.input)
                    if tool_use.name == "search_scientific_literature":
                        effective_input.pop("backend", None)
                        if literature_backend:
                            effective_input["backend"] = literature_backend
                    if tool_use.name == "run_analysis":
                        # mirror the execution-side strip in _execute_tool: a model-invented
                        # identity is discarded before the call, so logging or showing it
                        # would make a forged `user` read as identity in a log join. The
                        # authenticated pair is already on every line via log_prefix.
                        effective_input.pop("user", None)
                        effective_input.pop("session_id", None)
                    if secret:
                        logger.info(f"{log_prefix}Executing tool: {tool_use.name} (secret, input omitted)")
                    else:
                        logger.info(
                            f"{log_prefix}Executing tool: {tool_use.name} with input: "
                            f"{_loggable_tool_input(effective_input)}"
                        )
                    # structured, and carrying the input WHOLE: the client renders it as a
                    # collapsed disclosure it can expand, so nothing here needs to be sized
                    # for reading inline. `id` is what a later `script_result` correlates
                    # against to attach the outcome to this specific call.
                    yield StreamChunk(
                        type="tool_use",
                        content=json.dumps(
                            {
                                "id": tool_use.id,
                                "name": tool_use.name,
                                "input": effective_input,
                            },
                            default=str,
                        ),
                    )

                # separate subagent tool from regular tools for progress streaming
                subagent_tool = None
                regular_tool_uses = []
                for tu in tool_uses:
                    if tu.name == "launch_subagents" and self.subagent_service:
                        subagent_tool = tu
                    else:
                        regular_tool_uses.append(tu)

                raw_results_map: dict[str, dict[str, Any]] = {}

                # handle subagent tool with progress streaming
                if subagent_tool:
                    progress_queue: asyncio.Queue[str | None] = asyncio.Queue()

                    def _on_progress(msg: str) -> None:
                        progress_queue.put_nowait(msg)

                    async def _run_subagents() -> dict[str, Any]:
                        try:
                            result = await self.subagent_service.run_subagents(
                                subagent_tool.input.get("tasks", []),
                                progress_callback=_on_progress,
                            )
                            # log cost just like _execute_tool does
                            if result.get("success") and result.get("results"):
                                total_in = sum(r.get("input_tokens", 0) for r in result["results"])
                                total_out = sum(r.get("output_tokens", 0) for r in result["results"])
                                sa_model = settings.subagent_model or settings.fast_model
                                sa_cost = estimate_cost(sa_model, total_in, total_out)
                                logger.info(
                                    f"Subagents completed: {len(result['results'])} agents, "
                                    f"input_tokens={total_in} output_tokens={total_out} "
                                    f"estimated_cost=${sa_cost:.4f}"
                                )
                            return result
                        finally:
                            progress_queue.put_nowait(None)

                    # launch regular tools and subagents concurrently
                    subagent_task = asyncio.create_task(_run_subagents())
                    if regular_tool_uses:
                        regular_task = asyncio.create_task(
                            asyncio.gather(
                                *(self._execute_tool(tu.name, tu.input, literature_backend, user, session_id, gateway_asserted) for tu in regular_tool_uses)
                            )
                        )
                    else:
                        regular_task = None

                    # drain progress queue while subagents run
                    while True:
                        msg = await progress_queue.get()
                        if msg is None:
                            break
                        yield StreamChunk(type="text", content=f"\n\n*[{msg}]*\n\n")

                    subagent_result = await subagent_task
                    raw_results_map[subagent_tool.id] = subagent_result

                    if regular_task:
                        regular_results = await regular_task
                        for tu, res in zip(regular_tool_uses, regular_results):
                            raw_results_map[tu.id] = res
                else:
                    # no subagent tool — execute all tools in parallel as before
                    regular_results = await asyncio.gather(
                        *(self._execute_tool(tu.name, tu.input, literature_backend, user, session_id, gateway_asserted) for tu in regular_tool_uses)
                    )
                    for tu, res in zip(regular_tool_uses, regular_results):
                        raw_results_map[tu.id] = res

                # build ordered results list matching original tool_uses order
                raw_results = [raw_results_map[tu.id] for tu in tool_uses]

                # process results: extract images, truncate, build tool_results
                tool_results = []
                for tool_use, result in zip(tool_uses, raw_results):
                    if tool_use.name == "run_analysis" and isinstance(result, dict):
                        # metadata only, and emitted before the result is rendered into the
                        # tool_result block: the benchmark needs to know a script ran and how
                        # it ended, and nothing here carries source, output or artifact names
                        yield StreamChunk(
                            type="script_result",
                            content=json.dumps(
                                _script_result_payload(
                                    iteration, result, tool_use_id=tool_use.id
                                )
                            ),
                        )
                    # run_analysis returns a LIST: a script can write several figures, and
                    # unlike the single-plot tools it names each one. Stripped from the result
                    # for the same reason `image_base64` is — base64 in the tool_result is
                    # tokens the model pays for and cannot see.
                    if isinstance(result, dict) and isinstance(result.get("images"), list):
                        streamed_any = False
                        for image in result["images"]:
                            if not isinstance(image, dict):
                                continue
                            image_data = image.get("content_base64")
                            if not isinstance(image_data, str) or len(image_data) <= 100:
                                logger.warning(
                                    "invalid image artifact from %s: size=%s",
                                    tool_use.name,
                                    len(image_data) if isinstance(image_data, str) else 0,
                                )
                                continue
                            content_type = image.get("content_type") or "image/png"
                            name = image.get("name") or f"{tool_use.name} result"
                            logger.info(
                                "Streaming artifact image: name=%s type=%s size=%d chars",
                                name,
                                content_type,
                                len(image_data),
                            )
                            yield StreamChunk(
                                type="image",
                                content=image_data,
                                image_format=content_type.split("/", 1)[-1],
                                image_alt=str(name),
                            )
                            streamed_any = True
                        result = {k: v for k, v in result.items() if k != "images"}
                        if streamed_any:
                            result["note"] = (
                                "The image artifacts have been displayed to the user above. Do "
                                "not output any image placeholder or markdown - just describe "
                                "what the plot shows."
                            )

                    if isinstance(result, dict) and result.get("success") and result.get("image_base64"):
                        image_data = result["image_base64"]
                        image_format = result.get("image_format", "png")
                        if image_data and len(image_data) > 100:
                            logger.info(f"Streaming image: format={image_format}, size={len(image_data)} chars")
                            yield StreamChunk(
                                type="image",
                                content=image_data,
                                image_format=image_format,
                                image_alt=f"{tool_use.name} result",
                            )
                        else:
                            logger.warning(f"Invalid image data: size={len(image_data) if image_data else 0}")
                        result = {k: v for k, v in result.items() if k != "image_base64"}
                        result["note"] = "The image has been displayed to the user above. Do not output any image placeholder or markdown - just describe what the plot shows."

                    # convert download hints into INCLUDE_IN_RESPONSE links
                    result = _process_download_hints(result, owner=user, tool_name=tool_use.name)

                    result_json = json.dumps(result)

                    if len(result_json) > settings.mcp_max_result_size:
                        truncated_json = result_json[: settings.mcp_max_result_size - 1000]
                        result_json = truncated_json + _truncation_notice(result)

                    tool_results.append(
                        {"type": "tool_result", "tool_use_id": tool_use.id, "content": result_json}
                    )

                all_tool_results.extend(tool_results)

                # continue conversation with tool results
                request_params["messages"] = [
                    *request_params["messages"],
                    {"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in message.content]},
                    {"role": "user", "content": tool_results},
                ]

            if truncated:
                notice = "\n\n---\n*Response was cut short by the output token limit.*\n"
                yield StreamChunk(type="text", content=notice)
                all_content_blocks.append({"type": "text", "text": notice})

            if unfilled:
                notice = (
                    "\n\n---\n*The results above were left unfilled — the query was not "
                    "run. Ask again to retry.*\n"
                )
                yield StreamChunk(type="text", content=notice)
                all_content_blocks.append({"type": "text", "text": notice})

            if iteration >= max_iterations:
                yield StreamChunk(type="text", content=MAX_ITERATIONS_NOTICE)
                all_content_blocks.append({"type": "text", "text": MAX_ITERATIONS_NOTICE})

            logger.info(
                f"{log_prefix}Chat complete: model={model} iterations={iteration} "
                f"total_input_tokens={total_input_tokens} total_output_tokens={total_output_tokens} "
                f"total_cost=${total_cost:.4f}"
            )

            yield StreamChunk(
                type="done",
                message_content=all_content_blocks,
                tool_results=all_tool_results or None,
            )

            # after the terminator, never before it: `done` carries the message_content the
            # client persists, and an await placed ahead of it would put a SQLite busy wait
            # (up to the 5s default, whenever the nightly analysis job holds the write lock)
            # and a fresh cancellation point between the answer and its delivery. The
            # consumer in chat_api iterates this generator with a plain `async for` and no
            # `break`, so it is driven one step past this yield and the write still runs.
            #
            # Otherwise deliberately here and nowhere else: one metrics row exists exactly
            # when one "Chat complete" line exists, so the two can never disagree. A turn
            # that ended abnormally (exception, timeout, client disconnect) reaches neither
            # and is recorded by neither — its accumulators are partial and would bias the
            # cost-per-turn figures the benchmark gates on. A turn stopped by max_iterations
            # does reach both and is recorded: those are the expensive tail this exists to
            # measure.
            await self._record_turn_metrics(
                secret=secret,
                session_id=session_id,
                message_id=message_id,
                user_id=user,
                iterations=iteration,
                tool_call_count=tool_call_count,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                cache_read_tokens=total_cache_read,
                cache_create_tokens=total_cache_create,
                cost_usd=total_cost,
                wall_ms=int((time.monotonic() - turn_started) * 1000),
                tool_profile=tool_profile,
                model=model,
            )

        except asyncio.TimeoutError:
            logger.error("Anthropic streaming timed out after 300s")
            raise
        except Exception as e:
            logger.error(f"Error streaming Anthropic chat: {e}")
            raise

    async def _record_turn_metrics(self, *, secret: bool, **fields: Any) -> None:
        """Persist one completed turn's cost and roundtrip profile, best effort.

        Secret chat is skipped outright, before the database is even reached. The promise
        attached to it is that the conversation leaves no trace, and the same rule that
        keeps chat_messages empty has to keep this table empty: iteration counts and costs
        are not content, but a row keyed to a session id still says a conversation happened
        and how expensive it was.

        Everything else is swallowed. This runs inside a live SSE generator, so an
        exception here would truncate an answer the user already paid for — telemetry is
        never worth that. The write goes to a worker thread because chat_history.db lives
        on a ReadWriteOnce volume shared with the nightly analysis job: SQLite will block
        for its busy timeout when that job holds the write lock, and blocking the event
        loop would stall every other stream in this process, not just this one.
        """
        if secret:
            return
        try:
            from genetics_mcp_server.db.chat_history_db import get_chat_history_db

            await asyncio.to_thread(
                lambda: get_chat_history_db().record_turn_metrics(**fields)
            )
        except Exception as e:
            logger.warning(f"Failed to record turn metrics: {e}")

    async def _execute_tool(
        self,
        tool_name: str,
        tool_input: dict,
        literature_backend: str | None = None,
        user: str | None = None,
        session_id: str | None = None,
        gateway_asserted: bool = False,
    ) -> dict[str, Any]:
        """Execute a tool by name using the executor or external proxy."""
        try:
            # dispatch only what the resolved tool list actually advertised, the same
            # allowlist shape subagent.py's `_execute_subagent_tool` carries and for the
            # same reason: the local branch below calls any executor attribute the model
            # names, which makes withholding a tool advisory rather than enforced. The
            # model does not have to invent the name — ChatMessage.content accepts raw
            # blocks and _sanitize_tool_blocks drops only ORPHANED tool_use, so a
            # client-supplied history with a paired run_analysis tool_use/tool_result
            # survives verbatim and primes the model for a tool it was not given.
            # disabled_tools is the profile-independent complement (it is applied before
            # the profile filter at the resolution site above), so a name in it is
            # advertised by no profile.
            disabled = get_settings().disabled_tools
            if tool_name in disabled:
                logger.warning(
                    f"Refusing to dispatch '{tool_name}': not enabled in this deployment"
                )
                return {
                    "success": False,
                    "error": f"Tool '{tool_name}' is not available in this deployment.",
                    # run_analysis carries the sandbox's own operator-error type because
                    # that is what the script_result chunk and the benchmark read; the
                    # point of both is that a withheld tool must not look transient, or
                    # the model retries a deployment fact (genetics-results-suite-4h6.56)
                    "error_type": (
                        "SandboxNotConfigured"
                        if tool_name == "run_analysis"
                        else "ToolNotEnabled"
                    ),
                    "retryable": False,
                }

            # subagent tool
            if tool_name == "launch_subagents":
                if not self.subagent_service:
                    return {"success": False, "error": "Subagent service not initialized"}
                result = await self.subagent_service.run_subagents(tool_input.get("tasks", []))
                if result.get("success") and result.get("results"):
                    total_in = sum(r.get("input_tokens", 0) for r in result["results"])
                    total_out = sum(r.get("output_tokens", 0) for r in result["results"])
                    settings = get_settings()
                    model = settings.subagent_model or settings.fast_model
                    cost = estimate_cost(model, total_in, total_out)
                    logger.info(
                        f"Subagents completed: {len(result['results'])} agents, "
                        f"input_tokens={total_in} output_tokens={total_out} "
                        f"estimated_cost=${cost:.4f}"
                    )
                return result

            # check if this is an external tool
            if is_external_tool(tool_name):
                logger.info(f"Executing external tool: {tool_name}")
                result = await execute_external_tool(tool_name, tool_input)
                result_str = json.dumps(result)
                logger.info(
                    f"External tool {tool_name} result ({len(result_str)} chars): "
                    f"{result_str[:200]}{'...[truncated]' if len(result_str) > 200 else ''}"
                )
                return result

            # local tool execution
            if not self.executor:
                return {"success": False, "error": "Tool executor not initialized"}

            method = getattr(self.executor, tool_name, None)
            if method is None:
                return {"success": False, "error": f"Unknown tool: {tool_name}"}

            # the literature backend is the user's choice, never the model's: the tool exposes no
            # `backend` argument, and any value the model invents is discarded here
            if tool_name == "search_scientific_literature":
                tool_input = {k: v for k, v in tool_input.items() if k != "backend"}
                if literature_backend:
                    tool_input["backend"] = literature_backend

            # the identity a sandbox execution runs as is the authenticated one, never the
            # model's: it becomes the subject and session of the per-execution credential and
            # of the audit trail. The tool exposes no such arguments, but tool_input is
            # splatted verbatim, so a model-invented `user` would otherwise arrive here —
            # stripped first, then the real pair injected, same as `backend` above.
            if tool_name == "run_analysis":
                tool_input = {
                    k: v
                    for k, v in tool_input.items()
                    if k not in ("user", "session_id", "gateway_asserted")
                }
                tool_input["user"] = user
                tool_input["session_id"] = session_id
                # the provenance of `user`, not a claim about it: a model that emitted this
                # key is stripped above alongside the identity pair, for the same reason
                tool_input["gateway_asserted"] = gateway_asserted

            # read_artifact resolves a model-supplied NAME against the executions this SESSION
            # ran (genetics-results-suite-4h6.52), so the session is the authorization and is
            # injected the same way — stripped first, because the tool declares only `name`
            # and tool_input is splatted verbatim. A model-supplied session_id would be a
            # cross-session read of another conversation's artifacts.
            if tool_name == "read_artifact":
                tool_input = {
                    k: v for k, v in tool_input.items() if k not in ("user", "session_id")
                }
                tool_input["session_id"] = session_id

            return await method(**tool_input)

        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return {"success": False, "error": str(e)}

    async def close(self):
        """Close resources."""
        if self.executor:
            await self.executor.close()


# singleton instance
_llm_service: LLMService | None = None


def get_llm_service() -> LLMService:
    """Get singleton LLM service instance."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
