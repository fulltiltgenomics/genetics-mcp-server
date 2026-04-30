"""Analyze conversation history to gain insights for improving the genetics AI assistant.

Usage:
    python -m genetics_mcp_server.scripts.analyze_conversations --db /path/to/chat_history.db
    python -m genetics_mcp_server.scripts.analyze_conversations --db /path/to/db --no-llm
    python -m genetics_mcp_server.scripts.analyze_conversations --db /path/to/db --start-from 2026-03-01
    python -m genetics_mcp_server.scripts.analyze_conversations --db /path/to/db --output-dir ./analysis_output

Analyzes conversations for:
- Topic categorization (LLM-based or keyword fallback)
- LLM-based quality evaluation of full conversations
- Tool usage patterns and efficiency
- Success/failure metrics
- Eval dataset extraction
"""

import argparse
import asyncio
import json
import os
import re
import sqlite3
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(db_path: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load chat_sessions and chat_messages into polars DataFrames."""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute("SELECT * FROM chat_sessions")
        cols = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        sessions = pl.DataFrame(
            {col: [row[i] for row in rows] for i, col in enumerate(cols)},
            schema_overrides={"rating": pl.Int64},
        )

        cursor = conn.execute("SELECT * FROM chat_messages")
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


# ---------------------------------------------------------------------------
# Tool usage parsing
# ---------------------------------------------------------------------------

# matches *[Using tool: tool_name; param1: val1, ...]*  or  *[Using tool: tool_name...]*
TOOL_MARKER_RE = re.compile(r"\*\[Using tool: ([^;.\]]+)[^]]*\]\*")


def parse_tool_calls(content: str) -> list[str]:
    """Extract tool names from assistant message content."""
    return TOOL_MARKER_RE.findall(content)


def build_session_tool_stats(messages: pl.DataFrame) -> pl.DataFrame:
    """Build per-session tool usage statistics."""
    assistant_msgs = messages.filter(pl.col("role") == "assistant")

    rows = []
    for row in assistant_msgs.iter_rows(named=True):
        tools = parse_tool_calls(row["content"])
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
        })

    tool_df = pl.DataFrame(rows)

    # aggregate per session
    session_tools = tool_df.group_by("session_id").agg(
        pl.col("tool_calls").flatten().alias("all_tools"),
        pl.col("tool_count").sum().alias("total_tool_calls"),
    ).with_columns(
        pl.col("all_tools").list.n_unique().alias("unique_tools"),
        pl.col("all_tools").list.join(" -> ").alias("tool_sequence"),
    )

    return session_tools


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
# LLM-based topic categorization
# ---------------------------------------------------------------------------

async def categorize_with_llm(
    session_first_messages: list[dict[str, str]],
    model: str = "claude-sonnet-4-20250514",
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
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
            # extract JSON from response (may have markdown fences)
            json_match = re.search(r"\[.*\]", text, re.DOTALL)
            if json_match:
                classifications = json.loads(json_match.group())
                for c in classifications:
                    results[c["id"]] = {
                        "topic": c["topic"],
                        "complexity": c.get("complexity", 2),
                        "brief_reason": c.get("brief_reason", ""),
                    }
        except Exception as e:
            print(f"  LLM categorization failed for batch {i // batch_size + 1}: {e}",
                  file=sys.stderr)
            # fall back to keyword for this batch
            for m in batch:
                topic, _ = categorize_by_keywords(m["text"])
                results[m["id"]] = {
                    "topic": topic,
                    "complexity": 2,
                    "brief_reason": "keyword fallback",
                }

        if i + batch_size < len(session_first_messages):
            print(f"  Categorized {min(i + batch_size, len(session_first_messages))}"
                  f"/{len(session_first_messages)} sessions...", file=sys.stderr)

    return results


# ---------------------------------------------------------------------------
# LLM-based quality evaluation
# ---------------------------------------------------------------------------

def _format_conversation_for_eval(
    session_id: str, messages: pl.DataFrame, max_chars: int = 15000,
) -> str:
    """Format a conversation for LLM quality evaluation, truncating if needed."""
    session_msgs = messages.filter(
        pl.col("session_id") == session_id
    ).sort("created_at")

    parts = []
    total_len = 0
    for row in session_msgs.iter_rows(named=True):
        role = row["role"].upper()
        content = row["content"]
        # truncate individual long messages (e.g. huge tool outputs)
        if len(content) > 3000:
            content = content[:3000] + "\n[... truncated ...]"
        part = f"[{role}]\n{content}\n"
        total_len += len(part)
        if total_len > max_chars:
            parts.append("[... conversation truncated for length ...]")
            break
        parts.append(part)

    return "\n".join(parts)


async def evaluate_quality_with_llm(
    session_ids: list[str],
    messages: pl.DataFrame,
    model: str = "claude-sonnet-4-20250514",
) -> dict[str, dict]:
    """Evaluate conversation quality using Anthropic API.

    Sends full conversations (truncated to ~15K chars) one at a time.

    Returns:
        dict mapping session_id to quality assessment dict
    """
    import anthropic

    from .conversation_prompts import QUALITY_ASSESSMENT_PROMPT

    client = anthropic.AsyncAnthropic()
    results = {}
    total = len(session_ids)

    for idx, sid in enumerate(session_ids):
        conversation_text = _format_conversation_for_eval(sid, messages)

        # skip sessions with no real content
        if len(conversation_text.strip()) < 20:
            continue

        prompt = QUALITY_ASSESSMENT_PROMPT.format(conversation=conversation_text)

        try:
            response = await client.messages.create(
                model=model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                assessment = json.loads(json_match.group())
                results[sid] = assessment
        except Exception as e:
            print(f"  Quality eval failed for {sid[:8]}...: {e}", file=sys.stderr)

        if (idx + 1) % 20 == 0 or idx + 1 == total:
            print(f"  Evaluated {idx + 1}/{total} conversations...", file=sys.stderr)

    return results


# ---------------------------------------------------------------------------
# Success/failure metrics
# ---------------------------------------------------------------------------

@dataclass
class ConversationMetrics:
    session_id: str
    user_rating: int | None = None
    thumbs_up_count: int = 0
    thumbs_down_count: int = 0
    total_messages: int = 0
    user_messages: int = 0
    assistant_messages: int = 0
    total_tool_calls: int = 0
    unique_tools: int = 0
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
    # LLM quality assessment fields
    llm_quality_score: int | None = None
    llm_answered: str = ""
    llm_accurate: str = ""
    llm_efficient: str = ""
    llm_concluded: str = ""
    llm_issues: list[str] | None = None


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
        return round((q - 1) / 4.0, 3)

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


def label_success(score: float) -> str:
    if score >= 0.7:
        return "successful"
    elif score >= 0.4:
        return "neutral"
    else:
        return "unsuccessful"


def compute_all_metrics(
    sessions: pl.DataFrame,
    messages: pl.DataFrame,
    tool_stats: pl.DataFrame,
    topics: dict[str, dict],
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

    results = []
    for row in combined.iter_rows(named=True):
        sid = row["id"]
        topic_info = topics.get(sid, {})

        m = ConversationMetrics(
            session_id=sid,
            user_rating=row.get("rating"),
            thumbs_up_count=row.get("thumbs_up_count") or 0,
            thumbs_down_count=row.get("thumbs_down_count") or 0,
            total_messages=row.get("total_messages") or 0,
            user_messages=row.get("user_messages") or 0,
            assistant_messages=row.get("assistant_messages") or 0,
            total_tool_calls=row.get("total_tool_calls") or 0,
            unique_tools=row.get("unique_tools") or 0,
            has_error_response=sid in error_sessions,
            topic=topic_info.get("topic", "general_genetics"),
            complexity=topic_info.get("complexity", 2),
            topic_reason=topic_info.get("brief_reason", ""),
            tool_sequence=row.get("tool_sequence") or "",
            first_user_message=row.get("first_user_message") or "",
            tool_profile=row.get("tool_profile") or "",
        )
        m.success_score = compute_success_score(m)
        m.success_label = label_success(m.success_score)
        results.append(m)

    return results


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
            m.llm_issues = qa.get("issues")
            # recompute score now that LLM quality is available
            m.success_score = compute_success_score(m)
            m.success_label = label_success(m.success_score)


# ---------------------------------------------------------------------------
# Eval dataset export
# ---------------------------------------------------------------------------

def export_eval_dataset(
    metrics: list[ConversationMetrics],
    messages: pl.DataFrame,
    output_dir: Path,
    max_per_topic: int = 5,
):
    """Export representative conversations as eval test cases."""
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
            ).sort("created_at")

            turns = []
            for msg_row in session_msgs.iter_rows(named=True):
                turns.append({
                    "role": msg_row["role"],
                    "content": msg_row["content"][:2000],
                })

            tools_used = parse_tool_calls(
                " ".join(
                    msg_row["content"]
                    for msg_row in session_msgs.filter(pl.col("role") == "assistant").iter_rows(named=True)
                )
            )

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
                "turn_count": len(turns),
                "turns": turns,
            })

    # write JSON eval file
    eval_path = output_dir / "eval_dataset.json"
    with open(eval_path, "w") as f:
        json.dump(eval_cases, f, indent=2, default=str)
    print(f"  Wrote {len(eval_cases)} eval cases to {eval_path}", file=sys.stderr)

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

    print(f"  Wrote {len(eval_cases)} transcripts to {transcripts_dir}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    metrics: list[ConversationMetrics],
    sessions: pl.DataFrame,
    messages: pl.DataFrame,
    tool_stats: pl.DataFrame,
) -> str:
    """Generate a markdown analysis report."""
    lines = ["# Conversation Analysis Report\n"]

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
    lines.append("## Success Breakdown\n")
    success_counts = Counter(m.success_label for m in metrics)
    for label in ["successful", "neutral", "unsuccessful"]:
        count = success_counts.get(label, 0)
        pct = count / len(metrics) * 100
        lines.append(f"- **{label}**: {count} ({pct:.1f}%)")
    lines.append("")

    avg_score = sum(m.success_score for m in metrics) / len(metrics)
    lines.append(f"Average success score: {avg_score:.3f}\n")

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

    # --- tool usage patterns ---
    lines.append("## Tool Usage Patterns\n")
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

    # --- high tool-use sessions ---
    lines.append("## Sessions with Excessive Tool Use (>10 calls)\n")
    heavy = sorted([m for m in metrics if m.total_tool_calls > 10],
                   key=lambda m: m.total_tool_calls, reverse=True)
    if heavy:
        lines.append("| Session | Tools | Topic | Score | First Message |")
        lines.append("|---------|------:|-------|------:|---------------|")
        for m in heavy[:20]:
            msg_preview = m.first_user_message[:80].replace("|", "/").replace("\n", " ")
            lines.append(f"| {m.session_id[:8]}... | {m.total_tool_calls} | "
                         f"{m.topic} | {m.success_score:.2f} | {msg_preview} |")
    else:
        lines.append("None found.\n")
    lines.append("")

    # --- unsuccessful conversations ---
    lines.append("## Unsuccessful Conversations\n")
    unsuccessful = [m for m in metrics if m.success_label == "unsuccessful"]
    if unsuccessful:
        lines.append("| Session | Score | Topic | Tools | First Message |")
        lines.append("|---------|------:|-------|------:|---------------|")
        for m in sorted(unsuccessful, key=lambda m: m.success_score)[:20]:
            msg_preview = m.first_user_message[:80].replace("|", "/").replace("\n", " ")
            lines.append(f"| {m.session_id[:8]}... | {m.success_score:.2f} | "
                         f"{m.topic} | {m.total_tool_calls} | {msg_preview} |")
    else:
        lines.append("No unsuccessful conversations found.\n")
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
        user_id = row["user_id"]
        # anonymize email
        if "@" in user_id:
            local, domain = user_id.split("@", 1)
            user_id = f"{local[:3]}...@{domain}"
        lines.append(f"| {user_id} | {row['len']} |")
    lines.append("")

    # --- LLM quality evaluation summary ---
    evaluated = [m for m in metrics if m.llm_quality_score is not None]
    if evaluated:
        lines.append("## LLM Quality Evaluation\n")
        lines.append(f"- **Evaluated**: {len(evaluated)}/{len(metrics)} conversations")
        avg_q = sum(m.llm_quality_score for m in evaluated) / len(evaluated)
        lines.append(f"- **Average quality score**: {avg_q:.1f}/5\n")

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

        # most common issues
        all_issues: list[str] = []
        for m in evaluated:
            if m.llm_issues:
                all_issues.extend(m.llm_issues)
        if all_issues:
            issue_freq = Counter(all_issues)
            lines.append("### Most common issues\n")
            lines.append("| Issue | Count |")
            lines.append("|-------|------:|")
            for issue, count in issue_freq.most_common(15):
                lines.append(f"| {issue} | {count} |")
            lines.append("")

        # lowest quality conversations
        low_quality = sorted(evaluated, key=lambda m: m.llm_quality_score)[:10]
        lines.append("### Lowest quality conversations\n")
        lines.append("| Session | LLM Score | Topic | Tools | Answered | Issues |")
        lines.append("|---------|----------:|-------|------:|----------|--------|")
        for m in low_quality:
            issues_str = "; ".join(m.llm_issues[:2]) if m.llm_issues else ""
            lines.append(f"| {m.session_id[:8]}... | {m.llm_quality_score} | "
                         f"{m.topic} | {m.total_tool_calls} | {m.llm_answered} | "
                         f"{issues_str[:60]} |")
        lines.append("")

    # --- improvement recommendations ---
    lines.append("## Improvement Recommendations\n")

    # topics with low success (below overall average)
    overall_avg = sum(m.success_score for m in metrics) / len(metrics)
    for topic, count in topic_counts.most_common():
        if count < 5:
            continue
        topic_ms = [m for m in metrics if m.topic == topic]
        avg_s = sum(m.success_score for m in topic_ms) / len(topic_ms)
        if avg_s < overall_avg - 0.05:
            lines.append(f"- **{topic}** has below-average score ({avg_s:.2f} vs "
                         f"{overall_avg:.2f} overall) across {count} sessions")

    # tools that appear in unsuccessful conversations
    unsuccessful_tools: list[str] = []
    for m in unsuccessful:
        if m.tool_sequence:
            unsuccessful_tools.extend(m.tool_sequence.split(" -> "))
    if unsuccessful_tools:
        ut_freq = Counter(unsuccessful_tools)
        lines.append(f"- Tools most common in unsuccessful conversations: "
                     f"{', '.join(t for t, _ in ut_freq.most_common(5))}")

    if heavy:
        avg_heavy_tools = sum(m.total_tool_calls for m in heavy) / len(heavy)
        lines.append(f"- {len(heavy)} sessions used >10 tool calls "
                     f"(avg {avg_heavy_tools:.0f}) - consider optimizing tool strategies")

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
    parser.add_argument("--no-llm", action="store_true",
                        help="Use keyword categorization instead of LLM")
    parser.add_argument("--model", default="claude-sonnet-4-20250514",
                        help="Anthropic model for categorization")
    parser.add_argument("--start-from", default=None,
                        help="Only include sessions created on or after this date (YYYY-MM-DD)")
    parser.add_argument("--report-only", action="store_true",
                        help="Only print the report, skip eval export")
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"Error: database not found: {args.db}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.db).parent / "analysis_output"

    # --- load data ---
    print("Loading data...", file=sys.stderr)
    sessions, messages = load_data(args.db)

    if args.start_from:
        sessions = sessions.filter(pl.col("created_at") >= args.start_from)
        session_ids = sessions.select("id").to_series().to_list()
        messages = messages.filter(pl.col("session_id").is_in(session_ids))
        print(f"  Filtered to sessions from {args.start_from}", file=sys.stderr)

    print(f"  {sessions.height} sessions, {messages.height} messages", file=sys.stderr)

    # --- parse tool usage ---
    print("Parsing tool usage...", file=sys.stderr)
    tool_stats = build_session_tool_stats(messages)
    sessions_with_tools = tool_stats.height
    total_tool_calls = tool_stats.select(pl.col("total_tool_calls").sum()).item() if sessions_with_tools > 0 else 0
    print(f"  {sessions_with_tools} sessions used tools, "
          f"{total_tool_calls} total tool calls", file=sys.stderr)

    # --- categorize ---
    print("Categorizing conversations...", file=sys.stderr)

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

    # cache paths for LLM results
    cache_dir = output_dir / ".cache"
    topics_cache = cache_dir / "topics.json"
    quality_cache = cache_dir / "quality.json"

    if args.no_llm:
        print("  Using keyword categorization...", file=sys.stderr)
        topics = {}
        for m in session_first_msgs:
            topic, confidence = categorize_by_keywords(m["text"])
            topics[m["id"]] = {
                "topic": topic,
                "complexity": 2,
                "brief_reason": f"keyword match (confidence={confidence:.1f})",
            }
    else:
        # load cached topic classifications
        cached_topics = {}
        if topics_cache.exists():
            cached_topics = json.loads(topics_cache.read_text())
            print(f"  Loaded {len(cached_topics)} cached topic classifications", file=sys.stderr)

        uncached_msgs = [m for m in session_first_msgs if m["id"] not in cached_topics]
        if uncached_msgs:
            print(f"  Using LLM categorization for {len(uncached_msgs)} sessions "
                  f"(model={args.model})...", file=sys.stderr)
            new_topics = await categorize_with_llm(uncached_msgs, model=args.model)
            cached_topics.update(new_topics)
            cache_dir.mkdir(parents=True, exist_ok=True)
            topics_cache.write_text(json.dumps(cached_topics, indent=2))
        topics = cached_topics

    topic_dist = Counter(v["topic"] for v in topics.values())
    print(f"  Topics: {dict(topic_dist.most_common())}", file=sys.stderr)

    # --- compute metrics ---
    print("Computing success metrics...", file=sys.stderr)
    all_metrics = compute_all_metrics(sessions, messages, tool_stats, topics)

    # --- LLM quality evaluation ---
    if not args.no_llm:
        # load cached quality assessments
        cached_quality: dict[str, dict] = {}
        if quality_cache.exists():
            cached_quality = json.loads(quality_cache.read_text())
            print(f"  Loaded {len(cached_quality)} cached quality assessments", file=sys.stderr)

        session_ids = [m.session_id for m in all_metrics if m.session_id not in cached_quality]
        if session_ids:
            print(f"Evaluating conversation quality with LLM ({len(session_ids)} conversations)...",
                  file=sys.stderr)
            new_assessments = await evaluate_quality_with_llm(
                session_ids, messages, model=args.model,
            )
            cached_quality.update(new_assessments)
            cache_dir.mkdir(parents=True, exist_ok=True)
            quality_cache.write_text(json.dumps(cached_quality, indent=2))
        else:
            print("  All quality assessments cached, skipping LLM calls", file=sys.stderr)

        apply_quality_assessments(all_metrics, cached_quality)
        print(f"  {len(cached_quality)} conversations evaluated", file=sys.stderr)

    success_dist = Counter(m.success_label for m in all_metrics)
    print(f"  Success: {dict(success_dist)}", file=sys.stderr)

    # --- generate report ---
    print("Generating report...", file=sys.stderr)
    report = generate_report(all_metrics, sessions, messages, tool_stats)
    print(report)

    # --- export eval dataset ---
    if not args.report_only:
        print("Exporting eval dataset...", file=sys.stderr)
        export_eval_dataset(all_metrics, messages, output_dir)

    # --- save metrics as JSON ---
    metrics_path = output_dir / "metrics.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump([asdict(m) for m in all_metrics], f, indent=2, default=str)
    print(f"  Wrote metrics to {metrics_path}", file=sys.stderr)

    print("\nDone!", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
