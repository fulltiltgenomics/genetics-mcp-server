"""Tests for conversation analysis script."""

import json
import sqlite3
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import polars as pl
import pytest

import genetics_mcp_server.scripts.analyze_conversations as ac
from genetics_mcp_server.scripts.analyze_conversations import (
    ConversationMetrics,
    _attachment_note,
    _format_conversation_for_eval,
    _session_user_turns,
    apply_quality_assessments,
    build_session_tool_stats,
    build_tool_coverage,
    cached_topic_and_quality,
    categorize_by_keywords,
    categorize_issues_with_llm,
    categorize_with_llm,
    compute_all_metrics,
    compute_success_score,
    evaluate_quality_with_llm,
    export_eval_dataset,
    generate_report,
    label_from_disposition,
    label_success,
    load_data,
    mark_unscored_unknown,
    message_sort_keys,
    message_tool_calls,
    parse_tool_calls,
    parse_tool_calls_from_content_json,
)


def mock_llm_response(payload, *, leading_thinking: bool = False):
    """Mock a Messages API response whose blocks carry a .type, like the SDK's.

    Thinking-capable models can put a ThinkingBlock (no .text) first, so the
    mocks have to be block-type aware or they hide that failure mode.
    """
    blocks = []
    if leading_thinking:
        blocks.append(MagicMock(type="thinking", thinking="pondering..."))
    blocks.append(MagicMock(type="text", text=json.dumps(payload)))
    response = MagicMock()
    response.content = blocks
    return response


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_db(tmp_path):
    """Create a temporary SQLite DB with sample data."""
    db_path = str(tmp_path / "test_chat.db")
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE chat_sessions (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            rating INTEGER,
            comment TEXT,
            phenotype_code TEXT
        );
        CREATE TABLE chat_messages (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            thumbs_up BOOLEAN,
            content_json TEXT,
            literature_backend TEXT,
            tool_profile TEXT,
            FOREIGN KEY (session_id) REFERENCES chat_sessions(id)
        );
        CREATE TABLE chat_attachments (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            file_name TEXT NOT NULL,
            file_type TEXT NOT NULL,
            mime_type TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            storage_path TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES chat_sessions(id)
        );

        INSERT INTO chat_sessions VALUES
            ('s1', 'user1@test.com', 'Gene BRCA1', '2025-12-10', '2025-12-10', 5, NULL, NULL),
            ('s2', 'user1@test.com', 'Variant lookup', '2025-12-11', '2025-12-11', NULL, NULL, NULL),
            ('s3', 'user2@test.com', 'Literature search', '2025-12-12', '2025-12-12', 1, 'bad', NULL);

        INSERT INTO chat_messages VALUES
            ('m1', 's1', 'user', 'What do we know about gene BRCA1?', '2025-12-10 10:00:00', NULL, NULL, NULL, NULL),
            ('m2', 's1', 'assistant', 'I will look up BRCA1.\n\n*[Using tool: search_genes; query: BRCA1]*\n\n*[Using tool: get_credible_sets_by_gene; gene: BRCA1]*\n\nBRCA1 is associated with breast cancer.', '2025-12-10 10:00:01', NULL, NULL, NULL, NULL),
            ('m3', 's1', 'user', 'What about expression?', '2025-12-10 10:01:00', NULL, NULL, NULL, NULL),
            ('m4', 's1', 'assistant', '*[Using tool: get_gene_expression; gene: BRCA1]*\n\nBRCA1 is highly expressed in breast tissue.', '2025-12-10 10:01:01', 1, NULL, NULL, NULL),
            ('m5', 's2', 'user', 'What does variant 1:12345:A:G do?', '2025-12-11 10:00:00', NULL, NULL, NULL, 'api'),
            ('m6', 's2', 'assistant', '*[Using tool: get_variant_details; variant: 1:12345:A:G]*\n\nThis variant is in the coding region.', '2025-12-11 10:00:01', NULL, NULL, NULL, NULL),
            ('m7', 's3', 'user', 'Find papers about PCSK9', '2025-12-12 10:00:00', NULL, NULL, 'perplexity', NULL),
            ('m8', 's3', 'assistant', '*[Using tool: search_scientific_literature; query: PCSK9]*\n\nI found several papers. I apologize but I was unable to access the full text.', '2025-12-12 10:00:01', 0, NULL, NULL, NULL);
    """)
    conn.close()
    return db_path


# ---------------------------------------------------------------------------
# Tool parsing tests
# ---------------------------------------------------------------------------

class TestParseToolCalls:
    def test_basic_tool_marker(self):
        content = "*[Using tool: search_genes; query: BRCA1]*"
        assert parse_tool_calls(content) == ["search_genes"]

    def test_multiple_tools(self):
        content = (
            "*[Using tool: search_genes; query: TP53]*\n\n"
            "*[Using tool: get_credible_sets_by_gene; gene: TP53]*\n\n"
            "*[Using tool: get_gene_expression; gene: TP53]*"
        )
        assert parse_tool_calls(content) == [
            "search_genes", "get_credible_sets_by_gene", "get_gene_expression"
        ]

    def test_ellipsis_marker(self):
        content = "*[Using tool: get_phenotype_report...]*"
        assert parse_tool_calls(content) == ["get_phenotype_report"]

    def test_no_tools(self):
        content = "Here is some plain text with no tool calls."
        assert parse_tool_calls(content) == []

    def test_tool_in_surrounding_text(self):
        content = "Let me search.\n\n*[Using tool: web_search; query: genetics]*\n\nFound results."
        assert parse_tool_calls(content) == ["web_search"]


def _blocks(*blocks) -> str:
    return json.dumps(list(blocks))


class TestParseToolCallsFromContentJson:
    def test_tool_use_blocks(self):
        content_json = _blocks(
            {"type": "text", "text": "let me look"},
            {"type": "tool_use", "id": "t1", "name": "search_genes", "input": {"q": "TP53"}},
            {"type": "tool_use", "id": "t2", "name": "get_gene_expression", "input": {}},
        )
        assert parse_tool_calls_from_content_json(content_json) == [
            "search_genes", "get_gene_expression"
        ]

    def test_ignores_non_tool_use_blocks(self):
        content_json = _blocks(
            {"type": "thinking", "thinking": "hmm"},
            {"type": "text", "text": "hello"},
            {"type": "tool_result", "tool_use_id": "t1", "content": "..."},
        )
        assert parse_tool_calls_from_content_json(content_json) == []

    def test_recorded_but_toolless_is_empty_not_none(self):
        # [] means "recorded, no tools"; None means "nothing recorded" -> fallback
        assert parse_tool_calls_from_content_json("[]") == []

    def test_missing_returns_none(self):
        assert parse_tool_calls_from_content_json(None) is None
        assert parse_tool_calls_from_content_json("") is None

    def test_malformed_json_returns_none(self):
        assert parse_tool_calls_from_content_json("{not json") is None

    def test_non_list_returns_none(self):
        assert parse_tool_calls_from_content_json('{"type": "tool_use"}') is None

    def test_junk_blocks_are_skipped_without_raising(self):
        content_json = _blocks(
            "a bare string",
            None,
            {"type": "tool_use"},               # no name
            {"type": "tool_use", "name": ""},   # empty name
            {"type": "tool_use", "name": 7},    # wrong type
            {"type": "tool_use", "name": "ok_tool"},
        )
        assert parse_tool_calls_from_content_json(content_json) == ["ok_tool"]


class TestMessageToolCalls:
    def test_content_json_wins_over_markers(self):
        row = {
            "content": "*[Using tool: stale_marker; q: x]*",
            "content_json": _blocks({"type": "tool_use", "name": "real_tool"}),
        }
        assert message_tool_calls(row) == (["real_tool"], True)

    def test_recorded_toolless_row_does_not_fall_back(self):
        row = {
            "content": "*[Using tool: imitated_marker]*",
            "content_json": _blocks({"type": "text", "text": "hi"}),
        }
        assert message_tool_calls(row) == ([], True)

    def test_falls_back_to_markers_without_content_json(self):
        row = {"content": "*[Using tool: legacy_tool; q: x]*", "content_json": None}
        assert message_tool_calls(row) == (["legacy_tool"], False)

    def test_fallback_is_not_authoritative_even_when_empty(self):
        row = {"content": "plain prose", "content_json": None}
        assert message_tool_calls(row) == ([], False)


class TestToolCoverage:
    @staticmethod
    def _messages(rows):
        return pl.DataFrame(rows, schema={
            "session_id": pl.Utf8, "role": pl.Utf8,
            "content": pl.Utf8, "content_json": pl.Utf8,
        })

    def test_counts_split_by_source(self):
        messages = self._messages([
            {"session_id": "a", "role": "user", "content": "q", "content_json": None},
            {"session_id": "a", "role": "assistant", "content": "x",
             "content_json": _blocks({"type": "tool_use", "name": "t1"},
                                     {"type": "tool_use", "name": "t2"})},
            {"session_id": "b", "role": "assistant",
             "content": "*[Using tool: legacy]*", "content_json": None},
        ])
        frame, coverage = build_tool_coverage(messages)

        assert coverage.assistant_messages == 2
        assert coverage.messages_from_content_json == 1
        assert coverage.messages_from_markers == 1
        assert coverage.tool_calls_from_content_json == 2
        assert coverage.tool_calls_from_markers == 1
        assert coverage.total_tool_calls == 3
        assert coverage.sessions == 2
        assert coverage.fully_covered_sessions == 1
        assert coverage.is_exact is False

        flags = dict(zip(frame["session_id"], frame["tool_count_is_lower_bound"], strict=True))
        assert flags == {"a": False, "b": True}

    def test_partially_covered_session_is_a_lower_bound(self):
        messages = self._messages([
            {"session_id": "a", "role": "assistant", "content": "x",
             "content_json": _blocks({"type": "tool_use", "name": "t1"})},
            {"session_id": "a", "role": "assistant",
             "content": "*[Using tool: legacy]*", "content_json": None},
        ])
        frame, coverage = build_tool_coverage(messages)
        assert coverage.fully_covered_sessions == 0
        assert frame["tool_count_is_lower_bound"].item() is True

    def test_exact_when_everything_has_content_json(self):
        messages = self._messages([
            {"session_id": "a", "role": "assistant", "content": "x", "content_json": "[]"},
        ])
        _, coverage = build_tool_coverage(messages)
        assert coverage.is_exact is True
        assert "exact" in " ".join(coverage.summary_lines())

    def test_empty_frame(self):
        frame, coverage = build_tool_coverage(self._messages([]))
        assert frame.height == 0
        assert coverage.sessions == 0
        assert coverage.summary_lines()  # must not divide by zero

    def test_empty_frame_is_not_exact(self):
        # nothing was counted, so there is nothing to have counted exactly; claiming
        # "all tool counts are exact" over zero data is a vacuous truth in a report
        _, coverage = build_tool_coverage(self._messages([]))
        assert coverage.is_exact is False
        assert any("no assistant messages" in line.lower() for line in coverage.summary_lines())
        assert not any("**exact**" in line for line in coverage.summary_lines())

    def test_session_stats_prefer_content_json(self, sample_db):
        # s1's markers claim 3 calls; content_json on one message overrides that message
        conn = sqlite3.connect(sample_db)
        conn.execute(
            "UPDATE chat_messages SET content_json = ? WHERE id = 'm2'",
            (_blocks({"type": "tool_use", "name": "real_a"},
                     {"type": "tool_use", "name": "real_b"},
                     {"type": "tool_use", "name": "real_c"}),),
        )
        conn.commit()
        conn.close()

        _, messages = load_data(sample_db)
        stats = build_session_tool_stats(messages)
        s1 = stats.filter(pl.col("session_id") == "s1")
        assert s1["total_tool_calls"].item() == 4  # 3 real from m2 + 1 marker from m4
        assert "real_a" in s1["tool_sequence"].item()


class TestToolCoverageInMetricsAndReport:
    def test_lower_bound_flag_and_report_caveat(self, sample_db):
        sessions, messages = load_data(sample_db)
        tool_stats = build_session_tool_stats(messages)
        metrics = compute_all_metrics(sessions, messages, tool_stats, {})

        # the fixture has no content_json anywhere, so every session is marker-counted
        assert all(m.tool_count_is_lower_bound for m in metrics)

        report = generate_report(metrics, sessions, messages, tool_stats)
        assert "Counting coverage" in report
        assert "LOWER BOUND" in report
        assert "lower bound" in report

    def test_report_says_exact_when_fully_covered(self, sample_db):
        conn = sqlite3.connect(sample_db)
        conn.execute(
            "UPDATE chat_messages SET content_json = ? WHERE role = 'assistant'",
            (_blocks({"type": "tool_use", "name": "real_tool"}),),
        )
        conn.commit()
        conn.close()

        sessions, messages = load_data(sample_db)
        tool_stats = build_session_tool_stats(messages)
        metrics = compute_all_metrics(sessions, messages, tool_stats, {})
        assert not any(m.tool_count_is_lower_bound for m in metrics)

        report = generate_report(metrics, sessions, messages, tool_stats)
        assert "LOWER BOUND" not in report
        assert "exact" in report


# ---------------------------------------------------------------------------
# Keyword categorization tests
# ---------------------------------------------------------------------------

class TestKeywordCategorization:
    def test_gene_query(self):
        topic, conf = categorize_by_keywords("What do we know about gene BRCA1?")
        assert topic == "gene_lookup"
        assert conf > 0

    def test_variant_query(self):
        topic, _ = categorize_by_keywords("What does variant 1:12345:A:G do?")
        assert topic == "variant_interpretation"

    def test_variant_rsid(self):
        topic, _ = categorize_by_keywords("Tell me about rs12345")
        assert topic == "variant_interpretation"

    def test_phenotype_query(self):
        topic, _ = categorize_by_keywords("Show me GWAS associations for diabetes")
        assert topic == "phenotype_exploration"

    def test_literature_query(self):
        topic, _ = categorize_by_keywords("Find papers about PCSK9 in PubMed")
        assert topic == "literature_search"

    def test_clinical_query(self):
        topic, _ = categorize_by_keywords(
            "Patient has heterozygous frameshift in SLC9B1"
        )
        assert topic == "clinical_genetics"

    def test_general_fallback(self):
        topic, conf = categorize_by_keywords("Hello how are you?")
        assert topic == "general_genetics"
        assert conf == 0.3

    def test_data_source_question(self):
        topic, _ = categorize_by_keywords("What are your sources of data?")
        assert topic == "data_source_question"


# ---------------------------------------------------------------------------
# Success metrics tests
# ---------------------------------------------------------------------------

class TestSuccessMetrics:
    def test_rated_session_uses_rating(self):
        m = ConversationMetrics(session_id="s1", user_rating=5)
        score = compute_success_score(m)
        assert score == 1.0

    def test_rated_session_low_rating(self):
        m = ConversationMetrics(session_id="s1", user_rating=1)
        score = compute_success_score(m)
        assert score == 0.0

    def test_unrated_baseline(self):
        m = ConversationMetrics(session_id="s1", user_messages=2, total_tool_calls=3)
        score = compute_success_score(m)
        assert score == 0.5

    def test_error_penalty(self):
        m = ConversationMetrics(
            session_id="s1", user_messages=2, total_tool_calls=3, has_error_response=True,
        )
        score = compute_success_score(m)
        assert score < 0.5

    def test_excessive_tools_penalty(self):
        m = ConversationMetrics(
            session_id="s1", user_messages=1, total_tool_calls=15,
        )
        score = compute_success_score(m)
        assert score < 0.5

    def test_multi_turn_bonus(self):
        m = ConversationMetrics(
            session_id="s1", user_messages=5, total_tool_calls=5,
        )
        score = compute_success_score(m)
        assert score > 0.5

    def test_abandoned_session_penalty(self):
        m = ConversationMetrics(
            session_id="s1", user_messages=1, assistant_messages=1,
            total_tool_calls=0,
        )
        score = compute_success_score(m)
        assert score < 0.5

    def test_thumbs_up_boost(self):
        m = ConversationMetrics(
            session_id="s1", user_messages=2, total_tool_calls=3,
            thumbs_up_count=2, thumbs_down_count=0,
        )
        score = compute_success_score(m)
        assert score > 0.5

    def test_label_successful(self):
        assert label_success(0.8) == "successful"

    def test_label_neutral(self):
        assert label_success(0.5) == "neutral"

    def test_label_unsuccessful(self):
        assert label_success(0.2) == "unsuccessful"


# ---------------------------------------------------------------------------
# Data loading tests
# ---------------------------------------------------------------------------

class TestLoadData:
    def test_load_sessions_and_messages(self, sample_db):
        sessions, messages = load_data(sample_db)
        assert sessions.height == 3
        assert messages.height == 8
        assert "id" in sessions.columns
        assert "session_id" in messages.columns

    def test_rating_column_type(self, sample_db):
        sessions, _ = load_data(sample_db)
        assert sessions.schema["rating"] == pl.Int64

    def test_thumbs_up_column_type(self, sample_db):
        _, messages = load_data(sample_db)
        assert messages.schema["thumbs_up"] == pl.Boolean


# ---------------------------------------------------------------------------
# Tool stats tests
# ---------------------------------------------------------------------------

class TestBuildSessionToolStats:
    def test_tool_stats(self, sample_db):
        _, messages = load_data(sample_db)
        stats = build_session_tool_stats(messages)
        assert stats.height > 0
        assert "total_tool_calls" in stats.columns

        s1_stats = stats.filter(pl.col("session_id") == "s1")
        assert s1_stats.height == 1
        assert s1_stats["total_tool_calls"].item() == 3  # search_genes, get_credible_sets, get_gene_expression
        # m2 calls 2 tools, m4 calls 1 -> peak in a single message is 2
        assert "max_tools_in_message" in stats.columns
        assert s1_stats["max_tools_in_message"].item() == 2

    def test_empty_messages(self):
        messages = pl.DataFrame({
            "session_id": pl.Series([], dtype=pl.Utf8),
            "role": pl.Series([], dtype=pl.Utf8),
            "content": pl.Series([], dtype=pl.Utf8),
        })
        stats = build_session_tool_stats(messages)
        assert stats.height == 0


# ---------------------------------------------------------------------------
# Integration: compute_all_metrics
# ---------------------------------------------------------------------------

class TestComputeAllMetrics:
    def test_all_metrics(self, sample_db):
        sessions, messages = load_data(sample_db)
        tool_stats = build_session_tool_stats(messages)

        # keyword categorization
        first_msgs = (
            messages.filter(pl.col("role") == "user")
            .sort("created_at")
            .group_by("session_id").first()
        )
        topics = {}
        for row in first_msgs.iter_rows(named=True):
            topic, _ = categorize_by_keywords(row["content"])
            topics[row["session_id"]] = {"topic": topic, "complexity": 2, "brief_reason": "test"}

        metrics = compute_all_metrics(sessions, messages, tool_stats, topics)
        assert len(metrics) == 3

        # s1 has rating=5
        s1 = next(m for m in metrics if m.session_id == "s1")
        assert s1.user_rating == 5
        assert s1.success_score == 1.0
        assert s1.total_tool_calls == 3

        # s3 has rating=1 and error
        s3 = next(m for m in metrics if m.session_id == "s3")
        assert s3.user_rating == 1
        assert s3.success_score == 0.0


# ---------------------------------------------------------------------------
# Report generation test
# ---------------------------------------------------------------------------

class TestGenerateReport:
    def test_report_contains_sections(self, sample_db):
        sessions, messages = load_data(sample_db)
        tool_stats = build_session_tool_stats(messages)
        topics = {
            "s1": {"topic": "gene_lookup", "complexity": 1, "brief_reason": ""},
            "s2": {"topic": "variant_interpretation", "complexity": 1, "brief_reason": ""},
            "s3": {"topic": "literature_search", "complexity": 1, "brief_reason": ""},
        }
        metrics = compute_all_metrics(sessions, messages, tool_stats, topics)
        report = generate_report(metrics, sessions, messages, tool_stats)

        assert "# Conversation Analysis Report" in report
        assert "## Overview" in report
        assert "## Topic Distribution" in report
        assert "## Tool Usage Patterns" in report
        assert "Total sessions**: 3" in report

    def test_issue_categories_aggregated(self, sample_db):
        sessions, messages = load_data(sample_db)
        tool_stats = build_session_tool_stats(messages)
        topics = {
            "s1": {"topic": "gene_lookup", "complexity": 1, "brief_reason": ""},
            "s2": {"topic": "variant_interpretation", "complexity": 1, "brief_reason": ""},
            "s3": {"topic": "literature_search", "complexity": 1, "brief_reason": ""},
        }
        metrics = compute_all_metrics(sessions, messages, tool_stats, topics)
        # give two conversations distinct raw issues that share a category
        metrics[0].llm_quality_score = 2
        metrics[0].llm_issues = ["did not query exome data despite it being available"]
        metrics[1].llm_quality_score = 3
        metrics[1].llm_issues = ["ignored the available eQTL dataset for this gene"]
        issue_categories = {
            "did not query exome data despite it being available": "missed_data_source",
            "ignored the available eQTL dataset for this gene": "missed_data_source",
        }
        report = generate_report(metrics, sessions, messages, tool_stats,
                                 issue_categories=issue_categories)

        # both distinct raw issues collapse to one category with count 2
        assert "Most common issue categories" in report
        assert "| missed_data_source | 2 |" in report

    def test_raw_issues_fallback_without_categories(self, sample_db):
        sessions, messages = load_data(sample_db)
        tool_stats = build_session_tool_stats(messages)
        topics = {
            "s1": {"topic": "gene_lookup", "complexity": 1, "brief_reason": ""},
            "s2": {"topic": "variant_interpretation", "complexity": 1, "brief_reason": ""},
            "s3": {"topic": "literature_search", "complexity": 1, "brief_reason": ""},
        }
        metrics = compute_all_metrics(sessions, messages, tool_stats, topics)
        metrics[0].llm_quality_score = 2
        metrics[0].llm_issues = ["some specific issue"]
        # no issue_categories passed -> falls back to raw issue frequency
        report = generate_report(metrics, sessions, messages, tool_stats)
        assert "Most common issues" in report
        assert "some specific issue" in report


# ---------------------------------------------------------------------------
# Eval export test
# ---------------------------------------------------------------------------

class TestExportEvalDataset:
    def test_export_creates_files(self, sample_db, tmp_path):
        sessions, messages = load_data(sample_db)
        tool_stats = build_session_tool_stats(messages)
        topics = {
            "s1": {"topic": "gene_lookup", "complexity": 1, "brief_reason": ""},
            "s2": {"topic": "variant_interpretation", "complexity": 1, "brief_reason": ""},
            "s3": {"topic": "literature_search", "complexity": 1, "brief_reason": ""},
        }
        metrics = compute_all_metrics(sessions, messages, tool_stats, topics)

        output_dir = tmp_path / "eval_output"
        export_eval_dataset(metrics, messages, output_dir)

        assert (output_dir / "eval_dataset.json").exists()
        assert (output_dir / "transcripts").is_dir()

        with open(output_dir / "eval_dataset.json") as f:
            data = json.load(f)
        assert len(data) > 0
        assert "session_id" in data[0]
        assert "topic" in data[0]
        assert "turns" in data[0]
        # pre-existing keys must survive for any consumer of the old shape
        for key in ("first_user_message", "tools_used", "total_tool_calls", "turn_count"):
            assert key in data[0]


@pytest.fixture
def replay_db(tmp_path):
    """A DB with per-turn options, tied timestamps, and multi-turn sessions."""
    db_path = str(tmp_path / "replay_chat.db")
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE chat_sessions (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT,
            created_at TIMESTAMP,
            updated_at TIMESTAMP,
            rating INTEGER,
            comment TEXT,
            phenotype_code TEXT
        );
        CREATE TABLE chat_messages (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP,
            thumbs_up BOOLEAN,
            content_json TEXT,
            literature_backend TEXT,
            tool_profile TEXT,
            instruction_set_id TEXT,
            verbosity TEXT
        );

        INSERT INTO chat_sessions VALUES
            ('r1', 'user1@test.com', 'Multi turn', '2025-12-10', '2025-12-10', NULL, NULL, NULL);

        -- every row shares one timestamp: created_at has one-second resolution, so only
        -- the rowid tiebreak keeps this order deterministic.
        -- the client sends all four option keys on every save, and a null tool_profile is
        -- the real value for "all tools" — it is never persisted as the string 'all'
        INSERT INTO chat_messages VALUES
            ('q1', 'r1', 'user', 'first question', '2025-12-10 10:00:00', NULL, NULL,
             'europepmc', NULL, 'iset-1', 'brief'),
            ('a1', 'r1', 'assistant', 'first answer', '2025-12-10 10:00:00', NULL, NULL,
             'europepmc', NULL, 'iset-1', 'brief'),
            ('q2', 'r1', 'user', 'second question', '2025-12-10 10:00:00', NULL, NULL,
             'europepmc', NULL, 'iset-1', 'detailed'),
            ('a2', 'r1', 'assistant', 'second answer', '2025-12-10 10:00:00', NULL, NULL,
             'europepmc', 'bigquery', 'iset-1', 'detailed'),
            ('q3', 'r1', 'user', 'third question', '2025-12-10 10:00:00', NULL, NULL,
             'europepmc', NULL, 'iset-2', 'detailed');
    """)
    conn.close()
    return db_path


class TestUserTurnExport:
    def _case(self, db_path, tmp_path):
        sessions, messages = load_data(db_path)
        tool_stats = build_session_tool_stats(messages)
        metrics = compute_all_metrics(sessions, messages, tool_stats, {})
        output_dir = tmp_path / "replay_output"
        export_eval_dataset(metrics, messages, output_dir)
        with open(output_dir / "eval_dataset.json") as f:
            return json.load(f)[0]

    def test_all_user_turns_exported_in_order(self, replay_db, tmp_path):
        case = self._case(replay_db, tmp_path)
        assert case["user_turn_count"] == 3
        assert [t["content"] for t in case["user_turns"]] == [
            "first question", "second question", "third question"
        ]
        assert [t["index"] for t in case["user_turns"]] == [0, 1, 2]
        assert [t["message_id"] for t in case["user_turns"]] == ["q1", "q2", "q3"]

    def test_options_in_force_are_each_rows_own_values(self, replay_db, tmp_path):
        case = self._case(replay_db, tmp_path)
        opts = [t["options"] for t in case["user_turns"]]

        assert opts[0] == {
            "verbosity": "brief", "tool_profile": None,
            "instruction_set_id": "iset-1", "literature_backend": "europepmc",
        }
        assert opts[1] == {
            "verbosity": "detailed", "tool_profile": None,
            "instruction_set_id": "iset-1", "literature_backend": "europepmc",
        }
        # a2 ran under tool_profile='bigquery'; q3 switched back to all tools, which the
        # client persists as NULL. carrying the last non-null value forward would export
        # q3 as still running 'bigquery' and a replay would reproduce the wrong conditions.
        assert opts[2] == {
            "verbosity": "detailed", "tool_profile": None,
            "instruction_set_id": "iset-2", "literature_backend": "europepmc",
        }

    def test_options_carry_forward_only_on_fully_null_rows(self):
        # a row written by the current client always carries at least verbosity and
        # literature_backend, so an all-null row predates that wiring and inherits
        messages = pl.DataFrame({
            "id": ["q1", "a1", "q2"],
            "role": ["user", "assistant", "user"],
            "content": ["one", "answer", "two"],
            "created_at": ["2025-12-10 10:00:00"] * 3,
            "verbosity": ["brief", None, None],
            "tool_profile": ["api", None, None],
            "instruction_set_id": ["iset-1", None, None],
            "literature_backend": ["perplexity", None, None],
        })
        turns = _session_user_turns(messages)

        assert turns[1]["options"] == {
            "verbosity": "brief", "tool_profile": "api",
            "instruction_set_id": "iset-1", "literature_backend": "perplexity",
        }

    def test_user_turn_content_is_not_truncated(self, replay_db, tmp_path):
        long_text = "x" * 5000
        conn = sqlite3.connect(replay_db)
        conn.execute("UPDATE chat_messages SET content = ? WHERE id = 'q1'", (long_text,))
        conn.commit()
        conn.close()

        case = self._case(replay_db, tmp_path)
        assert case["user_turns"][0]["content"] == long_text
        # the display transcript stays capped
        assert len(case["turns"][0]["content"]) == 2000

    def test_sort_keys_use_rowid_when_available(self, replay_db):
        _, messages = load_data(replay_db)
        assert message_sort_keys(messages) == ["created_at", "_rowid"]
        assert message_sort_keys(messages.drop("_rowid")) == ["created_at"]

    def test_missing_option_columns_degrade_to_none(self, sample_db):
        # sample_db predates the verbosity / instruction_set_id migrations
        _, messages = load_data(sample_db)
        session_msgs = messages.filter(pl.col("session_id") == "s2").sort(
            message_sort_keys(messages)
        )
        turns = _session_user_turns(session_msgs)
        assert len(turns) == 1
        assert turns[0]["options"] == {
            "verbosity": None, "tool_profile": "api",
            "instruction_set_id": None, "literature_backend": None,
        }


# ---------------------------------------------------------------------------
# LLM categorization tests (mocked)
# ---------------------------------------------------------------------------

class TestLLMCategorization:
    @pytest.mark.asyncio
    async def test_successful_categorization(self):
        mock_response = mock_llm_response([
            {"id": "s1", "topic": "gene_lookup", "complexity": 1, "brief_reason": "gene query"},
            {"id": "s2", "topic": "variant_interpretation", "complexity": 2, "brief_reason": "variant"},
        ])

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch.dict("sys.modules", {"anthropic": MagicMock(AsyncAnthropic=lambda: mock_client)}):
            result = await categorize_with_llm([
                {"id": "s1", "text": "What about BRCA1?"},
                {"id": "s2", "text": "What does 1:123:A:G do?"},
            ])

        assert result["s1"]["topic"] == "gene_lookup"
        assert result["s2"]["topic"] == "variant_interpretation"
        # thinking is off: these calls only need a JSON object back
        assert mock_client.messages.create.await_args.kwargs["thinking"] == {"type": "disabled"}

    @pytest.mark.asyncio
    async def test_leading_thinking_block_is_skipped(self):
        """A ThinkingBlock ahead of the answer must not break parsing."""
        mock_response = mock_llm_response(
            [{"id": "s1", "topic": "gene_lookup", "complexity": 1, "brief_reason": "gene query"}],
            leading_thinking=True,
        )
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch.dict("sys.modules", {"anthropic": MagicMock(AsyncAnthropic=lambda: mock_client)}):
            result = await categorize_with_llm([{"id": "s1", "text": "What about BRCA1?"}])

        assert result["s1"]["topic"] == "gene_lookup"

    @pytest.mark.asyncio
    async def test_fallback_on_api_error(self):
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(side_effect=Exception("API error"))

        with patch.dict("sys.modules", {"anthropic": MagicMock(AsyncAnthropic=lambda: mock_client)}):
            result = await categorize_with_llm([
                {"id": "s1", "text": "What about gene BRCA1?"},
            ])

        assert "s1" in result
        # should fall back to keyword categorization
        assert result["s1"]["brief_reason"] == "keyword fallback"


# ---------------------------------------------------------------------------
# Issue categorization tests
# ---------------------------------------------------------------------------

class TestIssueCategorization:
    @pytest.mark.asyncio
    async def test_maps_issues_to_categories(self):
        mock_response = mock_llm_response([
            {"id": 0, "category": "missed_data_source"},
            {"id": 1, "category": "inefficient_tool_use"},
        ])
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch.dict("sys.modules", {"anthropic": MagicMock(AsyncAnthropic=lambda: mock_client)}):
            result = await categorize_issues_with_llm(
                ["did not query exome data", "made 12 redundant tool calls"],
            )

        assert result["did not query exome data"] == "missed_data_source"
        assert result["made 12 redundant tool calls"] == "inefficient_tool_use"

    @pytest.mark.asyncio
    async def test_unknown_category_coerced_to_other(self):
        mock_response = mock_llm_response([
            {"id": 0, "category": "totally_made_up"},
        ])
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch.dict("sys.modules", {"anthropic": MagicMock(AsyncAnthropic=lambda: mock_client)}):
            result = await categorize_issues_with_llm(["some issue"])

        assert result["some issue"] == "other"

    @pytest.mark.asyncio
    async def test_api_error_falls_back_to_other(self):
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(side_effect=Exception("API error"))

        with patch.dict("sys.modules", {"anthropic": MagicMock(AsyncAnthropic=lambda: mock_client)}):
            result = await categorize_issues_with_llm(["issue a", "issue b"])

        # every input still gets a category even when the call fails
        assert result == {"issue a": "other", "issue b": "other"}


# ---------------------------------------------------------------------------
# Quality evaluation tests
# ---------------------------------------------------------------------------

class TestQualityEvaluation:
    def test_format_conversation(self, sample_db):
        _, messages = load_data(sample_db)
        text = _format_conversation_for_eval("s1", messages)
        assert "[USER]" in text
        assert "[ASSISTANT]" in text
        assert "BRCA1" in text

    def test_format_truncates_long_messages(self, sample_db):
        _, messages = load_data(sample_db)
        text = _format_conversation_for_eval("s1", messages, max_chars=100)
        assert "truncated" in text

    def test_attachment_note_from_content_json(self):
        cj = json.dumps({"attachments": [
            {"id": "x", "name": "PANSINUSITIS.tsv", "size": 67386, "type": "tsv"},
        ]})
        note = _attachment_note(cj)
        assert "PANSINUSITIS.tsv" in note
        assert "tsv" in note
        assert "67386 bytes" in note

    def test_attachment_note_empty_cases(self):
        assert _attachment_note(None) == ""
        assert _attachment_note("") == ""
        assert _attachment_note("not json") == ""
        assert _attachment_note(json.dumps({"attachments": []})) == ""
        assert _attachment_note(json.dumps({"other": 1})) == ""

    def test_format_includes_attachment_note(self):
        # a user turn with an attachment recorded only in content_json
        messages = pl.DataFrame({
            "session_id": ["s9", "s9"],
            "role": ["user", "assistant"],
            "content": ["what do you think about the results?", "Here is the analysis."],
            "created_at": ["2026-01-01 10:00:00", "2026-01-01 10:00:01"],
            "content_json": [
                json.dumps({"attachments": [
                    {"name": "results.tsv", "size": 1000, "type": "tsv"}]}),
                None,
            ],
        })
        text = _format_conversation_for_eval("s9", messages)
        assert "User attached file(s): results.tsv" in text

    def test_format_orders_same_second_turns_by_rowid(self):
        # created_at has one-second resolution, so a user turn and its own reply routinely
        # share a timestamp; only rowid separates them. Rows are handed over reply-first so
        # a sort on created_at alone would leave the answer above the question.
        messages = pl.DataFrame({
            "session_id": ["s10", "s10"],
            "role": ["assistant", "user"],
            "content": ["BRCA1 is on chromosome 17.", "where is BRCA1?"],
            "created_at": ["2026-01-01 10:00:00", "2026-01-01 10:00:00"],
            "content_json": [None, None],
            "_rowid": [2, 1],
        })
        # the guard in message_sort_keys must not have fallen back to the single key
        assert message_sort_keys(messages) == ["created_at", "_rowid"]

        text = _format_conversation_for_eval("s10", messages)
        assert text.index("where is BRCA1?") < text.index("BRCA1 is on chromosome 17.")
        assert text.index("[USER]") < text.index("[ASSISTANT]")

    def test_apply_quality_assessments(self):
        metrics = [
            ConversationMetrics(session_id="s1", user_messages=2, total_tool_calls=3),
            ConversationMetrics(session_id="s2", user_messages=1, total_tool_calls=1),
        ]
        assessments = {
            "s1": {
                "quality_score": 4,
                "answered": "yes",
                "accurate": "yes",
                "efficient": "mostly",
                "concluded": "yes",
                "issues": [],
            },
        }
        apply_quality_assessments(metrics, assessments)

        assert metrics[0].llm_quality_score == 4
        # (4-1)/4 = 0.75 base, +0.05 for answered="yes"
        assert metrics[0].success_score == 0.8
        assert metrics[0].success_label == "successful"
        # s2 should be unchanged (no assessment)
        assert metrics[1].llm_quality_score is None

    def test_llm_score_overrides_heuristic(self):
        m = ConversationMetrics(
            session_id="s1", user_messages=1, total_tool_calls=0,
            assistant_messages=1, llm_quality_score=5,
        )
        # without LLM score this would be penalized as abandoned
        score = compute_success_score(m)
        assert score == 1.0

    def test_binary_flags_adjust_quality_score(self):
        # answered=no and concluded=no pull a middling quality_score down
        m = ConversationMetrics(
            session_id="s1", llm_quality_score=3,
            llm_answered="no", llm_concluded="no",
        )
        # (3-1)/4 = 0.5 base, -0.1 (answered=no) -0.05 (concluded=no)
        assert compute_success_score(m) == 0.35
        # answered=yes nudges it up
        m2 = ConversationMetrics(
            session_id="s2", llm_quality_score=3, llm_answered="yes",
        )
        assert compute_success_score(m2) == 0.55

    def test_user_rating_still_takes_priority(self):
        m = ConversationMetrics(
            session_id="s1", user_rating=1, llm_quality_score=5,
        )
        score = compute_success_score(m)
        assert score == 0.0  # user rating wins


class TestDisposition:
    def test_label_from_disposition_non_quality(self):
        # non-quality dispositions keep their own label regardless of score
        for disp in ["technical_failure", "out_of_scope", "unfinished", "weird_or_unclear"]:
            assert label_from_disposition(disp, 0.1) == disp
            assert label_from_disposition(disp, 0.9) == disp

    def test_label_from_disposition_quality(self):
        # good_answer / agent_failure (and empty fallback) collapse to score-based
        assert label_from_disposition("good_answer", 0.9) == "successful"
        assert label_from_disposition("agent_failure", 0.1) == "unsuccessful"
        assert label_from_disposition("", 0.5) == "neutral"

    def test_out_of_scope_not_unsuccessful_despite_low_score(self):
        # a low-scored but out-of-scope conversation must NOT be labelled unsuccessful
        metrics = [ConversationMetrics(session_id="s1", user_messages=1)]
        apply_quality_assessments(metrics, {
            "s1": {"quality_score": 1, "answered": "no", "disposition": "out_of_scope",
                   "issues": []},
        })
        assert metrics[0].llm_disposition == "out_of_scope"
        assert metrics[0].success_label == "out_of_scope"
        assert metrics[0].success_label not in {"successful", "neutral", "unsuccessful"}

    def test_technical_failure_labelled_separately(self):
        metrics = [ConversationMetrics(session_id="s1", user_messages=1)]
        apply_quality_assessments(metrics, {
            "s1": {"quality_score": 1, "answered": "no",
                   "disposition": "technical_failure", "issues": []},
        })
        assert metrics[0].success_label == "technical_failure"

    def test_agent_failure_still_unsuccessful(self):
        metrics = [ConversationMetrics(session_id="s1", user_messages=1)]
        apply_quality_assessments(metrics, {
            "s1": {"quality_score": 1, "answered": "no",
                   "disposition": "agent_failure", "issues": ["gave up"]},
        })
        assert metrics[0].success_label == "unsuccessful"

    def test_mark_unscored_unknown(self):
        metrics = [
            # judged -> keeps its disposition-derived label
            ConversationMetrics(session_id="s1", llm_quality_score=5,
                                success_label="successful"),
            # no LLM score, no rating -> unknown
            ConversationMetrics(session_id="s2", success_label="neutral"),
            # no LLM score but has a user rating -> rating is a real signal, keep it
            ConversationMetrics(session_id="s3", user_rating=4,
                                success_label="successful"),
        ]
        n = mark_unscored_unknown(metrics)
        assert n == 1
        assert metrics[0].success_label == "successful"
        assert metrics[1].success_label == "unknown"
        assert metrics[2].success_label == "successful"

    @pytest.mark.asyncio
    async def test_evaluate_quality_with_llm(self):
        mock_response = mock_llm_response({
            "answered": "yes", "accurate": "yes", "efficient": "yes",
            "concluded": "yes", "quality_score": 5, "issues": [],
        })

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        _, messages = load_data.__wrapped__("nonexistent") if hasattr(load_data, "__wrapped__") else (None, None)
        # build minimal messages DataFrame
        messages = pl.DataFrame({
            "id": ["m1", "m2"],
            "session_id": ["s1", "s1"],
            "role": ["user", "assistant"],
            "content": ["What about BRCA1?", "BRCA1 is a tumor suppressor gene."],
            "created_at": ["2025-01-01", "2025-01-01"],
            "thumbs_up": [None, None],
            "content_json": [None, None],
            "literature_backend": [None, None],
            "tool_profile": [None, None],
        })

        with patch.dict("sys.modules", {"anthropic": MagicMock(AsyncAnthropic=lambda: mock_client)}):
            result = await evaluate_quality_with_llm(["s1"], messages)

        assert "s1" in result
        assert result["s1"]["quality_score"] == 5
        assert mock_client.messages.create.await_args.kwargs["thinking"] == {"type": "disabled"}

    @pytest.mark.asyncio
    async def test_evaluate_quality_skips_leading_thinking_block(self):
        """Regression: judging a conversation on a thinking-capable model."""
        mock_response = mock_llm_response({
            "answered": "yes", "accurate": "yes", "efficient": "yes",
            "concluded": "yes", "quality_score": 4, "issues": [],
        }, leading_thinking=True)

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        messages = pl.DataFrame({
            "id": ["m1", "m2"],
            "session_id": ["s1", "s1"],
            "role": ["user", "assistant"],
            "content": ["What about BRCA1?", "BRCA1 is a tumor suppressor gene."],
            "created_at": ["2025-01-01", "2025-01-01"],
            "thumbs_up": [None, None],
            "content_json": [None, None],
            "literature_backend": [None, None],
            "tool_profile": [None, None],
        })

        with patch.dict("sys.modules", {"anthropic": MagicMock(AsyncAnthropic=lambda: mock_client)}):
            result = await evaluate_quality_with_llm(["s1"], messages)

        assert result["s1"]["quality_score"] == 4


# ---------------------------------------------------------------------------
# Date filter tests
# ---------------------------------------------------------------------------

class TestDateFilter:
    def test_start_from_filters_sessions(self, sample_db):
        sessions, messages = load_data(sample_db)
        # filter to only sessions from 2025-12-11 onwards
        filtered_sessions = sessions.filter(pl.col("created_at") >= "2025-12-11")
        session_ids = filtered_sessions.select("id").to_series().to_list()
        filtered_messages = messages.filter(pl.col("session_id").is_in(session_ids))

        assert filtered_sessions.height == 2  # s2 and s3
        assert filtered_messages.height == 4  # m5-m8


# ---------------------------------------------------------------------------
# Cached-row reconstruction (pure helper)
# ---------------------------------------------------------------------------

class TestCachedTopicAndQuality:
    def test_reconstructs_only_current_version_rows(self):
        m_current = ConversationMetrics(
            session_id="s1", topic="gene_lookup", complexity=3, topic_reason="gene",
            llm_quality_score=4, llm_answered="yes", llm_disposition="good_answer",
            llm_issues=["x"], llm_issue_categories=["other"],
        )
        analysis_map = {
            "s1": {
                "analyzer_version": ac.ANALYZER_VERSION,
                "metrics_json": json.dumps(vars(m_current)),
            },
            # stale version: must be treated as not cached
            "s2": {
                "analyzer_version": ac.ANALYZER_VERSION + 1,
                "metrics_json": json.dumps({"topic": "x", "llm_quality_score": 2}),
            },
        }
        topics, quality, issue_cats = cached_topic_and_quality(analysis_map)

        assert set(topics) == {"s1"}
        assert topics["s1"]["topic"] == "gene_lookup"
        assert topics["s1"]["complexity"] == 3
        assert quality["s1"]["quality_score"] == 4
        assert quality["s1"]["disposition"] == "good_answer"
        assert issue_cats["s1"] == ["other"]
        # stale-version row excluded entirely
        assert "s2" not in topics and "s2" not in quality

    def test_unjudged_session_has_no_quality_entry(self):
        m = ConversationMetrics(session_id="s1", topic="gene_lookup")  # no llm score
        analysis_map = {
            "s1": {
                "analyzer_version": ac.ANALYZER_VERSION,
                "metrics_json": json.dumps(vars(m)),
            },
        }
        topics, quality, _ = cached_topic_and_quality(analysis_map)
        assert "s1" in topics
        assert "s1" not in quality  # must not fabricate a judgment


# ---------------------------------------------------------------------------
# DB-backed cache: end-to-end main() behavior
# ---------------------------------------------------------------------------

def _patch_llm():
    """Patch the three network-calling LLM helpers with counting AsyncMocks.

    topic + issue mocks return per-input results; quality returns a judgment for
    every requested session so each conversation gets a quality score.
    """
    async def fake_topics(msgs, **kw):
        return {m["id"]: {"topic": "gene_lookup", "complexity": 1,
                          "brief_reason": "t"} for m in msgs}

    async def fake_quality(session_ids, messages, **kw):
        return {sid: {"quality_score": 4, "answered": "yes", "accurate": "yes",
                      "efficient": "yes", "concluded": "yes",
                      "disposition": "good_answer", "issues": ["some issue"]}
                for sid in session_ids}

    async def fake_issue_cats(issues, **kw):
        return {i: "other" for i in issues}

    return (
        patch.object(ac, "categorize_with_llm", AsyncMock(side_effect=fake_topics)),
        patch.object(ac, "evaluate_quality_with_llm", AsyncMock(side_effect=fake_quality)),
        patch.object(ac, "categorize_issues_with_llm", AsyncMock(side_effect=fake_issue_cats)),
    )


async def _run_main(db_path, extra_args=None):
    """Run analyze_conversations.main() against db_path with LLM mocked.

    Returns (topic_mock, quality_mock) so tests can assert recompute happened or
    was skipped.
    """
    from genetics_mcp_server.db.chat_history_db import ChatHistoryDB
    from genetics_mcp_server.db.singleton import Singleton

    # a fresh DB instance each run so the singleton doesn't pin a stale path
    if ChatHistoryDB in Singleton._instances:
        del Singleton._instances[ChatHistoryDB]

    topic_p, quality_p, issue_p = _patch_llm()
    argv = ["analyze_conversations", "--db", db_path, "--report-only"]
    if extra_args:
        argv += extra_args
    with patch.object(sys, "argv", argv), topic_p as tm, quality_p as qm, issue_p:
        await ac.main()
    return tm, qm


def _analysis_rows(db_path):
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT session_id, analyzer_version, source_updated_at, message_count, "
            "llm_quality_score, success_label, topic FROM conversation_analysis"
        ).fetchall()
        issues = conn.execute(
            "SELECT session_id, category FROM conversation_issue"
        ).fetchall()
    finally:
        conn.close()
    return rows, issues


class TestDBBackedCache:
    @pytest.mark.asyncio
    async def test_results_written_to_db(self, sample_db):
        await _run_main(sample_db)

        rows, issues = _analysis_rows(sample_db)
        assert len(rows) == 3  # one analysis row per session
        by_sid = {r[0]: r for r in rows}
        # every row stored at the current analyzer version
        assert all(r[1] == ac.ANALYZER_VERSION for r in rows)
        # source_updated_at is the raw stored chat_sessions.updated_at (space, not 'T')
        assert by_sid["s1"][2] == "2025-12-10"
        assert "T" not in (by_sid["s1"][2] or "")
        # message_count and quality persisted
        assert by_sid["s1"][3] == 4  # s1 has 4 messages
        assert by_sid["s1"][4] == 4  # llm_quality_score from the mock
        # normalized issue rows written
        assert all(cat == "other" for _, cat in issues)
        assert len(issues) == 3  # each session got one "some issue" -> one category

    @pytest.mark.asyncio
    async def test_second_run_skips_llm_recompute(self, sample_db):
        await _run_main(sample_db)
        # second run: every session already analyzed at the current version
        tm, qm = await _run_main(sample_db)
        # no session needs topic classification or quality judging
        assert tm.call_count == 0
        assert qm.call_count == 0

    @pytest.mark.asyncio
    async def test_no_cache_forces_recompute(self, sample_db):
        await _run_main(sample_db)
        tm, qm = await _run_main(sample_db, ["--no-cache"])
        # --no-cache discards the DB cache, so the LLM is invoked again
        assert tm.call_count == 1  # one batched topic call
        assert qm.call_count == 1  # one quality call (covering all sessions)

    @pytest.mark.asyncio
    async def test_refresh_quality_recomputes_only_quality(self, sample_db):
        await _run_main(sample_db)
        tm, qm = await _run_main(sample_db, ["--refresh-quality"])
        # topics stay cached, quality is re-judged
        assert tm.call_count == 0
        assert qm.call_count == 1

    @pytest.mark.asyncio
    async def test_force_recomputes_everything(self, sample_db):
        await _run_main(sample_db)
        # --force ignores the cache for every in-range session: both LLM passes run
        tm, qm = await _run_main(sample_db, ["--force"])
        assert tm.call_count == 1  # one batched topic call covering all sessions
        assert qm.call_count == 1  # one quality call covering all sessions
        rows, _ = _analysis_rows(sample_db)
        assert all(r[1] == ac.ANALYZER_VERSION for r in rows)

    @pytest.mark.asyncio
    async def test_continued_conversation_reanalyzed_others_skipped(self, sample_db):
        await _run_main(sample_db)

        # continue s1: bump its updated_at past its analyzed_at. analyzed_at was set
        # to the run's CURRENT_TIMESTAMP (UTC now), so use a clearly-later but not
        # absurd-future value relative to that; a far-future date keeps it > analyzed_at.
        conn = sqlite3.connect(sample_db)
        try:
            bumped = "2099-01-01 00:00:00"
            conn.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE id = 's1'", (bumped,)
            )
            conn.commit()
        finally:
            conn.close()

        # only s1 is stale now, so the quality judge must see exactly s1
        async def fake_quality_capture(session_ids, messages, **kw):
            fake_quality_capture.seen = list(session_ids)
            return {sid: {"quality_score": 4, "answered": "yes", "accurate": "yes",
                          "efficient": "yes", "concluded": "yes",
                          "disposition": "good_answer", "issues": ["some issue"]}
                    for sid in session_ids}

        from genetics_mcp_server.db.chat_history_db import ChatHistoryDB
        from genetics_mcp_server.db.singleton import Singleton
        if ChatHistoryDB in Singleton._instances:
            del Singleton._instances[ChatHistoryDB]

        topic_p, _, issue_p = _patch_llm()
        quality_p = patch.object(
            ac, "evaluate_quality_with_llm", AsyncMock(side_effect=fake_quality_capture)
        )
        argv = ["analyze_conversations", "--db", sample_db, "--report-only"]
        with patch.object(sys, "argv", argv), topic_p, quality_p, issue_p:
            await ac.main()

        # exactly the continued session was re-judged; s2/s3 were skipped
        assert fake_quality_capture.seen == ["s1"]

        # round-trip: the run above re-persisted s1 with analyzed_at=CURRENT_TIMESTAMP
        # (the run's UTC wall clock). The continued-conversation requirement is that a
        # session whose updated_at is at or before that wall clock is no longer stale.
        # The 2099 value above is deliberately later-than-now to drive the reanalysis;
        # reset it to a realistic past timestamp (as a real continued conversation has)
        # and confirm the freshly written analyzed_at now dominates it.
        conn = sqlite3.connect(sample_db)
        try:
            conn.execute(
                "UPDATE chat_sessions SET updated_at = '2025-12-10 09:00:00' WHERE id = 's1'"
            )
            conn.commit()
        finally:
            conn.close()
        if ChatHistoryDB in Singleton._instances:
            del Singleton._instances[ChatHistoryDB]
        db = ChatHistoryDB(sample_db)
        stale = db.get_stale_or_missing_session_ids(
            force=False, analyzer_version=ac.ANALYZER_VERSION
        )
        assert "s1" not in stale

    @pytest.mark.asyncio
    async def test_version_bump_invalidates_cache(self, sample_db):
        await _run_main(sample_db)
        # simulate a new analyzer version: old rows must be treated as missing
        bumped = ac.ANALYZER_VERSION + 1
        with patch.object(ac, "ANALYZER_VERSION", bumped):
            tm, qm = await _run_main(sample_db)
            assert tm.call_count == 1
            assert qm.call_count == 1
            rows, _ = _analysis_rows(sample_db)
            # rows rewritten at the bumped version
            assert all(r[1] == bumped for r in rows)


# ---------------------------------------------------------------------------
# Instruction set breakdown
# ---------------------------------------------------------------------------

def _messages_with_instruction_sets(messages):
    """Attach instruction_set_id to the sample messages: s1 -> set-a, s3 -> set-b, s2 none."""
    per_session = {"s1": "set-a", "s3": "set-b"}
    return messages.with_columns(
        pl.Series(
            "instruction_set_id",
            [per_session.get(sid) for sid in messages["session_id"].to_list()],
            dtype=pl.Utf8,
        )
    )


def _messages_switching_sets_midway(messages):
    """s1's two user messages name different sets: m1 -> set-a, then m3 -> set-b."""
    per_message = {"m1": "set-a", "m3": "set-b"}
    return messages.with_columns(
        pl.Series(
            "instruction_set_id",
            [per_message.get(mid) for mid in messages["id"].to_list()],
            dtype=pl.Utf8,
        )
    )


def _topics():
    return {
        "s1": {"topic": "gene_lookup", "complexity": 1, "brief_reason": ""},
        "s2": {"topic": "variant_interpretation", "complexity": 1, "brief_reason": ""},
        "s3": {"topic": "literature_search", "complexity": 1, "brief_reason": ""},
    }


class TestInstructionSetMetrics:
    def test_missing_column_leaves_sets_empty(self, sample_db):
        """chat_messages predating the migration has no instruction_set_id column."""
        sessions, messages = load_data(sample_db)
        assert "instruction_set_id" not in messages.columns
        metrics = compute_all_metrics(
            sessions, messages, build_session_tool_stats(messages), _topics()
        )
        assert all(m.instruction_set_id == "" for m in metrics)
        assert all(m.instruction_set_name == "" for m in metrics)

    def test_names_resolved_from_config_db(self, sample_db):
        sessions, messages = load_data(sample_db)
        messages = _messages_with_instruction_sets(messages)
        metrics = compute_all_metrics(
            sessions,
            messages,
            build_session_tool_stats(messages),
            _topics(),
            {"set-a": "Statistician", "set-b": "Terse"},
        )
        by_id = {m.session_id: m for m in metrics}
        assert by_id["s1"].instruction_set_id == "set-a"
        assert by_id["s1"].instruction_set_name == "Statistician"
        assert by_id["s3"].instruction_set_name == "Terse"
        assert by_id["s2"].instruction_set_id == ""
        assert by_id["s2"].instruction_set_name == ""

    def test_unresolvable_id_groups_under_the_id(self, sample_db):
        sessions, messages = load_data(sample_db)
        messages = _messages_with_instruction_sets(messages)
        metrics = compute_all_metrics(
            sessions, messages, build_session_tool_stats(messages), _topics()
        )
        by_id = {m.session_id: m for m in metrics}
        assert by_id["s1"].instruction_set_name == "set-a"

    def test_last_set_named_in_a_session_wins(self, sample_db):
        """The selector can move mid-conversation; the session runs under the newest one."""
        sessions, messages = load_data(sample_db)
        messages = _messages_switching_sets_midway(messages)
        metrics = compute_all_metrics(
            sessions,
            messages,
            build_session_tool_stats(messages),
            _topics(),
            {"set-a": "Statistician", "set-b": "Terse"},
        )
        by_id = {m.session_id: m for m in metrics}
        assert by_id["s1"].instruction_set_id == "set-b"
        assert by_id["s1"].instruction_set_name == "Terse"

    def test_report_groups_by_instruction_set(self, sample_db):
        sessions, messages = load_data(sample_db)
        messages = _messages_with_instruction_sets(messages)
        tool_stats = build_session_tool_stats(messages)
        metrics = compute_all_metrics(
            sessions, messages, tool_stats, _topics(),
            {"set-a": "Statistician", "set-b": "Terse"},
        )
        report = generate_report(metrics, sessions, messages, tool_stats)

        assert "## Instruction Set Usage" in report
        assert "| Instructions | Count | Avg Score |" in report
        assert "| Statistician | 1 |" in report
        assert "| Terse | 1 |" in report
        # the session that used none still shows, so the baseline is comparable
        assert "| (none) | 1 |" in report


class TestLoadInstructionSetNames:
    def test_missing_file_returns_empty(self, tmp_path):
        assert ac.load_instruction_set_names(str(tmp_path / "nope.db")) == {}

    def test_missing_table_returns_empty(self, tmp_path):
        path = str(tmp_path / "empty.db")
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE unrelated (id TEXT)")
        conn.close()
        assert ac.load_instruction_set_names(path) == {}

    def test_reads_id_to_name(self, tmp_path):
        path = str(tmp_path / "llm_config.db")
        conn = sqlite3.connect(path)
        conn.execute(
            "CREATE TABLE user_instruction_sets (id TEXT PRIMARY KEY, user_id TEXT, "
            "name TEXT, body TEXT)"
        )
        conn.execute(
            "INSERT INTO user_instruction_sets VALUES ('s1', 'u@x.com', 'Statistician', 'b')"
        )
        conn.commit()
        conn.close()
        assert ac.load_instruction_set_names(path) == {"s1": "Statistician"}


class TestResolveLlmConfigDb:
    """The deployed CronJob passes only --db and relies on this default, so the derivation is
    load bearing. It used to live inline in main() where no test could reach it
    (genetics-results-suite-uvh 8)."""

    def test_defaults_to_the_sibling_of_the_chat_db(self):
        from genetics_mcp_server.scripts.analyze_conversations import resolve_llm_config_db

        assert resolve_llm_config_db("/data/chat_history.db", None) == "/data/llm_config.db"

    def test_an_explicit_path_wins(self):
        from genetics_mcp_server.scripts.analyze_conversations import resolve_llm_config_db

        assert resolve_llm_config_db("/data/chat_history.db", "/other/x.db") == "/other/x.db"

    def test_a_bare_filename_resolves_beside_it_rather_than_at_the_root(self):
        from genetics_mcp_server.scripts.analyze_conversations import resolve_llm_config_db

        assert resolve_llm_config_db("chat_history.db", None) == "llm_config.db"
