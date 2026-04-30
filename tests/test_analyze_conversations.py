"""Tests for conversation analysis script."""

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import polars as pl
import pytest

from genetics_mcp_server.scripts.analyze_conversations import (
    ConversationMetrics,
    _format_conversation_for_eval,
    apply_quality_assessments,
    build_session_tool_stats,
    categorize_by_keywords,
    categorize_with_llm,
    compute_all_metrics,
    compute_success_score,
    evaluate_quality_with_llm,
    export_eval_dataset,
    generate_report,
    label_success,
    load_data,
    parse_tool_calls,
)


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


# ---------------------------------------------------------------------------
# LLM categorization tests (mocked)
# ---------------------------------------------------------------------------

class TestLLMCategorization:
    @pytest.mark.asyncio
    async def test_successful_categorization(self):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps([
            {"id": "s1", "topic": "gene_lookup", "complexity": 1, "brief_reason": "gene query"},
            {"id": "s2", "topic": "variant_interpretation", "complexity": 2, "brief_reason": "variant"},
        ]))]

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch.dict("sys.modules", {"anthropic": MagicMock(AsyncAnthropic=lambda: mock_client)}):
            result = await categorize_with_llm([
                {"id": "s1", "text": "What about BRCA1?"},
                {"id": "s2", "text": "What does 1:123:A:G do?"},
            ])

        assert result["s1"]["topic"] == "gene_lookup"
        assert result["s2"]["topic"] == "variant_interpretation"

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
        assert metrics[0].success_score == 0.75  # (4-1)/4
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

    def test_user_rating_still_takes_priority(self):
        m = ConversationMetrics(
            session_id="s1", user_rating=1, llm_quality_score=5,
        )
        score = compute_success_score(m)
        assert score == 0.0  # user rating wins

    @pytest.mark.asyncio
    async def test_evaluate_quality_with_llm(self):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps({
            "answered": "yes", "accurate": "yes", "efficient": "yes",
            "concluded": "yes", "quality_score": 5, "issues": [],
        }))]

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
