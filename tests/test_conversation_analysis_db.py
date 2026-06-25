"""Unit tests for the conversation analysis cache tables and methods."""

from dataclasses import dataclass


@dataclass
class FakeMetrics:
    """Minimal ConversationMetrics-like object for upsert_analysis tests."""
    session_id: str
    created_at: str = ""
    user_rating: int | None = None
    success_label: str = "unknown"
    topic: str = "general_genetics"
    complexity: int = 2
    llm_quality_score: int | None = None
    llm_disposition: str = ""
    llm_issue_categories: list[str] | None = None


def _set_updated_at(db, session_id: str, value: str) -> None:
    """Force a deterministic chat_sessions.updated_at (CURRENT_TIMESTAMP has 1s resolution)."""
    conn = db._conn
    conn.execute(
        "UPDATE chat_sessions SET updated_at = ? WHERE id = ?", (value, session_id)
    )
    conn.commit()


def _set_analyzed_at(db, session_id: str, value: str) -> None:
    conn = db._conn
    conn.execute(
        "UPDATE conversation_analysis SET analyzed_at = ? WHERE session_id = ?",
        (value, session_id),
    )
    conn.commit()


class TestConversationAnalysisSchema:
    def test_tables_created(self, chat_history_db):
        cursor = chat_history_db._conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN "
            "('conversation_analysis', 'conversation_issue')"
        )
        names = {row["name"] for row in cursor.fetchall()}
        assert names == {"conversation_analysis", "conversation_issue"}

    def test_wal_enabled(self, chat_history_db):
        cursor = chat_history_db._conn.cursor()
        cursor.execute("PRAGMA journal_mode")
        assert cursor.fetchone()[0].lower() == "wal"


class TestUpsertAnalysis:
    def test_writes_analysis_and_issue_rows(self, chat_history_db):
        session = chat_history_db.create_session("user@example.com")
        metrics = FakeMetrics(
            session_id=session.id,
            user_rating=4,
            success_label="successful",
            topic="variant_lookup",
            complexity=3,
            llm_quality_score=5,
            llm_disposition="resolved",
            llm_issue_categories=["hallucination", "missing_context"],
        )

        chat_history_db.upsert_analysis(
            metrics, analyzer_version=1, source_updated_at="2026-06-25T10:00:00", message_count=6
        )

        amap = chat_history_db.get_analysis_map()
        assert session.id in amap
        row = amap[session.id]
        assert row["user_rating"] == 4
        assert row["llm_quality_score"] == 5
        assert row["success_label"] == "successful"
        assert row["llm_disposition"] == "resolved"
        assert row["topic"] == "variant_lookup"
        assert row["complexity"] == 3
        assert row["analyzer_version"] == 1
        assert row["message_count"] == 6
        assert row["metrics_json"]

        cursor = chat_history_db._conn.cursor()
        cursor.execute(
            "SELECT category FROM conversation_issue WHERE session_id = ? ORDER BY category",
            (session.id,),
        )
        cats = [r["category"] for r in cursor.fetchall()]
        assert cats == ["hallucination", "missing_context"]

    def test_accepts_dict(self, chat_history_db):
        session = chat_history_db.create_session("user@example.com")
        chat_history_db.upsert_analysis(
            {"session_id": session.id, "llm_issue_categories": ["foo"]},
            analyzer_version=2,
            source_updated_at=None,
            message_count=1,
        )
        amap = chat_history_db.get_analysis_map()
        assert amap[session.id]["analyzer_version"] == 2

    def test_idempotent_replaces_issues(self, chat_history_db):
        session = chat_history_db.create_session("user@example.com")
        chat_history_db.upsert_analysis(
            FakeMetrics(session_id=session.id, llm_issue_categories=["a", "b", "c"]),
            analyzer_version=1, source_updated_at=None, message_count=2,
        )
        # re-analysis with a different issue set must fully replace, not append
        chat_history_db.upsert_analysis(
            FakeMetrics(session_id=session.id, llm_issue_categories=["d"]),
            analyzer_version=1, source_updated_at=None, message_count=3,
        )

        cursor = chat_history_db._conn.cursor()
        cursor.execute(
            "SELECT category FROM conversation_issue WHERE session_id = ?", (session.id,)
        )
        cats = [r["category"] for r in cursor.fetchall()]
        assert cats == ["d"]

        cursor.execute(
            "SELECT COUNT(*) AS c FROM conversation_analysis WHERE session_id = ?",
            (session.id,),
        )
        assert cursor.fetchone()["c"] == 1
        assert chat_history_db.get_analysis_map()[session.id]["message_count"] == 3

    def test_empty_issue_categories(self, chat_history_db):
        session = chat_history_db.create_session("user@example.com")
        chat_history_db.upsert_analysis(
            FakeMetrics(session_id=session.id, llm_issue_categories=None),
            analyzer_version=1, source_updated_at=None, message_count=1,
        )
        cursor = chat_history_db._conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) AS c FROM conversation_issue WHERE session_id = ?",
            (session.id,),
        )
        assert cursor.fetchone()["c"] == 0


class TestStaleOrMissing:
    def test_missing_row_is_stale(self, chat_history_db):
        s1 = chat_history_db.create_session("user@example.com")
        s2 = chat_history_db.create_session("user@example.com")
        chat_history_db.upsert_analysis(
            FakeMetrics(session_id=s1.id), analyzer_version=1,
            source_updated_at=None, message_count=1,
        )
        # keep s1 fresh: analyzed after its updated_at
        _set_updated_at(chat_history_db, s1.id, "2026-06-25T10:00:00")
        _set_analyzed_at(chat_history_db, s1.id, "2026-06-25T11:00:00")

        stale = chat_history_db.get_stale_or_missing_session_ids(force=False, analyzer_version=1)
        assert s2.id in stale
        assert s1.id not in stale

    def test_continued_conversation_is_stale(self, chat_history_db):
        s = chat_history_db.create_session("user@example.com")
        chat_history_db.upsert_analysis(
            FakeMetrics(session_id=s.id), analyzer_version=1,
            source_updated_at=None, message_count=1,
        )
        _set_analyzed_at(chat_history_db, s.id, "2026-06-25T10:00:00")
        # new messages arrived after analysis
        _set_updated_at(chat_history_db, s.id, "2026-06-25T12:00:00")

        stale = chat_history_db.get_stale_or_missing_session_ids(force=False, analyzer_version=1)
        assert s.id in stale

    def test_analyzer_version_mismatch_is_stale(self, chat_history_db):
        s = chat_history_db.create_session("user@example.com")
        chat_history_db.upsert_analysis(
            FakeMetrics(session_id=s.id), analyzer_version=1,
            source_updated_at=None, message_count=1,
        )
        _set_updated_at(chat_history_db, s.id, "2026-06-25T10:00:00")
        _set_analyzed_at(chat_history_db, s.id, "2026-06-25T11:00:00")

        # current version bumped -> needs reanalysis
        stale = chat_history_db.get_stale_or_missing_session_ids(force=False, analyzer_version=2)
        assert s.id in stale
        # matching version -> not stale
        fresh = chat_history_db.get_stale_or_missing_session_ids(force=False, analyzer_version=1)
        assert s.id not in fresh

    def test_force_returns_all(self, chat_history_db):
        s1 = chat_history_db.create_session("user@example.com")
        s2 = chat_history_db.create_session("user@example.com")
        chat_history_db.upsert_analysis(
            FakeMetrics(session_id=s1.id), analyzer_version=1,
            source_updated_at=None, message_count=1,
        )
        _set_updated_at(chat_history_db, s1.id, "2026-06-25T10:00:00")
        _set_analyzed_at(chat_history_db, s1.id, "2026-06-25T11:00:00")

        all_ids = chat_history_db.get_stale_or_missing_session_ids(force=True, analyzer_version=1)
        assert set(all_ids) == {s1.id, s2.id}
