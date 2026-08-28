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


class TestPlotLoadFromDbDelegates:
    """plot_conversation_scores.load_from_db must use the one authoritative
    DB-read path (list_all_analysis_rows) and feed build_all_series correctly."""

    def test_load_from_db_maps_keys_and_keeps_undated_metrics(self, chat_history_db):
        from genetics_mcp_server.scripts import analysis_timeseries as ats
        from genetics_mcp_server.scripts.plot_conversation_scores import load_from_db

        s = chat_history_db.create_session("user@example.com")
        # native chat_sessions.created_at is the authoritative date the plots key
        # on; force it so the assertion is deterministic
        _set_updated_at(chat_history_db, s.id, "2026-01-10T12:00:00")
        chat_history_db._conn.execute(
            "UPDATE chat_sessions SET created_at = ? WHERE id = ?",
            ("2026-01-10T12:00:00", s.id),
        )
        chat_history_db._conn.commit()
        # metrics_json deliberately has no created_at: the old metrics_json-based
        # path dropped this row; the unified path must keep it via chat_sessions
        chat_history_db.upsert_analysis(
            FakeMetrics(
                session_id=s.id,
                llm_quality_score=4,
                success_label="successful",
                llm_disposition="good_answer",
                llm_issue_categories=["hallucination"],
            ),
            analyzer_version=1, source_updated_at=None, message_count=2,
        )

        records = load_from_db(chat_history_db.db_path)
        assert len(records) == 1
        r = records[0]
        # keys the plotter / analysis_timeseries consume
        assert set(r) >= {
            "created_at", "llm_quality_score", "llm_disposition",
            "success_label", "llm_issue_categories",
        }
        assert r["created_at"] == "2026-01-10T12:00:00"
        assert r["llm_quality_score"] == 4
        assert r["success_label"] == "successful"
        assert r["llm_disposition"] == "good_answer"
        # issue_categories -> llm_issue_categories mapping
        assert r["llm_issue_categories"] == ["hallucination"]

        # records flow through to all four panels without error
        series = ats.build_all_series(records, window=7, min_n=1)
        assert series["meta"]["empty"] is False
        assert series["meta"]["total"] == 1
        assert series["meta"]["scored"] == 1


class TestJudgeModelColumns:
    """topic_model / quality_model: the migration onto a long-lived production file,
    and the COALESCE that keeps a cached re-upsert from erasing recorded provenance
    (genetics-results-suite-9wv)."""

    OLD_SCHEMA = """
        CREATE TABLE conversation_analysis (
            session_id TEXT PRIMARY KEY REFERENCES chat_sessions(id) ON DELETE CASCADE,
            analyzed_at TIMESTAMP,
            analyzer_version INTEGER,
            source_updated_at TIMESTAMP,
            message_count INTEGER,
            user_rating INTEGER,
            llm_quality_score INTEGER,
            success_label TEXT,
            llm_disposition TEXT,
            topic TEXT,
            complexity INTEGER,
            metrics_json TEXT
        )
    """

    @staticmethod
    def _columns(db):
        cursor = db._conn.cursor()
        cursor.execute("PRAGMA table_info(conversation_analysis)")
        return [row[1] for row in cursor.fetchall()]

    def test_migrates_old_schema_and_is_idempotent(self, tmp_path):
        """A production file whose conversation_analysis predates the two columns must
        gain them on open (CREATE TABLE IF NOT EXISTS is a no-op there), and a second
        open must not attempt the ALTERs again."""
        import sqlite3

        from genetics_mcp_server.db.chat_history_db import ChatHistoryDB
        from genetics_mcp_server.db.singleton import Singleton

        db_path = str(tmp_path / "old_schema.db")
        conn = sqlite3.connect(db_path)
        conn.execute(self.OLD_SCHEMA)
        conn.execute(
            "INSERT INTO conversation_analysis (session_id, analyzer_version, topic) "
            "VALUES ('legacy', 1, 'gene_lookup')"
        )
        conn.commit()
        conn.close()

        dbs = []
        try:
            for _ in range(2):
                if ChatHistoryDB in Singleton._instances:
                    del Singleton._instances[ChatHistoryDB]
                db = ChatHistoryDB(db_path)
                dbs.append(db)
                cols = self._columns(db)
                assert "topic_model" in cols
                assert "quality_model" in cols
                # exactly one of each: a second ALTER would have raised, and a
                # duplicate column is impossible in SQLite anyway
                assert cols.count("topic_model") == 1
                assert cols.count("quality_model") == 1

            # the pre-existing row survives the migration with NULL models: which
            # judge produced it is not recoverable, and NULL says so honestly
            cursor = dbs[-1]._conn.cursor()
            cursor.execute(
                "SELECT topic, topic_model, quality_model FROM conversation_analysis "
                "WHERE session_id = 'legacy'"
            )
            row = cursor.fetchone()
            assert row["topic"] == "gene_lookup"
            assert row["topic_model"] is None
            assert row["quality_model"] is None
        finally:
            if ChatHistoryDB in Singleton._instances:
                del Singleton._instances[ChatHistoryDB]
            for db in dbs:
                for c in db._connections.values():
                    try:
                        c.close()
                    except Exception:
                        pass
                db._connections.clear()

    def test_models_round_trip(self, chat_history_db):
        session = chat_history_db.create_session("user@example.com")
        chat_history_db.upsert_analysis(
            FakeMetrics(session_id=session.id),
            analyzer_version=1, source_updated_at=None, message_count=2,
            topic_model="claude-sonnet-4-6", quality_model="claude-opus-4-8",
        )
        row = chat_history_db.get_analysis_map()[session.id]
        assert row["topic_model"] == "claude-sonnet-4-6"
        assert row["quality_model"] == "claude-opus-4-8"

    def test_defaults_stay_null(self, chat_history_db):
        """A caller that never passes the kwargs leaves the models NULL, not ''."""
        session = chat_history_db.create_session("user@example.com")
        chat_history_db.upsert_analysis(
            FakeMetrics(session_id=session.id),
            analyzer_version=1, source_updated_at=None, message_count=2,
        )
        row = chat_history_db.get_analysis_map()[session.id]
        assert row["topic_model"] is None
        assert row["quality_model"] is None

    def test_cached_reupsert_preserves_recorded_models(self, chat_history_db):
        """The COALESCE: a re-upsert that recomputed nothing passes None and must
        NOT null out (or restamp) the models recorded when the row was judged."""
        session = chat_history_db.create_session("user@example.com")
        chat_history_db.upsert_analysis(
            FakeMetrics(session_id=session.id, llm_quality_score=4),
            analyzer_version=1, source_updated_at=None, message_count=2,
            topic_model="claude-sonnet-4-6", quality_model="claude-opus-4-8",
        )
        # replayed from cache on a later run: no LLM ran, so both models are None
        chat_history_db.upsert_analysis(
            FakeMetrics(session_id=session.id, llm_quality_score=4),
            analyzer_version=1, source_updated_at=None, message_count=3,
        )
        row = chat_history_db.get_analysis_map()[session.id]
        assert row["message_count"] == 3  # the rest of the row still updates
        assert row["topic_model"] == "claude-sonnet-4-6"
        assert row["quality_model"] == "claude-opus-4-8"

        # a half-recompute (quality re-judged, topic replayed) updates only quality
        chat_history_db.upsert_analysis(
            FakeMetrics(session_id=session.id, llm_quality_score=5),
            analyzer_version=1, source_updated_at=None, message_count=3,
            quality_model="claude-opus-5",
        )
        row = chat_history_db.get_analysis_map()[session.id]
        assert row["topic_model"] == "claude-sonnet-4-6"
        assert row["quality_model"] == "claude-opus-5"
