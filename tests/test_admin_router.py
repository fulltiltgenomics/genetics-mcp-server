"""Integration tests for admin router endpoints."""

import os
import tempfile
import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from genetics_mcp_server.auth import admin_required, auth_required
from genetics_mcp_server.chat_api import app
from genetics_mcp_server.db.chat_history_db import ChatHistoryDB
from genetics_mcp_server.db.llm_config_db import LLMConfigDB
from genetics_mcp_server.db.singleton import Singleton


@pytest.fixture
def test_db():
    """Create a temporary database for testing."""
    if ChatHistoryDB in Singleton._instances:
        del Singleton._instances[ChatHistoryDB]

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    db = ChatHistoryDB(db_path)
    yield db

    if ChatHistoryDB in Singleton._instances:
        del Singleton._instances[ChatHistoryDB]
    os.unlink(db_path)


@pytest.fixture
def seeded_db(test_db):
    """Create test DB with sessions and messages from multiple users."""
    for i, user in enumerate(["alice@example.com", "bob@example.com", "alice@example.com"]):
        session = test_db.create_session(user)
        test_db.add_message(session.id, f"msg-{i}-1", "user", f"Hello from {user}")
        test_db.add_message(session.id, f"msg-{i}-2", "assistant", f"Hi {user}!")
    return test_db


@pytest.fixture
def admin_client(seeded_db):
    """Test client with admin auth override and ENABLE_ADMIN_PAGE=true."""
    from genetics_mcp_server.config.settings import get_settings

    async def mock_admin():
        return "admin@example.com"

    app.dependency_overrides[admin_required] = mock_admin

    with patch.dict(os.environ, {"ENABLE_ADMIN_PAGE": "true"}):
        get_settings.cache_clear()
        with patch("genetics_mcp_server.routers.admin.get_chat_history_db", return_value=seeded_db):
            with TestClient(app) as client:
                yield client
        get_settings.cache_clear()

    app.dependency_overrides.clear()


class TestAdminSessions:

    def test_list_all_sessions(self, admin_client):
        response = admin_client.get("/chat/v1/admin/sessions")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert len(data["sessions"]) == 3
        users = {s["user_id"] for s in data["sessions"]}
        assert "alice@example.com" in users
        assert "bob@example.com" in users

    def test_list_sessions_filter_by_user(self, admin_client):
        response = admin_client.get("/chat/v1/admin/sessions?user=bob")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["sessions"][0]["user_id"] == "bob@example.com"

    def test_list_sessions_pagination(self, admin_client):
        response = admin_client.get("/chat/v1/admin/sessions?limit=2&offset=0")
        data = response.json()
        assert len(data["sessions"]) == 2
        assert data["total"] == 3

        response2 = admin_client.get("/chat/v1/admin/sessions?limit=2&offset=2")
        assert len(response2.json()["sessions"]) == 1

    def test_list_sessions_has_message_count_and_preview(self, admin_client):
        response = admin_client.get("/chat/v1/admin/sessions")
        for s in response.json()["sessions"]:
            assert s["message_count"] == 2
            assert s["preview"] is not None
            assert "Hello from" in s["preview"]


class TestAdminSessionDetail:

    def test_get_session_detail(self, admin_client, seeded_db):
        sessions, _ = seeded_db.list_all_sessions()
        session_id = sessions[0].id

        response = admin_client.get(f"/chat/v1/admin/sessions/{session_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == session_id
        assert len(data["messages"]) == 2
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][1]["role"] == "assistant"

    def test_get_session_not_found(self, admin_client):
        response = admin_client.get("/chat/v1/admin/sessions/nonexistent-id")
        assert response.status_code == 404


class TestAdminAnalytics:

    def test_analytics_week(self, admin_client):
        response = admin_client.get("/chat/v1/admin/analytics/usage?period=week")
        assert response.status_code == 200
        data = response.json()
        assert data["period"] == "week"
        assert len(data["data"]) >= 1
        point = data["data"][0]
        assert point["unique_users"] == 2
        assert point["conversations"] == 3

    def test_analytics_month(self, admin_client):
        response = admin_client.get("/chat/v1/admin/analytics/usage?period=month")
        assert response.status_code == 200
        assert response.json()["period"] == "month"

    def test_analytics_year(self, admin_client):
        response = admin_client.get("/chat/v1/admin/analytics/usage?period=year")
        assert response.status_code == 200

    def test_analytics_invalid_period(self, admin_client):
        response = admin_client.get("/chat/v1/admin/analytics/usage?period=invalid")
        assert response.status_code == 400


class TestAdminAuthGuards:

    def test_non_admin_denied_when_auth_required(self, seeded_db):
        """Non-admin user gets 403 when REQUIRE_AUTH is true."""
        from genetics_mcp_server.config.settings import get_settings

        async def mock_auth():
            return "regular@example.com"

        app.dependency_overrides[auth_required] = mock_auth

        env = {"REQUIRE_AUTH": "true", "ADMIN_USERS": "admin@example.com", "ENABLE_ADMIN_PAGE": "true"}
        with patch("genetics_mcp_server.routers.admin.get_chat_history_db", return_value=seeded_db):
            with patch.dict(os.environ, env):
                get_settings.cache_clear()
                with patch("genetics_mcp_server.auth.dependencies._require_auth", True):
                    with TestClient(app) as client:
                        response = client.get("/chat/v1/admin/sessions")
                        assert response.status_code == 403
                get_settings.cache_clear()

        app.dependency_overrides.clear()

    def test_dev_mode_allows_any_user(self, seeded_db):
        """When REQUIRE_AUTH is false, any user can access admin."""
        from genetics_mcp_server.config.settings import get_settings

        with patch("genetics_mcp_server.routers.admin.get_chat_history_db", return_value=seeded_db):
            with patch.dict(os.environ, {"ENABLE_ADMIN_PAGE": "true"}):
                get_settings.cache_clear()
                with patch("genetics_mcp_server.auth.dependencies._require_auth", False):
                    with TestClient(app) as client:
                        response = client.get("/chat/v1/admin/sessions")
                        assert response.status_code == 200
                get_settings.cache_clear()

        app.dependency_overrides.clear()

    def test_admin_disabled_returns_404(self, seeded_db):
        """When ENABLE_ADMIN_PAGE is false, admin endpoints return 404."""
        from genetics_mcp_server.config.settings import get_settings

        with patch("genetics_mcp_server.routers.admin.get_chat_history_db", return_value=seeded_db):
            with patch.dict(os.environ, {"ENABLE_ADMIN_PAGE": "false"}):
                get_settings.cache_clear()
                with TestClient(app) as client:
                    response = client.get("/chat/v1/admin/sessions")
                    assert response.status_code == 404
                get_settings.cache_clear()

        app.dependency_overrides.clear()


class TestAdminDBMethods:

    def test_list_all_sessions_no_filter(self, seeded_db):
        sessions, total = seeded_db.list_all_sessions()
        assert total == 3
        assert len(sessions) == 3

    def test_list_all_sessions_user_filter(self, seeded_db):
        sessions, total = seeded_db.list_all_sessions(user_filter="alice")
        assert total == 2

    def test_list_all_sessions_pagination(self, seeded_db):
        sessions, total = seeded_db.list_all_sessions(limit=1, offset=0)
        assert total == 3
        assert len(sessions) == 1

    def test_get_session_any_user(self, seeded_db):
        sessions, _ = seeded_db.list_all_sessions()
        session = seeded_db.get_session_any_user(sessions[0].id)
        assert session is not None

    def test_get_session_any_user_not_found(self, seeded_db):
        assert seeded_db.get_session_any_user("nonexistent") is None

    def test_get_usage_analytics(self, seeded_db):
        data = seeded_db.get_usage_analytics("week")
        assert len(data) >= 1
        assert data[0]["unique_users"] == 2
        assert data[0]["conversations"] == 3

    def test_get_usage_analytics_periods(self, seeded_db):
        for period in ("week", "month", "year"):
            data = seeded_db.get_usage_analytics(period)
            assert isinstance(data, list)

    def test_list_all_sessions_default_filters_unset(self, seeded_db):
        """New analysis filter params default to None and don't drop rows."""
        sessions, total = seeded_db.list_all_sessions(
            disposition=None, rating=None, success_label=None, min_issues=None
        )
        assert total == 3

    def test_list_all_user_comments(self):
        """list_all_user_comments returns all comments ordered by created_at DESC."""
        if LLMConfigDB in Singleton._instances:
            del Singleton._instances[LLMConfigDB]

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            config_db = LLMConfigDB(db_path)
            # insert with explicit timestamps for deterministic ordering
            cursor = config_db._conn.cursor()
            cursor.execute(
                "INSERT INTO user_comments (user_id, comment, created_at) VALUES (?, ?, ?)",
                ("alice@example.com", "first comment", "2025-01-01T10:00:00"),
            )
            cursor.execute(
                "INSERT INTO user_comments (user_id, comment, created_at) VALUES (?, ?, ?)",
                ("bob@example.com", "second comment", "2025-01-01T11:00:00"),
            )
            cursor.execute(
                "INSERT INTO user_comments (user_id, comment, created_at) VALUES (?, ?, ?)",
                ("alice@example.com", "third comment", "2025-01-01T12:00:00"),
            )
            config_db._conn.commit()

            results = config_db.list_all_user_comments()
            assert len(results) == 3
            assert results[0].comment == "third comment"
            assert results[1].comment == "second comment"
            assert results[2].comment == "first comment"
            # verify multiple users are present
            users = {r.user_id for r in results}
            assert users == {"alice@example.com", "bob@example.com"}
        finally:
            if LLMConfigDB in Singleton._instances:
                del Singleton._instances[LLMConfigDB]
            os.unlink(db_path)

    def test_list_sessions_with_comments(self, test_db):
        """list_sessions_with_comments returns only sessions with non-empty comments."""
        # session with a comment
        s1 = test_db.create_session("alice@example.com")
        test_db.update_session(s1.id, "alice@example.com", comment="great session")

        # session without a comment
        s2 = test_db.create_session("bob@example.com")
        test_db.add_message(s2.id, "msg-1", "user", "hi")

        # another session with a comment
        s3 = test_db.create_session("carol@example.com")
        test_db.update_session(s3.id, "carol@example.com", comment="needs improvement")

        results = test_db.list_sessions_with_comments()
        assert len(results) == 2
        session_ids = {r["session_id"] for r in results}
        assert s1.id in session_ids
        assert s3.id in session_ids
        assert s2.id not in session_ids
        # verify returned fields
        for r in results:
            assert "user_id" in r
            assert "comment" in r
            assert "created_at" in r
            assert "session_id" in r


class TestAdminFeedbackEndpoint:

    @pytest.fixture
    def feedback_client(self, test_db):
        """Client with both chat_history_db and llm_config_db mocked for feedback tests."""
        from genetics_mcp_server.config.settings import get_settings

        if LLMConfigDB in Singleton._instances:
            del Singleton._instances[LLMConfigDB]

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            config_db_path = f.name

        config_db = LLMConfigDB(config_db_path)

        async def mock_admin():
            return "admin@example.com"

        app.dependency_overrides[admin_required] = mock_admin

        with patch.dict(os.environ, {"ENABLE_ADMIN_PAGE": "true"}):
            get_settings.cache_clear()
            with (
                patch("genetics_mcp_server.routers.admin.get_chat_history_db", return_value=test_db),
                patch("genetics_mcp_server.routers.admin.get_llm_config_db", return_value=config_db),
            ):
                with TestClient(app) as client:
                    yield client, test_db, config_db
            get_settings.cache_clear()

        app.dependency_overrides.clear()
        if LLMConfigDB in Singleton._instances:
            del Singleton._instances[LLMConfigDB]
        os.unlink(config_db_path)

    def test_list_feedback_empty(self, feedback_client):
        client, _, _ = feedback_client
        response = client.get("/chat/v1/admin/feedback")
        assert response.status_code == 200
        data = response.json()
        assert data["items"] == []
        assert data["total"] == 0
        assert data["latest_at"] is None

    def test_list_feedback_combined(self, feedback_client):
        """Feedback from both user_comments and session comments are merged and sorted."""
        client, chat_db, config_db = feedback_client

        # add feedback dialog comments
        config_db.add_user_comment("alice@example.com", "feedback dialog comment 1")
        time.sleep(0.05)

        # add session with comment
        s = chat_db.create_session("bob@example.com")
        chat_db.update_session(s.id, "bob@example.com", comment="session comment")
        time.sleep(0.05)

        config_db.add_user_comment("carol@example.com", "feedback dialog comment 2")

        response = client.get("/chat/v1/admin/feedback")
        assert response.status_code == 200
        data = response.json()

        assert data["total"] == 3
        assert len(data["items"]) == 3
        assert data["latest_at"] is not None

        # verify sorted by created_at DESC — latest first
        timestamps = [item["created_at"] for item in data["items"]]
        assert timestamps == sorted(timestamps, reverse=True)

        # verify source fields
        sources = {item["source"] for item in data["items"]}
        assert sources == {"feedback_dialog", "session_comment"}

        # session_comment item should have session_id
        session_items = [i for i in data["items"] if i["source"] == "session_comment"]
        assert len(session_items) == 1
        assert session_items[0]["session_id"] == s.id
        assert session_items[0]["user"] == "bob@example.com"

        # feedback_dialog items should have no session_id
        dialog_items = [i for i in data["items"] if i["source"] == "feedback_dialog"]
        assert all(i["session_id"] is None for i in dialog_items)

    def test_list_feedback_pagination(self, feedback_client):
        """Pagination limit/offset work on the merged feed."""
        client, chat_db, config_db = feedback_client

        # seed 5 items total
        for i in range(3):
            config_db.add_user_comment(f"user{i}@example.com", f"dialog comment {i}")
            time.sleep(0.02)

        for i in range(2):
            s = chat_db.create_session(f"sess_user{i}@example.com")
            chat_db.update_session(s.id, f"sess_user{i}@example.com", comment=f"session comment {i}")
            time.sleep(0.02)

        # full list
        resp_all = client.get("/chat/v1/admin/feedback?limit=50&offset=0")
        assert resp_all.json()["total"] == 5

        # first page
        resp_p1 = client.get("/chat/v1/admin/feedback?limit=2&offset=0")
        data_p1 = resp_p1.json()
        assert len(data_p1["items"]) == 2
        assert data_p1["total"] == 5

        # second page
        resp_p2 = client.get("/chat/v1/admin/feedback?limit=2&offset=2")
        data_p2 = resp_p2.json()
        assert len(data_p2["items"]) == 2

        # third page (last item)
        resp_p3 = client.get("/chat/v1/admin/feedback?limit=2&offset=4")
        data_p3 = resp_p3.json()
        assert len(data_p3["items"]) == 1

        # no overlap between pages
        all_comments = [i["comment"] for i in data_p1["items"] + data_p2["items"] + data_p3["items"]]
        assert len(all_comments) == len(set(all_comments))


class TestAdminSessionsAnalysisJoin:
    """DB-level tests for the conversation_analysis join in list_all_sessions
    and for list_all_analysis_rows (task genetics-results-suite-w8o.5)."""

    @pytest.fixture
    def analyzed_db(self, test_db):
        """Three sessions; two carry analysis rows, one is left unanalyzed."""
        s1 = test_db.create_session("alice@example.com")
        s2 = test_db.create_session("bob@example.com")
        s3 = test_db.create_session("carol@example.com")

        test_db.upsert_analysis(
            {
                "session_id": s1.id,
                "llm_quality_score": 5,
                "success_label": "successful",
                "llm_disposition": "answered",
                "llm_issue_categories": ["hallucination", "tone"],
            },
            analyzer_version=1,
            source_updated_at=None,
            message_count=4,
        )
        test_db.upsert_analysis(
            {
                "session_id": s2.id,
                "llm_quality_score": 2,
                "success_label": "unsuccessful",
                "llm_disposition": "refused",
                "llm_issue_categories": ["refusal"],
            },
            analyzer_version=1,
            source_updated_at=None,
            message_count=2,
        )
        return test_db, s1.id, s2.id, s3.id

    def test_analysis_fields_joined(self, analyzed_db):
        db, s1, s2, s3 = analyzed_db
        sessions, total = db.list_all_sessions()
        assert total == 3
        by_id = {s.id: s for s in sessions}
        assert by_id[s1].disposition == "answered"
        assert by_id[s1].llm_quality_score == 5
        assert by_id[s1].success_label == "successful"
        assert by_id[s1].issue_count == 2
        assert set(by_id[s1].issue_categories) == {"hallucination", "tone"}

    def test_unanalyzed_session_has_null_analysis(self, analyzed_db):
        db, s1, s2, s3 = analyzed_db
        sessions, _ = db.list_all_sessions()
        by_id = {s.id: s for s in sessions}
        assert by_id[s3].disposition is None
        assert by_id[s3].llm_quality_score is None
        assert by_id[s3].success_label is None
        assert by_id[s3].issue_count == 0
        assert by_id[s3].issue_categories == []

    def test_filter_disposition(self, analyzed_db):
        db, s1, s2, s3 = analyzed_db
        sessions, total = db.list_all_sessions(disposition="answered")
        assert total == 1
        assert sessions[0].id == s1

    def test_filter_rating(self, analyzed_db):
        db, s1, s2, s3 = analyzed_db
        sessions, total = db.list_all_sessions(rating=2)
        assert total == 1
        assert sessions[0].id == s2

    def test_filter_success_label(self, analyzed_db):
        db, s1, s2, s3 = analyzed_db
        sessions, total = db.list_all_sessions(success_label="successful")
        assert total == 1
        assert sessions[0].id == s1

    def test_filter_min_issues(self, analyzed_db):
        db, s1, s2, s3 = analyzed_db
        # >=2 keeps only s1 (2 issues); s2 has 1, s3 has 0
        sessions, total = db.list_all_sessions(min_issues=2)
        assert total == 1
        assert sessions[0].id == s1
        # >=1 keeps s1 and s2
        sessions, total = db.list_all_sessions(min_issues=1)
        assert total == 2

    def test_filter_total_respects_pagination(self, analyzed_db):
        """The total must reflect filters even when a page is sliced."""
        db, s1, s2, s3 = analyzed_db
        sessions, total = db.list_all_sessions(min_issues=1, limit=1, offset=0)
        assert total == 2
        assert len(sessions) == 1

    def test_filters_compose_with_user_filter(self, analyzed_db):
        db, s1, s2, s3 = analyzed_db
        sessions, total = db.list_all_sessions(
            user_filter="alice", success_label="successful"
        )
        assert total == 1
        assert sessions[0].id == s1
        # composing with a non-matching user yields nothing
        _, total = db.list_all_sessions(
            user_filter="bob", success_label="successful"
        )
        assert total == 0

    def test_list_all_analysis_rows(self, analyzed_db):
        db, s1, s2, s3 = analyzed_db
        rows = db.list_all_analysis_rows()
        # only analyzed sessions appear
        assert {r["session_id"] for r in rows} == {s1, s2}
        by_id = {r["session_id"]: r for r in rows}
        assert by_id[s1]["llm_quality_score"] == 5
        assert by_id[s1]["llm_disposition"] == "answered"
        assert by_id[s1]["success_label"] == "successful"
        assert set(by_id[s1]["issue_categories"]) == {"hallucination", "tone"}
        assert by_id[s2]["issue_categories"] == ["refusal"]
        # ordered by created_at ascending
        assert "created_at" in rows[0]
