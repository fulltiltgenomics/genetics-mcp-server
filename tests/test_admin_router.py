"""Integration tests for admin router endpoints."""

import os
import tempfile
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from genetics_mcp_server.auth import admin_required, auth_required
from genetics_mcp_server.chat_api import app
from genetics_mcp_server.db.chat_history_db import ChatHistoryDB
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
