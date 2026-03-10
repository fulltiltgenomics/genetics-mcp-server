"""Integration tests for chat history router endpoints."""

import os
import tempfile
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from genetics_mcp_server.chat_api import app
from genetics_mcp_server.auth import auth_required
from genetics_mcp_server.db.chat_history_db import ChatHistoryDB
from genetics_mcp_server.db.singleton import Singleton


@pytest.fixture
def test_db():
    """Create a temporary database for testing."""
    # clear singleton to allow fresh instance
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
def client_with_auth(test_db):
    """Create a test client with mocked authentication and database."""
    # override auth dependency to return a test user
    async def mock_auth():
        return "test@example.com"

    app.dependency_overrides[auth_required] = mock_auth

    # patch get_chat_history_db to return our test database
    with patch("genetics_mcp_server.routers.chat_history.get_chat_history_db", return_value=test_db):
        with TestClient(app) as client:
            yield client

    # cleanup overrides
    app.dependency_overrides.clear()


class TestSessionEndpoints:
    """Tests for chat session endpoints."""

    def test_create_session(self, client_with_auth):
        """Test creating a new chat session."""
        response = client_with_auth.post(
            "/chat/v1/chat/sessions",
            json={},
        )

        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "created_at" in data

    def test_create_session_with_phenotype(self, client_with_auth):
        """Test creating a session with phenotype code."""
        response = client_with_auth.post(
            "/chat/v1/chat/sessions",
            json={"phenotype_code": "T2D"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "id" in data

    def test_list_sessions_empty(self, client_with_auth):
        """Test listing sessions when none exist."""
        response = client_with_auth.get("/chat/v1/chat/sessions")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 0

    def test_list_sessions(self, client_with_auth):
        """Test listing sessions after creating some."""
        # create sessions
        client_with_auth.post("/chat/v1/chat/sessions", json={})
        client_with_auth.post("/chat/v1/chat/sessions", json={})

        response = client_with_auth.get("/chat/v1/chat/sessions")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

    def test_list_sessions_with_limit(self, client_with_auth):
        """Test limiting session list results."""
        for _ in range(5):
            client_with_auth.post("/chat/v1/chat/sessions", json={})

        response = client_with_auth.get("/chat/v1/chat/sessions?limit=3")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3

    def test_get_session(self, client_with_auth):
        """Test getting a specific session."""
        create_resp = client_with_auth.post("/chat/v1/chat/sessions", json={})
        session_id = create_resp.json()["id"]

        response = client_with_auth.get(f"/chat/v1/chat/sessions/{session_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == session_id
        assert "messages" in data

    def test_get_session_not_found(self, client_with_auth):
        """Test getting a non-existent session."""
        response = client_with_auth.get("/chat/v1/chat/sessions/nonexistent-id")

        assert response.status_code == 404

    def test_update_session_title(self, client_with_auth):
        """Test updating session title."""
        create_resp = client_with_auth.post("/chat/v1/chat/sessions", json={})
        session_id = create_resp.json()["id"]

        response = client_with_auth.put(
            f"/chat/v1/chat/sessions/{session_id}",
            json={"title": "Test Chat Title"},
        )

        assert response.status_code == 200
        assert response.json()["updated"] is True

        # verify the title was set
        get_resp = client_with_auth.get(f"/chat/v1/chat/sessions/{session_id}")
        assert get_resp.json()["title"] == "Test Chat Title"

    def test_update_session_rating(self, client_with_auth):
        """Test updating session rating."""
        create_resp = client_with_auth.post("/chat/v1/chat/sessions", json={})
        session_id = create_resp.json()["id"]

        response = client_with_auth.put(
            f"/chat/v1/chat/sessions/{session_id}",
            json={"rating": 5},
        )

        assert response.status_code == 200

    def test_update_session_not_found(self, client_with_auth):
        """Test updating a non-existent session."""
        response = client_with_auth.put(
            "/chat/v1/chat/sessions/nonexistent-id",
            json={"title": "Test"},
        )

        assert response.status_code == 404

    def test_delete_session(self, client_with_auth):
        """Test deleting a session."""
        create_resp = client_with_auth.post("/chat/v1/chat/sessions", json={})
        session_id = create_resp.json()["id"]

        response = client_with_auth.delete(f"/chat/v1/chat/sessions/{session_id}")

        assert response.status_code == 200
        assert response.json()["deleted"] is True

        # verify the session is gone
        get_resp = client_with_auth.get(f"/chat/v1/chat/sessions/{session_id}")
        assert get_resp.status_code == 404

    def test_delete_session_not_found(self, client_with_auth):
        """Test deleting a non-existent session."""
        response = client_with_auth.delete("/chat/v1/chat/sessions/nonexistent-id")

        assert response.status_code == 404


class TestMessageEndpoints:
    """Tests for chat message endpoints."""

    def test_save_message(self, client_with_auth):
        """Test saving a message to a session."""
        create_resp = client_with_auth.post("/chat/v1/chat/sessions", json={})
        session_id = create_resp.json()["id"]

        response = client_with_auth.post(
            f"/chat/v1/chat/sessions/{session_id}/messages",
            json={
                "id": "msg-1",
                "role": "user",
                "content": "Hello, world!",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "msg-1"
        assert data["role"] == "user"
        assert data["content"] == "Hello, world!"

    def test_save_message_with_content_json(self, client_with_auth):
        """Test saving a message with JSON content blocks."""
        create_resp = client_with_auth.post("/chat/v1/chat/sessions", json={})
        session_id = create_resp.json()["id"]

        content_json = '[{"type": "text", "text": "Hello"}]'
        response = client_with_auth.post(
            f"/chat/v1/chat/sessions/{session_id}/messages",
            json={
                "id": "msg-1",
                "role": "assistant",
                "content": "Hello",
                "content_json": content_json,
            },
        )

        assert response.status_code == 200
        assert response.json()["content_json"] == content_json

    def test_save_message_invalid_role(self, client_with_auth):
        """Test that invalid role is rejected."""
        create_resp = client_with_auth.post("/chat/v1/chat/sessions", json={})
        session_id = create_resp.json()["id"]

        response = client_with_auth.post(
            f"/chat/v1/chat/sessions/{session_id}/messages",
            json={
                "id": "msg-1",
                "role": "system",
                "content": "Hello",
            },
        )

        assert response.status_code == 400

    def test_save_message_session_not_found(self, client_with_auth):
        """Test saving message to non-existent session."""
        response = client_with_auth.post(
            "/chat/v1/chat/sessions/nonexistent-id/messages",
            json={
                "id": "msg-1",
                "role": "user",
                "content": "Hello",
            },
        )

        assert response.status_code == 404

    def test_get_session_messages(self, client_with_auth):
        """Test retrieving messages from a session."""
        create_resp = client_with_auth.post("/chat/v1/chat/sessions", json={})
        session_id = create_resp.json()["id"]

        # add messages
        client_with_auth.post(
            f"/chat/v1/chat/sessions/{session_id}/messages",
            json={"id": "msg-1", "role": "user", "content": "Hello"},
        )
        client_with_auth.post(
            f"/chat/v1/chat/sessions/{session_id}/messages",
            json={"id": "msg-2", "role": "assistant", "content": "Hi there!"},
        )

        response = client_with_auth.get(f"/chat/v1/chat/sessions/{session_id}")

        assert response.status_code == 200
        messages = response.json()["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_rate_message_thumbs_up(self, client_with_auth):
        """Test rating a message with thumbs up."""
        create_resp = client_with_auth.post("/chat/v1/chat/sessions", json={})
        session_id = create_resp.json()["id"]

        client_with_auth.post(
            f"/chat/v1/chat/sessions/{session_id}/messages",
            json={"id": "msg-1", "role": "assistant", "content": "Response"},
        )

        response = client_with_auth.put(
            "/chat/v1/chat/messages/msg-1/rating",
            json={"thumbs_up": True},
        )

        assert response.status_code == 200
        assert response.json()["updated"] is True

    def test_rate_message_thumbs_down(self, client_with_auth):
        """Test rating a message with thumbs down."""
        create_resp = client_with_auth.post("/chat/v1/chat/sessions", json={})
        session_id = create_resp.json()["id"]

        client_with_auth.post(
            f"/chat/v1/chat/sessions/{session_id}/messages",
            json={"id": "msg-1", "role": "assistant", "content": "Response"},
        )

        response = client_with_auth.put(
            "/chat/v1/chat/messages/msg-1/rating",
            json={"thumbs_up": False},
        )

        assert response.status_code == 200

    def test_rate_message_clear(self, client_with_auth):
        """Test clearing a message rating."""
        create_resp = client_with_auth.post("/chat/v1/chat/sessions", json={})
        session_id = create_resp.json()["id"]

        client_with_auth.post(
            f"/chat/v1/chat/sessions/{session_id}/messages",
            json={"id": "msg-1", "role": "assistant", "content": "Response"},
        )

        response = client_with_auth.put(
            "/chat/v1/chat/messages/msg-1/rating",
            json={"thumbs_up": None},
        )

        assert response.status_code == 200

    def test_rate_message_not_found(self, client_with_auth):
        """Test rating a non-existent message."""
        response = client_with_auth.put(
            "/chat/v1/chat/messages/nonexistent/rating",
            json={"thumbs_up": True},
        )

        assert response.status_code == 404


class TestSessionPreview:
    """Tests for session preview functionality."""

    def test_session_preview_from_first_message(self, client_with_auth):
        """Test that session preview is generated from first user message."""
        create_resp = client_with_auth.post("/chat/v1/chat/sessions", json={})
        session_id = create_resp.json()["id"]

        # add a user message
        client_with_auth.post(
            f"/chat/v1/chat/sessions/{session_id}/messages",
            json={"id": "msg-1", "role": "user", "content": "What is APOE?"},
        )

        # list sessions to get preview
        response = client_with_auth.get("/chat/v1/chat/sessions")

        assert response.status_code == 200
        sessions = response.json()
        assert len(sessions) == 1
        assert sessions[0]["preview"] == "What is APOE?"

    def test_session_title_overrides_preview(self, client_with_auth):
        """Test that explicit title overrides preview."""
        create_resp = client_with_auth.post("/chat/v1/chat/sessions", json={})
        session_id = create_resp.json()["id"]

        # add message and set title
        client_with_auth.post(
            f"/chat/v1/chat/sessions/{session_id}/messages",
            json={"id": "msg-1", "role": "user", "content": "What is APOE?"},
        )
        client_with_auth.put(
            f"/chat/v1/chat/sessions/{session_id}",
            json={"title": "APOE Discussion"},
        )

        # list sessions
        response = client_with_auth.get("/chat/v1/chat/sessions")

        assert response.status_code == 200
        sessions = response.json()
        assert sessions[0]["title"] == "APOE Discussion"
        assert sessions[0]["preview"] is None
