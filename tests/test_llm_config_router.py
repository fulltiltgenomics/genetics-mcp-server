"""Integration tests for LLM config router endpoints."""

import os
import tempfile
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from genetics_mcp_server.chat_api import app
from genetics_mcp_server.auth import auth_required
from genetics_mcp_server.db.llm_config_db import LLMConfigDB
from genetics_mcp_server.db.singleton import Singleton


@pytest.fixture
def test_db():
    """Create a temporary database for testing."""
    # clear singleton to allow fresh instance
    if LLMConfigDB in Singleton._instances:
        del Singleton._instances[LLMConfigDB]

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    db = LLMConfigDB(db_path)
    yield db

    if LLMConfigDB in Singleton._instances:
        del Singleton._instances[LLMConfigDB]
    os.unlink(db_path)


@pytest.fixture
def client_with_auth(test_db):
    """Create a test client with mocked authentication and database."""
    async def mock_auth():
        return "test@example.com"

    app.dependency_overrides[auth_required] = mock_auth

    # patch get_llm_config_db to return our test database
    with patch("genetics_mcp_server.routers.llm_config.get_llm_config_db", return_value=test_db):
        with TestClient(app) as client:
            yield client

    app.dependency_overrides.clear()


class TestDefaultsEndpoint:
    """Tests for /chat/v1/llm-config/defaults endpoint."""

    def test_get_defaults(self, client_with_auth):
        """Test getting default tool descriptions."""
        response = client_with_auth.get("/chat/v1/llm-config/defaults")

        assert response.status_code == 200
        data = response.json()
        assert "tool_descriptions" in data
        assert isinstance(data["tool_descriptions"], list)
        assert len(data["tool_descriptions"]) > 0

    def test_default_tool_structure(self, client_with_auth):
        """Test that default tools have correct structure."""
        response = client_with_auth.get("/chat/v1/llm-config/defaults")

        data = response.json()
        tool = data["tool_descriptions"][0]
        assert "tool_name" in tool
        assert "description" in tool


class TestUserCommentEndpoints:
    """Tests for user comment endpoints."""

    def test_get_user_comments_empty(self, client_with_auth):
        """Test getting comments when none exist."""
        response = client_with_auth.get("/chat/v1/llm-config/user/comments")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 0

    def test_add_user_comment(self, client_with_auth):
        """Test adding a comment."""
        response = client_with_auth.post(
            "/chat/v1/llm-config/user/comments",
            json={"comment": "This is a note about my preferences."},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["comment"] == "This is a note about my preferences."
        assert "id" in data
        assert "created_at" in data

    def test_get_user_comments(self, client_with_auth):
        """Test getting saved comments."""
        import time

        # add comments
        client_with_auth.post(
            "/chat/v1/llm-config/user/comments",
            json={"comment": "First comment"},
        )
        # wait for different timestamp
        time.sleep(1.1)
        client_with_auth.post(
            "/chat/v1/llm-config/user/comments",
            json={"comment": "Second comment"},
        )

        response = client_with_auth.get("/chat/v1/llm-config/user/comments")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        # newest first
        assert data[0]["comment"] == "Second comment"
        assert data[1]["comment"] == "First comment"

    def test_add_user_comment_empty_rejected(self, client_with_auth):
        """Test that empty comment is rejected."""
        response = client_with_auth.post(
            "/chat/v1/llm-config/user/comments",
            json={"comment": "   "},
        )

        assert response.status_code == 400

    def test_delete_user_comment(self, client_with_auth):
        """Test deleting a comment."""
        # add comment first
        add_resp = client_with_auth.post(
            "/chat/v1/llm-config/user/comments",
            json={"comment": "To be deleted"},
        )
        comment_id = add_resp.json()["id"]

        response = client_with_auth.delete(
            f"/chat/v1/llm-config/user/comments/{comment_id}"
        )

        assert response.status_code == 200
        assert response.json()["deleted"] is True

        # verify comment is gone
        get_resp = client_with_auth.get("/chat/v1/llm-config/user/comments")
        assert len(get_resp.json()) == 0

    def test_delete_user_comment_not_found(self, client_with_auth):
        """Test deleting a non-existent comment."""
        response = client_with_auth.delete("/chat/v1/llm-config/user/comments/99999")

        assert response.status_code == 404


class TestGlobalToolDescriptionEndpoints:
    """Tests for global tool description endpoints (legacy)."""

    def test_get_tool_descriptions_empty(self, client_with_auth):
        """Test getting tool descriptions when none are saved."""
        response = client_with_auth.get("/chat/v1/llm-config/tool-descriptions")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)

    def test_update_tool_description(self, client_with_auth):
        """Test updating a tool description globally."""
        response = client_with_auth.put(
            "/chat/v1/llm-config/tool-descriptions/search_genes",
            json={
                "description": "Global custom description",
                "comment": "Global change",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["tool_name"] == "search_genes"
        assert data["description"] == "Global custom description"
        assert data["changed_by"] == "test@example.com"

    def test_get_tool_description(self, client_with_auth):
        """Test getting a specific tool description."""
        # save first
        client_with_auth.put(
            "/chat/v1/llm-config/tool-descriptions/search_genes",
            json={"description": "Custom description"},
        )

        response = client_with_auth.get(
            "/chat/v1/llm-config/tool-descriptions/search_genes"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["description"] == "Custom description"

    def test_get_tool_description_not_found(self, client_with_auth):
        """Test getting a tool description that hasn't been customized."""
        response = client_with_auth.get(
            "/chat/v1/llm-config/tool-descriptions/nonexistent_tool"
        )

        assert response.status_code == 200
        assert response.json() is None

    def test_get_tool_description_history(self, client_with_auth):
        """Test getting tool description change history."""
        import time

        # make multiple changes
        client_with_auth.put(
            "/chat/v1/llm-config/tool-descriptions/search_genes",
            json={"description": "Version 1"},
        )
        # wait for different timestamp
        time.sleep(1.1)
        client_with_auth.put(
            "/chat/v1/llm-config/tool-descriptions/search_genes",
            json={"description": "Version 2"},
        )

        response = client_with_auth.get(
            "/chat/v1/llm-config/tool-descriptions/search_genes/history"
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["description"] == "Version 2"
        assert data[1]["description"] == "Version 1"

    def test_update_tool_description_empty_rejected(self, client_with_auth):
        """Test that empty description is rejected."""
        response = client_with_auth.put(
            "/chat/v1/llm-config/tool-descriptions/search_genes",
            json={"description": "  "},
        )

        assert response.status_code == 400
