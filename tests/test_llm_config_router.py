"""Integration tests for LLM config router endpoints."""

import tempfile
from unittest.mock import patch

import pytest
from conftest import close_and_unlink
from fastapi.testclient import TestClient

from genetics_mcp_server.auth import auth_required
from genetics_mcp_server.chat_api import app
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
    close_and_unlink(db, db_path)


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

    def test_tool_description_history_limit_is_validated(self, client_with_auth):
        """Same unbounded/overflow hole as the instruction-set history endpoint."""
        url = "/chat/v1/llm-config/tool-descriptions/search_genes/history"
        client_with_auth.put(
            "/chat/v1/llm-config/tool-descriptions/search_genes",
            json={"description": "Version 1"},
        )

        assert client_with_auth.get(f"{url}?limit=-1").status_code == 422
        assert client_with_auth.get(f"{url}?limit=0").status_code == 422
        assert client_with_auth.get(f"{url}?limit=999999999999999999999").status_code == 422
        assert client_with_auth.get(f"{url}?limit=101").status_code == 422
        assert len(client_with_auth.get(f"{url}?limit=1").json()) == 1
        assert client_with_auth.get(f"{url}?limit=100").status_code == 200

    def test_update_tool_description_empty_rejected(self, client_with_auth):
        """Test that empty description is rejected."""
        response = client_with_auth.put(
            "/chat/v1/llm-config/tool-descriptions/search_genes",
            json={"description": "  "},
        )

        assert response.status_code == 400


@pytest.fixture
def multi_user_client(test_db):
    """Test client whose authenticated user can be switched mid-test.

    Cross-user scoping is the whole point of these endpoints, so it needs two identities
    against one database rather than two databases.
    """
    current = {"user": "alice@example.com"}

    async def mock_auth():
        return current["user"]

    app.dependency_overrides[auth_required] = mock_auth

    with patch("genetics_mcp_server.routers.llm_config.get_llm_config_db", return_value=test_db):
        with TestClient(app) as client:
            yield client, current

    app.dependency_overrides.clear()


BASE = "/chat/v1/llm-config/user/instruction-sets"


def _create(client, name="Statistician", body="Answer like a statistician."):
    return client.post(BASE, json={"name": name, "body": body})


class TestInstructionSetCrud:
    """Tests for the instruction set CRUD endpoints."""

    def test_list_empty(self, client_with_auth):
        resp = client_with_auth.get(BASE)
        assert resp.status_code == 200
        assert resp.json() == []

    def test_create_and_list(self, client_with_auth):
        created = _create(client_with_auth)
        assert created.status_code == 200
        data = created.json()
        assert data["name"] == "Statistician"
        assert data["body"] == "Answer like a statistician."
        assert data["body_over_cap"] is False
        assert data["id"]

        listed = client_with_auth.get(BASE).json()
        assert [s["id"] for s in listed] == [data["id"]]

    def test_create_strips_the_name(self, client_with_auth):
        assert _create(client_with_auth, name="  Padded  ").json()["name"] == "Padded"

    def test_create_empty_name_rejected(self, client_with_auth):
        assert _create(client_with_auth, name="   ").status_code == 400

    def test_create_empty_body_rejected(self, client_with_auth):
        assert _create(client_with_auth, body="   ").status_code == 400

    def test_create_over_body_cap_is_413(self, client_with_auth):
        from genetics_mcp_server.db.llm_config_db import INSTRUCTION_SET_MAX_BODY_CHARS

        resp = _create(client_with_auth, body="x" * (INSTRUCTION_SET_MAX_BODY_CHARS + 1))
        assert resp.status_code == 413

    def test_create_over_count_cap_is_409(self, client_with_auth, test_db):
        from genetics_mcp_server.db.llm_config_db import INSTRUCTION_SET_MAX_PER_USER

        for i in range(INSTRUCTION_SET_MAX_PER_USER):
            test_db.create_instruction_set("test@example.com", f"set {i}", "body")

        assert _create(client_with_auth).status_code == 409

    def test_over_cap_body_is_flagged_on_read(self, client_with_auth, test_db):
        """A set stored before the cap existed must be reported, not hidden or truncated."""
        from genetics_mcp_server.db.llm_config_db import INSTRUCTION_SET_MAX_BODY_CHARS

        long_body = "y" * (INSTRUCTION_SET_MAX_BODY_CHARS + 10)
        test_db._conn.execute(
            "INSERT INTO user_instruction_sets (id, user_id, name, body) VALUES (?, ?, ?, ?)",
            ("legacy-set", "test@example.com", "Legacy", long_body),
        )
        test_db._conn.commit()

        listed = client_with_auth.get(BASE).json()
        assert listed[0]["body_over_cap"] is True
        assert len(listed[0]["body"]) == len(long_body)

    def test_update_name_and_body(self, client_with_auth):
        set_id = _create(client_with_auth).json()["id"]

        resp = client_with_auth.put(
            f"{BASE}/{set_id}", json={"name": "Terse", "body": "Be terse."}
        )
        assert resp.status_code == 200
        assert resp.json()["name"] == "Terse"
        assert resp.json()["body"] == "Be terse."

    def test_update_keeps_omitted_fields(self, client_with_auth):
        set_id = _create(client_with_auth).json()["id"]

        resp = client_with_auth.put(f"{BASE}/{set_id}", json={"name": "Renamed"})
        assert resp.status_code == 200
        assert resp.json()["body"] == "Answer like a statistician."

    def test_update_empty_name_rejected(self, client_with_auth):
        set_id = _create(client_with_auth).json()["id"]
        assert client_with_auth.put(f"{BASE}/{set_id}", json={"name": " "}).status_code == 400

    def test_update_empty_body_rejected(self, client_with_auth):
        set_id = _create(client_with_auth).json()["id"]
        assert client_with_auth.put(f"{BASE}/{set_id}", json={"body": " "}).status_code == 400

    def test_update_over_body_cap_is_413(self, client_with_auth):
        from genetics_mcp_server.db.llm_config_db import INSTRUCTION_SET_MAX_BODY_CHARS

        set_id = _create(client_with_auth).json()["id"]
        resp = client_with_auth.put(
            f"{BASE}/{set_id}", json={"body": "x" * (INSTRUCTION_SET_MAX_BODY_CHARS + 1)}
        )
        assert resp.status_code == 413

    def test_update_unknown_id_is_404(self, client_with_auth):
        assert client_with_auth.put(f"{BASE}/nope", json={"name": "x"}).status_code == 404

    def test_delete_archives(self, client_with_auth):
        set_id = _create(client_with_auth).json()["id"]

        resp = client_with_auth.delete(f"{BASE}/{set_id}")
        assert resp.status_code == 204
        assert client_with_auth.get(BASE).json() == []

    def test_delete_twice_is_404(self, client_with_auth):
        set_id = _create(client_with_auth).json()["id"]
        client_with_auth.delete(f"{BASE}/{set_id}")
        assert client_with_auth.delete(f"{BASE}/{set_id}").status_code == 404

    def test_delete_unknown_id_is_404(self, client_with_auth):
        assert client_with_auth.delete(f"{BASE}/nope").status_code == 404

    def test_update_after_delete_is_404(self, client_with_auth):
        """An archived set is deleted as far as the user is concerned: editing must not 200."""
        set_id = _create(client_with_auth).json()["id"]
        client_with_auth.delete(f"{BASE}/{set_id}")

        assert client_with_auth.put(
            f"{BASE}/{set_id}", json={"body": "Be terse."}
        ).status_code == 404


class TestInstructionSetHistory:
    def test_history_after_create_and_update(self, client_with_auth):
        set_id = _create(client_with_auth).json()["id"]
        client_with_auth.put(f"{BASE}/{set_id}", json={"body": "v2", "comment": "tighten"})

        resp = client_with_auth.get(f"{BASE}/{set_id}/history")
        assert resp.status_code == 200
        history = resp.json()
        # newest first even though both versions land in the same second
        assert [v["body"] for v in history] == ["v2", "Answer like a statistician."]
        assert history[0]["comment"] == "tighten"
        assert history[0]["set_id"] == set_id

    def test_history_limit(self, client_with_auth):
        set_id = _create(client_with_auth).json()["id"]
        for i in range(3):
            client_with_auth.put(f"{BASE}/{set_id}", json={"body": f"v{i}"})

        history = client_with_auth.get(f"{BASE}/{set_id}/history?limit=2").json()
        assert len(history) == 2
        assert history[0]["body"] == "v2"

    def test_history_survives_delete(self, client_with_auth):
        set_id = _create(client_with_auth).json()["id"]
        client_with_auth.delete(f"{BASE}/{set_id}")

        resp = client_with_auth.get(f"{BASE}/{set_id}/history")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_history_unknown_id_is_404(self, client_with_auth):
        assert client_with_auth.get(f"{BASE}/nope/history").status_code == 404

    def test_negative_limit_is_rejected(self, client_with_auth):
        """SQLite reads a negative LIMIT as unbounded, so it would dump the whole history."""
        set_id = _create(client_with_auth).json()["id"]
        client_with_auth.put(f"{BASE}/{set_id}", json={"body": "v2"})

        assert client_with_auth.get(f"{BASE}/{set_id}/history?limit=-1").status_code == 422
        assert client_with_auth.get(f"{BASE}/{set_id}/history?limit=0").status_code == 422

    def test_oversized_limit_is_rejected(self, client_with_auth):
        """An int wider than SQLite's INTEGER used to escape as an uncaught 500."""
        set_id = _create(client_with_auth).json()["id"]

        assert client_with_auth.get(
            f"{BASE}/{set_id}/history?limit=999999999999999999999"
        ).status_code == 422
        assert client_with_auth.get(f"{BASE}/{set_id}/history?limit=101").status_code == 422

    def test_limit_boundaries_are_accepted(self, client_with_auth):
        set_id = _create(client_with_auth).json()["id"]

        assert len(client_with_auth.get(f"{BASE}/{set_id}/history?limit=1").json()) == 1
        assert client_with_auth.get(f"{BASE}/{set_id}/history?limit=100").status_code == 200


class TestInstructionSetCrossUserScoping:
    """User B must not read, update, archive or inspect the history of user A's set."""

    def test_other_users_set_is_not_listed(self, multi_user_client):
        client, current = multi_user_client
        _create(client)

        current["user"] = "bob@example.com"
        assert client.get(BASE).json() == []

    def test_other_users_set_cannot_be_updated(self, multi_user_client):
        client, current = multi_user_client
        set_id = _create(client).json()["id"]

        current["user"] = "bob@example.com"
        assert client.put(f"{BASE}/{set_id}", json={"body": "mine now"}).status_code == 404

        current["user"] = "alice@example.com"
        assert client.get(BASE).json()[0]["body"] == "Answer like a statistician."

    def test_other_users_set_cannot_be_archived(self, multi_user_client):
        client, current = multi_user_client
        set_id = _create(client).json()["id"]

        current["user"] = "bob@example.com"
        assert client.delete(f"{BASE}/{set_id}").status_code == 404

        current["user"] = "alice@example.com"
        assert [s["id"] for s in client.get(BASE).json()] == [set_id]

    def test_other_users_history_is_not_readable(self, multi_user_client):
        client, current = multi_user_client
        set_id = _create(client).json()["id"]

        current["user"] = "bob@example.com"
        assert client.get(f"{BASE}/{set_id}/history").status_code == 404

    def test_count_cap_is_per_user(self, multi_user_client, test_db):
        from genetics_mcp_server.db.llm_config_db import INSTRUCTION_SET_MAX_PER_USER

        client, current = multi_user_client
        for i in range(INSTRUCTION_SET_MAX_PER_USER):
            test_db.create_instruction_set("alice@example.com", f"set {i}", "body")
        assert _create(client).status_code == 409

        current["user"] = "bob@example.com"
        assert _create(client).status_code == 200
