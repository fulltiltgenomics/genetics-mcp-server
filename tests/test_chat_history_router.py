"""Integration tests for chat history router endpoints."""

import os
import tempfile
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from genetics_mcp_server.auth import auth_required
from genetics_mcp_server.chat_api import app
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

    def test_save_message_with_tool_results_json(self, client_with_auth):
        """Test that tool_results_json round-trips through save and get."""
        create_resp = client_with_auth.post("/chat/v1/chat/sessions", json={})
        session_id = create_resp.json()["id"]

        content_json = '[{"type": "tool_use", "id": "tu_1", "name": "x", "input": {}}]'
        tool_results_json = (
            '[{"type": "tool_result", "tool_use_id": "tu_1", "content": "{\\"rows\\": 5}"}]'
        )
        response = client_with_auth.post(
            f"/chat/v1/chat/sessions/{session_id}/messages",
            json={
                "id": "msg-1",
                "role": "assistant",
                "content": "Done",
                "content_json": content_json,
                "tool_results_json": tool_results_json,
            },
        )

        assert response.status_code == 200
        assert response.json()["tool_results_json"] == tool_results_json

        # round-trips through GET session detail
        detail = client_with_auth.get(f"/chat/v1/chat/sessions/{session_id}")
        assert detail.status_code == 200
        msg = detail.json()["messages"][0]
        assert msg["tool_results_json"] == tool_results_json

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


class TestShareAndForkEndpoints:
    """Tests for session sharing and forking endpoints."""

    @pytest.fixture
    def two_user_clients(self, test_db):
        """Create two test clients with different authenticated users."""
        # owner client
        async def mock_auth_owner():
            return "owner@example.com"

        app.dependency_overrides[auth_required] = mock_auth_owner

        with patch("genetics_mcp_server.routers.chat_history.get_chat_history_db", return_value=test_db):
            with TestClient(app) as owner_client:
                # create a session with a message as owner
                create_resp = owner_client.post("/chat/v1/chat/sessions", json={})
                session_id = create_resp.json()["id"]
                owner_client.post(
                    f"/chat/v1/chat/sessions/{session_id}/messages",
                    json={"id": "msg-1", "role": "user", "content": "Hello from owner"},
                )

        app.dependency_overrides.clear()

        yield test_db, session_id

    def _make_client(self, test_db, user_email):
        """Helper to create a test client for a specific user."""
        async def mock_auth():
            return user_email

        app.dependency_overrides[auth_required] = mock_auth

        return patch("genetics_mcp_server.routers.chat_history.get_chat_history_db", return_value=test_db)

    def test_owner_can_share_session(self, two_user_clients):
        """Test that the session owner can share a session."""
        test_db, session_id = two_user_clients

        with self._make_client(test_db, "owner@example.com"):
            with TestClient(app) as client:
                response = client.put(
                    f"/chat/v1/chat/sessions/{session_id}/share",
                    json={"shared": True},
                )

        app.dependency_overrides.clear()

        assert response.status_code == 200
        assert response.json()["shared"] is True

    def test_non_owner_cannot_share_session(self, two_user_clients):
        """Test that a non-owner gets 403 when trying to share."""
        test_db, session_id = two_user_clients

        with self._make_client(test_db, "other@example.com"):
            with TestClient(app) as client:
                response = client.put(
                    f"/chat/v1/chat/sessions/{session_id}/share",
                    json={"shared": True},
                )

        app.dependency_overrides.clear()

        assert response.status_code == 403

    def test_non_owner_can_access_shared_session(self, two_user_clients):
        """Test that a non-owner can access a shared session via GET."""
        test_db, session_id = two_user_clients

        # owner shares the session
        with self._make_client(test_db, "owner@example.com"):
            with TestClient(app) as client:
                client.put(
                    f"/chat/v1/chat/sessions/{session_id}/share",
                    json={"shared": True},
                )
        app.dependency_overrides.clear()

        # non-owner accesses the shared session
        with self._make_client(test_db, "other@example.com"):
            with TestClient(app) as client:
                response = client.get(f"/chat/v1/chat/sessions/{session_id}")
        app.dependency_overrides.clear()

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == session_id
        assert data["is_owner"] is False
        assert len(data["messages"]) == 1

    def test_non_owner_cannot_access_non_shared_session(self, two_user_clients):
        """Test that a non-owner gets 404 for a non-shared session."""
        test_db, session_id = two_user_clients

        with self._make_client(test_db, "other@example.com"):
            with TestClient(app) as client:
                response = client.get(f"/chat/v1/chat/sessions/{session_id}")
        app.dependency_overrides.clear()

        assert response.status_code == 404

    def test_fork_shared_session(self, two_user_clients):
        """Test that an authenticated user can fork a shared session."""
        test_db, session_id = two_user_clients

        # owner shares the session
        with self._make_client(test_db, "owner@example.com"):
            with TestClient(app) as client:
                client.put(
                    f"/chat/v1/chat/sessions/{session_id}/share",
                    json={"shared": True},
                )
        app.dependency_overrides.clear()

        # other user forks the session
        with self._make_client(test_db, "other@example.com"):
            with TestClient(app) as client:
                response = client.post(f"/chat/v1/chat/sessions/{session_id}/fork")
        app.dependency_overrides.clear()

        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["id"] != session_id

        # verify the forked session has copied messages
        with self._make_client(test_db, "other@example.com"):
            with TestClient(app) as client:
                get_resp = client.get(f"/chat/v1/chat/sessions/{data['id']}")
        app.dependency_overrides.clear()

        assert get_resp.status_code == 200
        forked = get_resp.json()
        assert forked["is_owner"] is True
        assert len(forked["messages"]) == 1
        assert forked["messages"][0]["content"] == "Hello from owner"

    def test_fork_non_shared_session_fails(self, two_user_clients):
        """Test that forking a non-shared session returns 404."""
        test_db, session_id = two_user_clients

        with self._make_client(test_db, "other@example.com"):
            with TestClient(app) as client:
                response = client.post(f"/chat/v1/chat/sessions/{session_id}/fork")
        app.dependency_overrides.clear()

        assert response.status_code == 404

    def test_unshare_blocks_non_owner_access(self, two_user_clients):
        """Test that after unsharing, non-owner gets 404."""
        test_db, session_id = two_user_clients

        # owner shares the session
        with self._make_client(test_db, "owner@example.com"):
            with TestClient(app) as client:
                client.put(
                    f"/chat/v1/chat/sessions/{session_id}/share",
                    json={"shared": True},
                )
        app.dependency_overrides.clear()

        # verify non-owner can access
        with self._make_client(test_db, "other@example.com"):
            with TestClient(app) as client:
                response = client.get(f"/chat/v1/chat/sessions/{session_id}")
        app.dependency_overrides.clear()
        assert response.status_code == 200

        # owner unshares the session
        with self._make_client(test_db, "owner@example.com"):
            with TestClient(app) as client:
                client.put(
                    f"/chat/v1/chat/sessions/{session_id}/share",
                    json={"shared": False},
                )
        app.dependency_overrides.clear()

        # non-owner can no longer access
        with self._make_client(test_db, "other@example.com"):
            with TestClient(app) as client:
                response = client.get(f"/chat/v1/chat/sessions/{session_id}")
        app.dependency_overrides.clear()

        assert response.status_code == 404


def _make_xlsx(sheets: dict[str, list[list]]) -> bytes:
    """Build an in-memory .xlsx from {sheet_name: rows} for tests."""
    import io

    import xlsxwriter

    buf = io.BytesIO()
    wb = xlsxwriter.Workbook(buf, {"in_memory": True})
    for name, rows in sheets.items():
        ws = wb.add_worksheet(name)
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                ws.write(r, c, val)
    wb.close()
    return buf.getvalue()


_XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


class TestAttachmentEndpoints:
    """Tests for file attachment upload/download, focused on Excel parsing."""

    @pytest.fixture
    def tmp_storage(self, tmp_path, monkeypatch):
        # redirect attachment storage to a temp dir for the duration of a test
        from genetics_mcp_server.config import get_settings

        settings = get_settings()
        monkeypatch.setattr(settings, "attachment_storage_path", str(tmp_path))
        return tmp_path

    def _create_session(self, client) -> str:
        return client.post("/chat/v1/chat/sessions", json={}).json()["id"]

    def test_excel_upload_parses_to_tsv(self, client_with_auth, tmp_storage):
        session_id = self._create_session(client_with_auth)
        xlsx = _make_xlsx({"results": [["gene", "trait"], ["BRCA1", "cancer"], ["TP53", "cancer"]]})

        resp = client_with_auth.post(
            f"/chat/v1/chat/sessions/{session_id}/attachments",
            files={"file": ("data.xlsx", xlsx, _XLSX_MIME)},
        )
        assert resp.status_code == 200
        att = resp.json()
        assert att["type"] == "excel"

        # raw download still returns the original spreadsheet bytes
        raw = client_with_auth.get(
            f"/chat/v1/chat/sessions/{session_id}/attachments/{att['id']}"
        )
        assert raw.status_code == 200
        assert raw.content == xlsx

        # ?as=text returns the model-ready parsed TSV
        text = client_with_auth.get(
            f"/chat/v1/chat/sessions/{session_id}/attachments/{att['id']}?as=text"
        )
        assert text.status_code == 200
        assert "gene\ttrait" in text.text
        assert "BRCA1\tcancer" in text.text

    def test_multi_sheet_excel_includes_sheet_headers(self, client_with_auth, tmp_storage):
        session_id = self._create_session(client_with_auth)
        xlsx = _make_xlsx(
            {
                "genes": [["gene"], ["BRCA1"]],
                "variants": [["rsid"], ["rs123"]],
            }
        )
        att = client_with_auth.post(
            f"/chat/v1/chat/sessions/{session_id}/attachments",
            files={"file": ("multi.xlsx", xlsx, _XLSX_MIME)},
        ).json()

        text = client_with_auth.get(
            f"/chat/v1/chat/sessions/{session_id}/attachments/{att['id']}?as=text"
        ).text
        assert "# Sheet: genes" in text
        assert "# Sheet: variants" in text

    def test_invalid_excel_rejected_and_nothing_stored(self, client_with_auth, tmp_storage):
        session_id = self._create_session(client_with_auth)
        resp = client_with_auth.post(
            f"/chat/v1/chat/sessions/{session_id}/attachments",
            files={"file": ("bad.xlsx", b"this is not a spreadsheet", _XLSX_MIME)},
        )
        assert resp.status_code == 400
        # the bad upload left no files behind
        assert not any((tmp_storage / session_id).glob("*")) if (tmp_storage / session_id).exists() else True

    def test_tsv_as_text_returns_original(self, client_with_auth, tmp_storage):
        session_id = self._create_session(client_with_auth)
        resp = client_with_auth.post(
            f"/chat/v1/chat/sessions/{session_id}/attachments",
            files={"file": ("data.tsv", b"a\tb\n1\t2\n", "text/tab-separated-values")},
        )
        att = resp.json()
        assert att["type"] == "tsv"
        text = client_with_auth.get(
            f"/chat/v1/chat/sessions/{session_id}/attachments/{att['id']}?as=text"
        )
        assert text.status_code == 200
        assert text.text == "a\tb\n1\t2\n"
