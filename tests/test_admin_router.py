"""Integration tests for admin router endpoints."""

import os
import tempfile
import time
from contextlib import contextmanager, nullcontext
from unittest.mock import patch

import pytest
from conftest import close_and_unlink, settings_env
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
    close_and_unlink(db, db_path)


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


@pytest.fixture
def analyzed_admin_client(test_db):
    """Admin client over a DB with analysis rows on two of three sessions.

    s1: rating 5, successful, answered, 2 issues (hallucination, tone)
    s2: rating 2, unsuccessful, refused, 1 issue (refusal)
    s3: unanalyzed (NA rating, no issues)
    """
    from genetics_mcp_server.config.settings import get_settings

    s1 = test_db.create_session("alice@example.com")
    s2 = test_db.create_session("bob@example.com")
    s3 = test_db.create_session("carol@example.com")
    for s in (s1, s2, s3):
        test_db.add_message(s.id, f"msg-{s.id}", "user", "hello")

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
        message_count=2,
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

    async def mock_admin():
        return "admin@example.com"

    app.dependency_overrides[admin_required] = mock_admin

    with patch.dict(os.environ, {"ENABLE_ADMIN_PAGE": "true"}):
        get_settings.cache_clear()
        with patch("genetics_mcp_server.routers.admin.get_chat_history_db", return_value=test_db):
            with TestClient(app) as client:
                yield client, s1.id, s2.id, s3.id
        get_settings.cache_clear()

    app.dependency_overrides.clear()


class TestAdminSessionsAnalysisFields:
    """HTTP-level coverage of the analysis fields + filters on /admin/sessions."""

    def test_analysis_fields_exposed(self, analyzed_admin_client):
        client, s1, s2, s3 = analyzed_admin_client
        data = client.get("/chat/v1/admin/sessions").json()
        by_id = {s["id"]: s for s in data["sessions"]}

        assert by_id[s1]["disposition"] == "answered"
        assert by_id[s1]["llm_rating"] == 5
        assert by_id[s1]["success_label"] == "successful"
        assert by_id[s1]["issue_count"] == 2
        assert set(by_id[s1]["issue_categories"]) == {"hallucination", "tone"}

        # unanalyzed session surfaces NA/empty defaults
        assert by_id[s3]["disposition"] is None
        assert by_id[s3]["llm_rating"] is None
        assert by_id[s3]["success_label"] is None
        assert by_id[s3]["issue_count"] == 0
        assert by_id[s3]["issue_categories"] == []

    def test_filter_disposition(self, analyzed_admin_client):
        client, s1, s2, s3 = analyzed_admin_client
        data = client.get("/chat/v1/admin/sessions?disposition=answered").json()
        assert data["total"] == 1
        assert data["sessions"][0]["id"] == s1

    def test_filter_success_label(self, analyzed_admin_client):
        client, s1, s2, s3 = analyzed_admin_client
        data = client.get("/chat/v1/admin/sessions?success_label=unsuccessful").json()
        assert data["total"] == 1
        assert data["sessions"][0]["id"] == s2

    def test_filter_min_issues(self, analyzed_admin_client):
        client, s1, s2, s3 = analyzed_admin_client
        assert client.get("/chat/v1/admin/sessions?min_issues=2").json()["total"] == 1
        assert client.get("/chat/v1/admin/sessions?min_issues=1").json()["total"] == 2

    def test_filter_rating_numeric(self, analyzed_admin_client):
        client, s1, s2, s3 = analyzed_admin_client
        data = client.get("/chat/v1/admin/sessions?rating=5").json()
        assert data["total"] == 1
        assert data["sessions"][0]["id"] == s1

    def test_filter_rating_na(self, analyzed_admin_client):
        """rating=NA filters to unrated sessions (s3 has no analysis row)."""
        client, s1, s2, s3 = analyzed_admin_client
        data = client.get("/chat/v1/admin/sessions?rating=NA").json()
        assert data["total"] == 1
        assert data["sessions"][0]["id"] == s3

    def test_filter_rating_invalid(self, analyzed_admin_client):
        client, s1, s2, s3 = analyzed_admin_client
        resp = client.get("/chat/v1/admin/sessions?rating=bogus")
        assert resp.status_code == 400


class TestAdminQualityAnalytics:

    def test_quality_rows(self, analyzed_admin_client):
        client, s1, s2, s3 = analyzed_admin_client
        resp = client.get("/chat/v1/admin/analytics/quality")
        assert resp.status_code == 200
        rows = resp.json()["rows"]
        # only analyzed sessions appear
        assert {r["session_id"] for r in rows} == {s1, s2}
        by_id = {r["session_id"]: r for r in rows}
        assert by_id[s1]["llm_quality_score"] == 5
        assert by_id[s1]["llm_disposition"] == "answered"
        assert by_id[s1]["success_label"] == "successful"
        assert set(by_id[s1]["issue_categories"]) == {"hallucination", "tone"}
        assert by_id[s2]["issue_categories"] == ["refusal"]
        assert "created_at" in by_id[s1]


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
        async def mock_auth():
            return "regular@example.com"

        app.dependency_overrides[auth_required] = mock_auth

        with patch("genetics_mcp_server.routers.admin.get_chat_history_db", return_value=seeded_db):
            with settings_env(
                REQUIRE_AUTH="true",
                ADMIN_USERS="admin@example.com",
                ENABLE_ADMIN_PAGE="true",
            ):
                with TestClient(app) as client:
                    response = client.get("/chat/v1/admin/sessions")
                    assert response.status_code == 403

        app.dependency_overrides.clear()

    def test_dev_mode_allows_any_user(self, seeded_db):
        """When REQUIRE_AUTH is false, any user can access admin."""
        with patch("genetics_mcp_server.routers.admin.get_chat_history_db", return_value=seeded_db):
            with settings_env(REQUIRE_AUTH="false", ENABLE_ADMIN_PAGE="true"):
                with TestClient(app) as client:
                    response = client.get("/chat/v1/admin/sessions")
                    assert response.status_code == 200

        app.dependency_overrides.clear()

    @pytest.mark.parametrize(
        "require_auth,user,expect_admin",
        [
            ("true", "regular@example.com", False),
            ("true", "admin@example.com", True),
            ("false", "regular@example.com", True),
        ],
    )
    def test_the_reported_is_admin_matches_the_gate_on_the_admin_endpoints(
        self, seeded_db, require_auth, user, expect_admin
    ):
        """/chat/v1/auth's is_admin is what the frontend shows the admin UI on, and
        admin_required is what actually guards the endpoints behind it. They used to read
        REQUIRE_AUTH from two places — a module global snapshotted at import time and a per
        request os.environ lookup — so moving one and not the other made them disagree: the
        REQUIRE_AUTH=true row below reported is_admin False for a regular user while the gate,
        still holding its import-time False, served that same user every admin endpoint
        (genetics-results-suite-pol).
        """
        headers = {"X-Goog-Authenticated-User-Email": f"accounts.google.com:{user}"}
        with patch("genetics_mcp_server.routers.admin.get_chat_history_db", return_value=seeded_db):
            with settings_env(
                REQUIRE_AUTH=require_auth,
                ADMIN_USERS="admin@example.com",
                ENABLE_ADMIN_PAGE="true",
            ):
                with TestClient(app) as client:
                    reported = client.get("/chat/v1/auth", headers=headers).json()["is_admin"]
                    gated = client.get("/chat/v1/admin/sessions", headers=headers)

        assert reported is expect_admin
        assert (gated.status_code == 200) is expect_admin

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

        config_db = None
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
            close_and_unlink(config_db, db_path)

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


@pytest.fixture
def feedback_client(test_db):
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
    close_and_unlink(config_db, config_db_path)


class TestAdminFeedbackEndpoint:

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


class TestAdminFeedbackPaginationStability:
    """created_at comes from CURRENT_TIMESTAMP in both feedback tables, so it has one-second
    resolution and any two submissions inside that second tie on it. Neither query orders the
    tied rows, so the order they arrive in is the engine's to choose and may differ between two
    requests; the feed sorts on created_at alone and hands out a slice of the result, so a page
    boundary landing inside a tie shows an admin one item twice and never shows another
    (genetics-results-suite-qdf).
    """

    STAMP = "2026-01-01 00:00:00"

    @staticmethod
    @contextmanager
    def sources_reversed(config_db, chat_db):
        """Return each source's rows in the reverse of the order the DB gave them.

        That is one of the orders the engine is free to return for rows tied on created_at —
        the same freedom PRAGMA reverse_unordered_selects stands in for in tests/test_db.py,
        applied here instead because the request runs on a different thread from the test and
        so on a different connection from the one a pragma would be set on.
        """
        originals = [
            (config_db, "list_all_user_comments", config_db.list_all_user_comments),
            (chat_db, "list_sessions_with_comments", chat_db.list_sessions_with_comments),
        ]
        for db, name, original in originals:
            setattr(db, name, lambda original=original: list(reversed(original())))
        try:
            yield
        finally:
            for db, name, _ in originals:
                delattr(db, name)

    @pytest.fixture
    def tied_feedback(self, feedback_client):
        """Three comments from each source, every one of them stamped the same second."""
        client, chat_db, config_db = feedback_client
        for i in range(3):
            config_db.add_user_comment(f"user{i}@example.com", f"dialog comment {i}")
        for i in range(3):
            s = chat_db.create_session(f"sess_user{i}@example.com")
            chat_db.update_session(s.id, f"sess_user{i}@example.com", comment=f"session comment {i}")
        # by SQL rather than by racing the clock: landing six writes in one second is the one
        # thing a timing test cannot arrange
        config_db._conn.execute("UPDATE user_comments SET created_at = ?", (self.STAMP,))
        config_db._conn.commit()
        chat_db._conn.execute("UPDATE chat_sessions SET created_at = ?", (self.STAMP,))
        chat_db._conn.commit()
        return client, chat_db, config_db

    def test_paging_through_a_tie_returns_every_item_exactly_once(self, tied_feedback):
        """The pages are walked with the row order flipped under the middle request, which is
        what an admin clicking through the feed can hit: nothing pins the order between two
        requests."""
        client, chat_db, config_db = tied_feedback

        seen = []
        for offset in (0, 2, 4):
            flip = offset == 2
            with self.sources_reversed(config_db, chat_db) if flip else nullcontext():
                page = client.get(f"/chat/v1/admin/feedback?limit=2&offset={offset}").json()
            assert page["total"] == 6
            seen.extend(item["comment"] for item in page["items"])

        assert len(seen) == 6
        assert set(seen) == {f"dialog comment {i}" for i in range(3)} | {
            f"session comment {i}" for i in range(3)
        }

    def test_the_same_page_is_the_same_whichever_order_the_rows_arrive_in(self, tied_feedback):
        """The page is a pure function of the stored rows, not of how they reached the merge."""
        client, chat_db, config_db = tied_feedback

        for offset in (0, 2, 4):
            url = f"/chat/v1/admin/feedback?limit=2&offset={offset}"
            forward = client.get(url).json()["items"]
            with self.sources_reversed(config_db, chat_db):
                backward = client.get(url).json()["items"]
            assert forward == backward

    def test_dialog_comments_tied_in_one_second_stay_newest_first(self, feedback_client):
        """Making the merge total is not enough — it has to be total in the *same* order the feed
        claims to be in, and the one the source query already put the rows in.

        The tiebreak carries user_comments.id, an autoincrement, and eleven rows are enough for
        the two orders to disagree: compared as text, '9' sorts above '10', so a stringified id
        renders 9, 8, 11, 10 where the ids say 11, 10, 9, 8. That reverses pairs of adjacent rows
        inside every tie and contradicts list_all_user_comments' own ORDER BY created_at DESC,
        id DESC, which exists for exactly this feed. Session comments are tied into the same
        second alongside them so the merge still has to compare across the two id spaces.
        """
        client, chat_db, config_db = feedback_client
        for i in range(11):
            config_db.add_user_comment("user@example.com", f"dialog comment {i}")
        s = chat_db.create_session("sess@example.com")
        chat_db.update_session(s.id, "sess@example.com", comment="session comment")
        config_db._conn.execute("UPDATE user_comments SET created_at = ?", (self.STAMP,))
        config_db._conn.commit()
        chat_db._conn.execute("UPDATE chat_sessions SET created_at = ?", (self.STAMP,))
        chat_db._conn.commit()

        page = client.get("/chat/v1/admin/feedback?limit=50").json()
        assert page["total"] == 12
        dialog = [i["comment"] for i in page["items"] if i["source"] == "feedback_dialog"]
        assert dialog == [c.comment for c in config_db.list_all_user_comments()]
        assert dialog == [f"dialog comment {i}" for i in reversed(range(11))]

    def test_both_sources_stamp_created_at_in_the_same_shape(self, tied_feedback):
        """The ni9 invariant, which nothing else guards (genetics-results-suite-uvh 5).

        The merge sorts on the isoformat STRING, so the two sources' stamps have to be
        lexicographically comparable. If one carried a +00:00 offset and the other did not, the
        two would still sort consistently — every offset-bearing string sorts above every bare one
        — so the pagination tests above would stay green while the feed silently grouped by source
        instead of by time. The visible half of the same invariant is on the client: JS parses a
        bare 'T12:00:00' as LOCAL time and the offset form as UTC, so a half-move shifts one
        source's rendered times by the viewer's offset (genetics-results-suite-eis).
        """
        client, _, _ = tied_feedback

        items = client.get("/chat/v1/admin/feedback?limit=50").json()["items"]
        by_source = {}
        for item in items:
            by_source.setdefault(item["source"], []).append(item["created_at"])
        assert set(by_source) == {"feedback_dialog", "session_comment"}

        def shape(stamp: str) -> bool:
            return stamp.endswith("+00:00")

        shapes = {source: {shape(v) for v in vals} for source, vals in by_source.items()}
        assert shapes["feedback_dialog"] == shapes["session_comment"], (
            f"the two sources disagree on the timestamp shape: {shapes}"
        )

    def test_a_tie_orders_across_sources_rather_than_grouping_by_source(self, feedback_client):
        """The other half of the same invariant: with equal stamps the tiebreak must interleave
        the two sources, not stack one above the other."""
        client, chat_db, config_db = feedback_client
        config_db.add_user_comment("a@example.com", "dialog comment")
        s = chat_db.create_session("b@example.com")
        chat_db.update_session(s.id, "b@example.com", comment="session comment")
        config_db._conn.execute("UPDATE user_comments SET created_at = ?", (self.STAMP,))
        config_db._conn.commit()
        chat_db._conn.execute("UPDATE chat_sessions SET created_at = ?", (self.STAMP,))
        chat_db._conn.commit()

        items = client.get("/chat/v1/admin/feedback?limit=50").json()["items"]
        assert len(items) == 2
        # both stamps are the same instant, so they must compare equal as strings
        assert items[0]["created_at"] == items[1]["created_at"]


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

    def test_filter_unrated(self, analyzed_db):
        """unrated=True keeps sessions with no llm_quality_score (the 'NA' case):
        an unanalyzed session, or an analyzed one whose score is NULL."""
        db, s1, s2, s3 = analyzed_db
        # s3 has no analysis row at all -> NULL score
        sessions, total = db.list_all_sessions(unrated=True)
        assert total == 1
        assert sessions[0].id == s3
        # an analyzed session with a NULL quality score also counts as unrated
        s4 = db.create_session("dave@example.com")
        db.upsert_analysis(
            {
                "session_id": s4.id,
                "llm_quality_score": None,
                "success_label": "unknown",
                "llm_disposition": "out_of_scope",
                "llm_issue_categories": [],
            },
            analyzer_version=1,
            source_updated_at=None,
            message_count=2,
        )
        sessions, total = db.list_all_sessions(unrated=True)
        assert total == 2
        assert {s.id for s in sessions} == {s3, s4.id}

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


@pytest.fixture
def instruction_set_admin_client(test_db):
    """Admin client over a session whose messages recorded an instruction set."""
    from genetics_mcp_server.config.settings import get_settings

    if LLMConfigDB in Singleton._instances:
        del Singleton._instances[LLMConfigDB]
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        config_path = f.name
    config_db = LLMConfigDB(config_path)

    named = config_db.create_instruction_set("alice@example.com", "Statistician", "be precise")
    renamed = config_db.create_instruction_set("alice@example.com", "Terse", "be terse")

    # s1 switches sets mid-conversation, s2 never used one, s3 references a set nobody owns
    s1 = test_db.create_session("alice@example.com")
    test_db.add_message(s1.id, "s1-m1", "user", "hi", instruction_set_id=named.id)
    test_db.add_message(s1.id, "s1-m2", "user", "again", instruction_set_id=renamed.id)
    s2 = test_db.create_session("alice@example.com")
    test_db.add_message(s2.id, "s2-m1", "user", "hi")
    s3 = test_db.create_session("bob@example.com")
    test_db.add_message(s3.id, "s3-m1", "user", "hi", instruction_set_id="ghost-set")
    # bob naming a set id that exists but belongs to alice: the memo must be keyed on
    # (user_id, set_id), not set_id alone
    s4 = test_db.create_session("bob@example.com")
    test_db.add_message(s4.id, "s4-m1", "user", "hi", instruction_set_id=named.id)
    sessions = {"alice_switch": s1.id, "alice_none": s2.id, "bob_ghost": s3.id,
                "bob_borrowed": s4.id}

    async def mock_admin():
        return "admin@example.com"

    app.dependency_overrides[admin_required] = mock_admin

    with patch.dict(os.environ, {"ENABLE_ADMIN_PAGE": "true"}):
        get_settings.cache_clear()
        with patch("genetics_mcp_server.routers.admin.get_chat_history_db", return_value=test_db), \
                patch("genetics_mcp_server.routers.admin.get_llm_config_db",
                      return_value=config_db):
            with TestClient(app) as client:
                yield client, config_db, sessions
        get_settings.cache_clear()

    app.dependency_overrides.clear()
    if LLMConfigDB in Singleton._instances:
        del Singleton._instances[LLMConfigDB]
    close_and_unlink(config_db, config_path)


class TestAdminSessionsInstructionSet:
    """The operator needs to see which instructions were in force on a session."""

    def _by_session(self, client_and_db):
        client = client_and_db[0]
        resp = client.get("/chat/v1/admin/sessions")
        assert resp.status_code == 200
        return {s["id"]: s for s in resp.json()["sessions"]}

    def test_latest_set_name_is_reported(self, instruction_set_admin_client):
        sessions = self._by_session(instruction_set_admin_client)
        with_sets = [s for s in sessions.values() if s["instruction_set_name"]]
        assert [s["instruction_set_name"] for s in with_sets] == ["Terse"]

    def test_session_without_a_set_reports_none(self, instruction_set_admin_client):
        sessions = self._by_session(instruction_set_admin_client)
        assert any(
            s["instruction_set_name"] is None and s["message_count"] == 1
            for s in sessions.values()
        )

    def test_unresolvable_set_reports_none(self, instruction_set_admin_client):
        """A message can reference an id from another user or another database file."""
        sessions = self._by_session(instruction_set_admin_client)
        ids = instruction_set_admin_client[2]
        assert sessions[ids["bob_ghost"]]["instruction_set_name"] is None

    def test_other_users_real_set_id_reports_none(self, instruction_set_admin_client):
        """Resolution is scoped to the session's owner: alice's real id must not name
        itself on bob's session."""
        sessions = self._by_session(instruction_set_admin_client)
        ids = instruction_set_admin_client[2]
        assert sessions[ids["bob_borrowed"]]["instruction_set_name"] is None

    def test_archived_set_still_names_itself(self, instruction_set_admin_client):
        """Archiving is a soft delete precisely so history stays resolvable."""
        config_db = instruction_set_admin_client[1]
        sessions_before = self._by_session(instruction_set_admin_client)
        target = next(
            s for s in sessions_before.values() if s["instruction_set_name"] == "Terse"
        )

        terse = next(s for s in config_db.list_instruction_sets("alice@example.com")
                     if s.name == "Terse")
        assert config_db.archive_instruction_set("alice@example.com", terse.id)

        after = self._by_session(instruction_set_admin_client)
        assert after[target["id"]]["instruction_set_name"] == "Terse"
