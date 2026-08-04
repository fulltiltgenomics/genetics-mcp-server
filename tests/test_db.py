"""Unit tests for database layer."""

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import pytest
from conftest import block_writes, unblock_writes

USER = "user@example.com"


@contextmanager
def failing_commits(db, failures=1):
    """Swap this thread's cached connection for one whose first `failures` commits raise.

    SQLite can return SQLITE_BUSY from COMMIT itself, and unlike a failed statement that leaves
    nothing pending, a failed commit leaves the write itself pending. `failures=1` lets the
    connection recover afterwards, so a later accessor can show what it does with what was left.
    """

    class FailingCommit(sqlite3.Connection):
        remaining = failures

        def commit(self):
            if self.remaining > 0:
                self.remaining -= 1
                raise sqlite3.OperationalError("database is locked")
            super().commit()

    ident = threading.get_ident()
    healthy = db._connections[ident]
    failing = sqlite3.connect(db.db_path, factory=FailingCommit)
    failing.row_factory = sqlite3.Row
    db._connections[ident] = failing
    try:
        yield failing
    finally:
        failing.close()
        db._connections[ident] = healthy


def token_row(db, token_id):
    return next(t for t in db.list_api_tokens(USER) if t.id == token_id)


class TestChatHistoryDB:
    """Tests for ChatHistoryDB."""

    def test_create_session(self, chat_history_db):
        """Test creating a new chat session."""
        session = chat_history_db.create_session("user@example.com")

        assert session.id is not None
        assert session.user_id == "user@example.com"
        assert session.title is None
        assert session.rating is None
        assert session.phenotype_code is None

    def test_create_session_with_phenotype(self, chat_history_db):
        """Test creating a session with phenotype code."""
        session = chat_history_db.create_session(
            "user@example.com", phenotype_code="T2D"
        )

        assert session.phenotype_code == "T2D"

    def test_get_session(self, chat_history_db):
        """Test retrieving a session by ID."""
        created = chat_history_db.create_session("user@example.com")
        retrieved = chat_history_db.get_session(created.id, "user@example.com")

        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.user_id == "user@example.com"

    def test_get_session_wrong_user(self, chat_history_db):
        """Test that users can only access their own sessions."""
        created = chat_history_db.create_session("user@example.com")
        retrieved = chat_history_db.get_session(created.id, "other@example.com")

        assert retrieved is None

    def test_get_session_not_found(self, chat_history_db):
        """Test getting a non-existent session."""
        retrieved = chat_history_db.get_session("nonexistent-id", "user@example.com")

        assert retrieved is None

    def test_list_sessions(self, chat_history_db):
        """Test listing user sessions."""
        chat_history_db.create_session("user@example.com")
        chat_history_db.create_session("user@example.com")
        chat_history_db.create_session("other@example.com")

        sessions = chat_history_db.list_sessions("user@example.com")

        assert len(sessions) == 2
        assert all(s.user_id == "user@example.com" for s in sessions)

    def test_list_sessions_order(self, chat_history_db):
        """Test that sessions are ordered by updated_at descending."""
        s1 = chat_history_db.create_session("user@example.com")
        s2 = chat_history_db.create_session("user@example.com")

        # touch s1 to make it more recent
        chat_history_db.touch_session(s1.id)

        sessions = chat_history_db.list_sessions("user@example.com")

        assert sessions[0].id == s1.id
        assert sessions[1].id == s2.id

    def test_list_sessions_limit(self, chat_history_db):
        """Test session list limit."""
        for _ in range(5):
            chat_history_db.create_session("user@example.com")

        sessions = chat_history_db.list_sessions("user@example.com", limit=3)

        assert len(sessions) == 3

    def test_update_session_title(self, chat_history_db):
        """Test updating session title."""
        session = chat_history_db.create_session("user@example.com")
        updated = chat_history_db.update_session(
            session.id, "user@example.com", title="Test Chat"
        )

        assert updated is True

        retrieved = chat_history_db.get_session(session.id, "user@example.com")
        assert retrieved.title == "Test Chat"

    def test_update_session_rating(self, chat_history_db):
        """Test updating session rating."""
        session = chat_history_db.create_session("user@example.com")
        chat_history_db.update_session(session.id, "user@example.com", rating=5)

        retrieved = chat_history_db.get_session(session.id, "user@example.com")
        assert retrieved.rating == 5

    def test_update_session_comment(self, chat_history_db):
        """Test updating session comment."""
        session = chat_history_db.create_session("user@example.com")
        chat_history_db.update_session(
            session.id, "user@example.com", comment="Great chat!"
        )

        retrieved = chat_history_db.get_session(session.id, "user@example.com")
        assert retrieved.comment == "Great chat!"

    def test_update_session_wrong_user(self, chat_history_db):
        """Test that users can only update their own sessions."""
        session = chat_history_db.create_session("user@example.com")
        updated = chat_history_db.update_session(
            session.id, "other@example.com", title="Hacked"
        )

        assert updated is False

    def test_update_session_not_found(self, chat_history_db):
        """Test updating non-existent session."""
        updated = chat_history_db.update_session(
            "nonexistent-id", "user@example.com", title="Test"
        )

        assert updated is False

    def test_delete_session(self, chat_history_db):
        """Test deleting a session."""
        session = chat_history_db.create_session("user@example.com")
        deleted = chat_history_db.delete_session(session.id, "user@example.com")

        assert deleted is True
        assert chat_history_db.get_session(session.id, "user@example.com") is None

    def test_delete_session_wrong_user(self, chat_history_db):
        """Test that users can only delete their own sessions."""
        session = chat_history_db.create_session("user@example.com")
        deleted = chat_history_db.delete_session(session.id, "other@example.com")

        assert deleted is False
        assert chat_history_db.get_session(session.id, "user@example.com") is not None

    def test_delete_session_cascades_messages(self, chat_history_db):
        """Test that deleting a session also deletes its messages."""
        session = chat_history_db.create_session("user@example.com")
        chat_history_db.add_message(session.id, "msg1", "user", "Hello")
        chat_history_db.add_message(session.id, "msg2", "assistant", "Hi there!")

        chat_history_db.delete_session(session.id, "user@example.com")

        messages = chat_history_db.get_messages(session.id)
        assert len(messages) == 0

    def test_add_message(self, chat_history_db):
        """Test adding a message to a session."""
        session = chat_history_db.create_session("user@example.com")
        msg = chat_history_db.add_message(session.id, "msg1", "user", "Hello")

        assert msg.id == "msg1"
        assert msg.session_id == session.id
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.thumbs_up is None

    def test_add_message_with_content_json(self, chat_history_db):
        """Test adding a message with JSON content blocks."""
        session = chat_history_db.create_session("user@example.com")
        content_json = '[{"type": "text", "text": "Hello"}]'
        msg = chat_history_db.add_message(
            session.id, "msg1", "user", "Hello", content_json=content_json
        )

        assert msg.content_json == content_json

    def test_add_message_with_literature_backend(self, chat_history_db):
        """Test adding a message with literature backend choice."""
        session = chat_history_db.create_session("user@example.com")
        msg = chat_history_db.add_message(
            session.id, "msg1", "user", "Search for BRCA1",
            literature_backend="perplexity"
        )

        assert msg.literature_backend == "perplexity"

        # verify it persists when retrieved
        messages = chat_history_db.get_messages(session.id)
        assert messages[0].literature_backend == "perplexity"

    def test_add_message_with_tool_results_json(self, chat_history_db):
        """Test adding an assistant message with persisted tool_result blocks."""
        session = chat_history_db.create_session("user@example.com")
        tool_results_json = (
            '[{"type": "tool_result", "tool_use_id": "tu_1", "content": "{\\"rows\\": 5}"}]'
        )
        msg = chat_history_db.add_message(
            session.id, "msg1", "assistant", "Done",
            content_json='[{"type": "tool_use", "id": "tu_1", "name": "x", "input": {}}]',
            tool_results_json=tool_results_json,
        )

        assert msg.tool_results_json == tool_results_json

        # verify it persists when retrieved
        messages = chat_history_db.get_messages(session.id)
        assert messages[0].tool_results_json == tool_results_json

    def test_add_message_upsert(self, chat_history_db):
        """Test that adding a message with same ID updates it (incl. tool_results_json)."""
        session = chat_history_db.create_session("user@example.com")
        chat_history_db.add_message(
            session.id, "msg1", "assistant", "Hello", tool_results_json='[{"a": 1}]'
        )
        chat_history_db.add_message(
            session.id, "msg1", "assistant", "Updated Hello", tool_results_json='[{"a": 2}]'
        )

        messages = chat_history_db.get_messages(session.id)
        assert len(messages) == 1
        assert messages[0].content == "Updated Hello"
        assert messages[0].tool_results_json == '[{"a": 2}]'

    def test_tool_results_json_defaults_none_for_old_messages(self, chat_history_db):
        """Messages saved without tool results have tool_results_json = None (back-compat)."""
        session = chat_history_db.create_session("user@example.com")
        chat_history_db.add_message(session.id, "msg1", "user", "Hello")
        messages = chat_history_db.get_messages(session.id)
        assert messages[0].tool_results_json is None

    def test_get_messages(self, chat_history_db):
        """Test retrieving messages for a session."""
        session = chat_history_db.create_session("user@example.com")
        chat_history_db.add_message(session.id, "msg1", "user", "Hello")
        chat_history_db.add_message(session.id, "msg2", "assistant", "Hi!")

        messages = chat_history_db.get_messages(session.id)

        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"

    def test_get_messages_order(self, chat_history_db):
        """Test that messages are ordered by creation time."""
        session = chat_history_db.create_session("user@example.com")
        chat_history_db.add_message(session.id, "msg1", "user", "First")
        chat_history_db.add_message(session.id, "msg2", "assistant", "Second")
        chat_history_db.add_message(session.id, "msg3", "user", "Third")

        messages = chat_history_db.get_messages(session.id)

        assert messages[0].content == "First"
        assert messages[1].content == "Second"
        assert messages[2].content == "Third"

    def test_get_first_user_message(self, chat_history_db):
        """Test getting the first user message for preview."""
        session = chat_history_db.create_session("user@example.com")
        chat_history_db.add_message(session.id, "msg1", "user", "First user message")
        chat_history_db.add_message(session.id, "msg2", "assistant", "Response")
        chat_history_db.add_message(session.id, "msg3", "user", "Second user message")

        first_msg = chat_history_db.get_first_user_message(session.id)

        assert first_msg == "First user message"

    def test_get_first_user_message_no_messages(self, chat_history_db):
        """Test getting first user message when no messages exist."""
        session = chat_history_db.create_session("user@example.com")

        first_msg = chat_history_db.get_first_user_message(session.id)

        assert first_msg is None

    def test_rate_message_thumbs_up(self, chat_history_db):
        """Test rating a message with thumbs up."""
        session = chat_history_db.create_session("user@example.com")
        chat_history_db.add_message(session.id, "msg1", "assistant", "Response")

        updated = chat_history_db.rate_message("msg1", thumbs_up=True)

        assert updated is True
        messages = chat_history_db.get_messages(session.id)
        # SQLite stores boolean as 0/1
        assert messages[0].thumbs_up in (True, 1)

    def test_rate_message_thumbs_down(self, chat_history_db):
        """Test rating a message with thumbs down."""
        session = chat_history_db.create_session("user@example.com")
        chat_history_db.add_message(session.id, "msg1", "assistant", "Response")

        chat_history_db.rate_message("msg1", thumbs_up=False)

        messages = chat_history_db.get_messages(session.id)
        # SQLite stores boolean as 0/1
        assert messages[0].thumbs_up in (False, 0)

    def test_rate_message_clear(self, chat_history_db):
        """Test clearing a message rating."""
        session = chat_history_db.create_session("user@example.com")
        chat_history_db.add_message(session.id, "msg1", "assistant", "Response")
        chat_history_db.rate_message("msg1", thumbs_up=True)
        chat_history_db.rate_message("msg1", thumbs_up=None)

        messages = chat_history_db.get_messages(session.id)
        assert messages[0].thumbs_up is None

    def test_rate_message_not_found(self, chat_history_db):
        """Test rating a non-existent message."""
        updated = chat_history_db.rate_message("nonexistent", thumbs_up=True)

        assert updated is False

    def test_touch_session(self, chat_history_db):
        """Test touching a session updates its timestamp."""
        session = chat_history_db.create_session("user@example.com")
        original_updated = chat_history_db.get_session(
            session.id, "user@example.com"
        ).updated_at

        import time
        # SQLite timestamp precision is seconds, so wait > 1 second
        time.sleep(1.1)

        chat_history_db.touch_session(session.id)

        updated_session = chat_history_db.get_session(session.id, "user@example.com")
        assert updated_session.updated_at > original_updated


class TestLLMConfigDB:
    """Tests for LLMConfigDB."""

    def test_save_and_get_tool_description(self, llm_config_db):
        """Test saving and retrieving a tool description."""
        desc = llm_config_db.save_tool_description(
            tool_name="search_genes",
            description="Search for genes by name",
            user="admin@example.com",
            comment="Initial description",
        )

        assert desc.tool_name == "search_genes"
        assert desc.description == "Search for genes by name"
        assert desc.changed_by == "admin@example.com"
        assert desc.comment == "Initial description"

        retrieved = llm_config_db.get_tool_description("search_genes")
        assert retrieved.description == "Search for genes by name"

    def test_get_tool_description_not_found(self, llm_config_db):
        """Test getting a non-existent tool description."""
        desc = llm_config_db.get_tool_description("nonexistent_tool")

        assert desc is None

    def test_get_tool_descriptions(self, llm_config_db):
        """Test getting all tool descriptions."""
        llm_config_db.save_tool_description(
            "tool1", "Description 1", "user@example.com"
        )
        llm_config_db.save_tool_description(
            "tool2", "Description 2", "user@example.com"
        )

        descriptions = llm_config_db.get_tool_descriptions()

        assert "tool1" in descriptions
        assert "tool2" in descriptions
        assert descriptions["tool1"].description == "Description 1"

    def test_tool_description_versioning(self, llm_config_db):
        """Test that tool descriptions are versioned."""
        import time

        llm_config_db.save_tool_description(
            "search_genes", "Version 1", "user@example.com"
        )
        # wait to ensure different timestamp
        time.sleep(1.1)
        llm_config_db.save_tool_description(
            "search_genes", "Version 2", "user@example.com"
        )

        # latest should be version 2
        latest = llm_config_db.get_tool_description("search_genes")
        assert latest.description == "Version 2"

        # history should have both versions
        history = llm_config_db.get_tool_description_history("search_genes")
        assert len(history) == 2
        assert history[0].description == "Version 2"
        assert history[1].description == "Version 1"

    def test_tool_description_history_limit(self, llm_config_db):
        """Test tool description history limit."""
        for i in range(5):
            llm_config_db.save_tool_description(
                "test_tool", f"Version {i}", "user@example.com"
            )

        history = llm_config_db.get_tool_description_history("test_tool", limit=3)

        assert len(history) == 3

    def test_add_and_get_user_comments(self, llm_config_db):
        """Test adding and retrieving user comments."""
        comment = llm_config_db.add_user_comment(
            "user@example.com", "This is a note"
        )

        assert comment.user_id == "user@example.com"
        assert comment.comment == "This is a note"

        comments = llm_config_db.get_user_comments("user@example.com")
        assert len(comments) == 1
        assert comments[0].comment == "This is a note"

    def test_user_comments_order(self, llm_config_db):
        """Test that user comments are ordered by creation time descending."""
        import time

        llm_config_db.add_user_comment("user@example.com", "First comment")
        # wait to ensure different timestamp
        time.sleep(1.1)
        llm_config_db.add_user_comment("user@example.com", "Second comment")

        comments = llm_config_db.get_user_comments("user@example.com")

        assert comments[0].comment == "Second comment"
        assert comments[1].comment == "First comment"

    def test_delete_user_comment(self, llm_config_db):
        """Test deleting a user comment."""
        comment = llm_config_db.add_user_comment(
            "user@example.com", "Comment to delete"
        )
        deleted = llm_config_db.delete_user_comment("user@example.com", comment.id)

        assert deleted is True
        comments = llm_config_db.get_user_comments("user@example.com")
        assert len(comments) == 0

    def test_delete_user_comment_wrong_user(self, llm_config_db):
        """Test that users can only delete their own comments."""
        comment = llm_config_db.add_user_comment(
            "user@example.com", "My comment"
        )
        deleted = llm_config_db.delete_user_comment("other@example.com", comment.id)

        assert deleted is False

    def test_delete_user_comment_not_found(self, llm_config_db):
        """Test deleting a non-existent comment."""
        deleted = llm_config_db.delete_user_comment("user@example.com", 99999)

        assert deleted is False

    # user settings tests

    def test_save_and_get_user_setting(self, llm_config_db):
        """Test saving and retrieving a user setting."""
        setting = llm_config_db.save_user_setting(
            user_id="user@example.com",
            setting_key="literature_search_backend",
            setting_value="perplexity",
            comment="Switched to Perplexity",
        )

        assert setting.user_id == "user@example.com"
        assert setting.setting_key == "literature_search_backend"
        assert setting.setting_value == "perplexity"
        assert setting.comment == "Switched to Perplexity"

        retrieved = llm_config_db.get_user_setting(
            "user@example.com", "literature_search_backend"
        )
        assert retrieved.setting_value == "perplexity"

    def test_get_user_setting_not_found(self, llm_config_db):
        """Test getting a setting that doesn't exist."""
        setting = llm_config_db.get_user_setting(
            "user@example.com", "nonexistent_setting"
        )

        assert setting is None

    def test_get_user_settings(self, llm_config_db):
        """Test getting all settings for a user."""
        llm_config_db.save_user_setting(
            "user@example.com", "setting1", "value1"
        )
        llm_config_db.save_user_setting(
            "user@example.com", "setting2", "value2"
        )

        settings = llm_config_db.get_user_settings("user@example.com")

        assert "setting1" in settings
        assert "setting2" in settings
        assert settings["setting1"].setting_value == "value1"
        assert settings["setting2"].setting_value == "value2"

    def test_user_settings_isolation(self, llm_config_db):
        """Test that settings are isolated per user."""
        llm_config_db.save_user_setting(
            "user1@example.com", "theme", "dark"
        )
        llm_config_db.save_user_setting(
            "user2@example.com", "theme", "light"
        )

        user1_settings = llm_config_db.get_user_settings("user1@example.com")
        user2_settings = llm_config_db.get_user_settings("user2@example.com")

        assert user1_settings["theme"].setting_value == "dark"
        assert user2_settings["theme"].setting_value == "light"

    def test_user_setting_versioning(self, llm_config_db):
        """Test that settings are versioned (latest wins)."""
        import time

        llm_config_db.save_user_setting(
            "user@example.com", "backend", "europepmc"
        )
        time.sleep(1.1)
        llm_config_db.save_user_setting(
            "user@example.com", "backend", "perplexity"
        )

        latest = llm_config_db.get_user_setting("user@example.com", "backend")
        assert latest.setting_value == "perplexity"

    def test_delete_user_setting(self, llm_config_db):
        """Test deleting a user setting (soft delete)."""
        import time

        llm_config_db.save_user_setting(
            "user@example.com", "backend", "perplexity"
        )
        time.sleep(1.1)
        llm_config_db.delete_user_setting("user@example.com", "backend")

        settings = llm_config_db.get_user_settings("user@example.com")
        # deleted settings should not appear
        assert "backend" not in settings


class TestLLMConfigJournalMode:
    """This file is read and written on the same request paths: validate_api_token does a SELECT
    plus a bookkeeping UPDATE and COMMIT on every MCP request, and the settings and
    tool-description routers read and write it per request, all from one process. Under
    the default rollback journal those two block each other whenever a write reaches the file:
    the writer needs every reader gone before it can commit, and a reader that arrives while it
    is committing is turned away with "database is locked". WAL gives readers a consistent
    snapshot that no writer has to wait for."""

    def test_the_connection_is_in_wal_mode(self, llm_config_db):
        cursor = llm_config_db._conn.cursor()
        cursor.execute("PRAGMA journal_mode")
        assert cursor.fetchone()[0].lower() == "wal"

    def test_a_writer_can_commit_while_a_reader_holds_a_read_transaction(self, llm_config_db):
        """Under a rollback journal the commit needs an exclusive lock, so an open read holds
        the writer off until its own 5s busy_timeout runs out and the write fails outright."""
        reader = sqlite3.connect(llm_config_db.db_path, timeout=0)
        try:
            reader.execute("BEGIN")
            reader.execute("SELECT COUNT(*) FROM user_settings_history").fetchone()

            llm_config_db.save_user_setting(USER, "backend", "perplexity")

            assert llm_config_db.get_user_setting(USER, "backend").setting_value == "perplexity"
        finally:
            reader.rollback()
            reader.close()

    def test_reads_are_never_locked_out_by_a_writer(self, llm_config_db):
        """The reverse direction, and the one a chat turn feels: a reader arriving while a
        write is being applied is refused, and under load that is most of the time. timeout=0
        stands in for a reader that has exhausted its retries rather than one that never waits.
        """
        writes = 50
        done = threading.Event()
        failures = []

        def write_repeatedly():
            try:
                for i in range(writes):
                    llm_config_db.save_user_setting(USER, f"key_{i}", "value")
            except Exception as exc:  # surfaced below rather than lost in the thread
                failures.append(repr(exc))
            finally:
                done.set()

        writer = threading.Thread(target=write_repeatedly)
        reader = sqlite3.connect(llm_config_db.db_path, timeout=0)
        reads = 0
        locked = 0
        try:
            writer.start()
            while not done.is_set():
                try:
                    reader.execute("SELECT COUNT(*) FROM user_settings_history").fetchone()
                    reads += 1
                except sqlite3.OperationalError:
                    locked += 1
        finally:
            writer.join(timeout=30)
            reader.close()

        assert not failures, failures
        assert reads > 0
        assert locked == 0


class TestUserApiTokenExpiry:
    """Tests for the idle (rolling) expiry on per-user API tokens.

    Expiry is measured from a token's last use, so a token in regular use never expires
    while an abandoned one does. `expires_at` is rewritten on every successful validation.
    """

    @staticmethod
    def _set_last_activity(db, token_id: int, days_ago: int):
        """Backdate a token as if it had last been used `days_ago` days ago."""
        stamp = db._utc_stamp(datetime.now(timezone.utc) - timedelta(days=days_ago))
        cursor = db._conn.cursor()
        cursor.execute(
            "UPDATE user_api_tokens SET last_used_at = ?, created_at = ?, expires_at = NULL "
            "WHERE id = ?",
            (stamp, stamp, token_id),
        )
        db._conn.commit()

    def test_fresh_token_validates(self, llm_config_db, monkeypatch):
        monkeypatch.setenv("API_TOKEN_TTL_DAYS", "90")
        _, plaintext = llm_config_db.create_api_token("user@example.com")

        assert llm_config_db.validate_api_token(plaintext) == "user@example.com"

    def test_use_pushes_the_deadline_forward(self, llm_config_db, monkeypatch):
        monkeypatch.setenv("API_TOKEN_TTL_DAYS", "90")
        token_id, plaintext = llm_config_db.create_api_token("user@example.com")

        # 89 days idle: still inside the window, and using it should reset the clock
        self._set_last_activity(llm_config_db, token_id, 89)
        assert llm_config_db.validate_api_token(plaintext) == "user@example.com"

        expires_at = next(
            t.expires_at
            for t in llm_config_db.list_api_tokens("user@example.com")
            if t.id == token_id
        )
        remaining = expires_at - datetime.now(timezone.utc)
        assert timedelta(days=89) < remaining <= timedelta(days=90)

    def test_idle_token_expires(self, llm_config_db, monkeypatch):
        monkeypatch.setenv("API_TOKEN_TTL_DAYS", "90")
        token_id, plaintext = llm_config_db.create_api_token("user@example.com")

        self._set_last_activity(llm_config_db, token_id, 91)
        assert llm_config_db.validate_api_token(plaintext) is None

    def test_legacy_token_without_expires_at_is_judged_on_last_use(
        self, llm_config_db, monkeypatch
    ):
        """Rows predating the expiry column must not be killed off if still in active use."""
        monkeypatch.setenv("API_TOKEN_TTL_DAYS", "90")
        token_id, plaintext = llm_config_db.create_api_token("user@example.com")

        # created long ago but used yesterday — an actively-used legacy token
        cursor = llm_config_db._conn.cursor()
        cursor.execute(
            "UPDATE user_api_tokens SET created_at = ?, last_used_at = ?, expires_at = NULL "
            "WHERE id = ?",
            (
                llm_config_db._utc_stamp(datetime.now(timezone.utc) - timedelta(days=400)),
                llm_config_db._utc_stamp(datetime.now(timezone.utc) - timedelta(days=1)),
                token_id,
            ),
        )
        llm_config_db._conn.commit()

        assert llm_config_db.validate_api_token(plaintext) == "user@example.com"

    def test_ttl_zero_disables_expiry(self, llm_config_db, monkeypatch):
        monkeypatch.setenv("API_TOKEN_TTL_DAYS", "0")
        token_id, plaintext = llm_config_db.create_api_token("user@example.com")

        self._set_last_activity(llm_config_db, token_id, 5000)
        assert llm_config_db.validate_api_token(plaintext) == "user@example.com"

        expires_at = next(
            t.expires_at
            for t in llm_config_db.list_api_tokens("user@example.com")
            if t.id == token_id
        )
        assert expires_at is None

    def test_revoked_token_stays_invalid(self, llm_config_db, monkeypatch):
        monkeypatch.setenv("API_TOKEN_TTL_DAYS", "90")
        token_id, plaintext = llm_config_db.create_api_token("user@example.com")
        llm_config_db.revoke_api_token("user@example.com", token_id)

        assert llm_config_db.validate_api_token(plaintext) is None


# (label, table written, trigger event, setup -> ctx, call). Every one of these is a single DML
# followed by a commit, so the setup exists only to give the DML a row to act on
WRITE_ACCESSORS = [
    (
        "save_tool_description",
        "tool_description_history",
        "INSERT",
        lambda db: None,
        lambda db, ctx: db.save_tool_description("search_genes", "desc", USER),
    ),
    (
        "add_user_comment",
        "user_comments",
        "INSERT",
        lambda db: None,
        lambda db, ctx: db.add_user_comment(USER, "note"),
    ),
    (
        "delete_user_comment",
        "user_comments",
        "DELETE",
        lambda db: db.add_user_comment(USER, "note").id,
        lambda db, ctx: db.delete_user_comment(USER, ctx),
    ),
    (
        "save_user_setting",
        "user_settings_history",
        "INSERT",
        lambda db: None,
        lambda db, ctx: db.save_user_setting(USER, "backend", "perplexity"),
    ),
    (
        "delete_user_setting",
        "user_settings_history",
        "INSERT",
        lambda db: db.save_user_setting(USER, "backend", "perplexity"),
        lambda db, ctx: db.delete_user_setting(USER, "backend"),
    ),
    (
        "create_api_token",
        "user_api_tokens",
        "INSERT",
        lambda db: None,
        lambda db, ctx: db.create_api_token(USER),
    ),
    (
        "revoke_api_token",
        "user_api_tokens",
        "UPDATE",
        lambda db: db.create_api_token(USER)[0],
        lambda db, ctx: db.revoke_api_token(USER, ctx),
    ),
]

WRITE_ACCESSOR_IDS = [case[0] for case in WRITE_ACCESSORS]


class TestLLMConfigWriteTransactionSafety:
    """Connections are cached per thread and server threads are long-lived, so a transaction one
    call leaves open is inherited by every later call on that thread: it holds the write lock
    against every other writer of this file, and turns a write the caller was told had failed
    into a durable one as soon as anything else on that connection commits."""

    @pytest.mark.parametrize(
        "label,table,event,setup,call", WRITE_ACCESSORS, ids=WRITE_ACCESSOR_IDS
    )
    def test_a_failed_dml_leaves_no_open_transaction(
        self, llm_config_db, label, table, event, setup, call
    ):
        ctx = setup(llm_config_db)
        block_writes(llm_config_db, table, event=event)
        try:
            with pytest.raises(sqlite3.IntegrityError):
                call(llm_config_db, ctx)

            assert llm_config_db._conn.in_transaction is False
        finally:
            unblock_writes(llm_config_db, table)

    @pytest.mark.parametrize(
        "label,table,event,setup,call", WRITE_ACCESSORS, ids=WRITE_ACCESSOR_IDS
    )
    def test_a_failed_commit_leaves_no_open_transaction(
        self, llm_config_db, label, table, event, setup, call
    ):
        ctx = setup(llm_config_db)
        with failing_commits(llm_config_db) as failing:
            with pytest.raises(sqlite3.OperationalError):
                call(llm_config_db, ctx)

            assert failing.in_transaction is False

    def test_a_failed_write_releases_the_lock_for_other_connections(self, llm_config_db):
        """The damage is the retained lock: every other writer of this file — the instruction
        sets, the settings store, the token paths — blocks on it until this thread happens to
        commit something else, which for a server thread may be never."""
        block_writes(llm_config_db, "user_settings_history")
        try:
            with pytest.raises(sqlite3.IntegrityError):
                llm_config_db.save_user_setting(USER, "backend", "perplexity")
        finally:
            unblock_writes(llm_config_db, "user_settings_history")

        # timeout=0 so a held write lock reports itself at once instead of waiting it out
        other = sqlite3.connect(llm_config_db.db_path, timeout=0)
        try:
            other.execute(
                "INSERT INTO user_comments (user_id, comment) VALUES (?, ?)",
                (USER, "from another connection"),
            )
            other.commit()
        finally:
            other.close()

    def test_a_write_that_could_not_commit_is_not_committed_by_the_next_accessor(
        self, llm_config_db
    ):
        """A failed statement leaves nothing pending, but a failed COMMIT leaves the write
        itself pending: the next accessor on this connection would store what its caller was
        told had not been saved."""
        with failing_commits(llm_config_db):
            with pytest.raises(sqlite3.OperationalError):
                llm_config_db.save_user_setting(USER, "backend", "perplexity")

            llm_config_db.add_user_comment(USER, "note")

        assert llm_config_db.get_user_setting(USER, "backend") is None
        assert [c.comment for c in llm_config_db.get_user_comments(USER)] == ["note"]

    def test_reads_do_not_return_rows_an_earlier_failed_write_left_pending(self, llm_config_db):
        """A connection sees its own uncommitted rows, so anything a write left behind stays
        visible to every later read on that thread until something ends the transaction. The raw
        INSERT stands in for the DML of a write that raised after executing it."""
        llm_config_db.save_user_setting(USER, "backend", "europepmc")
        conn = llm_config_db._conn
        conn.execute(
            "INSERT INTO user_settings_history (user_id, setting_key, setting_value, changed_at) "
            "VALUES (?, 'backend', 'never committed', '2099-01-01 00:00:00')",
            (USER,),
        )
        assert conn.in_transaction

        assert llm_config_db.get_user_setting(USER, "backend").setting_value == "europepmc"
        assert llm_config_db.get_user_settings(USER)["backend"].setting_value == "europepmc"

        # discarded rather than carried, so no later write can turn it into a stored value
        assert conn.in_transaction is False
        llm_config_db.add_user_comment(USER, "note")
        assert llm_config_db.get_user_setting(USER, "backend").setting_value == "europepmc"


class TestValidateApiTokenTransactionSafety:
    """validate_api_token is the authentication hot path and the one write here that must not
    turn a failed write into a failure: the token's validity is settled by its SELECT, and the
    UPDATE only pushes the rolling idle deadline forward."""

    @staticmethod
    def _set_deadline(db, token_id, days_from_now):
        stamp = db._utc_stamp(datetime.now(timezone.utc) + timedelta(days=days_from_now))
        db._conn.execute(
            "UPDATE user_api_tokens SET expires_at = ?, last_used_at = NULL WHERE id = ?",
            (stamp, token_id),
        )
        db._conn.commit()
        return stamp

    def test_a_failed_deadline_update_still_authenticates(
        self, llm_config_db, monkeypatch, caplog
    ):
        monkeypatch.setenv("API_TOKEN_TTL_DAYS", "90")
        _, plaintext = llm_config_db.create_api_token(USER)
        block_writes(llm_config_db, "user_api_tokens", event="UPDATE")
        try:
            with caplog.at_level("WARNING"):
                assert llm_config_db.validate_api_token(plaintext) == USER

            assert "could not record use of API token" in caplog.text
            assert llm_config_db._conn.in_transaction is False
        finally:
            unblock_writes(llm_config_db, "user_api_tokens")

    def test_a_failed_commit_still_authenticates_and_leaves_no_open_transaction(
        self, llm_config_db, monkeypatch
    ):
        monkeypatch.setenv("API_TOKEN_TTL_DAYS", "90")
        _, plaintext = llm_config_db.create_api_token(USER)

        with failing_commits(llm_config_db) as failing:
            assert llm_config_db.validate_api_token(plaintext) == USER

            assert failing.in_transaction is False

    def test_a_deadline_that_could_not_commit_is_not_committed_by_the_next_accessor(
        self, llm_config_db, monkeypatch
    ):
        monkeypatch.setenv("API_TOKEN_TTL_DAYS", "90")
        token_id, plaintext = llm_config_db.create_api_token(USER)
        deadline = self._set_deadline(llm_config_db, token_id, 30)

        with failing_commits(llm_config_db):
            assert llm_config_db.validate_api_token(plaintext) == USER

            llm_config_db.add_user_comment(USER, "note")

        stored = token_row(llm_config_db, token_id)
        assert llm_config_db._utc_stamp(stored.expires_at) == deadline
        assert stored.last_used_at is None

    def test_an_expired_token_is_still_rejected_after_a_failed_write(
        self, llm_config_db, monkeypatch, caplog
    ):
        """A guard on the swallow rather than on the old bug: the expiry check runs before the
        update, so widening what is tolerated must never reach it."""
        monkeypatch.setenv("API_TOKEN_TTL_DAYS", "90")
        token_id, plaintext = llm_config_db.create_api_token(USER)
        self._set_deadline(llm_config_db, token_id, -1)
        block_writes(llm_config_db, "user_api_tokens", event="UPDATE")
        try:
            with caplog.at_level("WARNING"):
                assert llm_config_db.validate_api_token(plaintext) is None

            assert "could not record use of API token" not in caplog.text
        finally:
            unblock_writes(llm_config_db, "user_api_tokens")

    def test_a_revoked_token_is_still_rejected_after_a_failed_write(
        self, llm_config_db, monkeypatch, caplog
    ):
        """The other half of that guard: revocation is enforced by the SELECT's is_active = 1,
        so a revoked token must never reach the update whose failure is tolerated."""
        monkeypatch.setenv("API_TOKEN_TTL_DAYS", "90")
        token_id, plaintext = llm_config_db.create_api_token(USER)
        assert llm_config_db.revoke_api_token(USER, token_id) is True
        block_writes(llm_config_db, "user_api_tokens", event="UPDATE")
        try:
            with caplog.at_level("WARNING"):
                assert llm_config_db.validate_api_token(plaintext) is None

            assert "could not record use of API token" not in caplog.text
        finally:
            unblock_writes(llm_config_db, "user_api_tokens")
