"""Unit tests for database layer."""

import sqlite3
import threading
from contextlib import contextmanager, nullcontext
from datetime import datetime, timedelta, timezone

import pytest
from conftest import block_writes, unblock_writes

USER = "user@example.com"
# what a read degrades an unusable stored timestamp to, rather than raising
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


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

    def test_add_message_with_instruction_set_id(self, chat_history_db):
        """The instruction set in force for a turn round-trips through save and read."""
        session = chat_history_db.create_session(USER)
        msg = chat_history_db.add_message(
            session.id, "msg1", "user", "Hello", instruction_set_id="set-abc"
        )

        assert msg.instruction_set_id == "set-abc"

        messages = chat_history_db.get_messages(session.id)
        assert messages[0].instruction_set_id == "set-abc"

    def test_add_message_upsert_updates_instruction_set_id(self, chat_history_db):
        session = chat_history_db.create_session(USER)
        chat_history_db.add_message(
            session.id, "msg1", "assistant", "Hello", instruction_set_id="set-abc"
        )
        chat_history_db.add_message(
            session.id, "msg1", "assistant", "Hello", instruction_set_id="set-def"
        )

        messages = chat_history_db.get_messages(session.id)
        assert len(messages) == 1
        assert messages[0].instruction_set_id == "set-def"

    def test_instruction_set_id_defaults_none_for_old_messages(self, chat_history_db):
        """Rows written before the column existed read NULL and behave as today."""
        session = chat_history_db.create_session(USER)
        conn = chat_history_db._conn
        conn.execute(
            "INSERT INTO chat_messages (id, session_id, role, content) VALUES (?, ?, ?, ?)",
            ("legacy", session.id, "user", "Hello"),
        )
        conn.commit()

        messages = chat_history_db.get_messages(session.id)
        assert len(messages) == 1
        assert messages[0].instruction_set_id is None
        assert messages[0].content == "Hello"

    def test_instruction_set_id_survives_fork(self, chat_history_db):
        session = chat_history_db.create_session(USER)
        chat_history_db.add_message(
            session.id, "msg1", "user", "Hello", instruction_set_id="set-abc"
        )
        chat_history_db.set_shared(session.id, USER, True)

        forked = chat_history_db.fork_session(session.id, "other@example.com")
        assert chat_history_db.get_messages(forked.id)[0].instruction_set_id == "set-abc"

    def test_instruction_set_migration_is_idempotent(self, chat_history_db):
        """Re-running the migration over a populated database is a no-op."""
        session = chat_history_db.create_session(USER)
        chat_history_db.add_message(
            session.id, "msg1", "user", "Hello", instruction_set_id="set-abc"
        )

        chat_history_db._init_db()

        columns = [
            row[1]
            for row in chat_history_db._conn.execute("PRAGMA table_info(chat_messages)")
        ]
        assert columns.count("instruction_set_id") == 1
        assert chat_history_db.get_messages(session.id)[0].instruction_set_id == "set-abc"

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


class TestSameSecondTiebreak:
    """changed_at is CURRENT_TIMESTAMP, whose resolution is one second, so any pair of saves to
    the same key inside that second ties on it. Resolving "latest" by timestamp alone therefore
    leaves the winner to the query plan, and a setting the user changed twice in quick succession
    reverts to the first value with no error anywhere. A save is a single INSERT behind one PUT of
    /llm-config/user/settings/{key}, so nothing spaces two writes further apart than a round trip.

    The rows are forced into one second with SQL rather than by racing the clock — sleeping to
    land two writes in the same second is the one thing a timing test cannot arrange.
    """

    STAMP = "2026-01-01 00:00:00"

    @staticmethod
    def tie_timestamps(db, table, stamp=STAMP):
        """Collapse every row of a history table into the same second."""
        db._conn.execute(f"UPDATE {table} SET changed_at = ?", (stamp,))
        db._conn.commit()

    @staticmethod
    def unstamp(db, table, where, params):
        """Blank the timestamps of the matching rows. changed_at carries no NOT NULL, so a row
        that reached the table by any route other than the INSERTs in this class — a migration,
        a manual fix — can have one."""
        db._conn.execute(f"UPDATE {table} SET changed_at = NULL WHERE {where}", params)
        db._conn.commit()

    @staticmethod
    @contextmanager
    def unordered_rows_reversed(db):
        """SQLite's own testing pragma: emit the rows of any select whose result order is
        unconstrained in the reverse of the order this build would otherwise pick. That is one
        alternative order, not every order the engine is free to choose, but it is exactly the
        degree of freedom the pre-tiebreak queries left open — they returned every row tied for
        the newest second and let the loop keep whichever arrived last — and flipping it flips
        their answer. The fix removes the freedom instead of pinning it down: one row per key
        comes back, so there is no order left for the pragma to change.
        """
        db._conn.execute("PRAGMA reverse_unordered_selects = ON")
        try:
            yield
        finally:
            db._conn.execute("PRAGMA reverse_unordered_selects = OFF")

    def test_get_user_setting_returns_the_later_of_two_saves_in_one_second(self, llm_config_db):
        llm_config_db.save_user_setting(USER, "backend", "europepmc")
        later = llm_config_db.save_user_setting(USER, "backend", "perplexity")
        self.tie_timestamps(llm_config_db, "user_settings_history")

        for _ in range(5):
            got = llm_config_db.get_user_setting(USER, "backend")
            assert (got.setting_value, got.id) == ("perplexity", later.id)

    def test_get_user_setting_returns_the_last_of_a_run_of_saves_in_one_second(
        self, llm_config_db
    ):
        """One PUT stores one version, so a client that saves on every change of a control writes
        the whole run of them into a second or two."""
        saved = [llm_config_db.save_user_setting(USER, "backend", f"v{i}") for i in range(6)]
        self.tie_timestamps(llm_config_db, "user_settings_history")

        for _ in range(5):
            got = llm_config_db.get_user_setting(USER, "backend")
            assert (got.setting_value, got.id) == ("v5", saved[-1].id)

    def test_get_user_settings_returns_the_later_of_two_saves_in_one_second(self, llm_config_db):
        llm_config_db.save_user_setting(USER, "backend", "europepmc")
        later = llm_config_db.save_user_setting(USER, "backend", "perplexity")
        llm_config_db.save_user_setting(USER, "verbosity", "brief")
        self.tie_timestamps(llm_config_db, "user_settings_history")

        # both plans, because which one runs is not this code's to decide
        for reversed_rows in (False, True):
            with self.unordered_rows_reversed(llm_config_db) if reversed_rows else nullcontext():
                for _ in range(5):
                    settings = llm_config_db.get_user_settings(USER)
                    assert settings["backend"].setting_value == "perplexity"
                    assert settings["backend"].id == later.id
                    assert settings["verbosity"].setting_value == "brief"

    def test_both_settings_accessors_pick_the_same_winner_within_one_second(self, llm_config_db):
        """The two are read on different request paths for the same value, so they disagreeing
        is a setting that reads back differently depending on which endpoint was called."""
        for value in ("a", "b", "c"):
            llm_config_db.save_user_setting(USER, "backend", value)
        self.tie_timestamps(llm_config_db, "user_settings_history")

        for reversed_rows in (False, True):
            with self.unordered_rows_reversed(llm_config_db) if reversed_rows else nullcontext():
                one = llm_config_db.get_user_setting(USER, "backend")
                many = llm_config_db.get_user_settings(USER)["backend"]
                assert one.id == many.id
                assert one.setting_value == many.setting_value == "c"

    def test_every_key_gets_its_own_winner_when_several_have_several_tied_versions(
        self, llm_config_db
    ):
        """One key with a tie is the easy case. Several keys each carrying several tied rows is
        where a rewrite that resolves the tie per row rather than per key goes wrong quietly: it
        returns two rows for one key and the loop keeps the wrong one, or none for another and
        the key vanishes. A second user writing the same keys in the same second is here because
        the winner is per (user, key), not per key."""
        expected = {}
        for round_ in range(4):
            for key in ("backend", "verbosity", "model", "profile"):
                expected[key] = llm_config_db.save_user_setting(USER, key, f"{key}-{round_}")
        for key in ("backend", "verbosity"):
            llm_config_db.save_user_setting("other@example.com", key, "other user's value")
        self.tie_timestamps(llm_config_db, "user_settings_history")

        for reversed_rows in (False, True):
            with self.unordered_rows_reversed(llm_config_db) if reversed_rows else nullcontext():
                settings = llm_config_db.get_user_settings(USER)
                assert set(settings) == set(expected)
                for key, last_save in expected.items():
                    assert settings[key].id == last_save.id
                    assert settings[key].setting_value == f"{key}-3"
                    assert settings[key].user_id == USER
                    assert llm_config_db.get_user_setting(USER, key).id == last_save.id
                other = llm_config_db.get_user_settings("other@example.com")
                assert {k: v.setting_value for k, v in other.items()} == {
                    "backend": "other user's value",
                    "verbosity": "other user's value",
                }

    def test_a_reset_in_the_same_second_as_a_save_wins(self, llm_config_db):
        """delete_user_setting is a soft delete that appends an empty value, so a reset landing in
        the same second as the save it followed still beats it, exactly as any later write does:
        get_user_setting reads the empty value back and get_user_settings drops the key, rather
        than resurrecting the value the user just cleared."""
        llm_config_db.save_user_setting(USER, "backend", "perplexity")
        llm_config_db.delete_user_setting(USER, "backend")
        self.tie_timestamps(llm_config_db, "user_settings_history")

        for _ in range(5):
            assert llm_config_db.get_user_setting(USER, "backend").setting_value == ""
            assert "backend" not in llm_config_db.get_user_settings(USER)

    def test_user_comments_written_in_one_second_are_ordered_by_id(self, llm_config_db):
        """user_comments.created_at is the same CURRENT_TIMESTAMP with the same one-second
        resolution, and two comments landing inside it are ordered by nothing at all: SQLite may
        return them either way round, and differently between two calls. The admin feedback feed
        pages a merge of these rows, so an undecided order there moves an item across a page
        boundary between requests (genetics-results-suite-qdf)."""
        saved = [llm_config_db.add_user_comment(USER, f"comment {i}") for i in range(5)]
        llm_config_db._conn.execute("UPDATE user_comments SET created_at = ?", (self.STAMP,))
        llm_config_db._conn.commit()
        newest_first = [c.id for c in reversed(saved)]

        for reversed_rows in (False, True):
            with self.unordered_rows_reversed(llm_config_db) if reversed_rows else nullcontext():
                assert [c.id for c in llm_config_db.get_user_comments(USER)] == newest_first
                assert [c.id for c in llm_config_db.list_all_user_comments()] == newest_first

    def test_get_tool_description_returns_the_later_of_two_saves_in_one_second(
        self, llm_config_db
    ):
        llm_config_db.save_tool_description("search_genes", "Version 1", USER)
        llm_config_db.save_tool_description("search_genes", "Version 2", USER)
        self.tie_timestamps(llm_config_db, "tool_description_history")

        for _ in range(5):
            assert llm_config_db.get_tool_description("search_genes").description == "Version 2"

    def test_get_tool_descriptions_returns_the_later_of_two_saves_in_one_second(
        self, llm_config_db
    ):
        llm_config_db.save_tool_description("search_genes", "Version 1", USER)
        llm_config_db.save_tool_description("search_genes", "Version 2", USER)
        llm_config_db.save_tool_description("search_variants", "Only version", USER)
        self.tie_timestamps(llm_config_db, "tool_description_history")

        for reversed_rows in (False, True):
            with self.unordered_rows_reversed(llm_config_db) if reversed_rows else nullcontext():
                for _ in range(5):
                    descriptions = llm_config_db.get_tool_descriptions()
                    assert descriptions["search_genes"].description == "Version 2"
                    assert descriptions["search_variants"].description == "Only version"

    def test_every_tool_gets_its_own_winner_when_several_have_several_tied_versions(
        self, llm_config_db
    ):
        """As for settings: one row per tool, the last one written, with several tools tied at
        once."""
        tools = ("search_genes", "search_variants", "get_phewas", "run_query")
        for round_ in range(4):
            for tool in tools:
                llm_config_db.save_tool_description(tool, f"{tool}-{round_}", USER)
        self.tie_timestamps(llm_config_db, "tool_description_history")

        for reversed_rows in (False, True):
            with self.unordered_rows_reversed(llm_config_db) if reversed_rows else nullcontext():
                descriptions = llm_config_db.get_tool_descriptions()
                assert {n: d.description for n, d in descriptions.items()} == {
                    tool: f"{tool}-3" for tool in tools
                }
                for tool in tools:
                    assert descriptions[tool].id == llm_config_db.get_tool_description(tool).id

    def test_a_key_with_only_blank_timestamps_is_read_by_both_accessors(self, llm_config_db):
        """MAX ignores NULLs, so a key whose rows are all unstamped has no newest row for an
        equality join to match: the plural accessors used to drop it while the singular ones
        returned it and raised TypeError parsing the stamp. Both now hand back the same row —
        the highest id, since nothing else separates the tied rows — with the unusable stamp
        degraded to the epoch, and the keys around it are unaffected either way."""
        llm_config_db.save_user_setting(USER, "backend", "europepmc")
        newest_setting = llm_config_db.save_user_setting(USER, "backend", "perplexity")
        llm_config_db.save_user_setting(USER, "verbosity", "brief")
        llm_config_db.save_tool_description("search_genes", "Version 1", USER)
        newest_description = llm_config_db.save_tool_description("search_genes", "Version 2", USER)
        llm_config_db.save_tool_description("search_variants", "Only version", USER)
        self.tie_timestamps(llm_config_db, "user_settings_history")
        self.tie_timestamps(llm_config_db, "tool_description_history")
        self.unstamp(llm_config_db, "user_settings_history", "setting_key = ?", ("backend",))
        self.unstamp(llm_config_db, "tool_description_history", "tool_name = ?", ("search_genes",))

        for reversed_rows in (False, True):
            with self.unordered_rows_reversed(llm_config_db) if reversed_rows else nullcontext():
                settings = llm_config_db.get_user_settings(USER)
                assert settings["backend"].id == newest_setting.id
                assert settings["backend"].setting_value == "perplexity"
                assert settings["backend"].changed_at == EPOCH
                assert settings["verbosity"].setting_value == "brief"
                one = llm_config_db.get_user_setting(USER, "backend")
                assert (one.id, one.setting_value, one.changed_at) == (
                    newest_setting.id,
                    "perplexity",
                    EPOCH,
                )

                descriptions = llm_config_db.get_tool_descriptions()
                assert descriptions["search_genes"].id == newest_description.id
                assert descriptions["search_genes"].description == "Version 2"
                assert descriptions["search_genes"].changed_at == EPOCH
                assert descriptions["search_variants"].description == "Only version"
                got = llm_config_db.get_tool_description("search_genes")
                assert (got.id, got.description, got.changed_at) == (
                    newest_description.id,
                    "Version 2",
                    EPOCH,
                )

    def test_a_blank_timestamp_does_not_outrank_a_real_one_in_the_same_second(self, llm_config_db):
        """The unstamped row here is the highest id of the three, so a tiebreak that fell back to
        id alone would hand it the key — and then fail to parse its timestamp. Both accessors have
        to pick the newest *stamped* row instead, and the same one."""
        llm_config_db.save_user_setting(USER, "backend", "europepmc")
        stamped_winner = llm_config_db.save_user_setting(USER, "backend", "perplexity")
        unstamped = llm_config_db.save_user_setting(USER, "backend", "never stamped")
        self.tie_timestamps(llm_config_db, "user_settings_history")
        self.unstamp(llm_config_db, "user_settings_history", "id = ?", (unstamped.id,))
        assert unstamped.id > stamped_winner.id

        for reversed_rows in (False, True):
            with self.unordered_rows_reversed(llm_config_db) if reversed_rows else nullcontext():
                assert llm_config_db.get_user_settings(USER)["backend"].id == stamped_winner.id
                assert llm_config_db.get_user_setting(USER, "backend").id == stamped_winner.id

    def test_a_blank_tool_timestamp_does_not_outrank_a_real_one_in_the_same_second(
        self, llm_config_db
    ):
        """The tool-description twin of the test above, and it is not redundant with it: the two
        accessor pairs run different SQL, so the setting test constrains nothing about this join.
        The unstamped row is the highest id of the three, so a plural query that stopped
        restricting the group to the newest changed_at — by dropping the join condition — would
        hand it the tool, and one that took MIN(id) instead of MAX inside the group would hand the
        tool to the *older* of the two stamped versions. Both are versions of the same failure the
        listing exists to prevent: the admin page showing a description that is not the one
        get_tool_description reports as current."""
        llm_config_db.save_tool_description("search_genes", "Version 1", USER)
        stamped_winner = llm_config_db.save_tool_description("search_genes", "Version 2", USER)
        unstamped = llm_config_db.save_tool_description("search_genes", "never stamped", USER)
        self.tie_timestamps(llm_config_db, "tool_description_history")
        self.unstamp(llm_config_db, "tool_description_history", "id = ?", (unstamped.id,))
        assert unstamped.id > stamped_winner.id

        for reversed_rows in (False, True):
            with self.unordered_rows_reversed(llm_config_db) if reversed_rows else nullcontext():
                current = llm_config_db.get_tool_descriptions()["search_genes"]
                assert (current.id, current.description) == (stamped_winner.id, "Version 2")
                one = llm_config_db.get_tool_description("search_genes")
                assert (one.id, one.description) == (stamped_winner.id, "Version 2")

    def test_tool_description_history_leads_with_the_version_in_force(self, llm_config_db):
        """The listing is the version history an admin reads — the endpoint serves it "for
        audit/rollback reference" — so its head disagreeing with get_tool_description means the
        row presented as current is not the one that accessor returns. The LIMIT is the sharper
        edge: a tie can push the newest versions off a page instead of the oldest."""
        for i in range(5):
            llm_config_db.save_tool_description("search_genes", f"Version {i}", USER)
        self.tie_timestamps(llm_config_db, "tool_description_history")

        for _ in range(5):
            history = llm_config_db.get_tool_description_history("search_genes")
            assert [v.description for v in history] == [f"Version {i}" for i in reversed(range(5))]
            assert history[0].description == (
                llm_config_db.get_tool_description("search_genes").description
            )
            paged = llm_config_db.get_tool_description_history("search_genes", limit=2)
            assert [v.description for v in paged] == ["Version 4", "Version 3"]


class TestMalformedTimestampReads:
    """changed_at carries no NOT NULL and no format check, and the column is untyped, so a row
    that reached the table by any route other than the accessors — a migration, a manual fix —
    can hold a NULL, a string nothing can parse, or a value that is not text at all.
    These are admin and config reads, so one such row must not 500 the whole
    listing: the stamp degrades to the epoch rather than raising. The two shapes the column
    actually reaches fail differently, NULL as a TypeError out of fromisoformat and a blank string
    as a ValueError, and neither used to be handled by the singular accessors
    (genetics-results-suite-2cl).
    """

    @staticmethod
    def restamp(db, table, value, where="1", params=()):
        db._conn.execute(f"UPDATE {table} SET changed_at = ? WHERE {where}", (value, *params))
        db._conn.commit()

    @pytest.mark.parametrize("stamp", ["", "   ", "not a timestamp"])
    def test_an_unparseable_setting_stamp_degrades_instead_of_raising(self, llm_config_db, stamp):
        """A blank string is the shape a bad migration leaves behind, and it is *not* the NULL
        case: it survives MAX and the join, so before this it took down the plural read too."""
        llm_config_db.save_user_setting(USER, "backend", "europepmc")
        newest = llm_config_db.save_user_setting(USER, "backend", "perplexity")
        self.restamp(llm_config_db, "user_settings_history", stamp)

        one = llm_config_db.get_user_setting(USER, "backend")
        assert (one.id, one.setting_value, one.changed_at) == (newest.id, "perplexity", EPOCH)
        many = llm_config_db.get_user_settings(USER)["backend"]
        assert (many.id, many.changed_at) == (newest.id, EPOCH)

    @pytest.mark.parametrize("stamp", ["", "   ", "not a timestamp"])
    def test_an_unparseable_tool_stamp_degrades_instead_of_raising(self, llm_config_db, stamp):
        llm_config_db.save_tool_description("search_genes", "Version 1", USER)
        newest = llm_config_db.save_tool_description("search_genes", "Version 2", USER)
        self.restamp(llm_config_db, "tool_description_history", stamp)

        one = llm_config_db.get_tool_description("search_genes")
        assert (one.id, one.description, one.changed_at) == (newest.id, "Version 2", EPOCH)
        many = llm_config_db.get_tool_descriptions()["search_genes"]
        assert (many.id, many.changed_at) == (newest.id, EPOCH)

    @pytest.mark.parametrize("sentinel", ["", 0])
    @pytest.mark.parametrize(
        "table,plural,singular",
        [
            (
                "user_settings_history",
                lambda db: db.get_user_settings(USER)["backend"],
                lambda db: db.get_user_setting(USER, "backend"),
            ),
            (
                "tool_description_history",
                lambda db: db.get_tool_descriptions()["search_genes"],
                lambda db: db.get_tool_description("search_genes"),
            ),
        ],
    )
    def test_a_group_holding_both_a_null_and_a_sentinel_stamp_agrees_across_accessors(
        self, llm_config_db, table, plural, singular, sentinel
    ):
        """The one group shape that distinguishes the null-safe `IS` join from a coalescing one,
        and the reason the join is written with IS rather than COALESCE on both sides.

        Neither sentinel is NULL: MAX ignores the NULL and returns the sentinel, so the group's
        newest stamp is the sentinel row's, and the singular accessor's DESC order puts both a
        blank string and a 0 above NULL for the same reason. Both accessors therefore have to
        return the *sentinel* row here even though the NULL row is newer by id. Coalescing the
        two sides to that same value collapses the two rows onto one key, MAX(id) inside the
        group then picks the NULL row, and the plural read starts reporting a different current
        row than the singular one — the exact disagreement this join is here to rule out, and the
        only one an equality-vs-IS test cannot see. Both sentinels are run because neither stands
        in for the other: SQLite sorts every integer below every string, so under a COALESCE to 0
        a blank row keeps a key of its own and only a 0-stamped row collapses onto the NULLs.
        """
        llm_config_db.save_user_setting(USER, "backend", "europepmc")
        llm_config_db.save_user_setting(USER, "backend", "perplexity")
        llm_config_db.save_tool_description("search_genes", "Version 1", USER)
        llm_config_db.save_tool_description("search_genes", "Version 2", USER)
        ids = [r[0] for r in llm_config_db._conn.execute(f"SELECT id FROM {table} ORDER BY id")]
        sentinel_row, null_row = ids[-2], ids[-1]
        self.restamp(llm_config_db, table, sentinel, "id = ?", (sentinel_row,))
        self.restamp(llm_config_db, table, None, "id = ?", (null_row,))

        for reversed_rows in (False, True):
            with (
                TestSameSecondTiebreak.unordered_rows_reversed(llm_config_db)
                if reversed_rows
                else nullcontext()
            ):
                assert plural(llm_config_db).id == sentinel_row
                assert singular(llm_config_db).id == sentinel_row
                assert plural(llm_config_db).changed_at == EPOCH

    def test_a_usable_stamp_is_still_parsed_as_utc(self, llm_config_db):
        """The degrade must not swallow the real stamps: SQLite stores naive UTC, so the read
        attaches the zone rather than leaving the value to be read as local time."""
        llm_config_db.save_user_setting(USER, "backend", "perplexity")
        llm_config_db.save_tool_description("search_genes", "Version 1", USER)
        stamped = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        self.restamp(llm_config_db, "user_settings_history", "2026-01-02 03:04:05")
        self.restamp(llm_config_db, "tool_description_history", "2026-01-02 03:04:05")

        assert llm_config_db.get_user_setting(USER, "backend").changed_at == stamped
        assert llm_config_db.get_user_settings(USER)["backend"].changed_at == stamped
        assert llm_config_db.get_tool_description("search_genes").changed_at == stamped
        assert llm_config_db.get_tool_descriptions()["search_genes"].changed_at == stamped

    @pytest.mark.parametrize("value", [None, "", "   ", "not a timestamp", 0, 1735689600])
    def test_the_helper_degrades_every_unusable_shape(self, llm_config_db, value):
        """Both raising paths in one place: the isinstance guard covers None and anything
        non-textual SQLite let into the column, the except covers strings that will not parse."""
        assert llm_config_db._as_utc_or_epoch(value) == EPOCH


class TestSavedAndReadStampsCarryTheSameZone:
    """The reads return aware UTC, so the writes have to as well. Neither save writes changed_at
    — the row takes DEFAULT CURRENT_TIMESTAMP, which SQLite writes as naive UTC — so the stamp a
    save hands back is one the process makes up for the response, and a naive local now() there
    puts a value on the PUT response that is the process's UTC offset away from the zone every
    subsequent GET on the same key reports. Same for the version history: it is rendered next to
    the description in force, so its head has to carry the stamp get_tool_description does, not
    the same instant under a different zone (genetics-results-suite-2cl).
    """

    @pytest.mark.parametrize(
        "save",
        [
            lambda db: db.save_tool_description("search_genes", "Version 1", USER),
            lambda db: db.save_user_setting(USER, "backend", "perplexity"),
        ],
        ids=["tool_description", "user_setting"],
    )
    def test_a_save_returns_an_aware_utc_stamp(self, llm_config_db, save):
        assert save(llm_config_db).changed_at.utcoffset() == timedelta(0)

    def test_the_history_head_carries_the_stamp_of_the_version_in_force(self, llm_config_db):
        llm_config_db.save_tool_description("search_genes", "Version 1", USER)

        current = llm_config_db.get_tool_description("search_genes")
        head = llm_config_db.get_tool_description_history("search_genes")[0]
        assert current.changed_at.utcoffset() == timedelta(0)
        assert head.changed_at.utcoffset() == timedelta(0)
        assert head.changed_at == current.changed_at


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


class TestMalformedCommentAndHistoryStamps:
    """The sibling reads of TestMalformedTimestampReads. created_at on user_comments and
    changed_at on tool_description_history carry no NOT NULL and no format check either, and
    these three accessors still called datetime.fromisoformat directly, so a NULL raised a
    TypeError and a blank string a ValueError. list_all_user_comments is the one that matters
    most: it backs GET /chat/v1/admin/feedback, so a single unusable row took the whole admin
    feedback feed down with a 500 (genetics-results-suite-ni9).
    """

    @staticmethod
    def restamp(db, table, column, value):
        db._conn.execute(f"UPDATE {table} SET {column} = ?", (value,))
        db._conn.commit()

    @pytest.mark.parametrize("stamp", [None, "", "   ", "not a timestamp"])
    def test_user_comment_reads_degrade_instead_of_raising(self, llm_config_db, stamp):
        llm_config_db.add_user_comment(USER, "first")
        llm_config_db.add_user_comment(USER, "second")
        self.restamp(llm_config_db, "user_comments", "created_at", stamp)

        mine = llm_config_db.get_user_comments(USER)
        assert [c.comment for c in mine] == ["second", "first"]
        assert {c.created_at for c in mine} == {EPOCH}

        every = llm_config_db.list_all_user_comments()
        assert [c.comment for c in every] == ["second", "first"]
        assert {c.created_at for c in every} == {EPOCH}

    @pytest.mark.parametrize("stamp", [None, "", "   ", "not a timestamp"])
    def test_tool_description_history_degrades_instead_of_raising(self, llm_config_db, stamp):
        llm_config_db.save_tool_description("search_genes", "Version 1", USER)
        llm_config_db.save_tool_description("search_genes", "Version 2", USER)
        self.restamp(llm_config_db, "tool_description_history", "changed_at", stamp)

        history = llm_config_db.get_tool_description_history("search_genes")
        assert [v.description for v in history] == ["Version 2", "Version 1"]
        assert {v.changed_at for v in history} == {EPOCH}

    def test_one_unusable_row_does_not_take_the_rest_of_the_feed_with_it(self, llm_config_db):
        """The 500 was the whole listing, not the one row: every other comment was readable."""
        good = llm_config_db.add_user_comment(USER, "readable")
        bad = llm_config_db.add_user_comment("other@example.com", "unstamped")
        llm_config_db._conn.execute(
            "UPDATE user_comments SET created_at = NULL WHERE id = ?", (bad.id,)
        )
        llm_config_db._conn.commit()

        by_id = {c.id: c for c in llm_config_db.list_all_user_comments()}
        assert set(by_id) == {good.id, bad.id}
        assert by_id[bad.id].created_at == EPOCH
        assert by_id[good.id].created_at.utcoffset() == timedelta(0)

    def test_the_comment_reads_and_the_write_agree_on_the_zone(self, llm_config_db):
        """add_user_comment's stamp is invented for the POST response — the row itself takes
        DEFAULT CURRENT_TIMESTAMP — so a naive local now() there reported the new comment at an
        instant the GET of the same row renders the process's UTC offset away."""
        written = llm_config_db.add_user_comment(USER, "note")
        read = llm_config_db.get_user_comments(USER)[0]

        assert written.created_at.utcoffset() == timedelta(0)
        assert read.created_at.utcoffset() == timedelta(0)
        assert abs((written.created_at - read.created_at).total_seconds()) < 5


# each case blocks one table so the accessor's own DML aborts, standing in for a disk error or a
# lock timeout. `setup` returns whatever the call needs to act on
CHAT_WRITE_ACCESSORS = [
    (
        "create_session",
        "chat_sessions",
        "INSERT",
        lambda db: None,
        lambda db, ctx: db.create_session(USER),
    ),
    (
        "update_session",
        "chat_sessions",
        "UPDATE",
        lambda db: db.create_session(USER).id,
        lambda db, ctx: db.update_session(ctx, USER, title="renamed"),
    ),
    (
        "touch_session",
        "chat_sessions",
        "UPDATE",
        lambda db: db.create_session(USER).id,
        lambda db, ctx: db.touch_session(ctx),
    ),
    (
        "set_shared",
        "chat_sessions",
        "UPDATE",
        lambda db: db.create_session(USER).id,
        lambda db, ctx: db.set_shared(ctx, USER, True),
    ),
    (
        "delete_session",
        "chat_sessions",
        "DELETE",
        lambda db: db.create_session(USER).id,
        lambda db, ctx: db.delete_session(ctx, USER),
    ),
    (
        "add_message",
        "chat_messages",
        "INSERT",
        lambda db: db.create_session(USER).id,
        lambda db, ctx: db.add_message(ctx, "msg-1", "user", "hello"),
    ),
    (
        "rate_message",
        "chat_messages",
        "UPDATE",
        lambda db: db.add_message(db.create_session(USER).id, "msg-1", "user", "hi").id,
        lambda db, ctx: db.rate_message(ctx, True),
    ),
    (
        "add_attachment",
        "chat_attachments",
        "INSERT",
        lambda db: db.create_session(USER).id,
        lambda db, ctx: db.add_attachment(
            "att-1", ctx, "f.tsv", "tsv", "text/tab-separated-values", 10, "/tmp/f.tsv"
        ),
    ),
    (
        "delete_attachment",
        "chat_attachments",
        "DELETE",
        lambda db: db.add_attachment(
            "att-1",
            db.create_session(USER).id,
            "f.tsv",
            "tsv",
            "text/tab-separated-values",
            10,
            "/tmp/f.tsv",
        ).session_id,
        lambda db, ctx: db.delete_attachment("att-1", ctx),
    ),
    (
        "fork_session",
        "chat_sessions",
        "INSERT",
        lambda db: _shared_session_with_a_message(db),
        lambda db, ctx: db.fork_session(ctx, "other@example.com"),
    ),
    (
        "upsert_analysis",
        "conversation_analysis",
        "INSERT",
        lambda db: db.create_session(USER).id,
        lambda db, ctx: db.upsert_analysis(
            {
                "session_id": ctx,
                "user_rating": 4,
                "llm_quality_score": 4,
                "success_label": "successful",
                "llm_disposition": "",
                "topic": "general_genetics",
                "complexity": 2,
                "llm_issue_categories": ["wrong_answer"],
            },
            1,
            None,
            2,
        ),
    ),
]

CHAT_WRITE_ACCESSOR_IDS = [case[0] for case in CHAT_WRITE_ACCESSORS]


def _shared_session_with_a_message(db):
    session = db.create_session(USER)
    db.set_shared(session.id, USER, True)
    db.add_message(session.id, "msg-1", "user", "hello")
    return session.id


class TestChatHistoryWriteTransactionSafety:
    """The same bug class as TestLLMConfigWriteTransactionSafety, in the other database
    (genetics-results-suite-4um). WAL removes the reader/writer contention but not this: python
    opens a transaction before any DML, the connection is cached for the life of the thread, and
    a write that raised without rolling back leaves it open — holding the write lock against
    every other writer of the file, and letting the next commit on that thread store the write
    its caller was told had failed.
    """

    @pytest.mark.parametrize(
        "label,table,event,setup,call", CHAT_WRITE_ACCESSORS, ids=CHAT_WRITE_ACCESSOR_IDS
    )
    def test_a_failed_dml_leaves_no_open_transaction(
        self, chat_history_db, label, table, event, setup, call
    ):
        ctx = setup(chat_history_db)
        block_writes(chat_history_db, table, event=event)
        try:
            with pytest.raises(sqlite3.IntegrityError):
                call(chat_history_db, ctx)

            assert chat_history_db._conn.in_transaction is False
        finally:
            unblock_writes(chat_history_db, table)

    @pytest.mark.parametrize(
        "label,table,event,setup,call", CHAT_WRITE_ACCESSORS, ids=CHAT_WRITE_ACCESSOR_IDS
    )
    def test_a_failed_commit_leaves_no_open_transaction(
        self, chat_history_db, label, table, event, setup, call
    ):
        ctx = setup(chat_history_db)
        with failing_commits(chat_history_db) as failing:
            with pytest.raises(sqlite3.OperationalError):
                call(chat_history_db, ctx)

            assert failing.in_transaction is False

    def test_a_failed_write_releases_the_lock_for_other_connections(self, chat_history_db):
        """The damage is the retained lock: every other writer of this file — the live chat
        threads and the nightly analysis job — blocks on it until this thread happens to commit
        something else, which for a server thread may be never."""
        block_writes(chat_history_db, "chat_sessions")
        try:
            with pytest.raises(sqlite3.IntegrityError):
                chat_history_db.create_session(USER)
        finally:
            unblock_writes(chat_history_db, "chat_sessions")

        # timeout=0 so a held write lock reports itself at once instead of waiting it out
        other = sqlite3.connect(chat_history_db.db_path, timeout=0)
        try:
            other.execute(
                "INSERT INTO chat_sessions (id, user_id) VALUES (?, ?)",
                ("from-another-connection", USER),
            )
            other.commit()
        finally:
            other.close()

    def test_a_write_that_could_not_commit_is_not_committed_by_the_next_accessor(
        self, chat_history_db
    ):
        """A failed statement leaves nothing pending, but a failed COMMIT leaves the write
        itself pending: the next accessor on this connection would store what its caller was
        told had not been saved."""
        session = chat_history_db.create_session(USER)
        with failing_commits(chat_history_db):
            with pytest.raises(sqlite3.OperationalError):
                chat_history_db.update_session(session.id, USER, title="never saved")

            chat_history_db.touch_session(session.id)

        assert chat_history_db.get_session(session.id, USER).title is None

    def test_reads_do_not_return_rows_an_earlier_failed_write_left_pending(self, chat_history_db):
        """A connection sees its own uncommitted rows, so anything a write left behind stays
        visible to every later read on that thread until something ends the transaction. The raw
        INSERT stands in for the DML of a write that raised after executing it."""
        conn = chat_history_db._conn
        conn.execute(
            "INSERT INTO chat_sessions (id, user_id, comment) VALUES (?, ?, ?)",
            ("never-committed", USER, "pending comment"),
        )
        assert conn.in_transaction

        assert chat_history_db.list_sessions(USER) == []
        assert conn.in_transaction is False
        assert chat_history_db.list_sessions_with_comments() == []

    def test_a_partial_multi_dml_write_is_rolled_back_whole(self, chat_history_db):
        """add_message writes the message and touches the session under one commit. Blocking the
        touch leaves the message inserted but uncommitted, so without the rollback it survives as
        soon as anything else on this connection commits — a message stored by a call that
        raised."""
        session = chat_history_db.create_session(USER)
        block_writes(chat_history_db, "chat_sessions", event="UPDATE")
        try:
            with pytest.raises(sqlite3.IntegrityError):
                chat_history_db.add_message(session.id, "msg-1", "user", "hello")
        finally:
            unblock_writes(chat_history_db, "chat_sessions")

        assert chat_history_db._conn.in_transaction is False
        chat_history_db.touch_session(session.id)
        assert chat_history_db.get_messages(session.id) == []

    def test_a_fork_that_fails_partway_leaves_no_half_copied_session(self, chat_history_db):
        """fork_session inserts the session and then copies every message. A failure in the copy
        used to leave the new session pending, so the target user could end up owning a session
        holding none of the conversation it claims to be a fork of."""
        source = _shared_session_with_a_message(chat_history_db)
        block_writes(chat_history_db, "chat_messages")
        try:
            with pytest.raises(sqlite3.IntegrityError):
                chat_history_db.fork_session(source, "other@example.com")
        finally:
            unblock_writes(chat_history_db, "chat_messages")

        assert chat_history_db._conn.in_transaction is False
        chat_history_db.touch_session(source)
        assert chat_history_db.list_sessions("other@example.com") == []


class TestMessageOrderIsTotal:
    """created_at comes from CURRENT_TIMESTAMP and has one-second resolution, so two messages in
    the same second tie on it. get_messages tiebreaks on rowid (insertion order) so that "which
    message was last" has one answer (genetics-results-suite-uvh 9).
    """

    def test_messages_tied_in_one_second_keep_insertion_order(self, chat_history_db):
        session = chat_history_db.create_session("user@example.com")
        for i in range(10):
            chat_history_db.add_message(session.id, f"m{i}", "user", f"message {i}")
        chat_history_db._conn.execute("UPDATE chat_messages SET created_at = '2026-01-01 00:00:00'")
        chat_history_db._conn.commit()

        ids = [m.id for m in chat_history_db.get_messages(session.id)]
        assert ids == [f"m{i}" for i in range(10)]

    def test_the_query_carries_a_total_order(self):
        """A shape guard, deliberately.

        The ambiguity cannot be forced from here: PRAGMA reverse_unordered_selects only reorders
        scans with no ORDER BY, and idx_chat_messages_session already returns ties in insertion
        order, so dropping the tiebreak leaves the behavioural test above still passing. What is
        actually at risk is a different query plan — after ANALYZE, or on another engine — so the
        guard is on the ORDER BY itself.
        """
        import inspect

        from genetics_mcp_server.db import chat_history_db

        source = inspect.getsource(chat_history_db.ChatHistoryDB.get_messages)
        assert "ORDER BY created_at ASC, rowid ASC" in source, (
            "get_messages must impose a total order: routers/admin.py and "
            "scripts/analyze_conversations.py both resolve 'the last set wins' from it"
        )


class TestChatTurnMetrics:
    """Tests for chat_turn_metrics, the per-turn cost and roundtrip telemetry."""

    def _record(self, db, **overrides):
        fields = dict(
            session_id="sess1",
            message_id="msg1",
            user_id=USER,
            iterations=3,
            tool_call_count=7,
            input_tokens=1000,
            output_tokens=200,
            cache_read_tokens=5000,
            cache_create_tokens=800,
            cost_usd=1.2345,
            wall_ms=4200,
            tool_profile="bigquery",
            model="claude-opus-5",
        )
        fields.update(overrides)
        return db.record_turn_metrics(**fields)

    def test_record_and_read_back(self, chat_history_db):
        row_id = self._record(chat_history_db)
        rows = chat_history_db.get_turn_metrics("sess1")

        assert len(rows) == 1
        row = rows[0]
        assert row["id"] == row_id
        assert row["message_id"] == "msg1"
        assert row["iterations"] == 3
        assert row["tool_call_count"] == 7
        assert row["input_tokens"] == 1000
        assert row["output_tokens"] == 200
        assert row["cache_read_tokens"] == 5000
        assert row["cache_create_tokens"] == 800
        assert row["cost_usd"] == pytest.approx(1.2345)
        assert row["wall_ms"] == 4200
        assert row["tool_profile"] == "bigquery"
        assert row["model"] == "claude-opus-5"
        assert row["created_at"] is not None

    def test_no_foreign_key_on_session_or_message(self, chat_history_db):
        """The row is written while the stream is open, before the client has POSTed the
        assistant message and — on a conversation's first turn — before the session row
        exists at all. A foreign key would abort the insert, PRAGMA foreign_keys being ON."""
        self._record(chat_history_db, session_id="never-created", message_id="never-saved")
        assert len(chat_history_db.get_turn_metrics("never-created")) == 1

    def test_both_ids_may_be_absent(self, chat_history_db):
        """A turn with no client ids still contributes its cost and iteration numbers."""
        self._record(chat_history_db, session_id=None, message_id=None)
        count = chat_history_db._conn.execute(
            "SELECT COUNT(*) FROM chat_turn_metrics"
        ).fetchone()[0]
        assert count == 1

    def test_null_message_ids_do_not_collide(self, chat_history_db):
        """The uniqueness guarantee is partial: NULL message ids are not deduplicated."""
        self._record(chat_history_db, message_id=None)
        self._record(chat_history_db, message_id=None)
        assert len(chat_history_db.get_turn_metrics("sess1")) == 2

    def test_same_message_id_overwrites(self, chat_history_db):
        self._record(chat_history_db, iterations=3)
        self._record(chat_history_db, iterations=9, cost_usd=2.0)

        rows = chat_history_db.get_turn_metrics("sess1")
        assert len(rows) == 1
        assert rows[0]["iterations"] == 9
        assert rows[0]["cost_usd"] == pytest.approx(2.0)

    def test_scoped_to_session(self, chat_history_db):
        self._record(chat_history_db, message_id="a", session_id="sess1")
        self._record(chat_history_db, message_id="b", session_id="sess1")
        self._record(chat_history_db, message_id="c", session_id="other")

        assert {r["message_id"] for r in chat_history_db.get_turn_metrics("sess1")} == {"a", "b"}
        assert [r["message_id"] for r in chat_history_db.get_turn_metrics("other")] == ["c"]

    def test_delete_session_removes_metrics(self, chat_history_db):
        """No foreign key means the chat_sessions cascade does not reach this table."""
        session = chat_history_db.create_session(USER)
        self._record(chat_history_db, session_id=session.id)

        chat_history_db.delete_session(session.id, USER)

        assert chat_history_db.get_turn_metrics(session.id) == []

    def test_another_users_message_id_cannot_overwrite_the_row(self, chat_history_db):
        """message_id is client-supplied. Sending someone else's used to silently replace
        their row: cost zeroed, session reassigned, no error and no log."""
        self._record(chat_history_db, user_id=USER, cost_usd=9.99, session_id="sess1")

        assert (
            self._record(
                chat_history_db,
                user_id="attacker@example.com",
                cost_usd=0.0,
                session_id="attacker-sess",
            )
            is None
        )

        rows = chat_history_db.get_turn_metrics("sess1")
        assert len(rows) == 1
        assert rows[0]["user_id"] == USER
        assert rows[0]["cost_usd"] == pytest.approx(9.99)
        # and nothing was inserted under the attacker's session either
        assert chat_history_db.get_turn_metrics("attacker-sess") == []

    def test_same_user_may_still_re_record(self, chat_history_db):
        """The user_id guard must not break the legitimate overwrite path."""
        self._record(chat_history_db, user_id=USER, cost_usd=1.0)
        assert self._record(chat_history_db, user_id=USER, cost_usd=2.0) is not None
        rows = chat_history_db.get_turn_metrics("sess1")
        assert len(rows) == 1
        assert rows[0]["cost_usd"] == pytest.approx(2.0)

    def test_null_user_id_re_records(self, chat_history_db):
        """An unauthenticated deployment writes NULL user_id; `IS` makes it match itself."""
        self._record(chat_history_db, user_id=None, cost_usd=1.0)
        assert self._record(chat_history_db, user_id=None, cost_usd=2.0) is not None
        assert chat_history_db.get_turn_metrics("sess1")[0]["cost_usd"] == pytest.approx(2.0)

    def test_delete_session_reaches_this_users_unattributable_rows(self, chat_history_db):
        """Every conversation's opening turn streams before the session row exists and
        without a message id, so it can never be joined to anything. Leaving it behind
        would keep a permanent record that this user held a conversation and its cost."""
        session = chat_history_db.create_session(USER)
        self._record(chat_history_db, session_id=session.id, message_id="m-attributed")
        self._record(chat_history_db, session_id=None, message_id=None, user_id=USER)
        self._record(
            chat_history_db, session_id=None, message_id=None, user_id="other@example.com"
        )

        chat_history_db.delete_session(session.id, USER)

        remaining = chat_history_db._conn.execute(
            "SELECT user_id FROM chat_turn_metrics"
        ).fetchall()
        assert [r["user_id"] for r in remaining] == ["other@example.com"]

    def test_delete_by_another_user_keeps_metrics(self, chat_history_db):
        session = chat_history_db.create_session(USER)
        self._record(chat_history_db, session_id=session.id)

        assert chat_history_db.delete_session(session.id, "someone@else.com") is False
        assert len(chat_history_db.get_turn_metrics(session.id)) == 1

    def test_failed_commit_leaves_nothing_pending(self, chat_history_db):
        with failing_commits(chat_history_db):
            with pytest.raises(sqlite3.OperationalError):
                self._record(chat_history_db)

        # the rollback released the write lock, so the next writer succeeds
        self._record(chat_history_db, message_id="msg2")
        assert [r["message_id"] for r in chat_history_db.get_turn_metrics("sess1")] == ["msg2"]
