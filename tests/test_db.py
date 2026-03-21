"""Unit tests for database layer."""

import pytest


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

    def test_add_message_upsert(self, chat_history_db):
        """Test that adding a message with same ID updates it."""
        session = chat_history_db.create_session("user@example.com")
        chat_history_db.add_message(session.id, "msg1", "user", "Hello")
        chat_history_db.add_message(session.id, "msg1", "user", "Updated Hello")

        messages = chat_history_db.get_messages(session.id)
        assert len(messages) == 1
        assert messages[0].content == "Updated Hello"

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
