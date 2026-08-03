"""Unit tests for the one-shot import of legacy per-user instructions.

The import in LLMConfigDB._migrate_to_history_tables runs once against a live
database and cannot be replayed once a user owns a set, so every branch of the
guard is pinned here.
"""

import sqlite3
from datetime import datetime

import pytest

LEGACY_DDL = """
    CREATE TABLE user_instructions_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        instructions TEXT NOT NULL,
        changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        comment TEXT
    )
"""

ANCIENT_DDL = """
    CREATE TABLE user_instructions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        instructions TEXT NOT NULL,
        changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        comment TEXT
    )
"""


@pytest.fixture
def db_path(tmp_path):
    """Path to a database file that no LLMConfigDB has opened yet."""
    from genetics_mcp_server.db.llm_config_db import LLMConfigDB
    from genetics_mcp_server.db.singleton import Singleton

    Singleton._instances.pop(LLMConfigDB, None)
    yield str(tmp_path / "llm_config.db")
    Singleton._instances.pop(LLMConfigDB, None)


def open_db(db_path):
    """Open (or reopen) the database, running _init_db and the migration."""
    from genetics_mcp_server.db.llm_config_db import LLMConfigDB
    from genetics_mcp_server.db.singleton import Singleton

    Singleton._instances.pop(LLMConfigDB, None)
    return LLMConfigDB(db_path)


def seed_legacy(db_path, rows, table="user_instructions_history", ddl=LEGACY_DDL):
    """Create a legacy table and insert (user_id, instructions, changed_at) rows."""
    conn = sqlite3.connect(db_path)
    conn.execute(ddl)
    for user_id, instructions, changed_at in rows:
        conn.execute(
            f"INSERT INTO {table} (user_id, instructions, changed_at) VALUES (?, ?, ?)",
            (user_id, instructions, changed_at),
        )
    conn.commit()
    conn.close()


def query(db_path, sql, params=()):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def sets_of(db_path, user_id=None):
    if user_id is None:
        return query(db_path, "SELECT * FROM user_instruction_sets ORDER BY user_id")
    return query(db_path, "SELECT * FROM user_instruction_sets WHERE user_id = ?", (user_id,))


class TestLegacyInstructionsImport:
    """Tests for the legacy user_instructions_history -> user_instruction_sets import."""

    def test_imports_latest_row_per_user(self, db_path):
        """Only each user's newest surviving row becomes a set."""
        seed_legacy(
            db_path,
            [
                ("a@example.com", "old a", "2026-01-01 00:00:00"),
                ("a@example.com", "new a", "2026-02-01 00:00:00"),
                ("b@example.com", "only b", "2026-01-15 00:00:00"),
            ],
        )
        open_db(db_path)

        rows = sets_of(db_path)
        assert [(r["user_id"], r["body"]) for r in rows] == [
            ("a@example.com", "new a"),
            ("b@example.com", "only b"),
        ]
        assert all(r["name"] == "Imported" for r in rows)
        assert all(r["archived_at"] is None for r in rows)

    def test_preserves_original_timestamp(self, db_path):
        """The imported set keeps the legacy changed_at rather than claiming to be new."""
        seed_legacy(db_path, [("a@example.com", "body", "2026-01-01 12:34:56")])
        open_db(db_path)

        row = sets_of(db_path, "a@example.com")[0]
        assert row["created_at"] == "2026-01-01 12:34:56"
        assert row["updated_at"] == "2026-01-01 12:34:56"

    def test_writes_history_row(self, db_path):
        """The import is recorded in the append-only history table."""
        seed_legacy(db_path, [("a@example.com", "body", "2026-01-01 00:00:00")])
        open_db(db_path)

        set_row = sets_of(db_path, "a@example.com")[0]
        history = query(db_path, "SELECT * FROM user_instruction_set_history")
        assert len(history) == 1
        assert history[0]["set_id"] == set_row["id"]
        assert history[0]["user_id"] == "a@example.com"
        assert history[0]["name"] == "Imported"
        assert history[0]["body"] == "body"
        assert history[0]["comment"]

    def test_tie_on_changed_at_broken_by_id_desc(self, db_path):
        """Rows sharing a timestamp resolve to the highest id, i.e. the last write."""
        seed_legacy(
            db_path,
            [
                ("a@example.com", "first", "2026-01-01 00:00:00"),
                ("a@example.com", "second", "2026-01-01 00:00:00"),
            ],
        )
        open_db(db_path)

        rows = sets_of(db_path, "a@example.com")
        assert len(rows) == 1
        assert rows[0]["body"] == "second"

    def test_empty_tombstone_skipped(self, db_path):
        """The removed delete_user_instructions wrote '' as a tombstone; it is not a set."""
        seed_legacy(
            db_path,
            [
                ("a@example.com", "deleted later", "2026-01-01 00:00:00"),
                ("a@example.com", "", "2026-02-01 00:00:00"),
            ],
        )
        open_db(db_path)

        assert sets_of(db_path) == []

    @pytest.mark.parametrize("body", ["\n  \n", "\n", "   ", "\t", "\r\n", " \t\r\n "])
    def test_whitespace_only_body_skipped(self, db_path, body):
        """SQLite's one-argument TRIM strips spaces only, so newlines must be handled too."""
        seed_legacy(db_path, [("a@example.com", body, "2026-01-01 00:00:00")])
        open_db(db_path)

        assert sets_of(db_path) == []

    def test_body_with_surrounding_whitespace_still_imported(self, db_path):
        """Only entirely blank bodies are dropped; real text is kept verbatim."""
        seed_legacy(db_path, [("a@example.com", "\n  real text  \n", "2026-01-01 00:00:00")])
        open_db(db_path)

        rows = sets_of(db_path, "a@example.com")
        assert len(rows) == 1
        assert rows[0]["body"] == "\n  real text  \n"

    def test_user_with_existing_set_skipped_but_others_import(self, db_path):
        """The guard is per user: an early adopter must not strand everyone else."""
        open_db(db_path)  # creates the new tables the way the previous release does
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            INSERT INTO user_instruction_sets (id, user_id, name, body)
            VALUES ('own-set', 'a@example.com', 'Mine', 'hand written')
            """
        )
        conn.commit()
        conn.close()
        seed_legacy(
            db_path,
            [
                ("a@example.com", "legacy a", "2026-01-01 00:00:00"),
                ("b@example.com", "legacy b", "2026-01-01 00:00:00"),
            ],
        )
        open_db(db_path)

        a_sets = sets_of(db_path, "a@example.com")
        assert [r["body"] for r in a_sets] == ["hand written"]
        b_sets = sets_of(db_path, "b@example.com")
        assert [r["body"] for r in b_sets] == ["legacy b"]

    def test_archived_import_is_not_reimported(self, db_path):
        """Archiving is a soft delete, so the archived row keeps blocking re-import."""
        seed_legacy(db_path, [("a@example.com", "legacy", "2026-01-01 00:00:00")])
        open_db(db_path)
        conn = sqlite3.connect(db_path)
        conn.execute("UPDATE user_instruction_sets SET archived_at = CURRENT_TIMESTAMP")
        conn.commit()
        conn.close()
        open_db(db_path)

        rows = sets_of(db_path, "a@example.com")
        assert len(rows) == 1
        assert rows[0]["archived_at"] is not None

    def test_double_init_creates_no_duplicates(self, db_path):
        """Reopening the database must not import a second copy."""
        seed_legacy(
            db_path,
            [
                ("a@example.com", "legacy a", "2026-01-01 00:00:00"),
                ("b@example.com", "legacy b", "2026-01-01 00:00:00"),
            ],
        )
        open_db(db_path)
        open_db(db_path)
        open_db(db_path)

        assert len(sets_of(db_path)) == 2
        assert len(query(db_path, "SELECT * FROM user_instruction_set_history")) == 2

    def test_fresh_database_without_legacy_table(self, db_path):
        """A new deployment has no legacy table and imports nothing."""
        open_db(db_path)

        assert sets_of(db_path) == []
        assert (
            query(
                db_path,
                "SELECT name FROM sqlite_master WHERE type='table' AND name='user_instructions_history'",
            )
            == []
        )

    def test_empty_legacy_table(self, db_path):
        """A legacy table with no rows imports nothing."""
        seed_legacy(db_path, [])
        open_db(db_path)

        assert sets_of(db_path) == []

    def test_null_changed_at_does_not_produce_null_timestamps(self, db_path):
        """NULL would override the column DEFAULT and crash every reader in this module."""
        seed_legacy(db_path, [("a@example.com", "legacy", None)])
        open_db(db_path)

        row = sets_of(db_path, "a@example.com")[0]
        assert row["created_at"] is not None
        assert row["updated_at"] is not None
        assert isinstance(datetime.fromisoformat(row["created_at"]), datetime)
        history = query(db_path, "SELECT changed_at FROM user_instruction_set_history")
        assert history[0]["changed_at"] is not None

    def test_null_changed_at_row_wins_over_older_row(self, db_path):
        """A NULL sorts last under a plain ORDER BY, losing to a row it postdates."""
        seed_legacy(
            db_path,
            [
                ("a@example.com", "older", "2026-01-01 00:00:00"),
                ("a@example.com", "unstamped", None),
            ],
        )
        open_db(db_path)

        rows = sets_of(db_path, "a@example.com")
        assert len(rows) == 1
        assert rows[0]["body"] == "unstamped"

    def test_legacy_rows_are_not_deleted(self, db_path):
        """The legacy table survives the release so the import can be redone if needed."""
        seed_legacy(
            db_path,
            [
                ("a@example.com", "one", "2026-01-01 00:00:00"),
                ("a@example.com", "two", "2026-02-01 00:00:00"),
                ("b@example.com", "", "2026-01-01 00:00:00"),
            ],
        )
        open_db(db_path)

        rows = query(db_path, "SELECT * FROM user_instructions_history ORDER BY id")
        assert [r["instructions"] for r in rows] == ["one", "two", ""]

    def test_ancient_database_without_history_table_warns(self, db_path, caplog):
        """user_instructions with no history table is unreachable but must not be silent."""
        seed_legacy(
            db_path,
            [("a@example.com", "ancient", "2026-01-01 00:00:00")],
            table="user_instructions",
            ddl=ANCIENT_DDL,
        )
        with caplog.at_level("WARNING"):
            open_db(db_path)

        assert "user_instructions_history" in caplog.text
        assert sets_of(db_path) == []
        # the rows are stranded, not lost
        assert len(query(db_path, "SELECT * FROM user_instructions")) == 1

    def test_ancient_database_with_history_table_migrates_then_imports(self, db_path):
        """The two-step path still works: user_instructions -> history -> set."""
        seed_legacy(db_path, [])
        seed_legacy(
            db_path,
            [("a@example.com", "ancient", "2026-01-01 00:00:00")],
            table="user_instructions",
            ddl=ANCIENT_DDL,
        )
        open_db(db_path)

        rows = sets_of(db_path, "a@example.com")
        assert [r["body"] for r in rows] == ["ancient"]
        assert (
            query(
                db_path,
                "SELECT name FROM sqlite_master WHERE type='table' AND name='user_instructions'",
            )
            == []
        )
