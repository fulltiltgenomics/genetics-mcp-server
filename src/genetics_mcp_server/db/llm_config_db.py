"""
SQLite service for LLM configuration with change history.

Stores system prompts and tool descriptions with full audit trail
of who changed what, when, and why.
"""

import hashlib
import secrets
import sqlite3
import threading
from collections import defaultdict as dd
from dataclasses import dataclass
from datetime import datetime

from .singleton import Singleton


@dataclass
class ToolDescriptionVersion:
    """A version of a tool description with metadata."""

    id: int
    tool_name: str
    description: str
    changed_by: str
    changed_at: datetime
    comment: str | None


@dataclass
class UserComment:
    """Per-user random comment/note."""

    id: int
    user_id: str
    comment: str
    created_at: datetime


@dataclass
class UserSetting:
    """Per-user setting with versioned history."""

    id: int
    user_id: str
    setting_key: str
    setting_value: str
    changed_at: datetime
    comment: str | None


@dataclass
class UserApiToken:
    """Per-user API token for MCP server access."""

    id: int
    user_id: str
    token_prefix: str
    name: str | None
    created_at: datetime
    last_used_at: datetime | None
    is_active: bool


class LLMConfigDB(object, metaclass=Singleton):
    """
    SQLite database for LLM configuration with versioned history.

    Uses thread-local connections for thread safety.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._connections: dict[int, sqlite3.Connection] = dd(
            lambda: self._create_connection()
        )
        self._init_db()

    def _create_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @property
    def _conn(self) -> sqlite3.Connection:
        return self._connections[threading.get_ident()]

    def _migrate_to_history_tables(self, cursor: sqlite3.Cursor) -> None:
        """Migrate data from old tables to new history tables."""
        # check if old user_instructions table exists and has data
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='user_instructions'"
        )
        if cursor.fetchone():
            # check if history table is empty
            cursor.execute("SELECT COUNT(*) FROM user_instructions_history")
            if cursor.fetchone()[0] == 0:
                # migrate data
                cursor.execute(
                    """
                    INSERT INTO user_instructions_history (user_id, instructions, changed_at, comment)
                    SELECT user_id, instructions, changed_at, comment FROM user_instructions
                    """
                )
            # drop old table
            cursor.execute("DROP TABLE IF EXISTS user_instructions")

        # check if old user_tool_descriptions table exists and has data
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='user_tool_descriptions'"
        )
        if cursor.fetchone():
            # check if history table is empty
            cursor.execute("SELECT COUNT(*) FROM user_tool_descriptions_history")
            if cursor.fetchone()[0] == 0:
                # migrate data
                cursor.execute(
                    """
                    INSERT INTO user_tool_descriptions_history (user_id, tool_name, description, changed_at, comment)
                    SELECT user_id, tool_name, description, changed_at, comment FROM user_tool_descriptions
                    """
                )
            # drop old table
            cursor.execute("DROP TABLE IF EXISTS user_tool_descriptions")

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        cursor = self._conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tool_description_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_name TEXT NOT NULL,
                description TEXT NOT NULL,
                changed_by TEXT NOT NULL,
                changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                comment TEXT
            )
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_tool_desc_latest
            ON tool_description_history(tool_name, changed_at DESC)
        """
        )

        # per-user tables with history (no unique constraints - all changes are stored)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_instructions_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                instructions TEXT NOT NULL,
                changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                comment TEXT
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_tool_descriptions_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                description TEXT NOT NULL,
                changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                comment TEXT
            )
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_user_instructions_history_user
            ON user_instructions_history(user_id, changed_at DESC)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_user_tool_desc_history_user
            ON user_tool_descriptions_history(user_id, tool_name, changed_at DESC)
        """
        )

        # migrate data from old tables to history tables if they exist
        self._migrate_to_history_tables(cursor)

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_comments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                comment TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_user_comments_user
            ON user_comments(user_id)
        """
        )

        # user settings with history (key-value store for preferences)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_settings_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                setting_key TEXT NOT NULL,
                setting_value TEXT NOT NULL,
                changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                comment TEXT
            )
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_user_settings_history_user
            ON user_settings_history(user_id, setting_key, changed_at DESC)
        """
        )

        # user API tokens for MCP server access
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_api_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                token_hash TEXT NOT NULL,
                token_prefix TEXT NOT NULL,
                name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_used_at TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_user_api_tokens_user
            ON user_api_tokens(user_id, is_active)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_user_api_tokens_hash
            ON user_api_tokens(token_hash, is_active)
        """
        )

        self._conn.commit()

    def get_tool_descriptions(self) -> dict[str, ToolDescriptionVersion]:
        """Get the latest description for each tool."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT t1.id, t1.tool_name, t1.description, t1.changed_by, t1.changed_at, t1.comment
            FROM tool_description_history t1
            INNER JOIN (
                SELECT tool_name, MAX(changed_at) as max_changed_at
                FROM tool_description_history
                GROUP BY tool_name
            ) t2 ON t1.tool_name = t2.tool_name AND t1.changed_at = t2.max_changed_at
        """
        )
        result = {}
        for row in cursor.fetchall():
            result[row["tool_name"]] = ToolDescriptionVersion(
                id=row["id"],
                tool_name=row["tool_name"],
                description=row["description"],
                changed_by=row["changed_by"],
                changed_at=datetime.fromisoformat(row["changed_at"]),
                comment=row["comment"],
            )
        return result

    def get_tool_description(self, tool_name: str) -> ToolDescriptionVersion | None:
        """Get the latest description for a specific tool."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT id, tool_name, description, changed_by, changed_at, comment
            FROM tool_description_history
            WHERE tool_name = ?
            ORDER BY changed_at DESC
            LIMIT 1
            """,
            (tool_name,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return ToolDescriptionVersion(
            id=row["id"],
            tool_name=row["tool_name"],
            description=row["description"],
            changed_by=row["changed_by"],
            changed_at=datetime.fromisoformat(row["changed_at"]),
            comment=row["comment"],
        )

    def save_tool_description(
        self, tool_name: str, description: str, user: str, comment: str | None = None
    ) -> ToolDescriptionVersion:
        """Save a new version of a tool description."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT INTO tool_description_history (tool_name, description, changed_by, comment)
            VALUES (?, ?, ?, ?)
            """,
            (tool_name, description, user, comment),
        )
        self._conn.commit()

        return ToolDescriptionVersion(
            id=cursor.lastrowid,
            tool_name=tool_name,
            description=description,
            changed_by=user,
            changed_at=datetime.now(),
            comment=comment,
        )

    def get_tool_description_history(
        self, tool_name: str, limit: int = 20
    ) -> list[ToolDescriptionVersion]:
        """Get recent versions of a specific tool description."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT id, tool_name, description, changed_by, changed_at, comment
            FROM tool_description_history
            WHERE tool_name = ?
            ORDER BY changed_at DESC
            LIMIT ?
            """,
            (tool_name, limit),
        )
        return [
            ToolDescriptionVersion(
                id=row["id"],
                tool_name=row["tool_name"],
                description=row["description"],
                changed_by=row["changed_by"],
                changed_at=datetime.fromisoformat(row["changed_at"]),
                comment=row["comment"],
            )
            for row in cursor.fetchall()
        ]

    # user comments

    def get_user_comments(self, user_id: str) -> list[UserComment]:
        """Get all comments for a user."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT id, user_id, comment, created_at
            FROM user_comments
            WHERE user_id = ?
            ORDER BY created_at DESC
            """,
            (user_id,),
        )
        return [
            UserComment(
                id=row["id"],
                user_id=row["user_id"],
                comment=row["comment"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in cursor.fetchall()
        ]

    def add_user_comment(self, user_id: str, comment: str) -> UserComment:
        """Add a new comment for a user."""
        cursor = self._conn.cursor()
        cursor.execute(
            "INSERT INTO user_comments (user_id, comment) VALUES (?, ?)",
            (user_id, comment),
        )
        self._conn.commit()
        return UserComment(
            id=cursor.lastrowid,
            user_id=user_id,
            comment=comment,
            created_at=datetime.now(),
        )

    def delete_user_comment(self, user_id: str, comment_id: int) -> bool:
        """Delete a user's comment."""
        cursor = self._conn.cursor()
        cursor.execute(
            "DELETE FROM user_comments WHERE id = ? AND user_id = ?",
            (comment_id, user_id),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    # user settings

    def get_user_setting(self, user_id: str, setting_key: str) -> UserSetting | None:
        """Get a user's latest value for a specific setting."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT id, user_id, setting_key, setting_value, changed_at, comment
            FROM user_settings_history
            WHERE user_id = ? AND setting_key = ?
            ORDER BY changed_at DESC
            LIMIT 1
            """,
            (user_id, setting_key),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return UserSetting(
            id=row["id"],
            user_id=row["user_id"],
            setting_key=row["setting_key"],
            setting_value=row["setting_value"],
            changed_at=datetime.fromisoformat(row["changed_at"]),
            comment=row["comment"],
        )

    def get_user_settings(self, user_id: str) -> dict[str, UserSetting]:
        """Get all latest settings for a user."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT h1.id, h1.user_id, h1.setting_key, h1.setting_value, h1.changed_at, h1.comment
            FROM user_settings_history h1
            INNER JOIN (
                SELECT setting_key, MAX(changed_at) as max_changed_at
                FROM user_settings_history
                WHERE user_id = ?
                GROUP BY setting_key
            ) h2 ON h1.setting_key = h2.setting_key AND h1.changed_at = h2.max_changed_at
            WHERE h1.user_id = ?
            """,
            (user_id, user_id),
        )
        result = {}
        for row in cursor.fetchall():
            # skip empty values (deleted settings)
            if row["setting_value"]:
                result[row["setting_key"]] = UserSetting(
                    id=row["id"],
                    user_id=row["user_id"],
                    setting_key=row["setting_key"],
                    setting_value=row["setting_value"],
                    changed_at=datetime.fromisoformat(row["changed_at"]),
                    comment=row["comment"],
                )
        return result

    def save_user_setting(
        self,
        user_id: str,
        setting_key: str,
        setting_value: str,
        comment: str | None = None,
    ) -> UserSetting:
        """Save a new version of a user setting."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT INTO user_settings_history (user_id, setting_key, setting_value, comment)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, setting_key, setting_value, comment),
        )
        self._conn.commit()

        return UserSetting(
            id=cursor.lastrowid,
            user_id=user_id,
            setting_key=setting_key,
            setting_value=setting_value,
            changed_at=datetime.now(),
            comment=comment,
        )

    def delete_user_setting(self, user_id: str, setting_key: str) -> bool:
        """Delete a user setting (soft delete by inserting empty value)."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT INTO user_settings_history (user_id, setting_key, setting_value, comment)
            VALUES (?, ?, '', 'Reset to default')
            """,
            (user_id, setting_key),
        )
        self._conn.commit()
        return True

    # user API tokens

    @staticmethod
    def _hash_token(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()

    def create_api_token(self, user_id: str, name: str | None = None) -> tuple[int, str]:
        """Create a new API token. Returns (token_id, plaintext_token)."""
        plaintext = secrets.token_urlsafe(32)
        token_hash = self._hash_token(plaintext)
        token_prefix = plaintext[:8]

        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT INTO user_api_tokens (user_id, token_hash, token_prefix, name)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, token_hash, token_prefix, name),
        )
        self._conn.commit()
        return cursor.lastrowid, plaintext

    def list_api_tokens(self, user_id: str) -> list[UserApiToken]:
        """List all tokens for a user (active and inactive)."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT id, user_id, token_prefix, name, created_at, last_used_at, is_active
            FROM user_api_tokens
            WHERE user_id = ?
            ORDER BY created_at DESC
            """,
            (user_id,),
        )
        return [
            UserApiToken(
                id=row["id"],
                user_id=row["user_id"],
                token_prefix=row["token_prefix"],
                name=row["name"],
                created_at=datetime.fromisoformat(row["created_at"]),
                last_used_at=(
                    datetime.fromisoformat(row["last_used_at"])
                    if row["last_used_at"]
                    else None
                ),
                is_active=bool(row["is_active"]),
            )
            for row in cursor.fetchall()
        ]

    def revoke_api_token(self, user_id: str, token_id: int) -> bool:
        """Revoke a token (sets is_active = 0)."""
        cursor = self._conn.cursor()
        cursor.execute(
            "UPDATE user_api_tokens SET is_active = 0 WHERE id = ? AND user_id = ?",
            (token_id, user_id),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def validate_api_token(self, token: str) -> str | None:
        """Validate a token and return user_id if valid, else None."""
        token_hash = self._hash_token(token)
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT id, user_id FROM user_api_tokens WHERE token_hash = ? AND is_active = 1",
            (token_hash,),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        # update last_used_at
        cursor.execute(
            "UPDATE user_api_tokens SET last_used_at = CURRENT_TIMESTAMP WHERE id = ?",
            (row["id"],),
        )
        self._conn.commit()
        return row["user_id"]


# module-level instance, initialized lazily
_llm_config_db: LLMConfigDB | None = None


def get_llm_config_db() -> LLMConfigDB:
    """Get the singleton LLM config database instance."""
    global _llm_config_db
    if _llm_config_db is None:
        from genetics_mcp_server.config import get_settings

        settings = get_settings()
        db_path = getattr(settings, "llm_config_db", "/mnt/disks/data/llm_config.db")
        _llm_config_db = LLMConfigDB(db_path)
    return _llm_config_db
