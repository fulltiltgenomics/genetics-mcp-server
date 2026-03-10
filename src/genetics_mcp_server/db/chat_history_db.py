"""
SQLite service for chat history persistence.

Stores chat sessions and messages with support for ratings and comments.
"""

import sqlite3
import threading
import uuid
from collections import defaultdict as dd
from dataclasses import dataclass
from datetime import datetime

from .singleton import Singleton


@dataclass
class ChatSession:
    """A chat session with metadata."""
    id: str
    user_id: str
    title: str | None
    created_at: datetime
    updated_at: datetime
    rating: int | None
    comment: str | None
    phenotype_code: str | None


@dataclass
class ChatMessageRecord:
    """A chat message stored in the database."""
    id: str
    session_id: str
    role: str
    content: str
    created_at: datetime
    thumbs_up: bool | None
    content_json: str | None = None  # JSON string of full message content blocks
    literature_backend: str | None = None  # literature search backend: europepmc or perplexity
    tool_profile: str | None = None  # tool profile: api, bigquery, rag, or None (all)


@dataclass
class ChatAttachment:
    """A file attachment for a chat message."""
    id: str
    session_id: str
    file_name: str
    file_type: str  # "image", "tsv", "excel"
    mime_type: str
    file_size: int
    storage_path: str
    created_at: datetime


class ChatHistoryDB(object, metaclass=Singleton):
    """
    SQLite database for chat history persistence.

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
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @property
    def _conn(self) -> sqlite3.Connection:
        return self._connections[threading.get_ident()]

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        cursor = self._conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                rating INTEGER,
                comment TEXT,
                phenotype_code TEXT
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_chat_sessions_user
            ON chat_sessions(user_id, updated_at DESC)
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                thumbs_up BOOLEAN,
                FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_chat_messages_session
            ON chat_messages(session_id, created_at ASC)
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_attachments (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                file_name TEXT NOT NULL,
                file_type TEXT NOT NULL,
                mime_type TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                storage_path TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_chat_attachments_session
            ON chat_attachments(session_id)
        """)

        # migrations: add columns if they don't exist
        cursor.execute("PRAGMA table_info(chat_messages)")
        columns = {row[1] for row in cursor.fetchall()}
        if "content_json" not in columns:
            cursor.execute("ALTER TABLE chat_messages ADD COLUMN content_json TEXT")
        if "literature_backend" not in columns:
            cursor.execute("ALTER TABLE chat_messages ADD COLUMN literature_backend TEXT")
        if "tool_profile" not in columns:
            cursor.execute("ALTER TABLE chat_messages ADD COLUMN tool_profile TEXT")

        self._conn.commit()

    def create_session(
        self, user_id: str, phenotype_code: str | None = None
    ) -> ChatSession:
        """Create a new chat session."""
        session_id = str(uuid.uuid4())
        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT INTO chat_sessions (id, user_id, phenotype_code)
            VALUES (?, ?, ?)
            """,
            (session_id, user_id, phenotype_code),
        )
        self._conn.commit()

        return ChatSession(
            id=session_id,
            user_id=user_id,
            title=None,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            rating=None,
            comment=None,
            phenotype_code=phenotype_code,
        )

    def get_session(self, session_id: str, user_id: str) -> ChatSession | None:
        """Get a session by ID, verifying ownership."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT id, user_id, title, created_at, updated_at, rating, comment, phenotype_code
            FROM chat_sessions
            WHERE id = ? AND user_id = ?
            """,
            (session_id, user_id),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_session(row)

    def list_sessions(self, user_id: str, limit: int = 50) -> list[ChatSession]:
        """List user's chat sessions, most recent first."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT id, user_id, title, created_at, updated_at, rating, comment, phenotype_code
            FROM chat_sessions
            WHERE user_id = ?
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        )
        return [self._row_to_session(row) for row in cursor.fetchall()]

    def update_session(
        self,
        session_id: str,
        user_id: str,
        title: str | None = None,
        rating: int | None = None,
        comment: str | None = None,
    ) -> bool:
        """Update session metadata. Returns True if updated."""
        updates = []
        params = []

        if title is not None:
            updates.append("title = ?")
            params.append(title)
        if rating is not None:
            updates.append("rating = ?")
            params.append(rating)
        if comment is not None:
            updates.append("comment = ?")
            params.append(comment)

        if not updates:
            return False

        updates.append("updated_at = CURRENT_TIMESTAMP")
        params.extend([session_id, user_id])

        cursor = self._conn.cursor()
        cursor.execute(
            f"""
            UPDATE chat_sessions
            SET {", ".join(updates)}
            WHERE id = ? AND user_id = ?
            """,
            params,
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def touch_session(self, session_id: str) -> None:
        """Update the session's updated_at timestamp."""
        cursor = self._conn.cursor()
        cursor.execute(
            "UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (session_id,),
        )
        self._conn.commit()

    def delete_session(self, session_id: str, user_id: str) -> bool:
        """Delete a session and its messages. Returns True if deleted."""
        cursor = self._conn.cursor()
        cursor.execute(
            "DELETE FROM chat_sessions WHERE id = ? AND user_id = ?",
            (session_id, user_id),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def add_message(
        self,
        session_id: str,
        message_id: str,
        role: str,
        content: str,
        content_json: str | None = None,
        literature_backend: str | None = None,
        tool_profile: str | None = None,
    ) -> ChatMessageRecord:
        """Add a message to a session. If message ID already exists, update it."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT INTO chat_messages (id, session_id, role, content, content_json, literature_backend, tool_profile)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                content = excluded.content,
                content_json = excluded.content_json,
                literature_backend = excluded.literature_backend,
                tool_profile = excluded.tool_profile
            """,
            (message_id, session_id, role, content, content_json, literature_backend, tool_profile),
        )
        # also touch the session
        cursor.execute(
            "UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (session_id,),
        )
        self._conn.commit()

        return ChatMessageRecord(
            id=message_id,
            session_id=session_id,
            role=role,
            content=content,
            created_at=datetime.now(),
            thumbs_up=None,
            content_json=content_json,
            literature_backend=literature_backend,
            tool_profile=tool_profile,
        )

    def get_messages(self, session_id: str) -> list[ChatMessageRecord]:
        """Get all messages for a session, ordered by creation time."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT id, session_id, role, content, created_at, thumbs_up,
                   content_json, literature_backend, tool_profile
            FROM chat_messages
            WHERE session_id = ?
            ORDER BY created_at ASC
            """,
            (session_id,),
        )
        return [self._row_to_message(row) for row in cursor.fetchall()]

    def get_first_user_message(self, session_id: str) -> str | None:
        """Get the first user message content for preview."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT content FROM chat_messages
            WHERE session_id = ? AND role = 'user'
            ORDER BY created_at ASC
            LIMIT 1
            """,
            (session_id,),
        )
        row = cursor.fetchone()
        return row["content"] if row else None

    def rate_message(self, message_id: str, thumbs_up: bool | None) -> bool:
        """Rate a message with thumbs up/down. Returns True if updated."""
        cursor = self._conn.cursor()
        cursor.execute(
            "UPDATE chat_messages SET thumbs_up = ? WHERE id = ?",
            (thumbs_up, message_id),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def add_attachment(
        self,
        attachment_id: str,
        session_id: str,
        file_name: str,
        file_type: str,
        mime_type: str,
        file_size: int,
        storage_path: str,
    ) -> ChatAttachment:
        """Add an attachment to a session."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT INTO chat_attachments (id, session_id, file_name, file_type, mime_type, file_size, storage_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (attachment_id, session_id, file_name, file_type, mime_type, file_size, storage_path),
        )
        self._conn.commit()
        return ChatAttachment(
            id=attachment_id,
            session_id=session_id,
            file_name=file_name,
            file_type=file_type,
            mime_type=mime_type,
            file_size=file_size,
            storage_path=storage_path,
            created_at=datetime.now(),
        )

    def get_attachment(self, attachment_id: str, session_id: str) -> ChatAttachment | None:
        """Get an attachment by ID, verifying session ownership."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT id, session_id, file_name, file_type, mime_type, file_size, storage_path, created_at
            FROM chat_attachments
            WHERE id = ? AND session_id = ?
            """,
            (attachment_id, session_id),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_attachment(row)

    def get_session_attachments(self, session_id: str) -> list[ChatAttachment]:
        """Get all attachments for a session."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT id, session_id, file_name, file_type, mime_type, file_size, storage_path, created_at
            FROM chat_attachments
            WHERE session_id = ?
            ORDER BY created_at ASC
            """,
            (session_id,),
        )
        return [self._row_to_attachment(row) for row in cursor.fetchall()]

    def delete_attachment(self, attachment_id: str, session_id: str) -> bool:
        """Delete an attachment. Returns True if deleted."""
        cursor = self._conn.cursor()
        cursor.execute(
            "DELETE FROM chat_attachments WHERE id = ? AND session_id = ?",
            (attachment_id, session_id),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def _row_to_attachment(self, row: sqlite3.Row) -> ChatAttachment:
        return ChatAttachment(
            id=row["id"],
            session_id=row["session_id"],
            file_name=row["file_name"],
            file_type=row["file_type"],
            mime_type=row["mime_type"],
            file_size=row["file_size"],
            storage_path=row["storage_path"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def _row_to_session(self, row: sqlite3.Row) -> ChatSession:
        return ChatSession(
            id=row["id"],
            user_id=row["user_id"],
            title=row["title"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            rating=row["rating"],
            comment=row["comment"],
            phenotype_code=row["phenotype_code"],
        )

    def _row_to_message(self, row: sqlite3.Row) -> ChatMessageRecord:
        keys = row.keys()
        return ChatMessageRecord(
            id=row["id"],
            session_id=row["session_id"],
            role=row["role"],
            content=row["content"],
            created_at=datetime.fromisoformat(row["created_at"]),
            thumbs_up=row["thumbs_up"],
            content_json=row["content_json"] if "content_json" in keys else None,
            literature_backend=row["literature_backend"] if "literature_backend" in keys else None,
            tool_profile=row["tool_profile"] if "tool_profile" in keys else None,
        )


# module-level instance, initialized lazily
_chat_history_db: ChatHistoryDB | None = None


def get_chat_history_db() -> ChatHistoryDB:
    """Get the singleton chat history database instance."""
    global _chat_history_db
    if _chat_history_db is None:
        from genetics_mcp_server.config import get_settings

        settings = get_settings()
        db_path = getattr(settings, "chat_history_db", "/mnt/disks/data/chat_history.db")
        _chat_history_db = ChatHistoryDB(db_path)
    return _chat_history_db
