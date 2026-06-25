"""
SQLite service for chat history persistence.

Stores chat sessions and messages with support for ratings and comments.
"""

import json
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
    shared: bool = False


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
    tool_results_json: str | None = None  # JSON string of tool_result blocks for this assistant turn


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
    text_path: str | None = None  # parsed text/TSV representation (excel files)


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
        # WAL lets the nightly analysis job read/write without blocking live chat writers
        conn.execute("PRAGMA journal_mode = WAL")
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
        if "tool_results_json" not in columns:
            cursor.execute("ALTER TABLE chat_messages ADD COLUMN tool_results_json TEXT")

        # migrations: add columns to chat_sessions if they don't exist
        cursor.execute("PRAGMA table_info(chat_sessions)")
        session_columns = {row[1] for row in cursor.fetchall()}
        if "shared" not in session_columns:
            cursor.execute("ALTER TABLE chat_sessions ADD COLUMN shared BOOLEAN DEFAULT 0")

        # migrations: add columns to chat_attachments if they don't exist
        cursor.execute("PRAGMA table_info(chat_attachments)")
        attachment_columns = {row[1] for row in cursor.fetchall()}
        if "text_path" not in attachment_columns:
            cursor.execute("ALTER TABLE chat_attachments ADD COLUMN text_path TEXT")

        # cache of nightly conversation analysis results, keyed by session
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_analysis (
                session_id TEXT PRIMARY KEY REFERENCES chat_sessions(id) ON DELETE CASCADE,
                analyzed_at TIMESTAMP,
                analyzer_version INTEGER,
                source_updated_at TIMESTAMP,
                message_count INTEGER,
                user_rating INTEGER,
                llm_quality_score INTEGER,
                success_label TEXT,
                llm_disposition TEXT,
                topic TEXT,
                complexity INTEGER,
                metrics_json TEXT
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversation_analysis_analyzed_at
            ON conversation_analysis(analyzed_at)
        """)

        # issue categories normalized to one row each so filtering and the
        # issue-mix plot are plain SQL GROUP BY rather than JSON parsing
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_issue (
                session_id TEXT REFERENCES chat_sessions(id) ON DELETE CASCADE,
                category TEXT
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversation_issue_session
            ON conversation_issue(session_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversation_issue_category
            ON conversation_issue(category)
        """)

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
            shared=False,
        )

    def get_session(self, session_id: str, user_id: str) -> ChatSession | None:
        """Get a session by ID, verifying ownership."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT id, user_id, title, created_at, updated_at, rating, comment, phenotype_code, shared
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
            SELECT id, user_id, title, created_at, updated_at, rating, comment, phenotype_code, shared
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

    def set_shared(self, session_id: str, user_id: str, shared: bool) -> bool:
        """Set the shared flag on a session. Returns False if user doesn't own it."""
        cursor = self._conn.cursor()
        cursor.execute(
            "UPDATE chat_sessions SET shared = ? WHERE id = ? AND user_id = ?",
            (shared, session_id, user_id),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def get_session_for_access(
        self, session_id: str, user_id: str
    ) -> tuple[ChatSession, bool] | None:
        """Get a session if user owns it or it is shared.

        Returns (session, is_owner) or None if not accessible.
        """
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT id, user_id, title, created_at, updated_at, rating, comment, phenotype_code, shared
            FROM chat_sessions
            WHERE id = ?
            """,
            (session_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        session = self._row_to_session(row)
        if session.user_id == user_id:
            return (session, True)
        if session.shared:
            return (session, False)
        return None

    def add_message(
        self,
        session_id: str,
        message_id: str,
        role: str,
        content: str,
        content_json: str | None = None,
        literature_backend: str | None = None,
        tool_profile: str | None = None,
        tool_results_json: str | None = None,
    ) -> ChatMessageRecord:
        """Add a message to a session. If message ID already exists, update it."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT INTO chat_messages (id, session_id, role, content, content_json, literature_backend, tool_profile, tool_results_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                content = excluded.content,
                content_json = excluded.content_json,
                literature_backend = excluded.literature_backend,
                tool_profile = excluded.tool_profile,
                tool_results_json = excluded.tool_results_json
            """,
            (message_id, session_id, role, content, content_json, literature_backend, tool_profile, tool_results_json),
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
            tool_results_json=tool_results_json,
        )

    def get_messages(self, session_id: str) -> list[ChatMessageRecord]:
        """Get all messages for a session, ordered by creation time."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT id, session_id, role, content, created_at, thumbs_up,
                   content_json, literature_backend, tool_profile, tool_results_json
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
        text_path: str | None = None,
    ) -> ChatAttachment:
        """Add an attachment to a session."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT INTO chat_attachments (id, session_id, file_name, file_type, mime_type, file_size, storage_path, text_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (attachment_id, session_id, file_name, file_type, mime_type, file_size, storage_path, text_path),
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
            text_path=text_path,
        )

    def get_attachment(self, attachment_id: str, session_id: str) -> ChatAttachment | None:
        """Get an attachment by ID, verifying session ownership."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT id, session_id, file_name, file_type, mime_type, file_size, storage_path, created_at, text_path
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
            SELECT id, session_id, file_name, file_type, mime_type, file_size, storage_path, created_at, text_path
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

    def list_all_sessions(
        self,
        limit: int = 50,
        offset: int = 0,
        user_filter: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        session_id_filter: str | None = None,
    ) -> tuple[list[ChatSession], int]:
        """List all sessions across all users with optional filters.

        Returns (sessions, total_count) for pagination.
        """
        conditions = []
        params: list = []

        if user_filter:
            conditions.append("s.user_id LIKE ?")
            params.append(f"%{user_filter}%")
        if date_from:
            conditions.append("s.created_at >= ?")
            params.append(date_from)
        if date_to:
            conditions.append("s.created_at <= ?")
            params.append(date_to + " 23:59:59")
        if session_id_filter:
            conditions.append("s.id LIKE ?")
            params.append(f"%{session_id_filter}%")

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        cursor = self._conn.cursor()
        cursor.execute(
            f"SELECT COUNT(*) as cnt FROM chat_sessions s {where}",
            params,
        )
        total = cursor.fetchone()["cnt"]

        cursor.execute(
            f"""
            SELECT s.id, s.user_id, s.title, s.created_at, s.updated_at,
                   s.rating, s.comment, s.phenotype_code, s.shared
            FROM chat_sessions s
            {where}
            ORDER BY s.updated_at DESC
            LIMIT ? OFFSET ?
            """,
            params + [limit, offset],
        )
        sessions = [self._row_to_session(row) for row in cursor.fetchall()]
        return sessions, total

    def get_session_any_user(self, session_id: str) -> ChatSession | None:
        """Get a session by ID without user ownership check (admin use)."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT id, user_id, title, created_at, updated_at, rating, comment, phenotype_code, shared
            FROM chat_sessions WHERE id = ?
            """,
            (session_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_session(row)

    def fork_session(self, source_session_id: str, target_user_id: str) -> ChatSession | None:
        """Fork a shared session for another user.

        Copies the session and all messages with new UUIDs.
        Does NOT copy attachments.
        Returns the new ChatSession, or None if source is not shared/not found.
        """
        source = self.get_session_any_user(source_session_id)
        if source is None or not source.shared:
            return None

        title = f"Fork of: {source.title or 'Untitled'}"
        new_session_id = str(uuid.uuid4())
        now = datetime.now()

        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT INTO chat_sessions (id, user_id, title, phenotype_code)
            VALUES (?, ?, ?, ?)
            """,
            (new_session_id, target_user_id, title, source.phenotype_code),
        )

        # copy messages preserving order
        cursor.execute(
            """
            SELECT role, content, content_json, literature_backend, tool_profile, tool_results_json
            FROM chat_messages
            WHERE session_id = ?
            ORDER BY created_at ASC
            """,
            (source_session_id,),
        )
        for row in cursor.fetchall():
            new_msg_id = str(uuid.uuid4())
            cursor.execute(
                """
                INSERT INTO chat_messages (id, session_id, role, content, content_json, literature_backend, tool_profile, tool_results_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (new_msg_id, new_session_id, row["role"], row["content"],
                 row["content_json"], row["literature_backend"], row["tool_profile"], row["tool_results_json"]),
            )

        self._conn.commit()

        return ChatSession(
            id=new_session_id,
            user_id=target_user_id,
            title=title,
            created_at=now,
            updated_at=now,
            rating=None,
            comment=None,
            phenotype_code=source.phenotype_code,
            shared=False,
        )

    def get_usage_analytics(self, period: str = "week") -> list[dict]:
        """Get daily unique users and conversation counts for the given period.

        period: 'week' (7 days), 'month' (30 days), or 'year' (365 days)
        """
        days = {"week": 7, "month": 30, "year": 365}.get(period, 7)
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT date(created_at) as day,
                   COUNT(DISTINCT user_id) as unique_users,
                   COUNT(*) as conversations
            FROM chat_sessions
            WHERE created_at >= date('now', ?)
            GROUP BY date(created_at)
            ORDER BY day ASC
            """,
            (f"-{days} days",),
        )
        return [
            {"date": row["day"], "unique_users": row["unique_users"], "conversations": row["conversations"]}
            for row in cursor.fetchall()
        ]

    def list_sessions_with_comments(self) -> list[dict]:
        """List sessions with non-empty comments, ordered by created_at DESC.

        Returns dicts with user_id, comment, created_at, session_id for use
        in the unified admin feedback feed.
        """
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT user_id, comment, created_at, id as session_id
            FROM chat_sessions
            WHERE comment IS NOT NULL AND comment != ''
            ORDER BY created_at DESC
            """
        )
        return [
            {
                "user_id": row["user_id"],
                "comment": row["comment"],
                "created_at": datetime.fromisoformat(row["created_at"]),
                "session_id": row["session_id"],
            }
            for row in cursor.fetchall()
        ]

    def upsert_analysis(
        self,
        metrics: object,
        analyzer_version: int,
        source_updated_at: str | datetime | None,
        message_count: int,
    ) -> None:
        """Persist analysis for one session: one conversation_analysis row plus
        its conversation_issue rows, replacing any prior rows for the session.

        ``metrics`` is a ConversationMetrics instance or a dict with the same
        fields. The whole write is a single short transaction so it does not
        block live chat writers (no LLM calls or loops happen under the lock).
        """
        m = metrics if isinstance(metrics, dict) else vars(metrics)

        session_id = m["session_id"]
        issue_categories = m.get("llm_issue_categories") or []
        if isinstance(source_updated_at, datetime):
            source_updated_at = source_updated_at.isoformat()

        metrics_json = json.dumps(m, default=str)

        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT INTO conversation_analysis (
                session_id, analyzed_at, analyzer_version, source_updated_at,
                message_count, user_rating, llm_quality_score, success_label,
                llm_disposition, topic, complexity, metrics_json
            )
            VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                analyzed_at = CURRENT_TIMESTAMP,
                analyzer_version = excluded.analyzer_version,
                source_updated_at = excluded.source_updated_at,
                message_count = excluded.message_count,
                user_rating = excluded.user_rating,
                llm_quality_score = excluded.llm_quality_score,
                success_label = excluded.success_label,
                llm_disposition = excluded.llm_disposition,
                topic = excluded.topic,
                complexity = excluded.complexity,
                metrics_json = excluded.metrics_json
            """,
            (
                session_id,
                analyzer_version,
                source_updated_at,
                message_count,
                m.get("user_rating"),
                m.get("llm_quality_score"),
                m.get("success_label"),
                m.get("llm_disposition"),
                m.get("topic"),
                m.get("complexity"),
                metrics_json,
            ),
        )

        # replace the session's issue rows wholesale to keep them in sync
        cursor.execute(
            "DELETE FROM conversation_issue WHERE session_id = ?", (session_id,)
        )
        if issue_categories:
            cursor.executemany(
                "INSERT INTO conversation_issue (session_id, category) VALUES (?, ?)",
                [(session_id, category) for category in issue_categories],
            )

        self._conn.commit()

    def get_analysis_map(self) -> dict[str, dict]:
        """Return a session_id -> analysis-row dict map for fast lookup."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM conversation_analysis")
        return {row["session_id"]: dict(row) for row in cursor.fetchall()}

    def get_stale_or_missing_session_ids(
        self, force: bool, analyzer_version: int
    ) -> list[str]:
        """Return session ids needing (re)analysis.

        A session is stale when it has no analysis row, its chat_sessions
        updated_at is newer than its analyzed_at (conversation continued), or
        its analyzer_version differs from the current one. ``force`` selects
        every session regardless.
        """
        cursor = self._conn.cursor()
        if force:
            cursor.execute("SELECT id FROM chat_sessions")
            return [row["id"] for row in cursor.fetchall()]

        cursor.execute(
            """
            SELECT s.id
            FROM chat_sessions s
            LEFT JOIN conversation_analysis a ON a.session_id = s.id
            WHERE a.session_id IS NULL
               OR s.updated_at > a.analyzed_at
               OR a.analyzer_version != ?
            """,
            (analyzer_version,),
        )
        return [row["id"] for row in cursor.fetchall()]

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
            text_path=row["text_path"],
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
            shared=bool(row["shared"]) if row["shared"] is not None else False,
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
            tool_results_json=row["tool_results_json"] if "tool_results_json" in keys else None,
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
