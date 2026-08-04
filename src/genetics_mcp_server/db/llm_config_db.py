"""
SQLite service for LLM configuration with change history.

Stores system prompts and tool descriptions with full audit trail
of who changed what, when, and why.
"""

import hashlib
import logging
import os
import secrets
import sqlite3
import threading
import uuid
from collections import defaultdict as dd
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .singleton import Singleton

logger = logging.getLogger(__name__)

# the selected set's body will be appended to the system prompt of every chat request once
# prompt assembly consumes it, so both caps bound recurring token cost rather than storage.
# Both are enforced at write time; on read they behave differently. The count cap is applied
# by list_instruction_sets, which drops the rows past it, while an over-cap body is only
# reported (InstructionSet.body_over_cap) and never clamped: rows predating a cap exceed it —
# the legacy import applies none — and a read that truncated one would feed the truncation
# straight back into the next write
INSTRUCTION_SET_MAX_BODY_CHARS = 4000
INSTRUCTION_SET_MAX_PER_USER = 20


class InstructionSetBodyTooLong(ValueError):
    """Raised when an instruction set body exceeds INSTRUCTION_SET_MAX_BODY_CHARS."""


class InstructionSetLimitReached(ValueError):
    """Raised when a user already holds INSTRUCTION_SET_MAX_PER_USER non-archived sets."""


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
class InstructionSet:
    """A named set of user-authored instructions, to be appended to the chat system prompt."""

    id: str
    user_id: str
    name: str
    body: str
    created_at: datetime
    updated_at: datetime
    archived_at: datetime | None = None
    # computed on read, never stored: body is always the authoritative stored text, so a
    # consumer that must bound its length (prompt assembly) or explain why saving the set
    # unchanged is rejected (the edit dialog) needs to be told the row predates the cap
    body_over_cap: bool = False


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
    # rolling idle deadline; None means expiry is disabled for this token
    expires_at: datetime | None = None


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
        # under journal_mode=delete a reader is refused while a write is being applied (commit
        # takes an EXCLUSIVE lock), and a writer's commit is refused while any reader still
        # holds a read transaction. WAL removes both, as it already does for chat history
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    @property
    def _conn(self) -> sqlite3.Connection:
        return self._connections[threading.get_ident()]

    def _table_exists(self, cursor: sqlite3.Cursor, name: str) -> bool:
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
        )
        return cursor.fetchone() is not None

    def _migrate_to_history_tables(self, cursor: sqlite3.Cursor) -> None:
        """Migrate data from old tables to new history tables."""
        # _init_db no longer creates user_instructions_history, so a database old enough to
        # still hold user_instructions can reach here with no table to migrate into
        if self._table_exists(cursor, "user_instructions"):
            if not self._table_exists(cursor, "user_instructions_history"):
                logger.warning(
                    "user_instructions exists without user_instructions_history: its rows "
                    "are left unmigrated. No deployed database should be in this state"
                )
            else:
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

        # check if old user_tool_descriptions table exists
        if self._table_exists(cursor, "user_tool_descriptions"):
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

        # the per-user instructions feature was removed in 99fbdac (Mar 2026) and its text
        # never reached prompt assembly, but users wrote it and may still want it, so hand
        # each of them their last version as a named set they can edit or archive. The table
        # is no longer created for new databases and is deliberately not dropped here, so
        # deployed rows survive this release and stay available if the import needs redoing
        if self._table_exists(cursor, "user_instructions_history"):
            # the guard is per user, not table-wide: one user owning a set must not strand
            # everyone else's legacy text. Idempotency still holds because sets are archived
            # rather than deleted, so an imported set the user later archives blocks re-import.
            # changed_at is nullable in the legacy table, and binding NULL would override the
            # column DEFAULT: readers degrade a missing stamp to the epoch, so the imported set
            # would sort as the user's oldest and lose the date its text was actually written.
            # It is coalesced both where rows are ranked and where they are stored. SQLite's
            # one-argument TRIM strips 0x20 only, hence the explicit whitespace character set
            cursor.execute(
                """
                SELECT h.user_id, h.instructions,
                       COALESCE(h.changed_at, CURRENT_TIMESTAMP) AS changed_at
                FROM user_instructions_history h
                WHERE h.id = (
                    SELECT id FROM user_instructions_history
                    WHERE user_id = h.user_id
                    ORDER BY COALESCE(changed_at, CURRENT_TIMESTAMP) DESC, id DESC
                    LIMIT 1
                )
                AND TRIM(h.instructions, char(32)||char(9)||char(10)||char(13)) != ''
                AND NOT EXISTS (
                    SELECT 1 FROM user_instruction_sets s WHERE s.user_id = h.user_id
                )
                """
            )
            for row in cursor.fetchall():
                set_id = str(uuid.uuid4())
                # the original changed_at is kept rather than stamping now, so the
                # imported set does not claim to be the user's newest edit
                cursor.execute(
                    """
                    INSERT INTO user_instruction_sets (id, user_id, name, body, created_at, updated_at)
                    VALUES (?, ?, 'Imported', ?, ?, ?)
                    """,
                    (
                        set_id,
                        row["user_id"],
                        row["instructions"],
                        row["changed_at"],
                        row["changed_at"],
                    ),
                )
                cursor.execute(
                    """
                    INSERT INTO user_instruction_set_history (set_id, user_id, name, body, changed_at, comment)
                    VALUES (?, ?, 'Imported', ?, ?, ?)
                    """,
                    (
                        set_id,
                        row["user_id"],
                        row["instructions"],
                        row["changed_at"],
                        "imported from the removed per-user instructions feature",
                    ),
                )

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
            CREATE INDEX IF NOT EXISTS idx_user_tool_desc_history_user
            ON user_tool_descriptions_history(user_id, tool_name, changed_at DESC)
        """
        )

        # named instruction sets, one row per set (the id is a uuid4 assigned on create).
        # archived_at is a soft delete: chat messages will reference a set by id from a
        # separate SQLite file (chat_history.db), where no foreign key can enforce the
        # relationship, so a hard DELETE would silently orphan those references
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_instruction_sets (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                body TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                archived_at TIMESTAMP
            )
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_user_instruction_sets_user
            ON user_instruction_sets(user_id, archived_at, updated_at DESC)
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_instruction_set_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                set_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                body TEXT NOT NULL,
                changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                comment TEXT
            )
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_user_instruction_set_history_set
            ON user_instruction_set_history(set_id, changed_at DESC)
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

        # tokens were originally issued with no expiry, so a leaked one stayed valid forever
        # unless the user noticed and revoked it. Added by migration rather than in the CREATE
        # above so existing databases pick it up. expires_at holds a *rolling* deadline that
        # every successful validation pushes forward (see validate_api_token), so a token in
        # regular use never expires while an abandoned one does.
        cursor.execute("PRAGMA table_info(user_api_tokens)")
        if "expires_at" not in {row["name"] for row in cursor.fetchall()}:
            cursor.execute("ALTER TABLE user_api_tokens ADD COLUMN expires_at TIMESTAMP")

        self._conn.commit()

    def get_tool_descriptions(self) -> dict[str, ToolDescriptionVersion]:
        """Get the latest description for each tool."""
        conn = self._conn
        # every read in this class does this for the reason given in _discard_stale_transaction:
        # a connection sees its own uncommitted rows, so a DML left pending on it would be read
        # back here as if it had been stored. Defence in depth rather than a live hazard — every
        # write accessor rolls back on failure, so none of them leaves one behind
        self._discard_stale_transaction(conn)
        cursor = conn.cursor()
        # changed_at is CURRENT_TIMESTAMP, which has one-second resolution, so two saves to the
        # same tool inside one second are indistinguishable by it alone. Matching on MAX(changed_at)
        # alone returns *both* tied rows and the loop below keeps whichever the query plan happened
        # to emit last, which is not the newer one under any rule the engine promises to hold, so
        # the admin listing can show a superseded version as current. MAX(id) over the tied group
        # reduces that to one row per tool, ordered by (changed_at, id): id is an AUTOINCREMENT
        # sequence, so it strictly increases with insertion order even when the clock does not move.
        # Written as a grouped IN over the materialised MAX(changed_at) group rather than as a
        # per-row correlated subquery: this table is append-only and never pruned, and the
        # correlated form costs a full table scan that reads and discards every row's description
        # blob — measured an order of magnitude slower by 60k rows, and widening.
        # the join is on IS rather than =, so a tool whose rows all carry a NULL changed_at is
        # listed instead of vanishing: MAX ignores NULLs, so its group's max_changed_at is NULL
        # and = never matches it, while IS is null-safe and leaves the tool its MAX(id) row —
        # the same row get_tool_description returns for it, now that the parse degrades below.
        # It is only the *equality* that is relaxed, and only IS relaxes just that. Dropping the
        # condition lets the highest id in the group win outright, stamped or not, so an unstamped
        # row takes a tool that also holds stamped ones. COALESCE on both sides is null-safe too,
        # but it collapses the NULL rows onto whichever row holds the sentinel, and what that
        # costs depends on which sentinel. Coalescing to '' or 0 cannot cost a tool its usable
        # stamp: any row holding one outranks both sentinels in MAX ('' sorts below every other
        # string, and every integer below every string), so the coalesced key is never the
        # group max where a usable stamp exists. What it costs is agreement in the groups where
        # no row holds one — the plural read then hands the group its highest id, sentinel-stamped
        # or unstamped, while get_tool_description, whose DESC order puts '' and 0 above NULL,
        # keeps returning the sentinel row. Coalescing to CURRENT_TIMESTAMP is the one that costs
        # both: it equals every row written in the current second, so the NULL rows collapse onto
        # a *usably* stamped one, MAX(id) hands the tool to whichever is newest, and an unstamped
        # row wins a tool that holds a usable stamp
        cursor.execute(
            """
            SELECT id, tool_name, description, changed_by, changed_at, comment
            FROM tool_description_history
            WHERE id IN (
                SELECT MAX(t1.id)
                FROM tool_description_history t1
                INNER JOIN (
                    SELECT tool_name, MAX(changed_at) AS max_changed_at
                    FROM tool_description_history
                    GROUP BY tool_name
                ) t2 ON t1.tool_name = t2.tool_name AND t1.changed_at IS t2.max_changed_at
                GROUP BY t1.tool_name
            )
        """
        )
        result = {}
        for row in cursor.fetchall():
            result[row["tool_name"]] = ToolDescriptionVersion(
                id=row["id"],
                tool_name=row["tool_name"],
                description=row["description"],
                changed_by=row["changed_by"],
                changed_at=self._as_utc_or_epoch(row["changed_at"]),
                comment=row["comment"],
            )
        return result

    def get_tool_description(self, tool_name: str) -> ToolDescriptionVersion | None:
        """Get the latest description for a specific tool."""
        conn = self._conn
        self._discard_stale_transaction(conn)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, tool_name, description, changed_by, changed_at, comment
            FROM tool_description_history
            WHERE tool_name = ?
            ORDER BY changed_at DESC, id DESC
            LIMIT 1
            """,
            (tool_name,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        # the stamp is parsed leniently for the reason given on _as_utc_or_epoch: the column
        # carries no NOT NULL, so the winning row can hold a NULL or an unparseable string, and
        # raising here would fail the whole read where get_tool_descriptions returns a value
        return ToolDescriptionVersion(
            id=row["id"],
            tool_name=row["tool_name"],
            description=row["description"],
            changed_by=row["changed_by"],
            changed_at=self._as_utc_or_epoch(row["changed_at"]),
            comment=row["comment"],
        )

    def save_tool_description(
        self, tool_name: str, description: str, user: str, comment: str | None = None
    ) -> ToolDescriptionVersion:
        """Save a new version of a tool description."""
        conn = self._conn
        self._discard_stale_transaction(conn)
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                INSERT INTO tool_description_history (tool_name, description, changed_by, comment)
                VALUES (?, ?, ?, ?)
                """,
                (tool_name, description, user, comment),
            )
            # every write accessor in this class commits inside its try: python's legacy
            # isolation_level opens a transaction before the DML, and both a failed statement and
            # a COMMIT that raises leave it open, holding the write lock against every other
            # writer of this file for the life of this thread's cached connection. _init_db is
            # the one write not guarded this way — it runs from __init__, so a failure there
            # aborts construction and no accessor ever inherits the connection
            conn.commit()
        except BaseException:
            conn.rollback()
            raise

        # aware UTC, as the reads return: the row itself holds CURRENT_TIMESTAMP, which is UTC, so
        # a naive local now() would make this response disagree with the value just stored and
        # with the GET on the same key by the size of the process's UTC offset
        return ToolDescriptionVersion(
            id=cursor.lastrowid,
            tool_name=tool_name,
            description=description,
            changed_by=user,
            changed_at=datetime.now(timezone.utc),
            comment=comment,
        )

    def get_tool_description_history(
        self, tool_name: str, limit: int = 20
    ) -> list[ToolDescriptionVersion]:
        """Get recent versions of a specific tool description."""
        conn = self._conn
        self._discard_stale_transaction(conn)
        cursor = conn.cursor()
        # id DESC for the same reason as in get_tool_description, which this listing has to agree
        # with: versions saved in the same second tie on changed_at, so without it the row shown
        # first is not necessarily the one in force, and the LIMIT can drop the newest version
        # rather than the oldest.
        # _as_utc, not a bare fromisoformat, so the head of this history is the same aware UTC
        # value get_tool_description reports for the same row and the two string-match. It still
        # raises on a stamp it cannot parse, unlike the reads above: this listing has no winner to
        # pick, so there is nothing for a degraded row to get wrong, and the raise is tracked
        # separately
        cursor.execute(
            """
            SELECT id, tool_name, description, changed_by, changed_at, comment
            FROM tool_description_history
            WHERE tool_name = ?
            ORDER BY changed_at DESC, id DESC
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
                changed_at=self._as_utc(row["changed_at"]),
                comment=row["comment"],
            )
            for row in cursor.fetchall()
        ]

    # user comments

    def get_user_comments(self, user_id: str) -> list[UserComment]:
        """Get all comments for a user."""
        conn = self._conn
        self._discard_stale_transaction(conn)
        cursor = conn.cursor()
        # id DESC because created_at is CURRENT_TIMESTAMP and has one-second resolution: two
        # comments submitted inside one second tie on it, and SQLite is then free to return them
        # in either order, differently between two calls. id is an AUTOINCREMENT sequence, so it
        # orders the tied rows by the order they were written
        cursor.execute(
            """
            SELECT id, user_id, comment, created_at
            FROM user_comments
            WHERE user_id = ?
            ORDER BY created_at DESC, id DESC
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

    def list_all_user_comments(self) -> list[UserComment]:
        """List all user comments across all users, ordered by created_at DESC."""
        conn = self._conn
        self._discard_stale_transaction(conn)
        cursor = conn.cursor()
        # id DESC as in get_user_comments. It matters more here: this listing feeds the paginated
        # admin feedback feed, so an order that is only decided for the untied rows moves items
        # across a page boundary between two requests
        cursor.execute(
            """
            SELECT id, user_id, comment, created_at
            FROM user_comments
            ORDER BY created_at DESC, id DESC
            """
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
        conn = self._conn
        self._discard_stale_transaction(conn)
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO user_comments (user_id, comment) VALUES (?, ?)",
                (user_id, comment),
            )
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        return UserComment(
            id=cursor.lastrowid,
            user_id=user_id,
            comment=comment,
            created_at=datetime.now(),
        )

    def delete_user_comment(self, user_id: str, comment_id: int) -> bool:
        """Delete a user's comment."""
        conn = self._conn
        self._discard_stale_transaction(conn)
        cursor = conn.cursor()
        try:
            cursor.execute(
                "DELETE FROM user_comments WHERE id = ? AND user_id = ?",
                (comment_id, user_id),
            )
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        return cursor.rowcount > 0

    # user settings

    def get_user_setting(self, user_id: str, setting_key: str) -> UserSetting | None:
        """Get a user's latest value for a specific setting."""
        conn = self._conn
        self._discard_stale_transaction(conn)
        cursor = conn.cursor()
        # id DESC is what actually decides this: changed_at comes from CURRENT_TIMESTAMP and has
        # one-second resolution, so two saves to the same key inside one second tie on it and
        # LIMIT 1 keeps whichever the scan reached first — in practice the *older* row, so a
        # setting the user changed twice in quick succession silently reverts to the first value.
        # id is an AUTOINCREMENT sequence, so it orders the tied rows by the order they were
        # written even when the clock has not moved
        cursor.execute(
            """
            SELECT id, user_id, setting_key, setting_value, changed_at, comment
            FROM user_settings_history
            WHERE user_id = ? AND setting_key = ?
            ORDER BY changed_at DESC, id DESC
            LIMIT 1
            """,
            (user_id, setting_key),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        # as in get_tool_description: the winning row's stamp may be NULL or unparseable, and it
        # degrades to the epoch rather than raising so that one bad row does not 500 the read
        return UserSetting(
            id=row["id"],
            user_id=row["user_id"],
            setting_key=row["setting_key"],
            setting_value=row["setting_value"],
            changed_at=self._as_utc_or_epoch(row["changed_at"]),
            comment=row["comment"],
        )

    def get_user_settings(self, user_id: str) -> dict[str, UserSetting]:
        """Get all latest settings for a user."""
        conn = self._conn
        self._discard_stale_transaction(conn)
        cursor = conn.cursor()
        # as in get_user_setting, and it must agree with it: matching on MAX(changed_at) alone
        # returns every row that ties for the newest second, and the loop below then keeps
        # whichever of them the query plan emitted last. That is not a rule the engine promises
        # to hold — the same rows come back in the opposite order under a different plan — so the
        # value read here could disagree both with get_user_setting and with itself between calls.
        # MAX(id) over the tied group leaves exactly one row per key, the (changed_at, id) winner,
        # so the later write wins and row order decides nothing. The tiebreak stays inside the
        # MAX(changed_at) group rather than becoming a per-row correlated subquery so that both
        # sides of the join are built by seeking idx_user_settings_history_user on this user_id,
        # and nobody else's rows are touched. How far into that index each side seeks, and which
        # side drives the join, is the planner's choice and moves with the table statistics —
        # nothing in this project runs ANALYZE, so deployed databases have none at all — so no
        # particular plan is being relied on here. Correlating the tiebreak instead re-runs an
        # ORDER BY ... LIMIT 1 lookup once per row the user ever wrote, on a table that is never
        # pruned: measured at 10k rows for one user, ~20 ms against 3-6 ms for this shape
        # depending on statistics, and the endpoint runs it once per request. The join is on IS
        # rather than =, so a key whose rows all carry a NULL changed_at is returned instead of
        # vanishing: MAX ignores NULLs, so that group's max_changed_at is NULL and = never matches
        # it, while IS is null-safe and leaves the key its MAX(id) row — the row get_user_setting
        # returns for it, now that the parse below degrades. Only the equality is relaxed, and only
        # IS relaxes just that: dropping the condition lets the highest id in the group win
        # whatever it is stamped with, so an unstamped row takes a key that also holds stamped
        # ones, and coalescing both sides collapses the NULL rows onto whichever row holds the
        # sentinel — which costs this key its usable stamp only for a sentinel a usably stamped
        # row can itself hold, i.e. CURRENT_TIMESTAMP, and for '' or 0 costs instead the agreement
        # between this read and get_user_setting on keys where no row is usably stamped. See
        # get_tool_descriptions, which spells the two cases out
        cursor.execute(
            """
            SELECT id, user_id, setting_key, setting_value, changed_at, comment
            FROM user_settings_history
            WHERE id IN (
                SELECT MAX(h1.id)
                FROM user_settings_history h1
                INNER JOIN (
                    SELECT setting_key, MAX(changed_at) AS max_changed_at
                    FROM user_settings_history
                    WHERE user_id = ?
                    GROUP BY setting_key
                ) h2 ON h1.setting_key = h2.setting_key AND h1.changed_at IS h2.max_changed_at
                WHERE h1.user_id = ?
                GROUP BY h1.setting_key
            )
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
                    changed_at=self._as_utc_or_epoch(row["changed_at"]),
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
        conn = self._conn
        self._discard_stale_transaction(conn)
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                INSERT INTO user_settings_history (user_id, setting_key, setting_value, comment)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, setting_key, setting_value, comment),
            )
            conn.commit()
        except BaseException:
            conn.rollback()
            raise

        # aware UTC as in save_tool_description, and for the same reason
        return UserSetting(
            id=cursor.lastrowid,
            user_id=user_id,
            setting_key=setting_key,
            setting_value=setting_value,
            changed_at=datetime.now(timezone.utc),
            comment=comment,
        )

    def delete_user_setting(self, user_id: str, setting_key: str) -> bool:
        """Delete a user setting (soft delete by inserting empty value)."""
        conn = self._conn
        self._discard_stale_transaction(conn)
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                INSERT INTO user_settings_history (user_id, setting_key, setting_value, comment)
                VALUES (?, ?, '', 'Reset to default')
                """,
                (user_id, setting_key),
            )
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        return True

    # user instruction sets

    @classmethod
    def _instruction_set_from_row(cls, row: sqlite3.Row) -> InstructionSet:
        # the stored body is returned whole even when it exceeds the cap: a read that clamped
        # it would be handed back by the next edit and silently destroy the rest of the text.
        # Bounding the prompt is the job of whoever pays the token cost
        body = row["body"]
        return InstructionSet(
            id=row["id"],
            user_id=row["user_id"],
            name=row["name"],
            body=body,
            created_at=cls._as_utc_or_epoch(row["created_at"]),
            updated_at=cls._as_utc_or_epoch(row["updated_at"]),
            archived_at=(
                cls._as_utc_or_epoch(row["archived_at"]) if row["archived_at"] else None
            ),
            body_over_cap=len(body) > INSTRUCTION_SET_MAX_BODY_CHARS,
        )

    @staticmethod
    def _discard_stale_transaction(conn: sqlite3.Connection) -> None:
        """Roll back a transaction an earlier failed write left open on this connection.

        Defence in depth rather than a fix for a live leak: python's legacy isolation_level
        opens a transaction before any DML, but every write accessor in this class rolls one
        back on failure, so no ordinary failure path leaves one behind. What can still be found
        open on entry is a rollback() that itself raised, or a DML composed directly on
        db._conn rather than through an accessor (the tests do the latter deliberately). This
        makes either survivable instead of permanent: the connection is cached for the life of
        the thread, so an open transaction would otherwise hold locks against every other
        writer of the file until that thread happened to commit something else, the BEGIN
        IMMEDIATE in create_instruction_set and update_instruction_set would raise every time
        it was reached, and a plain UPDATE would silently commit the abandoned write with its
        own.

        Discarding is safe because whatever is pending is by construction an abandoned write:
        a write accessor commits before it returns, so anything still open belongs to a call
        that raised. The only code that runs many DMLs under a single commit is _init_db and
        the migration it calls, and no accessor can ever be inside it — it runs only from
        __init__, calls no accessor, commits before returning, and Singleton publishes the
        instance only once __init__ has succeeded. The connection belongs to this thread
        alone, so discarding what is pending can lose nothing a caller still expects saved.
        """
        if conn.in_transaction:
            logger.warning("rolling back a transaction left open by an earlier failed write")
            conn.rollback()

    @staticmethod
    def _check_body_cap(body: str) -> None:
        if len(body) > INSTRUCTION_SET_MAX_BODY_CHARS:
            raise InstructionSetBodyTooLong(
                f"instruction set body is {len(body)} chars, "
                f"the maximum is {INSTRUCTION_SET_MAX_BODY_CHARS}"
            )

    def list_instruction_sets(self, user_id: str) -> list[InstructionSet]:
        """List a user's non-archived instruction sets, most recently edited first.

        Returns at most INSTRUCTION_SET_MAX_PER_USER rows: the cap is re-applied here because
        a user can hold more than it allows once it is lowered under them.
        """
        conn = self._conn
        # a connection always sees its own uncommitted rows, so a DML left pending on it would
        # be read back here as if it had been saved. Defence in depth — every write accessor
        # rolls back on failure, so none of them leaves one behind — but where something else
        # does, ending that transaction is the only way for a read to see committed state,
        # and rolling back is the only end that keeps an abandoned write abandoned: committing
        # it here would store what its caller was told had failed
        self._discard_stale_transaction(conn)
        cursor = conn.cursor()
        # one row past the cap so an over-cap user can be reported without a second COUNT.
        # updated_at has one-second resolution, so the uuid id breaks ties: arbitrary, but
        # stable, where SQLite would otherwise be free to reorder between calls
        cursor.execute(
            """
            SELECT id, user_id, name, body, created_at, updated_at, archived_at
            FROM user_instruction_sets
            WHERE user_id = ? AND archived_at IS NULL
            ORDER BY updated_at DESC, id DESC
            LIMIT ?
            """,
            (user_id, INSTRUCTION_SET_MAX_PER_USER + 1),
        )
        rows = cursor.fetchall()
        if len(rows) > INSTRUCTION_SET_MAX_PER_USER:
            # only reachable by lowering the cap under an existing user. The hidden sets stay
            # in the table and each one is still readable and archivable by id, but nothing
            # hands those ids out once the listing drops them, so the user is editing a
            # truncated view of what they own. They can still get back under the cap unaided —
            # archiving enough of the sets that are listed brings the live count down to it and
            # the hidden ones reappear here — but they do it by deleting sets they can see to
            # recover sets they cannot, with no way to tell that is what is happening, hence
            # the loud log
            logger.warning(
                "user %s holds more than %d instruction sets, listing the newest %d",
                user_id,
                INSTRUCTION_SET_MAX_PER_USER,
                INSTRUCTION_SET_MAX_PER_USER,
            )
            rows = rows[:INSTRUCTION_SET_MAX_PER_USER]
        return [self._instruction_set_from_row(row) for row in rows]

    def get_instruction_set(self, user_id: str, set_id: str) -> InstructionSet | None:
        """Get one of a user's instruction sets by id, archived or not.

        Archived sets stay readable so a chat message that recorded the id remains resolvable.
        """
        conn = self._conn
        # as in list_instruction_sets: a DML left pending on this connection is still visible to
        # it, and would otherwise be handed to a chat turn as a stored set
        self._discard_stale_transaction(conn)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, user_id, name, body, created_at, updated_at, archived_at
            FROM user_instruction_sets
            WHERE id = ? AND user_id = ?
            """,
            (set_id, user_id),
        )
        row = cursor.fetchone()
        return self._instruction_set_from_row(row) if row else None

    def create_instruction_set(
        self, user_id: str, name: str, body: str, comment: str | None = None
    ) -> InstructionSet:
        """Create a new instruction set for a user.

        Raises InstructionSetBodyTooLong or InstructionSetLimitReached if a cap is exceeded.
        """
        self._check_body_cap(body)
        set_id = str(uuid.uuid4())
        conn = self._conn
        cursor = conn.cursor()
        try:
            self._discard_stale_transaction(conn)
            # BEGIN IMMEDIATE takes the database's write lock before the count is read. Left to
            # itself python opens a deferred transaction at the INSERT only, so the count and
            # the insert are two independently visible steps and concurrent requests — a double
            # click, a retry — all see room under the cap and all succeed. The excess rows then
            # stay live: list_instruction_sets clamps what it shows, but nothing archives them,
            # so the user has to. It sits inside the try so a BEGIN that fails anyway still
            # leaves the connection clean for the next call
            cursor.execute("BEGIN IMMEDIATE")
            cursor.execute(
                "SELECT COUNT(*) FROM user_instruction_sets "
                "WHERE user_id = ? AND archived_at IS NULL",
                (user_id,),
            )
            if cursor.fetchone()[0] >= INSTRUCTION_SET_MAX_PER_USER:
                raise InstructionSetLimitReached(
                    f"user already holds the maximum of {INSTRUCTION_SET_MAX_PER_USER} "
                    "instruction sets"
                )

            # the timestamps are omitted rather than bound as NULL, which would override the
            # column DEFAULTs: readers degrade a missing stamp to the epoch, so a brand new
            # set would sort as the user's oldest and never carry its real creation time
            cursor.execute(
                """
                INSERT INTO user_instruction_sets (id, user_id, name, body)
                VALUES (?, ?, ?, ?)
                """,
                (set_id, user_id, name, body),
            )
            cursor.execute(
                """
                INSERT INTO user_instruction_set_history (set_id, user_id, name, body, comment)
                VALUES (?, ?, ?, ?, ?)
                """,
                (set_id, user_id, name, body, comment),
            )
            # inside the try: COMMIT itself can return SQLITE_BUSY against a concurrent reader,
            # and a commit that raises leaves the transaction open, so the next accessor on this
            # long-lived thread would commit the rows this caller was told had not been written
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        # read back so the returned timestamps are the ones SQLite stored, not a local clock
        created = self.get_instruction_set(user_id, set_id)
        if created is None:  # pragma: no cover - the row was just committed for this user
            raise RuntimeError(f"instruction set {set_id} vanished right after it was created")
        return created

    def update_instruction_set(
        self,
        user_id: str,
        set_id: str,
        name: str | None = None,
        body: str | None = None,
        comment: str | None = None,
    ) -> InstructionSet | None:
        """Update a user's instruction set and append the new version to its history.

        Returns None if the set does not exist for this user or has been archived, including
        archived while this call was in flight — an archived set is deleted as far as the user
        is concerned, so editing one must not report success. Un-archiving, if it is ever
        offered, needs its own accessor that clears archived_at rather than this general editor.

        Omitted fields keep their current value. Raises InstructionSetBodyTooLong if the new
        body exceeds the cap, including when the caller is echoing back a stored over-cap body.
        """
        conn = self._conn
        # before the BEGIN: an abandoned write still open on this connection would fail it,
        # and would otherwise be part of the snapshot read here
        self._discard_stale_transaction(conn)
        cursor = conn.cursor()
        try:
            # as in create, BEGIN IMMEDIATE takes the write lock before the read. The SELECT
            # below supplies whichever of name and body the caller omitted and the UPDATE writes
            # both back, so the pair is a read-modify-write: unprotected, a rename committed
            # between them is read as the old name and silently written back over by a call that
            # only meant to edit the body, with both callers told they succeeded
            cursor.execute("BEGIN IMMEDIATE")
            cursor.execute(
                """
                SELECT name, body FROM user_instruction_sets
                WHERE id = ? AND user_id = ? AND archived_at IS NULL
                """,
                (set_id, user_id),
            )
            current = cursor.fetchone()
            if current is None:
                # the BEGIN above holds the write lock until the transaction is ended
                conn.rollback()
                return None

            if body is not None:
                self._check_body_cap(body)
            new_name = current["name"] if name is None else name
            new_body = current["body"] if body is None else body

            # updated_at must be assigned here: DEFAULT CURRENT_TIMESTAMP fires on INSERT only,
            # so leaving it alone would keep the listing ordered by creation time forever
            cursor.execute(
                """
                UPDATE user_instruction_sets
                SET name = ?, body = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND user_id = ? AND archived_at IS NULL
                """,
                (new_name, new_body, set_id, user_id),
            )
            if cursor.rowcount == 0:
                # archived between the SELECT and the UPDATE. The write lock now shuts other
                # connections out of that window, so this needs an archive reaching this same
                # connection from inside it — which discards the transaction taken above.
                # Nothing was stored, so the history row must not be appended and the caller
                # must get the same None an already-archived set returns; the read-back below
                # would happily return the archived row
                conn.rollback()
                return None
            cursor.execute(
                """
                INSERT INTO user_instruction_set_history (set_id, user_id, name, body, comment)
                VALUES (?, ?, ?, ?, ?)
                """,
                (set_id, user_id, new_name, new_body, comment),
            )
            # inside the try: a COMMIT that raises leaves the transaction open, and the next
            # accessor on this thread would commit the body this caller was told was not saved
            conn.commit()
        except BaseException:
            # without this the already-executed UPDATE stays pending and the next successful
            # write on this connection commits it, leaving a body with no history row
            conn.rollback()
            raise
        return self.get_instruction_set(user_id, set_id)

    def archive_instruction_set(self, user_id: str, set_id: str) -> bool:
        """Archive (soft delete) a user's instruction set.

        Returns False if the user has no such set or it is archived already. There is
        deliberately no hard delete: the row is the only record that a user has already been
        offered the legacy import, and chat messages will reference sets by id from a separate
        database file where no foreign key can catch a dangling reference.
        """
        conn = self._conn
        self._discard_stale_transaction(conn)
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                UPDATE user_instruction_sets
                SET archived_at = CURRENT_TIMESTAMP
                WHERE id = ? AND user_id = ? AND archived_at IS NULL
                """,
                (set_id, user_id),
            )
            conn.commit()
        except BaseException:
            # a failed UPDATE — or a COMMIT that raises, which SQLite allows — still leaves
            # python's implicit transaction open, holding the write lock against every other
            # writer of this file until the thread happens to commit something else
            conn.rollback()
            raise
        return cursor.rowcount > 0

    # user API tokens

    @staticmethod
    def _hash_token(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()

    @staticmethod
    def _idle_ttl_days() -> int:
        """Days of inactivity after which a token expires. 0 disables expiry entirely."""
        return int(os.environ.get("API_TOKEN_TTL_DAYS", "90"))

    @staticmethod
    def _as_utc(value: str) -> datetime:
        """Parse a stored timestamp as UTC. SQLite's CURRENT_TIMESTAMP is naive UTC."""
        parsed = datetime.fromisoformat(value)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

    @classmethod
    def _as_utc_or_epoch(cls, value: object) -> datetime:
        """Parse a stored timestamp that the schema permits to be NULL or malformed.

        No timestamp column in this file carries NOT NULL or a format check, and the column is
        untyped, so a row written outside the accessors — a migration, a manual fix — can hold
        anything SQLite accepts. These are admin and config reads: one unusable row must not 500
        a whole listing, so the stamp degrades to the epoch — which sorts oldest, the same place
        SQLite puts a NULL in a listing's DESC order — instead of raising. Both shapes the column
        actually reaches are covered, and they raise differently: fromisoformat(None) is a
        TypeError, caught here by the isinstance guard along with every other non-textual value,
        and fromisoformat("") a ValueError.
        """
        if isinstance(value, str):
            try:
                return cls._as_utc(value)
            except ValueError:
                pass
        logger.warning("unusable stored timestamp %r, falling back to the epoch", value)
        return datetime(1970, 1, 1, tzinfo=timezone.utc)

    @staticmethod
    def _utc_stamp(value: datetime) -> str:
        """Format as SQLite's own CURRENT_TIMESTAMP does, so the column stays sortable."""
        return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    def create_api_token(self, user_id: str, name: str | None = None) -> tuple[int, str]:
        """Create a new API token. Returns (token_id, plaintext_token)."""
        plaintext = secrets.token_urlsafe(32)
        token_hash = self._hash_token(plaintext)
        token_prefix = plaintext[:8]

        ttl_days = self._idle_ttl_days()
        expires_at = (
            self._utc_stamp(datetime.now(timezone.utc) + timedelta(days=ttl_days))
            if ttl_days > 0
            else None
        )

        conn = self._conn
        self._discard_stale_transaction(conn)
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                INSERT INTO user_api_tokens (user_id, token_hash, token_prefix, name, expires_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_id, token_hash, token_prefix, name, expires_at),
            )
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        return cursor.lastrowid, plaintext

    def list_api_tokens(self, user_id: str) -> list[UserApiToken]:
        """List all tokens for a user (active and inactive)."""
        conn = self._conn
        self._discard_stale_transaction(conn)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, user_id, token_prefix, name, created_at, last_used_at,
                   is_active, expires_at
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
                created_at=self._as_utc(row["created_at"]),
                last_used_at=(
                    self._as_utc(row["last_used_at"]) if row["last_used_at"] else None
                ),
                is_active=bool(row["is_active"]),
                expires_at=(
                    self._as_utc(row["expires_at"]) if row["expires_at"] else None
                ),
            )
            for row in cursor.fetchall()
        ]

    def revoke_api_token(self, user_id: str, token_id: int) -> bool:
        """Revoke a token (sets is_active = 0)."""
        conn = self._conn
        self._discard_stale_transaction(conn)
        cursor = conn.cursor()
        try:
            cursor.execute(
                "UPDATE user_api_tokens SET is_active = 0 WHERE id = ? AND user_id = ?",
                (token_id, user_id),
            )
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        return cursor.rowcount > 0

    def validate_api_token(self, token: str) -> str | None:
        """Validate a token and return user_id if valid, else None.

        Expiry is an *idle* deadline, not an absolute one: a token dies
        API_TOKEN_TTL_DAYS after its last use, and every successful validation pushes the
        deadline forward. A token in regular use therefore never expires, while an abandoned
        or leaked-and-unnoticed one stops working without the user having to rotate it.

        A token whose deadline cannot be written stays valid for this call: see the comment on
        the update below.
        """
        token_hash = self._hash_token(token)
        conn = self._conn
        self._discard_stale_transaction(conn)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, user_id, created_at, last_used_at, expires_at FROM user_api_tokens "
            "WHERE token_hash = ? AND is_active = 1",
            (token_hash,),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        ttl_days = self._idle_ttl_days()
        now = datetime.now(timezone.utc)

        if ttl_days > 0:
            if row["expires_at"]:
                deadline = self._as_utc(row["expires_at"])
            else:
                # issued before expiry existed: derive the deadline from the last activity we
                # know about, so an actively-used legacy token is not killed on first contact
                last_activity = row["last_used_at"] or row["created_at"]
                deadline = self._as_utc(last_activity) + timedelta(days=ttl_days)
            if deadline <= now:
                logger.info("rejected idle-expired API token id=%s", row["id"])
                return None

        new_expires_at = (
            self._utc_stamp(now + timedelta(days=ttl_days)) if ttl_days > 0 else None
        )
        try:
            cursor.execute(
                "UPDATE user_api_tokens SET last_used_at = ?, expires_at = ? WHERE id = ?",
                (self._utc_stamp(now), new_expires_at, row["id"]),
            )
            conn.commit()
        except Exception as exc:
            # the only write in this class that does not propagate its failure. The SELECT above
            # has already proved the token valid and unexpired; this update is bookkeeping, so
            # letting a locked or full database reject the token would take every MCP request
            # down with the write path. Dropping the update errs towards expiry rather than away
            # from it — a deadline that stops being pushed forward eventually ends the token —
            # and the rollback still releases the write lock the failed statement was holding
            conn.rollback()
            # no exc_info: this fires once per authenticated request, so a sustained lock or a
            # full disk would put a stack trace in the log sink for every one of them. sqlite3
            # raises from C, so that trace would be one of the two lines above and nothing more,
            # saying nothing the exception text and the token id do not already say
            logger.warning(
                "could not record use of API token id=%s; its idle deadline was not extended: %s",
                row["id"],
                exc,
            )
        except BaseException:
            conn.rollback()
            raise
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
