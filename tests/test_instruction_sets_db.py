"""Unit tests for the LLMConfigDB instruction-set accessors.

These sets are user-authored prompt text: every accessor is scoped by user_id, the caps are
enforced at write time and only reported on read (a read that clamped a body would feed the
truncation back into the next write), and nothing here may raise on a read path that a chat
turn depends on.
"""

import inspect
import re
import sqlite3
import threading
import uuid

import pytest

from genetics_mcp_server.db import llm_config_db as db_module
from genetics_mcp_server.db.llm_config_db import (
    INSTRUCTION_SET_MAX_BODY_CHARS,
    InstructionSetBodyTooLong,
    InstructionSetLimitReached,
)

USER_A = "a@example.com"
USER_B = "b@example.com"


def stamp_set(db, set_id, stamp, column="updated_at"):
    """Rewrite a stored timestamp, standing in for the passage of time."""
    db._conn.execute(f"UPDATE user_instruction_sets SET {column} = ? WHERE id = ?", (stamp, set_id))
    db._conn.commit()


def insert_raw_set(db, set_id, user_id, name, body):
    """Insert a set behind the accessors, the way the legacy import does."""
    db._conn.execute(
        "INSERT INTO user_instruction_sets (id, user_id, name, body) VALUES (?, ?, ?, ?)",
        (set_id, user_id, name, body),
    )
    db._conn.commit()


def stored_body(db, set_id):
    cursor = db._conn.cursor()
    cursor.execute("SELECT body FROM user_instruction_sets WHERE id = ?", (set_id,))
    return cursor.fetchone()["body"]


def live_count(db, user_id):
    cursor = db._conn.cursor()
    cursor.execute(
        "SELECT COUNT(*) FROM user_instruction_sets WHERE user_id = ? AND archived_at IS NULL",
        (user_id,),
    )
    return cursor.fetchone()[0]


def block_writes(db, table, event="INSERT"):
    """Make writes to a table abort, standing in for a disk error or a lock timeout.

    RAISE(ABORT) undoes the offending statement and leaves the transaction open, which is
    exactly what SQLite does to a failed write in real life.
    """
    db._conn.execute(
        f"CREATE TRIGGER block_{table} BEFORE {event} ON {table} "
        "BEGIN SELECT RAISE(ABORT, 'blocked'); END"
    )


def unblock_writes(db, table):
    db._conn.execute(f"DROP TRIGGER IF EXISTS block_{table}")


def history_of(db, set_id):
    cursor = db._conn.cursor()
    cursor.execute(
        "SELECT * FROM user_instruction_set_history WHERE set_id = ? ORDER BY id",
        (set_id,),
    )
    return cursor.fetchall()


class TestInstructionSetCreate:
    def test_create_returns_the_set(self, llm_config_db):
        created = llm_config_db.create_instruction_set(USER_A, "Statistician", "Use odds ratios")

        assert created.id
        assert created.user_id == USER_A
        assert created.name == "Statistician"
        assert created.body == "Use odds ratios"
        assert created.archived_at is None
        assert created.created_at is not None
        assert created.updated_at is not None

    def test_create_generates_distinct_uuid_ids(self, llm_config_db):
        first = llm_config_db.create_instruction_set(USER_A, "One", "body")
        second = llm_config_db.create_instruction_set(USER_A, "Two", "body")

        assert first.id != second.id
        assert uuid.UUID(first.id).version == 4

    def test_create_writes_a_history_row(self, llm_config_db):
        created = llm_config_db.create_instruction_set(USER_A, "One", "body", comment="first")

        rows = history_of(llm_config_db, created.id)
        assert len(rows) == 1
        assert rows[0]["user_id"] == USER_A
        assert rows[0]["name"] == "One"
        assert rows[0]["body"] == "body"
        assert rows[0]["comment"] == "first"
        assert rows[0]["changed_at"] is not None


class TestInstructionSetWriteCaps:
    def test_body_at_the_cap_is_accepted(self, llm_config_db):
        created = llm_config_db.create_instruction_set(
            USER_A, "Long", "x" * INSTRUCTION_SET_MAX_BODY_CHARS
        )

        assert len(created.body) == INSTRUCTION_SET_MAX_BODY_CHARS

    def test_body_over_the_cap_is_rejected(self, llm_config_db):
        with pytest.raises(InstructionSetBodyTooLong):
            llm_config_db.create_instruction_set(
                USER_A, "Long", "x" * (INSTRUCTION_SET_MAX_BODY_CHARS + 1)
            )

        assert llm_config_db.list_instruction_sets(USER_A) == []

    def test_update_body_over_the_cap_is_rejected(self, llm_config_db):
        created = llm_config_db.create_instruction_set(USER_A, "One", "short")

        with pytest.raises(InstructionSetBodyTooLong):
            llm_config_db.update_instruction_set(
                USER_A, created.id, body="x" * (INSTRUCTION_SET_MAX_BODY_CHARS + 1)
            )

        assert llm_config_db.get_instruction_set(USER_A, created.id).body == "short"
        assert len(history_of(llm_config_db, created.id)) == 1

    def test_count_cap_blocks_the_next_create(self, llm_config_db, monkeypatch):
        monkeypatch.setattr(db_module, "INSTRUCTION_SET_MAX_PER_USER", 3)
        for i in range(3):
            llm_config_db.create_instruction_set(USER_A, f"Set {i}", "body")

        with pytest.raises(InstructionSetLimitReached):
            llm_config_db.create_instruction_set(USER_A, "One too many", "body")

        assert len(llm_config_db.list_instruction_sets(USER_A)) == 3

    def test_archived_sets_do_not_count_towards_the_cap(self, llm_config_db, monkeypatch):
        monkeypatch.setattr(db_module, "INSTRUCTION_SET_MAX_PER_USER", 3)
        sets = [llm_config_db.create_instruction_set(USER_A, f"Set {i}", "body") for i in range(3)]
        llm_config_db.archive_instruction_set(USER_A, sets[0].id)

        replacement = llm_config_db.create_instruction_set(USER_A, "Replacement", "body")

        assert replacement.id in {s.id for s in llm_config_db.list_instruction_sets(USER_A)}

    def test_count_cap_is_per_user(self, llm_config_db, monkeypatch):
        monkeypatch.setattr(db_module, "INSTRUCTION_SET_MAX_PER_USER", 2)
        for i in range(2):
            llm_config_db.create_instruction_set(USER_B, f"B {i}", "body")

        llm_config_db.create_instruction_set(USER_A, "A only", "body")

        assert len(llm_config_db.list_instruction_sets(USER_A)) == 1
        assert len(llm_config_db.list_instruction_sets(USER_B)) == 2
        with pytest.raises(InstructionSetLimitReached):
            llm_config_db.create_instruction_set(USER_B, "B too many", "body")

    def test_concurrent_creates_cannot_exceed_the_count_cap(self, llm_config_db, monkeypatch):
        """The cap was a TOCTOU race: COUNT and INSERT ran in separate transactions, so
        parallel requests — a double click, a retry — all saw room and all succeeded."""
        cap = 6
        monkeypatch.setattr(db_module, "INSTRUCTION_SET_MAX_PER_USER", cap)
        for i in range(cap - 1):
            llm_config_db.create_instruction_set(USER_A, f"Set {i}", "body")

        racers = 12
        start = threading.Barrier(racers)
        outcomes = []
        lock = threading.Lock()

        def create_one(index):
            start.wait()
            try:
                llm_config_db.create_instruction_set(USER_A, f"Race {index}", "body")
                result = "created"
            except InstructionSetLimitReached:
                result = "capped"
            except Exception as exc:  # surfaced below rather than lost in the thread
                result = f"error: {exc!r}"
            with lock:
                outcomes.append(result)

        threads = [threading.Thread(target=create_one, args=(i,)) for i in range(racers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        assert not [o for o in outcomes if o.startswith("error")], outcomes
        assert outcomes.count("created") == 1, outcomes
        assert outcomes.count("capped") == racers - 1, outcomes
        assert live_count(llm_config_db, USER_A) == cap

    def test_the_body_cap_counts_code_points_not_bytes(self, llm_config_db):
        """A future len(body.encode()) would silently halve the cap for non-ASCII text."""
        body = "é" * INSTRUCTION_SET_MAX_BODY_CHARS

        created = llm_config_db.create_instruction_set(USER_A, "Accented", body)

        assert created.body == body
        assert len(created.body.encode()) > INSTRUCTION_SET_MAX_BODY_CHARS
        assert created.body_over_cap is False
        with pytest.raises(InstructionSetBodyTooLong):
            llm_config_db.create_instruction_set(
                USER_A, "One too long", "🧬" * (INSTRUCTION_SET_MAX_BODY_CHARS + 1)
            )


class TestInstructionSetValueValidation:
    """Emptiness is not rejected here, so the HTTP layer above cannot assume it was."""

    def test_empty_name_and_body_are_accepted_and_stored(self, llm_config_db):
        created = llm_config_db.create_instruction_set(USER_A, "", "")

        assert created.name == ""
        assert created.body == ""
        assert llm_config_db.get_instruction_set(USER_A, created.id).body == ""

    def test_whitespace_only_values_are_stored_verbatim(self, llm_config_db):
        created = llm_config_db.create_instruction_set(USER_A, "  ", "\n\t ")

        assert created.name == "  "
        assert created.body == "\n\t "
        assert history_of(llm_config_db, created.id)[0]["body"] == "\n\t "

    def test_update_can_empty_a_body(self, llm_config_db):
        created = llm_config_db.create_instruction_set(USER_A, "One", "body")

        updated = llm_config_db.update_instruction_set(USER_A, created.id, body="")

        assert updated.body == ""


class TestInstructionSetReadCaps:
    def test_over_cap_body_is_returned_whole_and_flagged_on_get(self, llm_config_db):
        insert_raw_set(llm_config_db, "legacy", USER_A, "Imported", "x" * 9000)

        got = llm_config_db.get_instruction_set(USER_A, "legacy")

        assert len(got.body) == 9000
        assert got.body_over_cap is True

    def test_over_cap_body_is_returned_whole_and_flagged_on_list(self, llm_config_db):
        insert_raw_set(llm_config_db, "legacy", USER_A, "Imported", "x" * 9000)

        listed = llm_config_db.list_instruction_sets(USER_A)

        assert len(listed) == 1
        assert len(listed[0].body) == 9000
        assert listed[0].body_over_cap is True

    def test_a_body_within_the_cap_is_not_flagged(self, llm_config_db):
        created = llm_config_db.create_instruction_set(
            USER_A, "At the cap", "x" * INSTRUCTION_SET_MAX_BODY_CHARS
        )

        assert created.body_over_cap is False

    def test_edit_round_trip_of_an_over_cap_set_cannot_shrink_it(self, llm_config_db):
        """The regression: a read that clamped the body handed the truncation to the edit
        dialog, whose save then stored 4000 of the original 9000 chars and lost the rest."""
        insert_raw_set(llm_config_db, "legacy", USER_A, "Imported", "x" * 9000)

        loaded = llm_config_db.get_instruction_set(USER_A, "legacy")
        with pytest.raises(InstructionSetBodyTooLong):
            llm_config_db.update_instruction_set(USER_A, "legacy", body=loaded.body)

        assert len(stored_body(llm_config_db, "legacy")) == 9000

    def test_rename_of_an_over_cap_set_keeps_the_full_body(self, llm_config_db):
        """A cap lowered under an existing row must not turn a rename into data loss."""
        insert_raw_set(llm_config_db, "legacy", USER_A, "Imported", "x" * 9000)

        updated = llm_config_db.update_instruction_set(USER_A, "legacy", name="Renamed")

        assert updated.name == "Renamed"
        assert len(stored_body(llm_config_db, "legacy")) == 9000
        assert len(history_of(llm_config_db, "legacy")[0]["body"]) == 9000

    def test_list_is_capped_when_the_limit_is_lowered(self, llm_config_db, monkeypatch, caplog):
        for i in range(4):
            llm_config_db.create_instruction_set(USER_A, f"Set {i}", "body")
        monkeypatch.setattr(db_module, "INSTRUCTION_SET_MAX_PER_USER", 2)

        with caplog.at_level("WARNING"):
            listed = llm_config_db.list_instruction_sets(USER_A)

        assert len(listed) == 2
        assert USER_A in caplog.text

    def test_sets_hidden_by_a_lowered_cap_stay_readable_by_id(self, llm_config_db, monkeypatch):
        sets = [llm_config_db.create_instruction_set(USER_A, f"Set {i}", "body") for i in range(4)]
        monkeypatch.setattr(db_module, "INSTRUCTION_SET_MAX_PER_USER", 2)

        listed_ids = {s.id for s in llm_config_db.list_instruction_sets(USER_A)}
        hidden = [s for s in sets if s.id not in listed_ids]

        assert hidden
        for s in hidden:
            assert llm_config_db.get_instruction_set(USER_A, s.id) is not None
            assert llm_config_db.archive_instruction_set(USER_A, s.id) is True


class TestInstructionSetUpdate:
    def test_update_changes_name_and_body(self, llm_config_db):
        created = llm_config_db.create_instruction_set(USER_A, "One", "body")

        updated = llm_config_db.update_instruction_set(
            USER_A, created.id, name="Two", body="new body"
        )

        assert updated.name == "Two"
        assert updated.body == "new body"
        assert llm_config_db.get_instruction_set(USER_A, created.id).body == "new body"

    def test_omitted_fields_keep_their_value(self, llm_config_db):
        created = llm_config_db.create_instruction_set(USER_A, "One", "body")

        updated = llm_config_db.update_instruction_set(USER_A, created.id, name="Renamed")

        assert updated.name == "Renamed"
        assert updated.body == "body"

    def test_update_appends_to_history(self, llm_config_db):
        created = llm_config_db.create_instruction_set(USER_A, "One", "body", comment="first")

        llm_config_db.update_instruction_set(USER_A, created.id, body="second", comment="why")

        rows = history_of(llm_config_db, created.id)
        assert [(r["body"], r["comment"]) for r in rows] == [
            ("body", "first"),
            ("second", "why"),
        ]

    def test_update_bumps_updated_at(self, llm_config_db):
        """DEFAULT CURRENT_TIMESTAMP fires on INSERT only, so the UPDATE must set it."""
        created = llm_config_db.create_instruction_set(USER_A, "One", "body")
        stamp_set(llm_config_db, created.id, "2020-01-01 00:00:00", column="created_at")
        stamp_set(llm_config_db, created.id, "2020-01-01 00:00:00")

        updated = llm_config_db.update_instruction_set(USER_A, created.id, body="new")

        assert updated.updated_at > updated.created_at
        assert updated.created_at.year == 2020

    def test_update_of_unknown_id_returns_none(self, llm_config_db):
        assert llm_config_db.update_instruction_set(USER_A, "nope", name="x") is None

    def test_update_of_an_archived_set_returns_none(self, llm_config_db):
        """An archived set is deleted as far as the user is concerned: an edit of one must
        not report success, bump updated_at, or append history."""
        created = llm_config_db.create_instruction_set(USER_A, "One", "body")
        llm_config_db.archive_instruction_set(USER_A, created.id)
        archived_at = llm_config_db.get_instruction_set(USER_A, created.id).archived_at

        assert (
            llm_config_db.update_instruction_set(USER_A, created.id, name="Two", body="new") is None
        )

        still = llm_config_db.get_instruction_set(USER_A, created.id)
        assert still.name == "One"
        assert still.body == "body"
        assert still.archived_at == archived_at
        assert len(history_of(llm_config_db, created.id)) == 1


class TestInstructionSetArchive:
    def test_archive_hides_from_list_but_keeps_it_gettable(self, llm_config_db):
        created = llm_config_db.create_instruction_set(USER_A, "One", "body")

        assert llm_config_db.archive_instruction_set(USER_A, created.id) is True

        assert llm_config_db.list_instruction_sets(USER_A) == []
        archived = llm_config_db.get_instruction_set(USER_A, created.id)
        assert archived is not None
        assert archived.archived_at is not None
        assert archived.body == "body"

    def test_archiving_twice_reports_no_change(self, llm_config_db):
        created = llm_config_db.create_instruction_set(USER_A, "One", "body")
        llm_config_db.archive_instruction_set(USER_A, created.id)
        first_archived_at = llm_config_db.get_instruction_set(USER_A, created.id).archived_at

        assert llm_config_db.archive_instruction_set(USER_A, created.id) is False
        assert (
            llm_config_db.get_instruction_set(USER_A, created.id).archived_at == first_archived_at
        )

    def test_archive_of_unknown_id_returns_false(self, llm_config_db):
        assert llm_config_db.archive_instruction_set(USER_A, "nope") is False

    def test_no_hard_delete_of_instruction_sets_exists(self):
        """A hard delete would let the legacy import resurrect a set the user removed, and
        would strand the chat messages that reference the set by id. Scanning the source
        catches it whatever the accessor ends up being called."""
        source = inspect.getsource(db_module)

        assert not re.search(r"DELETE\s+FROM\s+user_instruction_sets\b", source, re.IGNORECASE)
        assert not re.search(
            r"DROP\s+TABLE\s+(IF\s+EXISTS\s+)?user_instruction_sets\b", source, re.IGNORECASE
        )


class TestInstructionSetTransactionSafety:
    """Connections are cached per thread and server threads are long-lived, so a transaction
    one call leaves open is inherited by every later call on that thread: it fails the next
    BEGIN, holds the write lock against every other writer of the file, and turns an abandoned
    write into a durable one as soon as anything else commits."""

    def test_create_survives_a_transaction_left_open_by_an_earlier_failed_write(
        self, llm_config_db
    ):
        """Nothing outside these accessors rolls back, so any failed write in this class hands
        the next create a connection that is already inside a transaction."""
        insert_raw_set(llm_config_db, "taken", USER_A, "One", "body")
        conn = llm_config_db._conn
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO user_instruction_sets (id, user_id, name, body) VALUES (?, ?, ?, ?)",
                ("taken", USER_A, "Duplicate", "body"),
            )
        assert conn.in_transaction

        created = llm_config_db.create_instruction_set(USER_A, "After", "body")

        assert llm_config_db.get_instruction_set(USER_A, created.id) is not None
        # the failure must be self-limiting, not permanent for the life of the thread
        again = llm_config_db.create_instruction_set(USER_A, "After again", "body")
        assert llm_config_db.get_instruction_set(USER_A, again.id) is not None

    def test_a_failed_create_leaves_the_connection_clean(self, llm_config_db, monkeypatch):
        monkeypatch.setattr(db_module, "INSTRUCTION_SET_MAX_PER_USER", 1)
        llm_config_db.create_instruction_set(USER_A, "One", "body")

        with pytest.raises(InstructionSetLimitReached):
            llm_config_db.create_instruction_set(USER_A, "Two", "body")

        assert llm_config_db._conn.in_transaction is False

    def test_a_failed_history_insert_rolls_the_row_update_back(self, llm_config_db):
        """The append-only invariant: a body may never change without a history row. Left
        pending, the UPDATE is committed by the next successful write on the connection even
        though the caller was told the save failed."""
        created = llm_config_db.create_instruction_set(USER_A, "One", "body")
        block_writes(llm_config_db, "user_instruction_set_history")
        try:
            with pytest.raises(sqlite3.IntegrityError):
                llm_config_db.update_instruction_set(USER_A, created.id, body="never stored")

            assert llm_config_db._conn.in_transaction is False
            assert stored_body(llm_config_db, created.id) == "body"
        finally:
            unblock_writes(llm_config_db, "user_instruction_set_history")

        llm_config_db.create_instruction_set(USER_A, "Two", "body")

        assert stored_body(llm_config_db, created.id) == "body"
        assert len(history_of(llm_config_db, created.id)) == 1

    def test_a_failed_archive_does_not_keep_the_write_lock(self, llm_config_db):
        created = llm_config_db.create_instruction_set(USER_A, "One", "body")
        block_writes(llm_config_db, "user_instruction_sets", event="UPDATE")
        try:
            with pytest.raises(sqlite3.IntegrityError):
                llm_config_db.archive_instruction_set(USER_A, created.id)

            assert llm_config_db._conn.in_transaction is False
        finally:
            unblock_writes(llm_config_db, "user_instruction_sets")

    def test_an_update_that_races_an_archive_stores_nothing(self, llm_config_db, monkeypatch):
        """The archived check runs in update's SELECT, so a Delete landing between it and the
        UPDATE leaves the UPDATE matching no rows. Reported as success it returns a set the
        user deleted, stores nothing, and appends a history row for a body that never existed."""
        created = llm_config_db.create_instruction_set(USER_A, "One", "body")
        original_check = db_module.LLMConfigDB._check_body_cap

        def archive_then_check(body):
            # the one seam between the SELECT and the UPDATE, standing in for a DELETE request
            # arriving from another tab, a double click, or a retry
            llm_config_db.archive_instruction_set(USER_A, created.id)
            original_check(body)

        monkeypatch.setattr(
            db_module.LLMConfigDB, "_check_body_cap", staticmethod(archive_then_check)
        )

        assert llm_config_db.update_instruction_set(USER_A, created.id, body="never stored") is None

        assert stored_body(llm_config_db, created.id) == "body"
        assert len(history_of(llm_config_db, created.id)) == 1
        assert llm_config_db._conn.in_transaction is False

    def test_update_holds_the_write_lock_across_its_read_modify_write(
        self, llm_config_db, monkeypatch
    ):
        """update supplies the field the caller omitted from its own SELECT and writes both
        fields back, so SELECT and UPDATE are a read-modify-write. Unprotected, a rename
        committed in between is read as the old name and written back over: a rename and a body
        edit racing end as the old name with the new body, both callers told they succeeded.
        The probe stands in for the racing writer — it must be locked out while the update is
        in flight, and its rename must apply once the update lets go."""
        created = llm_config_db.create_instruction_set(USER_A, "Original", "body")
        original_check = db_module.LLMConfigDB._check_body_cap
        probe = {}

        def rename_from_another_connection():
            # timeout=0 so a held write lock reports itself at once instead of waiting it out
            other = sqlite3.connect(llm_config_db.db_path, timeout=0)
            try:
                other.execute(
                    "UPDATE user_instruction_sets SET name = 'Renamed' WHERE id = ?",
                    (created.id,),
                )
                other.commit()
                return True
            except sqlite3.OperationalError:
                return False
            finally:
                other.close()

        def rename_then_check(body):
            # the one seam between update's SELECT and its UPDATE
            probe["during"] = rename_from_another_connection()
            original_check(body)

        monkeypatch.setattr(
            db_module.LLMConfigDB, "_check_body_cap", staticmethod(rename_then_check)
        )

        updated = llm_config_db.update_instruction_set(USER_A, created.id, body="new body")

        assert probe["during"] is False, "a concurrent rename interleaved with the update"
        assert updated.body == "new body"
        assert rename_from_another_connection() is True
        after = llm_config_db.get_instruction_set(USER_A, created.id)
        assert (after.name, after.body) == ("Renamed", "new body")

    def test_a_failed_commit_does_not_leave_the_write_pending(self, llm_config_db):
        """SQLite can return SQLITE_BUSY from COMMIT itself, and this file is read on nearly
        every chat request. With the commit outside the try the transaction stayed open, and
        the next accessor on this long-lived thread committed the body — and its history row —
        that the caller had been told was not saved."""
        created = llm_config_db.create_instruction_set(USER_A, "One", "body")

        class FailingCommit(sqlite3.Connection):
            def commit(self):
                raise sqlite3.OperationalError("database is locked")

        ident = threading.get_ident()
        healthy = llm_config_db._connections[ident]
        failing = sqlite3.connect(llm_config_db.db_path, factory=FailingCommit)
        failing.row_factory = sqlite3.Row
        llm_config_db._connections[ident] = failing
        try:
            with pytest.raises(sqlite3.OperationalError):
                llm_config_db.update_instruction_set(USER_A, created.id, body="never stored")

            assert failing.in_transaction is False
        finally:
            failing.close()
            llm_config_db._connections[ident] = healthy

        assert stored_body(llm_config_db, created.id) == "body"
        assert len(history_of(llm_config_db, created.id)) == 1

    def test_reads_do_not_return_rows_an_earlier_failed_write_left_pending(self, llm_config_db):
        """A connection sees its own uncommitted rows, so an abandoned write stays visible to
        every later read on that thread until something ends the transaction — and the read is
        what feeds a chat turn's system prompt. The raw UPDATE below stands in for the DML a
        write that raised elsewhere in this class left behind."""
        created = llm_config_db.create_instruction_set(USER_A, "One", "body")
        conn = llm_config_db._conn
        conn.execute(
            "UPDATE user_instruction_sets SET body = 'never committed' WHERE id = ?",
            (created.id,),
        )
        assert conn.in_transaction

        assert llm_config_db.get_instruction_set(USER_A, created.id).body == "body"
        assert [s.body for s in llm_config_db.list_instruction_sets(USER_A)] == ["body"]

        # discarded rather than carried, so no later write can turn it into a stored body
        assert conn.in_transaction is False
        assert stored_body(llm_config_db, created.id) == "body"


class TestInstructionSetOrdering:
    def test_list_is_newest_edited_first_not_newest_created_first(self, llm_config_db):
        older = llm_config_db.create_instruction_set(USER_A, "Older", "body")
        newer = llm_config_db.create_instruction_set(USER_A, "Newer", "body")
        stamp_set(llm_config_db, older.id, "2020-01-01 00:00:00")
        stamp_set(llm_config_db, newer.id, "2021-01-01 00:00:00")

        assert [s.id for s in llm_config_db.list_instruction_sets(USER_A)] == [
            newer.id,
            older.id,
        ]

        llm_config_db.update_instruction_set(USER_A, older.id, body="edited")

        assert [s.id for s in llm_config_db.list_instruction_sets(USER_A)] == [
            older.id,
            newer.id,
        ]

    def test_sets_edited_in_the_same_second_keep_a_stable_order(self, llm_config_db):
        """updated_at has one-second resolution, so several sets routinely tie. Without the
        id tiebreaker SQLite is free to return the tied rows in a different order each call."""
        sets = [llm_config_db.create_instruction_set(USER_A, f"Set {i}", "body") for i in range(5)]
        for one in sets:
            stamp_set(llm_config_db, one.id, "2026-01-01 00:00:00")

        expected = sorted((s.id for s in sets), reverse=True)
        for _ in range(3):
            assert [s.id for s in llm_config_db.list_instruction_sets(USER_A)] == expected


class TestInstructionSetTimestamps:
    """The columns carry no NOT NULL, so a read must degrade rather than fail a chat turn."""

    def insert_with_stamps(self, db, set_id, created_at, updated_at):
        db._conn.execute(
            """
            INSERT INTO user_instruction_sets (id, user_id, name, body, created_at, updated_at)
            VALUES (?, ?, 'Odd', 'body', ?, ?)
            """,
            (set_id, USER_A, created_at, updated_at),
        )
        db._conn.commit()

    def test_null_timestamps_degrade_instead_of_raising(self, llm_config_db):
        self.insert_with_stamps(llm_config_db, "nulls", None, None)

        got = llm_config_db.get_instruction_set(USER_A, "nulls")

        assert got is not None
        assert got.created_at.year == 1970
        assert got.updated_at.year == 1970

    def test_null_timestamps_do_not_break_the_listing(self, llm_config_db):
        newer = llm_config_db.create_instruction_set(USER_A, "Newer", "body")
        self.insert_with_stamps(llm_config_db, "nulls", None, None)

        listed = llm_config_db.list_instruction_sets(USER_A)

        # a NULL updated_at sorts last in SQLite's DESC order, where the epoch fallback
        # also puts it: the degraded row never claims to be the newest edit
        assert [s.id for s in listed] == [newer.id, "nulls"]

    def test_an_unparseable_timestamp_degrades_instead_of_raising(self, llm_config_db):
        self.insert_with_stamps(llm_config_db, "junk", "not a timestamp", "not a timestamp")

        got = llm_config_db.get_instruction_set(USER_A, "junk")

        assert got.updated_at.year == 1970

    def test_a_null_stamped_set_stays_editable(self, llm_config_db):
        self.insert_with_stamps(llm_config_db, "nulls", None, None)

        updated = llm_config_db.update_instruction_set(USER_A, "nulls", name="Fixed")

        assert updated.name == "Fixed"
        assert updated.updated_at.year > 1970


class TestInstructionSetUserIsolation:
    def test_list_only_returns_the_callers_sets(self, llm_config_db):
        mine = llm_config_db.create_instruction_set(USER_A, "Mine", "body")
        llm_config_db.create_instruction_set(USER_B, "Theirs", "body")

        listed = llm_config_db.list_instruction_sets(USER_A)

        assert [s.id for s in listed] == [mine.id]

    def test_get_does_not_resolve_another_users_set(self, llm_config_db):
        theirs = llm_config_db.create_instruction_set(USER_B, "Theirs", "body")

        assert llm_config_db.get_instruction_set(USER_A, theirs.id) is None

    def test_update_does_not_touch_another_users_set(self, llm_config_db):
        theirs = llm_config_db.create_instruction_set(USER_B, "Theirs", "body")

        assert llm_config_db.update_instruction_set(USER_A, theirs.id, body="hijacked") is None

        assert llm_config_db.get_instruction_set(USER_B, theirs.id).body == "body"
        assert len(history_of(llm_config_db, theirs.id)) == 1

    def test_archive_does_not_touch_another_users_set(self, llm_config_db):
        theirs = llm_config_db.create_instruction_set(USER_B, "Theirs", "body")

        assert llm_config_db.archive_instruction_set(USER_A, theirs.id) is False

        assert llm_config_db.get_instruction_set(USER_B, theirs.id).archived_at is None
        assert len(llm_config_db.list_instruction_sets(USER_B)) == 1
