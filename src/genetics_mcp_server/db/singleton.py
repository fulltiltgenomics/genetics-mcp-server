"""Singleton metaclass and per-thread connection base for the SQLite databases."""

import sqlite3
import threading
from collections import defaultdict
from typing import Any


class Singleton(type):
    _instances: dict["Singleton", Any] = {}

    def __call__(cls: "Singleton", *args: str, **kwargs: dict[str, Any]) -> Any:
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class ThreadLocalDB(metaclass=Singleton):
    """One SQLite file, one connection per thread, opened on that thread's first use.

    A subclass says what schema the file must have (`_init_db`) and whether its
    connections enforce foreign keys.
    """

    _foreign_keys = False

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._connections: dict[int, sqlite3.Connection] = defaultdict(
            lambda: self._create_connection()
        )
        self._init_db()

    def _create_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        if self._foreign_keys:
            conn.execute("PRAGMA foreign_keys = ON")
        # under journal_mode=delete a reader is refused while a write is being applied (commit
        # takes an EXCLUSIVE lock), and a writer's commit is refused while any reader still
        # holds a read transaction. WAL removes both, which is what lets the nightly analysis
        # job read and write chat history without blocking live chat writers
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    @property
    def _conn(self) -> sqlite3.Connection:
        return self._connections[threading.get_ident()]
