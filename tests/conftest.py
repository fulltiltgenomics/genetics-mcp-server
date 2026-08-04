"""Shared test fixtures for genetics-mcp-server tests."""

import os
import tempfile
from contextlib import contextmanager
from unittest.mock import patch

import pytest

# set test environment before importing app modules
os.environ.setdefault("GENETICS_API_URL", "http://0.0.0.0:2000/api")
# disable auth for tests by default (overrides .env which may have REQUIRE_AUTH=true)
os.environ["REQUIRE_AUTH"] = "false"


@pytest.fixture(autouse=True)
def clear_settings_cache():
    """Defence in depth, not the fix for the order-dependent auth failures — those came from
    an importlib.reload() in test_temperature.py, which was removed. A stale lru_cache was
    investigated and refuted as the cause.

    Both handles are cleared because they can drift apart: production reads get_settings via
    the package (`genetics_mcp_server.config`), while a reload of the submodule rebinds the
    submodule's handle and leaves the package re-exporting the pre-reload function. Clearing
    only one would miss whichever half a future reload leaves behind. Today they are the same
    object, so the second clear is a no-op.
    """
    from genetics_mcp_server.config import get_settings as pkg_get_settings
    from genetics_mcp_server.config.settings import get_settings as mod_get_settings

    def clear_all():
        for fn in (pkg_get_settings, mod_get_settings):
            fn.cache_clear()

    clear_all()
    yield
    clear_all()


@contextmanager
def settings_env(**overrides):
    """Override environment variables and rebuild the settings snapshot from them.

    The only correct way to move REQUIRE_AUTH in a test since genetics-results-suite-pol: both
    the auth gate and /chat/v1/auth's is_admin read it through Settings, so patching one module
    global no longer moves the other — and no longer exists to be patched. The cache is cleared
    on both edges so neither the test nor whatever runs after it sees a snapshot built from the
    other's environment.
    """
    from genetics_mcp_server.config import get_settings as pkg_get_settings
    from genetics_mcp_server.config.settings import get_settings as mod_get_settings

    def clear_all():
        for fn in (pkg_get_settings, mod_get_settings):
            fn.cache_clear()

    with patch.dict(os.environ, {k: str(v) for k, v in overrides.items()}):
        clear_all()
        try:
            yield
        finally:
            clear_all()


def close_and_unlink(db, db_path):
    """Both databases run in WAL mode, which leaves `-wal`/`-shm` sidecars next to the file.
    SQLite only removes them on the last clean close, and the cached connections outlive the
    fixture, so unlinking the database alone strands two files per test in the temp dir.
    Connections opened on worker threads cannot be closed from here, hence the suppression;
    `db` may be None when construction itself failed.
    """
    if db is not None:
        for conn in db._connections.values():
            try:
                conn.close()
            except Exception:
                pass
        db._connections.clear()
    for suffix in ("", "-wal", "-shm"):
        try:
            os.unlink(db_path + suffix)
        except FileNotFoundError:
            pass


@pytest.fixture
def chat_history_db():
    """Create a temporary ChatHistoryDB instance for testing."""
    from genetics_mcp_server.db.chat_history_db import ChatHistoryDB
    from genetics_mcp_server.db.singleton import Singleton

    # clear singleton to allow fresh instance
    if ChatHistoryDB in Singleton._instances:
        del Singleton._instances[ChatHistoryDB]

    # use a temporary file instead of :memory: to allow multiple connections
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    db = ChatHistoryDB(db_path)
    yield db

    # cleanup
    if ChatHistoryDB in Singleton._instances:
        del Singleton._instances[ChatHistoryDB]
    close_and_unlink(db, db_path)


@pytest.fixture
def llm_config_db():
    """Create a temporary LLMConfigDB instance for testing."""
    from genetics_mcp_server.db.llm_config_db import LLMConfigDB
    from genetics_mcp_server.db.singleton import Singleton

    # clear singleton to allow fresh instance
    if LLMConfigDB in Singleton._instances:
        del Singleton._instances[LLMConfigDB]

    # use a temporary file instead of :memory: to allow multiple connections
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    db = LLMConfigDB(db_path)
    yield db

    # cleanup
    if LLMConfigDB in Singleton._instances:
        del Singleton._instances[LLMConfigDB]
    close_and_unlink(db, db_path)


def block_writes(db, table, event="INSERT"):
    """Make writes to a table abort, standing in for a disk error or a lock timeout.

    RAISE(ABORT) undoes the offending statement and leaves the transaction open, which is
    exactly what SQLite does to a failed write in real life. CREATE TRIGGER is not DML, so
    python does not open a transaction for it and it is committed as it runs.
    """
    db._conn.execute(
        f"CREATE TRIGGER block_{table} BEFORE {event} ON {table} "
        "BEGIN SELECT RAISE(ABORT, 'blocked'); END"
    )


def unblock_writes(db, table):
    db._conn.execute(f"DROP TRIGGER IF EXISTS block_{table}")


@pytest.fixture
def test_client():
    """Create a FastAPI TestClient for testing API endpoints."""
    from fastapi.testclient import TestClient

    from genetics_mcp_server.chat_api import app

    with TestClient(app) as client:
        yield client


@pytest.fixture
def tool_executor():
    """Create a ToolExecutor instance for testing."""
    from genetics_mcp_server.tools import ToolExecutor

    executor = ToolExecutor()
    yield executor
