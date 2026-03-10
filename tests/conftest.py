"""Shared test fixtures for genetics-mcp-server tests."""

import os
import tempfile

import pytest

# set test environment before importing app modules
os.environ.setdefault("GENETICS_API_URL", "http://0.0.0.0:2000/api")
# disable auth for tests by default (overrides .env which may have REQUIRE_AUTH=true)
os.environ["REQUIRE_AUTH"] = "false"


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
    os.unlink(db_path)


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
    os.unlink(db_path)


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
