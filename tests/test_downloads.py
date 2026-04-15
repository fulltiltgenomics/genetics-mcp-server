"""Tests for download store, TSV conversion, and download endpoint."""

import os
import time

import pytest

from genetics_mcp_server.download_store import EXPIRED_MESSAGE, DownloadStore
from genetics_mcp_server.llm_service import _convert_to_tsv, _process_download_hints


class TestDownloadStore:
    """Tests for disk-persisted download storage."""

    @pytest.fixture
    def store(self, tmp_path):
        return DownloadStore(str(tmp_path), ttl_seconds=3600)

    def test_store_and_retrieve(self, store):
        data = b"col1\tcol2\nval1\tval2\n"
        download_id = store.store(data, "test.tsv")
        assert download_id

        result = store.get(download_id)
        assert result is not None
        content, filename, content_type = result
        assert content == data
        assert filename == "test.tsv"
        assert content_type == "text/tab-separated-values"

    def test_get_missing_returns_none(self, store):
        assert store.get("nonexistent") is None

    def test_get_invalid_id_returns_none(self, store):
        assert store.get("../etc/passwd") is None
        assert store.get("foo/bar") is None

    def test_expired_entry_returns_none(self, tmp_path):
        store = DownloadStore(str(tmp_path), ttl_seconds=0)
        download_id = store.store(b"data", "test.tsv")
        time.sleep(0.01)
        assert store.get(download_id) is None

    def test_cleanup_expired(self, tmp_path):
        store = DownloadStore(str(tmp_path), ttl_seconds=0)
        store.store(b"data1", "a.tsv")
        store.store(b"data2", "b.tsv")
        time.sleep(0.01)
        removed = store.cleanup_expired()
        assert removed == 2
        # verify files are gone
        remaining = [f for f in os.listdir(str(tmp_path))]
        assert len(remaining) == 0

    def test_cleanup_keeps_valid(self, tmp_path):
        store = DownloadStore(str(tmp_path), ttl_seconds=3600)
        download_id = store.store(b"data", "keep.tsv")
        removed = store.cleanup_expired()
        assert removed == 0
        assert store.get(download_id) is not None


class TestConvertToTsv:
    """Tests for TSV conversion helper."""

    def test_list_of_dicts(self):
        data = {"results": [
            {"gene": "BRCA1", "pvalue": 1e-8},
            {"gene": "TP53", "pvalue": 1e-5},
        ]}
        tsv = _convert_to_tsv(data)
        lines = tsv.decode("utf-8").strip().split("\n")
        assert lines[0] == "gene\tpvalue"
        assert lines[1] == "BRCA1\t1e-08"
        assert lines[2] == "TP53\t1e-05"

    def test_columns_and_rows(self):
        data = {
            "columns": ["variant", "beta", "pvalue"],
            "rows": [
                ["1:100:A:T", 0.5, 1e-8],
                ["2:200:G:C", -0.3, 1e-5],
            ],
        }
        tsv = _convert_to_tsv(data)
        lines = tsv.decode("utf-8").strip().split("\n")
        assert lines[0] == "variant\tbeta\tpvalue"
        assert len(lines) == 3

    def test_empty_results(self):
        assert _convert_to_tsv({"results": []}) == b""

    def test_no_data(self):
        assert _convert_to_tsv({}) == b""


class TestProcessDownloadHints:
    """Tests for download hint processing."""

    def test_download_url_hint(self):
        result = {
            "success": True,
            "results": [{"x": 1}],
            "_download_url": "https://api.example.com/data?format=tsv",
        }
        processed = _process_download_hints(result)
        assert "INCLUDE_IN_RESPONSE" in processed
        assert "/data?format=tsv" in processed["INCLUDE_IN_RESPONSE"]
        assert "_download_url" not in processed

    def test_download_data_hint(self, tmp_path, monkeypatch):
        # patch settings to use temp dir
        monkeypatch.setenv("DOWNLOAD_STORAGE_PATH", str(tmp_path))

        # reset singletons
        import genetics_mcp_server.download_store as ds
        ds._store = None
        from genetics_mcp_server.config import settings as settings_mod
        settings_mod.get_settings.cache_clear()

        result = {
            "success": True,
            "results": [{"gene": "BRCA1"}],
            "_download_data": {
                "results": [{"gene": "BRCA1"}],
                "filename": "test.tsv",
            },
        }
        processed = _process_download_hints(result)
        assert "INCLUDE_IN_RESPONSE" in processed
        assert "/chat/v1/downloads/" in processed["INCLUDE_IN_RESPONSE"]
        assert "_download_data" not in processed

        # cleanup singletons
        ds._store = None
        settings_mod.get_settings.cache_clear()

    def test_failed_result_unchanged(self):
        result = {"success": False, "error": "some error"}
        processed = _process_download_hints(result)
        assert processed == result

    def test_no_hints_unchanged(self):
        result = {"success": True, "results": [{"x": 1}]}
        processed = _process_download_hints(result)
        assert "INCLUDE_IN_RESPONSE" not in processed


class TestStripTrailingLimit:
    """Tests for SQL LIMIT stripping."""

    def test_strips_trailing_limit(self):
        from genetics_mcp_server.tools.executor import ToolExecutor
        sql, stripped = ToolExecutor._strip_trailing_limit("SELECT * FROM t LIMIT 500")
        assert sql == "SELECT * FROM t"
        assert stripped is True

    def test_strips_limit_with_semicolon(self):
        from genetics_mcp_server.tools.executor import ToolExecutor
        sql, stripped = ToolExecutor._strip_trailing_limit("SELECT * FROM t LIMIT 500;")
        assert sql == "SELECT * FROM t"
        assert stripped is True

    def test_strips_limit_with_semicolon_and_space(self):
        from genetics_mcp_server.tools.executor import ToolExecutor
        sql, stripped = ToolExecutor._strip_trailing_limit("SELECT * FROM t LIMIT 500 ;")
        assert sql == "SELECT * FROM t"
        assert stripped is True

    def test_case_insensitive(self):
        from genetics_mcp_server.tools.executor import ToolExecutor
        sql, stripped = ToolExecutor._strip_trailing_limit("SELECT * FROM t limit 100")
        assert sql == "SELECT * FROM t"
        assert stripped is True

    def test_no_limit_unchanged(self):
        from genetics_mcp_server.tools.executor import ToolExecutor
        sql, stripped = ToolExecutor._strip_trailing_limit("SELECT * FROM t WHERE x > 1")
        assert sql == "SELECT * FROM t WHERE x > 1"
        assert stripped is False

    def test_no_limit_with_semicolon(self):
        from genetics_mcp_server.tools.executor import ToolExecutor
        sql, stripped = ToolExecutor._strip_trailing_limit("SELECT * FROM t WHERE x > 1;")
        assert sql == "SELECT * FROM t WHERE x > 1"
        assert stripped is False

    def test_subquery_limit_not_stripped(self):
        from genetics_mcp_server.tools.executor import ToolExecutor
        original = "SELECT * FROM (SELECT * FROM t LIMIT 10) sub ORDER BY x LIMIT 500"
        sql, stripped = ToolExecutor._strip_trailing_limit(original)
        assert sql == "SELECT * FROM (SELECT * FROM t LIMIT 10) sub ORDER BY x"
        assert stripped is True


class TestDownloadEndpoint:
    """Tests for the /chat/v1/downloads/{id} endpoint."""

    def test_missing_download_returns_404(self, test_client):
        response = test_client.get("/chat/v1/downloads/nonexistent")
        assert response.status_code == 404
        assert EXPIRED_MESSAGE in response.json()["detail"]

    def test_valid_download(self, test_client, tmp_path, monkeypatch):
        monkeypatch.setenv("DOWNLOAD_STORAGE_PATH", str(tmp_path))

        import genetics_mcp_server.download_store as ds
        ds._store = None
        from genetics_mcp_server.config import settings as settings_mod
        settings_mod.get_settings.cache_clear()

        store = DownloadStore(str(tmp_path), ttl_seconds=3600)
        ds._store = store

        tsv_data = b"col1\tcol2\nval1\tval2\n"
        download_id = store.store(tsv_data, "results.tsv")

        response = test_client.get(f"/chat/v1/downloads/{download_id}")
        assert response.status_code == 200
        assert response.content == tsv_data
        assert "results.tsv" in response.headers.get("content-disposition", "")

        # cleanup
        ds._store = None
        settings_mod.get_settings.cache_clear()
