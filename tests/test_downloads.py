"""Tests for download store, TSV conversion, and download endpoint."""

import json
import logging
import os
import time

import pytest

from genetics_mcp_server.download_store import EXPIRED_MESSAGE, DownloadStore
from genetics_mcp_server.llm_service import (
    DOWNLOAD_FAILED_NOTE,
    DOWNLOAD_SHAPE_NOTE,
    DownloadShapeError,
    _convert_to_tsv,
    _process_download_hints,
)


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
        """A recognized-but-empty payload is not a defect: there is simply nothing to download."""
        assert _convert_to_tsv({"results": []}) == b""

    def test_unrecognized_payload_raises(self):
        """An empty b"" return here used to make the download vanish with no error at all."""
        with pytest.raises(DownloadShapeError) as exc:
            _convert_to_tsv({})
        assert "'results'" in str(exc.value)


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


@pytest.fixture
def download_dir(tmp_path, monkeypatch):
    """Point the download-store singleton at a temp dir and reset it afterwards."""
    import genetics_mcp_server.download_store as ds
    from genetics_mcp_server.config import settings as settings_mod

    monkeypatch.setenv("DOWNLOAD_STORAGE_PATH", str(tmp_path))
    ds._store = None
    settings_mod.get_settings.cache_clear()
    yield tmp_path
    ds._store = None
    settings_mod.get_settings.cache_clear()


# payloads that a producer can plausibly emit by mistake, keyed by what went wrong.
# the first two are verbatim reproductions of the shapes that shipped to production in
# genetics-results-suite-bef (the five by-gene BigQuery tools) and -buc (get_hla_by_allele).
MALFORMED_DOWNLOADS = {
    "bef_positional_rows_under_results": {
        "results": [["19", 44908822, 12.3], ["19", 44908823, 9.1]],
        "filename": "APOE_asm_qtl.tsv",
    },
    "buc_positional_rows_under_results": {
        "results": [["DRB1*15:01", 0.42, 1e-9]],
        "filename": "DRB1_15_01_hla.tsv",
    },
    "results_is_a_dict": {"results": {"gene": "BRCA1"}, "filename": "x.tsv"},
    "results_mixes_dicts_and_tuples": {"results": [{"a": 1}, ("a", 2)], "filename": "x.tsv"},
    "rows_are_scalars": {"columns": ["a"], "rows": [1, 2], "filename": "x.tsv"},
    "rows_are_dicts": {"columns": ["a"], "rows": [{"a": 1}], "filename": "x.tsv"},
    "columns_without_rows": {"columns": ["a"], "filename": "x.tsv"},
    "neither_key": {"data": [{"a": 1}], "filename": "x.tsv"},
    "payload_is_a_list": [["a", 1]],
    # filename is json.dump'd into the sidecar after the .tsv is written, so a non-str
    # would raise TypeError past the allow-list and orphan the data file
    "filename_is_not_a_str": {"results": [{"a": 1}], "filename": b"x.tsv"},
}


class TestShapeDefectsAreNeverSwallowed:
    """The regression guard for genetics-results-suite-71c.

    `_process_download_hints` used to catch bare Exception, log a warning and return the
    result with `_download_data` already popped, so a malformed payload produced no link, no
    error to the user and no error to the model. That hid the identical positional-rows
    defect twice, months apart. These tests fail if any malformed payload is ever swallowed
    quietly again -- either it raises, or the user is told the download is missing.
    """

    @pytest.mark.parametrize("case", sorted(MALFORMED_DOWNLOADS))
    def test_malformed_payload_raises_naming_both_shapes(self, case):
        """The converter itself must reject every bad shape loudly.

        Asserted directly, not through `_process_download_hints`, which deliberately
        converts this into a note rather than losing the chat turn.
        """
        with pytest.raises(DownloadShapeError) as exc:
            _convert_to_tsv(MALFORMED_DOWNLOADS[case])
        message = str(exc.value)
        assert "_download_data" in message
        # the message must name the observed shape, not just the expected one, so the
        # producer is identifiable from the traceback alone
        assert any(token in message for token in ("list", "dict", "tuple", "int"))

    @pytest.mark.parametrize("case", sorted(MALFORMED_DOWNLOADS))
    def test_malformed_payload_never_returns_quietly(self, case, download_dir):
        """The exact failure mode of bef and buc: a normal-looking result, minus the link."""
        result = {"success": True, "results": [{"a": 1}], "_download_data": MALFORMED_DOWNLOADS[case]}
        try:
            processed = _process_download_hints(result, tool_name="get_hla_by_allele")
        except DownloadShapeError:
            return
        assert "INCLUDE_IN_RESPONSE" in processed, (
            f"{case} was swallowed: no exception and no note, which is exactly how "
            "genetics-results-suite-bef and -buc stayed invisible in production"
        )

    @pytest.mark.parametrize("case", sorted(MALFORMED_DOWNLOADS))
    def test_shape_defect_is_reported_not_fatal(self, case, download_dir, caplog):
        """A bad shape must not kill the chat turn (chat_api turns it into an SSE error).

        Most producers pass the results-api's parsed body through unvalidated, so this is
        as likely to be upstream drift as a local bug; the answer stays, the link goes.
        """
        result = {"success": True, "results": [{"a": 1}], "_download_data": MALFORMED_DOWNLOADS[case]}
        with caplog.at_level(logging.ERROR):
            processed = _process_download_hints(result, tool_name="get_summary_stats")

        assert processed["INCLUDE_IN_RESPONSE"] == DOWNLOAD_SHAPE_NOTE
        assert "_download_data" not in processed
        assert processed["results"] == [{"a": 1}]
        # the note must not send the user back around a deterministic failure, and must not
        # blame storage -- nothing was wrong with the disk
        assert "do not suggest re-running" in DOWNLOAD_SHAPE_NOTE.lower()
        assert "storage" not in DOWNLOAD_SHAPE_NOTE.lower()
        assert DOWNLOAD_SHAPE_NOTE != DOWNLOAD_FAILED_NOTE

        records = [r for r in caplog.records if r.levelno >= logging.ERROR]
        line = "\n".join(r.getMessage() for r in records)
        assert "DOWNLOAD_SHAPE_DEFECT" in line
        assert "get_summary_stats" in line
        # separable from a disk problem in the log
        assert "DOWNLOAD_FAILED" not in line
        assert any(r.exc_info for r in records), "the traceback must identify the producer"

    def test_programming_error_from_the_store_still_propagates(self, download_dir, monkeypatch):
        """Pins the narrow except: only OSError/UnicodeEncodeError may be caught."""
        import genetics_mcp_server.download_store as ds

        def boom(*args, **kwargs):
            raise TypeError("a bytes-like object is required, not 'str'")

        monkeypatch.setattr(ds.DownloadStore, "store", boom)
        result = {
            "success": True,
            "results": [{"a": 1}],
            "_download_data": {"results": [{"a": 1}], "filename": "x.tsv"},
        }
        with pytest.raises(TypeError):
            _process_download_hints(result, tool_name="get_mpra_by_gene")


class TestStorageFailuresAreSurfaced:
    """Failures we still catch must reach the user, not only the log."""

    def test_io_error_surfaces_note_and_error_log(self, download_dir, monkeypatch, caplog):
        import genetics_mcp_server.download_store as ds

        def enospc(*args, **kwargs):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(ds.DownloadStore, "store", enospc)
        result = {
            "success": True,
            "results": [{"gene": "BRCA1"}],
            "_download_data": {"results": [{"gene": "BRCA1"}], "filename": "x.tsv"},
        }
        with caplog.at_level(logging.ERROR):
            processed = _process_download_hints(result, tool_name="get_gene_to_peaks")

        assert processed["INCLUDE_IN_RESPONSE"] == DOWNLOAD_FAILED_NOTE
        assert "_download_data" not in processed
        assert processed["results"] == [{"gene": "BRCA1"}]

        line = "\n".join(r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR)
        assert "DOWNLOAD_FAILED" in line
        assert "get_gene_to_peaks" in line
        assert "results" in line and "dict" in line

    def test_unwritable_storage_path_surfaces_note(self, tmp_path, monkeypatch):
        """The real disk failure, without patching: the store cannot create its directory."""
        import genetics_mcp_server.download_store as ds
        from genetics_mcp_server.config import settings as settings_mod

        blocker = tmp_path / "not-a-dir"
        blocker.write_text("")
        monkeypatch.setenv("DOWNLOAD_STORAGE_PATH", str(blocker))
        ds._store = None
        settings_mod.get_settings.cache_clear()
        try:
            result = {
                "success": True,
                "results": [{"a": 1}],
                "_download_data": {"results": [{"a": 1}], "filename": "x.tsv"},
            }
            processed = _process_download_hints(result, tool_name="query_database")
            assert processed["INCLUDE_IN_RESPONSE"] == DOWNLOAD_FAILED_NOTE
        finally:
            ds._store = None
            settings_mod.get_settings.cache_clear()

    def test_unencodable_upstream_value_surfaces_note(self, download_dir):
        """json.loads accepts lone surrogates that utf-8 cannot encode -- bad data, not a defect."""
        lone_surrogate = json.loads('"\\ud800"')
        result = {
            "success": True,
            "results": [{"a": lone_surrogate}],
            "_download_data": {"results": [{"a": lone_surrogate}], "filename": "x.tsv"},
        }
        processed = _process_download_hints(result, tool_name="get_protein_annotations")
        assert processed["INCLUDE_IN_RESPONSE"] == DOWNLOAD_FAILED_NOTE


class TestHappyPathUnchanged:
    """The two supported shapes still produce a link, and empty results still produce none."""

    def test_row_form_stores_file_and_links_it(self, download_dir):
        result = {
            "success": True,
            "results": [{"gene": "BRCA1"}],
            "_download_data": {"results": [{"gene": "BRCA1"}], "filename": "test.tsv"},
        }
        processed = _process_download_hints(result, owner="alice", tool_name="get_gene_to_peaks")
        assert "/chat/v1/downloads/" in processed["INCLUDE_IN_RESPONSE"]
        assert "_download_data" not in processed

        download_id = processed["INCLUDE_IN_RESPONSE"].split("/chat/v1/downloads/")[1].rstrip(")")
        import genetics_mcp_server.download_store as ds

        stored = ds.get_download_store().get(download_id, requester="alice")
        assert stored is not None
        assert stored[0].decode() == "gene\nBRCA1\n"

    def test_columnar_form_stores_positional_rows(self, download_dir):
        result = {
            "success": True,
            "results": [{"chrom": "19", "pos": 44908822}],
            "_download_data": {
                "columns": ["chrom", "pos"],
                "rows": [["19", 44908822], ["19", 44908823]],
                "filename": "APOE.tsv",
            },
        }
        processed = _process_download_hints(result, tool_name="get_asm_qtl_by_gene")
        assert "/chat/v1/downloads/" in processed["INCLUDE_IN_RESPONSE"]

    def test_empty_results_yields_no_link_and_no_note(self, download_dir):
        result = {
            "success": True,
            "results": [],
            "_download_data": {"results": [], "filename": "x.tsv"},
        }
        processed = _process_download_hints(result, tool_name="get_mpra_by_gene")
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
