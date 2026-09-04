"""Tests for the ChEMBL client transport.

Split like tests/test_uniprot.py: the pure parts (URL building, the origin pin, the
record-key lookup, paging, caching) with no HTTP at all, and the sentinel shapes driven
through a mocked httpx client.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from genetics_mcp_server.tools import chembl
from genetics_mcp_server.tools.chembl import ChEMBLClient
from genetics_mcp_server.tools.executor import ToolExecutor
from genetics_mcp_server.tools.uniprot import _TTLCache


class _Settings:
    """Only the two attributes ChEMBLClient reads."""

    chembl_api_url = "https://www.ebi.ac.uk/chembl/api/data"
    chembl_cache_ttl = 3600


@pytest.fixture(autouse=True)
def clear_chembl_cache():
    """The TTL cache is a module singleton, so a hit would look like 'no HTTP call'."""
    chembl._CACHE.clear()
    yield
    chembl._CACHE.clear()


def _client(http=None) -> ChEMBLClient:
    return ChEMBLClient(http or MagicMock(), _Settings())


def _resp(body=None, status=200, text=""):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = body
    resp.text = text
    return resp


def _page(key, records, next_path=None, total=None):
    return {
        "page_meta": {
            "next": next_path,
            "total_count": len(records) if total is None else total,
        },
        key: list(records),
    }


class TestUrlBuilding:
    def test_only_is_always_sent_alongside_the_filters(self):
        url = _client()._build_url(
            "mechanism", {"target_chembl_id": "CHEMBL235"}, ["molecule_chembl_id", "action_type"]
        )
        assert str(url).startswith("https://www.ebi.ac.uk/chembl/api/data/mechanism.json?")
        assert url.params["target_chembl_id"] == "CHEMBL235"
        assert url.params["only"] == "molecule_chembl_id,action_type"

    def test_an_unprojected_request_is_refused_outright(self):
        with pytest.raises(ValueError):
            _client()._build_url("target", {}, [])

    def test_a_resource_name_cannot_climb_out_of_the_base_path(self):
        url = _client()._build_url("../../admin", {}, ["x"])
        assert url.path.startswith("/chembl/api/data/")

    def test_a_trailing_slash_on_the_configured_base_is_dropped(self):
        settings = _Settings()
        settings.chembl_api_url = "https://www.ebi.ac.uk/chembl/api/data/"
        client = ChEMBLClient(MagicMock(), settings)
        assert client._build_url("target", {}, ["x"]).path == "/chembl/api/data/target.json"


class TestOriginPin:
    @pytest.mark.asyncio
    async def test_a_foreign_host_is_refused_without_a_request(self):
        http = MagicMock()
        http.get = AsyncMock()
        result = await _client(http)._get_url(httpx.URL("http://169.254.169.254/latest/meta-data"))
        assert result["_origin_refused"] is True
        assert result["_status"] is None
        http.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_an_https_to_http_downgrade_is_refused(self):
        result = await _client()._get_url(httpx.URL("http://www.ebi.ac.uk/chembl/api/data/x.json"))
        assert result["_origin_refused"] is True

    @pytest.mark.asyncio
    async def test_a_next_pointing_off_host_is_refused_mid_walk(self):
        client = _client()
        first = _page("targets", [{"target_chembl_id": "CHEMBL235"}], next_path="https://evil.example/x.json")
        with patch.object(client, "_get", new=AsyncMock(return_value=first)):
            result = await client._get_all("target", {}, ["target_chembl_id"])
        assert result["_origin_refused"] is True


class TestRecordKeyLookup:
    def test_the_list_valued_key_that_is_not_page_meta_wins(self):
        for key in ("targets", "mechanisms", "molecules", "activities", "drug_indications"):
            body = _page(key, [{"a": 1}, {"a": 2}])
            assert ChEMBLClient._records(body) == [{"a": 1}, {"a": 2}]

    def test_a_body_with_no_record_list_is_not_a_page(self):
        assert ChEMBLClient._records({"page_meta": {"next": None}}) is None

    def test_an_ambiguous_body_is_not_a_page(self):
        assert ChEMBLClient._records({"page_meta": {}, "targets": [], "extras": []}) is None


@pytest.mark.asyncio
class TestPageWalking:
    async def test_pages_are_concatenated_and_total_count_kept(self):
        client = _client()
        pages = [
            _page("mechanisms", [{"i": 0}], next_path="/chembl/api/data/mechanism.json?offset=1", total=3),
            _page("mechanisms", [{"i": 1}], next_path="/chembl/api/data/mechanism.json?offset=2", total=3),
            _page("mechanisms", [{"i": 2}], total=3),
        ]
        with patch.object(client, "_get", new=AsyncMock(return_value=pages[0])), patch.object(
            client, "_get_url", new=AsyncMock(side_effect=pages[1:])
        ):
            result = await client._get_all("mechanism", {}, ["molecule_chembl_id"])
        assert result["records"] == [{"i": 0}, {"i": 1}, {"i": 2}]
        assert result["total_count"] == 3
        assert result["truncated"] is False

    async def test_the_page_cap_truncates_rather_than_walking_on(self):
        client = _client()
        page = _page("activities", [{"i": 0}], next_path="/chembl/api/data/activity.json?offset=1", total=2267)
        get_url = AsyncMock(return_value=page)
        with patch.object(client, "_get", new=AsyncMock(return_value=page)), patch.object(
            client, "_get_url", new=get_url
        ):
            result = await client._get_all("activity", {}, ["pchembl_value"], max_pages=3)
        assert result["truncated"] is True
        assert len(result["records"]) == 3
        assert result["total_count"] == 2267
        assert get_url.await_count == 2

    async def test_a_failing_page_returns_its_sentinel_not_a_partial_walk(self):
        client = _client()
        first = _page("targets", [{"i": 0}], next_path="/chembl/api/data/target.json?offset=1")
        sentinel = {"_error": "ChEMBL HTTP 500: ", "_status": 500}
        with patch.object(client, "_get", new=AsyncMock(return_value=first)), patch.object(
            client, "_get_url", new=AsyncMock(return_value=sentinel)
        ):
            result = await client._get_all("target", {}, ["target_chembl_id"])
        assert result is sentinel

    async def test_an_unrecognisable_page_is_reported_as_a_shape_error(self):
        client = _client()
        with patch.object(client, "_get", new=AsyncMock(return_value={"page_meta": {}})):
            result = await client._get_all("target", {}, ["target_chembl_id"])
        assert result["_unexpected_shape"] is True

    async def test_an_unparseable_next_is_a_sentinel_not_an_exception(self):
        client = _client()
        first = _page("targets", [{"i": 0}], next_path="http://a:notaport/x.json")
        get_url = AsyncMock()
        with patch.object(client, "_get", new=AsyncMock(return_value=first)), patch.object(
            client, "_get_url", new=get_url
        ):
            result = await client._get_all("target", {}, ["target_chembl_id"])
        assert result["_unusable_next"] is True
        get_url.assert_not_awaited()

    async def test_a_non_string_next_is_a_sentinel_and_makes_no_request(self):
        client = _client()
        first = _page("targets", [{"i": 0}], next_path={"href": "/chembl/api/data/target.json"})
        get_url = AsyncMock()
        with patch.object(client, "_get", new=AsyncMock(return_value=first)), patch.object(
            client, "_get_url", new=get_url
        ):
            result = await client._get_all("target", {}, ["target_chembl_id"])
        assert result["_unusable_next"] is True
        assert result["_status"] is None
        get_url.assert_not_awaited()


@pytest.mark.asyncio
class TestHttpBehaviour:
    """Driven through the shared external_client with a URL router, as test_uniprot does."""

    @pytest.fixture(autouse=True)
    async def setup_executor(self):
        self.executor = ToolExecutor()
        self.client = ChEMBLClient(self.executor.external_client, _Settings())
        yield
        await self.executor.close()

    def _patch_get(self, resolver):
        calls: list[str] = []

        def handler(url, *args, **kwargs):
            text = str(url)
            calls.append(text)
            resp = resolver(text)
            assert resp is not None, f"unexpected request: {text}"
            return resp

        patcher = patch.object(
            self.executor.external_client, "get", new_callable=AsyncMock, side_effect=handler
        )
        return patcher, calls

    async def test_a_404_carries_the_status_and_a_truncated_body(self):
        patcher, _calls = self._patch_get(lambda url: _resp(status=404, text="x" * 500))
        with patcher:
            result = await self.client._get(
                "molecule", {"molecule_chembl_id": "CHEMBL9999999"}, ["pref_name"]
            )
        assert result["_status"] == 404
        assert result["_error"].startswith("ChEMBL HTTP 404: ")
        assert len(result["_error"]) == len("ChEMBL HTTP 404: ") + 200

    async def test_a_500_is_a_sentinel_and_is_not_cached(self):
        patcher, calls = self._patch_get(lambda url: _resp(status=500, text="boom"))
        with patcher:
            first = await self.client._get("target", {"organism": "Homo sapiens"}, ["pref_name"])
            second = await self.client._get("target", {"organism": "Homo sapiens"}, ["pref_name"])
        assert first["_status"] == 500 and second["_status"] == 500
        assert len(calls) == 2

    async def test_a_hit_is_served_from_the_cache_until_the_ttl_lapses(self, monkeypatch):
        now = {"t": 0.0}
        monkeypatch.setattr(chembl, "_CACHE", _TTLCache(clock=lambda: now["t"]))
        body = _page("targets", [{"target_chembl_id": "CHEMBL235"}])
        patcher, calls = self._patch_get(lambda url: _resp(body=body))
        with patcher:
            assert await self.client._get("target", {}, ["target_chembl_id"]) == body
            assert await self.client._get("target", {}, ["target_chembl_id"]) == body
            assert len(calls) == 1
            now["t"] = _Settings.chembl_cache_ttl + 1
            assert await self.client._get("target", {}, ["target_chembl_id"]) == body
            assert len(calls) == 2

    async def test_a_different_projection_is_a_different_cache_entry(self, monkeypatch):
        monkeypatch.setattr(chembl, "_CACHE", _TTLCache(clock=lambda: 0.0))
        patcher, calls = self._patch_get(lambda url: _resp(body=_page("targets", [])))
        with patcher:
            await self.client._get("target", {}, ["target_chembl_id"])
            await self.client._get("target", {}, ["target_chembl_id", "pref_name"])
        assert len(calls) == 2

    async def test_redirects_are_not_followed(self):
        get = AsyncMock(return_value=_resp(body=_page("targets", [])))
        with patch.object(self.executor.external_client, "get", new=get):
            await self.client._get("target", {}, ["target_chembl_id"])
        assert get.await_args.kwargs["follow_redirects"] is False

    async def test_release_reads_chembl_db_version_without_a_projection(self):
        patcher, calls = self._patch_get(
            lambda url: _resp(body={"chembl_db_version": "ChEMBL_37"})
            if url.endswith("/status.json")
            else None
        )
        with patcher:
            assert await self.client.release() == "ChEMBL_37"
        assert calls == ["https://www.ebi.ac.uk/chembl/api/data/status.json"]

    async def test_release_is_none_when_status_is_unreadable(self):
        patcher, _calls = self._patch_get(lambda url: _resp(status=503, text="down"))
        with patcher:
            assert await self.client.release() is None
