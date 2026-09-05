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
from genetics_mcp_server.tools.executor import INTERNAL_ERROR_MSG, ToolExecutor
from genetics_mcp_server.tools.uniprot import _is_error, _TTLCache


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


def _client(http=None, uniprot=None) -> ChEMBLClient:
    return ChEMBLClient(http or MagicMock(), _Settings(), uniprot or MagicMock())


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
        client = ChEMBLClient(MagicMock(), settings, MagicMock())
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
        self.client = ChEMBLClient(
            self.executor.external_client, _Settings(), self.executor.uniprot
        )
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


def _target(chembl_id, pref_name, target_type):
    return {
        "target_chembl_id": chembl_id,
        "pref_name": pref_name,
        "target_type": target_type,
    }


# PPARG: CHEMBL235 is the protein itself, the rest are heteromers built on it. The
# SINGLE PROTEIN one is deliberately not first, so choosing it cannot be "took record 0".
_PPARG_TARGETS = [
    _target("CHEMBL2111342", "PPAR-alpha/gamma", "PROTEIN-PROTEIN INTERACTION"),
    _target("CHEMBL235", "Peroxisome proliferator-activated receptor gamma", "SINGLE PROTEIN"),
    _target("CHEMBL2094122", "PPAR-gamma/RXR-alpha", "PROTEIN-PROTEIN INTERACTION"),
]

_UNIPROT_PPARG = {
    "query": "PPARG",
    "input_kind": "symbol",
    "accession": "P37231",
    "entry_name": "PPARG_HUMAN",
    "ambiguous": False,
}

_UNIPROT_ACCESSION = {**_UNIPROT_PPARG, "query": "P37231", "input_kind": "accession"}

# P2RY12 is a real gene symbol that is also valid accession syntax, so only the resolver
# can say which reading is meant; clopidogrel's target hangs off the accession it returns
_P2RY12_TARGETS = [_target("CHEMBL1907600", "P2Y purinoceptor 12", "SINGLE PROTEIN")]

_UNIPROT_P2RY12 = {
    "query": "P2RY12",
    "input_kind": "symbol",
    "accession": "Q9H244",
    "entry_name": "P2RY12_HUMAN",
    "ambiguous": False,
}


@pytest.mark.asyncio
class TestTargetResolution:
    """resolve_target driven through the shared client, with the resolver stubbed."""

    @pytest.fixture(autouse=True)
    async def setup_executor(self):
        self.executor = ToolExecutor()
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

    def _stub_resolver(self, result):
        return patch.object(
            self.executor.uniprot, "resolve", new=AsyncMock(return_value=result)
        )

    async def test_a_symbol_resolves_through_uniprot_and_prefers_the_single_protein(self):
        patcher, calls = self._patch_get(lambda url: _resp(body=_page("targets", _PPARG_TARGETS)))
        with self._stub_resolver(_UNIPROT_PPARG) as resolve, patcher:
            result = await self.executor.chembl.resolve_target("PPARG")

        assert resolve.await_args.args == ("PPARG",)
        assert result["target_chembl_id"] == "CHEMBL235"
        assert result["target_type"] == "SINGLE PROTEIN"
        # the heteromers are visible rather than silently dropped
        assert [t["target_chembl_id"] for t in result["other_targets"]] == [
            "CHEMBL2111342",
            "CHEMBL2094122",
        ]
        resolution = result["resolution"]
        assert resolution["kind"] == "symbol"
        assert resolution["accession"] == "P37231"
        assert resolution["uniprot"] == _UNIPROT_PPARG
        assert resolution["n_targets"] == 3
        assert "P37231" in calls[0] and "organism=Homo+sapiens" in calls[0]
        assert resolution["organism"] == "Homo sapiens"

    async def test_an_accession_input_is_reported_as_one_by_the_resolver(self):
        patcher, calls = self._patch_get(lambda url: _resp(body=_page("targets", _PPARG_TARGETS)))
        with self._stub_resolver(_UNIPROT_ACCESSION) as resolve, patcher:
            result = await self.executor.chembl.resolve_target("P37231")

        assert resolve.await_args.args == ("P37231",)
        assert result["resolution"]["kind"] == "accession"
        assert result["resolution"]["uniprot"] == _UNIPROT_ACCESSION
        assert result["target_chembl_id"] == "CHEMBL235"
        assert "target_components__accession=P37231" in calls[0]

    async def test_an_accession_shaped_gene_symbol_still_goes_through_uniprot(self):
        patcher, calls = self._patch_get(lambda url: _resp(body=_page("targets", _P2RY12_TARGETS)))
        with self._stub_resolver(_UNIPROT_P2RY12) as resolve, patcher:
            result = await self.executor.chembl.resolve_target("P2RY12")

        assert resolve.await_args.args == ("P2RY12",)
        # the accession UniProt found, not the symbol that merely looks like one
        assert "target_components__accession=Q9H244" in calls[0]
        assert "P2RY12" not in calls[0]
        assert result["resolution"]["kind"] == "symbol"
        assert result["resolution"]["accession"] == "Q9H244"
        assert result["target_chembl_id"] == "CHEMBL1907600"

    async def test_a_chembl_target_id_is_used_as_given(self):
        single = [_target("CHEMBL235", "PPAR gamma", "SINGLE PROTEIN")]
        patcher, calls = self._patch_get(lambda url: _resp(body=_page("targets", single)))
        with self._stub_resolver(_UNIPROT_PPARG) as resolve, patcher:
            result = await self.executor.chembl.resolve_target("chembl235")

        resolve.assert_not_awaited()
        assert result["resolution"]["kind"] == "chembl_target"
        assert result["resolution"]["accession"] is None
        assert result["other_targets"] == []
        assert "target_chembl_id=CHEMBL235" in calls[0]
        assert "accession" not in calls[0]

    async def test_a_uniprot_failure_names_the_stage_and_makes_no_chembl_request(self):
        sentinel = {"_error": "UniProt: no match for ZZZZZ", "_status": None, "_no_match": True}
        patcher, calls = self._patch_get(lambda url: None)
        with self._stub_resolver(sentinel), patcher:
            result = await self.executor.chembl.resolve_target("ZZZZZ")

        assert _is_error(result)
        assert result["_stage"] == "uniprot"
        assert result["_no_match"] is True
        assert "UniProt: no match for ZZZZZ" in result["_error"]
        assert calls == []

    async def test_a_chembl_failure_is_a_sentinel_naming_the_chembl_stage(self):
        patcher, _calls = self._patch_get(lambda url: _resp(status=500, text="boom"))
        with self._stub_resolver(_UNIPROT_PPARG), patcher:
            result = await self.executor.chembl.resolve_target("PPARG")

        assert _is_error(result)
        assert result["_status"] == 500
        assert result["_stage"] == "chembl_target"

    async def test_a_gene_with_no_chembl_target_is_a_success_shaped_answer(self):
        patcher, _calls = self._patch_get(lambda url: _resp(body=_page("targets", [])))
        with self._stub_resolver(_UNIPROT_PPARG), patcher:
            result = await self.executor.chembl.resolve_target("PPARG")

        assert not _is_error(result)
        assert result["target_chembl_id"] is None
        assert result["other_targets"] == []
        assert result["resolution"]["n_targets"] == 0
        assert "no ChEMBL target" in result["resolution"]["note"]

    async def test_without_a_single_protein_the_first_match_is_returned_and_said_so(self):
        complexes = [t for t in _PPARG_TARGETS if t["target_type"] != "SINGLE PROTEIN"]
        patcher, _calls = self._patch_get(lambda url: _resp(body=_page("targets", complexes)))
        with self._stub_resolver(_UNIPROT_PPARG), patcher:
            result = await self.executor.chembl.resolve_target("PPARG")

        assert result["target_chembl_id"] == "CHEMBL2111342"
        assert result["target_type"] == "PROTEIN-PROTEIN INTERACTION"
        assert [t["target_chembl_id"] for t in result["other_targets"]] == ["CHEMBL2094122"]
        assert "no SINGLE PROTEIN target" in result["resolution"]["note"]


@pytest.mark.asyncio
class TestExecutorWiring:
    async def test_the_client_shares_the_executors_resolver_and_http_client(self):
        executor = ToolExecutor()
        try:
            assert executor.chembl._uniprot is executor.uniprot
            assert executor.chembl._client is executor.external_client
        finally:
            await executor.close()


def _pages(**by_resource):
    """A URL router: resource name -> the page bodies it serves, in order.

    The last body repeats, so a resource that always answers the same way needs one
    entry. A `(status, text)` tuple stands for an HTTP failure. Paging works because a
    `page_meta.next` for the same resource is simply the next body in its list.
    """
    served = dict.fromkeys(by_resource, 0)

    def resolve(url: str):
        for resource, bodies in by_resource.items():
            if f"/{resource}.json" in url:
                body = bodies[min(served[resource], len(bodies) - 1)]
                served[resource] += 1
                if isinstance(body, tuple):
                    return _resp(status=body[0], text=body[1])
                return _resp(body=body)
        return None

    return resolve


_STATUS = {"chembl_db_version": "ChEMBL_37"}
_ATTRIBUTION = "ChEMBL ChEMBL_37 (CC BY-SA 3.0), EMBL-EBI"


def _mechanism(molecule_id, action_type="AGONIST", moa="PPAR gamma agonist", max_phase=4):
    return {
        "molecule_chembl_id": molecule_id,
        "mechanism_of_action": moa,
        "action_type": action_type,
        "max_phase": max_phase,
    }


def _molecule(molecule_id, pref_name, max_phase=4, **rest):
    return {
        "molecule_chembl_id": molecule_id,
        "pref_name": pref_name,
        "max_phase": max_phase,
        "first_approval": rest.get("first_approval", 1999),
        "withdrawn_flag": rest.get("withdrawn_flag", False),
        "atc_classifications": rest.get("atc_classifications", ["A10BG03"]),
        "molecule_type": rest.get("molecule_type", "Small molecule"),
    }


class _ChEMBLToolCase:
    """Shared plumbing: a real executor, a stubbed UniProt resolver, a URL router."""

    @pytest.fixture(autouse=True)
    async def setup_executor(self):
        self.executor = ToolExecutor()
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

    def _stub_resolver(self, result=None):
        return patch.object(
            self.executor.uniprot,
            "resolve",
            new=AsyncMock(return_value=result or _UNIPROT_PPARG),
        )


# rosiglitazone has two annotated mechanisms on PPARG, so the (drug, mechanism) row shape
# is exercised rather than assumed; the second molecule is on the second batched page
_PPARG_MECHANISMS = [
    _mechanism("CHEMBL121", moa="PPAR gamma agonist"),
    _mechanism("CHEMBL121", moa="PPAR gamma partial agonist", action_type="PARTIAL AGONIST"),
    _mechanism("CHEMBL595", moa="PPAR gamma agonist", max_phase=3),
]
_ROSIGLITAZONE = _molecule("CHEMBL121", "ROSIGLITAZONE", max_phase=4)
_TROGLITAZONE = _molecule(
    "CHEMBL595", "TROGLITAZONE", max_phase=4, withdrawn_flag=True, first_approval=1997
)


def _molecule_pages(records, page_size=1):
    """Split molecule records across pages so the batched join has to walk."""
    pages = []
    for start in range(0, len(records), page_size):
        last = start + page_size >= len(records)
        pages.append(
            _page(
                "molecules",
                records[start : start + page_size],
                next_path=None
                if last
                else f"/chembl/api/data/molecule.json?offset={start + page_size}",
                total=len(records),
            )
        )
    return pages


@pytest.mark.asyncio
class TestDrugTargetsForGene(_ChEMBLToolCase):
    async def test_mechanisms_join_batched_molecules_into_drug_rows(self):
        resolver = _pages(
            status=[_STATUS],
            target=[_page("targets", _PPARG_TARGETS)],
            mechanism=[_page("mechanisms", _PPARG_MECHANISMS)],
            molecule=_molecule_pages([_ROSIGLITAZONE, _TROGLITAZONE]),
        )
        patcher, calls = self._patch_get(resolver)
        with self._stub_resolver(), patcher:
            result = await self.executor.chembl.get_drug_targets_for_gene("PPARG")

        assert result["success"] is True
        assert result["target_chembl_id"] == "CHEMBL235"
        assert result["attribution"] == _ATTRIBUTION
        assert result["n_mechanisms"] == 3
        assert result["count"] == 3
        assert result["resolution"]["accession"] == "P37231"
        # phase 4 first, then by name; the phase-3 mechanism row keeps the molecule's phase
        assert [(d["pref_name"], d["mechanism_of_action"]) for d in result["drugs"]] == [
            ("ROSIGLITAZONE", "PPAR gamma agonist"),
            ("ROSIGLITAZONE", "PPAR gamma partial agonist"),
            ("TROGLITAZONE", "PPAR gamma agonist"),
        ]
        troglitazone = result["drugs"][-1]
        assert troglitazone["max_phase"] == 4.0
        # the molecule is approved; only this mechanism annotation stopped at phase 3
        assert troglitazone["mechanism_max_phase"] == 3.0
        assert troglitazone["withdrawn_flag"] is True
        assert troglitazone["atc_codes"] == ["A10BG03"]
        assert troglitazone["first_approval"] == 1997
        assert result["drugs"][0]["molecule_type"] == "Small molecule"
        # the column is on every row, so the downloaded table's header is stable
        assert result["drugs"][0]["mechanism_max_phase"] is None
        assert "indications" not in result["drugs"][0]
        assert any("molecule_chembl_id__in=CHEMBL121%2CCHEMBL595" in c for c in calls)

    async def test_the_cached_molecule_records_are_never_mutated(self):
        record = _molecule("CHEMBL121", "ROSIGLITAZONE")
        before = dict(record, atc_classifications=list(record["atc_classifications"]))
        resolver = _pages(
            status=[_STATUS],
            target=[_page("targets", _PPARG_TARGETS)],
            mechanism=[_page("mechanisms", [_mechanism("CHEMBL121")])],
            molecule=[_page("molecules", [record])],
        )
        patcher, _calls = self._patch_get(resolver)
        with self._stub_resolver(), patcher:
            result = await self.executor.chembl.get_drug_targets_for_gene("PPARG")

        assert record == before
        # the row is a new dict, and its ATC list is not the cached one
        assert result["drugs"][0]["atc_codes"] is not record["atc_classifications"]

    async def test_a_gene_with_no_chembl_target_is_an_empty_success(self):
        resolver = _pages(
            status=[_STATUS], target=[_page("targets", [])], mechanism=[(500, "never")]
        )
        patcher, calls = self._patch_get(resolver)
        with self._stub_resolver(), patcher:
            result = await self.executor.chembl.get_drug_targets_for_gene("PPARG")

        assert result["success"] is True
        assert result["drugs"] == [] and result["count"] == 0
        assert result["n_mechanisms"] == 0
        assert "no ChEMBL target" in result["note"]
        assert not any("/mechanism.json" in c for c in calls)

    async def test_a_failing_mechanism_page_is_a_result_naming_the_stage(self):
        resolver = _pages(
            status=[_STATUS],
            target=[_page("targets", _PPARG_TARGETS)],
            mechanism=[(500, "boom")],
        )
        patcher, _calls = self._patch_get(resolver)
        with self._stub_resolver(), patcher:
            result = await self.executor.chembl.get_drug_targets_for_gene("PPARG")

        assert result["success"] is False
        assert result["stage"] == "mechanism"
        assert result["error"].startswith("ChEMBL HTTP 500")
        assert result["resolution"]["accession"] == "P37231"

    async def test_a_uniprot_failure_names_the_resolve_stage(self):
        patcher, calls = self._patch_get(lambda url: None)
        sentinel = {"_error": "UniProt: no match for ZZZZZ", "_status": None, "_no_match": True}
        with self._stub_resolver(sentinel), patcher:
            result = await self.executor.chembl.get_drug_targets_for_gene("ZZZZZ")

        assert result["success"] is False
        assert result["stage"] == "uniprot"
        assert result["resolution"] == {"query": "ZZZZZ"}
        assert calls == []

    async def test_an_unknown_phase_survives_min_phase_zero_and_is_dropped_above_it(self):
        # -1 and None are both ChEMBL's "unknown", and neither may be read as phase 0
        molecules = [
            _molecule("CHEMBL1", "APPROVED", max_phase=4),
            _molecule("CHEMBL2", "UNKNOWN_MINUS_ONE", max_phase=-1),
            _molecule("CHEMBL3", "UNKNOWN_NULL", max_phase=None),
            _molecule("CHEMBL4", "PHASE_TWO", max_phase=2),
        ]
        mechanisms = [_mechanism(m["molecule_chembl_id"], max_phase=None) for m in molecules]

        async def run(**kwargs):
            resolver = _pages(
                status=[_STATUS],
                target=[_page("targets", _PPARG_TARGETS)],
                mechanism=[_page("mechanisms", mechanisms)],
                molecule=[_page("molecules", molecules)],
            )
            patcher, _calls = self._patch_get(resolver)
            with self._stub_resolver(), patcher:
                return await self.executor.chembl.get_drug_targets_for_gene("PPARG", **kwargs)

        wide = await run()
        assert [d["pref_name"] for d in wide["drugs"]] == [
            "APPROVED",
            "PHASE_TWO",
            "UNKNOWN_MINUS_ONE",
            "UNKNOWN_NULL",
        ]
        assert wide["drugs"][2]["max_phase"] is None
        narrow = await run(min_phase=3)
        assert [d["pref_name"] for d in narrow["drugs"]] == ["APPROVED"]
        assert narrow["n_mechanisms"] == 4
        # a floor that arrives as a string filters as the number it spells
        as_text = await run(min_phase="3")
        assert [d["pref_name"] for d in as_text["drugs"]] == ["APPROVED"]
        # and a negative one is a floor of zero, so the unknown phases survive
        assert len((await run(min_phase=-2))["drugs"]) == 4

    async def test_max_results_is_clamped_to_one_at_the_bottom_and_100_at_the_top(self):
        molecules = [_molecule(f"CHEMBL{i}", f"DRUG{i:03d}") for i in range(120)]
        mechanisms = [_mechanism(m["molecule_chembl_id"]) for m in molecules]

        async def run(max_results):
            resolver = _pages(
                status=[_STATUS],
                target=[_page("targets", _PPARG_TARGETS)],
                mechanism=[_page("mechanisms", mechanisms)],
                molecule=[_page("molecules", molecules)],
            )
            patcher, _calls = self._patch_get(resolver)
            with self._stub_resolver(), patcher:
                return await self.executor.chembl.get_drug_targets_for_gene(
                    "PPARG", max_results=max_results
                )

        assert (await run(0))["count"] == 1
        capped = await run(500)
        assert capped["count"] == 100
        # what was dropped by the cap is said rather than left to be inferred
        assert capped["n_matching"] == 120
        assert capped["truncated"] is True
        # the batch is chunked at 50, so 120 ids are three requests, not 120
        assert capped["n_mechanisms"] == 120

    async def test_include_indications_batches_them_and_caps_each_drug_at_ten(self):
        indications = [
            {
                "molecule_chembl_id": "CHEMBL121",
                "efo_id": f"EFO:{i:07d}",
                "efo_term": f"condition {i}",
                "mesh_heading": f"Mesh {i}",
                "max_phase_for_ind": i % 5,
            }
            for i in range(12)
        ]
        resolver = _pages(
            status=[_STATUS],
            target=[_page("targets", _PPARG_TARGETS)],
            mechanism=[_page("mechanisms", [_mechanism("CHEMBL121")])],
            molecule=[_page("molecules", [_ROSIGLITAZONE])],
            drug_indication=[_page("drug_indications", indications)],
        )
        patcher, calls = self._patch_get(resolver)
        with self._stub_resolver(), patcher:
            result = await self.executor.chembl.get_drug_targets_for_gene(
                "PPARG", include_indications=True
            )

        row = result["drugs"][0]
        assert row["n_indications"] == 12
        assert len(row["indications"]) == 10
        # 12 indications cycling 0-4, so the kept ten start at the two 4s
        assert [i["max_phase_for_ind"] for i in row["indications"]][:3] == [4, 4, 3]
        assert set(row["indications"][0]) == {
            "efo_id",
            "efo_term",
            "mesh_heading",
            "max_phase_for_ind",
        }
        assert any("molecule_chembl_id__in=CHEMBL121" in c for c in calls)

    async def test_a_non_numeric_min_phase_fails_before_any_request(self):
        patcher, calls = self._patch_get(lambda url: None)
        with self._stub_resolver(), patcher:
            result = await self.executor.chembl.get_drug_targets_for_gene(
                "PPARG", min_phase="abc"
            )

        assert result["success"] is False
        assert result["stage"] == "input"
        assert "min_phase" in result["error"]
        assert calls == []


_METFORMIN = _molecule(
    "CHEMBL1431", "METFORMIN", max_phase=4, atc_classifications=["A10BA02"], first_approval=1995
)

_PPARG_TARGET_DETAIL = {
    "target_chembl_id": "CHEMBL235",
    "pref_name": "Peroxisome proliferator-activated receptor gamma",
    "target_type": "SINGLE PROTEIN",
    "organism": "Homo sapiens",
    "target_components": [
        {
            "accession": "P37231",
            "component_id": 241,
            "target_component_synonyms": [
                {"component_synonym": "PPARG", "syn_type": "GENE_SYMBOL"},
                {"component_synonym": "NR1C3", "syn_type": "GENE_SYMBOL_OTHER"},
                {"component_synonym": "PPAR-gamma", "syn_type": "UNIPROT"},
            ],
            # the live projection cannot narrow the nested objects, so this arrives too
            "target_component_xrefs": [{"xref_id": "8HHP", "xref_src_db": "PDBe"}],
        }
    ],
}


@pytest.mark.asyncio
class TestDrugProfile(_ChEMBLToolCase):
    async def test_a_preferred_name_hit_carries_mechanisms_targets_and_indications(self):
        indications = [
            {
                "efo_id": "EFO:0001360",
                "efo_term": "type II diabetes mellitus",
                "mesh_heading": "Diabetes Mellitus, Type 2",
                "max_phase_for_ind": 4,
            },
            {
                "efo_id": "EFO:0000305",
                "efo_term": "breast carcinoma",
                "mesh_heading": "Breast Neoplasms",
                "max_phase_for_ind": 2,
            },
        ]
        resolver = _pages(
            status=[_STATUS],
            molecule=[_page("molecules", [_ROSIGLITAZONE])],
            mechanism=[
                _page(
                    "mechanisms",
                    [
                        {
                            "target_chembl_id": "CHEMBL235",
                            "mechanism_of_action": "PPAR gamma agonist",
                            "action_type": "AGONIST",
                            "max_phase": 4,
                        }
                    ],
                )
            ],
            target=[_page("targets", [_PPARG_TARGET_DETAIL])],
            drug_indication=[_page("drug_indications", list(reversed(indications)))],
        )
        patcher, calls = self._patch_get(resolver)
        with patcher:
            result = await self.executor.chembl.get_drug_profile("rosiglitazone")

        assert result["success"] is True
        assert result["attribution"] == _ATTRIBUTION
        assert result["resolution"]["kind"] == "pref_name"
        assert result["resolution"]["n_candidates"] == 1
        assert result["drug"]["molecule_chembl_id"] == "CHEMBL121"
        assert result["drug"]["atc_codes"] == ["A10BG03"]
        mechanism = result["mechanisms"][0]
        assert mechanism["target_pref_name"].startswith("Peroxisome")
        assert mechanism["organism"] == "Homo sapiens"
        # only the accession and the GENE_SYMBOL synonyms survive; the xrefs do not
        assert mechanism["components"] == [{"accession": "P37231", "gene_symbols": ["PPARG"]}]
        assert result["n_indications"] == 2
        assert [i["efo_id"] for i in result["indications"]] == ["EFO:0001360", "EFO:0000305"]
        assert any("pref_name__iexact=rosiglitazone" in c for c in calls)

    async def test_a_chembl_id_skips_the_name_ladder(self):
        resolver = _pages(
            status=[_STATUS],
            molecule=[_page("molecules", [_METFORMIN])],
            mechanism=[_page("mechanisms", [])],
            drug_indication=[_page("drug_indications", [])],
        )
        patcher, calls = self._patch_get(resolver)
        with patcher:
            result = await self.executor.chembl.get_drug_profile("chembl1431")

        assert result["resolution"]["kind"] == "chembl_id"
        assert result["drug"]["atc_codes"] == ["A10BA02"]
        assert result["mechanisms"] == [] and result["indications"] == []
        assert any("molecule_chembl_id=CHEMBL1431" in c for c in calls)
        assert not any("iexact" in c for c in calls)

    async def test_a_preferred_name_miss_falls_through_to_the_synonym_list(self):
        resolver = _pages(
            status=[_STATUS],
            molecule=[_page("molecules", []), _page("molecules", [_METFORMIN])],
            mechanism=[_page("mechanisms", [])],
            drug_indication=[_page("drug_indications", [])],
        )
        patcher, calls = self._patch_get(resolver)
        with patcher:
            result = await self.executor.chembl.get_drug_profile("Glucophage")

        assert result["resolution"]["kind"] == "synonym"
        assert result["drug"]["pref_name"] == "METFORMIN"
        molecule_calls = [c for c in calls if "/molecule.json" in c]
        assert "pref_name__iexact=Glucophage" in molecule_calls[0]
        assert "molecule_synonyms__molecule_synonym__iexact=Glucophage" in molecule_calls[1]

    async def test_several_candidates_keep_the_highest_phase_and_list_the_rest(self):
        candidates = [
            _molecule("CHEMBL9", "ASPIRIN SALT", max_phase=None),
            _molecule("CHEMBL25", "ASPIRIN", max_phase=4),
            _molecule("CHEMBL10", "ASPIRIN COMPLEX", max_phase=2),
        ]
        resolver = _pages(
            status=[_STATUS],
            molecule=[_page("molecules", candidates)],
            mechanism=[_page("mechanisms", [])],
            drug_indication=[_page("drug_indications", [])],
        )
        patcher, _calls = self._patch_get(resolver)
        with patcher:
            result = await self.executor.chembl.get_drug_profile("aspirin")

        assert result["drug"]["molecule_chembl_id"] == "CHEMBL25"
        assert result["resolution"]["n_candidates"] == 3
        assert [c["molecule_chembl_id"] for c in result["resolution"]["other_candidates"]] == [
            "CHEMBL9",
            "CHEMBL10",
        ]

    async def test_the_indication_download_is_whole_while_the_result_is_capped(self):
        indications = [
            {
                "efo_id": f"EFO:{i:07d}",
                "efo_term": f"condition {i}",
                "mesh_heading": f"Mesh {i}",
                "max_phase_for_ind": 4,
            }
            for i in range(60)
        ]
        resolver = _pages(
            status=[_STATUS],
            molecule=[_page("molecules", [_METFORMIN])],
            mechanism=[_page("mechanisms", [])],
            drug_indication=[_page("drug_indications", indications)],
        )
        patcher, _calls = self._patch_get(resolver)
        with patcher:
            result = await self.executor.get_drug_profile("metformin")

        assert len(result["indications"]) == 50
        assert result["n_indications"] == 60
        assert "_all_indications" not in result
        assert len(result["_download_data"]["results"]) == 60

    async def test_a_tie_on_phase_prefers_the_parent_over_its_salt(self):
        salt = _molecule("CHEMBL1200653", "ASPIRIN SODIUM", max_phase=4)
        salt["molecule_hierarchy"] = {
            "molecule_chembl_id": "CHEMBL1200653",
            "parent_chembl_id": "CHEMBL25",
        }
        parent = _molecule("CHEMBL25", "ASPIRIN", max_phase=4)
        parent["molecule_hierarchy"] = {
            "molecule_chembl_id": "CHEMBL25",
            "parent_chembl_id": "CHEMBL25",
        }
        resolver = _pages(
            status=[_STATUS],
            # the salt is listed first, so choosing the parent cannot be "took record 0"
            molecule=[_page("molecules", [salt, parent])],
            mechanism=[_page("mechanisms", [])],
            drug_indication=[_page("drug_indications", [])],
        )
        patcher, _calls = self._patch_get(resolver)
        with patcher:
            result = await self.executor.chembl.get_drug_profile("aspirin")

        assert result["drug"]["molecule_chembl_id"] == "CHEMBL25"
        assert "preferring a parent molecule over its salts" in result["resolution"]["note"]

    async def test_no_molecule_anywhere_is_a_success_with_a_null_drug(self):
        resolver = _pages(status=[_STATUS], molecule=[_page("molecules", [])])
        patcher, _calls = self._patch_get(resolver)
        with patcher:
            result = await self.executor.chembl.get_drug_profile("notadrug")

        assert result["success"] is True
        assert result["drug"] is None
        assert result["n_indications"] == 0
        assert "no ChEMBL molecule matches" in result["note"]
        assert result["resolution"]["kind"] is None

    async def test_a_failing_mechanism_page_is_a_result_naming_the_stage(self):
        resolver = _pages(
            status=[_STATUS],
            molecule=[_page("molecules", [_METFORMIN])],
            mechanism=[(500, "boom")],
        )
        patcher, _calls = self._patch_get(resolver)
        with patcher:
            result = await self.executor.chembl.get_drug_profile("CHEMBL1431")

        assert result["success"] is False
        assert result["stage"] == "mechanism"
        assert result["resolution"]["kind"] == "chembl_id"

    async def test_a_failing_molecule_lookup_names_the_molecule_stage(self):
        resolver = _pages(status=[_STATUS], molecule=[(503, "down")])
        patcher, _calls = self._patch_get(resolver)
        with patcher:
            result = await self.executor.chembl.get_drug_profile("aspirin")

        assert result["success"] is False
        assert result["stage"] == "molecule"
        assert result["resolution"] == {"query": "aspirin"}


def _activity(molecule_id, pchembl, standard_type="IC50"):
    return {
        "molecule_chembl_id": molecule_id,
        "standard_type": standard_type,
        "standard_relation": "=",
        "pchembl_value": pchembl,
        "assay_chembl_id": "CHEMBL1234",
    }


@pytest.mark.asyncio
class TestTargetBioactivity(_ChEMBLToolCase):
    async def test_activities_collapse_to_one_row_per_molecule_best_first(self):
        activities = [
            _activity("CHEMBL121", "7.2"),
            _activity("CHEMBL121", "8.4", standard_type="Ki"),
            _activity("CHEMBL121", "6.1"),
            _activity("CHEMBL595", "9.0", standard_type="EC50"),
            _activity("CHEMBL9999", "6.5"),
        ]
        resolver = _pages(
            status=[_STATUS],
            target=[_page("targets", _PPARG_TARGETS)],
            activity=[_page("activities", activities, total=4210)],
            molecule=[_page("molecules", [_ROSIGLITAZONE, _TROGLITAZONE])],
        )
        patcher, calls = self._patch_get(resolver)
        with self._stub_resolver(), patcher:
            result = await self.executor.chembl.get_target_bioactivity("PPARG")

        assert result["success"] is True
        assert result["n_activities"] == 5
        assert result["total_count"] == 4210
        assert result["n_distinct_molecules"] == 3
        assert result["pchembl_min"] == 6.0
        # carried through from resolve_target, same as get_drug_targets_for_gene
        assert [t["target_chembl_id"] for t in result["other_targets"]] == [
            "CHEMBL2111342",
            "CHEMBL2094122",
        ]
        assert result["by_standard_type"] == {"IC50": 3, "EC50": 1, "Ki": 1}
        top = result["top_compounds"]
        assert [r["molecule_chembl_id"] for r in top] == ["CHEMBL595", "CHEMBL121", "CHEMBL9999"]
        assert top[0]["best_pchembl"] == 9.0 and top[0]["standard_type"] == "EC50"
        assert top[0]["pref_name"] == "TROGLITAZONE" and top[0]["max_phase"] == 4.0
        # the best measurement wins, and every measurement is counted
        assert top[1]["best_pchembl"] == 8.4 and top[1]["n_activities"] == 3
        # a molecule with no record in the batched projection still gets its row
        assert top[2]["pref_name"] is None
        # the top rows are copies, so naming them leaves the download's rows with the
        # columns that are filled for every molecule
        assert [r["molecule_chembl_id"] for r in result["_all_compounds"]] == [
            "CHEMBL595",
            "CHEMBL121",
            "CHEMBL9999",
        ]
        assert set(result["_all_compounds"][0]) == {
            "molecule_chembl_id",
            "best_pchembl",
            "standard_type",
            "n_activities",
        }
        assert any("pchembl_value__gte=6.0" in c for c in calls)

    async def test_the_walk_is_potency_ordered_and_stops_at_the_page_cap(self):
        # six pages are offered; the cap is what stops the walk, and the ordering is what
        # makes the rows it did fetch the top of the distribution rather than a prefix
        pages = [
            _page(
                "activities",
                [_activity(f"CHEMBL{page}{i}", "7.0") for i in range(2)],
                next_path=f"/chembl/api/data/activity.json?offset={page + 1}",
                total=6000,
            )
            for page in range(6)
        ]
        resolver = _pages(
            status=[_STATUS],
            target=[_page("targets", _PPARG_TARGETS)],
            activity=pages,
            molecule=[_page("molecules", [])],
        )
        patcher, calls = self._patch_get(resolver)
        with self._stub_resolver(), patcher:
            result = await self.executor.chembl.get_target_bioactivity("PPARG")

        activity_calls = [c for c in calls if "/activity.json" in c]
        assert len(activity_calls) == 5
        assert "order_by=-pchembl_value" in activity_calls[0]
        assert result["truncated"] is True
        assert result["n_activities"] == 10
        assert result["total_count"] == 6000

    async def test_zero_activities_is_a_success_stating_the_threshold(self):
        resolver = _pages(
            status=[_STATUS],
            target=[_page("targets", _PPARG_TARGETS)],
            activity=[_page("activities", [], total=0)],
        )
        patcher, calls = self._patch_get(resolver)
        with self._stub_resolver(), patcher:
            result = await self.executor.chembl.get_target_bioactivity("PPARG", pchembl_min=11)

        assert result["success"] is True
        assert result["top_compounds"] == [] and result["by_standard_type"] == {}
        assert result["n_activities"] == 0 and result["n_distinct_molecules"] == 0
        assert "pChEMBL 11.0" in result["note"]
        assert not any("/molecule.json" in c for c in calls)

    async def test_pchembl_min_and_max_results_are_clamped(self):
        activities = [_activity(f"CHEMBL{i}", str(6 + i / 100)) for i in range(120)]
        resolver = _pages(
            status=[_STATUS],
            target=[_page("targets", _PPARG_TARGETS)],
            activity=[_page("activities", activities, total=120)],
            molecule=[_page("molecules", [])],
        )
        patcher, calls = self._patch_get(resolver)
        with self._stub_resolver(), patcher:
            result = await self.executor.chembl.get_target_bioactivity(
                "PPARG", pchembl_min=99, max_results=500
            )

        assert result["pchembl_min"] == 14.0
        assert "pchembl_value__gte=14.0" in [c for c in calls if "/activity.json" in c][0]
        assert len(result["top_compounds"]) == 100
        assert len(result["_all_compounds"]) == 120

    async def test_a_negative_max_results_still_returns_one_row(self):
        resolver = _pages(
            status=[_STATUS],
            target=[_page("targets", _PPARG_TARGETS)],
            activity=[_page("activities", [_activity("CHEMBL121", "7.0")], total=1)],
            molecule=[_page("molecules", [_ROSIGLITAZONE])],
        )
        patcher, _calls = self._patch_get(resolver)
        with self._stub_resolver(), patcher:
            result = await self.executor.chembl.get_target_bioactivity("PPARG", max_results=0)

        assert len(result["top_compounds"]) == 1

    async def test_a_failing_activity_walk_is_a_result_naming_the_stage(self):
        resolver = _pages(
            status=[_STATUS], target=[_page("targets", _PPARG_TARGETS)], activity=[(500, "boom")]
        )
        patcher, _calls = self._patch_get(resolver)
        with self._stub_resolver(), patcher:
            result = await self.executor.chembl.get_target_bioactivity("PPARG")

        assert result["success"] is False
        assert result["stage"] == "activity"
        assert result["resolution"]["accession"] == "P37231"

    async def test_a_gene_with_no_target_is_an_empty_summary(self):
        resolver = _pages(status=[_STATUS], target=[_page("targets", [])])
        patcher, calls = self._patch_get(resolver)
        with self._stub_resolver(), patcher:
            result = await self.executor.chembl.get_target_bioactivity("PPARG")

        assert result["success"] is True
        assert result["n_activities"] == 0 and result["top_compounds"] == []
        assert "no ChEMBL target" in result["note"]
        assert not any("/activity.json" in c for c in calls)

    async def test_a_non_numeric_pchembl_min_fails_before_any_request(self):
        patcher, calls = self._patch_get(lambda url: None)
        with self._stub_resolver(), patcher:
            result = await self.executor.chembl.get_target_bioactivity("PPARG", pchembl_min="abc")

        assert result["success"] is False
        assert result["stage"] == "input"
        assert "pchembl_min" in result["error"]
        assert calls == []


@pytest.mark.asyncio
class TestChEMBLExecutorDelegates:
    """The delegates add a download hint and turn an unexpected exception into the
    generic message; everything else is the client's result verbatim."""

    @pytest.fixture(autouse=True)
    async def setup_executor(self):
        self.executor = ToolExecutor()
        yield
        await self.executor.close()

    def _raise(self, name):
        return patch.object(
            self.executor.chembl, name, new=AsyncMock(side_effect=RuntimeError("kaboom"))
        )

    async def test_get_drug_targets_for_gene_hides_an_unexpected_exception(self):
        with self._raise("get_drug_targets_for_gene"):
            result = await self.executor.get_drug_targets_for_gene("PPARG")
        assert result == {"success": False, "error": INTERNAL_ERROR_MSG}

    async def test_get_drug_profile_hides_an_unexpected_exception(self):
        with self._raise("get_drug_profile"):
            result = await self.executor.get_drug_profile("aspirin")
        assert result == {"success": False, "error": INTERNAL_ERROR_MSG}

    async def test_get_target_bioactivity_hides_an_unexpected_exception(self):
        with self._raise("get_target_bioactivity"):
            result = await self.executor.get_target_bioactivity("PPARG")
        assert result == {"success": False, "error": INTERNAL_ERROR_MSG}

    async def test_no_delegate_puts_the_query_in_a_download_filename(self):
        # the filename is interpolated into a Content-Disposition header unescaped, and
        # nothing resolved here, so every delegate must fall back to the constant
        hostile = 'BRAF" ; \u4e2d'
        rows = [{"molecule_chembl_id": str(i)} for i in range(30)]
        with patch.object(
            self.executor.chembl,
            "get_drug_targets_for_gene",
            new=AsyncMock(return_value={"success": True, "target_chembl_id": None, "drugs": rows}),
        ):
            drugs = await self.executor.get_drug_targets_for_gene(hostile)
        with patch.object(
            self.executor.chembl,
            "get_drug_profile",
            new=AsyncMock(return_value={"success": True, "drug": None, "indications": rows}),
        ):
            profile = await self.executor.get_drug_profile(hostile)
        with patch.object(
            self.executor.chembl,
            "get_target_bioactivity",
            new=AsyncMock(
                return_value={"success": True, "target_chembl_id": None, "_all_compounds": rows}
            ),
        ):
            bioactivity = await self.executor.get_target_bioactivity(hostile)

        names = [r["_download_data"]["filename"] for r in (drugs, profile, bioactivity)]
        assert names == [
            "chembl_chembl_drugs.tsv",
            "chembl_chembl_profile.tsv",
            "chembl_chembl_bioactivity.tsv",
        ]
        assert all(name.isascii() and '"' not in name for name in names)

    async def test_a_wide_drug_table_becomes_a_download_and_a_small_one_does_not(self):
        drugs = [{"pref_name": f"DRUG{i}"} for i in range(30)]
        with patch.object(
            self.executor.chembl,
            "get_drug_targets_for_gene",
            new=AsyncMock(
                return_value={"success": True, "target_chembl_id": "CHEMBL235", "drugs": drugs}
            ),
        ):
            wide = await self.executor.get_drug_targets_for_gene("PPARG")
        assert wide["_download_data"]["filename"] == "CHEMBL235_chembl_drugs.tsv"
        assert wide["_download_data"]["results"] is drugs

        with patch.object(
            self.executor.chembl,
            "get_drug_targets_for_gene",
            new=AsyncMock(return_value={"success": True, "drugs": drugs[:5]}),
        ):
            narrow = await self.executor.get_drug_targets_for_gene("PPARG")
        assert "_download_data" not in narrow

    async def test_the_bioactivity_download_carries_every_molecule_and_is_popped(self):
        summary = [{"molecule_chembl_id": f"CHEMBL{i}"} for i in range(30)]
        with patch.object(
            self.executor.chembl,
            "get_target_bioactivity",
            new=AsyncMock(
                return_value={
                    "success": True,
                    "target_chembl_id": "CHEMBL235",
                    "top_compounds": summary[:2],
                    "_all_compounds": summary,
                }
            ),
        ):
            result = await self.executor.get_target_bioactivity("PPARG")

        assert "_all_compounds" not in result
        assert result["_download_data"]["filename"] == "CHEMBL235_chembl_bioactivity.tsv"
        assert len(result["_download_data"]["results"]) == 30

    async def test_a_long_indication_list_becomes_a_download_named_by_the_molecule(self):
        indications = [{"efo_id": f"EFO:{i}"} for i in range(40)]
        with patch.object(
            self.executor.chembl,
            "get_drug_profile",
            new=AsyncMock(
                return_value={
                    "success": True,
                    "drug": {"molecule_chembl_id": "CHEMBL25"},
                    "indications": indications,
                }
            ),
        ):
            result = await self.executor.get_drug_profile("aspirin")

        assert result["_download_data"]["filename"] == "CHEMBL25_chembl_profile.tsv"
        assert len(result["_download_data"]["results"]) == 40
