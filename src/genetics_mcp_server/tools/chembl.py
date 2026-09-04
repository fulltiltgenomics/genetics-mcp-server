"""Client for the ChEMBL REST API (drug mechanisms, indications, bioactivity).

Transport only: URL building, the origin pin, the TTL cache, paging and error
sentinels. Target resolution and the tool methods live elsewhere, the way
tools/uniprot.py keeps its logic out of executor.py so it can be tested without a
ToolExecutor.
"""

import logging
import re
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

import httpx

from genetics_mcp_server.tools.uniprot import UniProtClient, _is_error, _TTLCache

if TYPE_CHECKING:
    # type-only for the same reason as in tools/uniprot.py: a real import of
    # config.settings would pull the module enumerating every internal env var name
    # into the SDK's import closure, and so into the sandbox image.
    from genetics_mcp_server.config.settings import Settings

logger = logging.getLogger(__name__)

_TIMEOUT = 20.0
# hard stop on a walk: a filter that matches half of ChEMBL (activity by target runs to
# thousands of rows) must bound the number of round trips, not just the rows returned
_MAX_PAGES = 10

# module level for the reason given above uniprot.py's `_CACHE`: clients are built per
# ToolExecutor and several are constructed independently, so a per-instance cache would
# not be shared.
_CACHE = _TTLCache()

_CHEMBL_TARGET_RE = re.compile(r"^CHEMBL[0-9]+$")

# a heteromer, a protein family and a cell line are all "targets" for the same accession,
# and only this one is the protein itself; the rest are reported rather than dropped
_SINGLE_PROTEIN = "SINGLE PROTEIN"
_TARGET_FIELDS = ["target_chembl_id", "pref_name", "target_type"]


class ChEMBLClient:
    """Async client for the ChEMBL REST API."""

    def __init__(
        self,
        external_client: httpx.AsyncClient,
        settings: "Settings",
        uniprot: UniProtClient,
    ):
        # the httpx client is injected, not constructed here: the executor passes its
        # _ResilientAsyncClient, so connection failures arrive as a synthetic 503
        # instead of raising, and no internal auth header reaches EBI
        self._client = external_client
        # the executor's own resolver, so a symbol resolved for a UniProt tool and the
        # same symbol resolved here cost one lookup between them and cannot disagree
        self._uniprot = uniprot
        self._base_url = settings.chembl_api_url.rstrip("/")
        self._cache_ttl = settings.chembl_cache_ttl
        # a request may only be made to the origin this deployment is configured for.
        # Derived from the base URL rather than hardcoded so a self-hosted mirror keeps
        # working; because the scheme is part of the origin this also refuses an
        # https -> http downgrade. The pin matters most for the paging walk, where the
        # next URL is server-supplied.
        self._allowed_origins = frozenset({self._origin(httpx.URL(self._base_url))})

    async def _get(
        self, resource: str, params: dict[str, Any], only: list[str]
    ) -> dict[str, Any]:
        """GET one page of `<base>/<resource>.json`, projected to `only`.

        Returns the parsed body or an error sentinel; use `_is_error` to tell them
        apart. `only` is mandatory because ChEMBL's unprojected records are far too
        large to reach the model or the cache: three target records are 55 KB, 200
        indications 76 KB, and a single molecule 6-8 KB.
        """
        return await self._get_url(self._build_url(resource, params, only))

    async def _get_all(
        self,
        resource: str,
        params: dict[str, Any],
        only: list[str],
        max_pages: int = _MAX_PAGES,
    ) -> dict[str, Any]:
        """Follow `page_meta.next` up to `max_pages` and concatenate the record lists.

        Returns `{"records", "total_count", "truncated"}` — `total_count` is ChEMBL's
        own count for the whole filter, so a caller can say how much was left behind —
        or the sentinel of the first page that failed. The records are the cached page
        objects themselves, so callers must not mutate what comes back.
        """
        records: list[Any] = []
        total_count: int | None = None
        body = await self._get(resource, params, only)
        pages = 0
        while True:
            if _is_error(body):
                return body
            page = self._records(body)
            if page is None:
                return self._shape_sentinel(resource, body)
            records.extend(page)
            pages += 1
            page_meta = body.get("page_meta") or {}
            if total_count is None:
                count = page_meta.get("total_count")
                total_count = count if isinstance(count, int) else None
            next_path = page_meta.get("next")
            if not next_path:
                return {"records": records, "total_count": total_count, "truncated": False}
            if pages >= max_pages:
                return {"records": records, "total_count": total_count, "truncated": True}
            # `next` is a path relative to the host ("/chembl/api/data/target.json?…"),
            # so it is joined and re-pinned like any other server-supplied URL
            if not isinstance(next_path, str):
                return self._unusable_next(next_path)
            try:
                next_url = httpx.URL(self._base_url).join(next_path)
            except (httpx.InvalidURL, ValueError):
                # a `next` httpx cannot parse — a non-numeric port, a host IDNA cannot
                # encode — must leave the walk as a sentinel like any other bad page,
                # not as an exception out of a tool method
                return self._unusable_next(next_path)
            body = await self._get_url(next_url)

    async def release(self) -> str | None:
        """The live release name, e.g. "ChEMBL_37", or None if status is unreadable.

        `status.json` is a single small record and takes no `only`, so it is the one
        request that does not go through `_get`.
        """
        body = await self._get_url(httpx.URL(f"{self._base_url}/status.json"))
        if _is_error(body):
            return None
        version = body.get("chembl_db_version")
        return str(version) if version else None

    async def resolve_target(self, query: str) -> dict[str, Any]:
        """Work out which ChEMBL target an agent-supplied gene, accession or id means.

        A `CHEMBL<number>` target id is used as given; everything else goes through the
        shared UniProt resolver, which reports whether it read the input as an accession
        or a symbol. Shape alone cannot make that call here — real HGNC symbols (P2RY12,
        B4GAT1, H2AC11) are valid accession syntax — and the resolver also maps a merged
        or secondary accession to the primary one. Using it rather than ChEMBL's own
        symbol synonyms, a second and differently curated naming authority, keeps one
        answer to "which protein is this".

        Returns `{resolution: {query, kind, accession, uniprot, organism, n_targets, note},
        target_chembl_id, pref_name, target_type, other_targets}`. A gene with no ChEMBL
        target is a normal answer, not an error: `target_chembl_id` is None and the note
        says so. Only a failed lookup is a sentinel (test it with `_is_error`); the
        sentinel carries `_stage` because "UniProt could not resolve the symbol" and
        "ChEMBL refused the target query" are different problems for the caller.
        """
        text = (query or "").strip()
        if not text:
            return {
                "_error": "ChEMBL: empty query, nothing to resolve",
                "_status": None,
                "_stage": "input",
            }
        candidate = text.upper()
        if _CHEMBL_TARGET_RE.match(candidate):
            return await self._targets(
                {"target_chembl_id": candidate}, text, "chembl_target", None, None
            )
        resolution = await self._uniprot.resolve(text)
        if _is_error(resolution) or not resolution.get("accession"):
            return self._uniprot_stage_sentinel(text, resolution)
        return await self._targets_for_accession(
            str(resolution["accession"]),
            text,
            str(resolution.get("input_kind") or "symbol"),
            resolution,
        )

    async def _targets_for_accession(
        self, accession: str, query: str, kind: str, uniprot: dict[str, Any] | None
    ) -> dict[str, Any]:
        return await self._targets(
            # organism is filtered here rather than after the fact because the same
            # accession's orthologues carry their own targets
            {"target_components__accession": accession, "organism": "Homo sapiens"},
            query,
            kind,
            accession,
            uniprot,
        )

    async def _targets(
        self,
        params: dict[str, Any],
        query: str,
        kind: str,
        accession: str | None,
        uniprot: dict[str, Any] | None,
    ) -> dict[str, Any]:
        body = await self._get_all("target", params, _TARGET_FIELDS)
        if _is_error(body):
            return {**body, "_stage": "chembl_target"}
        organism = params.get("organism")
        records = body["records"]
        singles = [r for r in records if r.get("target_type") == _SINGLE_PROTEIN]
        chosen = singles[0] if singles else (records[0] if records else None)
        others = [r for r in records if r is not chosen]
        if chosen is None:
            # the organism filter is silent, so a bare "no target" would state a false
            # reason for an accession whose targets are all non-human
            note = f"no ChEMBL target is annotated for {accession or query}"
            if organism:
                note += f" in {organism}"
        elif singles:
            note = f"chose the {_SINGLE_PROTEIN} target"
            if others:
                note += f"; {len(others)} other target(s) share this accession"
        else:
            note = (
                f"no {_SINGLE_PROTEIN} target for {accession or query}; "
                f"returning the {chosen.get('target_type')} target instead"
            )
        if body.get("truncated"):
            total = body.get("total_count")
            matched = f"{total}" if isinstance(total, int) else f"more than {len(records)}"
            note += f"; {matched} targets matched and the walk was capped"
        return {
            "resolution": {
                "query": query,
                "kind": kind,
                "accession": accession,
                "uniprot": uniprot,
                "organism": organism,
                "n_targets": len(records),
                "note": note,
            },
            # rebuilt rather than passed through: the records are the cached page objects
            "target_chembl_id": chosen.get("target_chembl_id") if chosen else None,
            "pref_name": chosen.get("pref_name") if chosen else None,
            "target_type": chosen.get("target_type") if chosen else None,
            "other_targets": [{k: r.get(k) for k in _TARGET_FIELDS} for r in others],
        }

    @staticmethod
    def _uniprot_stage_sentinel(query: str, resolution: dict[str, Any]) -> dict[str, Any]:
        """Sentinel for a symbol UniProt could not turn into an accession."""
        reason = resolution.get("_error") or "UniProt returned no accession"
        sentinel = {
            "_error": f"ChEMBL: could not resolve {query!r} to a UniProt accession: {reason}",
            "_status": resolution.get("_status"),
            "_stage": "uniprot",
        }
        if resolution.get("_no_match"):
            sentinel["_no_match"] = True
        return sentinel

    async def _get_url(self, url: httpx.URL) -> dict[str, Any]:
        """The single request funnel: origin pin, TTL cache, 20s timeout, sentinel.

        A cached body is handed to every hit, so callers must not mutate what comes
        back. Error sentinels are never cached.
        """
        if self._origin(url) not in self._allowed_origins:
            return self._refused_origin(url)
        cache_key = str(url)
        cached = _CACHE.get(cache_key)
        if cached is not _TTLCache._MISS:
            return cached
        resp = await self._client.get(
            url,
            headers={"Accept": "application/json"},
            timeout=_TIMEOUT,
            # stated rather than inherited from the client's default: the origin pin above
            # only holds if the hop it approves is the only one made
            follow_redirects=False,
        )
        if resp.status_code != 200:
            return self._error_sentinel(resp, url)
        try:
            body = resp.json()
        except ValueError as exc:
            logger.warning(f"ChEMBL returned an unparseable body from {url}: {exc}")
            return {"_error": f"ChEMBL returned an unparseable response: {exc}", "_status": 200}
        if not isinstance(body, dict):
            return self._shape_sentinel(str(url), body)
        _CACHE.set(cache_key, body, self._cache_ttl)
        return body

    def _build_url(self, resource: str, params: dict[str, Any], only: list[str]) -> httpx.URL:
        if not only:
            raise ValueError("ChEMBL requests must project with `only`")
        # quote() percent-encodes ?, = and & so a resource name cannot smuggle in a
        # query, and leaves no way out of the configured base path
        url = httpx.URL(f"{self._base_url}/{quote(resource.strip('/'), safe='')}.json")
        return url.copy_merge_params({**params, "only": ",".join(only)})

    @staticmethod
    def _records(body: dict[str, Any]) -> list[Any] | None:
        """The record list of a ChEMBL page, or None if the body is not one.

        The key is the resource name pluralised (targets, mechanisms, molecules,
        activities, drug_indications, assays), which is found by elimination rather
        than by a table of plurals so a resource nobody anticipated still reads.
        """
        lists = [
            value
            for key, value in body.items()
            if key != "page_meta" and isinstance(value, list)
        ]
        return lists[0] if len(lists) == 1 else None

    @staticmethod
    def _origin(url: httpx.URL) -> tuple[str, str, int | None]:
        return url.scheme, url.host, url.port

    @staticmethod
    def _refused_origin(url: httpx.URL) -> dict[str, Any]:
        """Sentinel for a URL outside the configured origin; no request is made."""
        logger.warning(f"ChEMBL: refused a request to an origin outside the configured API: {url}")
        return {
            "_error": "ChEMBL refused a request to an origin outside the configured API",
            "_status": None,
            "_origin_refused": True,
        }

    @staticmethod
    def _unusable_next(next_path: Any) -> dict[str, Any]:
        """Sentinel for a `page_meta.next` that cannot become a URL; no request is made."""
        logger.warning(f"ChEMBL: refused an unusable page_meta.next: {next_path!r}")
        return {
            "_error": "ChEMBL returned an unusable next-page link",
            "_status": None,
            "_unusable_next": True,
        }

    @staticmethod
    def _shape_sentinel(what: str, body: Any) -> dict[str, Any]:
        logger.warning(f"ChEMBL returned an unexpected shape for {what}: {type(body).__name__}")
        return {
            "_error": f"ChEMBL returned an unexpected response shape for {what}",
            "_status": 200,
            "_unexpected_shape": True,
        }

    @staticmethod
    def _error_sentinel(resp: httpx.Response, url: httpx.URL) -> dict[str, Any]:
        # 404 (no such record) and 400 (malformed filter or identifier) are the
        # documented answers to checking an agent-supplied identifier, i.e. input noise
        # rather than an incident
        log = logger.debug if resp.status_code in (400, 404) else logger.warning
        log(f"ChEMBL request failed: {url} -> {resp.status_code}")
        return {
            "_error": f"ChEMBL HTTP {resp.status_code}: {(resp.text or '')[:200]}",
            "_status": resp.status_code,
        }
