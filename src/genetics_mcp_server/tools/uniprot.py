"""Client for the UniProt REST API and the EBI Proteins API.

Protein annotation logic lives here rather than in executor.py so it can be unit tested
without a ToolExecutor; the executor exposes thin delegating tool methods.
"""

import logging
import re
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import quote

import httpx

from genetics_mcp_server.config.settings import Settings

logger = logging.getLogger(__name__)

_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_MAX_REDIRECTS = 5
# query parameters a redirect target owns outright, which the original request's query
# must not overwrite when it is re-applied to the hop: UniProt answers a merged
# accession with `?from={requested}` and that is the only staleness signal there is
_SERVER_OWNED_PARAMS = frozenset({"from"})
_CACHE_MAXSIZE = 512

# UniProtKB accession syntax. Fully anchored and alphanumeric-only, which is also what
# makes it safe to interpolate a match into a request path: _build_url's quote() leaves
# dot segments alone, so an unvalidated identifier could climb out of /uniprotkb/.
_ACCESSION_RE = re.compile(
    r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$"
)
# identity only: resolve answers "which protein is this", and the full entry (TTN's runs
# to megabytes) is fetched by the annotation tools that actually need it
_RESOLVE_FIELDS = "accession,id,protein_name,gene_names,organism_name"
_MAX_ALTERNATIVES = 5


def _is_error(data: Any) -> bool:
    """True when `data` is the sentinel returned by UniProtClient._get on failure.

    Callers must use this instead of `"_error" in data`: endpoints differ in shape
    (/uniprotkb/{accession} returns an object, the EBI /proteins?accession= and
    /coordinates?accession= endpoints return an array), and on a list `in` silently
    degrades into an element-membership test while `data.get("_error")` raises. Error
    sentinels are always dicts.
    """
    return isinstance(data, dict) and "_error" in data


def _copy_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """Detached copy of a response meta dict, safe to hand to a caller.

    A cached entry is returned to every hit, so without this they would all share one
    mutable dict (and one mutable `headers` dict inside it) and a caller annotating its
    own copy would rewrite what the next hit sees. Everything else in meta is an
    immutable str/int/None, so a two-level copy is deep enough.
    """
    return {**meta, "headers": dict(meta["headers"])}


def _is_reviewed(entry: dict[str, Any]) -> bool:
    return str(entry.get("entryType") or "").startswith("UniProtKB reviewed")


def _gene_names(entry: dict[str, Any]) -> list[str]:
    """Gene symbols of an entry, approved name first, then what TrEMBL offers instead."""
    names: list[str] = []
    for gene in entry.get("genes") or []:
        primary = (gene.get("geneName") or {}).get("value")
        if primary:
            names.append(primary)
            continue
        # unreviewed entries often carry no approved symbol at all
        for key in ("synonyms", "orfNames", "orderedLocusNames"):
            values = [item.get("value") for item in gene.get(key) or [] if item.get("value")]
            if values:
                names.extend(values)
                break
    return list(dict.fromkeys(names))


def _protein_name(entry: dict[str, Any]) -> str | None:
    description = entry.get("proteinDescription") or {}
    recommended = ((description.get("recommendedName") or {}).get("fullName") or {}).get("value")
    if recommended:
        return recommended
    # TrEMBL entries have no curated recommended name, only the submitter's
    for key in ("submissionNames", "alternativeNames"):
        for item in description.get(key) or []:
            value = (item.get("fullName") or {}).get("value")
            if value:
                return value
    return None


def _entry_summary(entry: dict[str, Any]) -> dict[str, Any]:
    """Identity block for one entry, built fresh: response bodies are shared by the cache."""
    organism = entry.get("organism") or {}
    return {
        "accession": entry.get("primaryAccession"),
        "entry_name": entry.get("uniProtkbId"),
        "protein_name": _protein_name(entry),
        "gene_names": _gene_names(entry),
        "organism": organism.get("scientificName"),
        "taxon_id": organism.get("taxonId"),
        "reviewed": _is_reviewed(entry),
    }


def _inactive_result(accession: str, entry: dict[str, Any]) -> dict[str, Any]:
    """Resolution result for an entry UniProt answers 200 for but has withdrawn.

    DEMERGED and DELETED entries carry no gene names and no sequence, so flattening one
    would produce a plausible-looking empty protein.
    """
    reason = entry.get("inactiveReason") or {}
    reason_type = reason.get("inactiveReasonType") or "INACTIVE"
    replaced_by = list(reason.get("mergeDemergeTo") or [])
    advice = (
        f"resolve one of {', '.join(replaced_by)} instead"
        if replaced_by
        else "it has no replacement in UniProtKB"
    )
    return {
        "query": accession,
        "input_kind": "accession",
        **_entry_summary(entry),
        # deliberately not the requested accession: nothing downstream may go on to
        # annotate a dead identifier
        "accession": None,
        "match_basis": "accession",
        "ambiguous": True,
        "alternatives": [],
        "inactive": True,
        "inactive_reason": reason_type,
        "replaced_by": replaced_by,
        "warning": f"{accession} is not an active UniProtKB entry ({reason_type}); {advice}",
    }


def _quoted_term(value: str) -> str:
    """A search term quoted so hyphens, colons and spaces cannot act as query syntax."""
    return '"{}"'.format(value.replace("\\", "\\\\").replace('"', '\\"'))


def _reviewed_first(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Swiss-Prot ahead of TrEMBL, relevance order preserved within each group.

    Only bites with reviewed_only=False: gene:PRSS55 answers with the curated Q6UWB4 plus
    three TrEMBL fragments, and relevance order alone need not put the curated one first.
    """
    return sorted(results, key=lambda entry: not _is_reviewed(entry))


def _total_results(meta: dict[str, Any], fallback: int) -> int:
    raw = (meta.get("headers") or {}).get("x-total-results")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return fallback


class _TTLCache:
    """Time-limited store with FIFO eviction, keyed by fully-qualified request URL.

    Entries hold an absolute deadline rather than an insertion time so the ttl is a
    property of the write, not of the cache: the cache outlives any one UniProtClient
    while the ttl comes from that client's injected Settings.
    """

    _MISS = object()

    def __init__(
        self,
        maxsize: int = _CACHE_MAXSIZE,
        clock: Callable[[], float] = time.monotonic,
    ):
        # the clock is injected so expiry is unit testable without sleeping, and
        # defaults to monotonic so a wall-clock step cannot void or extend a ttl
        self._clock = clock
        self._maxsize = maxsize
        self._entries: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Any:
        """The cached value, or `_TTLCache._MISS` — a stored value may itself be falsy."""
        entry = self._entries.get(key)
        if entry is None:
            return self._MISS
        expires_at, value = entry
        if self._clock() >= expires_at:
            del self._entries[key]
            return self._MISS
        return value

    def set(self, key: str, value: Any, ttl: float) -> None:
        # a non-positive ttl (UNIPROT_CACHE_TTL=0) turns caching off rather than storing
        # entries that are already expired when they land
        if ttl <= 0:
            return
        # re-insert instead of overwriting in place: a refreshed entry gets a new
        # deadline, so it belongs at the back of the eviction queue as well
        self._entries.pop(key, None)
        self._entries[key] = (self._clock() + ttl, value)
        while len(self._entries) > self._maxsize:
            # dicts iterate in insertion order, so the first key is the oldest write
            del self._entries[next(iter(self._entries))]

    def clear(self) -> None:
        self._entries.clear()


# module level, not per UniProtClient: a client is built from a ToolExecutor, and
# ToolExecutor is instantiated independently by LLMService, by the standalone MCP server
# and by the analyze_variants CLI, so a per-instance cache would be per-executor.
# Subagents do share the main agent's executor today (LLMService hands its own to
# SubagentService), but nothing in that API requires it, and a per-instance cache would
# silently stop being shared the moment any caller builds its own executor.
_CACHE = _TTLCache()


class UniProtClient:
    """Async client for UniProtKB and the EBI Proteins API."""

    def __init__(self, client: httpx.AsyncClient, settings: Settings):
        # the httpx client is injected, not constructed here: the executor passes its
        # _ResilientAsyncClient (30s timeout, no auth headers), so connection failures
        # arrive as a synthetic 503 instead of raising, and tests can mock it
        self._client = client
        self._uniprot_url = settings.uniprot_api_url.rstrip("/")
        self._ebi_url = settings.ebi_proteins_api_url.rstrip("/")
        self._cache_ttl = settings.uniprot_cache_ttl
        self._ebi_host = httpx.URL(self._ebi_url).host
        # a redirect may only be followed to an origin this deployment already talks to.
        # Derived from the configured URLs rather than hardcoded so a self-hosted mirror
        # keeps working; with the default https configuration this refuses both
        # cross-host hops (link-local metadata endpoints, attacker-chosen hosts) and
        # https -> http downgrades, since the scheme is part of the origin.
        self._allowed_origins = frozenset(
            self._origin(httpx.URL(url)) for url in (self._uniprot_url, self._ebi_url)
        )

    async def _get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        base: str | None = None,
    ) -> Any:
        """GET a JSON document from UniProt (or, with `base`, another host).

        Returns the parsed body — an object or an array depending on the endpoint — or
        an error sentinel dict on a non-200 response or an unparseable body. Use
        `_is_error` to tell the two apart; the sentinel carries the HTTP status under
        `_status`, so callers can distinguish an expected 404 (no such accession) from a
        genuine upstream failure without matching on the message text.

        Only the connection failures that _ResilientAsyncClient converts into a
        synthetic 503 (ConnectError, ConnectTimeout) are guaranteed not to raise. Read
        and pool timeouts and protocol errors still propagate; the tool methods that own
        a try/except deal with those.

        `_get_with_meta` documents the `path`, `params` and `base` contract.
        """
        body, _meta = await self._get_with_meta(path, params=params, base=base)
        return body

    async def _get_with_meta(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        base: str | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """`_get` plus response metadata: status, headers, final URL and redirect origin.

        UniProt reports the result total in the x-total-results header and the
        pagination cursor in Link, neither of which appear in the body, so anything
        needing those goes through here. Every request funnels through this one method
        so the TTL cache can wrap it in a single place: successful responses are served
        from the process-wide `_CACHE` for `settings.uniprot_cache_ttl` seconds, keyed
        on the full request URL, and the returned body must not be mutated because
        later hits share it. Error sentinels are never cached.

        `path` is either a query-free path relative to the base host, with all query
        data passed via `params`, or an absolute http(s) URL used verbatim — the latter
        so a server-supplied cursor from the Link header can be replayed as-is. Both
        `base` and an absolute `path` choose the host that is contacted and neither is
        validated, so both must only ever receive a configured value or a URL this
        upstream itself handed back, never a tool argument. Redirect targets, by
        contrast, ARE validated: every hop is pinned to `self._allowed_origins`.

        On failure the sentinel may carry, besides `_status`, `_location` and the
        redirect-specific markers `_redirect_refused` (a hop that would have left the
        configured origins, or a Location that cannot be turned into a URL) and
        `_redirect_limit` (the hop budget ran out), so neither has to be told from an
        ordinary redirect response by matching on the message text. `_status` is None on
        a refusal raised before any response was returned.

        A Location so malformed that httpx cannot even parse it (a bad port, say) still
        raises RemoteProtocolError, exactly as httpx's own redirect following does; that
        is a protocol error like any other and the tool methods' try/except owns it.
        """
        url = self._build_url(path, params, base)
        # keyed on the fully-qualified URL, which _build_url has already folded `params`
        # into: keying on the bare path would collide across field projections of the
        # same accession and serve a body missing the fields the caller asked for
        cache_key = str(url)
        cached = _CACHE.get(cache_key)
        if cached is not _TTLCache._MISS:
            body, meta = cached
            return body, _copy_meta(meta)
        origin = url
        # the initial request's full query, which for an absolute `path` also covers the
        # query that URL already carried; re-applied to every hop so field projection
        # and pagination cursors survive a redirect
        request_query = origin.params
        redirected = False

        # redirects are followed here rather than via httpx's follow_redirects=True
        # because httpx rebuilds the hop from the Location URL alone and so DROPS
        # `params`, which would silently defeat field projection (large entries such as
        # TTN must not reach the model in full). Redirects are load bearing: a merged
        # accession (A6NKB1 -> 303 -> /uniprotkb/Q8WZ42?from=A6NKB1) and the legacy
        # /uniprot/{acc} path (301 -> /uniprotkb/{acc}) only answer after the hop.
        for hop in range(_MAX_REDIRECTS + 1):
            try:
                resp = await self._client.get(
                    url,
                    headers={"Accept": "application/json"},
                    follow_redirects=False,
                )
            except (httpx.InvalidURL, ValueError) as exc:
                # httpx parses Location and builds the redirect request even when
                # follow_redirects is False (it fills response.next_request), so a
                # Location it cannot turn into a URL — `javascript:...`, `mailto:...` —
                # raises out of THIS call, from the previous hop's response, never from
                # the join below. httpx behaves identically with follow_redirects=True,
                # so this is not a regression from hand-rolling the loop, but an
                # InvalidURL reaching a tool method as an opaque failure is not useful.
                return self._unusable_location(url, origin, exc, redirected)
            # a blank Location is as absent as a missing one; left as-is it resolves to
            # the current path and burns the whole hop budget on a nonsense URL
            location = (resp.headers.get("location") or "").strip()
            if resp.status_code not in _REDIRECT_STATUSES or not location:
                break
            if hop == _MAX_REDIRECTS:
                label = self._label(url)
                logger.warning(f"{label}: more than {_MAX_REDIRECTS} hops from {origin}")
                return {
                    "_error": f"{label} redirected more than {_MAX_REDIRECTS} times",
                    "_status": resp.status_code,
                    "_location": location,
                    "_redirect_limit": True,
                }, self._meta(resp, origin, redirected=True)
            try:
                target = url.join(location)
            except (httpx.InvalidURL, ValueError):
                # only reachable with a client that does not pre-parse Location, i.e. a
                # test double returning a hand-built httpx.Response
                return self._refused_redirect(
                    url, resp, location, "an unparseable location"
                ), self._meta(resp, origin, redirected=True)
            if self._origin(target) not in self._allowed_origins:
                return self._refused_redirect(
                    url, resp, location, "an origin outside the configured UniProt and EBI APIs"
                ), self._meta(resp, origin, redirected=True)
            # re-apply the original query, which httpx's own redirect following would
            # drop, then hand back the parameters the redirect target owns: without the
            # restore a caller-supplied `from` would clobber UniProt's own `?from=`,
            # because merging REPLACES same-key values rather than keeping both
            server_owned = {
                key: target.params[key] for key in _SERVER_OWNED_PARAMS if key in target.params
            }
            if request_query:
                target = target.copy_merge_params(request_query)
            for key, value in server_owned.items():
                target = target.copy_set_param(key, value)
            url = target
            redirected = True

        meta = self._meta(resp, origin, redirected)

        if resp.status_code != 200:
            return self._error_sentinel(resp, url), meta

        try:
            body = resp.json()
        except ValueError:
            # UniProt serves an HTML placeholder page when its REST tier is unavailable,
            # which would otherwise surface as an opaque JSON decode error
            label = self._label(url)
            logger.warning(f"{label} returned non-JSON body: {url}")
            return {
                "_error": f"{label} returned a non-JSON response",
                "_status": resp.status_code,
            }, meta

        # only successes are stored: a 404, a synthetic 503 from _ResilientAsyncClient, a
        # refused or exhausted redirect and an HTML placeholder page are all transient or
        # input-dependent, and pinning any of them for a day would outlast the condition
        # that caused it. Every one of those paths returns above, so the _is_error check
        # is a guard on the invariant rather than the mechanism enforcing it — it keeps
        # "no sentinel is ever cached" checkable here instead of by auditing five returns.
        # meta is cached alongside the body because a hit has no live response to rebuild
        # it from, and dropping it would make a cached call lose x-total-results, the Link
        # cursor and redirected_from, i.e. behave differently from an uncached one. The
        # body is handed out by reference (copying a full entry on every hit would defeat
        # the cache), so callers must treat it as read-only.
        if not _is_error(body):
            _CACHE.set(cache_key, (body, _copy_meta(meta)), self._cache_ttl)
        return body, meta

    async def resolve(
        self,
        query: str,
        organism_id: int | None = 9606,
        reviewed_only: bool = True,
    ) -> dict[str, Any]:
        """Work out which protein an agent-supplied gene symbol or accession refers to.

        Returns `{query, input_kind, accession, entry_name, protein_name, gene_names,
        organism, taxon_id, reviewed, match_basis, ambiguous, alternatives}`, or an error
        sentinel (test it with `_is_error`) when nothing matched or the request failed.

        An accession input is reported with the gene names of the entry it actually names,
        so an accession that is not the protein the caller meant is visible in the very
        same result — Q92626 supplied where TPO was intended comes back as PXDN_HUMAN
        rather than being annotated as if it were thyroid peroxidase. Withdrawn entries
        additionally carry `inactive`, `inactive_reason` and `replaced_by`, and an
        accession merged into another one carries `stale_accession`.

        A symbol is matched through three tiers: an exact gene-name match, then any gene
        name or synonym, then free text; it also reports `total_matches`. `ambiguous` is
        true unless the exact tier matched exactly one entry, which is the only outcome
        that pins a symbol to one protein.
        """
        text = (query or "").strip()
        if not text:
            return {
                "_error": "UniProt: empty query, nothing to resolve",
                "_status": None,
                "_no_match": True,
            }
        # matched before the value reaches _resolve_accession, which interpolates it into
        # a request path; a symbol never reaches a path at all
        candidate = text.upper()
        if _ACCESSION_RE.match(candidate):
            return await self._resolve_accession(candidate)
        return await self._resolve_symbol(text, organism_id, reviewed_only)

    async def _resolve_accession(self, accession: str) -> dict[str, Any]:
        body, meta = await self._get_with_meta(
            f"/uniprotkb/{accession}", params={"fields": _RESOLVE_FIELDS}
        )
        if _is_error(body):
            return body
        if body.get("entryType") == "Inactive":
            return _inactive_result(accession, body)

        result = {
            "query": accession,
            "input_kind": "accession",
            **_entry_summary(body),
            "match_basis": "accession",
            "ambiguous": False,
            "alternatives": [],
        }
        # a merged accession answers 303 and _get_with_meta follows it, so the entry in
        # hand is already the live one and `?from=` on the final URL is the only trace
        # that the caller's accession is stale. meta["redirected_from"] is not that
        # signal: the legacy /uniprot/ -> /uniprotkb/ 301 sets it on current accessions.
        merged_from = httpx.URL(meta.get("url") or "").params.get("from")
        if merged_from:
            result["stale_accession"] = merged_from
            result["warning"] = (
                f"{merged_from} is a secondary accession merged into {result['accession']}"
            )
        return result

    async def _resolve_symbol(
        self, symbol: str, organism_id: int | None, reviewed_only: bool
    ) -> dict[str, Any]:
        try:
            scope = f" AND organism_id:{int(organism_id)}" if organism_id else ""
        except (TypeError, ValueError):
            return {
                "_error": f"UniProt: organism_id must be a taxon id, got {organism_id!r}",
                "_status": None,
            }
        if reviewed_only:
            scope += " AND reviewed:true"
        term = _quoted_term(symbol)
        # most precise tier first: gene_exact is the only one that cannot match another
        # gene's synonym, and free text is the tier that ranks thrombopoietin (TPO_HUMAN,
        # gene THPO) second behind the real TPO
        tiers = (
            (f"gene_exact:{term}{scope}", "gene_exact_reviewed"),
            (f"gene:{term}{scope}", "gene_synonym_reviewed"),
            (f"{term}{scope}", "text_search"),
        )

        exact_total: int | None = None
        for tier_query, match_basis in tiers:
            body, meta = await self._get_with_meta(
                "/uniprotkb/search",
                params={
                    "query": tier_query,
                    "fields": _RESOLVE_FIELDS,
                    "size": _MAX_ALTERNATIVES + 1,
                },
            )
            if _is_error(body):
                return body
            hits = _reviewed_first(body.get("results") or [])
            total = _total_results(meta, len(hits))
            if exact_total is None:
                exact_total = total
            if not hits:
                continue
            return {
                "query": symbol,
                "input_kind": "symbol",
                **_entry_summary(hits[0]),
                "match_basis": match_basis,
                "ambiguous": exact_total != 1,
                # within the tier that answered, so a fallback tier's count does not read
                # as agreement about the symbol
                "total_matches": total,
                "alternatives": [_entry_summary(hit) for hit in hits[1:]],
            }

        return {
            "_error": (
                f"UniProt: no entry matched gene symbol '{symbol}'"
                f"{' in organism ' + str(organism_id) if organism_id else ''}"
                f"{' among reviewed entries' if reviewed_only else ''}"
            ),
            "_status": None,
            "_no_match": True,
        }

    @staticmethod
    def _meta(resp: httpx.Response, origin: httpx.URL, redirected: bool) -> dict[str, Any]:
        return {
            "status": resp.status_code,
            # only the headers carrying data absent from the body, as a plain
            # serialisable dict: httpx.Headers is neither JSON-serialisable nor safe to
            # hand out from the cache, where every hit would alias one mutable object
            "headers": {
                name: resp.headers[name]
                for name in ("x-total-results", "link")
                if name in resp.headers
            },
            "url": str(resp.url),
            # set whenever ANY hop happened, which includes the harmless legacy
            # /uniprot/ -> /uniprotkb/ 301 on a perfectly current accession. It is
            # therefore NOT evidence that the requested identifier is stale — the
            # `?from=` parameter on the final URL is that signal.
            "redirected_from": str(origin) if redirected else None,
        }

    def _build_url(self, path: str, params: dict[str, Any] | None, base: str | None) -> httpx.URL:
        """Absolute URL with `params` merged over any query the URL already carries.

        The query is folded into the URL instead of being passed to httpx as `params=`
        because httpx REPLACES an existing query in that case, which would discard the
        `?from=` of a redirect hop and the cursor of a Link URL.
        """
        if path.startswith(("http://", "https://")):
            url = httpx.URL(path)
        else:
            # quote() percent-encodes ?, = and &, so a relative path cannot carry a
            # query — accession-shaped path segments are all this needs to escape
            url = httpx.URL(f"{base or self._uniprot_url}/{quote(path.lstrip('/'), safe='/')}")
        return url.copy_merge_params(params) if params else url

    @staticmethod
    def _origin(url: httpx.URL) -> tuple[str, str, int | None]:
        """Scheme, host and port — what a redirect target is allowed to change to."""
        return url.scheme, url.host, url.port

    def _label(self, url: httpx.URL) -> str:
        """Which upstream an error came from, since `base` also serves the EBI host."""
        return "EBI Proteins API" if url.host == self._ebi_host else "UniProt"

    def _refused_redirect(
        self, url: httpx.URL, resp: httpx.Response, location: str, reason: str
    ) -> dict[str, Any]:
        label = self._label(url)
        logger.warning(f"{label}: refused redirect from {url} to {reason}: {location[:200]}")
        return {
            "_error": f"{label} redirected to {reason}: {location[:200]}",
            "_status": resp.status_code,
            "_location": location,
            "_redirect_refused": True,
        }

    def _unusable_location(
        self, url: httpx.URL, origin: httpx.URL, exc: Exception, redirected: bool
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Sentinel for a Location httpx rejected while building the next request.

        httpx closes and discards that response before raising, so unlike every other
        failure here there is no status or header to report and `_status` is None.
        """
        label = self._label(url)
        logger.warning(f"{label}: unusable redirect location from {url}: {exc}")
        return {
            "_error": f"{label} returned an unusable redirect location: {exc}",
            "_status": None,
            "_redirect_refused": True,
        }, {
            "status": None,
            "headers": {},
            "url": str(url),
            "redirected_from": str(origin) if redirected else None,
        }

    def _error_sentinel(self, resp: httpx.Response, url: httpx.URL) -> dict[str, Any]:
        label = self._label(url)
        # 404 (no such accession) and 400 (malformed accession or query) are the
        # documented answers to checking an agent-supplied identifier, i.e. input noise
        # rather than an incident
        log = logger.debug if resp.status_code in (400, 404) else logger.warning
        log(f"{label} request failed: {url} -> {resp.status_code}")

        # truncate body to keep error messages bounded
        body = (resp.text or "")[:200]
        sentinel: dict[str, Any] = {
            "_error": f"{label} HTTP {resp.status_code}: {body}",
            "_status": resp.status_code,
        }
        location = resp.headers.get("location")
        if location:
            sentinel["_location"] = location
        return sentinel
