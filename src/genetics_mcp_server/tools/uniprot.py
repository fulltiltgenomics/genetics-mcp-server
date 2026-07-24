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
# Lucene wildcards, which survive _quoted_term because UniProt's parser honours them
# inside a quoted phrase and does NOT honour a backslash escape (verified live:
# gene:"*" and gene:"\*" both return every reviewed human entry). A gene symbol or
# accession never contains either character, so removing them is lossless for real input.
_WILDCARD_RE = re.compile(r"[*?]")
# identity only: resolve answers "which protein is this", and the full entry (TTN's runs
# to megabytes) is fetched by the annotation tools that actually need it
_RESOLVE_FIELDS = "accession,id,protein_name,gene_names,organism_name"
_MAX_ALTERNATIVES = 5

# GRCh38 RefSeq accessions per chromosome — the versioned NC_ identifiers the EBI
# variation/hgvs endpoint keys genomic HGVS on. Build-specific (the version suffix is
# GRCh38.p14) and pinned here so a chr:pos:ref:alt is only ever mapped against GRCh38;
# a bare '12' has no assembly and guessing one would silently answer for the wrong build.
_GRCH38_REFSEQ: dict[str, str] = {
    "1": "NC_000001.11", "2": "NC_000002.12", "3": "NC_000003.12", "4": "NC_000004.12",
    "5": "NC_000005.10", "6": "NC_000006.12", "7": "NC_000007.14", "8": "NC_000008.11",
    "9": "NC_000009.12", "10": "NC_000010.11", "11": "NC_000011.10", "12": "NC_000012.12",
    "13": "NC_000013.11", "14": "NC_000014.9", "15": "NC_000015.10", "16": "NC_000016.10",
    "17": "NC_000017.11", "18": "NC_000018.10", "19": "NC_000019.10", "20": "NC_000020.11",
    "21": "NC_000021.9", "22": "NC_000022.11", "X": "NC_000023.11", "Y": "NC_000024.10",
    "MT": "NC_012920.1",
}
# chr:pos:ref:alt, the variant id every other tool here already speaks. A leading 'chr'
# and 'M'/'MT'/'23'/'24' aliases are normalised in _genomic_hgvs, not matched here.
_GENOMIC_VARIANT_RE = re.compile(
    r"^(?P<chr>[0-9A-Za-z]+):(?P<pos>\d+):(?P<ref>[ACGTN]+):(?P<alt>[ACGTN]+)$",
    re.IGNORECASE,
)


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


def _inactive_result(accession: str, entry: dict[str, Any], reviewed_only: bool) -> dict[str, Any]:
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


def _organism_taxon(organism_id: Any) -> int | None:
    """`organism_id` as a positive taxon id, or None for "any organism".

    Raises TypeError/ValueError on anything else. The explicit positivity check is what
    makes organism_id=-1 fail the same way organism_id='abc' does: int() accepts it
    happily and UniProt answers the resulting query with a raw HTTP 400.
    """
    if organism_id is None:
        return None
    taxon = int(organism_id)
    if taxon <= 0:
        raise ValueError(f"taxon id must be positive, got {taxon}")
    return taxon


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
        self._uniprot_host = httpx.URL(self._uniprot_url).host
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
        `base` and an absolute `path` choose the host that is contacted, and both are
        pinned to `self._allowed_origins`, the same check every redirect hop passes: the
        guarantee is structural, not a trusted-callers-only convention, so a server-
        supplied Link cursor — the same trust class as a Location header — can never
        reach a host the deployment does not already talk to.

        On failure the sentinel may carry, besides `_status`, `_location` and the
        origin markers `_origin_refused` (the initial URL was outside the configured
        origins), `_redirect_refused` (a hop that would have left them, or a Location that
        cannot be turned into a URL) and `_redirect_limit` (the hop budget ran out), so
        none has to be told from an ordinary redirect response by matching on the message
        text. `_status` is None on a refusal raised before any response was returned.

        A Location so malformed that httpx cannot even parse it (a bad port, say) still
        raises RemoteProtocolError, exactly as httpx's own redirect following does; that
        is a protocol error like any other and the tool methods' try/except owns it.
        """
        url = self._build_url(path, params, base)
        # pin the URL the request STARTS at, not just the redirect hops: `base` and an
        # absolute `path` (a replayed Link cursor) choose the host without passing through
        # the hop loop's origin check, and must be held to the same allowed origins
        if self._origin(url) not in self._allowed_origins:
            return self._refused_origin(url)
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
            except httpx.InvalidURL as exc:
                # httpx parses Location and builds the redirect request even when
                # follow_redirects is False (it fills response.next_request), so a
                # Location it cannot turn into a URL — `javascript:...`, `mailto:...` —
                # raises out of THIS call, from this call's own response before it is
                # returned, never from the join below. httpx behaves identically with
                # follow_redirects=True, so this is not a regression from hand-rolling the
                # loop, but an InvalidURL reaching a tool method as an opaque failure is
                # not useful. Only InvalidURL is caught: it derives from Exception, not
                # ValueError, so a wider clause would only risk relabelling an unrelated
                # transport ValueError as an unusable redirect location.
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
                }, self._meta(resp, origin, redirected)
            try:
                target = url.join(location)
            except (httpx.InvalidURL, ValueError):
                # only reachable with a client that does not pre-parse Location, i.e. a
                # test double returning a hand-built httpx.Response
                return self._refused_redirect(
                    url, resp, location, "an unparseable location"
                ), self._meta(resp, origin, redirected)
            if self._origin(target) not in self._allowed_origins:
                return self._refused_redirect(
                    url, resp, location, "an origin outside the configured UniProt and EBI APIs"
                ), self._meta(resp, origin, redirected)
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

        Both result shapes share `{query, input_kind, accession, entry_name, protein_name,
        gene_names, organism, taxon_id, reviewed, match_basis, ambiguous, alternatives,
        reviewed_only}`; on failure an error sentinel is returned instead (test it with
        `_is_error`), carrying `_no_match` whenever the input simply matched nothing.
        `reviewed_only` echoes the argument, because the `_reviewed` suffix on a
        match_basis names the tier that answered and not whether a reviewed:true clause
        was actually sent.

        input_kind 'accession' adds `stale_accession` when the caller's accession has been
        merged into another, and `inactive`/`inactive_reason`/`replaced_by` when UniProt
        has withdrawn it. input_kind 'symbol' adds `total_matches` and, when an
        accession-shaped input had to be reinterpreted, `accession_interpretation`.

        An accession input is reported with the gene names of the entry it actually names,
        so an accession that is not the protein the caller meant is visible in the very
        same result — Q92626 supplied where TPO was intended comes back as PXDN_HUMAN
        rather than being annotated as if it were thyroid peroxidase.

        A symbol is matched through three tiers: an exact gene-name match, then any gene
        name or synonym, then free text. `ambiguous` is true unless the exact tier matched
        exactly one entry, which is the only outcome that pins a symbol to one protein.

        Accession syntax and gene symbols overlap — P2RY12, B4GAT1, B3GNT2, R3HDM1 and the
        whole H2AC*/H2BC* histone families are all valid accession patterns — so an
        accession reading that does not hold up is retried as a symbol; see
        `_resolve_accession`. Isoform identifiers (P07202-2) are deliberately NOT accession
        shaped and fall through to the symbol ladder, where free text still finds the
        parent entry. Lucene wildcards are stripped from `query` before anything else: they
        survive quoting, and `*` alone would otherwise "resolve" to all 20k human entries.
        """
        text = _WILDCARD_RE.sub("", query or "").strip()
        if not text:
            return {
                "_error": "UniProt: empty query, nothing to resolve",
                "_status": None,
                "_no_match": True,
            }
        try:
            taxon_id = _organism_taxon(organism_id)
        except (TypeError, ValueError):
            return {
                "_error": f"UniProt: organism_id must be a positive taxon id, got {organism_id!r}",
                "_status": None,
            }
        # matched before the value reaches _resolve_accession, which interpolates it into
        # a request path; a symbol never reaches a path at all
        candidate = text.upper()
        if _ACCESSION_RE.match(candidate):
            return await self._resolve_accession(candidate, taxon_id, reviewed_only)
        return await self._resolve_symbol(text, taxon_id, reviewed_only)

    async def _resolve_accession(
        self, accession: str, organism_id: int | None, reviewed_only: bool
    ) -> dict[str, Any]:
        """Resolve an accession-shaped input, retrying it as a gene symbol if it fails.

        The regex is right — it admits every valid accession and nothing malformed — but
        real HGNC symbols land inside that syntax, so matching it is not proof the caller
        meant an accession. Three signals say the accession reading is wrong, and each
        costs nothing extra to observe because the entry fetch already happened: UniProt
        404s, the entry is withdrawn, or its organism is not the one that was asked for.
        Only those three trigger the retry, so an ordinary accession lookup stays at one
        request; a colliding symbol costs a second one, which the TTL cache then absorbs.

        Free text is excluded from the retry: it matches on description prose, which is no
        evidence that an accession-shaped token is a gene symbol, and accepting it would
        let an unrelated entry displace a correct accession answer.
        """
        body, meta = await self._get_with_meta(
            f"/uniprotkb/{accession}", params={"fields": _RESOLVE_FIELDS}
        )
        if _is_error(body):
            if body.get("_status") != 404:
                return body
            return await self._as_symbol(
                accession,
                organism_id,
                reviewed_only,
                outcome={"outcome": "not_found"},
                rejected=f"{accession} is not a UniProtKB accession",
                on_failure={
                    "_error": (
                        f"UniProt: '{accession}' matched no UniProtKB entry — it is neither "
                        f"a known accession nor a gene symbol"
                        f"{' in organism ' + str(organism_id) if organism_id else ''}"
                        f"{' among reviewed entries' if reviewed_only else ''}"
                    ),
                    "_status": 404,
                    "_no_match": True,
                },
            )

        if body.get("entryType") == "Inactive":
            inactive = _inactive_result(accession, body)
            return await self._as_symbol(
                accession,
                organism_id,
                reviewed_only,
                outcome={
                    "outcome": "inactive",
                    "inactive_reason": inactive["inactive_reason"],
                    "replaced_by": list(inactive["replaced_by"]),
                },
                rejected=inactive["warning"],
                on_failure=inactive,
            )

        result = {
            "query": accession,
            "input_kind": "accession",
            **_entry_summary(body),
            "match_basis": "accession",
            "ambiguous": False,
            "alternatives": [],
            "reviewed_only": reviewed_only,
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

        if organism_id is not None and result["taxon_id"] != organism_id:
            # B4GAT1 is both a human gene and a Drosophila persimilis accession; without
            # this the fruit fly entry came back as an unambiguous answer to a human query
            mismatch = (
                f"accession {accession} is {result['organism']} (taxon {result['taxon_id']}), "
                f"not the requested organism {organism_id}"
            )
            result["taxon_mismatch"] = True
            result["ambiguous"] = True
            result["warning"] = f"{result['warning']}; {mismatch}" if merged_from else mismatch
            return await self._as_symbol(
                accession,
                organism_id,
                reviewed_only,
                outcome={
                    "outcome": "organism_mismatch",
                    "accession": result["accession"],
                    "entry_name": result["entry_name"],
                    "organism": result["organism"],
                    "taxon_id": result["taxon_id"],
                },
                rejected=mismatch,
                on_failure=result,
            )
        return result

    async def _as_symbol(
        self,
        accession: str,
        organism_id: int | None,
        reviewed_only: bool,
        outcome: dict[str, Any],
        rejected: str,
        on_failure: dict[str, Any],
    ) -> dict[str, Any]:
        """Retry a failed accession reading as a gene symbol, keeping both interpretations.

        Returns `on_failure` unchanged when the symbol ladder finds nothing either, so a
        genuinely dead or genuinely foreign accession still reports as itself.
        """
        result = await self._resolve_symbol(
            accession, organism_id, reviewed_only, allow_text=False
        )
        if _is_error(result):
            return on_failure
        # the discarded reading is preserved rather than dropped: the caller may well have
        # meant the accession, and this is the only place that says what it would have been
        result["accession_interpretation"] = {"queried_as": accession, **outcome}
        result["warning"] = f"{rejected}; resolved as a gene symbol instead"
        return result

    async def _resolve_symbol(
        self,
        symbol: str,
        organism_id: int | None,
        reviewed_only: bool,
        allow_text: bool = True,
    ) -> dict[str, Any]:
        scope = f" AND organism_id:{organism_id}" if organism_id is not None else ""
        if reviewed_only:
            scope += " AND reviewed:true"
        term = _quoted_term(symbol)
        # most precise tier first: gene_exact is the only one that cannot match another
        # gene's synonym, and free text is the tier that ranks thrombopoietin (TPO_HUMAN,
        # gene THPO) second behind the real TPO
        tiers = [
            (f"gene_exact:{term}{scope}", "gene_exact_reviewed"),
            (f"gene:{term}{scope}", "gene_synonym_reviewed"),
        ]
        if allow_text:
            tiers.append((f"{term}{scope}", "text_search"))

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
                # judged on the exact tier alone, so a fallback tier answering does not
                # read as agreement about the symbol
                "ambiguous": exact_total != 1,
                # counted within the tier that answered
                "total_matches": total,
                "alternatives": [_entry_summary(hit) for hit in hits[1:]],
                "reviewed_only": reviewed_only,
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
        """Which upstream an error came from, since `base` also serves the EBI host.

        UniProt is the default so a self-hosted mirror pointing both settings URLs at one
        host does not label every message 'EBI Proteins API'; only a host that is the EBI
        host and not also the UniProt host is named as EBI.
        """
        if url.host == self._ebi_host and self._ebi_host != self._uniprot_host:
            return "EBI Proteins API"
        return "UniProt"

    def _refused_origin(self, url: httpx.URL) -> tuple[dict[str, Any], dict[str, Any]]:
        """Sentinel for an initial request URL outside the configured origins.

        The redirect loop pins every hop; this pins the URL the request STARTS at, so
        that neither `base`, an absolute `path`, nor a server-supplied Link cursor
        replayed as one can reach a host the deployment does not already talk to. No
        request is made, so there is no status or header to report.
        """
        label = self._label(url)
        logger.warning(f"{label}: refused a request to an origin outside the configured APIs: {url}")
        return {
            "_error": (
                f"{label} refused a request to an origin outside the configured "
                f"UniProt and EBI APIs"
            ),
            "_status": None,
            "_origin_refused": True,
        }, {
            "status": None,
            "headers": {},
            "url": str(url),
            "redirected_from": None,
        }

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

    async def fetch_batch(
        self,
        accessions: list[str] | str,
        fields: str | None = None,
    ) -> dict[str, Any]:
        """Fetch many entries in one round trip, in chunks of 100 (the endpoint's limit).

        Serves table-building over a gene list — a 167-gene zymogen feature table is
        currently curated by hand — where one request per accession is both slow and, via
        the per-entry error paths, easy to lose rows in silently.

        `missing` is the point of the result shape: an accession UniProt did not answer
        for is reported by name rather than dropped, so a table can never quietly come
        back short. `invalid` holds inputs that are not accession-shaped at all (a gene
        symbol among the accessions), which must go through `resolve` first.
        """
        requested = [
            token
            for token in dict.fromkeys(
                (item or "").strip().upper() for item in _as_list(accessions)
            )
            if token
        ]
        valid = [item for item in requested if _ACCESSION_RE.match(item)]
        invalid = [item for item in requested if not _ACCESSION_RE.match(item)]

        results: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        for start in range(0, len(valid), _BATCH_SIZE):
            chunk = valid[start : start + _BATCH_SIZE]
            params: dict[str, Any] = {"accessions": ",".join(chunk)}
            if fields:
                params["fields"] = fields
            body = await self._get("/uniprotkb/accessions", params=params)
            if _is_error(body):
                errors.append({**body, "accessions": chunk})
                continue
            results.extend(body.get("results") or [])

        found: set[str] = set()
        for entry in results:
            primary = entry.get("primaryAccession")
            if primary:
                found.add(primary)
            # a requested secondary accession is answered under its primary, so without
            # this it would be reported missing while its entry is right there
            found.update(entry.get("secondaryAccessions") or [])

        return {
            "requested": requested,
            "invalid": invalid,
            "results": results,
            "count": len(results),
            "missing": [item for item in valid if item not in found],
            "errors": errors,
        }

    async def get_protein_annotations(
        self,
        query: str,
        organism_id: int | None = None,
        include: list[str] | str | None = None,
        feature_types: list[str] | str | None = None,
        residue_range: Any = None,
    ) -> dict[str, Any]:
        """Flattened annotations for whichever protein `query` actually resolves to.

        The result always carries a `resolution` block — the full `resolve` answer,
        including its warnings, alternatives and ambiguity flag — so protein data can
        never reach a caller without saying which protein it is. Nothing is annotated
        when resolution produced no live accession (a withdrawn entry), because a
        plausible-looking empty protein is worse than an error.

        `include` selects the sections too large to return unasked: 'features' (per
        residue, TTN alone has thousands), 'sequence' and 'isoforms'. Passing
        `feature_types` or `residue_range` implies 'features', since asking to filter
        features is asking for them.

        A list `query` is a table request (the 167-gene zymogen case) and answers with a
        flat `results` row per input, each row carrying its own identity and match_basis
        so a table can never attribute a row to the wrong protein.
        """
        sections = {item.strip().lower() for item in _as_list(include) if item}
        wants_features = bool(feature_types) or residue_range is not None or "features" in sections
        fields = _ENTRY_FIELDS
        if wants_features:
            fields = f"{fields},{_feature_fields(feature_types)}"

        if isinstance(query, (list, tuple, set, frozenset)):
            return await self._annotate_batch(
                list(query), organism_id, fields, wants_features, feature_types, residue_range
            )

        resolution = await self.resolve(query, organism_id=organism_id)
        if _is_error(resolution):
            return {**resolution, "query": query}
        accession = resolution.get("accession")
        if not accession:
            return {
                "_error": (
                    f"UniProt: '{query}' resolved to no annotatable entry"
                    f"{': ' + resolution['warning'] if resolution.get('warning') else ''}"
                ),
                "_status": None,
                "resolution": resolution,
            }

        body = await self._get(f"/uniprotkb/{accession}", params={"fields": fields})
        if _is_error(body):
            return {**body, "resolution": resolution}

        result: dict[str, Any] = {
            "success": True,
            "resolution": resolution,
            "accession": accession,
            "entry": flatten_entry(body),
        }
        if wants_features:
            result["features"] = flatten_features(
                body.get("features"), feature_types=feature_types, residue_range=residue_range
            )
            result["feature_filter"] = {
                "feature_types": _as_list(feature_types) or None,
                "residue_range": list(_residue_range(residue_range) or ()) or None,
            }
        if "sequence" in sections:
            result["sequence"] = (body.get("sequence") or {}).get("value")
        if "isoforms" in sections:
            result["isoforms"] = _isoforms(body)
        return result

    async def map_protein_variants(
        self,
        variants: list[str] | str,
        query: str,
        organism_id: int | None = None,
    ) -> dict[str, Any]:
        """Map protein-level variants onto genomic coordinates, one resolution for all.

        Each variant string carries its own expected residue ('P70A', 'p.Pro70Ala'), so
        the residue agreement check is part of the input and cannot be skipped by an
        omitted argument. A bare position ('70') is accepted and simply reports no
        expectation to check against.

        Coordinates are never suppressed on disagreement: the mapping is returned with
        `agrees: false` plus the evidence that explains it — sequence length, isoform
        count and any sequence-conflict or VAR_SEQ feature over that residue. P07202
        position 70 is exactly that case (Pro at chr2:1,433,466-1,433,468 with a
        sequence conflict at the same residue).
        """
        resolution = await self.resolve(query, organism_id=organism_id)
        if _is_error(resolution):
            return {**resolution, "query": query}
        accession = resolution.get("accession")
        if not accession:
            return {
                "_error": (
                    f"UniProt: '{query}' resolved to no mappable entry"
                    f"{': ' + resolution['warning'] if resolution.get('warning') else ''}"
                ),
                "_status": None,
                "resolution": resolution,
            }

        entry = await self._get(
            f"/uniprotkb/{accession}",
            params={
                "fields": f"accession,sequence,cc_alternative_products,"
                f"{_CONFLICT_FIELDS},ft_variant"
            },
        )
        entry = {} if _is_error(entry) else entry
        curated = flatten_features(entry.get("features"), feature_types="variant")

        mapped: list[dict[str, Any]] = []
        for raw in _as_list(variants):
            variant: dict[str, Any] = {"variant": raw}
            try:
                position, expected_aa = parse_protein_variant(raw)
            except ValueError as exc:
                variant["_error"] = str(exc)
                mapped.append(variant)
                continue
            variant["position"] = position
            variant["expected_aa"] = expected_aa
            body = await self._get_coordinates(accession, position)
            if _is_error(body):
                variant["_error"] = body["_error"]
                variant["_status"] = body.get("_status")
                mapped.append(variant)
                continue
            locations = body.get("locations") if isinstance(body, dict) else body
            rows = collapse_transcripts(locations, expected_aa=expected_aa)
            variant["locations"] = rows
            variant["agrees"] = (
                None
                if expected_aa is None or not rows
                else all(row.get("agrees") for row in rows)
            )
            if variant["agrees"] is False:
                variant["disagreement_evidence"] = _disagreement_evidence(entry, position)
            matches = [c for c in curated if c.get("start") == position]
            if matches:
                variant["curated_variants"] = matches
            mapped.append(variant)

        return {
            "success": True,
            "resolution": resolution,
            "accession": accession,
            "variants": mapped,
        }

    async def get_variant_protein_effect(
        self, variants: list[str] | str
    ) -> dict[str, Any]:
        """Map genomic coding SNVs onto their curated UniProt protein consequence.

        The direction map_protein_variants does not cover: a genomic `chr:pos:ref:alt`
        (or the coordinate an rsID lookup already produced) in, the amino-acid change and
        UniProt/ClinVar variant annotation out. Each SNV becomes a GRCh38 RefSeq genomic
        HGVS and is looked up through the EBI Proteins variation/hgvs endpoint, so the
        residue change, disease association, clinical significance and population
        frequency are curated values, not inferred from the reference sequence.

        Only the reviewed (Swiss-Prot) protein and its isoforms are reported; the TrEMBL
        predicted entries the endpoint also returns are dropped. A variant with no coding
        consequence (intronic, intergenic, or simply unannotated) comes back with an
        explicit note rather than an empty result that reads as a failed call, and a
        non-SNV allele is reported as unsupported rather than silently mapped to nothing.
        """
        results: list[dict[str, Any]] = []
        for raw in _as_list(variants):
            row: dict[str, Any] = {"variant": raw}
            try:
                hgvs, normalised = _genomic_hgvs(raw)
            except ValueError as exc:
                row["note"] = str(exc)
                results.append(row)
                continue
            row["normalized"] = normalised
            row["genomic_hgvs"] = hgvs
            body = await self._get(f"{self._ebi_url}/variation/hgvs/{hgvs}")
            if _is_error(body):
                row["_error"] = body["_error"]
                row["_status"] = body.get("_status")
                results.append(row)
                continue
            entries = body if isinstance(body, list) else []
            effects = [
                effect
                for entry in entries
                if isinstance(entry, dict) and not _variation_entry_is_predicted(entry)
                for effect in _flatten_variation_entry(entry)
            ]
            effects.sort(key=lambda e: (not e.get("canonical"), str(e.get("accession") or "")))
            if effects:
                row["effects"] = effects
            else:
                row["note"] = (
                    "no curated protein consequence for this variant in UniProt "
                    "(non-coding, or not annotated on a reviewed entry)"
                )
            results.append(row)
        return {"success": True, "assembly": "GRCh38", "results": results}

    async def search_uniprot(
        self,
        query: str | None = None,
        keyword: str | None = None,
        organism_id: int | None = None,
        reviewed_only: bool = True,
        fields: list[str] | str | None = None,
        size: int | None = None,
        count_only: bool = False,
    ) -> dict[str, Any]:
        """Search UniProtKB, reporting the query that was actually sent.

        `resolution` here answers the same question as it does for a single protein: what
        was searched for, in which organism, over reviewed entries or not. Without it a
        count is a bare number with no way to tell what it counted.

        `count_only` reads the x-total-results header rather than paging the hits
        (keyword:KW-0865 AND organism_id:9606 AND reviewed:true -> 215).
        """
        try:
            taxon_id = _organism_taxon(organism_id)
        except (TypeError, ValueError):
            return {
                "_error": f"UniProt: organism_id must be a positive taxon id, got {organism_id!r}",
                "_status": None,
            }

        clauses: list[str] = []
        text = _WILDCARD_RE.sub("", query or "").strip()
        if text:
            clauses.append(text)
        if keyword:
            clauses.append(f"keyword:{_keyword_term(keyword)}")
        if taxon_id is not None:
            clauses.append(f"organism_id:{taxon_id}")
        if reviewed_only:
            clauses.append("reviewed:true")
        if not clauses:
            return {
                "_error": "UniProt: a search needs at least a query or a keyword",
                "_status": None,
            }
        query_sent = " AND ".join(clauses)
        resolution = {
            "query_sent": query_sent,
            "query": query,
            "keyword": keyword,
            "organism_id": taxon_id,
            "reviewed_only": reviewed_only,
            "match_basis": "search",
        }

        params: dict[str, Any] = {
            "query": query_sent,
            "fields": ",".join(str(item) for item in _as_list(fields)) or _RESOLVE_FIELDS,
            # a count still needs a request; one hit is the smallest page that is
            # certainly accepted, and the total comes from the header either way
            "size": 1 if count_only else max(1, min(int(size or _DEFAULT_SEARCH_SIZE), 500)),
        }
        body, meta = await self._get_with_meta("/uniprotkb/search", params=params)
        if _is_error(body):
            return {**body, "resolution": resolution}

        hits = _reviewed_first(body.get("results") or [])
        total = _total_results(meta, len(hits))
        if count_only:
            return {"success": True, "resolution": resolution, "count": total}
        return {
            "success": True,
            "resolution": resolution,
            "count": total,
            "returned": len(hits),
            "results": [_entry_summary(hit) for hit in hits],
        }

    async def _get_coordinates(self, accession: str, position: int) -> Any:
        """EBI genomic coordinates for one residue of one accession.

        Built as an absolute URL because the endpoint's path segment is
        `{accession}:{position}` and _build_url percent-encodes the colon, which EBI
        answers with a 400. Both parts are therefore re-checked here: the accession
        against the same anchored, alphanumeric-only pattern resolve uses, and the
        position as an int, so neither can carry path syntax.
        """
        if not _ACCESSION_RE.match(accession):
            return {"_error": f"EBI Proteins API: not an accession: {accession!r}", "_status": None}
        return await self._get(f"{self._ebi_url}/coordinates/location/{accession}:{int(position)}")

    async def _annotate_batch(
        self,
        queries: list[Any],
        organism_id: int | None,
        fields: str,
        wants_features: bool,
        feature_types: list[str] | str | None,
        residue_range: Any,
    ) -> dict[str, Any]:
        """Annotate a list of inputs as one table of rows.

        Accession-shaped inputs skip resolution and go through fetch_batch, because the
        entry that comes back states its own identity — the same evidence resolve would
        have produced, at a hundredth of the requests. Symbols still resolve one at a
        time, since only the gene_exact ladder can pin a symbol to a single protein.

        Every row carries its own `query`, `match_basis` and full identity, so the table
        says per row which protein answered for which input; an input that resolved to
        nothing is listed in `unresolved` rather than dropped, so the table cannot come
        back quietly short.
        """
        requested: dict[str, str] = {}
        basis: dict[str, dict[str, Any]] = {}
        unresolved: list[dict[str, Any]] = []
        for item in queries:
            text = _WILDCARD_RE.sub("", str(item or "")).strip()
            if not text:
                continue
            candidate = text.upper()
            if _ACCESSION_RE.match(candidate):
                requested.setdefault(candidate, text)
                basis.setdefault(
                    candidate, {"query": text, "match_basis": "accession", "ambiguous": False}
                )
                continue
            resolution = await self.resolve(text, organism_id=organism_id)
            if _is_error(resolution) or not resolution.get("accession"):
                unresolved.append(
                    {
                        "query": text,
                        "error": (
                            resolution.get("_error")
                            or resolution.get("warning")
                            or "resolved to no live UniProtKB entry"
                        ),
                    }
                )
                continue
            accession = resolution["accession"]
            requested.setdefault(accession, text)
            basis.setdefault(
                accession,
                {
                    "query": text,
                    "match_basis": resolution.get("match_basis"),
                    "ambiguous": resolution.get("ambiguous"),
                    "warning": resolution.get("warning"),
                },
            )

        batch = await self.fetch_batch(list(requested), fields=fields)
        by_accession: dict[str, dict[str, Any]] = {}
        for entry in batch["results"]:
            accession = entry.get("primaryAccession")
            if accession:
                by_accession[accession] = entry
            for secondary in entry.get("secondaryAccessions") or []:
                by_accession.setdefault(secondary, entry)

        rows: list[dict[str, Any]] = []
        for accession, text in requested.items():
            entry = by_accession.get(accession)
            if entry is None:
                unresolved.append({"query": text, "error": f"{accession} returned no entry"})
                continue
            row = {**basis.get(accession, {"query": text}), **flatten_entry(entry)}
            if wants_features:
                row["features"] = flatten_features(
                    entry.get("features"), feature_types=feature_types, residue_range=residue_range
                )
            rows.append(row)

        return {
            "success": True,
            "results": rows,
            "count": len(rows),
            "unresolved": unresolved,
            "errors": batch["errors"],
        }


# UniProtKB accessions endpoint limit, verified live
_BATCH_SIZE = 100
_DEFAULT_SEARCH_SIZE = 25
# everything flatten_entry reads, and nothing else: a full TTN entry is megabytes
_ENTRY_FIELDS = (
    "accession,id,protein_name,gene_names,organism_name,"
    "cc_function,cc_subcellular_location,keyword,sequence,cc_alternative_products"
)
_CONFLICT_FIELDS = "ft_conflict,ft_var_seq"
# UniProt return-field name per supported feature type, so a per-residue request over a
# titin-sized entry asks for the domains it wants rather than the whole feature table
_FEATURE_FIELDS: dict[str, str] = {
    "active site": "ft_act_site",
    "propeptide": "ft_propep",
    "signal": "ft_signal",
    "domain": "ft_domain",
    "modified residue": "ft_mod_res",
    "topological domain": "ft_topo_dom",
    "sequence conflict": "ft_conflict",
    "alternative sequence": "ft_var_seq",
    "natural variant": "ft_variant",
}
# the same type under the names the two APIs and the flag files use: UniProtKB JSON says
# "Modified residue", the EBI payloads say "modified residue", and the flag-file codes
# (MOD_RES, VAR_SEQ) are what a curator types
_FEATURE_ALIASES: dict[str, str] = {
    "act site": "active site",
    "propep": "propeptide",
    "signal peptide": "signal",
    "mod res": "modified residue",
    "topo dom": "topological domain",
    "conflict": "sequence conflict",
    "var seq": "alternative sequence",
    "varseq": "alternative sequence",
    "splice variant": "alternative sequence",
    "variant": "natural variant",
    "variants": "natural variant",
}
_AA3_TO_1: dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O", "ASX": "B", "GLX": "Z", "XAA": "X",
    "TER": "*", "STOP": "*",
}
_AA1 = frozenset("ACDEFGHIKLMNPQRSTVWYUOBZX*")
# 'P70A', 'p.Pro70Ala', 'Pro70Ala', '70'; the reference residue is optional only so a
# bare position stays usable, never so a caller can drop it from a full variant string
_VARIANT_RE = re.compile(
    r"^(?:p\.)?(?P<ref>[A-Za-z]{3}|[A-Za-z*])?\s*(?P<pos>\d+)\s*(?P<alt>[A-Za-z]{3}|[A-Za-z*=])?$"
)


def _as_list(value: Any) -> list[Any]:
    """A scalar, a sequence or None as a list — tool arguments arrive as any of the three."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set, frozenset)):
        return list(value)
    return [value]


def _normalize_feature_type(value: Any) -> str:
    text = str(value or "").strip().lower().replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return _FEATURE_ALIASES.get(text, text)


def _feature_fields(feature_types: Any) -> str:
    """UniProt return fields covering `feature_types`, or all supported types."""
    wanted = {_normalize_feature_type(item) for item in _as_list(feature_types)}
    fields = [field for key, field in _FEATURE_FIELDS.items() if not wanted or key in wanted]
    # an unrecognised type would otherwise project no feature fields at all and look like
    # "this protein has no such features" instead of "that is not a type I know"
    return ",".join(fields or _FEATURE_FIELDS.values())


def _keyword_term(keyword: str) -> str:
    """A keyword as UniProt indexes it: the KW- identifier bare, a keyword name quoted."""
    text = str(keyword).strip()
    return text.upper() if re.match(r"^KW-\d+$", text, re.IGNORECASE) else _quoted_term(text)


def _residue_range(value: Any) -> tuple[int, int] | None:
    """`residue_range` as an inclusive (start, end), accepting 70, (1, 100) and '1-100'."""
    if value is None:
        return None
    if isinstance(value, str):
        parts = [part for part in re.split(r"[-:.\s]+", value.strip()) if part]
    elif isinstance(value, (list, tuple)):
        parts = [str(part) for part in value]
    else:
        parts = [str(value)]
    if not parts:
        return None
    try:
        bounds = [int(part) for part in parts[:2]]
    except (TypeError, ValueError):
        raise ValueError(f"residue_range must be numeric, got {value!r}") from None
    start, end = (bounds * 2)[:2]
    return (start, end) if start <= end else (end, start)


def one_letter_aa(value: Any) -> str | None:
    """'Pro' or 'P' as 'P'; None when it is neither. Three-letter input is what the EBI
    coordinates endpoint reports and one-letter is what a variant string carries, so the
    two can only be compared after both pass through here."""
    text = str(value or "").strip()
    if not text:
        return None
    if len(text) == 1:
        return text.upper() if text.upper() in _AA1 else None
    return _AA3_TO_1.get(text.upper())


def _evidence(feature: dict[str, Any]) -> list[str]:
    """Evidence codes of a feature, from either API's spelling of them."""
    codes: list[str] = []
    for item in (feature.get("evidences") or feature.get("evidence") or []):
        if not isinstance(item, dict):
            continue
        code = item.get("evidenceCode") or item.get("code")
        if not code:
            continue
        source, ref = item.get("source"), item.get("id")
        codes.append(f"{code}|{source}:{ref}" if source and ref else str(code))
    return codes


def _position(node: Any) -> int | None:
    if isinstance(node, dict):
        value = node.get("value", node.get("position"))
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    try:
        return int(node)
    except (TypeError, ValueError):
        return None


def _feature_bounds(feature: dict[str, Any]) -> tuple[int | None, int | None]:
    """Start and end of a feature, over the three location shapes in play: UniProtKB's
    start/end, the EBI's begin/end, and the single-residue position of both."""
    location = feature.get("location") or {}
    if "position" in location:
        point = _position(location.get("position"))
        return point, point
    start = _position(location.get("start", location.get("begin")))
    end = _position(location.get("end"))
    return start, end if end is not None else start


def flatten_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """A UniProtKB entry as the compact record a model can actually read.

    Identity comes from _entry_summary, so a flattened entry says which protein it is in
    the same fields resolution does. Built fresh rather than edited in place: entry
    bodies are shared by the response cache.
    """
    entry = entry or {}
    functions: list[str] = []
    locations: list[str] = []
    for comment in entry.get("comments") or []:
        kind = str(comment.get("commentType") or "").upper()
        if kind == "FUNCTION":
            functions.extend(
                text.get("value") for text in comment.get("texts") or [] if text.get("value")
            )
        elif kind == "SUBCELLULAR LOCATION":
            for item in comment.get("subcellularLocations") or []:
                value = (item.get("location") or {}).get("value")
                if value:
                    locations.append(value)
    sequence = entry.get("sequence") or {}
    return {
        **_entry_summary(entry),
        "function": " ".join(functions) or None,
        "subcellular_location": list(dict.fromkeys(locations)),
        "keywords": [kw.get("name") for kw in entry.get("keywords") or [] if kw.get("name")],
        "sequence_length": sequence.get("length"),
        "isoform_count": len(_isoforms(entry)) or 1,
    }


def _isoforms(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Isoforms of an entry, from its ALTERNATIVE PRODUCTS comment."""
    isoforms: list[dict[str, Any]] = []
    for comment in (entry or {}).get("comments") or []:
        if str(comment.get("commentType") or "").upper() != "ALTERNATIVE PRODUCTS":
            continue
        for isoform in comment.get("isoforms") or []:
            isoforms.append(
                {
                    "name": (isoform.get("name") or {}).get("value"),
                    "ids": list(isoform.get("isoformIds") or []),
                    "sequence_status": isoform.get("isoformSequenceStatus"),
                }
            )
    return isoforms


def flatten_features(
    features: list[dict[str, Any]] | None,
    feature_types: list[str] | str | None = None,
    residue_range: Any = None,
) -> list[dict[str, Any]]:
    """Sequence features as flat {type, start, end, description, evidence} rows.

    `feature_types` matches a type under any of its spellings (MOD_RES, 'modified
    residue', 'Modified residue'); `residue_range` keeps every feature OVERLAPPING the
    range rather than only those contained in it, because a TTN Ig domain spanning the
    residue of interest is precisely what the question is about, and an active site at a
    single residue must survive a range query that brackets it.
    """
    wanted = {_normalize_feature_type(item) for item in _as_list(feature_types)}
    bounds = _residue_range(residue_range)
    rows: list[dict[str, Any]] = []
    for feature in features or []:
        if not isinstance(feature, dict):
            continue
        kind = feature.get("type")
        if wanted and _normalize_feature_type(kind) not in wanted:
            continue
        start, end = _feature_bounds(feature)
        if bounds is not None:
            if start is None or end is None:
                continue
            if end < bounds[0] or start > bounds[1]:
                continue
        description = feature.get("description")
        if not description and feature.get("original"):
            variation = ", ".join(str(item) for item in _as_list(feature.get("variation")))
            description = f"{feature['original']} -> {variation}" if variation else None
        row = {
            "type": kind,
            "start": start,
            "end": end,
            "description": description or None,
            "evidence": _evidence(feature),
        }
        if _normalize_feature_type(kind) == "natural variant":
            row.update(_variant_change(feature))
        rows.append(row)
    return rows


def _variant_change(feature: dict[str, Any]) -> dict[str, Any]:
    """The residue change and cross-references of a Natural variant feature.

    The `alternativeSequence` shape is UniProtKB's ('originalSequence' plus a list of
    'alternativeSequences'); `original`/`variation` is the EBI Proteins spelling of the
    same thing, already folded into `description` above but repeated here as structured
    fields. The dbSNP identifier a curator recorded lives in `featureCrossReferences`,
    which is what turns a residue change into something the genomic tools can look up.
    """
    alt = feature.get("alternativeSequence") or {}
    original = alt.get("originalSequence") or feature.get("original")
    variants = _as_list(alt.get("alternativeSequences")) or _as_list(feature.get("variation"))
    xrefs: dict[str, str] = {}
    for ref in feature.get("featureCrossReferences") or feature.get("xrefs") or []:
        if not isinstance(ref, dict):
            continue
        database, identifier = ref.get("database") or ref.get("name"), ref.get("id")
        if database and identifier:
            xrefs.setdefault(str(database), str(identifier))
    change: dict[str, Any] = {}
    if original is not None:
        change["original_aa"] = original
    if variants:
        change["variant_aa"] = variants if len(variants) > 1 else variants[0]
    if feature.get("featureId") or feature.get("ftId"):
        change["feature_id"] = feature.get("featureId") or feature.get("ftId")
    if xrefs.get("dbSNP"):
        change["dbsnp"] = xrefs["dbSNP"]
    if xrefs:
        change["xrefs"] = xrefs
    return change


def _genomic_hgvs(variant: str) -> tuple[str, str]:
    """A `chr:pos:ref:alt` GRCh38 variant as (RefSeq genomic HGVS, normalised id).

    Only SNVs are converted: the EBI variation/hgvs endpoint answers indels and MNVs
    unreliably (an insertion HGVS that is valid but unnormalised comes back empty, which
    reads as 'no consequence' when it means 'not looked up'). A non-SNV therefore raises
    with a message the tool surfaces as a per-variant note rather than a silent empty.

    Raises ValueError on an unparseable id, an unknown chromosome, or a non-SNV allele.
    """
    match = _GENOMIC_VARIANT_RE.match(str(variant or "").strip())
    if not match:
        raise ValueError(
            f"cannot parse genomic variant {variant!r}; expected 'chr:pos:ref:alt' "
            f"such as '12:40340400:G:A'"
        )
    chrom = match.group("chr").upper()
    if chrom.startswith("CHR"):
        chrom = chrom[3:]
    chrom = {"M": "MT", "23": "X", "24": "Y"}.get(chrom, chrom)
    refseq = _GRCH38_REFSEQ.get(chrom)
    if refseq is None:
        raise ValueError(f"unknown GRCh38 chromosome {match.group('chr')!r} in {variant!r}")
    ref, alt = match.group("ref").upper(), match.group("alt").upper()
    if len(ref) != 1 or len(alt) != 1:
        raise ValueError(
            f"{variant!r} is not an SNV; genomic protein-effect mapping here supports "
            f"single-nucleotide substitutions only"
        )
    normalised = f"{chrom}:{match.group('pos')}:{ref}:{alt}"
    return f"{refseq}:g.{match.group('pos')}{ref}>{alt}", normalised


def _flatten_variation_entry(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """The protein-consequence rows of one EBI variation/hgvs entry.

    One entry is one protein (the canonical accession or a specific isoform); each of its
    features is the consequence of the queried genomic change on that sequence. `wildType`
    /`mutatedType` are authoritative for the residue change — `consequenceType` mislabels
    a stop as 'missense' — so a stop is recognised from a '*' mutatedType here.
    """
    accession = entry.get("accession")
    reviewed = not _variation_entry_is_predicted(entry)
    rows: list[dict[str, Any]] = []
    for feature in entry.get("features") or []:
        if not isinstance(feature, dict):
            continue
        wild, mutated = feature.get("wildType"), feature.get("mutatedType")
        consequence = feature.get("consequenceType")
        if mutated == "*" or (isinstance(mutated, str) and mutated.upper() in {"*", "TER", "STOP"}):
            consequence = "stop_gained"
        protein_change = None
        for location in feature.get("locations") or []:
            loc = location.get("loc") if isinstance(location, dict) else None
            if isinstance(loc, str) and loc.startswith("p."):
                protein_change = loc
                break
        rows.append(
            {
                "accession": accession,
                "entry_name": entry.get("entryName"),
                "gene": entry.get("geneName"),
                "protein_name": entry.get("proteinName"),
                "reviewed": reviewed,
                "canonical": bool(accession) and "-" not in str(accession),
                "position": _position(feature.get("begin")),
                "wild_type_aa": wild,
                "variant_aa": mutated,
                "protein_change": protein_change,
                "consequence": consequence,
                "clinical_significance": [
                    sig.get("type")
                    for sig in feature.get("clinicalSignificances") or []
                    if isinstance(sig, dict) and sig.get("type")
                ],
                "diseases": [
                    assoc.get("name")
                    for assoc in feature.get("association") or []
                    if isinstance(assoc, dict) and assoc.get("name")
                ],
                "population_frequencies": [
                    {
                        "source": freq.get("source"),
                        "population": freq.get("populationName"),
                        "frequency": freq.get("frequency"),
                    }
                    for freq in feature.get("populationFrequencies") or []
                    if isinstance(freq, dict)
                ],
                "xrefs": _variation_xrefs(feature),
                "feature_id": feature.get("ftId"),
            }
        )
    return rows


def _variation_entry_is_predicted(entry: dict[str, Any]) -> bool:
    """True for an unreviewed (TrEMBL) variation entry.

    The variation payload carries no reviewed flag, so two signals stand in: a 'Predicted'
    proteinExistence, and a mnemonic entryName equal to the accession (Swiss-Prot names are
    always alphabetic — LRRK2_HUMAN — while TrEMBL reuses the accession, A0ACI8UJW1_HUMAN).
    """
    if str(entry.get("proteinExistence") or "").strip().lower() == "predicted":
        return True
    accession = str(entry.get("accession") or "")
    entry_name = str(entry.get("entryName") or "")
    base = accession.split("-", 1)[0]
    return bool(base) and entry_name.upper().startswith(base.upper())


def _variation_xrefs(feature: dict[str, Any]) -> dict[str, str]:
    """First identifier per database from a variation feature's xref list (dbSNP, ClinVar)."""
    xrefs: dict[str, str] = {}
    for ref in feature.get("xrefs") or []:
        if not isinstance(ref, dict):
            continue
        name, identifier = ref.get("name"), ref.get("id")
        if name and identifier:
            xrefs.setdefault(str(name), str(identifier))
    return xrefs


def parse_protein_variant(variant: str) -> tuple[int, str | None]:
    """A protein variant string as (position, expected_aa in one-letter form).

    Accepts 'P70A', 'p.Pro70Ala', 'Pro70Ala' and a bare '70'. The expected residue is
    read out of the caller's own string and never taken from a separate parameter, so a
    mapping cannot be run with the residue check silently skipped — an omitted residue is
    visible in the input itself, as a bare position.

    Raises ValueError on anything else, including a reference that is not an amino acid.
    """
    text = str(variant or "").strip()
    match = _VARIANT_RE.match(text)
    if not match:
        raise ValueError(
            f"cannot parse protein variant {variant!r}; expected forms like "
            f"'P70A', 'p.Pro70Ala' or a bare position '70'"
        )
    position = int(match.group("pos"))
    if position <= 0:
        raise ValueError(f"protein position must be positive, got {position}")
    ref = match.group("ref")
    if ref is None:
        return position, None
    expected = one_letter_aa(ref)
    if expected is None:
        raise ValueError(f"{ref!r} in {variant!r} is not an amino acid")
    return position, expected


def collapse_transcripts(
    locations: list[dict[str, Any]] | None,
    expected_aa: str | None = None,
) -> list[dict[str, Any]]:
    """Collapse the EBI coordinates rows for one residue into distinct genomic intervals.

    The endpoint answers with one row per Ensembl transcript — P07202 residue 70 returns
    dozens — and they overwhelmingly agree. Taking the first would hide the case where
    they do not, so rows are grouped by (chromosome, geneStart, geneEnd, aminoAcids) and
    every transcript that voted for an interval is listed on it.

    With `expected_aa`, each interval reports `agrees` against the residue the EBI places
    there. A disagreement never removes the coordinates: the row keeps them and gains the
    overlapping sequence-conflict and VAR_SEQ features the payload carries, which is what
    explains a mismatch at a residue like P07202:70.
    """
    expected = one_letter_aa(expected_aa)
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in locations or []:
        if not isinstance(row, dict):
            continue
        amino_acids = row.get("aminoAcids")
        key = (row.get("chromosome"), row.get("geneStart"), row.get("geneEnd"), amino_acids)
        entry = grouped.get(key)
        if entry is None:
            observed = one_letter_aa(amino_acids)
            entry = {
                "chromosome": row.get("chromosome"),
                "gene_start": row.get("geneStart"),
                "gene_end": row.get("geneEnd"),
                "strand": "reverse" if row.get("reverseStrand") else "forward",
                "assembly": row.get("assemblyName"),
                "protein_start": row.get("proteinStart"),
                "protein_end": row.get("proteinEnd"),
                "amino_acids": amino_acids,
                "amino_acid": observed,
                "expected_aa": expected,
                "agrees": None if expected is None else observed == expected,
                "genes": [],
                "transcripts": [],
            }
            if expected is not None and observed != expected:
                position = row.get("proteinStart") or row.get("proteinEnd")
                entry["conflicting_features"] = flatten_features(
                    row.get("features"),
                    feature_types=["sequence conflict", "VAR_SEQ"],
                    residue_range=(position, row.get("proteinEnd") or position)
                    if position
                    else None,
                )
            grouped[key] = entry
        for source, target in (("ensemblTranscriptId", "transcripts"), ("ensemblGeneId", "genes")):
            value = row.get(source)
            if value and value not in entry[target]:
                entry[target].append(value)
    return list(grouped.values())


def _disagreement_evidence(entry: dict[str, Any], position: int) -> dict[str, Any]:
    """What to show when the mapped residue is not the one the caller expected.

    Length and isoform count first, because the commonest cause is a position numbered
    against a different isoform, then any curated conflict over that exact residue.
    """
    return {
        "sequence_length": ((entry or {}).get("sequence") or {}).get("length"),
        "isoform_count": len(_isoforms(entry)) or 1,
        "features": flatten_features(
            (entry or {}).get("features"),
            feature_types=["sequence conflict", "VAR_SEQ"],
            residue_range=(position, position),
        ),
    }
