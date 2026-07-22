"""Client for the UniProt REST API and the EBI Proteins API.

Protein annotation logic lives here rather than in executor.py so it can be unit tested
without a ToolExecutor; the executor exposes thin delegating tool methods.
"""

import logging
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


def _is_error(data: Any) -> bool:
    """True when `data` is the sentinel returned by UniProtClient._get on failure.

    Callers must use this instead of `"_error" in data`: endpoints differ in shape
    (/uniprotkb/{accession} returns an object, the EBI /proteins?accession= and
    /coordinates?accession= endpoints return an array), and on a list `in` silently
    degrades into an element-membership test while `data.get("_error")` raises. Error
    sentinels are always dicts.
    """
    return isinstance(data, dict) and "_error" in data


class UniProtClient:
    """Async client for UniProtKB and the EBI Proteins API."""

    def __init__(self, client: httpx.AsyncClient, settings: Settings):
        # the httpx client is injected, not constructed here: the executor passes its
        # _ResilientAsyncClient (30s timeout, no auth headers), so connection failures
        # arrive as a synthetic 503 instead of raising, and tests can mock it
        self._client = client
        self._uniprot_url = settings.uniprot_api_url.rstrip("/")
        self._ebi_url = settings.ebi_proteins_api_url.rstrip("/")
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
        so the TTL cache can wrap it in a single place.

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
            return resp.json(), meta
        except ValueError:
            # UniProt serves an HTML placeholder page when its REST tier is unavailable,
            # which would otherwise surface as an opaque JSON decode error
            label = self._label(url)
            logger.warning(f"{label} returned non-JSON body: {url}")
            return {
                "_error": f"{label} returned a non-JSON response",
                "_status": resp.status_code,
            }, meta

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
