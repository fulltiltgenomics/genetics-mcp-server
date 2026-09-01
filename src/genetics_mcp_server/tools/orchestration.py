"""The half of the tool executor that orchestrates other services.

`tools/executor.py` is the only genetics_mcp_server module outside `sdk/` that the sandbox
image ships: `sdk/client.py` imports `ToolExecutor` and every SDK function delegates to it.
What is here are the entry points that hand work to something else — the sandbox, Tavily or
DuckDuckGo, Europe PMC or Perplexity — and the private helpers only they use. Between them
they reach `ddgs`, `sandbox_client`, `auth.core` and `sandbox_token`, none of which that
image contains, so none of it can run there; and the source is the run_analysis gateway, the
identity it refuses to dispatch without, and the artifact authorization model, none of which
a reader needs in order to import the SDK.

A SUBCLASS rather than a mixin, so that nothing in the shipped module names this one. A
mixin would put a module-level import of this file into `executor.py` plus a fallback base
class for the image that lacks it, and that fallback makes the absence unobservable to the
build's `import genetics_mcp_server.sdk`. The dependency runs one way instead, so re-merging
the halves fails that build rather than shipping the source.

Server processes construct `ServerToolExecutor` (`genetics_mcp_server.tools` re-exports it
lazily, so importing the SDK's half does not pull this one in). `sdk/client.py` constructs
the base, so the executor a script reaches through `GeneticsClient._executor` is the
data-access half — everywhere, not only in the sandbox.
"""

import asyncio
import logging
import mimetypes
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Iterable
from functools import cached_property
from pathlib import Path
from typing import Any
from urllib.parse import quote

from genetics_mcp_server.tools.executor import ToolExecutor, _resolve_settings

logger = logging.getLogger(__name__)


# Modules the sandbox image does not contain, read by two different things here.
#
# `_absent_capability_named` reads the inventory back out of the text the supervisor
# forwards from a failed execution, so `_analysis_hint` can tell the model a capability is
# ABSENT rather than being called wrongly. That is the live reader and it runs server-side:
# a script can still `import ddgs` itself, or reach `tools.definitions` through the lazy
# hook in `tools/__init__.py`, and produce exactly these errors. Re-derive the set with an
# AST walk over what the image ships; do not trust the list.
#
# `_pruned_module` guards the deferred imports below. This module is not shipped to the
# sandbox, so those branches CANNOT fire there — the code holding them is not in the image —
# and what they now answer for is an incomplete install of the distribution. Every name below
# is pinned or packaged, so a complete install has all of them: `ddgs` is a hard pin in
# pyproject's `[project].dependencies` and the rest are modules of this package. Keep the
# guards anyway, because the two are not equally reachable: `ddgs` is a separate
# distribution, so `pip install --no-deps` or a half-applied upgrade can leave it out of an
# otherwise working environment, while the rest can only be missing if this package's own
# files are damaged.
#
# Naming a module here does NOT make its capability work; it makes the absence legible.
# Widening the image instead was considered and rejected: the sandbox NetworkPolicy grants
# no DNS, so ddgs would stall the glibc resolver rather than fail fast, and shipping
# sandbox_client would hand a prompt-injected script a confused-deputy surface with no
# egress rule able to use it.
_SANDBOX_PRUNED_MODULES = (
    "ddgs",
    "genetics_mcp_server.auth",
    "genetics_mcp_server.auth.core",
    "genetics_mcp_server.sandbox_client",
    "genetics_mcp_server.sandbox_token",
    "genetics_mcp_server.tools.definitions",
    # this module. `from genetics_mcp_server.tools import ServerToolExecutor` inside the
    # sandbox raises for it, and without the entry the hint would send the model to
    # list_capabilities for something no image will ever list.
    "genetics_mcp_server.tools.orchestration",
)

# stamped on every result the guards below return. Deliberately NOT SandboxNotConfigured:
# that is an operator fault on a server that HAS the code and lacks an address, which an
# operator can fix. This is the code not being installed at all, so there is nothing to
# configure.
CAPABILITY_UNAVAILABLE = "CapabilityUnavailable"


def _pruned_module(exc: ModuleNotFoundError, *names: str) -> str | None:
    """The name among `names` that `exc` reports missing, or None if the fault is elsewhere.

    The same discrimination `_resolve_settings` makes, for the same reason: a
    ModuleNotFoundError raised from DEEPER INSIDE a module that IS installed — a missing
    third-party dependency of it, today or tomorrow — is a broken install, and swallowing
    it as a pruned sandbox would hide a real fault outside the sandbox.
    """
    return exc.name if exc.name in names else None


def _capability_absent_message(capability: str, module: str) -> str:
    return (
        f"{capability} is not available in this environment: `{module}` is not installed "
        "here, so no rewrite of the script can reach the capability."
    )


def _absent_capability_named(message: Any) -> str | None:
    """The pruned module a supervisor-reported error message names, if any.

    The supervisor forwards the child's exception as text, so `exc.name` does not survive
    the hop and the name has to be read back out of the message. Matched with the quotes
    CPython writes, so `genetics_mcp_server.auth` does not match a report about
    `genetics_mcp_server.auth.core`.

    TWO MESSAGE SHAPES, and the second is not optional. A plain `import a.b.c` of a pruned
    module reports `No module named 'a.b.c'`. But `from a.b import c` where `c` was pruned
    reports an **ImportError** — `cannot import name 'c' from 'a.b'` — which contains the
    package and the attribute but NEVER the dotted path, so a search for the full name
    misses it entirely. That is the shape `tools/__init__.py`'s lazy `__getattr__` produces
    in the image, verified in the deployed sandbox pod, and ImportError is itself in
    `_SDK_MISUSE_ERROR_TYPES` — so without this branch the one instance reached through a
    package attribute would collect exactly the misleading hint this function exists to
    prevent.
    """
    if not isinstance(message, str):
        return None
    for name in _SANDBOX_PRUNED_MODULES:
        if "No module named" in message and f"'{name}'" in message:
            return name
        package, _, attribute = name.rpartition(".")
        if package and f"cannot import name '{attribute}' from '{package}'" in message:
            return name
    return None


# Mirrors ARTIFACT_READ_MAX_BYTES in genetics_mcp_server.sandbox_client, which mirrors
# sandbox/supervisor.py. Duplicated rather than imported ON PURPOSE: this module is imported
# by the standalone MCP server, and tests/test_mcp_server.py asserts in a subprocess that
# importing it does not pull sandbox_client into sys.modules. tests/test_code_execution_tools.py
# asserts the two numbers are equal, so the copy cannot drift silently.
ARTIFACT_READ_MAX_BYTES = 512 * 1024

# sandbox/supervisor.py's RETENTION_S. THIS IS A LIFETIME, NOT A POLICY KNOB: the supervisor
# deletes /scratch/<id>/artifacts this many seconds after an execution completes, and the
# per-execution key that decrypts a sealed artifact lives only in that process's memory. So
# nothing chat-backend records about an execution can be worth more than this — a longer-lived
# record would promise reads that can only come back 404 or 409.
ARTIFACT_RETENTION_S = 300


def _is_identity(value: Any) -> bool:
    """A usable half of an artifact key: a non-empty string, and nothing else.

    `isinstance` rather than truthiness so every guard below is TOTAL. A non-str identity is
    unhashable and would raise out of the `(sub, sid)` tuple lookup instead of failing closed
    — a distinct response shape on a security path. `get_authenticated_user` returns
    `str | None`, so nothing produces one today; the guard is meant to hold regardless.
    """
    return isinstance(value, str) and bool(value)


class _ArtifactManifests:
    """`(sub, sid)` -> the artifacts each of that session's recent executions reported writing.

    THE AUTHORIZATION STATE FOR `read_artifact` (genetics-results-suite-4h6.52). `run_analysis`
    records a completed execution's manifest here against the AUTHENTICATED user AND session;
    the tool resolves a model-supplied NAME against it and can address nothing else. A name
    recorded under another key is not merely refused, it is invisible: `resolve` looks in one
    key's rows and returns `None` for everything else, which is the same answer a name that
    never existed gets.

    THE KEY CARRIES A USER TERM, AND THAT IS THE WHOLE OF genetics-results-suite-dh3. `sid` is
    the CLIENT'S value: it arrives in the `/v1/chat` body and is never checked against the
    caller, so keying on it alone meant user B could put user A's session id in B's own request
    and have B's model read A's artifacts. `sub` is the half that is not the caller's to choose,
    but only because BOTH tools that touch this map require the auth-gateway provenance
    assertion: `get_authenticated_user` honours the proxy identity header from any caller
    presenting INTERNAL_API_SECRET, so `sub` is exactly as forgeable as `sid` to a marker holder
    until `gateway_asserted` is required. `run_analysis` has required it since
    genetics-results-suite-4h6.84 and `read_artifact` now does too — a read path without it made
    the user term a value the attacker supplied. With the gate on both, pairing the two makes a
    stolen or guessed `sid` resolve to nothing for anyone but its owner. The
    supervisor deliberately has no matching check (sandbox/supervisor.py: the sid-scoped
    resolution "belongs in chat-backend, the only side that knows which session owns which
    execution"), so this key is the only place the ownership question is asked.

    FAILS CLOSED ON A MISSING USER. `record` with no `sub` stores nothing and `resolve` with no
    `sub` returns `None`, rather than falling back to a session-only key — an unauthenticated or
    internal-only caller must not be able to write into, or read out of, a scope that a real
    user's `sub` would otherwise share.

    IN MEMORY, WITH A TTL MATCHED TO THE SUPERVISOR'S RETENTION, AND DELIBERATELY NOT
    PERSISTED. The thing this maps to is deleted by the supervisor `ARTIFACT_RETENTION_S`
    after completion, and its decryption key is held in the supervisor's process memory only,
    so a row that outlives either — across a restart of this process, or of the sandbox — points
    at bytes nobody can serve. Persisting the map would buy nothing but a longer window in which
    `read_artifact` promises a read that ends in a 404 or a 409. Expiring with the artifacts
    keeps the two sides failing together. `chat-backend` is `replicas: 1` (k8s/deployments/
    chat-backend.yaml), the same premise db-api's per-execution byte counter and results-api's
    sandbox budget already run on, so there is no second process to disagree with.

    BOUNDED, because this is per-process state fed by user activity: at most `_MAX_SESSIONS`
    sessions (LRU by last write) and `_MAX_EXECUTIONS` executions within each. NEITHER BOUND IS
    ABOVE WHAT THE RETENTION WINDOW CAN HOLD, and the earlier claim that both were ("an
    execution takes tens of seconds") was false for the scripts that actually run: a trivial
    script completes in well under a second, so one session can in principle stack ~600
    executions into a 300 s window. These are MEMORY bounds, but the dominant term is row
    SIZE, not row COUNT: `record()` puts no cap of its own on a row's name set, which is
    bounded only by the supervisor's `ARTIFACT_ENTRY_BUDGET = 1024` (sandbox/supervisor.py)
    x NAME_MAX 255 -- a worst-case row of 328.4 KiB against a typical 487 bytes. The naive
    512 x 128 x 328.4 KiB product is ~20.5 GiB against chat-backend's `limits.memory: 2Gi`
    (k8s/deployments/chat-backend.yaml); that is unreachable rather than a live risk, since
    filling it needs 65,536 executions inside one 300 s TTL through a single serialized
    sandbox, so the real worst case is throughput-bounded and one session sitting at the cap
    costs 41 MiB. (The OLD cap of 16 already produced 2.57 GiB on this same naive product, so
    16 -> 128 did not cross a threshold that 16 was on the safe side of -- `_MAX_EXECUTIONS`
    was never the binding term.) These are a chosen, benign failure mode, not backstops
    against a leak:

      * `_MAX_EXECUTIONS = 128` is sized off measurement, not off the window: the epic's
        benchmark turns reach 58 tool calls, so 128 artifact-producing executions inside one
        retention window leaves better than 2x headroom over anything observed. Past it the
        OLDEST row is evicted while its artifacts may still be on the sandbox's disk, and a
        read of one of those names answers ArtifactNotFound — non-retryable, worded "re-run the
        script if you need it again", which is the correct instruction for that state. That is
        the trade: a bounded map, and a legitimate read can 404 in a session that ran more than
        128 artifact-producing analyses in five minutes.
      * `_MAX_SESSIONS = 512` is an LRU cap on concurrently active `(sub, sid)` KEYS, evicted
        by last write — one user's one session is one key, so a user with several live chats
        holds several. Rows also expire with the retention window on every `record`/`resolve`,
        so this only binds when more than 512 such keys are live at once in one replica.
    """

    _MAX_SESSIONS = 512
    _MAX_EXECUTIONS = 128

    def __init__(self, ttl_s: float = ARTIFACT_RETENTION_S) -> None:
        self._ttl_s = ttl_s
        self._lock = threading.Lock()
        # (sub, sid) -> list of (recorded_at, execution_id, {names}), oldest first
        self._sessions: OrderedDict[tuple[str, str], list[tuple[float, str, set[str]]]] = (
            OrderedDict()
        )

    def record(
        self, user: str | None, session_id: str | None, execution_id: str, names: Iterable[str]
    ) -> None:
        wanted = {n for n in names if isinstance(n, str) and n}
        # no `user` means no owner to attribute the artifacts to, so nothing is recorded
        # rather than recorded under a session-only key another identity could reach
        if not _is_identity(user) or not _is_identity(session_id) or not execution_id or not wanted:
            return
        key = (user, session_id)
        now = time.monotonic()
        with self._lock:
            rows = self._sessions.get(key)
            if rows is None:
                rows = []
                self._sessions[key] = rows
            rows.append((now, execution_id, wanted))
            # if _MAX_EXECUTIONS is ever set to 0 this is a no-op (rows[:-0] == rows[:0]),
            # so the trim silently stops trimming; correct at 128, nothing sets it to 0 today
            del rows[: -self._MAX_EXECUTIONS]
            self._sessions.move_to_end(key)
            self._expire(now)
            while len(self._sessions) > self._MAX_SESSIONS:
                self._sessions.popitem(last=False)

    def resolve(self, user: str | None, session_id: str | None, name: str) -> str | None:
        """The execution this user's session `name` refers to, or None.

        MOST RECENT WINS. Two executions in one session can both write `manhattan.png`, and the
        model is asking about the one it was just told about; a stale-first rule would quietly
        hand back a previous turn's plot under the right name, which is a wrong answer rather
        than a loud failure.

        A MISSING `user` RESOLVES TO NOTHING, deliberately and not incidentally: without a
        server-derived subject there is no authorization to make, and a session-only lookup is
        exactly the cross-user read this key exists to prevent.
        """
        if not _is_identity(user) or not _is_identity(session_id) or not name:
            return None
        key = (user, session_id)
        now = time.monotonic()
        with self._lock:
            self._expire(now)
            rows = self._sessions.get(key)
            if not rows:
                return None
            for _, execution_id, names in reversed(rows):
                if name in names:
                    return execution_id
        return None

    def _expire(self, now: float) -> None:
        """Caller holds the lock. Drop rows past the retention window, then empty sessions."""
        cutoff = now - self._ttl_s
        for key in list(self._sessions):
            rows = [row for row in self._sessions[key] if row[0] > cutoff]
            if rows:
                self._sessions[key] = rows
            else:
                del self._sessions[key]

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()


# process-wide rather than per-ToolExecutor: the recording call and the resolving call are two
# different chat turns, and nothing guarantees they are served by the same executor instance
_ARTIFACT_MANIFESTS = _ArtifactManifests()


ARTIFACTS_RETAINED_IN_CLEAR_NOTE = (
    "The sandbox could not remove this run's output files from its shared scratch "
    "space, so they remain readable there by other code running in the sandbox "
    "until they are reaped. Nothing about YOUR data was disclosed to the user's "
    "detriment by this alone, and the analysis result above still stands — but "
    "mention to the user that the run's output files could not be cleaned up, do "
    "not re-read these artifacts as trusted input, and re-run the analysis if a "
    "conclusion depends on their exact contents."
)
"""MODULE-LEVEL BECAUSE llm_service RE-ATTACHES IT AFTER TRUNCATION (4h6.97 round 2).

A tool result over `mcp_max_result_size` is cut to a PREFIX, and `output` is script-
controlled and can be 64 KiB — so a script that both provokes the retained-in-clear
condition and prints ~50 KB deletes the warning from what the model reads. Ordering in
`_render_analysis` puts the field ahead of `output`, and `_truncation_notice` re-states it
from this constant; the two defences are independent on purpose, because the first relies
on JSON serialisation order and the second does not.
"""


class ServerToolExecutor(ToolExecutor):
    """The tool executor as the chat backend and the MCP server construct it."""

    # -------------------------------------------------------------------------
    # External Search Tools
    # -------------------------------------------------------------------------

    async def search_scientific_literature(
        self,
        query: str,
        max_results: int = 10,
        include_preprints: bool = True,
        date_range: str | None = None,
        backend: str | None = None,
    ) -> dict[str, Any]:
        """Search scientific literature via Europe PMC or Perplexity."""
        selected_backend = (
            backend
            or os.environ.get("LITERATURE_SEARCH_BACKEND", "perplexity")
        ).lower()

        if selected_backend == "perplexity":
            perplexity_api_key = os.environ.get("PERPLEXITY_API_KEY")
            if not perplexity_api_key:
                logger.error("Perplexity backend requested but PERPLEXITY_API_KEY not configured")
                return {
                    "success": False,
                    "backend": "perplexity",
                    "error": "Literature search with Perplexity is currently unavailable (API key not configured)",
                }
            try:
                result = await self._search_perplexity_literature(
                    query, max_results, perplexity_api_key, include_preprints, date_range
                )
            except Exception as e:
                logger.error(f"Perplexity search failed: {e}")
                return {
                    "success": False,
                    "backend": "perplexity",
                    "error": f"Literature search with Perplexity is currently unavailable: {e}",
                }
        else:
            result = await self._search_europepmc_literature(
                query, max_results, include_preprints, date_range
            )

        # the caller-supplied `backend` argument may have been overridden upstream, so the
        # response states which API was actually queried rather than which one was asked for
        result["backend"] = selected_backend
        return result

    async def _search_europepmc_literature(
        self,
        query: str,
        max_results: int,
        include_preprints: bool,
        date_range: str | None,
    ) -> dict[str, Any]:
        """Search scientific literature via Europe PMC."""
        epmc_query = query

        if not include_preprints:
            epmc_query += " (SRC:MED OR SRC:PMC)"

        if date_range:
            epmc_query += self._build_date_filter(date_range)

        url = (
            f"https://www.ebi.ac.uk/europepmc/webservices/rest/search"
            f"?query={quote(epmc_query)}"
            f"&format=json"
            f"&pageSize={min(max_results, 25)}"
            f"&resultType=core"
        )

        try:
            resp = await self.external_client.get(url, timeout=15.0)
            if resp.status_code == 200:
                data = resp.json()
                results = self._format_literature_results(
                    data.get("resultList", {}).get("result", [])
                )
                return {
                    "success": True,
                    "query": query,
                    "total_found": data.get("hitCount", 0),
                    "returned": len(results),
                    "results": results,
                    "source": "europepmc",
                }
            return {
                "success": False,
                "error": f"Europe PMC error: HTTP {resp.status_code}",
            }
        except Exception as e:
            logger.error(f"Europe PMC search error: {e}")
            return {"success": False, "error": f"Literature search failed: {str(e)}"}

    async def _search_perplexity_literature(
        self,
        query: str,
        max_results: int,
        api_key: str,
        include_preprints: bool,
        date_range: str | None,
    ) -> dict[str, Any]:
        """Search scientific literature using Perplexity Sonar API."""
        enhanced_query = f"Find scientific research papers about: {query}"

        # domain filter for scientific sources
        domains = [
            "pubmed.ncbi.nlm.nih.gov",
            "ncbi.nlm.nih.gov",
            "doi.org",
            "nature.com",
            "science.org",
            "cell.com",
            "nejm.org",
            "thelancet.com",
            "pnas.org",
            "jci.org",
        ]
        if include_preprints:
            domains.extend(["biorxiv.org", "medrxiv.org"])

        web_search_options: dict[str, Any] = {
            "search_domain_filter": domains,
        }

        # date filtering
        if date_range == "last_year":
            web_search_options["search_recency_filter"] = "year"
        elif date_range == "last_5_years":
            # perplexity doesn't have exact 5-year filter, skip
            pass

        payload = {
            "model": "sonar",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a scientific literature search assistant. "
                        "Return information about relevant research papers including "
                        "title, authors, journal, year, and key findings. "
                        "Include DOI or PMID when available."
                    ),
                },
                {"role": "user", "content": enhanced_query},
            ],
            "web_search_options": web_search_options,
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        resp = await self.external_client.post(
            "https://api.perplexity.ai/chat/completions",
            json=payload,
            headers=headers,
            timeout=30.0,
        )

        if resp.status_code == 200:
            data = resp.json()
            formatted = self._format_perplexity_literature_results(data, query, max_results)
            await self._hydrate_literature_metadata(formatted["results"])
            return formatted

        raise Exception(f"Perplexity API error: HTTP {resp.status_code}")

    def _format_perplexity_literature_results(
        self,
        data: dict,
        query: str,
        max_results: int,
    ) -> dict[str, Any]:
        """Format Perplexity response to match Europe PMC output structure."""
        import re

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        search_results = data.get("search_results") or []
        citations = data.get("citations") or []
        # search_results carries title/date/snippet per hit; citations is a bare URL list
        entries = search_results or [{"url": url} for url in citations]

        results = []
        for entry in entries[:max_results]:
            url = entry.get("url") or ""

            # extract DOI/PMID/PMCID from URL if possible
            doi = None
            pmid = None
            pmcid = None
            if "doi.org/" in url:
                doi = url.split("doi.org/")[-1]
            if "pubmed.ncbi.nlm.nih.gov/" in url:
                match = re.search(r"/(\d+)", url)
                if match:
                    pmid = match.group(1)
            match = re.search(r"(PMC\d+)", url)
            if match:
                pmcid = match.group(1)

            is_preprint = "biorxiv.org" in url or "medrxiv.org" in url
            date = entry.get("date") or ""

            results.append({
                "title": entry.get("title") or "",
                "authors": "",
                "journal": "",
                "year": date[:4],
                "abstract": entry.get("snippet") or "",
                "doi": doi,
                "pmid": pmid,
                "pmcid": pmcid,
                "source": "perplexity",
                "metadata_source": "perplexity",
                "is_preprint": is_preprint,
                "url": url,
            })

        return {
            "success": True,
            "query": query,
            "total_found": len(entries),
            "returned": len(results),
            "results": results,
            "summary": content,
            "source": "perplexity",
        }

    async def _hydrate_literature_metadata(self, results: list[dict]) -> None:
        """Fill in authors/journal/title for Perplexity hits that carry a PMID, DOI or PMCID.

        Perplexity returns no bibliographic metadata beyond a title, so records are looked up
        in Europe PMC in one batched query. Best-effort: the hits stay usable if it fails.
        """
        clauses = []
        for record in results:
            if record.get("pmid"):
                clauses.append(f"EXT_ID:{record['pmid']}")
            elif record.get("doi"):
                clauses.append(f'DOI:"{record["doi"]}"')
            elif record.get("pmcid"):
                clauses.append(f"PMCID:{record['pmcid']}")

        if not clauses:
            return

        url = (
            f"https://www.ebi.ac.uk/europepmc/webservices/rest/search"
            f"?query={quote(' OR '.join(clauses))}"
            f"&format=json"
            f"&pageSize={len(clauses)}"
            f"&resultType=core"
        )

        try:
            resp = await self.external_client.get(url, timeout=15.0)
            if resp.status_code != 200:
                logger.warning(
                    f"Literature metadata hydration skipped: Europe PMC HTTP {resp.status_code}"
                )
                return
            records = resp.json().get("resultList", {}).get("result", [])
        except Exception as e:
            logger.warning(f"Literature metadata hydration failed: {e}")
            return

        by_id: dict[str, dict] = {}
        for raw, paper in zip(records, self._format_literature_results(records)):
            for key in (raw.get("pmid"), (raw.get("doi") or "").lower(), raw.get("pmcid")):
                if key:
                    by_id[key] = paper

        for record in results:
            match = (
                by_id.get(record.get("pmid") or "")
                or by_id.get((record.get("doi") or "").lower())
                or by_id.get(record.get("pmcid") or "")
            )
            if not match:
                continue
            for field in ("title", "authors", "journal", "year", "abstract"):
                if match.get(field):
                    record[field] = match[field]
            record["doi"] = record.get("doi") or match.get("doi")
            record["pmid"] = record.get("pmid") or match.get("pmid")
            record["metadata_source"] = "europepmc"

    async def web_search(
        self,
        query: str,
        max_results: int = 5,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
    ) -> dict[str, Any]:
        """Search web using Tavily (if configured) or DuckDuckGo."""
        tavily_api_key = os.environ.get("TAVILY_API_KEY")
        if tavily_api_key:
            try:
                return await self._search_tavily(
                    query,
                    min(max_results, 10),
                    tavily_api_key,
                    include_domains,
                    exclude_domains,
                )
            except Exception as e:
                logger.warning(f"Tavily search failed, falling back to DuckDuckGo: {e}")

        return await self._search_duckduckgo(query, min(max_results, 10))

    def _build_date_filter(self, date_range: str) -> str:
        """Build Europe PMC date filter clause."""
        from datetime import datetime, timedelta

        today = datetime.now()

        if date_range == "last_year":
            start = (today - timedelta(days=365)).strftime("%Y-%m-%d")
            return f" AND (FIRST_PDATE:[{start} TO {today.strftime('%Y-%m-%d')}])"
        elif date_range == "last_5_years":
            start = (today - timedelta(days=5 * 365)).strftime("%Y-%m-%d")
            return f" AND (FIRST_PDATE:[{start} TO {today.strftime('%Y-%m-%d')}])"
        elif "-" in date_range:
            years = date_range.split("-")
            if len(years) == 2:
                return f" AND (PUB_YEAR:[{years[0]} TO {years[1]}])"
        return ""

    def _format_literature_results(self, results: list) -> list:
        """Format Europe PMC results for LLM consumption."""
        import html
        import re

        def strip_html(text: str) -> str:
            if not text:
                return ""
            text = html.unescape(text)
            return re.sub(r"<[^>]+>", "", text)

        formatted = []
        for paper in results:
            pmid = paper.get("pmid")
            doi = paper.get("doi")
            source = paper.get("source", "")

            if pmid:
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            elif doi:
                url = f"https://doi.org/{doi}"
            else:
                url = None

            formatted.append(
                {
                    "title": strip_html(paper.get("title", "")),
                    "authors": paper.get("authorString", ""),
                    "journal": paper.get("journalTitle", "")
                    or paper.get("bookOrReportDetails", {}).get("publisher", ""),
                    "year": paper.get("pubYear", ""),
                    "abstract": strip_html(paper.get("abstractText", "") or "")[:1500],
                    "doi": doi,
                    "pmid": pmid,
                    "source": source,
                    "is_preprint": source == "PPR",
                    "url": url,
                }
            )
        return formatted

    async def _search_tavily(
        self,
        query: str,
        max_results: int,
        api_key: str,
        include_domains: list[str] | None,
        exclude_domains: list[str] | None,
    ) -> dict[str, Any]:
        """Search using Tavily API."""
        payload: dict[str, Any] = {
            "api_key": api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": max_results,
            "include_answer": True,
        }

        if include_domains:
            payload["include_domains"] = include_domains
        if exclude_domains:
            payload["exclude_domains"] = exclude_domains

        resp = await self.external_client.post(
            "https://api.tavily.com/search", json=payload, timeout=20.0
        )

        if resp.status_code == 200:
            data = resp.json()
            results = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": (r.get("content", "") or "")[:500],
                    "score": r.get("score", 0),
                }
                for r in data.get("results", [])
            ]
            return {
                "success": True,
                "query": query,
                "source": "tavily",
                "answer": data.get("answer"),
                "results": results,
            }

        raise Exception(f"Tavily API error: HTTP {resp.status_code}")

    async def _search_duckduckgo(self, query: str, max_results: int) -> dict[str, Any]:
        """Search using DuckDuckGo (free fallback)."""
        import asyncio

        try:
            from ddgs import DDGS
        except ModuleNotFoundError as exc:
            if not _pruned_module(exc, "ddgs"):
                raise
            logger.error("web search is unavailable: ddgs is not installed in this environment")
            return {
                "success": False,
                "query": query,
                "source": "duckduckgo",
                "error": _capability_absent_message("Web search", "ddgs"),
                "error_type": CAPABILITY_UNAVAILABLE,
            }

        def sync_search():
            return DDGS().text(query, max_results=max_results)

        try:
            results = await asyncio.to_thread(sync_search)
            formatted = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "content": (r.get("body", "") or "")[:500],
                }
                for r in results
            ]
            return {
                "success": True,
                "query": query,
                "source": "duckduckgo",
                "results": formatted,
            }
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return {"success": False, "error": f"Web search failed: {str(e)}"}

    # THE CAP IS THE TRANSPORT'S, NOT THIS LAYER'S (genetics-results-suite-4h6.52). It used
    # to be 4 MiB, chosen for a local read that no longer exists: every read now goes over
    # `GET /artifact`, and the supervisor answers 413 above ARTIFACT_READ_MAX_BYTES, so a
    # larger number here would only promise the model bytes the sandbox will refuse. This is
    # a REDUCTION in what `read_artifact` can return — 4 MiB -> 512 KiB — and it is the same
    # ceiling `_fetch_analysis_images` has always skipped oversize images against. A truncated
    # PNG is garbage rather than a short answer, so an oversized binary is refused, not cut.
    _MAX_ARTIFACT_BYTES = ARTIFACT_READ_MAX_BYTES

    # ...but text is still truncated, and the two bounds are not the same bound. The byte cap
    # is about the wire; this one is about the MODEL'S CONTEXT, which no transport limit
    # protects — 512 KiB of TSV is well over 100k tokens dropped into one tool result. It
    # survived the move for that reason, and it is the only cap of the two that a caller can
    # act on: `truncated: true` tells the model to have the script summarise instead.
    _MAX_ARTIFACT_TEXT_CHARS = 100_000

    async def read_artifact(
        self,
        name: str,
        *,
        user: str | None = None,
        session_id: str | None = None,
        gateway_asserted: bool = False,
    ) -> dict[str, Any]:
        """Read one artifact of THIS USER'S CHAT SESSION'S recent runs, over HTTP from the sandbox.

        `name` is a bare file name, never a path and never an execution id: the model learns
        names from a run's artifact manifest, and nothing else is addressable. `user` and
        `session_id` are supplied by the CALLER (llm_service injects the authenticated pair,
        and strips any same-named key the model emitted); the declared schema has one
        parameter, `name`.

        RESOLUTION IS SERVER-SIDE AND IS THE AUTHORIZATION. The name is resolved against
        `_ARTIFACT_MANIFESTS`, which `run_analysis` populates with what each execution of this
        `(sub, sid)` reported producing. An artifact belonging to another user or another
        session resolves to nothing and returns the SAME "not found" as a name that never
        existed — the three are deliberately indistinguishable, because knowing that a name
        exists somewhere is already a cross-session fact. Within one key, a name that two
        executions both produced resolves to the most recently completed one still inside the
        retention window (`_ARTIFACT_MANIFESTS.resolve`).

        THE USER TERM IS NOT DECORATION (genetics-results-suite-dh3). `session_id` is
        client-supplied and unvalidated, so it authorizes nothing on its own; `user` is the
        subject `get_authenticated_user` produced, the same value that becomes the `sub` of
        the per-execution credential. A read presenting somebody else's session id lands on a
        key that does not exist.

        AND `user` IS ONLY UNFORGEABLE BECAUSE OF THE GATE BELOW, which is why this tool has
        the same one `run_analysis` does. `get_authenticated_user` honours
        `X-Goog-Authenticated-User-Email` from any caller presenting INTERNAL_API_SECRET —
        mcp-server and results-api hold it by design and the NetworkPolicy admits mcp-server
        to chat-backend:8000 — so without `gateway_asserted` a marker holder could name the
        victim as `user`, supply the victim's `sid`, and the (sub, sid) key would resolve. The
        write path was gated and the read path was not, which made keying on `user` a
        statement about a value the attacker chose. `gateway_asserted` is the auth-gateway
        provenance assertion (genetics-results-suite-4h6.84); it defaults False so a caller
        that states no provenance fails closed. Nothing legitimate loses access: an artifact
        only exists to be read because `run_analysis` recorded it, and that dispatch already
        refused every caller this gate refuses.

        NO LOCAL FILESYSTEM READ HAPPENS HERE, and that is the point of the change rather
        than an implementation detail. This process is chat-backend, whose `/data` PVC holds
        `chat_history.db` and `llm_config.db`; the descriptor-based checks that used to guard
        a local read (`O_NOFOLLOW` on the directory and on the file, `/proc/self/fd`
        verification, `S_ISREG`, `st_nlink == 1`) now run INSIDE THE SANDBOX in
        `read_artifact_bytes`, against `/scratch/<id>/artifacts` — where the hostile party
        actually is. `SANDBOX_ARTIFACTS_DIR` gains no reader here and neither does
        `SUBAGENT_ALLOWED_PATHS`.
        """
        if not gateway_asserted and getattr(_resolve_settings(), "require_auth", True):
            # THE SAME GATE `run_analysis` HAS, for the same reason and in the same shape
            # (genetics-results-suite-4h6.84, dh3). A holder of INTERNAL_API_SECRET can send
            #     X-Internal-Auth: <secret>  +  X-Goog-Authenticated-User-Email: victim@…
            # and arrive with `user` set to anybody; only `X-Gateway-Auth` separates a real
            # browser session from that, and auth-gateway is the only other holder of it.
            # Gated on require_auth exactly as the dispatch is: REQUIRE_AUTH=false is local
            # dev, where there is no gateway to assert anything.
            #
            # Refused BEFORE the name is even examined, and with the operator error rather
            # than `not_found`: this is a property of the caller, not of the name, so it
            # leaks nothing about what exists — the answer is identical for every `name`.
            logger.error(
                "read_artifact refused for an identity the gateway did not assert "
                "(session=%s): the caller presented the internal marker without the "
                "auth-gateway provenance secret",
                session_id,
            )
            return self._sandbox_operator_error(
                "Reading analysis artifacts requires an authenticated user session and is "
                "not available to service callers."
            )

        if not isinstance(name, str) or not name.strip():
            return {"success": False, "error": "An artifact name is required."}
        name = name.strip()
        if (
            name in (".", "..")
            or "/" in name
            or "\\" in name
            or "\x00" in name
            or os.path.isabs(name)
            or Path(name).name != name
        ):
            return {
                "success": False,
                "error": (
                    f"Invalid artifact name '{name}': pass the bare file name from the "
                    f"run's artifact manifest, not a path."
                ),
            }

        not_found = {
            "success": False,
            "error": (
                f"Artifact not found: {name}. Artifacts are readable only from analyses run "
                f"in this conversation, and only for about {ARTIFACT_RETENTION_S // 60} "
                f"minutes after the run finishes. Re-run the script if you need it again."
            ),
            "error_type": "ArtifactNotFound",
            "retryable": False,
        }

        if not _is_identity(user) or not _is_identity(session_id):
            # a wiring fault, not a model error: nothing dispatches this tool without an
            # authenticated user and a session today (it is excluded from MCP and from every
            # subagent skill), so reaching here means a new caller forgot the injection.
            # Refused rather than resolved on whichever half arrived — half a key is not a
            # weaker authorization, it is none.
            #
            # `isinstance` and not just truthiness so the guard is TOTAL: a non-str identity
            # (a list, a dict) is unhashable and would raise out of the tuple lookup instead
            # of failing closed, giving a caller a response shape distinct from
            # ArtifactNotFound. `get_authenticated_user` returns `str | None` today, so this
            # is unreachable — a security guard that is total by construction is worth more
            # than one that is total by the current type of its argument.
            logger.error(
                "read_artifact called without an authenticated identity (user=%s session=%s); "
                "refusing the read",
                bool(user),
                bool(session_id),
            )
            return not_found

        execution_id = _ARTIFACT_MANIFESTS.resolve(user, session_id, name)
        if execution_id is None:
            return not_found

        sandbox, not_configured = self._sandbox_or_operator_error("no artifact can be read")
        if not_configured is not None:
            return not_configured

        result = await sandbox.get_artifact(execution_id, name)
        if not result.ok or result.data is None:
            return self._artifact_error(name, result)

        raw = result.data
        content_type = result.content_type or (
            mimetypes.guess_type(name)[0] or "application/octet-stream"
        )
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return {
                "success": True,
                "name": name,
                "size": len(raw),
                "content_type": content_type,
                "encoding": "base64",
                "content": result.content_base64,
            }

        truncated = len(text) > self._MAX_ARTIFACT_TEXT_CHARS
        return {
            "success": True,
            "name": name,
            "size": len(raw),
            "content_type": content_type,
            "encoding": "utf-8",
            "content": text[: self._MAX_ARTIFACT_TEXT_CHARS],
            "truncated": truncated,
        }

    def _artifact_error(self, name: str, result: Any) -> dict[str, Any]:
        """Turn a failed `GET /artifact` into something the model can act on.

        TWO OF THESE STATUSES ARE NEW TO THIS TOOL and neither may arrive as a generic
        failure. `409 ArtifactModified` is the supervisor refusing to serve bytes that no
        longer match the manifest it advertised — a same-uid peer rewrote or replaced the file
        during retention (`4h6.82`/`4h6.88`). It is NOT a transient error and re-asking cannot
        fix it, so it is reported as non-retryable with the one repair that works: run the
        analysis again. `413` is now reachable at 512 KiB rather than 4 MiB, and its answer is
        for the script to write a smaller summary, not for the model to retry the read.

        `retryable` MEANS "A SECOND ASK COULD PLAUSIBLY SUCCEED", and only a transport-level
        fault qualifies. A malformed 200 body and an execution_id the client itself refused are
        both DETERMINISTIC — the same body comes back, or the same id is refused again with no
        request issued — so they are non-retryable even though both look like server-side
        faults. There is no total budget above this tool, unlike `run_analysis`'s 300s
        `wait_for`, so a retryable answer here is retried on the model's judgement alone.
        """
        try:
            from genetics_mcp_server.sandbox_client import (
                ERROR_BAD_EXECUTION_ID,
                ERROR_MALFORMED_RESPONSE,
            )
        except ModuleNotFoundError as exc:
            if not _pruned_module(exc, "genetics_mcp_server.sandbox_client"):
                raise
            return self._capability_unavailable_error(
                "Artifact reading", "genetics_mcp_server.sandbox_client"
            )

        status = getattr(result, "status_code", None)
        error_type = getattr(result, "error_type", None)
        if status == 409 or error_type == "ArtifactModified":
            logger.error("sandbox refused artifact %r as modified", name)
            return {
                "success": False,
                "error": (
                    f"Artifact '{name}' could not be served because its contents no longer "
                    f"match what the run reported writing, so the sandbox refuses it. Do not "
                    f"retry the read — re-run the analysis to produce the file again, and "
                    f"tell the user the earlier output could not be trusted."
                ),
                "error_type": "ArtifactModified",
                "retryable": False,
            }
        if status == 413 or error_type == "ArtifactTooLarge":
            return {
                "success": False,
                "error": (
                    f"Artifact '{name}' is over the {self._MAX_ARTIFACT_BYTES} byte read "
                    f"limit. Re-run the analysis and have the script print or write a smaller "
                    f"summary instead."
                ),
                "error_type": "ArtifactTooLarge",
                "retryable": False,
            }
        if status in (404, 400) or error_type == ERROR_BAD_EXECUTION_ID:
            # 400 means the supervisor rejected the execution_id WE resolved, which is our
            # bug, not the model's; it gets the same answer as a missing name rather than a
            # description of our internals. ERROR_BAD_EXECUTION_ID is the same fault caught one
            # hop earlier — the client's own pre-flight refused the id and issued NO request —
            # so it belongs here rather than in the retryable fallback below: there is no
            # request to repeat and re-asking would re-reject the identical id
            # (genetics-results-suite-4h6.52).
            return {
                "success": False,
                "error": (
                    f"Artifact not found: {name}. It may already have passed the "
                    f"{ARTIFACT_RETENTION_S // 60}-minute retention window."
                ),
                "error_type": "ArtifactNotFound",
                "retryable": False,
            }
        if error_type == ERROR_MALFORMED_RESPONSE:
            # A 200 WITH AN UNUSABLE BODY IS DETERMINISTIC, not transient: the supervisor
            # answered, and the same execution_id and name will produce the same body. Marking
            # it retryable spent model roundtrips on a re-ask that cannot succeed, and this
            # tool has no total budget above it the way run_analysis has its 300s wait_for
            # (genetics-results-suite-4h6.52).
            logger.error("artifact %r came back as a malformed 200 body", name)
            return {
                "success": False,
                "error": (
                    f"Artifact '{name}' could not be read: the analysis sandbox answered with "
                    f"a response this server could not parse. Do not retry the read — it will "
                    f"return the same thing. Re-run the analysis if you need the file."
                ),
                "error_type": "ArtifactUnavailable",
                "retryable": False,
            }
        logger.warning(
            "artifact %r could not be read: status=%s type=%s", name, status, error_type
        )
        return {
            "success": False,
            "error": (
                f"Artifact '{name}' could not be read from the analysis sandbox. This is a "
                f"server-side fault rather than a problem with the name."
            ),
            "error_type": "ArtifactUnavailable",
            "retryable": True,
        }

    # The whole chat turn's budget for one run_analysis call, across every retry the client
    # makes. The client bounds each ATTEMPT correctly and deliberately offers no total: its
    # per-attempt read deadline is derived from the supervisor's own worst-case hold time
    # (120s queued + timeout_s + 15s margin) and must not be shortened. But the attempts sum:
    # connect 5 + write 10 + read 255 + a 60s Retry-After + a second 270 is ~585s, and ten
    # minutes inside a single tool call is not a chat turn. This layer owns that budget, so
    # the cap is here.
    #
    # 300 is the smallest value that never truncates a legitimate single attempt: at the
    # maximum timeout_s of 120 one attempt is at most 5 + 10 + 255 = 270s. At the default
    # timeout_s of 60 an attempt is at most 210s, so 300 still leaves room for a 429's
    # Retry-After wait and a partial retry. Raising timeout_s therefore trades away the
    # retry, which is the right way round: a script that asked for the full 120s has already
    # been promised most of the turn.
    _RUN_ANALYSIS_DEADLINE_S = 300

    # error types that say the script called the SDK wrong rather than that the data was
    # wrong. Advisory only — `error.type` is an OPEN string (the child's own exception class
    # name), so this is used to ADD a hint, never to decide whether something is an error.
    _SDK_MISUSE_ERROR_TYPES = frozenset(
        {"TypeError", "AttributeError", "NameError", "ImportError", "ModuleNotFoundError"}
    )

    @cached_property
    def _sandbox(self) -> Any:
        """The sandbox transport, imported lazily and built once.

        Deferred import for the same reason `read_artifact` defers its own: this module is
        imported by the standalone MCP server, and the sandbox client has no business in
        that import graph. Tests replace this by assigning to the attribute.
        """
        try:
            from genetics_mcp_server.sandbox_client import SandboxClient
        except ModuleNotFoundError as exc:
            if not _pruned_module(exc, "genetics_mcp_server.sandbox_client"):
                raise
            # RAISED, not shaped: this is a property whose contract is "the transport".
            # Every entry point resolves it through `_sandbox_or_operator_error`, which
            # guards the same import and answers with a shaped error first, so the only
            # way here is a direct attribute access that has nothing to return.
            raise RuntimeError(
                _capability_absent_message("Code execution", "genetics_mcp_server.sandbox_client")
            ) from exc

        return SandboxClient()

    def _sandbox_or_operator_error(
        self, consequence: str
    ) -> tuple[Any, dict[str, Any] | None]:
        """The transport, or — with nothing configured — the shaped error to return in its place.

        THE ONE PLACE `SandboxNotConfigured` STOPS BEING AN EXCEPTION. `_sandbox` is a
        cached_property, and its constructor now raises when SANDBOX_URL is unset
        (genetics-results-suite-6um), so "no address" surfaces at a bare ATTRIBUTE ACCESS
        inside methods that otherwise only ever RETURN a shaped failure. Every entry point
        resolves the transport through here, so a new one fails in the house style by
        construction rather than by remembering a handler of its own.

        Caught by name rather than left to `except SandboxError`: it is in that family so
        nothing in the request flow escapes the family clause, but the family's fallback
        reports `retryable: True`, and no second ask can supply a missing address.
        """
        try:
            from genetics_mcp_server.sandbox_client import SandboxNotConfigured
        except ModuleNotFoundError as exc:
            if not _pruned_module(exc, "genetics_mcp_server.sandbox_client"):
                raise
            return None, self._capability_unavailable_error(
                "Code execution", "genetics_mcp_server.sandbox_client"
            )

        try:
            return self._sandbox, None
        except SandboxNotConfigured as e:
            logger.error("no sandbox address is configured: %s", e)
            return None, self._sandbox_operator_error(
                f"Code execution is not configured on this server, so {consequence}. "
                "This is a server configuration fault and will not be fixed by retrying."
            )

    async def run_analysis(
        self,
        code: str,
        timeout_s: int | None = None,
        *,
        user: str | None = None,
        session_id: str | None = None,
        gateway_asserted: bool = False,
    ) -> dict[str, Any]:
        """Run one script in the sandbox and render the supervisor's result for the model.

        `user` and `session_id` are supplied by the CALLER, never by the model: they are the
        subject and the session of the per-execution credential, and llm_service strips any
        same-named key the model emits before injecting the authenticated pair. A tool
        invocation with neither is a wiring fault, not a script fault, and is reported as one.

        `gateway_asserted` says WHERE `user` came from — auth-gateway having verified an
        oauth2-proxy session, or some other holder of INTERNAL_API_SECRET simply asserting an
        address. It defaults to False so that a caller which does not state a provenance is
        refused rather than trusted (genetics-results-suite-4h6.84).

        **There is deliberately no `except Exception` in this method**, which is a departure
        from the ~40 handlers above it. `mint_execution_tokens` raises `SandboxTokenUnavailable`
        — a plain `RuntimeError` — when `SANDBOX_TOKEN_SIGNING_KEY` is unset, and the house
        style would catch it and hand the model an ordinary "tool failed". That is not a
        security hole (the client raises before any request, so no credential is ever sent)
        but it converts an OPERATOR-VISIBLE misconfiguration into a MODEL-VISIBLE failure the
        model then retries, forever, against a sandbox that cannot work. It is caught here
        first and by name, and reported with `retryable: False`; anything genuinely unforeseen
        propagates rather than being flattened into that same shape.

        The 200 body is rendered field by field rather than passed through. Two reasons, and
        neither is that the contract's field set is closed — it is not, and an unknown
        `status` or `error.type` renders as itself instead of crashing. First, `execution_id`
        must not reach the model: it is the join key for the audit trail and for the manifest
        chat-backend records against the `jti`/`sid`, and putting it in context invites a
        model-supplied one back in, which is exactly what artifact resolution rules out.
        Second, an artifact entry is `name`/`size`/`content_type` and nothing else — no path,
        no id, no URL — so it is rebuilt to that shape rather than forwarded.
        """
        # auth.core is deferred for the same reason the sandbox imports are: it pulls
        # FastAPI in, and this module is imported by the standalone MCP server, whose
        # import graph has no business with either
        try:
            from genetics_mcp_server.auth.core import SERVICE_IDENTITY
            from genetics_mcp_server.sandbox_client import (
                MAX_TIMEOUT_S,
                SandboxBusy,
                SandboxDeadlineExceeded,
                SandboxError,
                SandboxRejected,
                SandboxUnavailable,
            )
            from genetics_mcp_server.sandbox_token import SandboxTokenUnavailable
        except ModuleNotFoundError as exc:
            missing = _pruned_module(
                exc,
                "genetics_mcp_server.auth",
                "genetics_mcp_server.auth.core",
                "genetics_mcp_server.sandbox_client",
                "genetics_mcp_server.sandbox_token",
            )
            if not missing:
                raise
            # a script calling run_analysis from INSIDE the sandbox: nesting an execution
            # is not a capability the image carries, and the model must be able to tell
            # that from "the sandbox is down", which is retryable and this is not
            return self._capability_unavailable_error("Code execution", missing)

        if not isinstance(code, str) or not code.strip():
            # error_type is not decoration: without it this shape reaches the `script_result`
            # SSE chunk as a bare "unknown", indistinguishable from a transport fault, and a
            # benchmark then books the model emitting no code as the sandbox being flaky
            return {
                "success": False,
                "error": "A non-empty Python script is required.",
                "error_type": "EmptyScript",
                "retryable": True,
            }
        if not user or not session_id:
            # the identity is the credential's subject; without it there is nothing to mint
            # against and the fault is in the wiring, not in anything the model can change
            logger.error(
                "run_analysis called without an authenticated identity "
                "(user=%s session=%s); the caller must supply both",
                bool(user),
                bool(session_id),
            )
            return self._sandbox_operator_error(
                "Code execution is not available in this context: no authenticated session."
            )
        if user == SERVICE_IDENTITY:
            # THE MCP EXCLUSION BOUNDARY, enforced here rather than at the HTTP route
            # (genetics-results-suite-4h6.27). The NetworkPolicy closes mcp-server -> sandbox
            # but not mcp-server -> chat-backend -> sandbox: mcp-server holds
            # INTERNAL_API_SECRET and is admitted to chat-backend:8000, and a valid marker
            # with no identity header resolves to exactly this one service string
            # (genetics-results-suite-th2). So "authenticated caller" is not the property
            # this dispatch needs — a real person is, because `user` becomes the `sub` of
            # both per-execution JWTs, the artifact retention scope and every audit record,
            # and a service marker names nobody to attribute or revoke.
            #
            # At the dispatch and not at the route because THIS is the narrow waist every
            # sandbox execution passes through — the streaming chat path, subagent dispatch
            # and any future caller — and because it sits immediately
            # before mint_execution_tokens, so no credential can be minted for a subject
            # that was refused. A route-level check would guard only the routes someone
            # remembered to decorate and would also refuse chat itself, which the marker
            # identity is legitimately allowed to use.
            logger.error(
                "run_analysis refused for the %s service identity (session=%s): code "
                "execution requires an authenticated user",
                SERVICE_IDENTITY,
                session_id,
            )
            return self._sandbox_operator_error(
                "Code execution requires an authenticated user session and is not available "
                "to service callers."
            )
        if not gateway_asserted and getattr(_resolve_settings(), "require_auth", True):
            # THE RESIDUAL THE GUARD ABOVE DOES NOT COVER (genetics-results-suite-4h6.84).
            # `user == SERVICE_IDENTITY` catches the marker-ALONE caller — auth_required's
            # case 3. It does not catch case 1: marker PLUS an identity header wins over
            # case 3, so any holder of INTERNAL_API_SECRET (mcp-server included, and the
            # NetworkPolicy admits it to chat-backend:8000) could send
            #     X-Internal-Auth: <secret>  +  X-Goog-Authenticated-User-Email: anyone@…
            # and arrive here as `anyone@…`, indistinguishable from a browser session. Both
            # per-execution JWTs would then carry that `sub`, and `session_id` is
            # client-supplied, so the artifact scope AND the audit trail would name a person
            # who never made the request.
            #
            # What separates the two is a SECOND SECRET the other holders do not have:
            # auth-gateway sends `X-Gateway-Auth: <GATEWAY_IDENTITY_SECRET>` on the two
            # locations that proxy here, after an `auth_request /oauth2/auth`, and that key
            # is mounted only into auth-gateway and chat-backend.
            # `auth.dependencies.gateway_asserted_identity` reduces it to this bool, and the
            # chat route passes it down. Defaulting it False makes a caller that never states
            # a provenance fail closed rather than inherit trust.
            #
            # NOT a check on the transport. The first draft of this gate demanded that the
            # marker arrive in `X-Internal-Auth` rather than in `Authorization: Bearer`, and
            # that was measurably bypassable: mcp-server and results-api hold
            # INTERNAL_API_SECRET by design and can copy it into any header they choose, so
            # the gate asked them to not rename a header. A header name is not a secret.
            #
            # Bound, stated so nobody reads it as more than it is: a compromised auth-gateway
            # — or GATEWAY_IDENTITY_SECRET leaked to another pod, which can then reach
            # chat-backend:8000 directly — still reaches this dispatch, and the identity it
            # asserts is still only allow-list-checked. It closes the transitive
            # mcp-server -> chat-backend -> sandbox path, which is what 4h6.84 is about; it
            # is not authentication.
            #
            # Gated on require_auth for the same reason auth_required's first branch is:
            # REQUIRE_AUTH=false means there is no oauth2-proxy and no gateway to assert
            # anything, so this could never be true in local dev. Production sets it true.
            logger.error(
                "run_analysis refused for an identity the gateway did not assert "
                "(session=%s): the caller presented the internal marker without the "
                "auth-gateway provenance secret",
                session_id,
            )
            return self._sandbox_operator_error(
                "Code execution requires an authenticated user session and is not available "
                "to service callers."
            )

        sandbox, not_configured = self._sandbox_or_operator_error("no script can run")
        if not_configured is not None:
            return not_configured

        try:
            result = await asyncio.wait_for(
                sandbox.execute(
                    code=code,
                    user=user,
                    session_id=session_id,
                    timeout_s=timeout_s,
                ),
                timeout=self._RUN_ANALYSIS_DEADLINE_S,
            )
        except SandboxTokenUnavailable as e:
            # FIRST, and by name. See the docstring: this is the one failure that must not
            # read to the model as a retryable tool error.
            logger.error("run_analysis cannot mint execution tokens: %s", e)
            return self._sandbox_operator_error(
                "Code execution is not configured on this server, so no script can run. "
                "This is a server configuration fault and will not be fixed by retrying."
            )
        except asyncio.TimeoutError:
            # our own cap, not the client's per-attempt one: the sandbox may still be running
            # or queueing this script. Deliberately not framed as a script failure.
            logger.warning(
                "run_analysis exceeded the %ss turn budget (session=%s)",
                self._RUN_ANALYSIS_DEADLINE_S,
                session_id,
            )
            return {
                "success": False,
                "error": (
                    f"The analysis did not finish within {self._RUN_ANALYSIS_DEADLINE_S}s of "
                    "this turn's budget, which includes time spent queued behind another run. "
                    "The script itself may still be running. Try again with a smaller job."
                ),
                "error_type": "TurnBudgetExceeded",
                "retryable": True,
            }
        except SandboxUnavailable as e:
            # NOT a script failure, and it must never be worded as one: `strategy: Recreate`
            # plus terminationGracePeriodSeconds 130 means a deploy landing on an in-flight
            # execution leaves no sandbox at all for up to ~130s.
            logger.warning("sandbox unavailable for run_analysis: %s", e)
            return {
                "success": False,
                "error": (
                    "The analysis sandbox is temporarily unavailable — this is not a problem "
                    "with the script. It is usually a restart and clears within a couple of "
                    "minutes; wait and try the same script again, or answer from other tools."
                ),
                "error_type": "SandboxUnavailable",
                "retryable": True,
            }
        except SandboxBusy as e:
            return {
                "success": False,
                "error": (
                    "The analysis sandbox is busy with other runs and could not take this one. "
                    "Wait a moment and retry the same script, or use other tools."
                ),
                "error_type": "SandboxBusy",
                "retry_after_s": e.retry_after,
                "retryable": True,
            }
        except SandboxDeadlineExceeded as e:
            logger.warning("sandbox did not answer run_analysis: %s", e)
            return {
                "success": False,
                "error": (
                    "The analysis sandbox accepted the script but never answered. The script "
                    "may still be running; do not assume it failed."
                ),
                "error_type": "SandboxDeadlineExceeded",
                "retryable": True,
            }
        except SandboxRejected as e:
            # a caller bug — including a timeout_s or a script size this client refused to
            # send. Actionable, because the model chose the value.
            logger.warning("sandbox refused run_analysis: %s", e)
            return {
                "success": False,
                "error": (
                    f"The analysis request was rejected: {e}. Fix the request rather than "
                    f"repeating it — timeout_s must be 1-{MAX_TIMEOUT_S} seconds."
                ),
                "error_type": "SandboxRejected",
                "retryable": False,
            }
        except SandboxError as e:
            # SandboxInternalError, SandboxProtocolError, and any subclass added later. An
            # unrecognised member of the family is reported, never dropped.
            logger.error("sandbox error in run_analysis: %s", e)
            return {
                "success": False,
                "error": (
                    "The analysis sandbox failed to run the script. This is a server-side "
                    "fault rather than an error in the script."
                ),
                "error_type": type(e).__name__,
                "retryable": True,
            }

        self._record_artifact_manifest(result, user, session_id)
        return self._render_analysis(result, images=await self._fetch_analysis_images(result))

    @staticmethod
    def _record_artifact_manifest(
        result: dict[str, Any], user: str | None, session_id: str | None
    ) -> None:
        """Bind what this execution wrote to the user and session that ran it, for `read_artifact`.

        THE ONLY PLACE THE (sub, sid, execution_id) TRIPLE EXISTS. `execution_id` is minted per
        call and deliberately never shown to the model, so this is the last point at which the
        parts are all in scope; without the record there is nothing for a name to resolve
        against and `read_artifact` can only 404.

        `user` is the same server-derived subject `run_analysis` refused to dispatch without and
        minted both per-execution JWTs against, so a row can never be attributed to an identity
        the caller merely asserted (genetics-results-suite-dh3).

        Recorded for a FAILED status too, when the supervisor still reported a manifest: a
        script that raised after writing its plot has produced a real, retained artifact, and
        the manifest is what the supervisor is willing to serve either way.
        """
        if not user or not session_id or not isinstance(result, dict):
            return
        execution_id = result.get("execution_id")
        entries = result.get("artifacts")
        if not isinstance(execution_id, str) or not isinstance(entries, list):
            return
        names = [
            entry["name"]
            for entry in entries
            if isinstance(entry, dict) and isinstance(entry.get("name"), str) and entry["name"]
        ]
        _ARTIFACT_MANIFESTS.record(user, session_id, execution_id, names)

    # How many image artifacts one script may have shown for it. A script that writes fifty
    # PNGs is not asking for fifty pictures in the transcript, and each one is a fetch the
    # user waits through after the analysis has already finished.
    _MAX_ANALYSIS_IMAGES = 4

    async def _fetch_analysis_images(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        """Pull the image artifacts of a completed execution back out of the sandbox.

        Automatic rather than a tool the model calls: an image is for the USER to look at,
        and routing it through the model would cost a roundtrip to fetch something the model
        cannot see anyway. `read_artifact` is the general retrieval path for everything else
        and now goes through the SAME client method (`SandboxClient.get_artifact`) with the
        same cap, so the two readers cannot drift apart again.

        `execution_id` comes from the supervisor's own echoed response and is used here and
        nowhere else; `_render_analysis` still keeps it out of what the model reads.
        """
        if not isinstance(result, dict) or result.get("status") != "ok":
            return []
        execution_id = result.get("execution_id")
        if not isinstance(execution_id, str) or not execution_id:
            return []
        entries = result.get("artifacts")
        if not isinstance(entries, list):
            return []

        try:
            from genetics_mcp_server.sandbox_client import ARTIFACT_READ_MAX_BYTES
        except ModuleNotFoundError as exc:
            if not _pruned_module(exc, "genetics_mcp_server.sandbox_client"):
                raise
            # structurally unreachable: this runs only on a result `run_analysis` produced,
            # and that method guards the same import and returns before any result exists.
            # Logged rather than raised or shaped because the caller has a SUCCESSFUL
            # analysis in hand, and losing it over the image plot would be the worse
            # failure; the empty list is not silent, it is this line.
            logger.error(
                "cannot fetch analysis images: genetics_mcp_server.sandbox_client is not "
                "installed in this image"
            )
            return []

        wanted = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            content_type = entry.get("content_type")
            size = entry.get("size")
            if not isinstance(name, str) or not name:
                continue
            if not isinstance(content_type, str) or not content_type.startswith("image/"):
                continue
            # skipped locally rather than fetched and refused: the supervisor would answer
            # 413 and the round trip buys nothing
            if isinstance(size, int) and not isinstance(size, bool) and size > ARTIFACT_READ_MAX_BYTES:
                logger.info("skipping oversize image artifact %s (%d bytes)", name, size)
                continue
            wanted.append(name)
            if len(wanted) >= self._MAX_ANALYSIS_IMAGES:
                break

        images = []
        for name in wanted:
            fetched = await self._sandbox.fetch_artifact(execution_id, name)
            if fetched:
                images.append(fetched)
        return images

    @staticmethod
    def _capability_unavailable_error(capability: str, module: str) -> dict[str, Any]:
        """The shaped failure for a capability whose code this environment does not contain.

        Distinct from `_sandbox_operator_error` on purpose. That one says an operator has
        not configured a thing the process could otherwise do; this says the module is not
        installed, so there is nothing to configure. Both are non-retryable, but only one of
        them is something an operator can act on.
        """
        logger.error(
            "%s is unavailable: %s is not installed in this environment", capability, module
        )
        return {
            "success": False,
            "error": _capability_absent_message(capability, module),
            "error_type": CAPABILITY_UNAVAILABLE,
            "retryable": False,
        }

    @staticmethod
    def _sandbox_operator_error(message: str) -> dict[str, Any]:
        """A misconfiguration the model cannot route around, marked so it stops trying."""
        return {
            "success": False,
            "error": message,
            "error_type": "SandboxNotConfigured",
            "retryable": False,
        }

    def _render_analysis(
        self, result: dict[str, Any], images: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """Turn the supervisor's 200 body into the tool result the model reads.

        Tolerant by construction: every field is read defensively, an unknown `status` is
        reported as itself and counts as not-ok, and an unrecognised `error.type` is a label
        to display. The contract reserves the supervisor's names as a MINIMUM.
        """
        status = result.get("status")
        status_text = status if isinstance(status, str) else "unknown"
        ok = status_text == "ok"

        artifacts = []
        raw_artifacts = result.get("artifacts")
        if isinstance(raw_artifacts, list):
            for entry in raw_artifacts:
                if not isinstance(entry, dict):
                    continue
                name = entry.get("name")
                if not isinstance(name, str) or not name:
                    continue
                size = entry.get("size")
                content_type = entry.get("content_type")
                artifacts.append(
                    {
                        "name": name,
                        "size": size if isinstance(size, int) and not isinstance(size, bool) else None,
                        "content_type": content_type if isinstance(content_type, str) else None,
                    }
                )

        rendered: dict[str, Any] = {"success": ok, "status": status_text}
        if result.get("artifacts_retained_in_clear") is True:
            # BEFORE `output`, DELIBERATELY, and this is a security ordering rather than a
            # cosmetic one (genetics-results-suite-4h6.97). llm_service truncates a serialised
            # tool result over `mcp_max_result_size` to a PREFIX, and `output` is
            # SCRIPT-CONTROLLED up to the supervisor's 64 KiB cap — so with this field after
            # it, a script that both provokes the condition and prints ~50 KB cuts the warning
            # out of what the model reads. MEASURED before the move: a 66,569-byte result
            # truncated at 50,000 contained neither the flag nor the note. Placing it here
            # makes that impossible for any output size, since json.dumps preserves insertion
            # order. It is belt-and-braces with `_truncation_notice`, which re-states the note
            # from ARTIFACTS_RETAINED_IN_CLEAR_NOTE without depending on ordering at all.
            #
            # RENDERED ONLY WHEN TRUE, the same shape `artifacts_omitted` uses: the ordinary
            # case is every run, and a field that is false on every run is noise the model
            # learns to skip past.
            #
            # WHAT IT MEANS, AND WHO IT IS FOR. The supervisor sets it when the seal pass could
            # NEITHER encrypt NOR delete this execution's outputs, so they sit in plaintext on
            # a /scratch that every process at the shared uid can read until the reaper removes
            # the directory. The primary audience is the OPERATOR and that half already works
            # (`LOG.error` in the supervisor, and it is adversarial-only in practice — a
            # same-uid peer chmod-ing artifacts/, not ENOSPC).
            #
            # SO THE MODEL'S JOB IS NARROW, AND THE WORDING SAYS SO. The exposure is to OTHER
            # TENANTS' CODE, not to this user — the user's own artifacts are theirs to see, and
            # telling them "your results may be compromised" would be wrong on both halves.
            # What the model can honestly do is (a) say the outputs could not be removed from
            # shared scratch, and (b) not treat the artifacts as trustworthy inputs to a
            # further conclusion, because bytes readable by a peer are also writable by one —
            # which is the same condition `read_artifact`'s 409 exists to catch.
            rendered["artifacts_retained_in_clear"] = True
            rendered["artifacts_retained_in_clear_note"] = ARTIFACTS_RETAINED_IN_CLEAR_NOTE
        rendered["output"] = result.get("output") if isinstance(result.get("output"), str) else ""
        rendered["output_truncated"] = bool(result.get("output_truncated"))
        rendered["artifacts"] = artifacts
        duration_ms = result.get("duration_ms")
        if isinstance(duration_ms, int) and not isinstance(duration_ms, bool):
            rendered["duration_ms"] = duration_ms
        omitted = result.get("artifacts_omitted")
        if isinstance(omitted, int) and not isinstance(omitted, bool) and omitted > 0:
            rendered["artifacts_omitted"] = omitted
        if images:
            # base64 payloads, stripped by llm_service before this dict is serialised into the
            # tool_result — they are for the browser, and the model cannot see an image it is
            # handed as base64 anyway. The note that replaces them is set there too.
            rendered["images"] = images

        if artifacts:
            # said once, here, rather than left to the model to infer from the manifest.
            # Images are fetched automatically (see `_fetch_analysis_images`); everything else
            # is now readable with `read_artifact` by NAME, resolved against this session
            # (`genetics-results-suite-4h6.52`). The retention window is stated because it is
            # short and because the model cannot discover it any other way.
            shown = {
                image["name"]
                for image in images or []
                if isinstance(image, dict) and isinstance(image.get("name"), str)
            }
            readable = [entry["name"] for entry in artifacts if entry["name"] not in shown]
            note = ""
            if shown:
                note = (
                    "Image artifacts have been displayed to the user already; describe what "
                    "the plot shows rather than emitting a placeholder or a markdown image."
                )
            if readable:
                note += (
                    f"{' ' if note else ''}Read any other artifact with read_artifact, passing "
                    f"its name from this manifest — it is available for about "
                    f"{ARTIFACT_RETENTION_S // 60} minutes and only from this conversation. "
                    f"Large files come back truncated, so prefer printing a summary from the "
                    f"script when you only need a few numbers."
                )
            rendered["artifacts_note"] = note

        if not ok:
            rendered.update(self._analysis_error_fields(result, status_text))

        return rendered

    def _analysis_error_fields(self, result: dict[str, Any], status_text: str) -> dict[str, Any]:
        """The actionable half of a failed run: what raised, where, and which limit fired.

        A failing script costs a whole model roundtrip, and the measured distribution of
        roundtrips has a tail at 8+ that burns a third of the spend — so the error carries
        the exception type, the traceback tail and the specific limit rather than a summary,
        on the theory that the model repairs in one attempt instead of three.
        """
        error = result.get("error")
        error = error if isinstance(error, dict) else {}
        error_type = error.get("type")
        error_type = error_type if isinstance(error_type, str) else None
        message = error.get("message")
        tb = error.get("traceback")
        limit = error.get("limit")

        fields: dict[str, Any] = {
            "error": message if isinstance(message, str) and message else f"Script {status_text}",
        }
        if error_type:
            fields["error_type"] = error_type
        if isinstance(tb, str) and tb:
            fields["traceback"] = tb
        if isinstance(limit, str) and limit:
            fields["limit_exceeded"] = limit

        hint = self._analysis_hint(status_text, error_type, limit, message)
        if hint:
            fields["hint"] = hint
        # a script that ran and failed is repairable by rewriting it, which is the model's
        # job; that is a different thing from the transport failures above, where retrying
        # the SAME script is the correct move.
        fields["retryable"] = False
        return fields

    @staticmethod
    def _analysis_hint(
        status_text: str, error_type: str | None, limit: Any, message: Any = None
    ) -> str | None:
        if status_text == "timeout":
            return (
                "The wall clock fired. Narrow the query or process fewer rows; raising "
                "timeout_s only helps if the script was genuinely close to finishing."
            )
        limit_name = limit if isinstance(limit, str) else error_type
        if status_text == "limit":
            limits = {
                "OutputLimit": "The script printed too much. Print a summary, not every row.",
                "MemoryLimit": "The script ran out of memory. Aggregate in the query rather "
                "than pulling every row into memory.",
                "ArtifactQuota": "The script wrote more artifact bytes than one execution is "
                "allowed. Write fewer or smaller files.",
                "ScratchQuota": "The script wrote more scratch bytes than one execution is "
                "allowed.",
                "PidLimit": "The script started too many processes. It should not need "
                "subprocesses at all.",
            }
            return limits.get(
                limit_name or "",
                f"A sandbox limit fired ({limit_name or 'unknown'}). Do less work per run.",
            )
        if error_type in ServerToolExecutor._SDK_MISUSE_ERROR_TYPES:
            # a ModuleNotFoundError naming one of the modules the image prunes is NOT the
            # script calling the SDK wrong, and sending the model to list_capabilities for
            # it is actively misleading: none of them appear there, so the list it is told
            # to consult cannot explain the failure it just saw.
            #
            # Not made dead by the guards above: those are in this module, which the
            # sandbox image does not contain, so they never run in the child. What runs
            # there is the script's own `import`, and this reads the raw report of it that
            # the supervisor forwards.
            absent = _absent_capability_named(message)
            if absent:
                return (
                    f"`{absent}` is not installed in the sandbox image, so this capability "
                    "is absent from the environment rather than being called wrongly. It "
                    "will not appear in list_capabilities and no rewrite reaches it — use "
                    "the SDK functions list_capabilities does report."
                )
            return (
                "This looks like the SDK being called differently from how it is defined. "
                "Call list_capabilities for the exact signatures before rewriting."
            )
        return None
