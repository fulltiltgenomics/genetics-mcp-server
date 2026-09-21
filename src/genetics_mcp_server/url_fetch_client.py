"""HTTP transport between chat-backend and the url-fetcher, and the per-user fetch cache.

Wire contract of record: ``url-fetcher/server.py``, ``url-fetcher/fetch.py`` and
``url-fetcher/guard.py`` in genetics-results-suite, with the prose in
``docs/code-execution-security.md``. The two ends **cannot share a module** — the fetcher is a
distroless pod with no third-party dependency and no secret, deliberately holding nothing but
its route — so every number and every field name the two must agree on is a named constant
here, quoted once from that source, rather than a literal at a call site. Same split, and the
same remedy, as :mod:`genetics_mcp_server.sandbox_client`.

Scope is transport plus cache. Nothing here knows about ``run_analysis``, the tool definitions
or :class:`~genetics_mcp_server.sandbox_client.SandboxInput`; turning a :class:`FetchedFile`
into a sandbox input is the caller's job.

Three rules this module exists to keep:

* **A refusal is never retried.** The fetcher's error envelope carries ``retryable`` on every
  error, and that flag — not the HTTP status — is the discriminant. The status cannot serve:
  ``502`` is retryable for a connection that failed mid-body and non-retryable for a TLS
  failure or an upstream ``404``, and both arrive as ``502``.
* **No credential is ever sent.** The fetcher is unauthenticated by design; its NetworkPolicy
  is the control. Adding an ``Authorization`` header here would put a credential on the wire
  to the one pod in the namespace that talks to the open internet.
* **The cache is per-user and in memory.** See :class:`FetchCache`.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

from .config.settings import get_settings

logger = logging.getLogger(__name__)


# --- the contract's numbers, in one place -----------------------------------------------
# Changing any of these desynchronises this client from the fetcher. They are the fetcher's,
# not this module's, and none of them is configurable.

FETCH_PATH = "/fetch"
HEALTH_PATH = "/healthz"
"""``server.py``: two routes and no third."""

HEALTH_ALLOWED_HOSTS_FIELD = "allowed_hosts"
"""``server.do_GET``: the guard's host allow-list, carried on the health route. A fetcher that
does not send the field is not the same as one that allows nothing, so absence reads as
unknown rather than as an empty list."""

MAX_REQUEST_BYTES = 8 * 1024
"""``server.MAX_REQUEST_BYTES``. The route takes a url and nothing else, and answers 413 to a
larger body, so the serialised payload is measured here and refused locally instead."""

MAX_FETCH_BYTES = 512 * 1024
"""``fetch.MAX_BYTES``. The fetcher **aborts** at the cap rather than truncating, so a success
body carrying more than this means the two ends disagree about the number."""

MAX_REDIRECTS = 5
"""``fetch.MAX_REDIRECTS``. Mirrored for the reported ``redirects`` count, which cannot
legitimately exceed it."""

NAME_MAX = 128
"""``fetch.NAME_MAX``. The cap ``fetch.filename_for`` truncates its output to."""

def _is_fetcher_name_char(ch: str) -> bool:
    """``fetch.filename_for`` keeps a character when ``ch.isalnum() or ch in "._-"`` —
    ``str.isalnum`` is Unicode-aware, so the rule mirrored here is the same one, not its ASCII
    subset. A regex of ``[A-Za-z0-9._-]`` would refuse a name the fetcher legitimately produced
    for a url like ``https://x/caf%C3%A9.txt``."""
    return ch.isalnum() or ch in "._-"


def is_fetcher_name(name: str) -> bool:
    """The rest of ``fetch.filename_for``'s output contract: every character satisfies
    :func:`_is_fetcher_name_char`, leading dots are stripped (so none survive) and an empty
    result becomes ``download`` (so it is never empty), at most :data:`NAME_MAX` characters. A
    ``name`` outside this shape is a fetcher we did not mirror, and it reaches a directory entry
    and a log line here, so it is checked like the digest and the size rather than trusted."""
    return (
        bool(name)
        and len(name) <= NAME_MAX
        and not name.startswith(".")
        and all(_is_fetcher_name_char(ch) for ch in name)
    )


FETCH_TIMEOUT_S = 30
"""``fetch.TIMEOUT_S`` — the fetcher's wall clock over connect, TLS, headers and body. It does
**not** bound name resolution, which ``guard.check`` runs outside it."""

FETCHER_CONNECT_TIMEOUT_S = 5.0
"""This client's connect timeout to the fetcher — not a mirrored number, and deliberately not
spelled ``CONNECT_TIMEOUT_S``, which ``fetch.py`` uses for its own connect timeout to the
upstream with a different value. A name-by-name comparison with the fetcher is how desync is
found here, so a name that collides across the two ends with two meanings defeats it.

Short on purpose, for the same reason the sandbox client's is: "no fetcher at all" is a deploy
state that must be distinguishable from a slow upstream."""

ALLOW_LIST_PROBE_TIMEOUT_S = 1.0
"""Per-phase bound on the allow-list probe, giving it a ~2 s worst case against the 10 s
:data:`FETCHER_CONNECT_TIMEOUT_S` allows. It is deliberately tighter than a health check:
this probe runs on system-prompt assembly and buys one sentence of the prompt, so a fetcher
that accepts a connection and then says nothing must not hold a chat turn for ten seconds."""

RESPONSE_MARGIN_S = 10
"""The client's allowance over the fetcher's wall clock for base64, serialisation and the
hop."""

# A memory bound against a responder that is not the fetcher we mirrored, not a contract check
# on it. The base64 of a MAX_FETCH_BYTES payload is ~0.67 MiB, but the rest of the envelope is
# NOT bounded by the fetcher: fetch.py follows a Location header with no length check (up to
# 65536 bytes, http.client's own limit), so the reported final url can run far past
# MAX_REQUEST_BYTES, and server._send's json.dumps escapes non-ASCII at 6 bytes out per input
# character, so a non-ASCII name or url inflates further still. A ceiling derived tightly from
# those fields would refuse a legitimate success; 2 MiB instead bounds a misbehaving responder
# to a few MiB resident rather than tens, which is the whole purpose.
MAX_RESPONSE_BYTES = 4 * MAX_FETCH_BYTES


def client_deadline_s() -> float:
    """The client's budget for one fetch: the fetcher's wall clock plus a margin.

    :data:`FETCH_TIMEOUT_S` bounds the fetcher's connect, TLS, header and body phases, and
    :data:`RESPONSE_MARGIN_S` covers what happens after them — base64, serialisation, the hop.

    It is **not** a bound on the fetcher's worst case, because the fetcher has none: its
    ``guard.check`` resolves every hop with ``socket.getaddrinfo``, which takes no timeout and
    runs outside the wall clock it then starts. A fetcher stuck in resolution is therefore
    reported here as unavailable while it is still working, and its eventual answer is dropped.
    That is the deliberate trade — a caller that waits for a resolver has no deadline at all.
    """
    return float(FETCH_TIMEOUT_S + RESPONSE_MARGIN_S)


# The ``error.type`` values the fetcher emits. ``type`` is carried onto the exception as an
# opaque label and is NOT what decides retry — ``retryable`` is. These exist so a caller can
# branch on a specific cause without spelling the string again.
ERROR_INVALID_REQUEST = "invalid_request"
ERROR_REFUSED_BY_POLICY = "refused_by_policy"
"""``guard.py``: scheme, userinfo, host allow-list, port, address class, metadata deny-list,
and the redirect cap. Every one of them names the policy in its message, which is the whole
point of relaying the message to the caller."""
ERROR_UNREACHABLE = "unreachable"
ERROR_TIMED_OUT = "timed_out"
ERROR_TOO_LARGE = "too_large"
ERROR_TRUNCATED = "truncated"
ERROR_UPSTREAM_STATUS = "upstream_status"
ERROR_UNREQUESTED_ENCODING = "unrequested_encoding"
ERROR_INTERNAL = "internal_error"

# Synthesised by this client, not the fetcher: snake_case to sit in the same namespace as the
# wire's own labels, because a caller reads them from one field.
ERROR_FETCHER_UNREACHABLE = "fetcher_unreachable"
"""No HTTP response was obtained at all."""
ERROR_FETCHER_PROTOCOL = "fetcher_protocol"
"""A response the contract does not describe."""

UPSTREAM_DETAIL_FIELDS = ("upstream_status", "upstream_content_type", "upstream_body_prefix_b64")
"""The extra fields ``fetch._raise_upstream`` puts beside ``type``/``message``/``retryable``.
They are the evidence that an HTML error page served as a ``.tsv`` is an error page, so they
are carried onto the exception rather than dropped."""

MAX_ERROR_TYPE_CHARS = 256
MAX_ERROR_MESSAGE_CHARS = 2048
MAX_DETAIL_CHARS = 512


class UrlFetchError(RuntimeError):
    """Base for every failure to obtain the bytes.

    ``retryable`` is the contract's own flag where the fetcher answered, and this client's
    judgement where it did not. The two subclasses below are that flag made checkable.
    """

    retryable = False

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_type: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.details = details or {}


class UrlFetchRefused(UrlFetchError):
    """The request was answered with a no that a second ask cannot turn into a yes.

    The rule, rather than a list that would rot: every failure the fetcher marks non-retryable
    except its own ``internal_error`` — which is a bug in the fetcher and not a decision about
    the request, so it belongs to :class:`UrlFetchProtocolError` — plus this client's own
    pre-flight rejections.

    Widening the fetcher's allow-list is a config change, so the message is relayed to the
    caller verbatim: a refusal that names its policy is a visible request to widen it, and a
    refusal the user never sees is not.
    """


class UrlFetchUnavailable(UrlFetchError):
    """A transient failure: retrying the same request may succeed.

    A slow or flapping upstream, a connection that failed mid-body, a short body, an upstream
    5xx, or no fetcher to talk to. Deliberately distinct from :class:`UrlFetchRefused` because
    the two call for opposite next moves, and telling a user their URL is not allowed when the
    host was merely slow is the more expensive mistake.
    """

    retryable = True


class UrlFetchNotConfigured(UrlFetchError):
    """No ``URL_FETCHER_URL``, so there is no address to talk to and no request is ever sent.

    An error at construction rather than a default URL, mirroring
    :class:`~genetics_mcp_server.sandbox_client.SandboxNotConfigured` and for the same measured
    reason (genetics-results-suite-6um): every fallback from "no address" points a POST at
    whatever occupies a common port on a dev machine, and a real service's 404s and auth errors
    then classify as fetcher failures, hiding the one diagnostic the operator needs.

    In this family so nothing escapes ``except UrlFetchError``, and non-retryable because a
    second ask cannot supply a missing address.
    """


class UrlFetchProtocolError(UrlFetchError):
    """The fetcher did not behave as the contract describes: a 200 whose body is unparseable,
    over a cap, or carries a field that does not describe the bytes it came with — or the
    fetcher's own ``internal_error``, which reports a bug on that side and is no more a
    decision about this request than a malformed body is."""


@dataclass(frozen=True)
class FetchedFile:
    """One fetched file: every field of the fetcher's 200 body, with the bytes decoded.

    ``content`` replaces the wire's ``content_b64``; ``size_bytes`` and ``sha256`` are the
    fetcher's own and are **checked against** ``content`` before this object exists, so a
    caller may use either without re-deriving it.

    ``url`` is the FINAL url after ``redirects`` hops, which is not necessarily the one that
    was asked for — the cache is keyed on the requested url, never on this one.

    ``content_type`` and ``content_encoding`` are labels the upstream supplied, stripped of
    control characters by the fetcher. Nothing downstream may trust them to describe the bytes.
    """

    url: str
    name: str
    content: bytes
    size_bytes: int
    sha256: str
    content_type: str | None
    content_encoding: str | None
    redirects: int


def _cap(value: str, limit: int) -> str:
    return value[:limit]


def _error_fields(response: httpx.Response) -> tuple[str | None, str, bool | None, dict[str, Any]]:
    """Read ``{"error": {"type", "message", "retryable", ...}}`` out of a body, tolerating
    anything. A client that raises while parsing an error response turns a diagnosable failure
    into an undiagnosable one, so every departure from the shape degrades to "nothing known".

    ``retryable`` comes back ``None`` when it is absent or not a bool, which is the signal for
    the caller to fall back to its own judgement rather than to read a missing flag as False.
    """
    try:
        body = response.json()
    except (json.JSONDecodeError, ValueError):
        return None, "", None, {}
    if not isinstance(body, dict):
        return None, "", None, {}
    error = body.get("error")
    if not isinstance(error, dict):
        return None, "", None, {}
    error_type = error.get("type")
    message = error.get("message")
    retryable = error.get("retryable")
    details = {}
    for key in UPSTREAM_DETAIL_FIELDS:
        value = error.get(key)
        if isinstance(value, str):
            details[key] = _cap(value, MAX_DETAIL_CHARS)
        elif isinstance(value, int) and not isinstance(value, bool):
            details[key] = value
    return (
        error_type if isinstance(error_type, str) else None,
        message if isinstance(message, str) else "",
        retryable if isinstance(retryable, bool) else None,
        details,
    )


class FetchCache:
    """Successful fetches, in memory, keyed by ``(user, url)``.

    **The key is a tuple, never a joined string.** That is the security property, not a style
    choice: any ``f"{user}|{url}"`` form makes collisions a question of which separator a user
    can put in their own identifier, and one user's bytes reaching another is the single
    failure this module must not have. A tuple of two strings has no separator to attack.

    **In memory only, never the chat PVC.** Remotely fetched bytes must not land on the volume
    that holds conversations: that volume carries retention and deletion obligations this
    content has no business inheriting, and a process restart is a perfectly acceptable way to
    lose a latency optimisation.

    **Invisible.** It holds successes only and it is consulted only by
    :meth:`UrlFetchClient.fetch`. Deleting this class must change nothing but latency, which is
    why failures are not cached — a negative entry would make a refusal survive the config
    change that widens the allow-list, and a blip survive the upstream's recovery.

    Bounds are a TTL and a total decoded-byte ceiling, both configuration. Eviction is
    insertion order: a plain dict, a timestamp per entry and a running total. There is no
    background sweeper — expiry is checked on read and stale entries are dropped there and on
    insert, so the cache costs nothing while idle.
    """

    def __init__(self, *, ttl_s: float, max_bytes: int, clock=time.monotonic) -> None:
        self._ttl_s = float(ttl_s)
        self._max_bytes = int(max_bytes)
        self._clock = clock
        # insertion-ordered; the oldest entry is the first key
        self._entries: dict[tuple[str, str], tuple[float, FetchedFile]] = {}
        self._total_bytes = 0

    @property
    def enabled(self) -> bool:
        """A non-positive TTL or ceiling disables the cache outright — the documented way to
        turn it off, and the state the client is in when it is constructed without one."""
        return self._ttl_s > 0 and self._max_bytes > 0

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, user: str, url: str) -> FetchedFile | None:
        if not self.enabled or not user:
            return None
        entry = self._entries.get((user, url))
        if entry is None:
            return None
        stored_at, result = entry
        if self._clock() - stored_at >= self._ttl_s:
            self._drop((user, url))
            return None
        return result

    def put(self, user: str, url: str, result: FetchedFile) -> None:
        if not self.enabled or not user:
            return
        size = len(result.content)
        if size > self._max_bytes:
            # one entry that cannot coexist with any other would empty the cache to hold
            # itself; not storing it leaves every other user's entries alone
            return
        self._expire()
        self._drop((user, url))
        while self._entries and self._total_bytes + size > self._max_bytes:
            self._drop(next(iter(self._entries)))
        self._entries[(user, url)] = (self._clock(), result)
        self._total_bytes += size

    def clear(self) -> None:
        self._entries.clear()
        self._total_bytes = 0

    def _expire(self) -> None:
        now = self._clock()
        for key in [k for k, (at, _) in self._entries.items() if now - at >= self._ttl_s]:
            self._drop(key)

    def _drop(self, key: tuple[str, str]) -> None:
        entry = self._entries.pop(key, None)
        if entry is not None:
            self._total_bytes -= len(entry[1].content)


class UrlFetchClient:
    """One configuration value — a base URL — and no branch on where the fetcher runs.

    The URL points at the in-cluster Service in production and at ``devserver.py`` locally; the
    contract is identical in both, so nothing downstream may branch on which one this is.

    The client does **not** retry. The fetcher has no queue and answers no 429, so a transient
    failure is the upstream's, and whether a chat turn can afford a second 30-second wait is
    the caller's decision, not this layer's.
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        cache: FetchCache | None = None,
    ) -> None:
        resolved = base_url or get_settings().url_fetcher_url
        if not resolved:
            raise UrlFetchNotConfigured(
                "URL_FETCHER_URL is not set: refusing to guess where the url-fetcher is. Set "
                "it to the in-cluster Service (http://url-fetcher.genetics.svc.cluster.local"
                ":8090) or, locally, to what url-fetcher/devserver.py publishes "
                "(http://127.0.0.1:8090).",
                error_type=ERROR_FETCHER_UNREACHABLE,
            )
        self.base_url = resolved.rstrip("/")
        self.cache = cache if cache is not None else FetchCache(ttl_s=0, max_bytes=0)
        self._transport = transport
        self._allowed_hosts: tuple[str, ...] | None = None

    def _client(self, timeout: httpx.Timeout) -> httpx.AsyncClient:
        # a fresh client per call, so nothing here keeps a connection between fetches. Leaving
        # the ``async with`` closes an INJECTED transport too, which would make a second fetch
        # through the same client object fail: harmless while the only injected transport is a
        # test's MockTransport, whose close is a no-op, and real the moment something passes a
        # transport that holds a pool or a socket. Giving this client a long-lived transport
        # means giving it ownership of that transport's lifetime, not just calling it twice.
        return httpx.AsyncClient(
            base_url=self.base_url, timeout=timeout, transport=self._transport
        )

    async def _health(self, phase_timeout_s: float = FETCHER_CONNECT_TIMEOUT_S) -> httpx.Response | None:
        """``GET /healthz``, or None if the fetcher could not be reached at all.

        ``phase_timeout_s`` bounds connect and read separately, so the worst case a caller can
        wait is twice it: a connect that stalls to its limit, then a response that never
        arrives."""
        timeout = httpx.Timeout(
            phase_timeout_s, connect=phase_timeout_s, read=phase_timeout_s
        )
        try:
            async with self._client(timeout) as client:
                return await client.get(HEALTH_PATH)
        except httpx.HTTPError:
            return None

    async def healthy(self) -> bool:
        """``GET /healthz``. True iff the fetcher answered 200; never raises, because every
        caller of this wants "is it there" rather than a diagnosis."""
        response = await self._health()
        return response is not None and response.status_code == 200

    async def allowed_hosts(self) -> tuple[str, ...] | None:
        """The hosts the fetcher's guard will fetch from, or None when it cannot say.

        None covers every way the answer is not knowable — unreachable, non-200, the field
        absent because the fetcher predates it, or a shape that is not a list of strings —
        because a caller that cannot distinguish them would have to guess, and the one caller
        there is (the system prompt) says nothing rather than guessing.

        Cached on the instance after the first success, and never re-read. That is a cache in
        a **different pod** from the one whose configuration it holds, so it does go stale:
        widening the fetcher's list leaves this process naming the old one until it restarts.
        What makes that acceptable is that nothing here is the decision — the fetcher still
        refuses or allows the fetch itself, and the only cost of a stale copy is that the model
        is told about a host it no longer has to avoid, or not told about one it could now use,
        for one refused fetch. Narrowing the list is never unsafe for the same reason.

        It is NEVER on the request path — :meth:`fetch` must not wait on a second round trip to
        learn what a refusal would tell it anyway — so it is probed on a shorter budget than
        :meth:`healthy`: a prompt sentence is worth less waiting than a deploy diagnosis.
        """
        if self._allowed_hosts is not None:
            return self._allowed_hosts
        response = await self._health(ALLOW_LIST_PROBE_TIMEOUT_S)
        if response is None or response.status_code != 200:
            return None
        try:
            body = response.json()
        except ValueError:
            return None
        if not isinstance(body, dict):
            return None
        hosts = body.get(HEALTH_ALLOWED_HOSTS_FIELD)
        if not isinstance(hosts, list) or not all(isinstance(h, str) for h in hosts):
            return None
        self._allowed_hosts = tuple(hosts)
        return self._allowed_hosts

    async def fetch(self, url: str, *, user: str) -> FetchedFile:
        """The bytes at ``url``, fetched on behalf of ``user``.

        ``user`` is mandatory and is the cache key's first element, **exactly as given**: it is
        not trimmed or folded, because normalising it here would make two spellings of an
        identifier share one cache entry while every other user-scoped store in this process
        (``download_store``) compares owners by plain equality. An identity this layer accepts
        as equal and that one does not is how one user's bytes reach another.

        It never reaches the wire: the fetcher's request body carries a url and nothing else,
        and it refuses an unknown field rather than ignoring it — which is exactly why the
        per-user cache lives here and not there.
        """
        if not isinstance(user, str) or not user.strip():
            # a local pre-flight refusal, in the same family as the wire's, because an empty
            # user would put every caller's bytes in one cache bucket
            raise UrlFetchRefused(
                "a fetch must name the user it is for; the cache key is (user, url)",
                error_type=ERROR_INVALID_REQUEST,
            )
        if not isinstance(url, str) or not url.strip():
            raise UrlFetchRefused(
                "'url' must be a non-empty string", error_type=ERROR_INVALID_REQUEST
            )
        url = url.strip()

        cached = self.cache.get(user, url)
        if cached is not None:
            return cached

        # serialised here rather than by httpx so the 8 KiB cap is measured on the bytes that
        # actually go on the wire
        payload = json.dumps({"url": url}, ensure_ascii=False).encode("utf-8")
        if len(payload) > MAX_REQUEST_BYTES:
            raise UrlFetchRefused(
                f"the request body is {len(payload)} bytes, over the {MAX_REQUEST_BYTES} byte "
                "cap; this url is too long to fetch",
                error_type=ERROR_INVALID_REQUEST,
            )

        timeout = httpx.Timeout(
            client_deadline_s(), connect=FETCHER_CONNECT_TIMEOUT_S, read=client_deadline_s()
        )
        try:
            async with self._client(timeout) as client:
                async with client.stream(
                    "POST",
                    FETCH_PATH,
                    content=payload,
                    headers={"Content-Type": "application/json"},
                ) as streamed:
                    response = await self._read_bounded(streamed)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise UrlFetchUnavailable(
                f"url-fetcher is not reachable at {self.base_url}; it may be restarting",
                error_type=ERROR_FETCHER_UNREACHABLE,
            ) from exc
        except httpx.TimeoutException as exc:
            raise UrlFetchUnavailable(
                f"url-fetcher did not answer within {client_deadline_s():.0f}s",
                error_type=ERROR_FETCHER_UNREACHABLE,
            ) from exc
        except httpx.TransportError as exc:
            raise UrlFetchUnavailable(
                f"url-fetcher connection failed at {self.base_url}: {type(exc).__name__}",
                error_type=ERROR_FETCHER_UNREACHABLE,
            ) from exc

        result = self._interpret(response)
        self.cache.put(user, url, result)
        return result

    async def _read_bounded(self, streamed: httpx.Response) -> httpx.Response:
        """The response with its body read under a hard cap — and refused *unread* when the
        declared length is already over it.

        Measuring a body only after parsing and decoding it costs a multiple of the sender's
        byte count — 61.5 MB resident for a 16 MiB payload — in a single-replica process with a
        2Gi limit, so the frame is judged before anything is read. ``server._send`` writes a
        Content-Length on the success path, on every error path and on the unknown-verb path:
        an answer here without a parseable one is not the fetcher's and is not worth reading.
        The cap is then applied to the bytes as well — a declared length is the sender's claim
        about its body, not a limit on it.
        """

        def over(what: str) -> UrlFetchProtocolError:
            return UrlFetchProtocolError(
                f"url-fetcher answered with {what}, over the {MAX_RESPONSE_BYTES} byte ceiling "
                f"a {MAX_FETCH_BYTES} byte fetch can produce",
                status_code=streamed.status_code,
                error_type=ERROR_FETCHER_PROTOCOL,
            )

        try:
            declared = int(streamed.headers["Content-Length"])
        except (KeyError, ValueError):
            raise UrlFetchProtocolError(
                "url-fetcher answered without a usable Content-Length; every answer it frames "
                "carries one, so this body is not read at all",
                status_code=streamed.status_code,
                error_type=ERROR_FETCHER_PROTOCOL,
            ) from None
        if declared > MAX_RESPONSE_BYTES:
            raise over(f"a declared {declared} bytes")

        chunks: list[bytes] = []
        total = 0
        async for chunk in streamed.aiter_bytes():
            total += len(chunk)
            if total > MAX_RESPONSE_BYTES:
                raise over(f"a body of at least {total} bytes")
            chunks.append(chunk)

        # the declared length is dropped rather than copied: it is the sender's claim, and the
        # bytes below are what was actually read
        headers = httpx.Headers(streamed.headers)
        headers.pop("Content-Length", None)
        return httpx.Response(streamed.status_code, headers=headers, content=b"".join(chunks))

    def _interpret(self, response: httpx.Response) -> FetchedFile:
        if response.status_code == 200:
            return self._result(response)

        error_type, raw_message, retryable, details = _error_fields(response)
        error_type = _cap(error_type, MAX_ERROR_TYPE_CHARS) if error_type else None
        message = _cap(raw_message, MAX_ERROR_MESSAGE_CHARS)
        detail = f"{error_type or 'unknown'}: {message}" if message else (error_type or "unknown")

        if retryable is None:
            # nothing between this client and the fetcher is supposed to answer, so an envelope
            # without the flag came from something else — an ingress, a proxy, a crash page.
            # 5xx is the only shape of that worth a second attempt; everything else is treated
            # as a refusal, because retrying something we cannot classify costs the upstream
            # host the politeness this service is meant to keep.
            retryable = response.status_code >= 500
            logger.warning(
                "url-fetcher answered HTTP %s with no usable error envelope",
                response.status_code,
            )
        if error_type == ERROR_INTERNAL:
            # the fetcher says it crashed on its own side. Non-retryable, but not a decision
            # about this request: presenting it as a refusal would tell the user their URL is
            # not allowed when what happened is a bug in the service that fetches it.
            raise UrlFetchProtocolError(
                f"url-fetcher failed inside the service ({detail})",
                status_code=response.status_code,
                error_type=error_type,
                details=details,
            )
        failure = UrlFetchUnavailable if retryable else UrlFetchRefused
        raise failure(
            f"url-fetcher did not fetch the url ({detail})",
            status_code=response.status_code,
            error_type=error_type,
            details=details,
        )

    def _result(self, response: httpx.Response) -> FetchedFile:
        try:
            body = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise UrlFetchProtocolError(
                "url-fetcher returned a body that is not JSON",
                status_code=response.status_code,
                error_type=ERROR_FETCHER_PROTOCOL,
            ) from exc
        if not isinstance(body, dict):
            raise UrlFetchProtocolError(
                "url-fetcher returned a JSON body that is not an object",
                status_code=response.status_code,
                error_type=ERROR_FETCHER_PROTOCOL,
            )

        def malformed(reason: str) -> UrlFetchProtocolError:
            return UrlFetchProtocolError(
                f"url-fetcher returned a result that {reason}",
                status_code=response.status_code,
                error_type=ERROR_FETCHER_PROTOCOL,
            )

        for field_name in ("url", "name", "sha256", "content_b64"):
            if not isinstance(body.get(field_name), str):
                raise malformed(f"carries no {field_name}")
        # `name` is checked against the fetcher's output contract for the same reason the digest
        # and the size are: it is the one success field that leaves this module as data someone
        # acts on — a directory entry where the sandbox writes the file, and a log line here.
        name = body["name"]
        if not is_fetcher_name(name):
            raise malformed(
                f"carries a name of {len(name)} characters that fetch.filename_for cannot "
                f"produce; it keeps only characters where ch.isalnum() or ch in '._-', no "
                f"leading dot, at most {NAME_MAX}"
            )

        size_bytes = body.get("size_bytes")
        if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes < 0:
            raise malformed("carries no usable size_bytes")
        redirects = body.get("redirects")
        if not isinstance(redirects, int) or isinstance(redirects, bool) or redirects < 0:
            raise malformed("carries no usable redirects count")
        if redirects > MAX_REDIRECTS:
            raise malformed(f"followed {redirects} redirects, over the {MAX_REDIRECTS} cap")

        try:
            content = base64.b64decode(body["content_b64"], validate=True)
        except (binascii.Error, ValueError) as exc:
            raise malformed("carries content_b64 that is not base64") from exc

        # the two integrity checks, both against the bytes rather than against each other: the
        # fetcher aborts at the cap rather than truncating, so a body over it means the two ends
        # disagree about the number, and a digest that does not describe the content means
        # something between here and the fetch altered it.
        if len(content) > MAX_FETCH_BYTES:
            raise malformed(
                f"is {len(content)} bytes, over the {MAX_FETCH_BYTES} byte cap the fetcher "
                "aborts at"
            )
        if len(content) != size_bytes:
            raise malformed(
                f"declares {size_bytes} bytes and carries {len(content)}"
            )
        if hashlib.sha256(content).hexdigest() != body["sha256"]:
            raise malformed("carries a sha256 that does not describe its content")

        content_type = body.get("content_type")
        content_encoding = body.get("content_encoding")
        logger.info(
            "url-fetcher fetched %s (%s bytes, %s redirects)",
            name,
            size_bytes,
            redirects,
        )
        return FetchedFile(
            url=body["url"],
            name=name,
            content=content,
            size_bytes=size_bytes,
            sha256=body["sha256"],
            content_type=content_type if isinstance(content_type, str) else None,
            content_encoding=content_encoding if isinstance(content_encoding, str) else None,
            redirects=redirects,
        )


# singleton, so the cache outlives one request. Same shape as get_download_store().
_client: UrlFetchClient | None = None


def get_url_fetch_client() -> UrlFetchClient:
    """The process-wide client and its cache.

    Raises :class:`UrlFetchNotConfigured` when ``URL_FETCHER_URL`` is unset, every time — the
    failure is not memoised, so setting the variable and reloading works.
    """
    global _client
    if _client is None:
        settings = get_settings()
        _client = UrlFetchClient(
            cache=FetchCache(
                ttl_s=settings.url_fetch_cache_ttl_seconds,
                max_bytes=settings.url_fetch_cache_max_bytes,
            )
        )
    return _client


def reset_url_fetch_client() -> None:
    """Drop the singleton and everything it has cached."""
    global _client
    if _client is not None:
        _client.cache.clear()
    _client = None
