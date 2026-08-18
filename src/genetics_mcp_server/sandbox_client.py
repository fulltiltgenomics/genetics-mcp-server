"""HTTP transport between chat-backend and the sandbox supervisor.

Wire contract of record: ``docs/code-execution-security.md`` §2, "The HTTP contract between
chat-backend and the supervisor" (genetics-results-suite, ``4h6.38``). The supervisor
(``4h6.39``) is written against the same document and **cannot share a module with this one**
— the sandbox image pip-installs only the genetics SDK's import closure and
``sandbox/prune_venv.py`` deletes everything else. Every number the two ends must agree on is
therefore a named constant here, quoted once from that document, rather than a literal at a
call site: the timeouts, the queue bound and the retry behaviour are exactly where a client
and a server drift apart without either noticing.

Scope is transport only. ``run_analysis`` (`4h6.48`) owns the tool definition, the rendering
and the MCP exclusion; artifact retrieval over HTTP is `4h6.52`'s and no route for it exists
yet. :meth:`SandboxClient.execute` returns the supervisor's structured result **unchanged**.

Two rules this module exists to keep:

* **Fail closed.** ``mint_execution_tokens`` raises :class:`SandboxTokenUnavailable` when no
  signing key is configured. That exception is never caught here. Every fallback from "no
  sandbox token" is either sending no credential or sending the shared ``INTERNAL_API_SECRET``,
  which are the two outcomes §4 exists to prevent.
* **Never log a token.** Not at debug, not in an exception message, not in a repr. The bodies
  this module builds are never logged, and **both** halves of an error object the supervisor
  sends back — ``type`` as well as ``message`` — are scrubbed of the token strings and capped
  before any error text is built, logged or attached to an exception.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import httpx

from .config.settings import get_settings
from .sandbox_token import (
    DB_API_AUDIENCE,
    RESULTS_API_AUDIENCE,
    SandboxTokens,
    mint_execution_tokens,
)

logger = logging.getLogger(__name__)


# --- the contract's numbers, in one place -----------------------------------------------
# Changing any of these desynchronises this client from the supervisor. They are §2's, not
# this module's, and none of them is configurable: a per-deployment override would be a
# local/prod branch in the request flow, which the contract forbids.

DEFAULT_TIMEOUT_S = 60
"""Wall clock the supervisor applies when the body omits ``timeout_s``."""

MAX_TIMEOUT_S = 120
"""Hard ceiling. The supervisor **rejects** a larger value with 400 rather than clamping, so
this client rejects it locally too instead of sending a request it knows will fail."""

MAX_CODE_BYTES = 256 * 1024
"""Measured on ``len(code.encode("utf-8"))`` — the decoded string, not the JSON escaping."""

MAX_BODY_BYTES = 1024 * 1024
"""Measured on the raw bytes on the wire."""

MAX_QUEUED_WAIT_S = 120
"""The supervisor queues at most this long before answering 429. Queue depth is at most two
*waiting*, not counting the one executing, so a 429 means genuinely full."""

BODY_WRITE_DEADLINE_S = 10
"""The supervisor gives a request body 10s from the request line to the last byte, then 408.
Matching it here means a stalled write surfaces as our own error rather than as a 408."""

CONNECT_TIMEOUT_S = 5.0
"""Short on purpose: an unreachable sandbox must be distinguishable from a slow script, and
"no endpoint at all" is the expected state for up to ~130s of a deploy (Recreate plus
``terminationGracePeriodSeconds: 130``)."""

RESPONSE_MARGIN_S = 15
"""Slack over the supervisor's own worst case for reap, wipe, serialisation and the hop."""

RETRY_AFTER_FALLBACK_S = 60
"""What the supervisor sends on 429; used when the header is absent or unparseable."""

MAX_RETRY_WAIT_S = 60
"""Ceiling on an honoured ``Retry-After``, so a hostile or buggy header cannot park a chat
turn indefinitely."""

MAX_SUBMIT_ATTEMPTS = 2
"""One retry. Each attempt re-mints, so attempts multiply neither the queue nor the credential
lifetime — but a chat turn cannot afford many of them."""

EXECUTION_ID_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
"""§2's rule for ``execution_id``, verbatim. The supervisor makes it the ``/scratch/<id>``
directory name, so a laxer form here would hand it a path — and it 400s a non-matching value
anyway, which this client refuses to send."""

MAX_ERROR_TYPE_CHARS = 256
"""``error.type`` is as peer-controlled as ``error.message`` and is carried onto the exception,
so it gets the same scrub-and-cap treatment; the cap is smaller because it is a label."""

MAX_ERROR_MESSAGE_CHARS = 2048


def client_deadline_s(timeout_s: int) -> float:
    """The read deadline, which must sit **above** the supervisor's own worst case.

    The supervisor may hold a request for the full queued wait and *then* run it for the full
    ``timeout_s``, so the ceiling this must clear is ``MAX_QUEUED_WAIT_S + timeout_s``, not
    ``timeout_s``. Below that the client would time out on executions the supervisor is about
    to answer, and the user would read a live sandbox as a broken one.
    """
    return float(MAX_QUEUED_WAIT_S + timeout_s + RESPONSE_MARGIN_S)


# Reserved ``error.type`` names this client branches on. ``error.type`` is an OPEN string —
# half its range is the child's own exception class name — so these are the only values that
# get special handling and every other value is carried through as an opaque label.
ERROR_TOKEN_EXPIRED = "TokenExpired"
ERROR_DUPLICATE_EXECUTION_ID = "DuplicateExecutionId"
ERROR_BUSY = "Busy"
ERROR_NOT_READY = "NotReady"


class SandboxError(RuntimeError):
    """Base for every failure to *obtain* a result.

    A script that raised, timed out or hit a limit is **not** one of these: the supervisor
    answers 200 for all of those and :meth:`SandboxClient.execute` returns that body.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_type: str | None = None,
        retry_after: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.retry_after = retry_after


class SandboxUnavailable(SandboxError):
    """No sandbox to talk to: connection refused, DNS failure, 503 ``NotReady``, or a gateway
    error from something in between.

    Distinct from every other failure because it must not read to the user as "your script
    failed". ``strategy: Recreate`` plus ``terminationGracePeriodSeconds: 130`` means a deploy
    landing on an in-flight execution leaves no sandbox at all for up to ~130s, and that is a
    wait-and-retry condition, not a defect in the code the model wrote.
    """


class SandboxDeadlineExceeded(SandboxError):
    """The supervisor did not answer within :func:`client_deadline_s`.

    Also not a script failure: a script that runs too long comes back as ``200`` with
    ``status: "timeout"``. Reaching this means the supervisor itself is wedged or the response
    was lost, because the deadline already allows for the full queued wait plus the full run.
    """


class SandboxBusy(SandboxError):
    """429 after the retries were spent. One running plus two waiting is the whole capacity."""


class SandboxRejected(SandboxError):
    """The supervisor refused the request, or this client detected the violation before
    sending it. A caller bug in either case — retrying the identical request will not help."""


class SandboxInternalError(SandboxError):
    """500 ``InternalError`` — a supervisor bug."""


class SandboxProtocolError(SandboxError):
    """A response the contract does not describe: unparseable body, wrong shape, or an
    ``execution_id`` that is not the one we sent."""


@dataclass(frozen=True)
class SandboxHealth:
    """``GET /health``. The one route exempt from the uniform error shape — it answers both
    200 and 503 with this same body, so the 503 is parsed, not treated as an error object."""

    ready: bool
    """True iff the supervisor answered 200, i.e. it is accepting ``POST /execute``."""

    status: str
    """``"ok"`` | ``"starting"`` | ``"draining"``. Tolerated as an open string: an unknown
    value is reported, never a crash."""

    busy: bool | None
    queued: int | None
    """Requests *waiting*, not counting the one executing — the same definition the queue
    bound uses, so ``busy: true, queued: 0`` means one running and nothing behind it."""

    http_status: int


def _redact(text: str, tokens: SandboxTokens | None) -> str:
    """Defence in depth for the never-log-a-token rule.

    The supervisor is required never to echo a token, but this is the one place a value we
    minted could re-enter our own logs through a misbehaving or impersonating peer, and the
    check costs a substring scan on an error path.
    """
    if not tokens:
        return text
    for token in tokens.tokens.values():
        if token and token in text:
            text = text.replace(token, "[redacted]")
    return text


def _error_fields(response: httpx.Response) -> tuple[str | None, str]:
    """Read ``{"error": {"type", "message"}}`` out of a non-2xx body, tolerating anything.

    A client that raises while parsing an error response turns a diagnosable failure into an
    undiagnosable one, so every departure from the shape degrades to "no type, no message".
    """
    try:
        body = response.json()
    except (json.JSONDecodeError, ValueError):
        return None, ""
    if not isinstance(body, dict):
        return None, ""
    error = body.get("error")
    if not isinstance(error, dict):
        return None, ""
    error_type = error.get("type")
    message = error.get("message")
    return (
        error_type if isinstance(error_type, str) else None,
        message if isinstance(message, str) else "",
    )


def _retry_after(response: httpx.Response) -> int:
    raw = response.headers.get("Retry-After", "")
    try:
        seconds = int(raw.strip())
    except (TypeError, ValueError):
        return RETRY_AFTER_FALLBACK_S
    if seconds < 0:
        return RETRY_AFTER_FALLBACK_S
    return min(seconds, MAX_RETRY_WAIT_S)


class SandboxClient:
    """One configuration value — a base URL — and no branch on where the sandbox runs.

    The URL points at the in-cluster Service in production and at the local Docker container
    in development. Nothing else in the request flow differs between the two, which is what
    lets `4h6.40`'s plain-Docker backend and `4h6.49`'s local verification exercise the same
    code path the cluster does.
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self.base_url = (base_url or get_settings().sandbox_url).rstrip("/")
        # injection points for tests, which must run with no sandbox and no credentials
        self._transport = transport
        self._sleep = sleep or asyncio.sleep

    def _client(self, timeout: httpx.Timeout) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.base_url, timeout=timeout, transport=self._transport
        )

    async def health(self) -> SandboxHealth:
        timeout = httpx.Timeout(
            CONNECT_TIMEOUT_S,
            connect=CONNECT_TIMEOUT_S,
            read=CONNECT_TIMEOUT_S,
        )
        try:
            async with self._client(timeout) as client:
                response = await client.get("/health")
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise SandboxUnavailable(f"sandbox is not reachable at {self.base_url}") from exc
        except httpx.TimeoutException as exc:
            raise SandboxUnavailable(
                f"sandbox at {self.base_url} did not answer /health in {CONNECT_TIMEOUT_S}s"
            ) from exc
        except httpx.TransportError as exc:
            raise SandboxUnavailable(f"sandbox transport error at {self.base_url}") from exc

        body: Any
        try:
            body = response.json()
        except (json.JSONDecodeError, ValueError):
            body = {}
        if not isinstance(body, dict):
            body = {}
        status = body.get("status")
        busy = body.get("busy")
        queued = body.get("queued")
        return SandboxHealth(
            ready=response.status_code == 200,
            status=status if isinstance(status, str) else "unknown",
            busy=busy if isinstance(busy, bool) else None,
            queued=queued if isinstance(queued, int) and not isinstance(queued, bool) else None,
            http_status=response.status_code,
        )

    async def execute(
        self,
        *,
        code: str,
        user: str,
        session_id: str,
        timeout_s: int | None = None,
        execution_id: str | None = None,
    ) -> dict[str, Any]:
        """Submit one script and return the supervisor's result **unchanged**.

        ``execution_id`` is optional and is honoured on the first attempt only: it is minted
        here otherwise, and a retry always mints a fresh one because a repeated id is now
        refused with 409 ``DuplicateExecutionId``.

        Raises :class:`SandboxTokenUnavailable` — deliberately uncaught — when no signing key
        is configured, and one of the :class:`SandboxError` subclasses when no result could be
        obtained. A *script* failure is not an exception: it is a 200 with ``status`` of
        ``"error"``, ``"timeout"`` or ``"limit"``.
        """
        timeout_s = DEFAULT_TIMEOUT_S if timeout_s is None else timeout_s
        self._validate(code, timeout_s, user, session_id, execution_id)

        attempt = 0
        while True:
            attempt += 1
            # minted inside the loop, never before it: the tokens live 300s from mint, and a
            # queued execution eats that slack (a full 120s wait then a full 120s run leaves
            # ~60s of token life). Sitting on a minted pair while waiting to retry would spend
            # the credential on the wait.
            tokens = mint_execution_tokens(
                user=user, session_id=session_id, execution_id=execution_id
            )
            execution_id = None

            try:
                return await self._submit(tokens, code=code, timeout_s=timeout_s)
            except SandboxBusy as exc:
                if attempt >= MAX_SUBMIT_ATTEMPTS:
                    raise
                wait = exc.retry_after or RETRY_AFTER_FALLBACK_S
                logger.info(
                    "sandbox queue full (execution=%s), retrying in %ss with a fresh execution id",
                    tokens.execution_id,
                    wait,
                )
                await self._sleep(wait)
            except SandboxError as exc:
                if exc.error_type != ERROR_TOKEN_EXPIRED or attempt >= MAX_SUBMIT_ATTEMPTS:
                    raise
                # the credential died in the queue; re-mint and resubmit at once. Waiting
                # first would only spend more of the new one.
                logger.info(
                    "sandbox tokens expired while queued (execution=%s), re-minting",
                    tokens.execution_id,
                )

    def _validate(
        self,
        code: str,
        timeout_s: int,
        user: str,
        session_id: str,
        execution_id: str | None,
    ) -> None:
        # every caller-supplied value the body carries is checked here, so that a request the
        # supervisor is certain to 400 never leaves, and so that a caller catching SandboxError
        # sees these as refusals rather than as a bare ValueError out of the token minting.
        if not isinstance(code, str) or not code.strip():
            raise SandboxRejected("code must be a non-empty Python source string")
        if not isinstance(timeout_s, int) or isinstance(timeout_s, bool):
            raise SandboxRejected("timeout_s must be an integer number of seconds")
        if timeout_s < 1 or timeout_s > MAX_TIMEOUT_S:
            # rejected rather than clamped, for the same reason the supervisor rejects it: a
            # silent change on a model-influenceable path, and it would desync the deadlines.
            raise SandboxRejected(
                f"timeout_s must be between 1 and {MAX_TIMEOUT_S} seconds, got {timeout_s}"
            )
        code_bytes = len(code.encode("utf-8"))
        if code_bytes > MAX_CODE_BYTES:
            raise SandboxRejected(
                f"code is {code_bytes} bytes, over the {MAX_CODE_BYTES}-byte limit"
            )
        if not isinstance(user, str) or not user.strip():
            raise SandboxRejected("user must be the non-empty authenticated end-user identity")
        if not isinstance(session_id, str) or not session_id.strip():
            raise SandboxRejected("session_id must be a non-empty chat session id")
        # fullmatch, not match: Python's `$` also matches before a trailing newline, which the
        # contract's `^...$` does not mean and the supervisor's engine would not allow.
        if execution_id is not None and (
            not isinstance(execution_id, str) or not EXECUTION_ID_PATTERN.fullmatch(execution_id)
        ):
            raise SandboxRejected(
                "execution_id must be a lowercase-hex uuid4 in canonical form; it names the "
                "sandbox scratch directory"
            )

    async def _submit(
        self, tokens: SandboxTokens, *, code: str, timeout_s: int
    ) -> dict[str, Any]:
        # exactly the contract's fields and no others: unknown top-level fields are a 400,
        # so that a field added on one side and not the other fails on the first call.
        body = {
            "code": code,
            "execution_id": tokens.execution_id,
            "tokens": {
                DB_API_AUDIENCE: tokens.db_api,
                RESULTS_API_AUDIENCE: tokens.results_api,
            },
            "user": tokens.user,
            "session_id": tokens.session_id,
            "timeout_s": timeout_s,
        }
        # serialised here rather than by httpx so the 1 MiB cap is measured on the bytes that
        # actually go on the wire, which is where the supervisor measures it
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        if len(payload) > MAX_BODY_BYTES:
            raise SandboxRejected(
                f"request body is {len(payload)} bytes, over the {MAX_BODY_BYTES}-byte limit"
            )

        timeout = httpx.Timeout(
            CONNECT_TIMEOUT_S,
            connect=CONNECT_TIMEOUT_S,
            write=BODY_WRITE_DEADLINE_S,
            read=client_deadline_s(timeout_s),
        )
        try:
            async with self._client(timeout) as client:
                response = await client.post(
                    "/execute",
                    content=payload,
                    headers={"Content-Type": "application/json"},
                )
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise SandboxUnavailable(
                f"sandbox is not reachable at {self.base_url}; it may be restarting"
            ) from exc
        except httpx.TimeoutException as exc:
            raise SandboxDeadlineExceeded(
                f"sandbox did not answer within {client_deadline_s(timeout_s):.0f}s"
            ) from exc
        except httpx.TransportError as exc:
            raise SandboxUnavailable(
                f"sandbox connection failed at {self.base_url}: {type(exc).__name__}"
            ) from exc

        return self._interpret(response, tokens)

    def _interpret(self, response: httpx.Response, tokens: SandboxTokens) -> dict[str, Any]:
        status = response.status_code
        if status == 200:
            return self._result(response, tokens)

        error_type, raw_message = _error_fields(response)
        # scrubbed before `detail` is built and before it reaches the exception attribute: a
        # peer that echoes a token into `type` is exactly the peer `_redact` exists for.
        error_type = _redact(error_type, tokens)[:MAX_ERROR_TYPE_CHARS] if error_type else None
        message = _redact(raw_message, tokens)[:MAX_ERROR_MESSAGE_CHARS]
        detail = f"{error_type or 'unknown'}: {message}" if message else (error_type or "unknown")

        if status == 429:
            raise SandboxBusy(
                f"sandbox queue is full ({detail})",
                status_code=status,
                error_type=error_type or ERROR_BUSY,
                retry_after=_retry_after(response),
            )
        if status in (502, 503, 504):
            raise SandboxUnavailable(
                f"sandbox is not accepting executions ({detail})",
                status_code=status,
                error_type=error_type or (ERROR_NOT_READY if status == 503 else None),
            )
        if 400 <= status < 500:
            # includes both 409s. They are distinguished by error.type, never by the status
            # code: TokenExpired is retryable and handled by the caller loop, while
            # DuplicateExecutionId means the id was already spent and needs a fresh one.
            raise SandboxRejected(
                f"sandbox refused the request ({detail})",
                status_code=status,
                error_type=error_type,
            )
        if 500 <= status < 600:
            raise SandboxInternalError(
                f"sandbox supervisor failed ({detail})", status_code=status, error_type=error_type
            )
        raise SandboxProtocolError(
            f"sandbox answered an unexpected HTTP {status}", status_code=status
        )

    def _result(self, response: httpx.Response, tokens: SandboxTokens) -> dict[str, Any]:
        try:
            body = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise SandboxProtocolError("sandbox returned a body that is not JSON") from exc
        if not isinstance(body, dict):
            raise SandboxProtocolError("sandbox returned a JSON body that is not an object")
        if "status" not in body:
            raise SandboxProtocolError("sandbox result carries no status field")
        returned = body.get("execution_id")
        if returned != tokens.execution_id:
            # the id is the join key between chat-backend's manifest record, the audit stream
            # and db-api's access log. A result carrying a different one cannot be recorded.
            raise SandboxProtocolError(
                "sandbox result echoed a different execution_id than the one submitted"
            )
        logger.info(
            "sandbox execution %s finished: status=%s duration_ms=%s artifacts=%s",
            tokens.execution_id,
            body.get("status"),
            body.get("duration_ms"),
            len(body["artifacts"]) if isinstance(body.get("artifacts"), list) else "?",
        )
        return body
