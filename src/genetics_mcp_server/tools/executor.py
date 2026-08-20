"""Tool executor - handles HTTP calls to genetics API and external services."""

import asyncio
import base64
import inspect
import io
import json
import logging
import mimetypes
import os
import re
import stat
import threading
import traceback
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, urlencode
from xml.sax.saxutils import quoteattr

import httpx
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for server use
import matplotlib.pyplot as plt

from genetics_mcp_server.tools.phewas_categories import (
    categorize_phenotype,
    get_category_color,
)
from genetics_mcp_server.tools.sql_safety import (
    SqlValueError,
    normalize_literal,
    quote_literal,
    quote_literal_list,
    sql_float,
    sql_int,
)
from genetics_mcp_server.tools.uniprot import UniProtClient

if TYPE_CHECKING:
    from genetics_mcp_server.config.settings import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _PrunedInstallSettings:
    """Stand-in for Settings where config/settings.py is not installed.

    Exactly one install is like that: the sandbox image, which ships only the SDK's
    import closure because config/settings.py names the whole internal configuration
    surface (genetics-results-suite-l41). The values are Settings' own defaults rather
    than environment reads — the sandbox holds no internal secret by design, and reading
    the variables here would put their names back into the image this is removing them
    from. tests/test_sdk_import_closure.py asserts these agree with Settings' defaults.
    """

    internal_api_secret: str = ""
    myvariant_api_url: str = "https://myvariant.info/v1"
    uniprot_api_url: str = "https://rest.uniprot.org"
    ebi_proteins_api_url: str = "https://www.ebi.ac.uk/proteins/api"
    uniprot_cache_ttl: int = 86400


_PRUNED_INSTALL_SETTINGS = _PrunedInstallSettings()


_warned_pruned_install = False


def _resolve_settings() -> "Settings | _PrunedInstallSettings":
    """Settings, resolved at first use rather than at ToolExecutor construction.

    Deferred so that importing — or constructing — the executor never requires
    config/settings.py, which the sandbox image deliberately does not ship.

    Only config/settings.py itself going missing is the pruned install; a
    ModuleNotFoundError from anywhere else in its import chain (today python-dotenv,
    tomorrow any new optional dependency) is a broken install and must not degrade
    silently into the credential-less fallback.
    """
    global _warned_pruned_install
    try:
        from genetics_mcp_server.config.settings import get_settings
    except ModuleNotFoundError as exc:
        if exc.name not in (
            "genetics_mcp_server.config",
            "genetics_mcp_server.config.settings",
        ):
            raise
        if not _warned_pruned_install:
            _warned_pruned_install = True
            # once, not per call: in the sandbox this is the expected path
            logger.warning(
                "settings module is not installed; using the pruned-install defaults "
                "(no internal credentials, upstream endpoints at their public defaults)"
            )
        return _PRUNED_INSTALL_SETTINGS
    return get_settings()


def _endpoint_env(name: str, default: str | None = None) -> str | None:
    """An endpoint URL from the environment, read only after settings has had its chance.

    config/settings.py calls load_dotenv() at module scope, so a standalone process with
    a .env file sees these variables only once that module has been imported. These
    reads used to sit in __init__ *after* the settings import that has since been
    deferred; routing them through _resolve_settings() keeps them on the far side of
    load_dotenv() without putting the import back into construction. In the pruned
    install there is no settings module and the reads fall through to the environment
    the sandbox itself set.
    """
    _resolve_settings()
    return os.environ.get(name, default)


# --------------------------------------------------------------------------------------
# The per-execution sandbox credential (genetics-results-suite-4h6.44)
# --------------------------------------------------------------------------------------

# The supervisor writes /scratch/<execution_id>/tokens.json before it forks and names the
# PATH — never the tokens — to the child under this variable (genetics-results-suite
# sandbox/supervisor.py, ENV_TOKEN_FILE / TOKEN_FILE_NAME). The file is a JSON object
# keyed by audience: {"db-api": "<jws>", "results-api": "<jws>"}.
SANDBOX_TOKEN_FILE_ENV = "SANDBOX_TOKEN_FILE"
DB_API_AUDIENCE = "db-api"
RESULTS_API_AUDIENCE = "results-api"
SANDBOX_TOKEN_AUDIENCES = (DB_API_AUDIENCE, RESULTS_API_AUDIENCE)

# a token pair is a few hundred bytes; anything near this is not the supervisor's file
_SANDBOX_TOKEN_FILE_MAX_BYTES = 64 * 1024

_sandbox_tokens_lock = threading.Lock()
_sandbox_tokens: dict[str, str] | None = None
_sandbox_tokens_error: Exception | None = None
_sandbox_tokens_loaded = False


class SandboxCredentialError(RuntimeError):
    """The SDK could not obtain the per-execution token pair it was supposed to have.

    Either SANDBOX_TOKEN_FILE named a file that did not yield a usable pair, or a pruned
    install — the sandbox image, and only ever that — reached client construction with no
    token file and no INTERNAL_API_SECRET.
    """


def _reset_sandbox_tokens() -> None:
    """Test seam. The read is once-per-process by design; tests need to repeat it."""
    global _sandbox_tokens, _sandbox_tokens_error, _sandbox_tokens_loaded
    with _sandbox_tokens_lock:
        _sandbox_tokens = None
        _sandbox_tokens_error = None
        _sandbox_tokens_loaded = False


def _read_and_unlink(path: str) -> bytes:
    """Read the whole file once, then unlink it whether or not the read succeeded.

    O_NOFOLLOW for the same reason the supervisor writes with it: /scratch is writable by
    everything running under the shared uid, so a symlink planted at the path would
    otherwise redirect this read. The unlink is in a `finally` because a file left behind
    after a *failed read* is the worst of both outcomes.

    That `finally` covers reads, not opens: a path that cannot be OPENED at all — a
    symlink (ELOOP, by design) or a mode-000 file (EACCES) — raises before the `try` and
    is therefore left on disk. Nil impact in the pod, where the supervisor writes the file
    itself with O_EXCL|O_NOFOLLOW 0600 and reaps the whole /scratch/<execution_id>
    directory afterwards; the case exists only when something else has already put a file
    the SDK did not write at that path, which is not a state this function can improve.
    """
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        chunks: list[bytes] = []
        size = 0
        while size <= _SANDBOX_TOKEN_FILE_MAX_BYTES:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
        if size > _SANDBOX_TOKEN_FILE_MAX_BYTES:
            raise SandboxCredentialError(
                f"{path} is larger than {_SANDBOX_TOKEN_FILE_MAX_BYTES} bytes; "
                "that is not the supervisor's token file"
            )
        return b"".join(chunks)
    finally:
        try:
            os.close(fd)
        finally:
            try:
                os.unlink(path)
            except OSError as exc:
                # the credential outliving this call is worth a line even though the
                # execution can continue
                logger.warning("could not unlink %s: %s", path, exc)


def _load_sandbox_tokens() -> dict[str, str] | None:
    """The per-execution tokens, read ONCE and unlinked, or None outside the sandbox.

    READ-ONCE-AND-UNLINK IS NOT AN EXPOSURE BOUND and must not be described as one. The
    child is forked without exec from a supervisor that holds the tokens in its address
    space, and a raw /proc/self/mem scan in the child was measured to recover them —
    including from an execution that had already completed; a detached grandchild of an
    earlier execution was measured reading a live token file inside this very window.
    What bounds the exposure is genetics-results-suite-4h6.55 and nothing here.

    NEITHER IS ANY OF THIS A CONTROL OVER THE SCRIPT. The child is forked without exec and
    owns its own os.environ, so both behaviours below are conditioned on inputs the script
    can rewrite before the SDK's first use, and both were measured being rewritten:

    * Unlinking is the half of the delivery contract this side owns, and it happens only on
      the branch where SANDBOX_TOKEN_FILE is set. `os.environ.pop("SANDBOX_TOKEN_FILE")`
      before the first SDK call leaves the file on disk for the whole execution, removed
      only by the supervisor's reap. That widens a window this function would otherwise
      have closed in microseconds — but the same script could read the file itself, so it
      is hygiene against accident and misconfiguration, not containment.
    * Raising rather than degrading to no credential: SANDBOX_TOKEN_FILE being set is the
      sandbox saying it minted tokens, so failing to use them is a misconfiguration whose
      only other symptom is a bare 401 from a service that names no cause. Unsetting it
      instead takes the None branch, which is why `_build_client` refuses to build an
      uncredentialed client in a pruned install: the sandbox image is the one install where
      "no token and no secret" can never be legitimate. Note what the child still controls —
      the warning that accompanies any of this goes to this process's logger, i.e. into the
      execution's OWN captured stdout, which is returned to the model rather than to the
      pod log.
    """
    global _sandbox_tokens, _sandbox_tokens_error, _sandbox_tokens_loaded
    if _sandbox_tokens_loaded:
        if _sandbox_tokens_error is not None:
            raise _sandbox_tokens_error
        return _sandbox_tokens
    with _sandbox_tokens_lock:
        if _sandbox_tokens_loaded:
            if _sandbox_tokens_error is not None:
                raise _sandbox_tokens_error
            return _sandbox_tokens
        path = os.environ.get(SANDBOX_TOKEN_FILE_ENV, "").strip()
        tokens: dict[str, str] | None = None
        error: Exception | None = None
        if path:
            try:
                tokens = _parse_sandbox_tokens(_read_and_unlink(path), path)
            except SandboxCredentialError as exc:
                error = exc
            except OSError as exc:
                error = SandboxCredentialError(
                    f"{SANDBOX_TOKEN_FILE_ENV}={path} could not be read: {exc}"
                )
        _sandbox_tokens = tokens
        _sandbox_tokens_error = error
        _sandbox_tokens_loaded = True
    if error is not None:
        raise error
    return tokens


def _parse_sandbox_tokens(raw: bytes, path: str) -> dict[str, str]:
    try:
        body = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise SandboxCredentialError(f"{path} is not decodable JSON: {exc}") from None
    if not isinstance(body, dict):
        raise SandboxCredentialError(f"{path} is not a JSON object")
    tokens = {}
    for audience in SANDBOX_TOKEN_AUDIENCES:
        token = body.get(audience)
        if not isinstance(token, str) or not token:
            raise SandboxCredentialError(
                f"{path} carries no usable {audience!r} token; the pair is audience-bound "
                "and a cross-audience token is a hard 401 at both validators"
            )
        tokens[audience] = token
    return tokens


def _origin(url: httpx.URL) -> tuple[str, str, int | None]:
    return (url.scheme, url.host, url.port)


def _origin_display(url: httpx.URL) -> str:
    """scheme://host[:port], rebuilt from the parts rather than by blanking the URL.

    `url.copy_with(query=None, fragment=None, raw_path=b"/")` looks equivalent and is not:
    it RETAINS userinfo, so a caller-supplied password lands on a log line
    (http://user:hunter2@evil.test/collect -> "http://user:hunter2@evil.test/"). Not a
    sandbox credential, but not ours to log either.
    """
    scheme, host, port = _origin(url)
    if ":" in host:  # IPv6 literal; URL.host returns it unbracketed
        host = f"[{host}]"
    return f"{scheme}://{host}" if port is None else f"{scheme}://{host}:{port}"


class _SandboxTokenAuth(httpx.Auth):
    """Attaches the per-execution token BOUND TO THE DESTINATION of each request.

    One httpx client serves both upstreams, and the two tokens are not interchangeable:
    `aud` is validated exactly at genetics-results-db `api/sandbox_auth.py` and
    genetics-results-api `app/core/sandbox_token.py`, both of which additionally refuse a
    list-valued `aud`, so a token sent to the wrong service is a hard 401 rather than a
    degraded success. A default header on the client cannot express that, which is why
    this is per-request.

    A request to any other destination gets NO credential. That is hygiene against an
    accidental or misconfigured base URL, NOT a control over the script: the child is
    forked without exec and owns os.environ, while `base_url` and `bigquery_url` are
    cached_property reads of GENETICS_API_URL / BIGQUERY_API_URL resolved on FIRST USE and
    `sdk/__init__.py` holds `_client = None` until the first call — so a script that sets
    GENETICS_API_URL=http://evil.attacker.test/api before its first SDK call makes that
    host the matching destination and is handed the results-api token, as measured. The
    same script can read the token out of /proc/self/mem regardless
    (genetics-results-suite-4h6.55), so nothing here can be hardened into containment;
    what stops the request is the sandbox's deny-by-default egress allow-list.
    """

    def __init__(self, tokens: dict[str, str], destinations: list[tuple[str, str]]):
        self._tokens = tokens
        # longest path prefix first so two upstreams sharing an origin still resolve
        self._destinations = sorted(
            ((httpx.URL(base), audience) for base, audience in destinations),
            key=lambda item: len(str(item[0].path).rstrip("/")),
            reverse=True,
        )
        self._warned: set[str] = set()

    def audience_for(self, url: httpx.URL) -> str | None:
        """The audience whose base URL this request falls under, or None.

        The prefix match is on whole path SEGMENTS. A bare `startswith` makes `/api/bq` a
        prefix of `/api/bqx`, so with the two upstreams co-located on one origin —
        GENETICS_API_URL=http://svc:4000/api, BIGQUERY_API_URL=http://svc:4000/api/bq —
        a request to /api/bqx/steal would be handed the db-api token, which is the wrong
        audience, i.e. a token sent somewhere its base URL never said it could go. Not
        reachable in the dev stack or the cluster, where the two are distinct hosts; the
        `sorted()` above exists precisely to keep the co-located case working, so the
        matching rule has to actually support it.
        """
        path = str(url.path)
        for base, audience in self._destinations:
            if _origin(url) != _origin(base):
                continue
            prefix = str(base.path).rstrip("/")
            if path == prefix or path.startswith(f"{prefix}/"):
                return audience
        return None

    def auth_flow(self, request):
        audience = self.audience_for(request.url)
        if audience is None:
            origin = _origin_display(request.url)
            if origin not in self._warned:
                self._warned.add(origin)
                logger.warning(
                    "no per-execution token matches %s; sending it with no Authorization "
                    "header. The sandbox token is bound to db-api and results-api only.",
                    origin,
                )
        else:
            request.headers["Authorization"] = f"Bearer {self._tokens[audience]}"
        yield request


# distinguishes "caller did not ask for a row cap" from "caller asked for no cap at all",
# since None is a meaningful value for _row_limit
_KEEP_DEFAULT_ROW_LIMIT = object()


def _seg(value: Any) -> str:
    """Percent-encode a caller-supplied value for use as a URL *path segment*.

    Every by-gene/by-variant/by-region endpoint puts a caller string straight into the
    path of a request that carries the internal bearer token. Unencoded, `../../admin/users`
    resolves to a different endpoint entirely (httpx normalises `..`) and `x?a=b#c` appends
    an attacker-controlled query string — so an unvalidated segment defeats the typed
    surface. safe="" encodes `/`, `?`, `#` and everything else reserved; the bare-dot
    segments are encoded explicitly because `.` is unreserved and would survive quote().
    """
    encoded = quote(str(value), safe="")
    if encoded in (".", ".."):
        return encoded.replace(".", "%2E")
    return encoded

# generic error message returned to clients
INTERNAL_ERROR_MSG = "Internal server error. Check server logs for details."

# SANDBOX_ARTIFACTS_DIR must resolve under this prefix or read_artifact refuses. Without it
# the only thing between this code and behaviour docs/code-execution-security.md forbids is
# an env var staying unset: read_artifact is registered in the chat backend, so setting
# SANDBOX_ARTIFACTS_DIR=/data there would make chat_history.db and llm_config.db readable
# and base64'd back to the model. chat-backend has no /scratch volume and never will, so
# hardcoding the prefix makes that misconfiguration unreachable rather than merely unmade.
# Tests patch this to a temp path; nothing else may.
_ARTIFACTS_DIR_PREFIX = "/scratch/"

# returned when an upstream service (genetics API / BigQuery db) can't be connected to,
# as opposed to a genuine internal error — lets callers and the UI show something actionable
UPSTREAM_UNREACHABLE_MSG = (
    "The genetics data service is currently unreachable (connection failed). "
    "It may be down or restarting — please try again shortly."
)
# marker header on the synthetic response so callers can distinguish 'unreachable'
# from a real upstream 503
_UNREACHABLE_HEADER = "x-fg-upstream-unreachable"

# variant classifications for counting coding and loss-of-function variants
CODING_VARIANTS = {
    "missense_variant",
    "frameshift_variant",
    "inframe_insertion",
    "inframe_deletion",
    "transcript_ablation",
    "stop_gained",
    "stop_lost",
    "start_lost",
    "splice_acceptor_variant",
    "splice_donor_variant",
    "incomplete_terminal_codon_variant",
    "protein_altering_variant",
    "coding_sequence_variant",
}

LOF_VARIANTS = {
    "frameshift_variant",
    "stop_gained",
    "stop_lost",
    "start_lost",
    "splice_acceptor_variant",
    "splice_donor_variant",
    "transcript_ablation",
}


class _ResilientAsyncClient(httpx.AsyncClient):
    """httpx client that converts upstream connection failures into a synthetic 503
    response carrying a clear message. All get/post/etc. funnel through request(), so
    every tool surfaces 'service unreachable' instead of an opaque internal error (or an
    uncaught exception in methods without a try/except) when the API/db is simply down."""

    async def request(self, method, url, **kwargs):  # type: ignore[override]
        try:
            return await super().request(method, url, **kwargs)
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            logger.warning(f"upstream unreachable: {method} {url}: {e}")
            return httpx.Response(
                status_code=503,
                text=UPSTREAM_UNREACHABLE_MSG,
                headers={_UNREACHABLE_HEADER: "1"},
                request=httpx.Request(method, url),
            )


class ToolExecutor:
    """Executes MCP tools by making HTTP calls to the genetics API."""

    def __init__(
        self,
        api_base_url: str | None = None,
        public_api_url: str | None = None,
        bigquery_api_url: str | None = None,
        row_limit: Any = _KEEP_DEFAULT_ROW_LIMIT,
        expose_columns: bool = False,
    ):
        self._client_lock = threading.Lock()
        self._api_base_url_arg = api_base_url
        self._public_api_url_arg = public_api_url
        self._bigquery_api_url_arg = bigquery_api_url
        # separate client for third-party calls: carries no default auth so the
        # internal API secret is never leaked to external services (e.g. MouseMine,
        # myvariant.info). Per-call auth (Perplexity, Tavily) is passed explicitly.
        self.external_client = _ResilientAsyncClient(timeout=30.0)
        # lazily-fetched universe of expression resources (e.g. gtex, hpa), used to
        # tell "gene absent from this resource" apart from "resource unavailable"
        self._expression_resources: list[str] | None = None
        # lazily-fetched cBioPortal metadata. The study and profile lists change
        # only when cBioPortal reimports data, and the cancer-type denominators do
        # not depend on the gene being queried, so all three are cached for the
        # process lifetime rather than refetched per call.
        self._cbio_studies: list[dict[str, Any]] | None = None
        self._cbio_profiles: list[dict[str, Any]] | None = None
        self._cbio_denominators: dict[str, dict[str, Any]] | None = None
        # inline row cap; None disables capping (see _cap_rows)
        self._row_limit: int | None = (
            self._REGION_ROW_LIMIT if row_limit is _KEEP_DEFAULT_ROW_LIMIT else row_limit
        )
        # off by default: a tool result dict IS the MCP tool payload and the chat
        # backend's model input, and this epic forbids changing either. Only the SDK
        # (which builds its own executor) asks for it. See _columns_meta.
        self._expose_columns = expose_columns

    def _columns_meta(self, resp: httpx.Response) -> dict[str, list[str]]:
        """The results-api column names for this response, when it advertised them.

        results-api's JSON range responses are bare arrays, so an EMPTY one carries no
        schema at all and a client cannot build a named frame from it — the SDK raised
        ColumnNotFoundError on a no-hit query. It now advertises the file's own header in
        an `X-Columns` response header (genetics-results-api app/core/responses.py), which
        is populated even when zero rows matched.

        Returned as a dict to splice with `**` so that, when this is off or the endpoint
        does not advertise, the result dict is byte-identical to before. `column_names`
        rather than `columns`: db-api's `columns` is REQUIRED to read its positional rows,
        while these rows are already named dicts and this list matters only when there are
        none. Conflating them would put dict rows down the positional constructor.
        """
        if not self._expose_columns:
            return {}
        raw = resp.headers.get("X-Columns")
        return {"column_names": raw.split(",")} if raw else {}

    @cached_property
    def base_url(self) -> str:
        """The internal results-api endpoint, resolved on first use (see _endpoint_env)."""
        return self._api_base_url_arg or _endpoint_env(
            "GENETICS_API_URL", "http://0.0.0.0:2000/api"
        )

    @cached_property
    def public_url(self) -> str:
        """Public URL for download links shown to users."""
        return self._public_api_url_arg or _endpoint_env(
            "GENETICS_PUBLIC_API_URL", self.base_url
        )

    @cached_property
    def bigquery_url(self) -> str | None:
        """BigQuery API URL for direct SQL queries; None disables the SQL tools."""
        return self._bigquery_api_url_arg or _endpoint_env("BIGQUERY_API_URL")

    @property
    def client(self) -> _ResilientAsyncClient:
        """The internal API client, built on first use.

        Lazy because its Authorization header is the only reason __init__ needed
        settings; deferring it keeps construction free of config/settings.py. Assigning
        `executor.client = ...` still works — the setter below writes the instance dict,
        which is also where close() looks.

        Locked rather than a cached_property because functools dropped that descriptor's
        per-instance lock in 3.12: two threads racing the first access would each build a
        client, and only the winner's connection pool is the one close() ever sees. The
        service holds one shared executor across threads (mcp_server.py).
        """
        client = self.__dict__.get("client")
        if client is None:
            with self._client_lock:
                client = self.__dict__.get("client")
                if client is None:
                    client = self._build_client()
                    self.__dict__["client"] = client
        return client

    def _build_client(self) -> _ResilientAsyncClient:
        """One client, credentialled from the per-execution tokens when they exist.

        The sandbox path and the service path are mutually exclusive by design: an
        execution's token carries the real end user's `sub`, `sid` and `jti`, and those
        three are what results-api's per-execution counters are keyed on
        (app/core/sandbox_budget.py). INTERNAL_API_SECRET satisfies `is_internal_caller`
        and therefore reaches every handler while `_sandbox_principal` resolves nothing, so
        a sandbox request carrying it is served with NO ACCOUNTING AT ALL — that is
        genetics-results-suite-0lf, and it is why attaching both, or preferring the secret,
        would silently re-open it.

        db-api meters differently but on the same key: it has no request counter and no
        in-flight gate, and `_caps_for` (genetics-results-db api/main.py) grants the
        *relaxed* row and byte ceilings to the INTERNAL_API_SECRET principal with no `jti`,
        so the 200 GB `SANDBOX_AGGREGATE_BYTES_BUDGET` charged by `_charge_aggregate` is
        keyed on the execution id this token carries and is charged for no other caller.
        `genetics.sql()` is therefore metered by bytes, not by request count.
        """
        tokens = _load_sandbox_tokens()
        if tokens is not None:
            destinations = [(self.base_url, RESULTS_API_AUDIENCE)]
            if self.bigquery_url:
                destinations.append((self.bigquery_url, DB_API_AUDIENCE))
            return _ResilientAsyncClient(
                timeout=300.0, auth=_SandboxTokenAuth(tokens, destinations)
            )

        settings = _resolve_settings()
        api_secret = settings.internal_api_secret
        if not api_secret:
            if settings is _PRUNED_INSTALL_SETTINGS:
                # The pruned install is the sandbox image and nothing else (see
                # _PrunedInstallSettings), its `internal_api_secret` is "" by construction,
                # and the supervisor is the only thing that sets SANDBOX_TOKEN_FILE. So this
                # combination is never a legitimate state, which makes it the one place the
                # uncredentialed fallback can be refused outright instead of warned about.
                # Hygiene, NOT a control: the child owns os.environ and can unset the
                # variable to reach this line, but it can equally mint its own file — what
                # this buys is that an accident or a supervisor bug fails loudly here rather
                # than as a bare 401 from a service that names no cause.
                raise SandboxCredentialError(
                    f"{SANDBOX_TOKEN_FILE_ENV} is unset and no internal credential is "
                    "installed; a pruned (sandbox) install has no other way to authenticate "
                    "and must not fall back to sending requests unauthenticated"
                )
            # NOT raising outside the pruned install, and not silent either
            # (genetics-results-suite-618). Raising here would break a local run against an
            # unauthenticated results-api, which README documents as supported; the deployed
            # entrypoints call config.require_internal_api_secret() at startup so a
            # pod in this state never reaches this line. What is left is a
            # developer's own machine, where a bare 401 from results-api is the only
            # other symptom and does not name the cause.
            logger.warning(
                "no credential: %s names no per-execution token file and "
                "INTERNAL_API_SECRET is unset. Calls to %s and %s will be sent with no "
                "Authorization header and will be refused by any deployment that requires "
                "authentication",
                SANDBOX_TOKEN_FILE_ENV,
                self.base_url,
                self.bigquery_url or "the BigQuery API",
            )
        headers = {"Authorization": f"Bearer {api_secret}"} if api_secret else {}
        return _ResilientAsyncClient(timeout=300.0, headers=headers)

    @client.setter
    def client(self, value: _ResilientAsyncClient) -> None:
        self.__dict__["client"] = value

    @cached_property
    def uniprot(self) -> UniProtClient:
        """Shares external_client so UniProt/EBI outages arrive as the synthetic 503
        rather than raising, and so no internal auth header is ever sent to them."""
        return UniProtClient(self.external_client, _resolve_settings())

    # -------------------------------------------------------------------------
    # myvariant.info HGVS conversion
    # -------------------------------------------------------------------------

    @staticmethod
    def _variant_to_hgvs(variant_id: str) -> str:
        """Convert chr:pos:ref:alt to myvariant.info HGVS notation.

        Handles SNVs (chr1:g.12345A>G), deletions (chr1:g.12345_12347del),
        insertions (chr1:g.12345_12346insACG), and delins/MNVs
        (chr1:g.12345_12347delinsACG).
        """
        parts = re.split(r"[:|\-_]", variant_id, maxsplit=3)
        if len(parts) != 4:
            raise ValueError(f"Invalid variant format: {variant_id}")

        chrom, pos_str, ref, alt = parts
        pos = int(pos_str)

        # normalize chromosome
        chrom_str = chrom if chrom.startswith("chr") else f"chr{chrom}"

        if len(ref) == 1 and len(alt) == 1:
            # SNV
            return f"{chrom_str}:g.{pos}{ref}>{alt}"
        elif len(alt) == 0 or alt == "-":
            # pure deletion (no alt allele)
            if len(ref) == 1:
                return f"{chrom_str}:g.{pos}del"
            else:
                return f"{chrom_str}:g.{pos}_{pos + len(ref) - 1}del"
        elif len(ref) == 0 or ref == "-":
            # pure insertion (no ref allele)
            return f"{chrom_str}:g.{pos}_{pos + 1}ins{alt}"
        elif len(ref) > len(alt) and alt == ref[:len(alt)]:
            # deletion (ref starts with alt, remaining bases are deleted)
            del_start = pos + len(alt)
            del_end = pos + len(ref) - 1
            if del_start == del_end:
                return f"{chrom_str}:g.{del_start}del"
            else:
                return f"{chrom_str}:g.{del_start}_{del_end}del"
        elif len(alt) > len(ref) and alt[:len(ref)] == ref:
            # insertion (alt starts with ref, remaining bases are inserted)
            ins_after = pos + len(ref) - 1
            return f"{chrom_str}:g.{ins_after}_{ins_after + 1}ins{alt[len(ref):]}"
        else:
            # MNV or complex substitution → delins
            if len(ref) == 1:
                return f"{chrom_str}:g.{pos}delins{alt}"
            else:
                return f"{chrom_str}:g.{pos}_{pos + len(ref) - 1}delins{alt}"

    @staticmethod
    def _flatten_myvariant_result(data: dict[str, Any]) -> dict[str, Any]:
        """Extract key clinical/functional fields from a myvariant.info response."""
        result: dict[str, Any] = {}

        # ClinVar
        clinvar = data.get("clinvar")
        if clinvar:
            rcv = clinvar.get("rcv")
            if isinstance(rcv, list):
                significances = list({r.get("clinical_significance") for r in rcv if r.get("clinical_significance")})
                conditions = list({r.get("preferred_name") or r.get("conditions", {}).get("name", "") for r in rcv if r.get("preferred_name") or r.get("conditions")})
            elif isinstance(rcv, dict):
                significances = [rcv.get("clinical_significance")] if rcv.get("clinical_significance") else []
                cond = rcv.get("preferred_name") or (rcv.get("conditions", {}).get("name", "") if isinstance(rcv.get("conditions"), dict) else "")
                conditions = [cond] if cond else []
            else:
                significances = []
                conditions = []

            result["clinvar"] = {
                "clinical_significance": significances,
                "conditions": conditions,
                "review_status": clinvar.get("review", {}).get("review_status") if isinstance(clinvar.get("review"), dict) else None,
                "variant_id": clinvar.get("variant_id"),
            }

        # CADD
        cadd = data.get("cadd")
        if cadd:
            result["cadd"] = {
                "phred": cadd.get("phred"),
                "raw_score": cadd.get("rawscore"),
                "consequence": cadd.get("consequence"),
            }

        # dbNSFP functional predictions
        dbnsfp = data.get("dbnsfp")
        if dbnsfp:
            preds: dict[str, Any] = {}
            for predictor in ("sift", "polyphen2", "mutationtaster", "metalr", "metasvm", "fathmm"):
                pred_data = dbnsfp.get(predictor)
                if pred_data and isinstance(pred_data, dict):
                    entry: dict[str, Any] = {}
                    if "score" in pred_data:
                        entry["score"] = pred_data["score"]
                    if "pred" in pred_data:
                        entry["prediction"] = pred_data["pred"]
                    if "converted_rankscore" in pred_data:
                        entry["rankscore"] = pred_data["converted_rankscore"]
                    if entry:
                        preds[predictor] = entry
            if preds:
                result["functional_predictions"] = preds
            if dbnsfp.get("genename"):
                result["gene"] = dbnsfp["genename"]

        # COSMIC
        cosmic = data.get("cosmic")
        if cosmic:
            result["cosmic"] = {
                "cosmic_id": cosmic.get("cosmic_id"),
                "tumor_site": cosmic.get("tumor_site"),
            }

        # CIViC
        civic = data.get("civic")
        if civic:
            result["civic"] = {
                "variant_id": civic.get("variant_id"),
                "name": civic.get("name"),
                "gene": civic.get("entrez_name"),
            }

        # dbSNP
        dbsnp = data.get("dbsnp")
        if dbsnp:
            result["rsid"] = dbsnp.get("rsid")

        return result

    def _build_download_url(self, path: str, params: dict[str, Any] | None = None) -> str:
        """Build a public download URL for a genetics API endpoint."""
        url = f"{self.public_url}{path}"
        query = {"format": "tsv"}
        if params:
            query.update(params)
        return f"{url}?{urlencode(query)}"

    # a region query is unbounded by nature: a wide window can return tens of thousands
    # of rows, which is a table rather than an answer. Cap what goes into the context and
    # leave the full result behind the download URL.
    _REGION_ROW_LIMIT = 500

    def _cap_rows(self, rows: list, limit: int | None = None) -> tuple[list, bool]:
        """Cap inline rows, returning (rows, truncated).

        The cap protects the model's context, not the data — a caller that consumes rows
        programmatically (the SDK) sets `_row_limit = None` to receive the whole result set,
        while the tool surface keeps the default.
        """
        if limit is None:
            limit = self._row_limit
        if limit is None:
            return list(rows), False
        return rows[:limit], len(rows) > limit

    @staticmethod
    def _resources_param(resources: str | None) -> list[str] | None:
        """Split a comma-separated resource string into the repeated query params the API expects."""
        if not resources:
            return None
        return [r.strip() for r in resources.split(",") if r.strip()]

    async def close(self):
        """Close the HTTP clients.

        `client` is read out of the instance dict rather than through the attribute so
        closing an executor that never made an internal call does not build one first.
        """
        client = self.__dict__.get("client")
        if client is not None:
            await client.aclose()
        await self.external_client.aclose()

    # -------------------------------------------------------------------------
    # BigQuery Tools
    # -------------------------------------------------------------------------

    # bounds for values rendered into server-built SQL. The window bound also stops a caller
    # turning a gene-window scan into a whole-genome scan.
    _MAX_SQL_WINDOW = 10_000_000
    _MAX_SQL_LIMIT = 100_000

    @staticmethod
    def _query_metadata(
        payload: dict[str, Any], query_result: dict[str, Any], include: bool
    ) -> dict[str, Any]:
        """Optionally attach the column names and truncation flag of the underlying query.

        `results` already carries the names on every row, but an EMPTY result has no row
        to carry them: the SDK builds `pl.DataFrame({c: [] for c in columns})` so a script
        filtering a no-hit gene gets an empty frame rather than ColumnNotFound. `truncated`
        stays because silent truncation is the one failure a script cannot detect for
        itself. Off by default so the model's payload is not padded with either.
        """
        if include:
            payload["columns"] = query_result.get("columns", [])
            payload["truncated"] = query_result.get("truncated", False)
        return payload

    @staticmethod
    def _positional_rows(
        query_result: dict[str, Any],
    ) -> tuple[list[str], list[Any], dict[str, Any] | None]:
        """Split a db-api query result into column names and positional rows.

        Returns `(columns, rows, error)`; `error` is a ready-to-return failure payload
        when the two disagree, and None otherwise. `zip` truncates silently, so a row
        whose arity disagrees with `columns` must be an error rather than a shifted
        labelling: mislabelled genomic values are worse than no values. A row arriving
        as a dict is the same class of failure — zipping names against dict KEYS would
        produce plausible-looking garbage.
        """
        columns = query_result.get("columns") or []
        rows = query_result.get("rows") or []
        if rows and not columns:
            return columns, rows, {
                "success": False,
                "error": "query result has rows but no column names",
            }
        for row in rows:
            if not isinstance(row, (list, tuple)):
                return columns, rows, {
                    "success": False,
                    "error": f"query result row is {type(row).__name__}, expected a positional list",
                }
            if len(row) != len(columns):
                return columns, rows, {
                    "success": False,
                    "error": (
                        f"query result row has {len(row)} values but "
                        f"{len(columns)} column names"
                    ),
                }
        return columns, rows, None

    def _bq_gene_payload(
        self,
        gene: str,
        query_result: dict[str, Any],
        filename: str,
        with_metadata: bool,
    ) -> dict[str, Any]:
        """Shape a BigQuery by-gene query result for its two consumers.

        db-api returns rows POSITIONALLY (a list per row, names in a separate `columns`
        key), and the two consumers want opposite things with that:

        - `results` goes to the model, which cannot interpret `["19", 44908822, 12.3]`,
          so rows become dicts. This set is capped, so zipping is cheap.
        - `_download_data` keeps the positional `{columns, rows}` form that
          `_convert_to_tsv` handles directly — the download can carry 100k rows, and
          re-materialising each as a dict only to flatten it back is waste.
        """
        columns, rows, error = self._positional_rows(query_result)
        if error:
            return error
        payload = {
            "success": True,
            "gene": gene,
            "results": [dict(zip(columns, row)) for row in rows],
            "_download_data": {"columns": columns, "rows": rows, "filename": filename},
        }
        return self._query_metadata(payload, query_result, with_metadata)

    @staticmethod
    def _gene_window_cte(gene_literal: str) -> str:
        """CTE resolving a gene symbol to its chromosome and span.

        `gene_literal` must already have come through sql_safety.quote_literal.
        """
        return (
            f"WITH g AS ("
            f"  SELECT chr, MIN(gene_start) AS gstart, MAX(gene_end) AS gend"
            f"  FROM gene_annotations_v WHERE symbol = {gene_literal} GROUP BY chr"
            f") "
        )

    @staticmethod
    def _strip_trailing_limit(sql: str) -> tuple[str, bool]:
        """Strip a trailing LIMIT clause (and optional semicolon) from SQL. Returns (sql, was_stripped)."""
        stripped = re.sub(r'\bLIMIT\s+\d+\s*;?\s*$', '', sql, flags=re.IGNORECASE).strip()
        # also strip a bare trailing semicolon
        stripped = stripped.rstrip(';').rstrip()
        return stripped, stripped != sql.strip().rstrip(';').rstrip()

    async def query_database(
        self,
        sql: str,
        max_rows: int = 1000,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Execute a SQL query against the genetics BigQuery database."""
        if not self.bigquery_url:
            return {
                "success": False,
                "error": "BigQuery API URL not configured. Set BIGQUERY_API_URL environment variable.",
            }

        try:
            # strip SQL LIMIT so the download gets the full result set;
            # always fetch up to 100k rows for the download, the LLM
            # result is truncated to max_rows and further by mcp_max_result_size
            download_sql, _ = self._strip_trailing_limit(sql)
            fetch_max = max(max_rows, 100_000)

            resp = await self.client.post(
                f"{self.bigquery_url}/query",
                json={"sql": download_sql, "max_rows": fetch_max, "dry_run": dry_run},
                timeout=300.0,
            )
            if resp.status_code != 200:
                return {
                    "success": False,
                    "error": f"HTTP {resp.status_code}: {resp.text}",
                }

            data = resp.json()
            columns = data.get("columns", [])
            all_rows = data.get("rows", [])

            # download gets all fetched rows
            download_data = (
                {"columns": columns, "rows": all_rows, "filename": "finngenie_results.tsv"}
                if all_rows else None
            )

            # LLM gets at most max_rows
            llm_rows = all_rows[:max_rows] if len(all_rows) > max_rows else all_rows

            download_capped = data.get("truncated", False)
            result: dict[str, Any] = {
                "success": True,
                "sql": sql,
                "columns": columns,
                "rows": llm_rows,
                "total_rows": data.get("total_rows", 0),
                "rows_in_download": len(all_rows),
                "download_capped_at_100k": download_capped,
                "bytes_processed": data.get("bytes_processed", 0),
                "truncated": len(all_rows) > max_rows or download_capped,
            }
            if download_data:
                result["_download_data"] = download_data

            return result
        except Exception as e:
            logger.error(f"Error in query_database: {e}\n{traceback.format_exc()}")
            return {"success": False, "error": INTERNAL_ERROR_MSG}

    async def get_database_schema(self, table: str | None = None) -> dict[str, Any]:
        """Get schema information for the BigQuery tables."""
        if not self.bigquery_url:
            return {
                "success": False,
                "error": "BigQuery API URL not configured. Set BIGQUERY_API_URL environment variable.",
            }

        try:
            params = {}
            if table:
                params["table"] = table
            resp = await self.client.get(f"{self.bigquery_url}/schema", params=params)
            if resp.status_code == 200:
                return {"success": True, "schema": resp.json()}
            if resp.headers.get(_UNREACHABLE_HEADER):
                return {"success": False, "error": UPSTREAM_UNREACHABLE_MSG, "unreachable": True}
            return {
                "success": False,
                "error": f"HTTP {resp.status_code}: {resp.text}",
            }
        except Exception as e:
            logger.error(f"Error in get_database_schema: {e}\n{traceback.format_exc()}")
            return {"success": False, "error": INTERNAL_ERROR_MSG}

    # -------------------------------------------------------------------------
    # Search Tools
    # -------------------------------------------------------------------------

    async def search_phenotypes(self, query: str, limit: int = 100) -> dict[str, Any]:
        """Search phenotypes via autocomplete endpoint. Supports comma-separated trait names."""
        normalized_query = ",".join(term.strip() for term in query.split(","))
        resp = await self.client.get(
            f"{self.base_url}/v1/search",
            params={
                "q": normalized_query,
                "types": "phenotypes",
                "limit": limit,
                "format": "json",
            },
        )
        if resp.status_code == 200:
            return {"success": True, "results": resp.json()}
        return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}

    async def search_genes(self, query: str, limit: int = 10) -> dict[str, Any]:
        """Search genes via autocomplete endpoint. Supports comma-separated gene names."""
        normalized_query = ",".join(term.strip() for term in query.split(","))
        resp = await self.client.get(
            f"{self.base_url}/v1/search",
            params={"q": normalized_query, "types": "genes", "limit": limit, "format": "json"},
        )
        if resp.status_code == 200:
            return {"success": True, "results": resp.json()}
        return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}

    async def lookup_variants_by_rsid(self, rsids: str) -> dict[str, Any]:
        """Convert rsIDs to variant IDs (chr:pos:ref:alt format)."""
        if not rsids or not rsids.strip():
            return {"success": False, "error": "No rsIDs provided"}

        normalized_rsids = ",".join(term.strip() for term in rsids.split(","))
        resp = await self.client.get(
            f"{self.base_url}/v1/rsid/variants",
            params={"rsids": normalized_rsids},
        )
        if resp.status_code == 200:
            return {"success": True, "variants": resp.json()}
        return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}

    async def lookup_phenotype_names(self, codes: list[str]) -> dict[str, Any]:
        """Batch lookup phenotype codes to names."""
        if not codes:
            return {"success": False, "error": "No phenotype codes provided"}

        resp = await self.client.get(f"{self.base_url}/v1/trait_name_mapping")
        if resp.status_code == 200:
            full_mapping = resp.json()
            result = {
                code: full_mapping.get(code, f"Unknown: {code}") for code in codes
            }
            return {"success": True, "names": result}
        return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}

    # -------------------------------------------------------------------------
    # Credible Sets Tools
    # -------------------------------------------------------------------------

    async def get_credible_sets_by_gene(
        self,
        gene: str,
        window: int = 500000,
        resource: str | None = None,
        data_types: str | None = None,
        summarize: bool = True,
    ) -> dict[str, Any]:
        """Get credible sets for a gene region."""
        try:
            params: dict[str, Any] = {"window": window}
            if resource:
                params["resources"] = resource
            if data_types:
                params["data_types"] = data_types

            dl_params: dict[str, Any] = {"window": window}
            if resource:
                dl_params["resources"] = resource
            if data_types:
                dl_params["data_types"] = data_types
            download_url = self._build_download_url(f"/v1/credible_sets_by_gene/{_seg(gene)}", dl_params)

            if summarize:
                params["format"] = "tsv"
                resp = await self.client.get(
                    f"{self.base_url}/v1/credible_sets_by_gene/{_seg(gene)}", params=params
                )
                if resp.status_code == 200:
                    summary = self._summarize_credible_sets_simple(resp.text)
                    return {"success": True, **self._columns_meta(resp), "gene": gene, "_download_url": download_url, **summary}
                return {
                    "success": False,
                    "error": f"HTTP {resp.status_code}: {resp.text}",
                }
            else:
                params["format"] = "json"
                resp = await self.client.get(
                    f"{self.base_url}/v1/credible_sets_by_gene/{_seg(gene)}", params=params
                )
                if resp.status_code == 200:
                    results = resp.json()
                    results = self._prioritize_variants(results)
                    return {
                        "success": True,
                        **self._columns_meta(resp),
                        "gene": gene,
                        "total_count": len(results),
                        "results": results,
                        "_download_url": download_url,
                    }
                return {
                    "success": False,
                    "error": f"HTTP {resp.status_code}: {resp.text}",
                }
        except Exception as e:
            logger.error(
                f"Error in get_credible_sets_by_gene({gene}): {e}\n{traceback.format_exc()}"
            )
            return {"success": False, "error": INTERNAL_ERROR_MSG}

    async def get_credible_sets_by_variant(
        self,
        variant: str,
        resource: str | None = None,
        data_types: str | None = None,
        summarize: bool = True,
    ) -> dict[str, Any]:
        """Get credible sets containing a specific variant."""
        try:
            params: dict[str, Any] = {}
            if resource:
                params["resources"] = resource
            if data_types:
                params["data_types"] = data_types

            dl_params = {k: v for k, v in params.items()}
            download_url = self._build_download_url(f"/v1/credible_sets_by_variant/{_seg(variant)}", dl_params)

            if summarize:
                params["format"] = "tsv"
                resp = await self.client.get(
                    f"{self.base_url}/v1/credible_sets_by_variant/{_seg(variant)}",
                    params=params,
                )
                if resp.status_code == 200:
                    summary = self._summarize_credible_sets_simple(resp.text)
                    return {"success": True, **self._columns_meta(resp), "variant": variant, "_download_url": download_url, **summary}
                return {
                    "success": False,
                    "error": f"HTTP {resp.status_code}: {resp.text}",
                }
            else:
                params["format"] = "json"
                resp = await self.client.get(
                    f"{self.base_url}/v1/credible_sets_by_variant/{_seg(variant)}",
                    params=params,
                )
                if resp.status_code == 200:
                    results = resp.json()
                    results = self._prioritize_variants(results)
                    return {
                        "success": True,
                        **self._columns_meta(resp),
                        "variant": variant,
                        "total_count": len(results),
                        "results": results,
                        "_download_url": download_url,
                    }
                return {
                    "success": False,
                    "error": f"HTTP {resp.status_code}: {resp.text}",
                }
        except Exception as e:
            logger.error(
                f"Error in get_credible_sets_by_variant({variant}): {e}\n{traceback.format_exc()}"
            )
            return {"success": False, "error": INTERNAL_ERROR_MSG}

    async def get_credible_sets_by_region(
        self,
        region: str,
        resource: str | None = None,
        coding_only: bool = False,
        summarize: bool = True,
    ) -> dict[str, Any]:
        """Get credible sets overlapping a genomic region across resources."""
        try:
            params: dict[str, Any] = {}
            rlist = self._resources_param(resource)
            if rlist:
                params["resources"] = rlist
            if coding_only:
                params["coding_only"] = "true"

            download_url = self._build_download_url(
                f"/v1/credible_sets_by_region/{region}", dict(params)
            )
            url = f"{self.base_url}/v1/credible_sets_by_region/{_seg(region)}"

            if summarize:
                params["format"] = "tsv"
                resp = await self.client.get(url, params=params, timeout=300.0)
                if resp.status_code == 200:
                    summary = self._summarize_credible_sets_simple(resp.text)
                    return {
                        "success": True,
                        **self._columns_meta(resp),
                        "region": region,
                        "_download_url": download_url,
                        **summary,
                    }
                return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}

            params["format"] = "json"
            resp = await self.client.get(url, params=params, timeout=300.0)
            if resp.status_code == 200:
                results = self._prioritize_variants(resp.json())
                rows, truncated = self._cap_rows(results)
                return {
                    "success": True,
                    **self._columns_meta(resp),
                    "region": region,
                    "total_count": len(results),
                    "truncated": truncated,
                    "results": rows,
                    "_download_url": download_url,
                }
            return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}
        except Exception as e:
            logger.error(
                f"Error in get_credible_sets_by_region({region}): {e}\n{traceback.format_exc()}"
            )
            return {"success": False, "error": INTERNAL_ERROR_MSG}

    async def get_credible_sets_by_phenotype(
        self,
        phenotype: str,
        resource: str = "finngen",
        summarize: bool = True,
    ) -> dict[str, Any]:
        """Get credible sets for a phenotype."""
        try:
            download_url = self._build_download_url(
                f"/v1/credible_sets_by_phenotype/{resource}/{phenotype}"
            )

            if summarize:
                resp = await self.client.get(
                    f"{self.base_url}/v1/credible_sets_by_phenotype/{_seg(resource)}/{_seg(phenotype)}",
                    params={"format": "tsv"},
                )
                if resp.status_code == 200:
                    summary = self._summarize_credible_sets_trait(resp.text)
                    return {"success": True, **self._columns_meta(resp), "phenotype": phenotype, "_download_url": download_url, **summary}
                return {
                    "success": False,
                    "error": f"HTTP {resp.status_code}: {resp.text}",
                }
            else:
                resp = await self.client.get(
                    f"{self.base_url}/v1/credible_sets_by_phenotype/{_seg(resource)}/{_seg(phenotype)}",
                    params={"format": "json"},
                )
                if resp.status_code == 200:
                    return {
                        "success": True,
                        **self._columns_meta(resp),
                        "phenotype": phenotype,
                        "results": resp.json(),
                        "_download_url": download_url,
                    }
                return {
                    "success": False,
                    "error": f"HTTP {resp.status_code}: {resp.text}",
                }
        except Exception as e:
            logger.error(
                f"Error in get_credible_sets_by_phenotype({phenotype}): {e}\n{traceback.format_exc()}"
            )
            return {"success": False, "error": INTERNAL_ERROR_MSG}

    async def get_credible_set_leads_by_phenotype(
        self, phenotype: str, resource: str = "finngen"
    ) -> dict[str, Any]:
        """Get one lead variant per credible set for a phenotype."""
        try:
            download_url = self._build_download_url(
                f"/v1/credible_sets_by_phenotype_leads/{resource}/{phenotype}"
            )
            resp = await self.client.get(
                f"{self.base_url}/v1/credible_sets_by_phenotype_leads/{_seg(resource)}/{_seg(phenotype)}",
                params={"format": "json"},
                timeout=300.0,
            )
            if resp.status_code == 200:
                results = resp.json()
                return {
                    "success": True,
                    "phenotype": phenotype,
                    "resource": resource,
                    "count": len(results),
                    "results": results,
                    "_download_url": download_url,
                }
            if resp.status_code == 404:
                return {
                    "success": False,
                    "error": f"Not found: phenotype '{phenotype}' in resource '{resource}'",
                }
            return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}
        except Exception as e:
            logger.error(
                f"Error in get_credible_set_leads_by_phenotype({resource}, {phenotype}): "
                f"{e}\n{traceback.format_exc()}"
            )
            return {"success": False, "error": INTERNAL_ERROR_MSG}

    async def get_credible_set_by_id(
        self,
        resource: str,
        phenotype: str,
        credible_set_id: str,
    ) -> dict[str, Any]:
        """Get all variants in a specific credible set."""
        try:
            encoded_cs_id = quote(credible_set_id, safe="")
            download_url = self._build_download_url(
                f"/v1/credible_sets_by_id/{resource}/{phenotype}/{encoded_cs_id}"
            )
            resp = await self.client.get(
                f"{self.base_url}/v1/credible_sets_by_id/{_seg(resource)}/{_seg(phenotype)}/{encoded_cs_id}",
                params={"format": "json"},
            )
            if resp.status_code == 200:
                variants = resp.json()
                return {
                    "success": True,
                    **self._columns_meta(resp),
                    "resource": resource,
                    "phenotype": phenotype,
                    "credible_set_id": credible_set_id,
                    "n_variants": len(variants),
                    "variants": variants,
                    "_download_url": download_url,
                }
            return {
                "success": False,
                "error": f"HTTP {resp.status_code}: {resp.text}",
            }
        except Exception as e:
            logger.error(
                f"Error in get_credible_set_by_id({credible_set_id}): {e}\n{traceback.format_exc()}"
            )
            return {"success": False, "error": INTERNAL_ERROR_MSG}

    async def get_credible_sets_by_qtl_gene(
        self,
        gene: str,
        data_types: str | None = None,
        resource: str | None = None,
        summarize: bool = True,
    ) -> dict[str, Any]:
        """Get QTL credible sets where gene is the molecular trait.

        Defaults to the credible set-level summary, matching the sibling credible-set tools:
        the variant-level result for a well-studied gene exceeds the tool-result size cap by
        more than an order of magnitude and gets truncated to a positional prefix.
        """
        try:
            params: dict[str, Any] = {}
            if data_types:
                params["data_types"] = data_types
            if resource:
                params["resources"] = resource

            dl_params = {k: v for k, v in params.items()}
            download_url = self._build_download_url(f"/v1/credible_sets_by_qtl_gene/{_seg(gene)}", dl_params)

            if summarize:
                params["format"] = "tsv"
                resp = await self.client.get(
                    f"{self.base_url}/v1/credible_sets_by_qtl_gene/{_seg(gene)}",
                    params=params,
                )
                if resp.status_code == 200:
                    summary = self._summarize_credible_sets_simple(resp.text)
                    return {"success": True, **self._columns_meta(resp), "gene": gene, "_download_url": download_url, **summary}
                return {
                    "success": False,
                    "error": f"HTTP {resp.status_code}: {resp.text}",
                }
            else:
                params["format"] = "json"
                resp = await self.client.get(
                    f"{self.base_url}/v1/credible_sets_by_qtl_gene/{_seg(gene)}",
                    params=params,
                )
                if resp.status_code == 200:
                    return {"success": True, **self._columns_meta(resp), "gene": gene, "results": resp.json(), "_download_url": download_url}
                return {
                    "success": False,
                    "error": f"HTTP {resp.status_code}: {resp.text}",
                }
        except Exception as e:
            logger.error(
                f"Error in get_credible_sets_by_qtl_gene({gene}): {e}\n{traceback.format_exc()}"
            )
            return {"success": False, "error": INTERNAL_ERROR_MSG}

    # -------------------------------------------------------------------------
    # Gene Data Tools
    # -------------------------------------------------------------------------

    async def _get_expression_resources(self) -> list[str]:
        """Fetch (and cache) the full set of expression resources the API serves.

        Used to distinguish "gene not covered by a resource" from "resource
        unavailable" in get_gene_expression. Returns [] on failure so the caller
        degrades gracefully (it just omits the coverage annotation)."""
        if self._expression_resources is None:
            try:
                resp = await self.client.get(f"{self.base_url}/v1/resources")
                resp.raise_for_status()
                self._expression_resources = [
                    e["resource"] for e in resp.json().get("expression", [])
                ]
            except Exception:
                self._expression_resources = []
        return self._expression_resources

    async def get_gene_expression(self, gene: str) -> dict[str, Any]:
        """Get tissue expression for a gene."""
        resp = await self.client.get(
            f"{self.base_url}/v1/expression_by_gene/{_seg(gene)}", params={"format": "json"}
        )
        if resp.status_code == 200:
            results = resp.json()
            out: dict[str, Any] = {
                "success": True, **self._columns_meta(resp), "gene": gene, "results": results,
                "_download_data": {"results": results, "filename": f"{gene}_expression.tsv"},
            }
            # Annotate which expression resources returned no rows for this gene, so a
            # zero-coverage resource (e.g. HPA lacks IHC data for many genes) is not
            # mistaken for the resource being unavailable. The endpoint merges all
            # resources and silently omits those with no matching rows.
            expected = await self._get_expression_resources()
            if expected:
                returned = {r.get("resource") for r in results}
                no_data = sorted(r for r in expected if r not in returned)
                if no_data:
                    out["resources_with_no_data"] = no_data
            return out
        return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}

    async def get_asm_qtl_by_variant(
        self, variant: str, resources: str | None = None
    ) -> dict[str, Any]:
        """Get ASM-QTL data for a variant via the sumstats endpoint."""
        phenotypes = "CpG,MDS"
        if resources:
            # map resource names to phenotype codes
            pheno_map = {"decode_cpg": "CpG", "decode_mds": "MDS"}
            phenotypes = ",".join(
                pheno_map.get(r.strip(), r.strip())
                for r in resources.split(",")
            )
        params: dict[str, Any] = {
            "variants": variant,
            "phenotypes": phenotypes,
            "format": "json",
        }
        resp = await self.client.get(
            f"{self.base_url}/v1/summary_stats/decode/asmqtl", params=params
        )
        if resp.status_code == 200:
            results = resp.json()
            return {
                "success": True, **self._columns_meta(resp), "variant": variant, "results": results,
                "_download_data": {"results": results, "filename": f"{variant}_asm_qtl.tsv"},
            }
        return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}

    async def get_asm_qtl_by_gene(
        self,
        gene: str,
        resources: str | None = None,
        window: int = 500000,
        limit: int = 500,
        with_metadata: bool = False,
    ) -> dict[str, Any]:
        """Get ASM-QTL data for variants near a gene via BigQuery.

        Selects variants by genomic coordinates (gene body ± window) rather than
        by `gene_most_severe`: most-severe-consequence attribution is unreliable
        for regulatory variants and misses signals that sit near — but not inside —
        the gene. Gene coordinates come from `gene_annotations_v`.
        """
        try:
            gene = normalize_literal(gene, name="gene")
            gene_lit = quote_literal(gene, name="gene")
            window_sql = sql_int(window, name="window", minimum=0, maximum=self._MAX_SQL_WINDOW)
            limit_sql = sql_int(limit, name="limit", minimum=1, maximum=self._MAX_SQL_LIMIT)
            limit = int(limit_sql)
            dataset_filter = ""
            if resources:
                ds_map = {"decode_cpg": "deCODE_asmQTL_CpG", "decode_mds": "deCODE_asmQTL_MDS"}
                datasets = [ds_map.get(r.strip(), r.strip()) for r in resources.split(",")]
                quoted = quote_literal_list(datasets, name="resources")
                dataset_filter = f" AND a.dataset IN ({quoted})"
        except SqlValueError as e:
            return {"success": False, "error": str(e)}

        sql = (
            f"{self._gene_window_cte(gene_lit)}"
            f"SELECT a.* FROM asm_qtl_v a "
            f"JOIN g ON CAST(a.chr AS STRING) = CAST(g.chr AS STRING) "
            f"AND a.pos BETWEEN g.gstart - {window_sql} AND g.gend + {window_sql} "
            f"WHERE TRUE{dataset_filter} "
            f"ORDER BY a.mlog10p DESC LIMIT {limit_sql}"
        )
        result = await self.query_database(sql, max_rows=limit)
        if result.get("success"):
            return self._bq_gene_payload(gene, result, f"{gene}_asm_qtl.tsv", with_metadata)
        return result

    async def get_open_chromatin_by_variant(
        self, variant: str, resources: str | None = None
    ) -> dict[str, Any]:
        """Get open-chromatin atlas peaks overlapping a variant position via the results-api."""
        params: dict[str, Any] = {"format": "json"}
        if resources:
            # list value -> repeated query params, which the API's list[str] Query expects
            params["resources"] = [r.strip() for r in resources.split(",")]
        resp = await self.client.get(
            f"{self.base_url}/v1/open_chromatin/variant/{_seg(variant)}", params=params
        )
        if resp.status_code == 200:
            results = resp.json()
            return {
                "success": True, **self._columns_meta(resp), "variant": variant, "results": results,
                "_download_data": {"results": results, "filename": f"{variant}_open_chromatin.tsv"},
            }
        return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}

    async def get_open_chromatin_by_region(
        self, chrom: str, start: int, end: int, resources: str | None = None
    ) -> dict[str, Any]:
        """Get open-chromatin atlas peaks overlapping a genomic region via the results-api."""
        params: dict[str, Any] = {"format": "json"}
        if resources:
            params["resources"] = [r.strip() for r in resources.split(",")]
        resp = await self.client.get(
            f"{self.base_url}/v1/open_chromatin/region/{_seg(chrom)}/{_seg(start)}/{_seg(end)}", params=params
        )
        if resp.status_code == 200:
            results = resp.json()
            return {
                "success": True, **self._columns_meta(resp), "region": f"{chrom}:{start}-{end}", "results": results,
                "_download_data": {"results": results, "filename": f"{chrom}_{start}_{end}_open_chromatin.tsv"},
            }
        return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}

    async def get_open_chromatin_by_peak(
        self, peak_id: str, resources: str | None = None
    ) -> dict[str, Any]:
        """Get an open-chromatin atlas peak by its peak id via the results-api."""
        params: dict[str, Any] = {"format": "json"}
        rlist = self._resources_param(resources)
        if rlist:
            params["resources"] = rlist
        resp = await self.client.get(
            f"{self.base_url}/v1/open_chromatin/peak/{_seg(peak_id)}", params=params
        )
        if resp.status_code == 200:
            results = resp.json()
            return {
                "success": True, **self._columns_meta(resp), "peak_id": peak_id, "count": len(results), "results": results,
                "_download_data": {"results": results, "filename": f"{peak_id}_open_chromatin.tsv"},
            }
        return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}

    async def get_peak_to_genes(
        self, peak_id: str, resources: str | None = None, gencode_version: str | None = None
    ) -> dict[str, Any]:
        """Get the genes an Open4Gene chromatin peak is linked to."""
        params: dict[str, Any] = {"format": "json"}
        rlist = self._resources_param(resources)
        if rlist:
            params["resources"] = rlist
        if gencode_version:
            params["gencode_version"] = gencode_version
        resp = await self.client.get(
            f"{self.base_url}/v1/peak_to_genes/{_seg(peak_id)}", params=params, timeout=300.0
        )
        if resp.status_code == 200:
            results = resp.json()
            return {
                "success": True, **self._columns_meta(resp), "peak_id": peak_id, "count": len(results), "results": results,
                "_download_data": {"results": results, "filename": f"{peak_id}_peak_to_genes.tsv"},
            }
        if resp.status_code == 404:
            return {"success": False, "error": f"Not found: peak '{peak_id}'"}
        return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}

    async def get_gene_to_peaks(
        self, gene: str, resources: str | None = None, gencode_version: str | None = None
    ) -> dict[str, Any]:
        """Get the Open4Gene chromatin peaks linked to a gene, per cell type."""
        params: dict[str, Any] = {"format": "json"}
        rlist = self._resources_param(resources)
        if rlist:
            params["resources"] = rlist
        if gencode_version:
            params["gencode_version"] = gencode_version
        resp = await self.client.get(
            f"{self.base_url}/v1/gene_to_peaks/{_seg(gene)}", params=params, timeout=300.0
        )
        if resp.status_code == 200:
            results = resp.json()
            rows, truncated = self._cap_rows(results)
            return {
                "success": True,
                **self._columns_meta(resp),
                "gene": gene,
                "total_count": len(results),
                "truncated": truncated,
                "results": rows,
                "_download_data": {"results": results, "filename": f"{gene}_gene_to_peaks.tsv"},
            }
        if resp.status_code == 404:
            return {"success": False, "error": f"Not found: gene '{gene}'"}
        return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}

    async def get_open_chromatin_by_gene(
        self,
        gene: str,
        resources: str | None = None,
        window: int = 500000,
        limit: int = 500,
        with_metadata: bool = False,
    ) -> dict[str, Any]:
        """Get open-chromatin atlas peaks near a gene via BigQuery.

        The results-api has no open_chromatin by-gene endpoint, so peaks are selected by
        genomic coordinates (gene body ± window) against `open_chromatin_v` — region overlap,
        not `gene_most_severe` attribution, so nearby regulatory/enhancer peaks are not missed.
        Gene coordinates come from `gene_annotations_v`; resources are filtered on the view's
        derived `resource` column.
        """
        try:
            gene = normalize_literal(gene, name="gene")
            gene_lit = quote_literal(gene, name="gene")
            window_sql = sql_int(window, name="window", minimum=0, maximum=self._MAX_SQL_WINDOW)
            limit_sql = sql_int(limit, name="limit", minimum=1, maximum=self._MAX_SQL_LIMIT)
            limit = int(limit_sql)
            resource_filter = ""
            if resources:
                rlist = [r.strip() for r in resources.split(",")]
                quoted = quote_literal_list(rlist, name="resources")
                resource_filter = f" AND a.resource IN ({quoted})"
        except SqlValueError as e:
            return {"success": False, "error": str(e)}

        sql = (
            f"{self._gene_window_cte(gene_lit)}"
            f"SELECT a.* FROM open_chromatin_v a "
            f"JOIN g ON CAST(a.chr AS STRING) = CAST(g.chr AS STRING) "
            f"AND a.peak_start <= g.gend + {window_sql} AND a.peak_end >= g.gstart - {window_sql} "
            f"WHERE TRUE{resource_filter} "
            f"ORDER BY a.tissue, a.cell_type, a.peak_start LIMIT {limit_sql}"
        )
        result = await self.query_database(sql, max_rows=limit)
        if result.get("success"):
            return self._bq_gene_payload(gene, result, f"{gene}_open_chromatin.tsv", with_metadata)
        return result

    async def get_variant_effect_by_variant(
        self, variant: str, resources: str | None = None
    ) -> dict[str, Any]:
        """Get in-silico predicted variant effect on chromatin accessibility via the results-api.

        Returns per-model (chrombpnet/flare) per-cell-type predicted scores for the variant.
        FLARE scores are pan-context (cell_type may be null); ChromBPNet scores are per context.
        """
        params: dict[str, Any] = {"format": "json"}
        if resources:
            # list value -> repeated query params, which the API's list[str] Query expects
            params["resources"] = [r.strip() for r in resources.split(",")]
        resp = await self.client.get(
            f"{self.base_url}/v1/variant_effect/variant/{_seg(variant)}", params=params
        )
        if resp.status_code == 200:
            results = resp.json()
            return {
                "success": True, **self._columns_meta(resp), "variant": variant, "results": results,
                "_download_data": {"results": results, "filename": f"{variant}_variant_effect.tsv"},
            }
        return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}

    async def get_variant_effect_by_gene(
        self,
        gene: str,
        resources: str | None = None,
        window: int = 500000,
        limit: int = 500,
        with_metadata: bool = False,
    ) -> dict[str, Any]:
        """Get in-silico predicted variant effects on chromatin accessibility near a gene via BigQuery.

        Variants are selected by genomic coordinates (gene body ± window) against
        `variant_effect_v` — the table is variant-indexed (single `pos`), so a point-position
        overlap is used, not `gene_most_severe` attribution, so nearby regulatory variants are
        not missed. Gene coordinates come from `gene_annotations_v`; resources are filtered on
        the view's derived `resource` column. Rows carry per-model (chrombpnet/flare) predicted
        scores so the agent can summarize how strongly and in which cell types accessibility is
        predicted to be affected (FLARE is pan-context, so cell_type may be null).
        """
        try:
            gene = normalize_literal(gene, name="gene")
            gene_lit = quote_literal(gene, name="gene")
            window_sql = sql_int(window, name="window", minimum=0, maximum=self._MAX_SQL_WINDOW)
            limit_sql = sql_int(limit, name="limit", minimum=1, maximum=self._MAX_SQL_LIMIT)
            limit = int(limit_sql)
            resource_filter = ""
            if resources:
                rlist = [r.strip() for r in resources.split(",")]
                quoted = quote_literal_list(rlist, name="resources")
                resource_filter = f" AND a.resource IN ({quoted})"
        except SqlValueError as e:
            return {"success": False, "error": str(e)}

        sql = (
            f"{self._gene_window_cte(gene_lit)}"
            f"SELECT a.* FROM variant_effect_v a "
            f"JOIN g ON CAST(a.chr AS STRING) = CAST(g.chr AS STRING) "
            f"AND a.pos BETWEEN g.gstart - {window_sql} AND g.gend + {window_sql} "
            f"WHERE TRUE{resource_filter} "
            f"ORDER BY a.mlog10p DESC LIMIT {limit_sql}"
        )
        result = await self.query_database(sql, max_rows=limit)
        if result.get("success"):
            return self._bq_gene_payload(gene, result, f"{gene}_variant_effect.tsv", with_metadata)
        return result

    async def get_mpra_by_variant(
        self, variant: str, resources: str | None = None
    ) -> dict[str, Any]:
        """Get measured MPRA cis-regulatory allelic activity for a variant via the results-api.

        Returns LONG rows (one per cell_line, incl. 'meta') with emVar/active/log2Skew/log2FC —
        measured intrinsic allelic activity, distinct from in-silico variant_effect predictions.
        """
        params: dict[str, Any] = {"format": "json"}
        if resources:
            # list value -> repeated query params, which the API's list[str] Query expects
            params["resources"] = [r.strip() for r in resources.split(",")]
        resp = await self.client.get(
            f"{self.base_url}/v1/mpra/variant/{_seg(variant)}", params=params
        )
        if resp.status_code == 200:
            results = resp.json()
            return {
                "success": True, **self._columns_meta(resp), "variant": variant, "results": results,
                "_download_data": {"results": results, "filename": f"{variant}_mpra.tsv"},
            }
        return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}

    async def get_mpra_by_region(
        self, chrom: str, start: int, end: int, resources: str | None = None
    ) -> dict[str, Any]:
        """Get measured MPRA cis-regulatory allelic activity for variants in a region via the results-api."""
        params: dict[str, Any] = {"format": "json"}
        if resources:
            params["resources"] = [r.strip() for r in resources.split(",")]
        resp = await self.client.get(
            f"{self.base_url}/v1/mpra/region/{_seg(chrom)}/{_seg(start)}/{_seg(end)}", params=params
        )
        if resp.status_code == 200:
            results = resp.json()
            return {
                "success": True, **self._columns_meta(resp), "region": f"{chrom}:{start}-{end}", "results": results,
                "_download_data": {"results": results, "filename": f"{chrom}_{start}_{end}_mpra.tsv"},
            }
        return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}

    async def get_mpra_by_gene(
        self,
        gene: str,
        resources: str | None = None,
        window: int = 500000,
        limit: int = 500,
        with_metadata: bool = False,
    ) -> dict[str, Any]:
        """Get measured MPRA cis-regulatory allelic activity near a gene via BigQuery.

        Mirrors get_variant_effect_by_gene: mpra is variant-indexed (single `pos`), so variants
        are selected by coordinate overlap (gene body ± window) against `mpra_v`, not
        `gene_most_severe` attribution, so nearby regulatory variants are not missed. Gene
        coordinates come from `gene_annotations_v`; resources are filtered on the view's derived
        `resource` column. Rows are LONG (one per variant per cell_line, incl. 'meta') carrying
        emVar/active/log2Skew/log2FC so the agent can summarize which variants are functionally
        active and how strongly; ordered by allelic-skew significance.
        """
        try:
            gene = normalize_literal(gene, name="gene")
            gene_lit = quote_literal(gene, name="gene")
            window_sql = sql_int(window, name="window", minimum=0, maximum=self._MAX_SQL_WINDOW)
            limit_sql = sql_int(limit, name="limit", minimum=1, maximum=self._MAX_SQL_LIMIT)
            limit = int(limit_sql)
            resource_filter = ""
            if resources:
                rlist = [r.strip() for r in resources.split(",")]
                quoted = quote_literal_list(rlist, name="resources")
                resource_filter = f" AND a.resource IN ({quoted})"
        except SqlValueError as e:
            return {"success": False, "error": str(e)}

        sql = (
            f"{self._gene_window_cte(gene_lit)}"
            f"SELECT a.* FROM mpra_v a "
            f"JOIN g ON CAST(a.chr AS STRING) = CAST(g.chr AS STRING) "
            f"AND a.pos BETWEEN g.gstart - {window_sql} AND g.gend + {window_sql} "
            f"WHERE TRUE{resource_filter} "
            f"ORDER BY a.log2Skew_mlog10p DESC LIMIT {limit_sql}"
        )
        result = await self.query_database(sql, max_rows=limit)
        if result.get("success"):
            return self._bq_gene_payload(gene, result, f"{gene}_mpra.tsv", with_metadata)
        return result

    # -------------------------------------------------------------------------
    # HLA Tools
    # -------------------------------------------------------------------------

    # An HLA allele name is gene-stripped and two-field: letters/digits for the locus,
    # '*', then colon-separated numeric fields, optionally with a WHO expression suffix
    # letter (N/L/S/C/A/Q). Anything else is rejected rather than interpolated into SQL.
    _HLA_ALLELE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*\*\d{2,3}(:\d{2,3})+[NLSCAQ]?$")

    async def get_hla_by_phenotype(
        self,
        phenotypes: list[str],
        genes: str | None = None,
        resource: str = "finngen",
    ) -> dict[str, Any]:
        """Get every classical HLA allele association for the given phenotype(s).

        One read returns a trait's whole HLA profile (187 alleles across 10 genes), so
        this is the tool for "what drives the MHC signal for X?". There is no by-variant
        form: an HLA allele has no chr:pos:ref:alt identity. For the reverse question —
        which traits an allele is associated with — use get_hla_by_allele.
        """
        if not phenotypes:
            return {"success": False, "error": "No phenotypes provided"}
        pheno_param = ",".join(p.strip() for p in phenotypes if p.strip())
        if not pheno_param:
            return {"success": False, "error": "No phenotypes provided"}

        params: dict[str, Any] = {"phenotypes": pheno_param, "format": "json"}
        if genes:
            params["genes"] = ",".join(g.strip() for g in genes.split(",") if g.strip())

        try:
            path = f"/v1/hla/{_seg(resource)}"
            resp = await self.client.get(
                f"{self.base_url}{path}", params=params, timeout=300.0
            )
            if resp.status_code == 200:
                results = resp.json()
                rows, truncated = self._cap_rows(results)
                download_params = {k: v for k, v in params.items() if k != "format"}
                return {
                    "success": True,
                    **self._columns_meta(resp),
                    "resource": resource,
                    "phenotypes": phenotypes,
                    "total_count": len(results),
                    "truncated": truncated,
                    "results": rows,
                    "_download_url": self._build_download_url(path, download_params),
                }
            if resp.status_code in (404, 422):
                return {"success": False, "error": f"{resp.status_code}: {resp.text}"}
            return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}
        except Exception as e:
            logger.error(
                f"Error in get_hla_by_phenotype({phenotypes}): {e}\n{traceback.format_exc()}"
            )
            return {"success": False, "error": INTERNAL_ERROR_MSG}

    async def get_hla_by_allele(
        self,
        allele: str,
        min_mlogp: float = 7.3,
        min_info: float = 0.5,
        resource: str = "finngen",
        max_rows: int = 200,
        with_metadata: bool = False,
    ) -> dict[str, Any]:
        """Get every phenotype an HLA allele is associated with, via BigQuery.

        The per-phenotype files results-api serves cannot answer this — it spans all
        2,712 of them — so it goes through `hla_associations_v`. `min_info` defaults to
        0.5 because rare, badly-imputed alleles produce huge unstable betas that read as
        spectacular associations; pass 0 to see them anyway.
        """
        # the data stores alleles gene-stripped ('B*27:05'), but the conventional written
        # form carries the gene ('HLA-B*27:05') and that is what a user will type, so it
        # is normalized here rather than costing a turn on a rejection
        name = re.sub(r"^HLA-", "", allele.strip(), flags=re.IGNORECASE)
        if not self._HLA_ALLELE_RE.match(name):
            return {
                "success": False,
                "error": f"'{allele}' is not an HLA allele name. Expected a gene-stripped "
                f"two-field name such as 'B*27:05' or 'DQB1*02:01'.",
            }

        try:
            # normalized before quoting so the value echoed back in the payload is the one
            # the allow-list saw — uniformity with the `gene` sites, not a security fix:
            # this resource only lands in a JSON field, never a filename or a header
            resource = normalize_literal(resource, name="resource")
            resource_lit = quote_literal(resource, name="resource")
            limit_sql = sql_int(max_rows, name="max_rows", minimum=1, maximum=self._MAX_SQL_LIMIT)
        except SqlValueError as e:
            return {"success": False, "error": str(e)}

        info_filter = f" AND info >= {float(min_info)}" if min_info else ""
        sql = (
            f"SELECT phenotype, gene, allele, mlog10p, pval, beta, se, "
            f"af, af_cases, af_controls, info "
            f"FROM hla_associations_v "
            f"WHERE allele = '{name}' AND resource = {resource_lit} "
            f"AND mlog10p >= {float(min_mlogp)}{info_filter} "
            f"ORDER BY mlog10p DESC LIMIT {limit_sql}"
        )
        result = await self.query_database(sql, max_rows=max_rows)
        if result.get("success"):
            columns, rows, error = self._positional_rows(result)
            if error:
                return error
            payload = {
                "success": True,
                "allele": name,
                "resource": resource,
                "min_mlogp": min_mlogp,
                "min_info": min_info,
                "count": len(rows),
                # the model cannot tell mlog10p from beta from af in a bare positional
                # list, so it gets named rows; the download keeps the positional form
                # `_convert_to_tsv` handles, whose `results` branch needs dicts
                "results": [dict(zip(columns, row)) for row in rows],
                "_download_data": {
                    "columns": columns,
                    "rows": rows,
                    "filename": f"{name.replace('*', '_').replace(':', '_')}_hla.tsv",
                },
            }
            return self._query_metadata(payload, result, with_metadata)
        return result

    async def get_mpra_pip_concordance_by_gene(
        self,
        gene: str,
        window: int = 500000,
        resource: str = "finngen",
        min_pip: float = 0.1,
        limit: int = 500,
        with_metadata: bool = False,
    ) -> dict[str, Any]:
        """Cross-reference FinnGen fine-mapped credible-set PIP against MEASURED MPRA emVar calls.

        Serves the regulatory-buffering signal (Kanai et al.): whether high-PIP fine-mapped
        variants near a gene actually show measured cis-regulatory allelic activity in MPRA.
        Joins credible_sets_v (fine-mapped, filtered to `resource`, pip >= min_pip) to the MPRA
        meta row (mpra_v.cell_line='meta') on the shared chr:pos:ref:alt `variant` key — both
        views use the same variant convention (X=23), so the string join is exact. The meta row
        is used for the core call because it carries the cross-cell-line emVar/log2Skew summary.
        Distinct from get_mpra_by_gene, which returns MPRA rows without the PIP cross-reference.

        Equivalent raw BigQuery join a user could run via query_database:

            WITH g AS (
              SELECT chr, MIN(gene_start) AS gstart, MAX(gene_end) AS gend
              FROM gene_annotations_v
              WHERE symbol = 'PCSK9' GROUP BY chr
            )
            SELECT c.variant, c.pip, c.cs_id, c.trait, c.data_type,
                   m.emVar, m.active, m.log2Skew, m.log2Skew_mlog10p, m.log2FC, m.cohort
            FROM credible_sets_v c
            JOIN g ON CAST(c.chr AS STRING) = CAST(g.chr AS STRING)
              AND c.pos BETWEEN g.gstart - 500000 AND g.gend + 500000
            JOIN mpra_v m
              ON m.variant = c.variant AND m.cell_line = 'meta'
            WHERE c.resource = 'finngen' AND c.pip >= 0.1
            ORDER BY m.emVar DESC, c.pip DESC
        """
        try:
            gene = normalize_literal(gene, name="gene")
            gene_lit = quote_literal(gene, name="gene")
            resource = normalize_literal(resource, name="resource")
            resource_lit = quote_literal(resource, name="resource")
            window_sql = sql_int(window, name="window", minimum=0, maximum=self._MAX_SQL_WINDOW)
            min_pip_sql = sql_float(min_pip, name="min_pip", minimum=0.0, maximum=1.0)
            limit_sql = sql_int(limit, name="limit", minimum=1, maximum=self._MAX_SQL_LIMIT)
            limit = int(limit_sql)
        except SqlValueError as e:
            return {"success": False, "error": str(e)}

        sql = (
            f"{self._gene_window_cte(gene_lit)}"
            f"SELECT c.variant, c.pip, c.cs_id, c.trait, c.data_type, "
            f"c.mlog10p AS gwas_mlog10p, c.beta, "
            f"m.emVar, m.active, m.log2Skew, m.log2Skew_mlog10p, m.log2FC, m.cohort "
            f"FROM credible_sets_v c "
            f"JOIN g ON CAST(c.chr AS STRING) = CAST(g.chr AS STRING) "
            f"AND c.pos BETWEEN g.gstart - {window_sql} AND g.gend + {window_sql} "
            f"JOIN mpra_v m "
            f"ON m.variant = c.variant AND m.cell_line = 'meta' "
            f"WHERE c.resource = {resource_lit} AND c.pip >= {min_pip_sql} "
            f"ORDER BY m.emVar DESC, c.pip DESC LIMIT {limit_sql}"
        )
        result = await self.query_database(sql, max_rows=limit)
        if result.get("success"):
            return self._bq_gene_payload(
                gene, result, f"{gene}_mpra_pip_concordance.tsv", with_metadata
            )
        return result

    async def get_gene_disease_associations(self, gene: str) -> dict[str, Any]:
        """Get gene-disease associations."""
        resp = await self.client.get(
            f"{self.base_url}/v1/gene_disease/{_seg(gene)}", params={"format": "json"}
        )
        if resp.status_code == 200:
            results = resp.json()
            result: dict[str, Any] = {"success": True, "gene": gene, "results": results}
            if results:
                result["_download_data"] = {"results": results, "filename": f"{gene}_disease_associations.tsv"}
            return result
        elif resp.status_code == 404:
            return {
                "success": True,
                "gene": gene,
                "results": [],
                "message": "No Mendelian disease associations found",
            }
        return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}

    async def get_exome_results_by_gene(self, gene: str) -> dict[str, Any]:
        """Get exome sequencing results for a gene."""
        resp = await self.client.get(
            f"{self.base_url}/v1/exome_results_by_gene/{_seg(gene)}",
            params={"format": "json"},
        )
        if resp.status_code == 200:
            results = resp.json()
            return {
                "success": True, **self._columns_meta(resp), "gene": gene, "results": results,
                "_download_data": {"results": results, "filename": f"{gene}_exome_results.tsv"},
            }
        return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}

    async def get_exome_results_by_variant(
        self, variant: str, resources: str | None = None
    ) -> dict[str, Any]:
        """Get exome sequencing results for a single variant across resources."""
        params: dict[str, Any] = {"format": "json"}
        rlist = self._resources_param(resources)
        if rlist:
            params["resources"] = rlist
        resp = await self.client.get(
            f"{self.base_url}/v1/exome_results_by_variant/{_seg(variant)}", params=params
        )
        if resp.status_code == 200:
            results = resp.json()
            return {
                "success": True, **self._columns_meta(resp), "variant": variant, "count": len(results), "results": results,
                "_download_data": {"results": results, "filename": f"{variant}_exome_results.tsv"},
            }
        return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}

    async def get_exome_results_by_region(
        self, region: str, resources: str | None = None
    ) -> dict[str, Any]:
        """Get exome sequencing results overlapping a genomic region across resources."""
        params: dict[str, Any] = {"format": "json"}
        rlist = self._resources_param(resources)
        if rlist:
            params["resources"] = rlist
        download_url = self._build_download_url(
            f"/v1/exome_results_by_region/{region}",
            {"resources": rlist} if rlist else None,
        )
        resp = await self.client.get(
            f"{self.base_url}/v1/exome_results_by_region/{_seg(region)}", params=params, timeout=300.0
        )
        if resp.status_code == 200:
            results = resp.json()
            rows, truncated = self._cap_rows(results)
            return {
                "success": True,
                **self._columns_meta(resp),
                "region": region,
                "total_count": len(results),
                "truncated": truncated,
                "results": rows,
                "_download_url": download_url,
            }
        return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}

    async def get_exome_results_by_phenotype(
        self, resource: str, phenotype: str
    ) -> dict[str, Any]:
        """Get individual variant exome results for a specific phenotype."""
        try:
            resp = await self.client.get(
                f"{self.base_url}/v1/exome_results_by_phenotype/{_seg(resource)}/{_seg(phenotype)}",
                params={"format": "json"},
                timeout=300.0,
            )
            if resp.status_code == 200:
                results = resp.json()
                return {
                    "success": True,
                    **self._columns_meta(resp),
                    "resource": resource,
                    "phenotype": phenotype,
                    "count": len(results),
                    "results": results,
                    "_download_url": self._build_download_url(
                        f"/v1/exome_results_by_phenotype/{resource}/{phenotype}"
                    ),
                }
            if resp.status_code == 404:
                return {
                    "success": False,
                    "error": f"Not found: phenotype '{phenotype}' in resource '{resource}'",
                }
            return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}
        except Exception as e:
            logger.error(
                f"Error in get_exome_results_by_phenotype({resource}, {phenotype}): {e}\n{traceback.format_exc()}"
            )
            return {"success": False, "error": INTERNAL_ERROR_MSG}

    async def get_gene_based_results_by_phenotype(
        self, resource: str, phenotype: str
    ) -> dict[str, Any]:
        """Get unfiltered gene burden results for one phenotype."""
        try:
            resp = await self.client.get(
                f"{self.base_url}/v1/gene_based_results_by_phenotype/{_seg(resource)}/{_seg(phenotype)}",
                params={"format": "json"},
                timeout=300.0,
            )
            if resp.status_code == 200:
                results = resp.json()
                return {
                    "success": True,
                    "resource": resource,
                    "phenotype": phenotype,
                    "count": len(results),
                    "results": results,
                    "_download_url": self._build_download_url(
                        f"/v1/gene_based_results_by_phenotype/{resource}/{phenotype}"
                    ),
                }
            if resp.status_code == 404:
                return {
                    "success": False,
                    "error": f"Not found: phenotype '{phenotype}' in resource '{resource}'",
                }
            return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}
        except Exception as e:
            logger.error(
                f"Error in get_gene_based_results_by_phenotype({resource}, {phenotype}): {e}\n{traceback.format_exc()}"
            )
            return {"success": False, "error": INTERNAL_ERROR_MSG}

    async def get_gene_based_results(self, gene: str) -> dict[str, Any]:
        """Get gene-level burden test results (genebass, IBD, BipEx2, SCHEMA)."""
        import csv
        import io

        resp = await self.client.get(f"{self.base_url}/v1/gene_based/{_seg(gene)}")
        if resp.status_code == 200:
            reader = csv.DictReader(io.StringIO(resp.text), delimiter="\t")
            results = list(reader)
            result: dict[str, Any] = {
                "success": True,
                "gene": gene,
                "count": len(results),
                "results": results,
            }
            if results:
                result["_download_data"] = {
                    "results": results,
                    "filename": f"{gene}_gene_based_results.tsv",
                }
            return result
        return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}

    # -------------------------------------------------------------------------
    # LD Tools (FinnGen LD Server)
    # -------------------------------------------------------------------------

    def _parse_variant(self, variant: str) -> tuple[str, int, str, str]:
        """Parse variant ID into components.

        Args:
            variant: Variant ID in chr:pos:ref:alt format

        Returns:
            Tuple of (chromosome, position, ref, alt)

        Raises:
            ValueError: If variant format is invalid
        """
        parts = variant.split(":")
        if len(parts) != 4:
            raise ValueError(
                f"Invalid variant format: {variant}. Expected chr:pos:ref:alt"
            )
        chr_str, pos_str, ref, alt = parts
        try:
            pos = int(pos_str)
        except ValueError:
            raise ValueError(f"Invalid position in variant: {variant}")
        return chr_str, pos, ref, alt

    async def get_ld_between_variants(
        self,
        variant1: str,
        variant2: str,
        r2_threshold: float = 0.1,
        panel: str = "sisu42",
    ) -> dict[str, Any]:
        """Get LD statistics between two specific variants."""
        try:
            try:
                chr1, pos1, _, _ = self._parse_variant(variant1)
                chr2, pos2, _, _ = self._parse_variant(variant2)
            except ValueError as e:
                return {"success": False, "error": str(e)}

            # normalize chromosome format for comparison
            chr1_norm = chr1.replace("chr", "")
            chr2_norm = chr2.replace("chr", "")

            if chr1_norm != chr2_norm:
                return {
                    "success": False,
                    "error": f"Variants must be on same chromosome. Got chr{chr1_norm} and chr{chr2_norm}",
                }

            distance = abs(pos2 - pos1)

            # max 5 Mb distance
            if distance > 5_000_000:
                return {
                    "success": False,
                    "error": f"Variants are too far apart ({distance:,} bp). Maximum allowed distance is 5 Mb.",
                }

            # window = 2 * distance + 1000000 (API bug workaround)
            window = 2 * distance + 1_000_000

            resp = await self.external_client.get(
                "https://api.finngen.fi/api/ld",
                params={
                    "variant": variant1,
                    "window": window,
                    "panel": panel,
                    "r2_thresh": r2_threshold,
                },
                timeout=30.0,
            )

            if resp.status_code != 200:
                return {
                    "success": False,
                    "error": f"FinnGen LD API error: HTTP {resp.status_code}",
                }

            data = resp.json()
            ld_results = data.get("ld", [])

            # find variant2 in results (could be in variation1 or variation2 field)
            match = None
            for entry in ld_results:
                v1 = entry.get("variation1", "")
                v2 = entry.get("variation2", "")
                if v2 == variant2 or v1 == variant2:
                    match = entry
                    break

            if not match:
                return {
                    "success": True,
                    "variant1": variant1,
                    "variant2": variant2,
                    "in_ld": False,
                    "message": f"No LD found between variants (r2 < {r2_threshold} or variant not in reference panel)",
                }

            return {
                "success": True,
                "variant1": variant1,
                "variant2": variant2,
                "in_ld": True,
                "r2": match.get("r2"),
                "d_prime": match.get("d_prime"),
                "panel": panel,
            }

        except Exception as e:
            logger.error(
                f"Error in get_ld_between_variants({variant1}, {variant2}): {e}\n{traceback.format_exc()}"
            )
            return {"success": False, "error": INTERNAL_ERROR_MSG}

    async def get_variants_in_ld(
        self,
        variant: str,
        window: int = 1_500_000,
        r2_threshold: float = 0.6,
        panel: str = "sisu42",
    ) -> dict[str, Any]:
        """Get all variants in LD with a given variant."""
        try:
            try:
                self._parse_variant(variant)
            except ValueError as e:
                return {"success": False, "error": str(e)}

            resp = await self.external_client.get(
                "https://api.finngen.fi/api/ld",
                params={
                    "variant": variant,
                    "window": window,
                    "panel": panel,
                    "r2_thresh": r2_threshold,
                },
                timeout=30.0,
            )

            if resp.status_code != 200:
                return {
                    "success": False,
                    "error": f"FinnGen LD API error: HTTP {resp.status_code}",
                }

            data = resp.json()
            ld_results = data.get("ld", [])

            # extract variants in LD (the "other" variant from each pair)
            variants_in_ld = []
            for entry in ld_results:
                v1 = entry.get("variation1", "")
                v2 = entry.get("variation2", "")
                other_variant = v2 if v1 == variant else v1
                variants_in_ld.append({
                    "variant": other_variant,
                    "r2": entry.get("r2"),
                    "d_prime": entry.get("d_prime"),
                })

            # sort by r2 descending
            variants_in_ld.sort(key=lambda x: x.get("r2") or 0, reverse=True)

            result: dict[str, Any] = {
                "success": True,
                "query_variant": variant,
                "window": window,
                "r2_threshold": r2_threshold,
                "panel": panel,
                "n_variants": len(variants_in_ld),
                "variants": variants_in_ld,
            }
            if variants_in_ld:
                result["_download_data"] = {
                    "results": variants_in_ld,
                    "filename": f"{variant}_ld_variants.tsv",
                }
            return result

        except Exception as e:
            logger.error(
                f"Error in get_variants_in_ld({variant}): {e}\n{traceback.format_exc()}"
            )
            return {"success": False, "error": INTERNAL_ERROR_MSG}

    # -------------------------------------------------------------------------
    # Colocalization and Reports
    # -------------------------------------------------------------------------

    async def get_colocalization(self, variant: str) -> dict[str, Any]:
        """Get colocalization results for a variant."""
        resp = await self.client.get(
            f"{self.base_url}/v1/colocalization_by_variant/{_seg(variant)}",
            params={"format": "json"},
        )
        if resp.status_code == 200:
            results = resp.json()
            return {
                "success": True, **self._columns_meta(resp), "variant": variant, "results": results,
                "_download_data": {"results": results, "filename": f"{variant}_colocalization.tsv"},
            }
        return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}

    async def get_colocalization_by_credible_set(
        self,
        resource: str,
        phenotype: str,
        credible_set_id: str,
        dual_format: bool = False,
    ) -> dict[str, Any]:
        """Get credible sets that colocalize with one specific credible set."""
        encoded_cs_id = quote(credible_set_id, safe="")
        path = f"/v1/colocalization_by_credible_set_id/{_seg(resource)}/{_seg(phenotype)}/{encoded_cs_id}"
        params: dict[str, Any] = {"format": "json"}
        if dual_format:
            params["dual_format"] = "true"
        resp = await self.client.get(f"{self.base_url}{path}", params=params, timeout=300.0)
        if resp.status_code == 200:
            results = resp.json()
            return {
                "success": True,
                **self._columns_meta(resp),
                "resource": resource,
                "phenotype": phenotype,
                "credible_set_id": credible_set_id,
                "count": len(results),
                "results": results,
                "_download_url": self._build_download_url(
                    path, {"dual_format": "true"} if dual_format else None
                ),
            }
        if resp.status_code == 404:
            return {
                "success": False,
                "error": f"Not found: credible set '{credible_set_id}' "
                f"for phenotype '{phenotype}' in resource '{resource}'",
            }
        return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}

    async def get_phenotype_report(self, resource: str, phenotype_code: str) -> dict[str, Any]:
        """Get phenotype markdown report."""
        resp = await self.client.get(
            f"{self.base_url}/v1/phenotype/{_seg(resource)}/{_seg(phenotype_code)}/markdown",
        )
        if resp.status_code == 200:
            return {
                "success": True,
                "resource": resource,
                "phenotype_code": phenotype_code,
                "content": resp.text,
            }
        elif resp.status_code == 404:
            return {
                "success": False,
                "resource": resource,
                "phenotype_code": phenotype_code,
                "error": f"No report found for phenotype: {phenotype_code} in resource: {resource}",
            }
        return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}

    async def get_available_resources(self) -> dict[str, Any]:
        """Get catalog of available data resources."""
        resp = await self.client.get(f"{self.base_url}/v1/resources")
        if resp.status_code == 200:
            return {"success": True, "resources": resp.json()}
        return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}

    async def list_datasets(
        self, resource: str | None = None, include_stats: bool = True
    ) -> dict[str, Any]:
        """Get catalog of datasets with descriptions, products, and aggregate stats."""
        params: dict[str, Any] = {}
        if resource:
            params["resource"] = resource
        if not include_stats:
            params["include_stats"] = "false"
        resp = await self.client.get(f"{self.base_url}/v1/datasets", params=params)
        if resp.status_code == 200:
            return {"success": True, "datasets": resp.json()}
        return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}

    async def get_dataset_display_names(self) -> dict[str, Any]:
        """Get the display-name overrides keyed by the raw `dataset` column value."""
        resp = await self.client.get(f"{self.base_url}/v1/dataset_display_names")
        if resp.status_code == 200:
            return {"success": True, "display_names": resp.json()}
        return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}

    async def get_resource_metadata(self, resource: str) -> dict[str, Any]:
        """Get harmonized per-trait metadata (sample sizes, trait names) for a resource."""
        resp = await self.client.get(
            f"{self.base_url}/v1/resource_metadata/{_seg(resource)}",
            params={"format": "json"},
            timeout=300.0,
        )
        if resp.status_code == 200:
            results = resp.json()
            rows, truncated = self._cap_rows(results) if isinstance(results, list) else (results, False)
            return {
                "success": True,
                "resource": resource,
                "count": len(results) if isinstance(results, list) else None,
                "truncated": truncated,
                "metadata": rows,
                "_download_url": self._build_download_url(f"/v1/resource_metadata/{_seg(resource)}"),
            }
        if resp.status_code == 404:
            return {"success": False, "error": f"Not found: resource '{resource}'"}
        return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}

    # -------------------------------------------------------------------------
    # Variant Annotation Tools
    # -------------------------------------------------------------------------

    async def get_variant_annotations(
        self,
        variant: str | None = None,
        region: str | None = None,
        gene: str | None = None,
        variants: list[str] | None = None,
        source: str = "finngen",
    ) -> dict[str, Any]:
        """Get variant annotations by variant, region, gene, or batch variants."""
        # exactly one query type must be provided
        provided = sum(x is not None for x in [variant, region, gene, variants])
        if provided != 1:
            return {
                "success": False,
                "error": "Exactly one of 'variant', 'region', 'gene', or 'variants' must be provided",
            }
        if variants is not None and not variants:
            return {"success": False, "error": "No variants provided"}

        # determine query key/value for GET requests
        query_key: str | None = None
        query_value: str | None = None
        if variants is None:
            query_key = "variant" if variant is not None else "region" if region is not None else "gene"
            query_value = variant or region or gene

        try:
            if variants is not None:
                # POST endpoint for batch variant lookup
                resp = await self.client.post(
                    f"{self.base_url}/v1/variant_annotation/{_seg(source)}",
                    params={"format": "json"},
                    json={"variants": variants},
                    timeout=300.0,
                )
            else:
                # GET endpoint for single variant, region, or gene
                params: dict[str, Any] = {"format": "json", query_key: query_value}
                resp = await self.client.get(
                    f"{self.base_url}/v1/variant_annotation/{_seg(source)}",
                    params=params,
                    timeout=300.0,
                )

            if resp.status_code == 200:
                results = resp.json()
                query_desc = query_value if variants is None else f"{len(variants)} variants"
                result: dict[str, Any] = {
                    "success": True,
                    **self._columns_meta(resp),
                    "source": source,
                    "query": query_desc,
                    "count": len(results),
                    "results": results,
                }
                if results:
                    if variants is None:
                        result["_download_url"] = self._build_download_url(
                            f"/v1/variant_annotation/{source}", {query_key: query_value}
                        )
                    else:
                        result["_download_data"] = {
                            "results": results,
                            "filename": "variant_annotations.tsv",
                        }
                return result
            if resp.status_code == 404:
                return {"success": False, "error": f"Not found: {resp.text}"}
            return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}
        except Exception as e:
            logger.error(f"Error in get_variant_annotations: {e}\n{traceback.format_exc()}")
            return {"success": False, "error": INTERNAL_ERROR_MSG}

    # -------------------------------------------------------------------------
    # Summary Statistics Tools
    # -------------------------------------------------------------------------

    async def get_summary_stats(
        self,
        variants: list[str],
        phenotypes: list[str],
        resource: str = "finngen",
        data_type: str = "gwas",
    ) -> dict[str, Any]:
        """Get summary statistics for variant-phenotype pairs from a resource."""
        if not variants:
            return {"success": False, "error": "No variants provided"}
        if not phenotypes:
            return {"success": False, "error": "No phenotypes provided"}

        # normalize variant separators to dash (API expects CHR-POS-REF-ALT)
        normalized = []
        for v in variants:
            normalized.append(re.sub(r"[:_|]", "-", v.strip()))

        try:
            resp = await self.client.post(
                f"{self.base_url}/v1/summary_stats/{_seg(resource)}/{_seg(data_type)}",
                params={"format": "json"},
                json={"variants": normalized, "phenotypes": phenotypes},
                timeout=300.0,
            )
            if resp.status_code == 200:
                results = resp.json()
                result: dict[str, Any] = {
                    "success": True,
                    **self._columns_meta(resp),
                    "resource": resource,
                    "data_type": data_type,
                    "count": len(results),
                    "results": results,
                }
                if results:
                    result["_download_data"] = {"results": results, "filename": "summary_stats.tsv"}
                return result
            if resp.status_code == 404:
                return {
                    "success": False,
                    "error": f"Not found: {resp.text}",
                }
            return {
                "success": False,
                "error": f"HTTP {resp.status_code}: {resp.text}",
            }
        except Exception as e:
            logger.error(f"Error in get_summary_stats: {e}\n{traceback.format_exc()}")
            return {"success": False, "error": INTERNAL_ERROR_MSG}

    async def get_summary_stats_by_region(
        self,
        region: str,
        phenotypes: list[str],
        resource: str = "finngen",
        data_type: str = "gwas",
    ) -> dict[str, Any]:
        """Get every summary stat record in a genomic region for one or more phenotypes."""
        if not phenotypes:
            return {"success": False, "error": "No phenotypes provided"}
        try:
            path = f"/v1/summary_stats_by_region/{_seg(resource)}/{_seg(data_type)}/{_seg(region)}"
            pheno_param = ",".join(p.strip() for p in phenotypes if p.strip())
            resp = await self.client.get(
                f"{self.base_url}{path}",
                params={"phenotypes": pheno_param, "format": "json"},
                timeout=300.0,
            )
            if resp.status_code == 200:
                results = resp.json()
                rows, truncated = self._cap_rows(results)
                return {
                    "success": True,
                    **self._columns_meta(resp),
                    "region": region,
                    "resource": resource,
                    "data_type": data_type,
                    "phenotypes": phenotypes,
                    "total_count": len(results),
                    "truncated": truncated,
                    "results": rows,
                    "_download_url": self._build_download_url(
                        path, {"phenotypes": pheno_param}
                    ),
                }
            if resp.status_code in (404, 422):
                return {"success": False, "error": f"{resp.status_code}: {resp.text}"}
            return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}
        except Exception as e:
            logger.error(
                f"Error in get_summary_stats_by_region({region}): {e}\n{traceback.format_exc()}"
            )
            return {"success": False, "error": INTERNAL_ERROR_MSG}

    # -------------------------------------------------------------------------
    # Visualization Tools
    # -------------------------------------------------------------------------

    async def create_phewas_plot(
        self,
        variant: str,
        resource: str | None = None,
        significance_threshold: float = 7.3,
        min_mlog10p: float = 2.0,
    ) -> dict[str, Any]:
        """Create a PheWAS plot for a variant showing phenotype associations."""
        try:
            # fetch associations using existing method
            data = await self.get_credible_sets_by_variant(
                variant, resource=resource, summarize=False
            )
            if not data["success"]:
                return data

            # filter to GWAS associations above threshold
            results = [
                r for r in data["results"]
                if r.get("data_type") == "GWAS"
                and (r.get("mlog10p") or 0) >= min_mlog10p
            ]

            if not results:
                return {
                    "success": False,
                    "error": f"No GWAS associations found for {variant} with -log10(p) >= {min_mlog10p}",
                }

            # get phenotype names for categorization
            phenotype_codes = list(set(r.get("trait", "") for r in results if r.get("trait")))
            names_data = await self.lookup_phenotype_names(phenotype_codes)
            code_to_name = names_data.get("names", {}) if names_data.get("success") else {}

            # categorize each phenotype
            for r in results:
                code = r.get("trait", "")
                name = code_to_name.get(code, "")
                r["category"] = categorize_phenotype(code, name)
                r["phenotype_name"] = name if name and not name.startswith("Unknown:") else code

            # sort by category, then by mlog10p within category
            results.sort(key=lambda x: (x["category"], -(x.get("mlog10p") or 0)))

            # generate matplotlib figure
            fig = self._create_phewas_figure(results, variant, significance_threshold)

            # encode as base64 PNG
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", dpi=100, bbox_inches="tight", facecolor="white")
            buffer.seek(0)
            base64_png = base64.b64encode(buffer.read()).decode("utf-8")
            plt.close(fig)

            # compute summary stats
            n_significant = sum(1 for r in results if (r.get("mlog10p") or 0) >= significance_threshold)
            categories_found = sorted(set(r["category"] for r in results))

            return {
                "success": True,
                "variant": variant,
                "n_associations": len(results),
                "n_significant": n_significant,
                "categories": categories_found,
                "image_base64": base64_png,
                "image_format": "png",
            }

        except Exception as e:
            logger.error(f"Error in create_phewas_plot({variant}): {e}\n{traceback.format_exc()}")
            return {"success": False, "error": INTERNAL_ERROR_MSG}

    def _create_phewas_figure(
        self,
        results: list[dict],
        variant: str,
        significance_threshold: float,
    ) -> plt.Figure:
        """Create a matplotlib PheWAS figure."""
        fig, ax = plt.subplots(figsize=(14, 6))

        # assign x-positions - group by category
        categories = []
        x_positions = []
        y_values = []
        colors = []
        labels = []

        current_x = 0
        last_category = None
        category_ranges = {}  # track x-range for each category

        for r in results:
            category = r["category"]

            # add gap between categories
            if last_category is not None and category != last_category:
                current_x += 2

            if category not in category_ranges:
                category_ranges[category] = {"start": current_x, "end": current_x}

            x_positions.append(current_x)
            y_values.append(r.get("mlog10p") or 0)
            colors.append(get_category_color(category))
            labels.append(r["phenotype_name"])
            categories.append(category)

            category_ranges[category]["end"] = current_x
            last_category = category
            current_x += 1

        # scatter plot
        ax.scatter(x_positions, y_values, c=colors, s=50, alpha=0.7, edgecolors="none")

        # significance threshold line
        ax.axhline(
            y=significance_threshold,
            color="red",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            label="Genome-wide significance (p=5e-8)",
        )

        # category labels at bottom
        for category, ranges in category_ranges.items():
            mid_x = (ranges["start"] + ranges["end"]) / 2
            ax.text(
                mid_x,
                -0.5,
                category,
                ha="center",
                va="top",
                fontsize=8,
                rotation=45,
                color=get_category_color(category),
                fontweight="bold",
            )

        # annotate top significant hits
        significant_results = [
            (x, y, label) for x, y, label in zip(x_positions, y_values, labels)
            if y >= significance_threshold
        ]
        # sort by y descending and take top 10
        significant_results.sort(key=lambda t: -t[1])
        for x, y, label in significant_results[:10]:
            # truncate long labels
            short_label = label[:30] + "..." if len(label) > 30 else label
            ax.annotate(
                short_label,
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7,
                alpha=0.8,
            )

        # formatting
        ax.set_xlabel("Phenotype Category", fontsize=10)
        ax.set_ylabel("-log10(p-value)", fontsize=10)
        ax.set_title(f"PheWAS Plot for {variant}", fontsize=12, fontweight="bold")

        ax.set_xlim(-1, current_x)
        ax.set_ylim(bottom=0)
        ax.set_xticks([])  # hide x-axis ticks since we're using category labels

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        # legend for significance line
        ax.legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        return fig

    async def get_credible_sets_stats(
        self,
        resource_or_dataset: str,
        trait: str | None = None,
    ) -> dict[str, Any]:
        """Get credible sets statistics for a dataset."""
        try:
            resp = await self.client.get(
                f"{self.base_url}/v1/credible_sets/{_seg(resource_or_dataset)}/stats",
                params={"format": "json"},
            )
            if resp.status_code != 200:
                return {
                    "success": False,
                    "error": f"HTTP {resp.status_code}: {resp.text}",
                }

            data = resp.json()
            if not data:
                return {
                    "success": True,
                    "resource_or_dataset": resource_or_dataset,
                    "n_traits": 0,
                    "totals": {},
                    "traits": [],
                }

            # filter by trait if specified
            if trait:
                data = [row for row in data if row.get("trait") == trait]

            # compute aggregate totals
            stat_cols = [
                "n_risk_cs", "n_risk_cs_with_coding", "n_risk_cs_with_coding_pip_gt_0_05",
                "n_risk_cs_with_lof", "n_risk_cs_with_lof_pip_gt_0_05",
                "n_protective_cs", "n_protective_cs_with_coding", "n_protective_cs_with_coding_pip_gt_0_05",
                "n_protective_cs_with_lof", "n_protective_cs_with_lof_pip_gt_0_05",
            ]

            totals = {}
            for col in stat_cols:
                totals[col] = sum(row.get(col, 0) or 0 for row in data)

            download_url = self._build_download_url(
                f"/v1/credible_sets/{resource_or_dataset}/stats"
            )

            return {
                "_download_url": download_url,
                "success": True,
                "resource_or_dataset": resource_or_dataset,
                "n_traits": len(data),
                "totals": totals,
                "traits": data,
            }
        except Exception as e:
            logger.error(
                f"Error in get_credible_sets_stats({resource_or_dataset}): {e}\n{traceback.format_exc()}"
            )
            return {"success": False, "error": INTERNAL_ERROR_MSG}

    # -------------------------------------------------------------------------
    # Variant List Analysis
    # -------------------------------------------------------------------------

    async def analyze_variant_list(
        self,
        variants: str,
        resource: str | None = None,
    ) -> dict[str, Any]:
        """Analyze a list of variants for phenotype, QTL, and tissue patterns."""
        try:
            parsed = self._parse_variant_list(variants)
            if not parsed:
                return {"success": False, "error": "No valid variants found in input"}

            has_betas = any(v.get("beta") is not None for v in parsed)
            variant_ids = [v["variant"] for v in parsed]
            # build lookup for input betas
            input_betas: dict[str, float | None] = {
                v["variant"]: v.get("beta") for v in parsed
            }

            variants_payload = {"variants": "\n".join(variant_ids)}

            async def _fetch_cs_batch() -> list[dict[str, Any]]:
                params: dict[str, Any] = {"format": "json"}
                if resource:
                    params["resources"] = resource
                try:
                    resp = await self.client.post(
                        f"{self.base_url}/v1/credible_sets_by_variant",
                        params=params,
                        json=variants_payload,
                    )
                    if resp.status_code == 200:
                        return resp.json()
                    return []
                except Exception:
                    return []

            async def _fetch_nearest_batch() -> list[dict[str, Any]]:
                try:
                    resp = await self.client.post(
                        f"{self.base_url}/v1/nearest_genes",
                        params={"format": "json", "n": 1, "gene_type": "protein_coding"},
                        json=variants_payload,
                    )
                    if resp.status_code == 200:
                        return resp.json()
                    return []
                except Exception:
                    return []

            cs_raw, genes_raw = await asyncio.gather(
                _fetch_cs_batch(),
                _fetch_nearest_batch(),
            )

            def _normalize_vid(v: str) -> str:
                return v.replace("-", ":")

            # group credible set results by variant
            cs_by_variant: dict[str, list] = defaultdict(list)
            for r in cs_raw:
                if "variant" in r:
                    vid = _normalize_vid(r["variant"])
                elif "chr" in r and "pos" in r and "ref" in r and "alt" in r:
                    vid = f"{r['chr']}:{r['pos']}:{r['ref']}:{r['alt']}"
                else:
                    continue
                cs_by_variant[vid].append(r)
            cs_results = [{"variant": vid, "results": cs_by_variant.get(vid, [])} for vid in variant_ids]

            # build per-variant nearest gene results
            nearest_by_variant: dict[str, dict] = {}
            for g in genes_raw:
                vid = _normalize_vid(g.get("variant", ""))
                if vid and vid not in nearest_by_variant:
                    nearest_by_variant[vid] = {
                        "variant": vid,
                        "gene": g.get("name") or g.get("hgnc_symbol", ""),
                        "distance": g.get("distance", 0),
                    }
            gene_results = [
                nearest_by_variant.get(vid, {"variant": vid, "gene": "", "distance": -1})
                for vid in variant_ids
            ]

            # aggregate credible set data
            def _new_counts() -> dict:
                return {"variants": set(), "consistent": set(), "inconsistent": set(), "resources": set(), "datasets": set()}

            gwas_counts: dict[str, dict] = defaultdict(_new_counts)
            pqtl_counts: dict[str, dict] = defaultdict(_new_counts)
            eqtl_counts: dict[str, dict] = defaultdict(_new_counts)
            caqtl_counts: dict[str, dict] = defaultdict(lambda: {"variants": set(), "resources": set(), "datasets": set()})
            tissue_eqtl_variants: dict[str, set] = defaultdict(set)
            pqtl_variants: set[str] = set()
            variants_with_cs: set[str] = set()

            for cs_data in cs_results:
                vid = cs_data["variant"]
                for r in cs_data["results"]:
                    data_type = r.get("data_type")
                    if not data_type:
                        continue
                    variants_with_cs.add(vid)

                    cs_beta = r.get("beta")
                    in_beta = input_betas.get(vid)
                    cs_resource = r.get("resource", "")
                    cs_dataset = r.get("dataset", "")

                    if data_type == "GWAS":
                        trait = r.get("trait", "")
                        if trait:
                            gwas_counts[trait]["variants"].add(vid)
                            if cs_resource:
                                gwas_counts[trait]["resources"].add(cs_resource)
                            if cs_dataset:
                                gwas_counts[trait]["datasets"].add(cs_dataset)
                            if has_betas and in_beta is not None and cs_beta is not None:
                                if (in_beta > 0) == (cs_beta > 0):
                                    gwas_counts[trait]["consistent"].add(vid)
                                else:
                                    gwas_counts[trait]["inconsistent"].add(vid)

                    elif data_type == "pQTL":
                        gene = r.get("gene_most_severe", "")
                        if gene:
                            pqtl_counts[gene]["variants"].add(vid)
                            pqtl_variants.add(vid)
                            if cs_resource:
                                pqtl_counts[gene]["resources"].add(cs_resource)
                            if cs_dataset:
                                pqtl_counts[gene]["datasets"].add(cs_dataset)
                            if has_betas and in_beta is not None and cs_beta is not None:
                                if (in_beta > 0) == (cs_beta > 0):
                                    pqtl_counts[gene]["consistent"].add(vid)
                                else:
                                    pqtl_counts[gene]["inconsistent"].add(vid)

                    elif data_type == "eQTL":
                        gene = r.get("gene_most_severe", "")
                        tissue = r.get("cell_type", "")
                        if gene and tissue:
                            key = f"{gene}||{tissue}"
                            eqtl_counts[key]["variants"].add(vid)
                            tissue_eqtl_variants[tissue].add(vid)
                            if cs_resource:
                                eqtl_counts[key]["resources"].add(cs_resource)
                            if cs_dataset:
                                eqtl_counts[key]["datasets"].add(cs_dataset)
                            if has_betas and in_beta is not None and cs_beta is not None:
                                if (in_beta > 0) == (cs_beta > 0):
                                    eqtl_counts[key]["consistent"].add(vid)
                                else:
                                    eqtl_counts[key]["inconsistent"].add(vid)

                    elif data_type == "caQTL":
                        tissue = r.get("cell_type", "")
                        if tissue:
                            caqtl_counts[tissue]["variants"].add(vid)
                            if cs_resource:
                                caqtl_counts[tissue]["resources"].add(cs_resource)
                            if cs_dataset:
                                caqtl_counts[tissue]["datasets"].add(cs_dataset)

            # lookup phenotype names for GWAS traits
            trait_codes = list(gwas_counts.keys())
            code_to_name: dict[str, str] = {}
            if trait_codes:
                names_data = await self.lookup_phenotype_names(trait_codes)
                if names_data.get("success"):
                    code_to_name = names_data.get("names", {})

            # build output sorted by count descending
            gwas_phenotypes = sorted(
                [
                    {
                        "trait": trait,
                        "name": code_to_name.get(trait, ""),
                        "resource": ", ".join(sorted(d["resources"])),
                        "dataset": ", ".join(sorted(d["datasets"])),
                        "n_variants": len(d["variants"]),
                        **({"n_consistent": len(d["consistent"]), "n_inconsistent": len(d["inconsistent"])} if has_betas else {}),
                    }
                    for trait, d in gwas_counts.items()
                ],
                key=lambda x: -x["n_variants"],
            )

            pqtl_genes = sorted(
                [
                    {
                        "gene": gene,
                        "resource": ", ".join(sorted(d["resources"])),
                        "dataset": ", ".join(sorted(d["datasets"])),
                        "n_variants": len(d["variants"]),
                        **({"n_consistent": len(d["consistent"]), "n_inconsistent": len(d["inconsistent"])} if has_betas else {}),
                    }
                    for gene, d in pqtl_counts.items()
                ],
                key=lambda x: -x["n_variants"],
            )

            eqtl_genes = sorted(
                [
                    {
                        "gene": key.split("||")[0],
                        "tissue": key.split("||")[1],
                        "resource": ", ".join(sorted(d["resources"])),
                        "dataset": ", ".join(sorted(d["datasets"])),
                        "n_variants": len(d["variants"]),
                        **({"n_consistent": len(d["consistent"]), "n_inconsistent": len(d["inconsistent"])} if has_betas else {}),
                    }
                    for key, d in eqtl_counts.items()
                ],
                key=lambda x: -x["n_variants"],
            )

            caqtl_tissues = sorted(
                [
                    {
                        "tissue": tissue,
                        "resource": ", ".join(sorted(d["resources"])),
                        "dataset": ", ".join(sorted(d["datasets"])),
                        "n_variants": len(d["variants"]),
                    }
                    for tissue, d in caqtl_counts.items()
                ],
                key=lambda x: -x["n_variants"],
            )

            tissue_enrichment = sorted(
                [{"tissue": tissue, "n_eqtl_variants": len(vids)} for tissue, vids in tissue_eqtl_variants.items()],
                key=lambda x: -x["n_eqtl_variants"],
            )

            variant_genes = [
                {"variant": g["variant"], "nearest_gene": g["gene"], "distance": g["distance"]}
                for g in gene_results
            ]

            return {
                "success": True,
                "n_variants": len(variant_ids),
                "n_variants_with_cs": len(variants_with_cs),
                "input_has_betas": has_betas,
                "gwas_phenotypes": gwas_phenotypes,
                "pqtl_genes": pqtl_genes,
                "eqtl_genes": eqtl_genes,
                "caqtl_tissues": caqtl_tissues,
                "tissue_enrichment": tissue_enrichment,
                "pqtl_summary": {"n_variants_with_pqtl": len(pqtl_variants)},
                "variant_genes": variant_genes,
            }

        except Exception as e:
            logger.error(f"Error in analyze_variant_list: {e}\n{traceback.format_exc()}")
            return {"success": False, "error": INTERNAL_ERROR_MSG}

    @staticmethod
    def _parse_variant_list(text: str) -> list[dict[str, Any]]:
        """Parse a variant list with optional beta/se/pvalue columns.

        Accepts one variant per line, tab or comma separated.
        Variant format: chr:pos:ref:alt (chr prefix optional).
        Optional header row detected by first field not matching variant pattern.
        """
        # canonical form uses : as CPRA separator
        variant_pattern = re.compile(r"^(?:chr)?(\d{1,2}|X|Y|MT):(\d+):([ACGT]+):([ACGT]+)$", re.IGNORECASE)
        # accepted CPRA separators: : - _ | / \
        cpra_sep_pattern = re.compile(r"[-_|/\\]")
        results = []

        def _normalize_variant(raw: str) -> re.Match | None:
            """Try to match a variant, normalizing CPRA separators and chr23->X."""
            m = variant_pattern.match(raw)
            if m:
                return m
            normalized = cpra_sep_pattern.sub(":", raw)
            # convert chr23 -> chrX
            normalized = re.sub(r"^(?:chr)?23:", "X:", normalized, flags=re.IGNORECASE)
            return variant_pattern.match(normalized)

        # normalize literal \n (from inline pasting) to actual newlines
        text = text.replace("\\n", "\n")

        # if input is a single line with space-separated variants, split into multiple lines
        raw_lines = text.strip().splitlines()
        if len(raw_lines) == 1 and "\t" not in raw_lines[0] and "," not in raw_lines[0]:
            tokens = raw_lines[0].split()
            if len(tokens) > 1 and all(_normalize_variant(t) for t in tokens):
                raw_lines = tokens

        lines = [line.strip() for line in raw_lines if line.strip()]
        if not lines:
            return []

        # detect field separator: tab > spaces > comma
        if "\t" in lines[0]:
            sep = "\t"
        elif "," in lines[0]:
            sep = ","
        else:
            sep = None  # split on whitespace

        def _split_fields(line: str) -> list[str]:
            if sep is None:
                return line.split()
            return line.split(sep)

        # detect header row
        first_fields = _split_fields(lines[0])
        first_field = first_fields[0].strip()
        has_header = not _normalize_variant(first_field)
        start_idx = 1 if has_header else 0

        # detect column positions from header
        col_map: dict[str, int] = {}
        if has_header:
            for i, col in enumerate(first_fields):
                col_lower = col.strip().lower()
                if col_lower in ("variant", "varid", "var_id", "snp", "id"):
                    col_map["variant"] = i
                elif col_lower in ("beta", "effect"):
                    col_map["beta"] = i
                elif col_lower in ("se", "stderr", "standard_error"):
                    col_map["se"] = i
                elif col_lower in ("pvalue", "p", "pval", "p_value"):
                    col_map["pvalue"] = i

        for line in lines[start_idx:]:
            fields = [f.strip() for f in _split_fields(line)]
            if not fields:
                continue

            # get variant from mapped column or first field
            var_idx = col_map.get("variant", 0)
            if var_idx >= len(fields):
                continue
            raw_var = fields[var_idx]

            m = _normalize_variant(raw_var)
            if not m:
                continue

            chrom = m.group(1)
            if chrom == "23":
                chrom = "X"
            variant_id = f"{chrom}:{m.group(2)}:{m.group(3).upper()}:{m.group(4).upper()}"

            entry: dict[str, Any] = {"variant": variant_id}

            # parse optional stats columns
            for stat_name in ("beta", "se", "pvalue"):
                idx = col_map.get(stat_name, None)
                if idx is None and not has_header:
                    # positional: variant, beta, se, pvalue
                    positional = {"beta": 1, "se": 2, "pvalue": 3}
                    idx = positional.get(stat_name)
                if idx is not None and idx < len(fields):
                    try:
                        entry[stat_name] = float(fields[idx])
                    except (ValueError, TypeError):
                        entry[stat_name] = None
                else:
                    entry[stat_name] = None

            results.append(entry)

        return results

    async def get_nearest_genes(
        self,
        variant: str,
        gene_type: str = "protein_coding",
        n: int = 3,
        max_distance: int = 1000000,
        gencode_version: str | None = None,
        return_hgnc_symbol_if_only_ensg: bool = False,
    ) -> dict[str, Any]:
        """Get genes nearest to a variant."""
        try:
            params: dict[str, Any] = {
                "format": "json",
                "gene_type": gene_type,
                "n": n,
                "max_distance": max_distance,
                "return_hgnc_symbol_if_only_ensg": return_hgnc_symbol_if_only_ensg,
            }
            if gencode_version:
                params["gencode_version"] = gencode_version

            resp = await self.client.get(
                f"{self.base_url}/v1/nearest_genes/{_seg(variant)}",
                params=params,
            )
            if resp.status_code == 200:
                results = resp.json()
                return {
                    "success": True,
                    "variant": variant,
                    "genes": results,
                }
            return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}
        except Exception as e:
            logger.error(
                f"Error in get_nearest_genes({variant}): {e}\n{traceback.format_exc()}"
            )
            return {"success": False, "error": INTERNAL_ERROR_MSG}

    async def get_genes_in_region(
        self,
        chr: str,
        start: int,
        end: int,
        gene_type: str = "protein_coding",
        gencode_version: str | None = None,
    ) -> dict[str, Any]:
        """Get all genes in a genomic region."""
        try:
            params: dict[str, Any] = {
                "format": "json",
                "gene_type": gene_type,
            }
            if gencode_version:
                params["gencode_version"] = gencode_version

            resp = await self.client.get(
                f"{self.base_url}/v1/genes_in_region/{_seg(chr)}/{_seg(start)}/{_seg(end)}",
                params=params,
            )
            if resp.status_code == 200:
                results = resp.json()
                return {
                    "success": True,
                    "region": f"{chr}:{start}-{end}",
                    "genes": results,
                }
            return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}
        except Exception as e:
            logger.error(
                f"Error in get_genes_in_region({chr}:{start}-{end}): {e}\n{traceback.format_exc()}"
            )
            return {"success": False, "error": INTERNAL_ERROR_MSG}

    async def get_gene_group_members(
        self,
        group_id: int | None = None,
        group_name: str | None = None,
        exclude_olfactory: bool = True,
    ) -> dict[str, Any]:
        """Enumerate the member genes of an HGNC gene group by id or name."""
        # exactly one of group_id/group_name is required
        provided = sum(x is not None for x in (group_id, group_name))
        if provided != 1:
            return {
                "success": False,
                "error": "Provide exactly one of 'group_id' or 'group_name'",
            }

        params: dict[str, Any] = (
            {"group_id": group_id} if group_id is not None else {"group_name": group_name}
        )
        # default to excluding olfactory receptors: they are GPCRs that dominate
        # large families by sheer count and are rarely the analysis target
        params["exclude_olfactory"] = exclude_olfactory

        try:
            resp = await self.client.get(
                f"{self.base_url}/v1/gene_group/members", params=params
            )
            if resp.status_code == 200:
                data = resp.json()
                count = data.get("count", 0)
                result: dict[str, Any] = {
                    "success": True,
                    "group_id": data.get("group_id"),
                    "group_name": data.get("group_name"),
                    "exclude_olfactory": data.get("exclude_olfactory", exclude_olfactory),
                    "count": count,
                    "members": data.get("members", []),
                }
                if not count:
                    # the API loaded but has no members for this group yet (gene-group
                    # files may not be loaded) — not an error, just nothing to return
                    result["message"] = (
                        "Gene group has no members yet; gene-group data may not be loaded."
                    )
                return result
            if resp.status_code == 404:
                queried = f"group_id={group_id}" if group_id is not None else f"group_name={group_name!r}"
                return {"success": False, "error": f"Unknown gene group ({queried})"}
            return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}
        except Exception as e:
            logger.error(
                f"Error in get_gene_group_members(group_id={group_id}, group_name={group_name!r}): "
                f"{e}\n{traceback.format_exc()}"
            )
            return {"success": False, "error": INTERNAL_ERROR_MSG}

    async def normalize_gene_symbols(self, symbols: list[str]) -> dict[str, Any]:
        """Resolve previous/alias gene symbols to current approved HGNC symbols."""
        cleaned = [s.strip() for s in (symbols or []) if s and s.strip()]
        if not cleaned:
            return {
                "success": False,
                "error": "Provide a non-empty list of gene symbols",
            }

        # gene symbols never contain commas, so comma-join is a safe query encoding
        joined = ",".join(cleaned)

        try:
            resp = await self.client.get(
                f"{self.base_url}/v1/gene/normalize", params={"symbols": joined}
            )
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "success": True,
                    "mappings": data.get("mappings", []),
                    "unresolved": data.get("unresolved", []),
                }
            return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}
        except Exception as e:
            logger.error(
                f"Error in normalize_gene_symbols(symbols={cleaned}): "
                f"{e}\n{traceback.format_exc()}"
            )
            return {"success": False, "error": INTERNAL_ERROR_MSG}

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

    # -------------------------------------------------------------------------
    # MGI / MouseMine
    # -------------------------------------------------------------------------

    # MouseMine InterMine endpoint — JSON-encoded PathQuery results.
    # Templates would be friendlier but pin us to JAX-named templates that
    # change over releases. Inline PathQuery XML keeps this self-contained.
    _MOUSEMINE_URL = "https://www.mousemine.org/mousemine/service/query/results"
    _MGI_MARKER_URL = "https://www.informatics.jax.org/marker"
    _MGI_ALLELE_URL = "https://www.informatics.jax.org/allele"

    async def search_mgi(
        self,
        query: str,
        query_type: str = "gene_phenotypes",
        species: str = "mouse",
        max_results: int = 25,
    ) -> dict[str, Any]:
        """Search MGI (Mouse Genome Informatics) via MouseMine for curated
        mouse phenotypes, alleles, and human-mouse orthologs.
        """
        size = max(1, min(max_results, 100))
        try:
            if query_type == "gene_phenotypes":
                results = await self._mgi_gene_phenotypes(query, size)
            elif query_type == "phenotype_genes":
                results = await self._mgi_phenotype_genes(query, size)
            elif query_type == "allele":
                results = await self._mgi_allele(query, size)
            elif query_type == "ortholog":
                results = await self._mgi_ortholog(query, species, size)
            else:
                return {
                    "success": False,
                    "error": f"Unknown query_type: {query_type}",
                }

            # _mgi_* helpers may surface an upstream HTTP error as a dict
            if isinstance(results, dict) and results.get("_error"):
                return {"success": False, "error": results["_error"]}

            return {
                "success": True,
                "query": query,
                "query_type": query_type,
                "returned": len(results),
                "results": results,
                "source": "mgi",
            }
        except Exception as e:
            logger.error(
                f"Error in search_mgi({query!r}, type={query_type}): {e}\n"
                f"{traceback.format_exc()}"
            )
            return {"success": False, "error": INTERNAL_ERROR_MSG}

    async def _mousemine_query(
        self, path_query_xml: str, size: int
    ) -> dict[str, Any]:
        """Execute a MouseMine PathQuery and return the parsed JSON body.

        Returns the raw InterMine response dict on success, or a sentinel
        {'_error': '...'} dict on HTTP failure so the caller can surface the
        error without raising.
        """
        params = {
            "query": path_query_xml,
            "format": "json",
            "size": str(size),
        }
        resp = await self.external_client.get(
            self._MOUSEMINE_URL, params=params, timeout=20.0
        )
        if resp.status_code != 200:
            # truncate body to keep error messages bounded
            body = (resp.text or "")[:200]
            return {"_error": f"MouseMine HTTP {resp.status_code}: {body}"}
        return resp.json()

    @staticmethod
    def _marker_url(mgi_id: str | None) -> str | None:
        if not mgi_id:
            return None
        return f"{ToolExecutor._MGI_MARKER_URL}/{mgi_id}"

    @staticmethod
    def _allele_url(mgi_id: str | None) -> str | None:
        if not mgi_id:
            return None
        return f"{ToolExecutor._MGI_ALLELE_URL}/{mgi_id}"

    async def _mgi_gene_phenotypes(
        self, gene_symbol: str, size: int
    ) -> list[dict[str, Any]] | dict[str, Any]:
        """Look up MP phenotype terms and alleles for a gene symbol.

        MouseMine attaches MP annotations to genotypes that involve an allele
        of the gene. We aggregate per-gene so a single gene row carries its
        phenotype term set and allele list.
        """
        # gene -> MP annotations through genotypes; also pull allele identifiers.
        # MouseMine no longer exposes Genotype.phenotypeTerms; MP terms are now
        # reached via Genotype.ontologyAnnotations.ontologyTerm, and the MPTerm
        # type constraint keeps disease (DOID) annotations out of the phenotype
        # set. Mouse is selected by NCBI taxon id (10090) because its organism
        # shortName is "M. musculus/domesticus", not "M. musculus".
        pheno_xml = (
            '<query name="" model="genomic" '
            'view="Gene.primaryIdentifier Gene.symbol Gene.name '
            'Gene.alleles.genotypes.ontologyAnnotations.ontologyTerm.identifier '
            'Gene.alleles.genotypes.ontologyAnnotations.ontologyTerm.name '
            'Gene.alleles.primaryIdentifier Gene.alleles.symbol '
            'Gene.alleles.name">'
            f'<constraint path="Gene.symbol" op="=" value={quoteattr(gene_symbol)}/>'
            '<constraint path="Gene.organism.taxonId" op="=" value="10090"/>'
            '<constraint path="Gene.alleles.genotypes.ontologyAnnotations'
            '.ontologyTerm" type="MPTerm"/>'
            "</query>"
        )
        data = await self._mousemine_query(pheno_xml, size * 50)
        if "_error" in data:
            return data

        rows = data.get("results", [])
        # the view above produces one row per (gene, allele, MP term);
        # collapse by gene
        by_gene: dict[str, dict[str, Any]] = {}
        for r in rows:
            mgi_id = r[0]
            symbol = r[1]
            name = r[2]
            mp_id = r[3]
            mp_term = r[4]
            allele_id = r[5]
            allele_symbol = r[6]
            allele_name = r[7]

            entry = by_gene.setdefault(
                mgi_id or symbol,
                {
                    "mgi_id": mgi_id,
                    "symbol": symbol,
                    "name": name,
                    "phenotype_terms": [],
                    "alleles": [],
                    "url": self._marker_url(mgi_id),
                },
            )
            if mp_id and not any(
                p["mp_id"] == mp_id for p in entry["phenotype_terms"]
            ):
                entry["phenotype_terms"].append({"mp_id": mp_id, "term": mp_term})
            if allele_id and not any(
                a["mgi_id"] == allele_id for a in entry["alleles"]
            ):
                entry["alleles"].append({
                    "mgi_id": allele_id,
                    "symbol": allele_symbol,
                    "name": allele_name,
                    "url": self._allele_url(allele_id),
                })

        return list(by_gene.values())[:size]

    async def _mgi_phenotype_genes(
        self, term_query: str, size: int
    ) -> list[dict[str, Any]] | dict[str, Any]:
        """Look up genes annotated to an MP term (identifier or term name)."""
        # distinguish MP:NNNNNNN identifier from a free-text term name
        is_mp_id = term_query.upper().startswith("MP:")
        if is_mp_id:
            term_constraint = (
                '<constraint path="OntologyAnnotation.ontologyTerm.identifier" '
                f'op="=" value={quoteattr(term_query.upper())}/>'
            )
        else:
            # InterMine string-substring match is LIKE with * wildcards;
            # wrap with * on both sides for case-insensitive substring search
            term_constraint = (
                '<constraint path="OntologyAnnotation.ontologyTerm.name" '
                f'op="LIKE" value={quoteattr(f"*{term_query}*")}/>'
            )

        # MP annotations in MouseMine are attached to Genotype subjects; we
        # traverse subject.alleles.feature to reach the annotated gene. If this
        # path ever returns empty in practice, the alternative to try is
        # OntologyAnnotation.subject.featureGenotypes.feature.
        xml = (
            '<query name="" model="genomic" '
            'view="OntologyAnnotation.ontologyTerm.identifier '
            'OntologyAnnotation.ontologyTerm.name '
            'OntologyAnnotation.subject.alleles.feature.primaryIdentifier '
            'OntologyAnnotation.subject.alleles.feature.symbol '
            'OntologyAnnotation.subject.alleles.feature.name">'
            f"{term_constraint}"
            '<constraint path="OntologyAnnotation.ontologyTerm" type="MPTerm"/>'
            '<constraint path="OntologyAnnotation.subject" type="Genotype"/>'
            "</query>"
        )
        data = await self._mousemine_query(xml, size * 20)
        if "_error" in data:
            return data

        rows = data.get("results", [])
        by_gene: dict[str, dict[str, Any]] = {}
        for r in rows:
            mp_id = r[0]
            mp_term = r[1]
            mgi_id = r[2]
            symbol = r[3]
            name = r[4]
            if not mgi_id:
                continue
            entry = by_gene.setdefault(
                mgi_id,
                {
                    "mgi_id": mgi_id,
                    "symbol": symbol,
                    "name": name,
                    "phenotype_terms": [],
                    "url": self._marker_url(mgi_id),
                },
            )
            if mp_id and not any(
                p["mp_id"] == mp_id for p in entry["phenotype_terms"]
            ):
                entry["phenotype_terms"].append({"mp_id": mp_id, "term": mp_term})

        return list(by_gene.values())[:size]

    async def _mgi_allele(
        self, allele_query: str, size: int
    ) -> list[dict[str, Any]] | dict[str, Any]:
        """Look up allele details by MGI allele ID or symbol."""
        is_mgi_id = allele_query.upper().startswith("MGI:")
        constraint_path = (
            "Allele.primaryIdentifier" if is_mgi_id else "Allele.symbol"
        )
        constraint_value = allele_query.upper() if is_mgi_id else allele_query

        # MP terms hang off Genotype.ontologyAnnotations.ontologyTerm (the old
        # Genotype.phenotypeTerms field is gone); the MPTerm type constraint
        # keeps disease (DOID) annotations out of the phenotype set.
        xml = (
            '<query name="" model="genomic" '
            'view="Allele.primaryIdentifier Allele.symbol Allele.name '
            'Allele.alleleType Allele.feature.primaryIdentifier '
            'Allele.feature.symbol Allele.feature.name '
            'Allele.genotypes.ontologyAnnotations.ontologyTerm.identifier '
            'Allele.genotypes.ontologyAnnotations.ontologyTerm.name">'
            f'<constraint path="{constraint_path}" op="=" '
            f'value={quoteattr(constraint_value)}/>'
            '<constraint path="Allele.genotypes.ontologyAnnotations'
            '.ontologyTerm" type="MPTerm"/>'
            "</query>"
        )
        data = await self._mousemine_query(xml, size * 50)
        if "_error" in data:
            return data

        rows = data.get("results", [])
        by_allele: dict[str, dict[str, Any]] = {}
        for r in rows:
            allele_id = r[0]
            allele_symbol = r[1]
            allele_name = r[2]
            allele_type = r[3]
            gene_id = r[4]
            gene_symbol = r[5]
            gene_name = r[6]
            mp_id = r[7]
            mp_term = r[8]

            entry = by_allele.setdefault(
                allele_id or allele_symbol,
                {
                    "mgi_id": gene_id,
                    "symbol": gene_symbol,
                    "name": gene_name,
                    "alleles": [{
                        "mgi_id": allele_id,
                        "symbol": allele_symbol,
                        "name": allele_name,
                        "allele_type": allele_type,
                        "url": self._allele_url(allele_id),
                    }],
                    "phenotype_terms": [],
                    "url": self._marker_url(gene_id),
                },
            )
            if mp_id and not any(
                p["mp_id"] == mp_id for p in entry["phenotype_terms"]
            ):
                entry["phenotype_terms"].append({"mp_id": mp_id, "term": mp_term})

        return list(by_allele.values())[:size]

    async def _mgi_ortholog(
        self, gene_symbol: str, species: str, size: int
    ) -> list[dict[str, Any]] | dict[str, Any]:
        """Look up mouse-human orthologs of a gene symbol.

        species='human' means the input is a human gene and we want mouse
        orthologs; species='mouse' is the inverse direction.
        """
        # Select the source organism by NCBI taxon id (human 9606, mouse 10090)
        # rather than shortName, since mouse's shortName is "M. musculus/
        # domesticus" and an "M. musculus" match returns nothing.
        source_taxon = "9606" if species == "human" else "10090"
        xml = (
            '<query name="" model="genomic" '
            'view="Gene.primaryIdentifier Gene.symbol Gene.name '
            'Gene.homologues.homologue.primaryIdentifier '
            'Gene.homologues.homologue.symbol '
            'Gene.homologues.homologue.name '
            'Gene.homologues.homologue.organism.shortName">'
            f'<constraint path="Gene.symbol" op="=" value={quoteattr(gene_symbol)}/>'
            f'<constraint path="Gene.organism.taxonId" op="=" '
            f'value="{source_taxon}"/>'
            "</query>"
        )
        data = await self._mousemine_query(xml, size * 10)
        if "_error" in data:
            return data

        rows = data.get("results", [])
        by_gene: dict[str, dict[str, Any]] = {}
        for r in rows:
            mgi_id = r[0]
            symbol = r[1]
            name = r[2]
            ortho_id = r[3]
            ortho_symbol = r[4]
            ortho_name = r[5]
            ortho_organism = r[6]

            entry = by_gene.setdefault(
                mgi_id or symbol,
                {
                    "mgi_id": mgi_id,
                    "symbol": symbol,
                    "name": name,
                    "orthologs": [],
                    "url": self._marker_url(mgi_id),
                },
            )
            if ortho_id and not any(
                o["mgi_id"] == ortho_id for o in entry["orthologs"]
            ):
                entry["orthologs"].append({
                    "mgi_id": ortho_id,
                    "symbol": ortho_symbol,
                    "name": ortho_name,
                    "organism": ortho_organism,
                })

        return list(by_gene.values())[:size]

    # -------------------------------------------------------------------------
    # cBioPortal
    # -------------------------------------------------------------------------

    # Public cBioPortal REST API. Public studies need no authentication; the data
    # is ODbL-licensed, so every result carries the attribution string below and
    # study-level citations where we have them.
    _CBIO_API = "https://www.cbioportal.org/api"
    _CBIO_STUDY_URL = "https://www.cbioportal.org/study/summary"

    _CBIO_ATTRIBUTION = (
        "Data from cBioPortal (ODbL). Cite cBioPortal and the originating studies."
    )

    # cBioPortal is majority GRCh37 and does not lift over, so genomic coordinates
    # from it must never be compared against this suite's GRCh38 positions. Gene
    # symbols and protein changes are build-independent, which is why every query
    # type here keys on those and reports coordinates only tagged with their build.
    _CBIO_BUILD_NOTE = (
        "cBioPortal reports each record on the build of its source study (mostly "
        "GRCh37) and does not lift over. This suite is GRCh38, so do NOT compare "
        "these genomic coordinates with GRCh38 positions. Gene symbol and protein "
        "change are build-independent — match on those. To go from a GRCh38 "
        "variant to a protein change first, use get_variant_protein_effect."
    )

    # mutations/fetch returns one record per mutated sample, so a hot gene is huge
    # pan-cancer (TP53 is ~142k records / ~100 MB). Above this many records we drop
    # to the curated cohort below, which keeps TP53 at ~23k records / ~17 MB.
    _CBIO_MAX_MUTATION_RECORDS = 25000

    # TCGA PanCancer Atlas plus the MSK pan-cancer cohorts: large, broadly
    # non-overlapping, and deep enough to rank hotspots for genes too hot to fetch
    # whole. Used only as the fallback scope, never silently for normal genes.
    _CBIO_CURATED_STUDY_IDS = frozenset({"msk_impact_2017", "msk_chord_2024"})
    _CBIO_CURATED_SUFFIX = "pan_can_atlas"

    # discrete CNA levels as returned by genomic-data-counts
    _CBIO_CNA_LABELS = {
        "2": "amplification",
        "1": "gain",
        "0": "diploid",
        "-1": "shallow_deletion",
        "-2": "deep_deletion",
    }

    async def search_cbioportal(
        self,
        query: str,
        query_type: str = "gene_summary",
        cancer_types: list[str] | None = None,
        max_results: int = 25,
    ) -> dict[str, Any]:
        """Search cBioPortal for somatic alteration frequencies in cancer cohorts."""
        size = max(1, min(max_results, 100))
        try:
            if query_type == "gene_summary":
                result = await self._cbio_gene_summary(query)
            elif query_type == "gene_by_cancer_type":
                result = await self._cbio_gene_by_cancer_type(
                    query, cancer_types, size
                )
            elif query_type == "gene_mutations":
                result = await self._cbio_gene_mutations(query, size)
            elif query_type == "gene_fusions":
                result = await self._cbio_gene_fusions(query, size)
            elif query_type == "variant_hotspot":
                result = await self._cbio_variant_hotspot(query)
            elif query_type == "study_search":
                result = await self._cbio_study_search(query, size)
            else:
                return {
                    "success": False,
                    "error": f"Unknown query_type: {query_type}",
                }

            if isinstance(result, dict) and result.get("_error"):
                return {"success": False, "error": result["_error"]}

            return {
                "success": True,
                "query": query,
                "query_type": query_type,
                "source": "cbioportal",
                "attribution": self._CBIO_ATTRIBUTION,
                **result,
            }
        except Exception as e:
            logger.error(
                f"Error in search_cbioportal({query!r}, type={query_type}): {e}\n"
                f"{traceback.format_exc()}"
            )
            return {"success": False, "error": INTERNAL_ERROR_MSG}

    # --- transport -----------------------------------------------------------

    async def _cbio_get(
        self, path: str, params: dict[str, Any] | None = None, timeout: float = 30.0
    ) -> Any:
        """GET a cBioPortal endpoint. Returns parsed JSON, or a {'_error': ...}
        sentinel on HTTP failure so callers can surface it without raising."""
        resp = await self.external_client.get(
            f"{self._CBIO_API}{path}", params=params, timeout=timeout
        )
        if resp.status_code != 200:
            return {"_error": f"cBioPortal HTTP {resp.status_code}: {(resp.text or '')[:200]}"}
        return resp.json()

    async def _cbio_post(
        self,
        path: str,
        body: Any,
        params: dict[str, Any] | None = None,
        timeout: float = 120.0,
    ) -> Any:
        """POST to a cBioPortal endpoint, with the same error sentinel as _cbio_get.

        The timeout is generous because the pan-cancer fetches legitimately take
        several seconds against 539 studies.
        """
        resp = await self.external_client.post(
            f"{self._CBIO_API}{path}", json=body, params=params, timeout=timeout
        )
        if resp.status_code != 200:
            return {"_error": f"cBioPortal HTTP {resp.status_code}: {(resp.text or '')[:200]}"}
        return resp.json()

    async def _cbio_count(self, path: str, body: Any) -> int | dict[str, Any]:
        """Ask how many records a fetch would return, without downloading them.

        projection=META returns an empty body and a total-count header, which is
        what makes the size guard in _cbio_gene_mutations cheap (~0.4s).
        """
        resp = await self.external_client.post(
            f"{self._CBIO_API}{path}",
            json=body,
            params={"projection": "META"},
            timeout=60.0,
        )
        if resp.status_code != 200:
            return {"_error": f"cBioPortal HTTP {resp.status_code}: {(resp.text or '')[:200]}"}
        try:
            return int(resp.headers.get("total-count", 0))
        except ValueError:
            return {"_error": "cBioPortal returned a non-numeric total-count header"}

    # --- cached metadata -----------------------------------------------------

    async def _cbio_get_studies(self) -> list[dict[str, Any]] | dict[str, Any]:
        if self._cbio_studies is None:
            data = await self._cbio_get("/studies", {"projection": "SUMMARY"})
            if isinstance(data, dict):
                return data
            self._cbio_studies = data
        return self._cbio_studies

    async def _cbio_get_profiles(self) -> list[dict[str, Any]] | dict[str, Any]:
        if self._cbio_profiles is None:
            data = await self._cbio_get("/molecular-profiles", {"pageSize": 10000})
            if isinstance(data, dict):
                return data
            self._cbio_profiles = data
        return self._cbio_profiles

    async def _cbio_profile_ids(
        self, alteration_type: str, study_ids: set[str] | None = None
    ) -> list[str] | dict[str, Any]:
        profiles = await self._cbio_get_profiles()
        if isinstance(profiles, dict):
            return profiles
        return [
            p["molecularProfileId"]
            for p in profiles
            if p.get("molecularAlterationType") == alteration_type
            and (study_ids is None or p.get("studyId") in study_ids)
        ]

    async def _cbio_resolve_gene(self, symbol: str) -> dict[str, Any]:
        """Resolve a HUGO symbol to its Entrez gene id."""
        data = await self._cbio_get(f"/genes/{quote(symbol.strip().upper(), safe='')}")
        if isinstance(data, dict) and data.get("_error"):
            if "HTTP 404" in data["_error"]:
                return {"_error": f"Gene not found in cBioPortal: {symbol}"}
            return data
        return {
            "hugo_symbol": data.get("hugoGeneSymbol"),
            "entrez_gene_id": data.get("entrezGeneId"),
            "gene_type": data.get("type"),
        }

    @staticmethod
    def _cbio_normalize_label(label: str) -> str:
        """Fold the free-text CANCER_TYPE values studies use into one key.

        Studies spell the same disease differently ("Non Small Cell Lung Cancer"
        vs "Non-Small Cell Lung Cancer"), and leaving them split understates the
        sample counts of the biggest cancer types.
        """
        return re.sub(r"[^a-z0-9]+", " ", (label or "").lower()).strip()

    @staticmethod
    def _cbio_gene_filter(entrez_gene_id: int, symbol: str, profile_ids: list[str]) -> dict[str, Any]:
        """StudyViewFilter gene filter selecting samples altered in one gene.

        The include* flags mirror what the cBioPortal web UI sends with all
        annotation filters switched off — omitting them silently drops mutations
        whose driver status or germline/somatic status is unknown.
        """
        return {
            "molecularProfileIds": profile_ids,
            "geneQueries": [[{
                "hugoGeneSymbol": symbol,
                "entrezGeneId": entrez_gene_id,
                "includeSomatic": True,
                "includeGermline": True,
                "includeUnknownStatus": True,
                "includeDriver": True,
                "includeVUS": True,
                "includeUnknownOncogenicity": True,
                "includeUnknownTier": True,
                "tiersBooleanMap": {},
            }]],
        }

    @staticmethod
    def _cbio_counts_by_label(payload: Any) -> dict[str, dict[str, Any]]:
        """Fold a clinical-data-counts response into {normalized: {label, count}}."""
        entries = payload if isinstance(payload, list) else [payload]
        merged: dict[str, dict[str, Any]] = {}
        for entry in entries:
            for c in (entry or {}).get("counts", []):
                key = ToolExecutor._cbio_normalize_label(c.get("value", ""))
                if not key:
                    continue
                row = merged.setdefault(key, {"label": c.get("value"), "count": 0})
                row["count"] += c.get("count", 0)
        return merged

    async def _cbio_cancer_type_denominators(self) -> dict[str, dict[str, Any]]:
        """Samples with mutation data, broken down by cancer type.

        Gene-independent, so it is fetched once and reused. genomicProfiles
        restricts to samples that actually have a mutation profile — without it
        the counts include unsequenced samples and every frequency comes out low.
        """
        if self._cbio_denominators is None:
            studies = await self._cbio_get_studies()
            if isinstance(studies, dict):
                return studies
            body = {
                "attributes": [{"attributeId": "CANCER_TYPE"}],
                "studyViewFilter": {
                    "studyIds": [s["studyId"] for s in studies],
                    "genomicProfiles": [["mutations"]],
                },
            }
            data = await self._cbio_post("/clinical-data-counts/fetch", body)
            if isinstance(data, dict) and data.get("_error"):
                return data
            self._cbio_denominators = self._cbio_counts_by_label(data)
        return self._cbio_denominators

    # --- query handlers ------------------------------------------------------

    async def _cbio_gene_summary(self, symbol: str) -> dict[str, Any]:
        """Pan-cancer somatic mutation and copy-number frequency for one gene."""
        gene = await self._cbio_resolve_gene(symbol)
        if gene.get("_error"):
            return gene

        studies = await self._cbio_get_studies()
        if isinstance(studies, dict):
            return studies
        study_ids = [s["studyId"] for s in studies]

        gene_filter = [{"hugoGeneSymbol": gene["hugo_symbol"], "profileType": "mutations"}]
        cna_filter = [{"hugoGeneSymbol": gene["hugo_symbol"], "profileType": "cna"}]
        mut_body = {"genomicDataFilters": gene_filter, "studyViewFilter": {"studyIds": study_ids}}
        cna_body = {"genomicDataFilters": cna_filter, "studyViewFilter": {"studyIds": study_ids}}

        mut_data, cna_data = await asyncio.gather(
            self._cbio_post("/mutation-data-counts/fetch", mut_body, {"projection": "SUMMARY"}),
            self._cbio_post("/genomic-data-counts/fetch", cna_body),
        )
        if isinstance(mut_data, dict) and mut_data.get("_error"):
            return mut_data

        result: dict[str, Any] = {"gene": gene, "pan_cancer": {}}

        counts = {c["value"]: c["count"] for c in (mut_data[0]["counts"] if mut_data else [])}
        mutated = counts.get("MUTATED", 0)
        profiled = mutated + counts.get("NOT_MUTATED", 0)
        result["pan_cancer"]["mutation"] = {
            "altered_samples": mutated,
            "profiled_samples": profiled,
            "frequency": round(mutated / profiled, 5) if profiled else None,
            "not_profiled_samples": counts.get("NOT_PROFILED", 0),
        }

        # CNA counts are per discrete level; "NA" means the sample has no CNA call
        # for this gene and must be excluded from the denominator.
        if not (isinstance(cna_data, dict) and cna_data.get("_error")) and cna_data:
            levels = {c["value"]: c["count"] for c in cna_data[0].get("counts", [])}
            cna_profiled = sum(v for k, v in levels.items() if k != "NA")
            result["pan_cancer"]["copy_number"] = {
                "profiled_samples": cna_profiled,
                "levels": {
                    self._CBIO_CNA_LABELS.get(k, k): {
                        "samples": v,
                        "frequency": round(v / cna_profiled, 5) if cna_profiled else None,
                    }
                    for k, v in levels.items()
                    if k != "NA" and k != "0"
                },
            }

        result["cohort"] = self._cbio_cohort_summary(studies)
        result["genome_build_note"] = self._CBIO_BUILD_NOTE
        result["url"] = (
            f"https://www.cbioportal.org/results/cancerTypesSummary"
            f"?gene_list={quote(gene['hugo_symbol'] or symbol)}"
        )
        return result

    async def _cbio_gene_by_cancer_type(
        self, symbol: str, cancer_types: list[str] | None, size: int
    ) -> dict[str, Any]:
        """Somatic mutation frequency for one gene, broken down by cancer type.

        Cancer type comes from each sample's CANCER_TYPE clinical attribute rather
        than its study's headline cancer type — otherwise the large pan-cancer
        cohorts (MSK-IMPACT is ~10k samples) all collapse into "Mixed".
        """
        gene = await self._cbio_resolve_gene(symbol)
        if gene.get("_error"):
            return gene

        studies = await self._cbio_get_studies()
        if isinstance(studies, dict):
            return studies
        profile_ids = await self._cbio_profile_ids("MUTATION_EXTENDED")
        if isinstance(profile_ids, dict):
            return profile_ids

        num_body = {
            "attributes": [{"attributeId": "CANCER_TYPE"}],
            "studyViewFilter": {
                "studyIds": [s["studyId"] for s in studies],
                "geneFilters": [
                    self._cbio_gene_filter(
                        gene["entrez_gene_id"], gene["hugo_symbol"], profile_ids
                    )
                ],
            },
        }
        num_data, denominators = await asyncio.gather(
            self._cbio_post("/clinical-data-counts/fetch", num_body),
            self._cbio_cancer_type_denominators(),
        )
        if isinstance(num_data, dict) and num_data.get("_error"):
            return num_data
        if isinstance(denominators, dict) and denominators.get("_error"):
            return denominators

        numerators = self._cbio_counts_by_label(num_data)
        wanted = (
            {self._cbio_normalize_label(c) for c in cancer_types}
            if cancer_types
            else None
        )

        rows = []
        for key, num in numerators.items():
            if wanted is not None and key not in wanted:
                continue
            den = denominators.get(key)
            if not den or not den["count"]:
                continue
            rows.append({
                "cancer_type": den["label"],
                "altered_samples": num["count"],
                "profiled_samples": den["count"],
                "frequency": round(num["count"] / den["count"], 5),
            })

        # a 2-of-3 "frequency" from a tiny cohort outranks every real signal
        min_cohort = 100
        ranked = sorted(
            (r for r in rows if r["profiled_samples"] >= min_cohort),
            key=lambda r: -r["frequency"],
        )
        small = [r for r in rows if r["profiled_samples"] < min_cohort]

        return {
            "gene": gene,
            "returned": len(ranked[:size]),
            "results": ranked[:size],
            "cancer_types_matched": len(rows),
            "cancer_types_below_min_cohort": len(small),
            "min_cohort_samples": min_cohort,
            "denominator_basis": (
                "samples with any mutation profile in that cancer type. This is not "
                "gene-panel aware: a sample sequenced on a panel that omits this gene "
                "still counts in the denominator, so frequencies are lower bounds. "
                "Compare against not_profiled_samples from gene_summary to judge how "
                "much panel coverage varies for this gene."
            ),
            "genome_build_note": self._CBIO_BUILD_NOTE,
            "url": (
                f"https://www.cbioportal.org/results/cancerTypesSummary"
                f"?gene_list={quote(gene['hugo_symbol'] or symbol)}"
            ),
        }

    async def _cbio_gene_mutations(self, symbol: str, size: int) -> dict[str, Any]:
        """Recurrent protein changes (hotspots) for one gene across all cohorts."""
        gene = await self._cbio_resolve_gene(symbol)
        if gene.get("_error"):
            return gene

        profile_ids = await self._cbio_profile_ids("MUTATION_EXTENDED")
        if isinstance(profile_ids, dict):
            return profile_ids

        body = {"molecularProfileIds": profile_ids, "entrezGeneIds": [gene["entrez_gene_id"]]}
        total = await self._cbio_count("/mutations/fetch", body)
        if isinstance(total, dict):
            return total

        scope = "all_studies"
        if total > self._CBIO_MAX_MUTATION_RECORDS:
            studies = await self._cbio_get_studies()
            if isinstance(studies, dict):
                return studies
            curated = {
                s["studyId"]
                for s in studies
                if s["studyId"] in self._CBIO_CURATED_STUDY_IDS
                or self._CBIO_CURATED_SUFFIX in s["studyId"]
            }
            curated_ids = await self._cbio_profile_ids("MUTATION_EXTENDED", curated)
            if isinstance(curated_ids, dict):
                return curated_ids
            body = {"molecularProfileIds": curated_ids, "entrezGeneIds": [gene["entrez_gene_id"]]}
            scope = "curated_cohort"

        data = await self._cbio_post("/mutations/fetch", body, {"projection": "SUMMARY"})
        if isinstance(data, dict) and data.get("_error"):
            return data

        by_change: dict[str, dict[str, Any]] = {}
        seen: dict[str, set[str]] = defaultdict(set)
        builds: dict[str, int] = defaultdict(int)
        for rec in data:
            change = rec.get("proteinChange") or "(unspecified)"
            entry = by_change.setdefault(change, {
                "protein_change": change,
                "protein_position": rec.get("proteinPosStart"),
                "mutation_type": rec.get("mutationType"),
                "sample_count": 0,
                "coordinates_by_build": {},
            })
            key = rec.get("uniqueSampleKey") or f"{rec.get('studyId')}:{rec.get('sampleId')}"
            if key in seen[change]:
                continue
            seen[change].add(key)
            entry["sample_count"] += 1

            build = rec.get("ncbiBuild") or "unknown"
            builds[build] += 1
            # keyed by build and never merged: the same protein change sits at
            # different genomic positions in GRCh37 and GRCh38
            entry["coordinates_by_build"].setdefault(build, {
                "chromosome": rec.get("chr"),
                "start_position": rec.get("startPosition"),
                "reference_allele": rec.get("referenceAllele"),
                "variant_allele": rec.get("variantAllele"),
            })

        ranked = sorted(by_change.values(), key=lambda e: -e["sample_count"])
        return {
            "gene": gene,
            "returned": len(ranked[:size]),
            "results": ranked[:size],
            "distinct_protein_changes": len(by_change),
            "total_mutation_records": total,
            "records_analyzed": len(data),
            "scope": scope,
            "scope_note": (
                "Counts come from the TCGA PanCancer Atlas and MSK pan-cancer "
                "cohorts, not every study: the full set exceeded "
                f"{self._CBIO_MAX_MUTATION_RECORDS} records. Rankings hold, absolute "
                "counts are lower than pan-cancer totals."
                if scope == "curated_cohort"
                else "All cBioPortal studies with mutation data."
            ),
            "genome_builds": dict(builds),
            "genome_build_note": self._CBIO_BUILD_NOTE,
            "url": (
                f"https://www.cbioportal.org/results/mutations"
                f"?gene_list={quote(gene['hugo_symbol'] or symbol)}"
            ),
        }

    async def _cbio_gene_fusions(self, symbol: str, size: int) -> dict[str, Any]:
        """Structural-variant (fusion) partners observed for one gene."""
        gene = await self._cbio_resolve_gene(symbol)
        if gene.get("_error"):
            return gene

        profile_ids = await self._cbio_profile_ids("STRUCTURAL_VARIANT")
        if isinstance(profile_ids, dict):
            return profile_ids

        data = await self._cbio_post(
            "/structural-variant/fetch",
            {"molecularProfileIds": profile_ids, "entrezGeneIds": [gene["entrez_gene_id"]]},
        )
        if isinstance(data, dict) and data.get("_error"):
            return data

        partners: dict[str, dict[str, Any]] = {}
        for rec in data:
            site1 = rec.get("site1HugoSymbol") or ""
            site2 = rec.get("site2HugoSymbol") or ""
            partner = site2 if site1 == gene["hugo_symbol"] else site1
            partner = partner or "(intergenic)"
            entry = partners.setdefault(partner, {
                "partner_gene": partner,
                "sample_count": 0,
                "studies": set(),
                "variant_classes": set(),
            })
            entry["sample_count"] += 1
            entry["studies"].add(rec.get("studyId"))
            if rec.get("variantClass"):
                entry["variant_classes"].add(rec["variantClass"])

        ranked = sorted(partners.values(), key=lambda e: -e["sample_count"])[:size]
        for e in ranked:
            e["study_count"] = len(e["studies"])
            e["studies"] = sorted(s for s in e["studies"] if s)[:10]
            e["variant_classes"] = sorted(e["variant_classes"])

        return {
            "gene": gene,
            "returned": len(ranked),
            "results": ranked,
            "total_structural_variant_records": len(data),
            "distinct_partners": len(partners),
            "genome_build_note": self._CBIO_BUILD_NOTE,
            "url": (
                f"https://www.cbioportal.org/results/structuralVariants"
                f"?gene_list={quote(gene['hugo_symbol'] or symbol)}"
            ),
        }

    async def _cbio_variant_hotspot(self, query: str) -> dict[str, Any]:
        """How many samples carry a somatic mutation at one protein residue.

        Accepts 'TP53 R175H', 'TP53 R175' or 'TP53 175' — only the residue number
        is used for the count, so all three give the recurrence at that position.
        """
        parts = query.replace(":", " ").split()
        if len(parts) < 2:
            return {"_error": (
                "variant_hotspot needs a gene and a residue, e.g. 'TP53 R175H' or "
                "'TP53 175'."
            )}
        symbol, residue = parts[0], parts[1]
        match = re.search(r"(\d+)", residue)
        if not match:
            return {"_error": f"Could not read a residue position from {residue!r}."}
        position = int(match.group(1))

        gene = await self._cbio_resolve_gene(symbol)
        if gene.get("_error"):
            return gene

        data = await self._cbio_post(
            "/mutation-counts-by-position/fetch",
            [{
                "entrezGeneId": gene["entrez_gene_id"],
                "proteinPosStart": position,
                "proteinPosEnd": position,
            }],
        )
        if isinstance(data, dict) and data.get("_error"):
            return data

        count = data[0].get("count", 0) if data else 0
        return {
            "gene": gene,
            "protein_position": position,
            "sample_count": count,
            "note": (
                "Samples with any somatic mutation at this residue, across all "
                "cBioPortal studies — not restricted to one amino-acid substitution. "
                "Use gene_mutations for the per-substitution breakdown."
            ),
            "genome_build_note": self._CBIO_BUILD_NOTE,
            "url": (
                f"https://www.cbioportal.org/results/mutations"
                f"?gene_list={quote(gene['hugo_symbol'] or symbol)}"
            ),
        }

    async def _cbio_study_search(self, query: str, size: int) -> dict[str, Any]:
        """Find cBioPortal studies whose name, id or cancer type matches a term."""
        studies = await self._cbio_get_studies()
        if isinstance(studies, dict):
            return studies

        term = query.strip().lower()
        matches = [
            s for s in studies
            if term in (s.get("name") or "").lower()
            or term in (s.get("studyId") or "").lower()
            or term in (s.get("cancerTypeId") or "").lower()
            or term in (s.get("description") or "").lower()
        ]
        matches.sort(key=lambda s: -(s.get("allSampleCount") or 0))

        return {
            "returned": len(matches[:size]),
            "matched": len(matches),
            "results": [{
                "study_id": s.get("studyId"),
                "name": s.get("name"),
                "cancer_type_id": s.get("cancerTypeId"),
                "sample_count": s.get("allSampleCount"),
                "reference_genome": s.get("referenceGenome"),
                "citation": s.get("citation"),
                "pmid": s.get("pmid"),
                "url": f"{self._CBIO_STUDY_URL}?id={quote(s.get('studyId') or '')}",
            } for s in matches[:size]],
            "cohort": self._cbio_cohort_summary(studies),
        }

    @staticmethod
    def _cbio_cohort_summary(studies: list[dict[str, Any]]) -> dict[str, Any]:
        """Size and genome-build makeup of the cohort behind a result.

        Derived from the live study list rather than hardcoded, since cBioPortal
        adds studies and migrates them to hg38 over time.
        """
        builds: dict[str, int] = defaultdict(int)
        for s in studies:
            builds[s.get("referenceGenome") or "unknown"] += 1
        return {
            "studies": len(studies),
            "samples": sum(s.get("allSampleCount") or 0 for s in studies),
            "studies_by_reference_genome": dict(builds),
        }

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _prioritize_variants(self, results: list) -> list:
        """Sort results to show most informative variants first."""
        high_priority = {
            "missense_variant",
            "frameshift_variant",
            "stop_gained",
            "stop_lost",
            "start_lost",
            "splice_acceptor_variant",
            "splice_donor_variant",
            "splice_region_variant",
            "inframe_insertion",
            "inframe_deletion",
        }
        medium_priority = {
            "synonymous_variant",
            "5_prime_UTR_variant",
            "3_prime_UTR_variant",
        }

        def sort_key(item):
            consequence = item.get("most_severe") or ""
            pip = item.get("pip") or 0
            mlog10p = item.get("mlog10p") or 0

            if consequence in high_priority:
                priority = 0
            elif consequence in medium_priority:
                priority = 1
            else:
                priority = 2
            return (priority, -pip, -mlog10p)

        return sorted(results, key=sort_key)

    def _summarize_credible_sets_trait(self, tsv_data: str) -> dict:
        """Summarize variant-level TSV data into credible set-level summary with coding/LoF counts."""
        import io

        import polars as pl

        df = pl.read_csv(
            io.StringIO(tsv_data),
            separator="\t",
            null_values=["NA"],
            infer_schema_length=None,
        )

        if df.is_empty():
            return {"n_cs": 0, "cs": []}

        gene_counts = (
            df.group_by(["cs_id", "gene_most_severe", "most_severe"])
            .len()
            .group_by(["cs_id", "gene_most_severe"])
            .agg(pl.struct(["most_severe", "len"]).alias("consequence_count"))
            .group_by("cs_id")
            .agg(
                pl.struct(["gene_most_severe", "consequence_count"]).alias(
                    "gene_consequence_counts"
                )
            )
        )

        # get lead variant (max pip, then max mlog10p as tiebreaker) for each cs_id
        # use sort_by().first() inside agg to correctly sort within each group
        lead_variant_cols = [
            "cs_size",
            "chr",
            "pos",
            "ref",
            "alt",
            "mlog10p",
            "beta",
            "se",
            "pip",
            "aaf",
            "most_severe",
            "gene_most_severe",
        ]
        lead_variants = df.group_by("cs_id").agg(
            [
                pl.col(c)
                .sort_by(["pip", "mlog10p"], descending=[True, True], nulls_last=True)
                .first()
                for c in lead_variant_cols
            ]
        )

        # aggregate stats from all variants in each credible set
        cs_stats = df.group_by("cs_id").agg(
            [
                pl.col("aaf").min().alias("min_aaf"),
                pl.col("aaf").max().alias("max_aaf"),
                pl.col("most_severe").is_in(CODING_VARIANTS).sum().alias("n_coding"),
                pl.col("most_severe").is_in(LOF_VARIANTS).sum().alias("n_lof"),
            ]
        )

        result = (
            lead_variants.join(cs_stats, on="cs_id")
            .join(gene_counts, on="cs_id", how="left")
            .sort("mlog10p", descending=True, nulls_last=True)
        )

        summaries = []
        for row in result.to_dicts():
            gcc = []
            for gc in row.get("gene_consequence_counts") or []:
                gene = gc.get("gene_most_severe") or "unknown"
                counts = {
                    item["most_severe"]: item["len"]
                    for item in gc.get("consequence_count") or []
                }
                gcc.append({"gene": gene, "consequence_counts": counts})

            summaries.append(
                {
                    "cs_id": row["cs_id"],
                    "cs_size": row["cs_size"],
                    "n_coding": row["n_coding"],
                    "n_lof": row["n_lof"],
                    "min_aaf": row["min_aaf"],
                    "max_aaf": row["max_aaf"],
                    "lead_variant": {
                        "id": f"{row['chr']}:{row['pos']}:{row['ref']}:{row['alt']}",
                        "mlog10p": row["mlog10p"],
                        "beta": row["beta"],
                        "se": row["se"],
                        "pip": row["pip"],
                        "aaf": row["aaf"],
                        "most_severe": row["most_severe"],
                        "gene_most_severe": row["gene_most_severe"],
                    },
                    "gene_consequence_counts": gcc,
                }
            )

        return {"n_cs": len(summaries), "cs": summaries}

    def _summarize_credible_sets_simple(self, tsv_data: str) -> dict:
        """Simpler summary with coding/LoF counts, grouped by data_type."""
        import io

        import polars as pl

        df = pl.read_csv(
            io.StringIO(tsv_data),
            separator="\t",
            null_values=["NA"],
            infer_schema_length=None,
        )

        if df.is_empty():
            return {"n_cs": 0, "cs": {}}

        # A cs_id identifies a credible set only WITHIN one dataset's fine-mapping run of
        # one trait in one cell type; it is not globally unique. caQTL cs_ids are derived
        # from the peak, so the same one recurs in every cell type the peak was tested in
        # (IL7R: 46 distinct cs_ids but 129 real credible sets), and eQTL Catalogue cs_ids
        # like ENSG00000187608_L1 recur across QTD studies. Grouping on cs_id alone merged
        # those into one row each, undercounting credible sets by up to 4x and silently
        # dropping whole cell types from the summary (13 -> 9 for IL7R caQTL).
        key_cols = [
            c for c in ("resource", "dataset", "trait", "cell_type") if c in df.columns
        ]
        key_cols.append("cs_id")

        lead_variant_cols = [
            c
            for c in (
                "cs_size",
                "resource",
                "data_type",
                "cell_type",
                "trait",
                "chr",
                "pos",
                "ref",
                "alt",
                "mlog10p",
                "beta",
                "se",
                "pip",
                "aaf",
                "most_severe",
                "gene_most_severe",
            )
            if c not in key_cols
        ]

        # both aggregations in one pass: joining them would have to match on nullable key
        # columns (cell_type is null for GWAS), where null != null drops rows
        result = df.group_by(key_cols).agg(
            [
                pl.col(c)
                .sort_by(["pip", "mlog10p"], descending=[True, True], nulls_last=True)
                .first()
                for c in lead_variant_cols
            ]
            + [
                pl.col("aaf").min().alias("min_aaf"),
                pl.col("aaf").max().alias("max_aaf"),
                pl.col("most_severe").is_in(CODING_VARIANTS).sum().alias("n_coding"),
                pl.col("most_severe").is_in(LOF_VARIANTS).sum().alias("n_lof"),
            ]
        ).sort("mlog10p", descending=True, nulls_last=True)

        # group by data_type
        grouped: dict[str, list] = {}
        for row in result.to_dicts():
            data_type = row["data_type"] or "unknown"
            if data_type not in grouped:
                grouped[data_type] = []
            grouped[data_type].append(
                {
                    "cs_id": row["cs_id"],
                    "cs_size": row["cs_size"],
                    "resource": row["resource"],
                    "cell_type": row["cell_type"],
                    "trait": row["trait"],
                    "n_coding": row["n_coding"],
                    "n_lof": row["n_lof"],
                    "min_aaf": row["min_aaf"],
                    "max_aaf": row["max_aaf"],
                    "lead_variant": {
                        "id": f"{row['chr']}:{row['pos']}:{row['ref']}:{row['alt']}",
                        "mlog10p": row["mlog10p"],
                        "beta": row["beta"],
                        "se": row["se"],
                        "pip": row["pip"],
                        "aaf": row["aaf"],
                        "most_severe": row["most_severe"],
                        "gene_most_severe": row["gene_most_severe"],
                    },
                }
            )

        total_cs = sum(len(cs_list) for cs_list in grouped.values())
        return {
            "n_cs": total_cs,
            "counts": self._summary_counts(df, grouped),
            "cs": grouped,
        }

    @staticmethod
    def _summary_counts(df, grouped: dict[str, list]) -> dict[str, dict[str, int]]:
        """Per-data-type distinct counts, derived from the full variant-level frame.

        Placed before `cs` in the returned dict so it survives truncation: "how many
        peaks / cell types / associations" then has a definitive answer even when the
        per-credible-set list is cut off, instead of the model counting whatever rows
        happened to fit and reporting that as the total.
        """
        import polars as pl

        def _n_distinct(frame, column: str) -> int | None:
            if column not in frame.columns:
                return None
            return frame[column].drop_nulls().n_unique()

        counts: dict[str, dict[str, int]] = {}
        for data_type, cs_list in grouped.items():
            predicate = (
                pl.col("data_type").is_null()
                if data_type == "unknown"
                else pl.col("data_type") == data_type
            )
            sub = df.filter(predicate)

            variant_cols = [c for c in ("chr", "pos", "ref", "alt") if c in sub.columns]
            entry: dict[str, int | None] = {
                "n_credible_sets": len(cs_list),
                # variant-level row count: what an equivalent BigQuery COUNT(*) returns
                "n_associations": sub.height,
                "n_variants": (
                    sub.select(variant_cols).unique().height
                    if len(variant_cols) == 4
                    else None
                ),
                "n_traits": _n_distinct(sub, "trait"),
                "n_cell_types": _n_distinct(sub, "cell_type"),
                "n_datasets": _n_distinct(sub, "dataset"),
            }
            if data_type == "caQTL":
                # the caQTL molecular trait is a chromatin peak. trait_original holds the
                # peak id on every endpoint, while credible_sets_by_qtl_gene remaps trait
                # to the linked gene, so the peak count must come from trait_original
                entry["n_peaks"] = _n_distinct(sub, "trait_original")
            counts[data_type] = {k: v for k, v in entry.items() if v}
        return counts

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

        from ddgs import DDGS

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

    # -------------------------------------------------------------------------
    # myvariant.info
    # -------------------------------------------------------------------------

    _MYVARIANT_DEFAULT_FIELDS = "clinvar,cadd,dbnsfp,cosmic,civic,dbsnp"

    async def get_myvariant_annotations(
        self,
        variant: str | None = None,
        variants: list[str] | None = None,
        fields: str | None = None,
    ) -> dict[str, Any]:
        """Get clinical/functional variant annotations from myvariant.info."""
        base_url = _resolve_settings().myvariant_api_url

        req_fields = fields or self._MYVARIANT_DEFAULT_FIELDS
        params: dict[str, str] = {"fields": req_fields, "assembly": "hg38"}

        try:
            if variant:
                # single variant lookup
                try:
                    hgvs_id = self._variant_to_hgvs(variant)
                except ValueError as e:
                    return {"success": False, "error": str(e)}

                url = f"{base_url}/variant/{quote(hgvs_id, safe='')}"
                resp = await self.external_client.get(url, params=params, timeout=15.0)

                if resp.status_code == 404:
                    return {"success": True, "variant": variant, "found": False, "annotations": {}}
                if resp.status_code == 429:
                    return {"success": False, "error": "myvariant.info rate limit exceeded. Try again shortly."}
                resp.raise_for_status()

                data = resp.json()
                annotations = self._flatten_myvariant_result(data)
                return {"success": True, "variant": variant, "found": bool(annotations), "annotations": annotations}

            elif variants:
                if len(variants) > 1000:
                    return {"success": False, "error": "Maximum 1000 variants per batch query."}

                # convert all variant IDs to HGVS
                hgvs_ids = []
                conversion_errors = []
                for v in variants:
                    try:
                        hgvs_ids.append(self._variant_to_hgvs(v))
                    except ValueError:
                        conversion_errors.append(v)

                if not hgvs_ids:
                    return {"success": False, "error": f"Could not convert any variants to HGVS format. Invalid: {conversion_errors}"}

                # batch query via POST
                url = f"{base_url}/variant"
                post_data = {
                    "ids": ",".join(hgvs_ids),
                    "fields": req_fields,
                    "assembly": "hg38",
                }

                resp = await self.external_client.post(
                    url,
                    data=post_data,
                    headers={"content-type": "application/x-www-form-urlencoded"},
                    timeout=30.0,
                )

                if resp.status_code == 429:
                    return {"success": False, "error": "myvariant.info rate limit exceeded. Try again shortly."}
                resp.raise_for_status()

                results_list = resp.json()
                if not isinstance(results_list, list):
                    results_list = [results_list]

                # map back to original variant IDs
                hgvs_to_original = dict(zip(hgvs_ids, variants[:len(hgvs_ids)]))
                annotations = {}
                for item in results_list:
                    hgvs_key = item.get("_id", item.get("query", ""))
                    original_id = hgvs_to_original.get(hgvs_key, hgvs_key)
                    if item.get("notfound"):
                        annotations[original_id] = {"found": False}
                    else:
                        flat = self._flatten_myvariant_result(item)
                        annotations[original_id] = {"found": bool(flat), **flat}

                result: dict[str, Any] = {
                    "success": True,
                    "total_queried": len(variants),
                    "total_found": sum(1 for a in annotations.values() if a.get("found")),
                    "annotations": annotations,
                }
                if conversion_errors:
                    result["conversion_errors"] = conversion_errors
                return result

            else:
                return {"success": False, "error": "Provide either 'variant' or 'variants' parameter."}

        except httpx.TimeoutException:
            logger.error("myvariant.info request timed out")
            return {"success": False, "error": "myvariant.info request timed out."}
        except Exception as e:
            logger.error(f"myvariant.info error: {e}", exc_info=True)
            return {"success": False, "error": INTERNAL_ERROR_MSG}

    # -------------------------------------------------------------------------
    # UniProt / EBI Proteins
    # -------------------------------------------------------------------------

    # the logic lives in tools/uniprot.py; these exist because llm_service dispatches
    # tools with getattr(self.executor, tool_name). Each returns the client's result
    # unchanged apart from an optional download hint, so the resolution block it
    # carries — which protein was actually resolved — always reaches the agent.

    # a search wider than this is a table, not an answer: the rows go to a TSV link
    # instead of into the context
    _UNIPROT_DOWNLOAD_THRESHOLD = 25

    @staticmethod
    def _uniprot_download_hint(
        result: dict[str, Any], filename: str, min_rows: int = 1
    ) -> dict[str, Any]:
        """Attach a _download_data hint for the row list in a UniProt tool result."""
        if not isinstance(result, dict) or not result.get("success"):
            return result
        rows = result.get("results")
        if (
            isinstance(rows, list)
            and len(rows) >= min_rows
            and all(isinstance(r, dict) for r in rows)
        ):
            result["_download_data"] = {"results": rows, "filename": filename}
        return result

    async def get_protein_annotations(
        self,
        query: str | list[str],
        organism_id: int = 9606,
        include: list[str] | None = None,
        feature_types: list[str] | None = None,
        residue_range: str | None = None,
    ) -> dict[str, Any]:
        """Get UniProt annotations for a gene symbol, an accession, or a batch of them."""
        try:
            result = await self.uniprot.get_protein_annotations(
                query,
                organism_id=organism_id,
                include=include,
                feature_types=feature_types,
                residue_range=residue_range,
            )
        except Exception as e:
            logger.error(
                f"Error in get_protein_annotations({query!r}): {e}\n{traceback.format_exc()}"
            )
            return {"success": False, "error": INTERNAL_ERROR_MSG}
        # a batch of accessions is a table (the 167-gene zymogen case); one protein is not
        if isinstance(query, list):
            return self._uniprot_download_hint(result, "protein_annotations.tsv")
        return result

    async def map_protein_variants(
        self,
        variants: list[str],
        query: str,
        organism_id: int = 9606,
    ) -> dict[str, Any]:
        """Map protein-level variants (e.g. P70A) to genomic coordinates."""
        try:
            return await self.uniprot.map_protein_variants(
                variants, query, organism_id=organism_id
            )
        except Exception as e:
            logger.error(
                f"Error in map_protein_variants({query!r}, {variants!r}): {e}\n"
                f"{traceback.format_exc()}"
            )
            return {"success": False, "error": INTERNAL_ERROR_MSG}

    async def get_variant_protein_effect(
        self,
        variants: list[str] | str,
    ) -> dict[str, Any]:
        """Map genomic coding SNVs (e.g. 12:40340400:G:A) to their UniProt protein effect."""
        try:
            result = await self.uniprot.get_variant_protein_effect(variants)
        except Exception as e:
            logger.error(
                f"Error in get_variant_protein_effect({variants!r}): {e}\n"
                f"{traceback.format_exc()}"
            )
            return {"success": False, "error": INTERNAL_ERROR_MSG}
        return self._uniprot_download_hint(result, "variant_protein_effect.tsv")

    async def search_uniprot(
        self,
        query: str | None = None,
        keyword: str | None = None,
        organism_id: int = 9606,
        reviewed_only: bool = True,
        fields: list[str] | None = None,
        size: int = 25,
        count_only: bool = False,
    ) -> dict[str, Any]:
        """Search UniProtKB by free text or keyword, or count the matches."""
        try:
            result = await self.uniprot.search_uniprot(
                query=query,
                keyword=keyword,
                organism_id=organism_id,
                reviewed_only=reviewed_only,
                fields=fields,
                size=size,
                count_only=count_only,
            )
        except Exception as e:
            logger.error(
                f"Error in search_uniprot({query!r}, keyword={keyword!r}): {e}\n"
                f"{traceback.format_exc()}"
            )
            return {"success": False, "error": INTERNAL_ERROR_MSG}
        return self._uniprot_download_hint(
            result, "uniprot_search.tsv", min_rows=self._UNIPROT_DOWNLOAD_THRESHOLD
        )

    # -------------------------------------------------------------------------
    # Code execution support (genetics-results-suite-4h6)
    # -------------------------------------------------------------------------

    async def list_capabilities(self, module: str | None = None) -> dict[str, Any]:
        """Describe the `genetics` SDK surface one module at a time.

        The point of the tool is that the catalogue costs nothing until it is asked for:
        the model carries one short tool description instead of a signature per data
        product, so adding a dataset — which adds an SDK argument or function — costs no
        per-turn context. Signatures are read out of the live SDK objects rather than a
        checked-in copy, so they cannot drift from what a script can actually call.

        What it renders is per-function signatures and docstrings, and module-level
        docstrings deliberately not: sdk.__doc__ describes the deployment around the SDK,
        naming INTERNAL_API_SECRET, GENETICS_API_URL and BIGQUERY_API_URL and the services
        behind them, none of which is needed to write a call.

        This tool is NOT in mcp_server.py's _mcp_disabled, so an MCP client sees whatever
        it returns, and that IS new disclosure — the SDK is not the MCP tool surface, so
        none of it is a restatement of the tool list. It is judged acceptable on its
        content: signatures and function docstrings describe the SDK's shape, not data,
        session state or any execution.

        Be honest about what stripping module docs does NOT remove. Function docstrings
        are written to describe the SDK, so they disclose SDK internals by CATEGORY, and
        the categories are what to reason about — an enumeration of the individual strings
        has been re-derived twice and been wrong both times, so treat the examples as
        illustrative, not exhaustive:

        - the settings mechanism, including that endpoint URLs come from the environment
          and cannot be set from a script (`_URL_SETTINGS`, `configure`);
        - internal service and component names (`db-api`, the FinnGen LD server, the
          sandbox itself);
        - the execution model behind an argument — e.g. that `limit=` still runs the full
          join and ORDER BY server-side;
        - limit and quota values: the per-execution row cap, the per-query and
          per-execution byte quotas, and the SDK's own row ceilings.

        Rewriting the docstrings is a separate decision (it would also drift the generated
        sandbox stubs). Bare `<view>` names are NOT in this list: they already appear in
        MCP tool descriptions, so they are not new disclosure. The dataset that backs them
        is never named — db-api resolves it — so it cannot leak here either.
        """
        try:
            return _sdk_capabilities(module)
        except ValueError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(
                f"Error in list_capabilities({module!r}): {e}\n{traceback.format_exc()}"
            )
            return {"success": False, "error": INTERNAL_ERROR_MSG}

    # a read over 4 MiB is not something a chat turn can use; a truncated PNG is garbage
    # rather than a short answer, so oversized binaries are refused instead of cut
    _MAX_ARTIFACT_BYTES = 4 * 1024 * 1024
    _MAX_ARTIFACT_TEXT_CHARS = 100_000

    @staticmethod
    def _artifacts_dir() -> str:
        """The single directory `read_artifact` may read, or "" when there is none.

        This is the allow-list, and it is deliberately its own variable: the obvious
        alternative, SUBAGENT_ALLOWED_PATHS, is `/data` in the deployment — the PVC
        holding chat_history.db and llm_config.db — so wiring artifact reads to it would
        hand the model every conversation in the deployment. It is never read here.

        Only the process that owns the scratch directory sets SANDBOX_ARTIFACTS_DIR. In
        chat-backend it is unset, so this method refuses; retrieval there goes over HTTP
        to the sandbox pod, which is where the filesystem read and the validation below
        happen (genetics-results-suite docs/code-execution-security.md, section 6). The
        Neither the HTTP client nor the session-scoped name resolution exists yet:
        genetics-results-suite-4h6.52 owns both. Earlier comments named 4h6.11, which was
        the SDK extraction and closed without doing either — do not read its closed state
        as evidence the proxy is in place.

        Two structural checks, both of which fail closed to "" (= not enabled):

        - the configured directory may not itself be a symlink. `_validate_path` resolves
          both sides, so a symlinked allow-list root makes *every* file under its target
          validate. This is reachable, not merely operator error: /scratch/<id> is chown'd
          to the child uid (code-execution-security.md section 2), so the child can rmdir
          its `artifacts` and relink it at another execution's retained artifacts — the
          cross-session channel section 6.4 exists to prevent.
        - the resolved directory must sit under _ARTIFACTS_DIR_PREFIX.

        Both are advisory: they answer about a PATH, and the answer is stale the moment it
        returns, because the child owns /scratch/<id> and can swap `artifacts` for a
        symlink between this check and the open. `_open_artifacts_dir` is the enforcing
        layer — it checks an open descriptor instead.
        """
        configured = os.environ.get("SANDBOX_ARTIFACTS_DIR", "").strip()
        if not configured:
            return ""
        try:
            if stat.S_ISLNK(os.lstat(configured).st_mode):
                logger.error("SANDBOX_ARTIFACTS_DIR is a symlink; refusing artifact reads")
                return ""
            resolved = os.path.realpath(configured)
        except OSError:
            return ""
        prefix = _ARTIFACTS_DIR_PREFIX.rstrip("/") + "/"
        if not resolved.startswith(prefix):
            logger.error("SANDBOX_ARTIFACTS_DIR is outside %s; refusing artifact reads", prefix)
            return ""
        return resolved

    @staticmethod
    def _open_artifacts_dir() -> int | None:
        """Open the artifacts directory and verify the DESCRIPTOR, not the path.

        `_artifacts_dir` hands back a path string, and every subsequent use of that string
        re-walks the directory chain — so the `artifacts` component, which the child uid
        owns, can be rmdir'd and relinked at another execution's artifacts (or anywhere)
        after the check passed and before the file is opened. `_validate_path` cannot see
        it either: it resolves both sides through the same swapped link, so both land on
        the attacker's target and it agrees.

        So the directory is opened once, with O_NOFOLLOW (the configured name itself may
        not be a symlink) and O_DIRECTORY, and the prefix check is then made against
        /proc/self/fd/<dirfd> — the kernel's own name for the inode this fd holds, not a
        name re-resolved through whatever the directory chain says now. The caller opens
        the artifact relative to this fd, so a later swap changes a name the read no longer
        uses. Fails closed to None; the caller must close the fd.
        """
        configured = os.environ.get("SANDBOX_ARTIFACTS_DIR", "").strip()
        if not configured:
            return None
        try:
            dirfd = os.open(configured, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        except OSError:
            return None
        try:
            actual = os.readlink(f"/proc/self/fd/{dirfd}")
        except OSError:
            # no /proc, or the fd names nothing checkable: there is no way to verify the
            # descriptor, so there is no read
            os.close(dirfd)
            return None
        prefix = _ARTIFACTS_DIR_PREFIX.rstrip("/") + "/"
        # " (deleted)" is how the kernel renders an unlinked directory's fd; the path it
        # prints then describes where the inode used to be, so it proves nothing
        if not actual.startswith(prefix) or actual.endswith(" (deleted)"):
            logger.error("artifacts directory fd is outside %s; refusing artifact reads", prefix)
            os.close(dirfd)
            return None
        return dirfd

    async def read_artifact(self, name: str) -> dict[str, Any]:
        """Read one named file out of the directory named by SANDBOX_ARTIFACTS_DIR.

        Cross-execution scoping is NOT implemented here: there is no session or execution
        parameter, so which execution's artifacts are reachable rests entirely on that env
        var pointing at the right directory. Resolving a name against a session belongs to
        genetics-results-suite-4h6.52 and is NOT implemented anywhere today; until it lands
        this tool reads whatever directory it is pointed at (subject to `_artifacts_dir`'s
        structural checks).

        `name` is a bare file name, never a path and never an execution id: the model
        learns names from the run's artifact manifest, and nothing else is addressable.
        Validation is layered on purpose, and the layers are not equal. The name check
        (separators, traversal) and `_validate_path`'s resolved-path check are advisory:
        both answer about a path, and a script owns the directory, so it can swap either
        the final component or the `artifacts` directory itself between the check and the
        read. The enforcing layer is a pair of descriptors — the directory is opened once
        and verified as an fd (`_open_artifacts_dir`), the artifact is opened *relative to
        that fd* with O_NOFOLLOW, and every decision after that (regular file, link count,
        bytes) is taken from that one fd's fstat. After `_open_artifacts_dir` returns,
        nothing here addresses anything by path again.

        O_NONBLOCK is on the file open because O_RDONLY on a FIFO with no writer blocks in
        the kernel, before S_ISREG is ever reached — a script that does
        `os.mkfifo(artifacts/results.tsv)` would otherwise hang the calling coroutine (and
        so the chat backend) forever. It is inert for regular files, which are all that
        survives the S_ISREG check.

        Known and accepted: a refusal caused by `_validate_path` returns measurably faster
        than one caused by the open, so a caller can tell that a name IT planted resolves
        out of tree. A dangling symlink takes the same fast path, so this is not an
        existence oracle for anything the caller did not create.
        """
        from genetics_mcp_server.skills.sandbox_tools import _validate_path

        artifacts_dir = self._artifacts_dir()
        if not artifacts_dir:
            return {
                "success": False,
                "error": "Code execution is not enabled here, so there are no artifacts to read.",
            }

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

        not_found = {"success": False, "error": f"Artifact not found: {name}"}
        path = os.path.join(artifacts_dir, name)
        try:
            # belt and braces: catches a resolved path outside the allow-list before any
            # open, but its answer is advisory — the fd below is the enforcing layer.
            # OSError from resolve() is folded in here so it cannot escape carrying the
            # absolute path in its message
            _validate_path(path, [artifacts_dir])
        except (ValueError, OSError):
            # the same answer as a missing file: which names exist outside the allow-list
            # is not something a caller gets to learn by probing
            return not_found

        dirfd = self._open_artifacts_dir()
        if dirfd is None:
            # the directory passed _artifacts_dir a moment ago and does not verify now:
            # that is either a swap in progress or a teardown, and neither gets an answer
            return not_found

        try:
            try:
                fd = os.open(
                    name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dirfd
                )
            except OSError:
                # includes ELOOP: O_NOFOLLOW refuses a symlink at the final component,
                # which is the swap a script can perform after _validate_path resolved the
                # name. Resolution starts at dirfd, so the directory cannot be swapped out
                # from under it either
                return not_found

            try:
                st = os.fstat(fd)
                if not stat.S_ISREG(st.st_mode):
                    # FIFOs and devices land here rather than in a blocked open, thanks to
                    # O_NONBLOCK above
                    return not_found
                if st.st_nlink != 1:
                    # a hardlink has nothing to resolve, so both path layers see an in-tree
                    # path over an out-of-tree inode. Refusing st_nlink != 1 states the
                    # property here instead of inheriting it from fs.protected_hardlinks
                    return not_found
                if st.st_size > self._MAX_ARTIFACT_BYTES:
                    return {
                        "success": False,
                        # no byte count: an exact size would answer questions about files
                        # the caller cannot read
                        "error": (
                            f"Artifact '{name}' is over the {self._MAX_ARTIFACT_BYTES} byte "
                            f"read limit. Write a smaller summary from the script instead."
                        ),
                    }
                chunks: list[bytes] = []
                remaining = self._MAX_ARTIFACT_BYTES
                while remaining > 0:
                    chunk = os.read(fd, min(remaining, 1 << 20))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
                raw = b"".join(chunks)
            except OSError as e:
                logger.error(f"Error reading artifact {name!r}: {e}")
                return not_found
            finally:
                os.close(fd)
        finally:
            os.close(dirfd)

        # the size at open can disagree with what was read if the file grew mid-read, so
        # report the payload rather than the stat
        size = len(raw)

        content_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return {
                "success": True,
                "name": name,
                "size": size,
                "content_type": content_type,
                "encoding": "base64",
                "content": base64.b64encode(raw).decode("ascii"),
            }

        truncated = len(text) > self._MAX_ARTIFACT_TEXT_CHARS
        return {
            "success": True,
            "name": name,
            "size": size,
            "content_type": content_type,
            "encoding": "utf-8",
            "content": text[: self._MAX_ARTIFACT_TEXT_CHARS],
            "truncated": truncated,
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
        from genetics_mcp_server.sandbox_client import SandboxClient

        return SandboxClient()

    async def run_analysis(
        self,
        code: str,
        timeout_s: int | None = None,
        *,
        user: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Run one script in the sandbox and render the supervisor's result for the model.

        `user` and `session_id` are supplied by the CALLER, never by the model: they are the
        subject and the session of the per-execution credential, and llm_service strips any
        same-named key the model emits before injecting the authenticated pair. A tool
        invocation with neither is a wiring fault, not a script fault, and is reported as one.

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
            # sandbox execution passes through — the streaming and non-streaming chat paths,
            # subagent dispatch and any future caller — and because it sits immediately
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

        try:
            result = await asyncio.wait_for(
                self._sandbox.execute(
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

        return self._render_analysis(result, images=await self._fetch_analysis_images(result))

    # How many image artifacts one script may have shown for it. A script that writes fifty
    # PNGs is not asking for fifty pictures in the transcript, and each one is a fetch the
    # user waits through after the analysis has already finished.
    _MAX_ANALYSIS_IMAGES = 4

    async def _fetch_analysis_images(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        """Pull the image artifacts of a completed execution back out of the sandbox.

        Automatic rather than a tool the model calls: an image is for the USER to look at,
        and routing it through the model would cost a roundtrip to fetch something the model
        cannot see anyway. Everything else in `artifacts/` still has to be printed by the
        script — this is not general artifact retrieval (`genetics-results-suite-4h6.52`).

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

        from genetics_mcp_server.sandbox_client import ARTIFACT_READ_MAX_BYTES

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

        rendered: dict[str, Any] = {
            "success": ok,
            "status": status_text,
            "output": result.get("output") if isinstance(result.get("output"), str) else "",
            "output_truncated": bool(result.get("output_truncated")),
            "artifacts": artifacts,
        }
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
            # still has no retrieval path — `genetics-results-suite-4h6.52` owns general
            # artifact reads, and `read_artifact` in this process reads a local directory that
            # is not the sandbox's /scratch. Promising a fetch that returns "not enabled here"
            # costs a roundtrip.
            shown = {
                image["name"]
                for image in images or []
                if isinstance(image, dict) and isinstance(image.get("name"), str)
            }
            unretrievable = [entry["name"] for entry in artifacts if entry["name"] not in shown]
            if shown:
                rendered["artifacts_note"] = (
                    "Image artifacts have been displayed to the user already; describe what "
                    "the plot shows rather than emitting a placeholder or a markdown image."
                )
                if unretrievable:
                    rendered["artifacts_note"] += (
                        " The contents of the other artifacts cannot be retrieved — print "
                        "anything from the script that you need to read."
                    )
            else:
                rendered["artifacts_note"] = (
                    "Artifact contents cannot be retrieved. Print anything from the script "
                    "that you need to read. An image artifact would have been shown to the "
                    "user automatically."
                )

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

        hint = self._analysis_hint(status_text, error_type, limit)
        if hint:
            fields["hint"] = hint
        # a script that ran and failed is repairable by rewriting it, which is the model's
        # job; that is a different thing from the transport failures above, where retrying
        # the SAME script is the correct move.
        fields["retryable"] = False
        return fields

    @staticmethod
    def _analysis_hint(status_text: str, error_type: str | None, limit: Any) -> str | None:
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
        if error_type in ToolExecutor._SDK_MISUSE_ERROR_TYPES:
            return (
                "This looks like the SDK being called differently from how it is defined. "
                "Call list_capabilities for the exact signatures before rewriting."
            )
        return None


# --------------------------------------------------------------------------- SDK catalogue

# the modules a script sees, in the order the index reports them. They are the three the
# sandbox stubs are generated from (sandbox/stubs/*.pyi), so the tool and the shipped
# reference describe the same surface.
_SDK_MODULES = ("genetics", "client", "errors")

# one-line labels written here rather than taken from each module's __doc__. The catalogue
# renders per-function signatures and docstrings only: module docstrings describe the
# deployment around the SDK — the env vars endpoints and credentials come from, the
# services behind it, the per-execution row and byte quotas — none of which a script needs
# to write a call, and list_capabilities is reachable from MCP.
_SDK_MODULE_SUMMARIES = {
    "genetics": "the sync functions a script calls; every one returns a polars DataFrame",
    "client": "the awaitable GeneticsClient form of the same functions",
    "errors": "what a script catches",
}

# `genetics` re-exports these beyond the data functions; the data functions themselves come
# from sdk._FUNCTIONS, which is the SDK's own export list rather than a second copy of it
_SDK_EXTRA_FUNCTIONS = ("configure", "get_client", "close", "parse_region")

# fully-qualified reprs that inspect produces for evaluated annotations, written the way a
# script writes them
_ANNOTATION_ALIASES = {
    "polars.dataframe.frame.DataFrame": "pl.DataFrame",
    "genetics_mcp_server.sdk.client.GeneticsClient": "GeneticsClient",
}


def _render_annotations(text: str) -> str:
    for long_name, short_name in _ANNOTATION_ALIASES.items():
        text = text.replace(long_name, short_name)
    return text


def _render_def(name: str, func: Any, *, is_async: bool) -> str:
    signature = inspect.signature(func)
    params = list(signature.parameters.values())
    # every renderable object here is either a bound-form class method or a sync wrapper
    # whose __wrapped__ is one, so `self` is always present and never part of the surface
    if params and params[0].name == "self":
        signature = signature.replace(parameters=params[1:])
    keyword = "async def" if is_async else "def"
    lines = [_render_annotations(f"{keyword} {name}{signature}:")]
    doc = inspect.getdoc(func)
    if doc:
        body = "\n".join(f"    {line}".rstrip() for line in doc.splitlines())
        lines.append(f'    """{body.lstrip()}\n    """')
    else:
        lines.append("    ...")
    return "\n".join(lines)


def _render_class(name: str, cls: type) -> str:
    bases = ", ".join(base.__name__ for base in cls.__bases__)
    doc = inspect.getdoc(cls)
    if doc:
        body = "\n".join(f"    {line}".rstrip() for line in doc.splitlines())
        return f'class {name}({bases}):\n    """{body.lstrip()}\n    """'
    return f"class {name}({bases}):\n    ..."


def _sdk_members(module: str) -> list[tuple[str, Any]]:
    """(name, object) pairs for one SDK module, in the order they should be rendered."""
    from genetics_mcp_server import sdk
    from genetics_mcp_server.sdk import client as sdk_client
    from genetics_mcp_server.sdk import errors as sdk_errors

    if module == "genetics":
        names = list(sdk._FUNCTIONS) + list(_SDK_EXTRA_FUNCTIONS)
        return [(n, getattr(sdk, n)) for n in names if hasattr(sdk, n)]
    if module == "client":
        names = list(sdk._FUNCTIONS) + ["close"]
        return [
            (n, getattr(sdk_client.GeneticsClient, n))
            for n in names
            if hasattr(sdk_client.GeneticsClient, n)
        ]
    return [
        (n, obj)
        for n, obj in vars(sdk_errors).items()
        if inspect.isclass(obj) and obj.__module__ == sdk_errors.__name__
    ]


# The import line, on EVERY response rather than only the index (genetics-results-suite-706).
# It used to appear only when `module` was omitted — but a model that knows which module it
# wants calls straight through with `module="genetics"`, and that response carried signatures
# and nothing about how to reach them. The one other place the line exists is `sdk.__doc__`,
# which this catalogue deliberately strips (see `_SDK_MODULE_SUMMARIES`), so there was no
# reachable statement of it at all and sessions opened by probing `pkgutil.iter_modules()`.
#
# It names BOTH forms on purpose. `genetics` is the sandbox alias — the name every tool
# description, stub file and schema doc already uses, and the one a `run_analysis` script
# should write. `genetics_mcp_server.sdk` is the package, and is what an MCP client reading
# this catalogue outside the sandbox would have to import, since the alias ships only in the
# sandbox image.
_SDK_USAGE = (
    "import genetics  (in a run_analysis script; the package itself is "
    "genetics_mcp_server.sdk)"
)


def _sdk_capabilities(module: str | None = None) -> dict[str, Any]:
    from genetics_mcp_server.sdk import client as sdk_client

    if module is not None and module not in _SDK_MODULES:
        raise ValueError(
            f"unknown SDK module '{module}'; expected one of: {', '.join(_SDK_MODULES)}"
        )

    if module is None:
        return {
            "success": True,
            "usage": _SDK_USAGE,
            "modules": [
                {
                    "module": name,
                    "summary": _SDK_MODULE_SUMMARIES[name],
                    "names": [n for n, _ in _sdk_members(name)],
                }
                for name in _SDK_MODULES
            ],
            "next": "call list_capabilities(module=...) for signatures and docstrings",
        }

    blocks = []
    if module == "genetics" and hasattr(sdk_client, "MAX_ROWS"):
        blocks.append(f"MAX_ROWS: int = {sdk_client.MAX_ROWS}")
    for name, obj in _sdk_members(module):
        if inspect.isclass(obj):
            blocks.append(_render_class(name, obj))
        else:
            blocks.append(_render_def(name, obj, is_async=module == "client"))
    return {
        "success": True,
        "module": module,
        "usage": _SDK_USAGE,
        "signatures": "\n\n".join(blocks),
    }
