"""Authentication via X-Goog-Authenticated-User-Email header (set by IAP or oauth2-proxy).

The header alone is not a credential — anything that can reach chat-backend on the pod network
can set it to any string. It is honoured only when the request also carries the internal shared
secret, which marks the caller as one of the in-cluster proxies, and the identity it asserts is
then held to the same allow-list oauth2-proxy applies at the edge.

Two secrets live here and they answer different questions. INTERNAL_API_SECRET answers "is the
caller in-cluster" — auth-gateway, results-api and mcp-server all hold it, in either transport.
GATEWAY_IDENTITY_SECRET answers "did auth-gateway relay this after verifying a session", which
is strictly stronger because only auth-gateway and chat-backend hold it; sandbox dispatch keys
on that one (genetics-results-suite-4h6.84).
"""

import hmac
import logging

from fastapi import Request

logger = logging.getLogger(__name__)

IDENTITY_HEADER = "X-Goog-Authenticated-User-Email"
INTERNAL_MARKER_HEADER = "X-Internal-Auth"

# auth-gateway's provenance marker. It carries GATEWAY_IDENTITY_SECRET — a DIFFERENT secret
# from INTERNAL_API_SECRET, held by auth-gateway and chat-backend and by nothing else
# (k8s/deployments/auth-gateway.yaml, k8s/deployments/chat-backend.yaml). The header NAME is
# not the security property and never was: mcp-server and results-api hold INTERNAL_API_SECRET
# by design and are admitted to chat-backend:8000 by the NetworkPolicy, so either could put it
# under any header it likes. What they cannot do is produce a secret they do not have.
GATEWAY_MARKER_HEADER = "X-Gateway-Auth"

# what a caller resolves to when it presents the marker and asserts NO identity
# (`auth_required` case 3). It names a service, not a person, and every holder of
# INTERNAL_API_SECRET resolves to this one string — mcp-server included. Anything that must
# act for a real user has to reject it; `ToolExecutor.run_analysis` does, because the
# NetworkPolicy admits mcp-server to chat-backend and chat-backend is the pod admitted to
# the sandbox (genetics-results-suite-4h6.27, genetics-results-suite-th2).
SERVICE_IDENTITY = "mcp-tool"


def _expected_marker() -> bytes | None:
    """The configured INTERNAL_API_SECRET as comparison bytes, or None when unset."""
    from genetics_mcp_server.config import get_settings

    secret = get_settings().internal_api_secret
    if not secret:
        return None
    # compare as bytes: compare_digest on str raises TypeError for non-ASCII, and a 500 from a
    # forged header would be a worse failure mode than a 401.
    #
    # The two codecs differ on purpose — do not "fix" the latin-1 ones to utf-8. Starlette
    # decodes raw header bytes as latin-1, so re-encoding a header value with latin-1 undoes
    # that decode exactly; utf-8 would re-encode the mojibake instead (b"s\xc3\xa9cret" comes
    # back out as b"s\xc3\x83\xc2\xa9cret").
    #
    # It does NOT recover "the bytes the client sent" in general: measured off a real socket
    # the clients disagree with each other — node fetch/undici and python-requests put latin-1
    # on the wire, aiohttp puts utf-8, and httpx 0.28 (this process's own client) refuses to
    # send a non-ASCII header value at all. No codec is right for all of them, so under a
    # hypothetical non-ASCII secret this pairing would favour the aiohttp-shaped caller and 401
    # the others. What makes the comparison well defined is the ASCII invariant enforced by
    # config.require_internal_api_secret at startup; every codec coincides on ASCII.
    #
    # No try/except on the re-encodes, unlike results-api's is_internal_caller: this takes a
    # starlette Request, so the only strs it can see came from starlette's own latin-1 decode
    # and re-encode by construction. Add the guard if a str-taking entry point is introduced.
    return secret.encode("utf-8")


def _expected_gateway_marker() -> bytes | None:
    """The configured GATEWAY_IDENTITY_SECRET as comparison bytes, or None when unusable.

    None is the fail-closed answer and the only one available when the deployment has not
    provisioned the secret: `is_gateway_caller` then answers False for every request, so an
    unset gateway secret refuses sandbox dispatch rather than admitting it. Nothing in the
    configuration space turns that around.

    Non-ASCII is treated as unusable for the same reason `require_internal_api_secret`
    refuses it at startup (genetics-results-suite-ctq): HTTP clients disagree on how to put a
    non-ASCII header value on the wire, so no server-side codec recovers the same secret from
    every caller and the comparison below would be well defined for none of them. Here it
    degrades to "no gateway caller exists" instead of a startup refusal — see the note on
    `warn_unless_gateway_identity_secret`. The render-config initContainer in
    k8s/deployments/auth-gateway.yaml rejects such a value outright, so a cluster cannot reach
    this branch with a gateway that is also running.
    """
    from genetics_mcp_server.config import get_settings

    secret = get_settings().gateway_identity_secret
    # must agree with warn_unless_gateway_identity_secret's `secret.strip()` check: a value
    # the startup warning calls "unset or empty" cannot be a live, trivially guessable secret
    # here (whitespace-only was measured to dispatch against a whitespace-only header).
    if not secret.strip():
        return None
    if not secret.isascii():
        logger.error(
            "GATEWAY_IDENTITY_SECRET is non-ASCII, so no header encoding recovers it "
            "reliably; treating it as unset, which refuses sandbox dispatch"
        )
        return None
    # see the codec note in _expected_marker: utf-8 here, latin-1 on the header, and the two
    # coincide because of the ASCII check above
    return secret.encode("utf-8")


def is_gateway_caller(request: Request) -> bool:
    """True when the request carries GATEWAY_IDENTITY_SECRET, which only auth-gateway holds.

    STRICTLY NARROWER than `is_internal_caller`, and it is narrower by a SECRET rather than
    by a convention. auth-gateway sets this header on the two locations that proxy to
    chat-backend (`location /chat/v1/` and `location = /status`), after an
    `auth_request /oauth2/auth`, from a key that is mounted into auth-gateway and
    chat-backend and into no other Deployment. So a true answer means the identity header on
    this request was written by the proxy that had just verified an oauth2-proxy session for
    that address — not merely by *some* holder of INTERNAL_API_SECRET.

    That distinction is what `ToolExecutor.run_analysis` needs and what `is_internal_caller`
    cannot give it (genetics-results-suite-4h6.84): every marker holder can assert any
    allow-listed identity, so without this the sandbox's `sub`, artifact scope and audit
    trail rest on mcp-server *choosing* not to assert one.

    The FIRST attempt at this gated on the header NAME instead — the marker arriving in
    `X-Internal-Auth` rather than in `Authorization: Bearer`. Measured end to end, that
    refused a bearer caller and admitted the same caller after it copied its own secret into
    `X-Internal-Auth`, which mcp-server and results-api can both do unilaterally. A header
    name is not a secret; it replaced "mcp-server chooses not to assert an identity" with
    "mcp-server chooses not to rename a header". Do not reintroduce a name-based check here.

    nginx `proxy_set_header` redefines rather than appends, so a client-supplied
    `X-Gateway-Auth` cannot survive the gateway hop; a caller reaching chat-backend directly
    can set the header freely and still fails the comparison.
    """
    expected = _expected_gateway_marker()
    if expected is None:
        return False
    marker = request.headers.get(GATEWAY_MARKER_HEADER)
    return bool(marker) and hmac.compare_digest(marker.encode("latin-1"), expected)


def is_internal_caller(request: Request) -> bool:
    """True when the request carries the shared internal secret in either accepted transport.

    This is the trusted-proxy marker: only in-cluster services holding INTERNAL_API_SECRET
    (auth-gateway's chat locations, results-api, mcp-server) can produce it, so a pod that
    merely has network reach to chat-backend:8000 cannot.

    Two transports, deliberately:
      * `X-Internal-Auth` — auth-gateway's, on the browser-facing chat locations. A dedicated
        header keeps the marker off the caller's `Authorization`, so a chat-backend that has
        not yet rolled to this version simply ignores it and behaves exactly as before rather
        than collapsing every browser session onto the `mcp-tool` service identity during the
        gateway-leads-backend window.
      * `Authorization: Bearer` — results-api's and mcp-server's, unchanged. Those are
        service-to-service callers with no `Authorization` of their own to displace.

    The two are equivalent HERE — this answers "is the caller in-cluster", and both are.
    Deliberately so: the transport carries no authority, because any holder of the secret can
    choose either one. `is_gateway_caller` above does not tell these two apart; it asks for a
    second, different secret that neither results-api nor mcp-server holds, and that is what
    sandbox dispatch keys on.
    """
    expected = _expected_marker()
    if expected is None:
        return False

    marker = request.headers.get(INTERNAL_MARKER_HEADER)
    if marker and hmac.compare_digest(marker.encode("latin-1"), expected):
        return True

    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return False
    return hmac.compare_digest(auth_header[7:].encode("latin-1"), expected)


def _email_allowed(email: str) -> bool:
    """True when the address is covered by ALLOWED_EMAILS or ALLOWED_EMAIL_DOMAINS.

    Compared case-insensitively on both sides: oauth2-proxy lower-cases the address before its
    own domain check, so `User@FinnGen.fi` gets a session there and must not be rejected here.
    A literal `*` in ALLOWED_EMAIL_DOMAINS means "any domain", matching what oauth2-proxy does
    with the same value — without this it would match no domain at all and lock out every user
    of a deployment whose operator set `oauth_email_domain = "*"` deliberately.

    Fails OPEN when the deployment configured no allow-list at all. `allowed_email_domains`
    defaults to `finngen.fi`, so an unconfigured chat-backend would otherwise silently refuse
    every user of any other deployment — a total lockout, worse than the bug this file closes.
    The marker check above is the security-critical half and still applies; the allow-list is
    defence in depth against a compromised holder of INTERNAL_API_SECRET asserting an identity
    oauth2-proxy would never have issued.
    """
    from genetics_mcp_server.config import get_settings

    settings = get_settings()
    if not settings.allow_list_configured:
        logger.warning(
            "no ALLOWED_EMAILS/ALLOWED_EMAIL_DOMAINS configured: accepting any identity the "
            "trusted proxy asserts"
        )
        return True
    domains = {d.strip().lower() for d in settings.allowed_email_domains}
    if "*" in domains:
        return True
    email = email.strip().lower()
    domain = email.split("@")[-1] if "@" in email else ""
    return email in {e.strip().lower() for e in settings.allowed_emails} or domain in domains


def get_authenticated_user(request: Request) -> str | None:
    """Resolve the user email a trusted proxy asserted via the IAP/oauth2-proxy header.

    Returns None — never raises — when the header is absent, unmarked or not allow-listed, so
    every caller fails closed to unauthenticated.

    REQUIRE_AUTH=false means "no authentication at all" (local dev, no oauth2-proxy and no
    shared secret to compare against). The header is honoured as-is there, exactly as before:
    that mode already returned "anonymous" for anyone, so requiring a marker would only break
    the ability to develop as a named user without adding any protection.
    """
    from genetics_mcp_server.config import get_settings

    iap_email = request.headers.get(IDENTITY_HEADER)
    if not iap_email:
        return None
    # header format: "accounts.google.com:user@domain.com"
    email = iap_email.split(":")[-1] if ":" in iap_email else iap_email

    if not get_settings().require_auth:
        return email.strip().lower()

    if not is_internal_caller(request):
        logger.warning(
            "ignoring %s: caller did not present the internal secret", IDENTITY_HEADER
        )
        return None
    if not _email_allowed(email):
        logger.warning("proxied identity rejected: email not in the allow-list")
        return None
    # matching is case-insensitive and whitespace-tolerant, so the same person can arrive as
    # several spellings; return the normalized form or else chat sessions, downloads, API
    # tokens and the ADMIN_USERS check split one person across several identity strings
    return email.strip().lower()
