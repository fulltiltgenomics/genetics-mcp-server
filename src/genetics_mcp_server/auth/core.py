"""Authentication via X-Goog-Authenticated-User-Email header (set by IAP or oauth2-proxy).

The header alone is not a credential — anything that can reach chat-backend on the pod network
can set it to any string. It is honoured only when the request also carries the internal shared
secret, which marks the caller as one of the in-cluster proxies, and the identity it asserts is
then held to the same allow-list oauth2-proxy applies at the edge.
"""

import hmac
import logging

from fastapi import Request

logger = logging.getLogger(__name__)

IDENTITY_HEADER = "X-Goog-Authenticated-User-Email"
INTERNAL_MARKER_HEADER = "X-Internal-Auth"


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
    """
    from genetics_mcp_server.config import get_settings

    secret = get_settings().internal_api_secret
    if not secret:
        return False
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
    expected = secret.encode("utf-8")

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
    # several spellings; return the normalized form or chat sessions, downloads, API tokens and
    # the ADMIN_USERS check split one person across several identity strings
    return email.strip().lower()
