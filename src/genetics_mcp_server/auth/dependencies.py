"""FastAPI dependencies for authentication."""

import logging

from fastapi import Depends, HTTPException, Request

from genetics_mcp_server.auth.core import (
    IDENTITY_HEADER,
    SERVICE_IDENTITY,
    get_authenticated_user,
    is_gateway_caller,
    is_internal_caller,
)

logger = logging.getLogger(__name__)

def auth_is_required() -> bool:
    """Whether to require the X-Goog-Authenticated-User-Email header (set by IAP or oauth2-proxy).

    Read through settings rather than from a module global snapshotted at import time, so that
    this gate and the is_admin reported by /chat/v1/auth cannot disagree: they now consult the
    same cached Settings instance. The old module global is deliberately gone rather than kept
    as an alias — a test still patching it now fails loudly instead of silently moving nothing.

    Importing get_settings per call does NOT make this reload-proof: the import resolves through
    the package, and reloading `config.settings` rebinds only the submodule's handle while the
    package keeps re-exporting the pre-reload function, so this would still read the pre-reload
    cache. Move the setting with `conftest.settings_env`, which clears both handles, rather than
    by reloading. See the note on `clear_settings_cache` in tests/conftest.py.
    """
    from genetics_mcp_server.config import get_settings

    return get_settings().require_auth


def is_public_endpoint(request: Request) -> bool:
    """Check if the endpoint is marked as public."""
    route = request.scope.get("route")
    if route and getattr(route.endpoint, "is_public", False):
        return True
    return False


async def auth_required(request: Request) -> str | None:
    """Resolve the caller's identity from the internal-secret marker and the identity header.

    Precedence, in the order decided here:

    1. **marker + allow-listed identity header** -> that email. The shared secret only says "an
       in-cluster proxy sent this"; when that proxy also asserts *whose* request it is relaying,
       the asserted person is the caller. Checking the bearer first — which is what this did
       before — collapses every browser request to the generic ``mcp-tool`` and loses the real
       user for chat history, downloads, API tokens and the ADMIN_USERS check.
    2. **marker + identity header that is not allow-listed** -> 401. Deliberately *not*
       downgraded to ``mcp-tool``: a downgrade would let anything holding the shared secret
       launder a refused identity into a working, service-attributed request, and the refusal
       would never surface.
    3. **marker alone** (or with a literally empty header) -> ``mcp-tool``. results-api's
       /tokens/validate call and mcp-server's tool calls land here; unchanged.
    4. **identity header alone, no marker** -> 401. This is the hole being closed: the header is
       settable by anything with network reach to port 8000, and admin membership, token minting
       and every per-user route were decided from it.

    The secret used to be an ALTERNATIVE identity rather than a marker. It also used to accept a
    bare `X-Internal-MCP-Call: true` request header, which any client could send — full
    authentication for anyone the auth-gateway forgot to strip it for.
    """
    if not auth_is_required():
        # dev mode: no proxy, no secret. get_authenticated_user honours the header as-is here
        user = get_authenticated_user(request)
        return user or "anonymous"

    if is_public_endpoint(request):
        return None

    if is_internal_caller(request):
        if request.headers.get(IDENTITY_HEADER):
            # cases 1 and 2 — an asserted identity, once present, decides the outcome either way
            user = get_authenticated_user(request)
            if user is None:
                raise HTTPException(status_code=401, detail="Not authenticated")
            return user
        return SERVICE_IDENTITY  # case 3

    user = get_authenticated_user(request)  # case 4, always None without the marker
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


async def gateway_asserted_identity(request: Request) -> bool:
    """Whether THIS request's identity was asserted by auth-gateway, having verified a session.

    Sandbox dispatch keys on this and nothing else (genetics-results-suite-4h6.84). It is a
    separate dependency rather than a change to `auth_required` on purpose: `auth_required`'s
    precedence is unchanged and still resolves cases 1-4 exactly as before, because every
    route other than sandbox dispatch is legitimately reachable by any marker holder. What
    this adds is a second, narrower fact about the same request, consumed only where the
    identity has to be a person the gateway just authenticated rather than one a marker
    holder typed.

    Requires BOTH halves. The gateway secret alone is case 3 (`mcp-tool`), which names no
    person; the identity header alone is case 4, already a 401. Only the pair is a browser
    session.

    `is_gateway_caller` compares GATEWAY_IDENTITY_SECRET, a secret held by auth-gateway and
    chat-backend and by nothing else — NOT the header the marker arrived in. An earlier draft
    keyed on the header name and was measurably bypassable: mcp-server and results-api hold
    INTERNAL_API_SECRET by design and can write it under any header they choose.

    Unset gateway secret + REQUIRE_AUTH=true answers False for every request, so the
    misconfigured deployment refuses code execution instead of admitting it.

    REQUIRE_AUTH=false is "no authentication at all" — local dev with no oauth2-proxy, no
    gateway and no secret to send. `auth_required` already honours the bare header there and
    returns `anonymous` otherwise, so demanding a marker no one can produce would only make
    code execution undevelopable locally without protecting a deployment that has switched
    authentication off in the first place. Production sets REQUIRE_AUTH=true
    (k8s/deployments/chat-backend.yaml), which is where the gate binds.
    """
    if not auth_is_required():
        return True
    return bool(request.headers.get(IDENTITY_HEADER)) and is_gateway_caller(request)


async def admin_required(
    request: Request,
    user: str | None = Depends(auth_required),
) -> str:
    """Dependency that requires admin access.

    Returns 404 when ENABLE_ADMIN_PAGE is false.
    When REQUIRE_AUTH is false (dev mode), any authenticated user is an admin.
    When REQUIRE_AUTH is true, user must be in the ADMIN_USERS list.
    """
    from genetics_mcp_server.config import get_settings
    settings = get_settings()

    if not settings.enable_admin_page:
        raise HTTPException(status_code=404, detail="Not found")

    if not auth_is_required():
        return user or "anonymous"

    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    if user.lower() not in settings.admin_users_list:
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


def is_public(func):
    """Decorator to mark an endpoint as public (no auth required)."""
    setattr(func, "is_public", True)
    return func
