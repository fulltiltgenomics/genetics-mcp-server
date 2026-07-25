"""FastAPI dependencies for authentication."""

import hmac
import logging
import os

from fastapi import Depends, HTTPException, Request

from genetics_mcp_server.auth.core import get_authenticated_user

logger = logging.getLogger(__name__)

# when True, require X-Goog-Authenticated-User-Email header (set by IAP or oauth2-proxy)
_require_auth = os.environ.get("REQUIRE_AUTH", "").lower() in ("1", "true", "yes")
_internal_api_secret = os.environ.get("INTERNAL_API_SECRET", "")


def is_public_endpoint(request: Request) -> bool:
    """Check if the endpoint is marked as public."""
    route = request.scope.get("route")
    if route and getattr(route.endpoint, "is_public", False):
        return True
    return False


async def auth_required(request: Request) -> str | None:
    """Dependency that requires authentication via IAP/oauth2-proxy header."""
    if not _require_auth:
        # still check for IAP header in case it's present
        user = get_authenticated_user(request)
        return user or "anonymous"

    if is_public_endpoint(request):
        return None

    # internal service-to-service calls authenticate with the shared secret. This used to
    # accept a bare `X-Internal-MCP-Call: true` request header, which any client could send —
    # full authentication for anyone the auth-gateway forgot to strip it for. Nothing in the
    # stack ever sent that header; the real internal callers all use the bearer secret.
    if _internal_api_secret:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer ") and hmac.compare_digest(
            auth_header[7:], _internal_api_secret
        ):
            return "mcp-tool"

    user = get_authenticated_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


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

    if not _require_auth:
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
