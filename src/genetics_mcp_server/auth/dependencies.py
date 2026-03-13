"""FastAPI dependencies for authentication."""

import logging
import os

from fastapi import HTTPException, Request

from genetics_mcp_server.auth.core import get_authenticated_user

logger = logging.getLogger(__name__)

# when True, require X-Goog-Authenticated-User-Email header (set by IAP or oauth2-proxy)
_require_auth = os.environ.get("REQUIRE_AUTH", "").lower() in ("1", "true", "yes")


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

    # allow internal MCP tool calls
    if request.headers.get("X-Internal-MCP-Call") == "true":
        return "mcp-tool"

    user = get_authenticated_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


def is_public(func):
    """Decorator to mark an endpoint as public (no auth required)."""
    setattr(func, "is_public", True)
    return func
