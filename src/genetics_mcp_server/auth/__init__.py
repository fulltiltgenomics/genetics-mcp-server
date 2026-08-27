"""Authentication module for genetics MCP server."""

from genetics_mcp_server.auth.core import (
    GATEWAY_MARKER_HEADER,
    IDENTITY_HEADER,
    INTERNAL_MARKER_HEADER,
    SERVICE_IDENTITY,
    get_authenticated_user,
    is_gateway_caller,
    is_internal_caller,
)
from genetics_mcp_server.auth.dependencies import (
    admin_required,
    auth_required,
    gateway_asserted_identity,
    is_public,
)

__all__ = [
    "GATEWAY_MARKER_HEADER",
    "IDENTITY_HEADER",
    "INTERNAL_MARKER_HEADER",
    "SERVICE_IDENTITY",
    "get_authenticated_user",
    "is_gateway_caller",
    "is_internal_caller",
    "admin_required",
    "auth_required",
    "gateway_asserted_identity",
    "is_public",
]
