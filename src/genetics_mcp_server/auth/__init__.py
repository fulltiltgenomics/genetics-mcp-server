"""Authentication module for genetics MCP server."""

from genetics_mcp_server.auth.core import (
    IDENTITY_HEADER,
    INTERNAL_MARKER_HEADER,
    get_authenticated_user,
    is_internal_caller,
)
from genetics_mcp_server.auth.dependencies import admin_required, auth_required, is_public

__all__ = [
    "IDENTITY_HEADER",
    "INTERNAL_MARKER_HEADER",
    "get_authenticated_user",
    "is_internal_caller",
    "admin_required",
    "auth_required",
    "is_public",
]
