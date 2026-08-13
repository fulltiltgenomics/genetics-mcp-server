"""Configuration for genetics MCP server."""

from genetics_mcp_server.config.settings import (
    Settings,
    get_settings,
    model_rejects_disabled_thinking,
    model_rejects_temperature,
    model_supports_adaptive_thinking,
    require_internal_api_secret,
)

__all__ = [
    "Settings",
    "get_settings",
    "require_internal_api_secret",
    "model_rejects_disabled_thinking",
    "model_rejects_temperature",
    "model_supports_adaptive_thinking",
]
