"""MCP tools for genetics data access."""

from typing import TYPE_CHECKING, Any

from genetics_mcp_server.tools.executor import ToolExecutor

if TYPE_CHECKING:
    from genetics_mcp_server.tools.definitions import (
        BIGQUERY_TOOL_DEFINITIONS,
        SUBAGENT_TOOL_DEFINITIONS,
        TOOL_DEFINITIONS,
        TOOL_PROFILE_TOOLS,
        TOOL_PROFILES,
        get_anthropic_tools,
        register_mcp_tools,
    )

__all__ = [
    "ToolExecutor",
    "TOOL_DEFINITIONS",
    "BIGQUERY_TOOL_DEFINITIONS",
    "SUBAGENT_TOOL_DEFINITIONS",
    "TOOL_PROFILES",
    "TOOL_PROFILE_TOOLS",
    "register_mcp_tools",
    "get_anthropic_tools",
]

# `definitions` is resolved on first attribute access, not at import (genetics-results-suite-6bv).
# It does `from pydantic import Field` at module level and `sdk/client.py` reaches this package
# through `tools.executor`, so an eager re-export put pydantic — and the 2600-line map of every
# tool the suite exposes — inside the SDK's import closure, which is what the sandbox image
# ships and what sandbox/requirements.txt has to pin. Same cut as genetics-results-suite-l41
# made for config/settings.py. Every service caller (`from genetics_mcp_server.tools import
# TOOL_DEFINITIONS`, `tools.get_anthropic_tools`) is unaffected: `from X import Y` and attribute
# access both fall through to this hook.
# LATENT: this derives "lazy from definitions" as "everything in __all__ that is not
# ToolExecutor", which holds only because every other entry today does live in `definitions`.
# An __all__ entry added from a DIFFERENT submodule would be routed to
# `getattr(definitions, name)` and raise an AttributeError naming the wrong module. If that
# happens, list the definitions names explicitly instead of subtracting from __all__.
_LAZY_FROM_DEFINITIONS = frozenset(__all__) - {"ToolExecutor"}


def __getattr__(name: str) -> Any:
    if name in _LAZY_FROM_DEFINITIONS:
        from genetics_mcp_server.tools import definitions

        value = getattr(definitions, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
