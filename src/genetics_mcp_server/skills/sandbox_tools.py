"""Sandbox tools for subagent file access.

These tools are only available to subagents, not exposed via MCP or the main agent.
All operations are restricted to configured allowed paths. Code execution is not here:
it belongs to `run_analysis`, which runs in the sandbox pod under a per-execution
credential, and nothing in this process may execute model-authored code.
"""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _validate_path(path: str, allowed_paths: list[str]) -> Path:
    """Validate that a path is under one of the allowed directories.

    Raises ValueError if the path is outside allowed directories.
    """
    if not allowed_paths:
        raise ValueError("No allowed paths configured for this skill")

    resolved = Path(path).resolve()
    for allowed in allowed_paths:
        allowed_resolved = Path(allowed).resolve()
        if resolved == allowed_resolved or allowed_resolved in resolved.parents:
            return resolved

    raise ValueError(
        f"Path '{path}' is outside allowed directories: {allowed_paths}"
    )


async def read_file(path: str, allowed_paths: list[str]) -> dict[str, Any]:
    """Read a file within allowed directories."""
    try:
        resolved = _validate_path(path, allowed_paths)
        if not resolved.exists():
            return {"success": False, "error": f"File not found: {path}"}
        if not resolved.is_file():
            return {"success": False, "error": f"Not a file: {path}"}

        content = resolved.read_text(errors="replace")
        # truncate very large files
        max_size = 100_000
        truncated = len(content) > max_size
        if truncated:
            content = content[:max_size]

        return {
            "success": True,
            "path": str(resolved),
            "content": content,
            "truncated": truncated,
        }
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.error(f"Error reading file {path}: {e}")
        return {"success": False, "error": f"Failed to read file: {e}"}


async def list_directory(path: str, allowed_paths: list[str]) -> dict[str, Any]:
    """List directory contents within allowed directories."""
    try:
        resolved = _validate_path(path, allowed_paths)
        if not resolved.exists():
            return {"success": False, "error": f"Directory not found: {path}"}
        if not resolved.is_dir():
            return {"success": False, "error": f"Not a directory: {path}"}

        entries = []
        for entry in sorted(resolved.iterdir()):
            entries.append({
                "name": entry.name,
                "type": "directory" if entry.is_dir() else "file",
                "size": entry.stat().st_size if entry.is_file() else None,
            })

        return {"success": True, "path": str(resolved), "entries": entries}
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.error(f"Error listing directory {path}: {e}")
        return {"success": False, "error": f"Failed to list directory: {e}"}


def get_sandbox_tool_definitions(allow_file_read: bool) -> list[dict[str, Any]]:
    """Get Anthropic-format tool definitions for sandbox tools."""
    tools = []

    if allow_file_read:
        tools.append({
            "name": "read_file",
            "description": "Read the contents of a file. Only files within allowed directories can be read.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file to read",
                    },
                },
                "required": ["path"],
            },
        })
        tools.append({
            "name": "list_directory",
            "description": "List the contents of a directory. Only directories within allowed paths can be listed.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the directory to list",
                    },
                },
                "required": ["path"],
            },
        })

    return tools
