"""Sandbox tools for subagent file access and script execution.

These tools are only available to subagents, not exposed via MCP or the main agent.
All operations are restricted to configured allowed paths.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ALLOWED_INTERPRETERS = {"python3", "Rscript", "bash"}

# environment variables to strip from script execution
_SENSITIVE_ENV_PREFIXES = (
    "ANTHROPIC_",
    "OPENAI_",
    "TAVILY_",
    "PERPLEXITY_",
    "MCP_API_KEY",
    "GOOGLE_",
)


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


def _make_safe_env() -> dict[str, str]:
    """Create an environment dict with sensitive variables removed."""
    env = dict(os.environ)
    keys_to_remove = [
        k for k in env if any(k.startswith(prefix) for prefix in _SENSITIVE_ENV_PREFIXES)
    ]
    for k in keys_to_remove:
        del env[k]
    return env


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


async def execute_script(
    interpreter: str,
    script: str,
    working_dir: str,
    allowed_paths: list[str],
    timeout: int = 30,
) -> dict[str, Any]:
    """Execute a script using a whitelisted interpreter.

    The script content is passed via stdin to the interpreter.
    """
    if interpreter not in _ALLOWED_INTERPRETERS:
        return {
            "success": False,
            "error": f"Interpreter '{interpreter}' not allowed. Allowed: {_ALLOWED_INTERPRETERS}",
        }

    try:
        resolved_dir = _validate_path(working_dir, allowed_paths)
        if not resolved_dir.is_dir():
            return {"success": False, "error": f"Working directory not found: {working_dir}"}
    except ValueError as e:
        return {"success": False, "error": str(e)}

    safe_env = _make_safe_env()

    try:
        process = await asyncio.create_subprocess_exec(
            interpreter,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(resolved_dir),
            env=safe_env,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=script.encode()),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return {
                "success": False,
                "error": f"Script execution timed out after {timeout}s",
            }

        stdout_str = stdout.decode(errors="replace")
        stderr_str = stderr.decode(errors="replace")

        # truncate large outputs
        max_output = 50_000
        if len(stdout_str) > max_output:
            stdout_str = stdout_str[:max_output] + "\n[TRUNCATED]"
        if len(stderr_str) > max_output:
            stderr_str = stderr_str[:max_output] + "\n[TRUNCATED]"

        return {
            "success": process.returncode == 0,
            "return_code": process.returncode,
            "stdout": stdout_str,
            "stderr": stderr_str,
        }
    except Exception as e:
        logger.error(f"Error executing script with {interpreter}: {e}")
        return {"success": False, "error": f"Script execution failed: {e}"}


def get_sandbox_tool_definitions(
    allow_file_read: bool,
    allow_script_exec: bool,
) -> list[dict[str, Any]]:
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

    if allow_script_exec:
        tools.append({
            "name": "execute_script",
            "description": (
                "Execute a script using python3, Rscript, or bash. "
                "The script content is passed via stdin. "
                "Available Python libraries: matplotlib, polars, scipy, numpy, pandas. "
                "For plots, save to the working directory as PNG."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "interpreter": {
                        "type": "string",
                        "enum": list(_ALLOWED_INTERPRETERS),
                        "description": "Script interpreter to use",
                    },
                    "script": {
                        "type": "string",
                        "description": "Script content to execute",
                    },
                },
                "required": ["interpreter", "script"],
            },
        })

    return tools
