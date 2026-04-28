"""Subagent service for running parallel specialized agents.

Each subagent gets its own system prompt (skill instructions), tool set,
and agentic loop. Results are collected and returned to the main agent.
"""

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from genetics_mcp_server.config import get_settings
from genetics_mcp_server.mcp_proxy import (
    execute_external_tool,
    get_external_anthropic_tools,
    is_external_tool,
)
from genetics_mcp_server.skills.definitions import (
    SkillDefinition,
    get_skill,
    get_skill_instructions,
)
from genetics_mcp_server.skills.sandbox_tools import (
    execute_script,
    get_sandbox_tool_definitions,
    list_directory,
    read_file,
)
from genetics_mcp_server.tools import ToolExecutor, get_anthropic_tools

logger = logging.getLogger(__name__)

_VALUE_TRUNCATE_LEN = 50


def _format_tool_params(tool_input: dict, max_len: int = 80) -> str:
    """Produce a compact (key='val', ...) string from tool input dict."""
    if not tool_input:
        return ""

    parts = []
    for key, value in tool_input.items():
        if isinstance(value, (list, dict)):
            formatted = f"{key}=<{type(value).__name__}>"
        elif isinstance(value, str) and len(value) > _VALUE_TRUNCATE_LEN:
            formatted = f"{key}='{value[:_VALUE_TRUNCATE_LEN]}...'"
        elif isinstance(value, str):
            formatted = f"{key}='{value}'"
        else:
            formatted = f"{key}={value}"
        parts.append(formatted)

    result = f"({', '.join(parts)})"
    if len(result) > max_len:
        return result[: max_len - 3] + "..."
    return result


@dataclass
class SubagentResult:
    """Result from a single subagent execution."""

    skill_name: str
    query: str
    output: str
    subagent_id: str = ""
    tools_used: list[str] = field(default_factory=list)
    iterations: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    success: bool = True
    error: str | None = None


class SubagentService:
    """Service for running parallel subagents with specialized skills."""

    def __init__(self, anthropic_client: Any, executor: ToolExecutor):
        self._client = anthropic_client
        self._executor = executor

    async def run_subagents(
        self,
        tasks: list[dict[str, str]],
        progress_callback: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Run multiple subagents in parallel and return collected results.

        Args:
            tasks: List of dicts with 'skill', 'query', and optional 'context' keys.
            progress_callback: Optional callback invoked with progress messages
                at subagent start, tool calls, completion, and failure.
        """
        settings = get_settings()

        # validate skills before launching
        validated_tasks = []
        for task in tasks:
            skill_name = task.get("skill", "")
            skill = get_skill(skill_name)
            if skill is None:
                return {
                    "success": False,
                    "error": f"Unknown skill: '{skill_name}'. Available: {list(get_skill.__wrapped__.__code__.co_consts) if hasattr(get_skill, '__wrapped__') else 'check SKILL_REGISTRY'}",
                }
            validated_tasks.append((skill, task.get("query", ""), task.get("context")))

        logger.info(
            f"Launching {len(validated_tasks)} subagents: "
            f"{[s.name for s, _, _ in validated_tasks]}"
        )

        timeout = settings.subagent_timeout

        async def _run_with_timeout(
            skill: SkillDefinition, query: str, context: str | None,
            subagent_id: str,
        ) -> SubagentResult:
            try:
                return await asyncio.wait_for(
                    self._run_subagent(skill, query, context, progress_callback, subagent_id),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                logger.error(f"Subagent '{skill.name}' [{subagent_id}] timed out after {timeout}s")
                if progress_callback:
                    progress_callback(f"Subagent '{skill.name}' [{subagent_id}] timed out after {timeout}s")
                return SubagentResult(
                    skill_name=skill.name,
                    subagent_id=subagent_id,
                    query=query,
                    output="",
                    success=False,
                    error=f"Timed out after {timeout}s",
                )

        results = await asyncio.gather(
            *(
                _run_with_timeout(s, q, c, f"sa-{i + 1}")
                for i, (s, q, c) in enumerate(validated_tasks)
            ),
            return_exceptions=True,
        )

        # process results, handling any unexpected exceptions
        processed = []
        for i, result in enumerate(results):
            skill_name = validated_tasks[i][0].name
            subagent_id = f"sa-{i + 1}"
            if isinstance(result, Exception):
                logger.error(f"Subagent '{skill_name}' [{subagent_id}] failed with exception: {result}")
                processed.append({
                    "skill": skill_name,
                    "subagent_id": subagent_id,
                    "success": False,
                    "error": str(result),
                })
            else:
                processed.append({
                    "skill": result.skill_name,
                    "subagent_id": result.subagent_id,
                    "success": result.success,
                    "output": result.output,
                    "tools_used": result.tools_used,
                    "iterations": result.iterations,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "error": result.error,
                })

        return {"success": True, "results": processed}

    async def _run_subagent(
        self,
        skill: SkillDefinition,
        query: str,
        context: str | None = None,
        progress_callback: Callable[[str], None] | None = None,
        subagent_id: str = "sa-0",
    ) -> SubagentResult:
        """Run a single subagent with its own agentic loop."""
        settings = get_settings()
        model = skill.model or settings.subagent_model or settings.fast_model

        # build system prompt from skill instructions
        instructions = get_skill_instructions(skill)
        if not instructions:
            return SubagentResult(
                skill_name=skill.name,
                subagent_id=subagent_id,
                query=query,
                output="",
                success=False,
                error=f"No instructions found for skill '{skill.name}'",
            )

        # build tool list
        tool_definitions = self._get_tool_definitions(skill)

        # build messages
        user_content = query
        if context:
            user_content = f"Context:\n{context}\n\nTask:\n{query}"

        messages: list[dict[str, Any]] = [{"role": "user", "content": user_content}]

        max_iterations = skill.max_iterations
        max_tokens = skill.max_tokens
        tools_used: list[str] = []
        iteration = 0
        total_input_tokens = 0
        total_output_tokens = 0
        final_text = ""

        logger.info(
            f"Subagent '{skill.name}' [{subagent_id}] starting: model={model}, "
            f"tools={len(tool_definitions)}, max_iter={max_iterations}"
        )
        if progress_callback:
            progress_callback(f"Subagent '{skill.name}' [{subagent_id}] started")

        try:
            while iteration < max_iterations:
                iteration += 1

                request_params: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": settings.temperature,
                    "system": instructions,
                }
                if tool_definitions:
                    request_params["tools"] = tool_definitions

                message = await self._client.messages.create(**request_params)

                # accumulate token usage
                if hasattr(message, "usage") and message.usage:
                    total_input_tokens += message.usage.input_tokens
                    total_output_tokens += message.usage.output_tokens

                # extract text from response
                text_parts = []
                tool_uses = []
                for block in message.content:
                    if block.type == "text":
                        text_parts.append(block.text)
                    elif block.type == "tool_use":
                        tool_uses.append(block)

                final_text = "\n".join(text_parts)

                if not tool_uses:
                    break

                # execute tools
                tool_results = []
                for tool_use in tool_uses:
                    tools_used.append(tool_use.name)
                    params_str = _format_tool_params(tool_use.input)
                    logger.info(
                        f"Subagent '{skill.name}' [{subagent_id}] calling {tool_use.name}{params_str}"
                    )
                    if progress_callback:
                        progress_callback(f"Subagent '{skill.name}' [{subagent_id}] calling {tool_use.name}{params_str}")
                    result = await self._execute_subagent_tool(
                        tool_use.name, dict(tool_use.input), skill
                    )
                    result_json = json.dumps(result)

                    # truncate large results
                    if len(result_json) > settings.mcp_max_result_size:
                        result_json = (
                            result_json[: settings.mcp_max_result_size - 200]
                            + "\n\n[TRUNCATED: Response too large]"
                        )

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": result_json,
                    })

                # continue conversation with tool results
                messages = [
                    *messages,
                    {"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in message.content]},
                    {"role": "user", "content": tool_results},
                ]

            logger.info(
                f"Subagent '{skill.name}' [{subagent_id}] completed: {iteration} iterations, "
                f"{len(tools_used)} tool calls, "
                f"input_tokens={total_input_tokens} output_tokens={total_output_tokens}"
            )
            if progress_callback:
                progress_callback(f"Subagent '{skill.name}' [{subagent_id}] completed ({len(tools_used)} tool calls)")

            return SubagentResult(
                skill_name=skill.name,
                subagent_id=subagent_id,
                query=query,
                output=final_text,
                tools_used=tools_used,
                iterations=iteration,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                success=True,
            )

        except Exception as e:
            logger.error(f"Subagent '{skill.name}' [{subagent_id}] error: {e}")
            if progress_callback:
                progress_callback(f"Subagent '{skill.name}' [{subagent_id}] failed: {e}")
            return SubagentResult(
                skill_name=skill.name,
                subagent_id=subagent_id,
                query=query,
                output=final_text,
                tools_used=tools_used,
                iterations=iteration,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                success=False,
                error=str(e),
            )

    def _get_tool_definitions(self, skill: SkillDefinition) -> list[dict[str, Any]]:
        """Build tool definitions for a skill based on its categories and extras."""
        settings = get_settings()

        # determine tool_profile from categories
        # map skill categories to the closest tool_profile
        if {"api", "general"} <= skill.tool_categories:
            tool_profile = "api"
        elif {"bigquery", "general"} <= skill.tool_categories:
            tool_profile = "bigquery"
        else:
            tool_profile = "rag"  # general-only

        # exclude orchestration tools to prevent recursive subagent launches
        disabled = set(settings.disabled_tools) if settings.disabled_tools else set()
        disabled.add("launch_subagents")

        tools = get_anthropic_tools(
            tool_profile=tool_profile,
            disabled_tools=disabled,
        )

        # filter to only extra_tools if categories are minimal
        if skill.extra_tools:
            extra_names = set(skill.extra_tools)
            existing_names = {t["name"] for t in tools}
            # add any extra tools that aren't already included
            if not extra_names.issubset(existing_names):
                all_tools = get_anthropic_tools(disabled_tools=settings.disabled_tools)
                for tool in all_tools:
                    if tool["name"] in extra_names and tool["name"] not in existing_names:
                        tools.append(tool)

        # add sandbox tools
        sandbox_tools = get_sandbox_tool_definitions(
            allow_file_read=skill.allow_file_read and settings.enable_subagents,
            allow_script_exec=skill.allow_script_exec and settings.enable_script_execution,
        )
        tools.extend(sandbox_tools)

        if skill.include_external:
            tools.extend(get_external_anthropic_tools())

        return tools

    async def _execute_subagent_tool(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        skill: SkillDefinition,
    ) -> dict[str, Any]:
        """Execute a tool call from a subagent."""
        settings = get_settings()

        try:
            # sandbox tools
            if tool_name == "read_file":
                allowed = skill.allowed_paths or settings.subagent_allowed_paths_list
                return await read_file(tool_input["path"], allowed)

            if tool_name == "list_directory":
                allowed = skill.allowed_paths or settings.subagent_allowed_paths_list
                return await list_directory(tool_input["path"], allowed)

            if tool_name == "execute_script":
                allowed = skill.allowed_paths or settings.subagent_allowed_paths_list
                return await execute_script(
                    interpreter=tool_input["interpreter"],
                    script=tool_input["script"],
                    working_dir=allowed[0] if allowed else "/tmp",
                    allowed_paths=allowed,
                    timeout=settings.subagent_script_timeout,
                )

            # external tools
            if is_external_tool(tool_name):
                return await execute_external_tool(tool_name, tool_input)

            # local tools via executor
            method = getattr(self._executor, tool_name, None)
            if method is None:
                return {"success": False, "error": f"Unknown tool: {tool_name}"}

            result = await method(**tool_input)

            # lazy import to avoid circular dependency with llm_service
            from genetics_mcp_server.llm_service import _process_download_hints

            return _process_download_hints(result)

        except Exception as e:
            logger.error(f"Subagent tool '{tool_name}' error: {e}")
            return {"success": False, "error": str(e)}
