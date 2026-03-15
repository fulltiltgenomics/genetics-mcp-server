"""
LLM service for handling chat interactions with multiple providers.

Supports OpenAI and Anthropic with streaming responses.
Integrates with MCP tools for agentic queries when using Anthropic.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator

from genetics_mcp_server.config import get_settings
from genetics_mcp_server.mcp_proxy import (
    execute_external_tool,
    get_external_anthropic_tools,
    get_rag_anthropic_tools,
    initialize_external_servers,
    is_external_tool,
)
from genetics_mcp_server.tools import ToolExecutor, get_anthropic_tools

logger = logging.getLogger(__name__)


@dataclass
class StreamChunk:
    """A chunk from the LLM stream."""

    type: str  # "text", "done", "image"
    content: str = ""
    # full message content blocks for persistence (only set when type="done")
    message_content: list[dict[str, Any]] | None = None
    # image fields (only set when type="image")
    image_format: str | None = None
    image_alt: str | None = None


class LLMService:
    """Service for LLM chat streaming with multi-provider support."""

    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        self.executor: ToolExecutor | None = None
        self._initialize_clients()

    def _initialize_clients(self):
        """Initialize LLM provider clients based on available API keys."""
        settings = get_settings()

        if settings.openai_api_key:
            try:
                from openai import AsyncOpenAI

                self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
                logger.info("OpenAI client initialized")
            except ImportError:
                logger.warning("OpenAI package not installed")
            except Exception as e:
                logger.error(f"Error initializing OpenAI client: {e}")

        if settings.anthropic_api_key:
            try:
                from anthropic import AsyncAnthropic

                self.anthropic_client = AsyncAnthropic(api_key=settings.anthropic_api_key)
                logger.info("Anthropic client initialized")
            except ImportError:
                logger.warning("Anthropic package not installed")
            except Exception as e:
                logger.error(f"Error initializing Anthropic client: {e}")

        # initialize tool executor
        self.executor = ToolExecutor(
            api_base_url=settings.genetics_api_url,
            bigquery_api_url=settings.bigquery_api_url,
        )

        # initialize external MCP servers
        external_tool_count = initialize_external_servers()
        if external_tool_count > 0:
            logger.info(f"Initialized {external_tool_count} tools from external MCP servers")

    async def stream_chat(
        self,
        messages: list[dict],
        provider: str | None = None,
        model: str | None = None,
        system_prompt: str | None = None,
        enable_tools: bool = True,
        custom_tool_descriptions: dict[str, str] | None = None,
        literature_backend: str | None = None,
        tool_profile: str | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream chat responses from LLM provider.

        Args:
            messages: Chat history in OpenAI format [{"role": "user", "content": "..."}]
            provider: "openai" or "anthropic" (defaults to config setting)
            model: Specific model to use (defaults to provider's default model)
            system_prompt: System prompt to prepend
            enable_tools: Whether to enable MCP tools (Anthropic only)
            custom_tool_descriptions: Custom descriptions for tools
            literature_backend: Backend for literature search ('europepmc' or 'perplexity')
            tool_profile: Tool profile controlling which categories are available.
                None = all tools, "api" = general+api, "bigquery" = general+bigquery,
                "rag" = general+RAG external tools.

        Yields:
            StreamChunk objects with text content and final message structure
        """
        settings = get_settings()
        provider = provider or settings.default_provider

        if provider == "openai":
            async for chunk in self._stream_openai(messages, model, system_prompt):
                yield chunk
        elif provider == "anthropic":
            async for chunk in self._stream_anthropic(
                messages, model, system_prompt, enable_tools, custom_tool_descriptions,
                literature_backend, tool_profile,
            ):
                yield chunk
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    async def _stream_openai(
        self,
        messages: list[dict],
        model: str | None = None,
        system_prompt: str | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream chat from OpenAI."""
        if not self.openai_client:
            raise RuntimeError("OpenAI client not initialized. Check API key.")

        settings = get_settings()
        model = model or "gpt-4o"

        # add system prompt if provided
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages

        try:
            logger.info(f"Streaming OpenAI chat with model {model}")
            stream = await self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                max_tokens=settings.max_tokens,
                temperature=settings.temperature,
            )

            accumulated_text = ""
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    accumulated_text += text
                    yield StreamChunk(type="text", content=text)

            yield StreamChunk(
                type="done", message_content=[{"type": "text", "text": accumulated_text}]
            )

        except Exception as e:
            logger.error(f"Error streaming OpenAI chat: {e}")
            raise

    async def _stream_anthropic(
        self,
        messages: list[dict],
        model: str | None = None,
        system_prompt: str | None = None,
        enable_tools: bool = True,
        custom_tool_descriptions: dict[str, str] | None = None,
        literature_backend: str | None = None,
        tool_profile: str | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream chat from Anthropic with optional MCP tools and agentic loop."""
        if not self.anthropic_client:
            raise RuntimeError("Anthropic client not initialized. Check API key.")

        settings = get_settings()
        model = model or settings.default_model

        # convert messages to Anthropic format
        anthropic_messages = []
        for msg in messages:
            if msg["role"] != "system":
                anthropic_messages.append({"role": msg["role"], "content": msg["content"]})

        # prepare request parameters
        request_params: dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": settings.max_tokens,
            "temperature": settings.temperature,
        }

        if system_prompt:
            request_params["system"] = system_prompt

        # add tool definitions if enabled
        tool_definitions = None
        if enable_tools and settings.mcp_enabled:
            # get local tools filtered by profile
            tool_definitions = get_anthropic_tools(
                custom_tool_descriptions,
                tool_profile=tool_profile,
                disabled_tools=settings.disabled_tools,
            )
            local_count = len(tool_definitions)

            # always-on external tools (gnomAD, Open Targets) are always included
            external_tools = get_external_anthropic_tools()
            tool_definitions.extend(external_tools)

            # RAG tools only included when profile is None (all) or "rag"
            rag_tools = []
            if tool_profile is None or tool_profile == "rag":
                rag_tools = get_rag_anthropic_tools()
                tool_definitions.extend(rag_tools)

            request_params["tools"] = tool_definitions
            logger.info(
                f"Including {len(tool_definitions)} MCP tools "
                f"(profile={tool_profile or 'all'}, {local_count} local, "
                f"{len(external_tools)} external, {len(rag_tools)} RAG)"
            )

        try:
            logger.info(f"Streaming Anthropic chat with model {model}")
            max_iterations = settings.mcp_max_iterations
            iteration = 0

            # collect all content blocks for persistence
            all_content_blocks: list[dict[str, Any]] = []

            while iteration < max_iterations:
                iteration += 1

                # 5 min timeout per iteration to prevent indefinite hangs
                async with asyncio.timeout(300):
                    async with self.anthropic_client.messages.stream(**request_params) as stream:
                        async for text in stream.text_stream:
                            yield StreamChunk(type="text", content=text)

                        message = await stream.get_final_message()

                # add this iteration's content blocks
                for block in message.content:
                    all_content_blocks.append(block.model_dump(exclude_none=True))

                # check for tool use
                tool_uses = [b for b in message.content if b.type == "tool_use"]
                if not tool_uses or not self.executor:
                    break

                # execute tools and collect results
                tool_results = []
                for tool_use in tool_uses:
                    # build effective input for display (with injected params)
                    effective_input = dict(tool_use.input)
                    if tool_use.name == "search_scientific_literature" and literature_backend:
                        effective_input["backend"] = literature_backend

                    logger.info(f"Executing tool: {tool_use.name} with input: {effective_input}")
                    params_str = ", ".join(f"{k}: {v}" for k, v in effective_input.items())
                    yield StreamChunk(
                        type="text", content=f"\n\n*[Using tool: {tool_use.name}; {params_str}]*\n\n"
                    )

                    result = await self._execute_tool(tool_use.name, tool_use.input, literature_backend)

                    # check if result contains an image - stream it and remove from LLM context
                    if isinstance(result, dict) and result.get("success") and result.get("image_base64"):
                        image_data = result["image_base64"]
                        image_format = result.get("image_format", "png")
                        # validate base64 data is present and reasonably sized
                        if image_data and len(image_data) > 100:
                            logger.info(f"Streaming image: format={image_format}, size={len(image_data)} chars")
                            # stream image as a separate event type for reliable handling
                            yield StreamChunk(
                                type="image",
                                content=image_data,
                                image_format=image_format,
                                image_alt=f"{tool_use.name} result",
                            )
                        else:
                            logger.warning(f"Invalid image data: size={len(image_data) if image_data else 0}")
                        # remove image from result to avoid sending huge base64 to LLM
                        result = {k: v for k, v in result.items() if k != "image_base64"}
                        result["note"] = "The image has been displayed to the user above. Do not output any image placeholder or markdown - just describe what the plot shows."

                    result_json = json.dumps(result)

                    # truncate large results
                    if len(result_json) > settings.mcp_max_result_size:
                        total_count = None
                        if (
                            isinstance(result, dict)
                            and "results" in result
                            and isinstance(result["results"], list)
                        ):
                            total_count = len(result["results"])

                        truncated_json = result_json[: settings.mcp_max_result_size - 1000]
                        if total_count:
                            warning = f"\n\n[TRUNCATED: Showing partial data from {total_count} total results. Try adding filters.]"
                        else:
                            warning = "\n\n[TRUNCATED: Response too large. Try more specific filters.]"
                        result_json = truncated_json + warning

                    tool_results.append(
                        {"type": "tool_result", "tool_use_id": tool_use.id, "content": result_json}
                    )

                # continue conversation with tool results
                request_params["messages"] = [
                    *request_params["messages"],
                    {"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in message.content]},
                    {"role": "user", "content": tool_results},
                ]

            if iteration >= max_iterations:
                yield StreamChunk(type="text", content="\n\n*[Max tool iterations reached]*\n")
                all_content_blocks.append(
                    {"type": "text", "text": "\n\n*[Max tool iterations reached]*\n"}
                )

            yield StreamChunk(type="done", message_content=all_content_blocks)

        except asyncio.TimeoutError:
            logger.error("Anthropic streaming timed out after 300s")
            raise
        except Exception as e:
            logger.error(f"Error streaming Anthropic chat: {e}")
            raise

    async def _execute_tool(
        self,
        tool_name: str,
        tool_input: dict,
        literature_backend: str | None = None,
    ) -> dict[str, Any]:
        """Execute a tool by name using the executor or external proxy."""
        try:
            # check if this is an external tool
            if is_external_tool(tool_name):
                logger.debug(f"Executing external tool: {tool_name}")
                return await execute_external_tool(tool_name, tool_input)

            # local tool execution
            if not self.executor:
                return {"success": False, "error": "Tool executor not initialized"}

            method = getattr(self.executor, tool_name, None)
            if method is None:
                return {"success": False, "error": f"Unknown tool: {tool_name}"}

            # inject literature_backend for search_scientific_literature if specified
            if tool_name == "search_scientific_literature" and literature_backend:
                tool_input = {**tool_input, "backend": literature_backend}

            return await method(**tool_input)

        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return {"success": False, "error": str(e)}

    async def close(self):
        """Close resources."""
        if self.executor:
            await self.executor.close()


# singleton instance
_llm_service: LLMService | None = None


def get_llm_service() -> LLMService:
    """Get singleton LLM service instance."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
