"""
FastAPI chat router for LLM-powered conversations about genetics data.

Provides endpoints for:
- Streaming chat with OpenAI/Anthropic
- Tool calling via MCP tools

Run with: uvicorn genetics_mcp_server.chat_api:app --port 8000
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

load_dotenv()

from genetics_mcp_server.logging_config import setup_logging

setup_logging(os.environ.get("LOG_LEVEL", "INFO"))

from genetics_mcp_server.auth import auth_required, get_authenticated_user, is_public
from genetics_mcp_server.config import get_settings
from genetics_mcp_server.config.defaults import DEFAULT_SYSTEM_PROMPT
from genetics_mcp_server.download_store import EXPIRED_MESSAGE, get_download_store
from genetics_mcp_server.llm_service import get_llm_service
from genetics_mcp_server.rate_limit import check_rate_limit
from genetics_mcp_server.rate_limit import configure as configure_rate_limit
from genetics_mcp_server.routers import api_tokens_router, chat_history_router, llm_config_router
from genetics_mcp_server.tools import TOOL_DEFINITIONS

logger = logging.getLogger(__name__)


def _classify_error(e: Exception) -> str:
    """Map exceptions to safe, user-facing error messages."""
    name = type(e).__name__
    if name == "RateLimitError":
        return "Rate limit exceeded. Please wait a moment and try again."
    if name in ("AuthenticationError", "PermissionDeniedError"):
        return "LLM service authentication error. Check server configuration."
    if name in ("APITimeoutError", "TimeoutError") or isinstance(e, asyncio.TimeoutError):
        return "Request timed out. Please try again."
    if name == "APIConnectionError":
        return "Could not connect to the LLM service. Please try again later."
    if name in ("BadRequestError", "UnprocessableEntityError"):
        return "Invalid request sent to LLM service."
    if name == "InternalServerError":
        return "External LLM service returned an internal error. Please try again."
    if name == "APIStatusError":
        status = getattr(e, "status_code", None)
        if status and status >= 500:
            return "External LLM service returned an internal error. Please try again."
        return "External LLM service error. Please try again."
    return "An internal server error occurred. Please try again."


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    logger.info("Starting chat API server")
    configure_rate_limit(
        max_per_hour=int(os.environ.get("RATE_LIMIT_PER_HOUR", "20")),
        max_per_day=int(os.environ.get("RATE_LIMIT_PER_DAY", "100")),
    )
    # eagerly initialize LLM service and external MCP servers at startup
    get_llm_service()
    # initialize download store
    get_download_store()

    # periodic cleanup of expired downloads
    async def _download_cleanup_loop():
        while True:
            await asyncio.sleep(3600)
            try:
                get_download_store().cleanup_expired()
            except Exception as e:
                logger.error(f"Download cleanup error: {e}")

    cleanup_task = asyncio.create_task(_download_cleanup_loop())
    yield
    # cleanup
    cleanup_task.cancel()
    service = get_llm_service()
    await service.close()
    logger.info("Chat API server stopped")


app = FastAPI(
    title="Genetics Chat API",
    description="LLM-powered chat API for genetics data with MCP tools",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # configure as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# include routers
app.include_router(api_tokens_router, prefix="/chat/v1", tags=["api-tokens"])
app.include_router(chat_history_router, prefix="/chat/v1", tags=["chat-history"])
app.include_router(llm_config_router, prefix="/chat/v1", tags=["llm-config"])


class ChatMessage(BaseModel):
    """A single message in the chat history."""

    role: str = Field(..., description="Message role: 'user', 'assistant', or 'system'")
    content: str | list[dict[str, Any]] = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""

    messages: list[ChatMessage] = Field(..., description="Chat message history")
    provider: str | None = Field(
        None, description="LLM provider: 'openai' or 'anthropic'"
    )
    model: str | None = Field(None, description="Specific model to use")
    enable_tools: bool = Field(
        True, description="Enable MCP tools (only for Anthropic)"
    )
    system_prompt: str | None = Field(
        None, description="Custom system prompt (overrides default)"
    )
    literature_backend: str | None = Field(
        None, description="Backend for literature search: 'europepmc' or 'perplexity'"
    )
    tool_profile: str | None = Field(
        None,
        description="Tool profile controlling which tool categories are available. "
        "None = all tools, 'api' = general+API tools, 'bigquery' = general+BigQuery, "
        "'rag' = general+RAG external tools.",
    )
    secret: bool = Field(
        False,
        description="Secret chat mode - messages are not logged or persisted.",
    )


class ChatStatusResponse(BaseModel):
    """Response for chat status endpoint."""

    available_providers: list[str]
    default_provider: str
    default_model: str
    tools_enabled: bool
    available_tools: list[str]
    user: str | None = None


# -----------------------------------------------------------------------------
# Authentication endpoints
# -----------------------------------------------------------------------------


@app.get("/chat/v1/auth", include_in_schema=False)
@is_public
async def auth(request: Request):
    """Return current authentication status."""
    user = get_authenticated_user(request)
    return JSONResponse({
        "authenticated": user is not None,
        "user": user,
    })


@app.get("/chat/v1/me")
async def get_current_user_info(request: Request, user: str | None = Depends(auth_required)):
    """Get information about the current authenticated user."""
    return {"user": user}


# -----------------------------------------------------------------------------
# Chat API endpoints
# -----------------------------------------------------------------------------


@app.get("/status", response_model=ChatStatusResponse)
async def chat_status(
    request: Request, user: str | None = Depends(auth_required)
) -> ChatStatusResponse:
    """Get information about available LLM providers and tools."""
    settings = get_settings()
    service = get_llm_service()

    providers = []
    if service.anthropic_client:
        providers.append("anthropic")
    if service.openai_client:
        providers.append("openai")

    return ChatStatusResponse(
        available_providers=providers,
        default_provider=settings.default_provider,
        default_model=settings.default_model,
        tools_enabled=settings.mcp_enabled,
        available_tools=[t["name"] for t in TOOL_DEFINITIONS],
        user=user,
    )


@app.get("/chat/v1/tools")
async def list_tools(user: str | None = Depends(auth_required)) -> list[dict[str, Any]]:
    """List available MCP tools with their descriptions and parameters."""
    return TOOL_DEFINITIONS


@app.post("/chat/v1/chat")
async def stream_chat(
    request: ChatRequest,
    user: str | None = Depends(auth_required),
):
    """
    Stream chat responses as Server-Sent Events (SSE).

    The response is a stream of JSON objects:
    - {"type": "content", "content": "text chunk"}
    - {"type": "done", "message_content": [...]}
    - {"type": "error", "error": "message"}
    """
    settings = get_settings()
    service = get_llm_service()
    provider = request.provider or settings.default_provider

    # per-user rate limiting
    allowed, limit_reason = check_rate_limit(user)
    if not allowed:
        logger.warning(f"Rate limit exceeded for user={user}: {limit_reason}")
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded ({limit_reason}). Please try again later.",
        )

    # validate provider
    if provider == "anthropic" and not service.anthropic_client:
        raise HTTPException(
            status_code=400,
            detail="Anthropic provider not available. Check ANTHROPIC_API_KEY.",
        )
    if provider == "openai" and not service.openai_client:
        raise HTTPException(
            status_code=400,
            detail="OpenAI provider not available. Check OPENAI_API_KEY.",
        )

    # convert messages to dicts
    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

    # use custom or default system prompt
    system_prompt = request.system_prompt or DEFAULT_SYSTEM_PROMPT

    async def event_generator():
        """Generate SSE events from LLM stream."""
        try:
            async for chunk in service.stream_chat(
                messages=messages,
                provider=provider,
                model=request.model,
                system_prompt=system_prompt,
                enable_tools=request.enable_tools,
                literature_backend=request.literature_backend,
                tool_profile=request.tool_profile,
                secret=request.secret,
                user=user,
            ):
                if chunk.type == "text":
                    yield {
                        "event": "message",
                        "data": json.dumps(
                            {"type": "content", "content": chunk.content}
                        ),
                    }
                elif chunk.type == "image":
                    yield {
                        "event": "message",
                        "data": json.dumps({
                            "type": "image",
                            "image_data": chunk.content,
                            "image_format": chunk.image_format or "png",
                            "image_alt": chunk.image_alt or "Generated image",
                        }),
                    }
                elif chunk.type == "done":
                    yield {
                        "event": "message",
                        "data": json.dumps(
                            {"type": "done", "message_content": chunk.message_content}
                        ),
                    }

        except Exception as e:
            logger.error(f"Error in chat stream: {e}", exc_info=True)
            error_msg = _classify_error(e)
            yield {
                "event": "error",
                "data": json.dumps({"type": "error", "error": error_msg}),
            }

    return EventSourceResponse(
        event_generator(),
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/healthz")
@is_public
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


# -----------------------------------------------------------------------------
# Download endpoint
# -----------------------------------------------------------------------------


@app.get("/chat/v1/downloads/{download_id}")
@is_public
async def download_file(download_id: str):
    """Serve a stored download file (TSV)."""
    logger.info(f"Download requested: {download_id}")
    store = get_download_store()
    result = store.get(download_id)
    if result is None:
        logger.error(f"Download failed (404): {download_id}")
        raise HTTPException(status_code=404, detail=EXPIRED_MESSAGE)

    data, filename, content_type = result
    logger.info(f"Serving download {download_id}: {filename} ({len(data)} bytes)")
    from starlette.responses import Response
    return Response(
        content=data,
        media_type=content_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Run the Genetics Chat API server")
    parser.add_argument(
        "--port",
        type=int,
        default=4000,
        help="Port to run the server on (default: 4000)",
    )
    args = parser.parse_args()

    uvicorn.run(
        "genetics_mcp_server.chat_api:app", host="0.0.0.0", port=args.port, reload=True
    )
