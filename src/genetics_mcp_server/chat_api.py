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
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

load_dotenv()

from genetics_mcp_server.logging_config import setup_logging

setup_logging(os.environ.get("LOG_LEVEL", "INFO"))

from genetics_mcp_server.auth import (
    auth_required,
    gateway_asserted_identity,
    get_authenticated_user,
    is_public,
)
from genetics_mcp_server.config import (
    get_settings,
    require_internal_api_secret,
    warn_unless_gateway_identity_secret,
)
from genetics_mcp_server.config.defaults import (
    default_system_prompt,
    instruction_envelope,
    verbosity_prompt,
)
from genetics_mcp_server.db import get_llm_config_db
from genetics_mcp_server.db.llm_config_db import INSTRUCTION_SET_MAX_BODY_CHARS
from genetics_mcp_server.download_store import EXPIRED_MESSAGE, get_download_store
from genetics_mcp_server.llm_service import anthropic_error_type, get_llm_service
from genetics_mcp_server.rate_limit import check_rate_limit
from genetics_mcp_server.rate_limit import configure as configure_rate_limit
from genetics_mcp_server.routers import (
    admin_router,
    api_tokens_router,
    chat_history_router,
    llm_config_router,
)
from genetics_mcp_server.tools import TOOL_DEFINITIONS
from genetics_mcp_server.tools.definitions import TOOL_PROFILE_TOOLS, TOOL_PROFILES

logger = logging.getLogger(__name__)


def _classify_error(e: Exception) -> str:
    """Map exceptions to safe, user-facing error messages."""
    name = type(e).__name__
    # Overload/internal errors can arrive mid-stream as a base APIStatusError
    # with status_code=200, so the real type is read from the error body.
    err_type = anthropic_error_type(e)
    if name == "OverloadedError" or err_type == "overloaded_error":
        return (
            "Claude is temporarily overloaded due to high demand. We retried "
            "automatically but it's still unavailable. Please wait a moment and resend."
        )
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
    if name == "InternalServerError" or err_type in ("api_error", "internal_server_error"):
        return "Claude had a temporary upstream error. Please try again."
    if name == "APIStatusError":
        status = getattr(e, "status_code", None)
        if status and status >= 500:
            return "Claude had a temporary upstream error. Please try again."
        return "The LLM service returned an unexpected error. Please try again."
    return "An internal server error occurred. Please try again."


# prefix the frontend uses to inline data-file attachments as text blocks
_FILE_BLOCK_PREFIX = "[File: "


def _message_text_len(content) -> int:
    """Characters of text in one message, ignoring inline image/document data."""
    if isinstance(content, str):
        return len(content)
    if not isinstance(content, list):
        return 0
    total = 0
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            total += len(str(block.get("text", "")))
        elif block.get("type") == "tool_result":
            # replayed tool output, which is where the bulk of a long conversation lives
            inner = block.get("content")
            total += len(inner) if isinstance(inner, str) else _message_text_len(inner)
    return total


def _validate_request_size(messages: list["ChatMessage"]) -> None:
    """Bound the request as a whole. Raises HTTP 413.

    _validate_latest_message only ever inspects the newest user message, so a client-sent
    assistant turn and every replayed history turn were length-unbounded
    (genetics-results-suite-e0u). A forged assistant turn carries no system authority — the
    concern here is request size, not authority — but nothing capped it.

    Applying the per-message cap to every message would have been the tighter rule and the wrong
    one: replayed tool results are routinely larger than any typed message, so it would reject
    ordinary long conversations.
    """
    settings = get_settings()
    if len(messages) > settings.max_messages_per_request:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Too many messages in one request ({len(messages)}, limit "
                f"{settings.max_messages_per_request})."
            ),
        )
    total = sum(_message_text_len(m.content) for m in messages)
    if total > settings.max_request_chars:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Conversation too large ({total} characters, limit "
                f"{settings.max_request_chars}). Start a new chat to continue."
            ),
        )


def _validate_latest_message(messages: list["ChatMessage"]) -> None:
    """Enforce per-message limits on the newest user message.

    Caps typed text length (attachment blocks are excluded — bulk data should be
    uploaded as a file) and the number of attachment blocks. Raises HTTP 413.
    History/assistant turns are not re-validated; only the message being sent now.
    """
    settings = get_settings()
    latest = next((m for m in reversed(messages) if m.role == "user"), None)
    if latest is None:
        return

    content = latest.content
    if isinstance(content, str):
        text_len, attachment_count = len(content), 0
    else:
        # The frontend inlines data-file attachments (TSV/CSV/Excel) as text blocks
        # prefixed with "[File: <name>]" and images as image blocks. Both are
        # attachments and are excluded from the typed-text length, but counted.
        text_blocks = [
            b for b in content if isinstance(b, dict) and b.get("type") == "text"
        ]
        file_text_blocks = [
            b for b in text_blocks if str(b.get("text", "")).startswith(_FILE_BLOCK_PREFIX)
        ]
        text_len = sum(
            len(b.get("text", "")) for b in text_blocks if b not in file_text_blocks
        )
        attachment_count = len(file_text_blocks) + sum(
            1
            for b in content
            if isinstance(b, dict) and b.get("type") in ("image", "document")
        )

    if text_len > settings.max_message_chars:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Message too long ({text_len} characters, limit "
                f"{settings.max_message_chars}). For large data, upload a TSV/CSV file instead."
            ),
        )
    if attachment_count > settings.max_attachments_per_message:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Too many attachments ({attachment_count}, limit "
                f"{settings.max_attachments_per_message} per message)."
            ),
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    logger.info("Starting chat API server")
    # chat-backend runs the tool executor IN-PROCESS, so every results-api and db-api call it
    # makes carries INTERNAL_API_SECRET — or, until genetics-results-suite-618, no credential at
    # all and no signal that it was missing. Gated on require_auth because that is what
    # distinguishes a deployment from a local run against an open results-api: REQUIRE_AUTH is
    # "true" in k8s/deployments/chat-backend.yaml and false everywhere else.
    if get_settings().require_auth:
        require_internal_api_secret("chat-backend")
    # NOT fatal, unlike the line above — see warn_unless_gateway_identity_secret for why the
    # unset case is fail-closed at dispatch rather than at startup.
    warn_unless_gateway_identity_secret("chat-backend")
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
    allow_origins=get_settings().cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# include routers
app.include_router(api_tokens_router, prefix="/chat/v1", tags=["api-tokens"])
app.include_router(chat_history_router, prefix="/chat/v1", tags=["chat-history"])
app.include_router(llm_config_router, prefix="/chat/v1", tags=["llm-config"])

app.include_router(admin_router, prefix="/chat/v1", tags=["admin"])


class ChatMessage(BaseModel):
    """A single message in the chat history."""

    # constrained at the model boundary, not per provider: a client-sent 'system'
    # message used to reach the OpenAI path verbatim and land in a genuine system
    # slot after the server-assembled prompt, where recency favours it. Rejecting
    # here means a future third provider cannot reintroduce the hole. The Anthropic
    # path keeps its own system-role filter as defence in depth (llm_service.py),
    # since stream_chat also takes message dicts from callers other than this model.
    role: Literal["user", "assistant"] = Field(
        ..., description="Message role: 'user' or 'assistant'"
    )
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
    literature_backend: str | None = Field(
        None, description="Backend for literature search: 'europepmc' or 'perplexity'"
    )
    tool_profile: str | None = Field(
        None,
        description="Tool profile controlling which tools are available. "
        "None = all tools, 'api' = general+API tools, 'bigquery' = general+BigQuery, "
        "'rag' = general+RAG external tools, 'code' = the seven-tool code-execution "
        "surface (no external tools). Unrecognised values degrade to general-only.",
    )
    verbosity: str | None = Field(
        None,
        description="Response length: 'brief' (default) reports the analysis as its "
        "conclusions, 'detailed' lays out the full three-pass write-up. Unknown "
        "values fall back to 'brief'.",
    )
    instruction_set_id: str | None = Field(
        None,
        description="Id of one of the caller's stored instruction sets. Only the id "
        "travels: the body is loaded server-side, scoped to the authenticated user. An "
        "id that does not resolve for this user is ignored, not rejected.",
    )
    secret: bool = Field(
        False,
        description="Secret chat mode - messages are not logged or persisted.",
    )
    session_id: str | None = Field(
        None,
        description="Client conversation id. Logged (id only, never content) so distinct "
        "conversations can be counted, including secret ones.",
    )
    capture_thinking: bool = Field(
        False,
        description="Stream each iteration's summarized reasoning as `thinking_summary` "
        "chunks in addition to the contentless `thinking` keepalive. Off for the UI, which "
        "asks for neither; set by the replay benchmark so a transcript can show the "
        "reasoning that produced each tool call. Never affects what is persisted: the text "
        "is not part of `message_content`.",
    )
    message_id: str | None = Field(
        None,
        max_length=64,
        description="Client-generated id the assistant message of this turn will be saved "
        "under. Used only to key the turn's recorded metrics to chat_messages; the metrics "
        "row is written with or without it, and never for a secret chat. Bounded at 64 "
        "characters (uuid4 is 36): unlike session_id, which is only logged, this value is "
        "persisted, and the service has no request-body-size middleware.",
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
    settings = get_settings()
    # settings.require_auth, not a second read of os.environ: this is_admin has to agree with the
    # gate admin_required applies to the endpoints the frontend shows once it is True
    is_admin = False
    if settings.enable_admin_page:
        if not settings.require_auth:
            is_admin = True
        elif user and user.lower() in settings.admin_users_list:
            is_admin = True
    return JSONResponse({
        "authenticated": user is not None,
        "user": user,
        "is_admin": is_admin,
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


@app.get("/chat/v1/tools/resolved")
async def list_resolved_tools(
    tool_profile: str | None = None,
    enable_tools: bool = True,
    user: str | None = Depends(auth_required),
) -> dict[str, Any]:
    """The LOCAL tool names a chat request with these settings would actually be given.

    `/chat/v1/tools` above answers a different question — it returns TOOL_DEFINITIONS raw,
    with no profile filter, no feature flags, and neither the BigQuery nor the subagent
    definition list — so it cannot be used to check what an arm of a benchmark ran with.
    This one resolves through `service.resolve_local_tool_names`, the SAME call the system
    prompt is assembled from (genetics-results-suite-4h6.69), so what it reports is what the
    model was handed.

    IT EXISTS TO MAKE THE SILENT FALLBACK LOUD. `get_anthropic_tools` degrades an
    unrecognised profile to general-only rather than raising, deliberately, because the
    value is read back from `chat_messages` rows written by older clients — so a typo costs
    the model most of its tools and nothing anywhere says so. A benchmark arm misspelled
    that way runs fine and reports plausible numbers. `known_profile: false` is the flag
    that turns that into something a caller can see.

    `count` is LOCAL tools only. External (gnomAD / Open Targets) and RAG tools are proxied
    surfaces resolved separately and are not included; see docs/chat-tool-reference.md § 3
    for the per-profile external/RAG columns.
    """
    service = get_llm_service()
    names = sorted(service.resolve_local_tool_names(tool_profile, enable_tools))
    known = tool_profile is None or tool_profile in TOOL_PROFILES or tool_profile in TOOL_PROFILE_TOOLS
    return {
        "tool_profile": tool_profile,
        "enable_tools": enable_tools,
        "known_profile": known,
        "count": len(names),
        "names": names,
    }


@app.get("/chat/v1/schema")
async def get_schema(
    table: str | None = None,
    user: str | None = Depends(auth_required),
) -> dict[str, Any]:
    """proxy to genetics-results-db /schema so the browser can fetch the BigQuery
    view catalog (resources + tables) without needing a separate URL or env var.
    reuses the executor's httpx client and BIGQUERY_API_URL."""
    service = get_llm_service()
    if not service.executor:
        raise HTTPException(status_code=503, detail="Tool executor not initialized")
    result = await service.executor.get_database_schema(table=table)
    if not result.get("success"):
        # 503 when the db service is simply unreachable (down/restarting), 502 for other upstream errors
        status = 503 if result.get("unreachable") else 502
        raise HTTPException(status_code=status, detail=result.get("error", "schema fetch failed"))
    return result["schema"]


def _resolve_user_instructions(
    user: str | None, set_id: str | None, secret: bool = False
) -> str | None:
    """Envelope fragment for the caller's selected instruction set, or None.

    Every failure path degrades to 'no instructions' rather than raising: an id that
    belongs to another user or does not exist, a set the user archived, an unavailable
    database, an over-long body, a stored body that is not text. Instructions are a
    presentation preference, and a presentation preference must never fail a chat turn.

    In secret mode only the id is logged: the set name is free text the user wrote, and
    the turn it is attached to was explicitly asked not to be logged.
    """
    if not user or not set_id:
        return None
    # the whole resolution runs under the handler, not just the fetch: a body that is not
    # a str fails inside the slice or the envelope, and that must degrade too
    try:
        instruction_set = get_llm_config_db().get_instruction_set(user, set_id)
        if instruction_set is None or instruction_set.archived_at is not None:
            logger.info(f"Instruction set {set_id} does not resolve for this user, ignoring")
            return None

        # the accessor returns the authoritative stored body, which may predate the cap or
        # survive a lowering of it, so the prompt is bounded here. len() is code points
        body = instruction_set.body[:INSTRUCTION_SET_MAX_BODY_CHARS]
        # truncate before wrapping so the envelope's fence is computed over the text that
        # actually ships — a cut landing inside a backtick run would otherwise escape it
        fragment = instruction_envelope(body) or None
        if fragment is None:
            return None

        # id (and name, outside secret mode) only: the body is user-authored text and
        # never reaches the log. logged here so nothing claims to apply a set that the
        # whitespace-only check above just dropped
        applied = f"Applying instruction set {instruction_set.id}"
        if not secret:
            applied += f" ({instruction_set.name!r})"
        logger.info(
            applied + (" [truncated to the length cap]" if instruction_set.body_over_cap else "")
        )
        return fragment
    except Exception as e:
        logger.warning(f"Could not load instruction set {set_id}, continuing without it: {e}")
        return None


@app.post("/chat/v1/chat")
async def stream_chat(
    request: ChatRequest,
    user: str | None = Depends(auth_required),
    gateway_asserted: bool = Depends(gateway_asserted_identity),
):
    """
    Stream chat responses as Server-Sent Events (SSE).

    The response is a stream of JSON objects:
    - {"type": "content", "content": "text chunk"}
    - {"type": "thinking"}  (keepalive while the model reasons; no content)
    - {"type": "thinking_summary", "iteration": N, "text": "..."}  (only when the request
      set `capture_thinking`; the model's SUMMARIZED reasoning, never the raw chain)
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

    # enforce per-message size limits before any model call
    _validate_request_size(request.messages)
    _validate_latest_message(request.messages)

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

    # the system prompt is assembled server-side and is never client-supplied: it carries
    # the grounding, citation, truncation and out-of-scope rules, so a request may not
    # replace it. The response-length fragment is concatenated onto it because both are
    # identical for every user, so each setting keeps its own prompt-cache entry rather
    # than invalidating the other's. The user's instructions are NOT concatenated: they
    # travel separately and land in their own cache block after this one, which both
    # keeps the shared block cacheable across users and puts the envelope's guardrail
    # postamble last, where recency favours it.
    # assembled against the tool list THIS request will actually get, so the prompt never
    # documents a tool the model was not given (genetics-results-suite-4h6.69). Resolution
    # goes through the service rather than being recomputed here: one home for the
    # profile + feature-flag + subagent-liveness filtering that also builds the tool list.
    system_prompt = default_system_prompt(
        settings.app_name,
        tool_names=service.resolve_local_tool_names(request.tool_profile, request.enable_tools),
    )
    system_prompt += verbosity_prompt(request.verbosity)
    user_instructions = _resolve_user_instructions(
        user, request.instruction_set_id, secret=request.secret
    )

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
                session_id=request.session_id,
                user_instructions=user_instructions,
                message_id=request.message_id,
                capture_thinking=request.capture_thinking,
                gateway_asserted=gateway_asserted,
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
                elif chunk.type == "tool_use":
                    # one per tool call, carrying the input WHOLE — a run_analysis script is
                    # the thing the user most needs to read, and the prose marker this
                    # replaced cut it off at 400 chars with no way to expand it. The client
                    # renders a collapsed disclosure; a client that does not know this type
                    # drops it and simply shows no tool indicator.
                    yield {
                        "event": "message",
                        "data": json.dumps(
                            {"type": "tool_use", **json.loads(chunk.content)}
                        ),
                    }
                elif chunk.type == "thinking":
                    # keepalive only: carries no reasoning content, and exists so a long
                    # thinking phase doesn't read as a stalled stream to the client
                    yield {
                        "event": "message",
                        "data": json.dumps({"type": "thinking"}),
                    }
                elif chunk.type == "thinking_summary":
                    # only reached when the request opted in; the browser never does, so this
                    # branch is dead for ordinary chats rather than something they filter out
                    yield {
                        "event": "message",
                        "data": json.dumps(
                            {"type": "thinking_summary", **json.loads(chunk.content)}
                        ),
                    }
                elif chunk.type == "usage":
                    yield {
                        "event": "message",
                        "data": json.dumps({"type": "usage", **json.loads(chunk.content)}),
                    }
                elif chunk.type == "script_result":
                    # one per completed run_analysis. Metadata only (outcome, exception type,
                    # duration) — the script's source and output travel in the tool_result,
                    # not here. Unhandled chunk types are dropped silently by this dispatch,
                    # which is why the replay benchmark's script metrics need this branch.
                    yield {
                        "event": "message",
                        "data": json.dumps(
                            {"type": "script_result", **json.loads(chunk.content)}
                        ),
                    }
                elif chunk.type == "done":
                    yield {
                        "event": "message",
                        "data": json.dumps(
                            {
                                "type": "done",
                                "message_content": chunk.message_content,
                                "tool_results": chunk.tool_results,
                            }
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
async def download_file(download_id: str, user: str | None = Depends(auth_required)):
    """Serve a stored download file (TSV), to the user who generated it."""
    logger.info(f"Download requested: {download_id}")
    store = get_download_store()
    result = store.get(download_id, requester=user)
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
        "genetics_mcp_server.chat_api:app", host="0.0.0.0", port=args.port, reload=True, log_config=None
    )
