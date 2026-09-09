"""
Chat history router for managing persistent chat sessions.

Provides endpoints for:
- Creating and listing chat sessions
- Saving and retrieving messages
- Rating sessions and individual messages
- Generating chat titles via LLM
- Uploading and managing file attachments
"""

import io
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from genetics_mcp_server.auth import auth_required
from genetics_mcp_server.config import get_settings, model_rejects_disabled_thinking
from genetics_mcp_server.db import get_chat_history_db
from genetics_mcp_server.memory_digest import MAX_DIGEST_CHARS, render_digest
from genetics_mcp_server.memory_gate import (
    MEMORY_PROJECT_SESSION_CAP,
    is_identifiable_user,
    memory_setting_on,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# --- Pydantic Models ---

class SessionListItem(BaseModel):
    """Session summary for list view."""
    id: str
    title: Optional[str]
    created_at: str
    updated_at: str
    preview: Optional[str] = None
    rating: Optional[int] = None
    pinned: bool = False
    project_id: Optional[str] = None


class SessionCreateRequest(BaseModel):
    """Request to create a new session, optionally filed into a project."""
    phenotype_code: Optional[str] = None
    project_id: Optional[str] = None


class SessionCreateResponse(BaseModel):
    """Response after creating a session."""
    id: str
    created_at: str
    project_id: Optional[str] = None


class SessionUpdateRequest(BaseModel):
    """Request to update session metadata."""
    title: Optional[str] = None
    rating: Optional[int] = Field(None, ge=1, le=5)
    comment: Optional[str] = None


class MessageResponse(BaseModel):
    """A chat message."""
    id: str
    role: str
    content: str
    created_at: str
    thumbs_up: Optional[bool] = None
    content_json: Optional[str] = None  # JSON string of full message content blocks
    literature_backend: Optional[str] = None  # europepmc or perplexity
    tool_profile: Optional[str] = None  # api, bigquery, rag, code, or None (all)
    tool_results_json: Optional[str] = None  # JSON string of tool_result blocks for this assistant turn
    instruction_set_id: Optional[str] = None  # user instruction set in force for this turn
    verbosity: Optional[str] = None  # answer detail in force for this turn: brief or detailed


class SessionDetailResponse(BaseModel):
    """Full session details with messages."""
    id: str
    title: Optional[str]
    created_at: str
    updated_at: str
    rating: Optional[int] = None
    comment: Optional[str] = None
    phenotype_code: Optional[str] = None
    messages: list[MessageResponse]
    is_owner: Optional[bool] = None
    shared: Optional[bool] = None
    project_id: Optional[str] = None


class ShareRequest(BaseModel):
    """Request to toggle session sharing."""
    shared: bool


class PinRequest(BaseModel):
    """Request to pin or unpin a session."""
    pinned: bool


class MemorySessionItem(BaseModel):
    """One session as it enters the memory digest."""
    id: str
    title: Optional[str]
    pinned: bool
    created_at: str


class MemoryResponse(BaseModel):
    """What the memory dialog shows: the opt-in state and a preview of the digest."""
    enabled: bool
    digest: str
    sessions: list[MemorySessionItem]
    char_cap: int


class ProjectCreateRequest(BaseModel):
    """Request to create a project."""
    name: str


class ProjectUpdateRequest(BaseModel):
    """Request to rename a project."""
    name: str


class ProjectResponse(BaseModel):
    """A project as the sidebar shows it."""
    id: str
    name: str
    created_at: str
    updated_at: str
    last_activity_at: Optional[str] = None


class SessionProjectRequest(BaseModel):
    """Request to file a session into a project, or unfile it with project_id=None."""
    project_id: Optional[str] = None


class SessionProjectResponse(BaseModel):
    """Response after filing or unfiling a session."""
    id: str
    project_id: Optional[str] = None


class MessageSaveRequest(BaseModel):
    """Request to save a message."""
    id: str = Field(..., description="Message ID (generated on frontend)")
    role: str = Field(..., description="user or assistant")
    content: str
    content_json: Optional[str] = Field(None, description="JSON string of full message content blocks")
    literature_backend: Optional[str] = Field(None, description="Literature search backend: europepmc or perplexity")
    tool_profile: Optional[str] = Field(None, description="Tool profile: api, bigquery, rag, code, or null (all)")
    tool_results_json: Optional[str] = Field(None, description="JSON string of tool_result blocks for this assistant turn")
    # add_message re-saves the whole row on conflict, so a re-save that omits this clears the
    # stored value — same semantics as tool_profile above. Every save of a message must carry it
    instruction_set_id: Optional[str] = Field(None, description="Instruction set in force for this turn")
    verbosity: Optional[str] = Field(None, description="Answer detail in force for this turn: brief or detailed")


class MessageRatingRequest(BaseModel):
    """Request to rate a message."""
    thumbs_up: Optional[bool] = Field(None, description="true=up, false=down, null=clear")


class TitleGenerateResponse(BaseModel):
    """Response with generated title."""
    title: str


class AttachmentResponse(BaseModel):
    """A file attachment."""
    id: str
    name: str
    type: str  # image, tsv, excel
    mime_type: str
    size: int
    created_at: str


# allowed MIME types for attachments
ALLOWED_MIME_TYPES = {
    # images
    "image/png": "image",
    "image/jpeg": "image",
    "image/gif": "image",
    "image/webp": "image",
    # tabular data
    "text/tab-separated-values": "tsv",
    "text/csv": "tsv",
    "text/plain": "tsv",  # sometimes TSV comes as text/plain
    # excel
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "excel",
    "application/vnd.ms-excel": "excel",
}

# file extension to type mapping as fallback
EXTENSION_TO_TYPE = {
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".gif": "image",
    ".webp": "image",
    ".tsv": "tsv",
    ".csv": "tsv",
    ".xlsx": "excel",
    ".xls": "excel",
}


def get_attachment_type(mime_type: str, filename: str) -> str | None:
    """Determine the attachment type from MIME type or file extension."""
    if mime_type in ALLOWED_MIME_TYPES:
        return ALLOWED_MIME_TYPES[mime_type]
    # fallback to extension
    ext = Path(filename).suffix.lower()
    return EXTENSION_TO_TYPE.get(ext)


def excel_to_tsv(content: bytes) -> str:
    """Convert an Excel workbook (.xlsx/.xls) to TSV text.

    Excel is a binary format, so it must be decoded to text before the model
    can read it. Reads every sheet; with more than one, each is prefixed with a
    "# Sheet: <name>" header so the model can tell them apart.
    """
    import polars as pl

    # sheet_id=0 returns all sheets as {name: DataFrame}
    sheets = pl.read_excel(io.BytesIO(content), sheet_id=0)
    multi = len(sheets) > 1
    parts = []
    for name, df in sheets.items():
        body = df.write_csv(separator="\t")
        parts.append(f"# Sheet: {name}\n{body}" if multi else body)
    return "\n".join(parts)


# --- Session Endpoints ---

@router.get(
    "/chat/sessions",
    summary="List user's chat sessions",
    response_model=list[SessionListItem],
)
async def list_sessions(
    limit: int = 50,
    user: str = Depends(auth_required),
):
    """Get a list of the user's chat sessions, most recent first."""
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    db = get_chat_history_db()
    sessions = db.list_sessions(user, limit=limit)

    result = []
    for session in sessions:
        preview = None
        if not session.title:
            # get first user message as preview
            preview_text = db.get_first_user_message(session.id)
            if preview_text:
                preview = preview_text[:80] + "..." if len(preview_text) > 80 else preview_text

        result.append(SessionListItem(
            id=session.id,
            title=session.title,
            created_at=session.created_at.isoformat(),
            updated_at=session.updated_at.isoformat(),
            preview=preview,
            rating=session.rating,
            pinned=session.pinned_at is not None,
            project_id=session.project_id,
        ))

    return result


@router.post(
    "/chat/sessions",
    summary="Create a new chat session",
    response_model=SessionCreateResponse,
)
async def create_session(
    request: SessionCreateRequest,
    user: str = Depends(auth_required),
):
    """Create a new chat session for the current user, optionally filed into a project."""
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    db = get_chat_history_db()
    try:
        session = db.create_session(
            user, phenotype_code=request.phenotype_code, project_id=request.project_id
        )
    except ValueError:
        if request.project_id is not None:
            # a foreign or missing project_id, same 404 shape as any other owner-only check
            raise HTTPException(status_code=404, detail="Project not found") from None
        raise
    logger.info(f"Chat session created by {user}: {session.id}")

    return SessionCreateResponse(
        id=session.id,
        created_at=session.created_at.isoformat(),
        project_id=session.project_id,
    )


@router.get(
    "/chat/sessions/{session_id}",
    summary="Get session details with messages",
    response_model=SessionDetailResponse,
)
async def get_session(
    session_id: str,
    user: str = Depends(auth_required),
):
    """Get a chat session with all its messages. Returns session if owned or shared."""
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    db = get_chat_history_db()
    result = db.get_session_for_access(session_id, user)
    if result is None:
        raise HTTPException(status_code=404, detail="Session not found")

    session, is_owner = result
    messages = db.get_messages(session_id)

    return SessionDetailResponse(
        id=session.id,
        title=session.title,
        created_at=session.created_at.isoformat(),
        updated_at=session.updated_at.isoformat(),
        rating=session.rating,
        comment=session.comment,
        phenotype_code=session.phenotype_code,
        is_owner=is_owner,
        shared=session.shared,
        project_id=session.project_id,
        messages=[
            MessageResponse(
                id=msg.id,
                role=msg.role,
                content=msg.content,
                created_at=msg.created_at.isoformat(),
                thumbs_up=msg.thumbs_up,
                content_json=msg.content_json,
                literature_backend=msg.literature_backend,
                tool_profile=msg.tool_profile,
                tool_results_json=msg.tool_results_json,
                instruction_set_id=msg.instruction_set_id,
                verbosity=msg.verbosity,
            )
            for msg in messages
        ],
    )


@router.put(
    "/chat/sessions/{session_id}",
    summary="Update session metadata",
)
async def update_session(
    session_id: str,
    request: SessionUpdateRequest,
    user: str = Depends(auth_required),
):
    """Update session title, rating, or comment."""
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    db = get_chat_history_db()
    updated = db.update_session(
        session_id,
        user,
        title=request.title,
        rating=request.rating,
        comment=request.comment,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Session not found")

    logger.info(f"Chat session {session_id} updated by {user}")
    return {"updated": True}


@router.delete(
    "/chat/sessions/{session_id}",
    summary="Delete a chat session",
)
async def delete_session(
    session_id: str,
    user: str = Depends(auth_required),
):
    """Delete a chat session and all its messages."""
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    db = get_chat_history_db()
    deleted = db.delete_session(session_id, user)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")

    logger.info(f"Chat session {session_id} deleted by {user}")
    return {"deleted": True}


@router.put(
    "/chat/sessions/{session_id}/share",
    summary="Toggle session sharing",
)
async def share_session(
    session_id: str,
    request: ShareRequest,
    user: str = Depends(auth_required),
):
    """Toggle the shared flag on a session. Only the owner can share/unshare."""
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    db = get_chat_history_db()
    updated = db.set_shared(session_id, user, request.shared)
    if not updated:
        # check if session exists but user doesn't own it
        session = db.get_session_any_user(session_id)
        if session is not None:
            raise HTTPException(status_code=403, detail="Not the session owner")
        raise HTTPException(status_code=404, detail="Session not found")

    logger.info(f"Session {session_id} shared={request.shared} by {user}")
    return {"shared": request.shared}


@router.put(
    "/chat/sessions/{session_id}/pin",
    summary="Pin or unpin a session for chat memory",
)
async def pin_session(
    session_id: str,
    request: PinRequest,
    user: str = Depends(auth_required),
):
    """Pin a session so it keeps surfacing in the memory digest. Owner only.

    A session the caller does not own is reported as missing rather than forbidden: the id
    is caller-supplied, and a 403 would confirm that somebody else's session exists. Secret
    chats never wrote a row, so they are 404 here by construction.
    """
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    # a pin is a memory control, so it takes the same caller the project memory
    # endpoint takes: with
    # REQUIRE_AUTH off every local caller is the same `anonymous`, and a pin from one of
    # them would steer a digest that is not theirs
    if not is_identifiable_user(user):
        raise HTTPException(status_code=404, detail="Not found")

    db = get_chat_history_db()
    if not db.set_pinned(session_id, user, request.pinned):
        raise HTTPException(status_code=404, detail="Session not found")

    logger.info(f"Session {session_id} pinned={request.pinned} by {user}")
    return {"id": session_id, "pinned": request.pinned}


@router.put(
    "/chat/sessions/{session_id}/project",
    summary="File a session into a project, or unfile it",
    response_model=SessionProjectResponse,
)
async def set_session_project(
    session_id: str,
    request: SessionProjectRequest,
    user: str = Depends(auth_required),
):
    """File the session into request.project_id, or unfile it with project_id=None.

    Owner-only on both ends: set_session_project returns False for a session that is not
    the caller's or a project_id that names another user's project, and either is reported
    as a plain 404 — the same shape a caller-supplied foreign id gets everywhere else in
    this router.
    """
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    # same gate the pin endpoint carries: a shared identity must not steer a digest
    # that is not theirs, and filing a session into a project does exactly that
    if not is_identifiable_user(user):
        raise HTTPException(status_code=404, detail="Not found")

    db = get_chat_history_db()
    if not db.set_session_project(user, session_id, request.project_id):
        raise HTTPException(status_code=404, detail="Session or project not found")

    logger.info(f"Session {session_id} filed into project {request.project_id} by {user}")
    return SessionProjectResponse(id=session_id, project_id=request.project_id)


# --- Project Endpoints ---

def _find_project(db, user: str, project_id: str):
    """The caller's own project by id, or None. Owner-only lookup mirroring get_session."""
    for project in db.list_projects(user):
        if project.id == project_id:
            return project
    return None


def _project_response(project) -> ProjectResponse:
    return ProjectResponse(
        id=project.id,
        name=project.name,
        created_at=project.created_at.isoformat(),
        updated_at=project.updated_at.isoformat(),
        last_activity_at=(
            project.last_activity_at.isoformat() if project.last_activity_at else None
        ),
    )


@router.get(
    "/projects",
    summary="List the caller's projects",
    response_model=list[ProjectResponse],
)
async def list_projects(user: str = Depends(auth_required)):
    """The caller's own projects, most recently active first."""
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    if not is_identifiable_user(user):
        raise HTTPException(status_code=404, detail="Not found")

    db = get_chat_history_db()
    return [_project_response(p) for p in db.list_projects(user)]


@router.post(
    "/projects",
    summary="Create a project",
    response_model=ProjectResponse,
)
async def create_project(
    request: ProjectCreateRequest,
    user: str = Depends(auth_required),
):
    """Create a project for the caller. 400 on a blank name or past the per-user cap."""
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    if not is_identifiable_user(user):
        raise HTTPException(status_code=404, detail="Not found")

    db = get_chat_history_db()
    try:
        project = db.create_project(user, request.name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None

    logger.info(f"Project {project.id} created by {user}")
    return _project_response(project)


@router.put(
    "/projects/{project_id}",
    summary="Rename a project",
    response_model=ProjectResponse,
)
async def update_project(
    project_id: str,
    request: ProjectUpdateRequest,
    user: str = Depends(auth_required),
):
    """Rename the caller's own project. 400 on a blank name, 404 if not theirs."""
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    if not is_identifiable_user(user):
        raise HTTPException(status_code=404, detail="Not found")

    db = get_chat_history_db()
    try:
        renamed = db.rename_project(user, project_id, request.name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    if not renamed:
        raise HTTPException(status_code=404, detail="Project not found")

    logger.info(f"Project {project_id} renamed by {user}")
    project = _find_project(db, user, project_id)
    if project is None:
        # renamed True but the row is gone by the time we re-read it: archived or a
        # concurrent delete raced us between the rename and this lookup
        raise HTTPException(status_code=404, detail="Project not found")
    return _project_response(project)


@router.delete(
    "/projects/{project_id}",
    summary="Delete a project",
)
async def delete_project(
    project_id: str,
    with_sessions: bool = Query(
        False, description="Also delete every session filed in the project"
    ),
    user: str = Depends(auth_required),
):
    """Delete the caller's own project.

    By default the project's sessions survive, unfiled. with_sessions=true deletes them
    too, one delete_session call per row so the message cascade and turn-metrics cleanup
    both run for each.
    """
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    if not is_identifiable_user(user):
        raise HTTPException(status_code=404, detail="Not found")

    db = get_chat_history_db()
    deleted = (
        db.delete_project_with_sessions(user, project_id)
        if with_sessions
        else db.delete_project(user, project_id)
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="Project not found")

    logger.info(f"Project {project_id} deleted by {user} (with_sessions={with_sessions})")
    return {"deleted": True}


@router.get(
    "/projects/{project_id}/sessions",
    summary="List the sessions filed in a project",
    response_model=list[SessionListItem],
)
async def list_project_sessions(
    project_id: str,
    user: str = Depends(auth_required),
):
    """The caller's own sessions filed into this project, most recent first."""
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    if not is_identifiable_user(user):
        raise HTTPException(status_code=404, detail="Not found")

    db = get_chat_history_db()
    if _find_project(db, user, project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")

    # generous enough to hand back a whole project's worth of sessions in one page;
    # list_sessions' 50-item default is sized for the unfiled/all-sessions view instead
    sessions = db.list_sessions_in_project(user, project_id, limit=500)

    result = []
    for session in sessions:
        preview = None
        if not session.title:
            preview_text = db.get_first_user_message(session.id)
            if preview_text:
                preview = preview_text[:80] + "..." if len(preview_text) > 80 else preview_text

        result.append(SessionListItem(
            id=session.id,
            title=session.title,
            created_at=session.created_at.isoformat(),
            updated_at=session.updated_at.isoformat(),
            preview=preview,
            rating=session.rating,
            pinned=session.pinned_at is not None,
            project_id=session.project_id,
        ))

    return result


@router.get(
    "/projects/{project_id}/memory",
    summary="What the next session filed in this project will remember",
    response_model=MemoryResponse,
)
async def get_project_memory(
    project_id: str,
    user: str = Depends(auth_required),
):
    """The caller's memory digest for this project, rendered fresh.

    This is what the NEXT session filed here will be seeded with, not what an existing
    session carries: a session's digest is frozen on its first turn, so reading that copy
    back would show the user a stale index of the project's history.

    The preview is returned whether or not the chat_memory setting is on, so the dialog can
    show what would be remembered before the user opts in. Nothing is written either way.

    Unlike prompt injection, this read does NOT require `gateway_asserted`. Every private
    read in this router — session detail, messages, attachments — authorizes on the
    resolved `user` alone, and the digest is rendered from those same rows, so this
    endpoint discloses nothing the router does not already hand out. Requiring the flag
    here would harden one endpoint while its source rows stayed reachable next door.
    """
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    # a service identity and the shared `anonymous` of an auth-less deployment name no
    # person, so they own no project and no memory — reported as missing, like an id that
    # resolves to nobody, rather than as a permission the caller could acquire
    if not is_identifiable_user(user):
        raise HTTPException(status_code=404, detail="Not found")

    db = get_chat_history_db()
    if _find_project(db, user, project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")

    sessions = db.get_recent_sessions_for_digest(
        user, MEMORY_PROJECT_SESSION_CAP, project_id=project_id, include_pinned=True
    )
    return MemoryResponse(
        enabled=memory_setting_on(user),
        digest=render_digest(sessions, datetime.now(timezone.utc)),
        # the rows that were handed to the renderer, in the order it saw them. The renderer
        # drops its oldest entries when the character cap binds, so a long list can name a
        # session whose line did not survive into `digest`
        sessions=[
            MemorySessionItem(
                id=row["id"],
                title=row["title"],
                pinned=row["pinned_at"] is not None,
                created_at=str(row["created_at"]),
            )
            for row in sessions
        ],
        char_cap=MAX_DIGEST_CHARS,
    )


@router.post(
    "/chat/sessions/{session_id}/fork",
    summary="Fork a shared session",
    response_model=SessionCreateResponse,
)
async def fork_session(
    session_id: str,
    user: str = Depends(auth_required),
):
    """Fork a shared session for the current user. Creates a copy with all messages."""
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    db = get_chat_history_db()
    new_session = db.fork_session(session_id, user)
    if new_session is None:
        raise HTTPException(status_code=404, detail="Session not found or not shared")

    logger.info(f"Session {session_id} forked by {user} as {new_session.id}")
    return SessionCreateResponse(
        id=new_session.id,
        created_at=new_session.created_at.isoformat(),
        project_id=new_session.project_id,
    )


# --- Message Endpoints ---

@router.post(
    "/chat/sessions/{session_id}/messages",
    summary="Save a message to a session",
    response_model=MessageResponse,
)
async def save_message(
    session_id: str,
    request: MessageSaveRequest,
    user: str = Depends(auth_required),
):
    """Save a chat message to a session."""
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    if request.role not in ("user", "assistant"):
        raise HTTPException(status_code=400, detail="Role must be 'user' or 'assistant'")

    db = get_chat_history_db()

    # verify session ownership
    session = db.get_session(session_id, user)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    msg = db.add_message(
        session_id,
        request.id,
        request.role,
        request.content,
        request.content_json,
        request.literature_backend,
        request.tool_profile,
        request.tool_results_json,
        request.instruction_set_id,
        request.verbosity,
    )

    return MessageResponse(
        id=msg.id,
        role=msg.role,
        content=msg.content,
        created_at=msg.created_at.isoformat(),
        thumbs_up=msg.thumbs_up,
        content_json=msg.content_json,
        literature_backend=msg.literature_backend,
        tool_profile=msg.tool_profile,
        tool_results_json=msg.tool_results_json,
        instruction_set_id=msg.instruction_set_id,
        verbosity=msg.verbosity,
    )


@router.put(
    "/chat/messages/{message_id}/rating",
    summary="Rate a message",
)
async def rate_message(
    message_id: str,
    request: MessageRatingRequest,
    user: str = Depends(auth_required),
):
    """Rate a message with thumbs up or down."""
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    db = get_chat_history_db()
    updated = db.rate_message(message_id, request.thumbs_up)
    if not updated:
        raise HTTPException(status_code=404, detail="Message not found")

    return {"updated": True}


# --- Title Generation ---

@router.post(
    "/chat/sessions/{session_id}/generate-title",
    summary="Generate a title using LLM",
    response_model=TitleGenerateResponse,
)
async def generate_title(
    session_id: str,
    user: str = Depends(auth_required),
):
    """Generate a short title for the chat using the LLM."""
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    db = get_chat_history_db()
    session = db.get_session(session_id, user)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = db.get_messages(session_id)
    if not messages:
        raise HTTPException(status_code=400, detail="No messages to generate title from")

    # build context from first few messages
    context_messages = messages[:4]
    conversation_text = "\n".join(
        f"{msg.role}: {msg.content[:500]}" for msg in context_messages
    )

    # generate title using Anthropic (simpler non-streaming call)
    try:
        from anthropic import Anthropic

        settings = get_settings()
        client = Anthropic(api_key=settings.anthropic_api_key)
        # a 3-6 word title needs no reasoning, and max_tokens covers thinking and
        # visible text together — 50 tokens of thinking would leave nothing for the
        # title. Models that think by default (Opus 5) have to be told explicitly.
        thinking_kwargs = (
            {} if model_rejects_disabled_thinking(settings.fast_model)
            else {"thinking": {"type": "disabled"}}
        )
        response = client.messages.create(
            model=settings.fast_model,
            max_tokens=50,
            messages=[
                {
                    "role": "user",
                    "content": f"Generate a very short title (3-6 words) for this chat conversation. Return only the title, no quotes or explanation.\n\nConversation:\n{conversation_text}",
                }
            ],
            **thinking_kwargs,
        )
        # never index content[0]: a thinking-capable model leads with a block that
        # has no .text
        title = "".join(
            b.text for b in response.content if b.type == "text"
        ).strip().strip('"\'')
        if not title:
            # e.g. the whole budget went to thinking on a model that can't be
            # turned off — fall back to the first user message rather than
            # storing an empty title
            raise ValueError(f"model returned no title text (stop_reason={response.stop_reason})")

        # save title to database
        db.update_session(session_id, user, title=title)
        logger.info(f"Title generated for session {session_id}: {title}")

        return TitleGenerateResponse(title=title)

    except Exception as e:
        logger.error(f"Error generating title: {e}")
        # fallback: use first user message
        first_msg = next((m for m in messages if m.role == "user"), None)
        if first_msg:
            fallback_title = first_msg.content[:40] + "..." if len(first_msg.content) > 40 else first_msg.content
            db.update_session(session_id, user, title=fallback_title)
            return TitleGenerateResponse(title=fallback_title)
        raise HTTPException(status_code=500, detail="Failed to generate title")


# --- Attachment Endpoints ---

@router.post(
    "/chat/sessions/{session_id}/attachments",
    summary="Upload a file attachment",
    response_model=AttachmentResponse,
)
async def upload_attachment(
    session_id: str,
    file: UploadFile = File(...),
    user: str = Depends(auth_required),
):
    """Upload a file attachment to a session."""
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    db = get_chat_history_db()
    settings = get_settings()

    # verify session ownership
    session = db.get_session(session_id, user)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # validate file type
    file_type = get_attachment_type(file.content_type or "", file.filename or "")
    if file_type is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Allowed: images, TSV, CSV, Excel"
        )

    # read file content
    content = await file.read()
    file_size = len(content)

    # check size limit
    if file_size > settings.max_attachment_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {settings.max_attachment_size // (1024*1024)}MB"
        )

    # parse Excel to TSV up front so a bad file fails before anything is written
    parsed_tsv: str | None = None
    if file_type == "excel":
        try:
            parsed_tsv = excel_to_tsv(content)
        except Exception as e:
            logger.warning(f"Failed to parse Excel attachment {file.filename}: {e}")
            raise HTTPException(
                status_code=400,
                detail="Could not read this Excel file. Please re-save it or upload as CSV/TSV.",
            )

    # generate unique ID and storage path
    attachment_id = str(uuid.uuid4())
    storage_dir = Path(settings.attachment_storage_path) / session_id
    storage_dir.mkdir(parents=True, exist_ok=True)

    # sanitize filename and create storage path
    safe_filename = Path(file.filename or "attachment").name
    storage_path = storage_dir / f"{attachment_id}_{safe_filename}"

    # write file to disk (original bytes preserved; parsed TSV stored as a sidecar)
    text_path: str | None = None
    try:
        with open(storage_path, "wb") as f:
            f.write(content)
        if parsed_tsv is not None:
            text_path = f"{storage_path}.tsv"
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(parsed_tsv)
    except Exception as e:
        logger.error(f"Failed to write attachment file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save file")

    # save to database
    attachment = db.add_attachment(
        attachment_id=attachment_id,
        session_id=session_id,
        file_name=safe_filename,
        file_type=file_type,
        mime_type=file.content_type or "application/octet-stream",
        file_size=file_size,
        storage_path=str(storage_path),
        text_path=text_path,
    )

    logger.info(f"Attachment uploaded: {attachment_id} ({safe_filename}) to session {session_id}")

    return AttachmentResponse(
        id=attachment.id,
        name=attachment.file_name,
        type=attachment.file_type,
        mime_type=attachment.mime_type,
        size=attachment.file_size,
        created_at=attachment.created_at.isoformat(),
    )


@router.get(
    "/chat/sessions/{session_id}/attachments",
    summary="List session attachments",
    response_model=list[AttachmentResponse],
)
async def list_attachments(
    session_id: str,
    user: str = Depends(auth_required),
):
    """List all attachments for a session."""
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    db = get_chat_history_db()

    # verify session ownership
    session = db.get_session(session_id, user)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    attachments = db.get_session_attachments(session_id)

    return [
        AttachmentResponse(
            id=att.id,
            name=att.file_name,
            type=att.file_type,
            mime_type=att.mime_type,
            size=att.file_size,
            created_at=att.created_at.isoformat(),
        )
        for att in attachments
    ]


@router.get(
    "/chat/sessions/{session_id}/attachments/{attachment_id}",
    summary="Download an attachment",
)
async def get_attachment(
    session_id: str,
    attachment_id: str,
    as_: str | None = Query(default=None, alias="as"),
    user: str = Depends(auth_required),
):
    """Download a file attachment.

    With ``?as=text`` the model-ready text representation is returned instead of
    the raw bytes: parsed TSV for Excel, the original text for TSV/CSV. Clients
    should inline this (not the raw file) so the model receives readable text.
    """
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    db = get_chat_history_db()

    # verify session ownership
    session = db.get_session(session_id, user)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    attachment = db.get_attachment(attachment_id, session_id)
    if attachment is None:
        raise HTTPException(status_code=404, detail="Attachment not found")

    if as_ == "text":
        # parsed sidecar for Excel; tsv/csv are already text and served as-is
        if attachment.text_path and os.path.exists(attachment.text_path):
            return FileResponse(
                path=attachment.text_path,
                filename=f"{attachment.file_name}.tsv",
                media_type="text/tab-separated-values",
            )
        if attachment.file_type == "tsv" and os.path.exists(attachment.storage_path):
            return FileResponse(
                path=attachment.storage_path,
                filename=attachment.file_name,
                media_type="text/tab-separated-values",
            )
        raise HTTPException(
            status_code=415, detail="No text representation available for this attachment"
        )

    # verify file exists
    if not os.path.exists(attachment.storage_path):
        logger.error(f"Attachment file not found on disk: {attachment.storage_path}")
        raise HTTPException(status_code=404, detail="Attachment file not found")

    return FileResponse(
        path=attachment.storage_path,
        filename=attachment.file_name,
        media_type=attachment.mime_type,
    )


@router.delete(
    "/chat/sessions/{session_id}/attachments/{attachment_id}",
    summary="Delete an attachment",
)
async def delete_attachment(
    session_id: str,
    attachment_id: str,
    user: str = Depends(auth_required),
):
    """Delete a file attachment."""
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    db = get_chat_history_db()

    # verify session ownership
    session = db.get_session(session_id, user)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    attachment = db.get_attachment(attachment_id, session_id)
    if attachment is None:
        raise HTTPException(status_code=404, detail="Attachment not found")

    # delete file from disk (and the parsed sidecar, if any)
    try:
        if os.path.exists(attachment.storage_path):
            os.remove(attachment.storage_path)
        if attachment.text_path and os.path.exists(attachment.text_path):
            os.remove(attachment.text_path)
    except Exception as e:
        logger.error(f"Failed to delete attachment file: {e}")

    # delete from database
    db.delete_attachment(attachment_id, session_id)

    logger.info(f"Attachment deleted: {attachment_id} from session {session_id}")
    return {"deleted": True}
