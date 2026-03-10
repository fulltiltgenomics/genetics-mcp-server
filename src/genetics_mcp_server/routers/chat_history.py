"""
Chat history router for managing persistent chat sessions.

Provides endpoints for:
- Creating and listing chat sessions
- Saving and retrieving messages
- Rating sessions and individual messages
- Generating chat titles via LLM
- Uploading and managing file attachments
"""

import logging
import os
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from genetics_mcp_server.auth import auth_required
from genetics_mcp_server.config import get_settings
from genetics_mcp_server.db import get_chat_history_db

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


class SessionCreateRequest(BaseModel):
    """Request to create a new session."""
    phenotype_code: Optional[str] = None


class SessionCreateResponse(BaseModel):
    """Response after creating a session."""
    id: str
    created_at: str


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
    tool_profile: Optional[str] = None  # api, bigquery, rag, or None (all)


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


class MessageSaveRequest(BaseModel):
    """Request to save a message."""
    id: str = Field(..., description="Message ID (generated on frontend)")
    role: str = Field(..., description="user or assistant")
    content: str
    content_json: Optional[str] = Field(None, description="JSON string of full message content blocks")
    literature_backend: Optional[str] = Field(None, description="Literature search backend: europepmc or perplexity")
    tool_profile: Optional[str] = Field(None, description="Tool profile: api, bigquery, rag, or null (all)")


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
    """Create a new chat session for the current user."""
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    db = get_chat_history_db()
    session = db.create_session(user, phenotype_code=request.phenotype_code)
    logger.info(f"Chat session created by {user}: {session.id}")

    return SessionCreateResponse(
        id=session.id,
        created_at=session.created_at.isoformat(),
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
    """Get a chat session with all its messages."""
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    db = get_chat_history_db()
    session = db.get_session(session_id, user)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = db.get_messages(session_id)

    return SessionDetailResponse(
        id=session.id,
        title=session.title,
        created_at=session.created_at.isoformat(),
        updated_at=session.updated_at.isoformat(),
        rating=session.rating,
        comment=session.comment,
        phenotype_code=session.phenotype_code,
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
        response = client.messages.create(
            model=settings.fast_model,
            max_tokens=50,
            messages=[
                {
                    "role": "user",
                    "content": f"Generate a very short title (3-6 words) for this chat conversation. Return only the title, no quotes or explanation.\n\nConversation:\n{conversation_text}",
                }
            ],
        )
        title = response.content[0].text.strip().strip('"\'')

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

    # generate unique ID and storage path
    attachment_id = str(uuid.uuid4())
    storage_dir = Path(settings.attachment_storage_path) / session_id
    storage_dir.mkdir(parents=True, exist_ok=True)

    # sanitize filename and create storage path
    safe_filename = Path(file.filename or "attachment").name
    storage_path = storage_dir / f"{attachment_id}_{safe_filename}"

    # write file to disk
    try:
        with open(storage_path, "wb") as f:
            f.write(content)
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
    user: str = Depends(auth_required),
):
    """Download a file attachment."""
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

    # delete file from disk
    try:
        if os.path.exists(attachment.storage_path):
            os.remove(attachment.storage_path)
    except Exception as e:
        logger.error(f"Failed to delete attachment file: {e}")

    # delete from database
    db.delete_attachment(attachment_id, session_id)

    logger.info(f"Attachment deleted: {attachment_id} from session {session_id}")
    return {"deleted": True}
