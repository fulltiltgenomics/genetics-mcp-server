"""
Admin router for viewing all conversations and usage analytics.

Requires ENABLE_ADMIN_PAGE=true and user in ADMIN_USERS list.
When REQUIRE_AUTH=false (dev mode), any user can access.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from genetics_mcp_server.auth import admin_required
from genetics_mcp_server.db import get_chat_history_db, get_llm_config_db

logger = logging.getLogger(__name__)

router = APIRouter()


# --- Pydantic Models ---


class AdminSessionItem(BaseModel):
    id: str
    user_id: str
    title: Optional[str]
    created_at: str
    updated_at: str
    rating: Optional[int] = None
    comment: Optional[str] = None
    phenotype_code: Optional[str] = None
    message_count: int = 0
    preview: Optional[str] = None


class AdminSessionListResponse(BaseModel):
    sessions: list[AdminSessionItem]
    total: int
    limit: int
    offset: int


class AdminMessageItem(BaseModel):
    id: str
    role: str
    content: str
    created_at: str
    thumbs_up: Optional[bool] = None


class AdminSessionDetailResponse(BaseModel):
    id: str
    user_id: str
    title: Optional[str]
    created_at: str
    updated_at: str
    rating: Optional[int] = None
    comment: Optional[str] = None
    phenotype_code: Optional[str] = None
    messages: list[AdminMessageItem]


class UsageDataPoint(BaseModel):
    date: str
    unique_users: int
    conversations: int


class UsageAnalyticsResponse(BaseModel):
    period: str
    data: list[UsageDataPoint]


class FeedbackItem(BaseModel):
    user: str
    comment: str
    preview: str
    created_at: str
    source: str
    session_id: Optional[str] = None


class FeedbackListResponse(BaseModel):
    items: list[FeedbackItem]
    total: int
    latest_at: Optional[str] = None
    limit: int
    offset: int


# --- Endpoints ---


@router.get("/admin/sessions", response_model=AdminSessionListResponse)
async def list_all_sessions(
    limit: int = 50,
    offset: int = 0,
    user: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    session_id: Optional[str] = None,
    admin_user: str = Depends(admin_required),
):
    """List all sessions across all users with optional filters."""
    db = get_chat_history_db()
    sessions, total = db.list_all_sessions(
        limit=limit,
        offset=offset,
        user_filter=user,
        date_from=date_from,
        date_to=date_to,
        session_id_filter=session_id,
    )

    items = []
    for s in sessions:
        preview = db.get_first_user_message(s.id)
        messages = db.get_messages(s.id)
        items.append(AdminSessionItem(
            id=s.id,
            user_id=s.user_id,
            title=s.title,
            created_at=s.created_at.isoformat(),
            updated_at=s.updated_at.isoformat(),
            rating=s.rating,
            comment=s.comment,
            phenotype_code=s.phenotype_code,
            message_count=len(messages),
            preview=preview[:100] if preview else None,
        ))

    return AdminSessionListResponse(
        sessions=items,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/admin/sessions/{session_id}", response_model=AdminSessionDetailResponse)
async def get_session_detail(
    session_id: str,
    admin_user: str = Depends(admin_required),
):
    """Get full session detail with all messages (admin access, no user check)."""
    db = get_chat_history_db()
    session = db.get_session_any_user(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = db.get_messages(session_id)
    return AdminSessionDetailResponse(
        id=session.id,
        user_id=session.user_id,
        title=session.title,
        created_at=session.created_at.isoformat(),
        updated_at=session.updated_at.isoformat(),
        rating=session.rating,
        comment=session.comment,
        phenotype_code=session.phenotype_code,
        messages=[
            AdminMessageItem(
                id=m.id,
                role=m.role,
                content=m.content,
                created_at=m.created_at.isoformat(),
                thumbs_up=m.thumbs_up,
            )
            for m in messages
        ],
    )


@router.get("/admin/analytics/usage", response_model=UsageAnalyticsResponse)
async def get_usage_analytics(
    period: str = "week",
    admin_user: str = Depends(admin_required),
):
    """Get daily usage analytics (unique users and conversations per day)."""
    if period not in ("week", "month", "year"):
        raise HTTPException(status_code=400, detail="period must be 'week', 'month', or 'year'")

    db = get_chat_history_db()
    data = db.get_usage_analytics(period)
    return UsageAnalyticsResponse(
        period=period,
        data=[UsageDataPoint(**d) for d in data],
    )


@router.get("/admin/feedback", response_model=FeedbackListResponse)
async def list_feedback(
    limit: int = 50,
    offset: int = 0,
    admin_user: str = Depends(admin_required),
):
    """List all user feedback from both feedback dialogs and session comments."""
    # collect feedback from both sources
    items: list[FeedbackItem] = []

    for uc in get_llm_config_db().list_all_user_comments():
        items.append(FeedbackItem(
            user=uc.user_id,
            comment=uc.comment,
            preview=uc.comment[:100],
            created_at=uc.created_at.isoformat(),
            source="feedback_dialog",
        ))

    for sc in get_chat_history_db().list_sessions_with_comments():
        comment = sc["comment"]
        items.append(FeedbackItem(
            user=sc["user_id"],
            comment=comment,
            preview=comment[:100],
            created_at=sc["created_at"].isoformat(),
            source="session_comment",
            session_id=sc["session_id"],
        ))

    # sort by created_at descending
    items.sort(key=lambda x: x.created_at, reverse=True)

    total = len(items)
    latest_at = items[0].created_at if items else None
    page = items[offset:offset + limit]

    return FeedbackListResponse(
        items=page,
        total=total,
        latest_at=latest_at,
        limit=limit,
        offset=offset,
    )
