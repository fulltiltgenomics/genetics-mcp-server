"""
LLM Configuration router for managing editable system prompts and tool descriptions.

Provides endpoints for:
- Viewing/editing system prompts with version history
- Viewing/editing tool descriptions with version history
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from genetics_mcp_server.auth import auth_required
from genetics_mcp_server.db import get_llm_config_db
from genetics_mcp_server.tools import TOOL_DEFINITIONS

logger = logging.getLogger(__name__)

router = APIRouter()


class ToolDescriptionResponse(BaseModel):
    """Response containing tool description with metadata."""

    id: int
    tool_name: str
    description: str
    changed_by: str
    changed_at: str
    comment: Optional[str] = None


class ToolDescriptionUpdate(BaseModel):
    """Request body for updating tool description."""

    description: str = Field(..., description="New tool description")
    comment: Optional[str] = Field(None, description="Optional comment describing the change")


class DefaultToolDescription(BaseModel):
    """Default tool description from code."""

    tool_name: str
    description: str


class LLMConfigDefaults(BaseModel):
    """Default LLM configuration values from code."""

    tool_descriptions: list[DefaultToolDescription]


class UserCommentResponse(BaseModel):
    """Response containing a user's comment/note."""

    id: int
    comment: str
    created_at: str


class UserCommentCreate(BaseModel):
    """Request body for creating a user comment."""

    comment: str = Field(..., description="Comment text")


class UserSettingResponse(BaseModel):
    """Response containing a user setting."""

    id: int
    setting_key: str
    setting_value: str
    changed_at: str
    comment: Optional[str] = None


class UserSettingUpdate(BaseModel):
    """Request body for updating a user setting."""

    setting_value: str = Field(..., description="Setting value")
    comment: Optional[str] = Field(None, description="Optional comment describing the change")


@router.get(
    "/llm-config/defaults",
    summary="Get default tool descriptions",
    response_model=LLMConfigDefaults,
)
async def get_defaults(user: str = Depends(auth_required)):
    """
    Get the default tool descriptions from code.
    These are the fallback values used when no custom descriptions are saved.
    Note: System prompt is not exposed - users can only add additional instructions.
    """
    tool_defs = []
    for tool in TOOL_DEFINITIONS:
        tool_defs.append(DefaultToolDescription(
            tool_name=tool["name"],
            description=tool["description"],
        ))

    return LLMConfigDefaults(
        tool_descriptions=tool_defs,
    )


# user comments

@router.get(
    "/llm-config/user/comments",
    summary="Get user's comments",
    response_model=list[UserCommentResponse],
)
async def get_user_comments(user: str = Depends(auth_required)):
    """
    Get all comments/notes for the current user.
    """
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    db = get_llm_config_db()
    comments = db.get_user_comments(user)
    return [
        UserCommentResponse(
            id=c.id,
            comment=c.comment,
            created_at=c.created_at.isoformat(),
        )
        for c in comments
    ]


@router.post(
    "/llm-config/user/comments",
    summary="Add a user comment",
    response_model=UserCommentResponse,
)
async def add_user_comment(
    create: UserCommentCreate,
    user: str = Depends(auth_required),
):
    """
    Add a new comment/note for the current user.
    """
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    if not create.comment.strip():
        raise HTTPException(status_code=400, detail="Comment cannot be empty")

    db = get_llm_config_db()
    comment = db.add_user_comment(user, create.comment)
    logger.info(f"User comment added by {user}")
    return UserCommentResponse(
        id=comment.id,
        comment=comment.comment,
        created_at=comment.created_at.isoformat(),
    )


@router.delete(
    "/llm-config/user/comments/{comment_id}",
    summary="Delete a user comment",
)
async def delete_user_comment(
    comment_id: int,
    user: str = Depends(auth_required),
):
    """
    Delete a comment/note for the current user.
    """
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    db = get_llm_config_db()
    deleted = db.delete_user_comment(user, comment_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Comment not found")
    logger.info(f"User comment {comment_id} deleted by {user}")
    return {"deleted": True}


# user settings

@router.get(
    "/llm-config/user/settings",
    summary="Get all user settings",
    response_model=dict[str, UserSettingResponse],
)
async def get_user_settings(user: str = Depends(auth_required)):
    """
    Get all settings for the current user.
    Returns a dictionary keyed by setting name.
    """
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    db = get_llm_config_db()
    settings = db.get_user_settings(user)
    return {
        key: UserSettingResponse(
            id=s.id,
            setting_key=s.setting_key,
            setting_value=s.setting_value,
            changed_at=s.changed_at.isoformat(),
            comment=s.comment,
        )
        for key, s in settings.items()
    }


@router.get(
    "/llm-config/user/settings/{setting_key}",
    summary="Get a specific user setting",
    response_model=Optional[UserSettingResponse],
)
async def get_user_setting(
    setting_key: str,
    user: str = Depends(auth_required),
):
    """
    Get a specific setting for the current user.
    Returns null if the setting has not been customized.
    """
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    db = get_llm_config_db()
    setting = db.get_user_setting(user, setting_key)
    if setting is None or not setting.setting_value:
        return None
    return UserSettingResponse(
        id=setting.id,
        setting_key=setting.setting_key,
        setting_value=setting.setting_value,
        changed_at=setting.changed_at.isoformat(),
        comment=setting.comment,
    )


@router.put(
    "/llm-config/user/settings/{setting_key}",
    summary="Update a user setting",
    response_model=UserSettingResponse,
)
async def update_user_setting(
    setting_key: str,
    update: UserSettingUpdate,
    user: str = Depends(auth_required),
):
    """
    Save a setting for the current user.
    """
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    if not update.setting_value.strip():
        raise HTTPException(status_code=400, detail="Setting value cannot be empty")

    db = get_llm_config_db()
    setting = db.save_user_setting(
        user_id=user,
        setting_key=setting_key,
        setting_value=update.setting_value,
        comment=update.comment,
    )
    logger.info(f"User setting '{setting_key}' updated by {user}")
    return UserSettingResponse(
        id=setting.id,
        setting_key=setting.setting_key,
        setting_value=setting.setting_value,
        changed_at=setting.changed_at.isoformat(),
        comment=setting.comment,
    )


@router.delete(
    "/llm-config/user/settings/{setting_key}",
    summary="Delete a user setting",
)
async def delete_user_setting(
    setting_key: str,
    user: str = Depends(auth_required),
):
    """
    Delete a user setting (reset to default).
    """
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    db = get_llm_config_db()
    db.delete_user_setting(user, setting_key)
    logger.info(f"User setting '{setting_key}' deleted by {user}")
    return {"deleted": True}


# legacy global endpoints (kept for backward compatibility but deprecated)

@router.get(
    "/llm-config/tool-descriptions",
    summary="Get all tool descriptions",
    response_model=dict[str, ToolDescriptionResponse],
)
async def get_tool_descriptions(user: str = Depends(auth_required)):
    """
    Get the latest description for each tool.
    Returns a dictionary keyed by tool name.
    """
    db = get_llm_config_db()
    descriptions = db.get_tool_descriptions()
    return {
        name: ToolDescriptionResponse(
            id=desc.id,
            tool_name=desc.tool_name,
            description=desc.description,
            changed_by=desc.changed_by,
            changed_at=desc.changed_at.isoformat(),
            comment=desc.comment,
        )
        for name, desc in descriptions.items()
    }


@router.get(
    "/llm-config/tool-descriptions/{tool_name}",
    summary="Get description for a specific tool",
    response_model=Optional[ToolDescriptionResponse],
)
async def get_tool_description(
    tool_name: str,
    user: str = Depends(auth_required),
):
    """
    Get the latest description for a specific tool.
    Returns null if no custom description has been saved.
    """
    db = get_llm_config_db()
    desc = db.get_tool_description(tool_name)
    if desc is None:
        return None
    return ToolDescriptionResponse(
        id=desc.id,
        tool_name=desc.tool_name,
        description=desc.description,
        changed_by=desc.changed_by,
        changed_at=desc.changed_at.isoformat(),
        comment=desc.comment,
    )


@router.put(
    "/llm-config/tool-descriptions/{tool_name}",
    summary="Update tool description",
    response_model=ToolDescriptionResponse,
)
async def update_tool_description(
    tool_name: str,
    update: ToolDescriptionUpdate,
    user: str = Depends(auth_required),
):
    """
    Save a new version of a tool description.
    The change is tracked with the user's email and optional comment.
    """
    if not update.description.strip():
        raise HTTPException(status_code=400, detail="Tool description cannot be empty")

    db = get_llm_config_db()
    desc = db.save_tool_description(
        tool_name=tool_name,
        description=update.description,
        user=user or "anonymous",
        comment=update.comment,
    )
    logger.info(f"Tool '{tool_name}' description updated by {user or 'anonymous'}: {update.comment or 'no comment'}")
    return ToolDescriptionResponse(
        id=desc.id,
        tool_name=desc.tool_name,
        description=desc.description,
        changed_by=desc.changed_by,
        changed_at=desc.changed_at.isoformat(),
        comment=desc.comment,
    )


@router.get(
    "/llm-config/tool-descriptions/{tool_name}/history",
    summary="Get tool description change history",
    response_model=list[ToolDescriptionResponse],
)
async def get_tool_description_history(
    tool_name: str,
    limit: int = 20,
    user: str = Depends(auth_required),
):
    """
    Get recent versions of a tool description for audit/rollback reference.
    """
    db = get_llm_config_db()
    history = db.get_tool_description_history(tool_name, limit=limit)
    return [
        ToolDescriptionResponse(
            id=desc.id,
            tool_name=desc.tool_name,
            description=desc.description,
            changed_by=desc.changed_by,
            changed_at=desc.changed_at.isoformat(),
            comment=desc.comment,
        )
        for desc in history
    ]
