"""API token router for managing per-user MCP server access tokens."""

import hmac
import logging
import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from genetics_mcp_server.auth import auth_required
from genetics_mcp_server.db import get_llm_config_db

logger = logging.getLogger(__name__)

router = APIRouter()

_internal_api_secret = os.environ.get("INTERNAL_API_SECRET", "")


class TokenCreateRequest(BaseModel):
    name: Optional[str] = Field(None, description="Optional label for the token")


class TokenCreateResponse(BaseModel):
    id: int
    token: str
    prefix: str
    name: Optional[str]
    created_at: str


class TokenResponse(BaseModel):
    id: int
    prefix: str
    name: Optional[str]
    created_at: str
    last_used_at: Optional[str]
    is_active: bool


@router.post("/tokens", response_model=TokenCreateResponse)
async def create_token(
    body: TokenCreateRequest = TokenCreateRequest(),
    user: str = Depends(auth_required),
):
    """Create a new API token. The plaintext token is only returned once."""
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    db = get_llm_config_db()
    token_id, plaintext = db.create_api_token(user, body.name)
    tokens = db.list_api_tokens(user)
    created = next(t for t in tokens if t.id == token_id)

    return TokenCreateResponse(
        id=token_id,
        token=plaintext,
        prefix=created.token_prefix,
        name=created.name,
        created_at=created.created_at.isoformat(),
    )


@router.get("/tokens", response_model=list[TokenResponse])
async def list_tokens(user: str = Depends(auth_required)):
    """List all tokens for the current user."""
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    db = get_llm_config_db()
    tokens = db.list_api_tokens(user)
    return [
        TokenResponse(
            id=t.id,
            prefix=t.token_prefix,
            name=t.name,
            created_at=t.created_at.isoformat(),
            last_used_at=t.last_used_at.isoformat() if t.last_used_at else None,
            is_active=t.is_active,
        )
        for t in tokens
    ]


@router.delete("/tokens/{token_id}")
async def revoke_token(token_id: int, user: str = Depends(auth_required)):
    """Revoke a token."""
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    db = get_llm_config_db()
    if not db.revoke_api_token(user, token_id):
        raise HTTPException(status_code=404, detail="Token not found")
    return {"status": "revoked"}


class TokenValidateRequest(BaseModel):
    token: str


@router.post("/tokens/validate")
async def validate_token(body: TokenValidateRequest, request: Request):
    """Validate a token and return the user_id. Internal use only."""
    # only allow internal callers
    is_internal = request.headers.get("X-Internal-MCP-Call") == "true"
    if not is_internal and _internal_api_secret:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            is_internal = hmac.compare_digest(auth_header[7:], _internal_api_secret)
    if not is_internal:
        raise HTTPException(status_code=403, detail="Internal endpoint")

    db = get_llm_config_db()
    user_id = db.validate_api_token(body.token)
    if user_id:
        return {"valid": True, "user_id": user_id}
    return {"valid": False}
