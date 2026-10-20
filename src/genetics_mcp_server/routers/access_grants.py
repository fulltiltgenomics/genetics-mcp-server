"""Admin endpoints for runtime access grants, plus the internal check other services call.

See access_grants.py for how a grant reaches the three gates.
"""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from genetics_mcp_server import access_grants as svc
from genetics_mcp_server.auth import IDENTITY_HEADER, admin_required, is_internal_caller
from genetics_mcp_server.config import get_settings
from genetics_mcp_server.db import get_llm_config_db
from genetics_mcp_server.db.llm_config_db import AccessGrant

logger = logging.getLogger(__name__)

router = APIRouter()


class AccessGrantItem(BaseModel):
    email: str
    granted_by: str
    granted_at: str
    expires_at: str
    revoked_at: str | None
    email_sent_at: str | None
    email_error: str | None
    active: bool


class GrantRequest(BaseModel):
    email: str
    # the admin has seen the "does not look like a Google, Apple or Microsoft address" warning
    confirm_unknown_provider: bool = False


class GrantResponse(BaseModel):
    granted: bool
    already_active: bool = False
    provider: str
    grant: AccessGrantItem | None = None
    email_sent: bool | None = None
    email_error: str | None = None


class EmailResponse(BaseModel):
    email_sent: bool
    email_error: str | None


def _item(grant: AccessGrant, now: datetime) -> AccessGrantItem:
    def iso(value: datetime | None) -> str | None:
        return value.isoformat() if value else None

    return AccessGrantItem(
        email=grant.email,
        granted_by=grant.granted_by,
        granted_at=grant.granted_at.isoformat(),
        expires_at=grant.expires_at.isoformat(),
        revoked_at=iso(grant.revoked_at),
        email_sent_at=iso(grant.email_sent_at),
        email_error=grant.email_error,
        active=grant.is_active(now),
    )


def _feature_on() -> None:
    if not get_settings().access_grants_enabled:
        raise HTTPException(status_code=404, detail="Not found")


def _kick() -> None:
    reconciler = svc.get_reconciler()
    if reconciler is not None:
        reconciler.kick()


@router.get("/admin/access-grants", response_model=list[AccessGrantItem])
async def list_grants(user: str = Depends(admin_required)):
    _feature_on()
    now = datetime.now(timezone.utc)
    return [_item(g, now) for g in get_llm_config_db().list_access_grants()]


@router.post("/admin/access-grants", response_model=GrantResponse)
async def grant(body: GrantRequest, user: str = Depends(admin_required)):
    _feature_on()
    email = svc.normalize_email(body.email)
    if not svc.is_valid_email(email):
        raise HTTPException(status_code=400, detail="Not a valid e-mail address")
    expires_at = svc.parse_expires_at(get_settings().access_grant_expires_at)
    if expires_at is None:
        raise HTTPException(
            status_code=500, detail="ACCESS_GRANT_EXPIRES_AT is not configured on this deployment"
        )
    now = datetime.now(timezone.utc)
    if expires_at <= now:
        raise HTTPException(
            status_code=400, detail=f"The configured access deadline {expires_at.isoformat()} has passed"
        )

    db = get_llm_config_db()
    existing = db.get_access_grant(email)
    if existing is not None and existing.is_active(now):
        # idempotent: a second click must not mail the person twice
        return GrantResponse(
            granted=True,
            already_active=True,
            provider=await svc.classify_provider(email),
            grant=_item(existing, now),
            email_sent=existing.email_sent_at is not None,
            email_error=existing.email_error,
        )

    provider = await svc.classify_provider(email)
    if provider == "unknown" and not body.confirm_unknown_provider:
        return GrantResponse(granted=False, provider=provider)

    created = db.upsert_access_grant(email, granted_by=user, expires_at=expires_at)
    logger.info("access granted to %s by %s until %s", email, user, expires_at.isoformat())
    _kick()
    error = await svc.notify(email)
    refreshed = db.get_access_grant(email) or created
    return GrantResponse(
        granted=True,
        provider=provider,
        grant=_item(refreshed, now),
        email_sent=error is None,
        email_error=error,
    )


@router.delete("/admin/access-grants/{email}")
async def revoke(email: str, user: str = Depends(admin_required)):
    _feature_on()
    if not get_llm_config_db().revoke_access_grant(email):
        raise HTTPException(status_code=404, detail="No active grant for that address")
    logger.info("access grant for %s revoked by %s", svc.normalize_email(email), user)
    _kick()
    return {"status": "revoked"}


@router.post("/admin/access-grants/{email}/email", response_model=EmailResponse)
async def resend(email: str, user: str = Depends(admin_required)):
    _feature_on()
    db = get_llm_config_db()
    grant = db.get_access_grant(email)
    if grant is None or not grant.is_active(datetime.now(timezone.utc)):
        raise HTTPException(status_code=404, detail="No active grant for that address")
    error = await svc.notify(grant.email)
    return EmailResponse(email_sent=error is None, email_error=error)


@router.get("/access-grants/check")
async def check(email: str, request: Request):
    """Whether the address holds an active grant. Internal use only: results-api and
    mcp-server call it after their own static allow-list misses."""
    # same caller rule as /tokens/validate: the marker, and no proxied identity on the request
    if not is_internal_caller(request) or request.headers.get(IDENTITY_HEADER):
        raise HTTPException(status_code=403, detail="Internal endpoint")
    return {"allowed": get_llm_config_db().has_active_access_grant(email)}
