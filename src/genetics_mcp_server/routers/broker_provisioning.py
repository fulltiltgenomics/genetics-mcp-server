"""POST /provision/{alias}: a partner app provisions one of its users for the jwt-bearer grant.

See broker_provisioning.py for what it does and why the caller is the partner's own client.
"""

import logging

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from genetics_mcp_server import broker_provisioning as svc
from genetics_mcp_server.auth.core import _matches_allow_list
from genetics_mcp_server.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter()


class ProvisionRequest(BaseModel):
    email: str


class ProvisionResponse(BaseModel):
    email: str
    created: bool
    newly_linked: bool


@router.post("/provision/{alias}", response_model=ProvisionResponse)
async def provision(alias: str, body: ProvisionRequest, request: Request) -> ProvisionResponse:
    settings = get_settings()
    if alias not in settings.broker_provisioning_clients or not svc.configured(settings):
        raise HTTPException(status_code=404, detail="Not found")

    auth = request.headers.get("Authorization", "")
    scheme, _, token = auth.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(
            status_code=401, detail="Bearer token required", headers={"WWW-Authenticate": "Bearer"}
        )
    if await svc.verify_caller(token.strip(), settings) != alias:
        # a token of another client, a user token of this one, or no valid token at all
        raise HTTPException(status_code=401, detail="Not the partner's service account")

    email = svc.normalise_email(body.email)
    if email is None:
        raise HTTPException(status_code=400, detail="Not an e-mail address")
    # allow_wildcard=False as on every path no gateway fronts (see auth.core._matches_allow_list)
    if not _matches_allow_list(email, settings, allow_wildcard=False):
        raise HTTPException(status_code=403, detail="Address is not allow-listed")

    try:
        result = await svc.provisioner(settings).ensure_linked(email, alias)
    except svc.AccountDisabled:
        raise HTTPException(status_code=409, detail="Account is disabled")
    except httpx.HTTPError as e:
        logger.warning("provisioning %s for %s failed against Keycloak: %s", email, alias, e)
        raise HTTPException(status_code=502, detail="Identity broker unavailable")
    return ProvisionResponse(**result.__dict__)
