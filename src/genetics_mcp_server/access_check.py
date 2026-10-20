"""Ask chat-backend whether an address holds a runtime access grant.

mcp-server has no copy of the grant table — the config database is chat-backend's PVC — so
after its own static allow-list misses it asks /chat/v1/access-grants/check, the way it
already asks /tokens/validate. Answers are cached briefly in both directions: the cost is a
grant or revocation taking up to CACHE_TTL_S to be seen here, which is the same order as the
ConfigMap propagation oauth2-proxy lives with. Any failure answers False.
"""

import logging
import os
import time

import httpx

from genetics_mcp_server.config.settings import get_settings

logger = logging.getLogger(__name__)

CACHE_TTL_S = 60.0
_cache: dict[str, tuple[bool, float]] = {}


def has_access_grant(email: str) -> bool:
    email = email.strip().lower()
    now = time.monotonic()
    cached = _cache.get(email)
    if cached and cached[1] > now:
        return cached[0]

    chat_url = os.environ.get("CHAT_BACKEND_URL", "")
    if not chat_url:
        return False
    allowed = False
    try:
        secret = get_settings().internal_api_secret
        headers = {"Authorization": f"Bearer {secret}"} if secret else {}
        resp = httpx.get(
            f"{chat_url}/v1/access-grants/check",
            params={"email": email},
            headers=headers,
            timeout=5.0,
        )
        if resp.status_code == 200:
            allowed = bool(resp.json().get("allowed"))
        else:
            logger.warning("access-grant check answered %s", resp.status_code)
    except Exception as exc:
        logger.warning("access-grant check failed: %s", exc)
    _cache[email] = (allowed, now + CACHE_TTL_S)
    return allowed


def clear_cache() -> None:
    _cache.clear()
