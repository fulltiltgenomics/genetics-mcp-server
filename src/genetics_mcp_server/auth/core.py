"""Authentication via X-Goog-Authenticated-User-Email header (set by IAP or oauth2-proxy)."""

import logging

from fastapi import Request

logger = logging.getLogger(__name__)


def get_authenticated_user(request: Request) -> str | None:
    """Extract authenticated user email from the IAP/oauth2-proxy header."""
    iap_email = request.headers.get("X-Goog-Authenticated-User-Email")
    if not iap_email:
        return None
    # header format: "accounts.google.com:user@domain.com"
    return iap_email.split(":")[-1] if ":" in iap_email else iap_email
