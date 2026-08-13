"""Mint the per-execution, audience-scoped tokens the sandbox uses to reach db-api and
results-api.

Design of record: ``docs/code-execution-security.md`` §4 in genetics-results-suite.

The sandbox must never hold ``INTERNAL_API_SECRET``: that secret authenticates the *service*,
never expires, and a script that reads it can reach both backends forever. Instead chat-backend
— the only caller of the sandbox, and the only component that knows the end user and the chat
session — mints one short-lived HS256 JWT *per audience per execution* and ships them in the
body of the POST that submits the script.

Minting is fail-closed: with no signing key configured there is no token, and the caller must
surface that rather than fall back to a service credential.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

import jwt

from .config.settings import get_settings

ISSUER = "chat-backend"

DB_API_AUDIENCE = "db-api"
RESULTS_API_AUDIENCE = "results-api"
AUDIENCES = (DB_API_AUDIENCE, RESULTS_API_AUDIENCE)

ALGORITHM = "HS256"

# exp = iat + 300. The hard wall clock on an execution is 120s; the slack covers a slow
# BigQuery job started at the last moment, and leaves room for clock skew between pods.
TTL_SECONDS = 300

# coarse capability string. v1 validators check only that it is present — it is a hook for
# later per-view narrowing and enforces nothing today.
SCOPE = "query:views"


class SandboxTokenUnavailable(RuntimeError):
    """Raised when SANDBOX_TOKEN_SIGNING_KEY is unset, so no token can be minted.

    Deliberately an error rather than a ``None`` return: every fallback from "no sandbox
    token" is either "send no credential" or "send the shared secret", and both are the
    failure modes this whole mechanism exists to prevent.
    """


@dataclass(frozen=True)
class SandboxTokens:
    """The credentials for one ``run_analysis`` execution."""

    execution_id: str
    """uuid4, also the ``/scratch/<execution-id>`` directory name and the ``jti`` of both
    tokens, so logs join across chat-backend, the sandbox SDK and db-api."""

    session_id: str
    user: str
    expires_at: int
    tokens: dict[str, str]
    """audience -> encoded JWT."""

    @property
    def db_api(self) -> str:
        return self.tokens[DB_API_AUDIENCE]

    @property
    def results_api(self) -> str:
        return self.tokens[RESULTS_API_AUDIENCE]


def _signing_key() -> str:
    key = get_settings().sandbox_token_signing_key
    if not key:
        raise SandboxTokenUnavailable(
            "SANDBOX_TOKEN_SIGNING_KEY is not set: refusing to run sandboxed code without a "
            "scoped credential"
        )
    return key


def mint_sandbox_token(
    *,
    audience: str,
    user: str,
    session_id: str,
    execution_id: str,
    issued_at: int | None = None,
    ttl_seconds: int = TTL_SECONDS,
) -> str:
    """Mint one audience-bound token. Every claim here is required by both validators."""
    if audience not in AUDIENCES:
        raise ValueError(f"unknown sandbox token audience: {audience!r}")
    if not user or not session_id or not execution_id:
        raise ValueError("sandbox tokens require user, session_id and execution_id")

    iat = int(time.time()) if issued_at is None else int(issued_at)
    claims = {
        "iss": ISSUER,
        "aud": audience,
        "sub": user,
        "sid": session_id,
        "jti": execution_id,
        "iat": iat,
        "exp": iat + ttl_seconds,
        "scope": SCOPE,
    }
    return jwt.encode(claims, _signing_key(), algorithm=ALGORITHM)


def mint_execution_tokens(
    *,
    user: str,
    session_id: str,
    execution_id: str | None = None,
    ttl_seconds: int = TTL_SECONDS,
) -> SandboxTokens:
    """Mint the pair of tokens for one execution — the entry point `4h6.14` calls.

    One token per audience so a token captured from a results-api request cannot be replayed
    at db-api. Both carry the same ``jti``.
    """
    execution_id = execution_id or str(uuid.uuid4())
    issued_at = int(time.time())
    tokens = {
        audience: mint_sandbox_token(
            audience=audience,
            user=user,
            session_id=session_id,
            execution_id=execution_id,
            issued_at=issued_at,
            ttl_seconds=ttl_seconds,
        )
        for audience in AUDIENCES
    }
    return SandboxTokens(
        execution_id=execution_id,
        session_id=session_id,
        user=user,
        expires_at=issued_at + ttl_seconds,
        tokens=tokens,
    )
