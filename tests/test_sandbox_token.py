"""Tests for minting the per-execution sandbox tokens.

Design of record: docs/code-execution-security.md section 4 in genetics-results-suite. The
contract these tests pin is what `4h6.14` consumes: two audience-bound tokens per execution,
sharing one `jti` that is also the `/scratch/<execution-id>` directory name, a 5-minute
lifetime, and a hard failure — never a fallback to a service credential — when no signing key
is configured.
"""

import time

import jwt
import pytest

from genetics_mcp_server import sandbox_token
from genetics_mcp_server.config import settings as settings_module

SIGNING_KEY = "test-sandbox-signing-key-that-is-32-bytes+"


@pytest.fixture(autouse=True)
def signing_key(monkeypatch):
    settings_module.get_settings.cache_clear()
    monkeypatch.setenv("SANDBOX_TOKEN_SIGNING_KEY", SIGNING_KEY)
    yield
    settings_module.get_settings.cache_clear()


def _decode(token, audience):
    return jwt.decode(
        token,
        SIGNING_KEY,
        algorithms=["HS256"],
        audience=audience,
        issuer="chat-backend",
        options={"require": ["iss", "aud", "sub", "sid", "jti", "iat", "exp"]},
    )


def test_mints_one_token_per_audience():
    """A token captured from a results-api request must not be replayable at db-api."""
    minted = sandbox_token.mint_execution_tokens(user="u@finngen.fi", session_id="s1")
    assert set(minted.tokens) == {"db-api", "results-api"}
    assert _decode(minted.db_api, "db-api")["aud"] == "db-api"
    assert _decode(minted.results_api, "results-api")["aud"] == "results-api"


def test_the_pair_shares_one_execution_id():
    """`jti` is the execution id and the /scratch directory name — it joins the log streams."""
    minted = sandbox_token.mint_execution_tokens(user="u@finngen.fi", session_id="s1")
    assert (
        _decode(minted.db_api, "db-api")["jti"]
        == _decode(minted.results_api, "results-api")["jti"]
        == minted.execution_id
    )


def test_claims_carry_the_user_the_session_and_the_scope():
    minted = sandbox_token.mint_execution_tokens(user="u@finngen.fi", session_id="conv-9")
    claims = _decode(minted.db_api, "db-api")
    assert claims["iss"] == "chat-backend"
    assert claims["sub"] == "u@finngen.fi"
    assert claims["sid"] == "conv-9"
    assert claims["scope"] == "query:views"


def test_lifetime_is_five_minutes():
    claims = _decode(
        sandbox_token.mint_execution_tokens(user="u@finngen.fi", session_id="s1").db_api,
        "db-api",
    )
    assert claims["exp"] - claims["iat"] == 300
    assert abs(claims["iat"] - int(time.time())) <= 5


def test_algorithm_is_hs256_so_the_validators_route_it_correctly():
    """The validators discriminate on the JOSE alg header, not on dot count."""
    token = sandbox_token.mint_execution_tokens(user="u@finngen.fi", session_id="s1").db_api
    assert jwt.get_unverified_header(token)["alg"] == "HS256"


def test_a_token_for_one_audience_does_not_validate_for_the_other():
    minted = sandbox_token.mint_execution_tokens(user="u@finngen.fi", session_id="s1")
    with pytest.raises(jwt.InvalidAudienceError):
        _decode(minted.results_api, "db-api")


def test_minting_fails_closed_without_a_signing_key(monkeypatch):
    """No fallback exists that is not 'send no credential' or 'send the shared secret'."""
    monkeypatch.delenv("SANDBOX_TOKEN_SIGNING_KEY", raising=False)
    settings_module.get_settings.cache_clear()
    with pytest.raises(sandbox_token.SandboxTokenUnavailable):
        sandbox_token.mint_execution_tokens(user="u@finngen.fi", session_id="s1")


def test_unknown_audience_is_refused():
    with pytest.raises(ValueError):
        sandbox_token.mint_sandbox_token(
            audience="bigquery", user="u@finngen.fi", session_id="s1", execution_id="e1"
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"user": "", "session_id": "s1"},
        {"user": "u@finngen.fi", "session_id": ""},
    ],
)
def test_attribution_claims_cannot_be_empty(kwargs):
    """A token with no user or no session is unattributable, which defeats the design."""
    with pytest.raises(ValueError):
        sandbox_token.mint_execution_tokens(**kwargs)


def test_execution_ids_are_unique_per_execution():
    a = sandbox_token.mint_execution_tokens(user="u@finngen.fi", session_id="s1")
    b = sandbox_token.mint_execution_tokens(user="u@finngen.fi", session_id="s1")
    assert a.execution_id != b.execution_id
