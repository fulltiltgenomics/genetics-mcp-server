"""POST /provision/{alias}: a partner's service account creates and links one of its users for the
jwt-bearer grant (broker_provisioning.py, routers/broker_provisioning.py)."""

import json
import time
from unittest.mock import patch

import httpx
import jwt
import pytest
from conftest import settings_env
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from genetics_mcp_server import broker_provisioning as svc
from genetics_mcp_server.chat_api import app
from genetics_mcp_server.config import get_settings

ISSUER = "https://host/auth/realms/genetics"
ALIAS = "brainzzz"
EMAIL = "someone@broadinstitute.org"

CONFIGURED = dict(
    REQUIRE_AUTH="true",
    INTERNAL_API_SECRET="test-internal-secret",
    ALLOWED_EMAIL_DOMAINS="broadinstitute.org",
    BROKER_PROVISIONING_CLIENTS=ALIAS,
    KEYCLOAK_ADMIN_URL="http://kc",
    KEYCLOAK_ACCESS_GRANTS_CLIENT_SECRET="s3cret",
    OAUTH_ISSUER=ISSUER,
)


@pytest.fixture
def client(llm_config_db, chat_history_db):
    with settings_env(**CONFIGURED), TestClient(app) as c:
        yield c


def _as(caller):
    """Stand in for the token check: the bearer token's text names the caller it verifies as."""

    async def verify(token, settings):
        return caller

    return patch.object(svc, "verify_caller", verify)


class FakeKeycloak:
    """The admin API calls ensure_linked makes, recorded; `users` is what the realm holds."""

    def __init__(self, users=None, links=None):
        self.users = users or []
        self.links = links or []
        self.calls = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.calls.append((request.method, path))
        if path.endswith("/protocol/openid-connect/token"):
            return httpx.Response(200, json={"access_token": "admin-token"})
        if path == "/admin/realms/genetics/users" and request.method == "GET":
            return httpx.Response(200, json=self.users)
        if path == "/admin/realms/genetics/users" and request.method == "POST":
            body = json.loads(request.content)
            assert body == {
                "username": EMAIL,
                "email": EMAIL,
                "enabled": True,
                "emailVerified": True,
            }
            return httpx.Response(
                201, headers={"Location": "http://kc/admin/realms/genetics/users/new-id"}
            )
        if path.endswith("/federated-identity") and request.method == "GET":
            return httpx.Response(200, json=self.links)
        if path.endswith(f"/federated-identity/{ALIAS}") and request.method == "POST":
            assert json.loads(request.content) == {
                "identityProvider": ALIAS,
                "userId": EMAIL,
                "userName": EMAIL,
            }
            return httpx.Response(204)
        return httpx.Response(500, text=f"unexpected {request.method} {path}")

    def patched(self):
        real = httpx.AsyncClient

        def mocked(**kw):
            return real(transport=httpx.MockTransport(self.handler), **kw)

        return patch.object(svc.httpx, "AsyncClient", mocked)


def test_404_when_no_partner_is_configured(llm_config_db, chat_history_db):
    with settings_env(**{**CONFIGURED, "BROKER_PROVISIONING_CLIENTS": ""}), TestClient(app) as c:
        assert c.post(f"/provision/{ALIAS}", json={"email": EMAIL}).status_code == 404


def test_404_for_an_alias_not_on_the_list(client):
    with _as("other"):
        r = client.post(
            "/provision/other", json={"email": EMAIL}, headers={"Authorization": "Bearer t"}
        )
    assert r.status_code == 404


def test_401_without_a_bearer_token(client):
    r = client.post(f"/provision/{ALIAS}", json={"email": EMAIL})
    assert r.status_code == 401
    assert r.headers["WWW-Authenticate"] == "Bearer"


def test_401_when_the_token_is_not_the_partners_service_account(client):
    # another client's service account, or a user token obtained through the grant
    for caller in ("other-client", None):
        with _as(caller):
            r = client.post(
                f"/provision/{ALIAS}", json={"email": EMAIL}, headers={"Authorization": "Bearer t"}
            )
        assert r.status_code == 401, caller


def test_403_outside_the_allow_list(client):
    with _as(ALIAS):
        r = client.post(
            f"/provision/{ALIAS}",
            json={"email": "outsider@example.com"},
            headers={"Authorization": "Bearer t"},
        )
    assert r.status_code == 403


def test_400_for_something_that_is_not_an_address(client):
    with _as(ALIAS):
        r = client.post(
            f"/provision/{ALIAS}",
            json={"email": "not an address"},
            headers={"Authorization": "Bearer t"},
        )
    assert r.status_code == 400


def test_creates_and_links_a_new_user(client):
    kc = FakeKeycloak()
    with _as(ALIAS), kc.patched():
        r = client.post(
            f"/provision/{ALIAS}",
            json={"email": " Someone@BroadInstitute.org "},
            headers={"Authorization": "Bearer t"},
        )
    assert r.status_code == 200
    assert r.json() == {"email": EMAIL, "created": True, "newly_linked": True}
    assert ("POST", "/admin/realms/genetics/users") in kc.calls
    assert ("POST", f"/admin/realms/genetics/users/new-id/federated-identity/{ALIAS}") in kc.calls


def test_an_existing_unlinked_user_is_only_linked(client):
    kc = FakeKeycloak(users=[{"id": "u1", "email": EMAIL, "enabled": True}])
    with _as(ALIAS), kc.patched():
        r = client.post(
            f"/provision/{ALIAS}", json={"email": EMAIL}, headers={"Authorization": "Bearer t"}
        )
    assert r.json() == {"email": EMAIL, "created": False, "newly_linked": True}
    assert ("POST", "/admin/realms/genetics/users") not in kc.calls
    assert ("POST", f"/admin/realms/genetics/users/u1/federated-identity/{ALIAS}") in kc.calls


def test_a_linked_user_is_a_no_op(client):
    kc = FakeKeycloak(
        users=[{"id": "u1", "email": EMAIL, "enabled": True}],
        links=[{"identityProvider": ALIAS, "userId": EMAIL}],
    )
    with _as(ALIAS), kc.patched():
        r = client.post(
            f"/provision/{ALIAS}", json={"email": EMAIL}, headers={"Authorization": "Bearer t"}
        )
    assert r.json() == {"email": EMAIL, "created": False, "newly_linked": False}
    assert not any(m == "POST" and "users" in p for m, p in kc.calls if "token" not in p)


def test_a_disabled_account_is_409_and_stays_unlinked(client):
    kc = FakeKeycloak(users=[{"id": "u1", "email": EMAIL, "enabled": False}])
    with _as(ALIAS), kc.patched():
        r = client.post(
            f"/provision/{ALIAS}", json={"email": EMAIL}, headers={"Authorization": "Bearer t"}
        )
    assert r.status_code == 409
    assert not any(p.endswith(f"/federated-identity/{ALIAS}") for _, p in kc.calls)


def test_502_when_keycloak_fails(client):
    def handler(request):
        return httpx.Response(503, text="down")

    real = httpx.AsyncClient
    with (
        _as(ALIAS),
        patch.object(
            svc.httpx,
            "AsyncClient",
            lambda **kw: real(transport=httpx.MockTransport(handler), **kw),
        ),
    ):
        r = client.post(
            f"/provision/{ALIAS}", json={"email": EMAIL}, headers={"Authorization": "Bearer t"}
        )
    assert r.status_code == 502


# --- the token check itself, with a real signature ------------------------------------------------


@pytest.fixture
def keypair():
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = private.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    return pem, private.public_key()


def _token(pem, **claims):
    now = int(time.time())
    base = {"iss": ISSUER, "exp": now + 60, "iat": now, "aud": "https://host/mcp"}
    return jwt.encode({**base, **claims}, pem, algorithm="RS256")


def test_caller_client_id_accepts_the_service_account_and_nothing_else(keypair):
    pem, public = keypair
    with settings_env(**CONFIGURED), patch.object(svc, "_signing_key", lambda uri, tok: public):
        settings = get_settings()
        ok = _token(pem, azp=ALIAS, preferred_username=f"service-account-{ALIAS}")
        assert svc.caller_client_id(ok, settings) == ALIAS
        # a person's token from the same client: the grant issues these with azp=brainzzz too
        user = _token(pem, azp=ALIAS, preferred_username=EMAIL, email=EMAIL)
        assert svc.caller_client_id(user, settings) is None
        other_issuer = _token(
            pem,
            iss="https://elsewhere/realms/x",
            azp=ALIAS,
            preferred_username=f"service-account-{ALIAS}",
        )
        assert svc.caller_client_id(other_issuer, settings) is None
        expired = _token(
            pem, exp=int(time.time()) - 5, azp=ALIAS, preferred_username=f"service-account-{ALIAS}"
        )
        assert svc.caller_client_id(expired, settings) is None


def test_caller_client_id_refuses_a_token_signed_by_another_key(keypair):
    pem, _ = keypair
    other_public = rsa.generate_private_key(public_exponent=65537, key_size=2048).public_key()
    with (
        settings_env(**CONFIGURED),
        patch.object(svc, "_signing_key", lambda uri, tok: other_public),
    ):
        tok = _token(pem, azp=ALIAS, preferred_username=f"service-account-{ALIAS}")
        assert svc.caller_client_id(tok, get_settings()) is None


def test_jwks_uri_prefers_the_in_cluster_keycloak():
    with settings_env(**CONFIGURED):
        assert (
            svc._jwks_uri(get_settings())
            == "http://kc/realms/genetics/protocol/openid-connect/certs"
        )
    with settings_env(**{**CONFIGURED, "KEYCLOAK_ADMIN_URL": ""}):
        assert svc._jwks_uri(get_settings()) == f"{ISSUER}/protocol/openid-connect/certs"
