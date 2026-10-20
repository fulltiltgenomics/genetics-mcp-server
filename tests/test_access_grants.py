"""Runtime access grants: the table, the auth paths that consult it, the admin API and the
reconciler that pushes the table out to oauth2-proxy and Keycloak (access_grants.py)."""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import httpx
import pytest
from conftest import settings_env
from fastapi.testclient import TestClient

from genetics_mcp_server import access_grants as svc
from genetics_mcp_server.auth import core
from genetics_mcp_server.chat_api import app
from genetics_mcp_server.db import llm_config_db as db_module

INTERNAL_SECRET = "test-internal-secret"
ADMIN = "admin@finngen.fi"
GUEST = "visitor@gmail.com"
FUTURE = "2099-01-01T00:00:00+00:00"

NOW = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)
LATER = NOW + timedelta(days=4)


@pytest.fixture
def grant_db(llm_config_db, monkeypatch):
    """Route get_llm_config_db() at the temporary database for everything under test."""
    monkeypatch.setattr(db_module, "_llm_config_db", llm_config_db)
    yield llm_config_db
    svc.reset_reconciler()


def _admin_headers(user: str = ADMIN) -> dict[str, str]:
    return {
        core.INTERNAL_MARKER_HEADER: INTERNAL_SECRET,
        core.IDENTITY_HEADER: f"accounts.google.com:{user}",
    }


@pytest.fixture
def deployed(grant_db):
    """The daly shape: auth on, an admin, a static allow-list and the feature configured."""
    with settings_env(
        REQUIRE_AUTH="true",
        INTERNAL_API_SECRET=INTERNAL_SECRET,
        ENABLE_ADMIN_PAGE="true",
        ADMIN_USERS=ADMIN,
        ALLOWED_EMAIL_DOMAINS="finngen.fi",
        ALLOWED_EMAILS="apple.user@privaterelay.appleid.com",
        ACCESS_GRANT_BUTTON_LABEL="ASHG special",
        ACCESS_GRANT_EXPIRES_AT=FUTURE,
    ):
        yield grant_db


@pytest.fixture
def client(deployed):
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# the table
# ---------------------------------------------------------------------------


class TestAccessGrantTable:
    def test_grant_is_active_until_its_deadline(self, grant_db):
        grant = grant_db.upsert_access_grant(" Visitor@Gmail.com ", ADMIN, LATER)
        assert grant.email == "visitor@gmail.com"
        assert grant.is_active(NOW)
        assert not grant.is_active(LATER)
        assert grant_db.active_access_grant_emails(NOW) == {"visitor@gmail.com"}
        assert grant_db.active_access_grant_emails(LATER) == set()

    def test_revoke_ends_it_and_is_reported_once(self, grant_db):
        grant_db.upsert_access_grant(GUEST, ADMIN, LATER)
        assert grant_db.revoke_access_grant(GUEST) is True
        assert grant_db.revoke_access_grant(GUEST) is False
        assert grant_db.has_active_access_grant(GUEST, NOW) is False

    def test_regrant_rearms_a_revoked_or_deactivated_row(self, grant_db):
        grant_db.upsert_access_grant(GUEST, ADMIN, LATER)
        grant_db.record_access_grant_email(GUEST, "smtp down")
        grant_db.revoke_access_grant(GUEST)
        grant_db.mark_access_grant_deactivated(GUEST)
        again = grant_db.upsert_access_grant(GUEST, "other@finngen.fi", LATER + timedelta(days=1))
        assert again.revoked_at is None
        assert again.deactivated_at is None
        assert again.email_error is None and again.email_sent_at is None
        assert again.granted_by == "other@finngen.fi"
        assert len(grant_db.list_access_grants()) == 1

    def test_email_outcome_is_recorded_either_way(self, grant_db):
        grant_db.upsert_access_grant(GUEST, ADMIN, LATER)
        grant_db.record_access_grant_email(GUEST, "boom")
        g = grant_db.get_access_grant(GUEST)
        assert g.email_error == "boom" and g.email_sent_at is None
        grant_db.record_access_grant_email(GUEST, None)
        g = grant_db.get_access_grant(GUEST)
        assert g.email_error is None and g.email_sent_at is not None


# ---------------------------------------------------------------------------
# the auth paths
# ---------------------------------------------------------------------------


class TestAllowListConsultsGrants:
    def test_static_miss_with_active_grant_is_allowed(self, deployed):
        assert core._email_allowed(GUEST) is False
        deployed.upsert_access_grant(GUEST, ADMIN, datetime.now(timezone.utc) + timedelta(hours=1))
        assert core._email_allowed(GUEST) is True

    def test_expired_or_revoked_grant_does_not_admit(self, deployed):
        deployed.upsert_access_grant(GUEST, ADMIN, datetime.now(timezone.utc) - timedelta(seconds=1))
        assert core._email_allowed(GUEST) is False
        deployed.upsert_access_grant(GUEST, ADMIN, datetime.now(timezone.utc) + timedelta(hours=1))
        deployed.revoke_access_grant(GUEST)
        assert core._email_allowed(GUEST) is False

    def test_static_users_never_touch_the_table(self, deployed):
        with patch.object(db_module, "_llm_config_db", None), patch.object(
            db_module, "LLMConfigDB", side_effect=AssertionError("table consulted")
        ):
            assert core._email_allowed("anyone@finngen.fi") is True

    def test_unreadable_table_answers_not_granted(self, deployed):
        with patch.object(deployed, "has_active_access_grant", side_effect=RuntimeError("locked")):
            assert core.has_access_grant(GUEST) is False

    def test_proxied_identity_with_grant_resolves(self, client, deployed):
        deployed.upsert_access_grant(GUEST, ADMIN, datetime.now(timezone.utc) + timedelta(hours=1))
        resp = client.get("/chat/v1/me", headers=_admin_headers(GUEST))
        assert resp.status_code == 200 and resp.json()["user"] == GUEST
        deployed.revoke_access_grant(GUEST)
        assert client.get("/chat/v1/me", headers=_admin_headers(GUEST)).status_code == 401


class TestApiTokensFollowTheGrant:
    def _validate(self, client, token):
        return client.post(
            "/chat/v1/tokens/validate",
            json={"token": token},
            headers={"Authorization": f"Bearer {INTERNAL_SECRET}"},
        ).json()

    def test_token_dies_with_the_grant_and_revives_with_it(self, client, deployed):
        deployed.upsert_access_grant(GUEST, ADMIN, datetime.now(timezone.utc) + timedelta(hours=1))
        _, token = deployed.create_api_token(GUEST, "conference")
        assert self._validate(client, token) == {"valid": True, "user_id": GUEST}
        deployed.revoke_access_grant(GUEST)
        assert self._validate(client, token) == {"valid": False}
        deployed.upsert_access_grant(GUEST, ADMIN, datetime.now(timezone.utc) + timedelta(hours=1))
        assert self._validate(client, token)["valid"] is True

    def test_static_users_tokens_are_unaffected(self, client, deployed):
        _, token = deployed.create_api_token("someone@finngen.fi")
        assert self._validate(client, token)["valid"] is True


# ---------------------------------------------------------------------------
# the admin API
# ---------------------------------------------------------------------------


class TestAdminApi:
    def test_auth_advertises_the_button_to_admins_only(self, client):
        admin = client.get("/chat/v1/auth", headers=_admin_headers()).json()
        assert admin["access_grants"] == {"label": "ASHG special", "expires_at": FUTURE}
        user = client.get("/chat/v1/auth", headers=_admin_headers("someone@finngen.fi")).json()
        assert user["is_admin"] is False and user["access_grants"] is None

    def test_button_absent_when_unconfigured(self, grant_db):
        with settings_env(
            REQUIRE_AUTH="true", INTERNAL_API_SECRET=INTERNAL_SECRET, ENABLE_ADMIN_PAGE="true",
            ADMIN_USERS=ADMIN, ALLOWED_EMAIL_DOMAINS="finngen.fi", ACCESS_GRANT_BUTTON_LABEL="",
        ):
            with TestClient(app) as c:
                assert c.get("/chat/v1/auth", headers=_admin_headers()).json()["access_grants"] is None
                assert c.get("/chat/v1/admin/access-grants", headers=_admin_headers()).status_code == 404

    def test_non_admin_is_refused(self, client):
        resp = client.get("/chat/v1/admin/access-grants", headers=_admin_headers("someone@finngen.fi"))
        assert resp.status_code == 403

    def test_grant_known_provider_records_and_mails(self, client, deployed):
        with patch.object(svc, "classify_provider", return_value="google") as classify, patch.object(
            svc, "notify", return_value=None
        ) as notify:
            resp = client.post(
                "/chat/v1/admin/access-grants", json={"email": " Visitor@Gmail.com "},
                headers=_admin_headers(),
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["granted"] is True and body["provider"] == "google"
        assert body["email_sent"] is True and body["grant"]["email"] == GUEST
        assert body["grant"]["expires_at"] == FUTURE
        assert body["grant"]["granted_by"] == ADMIN
        classify.assert_awaited_once_with(GUEST)
        notify.assert_awaited_once_with(GUEST)
        assert deployed.has_active_access_grant(GUEST)

    def test_unknown_provider_needs_confirmation(self, client, deployed):
        with patch.object(svc, "classify_provider", return_value="unknown"), patch.object(
            svc, "notify", return_value=None
        ) as notify:
            first = client.post(
                "/chat/v1/admin/access-grants", json={"email": "x@uni.example"},
                headers=_admin_headers(),
            ).json()
            assert first == {
                "granted": False, "already_active": False, "provider": "unknown",
                "grant": None, "email_sent": None, "email_error": None,
            }
            assert not deployed.has_active_access_grant("x@uni.example")
            notify.assert_not_awaited()
            second = client.post(
                "/chat/v1/admin/access-grants",
                json={"email": "x@uni.example", "confirm_unknown_provider": True},
                headers=_admin_headers(),
            ).json()
        assert second["granted"] is True
        assert deployed.has_active_access_grant("x@uni.example")

    def test_regrant_of_active_address_does_not_mail_again(self, client, deployed):
        with patch.object(svc, "classify_provider", return_value="google"), patch.object(
            svc, "notify", return_value=None
        ) as notify:
            client.post("/chat/v1/admin/access-grants", json={"email": GUEST}, headers=_admin_headers())
            again = client.post(
                "/chat/v1/admin/access-grants", json={"email": GUEST}, headers=_admin_headers()
            ).json()
        assert again["granted"] is True and again["already_active"] is True
        assert notify.await_count == 1

    def test_mail_failure_is_reported_but_access_stands(self, client, deployed):
        with patch.object(svc, "classify_provider", return_value="google"), patch.object(
            svc, "send_notification", side_effect=OSError("connection refused")
        ), settings_env(
            SMTP_HOST="smtp.example", ACCESS_GRANT_EMAIL_FROM="a@b", ACCESS_GRANT_EMAIL_BODY_FILE="/x",
        ):
            body = client.post(
                "/chat/v1/admin/access-grants", json={"email": GUEST}, headers=_admin_headers()
            ).json()
        assert body["granted"] is True and body["email_sent"] is False
        assert "connection refused" in body["email_error"]
        assert deployed.has_active_access_grant(GUEST)
        assert deployed.get_access_grant(GUEST).email_error == body["email_error"]

    def test_invalid_address_and_passed_deadline(self, client, deployed):
        resp = client.post("/chat/v1/admin/access-grants", json={"email": "nope"}, headers=_admin_headers())
        assert resp.status_code == 400
        with settings_env(ACCESS_GRANT_EXPIRES_AT="2020-01-01T00:00:00+00:00"):
            resp = client.post(
                "/chat/v1/admin/access-grants", json={"email": GUEST}, headers=_admin_headers()
            )
        assert resp.status_code == 400 and "passed" in resp.json()["detail"]

    def test_list_revoke_resend(self, client, deployed):
        with patch.object(svc, "classify_provider", return_value="google"), patch.object(
            svc, "send_notification", side_effect=RuntimeError("smtp down")
        ), settings_env(
            SMTP_HOST="smtp.example", ACCESS_GRANT_EMAIL_FROM="a@b", ACCESS_GRANT_EMAIL_BODY_FILE="/x",
        ):
            client.post("/chat/v1/admin/access-grants", json={"email": GUEST}, headers=_admin_headers())
        listed = client.get("/chat/v1/admin/access-grants", headers=_admin_headers()).json()
        assert [g["email"] for g in listed] == [GUEST]
        assert listed[0]["active"] is True and listed[0]["email_error"] == "RuntimeError: smtp down"

        with patch.object(svc, "notify", return_value=None):
            resend = client.post(
                f"/chat/v1/admin/access-grants/{GUEST}/email", headers=_admin_headers()
            ).json()
        assert resend == {"email_sent": True, "email_error": None}

        assert client.delete(f"/chat/v1/admin/access-grants/{GUEST}", headers=_admin_headers()).json() == {
            "status": "revoked"
        }
        assert client.delete(f"/chat/v1/admin/access-grants/{GUEST}", headers=_admin_headers()).status_code == 404
        assert client.post(f"/chat/v1/admin/access-grants/{GUEST}/email", headers=_admin_headers()).status_code == 404
        assert client.get("/chat/v1/admin/access-grants", headers=_admin_headers()).json()[0]["active"] is False


class TestInternalCheck:
    def test_requires_the_marker_and_no_identity(self, client, deployed):
        deployed.upsert_access_grant(GUEST, ADMIN, datetime.now(timezone.utc) + timedelta(hours=1))
        assert client.get("/chat/v1/access-grants/check", params={"email": GUEST}).status_code == 403
        assert client.get(
            "/chat/v1/access-grants/check", params={"email": GUEST}, headers=_admin_headers()
        ).status_code == 403
        ok = client.get(
            "/chat/v1/access-grants/check", params={"email": GUEST},
            headers={"Authorization": f"Bearer {INTERNAL_SECRET}"},
        )
        assert ok.json() == {"allowed": True}
        other = client.get(
            "/chat/v1/access-grants/check", params={"email": "nobody@gmail.com"},
            headers={"Authorization": f"Bearer {INTERNAL_SECRET}"},
        )
        assert other.json() == {"allowed": False}


# ---------------------------------------------------------------------------
# provider classification and the deadline
# ---------------------------------------------------------------------------


def _doh(mx_hosts: list[str] | None, status: int = 200):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "dns.google"
        if mx_hosts is None:
            return httpx.Response(status)
        answer = [{"data": f"{10 + i} {h}"} for i, h in enumerate(mx_hosts)]
        return httpx.Response(200, json={"Answer": answer})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class TestClassifyProvider:
    async def test_consumer_domains_need_no_lookup(self):
        async with _doh(None, 500) as client:
            assert await svc.classify_provider("A@GMAIL.com", client) == "google"
            assert await svc.classify_provider("b@icloud.com", client) == "apple"
            assert await svc.classify_provider("c@privaterelay.appleid.com", client) == "apple"
            assert await svc.classify_provider("d@outlook.com", client) == "microsoft"
            assert await svc.classify_provider("e@Hotmail.com", client) == "microsoft"

    async def test_mx_decides_for_other_domains(self):
        async with _doh(["aspmx.l.google.com.", "alt1.aspmx.l.google.com."]) as client:
            assert await svc.classify_provider("x@broadinstitute.org", client) == "google-workspace"
        async with _doh(["mx01.mail.icloud.com."]) as client:
            assert await svc.classify_provider("x@custom.example", client) == "apple"
        async with _doh(["hotmail-co-uk.olc.protection.outlook.com."]) as client:
            assert await svc.classify_provider("x@hotmail.co.uk", client) == "microsoft"
        async with _doh(["msft-example.mail.protection.outlook.com."]) as client:
            assert await svc.classify_provider("x@msft.example", client) == "microsoft-365"
        async with _doh(["mx.zoho.eu."]) as client:
            assert await svc.classify_provider("x@zoho.example", client) == "unknown"

    async def test_lookup_failure_is_unknown(self):
        async with _doh(None, 503) as client:
            assert await svc.classify_provider("x@down.example", client) == "unknown"
        async with _doh([]) as client:
            assert await svc.classify_provider("x@nomx.example", client) == "unknown"


class TestDeadline:
    def test_offset_required(self):
        assert svc.parse_expires_at("") is None
        assert svc.parse_expires_at("2026-10-24T23:59:59") is None
        assert svc.parse_expires_at("not a date") is None

    def test_converted_to_utc(self):
        parsed = svc.parse_expires_at("2026-10-24T23:59:59-04:00")
        assert parsed == datetime(2026, 10, 25, 3, 59, 59, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# the reconciler
# ---------------------------------------------------------------------------


class FakeConfigMap:
    def __init__(self):
        self.synced: list[set[str]] = []

    async def sync(self, desired):
        self.synced.append(set(desired))
        return True


class FakeKeycloak:
    def __init__(self, accounts: set[str]):
        self.accounts = accounts
        self.calls: list[tuple[set[str], list[str], list[str]]] = []

    async def sync(self, desired, ended, reactivated):
        self.calls.append((set(desired), list(ended), list(reactivated)))
        return [e for e in ended if e in self.accounts]


class TestReconciler:
    @pytest.fixture
    def reconciler(self, deployed):
        r = svc.AccessGrantReconciler()
        assert r.configmap is None and r.keycloak is None, "no cluster here"
        r.configmap = FakeConfigMap()
        r.keycloak = FakeKeycloak(accounts={"gone@gmail.com"})
        return r

    async def test_desired_is_static_plus_active(self, reconciler, deployed):
        active_until = datetime.now(timezone.utc) + timedelta(hours=1)
        deployed.upsert_access_grant(GUEST, ADMIN, active_until)
        deployed.upsert_access_grant("gone@gmail.com", ADMIN, datetime.now(timezone.utc) - timedelta(hours=1))
        deployed.upsert_access_grant("never@gmail.com", ADMIN, datetime.now(timezone.utc) - timedelta(hours=1))

        result = await reconciler.run_once()

        assert result.desired == {"apple.user@privaterelay.appleid.com", GUEST}
        assert reconciler.configmap.synced == [result.desired]
        desired, ended, reactivated = reconciler.keycloak.calls[0]
        assert set(ended) == {"gone@gmail.com", "never@gmail.com"} and reactivated == [GUEST]
        assert result.keycloak_disabled == ["gone@gmail.com"]
        # both ended rows are marked: the one whose account was disabled, and the one that
        # never had an account — neither is queried again
        assert deployed.get_access_grant("gone@gmail.com").deactivated_at is not None
        assert deployed.get_access_grant("never@gmail.com").deactivated_at is not None
        assert deployed.get_access_grant(GUEST).deactivated_at is None

        await reconciler.run_once()
        assert reconciler.keycloak.calls[1][1] == []

    async def test_keycloak_failure_leaves_ended_rows_for_the_next_pass(self, reconciler, deployed):
        deployed.upsert_access_grant("gone@gmail.com", ADMIN, datetime.now(timezone.utc) - timedelta(hours=1))

        async def boom(*_a, **_k):
            raise httpx.ConnectError("down")

        reconciler.keycloak.sync = boom
        result = await reconciler.run_once()
        assert result.configmap_patched is True
        assert deployed.get_access_grant("gone@gmail.com").deactivated_at is None

    async def test_without_keycloak_target_ended_rows_are_just_marked(self, reconciler, deployed):
        reconciler.keycloak = None
        deployed.upsert_access_grant("gone@gmail.com", ADMIN, datetime.now(timezone.utc) - timedelta(hours=1))
        await reconciler.run_once()
        assert deployed.get_access_grant("gone@gmail.com").deactivated_at is not None


class TestTargets:
    async def test_configmap_patches_only_on_difference(self, tmp_path, monkeypatch):
        sa = tmp_path / "sa"
        sa.mkdir()
        (sa / "token").write_text("tok")
        (sa / "namespace").write_text("genetics")
        (sa / "ca.crt").write_text("")
        monkeypatch.setattr(svc, "SA_DIR", sa)
        monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "kube")
        monkeypatch.setenv("KUBERNETES_SERVICE_PORT", "443")

        state = {"text": "b@x.y\na@x.y\n", "patches": []}

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.headers["Authorization"] == "Bearer tok"
            assert request.url.path == "/api/v1/namespaces/genetics/configmaps/oauth2-allowed-emails"
            if request.method == "GET":
                return httpx.Response(200, json={"data": {"allowed-emails.txt": state["text"]}})
            assert request.headers["Content-Type"] == "application/merge-patch+json"
            state["patches"].append(json.loads(request.content)["data"]["allowed-emails.txt"])
            return httpx.Response(200, json={})

        target = svc.ConfigMapTarget("oauth2-allowed-emails")
        monkeypatch.setattr(
            target, "_client",
            lambda: httpx.AsyncClient(
                base_url="https://kube:443", transport=httpx.MockTransport(handler),
                headers={"Authorization": "Bearer tok"},
            ),
        )
        assert await target.sync({"a@x.y", "b@x.y"}) is False
        assert await target.sync({"a@x.y", "b@x.y", GUEST}) is True
        assert state["patches"] == [f"a@x.y\nb@x.y\n{GUEST}"]

    async def test_keycloak_round_trip(self):
        realm = {"realm": "genetics", "attributes": {"allowedEmails": "a@x.y", "other": "kept"}}
        users = {
            "gone@gmail.com": {"id": "u1", "email": "gone@gmail.com", "enabled": True},
            GUEST: {"id": "u2", "email": GUEST, "enabled": False},
        }
        log: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path.endswith("/protocol/openid-connect/token"):
                assert b"client_credentials" in request.content
                return httpx.Response(200, json={"access_token": "kc-tok"})
            assert request.headers["Authorization"] == "Bearer kc-tok"
            if path == "/admin/realms/genetics":
                if request.method == "GET":
                    return httpx.Response(200, json=realm)
                realm.update(json.loads(request.content))
                log.append("realm-put")
                return httpx.Response(204)
            if path == "/admin/realms/genetics/users":
                email = request.url.params["email"]
                return httpx.Response(200, json=[users[email]] if email in users else [])
            if path.startswith("/admin/realms/genetics/users/"):
                body = json.loads(request.content)
                users[body["email"]] = body
                log.append(f"user-put {body['email']} enabled={body['enabled']}")
                return httpx.Response(204)
            raise AssertionError(path)

        target = svc.KeycloakTarget("http://kc", "genetics", "access-grants", "s3cret")
        real_client = httpx.AsyncClient

        def mocked_client(**kw):
            return real_client(transport=httpx.MockTransport(handler), **kw)

        with patch.object(svc.httpx, "AsyncClient", mocked_client):
            disabled = await target.sync(
                {"a@x.y", GUEST}, ended=["gone@gmail.com", "never@gmail.com"], reactivated=[GUEST]
            )
        assert disabled == ["gone@gmail.com"]
        assert realm["attributes"] == {"allowedEmails": f"a@x.y,{GUEST}", "other": "kept"}
        assert log == ["realm-put", "user-put gone@gmail.com enabled=False", f"user-put {GUEST} enabled=True"]
