"""INTERNAL_API_SECRET has four consumers; they must all see the same live value.

Same defect class as genetics-results-suite-pol, different variable (genetics-results-suite-avt).
Two of the four snapshotted the value into a module global at import time, so a secret set only in
.env was silently ignored wherever the import ran before load_dotenv(). It was fail-closed — an
empty secret skips the bearer branch and falls through to requiring the IAP header — but the four
reads could disagree, and nothing here noticed because nothing tested it.

Every case below sets the environment *after* import, which is exactly what an import-time
snapshot cannot see.
"""

import hmac

import pytest
from conftest import settings_env
from fastapi.testclient import TestClient

from genetics_mcp_server.chat_api import app

SECRET = "set-after-import"


class TestAllConsumersReadTheSameLiveValue:
    def test_auth_dependency_accepts_the_bearer_form(self):
        """auth_required's bearer branch is the gate the other services authenticate through.

        /chat/v1/me rather than /chat/v1/auth: the latter is @is_public and reports status, so it
        answers 200 whatever is presented and could never show a rejection."""
        with settings_env(REQUIRE_AUTH="true", INTERNAL_API_SECRET=SECRET):
            with TestClient(app) as client:
                response = client.get(
                    "/chat/v1/me", headers={"Authorization": f"Bearer {SECRET}"}
                )
                assert response.status_code == 200

    def test_auth_dependency_rejects_a_wrong_bearer(self):
        with settings_env(REQUIRE_AUTH="true", INTERNAL_API_SECRET=SECRET):
            with TestClient(app) as client:
                response = client.get(
                    "/chat/v1/me", headers={"Authorization": "Bearer wrong"}
                )
                assert response.status_code == 401

    def test_tokens_validate_accepts_the_same_secret(self):
        with settings_env(REQUIRE_AUTH="true", INTERNAL_API_SECRET=SECRET):
            with TestClient(app) as client:
                response = client.post(
                    "/chat/v1/tokens/validate",
                    json={"token": "irrelevant"},
                    headers={"Authorization": f"Bearer {SECRET}"},
                )
                # 403 is the "not internal" answer; anything else means the secret was read
                assert response.status_code != 403

    def test_tokens_validate_rejects_a_wrong_bearer(self):
        with settings_env(REQUIRE_AUTH="true", INTERNAL_API_SECRET=SECRET):
            with TestClient(app) as client:
                response = client.post(
                    "/chat/v1/tokens/validate",
                    json={"token": "irrelevant"},
                    headers={"Authorization": "Bearer wrong"},
                )
                assert response.status_code == 403

    def test_tokens_validate_rejects_the_marker_alongside_an_asserted_identity(self):
        """The genuine callers are service-to-service and never assert an identity, so the pair
        can only mean a proxy built this request for a browser user. Minting a token answer from
        one would be the route acting on an end-user request it has no dependency guarding."""
        with settings_env(REQUIRE_AUTH="true", INTERNAL_API_SECRET=SECRET):
            with TestClient(app) as client:
                for marker in (
                    {"Authorization": f"Bearer {SECRET}"},
                    {"X-Internal-Auth": SECRET},
                ):
                    response = client.post(
                        "/chat/v1/tokens/validate",
                        json={"token": "irrelevant"},
                        headers={
                            **marker,
                            "X-Goog-Authenticated-User-Email": "someone@finngen.fi",
                        },
                    )
                    assert response.status_code == 403

    async def test_tokens_validate_rejects_a_non_ascii_bearer_rather_than_raising(self):
        """It used to compare_digest on str here, which raises TypeError — a 500 — for a
        non-ASCII bearer. Sharing is_internal_caller is what keeps the two comparisons alike.

        Driven below TestClient because httpx refuses to encode a non-ASCII header itself; the
        forged request this guards against is written straight onto the socket."""
        import fastapi
        import pytest as _pytest

        from genetics_mcp_server.routers.api_tokens import (
            TokenValidateRequest,
            validate_token,
        )

        request = fastapi.Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/chat/v1/tokens/validate",
                "headers": [(b"authorization", "Bearer säkerhet".encode("utf-8"))],
            }
        )
        with settings_env(REQUIRE_AUTH="true", INTERNAL_API_SECRET=SECRET):
            with _pytest.raises(fastapi.HTTPException) as exc:
                await validate_token(TokenValidateRequest(token="irrelevant"), request)
            assert exc.value.status_code == 403

    def test_the_tool_executor_sends_the_same_secret(self):
        """executor.py was already a live read, but it has to agree with the other three."""
        from genetics_mcp_server.tools.executor import ToolExecutor

        with settings_env(INTERNAL_API_SECRET=SECRET):
            executor = ToolExecutor()
            assert executor.client.headers.get("Authorization") == f"Bearer {SECRET}"
            # the third-party client must never carry it
            assert "Authorization" not in executor.external_client.headers

    def test_the_mcp_token_validation_sends_the_same_secret(self, monkeypatch):
        """mcp_server's cross-pod fallback authenticates to /tokens/validate with it."""
        from genetics_mcp_server import mcp_server

        sent = {}

        class _Response:
            status_code = 200

            def json(self):
                return {"valid": True}

        def fake_post(url, json=None, headers=None, timeout=None):
            sent["headers"] = headers
            return _Response()

        import httpx

        monkeypatch.setattr(httpx, "post", fake_post)
        # the local-DB branch runs first and falls through on its own: it imports
        # get_llm_config_db inside the function, so the module attribute is not patchable from
        # here, and a token that matches nothing simply returns None

        with settings_env(INTERNAL_API_SECRET=SECRET, CHAT_BACKEND_URL="http://chat"):
            assert mcp_server._validate_user_token("token") is True

        assert sent["headers"]["Authorization"] == f"Bearer {SECRET}"


class TestEmptySecretStaysFailClosed:
    """An unset secret must not become a way in — the bearer branch is skipped entirely rather
    than comparing against ""."""

    @pytest.mark.parametrize("presented", ["", "anything"])
    def test_no_bearer_is_accepted_when_the_secret_is_unset(self, presented):
        # TestClient WITHOUT the context manager, so the lifespan does not run: since
        # genetics-results-suite-618 a REQUIRE_AUTH=true chat-backend with no secret refuses to
        # start at all (TestTheStartupGuard below). This case is about the inbound gate, which
        # must stay fail-closed for the same reason it always did — the guard is one entrypoint's
        # check, not a property of auth_required.
        client = TestClient(app)
        with settings_env(REQUIRE_AUTH="true", INTERNAL_API_SECRET=""):
            response = client.get(
                "/chat/v1/me", headers={"Authorization": f"Bearer {presented}"}
            )
            assert response.status_code == 401

    def test_compare_digest_is_used_rather_than_equality(self):
        """Guards the shape of the comparison, which is what makes it constant-time.

        Lives in auth.core since genetics-results-suite-th2 — auth_required and the identity
        header now share one is_internal_caller instead of comparing the secret in two places.
        """
        import inspect

        from genetics_mcp_server.auth import core

        source = inspect.getsource(core)
        assert "hmac.compare_digest" in source
        assert hmac.compare_digest(SECRET, SECRET)


class TestTheStartupGuard:
    """genetics-results-suite-618: a deployed service must never fall back to no credential.

    The executor built its header as `{"Authorization": ...} if api_secret else {}`, so an unset
    variable meant every call to results-api and db-api went out ANONYMOUSLY — at request time,
    with no local signal, and indistinguishable at the far end from an authenticated internal
    call (results-api's usage log attributes via the secret and never sees a principal on a
    route that resolves none). The fix is to fail where a human is looking: at startup, naming
    the variable.
    """

    def test_the_helper_raises_and_names_the_variable(self):
        from genetics_mcp_server.config import require_internal_api_secret

        for value in ("", "   "):
            with settings_env(INTERNAL_API_SECRET=value):
                with pytest.raises(RuntimeError, match="INTERNAL_API_SECRET"):
                    require_internal_api_secret("a service")

    def test_the_helper_returns_the_live_secret(self):
        from genetics_mcp_server.config import require_internal_api_secret

        with settings_env(INTERNAL_API_SECRET=SECRET):
            assert require_internal_api_secret("a service") == SECRET

    def test_chat_backend_refuses_to_start_without_it(self):
        """The whole point: a crash-looping pod with a message, not a working pod making
        anonymous requests."""
        with settings_env(REQUIRE_AUTH="true", INTERNAL_API_SECRET=""):
            with pytest.raises(RuntimeError, match="INTERNAL_API_SECRET"):
                with TestClient(app):
                    pass

    def test_a_local_run_with_auth_off_is_not_blocked(self, monkeypatch):
        """REQUIRE_AUTH=false is a local run against an unauthenticated results-api, where the
        secret is genuinely optional (README documents it that way). The guard must not turn
        that into a startup failure."""
        from genetics_mcp_server import chat_api

        def _fail(component):
            raise AssertionError(f"the guard ran for {component} with REQUIRE_AUTH=false")

        monkeypatch.setattr(chat_api, "require_internal_api_secret", _fail)
        with settings_env(REQUIRE_AUTH="false", INTERNAL_API_SECRET=""):
            with TestClient(app):
                pass

    def test_the_mcp_server_entrypoint_checks_it_too(self, monkeypatch):
        """mcp-server is the other deployed process running the same executor.

        Driven through `main()` rather than asserted on `inspect.getsource(main)`, which the
        earlier version did: a source-substring assertion passes just as well when the call is
        commented out. `uvicorn.run` is replaced so nothing binds, and the guard sits ahead of
        it, so reaching the replacement at all is the failure this pins."""
        import sys

        import uvicorn

        from genetics_mcp_server import mcp_server

        served = []
        monkeypatch.setattr(uvicorn, "run", lambda *a, **k: served.append(True))
        monkeypatch.setattr(sys, "argv", ["mcp-server", "--transport", "sse"])
        with settings_env(MCP_API_KEY="a-key", INTERNAL_API_SECRET=""):
            with pytest.raises(RuntimeError, match="INTERNAL_API_SECRET"):
                mcp_server.main()
        assert not served, "main() served the app instead of refusing to start"
