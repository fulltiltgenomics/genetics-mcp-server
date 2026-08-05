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
        with settings_env(REQUIRE_AUTH="true", INTERNAL_API_SECRET=""):
            with TestClient(app) as client:
                response = client.get(
                    "/chat/v1/me", headers={"Authorization": f"Bearer {presented}"}
                )
                assert response.status_code == 401

    def test_compare_digest_is_used_rather_than_equality(self):
        """Guards the shape of the comparison, which is what makes it constant-time."""
        import inspect

        from genetics_mcp_server.auth import dependencies

        source = inspect.getsource(dependencies)
        assert "hmac.compare_digest" in source
        assert hmac.compare_digest(SECRET, SECRET)
