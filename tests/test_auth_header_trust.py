"""The X-Goog-Authenticated-User-Email trust rule for chat-backend (genetics-results-suite-th2).

The header is settable by anything that can reach chat-backend on the pod network, so it must be
honoured only when the caller also presents INTERNAL_API_SECRET — the marker only an in-cluster
proxy can produce. Before this, forging one header granted admin (ADMIN_USERS is tested against
the forged string), minted plaintext API tokens that results-api and mcp-server both accept, and
read every user's chat history.

Same defect class as genetics-results-suite-fad in results-api; the precedence table below is
deliberately the same one, so the two services cannot drift apart.
"""

import pytest
from conftest import settings_env
from fastapi import Request
from fastapi.testclient import TestClient

from genetics_mcp_server.auth import core
from genetics_mcp_server.auth.dependencies import auth_required
from genetics_mcp_server.chat_api import app

INTERNAL_SECRET = "test-internal-secret"
USER_HEADER = "X-Goog-Authenticated-User-Email"


def _request(headers: dict[str, str]) -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/chat/v1/me",
            "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        }
    )


async def _resolve(headers: dict[str, str]):
    """auth_required against a bare request — no route, so is_public_endpoint is False."""
    return await auth_required(_request(headers))


@pytest.fixture
def prod_auth():
    """REQUIRE_AUTH=true with a secret and an explicit allow-list, i.e. the deployed shape."""
    with settings_env(
        REQUIRE_AUTH="true",
        INTERNAL_API_SECRET=INTERNAL_SECRET,
        ALLOWED_EMAIL_DOMAINS="finngen.fi",
        ALLOWED_EMAILS="guest@example.org",
    ):
        yield


# ---------------------------------------------------------------------------
# the vulnerability: the header alone must not authenticate
# ---------------------------------------------------------------------------


class TestHeaderAloneIsNotACredential:
    def test_forged_header_is_ignored(self, prod_auth):
        req = _request({USER_HEADER: "accounts.google.com:anyone@finngen.fi"})
        assert core.get_authenticated_user(req) is None

    async def test_forged_header_is_401(self, prod_auth):
        with pytest.raises(Exception) as exc:
            await _resolve({USER_HEADER: "accounts.google.com:anyone@finngen.fi"})
        assert exc.value.status_code == 401

    async def test_forged_admin_identity_is_401(self, prod_auth):
        """The motivating forgery: one header made the caller an ADMIN_USERS member."""
        with settings_env(ADMIN_USERS="admin@finngen.fi", ENABLE_ADMIN_PAGE="true"):
            with pytest.raises(Exception) as exc:
                await _resolve({USER_HEADER: "accounts.google.com:admin@finngen.fi"})
        assert exc.value.status_code == 401

    def test_bare_string_without_colon_is_ignored(self, prod_auth):
        assert core.get_authenticated_user(_request({USER_HEADER: "attacker"})) is None

    def test_wrong_bearer_does_not_unlock_the_header(self, prod_auth):
        req = _request(
            {
                USER_HEADER: "accounts.google.com:anyone@finngen.fi",
                "Authorization": "Bearer not-the-secret",
            }
        )
        assert core.get_authenticated_user(req) is None

    def test_header_ignored_when_no_secret_is_configured(self):
        """Fail closed rather than open when the deployment has no secret to compare against."""
        with settings_env(REQUIRE_AUTH="true", INTERNAL_API_SECRET=""):
            req = _request(
                {
                    USER_HEADER: "accounts.google.com:anyone@finngen.fi",
                    "Authorization": "Bearer ",
                }
            )
            assert core.get_authenticated_user(req) is None

    def test_non_ascii_bearer_fails_closed_rather_than_raising(self, prod_auth):
        """compare_digest on str raises TypeError for non-ASCII; it must compare bytes."""
        req = _request(
            {
                USER_HEADER: "accounts.google.com:anyone@finngen.fi",
                "Authorization": "Bearer säkerhet",
            }
        )
        assert core.get_authenticated_user(req) is None
        assert core.is_internal_caller(req) is False
        assert core.is_internal_caller(_request({"X-Internal-Auth": "säkerhet"})) is False

    def test_non_ascii_secret_compares_the_bytes_actually_sent(self):
        """Starlette decodes raw header bytes as latin-1, so re-encoding the presented
        credential with utf-8 compared mojibake against the secret. latin-1 undoes that decode
        exactly — for a caller that put utf-8 on the wire, which is what `_request` builds; see
        `test_which_wire_bytes_authenticate_a_non_ascii_secret` for the callers that do not."""
        secret = "sécret"
        with settings_env(REQUIRE_AUTH="true", INTERNAL_API_SECRET=secret):
            assert core.is_internal_caller(_request({"Authorization": f"Bearer {secret}"})) is True
            assert core.is_internal_caller(_request({core.INTERNAL_MARKER_HEADER: secret})) is True
            assert core.is_internal_caller(_request({"Authorization": "Bearer sekret"})) is False

    def test_which_wire_bytes_authenticate_a_non_ascii_secret(self):
        """Pin the accept/reject map over RAW wire bytes — what genetics-results-suite-ctq
        changed, and what no other test covers.

        Hand-built ASGI scope because TestClient cannot express it: `starlette/testclient.py`
        does `value.encode()` (utf-8) on httpx's already-decoded header str, so latin-1 wire
        bytes are rewritten to utf-8 before the app sees them and both cases below would look
        identical. `require_internal_api_secret` refuses a non-ASCII secret at startup, so this
        map is unreachable in a deployment; the comparison is still reachable, so it is pinned
        at that level.
        """

        def _raw(raw: bytes) -> Request:
            return Request(
                {
                    "type": "http",
                    "method": "GET",
                    "path": "/chat/v1/me",
                    "headers": [(b"authorization", raw)],
                }
            )

        with settings_env(REQUIRE_AUTH="true", INTERNAL_API_SECRET="sécret"):
            # utf-8 on the wire (aiohttp-shaped): authenticates now, 401 before ctq
            assert core.is_internal_caller(_raw(b"Bearer s\xc3\xa9cret")) is True
            # latin-1 on the wire (node/undici- and requests-shaped): 401 now, was accepted
            assert core.is_internal_caller(_raw(b"Bearer s\xe9cret")) is False

    def test_ascii_secret_is_unaffected(self, prod_auth):
        """The codecs agree on ASCII, so the deployed shape behaves exactly as before."""
        assert core.is_internal_caller(
            _request({"Authorization": f"Bearer {INTERNAL_SECRET}"})
        ) is True
        assert core.is_internal_caller(
            _request({core.INTERNAL_MARKER_HEADER: INTERNAL_SECRET})
        ) is True


# ---------------------------------------------------------------------------
# the precedence table
# ---------------------------------------------------------------------------


class TestPrecedence:
    async def test_marker_plus_allow_listed_header_is_that_user(self, prod_auth):
        user = await _resolve(
            {
                USER_HEADER: "accounts.google.com:someone@finngen.fi",
                "Authorization": f"Bearer {INTERNAL_SECRET}",
            }
        )
        assert user == "someone@finngen.fi"

    async def test_marker_plus_non_allow_listed_header_is_401_not_a_downgrade(self, prod_auth):
        """Rejecting rather than falling back to mcp-tool is the point: a downgrade would let
        anything holding the shared secret launder a refused identity into a working request."""
        with pytest.raises(Exception) as exc:
            await _resolve(
                {
                    USER_HEADER: "accounts.google.com:outsider@evil.example",
                    "Authorization": f"Bearer {INTERNAL_SECRET}",
                }
            )
        assert exc.value.status_code == 401

    async def test_marker_alone_is_the_service_identity(self, prod_auth):
        assert await _resolve({"Authorization": f"Bearer {INTERNAL_SECRET}"}) == "mcp-tool"

    async def test_the_gateway_header_transport_is_equivalent_to_the_bearer(self, prod_auth):
        """auth-gateway sends the marker as X-Internal-Auth so that a chat-backend still on the
        previous image ignores it — with the marker on Authorization, every browser user in the
        rollout window would have resolved to the single `mcp-tool` identity instead."""
        assert (
            await _resolve(
                {
                    USER_HEADER: "accounts.google.com:someone@finngen.fi",
                    "X-Internal-Auth": INTERNAL_SECRET,
                }
            )
            == "someone@finngen.fi"
        )
        assert await _resolve({"X-Internal-Auth": INTERNAL_SECRET}) == "mcp-tool"

    async def test_a_wrong_gateway_header_is_not_a_marker(self, prod_auth):
        with pytest.raises(Exception) as exc:
            await _resolve(
                {
                    USER_HEADER: "accounts.google.com:someone@finngen.fi",
                    "X-Internal-Auth": "not-the-secret",
                }
            )
        assert exc.value.status_code == 401

    async def test_marker_with_an_empty_header_is_the_service_identity(self, prod_auth):
        """oauth2-proxy sets an empty value when it has no session; nginx passes it through."""
        assert (
            await _resolve(
                {USER_HEADER: "", "Authorization": f"Bearer {INTERNAL_SECRET}"}
            )
            == "mcp-tool"
        )

    async def test_no_credential_at_all_is_401(self, prod_auth):
        with pytest.raises(Exception) as exc:
            await _resolve({})
        assert exc.value.status_code == 401


# ---------------------------------------------------------------------------
# allow-list semantics — must agree with oauth2-proxy, which admitted the user already
# ---------------------------------------------------------------------------


class TestAllowList:
    async def test_matching_is_case_insensitive_and_normalises(self, prod_auth):
        """oauth2-proxy lower-cases before its own domain check, so User@FinnGen.fi has a
        session there and must not be rejected here. The resolved identity is normalised so one
        person cannot split across several chat-history / API-token / ADMIN_USERS identities."""
        user = await _resolve(
            {
                USER_HEADER: "accounts.google.com: User@FinnGen.fi ",
                "Authorization": f"Bearer {INTERNAL_SECRET}",
            }
        )
        assert user == "user@finngen.fi"

    def test_explicit_address_outside_the_domain_is_allowed(self, prod_auth):
        req = _request(
            {
                USER_HEADER: "accounts.google.com:Guest@Example.ORG",
                "Authorization": f"Bearer {INTERNAL_SECRET}",
            }
        )
        assert core.get_authenticated_user(req) == "guest@example.org"

    def test_star_domain_means_allow_all(self):
        """oauth2-proxy treats `*` as any domain; matching it literally would allow nobody."""
        with settings_env(
            REQUIRE_AUTH="true", INTERNAL_API_SECRET=INTERNAL_SECRET, ALLOWED_EMAIL_DOMAINS="*"
        ):
            req = _request(
                {
                    USER_HEADER: "accounts.google.com:anyone@anywhere.example",
                    "Authorization": f"Bearer {INTERNAL_SECRET}",
                }
            )
            assert core.get_authenticated_user(req) == "anyone@anywhere.example"

    def test_unconfigured_allow_list_fails_open_but_still_needs_the_marker(self):
        """chat-backend only started reading ALLOWED_EMAIL* in th2. A pod that has not picked up
        the bearer-auth-allowed ConfigMap would otherwise inherit the finngen.fi default and
        lock out every user of a non-finngen deployment — worse than the bug. The marker, which
        is the half that actually closes the hole, is still required."""
        with settings_env(REQUIRE_AUTH="true", INTERNAL_API_SECRET=INTERNAL_SECRET):
            marked = _request(
                {
                    USER_HEADER: "accounts.google.com:someone@other.example",
                    "Authorization": f"Bearer {INTERNAL_SECRET}",
                }
            )
            assert core.get_authenticated_user(marked) == "someone@other.example"

            unmarked = _request({USER_HEADER: "accounts.google.com:someone@other.example"})
            assert core.get_authenticated_user(unmarked) is None


# ---------------------------------------------------------------------------
# ordered rollout: auth-gateway (sending side) ships before chat-backend
# ---------------------------------------------------------------------------


class TestOrderedRollout:
    """Which single-sided state is safe is a deployment decision this pins.

    auth-gateway sending the bearer while chat-backend still runs the old code is safe: the old
    code checked the bearer FIRST, so those requests would have resolved to mcp-tool — which is
    exactly why the gateway must not ship alone for long, but it is not a lockout. The reverse
    (chat-backend first) is a total lockout, and is what the ordering exists to prevent.
    """

    async def test_gateway_wire_shape_resolves_the_real_user_not_mcp_tool(self, prod_auth):
        """The exact headers auth-gateway now sends. If this returned mcp-tool, every user's
        chat history, downloads and API tokens would collapse into one shared identity."""
        user = await _resolve(
            {
                USER_HEADER: "accounts.google.com:real.user@finngen.fi",
                "Authorization": f"Bearer {INTERNAL_SECRET}",
                "X-Internal-MCP-Call": "",
            }
        )
        assert user == "real.user@finngen.fi"

    async def test_old_gateway_new_backend_is_the_lockout_state(self, prod_auth):
        """auth-gateway before the fix sends the header with no Authorization at all."""
        with pytest.raises(Exception) as exc:
            await _resolve({USER_HEADER: "accounts.google.com:real.user@finngen.fi"})
        assert exc.value.status_code == 401


# ---------------------------------------------------------------------------
# the endpoints the forged identity reached
# ---------------------------------------------------------------------------


class TestEndpointsThroughTheApp:
    def test_public_auth_endpoint_no_longer_reflects_a_forged_identity(self):
        """GET /chat/v1/auth is @is_public and echoes the identity plus is_admin — an
        unauthenticated oracle for enumerating ADMIN_USERS. It needs no change of its own: it
        resolves through get_authenticated_user, which now requires the marker."""
        with settings_env(
            REQUIRE_AUTH="true",
            INTERNAL_API_SECRET=INTERNAL_SECRET,
            ADMIN_USERS="admin@finngen.fi",
            ENABLE_ADMIN_PAGE="true",
        ):
            with TestClient(app) as client:
                forged = client.get(
                    "/chat/v1/auth",
                    headers={USER_HEADER: "accounts.google.com:admin@finngen.fi"},
                ).json()
                assert forged["authenticated"] is False
                assert forged["user"] is None
                assert forged["is_admin"] is False

                proxied = client.get(
                    "/chat/v1/auth",
                    headers={
                        USER_HEADER: "accounts.google.com:admin@finngen.fi",
                        "Authorization": f"Bearer {INTERNAL_SECRET}",
                    },
                ).json()
                assert proxied["user"] == "admin@finngen.fi"
                assert proxied["is_admin"] is True

    def test_token_minting_rejects_a_forged_identity(self):
        """POST /chat/v1/tokens returns the PLAINTEXT token, which mcp-server and results-api
        both accept — the pivot out of chat-backend."""
        with settings_env(REQUIRE_AUTH="true", INTERNAL_API_SECRET=INTERNAL_SECRET):
            with TestClient(app) as client:
                response = client.post(
                    "/chat/v1/tokens",
                    json={"name": "forged"},
                    headers={USER_HEADER: "accounts.google.com:victim@finngen.fi"},
                )
                assert response.status_code == 401

    def test_dev_mode_still_honours_the_header(self):
        """REQUIRE_AUTH=false is "no authentication at all" — no proxy and no secret to compare
        against. It returned "anonymous" for anyone before and still does, so requiring a marker
        there would break developing as a named user without protecting anything."""
        with settings_env(REQUIRE_AUTH="false"):
            with TestClient(app) as client:
                response = client.get(
                    "/chat/v1/me",
                    headers={USER_HEADER: "accounts.google.com:dev@localhost"},
                )
                assert response.status_code == 200
                assert response.json()["user"] == "dev@localhost"
