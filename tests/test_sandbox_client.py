"""Tests for the sandbox HTTP transport (`4h6.47`).

Design of record: docs/code-execution-security.md section 2, "The HTTP contract between
chat-backend and the supervisor" (genetics-results-suite, `4h6.38`). The supervisor is built
against the same document by a different implementer and cannot share a module with this one,
so these tests pin the parts where two independently written ends drift apart silently: the
request field list, the deadlines, the 429 retry path and its FRESH execution id, and which
failures must not read to a user as "your script failed".

Everything here runs with no sandbox and no credentials: the HTTP layer is an
`httpx.MockTransport` and the signing key is a fixture.
"""

import base64
import hashlib
import json
import logging

import httpx
import jwt
import pytest

from genetics_mcp_server import sandbox_client
from genetics_mcp_server.config import settings as settings_module
from genetics_mcp_server.sandbox_client import (
    MAX_TIMEOUT_S,
    SandboxBusy,
    SandboxClient,
    SandboxDeadlineExceeded,
    SandboxInput,
    SandboxInternalError,
    SandboxNotConfigured,
    SandboxProtocolError,
    SandboxRejected,
    SandboxUnavailable,
    client_deadline_s,
)
from genetics_mcp_server.sandbox_token import SandboxTokenUnavailable

SIGNING_KEY = "test-sandbox-signing-key-that-is-32-bytes+"
USER = "u@finngen.fi"
SESSION = "conv-9"
CODE = "print('hello')"


@pytest.fixture(autouse=True)
def signing_key(monkeypatch):
    settings_module.get_settings.cache_clear()
    monkeypatch.setenv("SANDBOX_TOKEN_SIGNING_KEY", SIGNING_KEY)
    yield
    settings_module.get_settings.cache_clear()


def _ok_body(execution_id, **overrides):
    body = {
        "execution_id": execution_id,
        "status": "ok",
        "exit_code": 0,
        "signal": None,
        "duration_ms": 12,
        "output": "hello\n",
        "output_bytes": 6,
        "output_truncated": False,
        "error": None,
        "artifacts": [],
        "artifacts_omitted": 0,
    }
    body.update(overrides)
    return body


def _error_response(status, error_type, message="", headers=None):
    return httpx.Response(
        status,
        json={"execution_id": None, "error": {"type": error_type, "message": message}},
        headers=headers or {},
    )


class _Recorder:
    """A MockTransport handler that records every request and replays a scripted response."""

    def __init__(self, *responses):
        self.requests = []
        self.bodies = []
        self._responses = list(responses)

    def __call__(self, request):
        self.requests.append(request)
        body = json.loads(request.content.decode("utf-8")) if request.content else None
        self.bodies.append(body)
        response = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        if callable(response):
            return response(request, body)
        return response

    @property
    def transport(self):
        return httpx.MockTransport(self)


def _client(handler, **kwargs):
    slept = kwargs.pop("slept", None)

    async def sleep(seconds):
        if slept is not None:
            slept.append(seconds)

    return SandboxClient(
        "http://sandbox:8080", transport=httpx.MockTransport(handler), sleep=sleep, **kwargs
    )


def _echoing_ok(request, body):
    return httpx.Response(200, json=_ok_body(body["execution_id"]))


class TestTheRequestBody:
    async def test_carries_exactly_the_contract_fields(self):
        """Unknown top-level fields are a 400, so an extra field here fails every call."""
        recorder = _Recorder(_echoing_ok)
        client = _client(recorder)
        await client.execute(code=CODE, user=USER, session_id=SESSION)

        body = recorder.bodies[0]
        assert set(body) == {
            "code",
            "execution_id",
            "tokens",
            "user",
            "session_id",
            "timeout_s",
        }
        assert body["code"] == CODE
        assert body["user"] == USER
        assert body["session_id"] == SESSION
        assert body["timeout_s"] == sandbox_client.DEFAULT_TIMEOUT_S
        assert set(body["tokens"]) == {"db-api", "results-api"}

    async def test_posts_json_to_execute(self):
        recorder = _Recorder(_echoing_ok)
        await _client(recorder).execute(code=CODE, user=USER, session_id=SESSION)
        request = recorder.requests[0]
        assert request.method == "POST"
        assert str(request.url) == "http://sandbox:8080/execute"
        assert request.headers["content-type"] == "application/json"

    async def test_nothing_travels_in_a_header(self):
        """Headers are what proxies log, and the tokens must never be logged."""
        recorder = _Recorder(_echoing_ok)
        await _client(recorder).execute(code=CODE, user=USER, session_id=SESSION)
        headers = recorder.requests[0].headers
        assert "authorization" not in headers
        assert not [name for name in headers if name.lower().startswith("x-execution")]

    async def test_the_execution_id_is_the_jti_of_both_tokens(self):
        """One value in three roles — the /scratch directory, both jtis, the log join key."""
        recorder = _Recorder(_echoing_ok)
        await _client(recorder).execute(code=CODE, user=USER, session_id=SESSION)
        body = recorder.bodies[0]
        for audience, token in body["tokens"].items():
            claims = jwt.decode(
                token, SIGNING_KEY, algorithms=["HS256"], audience=audience, issuer="chat-backend"
            )
            assert claims["jti"] == body["execution_id"]
            assert claims["sub"] == body["user"]
            assert claims["sid"] == body["session_id"]

    async def test_a_caller_supplied_execution_id_is_used_verbatim(self):
        """The caller may have already chosen the /scratch directory name."""
        recorder = _Recorder(_echoing_ok)
        chosen = "11111111-2222-4333-8444-555555555555"
        result = await _client(recorder).execute(
            code=CODE, user=USER, session_id=SESSION, execution_id=chosen
        )
        assert recorder.bodies[0]["execution_id"] == chosen
        assert result["execution_id"] == chosen


class TestFailClosed:
    async def test_an_unset_signing_key_raises_and_sends_nothing(self, monkeypatch):
        """Every fallback from "no sandbox token" is either no credential or the shared one."""
        settings_module.get_settings.cache_clear()
        monkeypatch.delenv("SANDBOX_TOKEN_SIGNING_KEY", raising=False)
        recorder = _Recorder(_echoing_ok)
        with pytest.raises(SandboxTokenUnavailable):
            await _client(recorder).execute(code=CODE, user=USER, session_id=SESSION)
        assert recorder.requests == []

    async def test_token_unavailable_is_not_a_sandbox_error(self):
        """It must not be swallowed by a caller catching the transport's own failures."""
        assert not issubclass(SandboxTokenUnavailable, sandbox_client.SandboxError)

    async def test_no_token_reaches_a_log_or_an_exception(self, caplog):
        """The supervisor must never echo a token; if it does, ours never re-enters our logs."""
        captured = {}

        def handler(request, body=None):
            payload = json.loads(request.content.decode("utf-8"))
            captured["token"] = payload["tokens"]["db-api"]
            return httpx.Response(
                400,
                json={
                    "execution_id": None,
                    "error": {"type": "InvalidRequest", "message": captured["token"]},
                },
            )

        with caplog.at_level("DEBUG"):
            with pytest.raises(SandboxRejected) as excinfo:
                await _client(handler).execute(code=CODE, user=USER, session_id=SESSION)
        assert captured["token"] not in str(excinfo.value)
        assert captured["token"] not in caplog.text

    async def test_no_token_reaches_a_log_or_an_exception_through_error_type(self, caplog):
        """`error.type` is as peer-controlled as `error.message`, and unlike it, it is also
        carried onto the exception attribute — so it must be scrubbed in both places."""
        captured = {}

        def handler(request, body=None):
            payload = json.loads(request.content.decode("utf-8"))
            captured["token"] = payload["tokens"]["db-api"]
            return httpx.Response(
                400,
                json={
                    "execution_id": None,
                    "error": {"type": captured["token"], "message": "invalid request"},
                },
            )

        with caplog.at_level("DEBUG"):
            with pytest.raises(SandboxRejected) as excinfo:
                await _client(handler).execute(code=CODE, user=USER, session_id=SESSION)
        assert captured["token"] not in str(excinfo.value)
        assert captured["token"] not in (excinfo.value.error_type or "")
        assert captured["token"] not in caplog.text


class TestTheResult:
    async def test_a_success_is_returned_unchanged(self):
        """The tool layer renders; this module does not reshape, summarise or filter."""
        body_holder = {}

        def handler(request, parsed=None):
            payload = json.loads(request.content.decode("utf-8"))
            body_holder["sent"] = _ok_body(
                payload["execution_id"],
                artifacts=[{"name": "plot.png", "size": 12, "content_type": "image/png"}],
                a_field_added_later="tolerated",
            )
            return httpx.Response(200, json=body_holder["sent"])

        result = await _client(handler).execute(code=CODE, user=USER, session_id=SESSION)
        assert result == body_holder["sent"]

    @pytest.mark.parametrize("status", ["error", "timeout", "limit"])
    async def test_a_failing_script_is_a_result_not_an_exception(self, status):
        """A failing script is not an HTTP failure, and must not become one here."""

        def handler(request, parsed=None):
            payload = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                200,
                json=_ok_body(
                    payload["execution_id"],
                    status=status,
                    exit_code=1,
                    error={
                        "type": "ValueError",
                        "message": "boom",
                        "traceback": "...",
                        "limit": None,
                    },
                ),
            )

        result = await _client(handler).execute(code=CODE, user=USER, session_id=SESSION)
        assert result["status"] == status
        assert result["error"]["type"] == "ValueError"

    async def test_an_unrecognised_error_type_does_not_crash(self):
        """error.type is an OPEN string: half its range is the child's exception class name."""

        def handler(request, parsed=None):
            payload = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                200,
                json=_ok_body(
                    payload["execution_id"],
                    status="error",
                    error={"type": "SomeLibrarySpecificError", "message": "x", "traceback": None},
                ),
            )

        result = await _client(handler).execute(code=CODE, user=USER, session_id=SESSION)
        assert result["error"]["type"] == "SomeLibrarySpecificError"

    async def test_a_mismatched_execution_id_is_a_protocol_error(self):
        """The id is the join key; a result carrying another one cannot be recorded."""
        handler = _Recorder(httpx.Response(200, json=_ok_body("00000000-0000-4000-8000-000000000000")))
        with pytest.raises(SandboxProtocolError):
            await _client(handler).execute(code=CODE, user=USER, session_id=SESSION)

    @pytest.mark.parametrize(
        "response",
        [
            httpx.Response(200, content=b"not json"),
            httpx.Response(200, json=["not", "an", "object"]),
        ],
    )
    async def test_a_body_the_contract_does_not_describe_is_a_protocol_error(self, response):
        with pytest.raises(SandboxProtocolError):
            await _client(_Recorder(response)).execute(code=CODE, user=USER, session_id=SESSION)


class TestTheQueueAndTheRetry:
    async def test_a_429_retries_with_a_fresh_execution_id(self):
        """A repeated id is now refused with 409 DuplicateExecutionId, so the retry re-mints."""
        slept = []
        recorder = _Recorder(
            _error_response(429, "Busy", headers={"Retry-After": "7"}), _echoing_ok
        )
        client = _client(recorder, slept=slept)
        result = await client.execute(code=CODE, user=USER, session_id=SESSION)

        first, second = recorder.bodies
        assert first["execution_id"] != second["execution_id"]
        assert first["tokens"] != second["tokens"]
        assert result["execution_id"] == second["execution_id"]
        assert slept == [7]

    async def test_a_supplied_execution_id_is_not_reused_on_the_retry(self):
        slept = []
        recorder = _Recorder(_error_response(429, "Busy"), _echoing_ok)
        chosen = "11111111-2222-4333-8444-555555555555"
        await _client(recorder, slept=slept).execute(
            code=CODE, user=USER, session_id=SESSION, execution_id=chosen
        )
        assert recorder.bodies[0]["execution_id"] == chosen
        assert recorder.bodies[1]["execution_id"] != chosen

    async def test_a_persistent_429_surfaces_as_busy(self):
        slept = []
        recorder = _Recorder(_error_response(429, "Busy", headers={"Retry-After": "60"}))
        with pytest.raises(SandboxBusy) as excinfo:
            await _client(recorder, slept=slept).execute(code=CODE, user=USER, session_id=SESSION)
        assert excinfo.value.retry_after == 60
        assert len(recorder.requests) == sandbox_client.MAX_SUBMIT_ATTEMPTS

    async def test_an_absurd_retry_after_is_capped(self):
        slept = []
        recorder = _Recorder(
            _error_response(429, "Busy", headers={"Retry-After": "86400"}), _echoing_ok
        )
        await _client(recorder, slept=slept).execute(code=CODE, user=USER, session_id=SESSION)
        assert slept == [sandbox_client.MAX_RETRY_WAIT_S]

    async def test_a_missing_retry_after_falls_back(self):
        slept = []
        recorder = _Recorder(_error_response(429, "Busy"), _echoing_ok)
        await _client(recorder, slept=slept).execute(code=CODE, user=USER, session_id=SESSION)
        assert slept == [sandbox_client.RETRY_AFTER_FALLBACK_S]

    async def test_token_expired_retries_immediately_with_fresh_tokens(self):
        """409 TokenExpired means "you waited too long; re-mint and retry" — waiting again
        would only spend more of the new credential."""
        slept = []
        recorder = _Recorder(_error_response(409, "TokenExpired"), _echoing_ok)
        await _client(recorder, slept=slept).execute(code=CODE, user=USER, session_id=SESSION)
        assert slept == []
        assert recorder.bodies[0]["execution_id"] != recorder.bodies[1]["execution_id"]

    async def test_duplicate_execution_id_is_not_retried(self):
        """The two 409s are told apart by error.type and want opposite responses."""
        recorder = _Recorder(_error_response(409, "DuplicateExecutionId"))
        with pytest.raises(SandboxRejected) as excinfo:
            await _client(recorder).execute(code=CODE, user=USER, session_id=SESSION)
        assert excinfo.value.error_type == "DuplicateExecutionId"
        assert len(recorder.requests) == 1


class TestTheDeadlines:
    def test_the_client_deadline_clears_the_supervisors_worst_case(self):
        """Queue wait plus the full run, not just the run: below that the client would time
        out on executions the supervisor is about to answer."""
        assert client_deadline_s(MAX_TIMEOUT_S) > sandbox_client.MAX_QUEUED_WAIT_S + MAX_TIMEOUT_S
        assert client_deadline_s(sandbox_client.DEFAULT_TIMEOUT_S) > MAX_TIMEOUT_S

    async def test_the_read_timeout_on_the_wire_is_that_deadline(self):
        seen = {}

        def handler(request, parsed=None):
            seen.update(request.extensions["timeout"])
            payload = json.loads(request.content.decode("utf-8"))
            return httpx.Response(200, json=_ok_body(payload["execution_id"]))

        await _client(handler).execute(
            code=CODE, user=USER, session_id=SESSION, timeout_s=MAX_TIMEOUT_S
        )
        assert seen["read"] == client_deadline_s(MAX_TIMEOUT_S)
        assert seen["connect"] == sandbox_client.CONNECT_TIMEOUT_S

    async def test_a_read_timeout_is_not_a_script_failure(self):
        def handler(request, parsed=None):
            raise httpx.ReadTimeout("wedged")

        with pytest.raises(SandboxDeadlineExceeded):
            await _client(handler).execute(code=CODE, user=USER, session_id=SESSION)


class TestSandboxUnreachable:
    @pytest.mark.parametrize(
        "exc", [httpx.ConnectError("refused"), httpx.ConnectTimeout("no route")]
    )
    async def test_a_refused_connection_is_distinct(self, exc):
        """Recreate plus a 130s grace leaves no sandbox for up to ~130s of a deploy, and that
        must not read as "your analysis failed"."""

        def handler(request, parsed=None):
            raise exc

        with pytest.raises(SandboxUnavailable):
            await _client(handler).execute(code=CODE, user=USER, session_id=SESSION)

    async def test_not_ready_is_the_same_class_of_failure(self):
        recorder = _Recorder(_error_response(503, "NotReady"))
        with pytest.raises(SandboxUnavailable) as excinfo:
            await _client(recorder).execute(code=CODE, user=USER, session_id=SESSION)
        assert excinfo.value.status_code == 503

    async def test_a_gateway_error_is_the_same_class_of_failure(self):
        recorder = _Recorder(httpx.Response(502, content=b"<html>bad gateway</html>"))
        with pytest.raises(SandboxUnavailable):
            await _client(recorder).execute(code=CODE, user=USER, session_id=SESSION)


class TestRefusals:
    @pytest.mark.parametrize(
        "status,error_type",
        [
            (400, "InvalidRequest"),
            (404, "NotFound"),
            (405, "MethodNotAllowed"),
            (408, "RequestTimeout"),
            (413, "PayloadTooLarge"),
            (415, "UnsupportedMediaType"),
        ],
    )
    async def test_a_refusal_carries_its_type(self, status, error_type):
        recorder = _Recorder(_error_response(status, error_type, "no"))
        with pytest.raises(SandboxRejected) as excinfo:
            await _client(recorder).execute(code=CODE, user=USER, session_id=SESSION)
        assert excinfo.value.status_code == status
        assert excinfo.value.error_type == error_type

    async def test_a_supervisor_bug_is_its_own_class(self):
        recorder = _Recorder(_error_response(500, "InternalError"))
        with pytest.raises(SandboxInternalError):
            await _client(recorder).execute(code=CODE, user=USER, session_id=SESSION)

    async def test_an_unparseable_error_body_does_not_crash_the_client(self):
        recorder = _Recorder(httpx.Response(400, content=b"\xff\xfe not json"))
        with pytest.raises(SandboxRejected):
            await _client(recorder).execute(code=CODE, user=USER, session_id=SESSION)


class TestLocalValidation:
    async def test_an_over_ceiling_timeout_is_rejected_not_clamped(self):
        """The supervisor rejects it too: clamping is a silent behaviour change on a path fed
        from a model-influenceable direction, and it would desync the two deadlines."""
        recorder = _Recorder(_echoing_ok)
        with pytest.raises(SandboxRejected):
            await _client(recorder).execute(
                code=CODE, user=USER, session_id=SESSION, timeout_s=MAX_TIMEOUT_S + 1
            )
        assert recorder.requests == []

    @pytest.mark.parametrize("timeout_s", [0, -1, 1.5, True])
    async def test_a_nonsense_timeout_never_reaches_the_wire(self, timeout_s):
        recorder = _Recorder(_echoing_ok)
        with pytest.raises(SandboxRejected):
            await _client(recorder).execute(
                code=CODE, user=USER, session_id=SESSION, timeout_s=timeout_s
            )
        assert recorder.requests == []

    @pytest.mark.parametrize("code", ["", "   \n\t "])
    async def test_empty_code_never_reaches_the_wire(self, code):
        recorder = _Recorder(_echoing_ok)
        with pytest.raises(SandboxRejected):
            await _client(recorder).execute(code=code, user=USER, session_id=SESSION)
        assert recorder.requests == []

    async def test_oversized_code_is_measured_on_its_utf8_encoding(self):
        """Not on the JSON escaping, which can triple the same program's on-wire length."""
        recorder = _Recorder(_echoing_ok)
        over = "é" * (sandbox_client.MAX_CODE_BYTES // 2 + 1)
        assert len(over) < sandbox_client.MAX_CODE_BYTES
        with pytest.raises(SandboxRejected):
            await _client(recorder).execute(code=over, user=USER, session_id=SESSION)
        assert recorder.requests == []

    @pytest.mark.parametrize(
        "execution_id",
        [
            "../../etc/passwd",
            "not-a-uuid",
            "3F2504E0-4F89-41D3-9A0C-0305E82C3301",  # uppercase is not the contract's form
            "3f2504e0-4f89-41d3-9a0c-0305e82c3301\n",
            "3f2504e0-4f89-41d3-9a0c-0305e82c330",
            "",
            42,
        ],
    )
    async def test_a_malformed_execution_id_never_reaches_the_wire(self, execution_id):
        """It becomes the /scratch/<id> directory name, and the supervisor 400s anything that
        does not match §2's regex — so this client refuses it locally."""
        recorder = _Recorder(_echoing_ok)
        with pytest.raises(SandboxRejected):
            await _client(recorder).execute(
                code=CODE, user=USER, session_id=SESSION, execution_id=execution_id
            )
        assert recorder.requests == []

    async def test_a_contract_shaped_execution_id_is_accepted_and_sent(self):
        recorder = _Recorder(_echoing_ok)
        given = "3f2504e0-4f89-41d3-9a0c-0305e82c3301"
        await _client(recorder).execute(
            code=CODE, user=USER, session_id=SESSION, execution_id=given
        )
        assert recorder.bodies[0]["execution_id"] == given

    @pytest.mark.parametrize("user", ["", "   ", None, 42])
    async def test_an_empty_user_is_a_refusal_not_a_bare_value_error(self, user):
        """A caller catching SandboxError must not miss it: the tokens' `sub` is required and
        minting would otherwise raise something outside this module's hierarchy."""
        recorder = _Recorder(_echoing_ok)
        with pytest.raises(SandboxRejected):
            await _client(recorder).execute(code=CODE, user=user, session_id=SESSION)
        assert recorder.requests == []

    @pytest.mark.parametrize("session_id", ["", "   ", None, 42])
    async def test_an_empty_session_id_is_a_refusal(self, session_id):
        recorder = _Recorder(_echoing_ok)
        with pytest.raises(SandboxRejected):
            await _client(recorder).execute(code=CODE, user=USER, session_id=session_id)
        assert recorder.requests == []


class TestHealth:
    async def test_ok(self):
        recorder = _Recorder(httpx.Response(200, json={"status": "ok", "busy": True, "queued": 1}))
        health = await _client(recorder).health()
        assert (health.ready, health.status, health.busy, health.queued) == (True, "ok", True, 1)
        assert str(recorder.requests[0].url) == "http://sandbox:8080/health"

    @pytest.mark.parametrize("status", ["starting", "draining"])
    async def test_a_503_is_parsed_as_a_health_body_not_an_error_object(self, status):
        """/health is the single exception to the uniform error shape; a client that parsed
        this 503 as {execution_id, error} would KeyError on every startup."""
        recorder = _Recorder(
            httpx.Response(503, json={"status": status, "busy": False, "queued": 0})
        )
        health = await _client(recorder).health()
        assert health.ready is False
        assert health.status == status
        assert health.queued == 0

    async def test_a_garbage_body_degrades_rather_than_raising(self):
        recorder = _Recorder(httpx.Response(200, content=b"nope"))
        health = await _client(recorder).health()
        assert health.status == "unknown"
        assert health.busy is None and health.queued is None

    async def test_an_unreachable_sandbox_is_unavailable(self):
        def handler(request, parsed=None):
            raise httpx.ConnectError("refused")

        with pytest.raises(SandboxUnavailable):
            await _client(handler).health()


class TestFetchArtifact:
    """`fetch_artifact` runs after the analysis has already succeeded, so it never raises.

    Every failure it can meet — reaped, evicted, oversize, restarted sandbox, a body that is
    not what the contract says — means the same thing to its caller: there is no picture to
    show. Turning any of them into an exception would lose the analysis to save the figure.
    """

    EID = "3f1a2b3c-4d5e-4f60-8a1b-2c3d4e5f6071"

    def _served(self, data: bytes, content_type="image/png"):
        import base64

        return _Recorder(
            httpx.Response(
                200,
                json={
                    "execution_id": self.EID,
                    "name": "plot.png",
                    "content_type": content_type,
                    "size": len(data),
                    "content_base64": base64.b64encode(data).decode("ascii"),
                },
            )
        )

    async def test_returns_the_encoded_bytes_and_asks_by_name(self):
        recorder = self._served(b"\x89PNG-data")
        fetched = await _client(recorder).fetch_artifact(self.EID, "plot.png")
        assert fetched["content_type"] == "image/png"
        assert fetched["size"] == len(b"\x89PNG-data")

        url = recorder.requests[0].url
        assert url.path == "/artifact"
        assert dict(url.params) == {"execution_id": self.EID, "name": "plot.png"}

    @pytest.mark.parametrize(
        "response",
        [
            _error_response(404, "NotFound"),
            _error_response(413, "ArtifactTooLarge"),
            _error_response(500, "InternalError"),
            httpx.Response(200, content=b"not json"),
            httpx.Response(200, json={"execution_id": EID, "content_base64": "!!!not base64"}),
            httpx.Response(200, json={"execution_id": EID}),
        ],
    )
    async def test_every_refusal_and_malformed_body_is_none_not_an_exception(self, response):
        assert await _client(_Recorder(response)).fetch_artifact(self.EID, "plot.png") is None

    async def test_an_unreachable_sandbox_is_none(self):
        def handler(request, parsed=None):
            raise httpx.ConnectError("refused")

        assert await _client(handler).fetch_artifact(self.EID, "plot.png") is None

    async def test_a_malformed_execution_id_is_never_sent(self):
        recorder = _Recorder(httpx.Response(200, json={}))
        assert await _client(recorder).fetch_artifact("../etc", "plot.png") is None
        assert recorder.requests == []

    async def test_a_body_over_the_size_cap_is_refused_locally(self):
        oversize = b"x" * (sandbox_client.ARTIFACT_READ_MAX_BYTES + 1)
        assert await _client(self._served(oversize)).fetch_artifact(self.EID, "plot.png") is None


class TestConfiguration:
    def test_the_base_url_is_one_setting_with_no_local_prod_branch(self, monkeypatch):
        settings_module.get_settings.cache_clear()
        monkeypatch.setenv("SANDBOX_URL", "http://sandbox.genetics.svc.cluster.local:8080")
        try:
            assert SandboxClient().base_url == "http://sandbox.genetics.svc.cluster.local:8080"
        finally:
            settings_module.get_settings.cache_clear()

    def test_a_trailing_slash_does_not_double_up_the_path(self, monkeypatch):
        assert SandboxClient("http://sandbox:8080/").base_url == "http://sandbox:8080"

    def test_an_unset_sandbox_url_refuses_to_build_a_client(self, monkeypatch):
        """No default: the old one was db-api's port on a dev machine (`6um`), so "nothing
        configured" has to be loud rather than a POST at whatever answers on 8080."""
        settings_module.get_settings.cache_clear()
        monkeypatch.delenv("SANDBOX_URL", raising=False)
        try:
            with pytest.raises(SandboxNotConfigured) as excinfo:
                SandboxClient()
            assert "SANDBOX_URL" in str(excinfo.value)
        finally:
            settings_module.get_settings.cache_clear()

    def test_an_explicit_base_url_needs_no_environment(self, monkeypatch):
        settings_module.get_settings.cache_clear()
        monkeypatch.delenv("SANDBOX_URL", raising=False)
        try:
            assert SandboxClient("http://127.0.0.1:8081").base_url == "http://127.0.0.1:8081"
        finally:
            settings_module.get_settings.cache_clear()


class TestInputs:
    """`inputs` is the one field whose rules live in two independently written ends, so each
    rule the supervisor enforces is asserted here to be enforced BEFORE the wire — a request
    it is certain to 413 or 400 must never leave — and the wire element is pinned field for
    field."""

    async def test_an_input_is_carried_as_the_contract_element(self):
        recorder = _Recorder(_echoing_ok)
        await _client(recorder).execute(
            code=CODE,
            user=USER,
            session_id=SESSION,
            inputs=[SandboxInput(name="data.tsv", content=b"a\tb\n", content_type="text/csv")],
        )
        body = recorder.bodies[0]
        assert set(body) == {
            "code",
            "execution_id",
            "tokens",
            "user",
            "session_id",
            "timeout_s",
            "inputs",
        }
        (element,) = body["inputs"]
        assert set(element) == {"name", "content_base64", "sha256", "content_type"}
        assert element["name"] == "data.tsv"
        assert base64.b64decode(element["content_base64"]) == b"a\tb\n"
        assert element["sha256"] == hashlib.sha256(b"a\tb\n").hexdigest()
        assert element["content_type"] == "text/csv"

    async def test_content_type_is_omitted_rather_than_sent_as_null(self):
        recorder = _Recorder(_echoing_ok)
        await _client(recorder).execute(
            code=CODE,
            user=USER,
            session_id=SESSION,
            inputs=[SandboxInput(name="a.txt", content=b"x")],
        )
        assert set(recorder.bodies[0]["inputs"][0]) == {"name", "content_base64", "sha256"}

    @pytest.mark.parametrize("inputs", [None, [], ()])
    async def test_the_field_is_absent_when_there_are_none(self, inputs):
        """Unknown top-level fields are a 400, so a sandbox predating this field must still
        serve every execution that does not use it."""
        recorder = _Recorder(_echoing_ok)
        await _client(recorder).execute(
            code=CODE, user=USER, session_id=SESSION, inputs=inputs
        )
        assert "inputs" not in recorder.bodies[0]

    async def test_an_input_at_the_per_input_cap_is_accepted(self):
        recorder = _Recorder(_echoing_ok)
        await _client(recorder).execute(
            code=CODE,
            user=USER,
            session_id=SESSION,
            inputs=[SandboxInput(name="big.bin", content=b"x" * sandbox_client.MAX_INPUT_BYTES)],
        )
        assert len(recorder.requests) == 1

    async def test_one_byte_over_the_per_input_cap_never_reaches_the_wire(self):
        recorder = _Recorder(_echoing_ok)
        with pytest.raises(SandboxRejected):
            await _client(recorder).execute(
                code=CODE,
                user=USER,
                session_id=SESSION,
                inputs=[
                    SandboxInput(name="big.bin", content=b"x" * (sandbox_client.MAX_INPUT_BYTES + 1))
                ],
            )
        assert recorder.requests == []

    async def test_inputs_summing_to_the_total_cap_are_accepted(self):
        half = sandbox_client.MAX_INPUTS_TOTAL_BYTES // 2
        recorder = _Recorder(_echoing_ok)
        await _client(recorder).execute(
            code=CODE,
            user=USER,
            session_id=SESSION,
            inputs=[
                SandboxInput(name="a.bin", content=b"a" * half),
                SandboxInput(name="b.bin", content=b"b" * (sandbox_client.MAX_INPUTS_TOTAL_BYTES - half)),
            ],
        )
        assert len(recorder.requests) == 1

    async def test_one_byte_over_the_total_cap_never_reaches_the_wire(self):
        """Each input is under the per-input cap; only their sum is over it."""
        half = sandbox_client.MAX_INPUTS_TOTAL_BYTES // 2
        recorder = _Recorder(_echoing_ok)
        with pytest.raises(SandboxRejected) as excinfo:
            await _client(recorder).execute(
                code=CODE,
                user=USER,
                session_id=SESSION,
                inputs=[
                    SandboxInput(name="a.bin", content=b"a" * half),
                    SandboxInput(
                        name="b.bin",
                        content=b"b" * (sandbox_client.MAX_INPUTS_TOTAL_BYTES - half + 1),
                    ),
                ],
            )
        assert "total" in str(excinfo.value)
        assert recorder.requests == []

    async def test_the_maximum_number_of_inputs_is_accepted(self):
        recorder = _Recorder(_echoing_ok)
        await _client(recorder).execute(
            code=CODE,
            user=USER,
            session_id=SESSION,
            inputs=[
                SandboxInput(name=f"f{i}.txt", content=b"x")
                for i in range(sandbox_client.MAX_INPUTS)
            ],
        )
        assert len(recorder.bodies[0]["inputs"]) == sandbox_client.MAX_INPUTS

    async def test_one_input_too_many_never_reaches_the_wire(self):
        recorder = _Recorder(_echoing_ok)
        with pytest.raises(SandboxRejected):
            await _client(recorder).execute(
                code=CODE,
                user=USER,
                session_id=SESSION,
                inputs=[
                    SandboxInput(name=f"f{i}.txt", content=b"x")
                    for i in range(sandbox_client.MAX_INPUTS + 1)
                ],
            )
        assert recorder.requests == []

    @pytest.mark.parametrize(
        "name",
        [
            "",
            ".",
            "..",
            ".hidden",
            "a/b.txt",
            "a\\b.txt",
            "../etc/passwd",
            "/etc/passwd",
            "a b.txt",
            "a.txt\n",
            "x" * 65,
            42,
            None,
        ],
    )
    async def test_a_name_the_supervisor_would_refuse_never_reaches_the_wire(self, name):
        """The name becomes a file inside the execution's inputs directory, so it is bounded
        the way execution_id is: refused here rather than sent and 400ed."""
        recorder = _Recorder(_echoing_ok)
        with pytest.raises(SandboxRejected):
            await _client(recorder).execute(
                code=CODE,
                user=USER,
                session_id=SESSION,
                inputs=[SandboxInput(name=name, content=b"x")],
            )
        assert recorder.requests == []

    @pytest.mark.parametrize("name", ["a", "data.tsv", "A_1-2.vcf.gz", "x" * 64])
    async def test_a_contract_shaped_name_is_sent_unchanged(self, name):
        recorder = _Recorder(_echoing_ok)
        await _client(recorder).execute(
            code=CODE,
            user=USER,
            session_id=SESSION,
            inputs=[SandboxInput(name=name, content=b"x")],
        )
        assert recorder.bodies[0]["inputs"][0]["name"] == name

    async def test_two_inputs_with_the_same_name_are_refused_not_deduplicated(self):
        """One would overwrite the other in the inputs directory, and which one survived
        would decide what the script read."""
        recorder = _Recorder(_echoing_ok)
        with pytest.raises(SandboxRejected) as excinfo:
            await _client(recorder).execute(
                code=CODE,
                user=USER,
                session_id=SESSION,
                inputs=[
                    SandboxInput(name="a.txt", content=b"first"),
                    SandboxInput(name="a.txt", content=b"second"),
                ],
            )
        assert "duplicates" in str(excinfo.value)
        assert recorder.requests == []

    @pytest.mark.parametrize("content_type", ["csv", "", "text/", "text/csv; ", 42])
    async def test_a_content_type_that_is_not_a_media_type_never_reaches_the_wire(
        self, content_type
    ):
        """The supervisor validates it against RFC 7231 and 400s the rest — including the
        empty string, which must be refused rather than quietly dropped from the body."""
        recorder = _Recorder(_echoing_ok)
        with pytest.raises(SandboxRejected):
            await _client(recorder).execute(
                code=CODE,
                user=USER,
                session_id=SESSION,
                inputs=[SandboxInput(name="a.txt", content=b"x", content_type=content_type)],
            )
        assert recorder.requests == []

    @pytest.mark.parametrize(
        "content_type", ["text/csv", "text/csv; charset=utf-8", 'text/csv; name="a b"']
    )
    async def test_a_real_fetchers_content_type_is_accepted(self, content_type):
        recorder = _Recorder(_echoing_ok)
        await _client(recorder).execute(
            code=CODE,
            user=USER,
            session_id=SESSION,
            inputs=[SandboxInput(name="a.txt", content=b"x", content_type=content_type)],
        )
        assert recorder.bodies[0]["inputs"][0]["content_type"] == content_type

    async def test_a_mutable_buffer_is_refused_as_content(self):
        """SandboxInput is frozen, a bytearray is not, and _submit re-encodes on every retry
        while validation runs once — a buffer grown afterwards would pass the cap here and
        be refused by the supervisor as if the two ends disagreed about the numbers."""
        recorder = _Recorder(_echoing_ok)
        with pytest.raises(SandboxRejected):
            await _client(recorder).execute(
                code=CODE,
                user=USER,
                session_id=SESSION,
                inputs=[SandboxInput(name="a.txt", content=bytearray(b"x"))],
            )
        assert recorder.requests == []

    async def test_a_matching_expected_digest_is_verified_and_then_discarded(self):
        """It closes the chain to whoever fetched the bytes, and must NOT go on the wire: the
        supervisor computes its own from what it decoded, which is the only way its
        DigestMismatch still covers the hop between these two processes."""
        content = b"fetched bytes"
        recorder = _Recorder(_echoing_ok)
        await _client(recorder).execute(
            code=CODE,
            user=USER,
            session_id=SESSION,
            inputs=[
                SandboxInput(
                    name="a.txt",
                    content=content,
                    expected_sha256=hashlib.sha256(content).hexdigest(),
                )
            ],
        )
        (element,) = recorder.bodies[0]["inputs"]
        assert "expected_sha256" not in element
        assert element["sha256"] == hashlib.sha256(content).hexdigest()

    async def test_a_mismatching_expected_digest_is_refused_before_anything_leaves(self):
        recorder = _Recorder(_echoing_ok)
        with pytest.raises(SandboxRejected) as excinfo:
            await _client(recorder).execute(
                code=CODE,
                user=USER,
                session_id=SESSION,
                inputs=[
                    SandboxInput(
                        name="a.txt", content=b"these bytes", expected_sha256="0" * 64
                    )
                ],
            )
        assert "expected sha256" in str(excinfo.value)
        assert recorder.requests == []

    @pytest.mark.parametrize(
        "expected", ["", "deadbeef", "0" * 63, "A" * 64, 42]
    )
    async def test_a_malformed_expected_digest_is_refused(self, expected):
        recorder = _Recorder(_echoing_ok)
        with pytest.raises(SandboxRejected):
            await _client(recorder).execute(
                code=CODE,
                user=USER,
                session_id=SESSION,
                inputs=[SandboxInput(name="a.txt", content=b"x", expected_sha256=expected)],
            )
        assert recorder.requests == []

    async def test_a_retry_re_sends_the_inputs_under_a_fresh_execution_id(self):
        """Inputs are ephemeral to one execution and nothing about them may be keyed to an id
        that does not exist when the body is built — a design that uploaded them once, against
        the first id, delivers nothing to the execution that actually runs."""
        slept = []
        recorder = _Recorder(_error_response(429, "Busy"), _echoing_ok)
        inputs = [SandboxInput(name="a.txt", content=b"payload", content_type="text/plain")]
        result = await _client(recorder, slept=slept).execute(
            code=CODE, user=USER, session_id=SESSION, inputs=inputs
        )
        first, second = recorder.bodies
        assert first["execution_id"] != second["execution_id"]
        assert first["inputs"] == second["inputs"]
        assert base64.b64decode(second["inputs"][0]["content_base64"]) == b"payload"
        assert result["execution_id"] == second["execution_id"]

    async def test_inputs_that_are_not_a_list_are_refused(self):
        recorder = _Recorder(_echoing_ok)
        for bad in (SandboxInput(name="a.txt", content=b"x"), "a.txt", {"a.txt": b"x"}):
            with pytest.raises(SandboxRejected):
                await _client(recorder).execute(
                    code=CODE, user=USER, session_id=SESSION, inputs=bad
                )
        assert recorder.requests == []

    async def test_an_element_that_is_not_a_sandbox_input_is_refused(self):
        recorder = _Recorder(_echoing_ok)
        with pytest.raises(SandboxRejected):
            await _client(recorder).execute(
                code=CODE,
                user=USER,
                session_id=SESSION,
                inputs=[{"name": "a.txt", "content_base64": "eA=="}],
            )
        assert recorder.requests == []

    async def test_an_inputs_too_large_413_is_reported_as_drift(self, caplog):
        """Every cap is mirrored, so this answer can only mean the two ends disagree about
        the numbers — which is a deployment problem, not the caller's bug."""
        recorder = _Recorder(_error_response(413, sandbox_client.ERROR_INPUTS_TOO_LARGE, "too big"))
        with caplog.at_level(logging.WARNING):
            with pytest.raises(SandboxRejected):
                await _client(recorder).execute(
                    code=CODE,
                    user=USER,
                    session_id=SESSION,
                    inputs=[SandboxInput(name="a.txt", content=b"x")],
                )
        assert any("drift" in r.message or "had accepted" in r.getMessage() for r in caplog.records)
