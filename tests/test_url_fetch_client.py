"""Tests for the url-fetcher transport and the per-user fetch cache (`vxtv.12`).

Wire contract of record: `url-fetcher/server.py`, `fetch.py` and `guard.py` in
genetics-results-suite. The fetcher is built by a different implementer and cannot share a
module with this one, so these tests pin the places where two independently written ends drift
apart silently: the request body's one field, the success shape's every field, and — the one
that is not a shape at all — which failures may be retried and which never may.

The cross-user tests are driven as the failure they exist to prevent: user B asks for the URL
user A already fetched, and must reach the wire.

Everything here runs with no fetcher: the HTTP layer is an `httpx.MockTransport`.
"""

import base64
import hashlib
import json

import httpx
import pytest

from genetics_mcp_server import url_fetch_client
from genetics_mcp_server.config import settings as settings_module
from genetics_mcp_server.url_fetch_client import (
    MAX_FETCH_BYTES,
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    NAME_MAX,
    FetchCache,
    UrlFetchClient,
    UrlFetchNotConfigured,
    UrlFetchProtocolError,
    UrlFetchRefused,
    UrlFetchUnavailable,
    get_url_fetch_client,
    reset_url_fetch_client,
)

URL = "https://raw.githubusercontent.com/org/repo/main/data.tsv"
OTHER_URL = "https://zenodo.org/record/1/files/other.tsv"
ALICE = "alice@finngen.fi"
BOB = "bob@finngen.fi"


def _ok_body(content=b"rsid\tbeta\n", **overrides):
    body = {
        "url": URL,
        "name": "data.tsv",
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "content_type": "text/tab-separated-values",
        "content_encoding": None,
        "redirects": 0,
        "content_b64": base64.b64encode(content).decode("ascii"),
    }
    body.update(overrides)
    return body


def _error_response(status, error_type, message="", retryable=False, **details):
    """The envelope `server.error_body` builds: type, message, retryable, plus whatever extra
    fields the Refused carried."""
    body = {"type": error_type, "message": message, "retryable": retryable}
    body.update(details)
    return httpx.Response(status, json={"error": body})


class _WatchedBody(httpx.AsyncByteStream):
    """A response body that records whether anything read it, so a test can assert a refusal
    happened *before* the bytes were taken in."""

    def __init__(self, data):
        self._data = data
        self.read = False

    async def __aiter__(self):
        self.read = True
        yield self._data


def _streamed_response(body, *, status=200, content_length=None):
    """A response whose Content-Length is set independently of its body, which is the only way
    to drive a header that lies about what follows."""
    stream = _WatchedBody(body)
    headers = {"Content-Type": "application/json"}
    declared = len(body) if content_length is None else content_length
    if declared is not False:
        headers["Content-Length"] = str(declared)
    return httpx.Response(status, headers=headers, stream=stream), stream


class _Recorder:
    """A MockTransport handler that records every request and replays a scripted response."""

    def __init__(self, *responses):
        self.requests = []
        self.bodies = []
        self._responses = list(responses) or [httpx.Response(200, json=_ok_body())]

    def __call__(self, request):
        self.requests.append(request)
        self.bodies.append(json.loads(request.content.decode("utf-8")) if request.content else None)
        response = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        return response(request) if callable(response) else response

    @property
    def calls(self):
        return len(self.requests)


def _client(*responses, cache=None):
    recorder = _Recorder(*responses)
    client = UrlFetchClient(
        "http://url-fetcher:8090", transport=httpx.MockTransport(recorder), cache=cache
    )
    return client, recorder


def _caching_client(*responses, ttl_s=900, max_bytes=MAX_FETCH_BYTES * 8, clock=None):
    cache = FetchCache(ttl_s=ttl_s, max_bytes=max_bytes, **({"clock": clock} if clock else {}))
    return _client(*responses, cache=cache)


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class TestTheRequestBody:
    async def test_carries_the_url_and_nothing_else(self):
        """`server._read_body` REFUSES an unknown field rather than ignoring it, so an extra
        field here would 400 every call — including `user`, which must never reach the wire."""
        client, recorder = _client()
        await client.fetch(URL, user=ALICE)
        assert recorder.bodies[0] == {"url": URL}

    async def test_posts_json_to_fetch(self):
        client, recorder = _client()
        await client.fetch(URL, user=ALICE)
        request = recorder.requests[0]
        assert request.method == "POST"
        assert str(request.url) == "http://url-fetcher:8090/fetch"
        assert request.headers["content-type"] == "application/json"

    async def test_no_credential_travels_to_the_fetcher(self):
        """The fetcher is unauthenticated by design; a credential here would be one on the wire
        to the only pod in the namespace that dials the open internet."""
        client, recorder = _client()
        await client.fetch(URL, user=ALICE)
        headers = recorder.requests[0].headers
        assert "authorization" not in headers
        assert not any(ALICE in value for value in headers.values())

    async def test_an_oversize_url_is_refused_without_a_round_trip(self):
        client, recorder = _client()
        with pytest.raises(UrlFetchRefused):
            await client.fetch("https://zenodo.org/" + "a" * MAX_REQUEST_BYTES, user=ALICE)
        assert recorder.calls == 0


class TestTheSuccessShape:
    async def test_every_field_of_the_200_reaches_the_result(self):
        content = b"col\tvalue\n1\t2\n"
        client, _ = _client(
            httpx.Response(
                200,
                json=_ok_body(
                    content,
                    url="https://zenodo.org/final.tsv",
                    name="final.tsv",
                    content_type="text/csv; charset=utf-8",
                    content_encoding="identity",
                    redirects=2,
                ),
            )
        )
        result = await client.fetch(URL, user=ALICE)
        assert result.url == "https://zenodo.org/final.tsv"
        assert result.name == "final.tsv"
        assert result.content == content
        assert result.size_bytes == len(content)
        assert result.sha256 == hashlib.sha256(content).hexdigest()
        assert result.content_type == "text/csv; charset=utf-8"
        assert result.content_encoding == "identity"
        assert result.redirects == 2

    async def test_a_digest_that_does_not_describe_the_bytes_is_a_protocol_error(self):
        client, _ = _client(httpx.Response(200, json=_ok_body(sha256="0" * 64)))
        with pytest.raises(UrlFetchProtocolError, match="sha256"):
            await client.fetch(URL, user=ALICE)

    async def test_a_size_that_does_not_describe_the_bytes_is_a_protocol_error(self):
        client, _ = _client(httpx.Response(200, json=_ok_body(size_bytes=99)))
        with pytest.raises(UrlFetchProtocolError, match="declares 99"):
            await client.fetch(URL, user=ALICE)

    async def test_a_body_over_the_cap_the_fetcher_aborts_at_is_a_protocol_error(self):
        """`fetch._read_capped` aborts rather than truncating, so a larger body means the two
        ends disagree about MAX_BYTES — not that the file is big."""
        client, _ = _client(httpx.Response(200, json=_ok_body(b"x" * (MAX_FETCH_BYTES + 1))))
        with pytest.raises(UrlFetchProtocolError, match="cap"):
            await client.fetch(URL, user=ALICE)

    @pytest.mark.parametrize(
        "overrides",
        [
            {"name": None},
            {"sha256": 12},
            {"redirects": "0"},
            {"redirects": 9},
            {"size_bytes": -1},
            {"content_b64": "not base64!!"},
        ],
    )
    async def test_a_200_the_contract_does_not_describe_is_a_protocol_error(self, overrides):
        client, _ = _client(httpx.Response(200, json=_ok_body(**overrides)))
        with pytest.raises(UrlFetchProtocolError):
            await client.fetch(URL, user=ALICE)

    async def test_a_200_that_is_not_json_is_a_protocol_error(self):
        client, _ = _client(httpx.Response(200, content=b"<html>"))
        with pytest.raises(UrlFetchProtocolError):
            await client.fetch(URL, user=ALICE)

    @pytest.mark.parametrize(
        "name",
        [
            "../../../etc/passwd",
            "x" * (NAME_MAX + 1),
            "",
            ".bashrc",
            "data.tsv\nurl-fetcher fetched something-else.tsv",
        ],
    )
    async def test_a_name_fetch_filename_for_cannot_produce_is_a_protocol_error(self, name):
        """`filename_for` emits [A-Za-z0-9._-], strips leading dots and truncates at NAME_MAX,
        so anything else is a fetcher we did not mirror. It is the only success field that
        leaves here as something acted on — a file name, and a log line a newline could forge."""
        client, _ = _client(httpx.Response(200, json=_ok_body(name=name)))
        with pytest.raises(UrlFetchProtocolError, match="name"):
            await client.fetch(URL, user=ALICE)

    @pytest.mark.parametrize("name", ["café.txt", "日本.csv"])
    async def test_a_name_fetch_filename_for_can_produce_from_a_non_ascii_url_is_accepted(
        self, name
    ):
        """``str.isalnum`` is Unicode-aware, and that is the rule ``filename_for`` uses, so a
        name it legitimately emits from ``https://x/caf%C3%A9.txt`` must not be refused as a
        fetcher we did not mirror."""
        client, _ = _client(httpx.Response(200, json=_ok_body(name=name)))
        result = await client.fetch(URL, user=ALICE)
        assert result.name == name

    async def test_a_label_that_is_not_a_string_is_dropped_rather_than_carried(self):
        client, _ = _client(
            httpx.Response(200, json=_ok_body(content_type=7, content_encoding=["gzip"]))
        )
        result = await client.fetch(URL, user=ALICE)
        assert result.content_type is None
        assert result.content_encoding is None


class TestTheResponseBound:
    """Parsing and decoding a 200 before capping it costs a multiple of the bytes on the wire —
    measured at 61.5 MB resident for a 16 MiB payload, in a single-replica process with a 2Gi
    limit. So the frame is judged before the body, and the body is capped whatever it claims."""

    async def test_an_over_ceiling_content_length_is_refused_without_reading_the_body(self):
        response, body = _streamed_response(b"{}", content_length=MAX_RESPONSE_BYTES + 1)
        client, _ = _client(response)
        with pytest.raises(UrlFetchProtocolError, match="ceiling"):
            await client.fetch(URL, user=ALICE)
        assert body.read is False, "the body was taken in before the declared size was judged"

    async def test_a_missing_content_length_is_refused_without_reading_the_body(self):
        """`server._send` frames every answer on both routes, so an unframed one is not the
        fetcher's and reading it is unbounded work."""
        response, body = _streamed_response(b"{}", content_length=False)
        client, _ = _client(response)
        with pytest.raises(UrlFetchProtocolError, match="Content-Length"):
            await client.fetch(URL, user=ALICE)
        assert body.read is False

    async def test_an_unparseable_content_length_is_refused(self):
        response, _ = _streamed_response(b"{}", content_length="not-a-number")
        client, _ = _client(response)
        with pytest.raises(UrlFetchProtocolError, match="Content-Length"):
            await client.fetch(URL, user=ALICE)

    async def test_a_body_longer_than_it_declares_is_still_capped(self):
        """The header is the sender's claim about the body, not a limit on it."""
        response, body = _streamed_response(b"x" * (MAX_RESPONSE_BYTES + 1), content_length=10)
        client, _ = _client(response)
        with pytest.raises(UrlFetchProtocolError, match="ceiling"):
            await client.fetch(URL, user=ALICE)
        assert body.read is True

    async def test_the_ceiling_admits_the_largest_fetch_the_contract_allows(self):
        """Derived from MAX_FETCH_BYTES, so a file at the fetcher's own cap must still pass."""
        content = b"x" * MAX_FETCH_BYTES
        client, _ = _client(httpx.Response(200, json=_ok_body(content)))
        result = await client.fetch(URL, user=ALICE)
        assert result.size_bytes == MAX_FETCH_BYTES

    async def test_the_ceiling_admits_the_worst_legitimate_envelope(self):
        """The ceiling is a memory bound, not a re-derivation of the request cap: a redirect's
        final url is unbounded by ``fetch.py`` (``http.client`` allows a 65536-byte header) and
        ``server._send``'s ``json.dumps`` escapes non-ASCII at 6 bytes out per character. This
        builds the worst envelope those two facts allow and serialises it exactly as
        ``server._send`` does, to measure it against the ceiling rather than assume it fits."""
        content = b"x" * MAX_FETCH_BYTES
        body = _ok_body(
            content,
            url="https://example.org/" + "a" * 60_000,
            name="é" * 128,
            content_type='"' * 200,
            content_encoding='"' * 200,
        )
        serialised = json.dumps(body).encode("utf-8")
        print(f"worst-case envelope: {len(serialised)} bytes; ceiling: {MAX_RESPONSE_BYTES}")
        assert len(serialised) < MAX_RESPONSE_BYTES


class TestRefusalVersusTransient:
    """`error.retryable` is the discriminant, never the status code: 502 carries both."""

    @pytest.mark.parametrize(
        "status,error_type",
        [
            (403, "refused_by_policy"),
            (400, "invalid_request"),
            (413, "too_large"),
            (502, "unrequested_encoding"),
        ],
    )
    async def test_a_refusal_is_never_retryable(self, status, error_type):
        client, _ = _client(_error_response(status, error_type, "no", retryable=False))
        with pytest.raises(UrlFetchRefused) as excinfo:
            await client.fetch(URL, user=ALICE)
        assert excinfo.value.retryable is False
        assert excinfo.value.error_type == error_type
        assert excinfo.value.status_code == status

    @pytest.mark.parametrize(
        "status,error_type",
        [
            (504, "timed_out"),
            (502, "unreachable"),
            (502, "truncated"),
        ],
    )
    async def test_a_transient_failure_is_retryable(self, status, error_type):
        client, _ = _client(_error_response(status, error_type, "later", retryable=True))
        with pytest.raises(UrlFetchUnavailable) as excinfo:
            await client.fetch(URL, user=ALICE)
        assert excinfo.value.retryable is True

    async def test_the_same_status_splits_on_the_flag_alone(self):
        """A TLS failure and a connection lost mid-body are both `unreachable` 502s, and only
        one of them is worth asking again."""
        client, _ = _client(_error_response(502, "unreachable", "TLS failed", retryable=False))
        with pytest.raises(UrlFetchRefused):
            await client.fetch(URL, user=ALICE)

        client, _ = _client(_error_response(502, "unreachable", "mid-body", retryable=True))
        with pytest.raises(UrlFetchUnavailable):
            await client.fetch(URL, user=ALICE)

    async def test_an_upstream_status_carries_its_evidence(self):
        """`fetch._raise_upstream` sends the first bytes so an HTML error page served as a .tsv
        is recognisable as one. Dropping them loses the only diagnosis the user gets."""
        prefix = base64.b64encode(b"<html>404").decode("ascii")
        client, _ = _client(
            _error_response(
                502,
                "upstream_status",
                "host answered 404 (text/html)",
                retryable=False,
                upstream_status=404,
                upstream_content_type="text/html",
                upstream_body_prefix_b64=prefix,
            )
        )
        with pytest.raises(UrlFetchRefused) as excinfo:
            await client.fetch(URL, user=ALICE)
        assert excinfo.value.details["upstream_status"] == 404
        assert excinfo.value.details["upstream_body_prefix_b64"] == prefix

    async def test_the_refusal_message_reaches_the_caller(self):
        """A refusal names its policy so a blocked URL is a visible request to widen the
        allow-list; swallowing the message makes it invisible."""
        client, _ = _client(
            _error_response(403, "refused_by_policy", "example.com is not on the allow-list")
        )
        with pytest.raises(UrlFetchRefused, match="not on the allow-list"):
            await client.fetch(URL, user=ALICE)

    @pytest.mark.parametrize("status,expected", [(404, UrlFetchRefused), (503, UrlFetchUnavailable)])
    async def test_an_envelope_without_the_flag_falls_back_to_the_status(self, status, expected):
        """Something other than the fetcher answered — an ingress, a crash page. 5xx is the
        only shape worth a second attempt."""
        client, _ = _client(httpx.Response(status, content=b"<html>nope"))
        with pytest.raises(expected):
            await client.fetch(URL, user=ALICE)

    async def test_the_fetchers_own_internal_error_is_not_a_policy_refusal(self):
        """`server.do_POST` sends `internal_error` for a bug on its side: non-retryable, but
        not a decision about the request. Presented as a refusal it would tell the user their
        URL is not allowed when nothing was ever said about the URL."""
        client, _ = _client(
            _error_response(500, "internal_error", "the fetch failed inside the service: KeyError")
        )
        with pytest.raises(UrlFetchProtocolError) as excinfo:
            await client.fetch(URL, user=ALICE)
        assert not isinstance(excinfo.value, UrlFetchRefused)
        assert excinfo.value.error_type == "internal_error"
        assert excinfo.value.retryable is False

    @pytest.mark.parametrize("body", [[1, 2], {"error": "refused"}, "nope"])
    async def test_an_error_body_that_is_not_the_envelope_degrades_to_the_status(self, body):
        """Raising while parsing an error response turns a diagnosable failure into an
        undiagnosable one, so every departure from the shape degrades to 'nothing known'."""
        client, _ = _client(httpx.Response(503, json=body))
        with pytest.raises(UrlFetchUnavailable) as excinfo:
            await client.fetch(URL, user=ALICE)
        assert excinfo.value.error_type is None

    async def test_no_fetcher_at_all_is_transient(self):
        def boom(request):
            raise httpx.ConnectError("refused", request=request)

        client, _ = _client(boom)
        with pytest.raises(UrlFetchUnavailable) as excinfo:
            await client.fetch(URL, user=ALICE)
        assert excinfo.value.error_type == url_fetch_client.ERROR_FETCHER_UNREACHABLE

    async def test_a_timeout_talking_to_the_fetcher_is_transient(self):
        def slow(request):
            raise httpx.ReadTimeout("too slow", request=request)

        client, _ = _client(slow)
        with pytest.raises(UrlFetchUnavailable):
            await client.fetch(URL, user=ALICE)


class TestCrossUserIsolation:
    """The one security property of this module. Each test is driven as the failure."""

    async def test_one_users_fetch_is_never_served_to_another(self):
        content = b"alice's private file\n"
        client, recorder = _caching_client(httpx.Response(200, json=_ok_body(content)))
        await client.fetch(URL, user=ALICE)
        assert recorder.calls == 1

        await client.fetch(URL, user=BOB)
        assert recorder.calls == 2, "Bob was served Alice's cached bytes"

    async def test_the_key_is_a_tuple_so_no_separator_can_forge_a_collision(self):
        """A joined-string key makes a collision a question of what a user may put in their own
        identifier; a tuple has no separator to attack."""
        cache = FetchCache(ttl_s=900, max_bytes=MAX_FETCH_BYTES)
        result = url_fetch_client.FetchedFile(
            url=URL, name="d", content=b"x", size_bytes=1,
            sha256=hashlib.sha256(b"x").hexdigest(), content_type=None,
            content_encoding=None, redirects=0,
        )
        cache.put("a|b", "c", result)
        assert cache.get("a", "b|c") is None
        assert cache.get("a|b", "c") is result

    async def test_a_fetch_without_a_user_is_refused_before_the_wire(self):
        client, recorder = _caching_client()
        for missing in ("", "   ", None):
            with pytest.raises(UrlFetchRefused):
                await client.fetch(URL, user=missing)
        assert recorder.calls == 0

    async def test_two_spellings_of_one_identifier_are_two_users(self):
        """Trimming here made ` alice\\n` and `alice` one cache entry, while `download_store`,
        the other user-scoped store in this process, compares owners by plain equality. An
        identity this layer folds and that one does not is how one user's bytes reach another,
        so the key is the string exactly as given."""
        client, recorder = _caching_client()
        await client.fetch(URL, user="alice")
        await client.fetch(URL, user=" alice\n")
        assert recorder.calls == 2, "a trimmed identity was served another spelling's bytes"

        await client.fetch(URL, user="alice")
        await client.fetch(URL, user=" alice\n")
        assert recorder.calls == 2, "the two spellings are cached separately"

    async def test_the_same_user_is_served_from_the_cache(self):
        client, recorder = _caching_client()
        first = await client.fetch(URL, user=ALICE)
        second = await client.fetch(URL, user=ALICE)
        assert recorder.calls == 1
        assert second is first

    async def test_a_different_url_for_the_same_user_is_a_miss(self):
        client, recorder = _caching_client()
        await client.fetch(URL, user=ALICE)
        await client.fetch(OTHER_URL, user=ALICE)
        assert recorder.calls == 2


class TestTheCacheBounds:
    async def test_an_entry_past_the_ttl_is_not_served(self):
        clock = _Clock()
        client, recorder = _caching_client(ttl_s=60, clock=clock)
        await client.fetch(URL, user=ALICE)
        clock.advance(61)
        await client.fetch(URL, user=ALICE)
        assert recorder.calls == 2, "an expired entry was served"

    async def test_an_entry_inside_the_ttl_is_served(self):
        clock = _Clock()
        client, recorder = _caching_client(ttl_s=60, clock=clock)
        await client.fetch(URL, user=ALICE)
        clock.advance(59)
        await client.fetch(URL, user=ALICE)
        assert recorder.calls == 1

    async def test_an_expired_entry_releases_its_bytes(self):
        clock = _Clock()
        cache = FetchCache(ttl_s=60, max_bytes=1000, clock=clock)
        client, _ = _client(httpx.Response(200, json=_ok_body(b"x" * 100)), cache=cache)
        await client.fetch(URL, user=ALICE)
        assert cache.total_bytes == 100
        clock.advance(61)
        assert cache.get(ALICE, URL) is None
        assert cache.total_bytes == 0

    async def test_the_size_bound_evicts_the_oldest_entry(self):
        cache = FetchCache(ttl_s=900, max_bytes=250)
        client, recorder = _client(
            httpx.Response(200, json=_ok_body(b"a" * 100)),
            httpx.Response(200, json=_ok_body(b"b" * 100)),
            httpx.Response(200, json=_ok_body(b"c" * 100)),
            httpx.Response(200, json=_ok_body(b"a" * 100)),
            cache=cache,
        )
        await client.fetch(URL, user=ALICE)
        await client.fetch(OTHER_URL, user=ALICE)
        assert cache.total_bytes == 200
        await client.fetch("https://github.com/x/third.tsv", user=ALICE)

        assert cache.total_bytes <= 250
        assert len(cache) == 2
        await client.fetch(URL, user=ALICE)
        assert recorder.calls == 4, "the oldest entry survived the size bound"

    async def test_an_entry_larger_than_the_whole_bound_is_not_stored(self):
        """Storing it would evict every other user's entries to hold one file."""
        cache = FetchCache(ttl_s=900, max_bytes=50)
        client, recorder = _client(httpx.Response(200, json=_ok_body(b"x" * 100)), cache=cache)
        await client.fetch(URL, user=ALICE)
        assert len(cache) == 0
        await client.fetch(URL, user=ALICE)
        assert recorder.calls == 2

    @pytest.mark.parametrize("ttl_s,max_bytes", [(0, 1000), (900, 0), (-1, -1)])
    async def test_a_non_positive_bound_disables_the_cache(self, ttl_s, max_bytes):
        cache = FetchCache(ttl_s=ttl_s, max_bytes=max_bytes)
        client, recorder = _client(cache=cache)
        await client.fetch(URL, user=ALICE)
        await client.fetch(URL, user=ALICE)
        assert recorder.calls == 2
        assert len(cache) == 0

    async def test_the_default_client_caches_nothing(self):
        """Constructed without a cache, the client must still work — and must not accumulate."""
        client, recorder = _client()
        await client.fetch(URL, user=ALICE)
        await client.fetch(URL, user=ALICE)
        assert recorder.calls == 2


class TestWhatIsNeverCached:
    @pytest.mark.parametrize(
        "response",
        [
            _error_response(403, "refused_by_policy", "not allow-listed"),
            _error_response(504, "timed_out", "slow", retryable=True),
        ],
    )
    async def test_a_failure_is_never_cached(self, response):
        """A negative entry would make a refusal survive the config change that widens the
        allow-list, and a blip survive the upstream's recovery."""
        cache = FetchCache(ttl_s=900, max_bytes=MAX_FETCH_BYTES)
        client, recorder = _client(response, httpx.Response(200, json=_ok_body()), cache=cache)
        with pytest.raises(url_fetch_client.UrlFetchError):
            await client.fetch(URL, user=ALICE)
        assert len(cache) == 0

        assert (await client.fetch(URL, user=ALICE)).name == "data.tsv"
        assert recorder.calls == 2

    async def test_a_protocol_error_is_never_cached(self):
        cache = FetchCache(ttl_s=900, max_bytes=MAX_FETCH_BYTES)
        client, _ = _client(httpx.Response(200, json=_ok_body(sha256="0" * 64)), cache=cache)
        with pytest.raises(UrlFetchProtocolError):
            await client.fetch(URL, user=ALICE)
        assert len(cache) == 0

    async def test_the_cache_is_keyed_on_the_requested_url_not_the_final_one(self):
        """The 200 reports the url after redirects. Keying on that would make a second ask for
        the ORIGINAL url a miss forever, and would let one redirect target answer for a url the
        guard has not been asked about."""
        cache = FetchCache(ttl_s=900, max_bytes=MAX_FETCH_BYTES)
        client, recorder = _client(
            httpx.Response(200, json=_ok_body(url="https://zenodo.org/after-redirect.tsv",
                                              redirects=1)),
            cache=cache,
        )
        await client.fetch(URL, user=ALICE)
        await client.fetch(URL, user=ALICE)
        assert recorder.calls == 1
        assert cache.get(ALICE, "https://zenodo.org/after-redirect.tsv") is None


class TestConfiguration:
    @pytest.fixture(autouse=True)
    def _clean(self, monkeypatch):
        settings_module.get_settings.cache_clear()
        reset_url_fetch_client()
        yield
        settings_module.get_settings.cache_clear()
        reset_url_fetch_client()

    def test_an_unconfigured_fetcher_refuses_to_guess(self, monkeypatch):
        monkeypatch.setenv("URL_FETCHER_URL", "")
        with pytest.raises(UrlFetchNotConfigured):
            UrlFetchClient()

    def test_the_singleton_reads_the_configured_bounds(self, monkeypatch):
        monkeypatch.setenv("URL_FETCHER_URL", "http://url-fetcher:8090/")
        monkeypatch.setenv("URL_FETCH_CACHE_TTL_SECONDS", "42")
        monkeypatch.setenv("URL_FETCH_CACHE_MAX_BYTES", "4096")
        client = get_url_fetch_client()
        assert client is get_url_fetch_client()
        assert client.base_url == "http://url-fetcher:8090"
        assert client.cache._ttl_s == 42
        assert client.cache._max_bytes == 4096

    def test_the_unconfigured_failure_is_not_memoised(self, monkeypatch):
        monkeypatch.setenv("URL_FETCHER_URL", "")
        with pytest.raises(UrlFetchNotConfigured):
            get_url_fetch_client()
        monkeypatch.setenv("URL_FETCHER_URL", "http://url-fetcher:8090")
        settings_module.get_settings.cache_clear()
        assert get_url_fetch_client().base_url == "http://url-fetcher:8090"


class TestHealth:
    async def test_healthy_is_true_only_on_200(self):
        client, _ = _client(httpx.Response(200, json={"status": "ok"}))
        assert await client.healthy() is True

        client, _ = _client(httpx.Response(503, json={"error": {"type": "x"}}))
        assert await client.healthy() is False

    async def test_healthy_never_raises(self):
        def boom(request):
            raise httpx.ConnectError("refused", request=request)

        client, _ = _client(boom)
        assert await client.healthy() is False


class TestTheAllowList:
    """`allowed_hosts()` — the prompt's source for which hosts a URL input can come from."""

    async def test_the_list_is_read_once_and_cached(self):
        client, recorder = _client(
            httpx.Response(200, json={"status": "ok", "allowed_hosts": ["a.example", "b.example"]})
        )
        assert await client.allowed_hosts() == ("a.example", "b.example")
        assert await client.allowed_hosts() == ("a.example", "b.example")
        assert recorder.calls == 1

    async def test_an_absent_field_is_unknown_rather_than_empty(self):
        client, _ = _client(httpx.Response(200, json={"status": "ok"}))
        assert await client.allowed_hosts() is None

    async def test_an_unreachable_fetcher_is_none(self):
        def boom(request):
            raise httpx.ConnectError("refused", request=request)

        client, _ = _client(boom)
        assert await client.allowed_hosts() is None

    async def test_a_non_200_is_none(self):
        client, _ = _client(httpx.Response(503, json={"error": {"type": "x"}}))
        assert await client.allowed_hosts() is None

    @pytest.mark.parametrize(
        "body",
        [
            {"allowed_hosts": "a.example"},
            {"allowed_hosts": ["a.example", 7]},
            {"allowed_hosts": {"a.example": True}},
            {"allowed_hosts": None},
            ["a.example"],
        ],
    )
    async def test_a_malformed_list_is_none(self, body):
        client, _ = _client(httpx.Response(200, json=body))
        assert await client.allowed_hosts() is None

    async def test_a_body_that_is_not_json_is_none(self):
        client, _ = _client(
            httpx.Response(200, content=b"not json", headers={"Content-Type": "application/json"})
        )
        assert await client.allowed_hosts() is None

    async def test_a_failure_is_not_cached(self):
        """The fetcher can come up after this process did; only a success is remembered."""
        client, recorder = _client(
            httpx.Response(200, json={"status": "ok"}),
            httpx.Response(200, json={"status": "ok", "allowed_hosts": ["a.example"]}),
        )
        assert await client.allowed_hosts() is None
        assert await client.allowed_hosts() == ("a.example",)
        assert recorder.calls == 2

    async def test_an_empty_list_is_an_answer_and_is_cached(self):
        """A fetcher that can reach nothing is knowable; only "cannot say" is None."""
        client, recorder = _client(httpx.Response(200, json={"status": "ok", "allowed_hosts": []}))
        assert await client.allowed_hosts() == ()
        assert await client.allowed_hosts() == ()
        assert recorder.calls == 1
