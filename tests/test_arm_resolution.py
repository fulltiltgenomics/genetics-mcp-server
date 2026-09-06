"""The arm preflight exists to stop a run that would measure the wrong surface.

`get_anthropic_tools` resolves an unrecognised profile to the no-code surface rather than
raising, so a misspelled arm produces a plausible-looking benchmark against the wrong tools.
Every profile name except `code` now selects that same surface, so two correctly spelled
arms can also be one surface. These pin the outcomes: refuse a misspelling, refuse two arms
that resolve alike, record what was resolved, and degrade-with-a-warning.
"""

import httpx
import pytest

from genetics_mcp_server.scripts.replay_benchmark import (
    ALL_TOOLS_ARM,
    ArmResolutionError,
    resolve_arm_tools,
)


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _ok(profile, count, known=True, names=None):
    return httpx.Response(
        200,
        json={
            "tool_profile": profile,
            "known_profile": known,
            "count": count,
            "names": names if names is not None else [f"t{i}" for i in range(count)],
        },
    )


@pytest.mark.asyncio
async def test_an_unknown_profile_aborts_before_anything_is_spent():
    def handler(request):
        profile = request.url.params.get("tool_profile")
        # the server's real behaviour for a typo: served, not an error, just wrong
        return _ok(profile, 18, known=False) if profile == "nocod" else _ok(profile, 7)

    async with _client(handler) as client:
        with pytest.raises(ArmResolutionError) as exc:
            await resolve_arm_tools(client, "http://x", ("nocod", "code"))

    assert "nocod" in str(exc.value)
    assert "18" in str(exc.value), "the message must say what it would have measured instead"


@pytest.mark.asyncio
async def test_known_arms_are_recorded_with_their_counts_and_names():
    def handler(request):
        profile = request.url.params.get("tool_profile")
        return _ok(profile, 62 if profile == "nocode" else 7)

    async with _client(handler) as client:
        out = await resolve_arm_tools(client, "http://x", ("nocode", "code"))

    assert out["nocode"]["count"] == 62
    assert out["code"]["count"] == 7
    assert len(out["nocode"]["names"]) == 62, "names are recorded, not just the count"


@pytest.mark.asyncio
async def test_the_all_arm_is_sent_as_no_profile_not_as_the_literal_string():
    # "all" is the harness's spelling for tool_profile: null; sending it verbatim would hit
    # the very fallback this preflight exists to catch
    seen = []

    def handler(request):
        profile = request.url.params.get("tool_profile")
        seen.append(profile)
        # distinct counts so the two arms do not resolve to one surface, which is its own
        # refusal and would mask what this test is asserting
        return _ok(profile, 64 if profile is None else 18)

    async with _client(handler) as client:
        await resolve_arm_tools(client, "http://x", (ALL_TOOLS_ARM, "code"))

    assert seen[0] is None, f"the all arm must send no tool_profile param, sent {seen[0]!r}"


@pytest.mark.asyncio
async def test_a_server_without_the_endpoint_warns_rather_than_failing_the_run():
    async with _client(lambda request: httpx.Response(404)) as client:
        out = await resolve_arm_tools(client, "http://x", ("nocode", "code"))
    assert "unavailable" in out, "an older server loses the proof, not the run"


@pytest.mark.asyncio
async def test_a_transport_error_on_one_arm_does_not_abort_the_run():
    def handler(request):
        if request.url.params.get("tool_profile") == "nocode":
            raise httpx.ConnectError("boom")
        return _ok("code", 7)

    async with _client(handler) as client:
        out = await resolve_arm_tools(client, "http://x", ("nocode", "code"))

    assert "error" in out["nocode"], "the failure is recorded"
    assert out["code"]["count"] == 7, "and the other arm still resolves"


@pytest.mark.asyncio
async def test_two_arms_that_resolve_to_one_surface_abort_the_run():
    # the shipped defaults are exactly this shape: `all` and `bigquery` are both the
    # no-code surface, so the run would report a zero difference that means nothing
    def handler(request):
        return _ok(request.url.params.get("tool_profile"), 64)

    async with _client(handler) as client:
        with pytest.raises(ArmResolutionError) as exc:
            await resolve_arm_tools(client, "http://x", (ALL_TOOLS_ARM, "bigquery"))

    message = str(exc.value)
    assert ALL_TOOLS_ARM in message and "bigquery" in message, "both arms are named"
    assert "64" in message, "and the surface they share is quantified"


@pytest.mark.asyncio
async def test_arms_that_differ_by_one_tool_are_not_treated_as_one_surface():
    # the guard compares name sets, not counts: two surfaces of equal size are still two
    def handler(request):
        profile = request.url.params.get("tool_profile")
        names = ["a", "b"] if profile is None else ["a", "c"]
        return _ok(profile, len(names), names=names)

    async with _client(handler) as client:
        out = await resolve_arm_tools(client, "http://x", (ALL_TOOLS_ARM, "code"))

    assert out["code"]["names"] == ["a", "c"]
