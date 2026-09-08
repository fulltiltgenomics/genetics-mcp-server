"""The arm preflight exists to stop a run that would measure the wrong surface.

Three ways a run can look fine and mean nothing, one test each: a misspelled arm
(`code_execution_requested` coerces it to the no-code surface rather than raising), an arm
that is not the surface it names (the flag subtraction runs after the surface resolves, so a
`code` arm can arrive without `run_analysis`), and two arms that resolve alike (every profile
name except `code` selects one surface). Plus the two outcomes that are deliberately NOT
fatal: an older server without the endpoint, and one arm failing to resolve.
"""

import httpx
import pytest

from genetics_mcp_server.scripts.replay_benchmark import (
    ALL_TOOLS_ARM,
    CODE_ARM,
    CODE_EXECUTION_TOOL,
    NOCODE_ARM,
    ArmResolutionError,
    build_parser,
    resolve_arm_tools,
)
from genetics_mcp_server.tools.definitions import code_execution_requested, resolve_tools


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _names(count, *, code):
    """A resolved name list of `count` names, with or without the code-execution tool."""
    filler = [f"t{i}" for i in range(count - (1 if code else 0))]
    return ([CODE_EXECUTION_TOOL] + filler) if code else filler


def _ok(profile, count, known=True, names=None):
    if names is None:
        names = _names(count, code=profile == CODE_ARM)
    return httpx.Response(
        200,
        json={
            "tool_profile": profile,
            "known_profile": known,
            "count": count,
            "names": names,
        },
    )


def test_the_defaults_are_the_two_arms_that_straddle_the_boolean():
    args = build_parser().parse_args([])
    assert (args.arm_a, args.arm_b) == (NOCODE_ARM, CODE_ARM)


def test_the_harnesss_own_literals_still_describe_this_servers_surfaces():
    # the harness keeps its own copies because it reads a REMOTE server; this is what stops
    # them drifting from the build they were written against
    assert code_execution_requested(CODE_ARM) is True
    assert code_execution_requested(NOCODE_ARM) is False
    assert CODE_EXECUTION_TOOL in {t["name"] for t in resolve_tools(code_execution=True)}
    assert CODE_EXECUTION_TOOL not in {t["name"] for t in resolve_tools(code_execution=False)}


@pytest.mark.asyncio
async def test_an_unknown_profile_aborts_before_anything_is_spent():
    def handler(request):
        profile = request.url.params.get("tool_profile")
        # the server's real behaviour for a typo: served, not an error, just wrong
        return _ok(profile, 18, known=False) if profile == "nocod" else _ok(profile, 18)

    async with _client(handler) as client:
        with pytest.raises(ArmResolutionError) as exc:
            await resolve_arm_tools(client, "http://x", ("nocod", CODE_ARM))

    assert "nocod" in str(exc.value)
    assert "18" in str(exc.value), "the message must say what it would have measured instead"


@pytest.mark.asyncio
async def test_a_code_arm_without_run_analysis_aborts_the_run():
    # SANDBOX_ENABLED=false subtracts run_analysis AFTER the profile resolves, so the name is
    # recognised and the arm is still not the surface it claims
    def handler(request):
        profile = request.url.params.get("tool_profile")
        names = None if profile == NOCODE_ARM else _names(17, code=False)
        return _ok(profile, 64 if profile == NOCODE_ARM else 17, names=names)

    async with _client(handler) as client:
        with pytest.raises(ArmResolutionError) as exc:
            await resolve_arm_tools(client, "http://x", (NOCODE_ARM, CODE_ARM))

    message = str(exc.value)
    assert CODE_ARM in message and CODE_EXECUTION_TOOL in message
    assert "SANDBOX_ENABLED" in message, "the message must name the usual cause"


@pytest.mark.asyncio
async def test_a_baseline_arm_carrying_run_analysis_aborts_the_run():
    # a server predating the collapse: every profile there carried the code-execution tool,
    # so both arms could run scripts and the comparison is not old against new
    def handler(request):
        return _ok(request.url.params.get("tool_profile"), 65, names=_names(65, code=True))

    async with _client(handler) as client:
        with pytest.raises(ArmResolutionError) as exc:
            await resolve_arm_tools(client, "http://x", (NOCODE_ARM, CODE_ARM))

    assert NOCODE_ARM in str(exc.value) and CODE_EXECUTION_TOOL in str(exc.value)


@pytest.mark.asyncio
async def test_known_arms_are_recorded_with_their_counts_and_names():
    def handler(request):
        profile = request.url.params.get("tool_profile")
        return _ok(profile, 64 if profile == NOCODE_ARM else 18)

    async with _client(handler) as client:
        out = await resolve_arm_tools(client, "http://x", (NOCODE_ARM, CODE_ARM))

    assert out[NOCODE_ARM]["count"] == 64
    assert out[CODE_ARM]["count"] == 18
    assert len(out[NOCODE_ARM]["names"]) == 64, "names are recorded, not just the count"


@pytest.mark.asyncio
async def test_the_all_arm_is_sent_as_no_profile_not_as_the_literal_string():
    # "all" is the harness's spelling for tool_profile: null; sending it verbatim would hit
    # the very fallback this preflight exists to catch
    seen = []

    def handler(request):
        profile = request.url.params.get("tool_profile")
        seen.append(profile)
        # distinct surfaces, or the identical-surface refusal would mask what this asserts
        return _ok(profile, 64 if profile is None else 18)

    async with _client(handler) as client:
        await resolve_arm_tools(client, "http://x", (ALL_TOOLS_ARM, CODE_ARM))

    assert seen[0] is None, f"the all arm must send no tool_profile param, sent {seen[0]!r}"


@pytest.mark.asyncio
async def test_a_server_without_the_endpoint_warns_rather_than_failing_the_run():
    async with _client(lambda request: httpx.Response(404)) as client:
        out = await resolve_arm_tools(client, "http://x", (NOCODE_ARM, CODE_ARM))
    assert "unavailable" in out, "an older server loses the proof, not the run"


@pytest.mark.asyncio
async def test_a_transport_error_on_one_arm_does_not_abort_the_run():
    def handler(request):
        if request.url.params.get("tool_profile") == NOCODE_ARM:
            raise httpx.ConnectError("boom")
        return _ok(CODE_ARM, 18)

    async with _client(handler) as client:
        out = await resolve_arm_tools(client, "http://x", (NOCODE_ARM, CODE_ARM))

    assert "error" in out[NOCODE_ARM], "the failure is recorded"
    assert out[CODE_ARM]["count"] == 18, "and the other arm still resolves"


@pytest.mark.asyncio
async def test_two_arms_that_resolve_to_one_surface_abort_the_run():
    # `code` twice passes the per-arm check and is still a surface compared against itself
    def handler(request):
        return _ok(request.url.params.get("tool_profile"), 18, names=_names(18, code=True))

    async with _client(handler) as client:
        with pytest.raises(ArmResolutionError) as exc:
            await resolve_arm_tools(client, "http://x", (CODE_ARM, CODE_ARM))

    message = str(exc.value)
    assert CODE_ARM in message, "the arms are named"
    assert "18" in message, "and the surface they share is quantified"


@pytest.mark.asyncio
async def test_the_null_arm_and_nocode_are_two_names_for_one_surface():
    def handler(request):
        return _ok(request.url.params.get("tool_profile"), 64)

    async with _client(handler) as client:
        with pytest.raises(ArmResolutionError) as exc:
            await resolve_arm_tools(client, "http://x", (ALL_TOOLS_ARM, NOCODE_ARM))

    assert ALL_TOOLS_ARM in str(exc.value) and NOCODE_ARM in str(exc.value)


@pytest.mark.asyncio
async def test_arms_that_differ_by_one_tool_are_not_treated_as_one_surface():
    # the guard compares name sets, not counts: two surfaces of equal size are still two
    def handler(request):
        profile = request.url.params.get("tool_profile")
        names = ["a", "b"] if profile is None else [CODE_EXECUTION_TOOL, "a"]
        return _ok(profile, len(names), names=names)

    async with _client(handler) as client:
        out = await resolve_arm_tools(client, "http://x", (ALL_TOOLS_ARM, CODE_ARM))

    assert out[CODE_ARM]["names"] == [CODE_EXECUTION_TOOL, "a"]


def test_a_refused_pair_exits_2_on_the_paid_path_too(tmp_path, monkeypatch, capsys):
    """The preflight is reached from main() on the real path, not only under --dry-run.

    Raising out of run_benchmark exits 1 with a traceback and no ERROR line, which is
    neither what the runbook promises nor something a caller can act on.
    """
    import json

    import genetics_mcp_server.scripts.replay_benchmark as rb

    dataset = tmp_path / "cases.json"
    dataset.write_text(json.dumps([{"session_id": "a", "user_turns": ["q"]}]))

    async def refuse(client, base_url, arms):
        raise ArmResolutionError("arms 'x' and 'y' both resolve to the same surface")

    async def spent(**kwargs):
        raise AssertionError("a refused pair must not reach the chat service")

    monkeypatch.setattr(rb, "resolve_arm_tools", refuse)
    monkeypatch.setattr(rb, "replay_case", spent)

    code = rb.main(
        [
            "--dataset",
            str(dataset),
            "--base-url",
            "http://x",
            "--arm-a",
            NOCODE_ARM,
            "--arm-b",
            CODE_ARM,
        ]
    )

    assert code == 2
    assert "ERROR: arms" in capsys.readouterr().err
