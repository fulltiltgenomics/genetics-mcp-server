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
    parse_arm,
    resolve_arm_tools,
)
from genetics_mcp_server.tools.definitions import code_execution_requested, resolve_tools


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _arms(*specs, base_url="http://x"):
    """Arms from bare profile names, all on one server — the single-server shape."""
    return tuple(parse_arm(s, base_url) for s in specs)


def _names(count, *, code):
    """A resolved name list of `count` names, with or without the code-execution tool."""
    filler = [f"t{i}" for i in range(count - (1 if code else 0))]
    return ([CODE_EXECUTION_TOOL] + filler) if code else filler


def _ok(profile, count, known=True, names=None, variant="current"):
    if names is None:
        names = _names(count, code=profile == CODE_ARM)
    body = {
        "tool_profile": profile,
        "known_profile": known,
        "count": count,
        "names": names,
    }
    if variant is not None:
        body["prompt_variant"] = variant
    return httpx.Response(200, json=body)


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
            await resolve_arm_tools(client, _arms("nocod", CODE_ARM))

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
            await resolve_arm_tools(client, _arms(NOCODE_ARM, CODE_ARM))

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
            await resolve_arm_tools(client, _arms(NOCODE_ARM, CODE_ARM))

    assert NOCODE_ARM in str(exc.value) and CODE_EXECUTION_TOOL in str(exc.value)


@pytest.mark.asyncio
async def test_known_arms_are_recorded_with_their_counts_and_names():
    def handler(request):
        profile = request.url.params.get("tool_profile")
        return _ok(profile, 64 if profile == NOCODE_ARM else 18)

    async with _client(handler) as client:
        out = await resolve_arm_tools(client, _arms(NOCODE_ARM, CODE_ARM))

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
        await resolve_arm_tools(client, _arms(ALL_TOOLS_ARM, CODE_ARM))

    assert seen[0] is None, f"the all arm must send no tool_profile param, sent {seen[0]!r}"


@pytest.mark.asyncio
async def test_a_server_without_the_endpoint_warns_rather_than_failing_the_run():
    async with _client(lambda request: httpx.Response(404)) as client:
        out = await resolve_arm_tools(client, _arms(NOCODE_ARM, CODE_ARM))

    # recorded PER ARM rather than as one whole-run verdict: with an arm's base URL now
    # its own, one server too old to answer must not discard the proof the other can
    # still give
    assert all("unavailable" in out[arm] for arm in (NOCODE_ARM, CODE_ARM)), (
        "an older server loses the proof, not the run"
    )


@pytest.mark.asyncio
async def test_a_transport_error_on_one_arm_does_not_abort_the_run():
    def handler(request):
        if request.url.params.get("tool_profile") == NOCODE_ARM:
            raise httpx.ConnectError("boom")
        return _ok(CODE_ARM, 18)

    async with _client(handler) as client:
        out = await resolve_arm_tools(client, _arms(NOCODE_ARM, CODE_ARM))

    assert "error" in out[NOCODE_ARM], "the failure is recorded"
    assert out[CODE_ARM]["count"] == 18, "and the other arm still resolves"


@pytest.mark.asyncio
async def test_two_arms_that_resolve_to_one_surface_on_one_server_abort_the_run():
    # `code` twice passes the per-arm check and is still a surface compared against itself
    def handler(request):
        return _ok(request.url.params.get("tool_profile"), 18, names=_names(18, code=True))

    async with _client(handler) as client:
        with pytest.raises(ArmResolutionError) as exc:
            # same server, so this cannot be a prompt A/B either
            await resolve_arm_tools(client, _arms(CODE_ARM, CODE_ARM))

    message = str(exc.value)
    assert CODE_ARM in message, "the arms are named"
    assert "18" in message, "and the surface they share is quantified"


@pytest.mark.asyncio
async def test_the_null_arm_and_nocode_are_two_names_for_one_surface():
    def handler(request):
        return _ok(request.url.params.get("tool_profile"), 64)

    async with _client(handler) as client:
        with pytest.raises(ArmResolutionError) as exc:
            await resolve_arm_tools(client, _arms(ALL_TOOLS_ARM, NOCODE_ARM))

    assert ALL_TOOLS_ARM in str(exc.value) and NOCODE_ARM in str(exc.value)


@pytest.mark.asyncio
async def test_arms_that_differ_by_one_tool_are_not_treated_as_one_surface():
    # the guard compares name sets, not counts: two surfaces of equal size are still two
    def handler(request):
        profile = request.url.params.get("tool_profile")
        names = ["a", "b"] if profile is None else [CODE_EXECUTION_TOOL, "a"]
        return _ok(profile, len(names), names=names)

    async with _client(handler) as client:
        out = await resolve_arm_tools(client, _arms(ALL_TOOLS_ARM, CODE_ARM))

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

    async def refuse(client, arms):
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


# ---------------------------------------------------------------------------
# The prompt-variant dimension: two arms that differ by which SERVER, and so by
# which system prompt, rather than by tool surface.
# ---------------------------------------------------------------------------

A_URL = "http://a:8000"
B_URL = "http://b:8001"


def _two_servers(variant_a="current", variant_b="candidate", names=None):
    """A handler serving the same surface from two hosts under different variants."""

    def handler(request):
        profile = request.url.params.get("tool_profile")
        variant = variant_a if request.url.host == "a" else variant_b
        return _ok(profile, 20, names=names or _names(20, code=True), variant=variant)

    return handler


def test_an_arm_spec_carries_a_profile_a_url_and_an_optional_label():
    bare = parse_arm(CODE_ARM, A_URL)
    assert (bare.label, bare.profile, bare.base_url) == (CODE_ARM, CODE_ARM, A_URL)

    # no label and an explicit URL: the netloc keeps two same-profile arms apart
    urled = parse_arm(f"{CODE_ARM}@{B_URL}", A_URL)
    assert urled.label == f"{CODE_ARM}@b:8001" and urled.base_url == B_URL

    labelled = parse_arm(f"candidate={CODE_ARM}@{B_URL}/", A_URL)
    assert (labelled.label, labelled.profile, labelled.base_url) == (
        "candidate",
        CODE_ARM,
        B_URL,
    ), "a trailing slash is stripped so the URL joins cleanly"

    # `all` is the one profile that goes on the wire as null rather than as its own name
    assert parse_arm(ALL_TOOLS_ARM, A_URL).tool_profile is None
    assert parse_arm(CODE_ARM, A_URL).tool_profile == CODE_ARM


@pytest.mark.asyncio
async def test_one_surface_on_two_servers_is_allowed_when_the_prompts_differ():
    """The comparison the old identical-surface guard would have refused outright."""
    arms = _arms(f"current={CODE_ARM}@{A_URL}", f"candidate={CODE_ARM}@{B_URL}")

    async with _client(_two_servers()) as client:
        out = await resolve_arm_tools(client, arms)

    assert out["current"]["prompt_variant"] == "current"
    assert out["candidate"]["prompt_variant"] == "candidate"
    assert out["current"]["names"] == out["candidate"]["names"], (
        "identical tools is the POINT here — only the prompt is under test"
    )
    assert out["current"]["base_url"] == A_URL and out["candidate"]["base_url"] == B_URL


@pytest.mark.asyncio
async def test_two_servers_serving_one_variant_abort_the_run():
    """The same failure as the identical-surface guard, wearing different clothes.

    PROMPT_VARIANT coerces an unknown name to the default rather than raising, so a typo
    on one process leaves both arms on the default prompt and the run measures a prompt
    against itself.
    """
    arms = _arms(f"current={CODE_ARM}@{A_URL}", f"candidate={CODE_ARM}@{B_URL}")

    async with _client(_two_servers(variant_b="current")) as client:
        with pytest.raises(ArmResolutionError) as exc:
            await resolve_arm_tools(client, arms)

    message = str(exc.value)
    assert "current" in message and "candidate" in message, "both arms are named"
    assert A_URL in message and B_URL in message, "and so are the servers"
    assert "PROMPT_VARIANT" in message, "with the usual cause pointed at"


@pytest.mark.asyncio
async def test_a_server_too_old_to_report_its_variant_warns_rather_than_failing(caplog):
    """Unprovable is not the same as equal, and refusing would be the wrong trade."""
    arms = _arms(f"current={CODE_ARM}@{A_URL}", f"candidate={CODE_ARM}@{B_URL}")

    with caplog.at_level("WARNING"):
        async with _client(_two_servers(variant_b=None)) as client:
            out = await resolve_arm_tools(client, arms)

    assert out["candidate"]["prompt_variant"] is None
    assert "CANNOT prove" in caplog.text


@pytest.mark.asyncio
async def test_two_prompts_on_two_surfaces_warns_that_the_cause_is_ambiguous(caplog):
    """Legal, but it is no longer a prompt A/B and nothing else would say so."""

    def handler(request):
        profile = request.url.params.get("tool_profile")
        if request.url.host == "a":
            return _ok(profile, 2, names=["a", "b"], variant="current")
        return _ok(profile, 2, names=[CODE_EXECUTION_TOOL, "a"], variant="candidate")

    arms = _arms(f"current={NOCODE_ARM}@{A_URL}", f"candidate={CODE_ARM}@{B_URL}")

    with caplog.at_level("WARNING"):
        async with _client(handler) as client:
            out = await resolve_arm_tools(client, arms)

    assert out["candidate"]["prompt_variant"] == "candidate"
    assert "cannot be attributed to the prompt alone" in caplog.text


def test_two_arms_with_the_same_spec_are_refused_before_the_dataset_is_read(capsys):
    """Equal labels mean equal specs: `parse_arm` gives same-profile arms distinct labels
    as soon as their URLs differ, so this can only fire on a genuine duplicate."""
    import genetics_mcp_server.scripts.replay_benchmark as rb

    code = rb.main(
        ["--dataset", "/nonexistent.json", "--arm-a", CODE_ARM, "--arm-b", CODE_ARM]
    )

    assert code == 2
    err = capsys.readouterr().err
    assert "must differ" in err
    assert "prompt variants" in err, "and the two-server form is offered as the way out"
