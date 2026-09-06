"""An unrecognised tool_profile is loud to an operator, exactly once per distinct value.

genetics-results-suite-4h6.74: the browser and the server each enumerate the profiles and
nothing pins them together, so a server-side rename leaves the browser offering a dead name
that degrades to general-only. The degrade stays (stored rows from older clients depend on
it); what changes is that it is no longer invisible.

Once per DISTINCT VALUE is the load-bearing half. The value is persisted per message, so a
session that stored a dead profile re-sends it on every turn — a per-request warning would
be a flood, and a flooded warning is one nobody reads.
"""

import logging

import pytest

from genetics_mcp_server.tools import definitions
from genetics_mcp_server.tools.definitions import (
    code_execution_requested,
    get_anthropic_tools,
)


@pytest.fixture(autouse=True)
def _clear_warned_profiles():
    definitions._WARNED_UNKNOWN_PROFILES.clear()
    yield
    definitions._WARNED_UNKNOWN_PROFILES.clear()


def _warnings(caplog):
    return [r for r in caplog.records if r.levelno == logging.WARNING]


def test_unknown_profile_warns_naming_the_value_and_the_known_set(caplog):
    with caplog.at_level(logging.WARNING, logger=definitions.__name__):
        code_execution_requested("cdoe")

    records = _warnings(caplog)
    assert len(records) == 1
    message = records[0].getMessage()
    assert "cdoe" in message
    # what it RESOLVED TO has to be in the warning, not just that the value was unknown:
    # the degrade is silent to the caller, so this line is the only place an operator
    # learns the turn ran without code execution
    assert "no-code" in message
    # the known set has to be IN the warning too: "unknown profile" alone does not tell an
    # operator whether the browser or the server is the side that drifted
    for known in definitions.KNOWN_TOOL_PROFILES:
        assert known in message


def test_the_same_unknown_value_warns_only_once(caplog):
    with caplog.at_level(logging.WARNING, logger=definitions.__name__):
        for _ in range(25):
            code_execution_requested("cdoe")

    assert len(_warnings(caplog)) == 1


def test_a_second_distinct_unknown_value_still_warns(caplog):
    with caplog.at_level(logging.WARNING, logger=definitions.__name__):
        code_execution_requested("cdoe")
        code_execution_requested("bigqeury")

    assert {"cdoe", "bigqeury"} <= {r.getMessage().split("'")[1] for r in _warnings(caplog)}


def test_known_profiles_and_no_profile_stay_quiet(caplog):
    with caplog.at_level(logging.WARNING, logger=definitions.__name__):
        for profile in (None, *definitions.KNOWN_TOOL_PROFILES):
            code_execution_requested(profile)

    assert _warnings(caplog) == []


def test_distinct_unknown_values_are_bounded(caplog):
    """A client inventing a value per request must not flood the log or grow the set."""
    with caplog.at_level(logging.WARNING, logger=definitions.__name__):
        for i in range(definitions._MAX_WARNED_UNKNOWN_PROFILES + 20):
            code_execution_requested(f"junk-{i}")

    assert len(_warnings(caplog)) == definitions._MAX_WARNED_UNKNOWN_PROFILES
    assert len(definitions._WARNED_UNKNOWN_PROFILES) == definitions._MAX_WARNED_UNKNOWN_PROFILES


def test_the_degrade_itself_is_unchanged(caplog):
    """The warning is additive: an unknown profile still resolves to the no-code surface."""
    with caplog.at_level(logging.WARNING, logger=definitions.__name__):
        code_execution = code_execution_requested("cdoe")

    assert code_execution is code_execution_requested("nocode")
    assert {t["name"] for t in get_anthropic_tools(code_execution=code_execution)} == {
        t["name"] for t in get_anthropic_tools(code_execution=False)
    }


def test_the_profile_key_set_is_pinned_against_the_admin_default_and_the_browser():
    """Adding or renaming a profile here must be a deliberate decision, checked two ways.

    KNOWN_TOOL_PROFILES is what routers/llm_config.py validates DEFAULT_TOOL_PROFILE
    against (an unrecognised deployment default is rejected, not silently degraded), so
    the full literal below pins that admin-facing set.

    The browser's ToolProfile union in
    genetics-results-browser/src/features/chat/chat.types.ts is now just "code" | "nocode"
    — every other name here is a legacy value the browser no longer emits but old stored
    rows can still carry, so it isn't part of the union to pin. The one thing the two repos
    still share is that "code" and "nocode" both have to resolve here, which is the
    remaining cross-repo pin: if the browser ever emits a
    third value, this assertion alone won't catch it, but the degrade above ensures it
    still lands on the no-code surface rather than failing the request.

    If this fails: update the literal below, and if a profile name changed rather than
    being added, update the "Profile behavior" table in docs/project-spec.md.
    """
    assert set(definitions.KNOWN_TOOL_PROFILES) == {
        "api",
        "bigquery",
        "rag",
        "nocode",
        "code",
    }
    assert {"code", "nocode"} <= set(definitions.KNOWN_TOOL_PROFILES)
    # the three legacy names now resolve to the same surface as "nocode": the edge keeps
    # them accepted, and only "code" resolves anywhere else
    for legacy in ("api", "bigquery", "rag"):
        assert code_execution_requested(legacy) is code_execution_requested("nocode")
