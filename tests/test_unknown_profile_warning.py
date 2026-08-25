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
from genetics_mcp_server.tools.definitions import get_anthropic_tools


@pytest.fixture(autouse=True)
def _clear_warned_profiles():
    definitions._WARNED_UNKNOWN_PROFILES.clear()
    yield
    definitions._WARNED_UNKNOWN_PROFILES.clear()


def _warnings(caplog):
    return [r for r in caplog.records if r.levelno == logging.WARNING]


def test_unknown_profile_warns_naming_the_value_and_the_known_set(caplog):
    with caplog.at_level(logging.WARNING, logger=definitions.__name__):
        get_anthropic_tools(tool_profile="cdoe")

    records = _warnings(caplog)
    assert len(records) == 1
    message = records[0].getMessage()
    assert "cdoe" in message
    # the known set has to be IN the warning: "unknown profile" alone does not tell an
    # operator whether the browser or the server is the side that drifted
    for known in (*definitions.TOOL_PROFILES, *definitions.TOOL_PROFILE_TOOLS):
        assert known in message


def test_the_same_unknown_value_warns_only_once(caplog):
    with caplog.at_level(logging.WARNING, logger=definitions.__name__):
        for _ in range(25):
            get_anthropic_tools(tool_profile="cdoe")

    assert len(_warnings(caplog)) == 1


def test_a_second_distinct_unknown_value_still_warns(caplog):
    with caplog.at_level(logging.WARNING, logger=definitions.__name__):
        get_anthropic_tools(tool_profile="cdoe")
        get_anthropic_tools(tool_profile="bigqeury")

    assert {"cdoe", "bigqeury"} <= {r.getMessage().split("'")[1] for r in _warnings(caplog)}


def test_known_profiles_and_no_profile_stay_quiet(caplog):
    with caplog.at_level(logging.WARNING, logger=definitions.__name__):
        for profile in (None, *definitions.TOOL_PROFILES, *definitions.TOOL_PROFILE_TOOLS):
            get_anthropic_tools(tool_profile=profile)

    assert _warnings(caplog) == []


def test_distinct_unknown_values_are_bounded(caplog):
    """A client inventing a value per request must not flood the log or grow the set."""
    with caplog.at_level(logging.WARNING, logger=definitions.__name__):
        for i in range(definitions._MAX_WARNED_UNKNOWN_PROFILES + 20):
            get_anthropic_tools(tool_profile=f"junk-{i}")

    assert len(_warnings(caplog)) == definitions._MAX_WARNED_UNKNOWN_PROFILES
    assert len(definitions._WARNED_UNKNOWN_PROFILES) == definitions._MAX_WARNED_UNKNOWN_PROFILES


def test_the_degrade_itself_is_unchanged(caplog):
    """The warning is additive: an unknown profile still resolves to general-only."""
    with caplog.at_level(logging.WARNING, logger=definitions.__name__):
        names = {t["name"] for t in get_anthropic_tools(tool_profile="cdoe")}

    assert names == {t["name"] for t in get_anthropic_tools(tool_profile="rag")}


def test_the_profile_key_set_is_pinned_against_the_browsers_copy():
    """Adding or renaming a profile here must be a deliberate, two-repo decision.

    The other list is TOOL_PROFILES in
    genetics-results-browser/src/features/chat/chat.types.ts, plus TOOL_PROFILE_LABELS in
    src/features/chat/LLMChat.tsx which decides whether the Tools control offers it. The
    two repos cannot import each other, so nothing but a literal on each side pins them
    together, and BOTH drift directions are silent by construction:

      - a name the browser offers and this server dropped resolves to general-only here
        (the degrade above), so the user gets a much smaller surface than the one named;
      - a name added HERE that the browser predates is narrowed to null there, and null
        means no `tool_profile` on the request, which is NO filtering — the user's stored,
        narrower choice silently becomes the full surface.

    The browser has had the mirror of this test since useChatOptions.test.ts:196 ("lists
    code alongside the three original profiles"), which is why its own drift already fails
    a test. This is the missing half (genetics-results-suite-4h6.74).

    If this fails: update the browser's TOOL_PROFILES and decide whether the new profile
    gets a TOOL_PROFILE_LABELS entry (offered in the UI) or `null` (resolvable but not
    selectable, as `rag` is), then update the literal below and the "Profile behavior"
    table in docs/project-spec.md.
    """
    assert set(definitions.TOOL_PROFILES) | set(definitions.TOOL_PROFILE_TOOLS) == {
        "api",
        "bigquery",
        "rag",
        "nocode",
        "code",
    }
    # the two mechanisms must stay disjoint: TOOL_PROFILE_TOOLS wins where they overlap, so a
    # name in both would resolve to its allow-list and quietly ignore its categories
    assert not set(definitions.TOOL_PROFILES) & set(definitions.TOOL_PROFILE_TOOLS)
