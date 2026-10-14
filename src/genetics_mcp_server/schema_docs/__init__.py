"""The BigQuery view documentation, inlined into the system prompt.

The `.md` files beside this one are GENERATED — genetics-results-suite's
`scripts/gen-sandbox-docs.py` renders them from that repo's `configs/datasets.yaml` and
writes the identical bytes here and into the sandbox image, and its `--check` mode fails
the image build when the two diverge. Do not edit them; edit `configs/datasets.yaml`.

WHY THIS IS IN THE PROMPT rather than read on demand. It used to be read on demand, and
that is what the round trip cost: measured over benchmark run a08b371d, 32 of 138
`run_analysis` scripts did nothing but `print(open(GENETICS_SCHEMA_DIR + '/<view>.md'))`,
18 of 20 cases opened with one, and they accounted for 658s of 4272s of model time — to
fetch bytes that were already in the container and never change between builds. Prompt
caching makes the same bytes a one-off cache write instead.

The files stay in the sandbox image regardless: a script may still read one, and the
image's own docs are what a script's `GENETICS_SCHEMA_DIR` names. What changed is that
the prompt no longer *instructs* a read, because the answer is already above it.
"""

import functools
import json
import logging
import os
import re

logger = logging.getLogger(__name__)

_DIR = os.path.dirname(os.path.abspath(__file__))
# the generator writes exactly one file per view plus this index; README is rendered
# first because it carries the cross-view rules (bare view names, the scan ceilings)
# that a single view's file does not repeat
_INDEX = "README.md"


_AVAILABILITY = "_availability.json"
# deliberately not `PROFILE`: DEFAULT_TOOL_PROFILE is an unrelated tool-set selector
_PROFILE_ENV = "CONFIG_PROFILE"


def _profile() -> str | None:
    return os.environ.get(_PROFILE_ENV) or None


@functools.lru_cache(maxsize=None)
def _hidden_views(profile: str | None) -> frozenset[str]:
    """Gated views the deployment `profile` may not see.

    Mirrors `filter_schema_docs` in genetics-results-suite's `scripts/gen-sandbox-docs.py`.
    Fails closed on the profile: unset or unknown hides every gated view and logs an
    ERROR (once per distinct value, via the cache), so a misconfigured pod loses a view
    rather than advertising one its data backend lacks. A missing or unparseable json
    raises instead of meaning "no gating": images are built from git clones, so a json
    left out of a commit would otherwise advertise every gated view on every profile.
    Ungated views are never affected by the profile.
    """
    try:
        with open(os.path.join(_DIR, _AVAILABILITY)) as fh:
            availability = json.load(fh)
        availability["profiles"], availability["views"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(
            f"schema_docs/{_AVAILABILITY} is missing or malformed ({exc!r}); it is generated "
            "by genetics-results-suite scripts/gen-sandbox-docs.py and must be committed"
        ) from exc
    known = profile in availability["profiles"]
    gated = availability["views"]
    if not known and gated:
        logger.error(
            "%s=%r is not one of %s; hiding gated schema views %s",
            _PROFILE_ENV, profile, availability["profiles"], sorted(gated),
        )
    return frozenset(v for v, allowed in gated.items() if not known or profile not in allowed)


def _visible(hidden: frozenset[str]) -> list[str]:
    return sorted(
        f[:-3] for f in os.listdir(_DIR)
        if f.endswith(".md") and f != _INDEX and f[:-3] not in hidden
    )


def view_names() -> list[str]:
    """The views documented here and visible to this deployment's profile."""
    return _visible(_hidden_views(_profile()))


def schema_reference() -> str:
    """Every visible view's documentation as ONE prompt section.

    Built once per distinct `CONFIG_PROFILE`; an env that never changes in-process (the
    deployed case) therefore costs one build, as before.
    """
    return _reference(_profile())


@functools.lru_cache(maxsize=None)
def _reference(profile: str | None) -> str:
    """Headings are demoted one level on the way in. As files each view's doc is its own
    document and starts at `#`; concatenated into a prompt whose own sections are `##`,
    seventeen `#` headings would read as seventeen top-level sections competing with
    "Response Style" and "Prohibited" rather than as one reference under one of them.
    Demoting makes the view the `##` and its Columns/Worked-examples the `###`, which is
    what the nesting actually is.

    Cached: the bytes are fixed at build time, and `default_system_prompt` is called per
    request.
    """
    hidden = _hidden_views(profile)
    # the index's own title becomes a near-duplicate of the prompt section heading above
    # it ("BigQuery views" under "BigQuery view reference"), so it is dropped and its
    # body — the cross-view rules and the view list — sits directly under that heading
    parts = [_strip_title(_demote(_drop_rows(_read(_INDEX), hidden)))]
    parts += [_demote(_read(f"{name}.md")) for name in _visible(hidden)]
    return "\n\n".join(parts)


def _drop_rows(text: str, hidden: frozenset[str]) -> str:
    """Remove each hidden view's index row (`| `<view>` | ...`).

    The shape is the generator's; its test asserts no surviving file names a hidden view,
    which is what catches the index mentioning one in any other form.
    """
    rows = [re.compile(rf"^\| `{re.escape(v)}` \|") for v in hidden]
    return "".join(
        line for line in text.splitlines(keepends=True)
        if not any(row.match(line) for row in rows)
    )


def _strip_title(text: str) -> str:
    lines = text.split("\n")
    if lines and lines[0].startswith("#"):
        lines = lines[1:]
    return "\n".join(lines).strip()


def _demote(text: str) -> str:
    """Add one `#` to every ATX heading, leaving fenced code blocks alone.

    The worked examples are ```sql fences; a `#` inside one is a SQL comment, not a
    heading, and rewriting it would corrupt an example the model is told to copy.
    """
    out, in_fence = [], False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        elif not in_fence and line.startswith("#"):
            line = "#" + line
        out.append(line)
    return "\n".join(out)


def _read(name: str) -> str:
    with open(os.path.join(_DIR, name)) as fh:
        text = fh.read()
    # the generated banner tells a human not to edit the file; in a prompt it is noise
    # that the model could mistake for an instruction about its own output
    lines = text.splitlines()
    if lines and lines[0].startswith("<!--"):
        end = next((i for i, l in enumerate(lines) if l.rstrip().endswith("-->")), 0)
        lines = lines[end + 1 :]
    return "\n".join(lines).strip()
