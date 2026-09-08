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
import os

_DIR = os.path.dirname(os.path.abspath(__file__))
# the generator writes exactly one file per view plus this index; README is rendered
# first because it carries the cross-view rules (bare view names, the scan ceilings)
# that a single view's file does not repeat
_INDEX = "README.md"


def view_names() -> list[str]:
    """The views documented here, derived from the directory rather than listed."""
    return sorted(
        f[:-3] for f in os.listdir(_DIR) if f.endswith(".md") and f != _INDEX
    )


@functools.lru_cache(maxsize=1)
def schema_reference() -> str:
    """Every view's documentation as ONE prompt section.

    Headings are demoted one level on the way in. As files each view's doc is its own
    document and starts at `#`; concatenated into a prompt whose own sections are `##`,
    seventeen `#` headings would read as seventeen top-level sections competing with
    "Response Style" and "Prohibited" rather than as one reference under one of them.
    Demoting makes the view the `##` and its Columns/Worked-examples the `###`, which is
    what the nesting actually is.

    Cached: the bytes are fixed at build time, and `default_system_prompt` is called per
    request.
    """
    # the index's own title becomes a near-duplicate of the prompt section heading above
    # it ("BigQuery views" under "BigQuery view reference"), so it is dropped and its
    # body — the cross-view rules and the view list — sits directly under that heading
    parts = [_strip_title(_demote(_read(_INDEX)))]
    parts += [_demote(_read(f"{name}.md")) for name in view_names()]
    return "\n\n".join(parts)


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
