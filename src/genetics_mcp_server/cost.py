"""Estimate API call cost from Anthropic token usage."""

import re

# USD per million tokens: (input, output, cache_read, cache_creation), keyed by model family
# and the lowest version the row applies to. Rows within a family are ordered newest first
# and the first row whose floor the model meets wins, so a new minor release inherits its
# family's latest row until a price change adds one above it.
#
# cache_creation is the 5-minute TTL write (1.25x input); the 1-hour TTL writes at 2x, but
# nothing here sends `ttl: "1h"`. Cache reads are 0.1x input everywhere except Fable 5.1,
# where Anthropic cut them to 0.025x.
_PRICING: list[tuple[str, tuple[int, int], tuple[float, float, float, float]]] = [
    ("fable",  (5, 1), (10.0, 50.0, 0.25, 12.5)),
    ("fable",  (5, 0), (10.0, 50.0, 1.0,  12.5)),
    ("opus",   (4, 5), (5.0,  25.0, 0.50, 6.25)),
    ("opus",   (0, 0), (15.0, 75.0, 1.50, 18.75)),
    ("sonnet", (5, 0), (2.0,  10.0, 0.20, 2.5)),
    ("sonnet", (0, 0), (3.0,  15.0, 0.30, 3.75)),
    ("haiku",  (4, 5), (1.0,  5.0,  0.10, 1.25)),
    ("haiku",  (0, 0), (0.80, 4.0,  0.08, 1.0)),
]

# what an unrecognised model is priced at; `has_pricing` lets callers refuse instead
_FALLBACK_PRICING = (3.0, 15.0, 0.30, 3.75)

_CONTEXT_WINDOWS: dict[str, int] = {
    "fable":  1_000_000,
    "opus":   1_000_000,
    "sonnet": 200_000,
    "haiku":  200_000,
}

# `claude-<family>-<major>[-<minor>][-<yyyymmdd>]`. The minor group also swallows a date
# suffix on a model with no minor version (claude-sonnet-4-20250514), so anything eight or
# more digits long is a date, not a version.
_MODEL_RE = re.compile(r"claude-(fable|opus|sonnet|haiku)-(\d+)(?:-(\d+))?")


def _parse_model(model: str) -> tuple[str, tuple[int, int]] | None:
    match = _MODEL_RE.search(model)
    if not match:
        return None
    family, major, minor = match.group(1), int(match.group(2)), match.group(3)
    minor_version = int(minor) if minor and len(minor) < 8 else 0
    return family, (major, minor_version)


def get_context_window(model: str) -> int:
    """Return context window size (tokens) for the given model."""
    parsed = _parse_model(model)
    if parsed:
        return _CONTEXT_WINDOWS[parsed[0]]
    # fallback to 200k
    return 200_000


def has_pricing(model: str) -> bool:
    """True when `model` matches a known pricing entry exactly.

    `_match_pricing` falls back to Sonnet for anything unrecognised, which is fine for a
    rough in-flight estimate but not for a report that gates a spend decision: `gpt-4o`
    or a transposed `claude-4-opus` would be priced confidently and wrongly. Callers that
    must not fabricate a number check this first.
    """
    return _parse_model(model) is not None


def _match_pricing(model: str) -> tuple[float, float, float, float]:
    """Find pricing by model family and version."""
    parsed = _parse_model(model)
    if parsed:
        family, version = parsed
        for row_family, floor, pricing in _PRICING:
            if row_family == family and version >= floor:
                return pricing
    # fallback to sonnet pricing
    return _FALLBACK_PRICING


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
) -> float:
    """Return estimated cost in USD."""
    inp, out, cache_rd, cache_cr = _match_pricing(model)
    cost = (
        input_tokens * inp
        + output_tokens * out
        + cache_read_tokens * cache_rd
        + cache_creation_tokens * cache_cr
    ) / 1_000_000
    return cost
