"""Estimate API call cost from Anthropic token usage."""

# pricing per million tokens (USD) as of 2025
# format: model_prefix -> (input, output, cache_read, cache_creation)
_PRICING: dict[str, tuple[float, float, float, float]] = {
    "claude-opus":   (15.0, 75.0, 1.50, 18.75),
    "claude-sonnet": (3.0,  15.0, 0.30, 3.75),
    "claude-haiku":  (0.80, 4.0,  0.08, 1.0),
}


_CONTEXT_WINDOWS: dict[str, int] = {
    "claude-opus":   200_000,
    "claude-sonnet": 200_000,
    "claude-haiku":  200_000,
}


def get_context_window(model: str) -> int:
    """Return context window size (tokens) for the given model."""
    for prefix, window in _CONTEXT_WINDOWS.items():
        if prefix in model:
            return window
    # fallback to 200k
    return 200_000


def _match_pricing(model: str) -> tuple[float, float, float, float]:
    """Find pricing by matching model name prefix."""
    for prefix, pricing in _PRICING.items():
        if prefix in model:
            return pricing
    # fallback to sonnet pricing
    return _PRICING["claude-sonnet"]


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
