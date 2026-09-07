"""Tests for cost estimation utilities."""

import pytest

from genetics_mcp_server.cost import estimate_cost, get_context_window, has_pricing


class TestGetContextWindow:
    def test_fable_model(self):
        assert get_context_window("claude-fable-5-1") == 1_000_000

    def test_opus_model(self):
        assert get_context_window("claude-opus-4-20250514") == 1_000_000

    def test_sonnet_model(self):
        assert get_context_window("claude-sonnet-4-20250514") == 200_000

    def test_haiku_model(self):
        assert get_context_window("claude-haiku-4-5") == 200_000

    def test_unknown_model_falls_back(self):
        assert get_context_window("gpt-4o") == 200_000


# USD per million tokens: input, output, cache read, cache write (5-minute TTL)
@pytest.mark.parametrize(
    "model, rates",
    [
        ("claude-fable-5-1", (10.0, 50.0, 0.25, 12.5)),
        ("claude-fable-5", (10.0, 50.0, 1.0, 12.5)),
        ("claude-opus-5", (5.0, 25.0, 0.50, 6.25)),
        ("claude-opus-4-8", (5.0, 25.0, 0.50, 6.25)),
        ("claude-opus-4-5-20251101", (5.0, 25.0, 0.50, 6.25)),
        ("claude-opus-4-1-20250805", (15.0, 75.0, 1.50, 18.75)),
        ("claude-opus-4-20250514", (15.0, 75.0, 1.50, 18.75)),
        ("claude-sonnet-5", (2.0, 10.0, 0.20, 2.5)),
        ("claude-sonnet-4-6", (3.0, 15.0, 0.30, 3.75)),
        ("claude-sonnet-4-20250514", (3.0, 15.0, 0.30, 3.75)),
        ("claude-haiku-4-5", (1.0, 5.0, 0.10, 1.25)),
        ("claude-haiku-4-5-20251001", (1.0, 5.0, 0.10, 1.25)),
    ],
)
def test_per_million_token_rates(model, rates):
    inp, out, cache_read, cache_write = rates
    million = 1_000_000
    assert estimate_cost(model, million, 0) == pytest.approx(inp)
    assert estimate_cost(model, 0, million) == pytest.approx(out)
    assert estimate_cost(model, 0, 0, cache_read_tokens=million) == pytest.approx(cache_read)
    assert estimate_cost(model, 0, 0, cache_creation_tokens=million) == pytest.approx(cache_write)


class TestEstimateCost:
    def test_basic_cost(self):
        cost = estimate_cost("claude-sonnet-4-20250514", input_tokens=1000, output_tokens=500)
        expected = (1000 * 3.0 + 500 * 15.0) / 1_000_000
        assert cost == expected

    def test_with_cache_tokens(self):
        cost = estimate_cost(
            "claude-sonnet-4-20250514",
            input_tokens=1000,
            output_tokens=500,
            cache_read_tokens=2000,
            cache_creation_tokens=300,
        )
        expected = (1000 * 3.0 + 500 * 15.0 + 2000 * 0.30 + 300 * 3.75) / 1_000_000
        assert cost == expected

    def test_a_date_suffix_is_not_a_minor_version(self):
        """claude-sonnet-4-20250514 is Sonnet 4.0, not Sonnet 4.20250514 priced as Sonnet 5."""
        assert estimate_cost("claude-sonnet-4-20250514", 1_000_000, 0) == pytest.approx(3.0)

    def test_unknown_model_falls_back_to_sonnet_pricing(self):
        assert estimate_cost("gpt-4o", 1_000_000, 0) == pytest.approx(3.0)


class TestHasPricing:
    @pytest.mark.parametrize(
        "model", ["claude-fable-5-1", "claude-opus-5", "claude-haiku-4-5-20251001"]
    )
    def test_known_families(self, model):
        assert has_pricing(model)

    @pytest.mark.parametrize("model", ["gpt-4o", "claude-4-opus", "claude-mythos-5-1"])
    def test_unknown_models(self, model):
        assert not has_pricing(model)
