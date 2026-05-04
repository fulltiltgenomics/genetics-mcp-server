"""Tests for cost estimation utilities."""

from genetics_mcp_server.cost import estimate_cost, get_context_window


class TestGetContextWindow:
    def test_opus_model(self):
        assert get_context_window("claude-opus-4-20250514") == 200_000

    def test_sonnet_model(self):
        assert get_context_window("claude-sonnet-4-20250514") == 200_000

    def test_haiku_model(self):
        assert get_context_window("claude-haiku-3-5-20241022") == 200_000

    def test_unknown_model_falls_back(self):
        assert get_context_window("gpt-4o") == 200_000


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
