"""Tests for keeping short structured calls thinking-free.

Opus 5 thinks unless told not to, which both eats `max_tokens` and puts a
ThinkingBlock (no `.text`) first in `content`. Calls whose whole output is a
title or a JSON object opt out — except on Fable/Mythos, which reject the
opt-out.
"""

import pytest

from genetics_mcp_server.config import model_rejects_disabled_thinking
from genetics_mcp_server.scripts.analyze_conversations import thinking_off_kwargs


@pytest.mark.parametrize(
    "model,rejects",
    [
        ("claude-fable-5", True),
        ("claude-mythos-5", True),
        ("claude-opus-5", False),
        ("claude-opus-4-8", False),
        ("claude-sonnet-4-6", False),
        ("claude-haiku-4-5", False),
    ],
)
def test_model_rejects_disabled_thinking(model, rejects):
    assert model_rejects_disabled_thinking(model) is rejects


def test_thinking_off_kwargs_disables_where_supported():
    assert thinking_off_kwargs("claude-opus-5") == {"thinking": {"type": "disabled"}}


def test_thinking_off_kwargs_omits_param_on_always_thinking_models():
    # sending {"type": "disabled"} to Fable is a 400, so the param has to go
    assert thinking_off_kwargs("claude-fable-5") == {}
