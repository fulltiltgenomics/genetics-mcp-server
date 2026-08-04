"""Tests for temperature handling: off by default, model-specific rejection."""

import pytest

from genetics_mcp_server.config import model_rejects_temperature
from genetics_mcp_server.config import settings as settings_module


@pytest.mark.parametrize(
    "model,rejects",
    [
        ("claude-fable-5", True),
        ("claude-fable-7", True),
        ("claude-opus-4-7", True),
        ("claude-opus-4-8", True),
        ("claude-opus-5", True),
        ("claude-opus-6", True),
        ("claude-opus-4-6", False),
        ("claude-sonnet-4-6", False),
        ("claude-haiku-4-5", False),
        ("gpt-4o", False),
    ],
)
def test_model_rejects_temperature(model, rejects):
    assert model_rejects_temperature(model) is rejects


# every Settings field reads the environment from its default_factory, i.e. at
# instantiation, so these need no module reload. Reloading would rebind
# config.settings.get_settings to a fresh function while config/__init__ (and every
# module that imported from it) keeps the old one and its stale cache.
def test_temperature_off_by_default(monkeypatch):
    monkeypatch.delenv("TEMPERATURE", raising=False)
    assert settings_module.Settings().temperature is None


def test_temperature_opt_in_via_env(monkeypatch):
    monkeypatch.setenv("TEMPERATURE", "0.5")
    assert settings_module.Settings().temperature == 0.5
