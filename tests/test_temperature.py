"""Tests for temperature handling: off by default, model-specific rejection."""

import importlib

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
        ("claude-opus-4-6", False),
        ("claude-sonnet-4-6", False),
        ("claude-haiku-4-5", False),
        ("gpt-4o", False),
    ],
)
def test_model_rejects_temperature(model, rejects):
    assert model_rejects_temperature(model) is rejects


def test_temperature_off_by_default(monkeypatch):
    monkeypatch.delenv("TEMPERATURE", raising=False)
    importlib.reload(settings_module)
    assert settings_module.Settings().temperature is None


def test_temperature_opt_in_via_env(monkeypatch):
    monkeypatch.setenv("TEMPERATURE", "0.5")
    importlib.reload(settings_module)
    assert settings_module.Settings().temperature == 0.5
    # restore module state for other tests
    monkeypatch.delenv("TEMPERATURE", raising=False)
    importlib.reload(settings_module)
