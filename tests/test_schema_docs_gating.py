"""Per-profile view gating in `schema_docs`, against a fixture tree with one gated view."""

import importlib
import importlib.resources
import json
import logging

import pytest

from genetics_mcp_server import schema_docs

README = """<!-- banner
     more -->
# BigQuery views

Cross-view rules.

| view | what |
|---|---|
| `open_v` | the open view |
| `gated_v` | the gated view |
"""
VIEW = "<!-- banner -->\n# {name}\n\nbody of {name}\n\n## Columns\n"


@pytest.fixture
def docs(tmp_path, monkeypatch):
    (tmp_path / "README.md").write_text(README)
    for name in ("open_v", "gated_v"):
        (tmp_path / f"{name}.md").write_text(VIEW.format(name=name))
    (tmp_path / "_availability.json").write_text(
        json.dumps({"profiles": ["daly", "finngen"], "views": {"gated_v": ["daly"]}})
    )
    monkeypatch.setattr(schema_docs, "_DIR", str(tmp_path))
    monkeypatch.delenv("CONFIG_PROFILE", raising=False)
    schema_docs._hidden_views.cache_clear()
    schema_docs._reference.cache_clear()
    yield tmp_path
    schema_docs._hidden_views.cache_clear()
    schema_docs._reference.cache_clear()


def test_daly_sees_everything(docs, monkeypatch):
    monkeypatch.setenv("CONFIG_PROFILE", "daly")
    assert schema_docs.view_names() == ["gated_v", "open_v"]
    ref = schema_docs.schema_reference()
    assert "| `gated_v` |" in ref and "body of gated_v" in ref


def test_finngen_hides_gated_view_and_its_row(docs, monkeypatch):
    monkeypatch.setenv("CONFIG_PROFILE", "finngen")
    assert schema_docs.view_names() == ["open_v"]
    ref = schema_docs.schema_reference()
    assert "gated_v" not in ref
    assert "| `open_v` |" in ref and "| `gated_v` |" not in ref
    assert "body of open_v" in ref


@pytest.mark.parametrize("value", [None, "", "nope"])
def test_unset_or_unknown_fails_closed_and_logs_error_once(docs, monkeypatch, caplog, value):
    if value is not None:
        monkeypatch.setenv("CONFIG_PROFILE", value)
    with caplog.at_level(logging.ERROR):
        assert schema_docs.view_names() == ["open_v"]
        assert "gated_v" not in schema_docs.schema_reference()
        schema_docs.view_names()
    assert len([r for r in caplog.records if r.levelno == logging.ERROR]) == 1


def test_profile_change_is_not_served_from_a_stale_cache(docs, monkeypatch):
    monkeypatch.setenv("CONFIG_PROFILE", "daly")
    assert "gated_v" in schema_docs.schema_reference()
    monkeypatch.setenv("CONFIG_PROFILE", "finngen")
    assert "gated_v" not in schema_docs.schema_reference()


@pytest.mark.parametrize("content", [None, "{not json", "[]", '{"profiles": []}'])
def test_missing_or_malformed_json_raises(docs, monkeypatch, content):
    if content is None:
        (docs / "_availability.json").unlink()
    else:
        (docs / "_availability.json").write_text(content)
    monkeypatch.setenv("CONFIG_PROFILE", "finngen")
    with pytest.raises(RuntimeError, match="_availability.json"):
        schema_docs.view_names()
    with pytest.raises(RuntimeError):
        schema_docs.schema_reference()


def test_ungated_json_never_logs_for_unknown_profile(docs, caplog):
    (docs / "_availability.json").write_text('{"profiles": ["daly"], "views": {}}')
    with caplog.at_level(logging.ERROR):
        assert schema_docs.view_names() == ["gated_v", "open_v"]
    assert not caplog.records


def test_shipped_availability_json_is_valid_and_in_the_package():
    shipped = importlib.resources.files("genetics_mcp_server.schema_docs") / "_availability.json"
    assert shipped.is_file()
    data = json.loads(shipped.read_text())
    assert set(data) == {"profiles", "views"}
    assert all(set(p) <= set(data["profiles"]) for p in data["views"].values())


def test_prompt_and_tool_description_inherit_the_filter(docs, monkeypatch):
    # all three consumers evaluate schema_docs at import, so reload them under the fixture
    from genetics_mcp_server.config import defaults, prompt_condensed
    from genetics_mcp_server.tools import definitions

    modules = (prompt_condensed, defaults, definitions)

    def schema_text():
        desc = next(
            t["parameters"]["table"]["description"]
            for t in definitions.BIGQUERY_TOOL_DEFINITIONS
            if t["name"] == "get_database_schema"
        )
        blocks = " ".join(
            b.text for b in (*defaults._PROMPT_BLOCKS, *prompt_condensed.CONDENSED_PROMPT_BLOCKS)
        )
        return desc, blocks

    try:
        monkeypatch.setenv("CONFIG_PROFILE", "finngen")
        for m in modules:
            importlib.reload(m)
        desc, blocks = schema_text()
        assert "gated_v" not in desc and "open_v" in desc
        assert "gated_v" not in blocks and "open_v" in blocks

        monkeypatch.setenv("CONFIG_PROFILE", "daly")
        for m in modules:
            importlib.reload(m)
        desc, blocks = schema_text()
        assert "gated_v" in desc and "gated_v" in blocks
    finally:
        monkeypatch.undo()
        schema_docs._hidden_views.cache_clear()
        schema_docs._reference.cache_clear()
        for m in modules:
            importlib.reload(m)
