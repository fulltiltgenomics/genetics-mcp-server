"""The generated blocks in docs/project-spec.md match the code they are derived from.

This is the gate that makes them generated rather than hand-kept: a change to a router, the
turn-metrics schema, a Settings default or the package tree fails here until
`scripts/gen-doc-blocks.py` is re-run and its output committed.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "gen-doc-blocks.py"
SPEC = "docs/project-spec.md"


def _generator():
    spec = importlib.util.spec_from_file_location("gen_doc_blocks", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_generated_blocks_are_current():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"], capture_output=True, text=True, cwd=ROOT
    )
    assert result.returncode == 0, (
        f"exit {result.returncode}; run scripts/gen-doc-blocks.py and commit the result\n"
        f"{result.stdout}{result.stderr}"
    )


def test_a_tampered_block_is_rewritten():
    gen = _generator()
    current = (ROOT / SPEC).read_text()
    tampered = current.replace(
        "| `cost_usd` | REAL | not null |", "| `cost_usd` | TEXT | nullable |"
    )
    assert tampered != current
    assert gen.render(SPEC, tampered) == current


def test_a_missing_marker_is_refused():
    gen = _generator()
    current = (ROOT / SPEC).read_text()
    without = current.replace("<!-- BEGIN GENERATED: module-tree -->", "")
    with pytest.raises(SystemExit) as exc:
        gen.render(SPEC, without)
    assert exc.value.code == 2
