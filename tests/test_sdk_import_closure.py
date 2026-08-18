"""The import closure of `genetics_mcp_server.sdk` is a security boundary, not a detail.

The sandbox image (genetics-results-suite-4h6.6) pip-installs this distribution and then
deletes every genetics_mcp_server module outside this closure, because a prompt-injected
script in the sandbox reads source files — it does not need them to import. Before
genetics-results-suite-l41 the closure reached `config/settings.py`, which enumerates the
suite's entire internal configuration surface by name (INTERNAL_API_SECRET, the four LLM
provider keys, the allow-lists, the OAuth endpoints, the database paths).

The closure regrows silently: one new module-level import anywhere in `tools/` and the
sandbox ships another file. The sandbox build asserts the *shipped* set equals its own
allow-list, but that build lives in another repo and runs only when the image is built.
This test is the one that fails in this repo, in the commit that widens it.

Measured the way that allow-list was: import the package and enumerate `sys.modules`, in
a fresh interpreter so nothing another test imported can be mistaken for a dependency.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# every probe below runs in a fresh interpreter, which resolves genetics_mcp_server from
# its own sys.path — an editable install elsewhere on the machine would otherwise let this
# test measure a different checkout and pass or fail independently of the code under it.
# _SRC is prepended, and each probe asserts the package it imported actually came from it.
_SRC = Path(__file__).resolve().parents[1] / "src"

_ORIGIN_GUARD = f"""
import pathlib, sys
import genetics_mcp_server
_src = pathlib.Path({str(_SRC)!r}).resolve()
_origin = pathlib.Path(genetics_mcp_server.__file__).resolve()
assert _src in _origin.parents, f"imported {{_origin}}, not the checkout under test ({{_src}})"
"""


def _run_probe(body: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(_SRC), os.environ.get("PYTHONPATH", "")])}
    return subprocess.run(
        [sys.executable, "-c", body], capture_output=True, text=True, check=False, env=env
    )

# Widening this set means the sandbox image ships another genetics_mcp_server source file.
# Do that deliberately — and update SDK_ALLOWLIST in genetics-results-suite
# sandbox/prune_venv.py in the same change, or the sandbox build fails.
SDK_IMPORT_CLOSURE = frozenset(
    {
        "genetics_mcp_server",
        "genetics_mcp_server.sdk",
        "genetics_mcp_server.sdk._runner",
        "genetics_mcp_server.sdk.client",
        "genetics_mcp_server.sdk.errors",
        "genetics_mcp_server.tools",
        "genetics_mcp_server.tools.definitions",
        "genetics_mcp_server.tools.executor",
        "genetics_mcp_server.tools.phewas_categories",
        "genetics_mcp_server.tools.sql_safety",
        "genetics_mcp_server.tools.uniprot",
    }
)

_PROBE = (
    _ORIGIN_GUARD
    + """
import json, sys
import genetics_mcp_server.sdk  # noqa: F401
print(json.dumps(sorted(m for m in sys.modules if m.split(".")[0] == "genetics_mcp_server")))
"""
)


def _measured_closure() -> set[str]:
    result = _run_probe(_PROBE)
    assert result.returncode == 0, f"importing the SDK failed: {result.stderr}"
    return set(json.loads(result.stdout.strip().splitlines()[-1]))


def test_the_closure_is_exactly_the_pinned_set():
    measured = _measured_closure()
    assert measured == set(SDK_IMPORT_CLOSURE), (
        f"grew: {sorted(measured - SDK_IMPORT_CLOSURE)}, "
        f"shrank: {sorted(SDK_IMPORT_CLOSURE - measured)}"
    )


@pytest.mark.parametrize(
    "module",
    [
        # names every internal env var (genetics-results-suite-l41)
        "genetics_mcp_server.config.settings",
        # the identity model of every service in the suite, including the
        # X-Goog-Authenticated-User-Email trust rules
        "genetics_mcp_server.auth.core",
        "genetics_mcp_server.chat_api",
        "genetics_mcp_server.mcp_server",
        "genetics_mcp_server.llm_service",
    ],
)
def test_named_modules_stay_out(module):
    assert module not in _measured_closure()


def test_importing_the_sdk_does_not_need_dotenv():
    """`config.settings` calls load_dotenv() at module scope, so while it was in the
    closure the sandbox image had to pin python-dotenv purely to make `import
    genetics_mcp_server.sdk` succeed. Nothing in the closure should need it now."""
    probe = _ORIGIN_GUARD + (
        "import sys; sys.modules['dotenv'] = None; "
        "import genetics_mcp_server.sdk"  # a real import of dotenv would raise here
    )
    result = _run_probe(probe)
    assert result.returncode == 0, result.stderr


def test_the_executor_can_be_built_and_used_without_config_settings(tmp_path):
    """The sandbox deletes config/settings.py, so `ToolExecutor()` — which the SDK builds
    on first use — must not reach for it. Blocking the import here reproduces that image
    without needing one.

    The probe runs BOTH credential states of that image, because the settings resolution is
    what both of them turn on:

    * with no token file, `_build_client` must reach `_PRUNED_INSTALL_SETTINGS`, read its
      empty `internal_api_secret` and refuse (genetics-results-suite-4h6.44) — reaching the
      `settings is _PRUNED_INSTALL_SETTINGS` discriminator at all is the proof that the
      import was deferred rather than merely absent;
    * with the token file the supervisor actually writes, the client builds, and the
      per-destination auth means there is still no default Authorization header.

    Neither the URL reads nor `uniprot` short-circuit on the token path, so the assertions
    that the deferred resolution survives config/settings.py's absence are unchanged.
    """
    tokens = tmp_path / "tokens.json"
    tokens.write_text(json.dumps({"db-api": "db.token", "results-api": "results.token"}))
    probe = (
        _ORIGIN_GUARD
        + f"""
import sys
sys.modules["genetics_mcp_server.config"] = None
sys.modules["genetics_mcp_server.config.settings"] = None
import genetics_mcp_server.sdk as genetics  # noqa: F401
import os
from genetics_mcp_server.tools import executor as executor_mod
from genetics_mcp_server.tools.executor import SandboxCredentialError, ToolExecutor

os.environ.pop("SANDBOX_TOKEN_FILE", None)
os.environ.pop("INTERNAL_API_SECRET", None)
try:
    ToolExecutor(row_limit=None).client
except SandboxCredentialError:
    pass
else:
    raise AssertionError("a pruned install with no token file must not go uncredentialed")

executor_mod._reset_sandbox_tokens()
os.environ["SANDBOX_TOKEN_FILE"] = {str(tokens)!r}
executor = ToolExecutor(row_limit=None)
assert "Authorization" not in executor.client.headers
assert executor.uniprot._uniprot_url == "https://rest.uniprot.org"
# the URL reads route through the same deferred settings resolution as the secret, so
# they must survive its absence too
assert executor.base_url == "http://0.0.0.0:2000/api"
assert executor.bigquery_url is None
print("ok")
"""
    )
    result = _run_probe(probe)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_the_executor_still_sees_a_dotenv_file(tmp_path):
    """Deferring the settings import must not move the endpoint reads ahead of load_dotenv().

    config/settings.py calls load_dotenv() at module scope, and that used to be the first
    statement of ToolExecutor.__init__, so os.environ was populated before the three
    GENETICS_API_URL / GENETICS_PUBLIC_API_URL / BIGQUERY_API_URL reads ran. Deferring the
    import for the sandbox once put those reads *first*: the executor landed on the
    hard-coded default URL while still attaching a .env-supplied secret to it, and
    bigquery_url came back None, silently disabling the SQL tools. Affects every standalone
    entry point (scripts/analyze_variants.py, the SDK used outside the service) and nothing
    else pins it.
    """
    (tmp_path / ".env").write_text(
        "GENETICS_API_URL=http://dotenv.example/api\n"
        "GENETICS_PUBLIC_API_URL=http://public.dotenv.example/api\n"
        "BIGQUERY_API_URL=http://dotenv.example:8080\n"
    )
    probe = (
        _ORIGIN_GUARD
        + """
from genetics_mcp_server.tools.executor import ToolExecutor

executor = ToolExecutor()
print(executor.base_url)
print(executor.public_url)
print(executor.bigquery_url)
"""
    )
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("GENETICS_API_URL", "GENETICS_PUBLIC_API_URL", "BIGQUERY_API_URL")
    }
    env["PYTHONPATH"] = os.pathsep.join([str(_SRC), env.get("PYTHONPATH", "")])
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == [
        "http://dotenv.example/api",
        "http://public.dotenv.example/api",
        "http://dotenv.example:8080",
    ], result.stdout


def test_the_pruned_install_fallback_matches_settings_defaults():
    """The fallback restates five of Settings' defaults rather than importing them; this
    is what stops the two copies drifting."""
    from genetics_mcp_server.config.settings import Settings
    from genetics_mcp_server.tools.executor import _PrunedInstallSettings

    fallback = _PrunedInstallSettings()
    for name in (
        "myvariant_api_url",
        "uniprot_api_url",
        "ebi_proteins_api_url",
        "uniprot_cache_ttl",
    ):
        # Settings reads the environment, so compare against its declared default rather
        # than against a live instance
        default = Settings.__dataclass_fields__[name].default_factory
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {}, clear=True):
            assert getattr(fallback, name) == default(), name
    # deliberately not read from the environment: the sandbox holds no internal secret,
    # and naming the variable in a shipped file is the disclosure l41 removes
    assert fallback.internal_api_secret == ""
