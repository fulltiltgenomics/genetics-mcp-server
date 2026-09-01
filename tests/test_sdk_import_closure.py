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
        # `from pydantic import Field` at module level, and the complete map of every tool
        # the suite exposes — cut out of the closure by genetics-results-suite-6bv
        "genetics_mcp_server.tools.definitions",
        # the identity model of every service in the suite, including the
        # X-Goog-Authenticated-User-Email trust rules
        "genetics_mcp_server.auth.core",
        # the run_analysis gateway, its identity refusal and the artifact authorization
        # model. It subclasses the executor rather than being mixed into it precisely so
        # that the shipped half never names it; this is what asserts that stayed true.
        "genetics_mcp_server.tools.orchestration",
        "genetics_mcp_server.chat_api",
        "genetics_mcp_server.mcp_server",
        "genetics_mcp_server.llm_service",
    ],
)
def test_named_modules_stay_out(module):
    assert module not in _measured_closure()


# The distributions genetics-results-suite/sandbox/requirements.txt pins. pip installs those
# WITH their dependencies; the SDK wheel then goes in --no-deps (docs/code-execution-security.md,
# "Deviation 2"), so this list plus its transitive requirements is the entire third-party surface
# the sandbox interpreter has. Changing it means changing that file, and vice versa.
SANDBOX_PINNED_DISTRIBUTIONS = ("numpy", "scipy", "polars", "matplotlib", "httpx")

_BLOCKED_IMPORT_PROBE = """
import importlib.metadata as _md, json, sys

# The walk skips a requirement only when its marker mentions `extra`; EVERY other environment
# marker is ignored, so the requirement joins the allowed set whether or not pip would install
# it in the image. That is the one direction in which this guard is LAXER than the image, and
# it is the dangerous direction: `anyio` requires `exceptiongroup>=1.0.2; python_version <
# "3.11"`, the sandbox is Python 3.11 so pip does NOT install it there, yet `exceptiongroup`
# lands in `_dists` here — a closure module importing it would pass this test and fail the
# image build. `typing_extensions` (`; python_version < "3.13"`) has the same shape and is
# safe only by coincidence. The fix is to evaluate markers against the sandbox's Python
# version, or to walk a resolved lock for that image instead of this venv's dev metadata;
# filed separately, deliberately not done here.
#
# Every other divergence runs the other way — over-strict, which fails loudly rather than
# silently. `importlib.metadata.requires("anyio")` on this release does not declare `sniffio`
# at all, so `sniffio` stays blocked; that is CORRECT (the image installs it only as httpx's
# own dependency, and blocking it merely makes the probe stricter than the image). Measured:
# `import anyio` and a full `httpx.AsyncClient` request both succeed with it blocked.
_dists, _stack = set(), list({pinned!r})
while _stack:                                    # transitive requirements, extras excluded,
    _d = _stack.pop()                            # the way `pip install -r` resolves them
    _key = _d.lower().replace("-", "_")
    if _key in _dists:
        continue
    _dists.add(_key)
    try:
        _reqs = _md.requires(_d) or []
    except _md.PackageNotFoundError:
        continue
    for _r in _reqs:
        _name, _, _marker = _r.partition(";")
        if "extra" in _marker:
            continue
        _n = _name.split("[")[0].split("(")[0].strip()
        for _sep in ">=<!~ ":
            _n = _n.split(_sep)[0]
        if _n:
            _stack.append(_n)

# every other top-level module this venv installed: pydantic, dotenv, mcp, anthropic, fastapi,
# google.*, jwt ... Blocking by installed-distribution rather than by an allow-list of names
# leaves the standard library and the interpreter's own private modules untouched.
_blocked = {{
    _mod
    for _mod, _owners in _md.packages_distributions().items()
    if not any(_o.lower().replace("-", "_") in _dists for _o in _owners)
}} - {{"genetics_mcp_server"}}
_hit = set()


class _AbsentFromTheSandboxImage:
    def find_spec(self, fullname, path=None, target=None):
        top = fullname.split(".")[0]
        if top not in _blocked:
            return None
        _hit.add(top)
        raise ModuleNotFoundError("absent from the sandbox image: " + fullname, name=fullname)


sys.meta_path.insert(0, _AbsentFromTheSandboxImage())
import genetics_mcp_server.sdk  # noqa: F401
print(json.dumps({{"blocked": sorted(_blocked), "attempted": sorted(_hit)}}))
"""


def test_the_sdk_imports_with_every_unpinned_third_party_module_blocked():
    """The generalised form of the guard, and the point of it.

    Two separate offenders have widened the closure by one module-level import into a package
    the sandbox does not have: `config.settings` -> `dotenv` (genetics-results-suite-l41), then
    `tools/definitions.py` -> `pydantic` (genetics-results-suite-6bv). A per-offender test
    ("the SDK does not need dotenv") only ever catches the one already fixed, so this asserts
    the property instead: the SDK imports when EVERY distribution outside
    SANDBOX_PINNED_DISTRIBUTIONS' dependency closure is unavailable. The third offender fails
    here, in the commit that introduces it, rather than in another repo's image build.

    A module the SDK reaches for and does not get is not automatically a failure — httpx's CLI
    entry point is a real `try: ... except ImportError` and is expected to be denied — which is
    exactly why this asserts on the import succeeding rather than on nothing being attempted.
    """
    result = _run_probe(_ORIGIN_GUARD + _BLOCKED_IMPORT_PROBE.format(pinned=list(SANDBOX_PINNED_DISTRIBUTIONS)))
    assert result.returncode == 0, (
        "importing the SDK reached a package the sandbox image does not install; "
        "either defer that import out of the closure or widen sandbox/requirements.txt "
        f"deliberately:\n{result.stderr}"
    )
    measured = json.loads(result.stdout.strip().splitlines()[-1])
    # the guard is only meaningful if it is actually blocking; both known offenders must be in
    # the blocked set, or a future `pip install pydantic` into the dev venv could quietly
    # neuter this test
    for offender in ("dotenv", "pydantic"):
        assert offender in measured["blocked"], measured["blocked"]


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
