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

Importing is only half of it. The image ships source, and source that runs at CALL time is
invisible to anything that imports the package and looks at what loaded — so the shipped
files are also read with `ast`, which is the only half that can see a deferred import.
"""

import ast
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

# The top-level module names those pins provide, and the only third-party names a shipped source
# file may name. Written down rather than read out of installed metadata, because metadata makes
# the verdict a function of THIS venv rather than of the image: a pin that is not installed here
# (scipy is not) drops out of the set silently, and two developers get different answers from the
# same commit. Written down, the answer is the same everywhere and the test below is what stops
# it rotting.
#
# Deliberately narrower than the image, which also holds everything pip pulls in behind those
# pins — anyio, certifi, contourpy, six and the rest. Those are legitimately importable there,
# but a shipped file reaching past the declared pins into httpx's or matplotlib's private
# dependency set is a widening that should be argued for; refusing it here fails loudly, which
# is the safe direction.
SANDBOX_MODULE_NAMES = frozenset(
    {"numpy", "scipy", "polars", "matplotlib", "mpl_toolkits", "pylab", "httpx"}
)


def test_the_pinned_module_names_still_match_the_pinned_distributions():
    """SANDBOX_MODULE_NAMES is written down, so something has to notice it going stale.

    Only the pins that happen to be installed here can be checked, and that is the point of
    writing the set down in the first place: this narrows as the venv narrows instead of
    changing the verdict of the guards that use it.
    """
    import importlib.metadata as md

    pinned = {d.lower().replace("-", "_") for d in SANDBOX_PINNED_DISTRIBUTIONS}
    owners_of = md.packages_distributions()
    provided = {
        module
        for module, owners in owners_of.items()
        if {o.lower().replace("-", "_") for o in owners} & pinned
    }
    assert provided <= SANDBOX_MODULE_NAMES, (
        f"a pin provides a module the set does not list: {sorted(provided - SANDBOX_MODULE_NAMES)}"
    )
    for name in sorted(SANDBOX_MODULE_NAMES):
        owners = {o.lower().replace("-", "_") for o in owners_of.get(name, ())}
        assert not owners or owners & pinned, f"{name} comes from {sorted(owners)}, not a pin"


# Installs a meta_path finder that answers for the image's module surface rather than for this
# venv's. The obvious alternative — block every installed distribution outside the pins — is
# rejected on two counts, and both are why this admits nothing by default:
#
#   * a top-level module `importlib.metadata.packages_distributions()` cannot attribute to any
#     distribution falls outside a blocked set entirely and imports freely: `.pth`-installed
#     modules such as `_virtualenv`, egg-link and develop installs, loose files dropped into
#     site-packages;
#   * deriving a blocked set means walking requirement metadata, which means evaluating
#     environment markers or ignoring them. Ignoring them admits `exceptiongroup` — `anyio`
#     requires it only below Python 3.11 and the image is 3.11 — which is the one direction in
#     which a guard here can be LAXER than the image. Nothing walks metadata now.
_IMAGE_MODULE_POLICY = """
import os, sys

_allowed = set(sys.stdlib_module_names) | set({allowed!r}) | {{"genetics_mcp_server"}}
_hit = set()

# the finder is process-global, so it must not answer for imports httpx or matplotlib make on
# their own behalf: pip installs the pins WITH their dependencies, so those imports are honest
# in the image. Attribute each import to the innermost frame outside the import machinery and
# apply the policy only where that frame is a genetics_mcp_server source file.
_MACHINERY = ("<frozen importlib", os.path.join(os.path.dirname(os.__file__), "importlib"))


class _AbsentFromTheSandboxImage:
    def find_spec(self, fullname, path=None, target=None):
        top = fullname.split(".")[0]
        if top in _allowed:
            return None
        frame = sys._getframe(1)
        while frame is not None and frame.f_code.co_filename.startswith(_MACHINERY):
            frame = frame.f_back
        if frame is None:
            return None
        if "genetics_mcp_server" not in frame.f_code.co_filename.replace(os.sep, "/").split("/"):
            return None
        _hit.add(top)
        raise ModuleNotFoundError("absent from the sandbox image: " + fullname, name=fullname)


sys.meta_path.insert(0, _AbsentFromTheSandboxImage())
"""

# only the policy carries substitutions; the body below is a plain string so its own braces
# stay its own
_BLOCKED_IMPORT_PROBE_BODY = """
import json
import genetics_mcp_server.sdk  # noqa: F401

# The policy has to be shown to be live, and the shape of that proof is the point. Asserting
# that some named dev-only distribution lands in a blocked set couples a security guard to
# whatever the developer's venv holds, and fails hardest in a venv built from the sandbox's own
# requirements plus the SDK — the environment CLOSEST to the thing under test. Drive the policy
# directly instead: a frame carrying a shipped file's name, a module name that exists nowhere,
# and the same answer in every environment.
_pkg = os.path.dirname(genetics_mcp_server.__file__)


def _canary(source, filename):
    try:
        exec(compile(source, filename, "exec"), {})
    except ModuleNotFoundError as exc:
        return str(exc)
    return ""


print(json.dumps({
    "attempted": sorted(_hit),
    "blocked_in_a_shipped_file": _canary(
        "import __not_in_the_sandbox_image__", os.path.join(_pkg, "_canary.py")
    ),
    "allowed_in_a_shipped_file": _canary("import json", os.path.join(_pkg, "_canary.py")),
    "unattributed": _canary(
        "import __not_in_the_sandbox_image__",
        os.path.join(os.path.dirname(_pkg), "_canary.py"),
    ),
}))
"""


def test_the_sdk_imports_with_every_module_the_image_lacks_blocked():
    """The generalised form of the guard, and the point of it.

    Two separate offenders have widened the closure by one module-level import into a package
    the sandbox does not have: `config.settings` -> `dotenv` (genetics-results-suite-l41), then
    `tools/definitions.py` -> `pydantic` (genetics-results-suite-6bv). A per-offender test
    ("the SDK does not need dotenv") only ever catches the one already fixed, so this asserts
    the property instead: the SDK imports when every module outside the image's surface is
    unavailable to a shipped file. The third offender fails here, in the commit that introduces
    it, rather than in another repo's image build.

    This sees only what runs during the import. A deferred import runs at call time and is
    invisible to it by construction, which is what the static scan below is for.
    """
    result = _run_probe(
        _ORIGIN_GUARD
        + _IMAGE_MODULE_POLICY.format(allowed=sorted(SANDBOX_MODULE_NAMES))
        + _BLOCKED_IMPORT_PROBE_BODY
    )
    assert result.returncode == 0, (
        "importing the SDK reached a module the sandbox image does not install; "
        "either defer that import out of the closure or widen sandbox/requirements.txt "
        f"deliberately:\n{result.stderr}"
    )
    measured = json.loads(result.stdout.strip().splitlines()[-1])
    assert "absent from the sandbox image" in measured["blocked_in_a_shipped_file"], measured
    assert measured["allowed_in_a_shipped_file"] == "", measured
    assert "absent from the sandbox image" not in measured["unattributed"], measured


_CLOSURE_FILES_PROBE = (
    _ORIGIN_GUARD
    + """
import json, sys
import genetics_mcp_server.sdk  # noqa: F401
print(json.dumps({
    m: getattr(sys.modules[m], "__file__", None)
    for m in sys.modules
    if m.split(".")[0] == "genetics_mcp_server"
}))
"""
)


def _shipped_sources() -> dict[str, str]:
    """The files the sandbox image ships, measured rather than listed a second time.

    prune_venv.py's SDK_ALLOWLIST is the same set expressed as paths; deriving these from the
    live closure keeps this from becoming a third copy of it, and the equality test above is
    what pins the set itself.
    """
    result = _run_probe(_CLOSURE_FILES_PROBE)
    assert result.returncode == 0, result.stderr
    files = json.loads(result.stdout.strip().splitlines()[-1])
    assert all(files.values()), f"a closure module has no source file: {files}"
    return files


def _dynamic_import_target(node: ast.Call) -> str | None:
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
    if name not in ("__import__", "import_module") or not node.args:
        return None
    arg = node.args[0]
    return arg.value if isinstance(arg, ast.Constant) and isinstance(arg.value, str) else None


def _imported_top_level_names(source: str) -> list[tuple[str, int]]:
    """Every top-level module name an import in `source` names, at ANY nesting depth.

    `ast.walk` rather than a scan of the module body, because depth is the whole point: a
    function-level import, one inside `try`/`except ImportError`, one under `if TYPE_CHECKING`
    and one in a module `__getattr__` are all invisible to anything that imports the package
    and looks at what loaded.

    Relative imports are skipped — they resolve inside genetics_mcp_server by construction, and
    what is inside the package is the closure test's question, not this one.
    """
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found += [(alias.name.split(".")[0], node.lineno) for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and not node.level:
            found.append(((node.module or "").split(".")[0], node.lineno))
        elif isinstance(node, ast.Call):
            target = _dynamic_import_target(node)
            if target:
                found.append((target.split(".")[0], node.lineno))
    return [(name, lineno) for name, lineno in found if name]


def test_no_shipped_source_names_a_module_the_image_lacks():
    """The deferred-import shape, which every runtime guard is blind to.

    `import genetics_mcp_server.sdk` executes module bodies and nothing else, so a
    `from ddgs import DDGS` inside a method passes every check this file makes and then raises
    ModuleNotFoundError inside a container with no shell and no package manager — which has
    shipped. Deferring an import is also this codebase's house style for adding capability, so
    the blind spot sits precisely where new code lands.

    `if TYPE_CHECKING` and `try/except ImportError` guards are held to the same rule, for the
    reason the assertion below states; `typing.get_type_hints()` on the annotations would try
    to resolve a TYPE_CHECKING import for real regardless of the guard.

    Over-strictness is the hazard here rather than laxness — a false positive breaks the build
    for everyone — so the allowed set is the union of the standard library, the pins, and the
    package itself, and `__future__` and relative imports are inside it by construction.
    """
    allowed = set(sys.stdlib_module_names) | SANDBOX_MODULE_NAMES | {"genetics_mcp_server"}
    offenders = []
    for module, path in sorted(_shipped_sources().items()):
        for name, lineno in _imported_top_level_names(Path(path).read_text()):
            if name not in allowed:
                offenders.append(f"{path}:{lineno} imports {name!r} ({module})")
    assert not offenders, (
        "the sandbox image ships these files, and they name modules it does not install. "
        "Deferring the import does not help — it moves the failure from the build to a "
        "call inside the sandbox — and neither does a `TYPE_CHECKING` or `try/except "
        "ImportError` guard: the name is in the shipped source whether or not that line "
        "runs. Move the code out of the SDK's import closure, or widen "
        "genetics-results-suite/sandbox/requirements.txt deliberately:\n  "
        + "\n  ".join(offenders)
    )


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
