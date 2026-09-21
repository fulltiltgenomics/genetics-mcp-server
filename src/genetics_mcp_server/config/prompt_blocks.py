"""The system-prompt block type, shared by every prompt variant.

It lives apart from `defaults` so a variant module can import it without importing the
module that holds the registry those variants are registered in — the cycle that would
otherwise force the registration to happen at the bottom of a file, where it breaks
depending on which module the process imports first.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Block:
    r"""One fragment of the system prompt, with the conditions for emitting it.

    `text` owns its surrounding newlines so that concatenating the emitted blocks
    reproduces the document structure with no re-joining.

    Three rules for writing the text, none of which the gate can enforce for you:

    1. NAME TOOLS EXACTLY. The gate matches `(?<!\w)NAME\b` (see `tools_named_in`), so a
       plural or suffixed mention — `get_hla_by_alleles` where the tool is
       `get_hla_by_allele` — is not seen as a mention at all, and the block is then
       emitted on surfaces that do not have the tool. No test catches that:
       tests/test_system_prompt.py's scan is independent of the gate on the ALGORITHM
       (tokenise-then-intersect vs per-name regex) but shares its NORMALISATION, so the
       two agree with each other while both being wrong. There is no live instance today
       (genetics-results-suite-4h6.78); keep it that way by naming tools verbatim and
       rephrasing the sentence around the exact name.

    2. A PROHIBITION GATES POSITIVELY. `_assemble` asks only WHICH names appear in the
       text, never with what polarity, so a block written to warn AGAINST a tool requires
       that tool to be available. Remove the tool from a surface and the warning
       disappears — along with everything else that block carries. The
       genetics-results-suite-4h6.17/.69 cycle fixed the live instance (the HLA block,
       where `get_summary_stats` appeared only inside a negation); the property remains.
       If a rule has to outlive the tool it warns about, put the warning in its own block.

    3. GATE ON WHAT A RULE NEEDS, NOT ON WHAT IT IS ABOUT. Science and grounding belong in
       blocks that name no tool; only the "which tool" clause is gated. The line between
       the two is whether the rule can be OBEYED without rows:
       - An obligation that attaches when the model PRESENTS data holds on every surface,
         because a surface with no data path can still present data it retrieved from a
         document. So the pseudo-credible-set labelling duty ("not statistically
         fine-mapped", "always tell the user explicitly") and the construction facts
         needed to read such a result (the r² membership criteria, the PIP caution) are
         ungated, and reach `rag`.
       - A rule that can only be carried out by FETCHING rows — membership is whatever
         `credible_sets_v` returns, re-query rather than answer from memory — is gated on
         having a path to those rows: on a surface without one it names an action the
         model cannot take, and "verify it" with nothing to verify against is worse than
         silence (genetics-results-suite-4h6.79).
    """

    text: str
    # emitted only if at least one of these is available; use for a section whose own text
    # names no tool but which presupposes a capability (e.g. SQL guidance, reachable either
    # through query_database or through the SDK's sql() inside run_analysis)
    requires_any: frozenset[str] = field(default_factory=frozenset)
    # suppressed if any of these is available; use to pick between mutually exclusive
    # wordings of the same guidance for different tool surfaces
    excludes: frozenset[str] = field(default_factory=frozenset)
    # emitted only if ALL of these are available. The text-derived name gate is itself an
    # implicit requires_all, so guidance whose emission is a real precondition on a tool
    # used to be expressed by happening to name that tool — which made it hostage to every
    # OTHER name in the same text, including illustrative "e.g." asides. State the
    # precondition here and keep the asides in their own blocks instead.
    requires_all: frozenset[str] = field(default_factory=frozenset)
    # supplies the block's text at assemble time instead of `text`, and returning None drops
    # the block. For a rule whose wording depends on the deployment rather than on the tool
    # list — the one gate above cannot express "say this only if the value exists". The
    # tool-name gate reads `text`, which is empty for such a block, so a dynamic block must
    # state its tool precondition in `requires_any` rather than by naming the tool.
    render: Callable[[], str | None] | None = None


def _fs(*names: str) -> frozenset[str]:
    return frozenset(names)


def block_text(block: _Block) -> str | None:
    return block.text if block.render is None else block.render()


# how long "the fetcher did not answer" is believed before probing again. Prompt assembly runs
# on every chat request, so an unavailable fetcher must not cost a connect attempt per turn:
# only successes are cached, so without this window every request on the replica would stall
# on the probe's worst case (~2 s: a connect that reaches its limit, then a read that does).
# A replica that starts while the fetcher is unreachable therefore changes the shared prompt
# prefix once, within its first minute, when the retry succeeds.
_URL_FETCH_HOSTS_RETRY_S = 60.0
# (hosts, when it was read). A successful read is kept for the life of the process and goes
# stale when the fetcher's list is widened; url_fetch_client.UrlFetchClient.allowed_hosts says
# why that is affordable and what an operator has to do about it.
_url_fetch_hosts: tuple[tuple[str, ...] | None, float] | None = None


def _probe_url_fetch_allowed_hosts() -> tuple[str, ...] | None:
    """The fetcher's allow-list, read through the process-wide client.

    Imported inside the call: `config` is imported by nearly everything, and
    `url_fetch_client` imports settings and httpx.

    The client's API is async and prompt assembly is not, so when a request's event loop is
    already running the one probe happens on a thread of its own — `asyncio.run` cannot nest.
    """
    from genetics_mcp_server.url_fetch_client import (
        UrlFetchNotConfigured,
        get_url_fetch_client,
    )

    try:
        client = get_url_fetch_client()
    except UrlFetchNotConfigured:
        return None
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(client.allowed_hosts())
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(client.allowed_hosts())).result()


def url_fetch_allowed_hosts() -> tuple[str, ...] | None:
    global _url_fetch_hosts
    now = time.monotonic()
    if _url_fetch_hosts is not None:
        hosts, read_at = _url_fetch_hosts
        if hosts is not None or now - read_at < _URL_FETCH_HOSTS_RETRY_S:
            return hosts
    try:
        hosts = _probe_url_fetch_allowed_hosts()
    except Exception:  # noqa: BLE001 - a prompt is never worth failing a chat turn for
        logger.warning("could not read the url-fetcher allow-list", exc_info=True)
        hosts = None
    _url_fetch_hosts = (hosts, now)
    return hosts


def url_input_hosts_rule() -> str | None:
    """Names the hosts a URL input can come from, or says nothing at all.

    Silence is the right answer when the list is UNKNOWN — no fetcher configured, unreachable,
    or a fetcher too old to say: the model learns the policy from a refusal either way, and a
    prompt that states the wrong hosts, or states a policy on a deployment that has no fetcher,
    costs more than one that states none. An EMPTY list is not unknown, and gets its own
    sentence.
    """
    hosts = url_fetch_allowed_hosts()
    if hosts is None:
        return None
    if not hosts:
        # a fetcher configured to reach nowhere is a knowable answer, not an unknown one: the
        # model is better off asking for an upload than spending a turn discovering it
        return (
            "- **URL inputs are unavailable in this deployment; ask the user to upload the "
            "file.**\n"
        )
    return (
        f"- **URL inputs are fetched only from these hosts: {', '.join(hosts)}.** A file "
        "anywhere else is unreachable — do not try another host; ask the user to upload it.\n"
    )
