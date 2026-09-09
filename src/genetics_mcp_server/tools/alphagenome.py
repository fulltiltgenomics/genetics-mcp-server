"""Client for the AlphaGenome Atlas API (regulatory-track variant effect prediction).

Transport (the gRPC Atlas client, the per-minute token bucket, quota backoff, the track
metadata cache), cell-type resolution, AnnData flattening and the tool methods all live
here, the way tools/uniprot.py and tools/chembl.py keep their logic out of executor.py so
they can be tested without a ToolExecutor. No method raises for an AlphaGenome-side
problem: a tool method returns a `success: False` dict naming the stage, everything below
it returns an error sentinel.

Reached from ServerToolExecutor only. `alphagenome` is not installed in the sandbox image
and this module is not on the suite's sandbox/prune_venv.py SDK_ALLOWLIST, so it must
never be imported from tools/executor.py -- not even from inside a method, where the
deferred intra-package import satisfies every build gate and then raises
ModuleNotFoundError at call time in a container with no shell.

The two quantities this client can report are not interchangeable and the choice is
per-modality, not per-caller: see MODALITIES.
"""

import asyncio
import logging
import math
import os
import re
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

# hard runtime dependency, not a convenience: the Atlas API answers in AnnData and this
# module is the boundary where that stops -- nothing above it may see an AnnData object.
# It arrives with the `alphagenome` distribution, so it costs nothing extra.
import anndata
from alphagenome.atlas import atlas
from alphagenome.data import genome

if TYPE_CHECKING:
    # type-only for the same reason as in tools/uniprot.py: a real import of
    # config.settings would pull the module enumerating every internal env var name into
    # the SDK's import closure.
    from genetics_mcp_server.config.settings import Settings

logger = logging.getLogger(__name__)

# gRPC to gdmscience.googleapis.com; a scored batch is several seconds of model time
_TIMEOUT = 120.0

# Measured against the live API: request 1321 inside 60.5s is refused with
# RESOURCE_EXHAUSTED and no retry-after, so 1320 in a rolling minute is the ceiling and
# the only useful response to reaching it is not to make the request.
_RATE_LIMIT_PER_MINUTE = 1320
_RATE_WINDOW = 60.0

# The quota carries no retry-after, so the schedule is ours. It is per-minute, so waiting
# out a full window is the worst case and three doublings from 2s stay inside it.
_QUOTA_RETRIES = 3
_QUOTA_BACKOFF_BASE = 2.0

# A batch is one RPC per variant behind the SDK's thread pool -- plus one `scorer_metadata`
# call `query_variants` makes for itself on every batch -- and all-or-nothing in front of it:
# it re-raises the first failure and returns nothing for any variant. A failed batch is
# therefore re-run one variant at a time, which is what bounds the batch size rather than any
# API limit. The metadata call would stop counting if the SDK ever cached it across calls.
_MAX_VARIANTS = 25

# per-modality track detail on the response, ordered by |value|
_TOP_TRACKS = 5
# a gene-resolved modality answers one row per gene, and a variant can sit near many
_MAX_GENES = 20

_SIGNED = "signed"
_MAGNITUDE = "magnitude"

_CALIBRATED = "calibrated"
_UNVALIDATED = "unvalidated"


@dataclass(frozen=True)
class Modality:
    """One AlphaGenome scorer and the rule for reading its delta.

    `quantity` is the whole point of this table. Measured on this suite's own data, the
    signed delta beats |delta| decisively where the sign means something (DNASE against
    caQTL beta: rho +0.478 signed, +0.200 magnitude), and the splice sign means nothing
    the suite can line up against -- sQTL beta orients to a leafcutter intron cluster and
    the splice delta has no corresponding orientation, so its signed correlation's CI
    spans zero. One rule for every modality therefore either throws away most of the
    signal or invents a direction.

    `population_rho` is a cohort-level Spearman correlation against
    `calibrated_against`. It describes the modality, never the variant in hand.
    """

    name: str
    tier: int
    quantity: str
    calibrated_against: str | None = None
    population_rho: float | None = None
    gene_resolved: bool = False

    @property
    def status(self) -> str:
        return _CALIBRATED if self.population_rho is not None else _UNVALIDATED


# Tiers 1-3 were calibrated against this suite's own QTL and MPRA measurements; tier 4 has
# no local substrate, so it ships flagged and without a number rather than with a borrowed
# one.
# Tier 4's `quantity` is a judgement call, not a measurement: the sign is kept where
# it denotes more or less of something a caller can name (binding, initiation,
# polyadenylation) and dropped for CONTACT_MAPS, whose score aggregates a matrix and has
# no orientation to keep. The names are the SDK's own scorer keys.
# `gene_resolved` is read off the SDK's scorer classes, not off a live response:
# SPLICE_SITES and SPLICE_SITE_USAGE are GeneMaskSplicingScorer, SPLICE_JUNCTIONS is
# SpliceJunctionScorer, POLYADENYLATION is PolyadenylationScorer, and all of them carry the
# gene metadata `atlas.convert_variant_scores_to_anndata` turns into one obs row per gene.
# Flagging is the safe direction: the per-gene path reads a one-row response correctly,
# while the single-row path silently keeps whichever gene the API returned first. Only
# RNA_SEQ was confirmed against a live response; the claim is false if one of the other four
# ever answers with a row that carries no gene at all.
MODALITIES: dict[str, Modality] = {
    m.name: m
    for m in (
        Modality("DNASE", 1, _SIGNED, "caQTL beta", 0.478),
        Modality("ATAC", 1, _SIGNED, "caQTL beta", 0.489),
        Modality("CHIP_HISTONE", 1, _SIGNED, "MPRA log2Skew", 0.472),
        Modality("SPLICE_SITES", 2, _MAGNITUDE, "sQTL beta", 0.249, gene_resolved=True),
        Modality("SPLICE_JUNCTIONS", 2, _MAGNITUDE, "sQTL beta", 0.231, gene_resolved=True),
        Modality("RNA_SEQ", 3, _SIGNED, "eQTL beta", 0.112, gene_resolved=True),
        Modality("SPLICE_SITE_USAGE", 3, _MAGNITUDE, "sQTL beta", 0.148, gene_resolved=True),
        Modality("CHIP_TF", 4, _SIGNED),
        Modality("CAGE", 4, _SIGNED),
        Modality("PROCAP", 4, _SIGNED),
        Modality("POLYADENYLATION", 4, _SIGNED, gene_resolved=True),
        Modality("CONTACT_MAPS", 4, _MAGNITUDE),
    )
}

# 22 scorers and 9,440 tracks are available; these twelve are the ones with a stated
# reading rule, and the default asks only for the calibrated ones.
DEFAULT_MODALITIES: tuple[str, ...] = tuple(
    name for name, m in MODALITIES.items() if m.tier <= 3
)

# The suite's BigQuery views encode X as 23. AlphaGenome accepts 'chrX' and rejects
# 'chr23', '23' and bare 'X' with ValueError('Chromosome ... not found.'), so this map is
# mandatory rather than cosmetic.
# UNVERIFIED: only 23 -> chrX was checked against the live API. 24/25/26 -> chrY/chrM are
# the same convention read off the same views, believed but not measured.
_CHROM_ALIASES = {
    "23": "chrX",
    "24": "chrY",
    "25": "chrM",
    "26": "chrM",
    "X": "chrX",
    "Y": "chrY",
    "M": "chrM",
    "MT": "chrM",
}

_VARIANT_RE = re.compile(
    r"^(?P<chr>[0-9A-Za-z]+)[:_-](?P<pos>\d+)[:_-](?P<ref>[ACGTN]+)[:_-](?P<alt>[ACGTN]+)$",
    re.IGNORECASE,
)

_ATTRIBUTION = {
    "source": "AlphaGenome (Google DeepMind)",
    "assembly": "GRCh38/hg38",
    "note": (
        "Predicted regulatory effects, not measurements. Each modality carries its "
        "validation tier and the population-level rho behind it."
    ),
}


def _is_error(data: Any) -> bool:
    """True when `data` is an error sentinel rather than a result.

    Callers must use this rather than `"_error" in data`: some of what flows through here
    is a list or a mapping of AnnData, on which `in` degrades to a membership test.
    """
    return isinstance(data, dict) and "_error" in data


def _error(stage: str, message: str) -> dict[str, Any]:
    return {"_error": message, "_stage": stage}


def normalise_chromosome(value: Any) -> str | None:
    """AlphaGenome's chromosome spelling, or None when there is no valid one.

    Accepts what the suite's views and the other tools here emit -- 23, 'chr23', 'X',
    'chrX' -- and answers only in the 'chrN'/'chrX' form AlphaGenome will take.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text[:3].lower() == "chr":
        text = text[3:]
    upper = text.upper()
    if upper in _CHROM_ALIASES:
        return _CHROM_ALIASES[upper]
    if upper.isdigit() and 1 <= int(upper) <= 22:
        return f"chr{int(upper)}"
    return None


def parse_variant(variant_id: str) -> dict[str, Any]:
    """`chr:pos:ref:alt` -> a normalised variant dict, or an error sentinel.

    Everything AlphaGenome can be told about without asking it is rejected here, before
    anything reaches the API, because a batch is all-or-nothing: one variant the API
    dislikes raises for every variant in the batch.
    """
    text = str(variant_id or "").strip()
    match = _VARIANT_RE.match(text)
    if not match:
        return _error("input", f"{text!r} is not a chr:pos:ref:alt variant id")
    chromosome = normalise_chromosome(match.group("chr"))
    if chromosome is None:
        return _error("input", f"{match.group('chr')!r} is not a chromosome AlphaGenome knows")
    return {
        "id": text,
        "chromosome": chromosome,
        "position": int(match.group("pos")),
        "reference_bases": match.group("ref").upper(),
        "alternate_bases": match.group("alt").upper(),
    }


def population_rho_for_modality(name: str) -> float | None:
    """Cohort-level Spearman rho for a modality against its calibration substrate.

    NOT a per-variant confidence, and it must never be rendered as one: it is the
    correlation of this modality's predictions with measured effects across a cohort of
    variants, and says nothing about the variant in hand. None means uncalibrated.
    """
    modality = MODALITIES.get(name)
    return modality.population_rho if modality else None


def validation_metadata(name: str) -> dict[str, Any]:
    """A modality's validation status as data, so callers render it structurally."""
    modality = MODALITIES.get(name)
    if modality is None:
        return {"tier": None, "status": _UNVALIDATED, "population_rho": None}
    return {
        "tier": modality.tier,
        "status": modality.status,
        "quantity": modality.quantity,
        "calibrated_against": modality.calibrated_against,
        "population_rho": modality.population_rho,
        # names the scope of the number above so no renderer has to infer it
        "rho_scope": "population" if modality.population_rho is not None else None,
    }


# --------------------------------------------------------------------------- #
# AnnData -> plain Python
# --------------------------------------------------------------------------- #


def _plain(value: Any) -> Any:
    """A numpy/pandas scalar as a JSON-able Python one; NaN becomes None."""
    item = getattr(value, "item", None)
    if callable(item) and not isinstance(value, (str, bytes)):
        try:
            value = item()
        except (ValueError, TypeError):
            return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def _rows(frame: Any) -> list[dict[str, Any]]:
    """An AnnData .obs/.var as a list of plain dicts."""
    if frame is None:
        return []
    to_dict = getattr(frame, "to_dict", None)
    if callable(to_dict):
        try:
            records = to_dict("records")
        except TypeError:
            records = None
        if isinstance(records, list):
            return [{k: _plain(v) for k, v in row.items()} for row in records]
    if isinstance(frame, list):
        return [{k: _plain(v) for k, v in row.items()} for row in frame]
    return []


def _matrix(values: Any) -> list[list[Any]]:
    """An AnnData .X or layer as a list of rows of plain floats."""
    if values is None:
        return []
    tolist = getattr(values, "tolist", None)
    raw = tolist() if callable(tolist) else values
    if not isinstance(raw, list):
        return []
    if raw and not isinstance(raw[0], list):
        raw = [raw]
    return [[_plain(cell) for cell in row] for row in raw]


def _first(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def _variant_key(value: Any) -> str:
    """A comparable key for whatever sits in an obs `variant` cell."""
    fields = ("chromosome", "position", "reference_bases", "alternate_bases")
    if all(hasattr(value, f) for f in fields):
        return ":".join(str(getattr(value, f)) for f in fields)
    if isinstance(value, dict):
        return ":".join(str(value.get(f, "")) for f in fields)
    return str(value)


# --------------------------------------------------------------------------- #
# track selection
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TrackSelection:
    """Which track columns a cell-type request resolved to."""

    requested: str | None
    indices: tuple[int, ...]
    biosamples: tuple[str, ...]
    matched: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "resolved_biosamples": list(self.biosamples),
            "matched": self.matched,
        }


_ALL_TRACKS = TrackSelection(requested=None, indices=(), biosamples=(), matched=False)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


# --------------------------------------------------------------------------- #
# rate limiting
# --------------------------------------------------------------------------- #


class _RateLimiter:
    """Sliding-window limiter for AlphaGenome's per-minute burst quota.

    A window, not a steady rate: the measured refusal is of the 1321st request inside one
    minute, so smoothing requests out would cost throughput and buy nothing. The clock and
    sleep are injected so expiry is testable without waiting a minute.
    """

    def __init__(
        self,
        limit: int = _RATE_LIMIT_PER_MINUTE,
        window: float = _RATE_WINDOW,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Any] = asyncio.sleep,
    ):
        self._limit = limit
        self._window = window
        self._clock = clock
        self._sleep = sleep
        self._hits: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self, cost: int = 1) -> None:
        """Take `cost` slots. The caller states the cost because only it knows how many
        RPCs its call becomes: the SDK fans a batch out into one per variant behind a thread
        pool and fetches scorer metadata alongside, so charging a batch as one would let a
        single call walk straight through the quota."""
        cost = max(1, min(cost, self._limit))
        # the lock spans the wait, so queued callers leave in arrival order rather than
        # all waking to race for the slots that freed
        async with self._lock:
            while True:
                now = self._clock()
                cutoff = now - self._window
                while self._hits and self._hits[0] <= cutoff:
                    self._hits.popleft()
                if len(self._hits) + cost <= self._limit:
                    self._hits.extend([now] * cost)
                    return
                await self._sleep(max(self._hits[0] + self._window - now, 0.0))

    def clear(self) -> None:
        self._hits.clear()


# module level, not per client: the quota belongs to the API key, and clients are
# constructed independently by each executor, so a per-instance limiter would each think
# it had the whole minute to itself.
_LIMITER = _RateLimiter()

# The scorer metadata is one round trip that does not change within a process, and it is
# what cell-type matching resolves against -- matching is worth ~0.06 rho over taking the
# extreme across all tissues. Module level for the same reason as the limiter.
_METADATA_CACHE: dict[str, Any] = {}
_METADATA_LOCK = asyncio.Lock()


def _grpc_code(exc: BaseException) -> str | None:
    """The gRPC status name of an error, without importing grpc to find out."""
    code = getattr(exc, "code", None)
    if not callable(code):
        return None
    try:
        value = code()
    except Exception:  # noqa: BLE001 - a foreign object's code() is not our contract
        return None
    return getattr(value, "name", None) or str(value)


def _is_quota(exc: BaseException) -> bool:
    return _grpc_code(exc) == "RESOURCE_EXHAUSTED"


# --------------------------------------------------------------------------- #
# the client
# --------------------------------------------------------------------------- #


class AlphaGenomeClient:
    """Async client for the AlphaGenome Atlas variant scorers.

    The SDK's transport is synchronous gRPC, so every call crosses `asyncio.to_thread`;
    there is no httpx client to inject, unlike the REST clients next door.
    """

    def __init__(
        self,
        settings: "Settings | None" = None,
        *,
        api_key: str | None = None,
        atlas_factory: Callable[[str], Any] | None = None,
        limiter: _RateLimiter | None = None,
        sleep: Callable[[float], Any] = asyncio.sleep,
    ):
        # getattr rather than an attribute access: the settings field is added by its own
        # subtask, and the environment is the source of record for the key either way.
        configured = api_key or getattr(settings, "alphagenome_api_key", None)
        self._api_key = configured or os.environ.get("ALPHAGENOME_API_KEY") or ""
        self._address = getattr(settings, "alphagenome_address", None)
        self._timeout = float(getattr(settings, "alphagenome_timeout", _TIMEOUT) or _TIMEOUT)
        self._atlas_factory = atlas_factory or self._default_factory
        self._limiter = limiter or _LIMITER
        self._sleep = sleep
        self._atlas: Any = None

    def _default_factory(self, api_key: str) -> Any:
        kwargs: dict[str, Any] = {"timeout": self._timeout}
        if self._address:
            kwargs["address"] = self._address
        return atlas.create(api_key, **kwargs)

    # ---- transport -------------------------------------------------------------

    def _redact(self, text: str) -> str:
        """No error message, log line or tool result may carry the API key."""
        if self._api_key and self._api_key in text:
            return text.replace(self._api_key, "***")
        return text

    def _classify(self, exc: BaseException, stage: str) -> dict[str, Any]:
        """Map an SDK exception onto the error taxonomy, as a sentinel.

        The SDK translates most gRPC statuses for us -- INVALID_ARGUMENT and NOT_FOUND
        become ValueError, DEADLINE_EXCEEDED becomes TimeoutError -- but RESOURCE_EXHAUSTED
        has no case there and arrives as the raw grpc error, which is why the quota is
        recognised by status name rather than by type.
        """
        message = self._redact(str(exc) or exc.__class__.__name__)
        if _is_quota(exc):
            return _error("quota", f"AlphaGenome quota exhausted: {message}")
        if isinstance(exc, (TimeoutError, asyncio.TimeoutError)) or (
            _grpc_code(exc) == "DEADLINE_EXCEEDED"
        ):
            return _error("timeout", f"AlphaGenome timed out: {message}")
        if isinstance(exc, PermissionError):
            return _error("auth", "AlphaGenome refused the API key")
        if isinstance(exc, ValueError):
            lowered = message.lower()
            if "chromosome" in lowered:
                return _error("chromosome", f"AlphaGenome rejected the chromosome: {message}")
            if "reference" in lowered or "expected" in lowered:
                return _error(
                    "reference_mismatch",
                    f"reference base does not match GRCh38: {message}",
                )
        return _error(stage, f"AlphaGenome {stage} failed: {message}")

    async def _call(self, stage: str, func: Callable[[], Any], cost: int = 1) -> Any:
        """One rate-limited, quota-retrying call into the synchronous SDK.

        Returns an error sentinel rather than raising; nothing above this re-raises.
        """
        if not self._api_key:
            return _error("config", "ALPHAGENOME_API_KEY is not set")
        delay = _QUOTA_BACKOFF_BASE
        for attempt in range(_QUOTA_RETRIES + 1):
            try:
                # inside the try: the limiter's lock is created at import and shared
                # process-wide, so a contended acquire from a second event loop raises
                # instead of waiting, and nothing above this may see an exception
                await self._limiter.acquire(cost)
                return await asyncio.to_thread(func)
            except Exception as exc:  # noqa: BLE001 - the contract is never to raise
                if _is_quota(exc) and attempt < _QUOTA_RETRIES:
                    logger.warning("AlphaGenome quota hit, backing off %.1fs", delay)
                    await self._sleep(delay)
                    delay *= 2
                    continue
                logger.warning("AlphaGenome %s failed: %s", stage, self._redact(str(exc)))
                return self._classify(exc, stage)
        return _error("quota", "AlphaGenome quota exhausted after backoff")

    def _atlas_client(self) -> Any:
        if self._atlas is None:
            self._atlas = self._atlas_factory(self._api_key)
        return self._atlas

    # ---- track metadata and cell-type matching ---------------------------------

    async def track_metadata(self) -> Any:
        """Track metadata rows per modality, fetched once per process and cached."""
        key = self._address or "default"
        cached = _METADATA_CACHE.get(key)
        if cached is not None:
            return cached
        async with _METADATA_LOCK:
            cached = _METADATA_CACHE.get(key)
            if cached is not None:
                return cached
            raw = await self._call("metadata", lambda: self._atlas_client().scorer_metadata())
            if _is_error(raw):
                return raw
            rows = _metadata_by_modality(raw)
            _METADATA_CACHE[key] = rows
            return rows

    async def resolve_tracks(self, modality: str, cell_type: str | None) -> TrackSelection:
        """Track columns for a requested cell type or tissue.

        Cell-type matching is a client feature rather than a caller concern: matched
        tracks are worth ~0.06 rho over taking the extreme across all tissues, and the
        caller has no way to reach the biosample vocabulary. An unresolvable request falls
        back to all tracks with `matched` False rather than to no answer.
        """
        if not cell_type:
            return _ALL_TRACKS
        metadata = await self.track_metadata()
        if _is_error(metadata):
            return TrackSelection(cell_type, (), (), False)
        needle = _slug(cell_type)
        indices: list[int] = []
        biosamples: list[str] = []
        for index, row in enumerate(metadata.get(modality, [])):
            name = _first(row, "biosample_name", "gtex_tissue", "biosample", "tissue")
            if name is None:
                continue
            slug = _slug(str(name))
            if not slug:
                continue
            if needle in slug or slug in needle:
                indices.append(index)
                if str(name) not in biosamples:
                    biosamples.append(str(name))
        return TrackSelection(cell_type, tuple(indices), tuple(biosamples), bool(indices))

    # ---- tool methods ----------------------------------------------------------

    async def score_variant(
        self,
        variant_id: str,
        cell_type: str | None = None,
        modalities: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Predicted regulatory effect of one variant, per modality."""
        batch = await self.score_variants([variant_id], cell_type, modalities)
        if not batch.get("success"):
            return batch
        return {**(batch.get("results") or [{}])[0], "attribution": _ATTRIBUTION}

    async def score_variants(
        self,
        variant_ids: Sequence[str],
        cell_type: str | None = None,
        modalities: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Predicted regulatory effects for a batch, one result per requested variant."""
        requested = list(variant_ids or [])
        if not requested:
            return _failed("input", "no variants given", {"n_requested": 0})
        if len(requested) > _MAX_VARIANTS:
            return _failed(
                "input",
                f"at most {_MAX_VARIANTS} variants per call, got {len(requested)}",
                {"n_requested": len(requested)},
            )
        names = _select_modalities(modalities)
        if _is_error(names):
            return _failed(names["_stage"], names["_error"], {"n_requested": len(requested)})

        parsed = [(vid, parse_variant(vid)) for vid in requested]
        valid = [variant for _, variant in parsed if not _is_error(variant)]
        selections = {name: await self.resolve_tracks(name, cell_type) for name in names}
        scored = await self._score_batch(valid, names) if valid else {}

        results: list[dict[str, Any]] = []
        for vid, variant in parsed:
            if _is_error(variant):
                results.append(_failed(variant["_stage"], variant["_error"], {"variant_id": vid}))
                continue
            outcome = scored.get(vid)
            if outcome is None or _is_error(outcome):
                sentinel = outcome or _error("prediction", "no prediction returned")
                results.append(
                    _failed(sentinel["_stage"], sentinel["_error"], {"variant_id": vid})
                )
                continue
            results.append(
                {
                    "success": True,
                    "variant": variant,
                    "cell_type": cell_type,
                    "modalities": _flatten_all(outcome, names, selections),
                }
            )
        return {
            "success": True,
            "n_requested": len(requested),
            "n_scored": sum(1 for r in results if r.get("success")),
            "results": results,
            "attribution": _ATTRIBUTION,
        }

    async def _score_batch(
        self, variants: list[dict[str, Any]], names: Sequence[str]
    ) -> dict[str, Any]:
        """Score a batch, isolating the offender when the batch is destroyed.

        `query_variants` fans out one RPC per variant behind a thread pool and re-raises
        the first failure, so a single variant AlphaGenome dislikes returns nothing for
        every variant. Everything that can be checked without asking is already rejected
        in `parse_variant`; what survives to here is what only the API can judge, chiefly
        a reference base that does not match GRCh38.
        """
        if len(variants) > 1:
            batch = await self._call(
                "prediction",
                lambda: self._atlas_client().query_variants(
                    [_sdk_variant(v) for v in variants],
                    requested_scorers=list(names),
                    progress_bar=False,
                ),
                # +1 for the scorer_metadata round trip query_variants makes per batch
                cost=len(variants) + 1,
            )
            if not _is_error(batch):
                split = _split_by_variant(batch, variants)
                if split is not None:
                    return split
                logger.warning("AlphaGenome batch could not be split by variant; isolating")
        return {v["id"]: await self._query_one(v, names) for v in variants}

    async def _query_one(self, variant: dict[str, Any], names: Sequence[str]) -> Any:
        return await self._call(
            "prediction",
            lambda: self._atlas_client().query_variant(
                _sdk_variant(variant), requested_scorers=list(names)
            ),
        )


def _sdk_variant(variant: dict[str, Any]) -> Any:
    """The SDK's Variant, carrying the caller's id so a batch can be split back apart."""
    return genome.Variant(
        chromosome=variant["chromosome"],
        position=variant["position"],
        reference_bases=variant["reference_bases"],
        alternate_bases=variant["alternate_bases"],
        name=variant["id"],
    )


def _select_modalities(requested: Sequence[str] | None) -> Any:
    if not requested:
        return list(DEFAULT_MODALITIES)
    names = [str(name).strip().upper() for name in requested]
    unknown = [name for name in names if name not in MODALITIES]
    if unknown:
        return _error("input", f"unknown modalities: {', '.join(sorted(unknown))}")
    return names


def _metadata_by_modality(raw: Any) -> dict[str, list[dict[str, Any]]]:
    """`scorer_metadata()` as {modality: [track row, ...]}.

    An SDK answer that is not a mapping yields nothing rather than an AttributeError, which
    would leave `score_variants` by a path that has no `success: False` to return.
    """
    out: dict[str, list[dict[str, Any]]] = {}
    if not hasattr(raw, "items"):
        return out
    for key, value in raw.items():
        frame = getattr(value, "track_metadata", value)
        out[str(key).upper()] = _rows(frame)
    return out


def _split_by_variant(
    batch: Any, variants: list[dict[str, Any]]
) -> dict[str, dict[str, Any]] | None:
    """Take a {modality: AnnData} batch response apart into one per variant.

    A batch answers with one AnnData per modality whose obs rows carry the variant they
    came from, so the split is by obs row rather than positional: a modality that scored
    a variant against several genes contributes several rows, and one that scored it not
    at all contributes none. Returns None when a modality's obs does not identify the
    variant, because guessing would attribute one variant's scores to another.
    """
    if not isinstance(batch, dict) or not batch:
        return None
    wanted = {_variant_key(_sdk_variant(v)): v["id"] for v in variants}
    if len(wanted) != len(variants):
        # two spellings of one locus ("1:100:A:G" and "chr1:100:A:G") collapse to one key,
        # and the id that loses would be reported as scored with every modality null
        return None
    out: dict[str, dict[str, Any]] = {v["id"]: {} for v in variants}
    for modality, scores in batch.items():
        obs = getattr(scores, "obs", None)
        if obs is None or "variant" not in getattr(obs, "columns", ()):
            return None
        keys = [_variant_key(value) for value in obs["variant"]]
        for key, variant_id in wanted.items():
            mask = [k == key for k in keys]
            if any(mask):
                out[variant_id][str(modality).upper()] = scores[mask]
    return out


# --------------------------------------------------------------------------- #
# flattening
# --------------------------------------------------------------------------- #


def _flatten_all(
    prediction: Any,
    names: Sequence[str],
    selections: dict[str, TrackSelection],
) -> dict[str, Any]:
    """Every requested modality's flattened scores, keyed by modality name."""
    by_modality = {str(k).upper(): v for k, v in (prediction or {}).items()}
    out: dict[str, Any] = {}
    for name in names:
        selection = selections.get(name, _ALL_TRACKS)
        scores = by_modality.get(name)
        if not isinstance(scores, anndata.AnnData):
            out[name] = {
                "modality": name,
                "quantity": MODALITIES[name].quantity if name in MODALITIES else None,
                "value": None,
                "quantile": None,
                "top_tracks": [],
                "note": "AlphaGenome returned no scores for this modality",
                "cell_type_match": selection.as_dict(),
                "validation": validation_metadata(name),
            }
            continue
        out[name] = flatten_scores(scores, name, selection)
    return out


def flatten_scores(
    scores: Any, modality_name: str, selection: TrackSelection = _ALL_TRACKS
) -> dict[str, Any]:
    """An AnnData of scores as plain JSON-able Python.

    This is the boundary: no AnnData object travels above it. The exposed quantity follows
    the modality's rule -- the signed delta where the sign is commensurable with something
    the suite measures, the magnitude where it is not -- and the quantile from
    `layers['quantiles']` travels with it, because the raw number alone misleads: APOE
    rs429358's AVI of 0.4999 is the 98.7th percentile.
    """
    modality = MODALITIES.get(modality_name)
    quantity = modality.quantity if modality else _MAGNITUDE
    values = _matrix(getattr(scores, "X", None))
    layers = getattr(scores, "layers", None)
    quantiles = _matrix(layers.get("quantiles")) if layers is not None else []
    var_rows = _rows(getattr(scores, "var", None))
    obs_rows = _rows(getattr(scores, "obs", None))

    n_tracks = len(values[0]) if values else len(var_rows)
    columns, matched = _columns(selection, var_rows, n_tracks)

    base: dict[str, Any] = {
        "modality": modality_name,
        "quantity": quantity,
        "n_tracks_scored": len(columns),
        "cell_type_match": {**selection.as_dict(), "matched": matched},
        "validation": validation_metadata(modality_name),
    }

    if not values:
        # measured: RNA_SEQ comes back (0, 371) when no gene is near enough to score
        return {
            **base,
            "value": None,
            "quantile": None,
            "top_tracks": [],
            **({"genes": []} if modality and modality.gene_resolved else {}),
            "note": "AlphaGenome returned no scored rows for this variant",
        }

    if modality is not None and modality.gene_resolved:
        genes = [
            {
                "gene_id": _first(obs_rows[i], "gene_id", "gene") if i < len(obs_rows) else None,
                "gene_name": (
                    _first(obs_rows[i], "gene_name", "symbol") if i < len(obs_rows) else None
                ),
                **_summarise_row(values[i], _row(quantiles, i), columns, quantity, var_rows),
            }
            for i in range(len(values))
        ]
        genes.sort(key=_by_effect, reverse=True)
        genes = genes[:_MAX_GENES]
        top = genes[0] if genes else {}
        return {
            **base,
            "value": top.get("value"),
            "quantile": top.get("quantile"),
            "top_tracks": top.get("top_tracks", []),
            "genes": genes,
        }

    return {**base, **_summarise_row(values[0], _row(quantiles, 0), columns, quantity, var_rows)}


def _columns(
    selection: TrackSelection, var_rows: list[dict[str, Any]], n_tracks: int
) -> tuple[list[int], bool]:
    """The track columns to read, and whether the cell-type request actually matched.

    Resolution happens against the cached scorer metadata, but the columns are taken from
    the response's own var wherever it names biosamples, so a response whose tracks were
    filtered server-side cannot be read through stale positions.
    """
    if selection.biosamples and var_rows:
        wanted = {_slug(name) for name in selection.biosamples}
        by_name = [
            i
            for i, row in enumerate(var_rows[:n_tracks])
            if _slug(str(_first(row, "biosample_name", "gtex_tissue", "biosample") or ""))
            in wanted
        ]
        if by_name:
            return by_name, True
    columns = [i for i in selection.indices if 0 <= i < n_tracks]
    if columns:
        return columns, selection.matched
    return list(range(n_tracks)), False


def _by_effect(entry: dict[str, Any]) -> float:
    value = entry.get("value")
    return abs(value) if value is not None else -1.0


def _row(matrix: list[list[Any]], index: int) -> list[Any] | None:
    return matrix[index] if index < len(matrix) else None


def _summarise_row(
    values: list[Any],
    quantiles: list[Any] | None,
    columns: list[int],
    quantity: str,
    var_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """The reported quantity for one row, plus the tracks that produced it."""
    tracks = [
        {
            "track": _track_field(var_rows, i, "name", "track", "track_name"),
            "biosample": _track_field(var_rows, i, "biosample_name", "gtex_tissue", "biosample"),
            "value": _expose(values[i], quantity),
            "quantile": (
                _expose(quantiles[i], quantity)
                if quantiles is not None and i < len(quantiles)
                else None
            ),
        }
        for i in columns
        if i < len(values) and values[i] is not None
    ]
    if not tracks:
        return {"value": None, "quantile": None, "top_tracks": []}
    # the extreme track by absolute effect, whichever quantity is exposed: averaging over
    # a matched biosample's tracks would wash a real effect out against the tracks in the
    # same tissue that do not carry it
    tracks.sort(key=_by_effect, reverse=True)
    # both fields of the extreme track are already exposed, so the row-level pair -- and the
    # per-gene and modality-level copies taken from it -- inherit the rule
    top = tracks[0]
    return {
        "value": top["value"],
        "quantile": top["quantile"],
        "top_tracks": tracks[:_TOP_TRACKS],
    }


def _track_field(var_rows: list[dict[str, Any]], index: int, *keys: str) -> Any:
    return _first(var_rows[index], *keys) if index < len(var_rows) else None


def _expose(value: Any, quantity: str) -> float | None:
    """A value as the modality's rule allows it to be seen.

    Applied to the quantile as well as the score. AlphaGenome's quantile ranks the SIGNED
    raw score in a background distribution, so exposing it verbatim beside a magnitude value
    would hand back both the direction the rule withholds and, from the pair, the signed
    delta. The splice sign is not commensurable with anything the suite measures, so
    surfacing it would read as a claim about direction that no measurement here supports. AVI is signed even where scorer metadata
    reports is_signed False, so this cannot be driven off that flag.
    """
    if value is None:
        return None
    number = float(value)
    return number if quantity == _SIGNED else abs(number)


def _failed(stage: str, message: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Turn a sentinel into a tool result; never raises past a tool method."""
    return {"success": False, "error": message, "stage": stage, **(context or {})}
