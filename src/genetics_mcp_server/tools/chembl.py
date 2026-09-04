"""Client for the ChEMBL REST API (drug mechanisms, indications, bioactivity).

Transport (URL building, the origin pin, the TTL cache, paging, error sentinels),
resolution and the tool methods all live here, the way tools/uniprot.py keeps its logic
out of executor.py so it can be tested without a ToolExecutor; the executor holds only
delegates. No method raises for a ChEMBL-side problem: a tool method returns a
`success: False` dict naming the stage, everything below it returns an error sentinel.
"""

import logging
import re
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

import httpx

from genetics_mcp_server.tools.uniprot import UniProtClient, _is_error, _TTLCache

if TYPE_CHECKING:
    # type-only for the same reason as in tools/uniprot.py: a real import of
    # config.settings would pull the module enumerating every internal env var name
    # into the SDK's import closure, and so into the sandbox image.
    from genetics_mcp_server.config.settings import Settings

logger = logging.getLogger(__name__)

_TIMEOUT = 20.0
# hard stop on a walk: a filter that matches half of ChEMBL (activity by target runs to
# thousands of rows) must bound the number of round trips, not just the rows returned
_MAX_PAGES = 10

# module level for the reason given above uniprot.py's `_CACHE`: clients are built per
# ToolExecutor and several are constructed independently, so a per-instance cache would
# not be shared.
_CACHE = _TTLCache()

_CHEMBL_ID_RE = re.compile(r"^CHEMBL[0-9]+$")

# a heteromer, a protein family and a cell line are all "targets" for the same accession,
# and only this one is the protein itself; the rest are reported rather than dropped
_SINGLE_PROTEIN = "SINGLE PROTEIN"
_TARGET_FIELDS = ["target_chembl_id", "pref_name", "target_type"]
_TARGET_DETAIL_FIELDS = [*_TARGET_FIELDS, "organism", "target_components"]
_MECHANISM_FIELDS = ["molecule_chembl_id", "mechanism_of_action", "action_type", "max_phase"]
_PROFILE_MECHANISM_FIELDS = [
    "target_chembl_id",
    "mechanism_of_action",
    "action_type",
    "max_phase",
]
_MOLECULE_FIELDS = [
    "molecule_chembl_id",
    "pref_name",
    "max_phase",
    "first_approval",
    "withdrawn_flag",
    "atc_classifications",
    "molecule_type",
]
# only the name ladder needs the hierarchy, and it is a nested object: projecting it on
# the mechanism -> molecule join would widen every row of that batch for nothing
_CANDIDATE_FIELDS = [*_MOLECULE_FIELDS, "molecule_hierarchy"]
_TOP_FIELDS = ["molecule_chembl_id", "pref_name", "max_phase"]
_INDICATION_FIELDS = ["efo_id", "efo_term", "mesh_heading", "max_phase_for_ind"]
_INDICATION_BATCH_FIELDS = ["molecule_chembl_id", *_INDICATION_FIELDS]
_ACTIVITY_FIELDS = [
    "molecule_chembl_id",
    "standard_type",
    "standard_relation",
    "pchembl_value",
    "assay_chembl_id",
]

# ChEMBL pages at 20 by default and allows up to 1000; asking for more per page is the
# cheapest way to keep a join inside the `_MAX_PAGES` walk
_PAGE_LIMIT = 100
_ACTIVITY_PAGE_LIMIT = 1000
# 5 x _ACTIVITY_PAGE_LIMIT rows is the ceiling on one target's activity walk, ordered by
# descending pchembl_value. top_compounds is exact unless the walk is capped: a heavily
# assayed target can spend every page on its own top molecules; `truncated` reports that.
_ACTIVITY_MAX_PAGES = 5
# the id list travels in the URL, so a `__in` batch is split
_ID_CHUNK = 50
# per-drug indication lists in a many-drug table are context, not the answer; the whole
# list is one get_drug_profile call away
_INDICATIONS_PER_DRUG = 10
_PROFILE_INDICATION_CAP = 50


def _phase(value: Any) -> float | None:
    """ChEMBL max_phase: 0-4, float-ish, with -1 or None meaning unknown.

    Unknown stays None. Coercing it to 0 would read as "preclinical", which is a claim
    about the drug rather than about the annotation. 4 means approved *somewhere*, which
    is not the same as approved by any particular regulator.
    """
    if value is None:
        return None
    phase = _number(value)
    if phase is None or phase < 0:
        return None
    return phase


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sort_phase(value: Any) -> float:
    """Phase for ordering only: an unknown one sinks below phase 0."""
    phase = _phase(value)
    return phase if phase is not None else -1.0


def _passes_phase(phase: float | None, min_phase: float) -> bool:
    """An unknown phase is kept only when nothing was asked of it."""
    if phase is None:
        return min_phase == 0
    return phase >= min_phase


def _unique_ids(records: list[Any], field: str) -> list[str]:
    """The distinct non-empty values of `field`, in first-seen order."""
    seen: dict[str, None] = {}
    for record in records:
        value = record.get(field)
        if value:
            seen.setdefault(str(value), None)
    return list(seen)


def _index_by(records: list[Any], field: str) -> dict[str, Any]:
    return {r[field]: r for r in records if isinstance(r, dict) and r.get(field)}


class ChEMBLClient:
    """Async client for the ChEMBL REST API."""

    def __init__(
        self,
        external_client: httpx.AsyncClient,
        settings: "Settings",
        uniprot: UniProtClient,
    ):
        # the httpx client is injected, not constructed here: the executor passes its
        # _ResilientAsyncClient, so connection failures arrive as a synthetic 503
        # instead of raising, and no internal auth header reaches EBI
        self._client = external_client
        # the executor's own resolver, so a symbol resolved for a UniProt tool and the
        # same symbol resolved here cost one lookup between them and cannot disagree
        self._uniprot = uniprot
        self._base_url = settings.chembl_api_url.rstrip("/")
        self._cache_ttl = settings.chembl_cache_ttl
        # a request may only be made to the origin this deployment is configured for.
        # Derived from the base URL rather than hardcoded so a self-hosted mirror keeps
        # working; because the scheme is part of the origin this also refuses an
        # https -> http downgrade. The pin matters most for the paging walk, where the
        # next URL is server-supplied.
        self._allowed_origins = frozenset({self._origin(httpx.URL(self._base_url))})

    async def _get(
        self, resource: str, params: dict[str, Any], only: list[str]
    ) -> dict[str, Any]:
        """GET one page of `<base>/<resource>.json`, projected to `only`.

        Returns the parsed body or an error sentinel; use `_is_error` to tell them
        apart. `only` is mandatory because ChEMBL's unprojected records are far too
        large to reach the model or the cache: three target records are 55 KB, 200
        indications 76 KB, and a single molecule 6-8 KB.
        """
        return await self._get_url(self._build_url(resource, params, only))

    async def _get_all(
        self,
        resource: str,
        params: dict[str, Any],
        only: list[str],
        max_pages: int = _MAX_PAGES,
    ) -> dict[str, Any]:
        """Follow `page_meta.next` up to `max_pages` and concatenate the record lists.

        Returns `{"records", "total_count", "truncated"}` — `total_count` is ChEMBL's
        own count for the whole filter, so a caller can say how much was left behind —
        or the sentinel of the first page that failed. The records are the cached page
        objects themselves, so callers must not mutate what comes back.
        """
        records: list[Any] = []
        total_count: int | None = None
        body = await self._get(resource, params, only)
        pages = 0
        while True:
            if _is_error(body):
                return body
            page = self._records(body)
            if page is None:
                return self._shape_sentinel(resource, body)
            records.extend(page)
            pages += 1
            page_meta = body.get("page_meta") or {}
            if total_count is None:
                count = page_meta.get("total_count")
                total_count = count if isinstance(count, int) else None
            next_path = page_meta.get("next")
            if not next_path:
                return {"records": records, "total_count": total_count, "truncated": False}
            if pages >= max_pages:
                return {"records": records, "total_count": total_count, "truncated": True}
            # `next` is a path relative to the host ("/chembl/api/data/target.json?…"),
            # so it is joined and re-pinned like any other server-supplied URL
            if not isinstance(next_path, str):
                return self._unusable_next(next_path)
            try:
                next_url = httpx.URL(self._base_url).join(next_path)
            except (httpx.InvalidURL, ValueError):
                # a `next` httpx cannot parse — a non-numeric port, a host IDNA cannot
                # encode — must leave the walk as a sentinel like any other bad page,
                # not as an exception out of a tool method
                return self._unusable_next(next_path)
            body = await self._get_url(next_url)

    async def release(self) -> str | None:
        """The live release name, e.g. "ChEMBL_37", or None if status is unreadable.

        `status.json` is a single small record and takes no `only`, so it is the one
        request that does not go through `_get`.
        """
        body = await self._get_url(httpx.URL(f"{self._base_url}/status.json"))
        if _is_error(body):
            return None
        version = body.get("chembl_db_version")
        return str(version) if version else None

    async def resolve_target(self, query: str) -> dict[str, Any]:
        """Work out which ChEMBL target an agent-supplied gene, accession or id means.

        A `CHEMBL<number>` target id is used as given; everything else goes through the
        shared UniProt resolver, which reports whether it read the input as an accession
        or a symbol. Shape alone cannot make that call here — real HGNC symbols (P2RY12,
        B4GAT1, H2AC11) are valid accession syntax — and the resolver also maps a merged
        or secondary accession to the primary one. Using it rather than ChEMBL's own
        symbol synonyms, a second and differently curated naming authority, keeps one
        answer to "which protein is this".

        Returns `{resolution: {query, kind, accession, uniprot, organism, n_targets, note},
        target_chembl_id, pref_name, target_type, other_targets}`. A gene with no ChEMBL
        target is a normal answer, not an error: `target_chembl_id` is None and the note
        says so. Only a failed lookup is a sentinel (test it with `_is_error`); the
        sentinel carries `_stage` because "UniProt could not resolve the symbol" and
        "ChEMBL refused the target query" are different problems for the caller.
        """
        text = (query or "").strip()
        if not text:
            return {
                "_error": "ChEMBL: empty query, nothing to resolve",
                "_status": None,
                "_stage": "input",
            }
        candidate = text.upper()
        if _CHEMBL_ID_RE.match(candidate):
            return await self._targets(
                {"target_chembl_id": candidate}, text, "chembl_target", None, None
            )
        resolution = await self._uniprot.resolve(text)
        if _is_error(resolution) or not resolution.get("accession"):
            return self._uniprot_stage_sentinel(text, resolution)
        return await self._targets_for_accession(
            str(resolution["accession"]),
            text,
            str(resolution.get("input_kind") or "symbol"),
            resolution,
        )

    async def _targets_for_accession(
        self, accession: str, query: str, kind: str, uniprot: dict[str, Any] | None
    ) -> dict[str, Any]:
        return await self._targets(
            # organism is filtered here rather than after the fact because the same
            # accession's orthologues carry their own targets
            {"target_components__accession": accession, "organism": "Homo sapiens"},
            query,
            kind,
            accession,
            uniprot,
        )

    async def _targets(
        self,
        params: dict[str, Any],
        query: str,
        kind: str,
        accession: str | None,
        uniprot: dict[str, Any] | None,
    ) -> dict[str, Any]:
        body = await self._get_all("target", params, _TARGET_FIELDS)
        if _is_error(body):
            return {**body, "_stage": "chembl_target"}
        organism = params.get("organism")
        records = body["records"]
        singles = [r for r in records if r.get("target_type") == _SINGLE_PROTEIN]
        chosen = singles[0] if singles else (records[0] if records else None)
        others = [r for r in records if r is not chosen]
        if chosen is None:
            # the organism filter is silent, so a bare "no target" would state a false
            # reason for an accession whose targets are all non-human
            note = f"no ChEMBL target is annotated for {accession or query}"
            if organism:
                note += f" in {organism}"
        elif singles:
            note = f"chose the {_SINGLE_PROTEIN} target"
            if others:
                note += f"; {len(others)} other target(s) share this accession"
        else:
            note = (
                f"no {_SINGLE_PROTEIN} target for {accession or query}; "
                f"returning the {chosen.get('target_type')} target instead"
            )
        if body.get("truncated"):
            total = body.get("total_count")
            matched = f"{total}" if isinstance(total, int) else f"more than {len(records)}"
            note += f"; {matched} targets matched and the walk was capped"
        return {
            "resolution": {
                "query": query,
                "kind": kind,
                "accession": accession,
                "uniprot": uniprot,
                "organism": organism,
                "n_targets": len(records),
                "note": note,
            },
            # rebuilt rather than passed through: the records are the cached page objects
            "target_chembl_id": chosen.get("target_chembl_id") if chosen else None,
            "pref_name": chosen.get("pref_name") if chosen else None,
            "target_type": chosen.get("target_type") if chosen else None,
            "other_targets": [{k: r.get(k) for k in _TARGET_FIELDS} for r in others],
        }

    @staticmethod
    def _uniprot_stage_sentinel(query: str, resolution: dict[str, Any]) -> dict[str, Any]:
        """Sentinel for a symbol UniProt could not turn into an accession."""
        reason = resolution.get("_error") or "UniProt returned no accession"
        sentinel = {
            "_error": f"ChEMBL: could not resolve {query!r} to a UniProt accession: {reason}",
            "_status": resolution.get("_status"),
            "_stage": "uniprot",
        }
        if resolution.get("_no_match"):
            sentinel["_no_match"] = True
        return sentinel

    # -------------------------------------------------------------------------
    # tool methods
    # -------------------------------------------------------------------------

    async def attribution(self) -> str:
        """The credit ChEMBL's CC BY-SA licence requires on every result.

        Unversioned when status.json cannot be read: an attribution lookup must never
        be what fails a tool. The release goes through the same TTL cache as every
        other request, so this costs one call per cache window, not one per tool call.
        """
        release = await self.release()
        if not release:
            return "ChEMBL (CC BY-SA 3.0), EMBL-EBI"
        return f"ChEMBL {release} (CC BY-SA 3.0), EMBL-EBI"

    async def get_drug_targets_for_gene(
        self,
        query: str,
        min_phase: float = 0,
        include_indications: bool = False,
        max_results: int = 25,
    ) -> dict[str, Any]:
        """Drugs and clinical candidates annotated as acting on a gene's ChEMBL target.

        One row per (drug, mechanism): a drug with two annotated mechanisms on the same
        target is two rows. `max_phase` is the molecule's — the mechanism row carries a
        per-mechanism phase that can be lower, carried as `mechanism_max_phase` and None
        where the two agree.
        """
        try:
            # clamped like pchembl_min in get_target_bioactivity: phases run 0-4, and a
            # negative floor would otherwise read as "asked for something" and drop the
            # unknown-phase rows that a floor of 0 is meant to keep
            floor = min(4.0, max(0.0, float(min_phase)))
        except (TypeError, ValueError):
            return self._bad_param("min_phase", min_phase, query)
        try:
            # the schema bound in definitions.py must match this expression — widen one
            # without the other and rows are silently dropped below
            result_cap = max(1, min(int(max_results), 100))
        except (TypeError, ValueError):
            return self._bad_param("max_results", max_results, query)

        resolved = await self.resolve_target(query)
        if _is_error(resolved):
            return self._failed(resolved, "resolve_target", {"query": query})
        resolution = resolved["resolution"]
        target_id = resolved["target_chembl_id"]
        base = {
            "success": True,
            "query": query,
            "target_chembl_id": target_id,
            "target_pref_name": resolved["pref_name"],
            "target_type": resolved["target_type"],
            "other_targets": resolved["other_targets"],
            "resolution": resolution,
            "attribution": await self.attribution(),
        }
        if target_id is None:
            return {
                **base,
                "drugs": [],
                "count": 0,
                "n_matching": 0,
                "n_mechanisms": 0,
                "truncated": False,
                "note": resolution["note"],
            }

        mechanisms = await self._get_all(
            "mechanism",
            {"target_chembl_id": target_id, "limit": _PAGE_LIMIT},
            _MECHANISM_FIELDS,
        )
        if _is_error(mechanisms):
            return self._failed(mechanisms, "mechanism", resolution)
        mechanism_records = mechanisms["records"]
        molecule_ids = _unique_ids(mechanism_records, "molecule_chembl_id")
        molecules = await self._by_ids(
            "molecule", "molecule_chembl_id", molecule_ids, _MOLECULE_FIELDS
        )
        if _is_error(molecules):
            return self._failed(molecules, "molecule", resolution)
        by_molecule = _index_by(molecules["records"], "molecule_chembl_id")

        drugs = []
        for record in mechanism_records:
            molecule_id = record.get("molecule_chembl_id")
            molecule = by_molecule.get(molecule_id) or {}
            row = {
                **self._molecule_row(molecule, molecule_id),
                "action_type": record.get("action_type"),
                "mechanism_of_action": record.get("mechanism_of_action"),
            }
            mechanism_phase = _phase(record.get("max_phase"))
            # always present, None when it agrees with the molecule's, so the downloaded
            # table's columns do not depend on which rows came back
            row["mechanism_max_phase"] = (
                mechanism_phase if mechanism_phase != row["max_phase"] else None
            )
            drugs.append(row)

        drugs = [d for d in drugs if _passes_phase(d["max_phase"], floor)]
        drugs.sort(key=lambda d: (-_sort_phase(d["max_phase"]), d["pref_name"] or ""))
        n_matching = len(drugs)
        drugs = drugs[:result_cap]

        indications_truncated = False
        if include_indications:
            indications = await self._by_ids(
                "drug_indication",
                "molecule_chembl_id",
                _unique_ids(drugs, "molecule_chembl_id"),
                _INDICATION_BATCH_FIELDS,
                limit=_ACTIVITY_PAGE_LIMIT,
            )
            if _is_error(indications):
                return self._failed(indications, "drug_indication", resolution)
            indications_truncated = bool(indications["truncated"])
            grouped = self._group_indications(indications["records"])
            for row in drugs:
                for_drug = grouped.get(row["molecule_chembl_id"], [])
                row["indications"] = for_drug[:_INDICATIONS_PER_DRUG]
                row["n_indications"] = len(for_drug)

        return {
            **base,
            "drugs": drugs,
            "count": len(drugs),
            "n_matching": n_matching,
            "n_mechanisms": len(mechanism_records),
            "truncated": bool(
                mechanisms["truncated"]
                or molecules["truncated"]
                or indications_truncated
                or n_matching > len(drugs)
            ),
            "note": resolution["note"],
        }

    async def get_drug_profile(self, query: str) -> dict[str, Any]:
        """What ChEMBL holds about one drug: phase, ATC, mechanisms, indications."""
        resolved = await self._resolve_molecule(query)
        if _is_error(resolved):
            return self._failed(resolved, "molecule", {"query": query})
        resolution = resolved["resolution"]
        molecule = resolved["molecule"]
        base = {
            "success": True,
            "query": query,
            "resolution": resolution,
            "attribution": await self.attribution(),
        }
        if molecule is None:
            return {
                **base,
                "drug": None,
                "mechanisms": [],
                "indications": [],
                "n_indications": 0,
                "note": resolution["note"],
            }

        molecule_id = molecule["molecule_chembl_id"]
        mechanisms = await self._get_all(
            "mechanism",
            {"molecule_chembl_id": molecule_id, "limit": _PAGE_LIMIT},
            _PROFILE_MECHANISM_FIELDS,
        )
        if _is_error(mechanisms):
            return self._failed(mechanisms, "mechanism", resolution)
        targets = await self._by_ids(
            "target",
            "target_chembl_id",
            _unique_ids(mechanisms["records"], "target_chembl_id"),
            _TARGET_DETAIL_FIELDS,
        )
        if _is_error(targets):
            return self._failed(targets, "target", resolution)
        by_target = _index_by(targets["records"], "target_chembl_id")
        mechanism_rows = [
            {
                "target_chembl_id": record.get("target_chembl_id"),
                "target_pref_name": (by_target.get(record.get("target_chembl_id")) or {}).get(
                    "pref_name"
                ),
                "target_type": (by_target.get(record.get("target_chembl_id")) or {}).get(
                    "target_type"
                ),
                "organism": (by_target.get(record.get("target_chembl_id")) or {}).get("organism"),
                "components": self._target_components(
                    by_target.get(record.get("target_chembl_id")) or {}
                ),
                "action_type": record.get("action_type"),
                "mechanism_of_action": record.get("mechanism_of_action"),
                "max_phase": _phase(record.get("max_phase")),
            }
            for record in mechanisms["records"]
        ]

        indications = await self._get_all(
            "drug_indication",
            {"molecule_chembl_id": molecule_id, "limit": _PAGE_LIMIT},
            _INDICATION_FIELDS,
        )
        if _is_error(indications):
            return self._failed(indications, "drug_indication", resolution)
        indication_rows = [
            {k: record.get(k) for k in _INDICATION_FIELDS} for record in indications["records"]
        ]
        indication_rows.sort(key=lambda i: -_sort_phase(i.get("max_phase_for_ind")))

        return {
            **base,
            "drug": molecule,
            "mechanisms": mechanism_rows,
            "n_mechanisms": len(mechanism_rows),
            "indications": indication_rows[:_PROFILE_INDICATION_CAP],
            "n_indications": len(indication_rows),
            "truncated": bool(
                mechanisms["truncated"] or targets["truncated"] or indications["truncated"]
            ),
            # popped into the download hint like _all_compounds: the full list is a TSV,
            # while the result carries the best-evidenced _PROFILE_INDICATION_CAP
            "_all_indications": indication_rows,
        }

    async def get_target_bioactivity(
        self, query: str, pchembl_min: float = 6.0, max_results: int = 25
    ) -> dict[str, Any]:
        """Potency summary for a target: how many compounds bind it, and the best ones.

        The activity table is the largest resource in ChEMBL, so this returns counts and
        one row per molecule rather than per assay measurement.
        """
        try:
            # pChEMBL is -log10(molar), so anything outside 0-14 is a typo rather than a filter
            threshold = min(14.0, max(0.0, float(pchembl_min)))
        except (TypeError, ValueError):
            return self._bad_param("pchembl_min", pchembl_min, query)
        try:
            # the schema bound in definitions.py must match this expression — widen one
            # without the other and rows are silently dropped below
            result_cap = max(1, min(int(max_results), 100))
        except (TypeError, ValueError):
            return self._bad_param("max_results", max_results, query)

        resolved = await self.resolve_target(query)
        if _is_error(resolved):
            return self._failed(resolved, "resolve_target", {"query": query})
        resolution = resolved["resolution"]
        target_id = resolved["target_chembl_id"]
        base = {
            "success": True,
            "query": query,
            "target_chembl_id": target_id,
            "target_pref_name": resolved["pref_name"],
            "target_type": resolved["target_type"],
            "other_targets": resolved["other_targets"],
            "pchembl_min": threshold,
            "resolution": resolution,
            "attribution": await self.attribution(),
        }
        empty = {
            "n_activities": 0,
            "total_count": 0,
            "truncated": False,
            "n_distinct_molecules": 0,
            "by_standard_type": {},
            "top_compounds": [],
        }
        if target_id is None:
            return {**base, **empty, "note": resolution["note"]}

        activities = await self._get_all(
            "activity",
            {
                "target_chembl_id": target_id,
                "pchembl_value__gte": threshold,
                "limit": _ACTIVITY_PAGE_LIMIT,
                "order_by": "-pchembl_value",
            },
            _ACTIVITY_FIELDS,
            max_pages=_ACTIVITY_MAX_PAGES,
        )
        if _is_error(activities):
            return self._failed(activities, "activity", resolution)
        records = activities["records"]
        if not records:
            return {
                **base,
                **empty,
                "total_count": activities["total_count"],
                "note": f"no activity at or above pChEMBL {threshold} is recorded for {target_id}",
            }

        by_standard_type: dict[str, int] = {}
        summary: dict[str, dict[str, Any]] = {}
        for record in records:
            standard_type = record.get("standard_type")
            if standard_type:
                by_standard_type[standard_type] = by_standard_type.get(standard_type, 0) + 1
            molecule_id = record.get("molecule_chembl_id")
            if not molecule_id:
                continue
            entry = summary.setdefault(
                molecule_id,
                {
                    "molecule_chembl_id": molecule_id,
                    "best_pchembl": None,
                    "standard_type": None,
                    "n_activities": 0,
                },
            )
            entry["n_activities"] += 1
            value = _number(record.get("pchembl_value"))
            if value is not None and (
                entry["best_pchembl"] is None or value > entry["best_pchembl"]
            ):
                entry["best_pchembl"] = value
                entry["standard_type"] = standard_type

        ordered = sorted(
            summary.values(),
            key=lambda r: (
                -(r["best_pchembl"] if r["best_pchembl"] is not None else -1.0),
                r["molecule_chembl_id"],
            ),
        )
        # copied out of `ordered` so naming these rows does not add a pref_name column to
        # every row of the download, where no name is ever fetched
        top = [dict(row) for row in ordered[:result_cap]]
        molecules = await self._by_ids(
            "molecule", "molecule_chembl_id", [r["molecule_chembl_id"] for r in top], _TOP_FIELDS
        )
        if _is_error(molecules):
            return self._failed(molecules, "molecule", resolution)
        by_molecule = _index_by(molecules["records"], "molecule_chembl_id")
        for row in top:
            molecule = by_molecule.get(row["molecule_chembl_id"]) or {}
            row["pref_name"] = molecule.get("pref_name")
            row["max_phase"] = _phase(molecule.get("max_phase"))

        return {
            **base,
            "n_activities": len(records),
            "total_count": activities["total_count"],
            "truncated": bool(activities["truncated"] or molecules["truncated"]),
            "n_distinct_molecules": len(summary),
            "by_standard_type": dict(
                sorted(by_standard_type.items(), key=lambda kv: (-kv[1], kv[0]))
            ),
            "top_compounds": top,
            # the executor pops this into the download hint; the whole per-molecule
            # summary is a TSV, not something to put in front of the model
            "_all_compounds": ordered,
        }

    async def _resolve_molecule(self, query: str) -> dict[str, Any]:
        """Work out which ChEMBL molecule a drug name or id means.

        A `CHEMBL<number>` is used as given; a name is tried as the preferred name
        before the synonym list, because pref_name is one per molecule while a synonym
        ("Advil") is shared by every formulation carrying it. Several matches are not an
        error: `_candidate_rank` picks one and the rest are listed.
        """
        text = (query or "").strip()
        if not text:
            return {
                "_error": "ChEMBL: empty query, nothing to resolve",
                "_status": None,
                "_stage": "input",
            }
        if _CHEMBL_ID_RE.match(text.upper()):
            ladder = [("chembl_id", {"molecule_chembl_id": text.upper()})]
        else:
            ladder = [
                ("pref_name", {"pref_name__iexact": text}),
                ("synonym", {"molecule_synonyms__molecule_synonym__iexact": text}),
            ]
        for kind, params in ladder:
            body = await self._get_all(
                "molecule", {**params, "limit": _PAGE_LIMIT}, _CANDIDATE_FIELDS
            )
            if _is_error(body):
                return {**body, "_stage": "molecule"}
            records = body["records"]
            if not records:
                continue
            chosen = min(records, key=self._candidate_rank)
            others = [
                {
                    "molecule_chembl_id": r.get("molecule_chembl_id"),
                    "pref_name": r.get("pref_name"),
                }
                for r in records
                if r is not chosen
            ]
            note = f"matched {query!r} on {kind}"
            if others:
                note += (
                    f"; {len(others)} other candidate(s) matched, kept the highest max_phase, "
                    "preferring a parent molecule over its salts and then the lowest id"
                )
            return {
                "resolution": {
                    "query": query,
                    "kind": kind,
                    "n_candidates": len(records),
                    "other_candidates": others,
                    "note": note,
                },
                "molecule": self._molecule_row(chosen, chosen.get("molecule_chembl_id")),
            }
        return {
            "resolution": {
                "query": query,
                "kind": None,
                "n_candidates": 0,
                "other_candidates": [],
                "note": f"no ChEMBL molecule matches {query!r} by preferred name or synonym",
            },
            "molecule": None,
        }

    @staticmethod
    def _candidate_rank(record: dict[str, Any]) -> tuple[float, int, str]:
        """Sort key for the name ladder's candidates; the smallest wins.

        Phase alone leaves a salt and its parent tied, and a tie was previously settled
        by whichever ChEMBL listed first. The parent is the molecule the annotations hang
        off, so it wins the tie; the id settles the rest so the answer is stable.
        """
        molecule_id = str(record.get("molecule_chembl_id") or "")
        hierarchy = record.get("molecule_hierarchy")
        is_parent = (
            isinstance(hierarchy, dict) and hierarchy.get("parent_chembl_id") == molecule_id
        )
        return (-_sort_phase(record.get("max_phase")), 0 if is_parent else 1, molecule_id)

    async def _by_ids(
        self,
        resource: str,
        field: str,
        ids: list[str],
        only: list[str],
        limit: int = _PAGE_LIMIT,
        max_pages: int = _MAX_PAGES,
    ) -> dict[str, Any]:
        """Fetch `resource` rows for many ids through `<field>__in`, in chunks.

        The alternative — one request per drug — is what the mechanism -> molecule join
        costs otherwise. Chunked because the id list travels in the URL.
        Returns `{"records", "truncated"}`, or the sentinel of the chunk that failed.
        """
        records: list[Any] = []
        truncated = False
        for start in range(0, len(ids), _ID_CHUNK):
            body = await self._get_all(
                resource,
                {f"{field}__in": ",".join(ids[start : start + _ID_CHUNK]), "limit": limit},
                only,
                max_pages,
            )
            if _is_error(body):
                return body
            records.extend(body["records"])
            truncated = truncated or bool(body["truncated"])
        return {"records": records, "truncated": truncated}

    @staticmethod
    def _molecule_row(molecule: dict[str, Any], molecule_id: str | None) -> dict[str, Any]:
        """The drug fields of a result row, rebuilt out of a cached molecule record."""
        return {
            "pref_name": molecule.get("pref_name"),
            "molecule_chembl_id": molecule.get("molecule_chembl_id") or molecule_id,
            "molecule_type": molecule.get("molecule_type"),
            "max_phase": _phase(molecule.get("max_phase")),
            "first_approval": molecule.get("first_approval"),
            "withdrawn_flag": molecule.get("withdrawn_flag"),
            "atc_codes": list(molecule.get("atc_classifications") or []),
        }

    @staticmethod
    def _group_indications(records: list[Any]) -> dict[str, list[dict[str, Any]]]:
        """drug_indication rows per molecule, best-evidenced phase first."""
        grouped: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            molecule_id = record.get("molecule_chembl_id")
            if not molecule_id:
                continue
            grouped.setdefault(molecule_id, []).append(
                {k: record.get(k) for k in _INDICATION_FIELDS}
            )
        for rows in grouped.values():
            rows.sort(key=lambda i: -_sort_phase(i.get("max_phase_for_ind")))
        return grouped

    @staticmethod
    def _target_components(target: dict[str, Any]) -> list[dict[str, Any]]:
        """Accessions and gene symbols out of a target's components.

        `only=target_components` projects the column, not the objects inside it, so the
        record arrives with each component's full xref list — hundreds of PDBe entries
        for a well-studied target. Everything but the accession and the GENE_SYMBOL
        synonyms is dropped here rather than carried into a result.
        """
        components = []
        for component in target.get("target_components") or []:
            if not isinstance(component, dict):
                continue
            symbols = [
                synonym.get("component_synonym")
                for synonym in component.get("target_component_synonyms") or []
                if isinstance(synonym, dict)
                and synonym.get("syn_type") == "GENE_SYMBOL"
                and synonym.get("component_synonym")
            ]
            components.append(
                {"accession": component.get("accession"), "gene_symbols": symbols}
            )
        return components

    @staticmethod
    def _failed(
        sentinel: dict[str, Any], stage: str, resolution: dict[str, Any]
    ) -> dict[str, Any]:
        """Turn a stage's sentinel into a tool result; never raises past a tool method."""
        return {
            "success": False,
            "error": sentinel.get("_error"),
            "stage": sentinel.get("_stage") or stage,
            "resolution": resolution,
        }

    @staticmethod
    def _bad_param(name: str, value: Any, query: str) -> dict[str, Any]:
        """Result for an argument that failed to coerce, caught before any request is made."""
        return {
            "success": False,
            "error": f"ChEMBL: {name}={value!r} is not a number",
            "stage": "input",
            "resolution": {"query": query},
        }

    async def _get_url(self, url: httpx.URL) -> dict[str, Any]:
        """The single request funnel: origin pin, TTL cache, 20s timeout, sentinel.

        A cached body is handed to every hit, so callers must not mutate what comes
        back. Error sentinels are never cached.
        """
        if self._origin(url) not in self._allowed_origins:
            return self._refused_origin(url)
        cache_key = str(url)
        cached = _CACHE.get(cache_key)
        if cached is not _TTLCache._MISS:
            return cached
        resp = await self._client.get(
            url,
            headers={"Accept": "application/json"},
            timeout=_TIMEOUT,
            # stated rather than inherited from the client's default: the origin pin above
            # only holds if the hop it approves is the only one made
            follow_redirects=False,
        )
        if resp.status_code != 200:
            return self._error_sentinel(resp, url)
        try:
            body = resp.json()
        except ValueError as exc:
            logger.warning(f"ChEMBL returned an unparseable body from {url}: {exc}")
            return {"_error": f"ChEMBL returned an unparseable response: {exc}", "_status": 200}
        if not isinstance(body, dict):
            return self._shape_sentinel(str(url), body)
        _CACHE.set(cache_key, body, self._cache_ttl)
        return body

    def _build_url(self, resource: str, params: dict[str, Any], only: list[str]) -> httpx.URL:
        if not only:
            raise ValueError("ChEMBL requests must project with `only`")
        # quote() percent-encodes ?, = and & so a resource name cannot smuggle in a
        # query, and leaves no way out of the configured base path
        url = httpx.URL(f"{self._base_url}/{quote(resource.strip('/'), safe='')}.json")
        return url.copy_merge_params({**params, "only": ",".join(only)})

    @staticmethod
    def _records(body: dict[str, Any]) -> list[Any] | None:
        """The record list of a ChEMBL page, or None if the body is not one.

        The key is the resource name pluralised (targets, mechanisms, molecules,
        activities, drug_indications, assays), which is found by elimination rather
        than by a table of plurals so a resource nobody anticipated still reads.
        """
        lists = [
            value
            for key, value in body.items()
            if key != "page_meta" and isinstance(value, list)
        ]
        return lists[0] if len(lists) == 1 else None

    @staticmethod
    def _origin(url: httpx.URL) -> tuple[str, str, int | None]:
        return url.scheme, url.host, url.port

    @staticmethod
    def _refused_origin(url: httpx.URL) -> dict[str, Any]:
        """Sentinel for a URL outside the configured origin; no request is made."""
        logger.warning(f"ChEMBL: refused a request to an origin outside the configured API: {url}")
        return {
            "_error": "ChEMBL refused a request to an origin outside the configured API",
            "_status": None,
            "_origin_refused": True,
        }

    @staticmethod
    def _unusable_next(next_path: Any) -> dict[str, Any]:
        """Sentinel for a `page_meta.next` that cannot become a URL; no request is made."""
        logger.warning(f"ChEMBL: refused an unusable page_meta.next: {next_path!r}")
        return {
            "_error": "ChEMBL returned an unusable next-page link",
            "_status": None,
            "_unusable_next": True,
        }

    @staticmethod
    def _shape_sentinel(what: str, body: Any) -> dict[str, Any]:
        logger.warning(f"ChEMBL returned an unexpected shape for {what}: {type(body).__name__}")
        return {
            "_error": f"ChEMBL returned an unexpected response shape for {what}",
            "_status": 200,
            "_unexpected_shape": True,
        }

    @staticmethod
    def _error_sentinel(resp: httpx.Response, url: httpx.URL) -> dict[str, Any]:
        # 404 (no such record) and 400 (malformed filter or identifier) are the
        # documented answers to checking an agent-supplied identifier, i.e. input noise
        # rather than an incident
        log = logger.debug if resp.status_code in (400, 404) else logger.warning
        log(f"ChEMBL request failed: {url} -> {resp.status_code}")
        return {
            "_error": f"ChEMBL HTTP {resp.status_code}: {(resp.text or '')[:200]}",
            "_status": resp.status_code,
        }
