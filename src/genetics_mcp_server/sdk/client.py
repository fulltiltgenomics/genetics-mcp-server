"""Async client behind the `genetics` SDK.

One method per data product. The by-gene / by-variant / by-region / by-phenotype
distinction that produced 42 near-duplicate MCP tools is an *argument* here, not a name:
`credible_sets(gene=...)`, `credible_sets(variant=...)`, `credible_sets(region=...)` all
go through one function that dispatches on which argument was supplied.

Every method is a thin adapter over an existing `ToolExecutor` method — the HTTP, tabix
and BigQuery logic is unchanged and still serves the MCP tool surface. What the adapter
adds is: exactly-one-of dispatch, `{"success": False}` turned into a raised
`GeneticsError`, and a polars DataFrame instead of a nested dict.
"""

import asyncio
import functools
import inspect
import logging
import os
import re
import sys
from typing import Any

import polars as pl

from genetics_mcp_server.sdk.errors import GeneticsError, GeneticsUsageError
from genetics_mcp_server.tools.executor import ToolExecutor

# db-api's per-query row ceiling, mirrored from the tool layer. `limit` on the BigQuery
# functions defaults to it rather than to the tool surface's 500: a script consumes rows
# programmatically, and a positional top-N prefix of an ORDER BY is a wrong answer rather
# than a short one.
MAX_ROWS = ToolExecutor._MAX_SQL_LIMIT

_REGION_RE = re.compile(r"^(?:chr)?([0-9]+|[XYxy]|MT|mt)[:_-](\d+)[-_](\d+)$")


def _frame(
    rows: Any, columns: list[str] | None = None, empty_columns: list[str] | None = None
) -> pl.DataFrame:
    """Build a DataFrame from either row dicts or positional rows plus column names.

    Both shapes occur upstream: the results-api returns JSON objects, while db-api returns
    a list per row with the names in a separate `columns` key. Handing positional rows to
    `pl.from_dicts` does not raise — it silently produces a transposed frame with
    fabricated `column_N` names and stringified values — so `columns`, when the payload
    exposes it, decides which constructor is used rather than being a hint.

    infer_schema_length=None scans every row, so a column that is null in the first rows
    and populated later still gets a usable dtype; strict=False is the fallback for mixed
    types within a column, which the upstream JSON does produce.

    `empty_columns` is the results-api counterpart and is deliberately NOT merged into
    `columns`. Those rows are named dicts that already carry their own schema, so the list
    is needed only when there are none; routing a non-empty results-api result through the
    positional constructor above would give up `from_dicts`' strict=False fallback for the
    mixed-type columns that upstream does produce.
    """
    if not rows and not columns and empty_columns:
        return pl.DataFrame({c: [] for c in empty_columns})
    if columns:
        # an empty result still has a schema: a script filtering on a column must get an
        # empty frame, not ColumnNotFoundError, when the gene simply has no hits
        if not rows:
            return pl.DataFrame({c: [] for c in columns})
        if isinstance(rows, dict):
            rows = [rows]
        if isinstance(rows[0], dict):
            # `columns` still decides order and schema; the dicts are re-flattened against
            # it rather than trusted to iterate in the query's column order
            rows = [[row.get(c) for c in columns] for row in rows]
        return pl.DataFrame(rows, schema=columns, orient="row", infer_schema_length=None)
    if not rows:
        return pl.DataFrame()
    if isinstance(rows, dict):
        rows = [rows]
    try:
        return pl.from_dicts(rows, infer_schema_length=None)
    except Exception:
        return pl.from_dicts(rows, infer_schema_length=None, strict=False)


def _one_of(**candidates: Any) -> tuple[str, Any]:
    """Return the single supplied argument, or raise. This is the grid collapse."""
    given = [(k, v) for k, v in candidates.items() if v is not None]
    if len(given) != 1:
        names = ", ".join(candidates)
        raise GeneticsUsageError(
            f"provide exactly one of: {names} (got {len(given)}: "
            f"{', '.join(k for k, _ in given) or 'none'})"
        )
    return given[0]


def _reject(branch: str, **ignored: Any) -> None:
    """Raise if the branch that was selected cannot honour an argument the caller supplied.

    Dropping a filter silently is worse than refusing it: `exome(gene=..., resources=[...])`
    that ignores `resources` returns MORE rows than the caller asked for, and nothing in
    the returned frame says which arguments were applied.
    """
    supplied = sorted(k for k, v in ignored.items() if v is not None and v is not False)
    if supplied:
        raise GeneticsUsageError(
            f"{branch} does not accept {', '.join(supplied)}; "
            f"it would be ignored, not applied"
        )


def _csv(value: str | list[str] | None) -> str | None:
    """Accept either a comma-separated string or a list; the executor wants the string."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return ",".join(str(v) for v in value)


def parse_region(region: str) -> tuple[str, int, int]:
    """Split 'chr1:100-200' (or '1:100-200') into (chrom, start, end)."""
    match = _REGION_RE.match(str(region).strip().replace(",", ""))
    if not match:
        raise GeneticsUsageError(
            f"invalid region {region!r}; expected 'chr:start-end', e.g. '19:44900000-44910000'"
        )
    return match.group(1), int(match.group(2)), int(match.group(3))


class GeneticsClient:
    """Async data client. `genetics.<function>` are synchronous wrappers over these."""

    def __init__(self, executor: ToolExecutor | None = None) -> None:
        """Endpoint URLs are read from the environment and are NOT arguments.

        Inside the sandbox the client attaches the PER-EXECUTION token the supervisor
        delivered, audience-bound to the destination it is going to, and never
        INTERNAL_API_SECRET (genetics-results-suite-4h6.44; the machinery is
        `tools/executor.py`'s `_load_sandbox_tokens` and `_SandboxTokenAuth`). That is what
        makes results-api's four per-execution counters apply at all: the shared secret
        satisfies `is_internal_caller` and reaches every handler while resolving no sandbox
        principal, so a request carrying it is served with no accounting whatsoever
        (genetics-results-suite-0lf). Any way for a script to point this at a host of its
        choosing is still a way to hand that token somewhere it should not go, which is why
        the endpoints are not parameters. `executor` remains injectable for the in-process
        callers that already hold a configured one.

        NOT A CONFIDENTIALITY BOUNDARY: a script that can import this module can read the
        token out of the client, out of the supervisor's inherited address space, and — for
        a resident process left by an earlier execution — out of another execution's token
        file. What bounds that is genetics-results-suite-4h6.55, not anything here. The
        token's value is that it is short-lived, scoped, and ATTRIBUTABLE, so the quota
        controls above it are no longer inert.
        """
        # the executor's 500-row inline cap protects the model's context window. A script
        # consumes rows programmatically and can filter them itself, so it gets everything
        # the upstream returned instead of a positional prefix. This is passed to the
        # constructor rather than assigned afterwards: an executor handed in by a caller
        # may be the running service's shared one, and lifting its cap in place would
        # flood the model's context on every MCP call site that relies on it.
        # expose_columns is asked for HERE and not defaulted on in ToolExecutor because
        # the extra key would otherwise land in the MCP tool payload and the chat
        # backend's model input. An INJECTED executor keeps whatever it was built with,
        # so an empty results-api result through one falls back to a bare empty frame.
        #
        # THE UNDERSCORE IS CURATION, NOT ENFORCEMENT. It marks the executor as "not part
        # of the curated SDK surface" so that neither a reader nor a model treats it as a
        # recommended entry point — it does NOT make it unreachable, and nothing here
        # should ever be cited as though it did. `_executor` is one attribute access away
        # from every unwrapped tool method THE SHIPPED EXECUTOR HAS, and a script needs no
        # client at all to get one: `tools/executor.py` is on the SDK_ALLOWLIST in
        # genetics-results-suite `sandbox/prune_venv.py` (it must ship because this module
        # imports ToolExecutor directly), so sandboxed code can just
        # `from genetics_mcp_server.tools.executor import ToolExecutor` and build its own.
        # What it is NOT one attribute access away from is `tools/orchestration.py` — the
        # sandbox gateway, artifact reads and web search — which is a different statement:
        # that code is absent from the image because it cannot run there and the SDK does
        # not need it, not because anything is being withheld from a script.
        # `tools/uniprot.py` is on that same allow-list — a third-party HTTP client ships
        # into the image — so "the third-party tools are unreachable" rests ENTIRELY on
        # egress, with no second layer pruning them out.
        # The real containment boundary, AS SPECIFIED rather than as deployed (the sandbox
        # is not running and the policy is decoration until genetics-results-suite-4h6.7
        # ships a Deployment with the labels it selects), is the sandbox's NETWORK EGRESS
        # ALLOW-LIST — deny-by-default, permitting only db-api:8080 and results-api:4000 —
        # specified in genetics-results-suite `docs/code-execution-security.md`. The
        # constructor PARAMETER stays public (`executor=`) because in-process callers
        # legitimately inject a configured one.
        self._executor = executor or ToolExecutor(row_limit=None, expose_columns=True)

    async def close(self) -> None:
        await self._executor.close()

    # ------------------------------------------------------------------ plumbing

    @staticmethod
    def _payload(result: dict[str, Any]) -> dict[str, Any]:
        if not result.get("success"):
            raise GeneticsError(result.get("error") or "request failed")
        return result

    @classmethod
    def _rows(cls, result: dict[str, Any], key: str = "results") -> pl.DataFrame:
        payload = cls._payload(result)
        cls._check_truncation(payload)
        return _frame(
            payload.get(key) or [],
            columns=payload.get("columns"),
            empty_columns=payload.get("column_names"),
        )

    @staticmethod
    def _check_truncation(payload: dict[str, Any]) -> None:
        """Refuse to hand back a silently shortened result.

        Truncation is the one failure a script cannot detect for itself: the frame it
        receives is well-formed and simply missing rows, so every downstream count, mean
        and join is wrong with no signal. Raising is the only honest option — the caller
        can lower `limit`, narrow `window` or move to `sql()` once told.
        """
        if not payload.get("truncated"):
            return
        # Two different cuts reach this flag, and they need different advice. `capped_by_server`
        # means db-api itself cut the result set at the row cap for this credential — raising
        # `limit`/`max_rows` past that cap does nothing, which the old message never said.
        # The cap's VALUE is db-api's constant (SANDBOX_MAX_ROWS for a sandbox execution, four
        # times higher for a caller holding INTERNAL_API_SECRET), so it is quoted from the
        # response rather than hardcoded here; a db-api that does not report it, or a payload
        # from another path, simply gets the sentence without a number.
        if payload.get("capped_by_server"):
            cap = payload.get("server_row_cap")
            ceiling = f" of {cap:,} rows" if isinstance(cap, int) and cap > 0 else ""
            raise GeneticsError(
                f"the query hit db-api's row cap{ceiling}, so rows are missing and what came "
                "back is a positional prefix. Raising `max_rows` past that cap does nothing — "
                "aggregate or filter in SQL instead of fetching every row."
            )
        raise GeneticsError(
            "result was truncated, so the rows returned are a positional prefix and "
            "not the whole answer. Narrow the query (smaller window/region, more "
            "specific resource) or raise `limit`."
        )

    # ------------------------------------------------------------------ credible sets

    async def credible_sets(
        self,
        *,
        gene: str | None = None,
        qtl_gene: str | None = None,
        variant: str | None = None,
        region: str | None = None,
        phenotype: str | None = None,
        credible_set_id: str | None = None,
        resource: str | None = None,
        data_types: str | list[str] | None = None,
        window: int | None = None,
        coding_only: bool = False,
        leads_only: bool = False,
    ) -> pl.DataFrame:
        """Fine-mapped credible sets, selected by whichever key is supplied.

        gene      — GWAS/QTL credible sets overlapping the gene region +/- window
        qtl_gene  — QTL credible sets where the gene is the molecular trait
        variant   — credible sets containing the variant
        region    — credible sets overlapping 'chr:start-end'
        phenotype — credible sets for one trait (needs `resource`, default finngen);
                    with `credible_set_id`, the variants of that one set;
                    with `leads_only=True`, one lead variant per set
        """
        data_types = _csv(data_types)
        if credible_set_id is not None:
            if phenotype is None:
                raise GeneticsUsageError("credible_set_id requires phenotype")
            _reject(
                "credible_sets(credible_set_id=...)",
                gene=gene,
                qtl_gene=qtl_gene,
                variant=variant,
                region=region,
                data_types=data_types,
                window=window,
                coding_only=coding_only,
                leads_only=leads_only,
            )
            return self._rows(
                await self._executor.get_credible_set_by_id(
                    resource or "finngen", phenotype, credible_set_id
                ),
                key="variants",
            )

        key, value = _one_of(
            gene=gene, qtl_gene=qtl_gene, variant=variant, region=region, phenotype=phenotype
        )
        if key != "gene":
            _reject(f"credible_sets({key}=...)", window=window)
        if key != "region":
            _reject(f"credible_sets({key}=...)", coding_only=coding_only)
        if key != "phenotype":
            _reject(f"credible_sets({key}=...)", leads_only=leads_only)
        if key in ("region", "phenotype"):
            _reject(f"credible_sets({key}=...)", data_types=data_types)

        if key == "gene":
            result = await self._executor.get_credible_sets_by_gene(
                value,
                window=500_000 if window is None else window,
                resource=resource,
                data_types=data_types,
                summarize=False,
            )
        elif key == "qtl_gene":
            result = await self._executor.get_credible_sets_by_qtl_gene(
                value, data_types=data_types, resource=resource, summarize=False
            )
        elif key == "variant":
            result = await self._executor.get_credible_sets_by_variant(
                value, resource=resource, data_types=data_types, summarize=False
            )
        elif key == "region":
            result = await self._executor.get_credible_sets_by_region(
                value, resource=resource, coding_only=coding_only, summarize=False
            )
        elif leads_only:
            result = await self._executor.get_credible_set_leads_by_phenotype(
                value, resource=resource or "finngen"
            )
        else:
            result = await self._executor.get_credible_sets_by_phenotype(
                value, resource=resource or "finngen", summarize=False
            )
        return self._rows(result)

    # ------------------------------------------------------------------ colocalization

    async def colocalization(
        self,
        *,
        variant: str | None = None,
        credible_set_id: str | None = None,
        resource: str | None = None,
        phenotype: str | None = None,
        dual_format: bool = False,
    ) -> pl.DataFrame:
        """Colocalizing credible sets, by variant or by one credible set."""
        if credible_set_id is not None:
            if phenotype is None:
                raise GeneticsUsageError("credible_set_id requires phenotype")
            _reject("colocalization(credible_set_id=...)", variant=variant)
            return self._rows(
                await self._executor.get_colocalization_by_credible_set(
                    resource or "finngen", phenotype, credible_set_id, dual_format=dual_format
                )
            )
        _one_of(variant=variant)
        _reject("colocalization(variant=...)", resource=resource, dual_format=dual_format)
        return self._rows(await self._executor.get_colocalization(variant))

    # ------------------------------------------------------------------ exome / burden

    async def exome(
        self,
        *,
        gene: str | None = None,
        variant: str | None = None,
        region: str | None = None,
        phenotype: str | None = None,
        resource: str | None = None,
        resources: str | list[str] | None = None,
    ) -> pl.DataFrame:
        """Single-variant exome association results."""
        key, value = _one_of(gene=gene, variant=variant, region=region, phenotype=phenotype)
        if key in ("gene", "phenotype"):
            _reject(f"exome({key}=...)", resources=resources)
        if key != "phenotype":
            _reject(f"exome({key}=...)", resource=resource)
        if key == "gene":
            result = await self._executor.get_exome_results_by_gene(value)
        elif key == "variant":
            result = await self._executor.get_exome_results_by_variant(
                value, resources=_csv(resources)
            )
        elif key == "region":
            result = await self._executor.get_exome_results_by_region(
                value, resources=_csv(resources)
            )
        else:
            result = await self._executor.get_exome_results_by_phenotype(
                resource or "finngen", value
            )
        return self._rows(result)

    async def gene_burden(
        self,
        *,
        gene: str | None = None,
        phenotype: str | None = None,
        resource: str | None = None,
    ) -> pl.DataFrame:
        """Gene-level burden test results (genebass, IBD, BipEx2, SCHEMA, ...)."""
        key, value = _one_of(gene=gene, phenotype=phenotype)
        if key == "gene":
            _reject("gene_burden(gene=...)", resource=resource)
            result = await self._executor.get_gene_based_results(value)
        else:
            result = await self._executor.get_gene_based_results_by_phenotype(
                resource or "finngen", value
            )
        return self._rows(result)

    # ------------------------------------------------------------------ HLA

    async def hla(
        self,
        *,
        phenotype: str | list[str] | None = None,
        allele: str | None = None,
        genes: str | list[str] | None = None,
        resource: str = "finngen",
        min_mlogp: float | None = None,
        min_info: float | None = None,
        limit: int | None = None,
    ) -> pl.DataFrame:
        """Classical HLA allele associations, from whichever end is fixed.

        phenotype — every imputed allele (187 across 10 genes) tested against one or more
                    traits; `genes` narrows to e.g. 'HLA-B,HLA-DQB1'
        allele    — the PheWAS view: every trait one allele is associated with, filtered
                    by `min_mlogp` (default 7.3) and `min_info` (default 0.5; pass 0 to
                    see badly-imputed rare alleles, whose betas are artifacts)

        Pass `allele` gene-stripped and two-field ('B*27:05', 'DQB1*02:01'); the written
        'HLA-B*27:05' form is normalized for you. There is no by-variant shape — an HLA
        allele has no chr:pos:ref:alt identity.

        Both shapes spell the statistics the same way — `mlog10p`, `se`, `af`, `af_cases`,
        `af_controls` — even though they read different stores, so per-column access is
        uniform across the two. The column SETS still differ: `allele=` returns the 11
        columns common to both (`phenotype`, `gene`, `allele`, `mlog10p`, `pval`, `beta`,
        `se`, `af`, `af_cases`, `af_controls`, `info`), while `phenotype=` returns those
        plus `resource`, `version`, `chr` and `pos` — 15 in all. A bare concat of the two
        therefore fails on width; select the 11 shared columns on both sides first. The
        trait column is `phenotype` in both, not `trait` or `phenocode` as the rest of the
        suite uses.

        Rank on `mlog10p` — `pval` underflows to a literal 0 for the strongest signals
        (coeliac DQB1*02:01 is mlog10p 1596). Both shapes carry their column names through
        an empty result, so filtering a no-hit phenotype gives an empty frame rather than
        ColumnNotFoundError.
        """
        key, value = _one_of(phenotype=phenotype, allele=allele)
        if key == "phenotype":
            _reject(
                "hla(phenotype=...)", min_mlogp=min_mlogp, min_info=min_info, limit=limit
            )
            phenotypes = [value] if isinstance(value, str) else list(value)
            return self._rows(
                await self._executor.get_hla_by_phenotype(
                    phenotypes, genes=_csv(genes), resource=resource
                )
            )
        _reject("hla(allele=...)", genes=_csv(genes))
        return self._rows(
            await self._executor.get_hla_by_allele(
                value,
                min_mlogp=7.3 if min_mlogp is None else min_mlogp,
                min_info=0.5 if min_info is None else min_info,
                resource=resource,
                max_rows=MAX_ROWS if limit is None else limit,
                with_metadata=True,
            )
        )

    # ------------------------------------------------------------------ regulatory

    async def asm_qtl(
        self,
        *,
        variant: str | None = None,
        gene: str | None = None,
        resources: str | list[str] | None = None,
        window: int = 500_000,
        limit: int = MAX_ROWS,
    ) -> pl.DataFrame:
        """Allele-specific methylation QTL results.

        `limit` is a DISPLAY bound, not a work bound: db-api strips the trailing LIMIT so
        the full join and ORDER BY still run server-side. Lowering it makes the query no
        cheaper, and a truncated result raises rather than returning a prefix.
        """
        key, value = _one_of(variant=variant, gene=gene)
        if key == "variant":
            result = await self._executor.get_asm_qtl_by_variant(value, resources=_csv(resources))
        else:
            result = await self._executor.get_asm_qtl_by_gene(
                value,
                resources=_csv(resources),
                window=window,
                limit=limit,
                with_metadata=True,
            )
        return self._rows(result)

    async def open_chromatin(
        self,
        *,
        variant: str | None = None,
        region: str | None = None,
        peak: str | None = None,
        gene: str | None = None,
        resources: str | list[str] | None = None,
        window: int = 500_000,
        limit: int = MAX_ROWS,
    ) -> pl.DataFrame:
        """Open-chromatin atlas peaks.

        `limit` is a DISPLAY bound, not a work bound: db-api strips the trailing LIMIT so
        the full join and ORDER BY still run server-side. Lowering it makes the query no
        cheaper, and a truncated result raises rather than returning a prefix.
        """
        key, value = _one_of(variant=variant, region=region, peak=peak, gene=gene)
        if key == "variant":
            result = await self._executor.get_open_chromatin_by_variant(
                value, resources=_csv(resources)
            )
        elif key == "region":
            chrom, start, end = parse_region(value)
            result = await self._executor.get_open_chromatin_by_region(
                chrom, start, end, resources=_csv(resources)
            )
        elif key == "peak":
            result = await self._executor.get_open_chromatin_by_peak(
                value, resources=_csv(resources)
            )
        else:
            result = await self._executor.get_open_chromatin_by_gene(
                value,
                resources=_csv(resources),
                window=window,
                limit=limit,
                with_metadata=True,
            )
        return self._rows(result)

    async def peak_to_gene(
        self,
        *,
        peak: str | None = None,
        gene: str | None = None,
        resources: str | list[str] | None = None,
        gencode_version: str | None = None,
    ) -> pl.DataFrame:
        """Open4Gene peak-to-gene links, from either end."""
        key, value = _one_of(peak=peak, gene=gene)
        method = (
            self._executor.get_peak_to_genes if key == "peak" else self._executor.get_gene_to_peaks
        )
        return self._rows(
            await method(value, resources=_csv(resources), gencode_version=gencode_version)
        )

    async def variant_effect(
        self,
        *,
        variant: str | None = None,
        gene: str | None = None,
        resources: str | list[str] | None = None,
        window: int = 500_000,
        limit: int = MAX_ROWS,
    ) -> pl.DataFrame:
        """In-silico predicted variant effects on chromatin accessibility.

        `limit` is a DISPLAY bound, not a work bound: db-api strips the trailing LIMIT so
        the full join and ORDER BY still run server-side. Lowering it makes the query no
        cheaper, and a truncated result raises rather than returning a prefix.
        """
        key, value = _one_of(variant=variant, gene=gene)
        if key == "variant":
            result = await self._executor.get_variant_effect_by_variant(
                value, resources=_csv(resources)
            )
        else:
            result = await self._executor.get_variant_effect_by_gene(
                value,
                resources=_csv(resources),
                window=window,
                limit=limit,
                with_metadata=True,
            )
        return self._rows(result)

    async def mpra(
        self,
        *,
        variant: str | None = None,
        region: str | None = None,
        gene: str | None = None,
        resources: str | list[str] | None = None,
        window: int = 500_000,
        limit: int = MAX_ROWS,
    ) -> pl.DataFrame:
        """Measured MPRA cis-regulatory allelic activity (long: one row per cell line).

        `limit` is a DISPLAY bound, not a work bound: db-api strips the trailing LIMIT so
        the full join and ORDER BY still run server-side. Lowering it makes the query no
        cheaper, and a truncated result raises rather than returning a prefix.
        """
        key, value = _one_of(variant=variant, region=region, gene=gene)
        if key == "variant":
            result = await self._executor.get_mpra_by_variant(value, resources=_csv(resources))
        elif key == "region":
            chrom, start, end = parse_region(value)
            result = await self._executor.get_mpra_by_region(
                chrom, start, end, resources=_csv(resources)
            )
        else:
            result = await self._executor.get_mpra_by_gene(
                value,
                resources=_csv(resources),
                window=window,
                limit=limit,
                with_metadata=True,
            )
        return self._rows(result)

    async def mpra_pip_concordance(
        self,
        gene: str,
        *,
        window: int = 500_000,
        resource: str = "finngen",
        min_pip: float = 0.1,
        limit: int = MAX_ROWS,
    ) -> pl.DataFrame:
        """Fine-mapped PIP cross-referenced against measured MPRA emVar calls.

        Separate from `mpra()` rather than a keyword on it: this is a join of two views and
        returns credible-set columns alongside MPRA columns, so the row shape differs.

        `limit` is a DISPLAY bound, not a work bound: db-api strips the trailing LIMIT so
        the full join and ORDER BY still run server-side. Lowering it makes the query no
        cheaper, and a truncated result raises rather than returning a prefix.
        """
        return self._rows(
            await self._executor.get_mpra_pip_concordance_by_gene(
                gene,
                window=window,
                resource=resource,
                min_pip=min_pip,
                limit=limit,
                with_metadata=True,
            )
        )

    # ------------------------------------------------------------------ annotation

    async def variant_annotation(
        self,
        *,
        variant: str | None = None,
        region: str | None = None,
        gene: str | None = None,
        variants: list[str] | None = None,
        source: str = "finngen",
    ) -> pl.DataFrame:
        """Variant annotations (consequence, AF, gene) for one variant, a region, a gene or a batch."""
        _one_of(variant=variant, region=region, gene=gene, variants=variants)
        return self._rows(
            await self._executor.get_variant_annotations(
                variant=variant, region=region, gene=gene, variants=variants, source=source
            )
        )

    async def gene_annotations(
        self,
        *,
        region: str | None = None,
        nearest_to: str | None = None,
        group: int | str | None = None,
        gene_type: str = "protein_coding",
        n: int | None = None,
        max_distance: int | None = None,
        gencode_version: str | None = None,
        exclude_olfactory: bool | None = None,
    ) -> pl.DataFrame:
        """Genes, selected by region, by proximity to a variant, or by HGNC gene group.

        `n`/`max_distance` apply to `nearest_to` only and `exclude_olfactory` to `group`
        only; supplying one outside its branch is refused rather than ignored.
        """
        key, value = _one_of(region=region, nearest_to=nearest_to, group=group)
        if key != "nearest_to":
            _reject(f"gene_annotations({key}=...)", n=n, max_distance=max_distance)
        if key != "group":
            _reject(f"gene_annotations({key}=...)", exclude_olfactory=exclude_olfactory)
        if key == "group":
            _reject("gene_annotations(group=...)", gencode_version=gencode_version)
        if key == "region":
            chrom, start, end = parse_region(value)
            return self._rows(
                await self._executor.get_genes_in_region(
                    chrom, start, end, gene_type=gene_type, gencode_version=gencode_version
                ),
                key="genes",
            )
        if key == "nearest_to":
            return self._rows(
                await self._executor.get_nearest_genes(
                    value,
                    gene_type=gene_type,
                    n=3 if n is None else n,
                    max_distance=1_000_000 if max_distance is None else max_distance,
                    gencode_version=gencode_version,
                ),
                key="genes",
            )
        by_id = isinstance(value, int) or (isinstance(value, str) and value.isdigit())
        return self._rows(
            await self._executor.get_gene_group_members(
                group_id=int(value) if by_id else None,
                group_name=None if by_id else value,
                exclude_olfactory=True if exclude_olfactory is None else exclude_olfactory,
            ),
            key="members",
        )

    async def expression(self, gene: str) -> pl.DataFrame:
        """Tissue expression for a gene, across expression resources."""
        return self._rows(await self._executor.get_gene_expression(gene))

    async def gene_disease(self, gene: str) -> pl.DataFrame:
        """Mendelian gene-disease associations."""
        return self._rows(await self._executor.get_gene_disease_associations(gene))

    # ------------------------------------------------------------------ sumstats / LD

    async def summary_stats(
        self,
        *,
        phenotypes: list[str] | str,
        variants: list[str] | None = None,
        region: str | None = None,
        resource: str = "finngen",
        data_type: str = "gwas",
    ) -> pl.DataFrame:
        """Summary statistics, for explicit variant-phenotype pairs or for a whole region."""
        pheno_list = [phenotypes] if isinstance(phenotypes, str) else list(phenotypes)
        key, value = _one_of(variants=variants, region=region)
        if key == "variants":
            result = await self._executor.get_summary_stats(
                value, pheno_list, resource=resource, data_type=data_type
            )
        else:
            result = await self._executor.get_summary_stats_by_region(
                value, pheno_list, resource=resource, data_type=data_type
            )
        return self._rows(result)

    async def ld(
        self,
        variant: str,
        other: str | None = None,
        *,
        window: int | None = None,
        r2_threshold: float | None = None,
        panel: str = "sisu42",
    ) -> pl.DataFrame:
        """LD from the FinnGen LD server.

        With `other`, one row for that pair; without it, every variant in LD above the
        threshold. Defaults differ per shape and match the tool layer (0.1 for a named
        pair, 0.6 for a neighbourhood scan).
        """
        if other is not None:
            _reject("ld(variant, other)", window=window)
            result = self._payload(
                await self._executor.get_ld_between_variants(
                    variant, other, r2_threshold=0.1 if r2_threshold is None else r2_threshold,
                    panel=panel,
                )
            )
            return _frame([{
                "variant1": result["variant1"],
                "variant2": result["variant2"],
                "in_ld": result["in_ld"],
                "r2": result.get("r2"),
                "d_prime": result.get("d_prime"),
                "panel": panel,
            }])
        return self._rows(
            await self._executor.get_variants_in_ld(
                variant,
                window=1_500_000 if window is None else window,
                r2_threshold=0.6 if r2_threshold is None else r2_threshold,
                panel=panel,
            ),
            key="variants",
        )

    # ------------------------------------------------------------------ search / SQL

    async def search(
        self,
        query: str | None = None,
        *,
        kind: str = "phenotypes",
        rsids: str | list[str] | None = None,
        limit: int | None = None,
    ) -> pl.DataFrame:
        """Fuzzy search over the phenotype/gene index, or rsID -> variant id lookup."""
        if rsids is not None:
            _reject("search(rsids=...)", query=query, limit=limit)
            return self._rows(
                await self._executor.lookup_variants_by_rsid(_csv(rsids)), key="variants"
            )
        if query is None:
            raise GeneticsUsageError("provide either query= or rsids=")
        if kind == "genes":
            result = await self._executor.search_genes(query, limit=limit or 10)
        elif kind == "phenotypes":
            result = await self._executor.search_phenotypes(query, limit=limit or 100)
        else:
            raise GeneticsUsageError(f"kind must be 'phenotypes' or 'genes', got {kind!r}")
        return self._rows(result, key="results")

    async def sql(self, query: str, *, max_rows: int = 100_000) -> pl.DataFrame:
        """Run read-only SQL against the genetics BigQuery views.

        The db-api rejects anything that is not a plain SELECT over the exposed views, so
        this is the escape hatch for joins the typed functions above do not cover. Name every
        table by its bare view name (`FROM credible_sets_v`) — do NOT prefix it with a
        project or dataset, and do not wrap it in backticks. db-api resolves each name
        against the dataset it was configured with, so the same script runs unchanged
        against dev and production data. The schema docs' examples are written that way.

        Like the `limit=` functions, a truncated result RAISES rather than returning a
        prefix — a short frame with no signal would make every downstream count and join
        silently wrong. Two ceilings can trip it, and neither is raised by passing a bigger
        `max_rows`:

          rows  — db-api caps a sandbox execution at 25,000 returned rows regardless of
                  `max_rows`, and reports both the cut and the cap it applied, so the
                  GeneticsError names the ceiling you actually hit.
          bytes — 50 GB scanned per query and 200 GB across one execution; over either is a
                  GeneticsError too, not a short result.

        So aggregate, filter or add the partition predicate in SQL rather than fetching
        every row and reducing in polars.
        """
        result = self._payload(await self._executor.query_database(query, max_rows=max_rows))
        self._check_truncation(result)
        return _frame(result.get("rows") or [], columns=result.get("columns") or None)

    # ------------------------------------------------------------------ code -> name

    async def lookup_phenotype_names(self, codes: str | list[str]) -> pl.DataFrame:
        """Resolve trait codes ('I9_CHD') to their human-readable names.

        credible_sets, summary_stats and gene_burden all return codes, and
        `search(kind="phenotypes")` is the fuzzy ranked index rather than a lookup, so it
        cannot answer "what is I9_CHD". One row per code; unknown codes come back with the
        upstream's 'Unknown: <code>' placeholder.
        """
        code_list = [codes] if isinstance(codes, str) else list(codes)
        names = self._payload(await self._executor.lookup_phenotype_names(code_list))["names"]
        return _frame(
            [[code, names.get(code)] for code in code_list],
            columns=["phenotype", "name"],
        )

    async def get_dataset_display_names(self) -> pl.DataFrame:
        """Display-name overrides keyed by the raw `dataset` column value."""
        mapping = self._payload(await self._executor.get_dataset_display_names())["display_names"]
        return _frame(
            [[key, value] for key, value in mapping.items()],
            columns=["dataset", "display_name"],
        )

    async def normalize_gene_symbols(self, symbols: str | list[str]) -> pl.DataFrame:
        """Resolve aliases and previous symbols to current approved HGNC symbols.

        One row per input rather than mappings plus a separate `unresolved` list: the tidy
        form is a frame, and `df.filter(pl.col("symbol").is_null())` recovers the inputs
        that did not resolve. Canonicalising a user-supplied gene list this way is what
        makes a later `credible_sets(gene=...)` reliable.
        """
        given = [symbols] if isinstance(symbols, str) else list(symbols)
        cleaned = [str(s).strip() for s in given if s and str(s).strip()]
        payload = self._payload(await self._executor.normalize_gene_symbols(cleaned))
        by_input = {m.get("input"): m for m in payload.get("mappings") or []}
        return _frame(
            [
                [
                    symbol,
                    by_input.get(symbol, {}).get("approved"),
                    symbol in by_input,
                    by_input.get(symbol, {}).get("matched_on"),
                ]
                for symbol in cleaned
            ],
            columns=["input", "symbol", "resolved", "matched_on"],
        )

    async def schema(self, table: str | None = None) -> dict[str, Any]:
        """Column-level schema of the BigQuery views. A dict, not a frame: it is nested."""
        return self._payload(await self._executor.get_database_schema(table))["schema"]

    async def resources(self) -> dict[str, Any]:
        """Catalog of available data resources, grouped by data product."""
        return self._payload(await self._executor.get_available_resources())["resources"]

    async def datasets(
        self, resource: str | None = None, include_stats: bool = True
    ) -> dict[str, Any]:
        """Dataset catalog with descriptions and aggregate stats."""
        return self._payload(
            await self._executor.list_datasets(resource=resource, include_stats=include_stats)
        )["datasets"]


# --------------------------------------------------------------------------- audit trail

# genetics-results-suite-4h6.12. Today every data access is an MCP tool call and chat-backend
# logs one `Executing tool: <name> with input: <dict>` line per call under a
# `[user=…] [session=…] ` prefix (llm_service.py). Once access moves inside a sandboxed
# script that trail disappears: one `run_analysis` tool call stands for an unbounded number
# of queries. These lines put it back, one per SDK function call, in the same shape so the
# same log queries answer the same questions — `Executing tool:` still matches exactly what
# it matched before, and `Executing SDK function:` is its counterpart for script access.
#
# WHAT THE TRAIL COVERS, AND WHAT IT DOES NOT. These lines cover calls made through the
# curated SDK surface — `GeneticsClient` coroutine methods and the `genetics.<fn>` sync
# wrappers that delegate to them. They do NOT cover `client._executor.<method>()`, which is
# one attribute access away and returns the same data with no line at all (see the
# constructor's note: `tools/executor.py` ships in the sandbox image, so a script can build
# its own executor without a client). Closing that means instrumenting `ToolExecutor`
# itself, which is a much larger change: every MCP tool call goes through the same methods,
# so it would need chat-backend's existing `Executing tool:` line suppressed or deduplicated
# against it. Until then this is "audited if you use the recommended surface", not
# "unbypassable", and nothing should cite it as the latter.
#
# WHERE THE LINE GOES, AND WHAT HAPPENS TO IT THERE. In-process (chat-backend, tests) it is
# an ordinary record on `genetics_mcp_server.sdk.audit` and lands wherever `Executing tool:`
# lands. In the sandbox the SDK runs in a forked child that configures no logging at all, so
# a handler is installed on first use. The two shapes are not equivalent:
#
#   dedicated fd (GENETICS_SDK_AUDIT_FD) — a pipe whose READ END the supervisor holds
#       (genetics-results-suite-4h6.45, `sandbox/supervisor.py` `_AuditForwarder`).
#       Propagation is switched off so inherited handlers cannot put the records back on a
#       shared stream. What the supervisor does with what arrives is the part that matters
#       here: it treats this stream as UNTRUSTED INPUT. Each line is re-parsed, matched whole
#       against the shapes below, and re-emitted in the supervisor's own framing on the POD'S
#       STDOUT — everything before the marker, including the `[user=…] [session=…]
#       [execution=…]` prefix rendered below, is DISCARDED and replaced with the sandbox
#       token's `sub`/`sid`/`jti`. The rate, byte and per-line caps are applied there too. So
#       on this path the identity in the collected record is the supervisor's, and the shape
#       below is what makes a record admissible rather than what makes it true.
#   anything else — the records share a stream (stderr, or the host's logging config) that
#       the audited script writes to itself with `print(..., file=sys.stderr)`. Per-value
#       escaping is irrelevant when the writer owns the stream: a forged line parses cleanly
#       through this repo's own parser and can name any user and session it likes. When no
#       fd is configured the channel says so, once, in the stream itself — see
#       `_warn_shared_stream` — and `scripts/analyze_conversations.py` repeats the warning in
#       any report built from such a log.
#
# WHICH SIDE IS AUTHORITATIVE, since this file and the supervisor both render the identity: ON
# THE FD PATH, THE SUPERVISOR'S STAMP IS. It comes from the token's signed claims; the values
# below come from `SANDBOX_USER`/`SANDBOX_SESSION_ID`/`SANDBOX_EXECUTION_ID`, which the
# supervisor sets from the same claims but the audited script owns and can rewrite between two
# calls. They are still rendered — the in-process path has no supervisor to stamp anything —
# but on the fd path they never leave the pod. THE ENV PREFIX AND THE SIGNED CLAIMS ARE NOT
# THE SAME EVIDENCE, and only one of them survives to a collector.
#
# AND NOTHING IN THIS FILE CAN CHANGE THAT. The audited script shares this address space, so
# every in-process control here is reachable from the code it is supposed to constrain. This
# is not a list of gaps to be patched one by one; it is the consequence of where the code
# runs, and each of these was demonstrated against an earlier revision of this module:
#
#   FORGERY   — `logging.getLogger("genetics_mcp_server.sdk.audit").info(...)` emits a record
#               indistinguishable from ours, and `os.write(int(os.environ["…AUDIT_FD"]), …)`
#               puts arbitrary bytes on the dedicated fd itself. The fd number has to be in
#               the child's environment for the SDK to find it, so the script can find it too.
#   SUPPRESSION — `logger.disabled = True`, `setLevel`, `addFilter`, `removeHandler` all
#               silence real records, from the script, with one line.
#   BUDGET RESET — any per-execution counter keyed on an environment variable is reset by
#               rewriting that variable, which is why the budget below is keyed on nothing
#               the script can write.
#
# SO: THESE RECORDS ARE A BEST-EFFORT OBSERVABILITY SIGNAL, NOT A SECURITY CONTROL. They tell
# an operator what a well-behaved script did. They do not bound what a hostile one can claim,
# hide or emit, and nothing — this repo's docs, a review, an incident write-up — may cite them
# as evidence of what happened under an assumption of compromise. The controls that DO hold
# against a hostile script are the ones outside the process: the sandbox's network policy,
# db-api's own `endpoint_access` lines, and the byte/row quotas.
#
# WHAT ACTUALLY WORKS IS OUT HERE, AND genetics-results-suite-4h6.45 HAS LANDED IT:
#
#   1. THE SUPERVISOR READS THE FD AND ENFORCES ON THE READ END. The child holds only the
#      write end; the rate cap, the byte cap and the per-line length cap are applied by the
#      supervisor as it reads, where no code in the child can raise, reset or bypass them.
#      Overrun is the supervisor's decision — it drops, counts, announces the cut once and
#      emits a per-execution summary — not the child's.
#   2. THE SUPERVISOR STAMPS IDENTITY, from the sandbox token's `sub`/`sid`/`jti`, over
#      whatever `_audit_identity` below rendered. The child is not asked who it is.
#   3. THE CHILD'S FRAMING IS UNTRUSTED INPUT. The supervisor matches each line whole against
#      the record shapes, drops what does not match, and re-emits the rest in its own framing.
#      The sanitisation below reduces the damage a malformed line does in the SHARED-stream
#      case; it is not what makes the collected record safe.
#
# WHAT STILL DOES NOT FOLLOW, so that nothing reads the paragraph above as more than it says:
# these records still do not bound what a hostile script DID. It can emit well-formed records
# for calls it never made, and `_executor` reads nothing here can see. The supervisor bounds
# who a record is attributed to and what shape it may take — not whether it happened.
_audit_logger = logging.getLogger("genetics_mcp_server.sdk.audit")

# A logger with no level of its own inherits root's. A host that leaves root at WARNING would
# therefore discard this entire control with no error and no missing-handler warning, so the
# level is pinned here at import instead of only on the fallback path. (Handler levels in the
# host's own config still apply; nothing here can override those.)
_audit_logger.setLevel(logging.INFO)

# Statements ABOUT the channel — it is shared, it was truncated — go on a child logger. They
# propagate into the same sink, so an operator and the analyzer both see them, but they are
# not data accesses: they must not parse as an SDK call or be counted as one.
_audit_meta_logger = logging.getLogger("genetics_mcp_server.sdk.audit.meta")
_audit_meta_logger.setLevel(logging.INFO)

_AUDIT_FD_ENV = "GENETICS_SDK_AUDIT_FD"

SHARED_STREAM_WARNING = (
    "SDK audit records here are NOT a tamper-evident audit trail: no "
    f"{_AUDIT_FD_ENV} was configured, so they share a stream the audited script can write to "
    "itself. Any line below may be forged, including its user and session. A log collected "
    "from the sandbox does NOT look like this: there the supervisor holds the fd's read end, "
    "re-frames every record and stamps identity from the token (4h6.45), so this warning "
    "appearing at all means the stream came from somewhere else."
)

# WHAT IS BOUNDED, AND WHAT MUST NEVER BE. Only records for calls that NEVER REACHED THE
# EXECUTOR — refusals — are bounded. Records for calls that did reach it are never dropped.
#
# The two are not the same primitive. A refusal costs the script nothing: no socket, no
# upstream, no bytes, just local argument validation raising, which is why 1000 of them were
# driven through in ~50ms. That is the flooding primitive, and it is worth bounding. A call
# that reached the executor is already rate-limited by an HTTP round-trip to db-api and is
# charged against the byte and row quotas the rest of the sandbox's resource controls rely
# on, so it cannot be driven at flood rates in the first place — and a bound on it is not a
# bound at all but a SUPPRESSION PRIMITIVE: an earlier revision counted both against one
# ceiling, so 1001 cheap refusals bought silence for every genuine `sql()` read that followed.
# Silently dropping the records that matter, on demand, is strictly worse than the noisy flood
# the ceiling replaced. Hence: bound the cheap side, never the side that read data.
#
# KEYED ON A MODULE-LEVEL PROCESS COUNTER, deliberately, and on nothing else. Any key the
# script can write is a reset button: keying this on SANDBOX_EXECUTION_ID meant a loop that
# rewrote that variable restored the full flood (19,622 lines/s — higher than before the
# ceiling existed). A process-global counter is not resettable from outside this module, and a
# script that can reach into this module's globals can also just call the logger directly, so
# nothing is lost by not defending that case here. This does mean a supervisor that reuses one
# process across executions shares one refusal budget across them; that is the right trade,
# and the supervisor-side enforcement described above is what removes the need for it.
#
# WHY 1000: refusals are a bug signal, not a workload. The heaviest observed real session is
# 58 tool calls; a script emitting a four-figure count of refusals is looping on a mistake,
# and the first thousand say so as well as a million would.
_AUDIT_MAX_REFUSALS = 1000

# Statements ABOUT the channel are themselves emitted from script-driven paths, so the meta
# channel needs its own process-global bound: the truncation notice used to fire once per
# execution id, and rotating that id produced 3,873 notices in one second. Small, because the
# only things that belong here are the shared-stream warning and the refusal cut, once each.
_AUDIT_MAX_META_RECORDS = 8

_audit_handler_pid: int | None = None
_audit_dedicated_fd = False
_audit_refusals = 0
_audit_meta_records = 0
_audit_emit_failures = 0


def _install_audit_handler(stream: Any) -> None:
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    # marked so a re-check after fork replaces OUR handler and leaves the host's alone
    handler._sdk_audit_owned = True  # type: ignore[attr-defined]
    _audit_logger.addHandler(handler)


def _ensure_audit_handler() -> None:
    """Point the audit logger at a dedicated fd if there is one, else at a shared stream.

    KEYED ON THE PID, not on a one-shot flag. The sandbox child is forked from a supervisor
    that may already have made SDK calls, and both this module's state and the parent's
    handlers survive `fork()` — a one-shot flag therefore left the child writing into the
    parent's inherited sink and never reading GENETICS_SDK_AUDIT_FD at all, which made the
    one separation mechanism unreachable in exactly the shape it was written for.

    A host process (chat-backend, the MCP server, pytest) owns its own logging config, and
    adding a second handler there would duplicate every line into a different sink — so the
    shared-stream branch installs one only when nothing else has configured logging. The
    dedicated-fd branch always installs, and stops propagation: inherited handlers would
    otherwise copy every record back onto the shared stream the fd exists to escape.
    """
    global _audit_handler_pid, _audit_dedicated_fd
    pid = os.getpid()
    if _audit_handler_pid == pid:
        return
    _audit_handler_pid = pid
    _audit_dedicated_fd = False
    for handler in [h for h in _audit_logger.handlers if getattr(h, "_sdk_audit_owned", False)]:
        _audit_logger.removeHandler(handler)

    fd = os.environ.get(_AUDIT_FD_ENV, "").strip()
    if fd.isdigit():
        try:
            # closefd=False: the fd belongs to the supervisor that passed it in
            stream = os.fdopen(int(fd), "w", buffering=1, closefd=False)
        except OSError:
            stream = None
        if stream is not None:
            _install_audit_handler(stream)
            _audit_logger.propagate = False
            _audit_dedicated_fd = True
            return

    _audit_logger.propagate = True
    if not logging.getLogger().handlers and not _audit_logger.handlers:
        _install_audit_handler(sys.stderr)
    _warn_shared_stream()


def _warn_shared_stream() -> None:
    """Say in the stream itself that the stream is not trustworthy.

    Emitted once per process, before any record it qualifies, so a collected log carries its
    own provenance: a reader (or the analyzer) does not have to know how the process was
    launched to know whether the lines below can be forged.
    """
    _emit_meta(SHARED_STREAM_WARNING)


def _emit_meta(message: str) -> None:
    """Emit a statement about the channel, bounded per process, carrying no script-chosen text.

    Both properties are load-bearing. The bound: these are emitted from paths a script drives,
    so an unbounded meta channel is the same flooding primitive one level up. No script-chosen
    text: the truncation notice used to interpolate `[execution=…]`, which is whatever the
    script last wrote to SANDBOX_EXECUTION_ID — chosen bytes on the one channel that exists to
    describe the log rather than to carry the script's input. Callers must pass literals.
    """
    global _audit_meta_records
    if _audit_meta_records >= _AUDIT_MAX_META_RECORDS:
        return
    _audit_meta_records += 1
    _emit(_audit_meta_logger, logging.WARNING, message)


def _emit(logger: logging.Logger, level: int, message: str) -> None:
    """Emit without ever letting logging break the call it describes.

    The wrapper's contract is transparency: a handler that raises (a full disk, a closed fd
    passed by a supervisor that went away) must not turn a SUCCESSFUL data access into a
    failed one. Swallowing silently would be its own defect, so the first failure is reported
    on the process's original stderr — `sys.__stderr__`, not `sys.stderr`, which a sandboxed
    script may have rebound — and the count stays readable for tests and for callers.
    """
    global _audit_emit_failures
    try:
        logger.log(level, message)
    except Exception as exc:
        _audit_emit_failures += 1
        if _audit_emit_failures == 1:
            try:
                print(
                    f"genetics sdk audit: emitting a record failed ({type(exc).__name__}); "
                    "SDK calls are no longer being recorded",
                    file=sys.__stderr__,
                    flush=True,
                )
            except Exception:
                pass


def _audit_refusal_budget() -> bool:
    """False once this PROCESS has spent its refusal budget; announces the cut exactly once.

    Applies only to calls that never reached the executor — see `_AUDIT_MAX_REFUSALS` for why
    reads are deliberately not bounded here, and why the key is a process global rather than
    anything the audited script can write.
    """
    global _audit_refusals
    if _audit_refusals < _AUDIT_MAX_REFUSALS:
        _audit_refusals += 1
        return True
    if _audit_refusals == _AUDIT_MAX_REFUSALS:
        _audit_refusals += 1
        _emit_meta(
            f"SDK audit truncated after {_AUDIT_MAX_REFUSALS} records; further REFUSED SDK "
            "calls in this process are NOT recorded. Calls that reached the executor are "
            "still recorded in full."
        )
    return False


# Identity is read from the environment and the environment is script-writable, so these three
# fields get the SAME treatment as an argument value — they were interpolated raw and unbounded
# until `4h6.12`'s last revision, which is a defect on any architecture and not merely a
# sandbox one. Two consequences were measured: `SANDBOX_USER` of 100 KB put 100,431 bytes on
# the stream for ONE legitimate call, and `SANDBOX_USER = "alice\n[user=admin@finngen.fi"`
# rendered a line this repo's OWN parser (`scripts/analyze_conversations.py`) read back as
# `user='admin@finngen.fi'` — a forged attribution needing no logger access at all.
#
# A value that fails is REPLACED, never truncated. Truncating "admin@finngen.fi.attacker.test"
# to a plausible prefix manufactures a different, credible-looking identity; `<invalid>` is
# unmistakably not a user id, keeps the field's position so the line still parses, and contains
# no `]` so it cannot escape the bracket the parser keys on.
_AUDIT_SAFE_IDENTITY_RE = re.compile(r"\A[A-Za-z0-9_.:/@|+-]{1,64}\Z")
_AUDIT_BAD_IDENTITY = "<invalid>"


def _audit_identifier(value: str) -> str:
    return value if _AUDIT_SAFE_IDENTITY_RE.match(value) else _AUDIT_BAD_IDENTITY


def _audit_identity() -> tuple[str, str, str]:
    """(user, session, execution) for the audit prefix, sanitised.

    Read from the environment per call rather than cached: the supervisor sets them on the
    forked child, and the SDK may well be imported before that happens. Which also means the
    audited script can rewrite them at will — this function reads exactly what the script
    controls, which is why the supervisor, not the child, has to stamp identity (`4h6.45`; see
    the module header). Sanitising here bounds the damage; it does not make the values true.

    IN THE SANDBOX ALL THREE ARE SET AND NONE OF THEM IS WHAT GETS COLLECTED. The supervisor
    puts the sandbox token's `sub`, `sid` and `jti` into the child's environment
    (`ExecutionDirs.child_env`), so the line renders; and then, on the dedicated-fd path, it
    DISCARDS this prefix and re-stamps from the same claims on its own side (`4h6.45`).
    Reading them here is for the in-process path and for the shape of the line — never for
    attribution, because the script owns this environment. `jti` is also the
    `/scratch/<execution-id>` directory name, which is what makes the collected line joinable
    with db-api's `endpoint_access` lines and with chat-backend's own.
    """
    env = os.environ
    return (
        _audit_identifier(env.get("SANDBOX_USER") or "unknown"),
        _audit_identifier(env.get("SANDBOX_SESSION_ID") or "unknown"),
        _audit_identifier(env.get("SANDBOX_EXECUTION_ID") or "unknown"),
    )


# An identifier-shaped string is logged verbatim; anything else is reduced to its type and
# length. This is a DISCLOSURE decision, not formatting. The tool layer logs raw inputs
# because they are model-authored and bounded by a schema; SDK arguments are SCRIPT-authored
# and unbounded, so raw logging would (a) let an injected script write chosen text —
# including forged `\n`-separated log lines — into the operator's log pipeline, and (b) copy
# whole `sql()` query bodies, which carry free text and are the argument most likely to
# embed data, into a sink read by people who did not ask for it. The charset admits gene
# symbols, variant ids, rsids, phenotype codes, regions, dataset and view names — the values
# that answer "what did that script actually read?" — and excludes whitespace, quotes and
# every control character, so the rendered line can be neither forged nor expanded.
#
# `\Z`, NOT `$`. `$` matches before a terminal newline, so `'IL7R\n'` passed as
# identifier-shaped and the verbatim bound was 65 characters rather than 64. Only the `repr()`
# below defanged it, which makes the whole property rest on a formatting choice a future
# readability tweak would drop; `\Z` puts it back in the charset where the comment says it is.
# What it does NOT cover: `sql()` bodies and every list argument fall to `<str:N>`/`<list:N>`,
# so for the two most powerful shapes the line records that a read happened and not what was
# read — see docs/project-spec.md, which states the limitation next to the claim.
_AUDIT_SAFE_VALUE_RE = re.compile(r"\A[A-Za-z0-9_.:/@|+-]{1,64}\Z")


def _summarize_value(value: Any) -> str:
    if value is None or isinstance(value, (bool, int, float)):
        return repr(value)
    if isinstance(value, str):
        return repr(value) if _AUDIT_SAFE_VALUE_RE.match(value) else f"<str:{len(value)}>"
    if isinstance(value, (list, tuple, set, frozenset, dict)):
        return f"<{type(value).__name__}:{len(value)}>"
    # `__name__` is whatever the caller's class says it is — arbitrary text, including
    # non-ASCII — so this line is NOT a bound, and the charset above does not cover it. The
    # sandbox supervisor's read end is where that is held (printable ASCII minus brackets and
    # backslash); on any shared stream nothing holds it at all.
    return f"<{type(value).__name__}>"


def _summarize_arguments(signature: inspect.Signature, args: tuple, kwargs: dict) -> str:
    """Render the supplied arguments dict-style, mirroring the tool line's `{...}` input.

    Only what the caller actually passed appears, as on the tool side: defaults are the
    SDK's, not the script's, and a line full of them hides the two arguments that mattered.
    """
    try:
        # `signature` is the UNBOUND method's, so `self` has to occupy the first slot or a
        # positional argument binds to it and is then dropped by the filter below
        bound = signature.bind_partial(None, *args, **kwargs)
    except TypeError:
        # the call is about to raise anyway; the attempt is still worth a line
        #
        # THIS SHAPE HAS NO BRACES, and it is on the sandbox supervisor's admit-list for that
        # reason (genetics-results-suite, supervisor.py `_AUDIT_ARGS_RE`): the read end matches
        # every line whole and drops what does not match, so a `{...}`-only pattern there put
        # the record of an ordinary argument mistake — one extra positional, one unknown
        # keyword — into `dropped_unparseable`, which an operator reads as tampering. Changing
        # this string is changing that wire shape.
        return "<unavailable>"
    items = [
        f"{name!r}: {_summarize_value(value)}"
        for name, value in bound.arguments.items()
        if name != "self"
    ]
    return "{" + ", ".join(items) + "}"


def _result_rows(result: Any) -> int:
    """Rows the call returned.

    `schema`/`resources`/`datasets` return nested dicts, where this is the number of
    top-level entries — the only honest count for a non-tabular answer, and still the signal
    that separates "read everything" from "read nothing".
    """
    if isinstance(result, pl.DataFrame):
        return result.height
    if isinstance(result, (list, tuple, set, frozenset, dict)):
        return len(result)
    return 1


def _audit(
    function: str,
    summary: str,
    *,
    rows: int | None,
    error: str | None = None,
    cancelled: bool = False,
) -> None:
    """One record per call. `rows=None` means the call never reached the executor.

    A call refused by local argument validation read NOTHING, so it gets a different marker
    and no `rows:` field at all. Recording it in the same shape as a real data access both
    inflated the flood (the refusal path is the cheap one — no network, no upstream) and
    polluted the answer to "what did that script read?" with calls that read nothing.

    It is also the ONLY branch the budget applies to: a refusal is free to the caller and can
    therefore be driven at flood rates, while a call that reached the executor paid an HTTP
    round-trip and is charged against the sandbox's byte and row quotas. Dropping the latter
    is not a flood control, it is a way for a script to buy silence — see `_AUDIT_MAX_REFUSALS`.
    """
    _ensure_audit_handler()
    user, session, execution = _audit_identity()
    prefix = f"[user={user}] [session={session}] [execution={execution}] "
    if rows is None:
        if not _audit_refusal_budget():
            return
        _emit(
            _audit_logger,
            logging.INFO,
            f"{prefix}Rejected SDK function: {function} with input: {summary} error: {error}",
        )
        return
    line = f"{prefix}Executing SDK function: {function} with input: {summary} rows: {rows}"
    if error:
        line = f"{line} error: {error}"
    if cancelled:
        line = f"{line} cancelled"
    _emit(_audit_logger, logging.INFO, line)


def _audited(method):
    """Wrap one `GeneticsClient` coroutine method so every call emits exactly one line.

    functools.wraps plus an untouched `__signature__` is load-bearing, not hygiene:
    `list_capabilities` renders the SDK catalogue out of these live objects with `inspect`,
    and `sdk._make_sync` builds the sync surface by slicing the same signature.
    """
    signature = inspect.signature(method)

    @functools.wraps(method)
    async def wrapper(self, *args: Any, **kwargs: Any):
        summary = _summarize_arguments(signature, args, kwargs)
        try:
            result = await method(self, *args, **kwargs)
        except GeneticsUsageError as e:
            # every GeneticsUsageError in this module is raised by _one_of / _reject /
            # parse_region BEFORE any executor call, so this branch is exactly "the call
            # never reached the executor" and is recorded as a refusal, not as a read
            _audit(method.__name__, summary, rows=None, error=type(e).__name__)
            raise
        except asyncio.CancelledError:
            # a cancelled call is not a failed read — the task was taken away from it, and
            # `error: CancelledError` would file every shutdown and timeout as a failure
            _audit(method.__name__, summary, rows=0, cancelled=True)
            raise
        except BaseException as e:
            # the exception TYPE only: upstream error strings are not ours to copy into a
            # sink read by people who did not ask for them
            _audit(method.__name__, summary, rows=0, error=type(e).__name__)
            raise
        _audit(method.__name__, summary, rows=_result_rows(result))
        return result

    return wrapper


def _instrument(cls: type) -> None:
    """Instrument at the client, not at `sdk._make_sync`: the sync functions delegate here,
    so one wrapper covers both surfaces and neither double-counts."""
    for name, method in list(vars(cls).items()):
        if name.startswith("_") or name == "close" or not inspect.iscoroutinefunction(method):
            continue
        setattr(cls, name, _audited(method))


_instrument(GeneticsClient)
