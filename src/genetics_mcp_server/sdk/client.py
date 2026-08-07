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

import re
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


def _frame(rows: Any, columns: list[str] | None = None) -> pl.DataFrame:
    """Build a DataFrame from either row dicts or positional rows plus column names.

    Both shapes occur upstream: the results-api returns JSON objects, while db-api returns
    a list per row with the names in a separate `columns` key. Handing positional rows to
    `pl.from_dicts` does not raise — it silently produces a transposed frame with
    fabricated `column_N` names and stringified values — so `columns`, when the payload
    exposes it, decides which constructor is used rather than being a hint.

    infer_schema_length=None scans every row, so a column that is null in the first rows
    and populated later still gets a usable dtype; strict=False is the fallback for mixed
    types within a column, which the upstream JSON does produce.
    """
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

        The client attaches INTERNAL_API_SECRET to every request it makes, so any way for
        a script to point it at a host of its choosing is a way to exfiltrate the secret
        that authenticates to both results-api and db-api. `executor` remains injectable
        for the in-process callers that already hold a configured one.

        MITIGATION, not the answer: a script that can import this module can also read
        os.environ. The SDK must eventually carry a short-lived scoped token instead of
        INTERNAL_API_SECRET — see genetics-results-suite docs/code-execution-security.md
        and tasks genetics-results-suite-4h6.9 / .14.
        """
        # the executor's 500-row inline cap protects the model's context window. A script
        # consumes rows programmatically and can filter them itself, so it gets everything
        # the upstream returned instead of a positional prefix. This is passed to the
        # constructor rather than assigned afterwards: an executor handed in by a caller
        # may be the running service's shared one, and lifting its cap in place would
        # flood the model's context on every MCP call site that relies on it.
        self.executor = executor or ToolExecutor(row_limit=None)

    async def close(self) -> None:
        await self.executor.close()

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
        return _frame(payload.get(key) or [], columns=payload.get("columns"))

    @staticmethod
    def _check_truncation(payload: dict[str, Any]) -> None:
        """Refuse to hand back a silently shortened result.

        Truncation is the one failure a script cannot detect for itself: the frame it
        receives is well-formed and simply missing rows, so every downstream count, mean
        and join is wrong with no signal. Raising is the only honest option — the caller
        can lower `limit`, narrow `window` or move to `sql()` once told.
        """
        if payload.get("truncated"):
            raise GeneticsError(
                "result was truncated, so the rows returned are a positional prefix and "
                "not the whole answer. Narrow the query (smaller window/region, more "
                "specific resource) or raise `limit`."
            )
        if payload.get("download_capped_at_100k"):
            raise GeneticsError(
                "the query hit db-api's 100,000-row transfer cap, so rows are missing. "
                "Aggregate or filter in SQL instead of fetching every row."
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
                await self.executor.get_credible_set_by_id(
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
            result = await self.executor.get_credible_sets_by_gene(
                value,
                window=500_000 if window is None else window,
                resource=resource,
                data_types=data_types,
                summarize=False,
            )
        elif key == "qtl_gene":
            result = await self.executor.get_credible_sets_by_qtl_gene(
                value, data_types=data_types, resource=resource, summarize=False
            )
        elif key == "variant":
            result = await self.executor.get_credible_sets_by_variant(
                value, resource=resource, data_types=data_types, summarize=False
            )
        elif key == "region":
            result = await self.executor.get_credible_sets_by_region(
                value, resource=resource, coding_only=coding_only, summarize=False
            )
        elif leads_only:
            result = await self.executor.get_credible_set_leads_by_phenotype(
                value, resource=resource or "finngen"
            )
        else:
            result = await self.executor.get_credible_sets_by_phenotype(
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
                await self.executor.get_colocalization_by_credible_set(
                    resource or "finngen", phenotype, credible_set_id, dual_format=dual_format
                )
            )
        _one_of(variant=variant)
        _reject("colocalization(variant=...)", resource=resource, dual_format=dual_format)
        return self._rows(await self.executor.get_colocalization(variant))

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
            result = await self.executor.get_exome_results_by_gene(value)
        elif key == "variant":
            result = await self.executor.get_exome_results_by_variant(
                value, resources=_csv(resources)
            )
        elif key == "region":
            result = await self.executor.get_exome_results_by_region(
                value, resources=_csv(resources)
            )
        else:
            result = await self.executor.get_exome_results_by_phenotype(
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
            result = await self.executor.get_gene_based_results(value)
        else:
            result = await self.executor.get_gene_based_results_by_phenotype(
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

        THE TWO SHAPES DO NOT RETURN THE SAME COLUMNS, because they read different stores.
        The trait column is `phenotype` in both (not `trait` or `phenocode`, which the rest
        of the suite uses), but the statistics are spelled differently:

            phenotype=  (per-trait files)  mlog10p  se      af      af_cases      af_controls
            allele=     (hla_associations_v)  mlogp  sebeta  af_alt  af_alt_cases  af_alt_controls

        Rank on the -log10 p-value in both cases — `pval` underflows to a literal 0 for the
        strongest signals (coeliac DQB1*02:01 is mlog10p 1596). A script that consumes both
        shapes must rename rather than assume. Only the `allele=` shape carries its column
        names through an empty result; results-api returns a bare `[]` with no schema.
        """
        key, value = _one_of(phenotype=phenotype, allele=allele)
        if key == "phenotype":
            _reject(
                "hla(phenotype=...)", min_mlogp=min_mlogp, min_info=min_info, limit=limit
            )
            phenotypes = [value] if isinstance(value, str) else list(value)
            return self._rows(
                await self.executor.get_hla_by_phenotype(
                    phenotypes, genes=_csv(genes), resource=resource
                )
            )
        _reject("hla(allele=...)", genes=_csv(genes))
        return self._rows(
            await self.executor.get_hla_by_allele(
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
            result = await self.executor.get_asm_qtl_by_variant(value, resources=_csv(resources))
        else:
            result = await self.executor.get_asm_qtl_by_gene(
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
            result = await self.executor.get_open_chromatin_by_variant(
                value, resources=_csv(resources)
            )
        elif key == "region":
            chrom, start, end = parse_region(value)
            result = await self.executor.get_open_chromatin_by_region(
                chrom, start, end, resources=_csv(resources)
            )
        elif key == "peak":
            result = await self.executor.get_open_chromatin_by_peak(
                value, resources=_csv(resources)
            )
        else:
            result = await self.executor.get_open_chromatin_by_gene(
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
            self.executor.get_peak_to_genes if key == "peak" else self.executor.get_gene_to_peaks
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
            result = await self.executor.get_variant_effect_by_variant(
                value, resources=_csv(resources)
            )
        else:
            result = await self.executor.get_variant_effect_by_gene(
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
            result = await self.executor.get_mpra_by_variant(value, resources=_csv(resources))
        elif key == "region":
            chrom, start, end = parse_region(value)
            result = await self.executor.get_mpra_by_region(
                chrom, start, end, resources=_csv(resources)
            )
        else:
            result = await self.executor.get_mpra_by_gene(
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
            await self.executor.get_mpra_pip_concordance_by_gene(
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
            await self.executor.get_variant_annotations(
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
                await self.executor.get_genes_in_region(
                    chrom, start, end, gene_type=gene_type, gencode_version=gencode_version
                ),
                key="genes",
            )
        if key == "nearest_to":
            return self._rows(
                await self.executor.get_nearest_genes(
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
            await self.executor.get_gene_group_members(
                group_id=int(value) if by_id else None,
                group_name=None if by_id else value,
                exclude_olfactory=True if exclude_olfactory is None else exclude_olfactory,
            ),
            key="members",
        )

    async def expression(self, gene: str) -> pl.DataFrame:
        """Tissue expression for a gene, across expression resources."""
        return self._rows(await self.executor.get_gene_expression(gene))

    async def gene_disease(self, gene: str) -> pl.DataFrame:
        """Mendelian gene-disease associations."""
        return self._rows(await self.executor.get_gene_disease_associations(gene))

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
            result = await self.executor.get_summary_stats(
                value, pheno_list, resource=resource, data_type=data_type
            )
        else:
            result = await self.executor.get_summary_stats_by_region(
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
                await self.executor.get_ld_between_variants(
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
            await self.executor.get_variants_in_ld(
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
                await self.executor.lookup_variants_by_rsid(_csv(rsids)), key="variants"
            )
        if query is None:
            raise GeneticsUsageError("provide either query= or rsids=")
        if kind == "genes":
            result = await self.executor.search_genes(query, limit=limit or 10)
        elif kind == "phenotypes":
            result = await self.executor.search_phenotypes(query, limit=limit or 100)
        else:
            raise GeneticsUsageError(f"kind must be 'phenotypes' or 'genes', got {kind!r}")
        return self._rows(result, key="results")

    async def sql(self, query: str, *, max_rows: int = 100_000) -> pl.DataFrame:
        """Run read-only SQL against the genetics BigQuery views.

        The db-api rejects anything that is not a plain SELECT over the exposed views, so
        this is the escape hatch for joins the typed functions above do not cover.
        """
        result = self._payload(await self.executor.query_database(query, max_rows=max_rows))
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
        names = self._payload(await self.executor.lookup_phenotype_names(code_list))["names"]
        return _frame(
            [[code, names.get(code)] for code in code_list],
            columns=["phenotype", "name"],
        )

    async def get_dataset_display_names(self) -> pl.DataFrame:
        """Display-name overrides keyed by the raw `dataset` column value."""
        mapping = self._payload(await self.executor.get_dataset_display_names())["display_names"]
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
        payload = self._payload(await self.executor.normalize_gene_symbols(cleaned))
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
        return self._payload(await self.executor.get_database_schema(table))["schema"]

    async def resources(self) -> dict[str, Any]:
        """Catalog of available data resources, grouped by data product."""
        return self._payload(await self.executor.get_available_resources())["resources"]

    async def datasets(
        self, resource: str | None = None, include_stats: bool = True
    ) -> dict[str, Any]:
        """Dataset catalog with descriptions and aggregate stats."""
        return self._payload(
            await self.executor.list_datasets(resource=resource, include_stats=include_stats)
        )["datasets"]
