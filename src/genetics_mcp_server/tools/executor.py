"""Tool executor - handles HTTP calls to genetics API and external services."""

import base64
import io
import logging
import os
import traceback
from typing import Any
from urllib.parse import quote

import httpx
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for server use
import matplotlib.pyplot as plt

from genetics_mcp_server.tools.phewas_categories import (
    categorize_phenotype,
    get_category_color,
)

logger = logging.getLogger(__name__)

# generic error message returned to clients
INTERNAL_ERROR_MSG = "Internal server error. Check server logs for details."

# variant classifications for counting coding and loss-of-function variants
CODING_VARIANTS = {
    "missense_variant",
    "frameshift_variant",
    "inframe_insertion",
    "inframe_deletion",
    "transcript_ablation",
    "stop_gained",
    "stop_lost",
    "start_lost",
    "splice_acceptor_variant",
    "splice_donor_variant",
    "incomplete_terminal_codon_variant",
    "protein_altering_variant",
    "coding_sequence_variant",
}

LOF_VARIANTS = {
    "frameshift_variant",
    "stop_gained",
    "stop_lost",
    "start_lost",
    "splice_acceptor_variant",
    "splice_donor_variant",
    "transcript_ablation",
}


class ToolExecutor:
    """Executes MCP tools by making HTTP calls to the genetics API."""

    def __init__(
        self,
        api_base_url: str | None = None,
        public_api_url: str | None = None,
        bigquery_api_url: str | None = None,
    ):
        self.base_url = api_base_url or os.environ.get(
            "GENETICS_API_URL", "http://0.0.0.0:2000/api"
        )
        # public URL for download links shown to users
        self.public_url = public_api_url or os.environ.get(
            "GENETICS_PUBLIC_API_URL", self.base_url
        )
        # BigQuery API URL for direct SQL queries
        self.bigquery_url = bigquery_api_url or os.environ.get("BIGQUERY_API_URL")
        # authenticate to results-api with shared secret if configured
        api_secret = os.environ.get("INTERNAL_API_SECRET", "")
        headers = {"Authorization": f"Bearer {api_secret}"} if api_secret else {}
        self.client = httpx.AsyncClient(timeout=30.0, headers=headers)

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    # -------------------------------------------------------------------------
    # BigQuery Tools
    # -------------------------------------------------------------------------

    async def query_bigquery(
        self,
        sql: str,
        max_rows: int = 1000,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Execute a SQL query against the genetics BigQuery database."""
        if not self.bigquery_url:
            return {
                "success": False,
                "error": "BigQuery API URL not configured. Set BIGQUERY_API_URL environment variable.",
            }

        try:
            resp = await self.client.post(
                f"{self.bigquery_url}/query",
                json={"sql": sql, "max_rows": max_rows, "dry_run": dry_run},
                timeout=60.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "success": True,
                    "sql": sql,
                    "columns": data.get("columns", []),
                    "rows": data.get("rows", []),
                    "total_rows": data.get("total_rows", 0),
                    "bytes_processed": data.get("bytes_processed", 0),
                    "truncated": data.get("truncated", False),
                }
            return {
                "success": False,
                "error": f"HTTP {resp.status_code}: {resp.text}",
            }
        except Exception as e:
            logger.error(f"Error in query_bigquery: {e}\n{traceback.format_exc()}")
            return {"success": False, "error": INTERNAL_ERROR_MSG}

    async def get_bigquery_schema(self) -> dict[str, Any]:
        """Get schema information for the BigQuery tables."""
        if not self.bigquery_url:
            return {
                "success": False,
                "error": "BigQuery API URL not configured. Set BIGQUERY_API_URL environment variable.",
            }

        try:
            resp = await self.client.get(f"{self.bigquery_url}/schema")
            if resp.status_code == 200:
                return {"success": True, "schema": resp.json()}
            return {
                "success": False,
                "error": f"HTTP {resp.status_code}: {resp.text}",
            }
        except Exception as e:
            logger.error(f"Error in get_bigquery_schema: {e}\n{traceback.format_exc()}")
            return {"success": False, "error": INTERNAL_ERROR_MSG}

    # -------------------------------------------------------------------------
    # Search Tools
    # -------------------------------------------------------------------------

    async def search_phenotypes(self, query: str, limit: int = 100) -> dict[str, Any]:
        """Search phenotypes via autocomplete endpoint. Supports comma-separated trait names."""
        normalized_query = ",".join(term.strip() for term in query.split(","))
        resp = await self.client.get(
            f"{self.base_url}/v1/search",
            params={
                "q": normalized_query,
                "types": "phenotypes",
                "limit": limit,
                "format": "json",
            },
        )
        if resp.status_code == 200:
            return {"success": True, "results": resp.json()}
        return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}

    async def search_genes(self, query: str, limit: int = 10) -> dict[str, Any]:
        """Search genes via autocomplete endpoint. Supports comma-separated gene names."""
        normalized_query = ",".join(term.strip() for term in query.split(","))
        resp = await self.client.get(
            f"{self.base_url}/v1/search",
            params={"q": normalized_query, "types": "genes", "limit": limit, "format": "json"},
        )
        if resp.status_code == 200:
            return {"success": True, "results": resp.json()}
        return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}

    async def lookup_variants_by_rsid(self, rsids: str) -> dict[str, Any]:
        """Convert rsIDs to variant IDs (chr:pos:ref:alt format)."""
        if not rsids or not rsids.strip():
            return {"success": False, "error": "No rsIDs provided"}

        normalized_rsids = ",".join(term.strip() for term in rsids.split(","))
        resp = await self.client.get(
            f"{self.base_url}/v1/rsid/variants",
            params={"rsids": normalized_rsids},
        )
        if resp.status_code == 200:
            return {"success": True, "variants": resp.json()}
        return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}

    async def lookup_phenotype_names(self, codes: list[str]) -> dict[str, Any]:
        """Batch lookup phenotype codes to names."""
        if not codes:
            return {"success": False, "error": "No phenotype codes provided"}

        resp = await self.client.get(f"{self.base_url}/v1/trait_name_mapping")
        if resp.status_code == 200:
            full_mapping = resp.json()
            result = {
                code: full_mapping.get(code, f"Unknown: {code}") for code in codes
            }
            return {"success": True, "names": result}
        return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}

    # -------------------------------------------------------------------------
    # Credible Sets Tools
    # -------------------------------------------------------------------------

    async def get_credible_sets_by_gene(
        self,
        gene: str,
        window: int = 100000,
        resource: str | None = None,
        data_types: str | None = None,
        summarize: bool = True,
    ) -> dict[str, Any]:
        """Get credible sets for a gene region."""
        try:
            params: dict[str, Any] = {"window": window}
            if resource:
                params["resources"] = resource
            if data_types:
                params["data_types"] = data_types

            if summarize:
                params["format"] = "tsv"
                resp = await self.client.get(
                    f"{self.base_url}/v1/credible_sets_by_gene/{gene}", params=params
                )
                if resp.status_code == 200:
                    summary = self._summarize_credible_sets_simple(resp.text)
                    return {"success": True, "gene": gene, **summary}
                return {
                    "success": False,
                    "error": f"HTTP {resp.status_code}: {resp.text}",
                }
            else:
                params["format"] = "json"
                resp = await self.client.get(
                    f"{self.base_url}/v1/credible_sets_by_gene/{gene}", params=params
                )
                if resp.status_code == 200:
                    results = resp.json()
                    results = self._prioritize_variants(results)
                    return {
                        "success": True,
                        "gene": gene,
                        "total_count": len(results),
                        "results": results,
                    }
                return {
                    "success": False,
                    "error": f"HTTP {resp.status_code}: {resp.text}",
                }
        except Exception as e:
            logger.error(
                f"Error in get_credible_sets_by_gene({gene}): {e}\n{traceback.format_exc()}"
            )
            return {"success": False, "error": INTERNAL_ERROR_MSG}

    async def get_credible_sets_by_variant(
        self,
        variant: str,
        resource: str | None = None,
        data_types: str | None = None,
        summarize: bool = True,
    ) -> dict[str, Any]:
        """Get credible sets containing a specific variant."""
        try:
            params: dict[str, Any] = {}
            if resource:
                params["resources"] = resource
            if data_types:
                params["data_types"] = data_types

            if summarize:
                params["format"] = "tsv"
                resp = await self.client.get(
                    f"{self.base_url}/v1/credible_sets_by_variant/{variant}",
                    params=params,
                )
                if resp.status_code == 200:
                    summary = self._summarize_credible_sets_simple(resp.text)
                    return {"success": True, "variant": variant, **summary}
                return {
                    "success": False,
                    "error": f"HTTP {resp.status_code}: {resp.text}",
                }
            else:
                params["format"] = "json"
                resp = await self.client.get(
                    f"{self.base_url}/v1/credible_sets_by_variant/{variant}",
                    params=params,
                )
                if resp.status_code == 200:
                    results = resp.json()
                    results = self._prioritize_variants(results)
                    return {
                        "success": True,
                        "variant": variant,
                        "total_count": len(results),
                        "results": results,
                    }
                return {
                    "success": False,
                    "error": f"HTTP {resp.status_code}: {resp.text}",
                }
        except Exception as e:
            logger.error(
                f"Error in get_credible_sets_by_variant({variant}): {e}\n{traceback.format_exc()}"
            )
            return {"success": False, "error": INTERNAL_ERROR_MSG}

    async def get_credible_sets_by_phenotype(
        self,
        phenotype: str,
        resource: str = "finngen",
        summarize: bool = True,
    ) -> dict[str, Any]:
        """Get credible sets for a phenotype."""
        try:
            if summarize:
                resp = await self.client.get(
                    f"{self.base_url}/v1/credible_sets_by_phenotype/{resource}/{phenotype}",
                    params={"format": "tsv"},
                )
                if resp.status_code == 200:
                    summary = self._summarize_credible_sets_trait(resp.text)
                    return {"success": True, "phenotype": phenotype, **summary}
                return {
                    "success": False,
                    "error": f"HTTP {resp.status_code}: {resp.text}",
                }
            else:
                resp = await self.client.get(
                    f"{self.base_url}/v1/credible_sets_by_phenotype/{resource}/{phenotype}",
                    params={"format": "json"},
                )
                if resp.status_code == 200:
                    return {
                        "success": True,
                        "phenotype": phenotype,
                        "results": resp.json(),
                    }
                return {
                    "success": False,
                    "error": f"HTTP {resp.status_code}: {resp.text}",
                }
        except Exception as e:
            logger.error(
                f"Error in get_credible_sets_by_phenotype({phenotype}): {e}\n{traceback.format_exc()}"
            )
            return {"success": False, "error": INTERNAL_ERROR_MSG}

    async def get_credible_set_by_id(
        self,
        resource: str,
        phenotype: str,
        credible_set_id: str,
    ) -> dict[str, Any]:
        """Get all variants in a specific credible set."""
        try:
            encoded_cs_id = quote(credible_set_id, safe="")
            resp = await self.client.get(
                f"{self.base_url}/v1/credible_sets_by_id/{resource}/{phenotype}/{encoded_cs_id}",
                params={"format": "json"},
            )
            if resp.status_code == 200:
                variants = resp.json()
                return {
                    "success": True,
                    "resource": resource,
                    "phenotype": phenotype,
                    "credible_set_id": credible_set_id,
                    "n_variants": len(variants),
                    "variants": variants,
                }
            return {
                "success": False,
                "error": f"HTTP {resp.status_code}: {resp.text}",
            }
        except Exception as e:
            logger.error(
                f"Error in get_credible_set_by_id({credible_set_id}): {e}\n{traceback.format_exc()}"
            )
            return {"success": False, "error": INTERNAL_ERROR_MSG}

    async def get_credible_sets_by_qtl_gene(
        self,
        gene: str,
        data_types: str | None = None,
        resource: str | None = None,
        summarize: bool = False,
    ) -> dict[str, Any]:
        """Get QTL credible sets where gene is the molecular trait."""
        try:
            params: dict[str, Any] = {}
            if data_types:
                params["data_types"] = data_types
            if resource:
                params["resources"] = resource

            if summarize:
                params["format"] = "tsv"
                resp = await self.client.get(
                    f"{self.base_url}/v1/credible_sets_by_qtl_gene/{gene}",
                    params=params,
                )
                if resp.status_code == 200:
                    summary = self._summarize_credible_sets_simple(resp.text)
                    return {"success": True, "gene": gene, **summary}
                return {
                    "success": False,
                    "error": f"HTTP {resp.status_code}: {resp.text}",
                }
            else:
                params["format"] = "json"
                resp = await self.client.get(
                    f"{self.base_url}/v1/credible_sets_by_qtl_gene/{gene}",
                    params=params,
                )
                if resp.status_code == 200:
                    return {"success": True, "gene": gene, "results": resp.json()}
                return {
                    "success": False,
                    "error": f"HTTP {resp.status_code}: {resp.text}",
                }
        except Exception as e:
            logger.error(
                f"Error in get_credible_sets_by_qtl_gene({gene}): {e}\n{traceback.format_exc()}"
            )
            return {"success": False, "error": INTERNAL_ERROR_MSG}

    # -------------------------------------------------------------------------
    # Gene Data Tools
    # -------------------------------------------------------------------------

    async def get_gene_expression(self, gene: str) -> dict[str, Any]:
        """Get tissue expression for a gene."""
        resp = await self.client.get(
            f"{self.base_url}/v1/expression_by_gene/{gene}", params={"format": "json"}
        )
        if resp.status_code == 200:
            return {"success": True, "gene": gene, "results": resp.json()}
        return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}

    async def get_gene_disease_associations(self, gene: str) -> dict[str, Any]:
        """Get gene-disease associations."""
        resp = await self.client.get(
            f"{self.base_url}/v1/gene_disease/{gene}", params={"format": "json"}
        )
        if resp.status_code == 200:
            return {"success": True, "gene": gene, "results": resp.json()}
        elif resp.status_code == 404:
            return {
                "success": True,
                "gene": gene,
                "results": [],
                "message": "No Mendelian disease associations found",
            }
        return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}

    async def get_exome_results_by_gene(self, gene: str) -> dict[str, Any]:
        """Get exome sequencing results for a gene."""
        resp = await self.client.get(
            f"{self.base_url}/v1/exome_results_by_gene/{gene}",
            params={"format": "json"},
        )
        if resp.status_code == 200:
            return {"success": True, "gene": gene, "results": resp.json()}
        return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}

    # -------------------------------------------------------------------------
    # LD Tools (FinnGen LD Server)
    # -------------------------------------------------------------------------

    def _parse_variant(self, variant: str) -> tuple[str, int, str, str]:
        """Parse variant ID into components.

        Args:
            variant: Variant ID in chr:pos:ref:alt format

        Returns:
            Tuple of (chromosome, position, ref, alt)

        Raises:
            ValueError: If variant format is invalid
        """
        parts = variant.split(":")
        if len(parts) != 4:
            raise ValueError(
                f"Invalid variant format: {variant}. Expected chr:pos:ref:alt"
            )
        chr_str, pos_str, ref, alt = parts
        try:
            pos = int(pos_str)
        except ValueError:
            raise ValueError(f"Invalid position in variant: {variant}")
        return chr_str, pos, ref, alt

    async def get_ld_between_variants(
        self,
        variant1: str,
        variant2: str,
        r2_threshold: float = 0.1,
        panel: str = "sisu42",
    ) -> dict[str, Any]:
        """Get LD statistics between two specific variants."""
        try:
            try:
                chr1, pos1, _, _ = self._parse_variant(variant1)
                chr2, pos2, _, _ = self._parse_variant(variant2)
            except ValueError as e:
                return {"success": False, "error": str(e)}

            # normalize chromosome format for comparison
            chr1_norm = chr1.replace("chr", "")
            chr2_norm = chr2.replace("chr", "")

            if chr1_norm != chr2_norm:
                return {
                    "success": False,
                    "error": f"Variants must be on same chromosome. Got chr{chr1_norm} and chr{chr2_norm}",
                }

            distance = abs(pos2 - pos1)

            # max 5 Mb distance
            if distance > 5_000_000:
                return {
                    "success": False,
                    "error": f"Variants are too far apart ({distance:,} bp). Maximum allowed distance is 5 Mb.",
                }

            # window = 2 * distance + 1000000 (API bug workaround)
            window = 2 * distance + 1_000_000

            resp = await self.client.get(
                "https://api.finngen.fi/api/ld",
                params={
                    "variant": variant1,
                    "window": window,
                    "panel": panel,
                    "r2_thresh": r2_threshold,
                },
                timeout=30.0,
            )

            if resp.status_code != 200:
                return {
                    "success": False,
                    "error": f"FinnGen LD API error: HTTP {resp.status_code}",
                }

            data = resp.json()
            ld_results = data.get("ld", [])

            # find variant2 in results (could be in variation1 or variation2 field)
            match = None
            for entry in ld_results:
                v1 = entry.get("variation1", "")
                v2 = entry.get("variation2", "")
                if v2 == variant2 or v1 == variant2:
                    match = entry
                    break

            if not match:
                return {
                    "success": True,
                    "variant1": variant1,
                    "variant2": variant2,
                    "in_ld": False,
                    "message": f"No LD found between variants (r2 < {r2_threshold} or variant not in reference panel)",
                }

            return {
                "success": True,
                "variant1": variant1,
                "variant2": variant2,
                "in_ld": True,
                "r2": match.get("r2"),
                "d_prime": match.get("d_prime"),
                "panel": panel,
            }

        except Exception as e:
            logger.error(
                f"Error in get_ld_between_variants({variant1}, {variant2}): {e}\n{traceback.format_exc()}"
            )
            return {"success": False, "error": INTERNAL_ERROR_MSG}

    async def get_variants_in_ld(
        self,
        variant: str,
        window: int = 1_500_000,
        r2_threshold: float = 0.6,
        panel: str = "sisu42",
    ) -> dict[str, Any]:
        """Get all variants in LD with a given variant."""
        try:
            try:
                self._parse_variant(variant)
            except ValueError as e:
                return {"success": False, "error": str(e)}

            resp = await self.client.get(
                "https://api.finngen.fi/api/ld",
                params={
                    "variant": variant,
                    "window": window,
                    "panel": panel,
                    "r2_thresh": r2_threshold,
                },
                timeout=30.0,
            )

            if resp.status_code != 200:
                return {
                    "success": False,
                    "error": f"FinnGen LD API error: HTTP {resp.status_code}",
                }

            data = resp.json()
            ld_results = data.get("ld", [])

            # extract variants in LD (the "other" variant from each pair)
            variants_in_ld = []
            for entry in ld_results:
                v1 = entry.get("variation1", "")
                v2 = entry.get("variation2", "")
                other_variant = v2 if v1 == variant else v1
                variants_in_ld.append({
                    "variant": other_variant,
                    "r2": entry.get("r2"),
                    "d_prime": entry.get("d_prime"),
                })

            # sort by r2 descending
            variants_in_ld.sort(key=lambda x: x.get("r2") or 0, reverse=True)

            return {
                "success": True,
                "query_variant": variant,
                "window": window,
                "r2_threshold": r2_threshold,
                "panel": panel,
                "n_variants": len(variants_in_ld),
                "variants": variants_in_ld,
            }

        except Exception as e:
            logger.error(
                f"Error in get_variants_in_ld({variant}): {e}\n{traceback.format_exc()}"
            )
            return {"success": False, "error": INTERNAL_ERROR_MSG}

    # -------------------------------------------------------------------------
    # Colocalization and Reports
    # -------------------------------------------------------------------------

    async def get_colocalization(self, variant: str) -> dict[str, Any]:
        """Get colocalization results for a variant."""
        resp = await self.client.get(
            f"{self.base_url}/v1/colocalization_by_variant/{variant}",
            params={"format": "json"},
        )
        if resp.status_code == 200:
            return {"success": True, "variant": variant, "results": resp.json()}
        return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}

    async def get_phenotype_report(self, resource: str, phenotype_code: str) -> dict[str, Any]:
        """Get phenotype markdown report."""
        resp = await self.client.get(
            f"{self.base_url}/v1/phenotype/{resource}/{phenotype_code}/markdown",
        )
        if resp.status_code == 200:
            return {
                "success": True,
                "resource": resource,
                "phenotype_code": phenotype_code,
                "content": resp.text,
            }
        elif resp.status_code == 404:
            return {
                "success": False,
                "resource": resource,
                "phenotype_code": phenotype_code,
                "error": f"No report found for phenotype: {phenotype_code} in resource: {resource}",
            }
        return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}

    async def get_available_resources(self) -> dict[str, Any]:
        """Get catalog of available data resources."""
        resp = await self.client.get(f"{self.base_url}/v1/resources")
        if resp.status_code == 200:
            return {"success": True, "resources": resp.json()}
        return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}

    # -------------------------------------------------------------------------
    # Visualization Tools
    # -------------------------------------------------------------------------

    async def create_phewas_plot(
        self,
        variant: str,
        resource: str | None = None,
        significance_threshold: float = 7.3,
        min_mlog10p: float = 2.0,
    ) -> dict[str, Any]:
        """Create a PheWAS plot for a variant showing phenotype associations."""
        try:
            # fetch associations using existing method
            data = await self.get_credible_sets_by_variant(
                variant, resource=resource, summarize=False
            )
            if not data["success"]:
                return data

            # filter to GWAS associations above threshold
            results = [
                r for r in data["results"]
                if r.get("data_type") == "GWAS"
                and (r.get("mlog10p") or 0) >= min_mlog10p
            ]

            if not results:
                return {
                    "success": False,
                    "error": f"No GWAS associations found for {variant} with -log10(p) >= {min_mlog10p}",
                }

            # get phenotype names for categorization
            phenotype_codes = list(set(r.get("trait", "") for r in results if r.get("trait")))
            names_data = await self.lookup_phenotype_names(phenotype_codes)
            code_to_name = names_data.get("names", {}) if names_data.get("success") else {}

            # categorize each phenotype
            for r in results:
                code = r.get("trait", "")
                name = code_to_name.get(code, "")
                r["category"] = categorize_phenotype(code, name)
                r["phenotype_name"] = name if name and not name.startswith("Unknown:") else code

            # sort by category, then by mlog10p within category
            results.sort(key=lambda x: (x["category"], -(x.get("mlog10p") or 0)))

            # generate matplotlib figure
            fig = self._create_phewas_figure(results, variant, significance_threshold)

            # encode as base64 PNG
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", dpi=100, bbox_inches="tight", facecolor="white")
            buffer.seek(0)
            base64_png = base64.b64encode(buffer.read()).decode("utf-8")
            plt.close(fig)

            # compute summary stats
            n_significant = sum(1 for r in results if (r.get("mlog10p") or 0) >= significance_threshold)
            categories_found = sorted(set(r["category"] for r in results))

            return {
                "success": True,
                "variant": variant,
                "n_associations": len(results),
                "n_significant": n_significant,
                "categories": categories_found,
                "image_base64": base64_png,
                "image_format": "png",
            }

        except Exception as e:
            logger.error(f"Error in create_phewas_plot({variant}): {e}\n{traceback.format_exc()}")
            return {"success": False, "error": INTERNAL_ERROR_MSG}

    def _create_phewas_figure(
        self,
        results: list[dict],
        variant: str,
        significance_threshold: float,
    ) -> plt.Figure:
        """Create a matplotlib PheWAS figure."""
        fig, ax = plt.subplots(figsize=(14, 6))

        # assign x-positions - group by category
        categories = []
        x_positions = []
        y_values = []
        colors = []
        labels = []

        current_x = 0
        last_category = None
        category_ranges = {}  # track x-range for each category

        for r in results:
            category = r["category"]

            # add gap between categories
            if last_category is not None and category != last_category:
                current_x += 2

            if category not in category_ranges:
                category_ranges[category] = {"start": current_x, "end": current_x}

            x_positions.append(current_x)
            y_values.append(r.get("mlog10p") or 0)
            colors.append(get_category_color(category))
            labels.append(r["phenotype_name"])
            categories.append(category)

            category_ranges[category]["end"] = current_x
            last_category = category
            current_x += 1

        # scatter plot
        ax.scatter(x_positions, y_values, c=colors, s=50, alpha=0.7, edgecolors="none")

        # significance threshold line
        ax.axhline(
            y=significance_threshold,
            color="red",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            label="Genome-wide significance (p=5e-8)",
        )

        # category labels at bottom
        for category, ranges in category_ranges.items():
            mid_x = (ranges["start"] + ranges["end"]) / 2
            ax.text(
                mid_x,
                -0.5,
                category,
                ha="center",
                va="top",
                fontsize=8,
                rotation=45,
                color=get_category_color(category),
                fontweight="bold",
            )

        # annotate top significant hits
        significant_results = [
            (x, y, label) for x, y, label in zip(x_positions, y_values, labels)
            if y >= significance_threshold
        ]
        # sort by y descending and take top 10
        significant_results.sort(key=lambda t: -t[1])
        for x, y, label in significant_results[:10]:
            # truncate long labels
            short_label = label[:30] + "..." if len(label) > 30 else label
            ax.annotate(
                short_label,
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7,
                alpha=0.8,
            )

        # formatting
        ax.set_xlabel("Phenotype Category", fontsize=10)
        ax.set_ylabel("-log10(p-value)", fontsize=10)
        ax.set_title(f"PheWAS Plot for {variant}", fontsize=12, fontweight="bold")

        ax.set_xlim(-1, current_x)
        ax.set_ylim(bottom=0)
        ax.set_xticks([])  # hide x-axis ticks since we're using category labels

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        # legend for significance line
        ax.legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        return fig

    async def get_credible_sets_stats(
        self,
        resource_or_dataset: str,
        trait: str | None = None,
    ) -> dict[str, Any]:
        """Get credible sets statistics for a dataset."""
        try:
            resp = await self.client.get(
                f"{self.base_url}/v1/credible_sets/{resource_or_dataset}/stats",
                params={"format": "json"},
            )
            if resp.status_code != 200:
                return {
                    "success": False,
                    "error": f"HTTP {resp.status_code}: {resp.text}",
                }

            data = resp.json()
            if not data:
                return {
                    "success": True,
                    "resource_or_dataset": resource_or_dataset,
                    "n_traits": 0,
                    "totals": {},
                    "traits": [],
                }

            # filter by trait if specified
            if trait:
                data = [row for row in data if row.get("trait") == trait]

            # compute aggregate totals
            stat_cols = [
                "n_risk_cs", "n_risk_cs_with_coding", "n_risk_cs_with_coding_pip_gt_0_05",
                "n_risk_cs_with_lof", "n_risk_cs_with_lof_pip_gt_0_05",
                "n_protective_cs", "n_protective_cs_with_coding", "n_protective_cs_with_coding_pip_gt_0_05",
                "n_protective_cs_with_lof", "n_protective_cs_with_lof_pip_gt_0_05",
            ]

            totals = {}
            for col in stat_cols:
                totals[col] = sum(row.get(col, 0) or 0 for row in data)

            download_url = f"{self.public_url}/v1/credible_sets/{resource_or_dataset}/stats"

            return {
                "INCLUDE_IN_RESPONSE": f"📥 [Download full data as TSV]({download_url})",
                "success": True,
                "resource_or_dataset": resource_or_dataset,
                "n_traits": len(data),
                "totals": totals,
                "traits": data,
            }
        except Exception as e:
            logger.error(
                f"Error in get_credible_sets_stats({resource_or_dataset}): {e}\n{traceback.format_exc()}"
            )
            return {"success": False, "error": INTERNAL_ERROR_MSG}

    async def get_nearest_genes(
        self,
        variant: str,
        gene_type: str = "protein_coding",
        n: int = 3,
        max_distance: int = 1000000,
        gencode_version: str | None = None,
        return_hgnc_symbol_if_only_ensg: bool = False,
    ) -> dict[str, Any]:
        """Get genes nearest to a variant."""
        try:
            params: dict[str, Any] = {
                "format": "json",
                "gene_type": gene_type,
                "n": n,
                "max_distance": max_distance,
                "return_hgnc_symbol_if_only_ensg": return_hgnc_symbol_if_only_ensg,
            }
            if gencode_version:
                params["gencode_version"] = gencode_version

            resp = await self.client.get(
                f"{self.base_url}/v1/nearest_genes/{variant}",
                params=params,
            )
            if resp.status_code == 200:
                results = resp.json()
                return {
                    "success": True,
                    "variant": variant,
                    "genes": results,
                }
            return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}
        except Exception as e:
            logger.error(
                f"Error in get_nearest_genes({variant}): {e}\n{traceback.format_exc()}"
            )
            return {"success": False, "error": INTERNAL_ERROR_MSG}

    async def get_genes_in_region(
        self,
        chr: str,
        start: int,
        end: int,
        gene_type: str = "protein_coding",
        gencode_version: str | None = None,
    ) -> dict[str, Any]:
        """Get all genes in a genomic region."""
        try:
            params: dict[str, Any] = {
                "format": "json",
                "gene_type": gene_type,
            }
            if gencode_version:
                params["gencode_version"] = gencode_version

            resp = await self.client.get(
                f"{self.base_url}/v1/genes_in_region/{chr}/{start}/{end}",
                params=params,
            )
            if resp.status_code == 200:
                results = resp.json()
                return {
                    "success": True,
                    "region": f"{chr}:{start}-{end}",
                    "genes": results,
                }
            return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}
        except Exception as e:
            logger.error(
                f"Error in get_genes_in_region({chr}:{start}-{end}): {e}\n{traceback.format_exc()}"
            )
            return {"success": False, "error": INTERNAL_ERROR_MSG}

    # -------------------------------------------------------------------------
    # External Search Tools
    # -------------------------------------------------------------------------

    async def search_scientific_literature(
        self,
        query: str,
        max_results: int = 10,
        include_preprints: bool = True,
        date_range: str | None = None,
        backend: str | None = None,
    ) -> dict[str, Any]:
        """Search scientific literature via Europe PMC or Perplexity."""
        selected_backend = (
            backend
            or os.environ.get("LITERATURE_SEARCH_BACKEND", "europepmc")
        ).lower()

        if selected_backend == "perplexity":
            perplexity_api_key = os.environ.get("PERPLEXITY_API_KEY")
            if not perplexity_api_key:
                logger.error("Perplexity backend requested but PERPLEXITY_API_KEY not configured")
                return {
                    "success": False,
                    "error": "Literature search with Perplexity is currently unavailable (API key not configured)",
                }
            try:
                return await self._search_perplexity_literature(
                    query, max_results, perplexity_api_key, include_preprints, date_range
                )
            except Exception as e:
                logger.error(f"Perplexity search failed: {e}")
                return {
                    "success": False,
                    "error": f"Literature search with Perplexity is currently unavailable: {e}",
                }

        # europepmc backend
        return await self._search_europepmc_literature(
            query, max_results, include_preprints, date_range
        )

    async def _search_europepmc_literature(
        self,
        query: str,
        max_results: int,
        include_preprints: bool,
        date_range: str | None,
    ) -> dict[str, Any]:
        """Search scientific literature via Europe PMC."""
        epmc_query = query

        if not include_preprints:
            epmc_query += " (SRC:MED OR SRC:PMC)"

        if date_range:
            epmc_query += self._build_date_filter(date_range)

        url = (
            f"https://www.ebi.ac.uk/europepmc/webservices/rest/search"
            f"?query={quote(epmc_query)}"
            f"&format=json"
            f"&pageSize={min(max_results, 25)}"
            f"&resultType=core"
        )

        try:
            resp = await self.client.get(url, timeout=15.0)
            if resp.status_code == 200:
                data = resp.json()
                results = self._format_literature_results(
                    data.get("resultList", {}).get("result", [])
                )
                return {
                    "success": True,
                    "query": query,
                    "total_found": data.get("hitCount", 0),
                    "returned": len(results),
                    "results": results,
                    "source": "europepmc",
                }
            return {
                "success": False,
                "error": f"Europe PMC error: HTTP {resp.status_code}",
            }
        except Exception as e:
            logger.error(f"Europe PMC search error: {e}")
            return {"success": False, "error": f"Literature search failed: {str(e)}"}

    async def _search_perplexity_literature(
        self,
        query: str,
        max_results: int,
        api_key: str,
        include_preprints: bool,
        date_range: str | None,
    ) -> dict[str, Any]:
        """Search scientific literature using Perplexity Sonar API."""
        enhanced_query = f"Find scientific research papers about: {query}"

        # domain filter for scientific sources
        domains = [
            "pubmed.ncbi.nlm.nih.gov",
            "ncbi.nlm.nih.gov",
            "doi.org",
            "nature.com",
            "science.org",
            "cell.com",
            "nejm.org",
            "thelancet.com",
            "pnas.org",
            "jci.org",
        ]
        if include_preprints:
            domains.extend(["biorxiv.org", "medrxiv.org"])

        web_search_options: dict[str, Any] = {
            "search_domain_filter": domains,
        }

        # date filtering
        if date_range == "last_year":
            web_search_options["search_recency_filter"] = "year"
        elif date_range == "last_5_years":
            # perplexity doesn't have exact 5-year filter, skip
            pass

        payload = {
            "model": "sonar",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a scientific literature search assistant. "
                        "Return information about relevant research papers including "
                        "title, authors, journal, year, and key findings. "
                        "Include DOI or PMID when available."
                    ),
                },
                {"role": "user", "content": enhanced_query},
            ],
            "web_search_options": web_search_options,
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        resp = await self.client.post(
            "https://api.perplexity.ai/chat/completions",
            json=payload,
            headers=headers,
            timeout=30.0,
        )

        if resp.status_code == 200:
            data = resp.json()
            return self._format_perplexity_literature_results(data, query, max_results)

        raise Exception(f"Perplexity API error: HTTP {resp.status_code}")

    def _format_perplexity_literature_results(
        self,
        data: dict,
        query: str,
        max_results: int,
    ) -> dict[str, Any]:
        """Format Perplexity response to match Europe PMC output structure."""
        import re

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        citations = data.get("citations", [])

        results = []
        for i, url in enumerate(citations[:max_results]):
            # extract DOI/PMID from URL if possible
            doi = None
            pmid = None
            if "doi.org/" in url:
                doi = url.split("doi.org/")[-1]
            if "pubmed.ncbi.nlm.nih.gov/" in url:
                match = re.search(r"/(\d+)", url)
                if match:
                    pmid = match.group(1)

            is_preprint = "biorxiv.org" in url or "medrxiv.org" in url

            results.append({
                "title": "",
                "authors": "",
                "journal": "",
                "year": "",
                "abstract": "",
                "doi": doi,
                "pmid": pmid,
                "source": "perplexity",
                "is_preprint": is_preprint,
                "url": url,
            })

        return {
            "success": True,
            "query": query,
            "total_found": len(citations),
            "returned": len(results),
            "results": results,
            "summary": content,
            "source": "perplexity",
        }

    async def web_search(
        self,
        query: str,
        max_results: int = 5,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
    ) -> dict[str, Any]:
        """Search web using Tavily (if configured) or DuckDuckGo."""
        tavily_api_key = os.environ.get("TAVILY_API_KEY")
        if tavily_api_key:
            try:
                return await self._search_tavily(
                    query,
                    min(max_results, 10),
                    tavily_api_key,
                    include_domains,
                    exclude_domains,
                )
            except Exception as e:
                logger.warning(f"Tavily search failed, falling back to DuckDuckGo: {e}")

        return await self._search_duckduckgo(query, min(max_results, 10))

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _prioritize_variants(self, results: list) -> list:
        """Sort results to show most informative variants first."""
        high_priority = {
            "missense_variant",
            "frameshift_variant",
            "stop_gained",
            "stop_lost",
            "start_lost",
            "splice_acceptor_variant",
            "splice_donor_variant",
            "splice_region_variant",
            "inframe_insertion",
            "inframe_deletion",
        }
        medium_priority = {
            "synonymous_variant",
            "5_prime_UTR_variant",
            "3_prime_UTR_variant",
        }

        def sort_key(item):
            consequence = item.get("most_severe") or ""
            pip = item.get("pip") or 0
            mlog10p = item.get("mlog10p") or 0

            if consequence in high_priority:
                priority = 0
            elif consequence in medium_priority:
                priority = 1
            else:
                priority = 2
            return (priority, -pip, -mlog10p)

        return sorted(results, key=sort_key)

    def _summarize_credible_sets_trait(self, tsv_data: str) -> dict:
        """Summarize variant-level TSV data into credible set-level summary with coding/LoF counts."""
        import io

        import polars as pl

        df = pl.read_csv(
            io.StringIO(tsv_data),
            separator="\t",
            null_values=["NA"],
            infer_schema_length=None,
        )

        if df.is_empty():
            return {"n_cs": 0, "cs": []}

        gene_counts = (
            df.group_by(["cs_id", "gene_most_severe", "most_severe"])
            .len()
            .group_by(["cs_id", "gene_most_severe"])
            .agg(pl.struct(["most_severe", "len"]).alias("consequence_count"))
            .group_by("cs_id")
            .agg(
                pl.struct(["gene_most_severe", "consequence_count"]).alias(
                    "gene_consequence_counts"
                )
            )
        )

        # get lead variant (max pip, then max mlog10p as tiebreaker) for each cs_id
        # use sort_by().first() inside agg to correctly sort within each group
        lead_variant_cols = [
            "cs_size",
            "chr",
            "pos",
            "ref",
            "alt",
            "mlog10p",
            "beta",
            "se",
            "pip",
            "aaf",
            "most_severe",
            "gene_most_severe",
        ]
        lead_variants = df.group_by("cs_id").agg(
            [
                pl.col(c)
                .sort_by(["pip", "mlog10p"], descending=[True, True], nulls_last=True)
                .first()
                for c in lead_variant_cols
            ]
        )

        # aggregate stats from all variants in each credible set
        cs_stats = df.group_by("cs_id").agg(
            [
                pl.col("aaf").min().alias("min_aaf"),
                pl.col("aaf").max().alias("max_aaf"),
                pl.col("most_severe").is_in(CODING_VARIANTS).sum().alias("n_coding"),
                pl.col("most_severe").is_in(LOF_VARIANTS).sum().alias("n_lof"),
            ]
        )

        result = (
            lead_variants.join(cs_stats, on="cs_id")
            .join(gene_counts, on="cs_id", how="left")
            .sort("mlog10p", descending=True, nulls_last=True)
        )

        summaries = []
        for row in result.to_dicts():
            gcc = []
            for gc in row.get("gene_consequence_counts") or []:
                gene = gc.get("gene_most_severe") or "unknown"
                counts = {
                    item["most_severe"]: item["len"]
                    for item in gc.get("consequence_count") or []
                }
                gcc.append({"gene": gene, "consequence_counts": counts})

            summaries.append(
                {
                    "cs_id": row["cs_id"],
                    "cs_size": row["cs_size"],
                    "n_coding": row["n_coding"],
                    "n_lof": row["n_lof"],
                    "min_aaf": row["min_aaf"],
                    "max_aaf": row["max_aaf"],
                    "lead_variant": {
                        "id": f"{row['chr']}:{row['pos']}:{row['ref']}:{row['alt']}",
                        "mlog10p": row["mlog10p"],
                        "beta": row["beta"],
                        "se": row["se"],
                        "pip": row["pip"],
                        "aaf": row["aaf"],
                        "most_severe": row["most_severe"],
                        "gene_most_severe": row["gene_most_severe"],
                    },
                    "gene_consequence_counts": gcc,
                }
            )

        return {"n_cs": len(summaries), "cs": summaries}

    def _summarize_credible_sets_simple(self, tsv_data: str) -> dict:
        """Simpler summary with coding/LoF counts, grouped by data_type."""
        import io

        import polars as pl

        df = pl.read_csv(
            io.StringIO(tsv_data),
            separator="\t",
            null_values=["NA"],
            infer_schema_length=None,
        )

        if df.is_empty():
            return {"n_cs": 0, "cs": {}}

        lead_variant_cols = [
            "cs_size",
            "resource",
            "data_type",
            "cell_type",
            "trait",
            "chr",
            "pos",
            "ref",
            "alt",
            "mlog10p",
            "beta",
            "se",
            "pip",
            "aaf",
            "most_severe",
            "gene_most_severe",
        ]
        lead_variants = df.group_by("cs_id").agg(
            [
                pl.col(c)
                .sort_by(["pip", "mlog10p"], descending=[True, True], nulls_last=True)
                .first()
                for c in lead_variant_cols
            ]
        )

        # aggregate stats from all variants in each credible set
        cs_stats = df.group_by("cs_id").agg(
            [
                pl.col("aaf").min().alias("min_aaf"),
                pl.col("aaf").max().alias("max_aaf"),
                pl.col("most_severe").is_in(CODING_VARIANTS).sum().alias("n_coding"),
                pl.col("most_severe").is_in(LOF_VARIANTS).sum().alias("n_lof"),
            ]
        )

        result = lead_variants.join(cs_stats, on="cs_id").sort(
            "mlog10p", descending=True, nulls_last=True
        )

        # group by data_type
        grouped: dict[str, list] = {}
        for row in result.to_dicts():
            data_type = row["data_type"] or "unknown"
            if data_type not in grouped:
                grouped[data_type] = []
            grouped[data_type].append(
                {
                    "cs_id": row["cs_id"],
                    "cs_size": row["cs_size"],
                    "resource": row["resource"],
                    "cell_type": row["cell_type"],
                    "trait": row["trait"],
                    "n_coding": row["n_coding"],
                    "n_lof": row["n_lof"],
                    "min_aaf": row["min_aaf"],
                    "max_aaf": row["max_aaf"],
                    "lead_variant": {
                        "id": f"{row['chr']}:{row['pos']}:{row['ref']}:{row['alt']}",
                        "mlog10p": row["mlog10p"],
                        "beta": row["beta"],
                        "se": row["se"],
                        "pip": row["pip"],
                        "aaf": row["aaf"],
                        "most_severe": row["most_severe"],
                        "gene_most_severe": row["gene_most_severe"],
                    },
                }
            )

        total_cs = sum(len(cs_list) for cs_list in grouped.values())
        return {"n_cs": total_cs, "cs": grouped}

    def _build_date_filter(self, date_range: str) -> str:
        """Build Europe PMC date filter clause."""
        from datetime import datetime, timedelta

        today = datetime.now()

        if date_range == "last_year":
            start = (today - timedelta(days=365)).strftime("%Y-%m-%d")
            return f" AND (FIRST_PDATE:[{start} TO {today.strftime('%Y-%m-%d')}])"
        elif date_range == "last_5_years":
            start = (today - timedelta(days=5 * 365)).strftime("%Y-%m-%d")
            return f" AND (FIRST_PDATE:[{start} TO {today.strftime('%Y-%m-%d')}])"
        elif "-" in date_range:
            years = date_range.split("-")
            if len(years) == 2:
                return f" AND (PUB_YEAR:[{years[0]} TO {years[1]}])"
        return ""

    def _format_literature_results(self, results: list) -> list:
        """Format Europe PMC results for LLM consumption."""
        import html
        import re

        def strip_html(text: str) -> str:
            if not text:
                return ""
            text = html.unescape(text)
            return re.sub(r"<[^>]+>", "", text)

        formatted = []
        for paper in results:
            pmid = paper.get("pmid")
            doi = paper.get("doi")
            source = paper.get("source", "")

            if pmid:
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            elif doi:
                url = f"https://doi.org/{doi}"
            else:
                url = None

            formatted.append(
                {
                    "title": strip_html(paper.get("title", "")),
                    "authors": paper.get("authorString", ""),
                    "journal": paper.get("journalTitle", "")
                    or paper.get("bookOrReportDetails", {}).get("publisher", ""),
                    "year": paper.get("pubYear", ""),
                    "abstract": strip_html(paper.get("abstractText", "") or "")[:1500],
                    "doi": doi,
                    "pmid": pmid,
                    "source": source,
                    "is_preprint": source == "PPR",
                    "url": url,
                }
            )
        return formatted

    async def _search_tavily(
        self,
        query: str,
        max_results: int,
        api_key: str,
        include_domains: list[str] | None,
        exclude_domains: list[str] | None,
    ) -> dict[str, Any]:
        """Search using Tavily API."""
        payload: dict[str, Any] = {
            "api_key": api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": max_results,
            "include_answer": True,
        }

        if include_domains:
            payload["include_domains"] = include_domains
        if exclude_domains:
            payload["exclude_domains"] = exclude_domains

        resp = await self.client.post(
            "https://api.tavily.com/search", json=payload, timeout=20.0
        )

        if resp.status_code == 200:
            data = resp.json()
            results = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": (r.get("content", "") or "")[:500],
                    "score": r.get("score", 0),
                }
                for r in data.get("results", [])
            ]
            return {
                "success": True,
                "query": query,
                "source": "tavily",
                "answer": data.get("answer"),
                "results": results,
            }

        raise Exception(f"Tavily API error: HTTP {resp.status_code}")

    async def _search_duckduckgo(self, query: str, max_results: int) -> dict[str, Any]:
        """Search using DuckDuckGo (free fallback)."""
        import asyncio

        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        def sync_search():
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=max_results))

        try:
            results = await asyncio.to_thread(sync_search)
            formatted = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "content": (r.get("body", "") or "")[:500],
                }
                for r in results
            ]
            return {
                "success": True,
                "query": query,
                "source": "duckduckgo",
                "results": formatted,
            }
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return {"success": False, "error": f"Web search failed: {str(e)}"}
