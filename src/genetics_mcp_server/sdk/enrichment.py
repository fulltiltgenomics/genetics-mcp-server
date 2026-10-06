"""Gene set enrichment: which gene sets a list of genes over-represents.

    import genetics
    res = genetics.enrichment.ora(my_genes, gene_sets, universe=background_genes)
    res["results"]     # one row per tested set: size, overlap, expected, p, BH q, overlap genes
    res["parameters"]  # the universe convention, |U|, the list size and size filter used

Over-representation analysis (ORA) asks, for each gene set, whether more of the list's genes
fall in it than chance would put there. Validated against R (`phyper`, `fgsea::fora` 1.32.4)
on the fixtures in the package's tests.

THE UNIVERSE IS THE ANALYSIS. The p-value is the hypergeometric upper tail
P(X >= k), X ~ Hypergeom(|U|, |set ∩ U|, |list ∩ U|): drawing the list's genes from a
background U, how often would a set catch k or more? Everything outside U is dropped from
the list and from every set before anything is counted, and the size filter applies to the
sets' sizes AFTER that restriction. Getting U wrong moves p-values by far more than any
numerical detail: a list of protein-coding genes tested against all ~60k genes looks
enriched for everything protein-coding.

- `universe` should be the genes that COULD have made the list — every gene tested in the
  study, or every gene near a variant that was tested — not the genome.
- `universe_mode="annotated"` (default) further intersects U with the genes annotated to any
  set in the collection, as clusterProfiler and most published ORA do: an unannotated gene
  cannot fall in a set, so counting it in U only dilutes. `"given"` uses U exactly as
  supplied, as `fgsea::fora` does. The two differ by tens of percent in p on realistic
  inputs; say which one was used.
- Without `universe`, U is every gene annotated to any set, under either mode.

Returns a dict of a polars DataFrame and the parameters used, so a defaulted analysis can be
restated exactly.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from itertools import chain
from typing import Any

import numpy as np
import polars as pl
from scipy import sparse
from scipy.stats import hypergeom

from genetics_mcp_server.sdk.errors import GeneticsUsageError

__all__ = ["ora"]

_UNIVERSE_MODES = ("annotated", "given")


def _gene_list(value: Any, what: str) -> list[str]:
    """Distinct gene ids in first-seen order; refuses a bare string, which would iterate as
    characters."""
    if isinstance(value, str) or not isinstance(value, Iterable):
        raise GeneticsUsageError(f"{what} must be a list of gene ids, got {type(value).__name__}")
    if isinstance(value, pl.Series):
        value = value.to_list()
    out = list(dict.fromkeys(value))
    # refused rather than dropped: a silent drop would change |U| or the list size unseen
    if any(g is None for g in out):
        raise GeneticsUsageError(f"{what} holds a null gene id")
    for g in out:
        if not isinstance(g, str):
            raise GeneticsUsageError(
                f"{what} must hold gene ids as strings, got {type(g).__name__} ({g!r})"
            )
    return out


def _flatten(gene_sets: Mapping[str, Iterable[str]]) -> tuple[list[str], pl.DataFrame]:
    """The collection as set ids and one (row, gene) pair per membership, row indexing the
    ids."""
    if not isinstance(gene_sets, Mapping):
        raise GeneticsUsageError(
            f"gene_sets must be a mapping of set id -> gene ids, got {type(gene_sets).__name__}"
        )
    set_ids, members = [], []
    for s, m in gene_sets.items():
        if isinstance(m, str) or not isinstance(m, Iterable):
            raise GeneticsUsageError(f"gene set {s!r} must be a list of gene ids")
        set_ids.append(str(s))
        members.append(m if isinstance(m, (list, tuple)) else list(m))
    if any(g is None for m in members for g in m):
        raise GeneticsUsageError("gene sets hold a null gene id")
    lengths = np.fromiter((len(m) for m in members), dtype=np.int64, count=len(members))
    try:
        genes = pl.Series(list(chain.from_iterable(members)), dtype=pl.Utf8, strict=True)
    except (TypeError, pl.exceptions.PolarsError) as exc:
        raise GeneticsUsageError("gene sets must hold gene ids as strings") from exc
    rows = np.repeat(np.arange(len(members), dtype=np.int64), lengths)
    return set_ids, pl.DataFrame({"row": rows, "gene": genes})


def _membership(flat: pl.DataFrame, n_sets: int, index: dict[str, int]) -> sparse.csr_matrix:
    """Sets x genes 0/1 matrix over the genes in `index`; members outside it are dropped and
    repeated members count once."""
    universe = pl.DataFrame({
        "gene": pl.Series(list(index), dtype=pl.Utf8),
        "col": np.fromiter(index.values(), dtype=np.int64, count=len(index)),
    })
    hit = flat.join(universe, on="gene", how="inner")
    rows, cols = hit["row"].to_numpy(), hit["col"].to_numpy()
    # the COO -> CSR conversion sums repeated (set, gene) pairs; resetting data dedupes them
    m = sparse.csr_matrix(
        (np.ones(rows.size, dtype=np.int32), (rows, cols)), shape=(n_sets, len(index))
    )
    m.data[:] = 1
    return m


def _bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, as R's p.adjust(method = "BH")."""
    n = p.size
    if n == 0:
        return p.copy()
    order = np.argsort(p)[::-1]
    ranked = p[order] * n / np.arange(n, 0, -1)
    q = np.minimum(1.0, np.minimum.accumulate(ranked))
    out = np.empty(n)
    out[order] = q
    return out


def ora(
    genes: Any,
    gene_sets: Mapping[str, Iterable[str]],
    universe: Any = None,
    *,
    universe_mode: str = "annotated",
    min_size: int = 5,
    max_size: int = 300,
) -> dict[str, Any]:
    """Over-representation of `genes` in each gene set: hypergeometric upper tail and BH FDR.

    `genes` — the list to test (e.g. nearest genes to lead variants, collapsed to one per
    locus), as gene ids. Duplicates count once; genes outside the universe are dropped and
    counted in `parameters`.
    `gene_sets` — a mapping of set id -> gene ids. The ids must be in the same namespace as
    `genes` and `universe` (all Ensembl, or all symbols); nothing is translated.
    `universe` — the background the list was drawn from. Strongly recommended; see the module
    docstring. Default: every gene annotated to any set.
    `universe_mode` — `"annotated"` (default) intersects `universe` with the genes annotated
    to any set; `"given"` uses it as supplied.
    `min_size`, `max_size` — only sets whose size after restriction to the universe lies in
    `[min_size, max_size]` are tested, and the BH correction runs over exactly those.

    Returns `{"results", "parameters"}`. `results` has one row per tested set, ordered by p:
    `set_id`, `size` (|set ∩ U|), `overlap` (k), `expected` (list size · size / |U|),
    `fold_enrichment` (overlap / expected), `pval` (P(X >= k)), `padj` (BH q) and
    `overlap_genes` (sorted). `parameters` states `universe_mode`, `universe_source`,
    `universe_size` (|U|), `list_size` (|list ∩ U|), `list_dropped` (list genes outside U),
    `min_size`, `max_size`, `sets_given` and `sets_tested`.
    """
    if universe_mode not in _UNIVERSE_MODES:
        raise GeneticsUsageError(
            f"universe_mode must be one of {_UNIVERSE_MODES}, got {universe_mode!r}"
        )
    if int(min_size) != min_size or int(max_size) != max_size or not 1 <= min_size <= max_size:
        raise GeneticsUsageError(
            f"need integers 1 <= min_size <= max_size, got min_size={min_size!r}, "
            f"max_size={max_size!r}"
        )
    query = _gene_list(genes, "genes")

    set_ids, flat = _flatten(gene_sets)
    annotated = flat["gene"].unique(maintain_order=True).to_list()
    if universe is None:
        u_genes, source = annotated, "annotated genes (no universe given)"
    else:
        u_genes = _gene_list(universe, "universe")
        if universe_mode == "annotated":
            ann = set(annotated)
            u_genes = [g for g in u_genes if g in ann]
            source = "given universe ∩ annotated genes"
        else:
            source = "given universe"
    u_index = {g: i for i, g in enumerate(u_genes)}
    m = _membership(flat, len(set_ids), u_index)

    n_universe = len(u_index)
    in_list = np.zeros(n_universe, dtype=bool)
    list_cols = [u_index[g] for g in query if g in u_index]
    in_list[list_cols] = True
    n_list = len(list_cols)
    if n_list == 0:
        raise GeneticsUsageError(
            f"none of the {len(query)} genes is in the universe of {n_universe} genes; check "
            "that the gene ids and the gene sets use the same identifiers"
        )

    sizes = np.diff(m.indptr)
    overlap = m @ in_list.astype(np.int32)
    tested = np.flatnonzero((sizes >= min_size) & (sizes <= max_size))
    size_t, k_t = sizes[tested], overlap[tested]
    # P(X >= 0) is exactly 1, and scipy's sf costs ~15 µs a set at collection scale
    pval = np.ones(tested.size)
    hit = k_t > 0
    pval[hit] = hypergeom.sf(k_t[hit] - 1, n_universe, size_t[hit], n_list)
    expected = n_list * size_t / n_universe

    sub = m[tested].tocoo()
    keep = in_list[sub.col]
    overlaps = (
        pl.DataFrame({
            "row": sub.row[keep].astype(np.int64),
            "gene": pl.Series(u_genes, dtype=pl.Utf8).gather(sub.col[keep]),
        })
        .sort("row", "gene")
        .group_by("row", maintain_order=True)
        .agg(overlap_genes=pl.col("gene"))
    )

    results = (
        pl.DataFrame({
            "row": np.arange(tested.size, dtype=np.int64),
            "set_id": pl.Series([set_ids[i] for i in tested], dtype=pl.Utf8),
            "size": size_t.astype(np.int64),
            "overlap": k_t.astype(np.int64),
            "expected": expected,
            "fold_enrichment": k_t / expected,
            "pval": pval,
            "padj": _bh(pval),
        })
        .join(overlaps, on="row", how="left")
        .with_columns(pl.col("overlap_genes").fill_null(pl.lit([], dtype=pl.List(pl.Utf8))))
        .drop("row")
        .sort(["pval", "set_id"])
    )
    return {
        "results": results,
        "parameters": {
            "universe_mode": universe_mode,
            "universe_source": source,
            "universe_size": n_universe,
            "list_size": n_list,
            "list_dropped": len(query) - n_list,
            "min_size": int(min_size),
            "max_size": int(max_size),
            "sets_given": len(set_ids),
            "sets_tested": int(tested.size),
        },
    }
