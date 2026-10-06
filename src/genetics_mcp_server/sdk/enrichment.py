"""Gene set enrichment: which gene sets a list of genes, or a gene-level statistic, points at.

    import genetics
    res = genetics.enrichment.ora(my_genes, gene_sets, universe=background_genes)
    res["results"]     # one row per tested set: size, overlap, expected, p, BH q, overlap genes
    res["parameters"]  # the universe convention, |U|, the list size and size filter used

Three analyses, each validated against R on the fixtures in the package's tests:

- `ora(genes, gene_sets, universe)` — over-representation of a gene LIST (`phyper`,
  `fgsea::fora` 1.32.4).
- `gsea_preranked(stats, gene_sets)` — preranked GSEA on one statistic per gene, signed so
  that large positive and large negative are both interesting (`fgsea::fgseaSimple`).
- `competitive(gene_stats, gene_sets)` — MAGMA-style competitive test: are the set's gene
  z-scores higher than other genes', optionally adjusting for covariates such as gene
  length (`lm`, one-sided `pt`).

For the two statistic-based tests the universe is the genes that carry a statistic: every
set is restricted to them before the size filter, as for ORA. They are not GWAS gene-level
tests — a z from a burden test or a user's own per-gene analysis is fine; turning GWAS
variant statistics into gene statistics needs an LD reference that is not here.

ORA — the rest of this docstring.

Over-representation analysis (ORA) asks, for each gene set, whether more of the list's genes
fall in it than chance would put there.

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
from scipy.stats import t as t_dist

from genetics_mcp_server.sdk.errors import GeneticsUsageError

__all__ = ["competitive", "gsea_preranked", "ora"]

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


def _check_sizes(min_size: Any, max_size: Any) -> None:
    if int(min_size) != min_size or int(max_size) != max_size or not 1 <= min_size <= max_size:
        raise GeneticsUsageError(
            f"need integers 1 <= min_size <= max_size, got min_size={min_size!r}, "
            f"max_size={max_size!r}"
        )


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
    _check_sizes(min_size, max_size)
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


def _gene_values(value: Any, what: str) -> tuple[list[str], np.ndarray]:
    """Gene ids and one finite number per gene, from a mapping or a two-column DataFrame
    (gene id, then value)."""
    if isinstance(value, pl.DataFrame):
        if value.width != 2:
            raise GeneticsUsageError(
                f"{what} as a DataFrame must have two columns, gene id then value; got "
                f"{value.columns}"
            )
        ids_s, vals_s = value.to_series(0), value.to_series(1)
        if ids_s.null_count() or vals_s.null_count():
            raise GeneticsUsageError(f"{what} holds a null gene id or value")
        if ids_s.dtype != pl.Utf8 or not vals_s.dtype.is_numeric():
            raise GeneticsUsageError(
                f"{what} needs string gene ids and numeric values, got {ids_s.dtype} and "
                f"{vals_s.dtype}"
            )
        ids, vals = ids_s.to_list(), vals_s.cast(pl.Float64).to_numpy()
    elif isinstance(value, Mapping):
        ids, raw = list(value.keys()), list(value.values())
        if any(g is None for g in ids) or any(v is None for v in raw):
            raise GeneticsUsageError(f"{what} holds a null gene id or value")
        if not all(isinstance(g, str) for g in ids):
            raise GeneticsUsageError(f"{what} must be keyed by gene ids as strings")
        if not all(
            isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
            for v in raw
        ):
            raise GeneticsUsageError(f"{what} values must be numbers")
        vals = np.array(raw, dtype=np.float64)
    else:
        raise GeneticsUsageError(
            f"{what} must be a mapping of gene id -> value or a two-column DataFrame, got "
            f"{type(value).__name__}"
        )
    if not np.isfinite(vals).all():
        raise GeneticsUsageError(f"{what} holds a NaN or infinite value")
    if len(set(ids)) != len(ids):
        raise GeneticsUsageError(f"{what} gives more than one value for a gene id")
    if not ids:
        raise GeneticsUsageError(f"{what} is empty")
    return ids, vals


# the null's cost in units of ~16 ns of single-thread CPU on the dev VM (numpy 2.4), per
# permutation: one unit per rank position of every distinct size (the merge and running sum
# touch each), ten per position of the largest size (the draw's Python-level step per gene)
# and a twentieth per gene (resetting the permutation). Fit to `_null_es` there: GO BP scale,
# sizes 5..300 over 18,215 genes, took 8.1 s for 10,000 permutations (4.9e8 units); sizes
# 5..1000 took 7.0 s per 1,000. The budget is 20 s of that CPU, a third of the sandbox's
# 60 s wall, leaving the rest for gVisor, its 1.5 CPUs and the remainder of the call. A
# changed `_null_es` or a sandbox more than ~3x slower than the VM per unit invalidates it.
_NULL_BUDGET = 1.25e9


def _null_cost(sizes: np.ndarray, n_genes: int, nperm: int) -> float:
    if sizes.size == 0:
        return 0.0
    return nperm * (float(sizes.sum()) + 10.0 * float(sizes[-1]) + n_genes / 20)


def _is_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _running_es(
    w: np.ndarray, pos: np.ndarray, n_genes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """fgsea's `calcGseaStat`, one set per row of `pos`.

    `pos` holds the members' 0-based rank positions, ascending, in a list of `n_genes` whose
    weights in rank order are `w`. Returns the ES and the indices of the running sum's maximum
    and minimum, the ends of the leading edge."""
    m = pos.shape[1]
    r = w[pos]
    nr = r.sum(axis=1, keepdims=True)
    zero = nr[:, 0] == 0
    with np.errstate(invalid="ignore", divide="ignore"):
        hit = np.cumsum(r, axis=1) / nr
        step = r / nr
    if zero.any():
        # all members weigh nothing: fgsea steps up by 1/m per member instead
        hit[zero] = np.arange(1, m + 1) / m
        step[zero] = 1 / m
    tops = hit - (pos - np.arange(m)) / (n_genes - m)
    bottoms = tops - step
    rows = np.arange(pos.shape[0])
    imax, imin = tops.argmax(axis=1), bottoms.argmin(axis=1)
    max_p, min_p = tops[rows, imax], bottoms[rows, imin]
    es = np.where(max_p == -min_p, 0.0, np.where(max_p > -min_p, max_p, min_p))
    return es, imax, imin


def _null_es(w: np.ndarray, sizes: np.ndarray, nperm: int, rng: np.random.Generator) -> np.ndarray:
    """Permutation-null ES, `(nperm, len(sizes))`, for ascending distinct set `sizes`.

    Each permutation draws `max(sizes)` genes without replacement and the random set of size
    m is the first m of them — fgseaSimple's scheme, so every set of one size is held to the
    same null and the cost scales with the distinct sizes, not the number of sets."""
    n_genes = w.size
    k = int(sizes[-1])
    out = np.empty((nperm, sizes.size))
    # measured at 18k genes x 300 distinct sizes: 256 rows keep the per-size work arrays in
    # cache (921 rows ran 1.7x slower) while amortising the per-gene Python loop below
    chunk = min(nperm, 256)
    for start in range(0, nperm, chunk):
        n = min(chunk, nperm - start)
        rows = np.arange(n)
        perm = np.tile(np.arange(n_genes, dtype=np.int32), (n, 1))
        for i in range(k):
            j = rng.integers(i, n_genes, size=n)
            picked = perm[rows, j]
            perm[rows, j] = perm[:, i]
            perm[:, i] = picked
        drawn = perm[:, :k].copy()
        del perm
        srt = np.empty((n, 0), dtype=np.int32)
        prev = 0
        for c, m in enumerate(sizes):
            block = np.sort(drawn[:, prev:m], axis=1)
            # a stable sort is timsort, which merges two presorted runs in linear time
            srt = np.sort(np.concatenate([srt, block], axis=1), axis=1, kind="stable")
            prev = m
            out[start : start + n, c] = _running_es(w, srt, n_genes)[0]
    return out


def gsea_preranked(
    stats: Any,
    gene_sets: Mapping[str, Iterable[str]],
    *,
    min_size: int = 5,
    max_size: int = 300,
    nperm: int = 10_000,
    seed: int | None = None,
    weight: float = 1.0,
) -> dict[str, Any]:
    """Preranked GSEA: do a set's genes sit at the top or the bottom of a ranked list?

    `stats` — one signed statistic per gene (e.g. signed -log10 p, or a z), as a mapping of
    gene id -> value or a two-column DataFrame (gene id, value). Genes are ranked by value,
    largest first; tied values keep their input order, which is what fgsea's (stable) sort
    does too, so sort the input first if a different tie order matters. Nulls, NaN and a gene
    given twice are refused.
    `gene_sets` — a mapping of set id -> gene ids, in the same namespace as `stats`.
    `min_size`, `max_size` — only sets with this many genes carrying a statistic are tested
    (the universe is the genes in `stats`; set members without one are dropped first).
    `nperm` — permutations. Each permutation draws random gene sets without replacement;
    sets of the same size share one null distribution. The null's cost grows with nperm
    times the number and size of the distinct set sizes tested, and a run that would not
    fit the sandbox's time limit is refused up front: lower `nperm` or `max_size`.
    `seed` — fixes the permutations; when omitted one is drawn and reported in `parameters`
    so the run can be repeated exactly.
    `weight` — exponent on |stat| in the running sum (fgsea's `gseaParam`; 1 is standard,
    0 is the unweighted Kolmogorov-Smirnov form).

    This is `fgsea::fgseaSimple`: ES is the running-sum extreme; for each set the
    permutation ESs of the same sign give NES = ES / their mean, and
    p = min((1 + #null <= ES) / (1 + #null <= 0), (1 + #null >= ES) / (1 + #null >= 0)),
    so p cannot go below about 2 / nperm — a set at that floor is "at least this
    significant", and a smaller p needs more permutations. `padj` is BH over tested sets.

    Returns `{"results", "parameters"}`. `results` has one row per tested set, ordered by p:
    `set_id`, `size`, `es`, `nes`, `pval`, `padj`, `n_more_extreme` (same-sign permutation
    ESs at least as extreme) and `leading_edge` (the genes up to the running-sum peak, most
    extreme first). `parameters` states the universe, `n_genes`, `tied_stats` (non-zero
    statistics repeating an earlier value), `nperm`, `seed`, `weight`, the size filter,
    `sets_given` and `sets_tested`.
    """
    _check_sizes(min_size, max_size)
    if not _is_int(nperm) or nperm < 1:
        raise GeneticsUsageError(f"nperm must be a positive integer, got {nperm!r}")
    if (
        not isinstance(weight, (int, float, np.integer, np.floating))
        or isinstance(weight, bool)
        or not np.isfinite(weight)
        or weight < 0
    ):
        raise GeneticsUsageError(f"weight must be a finite number >= 0, got {weight!r}")
    if seed is None:
        seed = int(np.random.SeedSequence().generate_state(1)[0])
    elif not _is_int(seed) or seed < 0:
        raise GeneticsUsageError(f"seed must be a non-negative integer, got {seed!r}")
    nperm, seed = int(nperm), int(seed)
    genes, values = _gene_values(stats, "stats")
    order = np.argsort(-values, kind="stable")
    ranked = [genes[i] for i in order]
    w = np.abs(values[order]) ** weight
    n_genes = len(ranked)

    set_ids, flat = _flatten(gene_sets)
    m = _membership(flat, len(set_ids), {g: i for i, g in enumerate(ranked)})
    m.sort_indices()
    sizes = np.diff(m.indptr)
    # a set holding every ranked gene has no running sum to speak of
    tested = np.flatnonzero((sizes >= min_size) & (sizes <= min(max_size, n_genes - 1)))
    uniq, size_col = np.unique(sizes[tested], return_inverse=True)
    cost = _null_cost(uniq, n_genes, nperm)
    if cost > _NULL_BUDGET:
        raise GeneticsUsageError(
            f"{nperm} permutations over {uniq.size} distinct set sizes up to {uniq[-1]} "
            f"would take about {cost / _NULL_BUDGET:.1f}x the time this call is allowed; "
            "lower nperm or max_size"
        )

    es = np.empty(tested.size)
    leading: list[list[str]] = [[] for _ in tested]
    # sets of one size stack into one array, so the observed ES costs a call per size
    for c, size in enumerate(uniq):
        sel = np.flatnonzero(size_col == c)
        starts = m.indptr[tested[sel]]
        pos = m.indices[starts[:, None] + np.arange(size)]
        e, imax, imin = _running_es(w, pos, n_genes)
        es[sel] = e
        for t, p, e1, hi, lo in zip(sel, pos, e, imax, imin, strict=True):
            lead = p[: hi + 1] if e1 > 0 else p[lo:][::-1] if e1 < 0 else p[:0]
            leading[t] = [ranked[i] for i in lead]

    null = (
        _null_es(w, uniq, nperm, np.random.default_rng(seed))
        if uniq.size
        else np.empty((nperm, 0))
    )
    null.sort(axis=0)
    pval, nes = np.empty(tested.size), np.full(tested.size, np.nan)
    n_more = np.empty(tested.size, dtype=np.int64)
    for c in range(uniq.size):
        sel = np.flatnonzero(size_col == c)
        col, e = null[:, c], es[sel]
        n_le = np.searchsorted(col, e, side="right")
        n_ge = nperm - np.searchsorted(col, e, side="left")
        le0 = int(np.searchsorted(col, 0.0, side="right"))
        ge0 = nperm - int(np.searchsorted(col, 0.0, side="left"))
        pval[sel] = np.minimum((1 + n_le) / (1 + le0), (1 + n_ge) / (1 + ge0))
        n_more[sel] = np.where(e > 0, n_ge, n_le)
        ge_mean = col[nperm - ge0 :].sum() / ge0 if ge0 else 0.0
        le_mean = abs(col[:le0].sum() / le0) if le0 else 0.0
        denom = np.where(e > 0, ge_mean, le_mean)
        ok = denom != 0
        nes[sel[ok]] = e[ok] / denom[ok]

    results = pl.DataFrame({
        "set_id": pl.Series([set_ids[i] for i in tested], dtype=pl.Utf8),
        "size": sizes[tested].astype(np.int64),
        "es": es,
        "nes": nes,
        "pval": pval,
        "padj": _bh(pval),
        "n_more_extreme": n_more,
        "leading_edge": pl.Series(leading, dtype=pl.List(pl.Utf8)),
    }).sort(["pval", "set_id"])
    nonzero = values[values != 0]
    return {
        "results": results,
        "parameters": {
            "universe": "genes with a statistic",
            "n_genes": n_genes,
            "tied_stats": int(nonzero.size - np.unique(nonzero).size),
            "nperm": nperm,
            "seed": seed,
            "weight": float(weight),
            "min_size": int(min_size),
            "max_size": int(max_size),
            "sets_given": len(set_ids),
            "sets_tested": int(tested.size),
        },
    }


def competitive(
    gene_stats: Any,
    gene_sets: Mapping[str, Iterable[str]],
    *,
    covariates: pl.DataFrame | None = None,
    min_size: int = 5,
    max_size: int = 300,
) -> dict[str, Any]:
    """Competitive gene set test: are a set's gene z-scores higher than the other genes'?

    MAGMA's competitive gene-set analysis on gene-level statistics: for each set, the
    ordinary least squares fit `z ~ 1 + in_set + covariates` over every gene with a statistic,
    and the one-sided t-test of the `in_set` coefficient being > 0. Matches R's
    `lm(z ~ in_set + log_len)` with `pt(t, df, lower.tail = FALSE)`.

    `gene_stats` — one z per gene (higher = more associated), as a mapping of gene id -> z or
    a two-column DataFrame (gene id, z): e.g. a burden test's z, or a user's own gene-level
    analysis. Nulls, NaN and a gene given twice are refused.
    `gene_sets` — a mapping of set id -> gene ids, in the same namespace.
    `covariates` — optional DataFrame, gene id in the first column and one numeric covariate
    per further column (e.g. log gene length, gene density — what MAGMA adjusts for, since
    long genes collect larger statistics). Every gene in `gene_stats` must have a row.
    `min_size`, `max_size` — only sets with this many genes carrying a statistic are tested
    (the universe is the genes in `gene_stats`; members without one are dropped first).

    Returns `{"results", "parameters"}`. `results` has one row per tested set, ordered by p:
    `set_id`, `size`, `beta` (the membership coefficient: the mean z shift of members), `se`,
    `t`, `pval` (one-sided) and `padj` (BH over tested sets). `parameters` states the
    universe, `n_genes`, `covariates`, residual `df`, the size filter, `sets_given`,
    `sets_tested` and `sets_aliased` (sets whose membership the covariates explain entirely,
    which have no estimate and are not tested).
    """
    _check_sizes(min_size, max_size)
    genes, z = _gene_values(gene_stats, "gene_stats")
    n_genes = len(genes)
    index = {g: i for i, g in enumerate(genes)}
    design, names = np.ones((n_genes, 1)), []
    if covariates is not None:
        if not isinstance(covariates, pl.DataFrame) or covariates.width < 2:
            raise GeneticsUsageError(
                "covariates must be a DataFrame with gene ids in the first column and one "
                "numeric covariate per further column"
            )
        ids = covariates.to_series(0)
        names = covariates.columns[1:]
        cov = covariates.select(names)
        if ids.dtype != pl.Utf8 or not all(d.is_numeric() for d in cov.dtypes):
            raise GeneticsUsageError("covariates need string gene ids and numeric covariates")
        if ids.null_count() or any(cov.null_count().row(0)):
            raise GeneticsUsageError("covariates hold a null gene id or value")
        if ids.is_duplicated().any():
            raise GeneticsUsageError("covariates give more than one row for a gene id")
        cov_index = {g: i for i, g in enumerate(ids.to_list())}
        missing = [g for g in genes if g not in cov_index]
        if missing:
            raise GeneticsUsageError(
                f"{len(missing)} genes with a statistic have no covariates, e.g. {missing[:3]}"
            )
        mat = cov.cast(pl.Float64).to_numpy()[[cov_index[g] for g in genes]]
        if not np.isfinite(mat).all():
            raise GeneticsUsageError("covariates hold a NaN or infinite value")
        design = np.column_stack([design, mat])
    # unit-norm columns, so the rank check compares directions rather than units: a
    # covariate in tiny units is not collinear, and the span (hence q) is unchanged
    norms = np.linalg.norm(design, axis=0)
    diag = np.zeros(design.shape[1])
    if (norms > 0).all():
        q, r = np.linalg.qr(design / norms)
        diag = np.abs(np.diag(r))
    if (diag <= 1e-10).any():
        raise GeneticsUsageError(
            "the covariates are collinear with each other or with the intercept (e.g. a "
            "constant column)"
        )
    df = n_genes - design.shape[1] - 1
    if df < 1:
        raise GeneticsUsageError(f"{n_genes} genes are too few for the regression")

    # Frisch-Waugh: with the covariate space projected out of z and of every membership
    # vector, each set's fit reduces to sparse products, with no per-set solve
    z_res = z - q @ (q.T @ z)
    # a residual this small next to z is rounding (|z_res| / |z| ~ 1e-16 for constant z),
    # and t-statistics built on it are noise; 1e-10 in norm leaves room above that floor at
    # any realistic gene count
    if z_res @ z_res <= 1e-20 * (z @ z):
        raise GeneticsUsageError(
            "gene_stats leave no variation once the intercept and covariates are fitted "
            "(e.g. every z is the same)"
        )
    set_ids, flat = _flatten(gene_sets)
    m = _membership(flat, len(set_ids), index)
    sizes = np.diff(m.indptr)
    tested = np.flatnonzero((sizes >= min_size) & (sizes <= max_size))
    mt = m[tested].astype(np.float64)
    sxx = sizes[tested] - np.asarray(mt @ q) ** 2 @ np.ones(q.shape[1])
    keep = sxx > 1e-9 * sizes[tested]
    aliased = int((~keep).sum())
    tested, mt, sxx = tested[keep], mt[keep], sxx[keep]
    sxy = mt @ z_res
    beta = sxy / sxx
    rss = z_res @ z_res - sxy**2 / sxx
    se = np.sqrt(rss / df / sxx)
    t = beta / se
    pval = t_dist.sf(t, df)

    results = pl.DataFrame({
        "set_id": pl.Series([set_ids[i] for i in tested], dtype=pl.Utf8),
        "size": sizes[tested].astype(np.int64),
        "beta": beta,
        "se": se,
        "t": t,
        "pval": pval,
        "padj": _bh(pval),
    }).sort(["pval", "set_id"])
    return {
        "results": results,
        "parameters": {
            "universe": "genes with a statistic",
            "n_genes": n_genes,
            "covariates": list(names),
            "df": int(df),
            "min_size": int(min_size),
            "max_size": int(max_size),
            "sets_given": len(set_ids),
            "sets_tested": int(tested.size),
            "sets_aliased": aliased,
        },
    }
