"""Gene set enrichment: which gene sets a list of genes, or a gene-level statistic, points at.

    import genetics
    genetics.enrichment.collections()   # gene-set collections in this deployment, with licences
    res = genetics.enrichment.ora(["LDLR", "APOB", "PCSK9"], "reactome", universe=tested)
    res["results"]      # one row per tested set: size, overlap, expected, p, BH q, overlap genes
    res["parameters"]   # universe convention, |U|, list size, size filter and id resolution used
    res["attribution"]  # bundle release and, per collection used, the text to cite

Three analyses:

- `ora(genes, gene_sets, universe)` — over-representation of a gene LIST: do more of the
  genes fall in a set than chance would put there (hypergeometric upper tail)? Use for a
  hit list such as significant genes or a credible-set gene list.
- `gsea_preranked(stats, gene_sets)` — preranked GSEA on one statistic per gene, signed so
  that large positive and large negative are both interesting. Use when every tested gene
  has a value and no cut-off is wanted.
- `competitive(gene_stats, gene_sets)` — MAGMA-style competitive test: are the set's gene
  z-scores higher than other genes', optionally adjusting for covariates such as gene
  length. Use for gene-level z-scores from a burden test or your own per-gene analysis;
  turning GWAS variant statistics into gene statistics needs an LD reference that is not here.

GENE SETS. `gene_sets` is a collection name, a list of names, or a mapping
{set name: gene ids} used exactly as given. Names refer to this deployment's bundle
(`GENETICS_GENESETS_DIR`); `collections()` lists what is present, with licence, upstream
release and attribution text, and `gene_sets(collection)` fetches one. Which collections
exist differs between deployments. With a collection name, gene ids may be HGNC symbols,
NCBI Entrez ids (as strings) or Ensembl gene ids with or without a version suffix, mixed
freely; unresolvable inputs are dropped and counted in `parameters["id_resolution"]`, and
results also carry each set's `name` and `collection` and the overlap genes as symbols.
Always cite the `attribution` entry of a result that used bundled collections.

THE UNIVERSE IS THE ANALYSIS. ORA's background U is the genes that COULD have made the
list — every gene tested in the study, not the genome; a list of protein-coding genes tested
against all ~60k genes looks enriched for everything protein-coding. Everything outside U
is dropped from the list and from every set before counting, and the size filter applies to
set sizes after that restriction.

- `universe_mode="annotated"` (default) further intersects U with the genes annotated to any
  set in the collection, as clusterProfiler does; `"given"` uses U exactly as supplied, as
  `fgsea::fora` does. The two can give materially different p (by about a factor of two on the
  test fixture); say which was used.
- Without `universe`, U is every gene annotated to any set, under either mode.

For `gsea_preranked` and `competitive` the universe is the genes that carry a statistic;
sets are restricted to them before the size filter.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping
from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy import sparse
from scipy.stats import hypergeom
from scipy.stats import t as t_dist

from genetics_mcp_server.sdk.errors import GeneticsUsageError

__all__ = ["collections", "competitive", "gene_sets", "gsea_preranked", "ora"]

_UNIVERSE_MODES = ("annotated", "given")

GENESETS_ENV = "GENETICS_GENESETS_DIR"
_GENESETS_DEFAULT = "/genetics/genesets"
_UNRESOLVED_SAMPLE = 10
_ENSEMBL = re.compile(r"ENSG\d+(\.\d+)?")
_ENTREZ = re.compile(r"\d+")


class _Bundle:
    """One staged gene set bundle. Set metadata is read up front (tens of thousands of rows);
    memberships are read per collection on first use and kept as polars frames, never as a
    Python dict of lists. Measured on daly-staging (gVisor, 2 CPUs, polars 2 threads): reading
    all three bundle files and running ORA over all 20,299 sets peaked at VmSize 1,410,156 kB
    of the 2560 MiB RLIMIT_AS. The sandbox image pins POLARS_MAX_THREADS=2 because each polars
    thread's malloc arena reserves address space (a 4-CPU host with ~16 threads hit the limit)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        manifest = json.loads((path / "manifest.json").read_text())
        self.release_id = manifest.get("bundle", {}).get("release_id") or manifest.get(
            "release_id"
        )
        self.meta = {c["collection"]: c for c in manifest.get("collections", [])}
        self.sets = pl.read_parquet(path / "sets.parquet").select(
            "set_id", "collection", "name", "url_or_stable_id", "n_given", "n_mapped",
            "n_unmapped",
        )
        held = self.sets["collection"].unique().to_list()
        # a collection with no manifest entry has no licence or attribution to serve it under
        unlisted = sorted(set(held) - set(self.meta))
        if unlisted:
            raise GeneticsUsageError(
                f"gene set bundle at {path} holds collections its manifest does not describe: "
                f"{unlisted}"
            )
        self.order = [c for c in self.meta if c in held]
        self._members: dict[str, pl.DataFrame] = {}
        self._genes: _GeneIndex | None = None

    def members(self, collection: str) -> pl.DataFrame:
        if collection not in self._members:
            ids = self.sets.filter(pl.col("collection") == collection)["set_id"]
            self._members[collection] = (
                pl.scan_parquet(self.path / "members.parquet")
                .filter(pl.col("set_id").is_in(ids.implode()))
                .select("set_id", "ensembl_gene_id")
                .collect()
            )
        return self._members[collection]

    @property
    def genes(self) -> _GeneIndex:
        if self._genes is None:
            self._genes = _GeneIndex(pl.read_parquet(self.path / "genes.parquet"))
        return self._genes

    def check_names(self, names: list[str]) -> None:
        unknown = [n for n in names if n not in self.order]
        if unknown:
            raise GeneticsUsageError(
                f"no gene set collection {unknown[0]!r} in this deployment; it holds "
                f"{self.order} (see genetics.enrichment.collections())"
            )

    def attribution(self, names: list[str], used: list[str] | None = None) -> dict[str, Any]:
        """`names` are the collections asked for; `collections` cites only those in `used`
        (default: all of them), the ones whose sets actually reached the results."""
        used = names if used is None else used
        return {
            "release_id": self.release_id,
            "requested": list(names),
            "collections": [
                {
                    "collection": n,
                    "licence": self.meta[n].get("spdx_license"),
                    **_release_fields(self.meta[n]),
                    "attribution": self.meta[n].get("attribution"),
                }
                for n in names
                if n in used
            ],
        }


def _release_fields(meta: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in meta.items() if k.endswith("_release")}


class _GeneIndex:
    """Symbol and Entrez lookups to unversioned Ensembl over the bundle's gene table.

    Only the current approved HGNC symbol is known: previous and alias symbols are not in
    the table, so an outdated symbol resolves to nothing and is reported, never guessed. A
    symbol is matched exactly first, then ignoring case, but only where the case-folded
    form names a single gene."""

    def __init__(self, genes: pl.DataFrame) -> None:
        ens, sym = genes["ensembl_gene_id"].to_list(), genes["symbol"].to_list()
        self.symbol_of = {e: s for e, s in zip(ens, sym, strict=True) if s is not None}
        self.by_symbol = {s: e for e, s in self.symbol_of.items()}
        upper: dict[str, str | None] = {}
        for s, e in self.by_symbol.items():
            u = s.upper()
            upper[u] = None if u in upper and upper[u] != e else e
        self.by_upper = {u: e for u, e in upper.items() if e is not None}
        entrez = genes.select("entrez_id", "ensembl_gene_id").drop_nulls()
        self.by_entrez = dict(
            zip(entrez["entrez_id"].to_list(), entrez["ensembl_gene_id"].to_list(), strict=True)
        )

    def resolve(self, ids: list[str], what: str) -> tuple[list[str | None], dict[str, Any]]:
        """One unversioned Ensembl id (or None) per input, and the report of how they
        resolved. An Ensembl-shaped id is taken as one even when the table lacks it, so a
        supplied universe is not shrunk by the table's coverage."""
        out: list[str | None] = []
        by = {"ensembl": 0, "entrez": 0, "symbol": 0, "symbol_ignoring_case": 0}
        for g in ids:
            if _ENSEMBL.fullmatch(g):
                e, how = g.split(".", 1)[0], "ensembl"
            elif _ENTREZ.fullmatch(g):
                e, how = self.by_entrez.get(g), "entrez"
            elif g in self.by_symbol:
                e, how = self.by_symbol[g], "symbol"
            else:
                e, how = self.by_upper.get(g.upper()), "symbol_ignoring_case"
            if e is not None:
                by[how] += 1
            out.append(e)
        unresolved = [g for g, e in zip(ids, out, strict=True) if e is None]
        return out, {
            "given": len(ids),
            "resolved": len(ids) - len(unresolved),
            "unresolved": len(unresolved),
            "unresolved_sample": unresolved[:_UNRESOLVED_SAMPLE],
            "resolved_by": by,
            "what": what,
        }


_BUNDLES: dict[Path, _Bundle] = {}


def _bundle() -> _Bundle:
    path = Path(os.environ.get(GENESETS_ENV) or _GENESETS_DEFAULT)
    if path not in _BUNDLES:
        if not (path / "manifest.json").is_file():
            raise GeneticsUsageError(
                f"no gene set bundle at {path}: {GENESETS_ENV} must name a directory holding "
                "manifest.json, sets.parquet, members.parquet and genes.parquet. The sandbox "
                f"image stages one at {_GENESETS_DEFAULT}; outside it (local development) "
                "there is none unless you point the variable at one — or pass gene_sets as a "
                "mapping of set id -> gene ids instead of a collection name"
            )
        _BUNDLES[path] = _Bundle(path)
    return _BUNDLES[path]


def collections() -> pl.DataFrame:
    """The gene set collections this deployment's image holds, one row each.

    Columns: `collection` (the name to pass as `gene_sets`), `n_sets`, `licence` (SPDX id),
    the upstream release fields (`upstream_release`, and e.g. `go_release` / `hpo_release`
    where the source states one; null elsewhere), `attribution` (the text to cite when
    reporting results from it) and `release_id` (the bundle's). Which collections appear
    depends on the deployment's licence allowlist, so ask rather than assume."""
    b = _bundle()
    counts = dict(b.sets.group_by("collection").len().iter_rows())
    release_keys = sorted({k for n in b.order for k in _release_fields(b.meta[n])})
    rows = [
        {
            "collection": n,
            "n_sets": counts[n],
            "licence": b.meta[n].get("spdx_license"),
            **{k: b.meta[n].get(k) for k in release_keys},
            "attribution": b.meta[n].get("attribution"),
            "release_id": b.release_id,
        }
        for n in b.order
    ]
    schema = {"collection": pl.Utf8, "n_sets": pl.Int64, "licence": pl.Utf8}
    schema |= {k: pl.Utf8 for k in release_keys}
    schema |= {"attribution": pl.Utf8, "release_id": pl.Utf8}
    return pl.DataFrame(rows, schema=schema)


def _collection_names(value: Any) -> list[str]:
    names = [value] if isinstance(value, str) else value
    if (
        not isinstance(names, (list, tuple))
        or not names
        or not all(isinstance(n, str) for n in names)
    ):
        raise GeneticsUsageError(
            "gene_sets must be a collection name, a list of collection names, or a mapping "
            f"of set id -> gene ids; got {type(value).__name__}"
        )
    if len(set(names)) != len(names):
        raise GeneticsUsageError(f"a collection is named twice in {list(names)}")
    return list(names)


def gene_sets(collection: str | list[str], *, ids: str = "ensembl") -> dict[str, Any]:
    """The gene sets of one or more shipped collections, to inspect, filter, or pass on.

    `collection` — a name from `collections()`, or a list of them.
    `ids` — `"ensembl"` (unversioned Ensembl gene ids, the bundle's own) or `"symbol"`
    (current HGNC symbols).

    Returns `{"gene_sets", "sets", "attribution"}`: `gene_sets` maps set id -> gene ids,
    `sets` is one row per set (`set_id`, `collection`, `name`, `url_or_stable_id`, and the
    member counts `n_given`, `n_mapped`, `n_unmapped` from harmonising the source to Ensembl),
    and `attribution` holds the bundle release and the text to cite. A mapping filtered from
    this and passed back to an analysis is used as given: give the analysis ids of the same
    kind as `ids`."""
    if ids not in ("ensembl", "symbol"):
        raise GeneticsUsageError(f"ids must be 'ensembl' or 'symbol', got {ids!r}")
    src = _Bundled(_collection_names(collection))
    set_ids, flat = src.flat()
    gene = pl.col("gene")
    if ids == "symbol":
        gene = gene.replace(src.bundle.genes.symbol_of)
    grouped = (
        flat.with_columns(gene)
        .group_by("row", maintain_order=True)
        .agg(pl.col("gene").unique(maintain_order=True))
    )
    mapping: dict[str, list[str]] = {s: [] for s in set_ids}
    for row, members in grouped.iter_rows():
        mapping[set_ids[row]] = members
    return {
        "gene_sets": mapping,
        "sets": src.sets,
        "attribution": src.bundle.attribution(src.names),
    }


class _Bundled:
    """An analysis's `gene_sets` given as shipped collection names."""

    def __init__(self, names: list[str]) -> None:
        self.bundle = _bundle()
        self.bundle.check_names(names)
        self.names = names
        rank = {n: i for i, n in enumerate(names)}
        self.sets = (
            self.bundle.sets.filter(pl.col("collection").is_in(names))
            .with_columns(pl.col("collection").replace_strict(rank).alias("_rank"))
            .sort("_rank", "set_id")
            .drop("_rank")
        )
        self.reports: dict[str, dict[str, Any]] = {}

    def flat(self) -> tuple[list[str], pl.DataFrame]:
        """The same (set ids, (row, gene) pairs) `_flatten` makes of a mapping."""
        rows = self.sets.select("set_id").with_row_index("row").with_columns(
            pl.col("row").cast(pl.Int64)
        )
        members = pl.concat([self.bundle.members(n) for n in self.names])
        flat = members.join(rows, on="set_id", how="inner").select(
            "row", pl.col("ensembl_gene_id").alias("gene")
        )
        return self.sets["set_id"].to_list(), flat

    def resolve(self, ids: list[str], what: str) -> list[str | None]:
        out, report = self.bundle.genes.resolve(ids, what)
        del report["what"]
        self.reports[what] = report
        return out

    def resolve_list(self, ids: list[str], what: str) -> list[str]:
        """Resolved, distinct, unresolved inputs dropped (and reported)."""
        return list(dict.fromkeys(e for e in self.resolve(ids, what) if e is not None))

    def resolve_keyed(self, ids: list[str], what: str) -> tuple[list[str], np.ndarray]:
        """Resolved ids and the input positions they came from; two inputs naming one gene
        are refused, since their values cannot both stand for it."""
        out = self.resolve(ids, what)
        seen: dict[str, int] = {}
        for i, e in enumerate(out):
            if e is None:
                continue
            if e in seen:
                raise GeneticsUsageError(
                    f"{what}: {ids[seen[e]]!r} and {ids[i]!r} are the same gene ({e})"
                )
            seen[e] = i
        return list(seen), np.fromiter(seen.values(), dtype=np.int64, count=len(seen))

    def finish(self, out: dict[str, Any], gene_cols: tuple[str, ...]) -> dict[str, Any]:
        """Adds set names, collections and symbol columns to the results, the id resolution
        report to the parameters, and the attribution alongside."""
        res = out["results"]
        sym = self.bundle.genes.symbol_of
        res = res.join(
            self.sets.select("set_id", "name", "collection"), on="set_id", how="left"
        ).with_columns(
            pl.col(c).list.eval(pl.element().replace(sym)).alias(_symbols_column(c))
            for c in gene_cols
        )
        front = ["set_id", "name", "collection"]
        res = res.select(front + [c for c in res.columns if c not in front])
        out["results"] = res
        out["parameters"]["id_resolution"] = self.reports
        tested = set(res["collection"].to_list())
        out["attribution"] = self.bundle.attribution(
            self.names, [n for n in self.names if n in tested]
        )
        return out


def _symbols_column(gene_col: str) -> str:
    return gene_col.removesuffix("_genes") + "_symbols"


def _bundled(gene_sets: Any) -> _Bundled | None:
    return None if isinstance(gene_sets, Mapping) else _Bundled(_collection_names(gene_sets))


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
    gene_sets: Mapping[str, Iterable[str]] | str | list[str],
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
    `gene_sets` — a shipped collection name or list of names (see `collections()`), or a
    mapping of set id -> gene ids. With a name, `genes` and `universe` may be symbols, Entrez
    or Ensembl ids, resolved as the module docstring says. With a mapping, the ids must be in
    the same namespace as `genes` and `universe` (all Ensembl, or all symbols); nothing is
    translated.
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

    With a collection name, `results` also has `name`, `collection` and `overlap_symbols`,
    `parameters["id_resolution"]` reports how `genes` and `universe` resolved (counts and a
    sample of the unresolved), and an `"attribution"` key holds the bundle release and the
    text to cite per collection.
    """
    if universe_mode not in _UNIVERSE_MODES:
        raise GeneticsUsageError(
            f"universe_mode must be one of {_UNIVERSE_MODES}, got {universe_mode!r}"
        )
    _check_sizes(min_size, max_size)
    query = _gene_list(genes, "genes")

    bundled = _bundled(gene_sets)
    set_ids, flat = bundled.flat() if bundled else _flatten(gene_sets)
    if bundled:
        query = bundled.resolve_list(query, "genes")
    annotated = flat["gene"].unique(maintain_order=True).to_list()
    if universe is None:
        u_genes, source = annotated, "annotated genes (no universe given)"
    else:
        u_genes = _gene_list(universe, "universe")
        if bundled:
            u_genes = bundled.resolve_list(u_genes, "universe")
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
    out = {
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
    return bundled.finish(out, ("overlap_genes",)) if bundled else out


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
    gene_sets: Mapping[str, Iterable[str]] | str | list[str],
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
    `gene_sets` — a shipped collection name or list of names, against which `stats` may be
    keyed by symbol, Entrez or Ensembl id; or a mapping of set id -> gene ids in the same
    namespace as `stats`. Two keys naming one gene are refused.
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
    statistics repeating an earlier value — zeros are not counted, yet their input order
    still places set genes in the ranking: with every statistic zero, input order alone
    yields a significant-looking ES, as in fgsea), `nperm`, `seed`, `weight`, the size filter,
    `sets_given` and `sets_tested`. With a collection name, as for `ora`: `name`,
    `collection` and `leading_edge_symbols` columns, `parameters["id_resolution"]` (keys that
    resolve to no gene are dropped before ranking) and an `"attribution"` key.
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
    bundled = _bundled(gene_sets)
    if bundled:
        genes, kept = bundled.resolve_keyed(genes, "stats")
        values = values[kept]
        if not genes:
            raise GeneticsUsageError("none of the stats' gene ids resolves to a known gene")
    order = np.argsort(-values, kind="stable")
    ranked = [genes[i] for i in order]
    w = np.abs(values[order]) ** weight
    n_genes = len(ranked)

    set_ids, flat = bundled.flat() if bundled else _flatten(gene_sets)
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
    out = {
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
    return bundled.finish(out, ("leading_edge",)) if bundled else out


def competitive(
    gene_stats: Any,
    gene_sets: Mapping[str, Iterable[str]] | str | list[str],
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
    `gene_sets` — a shipped collection name or list of names, against which `gene_stats` and
    the covariates' ids may be symbols, Entrez or Ensembl ids; or a mapping of set id -> gene
    ids in the same namespace as `gene_stats`.
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
    which have no estimate and are not tested). With a collection name, as for `ora`: `name`
    and `collection` columns, `parameters["id_resolution"]` (genes that resolve to nothing are
    dropped) and an `"attribution"` key.
    """
    _check_sizes(min_size, max_size)
    genes, z = _gene_values(gene_stats, "gene_stats")
    bundled = _bundled(gene_sets)
    if bundled:
        genes, kept = bundled.resolve_keyed(genes, "gene_stats")
        z = z[kept]
        if not genes:
            raise GeneticsUsageError("none of gene_stats' gene ids resolves to a known gene")
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
        cov_ids = ids.to_list()
        if bundled:
            resolved, rows = bundled.resolve_keyed(cov_ids, "covariates")
            cov_index = dict(zip(resolved, rows.tolist(), strict=True))
        else:
            cov_index = {g: i for i, g in enumerate(cov_ids)}
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
    set_ids, flat = bundled.flat() if bundled else _flatten(gene_sets)
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
    out = {
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
    return bundled.finish(out, ()) if bundled else out
