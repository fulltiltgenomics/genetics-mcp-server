"""Gene set enrichment: which gene sets a list of genes, or a gene-level statistic, points at.

    import genetics
    genetics.enrichment.collections()   # gene-set collections in this deployment, with licences
    res = genetics.enrichment.ora(["LDLR", "APOB", "PCSK9"], "reactome", universe=tested)
    res["results"]      # one row per tested set: size, overlap, expected, p, BH q, overlap genes
    res["parameters"]   # universe convention, |U|, list size, size filter and id resolution used
    res["attribution"]  # bundle release and, per collection used, the text to cite

Four analyses:

- `ora(genes, gene_sets, universe)` — over-representation of a gene LIST: do more of the
  genes fall in a set than chance would put there (hypergeometric upper tail)? Use for a
  hit list such as significant genes from a gene-level test. Not for nearest genes of GWAS
  leads: an isolated gene in a gene-sparse region is the nearest gene of far more of the
  genome than one in a dense cluster, so sets rich in isolated genes look enriched; and a
  cluster of related genes at one signal counts several times — use `locus_enrichment`.
- `gsea_preranked(stats, gene_sets)` — preranked GSEA on one statistic per gene, signed so
  that large positive and large negative are both interesting. Use when every tested gene
  has a value and no cut-off is wanted.
- `competitive(gene_stats, gene_sets)` — MAGMA-style competitive test: are the set's gene
  z-scores higher than other genes', optionally adjusting for covariates such as gene
  length. Use for gene-level z-scores from a burden test or your own per-gene analysis;
  turning GWAS variant statistics into gene statistics needs an LD reference that is not
  here, and z-scores built from credible-set variants alone are not MAGMA (nearly every gene
  is 0, and genes sharing one signal count as independent) — for GWAS leads or credible sets
  use `locus_enrichment`.
- `locus_enrichment(loci, gene_sets)` — for GWAS lead variants (e.g. credible-set leads):
  how many independent loci map to a gene in the set, against random loci of matched local
  gene density. Needs only lead positions; `genes_for_loci(loci)` shows the mapping it uses
  (nearest gene or a window, from the bundle's GRCh38 gene coordinates) without testing.

`simplify(result)` collapses the significant sets of any of these results whose member genes
overlap heavily (e.g. nested GO terms) into one representative each.

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

__all__ = [
    "collections",
    "competitive",
    "gene_sets",
    "genes_for_loci",
    "gsea_preranked",
    "locus_enrichment",
    "ora",
    "simplify",
]

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
        self.release_id = manifest.get("bundle", {}).get("release_id") or manifest.get("release_id")
        self.meta = {c["collection"]: c for c in manifest.get("collections", [])}
        self.sets = pl.read_parquet(path / "sets.parquet").select(
            "set_id",
            "collection",
            "name",
            "url_or_stable_id",
            "n_given",
            "n_mapped",
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
        self.coords: dict[bool, _Coords] = {}

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
        rows = (
            self.sets.select("set_id")
            .with_row_index("row")
            .with_columns(pl.col("row").cast(pl.Int64))
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
    universe = pl.DataFrame(
        {
            "gene": pl.Series(list(index), dtype=pl.Utf8),
            "col": np.fromiter(index.values(), dtype=np.int64, count=len(index)),
        }
    )
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
        pl.DataFrame(
            {
                "row": sub.row[keep].astype(np.int64),
                "gene": pl.Series(u_genes, dtype=pl.Utf8).gather(sub.col[keep]),
            }
        )
        .sort("row", "gene")
        .group_by("row", maintain_order=True)
        .agg(overlap_genes=pl.col("gene"))
    )

    results = (
        pl.DataFrame(
            {
                "row": np.arange(tested.size, dtype=np.int64),
                "set_id": pl.Series([set_ids[i] for i in tested], dtype=pl.Utf8),
                "size": size_t.astype(np.int64),
                "overlap": k_t.astype(np.int64),
                "expected": expected,
                "fold_enrichment": k_t / expected,
                "pval": pval,
                "padj": _bh(pval),
            }
        )
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
        _null_es(w, uniq, nperm, np.random.default_rng(seed)) if uniq.size else np.empty((nperm, 0))
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

    results = pl.DataFrame(
        {
            "set_id": pl.Series([set_ids[i] for i in tested], dtype=pl.Utf8),
            "size": sizes[tested].astype(np.int64),
            "es": es,
            "nes": nes,
            "pval": pval,
            "padj": _bh(pval),
            "n_more_extreme": n_more,
            "leading_edge": pl.Series(leading, dtype=pl.List(pl.Utf8)),
        }
    ).sort(["pval", "set_id"])
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

    results = pl.DataFrame(
        {
            "set_id": pl.Series([set_ids[i] for i in tested], dtype=pl.Utf8),
            "size": sizes[tested].astype(np.int64),
            "beta": beta,
            "se": se,
            "t": t,
            "pval": pval,
            "padj": _bh(pval),
        }
    ).sort(["pval", "set_id"])
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


# GRCh38 extended MHC, inclusive; its long-range LD and gene density make one signal look
# like dozens of loci
_MHC = (6, 25_000_000, 34_000_000)
# GENCODE's main annotation (`gencode.v<N>.annotation`) is on GRCh38; the bundle's coordinates
# come from it, so leads must be GRCh38 too
_GENOME_BUILD = "GRCh38"
_PROTEIN_CODING = "HGNC locus_group == 'protein-coding gene'"
_CHROMS = {**{str(i): i for i in range(1, 25)}, "26": 26, "X": 23, "Y": 24, "M": 26, "MT": 26}
_CHROM_LABEL = {23: "X", 24: "Y", 26: "MT"}
_VARIANT = re.compile(
    r"(?:chr)?([0-9]{1,2}|X|Y|MT?)[:_-](\d+)(?:[:_-][A-Z*]+[:_-][A-Z*,]+)?", re.IGNORECASE
)
_LEAD_ID_COLUMNS = ("id", "cs_id", "variant")
_LOCUS_MODES = ("nearest", "window")


class _Chrom:
    """One chromosome's genes sorted by (start, end, Ensembl id), for interval lookups."""

    def __init__(self, offset: int, start: np.ndarray, end: np.ndarray) -> None:
        self.offset, self.start, self.end = offset, start, end
        # running maximum of the ends: the first index where it reaches p is the earliest
        # gene (in sort order) whose body extends to p from the left
        self.reach = np.maximum.accumulate(end)
        self.end_sorted = np.sort(end)
        self.max_len = int((end - start).max())

    def nearest(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Nearest gene (local index) and distance per position; 0 inside a body, ties to the
        gene earliest in sort order."""
        n = self.start.size
        k = np.searchsorted(self.start, p, "right")  # genes [0, k) start at or before p
        first = np.searchsorted(self.reach, p, "left")
        inside = first < k
        last = np.maximum(k - 1, 0)
        big = np.iinfo(np.int64).max
        d_left = np.where(k > 0, p - self.reach[last], big)
        left = np.searchsorted(self.reach, self.reach[last], "left")
        d_right = np.where(k < n, self.start[np.minimum(k, n - 1)] - p, big)
        gene = np.where(inside, first, np.where(d_left <= d_right, left, k))
        return gene, np.where(inside, 0, np.minimum(d_left, d_right))

    def window(
        self, p: np.ndarray, w: int, chunk: int = 50_000
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(position index, local gene index, distance) for every gene body overlapping
        [p - w, p + w]."""
        out_q, out_g, out_d = [], [], []
        for c0 in range(0, p.size, chunk):
            pc = p[c0 : c0 + chunk]
            # a gene reaching p - w starts no earlier than p - w - max_len
            lo = np.searchsorted(self.start, pc - w - self.max_len, "left")
            hi = np.searchsorted(self.start, pc + w, "right")
            counts = hi - lo
            q = np.repeat(np.arange(pc.size), counts)
            first = np.repeat(np.cumsum(counts) - counts, counts)
            g = np.repeat(lo, counts) + np.arange(q.size) - first
            keep = self.end[g] >= pc[q] - w
            q, g = q[keep], g[keep]
            d = np.maximum(np.maximum(self.start[g] - pc[q], pc[q] - self.end[g]), 0)
            out_q.append(q + c0)
            out_g.append(g)
            out_d.append(d)
        if not out_q:
            e = np.empty(0, dtype=np.int64)
            return e, e, e
        return np.concatenate(out_q), np.concatenate(out_g), np.concatenate(out_d)

    def density(self, p: np.ndarray, dw: int) -> np.ndarray:
        # genes ending before p - dw all start before p + dw, so the difference counts overlaps
        return np.searchsorted(self.start, p + dw, "right") - np.searchsorted(
            self.end_sorted, p - dw, "left"
        )


class _Coords:
    """The bundle's genes with coordinates (optionally protein-coding only), indexed 0..n-1 in
    (chr, start, end, Ensembl id) order."""

    def __init__(self, genes: pl.DataFrame, protein_coding: bool) -> None:
        g = genes.drop_nulls(["chr", "gene_start", "gene_end"])
        if protein_coding:
            g = g.filter(pl.col("locus_group") == "protein-coding gene")
        g = g.sort("chr", "gene_start", "gene_end", "ensembl_gene_id")
        self.ensembl = g["ensembl_gene_id"].to_list()
        self.symbol = [s or e for s, e in zip(g["symbol"].to_list(), self.ensembl, strict=True)]
        self.index = {e: i for i, e in enumerate(self.ensembl)}
        chrom = g["chr"].to_numpy()
        start = g["gene_start"].to_numpy().astype(np.int64)
        end = g["gene_end"].to_numpy().astype(np.int64)
        self.by_chr: dict[int, _Chrom] = {}
        for c in np.unique(chrom):
            lo, hi = np.searchsorted(chrom, c, "left"), np.searchsorted(chrom, c, "right")
            self.by_chr[int(c)] = _Chrom(int(lo), start[lo:hi], end[lo:hi])

    def map(
        self, chrom: np.ndarray, pos: np.ndarray, mode: str, window: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(query index, gene index, distance) under the mapping rule."""
        qs, gs, ds = [], [], []
        for c in np.unique(chrom):
            ch = self.by_chr.get(int(c))
            if ch is None:
                continue
            sel = np.flatnonzero(chrom == c)
            if mode == "nearest":
                g, d = ch.nearest(pos[sel])
                q = sel
            else:
                qi, g, d = ch.window(pos[sel], window)
                q = sel[qi]
            qs.append(q)
            gs.append(g + ch.offset)
            ds.append(d)
        if not qs:
            e = np.empty(0, dtype=np.int64)
            return e, e, e
        return np.concatenate(qs), np.concatenate(gs), np.concatenate(ds)


def _coords(bundle: _Bundle, protein_coding: bool) -> _Coords:
    if protein_coding not in bundle.coords:
        genes = pl.read_parquet(bundle.path / "genes.parquet")
        bundle.coords[protein_coding] = _Coords(genes, protein_coding)
    return bundle.coords[protein_coding]


def _chrom_code(value: Any) -> int | None:
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        value = str(int(value))
    if not isinstance(value, str):
        return None
    v = value.strip().upper()
    return _CHROMS.get(v[3:] if v.startswith("CHR") else v)


def _leads(loci: Any, id_col: str | None) -> pl.DataFrame:
    """Lead variants as (id, chr, pos), chr coded 1-22, X = 23, Y = 24, MT = 26."""
    if isinstance(loci, pl.DataFrame):
        missing = [c for c in ("chr", "pos") if c not in loci.columns]
        if missing:
            raise GeneticsUsageError(
                f"loci needs columns 'chr' and 'pos' (and optionally an id column); it has "
                f"{loci.columns}"
            )
        if id_col is None:
            id_col = next((c for c in _LEAD_ID_COLUMNS if c in loci.columns), None)
        elif id_col not in loci.columns:
            raise GeneticsUsageError(f"id_col {id_col!r} is not a column of loci")
        if loci["chr"].null_count() or loci["pos"].null_count():
            raise GeneticsUsageError("loci holds a null chr or pos")
        if not loci["pos"].dtype.is_integer():
            raise GeneticsUsageError(f"loci's pos must be integers, got {loci['pos'].dtype}")
        raw_chr = loci["chr"].to_list()
        pos = loci["pos"].cast(pl.Int64).to_list()
        ids = (
            [str(v) for v in loci[id_col].to_list()]
            if id_col
            else [f"{c}:{p}" for c, p in zip(raw_chr, pos, strict=True)]
        )
    elif isinstance(loci, (list, tuple, pl.Series)) and not isinstance(loci, str):
        ids, raw_chr, pos = [], [], []
        for v in loci:
            m = _VARIANT.fullmatch(v.strip()) if isinstance(v, str) else None
            if m is None:
                raise GeneticsUsageError(
                    f"cannot read {v!r} as a variant: give 'chr:pos', 'chr:pos:ref:alt' or "
                    "'chr_pos_ref_alt' strings, or a DataFrame with chr and pos columns"
                )
            ids.append(v)
            raw_chr.append(m.group(1))
            pos.append(int(m.group(2)))
    else:
        raise GeneticsUsageError(
            "loci must be a polars DataFrame with chr and pos columns or a list of variant "
            f"strings like '19:44908684:T:C'; got {type(loci).__name__}"
        )
    if not ids:
        raise GeneticsUsageError("loci is empty")
    chrom = [_chrom_code(c) for c in raw_chr]
    bad = sorted({str(r) for r, c in zip(raw_chr, chrom, strict=True) if c is None})
    if bad:
        raise GeneticsUsageError(
            f"unrecognised chromosome(s) {bad[:5]}: use 1-22, X (or 23), Y, MT, with or "
            "without a 'chr' prefix"
        )
    if min(pos) < 1:
        raise GeneticsUsageError("loci positions must be >= 1")
    return pl.DataFrame(
        {"id": ids, "chr": chrom, "pos": pos},
        schema={"id": pl.Utf8, "chr": pl.Int64, "pos": pl.Int64},
    )


def _check_loci_args(mode: str, window: Any, merge_distance: Any) -> None:
    if mode not in _LOCUS_MODES:
        raise GeneticsUsageError(f"mode must be one of {_LOCUS_MODES}, got {mode!r}")
    for name, v in (("window", window), ("merge_distance", merge_distance)):
        if not _is_int(v) or v < 0:
            raise GeneticsUsageError(f"{name} must be a non-negative integer, got {v!r}")


def _in_mhc(leads: pl.DataFrame) -> pl.Series:
    c, lo, hi = _MHC
    return (leads["chr"] == c) & leads["pos"].is_between(lo, hi)


def _merge(leads: pl.DataFrame, merge_distance: int) -> pl.DataFrame:
    """Adds `locus` (0-based, genomic order): single-linkage over sorted positions."""
    leads = leads.sort("chr", "pos", "id")
    chrom, pos = leads["chr"].to_numpy(), leads["pos"].to_numpy()
    new = np.ones(pos.size, dtype=bool)
    if merge_distance > 0:
        new[1:] = (chrom[1:] != chrom[:-1]) | (np.diff(pos) > merge_distance)
    return leads.with_columns(locus=pl.Series(np.cumsum(new) - 1, dtype=pl.Int64))


def _coordinates_meta(bundle: _Bundle) -> dict[str, Any]:
    raw = json.loads((bundle.path / "manifest.json").read_text())
    gencode = (raw.get("bundle", {}).get("gencode") or {}).get("version")
    return {"release_id": bundle.release_id, "gencode": gencode, "genome_build": _GENOME_BUILD}


def _map_loci(
    leads: pl.DataFrame, coords: _Coords, mode: str, window: int, merge_distance: int
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """(leads with locus, loci frame, genes frame) for leads already filtered."""
    leads = _merge(leads, merge_distance)
    qi, gi, dist = coords.map(leads["chr"].to_numpy(), leads["pos"].to_numpy(), mode, window)
    locus = leads["locus"].to_numpy()
    genes = (
        pl.DataFrame(
            {"locus": locus[qi], "gene": gi.astype(np.int64), "distance": dist.astype(np.int64)}
        )
        .group_by("locus", "gene")
        .agg(pl.col("distance").min())
    )
    genes = genes.with_columns(
        ensembl_gene_id=pl.Series(coords.ensembl, dtype=pl.Utf8).gather(genes["gene"]),
        symbol=pl.Series(coords.symbol, dtype=pl.Utf8).gather(genes["gene"]),
    ).sort("locus", "distance", "symbol")
    label = pl.col("chr").replace_strict(
        _CHROM_LABEL, default=pl.col("chr").cast(pl.Utf8), return_dtype=pl.Utf8
    )
    loci = (
        leads.group_by("locus", maintain_order=True)
        .agg(
            pl.col("chr").first(),
            start=pl.col("pos").min(),
            end=pl.col("pos").max(),
            n_leads=pl.len().cast(pl.Int64),
            lead_ids=pl.col("id"),
        )
        .with_columns(locus_id=pl.format("{}:{}-{}", label, pl.col("start"), pl.col("end")))
        # identical positions stay separate loci only with merge_distance=0
        .with_columns(
            locus_id=pl.when(pl.col("locus_id").cum_count().over("locus_id") > 1)
            .then(pl.format("{}#{}", "locus_id", pl.col("locus_id").cum_count().over("locus_id")))
            .otherwise("locus_id")
        )
        .join(
            genes.group_by("locus", maintain_order=True).agg(gene_symbols=pl.col("symbol")),
            on="locus",
            how="left",
        )
        .with_columns(pl.col("gene_symbols").fill_null(pl.lit([], dtype=pl.List(pl.Utf8))))
        .sort("locus")
    )
    genes = genes.join(loci.select("locus", "locus_id"), on="locus").select(
        "locus", "locus_id", "gene", "ensembl_gene_id", "symbol", "distance"
    )
    return leads, loci, genes


def genes_for_loci(
    loci: Any,
    *,
    mode: str = "nearest",
    window: int = 50_000,
    merge_distance: int = 500_000,
    protein_coding: bool = True,
    exclude_mhc: bool = False,
    id_col: str | None = None,
) -> dict[str, Any]:
    """Map lead variants to genes with the bundle's gene coordinates (no network).

    `loci` — a polars DataFrame with `chr` (1-22, X/23, Y, MT; a `chr` prefix is fine) and
    `pos` (GRCh38) columns, plus an optional id column carried through (`id_col`; by default
    the first of `id`, `cs_id`, `variant` present), or a list of variant strings such as
    `"19:44908684"`, `"19:44908684:T:C"` or `"chr19_44908684_T_C"`.
    `merge_distance` — leads on one chromosome within this many bp of the next one form one
    locus (single linkage on sorted positions), so several credible sets at one signal count
    once; 0 keeps every lead its own locus.
    `mode` — `"nearest"`: each lead's nearest gene (distance 0 inside a gene body, otherwise to
    the nearer end; ties go to the gene earliest by start, then end, then Ensembl id — so a gene
    nested inside a longer one that starts earlier is never a lead's nearest gene; window mode
    sees it). A locus's genes are the union over its leads. `"window"`: every gene whose body
    overlaps [pos - window, pos + window] of any lead.
    `protein_coding` — map to protein-coding genes only, as HGNC classifies them
    (`locus_group == "protein-coding gene"`, the bundle's gene identity being HGNC's).
    `exclude_mhc` — drop leads in GRCh38 chr6:25,000,000-34,000,000 (counted in parameters).

    Returns `{"loci", "genes", "parameters"}`. `loci`: one row per locus — `locus_id`
    (`chr:start-end` of its leads), `chr` (23 = X), `start`, `end`, `n_leads`, `lead_ids`,
    `gene_symbols` (nearest first). `genes`: one row per (locus, gene) — `locus_id`,
    `ensembl_gene_id`, `symbol`, `distance` (bp from the closest lead). `parameters`: the
    mapping settings, lead and locus counts, and the bundle release and genome build the
    coordinates come from.
    """
    _check_loci_args(mode, window, merge_distance)
    leads = _leads(loci, id_col)
    bundle = _bundle()
    coords = _coords(bundle, bool(protein_coding))
    mhc = _in_mhc(leads) if exclude_mhc else pl.Series([False] * leads.height)
    leads = leads.filter(~mhc)
    if leads.height == 0:
        raise GeneticsUsageError("no leads remain after dropping the MHC")
    _, loci_df, genes = _map_loci(leads, coords, mode, int(window), int(merge_distance))
    return {
        "loci": loci_df.drop("locus"),
        "genes": genes.drop("locus", "gene"),
        "parameters": {
            "mode": mode,
            "window": int(window) if mode == "window" else None,
            "merge_distance": int(merge_distance),
            "protein_coding": bool(protein_coding),
            "gene_type": _PROTEIN_CODING if protein_coding else "all genes with coordinates",
            "exclude_mhc": bool(exclude_mhc),
            "mhc_region": f"chr{_MHC[0]}:{_MHC[1]}-{_MHC[2]}",
            "leads_given": int(mhc.len()),
            "leads_dropped_mhc": int(mhc.sum()),
            "n_loci": loci_df.height,
            "loci_without_gene": int((loci_df["gene_symbols"].list.len() == 0).sum()),
            "n_genes": genes["gene"].n_unique(),
            "coordinates": _coordinates_meta(bundle),
        },
    }


def _null_q(
    coords: _Coords,
    mt: sparse.csr_matrix,
    chrom: np.ndarray,
    centre: np.ndarray,
    offsets: np.ndarray,
    group: np.ndarray,
    n_groups: int,
    mode: str,
    window: int,
) -> np.ndarray:
    """(n_groups, n_sets): per group of placements, the fraction whose mapped genes hit each
    set. A placement puts a locus's leads at `centre + offsets`.

    Placements with the same gene tuple in one group are counted once with a weight, which is
    what keeps a 30k-placement group to a few thousand rows."""
    n_lead = offsets.size
    qi, gi, _ = coords.map(
        np.repeat(chrom, n_lead), (centre[:, None] + offsets[None, :]).ravel(), mode, window
    )
    out = np.zeros((n_groups, mt.shape[1]))
    if qi.size == 0:
        return out
    place = qi // n_lead
    per_place = (
        pl.DataFrame({"place": place, "gene": gi.astype(np.int64)})
        .unique()
        .group_by("place")
        .agg(pl.col("gene").sort())
    )
    tuples = (
        per_place.with_columns(group=group[per_place["place"].to_numpy()])
        .group_by("group", "gene")
        .len()
        .with_row_index("r")
    )
    flat = tuples.select("r", "gene").explode("gene")
    t = sparse.csr_matrix(
        (np.ones(flat.height), (flat["r"].to_numpy(), flat["gene"].to_numpy())),
        shape=(tuples.height, mt.shape[0]),
    )
    hit = (t @ mt).tocsr()
    hit.data[:] = 1.0
    n = np.bincount(group, minlength=n_groups).astype(np.float64)
    g = tuples["group"].to_numpy()
    w = sparse.csr_matrix(
        (tuples["len"].to_numpy() / n[g], (g, tuples["r"].to_numpy())),
        shape=(n_groups, tuples.height),
    )
    return (w @ hit).toarray()


def _poisson_binomial_sf(q: np.ndarray, obs: np.ndarray, chunk: int = 1024) -> np.ndarray:
    """P(X >= obs) for X = sum of independent Bernoulli(q[i, s]) over rows, per column s.

    Dynamic programming over rows with states 0..K-1 exact and K absorbing "at least K"; the
    tail is then a sum of non-negative terms, so tiny p keeps its relative precision where
    1 - CDF would cancel to zero. Columns are processed sorted by `obs`, so a chunk's K is its
    own largest observation rather than the call's."""
    out = np.ones(obs.size)
    todo = np.flatnonzero(obs > 0)
    todo = todo[np.argsort(obs[todo], kind="stable")]
    for c0 in range(0, todo.size, chunk):
        cols = todo[c0 : c0 + chunk]
        k_max = int(obs[cols].max())
        dp = np.zeros((k_max + 1, cols.size))
        dp[0] = 1.0
        for qi in q[:, cols]:
            top = dp[k_max] + dp[k_max - 1] * qi
            if k_max > 1:
                dp[1:k_max] = dp[1:k_max] * (1 - qi) + dp[: k_max - 1] * qi
            dp[0] *= 1 - qi
            dp[k_max] = top
        tail = np.cumsum(dp[::-1], axis=0)[::-1]
        out[cols] = np.minimum(1.0, tail[obs[cols], np.arange(cols.size)])
    return out


def _quantile_edges(density: np.ndarray, n_bins: int) -> np.ndarray:
    return np.unique(np.quantile(density, np.linspace(0, 1, n_bins + 1)[1:-1], method="lower"))


def locus_enrichment(
    loci: Any,
    gene_sets: Mapping[str, Iterable[str]] | str | list[str],
    *,
    mode: str = "nearest",
    window: int = 50_000,
    merge_distance: int = 500_000,
    density_window: int = 500_000,
    protein_coding: bool = True,
    exclude_mhc: bool = True,
    min_size: int = 5,
    max_size: int = 300,
    grid_step: int = 10_000,
    density_bins: int = 20,
    id_col: str | None = None,
) -> dict[str, Any]:
    """Locus-level gene set enrichment for GWAS leads, matched on local gene density
    (INRICH-style). Use this, not `ora` on nearest genes or `competitive` on credible-set z,
    when all you have is lead variants: it counts loci rather than genes, so a cluster of
    related genes at one signal (APOE/APOC1/TOMM40) counts once, and it compares each locus
    with random loci of matched local gene density. Density matters both ways: with nearest
    genes, an isolated gene in a sparse region is the nearest gene of a large territory and is
    picked by chance far more often than a gene in a dense cluster; with a window or a
    multi-lead locus, a locus in a dense region maps to more genes and hits more sets.

    `loci`, `mode`, `window`, `merge_distance`, `protein_coding`, `id_col` — as
    `genes_for_loci`. `exclude_mhc` defaults to True here (False there): leads in GRCh38
    chr6:25-34 Mb are dropped and the region is left out of the null. Leads on chromosomes
    other than 1-22 and X are dropped and counted (`genes_for_loci` keeps Y and MT). So
    `genes_for_loci(loci, exclude_mhc=True)` with the same mapping arguments shows the mapping
    used here, once any Y and MT leads are left out.
    `gene_sets` — a shipped collection name or list of names, or a mapping of set id -> gene
    ids (symbols, Entrez or Ensembl, resolved through the bundle's gene table, since the loci
    map to the bundle's genes). Sets are restricted to genes with coordinates of the gene type
    before the `min_size`/`max_size` filter.

    The test, per set S: the observed statistic is the number of loci with at least one mapped
    gene in S. Each locus i is compared with every placement of a locus of the same shape (its
    leads' offsets from the midpoint of its first and last lead) centred on a `grid_step` grid,
    among grid points of the same chromosome class (autosomes, or X for a locus on X) and the
    same gene-density stratum: density is the number of genes of the gene type overlapping
    centre ± `density_window`, and the strata are `density_bins` quantile bins of the
    autosomal grid's density (fewer where quantiles tie), so autosomal results do not depend on
    whether a locus is on X. A multi-lead placement is dropped when any of its leads would fall
    outside the chromosome's gene span or, with `exclude_mhc`, in the MHC. Grid points with no gene
    within `density_window` are left out — the bundle cannot tell assembly gaps and centromeres,
    where no lead can sit, from gene deserts; a real locus there is matched to the lowest
    stratum. q_iS is the fraction of placements, counting the real one, that hit S, and the
    p-value is the exact upper tail P(X >= observed) of X = sum of independent Bernoulli(q_iS)
    (Poisson-binomial), so it is not floored by a permutation count and survives BH over
    thousands of sets. Approximations: loci are treated as independent, and the null ignores
    that real loci cannot overlap one another. `padj` is BH over tested sets.

    Returns `{"results", "loci", "parameters"}` (and `"attribution"` with a collection name).
    `results`, one row per tested set ordered by p: `set_id`, `size`, `n_loci_hit`, `expected`
    (sum of q_iS), `fold_enrichment`, `pval`, `padj`, `hit_loci` (locus ids), `hit_genes` and
    `hit_symbols`; with a collection name also `name` and `collection`. `loci` is
    `genes_for_loci`'s frame plus each locus's `density` and `density_bin`. `parameters` states
    the mapping, the null (grid, density window, bin edges, placements per stratum,
    independence), lead and locus counts, set members dropped for lacking coordinates or being
    of another gene type, the coordinates' release and build, and the size filter.
    """
    _check_loci_args(mode, window, merge_distance)
    _check_sizes(min_size, max_size)
    for name, v, lo in (
        ("density_window", density_window, 0),
        ("grid_step", grid_step, 1),
        ("density_bins", density_bins, 1),
    ):
        if not _is_int(v) or v < lo:
            raise GeneticsUsageError(f"{name} must be an integer >= {lo}, got {v!r}")
    window, merge_distance, density_window = int(window), int(merge_distance), int(density_window)
    leads = _leads(loci, id_col)
    bundled = _bundled(gene_sets)
    bundle = bundled.bundle if bundled else _bundle()
    coords = _coords(bundle, bool(protein_coding))

    n_given = leads.height
    mhc = _in_mhc(leads) if exclude_mhc else pl.Series([False] * n_given)
    leads = leads.filter(~mhc)
    usable = leads["chr"].is_between(1, 23) & leads["chr"].is_in(list(coords.by_chr))
    dropped_chrom = int((~usable).sum())
    leads = leads.filter(usable)
    if leads.height == 0:
        raise GeneticsUsageError(
            "no leads remain after dropping the MHC and chromosomes other than 1-22 and X"
        )
    leads, loci_df, obs_genes = _map_loci(leads, coords, mode, window, merge_distance)
    n_loci = loci_df.height

    # sets over the genes with coordinates of the gene type
    set_ids, flat = bundled.flat() if bundled else _flatten(gene_sets)
    if not bundled:
        uniq = flat["gene"].unique(maintain_order=True)
        resolved, report = bundle.genes.resolve(uniq.to_list(), "gene_sets")
        del report["what"]
        flat = (
            flat.join(
                pl.DataFrame({"gene": uniq, "ens": pl.Series(resolved, dtype=pl.Utf8)}), on="gene"
            )
            .select("row", pl.col("ens").alias("gene"))
            .drop_nulls()
        )
    m = _membership(flat, len(set_ids), coords.index)
    unplaced = flat.filter(~pl.col("gene").is_in(list(coords.index)))
    sizes = np.diff(m.indptr)
    tested = np.flatnonzero((sizes >= min_size) & (sizes <= max_size))
    mt = m[tested].T.tocsr().astype(np.float64)

    # observed: loci x sets, a locus counting once however many of its genes are in a set
    lg = sparse.csr_matrix(
        (np.ones(obs_genes.height), (obs_genes["locus"].to_numpy(), obs_genes["gene"].to_numpy())),
        shape=(n_loci, len(coords.ensembl)),
    )
    obs_hit = (lg @ mt).tocsr()
    obs_hit.data[:] = 1.0
    obs = np.asarray(obs_hit.sum(axis=0)).ravel().astype(np.int64)

    # the null grid; X is gridded only for a locus on X, and is its own class, so autosomal
    # strata and placements are the same whether or not an X locus is present
    chroms = list(range(1, 23)) + ([23] if (loci_df["chr"] == 23).any() else [])
    chroms = [c for c in chroms if c in coords.by_chr]
    span = {c: (int(coords.by_chr[c].start[0]), int(coords.by_chr[c].end.max())) for c in chroms}
    g_chr, g_pos, g_den = [], [], []
    for c in chroms:
        ch = coords.by_chr[c]
        p = np.arange(span[c][0], span[c][1] + 1, int(grid_step), dtype=np.int64)
        d = ch.density(p, density_window)
        keep = d > 0
        if exclude_mhc and c == _MHC[0]:
            keep &= ~((p >= _MHC[1]) & (p <= _MHC[2]))
        g_chr.append(np.full(int(keep.sum()), c, dtype=np.int64))
        g_pos.append(p[keep])
        g_den.append(d[keep])
    g_chr, g_pos, g_den = np.concatenate(g_chr), np.concatenate(g_pos), np.concatenate(g_den)
    g_x = g_chr == 23
    edges = _quantile_edges(g_den[~g_x] if (~g_x).any() else g_den, int(density_bins))
    n_bins = edges.size + 1
    # stratum key: chromosome class * n_bins + density bin
    g_key = np.searchsorted(edges, g_den, "right") + n_bins * g_x
    per_key = np.bincount(g_key, minlength=2 * n_bins)

    # each locus by its leads' offsets from its centre
    lead_pos = leads.group_by("locus", maintain_order=True).agg(pl.col("pos")).sort("locus")
    centre = ((loci_df["start"] + loci_df["end"]) // 2).to_numpy()
    l_chr = loci_df["chr"].to_numpy()
    l_den = np.array(
        [
            int(coords.by_chr[int(c)].density(np.array([x]), density_window)[0])
            for c, x in zip(l_chr, centre, strict=True)
        ]
    )
    l_bin = np.searchsorted(edges, l_den, "right")
    l_key = l_bin + n_bins * (l_chr == 23)
    for i in np.flatnonzero(per_key[l_key] == 0):
        # an empty stratum (bin 0 always is; X, on autosomal edges, can have others): the
        # nearest non-empty bin of the locus's class, the lower one on a tie
        cls, b = divmod(int(l_key[i]), n_bins)
        filled = np.flatnonzero(per_key[cls * n_bins : (cls + 1) * n_bins])
        l_key[i] = cls * n_bins + int(filled[np.argmin(np.abs(filled - b))])
    l_bin = l_key % n_bins

    q = np.empty((n_loci, tested.size))
    n_place = per_key[l_key].astype(np.float64)
    single = loci_df["n_leads"].to_numpy() == 1
    if single.any():
        sel = np.flatnonzero(np.isin(g_key, np.unique(l_key[single])))
        q_key = _null_q(
            coords,
            mt,
            g_chr[sel],
            g_pos[sel],
            np.zeros(1, dtype=np.int64),
            g_key[sel],
            2 * n_bins,
            mode,
            window,
        )
        q[single] = q_key[l_key[single]]
    clipped = 0
    for i in np.flatnonzero(~single):
        offsets = np.asarray(lead_pos["pos"][int(i)].to_list(), dtype=np.int64) - centre[i]
        sel = np.flatnonzero(g_key == l_key[i])
        # a lead placed past either end gene maps to that end gene whatever the offset, which
        # would inflate the end gene's q; and with exclude_mhc no lead may sit in the MHC, as
        # no real lead does
        at = g_pos[sel, None] + offsets[None, :]
        lo = np.array([span[int(c)][0] for c in g_chr[sel]])[:, None]
        hi = np.array([span[int(c)][1] for c in g_chr[sel]])[:, None]
        ok = ((at >= lo) & (at <= hi)).all(axis=1)
        if exclude_mhc:
            ok &= ~((g_chr[sel, None] == _MHC[0]) & (at >= _MHC[1]) & (at <= _MHC[2])).any(axis=1)
        clipped += int((~ok).sum())
        sel = sel[ok]
        n_place[i] = sel.size
        q[i] = (
            _null_q(
                coords,
                mt,
                g_chr[sel],
                g_pos[sel],
                offsets,
                np.zeros(sel.size, dtype=np.int64),
                1,
                mode,
                window,
            )[0]
            if sel.size
            else 0.0
        )
    # each locus's own position counts as one of its placements, so q_iS > 0 for every set
    # the locus hits and no p-value can be 0
    q *= n_place[:, None]
    own = obs_hit.tocoo()
    q[own.row, own.col] += 1.0
    q /= n_place[:, None] + 1

    pval = _poisson_binomial_sf(q, obs)
    expected = q.sum(axis=0)

    hit_coo = obs_hit.tocoo()
    locus_ids = loci_df["locus_id"]
    order = np.lexsort((hit_coo.row, hit_coo.col))
    hit_loci = (
        pl.DataFrame(
            {
                "col": hit_coo.col[order].astype(np.int64),
                "locus_id": locus_ids.gather(hit_coo.row[order]),
            }
        )
        .group_by("col", maintain_order=True)
        .agg(hit_loci=pl.col("locus_id"))
    )
    observed_genes = np.zeros(len(coords.ensembl), dtype=bool)
    observed_genes[obs_genes["gene"].to_numpy()] = True
    sub = m[tested].tocoo()
    keep = observed_genes[sub.col]
    hit_genes = (
        pl.DataFrame(
            {
                "col": sub.row[keep].astype(np.int64),
                "gene": pl.Series(coords.ensembl, dtype=pl.Utf8).gather(sub.col[keep]),
            }
        )
        .sort("col", "gene")
        .group_by("col", maintain_order=True)
        .agg(hit_genes=pl.col("gene"))
    )
    empty = pl.lit([], dtype=pl.List(pl.Utf8))
    with np.errstate(divide="ignore", invalid="ignore"):
        fold = obs / expected
    results = (
        pl.DataFrame(
            {
                "col": np.arange(tested.size, dtype=np.int64),
                "set_id": pl.Series([set_ids[i] for i in tested], dtype=pl.Utf8),
                "size": sizes[tested].astype(np.int64),
                "n_loci_hit": obs,
                "expected": expected,
                "fold_enrichment": fold,
                "pval": pval,
                "padj": _bh(pval),
            }
        )
        .join(hit_loci, on="col", how="left")
        .join(hit_genes, on="col", how="left")
        .with_columns(pl.col("hit_loci").fill_null(empty), pl.col("hit_genes").fill_null(empty))
        .drop("col")
        .sort(["pval", "set_id"])
    )
    if not bundled:
        sym = dict(zip(coords.ensembl, coords.symbol, strict=True))
        results = results.with_columns(
            hit_symbols=pl.col("hit_genes").list.eval(pl.element().replace(sym))
        )
    multi = int((~single).sum())
    out = {
        "results": results,
        "loci": loci_df.drop("locus").with_columns(
            density=pl.Series(l_den, dtype=pl.Int64), density_bin=pl.Series(l_bin, dtype=pl.Int64)
        ),
        "parameters": {
            "mode": mode,
            "window": window if mode == "window" else None,
            "merge_distance": merge_distance,
            "protein_coding": bool(protein_coding),
            "gene_type": _PROTEIN_CODING if protein_coding else "all genes with coordinates",
            "exclude_mhc": bool(exclude_mhc),
            "mhc_region": f"chr{_MHC[0]}:{_MHC[1]}-{_MHC[2]}",
            "leads_given": n_given,
            "leads_dropped_mhc": int(mhc.sum()),
            "leads_dropped_chromosome": dropped_chrom,
            "n_loci": n_loci,
            "n_loci_multi_lead": multi,
            "null": {
                "statistic": "loci with >= 1 mapped gene in the set",
                "placements": (
                    "grid points of the locus's chromosome class (autosomes or X) and density "
                    "stratum, plus the locus itself"
                ),
                "grid_step": int(grid_step),
                "grid_chromosomes": [_CHROM_LABEL.get(c, str(c)) for c in chroms],
                "grid_points": int(g_pos.size),
                "density_window": density_window,
                "density": "genes of the gene type overlapping centre ± density_window",
                "density_bins": n_bins,
                "density_bin_edges": edges.astype(np.int64).tolist(),
                "edges_from": "autosomal grid",
                "placements_per_bin": per_key[:n_bins].astype(np.int64).tolist(),
                "placements_per_bin_x": (
                    per_key[n_bins:].astype(np.int64).tolist() if 23 in chroms else None
                ),
                "multi_lead_placements_dropped": clipped,
                "p_value": "exact Poisson-binomial upper tail over loci",
                "approximations": (
                    "loci independent; the null ignores that real loci cannot overlap"
                ),
            },
            "n_genes": len(coords.ensembl),
            "set_members_dropped": {
                "reason": "no coordinates in the bundle, or not of the gene type",
                "memberships": unplaced.height,
                "genes": unplaced["gene"].n_unique(),
            },
            "min_size": int(min_size),
            "max_size": int(max_size),
            "sets_given": len(set_ids),
            "sets_tested": int(tested.size),
            "coordinates": _coordinates_meta(bundle),
        },
    }
    if bundled:
        return bundled.finish(out, ("hit_genes",))
    out["parameters"]["id_resolution"] = {"gene_sets": report}
    return out


_HIT_COLUMNS = ("overlap_genes", "hit_genes", "leading_edge")
_SIMPLIFY_BASES = ("hits", "members")
_SIMPLIFY_MEASURES = ("containment", "overlap", "jaccard")


def simplify(
    result: Mapping[str, Any],
    *,
    padj: float = 0.05,
    cutoff: float = 0.7,
    basis: str = "hits",
    measure: str = "containment",
    gene_sets: Mapping[str, Iterable[str]] | None = None,
) -> dict[str, Any]:
    """Collapse redundant significant sets in an `ora`, `gsea_preranked`, `competitive` or
    `locus_enrichment` result.

    The sets with `padj <= padj` are taken in order of p (then padj, then set id); the best is
    kept as a representative and every remaining set whose similarity to it (`measure`) is at
    least `cutoff` is absorbed into it; then the best remaining set, and so on.

    `basis` — which genes are compared. `"hits"` (default): the genes driving each set's
    result — `ora`'s `overlap_genes`, `locus_enrichment`'s `hit_genes`, `gsea_preranked`'s
    `leading_edge` — so sets significant because of the same hits collapse. A result with no
    per-set hits (`competitive`) falls back to `"members"`, and `parameters` says so.
    `"members"`: the sets' full membership, from the bundle for a result over shipped
    collections or from `gene_sets` (set id -> gene ids) for one over a caller's mapping, not
    restricted to the genes that analysis tested.
    `measure` — how much of candidate A the representative R already accounts for.
    `"containment"` (default): |A ∩ R| / |A|, the fraction of A's own genes R holds — so a set
    is absorbed only into a representative that explains most of it, never into a better-p
    set whose few genes merely nest inside it. `"overlap"`: |A ∩ R| / min(|A|, |R|), symmetric,
    which does absorb a large set into a small nested one. `"jaccard"`: |A ∩ R| / |A ∪ R|.

    The bundle holds no GO graph, so gene overlap is the only similarity available: nested GO
    terms collapse because their genes nest, not because of the ontology.

    Returns a copy of `result` whose `results` holds the representatives only, with
    `n_absorbed`, `absorbed_ids` (and `absorbed_names` where the result has names) added;
    `clusters`, one row per significant set: `set_id`, `representative`, `similarity` (to its
    representative); and `parameters["simplify"]` recording `padj`, `cutoff`, `basis` (asked
    and used), `measure` and where the genes came from.
    """
    if not isinstance(result, Mapping) or not isinstance(result.get("results"), pl.DataFrame):
        raise GeneticsUsageError("simplify takes the dict an enrichment function returned")
    res = result["results"]
    missing_cols = [c for c in ("set_id", "pval", "padj") if c not in res.columns]
    if missing_cols:
        raise GeneticsUsageError(
            f"the result's results need set_id, pval and padj columns; missing {missing_cols}"
        )
    for name, v in (("padj", padj), ("cutoff", cutoff)):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 < v <= 1:
            raise GeneticsUsageError(f"{name} must be a number in (0, 1], got {v!r}")
    if basis not in _SIMPLIFY_BASES:
        raise GeneticsUsageError(f"basis must be one of {_SIMPLIFY_BASES}, got {basis!r}")
    if measure not in _SIMPLIFY_MEASURES:
        raise GeneticsUsageError(f"measure must be one of {_SIMPLIFY_MEASURES}, got {measure!r}")
    if gene_sets is not None and not isinstance(gene_sets, Mapping):
        raise GeneticsUsageError("gene_sets= must be a mapping of set id -> gene ids")
    sig = res.filter(pl.col("padj") <= padj).sort("pval", "padj", "set_id")
    ids = sig["set_id"].to_list()

    hit_col = next((c for c in _HIT_COLUMNS if c in res.columns), None)
    used = basis if hit_col else "members"
    if used == "hits":
        flat = (
            sig.select(pl.col(hit_col).alias("gene"))
            .with_row_index("row")
            .explode("gene")
            .drop_nulls()
            .select(pl.col("row").cast(pl.Int64), "gene")
        )
        source = f"the result's {hit_col}"
    elif gene_sets is not None:
        missing = [s for s in ids if s not in gene_sets]
        if missing:
            raise GeneticsUsageError(f"gene_sets has no members for {missing[:3]}")
        _, flat = _flatten({s: gene_sets[s] for s in ids})
        source = "gene_sets mapping as given"
    elif "attribution" in result and "collection" in res.columns:
        names = sorted(set(sig["collection"].to_list()))
        bundle = _bundle()
        bundle.check_names(names)
        rows = pl.DataFrame({"set_id": pl.Series(ids, dtype=pl.Utf8)}).with_row_index("row")
        members = (
            pl.concat([bundle.members(n) for n in names])
            if names
            else pl.DataFrame(schema={"set_id": pl.Utf8, "ensembl_gene_id": pl.Utf8})
        )
        flat = members.join(rows, on="set_id").select(
            pl.col("row").cast(pl.Int64), pl.col("ensembl_gene_id").alias("gene")
        )
        source = "full bundled membership"
    else:
        raise GeneticsUsageError(
            "this result is over a caller's mapping: pass the same mapping as gene_sets="
        )
    genes = flat["gene"].unique(maintain_order=True).to_list()
    m = _membership(flat, len(ids), {g: i for i, g in enumerate(genes)}).astype(np.float64)
    size = np.asarray(m.sum(axis=1)).ravel()

    rep_of = np.full(len(ids), -1)
    sim = np.zeros(len(ids))
    for i in range(len(ids)):
        if rep_of[i] >= 0:
            continue
        rep_of[i], sim[i] = i, 1.0
        inter = np.asarray((m @ m[i].T).todense()).ravel()
        denom = {
            "containment": size,
            "overlap": np.minimum(size, size[i]),
            "jaccard": size + size[i] - inter,
        }[measure]
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(denom > 0, inter / denom, 0.0)
        take = (rep_of < 0) & (s >= cutoff)
        rep_of[take], sim[take] = i, s[take]

    clusters = pl.DataFrame(
        {
            "set_id": pl.Series(ids, dtype=pl.Utf8),
            "representative": pl.Series([ids[r] for r in rep_of], dtype=pl.Utf8),
            "similarity": sim,
        }
    )
    named = "name" in sig.columns
    other = pl.col("set_id") != pl.col("representative")
    agg = [pl.col("set_id").filter(other).alias("absorbed_ids")]
    if named:
        agg.append(pl.col("name").filter(other).alias("absorbed_names"))
    absorbed = (
        clusters.join(sig.select(["set_id"] + (["name"] if named else [])), on="set_id")
        .group_by("representative", maintain_order=True)
        .agg(*agg)
        .with_columns(n_absorbed=pl.col("absorbed_ids").list.len().cast(pl.Int64))
    )
    reps = sig.join(absorbed, left_on="set_id", right_on="representative", how="inner")
    out = dict(result)
    out["results"] = reps.select(
        list(sig.columns) + ["n_absorbed", "absorbed_ids"] + (["absorbed_names"] if named else [])
    ).sort("pval", "padj", "set_id")
    out["clusters"] = clusters
    out["parameters"] = {
        **result.get("parameters", {}),
        "simplify": {
            "padj": float(padj),
            "cutoff": float(cutoff),
            "basis": basis,
            "basis_used": used,
            "measure": measure,
            "genes_from": source,
            "n_significant": len(ids),
            "n_representatives": out["results"].height,
        },
    }
    return out
