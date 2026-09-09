"""Measured suite data placed beside AlphaGenome's prediction for the same variant.

This module is the second of the two AlphaGenome capabilities and the one where the
labelling discipline stops being a footnote: it deliberately puts a model's number next
to a person's measurement in one response, so every value here carries its own `measured`
flag and, when measured, the view and column it came from. Nothing is merged, averaged or
reduced to a single figure.

Three rules are measured facts rather than presentation choices, and each is enforced
structurally:

1. THE QUANTITY RULE FROM `alphagenome.MODALITIES` HOLDS FOR THE COMPARISON TOO. Where the
   sign is commensurable with what the suite measured (accessibility, histone, expression)
   both sides are signed and a direction agreement is reported. Where it is not (all three
   splicing modalities) BOTH sides are reduced to magnitude and the response carries no
   `direction` key at all -- not a null one. sQTL beta orients to a leafcutter intron
   cluster and the splice delta has no corresponding orientation, so a stated agreement
   would be a fabricated claim, and handing back the two signs would let a reader
   fabricate it themselves.

2. CELL-TYPE MATCHING IS LOAD-BEARING, worth ~0.06 rho over taking the extreme across all
   tissues, so a measurement whose context is not the requested one is labelled
   `cross_tissue` rather than quietly compared.

3. POPULATION RHO IS A PROPERTY OF THE PAIRING, NEVER OF THE VARIANT. `Pairing.population_rho`
   is the cohort Spearman correlation of that modality against that substrate. Nothing here
   computes a per-variant statistic from a single pair, because with n = 1 every such number
   is either a tautology or a fiction.

The AlphaGenome side comes from `tools/alphagenome.py`; the measured side is SQL over
db-api, run by the caller and handed in as rows, so this module stays free of transport and
testable without either.

Reached from ServerToolExecutor only, for the reason `tools/alphagenome.py` states: neither
module may be imported from anything the sandbox image ships. The module name carries
"alphagenome" so the AST guard in tests/test_alphagenome.py covers it for free.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from genetics_mcp_server.tools import sql_safety
from genetics_mcp_server.tools.alphagenome import (
    _MAGNITUDE,
    _SIGNED,
    MODALITIES,
    _slug,
)

CREDIBLE_SETS_VIEW = "credible_sets_v"
MPRA_VIEW = "mpra_v"

# how many measurements a modality shows. A variant can be an eQTL in hundreds of
# study/tissue combinations, and a comparison the reader cannot hold is not a comparison;
# `n_measurements` carries the full count so the cut is visible.
_MAX_MEASUREMENTS = 5

# the variant predicate bounds the result set; `query_database` strips a trailing LIMIT, so
# this is passed as its `max_rows` instead of written into the SQL
MAX_ROWS = 500

_PREDICTION_SOURCE = "AlphaGenome (Google DeepMind)"

_NOT_REQUESTED = "not_requested"
_MATCHED = "matched"
_CROSS_TISSUE = "cross_tissue"


@dataclass(frozen=True)
class Pairing:
    """One (modality, measured substrate) pair that was actually calibrated here.

    Only pairs with a measured signed-Spearman behind them appear: a pairing invented for
    completeness would put a number beside a measurement that nothing ever checked. The
    tier-4 modalities have no entry at all, which is what makes "nothing measured to
    compare against" a derived answer rather than a hand-written special case.
    """

    substrate: str
    assay: str
    view: str
    value_column: str
    context_column: str
    # credible_sets_v holds every QTL flavour in one view; mpra_v needs no such filter
    data_type: str | None
    population_rho: float


def _qtl(substrate: str, rho: float) -> Pairing:
    return Pairing(
        substrate=substrate,
        assay=f"{substrate} beta",
        view=CREDIBLE_SETS_VIEW,
        value_column="beta",
        context_column="cell_type",
        data_type=substrate,
        population_rho=rho,
    )


def _mpra(rho: float) -> Pairing:
    return Pairing(
        substrate="MPRA",
        assay="MPRA log2Skew",
        view=MPRA_VIEW,
        value_column="log2Skew",
        context_column="cell_line",
        data_type=None,
        population_rho=rho,
    )


# The rho of each pair, from the calibration run, not the modality-level number in
# `alphagenome.MODALITIES`: DNASE was calibrated against both substrates and correlates
# differently with each (+0.478 caQTL, +0.503 MPRA), so quoting one for both would attach a
# number to a comparison it was not measured on.
#
# variant_effect_v is NOT here and must not be added: it holds ChromBPNet and FLARE
# PREDICTIONS, not measurements, so a row from it belongs on the `measured: false` side of a
# response -- its `predicted_direction` column and its `chrombpnet_abs_logfc` score_type are a
# model's sign and a model's magnitude, so an agreement against either is model-vs-model.
PAIRINGS: dict[str, tuple[Pairing, ...]] = {
    "DNASE": (_qtl("caQTL", 0.478), _mpra(0.503)),
    "ATAC": (_qtl("caQTL", 0.489),),
    "CHIP_HISTONE": (_mpra(0.472),),
    "SPLICE_SITES": (_qtl("sQTL", 0.249),),
    "SPLICE_JUNCTIONS": (_qtl("sQTL", 0.231),),
    "SPLICE_SITE_USAGE": (_qtl("sQTL", 0.148),),
    "RNA_SEQ": (_qtl("eQTL", 0.112),),
}


def pairings_for(modality: str) -> tuple[Pairing, ...]:
    return PAIRINGS.get(modality, ())


# The suite's views spell the chromosome as an integer with X as 23. Y and M follow the
# same convention read off the same views and are believed rather than measured, exactly as
# `alphagenome._CHROM_ALIASES` records for the other direction; a wrong spelling here yields
# no measured rows rather than the wrong ones, because the whole variant id must match.
_SUITE_CHROM = {"chrX": "23", "chrY": "24", "chrM": "25"}


def suite_variant_id(variant: dict[str, Any]) -> str | None:
    """A parsed AlphaGenome variant as the `variant` column of the suite's views spells it."""
    chromosome = variant.get("chromosome")
    if not isinstance(chromosome, str):
        return None
    number = _SUITE_CHROM.get(chromosome)
    if number is None:
        digits = chromosome[3:]
        if not digits.isdigit():
            return None
        number = str(int(digits))
    return ":".join(
        (
            number,
            str(variant["position"]),
            variant["reference_bases"],
            variant["alternate_bases"],
        )
    )


# BigQuery's own spelling of `alphagenome._slug`, so a context matches here exactly where it
# would match in the client
def _slug_sql(column: str) -> str:
    return f"REGEXP_REPLACE(LOWER({column}), r'[^a-z0-9]', '')"


def _matched_first(column: str, cell_type: str | None) -> str:
    """ORDER BY that keeps the requested context's rows ahead of everything else.

    Ranking in SQL rather than after the fetch is what stops `max_rows` from dropping the
    only rows the comparison is allowed to call matched.
    """
    if not cell_type:
        return ""
    needle = _slug(cell_type)
    if not needle:
        return ""
    try:
        literal = sql_safety.quote_literal(needle, name="cell_type")
    except sql_safety.SqlValueError:
        # a needle past the 128-character literal cap cannot match any context anyway, so
        # losing the ranking costs nothing; the sibling prediction tool answers for the
        # same input, and raising only here would make the pair disagree about validity
        return ""
    return f"CASE WHEN STRPOS({_slug_sql(column)}, {literal}) > 0 THEN 0 ELSE 1 END, "


def credible_sets_sql(
    variant_ids: Sequence[str], data_types: Iterable[str], cell_type: str | None
) -> str:
    """Measured QTL effect sizes for these variants, strongest and best-matched first.

    The `chr` predicate is not redundant with the variant id: it is what prunes partitions,
    and credible_sets_v is one of the large ones.
    """
    ids = sql_safety.quote_literal_list(variant_ids, name="variant")
    # the only unquoted interpolation in either builder: safe because every id reaching it
    # comes from `suite_variant_id`, which yields digits or None. A hand-built variant id
    # passed to this module-public builder would put its chromosome field straight into SQL.
    chroms = ", ".join(sorted({v.split(":", 1)[0] for v in variant_ids}))
    types = sql_safety.quote_literal_list(sorted(set(data_types)), name="data_type")
    return (
        "SELECT variant, data_type, trait, cell_type, beta, se, pip, cs_id, "
        "mlog10p, resource, dataset "
        f"FROM {CREDIBLE_SETS_VIEW} "
        f"WHERE chr IN ({chroms}) AND variant IN ({ids}) AND data_type IN ({types}) "
        f"ORDER BY {_matched_first('cell_type', cell_type)}ABS(beta) DESC"
    )


def mpra_sql(variant_ids: Sequence[str], cell_type: str | None) -> str:
    """Measured MPRA allelic skew for these variants, strongest and best-matched first."""
    ids = sql_safety.quote_literal_list(variant_ids, name="variant")
    # the only unquoted interpolation in either builder: safe because every id reaching it
    # comes from `suite_variant_id`, which yields digits or None. A hand-built variant id
    # passed to this module-public builder would put its chromosome field straight into SQL.
    chroms = ", ".join(sorted({v.split(":", 1)[0] for v in variant_ids}))
    return (
        "SELECT variant, cell_line, log2Skew, log2Skew_se, log2Skew_mlog10p, "
        "emVar, active, resource, dataset "
        f"FROM {MPRA_VIEW} "
        f"WHERE chr IN ({chroms}) AND variant IN ({ids}) "
        f"ORDER BY {_matched_first('cell_line', cell_type)}ABS(log2Skew) DESC"
    )


def _expose(value: Any, quantity: str) -> float | None:
    """A measured value under the modality's own reading rule.

    The same rule `alphagenome._expose` applies to the prediction, applied to the
    measurement for the same reason: beside a magnitude-only prediction, a signed
    measurement is a direction the reader will attribute to the pair.
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if quantity == _SIGNED else abs(number)


def _context_match(requested: str | None, context: Any) -> str:
    if not requested:
        return _NOT_REQUESTED
    needle = _slug(requested)
    slug = _slug(str(context or ""))
    if needle and slug and (needle in slug or slug in needle):
        return _MATCHED
    return _CROSS_TISSUE


def _measurement(
    row: dict[str, Any], pairing: Pairing, quantity: str, cell_type: str | None
) -> dict[str, Any]:
    context = row.get(pairing.context_column)
    match = _context_match(cell_type, context)
    source: dict[str, Any] = {
        "view": pairing.view,
        "column": pairing.value_column,
        "assay": pairing.assay,
        "resource": row.get("resource"),
        "dataset": row.get("dataset"),
    }
    if pairing.data_type:
        source["data_type"] = pairing.data_type
        # for caQTL the trait is an accessibility peak, for eQTL/sQTL a gene: either way it
        # names WHAT was measured, which the value alone does not
        source["trait"] = row.get("trait")
    entry: dict[str, Any] = {
        "measured": True,
        "value": _expose(row.get(pairing.value_column), quantity),
        "quantity": quantity,
        "source": source,
        "context": context,
        "context_match": match,
        "cross_tissue": match == _CROSS_TISSUE,
    }
    if pairing.view == MPRA_VIEW:
        entry["emVar"] = row.get("emVar")
        entry["active"] = row.get("active")
        if _slug(str(context or "")) == "meta":
            entry["context_note"] = (
                "cross-cell-line meta-analysis, not one measured cell line"
            )
    else:
        entry["se"] = row.get("se")
        entry["pip"] = row.get("pip")
    return entry


def _gene_names(block: dict[str, Any]) -> list[str]:
    return [
        str(gene.get("gene_name"))
        for gene in (block.get("genes") or [])
        if gene.get("gene_name")
    ]


def _predicted(block: dict[str, Any], gene_name: str | None) -> dict[str, Any]:
    """The prediction side, labelled, optionally read off one gene's row.

    A gene-resolved modality answers per gene and an eQTL or sQTL beta is measured on one
    gene, so where the two name the same gene that is the pair worth comparing -- comparing
    the strongest predicted gene against a measurement on a different one would be an
    accident of ordering.
    """
    entry = block
    matched_gene = None
    if gene_name:
        needle = _slug(gene_name)
        for gene in block.get("genes") or []:
            if _slug(str(gene.get("gene_name") or "")) == needle:
                entry, matched_gene = gene, gene.get("gene_name")
                break
    return {
        "measured": False,
        "data_kind": "model_prediction",
        "source": _PREDICTION_SOURCE,
        "quantity": block.get("quantity"),
        "value": entry.get("value"),
        "quantile": entry.get("quantile"),
        "gene": matched_gene if matched_gene else entry.get("gene_name"),
        "cell_type_match": block.get("cell_type_match"),
        "n_tracks_scored": block.get("n_tracks_scored"),
    }


def _rank(measurement: dict[str, Any], gene_names: set[str]) -> tuple:
    """Matched context first, then a measurement on a gene the model also scored, then size."""
    trait = _slug(str((measurement.get("source") or {}).get("trait") or ""))
    value = measurement.get("value")
    return (
        0 if measurement["context_match"] == _MATCHED else 1,
        0 if trait and trait in gene_names else 1,
        -abs(value) if value is not None else 1.0,
    )


def _concordance(
    predicted: dict[str, Any], measurement: dict[str, Any], quantity: str
) -> dict[str, Any]:
    """The one comparison, under the modality's quantity rule.

    For a magnitude modality this dict carries NO `direction` key -- not a null one. A key
    that is sometimes absent is harder to render carelessly than a key that is sometimes
    null, and the absence is the statement: no direction is available to report.
    """
    p, m = predicted.get("value"), measurement.get("value")
    both = p is not None and m is not None
    common: dict[str, Any] = {
        "quantity": quantity,
        "predicted_value": p,
        "measured_value": m,
        "compared_against": measurement.get("source"),
        "context_match": measurement.get("context_match"),
    }
    if quantity == _MAGNITUDE:
        return {
            **common,
            "direction_reported": False,
            "why_no_direction": (
                "the measured beta orients to a leafcutter intron cluster and the predicted "
                "splice delta has no corresponding orientation, so the two signs are not "
                "commensurable; magnitudes only. The signed measured beta is in "
                f"{measurement['source']['view']}, where its orientation is defined."
            ),
        }
    if not both or p == 0 or m == 0:
        return {
            **common,
            "direction": None,
            "why_no_direction": "one side has no value, or is exactly zero",
        }
    return {**common, "direction": "agrees" if (p > 0) == (m > 0) else "disagrees"}


def _no_substrate_note(modality: str) -> str:
    return (
        f"nothing measured to compare against: this suite has no assay that measures what "
        f"{modality} predicts, so no comparison is possible. The prediction stands alone "
        f"and `validation.status` says it was never checked against anything measured here."
    )


def compare_modality(
    block: dict[str, Any],
    rows: dict[str, list[dict[str, Any]]],
    cell_type: str | None,
    unavailable: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """One modality's prediction, the suite's measurements for it, and their concordance."""
    name = block.get("modality") or ""
    modality = MODALITIES.get(name)
    quantity = block.get("quantity") or (modality.quantity if modality else _MAGNITUDE)
    pairings = pairings_for(name)
    out: dict[str, Any] = {
        "modality": name,
        "quantity": quantity,
        "validation": block.get("validation"),
    }

    if not pairings:
        out["prediction"] = _predicted(block, None)
        out["measured_substrates"] = []
        out["measurements"] = []
        out["concordance"] = None
        out["note"] = _no_substrate_note(name)
        return out

    gene_names = {_slug(g) for g in _gene_names(block)}
    measurements: list[dict[str, Any]] = []
    substrates: list[dict[str, Any]] = []
    for pairing in pairings:
        substrate: dict[str, Any] = {
            "substrate": pairing.substrate,
            "assay": pairing.assay,
            "view": pairing.view,
            "column": pairing.value_column,
            # the pair's own cohort correlation. `rho_scope` says what it ranges over,
            # because beside a single variant it is the number most likely to be read
            # as that variant's confidence, which it is not.
            "population_rho": pairing.population_rho,
            "rho_scope": "population",
        }
        # a substrate listed with its rho and nothing else reads as one that was consulted
        # and had nothing to say, which is the opposite of what a failed lookup means
        if pairing.view in unavailable:
            substrate["lookup"] = "failed"
        substrates.append(substrate)
        for row in rows.get(pairing.view, []):
            if pairing.data_type and row.get("data_type") != pairing.data_type:
                continue
            measurements.append(_measurement(row, pairing, quantity, cell_type))

    measurements.sort(key=lambda m: _rank(m, gene_names))
    out["measured_substrates"] = substrates
    out["n_measurements"] = len(measurements)
    out["measurements"] = measurements[:_MAX_MEASUREMENTS]

    # an empty answer and an answer that never arrived look identical from here, and only
    # one of them means the suite measured nothing -- which stays true of the failed
    # substrate when a modality's other substrate did answer, so it is said either way
    blocked = sorted({p.view for p in pairings} & unavailable)
    blocked_note = ""
    if blocked:
        blocked_assays = " or ".join(
            sorted({p.assay for p in pairings if p.view in blocked})
        )
        blocked_note = (
            f"the measured lookup in {', '.join(blocked)} failed, so nothing can be said "
            f"about whether this suite has measured this variant as {blocked_assays}"
        )

    if not measurements:
        out["prediction"] = _predicted(block, None)
        out["concordance"] = None
        out["note"] = blocked_note or (
            "no "
            + " or ".join(sorted({p.assay for p in pairings}))
            + " for this variant in this suite; the prediction stands alone"
        )
        return out

    best = measurements[0]
    trait = (best.get("source") or {}).get("trait")
    on_predicted_gene = _slug(str(trait or "")) in gene_names
    predicted = _predicted(block, trait if on_predicted_gene else None)
    out["prediction"] = predicted
    concordance = _concordance(predicted, best, quantity)
    gene_resolved = modality is not None and modality.gene_resolved
    if gene_resolved and not on_predicted_gene and "direction" in concordance:
        # `genes` is capped, so the gene an eQTL beta was measured on need not be among the
        # ones scored. The block-level value is then the STRONGEST PREDICTED GENE, and an
        # agreement between it and a beta on a different gene is a statement about two
        # different genes -- exactly the fabrication this module exists to prevent.
        top_gene = next(iter(_gene_names(block)), None)
        other = (
            f"the strongest predicted gene {top_gene!r}"
            if top_gene
            else "no gene in particular"
        )
        concordance.pop("direction")
        concordance["why_no_direction"] = (
            f"the measurement is on {trait!r} and the prediction's value is for {other}: "
            "not the same gene, so the two signs are not a pair. Both values are shown, "
            "unpaired."
        )
    out["concordance"] = concordance
    if best["context_match"] == _CROSS_TISSUE:
        out["note"] = (
            f"cross-tissue comparison: the prediction was asked for {cell_type!r} and the "
            f"nearest measurement is in {best.get('context')!r}. Cell-type matching is worth "
            "about 0.06 rho, so an unmatched pair is weaker evidence than a matched one."
        )
    if blocked_note:
        out["note"] = f"{out['note']} {blocked_note}" if out.get("note") else blocked_note
    return out


def compare_variant(
    prediction: dict[str, Any],
    rows: dict[str, list[dict[str, Any]]],
    cell_type: str | None,
    unavailable: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """One variant's prediction entry turned into a side-by-side comparison."""
    if prediction.get("success") is not True:
        return {**prediction, "modalities": None}
    return {
        "success": True,
        "variant": prediction.get("variant"),
        "cell_type": prediction.get("cell_type"),
        "modalities": {
            name: compare_modality(block, rows, cell_type, unavailable)
            for name, block in (prediction.get("modalities") or {}).items()
        },
    }


def views_needed(modalities: Iterable[str]) -> dict[str, set[str]]:
    """{view: {data_type, ...}} for the substrates these modalities have; empty for tier 4."""
    needed: dict[str, set[str]] = {}
    for name in modalities:
        for pairing in pairings_for(name):
            types = needed.setdefault(pairing.view, set())
            if pairing.data_type:
                types.add(pairing.data_type)
    return needed


def rows_from_query(result: dict[str, Any]) -> list[dict[str, Any]]:
    """A `query_database` answer as row dicts; db-api answers in columns plus value lists."""
    if not result.get("success"):
        return []
    columns = result.get("columns") or []
    return [
        row if isinstance(row, dict) else dict(zip(columns, row))
        for row in (result.get("rows") or [])
    ]


def group_by_variant(
    rows_by_view: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """{suite variant id: {view: [row, ...]}} -- the shape `compare_variant` reads."""
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for view, rows in rows_by_view.items():
        for row in rows:
            key = str(row.get("variant") or "")
            out.setdefault(key, {}).setdefault(view, []).append(row)
    return out
