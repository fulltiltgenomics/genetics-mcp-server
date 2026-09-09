"""Entity extraction for cross-session memory.

The premise measurement and the session digest must agree on what an entity is, so both
call `extract_entities` here instead of each keying on its own list of parameter names.
Kinds are a closed set: gene, phenotype, variant, dataset, view.
"""

import base64
import json
import re
from typing import Any

Entity = tuple[str, str]

KINDS: tuple[str, ...] = ("gene", "phenotype", "variant", "dataset", "view")

# parameter names read out of tools/definitions.py. A name several tools share keeps one
# kind everywhere; a parameter whose declared value is a free-text search phrase rather than
# an identifier is deliberately absent — it carries sentences, not entities.
_PARAM_KIND: dict[str, str] = {
    "gene": "gene",
    "genes": "gene",
    "symbols": "gene",
    "phenotype": "phenotype",
    "phenotypes": "phenotype",
    "phenotype_code": "phenotype",
    "codes": "phenotype",
    "trait": "phenotype",
    "variant": "variant",
    "variants": "variant",
    "variant1": "variant",
    "variant2": "variant",
    "rsids": "variant",
    "resource": "dataset",
    "resources": "dataset",
    "resource_or_dataset": "dataset",
    "table": "view",
}

# `query` means whatever the tool taking it says it means, so it is keyed by tool name.
# The rule: a tool appears here when its `query` is declared to be an identifier (a gene
# symbol, a phenotype name); a tool whose `query` is declared to be a free-text search
# phrase is left out, because a sentence is not an entity.
_QUERY_KIND_BY_TOOL: dict[str, str] = {
    "search_genes": "gene",
    "search_phenotypes": "phenotype",
    "get_protein_annotations": "gene",
    "map_protein_variants": "gene",
    "get_drug_targets_for_gene": "gene",
    "get_target_bioactivity": "gene",
}

# two tools declare `query` as an identifier only under a condition their own arguments
# state, so the condition is checked instead of the tool name alone
_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9-]+$")


def _conditional_query_kind(tool_name: str, args: dict, value: Any) -> str | None:
    if tool_name == "search_cbioportal":
        # a gene symbol for the gene_* query types; a residue or a study term otherwise
        kind = str(args.get("query_type", "gene_summary"))
        return "gene" if kind.startswith("gene_") else None
    if tool_name == "search_mgi":
        # a symbol, an MP phenotype term or an MGI id, told apart by shape
        return (
            "gene"
            if isinstance(value, str) and _SYMBOL_RE.match(value.strip().upper())
            else None
        )
    return None

# parameters holding SQL or Python: mined with regexes rather than taken whole
_TEXT_PARAMS = frozenset({"sql", "code"})

_RSID_RE = re.compile(r"\brs\d{3,}\b", re.I)
# every BigQuery view in configs/datasets.yaml is named <something>_v
_VIEW_RE = re.compile(r"\b([a-z][a-z0-9_]*_v)\b")
_VARIANT_RE = re.compile(
    r"\b(?:chr)?(\d{1,2}|X|Y|MT)[-:_](\d+)[-:_]([ACGT]+)[-:_]([ACGT]+)\b", re.I
)
# a comparison or keyword argument against one or more quoted literals: `gene = 'APOE'`,
# `phenocode IN ('E4_DM2')`, `phenotypes=["I9_CHD"]` — SQL and script text read the same way
_COMPARISON_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:=|!=|<>|\bIN\b|\bLIKE\b)\s*[([]?((?:\s*['\"][^'\"]*['\"]\s*,?)+)",
    re.I,
)
_QUOTED_RE = re.compile(r"['\"]([^'\"]*)['\"]")
# checked in order: the first substring that matches decides the kind
_COLUMN_KIND: tuple[tuple[str, str], ...] = (
    ("gene", "gene"),
    ("symbol", "gene"),
    ("rsid", "variant"),
    ("variant", "variant"),
    ("phenocode", "phenotype"),
    ("phenotype", "phenotype"),
    ("endpoint", "phenotype"),
    ("trait", "phenotype"),
    ("resource", "dataset"),
    ("dataset", "dataset"),
)

_TOOLUSE_MARKER_RE = re.compile(r"\[TOOLUSE:([A-Za-z0-9+/=]+)\]")

_MAX_VALUE_CHARS = 64


def _column_kind(column: str) -> str | None:
    lowered = column.lower()
    for needle, kind in _COLUMN_KIND:
        if needle in lowered:
            return kind
    return None


def _normalise_variant(value: str) -> str:
    match = _VARIANT_RE.fullmatch(value)
    if match:
        chrom, pos, ref, alt = match.groups()
        return f"{chrom}-{pos}-{ref}-{alt}".lower()
    return value.lower()


def _normalise(kind: str, value: str) -> str | None:
    collapsed = " ".join(value.split())
    if len(collapsed) < 2 or len(collapsed) > _MAX_VALUE_CHARS or collapsed.isdigit():
        return None
    if kind == "variant":
        return _normalise_variant(collapsed)
    if kind in ("dataset", "view"):
        return collapsed.lower()
    # gene symbols and endpoint codes are upper-case by convention; a phenotype given as
    # free text (search_phenotypes takes "type 2 diabetes") has none, so it folds down
    return collapsed.lower() if " " in collapsed else collapsed.upper()


def _add(out: set[Entity], kind: str, raw: str) -> None:
    for part in raw.split(","):
        value = _normalise(kind, part)
        if value:
            out.add((kind, value))


def _add_value(out: set[Entity], kind: str, value: Any) -> None:
    if isinstance(value, str):
        _add(out, kind, value)
    elif isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, str):
                _add(out, kind, item)


def _mine_text(out: set[Entity], text: str) -> None:
    for rsid in _RSID_RE.findall(text):
        out.add(("variant", rsid.lower()))
    for view in _VIEW_RE.findall(text):
        out.add(("view", view.lower()))
    for match in _VARIANT_RE.finditer(text):
        chrom, pos, ref, alt = match.groups()
        out.add(("variant", f"{chrom}-{pos}-{ref}-{alt}".lower()))
    for column, literals in _COMPARISON_RE.findall(text):
        kind = _column_kind(column)
        if not kind:
            continue
        for literal in _QUOTED_RE.findall(literals):
            _add(out, kind, literal)


def _walk_input(out: set[Entity], tool_name: str, value: Any, depth: int = 0) -> None:
    if depth > 4:
        return
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in _TEXT_PARAMS and isinstance(item, str):
                _mine_text(out, item)
                continue
            kind = _PARAM_KIND.get(lowered)
            if kind is None and lowered == "query":
                kind = _QUERY_KIND_BY_TOOL.get(tool_name) or _conditional_query_kind(
                    tool_name, value, item
                )
            if kind:
                _add_value(out, kind, item)
            if isinstance(item, (dict, list, tuple)):
                _walk_input(out, tool_name, item, depth + 1)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _walk_input(out, tool_name, item, depth + 1)


def _loads(raw: Any) -> Any:
    if isinstance(raw, (list, dict)):
        return raw
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", "replace")
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return None


def extract_entities(content_json: Any, tool_results_json: Any = None) -> set[Entity]:
    """(kind, value) pairs from the `tool_use` INPUTS of one assistant message.

    `tool_results_json` is accepted and never read. Memory is built from what the user and
    the model asked for, never from the rows a tool returned, and keeping the parameter
    means callers written against this signature survive the digest renderer landing.
    """
    del tool_results_json
    out: set[Entity] = set()
    blocks = _loads(content_json)
    if isinstance(blocks, dict):
        blocks = blocks.get("content")
    if not isinstance(blocks, list):
        return out
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            _walk_input(out, str(block.get("name") or ""), block.get("input"))
    return out


def extract_marker_entities(content: Any) -> set[Entity]:
    """Entities from the `[TOOLUSE:<base64>]` markers the browser writes into `content`.

    Older stored assistant rows carry the tool call only as this display marker, with no
    `content_json` beside it; without this path those sessions look entity-free.
    """
    out: set[Entity] = set()
    if not isinstance(content, str):
        return out
    for blob in _TOOLUSE_MARKER_RE.findall(content):
        try:
            payload = json.loads(base64.b64decode(blob).decode("utf-8", "replace"))
        except (ValueError, TypeError, base64.binascii.Error):
            continue
        if not isinstance(payload, dict):
            continue
        name = str(payload.get("name") or "")
        _walk_input(out, name, payload.get("input", payload))
    return out


def extract_user_entities(text: Any) -> set[Entity]:
    """Entities a user typed: identifiers only, since prose is not an entity."""
    out: set[Entity] = set()
    if not isinstance(text, str):
        return out
    for rsid in _RSID_RE.findall(text):
        out.add(("variant", rsid.lower()))
    for match in _VARIANT_RE.finditer(text):
        chrom, pos, ref, alt = match.groups()
        out.add(("variant", f"{chrom}-{pos}-{ref}-{alt}".lower()))
    return out
