"""The protein structure layer: executor fetches, the three chat tools, the SDK functions, and
where each is offered.

results-api is faked at the transport over the lab reference bundle under
tests/fixtures/structure/ (real model files and PAE), so the geometry a tool reports can be
held to `genetics.structure` computing on the same files directly. The variant-to-residue
map and the feature rows are made up here; the routes are genetics-results-api's
`/v1/protein_structure/*` and `/v1/variant_annotation/protein`.
"""

import gzip
import importlib
import json
import re
from pathlib import Path

import httpx
import numpy as np
import polars as pl
import pytest
from test_tool_surface_golden import _mcp_hardcoded_exclusions

from genetics_mcp_server.config.defaults import PROMPT_VARIANTS, default_system_prompt
from genetics_mcp_server.config.settings import (
    PROTEIN_STRUCTURE_PROFILES,
    PROTEIN_STRUCTURE_TOOLS,
    Settings,
    get_settings,
)
from genetics_mcp_server.sdk import _structure_contract as contract
from genetics_mcp_server.sdk import structure as st
from genetics_mcp_server.sdk.client import GeneticsClient
from genetics_mcp_server.sdk.errors import GeneticsError, GeneticsUsageError
from genetics_mcp_server.tools import executor as executor_module
from genetics_mcp_server.tools.definitions import (
    TOOL_DEFINITIONS,
    get_anthropic_tools,
    register_mcp_tools,
)
from genetics_mcp_server.tools.executor import ToolExecutor

FIXTURES = Path(__file__).parent / "fixtures" / "structure"
RELEASE = "2026-01-01-000000"
SDK_FUNCTIONS = (
    "protein_models",
    "protein_residues",
    "protein_features",
    "protein_variants",
    "protein_structure_file",
    "protein_pae",
    "protein_structure",
)

# one file with PAE, and eight windows without it whose first and last share no file
SINGLE, SINGLE_FILE = "Q6RSH7", "AF-Q6RSH7-F1-model_v6"
WINDOWED = "Q02224"


def _models() -> list[dict]:
    lines = (FIXTURES / "models.tsv").read_text().splitlines()
    header = lines[0].lstrip("#").split("\t")
    integers = {"window_index", "n_windows", "protein_start", "protein_end", "n_residues"}
    integers |= {"owned_start", "owned_end"}
    rows = []
    for line in lines[1:]:
        row = dict(zip(header, line.split("\t")))
        rows.append(
            {
                k: int(v) if k in integers else (v == "true") if k == "has_pae" else v
                for k, v in row.items()
                if k in integers or k in ("model_file_id", "accession", "gene_name", "has_pae")
            }
        )
    return rows


MODELS = _models()
BY_FILE = {m["model_file_id"]: m for m in MODELS}


def _pae(mid: str) -> bytes:
    return gzip.decompress((FIXTURES / "pae" / f"{mid}.u8.gz").read_bytes())


SAMPLED = np.load(FIXTURES / "pae_rows.npz")


def _pae_row(mid: str, position: int) -> bytes:
    """One PAE row: from the whole matrix where the bundle has it, else from its sampled rows.

    A row the bundle does not hold is served as no-value codes, so a test that reads one
    gets nulls rather than numbers.
    """
    n = BY_FILE[mid]["n_residues"]
    if (FIXTURES / "pae" / f"{mid}.u8.gz").exists():
        return _pae(mid)[(position - 1) * n : position * n]
    sampled = SAMPLED[f"{mid}|file_positions"].tolist()
    if position in sampled:
        return SAMPLED[f"{mid}|rows"][sampled.index(position)].tobytes()
    return bytes([contract.PAE_NO_VALUE]) * n


def _structure(mid: str) -> bytes:
    return (FIXTURES / "structures" / f"{mid}.pdb.gz").read_bytes()


def _variant_row(
    variant: str, accession: str, gene: str, aa_pos: int, aa: str = "AV", gene_id: str = "NA"
) -> dict:
    chrom, pos, ref, alt = variant.split(":")
    model = next(
        m
        for m in MODELS
        if m["accession"] == accession and m["owned_start"] <= aa_pos <= m["owned_end"]
    )
    return {
        "chrom": chrom,
        "pos": pos,
        "ref": ref,
        "alt": alt,
        "variant": variant,
        "accession": accession,
        "gene_id": gene_id,
        "gene_name": gene,
        "aa_pos": str(aa_pos),
        "aa_ref": aa[0],
        "aa_alt": aa[1],
        "model_file_id": model["model_file_id"],
        "file_position": str(aa_pos - model["protein_start"] + 1),
        "mu_snp": "NA" if aa_pos == 10 else "1.5e-09",
    }


V10, V12, V12B, V130 = "3:1000:A:T", "3:1006:C:G", "3:1007:C:G", "3:1360:G:A"
W100, W2700 = "4:5000:A:C", "4:9000:T:C"
NO_ROW = "1:100:A:T"
VX = "X:700:A:T"
# two windows with PAE; both residues are owned by the second, at file positions 601 and 602
PAE_WINDOWED, PAE_WINDOWED_F2 = "C9JG08", "AF3-C9JG08-F2-model_v6"
P4401, P4402 = "9:4401:A:C", "9:4402:A:C"
SINGLE_GENE_ID = "ENSG00000189030"
VARIANT_MAP = {
    V10: [_variant_row(V10, SINGLE, "VHLL", 10)],
    V12: [_variant_row(V12, SINGLE, "VHLL", 12)],
    V12B: [_variant_row(V12B, SINGLE, "VHLL", 12)],
    V130: [_variant_row(V130, SINGLE, "VHLL", 130, gene_id=SINGLE_GENE_ID)],
    # the map spells X as 23; the caller did not
    VX: [_variant_row("23:700:A:T", SINGLE, "VHLL", 20)],
    P4401: [_variant_row(P4401, PAE_WINDOWED, "SPATA31H1", 4401)],
    P4402: [_variant_row(P4402, PAE_WINDOWED, "SPATA31H1", 4402)],
    W100: [_variant_row(W100, WINDOWED, "CENPE", 100)],
    W2700: [_variant_row(W2700, WINDOWED, "CENPE", 2700)],
}

FEATURES_TSV = (
    "gene_id\tchrom\tgene_start\tgene_end\tgene_name\taccession\tsource\tfeature_type\tstart\tend"
    "\tdescription\tevidence\tevidence_tier\n"
    'G\t3\t1\t9\tVHLL\tQ6RSH7\tuniprot\tChain\t1\t139\tvon Hippel-Lindau-like "protein"\tNA\tunknown\n'
    "G\t3\t1\t9\tVHLL\tQ6RSH7\tuniprot\tRegion\t5\t11\tDisordered\tECO:0000256\tpredicted\n"
    "G\t3\t1\t9\tVHLL\tQ6RSH7\tted\tTED domain\t40\t120\tTED01\tNA\tpredicted\n"
)
RESIDUES_TSV = (
    "gene_id\tchrom\tgene_start\tgene_end\tgene_name\taccession\tmodel_file_id\tfile_position"
    "\tprotein_position\tresidue\tis_owner\tca_x\tca_y\tca_z\tplddt\tmu\tn_with_mu\tn_possible\n"
    f"G\t3\t1\t9\tVHLL\tQ6RSH7\t{SINGLE_FILE}\t1\t1\tM\ttrue\t1.5\t-2.0\t3.25\t41.5\tNA\tNA\tNA\n"
    f"G\t3\t1\t9\tVHLL\tQ6RSH7\t{SINGLE_FILE}\t2\t2\tA\ttrue\t2.5\t-2.0\t3.25\t52.0\t2.5e-08\t6\t7\n"
)


def _positions(spec: str) -> list[int]:
    out: set[int] = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out |= set(range(int(lo), int(hi or lo) + 1))
    return sorted(out)


def _api(seen: list[httpx.Request] | None = None, fail: dict[str, httpx.Response] | None = None):
    """A results-api serving the fixture bundle; `fail` maps a path fragment to a refusal."""

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        path = request.url.path.removeprefix("/api/v1/")
        for fragment, response in (fail or {}).items():
            if fragment in path:
                return response
        version = {"X-Dataset-Version": RELEASE}
        if path == "variant_annotation/protein":
            rows = [
                r for v in json.loads(request.content)["variants"] for r in VARIANT_MAP.get(v, [])
            ]
            columns = ",".join(next(iter(VARIANT_MAP.values()))[0])
            return httpx.Response(200, json=rows, headers={**version, "X-Columns": columns})
        kind, _, rest = path.removeprefix("protein_structure/").partition("/")
        if kind == "models":
            rows = [m for m in MODELS if rest in (m["accession"], m["gene_name"])]
            if not rows:
                detail = (
                    f"{rest} is neither a UniProt accession in release {RELEASE} nor a known gene"
                )
                return httpx.Response(404, json={"detail": detail})
            return httpx.Response(200, json=rows, headers=version)
        if kind in ("features", "residues"):
            assert request.url.params["format"] == "tsv"
            if rest != SINGLE:
                detail = f"No {kind[:-1]} rows for {rest} in release {RELEASE}"
                return httpx.Response(404, json={"detail": detail})
            return httpx.Response(
                200, text=FEATURES_TSV if kind == "features" else RESIDUES_TSV, headers=version
            )
        if kind == "structure":
            return httpx.Response(200, content=_structure(rest), headers=version)
        mid, _, rows_route = rest.partition("/")
        model = BY_FILE[mid]
        if not model["has_pae"]:
            return httpx.Response(404, json={"detail": f"Model file {mid} has no PAE"})
        n = model["n_residues"]
        headers = {**version, "X-PAE-Size": str(n)}
        if not rows_route:
            return httpx.Response(200, content=_pae(mid), headers=headers)
        wanted = _positions(request.url.params["positions"])
        if len(wanted) > contract.pae_rows_per_request(n) or wanted[-1] > n:
            return httpx.Response(422, json={"detail": "positions past the model or too many"})
        body = b"".join(_pae_row(mid, p) for p in wanted)
        return httpx.Response(
            200, content=body, headers={**headers, "X-PAE-Rows": str(len(wanted))}
        )

    return handler


@pytest.fixture
def seen() -> list[httpx.Request]:
    return []


@pytest.fixture
async def executor(seen):
    ex = ToolExecutor(api_base_url="http://api.test/api", expose_columns=True)
    ex.client = httpx.AsyncClient(transport=httpx.MockTransport(_api(seen)))
    yield ex
    await ex.close()


def _failing(fail: dict[str, httpx.Response]) -> ToolExecutor:
    ex = ToolExecutor(api_base_url="http://api.test/api")
    ex.client = httpx.AsyncClient(transport=httpx.MockTransport(_api(fail=fail)))
    return ex


def _paths(seen: list[httpx.Request]) -> list[str]:
    return [r.url.path.removeprefix("/api/v1/") for r in seen]


def _reference(accession: str, pae: bool = True) -> st.Protein:
    rows = [m for m in MODELS if m["accession"] == accession]
    return st.protein(
        rows,
        {m["model_file_id"]: _structure(m["model_file_id"]) for m in rows},
        {m["model_file_id"]: _pae(m["model_file_id"]) for m in rows} if pae else None,
    )


# --------------------------------------------------------------------------- fetches


async def test_models_are_listed_by_accession_or_gene(executor, seen):
    by_accession = await executor.get_protein_models(SINGLE)
    by_gene = await executor.get_protein_models("CENPE")
    assert by_accession["success"] and by_accession["release"] == RELEASE
    assert [m["model_file_id"] for m in by_accession["results"]] == [SINGLE_FILE]
    assert by_gene["count"] == by_gene["results"][0]["n_windows"] > 1
    assert _paths(seen) == [
        f"protein_structure/models/{SINGLE}",
        "protein_structure/models/CENPE",
    ]


async def test_row_tables_are_read_as_tsv_with_na_as_null(executor):
    residues = await executor.get_protein_residues(SINGLE)
    features = await executor.get_protein_features(SINGLE)
    assert residues["results"][0]["mu"] is None and residues["results"][1]["mu"] == 2.5e-08
    assert residues["results"][0]["is_owner"] is True
    assert residues["results"][1]["file_position"] == 2
    assert residues["column_names"][-1] == "n_possible"
    # a quote in free text is data, not the start of a quoted field
    assert features["results"][0]["description"] == 'von Hippel-Lindau-like "protein"'
    assert [f["evidence_tier"] for f in features["results"]] == [
        "unknown",
        "predicted",
        "predicted",
    ]


async def test_variant_rows_are_typed_and_the_variants_without_one_are_named(executor):
    result = await executor.get_protein_variant_residues([V10, NO_ROW, V130])
    assert [(r["variant"], r["aa_pos"], r["file_position"]) for r in result["results"]] == [
        (V10, 10, 10),
        (V130, 130, 130),
    ]
    assert result["variants_without_residue_row"] == [NO_ROW]
    first = result["results"][0]
    assert first["pos"] == 1000 and first["mu_snp"] is None and first["gene_id"] is None
    assert result["results"][1]["gene_id"] == SINGLE_GENE_ID
    assert result["results"][1]["mu_snp"] == 1.5e-09


async def test_a_variant_spelled_differently_from_the_map_is_still_matched_to_its_row(executor):
    assert executor._variant_key("chrX-5-a-t") == "23:5:A:T"
    VARIANT_MAP[f"chr{V10}"] = VARIANT_MAP[V10]
    try:
        result = await executor.get_protein_variant_residues([f"chr{V10}", NO_ROW])
    finally:
        del VARIANT_MAP[f"chr{V10}"]
    assert result["variants_without_residue_row"] == [NO_ROW]


async def test_pae_whole_and_by_rows(executor, seen):
    whole = await executor.get_protein_pae(SINGLE_FILE)
    rows = await executor.get_protein_pae(SINGLE_FILE, [20, 7, 5, 6])
    n = whole["n_residues"]
    assert whole["pae"] == _pae(SINGLE_FILE) and n == 139
    assert sorted(rows["pae"]) == [5, 6, 7, 20]
    assert all(rows["pae"][p] == whole["pae"][(p - 1) * n : p * n] for p in rows["pae"])
    # the model's length is unknown to this call, so one row is asked for first
    assert [r.url.params.get("positions") for r in seen[1:]] == ["5", "6-7,20"]


async def test_a_matrix_over_the_response_limit_is_fetched_as_blocks_of_rows(
    executor, seen, monkeypatch
):
    n = BY_FILE[SINGLE_FILE]["n_residues"]
    monkeypatch.setattr(contract, "MAX_RESPONSE_BYTES", n * 50)
    result = await executor.get_protein_pae(SINGLE_FILE, None, n)
    assert result["pae"] == _pae(SINGLE_FILE)
    assert [r.url.params["positions"] for r in seen] == ["1-50", "51-100", f"101-{n}"]
    assert all(r.url.path.endswith("/rows") for r in seen)


async def test_inputs_for_some_positions_fetch_only_their_files_and_rows(executor, seen):
    inputs = await executor.get_protein_structure_inputs(variants=[V10, V130], positions=[12])
    assert inputs["accession"] == SINGLE and inputs["positions"] == [10, 12, 130]
    assert sorted(inputs["pae"][SINGLE_FILE]) == [10, 12, 130]
    windowed = await executor.get_protein_structure_inputs(protein="CENPE", positions=[100, 2700])
    assert windowed["pae"] is None
    assert len(windowed["models"]) == windowed["models"][0]["n_windows"]
    # each position's files only, and no PAE request for a protein the release has none for
    containing = {
        m["model_file_id"]
        for m in windowed["models"]
        if any(m["protein_start"] <= p <= m["protein_end"] for p in (100, 2700))
    }
    assert (
        set(windowed["structures"]) == containing < {m["model_file_id"] for m in windowed["models"]}
    )
    assert not any("/pae/" in p and WINDOWED in p for p in _paths(seen))


async def test_inputs_refuse_an_ambiguous_or_empty_selection(executor):
    two = await executor.get_protein_structure_inputs(variants=[V10, W100])
    none = await executor.get_protein_structure_inputs(variants=[NO_ROW])
    outside = await executor.get_protein_structure_inputs(protein=SINGLE, positions=[140])
    assert two["error_code"] == "ambiguous_protein" and SINGLE in two["error"]
    assert none["error_code"] == "no_residue_row"
    assert "not missense on the canonical UniProt isoform" in none["error"]
    assert outside["error_code"] == "invalid_request" and "1 to 139" in outside["error"]


@pytest.mark.parametrize(
    ("response", "code", "says"),
    [
        (
            httpx.Response(404, json={"detail": "Not Found"}),
            "not_available",
            "not available on this deployment",
        ),
        (
            httpx.Response(
                404, json={"detail": "Unknown annotation source 'protein'. Available: finngen"}
            ),
            "not_available",
            "not available on this deployment",
        ),
        (
            httpx.Response(404, json={"detail": "No model file 'x' in release r"}),
            "not_found",
            "No model file 'x'",
        ),
        (
            httpx.Response(503, json={"detail": "no release loaded"}),
            "release_not_loaded",
            "Try again",
        ),
        (
            httpx.Response(413, json={"detail": "over 16777216 bytes as JSON"}),
            "too_large",
            "positions=[...]",
        ),
        (
            httpx.Response(
                429, json={"detail": "Response exceeds", "code": "sandbox_response_bytes"}
            ),
            "too_large",
            "positions=[...]",
        ),
        (
            httpx.Response(422, json={"detail": "position 900 is past the model"}),
            "invalid_request",
            "900",
        ),
    ],
)
async def test_refusals_are_mapped_to_codes_a_caller_can_act_on(response, code, says):
    ex = _failing({"protein": response})
    client = GeneticsClient(executor=ex)
    try:
        results = [
            await ex.get_protein_models(SINGLE),
            await ex.get_protein_features(SINGLE),
            await ex.get_protein_structure_file(SINGLE_FILE),
            await ex.get_protein_pae(SINGLE_FILE, [1]),
            await ex.get_protein_variant_residues([V10]),
            await ex.get_protein_features_at_variant([V10]),
            await ex.get_3d_distance_between_variants([V10, V130]),
            await ex.get_3d_neighbourhood(protein=SINGLE, position=10),
        ]
        with pytest.raises(GeneticsError) as raised:
            await client.protein_structure(SINGLE)
    finally:
        await ex.close()
    assert {r.get("error_code") for r in results} == {code}
    assert all(says in r["error"] for r in results)
    assert raised.value.code == code and says in str(raised.value)
    assert not isinstance(raised.value, GeneticsUsageError)


async def test_an_unreachable_upstream_is_not_reported_as_a_release_still_loading():
    unreachable = httpx.Response(
        503, text="unreachable", headers={executor_module._UNREACHABLE_HEADER: "1"}
    )
    ex = _failing({"protein": unreachable})
    try:
        result = await ex.get_protein_models(SINGLE)
    finally:
        await ex.close()
    assert result.get("error_code") is None and "unreachable" in result["error"]


# --------------------------------------------------------------------------- tools


async def test_features_at_variant_gives_the_residue_its_features_and_its_plddt(executor):
    result = await executor.get_protein_features_at_variant([V10, V130, NO_ROW])
    plddt = st.parse_model(_structure(SINGLE_FILE))["plddt"].to_list()
    first, second = result["results"]
    assert first["residue_change"] == "A10V" and first["accession"] == SINGLE
    assert first["plddt"] == round(plddt[9], 2) and second["plddt"] == round(plddt[129], 2)
    assert [(f["feature_type"], f["evidence_tier"]) for f in first["features"]] == [
        ("Chain", "unknown"),
        ("Region", "predicted"),
    ]
    assert [f["feature_type"] for f in second["features"]] == ["Chain"]
    assert result["variants_without_residue_row"] == [NO_ROW]
    assert result["without_residue_row_means"] == (
        "no residue row: not missense on the canonical UniProt isoform under the release's "
        "annotation, or on a protein this deployment does not serve"
    )
    assert result["release"] == RELEASE


async def test_a_protein_without_feature_rows_has_an_empty_feature_list(executor):
    result = await executor.get_protein_features_at_variant([W100])
    assert result["results"][0]["features"] == []
    assert result["results"][0]["plddt"] is not None


async def test_distance_comes_with_pae_and_plddt_and_matches_the_geometry(executor):
    result = await executor.get_3d_distance_between_variants([V130, V10, V12])
    expected = {
        (d["position_a"], d["position_b"]): d
        for d in st.distances(_reference(SINGLE), [10, 12, 130]).to_dicts()
    }
    assert result["count"] == 3
    for pair in result["pairs"]:
        a, b = int(pair["residue_a"][1:]), int(pair["residue_b"][1:])
        want = expected[(a, b)]
        assert pair["status"] == "measured" and pair["model_file_id"] == SINGLE_FILE
        assert pair["distance"] == round(want["distance"], 2)
        assert pair["pae"] == want["pae"] and pair["pae"] is not None
        assert pair["min_plddt"] == round(want["min_plddt"], 2)


async def test_a_pair_sharing_no_model_file_gets_an_explicit_answer_and_no_number(executor):
    result = await executor.get_3d_distance_between_variants([W100, W2700])
    (pair,) = result["pairs"]
    assert pair["status"] == "no_shared_model_file"
    assert pair["distance"] is None
    assert pair["reason"] == "no single model file contains both residues"
    assert not {"pae", "min_plddt", "model_file_id"} & set(pair)


async def test_pairs_that_cannot_be_measured_say_why(executor):
    result = await executor.get_3d_distance_between_variants([V12, V12B, W100, NO_ROW])
    status = {frozenset((p["variant_a"], p["variant_b"])): p["status"] for p in result["pairs"]}
    assert status == {
        frozenset((V12, V12B)): "same_residue",
        frozenset((V12, W100)): "different_proteins",
        frozenset((V12B, W100)): "different_proteins",
    }
    assert not any("distance" in p for p in result["pairs"])
    assert result["variants_without_residue_row"] == [NO_ROW]


async def test_a_measured_pair_on_a_protein_without_pae_says_pae_is_unavailable(executor):
    VARIANT_MAP["4:5003:A:C"] = [_variant_row("4:5003:A:C", WINDOWED, "CENPE", 101)]
    try:
        result = await executor.get_3d_distance_between_variants([W100, "4:5003:A:C"])
    finally:
        del VARIANT_MAP["4:5003:A:C"]
    (pair,) = result["pairs"]
    assert pair["status"] == "measured" and pair["pae"] is None
    assert pair["distance"] > 0 and pair["min_plddt"] is not None
    assert "no PAE" in pair["pae_unavailable"]


async def test_distance_needs_two_variants_and_caps_the_batch(executor):
    one = await executor.get_3d_distance_between_variants([V10, V10])
    many = await executor.get_3d_distance_between_variants([f"1:{i}:A:T" for i in range(60)])
    assert not one["success"] and "at least 2" in one["error"]
    assert not many["success"] and "At most" in many["error"]


async def test_neighbourhood_defaults_to_15_angstrom_with_the_pae_filter(executor):
    result = await executor.get_3d_neighbourhood(variant=V10, variants=[V12, V130, W100, NO_ROW])
    want = st.neighbourhood(_reference(SINGLE), 10)["neighbours"]
    assert (result["radius"], result["pae_cutoff"], result["pae_filter_applied"]) == (
        15.0,
        15,
        True,
    )
    assert "pae_filter_note" not in result
    assert (
        sorted(n["protein_position"] for n in result["neighbours"])
        == want["protein_position"].to_list()
    )
    assert result["count"] == want.height and result["model_file_ids"] == [SINGLE_FILE]
    assert all({"distance", "pae", "plddt"} <= set(n) for n in result["neighbours"])
    assert result["neighbours"][0]["distance"] == 0.0
    assert result["centre"] == {
        "protein_position": 10,
        "residue": want.filter(pl.col("protein_position") == 10)["residue"][0],
        "plddt": result["neighbours"][0]["plddt"],
        "variant": V10,
    }
    placed = {v["variant"]: v for v in result["variants"]}
    inside = 12 in want["protein_position"].to_list()
    assert placed[V12]["in_neighbourhood"] is inside
    assert ({"distance", "pae", "plddt"} <= set(placed[V12])) is inside
    far = 130 in want["protein_position"].to_list()
    assert placed[V130]["in_neighbourhood"] is far and ("distance" in placed[V130]) is far
    assert placed[W100] == {"variant": W100, "status": "different_protein", "accession": WINDOWED}
    assert result["variants_without_residue_row"] == [NO_ROW]


async def test_neighbourhood_says_when_the_pae_filter_was_not_applied(executor):
    asked_off = await executor.get_3d_neighbourhood(
        protein=SINGLE, position=10, radius=8, apply_pae_filter=False
    )
    no_pae = await executor.get_3d_neighbourhood(protein=WINDOWED, position=100, radius=8)
    want = st.neighbourhood(_reference(SINGLE), 10, radius=8, pae_cutoff=None)["neighbours"]
    assert asked_off["pae_filter_applied"] is False and asked_off["pae_cutoff"] is None
    assert "no PAE filter was asked for" in asked_off["pae_filter_note"]
    assert asked_off["count"] == want.height
    # PAE is still reported where the release has it
    assert all(n["pae"] is not None for n in asked_off["neighbours"])
    assert no_pae["pae_filter_applied"] is False
    assert "holds no PAE" in no_pae["pae_filter_note"]
    assert all(n["pae"] is None and n["plddt"] is not None for n in no_pae["neighbours"])


async def test_neighbourhood_refuses_a_centre_it_cannot_place(executor):
    both = await executor.get_3d_neighbourhood(variant=V10, protein=SINGLE, position=10)
    neither = await executor.get_3d_neighbourhood(protein=SINGLE)
    no_row = await executor.get_3d_neighbourhood(variant=NO_ROW)
    wrong = await executor.get_3d_neighbourhood(variant=V10, protein=WINDOWED)
    fractional = await executor.get_3d_neighbourhood(protein=SINGLE, position=10, pae_cutoff=7.5)
    assert not both["success"] and not neither["success"]
    assert no_row["error_code"] == "no_residue_row" and NO_ROW in no_row["error"]
    assert wrong["error_code"] == "not_found" and SINGLE in wrong["error"]
    assert not fractional["success"] and "whole number" in fractional["error"]


# --------------------------------------------------------------------------- SDK


async def test_sdk_functions_return_frames_bytes_and_a_protein(executor):
    client = GeneticsClient(executor=executor)
    models = await client.protein_models(SINGLE)
    variants = await client.protein_variants([V10, NO_ROW])
    assert isinstance(models, pl.DataFrame) and models["n_residues"].to_list() == [139]
    assert (await client.protein_residues(SINGLE)).height == 2
    assert (await client.protein_features(SINGLE))["start"].dtype.is_integer()
    assert variants["variant"].to_list() == [V10] and variants["aa_pos"].to_list() == [10]
    assert await client.protein_structure_file(SINGLE_FILE) == _structure(SINGLE_FILE)
    assert await client.protein_pae(SINGLE_FILE) == _pae(SINGLE_FILE)
    assert sorted(await client.protein_pae(SINGLE_FILE, [3, 4])) == [3, 4]
    with pytest.raises(GeneticsUsageError):
        await client.protein_structure()
    with pytest.raises(GeneticsUsageError):
        await client.protein_variants([])


async def test_sdk_protein_structure_computes_what_the_whole_inputs_compute(executor):
    client = GeneticsClient(executor=executor)
    by_variants = await client.protein_structure(variants=[V10, V130])
    whole = await client.protein_structure(SINGLE)
    reference = _reference(SINGLE)
    assert isinstance(whole, st.Protein) and whole.accession == SINGLE
    assert st.distances(by_variants, [10, 130]).equals(st.distances(reference, [10, 130]))
    assert st.neighbourhood(by_variants, 10)["neighbours"].equals(
        st.neighbourhood(reference, 10)["neighbours"]
    )
    assert (st.adjacency(whole)["adjacency"] == st.adjacency(reference)["adjacency"]).all()
    # rows of two residues do not make a whole matrix, and the geometry says so
    with pytest.raises(GeneticsUsageError):
        st.adjacency(by_variants)
    no_pae = await client.protein_structure(SINGLE, positions=[10, 130], pae=False)
    assert st.distances(no_pae, [10, 130], pae=False)["pae"].to_list() == [None]


async def test_sdk_refuses_an_ambiguous_protein_with_the_choices(executor):
    client = GeneticsClient(executor=executor)
    with pytest.raises(GeneticsError) as raised:
        await client.protein_structure(variants=[V10, W100])
    assert raised.value.code == "ambiguous_protein"
    assert SINGLE in str(raised.value) and WINDOWED in str(raised.value)


@pytest.fixture
def sdk_for_profile(monkeypatch):
    import genetics_mcp_server.sdk as sdk

    def _load(profile):
        if profile is None:
            monkeypatch.delenv("CONFIG_PROFILE", raising=False)
        else:
            monkeypatch.setenv("CONFIG_PROFILE", profile)
        return importlib.reload(sdk)

    yield _load
    monkeypatch.undo()
    importlib.reload(sdk)


def test_sdk_binds_the_functions_on_daly_only(sdk_for_profile):
    sdk = sdk_for_profile("daly")
    assert set(SDK_FUNCTIONS) <= set(sdk.__all__)
    assert all(callable(getattr(sdk, name)) for name in SDK_FUNCTIONS)
    for profile in ("finngen", None, "unknown"):
        sdk = sdk_for_profile(profile)
        assert not set(SDK_FUNCTIONS) & set(sdk.__all__)
        assert not set(SDK_FUNCTIONS) & set(dir(sdk))
        # the geometry is pure computation and stays; only the fetches are a deployment's
        assert "structure" in sdk.__all__


def test_sdk_gating_table_is_the_tool_profile_rule():
    import genetics_mcp_server.sdk as sdk

    gated = {name for name in sdk._PROFILE_GATED if name.startswith("protein_")}
    assert gated == set(SDK_FUNCTIONS)
    assert all(set(sdk._PROFILE_GATED[n]) == set(PROTEIN_STRUCTURE_PROFILES) for n in gated)


async def test_list_capabilities_describes_the_structure_module():
    ex = ToolExecutor()
    try:
        index = await ex.list_capabilities()
        module = await ex.list_capabilities("structure")
    finally:
        await ex.close()
    listed = next(m for m in index["modules"] if m["module"] == "structure")
    assert listed["names"] == list(st.__all__)
    assert "def neighbourhood(" in module["signatures"]
    assert module["usage"].endswith("genetics.structure.parse_model")
    definition = next(
        t for t in get_anthropic_tools(code_execution=True) if t["name"] == "list_capabilities"
    )
    assert definition["input_schema"]["properties"]["module"]["enum"] == list(
        executor_module._SDK_MODULES
    )


# --------------------------------------------------------------------------- where offered


def test_the_three_tools_are_defined_and_are_the_gated_set():
    defined = {t["name"]: t for t in TOOL_DEFINITIONS}
    assert PROTEIN_STRUCTURE_TOOLS <= set(defined)
    for name in PROTEIN_STRUCTURE_TOOLS:
        assert defined[name]["sdk_replaceable"] is True
        assert callable(getattr(ToolExecutor, name))
    for name in ("get_3d_distance_between_variants", "get_3d_neighbourhood"):
        text = defined[name]["description"]
        assert "PAE" in text.upper() and "PLDDT" in text.upper()
    assert "no_shared_model_file" in defined["get_3d_distance_between_variants"]["description"]
    assert "pae_filter_applied" in defined["get_3d_neighbourhood"]["description"]


@pytest.mark.parametrize(
    ("profile", "enabled", "offered"),
    [
        ("daly", True, True),
        ("daly", False, False),
        ("finngen", True, False),
        (None, True, False),
        ("unknown", True, False),
    ],
)
def test_offered_only_where_enabled_on_a_profile_that_serves_the_release(profile, enabled, offered):
    disabled = Settings(config_profile=profile, protein_structure_enabled=enabled).disabled_tools
    names = {t["name"] for t in get_anthropic_tools(code_execution=False, disabled_tools=disabled)}
    assert (PROTEIN_STRUCTURE_TOOLS <= names) is offered
    assert bool(PROTEIN_STRUCTURE_TOOLS & names) is offered


def test_reads_the_enabled_flag_from_the_environment(monkeypatch):
    monkeypatch.setenv("CONFIG_PROFILE", "daly")
    monkeypatch.delenv("PROTEIN_STRUCTURE_ENABLED", raising=False)
    assert PROTEIN_STRUCTURE_TOOLS <= Settings().disabled_tools
    monkeypatch.setenv("PROTEIN_STRUCTURE_ENABLED", "true")
    assert not PROTEIN_STRUCTURE_TOOLS & Settings().disabled_tools
    monkeypatch.setenv("CONFIG_PROFILE", "finngen")
    assert PROTEIN_STRUCTURE_TOOLS <= Settings().disabled_tools


def test_sdk_replaceable_so_the_code_surface_uses_the_sdk():
    names = {t["name"] for t in get_anthropic_tools(code_execution=True)}
    assert not PROTEIN_STRUCTURE_TOOLS & names


def _feature(monkeypatch, profile: str | None, enabled: bool) -> None:
    for name, value in (("CONFIG_PROFILE", profile), ("PROTEIN_STRUCTURE_ENABLED", enabled)):
        if value in (None, False):
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, "true" if value is True else value)
    get_settings.cache_clear()


@pytest.fixture
def feature(monkeypatch):
    """Sets the deployment's profile and flag for the process-wide settings."""
    yield lambda profile, enabled: _feature(monkeypatch, profile, enabled)
    monkeypatch.undo()
    get_settings.cache_clear()


OFF_STATES = [("daly", False), ("finngen", True), (None, True), ("unknown", True)]


def _mcp_exclusions_with_the_feature_on() -> set[str]:
    """What /mcp withholds on a deployment that serves the layer.

    With the feature off the settings half of `_mcp_disabled` already names the three tools,
    and a test built on it stays green with the literal emptied.
    """
    serving = Settings(config_profile="daly", protein_structure_enabled=True).disabled_tools
    assert not PROTEIN_STRUCTURE_TOOLS & serving
    return serving | _mcp_hardcoded_exclusions()


def _mcp(disabled: set[str], executor: ToolExecutor):
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("test")
    if not hasattr(server, "_tool_manager"):
        pytest.skip("FastMCP version exposes no _tool_manager; cannot introspect tools")
    register_mcp_tools(server, executor, disabled_tools=disabled)
    return server._tool_manager


def test_not_on_mcp_where_the_deployment_serves_the_layer(executor):
    assert PROTEIN_STRUCTURE_TOOLS <= _mcp_hardcoded_exclusions()
    # the handlers exist, so it is the literal and nothing else that keeps them off
    assert PROTEIN_STRUCTURE_TOOLS <= set(_mcp(set(), executor)._tools)
    deployed = _mcp(_mcp_exclusions_with_the_feature_on(), executor)
    assert not PROTEIN_STRUCTURE_TOOLS & set(deployed._tools)
    assert "get_variant_annotations" in deployed._tools


async def test_the_protein_source_is_refused_on_mcp_and_served_where_nothing_is_withheld(
    executor, seen, feature
):
    feature("daly", True)
    arguments = {"variants": [V10], "source": "protein"}
    deployed = _mcp(_mcp_exclusions_with_the_feature_on(), executor)
    refused = await deployed.call_tool("get_variant_annotations", arguments)
    assert refused == executor_module.PROTEIN_STRUCTURE_NOT_AVAILABLE
    assert not seen
    served = await _mcp(set(), executor).call_tool("get_variant_annotations", arguments)
    assert served["success"] and served["results"][0]["aa_pos"] == "10"


@pytest.mark.parametrize("withheld", sorted(PROTEIN_STRUCTURE_TOOLS))
async def test_withholding_any_of_the_tools_from_mcp_withholds_the_source(
    executor, feature, withheld
):
    feature("daly", True)
    manager = _mcp({withheld}, executor)
    result = await manager.call_tool(
        "get_variant_annotations", {"variants": [V10], "source": "protein"}
    )
    assert result["error_code"] == "not_available"


@pytest.mark.parametrize(("profile", "enabled"), OFF_STATES)
async def test_the_protein_source_is_refused_where_the_feature_is_off(
    executor, seen, feature, profile, enabled
):
    feature(profile, enabled)
    for query in ({"variants": [V10]}, {"variant": V10}, {"gene": "VHLL"}):
        result = await executor.get_variant_annotations(**query, source="protein")
        assert result == executor_module.PROTEIN_STRUCTURE_NOT_AVAILABLE
    with pytest.raises(GeneticsError) as raised:
        await GeneticsClient(executor=executor).variant_annotation(variants=[V10], source="protein")
    assert raised.value.code == "not_available"
    assert not seen


async def test_the_protein_source_is_served_where_the_feature_is_on(executor, feature):
    feature("daly", True)
    result = await executor.get_variant_annotations(variants=[V10], source="protein")
    assert result["success"] and result["source"] == "protein" and result["count"] == 1


async def test_the_sandbox_has_no_flag_and_leaves_the_source_to_results_api(
    executor, feature, monkeypatch
):
    feature(None, False)
    monkeypatch.setattr(
        executor_module, "_resolve_settings", lambda: executor_module._PRUNED_INSTALL_SETTINGS
    )
    frame = await GeneticsClient(executor=executor).variant_annotation(
        variants=[V10], source="protein"
    )
    assert frame["aa_pos"].to_list() == ["10"]


async def test_the_layers_own_calls_do_not_depend_on_the_servers_flag(executor, feature):
    # the SDK functions are gated by profile at import and the tools by being offered
    feature(None, False)
    assert (await executor.get_protein_variant_residues([V10]))["success"]
    assert (await executor.get_protein_features_at_variant([V10]))["success"]


# --------------------------------------------------------------------------- arguments


@pytest.mark.parametrize("bad", ["", "   ", "a/b", "TP 53", "..\\x"])
async def test_a_name_that_cannot_be_a_path_segment_is_refused_before_any_request(
    executor, seen, bad
):
    results = [
        await executor.get_protein_models(bad),
        await executor.get_protein_residues(bad),
        await executor.get_protein_features(bad),
        await executor.get_protein_structure_file(bad),
        await executor.get_protein_pae(bad),
        await executor.get_protein_pae(bad, [1]),
        await executor.get_protein_structure_inputs(protein=bad, positions=[1]),
        await executor.get_3d_neighbourhood(protein=bad or " ", position=1),
    ]
    # a route miss would come back as FastAPI's own 404 and be read as "no such feature"
    assert {r.get("error_code") for r in results} == {"invalid_request"}
    assert not seen


@pytest.mark.parametrize(
    "protein", [SINGLE, SINGLE.lower(), "VHLL", "vhll", SINGLE_GENE_ID, f"{SINGLE_GENE_ID}.4"]
)
async def test_neighbourhood_takes_the_centres_protein_by_accession_symbol_or_gene_id(
    executor, protein
):
    result = await executor.get_3d_neighbourhood(variant=V130, protein=protein)
    assert result["success"] and result["accession"] == SINGLE
    assert result["centre"]["protein_position"] == 130


async def test_variants_come_back_spelled_as_the_caller_wrote_them(executor):
    written = [VX, f"chr{V10}"]
    VARIANT_MAP[f"chr{V10}"] = VARIANT_MAP[V10]
    try:
        features = await executor.get_protein_features_at_variant(written)
        distance = await executor.get_3d_distance_between_variants(written)
        near = await executor.get_3d_neighbourhood(variant=VX, variants=[f"chr{V10}"])
    finally:
        del VARIANT_MAP[f"chr{V10}"]
    assert [r["variant"] for r in features["results"]] == written
    (pair,) = distance["pairs"]
    assert {pair["variant_a"], pair["variant_b"]} == set(written)
    assert near["centre"]["variant"] == VX
    assert [v["variant"] for v in near["variants"]] == [f"chr{V10}"]
    # the SDK frame keeps the map's own spelling, which is what joins to the views
    rows = (await executor.get_protein_variant_residues([VX]))["results"]
    assert rows[0]["variant"] == "23:700:A:T"


# --------------------------------------------------------------------------- file positions


async def test_plddt_is_read_at_the_file_position_in_a_later_window(executor):
    result = await executor.get_protein_features_at_variant([W2700])
    (entry,) = result["results"]
    plddt = st.parse_model(_structure(entry["model_file_id"]))["plddt"].to_list()
    assert entry["protein_position"] == 2700 != entry["file_position"]
    assert len(plddt) < 2700
    assert entry["plddt"] == round(plddt[entry["file_position"] - 1], 2)


async def test_pae_rows_are_asked_for_at_file_positions_in_a_later_window(executor, seen):
    result = await executor.get_3d_distance_between_variants([P4401, P4402])
    rows = [m for m in MODELS if m["accession"] == PAE_WINDOWED]
    reference = st.protein(
        rows,
        {PAE_WINDOWED_F2: _structure(PAE_WINDOWED_F2)},
        {PAE_WINDOWED_F2: {601: _pae_row(PAE_WINDOWED_F2, 601)}},
    )
    (want,) = st.distances(reference, [4401, 4402]).to_dicts()
    (pair,) = result["pairs"]
    assert pair["status"] == "measured" and pair["model_file_id"] == PAE_WINDOWED_F2
    assert pair["distance"] == round(want["distance"], 2)
    assert pair["pae"] == want["pae"] and pair["pae"] is not None
    asked = {
        r.url.path.split("/")[-2]: r.url.params["positions"]
        for r in seen
        if r.url.path.endswith("/rows")
    }
    assert asked == {"AF3-C9JG08-F1-model_v6": "4401-4402", PAE_WINDOWED_F2: "601-602"}


# --------------------------------------------------------------------------- prompt


@pytest.mark.parametrize("variant", sorted(PROMPT_VARIANTS))
def test_prompt_block_follows_the_tools(variant):
    everything = {t["name"] for t in get_anthropic_tools(code_execution=False)}
    heading = "### Protein Structure (frozen release)"
    offered = default_system_prompt(tool_names=everything, variant=variant)
    withheld = default_system_prompt(
        tool_names=everything - PROTEIN_STRUCTURE_TOOLS, variant=variant
    )
    assert heading in offered and heading not in withheld
    block = offered.split(heading)[1].split("\n###")[0]
    assert all(f"`{name}`" in block for name in PROTEIN_STRUCTURE_TOOLS)
    assert "`get_protein_annotations`" in block
    assert re.search(r"every distance with its PAE and pLDDT", block)
    assert (
        "\"not missense on the canonical UniProt isoform under the release's annotation, or on "
        'a protein the deployment does not serve", nothing stronger' in block
    )
    assert len(block) < 1200


CODE_HEADING = "### Protein Structure (frozen release)"


def _code_surface_prompt(variant: str) -> str:
    names = {t["name"] for t in get_anthropic_tools(code_execution=True)}
    assert not PROTEIN_STRUCTURE_TOOLS & names and "run_analysis" in names
    return default_system_prompt(tool_names=names, variant=variant)


@pytest.mark.parametrize("variant", sorted(PROMPT_VARIANTS))
def test_code_surface_prompt_carries_the_sdk_block_where_the_layer_is_served(variant, feature):
    feature("daly", False)
    without = _code_surface_prompt(variant)
    feature("daly", True)
    prompt = _code_surface_prompt(variant)
    assert CODE_HEADING not in without
    assert prompt.count(CODE_HEADING) == 1
    block = CODE_HEADING + prompt.split(CODE_HEADING)[1].split("\n###")[0]
    assert len(prompt) - len(without) == len(block) + 1 <= 1050
    for name in (
        "`genetics.protein_variants`",
        "`genetics.protein_features`/`protein_residues`",
        "`genetics.protein_structure(...)`",
        "`genetics.structure` (`distances`, `neighbourhood`, `view` for a 3D view in the chat)",
        "`get_protein_annotations`",
        "its PAE and min pLDDT",
        "PAE (above about 15)",
        "pLDDT below 70",
        "below 50",
        "No distance for a pair sharing no model file",
        "Say which source a statement comes from",
        "\"not missense on the canonical UniProt isoform under the release's annotation, or "
        'on a protein the deployment does not serve", nothing stronger',
    ):
        assert name in block, name
    # the chat tools are not on this surface, so the block may not send the model to them
    assert not any(name in block for name in PROTEIN_STRUCTURE_TOOLS)


@pytest.mark.parametrize("variant", sorted(PROMPT_VARIANTS))
@pytest.mark.parametrize(("profile", "enabled"), OFF_STATES)
def test_code_surface_prompt_says_nothing_where_the_layer_is_not_served(
    variant, feature, profile, enabled
):
    feature(profile, enabled)
    assert CODE_HEADING not in _code_surface_prompt(variant)


@pytest.mark.parametrize("variant", sorted(PROMPT_VARIANTS))
def test_the_two_prompt_blocks_never_appear_together(variant, feature):
    feature("daly", True)
    nocode = {t["name"] for t in get_anthropic_tools(code_execution=False)}
    prompt = default_system_prompt(tool_names=nocode, variant=variant)
    assert prompt.count(CODE_HEADING) == 1 and "genetics.protein_variants" not in prompt
