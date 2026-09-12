"""Tests for the measured-vs-predicted comparison.

No network, no API key and no BigQuery: the Atlas boundary is the same fake AtlasClient
tests/test_alphagenome.py injects, and the db-api boundary is `query_database`, replaced on
the executor with a stub that answers in db-api's own shape (columns plus value rows).

The rules under test are measurements, not presentation choices, so each is pinned:
a signed modality reports direction agreement both ways; a splice modality reports NO
direction even when both signs are in hand; a tier-4 modality says nothing was measured
rather than inventing a pairing; a measurement in another tissue is labelled cross-tissue;
and every measured value carries `measured: true` and names where it came from.
"""

import pytest
from test_alphagenome import FakeAtlas, _client, entry

from genetics_mcp_server.tools import alphagenome
from genetics_mcp_server.tools import alphagenome_comparison as comparison


@pytest.fixture(autouse=True)
def clear_module_state():
    alphagenome._METADATA_CACHE.clear()
    alphagenome._LIMITER.clear()
    alphagenome._CACHE.clear()
    yield
    alphagenome._METADATA_CACHE.clear()
    alphagenome._LIMITER.clear()
    alphagenome._CACHE.clear()


TOOL = "compare_alphagenome_with_measured"

_CS_COLUMNS = [
    "variant",
    "data_type",
    "trait",
    "cell_type",
    "beta",
    "se",
    "pip",
    "cs_id",
    "mlog10p",
    "resource",
    "dataset",
]
_MPRA_COLUMNS = [
    "variant",
    "cell_line",
    "log2Skew",
    "log2Skew_se",
    "log2Skew_mlog10p",
    "emVar",
    "active",
    "resource",
    "dataset",
]


def _cs_row(beta, data_type="caQTL", trait="chr1-100-200", cell_type="CD4_T", variant="1:100:A:G"):
    return [variant, data_type, trait, cell_type, beta, 0.1, 0.9, "cs1", 8.0, "finngen_caqtl", "finngen"]


def _mpra_row(skew, cell_line="K562", variant="1:100:A:G"):
    return [variant, cell_line, skew, 0.1, 5.0, True, True, "siraj_mpra", "siraj_mpra"]


def _executor(fake_atlas, answers):
    """A ServerToolExecutor with a fake Atlas and a stubbed db-api.

    `answers` is {view: (columns, rows)}; anything not named answers empty, which is what a
    variant this suite has never measured looks like.
    """
    from genetics_mcp_server.tools.orchestration import ServerToolExecutor

    executor = ServerToolExecutor()
    executor.__dict__["alphagenome"] = _client(fake_atlas)
    seen: list[str] = []

    async def query_database(sql, max_rows=1000, dry_run=False):
        seen.append(sql)
        for view, answer in answers.items():
            if f"FROM {view} " in sql:
                if isinstance(answer, dict):
                    return answer
                columns, rows = answer
                return {"success": True, "columns": columns, "rows": rows}
        return {"success": True, "columns": [], "rows": []}

    executor.query_database = query_database
    executor.__dict__["_seen_sql"] = seen
    return executor


# --------------------------------------------------------------------------- #
# the pairings themselves
# --------------------------------------------------------------------------- #


def test_every_calibrated_modality_has_a_substrate_and_no_tier_4_one_does():
    """The two halves of the same fact: a modality carries a population rho exactly when
    something here was measured to produce it, so an entry added to one table and not the
    other is a pairing with no evidence or evidence with no pairing."""
    for name, modality in alphagenome.MODALITIES.items():
        pairings = comparison.pairings_for(name)
        assert bool(pairings) is (modality.population_rho is not None), name


def test_no_pairing_points_at_a_predictions_view():
    """variant_effect_v holds ChromBPNet and FLARE OUTPUT, not measurements. A row from it
    on the measured side of this response would be a model's number labelled as a person's."""
    views = {p.view for ps in comparison.PAIRINGS.values() for p in ps}
    assert views == {"credible_sets_v", "mpra_v"}


@pytest.mark.parametrize(
    "variant,expected",
    [
        ({"chromosome": "chr1", "position": 100, "reference_bases": "A", "alternate_bases": "G"}, "1:100:A:G"),
        ({"chromosome": "chrX", "position": 5, "reference_bases": "T", "alternate_bases": "C"}, "23:5:T:C"),
    ],
)
def test_the_suite_spelling_of_a_variant_is_what_the_views_use(variant, expected):
    assert comparison.suite_variant_id(variant) == expected


def test_the_sql_names_the_variant_the_chromosome_and_the_requested_context():
    sql = comparison.credible_sets_sql(["1:100:A:G"], ["caQTL"], "CD4 T")
    assert "chr IN (1)" in sql and "'1:100:A:G'" in sql and "'caQTL'" in sql
    # matched rows must survive the row cap, so the ranking is the server's job
    assert sql.index("CASE WHEN") < sql.index("ABS(beta)")


# --------------------------------------------------------------------------- #
# the science rules
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "predicted,measured,direction",
    [(0.5, 0.42, "agrees"), (-0.5, -0.42, "agrees"), (0.5, -0.42, "disagrees"), (-0.5, 0.42, "disagrees")],
)
async def test_a_signed_modality_reports_direction_agreement_both_ways(
    predicted, measured, direction
):
    fake = FakeAtlas(responses={"chr1:100": {"DNASE": [entry([predicted])]}})
    executor = _executor(fake, {"credible_sets_v": (_CS_COLUMNS, [_cs_row(measured)])})
    result = await executor.compare_alphagenome_with_measured(
        ["1:100:A:G"], modalities=["DNASE"]
    )
    block = result["results"][0]["modalities"]["DNASE"]
    assert block["concordance"]["direction"] == direction
    assert block["concordance"]["predicted_value"] == predicted
    assert block["concordance"]["measured_value"] == measured


async def test_a_splice_modality_never_reports_a_direction_even_with_two_signs():
    """sQTL beta orients to a leafcutter intron cluster and the splice delta has no
    corresponding orientation. Both signs are present in the inputs here and NEITHER
    reaches the output: no `direction` key anywhere, and no word that reads as one."""
    fake = FakeAtlas(responses={"chr1:100": {"SPLICE_SITES": [entry([-0.4], gene=("ENSG1", "GENEA"))]}})
    executor = _executor(
        fake,
        {"credible_sets_v": (_CS_COLUMNS, [_cs_row(-0.33, data_type="sQTL", trait="GENEA")])},
    )
    result = await executor.compare_alphagenome_with_measured(
        ["1:100:A:G"], modalities=["SPLICE_SITES"]
    )
    block = result["results"][0]["modalities"]["SPLICE_SITES"]
    concordance = block["concordance"]

    assert "direction" not in concordance
    assert concordance["direction_reported"] is False
    assert "agree" not in repr(block).lower()
    # and both sides are unsigned, so the direction cannot be reconstructed from the pair
    assert concordance["predicted_value"] == 0.4
    assert concordance["measured_value"] == pytest.approx(0.33)  # signed input, abs() here
    assert all(m["value"] >= 0 for m in block["measurements"])


async def test_a_tier_4_modality_says_nothing_was_measured_rather_than_pairing_something():
    fake = FakeAtlas(responses={"chr1:100": {"CHIP_TF": [entry([0.5])]}})
    executor = _executor(fake, {"credible_sets_v": (_CS_COLUMNS, [_cs_row(0.42)])})
    result = await executor.compare_alphagenome_with_measured(
        ["1:100:A:G"], modalities=["CHIP_TF"]
    )
    block = result["results"][0]["modalities"]["CHIP_TF"]
    assert block["measured_substrates"] == []
    assert block["measurements"] == []
    assert block["concordance"] is None
    assert "nothing measured to compare against" in block["note"]
    assert block["validation"]["status"] == "unvalidated"
    assert block["validation"]["population_rho"] is None


async def test_a_measurement_in_another_tissue_is_labelled_cross_tissue():
    fake = FakeAtlas(responses={"chr1:100": {"DNASE": [entry([0.5])]}})
    executor = _executor(
        fake, {"credible_sets_v": (_CS_COLUMNS, [_cs_row(0.42, cell_type="liver")])}
    )
    result = await executor.compare_alphagenome_with_measured(
        ["1:100:A:G"], cell_type="K562", modalities=["DNASE"]
    )
    block = result["results"][0]["modalities"]["DNASE"]
    [measurement] = block["measurements"]
    assert measurement["context_match"] == "cross_tissue"
    assert measurement["cross_tissue"] is True
    assert block["concordance"]["context_match"] == "cross_tissue"
    assert "cross-tissue" in block["note"]


async def test_a_measurement_in_the_requested_cell_line_is_matched():
    fake = FakeAtlas(responses={"chr1:100": {"CHIP_HISTONE": [entry([0.5])]}})
    executor = _executor(fake, {"mpra_v": (_MPRA_COLUMNS, [_mpra_row(0.7, cell_line="K562")])})
    result = await executor.compare_alphagenome_with_measured(
        ["1:100:A:G"], cell_type="K562", modalities=["CHIP_HISTONE"]
    )
    block = result["results"][0]["modalities"]["CHIP_HISTONE"]
    [measurement] = block["measurements"]
    assert measurement["context_match"] == "matched"
    assert measurement["cross_tissue"] is False
    assert "note" not in block


async def test_every_measured_value_carries_the_flag_and_its_source():
    """The central design constraint: a prediction sits next to a measurement in one
    response, so no reader may have to infer which is which."""
    fake = FakeAtlas(responses={"chr1:100": {"DNASE": [entry([0.5])]}})
    executor = _executor(
        fake,
        {
            "credible_sets_v": (_CS_COLUMNS, [_cs_row(0.42)]),
            "mpra_v": (_MPRA_COLUMNS, [_mpra_row(0.7)]),
        },
    )
    result = await executor.compare_alphagenome_with_measured(
        ["1:100:A:G"], modalities=["DNASE"]
    )
    block = result["results"][0]["modalities"]["DNASE"]

    assert len(block["measurements"]) == 2
    for measurement in block["measurements"]:
        assert measurement["measured"] is True
        source = measurement["source"]
        assert source["view"] in ("credible_sets_v", "mpra_v")
        assert source["column"] in ("beta", "log2Skew")
        assert source["assay"] and source["resource"]
    assert block["prediction"]["measured"] is False
    assert block["prediction"]["source"] == "AlphaGenome (Google DeepMind)"
    # the envelope cannot claim one truth value for a payload holding both kinds
    assert "measured" not in result
    assert result["data_kind"] == "prediction_vs_measurement"


async def test_the_population_rho_belongs_to_the_pairing_and_says_so():
    """DNASE was calibrated against two substrates and correlates differently with each, so
    the number shown is the pair's own — and `rho_scope` names what it ranges over."""
    fake = FakeAtlas(responses={"chr1:100": {"DNASE": [entry([0.5])]}})
    executor = _executor(fake, {"credible_sets_v": (_CS_COLUMNS, [_cs_row(0.42)])})
    result = await executor.compare_alphagenome_with_measured(
        ["1:100:A:G"], modalities=["DNASE"]
    )
    substrates = result["results"][0]["modalities"]["DNASE"]["measured_substrates"]
    assert {s["substrate"]: s["population_rho"] for s in substrates} == {
        "caQTL": 0.478,
        "MPRA": 0.503,
    }
    assert all(s["rho_scope"] == "population" for s in substrates)
    # nothing per-variant is computed anywhere in the concordance
    concordance = result["results"][0]["modalities"]["DNASE"]["concordance"]
    assert set(concordance) <= {
        "quantity",
        "predicted_value",
        "measured_value",
        "compared_against",
        "context_match",
        "direction",
        "why_no_direction",
    }


async def test_a_gene_resolved_modality_compares_the_gene_that_was_measured():
    """An eQTL beta is measured on one gene, and RNA_SEQ answers per gene: pairing the
    measurement with the strongest predicted gene instead would compare two genes."""
    fake = FakeAtlas(
        responses={
            "chr1:100": {
                "RNA_SEQ": [
                    entry([0.9], gene=("ENSG1", "LOUD")),
                    entry([-0.2], gene=("ENSG2", "QUIET")),
                ]
            }
        }
    )
    executor = _executor(
        fake,
        {"credible_sets_v": (_CS_COLUMNS, [_cs_row(-0.5, data_type="eQTL", trait="QUIET")])},
    )
    result = await executor.compare_alphagenome_with_measured(
        ["1:100:A:G"], modalities=["RNA_SEQ"]
    )
    block = result["results"][0]["modalities"]["RNA_SEQ"]
    assert block["prediction"]["gene"] == "QUIET"
    assert block["concordance"]["predicted_value"] == pytest.approx(-0.2)
    assert block["concordance"]["direction"] == "agrees"


async def test_a_gene_resolved_modality_reports_no_direction_when_the_measured_gene_was_not_scored():
    """`genes` is capped, so the gene an eQTL beta was measured on need not be among the
    ones AlphaGenome scored. The block-level value is then the strongest predicted gene,
    and an agreement between it and that beta would be a claim about two different genes."""
    fake = FakeAtlas(
        responses={
            "chr1:100": {
                "RNA_SEQ": [
                    entry([0.9], gene=("ENSG1", "LOUD")),
                    entry([0.4], gene=("ENSG2", "QUIET")),
                ]
            }
        }
    )
    executor = _executor(
        fake,
        {
            "credible_sets_v": (
                _CS_COLUMNS,
                [_cs_row(0.5, data_type="eQTL", trait="UNSEEN_GENE")],
            )
        },
    )
    result = await executor.compare_alphagenome_with_measured(
        ["1:100:A:G"], modalities=["RNA_SEQ"]
    )
    concordance = result["results"][0]["modalities"]["RNA_SEQ"]["concordance"]

    assert "direction" not in concordance
    assert "UNSEEN_GENE" in concordance["why_no_direction"]
    assert "LOUD" in concordance["why_no_direction"]
    # both numbers still reach the reader, honestly unpaired
    assert concordance["predicted_value"] == pytest.approx(0.9)
    assert concordance["measured_value"] == pytest.approx(0.5)


async def test_a_substrate_whose_lookup_failed_is_marked_even_when_another_answered():
    """DNASE is paired with two substrates. When one view is down and the other answers,
    the modality has measurements, so the empty-measurements note never fires -- and a
    substrate listed with only its rho reads as one that was consulted and had nothing."""
    fake = FakeAtlas(responses={"chr1:100": {"DNASE": [entry([0.5])]}})
    executor = _executor(
        fake,
        {
            "credible_sets_v": (_CS_COLUMNS, [_cs_row(0.42)]),
            "mpra_v": {"success": False, "error": "HTTP 503: upstream"},
        },
    )
    result = await executor.compare_alphagenome_with_measured(
        ["1:100:A:G"], modalities=["DNASE"]
    )
    block = result["results"][0]["modalities"]["DNASE"]
    by_view = {s["view"]: s for s in block["measured_substrates"]}

    assert block["measurements"]
    assert by_view["mpra_v"]["lookup"] == "failed"
    assert "lookup" not in by_view["credible_sets_v"]
    assert "the measured lookup in mpra_v failed" in block["note"]


def test_a_cell_type_too_long_to_quote_drops_the_ranking_rather_than_raising():
    """`quote_literal` caps a literal at 128 characters. A needle that long cannot match
    any context, so losing the SQL ranking costs nothing -- and the sibling prediction tool
    answers for the same input, so raising only here would make the pair disagree."""
    long_cell_type = "a" * 200
    for sql in (
        comparison.credible_sets_sql(["1:100:A:G"], ["caQTL"], long_cell_type),
        comparison.mpra_sql(["1:100:A:G"], long_cell_type),
    ):
        assert sql.startswith("SELECT ")
        assert "CASE WHEN" not in sql
        assert long_cell_type not in sql


async def test_no_measured_rows_is_distinguished_from_a_failed_lookup():
    """Absence of evidence and absence of an answer look identical in the rows; only the
    note tells them apart, and only one of them means this suite measured nothing."""
    fake = FakeAtlas(responses={"chr1:100": {"ATAC": [entry([0.5])]}})
    silent = _executor(fake, {})
    result = await silent.compare_alphagenome_with_measured(["1:100:A:G"], modalities=["ATAC"])
    assert "for this variant in this suite" in result["results"][0]["modalities"]["ATAC"]["note"]

    alphagenome._CACHE.clear()
    broken = _executor(
        fake, {"credible_sets_v": {"success": False, "error": "HTTP 503: upstream"}}
    )
    result = await broken.compare_alphagenome_with_measured(["1:100:A:G"], modalities=["ATAC"])
    block = result["results"][0]["modalities"]["ATAC"]
    assert "the measured lookup in credible_sets_v failed" in block["note"]
    assert result["measured_lookup"]["credible_sets_v"]["success"] is False


async def test_a_tier_4_only_request_asks_db_api_for_nothing():
    """No substrate, so no query: the comparison must not spend a db-api round trip
    discovering that it has nothing to compare."""
    fake = FakeAtlas(responses={"chr1:100": {"CAGE": [entry([0.5])]}})
    executor = _executor(fake, {})
    await executor.compare_alphagenome_with_measured(["1:100:A:G"], modalities=["CAGE"])
    assert executor.__dict__["_seen_sql"] == []


# --------------------------------------------------------------------------- #
# the tool surface
# --------------------------------------------------------------------------- #


def _definition(name):
    from genetics_mcp_server.tools.definitions import all_local_tool_definitions

    [tool] = [t for t in all_local_tool_definitions() if t["name"] == name]
    return tool


def test_the_opt_in_wording_covers_both_tools_identically():
    """The user ruled that this wording is the entire enforcement, so a second capability
    under a second name must not carry a second, drifting copy of it."""
    from genetics_mcp_server.tools.definitions import _ALPHAGENOME_OPT_IN

    for name in ("get_alphagenome_variant_predictions", TOOL):
        assert _ALPHAGENOME_OPT_IN in _definition(name)["description"]


def test_the_comparison_tool_is_withheld_from_mcp_and_leaves_with_the_key():
    from genetics_mcp_server import mcp_server
    from genetics_mcp_server.config.settings import Settings

    assert TOOL in mcp_server._mcp_disabled
    assert TOOL not in Settings(alphagenome_enabled=True, alphagenome_api_key="k").disabled_tools
    assert TOOL in Settings(alphagenome_enabled=True, alphagenome_api_key="").disabled_tools
    assert TOOL in Settings(alphagenome_enabled=True, alphagenome_api_key=None).disabled_tools
    assert TOOL in Settings(alphagenome_enabled=False, alphagenome_api_key="k").disabled_tools


def test_the_comparison_tool_is_on_both_surfaces():
    from genetics_mcp_server.tools.definitions import resolve_tools

    assert _definition(TOOL)["sdk_replaceable"] is False
    for code_execution in (True, False):
        assert TOOL in {t["name"] for t in resolve_tools(code_execution)}


def test_the_declared_modalities_are_exactly_the_ones_the_client_knows():
    declared = _definition(TOOL)["parameters"]["modalities"]["items"]["enum"]
    assert sorted(declared) == sorted(alphagenome.MODALITIES)


def test_the_description_states_the_rules_the_response_shape_enforces():
    text = _definition(TOOL)["description"]
    for phrase in (
        "NO `direction` key at all",
        "never merge, average",
        "NEVER this variant's confidence",
        "nothing measured to compare against",
        "could not be checked",
    ):
        assert phrase.lower() in text.lower(), phrase
