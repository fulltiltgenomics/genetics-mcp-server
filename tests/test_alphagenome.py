"""Tests for the AlphaGenome Atlas client.

No ToolExecutor and no network, the way tests/test_uniprot.py's pure-function half works:
the SDK boundary is a fake AtlasClient injected through `atlas_factory`, and the responses
are real AnnData built the way alphagenome.atlas assembles them -- one per scorer, obs
rows carrying the variant they came from -- so the flattening and the batch split are
exercised against the shape the API actually answers in. Nothing here needs an API key.
"""

import anndata
import numpy as np
import pandas as pd
import pytest
from alphagenome.data import genome

from genetics_mcp_server.tools import alphagenome
from genetics_mcp_server.tools.alphagenome import AlphaGenomeClient

API_KEY = "test-key-not-a-real-credential"


@pytest.fixture(autouse=True)
def clear_module_state():
    """The two caches and the limiter are module singletons, shared across tests."""
    alphagenome._METADATA_CACHE.clear()
    alphagenome._LIMITER.clear()
    alphagenome._CACHE.clear()
    yield
    alphagenome._METADATA_CACHE.clear()
    alphagenome._LIMITER.clear()
    alphagenome._CACHE.clear()


def entry(values, quantiles=None, gene=None):
    """One obs row of a scorer's response: a track vector, its quantiles, its gene."""
    return {"values": list(values), "quantiles": quantiles, "gene": gene}


def _ann(pairs, biosamples=None):
    """{variant: [entry, ...]} rows as the AnnData alphagenome.atlas would return."""
    matrix = np.array([e["values"] for _, e in pairs], dtype=float)
    n_tracks = matrix.shape[1] if matrix.size or matrix.ndim == 2 else 0
    biosamples = biosamples or [f"biosample_{i}" for i in range(n_tracks)]
    var = pd.DataFrame(
        {"name": [f"track_{i}" for i in range(n_tracks)], "biosample_name": biosamples},
        index=[str(i) for i in range(n_tracks)],
    )
    obs_rows = []
    for variant, e in pairs:
        row = {"variant": variant}
        if e["gene"]:
            row["gene_id"], row["gene_name"] = e["gene"]
        obs_rows.append(row)
    obs = pd.DataFrame(obs_rows, index=[str(i) for i in range(len(pairs))])
    layers = {}
    if all(e["quantiles"] is not None for _, e in pairs) and pairs:
        layers["quantiles"] = np.array([e["quantiles"] for _, e in pairs], dtype=float)
    return anndata.AnnData(X=matrix, obs=obs, var=var, layers=layers)


def scores(values, quantiles=None, biosamples=None, genes=None):
    """A standalone AnnData for the flattening tests, with no variant column needed."""
    matrix = np.array(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    quantile_rows = quantiles if quantiles is not None else [None] * matrix.shape[0]
    genes = genes or [None] * matrix.shape[0]
    pairs = [
        (None, entry(row, q, g))
        for row, q, g in zip(matrix.tolist(), quantile_rows, genes)
    ]
    ann = _ann(pairs, biosamples=biosamples)
    if genes == [None] * matrix.shape[0]:
        ann.obs = ann.obs.drop(columns=["variant"])
    return ann


class FakeAtlas:
    """Stand-in for the gRPC AtlasClient: no transport, no key, no network."""

    def __init__(self, responses=None, batch_error=None, variant_errors=None, metadata=None):
        self.responses = responses or {}
        self.batch_error = batch_error
        self.variant_errors = variant_errors or {}
        self.metadata = metadata
        self.batch_calls = 0
        self.single_calls = []
        self.metadata_calls = 0
        self.progress_bars = []

    def _entries(self, variant):
        key = f"{variant.chromosome}:{variant.position}"
        error = self.variant_errors.get(key)
        if error is not None:
            raise error
        return self.responses.get(key, {})

    def query_variant(self, variant, *, requested_scorers, **kwargs):
        self.single_calls.append(variant)
        return {
            scorer: _ann([(variant, e) for e in entries])
            for scorer, entries in self._entries(variant).items()
            if scorer in requested_scorers
        }

    def query_variants(self, variants, *, requested_scorers, progress_bar=True, **kwargs):
        self.batch_calls += 1
        self.progress_bars.append(progress_bar)
        if self.batch_error is not None:
            raise self.batch_error
        by_scorer = {}
        for variant in variants:
            for scorer, entries in self._entries(variant).items():
                if scorer in requested_scorers:
                    by_scorer.setdefault(scorer, []).extend((variant, e) for e in entries)
        return {scorer: _ann(pairs) for scorer, pairs in by_scorer.items()}

    def scorer_metadata(self):
        self.metadata_calls += 1
        return self.metadata or {}


class FakeScorerMetadata:
    """The SDK's ScorerMetadata: a name, a signedness flag and a track DataFrame."""

    def __init__(self, name, biosamples):
        self.name = name
        self.is_signed = False
        self.track_metadata = pd.DataFrame({"biosample_name": biosamples})


async def _no_sleep(_delay):
    return None


def _client(atlas_obj, **kwargs):
    return AlphaGenomeClient(
        api_key=API_KEY, atlas_factory=lambda key: atlas_obj, sleep=_no_sleep, **kwargs
    )


class FakeGrpcError(Exception):
    """Shaped like grpc._InactiveRpcError: a code() returning a named status."""

    class _Code:
        def __init__(self, name):
            self.name = name

    def __init__(self, name, message=""):
        super().__init__(message or name)
        self._code = self._Code(name)

    def code(self):
        return self._code


# --------------------------------------------------------------------------- #
# chromosome normalisation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("23", "chrX"),
        (23, "chrX"),
        ("chr23", "chrX"),
        ("X", "chrX"),
        ("chrX", "chrX"),
        ("x", "chrX"),
        ("24", "chrY"),
        ("Y", "chrY"),
        ("25", "chrM"),
        ("26", "chrM"),
        ("MT", "chrM"),
        ("M", "chrM"),
        ("1", "chr1"),
        ("chr1", "chr1"),
        ("22", "chr22"),
        (" chr7 ", "chr7"),
    ],
)
def test_normalise_chromosome_accepts_every_spelling_the_suite_emits(raw, expected):
    assert alphagenome.normalise_chromosome(raw) == expected


@pytest.mark.parametrize("raw", ["0", "27", "99", "", "   ", None, "chrZ", "chr", "1.5"])
def test_normalise_chromosome_rejects_what_alphagenome_would(raw):
    assert alphagenome.normalise_chromosome(raw) is None


def test_parse_variant_normalises_x_and_keeps_the_caller_id():
    variant = alphagenome.parse_variant("23:154000000:A:G")
    assert variant["chromosome"] == "chrX"
    assert variant["position"] == 154000000
    assert variant["reference_bases"] == "A"
    assert variant["alternate_bases"] == "G"
    assert variant["id"] == "23:154000000:A:G"


@pytest.mark.parametrize(
    "variant_id", ["99:1:A:G", "chrZ:1:A:G", "not-a-variant", "1:A:G", "", "1:1:X:G"]
)
def test_parse_variant_returns_a_sentinel_rather_than_raising(variant_id):
    result = alphagenome.parse_variant(variant_id)
    assert alphagenome._is_error(result)
    assert result["_stage"] == "input"


def test_a_normalised_variant_is_what_the_sdk_takes():
    variant = alphagenome._sdk_variant(alphagenome.parse_variant("23:100:A:G"))
    assert isinstance(variant, genome.Variant)
    assert variant.chromosome == "chrX"
    assert variant.name == "23:100:A:G"


# --------------------------------------------------------------------------- #
# the per-modality quantity rule
# --------------------------------------------------------------------------- #


def test_signed_modality_keeps_the_direction():
    flat = alphagenome.flatten_scores(scores([[-0.42, 0.10]]), "DNASE")
    assert flat["quantity"] == "signed"
    assert flat["value"] == pytest.approx(-0.42)


def test_magnitude_modality_never_surfaces_a_direction():
    flat = alphagenome.flatten_scores(scores([[-0.42, 0.10]]), "SPLICE_SITES")
    assert flat["quantity"] == "magnitude"
    assert flat["value"] == pytest.approx(0.42)
    assert all(track["value"] >= 0 for track in flat["top_tracks"])


def test_every_shipped_modality_has_an_explicit_quantity():
    assert set(alphagenome.MODALITIES) == {
        "DNASE",
        "ATAC",
        "CHIP_HISTONE",
        "SPLICE_SITES",
        "SPLICE_JUNCTIONS",
        "RNA_SEQ",
        "SPLICE_SITE_USAGE",
        "CHIP_TF",
        "CAGE",
        "PROCAP",
        "POLYADENYLATION",
        "CONTACT_MAPS",
    }
    for name, modality in alphagenome.MODALITIES.items():
        assert modality.quantity in ("signed", "magnitude"), name


def test_accessibility_is_signed_and_splicing_is_magnitude():
    for name in ("DNASE", "ATAC", "CHIP_HISTONE", "RNA_SEQ"):
        assert alphagenome.MODALITIES[name].quantity == "signed"
    for name in ("SPLICE_SITES", "SPLICE_JUNCTIONS", "SPLICE_SITE_USAGE"):
        assert alphagenome.MODALITIES[name].quantity == "magnitude"


# --------------------------------------------------------------------------- #
# validation metadata travels as data
# --------------------------------------------------------------------------- #


def test_calibrated_modality_carries_tier_substrate_and_rho():
    meta = alphagenome.validation_metadata("DNASE")
    assert meta["tier"] == 1
    assert meta["status"] == "calibrated"
    assert meta["calibrated_against"] == "caQTL beta"
    assert meta["population_rho"] == pytest.approx(0.478)
    assert meta["rho_scope"] == "population"


def test_unvalidated_modality_says_so_and_carries_no_rho():
    meta = alphagenome.validation_metadata("CHIP_TF")
    assert meta["tier"] == 4
    assert meta["status"] == "unvalidated"
    assert meta["population_rho"] is None
    assert meta["rho_scope"] is None


def test_population_rho_is_only_defined_for_calibrated_tiers():
    for name, modality in alphagenome.MODALITIES.items():
        rho = alphagenome.population_rho_for_modality(name)
        assert (rho is None) == (modality.tier == 4), name
    assert alphagenome.population_rho_for_modality("NOT_A_MODALITY") is None


def test_flattened_response_carries_the_validation_block():
    flat = alphagenome.flatten_scores(scores([[0.3]]), "RNA_SEQ")
    assert flat["validation"]["tier"] == 3
    assert flat["validation"]["population_rho"] == pytest.approx(0.112)
    assert flat["validation"]["calibrated_against"] == "eQTL beta"


# --------------------------------------------------------------------------- #
# AnnData flattening
# --------------------------------------------------------------------------- #


def test_flattening_returns_plain_json_able_python():
    flat = alphagenome.flatten_scores(
        scores([[0.4999, -0.1]], quantiles=[[0.987, 0.4]]), "DNASE"
    )
    assert isinstance(flat["value"], float)
    for track in flat["top_tracks"]:
        assert set(track) == {"track", "biosample", "value", "quantile"}
        assert isinstance(track["track"], str)
        assert not isinstance(track["value"], np.generic)


def test_the_quantile_is_returned_alongside_the_raw_score():
    flat = alphagenome.flatten_scores(
        scores([[0.4999, -0.1]], quantiles=[[0.987, 0.4]]), "DNASE"
    )
    assert flat["value"] == pytest.approx(0.4999)
    assert flat["quantile"] == pytest.approx(0.987)


def test_missing_quantile_layer_is_none_not_an_error():
    flat = alphagenome.flatten_scores(scores([[0.2]]), "DNASE")
    assert flat["quantile"] is None


def test_empty_rna_seq_matrix_is_guarded():
    empty = anndata.AnnData(
        X=np.zeros((0, 371), dtype=float),
        obs=pd.DataFrame(index=pd.Index([], dtype=str)),
        var=pd.DataFrame(index=[str(i) for i in range(371)]),
    )
    flat = alphagenome.flatten_scores(empty, "RNA_SEQ")
    assert flat["value"] is None
    assert flat["genes"] == []
    assert "note" in flat


def test_rna_seq_is_reported_per_gene():
    ann = scores(
        [[0.1, 0.2], [-0.9, -0.4]],
        quantiles=[[0.5, 0.6], [0.01, 0.2]],
        genes=[("ENSG1", "AAA"), ("ENSG2", "BBB")],
    )
    flat = alphagenome.flatten_scores(ann, "RNA_SEQ")
    assert [g["gene_name"] for g in flat["genes"]] == ["BBB", "AAA"]
    assert flat["genes"][0]["value"] == pytest.approx(-0.9)
    assert flat["genes"][0]["quantile"] == pytest.approx(0.01)
    assert flat["value"] == pytest.approx(-0.9)


def test_negative_avi_survives_flattening():
    flat = alphagenome.flatten_scores(scores([[-0.77]]), "ATAC")
    assert flat["value"] == pytest.approx(-0.77)


@pytest.mark.parametrize(
    "modality",
    ["RNA_SEQ", "SPLICE_SITES", "SPLICE_SITE_USAGE", "SPLICE_JUNCTIONS", "POLYADENYLATION"],
)
def test_every_gene_based_scorer_reports_every_gene(modality):
    """The SDK emits one obs row per gene for these; reading row 0 would drop the rest."""
    ann = scores(
        [[0.1, 0.2], [-0.9, -0.4], [0.5, 0.3]],
        genes=[("ENSG1", "AAA"), ("ENSG2", "BBB"), ("ENSG3", "CCC")],
    )
    flat = alphagenome.flatten_scores(ann, modality)
    assert sorted(g["gene_name"] for g in flat["genes"]) == ["AAA", "BBB", "CCC"]
    assert sorted(g["gene_id"] for g in flat["genes"]) == ["ENSG1", "ENSG2", "ENSG3"]
    assert flat["genes"][0]["gene_name"] == "BBB"
    signed = alphagenome.MODALITIES[modality].quantity == "signed"
    assert flat["value"] == pytest.approx(-0.9 if signed else 0.9)


def test_a_magnitude_modality_hides_the_direction_in_the_quantile_too():
    """The quantile ranks the SIGNED score, so a negative one restores what value hides."""
    flat = alphagenome.flatten_scores(
        scores([[-0.42, 0.10]], quantiles=[[-0.83, 0.20]], genes=[("ENSG1", "AAA")]),
        "SPLICE_SITES",
    )
    assert flat["value"] == pytest.approx(0.42)
    assert flat["quantile"] == pytest.approx(0.83)
    assert flat["genes"][0]["quantile"] == pytest.approx(0.83)
    assert all(track["quantile"] >= 0 for track in flat["top_tracks"])


def test_a_signed_modality_keeps_a_negative_quantile():
    flat = alphagenome.flatten_scores(
        scores([[-0.42, 0.10]], quantiles=[[-0.83, 0.20]]), "DNASE"
    )
    assert flat["quantile"] == pytest.approx(-0.83)


# --------------------------------------------------------------------------- #
# cell-type matched track selection
# --------------------------------------------------------------------------- #


async def test_cell_type_resolves_to_tracks_and_narrows_the_answer():
    metadata = {"DNASE": FakeScorerMetadata("DNASE", ["liver", "K562"])}
    client = _client(FakeAtlas(metadata=metadata))
    selection = await client.resolve_tracks("DNASE", "liver")
    assert selection.indices == (0,)
    assert selection.matched is True

    flat = alphagenome.flatten_scores(
        scores([[0.1, 0.9]], biosamples=["liver", "K562"]), "DNASE", selection
    )
    assert flat["value"] == pytest.approx(0.1)
    assert flat["n_tracks_scored"] == 1
    assert flat["cell_type_match"]["resolved_biosamples"] == ["liver"]
    assert flat["cell_type_match"]["matched"] is True


async def test_unmatched_cell_type_falls_back_to_all_tracks_and_says_so():
    metadata = {"DNASE": FakeScorerMetadata("DNASE", ["liver"])}
    client = _client(FakeAtlas(metadata=metadata))
    selection = await client.resolve_tracks("DNASE", "pancreatic islet")
    assert selection.matched is False

    flat = alphagenome.flatten_scores(scores([[0.1, 0.9]]), "DNASE", selection)
    assert flat["n_tracks_scored"] == 2
    assert flat["cell_type_match"]["matched"] is False


async def test_track_metadata_is_fetched_once_per_process():
    fake = FakeAtlas(metadata={"DNASE": FakeScorerMetadata("DNASE", ["liver"])})
    client = _client(fake)
    await client.resolve_tracks("DNASE", "liver")
    await client.resolve_tracks("DNASE", "liver")
    await _client(fake).resolve_tracks("DNASE", "liver")
    assert fake.metadata_calls == 1


# --------------------------------------------------------------------------- #
# batches are all-or-nothing
# --------------------------------------------------------------------------- #


async def test_a_batch_failure_isolates_the_offender_instead_of_losing_everything():
    good = {"DNASE": [entry([0.5])]}
    fake = FakeAtlas(
        responses={"chr1:100": good, "chr2:200": good},
        batch_error=ValueError("Reference bases do not match, expected 'C'"),
        variant_errors={"chr3:300": ValueError("Reference bases do not match, expected 'C'")},
    )
    result = await _client(fake).score_variants(
        ["1:100:A:G", "2:200:A:G", "3:300:A:G"], modalities=["DNASE"]
    )
    assert result["success"] is True
    assert result["n_scored"] == 2
    assert fake.batch_calls == 1
    assert len(fake.single_calls) == 3
    by_id = {r.get("variant", {}).get("id") or r.get("variant_id"): r for r in result["results"]}
    assert by_id["1:100:A:G"]["success"] is True
    assert by_id["3:300:A:G"]["success"] is False
    assert by_id["3:300:A:G"]["stage"] == "reference_mismatch"


async def test_a_healthy_batch_is_one_call_split_back_apart_by_variant():
    fake = FakeAtlas(
        responses={
            "chr1:100": {"DNASE": [entry([0.5, 0.1])]},
            "chr2:200": {"DNASE": [entry([-0.8, 0.0])]},
        }
    )
    result = await _client(fake).score_variants(["1:100:A:G", "2:200:A:G"], modalities=["DNASE"])
    assert fake.batch_calls == 1
    assert fake.single_calls == []
    assert fake.progress_bars == [False]
    values = {r["variant"]["id"]: r["modalities"]["DNASE"]["value"] for r in result["results"]}
    assert values == {"1:100:A:G": pytest.approx(0.5), "2:200:A:G": pytest.approx(-0.8)}


async def test_two_ids_for_one_locus_are_both_scored():
    """Keyed by locus, the split would keep one id and report the other as all-null."""
    fake = FakeAtlas(responses={"chr1:100": {"DNASE": [entry([0.5])]}})
    result = await _client(fake).score_variants(
        ["1:100:A:G", "chr1:100:A:G"], modalities=["DNASE"]
    )
    assert len(fake.single_calls) == 2
    values = {r["variant"]["id"]: r["modalities"]["DNASE"]["value"] for r in result["results"]}
    assert values == {
        "1:100:A:G": pytest.approx(0.5),
        "chr1:100:A:G": pytest.approx(0.5),
    }


async def test_an_unparseable_variant_never_reaches_the_api():
    fake = FakeAtlas(responses={"chr1:100": {"DNASE": [entry([0.5])]}})
    result = await _client(fake).score_variants(["1:100:A:G", "99:1:A:G"], modalities=["DNASE"])
    # one valid variant survives validation, so the batch path is not taken at all
    assert fake.batch_calls == 0
    assert len(fake.single_calls) == 1
    assert result["n_scored"] == 1
    failed = [r for r in result["results"] if not r["success"]][0]
    assert failed["stage"] == "input"


async def test_too_many_variants_is_refused_before_any_request():
    fake = FakeAtlas()
    ids = [f"1:{i}:A:G" for i in range(alphagenome._MAX_VARIANTS + 1)]
    result = await _client(fake).score_variants(ids)
    assert result["success"] is False
    assert result["stage"] == "input"
    assert fake.batch_calls == 0


# --------------------------------------------------------------------------- #
# the error taxonomy maps onto success: False
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "exc,stage",
    [
        (ValueError("Chromosome chr23 not found."), "chromosome"),
        (ValueError("Reference bases do not match, expected 'C'"), "reference_mismatch"),
        (FakeGrpcError("RESOURCE_EXHAUSTED", "quota exceeded"), "quota"),
        (FakeGrpcError("DEADLINE_EXCEEDED", "deadline"), "timeout"),
        (TimeoutError("timed out"), "timeout"),
        (PermissionError("bad key"), "auth"),
        (RuntimeError("something else"), "prediction"),
    ],
)
async def test_every_error_arm_becomes_a_success_false_result(exc, stage):
    fake = FakeAtlas(variant_errors={"chr1:100": exc})
    result = await _client(fake).score_variant("1:100:A:G", modalities=["DNASE"])
    assert result["success"] is False
    assert result["stage"] == stage
    assert isinstance(result["error"], str)


async def test_a_missing_api_key_is_a_result_not_an_exception(monkeypatch):
    monkeypatch.delenv("ALPHAGENOME_API_KEY", raising=False)
    client = AlphaGenomeClient(atlas_factory=lambda key: FakeAtlas())
    result = await client.score_variant("1:100:A:G", modalities=["DNASE"])
    assert result["success"] is False
    assert result["stage"] == "config"


async def test_an_unknown_modality_is_refused_before_any_request():
    fake = FakeAtlas()
    result = await _client(fake).score_variants(["1:100:A:G"], modalities=["MASS_SPEC"])
    assert result["success"] is False
    assert result["stage"] == "input"
    assert fake.batch_calls == 0 and fake.single_calls == []


async def test_the_api_key_never_reaches_an_error_message():
    fake = FakeAtlas(variant_errors={"chr1:100": RuntimeError(f"auth failed for {API_KEY}")})
    result = await _client(fake).score_variant("1:100:A:G", modalities=["DNASE"])
    assert API_KEY not in result["error"]
    assert "***" in result["error"]


async def test_a_quota_error_is_retried_before_it_is_reported():
    calls = {"n": 0}

    class Flaky(FakeAtlas):
        def query_variant(self, variant, *, requested_scorers, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise FakeGrpcError("RESOURCE_EXHAUSTED")
            return {"DNASE": _ann([(variant, entry([0.3]))])}

    result = await _client(Flaky()).score_variant("1:100:A:G", modalities=["DNASE"])
    assert calls["n"] == 2
    assert result["success"] is True


async def test_quota_backoff_gives_up_and_reports_rather_than_looping():
    fake = FakeAtlas(variant_errors={"chr1:100": FakeGrpcError("RESOURCE_EXHAUSTED")})
    result = await _client(fake).score_variant("1:100:A:G", modalities=["DNASE"])
    assert result["success"] is False
    assert result["stage"] == "quota"
    assert len(fake.single_calls) == alphagenome._QUOTA_RETRIES + 1


# --------------------------------------------------------------------------- #
# the per-minute limiter
# --------------------------------------------------------------------------- #


async def test_the_limiter_admits_a_burst_then_waits_out_the_window():
    now = {"t": 0.0}
    waits = []

    async def sleep(delay):
        waits.append(delay)
        now["t"] += delay

    limiter = alphagenome._RateLimiter(limit=3, window=60.0, clock=lambda: now["t"], sleep=sleep)
    for _ in range(3):
        await limiter.acquire()
    assert waits == []
    await limiter.acquire()
    assert waits == [pytest.approx(60.0)]


async def test_a_batch_costs_one_slot_per_variant():
    limiter = alphagenome._RateLimiter(limit=100, window=60.0, sleep=_no_sleep)
    await limiter.acquire(cost=7)
    assert len(limiter._hits) == 7


async def test_a_batch_is_charged_for_its_metadata_call_too():
    fake = FakeAtlas(
        responses={
            "chr1:100": {"DNASE": [entry([0.5])]},
            "chr2:200": {"DNASE": [entry([0.1])]},
        }
    )
    await _client(fake).score_variants(["1:100:A:G", "2:200:A:G"], modalities=["DNASE"])
    assert len(alphagenome._LIMITER._hits) == 3


def test_the_limiter_default_is_the_measured_quota():
    assert alphagenome._RATE_LIMIT_PER_MINUTE == 1320
    assert alphagenome._RATE_WINDOW == 60.0


# --------------------------------------------------------------------------- #
# the whole result
# --------------------------------------------------------------------------- #


def _contains_anndata(value):
    if isinstance(value, anndata.AnnData):
        return True
    if isinstance(value, dict):
        return any(_contains_anndata(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_anndata(v) for v in value)
    return False


async def test_score_variant_returns_no_anndata_anywhere():
    fake = FakeAtlas(
        responses={
            "chrX:100": {
                "DNASE": [entry([0.4999], [0.987])],
                "SPLICE_SITES": [entry([-0.3])],
            }
        }
    )
    result = await _client(fake).score_variant("23:100:A:G", modalities=["DNASE", "SPLICE_SITES"])
    assert result["success"] is True
    assert result["variant"]["chromosome"] == "chrX"
    assert result["modalities"]["DNASE"]["value"] == pytest.approx(0.4999)
    assert result["modalities"]["DNASE"]["quantile"] == pytest.approx(0.987)
    assert result["modalities"]["SPLICE_SITES"]["value"] == pytest.approx(0.3)
    assert not _contains_anndata(result)


async def test_a_modality_with_no_scores_is_reported_rather_than_dropped():
    fake = FakeAtlas(responses={"chr1:100": {"DNASE": [entry([0.2])]}})
    result = await _client(fake).score_variant("1:100:A:G", modalities=["DNASE", "ATAC"])
    assert result["modalities"]["ATAC"]["value"] is None
    assert result["modalities"]["ATAC"]["validation"]["tier"] == 1


async def test_default_modalities_exclude_the_unvalidated_tier():
    fake = FakeAtlas(responses={"chr1:100": {}})
    result = await _client(fake).score_variant("1:100:A:G")
    assert set(result["modalities"]) == set(alphagenome.DEFAULT_MODALITIES)
    assert "CHIP_TF" not in result["modalities"]
    assert all(alphagenome.MODALITIES[n].tier <= 3 for n in result["modalities"])


# --------------------------------------------------------------------------- #
# the tool surface: definitions and the ServerToolExecutor delegate
# --------------------------------------------------------------------------- #

TOOL = "get_alphagenome_variant_predictions"


def _definition():
    from genetics_mcp_server.tools.definitions import all_local_tool_definitions

    [tool] = [t for t in all_local_tool_definitions() if t["name"] == TOOL]
    return tool


def _executor(atlas_obj, **kwargs):
    from genetics_mcp_server.tools.orchestration import ServerToolExecutor

    executor = ServerToolExecutor()
    executor.__dict__["alphagenome"] = _client(atlas_obj, **kwargs)
    return executor


def test_the_declared_modalities_are_exactly_the_ones_the_client_knows():
    """Spelled out in the schema, pinned here — the same arrangement as
    defaults._SUMMARIZE_PARAM_TOOLS, so definitions.py does not have to import a module
    that pulls the AlphaGenome SDK into every process that reads the tool catalogue."""
    declared = _definition()["parameters"]["modalities"]["items"]["enum"]
    assert sorted(declared) == sorted(alphagenome.MODALITIES)


def test_the_tool_is_on_both_surfaces_and_no_script_can_reach_it():
    """`sdk_replaceable` False: an outside resource, like search_uniprot. It is therefore
    advertised on the code-execution surface as well, where a SCRIPT cannot call it — the
    sandbox egress allow-list names db-api and results-api only. Intended for this phase."""
    from genetics_mcp_server.tools.definitions import resolve_tools

    assert _definition()["sdk_replaceable"] is False
    for code_execution in (True, False):
        assert TOOL in {t["name"] for t in resolve_tools(code_execution)}


# Every file the sandbox image ships, as a path under the genetics_mcp_server package
# root. Transcribed from SDK_ALLOWLIST in genetics-results-suite's sandbox/prune_venv.py,
# which is the build-time authority and lives in another repo, so nothing here can import
# it. What makes this list wrong: a module added to that allow-list and not to this one
# then ships unchecked while this test keeps passing. The existence assertion below catches
# the other direction, a file renamed or dropped here.
_SHIPPED_MODULES = (
    "__init__.py",
    "sdk/__init__.py",
    "sdk/_runner.py",
    "sdk/client.py",
    "sdk/errors.py",
    "sdk/plots.py",
    "tools/__init__.py",
    "tools/executor.py",
    "tools/sql_safety.py",
    "tools/chembl.py",
    "tools/uniprot.py",
)


def _alphagenome_imports(source: str) -> list[str]:
    """Every name `source` imports that mentions alphagenome, at any nesting depth.

    An ImportFrom contributes its module AND its imported names: `from . import
    alphagenome` and `from genetics_mcp_server.tools import alphagenome` put the module
    nowhere but `node.names`, and those are exactly the deferred intra-package forms
    prune_venv.py's own comment says defeat the build gate.
    """
    import ast

    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""] + [a.name for a in node.names]
        else:
            continue
        found += [n for n in names if "alphagenome" in n]
    return found


def test_the_sandbox_never_ships_a_path_to_the_client():
    """The hard boundary: the image ships the SDK's import closure, tools/orchestration.py
    is not in it, and `alphagenome` is not installed there. An import at any nesting depth
    in any shipped file would raise ModuleNotFoundError in a container with no shell."""
    import pathlib

    pkg = pathlib.Path(alphagenome.__file__).parent.parent
    for rel in _SHIPPED_MODULES:
        shipped = pkg / rel
        assert shipped.is_file(), f"{rel} is allow-listed for the image but not in the tree"
        found = _alphagenome_imports(shipped.read_text())
        assert not found, f"{rel} imports {found}"


@pytest.mark.parametrize(
    "source",
    [
        "import genetics_mcp_server.tools.alphagenome",
        "from genetics_mcp_server.tools.alphagenome import AlphaGenomeClient",
        "from . import alphagenome",
        "from genetics_mcp_server.tools import alphagenome",
        "def later():\n    from .alphagenome import AlphaGenomeClient\n",
    ],
)
def test_the_shipped_import_check_catches_the_relative_and_package_forms(source):
    """The forms the previous check missed: both name the module only in `node.names`."""
    assert _alphagenome_imports(source), f"not caught: {source!r}"


async def test_the_delegate_labels_the_result_as_a_prediction():
    fake = FakeAtlas(responses={"chr1:100": {"DNASE": [entry([0.5], quantiles=[0.9])]}})
    result = await _executor(fake).get_alphagenome_variant_predictions(
        ["1:100:A:G"], modalities=["DNASE"]
    )
    assert result["data_kind"] == "model_prediction"
    assert result["measured"] is False
    assert result["n_scored"] == 1


async def test_the_delegate_passes_the_validation_block_through_structurally():
    """Tier, quantity, substrate and the population rho reach the model as DATA. The rho
    is the modality's, and `rho_scope` says so, so nothing downstream has to infer that it
    is not a per-variant confidence."""
    fake = FakeAtlas(responses={"chr1:100": {"DNASE": [entry([0.5])]}})
    result = await _executor(fake).get_alphagenome_variant_predictions(["1:100:A:G"])
    validation = result["results"][0]["modalities"]["DNASE"]["validation"]
    assert validation == alphagenome.validation_metadata("DNASE")
    assert validation["rho_scope"] == "population"
    assert validation["population_rho"] == pytest.approx(alphagenome.MODALITIES["DNASE"].population_rho)


async def test_the_delegate_takes_a_comma_separated_string_too():
    fake = FakeAtlas(responses={"chr1:100": {"DNASE": [entry([0.5])]}})
    result = await _executor(fake).get_alphagenome_variant_predictions(
        "1:100:A:G, 99:1:A:G", modalities=["DNASE"]
    )
    assert result["n_requested"] == 2
    assert result["n_scored"] == 1


async def test_a_client_side_failure_stays_a_result_and_keeps_the_label():
    fake = FakeAtlas()
    result = await _executor(fake).get_alphagenome_variant_predictions([])
    assert result["success"] is False
    assert result["data_kind"] == "model_prediction"


# --------------------------------------------------------------------------- #
# the prediction cache
# --------------------------------------------------------------------------- #


async def test_a_repeated_variant_is_answered_without_a_second_request():
    fake = FakeAtlas(responses={"chr1:100": {"DNASE": [entry([0.5])]}})
    client = _client(fake)
    first = await client.score_variants(["1:100:A:G"], modalities=["DNASE"])
    second = await client.score_variants(["1:100:A:G"], modalities=["DNASE"])
    assert second["results"] == first["results"]
    assert len(fake.single_calls) == 1 and fake.batch_calls == 0


async def test_a_partly_cached_batch_only_asks_for_what_is_missing():
    fake = FakeAtlas(
        responses={
            "chr1:100": {"DNASE": [entry([0.5])]},
            "chr2:200": {"DNASE": [entry([0.7])]},
        }
    )
    client = _client(fake)
    await client.score_variants(["1:100:A:G"], modalities=["DNASE"])
    fake.single_calls.clear()
    result = await client.score_variants(["1:100:A:G", "2:200:A:G"], modalities=["DNASE"])
    assert result["n_scored"] == 2
    # two variants would have been a batch call; only the uncached one was asked for
    assert fake.batch_calls == 0 and len(fake.single_calls) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cell_type": "liver"},
        {"modalities": ["DNASE", "RNA_SEQ"]},
    ],
    ids=["cell_type", "modalities"],
)
async def test_the_key_separates_requests_that_would_answer_differently(kwargs):
    """The cell type especially: a key without it would return liver's prediction to a
    question about K562, labelled as K562."""
    metadata = {
        "DNASE": FakeScorerMetadata("DNASE", ["liver"]),
        "RNA_SEQ": FakeScorerMetadata("RNA_SEQ", ["liver"]),
    }
    fake = FakeAtlas(
        responses={"chr1:100": {"DNASE": [entry([0.5])], "RNA_SEQ": [entry([0.2])]}},
        metadata=metadata,
    )
    client = _client(fake)
    await client.score_variants(["1:100:A:G"], modalities=["DNASE"])
    fake.single_calls.clear()
    await client.score_variants(["1:100:A:G"], **{"modalities": ["DNASE"], **kwargs})
    assert len(fake.single_calls) == 1


def test_the_cache_key_carries_the_variant_the_cell_type_and_the_modalities():
    variant = alphagenome.parse_variant("1:100:A:G")
    base = alphagenome._cache_key(variant, "liver", ["DNASE"])
    assert base != alphagenome._cache_key(variant, "K562", ["DNASE"])
    assert base != alphagenome._cache_key(variant, None, ["DNASE"])
    assert base != alphagenome._cache_key(variant, "liver", ["DNASE", "RNA_SEQ"])
    assert base != alphagenome._cache_key(
        alphagenome.parse_variant("1:100:A:T"), "liver", ["DNASE"]
    )
    # order of the requested modalities is not a difference in the answer
    assert alphagenome._cache_key(variant, "liver", ["RNA_SEQ", "DNASE"]) == (
        alphagenome._cache_key(variant, "liver", ["DNASE", "RNA_SEQ"])
    )


async def test_a_failed_prediction_is_never_cached():
    fake = FakeAtlas(variant_errors={"chr1:100": RuntimeError("transient")})
    client = _client(fake)
    first = await client.score_variants(["1:100:A:G"], modalities=["DNASE"])
    assert first["results"][0]["success"] is False
    fake.variant_errors.clear()
    fake.responses = {"chr1:100": {"DNASE": [entry([0.5])]}}
    second = await client.score_variants(["1:100:A:G"], modalities=["DNASE"])
    assert second["results"][0]["success"] is True


async def test_a_zero_ttl_turns_the_cache_off():
    class _Settings:
        alphagenome_cache_ttl = 0

    fake = FakeAtlas(responses={"chr1:100": {"DNASE": [entry([0.5])]}})
    client = AlphaGenomeClient(
        _Settings(), api_key=API_KEY, atlas_factory=lambda key: fake, sleep=_no_sleep
    )
    await client.score_variants(["1:100:A:G"], modalities=["DNASE"])
    await client.score_variants(["1:100:A:G"], modalities=["DNASE"])
    assert len(fake.single_calls) == 2


async def test_an_expired_entry_is_fetched_again():
    now = {"t": 0.0}
    cache = alphagenome._TTLCache(clock=lambda: now["t"])
    fake = FakeAtlas(responses={"chr1:100": {"DNASE": [entry([0.5])]}})
    client = _client(fake, cache=cache)
    await client.score_variants(["1:100:A:G"], modalities=["DNASE"])
    now["t"] = alphagenome._CACHE_TTL + 1
    await client.score_variants(["1:100:A:G"], modalities=["DNASE"])
    assert len(fake.single_calls) == 2


async def test_an_all_hit_batch_makes_no_metadata_round_trip():
    metadata = {"DNASE": FakeScorerMetadata("DNASE", ["liver"])}
    fake = FakeAtlas(responses={"chr1:100": {"DNASE": [entry([0.5])]}}, metadata=metadata)
    client = _client(fake)
    await client.score_variants(["1:100:A:G"], cell_type="liver", modalities=["DNASE"])
    alphagenome._METADATA_CACHE.clear()
    fake.metadata_calls = 0
    await client.score_variants(["1:100:A:G"], cell_type="liver", modalities=["DNASE"])
    assert fake.metadata_calls == 0


async def test_a_metadata_failure_is_not_frozen_into_the_cache():
    """A degraded answer must not be served for the whole TTL after the Atlas recovers."""

    class FlakyMetadataAtlas(FakeAtlas):
        def scorer_metadata(self):
            self.metadata_calls += 1
            if self.metadata_calls == 1:
                raise FakeGrpcError("UNAVAILABLE", "atlas metadata is down")
            return self.metadata

    fake = FlakyMetadataAtlas(
        responses={"chr1:100": {"DNASE": [entry([0.5, 0.9])]}},
        metadata={"DNASE": FakeScorerMetadata("DNASE", ["biosample_1"])},
    )
    client = _client(fake)

    first = await client.score_variants(
        ["1:100:A:G"], cell_type="biosample_1", modalities=["DNASE"]
    )
    degraded = first["results"][0]["modalities"]["DNASE"]["cell_type_match"]
    assert first["results"][0]["success"] is True
    assert degraded["resolution_failed"] is True
    assert degraded["matched"] is False

    second = await client.score_variants(
        ["1:100:A:G"], cell_type="biosample_1", modalities=["DNASE"]
    )
    recovered = second["results"][0]["modalities"]["DNASE"]["cell_type_match"]
    assert recovered["resolution_failed"] is False
    assert recovered["matched"] is True
    assert recovered["resolved_biosamples"] == ["biosample_1"]
    assert len(fake.single_calls) == 2
