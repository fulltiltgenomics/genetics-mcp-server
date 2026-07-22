"""Tests for the UniProt / EBI Proteins integration.

Split in two: pure functions (no network, no HTTP mocking at all) and the three
composite executor methods driven through a mocked httpx client.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from genetics_mcp_server.tools import uniprot
from genetics_mcp_server.tools.executor import ToolExecutor


@pytest.fixture(autouse=True)
def clear_uniprot_cache():
    """The TTL cache is a module singleton, so a hit would look like 'no HTTP call'."""
    uniprot._CACHE.clear()
    yield
    uniprot._CACHE.clear()


def _one_of(data: dict, *keys):
    """Value of the first present key, so a test asserts semantics not a key spelling."""
    for key in keys:
        if key in data:
            return data[key]
    raise AssertionError(f"none of {keys} present in {sorted(data)}")


def _strings(value) -> set[str]:
    """Every string anywhere inside a nested structure."""
    if isinstance(value, str):
        return {value}
    if isinstance(value, dict):
        return set().union(*(_strings(v) for v in value.values())) if value else set()
    if isinstance(value, (list, tuple)):
        return set().union(*(_strings(v) for v in value)) if value else set()
    return set()


# UniProtKB REST entry, trimmed to the fields flatten_entry is specified to read
ENTRY_P07202 = {
    "primaryAccession": "P07202",
    "uniProtkbId": "PERT_HUMAN",
    "entryType": "UniProtKB reviewed (Swiss-Prot)",
    "proteinDescription": {"recommendedName": {"fullName": {"value": "Thyroid peroxidase"}}},
    "genes": [{"geneName": {"value": "TPO"}, "synonyms": [{"value": "MSA"}]}],
    "organism": {"scientificName": "Homo sapiens", "taxonId": 9606},
    "comments": [
        {
            "commentType": "FUNCTION",
            "texts": [{"value": "Iodination and coupling of the hormonogenic tyrosines."}],
        },
        {
            "commentType": "SUBCELLULAR LOCATION",
            "subcellularLocations": [{"location": {"value": "Apical cell membrane"}}],
        },
        {
            "commentType": "ALTERNATIVE PRODUCTS",
            "isoforms": [
                {"isoformIds": ["P07202-1"], "name": {"value": "1"}, "isoformSequenceStatus": "Displayed"},
                {"isoformIds": ["P07202-2"], "name": {"value": "2"}},
                {"isoformIds": ["P07202-3"], "name": {"value": "3"}},
            ],
        },
    ],
    "keywords": [{"id": "KW-0106", "name": "Calcium"}, {"id": "KW-0349", "name": "Heme"}],
    "sequence": {"value": "M" * 933, "length": 933, "molWeight": 102963},
    "features": [
        {
            "type": "Signal",
            "location": {"start": {"value": 1}, "end": {"value": 18}},
            "description": "",
            "evidences": [{"evidenceCode": "ECO:0000255"}],
        },
        {
            "type": "Domain",
            "location": {"start": {"value": 100}, "end": {"value": 200}},
            "description": "Sushi",
            "evidences": [{"evidenceCode": "ECO:0000259", "source": "PROSITE", "id": "PS50923"}],
        },
        {
            "type": "Active site",
            "location": {"start": {"value": 239}, "end": {"value": 239}},
            "description": "Proton acceptor",
            "evidences": [{"evidenceCode": "ECO:0000255", "source": "PROSITE-ProRule", "id": "PRU10012"}],
        },
        {
            "type": "Sequence conflict",
            "location": {"start": {"value": 70}, "end": {"value": 70}},
            "description": "in Ref. 1; AAA61217",
            "evidences": [{"evidenceCode": "ECO:0000305"}],
        },
    ],
}

# EBI /proteins/api/coordinates/location/P07202:70 -> body["locations"], real shape:
# three transcripts, two of which share one genomic interval
LOCATIONS_P07202_70 = [
    {
        "accession": "P07202",
        "entryType": "Swiss-Prot",
        "taxid": 9606,
        "ensemblGeneId": "ENSG00000115705",
        "ensemblTranscriptId": "ENST00000961834",
        "ensemblTranslationId": "ENSP00000631893",
        "proteinStart": 70,
        "proteinEnd": 70,
        "aminoAcids": "Pro",
        "chromosome": "2",
        "geneStart": 1433466,
        "geneEnd": 1433468,
        "reverseStrand": False,
        "assemblyName": "GRCh38",
    },
    {
        "accession": "P07202",
        "entryType": "Swiss-Prot",
        "taxid": 9606,
        "ensemblGeneId": "ENSG00000115705",
        "ensemblTranscriptId": "ENST00000898236",
        "ensemblTranslationId": "ENSP00000568295",
        "proteinStart": 70,
        "proteinEnd": 70,
        "aminoAcids": "Pro",
        "chromosome": "2",
        "geneStart": 1433466,
        "geneEnd": 1433468,
        "reverseStrand": False,
        "assemblyName": "GRCh38",
    },
    {
        "accession": "P07202",
        "entryType": "Swiss-Prot",
        "taxid": 9606,
        "ensemblGeneId": "ENSG00000115705",
        "ensemblTranscriptId": "ENST00000329066",
        "ensemblTranslationId": "ENSP00000328317",
        "proteinStart": 70,
        "proteinEnd": 70,
        "aminoAcids": "Pro",
        "chromosome": "2",
        "geneStart": 1449000,
        "geneEnd": 1449002,
        "reverseStrand": False,
        "assemblyName": "GRCh38",
    },
]


class TestUniProtPureFunctions:
    """No network and no HTTP mocking: parsing, flattening, grouping, cache expiry."""

    # -- parse_protein_variant -------------------------------------------------

    def test_parse_one_letter(self):
        assert uniprot.parse_protein_variant("P70A") == (70, "P")

    def test_parse_hgvs_three_letter(self):
        assert uniprot.parse_protein_variant("p.Pro70Ala") == (70, "P")

    def test_parse_bare_three_letter(self):
        assert uniprot.parse_protein_variant("Pro70Ala") == (70, "P")

    def test_parse_bare_position(self):
        assert uniprot.parse_protein_variant("70") == (70, None)

    def test_parse_three_letter_conversion_tryptophan(self):
        # W873C is one of the four TPO substitutions from the motivating session
        assert uniprot.parse_protein_variant("Trp873Cys") == (873, "W")

    def test_parse_one_letter_large_position(self):
        assert uniprot.parse_protein_variant("W873C") == (873, "W")

    def test_parse_lowercase_input(self):
        assert uniprot.parse_protein_variant("p70a") == (70, "P")

    def test_parse_surrounding_whitespace(self):
        assert uniprot.parse_protein_variant("  G393A  ") == (393, "G")

    def test_parse_hgvs_one_letter(self):
        assert uniprot.parse_protein_variant("p.R438H") == (438, "R")

    def test_parse_all_three_letter_codes_convert(self):
        codes = {
            "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
            "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
            "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
            "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
        }
        for three, one in codes.items():
            assert uniprot.parse_protein_variant(f"{three}10Ala") == (10, one), three

    def test_parse_rejects_garbage(self):
        with pytest.raises(ValueError):
            uniprot.parse_protein_variant("not-a-variant")

    def test_parse_rejects_empty(self):
        with pytest.raises(ValueError):
            uniprot.parse_protein_variant("")

    # -- flatten_entry ---------------------------------------------------------

    def test_flatten_entry_identity(self):
        flat = uniprot.flatten_entry(ENTRY_P07202)
        assert flat["accession"] == "P07202"
        assert flat["entry_name"] == "PERT_HUMAN"
        assert flat["protein_name"] == "Thyroid peroxidase"
        assert "TPO" in flat["gene_names"]
        assert flat["reviewed"] is True

    def test_flatten_entry_function_comment(self):
        flat = uniprot.flatten_entry(ENTRY_P07202)
        assert "hormonogenic tyrosines" in json.dumps(_one_of(flat, "function"))

    def test_flatten_entry_subcellular_location(self):
        flat = uniprot.flatten_entry(ENTRY_P07202)
        location = _one_of(flat, "subcellular_location", "subcellular_locations")
        assert "Apical cell membrane" in _strings(location)

    def test_flatten_entry_keywords(self):
        flat = uniprot.flatten_entry(ENTRY_P07202)
        assert {"Calcium", "Heme"} <= _strings(_one_of(flat, "keywords"))

    def test_flatten_entry_sequence_length(self):
        flat = uniprot.flatten_entry(ENTRY_P07202)
        assert _one_of(flat, "sequence_length", "length") == 933

    def test_flatten_entry_isoform_count(self):
        flat = uniprot.flatten_entry(ENTRY_P07202)
        assert _one_of(flat, "isoform_count", "isoforms") == 3

    def test_flatten_entry_unreviewed(self):
        entry = {
            "primaryAccession": "A0A1B0GTU1",
            "uniProtkbId": "A0A1B0GTU1_HUMAN",
            "entryType": "UniProtKB unreviewed (TrEMBL)",
            "proteinDescription": {"submissionNames": [{"fullName": {"value": "Serine protease 55"}}]},
            "genes": [{"geneName": {"value": "PRSS55"}}],
            "sequence": {"value": "M" * 42, "length": 42},
        }
        flat = uniprot.flatten_entry(entry)
        assert flat["reviewed"] is False
        assert flat["protein_name"] == "Serine protease 55"

    def test_flatten_entry_does_not_mutate_input(self):
        # the TTL cache hands out response bodies by reference; mutating one poisons
        # every later cache hit for that URL
        before = json.dumps(ENTRY_P07202, sort_keys=True)
        uniprot.flatten_entry(ENTRY_P07202)
        assert json.dumps(ENTRY_P07202, sort_keys=True) == before

    # -- flatten_features ------------------------------------------------------

    def test_flatten_features_normalises_all(self):
        features = uniprot.flatten_features(ENTRY_P07202["features"])
        assert len(features) == 4
        active = [f for f in features if "active site" in f["type"].lower()][0]
        assert active["start"] == 239
        assert active["end"] == 239
        assert active["description"] == "Proton acceptor"
        assert any("ECO:0000255" in item for item in _strings(active["evidence"]))

    def test_flatten_features_ranged_feature(self):
        features = uniprot.flatten_features(ENTRY_P07202["features"])
        domain = [f for f in features if f["type"].lower() == "domain"][0]
        assert (domain["start"], domain["end"]) == (100, 200)

    def test_flatten_features_type_filter_by_name(self):
        features = uniprot.flatten_features(ENTRY_P07202["features"], feature_types=["Active site"])
        assert len(features) == 1
        assert features[0]["start"] == 239

    def test_flatten_features_type_filter_by_uniprot_key(self):
        # the tool definition advertises ACT_SITE / DOMAIN keys to the model
        features = uniprot.flatten_features(ENTRY_P07202["features"], feature_types=["ACT_SITE"])
        assert [f["start"] for f in features] == [239]

    def test_flatten_features_type_filter_case_insensitive(self):
        features = uniprot.flatten_features(ENTRY_P07202["features"], feature_types=["domain"])
        assert [f["start"] for f in features] == [100]

    def test_flatten_features_type_filter_multiple(self):
        features = uniprot.flatten_features(
            ENTRY_P07202["features"], feature_types=["Active site", "Signal"]
        )
        assert sorted(f["start"] for f in features) == [1, 239]

    def test_flatten_features_type_filter_no_match(self):
        assert uniprot.flatten_features(ENTRY_P07202["features"], feature_types=["Propeptide"]) == []

    def test_flatten_features_range_keeps_overlapping_not_only_contained(self):
        # 150-160 lies strictly inside the 100-200 domain: the domain overlaps and must
        # be kept even though it is not contained in the window
        features = uniprot.flatten_features(ENTRY_P07202["features"], residue_range="150-160")
        assert [f["start"] for f in features] == [100]

    def test_flatten_features_range_boundary_inclusive(self):
        # the signal peptide ends exactly at 18, the window starts exactly at 18
        features = uniprot.flatten_features(ENTRY_P07202["features"], residue_range="18-30")
        assert [f["start"] for f in features] == [1]

    def test_flatten_features_range_point_feature(self):
        features = uniprot.flatten_features(ENTRY_P07202["features"], residue_range="239-239")
        assert [f["type"].lower() for f in features] == ["active site"]

    def test_flatten_features_range_excludes_non_overlapping(self):
        assert uniprot.flatten_features(ENTRY_P07202["features"], residue_range="300-400") == []

    def test_flatten_features_range_spanning_everything(self):
        features = uniprot.flatten_features(ENTRY_P07202["features"], residue_range="1-1000")
        assert len(features) == 4

    def test_flatten_features_type_and_range_combined(self):
        features = uniprot.flatten_features(
            ENTRY_P07202["features"], feature_types=["Domain", "Active site"], residue_range="1-150"
        )
        assert [f["start"] for f in features] == [100]

    def test_flatten_features_empty_input(self):
        assert uniprot.flatten_features([]) == []

    def test_flatten_features_ebi_begin_end_shape(self):
        # the EBI coordinates endpoint nests features under begin/end, not start/end
        ebi = [
            {
                "type": "sequence conflict",
                "location": {"begin": {"position": 70}, "end": {"position": 70}},
                "evidence": [{"code": "ECO:0000305"}],
            }
        ]
        features = uniprot.flatten_features(ebi)
        assert features[0]["start"] == 70
        assert features[0]["end"] == 70

    def test_flatten_features_does_not_mutate_input(self):
        before = json.dumps(ENTRY_P07202["features"], sort_keys=True)
        uniprot.flatten_features(ENTRY_P07202["features"], feature_types=["Domain"], residue_range="1-500")
        assert json.dumps(ENTRY_P07202["features"], sort_keys=True) == before

    # -- collapse_transcripts --------------------------------------------------

    def test_collapse_transcripts_groups_by_interval(self):
        rows = uniprot.collapse_transcripts(LOCATIONS_P07202_70)
        assert len(rows) == 2

    def test_collapse_transcripts_lists_agreeing_transcripts(self):
        rows = uniprot.collapse_transcripts(LOCATIONS_P07202_70)
        first = [r for r in rows if _one_of(r, "geneStart", "gene_start", "start") == 1433466][0]
        assert set(first["transcripts"]) == {"ENST00000961834", "ENST00000898236"}
        second = [r for r in rows if _one_of(r, "geneStart", "gene_start", "start") == 1449000][0]
        assert list(second["transcripts"]) == ["ENST00000329066"]

    def test_collapse_transcripts_keeps_coordinates(self):
        row = uniprot.collapse_transcripts(LOCATIONS_P07202_70)[0]
        assert str(_one_of(row, "chromosome", "chrom")) == "2"
        assert _one_of(row, "geneStart", "gene_start", "start") == 1433466
        assert _one_of(row, "geneEnd", "gene_end", "end") == 1433468

    def test_collapse_transcripts_converts_amino_acid_to_one_letter(self):
        row = uniprot.collapse_transcripts(LOCATIONS_P07202_70, expected_aa="P")[0]
        assert "P" in _strings({k: v for k, v in row.items() if k != "transcripts"})

    def test_collapse_transcripts_agrees_true(self):
        rows = uniprot.collapse_transcripts(LOCATIONS_P07202_70, expected_aa="P")
        assert all(r["agrees"] is True for r in rows)

    def test_collapse_transcripts_agrees_false(self):
        rows = uniprot.collapse_transcripts(LOCATIONS_P07202_70, expected_aa="A")
        assert all(r["agrees"] is False for r in rows)

    def test_collapse_transcripts_disagreement_does_not_suppress_coordinates(self):
        row = uniprot.collapse_transcripts(LOCATIONS_P07202_70, expected_aa="A")[0]
        assert _one_of(row, "geneStart", "gene_start", "start") == 1433466
        assert _one_of(row, "geneEnd", "gene_end", "end") == 1433468
        assert str(_one_of(row, "chromosome", "chrom")) == "2"

    def test_collapse_transcripts_no_expected_aa_makes_no_claim(self):
        rows = uniprot.collapse_transcripts(LOCATIONS_P07202_70)
        assert all(r.get("agrees") is None for r in rows)

    def test_collapse_transcripts_distinct_amino_acids_split(self):
        # same interval but a different residue is a different mapping, not the same one
        rows = uniprot.collapse_transcripts(
            [
                LOCATIONS_P07202_70[0],
                {**LOCATIONS_P07202_70[1], "aminoAcids": "Ala"},
            ],
            expected_aa="P",
        )
        assert len(rows) == 2
        assert sorted(r["agrees"] for r in rows) == [False, True]

    def test_collapse_transcripts_empty(self):
        assert uniprot.collapse_transcripts([]) == []

    def test_collapse_transcripts_does_not_mutate_input(self):
        before = json.dumps(LOCATIONS_P07202_70, sort_keys=True)
        uniprot.collapse_transcripts(LOCATIONS_P07202_70, expected_aa="P")
        assert json.dumps(LOCATIONS_P07202_70, sort_keys=True) == before

    # -- _TTLCache -------------------------------------------------------------

    def test_cache_hit_before_expiry(self):
        now = [1000.0]
        cache = uniprot._TTLCache(clock=lambda: now[0])
        cache.set("url", "body", ttl=60)
        now[0] += 59
        assert cache.get("url") == "body"

    def test_cache_expires_at_deadline(self):
        now = [1000.0]
        cache = uniprot._TTLCache(clock=lambda: now[0])
        cache.set("url", "body", ttl=60)
        now[0] += 60
        assert cache.get("url") is uniprot._TTLCache._MISS

    def test_cache_expiry_evicts_entry(self):
        now = [0.0]
        cache = uniprot._TTLCache(clock=lambda: now[0])
        cache.set("url", "body", ttl=10)
        now[0] = 100
        cache.get("url")
        now[0] = 100  # the entry is gone, not merely reported expired
        assert cache.get("url") is uniprot._TTLCache._MISS

    def test_cache_miss_for_unknown_key(self):
        cache = uniprot._TTLCache(clock=lambda: 0.0)
        assert cache.get("nope") is uniprot._TTLCache._MISS

    def test_cache_stores_falsy_value(self):
        cache = uniprot._TTLCache(clock=lambda: 0.0)
        cache.set("url", {}, ttl=60)
        assert cache.get("url") == {}

    def test_cache_zero_ttl_disables_caching(self):
        cache = uniprot._TTLCache(clock=lambda: 0.0)
        cache.set("url", "body", ttl=0)
        assert cache.get("url") is uniprot._TTLCache._MISS

    def test_cache_refresh_resets_deadline(self):
        now = [0.0]
        cache = uniprot._TTLCache(clock=lambda: now[0])
        cache.set("url", "old", ttl=10)
        now[0] = 9
        cache.set("url", "new", ttl=10)
        now[0] = 15
        assert cache.get("url") == "new"

    def test_cache_fifo_eviction_at_maxsize(self):
        cache = uniprot._TTLCache(maxsize=2, clock=lambda: 0.0)
        cache.set("a", 1, ttl=60)
        cache.set("b", 2, ttl=60)
        cache.set("c", 3, ttl=60)
        assert cache.get("a") is uniprot._TTLCache._MISS
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_cache_clear(self):
        cache = uniprot._TTLCache(clock=lambda: 0.0)
        cache.set("a", 1, ttl=60)
        cache.clear()
        assert cache.get("a") is uniprot._TTLCache._MISS


def _resp(body=None, status=200, headers=None, url="https://rest.uniprot.org/", text=""):
    resp = MagicMock()
    resp.status_code = status
    resp.headers = dict(headers or {})
    resp.json.return_value = body
    resp.text = text
    resp.url = url
    resp.raise_for_status = MagicMock()
    return resp


def _search_body(*entries):
    return {"results": list(entries)}


def _summary(accession, entry_name, protein_name, gene, reviewed=True, taxon=9606):
    return {
        "primaryAccession": accession,
        "uniProtkbId": entry_name,
        "entryType": (
            "UniProtKB reviewed (Swiss-Prot)" if reviewed else "UniProtKB unreviewed (TrEMBL)"
        ),
        "proteinDescription": {"recommendedName": {"fullName": {"value": protein_name}}},
        "genes": [{"geneName": {"value": gene}}],
        "organism": {"scientificName": "Homo sapiens", "taxonId": taxon},
    }


def _assert_failed(result: dict, status: int | None = None):
    """A failure is reported either as the executor's success/error envelope or as the
    client's `_error` sentinel — never as data."""
    assert result.get("success") is False or result.get("_error"), result
    if result.get("_error") and status is not None:
        assert result.get("_status") == status, result
    assert not result.get("entry")
    assert not result.get("results")


def _resolution(result: dict) -> dict:
    """The block naming the protein that was actually annotated."""
    for key in ("resolution", "resolved", "protein"):
        block = result.get(key)
        if isinstance(block, dict) and "accession" in block:
            return block
    if "accession" in result:
        return result
    raise AssertionError(f"no resolution block in result keys {sorted(result)}")


@pytest.mark.asyncio
class TestUniProtToolMethods:
    """Mocked-HTTP tests for get_protein_annotations / map_protein_variants / search_uniprot."""

    @pytest.fixture(autouse=True)
    async def setup_executor(self):
        self.executor = ToolExecutor()
        yield
        await self.executor.close()

    def _patch_get(self, resolver):
        """Patch the shared client's GET with a URL router; returns the patch and call log."""
        calls: list[str] = []

        def handler(url, *args, **kwargs):
            text = str(url)
            calls.append(text)
            resp = resolver(text)
            assert resp is not None, f"unexpected request: {text}"
            return resp

        patcher = patch.object(
            self.executor.external_client, "get", new_callable=AsyncMock, side_effect=handler
        )
        return patcher, calls

    async def test_tpo_resolves_to_p07202_not_thrombopoietin(self):
        # free text 'TPO' ranks P40225 (entry name literally TPO_HUMAN, thrombopoietin)
        # second; only the gene_exact tier pins the real thyroid peroxidase
        def resolver(url):
            if "gene_exact" in url:
                return _resp(_search_body(_summary("P07202", "PERT_HUMAN", "Thyroid peroxidase", "TPO")))
            if "/uniprotkb/search" in url:
                return _resp(
                    _search_body(
                        _summary("P07202", "PERT_HUMAN", "Thyroid peroxidase", "TPO"),
                        _summary("P40225", "TPO_HUMAN", "Thrombopoietin", "THPO"),
                    )
                )
            if "P07202" in url:
                return _resp(ENTRY_P07202)
            return None

        patcher, calls = self._patch_get(resolver)
        with patcher:
            result = await self.executor.get_protein_annotations(query="TPO")

        assert result.get("success") is not False
        resolution = _resolution(result)
        assert resolution["accession"] == "P07202"
        assert resolution["entry_name"] == "PERT_HUMAN"
        assert "P40225" not in json.dumps(result)
        assert any("gene_exact" in c for c in calls)

    async def test_prss55_reviewed_filter_beats_trembl(self):
        # gene:PRSS55 answers with one curated entry and three TrEMBL fragments; the
        # motivating session asserted Q7Z5A4 and Q9H0E5 before reaching Q6UWB4
        trembl = [
            _summary("A0A1B0GTU1", "A0A1B0GTU1_HUMAN", "Serine protease 55", "PRSS55", reviewed=False),
            _summary("H7C1P1", "H7C1P1_HUMAN", "Serine protease 55", "PRSS55", reviewed=False),
            _summary("C9JH25", "C9JH25_HUMAN", "Serine protease 55", "PRSS55", reviewed=False),
        ]
        reviewed = _summary("Q6UWB4", "PRS55_HUMAN", "Serine protease 55", "PRSS55")

        def resolver(url):
            if "gene_exact" in url:
                return _resp(_search_body())
            if "/uniprotkb/search" in url:
                # relevance order alone puts the unreviewed fragments first
                return _resp(_search_body(*trembl, reviewed))
            if "Q6UWB4" in url:
                return _resp({**ENTRY_P07202, "primaryAccession": "Q6UWB4", "uniProtkbId": "PRS55_HUMAN"})
            return None

        patcher, _calls = self._patch_get(resolver)
        with patcher:
            result = await self.executor.get_protein_annotations(query="PRSS55")

        resolution = _resolution(result)
        assert resolution["accession"] == "Q6UWB4"
        assert resolution["entry_name"] == "PRS55_HUMAN"

    async def test_map_protein_variants_p70a_agrees(self):
        def resolver(url):
            if "ebi.ac.uk" in url:
                return _resp({"locations": LOCATIONS_P07202_70})
            if "P07202" in url:
                return _resp(ENTRY_P07202)
            return None

        patcher, _calls = self._patch_get(resolver)
        with patcher:
            result = await self.executor.map_protein_variants(variants=["P70A"], query="P07202")

        assert result.get("success") is not False
        assert _resolution(result)["accession"] == "P07202"
        payload = json.dumps(result)
        assert '"agrees": true' in payload
        assert "1433466" in payload

    async def test_map_protein_variants_reports_mismatch_with_coordinates(self):
        def resolver(url):
            if "ebi.ac.uk" in url:
                return _resp({"locations": LOCATIONS_P07202_70})
            if "P07202" in url:
                return _resp(ENTRY_P07202)
            return None

        patcher, _calls = self._patch_get(resolver)
        with patcher:
            # residue 70 is Pro, not Ala: the mismatch must be reported, not suppressed
            result = await self.executor.map_protein_variants(variants=["A70P"], query="P07202")

        payload = json.dumps(result)
        assert '"agrees": false' in payload
        assert "1433466" in payload

    async def test_wrong_accession_surfaces_the_protein_it_actually_names(self):
        # Q92626 supplied where TPO was meant: the result must say it is PXDN_HUMAN
        pxdn = _summary("Q92626", "PXDN_HUMAN", "Peroxidasin homolog", "PXDN")

        def resolver(url):
            if "Q92626" in url:
                return _resp({**pxdn, "sequence": {"value": "M" * 1479, "length": 1479}})
            if "/uniprotkb/search" in url:
                return _resp(_search_body())
            return None

        patcher, _calls = self._patch_get(resolver)
        with patcher:
            result = await self.executor.get_protein_annotations(query="Q92626")

        resolution = _resolution(result)
        assert resolution["accession"] == "Q92626"
        assert resolution["entry_name"] == "PXDN_HUMAN"
        assert "PXDN" in _strings(resolution["gene_names"])
        assert "TPO" not in _strings(resolution.get("gene_names") or [])

    async def test_batch_chunks_at_100_accessions(self):
        accessions = [f"P1{i:03d}0" for i in range(150)]

        def resolver(url):
            if "accessions" in url:
                return _resp({"results": [_summary("P10000", "X_HUMAN", "X", "X")]})
            return None

        patcher, calls = self._patch_get(resolver)
        with patcher:
            result = await self.executor.uniprot.fetch_batch(accessions)

        batch_calls = [c for c in calls if "accessions" in c]
        assert len(batch_calls) == 2
        # commas survive as ',' or '%2C' depending on URL encoding
        separators = batch_calls[0].count(",") + batch_calls[0].count("%2C")
        assert separators == 99
        assert batch_calls[1].count(",") + batch_calls[1].count("%2C") == 49
        assert result["requested"] == accessions

    async def test_batch_single_chunk_at_exactly_100(self):
        accessions = [f"P1{i:03d}0" for i in range(100)]

        def resolver(url):
            if "accessions" in url:
                return _resp({"results": []})
            return None

        patcher, calls = self._patch_get(resolver)
        with patcher:
            await self.executor.uniprot.fetch_batch(accessions)

        assert len([c for c in calls if "accessions" in c]) == 1

    async def test_non_200_error_shape(self):
        def resolver(url):
            return _resp(None, status=500, text="upstream exploded")

        patcher, _calls = self._patch_get(resolver)
        with patcher:
            result = await self.executor.get_protein_annotations(query="TPO")

        _assert_failed(result, status=500)

    async def test_search_uniprot_returns_rows(self):
        def resolver(url):
            if "/uniprotkb/search" in url:
                return _resp(
                    _search_body(_summary("P07202", "PERT_HUMAN", "Thyroid peroxidase", "TPO")),
                    headers={"x-total-results": "1"},
                )
            return None

        patcher, _calls = self._patch_get(resolver)
        with patcher:
            result = await self.executor.search_uniprot(query="thyroid peroxidase")

        assert result.get("success") is not False
        assert "P07202" in json.dumps(result)

    async def test_search_uniprot_count_only_uses_total_header(self):
        def resolver(url):
            if "/uniprotkb/search" in url:
                return _resp(_search_body(), headers={"x-total-results": "215"})
            return None

        patcher, _calls = self._patch_get(resolver)
        with patcher:
            result = await self.executor.search_uniprot(keyword="KW-0865", count_only=True)

        assert _one_of(result, "count", "total", "total_results") == 215

    async def test_search_uniprot_non_200_error_shape(self):
        def resolver(url):
            return _resp(None, status=400, text="invalid query")

        patcher, _calls = self._patch_get(resolver)
        with patcher:
            result = await self.executor.search_uniprot(query="family:(")

        _assert_failed(result, status=400)
