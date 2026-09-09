"""Entity extraction and the premise-measurement script it feeds."""

import base64
import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta

from genetics_mcp_server.memory_digest import (
    KINDS,
    extract_entities,
    extract_marker_entities,
    extract_user_entities,
)
from genetics_mcp_server.scripts import memory_premise_stats as stats


def blocks(*items):
    """A stored assistant `content_json`: the list of content blocks the browser persists."""
    return json.dumps(list(items))


def tool_use(name, **kwargs):
    return {"type": "tool_use", "id": "toolu_01", "name": name, "input": kwargs}


def test_query_database_sql_yields_views_and_column_literals():
    sql = (
        "SELECT cs.phenocode, cs.pip FROM credible_sets_v cs "
        "JOIN gene_burden_results_v g USING (gene) "
        "WHERE cs.gene IN ('APOE', 'pcsk9') AND cs.phenocode = 'E4_DM2' "
        "AND cs.resource = 'finngen' AND cs.rsid = 'RS429358' LIMIT 100"
    )
    found = extract_entities(blocks(tool_use("query_database", sql=sql, max_rows=1000)))
    assert ("view", "credible_sets_v") in found
    assert ("view", "gene_burden_results_v") in found
    assert ("gene", "APOE") in found
    assert ("gene", "PCSK9") in found
    assert ("phenotype", "E4_DM2") in found
    assert ("dataset", "finngen") in found
    assert ("variant", "rs429358") in found
    assert all(kind in KINDS for kind, _ in found)


def test_run_analysis_script_text():
    code = (
        "import genetics\n"
        "df = genetics.query(\"SELECT * FROM colocalization_v WHERE gene = 'TP53'\")\n"
        "hits = genetics.get_summary_stats(variants=['rs7412'], phenotypes=['I9_CHD'])\n"
        "lead = '19-44908684-T-C'\n"
    )
    found = extract_entities(blocks(tool_use("run_analysis", code=code, timeout_s=120)))
    assert ("view", "colocalization_v") in found
    assert ("gene", "TP53") in found
    assert ("variant", "rs7412") in found
    assert ("phenotype", "I9_CHD") in found
    assert ("variant", "19-44908684-t-c") in found


def test_phenotype_and_gene_lookup_parameters():
    found = extract_entities(
        blocks(
            tool_use("get_phenotype_report", resource="finngen", phenotype_code="E4_DM2"),
            tool_use("lookup_phenotype_names", codes=["I9_CHD", "J10_ASTHMA"]),
            tool_use("search_phenotypes", query="type 2 diabetes"),
            tool_use("search_genes", query="APOE, LDLR"),
            tool_use("normalize_gene_symbols", symbols=["pcsk9"]),
            tool_use("lookup_variants_by_rsid", rsids=["rs429358"]),
        )
    )
    assert ("phenotype", "E4_DM2") in found
    assert ("phenotype", "I9_CHD") in found
    assert ("phenotype", "J10_ASTHMA") in found
    assert ("phenotype", "type 2 diabetes") in found
    assert ("gene", "APOE") in found
    assert ("gene", "LDLR") in found
    assert ("gene", "PCSK9") in found
    assert ("variant", "rs429358") in found
    assert ("dataset", "finngen") in found


def test_free_text_search_query_is_not_an_entity():
    found = extract_entities(
        blocks(
            tool_use("search_scientific_literature", query="APOE and Alzheimer's disease"),
            tool_use("web_search", query="latest GWAS news"),
            tool_use("search_uniprot", query="serine protease inhibitors"),
            tool_use("launch_subagents", query="summarise the coloc results"),
        )
    )
    assert found == set()


def test_conditional_query_parameters():
    """cBioPortal and MGI take a symbol only under a condition their own arguments state."""
    found = extract_entities(
        blocks(
            tool_use("search_cbioportal", query="TP53", query_type="gene_summary"),
            tool_use("search_cbioportal", query="EGFR"),  # gene_summary by default
            tool_use("search_mgi", query="Pcsk9", query_type="gene_phenotypes"),
        )
    )
    assert found == {("gene", "TP53"), ("gene", "EGFR"), ("gene", "PCSK9")}

    other = extract_entities(
        blocks(
            tool_use("search_cbioportal", query="TP53 R175H", query_type="variant_hotspot"),
            tool_use("search_cbioportal", query="lung adenocarcinoma", query_type="study_search"),
            tool_use("search_mgi", query="cardiac hypertrophy", query_type="phenotype_genes"),
            tool_use("search_mgi", query="MGI:1234567", query_type="allele"),
        )
    )
    assert other == set()


def test_tool_results_are_never_read():
    results = json.dumps(
        {"toolu_01": {"results": [{"gene": "BRCA1", "phenocode": "C3_BREAST"}]}}
    )
    content = blocks(
        tool_use("search_genes", query="APOE"),
        {"type": "tool_result", "tool_use_id": "toolu_01", "content": "BRCA1 C3_BREAST"},
    )
    found = extract_entities(content, results)
    assert found == {("gene", "APOE")}
    assert extract_entities(content) == extract_entities(content, results)


def test_malformed_and_empty_inputs():
    assert extract_entities("not json at all") == set()
    assert extract_entities("") == set()
    assert extract_entities(None) == set()
    assert extract_entities(blocks(tool_use("search_genes"))) == set()
    assert extract_entities(blocks(tool_use("search_genes", query=""))) == set()
    assert extract_entities(json.dumps({"unexpected": "shape"})) == set()
    assert extract_entities(blocks({"type": "text", "text": "APOE is interesting"})) == set()


def test_user_text_yields_identifiers_only():
    assert extract_user_entities("does rs429358 matter for APOE?") == {
        ("variant", "rs429358")
    }
    assert extract_user_entities("look at chr19:44908684:T:C") == {
        ("variant", "19-44908684-t-c")
    }
    assert extract_user_entities(None) == set()


def test_legacy_tooluse_marker():
    payload = base64.b64encode(
        json.dumps({"name": "search_genes", "input": {"query": "APOE"}}).encode()
    ).decode()
    assert extract_marker_entities(f"Working on it [TOOLUSE:{payload}] done") == {
        ("gene", "APOE")
    }
    assert extract_marker_entities("[TOOLUSE:notbase64!!]") == set()
    assert extract_marker_entities(None) == set()


# --- the premise script -------------------------------------------------------------

PRODUCTION_SCHEMA = """
CREATE TABLE chat_sessions (id TEXT PRIMARY KEY, user_id TEXT, title TEXT,
    created_at TIMESTAMP, updated_at TIMESTAMP, rating INTEGER, comment TEXT,
    phenotype_code TEXT, shared INTEGER);
CREATE TABLE chat_messages (id TEXT PRIMARY KEY, session_id TEXT, role TEXT,
    content TEXT, created_at TIMESTAMP, thumbs_up INTEGER, content_json TEXT,
    literature_backend TEXT, tool_profile TEXT, tool_results_json TEXT,
    instruction_set_id TEXT, verbosity TEXT);
CREATE TABLE conversation_analysis (session_id TEXT PRIMARY KEY, summary TEXT);
"""


def build_db(path):
    """Two users, three sessions, one of them a returning session that re-mentions APOE."""
    con = sqlite3.connect(path)
    con.executescript(PRODUCTION_SCHEMA)
    now = datetime.now()
    sessions = [
        ("s1", "alice", now - timedelta(days=10)),
        ("s2", "alice", now - timedelta(days=8)),
        ("s3", "bob", now - timedelta(days=1)),
        ("s4", "anonymous", now - timedelta(days=1)),
    ]
    for sid, uid, created in sessions:
        con.execute(
            "INSERT INTO chat_sessions (id, user_id, created_at) VALUES (?,?,?)",
            (sid, uid, created.strftime("%Y-%m-%d %H:%M:%S")),
        )
    messages = [
        ("m1", "s1", "user", "what is known about the APOE locus?", None),
        ("m2", "s1", "assistant", "Looking.", blocks(tool_use("search_genes", query="APOE"))),
        ("m3", "s2", "user", "back to apoe — any coding variants?", None),
        ("m4", "s2", "assistant", "Sure.", blocks(tool_use("search_genes", query="APOE"))),
        ("m5", "s3", "user", "hello, what can you do?", None),
        ("m6", "s4", "user", "anonymous traffic", None),
    ]
    for mid, sid, role, content, content_json in messages:
        con.execute(
            "INSERT INTO chat_messages (id, session_id, role, content, content_json, created_at)"
            " VALUES (?,?,?,?,?,?)",
            (mid, sid, role, content, content_json, now.strftime("%Y-%m-%d %H:%M:%S")),
        )
    con.commit()
    con.close()


def test_collect_shape_and_counts(tmp_path):
    db = tmp_path / "chat_history.db"
    build_db(str(db))
    out = stats.collect(str(db))

    assert set(out) >= {
        "generated_at",
        "span",
        "all_time",
        "last_90d",
        "chat_turn_metrics",
        "assistant_content_json_chars",
    }
    # production runs an image without the table; the script must say so, not crash
    assert out["chat_turn_metrics"] == "table missing"

    for window in ("all_time", "last_90d"):
        section = out[window]
        assert section["sessions"] == 3  # the anonymous session is excluded
        assert section["users"] == 2
        assert section["returning_sessions"] == 1
        assert section["returning_share"] == round(1 / 3, 3)
        assert section["first_msg_rementions_prior_entity"] == 1
        assert section["remention_share_of_returning"] == 1.0
        # APOE is a gene, so the strict rate agrees here; on real data it is the one that
        # is not inflated by the dataset every session names
        assert section["first_msg_rementions_prior_entity_strict"] == 1
        assert section["remention_share_strict"] == 1.0
        assert section["remention_kinds"] == {
            "gene": 1,
            "phenotype": 0,
            "variant": 0,
            "dataset": 0,
            "view": 0,
        }
        assert section["first_turn_toolarg_overlap"] == 1
        assert section["first_turn_toolarg_overlap_strict"] == 1
        assert section["toolarg_share_strict"] == 1.0
        assert section["toolarg_kinds"]["gene"] == 1
        assert section["first_msg_refer_back_phrase"] == 0
        assert section["inter_session_gap_hours"]["n"] == 1
        assert section["inter_session_gap_hours"]["share_lt_24h"] == 0.0
        assert section["sessions_per_user"] == {
            "users": 2,
            "median": 1.5,
            "p90": 1,  # nearest-rank on two users, not an interpolated 2
            "max": 2,
            "share_gt20": 0.0,
            "share_gt50": 0.0,
        }

    dumped = json.dumps(out)
    for secret in ("alice", "bob", "s1", "s2"):
        assert f'"{secret}"' not in dumped


def test_bundled_script_is_self_contained_and_agrees(tmp_path):
    db = tmp_path / "chat_history.db"
    build_db(str(db))
    bundle = tmp_path / "premise.py"
    source = stats.build_bundle()
    assert "genetics_mcp_server" not in source
    bundle.write_text(source)

    # an empty PYTHONPATH and a cwd holding nothing of this project mean an installed copy
    # of the package cannot stand in for a leak the bundle should not have
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    env = dict(os.environ, PYTHONPATH="")
    completed = subprocess.run(
        [sys.executable, str(bundle), str(db)],
        capture_output=True,
        text=True,
        cwd=str(run_dir),
        env=env,
        check=True,
    )
    assert completed.stderr == ""
    bundled = json.loads(completed.stdout)
    direct = stats.collect(str(db))
    bundled.pop("generated_at")
    direct.pop("generated_at")
    assert bundled == direct
