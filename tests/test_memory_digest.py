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


# --- render_digest -------------------------------------------------------------------

from genetics_mcp_server.memory_digest import (
    MAX_DIGEST_CHARS,
    MAX_PINNED_SESSIONS,
    _assemble,
    render_digest,
)


def text_block(t):
    return {"type": "text", "text": t}


def tool_result(tool_use_id, content):
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}


def session(
    id,
    title,
    updated_at,
    first_user_message,
    assistant_content_json=(),
    phenotype_code=None,
    pinned_at=None,
    created_at=None,
):
    return {
        "id": id,
        "title": title,
        "created_at": created_at or updated_at,
        "updated_at": updated_at,
        "phenotype_code": phenotype_code,
        "pinned_at": pinned_at,
        "first_user_message": first_user_message,
        "assistant_content_json": list(assistant_content_json),
    }


def test_render_digest_empty_input_is_empty_string():
    assert render_digest([], datetime(2026, 9, 9)) == ""


def test_render_digest_golden_output_two_recent_one_pinned():
    sessions = [
        session(
            "s2",
            "APOE lipid follow-up",
            "2026-09-05 12:30:00",
            "any   coding variants   in APOE we should check?\nthanks",
            [blocks(tool_use("search_genes", query="APOE"))],
            phenotype_code="E4_DM2",
        ),
        session(
            "s1",
            "first look",
            "2026-09-01 09:05:00",
            "what is known about the APOE locus?",
            [blocks(tool_use("search_genes", query="APOE"))],
        ),
        session(
            "s0",
            "PCSK9 reference",
            "2026-08-01 09:05:00",
            "give me the full background on PCSK9 and cardiovascular risk",
            [
                blocks(tool_use("search_genes", query="PCSK9")),
                blocks(text_block("PCSK9 inhibitors lower LDL.")),
            ],
            phenotype_code="E4_DM2",
            pinned_at="2026-08-01 09:10:00",
        ),
    ]

    expected = (
        "Earlier conversations (newest first):\n"
        "2026-09-05 | APOE lipid follow-up | E4_DM2 | gene:APOE | "
        "any coding variants in APOE we should check? thanks\n"
        "2026-09-01 | first look | gene:APOE | what is known about the APOE locus?\n"
        "Pinned:\n"
        "2026-08-01 | [pinned] | PCSK9 reference | E4_DM2 | gene:PCSK9\n"
        "  Q: give me the full background on PCSK9 and cardiovascular risk\n"
        "  A: PCSK9 inhibitors lower LDL."
    )
    assert render_digest(sessions, datetime(2026, 9, 9)) == expected


def test_render_digest_is_deterministic_regardless_of_entity_insertion_order():
    # tool_use inputs surface in a different order but the underlying entity set is the
    # same, and set iteration order must never leak into the rendered line
    s_a = session(
        "s1",
        "t",
        "2026-09-01 09:00:00",
        "hello",
        [blocks(tool_use("search_genes", query="APOE"), tool_use("lookup_variants_by_rsid", rsids=["rs429358"]))],
    )
    s_b = session(
        "s1",
        "t",
        "2026-09-01 09:00:00",
        "hello",
        [blocks(tool_use("lookup_variants_by_rsid", rsids=["rs429358"]), tool_use("search_genes", query="APOE"))],
    )
    out_a = render_digest([s_a], datetime(2026, 9, 9))
    out_b = render_digest([s_b], datetime(2026, 9, 9))
    assert out_a == out_b
    assert render_digest([s_a], datetime(2026, 9, 9)) == out_a


def test_render_digest_per_line_entity_cap():
    tool_uses = [tool_use("search_genes", query=f"GENE{i}") for i in range(15)]
    s = session("s1", "t", "2026-09-01 09:00:00", "hi", [blocks(*tool_uses)])
    out = render_digest([s], datetime(2026, 9, 9))
    line = out.splitlines()[1]
    assert line.count("gene:GENE") == 12
    assert "+3 more" in line


def test_render_digest_drops_oldest_unpinned_first_then_oldest_pinned():
    # newest-first, matching the accessor's own ordering guarantee: index 0 is the most
    # recent session, and each list is large enough that the 6000-char cap must bite
    recent = [
        session(f"u{i}", f"title {i}", f"2026-09-{40 - i:02d} 09:00:00", "x" * 120)
        for i in range(40)
    ]
    old_pinned = [
        session(
            f"p{i}",
            f"pinned {i}",
            f"2026-01-{5 - i:02d} 09:00:00",
            "y" * 380,
            pinned_at=f"2026-01-{5 - i:02d} 09:05:00",
        )
        for i in range(5)
    ]
    out = render_digest(recent + old_pinned, datetime(2026, 9, 9))
    assert len(out) <= MAX_DIGEST_CHARS
    # the newest unpinned entry and every pinned entry survive the cap
    assert "title 0" in out
    for i in range(5):
        assert f"pinned {i}" in out
    # the oldest unpinned entries are the ones dropped to make room
    assert "title 39" not in out


def test_render_digest_pinned_cap_keeps_newest():
    # 12 distinct months so every session has a unique, unambiguous sort key
    pinned_sorted = [
        session(
            f"p{i}",
            f"pinned {i}",
            f"2026-{12 - i:02d}-01 09:00:00",
            "hi",
            pinned_at=f"2026-{12 - i:02d}-01 09:05:00",
        )
        for i in range(12)
    ]
    out = render_digest(pinned_sorted, datetime(2026, 9, 9))
    rendered_titles = [line for line in out.splitlines() if "[pinned]" in line]
    assert len(rendered_titles) == MAX_PINNED_SESSIONS
    assert "pinned 0" in out
    assert "pinned 11" not in out


def test_render_digest_oversized_single_pinned_entry_still_fits_cap():
    s = session(
        "p1",
        "huge",
        "2026-01-01 09:00:00",
        "q" * 8000,
        [blocks(text_block("a" * 8000))],
        pinned_at="2026-01-01 09:05:00",
    )
    out = render_digest([s], datetime(2026, 9, 9))
    assert len(out) <= MAX_DIGEST_CHARS


def test_render_digest_never_surfaces_tool_result_text():
    s = session(
        "s1",
        "t",
        "2026-01-01 09:00:00",
        "hi",
        [blocks(tool_use("search_genes", query="APOE"), tool_result("toolu_01", "SECRET_ROW_DATA"))],
        pinned_at="2026-01-01 09:05:00",
    )
    out = render_digest([s], datetime(2026, 9, 9))
    assert "SECRET_ROW_DATA" not in out


def test_render_digest_collapses_newlines_in_first_message_head():
    s = session("s1", "t", "2026-01-01 09:00:00", "line one\nline two\r\nline three")
    out = render_digest([s], datetime(2026, 9, 9))
    assert "\n" not in out.splitlines()[1]
    assert "line one line two line three" in out


def test_render_digest_marks_artifact_looking_values_as_expires():
    s = session("s1", "t", "2026-01-01 09:00:00", "see report_final.csv or https://x.example/a.pdf?x=1")
    out = render_digest([s], datetime(2026, 9, 9))
    assert "report_final.csv" not in out
    assert "https://x.example" not in out
    assert "(expires)" in out


def test_render_digest_scrubs_artifact_link_out_of_title_unpinned():
    s = session("s1", "see https://x.example/r.csv", "2026-01-01 09:00:00", "hi")
    out = render_digest([s], datetime(2026, 9, 9))
    assert "https://x.example" not in out
    assert "(expires)" in out


def test_render_digest_scrubs_artifact_link_out_of_title_pinned():
    s = session(
        "s1",
        "see https://x.example/r.csv",
        "2026-01-01 09:00:00",
        "hi",
        pinned_at="2026-01-01 09:05:00",
    )
    out = render_digest([s], datetime(2026, 9, 9))
    assert "https://x.example" not in out
    assert "(expires)" in out


def test_render_digest_oversized_title_does_not_evict_other_sessions():
    huge = session("s1", "x" * 9000, "2026-09-02 09:00:00", "hi")
    normal = session("s2", "normal title", "2026-09-01 09:00:00", "hello there")
    out = render_digest([huge, normal], datetime(2026, 9, 9))
    assert len(out) <= MAX_DIGEST_CHARS
    assert "normal title" in out


def test_assemble_returns_empty_string_when_nothing_survives():
    assert _assemble([], []) == ""


def test_assemble_omits_unpinned_header_when_only_pinned_survive():
    out = _assemble([], ["2026-01-01 | [pinned] | t"])
    assert out == "Pinned:\n2026-01-01 | [pinned] | t"
    assert "Earlier conversations" not in out


def test_render_digest_collapses_newlines_in_phenotype_code():
    s = session("s1", "t", "2026-01-01 09:00:00", "hi", phenotype_code="E4\nPinned:\nDM2")
    out = render_digest([s], datetime(2026, 9, 9))
    lines = out.splitlines()
    assert len(lines) == 2
    assert "E4 Pinned: DM2" in lines[1]
