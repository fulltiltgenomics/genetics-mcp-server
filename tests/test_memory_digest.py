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
    cluster_sessions,
    extract_entities,
    extract_marker_entities,
    extract_user_entities,
)
from genetics_mcp_server.memory_gate import user_log_hash
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


# --- session clustering -------------------------------------------------------------


def test_single_linkage_joins_a_chain_and_leaves_strangers_alone():
    clusters = cluster_sessions(
        {
            "a": {("gene", "APOE")},
            "b": {("gene", "APOE"), ("gene", "LDLR")},
            "c": {("gene", "LDLR")},
            "d": {("phenotype", "I9_CHD")},
            "e": set(),
        }
    )
    assert clusters["a"] == clusters["b"] == clusters["c"]
    assert len({clusters["a"], clusters["d"], clusters["e"]}) == 3
    # numbered from zero in order of each cluster's smallest key
    assert clusters == {"a": 0, "b": 0, "c": 0, "d": 1, "e": 2}


def test_min_shared_two_needs_a_second_identifier():
    entity_sets = {
        "a": {("gene", "APOE"), ("gene", "LDLR")},
        "b": {("gene", "APOE"), ("gene", "LDLR")},
        "c": {("gene", "APOE")},
    }
    assert cluster_sessions(entity_sets, min_shared=1) == {"a": 0, "b": 0, "c": 0}
    loose = cluster_sessions(entity_sets, min_shared=2)
    assert loose["a"] == loose["b"] != loose["c"]


def test_only_strict_kinds_link_sessions():
    """Nearly every session names finngen and a view; linking on those merges everything."""
    clusters = cluster_sessions(
        {
            "a": {("dataset", "finngen"), ("view", "credible_sets_v")},
            "b": {("dataset", "finngen"), ("view", "credible_sets_v")},
        }
    )
    assert clusters["a"] != clusters["b"]


def test_cluster_ids_do_not_depend_on_iteration_order():
    entity_sets = {
        "s3": {("gene", "APOE")},
        "s1": {("gene", "TP53")},
        "s2": {("gene", "APOE")},
    }
    reversed_order = {key: entity_sets[key] for key in reversed(list(entity_sets))}
    assert cluster_sessions(entity_sets) == cluster_sessions(reversed_order)


def test_min_shared_below_one_is_rejected():
    try:
        cluster_sessions({}, min_shared=0)
    except ValueError:
        return
    raise AssertionError("min_shared=0 should be rejected")


def test_premise_pseudonym_matches_the_memory_log():
    """The bundle cannot import memory_gate, so its copy of the hash must be checked."""
    assert stats.user_hash(" Alice@Example.COM ") == user_log_hash("alice@example.com")


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

# chat_sessions with pinned_at, for the pin-aware M2/M3 tests below. Kept separate from
# PRODUCTION_SCHEMA because production predates the column — that gap is exactly what
# collect() has to tolerate, and PRODUCTION_SCHEMA is what stands in for it.
PRODUCTION_SCHEMA_WITH_PINNING = PRODUCTION_SCHEMA.replace(
    "phenotype_code TEXT, shared INTEGER);",
    "phenotype_code TEXT, shared INTEGER, pinned_at TIMESTAMP);",
)


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


def test_collect_succeeds_without_pinned_at(tmp_path):
    """Production's chat_sessions predates pinned_at; collect() must not raise for it."""
    db = tmp_path / "chat_history.db"
    build_db(str(db))
    out = stats.collect(str(db))
    assert out["clusters"]["pinned_visible"] is False


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


def build_cluster_db(path):
    """One user with two lines of work, opening the last session with a re-mention.

    Sessions 1-3 are a gene project (APOE-LDLR, joined as a chain), 4-6 a phenotype
    project, and session 7 comes back to APOE immediately after the phenotype work — so
    the session the recency digest would lead with belongs to the other project.
    """
    con = sqlite3.connect(path)
    con.executescript(PRODUCTION_SCHEMA)
    now = datetime.now()
    tools = [
        tool_use("search_genes", query="APOE"),
        tool_use("search_genes", query="APOE, LDLR"),
        tool_use("search_genes", query="LDLR"),
        tool_use("get_phenotype_report", phenotype_code="I9_CHD"),
        tool_use("lookup_phenotype_names", codes=["I9_CHD"]),
        tool_use("get_phenotype_report", phenotype_code="I9_CHD"),
        tool_use("search_genes", query="APOE"),
    ]
    firsts = ["start here"] * 6 + ["back to apoe — any coding variants?"]
    for index, (call, first) in enumerate(zip(tools, firsts)):
        session_id = f"c{index + 1}"
        stamp = (now - timedelta(days=20 - index)).strftime("%Y-%m-%d %H:%M:%S")
        con.execute(
            "INSERT INTO chat_sessions (id, user_id, created_at, updated_at)"
            " VALUES (?,?,?,?)",
            (session_id, "carol", stamp, stamp),
        )
        con.execute(
            "INSERT INTO chat_messages (id, session_id, role, content, created_at)"
            " VALUES (?,?,?,?,?)",
            (f"{session_id}u", session_id, "user", first, stamp),
        )
        con.execute(
            "INSERT INTO chat_messages (id, session_id, role, content, content_json,"
            " created_at) VALUES (?,?,?,?,?,?)",
            (f"{session_id}a", session_id, "assistant", "ok", blocks(call), stamp),
        )
    con.commit()
    con.close()


def test_cluster_metrics_and_quality_dump(tmp_path):
    db = tmp_path / "chat_history.db"
    build_cluster_db(str(db))
    out = stats.collect(str(db))
    loose = out["clusters"]["all_time"]["min_shared_1"]
    tight = out["clusters"]["all_time"]["min_shared_2"]

    # two clusters of >=3 sessions: the gene chain plus session 7, and the phenotype work
    assert loose["m1_users_with_min_sessions"] == 1
    assert loose["m1_multi_thread_share"] == 1.0
    assert loose["m2_remention_sessions"] == 1
    assert loose["m2_interleaving_share"] == 1.0
    # six prior sessions all sit inside the last-20 window, so nothing is a window miss
    assert loose["m3_window_miss_share"] == 0.0

    # at two shared entities nothing links, so APOE's earlier sessions land in different
    # clusters and the conservative rule counts the re-mention as no evidence
    assert tight["m1_multi_thread_share"] == 0.0
    assert tight["m2_interleaving_share"] == 0.0
    assert tight["remention_sessions_with_ambiguous_entity"] == 1

    dump = out["clusters"]["quality_all_time"]["min_shared_1"]["users"]
    assert [entry["sessions"] for entry in dump] == [7]
    assert dump[0]["user"] == user_log_hash("carol")
    biggest = dump[0]["clusters"][0]
    assert biggest["size"] == 4
    assert biggest["top_entities"][0] == {"entity": "gene:APOE", "sessions": 3}
    # singleton clusters carry a size and nothing else to say
    assert all("top_entities" not in c for c in dump[0]["clusters"] if c["size"] < 2)

    dumped = json.dumps(out)
    for secret in ("carol", "c1", "c7", "back to apoe — any coding variants?"):
        assert f'"{secret}"' not in dumped


def test_window_miss_counts_a_source_outside_the_recency_window(tmp_path, monkeypatch):
    """M3 on the same data with the digest window shrunk: APOE's session falls out of it."""
    db = tmp_path / "chat_history.db"
    build_cluster_db(str(db))
    monkeypatch.setattr(stats, "DIGEST_WINDOW_SESSIONS", 2)
    loose = stats.collect(str(db))["clusters"]["all_time"]["min_shared_1"]
    # the two most recent earlier sessions are both phenotype work; the APOE session the
    # digest would have needed is older than that, and in the returning session's cluster
    assert loose["m3_window_miss_share"] == 1.0


# --- M1/M2/M3 mutation coverage ------------------------------------------------------


def build_multi_user_db(path, users):
    """Sessions for several users, each given full control over timing and pin state.

    `users` maps user_id to a list of (id, created_at, updated_at, tool_use_or_None,
    first_user_message, pinned_at) tuples, inserted in the given order.
    """
    con = sqlite3.connect(path)
    con.executescript(PRODUCTION_SCHEMA_WITH_PINNING)
    for user_id, sessions in users.items():
        for sid, created, updated, call, first, pinned in sessions:
            con.execute(
                "INSERT INTO chat_sessions (id, user_id, created_at, updated_at, pinned_at)"
                " VALUES (?,?,?,?,?)",
                (sid, user_id, created, updated, pinned),
            )
            con.execute(
                "INSERT INTO chat_messages (id, session_id, role, content, created_at)"
                " VALUES (?,?,?,?,?)",
                (f"{sid}u", sid, "user", first, created),
            )
            if call:
                con.execute(
                    "INSERT INTO chat_messages (id, session_id, role, content, content_json,"
                    " created_at) VALUES (?,?,?,?,?,?)",
                    (f"{sid}a", sid, "assistant", "ok", blocks(call), created),
                )
    con.commit()
    con.close()


def build_sessions_db(path, user_id, sessions):
    """One user's sessions; see `build_multi_user_db`."""
    build_multi_user_db(path, {user_id: sessions})


def _day(n):
    return (datetime.now() + timedelta(days=n)).strftime("%Y-%m-%d %H:%M:%S")


def test_m3_ignores_a_source_in_a_different_cluster(tmp_path, monkeypatch):
    """(a) an out-of-window source outside the current session's cluster is not a miss.

    Kills dropping the `source_cluster == clusters[session_id]` condition: without it,
    any out-of-window source would count regardless of which project it belongs to.
    """
    db = tmp_path / "chat_history.db"
    monkeypatch.setattr(stats, "DIGEST_WINDOW_SESSIONS", 2)
    build_sessions_db(
        db,
        "u",
        [
            ("s1", _day(-10), _day(-10), tool_use("search_genes", query="APOE"), "start", None),
            ("s2", _day(-8), _day(-8), tool_use("search_genes", query="LDLR"), "start", None),
            ("s3", _day(-6), _day(-6), tool_use("search_genes", query="LDLR"), "start", None),
            (
                "s4",
                _day(0),
                _day(0),
                tool_use("search_genes", query="LDLR"),
                "back to apoe — any coding variants?",
                None,
            ),
        ],
    )
    loose = stats.collect(str(db))["clusters"]["all_time"]["min_shared_1"]
    assert loose["m2_remention_sessions"] == 1
    # s1 (APOE) is out of window, but s4 joined the LDLR cluster (s2/s3), not APOE's
    assert loose["m3_source_outside_window_same_cluster"] == 0
    assert loose["m3_window_miss_share"] == 0.0


def test_m3_window_boundary_is_exclusive(tmp_path, monkeypatch):
    """(b) a source at position DIGEST_WINDOW_SESSIONS is in window; one further back misses."""
    monkeypatch.setattr(stats, "DIGEST_WINDOW_SESSIONS", 3)

    def build(n_fillers):
        rows = [("src", _day(-100), _day(-100), tool_use("search_genes", query="APOE"), "s", None)]
        for i in range(n_fillers):
            rows.append(
                (
                    f"f{i}",
                    _day(-50 + i),
                    _day(-50 + i),
                    tool_use("search_genes", query=f"FILLER{i}"),
                    "s",
                    None,
                )
            )
        rows.append(
            ("cur", _day(0), _day(0), tool_use("search_genes", query="APOE"), "back to apoe", None)
        )
        return rows

    in_window_db = tmp_path / "in_window.db"
    build_sessions_db(in_window_db, "u", build(2))  # src at position 3 == window
    in_window = stats.collect(str(in_window_db))["clusters"]["all_time"]["min_shared_1"]
    assert in_window["m3_source_outside_window_same_cluster"] == 0

    miss_db = tmp_path / "miss.db"
    build_sessions_db(miss_db, "u2", build(3))  # src at position 4 == window + 1
    miss = stats.collect(str(miss_db))["clusters"]["all_time"]["min_shared_1"]
    assert miss["m3_source_outside_window_same_cluster"] == 1


def test_m1_needs_two_clusters_of_the_full_minimum_size(tmp_path):
    """(c) two clusters one short of M1_MIN_CLUSTER_SIZE must not satisfy M1.

    Kills M1_MIN_CLUSTER_SIZE 3->2: two pairs would satisfy a threshold of 2 but not 3.
    """
    db = tmp_path / "chat_history.db"
    build_sessions_db(
        db,
        "u",
        [
            ("s1", _day(-10), _day(-10), tool_use("search_genes", query="APOE"), "s", None),
            ("s2", _day(-9), _day(-9), tool_use("search_genes", query="APOE"), "s", None),
            ("s3", _day(-8), _day(-8), tool_use("search_genes", query="LDLR"), "s", None),
            ("s4", _day(-7), _day(-7), tool_use("search_genes", query="LDLR"), "s", None),
            ("s5", _day(-6), _day(-6), tool_use("search_genes", query="MTHFR"), "s", None),
        ],
    )
    loose = stats.collect(str(db))["clusters"]["all_time"]["min_shared_1"]
    assert loose["m1_users_with_min_sessions"] == 1
    assert loose["m1_multi_cluster_users"] == 0
    assert loose["m1_multi_thread_share"] == 0.0


def test_m2_preceding_session_is_by_recency_not_creation_order(tmp_path):
    """(d) a session re-touched after creation outranks its creation-order successor.

    Kills reading "preceding" off creation order: s2 was created after s1 but never
    touched again, while s1 was updated just before the re-mention — so the digest
    would have led with s1 (APOE's own cluster), not s2.
    """
    db = tmp_path / "chat_history.db"
    build_sessions_db(
        db,
        "u",
        [
            ("s1", _day(-10), _day(-1), tool_use("search_genes", query="APOE"), "s", None),
            ("s2", _day(-8), _day(-8), tool_use("search_genes", query="MTHFR"), "s", None),
            (
                "s3",
                _day(0),
                _day(0),
                tool_use("search_genes", query="LDLR"),
                "back to apoe — any coding variants?",
                None,
            ),
        ],
    )
    loose = stats.collect(str(db))["clusters"]["all_time"]["min_shared_1"]
    assert loose["m2_remention_sessions"] == 1
    # recency-order preceding is s1 (APOE's own cluster) == the entity's source cluster
    assert loose["m2_preceding_in_other_cluster"] == 0
    assert loose["m2_interleaving_share"] == 0.0


def test_m3_uses_the_newest_source_not_the_oldest(tmp_path, monkeypatch):
    """(e) two sources for one entity; only the newer one is inside the window.

    Kills picking the oldest occurrence: the digest surfaces the most recent one, so
    that is the one whose window membership decides the miss.
    """
    db = tmp_path / "chat_history.db"
    monkeypatch.setattr(stats, "DIGEST_WINDOW_SESSIONS", 1)
    build_sessions_db(
        db,
        "u",
        [
            ("s_old", _day(-10), _day(-10), tool_use("search_genes", query="TP53"), "s", None),
            ("s_new", _day(-5), _day(-5), tool_use("search_genes", query="TP53"), "s", None),
            (
                "s_cur",
                _day(0),
                _day(0),
                tool_use("search_genes", query="TP53"),
                "back to tp53 — following up",
                None,
            ),
        ],
    )
    loose = stats.collect(str(db))["clusters"]["all_time"]["min_shared_1"]
    assert loose["m2_remention_sessions"] == 1
    # the newest source (s_new) is the one in the 1-session recency window, so no miss
    assert loose["m3_source_outside_window_same_cluster"] == 0
    assert loose["m3_window_miss_share"] == 0.0


def test_pinned_source_is_not_a_window_miss(tmp_path, monkeypatch):
    """A pinned prior source counts as in-window, matching what the digest ships."""
    db = tmp_path / "chat_history.db"
    monkeypatch.setattr(stats, "DIGEST_WINDOW_SESSIONS", 1)
    build_sessions_db(
        db,
        "u",
        [
            (
                "s_src",
                _day(-10),
                _day(-10),
                tool_use("search_genes", query="APOE"),
                "s",
                _day(-10),  # pinned
            ),
            ("s_filler", _day(-5), _day(-5), tool_use("search_genes", query="MTHFR"), "s", None),
            (
                "s_cur",
                _day(0),
                _day(0),
                tool_use("search_genes", query="APOE"),
                "back to apoe",
                None,
            ),
        ],
    )
    loose = stats.collect(str(db))["clusters"]["all_time"]["min_shared_1"]
    assert loose["m2_remention_sessions"] == 1
    assert loose["m3_source_outside_window_same_cluster"] == 0


def test_m3_eligible_denominator_excludes_users_within_the_window(tmp_path, monkeypatch):
    """The diluted share and the eligible share diverge once one user is past the window."""
    db = tmp_path / "chat_history.db"
    monkeypatch.setattr(stats, "DIGEST_WINDOW_SESSIONS", 2)
    build_multi_user_db(
        db,
        {
            # short: 1 prior session, never enough to exceed the window -> ineligible
            "short": [
                ("a1", _day(-10), _day(-10), tool_use("search_genes", query="GENEX"), "s", None),
                (
                    "a2",
                    _day(0),
                    _day(0),
                    tool_use("search_genes", query="GENEX"),
                    "back to genex",
                    None,
                ),
            ],
            # long: 3 prior sessions (> window of 2), source pushed out of the window
            "long": [
                ("b1", _day(-10), _day(-10), tool_use("search_genes", query="GENEY"), "s", None),
                ("b2", _day(-8), _day(-8), tool_use("search_genes", query="FILLER1"), "s", None),
                ("b3", _day(-6), _day(-6), tool_use("search_genes", query="FILLER2"), "s", None),
                (
                    "b4",
                    _day(0),
                    _day(0),
                    tool_use("search_genes", query="GENEY"),
                    "back to geney",
                    None,
                ),
            ],
        },
    )

    loose = stats.collect(str(db))["clusters"]["all_time"]["min_shared_1"]
    assert loose["m2_remention_sessions"] == 2
    assert loose["m3_source_outside_window_same_cluster"] == 1
    assert loose["m3_window_miss_share"] == 0.5
    assert loose["m3_eligible_remention_sessions"] == 1
    assert loose["m3_window_miss_share_eligible"] == 1.0


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
