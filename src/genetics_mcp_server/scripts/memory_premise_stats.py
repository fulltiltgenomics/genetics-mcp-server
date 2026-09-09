"""Premise measurement for cross-session memory: is there anything to remember?

Two questions, one script: whether users return to earlier work at all (re-mention and
tool-argument rates), and whether that work falls into separate projects a recency digest
would confuse (M1/M2/M3, over `memory_digest.cluster_sessions`).

Reads chat_history.db read-only and prints ONE JSON object to stdout. Nothing else goes
to stdout, and no per-user row, user id, email or session id is ever emitted — the
production run pipes this file into `kubectl exec -i ... -- python -` and reads the
result back over the same pipe.

    python -m genetics_mcp_server.scripts.memory_premise_stats [DB_PATH]
    python -m genetics_mcp_server.scripts.memory_premise_stats --bundle > premise.py

The production image predates `memory_digest`, so the deployed interpreter cannot import
it. `--bundle` inlines that module's source above this one and cuts the bundle-only code
out again, printing a script that names nothing of this package; run that one against
production.
"""

import hashlib
import json
import os
import re
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from urllib.parse import quote

from genetics_mcp_server.memory_digest import (
    KINDS,
    STRICT_KINDS,
    cluster_sessions,
    extract_entities,
    extract_marker_entities,
    extract_user_entities,
)

DEFAULT_DB = "/data/chat_history.db"
EXCLUDED_USERS = ("anonymous", "mcp-tool")
WINDOW_DAYS = 90
# a value this short matches too much prose to mean anything as a re-mention
MIN_REMENTION_CHARS = 3
# memory_gate.MEMORY_PROJECT_SESSION_CAP, copied rather than imported because the bundle
# may import nothing of this package; a drift between the two makes M3 measure a window
# the product does not have
DIGEST_WINDOW_SESSIONS = 20
# both linkage thresholds are reported: at 1 a single shared gene merges two lines of
# work, at 2 a project has to be recognisable by more than one identifier
CLUSTER_MIN_SHARED = (1, 2)
# a user needs this many sessions before "does this user run several projects" is a
# question with an answer
M1_MIN_SESSIONS = 5
M1_MIN_CLUSTER_SIZE = 3
M1_MIN_CLUSTERS = 2
QUALITY_MAX_USERS = 40
QUALITY_TOP_ENTITIES = 5

REFER_BACK_RE = re.compile(
    r"\b(as before|last time|earlier (chat|conversation|session|analysis)|previous(ly)?|"
    r"we (discussed|looked|did|ran|found)|you (showed|gave|found|mentioned|said)|remember|"
    r"the (list|genes|analysis|query|plot|results?) (i|you|we) (gave|had|did|ran|made)|"
    r"continue (from|where)|again\b|same as (before|last))",
    re.I,
)

_TIMESTAMP_FORMATS = (
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
)


def parse_ts(raw):
    if raw is None:
        return None
    text = str(raw).replace("Z", "")
    for fmt in _TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(text[:26], fmt)
        except ValueError:
            pass
    return None


def percentile(sorted_values, fraction):
    if not sorted_values:
        return None
    return sorted_values[int(fraction * (len(sorted_values) - 1))]


def share(count, total):
    return round(count / total, 3) if total else None


def distribution(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    return {
        "n": len(values),
        "p25": percentile(values, 0.25),
        "median": percentile(values, 0.5),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.9),
        "mean": round(statistics.mean(values), 4),
    }


def session_entities(messages):
    """All entities of one session, plus the first user message and its first tool call."""
    entities = set()
    first_user = None
    first_tool_entities = set()
    seen_assistant = False
    for role, content, content_json in messages:
        if role == "user":
            if first_user is None:
                first_user = content or ""
            entities |= extract_user_entities(content)
        elif role == "assistant":
            found = extract_entities(content_json) | extract_marker_entities(content)
            # a text-only preamble row must not consume the first-turn slot
            if not seen_assistant and found:
                first_tool_entities = found
                seen_assistant = True
            entities |= found
    return entities, first_user, first_tool_entities


def mentioned_entities(text, entities):
    """The `entities` the text names as whole tokens — empty when it names none."""
    lowered = (text or "").lower()
    found = set()
    for kind, value in entities:
        if len(value) < MIN_REMENTION_CHARS:
            continue
        pattern = r"(?<![a-z0-9])" + re.escape(value.lower()) + r"(?![a-z0-9])"
        if re.search(pattern, lowered):
            found.add((kind, value))
    return found


def mentioned_kinds(text, entities):
    """Kinds of `entities` the text names — empty when it names none."""
    return {kind for kind, _ in mentioned_entities(text, entities)}


def user_hash(user_id):
    """The pseudonym memory_gate.user_log_hash produces, so the two logs name users alike.

    Inlined rather than imported for the same reason DIGEST_WINDOW_SESSIONS is; a test
    asserts the two agree.
    """
    return hashlib.sha256(str(user_id).strip().lower().encode()).hexdigest()[:12]


def recency_key(row):
    """Sort key reproducing get_recent_sessions_for_digest's `updated_at DESC, id DESC`.

    Sorted ascending, the last element is the row that accessor returns first. A row whose
    updated_at is null falls back to created_at rather than sorting as an empty string.
    """
    session_id, _user_id, created_at, updated_at, _pinned_at = row
    return (str(updated_at or created_at or ""), str(session_id))


def is_pinned(row):
    return row[4] is not None


def analyse_window(sessions_by_user, summary_by_session, window_start):
    counters = defaultdict(int)
    remention_kinds = defaultdict(int)
    toolarg_kinds = defaultdict(int)
    users = set()
    per_user_counts = defaultdict(int)
    gaps_hours = []
    for _user, sessions in sessions_by_user.items():
        # prior entities accumulate over ALL earlier sessions, not only in-window ones:
        # what a returning user re-mentions may have been said before the window opened
        prior_entities = set()
        prior_sessions = 0
        previous_start = None
        for session_id, user_id, created_at, _updated_at, _pinned_at in sessions:
            started = parse_ts(created_at)
            summary = summary_by_session.get(session_id)
            entities, first_user, first_tool_entities = summary or (set(), None, set())
            in_window = started is not None and started >= window_start
            if in_window:
                counters["sessions"] += 1
                users.add(user_id)
                per_user_counts[user_id] += 1
                if summary is None:
                    counters["empty_sessions"] += 1
                if previous_start is not None and started is not None:
                    gaps_hours.append((started - previous_start).total_seconds() / 3600)
                if prior_sessions >= 1:
                    counters["returning_sessions"] += 1
                    if first_user:
                        counters["returning_with_first_msg"] += 1
                        mentioned = mentioned_kinds(first_user, prior_entities)
                        if mentioned:
                            counters["first_msg_rementions_prior_entity"] += 1
                        if mentioned & STRICT_KINDS:
                            counters["first_msg_rementions_prior_entity_strict"] += 1
                        for kind in mentioned:
                            remention_kinds[kind] += 1
                        if REFER_BACK_RE.search(first_user):
                            counters["first_msg_refer_back_phrase"] += 1
                        overlap_kinds = {k for k, _ in first_tool_entities & prior_entities}
                        if overlap_kinds:
                            counters["first_turn_toolarg_overlap"] += 1
                        if overlap_kinds & STRICT_KINDS:
                            counters["first_turn_toolarg_overlap_strict"] += 1
                        for kind in overlap_kinds:
                            toolarg_kinds[kind] += 1
            prior_entities |= entities
            prior_sessions += 1
            if started is not None:
                previous_start = started

    total = counters["sessions"]
    returning = counters["returning_sessions"]
    with_first = counters["returning_with_first_msg"]
    gaps_hours.sort()
    counts = sorted(per_user_counts.values())
    return {
        "sessions": total,
        "empty_sessions": counters["empty_sessions"],
        "users": len(users),
        "returning_sessions": returning,
        "returning_share": share(returning, total),
        "returning_with_first_msg": with_first,
        "first_msg_rementions_prior_entity": counters["first_msg_rementions_prior_entity"],
        "remention_share_of_returning": share(
            counters["first_msg_rementions_prior_entity"], with_first
        ),
        "first_msg_rementions_prior_entity_strict": counters[
            "first_msg_rementions_prior_entity_strict"
        ],
        "remention_share_strict": share(
            counters["first_msg_rementions_prior_entity_strict"], with_first
        ),
        "remention_kinds": {kind: remention_kinds[kind] for kind in KINDS},
        "first_msg_refer_back_phrase": counters["first_msg_refer_back_phrase"],
        "refer_back_share_of_returning": share(
            counters["first_msg_refer_back_phrase"], with_first
        ),
        "first_turn_toolarg_overlap": counters["first_turn_toolarg_overlap"],
        "toolarg_share_of_returning": share(
            counters["first_turn_toolarg_overlap"], with_first
        ),
        "first_turn_toolarg_overlap_strict": counters["first_turn_toolarg_overlap_strict"],
        "toolarg_share_strict": share(
            counters["first_turn_toolarg_overlap_strict"], with_first
        ),
        "toolarg_kinds": {kind: toolarg_kinds[kind] for kind in KINDS},
        "inter_session_gap_hours": {
            "n": len(gaps_hours),
            "median": round(percentile(gaps_hours, 0.5), 2) if gaps_hours else None,
            "share_lt_5min": share(sum(g < 5 / 60 for g in gaps_hours), len(gaps_hours)),
            "share_lt_1h": share(sum(g < 1 for g in gaps_hours), len(gaps_hours)),
            "share_lt_24h": share(sum(g < 24 for g in gaps_hours), len(gaps_hours)),
        },
        "sessions_per_user": {
            "users": len(counts),
            "median": statistics.median(counts) if counts else None,
            "p90": percentile(counts, 0.9),
            "max": counts[-1] if counts else None,
            "share_gt20": share(sum(c > 20 for c in counts), len(counts)),
            "share_gt50": share(sum(c > 50 for c in counts), len(counts)),
        },
    }


def window_sessions(rows, window_start):
    """The user's sessions inside the window, in creation order."""
    kept = []
    for row in rows:
        started = parse_ts(row[2])
        if started is not None and started >= window_start:
            kept.append(row)
    return kept


def session_entity_sets(rows, summary_by_session):
    return {
        row[0]: (summary_by_session.get(row[0]) or (set(), None, set()))[0] for row in rows
    }


def analyse_clusters(sessions_by_user, summary_by_session, window_start, min_shared):
    """The epic's M1/M2/M3 for one window at one linkage threshold.

    Each user's in-window sessions are clustered by shared strict entities as a stand-in
    for "project". "Prior" means earlier in creation order, while the preceding session
    and the last-20 window are read off the recency order the digest itself uses, so M2
    and M3 describe what the shipped digest would have led with.
    """
    m1_users = m1_multi = 0
    rementions = different_cluster = window_miss = ambiguous_sessions = 0
    eligible_rementions = 0
    clusters_per_user = []
    for rows in sessions_by_user.values():
        in_window = window_sessions(rows, window_start)
        if not in_window:
            continue
        entity_sets = session_entity_sets(in_window, summary_by_session)
        clusters = cluster_sessions(entity_sets, min_shared)
        sizes = Counter(clusters.values())
        if len(in_window) >= M1_MIN_SESSIONS:
            m1_users += 1
            clusters_per_user.append(len(sizes))
            if sum(size >= M1_MIN_CLUSTER_SIZE for size in sizes.values()) >= M1_MIN_CLUSTERS:
                m1_multi += 1

        prior_rows = []
        prior_row_by_id = {}
        prior_by_entity = defaultdict(list)
        for row in in_window:
            session_id = row[0]
            entities, first_user, _first_tool = summary_by_session.get(session_id) or (
                set(),
                None,
                set(),
            )
            if prior_rows and first_user:
                matched = mentioned_entities(first_user, prior_by_entity)
                if matched:
                    rementions += 1
                    # a re-mention can only be a window miss when the user had more than
                    # DIGEST_WINDOW_SESSIONS prior in-window sessions at this point; below
                    # that, every prior session is in the recency window by construction
                    eligible = len(prior_rows) > DIGEST_WINDOW_SESSIONS
                    if eligible:
                        eligible_rementions += 1
                    by_recency = sorted(prior_rows, key=recency_key, reverse=True)
                    preceding_cluster = clusters[by_recency[0][0]]
                    # get_recent_sessions_for_digest returns pinned sessions in addition to
                    # the recency window, so a pinned prior source is never a miss
                    recent = {r[0] for r in by_recency[:DIGEST_WINDOW_SESSIONS]}
                    recent |= {r[0] for r in by_recency if is_pinned(r)}
                    is_different = is_miss = ambiguous = False
                    for entity in matched:
                        sources = prior_by_entity[entity]
                        source_clusters = {clusters[sid] for sid in sources}
                        if len(source_clusters) > 1:
                            # the entity was worked on in more than one cluster, so which
                            # project the user means is unknowable: counted as no evidence
                            ambiguous = True
                            continue
                        source_cluster = source_clusters.pop()
                        if source_cluster != preceding_cluster:
                            is_different = True
                        # the digest surfaces the most recent occurrence, so that is the
                        # one that has to fall outside the window for it to be a miss
                        newest = max(
                            (prior_row_by_id[sid] for sid in sources), key=recency_key
                        )
                        if newest[0] not in recent and source_cluster == clusters[session_id]:
                            is_miss = True
                    different_cluster += is_different
                    window_miss += is_miss
                    ambiguous_sessions += ambiguous

            prior_rows.append(row)
            prior_row_by_id[session_id] = row
            for entity in entities:
                if entity[0] in STRICT_KINDS:
                    prior_by_entity[entity].append(session_id)

    return {
        "min_shared": min_shared,
        "m1_users_with_min_sessions": m1_users,
        "m1_multi_cluster_users": m1_multi,
        "m1_multi_thread_share": share(m1_multi, m1_users),
        "m2_remention_sessions": rementions,
        "m2_preceding_in_other_cluster": different_cluster,
        "m2_interleaving_share": share(different_cluster, rementions),
        "m3_source_outside_window_same_cluster": window_miss,
        "m3_window_miss_share": share(window_miss, rementions),
        "m3_eligible_remention_sessions": eligible_rementions,
        "m3_window_miss_share_eligible": share(window_miss, eligible_rementions),
        "remention_sessions_with_ambiguous_entity": ambiguous_sessions,
        "clusters_per_user": distribution(clusters_per_user),
    }


def cluster_quality(sessions_by_user, summary_by_session, min_shared):
    """Per-user cluster shapes, to judge by eye whether a cluster looks like a project.

    A user is named only by the pseudonym the memory log uses. No session id, address or
    message text appears here; the entity values are the identifiers already counted above.
    """
    entries = []
    for user_id, rows in sessions_by_user.items():
        if len(rows) < 2:
            continue
        entity_sets = session_entity_sets(rows, summary_by_session)
        clusters = cluster_sessions(entity_sets, min_shared)
        members = defaultdict(list)
        for session_id, cluster_id in clusters.items():
            members[cluster_id].append(session_id)
        entry = {"user": user_hash(user_id), "sessions": len(rows), "clusters": []}
        for cluster_id, session_ids in sorted(
            members.items(), key=lambda kv: (-len(kv[1]), kv[0])
        ):
            cluster = {"size": len(session_ids)}
            if len(session_ids) >= 2:
                counts = Counter(
                    entity
                    for session_id in session_ids
                    for entity in entity_sets[session_id]
                    if entity[0] in STRICT_KINDS
                )
                cluster["top_entities"] = [
                    {"entity": f"{kind}:{value}", "sessions": count}
                    for (kind, value), count in sorted(
                        counts.items(), key=lambda kv: (-kv[1], kv[0])
                    )[:QUALITY_TOP_ENTITIES]
                ]
            entry["clusters"].append(cluster)
        entries.append(entry)
    entries.sort(key=lambda entry: (-entry["sessions"], entry["user"]))
    return {
        "users": entries[:QUALITY_MAX_USERS],
        "users_omitted": max(0, len(entries) - QUALITY_MAX_USERS),
    }


CLUSTER_NOTES = [
    "sessions are clustered per user by single-linkage over shared strict entities "
    "(gene/phenotype/variant); dataset and view are excluded because nearly every session "
    "carries them",
    "M1: share of users with >=5 in-window sessions that have >=2 clusters of >=3 sessions",
    "M2: among returning sessions whose first user message re-mentions a strict entity of "
    "an earlier session, the share where the session the digest would have led with "
    "(updated_at DESC, id DESC among earlier sessions) is in a different cluster from the "
    "entity's source cluster — the cluster every one of that entity's earlier sessions "
    "shares (ambiguous, and excluded, when they do not agree)",
    "M3: the same denominator; the share whose source session (the most recent earlier "
    "session naming that entity) falls outside the last-20 recency window yet sits in the "
    "current session's cluster — the miss a project memory recovers. A pinned source "
    "session counts as in-window regardless of recency, matching what the digest ships",
    "m3_window_miss_share's denominator is every re-mention session, most of which cannot "
    "structurally be a window miss because the user had <=20 prior in-window sessions at "
    "that point; m3_window_miss_share_eligible restricts the denominator to sessions where "
    "the user had more than 20, and is the share the ~10% gate should read",
    "conservative: an entity that occurred in earlier sessions of more than one cluster is "
    "counted as no evidence for either M2 or M3, and such sessions are reported separately "
    "as remention_sessions_with_ambiguous_entity",
    "the clustering universe is the window's own sessions, so last_90d clusters only "
    "sessions created in the last 90 days",
    "when clusters.pinned_visible is false, chat_sessions has no pinned_at column to read "
    "(an older production schema) and every source session outside the recency window "
    "counts as a miss even if it was pinned, which biases M3 up",
]


def sessions_pinned_column(con):
    """Whether chat_sessions has pinned_at, mirroring turn_metrics' table-missing pattern.

    Production's chat_sessions predates pinning; selecting the column unconditionally
    raises OperationalError before collect() gets past its first query.
    """
    columns = {row[1] for row in con.execute("PRAGMA table_info(chat_sessions)")}
    visible = "pinned_at" in columns
    return ("pinned_at" if visible else "NULL AS pinned_at"), visible


def turn_metrics(con, cutoff):
    """Per-turn cost from chat_turn_metrics, or a note that the table is not there.

    Production runs an image older than the table; that side of the premise comes from the
    BigQuery log sink instead (scripts/memory_premise_cost.sql).
    """
    try:
        con.execute("SELECT 1 FROM chat_turn_metrics LIMIT 1").fetchall()
    except sqlite3.OperationalError:
        return "table missing"
    columns = ("cost_usd", "input_tokens", "cache_read_tokens", "iterations")
    out = {}
    for label, where in (
        ("all_time", ""),
        ("last_90d", f" AND created_at >= '{cutoff:%Y-%m-%d}'"),
    ):
        out[label] = {
            column: distribution(
                row[0]
                for row in con.execute(
                    f"SELECT {column} FROM chat_turn_metrics "
                    f"WHERE {column} IS NOT NULL AND user_id NOT IN (?,?){where}",
                    EXCLUDED_USERS,
                )
            )
            for column in columns
        }
    return out


def summarise_sessions(con):
    """One summary per session, reducing each session's rows as they stream past.

    The production run shares a 2Gi limit with the live server, so the message text never
    accumulates: rows arrive grouped by session and only the finished summaries — sets of
    short strings plus one first message — outlive the session they came from.
    """
    summaries = {}
    current_id = None
    current_rows = []
    cursor = con.execute(
        "SELECT session_id, role, content, content_json FROM chat_messages "
        "WHERE session_id IN (SELECT id FROM chat_sessions WHERE user_id NOT IN (?,?)) "
        "ORDER BY session_id, created_at, rowid",
        EXCLUDED_USERS,
    )
    for session_id, role, content, content_json in cursor:
        if session_id != current_id:
            if current_id is not None:
                summaries[current_id] = session_entities(current_rows)
            current_id, current_rows = session_id, []
        current_rows.append((role, content, content_json))
    if current_id is not None:
        summaries[current_id] = session_entities(current_rows)
    return summaries


def assistant_payload_sizes(con):
    """Size distribution of stored assistant `content_json`, measured a row at a time."""
    sizes = sorted(
        len(raw or "")
        for (raw,) in con.execute(
            "SELECT content_json FROM chat_messages WHERE role='assistant'"
        )
    )
    return {
        "n": len(sizes),
        "median": percentile(sizes, 0.5),
        "p90": percentile(sizes, 0.9),
        "max": sizes[-1] if sizes else None,
    }


def collect(db_path):
    con = sqlite3.connect(f"file:{quote(db_path)}?mode=ro", uri=True)
    try:
        pinned_expr, pinned_visible = sessions_pinned_column(con)
        sessions = con.execute(
            f"SELECT id, user_id, created_at, updated_at, {pinned_expr} FROM chat_sessions "
            "WHERE user_id NOT IN (?,?) ORDER BY created_at, rowid",
            EXCLUDED_USERS,
        ).fetchall()
        summary_by_session = summarise_sessions(con)

        sessions_by_user = defaultdict(list)
        for row in sessions:
            sessions_by_user[row[1]].append(row)

        cutoff = datetime.now() - timedelta(days=WINDOW_DAYS)
        out = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "window_days": WINDOW_DAYS,
            "excluded_users": list(EXCLUDED_USERS),
            "span": {
                "first_session": sessions[0][2] if sessions else None,
                "last_session": sessions[-1][2] if sessions else None,
            },
            "all_time": analyse_window(
                sessions_by_user, summary_by_session, datetime(2000, 1, 1)
            ),
            "last_90d": analyse_window(sessions_by_user, summary_by_session, cutoff),
            "clusters": {
                "notes": CLUSTER_NOTES,
                "digest_window_sessions": DIGEST_WINDOW_SESSIONS,
                "pinned_visible": pinned_visible,
                "all_time": {
                    f"min_shared_{n}": analyse_clusters(
                        sessions_by_user, summary_by_session, datetime(2000, 1, 1), n
                    )
                    for n in CLUSTER_MIN_SHARED
                },
                "last_90d": {
                    f"min_shared_{n}": analyse_clusters(
                        sessions_by_user, summary_by_session, cutoff, n
                    )
                    for n in CLUSTER_MIN_SHARED
                },
                # one dump, over every session a user has: the shapes are what is being
                # judged, and repeating them per window would say the same thing twice
                "quality_all_time": {
                    f"min_shared_{n}": cluster_quality(
                        sessions_by_user, summary_by_session, n
                    )
                    for n in CLUSTER_MIN_SHARED
                },
            },
            "chat_turn_metrics": turn_metrics(con, cutoff),
            "assistant_content_json_chars": assistant_payload_sizes(con),
        }
        return out
    finally:
        con.close()


# BUNDLE-ONLY-START
# the import --bundle strips; the `^from` anchor is what keeps the pattern from matching
# the indented line that spells it
_BUNDLE_IMPORT_RE = re.compile(
    r"^from genetics_mcp_server\.memory_digest import \([^)]*\)\n", re.M
)
# this module's own header, replaced because the bundle is a different artifact
_BUNDLE_DOCSTRING_RE = re.compile(r'\A""".*?"""\n', re.S)
# the bundle-only region and the `--bundle` branch: both name this package, so a bundle
# that kept them would not run where nothing of this package is importable
_BUNDLE_REGION_RE = re.compile(r"^# BUNDLE-ONLY-START\n.*?^# BUNDLE-ONLY-END\n", re.S | re.M)
_BUNDLE_BRANCH_RE = re.compile(r"^.*# bundle-only$\n", re.M)

_BUNDLE_HEADER = '''"""Premise measurement for cross-session memory, as one self-contained script.

GENERATED by memory_premise_stats.py --bundle; do not edit. Imports nothing but the
standard library, so it runs inside a deployed image that predates the entity extractor.

    python premise.py [DB_PATH]
"""
'''


def build_bundle():
    """One self-contained script: memory_digest's source inlined above this module's."""
    from pathlib import Path

    from genetics_mcp_server import memory_digest

    digest_source = Path(memory_digest.__file__).read_text()
    inlined = "# --- memory_digest, inlined by --bundle ---\n" + digest_source + "\n\n"
    stats_source = Path(__file__).read_text()
    for pattern, replacement, expected in (
        (_BUNDLE_DOCSTRING_RE, _BUNDLE_HEADER, 1),
        (_BUNDLE_IMPORT_RE, inlined, 1),
        (_BUNDLE_REGION_RE, "", 1),
        (_BUNDLE_BRANCH_RE, "", 3),
    ):
        stats_source, replaced = pattern.subn(lambda _m, r=replacement: r, stats_source)
        if replaced != expected:
            raise RuntimeError(
                f"--bundle expected {expected} substitutions for {pattern.pattern!r}, "
                f"made {replaced}"
            )
    return stats_source
# BUNDLE-ONLY-END


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--bundle" in argv:  # bundle-only
        sys.stdout.write(build_bundle())  # bundle-only
        return 0  # bundle-only
    db_path = argv[0] if argv else os.environ.get("CHAT_HISTORY_DB", DEFAULT_DB)
    print(json.dumps(collect(db_path), indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
