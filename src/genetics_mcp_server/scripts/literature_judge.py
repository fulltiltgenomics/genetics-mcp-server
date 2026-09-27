"""Literature-evidence judge: how critically does a chat turn appraise the papers it cites?

The nightly quality judge (`analyze_conversations`) never sees tool results, so it cannot
tell a paper's finding from a Perplexity summary sentence or from the model's memory. This
judge reads each literature-bearing turn's answer BESIDE its persisted literature results and
reports findings in the eight categories of the human review in genetics-results-suite
`docs/research/literature-critical-evaluation/` (`rubric.md`; the labelled findings `[F..]`
and counter-examples `[G..]` are in its `review-*.md` files).

Three modes, all sharing one judge call (`LITERATURE_EVIDENCE_JUDGE_PROMPT`) and one cache:

    # baseline: every persisted literature-bearing turn of the named users, from prod
    python -m genetics_mcp_server.scripts.literature_judge --db --cache-dir DIR

    # calibration against the human labels, on one deterministic half of the labelled turns
    python -m genetics_mcp_server.scripts.literature_judge --calibrate dev --cache-dir DIR
    python -m genetics_mcp_server.scripts.literature_judge --calibrate heldout --cache-dir DIR

    # score a replay run
    python -m genetics_mcp_server.scripts.literature_judge --report replay.json --cache-dir DIR

Prod rows are read ONCE through `kubectl exec -i deploy/chat-backend -- python3 -` opening
`file:/data/chat_history.db?mode=ro`, printed to stdout and cached in `--cache-dir` as
`prod_rows-<hash>.json`, the hash taken over `--context` and the sorted `--users`, so a run
for other users or another cluster never reuses the rows; `--refresh-rows` re-reads. Judge
responses are cached in `judgements.jsonl` keyed by a hash of the model and the rendered
prompt. The date the prompt calls "today" is the turn's own `created_at` (the day the answer
was written), so the key does not change from one day to the next and re-running after a
prompt change re-pays only for what changed; a replay turn has no `created_at`, so it is
given the run's date, which is left out of its key. `ANTHROPIC_API_KEY` must be in the
environment.

Replay input (`--report`): a JSON list of turns, or an object whose `turns` key holds one.
Each turn is an object with
    question            str, the user's message
    answer              str, the assistant's final answer text
    literature_results  list; each item either a string (the tool result as returned) or an
                        object with `content` (str) and optionally `name` and `input`
    id                  optional str, used in the per-turn output
    earlier_results     optional list, same shape, literature results of earlier turns
A turn with an empty `literature_results` is still judged (a literature section written with
no search is a finding in its own right), but reported separately from literature-bearing
turns so the two rates are not mixed.

`--report` also accepts replay_benchmark.py's OWN `--output` JSON directly, with no
conversion step: it is an object with a `turns` key whose entries carry `user_question`,
`final_answer`, `status` and `literature_results` in place of `question`/`answer`/`id`. The
loader tells the two shapes apart by the presence of `status` (the documented shape never
has one) and drops every turn whose `status` isn't `"ok"` — an error/timeout/not_attempted
turn carries no answer to judge. `earlier_results` for this shape is rebuilt from the
turns' own `case_id`/`arm`/`turn_index`, the same per-(case, arm) accumulation the prod
path (`build_turns`) does per session, since the harness's own report carries no such
field per turn.

Calibration hazard. The human labels were made on a review dump that truncated literature
results at 9,000 characters, and turns before July 2026 have no persisted tool results at
all, so a label on such a turn was made from the answer text alone. Agreement is therefore
computed ONLY on labelled turns whose record is persisted (`content_json` and
`tool_results_json` both non-empty). Where a label was made against the truncated dump, the
judge reads more than the labeller did; the disagreement listing that `--calibrate` prints
for either half is where such a label shows up, and it should be re-checked against the full
record before being counted as a judge error. A label that such a re-check overturned carries
a `- VOID <date>: <reason>` line in its review file and is skipped by the loader.

Agreement. Labelled turns are split into halves by a hash of the message id (`half_of`), so
F and G labels of one turn always land together. For findings the unit is (turn, category):
the fraction of human (turn, category) pairs that the judge also reports on that turn. For
counter-examples most `[G..]` entries name no category, so the unit is the entry: it is
violated when the judge raises a finding on that turn that quotes the passage the reviewer
called good (a shared run of words with one of the entry's quotes), or, for an entry that
does name a category ("category 6 handled correctly"), any finding in that category. The
combined score pools both units. A judge finding with no rubric category is not a rubric
finding, whatever its text says, so it takes part in neither unit; the run reports how many
were dropped.

Category mix. Categories 1/2/5/8 are called record-decidable here and 3/4/7 claim-level, as
in the planning notes; a finding is record-only, claim-only, both, or other (category 6
alone).
"""

import argparse
import asyncio
import hashlib
import json
import logging
import re
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from genetics_mcp_server.scripts.analyze_conversations import (
    CostTracker,
    create_with_backoff,
    extract_first_json,
    response_text,
    thinking_off_kwargs,
)
from genetics_mcp_server.scripts.conversation_prompts import LITERATURE_EVIDENCE_JUDGE_PROMPT

load_dotenv()

logger = logging.getLogger("literature_judge")

LITERATURE_TOOL = "search_scientific_literature"
SUBAGENT_TOOL = "launch_subagents"
LITERATURE_SKILL = "literature_review"

PROD_CONTEXT = "gke_daly-finngenie_us-central1-a_finngenie"
DEFAULT_USERS = ("bneale", "mjdaly", "finucane")
DEFAULT_MODEL = "claude-opus-5"
# a turn's findings list runs to a few KB of JSON; the nightly judge's 1000 truncates it
DEFAULT_MAX_TOKENS = 8000

# what the answering model could see of one tool result (settings.mcp_max_result_size), so the
# judge sees no less of the record than the claim was written from
RESULT_CAP = 50_000
# earlier turns' results are context only, bounded in total by EARLIER_CAP; a per-result cap
# below that total keeps the most recent one from being omitted whole
EARLIER_RESULT_CAP = 20_000
ANSWER_CAP = 60_000
QUESTION_CAP = 4_000
EARLIER_CAP = 30_000

CATEGORIES = range(1, 9)
RECORD_DECIDABLE = frozenset({1, 2, 5, 8})
CLAIM_LEVEL = frozenset({3, 4, 7})

DEFAULT_LABELS_DIR = (
    Path(__file__).resolve().parents[3].parent
    / "genetics-results-suite/docs/research/literature-critical-evaluation"
)


# ---------------------------------------------------------------------------
# Turns
# ---------------------------------------------------------------------------

def _json_list(raw) -> list:
    if not raw:
        return []
    try:
        value = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        return []
    return value if isinstance(value, list) else []


def literature_tool_uses(content_json) -> list[dict]:
    """The literature tool_use blocks of one assistant message's content_json.

    This is the single definition of what counts as a literature call: a
    `search_scientific_literature` call, or a `launch_subagents` call with at least one
    `literature_review` task (its result is the subagents' digest of their own searches).
    `web_search` is excluded: the rubric is about papers, and web pages are judged only when
    the answer presents them as literature.
    """
    uses = []
    for block in _json_list(content_json):
        if not isinstance(block, dict) or block.get("type", "tool_use") != "tool_use":
            continue
        name = block.get("name")
        if name == LITERATURE_TOOL:
            uses.append(block)
        elif name == SUBAGENT_TOOL:
            tasks = (block.get("input") or {}).get("tasks") or []
            if any(isinstance(t, dict) and t.get("skill") == LITERATURE_SKILL for t in tasks):
                uses.append(block)
    return uses


def is_literature_bearing(row: dict) -> bool:
    return row.get("role") == "assistant" and bool(literature_tool_uses(row.get("content_json")))


def is_persisted(row: dict) -> bool:
    return bool(_json_list(row.get("content_json"))) and bool(_json_list(row.get("tool_results_json")))


def _result_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )
    return json.dumps(content, default=str)


def literature_results(row: dict) -> list[dict]:
    """Literature tool results of one assistant message, matched to their calls by tool_use_id."""
    by_id = {
        r.get("tool_use_id"): r
        for r in _json_list(row.get("tool_results_json"))
        if isinstance(r, dict)
    }
    out = []
    for use in literature_tool_uses(row.get("content_json")):
        result = by_id.get(use.get("id"))
        out.append({
            "name": use.get("name"),
            "input": use.get("input"),
            "content": _result_text(result.get("content")) if result else "",
        })
    return out


@dataclass
class Turn:
    id: str
    question: str
    answer: str
    literature_results: list[dict]
    earlier_results: list[dict] = field(default_factory=list)
    session_id: str = ""
    user: str = ""
    created_at: str = ""
    persisted: bool = True
    arm: str = ""  # only the harness's --output shape carries this (see load_replay)

    @property
    def literature_bearing(self) -> bool:
        return bool(self.literature_results)


def build_turns(rows: list[dict]) -> list[Turn]:
    """One Turn per assistant row; the question is the latest preceding user row of its session."""
    sessions: dict[str, list[dict]] = {}
    for r in rows:
        sessions.setdefault(r["session_id"], []).append(r)
    turns = []
    for sid, msgs in sessions.items():
        msgs.sort(key=lambda r: (r.get("created_at") or "", r.get("role") != "user"))
        question = ""
        earlier: list[dict] = []
        for r in msgs:
            if r.get("role") == "user":
                question = r.get("content") or ""
                continue
            if r.get("role") != "assistant":
                continue
            lit = literature_results(r)
            turns.append(Turn(
                id=r["id"], question=question, answer=r.get("content") or "",
                literature_results=lit, earlier_results=list(earlier), session_id=sid,
                user=(r.get("user_id") or "").split("@")[0],
                created_at=r.get("created_at") or "", persisted=is_persisted(r),
            ))
            earlier.extend(lit)
    return turns


def load_replay(path: Path) -> list[Turn]:
    """Load a replay-JSON turn list, in either the documented shape or the harness's own
    `--output` report shape — see the module docstring for both.
    """
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        data = data.get("turns")
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a list of turns or an object with 'turns'")

    def results(items) -> list[dict]:
        out = []
        for item in items or []:
            if isinstance(item, str):
                out.append({"name": LITERATURE_TOOL, "input": None, "content": item})
            elif isinstance(item, dict) and "content" in item:
                out.append({"name": item.get("name", LITERATURE_TOOL), "input": item.get("input"),
                            "content": _result_text(item["content"])})
            else:
                raise ValueError(f"{path}: a literature result must be a string or have 'content'")
        return out

    entries: list[dict | None] = []  # None marks a skipped (non-"ok") harness turn
    harness_by_group: dict[tuple, list[dict]] = {}
    skipped = 0
    for i, t in enumerate(data):
        if not isinstance(t, dict):
            raise ValueError(f"{path}: turn {i} is not an object")
        if "status" in t and "question" not in t:
            # replay_benchmark.py's own TurnRecord shape: only "ok" turns have an answer
            if t.get("status") != "ok":
                skipped += 1
                entries.append(None)
                continue
            entry = {
                "turn_id": f"{t.get('case_id', '?')}:{t.get('arm', '?')}:{t.get('turn_index', i)}",
                "question": t.get("user_question") or "",
                "answer": t.get("final_answer") or "",
                "literature_results": results(t.get("literature_results")),
                "turn_index": t.get("turn_index", i),
                "arm": t.get("arm") or "",
            }
            harness_by_group.setdefault((t.get("case_id"), t.get("arm")), []).append(entry)
            entries.append(entry)
        elif "question" in t and "answer" in t:
            entries.append({
                "turn_id": str(t.get("id", i)), "question": t["question"] or "",
                "answer": t["answer"] or "",
                "literature_results": results(t.get("literature_results")),
                "earlier_results": results(t.get("earlier_results")),
            })
        else:
            raise ValueError(
                f"{path}: turn {i} lacks 'question'/'answer' (or the harness's "
                "'user_question'/'final_answer' with status=='ok')"
            )

    # the harness's own report carries no earlier_results field per turn (TurnRecord has
    # none) — rebuild the same per-(case, arm) accumulation build_turns does per session,
    # in turn_index order, so a multi-turn replay case gets the context prod turns get.
    for group in harness_by_group.values():
        group.sort(key=lambda e: e["turn_index"])
        earlier: list[dict] = []
        for entry in group:
            entry["earlier_results"] = list(earlier)
            earlier.extend(entry["literature_results"])

    turns = [
        Turn(id=e["turn_id"], question=e["question"], answer=e["answer"],
             literature_results=e["literature_results"], earlier_results=e["earlier_results"],
             arm=e.get("arm", ""))
        for e in entries if e is not None
    ]
    if skipped:
        logger.info("--report: skipped %d harness turn(s) with status != 'ok'", skipped)
    return turns


# ---------------------------------------------------------------------------
# Human labels
# ---------------------------------------------------------------------------

LABEL_HEAD_RE = re.compile(r"^\s*-\s+\*\*\[(?P<kind>[FG])(?P<num>\d+)\]\s+(?P<attrs>[^*]*)\*\*")
ATTR_RE = re.compile(r"(\w+)=(\S+)")
PROBLEM_RE = re.compile(r"Problem:\s*(?P<cats>[\d,\s/]+)")
SEVERITY_RE = re.compile(r"Severity:\s*(?P<sev>[a-z/-]+)", re.I)
NAMED_CATEGORY_RE = re.compile(r"categor(?:y|ies)[\s-]*(\d)", re.I)
QUOTE_RE = re.compile(r"[\"“]([^\"”]{12,}?)[\"”]")
VOID_RE = re.compile(r"^\s*-\s+VOID\b", re.M)


@dataclass
class Label:
    kind: str
    num: int
    source: str
    session: str
    msg: str
    categories: frozenset
    severity: str
    quotes: list[str]
    text: str
    void: bool = False

    @property
    def key(self) -> str:
        return f"{self.source}:{self.kind}{self.num}"


def parse_labels(text: str, source: str) -> list[Label]:
    """The [F..] and [G..] entries of one review file.

    An entry runs from its bold header line to the next header or heading. Findings take
    their categories from the `Problem:` line; counter-examples name one only when the
    reviewer wrote "category N", so theirs is usually empty.
    """
    labels = []
    current: dict | None = None

    def close():
        if current is None:
            return
        body = "\n".join(current["lines"])
        if current["kind"] == "F":
            m = PROBLEM_RE.search(body)
            cats = {int(c) for c in re.findall(r"\d", m.group("cats"))} if m else set()
        else:
            cats = {int(c) for c in NAMED_CATEGORY_RE.findall(body)}
        sev = SEVERITY_RE.search(body)
        labels.append(Label(
            kind=current["kind"], num=current["num"], source=source,
            session=current["attrs"].get("session", ""), msg=current["attrs"].get("msg", ""),
            categories=frozenset(c for c in cats if c in CATEGORIES),
            severity=sev.group("sev").lower().rstrip(".") if sev else "",
            quotes=QUOTE_RE.findall(body), text=body, void=bool(VOID_RE.search(body)),
        ))

    for line in text.splitlines():
        head = LABEL_HEAD_RE.match(line)
        if head:
            close()
            current = {"kind": head["kind"], "num": int(head["num"]),
                       "attrs": dict(ATTR_RE.findall(head["attrs"])), "lines": [line]}
        elif line.startswith("#"):
            close()
            current = None
        elif current is not None:
            current["lines"].append(line)
    close()
    return labels


def load_labels(labels_dir: Path) -> tuple[list[Label], list[Label]]:
    """The live labels of every review file, and separately those a VOID line overturned."""
    labels = []
    for path in sorted(Path(labels_dir).glob("review-*.md")):
        labels.extend(parse_labels(path.read_text(), path.stem))
    return [lab for lab in labels if not lab.void], [lab for lab in labels if lab.void]


def resolve_labels(labels: list[Label], turns: list[Turn]) -> tuple[dict[str, list[Label]], list[Label]]:
    """Map labels to turns by message-id prefix; reviewers wrote both full ids and 8-char ones."""
    by_id = {t.id: t for t in turns}
    ids = sorted(by_id)
    resolved: dict[str, list[Label]] = {}
    unresolved = []
    for lab in labels:
        matches = [i for i in ids if lab.msg and i.startswith(lab.msg)
                   and (not lab.session or by_id[i].session_id.startswith(lab.session))]
        if len(matches) == 1:
            resolved.setdefault(matches[0], []).append(lab)
        else:
            unresolved.append(lab)
    return resolved, unresolved


def half_of(msg_id: str) -> str:
    return "dev" if hashlib.sha256(msg_id.encode()).digest()[0] % 2 == 0 else "heldout"


# ---------------------------------------------------------------------------
# Agreement
# ---------------------------------------------------------------------------

def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _shingles(text: str, n: int) -> set[tuple]:
    w = _words(text)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def quotes_overlap(quote: str, claim: str, n: int = 6) -> bool:
    """True when the two texts share a run of n words (fewer for a short quote, never under 4)."""
    k = min(n, len(_words(quote)))
    return k >= 4 and bool(_shingles(quote, k) & _shingles(claim, k))


def _finding_categories(finding: dict) -> set[int]:
    cats = finding.get("categories") or []
    out = set()
    for c in cats if isinstance(cats, list) else [cats]:
        try:
            out.add(int(c))
        except (TypeError, ValueError):
            continue
    return out & set(CATEGORIES)


@dataclass
class Agreement:
    finding_pairs: int = 0
    finding_hits: int = 0
    counter_examples: int = 0
    counter_kept: int = 0
    turns: int = 0
    uncategorised: int = 0
    missed: list = field(default_factory=list)
    violated: list = field(default_factory=list)

    @property
    def finding_recall(self) -> float | None:
        return self.finding_hits / self.finding_pairs if self.finding_pairs else None

    @property
    def counter_precision(self) -> float | None:
        return self.counter_kept / self.counter_examples if self.counter_examples else None

    @property
    def combined(self) -> float | None:
        n = self.finding_pairs + self.counter_examples
        return (self.finding_hits + self.counter_kept) / n if n else None


def score_agreement(labels_by_turn: dict[str, list[Label]], judgements: dict[str, dict]) -> Agreement:
    agg = Agreement()
    for turn_id, labels in labels_by_turn.items():
        judged = judgements.get(turn_id)
        if judged is None:
            continue
        agg.turns += 1
        raw = [f for f in judged.get("findings") or [] if isinstance(f, dict)]
        findings = [f for f in raw if _finding_categories(f)]
        agg.uncategorised += len(raw) - len(findings)
        judge_cats = set().union(*(_finding_categories(f) for f in findings)) if findings else set()
        human_cats = set().union(*(lab.categories for lab in labels if lab.kind == "F"))
        for cat in sorted(human_cats):
            agg.finding_pairs += 1
            if cat in judge_cats:
                agg.finding_hits += 1
            else:
                agg.missed.append((turn_id, cat))
        for lab in labels:
            if lab.kind != "G":
                continue
            agg.counter_examples += 1
            hit = [f for f in findings
                   if (lab.categories & _finding_categories(f))
                   or any(quotes_overlap(q, str(f.get("claim", ""))) for q in lab.quotes)]
            if hit:
                agg.violated.append((turn_id, lab.key, hit[0].get("claim", "")))
            else:
                agg.counter_kept += 1
    return agg


# ---------------------------------------------------------------------------
# Baseline summary
# ---------------------------------------------------------------------------

def category_mix(category_sets) -> Counter:
    mix = Counter()
    for cats in category_sets:
        rec, clm = bool(cats & RECORD_DECIDABLE), bool(cats & CLAIM_LEVEL)
        mix["both" if rec and clm else "record_only" if rec else "claim_only" if clm else "other"] += 1
    return mix


def summarize(judgements: dict[str, dict]) -> dict:
    n = len(judgements)
    raw = [f for j in judgements.values() for f in j.get("findings") or [] if isinstance(f, dict)]
    findings = [f for f in raw if _finding_categories(f)]
    by_cat = Counter(c for f in findings for c in _finding_categories(f))
    return {
        "turns": n,
        "findings": len(findings),
        "findings_per_turn": len(findings) / n if n else None,
        "uncategorised_dropped": len(raw) - len(findings),
        "turns_with_finding": sum(
            1 for j in judgements.values()
            if any(isinstance(f, dict) and _finding_categories(f) for f in j.get("findings") or [])),
        "high_severity": sum(1 for f in findings if str(f.get("severity", "")).lower() == "high"),
        "by_category": {c: by_cat.get(c, 0) for c in CATEGORIES},
        "category_mix": dict(category_mix(_finding_categories(f) for f in findings)),
        "counter_examples": sum(len(j.get("counter_examples") or []) for j in judgements.values()),
        "overcautious_turn_rate": (sum(1 for j in judgements.values() if j.get("overcautious")) / n) if n else None,
        "overcautious_flags": sum(len(j.get("overcautious") or []) for j in judgements.values()),
        "boilerplate_turn_rate": (sum(1 for j in judgements.values() if j.get("boilerplate_caveats")) / n) if n else None,
    }


def summarize_by_arm(turns: list[Turn], judgements: dict[str, dict]) -> dict[str, dict]:
    """Per-arm summaries, for a replay whose turns carry an arm (the harness's --output
    shape). Pooling across arms the way `summarize` does hides exactly the comparison a
    paired A/B run was for. The arm comes from `Turn.arm`, never from splitting the turn id:
    a label defaulted from a URL is `profile@host:port`, so the id's ':' fields are ambiguous.
    Empty when no turn has an arm (prod turns never do).
    """
    arms = sorted({t.arm for t in turns if t.arm})
    ids_by_arm = {arm: {t.id for t in turns if t.arm == arm} for arm in arms}
    return {
        arm: summarize({tid: j for tid, j in judgements.items() if tid in ids})
        for arm, ids in ids_by_arm.items()
    }


# ---------------------------------------------------------------------------
# Prod rows
# ---------------------------------------------------------------------------

_POD_PROGRAM = '''
import sqlite3, json
users = {users!r}
conn = sqlite3.connect("file:/data/chat_history.db?mode=ro", uri=True)
conn.row_factory = sqlite3.Row
where = " OR ".join("s.user_id = ? OR s.user_id LIKE ?" for _ in users)
params = [p for u in users for p in (u, u + "@%")]
rows = conn.execute(
    "SELECT m.id, m.session_id, m.role, m.content, m.created_at, m.content_json, "
    "m.tool_results_json, m.literature_backend, s.user_id FROM chat_messages m "
    "JOIN chat_sessions s ON s.id = m.session_id WHERE " + where, params)
print(json.dumps([dict(r) for r in rows], default=str))
'''


def rows_cache_path(context: str, users, cache_dir: Path) -> Path:
    tag = hashlib.sha256(json.dumps([context, sorted(users)]).encode()).hexdigest()[:12]
    return cache_dir / f"prod_rows-{tag}.json"


def pull_rows(context: str, users, cache_dir: Path, refresh: bool) -> list[dict]:
    cache = rows_cache_path(context, users, cache_dir)
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())
    r = subprocess.run(
        ["kubectl", "--context", context, "-n", "genetics", "exec", "-i", "deploy/chat-backend",
         "--", "python3", "-"],
        input=_POD_PROGRAM.format(users=tuple(users)), capture_output=True, text=True, timeout=600,
    )
    try:
        rows = json.loads(r.stdout)
    except ValueError:
        sys.exit(f"in-pod sqlite read produced no JSON (exit {r.returncode}): {r.stderr.strip()[:400]}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows))
    return rows


# ---------------------------------------------------------------------------
# Judge call
# ---------------------------------------------------------------------------

def _cap(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n[... {len(text) - limit} chars cut by the judge harness ...]"


def _format_results(results: list[dict], per_result: int, total: int | None = None) -> str:
    if not results:
        return "(none)"
    parts, used = [], 0
    for i, r in enumerate(results, 1):
        block = (f"### result {i}: {r.get('name')} input={json.dumps(r.get('input'), default=str)[:600]}\n"
                 f"{_cap(r.get('content') or '(no result persisted)', per_result)}")
        if total is not None and used + len(block) > total:
            parts.append(f"[... {len(results) - i + 1} earlier results omitted ...]")
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


def render_prompt(turn: Turn, today: str) -> str:
    # most recent earlier results first, so the cap drops the oldest
    return LITERATURE_EVIDENCE_JUDGE_PROMPT.format(
        today=today,
        question=_cap(turn.question, QUESTION_CAP),
        answer=_cap(turn.answer, ANSWER_CAP),
        literature_results=_format_results(turn.literature_results, RESULT_CAP),
        earlier_results=_format_results(list(reversed(turn.earlier_results)), EARLIER_RESULT_CAP, EARLIER_CAP),
    )


class JudgeCache:
    def __init__(self, path: Path):
        self.path = path
        self.entries: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.entries[rec["key"]] = rec

    def get(self, key: str) -> dict | None:
        return self.entries.get(key)

    def put(self, rec: dict) -> None:
        self.entries[rec["key"]] = rec
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")


async def judge_turns(turns: list[Turn], *, model: str, max_tokens: int, cache: JudgeCache,
                      cost: CostTracker, concurrency: int) -> dict[str, dict]:
    import anthropic

    client = anthropic.AsyncAnthropic()
    run_date = date.today().isoformat()
    sem = asyncio.Semaphore(concurrency)
    out: dict[str, dict] = {}

    async def one(turn: Turn):
        written = turn.created_at[:10]
        prompt = render_prompt(turn, written or run_date)
        key = hashlib.sha256(f"{model}\0{render_prompt(turn, written)}".encode()).hexdigest()
        hit = cache.get(key)
        if hit is not None:
            out[turn.id] = hit["judgement"]
            return
        async with sem:
            try:
                resp = await create_with_backoff(
                    client, model=model, max_tokens=max_tokens, **thinking_off_kwargs(model),
                    messages=[{"role": "user", "content": prompt}],
                )
            except Exception as e:
                logger.error(f"judge call failed for {turn.id}: {e}")
                return
        cost.add(model, resp.usage)
        parsed = extract_first_json(response_text(resp))
        if not isinstance(parsed, dict):
            logger.error(f"judge returned no JSON object for {turn.id} (stop={resp.stop_reason})")
            return
        cache.put({"key": key, "turn": turn.id, "model": model, "judgement": parsed,
                   "usage": {"input": resp.usage.input_tokens, "output": resp.usage.output_tokens}})
        out[turn.id] = parsed

    await asyncio.gather(*(one(t) for t in turns))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _print_summary(title: str, s: dict) -> None:
    print(f"\n== {title}")
    print(f"turns judged: {s['turns']}; findings: {s['findings']} "
          f"({s['findings_per_turn'] or 0:.2f}/turn); turns with >=1 finding: {s['turns_with_finding']}; "
          f"high severity: {s['high_severity']}")
    print("by category: " + ", ".join(f"{c}:{n}" for c, n in s["by_category"].items()))
    print(f"category mix: {s['category_mix']}")
    print(f"uncategorised findings dropped: {s['uncategorised_dropped']}")
    print(f"counter-examples: {s['counter_examples']}; overcautious turns: {_pct(s['overcautious_turn_rate'])} "
          f"({s['overcautious_flags']} flags); boilerplate-caveat turns: {_pct(s['boilerplate_turn_rate'])}")


def _in_window(turn: Turn, since: str, until: str) -> bool:
    return since <= turn.created_at[:10] < until


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--db", action="store_true", help="baseline over persisted prod turns")
    mode.add_argument("--calibrate", choices=("dev", "heldout"), help="agreement on one half of the labelled turns")
    mode.add_argument("--report", type=Path, help="judge a replay JSON (shape in the module docstring)")
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--context", default=PROD_CONTEXT)
    p.add_argument("--users", default=",".join(DEFAULT_USERS))
    p.add_argument("--since", default="2026-07-01")
    p.add_argument("--until", default="2026-10-01")
    p.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--refresh-rows", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="select turns and print counts; no API calls")
    p.add_argument("--out", type=Path, help="write per-turn judgements and the summary as JSON")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cache = JudgeCache(args.cache_dir / "judgements.jsonl")
    cost = CostTracker()
    result: dict = {}

    if args.report:
        turns = load_replay(args.report)
        print(f"{len(turns)} replay turns, {sum(t.literature_bearing for t in turns)} literature-bearing")
    else:
        rows = pull_rows(args.context, args.users.split(","), args.cache_dir, args.refresh_rows)
        all_turns = build_turns(rows)
        print(f"{len(rows)} rows, {len(all_turns)} assistant turns")
        if args.db:
            turns = [t for t in all_turns if t.persisted and t.literature_bearing
                     and _in_window(t, args.since, args.until)]
            print(f"{len(turns)} persisted literature-bearing turns in [{args.since}, {args.until})")
        else:
            labels, voided = load_labels(args.labels_dir)
            print(f"VOID labels skipped: {len(voided)} ({', '.join(lab.key for lab in voided) or 'none'})")
            resolved, unresolved = resolve_labels(labels, all_turns)
            by_id = {t.id: t for t in all_turns}
            usable = {tid: labs for tid, labs in resolved.items() if by_id[tid].persisted}
            half = {tid: labs for tid, labs in usable.items() if half_of(tid) == args.calibrate}
            turns = [by_id[tid] for tid in sorted(half)]
            n_lab = lambda d, k: sum(1 for labs in d.values() for lab in labs if lab.kind == k)  # noqa: E731
            print(f"labels: {n_lab({'x': labels}, 'F')} F / {n_lab({'x': labels}, 'G')} G; "
                  f"unresolved {len(unresolved)}; on persisted turns {n_lab(usable, 'F')} F / "
                  f"{n_lab(usable, 'G')} G over {len(usable)} turns")
            print(f"human category mix, all findings: "
                  f"{dict(category_mix(lab.categories for lab in labels if lab.kind == 'F'))}")
            print(f"{args.calibrate} half: {len(turns)} turns "
                  f"({sum(t.literature_bearing for t in turns)} literature-bearing), "
                  f"{n_lab(half, 'F')} F / {n_lab(half, 'G')} G")

    if args.dry_run:
        return
    judgements = asyncio.run(judge_turns(
        turns, model=args.model, max_tokens=args.max_tokens, cache=cache, cost=cost,
        concurrency=args.concurrency,
    ))
    summary = summarize(judgements)
    by_arm = summarize_by_arm(turns, judgements) if args.report else {}
    for arm, arm_summary in by_arm.items():
        _print_summary(f"judge summary (arm={arm})", arm_summary)
    _print_summary("judge summary (pooled)" if by_arm else "judge summary", summary)
    result = {"summary": summary, "judgements": judgements}
    if by_arm:
        result["summary_by_arm"] = by_arm

    if args.calibrate:
        agg = score_agreement(half, judgements)
        print(f"\n== agreement ({args.calibrate}, {agg.turns} turns)")
        print(f"findings: {agg.finding_hits}/{agg.finding_pairs} (turn, category) pairs = {_pct(agg.finding_recall)}")
        print(f"counter-examples not flagged: {agg.counter_kept}/{agg.counter_examples} = {_pct(agg.counter_precision)}")
        print(f"combined: {_pct(agg.combined)}")
        print(f"uncategorised findings dropped: {agg.uncategorised}")
        for tid, cat in agg.missed:
            labs = [lab.key for lab in half[tid] if cat in lab.categories]
            print(f"MISSED turn={tid} cat={cat} labels={labs}")
        for tid, key, claim in agg.violated:
            print(f"FLAGGED-GOOD turn={tid} {key}: {str(claim)[:200]!r}")
        result["agreement"] = {**asdict(agg), "finding_recall": agg.finding_recall,
                               "counter_precision": agg.counter_precision, "combined": agg.combined}

    if cost.usage:
        print("\n== API spend (this run, cache misses only)")
        print("\n".join(cost.summary_lines()))
        result["cost"] = {"usage": cost.usage, "usd": cost.total_cost()}
    if args.out:
        args.out.write_text(json.dumps(result, indent=1, default=str))


if __name__ == "__main__":
    main()
