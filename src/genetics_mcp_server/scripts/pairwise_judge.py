"""Blind, order-randomised pairwise LLM judging of the replay benchmark's matched pairs.

`replay_benchmark.py` measures cost and latency. This module answers the other half of
`genetics-results-suite-4h6.23`'s kill criterion — "must not REGRESS quality" — over the
SAME matched pairs the benchmark already computes (`replay_benchmark.matched_pairs`: the
`(case_id, turn_index)` keys that came back `ok` on BOTH arms).

WHY PAIRED AND NOT `analyze_conversations`' RUBRIC. That instrument scores ONE conversation
absolutely (topic, quality_score, issues) and was built for sampling and for tracking
quality over time. Scoring each arm absolutely and comparing means is a weak test here: the
rubric is coarse, the expected between-arm difference is small, and per-question difficulty
dominates the score. Judging the two answers to the same question side by side cancels that
difficulty, which is why A/B evaluations are done pairwise. The absolute rubric is still
worth having later — it is the only thing comparable with historical production numbers —
but it is a different question and it is not what the kill criterion asks.

The design is built around not fooling ourselves:

* BLIND. The judge is shown "Answer 1" and "Answer 2" and never an arm, a tool profile, a
  model or a mechanism. It is shown FINAL ANSWERS ONLY — see `final_answer_text`.
* JUDGED BOTH WAYS. Position bias in pairwise LLM judging is large, so every pair is
  judged twice with the answers swapped. A pair counts as a win only when BOTH passes name
  the same answer; anything else is a tie. That doubles judge cost and is the cheapest
  defence there is.
* ORDER RECORDED AND SEEDED. Which arm is shown first in pass 1 is derived from a SHA-256
  of `case_id|turn_index` — stable across processes (`hash()` is salted per process) and
  varying across pairs, so the same report judges the same way twice.
* TIES ARE FIRST-CLASS. Forcing a winner on two equally good answers manufactures signal.
* THE LOSS TAIL IS THE POINT. Wins/losses/ties are reported with counts, an exact sign
  test, and per-pair detail with the judge's own reason, so a human can read every loss.
* SPEND IS PRICED BEFORE IT HAPPENS and reported as its own line item, never folded into
  the benchmark's USD.
* THE HARNESS'S OWN DISTORTIONS ARE MEASURED PER ARM, not asserted to be even-handed. The
  answer-slicing rule, middle-elision and empty answers are each applied identically to
  both arms and each can still FIRE unequally, so the report prints per-arm dropped-,
  shown- and elided-character figures, and gives pairs where an answer could not be
  extracted the same restricted-table treatment provenance gets. A pair the harness broke
  must not read as a pair one arm lost.

Judging is OFF unless asked for: the benchmark produces cost and latency with no judge
calls at all, and a saved report can be judged later without re-running anything
(`python -m genetics_mcp_server.scripts.pairwise_judge --report report.json`).
"""

import argparse
import asyncio
import hashlib
import json
import logging
import math
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from genetics_mcp_server.cost import estimate_cost, has_pricing
from genetics_mcp_server.scripts.replay_benchmark import matched_pairs

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_MODEL = "claude-opus-5"

# the judge answers with a small JSON object; this is a ceiling, not an expectation, and the
# pre-flight estimate prices it as if every call hit it
JUDGE_MAX_TOKENS = 700

# both answers are elided in the MIDDLE when longer, so the opening and the conclusion — the
# two parts a reader of a genetics answer weighs most — both survive. Applied identically to
# both arms and flagged per pair, because a truncated answer is judged incompletely even
# when it is truncated fairly.
JUDGE_MAX_ANSWER_CHARS = 12_000
ELISION_MARKER = "\n\n[... middle of this answer elided for length ...]\n\n"

# preceding USER turns shown as context for follow-up questions ("what about the second
# one?"). Deliberately the user turns ONLY: they come from the recorded dataset and are
# identical for both arms, whereas each arm's own prior ASSISTANT turns differ and would
# leak arm identity straight into the context window.
JUDGE_MAX_PRIOR_TURNS = 4
JUDGE_MAX_PRIOR_CHARS = 1_500

# rough tokens-per-character for the PRE-FLIGHT estimate only. Never used to report spend:
# the post-run figure is priced from the API's own usage counts.
ESTIMATE_CHARS_PER_TOKEN = 4
# used only by `--dry-run`, which runs before any answer exists. Labelled nominal wherever
# it is printed, because it is a guess about text that has not been generated yet.
ESTIMATE_NOMINAL_ANSWER_CHARS = 3_000

PASSES_PER_PAIR = 2

# outcomes, disjoint and exhaustive over judged pairs
OUTCOME_WIN = "win"  # both passes named the same arm
OUTCOME_TIE_AGREED = "tie_agreed"  # both passes said tie
OUTCOME_TIE_UNSTABLE = "tie_unstable"  # one pass named a winner, the other said tie
OUTCOME_TIE_POSITION_FLIP = "tie_position_flip"  # the passes named DIFFERENT arms
OUTCOME_UNRESOLVED = "unresolved"  # a judge call failed or came back unparseable

TIE_OUTCOMES = (OUTCOME_TIE_AGREED, OUTCOME_TIE_UNSTABLE, OUTCOME_TIE_POSITION_FLIP)

MARGIN_RANK = {"none": 0, "slight": 1, "clear": 2}

# the smallest number of decisive (non-tie) pairs at which a two-sided sign test can reach
# p <= 0.05 AT ALL: below it even a clean sweep is indistinguishable from a coin, so a win
# rate over that many pairs must not be printed as if it were a solid number. Same reasoning
# as `replay_benchmark.MIN_N_FOR_PERCENTILE`, which refuses to quote a p95 that is really
# the sample maximum.
SIGN_TEST_ALPHA = 0.05
MIN_DECISIVE_PAIRS = next(n for n in range(1, 64) if 2 * 0.5**n <= SIGN_TEST_ALPHA)

# Text that identifies HOW an answer was produced, which is the same thing as identifying
# its arm. These are NOT scrubbed from the text the judge sees: rewriting an answer to hide
# its provenance changes what is being judged, and a regex that edits model prose will
# eventually delete something load-bearing. They are detected, counted, and the whole
# win/loss/tie table is reported a second time over the pairs where NEITHER answer carried
# one — so a reader can see whether the verdict survives dropping every pair whose arm was
# guessable. Cheap and honest, where scrubbing would be neither.
#
# THE LIST IS NOT SYMMETRIC AND CANNOT BE. Most of these (sandbox, python fence, script and
# SDK self-reference, artifact) are code-arm tells; only the tool and SQL self-references
# point the other way, and no phrase reliably marks "this answer came from many small tool
# calls" because that arm's answers read like ordinary prose. So the restricted subset drops
# code-arm-flavoured pairs preferentially, and a large gap between the full and restricted
# tables is evidence about the JUDGE, not about the arms. Read the two together. That
# asymmetry is printed with the numbers (see `format_judging`), not confessed only here:
# a caveat a reader has to open the source to find is a caveat that does not exist.
PROVENANCE_MARKERS: tuple[tuple[str, str], ...] = (
    ("run_analysis", r"run_analysis"),
    ("sandbox", r"\bsandbox(?:ed|es)?\b"),
    ("python_fence", r"```\s*(?:python|py)\b"),
    ("sql_fence", r"```\s*sql\b"),
    ("script_self_reference", r"\b(?:wrote|ran|run|running|ex(?:ecut|ecuted|ecuting))\w*\s+"
                              r"(?:a|the|this|my|some)?\s*(?:python\s+|analysis\s+)?"
                              r"(?:script|code|snippet)\b"),
    ("tool_self_reference", r"\[Using tool:|\bI (?:called|used|invoked) the\b[^.\n]{0,40}\btool\b"),
    ("sdk_self_reference", r"\bgenetics[_ ]sdk\b|\bthe SDK\b"),
    # deliberately NOT a bare `\bartifacts?\b`: in genetics prose "artifact" almost always
    # means a spurious signal ("a batch artifact of imputation", "an artifact of population
    # structure"), which fires on BOTH arms and shrinks the clean control subset for a
    # reason that has nothing to do with provenance
    ("artifact_reference", r"\bartifacts?\s+(?:file|files|directory|dir|path)\b"
                           r"|\bartifacts?\s+from\s+the\s+(?:script|run|analysis)\b"),
    # the iteration-cap notice (`llm_service.MAX_ITERATIONS_NOTICE`) survives
    # `final_answer_text` verbatim and is a near-perfect tell for the many-roundtrip arm.
    # Nothing is scrubbed here either — it is listed so the blinding audit stops reporting
    # zero identifiable pairs on a run where the arm was written on the answer.
    ("max_iterations_notice", r"Max tool iterations reached"),
)
_COMPILED_MARKERS = tuple((name, re.compile(rx, re.IGNORECASE)) for name, rx in PROVENANCE_MARKERS)


@dataclass
class JudgePair:
    """One matched (case, turn) with both arms' final answers. Arm-labelled HERE only."""

    case_id: str
    turn_index: int
    question: str
    prior_questions: list[str]
    answers: dict[str, str]  # arm -> final answer text
    # arm -> characters of text the answer-slicing rule discarded (see `final_answer_split`).
    # Carried so the report can show whether the rule cost the two arms the same amount.
    dropped_chars: dict[str, int] = field(default_factory=dict)


@dataclass
class PassResult:
    """One judging call: the answers in a fixed order, and what came back."""

    order: list[str]  # arms, in the order this pass presented them
    verdict: str | None  # arm name, "tie", or None when the call failed
    position_chosen: str | None  # "1", "2", "tie", or None — the raw positional answer
    margin: str | None
    # exactly what the judge wrote in `margin`, before it was normalised into MARGIN_RANK.
    # An unrecognised word normalises to "none", and without this the report would show a
    # win with clear=0, slight=0 and no explanation of where the strength went.
    margin_raw: str = ""
    reason: str = ""
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class PairVerdict:
    case_id: str
    turn_index: int
    outcome: str
    winner: str | None
    loser: str | None
    margin: str | None  # the weaker of the two passes' margins, when they agreed
    passes: list[PassResult] = field(default_factory=list)
    answer_chars: dict[str, int] = field(default_factory=dict)  # RAW, before elision
    # what the judge actually read, after middle-elision. Two answers of 40,000 and 20,000
    # characters are both 12,000 in the prompt, so any length diagnostic computed on
    # `answer_chars` describes a difference the judge was never shown.
    shown_answer_chars: dict[str, int] = field(default_factory=dict)
    # characters the answer-slicing rule discarded per arm; see `final_answer_split`
    dropped_chars: dict[str, int] = field(default_factory=dict)
    provenance_markers: dict[str, list[str]] = field(default_factory=dict)
    arm_identifiable: bool = False
    truncated_arms: list[str] = field(default_factory=list)
    # an arm whose final answer was empty is shown "(this answer was empty)", which reads as
    # a loss. That is right when the turn really answered nothing and WRONG if the harness
    # failed to extract the text, so it is counted and printed instead of being absorbed:
    # emptiness concentrated on one arm is a bug report, not a quality finding.
    empty_answer_arms: list[str] = field(default_factory=list)


def final_answer_split(message_content: list[dict[str, Any]] | None) -> tuple[str, int]:
    """(the turn's FINAL answer, how many characters of earlier text were discarded).

    WHAT THE JUDGE IS SHOWN, AND WHY IT IS ONLY THIS. `message_content` accumulates every
    block of every iteration — the model's running commentary, its `tool_use` blocks with
    their inputs, and finally the answer. Showing any of that to the judge would hand it the
    arm identity outright: a `run_analysis` call with a Python payload can only be the code
    arm, and a screenful of `get_*` calls can only be the all-tools arm. Blinding the judge
    while showing it the tool trace would be theatre.

    The cost of the choice, stated rather than hidden: the judge cannot see that one arm
    reached its answer through six redundant calls and the other through one. That is
    deliberate — efficiency is what the benchmark's OWN metrics measure exactly (iterations,
    tool calls, tokens, USD), and asking a blinded judge to re-estimate it from prose would
    be a worse instrument for a question already answered better elsewhere. The judge is
    asked only about the thing the metrics cannot see: whether the user got a worse answer.

    THE RULE IS NOT NEUTRAL BETWEEN THE ARMS, AND THAT IS WHY THE SECOND RETURN VALUE
    EXISTS. Intermediate commentary is usually provenance ("let me query BigQuery for that")
    and dropping it is right. But text before the last `tool_use` is not always commentary:
    a turn that lays out a table, calls one more tool and closes with "In summary, yes."
    loses the table, and a turn whose last block IS the `tool_use` loses everything. An arm
    that makes ONE EARLY tool call keeps nearly all its prose; an arm whose LAST call is
    LATE loses whatever it wrote between calls. That is a systematic, arm-dependent
    handicap, so the count of discarded characters is returned, recorded per turn, and its
    per-arm median is printed beside the length diagnostic: if the two arms' medians differ
    materially, the verdict is partly measuring this rule rather than answer quality.
    """
    if not message_content:
        return "", 0
    last_tool_use = -1
    for index, block in enumerate(message_content):
        if isinstance(block, dict) and block.get("type") == "tool_use":
            last_tool_use = index

    def _text(blocks: list[dict[str, Any]]) -> str:
        return "".join(
            block.get("text") or ""
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        )

    kept = _text(message_content[last_tool_use + 1 :]).strip()
    dropped = _text(message_content[: last_tool_use + 1]).strip()
    return kept, len(dropped)


def final_answer_text(message_content: list[dict[str, Any]] | None) -> str:
    """The turn's FINAL answer — see `final_answer_split`, whose first element this is."""
    return final_answer_split(message_content)[0]


def user_question_text(content: Any) -> str:
    """The user turn's text, whether it was recorded as a string or as content blocks."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(
            block.get("text") or ""
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ).strip()
    return ""


def presentation_order(case_id: str, turn_index: int, arms: tuple[str, str]) -> list[str]:
    """Which arm pass 1 shows first, seeded from the pair's identity.

    Seeded from `case_id|turn_index` through SHA-256 rather than `hash()`, whose per-process
    salt would make the same report judge differently on every run, and rather than a global
    RNG, whose draw depends on how many pairs happened to precede this one.

    With both-ways judging the ORDER cannot change a verdict — every pair is seen in both
    orders — so this is not the primary defence against position bias; the second pass is.
    It is still load-bearing twice over: it fixes which order is attempted first, so a pair
    whose second pass fails is not silently resolved from an arbitrarily chosen position,
    and it is recorded per pair so the per-pass verdicts can be read against position (see
    `OUTCOME_TIE_POSITION_FLIP`, which counts the pairs where position alone decided).
    """
    digest = hashlib.sha256(f"{case_id}|{turn_index}".encode()).digest()
    return list(arms) if digest[0] % 2 == 0 else list(reversed(arms))


def scan_provenance_markers(text: str) -> list[str]:
    """Names of the provenance markers present in `text` — see `PROVENANCE_MARKERS`."""
    return [name for name, pattern in _COMPILED_MARKERS if pattern.search(text)]


def elide_middle(text: str, limit: int = JUDGE_MAX_ANSWER_CHARS) -> str:
    """Keep the head and the tail, drop the middle, and say so where it was dropped."""
    if len(text) <= limit:
        return text
    keep = (limit - len(ELISION_MARKER)) // 2
    return text[:keep] + ELISION_MARKER + text[-keep:]


class _TurnView:
    """Adapts a serialised turn dict to the three attributes `matched_pairs` reads.

    A shim rather than a re-implementation on purpose: `matched_pairs` stays the single
    definition of what a comparable pair is, and a change to that rule reaches this path
    without anyone remembering to mirror it.
    """

    __slots__ = ("arm", "case_id", "status", "turn_index")

    def __init__(self, raw: dict[str, Any]):
        self.case_id = str(raw.get("case_id"))
        self.turn_index = int(raw.get("turn_index", -1))
        self.arm = str(raw.get("arm"))
        self.status = str(raw.get("status"))


def build_pairs(report: dict[str, Any]) -> tuple[list[JudgePair], tuple[str, str]]:
    """The judge's input: the harness's OWN matched pairs, from a benchmark report.

    Deliberately re-uses `replay_benchmark.matched_pairs` rather than re-deriving the set:
    the pairing rule (ok on BOTH arms, so an arm is not rewarded for failing on the hard
    turns) is one decision and belongs in one place.
    """
    arms = tuple(report.get("arms") or ())
    if len(arms) != 2:
        raise ValueError(f"expected exactly 2 arms in the report, found {list(arms)}")
    turns = [_TurnView(t) for t in report.get("turns") or []]
    matched, _dropped = matched_pairs(turns, arms)  # type: ignore[arg-type]

    by_key: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for raw in report.get("turns") or []:
        key = (str(raw.get("case_id")), int(raw.get("turn_index", -1)))
        by_key.setdefault(key, {})[str(raw.get("arm"))] = raw

    pairs: list[JudgePair] = []
    for key in sorted(matched):
        per_arm = by_key.get(key, {})
        if set(arms) - set(per_arm):
            continue
        question = str(per_arm[arms[0]].get("user_question") or "")
        priors: list[str] = []
        for (case_id, turn_index), arm_map in sorted(by_key.items()):
            if case_id != key[0] or turn_index >= key[1]:
                continue
            earlier = arm_map.get(arms[0])
            if earlier is not None and earlier.get("user_question"):
                priors.append(str(earlier["user_question"]))
        pairs.append(
            JudgePair(
                case_id=key[0],
                turn_index=key[1],
                question=question,
                prior_questions=priors[-JUDGE_MAX_PRIOR_TURNS:],
                answers={arm: str(per_arm[arm].get("final_answer") or "") for arm in arms},
                dropped_chars={
                    arm: _as_int(per_arm[arm].get("final_answer_dropped_chars")) for arm in arms
                },
            )
        )
    return pairs, arms  # type: ignore[return-value]


def build_prompt(pair: JudgePair, order: list[str], today: str | None = None) -> str:
    """Render the judging prompt with the two answers in `order`.

    Nothing arm-derived reaches the text: the arm names index `pair.answers` and are not
    interpolated. `test_the_arm_name_never_reaches_the_prompt` mutates an arm name to a
    conspicuous string and asserts it is absent from the rendered prompt.
    """
    from .conversation_prompts import PAIRWISE_ANSWER_JUDGE_PROMPT

    prior = ""
    if pair.prior_questions:
        joined = "\n".join(f"- {q[:JUDGE_MAX_PRIOR_CHARS]}" for q in pair.prior_questions)
        # NOT "both answers had the same conversation history": only the USER turns were
        # identical. Each arm's own prior ASSISTANT turns differ and are withheld (they
        # would leak arm identity). Telling the judge otherwise would be an instruction it
        # could act on — it might read a follow-up as under-specified for one answer.
        prior = (
            "\nEARLIER USER TURNS IN THIS CONVERSATION (context for the question above; "
            "these user turns were identical for both answers, and each assistant's own "
            "earlier replies are not shown to you):\n" + joined + "\n"
        )
    first, second = order
    return PAIRWISE_ANSWER_JUDGE_PROMPT.format(
        today=today or date.today().isoformat(),
        question=pair.question or "(the recorded question text is unavailable)",
        prior_context=prior,
        # `.strip()`, not truthiness: a whitespace-only answer is an empty answer, and
        # showing the judge a blank section would make it guess at what it was not shown
        answer_1=elide_middle(pair.answers[first].strip()) or "(this answer was empty)",
        answer_2=elide_middle(pair.answers[second].strip()) or "(this answer was empty)",
    )


def _parse_verdict(text: str) -> tuple[str, str, str, str] | None:
    """(position, margin, margin_raw, reason) from the judge's JSON, or None if unusable."""
    from .analyze_conversations import extract_first_json

    parsed = extract_first_json(text)
    if not isinstance(parsed, dict):
        return None
    position = str(parsed.get("verdict", "")).strip().lower()
    if position not in ("1", "2", "tie"):
        return None
    raw = str(parsed.get("margin", "")).strip().lower()[:40]
    # an unrecognised strength word is not evidence of strength — but it is not nothing
    # either, so the word itself is kept and surfaced rather than quietly becoming "none"
    margin = raw if raw in MARGIN_RANK else "none"
    if position == "tie":
        # a "tie" with a non-none margin is a self-contradiction; the verdict is what counts
        margin = "none"
    return position, margin, raw, str(parsed.get("reason", ""))[:500]


async def _judge_once(client: Any, model: str, pair: JudgePair, order: list[str]) -> PassResult:
    from .analyze_conversations import response_text, thinking_off_kwargs

    result = PassResult(order=list(order), verdict=None, position_chosen=None, margin=None)
    try:
        response = await client.messages.create(
            model=model,
            max_tokens=JUDGE_MAX_TOKENS,
            **thinking_off_kwargs(model),
            messages=[{"role": "user", "content": build_prompt(pair, order)}],
        )
    except Exception as e:  # one failed call must not abort the pass over the rest
        result.error = f"{type(e).__name__}: {e}"[:300]
        return result
    usage = getattr(response, "usage", None)
    # cached components are folded into the input count and priced at the full input rate.
    # These prompts carry no cache_control and each is unique, so both are expected to be 0;
    # if they ever are not, over-stating what judging cost is the safe direction for a figure
    # whose only job is to warn about spend.
    result.input_tokens = _as_int(getattr(usage, "input_tokens", 0)) + _as_int(
        getattr(usage, "cache_read_input_tokens", 0)
    ) + _as_int(getattr(usage, "cache_creation_input_tokens", 0))
    result.output_tokens = _as_int(getattr(usage, "output_tokens", 0))
    parsed = _parse_verdict(response_text(response))
    if parsed is None:
        result.error = "the judge's reply carried no usable {verdict, margin, reason} JSON"
        return result
    position, margin, margin_raw, reason = parsed
    result.position_chosen = position
    result.margin = margin
    result.margin_raw = margin_raw
    result.reason = reason
    result.verdict = "tie" if position == "tie" else order[int(position) - 1]
    return result


def _as_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _combine(passes: list[PassResult]) -> tuple[str, str | None, str | None]:
    """(outcome, winner, margin) from the two passes.

    A pair is a WIN only when both orders named the same arm. Everything else that is not a
    failure is a TIE, split by how it arose so the reader can tell an agreed tie from a
    verdict that only survived one presentation order:
      * both said tie                  -> tie_agreed
      * one named a winner, one tied   -> tie_unstable
      * they named DIFFERENT arms      -> tie_position_flip, i.e. position alone decided
    Resolving either of the last two in favour of the arm that "won once" is exactly how a
    position-biased judge manufactures a result.
    """
    if any(p.verdict is None for p in passes) or len(passes) < PASSES_PER_PAIR:
        return OUTCOME_UNRESOLVED, None, None
    first, second = passes[0].verdict, passes[1].verdict
    if first == "tie" and second == "tie":
        return OUTCOME_TIE_AGREED, None, None
    if first == "tie" or second == "tie":
        return OUTCOME_TIE_UNSTABLE, None, None
    if first != second:
        return OUTCOME_TIE_POSITION_FLIP, None, None
    # the WEAKER of the two margins, so a "clear" from one order cannot be quoted for a pair
    # the other order thought marginal. Ranked rather than tested for "slight": a model that
    # answers {"verdict": "1", "margin": "none"} would otherwise be read as "clear".
    weakest = min((p.margin or "none" for p in passes), key=MARGIN_RANK.__getitem__)
    return OUTCOME_WIN, first, weakest


async def judge_pairs(
    pairs: list[JudgePair],
    arms: tuple[str, str],
    model: str = DEFAULT_JUDGE_MODEL,
    concurrency: int = 4,
    client: Any = None,
) -> list[PairVerdict]:
    """Judge every pair twice, with the answers swapped between the two passes."""
    if client is None:
        import anthropic

        client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def one(pair: JudgePair) -> PairVerdict:
        order = presentation_order(pair.case_id, pair.turn_index, arms)
        async with semaphore:
            first = await _judge_once(client, model, pair, order)
            # a failed first pass makes the pair unresolved whatever the second says, so
            # the second call is not worth its money
            passes = [first]
            if first.verdict is not None:
                passes.append(await _judge_once(client, model, pair, list(reversed(order))))
        outcome, winner, margin = _combine(passes)
        markers = {arm: scan_provenance_markers(text) for arm, text in pair.answers.items()}
        return PairVerdict(
            case_id=pair.case_id,
            turn_index=pair.turn_index,
            outcome=outcome,
            winner=winner,
            loser=next((a for a in arms if a != winner), None) if winner else None,
            margin=margin,
            passes=passes,
            answer_chars={arm: len(text) for arm, text in pair.answers.items()},
            # the same transformation `build_prompt` applies, so this is literally what the
            # judge read — not the raw text it never saw
            shown_answer_chars={
                arm: len(elide_middle(text.strip())) for arm, text in pair.answers.items()
            },
            dropped_chars={arm: pair.dropped_chars.get(arm, 0) for arm in arms},
            provenance_markers={arm: found for arm, found in markers.items() if found},
            arm_identifiable=any(markers.values()),
            truncated_arms=[
                arm for arm, text in pair.answers.items() if len(text) > JUDGE_MAX_ANSWER_CHARS
            ],
            empty_answer_arms=[
                arm for arm, text in pair.answers.items() if not text.strip()
            ],
        )

    results = await asyncio.gather(*(one(p) for p in pairs))
    return list(results)


def sign_test_p(wins: int, losses: int) -> float | None:
    """Exact two-sided sign-test p-value for `wins` vs `losses`, ties excluded.

    Ties are dropped rather than split, which is the standard sign test: a tie is evidence
    of no difference on that pair, not half a win for each arm, and splitting them would
    shrink the p-value by inventing observations.
    """
    n = wins + losses
    if n == 0:
        return None
    extreme = max(wins, losses)
    tail = sum(math.comb(n, k) for k in range(extreme, n + 1)) / 2**n
    return min(1.0, 2 * tail)


def summarize_verdicts(
    verdicts: list[PairVerdict], arms: tuple[str, str], model: str
) -> dict[str, Any]:
    """Wins / losses / ties with counts, the sign test, and the bias diagnostics."""
    judged = [v for v in verdicts if v.outcome != OUTCOME_UNRESOLVED]
    wins = {arm: sum(1 for v in judged if v.winner == arm) for arm in arms}
    ties = {name: sum(1 for v in judged if v.outcome == name) for name in TIE_OUTCOMES}
    decisive = sum(wins.values())
    longer_won = sum(
        1
        for v in judged
        if v.winner and v.shown_answer_chars.get(v.winner, 0) > max(
            v.shown_answer_chars.get(a, 0) for a in arms if a != v.winner
        )
    )
    identifiable = [v for v in judged if v.arm_identifiable]
    clean = [v for v in judged if not v.arm_identifiable]
    nonempty = [v for v in judged if not v.empty_answer_arms]
    return {
        "model": model,
        "arms": list(arms),
        # the per-pair outcomes behind every aggregate below. Persisted because the
        # aggregates answer "which arm won" and cannot answer "on WHICH questions", which is
        # the actionable half: an arm that loses four of twenty is a different problem from
        # one that loses uniformly, and only this list distinguishes them. Kept deliberately
        # narrow — the judge's prose `reason` and the raw passes stay out, since this rides
        # inside every saved report.
        "pairs": [
            {
                "case_id": v.case_id,
                "turn_index": v.turn_index,
                "outcome": v.outcome,
                "winner": v.winner,
                "margin": v.margin,
                # a verdict the blinding failed on is still reported, flagged rather than
                # dropped, so a scorecard cannot present it as clean
                "arm_identifiable": v.arm_identifiable,
            }
            for v in verdicts
        ],
        "pairs_offered": len(verdicts),
        "pairs_judged": len(judged),
        "pairs_unresolved": len(verdicts) - len(judged),
        "judge_calls": sum(len(v.passes) for v in verdicts),
        "wins": wins,
        "ties_total": sum(ties.values()),
        "ties_by_kind": ties,
        "decisive_pairs": decisive,
        # the fraction of DECISIVE pairs won by each arm. Not "of all pairs": a rate whose
        # denominator includes ties moves when the judge's tie threshold moves.
        "win_share_of_decisive": {
            arm: (wins[arm] / decisive) if decisive else None for arm in arms
        },
        "sign_test_p": sign_test_p(wins[arms[0]], wins[arms[1]]),
        "sign_test_alpha": SIGN_TEST_ALPHA,
        "min_decisive_pairs_for_significance": MIN_DECISIVE_PAIRS,
        "underpowered": decisive < MIN_DECISIVE_PAIRS,
        # "none" is listed because a win CAN carry it — the weaker of the two passes' margins
        # is quoted, and an unrecognised strength word normalises to it. Without the column
        # the three numbers on the printed line would not sum to the win count.
        "margins": {
            arm: {
                m: sum(1 for v in judged if v.winner == arm and v.margin == m)
                for m in ("clear", "slight", "none")
            }
            for arm in arms
        },
        "unrecognised_margin_words": sorted(
            {
                p.margin_raw
                for v in verdicts
                for p in v.passes
                if p.margin_raw and p.margin_raw not in MARGIN_RANK
            }
        ),
        # length is the classic pairwise-judge confound and the arms have no reason to write
        # answers of the same length, so it is measured instead of assumed away — on the
        # SHOWN text, because a diagnostic whose job is to warn "your judge is rewarding
        # length" must not report a length difference elision had already removed
        "length_bias": {
            "measured_on": "answer text AS SHOWN to the judge (after middle-elision)",
            "decisive_pairs_won_by_the_longer_shown_answer": longer_won,
            "of_decisive_pairs": decisive,
            "median_shown_answer_chars": {
                arm: _median([v.shown_answer_chars.get(arm, 0) for v in judged]) for arm in arms
            },
            "median_raw_answer_chars": {
                arm: _median([v.answer_chars.get(arm, 0) for v in judged]) for arm in arms
            },
        },
        # what `final_answer_split` discarded before the judge saw anything. An arm whose
        # last tool call comes LATE loses more here, and that is a handicap the win/loss
        # table cannot distinguish from worse answers — see `final_answer_split`.
        "answer_slicing": {
            "median_dropped_pre_answer_chars": {
                arm: _median([v.dropped_chars.get(arm, 0) for v in judged]) for arm in arms
            },
            "total_dropped_pre_answer_chars": {
                arm: sum(v.dropped_chars.get(arm, 0) for v in judged) for arm in arms
            },
        },
        "blinding": {
            "pairs_with_provenance_markers": len(identifiable),
            # per arm, not a pooled total: the whole argument is about ONE-SIDEDNESS, and a
            # pooled `(sandbox=7)` cannot say whether all seven were the same arm's
            "marker_counts_per_arm": _marker_counts(judged, arms),
            "marker_list_is_asymmetric": (
                "most markers are code-arm tells, so this subset drops code-arm-flavoured "
                "pairs preferentially; a gap between the full and restricted tables is "
                "evidence about the JUDGE, not about the arms"
            ),
            # the same table over the pairs where NEITHER answer named its own machinery.
            # If the verdict only exists in the identifiable subset, the judge may have been
            # reading provenance rather than quality.
            "restricted_to_unidentifiable": _subset_table(clean, arms),
        },
        "truncated_answers": sum(1 for v in verdicts if v.truncated_arms),  # PAIRS, not arms
        # per arm, because "6 pairs were elided" reads as symmetric information loss when it
        # can mean one arm was judged on 30% of its answer and the other on all of it
        "truncated_answers_per_arm": {
            arm: sum(1 for v in verdicts if arm in v.truncated_arms) for arm in arms
        },
        "empty_answers_per_arm": {
            arm: sum(1 for v in verdicts if arm in v.empty_answer_arms) for arm in arms
        },
        # an empty answer whose turn DID produce text before its last tool call was emptied
        # by the slicing rule, not by the model. Without this split, `empty_answers_per_arm`
        # cannot tell a silent model from a harness that threw the answer away.
        "empty_answers_with_dropped_text_per_arm": {
            arm: sum(
                1
                for v in verdicts
                if arm in v.empty_answer_arms and v.dropped_chars.get(arm, 0) > 0
            )
            for arm in arms
        },
        "pairs_with_an_empty_answer": sum(1 for v in judged if v.empty_answer_arms),
        # an answer that could not be extracted was NOT MEASURED, so scoring it as a loss
        # measures the harness. Reported the way provenance is: the whole table again over
        # the pairs where both arms actually produced text.
        "restricted_to_pairs_with_both_answers": _subset_table(nonempty, arms),
        "pairs": [asdict(v) for v in verdicts],
    }


def _subset_table(subset: list[PairVerdict], arms: tuple[str, str]) -> dict[str, Any]:
    """The win/loss/tie table and sign test over a subset of judged pairs.

    Carries `underpowered` and the threshold with it, so a p-value the printed report
    refuses to quote is not left sitting quotable in the saved JSON — the report and the
    JSON must not disagree about whether a number means anything.
    """
    subset_wins = {arm: sum(1 for v in subset if v.winner == arm) for arm in arms}
    decisive = sum(subset_wins.values())
    return {
        "pairs_judged": len(subset),
        "wins": subset_wins,
        "ties_total": sum(1 for v in subset if v.outcome in TIE_OUTCOMES),
        "decisive_pairs": decisive,
        "sign_test_p": sign_test_p(subset_wins[arms[0]], subset_wins[arms[1]]),
        "underpowered": decisive < MIN_DECISIVE_PAIRS,
        "min_decisive_pairs_for_significance": MIN_DECISIVE_PAIRS,
    }


def _p_clause(table: dict[str, Any]) -> str:
    """A restricted table's sign test, rendered only when it can mean anything.

    The same power rule the headline obeys. A restricted table is where a reader goes when
    the headline looks too good, so it is the last place a p-value should be quotable at
    an n at which no outcome reaches alpha.
    """
    if table["sign_test_p"] is None or table["underpowered"]:
        return (
            f"; no sign test: {table['decisive_pairs']} decisive pair(s), below "
            f"{table['min_decisive_pairs_for_significance']}"
        )
    return f"; sign test p={table['sign_test_p']:.3f}"


def _median(values: list[int]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2


def _marker_counts(verdicts: list[PairVerdict], arms: tuple[str, str]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {arm: {} for arm in arms}
    for v in verdicts:
        for arm, found in v.provenance_markers.items():
            per_arm = counts.setdefault(arm, {})
            for name in found:
                per_arm[name] = per_arm.get(name, 0) + 1
    return {arm: dict(sorted(found.items())) for arm, found in counts.items()}


def estimate_judging_cost(
    pairs: list[JudgePair] | None,
    model: str,
    pair_count: int | None = None,
) -> dict[str, Any]:
    """Price the judging pass BEFORE it spends, the way `--dry-run` prices the benchmark.

    With `pairs` the input side is exact text (the prompts that will actually be sent) run
    through a chars-per-token approximation; with only `pair_count` — the `--dry-run` case,
    where no answer has been generated yet — the answers are assumed to be
    `ESTIMATE_NOMINAL_ANSWER_CHARS` and the result is labelled nominal. Output is priced at
    the `max_tokens` ceiling, so the USD figure is an UPPER bound on output cost and is
    named as one. A model with no pricing entry reports `usd: None`, never a guess: the
    harness does not hand out confident wrong numbers (`cost.has_pricing`).
    """
    if pairs is not None:
        # the order does not change the prompt's LENGTH, so it is priced in whichever order
        # the pair happens to hold; the two passes are then counted below
        prompt_chars = sum(len(build_prompt(p, list(p.answers))) for p in pairs)
        n = len(pairs)
        nominal = False
    else:
        n = pair_count or 0
        prompt_chars = n * (2 * ESTIMATE_NOMINAL_ANSWER_CHARS + 3_000)
        nominal = True
    calls = n * PASSES_PER_PAIR
    input_tokens = math.ceil(prompt_chars / ESTIMATE_CHARS_PER_TOKEN) * PASSES_PER_PAIR
    output_tokens = calls * JUDGE_MAX_TOKENS
    usd = (
        estimate_cost(model, input_tokens, output_tokens) if model and has_pricing(model) else None
    )
    return {
        "model": model,
        "pairs": n,
        "passes_per_pair": PASSES_PER_PAIR,
        "calls": calls,
        "input_tokens": input_tokens,
        "output_tokens_upper_bound": output_tokens,
        "usd_upper_bound": usd,
        "nominal": nominal,
    }


def estimate_lines(estimate: dict[str, Any]) -> list[str]:
    usd = estimate["usd_upper_bound"]
    price = (
        "NOT PRICED (no pricing entry for this model; the estimate is not guessed)"
        if usd is None
        else f"<= ${usd:,.2f}"
    )
    lines = [
        f"JUDGING ESTIMATE: {estimate['pairs']} matched pairs x "
        f"{estimate['passes_per_pair']} passes = {estimate['calls']} calls to "
        f"{estimate['model']}",
        f"  ~{estimate['input_tokens']:,} input tokens, <= "
        f"{estimate['output_tokens_upper_bound']:,} output tokens   {price}",
        "  output is priced at the max_tokens ceiling, so the USD figure is an UPPER bound; "
        "this is SEPARATE from the benchmark's own spend and is never folded into it.",
    ]
    if estimate["nominal"]:
        lines.append(
            f"  NOMINAL: no answers exist yet, so each answer is assumed to be "
            f"{ESTIMATE_NOMINAL_ANSWER_CHARS:,} characters. The real prompts are priced "
            "again, exactly, after the benchmark runs and before any judge call is made."
        )
    return lines


def actual_cost(summary: dict[str, Any], model: str) -> dict[str, Any]:
    """Judge spend from the API's own usage counts, as its own line item."""
    input_tokens = sum(
        p["input_tokens"] for pair in summary["pairs"] for p in pair["passes"]
    )
    output_tokens = sum(
        p["output_tokens"] for pair in summary["pairs"] for p in pair["passes"]
    )
    usd = estimate_cost(model, input_tokens, output_tokens) if has_pricing(model) else None
    return {"input_tokens": input_tokens, "output_tokens": output_tokens, "usd": usd}


def format_judging(summary: dict[str, Any]) -> list[str]:
    arms = summary["arms"]
    wins = summary["wins"]
    lines = ["=" * 78, "PAIRED QUALITY JUDGING (blind, both presentation orders)", "=" * 78]
    lines.append(
        f"judge model : {summary['model']}   pairs judged: {summary['pairs_judged']} of "
        f"{summary['pairs_offered']}   judge calls: {summary['judge_calls']}"
    )
    lines.append(
        "design      : final answers only, no tool traces; answers labelled 1/2 with the "
        "arm never named; every pair judged in BOTH orders and a disagreement recorded as "
        "a tie."
    )
    if summary["pairs_unresolved"]:
        lines.append(
            f"unresolved  : {summary['pairs_unresolved']} pair(s) had a failed or "
            "unparseable judge call. They are counted here and are in NEITHER the win nor "
            "the tie totals — a pair judged once is a pair judged from one position."
        )
    if summary["pairs_offered"] == 0:
        lines.append("NOTE        : there were no matched pairs to judge.")
        return lines

    lines.append("-" * 78)
    for arm in arms:
        margins = summary["margins"][arm]
        share = summary["win_share_of_decisive"][arm]
        # a rate over fewer than MIN_DECISIVE_PAIRS decisive pairs cannot reach alpha at any
        # outcome, so it is not printed as a number at all — the same rule the sign test
        # below obeys. "1 win = 100.0% of decisive" above a NOT CONCLUSIVE line is exactly
        # the solid-looking number this module refuses to hand out.
        if share is None:
            share_text = "no decisive pairs"
        elif summary["underpowered"]:
            share_text = (
                f"NO RATE PRINTED: only {summary['decisive_pairs']} decisive pair(s), "
                f"below {summary['min_decisive_pairs_for_significance']}"
            )
        else:
            share_text = f"{share:.1%} of decisive"
        lines.append(
            f"  wins  {arm:<24}{wins[arm]:>5}   ({share_text}; "
            f"clear={margins['clear']}, slight={margins['slight']}, none={margins['none']})"
        )
    if summary["unrecognised_margin_words"]:
        lines.append(
            "  margin words the judge used that are not clear/slight/none and were counted "
            "as none: " + ", ".join(repr(w) for w in summary["unrecognised_margin_words"])
        )
    ties = summary["ties_by_kind"]
    lines.append(
        f"  ties  {'(all kinds)':<24}{summary['ties_total']:>5}   "
        f"(both orders agreed it was a tie={ties[OUTCOME_TIE_AGREED]}, one order tied="
        f"{ties[OUTCOME_TIE_UNSTABLE]}, the two orders picked DIFFERENT answers="
        f"{ties[OUTCOME_TIE_POSITION_FLIP]})"
    )
    lines.append(
        "  READ THE TIES AS A LIMIT ON THE INSTRUMENT, NOT AS A RESULT: every tie counts "
        "toward passing \"must not regress\", but a large "
        f"{OUTCOME_TIE_POSITION_FLIP} count ({ties[OUTCOME_TIE_POSITION_FLIP]} here) means "
        "position moved this judge more than the answers did — i.e. a regression of this "
        "size could not have been detected, which is not the same as there not being one."
    )
    p = summary["sign_test_p"]
    if summary["underpowered"]:
        lines.append(
            f"  SIGN TEST   : NOT CONCLUSIVE AT ANY OUTCOME — {summary['decisive_pairs']} "
            f"decisive pair(s). A two-sided sign test needs at least "
            f"{summary['min_decisive_pairs_for_significance']} to reach p<="
            f"{summary['sign_test_alpha']} even on a clean sweep, so read the counts and "
            "the per-pair detail, not a rate."
        )
    else:
        verdict = "distinguishable from chance" if (p or 1) <= summary["sign_test_alpha"] else (
            "NOT distinguishable from chance"
        )
        lines.append(
            f"  SIGN TEST   : p={p:.3f} over {summary['decisive_pairs']} decisive pairs "
            f"(ties excluded, not split) — {verdict} at alpha="
            f"{summary['sign_test_alpha']}."
        )
    # both directions, derived rather than assumed: which arm is "the candidate" is not
    # something this module can read off the arm order, and naming the wrong one inverts the
    # sentence a reader uses to decide whether the criterion was met
    lines.append(
        f"  the loss tail is the kill criterion: {arms[0]} lost {wins[arms[1]]} pair(s) to "
        f"{arms[1]}; {arms[1]} lost {wins[arms[0]]} pair(s) to {arms[0]}. Whichever of the "
        "two is the candidate, its losses are listed below."
    )

    bias = summary["length_bias"]
    slicing = summary["answer_slicing"]
    lines.append("-" * 78)
    lines.append(
        f"  length check: the longer SHOWN answer won "
        f"{bias['decisive_pairs_won_by_the_longer_shown_answer']} of "
        f"{bias['of_decisive_pairs']} decisive pairs; median chars AS SHOWN to the judge "
        + ", ".join(f"{arm}={bias['median_shown_answer_chars'][arm]}" for arm in arms)
        + " (raw, before elision: "
        + ", ".join(f"{arm}={bias['median_raw_answer_chars'][arm]}" for arm in arms)
        + "). Pairwise judges favour length; a number near the total is a warning about the "
        "verdict, not about the arms. The SHOWN figure is the one that can have moved the "
        f"judge — everything over {JUDGE_MAX_ANSWER_CHARS:,} characters looks the same length "
        "in the prompt."
    )
    dropped = slicing["median_dropped_pre_answer_chars"]
    lines.append(
        "  slicing loss: median characters DISCARDED before the judge saw anything (text "
        "written before the answer's last tool call) "
        + ", ".join(f"{arm}={dropped[arm]}" for arm in arms)
        + ". An arm whose last tool call comes late loses more here. If these medians "
        "differ materially the verdict is partly measuring the slicing rule rather than "
        "answer quality — see `final_answer_split`."
    )
    blind = summary["blinding"]
    restricted = blind["restricted_to_unidentifiable"]
    per_arm_markers = blind["marker_counts_per_arm"]
    marker_text = "; ".join(
        f"{arm}: " + (", ".join(f"{k}={v}" for k, v in per_arm_markers.get(arm, {}).items()) or "none")
        for arm in arms
    )
    lines.append(
        f"  blinding    : {blind['pairs_with_provenance_markers']} pair(s) contained text "
        f"naming how an answer was produced ({marker_text})"
        + ". Such text is NOT edited out — rewriting an answer changes what is judged — so "
        "the same table over the pairs where neither answer carried any: "
        + ", ".join(f"{arm}={restricted['wins'][arm]}" for arm in arms)
        + f", ties={restricted['ties_total']}, over {restricted['pairs_judged']} pair(s)"
        + _p_clause(restricted)
        + "."
    )
    lines.append(
        "                THE MARKER LIST IS ASYMMETRIC and cannot be otherwise: most "
        "markers are tells for the arm that writes code, and no phrase reliably marks an "
        "answer assembled from many small tool calls. So the restricted subset drops one "
        "arm's pairs preferentially — read the per-arm counts above — and a gap between "
        "the full and restricted tables is evidence about the JUDGE, not about the arms."
    )
    empty = summary["empty_answers_per_arm"]
    if any(empty.values()):
        sliced_away = summary["empty_answers_with_dropped_text_per_arm"]
        both = summary["restricted_to_pairs_with_both_answers"]
        lines.append(
            "  empty answers: "
            + ", ".join(f"{arm}={empty[arm]}" for arm in arms)
            + " (of which the turn HAD produced text before its last tool call, i.e. the "
            "slicing rule emptied it, not the model: "
            + ", ".join(f"{arm}={sliced_away[arm]}" for arm in arms)
            + f"). {summary['pairs_with_an_empty_answer']} judged pair(s) carry one. An "
            "answer that could not be extracted was NOT MEASURED, so scoring it as a loss "
            "measures the harness — the table over the pairs where BOTH arms produced "
            "text: "
            + ", ".join(f"{arm}={both['wins'][arm]}" for arm in arms)
            + f", ties={both['ties_total']}, over {both['pairs_judged']} pair(s)"
            + _p_clause(both)
            + ". If the headline stands only in the full table, it is a bug report."
        )
    truncated_per_arm = summary["truncated_answers_per_arm"]
    if any(truncated_per_arm.values()):
        lines.append(
            f"  truncation  : answers over {JUDGE_MAX_ANSWER_CHARS:,} characters are elided "
            "in the MIDDLE. PER ARM (the rule is applied identically, but it does not FIRE "
            "equally): "
            + ", ".join(f"{arm}={truncated_per_arm[arm]}" for arm in arms)
            + f", over {summary['truncated_answers']} affected pair(s). An arm elided far "
            "more often was judged on less of what it wrote."
        )

    lines.append("-" * 78)
    lines.append("PER-PAIR DETAIL (read the losses; that is what the criterion is about)")
    for pair in summary["pairs"]:
        head = (
            f"  {pair['case_id'][:12]:<12} turn {pair['turn_index']:<3} "
            f"{pair['outcome']:<20}"
        )
        if pair["winner"]:
            head += f"winner={pair['winner']} ({pair['margin']})"
        lines.append(head)
        shown_first = pair["passes"][0]["order"][0] if pair["passes"] else "n/a"
        lines.append(
            f"      shown first in pass 1: {shown_first}"
            + (
                "   markers: "
                + ", ".join(
                    f"{arm}:{'/'.join(found)}"
                    for arm, found in pair["provenance_markers"].items()
                )
                if pair["provenance_markers"]
                else ""
            )
        )
        for index, p in enumerate(pair["passes"], start=1):
            said = p["error"] or f"{p['verdict']} ({p['margin']}): {p['reason']}"
            lines.append(f"      pass {index} [{p['order'][0]} first]: {said}")
    return lines


async def judge_report(
    report: dict[str, Any],
    model: str = DEFAULT_JUDGE_MODEL,
    concurrency: int = 4,
    client: Any = None,
    print_estimate: bool = True,
) -> dict[str, Any]:
    """Judge a benchmark report in place and return the judging block."""
    pairs, arms = build_pairs(report)
    # `or`, not `and`: a report where EVERY pair is missing ONE arm's answer is the
    # catastrophic case — every pair would be scored as a loss for the arm that has no text,
    # producing a clean sweep and a significant p-value out of a harness failure. Requiring
    # both to be empty caught only the strictly milder version of the same bug.
    missing = [p for p in pairs if not p.answers[arms[0]] or not p.answers[arms[1]]]
    if pairs and len(missing) == len(pairs):
        raise ValueError(
            "every matched pair is missing at least one arm's answer text, so every pair "
            "would score as a loss for whichever arm came back empty. A report produced "
            "before genetics-results-suite-4h6.72 has no `final_answer` on its turns at "
            "all and cannot be judged after the fact; re-run the benchmark."
        )
    estimate = estimate_judging_cost(pairs, model)
    if print_estimate:
        for line in estimate_lines(estimate):
            print(line)
    verdicts = await judge_pairs(pairs, arms, model=model, concurrency=concurrency, client=client)
    summary = summarize_verdicts(verdicts, arms, model)
    summary["cost_estimate_before_run"] = estimate
    summary["cost_actual"] = actual_cost(summary, model)
    report["judging"] = summary
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Blind pairwise LLM judging of a saved replay-benchmark report. "
        "Costs real money: two Opus-5 calls per matched pair.",
    )
    parser.add_argument("--report", type=Path, required=True, help="a replay_benchmark JSON report")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-concurrency", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="write the report back out with the judging block attached (defaults to "
        "--report, overwritten in place)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="price the judging pass over this report's matched pairs and stop",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    if not args.report.exists():
        print(f"report not found: {args.report}", file=sys.stderr)
        return 2
    with open(args.report) as f:
        report = json.load(f)

    if args.dry_run:
        pairs, _arms = build_pairs(report)
        for line in estimate_lines(estimate_judging_cost(pairs, args.judge_model)):
            print(line)
        return 0

    summary = asyncio.run(
        judge_report(report, model=args.judge_model, concurrency=args.judge_concurrency)
    )
    destination = args.output or args.report
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("wrote %s", destination)
    print("\n".join(format_judging(summary)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
