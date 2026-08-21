"""Tests for the blind pairwise judging pass over the replay benchmark's matched pairs.

Every judge call goes through a fake client — the real one is Opus-5 and this suite must
never spend money. What is tested is the machinery around the model: that the arm cannot
reach the judge, that both presentation orders are actually used, that a disagreement
between them cannot become a win, and that nothing is priced or reported as a solid number
when it is not one.
"""

import io
import json
import os
import subprocess
import sys
from contextlib import redirect_stdout

import pytest
from test_replay_benchmark import (  # noqa: F401  (stub_server is a fixture)
    StubChatServer,
    make_case,
    stub_server,
    write_dataset,
)

from genetics_mcp_server.scripts.pairwise_judge import (
    JUDGE_MAX_ANSWER_CHARS,
    MIN_DECISIVE_PAIRS,
    OUTCOME_TIE_AGREED,
    OUTCOME_TIE_POSITION_FLIP,
    OUTCOME_TIE_UNSTABLE,
    OUTCOME_UNRESOLVED,
    OUTCOME_WIN,
    JudgePair,
    build_pairs,
    build_prompt,
    elide_middle,
    estimate_judging_cost,
    estimate_lines,
    dropped_prose_blocks,
    final_answer_split,
    final_answer_text,
    format_judging,
    judge_pairs,
    judge_report,
    presentation_order,
    scan_provenance_markers,
    sign_test_p,
    summarize_verdicts,
    user_question_text,
)
from genetics_mcp_server.scripts.replay_benchmark import format_summary, run_benchmark

ARMS = ("all", "code")


# ------------------------------------------------------------------ fake judge client


class FakeResponse:
    def __init__(self, payload: dict, input_tokens: int = 1000, output_tokens: int = 60):
        self.content = [type("Block", (), {"type": "text", "text": json.dumps(payload)})()]
        self.usage = type(
            "Usage",
            (),
            {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        )()


class FakeJudge:
    """A programmable stand-in for anthropic.AsyncAnthropic.

    `decide(prompt, call_index)` returns the verdict payload, or raises to simulate a
    failed call. Every prompt is recorded so a test can assert on what the judge was shown.
    """

    def __init__(self, decide):
        self.prompts: list[str] = []
        self.stdout_at_first_call: str | None = None
        self._decide = decide
        outer = self

        class Messages:
            async def create(self, **kwargs):
                prompt = kwargs["messages"][0]["content"]
                if outer.stdout_at_first_call is None:
                    outer.stdout_at_first_call = _current_stdout()
                outer.prompts.append(prompt)
                return FakeResponse(outer._decide(prompt, len(outer.prompts) - 1))

        self.messages = Messages()


def _current_stdout() -> str:
    """Whatever has been printed so far, when stdout is a redirected StringIO."""
    return sys.stdout.getvalue() if isinstance(sys.stdout, io.StringIO) else ""


def always(verdict: str, margin: str = "clear"):
    return lambda prompt, index: {"verdict": verdict, "margin": margin, "reason": "because"}


def picks_answer_containing(needle: str):
    """A judge that always prefers whichever position holds `needle`."""

    def decide(prompt, index):
        first, second = prompt.split("--- ANSWER 2 ---")
        return {
            "verdict": "1" if needle in first else "2" if needle in second else "tie",
            "margin": "clear",
            "reason": "because",
        }

    return decide


def make_pairs(n: int, answers=None) -> list[JudgePair]:
    return [
        JudgePair(
            case_id=f"case{i}",
            turn_index=0,
            question=f"question {i}",
            prior_questions=[],
            # ALPHA/BETA rather than "baseline"/"candidate": the judging prompt itself
            # contains the phrase "two candidate answers", so a needle of "candidate" would
            # match the instructions instead of an answer and silently test nothing
            answers=answers or {ARMS[0]: f"ALPHA answer {i}", ARMS[1]: f"BETA answer {i}"},
        )
        for i in range(n)
    ]


# ------------------------------------------------------------------ what the judge sees


def test_final_answer_is_the_text_after_the_last_tool_use_not_the_commentary():
    # MUTATION-CHECKED: returning every text block (dropping the last_tool_use slice) makes
    # the commentary assertion fail, which is the whole point — that prose names the arm.
    blocks = [
        {"type": "text", "text": "Let me run a Python script for that."},
        {"type": "tool_use", "id": "t1", "name": "run_analysis", "input": {"code": "print(1)"}},
        {"type": "text", "text": "RPH1 has 3 credible sets."},
    ]
    answer = final_answer_text(blocks)
    assert answer == "RPH1 has 3 credible sets."
    assert "run_analysis" not in answer
    assert "Python script" not in answer


def test_the_slicing_rule_reports_how_much_it_threw_away():
    """The rule is not neutral between the arms, so its effect has to be MEASURABLE.

    MUTATED: returning a constant 0 for the dropped count, or measuring the kept text
    instead, fails here. The four measured behaviours of the rule are pinned together
    because the handicap is the DIFFERENCE between them, not any one of them.
    """
    early_call = [  # one early tool call: almost everything survives
        {"type": "text", "text": "x" * 10},
        {"type": "tool_use", "id": "t", "name": "run_analysis", "input": {}},
        {"type": "text", "text": "the answer"},
    ]
    late_call = [  # substantive prose between calls, then a late call: it is discarded
        {"type": "text", "text": "y" * 400},
        {"type": "tool_use", "id": "t1", "name": "get_a", "input": {}},
        {"type": "text", "text": "z" * 600},
        {"type": "tool_use", "id": "t2", "name": "get_b", "input": {}},
        {"type": "text", "text": "In summary, yes."},
    ]
    assert final_answer_split(early_call) == ("the answer", 10)
    assert final_answer_split(late_call) == ("In summary, yes.", 1000)
    # the two cases where the rule keeps NOTHING, and says how much that was
    assert final_answer_split([{"type": "text", "text": "answered before the call"},
                               {"type": "tool_use", "id": "t", "name": "g", "input": {}}]) == (
        "",
        len("answered before the call"),
    )
    # no tool call at all: nothing is dropped and every text block is the answer
    assert final_answer_split([{"type": "text", "text": "no tools needed"}]) == (
        "no tools needed",
        0,
    )
    assert final_answer_split(None) == ("", 0)


async def test_the_per_arm_slicing_loss_is_carried_into_the_report():
    """MUTATED: dropping `answer_slicing` from the summary hides a verdict that is really
    measuring which arm calls its last tool later — nothing else in the report can see it,
    because `answer_chars` and the whole length diagnostic are computed on the ALREADY
    SLICED text.
    """
    judge = FakeJudge(always("tie"))
    pairs = [
        JudgePair(
            case_id=f"c{i}",
            turn_index=0,
            question="q",
            prior_questions=[],
            answers={"all": "short tail", "code": "long tail from an early call"},
            dropped_chars={"all": 9_000, "code": 20},
        )
        for i in range(3)
    ]
    verdicts = await judge_pairs(pairs, ARMS, client=judge)
    assert verdicts[0].dropped_chars == {"all": 9_000, "code": 20}
    summary = summarize_verdicts(verdicts, ARMS, "fake")
    assert summary["answer_slicing"]["median_dropped_pre_answer_chars"] == {
        "all": 9_000.0,
        "code": 20.0,
    }
    text = "\n".join(format_judging(summary))
    assert "slicing loss" in text
    assert "all=9000.0, code=20.0" in text


async def test_the_benchmark_records_what_the_slicing_rule_discarded(stub_server, tmp_path):  # noqa: F811  (the imported fixture)
    """End to end: the count has to survive into the saved report or the judge cannot see it.

    MUTATED: leaving `final_answer_dropped_chars` off `TurnRecord` (or never assigning it)
    makes the JudgePair read 0 for every arm, which is precisely the silent state the
    diagnostic exists to end.
    """
    stub_server.plan = {
        None: [
            {"type": "usage", "iteration": 1, "input_tokens": 10, "output_tokens": 5,
             "total_input_tokens": 10, "total_output_tokens": 5, "context_window": 1000,
             "context_percent": 1.0},
            {"type": "done", "message_content": [
                {"type": "text", "text": "a table the judge will never see"},
                {"type": "tool_use", "id": "t1", "name": "get_x", "input": {}},
                {"type": "text", "text": "the answer"},
            ]},
        ],
    }
    stub_server.plan["bigquery"] = stub_server.plan[None]
    dataset = write_dataset(tmp_path, [make_case("s1")])
    report = await run_benchmark(
        dataset=dataset, base_url=stub_server.base_url, arms=("all", "bigquery"), limit=None,
        concurrency=1, model=None, timeout=30.0, max_turns=None, auth_token=None,
    )
    for turn in report["turns"]:
        assert turn["final_answer"] == "the answer"
        assert turn["final_answer_dropped_chars"] == len("a table the judge will never see")
    pairs, _ = build_pairs(report)
    assert pairs[0].dropped_chars == {
        "all": len("a table the judge will never see"),
        "bigquery": len("a table the judge will never see"),
    }


def test_final_answer_of_a_turn_that_used_no_tools_is_all_of_its_text():
    assert final_answer_text([{"type": "text", "text": "no tools needed"}]) == "no tools needed"
    assert final_answer_text(None) == ""


def test_no_tool_trace_reaches_the_judge_prompt():
    # the code arm's giveaway is a run_analysis call carrying Python; the all-tools arm's is
    # a pile of get_* calls. Neither may appear in the prompt.
    code_arm = final_answer_text(
        [
            {"type": "tool_use", "id": "a", "name": "run_analysis", "input": {"code": "import x"}},
            {"type": "text", "text": "answer B"},
        ]
    )
    tool_arm = final_answer_text(
        [
            {"type": "tool_use", "id": "b", "name": "get_credible_sets", "input": {"gene": "X"}},
            {"type": "text", "text": "answer A"},
        ]
    )
    pair = JudgePair("c", 0, "q?", [], {ARMS[0]: tool_arm, ARMS[1]: code_arm})
    prompt = build_prompt(pair, list(ARMS), today="2026-08-18")
    for forbidden in ("run_analysis", "get_credible_sets", "import x", "tool_use"):
        assert forbidden not in prompt


def test_the_arm_name_never_reaches_the_prompt():
    # MUTATED: interpolating `first`/`second` into the rendered prompt (e.g. labelling the
    # sections with the arm names) makes this fail in both orders.
    sentinel_arms = ("BASELINE-SENTINEL-ARM", "CANDIDATE-SENTINEL-ARM")
    pair = JudgePair("c", 0, "q?", [], {a: f"answer from {i}" for i, a in enumerate(sentinel_arms)})
    for order in (list(sentinel_arms), list(reversed(sentinel_arms))):
        prompt = build_prompt(pair, order, today="2026-08-18")
        assert "SENTINEL" not in prompt
        assert "Answer 1" in prompt or "ANSWER 1" in prompt


def test_prior_context_is_the_shared_user_turns_and_never_an_arms_own_answer():
    pair = JudgePair(
        case_id="c",
        turn_index=1,
        question="and the second one?",
        prior_questions=["what are the top loci for T2D?"],
        answers={ARMS[0]: "locus A", ARMS[1]: "locus B"},
    )
    prompt = build_prompt(pair, list(ARMS), today="2026-08-18")
    assert "top loci for T2D" in prompt
    assert "EARLIER USER TURNS" in prompt
    # MUTATED: the shipped "both answers had the same conversation history" is false and is
    # an INSTRUCTION to the judge, not a comment — only the USER turns were shared; each
    # arm's own prior assistant turns differ and are deliberately withheld, so a judge told
    # otherwise could read a follow-up as under-specified for one of the two answers.
    assert "both answers had the same conversation history" not in prompt
    assert "these user turns were identical for both answers" in prompt
    assert "earlier replies are not shown to you" in prompt


def test_a_long_answer_is_elided_in_the_middle_so_the_conclusion_survives():
    text = "HEAD" + ("x" * (JUDGE_MAX_ANSWER_CHARS * 2)) + "TAIL"
    elided = elide_middle(text)
    assert elided.startswith("HEAD")
    assert elided.endswith("TAIL")
    assert len(elided) <= JUDGE_MAX_ANSWER_CHARS + len("[... middle of this answer elided for length ...]") + 8


def test_user_question_text_handles_both_recorded_shapes():
    assert user_question_text("plain") == "plain"
    assert user_question_text([{"type": "text", "text": "blocks"}]) == "blocks"
    assert user_question_text(None) == ""


# ------------------------------------------------------------------ order randomisation


def test_presentation_order_varies_across_pairs():
    # MUTATED: returning `list(arms)` unconditionally collapses this to one distinct order.
    seen = {tuple(presentation_order(f"case{i}", 0, ARMS)) for i in range(40)}
    assert len(seen) == 2, "the seeded order never swapped across 40 pairs"


def test_presentation_order_is_stable_for_the_same_pair():
    assert presentation_order("case7", 3, ARMS) == presentation_order("case7", 3, ARMS)
    # the seed is the PAIR, not the arms: a different turn of the same case can differ
    assert {tuple(presentation_order("case7", t, ARMS)) for t in range(20)} == {
        (ARMS[0], ARMS[1]),
        (ARMS[1], ARMS[0]),
    }


def test_presentation_order_does_not_move_with_the_process_hash_seed():
    """The seed must be reproducible ACROSS RUNS, not merely within one.

    MUTATED: swapping the sha256 for the builtin `hash()` makes the two subprocesses
    disagree, because PYTHONHASHSEED salts str hashing per process. Run as subprocesses on
    purpose — PYTHONHASHSEED cannot be changed after interpreter start.
    """
    script = (
        "from genetics_mcp_server.scripts.pairwise_judge import presentation_order;"
        "print([presentation_order(f'case{i}', i % 3, ('all', 'code'))[0] for i in range(20)])"
    )
    outputs = []
    for seed in ("0", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        outputs.append(
            subprocess.run(
                [sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True
            ).stdout
        )
    assert outputs[0] == outputs[1]


# ------------------------------------------------------------------ both-ways judging


async def test_every_pair_is_judged_in_both_presentation_orders():
    judge = FakeJudge(always("tie"))
    pairs = make_pairs(3)
    await judge_pairs(pairs, ARMS, client=judge)
    assert len(judge.prompts) == 6
    for i in range(0, 6, 2):
        first_of_pass1 = judge.prompts[i].split("--- ANSWER 2 ---")[0]
        first_of_pass2 = judge.prompts[i + 1].split("--- ANSWER 2 ---")[0]
        # MUTATED: dropping the second pass, or passing the same order twice, makes the two
        # position-1 answers identical and fails here
        assert first_of_pass1 != first_of_pass2


async def test_a_judge_that_always_picks_the_first_answer_scores_no_wins_at_all():
    """The position-bias defence, stated as a test.

    A maximally position-biased judge names position 1 every time. Because each pair is
    also judged with the answers swapped, the two passes name DIFFERENT arms and the pair
    is a tie. MUTATED: resolving a disagreement in favour of either pass turns this into a
    clean sweep for one arm — which is exactly the fake signal this design exists to refuse.
    """
    judge = FakeJudge(always("1"))
    verdicts = await judge_pairs(make_pairs(5), ARMS, client=judge)
    assert {v.outcome for v in verdicts} == {OUTCOME_TIE_POSITION_FLIP}
    summary = summarize_verdicts(verdicts, ARMS, "fake")
    assert summary["wins"] == {"all": 0, "code": 0}
    assert summary["ties_total"] == 5


async def test_agreement_across_both_orders_is_a_win_for_that_arm():
    judge = FakeJudge(picks_answer_containing("BETA answer"))
    verdicts = await judge_pairs(make_pairs(4), ARMS, client=judge)
    assert {v.outcome for v in verdicts} == {OUTCOME_WIN}
    assert {v.winner for v in verdicts} == {"code"}
    summary = summarize_verdicts(verdicts, ARMS, "fake")
    assert summary["wins"] == {"all": 0, "code": 4}
    assert summary["decisive_pairs"] == 4


async def test_a_winner_in_one_order_and_a_tie_in_the_other_is_a_tie():
    judge = FakeJudge(lambda prompt, index: {"verdict": "1" if index % 2 == 0 else "tie",
                                             "margin": "clear", "reason": "r"})
    verdicts = await judge_pairs(make_pairs(3), ARMS, client=judge)
    assert {v.outcome for v in verdicts} == {OUTCOME_TIE_UNSTABLE}
    assert all(v.winner is None for v in verdicts)


async def test_ties_are_expressible_and_are_not_forced_into_a_winner():
    judge = FakeJudge(always("tie", margin="none"))
    verdicts = await judge_pairs(make_pairs(4), ARMS, client=judge)
    assert {v.outcome for v in verdicts} == {OUTCOME_TIE_AGREED}
    summary = summarize_verdicts(verdicts, ARMS, "fake")
    assert summary["decisive_pairs"] == 0
    assert summary["win_share_of_decisive"] == {"all": None, "code": None}
    assert summary["sign_test_p"] is None


async def test_a_failed_judge_call_leaves_the_pair_unresolved_rather_than_won():
    """A pair judged once is a pair judged from one position, so it is not judged."""

    def decide(prompt, index):
        if index % 2:  # the second pass of every pair
            raise RuntimeError("overloaded")
        return {"verdict": "1", "margin": "clear", "reason": "r"}

    judge = FakeJudge(decide)
    verdicts = await judge_pairs(make_pairs(3), ARMS, client=judge)
    assert {v.outcome for v in verdicts} == {OUTCOME_UNRESOLVED}
    summary = summarize_verdicts(verdicts, ARMS, "fake")
    assert summary["wins"] == {"all": 0, "code": 0}
    assert summary["ties_total"] == 0
    assert summary["pairs_unresolved"] == 3
    assert "unresolved" in "\n".join(format_judging(summary))


async def test_a_first_pass_failure_does_not_pay_for_a_second_call():
    def decide(prompt, index):
        raise RuntimeError("down")

    judge = FakeJudge(decide)
    await judge_pairs(make_pairs(3), ARMS, client=judge)
    assert len(judge.prompts) == 3, "a pair already unresolvable was judged twice anyway"


async def test_an_unparseable_reply_is_unresolved_not_a_tie():
    judge = FakeJudge(lambda prompt, index: {"verdict": "maybe", "margin": "clear", "reason": ""})
    verdicts = await judge_pairs(make_pairs(2), ARMS, client=judge)
    assert {v.outcome for v in verdicts} == {OUTCOME_UNRESOLVED}


# ------------------------------------------------------------------ reporting


async def test_a_win_is_quoted_at_the_weaker_of_the_two_margins():
    """MUTATED: `"slight" if "slight" in margins else "clear"` reads a {verdict:1,
    margin:none} pass as a clear win, promoting the weakest possible signal."""
    margins = iter(["clear", "none"])

    def decide(prompt, index):
        return {"verdict": "1" if "ALPHA" in prompt.split("--- ANSWER 2 ---")[0] else "2",
                "margin": next(margins), "reason": "r"}

    verdicts = await judge_pairs(make_pairs(1), ARMS, client=FakeJudge(decide))
    assert verdicts[0].outcome == OUTCOME_WIN and verdicts[0].winner == "all"
    assert verdicts[0].margin == "none"


async def test_an_empty_answer_is_counted_and_flagged_rather_than_silently_judged():
    """Emptiness concentrated on one arm is a harness bug, not a quality result."""
    judge = FakeJudge(always("2"))
    pairs = make_pairs(1, answers={"all": "a real answer", "code": "   "})
    verdicts = await judge_pairs(pairs, ARMS, client=judge)
    assert verdicts[0].empty_answer_arms == ["code"]
    assert "(this answer was empty)" in judge.prompts[0]
    summary = summarize_verdicts(verdicts, ARMS, "fake")
    assert summary["empty_answers_per_arm"] == {"all": 0, "code": 1}
    assert "empty answers: all=0, code=1" in "\n".join(format_judging(summary))


async def test_a_one_sided_empty_answer_cannot_carry_the_headline_unchallenged():
    """The reproduced defect: eight pairs where one arm's answer could not be extracted
    produce `wins all 8 (100.0% of decisive)`, `p=0.008 ... distinguishable from chance`
    and a loss tail — while a sentence four lines below correctly calls a one-sided empty
    count a bug report and neither restricts nor recomputes any of it.

    MUTATED: dropping `restricted_to_pairs_with_both_answers` (or leaving it out of the
    printed block) restores exactly that state — a significant headline with no second
    table, which is the opposite of how provenance, a strictly milder problem, is handled.
    """
    unextractable = [
        JudgePair(f"empty{i}", 0, "q", [], {"all": f"a real answer {i}", "code": ""})
        for i in range(8)
    ]
    measured = [
        JudgePair(f"real{i}", 0, "q", [], {"all": f"alpha {i}", "code": f"beta {i}"})
        for i in range(4)
    ]
    judge = FakeJudge(picks_answer_containing("a real answer"))
    verdicts = await judge_pairs(unextractable + measured, ARMS, client=judge)
    summary = summarize_verdicts(verdicts, ARMS, "fake")
    # the headline: a clean sweep and a significant p-value assembled entirely out of pairs
    # in which one arm's answer was never measured
    assert summary["wins"] == {"all": 8, "code": 0}
    assert summary["sign_test_p"] is not None and summary["sign_test_p"] < 0.05
    assert summary["underpowered"] is False
    assert summary["pairs_with_an_empty_answer"] == 8
    # the restriction that has to be printed beside it: over the pairs where both arms
    # actually produced text there is no winner at all
    both = summary["restricted_to_pairs_with_both_answers"]
    assert both["pairs_judged"] == 4
    assert both["wins"] == {"all": 0, "code": 0}
    assert both["ties_total"] == 4
    assert both["decisive_pairs"] == 0
    assert both["sign_test_p"] is None
    text = "\n".join(format_judging(summary))
    assert "NOT MEASURED" in text
    assert "all=0, code=0, ties=4, over 4 pair(s)" in text


def test_an_empty_answer_the_slicing_rule_caused_is_distinguishable_from_a_silent_model():
    """MUTATED: reporting only `empty_answers_per_arm` cannot tell "the model said nothing"
    from "the harness threw the answer away", which are a quality finding and a bug."""
    from genetics_mcp_server.scripts.pairwise_judge import PairVerdict

    ate_it = PairVerdict(
        case_id="sliced", turn_index=0, outcome=OUTCOME_WIN, winner="all", loser="code",
        margin="clear", answer_chars={"all": 10, "code": 0},
        shown_answer_chars={"all": 10, "code": 0},
        dropped_chars={"all": 0, "code": 4_000}, empty_answer_arms=["code"],
    )
    really_silent = PairVerdict(
        case_id="silent", turn_index=0, outcome=OUTCOME_WIN, winner="all", loser="code",
        margin="clear", answer_chars={"all": 10, "code": 0},
        shown_answer_chars={"all": 10, "code": 0},
        dropped_chars={"all": 0, "code": 0}, empty_answer_arms=["code"],
    )
    summary = summarize_verdicts([ate_it, really_silent], ARMS, "fake")
    assert summary["empty_answers_per_arm"] == {"all": 0, "code": 2}
    assert summary["empty_answers_with_dropped_text_per_arm"] == {"all": 0, "code": 1}
    assert "the slicing rule emptied it, not the model" in "\n".join(format_judging(summary))


async def test_every_pair_missing_ONE_arms_answer_is_refused_not_swept():
    """MUTATED: the shipped `and` fires only when BOTH arms are empty on EVERY pair, i.e.
    never in the catastrophic case — a report where one arm's extraction failed everywhere
    would be judged and would produce a clean sweep with a significant p-value out of a
    harness bug.
    """
    report = _report_with_answers()
    for turn in report["turns"]:
        if turn["arm"] == ARMS[1]:
            turn["final_answer"] = ""
    with pytest.raises(ValueError, match="at least one arm's answer text"):
        await judge_report(report, client=FakeJudge(always("tie")))


def test_sign_test_matches_the_exact_binomial():
    assert sign_test_p(0, 0) is None
    assert sign_test_p(5, 0) == pytest.approx(2 * 0.5**5)
    assert sign_test_p(6, 0) == pytest.approx(2 * 0.5**6)
    assert sign_test_p(3, 3) == 1.0


def test_a_win_rate_over_too_few_pairs_does_not_render_as_a_solid_number():
    """MUTATED: printing `p` unconditionally drops the NOT CONCLUSIVE line and fails here.

    Below MIN_DECISIVE_PAIRS no outcome can reach alpha, so a rate would be theatre — the
    same rule the benchmark applies to a p95 that is really the sample maximum.
    """
    assert MIN_DECISIVE_PAIRS == 6
    summary = _summary_with(wins_for_code=MIN_DECISIVE_PAIRS - 1)
    assert summary["underpowered"] is True
    text = "\n".join(format_judging(summary))
    assert "NOT CONCLUSIVE AT ANY OUTCOME" in text
    assert "p=" not in text
    # MUTATED: rendering `win_share_of_decisive` unconditionally prints "100.0% of decisive"
    # directly above the NOT CONCLUSIVE line. Asserting only on "p=" let that through, which
    # is how a solid-looking rate survived the rule that exists to forbid it.
    assert "% of decisive" not in text
    assert "100.0%" not in text
    assert "NO RATE PRINTED" in text

    enough = _summary_with(wins_for_code=MIN_DECISIVE_PAIRS)
    assert enough["underpowered"] is False
    enough_text = "\n".join(format_judging(enough))
    assert "p=0.031" in enough_text
    assert "100.0% of decisive" in enough_text


def _summary_with(wins_for_code: int, **pair_kwargs):
    from genetics_mcp_server.scripts.pairwise_judge import PairVerdict

    verdicts = [
        PairVerdict(
            case_id=f"c{i}",
            turn_index=0,
            outcome=OUTCOME_WIN,
            winner="code",
            loser="all",
            margin="clear",
            passes=[],
            answer_chars={"all": 100, "code": 200},
            shown_answer_chars={"all": 100, "code": 200},
            **pair_kwargs,
        )
        for i in range(wins_for_code)
    ]
    return summarize_verdicts(verdicts, ARMS, "fake")


def test_the_distribution_is_reported_with_per_pair_detail_not_a_bare_rate():
    judge_summary = _summary_with(wins_for_code=2)
    text = "\n".join(format_judging(judge_summary))
    assert "wins  all" in text and "wins  code" in text
    assert "ties" in text
    assert "PER-PAIR DETAIL" in text
    assert "c0" in text and "c1" in text


def test_the_length_confound_is_measured_and_printed():
    """Pairwise judges favour length; the report says how often the longer answer won."""
    summary = _summary_with(wins_for_code=3)  # code's answers are the longer ones above
    assert summary["length_bias"]["decisive_pairs_won_by_the_longer_shown_answer"] == 3
    assert "the longer SHOWN answer won 3 of 3" in "\n".join(format_judging(summary))


def test_the_length_diagnostic_measures_the_lengths_the_judge_ACTUALLY_SAW():
    """MUTATED: computing `longer_won` and the medians from `answer_chars` (the raw text)
    makes this fail, and that mutation IS the shipped-once defect: with 40,000 vs 20,000
    characters the judge is handed 11,999 each — identical — while the one diagnostic whose
    job is to warn "your judge is rewarding length" announces a 20,000-character gap that
    was never in the prompt.
    """
    from genetics_mcp_server.scripts.pairwise_judge import PairVerdict

    long_answer = "L" * 40_000
    short_answer = "S" * 20_000
    shown = {"all": len(elide_middle(long_answer)), "code": len(elide_middle(short_answer))}
    assert shown["all"] == shown["code"], "both are elided to the same size; that is the point"
    verdicts = [
        PairVerdict(
            case_id=f"c{i}",
            turn_index=0,
            outcome=OUTCOME_WIN,
            winner="all",
            loser="code",
            margin="clear",
            answer_chars={"all": 40_000, "code": 20_000},
            shown_answer_chars=dict(shown),
        )
        for i in range(3)
    ]
    summary = summarize_verdicts(verdicts, ARMS, "fake")
    bias = summary["length_bias"]
    assert bias["decisive_pairs_won_by_the_longer_shown_answer"] == 0
    assert bias["median_shown_answer_chars"]["all"] == bias["median_shown_answer_chars"]["code"]
    # the raw figure is kept, but labelled as the thing the judge did NOT see
    assert bias["median_raw_answer_chars"] == {"all": 40000.0, "code": 20000.0}
    text = "\n".join(format_judging(summary))
    assert "the longer SHOWN answer won 0 of 3" in text
    assert "raw, before elision" in text


def test_truncation_is_reported_per_arm_not_as_a_symmetric_pair_count():
    """MUTATED: printing only the pair count prints "3 pair(s) ... for both arms alike"
    when only one arm was ever elided — a reader concludes the information loss was
    symmetric while one arm was judged on a third of its answer and the other on all of it.
    """
    from genetics_mcp_server.scripts.pairwise_judge import PairVerdict

    verdicts = [
        PairVerdict(
            case_id=f"c{i}",
            turn_index=0,
            outcome=OUTCOME_WIN,
            winner="code",
            loser="all",
            margin="clear",
            answer_chars={"all": 40_000, "code": 500},
            shown_answer_chars={"all": JUDGE_MAX_ANSWER_CHARS, "code": 500},
            truncated_arms=["all"],
        )
        for i in range(3)
    ]
    summary = summarize_verdicts(verdicts, ARMS, "fake")
    assert summary["truncated_answers_per_arm"] == {"all": 3, "code": 0}
    text = "\n".join(format_judging(summary))
    assert "all=3, code=0" in text
    assert "both arms alike" not in text


async def test_an_unrecognised_margin_word_is_surfaced_rather_than_vanishing():
    """MUTATED: printing only clear/slight leaves `wins code 3 (clear=0, slight=0)` — three
    numbers that do not sum and no explanation, because `_parse_verdict` normalised an
    unrecognised strength word to "none" and dropped the word.
    """
    judge = FakeJudge(
        lambda prompt, index: {
            "verdict": "1" if "BETA" in prompt.split("--- ANSWER 2 ---")[0] else "2",
            "margin": "overwhelming",
            "reason": "r",
        }
    )
    verdicts = await judge_pairs(make_pairs(3), ARMS, client=judge)
    assert {v.outcome for v in verdicts} == {OUTCOME_WIN}
    assert all(v.margin == "none" for v in verdicts)
    summary = summarize_verdicts(verdicts, ARMS, "fake")
    assert summary["margins"]["code"] == {"clear": 0, "slight": 0, "none": 3}
    assert summary["unrecognised_margin_words"] == ["overwhelming"]
    text = "\n".join(format_judging(summary))
    assert "none=3" in text
    assert "'overwhelming'" in text


def test_the_loss_tail_line_names_both_directions_not_a_hard_coded_candidate():
    """MUTATED: `arms[1] lost wins[arms[0]] to arms[0]` reads correctly under the default
    arm order and inverts the sentence when the arms are swapped, so the direction is
    derived from the counts instead of assumed from the order.
    """
    summary = _summary_with(wins_for_code=2)  # every win belongs to `code`
    text = "\n".join(format_judging(summary))
    assert "all lost 2 pair(s) to code" in text
    assert "code lost 0 pair(s) to all" in text


# ------------------------------------------------------------------ blinding audit


def test_provenance_markers_are_detected():
    assert "run_analysis" in scan_provenance_markers("I called run_analysis for this")
    assert "script_self_reference" in scan_provenance_markers("I wrote a Python script to do it")
    assert "sandbox" in scan_provenance_markers("the sandbox returned 12 rows")
    assert "python_fence" in scan_provenance_markers("```python\nx=1\n```")
    assert scan_provenance_markers("RPH1 has three credible sets in FinnGen R12.") == []


def test_the_word_artifact_in_ordinary_genetics_prose_is_not_a_provenance_marker():
    """MUTATED: the shipped `\\bartifacts?\\b` fires on all three of these.

    In this domain "artifact" means a spurious signal far more often than a sandbox file,
    and it fires on BOTH arms — so the bare pattern shrinks the clean control subset for a
    reason that has nothing to do with guessing an arm, which is the one thing the subset
    is for.
    """
    for prose in (
        "this is likely a batch artifact of imputation",
        "an artifact of population structure rather than a real association",
        "a technical artifact of the genotyping array",
    ):
        assert scan_provenance_markers(prose) == [], prose
    assert "artifact_reference" in scan_provenance_markers("saved to the artifacts directory")
    assert "artifact_reference" in scan_provenance_markers("the artifact file from the run")


def test_the_iteration_cap_notice_counts_as_a_provenance_marker():
    """It survives `final_answer_text` verbatim into the judged answer and is a near-perfect
    tell for the many-roundtrip arm. MUTATED: leaving it off the list lets the blinding
    diagnostic report zero identifiable pairs on a run where the arm is written on the page.
    """
    from genetics_mcp_server.llm_service import MAX_ITERATIONS_NOTICE

    answer = final_answer_text(
        [
            {"type": "tool_use", "id": "t", "name": "get_x", "input": {}},
            {"type": "text", "text": "partial findings." + MAX_ITERATIONS_NOTICE},
        ]
    )
    assert "Max tool iterations reached" in answer, "the notice reaches the judge verbatim"
    assert "max_iterations_notice" in scan_provenance_markers(answer)
    assert "max_iterations_notice" in scan_provenance_markers(
        "[Max tool iterations reached (12). Partial results above.]"
    )


def test_the_blinding_line_states_the_marker_lists_asymmetry_where_the_numbers_are():
    """MUTATED: confessing the asymmetry only in a source comment (as shipped) leaves the
    printed report handing the reader a restricted table with no hint that the markers drop
    one arm's pairs preferentially — the exact misreading the table invites.
    """
    text = "\n".join(format_judging(_summary_with(wins_for_code=2)))
    assert "ASYMMETRIC" in text
    assert "evidence about the JUDGE" in text


def test_the_restricted_table_obeys_the_same_power_rule_as_the_headline():
    """MUTATED: leaving `underpowered` off the restricted table lets a p=0.031 over 6 pairs
    sit quotable in the saved JSON while the printed report prudently omits it — the report
    and the JSON disagreeing about whether a number means anything.
    """
    summary = _summary_with(wins_for_code=MIN_DECISIVE_PAIRS - 1)
    for table in (
        summary["blinding"]["restricted_to_unidentifiable"],
        summary["restricted_to_pairs_with_both_answers"],
    ):
        assert table["underpowered"] is True
        assert table["min_decisive_pairs_for_significance"] == MIN_DECISIVE_PAIRS
    text = "\n".join(format_judging(summary))
    assert "no sign test:" in text
    assert "p=" not in text


def test_pairs_whose_arm_is_guessable_are_flagged_and_reported_separately():
    """Blinding is audited, not assumed — and never by editing the answers.

    MUTATED: dropping `arm_identifiable` from the summary removes the restricted table, so
    a verdict that exists only in the pairs whose arm was guessable would read as clean.
    """
    from genetics_mcp_server.scripts.pairwise_judge import PairVerdict

    leaky = PairVerdict(
        case_id="leaky",
        turn_index=0,
        outcome=OUTCOME_WIN,
        winner="code",
        loser="all",
        margin="clear",
        answer_chars={"all": 10, "code": 20},
        shown_answer_chars={"all": 10, "code": 20},
        provenance_markers={"code": ["sandbox"]},
        arm_identifiable=True,
    )
    clean = PairVerdict(
        case_id="clean",
        turn_index=0,
        outcome=OUTCOME_WIN,
        winner="code",
        loser="all",
        margin="clear",
        answer_chars={"all": 10, "code": 20},
        shown_answer_chars={"all": 10, "code": 20},
    )
    summary = summarize_verdicts([leaky, clean], ARMS, "fake")
    assert summary["blinding"]["pairs_with_provenance_markers"] == 1
    # per ARM: a pooled count cannot say whether every marked pair was the same arm's, and
    # one-sidedness is the entire reason the restricted table exists
    assert summary["blinding"]["marker_counts_per_arm"] == {"all": {}, "code": {"sandbox": 1}}
    restricted = summary["blinding"]["restricted_to_unidentifiable"]
    assert restricted["pairs_judged"] == 1 and restricted["wins"]["code"] == 1
    assert "restricted" in json.dumps(summary["blinding"])
    assert "naming how an answer was produced" in "\n".join(format_judging(summary))


async def test_marker_bearing_answers_are_judged_verbatim_not_scrubbed():
    judge = FakeJudge(always("tie"))
    pairs = make_pairs(1, answers={"all": "plain answer", "code": "I ran a script in the sandbox"})
    verdicts = await judge_pairs(pairs, ARMS, client=judge)
    assert "I ran a script in the sandbox" in judge.prompts[0]
    # MUTATED: hard-coding arm_identifiable=False leaves the audit silent while the pair is
    # plainly attributable, so the flag is asserted on the judged verdict, not constructed
    assert verdicts[0].arm_identifiable is True
    assert verdicts[0].provenance_markers == {"code": ["sandbox", "script_self_reference"]}


# ------------------------------------------------------------------ cost


def test_the_estimate_is_priced_from_the_real_prompts_and_doubled_for_two_passes():
    pairs = make_pairs(4)
    estimate = estimate_judging_cost(pairs, "claude-opus-5")
    assert estimate["pairs"] == 4
    assert estimate["calls"] == 8
    assert estimate["usd_upper_bound"] > 0
    assert estimate["nominal"] is False
    single = estimate_judging_cost(make_pairs(2), "claude-opus-5")
    assert estimate["input_tokens"] > single["input_tokens"]


def test_an_unpriceable_judge_model_reports_no_usd_rather_than_a_guess():
    # MUTATED: dropping the has_pricing guard prices gpt-4o at Sonnet rates and returns a
    # confident wrong number, which is the defect cost.has_pricing exists to prevent.
    estimate = estimate_judging_cost(make_pairs(2), "gpt-4o")
    assert estimate["usd_upper_bound"] is None
    assert "NOT PRICED" in "\n".join(estimate_lines(estimate))


def test_the_dry_run_estimate_is_labelled_nominal():
    estimate = estimate_judging_cost(None, "claude-opus-5", pair_count=10)
    assert estimate["nominal"] is True
    assert "NOMINAL" in "\n".join(estimate_lines(estimate))


async def test_the_estimate_is_printed_before_the_first_judge_call():
    """Priced before it spends — the ordering, not merely the presence, of the line."""
    report = _report_with_answers()
    judge = FakeJudge(always("tie"))
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        await judge_report(report, model="claude-opus-5", client=judge)
    assert "JUDGING ESTIMATE" in judge.stdout_at_first_call


async def test_actual_spend_is_reported_from_usage_and_kept_off_the_arms_cost():
    report = _report_with_answers()
    judge = FakeJudge(always("tie"))
    with redirect_stdout(io.StringIO()):
        summary = await judge_report(report, model="claude-opus-5", client=judge)
    actual = summary["cost_actual"]
    assert actual["input_tokens"] == 1000 * summary["judge_calls"]
    assert actual["usd"] > 0
    footer = format_summary(report)
    assert "judging cost is a SEPARATE line item" in footer


# ------------------------------------------------------------------ wiring


def _report_with_answers(statuses=("ok", "ok")):
    """A real benchmark report, built through build_report so its shape cannot drift."""
    from genetics_mcp_server.scripts.replay_benchmark import CaseResult, TurnRecord, build_report

    case = CaseResult(case_id="s1", topic="t", arm_order=list(ARMS))
    for position, (arm, status) in enumerate(zip(ARMS, statuses)):
        case.turns.append(
            TurnRecord(
                case_id="s1",
                arm=arm,
                arm_position=position,
                turn_index=0,
                status=status,
                user_question="which genes associate with T2D?",
                final_answer=f"{arm} says TCF7L2",
            )
        )
    return build_report(
        [case], ARMS, {"base_url": "u", "dataset": "d", "model": None, "provider": None}
    )


def test_the_judge_input_is_the_harness_matched_pairs_not_every_turn():
    # MUTATED: pairing on status-agnostic keys admits the failed turn and yields 1 pair.
    pairs, arms = build_pairs(_report_with_answers(statuses=("ok", "error")))
    assert pairs == []
    assert arms == ARMS
    ok_pairs, _ = build_pairs(_report_with_answers())
    assert len(ok_pairs) == 1
    assert ok_pairs[0].answers == {"all": "all says TCF7L2", "code": "code says TCF7L2"}
    assert ok_pairs[0].question == "which genes associate with T2D?"


async def test_a_report_without_answer_text_is_refused_rather_than_judged_on_nothing():
    report = _report_with_answers()
    for turn in report["turns"]:
        turn.pop("final_answer")
    with pytest.raises(ValueError, match="4h6.72"):
        await judge_report(report, client=FakeJudge(always("tie")))


async def test_the_benchmark_records_the_question_and_the_final_answer(stub_server, tmp_path):  # noqa: F811  (the imported fixture)
    stub_server.plan = {
        None: [
            {"type": "usage", "iteration": 1, "input_tokens": 10, "output_tokens": 5,
             "total_input_tokens": 10, "total_output_tokens": 5, "context_window": 1000,
             "context_percent": 1.0},
            {"type": "done", "message_content": [
                {"type": "text", "text": "checking"},
                {"type": "tool_use", "id": "t1", "name": "get_x", "input": {}},
                {"type": "text", "text": "the answer"},
            ]},
        ],
    }
    stub_server.plan["bigquery"] = stub_server.plan[None]
    dataset = write_dataset(tmp_path, [make_case("s1")])
    report = await run_benchmark(
        dataset=dataset, base_url=stub_server.base_url, arms=("all", "bigquery"), limit=None,
        concurrency=1, model=None, timeout=30.0, max_turns=None, auth_token=None,
    )
    for turn in report["turns"]:
        assert turn["user_question"] == "question 0"
        assert turn["final_answer"] == "the answer"
    pairs, _ = build_pairs(report)
    assert len(pairs) == 1


def test_judging_is_off_unless_asked_for(tmp_path):
    from genetics_mcp_server.scripts.replay_benchmark import build_parser

    assert build_parser().parse_args([]).judge is False
    report = _report_with_answers()
    assert "judging" not in report
    # the cost/latency report renders with no judging section at all
    assert "PAIRED QUALITY JUDGING" not in format_summary(_full_report_shell())


def _full_report_shell():
    from genetics_mcp_server.scripts.replay_benchmark import build_report

    return build_report([], ARMS, {"base_url": "u", "dataset": "d", "model": None, "provider": None})


def test_a_saved_report_can_be_priced_without_re_running_the_benchmark(tmp_path, capsys):
    """The judge is independently runnable over a report the benchmark already wrote."""
    from genetics_mcp_server.scripts import pairwise_judge

    path = tmp_path / "report.json"
    path.write_text(json.dumps(_report_with_answers()))
    assert pairwise_judge.main(["--report", str(path), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "JUDGING ESTIMATE: 1 matched pairs x 2 passes = 2 calls" in out


def test_dropped_prose_is_returned_with_the_call_it_followed():
    blocks = [
        {"type": "text", "text": "Let me look that up."},
        {"type": "tool_use", "id": "t1", "name": "search_genes", "input": {}},
        {"type": "text", "text": "| gene | p |\n| LDLR | 1e-58 |"},
        {"type": "tool_use", "id": "t2", "name": "get_burden", "input": {}},
        {"type": "text", "text": "In summary, yes."},
    ]
    assert dropped_prose_blocks(blocks) == [
        {"after_call": None, "text": "Let me look that up."},
        {"after_call": 0, "text": "| gene | p |\n| LDLR | 1e-58 |"},
    ]
    # the boundary is the one final_answer_split uses, so the two cannot disagree
    kept, dropped_chars = final_answer_split(blocks)
    assert kept == "In summary, yes."
    assert dropped_chars > 0


def test_dropped_prose_is_empty_when_the_rule_discards_nothing():
    assert dropped_prose_blocks([{"type": "text", "text": "no tools needed"}]) == []
    assert dropped_prose_blocks(None) == []


def test_dropped_prose_keeps_everything_when_the_last_block_is_the_call():
    blocks = [
        {"type": "text", "text": "here is the table"},
        {"type": "tool_use", "id": "t1", "name": "x", "input": {}},
    ]
    assert dropped_prose_blocks(blocks) == [
        {"after_call": None, "text": "here is the table"}
    ]
    assert final_answer_split(blocks)[0] == ""
