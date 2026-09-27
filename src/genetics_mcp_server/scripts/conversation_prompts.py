"""Prompt templates for LLM-based conversation categorization."""

TOPIC_CLASSIFICATION_PROMPT = """\
You are classifying user questions from a genetics AI assistant into topic categories.

Below is a batch of first messages from different conversations, each prefixed with an ID.
Classify each into exactly ONE primary topic category from this list:

- gene_lookup: Questions about a specific gene (associations, expression, function)
- variant_interpretation: Questions about specific variants (what does variant X do, annotations)
- phenotype_exploration: Questions about diseases/traits (GWAS results, loci, phenotype reports)
- cross_phenotype_analysis: Comparing associations across multiple phenotypes or looking for shared signals
- colocalization_ld: Questions about colocalization, LD, or shared causal variants
- literature_search: Requests to find papers or literature on a topic
- data_source_question: Questions about available data, datasets, methods, or how the system works
- variant_list_analysis: Submitting a list of variants for batch analysis
- clinical_genetics: Clinical interpretation, Mendelian disease, patient variant interpretation
- bigquery_advanced: Complex analytical queries requiring SQL or BigQuery
- drug_target: Whether a gene or protein is a drug target, druggability, existing drugs or compounds and their mechanism, phase or repurposing
- general_genetics: General genetics questions not fitting other categories
- off_topic: Not related to genetics at all

Also assign a complexity score (1-3):
1 = simple lookup (single gene/variant/phenotype)
2 = moderate (requires multiple tools or cross-referencing)
3 = complex (multi-step analysis, interpretation, or novel questions)

Respond with a JSON array, one object per message:
[{{"id": "...", "topic": "...", "complexity": 1, "brief_reason": "..."}}]

Messages:
{messages}
"""

# fixed taxonomy for grouping detailed, per-conversation quality issues into
# recurring underlying problems. each detailed issue from the judge is mapped to
# exactly one of these so the report can count real patterns instead of unique strings.
#
# the descriptions are the categorizer's only guidance, so they are written to
# separate near neighbours: platform_error vs tool_failure_handling (the outage
# vs the assistant's handling of it), inaccurate_claim vs self_corrected_error
# (stayed wrong vs got fixed), capability_gap vs missed_data_source (data the
# system lacks vs data it has and did not use).
ISSUE_CATEGORIES = [
    ("incomplete_answer", "Answered only part of the question, omitted requested detail, or narrowed the scope without being asked"),
    ("missed_data_source", "Failed to use or find data the system does have; claimed no data when data exists; queried the wrong source"),
    ("capability_gap", "The requested data or analysis is not available through the assistant's tools (missing dataset, reference panel, plot type, external database, or API filter) and the assistant disclosed the limitation"),
    ("inaccurate_claim", "Stated something factually wrong, misleading, or unsupported, and it was not corrected within the conversation"),
    ("self_corrected_error", "Made an error that was later corrected, either by the assistant itself or after the user pushed back"),
    ("fabrication", "Invented data, results, numbers, citations, or tool output that was not actually returned"),
    ("weak_grounding", "Conclusions rest on proxies, extrapolation, or speculation and are presented with more rigor than the data supports, even if caveated"),
    ("inefficient_tool_use", "Redundant, repeated, or unnecessary tool calls; schema or column-name thrashing; per-item calls where a batch was possible"),
    ("tool_failure_handling", "A tool errored or returned nothing and the assistant gave up, did not retry, or failed to recover"),
    ("platform_error", "Infrastructure problem outside the assistant's control that the assistant handled acceptably: connection interrupted or response cut off mid-stream, attachment did not reach the assistant, download link or endpoint broken, external API rate-limited or down, tool output truncated"),
    ("misunderstood_question", "Misinterpreted what the user was actually asking, or proceeded on an unconfirmed assumption where a clarifying question was warranted"),
    ("no_conclusion", "Did not synthesize results into a clear answer; left the conversation hanging or ended awaiting a go-ahead"),
    ("missing_interpretation", "Returned raw data or tables without interpreting them for the user"),
    ("missing_deliverable", "A requested file, table, plot, or download was described or promised but not actually produced or usable"),
    ("formatting_readability", "Poor formatting or hard to read; dumped raw tables; overly long or dense output relative to the question"),
    ("overcautious", "Unnecessarily refused, hedged, or added excessive caveats"),
    ("not_an_issue", "Not a problem at all: praise, a neutral observation, a limitation the assistant handled well, or a note about a greeting or test message"),
    ("other", "A genuine issue that does not fit any category above"),
]

# categories that must not count as problems in the per-conversation category
# list, the conversation_issue table, the report or the admin chart
NON_ISSUE_CATEGORIES = frozenset({"not_an_issue"})

ISSUE_CATEGORIZATION_PROMPT = """\
You are grouping individual quality issues found across a genetics AI assistant's
conversations into recurring underlying problem categories.

Assign each issue below to exactly ONE category from this list:
{categories}

Pick the closest category; an issue rarely fits a description word for word. Use
"other" only when no category is even approximately right. If the text describes
a strength, or something the assistant handled well, rather than a problem, use
"not_an_issue".

Each issue is prefixed with a numeric ID. Respond with a single JSON array (not
one object per line), one object per issue, using only category names from the
list above:
[{{"id": 0, "category": "..."}}]

Issues:
{issues}
"""

QUALITY_ASSESSMENT_PROMPT = """\
You are evaluating the quality of an AI genetics assistant's conversation.
Today's date is {today}.

IMPORTANT — what you can and cannot verify:
- You are shown the assistant's rendered messages, but NOT the raw output the data
  tools returned. The assistant queries REAL genetics databases (FinnGen, GWAS
  summary stats, BigQuery, etc.). Specific, precise figures (sample sizes, variant
  counts, p-values) and recent dates are therefore EXPECTED and are almost always
  real tool output.
- Do NOT label something fabricated or hallucinated just because it is precise,
  unusual, or because you personally cannot verify it. Only call out fabrication
  when there is clear internal evidence (e.g. the assistant contradicts itself, or
  claims a tool result it never called). When unsure, assume the data are real.
- Dates in 2025 or early 2026 are in the PAST relative to today's date above; they
  are not "future dates" and are not evidence of hallucination.

Users may attach files (uploaded TSVs, images, etc.). An attachment is shown as a
line like "[User attached file(s): NAME (type, size)]". The assistant had access to
the FULL contents of any attached file even though those contents are not reproduced
below. So when a user references "the results" or "the file" and an attachment is
present, the assistant is NOT fabricating by analyzing it — treat the attached data
as legitimately available context, not invented.

First, classify the conversation's DISPOSITION (what kind of outcome it was). Pick
exactly one:
- good_answer: the assistant gave a good, complete answer to an answerable question.
- agent_failure: the question was answerable with the assistant's tools/data, but the
  assistant failed to answer it well (wrong, incomplete, gave up, ignored available data).
- technical_failure: a technical/infrastructure problem prevented a good answer
  (connection interrupted, backend/tool returned errors or empty results). NOT the
  assistant's fault, but the user was not served.
- out_of_scope: the user asked for something the system genuinely does not have or
  cannot do (e.g. data not available, an action outside its capabilities). Judge this
  by whether the request is answerable AT ALL, not by whether you can verify the data.
- unfinished: the conversation simply stops — the user asked something and did not
  continue — with no failure by the assistant.
- weird_or_unclear: the user's message is unclear, malformed, or appears to be missing
  context/attachments, so there is no well-formed question to answer.

Scoring rule for quality_score (1-5): score ONLY how well the assistant performed at
what it could control. Use the FULL 1-5 scale and discriminate — do NOT default to 5.
Reserve 5 for genuinely excellent answers; if there is ANY notable shortcoming, use 4
or lower. Calibrate against these anchors:
- 5 = excellent: fully and correctly answered, efficient, clearly synthesized; a domain
  expert would not meaningfully improve it.
- 4 = good: answered well but with a minor flaw (a small omission, some verbosity, a
  slightly inefficient tool path, a caveat that should have been stated).
- 3 = adequate/mixed: only partially answered, or has notable gaps; some real value but
  leaves a knowledgeable user wanting more.
- 2 = poor: largely failed to address the question, or significant inaccuracy/confusion,
  though some relevant content exists.
- 1 = very poor: did not answer, fundamentally wrong, or unusable.
Apply the anchors regardless of disposition, with these adjustments:
- Do NOT lower the score because the user's request was out_of_scope, unfinished, or
  weird_or_unclear — if the assistant handled it gracefully (clearly stated the limit
  and pointed elsewhere where possible), that is a 4 or 5.
- For technical_failure, a low score (1-2) is appropriate because the user was not
  served, even though it is not the assistant's fault.
- For agent_failure, score 1-2. For good_answer, score by the anchors above — most good
  answers with any minor flaw are a 4, not a 5.

Then assess:
1. Did the assistant answer the user's question? (yes/partially/no)
2. Was the information accurate and relevant? (yes/mostly/no)
3. Were tool calls efficient (no unnecessary calls)? (yes/mostly/no)
4. Did the conversation reach a natural conclusion? (yes/no)

Finally list what went wrong, split three ways. Each entry is one short sentence
naming a single concrete problem or strength:
- "issues": real problems that affected the answer, attributable to the assistant or
  to a tool/infrastructure failure the user experienced. Do NOT put strengths,
  "otherwise good" remarks, or well-handled limitations here.
- "nits": minor blemishes that did not affect the answer (a typo, rounding, one
  redundant call, slight verbosity). Keep these out of "issues".
- "strengths": notable things done well. Empty if nothing stands out.
Do not report that the conversation shown to you is elided or truncated — the
"[... chars elided ...]" markers are a display artifact of this evaluation, not
something the user saw. Greeting or test messages with no real question are not
an issue either; the disposition already captures them.

Respond with JSON:
{{"disposition": "good_answer|agent_failure|technical_failure|out_of_scope|unfinished|weird_or_unclear", "answered": "yes|partially|no", "accurate": "yes|mostly|no", "efficient": "yes|mostly|no", "concluded": "yes|no", "quality_score": 1-5, "issues": ["..."], "nits": ["..."], "strengths": ["..."]}}

Conversation:
{conversation}
"""

# PAIRED, not absolute. QUALITY_ASSESSMENT_PROMPT above scores one conversation on a 1-5
# rubric, which is the right instrument for sampling and for tracking quality over time and
# the wrong one for "is arm B worse than arm A on THIS turn": the rubric is coarse, the
# expected between-arm difference is small, and per-question difficulty dominates the score.
# Judging the two answers side by side cancels that difficulty, which is why A/B evaluations
# are done pairwise (genetics-results-suite-4h6.72).
#
# The two answers are labelled only "Answer 1" / "Answer 2". Nothing in this prompt names an
# arm, a tool profile, a model or a mechanism, and the caller shows FINAL ANSWERS ONLY — see
# `pairwise_judge.final_answer_text` for why tool traces are withheld.
PAIRWISE_ANSWER_JUDGE_PROMPT = """\
You are comparing two candidate answers from an AI genetics assistant to the SAME user
question. Today's date is {today}.

Decide which answer serves the user better, or whether they are equally good.

IMPORTANT — what you can and cannot verify:
- You are shown only the answers, NOT the raw output of the data tools behind them. The
  assistant queries REAL genetics databases (FinnGen, GWAS summary statistics, BigQuery
  and others). Specific, precise figures (sample sizes, variant counts, p-values, effect
  sizes) and recent dates are EXPECTED and are almost always real tool output.
- Do NOT prefer an answer merely because it contains more numbers, tables or citations,
  and do NOT call something fabricated merely because it is precise or you cannot verify
  it. Only treat a claim as unsupported when there is clear internal evidence (the answer
  contradicts itself, or reports a result it also says it could not obtain).
- Dates in 2025 or early 2026 are in the PAST relative to today's date above.

JUDGE ON, in this order of weight:
1. Correctness and internal consistency of what is claimed.
2. Whether it actually answers what was asked, including every part of a multi-part
   question.
3. Whether conclusions are supported by what the answer itself presents, and whether real
   limitations are stated.
4. Usefulness to a genetics researcher: is there an interpretation, or only raw output?

EXPLICITLY DO NOT JUDGE ON:
- Length. A longer answer is not a better answer; a short, complete, correct answer beats
  a long one that pads. Only prefer the longer answer when the extra content is content
  the question asked for.
- Formatting, tables, headings, bullets, markdown polish or tone.
- Confidence of phrasing. Appropriate hedging about a genuine uncertainty is a merit, not
  a fault; unnecessary hedging about something the answer establishes is a fault.
- How the answer was produced, or any hint of what machinery was used. If one answer
  mentions its own mechanism and the other does not, that is NOT a quality difference and
  must not move your verdict.
- Which answer is shown first. Position carries no information.

CALL A TIE whenever the two answers are of comparable quality, including when they differ
in style, in length or in emphasis but not in substance, and when both are equally wrong
or equally unhelpful. A tie is a real, expected outcome — do not invent a winner.

Respond with JSON only:
{{"verdict": "1|2|tie", "margin": "none|slight|clear", "reason": "one or two sentences naming the specific difference that decided it"}}
Use "margin": "none" exactly when "verdict" is "tie".

USER QUESTION:
{question}
{prior_context}
--- ANSWER 1 ---
{answer_1}

--- ANSWER 2 ---
{answer_2}
"""

# Unlike QUALITY_ASSESSMENT_PROMPT, this judge IS shown the literature tool results, because
# the failures it looks for (a claim absent from the record, a paper's hedge dropped, a
# Perplexity summary sentence relayed as a paper's finding) are invisible without them. The
# eight categories and the counter-example notion are those of the human review this judge
# is calibrated against (genetics-results-suite docs/research/literature-critical-evaluation/
# rubric.md); renumbering them breaks the agreement score in literature_judge.py.
LITERATURE_EVIDENCE_JUDGE_PROMPT = """\
You are auditing ONE turn of a genetics-research assistant ("FinnGenie") for how critically
it evaluated SCIENTIFIC LITERATURE. Today's date is {today}.

The assistant has loaded genetics results (FinnGen/UKBB GWAS, fine-mapping credible sets with
PIPs, QTL colocalization, burden tests, MGI mouse phenotypes) AND literature tools:
`search_scientific_literature` (backend `perplexity` returns an AI-generated `summary` with
[n] markers indexing its own `search_results` list, plus `records` hydrated from Europe PMC
with title/authors/journal/year/abstract/pmid/doi/is_preprint; backend `europepmc` returns
structured records only) and `launch_subagents` with the `literature_review` skill (a
subagent's written digest of its own searches).

You are shown the user's question, the assistant's final answer, and the FULL literature tool
results of this turn (and, marked as such, of earlier turns of the same conversation). You
are NOT shown the loaded genetics data; treat genetics numbers in the answer as real tool
output. Judge the LITERATURE claims only.

A senior user's complaint that motivates this audit: the assistant applies careful
guidelines to the loaded genetics results but "trusts at face value" conclusions from papers
and from Perplexity ("it said here's strong evidence of demyelination for schizophrenia that
was based on like 5 people").

Four failure patterns recur. Look for each explicitly:
A. The evidence rubric transfers to genetics papers and stops there: candidate-gene studies
   get sized and down-weighted, but functional, mechanistic, clinical and review papers are
   relayed as one-line findings with no model system, n, null arm or replication, and are
   sometimes COUNTED as independent lines of evidence.
B. Retrieved and recalled literature are indistinguishable: sentences from the model's
   memory welded to a search citation that does not contain them; "literature" sections
   written with no search; numbers attributed to a paper whose retrieved text has no such
   number.
C. Perplexity's assertions survive, its hedges do not: "not shown", "likely", "preprint",
   "in the provided results" dropped; "suggests" escalated to "confirms"/"establishes";
   Perplexity's reference list re-emitted as the assistant's citations; non-primary web
   pages given the weight of a paper.
D. Caveats live in the middle and die before the bottom line: "unverified"/"preprint" in a
   table, then counted as firm support in the conclusion; a blanket "AI-generated summaries
   were not verified" disclaimer that does not change the verdict it sits under.

CATEGORIES (a finding may carry more than one):
1. Face-value pass-through: a paper's or Perplexity's claim restated as established without
   sample size, study design (case report, n<50, in vitro, single mouse line, cell line,
   candidate-gene association, preprint, review), replication status, or effect size.
2. Evidence tier mismatch: the loaded genetics is hedged carefully (PIP, p thresholds, LD,
   winner's curse) but a literature claim is stated with equal or greater confidence than its
   study warrants. Quote both sides.
3. Source does not support claim: compare the RECORD to the answer. Overstated, wrong
   direction, wrong species, wrong phenotype, number not in the record, citation not present
   in any result shown, or a Perplexity summary sentence presented as the paper's finding.
4. Perplexity summary treated as primary source: the AI summary is relayed rather than the
   papers it cites being checked against the records.
5. Old candidate-gene / small-n association reported as support for a gene-disease link
   without noting such studies mostly do not replicate.
6. Literature contradicts the loaded data (or vice versa) and the answer does not reconcile
   or flag it.
7. Missing provenance: a literature claim with no citation; a citation with no PMID/DOI/link;
   or memory-derived and search-derived statements blended so a reader cannot separate them.
   A citation that appears in NO result shown to you (this turn or earlier) is category 7,
   and also 3 if it is attached to a specific claim.
8. Uncritical acceptance of review articles / consensus statements as the evidence itself.

Rules:
- Report only literature claims (papers, reviews, Perplexity or subagent text, textbook facts
  presented as literature). Do not audit the genetics analysis itself.
- Quote the claim VERBATIM from the answer (at most 3 lines). Quote the record you checked it
  against VERBATIM from the tool result (at most 3 lines), or write "not in any result shown"
  / "no search in this turn" when that is the point.
- One finding per distinct claim. Several claims with the same defect in one table may be
  reported as one finding quoting the most consequential one.
- Severity: high = could mislead a scientific or clinical decision (a central conclusion rests
  on it); medium = a material overstatement a careful reader would want corrected; low =
  a real but peripheral lapse.
- Do not flag what a critical reader would accept: well-known textbook facts stated as
  background with appropriate weight, a claim that the answer itself sizes and hedges, a
  record the answer correctly describes as weak.
- A finding is a DEFECT. Every finding carries at least one category. A passage you judge
  correctly handled is never a finding, not even a "low" one "for completeness"; it is a
  counter-example.
- Read the whole passage before flagging it. A claim whose own sentence or table row states
  the study's design, n, species, preprint status, or that it rests on the Perplexity
  summary and was not checked, is appraised, not passed through. Flag it only if a LATER
  statement (the conclusion, a verdict table, "independent support") drops that caveat, and
  then quote the later statement as the claim. A caveat attached to the specific claim
  ("the summary attributes this to X; not verifiable from the abstract") discloses it; a
  blanket disclaimer elsewhere in the answer does not.
- Minor imprecision inside a passage that already sizes and hedges the study (a rounded n,
  one arm's n given for the whole study) is not a finding unless it changes how much the
  evidence should weigh.
- Also record COUNTER-EXAMPLES: places where the answer DID appraise a paper (stated n or
  design, called out a preprint or case report, down-weighted a candidate-gene study,
  separated Perplexity's summary from the records, reconciled literature with the loaded
  data, said a search found nothing rather than filling from memory). Give the category the
  good behaviour addresses.
- Also flag OVERCAUTIOUS passages: hedging that withholds or undermines a conclusion the
  shown evidence clearly supports, or refuses to use a sound record. And BOILERPLATE
  caveats: generic disclaimers ("AI-generated, not independently verified") that do not
  change any conclusion they sit beside.
- If the answer contains no literature claims, return empty lists.

Respond with JSON only, no prose:
{{"findings": [{{"claim": "...", "record": "...", "categories": [1, 7], "severity": "high|medium|low", "problem": "one or two sentences"}}],
 "counter_examples": [{{"claim": "...", "categories": [5], "why": "one sentence"}}],
 "overcautious": [{{"claim": "...", "why": "one sentence"}}],
 "boilerplate_caveats": [{{"claim": "..."}}]}}

USER QUESTION:
{question}

ASSISTANT ANSWER:
{answer}

LITERATURE TOOL RESULTS OF THIS TURN:
{literature_results}

LITERATURE TOOL RESULTS OF EARLIER TURNS IN THIS CONVERSATION (context; the answer may cite them):
{earlier_results}
"""
