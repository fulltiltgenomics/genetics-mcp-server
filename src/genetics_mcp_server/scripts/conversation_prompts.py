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
ISSUE_CATEGORIES = [
    ("incomplete_answer", "Answered only part of the question or omitted requested detail"),
    ("missed_data_source", "Failed to use or find available data; claimed no data when data exists; queried the wrong source"),
    ("inaccurate_claim", "Stated something factually wrong, misleading, or unsupported"),
    ("fabrication", "Invented data, results, numbers, citations, or tool output that was not actually returned"),
    ("inefficient_tool_use", "Redundant, repeated, or unnecessary tool calls"),
    ("tool_failure_handling", "A tool errored or returned nothing and the assistant gave up or failed to recover"),
    ("misunderstood_question", "Misinterpreted what the user was actually asking"),
    ("no_conclusion", "Did not synthesize results into a clear answer; left the conversation hanging"),
    ("missing_interpretation", "Returned raw data or tables without interpreting them for the user"),
    ("formatting_readability", "Poor formatting or hard to read; dumped raw tables"),
    ("overcautious", "Unnecessarily refused, hedged, or added excessive caveats"),
    ("other", "A genuine issue that does not fit any category above"),
]

ISSUE_CATEGORIZATION_PROMPT = """\
You are grouping individual quality issues found across a genetics AI assistant's
conversations into recurring underlying problem categories.

Assign each issue below to exactly ONE category from this list:
{categories}

Each issue is prefixed with a numeric ID. Respond with a JSON array, one object
per issue, using only category names from the list above:
[{{"id": 0, "category": "..."}}]
If an issue genuinely fits none of the categories, use "other".

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

Respond with JSON:
{{"disposition": "good_answer|agent_failure|technical_failure|out_of_scope|unfinished|weird_or_unclear", "answered": "yes|partially|no", "accurate": "yes|mostly|no", "efficient": "yes|mostly|no", "concluded": "yes|no", "quality_score": 1-5, "issues": ["list of problems if any"]}}

Conversation:
{conversation}
"""
