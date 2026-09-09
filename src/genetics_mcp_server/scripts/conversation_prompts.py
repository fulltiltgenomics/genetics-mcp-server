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
