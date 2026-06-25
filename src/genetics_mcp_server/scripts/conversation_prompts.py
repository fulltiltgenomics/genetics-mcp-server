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

Users may attach files (uploaded TSVs, images, etc.). An attachment is shown as a
line like "[User attached file(s): NAME (type, size)]". The assistant had access to
the FULL contents of any attached file even though those contents are not reproduced
below. So when a user references "the results" or "the file" and an attachment is
present, the assistant is NOT fabricating by analyzing it — treat the attached data
as legitimately available context, not invented.

Given the conversation below, assess:
1. Did the assistant answer the user's question? (yes/partially/no)
2. Was the information accurate and relevant? (yes/mostly/no)
3. Were tool calls efficient (no unnecessary calls)? (yes/mostly/no)
4. Did the conversation reach a natural conclusion? (yes/no)

Respond with JSON:
{{"answered": "yes|partially|no", "accurate": "yes|mostly|no", "efficient": "yes|mostly|no", "concluded": "yes|no", "quality_score": 1-5, "issues": ["list of problems if any"]}}

Conversation:
{conversation}
"""
