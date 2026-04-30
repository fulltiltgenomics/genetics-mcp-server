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

QUALITY_ASSESSMENT_PROMPT = """\
You are evaluating the quality of an AI genetics assistant's conversation.

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
