"""Prompt templates. One per feature module, kept in one place so the
"answer only from context, cite everything, abstain when unsure" contract is
consistent across Q&A, risk analysis, conflict detection, etc.
"""

SYSTEM_PROMPT = """\
You are LegalMind, a document-grounded legal research assistant. You are not \
a lawyer and you do not give legal advice — you help a user understand what \
their documents say.

Rules:
1. Answer ONLY using the numbered context chunks provided. Never use outside
   knowledge of law, even if you are confident it is correct.
2. Every factual claim in your answer must be traceable to at least one
   chunk_id from the context. Cite it.
3. If the context does not contain enough information to answer, set
   insufficient_context=true and say so plainly — do not guess.
4. Quote exact language when the user's question is about a specific number,
   date, deadline, or defined term.
"""

QA_USER_PROMPT = """\
Question: {question}

Context chunks:
{context_block}

Return a CitedAnswer.
"""

QUERY_REWRITE_PROMPT = """\
Given the legal research question below, produce up to 3 alternative
phrasings or related search queries that would help retrieve relevant
clauses from a contract (synonyms, related legal terms, likely section
headings). One per line, no numbering, no extra commentary.

Question: {query}
"""

RISK_ANALYSIS_PROMPT = """\
Review the following clause for risk to the party receiving/signing this \
document. Classify it as RED (highly unfavorable / unusual), AMBER \
(worth negotiating), or GREEN (standard/fair), and give a one-sentence \
explanation a non-lawyer can understand.

Clause ({section_path}, page {page}):
{clause_text}
"""

CONFLICT_DETECTION_PROMPT = """\
Below are clauses of the same type ("{clause_type}") extracted from \
different documents. Identify any conflicts or inconsistencies between them \
(e.g. different notice periods, different governing law, contradictory \
obligations). For each conflict, name the two source chunk_ids and explain \
the contradiction in one or two sentences. If there is no conflict, say so.

Clauses:
{clauses_block}
"""

PLAYBOOK_COMPARE_PROMPT = """\
Compare the clause from the uploaded document against the standard/playbook \
version below. Summarise what changed (redline-style) and explain in plain \
English why the deviation matters to the signing party.

Standard clause:
{standard_clause}

Document clause ({section_path}, page {page}):
{document_clause}
"""

PLAIN_ENGLISH_PROMPT = """\
Rewrite the following legal clause in plain English for a non-lawyer. Keep \
all obligations, deadlines, and numbers accurate — do not soften or omit \
substantive terms, only the legalese phrasing.

Clause:
{clause_text}
"""

MEMO_EXPORT_PROMPT = """\
Produce a structured legal summary memo for the following document(s): \
parties, key dates, obligations, identified risks, and open questions a \
reviewer should follow up on. Base every item on the provided context and \
cite chunk_ids.

Context:
{context_block}
"""
