"""Generate (question, relevant passage) pairs from the synthetic legal corpus.

Two consumers:
  1. `scripts/finetune_legal_embeddings.py` — uses (question, passage) pairs as
     positive examples to fine-tune the BGE text-embedding model on legal
     phrasing (MultipleNegativesRankingLoss: in-batch negatives).
  2. `eval/legal_eval_dataset.json` (generated once, then checked in) — uses
     the same questions together with the manifest's page numbers as
     ground-truth retrieval labels (Precision@K / Recall@K).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .content import CaseJudgment, Contract, StatuteSection, generate_cases, generate_contracts, generate_statute_sections


@dataclass
class QAPair:
    question: str
    passage: str
    doc_title: str  # links back to the manifest entry for page lookup


def _contract_qa(contract: Contract) -> List[QAPair]:
    pairs = []
    parties_str = ", ".join(f"{role} {name}" for role, name in contract.parties)
    pairs.append(QAPair(
        f"Who are the parties to the {contract.doc_type.lower()} titled '{contract.title}'?",
        f"This {contract.doc_type} is between {parties_str}.", contract.title))
    for term, value in contract.key_terms:
        pairs.append(QAPair(
            f"What is the {term.lower()} in {contract.title}?",
            f"{term}: {value} (from {contract.title}).", contract.title))
    for head, body in contract.clauses:
        pairs.append(QAPair(
            f"What does the {head} clause say in a {contract.doc_type.lower()}?",
            body, contract.title))
        pairs.append(QAPair(f"Explain the {head} clause.", body, contract.title))
    return pairs


def _case_qa(case: CaseJudgment) -> List[QAPair]:
    pairs = [
        QAPair(f"What was the outcome in {case.case_name} ({case.citation})?",
               f"Outcome: {case.outcome}. {case.holding}", case.case_name),
        QAPair(f"What were the facts of the case {case.case_name}?", case.facts, case.case_name),
        QAPair(f"What did the court hold in {case.case_name}?", case.holding, case.case_name),
        QAPair(f"Which court decided {case.case_name} and on what date?",
               f"{case.case_name} was decided by the {case.court} on {case.date}, coram {case.judge}.",
               case.case_name),
    ]
    for issue in case.issues:
        pairs.append(QAPair(f"What was the legal issue in {case.case_name}?", issue, case.case_name))
    return pairs


def _statute_qa(section: StatuteSection) -> List[QAPair]:
    return [
        QAPair(f"What does Section {section.section_no} of the {section.act_name} say?",
               section.body, section.act_name),
        QAPair(f"Explain {section.heading} under the {section.act_name}.", section.body, section.act_name),
    ]


def generate_qa_pairs(n_contracts_per_type: int = 3, n_cases: int = 24) -> List[QAPair]:
    pairs: List[QAPair] = []
    for contract in generate_contracts(n_per_type=n_contracts_per_type):
        pairs.extend(_contract_qa(contract))
    for case in generate_cases(n=n_cases):
        pairs.extend(_case_qa(case))
    for section in generate_statute_sections():
        pairs.extend(_statute_qa(section))
    return pairs


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("legal_qa_pairs.json")
    pairs = generate_qa_pairs()
    out.write_text(json.dumps([p.__dict__ for p in pairs], indent=2), encoding="utf-8")
    print(f"Wrote {len(pairs)} QA pairs -> {out}")
