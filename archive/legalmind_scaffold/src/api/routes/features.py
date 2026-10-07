"""WOW/STRETCH feature endpoints: conflict detection, risk analysis,
playbook comparison, plain-English translation, memo export.
"""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from src.features.conflict_detection import ConflictFinding, detect_conflicts
from src.features.memo_export import Memo, generate_memo
from src.features.playbook_compare import PlaybookDeviation, compare_to_playbook
from src.features.risk_analysis import RiskFinding, analyze_risks
from src.features.translate import translate_to_plain_english

router = APIRouter()


class ConflictRequest(BaseModel):
    clause_type: str
    doc_ids: list[str]


@router.post("/conflicts", response_model=list[ConflictFinding])
def find_conflicts(request: ConflictRequest) -> list[ConflictFinding]:
    return detect_conflicts(request.clause_type, request.doc_ids)


@router.get("/risks/{doc_id}", response_model=list[RiskFinding])
def get_risks(doc_id: str) -> list[RiskFinding]:
    return analyze_risks(doc_id)


class PlaybookRequest(BaseModel):
    clause_type: str
    doc_id: str


@router.post("/playbook-compare", response_model=PlaybookDeviation | None)
def playbook_compare(request: PlaybookRequest) -> PlaybookDeviation | None:
    return compare_to_playbook(request.clause_type, request.doc_id)


class TranslateRequest(BaseModel):
    clause_text: str


@router.post("/translate")
def translate(request: TranslateRequest) -> dict[str, str]:
    return {"plain_english": translate_to_plain_english(request.clause_text)}


@router.get("/memo/{doc_id}", response_model=Memo)
def memo(doc_id: str) -> Memo:
    return generate_memo(doc_id)
