"""Clause extraction + risk flagging over a document's indexed text.

Uses the CUAD TF-IDF/SVM classifier trained by scripts/train_clause_classifier.py (macro-F1 0.66 on
the CUAD test split; see app/models/clause_classifier/report.json). Falls back to an empty result with a
clear warning if the model has not been trained. Confidence is the SVM margin, not a probability.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..config import BACKEND_DIR
from .risk import Risk, assess

MODEL_PATH = BACKEND_DIR / "app" / "models" / "clause_classifier" / "cuad.joblib"
_lock = threading.Lock()


@dataclass
class Clause:
    category: str
    text: str
    page_number: int
    chunk_id: str
    margin: float
    risk: Risk


@lru_cache(maxsize=1)
def _model():
    import joblib

    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


def model_available() -> bool:
    return MODEL_PATH.exists()


def extract_clauses(chunks: List[dict], min_margin: float = 0.0, exclude: Optional[set] = None) -> List[Clause]:
    """chunks: [{id, text, page_number}]. Returns classified, risk-assessed clauses, riskiest first."""
    model = _model()
    if model is None or not chunks:
        return []
    exclude = exclude or {"None", "Document Name", "Parties", "Agreement Date", "Effective Date", "Expiration Date"}
    texts = [c["text"] for c in chunks]
    with _lock:
        scores = model.decision_function(texts)
        classes = model.classes_
    out: List[Clause] = []
    for c, row in zip(chunks, scores):
        k = int(np.argmax(row))
        cat, margin = str(classes[k]), float(row[k])
        if margin < min_margin or cat in exclude:
            continue
        out.append(Clause(cat, c["text"], int(c.get("page_number", 0)), c["id"], round(margin, 3), assess(cat, c["text"])))
    out.sort(key=lambda x: ("low", "medium", "high").index(x.risk.level), reverse=True)
    return out
