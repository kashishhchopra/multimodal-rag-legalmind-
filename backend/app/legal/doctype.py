"""Document-type routing hint (judgment | statute | contract); see scripts/train_doctype_router.py for scope and limits."""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, Optional

from ..config import BACKEND_DIR

MODEL_PATH = BACKEND_DIR / "app" / "models" / "doctype_router" / "router.joblib"
TOOLS = {  # which analysis tools make sense for each type
    "judgment": ["summary", "timeline", "citations"],
    "contract": ["clauses", "compare", "timeline"],
    "statute": ["section_lookup"],
}


@lru_cache(maxsize=1)
def _model():
    import joblib

    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


def classify(text: str) -> Optional[Dict]:
    m = _model()
    if m is None or not text.strip():
        return None
    proba = m.predict_proba([text[:1500]])[0]
    best = int(proba.argmax())
    label = str(m.classes_[best])
    return {"type": label, "confidence": round(float(proba[best]), 3), "suggested_tools": TOOLS.get(label, []),
            "note": "Routing hint from a 3-class model; notices, affidavits and other types are not covered."}
