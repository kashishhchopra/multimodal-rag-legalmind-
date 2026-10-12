"""Rule-based PII redaction for outputs and logs.

Covers structured identifiers (Aadhaar, PAN, Indian mobile numbers, e-mail, bank-account-like numbers after an
"account" cue, IFSC). It does NOT detect personal names, addresses or the identity of victims/minors: that needs an NER
model plus judgment-specific rules, and masking must be reviewed by a person before publication of any judgment text.
"""
from __future__ import annotations

import re
from typing import List, Tuple

_RULES: List[Tuple[str, re.Pattern]] = [
    ("AADHAAR", re.compile(r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b")),
    ("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)")),
    ("IFSC", re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b")),
    ("ACCOUNT", re.compile(r"(?i)(account\s(?:no\.?|number)[\s:.-]{0,4})\d{9,18}")),
]


def redact(text: str) -> Tuple[str, dict]:
    """Returns (redacted text, {label: count})."""
    counts: dict = {}
    for label, rx in _RULES:
        repl = (r"\1[ACCOUNT REDACTED]") if label == "ACCOUNT" else f"[{label} REDACTED]"
        text, n = rx.subn(repl, text)
        if n:
            counts[label] = n
    return text, counts
