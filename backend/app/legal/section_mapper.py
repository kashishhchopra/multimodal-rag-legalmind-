"""Old-to-new criminal-law section mapper (IPC->BNS, CrPC->BNSS, Evidence Act->BSA).

Judgments before 1 July 2024 cite the old codes; users now ask with either. This module
detects section references in a query and appends the counterpart so both old and new
text are retrievable. Mapping data is a curated subset (see section_data.json) that can be
extended with the full official table dropped in data_lake/bns_mapping/*.csv.
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

_HERE = Path(__file__).resolve().parent
_LAKE_CSV_DIR = _HERE.parents[2] / "data_lake" / "bns_mapping"

# code key -> (old code label, new code label, json table name)
_CODES = {
    "ipc": ("IPC", "BNS", "ipc_to_bns"),
    "crpc": ("CrPC", "BNSS", "crpc_to_bnss"),
    "iea": ("Evidence Act", "BSA", "iea_to_bsa"),
}
_NEW_TO_OLD = {"bns": "ipc", "bnss": "crpc", "bsa": "iea"}

_ALIASES = {
    "ipc": "ipc", "indian penal code": "ipc", "bns": "bns", "bharatiya nyaya sanhita": "bns",
    "crpc": "crpc", "cr.p.c": "crpc", "cr.p.c.": "crpc", "code of criminal procedure": "crpc",
    "bnss": "bnss", "bharatiya nagarik suraksha sanhita": "bnss",
    "evidence act": "iea", "indian evidence act": "iea", "iea": "iea",
    "bsa": "bsa", "bharatiya sakshya adhiniyam": "bsa",
}
_ALIAS_RE = "|".join(sorted((re.escape(a) for a in _ALIASES), key=len, reverse=True))
# "Section 420 IPC", "s. 302 of the Indian Penal Code", "Sections 34 and 120B IPC"
_REF = re.compile(
    rf"(?:sections?|secs?\.?|ss?\.)\s*(?P<nums>\d+[A-Za-z]?(?:\([\w]+\))*(?:\s*(?:,|and|&|/)\s*\d+[A-Za-z]?(?:\([\w]+\))*)*)\s*(?:of\s+(?:the\s+)?)?(?P<code>{_ALIAS_RE})\b",
    re.IGNORECASE,
)
_NUM = re.compile(r"\d+[A-Za-z]?")


@dataclass(frozen=True)
class Mapping:
    old_code: str
    old_section: str
    new_code: str
    new_section: str
    title: str


@lru_cache(maxsize=1)
def _tables() -> Dict[str, Dict[str, Tuple[str, str]]]:
    data = json.loads((_HERE / "section_data.json").read_text())
    tables = {k: {s: (v[0], v[1]) for s, v in data[k].items()} for k in ("ipc_to_bns", "crpc_to_bnss", "iea_to_bsa")}
    csv_to_table = {"IPC": "ipc_to_bns", "CRPC": "crpc_to_bnss", "IEA": "iea_to_bsa", "EVIDENCE ACT": "iea_to_bsa"}
    if _LAKE_CSV_DIR.is_dir():
        for path in _LAKE_CSV_DIR.glob("*.csv"):
            with path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    t = csv_to_table.get(row["old_code"].strip().upper())
                    if t:
                        tables[t][row["old_section"].strip().upper()] = (row["new_section"].strip(), row.get("title", "").strip())
    return tables


def old_to_new(code: str, section: str) -> Mapping | None:
    key = _ALIASES.get(code.lower().strip())
    if key not in _CODES:
        return None
    old_label, new_label, table = _CODES[key]
    hit = _tables()[table].get(section.strip().upper())
    return Mapping(old_label, section.upper(), new_label, hit[0], hit[1]) if hit else None


def new_to_old(code: str, section: str) -> List[Mapping]:
    """Reverse lookup; may return several old sections (BNS 318 covers IPC 415 and 420)."""
    old_key = _NEW_TO_OLD.get(_ALIASES.get(code.lower().strip(), ""))
    if not old_key:
        return []
    old_label, new_label, table = _CODES[old_key]
    target = section.strip().lower()
    base = lambda s: re.sub(r"\(.*", "", s).lower()  # "318(4)" matches a query for "318"
    return [
        Mapping(old_label, old_s, new_label, new_s, title)
        for old_s, (new_s, title) in _tables()[table].items()
        if new_s.lower() == target or base(new_s) == target
    ]


def find_references(text: str) -> List[Tuple[str, str]]:
    """All (code_key, section) pairs mentioned in text, e.g. [('ipc','420')]."""
    refs = []
    for m in _REF.finditer(text):
        code = _ALIASES[m.group("code").lower()]
        refs += [(code, n.upper()) for n in _NUM.findall(m.group("nums"))]
    return refs


def expand_query(query: str) -> Tuple[str, List[str]]:
    """Append counterpart sections so 'Section 420 IPC' also retrieves BNS text.

    Returns (expanded_query, human-readable notes). Unmapped sections are left untouched.
    """
    extras: List[str] = []
    for code, sec in find_references(query):
        if code in _CODES:
            m = old_to_new(code, sec)
            if m:
                extras.append(f"{m.old_code} {m.old_section} corresponds to {m.new_code} {m.new_section} ({m.title})")
        elif code in _NEW_TO_OLD:
            for m in new_to_old(code, sec):
                extras.append(f"{m.new_code} {sec} corresponds to {m.old_code} {m.old_section} ({m.title})")
    return (f"{query} ({'; '.join(extras)})" if extras else query), extras
