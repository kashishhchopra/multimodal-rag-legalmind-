"""Date-ordered event extraction: every sentence containing a date, with the date normalised to ISO."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import List, Optional

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"], 1)}
_MON = "|".join(MONTHS)
_PATTERNS = [
    re.compile(rf"\b(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\s+(?P<m>{_MON})[,]?\s+(?P<y>\d{{4}})\b", re.I),      # 25 September 2024
    re.compile(rf"\b(?P<m>{_MON})\s+(?P<d>\d{{1,2}})(?:st|nd|rd|th)?[,]?\s+(?P<y>\d{{4}})\b", re.I),      # September 25, 2024
    re.compile(r"\b(?P<d>\d{1,2})[./-](?P<m>\d{1,2})[./-](?P<y>\d{4})\b"),                                 # 25.09.2024 (day first, Indian style)
]
_SENT = re.compile(r"(?<=[.;])\s+(?=[A-Z0-9(])")


@dataclass
class Event:
    date: str          # ISO yyyy-mm-dd
    text: str
    page_number: int
    chunk_id: str


def _to_date(m: re.Match) -> Optional[date]:
    try:
        mo = m.group("m")
        month = MONTHS[mo.lower()] if mo.isalpha() else int(mo)
        return date(int(m.group("y")), month, int(m.group("d")))
    except (ValueError, KeyError):
        return None


def extract_timeline(chunks: List[dict], max_events: int = 60) -> List[Event]:
    """chunks: [{id, text, page_number}]. Returns events sorted by date, de-duplicated by (date, sentence)."""
    seen, events = set(), []
    for c in chunks:
        for sent in _SENT.split(re.sub(r"\s+", " ", c["text"])):
            for pat in _PATTERNS:
                for m in pat.finditer(sent):
                    d = _to_date(m)
                    if d and 1800 <= d.year <= 2100 and (d, sent[:80]) not in seen:
                        seen.add((d, sent[:80]))
                        events.append(Event(d.isoformat(), sent.strip()[:400], int(c.get("page_number", 0)), c["id"]))
    events.sort(key=lambda e: (e.date, e.page_number))
    return events[:max_events]
