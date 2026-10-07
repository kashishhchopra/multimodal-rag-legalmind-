"""Section-wise parsing of Indian Act PDFs (BNS, BNSS, BSA, ... from India Code / MHA).

Sections are found as a monotonic chain 1, 2, 3 ... at line starts, which is robust to sub-clause numbers and
to the bilingual BNSS gazette (Hindi glyph noise is ignored). Each section becomes one chunk (long sections are
split on sub-clause boundaries) with act/section metadata so a query for 'BNS 318' can filter exactly.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

import fitz

_HEAD = re.compile(r"(?m)^\s*(\d{1,3})\.\s*(?=[A-Z(])")


@dataclass
class StatuteChunk:
    chunk_id: str
    act: str
    section: str
    text: str
    page_number: int


def parse_sections(pdf_path: str, act: str, max_chars: int = 1800) -> List[StatuteChunk]:
    doc = fitz.open(pdf_path)
    pages, offsets, full = [], [], ""
    for i, p in enumerate(doc, 1):
        offsets.append((len(full), i))
        full += p.get_text() + "\n"
    chain, nxt = [], 1
    for m in _HEAD.finditer(full):
        if int(m.group(1)) == nxt:
            chain.append((m.start(), nxt))
            nxt += 1
    page_at = lambda pos: max((pg for off, pg in offsets if off <= pos), default=1)
    out: List[StatuteChunk] = []
    for i, (start, n) in enumerate(chain):
        end = chain[i + 1][0] if i + 1 < len(chain) else len(full)
        body = re.sub(r"[ \t]+", " ", full[start:end]).strip()
        parts = [body] if len(body) <= max_chars else _split_subclauses(body, max_chars)
        for j, part in enumerate(parts):
            out.append(StatuteChunk(f"{act}_s{n}_{j}", act, str(n), part, page_at(start)))
    return out


def _split_subclauses(text: str, max_chars: int) -> List[str]:
    pieces, cur = [], ""
    for seg in re.split(r"(?m)(?=^\(\d+\)\s)", text):
        if cur and len(cur) + len(seg) > max_chars:
            pieces.append(cur.strip())
            cur = seg
        else:
            cur += seg
    pieces.append(cur.strip())
    return [p[: max_chars * 2] for p in pieces if p]
