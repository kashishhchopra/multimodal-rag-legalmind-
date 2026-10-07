"""Structure-aware chunking for legal text.

Fixed-size token windows are the wrong default for contracts: they cut
clauses in half and destroy the hierarchy (Article > Section > Clause >
sub-clause) that both a lawyer and a re-ranker rely on. This splitter walks
numbering patterns ("1.", "1.1", "(a)", "Article 12", "Section 12(b)") to find
clause boundaries first, and only falls back to a recursive character
splitter for prose that carries no numbering (e.g. recitals, free-text
notices).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.config import get_settings

# Matches common legal numbering styles at the start of a line:
#   "1.", "1.1", "1.1.1", "(a)", "(i)", "Article 12", "Section 12(b)"
_CLAUSE_HEADING_RE = re.compile(
    r"""^\s*(
        (?:ARTICLE|Article|SECTION|Section|Clause)\s+\d+[A-Za-z]?(\(\w+\))?
        |\d+(\.\d+)*\.?\s
        |\([a-zA-Z0-9]+\)\s
    )""",
    re.VERBOSE,
)


@dataclass
class ClauseSpan:
    text: str
    section_path: str | None
    start_char: int
    end_char: int


def split_into_clauses(page_text: str, *, parent_section_path: str | None = None) -> list[ClauseSpan]:
    """Split one page's text into clause-level spans.

    A running `section_path` stack tracks nesting depth so a sub-clause like
    "(b)" under "Section 12" is tagged "12 > 12(b)" rather than losing its
    parent context.
    """
    lines = page_text.splitlines(keepends=True)
    spans: list[ClauseSpan] = []
    path_stack: list[str] = [parent_section_path] if parent_section_path else []

    current_lines: list[str] = []
    current_heading: str | None = None
    char_offset = 0
    span_start = 0

    def flush():
        if current_lines:
            text = "".join(current_lines).strip()
            if text:
                spans.append(ClauseSpan(
                    text=text,
                    section_path=" > ".join(p for p in path_stack if p) or None,
                    start_char=span_start,
                    end_char=char_offset,
                ))

    for line in lines:
        match = _CLAUSE_HEADING_RE.match(line)
        if match:
            flush()
            heading = match.group(1).strip()
            # A new top-level heading replaces the stack; a sub-heading like
            # "(b)" appends under whatever numbered section came before it.
            if re.match(r"^(ARTICLE|Article|SECTION|Section|Clause)\b", heading) or re.match(r"^\d", heading):
                path_stack = [heading]
            else:
                path_stack = (path_stack[:1] if path_stack else []) + [heading]
            current_heading = heading
            current_lines = [line]
            span_start = char_offset
        else:
            current_lines.append(line)
        char_offset += len(line)

    flush()

    if not spans:
        # No numbering detected at all (e.g. a recital block) — fall back to
        # a recursive character split so we still return usable chunks.
        return _fallback_recursive_split(page_text, parent_section_path)

    return spans


def _fallback_recursive_split(text: str, section_path: str | None) -> list[ClauseSpan]:
    settings = get_settings()
    max_chars = settings.max_chunk_tokens * 4  # rough token->char heuristic
    overlap = settings.chunk_overlap_tokens * 4

    spans: list[ClauseSpan] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        # Prefer to break on a paragraph/sentence boundary near `end`.
        break_at = text.rfind("\n\n", start, end)
        if break_at == -1 or break_at <= start:
            break_at = end
        chunk = text[start:break_at].strip()
        if chunk:
            spans.append(ClauseSpan(text=chunk, section_path=section_path, start_char=start, end_char=break_at))
        start = max(break_at - overlap, break_at) if break_at < end else end
        if break_at >= len(text):
            break
    return spans
