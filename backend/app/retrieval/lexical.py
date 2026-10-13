"""Persistent lexical (BM25) index on SQLite FTS5, for corpora too large to hold in RAM.

The in-memory BM25Index in hybrid.py is fine for hundreds of chunks (uploaded documents) but needs gigabytes for
the 120k-chunk judgment corpus. FTS5 keeps postings on disk, ranks with its built-in bm25(), and supports the same metadata
filters through UNINDEXED columns. One database file lives next to the Chroma store (VECTOR_DB_PATH/lexical.sqlite3);
vector_store.add_text_chunks/delete_document keep it in sync, scripts/build_lexical_index.py backfills it.
Only the columns in FILTER_COLUMNS can be filtered here; for any other filter key the caller falls back to the in-memory path.
"""
from __future__ import annotations

import re
import sqlite3
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..config import get_settings
from .hybrid import _STOP

FILTER_COLUMNS = ("document_id", "year", "doc_type", "section", "act", "court", "source")
_lock = threading.Lock()
_conns: Dict[str, sqlite3.Connection] = {}


def _path() -> Path:
    return get_settings().vector_dir / "lexical.sqlite3"


def _conn() -> sqlite3.Connection:
    key = str(_path())
    con = _conns.get(key)
    if con is None:
        con = sqlite3.connect(key, check_same_thread=False)
        con.execute("PRAGMA journal_mode=WAL")
        cols = ", ".join(f"{c} UNINDEXED" for c in ("chunk_id", *FILTER_COLUMNS))
        con.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5({cols}, text, tokenize='unicode61 remove_diacritics 2')")
        _conns[key] = con
    return con


def count() -> int:
    with _lock:
        return _conn().execute("SELECT COUNT(*) FROM chunks").fetchone()[0]


def add(ids: Sequence[str], texts: Sequence[str], metas: Sequence[dict]) -> None:
    rows = [(i, *(str(m.get(c, "")) for c in FILTER_COLUMNS), t) for i, t, m in zip(ids, texts, metas)]
    with _lock:
        con = _conn()
        con.executemany(f"INSERT INTO chunks(chunk_id, {', '.join(FILTER_COLUMNS)}, text) VALUES ({', '.join('?' * (len(FILTER_COLUMNS) + 2))})", rows)
        con.commit()


def delete_document(document_id: str) -> None:
    with _lock:
        con = _conn()
        con.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        con.commit()


def can_filter(filters: Optional[Dict]) -> bool:
    return all(k in FILTER_COLUMNS for k in (filters or {}))


def search(query: str, k: int, document_id: Optional[str] = None, filters: Optional[Dict] = None) -> List[Tuple[str, float]]:
    """Top-k (chunk_id, score) by BM25 (higher is better). Terms are OR-ed, like the in-memory index."""
    # same word splitting as FTS5's unicode61 tokenizer, so 'R1' matches the indexed token 'r1'
    terms = sorted({t for t in re.findall(r"\w+", query.lower()) if t not in _STOP and len(t) > 1 or t.isdigit()})
    if not terms:
        return []
    match = " OR ".join('"' + t.replace('"', "") + '"' for t in terms)
    where, args = ["chunks MATCH ?"], [match]
    conds = dict(filters or {})
    if document_id:
        conds["document_id"] = document_id
    for col, val in conds.items():
        where.append(f"{col} = ?")
        args.append(str(val))
    sql = f"SELECT chunk_id, -bm25(chunks) AS score FROM chunks WHERE {' AND '.join(where)} ORDER BY score DESC LIMIT ?"
    with _lock:
        try:
            return [(r[0], float(r[1])) for r in _conn().execute(sql, (*args, k)).fetchall()]
        except sqlite3.OperationalError:  # malformed MATCH expression etc.: behave like "no lexical hits"
            return []


def breakdown() -> Dict[str, Dict[str, int]]:
    """{doc_type: {"documents": n, "chunks": m}} straight from the index (cheap; used by the admin dashboard)."""
    with _lock:
        rows = _conn().execute("SELECT doc_type, COUNT(DISTINCT document_id), COUNT(*) FROM chunks GROUP BY doc_type").fetchall()
    return {(r[0] or "document"): {"documents": r[1], "chunks": r[2]} for r in rows}
