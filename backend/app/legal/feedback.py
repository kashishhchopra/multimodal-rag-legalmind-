"""Thumbs up/down feedback store (SQLite) for later evaluation and training-data mining."""
from __future__ import annotations

import datetime as dt
import sqlite3
import threading
from pathlib import Path

_lock = threading.Lock()


def _conn(data_dir: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(data_dir / "feedback.db"))
    con.execute("""CREATE TABLE IF NOT EXISTS feedback(
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, document_id TEXT, question TEXT, answer TEXT,
        rating INTEGER, reason TEXT, status TEXT)""")
    return con


def add(data_dir: Path, document_id: str, question: str, answer: str, rating: int, reason: str, status: str) -> int:
    with _lock, _conn(data_dir) as con:
        cur = con.execute("INSERT INTO feedback(ts,document_id,question,answer,rating,reason,status) VALUES (?,?,?,?,?,?,?)",
                          (dt.datetime.utcnow().isoformat(), document_id, question[:2000], answer[:8000], rating, reason[:1000], status))
        return int(cur.lastrowid)


def stats(data_dir: Path) -> dict:
    with _lock, _conn(data_dir) as con:
        up, down = (con.execute("SELECT COUNT(*) FROM feedback WHERE rating=?", (r,)).fetchone()[0] for r in (1, -1))
        by_status = dict(con.execute("SELECT COALESCE(NULLIF(status,''),'ok'), COUNT(*) FROM feedback WHERE rating=-1 GROUP BY 1").fetchall())
    return {"up": up, "down": down, "down_by_answer_status": by_status}
