"""Per-query latency/size log (SQLite) for the admin dashboard and the latency/cost table in the report.

Token counts are estimates (characters / 4), not provider-reported usage; set LLM price per million tokens in the report
by hand. Stores no question or answer text, only sizes and status.
"""
from __future__ import annotations

import datetime as dt
import sqlite3
import threading
from pathlib import Path

_lock = threading.Lock()


def _con(data_dir: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(data_dir / "telemetry.db"))
    con.execute("CREATE TABLE IF NOT EXISTS queries(ts TEXT, scope TEXT, latency_ms REAL, status TEXT, chars_in INT, chars_out INT, evidence INT)")
    return con


def record(data_dir: Path, scope: str, latency_ms: float, status: str, chars_in: int, chars_out: int, evidence: int) -> None:
    with _lock, _con(data_dir) as con:
        con.execute("INSERT INTO queries VALUES (?,?,?,?,?,?,?)", (dt.datetime.utcnow().isoformat(), scope, latency_ms, status, chars_in, chars_out, evidence))


def summary(data_dir: Path, last: int = 1000) -> dict:
    with _lock, _con(data_dir) as con:
        rows = con.execute("SELECT latency_ms, status, chars_in, chars_out FROM queries ORDER BY ts DESC LIMIT ?", (last,)).fetchall()
    if not rows:
        return {"queries": 0}
    lat = sorted(r[0] for r in rows)
    pct = lambda p: round(lat[min(len(lat) - 1, int(p / 100 * (len(lat) - 1) + 0.5))], 1)
    by_status: dict = {}
    for r in rows:
        by_status[r[1] or "ok"] = by_status.get(r[1] or "ok", 0) + 1
    return {"queries": len(rows), "latency_p50_ms": pct(50), "latency_p95_ms": pct(95), "by_status": by_status,
            "est_tokens_in": sum(r[2] for r in rows) // 4, "est_tokens_out": sum(r[3] for r in rows) // 4}
