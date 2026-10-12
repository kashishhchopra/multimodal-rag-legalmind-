"""Opt-in authentication, roles, per-user document ownership and an audit log.

Off by default so the demo and tests behave as before. Turn on with AUTH_ENABLED=1 and a strong AUTH_SECRET (the app
refuses to start requests without one). Passwords: PBKDF2-HMAC-SHA256, 200k iterations, per-user salt. Tokens: HS256 JWT
built with the standard library (no extra dependency), 12 h expiry. The first registered user is the admin; later users
are "user". When enabled, every /api route except /api/health and /api/auth/* needs a Bearer token, users only see and
touch documents they uploaded, and admins see everything. Uploads, queries, deletes and logins are written to the
audit log (SQLite in the data directory).
"""
from __future__ import annotations

import base64
import datetime as dt
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from .config import get_settings

router = APIRouter(prefix="/api/auth", tags=["auth"])
_lock = threading.Lock()
TOKEN_TTL_S = 12 * 3600
_OPEN = ("/api/health", "/api/auth/")


def enabled() -> bool:
    return os.getenv("AUTH_ENABLED", "0").lower() in ("1", "true", "yes")


def _db() -> sqlite3.Connection:
    con = sqlite3.connect(str(get_settings().data_dir / "auth.db"))
    con.executescript("""
      CREATE TABLE IF NOT EXISTS users(username TEXT PRIMARY KEY, salt TEXT, pw TEXT, role TEXT);
      CREATE TABLE IF NOT EXISTS owners(document_id TEXT PRIMARY KEY, username TEXT);
      CREATE TABLE IF NOT EXISTS audit(id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, username TEXT, action TEXT, detail TEXT);
    """)
    return con


def _hash(pw: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, 200_000).hex()


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _secret() -> bytes:
    s = os.getenv("AUTH_SECRET", "")
    if len(s) < 16:
        raise HTTPException(500, "AUTH_ENABLED is set but AUTH_SECRET is missing or shorter than 16 characters")
    return s.encode()


def make_token(username: str, role: str, now: Optional[float] = None) -> str:
    head = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    body = _b64(json.dumps({"sub": username, "role": role, "exp": int((time.time() if now is None else now) + TOKEN_TTL_S)}).encode())
    sig = _b64(hmac.new(_secret(), f"{head}.{body}".encode(), hashlib.sha256).digest())
    return f"{head}.{body}.{sig}"


def read_token(token: str, now: Optional[float] = None) -> dict:
    try:
        head, body, sig = token.split(".")
        good = _b64(hmac.new(_secret(), f"{head}.{body}".encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, good):
            raise ValueError("bad signature")
        claims = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
        if claims["exp"] < (time.time() if now is None else now):
            raise ValueError("expired")
        return claims
    except HTTPException:
        raise
    except Exception:  # noqa: BLE001 - any malformed/expired token is simply unauthorised
        raise HTTPException(401, "Invalid or expired token")


def audit(username: str, action: str, detail: str = "") -> None:
    if not enabled():
        return
    with _lock, _db() as con:
        con.execute("INSERT INTO audit(ts,username,action,detail) VALUES (?,?,?,?)", (dt.datetime.utcnow().isoformat(), username, action, detail[:300]))


def set_owner(document_id: str, username: str) -> None:
    if enabled():
        with _lock, _db() as con:
            con.execute("INSERT OR REPLACE INTO owners VALUES (?,?)", (document_id, username))


def owned_by(username: str) -> set:
    with _lock, _db() as con:
        return {r[0] for r in con.execute("SELECT document_id FROM owners WHERE username=?", (username,))}


def current_user(request: Request) -> Optional[dict]:
    """Router-level dependency: None when auth is off; otherwise the verified claims or a 401/403."""
    if not enabled():
        return None
    if request.url.path.startswith(_OPEN):
        return None
    h = request.headers.get("authorization", "")
    if not h.lower().startswith("bearer "):
        raise HTTPException(401, "Authentication required")
    claims = read_token(h[7:])
    request.state.user = claims
    doc_id = request.path_params.get("document_id")
    if doc_id and claims["role"] != "admin":
        with _lock, _db() as con:
            row = con.execute("SELECT username FROM owners WHERE document_id=?", (doc_id,)).fetchone()
        if row and row[0] != claims["sub"]:
            raise HTTPException(403, "You do not have access to this document")
    return claims


class Credentials(BaseModel):
    username: str = Field(..., min_length=3, max_length=64, pattern=r"^[A-Za-z0-9_.@-]+$")
    password: str = Field(..., min_length=10, max_length=200)


@router.post("/register")
async def register(c: Credentials) -> dict:
    if not enabled():
        raise HTTPException(404, "Authentication is disabled")
    salt = secrets.token_bytes(16)
    with _lock, _db() as con:
        first = con.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0
        if con.execute("SELECT 1 FROM users WHERE username=?", (c.username,)).fetchone():
            raise HTTPException(409, "Username already taken")
        role = "admin" if first else "user"
        con.execute("INSERT INTO users VALUES (?,?,?,?)", (c.username, salt.hex(), _hash(c.password, salt), role))
    audit(c.username, "register", role)
    return {"username": c.username, "role": role}


@router.post("/login")
async def login(c: Credentials) -> dict:
    if not enabled():
        raise HTTPException(404, "Authentication is disabled")
    with _lock, _db() as con:
        row = con.execute("SELECT salt, pw, role FROM users WHERE username=?", (c.username,)).fetchone()
    ok = row and hmac.compare_digest(row[1], _hash(c.password, bytes.fromhex(row[0])))
    audit(c.username, "login_ok" if ok else "login_failed")
    if not ok:
        raise HTTPException(401, "Wrong username or password")
    return {"access_token": make_token(c.username, row[2]), "token_type": "bearer", "role": row[2]}


def require_admin(user: Optional[dict] = Depends(current_user)) -> Optional[dict]:
    if enabled() and (not user or user["role"] != "admin"):
        raise HTTPException(403, "Admin only")
    return user
