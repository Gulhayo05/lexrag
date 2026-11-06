"""Persistent storage: batch results (idempotency, async polling, >=30 day
retention) and the generation cache that makes answers reproducible."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time

from .config import settings

_lock = threading.Lock()
_con: sqlite3.Connection | None = None


def con() -> sqlite3.Connection:
    global _con
    if _con is None:
        _con = sqlite3.connect(settings.results_db, check_same_thread=False, isolation_level=None)
        _con.execute("PRAGMA journal_mode=WAL")
        _con.executescript(
            """
            CREATE TABLE IF NOT EXISTS batches(
              batch_id TEXT PRIMARY KEY, run_id TEXT, request_hash TEXT, status TEXT,
              created_at REAL, updated_at REAL, response TEXT);
            CREATE INDEX IF NOT EXISTS batches_run ON batches(run_id);
            CREATE TABLE IF NOT EXISTS gen_cache(key TEXT PRIMARY KEY, value TEXT, created_at REAL);
            """
        )
    return _con


def canonical_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def get_batch(batch_id: str) -> dict | None:
    with _lock:
        r = con().execute("SELECT batch_id, run_id, request_hash, status, response FROM batches WHERE batch_id=?", (batch_id,)).fetchone()
    if not r:
        return None
    return {"batch_id": r[0], "run_id": r[1], "request_hash": r[2], "status": r[3], "response": json.loads(r[4]) if r[4] else None}


def claim_batch(batch_id: str, run_id: str, request_hash: str) -> tuple[bool, dict | None]:
    """Atomically register a batch.  Returns (created, existing_row)."""
    with _lock:
        c = con()
        r = c.execute("SELECT request_hash, status, response FROM batches WHERE batch_id=?", (batch_id,)).fetchone()
        if r:
            return False, {"request_hash": r[0], "status": r[1], "response": json.loads(r[2]) if r[2] else None}
        now = time.time()
        c.execute("INSERT INTO batches VALUES (?,?,?,?,?,?,NULL)", (batch_id, run_id, request_hash, "processing", now, now))
        return True, None


def finish_batch(batch_id: str, status: str, response: dict | None) -> None:
    with _lock:
        con().execute(
            "UPDATE batches SET status=?, response=?, updated_at=? WHERE batch_id=?",
            (status, json.dumps(response, ensure_ascii=False) if response is not None else None, time.time(), batch_id),
        )


def release_batch(batch_id: str) -> None:
    """Forget a batch that failed as a whole so the tester can resend it."""
    with _lock:
        con().execute("DELETE FROM batches WHERE batch_id=?", (batch_id,))


def cache_get(key: str):
    with _lock:
        r = con().execute("SELECT value FROM gen_cache WHERE key=?", (key,)).fetchone()
    return json.loads(r[0]) if r else None


def cache_put(key: str, value) -> None:
    with _lock:
        con().execute("INSERT OR REPLACE INTO gen_cache VALUES (?,?,?)", (key, json.dumps(value, ensure_ascii=False), time.time()))
