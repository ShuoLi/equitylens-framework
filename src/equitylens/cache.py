"""SQLite-backed cache for engine outputs and raw data."""

from __future__ import annotations

import hashlib
import pickle
import sqlite3
import time
from pathlib import Path
from typing import Any


class Cache:
    """Simple SQLite cache with TTL.

    Parameters
    ----------
    path : str or Path
        Database file path.
    ttl_hours : float
        Default time-to-live for cached entries.
    """

    def __init__(self, path: str | Path = "equitylens_cache.db", ttl_hours: float = 72):
        self.path = Path(path)
        self.ttl_hours = ttl_hours
        self._init_db()

    def _init_db(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    created_at REAL,
                    ttl_hours REAL
                )
            """)
            conn.commit()

    @staticmethod
    def _make_key(*parts: Any) -> str:
        raw = ":".join(str(p) for p in parts)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, *key_parts: Any) -> Any | None:
        key = self._make_key(*key_parts)
        with sqlite3.connect(self.path) as conn:
            row = conn.execute(
                "SELECT value, created_at, ttl_hours FROM cache WHERE key = ?",
                (key,),
            ).fetchone()
            if row is None:
                return None
            value, created_at, ttl = row
            if time.time() - created_at > ttl * 3600:
                conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                conn.commit()
                return None
            return pickle.loads(value)

    def set(self, *key_parts: Any, value: Any, ttl_hours: float | None = None):
        key = self._make_key(*key_parts)
        blob = pickle.dumps(value)
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value, created_at, ttl_hours) VALUES (?, ?, ?, ?)",
                (key, blob, time.time(), ttl_hours or self.ttl_hours),
            )
            conn.commit()

    def clear(self):
        with sqlite3.connect(self.path) as conn:
            conn.execute("DELETE FROM cache")
            conn.commit()
