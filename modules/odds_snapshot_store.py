"""SQLite persistence for observed live 1X2 odds snapshots."""
from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_match_key(match_id: Any, match_url: Any, date: Any, league: Any,
                    home: Any, away: Any) -> str:
    """Build a stable fixture key, preferring provider identifiers."""
    for value in (match_id, match_url):
        text = str(value or "").strip()
        if text:
            return hashlib.sha256(text.encode("utf-8")).hexdigest()
    identity = "|".join(str(value or "").strip().casefold() for value in
                        (date, league, home, away))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def record_snapshot(match_key: str, odds: dict[str, Any], db_path: str | Path,
                    *, captured_at: datetime | None = None) -> bool:
    """Insert only when odds differ from the latest stored snapshot."""
    values = tuple(float(odds[key]) for key in ("1", "X", "2"))
    if not match_key or any(value <= 1 for value in values):
        raise ValueError("A match key and three decimal odds greater than 1 are required.")
    timestamp = (captured_at or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS odds_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_key TEXT NOT NULL,
                captured_at TEXT NOT NULL,
                odds_home REAL NOT NULL,
                odds_draw REAL NOT NULL,
                odds_away REAL NOT NULL
            )
        """)
        latest = connection.execute(
            "SELECT odds_home, odds_draw, odds_away FROM odds_snapshots "
            "WHERE match_key = ? ORDER BY id DESC LIMIT 1", (match_key,)
        ).fetchone()
        if latest and all(abs(float(old) - new) < 1e-9 for old, new in zip(latest, values)):
            return False
        connection.execute(
            "INSERT INTO odds_snapshots (match_key, captured_at, odds_home, odds_draw, odds_away) "
            "VALUES (?, ?, ?, ?, ?)", (match_key, timestamp, *values)
        )
    return True


def get_snapshots(match_key: str, db_path: str | Path, *, limit: int = 100) -> list[dict]:
    if not match_key:
        raise ValueError("match_key is required.")
    limit = min(max(int(limit), 1), 500)
    path = Path(db_path)
    if not path.is_file():
        return []
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='odds_snapshots'"
        ).fetchone()
        if not exists:
            return []
        rows = connection.execute(
            "SELECT captured_at, odds_home, odds_draw, odds_away FROM odds_snapshots "
            "WHERE match_key = ? ORDER BY id DESC LIMIT ?", (match_key, limit)
        ).fetchall()
    return [
        {
            "captured_at": row["captured_at"],
            "odds": {
                "1": float(row["odds_home"]),
                "X": float(row["odds_draw"]),
                "2": float(row["odds_away"]),
            },
        }
        for row in reversed(rows)
    ]
