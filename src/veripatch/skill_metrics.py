"""Durable metric receipts; bounded views do not discard deduplication identities."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any
from uuid import uuid4


def append_sample(
    root: Path,
    name: str,
    version: str,
    field: str,
    existing: list[dict[str, Any]],
    *,
    success: bool,
    latency_ms: int,
    operation_id: str | None,
) -> list[dict[str, Any]]:
    from veripatch.studio_skills import _linked

    root.mkdir(parents=True, exist_ok=True)
    path = root / "metrics.sqlite3"
    if any(_linked(Path(str(path) + suffix)) for suffix in ("", "-wal", "-shm", "-journal")):
        raise ValueError("技能指标数据库不能包含链接")
    connection = sqlite3.connect(path, timeout=5)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS metric_groups (
                name TEXT, version TEXT, field TEXT, PRIMARY KEY(name,version,field)
            );
            CREATE TABLE IF NOT EXISTS metric_receipts (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT, version TEXT, field TEXT, operation_id TEXT, sample TEXT,
                UNIQUE(name,version,field,operation_id)
            );
        """)
        key = (name, version, field)
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            fresh = connection.execute(
                "INSERT OR IGNORE INTO metric_groups VALUES(?,?,?)",
                key,
            ).rowcount
            if fresh:
                connection.executemany(
                    "INSERT INTO metric_receipts(name,version,field,operation_id,sample) "
                    "VALUES(?,?,?,?,?)",
                    [
                        (*key, f"legacy:{i}", json.dumps(sample))
                        for i, sample in enumerate(existing)
                    ],
                )
            sample = {"success": success, "latency_ms": latency_ms, "time": time.time()}
            connection.execute(
                "INSERT OR IGNORE INTO metric_receipts(name,version,field,operation_id,sample) "
                "VALUES(?,?,?,?,?)",
                (*key, operation_id or uuid4().hex, json.dumps(sample)),
            )
            rows = connection.execute(
                "SELECT sample FROM metric_receipts WHERE name=? AND version=? AND field=? "
                "ORDER BY sequence DESC LIMIT 100",
                key,
            ).fetchall()
        # Always rebuild the registry view, including on replay: this repairs a
        # crash between the metric commit and atomic registry replacement.
        return [json.loads(row[0]) for row in reversed(rows)]
    finally:
        connection.close()
