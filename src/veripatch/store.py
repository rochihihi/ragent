"""SQLite-backed event log and checkpoint storage."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from veripatch.domain import AgentRunState


class SQLiteRunStore:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    state_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_events_run_id ON events(run_id, sequence);
                """
            )

    def checkpoint(self, state: AgentRunState) -> None:
        state.updated_at = datetime.now(UTC)
        with self._connection() as connection:
            self._checkpoint(connection, state)

    @staticmethod
    def _checkpoint(connection: sqlite3.Connection, state: AgentRunState) -> None:
        connection.execute(
            """
            INSERT INTO runs(run_id, state_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                state_json = excluded.state_json,
                updated_at = excluded.updated_at
            """,
            (state.run_id, state.model_dump_json(), state.updated_at.isoformat()),
        )

    def record(self, state: AgentRunState, event_type: str, payload: dict[str, Any]) -> None:
        state.updated_at = datetime.now(UTC)
        with self._connection() as connection:
            self._checkpoint(connection, state)
            self._append_event(connection, state.run_id, event_type, payload)

    def append_event(self, run_id: str, event_type: str, payload: dict[str, Any]) -> None:
        with self._connection() as connection:
            self._append_event(connection, run_id, event_type, payload)

    @staticmethod
    def _append_event(
        connection: sqlite3.Connection,
        run_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        connection.execute(
            """
            INSERT INTO events(run_id, event_type, payload_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                run_id,
                event_type,
                json.dumps(payload, ensure_ascii=False, default=str),
                datetime.now(UTC).isoformat(),
            ),
        )

    def load(self, run_id: str) -> AgentRunState | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT state_json FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return AgentRunState.model_validate_json(row["state_json"]) if row else None

    def events(
        self, run_id: str, *, after_sequence: int = 0, limit: int = 500
    ) -> list[dict[str, Any]]:
        if not 1 <= limit <= 500:
            raise ValueError("Event limit must be between 1 and 500")
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT sequence, event_type, payload_json, created_at
                FROM events WHERE run_id = ? AND sequence > ? ORDER BY sequence LIMIT ?
                """,
                (run_id, after_sequence, limit),
            ).fetchall()
        return [
            {
                "sequence": row["sequence"],
                "event_type": row["event_type"],
                "payload": json.loads(row["payload_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def list_runs(self, *, limit: int = 50, offset: int = 0) -> list[AgentRunState]:
        if not 1 <= limit <= 100:
            raise ValueError("Run limit must be between 1 and 100")
        if offset < 0:
            raise ValueError("Run offset cannot be negative")
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT state_json FROM runs ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [AgentRunState.model_validate_json(row["state_json"]) for row in rows]
