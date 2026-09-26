"""SQLite persistence for interactive Studio sessions."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from veripatch.studio_domain import StudioSession


class StudioStore:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS studio_sessions (
                    session_id TEXT PRIMARY KEY,
                    state_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS studio_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_studio_events
                ON studio_events(session_id, sequence);
                CREATE TABLE IF NOT EXISTS studio_project_preferences (
                    repo_key TEXT PRIMARY KEY,
                    repo_root TEXT NOT NULL,
                    response_style TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

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

    def save(self, session: StudioSession, event_type: str, payload: dict[str, Any]) -> None:
        session.updated_at = datetime.now(UTC)
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO studio_sessions(session_id,state_json,updated_at) VALUES(?,?,?)
                ON CONFLICT(session_id) DO UPDATE SET
                state_json=excluded.state_json, updated_at=excluded.updated_at""",
                (session.session_id, session.model_dump_json(), session.updated_at.isoformat()),
            )
            connection.execute(
                """INSERT INTO studio_events(session_id,event_type,payload_json,created_at)
                VALUES(?,?,?,?)""",
                (
                    session.session_id,
                    event_type,
                    json.dumps(payload, ensure_ascii=False, default=str),
                    datetime.now(UTC).isoformat(),
                ),
            )

    def append_event(self, session_id: str, event_type: str, payload: dict[str, Any]) -> None:
        """Record concurrent UI input without overwriting the running session snapshot."""
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO studio_events(session_id,event_type,payload_json,created_at)
                VALUES(?,?,?,?)""",
                (
                    session_id,
                    event_type,
                    json.dumps(payload, ensure_ascii=False, default=str),
                    datetime.now(UTC).isoformat(),
                ),
            )

    def load(self, session_id: str) -> StudioSession | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT state_json FROM studio_sessions WHERE session_id=?", (session_id,)
            ).fetchone()
        return self._decode_session(row["state_json"]) if row else None

    def list_sessions(self, limit: int = 50) -> list[StudioSession]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT state_json FROM studio_sessions ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._decode_session(row["state_json"]) for row in rows]

    @staticmethod
    def _repo_key(repo_root: str) -> str:
        return str(Path(repo_root).resolve()).replace("\\", "/").casefold()

    def project_response_style(self, repo_root: str) -> str | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT response_style FROM studio_project_preferences WHERE repo_key=?",
                (self._repo_key(repo_root),),
            ).fetchone()
        return str(row["response_style"]) if row else None

    def save_project_response_style(self, repo_root: str, response_style: str) -> None:
        resolved = str(Path(repo_root).resolve())
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO studio_project_preferences(
                    repo_key,repo_root,response_style,updated_at
                ) VALUES(?,?,?,?)
                ON CONFLICT(repo_key) DO UPDATE SET
                repo_root=excluded.repo_root,
                response_style=excluded.response_style,
                updated_at=excluded.updated_at""",
                (
                    self._repo_key(resolved),
                    resolved,
                    response_style,
                    datetime.now(UTC).isoformat(),
                ),
            )

    @staticmethod
    def _decode_session(state_json: str) -> StudioSession:
        """Load older state defensively so one oversized list cannot brick startup."""
        payload = json.loads(state_json)
        if isinstance(payload, dict):
            for field in ("approved_paths", "approved_commands", "approved_capabilities"):
                values = payload.get(field)
                if isinstance(values, list) and len(values) > 40:
                    payload[field] = values[-40:]
        return StudioSession.model_validate(payload)

    def delete(self, session_id: str) -> bool:
        with self._connection() as connection:
            exists = connection.execute(
                "SELECT 1 FROM studio_sessions WHERE session_id=?", (session_id,)
            ).fetchone()
            if exists is None:
                return False
            connection.execute("DELETE FROM studio_events WHERE session_id=?", (session_id,))
            connection.execute("DELETE FROM studio_sessions WHERE session_id=?", (session_id,))
        return True

    def events(self, session_id: str, after: int = 0) -> list[dict[str, Any]]:
        with self._connection() as connection:
            if after:
                rows = connection.execute(
                    """SELECT sequence,event_type,payload_json,created_at FROM studio_events
                    WHERE session_id=? AND sequence>? ORDER BY sequence LIMIT 500""",
                    (session_id, after),
                ).fetchall()
            else:
                rows = connection.execute(
                    """SELECT sequence,event_type,payload_json,created_at FROM studio_events
                    WHERE session_id=? ORDER BY sequence DESC LIMIT 500""",
                    (session_id,),
                ).fetchall()
                rows = list(reversed(rows))
        return [
            {
                "sequence": row["sequence"],
                "event_type": row["event_type"],
                "payload": json.loads(row["payload_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]
