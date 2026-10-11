"""SQLite persistence for interactive Studio sessions."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from veripatch.studio_domain import StudioSession


class IdempotencyConflict(ValueError):
    """A request/operation identity was reused with different parameters."""


def payload_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()


@dataclass
class RequestReceipt:
    response: dict[str, str] | None = None
    replayed: bool = False


class StudioStore:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self._transaction = threading.local()
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
                CREATE TABLE IF NOT EXISTS studio_requests (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    digest TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    response_json TEXT,
                    state TEXT NOT NULL DEFAULT 'accepted',
                    UNIQUE(session_id,kind,request_id)
                );
                CREATE TABLE IF NOT EXISTS studio_operations (
                    session_id TEXT NOT NULL,
                    operation_id TEXT NOT NULL,
                    digest TEXT NOT NULL,
                    state TEXT NOT NULL,
                    result_json TEXT,
                    PRIMARY KEY(session_id,operation_id)
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
        current = getattr(self._transaction, "connection", None)
        if current is not None:
            yield current
            return
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @contextmanager
    def request_transaction(
        self,
        session_id: str,
        kind: str,
        request_id: str,
        payload: dict[str, Any],
    ) -> Iterator[RequestReceipt]:
        """Atomic receipt + state/events. Callers must not await or execute tools inside.

        Thread-local connection reuse deliberately does not propagate to asyncio
        tasks or worker threads; dispatch happens only after this context commits.
        """
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM studio_requests WHERE session_id=? AND kind=? AND request_id=?",
                (session_id, kind, request_id),
            ).fetchone()
            digest = payload_digest(payload)
            if row:
                if row["digest"] != digest:
                    raise IdempotencyConflict("同一个请求 ID 对应的内容不同，请使用新的请求 ID")
                yield RequestReceipt(response=json.loads(row["response_json"]), replayed=True)
                return
            connection.execute(
                "INSERT INTO studio_requests(session_id,kind,request_id,digest,payload_json) "
                "VALUES(?,?,?,?,?)",
                (session_id, kind, request_id, digest, json.dumps(payload, ensure_ascii=False)),
            )
            self._transaction.connection = connection
            receipt = RequestReceipt()
            try:
                yield receipt
                if receipt.response is None:
                    raise RuntimeError("Request receipt must include a response")
                connection.execute(
                    "UPDATE studio_requests SET response_json=? "
                    "WHERE session_id=? AND kind=? AND request_id=?",
                    (json.dumps(receipt.response), session_id, kind, request_id),
                )
            finally:
                del self._transaction.connection

    def request_started(self, session_id: str, kind: str, request_id: str) -> bool:
        with self._connection() as connection:
            return (
                connection.execute(
                    "UPDATE studio_requests SET state='started' "
                    "WHERE session_id=? AND kind=? AND request_id=? AND state='accepted'",
                    (session_id, kind, request_id),
                ).rowcount
                == 1
            )

    def request_finished(self, session_id: str, kind: str, request_id: str) -> None:
        with self._connection() as connection:
            connection.execute(
                "UPDATE studio_requests SET state='finished' "
                "WHERE session_id=? AND kind=? AND request_id=?",
                (session_id, kind, request_id),
            )

    def unfinished_requests(self, session_id: str) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM studio_requests WHERE session_id=? AND state!='finished' "
                "ORDER BY sequence",
                (session_id,),
            ).fetchall()
        return [
            {
                **dict(row),
                "payload": json.loads(row["payload_json"]),
                "response": json.loads(row["response_json"]),
            }
            for row in rows
        ]

    def next_steer(self, session_id: str) -> tuple[str, str] | None:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM studio_requests WHERE session_id=? AND kind='message' "
                "AND state='accepted' AND json_extract(response_json,'$.status')='queued' "
                "ORDER BY sequence LIMIT 1",
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                "UPDATE studio_requests SET state='started' WHERE sequence=?", (row["sequence"],)
            )
        return json.loads(row["payload_json"])["content"], row["request_id"]

    def operation(self, session_id: str, operation_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM studio_operations WHERE session_id=? AND operation_id=?",
                (session_id, operation_id),
            ).fetchone()
        return dict(row) if row else None

    def claim_operation(
        self,
        session_id: str,
        operation_id: str,
        digest: str,
        legacy: dict[str, Any] | None = None,
        *,
        resume_pending: bool = False,
    ) -> tuple[bool, dict[str, Any] | None]:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM studio_operations WHERE session_id=? AND operation_id=?",
                (session_id, operation_id),
            ).fetchone()
            if row and row["digest"] != digest:
                raise IdempotencyConflict("同一个工具操作 ID 的参数发生变化，已阻止执行")
            if row:
                state = row["state"]
                result = json.loads(row["result_json"]) if row["result_json"] else {"status": state}
                result["status"] = state
            else:
                state = legacy["status"] if legacy else "new"
                result = legacy
                connection.execute(
                    "INSERT INTO studio_operations VALUES(?,?,?,?,?)",
                    (
                        session_id,
                        operation_id,
                        digest,
                        "running" if state == "new" else state,
                        json.dumps(legacy) if legacy else None,
                    ),
                )
            claimed = state == "new" or (state == "pending" and resume_pending)
            if claimed:
                connection.execute(
                    "UPDATE studio_operations SET state='running',result_json=NULL "
                    "WHERE session_id=? AND operation_id=?",
                    (session_id, operation_id),
                )
            return claimed, result

    def claim_batch(self, session_id: str, entries: list[tuple[str, str]]) -> bool:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if any(
                connection.execute(
                    "SELECT 1 FROM studio_operations WHERE session_id=? AND operation_id=?",
                    (session_id, identity),
                ).fetchone()
                for identity, _ in entries
            ):
                return False
            connection.executemany(
                "INSERT INTO studio_operations VALUES(?,?,?,'running',NULL)",
                [(session_id, identity, digest) for identity, digest in entries],
            )
            return True

    def operation_unknown(self, session_id: str, operation_id: str) -> None:
        with self._connection() as connection:
            connection.execute(
                "UPDATE studio_operations SET state='unknown' "
                "WHERE session_id=? AND operation_id=?",
                (session_id, operation_id),
            )

    def save(self, session: StudioSession, event_type: str, payload: dict[str, Any]) -> None:
        session.updated_at = datetime.now(UTC)
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO studio_sessions(session_id,state_json,updated_at) VALUES(?,?,?)
                ON CONFLICT(session_id) DO UPDATE SET
                state_json=excluded.state_json, updated_at=excluded.updated_at""",
                (session.session_id, session.model_dump_json(), session.updated_at.isoformat()),
            )
            # Acknowledged tool results and the session snapshot commit together.
            for identity, result in session.tool_call_results.items():
                if result.get("status") in {"pending", "completed"}:
                    connection.execute(
                        "UPDATE studio_operations SET state=?,result_json=? "
                        "WHERE session_id=? AND operation_id=? AND state IN ('running','pending')",
                        (
                            result["status"],
                            json.dumps(result, ensure_ascii=False, default=str),
                            session.session_id,
                            identity,
                        ),
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
            connection.execute("DELETE FROM studio_requests WHERE session_id=?", (session_id,))
            connection.execute("DELETE FROM studio_operations WHERE session_id=?", (session_id,))
            connection.execute("DELETE FROM studio_sessions WHERE session_id=?", (session_id,))
        return True

    def events(
        self, session_id: str, after: int = 0, *, full: bool = False
    ) -> list[dict[str, Any]]:
        with self._connection() as connection:
            if full:
                rows = connection.execute(
                    """SELECT sequence,event_type,payload_json,created_at FROM studio_events
                    WHERE session_id=? AND sequence>? ORDER BY sequence""",
                    (session_id, after),
                ).fetchall()
            elif after:
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
