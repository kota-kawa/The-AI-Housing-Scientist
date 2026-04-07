from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


UTC = timezone.utc


def utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


class Database:
    def __init__(self, path: str):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def init(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;

                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    pending_action_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                );

                CREATE TABLE IF NOT EXISTS memories (
                    session_id TEXT PRIMARY KEY,
                    user_memory_json TEXT NOT NULL,
                    task_memory_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                );

                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    input_json TEXT NOT NULL,
                    output_json TEXT NOT NULL,
                    reasoning TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                );
                """
            )
            conn.commit()

    def create_session(self) -> tuple[str, str]:
        session_id = uuid.uuid4().hex
        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO sessions(id, status, pending_action_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, "active", None, now, now),
            )
            conn.execute(
                "INSERT INTO memories(session_id, user_memory_json, task_memory_json, updated_at) VALUES (?, ?, ?, ?)",
                (session_id, "{}", "{}", now),
            )
            conn.commit()
        return session_id, now

    def session_exists(self, session_id: str) -> bool:
        with self.connect() as conn:
            row = conn.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return row is not None

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "status": row["status"],
            "pending_action": json.loads(row["pending_action_json"]) if row["pending_action_json"] else None,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def set_pending_action(self, session_id: str, pending_action: dict[str, Any] | None) -> None:
        now = utc_now_iso()
        payload = json.dumps(pending_action, ensure_ascii=False) if pending_action else None
        with self.connect() as conn:
            conn.execute(
                "UPDATE sessions SET pending_action_json = ?, updated_at = ? WHERE id = ?",
                (payload, now, session_id),
            )
            conn.commit()

    def add_message(self, session_id: str, role: str, content: dict[str, Any]) -> None:
        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO messages(session_id, role, content_json, created_at) VALUES (?, ?, ?, ?)",
                (session_id, role, json.dumps(content, ensure_ascii=False), now),
            )
            conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))
            conn.commit()

    def list_messages(self, session_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT role, content_json, created_at FROM messages WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
        return [
            {
                "role": row["role"],
                "content": json.loads(row["content_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def get_memories(self, session_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT user_memory_json, task_memory_json FROM memories WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return {}, {}
        return json.loads(row["user_memory_json"]), json.loads(row["task_memory_json"])

    def update_memories(
        self, session_id: str, user_memory: dict[str, Any], task_memory: dict[str, Any]
    ) -> None:
        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                "UPDATE memories SET user_memory_json = ?, task_memory_json = ?, updated_at = ? WHERE session_id = ?",
                (
                    json.dumps(user_memory, ensure_ascii=False),
                    json.dumps(task_memory, ensure_ascii=False),
                    now,
                    session_id,
                ),
            )
            conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))
            conn.commit()

    def add_audit_event(
        self,
        session_id: str,
        stage: str,
        input_payload: dict[str, Any],
        output_payload: dict[str, Any],
        reasoning: str,
    ) -> None:
        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO audit_events(session_id, stage, input_json, output_json, reasoning, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    stage,
                    json.dumps(input_payload, ensure_ascii=False),
                    json.dumps(output_payload, ensure_ascii=False),
                    reasoning,
                    now,
                ),
            )
            conn.commit()

    def list_audit_events(self, session_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT id, stage, input_json, output_json, reasoning, created_at FROM audit_events WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
        events = []
        for row in rows:
            events.append(
                {
                    "id": row["id"],
                    "stage": row["stage"],
                    "input": json.loads(row["input_json"]),
                    "output": json.loads(row["output_json"]),
                    "reasoning": row["reasoning"],
                    "created_at": row["created_at"],
                }
            )
        return events
