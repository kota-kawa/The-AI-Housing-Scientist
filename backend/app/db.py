from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any
import uuid

from app.catalog import CATALOG_SEED, build_catalog_detail_url, build_catalog_image_url

UTC = timezone.utc


def utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


class Database:
    def __init__(self, path: str):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 30000;")
        return conn

    def init(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;

                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    profile_id TEXT NOT NULL,
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

                CREATE TABLE IF NOT EXISTS profiles (
                    id TEXT PRIMARY KEY,
                    user_memory_json TEXT NOT NULL,
                    profile_memory_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
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

                CREATE TABLE IF NOT EXISTS research_jobs (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    llm_config_json TEXT NOT NULL DEFAULT '{}',
                    approved_plan_json TEXT NOT NULL,
                    current_stage TEXT NOT NULL,
                    progress_percent INTEGER NOT NULL,
                    latest_summary TEXT NOT NULL,
                    result_json TEXT,
                    error_message TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                );

                CREATE TABLE IF NOT EXISTS research_journal_nodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    node_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    input_json TEXT NOT NULL,
                    output_json TEXT NOT NULL,
                    reasoning TEXT NOT NULL,
                    duration_ms INTEGER NOT NULL,
                    intent TEXT NOT NULL DEFAULT 'draft',
                    is_failed INTEGER NOT NULL DEFAULT 0,
                    debug_depth INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(job_id) REFERENCES research_jobs(id)
                );

                CREATE TABLE IF NOT EXISTS llm_call_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    job_id TEXT,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    prompt_chars INTEGER NOT NULL,
                    response_chars INTEGER NOT NULL,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    estimated_cost_usd REAL,
                    duration_ms INTEGER NOT NULL,
                    success INTEGER NOT NULL,
                    error_message TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS property_catalog (
                    property_id TEXT PRIMARY KEY,
                    detail_url TEXT NOT NULL UNIQUE,
                    image_url TEXT NOT NULL DEFAULT '',
                    building_name TEXT NOT NULL,
                    address TEXT NOT NULL,
                    area_name TEXT NOT NULL,
                    nearest_station TEXT NOT NULL,
                    line_name TEXT NOT NULL,
                    station_walk_min INTEGER NOT NULL,
                    layout TEXT NOT NULL,
                    area_m2 REAL NOT NULL,
                    rent INTEGER NOT NULL,
                    management_fee INTEGER NOT NULL,
                    deposit INTEGER NOT NULL,
                    key_money INTEGER NOT NULL,
                    available_date TEXT NOT NULL,
                    agency_name TEXT NOT NULL,
                    notes TEXT NOT NULL,
                    contract_text TEXT NOT NULL,
                    features_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            self._ensure_column(conn, "sessions", "profile_id", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "research_journal_nodes", "parent_node_id", "INTEGER")
            self._ensure_column(
                conn, "research_journal_nodes", "branch_id", "TEXT NOT NULL DEFAULT ''"
            )
            self._ensure_column(
                conn, "research_journal_nodes", "selected", "INTEGER NOT NULL DEFAULT 0"
            )
            self._ensure_column(
                conn,
                "research_journal_nodes",
                "intent",
                "TEXT NOT NULL DEFAULT 'draft'",
            )
            self._ensure_column(
                conn,
                "research_journal_nodes",
                "is_failed",
                "INTEGER NOT NULL DEFAULT 0",
            )
            self._ensure_column(
                conn,
                "research_journal_nodes",
                "debug_depth",
                "INTEGER NOT NULL DEFAULT 0",
            )
            self._ensure_column(
                conn,
                "research_journal_nodes",
                "metrics_json",
                "TEXT NOT NULL DEFAULT '{}'",
            )
            self._ensure_column(
                conn,
                "research_jobs",
                "llm_config_json",
                "TEXT NOT NULL DEFAULT '{}'",
            )
            self._ensure_column(
                conn, "llm_call_events", "prompt_tokens", "INTEGER NOT NULL DEFAULT 0"
            )
            self._ensure_column(
                conn, "llm_call_events", "completion_tokens", "INTEGER NOT NULL DEFAULT 0"
            )
            self._ensure_column(
                conn, "llm_call_events", "total_tokens", "INTEGER NOT NULL DEFAULT 0"
            )
            self._ensure_column(conn, "llm_call_events", "estimated_cost_usd", "REAL")
            self._ensure_column(conn, "property_catalog", "image_url", "TEXT NOT NULL DEFAULT ''")
            self._seed_property_catalog(conn)
            conn.commit()

    def _ensure_column(
        self,
        conn: sqlite3.Connection,
        table_name: str,
        column_name: str,
        definition: str,
    ) -> None:
        columns = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        if any(row["name"] == column_name for row in columns):
            return
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")

    def _seed_property_catalog(self, conn: sqlite3.Connection) -> None:
        row = conn.execute("SELECT COUNT(*) AS count FROM property_catalog").fetchone()
        count = int(row["count"]) if row is not None else 0
        if count > 0:
            self._backfill_property_catalog_image_urls(conn)
            return

        now = utc_now_iso()
        for item in CATALOG_SEED:
            conn.execute(
                """
                INSERT INTO property_catalog(
                    property_id,
                    detail_url,
                    image_url,
                    building_name,
                    address,
                    area_name,
                    nearest_station,
                    line_name,
                    station_walk_min,
                    layout,
                    area_m2,
                    rent,
                    management_fee,
                    deposit,
                    key_money,
                    available_date,
                    agency_name,
                    notes,
                    contract_text,
                    features_json,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item["property_id"],
                    build_catalog_detail_url(item["property_id"]),
                    item.get("image_url") or build_catalog_image_url(item["property_id"]),
                    item["building_name"],
                    item["address"],
                    item["area_name"],
                    item["nearest_station"],
                    item["line_name"],
                    item["station_walk_min"],
                    item["layout"],
                    item["area_m2"],
                    item["rent"],
                    item["management_fee"],
                    item["deposit"],
                    item["key_money"],
                    item["available_date"],
                    item["agency_name"],
                    item["notes"],
                    item["contract_text"],
                    json.dumps(item.get("features", []), ensure_ascii=False),
                    now,
                    now,
                ),
            )

    def _backfill_property_catalog_image_urls(self, conn: sqlite3.Connection) -> None:
        for item in CATALOG_SEED:
            image_url = str(
                item.get("image_url") or build_catalog_image_url(item.get("property_id")) or ""
            ).strip()
            property_id = str(item.get("property_id") or "").strip()
            if not image_url or not property_id:
                continue
            conn.execute(
                """
                UPDATE property_catalog
                SET image_url = ?, updated_at = ?
                WHERE property_id = ?
                  AND COALESCE(image_url, '') = ''
                """,
                (image_url, utc_now_iso(), property_id),
            )

    def get_or_create_profile(self, profile_id: str | None = None) -> tuple[str, dict[str, Any]]:
        resolved_profile_id = (profile_id or uuid.uuid4().hex).strip() or uuid.uuid4().hex
        profile = self.get_profile(resolved_profile_id)
        if profile is not None:
            return resolved_profile_id, profile

        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO profiles(
                    id,
                    user_memory_json,
                    profile_memory_json,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    resolved_profile_id,
                    "{}",
                    json.dumps(
                        {
                            "search_history": [],
                            "reaction_history": [],
                            "learned_preferences": {},
                            "strategy_memory": {},
                        },
                        ensure_ascii=False,
                    ),
                    now,
                    now,
                ),
            )
            conn.commit()

        created_profile = self.get_profile(resolved_profile_id)
        if created_profile is None:
            raise RuntimeError("profile creation failed")
        return resolved_profile_id, created_profile

    def get_profile(self, profile_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM profiles WHERE id = ?", (profile_id,)).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "user_memory": json.loads(row["user_memory_json"]),
            "profile_memory": json.loads(row["profile_memory_json"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def update_profile(
        self,
        profile_id: str,
        user_memory: dict[str, Any],
        profile_memory: dict[str, Any],
    ) -> None:
        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE profiles
                SET user_memory_json = ?, profile_memory_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    json.dumps(user_memory, ensure_ascii=False),
                    json.dumps(profile_memory, ensure_ascii=False),
                    now,
                    profile_id,
                ),
            )
            conn.commit()

    def create_session(
        self,
        profile_id: str | None = None,
        *,
        user_memory: dict[str, Any] | None = None,
        task_memory: dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        resolved_profile_id, _ = self.get_or_create_profile(profile_id)
        session_id = uuid.uuid4().hex
        now = utc_now_iso()
        initial_user_memory = dict(user_memory or {})
        initial_task_memory = {
            "profile_id": resolved_profile_id,
            "property_reactions": {},
            "comparison_property_ids": [],
            "profile_resume_pending": False,
        }
        initial_task_memory.update(task_memory or {})
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions(id, profile_id, status, pending_action_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, resolved_profile_id, "active", None, now, now),
            )
            conn.execute(
                "INSERT INTO memories(session_id, user_memory_json, task_memory_json, updated_at) VALUES (?, ?, ?, ?)",
                (
                    session_id,
                    json.dumps(initial_user_memory, ensure_ascii=False),
                    json.dumps(initial_task_memory, ensure_ascii=False),
                    now,
                ),
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
            "profile_id": row["profile_id"],
            "status": row["status"],
            "pending_action": json.loads(row["pending_action_json"])
            if row["pending_action_json"]
            else None,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def set_session_status(self, session_id: str, status: str) -> None:
        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                "UPDATE sessions SET status = ?, updated_at = ? WHERE id = ?",
                (status, now, session_id),
            )
            conn.commit()

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

    def list_catalog_properties(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    property_id,
                    detail_url,
                    image_url,
                    building_name,
                    address,
                    area_name,
                    nearest_station,
                    line_name,
                    station_walk_min,
                    layout,
                    area_m2,
                    rent,
                    management_fee,
                    deposit,
                    key_money,
                    available_date,
                    agency_name,
                    notes,
                    contract_text,
                    features_json
                FROM property_catalog
                ORDER BY rent ASC, station_walk_min ASC
                """
            ).fetchall()
        return [self._catalog_row_to_dict(row) for row in rows]

    def get_catalog_property_by_id(self, property_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT
                    property_id,
                    detail_url,
                    image_url,
                    building_name,
                    address,
                    area_name,
                    nearest_station,
                    line_name,
                    station_walk_min,
                    layout,
                    area_m2,
                    rent,
                    management_fee,
                    deposit,
                    key_money,
                    available_date,
                    agency_name,
                    notes,
                    contract_text,
                    features_json
                FROM property_catalog
                WHERE property_id = ?
                """,
                (property_id,),
            ).fetchone()
        if row is None:
            return None
        return self._catalog_row_to_dict(row)

    def update_catalog_property_notes(self, property_id: str, notes: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE property_catalog SET notes = ?, updated_at = ? WHERE property_id = ?",
                (notes, utc_now_iso(), property_id),
            )

    def get_catalog_property_by_url(self, detail_url: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT
                    property_id,
                    detail_url,
                    image_url,
                    building_name,
                    address,
                    area_name,
                    nearest_station,
                    line_name,
                    station_walk_min,
                    layout,
                    area_m2,
                    rent,
                    management_fee,
                    deposit,
                    key_money,
                    available_date,
                    agency_name,
                    notes,
                    contract_text,
                    features_json
                FROM property_catalog
                WHERE detail_url = ?
                """,
                (detail_url,),
            ).fetchone()
        if row is None:
            return None
        return self._catalog_row_to_dict(row)

    def _catalog_row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "property_id": row["property_id"],
            "detail_url": row["detail_url"],
            "image_url": row["image_url"] or build_catalog_image_url(row["property_id"]),
            "building_name": row["building_name"],
            "address": row["address"],
            "area_name": row["area_name"],
            "nearest_station": row["nearest_station"],
            "line_name": row["line_name"],
            "station_walk_min": row["station_walk_min"],
            "layout": row["layout"],
            "area_m2": row["area_m2"],
            "rent": row["rent"],
            "management_fee": row["management_fee"],
            "deposit": row["deposit"],
            "key_money": row["key_money"],
            "available_date": row["available_date"],
            "agency_name": row["agency_name"],
            "notes": row["notes"],
            "contract_text": row["contract_text"],
            "features": json.loads(row["features_json"]) if row["features_json"] else [],
        }

    def create_research_job(
        self,
        *,
        session_id: str,
        provider: str,
        llm_config: dict[str, Any],
        approved_plan: dict[str, Any],
    ) -> tuple[str, str]:
        job_id = uuid.uuid4().hex
        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO research_jobs(
                    id,
                    session_id,
                    status,
                    provider,
                    llm_config_json,
                    approved_plan_json,
                    current_stage,
                    progress_percent,
                    latest_summary,
                    result_json,
                    error_message,
                    created_at,
                    started_at,
                    finished_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    session_id,
                    "queued",
                    provider,
                    json.dumps(llm_config, ensure_ascii=False),
                    json.dumps(approved_plan, ensure_ascii=False),
                    "queued",
                    0,
                    "調査ジョブを登録しました。",
                    None,
                    "",
                    now,
                    None,
                    None,
                    now,
                ),
            )
            conn.commit()
        return job_id, now

    def get_research_job(self, job_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM research_jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        return self._research_job_row_to_dict(row)

    def get_latest_research_job(self, session_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM research_jobs
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return self._research_job_row_to_dict(row)

    def claim_next_research_job(self) -> dict[str, Any] | None:
        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT *
                FROM research_jobs
                WHERE status = 'queued'
                ORDER BY created_at ASC
                LIMIT 1
                """
            ).fetchone()
            if row is None:
                conn.commit()
                return None

            conn.execute(
                """
                UPDATE research_jobs
                SET status = ?, current_stage = ?, latest_summary = ?, started_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    "running",
                    "plan_finalize",
                    "調査を開始しました。",
                    now,
                    now,
                    row["id"],
                ),
            )
            conn.commit()

        return self.get_research_job(row["id"])

    def update_research_job(
        self,
        job_id: str,
        *,
        status: str | None = None,
        current_stage: str | None = None,
        progress_percent: int | None = None,
        latest_summary: str | None = None,
        result_payload: dict[str, Any] | None = None,
        error_message: str | None = None,
        started_at: str | None = None,
        finished_at: str | None = None,
    ) -> None:
        updates: list[str] = []
        params: list[Any] = []
        if status is not None:
            updates.append("status = ?")
            params.append(status)
        if current_stage is not None:
            updates.append("current_stage = ?")
            params.append(current_stage)
        if progress_percent is not None:
            updates.append("progress_percent = ?")
            params.append(progress_percent)
        if latest_summary is not None:
            updates.append("latest_summary = ?")
            params.append(latest_summary)
        if result_payload is not None:
            updates.append("result_json = ?")
            params.append(json.dumps(result_payload, ensure_ascii=False))
        if error_message is not None:
            updates.append("error_message = ?")
            params.append(error_message)
        if started_at is not None:
            updates.append("started_at = ?")
            params.append(started_at)
        if finished_at is not None:
            updates.append("finished_at = ?")
            params.append(finished_at)

        now = utc_now_iso()
        updates.append("updated_at = ?")
        params.append(now)
        params.append(job_id)

        with self.connect() as conn:
            conn.execute(
                f"UPDATE research_jobs SET {', '.join(updates)} WHERE id = ?",
                tuple(params),
            )
            conn.commit()

    def add_research_journal_node(
        self,
        *,
        job_id: str,
        stage: str,
        node_type: str,
        status: str,
        input_payload: dict[str, Any],
        output_payload: dict[str, Any],
        reasoning: str,
        duration_ms: int = 0,
        parent_node_id: int | None = None,
        branch_id: str = "",
        selected: bool = False,
        intent: str = "draft",
        is_failed: bool = False,
        debug_depth: int = 0,
        metrics_payload: dict[str, Any] | None = None,
    ) -> int:
        now = utc_now_iso()
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO research_journal_nodes(
                    job_id,
                    stage,
                    node_type,
                    status,
                    input_json,
                    output_json,
                    reasoning,
                    duration_ms,
                    parent_node_id,
                    branch_id,
                    selected,
                    intent,
                    is_failed,
                    debug_depth,
                    metrics_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    stage,
                    node_type,
                    status,
                    json.dumps(input_payload, ensure_ascii=False),
                    json.dumps(output_payload, ensure_ascii=False),
                    reasoning,
                    duration_ms,
                    parent_node_id,
                    branch_id,
                    1 if selected else 0,
                    intent,
                    1 if is_failed else 0,
                    debug_depth,
                    json.dumps(metrics_payload or {}, ensure_ascii=False),
                    now,
                ),
            )
            conn.commit()
        return int(cursor.lastrowid)

    def list_research_journal_nodes(self, job_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    id,
                    job_id,
                    stage,
                    node_type,
                    status,
                    input_json,
                    output_json,
                    reasoning,
                    duration_ms,
                    parent_node_id,
                    branch_id,
                    selected,
                    intent,
                    is_failed,
                    debug_depth,
                    metrics_json,
                    created_at
                FROM research_journal_nodes
                WHERE job_id = ?
                ORDER BY id ASC
                """,
                (job_id,),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "job_id": row["job_id"],
                "stage": row["stage"],
                "node_type": row["node_type"],
                "status": row["status"],
                "input": json.loads(row["input_json"]),
                "output": json.loads(row["output_json"]),
                "reasoning": row["reasoning"],
                "duration_ms": row["duration_ms"],
                "parent_node_id": row["parent_node_id"],
                "branch_id": row["branch_id"],
                "selected": bool(row["selected"]),
                "intent": str(row["intent"] or "draft"),
                "is_failed": bool(row["is_failed"]),
                "debug_depth": int(row["debug_depth"] or 0),
                "metrics": json.loads(row["metrics_json"]) if row["metrics_json"] else {},
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def update_research_journal_node(
        self,
        node_id: int,
        *,
        status: str | None = None,
        input_payload: dict[str, Any] | None = None,
        output_payload: dict[str, Any] | None = None,
        reasoning: str | None = None,
        duration_ms: int | None = None,
        parent_node_id: int | None = None,
        branch_id: str | None = None,
        selected: bool | None = None,
        intent: str | None = None,
        is_failed: bool | None = None,
        debug_depth: int | None = None,
        metrics_payload: dict[str, Any] | None = None,
    ) -> None:
        updates: list[str] = []
        params: list[Any] = []

        if status is not None:
            updates.append("status = ?")
            params.append(status)
        if input_payload is not None:
            updates.append("input_json = ?")
            params.append(json.dumps(input_payload, ensure_ascii=False))
        if output_payload is not None:
            updates.append("output_json = ?")
            params.append(json.dumps(output_payload, ensure_ascii=False))
        if reasoning is not None:
            updates.append("reasoning = ?")
            params.append(reasoning)
        if duration_ms is not None:
            updates.append("duration_ms = ?")
            params.append(duration_ms)
        if parent_node_id is not None:
            updates.append("parent_node_id = ?")
            params.append(parent_node_id)
        if branch_id is not None:
            updates.append("branch_id = ?")
            params.append(branch_id)
        if selected is not None:
            updates.append("selected = ?")
            params.append(1 if selected else 0)
        if intent is not None:
            updates.append("intent = ?")
            params.append(intent)
        if is_failed is not None:
            updates.append("is_failed = ?")
            params.append(1 if is_failed else 0)
        if debug_depth is not None:
            updates.append("debug_depth = ?")
            params.append(debug_depth)
        if metrics_payload is not None:
            updates.append("metrics_json = ?")
            params.append(json.dumps(metrics_payload, ensure_ascii=False))

        if not updates:
            return

        params.append(node_id)
        with self.connect() as conn:
            conn.execute(
                f"UPDATE research_journal_nodes SET {', '.join(updates)} WHERE id = ?",
                tuple(params),
            )
            conn.commit()

    def add_llm_call_event(
        self,
        *,
        session_id: str | None,
        job_id: str | None,
        provider: str,
        model: str,
        operation: str,
        prompt_chars: int,
        response_chars: int,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        estimated_cost_usd: float | None,
        duration_ms: int,
        success: bool,
        error_message: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> int:
        now = utc_now_iso()
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO llm_call_events(
                    session_id,
                    job_id,
                    provider,
                    model,
                    operation,
                    prompt_chars,
                    response_chars,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    estimated_cost_usd,
                    duration_ms,
                    success,
                    error_message,
                    metadata_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    job_id,
                    provider,
                    model,
                    operation,
                    prompt_chars,
                    response_chars,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    estimated_cost_usd,
                    duration_ms,
                    1 if success else 0,
                    error_message,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    now,
                ),
            )
            conn.commit()
        return int(cursor.lastrowid)

    def list_llm_call_events(
        self,
        *,
        session_id: str | None = None,
        job_id: str | None = None,
    ) -> list[dict[str, Any]]:
        query = """
            SELECT
                id,
                session_id,
                job_id,
                provider,
                model,
                operation,
                prompt_chars,
                response_chars,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                estimated_cost_usd,
                duration_ms,
                success,
                error_message,
                metadata_json,
                created_at
            FROM llm_call_events
        """
        params: list[Any] = []
        conditions: list[str] = []
        if session_id is not None:
            conditions.append("session_id = ?")
            params.append(session_id)
        if job_id is not None:
            conditions.append("job_id = ?")
            params.append(job_id)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY id ASC"

        with self.connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [
            {
                "id": row["id"],
                "session_id": row["session_id"],
                "job_id": row["job_id"],
                "provider": row["provider"],
                "model": row["model"],
                "operation": row["operation"],
                "prompt_chars": row["prompt_chars"],
                "response_chars": row["response_chars"],
                "prompt_tokens": row["prompt_tokens"],
                "completion_tokens": row["completion_tokens"],
                "total_tokens": row["total_tokens"],
                "estimated_cost_usd": row["estimated_cost_usd"],
                "duration_ms": row["duration_ms"],
                "success": bool(row["success"]),
                "error_message": row["error_message"],
                "metadata": json.loads(row["metadata_json"]) if row["metadata_json"] else {},
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def _research_job_row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "session_id": row["session_id"],
            "status": row["status"],
            "provider": row["provider"],
            "llm_config": json.loads(row["llm_config_json"]) if row["llm_config_json"] else {},
            "approved_plan": json.loads(row["approved_plan_json"]),
            "current_stage": row["current_stage"],
            "progress_percent": int(row["progress_percent"] or 0),
            "latest_summary": row["latest_summary"],
            "result": json.loads(row["result_json"]) if row["result_json"] else None,
            "error_message": row["error_message"],
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "updated_at": row["updated_at"],
        }
