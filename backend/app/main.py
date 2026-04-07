from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import ProviderName, load_settings
from app.db import Database
from app.models import (
    AuditEventResponse,
    ChatMessageRequest,
    ChatMessageResponse,
    ConfirmActionRequest,
    CreateSessionResponse,
    PreflightReport,
    SessionStateResponse,
)
from app.orchestrator import HousingOrchestrator
from app.preflight import run_preflight


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    db = Database(settings.database_path)
    db.init()

    preflight_report, preflight_ok = run_preflight(settings)

    app.state.settings = settings
    app.state.db = db
    app.state.orchestrator = HousingOrchestrator(settings=settings, db=db)
    app.state.preflight_report = preflight_report
    app.state.preflight_ok = preflight_ok

    if settings.run_preflight_on_startup and settings.preflight_fail_fast and not preflight_ok:
        raise RuntimeError("Preflight failed in fail-fast mode")

    yield


app = FastAPI(title="The-AI-Housing-Scientist", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/system/preflight", response_model=PreflightReport)
def get_preflight() -> PreflightReport:
    return app.state.preflight_report


@app.post("/api/chat/sessions", response_model=CreateSessionResponse)
def create_session() -> CreateSessionResponse:
    session_id, created_at = app.state.db.create_session()
    return CreateSessionResponse(
        session_id=session_id,
        created_at=datetime.fromisoformat(created_at),
    )


@app.get("/api/chat/sessions/{session_id}", response_model=SessionStateResponse)
def get_session_state(session_id: str) -> SessionStateResponse:
    session = app.state.db.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")

    user_memory, task_memory = app.state.db.get_memories(session_id)
    messages = app.state.db.list_messages(session_id)

    return SessionStateResponse(
        session_id=session_id,
        status=session["status"],
        pending_action=session["pending_action"],
        user_memory=user_memory,
        task_memory=task_memory,
        messages=messages,
    )


@app.post("/api/chat/sessions/{session_id}/messages", response_model=ChatMessageResponse)
def post_message(session_id: str, body: ChatMessageRequest) -> ChatMessageResponse:
    if not app.state.db.session_exists(session_id):
        raise HTTPException(status_code=404, detail="session not found")

    provider: ProviderName = body.provider or app.state.settings.llm_default_provider

    app.state.db.add_message(session_id, "user", {"message": body.message, "provider": provider})

    try:
        return app.state.orchestrator.process_user_message(
            session_id=session_id,
            message=body.message,
            provider=provider,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/chat/sessions/{session_id}/actions/confirm", response_model=ChatMessageResponse)
def confirm_action(session_id: str, body: ConfirmActionRequest) -> ChatMessageResponse:
    if not app.state.db.session_exists(session_id):
        raise HTTPException(status_code=404, detail="session not found")

    try:
        return app.state.orchestrator.confirm_action(
            session_id=session_id,
            action_type=body.action_type,
            approved=body.approved,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/audit/sessions/{session_id}", response_model=list[AuditEventResponse])
def get_audit_log(session_id: str) -> list[AuditEventResponse]:
    if not app.state.db.session_exists(session_id):
        raise HTTPException(status_code=404, detail="session not found")

    events = app.state.db.list_audit_events(session_id)
    return [
        AuditEventResponse(
            id=item["id"],
            stage=item["stage"],
            input=item["input"],
            output=item["output"],
            reasoning=item["reasoning"],
            created_at=datetime.fromisoformat(item["created_at"]),
        )
        for item in events
    ]
