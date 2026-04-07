from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import ProviderName, load_settings
from app.db import Database
from app.models import (
    ActionRequest,
    AuditEventResponse,
    ChatMessageRequest,
    ChatMessageResponse,
    LLMCallEventResponse,
    ConfirmActionRequest,
    CreateSessionRequest,
    CreateSessionResponse,
    PreflightReport,
    ResearchStateResponse,
    SessionStateResponse,
)
from app.orchestrator import HousingOrchestrator
from app.preflight import run_preflight


logger = logging.getLogger(__name__)


def _research_worker(stop_event: threading.Event, orchestrator: HousingOrchestrator) -> None:
    while not stop_event.is_set():
        try:
            processed = orchestrator.process_next_research_job()
        except Exception:
            logger.exception("research worker loop failed")
            processed = False
        if not processed:
            stop_event.wait(1.0)


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
    app.state.research_stop_event = threading.Event()
    app.state.research_worker = threading.Thread(
        target=_research_worker,
        args=(app.state.research_stop_event, app.state.orchestrator),
        daemon=True,
        name="research-worker",
    )
    app.state.research_worker.start()

    if settings.run_preflight_on_startup and settings.preflight_fail_fast and not preflight_ok:
        raise RuntimeError("Preflight failed in fail-fast mode")

    try:
        yield
    finally:
        app.state.research_stop_event.set()
        app.state.research_worker.join(timeout=2.0)


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
def create_session(body: CreateSessionRequest | None = None) -> CreateSessionResponse:
    profile_id, _ = app.state.db.get_or_create_profile(body.profile_id if body else None)
    session_id, created_at = app.state.db.create_session(profile_id=profile_id)
    initial_response = None
    if not (body.fresh_start if body else False):
        initial_response = app.state.orchestrator.build_session_initial_response(session_id)
    if initial_response is not None:
        app.state.db.add_message(session_id, "assistant", initial_response.model_dump())
    return CreateSessionResponse(
        session_id=session_id,
        profile_id=profile_id,
        created_at=datetime.fromisoformat(created_at),
        initial_response=initial_response,
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
        profile_id=session["profile_id"],
        status=session["status"],
        pending_action=session["pending_action"],
        user_memory=user_memory,
        task_memory=task_memory,
        messages=messages,
    )


@app.get("/api/chat/sessions/{session_id}/research", response_model=ResearchStateResponse)
def get_research_state(session_id: str) -> ResearchStateResponse:
    if not app.state.db.session_exists(session_id):
        raise HTTPException(status_code=404, detail="session not found")
    return app.state.orchestrator.get_research_state(session_id)


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


@app.post("/api/chat/sessions/{session_id}/actions", response_model=ChatMessageResponse)
def execute_action(session_id: str, body: ActionRequest) -> ChatMessageResponse:
    if not app.state.db.session_exists(session_id):
        raise HTTPException(status_code=404, detail="session not found")

    try:
        return app.state.orchestrator.execute_action(
            session_id=session_id,
            action_type=body.action_type,
            payload=body.payload,
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


@app.get("/api/audit/sessions/{session_id}/llm-calls", response_model=list[LLMCallEventResponse])
def get_llm_call_events(session_id: str) -> list[LLMCallEventResponse]:
    if not app.state.db.session_exists(session_id):
        raise HTTPException(status_code=404, detail="session not found")

    events = app.state.db.list_llm_call_events(session_id=session_id)
    return [
        LLMCallEventResponse(
            id=item["id"],
            session_id=item["session_id"],
            job_id=item["job_id"],
            provider=item["provider"],
            model=item["model"],
            operation=item["operation"],
            prompt_chars=item["prompt_chars"],
            response_chars=item["response_chars"],
            prompt_tokens=item["prompt_tokens"],
            completion_tokens=item["completion_tokens"],
            total_tokens=item["total_tokens"],
            estimated_cost_usd=item["estimated_cost_usd"],
            duration_ms=item["duration_ms"],
            success=item["success"],
            error_message=item["error_message"],
            metadata=item["metadata"],
            created_at=datetime.fromisoformat(item["created_at"]),
        )
        for item in events
    ]
