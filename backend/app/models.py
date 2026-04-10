from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class PropertyNormalized(BaseModel):
    property_id_norm: str
    source_id: str
    building_name: str = ""
    building_name_norm: str
    detail_url: str = ""
    image_url: str = ""
    address: str = ""
    address_norm: str
    area_name: str = ""
    nearest_station: str = ""
    line_name: str = ""
    layout: str = ""
    area_m2: float = 0.0
    rent: int = 0
    management_fee: int = 0
    deposit: int = 0
    key_money: int = 0
    station_walk_min: int = 0
    available_date: str = ""
    agency_name: str = ""
    notes: str = ""
    features: list[str] = Field(default_factory=list)
    floor_level: int = 0
    total_floors: int = 0
    has_autolock: bool | None = None
    contract_terms: dict[str, str] = Field(default_factory=dict)
    field_evidence: dict[str, str] = Field(default_factory=dict)
    field_confidence: dict[str, float] = Field(default_factory=dict)
    extraction_source: str = ""


class DuplicateGroup(BaseModel):
    key: str
    property_ids: list[str]
    confidence: float = 0.0
    reason: str = ""


class RankedProperty(BaseModel):
    property_id_norm: str
    score: float
    why_selected: str
    why_not_selected: str


class RiskItem(BaseModel):
    risk_type: Literal["renewal_fee", "early_termination", "notice_period", "guarantor", "other"]
    severity: Literal["high", "medium", "low"]
    evidence: str
    recommendation: str


class ChecklistItem(BaseModel):
    label: str
    checked: bool = False


class UIBlock(BaseModel):
    type: Literal[
        "text",
        "table",
        "checklist",
        "cards",
        "warning",
        "question",
        "actions",
        "plan",
        "timeline",
        "tree",
        "sources",
    ]
    title: str = ""
    content: dict[str, Any] = Field(default_factory=dict)
    display_label: str = ""


class PlannerAnswerPayload(BaseModel):
    slot: str = Field(min_length=1, max_length=80)
    value: str = Field(min_length=1, max_length=400)


class ChatMessageRequest(BaseModel):
    message: str = Field(default="", max_length=8000)
    planner_answers: list[PlannerAnswerPayload] = Field(default_factory=list)
    provider: Literal["openai", "gemini", "groq", "claude"] | None = None

    @model_validator(mode="after")
    def validate_message_or_answers(self) -> ChatMessageRequest:
        has_message = bool(self.message.strip())
        has_answers = bool(self.planner_answers)
        if not has_message and not has_answers:
            raise ValueError("message or planner_answers is required")
        return self


class LLMRouteConfigPayload(BaseModel):
    model: str = Field(min_length=1, max_length=200)


class LLMConfigPayload(BaseModel):
    preset: str = Field(default="custom", max_length=50)
    routes: dict[str, LLMRouteConfigPayload]


class SessionLLMConfigResponse(LLMConfigPayload):
    session_id: str
    editable: bool
    active_job_id: str | None = None


class LLMProviderCapability(BaseModel):
    key_present: bool
    reachable: bool
    default_model: str
    models: list[str] = Field(default_factory=list)
    details: str = ""


class LLMModelOptionResponse(BaseModel):
    model: str
    provider: str
    available: bool = True


class LLMRouteDefinitionResponse(BaseModel):
    key: str
    label: str
    description: str


class LLMCapabilitiesResponse(BaseModel):
    route_definitions: list[LLMRouteDefinitionResponse]
    providers: dict[str, LLMProviderCapability]
    models: list[LLMModelOptionResponse]
    default_config: LLMConfigPayload


class ChatMessageResponse(BaseModel):
    status: str
    assistant_message: str
    missing_slots: list[str] = Field(default_factory=list)
    next_action: str = ""
    blocks: list[UIBlock] = Field(default_factory=list)
    pending_confirmation: bool = False
    pending_action: dict[str, Any] | None = None
    status_label: str = ""


class CreateSessionResponse(BaseModel):
    session_id: str
    profile_id: str
    created_at: datetime
    initial_response: ChatMessageResponse | None = None


class CreateSessionRequest(BaseModel):
    profile_id: str | None = None
    fresh_start: bool = False


class ConfirmActionRequest(BaseModel):
    action_type: str
    approved: bool


class ActionRequest(BaseModel):
    action_type: str
    payload: dict[str, Any] = Field(default_factory=dict)


class SessionStateResponse(BaseModel):
    session_id: str
    profile_id: str
    status: str
    pending_action: dict[str, Any] | None
    user_memory: dict[str, Any]
    task_memory: dict[str, Any]
    messages: list[dict[str, Any]]


class ResearchStateResponse(BaseModel):
    session_id: str
    job_id: str | None = None
    status: str
    current_stage: str = ""
    progress_percent: int = 0
    latest_summary: str = ""
    response: ChatMessageResponse | None = None


class AuditEventResponse(BaseModel):
    id: int
    stage: str
    input: dict[str, Any]
    output: dict[str, Any]
    reasoning: str
    created_at: datetime


class LLMCallEventResponse(BaseModel):
    id: int
    session_id: str | None = None
    job_id: str | None = None
    provider: str
    model: str
    operation: str
    prompt_chars: int
    response_chars: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float | None = None
    duration_ms: int
    success: bool
    error_message: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class PreflightProviderReport(BaseModel):
    key_present: bool
    reachable: bool
    model_valid: bool
    details: str


class PreflightReport(BaseModel):
    strict_mode: bool
    brave_reachable: bool
    providers: dict[str, PreflightProviderReport]
