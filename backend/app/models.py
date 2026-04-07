from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class PropertyNormalized(BaseModel):
    property_id_norm: str
    source_id: str
    building_name_norm: str
    address_norm: str
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


class DuplicateGroup(BaseModel):
    key: str
    property_ids: list[str]


class RankedProperty(BaseModel):
    property_id_norm: str
    score: float
    why_selected: str
    why_not_selected: str


class RiskItem(BaseModel):
    risk_type: Literal[
        "renewal_fee", "early_termination", "notice_period", "guarantor", "other"
    ]
    severity: Literal["high", "medium", "low"]
    evidence: str
    recommendation: str


class ChecklistItem(BaseModel):
    label: str
    checked: bool = False


class UIBlock(BaseModel):
    type: Literal["text", "table", "checklist", "cards", "warning"]
    title: str = ""
    content: dict[str, Any] = Field(default_factory=dict)


class ChatMessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    provider: Literal["openai", "gemini", "groq", "claude"] | None = None


class ChatMessageResponse(BaseModel):
    status: str
    assistant_message: str
    missing_slots: list[str] = Field(default_factory=list)
    next_action: str = ""
    blocks: list[UIBlock] = Field(default_factory=list)
    pending_confirmation: bool = False
    pending_action: dict[str, Any] | None = None


class CreateSessionResponse(BaseModel):
    session_id: str
    created_at: datetime


class ConfirmActionRequest(BaseModel):
    action_type: str
    approved: bool


class SessionStateResponse(BaseModel):
    session_id: str
    status: str
    pending_action: dict[str, Any] | None
    user_memory: dict[str, Any]
    task_memory: dict[str, Any]
    messages: list[dict[str, Any]]


class AuditEventResponse(BaseModel):
    id: int
    stage: str
    input: dict[str, Any]
    output: dict[str, Any]
    reasoning: str
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
