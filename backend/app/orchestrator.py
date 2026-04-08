from __future__ import annotations

from app.llm_config_manager import LLMConfigManagerMixin
from app.orchestrator_modules import (
    RESEARCH_STAGE_ORDER,
    OrchestratorActionsMixin,
    OrchestratorCoreMixin,
    OrchestratorPlanningMixin,
    OrchestratorPresentationMixin,
    OrchestratorResearchMixin,
    _generate_llm_guidance_message,
    _generate_llm_plan_presentation,
    _generate_llm_resume_body,
    _generate_response_labels,
    _normalize_display_text,
    _normalize_display_texts,
)


class HousingOrchestrator(
    OrchestratorActionsMixin,
    OrchestratorResearchMixin,
    OrchestratorPresentationMixin,
    OrchestratorPlanningMixin,
    OrchestratorCoreMixin,
    LLMConfigManagerMixin,
):
    """Session orchestration entrypoint composed from focused mixins."""


__all__ = [
    "HousingOrchestrator",
    "RESEARCH_STAGE_ORDER",
    "_generate_llm_guidance_message",
    "_generate_llm_plan_presentation",
    "_generate_llm_resume_body",
    "_generate_response_labels",
    "_normalize_display_text",
    "_normalize_display_texts",
]
