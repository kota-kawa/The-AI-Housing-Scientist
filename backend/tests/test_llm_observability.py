from pathlib import Path

import pytest

from app.db import Database
from app.llm.base import LLMAdapter, LLMUsage
from app.llm.observability import (
    DatabaseLLMObserver,
    LLMObservationContext,
    ObservedLLMAdapter,
    build_cost_estimator,
)
from app.main import app, get_llm_call_events


class FakeLLMAdapter(LLMAdapter):
    def __init__(self):
        self._last_usage: LLMUsage | None = None

    def generate_text(self, *, system: str, user: str, temperature: float = 0.2) -> str:
        self._last_usage = LLMUsage(prompt_tokens=120, completion_tokens=0, total_tokens=120)
        raise RuntimeError("boom")

    def generate_structured(
        self,
        *,
        system: str,
        user: str,
        schema: dict,
        temperature: float = 0.2,
    ) -> dict:
        self._last_usage = LLMUsage(prompt_tokens=100, completion_tokens=40, total_tokens=140)
        return {"target_area": "江東区"}

    def list_models(self) -> list[str]:
        return ["fake-model"]

    def get_last_usage(self) -> LLMUsage | None:
        return self._last_usage


def test_observed_llm_adapter_records_success_and_failure(tmp_path: Path):
    db = Database(str(tmp_path / "housing.db"))
    db.init()
    session_id, _ = db.create_session()
    observed = ObservedLLMAdapter(
        wrapped=FakeLLMAdapter(),
        observer=DatabaseLLMObserver(
            db,
            cost_estimator=build_cost_estimator(
                '{"openai": {"fake-model": {"prompt_per_1m_tokens_usd": 1.0, "completion_per_1m_tokens_usd": 2.0}}}'
            ),
        ),
        context_factory=lambda operation, metadata: LLMObservationContext(
            session_id=session_id,
            job_id="job-1",
            operation=f"planner:{operation}",
            provider="openai",
            model="fake-model",
            metadata=metadata,
        ),
    )

    payload = observed.generate_structured(
        system="system",
        user="user",
        schema={
            "type": "object",
            "properties": {"target_area": {"type": "string"}},
        },
        temperature=0.0,
    )
    assert payload == {"target_area": "江東区"}

    with pytest.raises(RuntimeError, match="boom"):
        observed.generate_text(system="system", user="user", temperature=0.3)

    events = db.list_llm_call_events(session_id=session_id, job_id="job-1")
    assert len(events) == 2

    assert events[0]["operation"] == "planner:generate_structured"
    assert events[0]["success"] is True
    assert events[0]["metadata"]["schema_keys"] == ["target_area"]
    assert events[0]["response_chars"] > 0
    assert events[0]["prompt_tokens"] == 100
    assert events[0]["completion_tokens"] == 40
    assert events[0]["total_tokens"] == 140
    assert events[0]["estimated_cost_usd"] == pytest.approx(0.00018)

    assert events[1]["operation"] == "planner:generate_text"
    assert events[1]["success"] is False
    assert events[1]["error_message"] == "boom"
    assert events[1]["metadata"]["temperature"] == 0.3
    assert events[1]["prompt_tokens"] == 120
    assert events[1]["completion_tokens"] == 0

    app.state.db = db
    api_events = get_llm_call_events(session_id)
    assert len(api_events) == 2
    assert api_events[0].operation == "planner:generate_structured"
    assert api_events[0].estimated_cost_usd == pytest.approx(0.00018)
