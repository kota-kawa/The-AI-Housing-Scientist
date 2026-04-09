from pathlib import Path

from app.config import Settings
from app.db import Database
from app.llm.base import LLMAdapter
from app.orchestrator import HousingOrchestrator


def build_settings(database_path: str, *, with_keys: bool = False) -> Settings:
    return Settings(
        app_env="test",
        database_path=database_path,
        llm_default_provider="openai",
        model_strict_mode=False,
        run_preflight_on_startup=False,
        preflight_fail_fast=False,
        llm_timeout_seconds=3,
        llm_max_retries=0,
        openai_api_key="openai-key" if with_keys else "",
        gemini_api_key="gemini-key" if with_keys else "",
        groq_api_key="",
        claude_api_key="claude-key" if with_keys else "",
        brave_search_api_key="",
        openai_model="gpt-5.4-mini",
        gemini_model="gemini-3-flash",
        groq_model_primary="openai/gpt-oss-120b",
        groq_model_secondary="qwen/qwen3-32b",
        claude_model="claude-sonnet-4-6",
    )


class FakePlannerRouteAdapter(LLMAdapter):
    def generate_text(self, *, system: str, user: str, temperature: float = 0.2) -> str:
        raise AssertionError("generate_text should not be called in planner route tests")

    def generate_structured(
        self,
        *,
        system: str,
        user: str,
        schema: dict,
        temperature: float = 0.2,
    ) -> dict:
        return {
            "intent": "search",
            "user_memory": {
                "target_area": "江東区",
                "budget_max": 120000,
                "station_walk_max": 7,
                "layout_preference": "1LDK",
                "move_in_date": None,
                "must_conditions": [],
                "nice_to_have": [],
            },
            "missing_slots": [],
            "follow_up_questions": [],
            "next_action": "search_and_compare",
            "seed_queries": [
                "江東区 賃貸 12万円 1LDK",
                "江東区で家賃12万円以下の1LDK賃貸",
            ],
            "research_plan": {
                "summary": "条件に近い候補を比較します。",
                "goal": "問い合わせ候補を数件まで絞り込みます。",
                "strategy": [
                    "条件に合う候補を広めに集めます。",
                    "詳細ページで不足情報を確認します。",
                    "一致度の高い候補から並べます。",
                ],
                "rationale": "最初に母集団を作ってから絞る方が条件差分を見やすいためです。",
            },
            "condition_reasons": {
                "budget_max": "",
                "target_area": "",
                "station_walk_max": "",
                "move_in_date": "",
                "layout_preference": "",
                "must_conditions": "",
                "nice_to_have": "",
            },
        }

    def list_models(self) -> list[str]:
        return ["fake-planner-model"]


def test_session_llm_config_defaults_and_custom_update(tmp_path: Path, monkeypatch):
    database_path = str(tmp_path / "housing.db")
    db = Database(database_path)
    db.init()
    orchestrator = HousingOrchestrator(
        settings=build_settings(database_path, with_keys=True), db=db
    )
    session_id, _ = db.create_session()

    monkeypatch.setattr(
        orchestrator,
        "_list_models_for_provider",
        lambda provider: {
            "openai": ["gpt-5.4-mini"],
            "gemini": ["gemini-3-flash"],
            "groq": ["openai/gpt-oss-120b"],
            "claude": ["claude-sonnet-4-6"],
        }[provider],
    )

    default_config = orchestrator.get_session_llm_config(session_id)
    assert default_config["routes"]["planner"]["model"] == "gpt-5.4-mini"
    assert default_config["routes"]["research_default"]["model"] == "gpt-5.4-mini"
    assert default_config["routes"]["communication"]["model"] == "claude-sonnet-4-6"
    assert default_config["routes"]["risk_check"]["model"] == "gemini-3-flash"
    capabilities = orchestrator.get_llm_capabilities()
    assert [item["model"] for item in capabilities["models"]] == [
        "gpt-5.4-mini",
        "gemini-3-flash",
        "claude-sonnet-4-6",
        "openai/gpt-oss-120b",
        "qwen/qwen3-32b",
    ]

    updated = orchestrator.update_session_llm_config(
        session_id,
        {
            "preset": "custom",
            "routes": {
                "planner": {"model": "claude-sonnet-4-6"},
                "research_default": {"model": "gpt-5.4-mini"},
                "communication": {"model": "gpt-5.4-mini"},
                "risk_check": {"model": "gemini-3-flash"},
            },
        },
    )

    assert updated["preset"] == "custom"
    assert updated["routes"]["planner"]["model"] == "claude-sonnet-4-6"
    assert updated["routes"]["communication"]["model"] == "gpt-5.4-mini"

    _, task_memory = db.get_memories(session_id)
    assert task_memory["draft_llm_config"]["routes"]["planner"]["model"] == "claude-sonnet-4-6"


def test_research_job_uses_llm_config_snapshot_per_execution(tmp_path: Path):
    database_path = str(tmp_path / "housing.db")
    db = Database(database_path)
    db.init()
    orchestrator = HousingOrchestrator(
        settings=build_settings(database_path, with_keys=False), db=db
    )
    planner_adapter = FakePlannerRouteAdapter()
    orchestrator._get_adapter_for_route = lambda **kwargs: (
        planner_adapter if kwargs.get("route_key") == "planner" else None
    )
    session_id, _ = db.create_session()

    orchestrator.update_session_llm_config(
        session_id,
        {
            "preset": "custom",
            "routes": {
                "planner": {"model": "gpt-5.4-mini"},
                "research_default": {"model": "claude-sonnet-4-6"},
                "communication": {"model": "gpt-5.4-mini"},
                "risk_check": {"model": "gemini-3-flash"},
            },
        },
    )

    search_response = orchestrator.process_user_message(
        session_id=session_id,
        message="江東区で家賃12万円以下、駅徒歩7分以内の1LDKを探しています",
        provider="openai",
    )
    assert search_response.status == "awaiting_plan_confirmation"

    queued_response = orchestrator.execute_action(
        session_id=session_id,
        action_type="approve_research_plan",
        payload={},
    )
    assert queued_response.status == "research_queued"

    locked_config = orchestrator.get_session_llm_config(session_id)
    assert locked_config["editable"] is False

    first_job = db.get_latest_research_job(session_id)
    assert first_job is not None
    assert first_job["provider"] == "claude"
    assert first_job["llm_config"]["routes"]["research_default"]["model"] == "claude-sonnet-4-6"

    assert orchestrator.process_next_research_job() is True

    orchestrator.update_session_llm_config(
        session_id,
        {
            "preset": "custom",
            "routes": {
                "planner": {"model": "gpt-5.4-mini"},
                "research_default": {"model": "gemini-3-flash"},
                "communication": {"model": "claude-sonnet-4-6"},
                "risk_check": {"model": "gpt-5.4-mini"},
            },
        },
    )

    retry_response = orchestrator.execute_action(
        session_id=session_id,
        action_type="retry_research_job",
        payload={},
    )
    assert retry_response.status == "research_queued"

    second_job = db.get_latest_research_job(session_id)
    assert second_job is not None
    assert second_job["id"] != first_job["id"]
    assert second_job["provider"] == "gemini"
    assert second_job["llm_config"]["routes"]["research_default"]["model"] == "gemini-3-flash"
