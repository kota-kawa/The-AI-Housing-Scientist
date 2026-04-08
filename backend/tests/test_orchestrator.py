import json
from pathlib import Path

from app.config import Settings
from app.db import Database
from app.llm.base import LLMAdapter
from app.main import app, create_session as create_session_endpoint
from app.models import CreateSessionRequest
from app.orchestrator import HousingOrchestrator


def build_settings(database_path: str) -> Settings:
    return Settings(
        app_env="test",
        database_path=database_path,
        llm_default_provider="openai",
        model_strict_mode=False,
        run_preflight_on_startup=False,
        preflight_fail_fast=False,
        llm_timeout_seconds=3,
        llm_max_retries=0,
        openai_api_key="",
        gemini_api_key="",
        groq_api_key="",
        claude_api_key="",
        brave_search_api_key="",
        openai_model="gpt-5.4-mini",
        gemini_model="gemini-3-flash",
        groq_model_primary="openai/gpt-oss-120b",
        groq_model_secondary="qwen/qwen3-32b",
        claude_model="claude-sonnet-4-6",
    )


class FakeResearchSummaryAdapter(LLMAdapter):
    def __init__(self, summary: str):
        self.summary = summary
        self.last_text_user = ""

    def generate_text(self, *, system: str, user: str, temperature: float = 0.2) -> str:
        self.last_text_user = user
        return self.summary

    def generate_structured(
        self,
        *,
        system: str,
        user: str,
        schema: dict,
        temperature: float = 0.2,
    ) -> dict:
        properties = schema.get("properties", {})
        if "branches" in properties:
            return {
                "branches": [
                    {
                        "branch_id": "balanced",
                        "query_suggestions": [],
                        "description_hint": "バランス重視",
                    },
                    {
                        "branch_id": "strict",
                        "query_suggestions": [],
                        "description_hint": "厳しめ条件",
                    },
                    {
                        "branch_id": "broad",
                        "query_suggestions": [],
                        "description_hint": "広めに収集",
                    },
                ],
                "summary": "3本の探索分岐を維持します。",
            }
        if "assessments" in properties:
            return {"assessments": []}
        raise AssertionError(f"unexpected schema keys: {list(properties.keys())}")

    def list_models(self) -> list[str]:
        return ["fake-research-model"]


class FakePlannerRouteAdapter(LLMAdapter):
    def __init__(self, payload: dict):
        self.payload = payload

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
        return self.payload

    def list_models(self) -> list[str]:
        return ["fake-planner-model"]


def test_orchestrator_stage_flow_is_interactive(tmp_path: Path):
    database_path = str(tmp_path / "housing.db")
    db = Database(database_path)
    db.init()
    orchestrator = HousingOrchestrator(settings=build_settings(database_path), db=db)

    session_id, _ = db.create_session()

    search_response = orchestrator.process_user_message(
        session_id=session_id,
        message="江東区で家賃12万円以下、駅徒歩7分以内の1LDKを探しています",
        provider="openai",
    )

    assert search_response.status == "awaiting_plan_confirmation"
    assert search_response.next_action == "approve_research_plan"
    assert any(block.type == "plan" for block in search_response.blocks)

    queued_response = orchestrator.execute_action(
        session_id=session_id,
        action_type="approve_research_plan",
        payload={},
    )

    assert queued_response.status == "research_queued"
    assert any(block.type == "timeline" for block in queued_response.blocks)

    assert orchestrator.process_next_research_job() is True
    research_state = orchestrator.get_research_state(session_id)
    assert research_state.status == "completed"
    assert research_state.response is not None
    assert research_state.response.status == "research_completed"
    assert any(block.type == "cards" for block in research_state.response.blocks)

    user_memory, task_memory = db.get_memories(session_id)
    selected_property_id = task_memory["last_ranked_properties"][0]["property_id_norm"]
    comparison_response = orchestrator.execute_action(
        session_id=session_id,
        action_type="compare_selected_properties",
        payload={
            "property_ids": [
                item["property_id_norm"] for item in task_memory["last_ranked_properties"][:2]
            ]
        },
    )

    assert comparison_response.status == "research_completed"
    assert any(block.title == "選択物件の比較表" for block in comparison_response.blocks)

    inquiry_response = orchestrator.execute_action(
        session_id=session_id,
        action_type="generate_inquiry",
        payload={"property_id": selected_property_id},
    )

    assert inquiry_response.status == "inquiry_draft_ready"
    assert inquiry_response.pending_confirmation is True
    assert any(block.title == "問い合わせ下書き" for block in inquiry_response.blocks)

    contract_prompt = orchestrator.execute_action(
        session_id=session_id,
        action_type="start_contract_review",
        payload={"property_id": selected_property_id},
    )

    assert contract_prompt.status == "awaiting_contract_text"
    assert contract_prompt.next_action == "paste_contract_text"

    risk_response = orchestrator.process_user_message(
        session_id=session_id,
        message="更新料1ヶ月。短期解約違約金あり。解約予告2か月前。保証会社加入必須。",
        provider="openai",
    )

    assert risk_response.status == "risk_check_completed"
    assert any(block.title == "契約リスク一覧" for block in risk_response.blocks)
    _, updated_task_memory = db.get_memories(session_id)
    assert updated_task_memory["awaiting_contract_text"] is False
    assert updated_task_memory["risk_items"]


def test_orchestrator_uses_llm_generated_plan_content(tmp_path: Path):
    database_path = str(tmp_path / "housing.db")
    db = Database(database_path)
    db.init()
    orchestrator = HousingOrchestrator(settings=build_settings(database_path), db=db)
    planner_adapter = FakePlannerRouteAdapter(
        {
            "intent": "search",
            "extracted_slots": {
                "target_area": "江東区",
                "budget_max": 120000,
                "station_walk_max": 7,
                "layout_preference": "1LDK",
                "move_in_date": None,
                "must_conditions": ["ペット可"],
                "nice_to_have": ["在宅ワーク向け"],
            },
            "follow_up_questions": [
                {
                    "slot": "move_in_date",
                    "question": "ペットと一緒に住む前提で、いつ頃から入居したいですか？",
                    "examples": ["できるだけ早く", "来月中", "6月ごろ"],
                }
            ],
            "seed_queries": [
                "江東区 賃貸 12万円 1LDK ペット可",
                "江東区で家賃12万円以下、ペット可で在宅ワークしやすい賃貸",
                "江東区 ペット可 1LDK 在宅ワーク 賃貸",
            ],
            "research_plan": {
                "summary": "江東区でペット可かつ在宅ワークしやすい1LDKを優先して調べます。",
                "goal": "早めに問い合わせできる候補を3件前後まで絞り込む",
                "strategy": [
                    "ペット可物件は希少なので江東区内で候補を広めに集めます。",
                    "在宅ワーク向け設備や回線条件は詳細ページで重点確認します。",
                    "家賃と駅距離の両立度を見て問い合わせ候補を上位化します。",
                ],
                "rationale": "希少条件があるため、母集団を先に確保してから絞り込む進め方を取ります。",
            },
            "condition_reasons": {
                "target_area": "生活圏を固定したい条件なので検索の起点にします。",
                "budget_max": "毎月の予算上限なので厳格に見ます。",
                "station_walk_max": "通勤や移動負担に直結するためです。",
                "move_in_date": "",
                "layout_preference": "仕事スペースを確保できるかの判断軸にします。",
                "must_conditions": "ペット可は候補数が少ないため優先度が高い条件です。",
                "nice_to_have": "在宅ワーク条件は比較の差が出やすいので加点軸にします。",
            },
        }
    )
    orchestrator._get_adapter_for_route = (
        lambda **kwargs: planner_adapter if kwargs.get("route_key") == "planner" else None
    )

    session_id, _ = db.create_session()
    response = orchestrator.process_user_message(
        session_id=session_id,
        message="江東区で家賃12万円以下、ペット可、在宅ワークしやすい1LDKを探したい",
        provider="openai",
    )

    assert response.status == "awaiting_plan_confirmation"
    plan_block = next(block for block in response.blocks if block.type == "plan")
    assert plan_block.content["summary"] == "江東区でペット可かつ在宅ワークしやすい1LDKを優先して調べます。"
    assert plan_block.content["rationale"] == "希少条件があるため、母集団を先に確保してから絞り込む進め方を取ります。"
    assert plan_block.content["seed_queries"][1] == "江東区で家賃12万円以下、ペット可で在宅ワークしやすい賃貸"
    must_condition = next(
        item for item in plan_block.content["conditions"] if item["label"] == "必須条件"
    )
    assert must_condition["reason"] == "ペット可は候補数が少ないため優先度が高い条件です。"


def test_profile_memory_is_available_across_sessions(tmp_path: Path):
    database_path = str(tmp_path / "housing.db")
    db = Database(database_path)
    db.init()
    orchestrator = HousingOrchestrator(settings=build_settings(database_path), db=db)

    profile_id, _ = db.get_or_create_profile("local-profile-1")
    first_session_id, _ = db.create_session(profile_id=profile_id)

    orchestrator.process_user_message(
        session_id=first_session_id,
        message="江東区で家賃12万円以下、駅徒歩7分以内の1LDKを探しています",
        provider="openai",
    )
    orchestrator.execute_action(
        session_id=first_session_id,
        action_type="approve_research_plan",
        payload={},
    )
    assert orchestrator.process_next_research_job() is True
    _, first_task_memory = db.get_memories(first_session_id)
    selected_property_id = first_task_memory["last_ranked_properties"][0]["property_id_norm"]

    reaction_response = orchestrator.execute_action(
        session_id=first_session_id,
        action_type="record_property_reaction",
        payload={"property_id": selected_property_id, "reaction": "favorite"},
    )

    assert reaction_response.status == "research_completed"

    second_session_id, _ = db.create_session(profile_id=profile_id)
    initial_response = orchestrator.build_session_initial_response(second_session_id)

    assert initial_response is not None
    assert initial_response.status == "awaiting_profile_resume"
    assert initial_response.assistant_message == "前回の条件を引き継ぎますか？"
    assert any("江東区" in str(block.content.get("body", "")) for block in initial_response.blocks)


def test_research_completed_response_prefers_llm_summary(tmp_path: Path):
    database_path = str(tmp_path / "housing.db")
    db = Database(database_path)
    db.init()
    orchestrator = HousingOrchestrator(settings=build_settings(database_path), db=db)
    summary = (
        "江東区で家賃12万円以内・駅徒歩7分以内・1LDKの条件では3件を比較できました。"
        "最上位候補は駅徒歩4分で条件に最も近く、まずは問い合わせに進める候補です。"
    )
    adapter = FakeResearchSummaryAdapter(summary)
    orchestrator._get_adapter_for_route = (
        lambda **kwargs: adapter if kwargs.get("route_key") == "research_default" else None
    )

    session_id, _ = db.create_session()
    orchestrator.process_user_message(
        session_id=session_id,
        message="江東区で家賃12万円以下、駅徒歩7分以内の1LDKを探しています",
        provider="openai",
    )
    orchestrator.execute_action(
        session_id=session_id,
        action_type="approve_research_plan",
        payload={},
    )

    assert orchestrator.process_next_research_job() is True

    research_state = orchestrator.get_research_state(session_id)
    assert research_state.latest_summary == summary
    assert research_state.response is not None
    assert research_state.response.assistant_message == summary
    summary_block = next(
        block for block in research_state.response.blocks if block.title == "調査サマリー"
    )
    assert summary_block.content["body"] == summary

    _, task_memory = db.get_memories(session_id)
    assert task_memory["last_research_summary"] == summary


def test_research_summary_prompt_includes_branch_tradeoffs_and_followups(tmp_path: Path):
    database_path = str(tmp_path / "housing.db")
    db = Database(database_path)
    db.init()
    orchestrator = HousingOrchestrator(settings=build_settings(database_path), db=db)
    adapter = FakeResearchSummaryAdapter(
        "結論: 問い合わせ候補があります。\n"
        "理由: 条件一致度が最も高い分岐を採用しました。\n"
        "懸念: 一部の条件は再確認が必要です。\n"
        "次の確認事項: 初期費用と契約条件を確認してください。"
    )
    orchestrator._get_adapter_for_route = (
        lambda **kwargs: adapter if kwargs.get("route_key") == "research_default" else None
    )

    session_id, _ = db.create_session()
    orchestrator.process_user_message(
        session_id=session_id,
        message="江東区で家賃12万円以下、駅徒歩7分以内の1LDKを探しています",
        provider="openai",
    )
    orchestrator.execute_action(
        session_id=session_id,
        action_type="approve_research_plan",
        payload={},
    )

    assert orchestrator.process_next_research_job() is True

    payload = json.loads(adapter.last_text_user)
    assert payload["selected_branch"]["branch_id"]
    assert payload["alternative_branches"]
    assert payload["failure_summary"]["recommendations"]
    assert payload["confirmation_items"]


def test_fresh_start_session_skips_profile_resume_prompt(tmp_path: Path):
    database_path = str(tmp_path / "housing.db")
    db = Database(database_path)
    db.init()
    orchestrator = HousingOrchestrator(settings=build_settings(database_path), db=db)

    profile_id, _ = db.get_or_create_profile("local-profile-2")
    first_session_id, _ = db.create_session(profile_id=profile_id)
    orchestrator.process_user_message(
        session_id=first_session_id,
        message="吉祥寺で家賃12万円以下、駅徒歩7分以内、在宅ワーク向けで探したい",
        provider="openai",
    )
    orchestrator.execute_action(
        session_id=first_session_id,
        action_type="approve_research_plan",
        payload={},
    )
    assert orchestrator.process_next_research_job() is True

    app.state.db = db
    app.state.orchestrator = orchestrator

    response = create_session_endpoint(
        CreateSessionRequest(profile_id=profile_id, fresh_start=True)
    )

    assert response.profile_id == profile_id
    assert response.initial_response is None


def test_retry_and_reaction_feed_strategy_memory(tmp_path: Path):
    database_path = str(tmp_path / "housing.db")
    db = Database(database_path)
    db.init()
    orchestrator = HousingOrchestrator(settings=build_settings(database_path), db=db)

    profile_id, _ = db.get_or_create_profile("strategy-profile-1")
    session_id, _ = db.create_session(profile_id=profile_id)

    orchestrator.process_user_message(
        session_id=session_id,
        message="江東区で家賃12万円以下、駅徒歩7分以内の1LDKを探しています",
        provider="openai",
    )
    orchestrator.execute_action(
        session_id=session_id,
        action_type="approve_research_plan",
        payload={},
    )
    assert orchestrator.process_next_research_job() is True

    _, task_memory = db.get_memories(session_id)
    selected_property_id = task_memory["last_ranked_properties"][0]["property_id_norm"]
    orchestrator.execute_action(
        session_id=session_id,
        action_type="record_property_reaction",
        payload={"property_id": selected_property_id, "reaction": "favorite"},
    )

    profile = db.get_profile(profile_id)
    assert profile is not None
    strategy_memory = profile["profile_memory"].get("strategy_memory", {})
    assert strategy_memory["preferred_strategy_tags"]

    retry_response = orchestrator.execute_action(
        session_id=session_id,
        action_type="retry_research_job",
        payload={},
    )
    assert retry_response.status == "research_queued"

    latest_job = db.get_latest_research_job(session_id)
    assert latest_job is not None
    retry_context = latest_job["approved_plan"].get("retry_context", {})
    assert retry_context["selected_path"]
    assert retry_context["top_issues"] == task_memory["failure_summary"]["top_issues"]
