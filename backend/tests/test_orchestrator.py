from pathlib import Path

from app.config import Settings
from app.db import Database
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

    assert search_response.status == "search_results_ready"
    assert search_response.next_action == "select_property"
    assert any(block.type == "cards" for block in search_response.blocks)
    assert all(block.title != "問い合わせ下書き" for block in search_response.blocks)

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
    _, first_task_memory = db.get_memories(first_session_id)
    selected_property_id = first_task_memory["last_ranked_properties"][0]["property_id_norm"]

    reaction_response = orchestrator.execute_action(
        session_id=first_session_id,
        action_type="record_property_reaction",
        payload={"property_id": selected_property_id, "reaction": "favorite"},
    )

    assert reaction_response.status == "search_results_ready"

    second_session_id, _ = db.create_session(profile_id=profile_id)
    initial_response = orchestrator.build_session_initial_response(second_session_id)

    assert initial_response is not None
    assert initial_response.status == "awaiting_profile_resume"
    assert initial_response.assistant_message == "前回の条件を引き継ぎますか？"
    assert any("江東区" in str(block.content.get("body", "")) for block in initial_response.blocks)
