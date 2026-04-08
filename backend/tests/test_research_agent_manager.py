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


def test_research_job_records_branch_tree_and_evaluations(tmp_path: Path):
    database_path = str(tmp_path / "housing.db")
    db = Database(database_path)
    db.init()
    orchestrator = HousingOrchestrator(settings=build_settings(database_path), db=db)

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

    job = db.get_latest_research_job(session_id)
    assert job is not None

    nodes = db.list_research_journal_nodes(job["id"])
    completed_stage_names = [
        node["stage"]
        for node in nodes
        if node["node_type"] == "stage" and node["status"] == "completed"
    ]
    assert completed_stage_names == [
        "plan_finalize",
        "tree_search",
        "synthesize",
    ]

    search_roots = [node for node in nodes if node["node_type"] == "search_root"]
    assert len(search_roots) == 1

    candidate_nodes = [node for node in nodes if node["node_type"] == "search_candidate"]
    assert len(candidate_nodes) >= 2
    assert all(node["branch_id"] for node in candidate_nodes)
    assert all(node["parent_node_id"] is not None for node in candidate_nodes)

    branch_selections = [
        node for node in nodes if node["node_type"] == "search_selection" and node["selected"]
    ]
    assert len(branch_selections) == 1
    selected_node = branch_selections[0]
    assert selected_node["parent_node_id"] is not None
    assert selected_node["metrics"]["status"] == "completed"
    assert float(selected_node["metrics"]["branch_score"]) >= 0.0

    _, task_memory = db.get_memories(session_id)
    assert task_memory["selected_branch_id"] == selected_node["branch_id"]
    assert len(task_memory["branch_summaries"]) >= 2
    assert task_memory["selected_path"]
    assert task_memory["search_tree_summary"]["executed_node_count"] >= 2
    assert isinstance(task_memory["pruned_nodes"], list)
    assert task_memory["offline_evaluation"]["readiness"] in {"low", "medium", "high"}
    assert task_memory["failure_summary"]["recommendations"]
    assert task_memory["last_research_summary"]

    audit_stages = [event["stage"] for event in db.list_audit_events(session_id)]
    assert "search_normalize" in audit_stages
    assert "ranking" in audit_stages
    assert "offline_evaluator" in audit_stages

    research_state = orchestrator.get_research_state(session_id)
    assert research_state.response is not None
    block_titles = [block.title for block in research_state.response.blocks]
    assert "探索分岐の比較" in block_titles
    assert "オフライン評価" in block_titles
