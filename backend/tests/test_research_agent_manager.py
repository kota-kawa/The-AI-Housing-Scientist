from pathlib import Path

from app.config import Settings
from app.db import Database
from app.llm.base import LLMAdapter
from app.orchestrator import HousingOrchestrator
from app.research.agent_manager import (
    HousingResearchAgentManager,
    ResearchExecutionState,
    SearchNodeArtifacts,
    SearchNodePlan,
)


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


def test_research_job_records_branch_tree_and_evaluations(tmp_path: Path):
    database_path = str(tmp_path / "housing.db")
    db = Database(database_path)
    db.init()
    orchestrator = HousingOrchestrator(settings=build_settings(database_path), db=db)
    planner_adapter = FakePlannerRouteAdapter()
    orchestrator._get_adapter_for_route = (
        lambda **kwargs: planner_adapter if kwargs.get("route_key") == "planner" else None
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
    tree_block = next((block for block in research_state.response.blocks if block.type == "tree"), None)
    assert tree_block is not None
    tree_nodes = tree_block.content["nodes"]
    assert isinstance(tree_nodes, list)
    assert len(tree_nodes) >= len(search_roots) + len(candidate_nodes)
    assert tree_block.content["selected_branch_id"] == selected_node["branch_id"]


def test_register_frontier_node_uses_executed_count_for_tree_budget(tmp_path: Path):
    database_path = str(tmp_path / "housing.db")
    db = Database(database_path)
    db.init()

    session_id, _ = db.create_session()
    approved_plan = {
        "user_memory_snapshot": {
            "target_area": "江東区",
            "budget_max": 120000,
            "station_walk_max": 7,
            "layout_preference": "1LDK",
            "must_conditions": [],
            "nice_to_have": [],
            "learned_preferences": {},
        }
    }
    job_id, _ = db.create_research_job(
        session_id=session_id,
        provider="openai",
        llm_config={},
        approved_plan=approved_plan,
    )
    manager = HousingResearchAgentManager(
        db=db,
        session_id=session_id,
        job_id=job_id,
        approved_plan=approved_plan,
        user_memory=approved_plan["user_memory_snapshot"],
        task_memory={},
        provider="openai",
        research_adapter=None,
        build_research_queries=lambda user_memory, seed_queries: seed_queries,
        collect_search_results=lambda **kwargs: ([], {}),
        fetch_detail_html=lambda url: None,
        collect_source_items=lambda **kwargs: [],
        tree_max_nodes=4,
    )
    state = ResearchExecutionState()

    for index in range(4):
        plan = SearchNodePlan(
            node_key=f"existing-{index}",
            label=f"existing-{index}",
            description="existing plan",
            queries=[f"query-{index}"],
            ranking_profile={},
            strategy_tags=["existing"],
            depth=1,
        )
        state.node_plans[plan.node_key] = plan
        state.node_artifacts[plan.node_key] = SearchNodeArtifacts(
            plan=plan,
            query_hash=manager._hash_queries(plan.queries, plan.ranking_profile),
            frontier_score=60.0 - index,
            status="completed" if index < 3 else "queued",
        )

    state.branch_summaries = [
        {"branch_id": "existing-0", "status": "completed"},
        {"branch_id": "existing-1", "status": "completed"},
        {"branch_id": "existing-2", "status": "completed"},
    ]
    state.frontier = ["existing-3"]

    child_plan = SearchNodePlan(
        node_key="child-1",
        label="child",
        description="child plan",
        queries=["query-child"],
        ranking_profile={},
        strategy_tags=["child"],
        depth=2,
        parent_key="existing-2",
    )

    manager._register_frontier_node(
        state,
        plan=child_plan,
        parent_summary={"branch_score": 72.0},
    )

    assert child_plan.node_key in state.node_plans
    assert child_plan.node_key in state.node_artifacts
    assert state.frontier == ["existing-3", child_plan.node_key]
