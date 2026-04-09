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
    assert isinstance(task_memory["last_integrity_reviews"], list)
    assert isinstance(task_memory["last_dropped_property_ids"], list)
    assert isinstance(task_memory["last_branch_result_summary"], dict)
    assert str(task_memory["last_final_report"]).startswith("# ")

    audit_stages = [event["stage"] for event in db.list_audit_events(session_id)]
    assert "search_normalize" in audit_stages
    assert "integrity_review" in audit_stages
    assert "ranking" in audit_stages
    assert "offline_evaluator" in audit_stages
    assert "final_report" in audit_stages

    research_state = orchestrator.get_research_state(session_id)
    assert research_state.response is not None
    block_titles = [block.title for block in research_state.response.blocks]
    assert "最終レポート" in block_titles
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


def test_initial_node_plans_prioritize_success_path_and_exclude_avoided_strategies(tmp_path: Path):
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
            "learned_preferences": {
                "strategy_memory": {
                    "preferred_strategy_tags": ["detail_first", "schema_first"],
                    "avoided_strategy_tags": ["relax_for_coverage"],
                    "last_successful_path": ["source_diversify", "detail_first"],
                }
            },
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
    )
    state = ResearchExecutionState(query="江東区 賃貸 1LDK", seed_queries=["江東区 賃貸 1LDK"])

    plans = manager._initial_node_plans(state)
    operators = [plan.strategy_tags[0] for plan in plans]

    assert operators[0] == "source_diversify"
    assert "relax_for_coverage" not in operators
    assert "detail_first" in operators


def test_initial_node_plans_use_retry_issues_to_change_seed_strategies(tmp_path: Path):
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
        },
        "retry_context": {
            "top_issues": ["詳細ページ補完率が低い", "比較に必要な項目の欠損が多い"]
        },
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
    )
    state = ResearchExecutionState(query="江東区 賃貸 1LDK", seed_queries=["江東区 賃貸 1LDK"])

    plans = manager._initial_node_plans(state)
    operators = [plan.strategy_tags[0] for plan in plans]

    assert operators[:2] == ["detail_first", "schema_first"]


def test_research_tools_reuse_query_and_detail_caches(tmp_path: Path):
    database_path = str(tmp_path / "housing.db")
    db = Database(database_path)
    db.init()

    session_id, _ = db.create_session()
    approved_plan = {"user_memory_snapshot": {"learned_preferences": {}}}
    job_id, _ = db.create_research_job(
        session_id=session_id,
        provider="openai",
        llm_config={},
        approved_plan=approved_plan,
    )
    search_calls: list[str] = []
    detail_calls: list[str] = []

    def collect_search_results(**kwargs):
        query = kwargs["query"]
        search_calls.append(query)
        return (
            [{"url": f"https://example.com/{query}", "source_name": "catalog"}],
            {
                "catalog_result_count": 1,
                "brave_result_count": 0,
                "brave_error": "",
            },
        )

    def fetch_detail_html(url: str) -> str | None:
        detail_calls.append(url)
        return f"<html>{url}</html>"

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
        collect_search_results=collect_search_results,
        fetch_detail_html=fetch_detail_html,
        collect_source_items=lambda **kwargs: [],
    )
    plan_a = SearchNodePlan(
        node_key="a",
        label="a",
        description="a",
        queries=["q1", "q2"],
        ranking_profile={},
        strategy_tags=["a"],
        depth=1,
    )
    plan_b = SearchNodePlan(
        node_key="b",
        label="b",
        description="b",
        queries=["q2"],
        ranking_profile={},
        strategy_tags=["b"],
        depth=1,
    )

    retrieve_a = manager._tool_retrieve(context=manager.context, branch=plan_a)
    retrieve_b = manager._tool_retrieve(context=manager.context, branch=plan_b)
    enrich_a = manager._tool_enrich(
        context=manager.context,
        branch=plan_a,
        raw_results=retrieve_a["raw_results"],
    )
    enrich_b = manager._tool_enrich(
        context=manager.context,
        branch=plan_b,
        raw_results=retrieve_b["raw_results"],
    )

    assert search_calls == ["q1", "q2"]
    assert retrieve_b["summary"]["cache_hit_count"] == 1
    assert detail_calls == ["https://example.com/q1", "https://example.com/q2"]
    assert enrich_b["summary"]["cache_hit_count"] == 1


def test_select_frontier_nodes_balances_best_branch_and_alternative(tmp_path: Path):
    database_path = str(tmp_path / "housing.db")
    db = Database(database_path)
    db.init()

    session_id, _ = db.create_session()
    approved_plan = {"user_memory_snapshot": {"learned_preferences": {}}}
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
        tree_batch_size=2,
    )
    state = ResearchExecutionState(best_node_key="best")
    best_plan = SearchNodePlan(
        node_key="best",
        label="best",
        description="best",
        queries=["best"],
        ranking_profile={},
        strategy_tags=["best"],
        depth=1,
    )
    deep_a = SearchNodePlan(
        node_key="deep-a",
        label="deep-a",
        description="deep-a",
        queries=["deep-a"],
        ranking_profile={},
        strategy_tags=["deep"],
        depth=2,
        parent_key="best",
    )
    deep_b = SearchNodePlan(
        node_key="deep-b",
        label="deep-b",
        description="deep-b",
        queries=["deep-b"],
        ranking_profile={},
        strategy_tags=["deep"],
        depth=2,
        parent_key="best",
    )
    alt = SearchNodePlan(
        node_key="alt",
        label="alt",
        description="alt",
        queries=["alt"],
        ranking_profile={},
        strategy_tags=["alt"],
        depth=1,
    )
    for plan, score in ((best_plan, 99.0), (deep_a, 95.0), (deep_b, 92.0), (alt, 80.0)):
        state.node_plans[plan.node_key] = plan
        state.node_artifacts[plan.node_key] = SearchNodeArtifacts(
            plan=plan,
            query_hash=manager._hash_queries(plan.queries, plan.ranking_profile),
            frontier_score=score,
            status="completed" if plan.node_key == "best" else "queued",
            summary={"status": "completed"} if plan.node_key == "best" else {},
        )
    state.frontier = ["deep-a", "deep-b", "alt"]

    selected = manager._select_frontier_nodes(state)

    assert selected == ["deep-a", "alt"]


def test_best_node_readiness_uses_cached_artifact_value(tmp_path: Path, monkeypatch):
    database_path = str(tmp_path / "housing.db")
    db = Database(database_path)
    db.init()

    session_id, _ = db.create_session()
    approved_plan = {"user_memory_snapshot": {"learned_preferences": {}}}
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
    )
    state = ResearchExecutionState(best_node_key="node-1")
    plan = SearchNodePlan(
        node_key="node-1",
        label="cached",
        description="cached plan",
        queries=["query-1"],
        ranking_profile={},
        strategy_tags=["cached"],
        depth=1,
    )
    state.node_artifacts[plan.node_key] = SearchNodeArtifacts(
        plan=plan,
        query_hash=manager._hash_queries(plan.queries, plan.ranking_profile),
        frontier_score=88.0,
        readiness="high",
        summary={"branch_id": "node-1", "status": "completed"},
    )

    monkeypatch.setattr(
        "app.research.agent_manager_tree_mixin.evaluate_final_result",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("should not recompute readiness")),
    )
    monkeypatch.setattr(
        "app.research.agent_manager_tree_mixin.select_best_branch",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not rescan branches")),
    )

    assert manager._best_node_readiness(state) == "high"


def test_tree_search_waits_for_min_nodes_before_stable_stop(tmp_path: Path, monkeypatch):
    database_path = str(tmp_path / "housing.db")
    db = Database(database_path)
    db.init()

    session_id, _ = db.create_session()
    approved_plan = {"user_memory_snapshot": {"learned_preferences": {}}}
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
        tree_max_nodes=5,
        tree_batch_size=1,
        tree_stability_patience=1,
        tree_min_nodes_before_stable_stop=3,
        tree_min_best_score_gap=0,
    )
    plans = [
        SearchNodePlan(
            node_key=f"node-{index}",
            label=f"node-{index}",
            description="node",
            queries=[f"query-{index}"],
            ranking_profile={},
            strategy_tags=[f"op-{index}"],
            depth=1,
        )
        for index in range(3)
    ]
    monkeypatch.setattr(manager, "_initial_node_plans", lambda state: plans)

    def fake_execute_candidate(state, *, plan):
        order = len(state.branch_summaries)
        artifacts = state.node_artifacts[plan.node_key]
        summary = {
            "branch_id": plan.node_key,
            "node_key": plan.node_key,
            "label": plan.label,
            "status": "completed",
            "depth": plan.depth,
            "detail_coverage": 0.8,
            "avg_top3_score": 90.0,
            "normalized_count": 3,
            "branch_score": 95.0 - order,
            "frontier_score": 95.0 - order,
            "top_issue_class": "healthy",
            "prune_reasons": [],
            "parent_key": plan.parent_key or "",
            "strategy_tags": plan.strategy_tags,
        }
        artifacts.summary = summary
        artifacts.status = "completed"
        artifacts.readiness = "high"
        state.branch_summaries.append(summary)
        return summary

    monkeypatch.setattr(manager, "_execute_candidate", fake_execute_candidate)

    state = ResearchExecutionState(query="q", seed_queries=["q"])

    assert manager._handle_tree_search(state) is None
    assert len(state.branch_summaries) == 3
    assert state.termination_reason == "stable_high_readiness"


def test_tree_search_expands_recovery_nodes_after_initial_failures(tmp_path: Path):
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

    def fail_collect_search_results(**kwargs):
        raise RuntimeError("search backend unavailable")

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
        collect_search_results=fail_collect_search_results,
        fetch_detail_html=lambda url: None,
        collect_source_items=lambda **kwargs: [],
        tree_max_nodes=5,
    )
    state = ResearchExecutionState(
        query="江東区 賃貸 1LDK",
        seed_queries=["江東区 賃貸 1LDK"],
    )

    assert manager._handle_tree_search(state) is None

    assert len(state.branch_summaries) == 5
    assert any(int(summary.get("depth") or 0) >= 2 for summary in state.branch_summaries)
    assert any(str(summary.get("parent_key") or "").strip() for summary in state.branch_summaries)


def test_tree_search_attaches_branch_result_summary_before_final_selection(tmp_path: Path, monkeypatch):
    database_path = str(tmp_path / "housing.db")
    db = Database(database_path)
    db.init()

    session_id, _ = db.create_session()
    approved_plan = {"user_memory_snapshot": {"learned_preferences": {}}}
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
        tree_batch_size=1,
    )
    root_plan = SearchNodePlan(
        node_key="node-1",
        label="root-branch",
        description="root",
        queries=["query-1"],
        ranking_profile={},
        strategy_tags=["root"],
        depth=1,
    )
    child_plan = SearchNodePlan(
        node_key="node-2",
        label="child-branch",
        description="child",
        queries=["query-2"],
        ranking_profile={},
        strategy_tags=["detail_first"],
        depth=2,
        parent_key="node-1",
    )

    monkeypatch.setattr(manager, "_initial_node_plans", lambda state: [root_plan])
    monkeypatch.setattr(manager, "_refresh_best_node", lambda state, candidate_key: None)

    def fake_next_candidates(state, *, plan, summary):
        if plan.node_key == "node-1":
            return [child_plan]
        return []

    monkeypatch.setattr(manager, "_next_candidates_after_summary", fake_next_candidates)
    monkeypatch.setattr(
        "app.research.agent_manager_tree_mixin.run_result_summarizer",
        lambda **kwargs: {
            "物件候補リスト": [
                {
                    "property_id_norm": "p1",
                    "building_name": "東雲ベイテラス",
                    "address": "東京都江東区東雲1-4-8",
                    "rent": 118000,
                    "layout": "1LDK",
                    "station_walk_min": 6,
                    "area_m2": 42.1,
                    "detail_url": "https://example.com/p1",
                    "score": 90.0,
                    "reason": "条件一致度が高い",
                    "evidence": ["家賃と駅徒歩が条件内"],
                    "matched_queries": ["query-1", "query-2"],
                    "source_nodes": ["root-branch", "child-branch"],
                }
            ],
            "却下理由": [],
            "共通リスク": ["管理費の内訳が未確認"],
            "未解決の調査項目": ["東雲ベイテラス の管理費・初期費用内訳"],
            "summary": {
                "branch_node_count": 2,
                "unique_candidate_count": 1,
                "shortlisted_candidate_count": 1,
                "rejection_count": 0,
                "detail_page_hit_count": 1,
            },
        },
    )

    def fake_select_best_branch(branch_summaries):
        assert len(branch_summaries) == 2
        assert all("branch_result_summary" in summary for summary in branch_summaries)
        return branch_summaries[-1]

    monkeypatch.setattr(
        "app.research.agent_manager_tree_mixin.select_best_branch",
        fake_select_best_branch,
    )

    def fake_execute_candidate(state, *, plan):
        artifacts = state.node_artifacts[plan.node_key]
        artifacts.retrieve = {
            "raw_results": [{"title": "東雲ベイテラス", "url": "https://example.com/p1"}],
            "summary": {"detail_hit_count": 1},
        }
        artifacts.enrich = {
            "detail_html_map": {"https://example.com/p1": "<p>東雲ベイテラス</p>"},
            "summary": {"detail_hit_count": 1},
        }
        artifacts.normalize = {
            "normalized_properties": [
                {
                    "property_id_norm": "p1",
                    "building_name": "東雲ベイテラス",
                    "address": "東京都江東区東雲1-4-8",
                    "detail_url": "https://example.com/p1",
                    "rent": 118000,
                    "layout": "1LDK",
                    "station_walk_min": 6,
                    "area_m2": 42.1,
                }
            ],
            "duplicate_groups": [],
            "summary": {"normalized_count": 1, "detail_hit_count": 1},
        }
        artifacts.rank = {
            "ranked_properties": [
                {
                    "property_id_norm": "p1",
                    "score": 90.0 if plan.node_key == "node-2" else 82.0,
                    "why_selected": "条件一致度が高い",
                    "why_not_selected": "",
                }
            ]
        }
        summary = {
            "branch_id": plan.node_key,
            "node_key": plan.node_key,
            "label": plan.label,
            "status": "completed",
            "depth": plan.depth,
            "detail_coverage": 1.0,
            "avg_top3_score": 90.0 if plan.node_key == "node-2" else 82.0,
            "normalized_count": 1,
            "branch_score": 90.0 if plan.node_key == "node-2" else 82.0,
            "frontier_score": 90.0 if plan.node_key == "node-2" else 82.0,
            "top_issue_class": "healthy",
            "prune_reasons": [],
            "parent_key": plan.parent_key or "",
            "strategy_tags": plan.strategy_tags,
        }
        artifacts.summary = summary
        artifacts.status = "completed"
        artifacts.readiness = "medium"
        state.branch_summaries.append(summary)
        return summary

    monkeypatch.setattr(manager, "_execute_candidate", fake_execute_candidate)

    state = ResearchExecutionState(query="q", seed_queries=["q"])

    assert manager._handle_tree_search(state) is None
    assert state.selected_branch_summary["branch_id"] == "node-2"
    assert state.selected_branch_summary["branch_result_summary"]["summary"]["branch_node_count"] == 2
    selected_artifacts = manager._selected_artifacts(state)
    assert selected_artifacts is not None
    assert selected_artifacts.normalize["branch_result_summary"]["共通リスク"] == ["管理費の内訳が未確認"]
