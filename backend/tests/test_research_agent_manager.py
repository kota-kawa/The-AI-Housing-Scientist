from pathlib import Path
import threading

from app.config import Settings
from app.db import Database
from app.llm.base import LLMAdapter
from app.orchestrator import HousingOrchestrator
from app.research.agent_manager import (
    HousingResearchAgentManager as _HousingResearchAgentManager,
)
from app.research.agent_manager import (
    ResearchExecutionState,
    SearchNodeArtifacts,
    SearchNodePlan,
)
from app.stages.result_summarizer import PROPERTY_CANDIDATES_KEY


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
        gemini_model="gemini-2.5-flash",
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
                "listing_type": "賃貸",
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
                "listing_type": "",
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


class FakeLinkSelectionAdapter(LLMAdapter):
    def __init__(self, selected_indexes: list[int]):
        self.selected_indexes = selected_indexes
        self.calls = 0

    def generate_text(self, *, system: str, user: str, temperature: float = 0.2) -> str:
        raise AssertionError("generate_text should not be called in link selection tests")

    def generate_structured(
        self,
        *,
        system: str,
        user: str,
        schema: dict,
        temperature: float = 0.2,
    ) -> dict:
        self.calls += 1
        return {"selected_indexes": list(self.selected_indexes)}

    def list_models(self) -> list[str]:
        return ["fake-link-selection-model"]


def passthrough_branch_family_queries(
    user_memory: dict[str, object],
    seed_queries: list[str],
    **_: object,
) -> list[str]:
    return list(seed_queries)


def make_manager(**kwargs):
    return _HousingResearchAgentManager(
        build_branch_family_queries=passthrough_branch_family_queries,
        **kwargs,
    )


def test_research_job_records_branch_tree_and_evaluations(tmp_path: Path):
    database_path = str(tmp_path / "housing.db")
    db = Database(database_path)
    db.init()
    orchestrator = HousingOrchestrator(settings=build_settings(database_path), db=db)
    planner_adapter = FakePlannerRouteAdapter()
    orchestrator._get_adapter_for_route = lambda **kwargs: (
        planner_adapter if kwargs.get("route_key") == "planner" else None
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
    assert search_roots[0]["intent"] == "draft"
    assert search_roots[0]["is_failed"] is False
    assert search_roots[0]["debug_depth"] == 0

    candidate_nodes = [node for node in nodes if node["node_type"] == "search_candidate"]
    assert len(candidate_nodes) >= 2
    assert all(node["branch_id"] for node in candidate_nodes)
    assert all(node["parent_node_id"] is not None for node in candidate_nodes)
    assert all(
        node["intent"] in {"draft", "refine", "pivot", "recovery"} for node in candidate_nodes
    )
    assert all(isinstance(node["is_failed"], bool) for node in candidate_nodes)
    assert all(isinstance(node["debug_depth"], int) for node in candidate_nodes)

    branch_selections = [
        node for node in nodes if node["node_type"] == "search_selection" and node["selected"]
    ]
    assert len(branch_selections) == 1
    selected_node = branch_selections[0]
    assert selected_node["parent_node_id"] is not None
    assert selected_node["intent"] in {"draft", "refine", "pivot", "recovery"}
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
    tree_block = next(
        (block for block in research_state.response.blocks if block.type == "tree"), None
    )
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
    manager = make_manager(
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


def test_branch_result_nodes_keep_empty_integrity_result_instead_of_normalize_fallback(
    tmp_path: Path,
):
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
    manager = make_manager(
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
    state = ResearchExecutionState()
    plan = SearchNodePlan(
        node_key="strict-empty",
        label="strict",
        description="strict plan",
        queries=["町田 賃貸 ワンルーム"],
        ranking_profile={},
        strategy_tags=["strict_primary"],
        depth=1,
    )
    artifact = SearchNodeArtifacts(
        plan=plan,
        query_hash=manager._hash_queries(plan.queries, plan.ranking_profile),
        frontier_score=8.0,
        status="completed",
    )
    artifact.summary = {"status": "completed", "branch_id": plan.node_key}
    artifact.normalize = {
        "normalized_properties": [{"property_id_norm": "dropped", "building_name": "別エリア"}],
        "summary": {"normalized_count": 1},
    }
    artifact.integrity = {
        "normalized_properties": [],
        "dropped_properties": [{"property_id_norm": "dropped", "building_name": "別エリア"}],
        "integrity_reviews": [],
        "summary": {"integrity_dropped_count": 1},
    }
    state.node_artifacts[plan.node_key] = artifact

    nodes = manager._branch_result_nodes(state, node_key=plan.node_key)

    assert nodes[0]["normalized_properties"] == []
    assert nodes[0]["dropped_properties"][0]["property_id_norm"] == "dropped"


def test_display_candidate_pool_prefers_selected_path_aggregate_and_merges_snapshots(
    tmp_path: Path,
):
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
    manager = make_manager(
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
    state = ResearchExecutionState()

    root_plan = SearchNodePlan(
        node_key="root",
        label="root",
        description="root plan",
        queries=["江東区 賃貸"],
        ranking_profile={},
        strategy_tags=["broad"],
        depth=1,
    )
    child_plan = SearchNodePlan(
        node_key="child",
        label="child",
        description="child plan",
        queries=["江東区 1LDK 駅近"],
        ranking_profile={},
        strategy_tags=["refine"],
        depth=2,
        parent_key="root",
    )

    root_artifact = SearchNodeArtifacts(
        plan=root_plan,
        query_hash=manager._hash_queries(root_plan.queries, root_plan.ranking_profile),
        frontier_score=72.0,
        status="completed",
    )
    root_artifact.summary = {"status": "completed", "branch_id": "root"}
    root_artifact.integrity = {
        "normalized_properties": [
            {
                "property_id_norm": "p-root",
                "building_name": "親ノード物件",
                "detail_url": "https://example.com/p-root",
                "image_url": "https://img.example.com/p-root.jpg",
                "address": "江東区豊洲1-1-1",
                "rent": 119000,
                "layout": "1LDK",
                "station_walk_min": 6,
                "area_m2": 35.5,
                "features": ["角部屋"],
                "notes": "親ノードで発見した候補",
            },
            {
                "property_id_norm": "p-merge",
                "building_name": "統合対象物件",
                "detail_url": "https://example.com/p-merge",
                "address": "江東区東雲2-2-2",
                "layout": "1LDK",
                "features": ["南向き"],
                "notes": "親ノードの補足メモ",
            },
        ]
    }
    root_artifact.rank = {
        "ranked_properties": [
            {
                "property_id_norm": "p-root",
                "score": 84.0,
                "why_selected": "親ノードで条件一致が高かった",
                "why_not_selected": "",
            },
            {
                "property_id_norm": "p-merge",
                "score": 71.0,
                "why_selected": "母集団に残した候補",
                "why_not_selected": "",
            },
        ]
    }

    child_artifact = SearchNodeArtifacts(
        plan=child_plan,
        query_hash=manager._hash_queries(child_plan.queries, child_plan.ranking_profile),
        frontier_score=80.0,
        status="completed",
    )
    child_artifact.summary = {"status": "completed", "branch_id": "child"}
    child_artifact.integrity = {
        "normalized_properties": [
            {
                "property_id_norm": "p-merge",
                "building_name": "統合対象物件",
                "detail_url": "https://example.com/p-merge",
                "image_url": "https://img.example.com/p-merge.jpg",
                "address": "江東区東雲2-2-2",
                "rent": 118000,
                "layout": "1LDK",
                "station_walk_min": 5,
                "area_m2": 36.2,
                "features": ["追い焚き"],
            },
            {
                "property_id_norm": "p-child",
                "building_name": "子ノード物件",
                "detail_url": "https://example.com/p-child",
                "image_url": "https://img.example.com/p-child.jpg",
                "address": "江東区有明3-3-3",
                "rent": 121000,
                "layout": "1LDK",
                "station_walk_min": 4,
                "area_m2": 34.0,
                "features": ["宅配ボックス"],
            },
        ]
    }
    child_artifact.rank = {
        "ranked_properties": [
            {
                "property_id_norm": "p-merge",
                "score": 88.0,
                "why_selected": "子ノードで詳細が揃った",
                "why_not_selected": "管理費は要確認",
            },
            {
                "property_id_norm": "p-child",
                "score": 76.0,
                "why_selected": "駅近候補として残した",
                "why_not_selected": "",
            },
        ]
    }

    state.node_artifacts = {
        "root": root_artifact,
        "child": child_artifact,
    }
    state.selected_branch_summary = {"branch_id": "child"}

    display_ranked, display_normalized = manager._build_display_candidate_pool(
        state=state,
        branch_id="child",
        branch_result_summary={
            PROPERTY_CANDIDATES_KEY: [
                {
                    "property_id_norm": "p-root",
                    "detail_url": "https://example.com/p-root",
                    "building_name": "親ノード物件",
                    "reason": "親ノードだけで残っていた候補",
                },
                {
                    "property_id_norm": "p-merge",
                    "detail_url": "https://example.com/p-merge",
                    "building_name": "統合対象物件",
                    "reason": "パス全体で最も情報が揃った候補",
                },
                {
                    "property_id_norm": "missing",
                    "detail_url": "https://example.com/missing",
                    "building_name": "未解決候補",
                    "reason": "canonical snapshot が無い候補",
                },
            ]
        },
    )

    assert [item["property_id_norm"] for item in display_ranked] == [
        "p-root",
        "p-merge",
        "p-child",
    ]
    assert all(item["property_id_norm"] != "missing" for item in display_ranked)

    normalized_by_id = {item["property_id_norm"]: item for item in display_normalized}
    assert normalized_by_id["p-root"]["building_name"] == "親ノード物件"
    assert normalized_by_id["p-merge"]["rent"] == 118000
    assert normalized_by_id["p-merge"]["station_walk_min"] == 5
    assert normalized_by_id["p-merge"]["notes"] == "親ノードの補足メモ"
    assert normalized_by_id["p-merge"]["features"] == ["追い焚き", "南向き"]
    assert display_ranked[1]["why_selected"] == "子ノードで詳細が揃った"
    assert display_ranked[1]["why_not_selected"] == "管理費は要確認"


def test_ensure_minimum_display_candidates_backfills_from_raw_results(tmp_path: Path):
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
    manager = make_manager(
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
    state = ResearchExecutionState()
    plan = SearchNodePlan(
        node_key="strict-primary",
        label="strict",
        description="strict plan",
        queries=["江東区 賃貸 1LDK"],
        ranking_profile={},
        strategy_tags=["strict_primary"],
        depth=1,
        branch_family="strict_primary",
        area_scope="strict",
        constraint_mode="primary",
    )
    artifact = SearchNodeArtifacts(
        plan=plan,
        query_hash=manager._hash_queries(plan.queries, plan.ranking_profile),
        frontier_score=22.0,
        status="completed",
    )
    artifact.retrieve = {
        "raw_results": [
            {
                "title": "東雲ベイテラス",
                "url": "https://example.com/property/p1",
                "description": "東京都江東区東雲1-4-8 / 家賃118,000円 / 1LDK / 徒歩6分",
                "source_name": "catalog",
            },
            {
                "title": "豊洲リバーサイド",
                "url": "https://example.com/property/p2",
                "description": "東京都江東区豊洲2-1-9 / 家賃121,000円 / 1LDK / 徒歩7分",
                "source_name": "catalog",
            },
            {
                "title": "有明フロント",
                "url": "https://example.com/property/p3",
                "description": "東京都江東区有明1-2-3 / 家賃126,000円 / 1LDK / 徒歩9分",
                "source_name": "catalog",
            },
        ]
    }
    artifact.summary = {
        "branch_id": plan.node_key,
        "status": "completed",
        "label": plan.label,
        "branch_family": plan.branch_family,
        "branch_score": 22.0,
        "depth": 1,
    }
    state.node_artifacts[plan.node_key] = artifact
    state.branch_summaries = [artifact.summary]

    ranked, normalized, source = manager._ensure_minimum_display_candidates(
        state,
        display_ranked_properties=[],
        display_normalized_properties=[],
        alternative_display_groups=[],
    )

    assert source == "raw_result_fill"
    assert len(ranked) == 3
    assert len(normalized) == 3
    assert [item["building_name"] for item in ranked] == [
        "東雲ベイテラス",
        "豊洲リバーサイド",
        "有明フロント",
    ]
    assert all("参考候補" in item["why_selected"] for item in ranked)
    assert all(item["property_id_norm"] for item in normalized)


def test_ensure_minimum_display_candidates_skips_listing_pages_in_raw_results(tmp_path: Path):
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
    manager = make_manager(
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
    state = ResearchExecutionState()
    plan = SearchNodePlan(
        node_key="strict-primary",
        label="strict",
        description="strict plan",
        queries=["江東区 賃貸 1LDK"],
        ranking_profile={},
        strategy_tags=["strict_primary"],
        depth=1,
        branch_family="strict_primary",
        area_scope="strict",
        constraint_mode="primary",
    )
    artifact = SearchNodeArtifacts(
        plan=plan,
        query_hash=manager._hash_queries(plan.queries, plan.ranking_profile),
        frontier_score=22.0,
        status="completed",
    )
    artifact.retrieve = {
        "raw_results": [
            {
                "title": "江東区の賃貸物件一覧",
                "url": "https://example.com/search/koto?area=koto&page=1",
                "description": "該当物件 24件",
                "source_name": "catalog",
            },
            {
                "title": "東雲ベイテラス",
                "url": "https://example.com/property/p1",
                "description": "東京都江東区東雲1-4-8 / 家賃118,000円 / 1LDK / 徒歩6分",
                "source_name": "catalog",
            },
        ]
    }
    artifact.summary = {
        "branch_id": plan.node_key,
        "status": "completed",
        "label": plan.label,
        "branch_family": plan.branch_family,
        "branch_score": 22.0,
        "depth": 1,
    }
    state.node_artifacts[plan.node_key] = artifact
    state.branch_summaries = [artifact.summary]

    ranked, normalized, source = manager._ensure_minimum_display_candidates(
        state,
        display_ranked_properties=[],
        display_normalized_properties=[],
        alternative_display_groups=[],
    )

    assert source == "raw_result_fill"
    assert len(ranked) == 1
    assert len(normalized) == 1
    assert ranked[0]["building_name"] == "東雲ベイテラス"


def test_initial_node_plans_create_fixed_branch_families(tmp_path: Path):
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
                    "avoided_strategy_tags": ["detail_first"],
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
    manager = make_manager(
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
    assert [
        (plan.branch_family, plan.area_scope, plan.constraint_mode)
        for plan in plans
    ] == [
        ("strict_primary", "strict", "primary"),
        ("strict_relaxed", "strict", "relaxed"),
        ("nearby_primary", "nearby", "primary"),
        ("nearby_relaxed", "nearby", "relaxed"),
    ]
    assert [plan.strategy_tags for plan in plans] == [
        ["strict_primary"],
        ["strict_relaxed"],
        ["nearby_primary"],
        ["nearby_relaxed"],
    ]


def test_initial_node_plans_use_family_query_builder_and_nearby_hints(tmp_path: Path):
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
        "retry_context": {"top_issues": ["詳細ページ補完率が低い", "比較に必要な項目の欠損が多い"]},
    }
    job_id, _ = db.create_research_job(
        session_id=session_id,
        provider="openai",
        llm_config={},
        approved_plan=approved_plan,
    )
    manager = _HousingResearchAgentManager(
        db=db,
        session_id=session_id,
        job_id=job_id,
        approved_plan=approved_plan,
        user_memory=approved_plan["user_memory_snapshot"],
        task_memory={},
        provider="openai",
        research_adapter=None,
        build_research_queries=lambda user_memory, seed_queries: seed_queries,
        build_branch_family_queries=lambda user_memory, seed_queries, **kwargs: {
            ("strict", "primary"): ["江東区 賃貸 1LDK strict"],
            ("strict", "relaxed"): ["江東区 賃貸 1LDK relaxed"],
            ("nearby", "primary"): ["豊洲 賃貸 1LDK", "東雲 賃貸 1LDK"],
            ("nearby", "relaxed"): ["有明 賃貸 1LDK 緩和"],
        }[(kwargs["area_scope"], kwargs["constraint_mode"])],
        collect_search_results=lambda **kwargs: ([], {}),
        fetch_detail_html=lambda url: None,
        collect_source_items=lambda **kwargs: [],
    )
    state = ResearchExecutionState(query="江東区 賃貸 1LDK", seed_queries=["江東区 賃貸 1LDK"])

    plans = manager._initial_node_plans(state)
    plans_by_family = {plan.branch_family: plan for plan in plans}
    assert plans_by_family["strict_primary"].queries == ["江東区 賃貸 1LDK strict"]
    assert plans_by_family["strict_relaxed"].queries == ["江東区 賃貸 1LDK relaxed"]
    assert plans_by_family["nearby_primary"].queries == [
        "豊洲 賃貸 1LDK",
        "東雲 賃貸 1LDK",
    ]
    assert plans_by_family["nearby_primary"].nearby_hints == ["豊洲", "東雲"]
    assert plans_by_family["nearby_relaxed"].constraint_mode == "relaxed"


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

    manager = make_manager(
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
    manager._tool_enrich(
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


def test_expand_branch_batch_runs_candidates_in_parallel(tmp_path: Path, monkeypatch):
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
    manager = make_manager(
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
        tree_batch_size=3,
    )
    state = ResearchExecutionState()
    plans = [
        SearchNodePlan(
            node_key=f"node-{index}",
            label=f"node-{index}",
            description="parallel candidate",
            queries=[f"query-{index}"],
            ranking_profile={},
            strategy_tags=[f"tag-{index}"],
            depth=1,
        )
        for index in range(3)
    ]
    for plan in plans:
        state.node_artifacts[plan.node_key] = SearchNodeArtifacts(
            plan=plan,
            query_hash=manager._hash_queries(plan.queries, plan.ranking_profile),
            frontier_score=80.0,
        )

    started: list[str] = []
    release = threading.Event()
    lock = threading.Lock()
    active_workers = 0
    max_active_workers = 0

    def fake_execute_candidate(state, *, plan):
        nonlocal active_workers, max_active_workers
        with lock:
            started.append(plan.node_key)
            active_workers += 1
            max_active_workers = max(max_active_workers, active_workers)
            if len(started) == len(plans):
                release.set()
        try:
            assert release.wait(timeout=1.0)
            return {
                "branch_id": plan.node_key,
                "node_key": plan.node_key,
                "label": plan.label,
                "status": "completed",
                "depth": plan.depth,
                "detail_coverage": 0.8,
                "avg_top3_score": 88.0,
                "normalized_count": 2,
                "branch_score": 88.0,
                "frontier_score": 88.0,
                "top_issue_class": "healthy",
                "prune_reasons": [],
                "parent_key": plan.parent_key or "",
                "strategy_tags": plan.strategy_tags,
            }
        finally:
            with lock:
                active_workers -= 1

    monkeypatch.setattr(manager, "_execute_candidate", fake_execute_candidate)

    summaries = manager._expand_branch_batch(state, plans=plans)

    assert [summary["branch_id"] for summary in summaries] == [plan.node_key for plan in plans]
    assert [summary["branch_id"] for summary in state.branch_summaries] == [
        plan.node_key for plan in plans
    ]
    assert max_active_workers >= 2


def test_parallel_research_tools_share_singleflight_cache(tmp_path: Path):
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
    search_started = threading.Event()
    release_search = threading.Event()
    detail_started = threading.Event()
    release_detail = threading.Event()

    def collect_search_results(**kwargs):
        query = kwargs["query"]
        search_calls.append(query)
        search_started.set()
        assert release_search.wait(timeout=1.0)
        return (
            [{"url": "https://example.com/shared", "source_name": "catalog"}],
            {
                "catalog_result_count": 1,
                "brave_result_count": 0,
                "brave_error": "",
            },
        )

    def fetch_detail_html(url: str) -> str | None:
        detail_calls.append(url)
        detail_started.set()
        assert release_detail.wait(timeout=1.0)
        return f"<html>{url}</html>"

    manager = make_manager(
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
        tree_batch_size=2,
    )
    plan_a = SearchNodePlan(
        node_key="parallel-a",
        label="parallel-a",
        description="parallel-a",
        queries=["shared-query"],
        ranking_profile={},
        strategy_tags=["parallel"],
        depth=1,
    )
    plan_b = SearchNodePlan(
        node_key="parallel-b",
        label="parallel-b",
        description="parallel-b",
        queries=["shared-query"],
        ranking_profile={},
        strategy_tags=["parallel"],
        depth=1,
    )

    retrieve_results: dict[str, dict] = {}

    def run_retrieve(slot: str, plan: SearchNodePlan) -> None:
        retrieve_results[slot] = manager._tool_retrieve(context=manager.context, branch=plan)

    retrieve_a = threading.Thread(target=run_retrieve, args=("a", plan_a))
    retrieve_b = threading.Thread(target=run_retrieve, args=("b", plan_b))
    retrieve_a.start()
    assert search_started.wait(timeout=1.0)
    retrieve_b.start()
    release_search.set()
    retrieve_a.join(timeout=1.0)
    retrieve_b.join(timeout=1.0)

    assert search_calls == ["shared-query"]
    assert retrieve_results["a"]["raw_results"] == retrieve_results["b"]["raw_results"]
    assert (
        retrieve_results["a"]["summary"]["cache_hit_count"]
        + retrieve_results["b"]["summary"]["cache_hit_count"]
    ) == 1

    enrich_results: dict[str, dict] = {}

    def run_enrich(slot: str, plan: SearchNodePlan, raw_results: list[dict]) -> None:
        enrich_results[slot] = manager._tool_enrich(
            context=manager.context,
            branch=plan,
            raw_results=raw_results,
        )

    enrich_a = threading.Thread(
        target=run_enrich,
        args=("a", plan_a, retrieve_results["a"]["raw_results"]),
    )
    enrich_b = threading.Thread(
        target=run_enrich,
        args=("b", plan_b, retrieve_results["b"]["raw_results"]),
    )
    enrich_a.start()
    assert detail_started.wait(timeout=1.0)
    enrich_b.start()
    release_detail.set()
    enrich_a.join(timeout=1.0)
    enrich_b.join(timeout=1.0)

    assert detail_calls == ["https://example.com/shared"]
    assert enrich_results["a"]["detail_html_map"] == enrich_results["b"]["detail_html_map"]
    assert (
        enrich_results["a"]["summary"]["cache_hit_count"]
        + enrich_results["b"]["summary"]["cache_hit_count"]
    ) == 1


def test_tool_enrich_expands_listing_page_to_same_domain_detail_pages(tmp_path: Path):
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
    html_by_url = {
        "https://example.com/search/koto": """
        <html>
          <body>
            <h1>江東区の賃貸物件一覧</h1>
            <div class="card">
              <a href="/property/p1">詳細を見る</a>
              <p>東雲ベイテラス 家賃11.8万円 1LDK 徒歩6分</p>
            </div>
            <div class="card">
              <a href="/property/p2">物件詳細</a>
              <p>豊洲リバーサイド 家賃12.1万円 1LDK 徒歩7分</p>
            </div>
            <div class="card">
              <a href="https://other.example.com/property/p3">外部サイト</a>
              <p>他社の物件</p>
            </div>
          </body>
        </html>
        """,
        "https://example.com/property/p1": """
        <article data-kind="property-detail">
          <h1 data-field="building_name">東雲ベイテラス</h1>
          <p data-field="address">東京都江東区東雲1-4-8</p>
          <p data-field="layout">1LDK</p>
          <p data-field="rent">118000</p>
        </article>
        """,
        "https://example.com/property/p2": """
        <article data-kind="property-detail">
          <h1 data-field="building_name">豊洲リバーサイド</h1>
          <p data-field="address">東京都江東区豊洲2-1-9</p>
          <p data-field="layout">1LDK</p>
          <p data-field="rent">121000</p>
        </article>
        """,
    }
    manager = make_manager(
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
        fetch_detail_html=lambda url: html_by_url.get(url),
        collect_source_items=lambda **kwargs: [],
    )
    plan = SearchNodePlan(
        node_key="strict-primary",
        label="strict",
        description="strict plan",
        queries=["江東区 賃貸 1LDK"],
        ranking_profile={},
        strategy_tags=["strict_primary"],
        depth=1,
    )

    result = manager._tool_enrich(
        context=manager.context,
        branch=plan,
        raw_results=[
            {
                "title": "江東区の賃貸物件一覧",
                "url": "https://example.com/search/koto",
                "description": "江東区の1LDKをまとめた一覧",
                "source_name": "brave",
                "matched_queries": ["江東区 賃貸 1LDK"],
            }
        ],
    )

    expanded_urls = sorted(item["url"] for item in result["expanded_raw_results"])
    assert expanded_urls == [
        "https://example.com/property/p1",
        "https://example.com/property/p2",
    ]
    assert sorted(result["detail_html_map"].keys()) == sorted(expanded_urls)
    assert result["summary"]["listing_page_expand_count"] == 1
    assert result["summary"]["discovered_detail_count"] == 2
    assert result["summary"]["expanded_result_count"] == 2


def test_tool_enrich_uses_llm_for_ambiguous_listing_links(tmp_path: Path):
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
    adapter = FakeLinkSelectionAdapter([1])
    html_by_url = {
        "https://example.com/listing": """
        <html>
          <body>
            <h1>候補一覧</h1>
            <div class="slot">
              <a href="/estate/alpha">見る</a>
              <p>候補A 1LDK</p>
            </div>
            <div class="slot">
              <a href="/estate/beta">見る</a>
              <p>候補B 家賃11.9万円 1LDK 徒歩6分</p>
            </div>
          </body>
        </html>
        """,
        "https://example.com/estate/beta": """
        <article data-kind="property-detail">
          <h1 data-field="building_name">候補Bレジデンス</h1>
          <p data-field="address">東京都江東区木場1-2-3</p>
          <p data-field="layout">1LDK</p>
          <p data-field="rent">119000</p>
        </article>
        """,
    }
    manager = make_manager(
        db=db,
        session_id=session_id,
        job_id=job_id,
        approved_plan=approved_plan,
        user_memory=approved_plan["user_memory_snapshot"],
        task_memory={},
        provider="openai",
        research_adapter=adapter,
        build_research_queries=lambda user_memory, seed_queries: seed_queries,
        collect_search_results=lambda **kwargs: ([], {}),
        fetch_detail_html=lambda url: html_by_url.get(url),
        collect_source_items=lambda **kwargs: [],
    )
    plan = SearchNodePlan(
        node_key="strict-primary",
        label="strict",
        description="strict plan",
        queries=["江東区 賃貸 1LDK"],
        ranking_profile={},
        strategy_tags=["strict_primary"],
        depth=1,
    )

    result = manager._tool_enrich(
        context=manager.context,
        branch=plan,
        raw_results=[
            {
                "title": "一覧ページ",
                "url": "https://example.com/listing",
                "description": "曖昧なリンクが並ぶ一覧",
                "source_name": "brave",
                "matched_queries": ["江東区 賃貸 1LDK"],
            }
        ],
    )

    assert adapter.calls == 1
    assert [item["url"] for item in result["expanded_raw_results"]] == [
        "https://example.com/estate/beta"
    ]


def test_live_progress_summary_shows_current_action_and_recent_history(tmp_path: Path):
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
    manager = make_manager(
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

    manager._update_live_progress(
        stage_name="tree_search",
        progress_percent=36,
        current_action="検索結果を収集中",
        detail="クエリ 1/2: 江東区 賃貸 1LDK",
    )
    manager._update_live_progress(
        stage_name="tree_search",
        progress_percent=48,
        current_action="物件詳細ページを取得中",
        detail="2/5 件目",
        url="https://example.com/property/123",
    )

    job = db.get_research_job(job_id)
    assert job is not None
    assert "現在: 物件詳細ページを取得中" in job["latest_summary"]
    assert "対象: https://example.com/property/123" in job["latest_summary"]
    assert "直近:" in job["latest_summary"]
    assert "検索結果を収集中" in job["latest_summary"]


def test_tool_enrich_updates_job_with_live_detail_url(tmp_path: Path):
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
    manager = make_manager(
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
        fetch_detail_html=lambda url: """
        <article data-kind="property-detail">
          <h1 data-field="building_name">テスト物件</h1>
          <p data-field="address">東京都江東区豊洲1-2-3</p>
          <p data-field="layout">1LDK</p>
          <p data-field="rent">118000</p>
        </article>
        """,
        collect_source_items=lambda **kwargs: [],
    )
    plan = SearchNodePlan(
        node_key="node-1",
        label="情報源分散",
        description="detail fetch",
        queries=["江東区 賃貸 1LDK"],
        ranking_profile={},
        strategy_tags=["source_diversify"],
        depth=1,
    )

    result = manager._tool_enrich(
        context=manager.context,
        branch=plan,
        raw_results=[
            {"url": "https://example.com/property/abc", "source_name": "catalog"},
        ],
    )

    job = db.get_research_job(job_id)
    assert result["summary"]["detail_hit_count"] == 1
    assert job is not None
    assert "物件詳細ページを取得中" in job["latest_summary"]
    assert "https://example.com/property/abc" in job["latest_summary"]


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
    manager = make_manager(
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
    manager = make_manager(
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
    manager = make_manager(
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


def test_initial_node_plans_always_start_with_four_families(tmp_path: Path):
    database_path = str(tmp_path / "housing.db")
    db = Database(database_path)
    db.init()

    session_id, _ = db.create_session()
    approved_plan = {
        "user_memory_snapshot": {
            "budget_max": None,
            "must_conditions": ["バストイレ別", "独立洗面台", "2階以上", "南向き"],
            "nice_to_have": ["宅配ボックス", "オートロック", "角部屋"],
            "learned_preferences": {},
        },
        "retry_context": {
            "top_issues": ["検索結果が取得できていない"],
        },
    }
    job_id, _ = db.create_research_job(
        session_id=session_id,
        provider="openai",
        llm_config={},
        approved_plan=approved_plan,
    )
    manager = make_manager(
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

    plans = manager._initial_node_plans(ResearchExecutionState(query="q", seed_queries=["q"]))
    assert [plan.branch_family for plan in plans] == [
        "strict_primary",
        "strict_relaxed",
        "nearby_primary",
        "nearby_relaxed",
    ]


def test_initial_node_plans_keep_fixed_family_count_for_simple_tasks(tmp_path: Path):
    database_path = str(tmp_path / "housing.db")
    db = Database(database_path)
    db.init()

    session_id, _ = db.create_session()
    approved_plan = {
        "user_memory_snapshot": {
            "budget_max": 120000,
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
    manager = make_manager(
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

    plans = manager._initial_node_plans(ResearchExecutionState(query="q", seed_queries=["q"]))
    assert len(plans) == 4


def test_children_budget_for_summary_adapts_to_quality(tmp_path: Path):
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
    manager = make_manager(
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
        tree_children_per_expansion=2,
    )

    assert manager._children_budget_for({"branch_score": 82.0, "readiness": "high"}) == 1
    assert manager._children_budget_for({"branch_score": 45.0, "readiness": "low"}) == 3
    assert manager._children_budget_for({"branch_score": 62.0, "readiness": "medium"}) == 2


def test_recovery_operators_respect_adaptive_children_budget(tmp_path: Path):
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
    manager = make_manager(
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
        tree_children_per_expansion=2,
    )
    plan = SearchNodePlan(
        node_key="parent",
        label="parent",
        description="parent",
        queries=["query"],
        ranking_profile={},
        strategy_tags=[],
        depth=1,
    )

    operators = manager._recovery_operators_for_summary(
        plan=plan,
        summary={
            "status": "failed",
            "branch_score": 42.0,
            "readiness": "low",
            "failure_stage": "retrieve",
            "issues": [
                "検索結果が取得できていない",
                "上位候補の条件一致度が低い",
            ],
            "prune_reasons": [],
        },
    )

    assert operators == ["source_diversify", "detail_first", "exploit_best"]


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

    manager = make_manager(
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
    assert any(summary.get("intent") == "recovery" for summary in state.branch_summaries)
    assert any(int(summary.get("debug_depth") or 0) >= 1 for summary in state.branch_summaries)
    assert all(bool(summary.get("is_failed")) for summary in state.branch_summaries)


def test_tree_search_attaches_branch_result_summary_before_final_selection(
    tmp_path: Path, monkeypatch
):
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
    manager = make_manager(
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
    assert (
        state.selected_branch_summary["branch_result_summary"]["summary"]["branch_node_count"] == 2
    )
    selected_artifacts = manager._selected_artifacts(state)
    assert selected_artifacts is not None
    assert selected_artifacts.normalize["branch_result_summary"]["共通リスク"] == [
        "管理費の内訳が未確認"
    ]
