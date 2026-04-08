from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

from app.db import Database
from app.llm.base import LLMAdapter
from app.research.journal import ResearchJournal, ResearchNode
from app.research.offline_eval import (
    evaluate_branch,
    evaluate_final_result,
    select_best_branch,
    summarize_branch_failures,
)
from app.research.state_machine import ResearchStageDefinition, ResearchStateMachine
from app.research.tools import CallableResearchTool, ToolContext, ToolSpec, Toolbox
from app.stages.ranking import run_ranking
from app.stages.search_normalize import run_search_and_normalize


@dataclass(frozen=True)
class SearchNodePlan:
    node_key: str
    label: str
    description: str
    queries: list[str]
    ranking_profile: dict[str, Any]
    strategy_tags: list[str]
    depth: int
    parent_key: str | None = None
    parent_node_id: int | None = None


@dataclass
class SearchNodeArtifacts:
    plan: SearchNodePlan
    query_hash: str
    frontier_score: float
    retrieve: dict[str, Any] = field(default_factory=dict)
    enrich: dict[str, Any] = field(default_factory=dict)
    normalize: dict[str, Any] = field(default_factory=dict)
    rank: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    status: str = "queued"
    issue_streak: int = 0
    journal_node_id: int | None = None


@dataclass
class ResearchExecutionState:
    plan_result: dict[str, Any] = field(default_factory=dict)
    query: str = ""
    seed_queries: list[str] = field(default_factory=list)
    retry_context: dict[str, Any] = field(default_factory=dict)
    root_node: ResearchNode | None = None
    node_sequence: int = 0
    node_plans: dict[str, SearchNodePlan] = field(default_factory=dict)
    node_artifacts: dict[str, SearchNodeArtifacts] = field(default_factory=dict)
    frontier: list[str] = field(default_factory=list)
    branch_summaries: list[dict[str, Any]] = field(default_factory=list)
    branch_failures: dict[str, dict[str, Any]] = field(default_factory=dict)
    pruned_nodes: list[dict[str, Any]] = field(default_factory=list)
    selected_branch_summary: dict[str, Any] = field(default_factory=dict)
    selected_path: list[dict[str, Any]] = field(default_factory=list)
    best_node_key: str = ""
    best_node_stability: int = 0
    termination_reason: str = ""
    source_items: list[dict[str, Any]] = field(default_factory=list)
    search_summary: dict[str, Any] = field(default_factory=dict)
    offline_evaluation: dict[str, Any] = field(default_factory=dict)
    failure_summary: dict[str, Any] = field(default_factory=dict)
    research_summary: str = ""
    search_tree_summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchExecutionResult:
    query: str
    selected_branch_id: str
    branch_summaries: list[dict[str, Any]]
    normalized_properties: list[dict[str, Any]]
    ranked_properties: list[dict[str, Any]]
    duplicate_groups: list[dict[str, Any]]
    raw_results: list[dict[str, Any]]
    source_items: list[dict[str, Any]]
    search_summary: dict[str, Any]
    offline_evaluation: dict[str, Any]
    failure_summary: dict[str, Any]
    research_summary: str
    selected_path: list[dict[str, Any]]
    search_tree_summary: dict[str, Any]
    pruned_nodes: list[dict[str, Any]]


class HousingResearchAgentManager:
    def __init__(
        self,
        *,
        db: Database,
        session_id: str,
        job_id: str,
        approved_plan: dict[str, Any],
        user_memory: dict[str, Any],
        task_memory: dict[str, Any],
        provider: str,
        research_adapter: LLMAdapter | None,
        build_research_queries: Callable[[dict[str, Any], list[str]], list[str]],
        collect_search_results: Callable[..., tuple[list[dict[str, Any]], dict[str, Any]]],
        fetch_detail_html: Callable[[str], str | None],
        collect_source_items: Callable[..., list[dict[str, Any]]],
        tree_max_nodes: int = 12,
        tree_max_depth: int = 4,
        tree_batch_size: int = 2,
        tree_children_per_expansion: int = 2,
        tree_prune_score: int = 35,
        tree_stability_patience: int = 2,
    ):
        self.db = db
        self.session_id = session_id
        self.job_id = job_id
        self.approved_plan = approved_plan
        self.user_memory = user_memory
        self.task_memory = task_memory
        self.provider = provider
        self.research_adapter = research_adapter
        self.build_research_queries = build_research_queries
        self.collect_search_results = collect_search_results
        self.fetch_detail_html = fetch_detail_html
        self.collect_source_items = collect_source_items
        self.tree_max_nodes = max(4, tree_max_nodes)
        self.tree_max_depth = max(1, tree_max_depth)
        self.tree_batch_size = max(1, tree_batch_size)
        self.tree_children_per_expansion = max(1, tree_children_per_expansion)
        self.tree_prune_score = max(1, tree_prune_score)
        self.tree_stability_patience = max(1, tree_stability_patience)
        self.journal = ResearchJournal()
        self.context = ToolContext(
            session_id=session_id,
            job_id=job_id,
            user_memory=user_memory,
            task_memory=task_memory,
            approved_plan=approved_plan,
            provider=provider,
        )
        self.toolbox = self._build_toolbox()
        self.state_machine = self._build_state_machine()

    def _build_condition_summary(self, user_memory: dict[str, Any]) -> str:
        parts: list[str] = []
        area = str(user_memory.get("target_area") or "").strip()
        if area:
            parts.append(area)
        budget_max = int(user_memory.get("budget_max") or 0)
        if budget_max > 0:
            parts.append(f"家賃{budget_max:,}円以内")
        station_walk_max = int(user_memory.get("station_walk_max") or 0)
        if station_walk_max > 0:
            parts.append(f"駅徒歩{station_walk_max}分以内")
        layout = str(user_memory.get("layout_preference") or "").strip()
        if layout:
            parts.append(layout)
        return "・".join(parts)

    def _build_fallback_research_summary(
        self,
        *,
        ranked_properties: list[dict[str, Any]],
        normalized_properties: list[dict[str, Any]],
        search_summary: dict[str, Any],
        source_items: list[dict[str, Any]],
        offline_evaluation: dict[str, Any],
    ) -> str:
        user_memory = self.approved_plan.get("user_memory_snapshot", self.user_memory)
        condition_summary = self._build_condition_summary(user_memory)

        if not ranked_properties:
            lead = (
                f"{condition_summary}の条件で調査しましたが、問い合わせに進める候補は十分に揃いませんでした。"
                if condition_summary
                else "今回の条件で調査しましたが、問い合わせに進める候補は十分に揃いませんでした。"
            )
            detail_hit_count = int(search_summary.get("detail_hit_count") or 0)
            follow_up = (
                f"詳細ページを確認できた候補は{detail_hit_count}件にとどまっているため、条件を少し広げて再調査するのが安全です。"
            )
            return f"{lead}{follow_up}"

        by_id = {item["property_id_norm"]: item for item in normalized_properties}
        top_ranked = ranked_properties[0]
        top_property = by_id.get(top_ranked["property_id_norm"], {})
        candidate_count = len(ranked_properties)
        lead = (
            f"{condition_summary}の条件で{candidate_count}件を比較できました。"
            if condition_summary
            else f"今回の条件で{candidate_count}件を比較できました。"
        )
        top_name = str(top_property.get("building_name") or "第一候補の物件")
        rent = int(top_property.get("rent") or 0)
        station_walk = int(top_property.get("station_walk_min") or 0)
        top_detail_parts = [top_name]
        if rent > 0:
            top_detail_parts.append(f"家賃{rent:,}円")
        if station_walk > 0:
            top_detail_parts.append(f"駅徒歩{station_walk}分")
        top_detail = "、".join(top_detail_parts)

        detail_coverage = float(offline_evaluation.get("detail_coverage") or 0.0)
        top_reason = str(top_ranked.get("why_selected") or "").strip() or "条件との整合が高い候補です。"
        caution = str(top_ranked.get("why_not_selected") or "").strip()
        action = "まずは問い合わせに進める候補です。"
        if detail_coverage < 0.5 or not source_items:
            action = "ただし、掲載条件の最新性は問い合わせ前提で確認したい候補です。"
        elif caution:
            action = f"現時点では有力ですが、{caution}"
        return f"{lead}最上位候補は{top_detail}で、{top_reason}{action}"

    def _build_confirmation_items(
        self,
        *,
        ranked_properties: list[dict[str, Any]],
        normalized_properties: list[dict[str, Any]],
        source_items: list[dict[str, Any]],
        failure_summary: dict[str, Any],
    ) -> list[str]:
        by_id = {item["property_id_norm"]: item for item in normalized_properties}
        top_property = by_id.get(str(ranked_properties[0]["property_id_norm"]), {}) if ranked_properties else {}

        items: list[str] = []
        if source_items:
            items.append("掲載元ごとの差分条件")
        if top_property.get("notes"):
            items.append("募集条件の最新状況")
        if not top_property.get("rent"):
            items.append("家賃と管理費の内訳")
        if not top_property.get("station_walk_min"):
            items.append("駅徒歩分数")
        if not top_property.get("layout"):
            items.append("間取りと居室の広さ")
        items.extend(
            [
                "初期費用の内訳",
                "短期解約違約金・更新料・解約予告",
            ]
        )
        for recommendation in failure_summary.get("recommendations", []) or []:
            text = str(recommendation).strip()
            if text and text not in items:
                items.append(text)
        return items[:5]

    def _build_llm_research_summary(
        self,
        *,
        ranked_properties: list[dict[str, Any]],
        normalized_properties: list[dict[str, Any]],
        search_summary: dict[str, Any],
        source_items: list[dict[str, Any]],
        offline_evaluation: dict[str, Any],
        selected_branch_summary: dict[str, Any] | None = None,
        branch_summaries: list[dict[str, Any]] | None = None,
        failure_summary: dict[str, Any] | None = None,
        selected_path: list[dict[str, Any]] | None = None,
        search_tree_summary: dict[str, Any] | None = None,
    ) -> str:
        if self.research_adapter is None:
            return ""

        user_memory = self.approved_plan.get("user_memory_snapshot", self.user_memory)
        by_id = {item["property_id_norm"]: item for item in normalized_properties}
        top_candidates: list[dict[str, Any]] = []
        for ranked in ranked_properties[:3]:
            prop = by_id.get(ranked["property_id_norm"], {})
            top_candidates.append(
                {
                    "property_id_norm": ranked["property_id_norm"],
                    "building_name": str(prop.get("building_name") or "候補物件"),
                    "address": str(prop.get("address") or ""),
                    "rent": int(prop.get("rent") or 0),
                    "station_walk_min": int(prop.get("station_walk_min") or 0),
                    "layout": str(prop.get("layout") or ""),
                    "area_m2": float(prop.get("area_m2") or 0.0),
                    "score": float(ranked.get("score") or 0.0),
                    "why_selected": str(ranked.get("why_selected") or ""),
                    "why_not_selected": str(ranked.get("why_not_selected") or ""),
                }
            )
        selected_branch = selected_branch_summary or {}
        other_branches = [
            {
                "branch_id": str(item.get("branch_id") or ""),
                "label": str(item.get("label") or ""),
                "branch_score": float(item.get("branch_score") or 0.0),
                "frontier_score": float(item.get("frontier_score") or 0.0),
                "depth": int(item.get("depth") or 0),
                "detail_coverage": float(item.get("detail_coverage") or 0.0),
                "structured_ratio": float(item.get("structured_ratio") or 0.0),
                "normalized_count": int(item.get("normalized_count") or 0),
                "issues": [str(issue).strip() for issue in item.get("issues", [])[:2] if str(issue).strip()],
            }
            for item in (branch_summaries or [])
            if str(item.get("branch_id") or "") != str(selected_branch.get("branch_id") or "")
        ][:2]
        failure_info = failure_summary or {}
        payload = {
            "condition_summary": self._build_condition_summary(user_memory),
            "candidate_count": len(ranked_properties),
            "top_candidates": top_candidates,
            "selected_branch": {
                "branch_id": str(selected_branch.get("branch_id") or ""),
                "label": str(selected_branch.get("label") or ""),
                "branch_score": float(selected_branch.get("branch_score") or 0.0),
                "frontier_score": float(selected_branch.get("frontier_score") or 0.0),
                "depth": int(selected_branch.get("depth") or 0),
                "detail_coverage": float(selected_branch.get("detail_coverage") or 0.0),
                "structured_ratio": float(selected_branch.get("structured_ratio") or 0.0),
                "normalized_count": int(selected_branch.get("normalized_count") or 0),
                "summary": str(selected_branch.get("summary") or ""),
                "strategy_tags": [
                    str(tag).strip()
                    for tag in selected_branch.get("strategy_tags", []) or []
                    if str(tag).strip()
                ][:4],
            },
            "alternative_branches": other_branches,
            "selected_path": [
                {
                    "branch_id": str(item.get("branch_id") or ""),
                    "label": str(item.get("label") or ""),
                    "depth": int(item.get("depth") or 0),
                    "strategy_tags": [
                        str(tag).strip()
                        for tag in item.get("strategy_tags", []) or []
                        if str(tag).strip()
                    ][:4],
                    "branch_score": float(item.get("branch_score") or 0.0),
                }
                for item in (selected_path or [])[:4]
            ],
            "search_tree_summary": search_tree_summary or {},
            "search_summary": {
                "detail_hit_count": int(search_summary.get("detail_hit_count") or 0),
                "normalized_count": int(search_summary.get("normalized_count") or 0),
                "duplicate_group_count": int(search_summary.get("duplicate_group_count") or 0),
            },
            "offline_evaluation": {
                "readiness": str(offline_evaluation.get("readiness") or ""),
                "detail_coverage": float(offline_evaluation.get("detail_coverage") or 0.0),
                "structured_ratio": float(offline_evaluation.get("structured_ratio") or 0.0),
                "recommendations": [
                    str(item).strip()
                    for item in offline_evaluation.get("recommendations", [])[:3]
                    if str(item).strip()
                ],
            },
            "failure_summary": {
                "summary": str(failure_info.get("summary") or ""),
                "top_issues": [
                    str(item).strip()
                    for item in failure_info.get("top_issues", [])[:3]
                    if str(item).strip()
                ],
                "recommendations": [
                    str(item).strip()
                    for item in failure_info.get("recommendations", [])[:3]
                    if str(item).strip()
                ],
            },
            "source_count": len(source_items),
            "confirmation_items": self._build_confirmation_items(
                ranked_properties=ranked_properties,
                normalized_properties=normalized_properties,
                source_items=source_items,
                failure_summary=failure_info,
            ),
            "required_output_format": [
                "結論: ...",
                "理由: ...",
                "懸念: ...",
                "次の確認事項: ...",
            ],
        }
        summary = self.research_adapter.generate_text(
            system=(
                "You are a Japanese rental research assistant. "
                "Summarize the final research result for a user making a rental decision. "
                "Use only the provided facts. Do not invent properties, numbers, or conditions. "
                "Return exactly four lines in Japanese starting with "
                "'結論:', '理由:', '懸念:', and '次の確認事項:'."
            ),
            user=json.dumps(payload, ensure_ascii=False, indent=2),
            temperature=0.2,
        ).strip()
        return "\n".join(" ".join(line.split()) for line in summary.splitlines() if line.strip())

    def _build_state_machine(self) -> ResearchStateMachine:
        return ResearchStateMachine(
            [
                ResearchStageDefinition(
                    name="plan_finalize",
                    handler=self._handle_plan_finalize,
                    default_next_stage="tree_search",
                ),
                ResearchStageDefinition(
                    name="tree_search",
                    handler=self._handle_tree_search,
                    default_next_stage="synthesize",
                ),
                ResearchStageDefinition(
                    name="synthesize",
                    handler=self._handle_synthesize,
                    default_next_stage=None,
                ),
            ]
        )

    def _build_toolbox(self) -> Toolbox:
        node_schema = {
            "type": "object",
            "properties": {
                "node_key": {"type": "string"},
                "label": {"type": "string"},
                "description": {"type": "string"},
                "queries": {"type": "array", "items": {"type": "string"}},
                "ranking_profile": {"type": "object"},
                "strategy_tags": {"type": "array", "items": {"type": "string"}},
                "depth": {"type": "integer"},
                "parent_key": {"type": ["string", "null"]},
            },
            "required": [
                "node_key",
                "label",
                "description",
                "queries",
                "ranking_profile",
                "strategy_tags",
                "depth",
            ],
            "additionalProperties": True,
        }
        return Toolbox(
            [
                CallableResearchTool(
                    ToolSpec(
                        name="plan_finalize",
                        description="Fix an approved plan as the immutable starting point for tree search.",
                        output_schema={
                            "type": "object",
                            "properties": {
                                "summary": {"type": "string"},
                                "search_query": {"type": "string"},
                                "seed_queries": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["summary", "search_query", "seed_queries"],
                            "additionalProperties": True,
                        },
                    ),
                    self._tool_plan_finalize,
                ),
                CallableResearchTool(
                    ToolSpec(
                        name="retrieve",
                        description="Collect and merge search results for a search candidate.",
                        input_schema={
                            "type": "object",
                            "properties": {"branch": node_schema},
                            "required": ["branch"],
                            "additionalProperties": False,
                        },
                        output_schema={
                            "type": "object",
                            "properties": {
                                "raw_results": {"type": "array"},
                                "summary": {"type": "object"},
                                "per_query": {"type": "array"},
                            },
                            "required": ["raw_results", "summary", "per_query"],
                            "additionalProperties": True,
                        },
                    ),
                    self._tool_retrieve,
                ),
                CallableResearchTool(
                    ToolSpec(
                        name="enrich",
                        description="Fetch property detail pages for a search candidate.",
                        input_schema={
                            "type": "object",
                            "properties": {
                                "branch": node_schema,
                                "raw_results": {"type": "array"},
                            },
                            "required": ["branch", "raw_results"],
                            "additionalProperties": False,
                        },
                        output_schema={
                            "type": "object",
                            "properties": {
                                "detail_html_map": {"type": "object"},
                                "summary": {"type": "object"},
                            },
                            "required": ["detail_html_map", "summary"],
                            "additionalProperties": True,
                        },
                    ),
                    self._tool_enrich,
                ),
                CallableResearchTool(
                    ToolSpec(
                        name="normalize_dedupe",
                        description="Normalize raw results into a shared property schema and group duplicates.",
                        input_schema={
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "raw_results": {"type": "array"},
                                "detail_html_map": {"type": "object"},
                            },
                            "required": ["query", "raw_results", "detail_html_map"],
                            "additionalProperties": False,
                        },
                        output_schema={
                            "type": "object",
                            "properties": {
                                "normalized_properties": {"type": "array"},
                                "duplicate_groups": {"type": "array"},
                                "summary": {"type": "object"},
                            },
                            "required": ["normalized_properties", "duplicate_groups", "summary"],
                            "additionalProperties": True,
                        },
                    ),
                    self._tool_normalize,
                ),
                CallableResearchTool(
                    ToolSpec(
                        name="rank",
                        description="Rank normalized properties under a candidate-specific scoring profile.",
                        input_schema={
                            "type": "object",
                            "properties": {
                                "normalized_properties": {"type": "array"},
                                "ranking_profile": {"type": "object"},
                            },
                            "required": ["normalized_properties", "ranking_profile"],
                            "additionalProperties": False,
                        },
                        output_schema={
                            "type": "object",
                            "properties": {
                                "ranked_properties": {"type": "array"},
                                "why_selected": {"type": "object"},
                                "why_not_selected": {"type": "object"},
                                "ranking_profile": {"type": "object"},
                            },
                            "required": [
                                "ranked_properties",
                                "why_selected",
                                "why_not_selected",
                                "ranking_profile",
                            ],
                            "additionalProperties": True,
                        },
                    ),
                    self._tool_rank,
                ),
            ]
        )

    def _update_job(self, *, stage_name: str, progress_percent: int, latest_summary: str) -> None:
        self.db.update_research_job(
            self.job_id,
            current_stage=stage_name,
            progress_percent=progress_percent,
            latest_summary=latest_summary,
        )

    def _record_node(
        self,
        *,
        stage: str,
        node_type: str,
        status: str,
        input_payload: dict[str, Any],
        output_payload: dict[str, Any],
        reasoning: str,
        duration_ms: int = 0,
        parent_node_id: int | None = None,
        branch_id: str = "",
        selected: bool = False,
        metrics: dict[str, Any] | None = None,
    ) -> ResearchNode:
        row_id = self.db.add_research_journal_node(
            job_id=self.job_id,
            stage=stage,
            node_type=node_type,
            status=status,
            input_payload=input_payload,
            output_payload=output_payload,
            reasoning=reasoning,
            duration_ms=duration_ms,
            parent_node_id=parent_node_id,
            branch_id=branch_id,
            selected=selected,
            metrics_payload=metrics,
        )
        node = ResearchNode(
            id=row_id,
            stage=stage,
            node_type=node_type,
            status=status,
            input_payload=input_payload,
            output_payload=output_payload,
            reasoning=reasoning,
            duration_ms=duration_ms,
            parent_node_id=parent_node_id,
            branch_id=branch_id,
            selected=selected,
            metrics=metrics or {},
        )
        self.journal.append(node)
        return node

    def _run_stage(
        self,
        *,
        stage_name: str,
        progress_percent: int,
        latest_summary: str,
        input_payload: dict[str, Any],
        reasoning: str,
        runner: Callable[[], dict[str, Any]],
    ) -> tuple[ResearchNode, dict[str, Any]]:
        self._update_job(
            stage_name=stage_name,
            progress_percent=progress_percent,
            latest_summary=latest_summary,
        )
        started = time.perf_counter()
        try:
            output = runner()
        except Exception as exc:  # pragma: no cover - exercised through integration path
            duration_ms = int((time.perf_counter() - started) * 1000)
            self._record_node(
                stage=stage_name,
                node_type="stage",
                status="failed",
                input_payload=input_payload,
                output_payload={"error": str(exc)},
                reasoning=reasoning,
                duration_ms=duration_ms,
            )
            raise RuntimeError(str(exc)) from exc

        duration_ms = int((time.perf_counter() - started) * 1000)
        node = self._record_node(
            stage=stage_name,
            node_type="stage",
            status="completed",
            input_payload=input_payload,
            output_payload=output,
            reasoning=reasoning,
            duration_ms=duration_ms,
        )
        return node, output

    def _tool_plan_finalize(self, *, context: ToolContext) -> dict[str, Any]:
        seed_queries = [
            " ".join(str(item).split()).strip()
            for item in context.approved_plan.get("seed_queries", []) or []
            if " ".join(str(item).split()).strip()
        ]
        search_query = str(context.approved_plan.get("search_query") or "").strip()
        if not seed_queries and search_query:
            seed_queries = [search_query]
        return {
            "summary": (
                f"条件 {len(context.approved_plan.get('conditions', []))} 件で調査開始"
                f"（seed {len(seed_queries)}本）"
            ),
            "search_query": search_query or (seed_queries[0] if seed_queries else ""),
            "seed_queries": seed_queries,
        }

    def _tool_retrieve(self, *, context: ToolContext, branch: SearchNodePlan) -> dict[str, Any]:
        merged_by_url: dict[str, dict[str, Any]] = {}
        per_query: list[dict[str, Any]] = []
        catalog_total = 0
        brave_total = 0
        brave_errors: list[str] = []

        for index, query in enumerate(branch.queries, start=1):
            self._update_job(
                stage_name="tree_search",
                progress_percent=36,
                latest_summary=f"{branch.label}: {index}/{len(branch.queries)}件目を収集中",
            )
            results, source_summary = self.collect_search_results(
                query=query,
                user_memory=self._active_user_memory(),
                adapter=self.research_adapter,
            )
            catalog_total += int(source_summary["catalog_result_count"])
            brave_total += int(source_summary["brave_result_count"])
            if source_summary["brave_error"]:
                brave_errors.append(str(source_summary["brave_error"]))

            per_query.append(
                {
                    "query": query,
                    "result_count": len(results),
                    "catalog_result_count": source_summary["catalog_result_count"],
                    "brave_result_count": source_summary["brave_result_count"],
                }
            )

            for item in results:
                url = str(item.get("url") or "").strip()
                if not url:
                    continue
                source_name = str(item.get("source_name") or "unknown")
                if url not in merged_by_url:
                    merged_by_url[url] = {
                        **item,
                        "matched_queries": [query],
                        "source_names": [source_name],
                        "source_name": source_name,
                    }
                    continue
                existing = merged_by_url[url]
                if query not in existing["matched_queries"]:
                    existing["matched_queries"].append(query)
                if source_name not in existing["source_names"]:
                    existing["source_names"].append(source_name)
                snippets = list(existing.get("extra_snippets", []) or [])
                for snippet in item.get("extra_snippets", []) or []:
                    text = str(snippet).strip()
                    if text and text not in snippets:
                        snippets.append(text)
                existing["extra_snippets"] = snippets[:6]
                if len(existing["source_names"]) > 1:
                    existing["source_name"] = "multi_source"

        raw_results = list(merged_by_url.values())
        return {
            "raw_results": raw_results,
            "summary": {
                "query_count": len(branch.queries),
                "unique_url_count": len(raw_results),
                "catalog_result_count": catalog_total,
                "brave_result_count": brave_total,
                "brave_error_count": len(brave_errors),
            },
            "per_query": per_query,
        }

    def _tool_enrich(
        self,
        *,
        context: ToolContext,
        branch: SearchNodePlan,
        raw_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        detail_html_map: dict[str, str] = {}
        total = len(raw_results)
        for index, item in enumerate(raw_results, start=1):
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            detail_html = self.fetch_detail_html(url)
            if detail_html:
                detail_html_map[url] = detail_html
            if total and (index == 1 or index == total or index % 3 == 0):
                self._update_job(
                    stage_name="tree_search",
                    progress_percent=48,
                    latest_summary=f"{branch.label}: 詳細ページを補完中 {index}/{total}",
                )
        return {
            "detail_html_map": detail_html_map,
            "summary": {
                "detail_attempt_count": total,
                "detail_hit_count": len(detail_html_map),
                "summary": f"詳細ページを {len(detail_html_map)} 件取得",
            },
        }

    def _tool_normalize(
        self,
        *,
        context: ToolContext,
        query: str,
        raw_results: list[dict[str, Any]],
        detail_html_map: dict[str, str],
    ) -> dict[str, Any]:
        return run_search_and_normalize(
            query=query,
            search_results=raw_results,
            detail_fetcher=lambda url: detail_html_map.get(url),
            adapter=self.research_adapter,
        )

    def _tool_rank(
        self,
        *,
        context: ToolContext,
        normalized_properties: list[dict[str, Any]],
        ranking_profile: dict[str, Any],
    ) -> dict[str, Any]:
        return run_ranking(
            normalized_properties=normalized_properties,
            user_memory=self._active_user_memory(),
            ranking_profile=ranking_profile,
            adapter=self.research_adapter,
        )

    def _active_user_memory(self) -> dict[str, Any]:
        return self.approved_plan.get("user_memory_snapshot", self.user_memory)

    def _strategy_memory(self) -> dict[str, Any]:
        learned = self._active_user_memory().get("learned_preferences", {}) or {}
        strategy = learned.get("strategy_memory", {}) or {}
        if strategy:
            return strategy
        return self.task_memory.get("strategy_memory_snapshot", {}) or {}

    def _compose_query(self, *parts: Any) -> str:
        return " ".join(str(part).strip() for part in parts if str(part).strip()).strip()

    def _dedupe_queries(self, values: list[str], *, limit: int = 5) -> list[str]:
        deduped: list[str] = []
        for value in values:
            text = " ".join(str(value).split()).strip()
            if text and text not in deduped:
                deduped.append(text)
            if len(deduped) >= limit:
                break
        return deduped

    def _hash_queries(self, queries: list[str], ranking_profile: dict[str, Any]) -> str:
        payload = {
            "queries": [" ".join(str(item).split()) for item in queries],
            "ranking_profile": ranking_profile,
        }
        return hashlib.sha1(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]

    def _next_node_key(self, state: ResearchExecutionState, operator: str, depth: int) -> str:
        state.node_sequence += 1
        return f"{operator}-d{depth}-n{state.node_sequence}"

    def _merge_ranking_profile(
        self,
        base_profile: dict[str, Any],
        updates: dict[str, float],
    ) -> dict[str, Any]:
        merged = dict(base_profile)
        for key, value in updates.items():
            merged[key] = float(value)
        return merged

    def _llm_query_suggestions(
        self,
        *,
        operator: str,
        base_queries: list[str],
        user_memory: dict[str, Any],
    ) -> list[str]:
        if self.research_adapter is None:
            return []
        schema = {
            "type": "object",
            "properties": {
                "query_suggestions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 2,
                }
            },
            "required": ["query_suggestions"],
            "additionalProperties": False,
        }
        try:
            result = self.research_adapter.generate_structured(
                system=(
                    "You are proposing safe Japanese rental search query refinements. "
                    "Return at most two concise query suggestions grounded in the provided conditions. "
                    "Do not include site names or unsafe scraping instructions."
                ),
                user=json.dumps(
                    {
                        "operator": operator,
                        "base_queries": base_queries[:4],
                        "user_memory": {
                            "target_area": str(user_memory.get("target_area") or ""),
                            "budget_max": int(user_memory.get("budget_max") or 0),
                            "station_walk_max": int(user_memory.get("station_walk_max") or 0),
                            "layout_preference": str(user_memory.get("layout_preference") or ""),
                            "must_conditions": user_memory.get("must_conditions", []) or [],
                            "nice_to_have": user_memory.get("nice_to_have", []) or [],
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                schema=schema,
                temperature=0.2,
            )
        except Exception:
            return []
        return self._dedupe_queries(
            [str(item).strip() for item in result.get("query_suggestions", []) if str(item).strip()],
            limit=2,
        )

    def _queries_for_operator(
        self,
        *,
        base_queries: list[str],
        operator: str,
        user_memory: dict[str, Any],
    ) -> list[str]:
        area = str(user_memory.get("target_area") or "").strip()
        layout = str(user_memory.get("layout_preference") or "").strip()
        budget = int(user_memory.get("budget_max") or 0)
        walk = int(user_memory.get("station_walk_max") or 0)
        must_conditions = [
            str(item).strip()
            for item in user_memory.get("must_conditions", []) or []
            if str(item).strip()
        ]
        nice_to_have = [
            str(item).strip()
            for item in user_memory.get("nice_to_have", []) or []
            if str(item).strip()
        ]

        queries = list(base_queries)
        if operator == "tighten_match":
            queries.extend(
                [
                    self._compose_query(area, layout, " ".join(must_conditions[:2]), "賃貸"),
                    self._compose_query(area, layout, f"{int(budget / 10000)}万円" if budget else "", "賃貸"),
                    self._compose_query(area, layout, f"徒歩{walk}分" if walk else "", "賃貸"),
                ]
            )
        elif operator == "relax_for_coverage":
            queries.extend(
                [
                    self._compose_query(area, "賃貸"),
                    self._compose_query(area, "住みやすい", "賃貸"),
                    self._compose_query(area, " ".join(nice_to_have[:2]), "賃貸"),
                ]
            )
        elif operator == "source_diversify":
            queries.extend(
                [
                    self._compose_query(area, layout, "賃貸情報"),
                    self._compose_query(area, layout, "募集", "賃貸"),
                    self._compose_query(area, "不動産", "賃貸"),
                ]
            )
        elif operator == "detail_first":
            queries.extend(
                [
                    self._compose_query(area, layout, f"{int(budget / 10000)}万円" if budget else "", "設備", "賃貸"),
                    self._compose_query(area, layout, "詳細", "賃貸"),
                    self._compose_query(area, "初期費用", "賃貸"),
                ]
            )
        elif operator == "schema_first":
            queries.extend(
                [
                    self._compose_query(area, layout, "設備", "賃貸"),
                    self._compose_query(area, layout, "間取り", "賃貸"),
                    self._compose_query(area, "徒歩", "賃貸"),
                ]
            )
        elif operator == "exploit_best":
            queries.extend(
                [
                    self._compose_query(area, layout, f"{int(budget / 10000)}万円" if budget else "", "駅近", "賃貸"),
                    self._compose_query(area, layout, "候補", "賃貸"),
                ]
            )
        elif operator == "explore_adjacent":
            queries.extend(
                [
                    self._compose_query(area, " ".join(nice_to_have[:2]), "住みやすい", "賃貸"),
                    self._compose_query(area, layout, "広め", "賃貸"),
                ]
            )

        queries.extend(
            self._llm_query_suggestions(
                operator=operator,
                base_queries=queries[:4],
                user_memory=user_memory,
            )
        )
        return self._dedupe_queries(queries)

    def _profile_for_operator(
        self,
        *,
        base_profile: dict[str, Any],
        operator: str,
    ) -> dict[str, Any]:
        if operator == "tighten_match":
            return self._merge_ranking_profile(
                base_profile,
                {
                    "budget_match_bonus": 28.0,
                    "station_match_bonus": 18.0,
                    "layout_match_bonus": 14.0,
                    "rent_missing_penalty": 18.0,
                    "station_missing_penalty": 8.0,
                    "layout_missing_penalty": 7.0,
                },
            )
        if operator == "relax_for_coverage":
            return self._merge_ranking_profile(
                base_profile,
                {
                    "budget_near_bonus": 8.0,
                    "budget_far_penalty": 12.0,
                    "station_far_penalty": 6.0,
                    "rent_missing_penalty": 10.0,
                    "station_missing_penalty": 4.0,
                    "layout_missing_penalty": 4.0,
                },
            )
        if operator == "detail_first":
            return self._merge_ranking_profile(
                base_profile,
                {
                    "rent_missing_penalty": 24.0,
                    "station_missing_penalty": 12.0,
                    "layout_missing_penalty": 12.0,
                },
            )
        if operator == "schema_first":
            return self._merge_ranking_profile(
                base_profile,
                {
                    "rent_missing_penalty": 28.0,
                    "station_missing_penalty": 16.0,
                    "layout_missing_penalty": 16.0,
                },
            )
        if operator == "explore_adjacent":
            return self._merge_ranking_profile(
                base_profile,
                {
                    "budget_far_penalty": 10.0,
                    "station_far_penalty": 4.0,
                },
            )
        return dict(base_profile)

    def _operator_label(self, operator: str) -> str:
        labels = {
            "tighten_match": "tighten_match",
            "relax_for_coverage": "relax_for_coverage",
            "source_diversify": "source_diversify",
            "detail_first": "detail_first",
            "schema_first": "schema_first",
            "exploit_best": "exploit_best",
            "explore_adjacent": "explore_adjacent",
        }
        return labels.get(operator, operator)

    def _operator_description(self, operator: str) -> str:
        descriptions = {
            "tighten_match": "must 条件と予算一致度を強める探索",
            "relax_for_coverage": "候補数と詳細補完率を回復する探索",
            "source_diversify": "異なる検索表現で情報源の多様性を増やす探索",
            "detail_first": "詳細ページ取得率を優先する探索",
            "schema_first": "家賃・徒歩・間取りの取得率を優先する探索",
            "exploit_best": "有望な条件組み合わせを深掘りする探索",
            "explore_adjacent": "周辺条件に寄せて近傍探索する探索",
        }
        return descriptions.get(operator, "探索ノード")

    def _estimate_frontier_score(
        self,
        *,
        operator: str,
        depth: int,
        parent_summary: dict[str, Any] | None,
    ) -> float:
        base = float((parent_summary or {}).get("branch_score") or 60.0)
        if parent_summary is None:
            base = 60.0

        preferred = set(self._strategy_memory().get("preferred_strategy_tags", []) or [])
        avoided = set(self._strategy_memory().get("avoided_strategy_tags", []) or [])
        retry_issues = set(self._retry_context().get("top_issues", []) or [])

        tag_bonus = {
            "tighten_match": 4.0,
            "relax_for_coverage": 3.0,
            "source_diversify": 3.0,
            "detail_first": 4.0,
            "schema_first": 4.0,
            "exploit_best": 5.0,
            "explore_adjacent": 3.0,
        }.get(operator, 0.0)
        score = base + tag_bonus
        if operator in preferred:
            score += 4.0
        if operator in avoided:
            score -= 6.0
        if operator == "detail_first" and "詳細ページ補完率が低い" in retry_issues:
            score -= 4.0
        if operator == "schema_first" and "比較に必要な項目の欠損が多い" in retry_issues:
            score -= 4.0
        if depth >= 3:
            score -= (depth - 2) * 5.0
        return round(score, 2)

    def _make_node_plan(
        self,
        state: ResearchExecutionState,
        *,
        operator: str,
        base_queries: list[str],
        base_profile: dict[str, Any],
        parent_key: str | None,
        parent_node_id: int | None,
        depth: int,
        extra_tags: list[str] | None = None,
        parent_summary: dict[str, Any] | None = None,
    ) -> SearchNodePlan:
        strategy_tags = [operator] + [
            str(tag).strip()
            for tag in extra_tags or []
            if str(tag).strip() and str(tag).strip() != operator
        ]
        return SearchNodePlan(
            node_key=self._next_node_key(state, operator, depth),
            label=self._operator_label(operator),
            description=self._operator_description(operator),
            queries=self._queries_for_operator(
                base_queries=base_queries,
                operator=operator,
                user_memory=self._active_user_memory(),
            ),
            ranking_profile=self._profile_for_operator(
                base_profile=base_profile,
                operator=operator,
            ),
            strategy_tags=self._dedupe_queries(strategy_tags, limit=6),
            depth=depth,
            parent_key=parent_key,
            parent_node_id=parent_node_id,
        )

    def _initial_node_plans(self, state: ResearchExecutionState) -> list[SearchNodePlan]:
        user_memory = self._active_user_memory()
        seed_queries = self.seed_queries_for_search(state)
        base_queries = self.build_research_queries(user_memory, seed_queries)
        strategy_memory = self._strategy_memory()
        extra_tags = [
            str(tag).strip()
            for tag in strategy_memory.get("last_successful_path", []) or []
            if str(tag).strip()
        ]
        return [
            self._make_node_plan(
                state,
                operator="tighten_match",
                base_queries=base_queries,
                base_profile={},
                parent_key=None,
                parent_node_id=state.root_node.id if state.root_node else None,
                depth=1,
                extra_tags=extra_tags,
            ),
            self._make_node_plan(
                state,
                operator="relax_for_coverage",
                base_queries=base_queries,
                base_profile={},
                parent_key=None,
                parent_node_id=state.root_node.id if state.root_node else None,
                depth=1,
                extra_tags=extra_tags,
            ),
            self._make_node_plan(
                state,
                operator="source_diversify",
                base_queries=base_queries,
                base_profile={},
                parent_key=None,
                parent_node_id=state.root_node.id if state.root_node else None,
                depth=1,
                extra_tags=extra_tags,
            ),
        ]

    def seed_queries_for_search(self, state: ResearchExecutionState) -> list[str]:
        return state.seed_queries or ([state.query] if state.query else [])

    def _retry_context(self) -> dict[str, Any]:
        return self.approved_plan.get("retry_context", {}) or {}

    def _record_pruned_node(
        self,
        state: ResearchExecutionState,
        *,
        plan: SearchNodePlan,
        parent_node_id: int | None,
        prune_reasons: list[str],
        evaluation: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "node_key": plan.node_key,
            "label": plan.label,
            "depth": plan.depth,
            "strategy_tags": plan.strategy_tags,
            "prune_reasons": prune_reasons,
        }
        if evaluation:
            payload["evaluation"] = {
                "branch_score": float(evaluation.get("branch_score") or 0.0),
                "detail_coverage": float(evaluation.get("detail_coverage") or 0.0),
                "top_issue_class": str(evaluation.get("top_issue_class") or ""),
            }
        state.pruned_nodes.append(payload)
        self._record_node(
            stage="tree_search",
            node_type="search_pruned",
            status="completed",
            input_payload={
                "node_key": plan.node_key,
                "queries": plan.queries,
                "strategy_tags": plan.strategy_tags,
            },
            output_payload=payload,
            reasoning="重複・深さ・低品質・失敗再発の条件により探索を剪定する。",
            parent_node_id=parent_node_id,
            branch_id=plan.node_key,
            metrics=evaluation,
        )

    def _register_frontier_node(
        self,
        state: ResearchExecutionState,
        *,
        plan: SearchNodePlan,
        parent_summary: dict[str, Any] | None = None,
    ) -> None:
        if len(state.node_plans) >= self.tree_max_nodes:
            return
        if plan.depth > self.tree_max_depth:
            self._record_pruned_node(
                state,
                plan=plan,
                parent_node_id=plan.parent_node_id,
                prune_reasons=["depth_limit"],
            )
            return

        query_hash = self._hash_queries(plan.queries, plan.ranking_profile)
        if any(artifact.query_hash == query_hash for artifact in state.node_artifacts.values()):
            self._record_pruned_node(
                state,
                plan=plan,
                parent_node_id=plan.parent_node_id,
                prune_reasons=["duplicate_query_hash"],
            )
            return

        frontier_score = self._estimate_frontier_score(
            operator=plan.strategy_tags[0] if plan.strategy_tags else plan.label,
            depth=plan.depth,
            parent_summary=parent_summary,
        )
        state.node_plans[plan.node_key] = plan
        state.node_artifacts[plan.node_key] = SearchNodeArtifacts(
            plan=plan,
            query_hash=query_hash,
            frontier_score=frontier_score,
        )
        state.frontier.append(plan.node_key)

    def _select_frontier_nodes(self, state: ResearchExecutionState) -> list[str]:
        queued = [
            state.node_artifacts[key]
            for key in state.frontier
            if key in state.node_artifacts and state.node_artifacts[key].status == "queued"
        ]
        queued.sort(
            key=lambda artifact: (
                float(artifact.frontier_score),
                -int(artifact.plan.depth),
                artifact.plan.node_key,
            ),
            reverse=True,
        )
        return [artifact.plan.node_key for artifact in queued[: self.tree_batch_size]]

    def _compute_frontier_score(
        self,
        *,
        summary: dict[str, Any],
        parent_summary: dict[str, Any] | None,
        strategy_tags: list[str],
        depth: int,
    ) -> float:
        score = float(summary.get("branch_score") or 0.0)
        if parent_summary is not None:
            if float(summary.get("detail_coverage") or 0.0) > float(parent_summary.get("detail_coverage") or 0.0):
                score += 10.0
            if float(summary.get("structured_ratio") or 0.0) > float(parent_summary.get("structured_ratio") or 0.0):
                score += 8.0
            if str(summary.get("top_issue_class") or "") == str(parent_summary.get("top_issue_class") or ""):
                score -= 8.0
        parent_tags = set(parent_summary.get("strategy_tags", []) or []) if parent_summary else set()
        if any(tag not in parent_tags for tag in strategy_tags):
            score += 5.0

        preferred = set(self._strategy_memory().get("preferred_strategy_tags", []) or [])
        avoided = set(self._strategy_memory().get("avoided_strategy_tags", []) or [])
        score += 4.0 * len(preferred.intersection(strategy_tags))
        score -= 6.0 * len(avoided.intersection(strategy_tags))
        if depth >= 3:
            score -= (depth - 2) * 5.0
        return round(score, 2)

    def _prune_reasons_for_summary(
        self,
        state: ResearchExecutionState,
        *,
        summary: dict[str, Any],
        parent_summary: dict[str, Any] | None,
    ) -> list[str]:
        reasons: list[str] = []
        if float(summary.get("branch_score") or 0.0) < self.tree_prune_score:
            reasons.append("low_branch_score")
        if int(summary.get("depth") or 0) >= 1 and float(summary.get("detail_coverage") or 0.0) < 0.2:
            reasons.append("low_detail_coverage")
        if int(summary.get("depth") or 0) > self.tree_max_depth:
            reasons.append("depth_limit")
        if (
            parent_summary is not None
            and str(summary.get("top_issue_class") or "")
            and str(summary.get("top_issue_class") or "") == str(parent_summary.get("top_issue_class") or "")
        ):
            artifact = state.node_artifacts.get(str(summary.get("branch_id") or ""))
            if artifact is not None and artifact.issue_streak >= 2:
                reasons.append(f"repeated_issue:{summary.get('top_issue_class')}")
        return reasons

    def _build_failure_summary(
        self,
        *,
        plan: SearchNodePlan,
        artifacts: SearchNodeArtifacts,
        error_text: str,
        stage_name: str,
        parent_summary: dict[str, Any] | None,
    ) -> dict[str, Any]:
        search_summary = {}
        if artifacts.retrieve:
            search_summary |= artifacts.retrieve.get("summary", {})
        if artifacts.enrich:
            search_summary |= artifacts.enrich.get("summary", {})
        if artifacts.normalize:
            search_summary |= artifacts.normalize.get("summary", {})

        summary = evaluate_branch(
            branch_id=plan.node_key,
            label=plan.label,
            queries=plan.queries,
            raw_results=artifacts.retrieve.get("raw_results", []),
            normalized_properties=artifacts.normalize.get("normalized_properties", []),
            ranked_properties=artifacts.rank.get("ranked_properties", []),
            duplicate_groups=artifacts.normalize.get("duplicate_groups", []),
            search_summary=search_summary,
            parent_summary=parent_summary,
            strategy_tags=plan.strategy_tags,
            depth=plan.depth,
            query_hash=artifacts.query_hash,
        )
        summary["status"] = "failed"
        summary["frontier_score"] = 0.0
        summary["branch_score"] = 0.0
        summary["issues"] = list(summary.get("issues", [])) + [f"{stage_name}: {error_text}"]
        summary["recommendations"] = list(summary.get("recommendations", [])) + [
            "失敗した探索ノードを残しつつ別戦略へ切り替える"
        ]
        summary["summary"] = f"{plan.label}: failed at {stage_name} ({error_text})"
        summary["parent_key"] = plan.parent_key or ""
        summary["strategy_tags"] = plan.strategy_tags
        summary["query_hash"] = artifacts.query_hash
        summary["depth"] = plan.depth
        summary["prune_reasons"] = []
        return summary

    def _execute_candidate(
        self,
        state: ResearchExecutionState,
        *,
        plan: SearchNodePlan,
    ) -> dict[str, Any]:
        artifacts = state.node_artifacts[plan.node_key]
        parent_summary = (
            state.node_artifacts[plan.parent_key].summary
            if plan.parent_key and plan.parent_key in state.node_artifacts
            else None
        )
        started = time.perf_counter()
        try:
            retrieve_result = self.toolbox.run("retrieve", self.context, branch=plan)
            artifacts.retrieve = retrieve_result
            enrich_result = self.toolbox.run(
                "enrich",
                self.context,
                branch=plan,
                raw_results=retrieve_result.get("raw_results", []),
            )
            artifacts.enrich = enrich_result
            normalize_result = self.toolbox.run(
                "normalize_dedupe",
                self.context,
                query=state.query,
                raw_results=retrieve_result.get("raw_results", []),
                detail_html_map=enrich_result.get("detail_html_map", {}),
            )
            artifacts.normalize = normalize_result
            ranking_result = self.toolbox.run(
                "rank",
                self.context,
                normalized_properties=normalize_result.get("normalized_properties", []),
                ranking_profile=plan.ranking_profile,
            )
            artifacts.rank = ranking_result
            search_summary = (
                retrieve_result.get("summary", {})
                | enrich_result.get("summary", {})
                | normalize_result.get("summary", {})
            )
            summary = evaluate_branch(
                branch_id=plan.node_key,
                label=plan.label,
                queries=plan.queries,
                raw_results=retrieve_result.get("raw_results", []),
                normalized_properties=normalize_result.get("normalized_properties", []),
                ranked_properties=ranking_result.get("ranked_properties", []),
                duplicate_groups=normalize_result.get("duplicate_groups", []),
                search_summary=search_summary,
                parent_summary=parent_summary,
                strategy_tags=plan.strategy_tags,
                depth=plan.depth,
                query_hash=artifacts.query_hash,
            )
            summary["parent_key"] = plan.parent_key or ""
            summary["description"] = plan.description
            summary["frontier_score"] = self._compute_frontier_score(
                summary=summary,
                parent_summary=parent_summary,
                strategy_tags=plan.strategy_tags,
                depth=plan.depth,
            )
            parent_artifacts = state.node_artifacts.get(plan.parent_key or "")
            if parent_summary is not None and summary["top_issue_class"] == parent_summary.get("top_issue_class"):
                artifacts.issue_streak = (parent_artifacts.issue_streak if parent_artifacts else 0) + 1
            elif summary["top_issue_class"] != "healthy":
                artifacts.issue_streak = 1
            else:
                artifacts.issue_streak = 0
            summary["prune_reasons"] = self._prune_reasons_for_summary(
                state,
                summary=summary,
                parent_summary=parent_summary,
            )
            artifacts.summary = summary
            artifacts.frontier_score = float(summary.get("frontier_score") or artifacts.frontier_score)
            artifacts.status = "completed"
            duration_ms = int((time.perf_counter() - started) * 1000)
            node = self._record_node(
                stage="tree_search",
                node_type="search_candidate",
                status="completed",
                input_payload={
                    "node_key": plan.node_key,
                    "queries": plan.queries,
                    "strategy_tags": plan.strategy_tags,
                    "depth": plan.depth,
                },
                output_payload={
                    "retrieve_summary": retrieve_result.get("summary", {}),
                    "enrich_summary": enrich_result.get("summary", {}),
                    "normalize_summary": normalize_result.get("summary", {}),
                    "ranked_property_count": len(ranking_result.get("ranked_properties", [])),
                    "summary": summary.get("summary", ""),
                },
                reasoning="候補探索ノードを実行し、収集・補完・正規化・順位付けをまとめて評価する。",
                duration_ms=duration_ms,
                parent_node_id=plan.parent_node_id,
                branch_id=plan.node_key,
                metrics=summary,
            )
            artifacts.journal_node_id = node.id
            state.branch_summaries.append(summary)
            if summary["prune_reasons"]:
                self._record_pruned_node(
                    state,
                    plan=plan,
                    parent_node_id=node.id,
                    prune_reasons=summary["prune_reasons"],
                    evaluation=summary,
                )
            return summary
        except Exception as exc:
            duration_ms = int((time.perf_counter() - started) * 1000)
            failure_summary = self._build_failure_summary(
                plan=plan,
                artifacts=artifacts,
                error_text=str(exc),
                stage_name="tree_search",
                parent_summary=parent_summary,
            )
            artifacts.summary = failure_summary
            artifacts.status = "failed"
            state.branch_failures[plan.node_key] = failure_summary
            state.branch_summaries.append(failure_summary)
            node = self._record_node(
                stage="tree_search",
                node_type="search_candidate",
                status="failed",
                input_payload={
                    "node_key": plan.node_key,
                    "queries": plan.queries,
                    "strategy_tags": plan.strategy_tags,
                    "depth": plan.depth,
                },
                output_payload={"error": str(exc)},
                reasoning="探索ノード単位の失敗を全体失敗に直結させず、別ノード探索を継続する。",
                duration_ms=duration_ms,
                parent_node_id=plan.parent_node_id,
                branch_id=plan.node_key,
                metrics=failure_summary,
            )
            artifacts.journal_node_id = node.id
            return failure_summary

    def _expand_candidates_from_summary(
        self,
        state: ResearchExecutionState,
        *,
        plan: SearchNodePlan,
        summary: dict[str, Any],
    ) -> list[SearchNodePlan]:
        operators = [
            str(item).strip()
            for item in summary.get("expand_recommendations", []) or []
            if str(item).strip()
        ]
        if not operators:
            return []

        parent_artifacts = state.node_artifacts.get(plan.node_key)
        base_queries = plan.queries
        base_profile = plan.ranking_profile
        if parent_artifacts and parent_artifacts.rank.get("ranking_profile"):
            base_profile = dict(parent_artifacts.rank.get("ranking_profile", {}))
        children: list[SearchNodePlan] = []
        for operator in operators[: self.tree_children_per_expansion]:
            children.append(
                self._make_node_plan(
                    state,
                    operator=operator,
                    base_queries=base_queries,
                    base_profile=base_profile,
                    parent_key=plan.node_key,
                    parent_node_id=parent_artifacts.journal_node_id if parent_artifacts else plan.parent_node_id,
                    depth=plan.depth + 1,
                    extra_tags=plan.strategy_tags,
                    parent_summary=summary,
                )
            )
        return children

    def _default_selected_branch_summary(self) -> dict[str, Any]:
        return {
            "branch_id": "none",
            "node_key": "none",
            "label": "none",
            "status": "failed",
            "depth": 0,
            "query_count": 0,
            "queries": [],
            "raw_result_count": 0,
            "normalized_count": 0,
            "detail_hit_count": 0,
            "detail_coverage": 0.0,
            "duplicate_group_count": 0,
            "duplicate_ratio": 0.0,
            "structured_ratio": 0.0,
            "top_score": 0.0,
            "avg_top3_score": 0.0,
            "source_diversity": 0,
            "branch_score": 0.0,
            "frontier_score": 0.0,
            "issues": ["tree search の評価対象がありませんでした。"],
            "recommendations": ["検索条件とソース設定を見直して再試行する"],
            "summary": "tree search の結果がありません。",
            "strategy_tags": [],
            "prune_reasons": [],
        }

    def _selected_artifacts(self, state: ResearchExecutionState) -> SearchNodeArtifacts | None:
        return state.node_artifacts.get(str(state.selected_branch_summary.get("branch_id") or ""))

    def _build_selected_path(self, state: ResearchExecutionState) -> list[dict[str, Any]]:
        branch_id = str(state.selected_branch_summary.get("branch_id") or "")
        path: list[dict[str, Any]] = []
        current_key = branch_id
        while current_key:
            artifacts = state.node_artifacts.get(current_key)
            if artifacts is None:
                break
            summary = artifacts.summary or {}
            path.append(
                {
                    "branch_id": current_key,
                    "label": artifacts.plan.label,
                    "depth": artifacts.plan.depth,
                    "strategy_tags": artifacts.plan.strategy_tags,
                    "branch_score": float(summary.get("branch_score") or 0.0),
                    "frontier_score": float(summary.get("frontier_score") or artifacts.frontier_score),
                    "summary": str(summary.get("summary") or ""),
                }
            )
            current_key = artifacts.plan.parent_key or ""
        path.reverse()
        return path

    def _build_search_tree_summary(self, state: ResearchExecutionState) -> dict[str, Any]:
        issue_counter: Counter[str] = Counter()
        max_depth = 0
        for summary in state.branch_summaries:
            issue_class = str(summary.get("top_issue_class") or "").strip()
            if issue_class and issue_class != "healthy":
                issue_counter[issue_class] += 1
            max_depth = max(max_depth, int(summary.get("depth") or 0))
        selected_path_tags: list[str] = []
        for item in state.selected_path:
            for tag in item.get("strategy_tags", []) or []:
                text = str(tag).strip()
                if text and text not in selected_path_tags:
                    selected_path_tags.append(text)
        return {
            "executed_node_count": len(
                [item for item in state.branch_summaries if item.get("status") == "completed"]
            ),
            "failed_node_count": len(
                [item for item in state.branch_summaries if item.get("status") == "failed"]
            ),
            "pruned_node_count": len(state.pruned_nodes),
            "frontier_remaining": len(state.frontier),
            "termination_reason": state.termination_reason or "frontier_exhausted",
            "max_depth_reached": max_depth,
            "selected_branch_id": str(state.selected_branch_summary.get("branch_id") or ""),
            "selected_path_tags": selected_path_tags,
            "retry_context_used": bool(state.retry_context),
            "issue_distribution": {
                label: count for label, count in issue_counter.most_common(5)
            },
        }

    def _best_node_readiness(self, state: ResearchExecutionState) -> str:
        selected = select_best_branch(state.branch_summaries)
        if selected is None:
            return "low"
        artifacts = state.node_artifacts.get(str(selected.get("branch_id") or ""))
        ranked_properties = artifacts.rank.get("ranked_properties", []) if artifacts else []
        result = evaluate_final_result(
            selected_branch_summary=selected,
            visible_ranked_properties=ranked_properties,
            search_summary=(
                artifacts.retrieve.get("summary", {})
                | artifacts.enrich.get("summary", {})
                | artifacts.normalize.get("summary", {})
                if artifacts
                else {}
            ),
        )
        return str(result.get("readiness") or "low")

    def _handle_plan_finalize(self, state: ResearchExecutionState) -> str | None:
        _, state.plan_result = self._run_stage(
            stage_name="plan_finalize",
            progress_percent=10,
            latest_summary="承認済み計画を確認しています。",
            input_payload={"approved_plan": self.approved_plan},
            reasoning="ユーザー承認済みの計画を固定し、以降の調査に使う条件を確定する。",
            runner=lambda: self.toolbox.run("plan_finalize", self.context),
        )
        state.seed_queries = [
            " ".join(str(item).split()).strip()
            for item in state.plan_result.get("seed_queries", []) or []
            if " ".join(str(item).split()).strip()
        ]
        state.query = str(state.plan_result.get("search_query") or "")
        if not state.query and state.seed_queries:
            state.query = state.seed_queries[0]
        state.retry_context = self._retry_context()
        return None

    def _handle_tree_search(self, state: ResearchExecutionState) -> str | None:
        def runner() -> dict[str, Any]:
            retry_context = state.retry_context
            state.root_node = self._record_node(
                stage="tree_search",
                node_type="search_root",
                status="completed",
                input_payload={
                    "search_query": state.query,
                    "seed_queries": state.seed_queries,
                    "retry_context": retry_context,
                },
                output_payload={
                    "summary": "承認済み計画から tree search を開始",
                    "strategy_memory": self._strategy_memory(),
                },
                reasoning="承認済み計画、履歴戦略、再試行文脈を束ねた探索根を作る。",
            )

            for plan in self._initial_node_plans(state):
                self._register_frontier_node(state, plan=plan)

            while state.frontier and len(state.branch_summaries) < self.tree_max_nodes:
                selected_keys = self._select_frontier_nodes(state)
                if not selected_keys:
                    break

                should_stop = False
                for node_key in selected_keys:
                    if node_key not in state.frontier:
                        continue
                    state.frontier.remove(node_key)
                    plan = state.node_plans[node_key]
                    self._update_job(
                        stage_name="tree_search",
                        progress_percent=24 + min(58, len(state.branch_summaries) * 6),
                        latest_summary=f"{plan.label} を depth {plan.depth} で検証しています。",
                    )
                    summary = self._execute_candidate(state, plan=plan)

                    best_summary = select_best_branch(state.branch_summaries)
                    if best_summary is not None:
                        state.selected_branch_summary = best_summary
                        best_key = str(best_summary.get("branch_id") or "")
                        if best_key and best_key == state.best_node_key:
                            state.best_node_stability += 1
                        else:
                            state.best_node_key = best_key
                            state.best_node_stability = 1

                    if (
                        self._best_node_readiness(state) == "high"
                        and state.best_node_stability >= self.tree_stability_patience
                    ):
                        state.termination_reason = "stable_high_readiness"
                        should_stop = True
                        break

                    if len(state.branch_summaries) >= self.tree_max_nodes:
                        state.termination_reason = "node_budget_exhausted"
                        should_stop = True
                        break

                    if summary.get("status") == "completed" and not summary.get("prune_reasons"):
                        for child in self._expand_candidates_from_summary(
                            state,
                            plan=plan,
                            summary=summary,
                        ):
                            self._register_frontier_node(
                                state,
                                plan=child,
                                parent_summary=summary,
                            )

                if should_stop:
                    break

            if not state.termination_reason:
                state.termination_reason = (
                    "frontier_exhausted" if not state.frontier else "node_budget_exhausted"
                )

            state.selected_branch_summary = (
                select_best_branch(state.branch_summaries) or self._default_selected_branch_summary()
            )
            state.selected_path = self._build_selected_path(state)
            selected_artifacts = self._selected_artifacts(state)
            self._record_node(
                stage="tree_search",
                node_type="search_selection",
                status="completed",
                input_payload={"candidate_count": len(state.branch_summaries)},
                output_payload={
                    "selected_branch": state.selected_branch_summary,
                    "selected_path": state.selected_path,
                },
                reasoning="tree search の評価結果から最良ノードとその経路を採用する。",
                parent_node_id=(
                    selected_artifacts.journal_node_id
                    if selected_artifacts and selected_artifacts.journal_node_id
                    else state.root_node.id if state.root_node else None
                ),
                branch_id=str(state.selected_branch_summary.get("branch_id") or ""),
                selected=True,
                metrics=state.selected_branch_summary,
            )
            state.search_tree_summary = self._build_search_tree_summary(state)
            selected_label = str(state.selected_branch_summary.get("label") or "none")
            return {
                "summary": (
                    f"探索ノード{len(state.branch_summaries)}件を評価し、"
                    f"{selected_label} を採用"
                ),
                "selected_branch": state.selected_branch_summary,
                "selected_path": state.selected_path,
                "pruned_node_count": len(state.pruned_nodes),
                "search_tree_summary": state.search_tree_summary,
            }

        self._run_stage(
            stage_name="tree_search",
            progress_percent=22,
            latest_summary="動的 tree search を初期化しています。",
            input_payload={
                "search_query": state.query,
                "seed_queries": state.seed_queries,
                "retry_context": state.retry_context,
            },
            reasoning="固定分岐ではなく、評価に応じて次ノードを選ぶ動的探索を行う。",
            runner=runner,
        )
        return None

    def _handle_synthesize(self, state: ResearchExecutionState) -> str | None:
        selected_artifacts = self._selected_artifacts(state)
        selected_rank = selected_artifacts.rank if selected_artifacts else {}
        selected_normalize = selected_artifacts.normalize if selected_artifacts else {}
        selected_retrieve = selected_artifacts.retrieve if selected_artifacts else {}
        selected_enrich = selected_artifacts.enrich if selected_artifacts else {}

        state.source_items = self.collect_source_items(
            ranked_properties=selected_rank.get("ranked_properties", []),
            normalized_properties=selected_normalize.get("normalized_properties", []),
            raw_results=selected_retrieve.get("raw_results", []),
        )
        state.search_summary = (
            selected_normalize.get("summary", {})
            | selected_retrieve.get("summary", {})
            | selected_enrich.get("summary", {})
        )
        state.failure_summary = summarize_branch_failures(state.branch_summaries)
        state.offline_evaluation = evaluate_final_result(
            selected_branch_summary=state.selected_branch_summary,
            visible_ranked_properties=selected_rank.get("ranked_properties", []),
            search_summary=state.search_summary,
        )
        state.offline_evaluation["selected_path"] = state.selected_path
        state.offline_evaluation["search_tree_summary"] = state.search_tree_summary
        state.research_summary = self._build_fallback_research_summary(
            ranked_properties=selected_rank.get("ranked_properties", []),
            normalized_properties=selected_normalize.get("normalized_properties", []),
            search_summary=state.search_summary,
            source_items=state.source_items,
            offline_evaluation=state.offline_evaluation,
        )
        if self.research_adapter is not None:
            try:
                llm_summary = self._build_llm_research_summary(
                    ranked_properties=selected_rank.get("ranked_properties", []),
                    normalized_properties=selected_normalize.get("normalized_properties", []),
                    search_summary=state.search_summary,
                    source_items=state.source_items,
                    offline_evaluation=state.offline_evaluation,
                    selected_branch_summary=state.selected_branch_summary,
                    branch_summaries=state.branch_summaries,
                    failure_summary=state.failure_summary,
                    selected_path=state.selected_path,
                    search_tree_summary=state.search_tree_summary,
                )
                if llm_summary:
                    state.research_summary = llm_summary
            except Exception:
                pass

        self.db.add_audit_event(
            self.session_id,
            "search_normalize",
            {
                "query": state.query,
                "selected_branch_id": state.selected_branch_summary["branch_id"],
                "branch_summaries": state.branch_summaries,
                "selected_path": state.selected_path,
            },
            selected_normalize,
            "動的 tree search の最良ノードから正規化結果を採用",
        )
        self.db.add_audit_event(
            self.session_id,
            "ranking",
            {
                "selected_branch_id": state.selected_branch_summary["branch_id"],
                "branch_summaries": state.branch_summaries,
                "selected_path": state.selected_path,
            },
            selected_rank,
            "最良探索ノードの順位付け結果を採用",
        )
        self.db.add_audit_event(
            self.session_id,
            "offline_evaluator",
            {
                "selected_branch_id": state.selected_branch_summary["branch_id"],
                "search_tree_summary": state.search_tree_summary,
            },
            state.offline_evaluation,
            "探索品質をオフライン指標で評価し、次の改善候補を提示",
        )

        self._run_stage(
            stage_name="synthesize",
            progress_percent=94,
            latest_summary="結果をユーザー向けに整理しています。",
            input_payload={"selected_branch_id": state.selected_branch_summary["branch_id"]},
            reasoning="最良ノード、探索経路、失敗要因をまとめて次アクションへ接続する。",
            runner=lambda: {
                "selected_branch_id": state.selected_branch_summary["branch_id"],
                "source_item_count": len(state.source_items),
                "research_summary": state.research_summary,
                "offline_evaluation": state.offline_evaluation,
                "failure_summary": state.failure_summary,
                "selected_path": state.selected_path,
                "search_tree_summary": state.search_tree_summary,
            },
        )
        return None

    def execute(self) -> ResearchExecutionResult:
        state = ResearchExecutionState()
        self.state_machine.run(state, start_stage="plan_finalize")

        selected_artifacts = self._selected_artifacts(state)
        selected_rank = selected_artifacts.rank if selected_artifacts else {}
        selected_normalize = selected_artifacts.normalize if selected_artifacts else {}
        selected_retrieve = selected_artifacts.retrieve if selected_artifacts else {}

        return ResearchExecutionResult(
            query=state.query,
            selected_branch_id=str(state.selected_branch_summary.get("branch_id") or "none"),
            branch_summaries=state.branch_summaries,
            normalized_properties=selected_normalize.get("normalized_properties", []),
            ranked_properties=selected_rank.get("ranked_properties", []),
            duplicate_groups=selected_normalize.get("duplicate_groups", []),
            raw_results=selected_retrieve.get("raw_results", []),
            source_items=state.source_items,
            search_summary=state.search_summary,
            offline_evaluation=state.offline_evaluation,
            failure_summary=state.failure_summary,
            research_summary=state.research_summary,
            selected_path=state.selected_path,
            search_tree_summary=state.search_tree_summary,
            pruned_nodes=state.pruned_nodes,
        )
