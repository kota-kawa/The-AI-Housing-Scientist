from __future__ import annotations

import json
import time
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
class ResearchBranchPlan:
    branch_id: str
    label: str
    description: str
    queries: list[str]
    ranking_profile: dict[str, Any]


@dataclass
class BranchExecutionArtifacts:
    branch: ResearchBranchPlan
    retrieve: dict[str, Any] = field(default_factory=dict)
    enrich: dict[str, Any] = field(default_factory=dict)
    normalize: dict[str, Any] = field(default_factory=dict)
    rank: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchExecutionState:
    plan_result: dict[str, Any] = field(default_factory=dict)
    query: str = ""
    query_stage_node: ResearchNode | None = None
    branch_plans: list[ResearchBranchPlan] = field(default_factory=list)
    branch_roots: dict[str, ResearchNode] = field(default_factory=dict)
    branch_parent_ids: dict[str, int] = field(default_factory=dict)
    branch_artifacts: dict[str, BranchExecutionArtifacts] = field(default_factory=dict)
    branch_failures: dict[str, dict[str, Any]] = field(default_factory=dict)
    branch_summaries: list[dict[str, Any]] = field(default_factory=list)
    selected_branch_summary: dict[str, Any] = field(default_factory=dict)
    source_items: list[dict[str, Any]] = field(default_factory=list)
    search_summary: dict[str, Any] = field(default_factory=dict)
    offline_evaluation: dict[str, Any] = field(default_factory=dict)
    failure_summary: dict[str, Any] = field(default_factory=dict)
    research_summary: str = ""


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
        build_research_queries: Callable[[dict[str, Any], str], list[str]],
        collect_search_results: Callable[..., tuple[list[dict[str, Any]], dict[str, Any]]],
        fetch_detail_html: Callable[[str], str | None],
        collect_source_items: Callable[..., list[dict[str, Any]]],
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

    def _build_llm_research_summary(
        self,
        *,
        ranked_properties: list[dict[str, Any]],
        normalized_properties: list[dict[str, Any]],
        search_summary: dict[str, Any],
        source_items: list[dict[str, Any]],
        offline_evaluation: dict[str, Any],
    ) -> str:
        if self.research_adapter is None or not ranked_properties:
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
                    "rent": int(prop.get("rent") or 0),
                    "station_walk_min": int(prop.get("station_walk_min") or 0),
                    "layout": str(prop.get("layout") or ""),
                    "area_m2": float(prop.get("area_m2") or 0.0),
                    "why_selected": str(ranked.get("why_selected") or ""),
                    "why_not_selected": str(ranked.get("why_not_selected") or ""),
                }
            )
        payload = {
            "condition_summary": self._build_condition_summary(user_memory),
            "candidate_count": len(ranked_properties),
            "top_candidates": top_candidates,
            "search_summary": {
                "detail_hit_count": int(search_summary.get("detail_hit_count") or 0),
                "normalized_count": int(search_summary.get("normalized_count") or 0),
                "duplicate_group_count": int(search_summary.get("duplicate_group_count") or 0),
            },
            "offline_evaluation": {
                "readiness": str(offline_evaluation.get("readiness") or ""),
                "detail_coverage": float(offline_evaluation.get("detail_coverage") or 0.0),
                "recommendations": [
                    str(item).strip()
                    for item in offline_evaluation.get("recommendations", [])[:3]
                    if str(item).strip()
                ],
            },
            "source_count": len(source_items),
            "required_next_action": "候補がある場合は問い合わせに進めるかどうかを自然に述べる。候補が弱い場合は再調査や条件調整を促す。",
        }
        summary = self.research_adapter.generate_text(
            system=(
                "You are a Japanese rental research assistant. "
                "Write a concise natural-language summary of the overall search result. "
                "Use only the provided facts. Do not invent properties, numbers, or conditions. "
                "Return only a short paragraph in Japanese, ideally 2 to 3 sentences."
            ),
            user=json.dumps(payload, ensure_ascii=False, indent=2),
            temperature=0.2,
        ).strip()
        return " ".join(summary.split())

    def _build_state_machine(self) -> ResearchStateMachine:
        return ResearchStateMachine(
            [
                ResearchStageDefinition(
                    name="plan_finalize",
                    handler=self._handle_plan_finalize,
                    default_next_stage="query_expand",
                ),
                ResearchStageDefinition(
                    name="query_expand",
                    handler=self._handle_query_expand,
                    default_next_stage="retrieve",
                ),
                ResearchStageDefinition(
                    name="retrieve",
                    handler=self._handle_retrieve,
                    default_next_stage="enrich",
                ),
                ResearchStageDefinition(
                    name="enrich",
                    handler=self._handle_enrich,
                    default_next_stage="normalize_dedupe",
                ),
                ResearchStageDefinition(
                    name="normalize_dedupe",
                    handler=self._handle_normalize,
                    default_next_stage="rank",
                ),
                ResearchStageDefinition(
                    name="rank",
                    handler=self._handle_rank,
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
        branch_schema = {
            "type": "object",
            "properties": {
                "branch_id": {"type": "string"},
                "label": {"type": "string"},
                "description": {"type": "string"},
                "queries": {"type": "array", "items": {"type": "string"}},
                "ranking_profile": {"type": "object"},
            },
            "required": ["branch_id", "label", "description", "queries", "ranking_profile"],
            "additionalProperties": True,
        }
        return Toolbox(
            [
                CallableResearchTool(
                    ToolSpec(
                        name="plan_finalize",
                        description="Fix an approved plan as the immutable starting point for branch search.",
                        output_schema={
                            "type": "object",
                            "properties": {
                                "summary": {"type": "string"},
                                "search_query": {"type": "string"},
                            },
                            "required": ["summary", "search_query"],
                            "additionalProperties": True,
                        },
                    ),
                    self._tool_plan_finalize,
                ),
                CallableResearchTool(
                    ToolSpec(
                        name="query_expand",
                        description="Generate safe branch plans by varying query sets and ranking profiles.",
                        input_schema={
                            "type": "object",
                            "properties": {"seed_query": {"type": "string"}},
                            "required": ["seed_query"],
                            "additionalProperties": False,
                        },
                        output_schema={
                            "type": "object",
                            "properties": {
                                "branches": {"type": "array", "items": branch_schema},
                                "summary": {"type": "string"},
                            },
                            "required": ["branches", "summary"],
                            "additionalProperties": True,
                        },
                    ),
                    self._tool_query_expand,
                ),
                CallableResearchTool(
                    ToolSpec(
                        name="retrieve",
                        description="Collect and merge search results for a branch.",
                        input_schema={
                            "type": "object",
                            "properties": {"branch": branch_schema},
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
                        description="Fetch property detail pages for a branch.",
                        input_schema={
                            "type": "object",
                            "properties": {
                                "branch": branch_schema,
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
                        description="Rank normalized properties under a branch-specific scoring profile.",
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
        return {
            "summary": f"条件 {len(context.approved_plan.get('conditions', []))} 件で調査開始",
            "search_query": str(context.approved_plan.get("search_query") or ""),
        }

    def _make_branch_plans(self, seed_query: str) -> list[ResearchBranchPlan]:
        user_memory = self.approved_plan.get("user_memory_snapshot", self.user_memory)
        balanced_queries = self.build_research_queries(user_memory, seed_query)

        area = str(user_memory.get("target_area") or "").strip()
        layout = str(user_memory.get("layout_preference") or "").strip()
        budget = int(user_memory.get("budget_max") or 0)
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

        strict_queries = [seed_query]
        strict_tokens = [token for token in [area, layout, " ".join(must_conditions[:2]), "賃貸"] if token]
        if strict_tokens:
            strict_queries.append(" ".join(strict_tokens))
        if budget:
            budget_tokens = [token for token in [area, layout, f"{int(budget / 10000)}万円", "賃貸"] if token]
            if budget_tokens:
                strict_queries.append(" ".join(budget_tokens))

        broad_queries = [seed_query]
        broad_tokens = [token for token in [area, "賃貸"] if token]
        if broad_tokens:
            broad_queries.append(" ".join(broad_tokens))
        if area or nice_to_have:
            broad_queries.append(" ".join(token for token in [area, " ".join(nice_to_have[:2]), "賃貸"] if token))
        if area:
            broad_queries.append(" ".join(token for token in [area, "住みやすい", "賃貸"] if token))

        def dedupe(queries: list[str]) -> list[str]:
            unique: list[str] = []
            for query in queries:
                normalized = " ".join(query.split()).strip()
                if normalized and normalized not in unique:
                    unique.append(normalized)
            return unique[:5]

        return [
            ResearchBranchPlan(
                branch_id="balanced",
                label="balanced",
                description="条件と情報量のバランスを取る標準分岐",
                queries=dedupe(balanced_queries),
                ranking_profile={},
            ),
            ResearchBranchPlan(
                branch_id="strict",
                label="strict",
                description="must条件と欠損ペナルティを強める厳密分岐",
                queries=dedupe(strict_queries),
                ranking_profile={
                    "budget_match_bonus": 28.0,
                    "station_match_bonus": 18.0,
                    "layout_match_bonus": 14.0,
                    "rent_missing_penalty": 18.0,
                    "station_missing_penalty": 8.0,
                    "layout_missing_penalty": 7.0,
                },
            ),
            ResearchBranchPlan(
                branch_id="broad",
                label="broad",
                description="候補数と詳細補完率を優先して広めに拾う分岐",
                queries=dedupe(broad_queries),
                ranking_profile={
                    "budget_near_bonus": 8.0,
                    "budget_far_penalty": 12.0,
                    "station_far_penalty": 6.0,
                    "rent_missing_penalty": 10.0,
                    "station_missing_penalty": 4.0,
                    "layout_missing_penalty": 4.0,
                },
            ),
        ]

    def _tool_query_expand(self, *, context: ToolContext, seed_query: str) -> dict[str, Any]:
        branches = self._make_branch_plans(seed_query)
        llm_summary = ""
        if self.research_adapter is not None:
            schema = {
                "type": "object",
                "properties": {
                    "branches": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "branch_id": {
                                    "type": "string",
                                    "enum": ["balanced", "strict", "broad"],
                                },
                                "query_suggestions": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "description_hint": {"type": "string"},
                            },
                            "required": ["branch_id", "query_suggestions", "description_hint"],
                            "additionalProperties": False,
                        },
                    },
                    "summary": {"type": "string"},
                },
                "required": ["branches", "summary"],
                "additionalProperties": False,
            }
            try:
                llm_payload = self.research_adapter.generate_structured(
                    system=(
                        "You are designing safe query branches for Japanese rental research. "
                        "Suggest only query variations grounded in the approved plan. "
                        "Keep the three branch IDs fixed as balanced, strict, and broad."
                    ),
                    user=(
                        "承認済み計画:\n"
                        f"{self.approved_plan}\n\n"
                        f"seed_query: {seed_query}"
                    ),
                    schema=schema,
                    temperature=0.2,
                )
                by_branch_id = {
                    str(item.get("branch_id") or ""): item
                    for item in llm_payload.get("branches", [])
                }
                adjusted_branches: list[ResearchBranchPlan] = []
                for branch in branches:
                    suggestion = by_branch_id.get(branch.branch_id, {})
                    suggested_queries = [
                        str(item).strip()
                        for item in suggestion.get("query_suggestions", [])
                        if str(item).strip()
                    ]
                    merged_queries = branch.queries
                    if suggested_queries:
                        merged_queries = list(dict.fromkeys(branch.queries + suggested_queries))[:5]
                    description_hint = str(suggestion.get("description_hint") or "").strip()
                    adjusted_branches.append(
                        ResearchBranchPlan(
                            branch_id=branch.branch_id,
                            label=branch.label,
                            description=description_hint or branch.description,
                            queries=merged_queries,
                            ranking_profile=dict(branch.ranking_profile),
                        )
                    )
                branches = adjusted_branches
                llm_summary = str(llm_payload.get("summary") or "").strip()
            except Exception:
                llm_summary = ""
        return {
            "branches": [
                {
                    "branch_id": branch.branch_id,
                    "label": branch.label,
                    "description": branch.description,
                    "queries": branch.queries,
                    "ranking_profile": branch.ranking_profile,
                }
                for branch in branches
            ],
            "summary": llm_summary or f"{len(branches)}本の安全な検索分岐を作成",
        }

    def _tool_retrieve(self, *, context: ToolContext, branch: ResearchBranchPlan) -> dict[str, Any]:
        merged_by_url: dict[str, dict[str, Any]] = {}
        per_query: list[dict[str, Any]] = []
        catalog_total = 0
        brave_total = 0
        brave_errors: list[str] = []

        for index, query in enumerate(branch.queries, start=1):
            self._update_job(
                stage_name="retrieve",
                progress_percent=25,
                latest_summary=f"{branch.label}: {index}/{len(branch.queries)}件目を収集中",
            )
            results, source_summary = self.collect_search_results(
                query=query,
                user_memory=self.approved_plan.get("user_memory_snapshot", self.user_memory),
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
        branch: ResearchBranchPlan,
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
                    stage_name="enrich",
                    progress_percent=45,
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
            user_memory=self.approved_plan.get("user_memory_snapshot", self.user_memory),
            ranking_profile=ranking_profile,
            adapter=self.research_adapter,
        )

    def _ensure_branch_artifacts(
        self,
        state: ResearchExecutionState,
        branch: ResearchBranchPlan,
    ) -> BranchExecutionArtifacts:
        if branch.branch_id not in state.branch_artifacts:
            state.branch_artifacts[branch.branch_id] = BranchExecutionArtifacts(branch=branch)
        return state.branch_artifacts[branch.branch_id]

    def _active_branches(self, state: ResearchExecutionState) -> list[ResearchBranchPlan]:
        return [
            branch for branch in state.branch_plans if branch.branch_id not in state.branch_failures
        ]

    def _selected_artifacts(self, state: ResearchExecutionState) -> BranchExecutionArtifacts | None:
        return state.branch_artifacts.get(str(state.selected_branch_summary.get("branch_id") or ""))

    def _build_branch_failure_summary(
        self,
        *,
        branch: ResearchBranchPlan,
        stage_name: str,
        exc: Exception,
        artifacts: BranchExecutionArtifacts,
    ) -> dict[str, Any]:
        search_summary = {}
        if artifacts.retrieve:
            search_summary |= artifacts.retrieve.get("summary", {})
        if artifacts.enrich:
            search_summary |= artifacts.enrich.get("summary", {})
        if artifacts.normalize:
            search_summary |= artifacts.normalize.get("summary", {})

        summary = evaluate_branch(
            branch_id=branch.branch_id,
            label=branch.label,
            queries=branch.queries,
            raw_results=artifacts.retrieve.get("raw_results", []),
            normalized_properties=artifacts.normalize.get("normalized_properties", []),
            ranked_properties=artifacts.rank.get("ranked_properties", []),
            duplicate_groups=artifacts.normalize.get("duplicate_groups", []),
            search_summary=search_summary,
        )
        summary["status"] = "failed"
        summary["branch_score"] = 0.0
        summary["issues"] = list(summary.get("issues", [])) + [f"{stage_name}: {exc}"]
        summary["recommendations"] = list(summary.get("recommendations", [])) + [
            "失敗した分岐を残しつつ他分岐で再試行する"
        ]
        summary["summary"] = f"{branch.label}: failed at {stage_name} ({exc})"
        return summary

    def _mark_branch_failed(
        self,
        state: ResearchExecutionState,
        *,
        branch: ResearchBranchPlan,
        stage_name: str,
        parent_node_id: int | None,
        input_payload: dict[str, Any],
        exc: Exception,
    ) -> None:
        artifacts = self._ensure_branch_artifacts(state, branch)
        failure_summary = self._build_branch_failure_summary(
            branch=branch,
            stage_name=stage_name,
            exc=exc,
            artifacts=artifacts,
        )
        state.branch_failures[branch.branch_id] = failure_summary
        state.branch_summaries.append(failure_summary)
        self._record_node(
            stage=stage_name,
            node_type="branch_stage",
            status="failed",
            input_payload=input_payload,
            output_payload={"error": str(exc)},
            reasoning="分岐内の失敗を全体ジョブ失敗に直結させず、他分岐へ継続する。",
            parent_node_id=parent_node_id,
            branch_id=branch.branch_id,
            metrics=failure_summary,
        )

    def _default_selected_branch_summary(self) -> dict[str, Any]:
        return {
            "branch_id": "none",
            "label": "none",
            "status": "failed",
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
            "issues": ["評価対象の分岐がありませんでした。"],
            "recommendations": ["検索条件とソース設定を見直して再試行する"],
            "summary": "branch search の結果がありません。",
        }

    def _handle_plan_finalize(self, state: ResearchExecutionState) -> str | None:
        _, state.plan_result = self._run_stage(
            stage_name="plan_finalize",
            progress_percent=10,
            latest_summary="承認済み計画を確認しています。",
            input_payload={"approved_plan": self.approved_plan},
            reasoning="ユーザー承認済みの計画を固定し、以降の調査に使う条件を確定する。",
            runner=lambda: self.toolbox.run("plan_finalize", self.context),
        )
        state.query = str(state.plan_result.get("search_query") or "")
        return None

    def _handle_query_expand(self, state: ResearchExecutionState) -> str | None:
        state.query_stage_node, query_result = self._run_stage(
            stage_name="query_expand",
            progress_percent=18,
            latest_summary="branch search のための検索分岐を作成しています。",
            input_payload={"search_query": state.query},
            reasoning="単発検索ではなく、安全な複数分岐を比較して最良の探索計画を選ぶ。",
            runner=lambda: self.toolbox.run("query_expand", self.context, seed_query=state.query),
        )
        state.branch_plans = [
            ResearchBranchPlan(
                branch_id=item["branch_id"],
                label=item["label"],
                description=item["description"],
                queries=list(item["queries"]),
                ranking_profile=dict(item.get("ranking_profile", {})),
            )
            for item in query_result["branches"]
        ]
        for branch in state.branch_plans:
            branch_root = self._record_node(
                stage="query_expand",
                node_type="branch_root",
                status="completed",
                input_payload={"search_query": state.query},
                output_payload={
                    "label": branch.label,
                    "description": branch.description,
                    "queries": branch.queries,
                    "ranking_profile": branch.ranking_profile,
                },
                reasoning="固定ステージの中で安全な探索枝を表すノード。",
                branch_id=branch.branch_id,
                parent_node_id=state.query_stage_node.id if state.query_stage_node else None,
            )
            state.branch_roots[branch.branch_id] = branch_root
            state.branch_parent_ids[branch.branch_id] = branch_root.id or 0
            self._ensure_branch_artifacts(state, branch)
        return None

    def _handle_retrieve(self, state: ResearchExecutionState) -> str | None:
        def runner() -> dict[str, Any]:
            successful = 0
            for branch in state.branch_plans:
                parent_id = state.branch_parent_ids.get(branch.branch_id)
                try:
                    started = time.perf_counter()
                    retrieve_result = self.toolbox.run("retrieve", self.context, branch=branch)
                    duration_ms = int((time.perf_counter() - started) * 1000)
                    retrieve_node = self._record_node(
                        stage="retrieve",
                        node_type="branch_stage",
                        status="completed",
                        input_payload={"queries": branch.queries},
                        output_payload=retrieve_result,
                        reasoning="分岐ごとに検索結果を収集し、URL単位で統合する。",
                        duration_ms=duration_ms,
                        parent_node_id=parent_id,
                        branch_id=branch.branch_id,
                    )
                    state.branch_parent_ids[branch.branch_id] = retrieve_node.id or 0
                    self._ensure_branch_artifacts(state, branch).retrieve = retrieve_result
                    successful += 1
                except Exception as exc:
                    self._mark_branch_failed(
                        state,
                        branch=branch,
                        stage_name="retrieve",
                        parent_node_id=parent_id,
                        input_payload={"queries": branch.queries},
                        exc=exc,
                    )
            return {
                "branch_count": len(state.branch_plans),
                "successful_branch_count": successful,
                "failed_branch_count": len(state.branch_failures),
            }

        self._run_stage(
            stage_name="retrieve",
            progress_percent=32,
            latest_summary="branch search の収集結果を集約しています。",
            input_payload={"branch_count": len(state.branch_plans)},
            reasoning="複数分岐の検索を並列設計で比較可能な形に揃える。",
            runner=runner,
        )
        return None

    def _handle_enrich(self, state: ResearchExecutionState) -> str | None:
        def runner() -> dict[str, Any]:
            successful = 0
            for branch in self._active_branches(state):
                artifacts = self._ensure_branch_artifacts(state, branch)
                parent_id = state.branch_parent_ids.get(branch.branch_id)
                try:
                    started = time.perf_counter()
                    enrich_result = self.toolbox.run(
                        "enrich",
                        self.context,
                        branch=branch,
                        raw_results=artifacts.retrieve.get("raw_results", []),
                    )
                    duration_ms = int((time.perf_counter() - started) * 1000)
                    enrich_node = self._record_node(
                        stage="enrich",
                        node_type="branch_stage",
                        status="completed",
                        input_payload={"raw_result_count": len(artifacts.retrieve.get("raw_results", []))},
                        output_payload=enrich_result,
                        reasoning="分岐ごとに詳細ページを取得し、構造化情報を増やす。",
                        duration_ms=duration_ms,
                        parent_node_id=parent_id,
                        branch_id=branch.branch_id,
                    )
                    state.branch_parent_ids[branch.branch_id] = enrich_node.id or 0
                    artifacts.enrich = enrich_result
                    successful += 1
                except Exception as exc:
                    self._mark_branch_failed(
                        state,
                        branch=branch,
                        stage_name="enrich",
                        parent_node_id=parent_id,
                        input_payload={"raw_result_count": len(artifacts.retrieve.get("raw_results", []))},
                        exc=exc,
                    )
            return {
                "branch_count": len(state.branch_plans),
                "successful_branch_count": successful,
                "failed_branch_count": len(state.branch_failures),
            }

        self._run_stage(
            stage_name="enrich",
            progress_percent=48,
            latest_summary="詳細ページ補完結果を集約しています。",
            input_payload={"branch_count": len(state.branch_plans)},
            reasoning="各分岐の詳細取得状況を揃え、比較可能な情報量に近づける。",
            runner=runner,
        )
        return None

    def _handle_normalize(self, state: ResearchExecutionState) -> str | None:
        def runner() -> dict[str, Any]:
            successful = 0
            for branch in self._active_branches(state):
                artifacts = self._ensure_branch_artifacts(state, branch)
                parent_id = state.branch_parent_ids.get(branch.branch_id)
                try:
                    started = time.perf_counter()
                    normalize_result = self.toolbox.run(
                        "normalize_dedupe",
                        self.context,
                        query=state.query,
                        raw_results=artifacts.retrieve.get("raw_results", []),
                        detail_html_map=artifacts.enrich.get("detail_html_map", {}),
                    )
                    duration_ms = int((time.perf_counter() - started) * 1000)
                    normalize_node = self._record_node(
                        stage="normalize_dedupe",
                        node_type="branch_stage",
                        status="completed",
                        input_payload={
                            "query": state.query,
                            "raw_result_count": len(artifacts.retrieve.get("raw_results", [])),
                        },
                        output_payload=normalize_result,
                        reasoning="共通スキーマへ揃えて重複候補を統合する。",
                        duration_ms=duration_ms,
                        parent_node_id=parent_id,
                        branch_id=branch.branch_id,
                    )
                    state.branch_parent_ids[branch.branch_id] = normalize_node.id or 0
                    artifacts.normalize = normalize_result
                    successful += 1
                except Exception as exc:
                    self._mark_branch_failed(
                        state,
                        branch=branch,
                        stage_name="normalize_dedupe",
                        parent_node_id=parent_id,
                        input_payload={
                            "query": state.query,
                            "raw_result_count": len(artifacts.retrieve.get("raw_results", [])),
                        },
                        exc=exc,
                    )
            return {
                "branch_count": len(state.branch_plans),
                "successful_branch_count": successful,
                "failed_branch_count": len(state.branch_failures),
            }

        self._run_stage(
            stage_name="normalize_dedupe",
            progress_percent=66,
            latest_summary="正規化と重複統合の代表結果を保存しています。",
            input_payload={"branch_count": len(state.branch_plans)},
            reasoning="各分岐を共通スキーマに揃え、比較と選択を可能にする。",
            runner=runner,
        )
        return None

    def _handle_rank(self, state: ResearchExecutionState) -> str | None:
        def runner() -> dict[str, Any]:
            for branch in self._active_branches(state):
                artifacts = self._ensure_branch_artifacts(state, branch)
                parent_id = state.branch_parent_ids.get(branch.branch_id)
                try:
                    started = time.perf_counter()
                    ranking_result = self.toolbox.run(
                        "rank",
                        self.context,
                        normalized_properties=artifacts.normalize.get("normalized_properties", []),
                        ranking_profile=branch.ranking_profile,
                    )
                    duration_ms = int((time.perf_counter() - started) * 1000)
                    branch_summary = evaluate_branch(
                        branch_id=branch.branch_id,
                        label=branch.label,
                        queries=branch.queries,
                        raw_results=artifacts.retrieve.get("raw_results", []),
                        normalized_properties=artifacts.normalize.get("normalized_properties", []),
                        ranked_properties=ranking_result.get("ranked_properties", []),
                        duplicate_groups=artifacts.normalize.get("duplicate_groups", []),
                        search_summary=(
                            artifacts.retrieve.get("summary", {})
                            | artifacts.enrich.get("summary", {})
                            | artifacts.normalize.get("summary", {})
                        ),
                    )
                    rank_node = self._record_node(
                        stage="rank",
                        node_type="branch_stage",
                        status="completed",
                        input_payload={
                            "normalized_property_count": len(
                                artifacts.normalize.get("normalized_properties", [])
                            )
                        },
                        output_payload=ranking_result,
                        reasoning="分岐ごとの scoring profile で問い合わせ候補を評価する。",
                        duration_ms=duration_ms,
                        parent_node_id=parent_id,
                        branch_id=branch.branch_id,
                        metrics=branch_summary,
                    )
                    state.branch_parent_ids[branch.branch_id] = rank_node.id or 0
                    artifacts.rank = ranking_result
                    state.branch_summaries.append(branch_summary)
                except Exception as exc:
                    self._mark_branch_failed(
                        state,
                        branch=branch,
                        stage_name="rank",
                        parent_node_id=parent_id,
                        input_payload={
                            "normalized_property_count": len(
                                artifacts.normalize.get("normalized_properties", [])
                            )
                        },
                        exc=exc,
                    )

            state.selected_branch_summary = (
                select_best_branch(state.branch_summaries) or self._default_selected_branch_summary()
            )
            selected_root = self.journal.branch_root(state.selected_branch_summary["branch_id"])
            self._record_node(
                stage="rank",
                node_type="branch_selection",
                status="completed",
                input_payload={"branch_scores": state.branch_summaries},
                output_payload=state.selected_branch_summary,
                reasoning="deterministic offline evaluator により最良分岐を選択する。",
                parent_node_id=selected_root.id if selected_root else state.query_stage_node.id if state.query_stage_node else None,
                branch_id=state.selected_branch_summary["branch_id"],
                selected=True,
                metrics=state.selected_branch_summary,
            )
            return {
                "selected_branch": state.selected_branch_summary,
                "branch_count": len(state.branch_summaries),
            }

        self._run_stage(
            stage_name="rank",
            progress_percent=82,
            latest_summary="最良分岐を確定しています。",
            input_payload={"branch_count": len(state.branch_plans)},
            reasoning="分岐評価により最良の検索・順位付け結果を確定する。",
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
            },
            selected_normalize,
            "安全な branch search の中で最良分岐を選び、その正規化結果を採用",
        )
        self.db.add_audit_event(
            self.session_id,
            "ranking",
            {
                "selected_branch_id": state.selected_branch_summary["branch_id"],
                "branch_summaries": state.branch_summaries,
            },
            selected_rank,
            "branch evaluator により選ばれた分岐の順位付け結果を採用",
        )
        self.db.add_audit_event(
            self.session_id,
            "offline_evaluator",
            {"selected_branch_id": state.selected_branch_summary["branch_id"]},
            state.offline_evaluation,
            "検索品質をオフライン指標で評価し、次の改善候補を提示",
        )

        self._run_stage(
            stage_name="synthesize",
            progress_percent=94,
            latest_summary="結果をユーザー向けに整理しています。",
            input_payload={"selected_branch_id": state.selected_branch_summary["branch_id"]},
            reasoning="最良分岐、失敗要因、オフライン評価をまとめて次アクションへ接続する。",
            runner=lambda: {
                "selected_branch_id": state.selected_branch_summary["branch_id"],
                "source_item_count": len(state.source_items),
                "research_summary": state.research_summary,
                "offline_evaluation": state.offline_evaluation,
                "failure_summary": state.failure_summary,
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
        )
