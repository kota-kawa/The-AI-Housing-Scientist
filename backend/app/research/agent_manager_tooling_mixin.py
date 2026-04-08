from __future__ import annotations

import time
from typing import Any, Callable

from app.research.journal import ResearchNode
from app.research.state_machine import ResearchStageDefinition, ResearchStateMachine
from app.research.tools import CallableResearchTool, ToolContext, ToolSpec, Toolbox
from app.stages.ranking import run_ranking
from app.stages.search_normalize import run_search_and_normalize

from .agent_manager_types import SearchNodePlan


class AgentManagerToolingMixin:
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

    def _update_recorded_node(
        self,
        node_id: int | None,
        *,
        status: str | None = None,
        input_payload: dict[str, Any] | None = None,
        output_payload: dict[str, Any] | None = None,
        reasoning: str | None = None,
        duration_ms: int | None = None,
        parent_node_id: int | None = None,
        branch_id: str | None = None,
        selected: bool | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        if node_id is None:
            return

        self.db.update_research_journal_node(
            node_id,
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

        node = self.journal.get_node(node_id)
        if node is None:
            return
        if status is not None:
            node.status = status
        if input_payload is not None:
            node.input_payload = input_payload
        if output_payload is not None:
            node.output_payload = output_payload
        if reasoning is not None:
            node.reasoning = reasoning
        if duration_ms is not None:
            node.duration_ms = duration_ms
        if parent_node_id is not None:
            node.parent_node_id = parent_node_id
        if branch_id is not None:
            node.branch_id = branch_id
        if selected is not None:
            node.selected = selected
        if metrics is not None:
            node.metrics = metrics

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
