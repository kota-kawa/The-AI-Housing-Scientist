from __future__ import annotations

from collections.abc import Callable
import threading
import time
from typing import Any

from app.research.journal import ResearchIntent, ResearchNode
from app.research.state_machine import ResearchStageDefinition, ResearchStateMachine
from app.research.tools import CallableResearchTool, Toolbox, ToolContext, ToolSpec
from app.stages.integrity_review import run_integrity_review
from app.stages.ranking import run_ranking
from app.stages.search_normalize import run_search_and_normalize

from .agent_manager_types import SearchNodePlan


class AgentManagerToolingMixin:
    # JP: cached singleflight loadを処理する。
    # EN: Process cached singleflight load.
    def _cached_singleflight_load(
        self,
        *,
        cache: dict[str, Any],
        inflight: dict[str, dict[str, Any]],
        key: str,
        loader: Callable[[], Any],
    ) -> tuple[Any, bool]:
        with self._cache_lock:
            if key in cache:
                return self._cache_copy(cache[key]), True
            entry = inflight.get(key)
            if entry is None:
                entry = {
                    "event": threading.Event(),
                    "error": None,
                    "value": None,
                }
                inflight[key] = entry
                owner = True
            else:
                owner = False

        if owner:
            try:
                value = loader()
            except Exception as exc:
                with self._cache_lock:
                    inflight.pop(key, None)
                    entry["error"] = exc
                    entry["event"].set()
                raise

            cached_value = self._cache_copy(value)
            with self._cache_lock:
                cache[key] = cached_value
                inflight.pop(key, None)
                entry["value"] = self._cache_copy(cached_value)
                entry["event"].set()
            return self._cache_copy(cached_value), False

        entry["event"].wait()
        if entry["error"] is not None:
            raise entry["error"]
        return self._cache_copy(entry["value"]), True

    # JP: state machineを構築する。
    # EN: Build state machine.
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

    # JP: toolboxを構築する。
    # EN: Build toolbox.
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
                "intent": {"type": "string"},
                "debug_depth": {"type": "integer"},
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
                        name="integrity_review",
                        description="Review listing integrity and drop stale or contradictory candidates before ranking.",
                        input_schema={
                            "type": "object",
                            "properties": {
                                "normalized_properties": {"type": "array"},
                                "raw_results": {"type": "array"},
                                "detail_html_map": {"type": "object"},
                            },
                            "required": ["normalized_properties", "raw_results", "detail_html_map"],
                            "additionalProperties": False,
                        },
                        output_schema={
                            "type": "object",
                            "properties": {
                                "normalized_properties": {"type": "array"},
                                "integrity_reviews": {"type": "array"},
                                "dropped_property_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "summary": {"type": "object"},
                            },
                            "required": [
                                "normalized_properties",
                                "integrity_reviews",
                                "dropped_property_ids",
                                "summary",
                            ],
                            "additionalProperties": True,
                        },
                    ),
                    self._tool_integrity_review,
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

    # JP: jobを更新する。
    # EN: Update job.
    def _update_job(self, *, stage_name: str, progress_percent: int, latest_summary: str) -> None:
        with self._job_lock:
            resolved_progress = max(self._job_progress_percent, progress_percent)
            self._job_progress_percent = resolved_progress
            self.db.update_research_job(
                self.job_id,
                current_stage=stage_name,
                progress_percent=resolved_progress,
                latest_summary=latest_summary,
            )

    # JP: nodeを記録する。
    # EN: Record node.
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
        intent: ResearchIntent = "draft",
        is_failed: bool | None = None,
        debug_depth: int = 0,
        metrics: dict[str, Any] | None = None,
    ) -> ResearchNode:
        resolved_is_failed = status == "failed" if is_failed is None else is_failed
        with self._journal_lock:
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
                intent=intent,
                is_failed=resolved_is_failed,
                debug_depth=debug_depth,
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
                intent=intent,
                is_failed=resolved_is_failed,
                debug_depth=debug_depth,
                duration_ms=duration_ms,
                parent_node_id=parent_node_id,
                branch_id=branch_id,
                selected=selected,
                metrics=metrics or {},
            )
            self.journal.append(node)
            return node

    # JP: recorded nodeを更新する。
    # EN: Update recorded node.
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
        intent: ResearchIntent | None = None,
        is_failed: bool | None = None,
        debug_depth: int | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        if node_id is None:
            return

        with self._journal_lock:
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
                intent=intent,
                is_failed=is_failed,
                debug_depth=debug_depth,
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
            if intent is not None:
                node.intent = intent
            if is_failed is not None:
                node.is_failed = is_failed
            if debug_depth is not None:
                node.debug_depth = debug_depth
            if metrics is not None:
                node.metrics = metrics

    # JP: stageを実行する。
    # EN: Run stage.
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

    # JP: tool plan finalizeを処理する。
    # EN: Process tool plan finalize.
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

    # JP: tool retrieveを処理する。
    # EN: Process tool retrieve.
    def _tool_retrieve(self, *, context: ToolContext, branch: SearchNodePlan) -> dict[str, Any]:
        merged_by_url: dict[str, dict[str, Any]] = {}
        per_query: list[dict[str, Any]] = []
        catalog_total = 0
        brave_total = 0
        brave_errors: list[str] = []
        cache_hit_count = 0

        for index, query in enumerate(branch.queries, start=1):
            self._update_job(
                stage_name="tree_search",
                progress_percent=36,
                latest_summary=f"{branch.label}: {index}/{len(branch.queries)}件目を収集中",
            )
            (results, source_summary), cache_hit = self._cached_singleflight_load(
                cache=self.search_result_cache,
                inflight=self._search_result_inflight,
                key=query,
                loader=lambda query=query: self.collect_search_results(
                    query=query,
                    user_memory=self._active_user_memory(),
                    adapter=self.research_adapter,
                ),
            )
            if cache_hit:
                cache_hit_count += 1
            catalog_total += int(source_summary.get("catalog_result_count") or 0)
            brave_total += int(source_summary.get("brave_result_count") or 0)
            if source_summary.get("brave_error"):
                brave_errors.append(str(source_summary["brave_error"]))

            per_query.append(
                {
                    "query": query,
                    "result_count": len(results),
                    "catalog_result_count": int(source_summary.get("catalog_result_count") or 0),
                    "brave_result_count": int(source_summary.get("brave_result_count") or 0),
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
                "cache_hit_count": cache_hit_count,
            },
            "per_query": per_query,
        }

    # JP: tool enrichを処理する。
    # EN: Process tool enrich.
    def _tool_enrich(
        self,
        *,
        context: ToolContext,
        branch: SearchNodePlan,
        raw_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        detail_html_map: dict[str, str] = {}
        total = len(raw_results)
        cache_hit_count = 0
        for index, item in enumerate(raw_results, start=1):
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            detail_html, cache_hit = self._cached_singleflight_load(
                cache=self.detail_html_cache,
                inflight=self._detail_html_inflight,
                key=url,
                loader=lambda url=url: self.fetch_detail_html(url),
            )
            if cache_hit:
                cache_hit_count += 1
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
                "cache_hit_count": cache_hit_count,
                "summary": f"詳細ページを {len(detail_html_map)} 件取得",
            },
        }

    # JP: tool normalizeを処理する。
    # EN: Process tool normalize.
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
            image_resolver=(
                lambda item, prop, detail_html="": self.resolve_property_image(
                    search_result=item,
                    property_data=prop,
                    detail_html=detail_html,
                    adapter=self.research_adapter,
                )
            )
            if self.resolve_property_image is not None
            else None,
        )

    # JP: tool rankを処理する。
    # EN: Process tool rank.
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

    # JP: tool integrity reviewを処理する。
    # EN: Process tool integrity review.
    def _tool_integrity_review(
        self,
        *,
        context: ToolContext,
        normalized_properties: list[dict[str, Any]],
        raw_results: list[dict[str, Any]],
        detail_html_map: dict[str, str],
    ) -> dict[str, Any]:
        return run_integrity_review(
            normalized_properties=normalized_properties,
            raw_results=raw_results,
            detail_html_map=detail_html_map,
            adapter=self.research_adapter,
        )
