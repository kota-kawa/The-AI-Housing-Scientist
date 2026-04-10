from __future__ import annotations

from typing import Any

from app.db import utc_now_iso
from app.llm.base import LLMAdapter
from app.llm_config import route_config_for
from app.models import ChatMessageResponse, ResearchStateResponse, UIBlock
from app.research import HousingResearchAgentManager
from app.services import BraveSearchClient
from app.stages import run_final_report, run_risk_check
from app.stages.planner import _has_slot_value, run_planner

MAX_RESEARCH_QUERIES = 8
AREA_NEARBY_HINTS: dict[str, tuple[str, ...]] = {
    "町田": ("相模原", "橋本", "南町田"),
    "中野": ("東中野", "高円寺", "新井薬師前"),
    "江東区": ("門前仲町", "木場", "豊洲"),
    "渋谷": ("恵比寿", "代官山", "表参道"),
    "吉祥寺": ("三鷹", "西荻窪", "武蔵境"),
}
AREA_LINE_HINTS: dict[str, tuple[str, ...]] = {
    "町田": ("小田急線", "横浜線"),
    "中野": ("中央線", "東西線"),
    "江東区": ("東西線", "有楽町線"),
    "渋谷": ("山手線", "半蔵門線"),
    "吉祥寺": ("中央線", "井の頭線"),
}


# JP: query textを正規化する。
# EN: Normalize query text.
def _normalize_query_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


# JP: budget queryを整形する。
# EN: Format budget query.
def _format_budget_query(budget: int) -> str:
    if budget <= 0:
        return ""
    amount = budget / 10000
    if budget % 10000 == 0:
        return f"{int(amount)}万円以下"
    return f"{amount:.1f}".rstrip("0").rstrip(".") + "万円以下"


# JP: relaxed budget queryを処理する。
# EN: Process relaxed budget query.
def _relaxed_budget_query(budget: int) -> str:
    if budget <= 0:
        return ""
    return _format_budget_query(budget + 10000)


# JP: lookup area hintsを処理する。
# EN: Process lookup area hints.
def _lookup_area_hints(area: str, hint_map: dict[str, tuple[str, ...]]) -> list[str]:
    normalized = _normalize_query_text(area)
    if not normalized:
        return []
    for key, values in hint_map.items():
        if normalized == key or key in normalized or normalized in key:
            return list(values)
    return []


# JP: compose queryを処理する。
# EN: Process compose query.
def _compose_query(*parts: str) -> str:
    return " ".join(part for part in (_normalize_query_text(item) for item in parts) if part)


class OrchestratorResearchMixin:
    # JP: research queriesを構築する。
    # EN: Build research queries.
    def _build_research_queries(
        self, user_memory: dict[str, Any], seed_queries: list[str]
    ) -> list[str]:
        area = _normalize_query_text(user_memory.get("target_area"))
        layout = str(user_memory.get("layout_preference") or "").strip()
        budget = int(user_memory.get("budget_max") or 0)
        walk = int(user_memory.get("station_walk_max") or 0)
        listing_type_keyword = str(user_memory.get("listing_type") or "").strip()
        must_conditions = [
            _normalize_query_text(item)
            for item in user_memory.get("must_conditions", []) or []
            if _normalize_query_text(item)
        ]
        nice_to_have = [
            _normalize_query_text(item)
            for item in user_memory.get("nice_to_have", []) or []
            if _normalize_query_text(item)
        ]
        budget_token = _format_budget_query(budget)
        relaxed_budget_token = _relaxed_budget_query(budget)
        walk_token = f"徒歩{walk}分" if walk else ""
        core_must = " ".join(must_conditions[:2]).strip()
        core_nice = " ".join(nice_to_have[:2]).strip()
        nearby_areas = _lookup_area_hints(area, AREA_NEARBY_HINTS)
        line_hints = _lookup_area_hints(area, AREA_LINE_HINTS)

        normalized_seed_queries = [
            _normalize_query_text(item) for item in seed_queries if _normalize_query_text(item)
        ]
        strict_candidates: list[str] = []
        nearby_candidates: list[str] = []
        line_candidates: list[str] = []
        relaxed_candidates: list[str] = []
        signal_candidates: list[str] = []

        if area:
            strict_candidates.append(
                _compose_query(
                    area,
                    listing_type_keyword,
                    budget_token,
                    layout,
                    walk_token,
                    core_must,
                    core_nice,
                )
            )
        if area or layout:
            signal_candidates.append(
                _compose_query(area, layout, "住みやすい", listing_type_keyword, core_nice)
            )
        if walk:
            signal_candidates.append(
                _compose_query(area, "駅近", walk_token, listing_type_keyword, layout)
            )
        if must_conditions:
            strict_candidates.append(
                _compose_query(area, layout, core_must, listing_type_keyword, budget_token)
            )
            relaxed_candidates.append(
                _compose_query(
                    area, listing_type_keyword, budget_token, layout, walk_token, core_nice
                )
            )
        if nice_to_have:
            signal_candidates.append(
                _compose_query(area, layout, core_nice, listing_type_keyword, budget_token)
            )
        if budget:
            relaxed_candidates.append(
                _compose_query(
                    area, listing_type_keyword, relaxed_budget_token, layout, walk_token, core_must
                )
            )

        if area:
            # seed_queries に近隣エリアのクエリが既に含まれている場合は追加しない（二重追加防止）
            nearby_tokens = nearby_areas[:2] if nearby_areas else [f"{area}周辺"]
            seeds_cover_nearby = any(
                token.lower() in q.lower()
                for token in nearby_tokens
                for q in normalized_seed_queries
            )
            if not seeds_cover_nearby:
                if nearby_areas:
                    for nearby in nearby_areas[:2]:
                        nearby_candidates.append(
                            _compose_query(
                                nearby,
                                listing_type_keyword,
                                budget_token,
                                layout,
                                core_must or core_nice,
                            )
                        )
                else:
                    nearby_candidates.append(
                        _compose_query(
                            f"{area}周辺",
                            listing_type_keyword,
                            budget_token,
                            layout,
                            core_must or core_nice,
                        )
                    )

            # seed_queries に沿線クエリが既に含まれている場合は追加しない（二重追加防止）
            line_tokens = line_hints[:2] if line_hints else [f"{area}沿線"]
            seeds_cover_lines = any(
                token.lower() in q.lower() for token in line_tokens for q in normalized_seed_queries
            )
            if not seeds_cover_lines:
                if line_hints:
                    for line in line_hints[:2]:
                        line_candidates.append(
                            _compose_query(
                                line, area, listing_type_keyword, budget_token, layout, core_must
                            )
                        )
                else:
                    line_candidates.append(
                        _compose_query(area, "沿線", listing_type_keyword, budget_token, layout)
                    )

        candidates: list[str] = []
        candidates.extend(normalized_seed_queries[:2])
        if strict_candidates:
            candidates.append(strict_candidates[0])
        if nearby_candidates:
            candidates.append(nearby_candidates[0])
        if line_candidates:
            candidates.append(line_candidates[0])
        if relaxed_candidates:
            candidates.append(relaxed_candidates[0])
        if len(relaxed_candidates) > 1:
            candidates.append(relaxed_candidates[1])

        prioritized_buckets = [
            normalized_seed_queries[2:],
            strict_candidates[1:],
            nearby_candidates[1:],
            line_candidates[1:],
            relaxed_candidates[2:],
            signal_candidates,
        ]
        for bucket in prioritized_buckets:
            candidates.extend(bucket)

        deduped: list[str] = []
        for item in candidates:
            text = _normalize_query_text(item)
            if text and text not in deduped:
                deduped.append(text)
        return deduped[:MAX_RESEARCH_QUERIES]

    # JP: research source itemsを収集する。
    # EN: Collect research source items.
    def _collect_research_source_items(
        self,
        *,
        ranked_properties: list[dict[str, Any]],
        normalized_properties: list[dict[str, Any]],
        raw_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        by_id = {item["property_id_norm"]: item for item in normalized_properties}
        raw_by_url = {
            str(item.get("url") or ""): item
            for item in raw_results
            if str(item.get("url") or "").strip()
        }

        items: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        for ranked in ranked_properties[:6]:
            prop = by_id.get(ranked["property_id_norm"], {})
            url = str(prop.get("detail_url") or "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            raw = raw_by_url.get(url, {})
            queries = raw.get("matched_queries", []) or []
            items.append(
                {
                    "title": raw.get("title") or prop.get("building_name", "参照ソース"),
                    "url": url,
                    "source_name": raw.get("source_name", "source"),
                    "matched_property": prop.get("building_name", ""),
                    "reason": ranked.get("why_selected", ""),
                    "queries": queries[:3],
                }
            )
        return items

    # JP: search resultsを収集する。
    # EN: Collect search results.
    def _collect_search_results(
        self,
        *,
        query: str,
        user_memory: dict[str, Any],
        adapter: LLMAdapter | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        catalog_results = self.catalog.search(
            query=query,
            user_memory=user_memory,
            count=8,
            adapter=adapter,
        )
        brave_results: list[dict[str, Any]] = []
        brave_error = ""

        if self.settings.brave_search_api_key:
            try:
                brave_results = BraveSearchClient(
                    self.settings.brave_search_api_key,
                    timeout_seconds=self.settings.llm_timeout_seconds,
                ).search(query=query, count=6, adapter=adapter)
                for item in brave_results:
                    item["source_name"] = "brave"
            except Exception as exc:
                brave_error = str(exc)

        merged: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        for item in catalog_results + brave_results:
            url = str(item.get("url") or "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            merged.append(item)

        return merged, {
            "catalog_result_count": len(catalog_results),
            "brave_result_count": len(brave_results),
            "brave_error": brave_error,
        }

    # JP: process search messageを処理する。
    # EN: Process process search message.
    def _process_search_message(
        self,
        *,
        session_id: str,
        message: str,
        adapter: Any,
        llm_config: dict[str, Any],
        planner_result: dict[str, Any] | None = None,
        user_memory: dict[str, Any] | None = None,
        task_memory: dict[str, Any] | None = None,
    ) -> ChatMessageResponse:
        if user_memory is None or task_memory is None:
            user_memory, task_memory, llm_config = self._ensure_session_llm_config(session_id)
        profile_memory = self._get_profile_memory_for_session(session_id)
        if planner_result is None:
            planner_result = run_planner(
                message=message,
                user_memory=user_memory,
                adapter=adapter,
                profile_memory=profile_memory,
            )
        self.db.add_audit_event(
            session_id,
            "planner",
            {
                "message": message,
                "user_memory": user_memory,
                "profile_memory": profile_memory,
            },
            planner_result,
            "抽出済み条件と不足スロットを生成",
        )

        updated_user_memory = planner_result["user_memory"]
        follow_up_questions = planner_result.get("follow_up_questions", [])

        if planner_result["missing_slots"]:
            assistant_text = (
                "検索前に、まず大枠の条件を入力してください。選択肢でも自由入力でも進められます。"
            )
            task_memory["status"] = "awaiting_plan_inputs"
            task_memory["awaiting_contract_text"] = False
            task_memory["draft_research_plan"] = None
            task_memory["draft_llm_config"] = llm_config
            self.db.update_memories(session_id, updated_user_memory, task_memory)
            self.db.set_pending_action(session_id, None)
            self.db.set_session_status(session_id, "awaiting_plan_inputs")

            required_questions = self._build_planning_questions(
                user_memory=updated_user_memory,
                slots=list(planner_result["missing_slots"]),
                required=True,
                profile_memory=profile_memory,
            )
            blocks = [
                self._build_question_block(
                    questions=required_questions,
                    optional=False,
                )
            ]

            response = ChatMessageResponse(
                status="awaiting_plan_inputs",
                assistant_message=assistant_text,
                missing_slots=planner_result["missing_slots"],
                next_action=planner_result["next_action"],
                blocks=blocks,
                pending_confirmation=False,
                pending_action=None,
            )
            self.db.add_message(session_id, "assistant", response.model_dump())
            return response

        draft_plan = self._build_research_plan(
            user_memory=updated_user_memory,
            planner_result=planner_result,
            message=message,
            adapter=adapter,
            llm_config=llm_config,
        )

        task_memory["status"] = "awaiting_plan_confirmation"
        task_memory["awaiting_contract_text"] = False
        task_memory["profile_resume_pending"] = False
        task_memory["draft_research_plan"] = draft_plan
        task_memory["draft_llm_config"] = llm_config
        self.db.set_pending_action(session_id, None)
        self.db.update_memories(session_id, updated_user_memory, task_memory)
        self.db.set_session_status(session_id, "awaiting_plan_confirmation")

        blocks = [self._build_plan_block(draft_plan)]
        optional_slots = [
            slot
            for slot in ["station_walk_max", "move_in_date", "must_conditions", "nice_to_have"]
            if not _has_slot_value(slot, updated_user_memory)
        ]
        optional_questions = self._build_planning_questions(
            user_memory=updated_user_memory,
            slots=optional_slots,
            required=False,
            profile_memory=profile_memory,
        )
        if follow_up_questions and not optional_questions:
            optional_questions = follow_up_questions
        if optional_questions:
            blocks.append(
                self._build_question_block(
                    questions=optional_questions,
                    optional=True,
                )
            )
        blocks.append(
            UIBlock(
                type="actions",
                title="調査をどう進めますか",
                content={
                    "items": [
                        {
                            "label": "この計画で調査を始める",
                            "action_type": "approve_research_plan",
                            "payload": {},
                        },
                        {
                            "label": "条件を追加して計画を更新する",
                            "action_type": "revise_research_plan",
                            "payload": {},
                        },
                    ]
                },
            )
        )

        assistant_text = str(draft_plan.get("assistant_message") or "").strip()
        if not assistant_text:
            assistant_text = (
                "調査計画を作成しました。内容を確認してから、明示承認で調査を開始します。"
            )
        if optional_questions and "追加" not in assistant_text:
            assistant_text += "追加で分かる条件があれば、下の候補から反映できます。"

        response = ChatMessageResponse(
            status="awaiting_plan_confirmation",
            assistant_message=assistant_text,
            missing_slots=[],
            next_action="approve_research_plan",
            blocks=blocks,
            pending_confirmation=False,
            pending_action=None,
        )
        self.db.add_message(session_id, "assistant", response.model_dump())
        return response

    # JP: research jobを実行する。
    # EN: Execute research job.
    def _execute_research_job(self, job_id: str) -> dict[str, Any]:
        job = self.db.get_research_job(job_id)
        if job is None:
            raise RuntimeError("research job not found")

        session_id = job["session_id"]
        approved_plan = job["approved_plan"]
        user_memory, task_memory = self.db.get_memories(session_id)
        job_llm_config = self._normalize_llm_config(
            job.get("llm_config")
            or task_memory.get("approved_llm_config")
            or task_memory.get("draft_llm_config")
        )
        research_route = route_config_for(job_llm_config, "research_default")
        research_provider = str(
            self._resolve_provider_for_model(str(research_route["model"]))
            if str(research_route["model"]).strip()
            else job.get("provider") or self.settings.llm_default_provider
        )
        research_adapter = self._get_adapter_for_route(
            llm_config=job_llm_config,
            route_key="research_default",
            session_id=session_id,
            job_id=job_id,
            interaction_type="research",
        )
        manager = HousingResearchAgentManager(
            db=self.db,
            session_id=session_id,
            job_id=job_id,
            approved_plan=approved_plan,
            user_memory=user_memory,
            task_memory=task_memory,
            provider=research_provider,
            research_adapter=research_adapter,
            build_research_queries=self._build_research_queries,
            collect_search_results=self._collect_search_results,
            fetch_detail_html=self.catalog.fetch_detail_html,
            collect_source_items=self._collect_research_source_items,
            resolve_property_image=self.property_images.resolve,
            tree_max_nodes=self.settings.research_tree_max_nodes,
            tree_max_depth=self.settings.research_tree_max_depth,
            tree_batch_size=self.settings.research_tree_batch_size,
            tree_children_per_expansion=self.settings.research_tree_children_per_expansion,
            tree_prune_score=self.settings.research_tree_prune_score,
            tree_stability_patience=self.settings.research_tree_stability_patience,
            tree_min_nodes_before_stable_stop=(
                self.settings.research_tree_min_nodes_before_stable_stop
            ),
            tree_min_best_score_gap=self.settings.research_tree_min_best_score_gap,
        )
        execution_result = manager.execute()
        final_report_result = run_final_report(
            stage_nodes=manager.journal.stage_nodes,
            selected_branch_nodes=manager.journal.selected_branch_nodes(),
            adapter=research_adapter,
        )
        execution_result.final_report_markdown = str(
            final_report_result.get("report_markdown") or ""
        ).strip()

        session = self.db.get_session(session_id)
        profile_id = session["profile_id"] if session is not None else ""
        updated_user_memory = approved_plan.get("user_memory_snapshot", user_memory)
        if profile_id:
            updated_user_memory = self._sync_profile_after_search(
                profile_id=profile_id,
                user_memory=updated_user_memory,
                query=execution_result.query,
                adapter=research_adapter,
                search_outcome={
                    "selected_branch_id": execution_result.selected_branch_id,
                    "selected_path": execution_result.selected_path,
                    "search_tree_summary": execution_result.search_tree_summary,
                    "readiness": execution_result.offline_evaluation.get("readiness", ""),
                    "top_issues": execution_result.failure_summary.get("top_issues", []),
                },
            )

        task_memory["status"] = "research_completed"
        task_memory["awaiting_contract_text"] = False
        task_memory["profile_resume_pending"] = False
        task_memory["last_query"] = execution_result.query
        task_memory["last_normalized_properties"] = execution_result.normalized_properties
        task_memory["last_ranked_properties"] = execution_result.ranked_properties
        task_memory["last_duplicate_groups"] = execution_result.duplicate_groups
        task_memory["last_integrity_reviews"] = execution_result.integrity_reviews
        task_memory["last_dropped_property_ids"] = execution_result.dropped_property_ids
        task_memory["last_branch_result_summary"] = execution_result.branch_result_summary
        task_memory["last_final_report"] = execution_result.final_report_markdown
        task_memory["last_search_summary"] = execution_result.search_summary
        task_memory["last_source_items"] = execution_result.source_items
        task_memory["last_research_summary"] = execution_result.research_summary
        task_memory["selected_property_id"] = None
        task_memory["risk_items"] = []
        task_memory["property_reactions"] = {}
        task_memory["comparison_property_ids"] = []
        task_memory["approved_research_plan"] = approved_plan
        task_memory["approved_llm_config"] = job_llm_config
        task_memory["active_research_job_id"] = None
        task_memory["last_research_job_id"] = job_id
        task_memory["last_llm_config"] = job_llm_config
        task_memory["selected_branch_id"] = execution_result.selected_branch_id
        task_memory["branch_summaries"] = execution_result.branch_summaries
        task_memory["offline_evaluation"] = execution_result.offline_evaluation
        task_memory["failure_summary"] = execution_result.failure_summary
        task_memory["selected_path"] = execution_result.selected_path
        task_memory["search_tree_summary"] = execution_result.search_tree_summary
        task_memory["pruned_nodes"] = execution_result.pruned_nodes
        task_memory["latest_strategy_episode_id"] = job_id
        task_memory["strategy_memory_snapshot"] = (
            updated_user_memory.get("learned_preferences", {}) or {}
        ).get("strategy_memory", {})
        self.db.set_pending_action(session_id, None)
        self.db.update_memories(session_id, updated_user_memory, task_memory)
        self.db.add_audit_event(
            session_id,
            "final_report",
            {
                "selected_branch_id": execution_result.selected_branch_id,
                "selected_path": execution_result.selected_path,
            },
            final_report_result,
            "探索 journal と選択済み分岐を走査して最終レポートを生成",
        )

        self.db.update_research_job(
            job_id,
            status="completed",
            current_stage="synthesize",
            progress_percent=100,
            latest_summary=execution_result.research_summary or "調査が完了しました。",
            finished_at=utc_now_iso(),
        )
        completed_job = self.db.get_research_job(job_id)
        visible_ranked_properties = self._visible_ranked_properties(
            execution_result.ranked_properties,
            task_memory,
        )
        response = ChatMessageResponse(
            status="research_completed",
            assistant_message=(
                execution_result.research_summary
                or (
                    f"調査が完了しました。{len(visible_ranked_properties)}件の候補を比較し、"
                    "問い合わせに進める候補を整理しました。"
                )
            ),
            missing_slots=[],
            next_action="select_property",
            blocks=self._build_research_result_blocks(
                research_summary=execution_result.research_summary,
                final_report_markdown=execution_result.final_report_markdown,
                ranked_properties=visible_ranked_properties,
                normalized_properties=execution_result.normalized_properties,
                search_summary=execution_result.search_summary,
                source_items=execution_result.source_items,
                task_memory=task_memory,
                job_id=completed_job["id"] if completed_job else job_id,
            ),
            pending_confirmation=False,
            pending_action=None,
        )
        self.db.update_research_job(job_id, result_payload=response.model_dump())
        self.db.set_session_status(session_id, "research_completed")
        self.db.add_message(session_id, "assistant", response.model_dump())
        return response.model_dump()

    # JP: process next research jobを処理する。
    # EN: Process process next research job.
    def process_next_research_job(self) -> bool:
        job = self.db.claim_next_research_job()
        if job is None:
            return False

        try:
            self._execute_research_job(job["id"])
        except Exception as exc:
            self.db.update_research_job(
                job["id"],
                status="failed",
                latest_summary="調査に失敗しました。",
                error_message=str(exc),
                finished_at=utc_now_iso(),
            )
            failed_job = self.db.get_research_job(job["id"]) or job
            session_id = failed_job["session_id"]
            user_memory, task_memory = self.db.get_memories(session_id)
            task_memory["status"] = "research_failed"
            task_memory["active_research_job_id"] = None
            task_memory["last_research_job_id"] = job["id"]
            self.db.update_memories(session_id, user_memory, task_memory)
            response = ChatMessageResponse(
                status="research_failed",
                assistant_message="調査中にエラーが発生しました。条件は保持しているので再実行できます。",
                missing_slots=[],
                next_action="retry_research_job",
                blocks=self._build_research_progress_blocks(
                    failed_job,
                    task_memory=task_memory,
                )
                + [
                    UIBlock(
                        type="warning",
                        title="調査エラー",
                        content={"body": str(exc)},
                    ),
                    UIBlock(
                        type="actions",
                        title="次のアクション",
                        content={
                            "items": [
                                {
                                    "label": "同じ計画で再調査する",
                                    "action_type": "retry_research_job",
                                    "payload": {},
                                }
                            ]
                        },
                    ),
                ],
                pending_confirmation=False,
                pending_action=None,
            )
            self.db.update_research_job(
                job["id"],
                result_payload=response.model_dump(),
            )
            self.db.set_session_status(session_id, "research_failed")
            self.db.add_message(session_id, "assistant", response.model_dump())
        return True

    # JP: research stateを取得する。
    # EN: Get research state.
    def get_research_state(self, session_id: str) -> ResearchStateResponse:
        job = self.db.get_latest_research_job(session_id)
        if job is None:
            return ResearchStateResponse(session_id=session_id, status="idle")

        response_payload = job.get("result")
        response = None
        if response_payload:
            response = ChatMessageResponse(**response_payload)
        elif job["status"] in {"queued", "running"}:
            response = self._build_research_running_response(job)

        return ResearchStateResponse(
            session_id=session_id,
            job_id=job["id"],
            status=job["status"],
            current_stage=self._stage_label(job["current_stage"]),
            progress_percent=job["progress_percent"],
            latest_summary=job["latest_summary"],
            response=response,
        )

    # JP: process contract textを処理する。
    # EN: Process process contract text.
    def _process_contract_text(
        self,
        *,
        session_id: str,
        source_text: str,
    ) -> ChatMessageResponse:
        user_memory, task_memory, llm_config = self._ensure_session_llm_config(session_id)
        adapter = self._get_adapter_for_route(
            llm_config=llm_config,
            route_key="risk_check",
            session_id=session_id,
            interaction_type="risk_check",
        )

        risk_result = run_risk_check(source_text=source_text, adapter=adapter)
        self.db.add_audit_event(
            session_id,
            "risk_check",
            {"source_text": source_text},
            risk_result,
            "契約条項テキストをルール抽出",
        )

        task_memory["status"] = "risk_check_completed"
        task_memory["awaiting_contract_text"] = False
        task_memory["risk_items"] = risk_result["risk_items"]
        self.db.update_memories(session_id, user_memory, task_memory)
        self.db.set_session_status(session_id, "risk_check_completed")

        response = ChatMessageResponse(
            status="risk_check_completed",
            assistant_message="契約条項を確認しました。優先して確認すべきリスクを整理しました。",
            missing_slots=[],
            next_action="await_next_input",
            blocks=self._build_risk_blocks(risk_result),
            pending_confirmation=False,
            pending_action=None,
        )
        self.db.add_message(session_id, "assistant", response.model_dump())
        return response
