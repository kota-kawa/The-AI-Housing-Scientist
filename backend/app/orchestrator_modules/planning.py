from __future__ import annotations

from typing import Any

from app.llm.base import LLMAdapter
from app.models import ChatMessageResponse, UIBlock
from app.stages.planner import (
    PLANNING_SLOT_EXAMPLES,
    PLANNING_SLOT_INPUT_KIND,
    PLANNING_SLOT_KEYBOARD_HINT,
    PLANNING_SLOT_LABELS,
    PLANNING_SLOT_PLACEHOLDERS,
    PLANNING_SLOT_QUESTIONS,
    QUESTION_SLOT_ORDER,
)

from .shared import RESEARCH_STAGE_ORDER, _generate_llm_plan_presentation


class OrchestratorPlanningMixin:
    # JP: textを正規化する。
    # EN: Normalize text.
    def _normalize_planning_text(self, value: Any) -> str:
        return " ".join(str(value or "").split()).strip()

    # JP: profile area候補を集める。
    # EN: Collect profile area suggestions.
    def _profile_area_examples(self, profile_memory: dict[str, Any] | None) -> list[str]:
        profile_memory = profile_memory or {}
        suggestions: list[str] = []
        for entry in list(profile_memory.get("search_history", []) or [])[-5:]:
            user_memory = entry.get("user_memory", {}) or {}
            area = self._normalize_planning_text(user_memory.get("target_area"))
            if area and area not in suggestions:
                suggestions.append(area)
            if len(suggestions) >= 3:
                break
        if suggestions:
            return suggestions
        return ["中野", "吉祥寺", "横浜駅周辺"]

    # JP: slot examplesを取得する。
    # EN: Get slot examples.
    def _planning_slot_examples(
        self,
        slot: str,
        *,
        profile_memory: dict[str, Any] | None = None,
    ) -> list[str]:
        if slot == "target_area":
            return self._profile_area_examples(profile_memory)
        return list(PLANNING_SLOT_EXAMPLES.get(slot, []))

    # JP: current answerを構築する。
    # EN: Build current answer display.
    def _planning_current_answer(self, slot: str, user_memory: dict[str, Any]) -> str:
        value = user_memory.get(slot)
        if isinstance(value, list):
            return " / ".join(str(item) for item in value if str(item).strip())
        if slot == "budget_max" and value:
            amount = int(value)
            return f"{int(amount / 10000)}万円まで" if amount % 10000 == 0 else f"{amount:,}円まで"
        if slot == "station_walk_max" and value:
            return f"徒歩{int(value)}分以内"
        return self._normalize_planning_text(value)

    # JP: planning questionsを構築する。
    # EN: Build planning questions.
    def _build_planning_questions(
        self,
        *,
        user_memory: dict[str, Any],
        slots: list[str],
        required: bool,
        profile_memory: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        questions: list[dict[str, Any]] = []
        for slot in QUESTION_SLOT_ORDER:
            if slot not in slots:
                continue
            current_value = self._planning_current_answer(slot, user_memory)
            examples = self._planning_slot_examples(slot, profile_memory=profile_memory)
            selected_example = current_value if current_value and current_value in examples else ""
            free_text = "" if selected_example else current_value
            questions.append(
                {
                    "slot": slot,
                    "label": PLANNING_SLOT_LABELS.get(slot, slot),
                    "question": PLANNING_SLOT_QUESTIONS.get(slot, ""),
                    "examples": examples,
                    "required": required,
                    "input_kind": PLANNING_SLOT_INPUT_KIND.get(slot, "text"),
                    "text_placeholder": PLANNING_SLOT_PLACEHOLDERS.get(slot, ""),
                    "keyboard_hint": PLANNING_SLOT_KEYBOARD_HINT.get(slot, "default"),
                    "selected_example": selected_example,
                    "free_text": free_text,
                }
            )
        return questions

    # JP: tree prune reason labelを処理する。
    # EN: Process tree prune reason label.
    def _tree_prune_reason_label(self, reason: str) -> str:
        normalized = str(reason or "").strip()
        if not normalized:
            return ""
        if normalized.startswith("repeated_issue:"):
            issue = normalized.split(":", 1)[1].strip()
            return f"同じ課題が続いたため剪定 ({issue})" if issue else "同じ課題が続いたため剪定"
        labels = {
            "low_branch_score": "評価スコアが閾値未満",
            "low_detail_coverage": "詳細ページ補完率が低い",
            "depth_limit": "探索深さの上限に到達",
            "duplicate_query_hash": "類似クエリのため統合",
        }
        return labels.get(normalized, normalized)

    # JP: tree termination labelを処理する。
    # EN: Process tree termination label.
    def _tree_termination_label(self, reason: str) -> str:
        labels = {
            "queued": "キューで待機中",
            "in_progress": "探索を継続中",
            "stable_high_readiness": "有望分岐が安定したため終了",
            "frontier_exhausted": "探索候補を使い切って終了",
            "node_budget_exhausted": "探索ノード上限に到達",
            "failed": "探索エラーで終了",
        }
        return labels.get(reason, reason or "進行中")

    # JP: tree node payloadを構築する。
    # EN: Build tree node payload.
    def _build_tree_node_payload(self, node: dict[str, Any]) -> dict[str, Any]:
        node_type = str(node.get("node_type") or "")
        input_payload = node.get("input") or {}
        output_payload = node.get("output") or {}
        metrics = node.get("metrics") or {}
        depth = (
            int(metrics.get("depth") or 0)
            if metrics.get("depth") is not None
            else int(input_payload.get("depth") or 0)
        )
        branch_id = str(node.get("branch_id") or "")
        status = str(node.get("status") or "")
        display_status = "pruned" if node_type == "search_pruned" else status

        if node_type == "search_root":
            label = "探索開始"
        elif node_type == "search_pruned":
            label = str(
                output_payload.get("label")
                or metrics.get("label")
                or input_payload.get("label")
                or branch_id
                or "剪定ノード"
            )
        else:
            label = str(
                metrics.get("label")
                or input_payload.get("label")
                or output_payload.get("label")
                or branch_id
                or "探索ノード"
            )

        summary = str(
            output_payload.get("summary") or metrics.get("summary") or node.get("reasoning") or ""
        )
        queries = [
            str(item).strip()
            for item in input_payload.get("queries", []) or []
            if str(item).strip()
        ]
        strategy_tags = [
            str(item).strip()
            for item in (metrics.get("strategy_tags") or input_payload.get("strategy_tags") or [])
            if str(item).strip()
        ]
        prune_reasons = [
            self._tree_prune_reason_label(str(item))
            for item in (output_payload.get("prune_reasons") or metrics.get("prune_reasons") or [])
            if self._tree_prune_reason_label(str(item))
        ]
        return {
            "id": int(node["id"]),
            "parent_id": node.get("parent_node_id"),
            "branch_id": branch_id,
            "kind": (
                "root"
                if node_type == "search_root"
                else "pruned"
                if node_type == "search_pruned"
                else "candidate"
            ),
            "status": display_status,
            "node_type": node_type,
            "label": label,
            "description": str(
                input_payload.get("description") or metrics.get("description") or ""
            ),
            "summary": summary,
            "depth": 0 if node_type == "search_root" else depth,
            "intent": str(
                node.get("intent")
                or metrics.get("intent")
                or input_payload.get("intent")
                or "draft"
            ),
            "is_failed": bool(node.get("is_failed") or metrics.get("is_failed")),
            "debug_depth": int(
                node.get("debug_depth")
                or metrics.get("debug_depth")
                or input_payload.get("debug_depth")
                or 0
            ),
            "query_count": int(metrics.get("query_count") or len(queries)),
            "queries": queries[:3],
            "strategy_tags": strategy_tags[:4],
            "branch_score": metrics.get("branch_score"),
            "frontier_score": metrics.get("frontier_score") or output_payload.get("frontier_score"),
            "detail_coverage": metrics.get("detail_coverage"),
            "structured_ratio": metrics.get("structured_ratio"),
            "selected": bool(node.get("selected")),
            "prune_reasons": prune_reasons,
            "created_at": str(node.get("created_at") or ""),
        }

    # JP: tree blockを構築する。
    # EN: Build tree block.
    def _build_tree_block(
        self,
        job: dict[str, Any] | None,
        *,
        task_memory: dict[str, Any] | None = None,
    ) -> UIBlock:
        tree_nodes_raw = (
            [
                node
                for node in self.db.list_research_journal_nodes(job["id"])
                if node.get("stage") == "tree_search" and node.get("node_type") != "stage"
            ]
            if job is not None
            else []
        )
        selection_node = next(
            (
                node
                for node in reversed(tree_nodes_raw)
                if node.get("node_type") == "search_selection" and node.get("selected")
            ),
            None,
        )

        selected_branch_id = str((task_memory or {}).get("selected_branch_id") or "")
        if not selected_branch_id and selection_node is not None:
            selected_branch_id = str(selection_node.get("branch_id") or "")

        selected_path_source = list((task_memory or {}).get("selected_path") or [])
        if not selected_path_source and selection_node is not None:
            selected_path_source = list(selection_node.get("output", {}).get("selected_path") or [])
        selected_path_branch_ids: list[str] = []
        for item in selected_path_source:
            branch_id = str(item.get("branch_id") or "").strip()
            if branch_id and branch_id not in selected_path_branch_ids:
                selected_path_branch_ids.append(branch_id)

        tree_nodes = [
            self._build_tree_node_payload(node)
            for node in tree_nodes_raw
            if node.get("node_type") != "search_selection"
        ]
        nodes_by_id = {int(node["id"]): node for node in tree_nodes}
        for node in tree_nodes:
            parent = (
                nodes_by_id.get(int(node["parent_id"]))
                if node.get("parent_id") is not None
                else None
            )
            node["parent_label"] = str(parent.get("label") or "") if parent else ""
            branch_id = str(node.get("branch_id") or "")
            node["is_selected"] = bool(branch_id and branch_id == selected_branch_id)
            node["is_on_selected_path"] = bool(branch_id and branch_id in selected_path_branch_ids)

        completed_candidates = [
            node
            for node in tree_nodes
            if node.get("kind") == "candidate" and node.get("status") == "completed"
        ]

        selected_branch = {}
        for item in (task_memory or {}).get("branch_summaries", []) or []:
            if str(item.get("branch_id") or "") == selected_branch_id:
                selected_branch = item
                break
        if not selected_branch and selection_node is not None:
            selected_branch = dict(selection_node.get("output", {}).get("selected_branch") or {})

        if selected_branch:
            focus_kind = "selected"
            focus_branch = {
                "branch_id": str(selected_branch.get("branch_id") or ""),
                "label": str(selected_branch.get("label") or ""),
                "depth": int(selected_branch.get("depth") or 0),
                "branch_score": selected_branch.get("branch_score"),
                "detail_coverage": selected_branch.get("detail_coverage"),
                "frontier_score": selected_branch.get("frontier_score"),
                "summary": str(selected_branch.get("summary") or ""),
            }
        elif completed_candidates:
            leading = max(
                completed_candidates,
                key=lambda item: float(item.get("branch_score") or 0.0),
            )
            focus_kind = "leading"
            focus_branch = {
                "branch_id": str(leading.get("branch_id") or ""),
                "label": str(leading.get("label") or ""),
                "depth": int(leading.get("depth") or 0),
                "branch_score": leading.get("branch_score"),
                "detail_coverage": leading.get("detail_coverage"),
                "frontier_score": leading.get("frontier_score"),
                "summary": str(leading.get("summary") or ""),
            }
        else:
            focus_kind = "queued"
            focus_branch = {}

        computed_stats = {
            "executed_node_count": len(completed_candidates),
            "failed_node_count": len(
                [
                    node
                    for node in tree_nodes
                    if node.get("kind") == "candidate" and node.get("status") == "failed"
                ]
            ),
            "pruned_node_count": len([node for node in tree_nodes if node.get("kind") == "pruned"]),
            "frontier_remaining": len(
                [
                    node
                    for node in tree_nodes
                    if node.get("kind") == "candidate" and node.get("status") == "queued"
                ]
            ),
            "running_node_count": len(
                [
                    node
                    for node in tree_nodes
                    if node.get("kind") == "candidate" and node.get("status") == "running"
                ]
            ),
            "max_depth_reached": max(
                [int(node.get("depth") or 0) for node in tree_nodes], default=0
            ),
        }
        summary_source = dict((task_memory or {}).get("search_tree_summary") or {})
        termination_reason = str(summary_source.get("termination_reason") or "")
        if not termination_reason:
            if job is None or job.get("status") == "queued":
                termination_reason = "queued"
            elif job.get("status") == "running":
                termination_reason = "in_progress"
            elif job.get("status") == "failed":
                termination_reason = "failed"
            else:
                termination_reason = "frontier_exhausted"

        stats = {
            "executed_node_count": int(
                summary_source.get("executed_node_count") or computed_stats["executed_node_count"]
            ),
            "failed_node_count": int(
                summary_source.get("failed_node_count") or computed_stats["failed_node_count"]
            ),
            "pruned_node_count": int(
                summary_source.get("pruned_node_count") or computed_stats["pruned_node_count"]
            ),
            "frontier_remaining": int(
                summary_source.get("frontier_remaining") or computed_stats["frontier_remaining"]
            ),
            "running_node_count": computed_stats["running_node_count"],
            "max_depth_reached": int(
                summary_source.get("max_depth_reached") or computed_stats["max_depth_reached"]
            ),
            "termination_reason": termination_reason,
            "termination_label": self._tree_termination_label(termination_reason),
            "node_count": len(tree_nodes),
        }

        return UIBlock(
            type="tree",
            title="探索ツリー",
            content={
                "current_stage": self._stage_label(str(job.get("current_stage") or ""))
                if job
                else "",
                "summary": str(job.get("latest_summary") or "") if job else "",
                "is_live": bool(job and job.get("status") in {"queued", "running"}),
                "selected_branch_id": selected_branch_id,
                "selected_path_branch_ids": selected_path_branch_ids,
                "focus_kind": focus_kind,
                "focus_branch": focus_branch,
                "stats": stats,
                "nodes": tree_nodes,
            },
        )

    # JP: question blockを構築する。
    # EN: Build question block.
    def _build_question_block(
        self,
        *,
        questions: list[dict[str, Any]],
        optional: bool,
    ) -> UIBlock:
        intro = (
            "検索精度を上げるため、分かるものだけ追加入力してください。"
            if optional
            else "検索を始めるため、まずは次の条件を教えてください。"
        )
        title = "追加で確認したい条件" if optional else "検索前に確認したい条件"
        return UIBlock(
            type="question",
            title=title,
            content={
                "mode": "optional" if optional else "blocking",
                "intro": intro,
                "items": questions,
            },
        )

    # JP: stage labelを処理する。
    # EN: Process stage label.
    def _stage_label(self, stage_name: str) -> str:
        for name, label in RESEARCH_STAGE_ORDER:
            if name == stage_name:
                return label
        return stage_name

    # JP: plan conditionsを構築する。
    # EN: Build plan conditions.
    def _build_plan_conditions(
        self,
        user_memory: dict[str, Any],
        condition_reasons: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        condition_reasons = condition_reasons or {}
        conditions: list[dict[str, Any]] = []

        # JP: add conditionを処理する。
        # EN: Process add condition.
        def add_condition(key: str, label: str, value: str, *, priority: str) -> None:
            item = {"label": label, "value": value, "priority": priority, "changed": False}
            reason = str(condition_reasons.get(key) or "").strip()
            if reason:
                item["reason"] = reason
            conditions.append(item)

        if user_memory.get("listing_type"):
            add_condition(
                "listing_type",
                "物件種別",
                str(user_memory["listing_type"]),
                priority="required",
            )
        if user_memory.get("target_area"):
            add_condition(
                "target_area", "希望エリア", str(user_memory["target_area"]), priority="required"
            )
        if user_memory.get("budget_max"):
            add_condition(
                "budget_max",
                "家賃上限",
                self._format_money(user_memory["budget_max"]),
                priority="required",
            )
        if user_memory.get("layout_preference"):
            add_condition(
                "layout_preference",
                "間取り",
                str(user_memory["layout_preference"]),
                priority="required",
            )
        if user_memory.get("station_walk_max"):
            add_condition(
                "station_walk_max",
                "駅徒歩",
                self._format_walk(user_memory["station_walk_max"]),
                priority="preferred",
            )
        if user_memory.get("move_in_date"):
            add_condition(
                "move_in_date", "入居時期", str(user_memory["move_in_date"]), priority="context"
            )
        if user_memory.get("must_conditions"):
            add_condition(
                "must_conditions",
                "必須条件",
                " / ".join(str(item) for item in user_memory["must_conditions"]),
                priority="required",
            )
        if user_memory.get("nice_to_have"):
            add_condition(
                "nice_to_have",
                "あると良い条件",
                " / ".join(str(item) for item in user_memory["nice_to_have"]),
                priority="preferred",
            )
        return conditions

    # JP: research planを構築する。
    # EN: Build research plan.
    def _build_research_plan(
        self,
        *,
        user_memory: dict[str, Any],
        planner_result: dict[str, Any],
        message: str,
        adapter: LLMAdapter | None,
        llm_config: dict[str, Any],
    ) -> dict[str, Any]:
        follow_up_questions = planner_result.get("follow_up_questions", [])
        condition_reasons = planner_result.get("condition_reasons", {}) or {}
        conditions = self._build_plan_conditions(user_memory, condition_reasons)
        planner_plan = planner_result.get("research_plan", {}) or {}
        seed_queries = [
            " ".join(str(item).split()).strip()
            for item in planner_result.get("seed_queries", []) or []
            if " ".join(str(item).split()).strip()
        ]
        open_questions = [
            str(item.get("question") or "")
            for item in follow_up_questions
            if str(item.get("question") or "").strip()
        ]
        strategy = [
            str(item).strip()
            for item in planner_plan.get("strategy", []) or []
            if str(item).strip()
        ]
        if not strategy:
            strategy = [
                "希望条件を軸に複数クエリへ展開して候補を広めに収集します。",
                "詳細ページを優先して読み、表記ゆれと重複掲載を整理します。",
                "条件一致度と不足情報を比較し、問い合わせ向きの候補を上位化します。",
            ]
        summary_tokens = [item["value"] for item in conditions[:4]]
        summary = " / ".join(summary_tokens) if summary_tokens else "条件整理から調査を開始"
        llm_plan = _generate_llm_plan_presentation(
            user_message=message,
            conditions=conditions,
            follow_up_questions=follow_up_questions,
            seed_queries=seed_queries,
            planner_plan=planner_plan,
            adapter=adapter,
        )

        return {
            "assistant_message": str((llm_plan or {}).get("assistant_message") or "").strip(),
            "summary": str((llm_plan or {}).get("summary") or "").strip()
            or str(planner_plan.get("summary") or "").strip()
            or summary,
            "goal": str((llm_plan or {}).get("goal") or "").strip()
            or str(planner_plan.get("goal") or "").strip()
            or "条件に近い候補を比較し、問い合わせに進める物件を絞り込む",
            "rationale": str((llm_plan or {}).get("rationale") or "").strip()
            or str(planner_plan.get("rationale") or "").strip(),
            "conditions": conditions,
            "strategy": list((llm_plan or {}).get("strategy") or strategy),
            "open_questions": list((llm_plan or {}).get("open_questions") or open_questions),
            "seed_queries": seed_queries,
            "search_query": seed_queries[0] if seed_queries else "",
            "llm_config_snapshot": llm_config,
            "created_from_message": message,
            "user_memory_snapshot": user_memory,
        }

    # JP: plan blockを構築する。
    # EN: Build plan block.
    def _build_plan_block(self, plan: dict[str, Any]) -> UIBlock:
        return UIBlock(
            type="plan",
            title="今回の調査計画",
            content={
                "summary": plan.get("summary", ""),
                "goal": plan.get("goal", ""),
                "rationale": plan.get("rationale", ""),
                "conditions": plan.get("conditions", []),
                "strategy": plan.get("strategy", []),
                "open_questions": plan.get("open_questions", []),
                "seed_queries": plan.get("seed_queries", []),
                "search_query": plan.get("search_query", ""),
            },
        )

    # JP: timeline itemsを構築する。
    # EN: Build timeline items.
    def _build_timeline_items(self, job: dict[str, Any] | None) -> list[dict[str, str]]:
        if job is None:
            return []
        completed_nodes = {
            node["stage"]: node
            for node in self.db.list_research_journal_nodes(job["id"])
            if node["status"] == "completed" and node["node_type"] == "stage"
        }
        items: list[dict[str, str]] = []
        for stage_name, label in RESEARCH_STAGE_ORDER:
            status = "pending"
            detail = ""
            if stage_name in completed_nodes:
                status = "completed"
                detail = str(
                    completed_nodes[stage_name]["output"].get("summary")
                    or completed_nodes[stage_name]["reasoning"]
                )
            elif job["status"] == "failed" and stage_name == job.get("current_stage"):
                status = "failed"
                detail = str(job.get("error_message") or "処理に失敗しました。")
            elif job["status"] in {"queued", "running"} and stage_name == job.get("current_stage"):
                status = "running"
                detail = str(job.get("latest_summary") or "")
            elif job["status"] == "completed" and stage_name == "synthesize":
                status = "completed"
                detail = str(job.get("latest_summary") or "")
            items.append({"label": label, "status": status, "detail": detail})
        return items

    # JP: timeline blockを構築する。
    # EN: Build timeline block.
    def _build_timeline_block(self, job: dict[str, Any] | None) -> UIBlock:
        return UIBlock(
            type="timeline",
            title="調査の進捗",
            content={
                "progress_percent": int(job.get("progress_percent", 0)) if job else 0,
                "current_stage": self._stage_label(str(job.get("current_stage") or ""))
                if job
                else "",
                "summary": str(job.get("latest_summary") or "") if job else "",
                "items": self._build_timeline_items(job),
            },
        )

    # JP: research progress blocksを構築する。
    # EN: Build research progress blocks.
    def _build_research_progress_blocks(
        self,
        job: dict[str, Any] | None,
        *,
        task_memory: dict[str, Any] | None = None,
    ) -> list[UIBlock]:
        if job is None:
            return []
        return [
            self._build_timeline_block(job),
            self._build_tree_block(job, task_memory=task_memory),
        ]

    # JP: sources blockを構築する。
    # EN: Build sources block.
    def _build_sources_block(self, source_items: list[dict[str, Any]]) -> UIBlock:
        return UIBlock(
            type="sources",
            title="参照ソース",
            content={"items": source_items},
        )

    # JP: research running responseを構築する。
    # EN: Build research running response.
    def _build_research_running_response(self, job: dict[str, Any]) -> ChatMessageResponse:
        status = "research_running" if job["status"] == "running" else "research_queued"
        assistant_message = (
            "調査計画に沿って候補を収集中です。進捗はこのまま更新されます。"
            if job["status"] == "running"
            else "調査ジョブを登録しました。まもなく情報収集を始めます。"
        )
        return ChatMessageResponse(
            status=status,
            assistant_message=assistant_message,
            missing_slots=[],
            next_action="await_research_completion",
            blocks=self._build_research_progress_blocks(job),
            pending_confirmation=False,
            pending_action=None,
        )
