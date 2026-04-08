from __future__ import annotations

from typing import Any

from app.llm.base import LLMAdapter
from app.models import ChatMessageResponse, UIBlock

from .shared import RESEARCH_STAGE_ORDER, _generate_llm_plan_presentation


class OrchestratorPlanningMixin:
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

    def _stage_label(self, stage_name: str) -> str:
        for name, label in RESEARCH_STAGE_ORDER:
            if name == stage_name:
                return label
        return stage_name

    def _build_plan_conditions(
        self,
        user_memory: dict[str, Any],
        condition_reasons: dict[str, str] | None = None,
    ) -> list[dict[str, str]]:
        condition_reasons = condition_reasons or {}
        conditions: list[dict[str, str]] = []

        def add_condition(key: str, label: str, value: str) -> None:
            item = {"label": label, "value": value}
            reason = str(condition_reasons.get(key) or "").strip()
            if reason:
                item["reason"] = reason
            conditions.append(item)

        if user_memory.get("target_area"):
            add_condition("target_area", "希望エリア", str(user_memory["target_area"]))
        if user_memory.get("budget_max"):
            add_condition("budget_max", "家賃上限", self._format_money(user_memory["budget_max"]))
        if user_memory.get("layout_preference"):
            add_condition("layout_preference", "間取り", str(user_memory["layout_preference"]))
        if user_memory.get("station_walk_max"):
            add_condition("station_walk_max", "駅徒歩", self._format_walk(user_memory["station_walk_max"]))
        if user_memory.get("move_in_date"):
            add_condition("move_in_date", "入居時期", str(user_memory["move_in_date"]))
        if user_memory.get("must_conditions"):
            add_condition(
                "must_conditions",
                "必須条件",
                " / ".join(str(item) for item in user_memory["must_conditions"]),
            )
        if user_memory.get("nice_to_have"):
            add_condition(
                "nice_to_have",
                "あると良い条件",
                " / ".join(str(item) for item in user_memory["nice_to_have"]),
            )
        return conditions

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

    def _build_timeline_block(self, job: dict[str, Any] | None) -> UIBlock:
        return UIBlock(
            type="timeline",
            title="調査の進捗",
            content={
                "progress_percent": int(job.get("progress_percent", 0)) if job else 0,
                "current_stage": self._stage_label(str(job.get("current_stage") or "")) if job else "",
                "summary": str(job.get("latest_summary") or "") if job else "",
                "items": self._build_timeline_items(job),
            },
        )

    def _build_sources_block(self, source_items: list[dict[str, Any]]) -> UIBlock:
        return UIBlock(
            type="sources",
            title="参照ソース",
            content={"items": source_items},
        )

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
            blocks=[self._build_timeline_block(job)],
            pending_confirmation=False,
            pending_action=None,
        )
