from __future__ import annotations

from typing import Any

from app.config import ProviderName, Settings, get_provider_model
from app.db import Database
from app.llm import create_adapter
from app.models import ChatMessageResponse, UIBlock
from app.services.brave_search import BraveSearchClient
from app.stages import (
    run_communication,
    run_planner,
    run_ranking,
    run_risk_check,
    run_search_and_normalize,
)


class HousingOrchestrator:
    def __init__(self, settings: Settings, db: Database):
        self.settings = settings
        self.db = db
        self._model_cache: dict[str, list[str]] = {}

    def _get_adapter_or_none(self, provider: ProviderName):
        key_map = {
            "openai": self.settings.openai_api_key,
            "gemini": self.settings.gemini_api_key,
            "groq": self.settings.groq_api_key,
            "claude": self.settings.claude_api_key,
        }
        if not key_map[provider]:
            if self.settings.model_strict_mode:
                raise RuntimeError(f"strict mode: API key missing for {provider}")
            return None

        adapter = create_adapter(self.settings, provider)

        if self.settings.model_strict_mode:
            if provider not in self._model_cache:
                self._model_cache[provider] = adapter.list_models()
            models = self._model_cache[provider]
            wanted = get_provider_model(self.settings, provider)
            if provider == "groq":
                wanted_secondary = self.settings.groq_model_secondary
                if wanted not in models or wanted_secondary not in models:
                    raise RuntimeError(
                        f"strict mode: groq model IDs not available ({wanted}, {wanted_secondary})"
                    )
            elif wanted not in models:
                raise RuntimeError(f"strict mode: model ID not available for {provider}: {wanted}")

        return adapter

    def _build_search_query(self, user_memory: dict[str, Any], fallback_message: str) -> str:
        parts = []
        if user_memory.get("target_area"):
            parts.append(str(user_memory["target_area"]))
        parts.append("賃貸")

        budget_max = user_memory.get("budget_max")
        if budget_max:
            parts.append(f"{int(budget_max / 10000)}万円")

        if user_memory.get("layout_preference"):
            parts.append(str(user_memory["layout_preference"]))

        if user_memory.get("station_walk_max"):
            parts.append(f"徒歩{user_memory['station_walk_max']}分")

        query = " ".join(parts).strip()
        return query if query and query != "賃貸" else fallback_message

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

    def _build_blocks(
        self,
        *,
        ranked_properties: list[dict[str, Any]],
        normalized_properties: list[dict[str, Any]],
        communication: dict[str, Any],
        risk_result: dict[str, Any],
    ) -> list[UIBlock]:
        by_id = {item["property_id_norm"]: item for item in normalized_properties}

        cards = []
        for item in ranked_properties[:3]:
            prop = by_id.get(item["property_id_norm"], {})
            cards.append(
                {
                    "id": item["property_id_norm"],
                    "title": prop.get("building_name_norm", "候補物件"),
                    "score": item["score"],
                    "rent": prop.get("rent", 0),
                    "station_walk_min": prop.get("station_walk_min", 0),
                    "why_selected": item["why_selected"],
                    "why_not_selected": item["why_not_selected"],
                }
            )

        rows = []
        for item in ranked_properties[:8]:
            prop = by_id.get(item["property_id_norm"], {})
            rows.append(
                {
                    "property_id": item["property_id_norm"],
                    "score": item["score"],
                    "rent": prop.get("rent", 0),
                    "layout": prop.get("layout", ""),
                    "area_m2": prop.get("area_m2", 0),
                    "station_walk_min": prop.get("station_walk_min", 0),
                }
            )

        return [
            UIBlock(type="cards", title="推薦候補", content={"items": cards}),
            UIBlock(
                type="table",
                title="比較表",
                content={
                    "columns": [
                        "property_id",
                        "score",
                        "rent",
                        "layout",
                        "area_m2",
                        "station_walk_min",
                    ],
                    "rows": rows,
                },
            ),
            UIBlock(
                type="text",
                title="問い合わせ下書き",
                content={"body": communication["message_draft"]},
            ),
            UIBlock(
                type="checklist",
                title="問い合わせ前チェック",
                content={"items": [{"label": x, "checked": False} for x in communication["check_items"]]},
            ),
            UIBlock(
                type="checklist",
                title="契約前の必須確認",
                content={"items": [{"label": x, "checked": False} for x in risk_result["must_confirm_list"]]},
            ),
            UIBlock(
                type="warning",
                title="免責",
                content={"body": "契約判断は最終的にユーザーおよび専門家確認の上で実施してください。"},
            ),
        ]

    def process_user_message(
        self,
        *,
        session_id: str,
        message: str,
        provider: ProviderName,
    ) -> ChatMessageResponse:
        user_memory, task_memory = self.db.get_memories(session_id)

        adapter = self._get_adapter_or_none(provider)

        planner_result = run_planner(message=message, user_memory=user_memory, adapter=adapter)
        self.db.add_audit_event(
            session_id,
            "planner",
            {"message": message, "user_memory": user_memory},
            planner_result,
            "抽出済み条件と不足スロットを生成",
        )

        updated_user_memory = planner_result["user_memory"]
        follow_up_questions = planner_result.get("follow_up_questions", [])
        if planner_result["missing_slots"]:
            assistant_text = "検索を始める前に、条件を少しだけ確認させてください。"
            task_memory["status"] = "awaiting_user_slots"
            self.db.update_memories(session_id, updated_user_memory, task_memory)
            self.db.set_pending_action(session_id, None)

            blocks = []
            if follow_up_questions:
                blocks.append(
                    self._build_question_block(
                        questions=follow_up_questions,
                        optional=False,
                    )
                )
            else:
                blocks.append(
                    UIBlock(
                        type="warning",
                        title="不足情報",
                        content={"body": assistant_text},
                    )
                )

            response = ChatMessageResponse(
                status="awaiting_user_input",
                assistant_message=assistant_text,
                missing_slots=planner_result["missing_slots"],
                next_action=planner_result["next_action"],
                blocks=blocks,
                pending_confirmation=False,
                pending_action=None,
            )
            self.db.add_message(session_id, "assistant", response.model_dump())
            return response

        query = self._build_search_query(updated_user_memory, message)
        search_results = BraveSearchClient(
            self.settings.brave_search_api_key,
            timeout_seconds=self.settings.llm_timeout_seconds,
        ).search(query=query, count=20)

        search_result = run_search_and_normalize(query=query, search_results=search_results)
        self.db.add_audit_event(
            session_id,
            "search_normalize",
            {"query": query},
            search_result,
            "Brave検索結果を共通スキーマへ正規化",
        )

        ranking_result = run_ranking(
            normalized_properties=search_result["normalized_properties"],
            user_memory=updated_user_memory,
        )
        self.db.add_audit_event(
            session_id,
            "ranking",
            {"user_memory": updated_user_memory},
            ranking_result,
            "条件スコアに基づいて順位付け",
        )

        communication_result = run_communication(
            ranked_properties=ranking_result["ranked_properties"],
            normalized_properties=search_result["normalized_properties"],
            user_memory=updated_user_memory,
        )
        self.db.add_audit_event(
            session_id,
            "communication",
            {"top": ranking_result["ranked_properties"][:1]},
            communication_result,
            "問い合わせ文の生成",
        )

        risk_result = run_risk_check(source_text=message)
        self.db.add_audit_event(
            session_id,
            "risk_check",
            {"source_text": message},
            risk_result,
            "契約系キーワードをルール抽出",
        )

        task_memory["status"] = "message_draft_ready"
        task_memory["last_query"] = query
        task_memory["last_ranked_property_ids"] = [
            item["property_id_norm"] for item in ranking_result["ranked_properties"][:10]
        ]
        task_memory["risk_items"] = risk_result["risk_items"]

        pending_action = communication_result["pending_action"]
        self.db.set_pending_action(session_id, pending_action)
        self.db.update_memories(session_id, updated_user_memory, task_memory)

        blocks = self._build_blocks(
            ranked_properties=ranking_result["ranked_properties"],
            normalized_properties=search_result["normalized_properties"],
            communication=communication_result,
            risk_result=risk_result,
        )
        if follow_up_questions:
            blocks.append(
                self._build_question_block(
                    questions=follow_up_questions,
                    optional=True,
                )
            )

        assistant_message = "条件に沿って候補を比較し、問い合わせ文と契約前チェック項目を作成しました。"
        if follow_up_questions:
            assistant_message += "追加で分かると精度が上がる条件も下にまとめています。"
        assistant_message += "送信前に確認操作を実行してください。"

        response = ChatMessageResponse(
            status="completed",
            assistant_message=assistant_message,
            missing_slots=[],
            next_action="confirm_before_send",
            blocks=blocks,
            pending_confirmation=pending_action is not None,
            pending_action=pending_action,
        )
        self.db.add_message(session_id, "assistant", response.model_dump())
        return response

    def confirm_action(
        self,
        *,
        session_id: str,
        action_type: str,
        approved: bool,
    ) -> ChatMessageResponse:
        session = self.db.get_session(session_id)
        if session is None:
            raise RuntimeError("session not found")

        pending_action = session.get("pending_action")
        if pending_action is None or pending_action.get("action_type") != action_type:
            raise RuntimeError("no matching pending action")

        user_memory, task_memory = self.db.get_memories(session_id)

        if approved:
            task_memory["status"] = "inquiry_marked_as_sent"
            task_memory["last_action"] = {"action_type": action_type, "approved": True}
            message = "確認済みとして処理しました。実送信は外部連携時に実行されます。"
        else:
            task_memory["status"] = "inquiry_cancelled"
            task_memory["last_action"] = {"action_type": action_type, "approved": False}
            message = "送信操作をキャンセルしました。内容を修正して再確認できます。"

        self.db.set_pending_action(session_id, None)
        self.db.update_memories(session_id, user_memory, task_memory)

        response = ChatMessageResponse(
            status="completed",
            assistant_message=message,
            missing_slots=[],
            next_action="await_next_input",
            blocks=[UIBlock(type="text", title="操作結果", content={"body": message})],
            pending_confirmation=False,
            pending_action=None,
        )
        self.db.add_message(session_id, "assistant", response.model_dump())
        return response
