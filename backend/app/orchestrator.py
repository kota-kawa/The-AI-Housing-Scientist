from __future__ import annotations

from typing import Any

from app.config import ProviderName, Settings, get_provider_model
from app.db import Database, utc_now_iso
from app.llm import create_adapter
from app.models import ChatMessageResponse, UIBlock
from app.profile_memory import (
    build_profile_resume_summary,
    merge_learned_preferences,
    summarize_memory_labels,
    update_profile_memory_with_reaction,
    update_profile_memory_with_search,
)
from app.services import BraveSearchClient, PropertyCatalogService
from app.stages import run_communication, run_ranking, run_risk_check, run_search_and_normalize
from app.stages.planner import detect_search_signal, run_planner
from app.stages.risk_check import looks_like_contract_text


class HousingOrchestrator:
    def __init__(self, settings: Settings, db: Database):
        self.settings = settings
        self.db = db
        self.catalog = PropertyCatalogService(db)
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

        for token in user_memory.get("must_conditions", []) or []:
            text = str(token).strip()
            if text:
                parts.append(text)

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

    def _format_money(self, value: Any) -> str:
        amount = int(value or 0)
        return f"{amount:,}円" if amount > 0 else "要確認"

    def _format_area(self, value: Any) -> str:
        area = float(value or 0)
        return f"{area:.1f}㎡" if area > 0 else "要確認"

    def _format_walk(self, value: Any) -> str:
        minutes = int(value or 0)
        return f"{minutes}分" if minutes > 0 else "要確認"

    def _get_property_reactions(self, task_memory: dict[str, Any]) -> dict[str, str]:
        reactions = task_memory.get("property_reactions", {}) or {}
        return {
            str(property_id): str(reaction)
            for property_id, reaction in reactions.items()
            if str(property_id).strip() and str(reaction).strip()
        }

    def _visible_ranked_properties(
        self,
        ranked_properties: list[dict[str, Any]],
        task_memory: dict[str, Any],
    ) -> list[dict[str, Any]]:
        excluded_ids = {
            property_id
            for property_id, reaction in self._get_property_reactions(task_memory).items()
            if reaction == "exclude"
        }
        return [
            item for item in ranked_properties
            if item["property_id_norm"] not in excluded_ids
        ]

    def _sync_profile_after_search(
        self,
        *,
        profile_id: str,
        user_memory: dict[str, Any],
        query: str,
    ) -> dict[str, Any]:
        profile = self.db.get_profile(profile_id)
        if profile is None:
            return user_memory

        updated_profile_memory = update_profile_memory_with_search(
            profile["profile_memory"],
            query=query,
            user_memory=user_memory,
            searched_at=utc_now_iso(),
        )
        merged_user_memory = merge_learned_preferences(
            {key: value for key, value in user_memory.items() if key != "learned_preferences"},
            updated_profile_memory.get("learned_preferences", {}) or {},
        )
        self.db.update_profile(profile_id, merged_user_memory, updated_profile_memory)
        return merged_user_memory

    def _sync_profile_after_reaction(
        self,
        *,
        profile_id: str,
        property_snapshot: dict[str, Any],
        reaction: str,
    ) -> dict[str, Any] | None:
        profile = self.db.get_profile(profile_id)
        if profile is None:
            return None

        updated_profile_memory = update_profile_memory_with_reaction(
            profile["profile_memory"],
            reaction=reaction,
            property_snapshot=property_snapshot,
            recorded_at=utc_now_iso(),
        )
        merged_user_memory = merge_learned_preferences(
            {key: value for key, value in profile["user_memory"].items() if key != "learned_preferences"},
            updated_profile_memory.get("learned_preferences", {}) or {},
        )
        self.db.update_profile(profile_id, merged_user_memory, updated_profile_memory)
        return merged_user_memory

    def build_session_initial_response(self, session_id: str) -> ChatMessageResponse | None:
        session = self.db.get_session(session_id)
        if session is None:
            return None

        profile = self.db.get_profile(session["profile_id"])
        if profile is None:
            return None

        last_user_memory = {
            key: value
            for key, value in profile["user_memory"].items()
            if key != "learned_preferences"
        }
        if not summarize_memory_labels(last_user_memory):
            return None

        profile_summary = build_profile_resume_summary(last_user_memory, profile["profile_memory"])
        summary_lines: list[str] = []
        if profile_summary["last_search_labels"]:
            summary_lines.append(
                f"前回の条件: {' / '.join(profile_summary['last_search_labels'][:5])}"
            )
        if profile_summary["frequent_area"]:
            summary_lines.append(f"よく検索するエリア: {profile_summary['frequent_area']}")
        if profile_summary["stable_preferences"]:
            summary_lines.append(
                f"変えない条件の候補: {' / '.join(profile_summary['stable_preferences'][:3])}"
            )
        if profile_summary["liked_features"]:
            summary_lines.append(
                f"気になる傾向: {' / '.join(profile_summary['liked_features'][:3])}"
            )

        _, task_memory = self.db.get_memories(session_id)
        task_memory["profile_resume_pending"] = True
        task_memory["profile_resume_summary"] = profile_summary
        self.db.update_memories(session_id, {}, task_memory)

        return ChatMessageResponse(
            status="awaiting_profile_resume",
            assistant_message="前回の条件を引き継ぎますか？",
            missing_slots=[],
            next_action="resume_or_reset_profile",
            blocks=[
                UIBlock(
                    type="text",
                    title="引き継ぎ候補",
                    content={"body": "\n".join(summary_lines)},
                ),
                UIBlock(
                    type="actions",
                    title="開始方法",
                    content={
                        "items": [
                            {
                                "label": "前回の条件を引き継ぐ",
                                "action_type": "resume_profile_memory",
                                "payload": {},
                            },
                            {
                                "label": "新しく探し始める",
                                "action_type": "dismiss_profile_resume",
                                "payload": {},
                            },
                        ]
                    },
                ),
            ],
            pending_confirmation=False,
            pending_action=None,
        )

    def _build_property_cards(
        self,
        *,
        ranked_properties: list[dict[str, Any]],
        normalized_properties: list[dict[str, Any]],
        selectable: bool,
        property_reactions: dict[str, str] | None = None,
        max_items: int | None = 3,
    ) -> list[dict[str, Any]]:
        by_id = {item["property_id_norm"]: item for item in normalized_properties}
        cards: list[dict[str, Any]] = []
        reactions = property_reactions or {}

        slice_items = ranked_properties if max_items is None else ranked_properties[:max_items]
        for item in slice_items:
            prop = by_id.get(item["property_id_norm"], {})
            reaction_state = reactions.get(item["property_id_norm"], "")
            card = {
                "id": item["property_id_norm"],
                "title": prop.get("building_name", "候補物件"),
                "score": item["score"],
                "rent": prop.get("rent", 0),
                "station_walk_min": prop.get("station_walk_min", 0),
                "station": prop.get("nearest_station", ""),
                "address": prop.get("address", ""),
                "layout": prop.get("layout", ""),
                "area": self._format_area(prop.get("area_m2", 0)),
                "why_selected": item["why_selected"],
                "why_not_selected": item["why_not_selected"],
                "feature_tags": prop.get("features", [])[:3],
                "reaction_state": reaction_state,
            }
            if selectable:
                card["action"] = {
                    "action_type": "generate_inquiry",
                    "label": "この物件の問い合わせ文を作成する",
                    "payload": {"property_id": item["property_id_norm"]},
                }
                card["secondary_actions"] = [
                    {
                        "action_type": "record_property_reaction",
                        "label": "気になる解除" if reaction_state == "favorite" else "気になる",
                        "payload": {
                            "property_id": item["property_id_norm"],
                            "reaction": "clear" if reaction_state == "favorite" else "favorite",
                        },
                    },
                    {
                        "action_type": "record_property_reaction",
                        "label": "除外解除" if reaction_state == "exclude" else "除外する",
                        "payload": {
                            "property_id": item["property_id_norm"],
                            "reaction": "clear" if reaction_state == "exclude" else "exclude",
                        },
                    },
                ]
            cards.append(card)
        return cards

    def _build_search_blocks(
        self,
        *,
        ranked_properties: list[dict[str, Any]],
        normalized_properties: list[dict[str, Any]],
        search_summary: dict[str, Any],
        property_reactions: dict[str, str] | None = None,
    ) -> list[UIBlock]:
        by_id = {item["property_id_norm"]: item for item in normalized_properties}
        reactions = property_reactions or {}

        rows = []
        for item in ranked_properties[:8]:
            prop = by_id.get(item["property_id_norm"], {})
            rows.append(
                {
                    "building_name": prop.get("building_name", "候補物件"),
                    "score": item["score"],
                    "rent": self._format_money(prop.get("rent")),
                    "layout": prop.get("layout", "要確認"),
                    "area_m2": self._format_area(prop.get("area_m2", 0)),
                    "station": prop.get("nearest_station", "要確認"),
                    "station_walk_min": self._format_walk(prop.get("station_walk_min", 0)),
                    "reaction": reactions.get(item["property_id_norm"], ""),
                }
            )

        summary_body = (
            f"比較対象 {search_summary.get('normalized_count', 0)}件 / "
            f"詳細ページ解析 {search_summary.get('detail_parsed_count', 0)}件 / "
            f"スニペット補完 {search_summary.get('fallback_count', 0)}件 / "
            f"重複候補 {search_summary.get('duplicate_group_count', 0)}グループ"
        )

        blocks: list[UIBlock] = [
            UIBlock(type="text", title="検索サマリー", content={"body": summary_body}),
        ]

        if ranked_properties:
            blocks.extend(
                [
                    UIBlock(
                        type="cards",
                        title="推薦候補",
                        content={
                            "items": self._build_property_cards(
                                ranked_properties=ranked_properties,
                                normalized_properties=normalized_properties,
                                selectable=True,
                                property_reactions=reactions,
                            )
                        },
                    ),
                    UIBlock(
                        type="table",
                        title="比較表",
                        content={
                            "columns": [
                                "building_name",
                                "score",
                                "rent",
                                "layout",
                                "area_m2",
                                "station",
                                "station_walk_min",
                                "reaction",
                            ],
                            "rows": rows,
                        },
                    ),
                ]
            )
        else:
            blocks.append(
                UIBlock(
                    type="warning",
                    title="候補なし",
                    content={"body": "詳細ページまで解析できた候補が見つかりませんでした。条件を少し広げて再検索してください。"},
                )
            )

        return blocks

    def _build_inquiry_blocks(
        self,
        *,
        ranked_properties: list[dict[str, Any]],
        normalized_properties: list[dict[str, Any]],
        communication: dict[str, Any],
        selected_property_id: str,
    ) -> list[UIBlock]:
        selected_ranked = [
            item for item in ranked_properties if item["property_id_norm"] == selected_property_id
        ]
        return [
            UIBlock(
                type="cards",
                title="選択中の物件",
                content={
                    "items": self._build_property_cards(
                        ranked_properties=selected_ranked or ranked_properties[:1],
                        normalized_properties=normalized_properties,
                        selectable=False,
                    )
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
                type="actions",
                title="次のアクション",
                content={
                    "items": [
                        {
                            "label": "契約書チェックへ進む",
                            "action_type": "start_contract_review",
                            "payload": {"property_id": selected_property_id},
                        }
                    ]
                },
            ),
        ]

    def _build_compare_blocks(
        self,
        *,
        property_ids: list[str],
        ranked_properties: list[dict[str, Any]],
        normalized_properties: list[dict[str, Any]],
        property_reactions: dict[str, str] | None = None,
    ) -> list[UIBlock]:
        if not property_ids:
            return [
                UIBlock(
                    type="warning",
                    title="比較対象なし",
                    content={"body": "比較したい物件を2件以上選んでください。"},
                )
            ]

        ranking_by_id = {item["property_id_norm"]: item for item in ranked_properties}
        normalized_by_id = {item["property_id_norm"]: item for item in normalized_properties}
        selected_ranked = [
            ranking_by_id[property_id]
            for property_id in property_ids
            if property_id in ranking_by_id and property_id in normalized_by_id
        ]
        selected_properties = [
            normalized_by_id[property_id]
            for property_id in property_ids
            if property_id in ranking_by_id and property_id in normalized_by_id
        ]

        if len(selected_ranked) < 2:
            return [
                UIBlock(
                    type="warning",
                    title="比較対象不足",
                    content={"body": "比較には2件以上の候補が必要です。"},
                )
            ]

        rows = []
        for prop in selected_properties:
            ranking = ranking_by_id[prop["property_id_norm"]]
            rows.append(
                {
                    "building_name": prop.get("building_name", "候補物件"),
                    "score": ranking.get("score", 0),
                    "rent": self._format_money(prop.get("rent")),
                    "layout": prop.get("layout", "要確認"),
                    "area_m2": self._format_area(prop.get("area_m2")),
                    "station": prop.get("nearest_station", "要確認"),
                    "station_walk_min": self._format_walk(prop.get("station_walk_min")),
                    "features": " / ".join(prop.get("features", [])[:3]) or "要確認",
                    "reaction": (property_reactions or {}).get(prop["property_id_norm"], ""),
                }
            )

        cheapest = min(selected_properties, key=lambda item: int(item.get("rent") or 10**9))
        shortest_walk = min(
            selected_properties,
            key=lambda item: int(item.get("station_walk_min") or 10**9),
        )
        top_score = max(selected_ranked, key=lambda item: float(item.get("score") or 0))
        top_property = normalized_by_id[top_score["property_id_norm"]]

        summary_lines = [
            f"総合バランス: {top_property.get('building_name', '候補物件')}",
            f"最安水準: {cheapest.get('building_name', '候補物件')}",
            f"駅近: {shortest_walk.get('building_name', '候補物件')}",
        ]

        return [
            UIBlock(
                type="cards",
                title="選択した比較候補",
                content={
                    "items": self._build_property_cards(
                        ranked_properties=selected_ranked,
                        normalized_properties=normalized_properties,
                        selectable=True,
                        property_reactions=property_reactions,
                        max_items=None,
                    )
                },
            ),
            UIBlock(
                type="table",
                title="選択物件の比較表",
                content={
                    "columns": [
                        "building_name",
                        "score",
                        "rent",
                        "layout",
                        "area_m2",
                        "station",
                        "station_walk_min",
                        "features",
                        "reaction",
                    ],
                    "rows": rows,
                },
            ),
            UIBlock(
                type="text",
                title="比較メモ",
                content={"body": "\n".join(summary_lines)},
            ),
        ]

    def _build_contract_prompt_blocks(self, property_name: str) -> list[UIBlock]:
        return [
            UIBlock(
                type="text",
                title="契約書チェックの入力",
                content={
                    "body": (
                        f"{property_name}の契約書・重要事項説明・初期費用表などの文面を貼り付けてください。\n"
                        "更新料、違約金、解約予告、保証会社条件を重点的に抽出します。"
                    )
                },
            )
        ]

    def _build_risk_blocks(self, risk_result: dict[str, Any]) -> list[UIBlock]:
        rows = [
            {
                "severity": item["severity"],
                "risk_type": item["risk_type"],
                "evidence": item["evidence"],
                "recommendation": item["recommendation"],
            }
            for item in risk_result["risk_items"]
        ]
        return [
            UIBlock(
                type="table",
                title="契約リスク一覧",
                content={
                    "columns": ["severity", "risk_type", "evidence", "recommendation"],
                    "rows": rows,
                },
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

    def _build_guidance_response(
        self,
        *,
        session_id: str,
        task_memory: dict[str, Any],
        message: str,
    ) -> ChatMessageResponse:
        assistant_text = (
            "検索条件の追加・物件選択・契約書チェックのいずれを進めるかを指定してください。"
        )
        if task_memory.get("last_ranked_properties"):
            assistant_text = (
                "直前の候補は保持しています。物件カードのボタンで問い合わせ文を作るか、"
                "新しい条件を送るか、契約条項テキストを貼り付けてください。"
            )

        response = ChatMessageResponse(
            status="awaiting_user_input",
            assistant_message=assistant_text,
            missing_slots=[],
            next_action="await_specific_input",
            blocks=[
                UIBlock(
                    type="warning",
                    title="入力ガイド",
                    content={"body": assistant_text},
                )
            ],
            pending_confirmation=False,
            pending_action=None,
        )
        self.db.add_message(session_id, "assistant", response.model_dump())
        self.db.add_audit_event(
            session_id,
            "guidance",
            {"message": message},
            response.model_dump(),
            "検索条件または契約条項の入力を促す",
        )
        return response

    def _collect_search_results(
        self,
        *,
        query: str,
        user_memory: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        catalog_results = self.catalog.search(query=query, user_memory=user_memory, count=8)
        brave_results: list[dict[str, Any]] = []
        brave_error = ""

        if self.settings.brave_search_api_key:
            try:
                brave_results = BraveSearchClient(
                    self.settings.brave_search_api_key,
                    timeout_seconds=self.settings.llm_timeout_seconds,
                ).search(query=query, count=6)
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

    def _process_search_message(
        self,
        *,
        session_id: str,
        message: str,
        adapter: Any,
        provider: ProviderName,
    ) -> ChatMessageResponse:
        user_memory, task_memory = self.db.get_memories(session_id)

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
            task_memory["awaiting_contract_text"] = False
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
        merged_search_results, source_summary = self._collect_search_results(
            query=query,
            user_memory=updated_user_memory,
        )

        search_result = run_search_and_normalize(
            query=query,
            search_results=merged_search_results,
            detail_fetcher=self.catalog.fetch_detail_html,
        )
        self.db.add_audit_event(
            session_id,
            "search_normalize",
            {"query": query, "search_sources": source_summary},
            search_result,
            "検索結果URLを個別フェッチし、詳細ページから共通スキーマへ正規化",
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

        session = self.db.get_session(session_id)
        profile_id = session["profile_id"] if session is not None else ""
        if profile_id:
            updated_user_memory = self._sync_profile_after_search(
                profile_id=profile_id,
                user_memory=updated_user_memory,
                query=query,
            )

        task_memory["status"] = "search_results_ready"
        task_memory["awaiting_contract_text"] = False
        task_memory["profile_resume_pending"] = False
        task_memory["last_query"] = query
        task_memory["last_normalized_properties"] = search_result["normalized_properties"]
        task_memory["last_ranked_properties"] = ranking_result["ranked_properties"]
        task_memory["last_duplicate_groups"] = search_result["duplicate_groups"]
        task_memory["last_search_summary"] = search_result["summary"] | source_summary
        task_memory["selected_property_id"] = None
        task_memory["risk_items"] = []
        task_memory["property_reactions"] = {}
        task_memory["comparison_property_ids"] = []
        task_memory["last_provider"] = provider

        self.db.set_pending_action(session_id, None)
        self.db.update_memories(session_id, updated_user_memory, task_memory)

        visible_ranked_properties = self._visible_ranked_properties(
            ranking_result["ranked_properties"],
            task_memory,
        )

        blocks = self._build_search_blocks(
            ranked_properties=visible_ranked_properties,
            normalized_properties=search_result["normalized_properties"],
            search_summary=search_result["summary"],
            property_reactions=self._get_property_reactions(task_memory),
        )
        if follow_up_questions:
            blocks.append(
                self._build_question_block(
                    questions=follow_up_questions,
                    optional=True,
                )
            )

        assistant_text = (
            f"候補を{len(visible_ranked_properties)}件比較しました。"
            "気になる物件を選ぶと、次に問い合わせ文を作成します。"
        )
        if follow_up_questions:
            assistant_text += "追加で分かる条件があれば、下の候補から追加入力できます。"

        response = ChatMessageResponse(
            status="search_results_ready",
            assistant_message=assistant_text,
            missing_slots=[],
            next_action="select_property",
            blocks=blocks,
            pending_confirmation=False,
            pending_action=None,
        )
        self.db.add_message(session_id, "assistant", response.model_dump())
        return response

    def _process_contract_text(
        self,
        *,
        session_id: str,
        source_text: str,
    ) -> ChatMessageResponse:
        user_memory, task_memory = self.db.get_memories(session_id)

        risk_result = run_risk_check(source_text=source_text)
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

    def _find_property_name(self, task_memory: dict[str, Any], property_id: str | None) -> str:
        if not property_id:
            return "選択中の物件"
        for item in task_memory.get("last_normalized_properties", []):
            if item.get("property_id_norm") == property_id:
                return str(item.get("building_name") or "選択中の物件")
        return "選択中の物件"

    def process_user_message(
        self,
        *,
        session_id: str,
        message: str,
        provider: ProviderName,
    ) -> ChatMessageResponse:
        _, task_memory = self.db.get_memories(session_id)

        search_signal = detect_search_signal(message)
        contract_like = looks_like_contract_text(message)

        if task_memory.get("awaiting_contract_text") and not search_signal:
            return self._process_contract_text(session_id=session_id, source_text=message)

        if contract_like and not search_signal:
            return self._process_contract_text(session_id=session_id, source_text=message)

        if not search_signal:
            return self._build_guidance_response(
                session_id=session_id,
                task_memory=task_memory,
                message=message,
            )

        adapter = self._get_adapter_or_none(provider)
        return self._process_search_message(
            session_id=session_id,
            message=message,
            adapter=adapter,
            provider=provider,
        )

    def execute_action(
        self,
        *,
        session_id: str,
        action_type: str,
        payload: dict[str, Any],
    ) -> ChatMessageResponse:
        session = self.db.get_session(session_id)
        if session is None:
            raise RuntimeError("session not found")

        user_memory, task_memory = self.db.get_memories(session_id)
        normalized_properties = task_memory.get("last_normalized_properties", [])
        ranked_properties = task_memory.get("last_ranked_properties", [])
        property_reactions = self._get_property_reactions(task_memory)

        if action_type == "resume_profile_memory":
            profile = self.db.get_profile(session["profile_id"])
            if profile is None:
                raise RuntimeError("profile not found")

            restored_user_memory = profile["user_memory"]
            task_memory["profile_resume_pending"] = False
            self.db.update_memories(session_id, restored_user_memory, task_memory)

            labels = summarize_memory_labels(restored_user_memory)
            message = (
                f"前回の条件を引き継ぎました。現在の条件は {' / '.join(labels[:5])} です。"
                if labels
                else "前回の条件を引き継ぎました。必要なら新しい条件を追加してください。"
            )
            response = ChatMessageResponse(
                status="awaiting_user_input",
                assistant_message=message,
                missing_slots=[],
                next_action="await_search_input",
                blocks=[UIBlock(type="text", title="引き継いだ条件", content={"body": message})],
                pending_confirmation=False,
                pending_action=None,
            )
            self.db.add_message(session_id, "assistant", response.model_dump())
            return response

        if action_type == "dismiss_profile_resume":
            task_memory["profile_resume_pending"] = False
            self.db.update_memories(session_id, {}, task_memory)

            message = "新しい条件で住まい探しを始めます。希望エリアや家賃条件を入力してください。"
            response = ChatMessageResponse(
                status="awaiting_user_input",
                assistant_message=message,
                missing_slots=[],
                next_action="await_search_input",
                blocks=[UIBlock(type="text", title="新しい検索を開始", content={"body": message})],
                pending_confirmation=False,
                pending_action=None,
            )
            self.db.add_message(session_id, "assistant", response.model_dump())
            return response

        if action_type == "compare_selected_properties":
            property_ids = [
                str(item).strip()
                for item in payload.get("property_ids", []) or []
                if str(item).strip()
            ]
            task_memory["comparison_property_ids"] = property_ids
            self.db.update_memories(session_id, user_memory, task_memory)

            response = ChatMessageResponse(
                status="search_results_ready",
                assistant_message=f"選択した{len(property_ids)}件を比較しました。",
                missing_slots=[],
                next_action="select_property",
                blocks=self._build_compare_blocks(
                    property_ids=property_ids,
                    ranked_properties=self._visible_ranked_properties(ranked_properties, task_memory),
                    normalized_properties=normalized_properties,
                    property_reactions=property_reactions,
                ),
                pending_confirmation=False,
                pending_action=None,
            )
            self.db.add_message(session_id, "assistant", response.model_dump())
            return response

        if action_type == "record_property_reaction":
            property_id = str(payload.get("property_id") or "").strip()
            reaction = str(payload.get("reaction") or "").strip()
            if not property_id or reaction not in {"favorite", "exclude", "clear"}:
                raise RuntimeError("property_id and valid reaction are required")

            property_snapshot = next(
                (
                    item
                    for item in normalized_properties
                    if item.get("property_id_norm") == property_id
                ),
                None,
            )
            if property_snapshot is None:
                raise RuntimeError("property not found")

            updated_reactions = dict(property_reactions)
            if reaction == "clear":
                updated_reactions.pop(property_id, None)
            else:
                updated_reactions[property_id] = reaction

            task_memory["property_reactions"] = updated_reactions
            if reaction == "exclude" and task_memory.get("selected_property_id") == property_id:
                task_memory["selected_property_id"] = None

            profile_user_memory = None
            if session["profile_id"] and reaction in {"favorite", "exclude"}:
                profile_user_memory = self._sync_profile_after_reaction(
                    profile_id=session["profile_id"],
                    property_snapshot=property_snapshot,
                    reaction=reaction,
                )
            if profile_user_memory is not None:
                user_memory = merge_learned_preferences(
                    {
                        key: value
                        for key, value in user_memory.items()
                        if key != "learned_preferences"
                    },
                    profile_user_memory.get("learned_preferences", {}) or {},
                )

            self.db.update_memories(session_id, user_memory, task_memory)

            visible_ranked_properties = self._visible_ranked_properties(ranked_properties, task_memory)
            reaction_label = {
                "favorite": "気になる",
                "exclude": "除外",
                "clear": "解除",
            }[reaction]
            property_name = self._find_property_name(task_memory, property_id)
            response = ChatMessageResponse(
                status="search_results_ready",
                assistant_message=f"{property_name}を「{reaction_label}」として記録しました。",
                missing_slots=[],
                next_action="select_property",
                blocks=self._build_search_blocks(
                    ranked_properties=visible_ranked_properties,
                    normalized_properties=normalized_properties,
                    search_summary=task_memory.get("last_search_summary", {}),
                    property_reactions=updated_reactions,
                ),
                pending_confirmation=False,
                pending_action=None,
            )
            self.db.add_message(session_id, "assistant", response.model_dump())
            return response

        if action_type == "generate_inquiry":
            property_id = str(payload.get("property_id") or "")
            if not property_id:
                raise RuntimeError("property_id is required")

            provider = task_memory.get("last_provider") or self.settings.llm_default_provider
            adapter = self._get_adapter_or_none(provider)

            communication_result = run_communication(
                ranked_properties=ranked_properties,
                normalized_properties=normalized_properties,
                user_memory=user_memory,
                selected_property_id=property_id,
                adapter=adapter,
            )
            self.db.add_audit_event(
                session_id,
                "communication",
                {"property_id": property_id},
                communication_result,
                "選択物件の問い合わせ文を生成",
            )

            task_memory["status"] = "inquiry_draft_ready"
            task_memory["awaiting_contract_text"] = False
            task_memory["selected_property_id"] = property_id

            pending_action = communication_result["pending_action"]
            self.db.set_pending_action(session_id, pending_action)
            self.db.update_memories(session_id, user_memory, task_memory)

            property_name = self._find_property_name(task_memory, property_id)
            response = ChatMessageResponse(
                status="inquiry_draft_ready",
                assistant_message=f"{property_name}の問い合わせ文を作成しました。必要ならそのまま契約書チェックにも進めます。",
                missing_slots=[],
                next_action="confirm_before_send",
                blocks=self._build_inquiry_blocks(
                    ranked_properties=self._visible_ranked_properties(ranked_properties, task_memory),
                    normalized_properties=normalized_properties,
                    communication=communication_result,
                    selected_property_id=property_id,
                ),
                pending_confirmation=pending_action is not None,
                pending_action=pending_action,
            )
            self.db.add_message(session_id, "assistant", response.model_dump())
            return response

        if action_type == "start_contract_review":
            property_id = str(payload.get("property_id") or task_memory.get("selected_property_id") or "")
            task_memory["status"] = "awaiting_contract_text"
            task_memory["awaiting_contract_text"] = True
            if property_id:
                task_memory["selected_property_id"] = property_id
            self.db.update_memories(session_id, user_memory, task_memory)

            property_name = self._find_property_name(task_memory, property_id)
            response = ChatMessageResponse(
                status="awaiting_contract_text",
                assistant_message="契約書チェックモードに切り替えました。文面を貼り付けてください。",
                missing_slots=[],
                next_action="paste_contract_text",
                blocks=self._build_contract_prompt_blocks(property_name),
                pending_confirmation=session.get("pending_action") is not None,
                pending_action=session.get("pending_action"),
            )
            self.db.add_message(session_id, "assistant", response.model_dump())
            return response

        raise RuntimeError(f"unsupported action_type: {action_type}")

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
