from __future__ import annotations

import time
from typing import Any

from app.config import ProviderName, Settings, get_provider_model
from app.db import Database, utc_now_iso
from app.llm import create_adapter
from app.models import ChatMessageResponse, ResearchStateResponse, UIBlock
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


RESEARCH_STAGE_ORDER = [
    ("plan_finalize", "計画確認"),
    ("query_expand", "クエリ展開"),
    ("retrieve", "情報収集"),
    ("enrich", "詳細補完"),
    ("normalize_dedupe", "正規化と重複統合"),
    ("rank", "推薦順位付け"),
    ("synthesize", "結果要約"),
]


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

    def _stage_label(self, stage_name: str) -> str:
        for name, label in RESEARCH_STAGE_ORDER:
            if name == stage_name:
                return label
        return stage_name

    def _build_plan_conditions(self, user_memory: dict[str, Any]) -> list[dict[str, str]]:
        conditions: list[dict[str, str]] = []
        if user_memory.get("target_area"):
            conditions.append({"label": "希望エリア", "value": str(user_memory["target_area"])})
        if user_memory.get("budget_max"):
            conditions.append(
                {
                    "label": "家賃上限",
                    "value": self._format_money(user_memory["budget_max"]),
                }
            )
        if user_memory.get("layout_preference"):
            conditions.append(
                {"label": "間取り", "value": str(user_memory["layout_preference"])}
            )
        if user_memory.get("station_walk_max"):
            conditions.append(
                {
                    "label": "駅徒歩",
                    "value": self._format_walk(user_memory["station_walk_max"]),
                }
            )
        if user_memory.get("move_in_date"):
            conditions.append({"label": "入居時期", "value": str(user_memory["move_in_date"])})
        if user_memory.get("must_conditions"):
            conditions.append(
                {
                    "label": "必須条件",
                    "value": " / ".join(str(item) for item in user_memory["must_conditions"]),
                }
            )
        if user_memory.get("nice_to_have"):
            conditions.append(
                {
                    "label": "あると良い条件",
                    "value": " / ".join(str(item) for item in user_memory["nice_to_have"]),
                }
            )
        return conditions

    def _build_research_plan(
        self,
        *,
        user_memory: dict[str, Any],
        planner_result: dict[str, Any],
        message: str,
        provider: ProviderName,
    ) -> dict[str, Any]:
        follow_up_questions = planner_result.get("follow_up_questions", [])
        base_query = self._build_search_query(user_memory, message)
        open_questions = [str(item.get("question") or "") for item in follow_up_questions if str(item.get("question") or "").strip()]
        strategy = [
            "希望条件を軸に複数クエリへ展開して候補を広めに収集します。",
            "詳細ページを優先して読み、表記ゆれと重複掲載を整理します。",
            "条件一致度と不足情報を比較し、問い合わせ向きの候補を上位化します。",
        ]
        summary_tokens = [item["value"] for item in self._build_plan_conditions(user_memory)[:4]]
        summary = " / ".join(summary_tokens) if summary_tokens else "条件整理から調査を開始"

        return {
            "summary": summary,
            "goal": "条件に近い候補を比較し、問い合わせに進める物件を絞り込む",
            "conditions": self._build_plan_conditions(user_memory),
            "strategy": strategy,
            "open_questions": open_questions,
            "search_query": base_query,
            "provider": provider,
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
                "conditions": plan.get("conditions", []),
                "strategy": plan.get("strategy", []),
                "open_questions": plan.get("open_questions", []),
                "search_query": plan.get("search_query", ""),
            },
        )

    def _build_timeline_items(self, job: dict[str, Any] | None) -> list[dict[str, str]]:
        if job is None:
            return []
        completed_nodes = {
            node["stage"]: node
            for node in self.db.list_research_journal_nodes(job["id"])
            if node["status"] == "completed"
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

    def _build_research_summary_body(
        self,
        *,
        ranked_properties: list[dict[str, Any]],
        normalized_properties: list[dict[str, Any]],
        source_items: list[dict[str, Any]],
    ) -> str:
        if not ranked_properties:
            return (
                "結論: 現時点では問い合わせ候補を十分に絞り込めませんでした。\n"
                "理由: 条件に合う詳細付き候補が不足しています。\n"
                "不確実な点: 家賃・駅距離・間取りなどの不足情報が残っています。\n"
                "次の一手: 条件を少し広げるか、優先順位を教えて再調査してください。"
            )

        top_ranked = ranked_properties[0]
        by_id = {item["property_id_norm"]: item for item in normalized_properties}
        top_property = by_id.get(top_ranked["property_id_norm"], {})
        uncertainty = []
        if not top_property.get("rent"):
            uncertainty.append("家賃情報の再確認が必要")
        if not top_property.get("station_walk_min"):
            uncertainty.append("駅徒歩情報の再確認が必要")
        if not top_property.get("layout"):
            uncertainty.append("間取り情報の再確認が必要")
        if not uncertainty:
            uncertainty.append("掲載条件の最新性は問い合わせで最終確認が必要")

        confirmation_items = []
        if source_items:
            confirmation_items.append("掲載元ごとの差分条件")
        if top_property.get("notes"):
            confirmation_items.append("募集条件の最新状況")
        confirmation_items.extend(
            [
                "初期費用の内訳",
                "短期解約違約金・更新料・解約予告",
            ]
        )

        return (
            f"結論: 第一候補は {top_property.get('building_name', '候補物件')} です。\n"
            f"理由: {top_ranked.get('why_selected') or '主要条件との整合が高い候補です。'}\n"
            f"懸念: {top_ranked.get('why_not_selected') or '大きな懸念は見当たりません。'}\n"
            f"不確実な点: {' / '.join(uncertainty[:3])}\n"
            f"問い合わせで確認したい点: {' / '.join(confirmation_items[:4])}"
        )

    def _build_research_result_blocks(
        self,
        *,
        ranked_properties: list[dict[str, Any]],
        normalized_properties: list[dict[str, Any]],
        search_summary: dict[str, Any],
        source_items: list[dict[str, Any]],
        task_memory: dict[str, Any],
        job_id: str | None,
    ) -> list[UIBlock]:
        blocks: list[UIBlock] = []
        job = self.db.get_research_job(job_id) if job_id else None
        if job is not None:
            blocks.append(self._build_timeline_block(job))

        blocks.append(
            UIBlock(
                type="text",
                title="調査サマリー",
                content={
                    "body": self._build_research_summary_body(
                        ranked_properties=ranked_properties,
                        normalized_properties=normalized_properties,
                        source_items=source_items,
                    )
                },
            )
        )

        blocks.extend(
            self._build_search_blocks(
                ranked_properties=ranked_properties,
                normalized_properties=normalized_properties,
                search_summary=search_summary,
                property_reactions=self._get_property_reactions(task_memory),
            )
        )

        if source_items:
            blocks.append(self._build_sources_block(source_items))

        return blocks

    def _build_research_queries(self, user_memory: dict[str, Any], seed_query: str) -> list[str]:
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

        candidates = [seed_query]
        if area:
            tokens = [area, "賃貸"]
            if layout:
                tokens.append(layout)
            if budget:
                tokens.append(f"{int(budget / 10000)}万円")
            candidates.append(" ".join(tokens))

        if area or layout:
            tokens = [token for token in [area, layout, "住みやすい", "賃貸"] if token]
            candidates.append(" ".join(tokens))

        if walk:
            tokens = [token for token in [area, "駅近", f"徒歩{walk}分", "賃貸"] if token]
            candidates.append(" ".join(tokens))

        if must_conditions:
            tokens = [token for token in [area, layout, " ".join(must_conditions[:2]), "賃貸"] if token]
            candidates.append(" ".join(tokens))

        if nice_to_have:
            tokens = [token for token in [area, " ".join(nice_to_have[:2]), "賃貸"] if token]
            candidates.append(" ".join(tokens))

        deduped: list[str] = []
        for item in candidates:
            text = " ".join(part for part in str(item).split() if part).strip()
            if text and text not in deduped:
                deduped.append(text)
        return deduped[:5]

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
        if task_memory.get("status") == "awaiting_plan_confirmation" and task_memory.get("draft_research_plan"):
            assistant_text = (
                "調査計画は作成済みです。承認ボタンで開始するか、条件を追加して計画を更新してください。"
            )
        elif task_memory.get("last_ranked_properties"):
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
        self.db.set_session_status(session_id, response.status)
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
            task_memory["status"] = "awaiting_user_input"
            task_memory["awaiting_contract_text"] = False
            task_memory["draft_research_plan"] = None
            task_memory["last_provider"] = provider
            self.db.update_memories(session_id, updated_user_memory, task_memory)
            self.db.set_pending_action(session_id, None)
            self.db.set_session_status(session_id, "awaiting_user_input")

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

        draft_plan = self._build_research_plan(
            user_memory=updated_user_memory,
            planner_result=planner_result,
            message=message,
            provider=provider,
        )

        task_memory["status"] = "awaiting_plan_confirmation"
        task_memory["awaiting_contract_text"] = False
        task_memory["profile_resume_pending"] = False
        task_memory["draft_research_plan"] = draft_plan
        task_memory["last_provider"] = provider
        self.db.set_pending_action(session_id, None)
        self.db.update_memories(session_id, updated_user_memory, task_memory)
        self.db.set_session_status(session_id, "awaiting_plan_confirmation")

        blocks = [self._build_plan_block(draft_plan)]
        if follow_up_questions:
            blocks.append(
                self._build_question_block(
                    questions=follow_up_questions,
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

        assistant_text = "調査計画を作成しました。内容を確認してから、明示承認で調査を開始します。"
        if follow_up_questions:
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

    def _run_research_stage(
        self,
        *,
        job_id: str,
        stage_name: str,
        progress_percent: int,
        latest_summary: str,
        input_payload: dict[str, Any],
        reasoning: str,
        runner: Any,
    ) -> dict[str, Any]:
        self.db.update_research_job(
            job_id,
            current_stage=stage_name,
            progress_percent=progress_percent,
            latest_summary=latest_summary,
        )
        started = time.perf_counter()
        try:
            output = runner()
        except Exception as exc:
            duration_ms = int((time.perf_counter() - started) * 1000)
            self.db.add_research_journal_node(
                job_id=job_id,
                stage=stage_name,
                node_type="stage",
                status="failed",
                input_payload=input_payload,
                output_payload={"error": str(exc)},
                reasoning=reasoning,
                duration_ms=duration_ms,
            )
            raise

        duration_ms = int((time.perf_counter() - started) * 1000)
        self.db.add_research_journal_node(
            job_id=job_id,
            stage=stage_name,
            node_type="stage",
            status="completed",
            input_payload=input_payload,
            output_payload=output,
            reasoning=reasoning,
            duration_ms=duration_ms,
        )
        return output

    def _retrieve_across_queries(
        self,
        *,
        job_id: str,
        queries: list[str],
        user_memory: dict[str, Any],
    ) -> dict[str, Any]:
        merged_by_url: dict[str, dict[str, Any]] = {}
        per_query: list[dict[str, Any]] = []
        catalog_total = 0
        brave_total = 0
        brave_errors: list[str] = []

        for index, query in enumerate(queries, start=1):
            self.db.update_research_job(
                job_id,
                current_stage="retrieve",
                progress_percent=25,
                latest_summary=f"{index}/{len(queries)}件目のクエリを収集中: {query}",
            )
            results, source_summary = self._collect_search_results(
                query=query,
                user_memory=user_memory,
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
                "query_count": len(queries),
                "unique_url_count": len(raw_results),
                "catalog_result_count": catalog_total,
                "brave_result_count": brave_total,
                "brave_error_count": len(brave_errors),
            },
            "per_query": per_query,
        }

    def _prefetch_detail_pages(
        self,
        *,
        job_id: str,
        raw_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        detail_html_map: dict[str, str] = {}
        total = len(raw_results)
        for index, item in enumerate(raw_results, start=1):
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            detail_html = self.catalog.fetch_detail_html(url)
            if detail_html:
                detail_html_map[url] = detail_html
            if total and (index == 1 or index == total or index % 3 == 0):
                self.db.update_research_job(
                    job_id,
                    current_stage="enrich",
                    progress_percent=45,
                    latest_summary=f"詳細ページを補完中: {index}/{total}件",
                )
        return {
            "detail_html_map": detail_html_map,
            "summary": {
                "detail_attempt_count": total,
                "detail_hit_count": len(detail_html_map),
                "summary": f"詳細ページを {len(detail_html_map)} 件取得",
            },
        }

    def _execute_research_job(self, job_id: str) -> dict[str, Any]:
        job = self.db.get_research_job(job_id)
        if job is None:
            raise RuntimeError("research job not found")

        session_id = job["session_id"]
        approved_plan = job["approved_plan"]
        user_memory, task_memory = self.db.get_memories(session_id)
        provider = str(job.get("provider") or task_memory.get("last_provider") or self.settings.llm_default_provider)
        adapter = self._get_adapter_or_none(provider) if provider in {"openai", "gemini", "groq", "claude"} else None

        plan_result = self._run_research_stage(
            job_id=job_id,
            stage_name="plan_finalize",
            progress_percent=10,
            latest_summary="承認済み計画を確認しています。",
            input_payload={"approved_plan": approved_plan},
            reasoning="ユーザー承認済みの計画を固定し、以降の調査に使う条件を確定する。",
            runner=lambda: {
                "summary": f"条件 {len(approved_plan.get('conditions', []))} 件で調査開始",
                "search_query": approved_plan.get("search_query", ""),
            },
        )

        queries_result = self._run_research_stage(
            job_id=job_id,
            stage_name="query_expand",
            progress_percent=18,
            latest_summary="複数の検索クエリに展開しています。",
            input_payload={"search_query": plan_result["search_query"]},
            reasoning="単発検索を避け、条件ごとの観点で収集漏れを減らす。",
            runner=lambda: {
                "queries": self._build_research_queries(
                    approved_plan.get("user_memory_snapshot", user_memory),
                    str(approved_plan.get("search_query") or ""),
                ),
                "summary": "検索クエリを展開",
            },
        )
        queries = queries_result["queries"]

        retrieve_result = self._run_research_stage(
            job_id=job_id,
            stage_name="retrieve",
            progress_percent=25,
            latest_summary="検索結果を収集しています。",
            input_payload={"queries": queries},
            reasoning="複数クエリの結果をまとめて収集し、URL単位で統合する。",
            runner=lambda: self._retrieve_across_queries(
                job_id=job_id,
                queries=queries,
                user_memory=approved_plan.get("user_memory_snapshot", user_memory),
            ),
        )
        raw_results = retrieve_result["raw_results"]

        enrich_result = self._run_research_stage(
            job_id=job_id,
            stage_name="enrich",
            progress_percent=45,
            latest_summary="候補の詳細ページを確認しています。",
            input_payload={"raw_result_count": len(raw_results)},
            reasoning="詳細ページを優先して読み、賃料や駅距離などの精度を上げる。",
            runner=lambda: self._prefetch_detail_pages(job_id=job_id, raw_results=raw_results),
        )
        detail_html_map = enrich_result["detail_html_map"]

        query = str(approved_plan.get("search_query") or self._build_search_query(user_memory, "賃貸"))
        def normalize_runner() -> dict[str, Any]:
            return run_search_and_normalize(
                query=query,
                search_results=raw_results,
                detail_fetcher=lambda url: detail_html_map.get(url),
            )

        search_result = self._run_research_stage(
            job_id=job_id,
            stage_name="normalize_dedupe",
            progress_percent=62,
            latest_summary="表記ゆれと重複掲載を整理しています。",
            input_payload={"query": query, "raw_result_count": len(raw_results)},
            reasoning="詳細ページとスニペットを共通スキーマへ揃え、重複候補を統合する。",
            runner=normalize_runner,
        )
        self.db.add_audit_event(
            session_id,
            "search_normalize",
            {"query": query, "query_expand": queries, "retrieve_summary": retrieve_result["summary"]},
            search_result,
            "複数検索結果を詳細優先で正規化し、重複候補を整理",
        )

        ranking_result = self._run_research_stage(
            job_id=job_id,
            stage_name="rank",
            progress_percent=78,
            latest_summary="候補の優先順位を評価しています。",
            input_payload={"normalized_property_count": len(search_result["normalized_properties"])},
            reasoning="条件一致度と不足情報を見て問い合わせ候補を順位付けする。",
            runner=lambda: run_ranking(
                normalized_properties=search_result["normalized_properties"],
                user_memory=approved_plan.get("user_memory_snapshot", user_memory),
            ),
        )
        self.db.add_audit_event(
            session_id,
            "ranking",
            {"user_memory": approved_plan.get("user_memory_snapshot", user_memory)},
            ranking_result,
            "条件スコアと取得情報の充実度から順位付け",
        )

        session = self.db.get_session(session_id)
        profile_id = session["profile_id"] if session is not None else ""
        updated_user_memory = approved_plan.get("user_memory_snapshot", user_memory)
        if profile_id:
            updated_user_memory = self._sync_profile_after_search(
                profile_id=profile_id,
                user_memory=updated_user_memory,
                query=query,
            )

        source_items = self._collect_research_source_items(
            ranked_properties=ranking_result["ranked_properties"],
            normalized_properties=search_result["normalized_properties"],
            raw_results=raw_results,
        )
        search_summary = search_result["summary"] | retrieve_result["summary"] | enrich_result["summary"]

        self._run_research_stage(
            job_id=job_id,
            stage_name="synthesize",
            progress_percent=92,
            latest_summary="結果を人向けに整理しています。",
            input_payload={"ranked_property_count": len(ranking_result["ranked_properties"])},
            reasoning="比較結果・不確実性・参照ソースをユーザー向けに整理する。",
            runner=lambda: {
                "summary": "結果要約を作成",
                "source_item_count": len(source_items),
            },
        )

        task_memory["status"] = "research_completed"
        task_memory["awaiting_contract_text"] = False
        task_memory["profile_resume_pending"] = False
        task_memory["last_query"] = query
        task_memory["last_normalized_properties"] = search_result["normalized_properties"]
        task_memory["last_ranked_properties"] = ranking_result["ranked_properties"]
        task_memory["last_duplicate_groups"] = search_result["duplicate_groups"]
        task_memory["last_search_summary"] = search_summary
        task_memory["last_source_items"] = source_items
        task_memory["selected_property_id"] = None
        task_memory["risk_items"] = []
        task_memory["property_reactions"] = {}
        task_memory["comparison_property_ids"] = []
        task_memory["approved_research_plan"] = approved_plan
        task_memory["draft_research_plan"] = approved_plan
        task_memory["active_research_job_id"] = None
        task_memory["last_research_job_id"] = job_id
        task_memory["last_provider"] = provider
        self.db.set_pending_action(session_id, None)
        self.db.update_memories(session_id, updated_user_memory, task_memory)

        self.db.update_research_job(
            job_id,
            status="completed",
            current_stage="synthesize",
            progress_percent=100,
            latest_summary="調査が完了しました。",
            finished_at=utc_now_iso(),
        )
        completed_job = self.db.get_research_job(job_id)
        visible_ranked_properties = self._visible_ranked_properties(
            ranking_result["ranked_properties"],
            task_memory,
        )
        response = ChatMessageResponse(
            status="research_completed",
            assistant_message=(
                f"調査が完了しました。{len(visible_ranked_properties)}件の候補を比較し、"
                "問い合わせに進める候補を整理しました。"
            ),
            missing_slots=[],
            next_action="select_property",
            blocks=self._build_research_result_blocks(
                ranked_properties=visible_ranked_properties,
                normalized_properties=search_result["normalized_properties"],
                search_summary=search_summary,
                source_items=source_items,
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
                blocks=[
                    self._build_timeline_block(failed_job),
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
        active_job_id = str(task_memory.get("active_research_job_id") or "").strip()
        active_job = self.db.get_research_job(active_job_id) if active_job_id else None

        if task_memory.get("awaiting_contract_text") and not search_signal:
            return self._process_contract_text(session_id=session_id, source_text=message)

        if contract_like and not search_signal:
            return self._process_contract_text(session_id=session_id, source_text=message)

        if active_job and active_job["status"] in {"queued", "running"}:
            response = self._build_research_running_response(active_job)
            self.db.add_message(session_id, "assistant", response.model_dump())
            self.db.set_session_status(session_id, response.status)
            return response

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
        latest_job_id = str(task_memory.get("last_research_job_id") or "")
        latest_job = self.db.get_research_job(latest_job_id) if latest_job_id else None

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
            self.db.set_session_status(session_id, "awaiting_user_input")
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
            self.db.set_session_status(session_id, "awaiting_user_input")
            self.db.add_message(session_id, "assistant", response.model_dump())
            return response

        if action_type == "approve_research_plan":
            active_job_id = str(task_memory.get("active_research_job_id") or "").strip()
            if active_job_id:
                active_job = self.db.get_research_job(active_job_id)
                if active_job is not None and active_job["status"] in {"queued", "running"}:
                    response = self._build_research_running_response(active_job)
                    self.db.set_session_status(session_id, response.status)
                    self.db.add_message(session_id, "assistant", response.model_dump())
                    return response

            draft_plan = task_memory.get("draft_research_plan")
            if not isinstance(draft_plan, dict) or not draft_plan:
                raise RuntimeError("承認できる調査計画がありません")

            provider = str(task_memory.get("last_provider") or self.settings.llm_default_provider)
            approved_plan = {**draft_plan, "approved_at": utc_now_iso()}
            job_id, _ = self.db.create_research_job(
                session_id=session_id,
                provider=provider,
                approved_plan=approved_plan,
            )
            task_memory["status"] = "research_queued"
            task_memory["approved_research_plan"] = approved_plan
            task_memory["active_research_job_id"] = job_id
            task_memory["last_research_job_id"] = job_id
            task_memory["awaiting_contract_text"] = False
            task_memory["selected_property_id"] = None
            task_memory["property_reactions"] = {}
            task_memory["comparison_property_ids"] = []
            self.db.set_pending_action(session_id, None)
            self.db.update_memories(session_id, user_memory, task_memory)
            job = self.db.get_research_job(job_id)
            if job is None:
                raise RuntimeError("research job creation failed")
            response = self._build_research_running_response(job)
            self.db.set_session_status(session_id, response.status)
            self.db.add_message(session_id, "assistant", response.model_dump())
            return response

        if action_type == "revise_research_plan":
            message = "条件を追加入力すると、計画を更新してから再度確認できます。"
            response = ChatMessageResponse(
                status="awaiting_user_input",
                assistant_message=message,
                missing_slots=[],
                next_action="await_search_input",
                blocks=[UIBlock(type="text", title="計画の更新", content={"body": message})],
                pending_confirmation=False,
                pending_action=None,
            )
            self.db.set_session_status(session_id, "awaiting_user_input")
            self.db.add_message(session_id, "assistant", response.model_dump())
            return response

        if action_type == "retry_research_job":
            approved_plan = task_memory.get("approved_research_plan") or (
                latest_job["approved_plan"] if latest_job is not None else None
            )
            if not isinstance(approved_plan, dict) or not approved_plan:
                raise RuntimeError("再実行できる調査計画がありません")

            provider = str(task_memory.get("last_provider") or self.settings.llm_default_provider)
            job_id, _ = self.db.create_research_job(
                session_id=session_id,
                provider=provider,
                approved_plan=approved_plan,
            )
            task_memory["status"] = "research_queued"
            task_memory["active_research_job_id"] = job_id
            task_memory["last_research_job_id"] = job_id
            self.db.update_memories(session_id, user_memory, task_memory)
            job = self.db.get_research_job(job_id)
            if job is None:
                raise RuntimeError("research job creation failed")
            response = self._build_research_running_response(job)
            self.db.set_session_status(session_id, response.status)
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
                status="research_completed",
                assistant_message=f"選択した{len(property_ids)}件を比較しました。",
                missing_slots=[],
                next_action="select_property",
                blocks=(
                    ([self._build_timeline_block(latest_job)] if latest_job else [])
                    + self._build_compare_blocks(
                        property_ids=property_ids,
                        ranked_properties=self._visible_ranked_properties(ranked_properties, task_memory),
                        normalized_properties=normalized_properties,
                        property_reactions=property_reactions,
                    )
                ),
                pending_confirmation=False,
                pending_action=None,
            )
            self.db.set_session_status(session_id, "research_completed")
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
                status="research_completed",
                assistant_message=f"{property_name}を「{reaction_label}」として記録しました。",
                missing_slots=[],
                next_action="select_property",
                blocks=self._build_research_result_blocks(
                    ranked_properties=visible_ranked_properties,
                    normalized_properties=normalized_properties,
                    search_summary=task_memory.get("last_search_summary", {}),
                    source_items=task_memory.get("last_source_items", []) or [],
                    task_memory=task_memory,
                    job_id=latest_job_id or None,
                ),
                pending_confirmation=False,
                pending_action=None,
            )
            self.db.set_session_status(session_id, "research_completed")
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
            self.db.set_session_status(session_id, "inquiry_draft_ready")
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
            self.db.set_session_status(session_id, "awaiting_contract_text")
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
        self.db.set_session_status(session_id, "completed")
        self.db.add_message(session_id, "assistant", response.model_dump())
        return response
