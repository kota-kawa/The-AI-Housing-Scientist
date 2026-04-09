from __future__ import annotations

from typing import Any

from app.models import ChatMessageResponse, UIBlock

from .shared import _generate_llm_guidance_message


class OrchestratorPresentationMixin:
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
        research_summary: str,
        ranked_properties: list[dict[str, Any]],
        normalized_properties: list[dict[str, Any]],
        source_items: list[dict[str, Any]],
    ) -> str:
        if str(research_summary).strip():
            return str(research_summary).strip()

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
        research_summary: str,
        final_report_markdown: str = "",
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
            blocks.extend(self._build_research_progress_blocks(job, task_memory=task_memory))

        blocks.append(
            UIBlock(
                type="text",
                title="調査サマリー",
                content={
                    "body": self._build_research_summary_body(
                        research_summary=research_summary,
                        ranked_properties=ranked_properties,
                        normalized_properties=normalized_properties,
                        source_items=source_items,
                    )
                },
            )
        )

        if str(final_report_markdown).strip():
            blocks.append(
                UIBlock(
                    type="text",
                    title="最終レポート",
                    content={"body": str(final_report_markdown).strip()},
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

        branch_summaries = task_memory.get("branch_summaries") or []
        selected_branch_id = str(task_memory.get("selected_branch_id") or "")
        if branch_summaries:
            rows = []
            for item in sorted(
                branch_summaries,
                key=lambda summary: float(summary.get("branch_score") or 0.0),
                reverse=True,
            ):
                label = str(item.get("label") or item.get("branch_id") or "branch")
                if str(item.get("branch_id") or "") == selected_branch_id:
                    label = f"{label} (selected)"
                rows.append(
                    {
                        "branch": label,
                        "depth": int(item.get("depth") or 0),
                        "score": float(item.get("branch_score") or 0.0),
                        "frontier_score": float(item.get("frontier_score") or 0.0),
                        "query_count": int(item.get("query_count") or 0),
                        "normalized_count": int(item.get("normalized_count") or 0),
                        "detail_coverage": f"{round(float(item.get('detail_coverage') or 0.0) * 100)}%",
                        "summary": str(item.get("summary") or ""),
                    }
                )
            blocks.append(
                UIBlock(
                    type="table",
                    title="探索分岐の比較",
                    content={
                        "columns": [
                            "branch",
                            "depth",
                            "score",
                            "frontier_score",
                            "query_count",
                            "normalized_count",
                            "detail_coverage",
                            "summary",
                        ],
                        "rows": rows,
                    },
                )
            )

        selected_path = task_memory.get("selected_path") or []
        if selected_path:
            path_lines = []
            for index, item in enumerate(selected_path, start=1):
                label = str(item.get("label") or item.get("branch_id") or f"node-{index}")
                tags = [
                    str(tag).strip()
                    for tag in item.get("strategy_tags", []) or []
                    if str(tag).strip()
                ]
                extras = [f"depth {int(item.get('depth') or 0)}"]
                if tags:
                    extras.append("/".join(tags[:3]))
                score = float(item.get("branch_score") or 0.0)
                extras.append(f"score {round(score, 2)}")
                path_lines.append(f"{index}. {label} ({', '.join(extras)})")
            blocks.append(
                UIBlock(
                    type="text",
                    title="選択された探索パス",
                    content={"body": "\n".join(path_lines)},
                )
            )

        if source_items:
            blocks.append(self._build_sources_block(source_items))

        offline_evaluation = task_memory.get("offline_evaluation") or {}
        search_tree_summary = task_memory.get("search_tree_summary") or {}
        if offline_evaluation:
            blocks.append(
                UIBlock(
                    type="text",
                    title="オフライン評価",
                    content={
                        "body": (
                            f"readiness: {offline_evaluation.get('readiness', 'unknown')}\n"
                            f"候補数: {offline_evaluation.get('visible_candidate_count', 0)}\n"
                            f"詳細補完率: {round(float(offline_evaluation.get('detail_coverage', 0.0)) * 100)}%\n"
                            f"構造化率: {round(float(offline_evaluation.get('structured_ratio', 0.0)) * 100)}%\n"
                            f"探索終了理由: {search_tree_summary.get('termination_reason', 'unknown')}\n"
                            f"探索ノード数: {search_tree_summary.get('executed_node_count', 0)}\n"
                            f"次の改善候補: {' / '.join(offline_evaluation.get('recommendations', [])[:3])}"
                        )
                    },
                )
            )

        failure_summary = task_memory.get("failure_summary") or {}
        if failure_summary and failure_summary.get("top_issues"):
            blocks.append(
                UIBlock(
                    type="warning",
                    title="探索の改善余地",
                    content={
                        "body": (
                            f"{failure_summary.get('summary', '')}\n"
                            f"主な課題: {' / '.join(failure_summary.get('top_issues', [])[:3])}\n"
                            f"改善候補: {' / '.join(failure_summary.get('recommendations', [])[:3])}"
                        )
                    },
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

    def _build_guidance_blocks(self, task_memory: dict[str, Any]) -> list[UIBlock]:
        """状態に応じた次のアクションボタンを返す。情報不足時はブロックなし。"""
        if task_memory.get("status") == "awaiting_plan_confirmation" and task_memory.get("draft_research_plan"):
            return [
                UIBlock(
                    type="actions",
                    title="次のステップ",
                    content={
                        "items": [
                            {
                                "label": "調査計画を承認して開始する",
                                "action_type": "approve_research_plan",
                                "payload": {},
                            },
                            {
                                "label": "条件を修正する",
                                "action_type": "revise_research_plan",
                                "payload": {},
                            },
                        ]
                    },
                )
            ]

        ranked_properties = task_memory.get("last_ranked_properties") or []
        if ranked_properties:
            top_id = str(ranked_properties[0].get("property_id_norm") or "")
            action_items: list[dict[str, Any]] = []
            if top_id:
                action_items.append(
                    {
                        "label": "第一候補の問い合わせ文を作成する",
                        "action_type": "generate_inquiry",
                        "payload": {"property_id": top_id},
                    }
                )
            action_items.append(
                {
                    "label": "契約書チェックへ進む",
                    "action_type": "start_contract_review",
                    "payload": {},
                }
            )
            return [
                UIBlock(
                    type="actions",
                    title="次のステップ",
                    content={"items": action_items},
                )
            ]

        return []

    def _build_guidance_response(
        self,
        *,
        session_id: str,
        task_memory: dict[str, Any],
        message: str,
    ) -> ChatMessageResponse:
        # フォールバック: 状態ベースの固定テンプレート
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

        # LLMで文脈依存メッセージに差し替え（失敗時はフォールバックを維持）
        try:
            _, _, llm_config = self._ensure_session_llm_config(
                session_id, task_memory=task_memory
            )
            adapter = self._get_adapter_for_route(
                llm_config=llm_config,
                route_key="planner",
                session_id=session_id,
                interaction_type="guidance",
            )
            if adapter is not None:
                llm_text = _generate_llm_guidance_message(
                    task_memory=task_memory,
                    user_message=message,
                    adapter=adapter,
                )
                if llm_text:
                    assistant_text = llm_text
        except Exception:
            pass

        response = ChatMessageResponse(
            status="awaiting_user_input",
            assistant_message=assistant_text,
            missing_slots=[],
            next_action="await_specific_input",
            blocks=self._build_guidance_blocks(task_memory),
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

    @staticmethod
    def _annotate_response_labels(response: ChatMessageResponse) -> ChatMessageResponse:
        """UIBlock.display_label と ChatMessageResponse.status_label をコンテンツベースで設定する。"""
        if not response.status_label:
            status_map: dict[str, str] = {
                "awaiting_profile_resume": "前回条件の引き継ぎ確認",
                "awaiting_user_input": "追加条件の回答待ち",
                "awaiting_plan_confirmation": "調査計画の承認待ち",
                "research_queued": "調査キュー登録済み",
                "research_running": "調査進行中",
                "research_failed": "調査エラー",
                "inquiry_draft_ready": "問い合わせ文の確認待ち",
                "awaiting_contract_text": "契約書入力待ち",
                "risk_check_completed": "契約リスク確認完了",
            }
            # cards ブロックの件数を取得してラベルに反映
            cards_count = 0
            for block in response.blocks:
                if block.type == "cards":
                    items = block.content.get("items") or []
                    cards_count = len(items) if isinstance(items, list) else 0
                    break
            if response.status in ("research_completed", "search_results_ready") and cards_count:
                response.status_label = f"{cards_count}件の候補が揃いました"
            elif response.status in status_map:
                response.status_label = status_map[response.status]
            elif response.pending_confirmation:
                response.status_label = "確認待ち"
            else:
                response.status_label = "処理完了"

        for block in response.blocks:
            if block.display_label:
                continue
            if block.type == "cards":
                items = block.content.get("items") or []
                count = len(items) if isinstance(items, list) else 0
                block.display_label = f"{count}件の候補" if count > 0 else "候補物件"
            elif block.type == "checklist":
                items = block.content.get("items") or []
                count = len(items) if isinstance(items, list) else 0
                block.display_label = f"{count}項目確認" if count > 0 else "チェック"
            elif block.type == "table":
                rows = block.content.get("rows") or []
                count = len(rows) if isinstance(rows, list) else 0
                block.display_label = f"{count}件比較" if count > 0 else "比較表"

        return response
