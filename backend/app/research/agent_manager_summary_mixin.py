from __future__ import annotations

import json
from typing import Any

from app.stages.result_summarizer import (
    COMMON_RISKS_KEY,
    OPEN_QUESTIONS_KEY,
    PROPERTY_CANDIDATES_KEY,
)


class AgentManagerSummaryMixin:
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
        branch_result_summary: dict[str, Any] | None = None,
    ) -> str:
        user_memory = self.approved_plan.get("user_memory_snapshot", self.user_memory)
        condition_summary = self._build_condition_summary(user_memory)
        summarized_candidates = list((branch_result_summary or {}).get(PROPERTY_CANDIDATES_KEY, []) or [])

        if not ranked_properties:
            top_candidate = summarized_candidates[0] if summarized_candidates else {}
            lead = (
                f"{condition_summary}の条件で調査しましたが、問い合わせに進める候補は十分に揃いませんでした。"
                if condition_summary
                else "今回の条件で調査しましたが、問い合わせに進める候補は十分に揃いませんでした。"
            )
            detail_hit_count = int(search_summary.get("detail_hit_count") or 0)
            follow_up = (
                f"詳細ページを確認できた候補は{detail_hit_count}件にとどまっているため、条件を少し広げて再調査するのが安全です。"
            )
            if top_candidate:
                follow_up = (
                    f"圧縮要約では{top_candidate.get('building_name', '有望候補')}が残っていますが、"
                    "未確認項目が多いため再調査または問い合わせ前提で扱うのが安全です。"
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

    def _build_confirmation_items(
        self,
        *,
        ranked_properties: list[dict[str, Any]],
        normalized_properties: list[dict[str, Any]],
        source_items: list[dict[str, Any]],
        failure_summary: dict[str, Any],
        branch_result_summary: dict[str, Any] | None = None,
    ) -> list[str]:
        by_id = {item["property_id_norm"]: item for item in normalized_properties}
        top_property = by_id.get(str(ranked_properties[0]["property_id_norm"]), {}) if ranked_properties else {}

        items: list[str] = []
        if source_items:
            items.append("掲載元ごとの差分条件")
        if top_property.get("notes"):
            items.append("募集条件の最新状況")
        if not top_property.get("rent"):
            items.append("家賃と管理費の内訳")
        if not top_property.get("station_walk_min"):
            items.append("駅徒歩分数")
        if not top_property.get("layout"):
            items.append("間取りと居室の広さ")
        items.extend(
            [
                "初期費用の内訳",
                "短期解約違約金・更新料・解約予告",
            ]
        )
        for recommendation in failure_summary.get("recommendations", []) or []:
            text = str(recommendation).strip()
            if text and text not in items:
                items.append(text)
        for question in (branch_result_summary or {}).get(OPEN_QUESTIONS_KEY, []) or []:
            text = str(question).strip()
            if text and text not in items:
                items.append(text)
        return items[:5]

    def _build_llm_research_summary(
        self,
        *,
        ranked_properties: list[dict[str, Any]],
        normalized_properties: list[dict[str, Any]],
        search_summary: dict[str, Any],
        source_items: list[dict[str, Any]],
        offline_evaluation: dict[str, Any],
        branch_result_summary: dict[str, Any] | None = None,
        selected_branch_summary: dict[str, Any] | None = None,
        branch_summaries: list[dict[str, Any]] | None = None,
        failure_summary: dict[str, Any] | None = None,
        selected_path: list[dict[str, Any]] | None = None,
        search_tree_summary: dict[str, Any] | None = None,
    ) -> str:
        if self.research_adapter is None:
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
                    "address": str(prop.get("address") or ""),
                    "rent": int(prop.get("rent") or 0),
                    "station_walk_min": int(prop.get("station_walk_min") or 0),
                    "layout": str(prop.get("layout") or ""),
                    "area_m2": float(prop.get("area_m2") or 0.0),
                    "score": float(ranked.get("score") or 0.0),
                    "why_selected": str(ranked.get("why_selected") or ""),
                    "why_not_selected": str(ranked.get("why_not_selected") or ""),
                }
            )
        selected_branch = selected_branch_summary or {}
        other_branches = [
            {
                "branch_id": str(item.get("branch_id") or ""),
                "label": str(item.get("label") or ""),
                "branch_score": float(item.get("branch_score") or 0.0),
                "frontier_score": float(item.get("frontier_score") or 0.0),
                "depth": int(item.get("depth") or 0),
                "detail_coverage": float(item.get("detail_coverage") or 0.0),
                "structured_ratio": float(item.get("structured_ratio") or 0.0),
                "normalized_count": int(item.get("normalized_count") or 0),
                "issues": [str(issue).strip() for issue in item.get("issues", [])[:2] if str(issue).strip()],
            }
            for item in (branch_summaries or [])
            if str(item.get("branch_id") or "") != str(selected_branch.get("branch_id") or "")
        ][:2]
        failure_info = failure_summary or {}
        payload = {
            "condition_summary": self._build_condition_summary(user_memory),
            "candidate_count": len(ranked_properties),
            "top_candidates": top_candidates,
            "selected_branch": {
                "branch_id": str(selected_branch.get("branch_id") or ""),
                "label": str(selected_branch.get("label") or ""),
                "branch_score": float(selected_branch.get("branch_score") or 0.0),
                "frontier_score": float(selected_branch.get("frontier_score") or 0.0),
                "depth": int(selected_branch.get("depth") or 0),
                "detail_coverage": float(selected_branch.get("detail_coverage") or 0.0),
                "structured_ratio": float(selected_branch.get("structured_ratio") or 0.0),
                "normalized_count": int(selected_branch.get("normalized_count") or 0),
                "summary": str(selected_branch.get("summary") or ""),
                "strategy_tags": [
                    str(tag).strip()
                    for tag in selected_branch.get("strategy_tags", []) or []
                    if str(tag).strip()
                ][:4],
            },
            "alternative_branches": other_branches,
            "selected_path": [
                {
                    "branch_id": str(item.get("branch_id") or ""),
                    "label": str(item.get("label") or ""),
                    "depth": int(item.get("depth") or 0),
                    "strategy_tags": [
                        str(tag).strip()
                        for tag in item.get("strategy_tags", []) or []
                        if str(tag).strip()
                    ][:4],
                    "branch_score": float(item.get("branch_score") or 0.0),
                }
                for item in (selected_path or [])[:4]
            ],
            "search_tree_summary": search_tree_summary or {},
            "branch_result_summary": {
                "property_candidates": [
                    {
                        "building_name": str(item.get("building_name") or "候補物件"),
                        "rent": int(item.get("rent") or 0),
                        "layout": str(item.get("layout") or ""),
                        "station_walk_min": int(item.get("station_walk_min") or 0),
                        "area_m2": float(item.get("area_m2") or 0.0),
                        "reason": str(item.get("reason") or ""),
                    }
                    for item in (branch_result_summary or {}).get(PROPERTY_CANDIDATES_KEY, [])[:3]
                ],
                "common_risks": [
                    str(item).strip()
                    for item in (branch_result_summary or {}).get(COMMON_RISKS_KEY, [])[:3]
                    if str(item).strip()
                ],
                "open_questions": [
                    str(item).strip()
                    for item in (branch_result_summary or {}).get(OPEN_QUESTIONS_KEY, [])[:5]
                    if str(item).strip()
                ],
            },
            "search_summary": {
                "detail_hit_count": int(search_summary.get("detail_hit_count") or 0),
                "normalized_count": int(search_summary.get("normalized_count") or 0),
                "duplicate_group_count": int(search_summary.get("duplicate_group_count") or 0),
            },
            "offline_evaluation": {
                "readiness": str(offline_evaluation.get("readiness") or ""),
                "detail_coverage": float(offline_evaluation.get("detail_coverage") or 0.0),
                "structured_ratio": float(offline_evaluation.get("structured_ratio") or 0.0),
                "recommendations": [
                    str(item).strip()
                    for item in offline_evaluation.get("recommendations", [])[:3]
                    if str(item).strip()
                ],
            },
            "failure_summary": {
                "summary": str(failure_info.get("summary") or ""),
                "top_issues": [
                    str(item).strip()
                    for item in failure_info.get("top_issues", [])[:3]
                    if str(item).strip()
                ],
                "recommendations": [
                    str(item).strip()
                    for item in failure_info.get("recommendations", [])[:3]
                    if str(item).strip()
                ],
            },
            "source_count": len(source_items),
            "confirmation_items": self._build_confirmation_items(
                ranked_properties=ranked_properties,
                normalized_properties=normalized_properties,
                source_items=source_items,
                failure_summary=failure_info,
                branch_result_summary=branch_result_summary,
            ),
            "required_output_format": [
                "結論: ...",
                "理由: ...",
                "懸念: ...",
                "次の確認事項: ...",
            ],
        }
        summary = self.research_adapter.generate_text(
            system=(
                "You are a Japanese rental research assistant. "
                "Summarize the final research result for a user making a rental decision. "
                "Use only the provided facts. Do not invent properties, numbers, or conditions. "
                "Return exactly four lines in Japanese starting with "
                "'結論:', '理由:', '懸念:', and '次の確認事項:'."
            ),
            user=json.dumps(payload, ensure_ascii=False, indent=2),
            temperature=0.2,
        ).strip()
        return "\n".join(" ".join(line.split()) for line in summary.splitlines() if line.strip())
