from __future__ import annotations

from typing import Any

from app.models import RankedProperty


def _score_property(prop: dict[str, Any], user_memory: dict[str, Any]) -> tuple[float, str, str]:
    score = 50.0
    positives: list[str] = []
    negatives: list[str] = []

    budget_max = int(user_memory.get("budget_max") or 0)
    rent = int(prop.get("rent") or 0)
    if budget_max > 0 and rent > 0:
        if rent <= budget_max:
            score += 25
            positives.append(f"家賃{rent:,}円が上限{budget_max:,}円以内")
        elif rent <= budget_max + 20000:
            score += 5
            negatives.append(f"家賃{rent:,}円が上限をやや超過")
        else:
            score -= 20
            negatives.append(f"家賃{rent:,}円が上限を超過")

    station_walk_max = int(user_memory.get("station_walk_max") or 0)
    station_walk_min = int(prop.get("station_walk_min") or 0)
    if station_walk_max > 0 and station_walk_min > 0:
        if station_walk_min <= station_walk_max:
            score += 15
            positives.append(f"駅徒歩{station_walk_min}分で条件内")
        else:
            score -= 10
            negatives.append(f"駅徒歩{station_walk_min}分で条件超過")

    layout_pref = user_memory.get("layout_preference")
    layout = prop.get("layout") or ""
    if layout_pref:
        if layout == layout_pref:
            score += 10
            positives.append(f"間取り{layout}が希望一致")
        else:
            negatives.append(f"間取り{layout or '不明'}が希望{layout_pref}と不一致")

    if prop.get("area_m2", 0) and float(prop["area_m2"]) >= 25:
        score += 5
        positives.append("専有面積が25㎡以上")

    why_selected = "、".join(positives) if positives else "大きな加点要素は未確認"
    why_not_selected = "、".join(negatives) if negatives else "明確な減点要素は未確認"

    return round(score, 2), why_selected, why_not_selected


def run_ranking(*, normalized_properties: list[dict[str, Any]], user_memory: dict[str, Any]) -> dict[str, Any]:
    ranked: list[RankedProperty] = []
    for prop in normalized_properties:
        score, why_selected, why_not_selected = _score_property(prop, user_memory)
        ranked.append(
            RankedProperty(
                property_id_norm=prop["property_id_norm"],
                score=score,
                why_selected=why_selected,
                why_not_selected=why_not_selected,
            )
        )

    ranked.sort(key=lambda x: x.score, reverse=True)

    return {
        "ranked_properties": [item.model_dump() for item in ranked],
        "why_selected": {
            item.property_id_norm: item.why_selected
            for item in ranked
        },
        "why_not_selected": {
            item.property_id_norm: item.why_not_selected
            for item in ranked
        },
    }
