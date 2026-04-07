from __future__ import annotations

import json
from typing import Any

from app.llm.base import LLMAdapter
from app.models import RankedProperty


DEFAULT_RANKING_PROFILE = {
    "base_score": 50.0,
    "budget_match_bonus": 25.0,
    "budget_near_bonus": 5.0,
    "budget_far_penalty": 20.0,
    "station_match_bonus": 15.0,
    "station_far_penalty": 10.0,
    "layout_match_bonus": 10.0,
    "rent_missing_penalty": 15.0,
    "station_missing_penalty": 6.0,
    "layout_missing_penalty": 5.0,
    "large_area_bonus": 5.0,
    "frequent_area_bonus": 4.0,
    "nice_to_have_strong_bonus": 4.0,
    "nice_to_have_partial_bonus": 2.0,
    "nice_to_have_bonus_cap": 10.0,
    "liked_feature_bonus": 2.0,
    "excluded_feature_penalty": 3.0,
}


def _resolve_profile(profile: dict[str, Any] | None) -> dict[str, float]:
    resolved = dict(DEFAULT_RANKING_PROFILE)
    for key, value in (profile or {}).items():
        try:
            resolved[key] = float(value)
        except (TypeError, ValueError):
            continue
    return resolved


def _collect_nice_to_have(user_memory: dict[str, Any]) -> list[str]:
    deduped: list[str] = []
    for item in user_memory.get("nice_to_have", []) or []:
        text = str(item).strip()
        if text and text not in deduped:
            deduped.append(text)
    return deduped


def _build_property_haystack(prop: dict[str, Any]) -> str:
    return " ".join(
        [
            str(prop.get("building_name") or ""),
            str(prop.get("notes") or ""),
            " ".join(str(item) for item in prop.get("features", []) or []),
            str(prop.get("area_name") or prop.get("address") or ""),
            str(prop.get("layout") or ""),
            str(prop.get("nearest_station") or ""),
        ]
    )


def _compact_text(value: Any, *, max_chars: int = 220) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _score_property_rules(
    prop: dict[str, Any],
    user_memory: dict[str, Any],
    profile: dict[str, float],
) -> tuple[float, list[str], list[str]]:
    score = profile["base_score"]
    positives: list[str] = []
    negatives: list[str] = []

    budget_max = int(user_memory.get("budget_max") or 0)
    rent = int(prop.get("rent") or 0)
    if budget_max > 0 and rent > 0:
        if rent <= budget_max:
            score += profile["budget_match_bonus"]
            positives.append(f"家賃{rent:,}円が上限{budget_max:,}円以内")
        elif rent <= budget_max + 20000:
            score += profile["budget_near_bonus"]
            negatives.append(f"家賃{rent:,}円が上限をやや超過")
        else:
            score -= profile["budget_far_penalty"]
            negatives.append(f"家賃{rent:,}円が上限を超過")

    station_walk_max = int(user_memory.get("station_walk_max") or 0)
    station_walk_min = int(prop.get("station_walk_min") or 0)
    if station_walk_max > 0 and station_walk_min > 0:
        if station_walk_min <= station_walk_max:
            score += profile["station_match_bonus"]
            positives.append(f"駅徒歩{station_walk_min}分で条件内")
        else:
            score -= profile["station_far_penalty"]
            negatives.append(f"駅徒歩{station_walk_min}分で条件超過")

    layout_pref = user_memory.get("layout_preference")
    layout = prop.get("layout") or ""
    if layout_pref:
        if layout == layout_pref:
            score += profile["layout_match_bonus"]
            positives.append(f"間取り{layout}が希望一致")
        else:
            negatives.append(f"間取り{layout or '不明'}が希望{layout_pref}と不一致")

    if rent <= 0:
        score -= profile["rent_missing_penalty"]
        negatives.append("家賃情報が取得できていない")

    if station_walk_min <= 0:
        score -= profile["station_missing_penalty"]
        negatives.append("駅徒歩情報が取得できていない")

    if not layout:
        score -= profile["layout_missing_penalty"]
        negatives.append("間取り情報が取得できていない")

    if prop.get("area_m2", 0) and float(prop["area_m2"]) >= 25:
        score += profile["large_area_bonus"]
        positives.append("専有面積が25㎡以上")

    learned = user_memory.get("learned_preferences", {}) or {}
    frequent_area = str(learned.get("frequent_area") or "").strip()
    area_name = str(prop.get("area_name") or prop.get("address") or "")
    if frequent_area and frequent_area in area_name:
        score += profile["frequent_area_bonus"]
        positives.append(f"過去に好んだエリア {frequent_area} に近い")

    notes_haystack = _build_property_haystack(prop)
    for token in learned.get("liked_features", []) or []:
        text = str(token).strip()
        if text and text in notes_haystack:
            score += profile["liked_feature_bonus"]
            positives.append(f"過去の反応で好まれた条件 {text} に合致")

    for token in learned.get("excluded_features", []) or []:
        text = str(token).strip()
        if text and text in notes_haystack:
            score -= profile["excluded_feature_penalty"]
            negatives.append(f"過去に除外した条件 {text} と重なる")

    return round(score, 2), positives, negatives


def _build_fallback_nice_to_have_assessments(
    *,
    prop: dict[str, Any],
    user_memory: dict[str, Any],
) -> list[dict[str, str]]:
    haystack = _build_property_haystack(prop)
    assessments: list[dict[str, str]] = []
    for condition in _collect_nice_to_have(user_memory):
        match_level = "strong" if condition in haystack else "none"
        evidence = f"{condition}に関する記載あり" if match_level == "strong" else ""
        assessments.append(
            {
                "condition": condition,
                "match_level": match_level,
                "evidence": evidence,
            }
        )
    return assessments


def _score_nice_to_have_assessments(
    assessments: list[dict[str, str]],
    profile: dict[str, float],
) -> tuple[float, list[str], list[str]]:
    score_delta = 0.0
    positives: list[str] = []
    negatives: list[str] = []

    for assessment in assessments:
        condition = str(assessment.get("condition") or "").strip()
        match_level = str(assessment.get("match_level") or "none").strip()
        evidence = str(assessment.get("evidence") or "").strip()
        evidence_text = evidence or condition

        if not condition:
            continue

        if match_level == "strong":
            score_delta += profile["nice_to_have_strong_bonus"]
            positives.append(f"「{condition}」に合う要素として{evidence_text}")
        elif match_level == "partial":
            score_delta += profile["nice_to_have_partial_bonus"]
            positives.append(f"「{condition}」に近い要素として{evidence_text}")
        else:
            negatives.append(f"「{condition}」に合う明確な記載は見当たりません")

    return round(min(score_delta, profile["nice_to_have_bonus_cap"]), 2), positives, negatives


def _build_fallback_selected_reason(
    positives: list[str],
    nice_to_have_positives: list[str],
) -> str:
    highlights = [item for item in positives[:2] if item]
    if nice_to_have_positives:
        highlights.append(nice_to_have_positives[0])
    elif len(positives) > 2:
        highlights.append(positives[2])
    if not highlights:
        return "主要条件との相性は悪くありませんが、決め手になる強みはまだ限定的です。"
    return f"{'、'.join(highlights)}ため、条件との相性が良い候補です。"


def _build_fallback_not_selected_reason(
    negatives: list[str],
    nice_to_have_negatives: list[str],
) -> str:
    concerns = [item for item in negatives[:2] if item]
    if nice_to_have_negatives:
        concerns.append(nice_to_have_negatives[0])
    elif len(negatives) > 2:
        concerns.append(negatives[2])
    if not concerns:
        return "致命的な懸念はありませんが、募集条件や細かな住み心地は追加確認したい候補です。"
    return f"{'、'.join(concerns)}ため、優先度はやや下がります。"


def _build_llm_ranking_enhancements(
    *,
    normalized_properties: list[dict[str, Any]],
    user_memory: dict[str, Any],
    rule_results_by_id: dict[str, dict[str, Any]],
    adapter: LLMAdapter,
) -> dict[str, dict[str, Any]]:
    nice_to_have = _collect_nice_to_have(user_memory)
    properties_payload = [
        {
            "property_id_norm": prop.get("property_id_norm"),
            "building_name": _compact_text(prop.get("building_name") or "対象物件"),
            "address": _compact_text(prop.get("address") or ""),
            "area_name": _compact_text(prop.get("area_name") or ""),
            "nearest_station": _compact_text(prop.get("nearest_station") or ""),
            "station_walk_min": int(prop.get("station_walk_min") or 0),
            "rent": int(prop.get("rent") or 0),
            "layout": _compact_text(prop.get("layout") or ""),
            "area_m2": float(prop.get("area_m2") or 0.0),
            "features": [
                _compact_text(item, max_chars=80)
                for item in (prop.get("features", []) or [])[:8]
                if str(item).strip()
            ],
            "notes": _compact_text(prop.get("notes") or "", max_chars=280),
            "rule_based_positives": rule_results_by_id.get(prop["property_id_norm"], {}).get(
                "positives", []
            ),
            "rule_based_negatives": rule_results_by_id.get(prop["property_id_norm"], {}).get(
                "negatives", []
            ),
        }
        for prop in normalized_properties
    ]
    schema = {
        "type": "object",
        "properties": {
            "assessments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "property_id_norm": {"type": "string"},
                        "why_selected": {"type": "string"},
                        "why_not_selected": {"type": "string"},
                        "nice_to_have_assessments": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "condition": {"type": "string"},
                                    "match_level": {
                                        "type": "string",
                                        "enum": ["strong", "partial", "none"],
                                    },
                                    "evidence": {"type": "string"},
                                },
                                "required": ["condition", "match_level", "evidence"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": [
                        "property_id_norm",
                        "why_selected",
                        "why_not_selected",
                        "nice_to_have_assessments",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["assessments"],
        "additionalProperties": False,
    }
    payload = {
        "user_preferences": {
            "budget_max": int(user_memory.get("budget_max") or 0),
            "station_walk_max": int(user_memory.get("station_walk_max") or 0),
            "layout_preference": str(user_memory.get("layout_preference") or ""),
            "must_conditions": [
                str(item).strip()
                for item in user_memory.get("must_conditions", []) or []
                if str(item).strip()
            ],
            "nice_to_have": nice_to_have,
        },
        "properties": properties_payload,
        "output_rules": [
            "nice_to_have_assessments は user_preferences.nice_to_have を対象に評価する",
            "match_level は strong, partial, none のいずれか",
            "evidence には物件情報にある明示的な根拠だけを書く",
            "why_selected は 1-2文の自然な日本語で、主な魅力を具体的に述べる",
            "why_not_selected は 1-2文の自然な日本語で、懸念点や未確認点を具体的に述べる",
            "与えられていない設備・条件は推測しない",
        ],
    }
    result = adapter.generate_structured(
        system=(
            "You are a Japanese rental ranking assistant. "
            "Judge each nice-to-have condition only from explicit evidence in the property text. "
            "Then rewrite the selection reasons into natural Japanese without inventing facts."
        ),
        user=json.dumps(payload, ensure_ascii=False, indent=2),
        schema=schema,
        temperature=0.1,
    )

    enhancements: dict[str, dict[str, Any]] = {}
    for raw_item in result.get("assessments", []) or []:
        property_id = str(raw_item.get("property_id_norm") or "").strip()
        if not property_id:
            continue
        assessments: list[dict[str, str]] = []
        for raw_assessment in raw_item.get("nice_to_have_assessments", []) or []:
            condition = str(raw_assessment.get("condition") or "").strip()
            if not condition:
                continue
            assessments.append(
                {
                    "condition": condition,
                    "match_level": str(raw_assessment.get("match_level") or "none").strip(),
                    "evidence": str(raw_assessment.get("evidence") or "").strip(),
                }
            )
        enhancements[property_id] = {
            "why_selected": str(raw_item.get("why_selected") or "").strip(),
            "why_not_selected": str(raw_item.get("why_not_selected") or "").strip(),
            "nice_to_have_assessments": assessments,
        }

    return enhancements


def run_ranking(
    *,
    normalized_properties: list[dict[str, Any]],
    user_memory: dict[str, Any],
    ranking_profile: dict[str, Any] | None = None,
    adapter: LLMAdapter | None = None,
) -> dict[str, Any]:
    profile = _resolve_profile(ranking_profile)
    rule_results_by_id: dict[str, dict[str, Any]] = {}
    llm_enhancements: dict[str, dict[str, Any]] = {}

    for prop in normalized_properties:
        score, positives, negatives = _score_property_rules(prop, user_memory, profile)
        rule_results_by_id[prop["property_id_norm"]] = {
            "score": score,
            "positives": positives,
            "negatives": negatives,
        }

    if adapter is not None and normalized_properties:
        try:
            llm_enhancements = _build_llm_ranking_enhancements(
                normalized_properties=normalized_properties,
                user_memory=user_memory,
                rule_results_by_id=rule_results_by_id,
                adapter=adapter,
            )
        except Exception:
            llm_enhancements = {}

    ranked: list[RankedProperty] = []
    nice_to_have_assessments_by_id: dict[str, list[dict[str, str]]] = {}
    for prop in normalized_properties:
        property_id = prop["property_id_norm"]
        rule_result = rule_results_by_id[property_id]
        enhancement = llm_enhancements.get(property_id, {})
        assessments = enhancement.get("nice_to_have_assessments") or _build_fallback_nice_to_have_assessments(
            prop=prop,
            user_memory=user_memory,
        )
        score_delta, nice_to_have_positives, nice_to_have_negatives = _score_nice_to_have_assessments(
            assessments,
            profile,
        )
        score = round(float(rule_result["score"]) + score_delta, 2)
        why_selected = str(enhancement.get("why_selected") or "").strip() or _build_fallback_selected_reason(
            rule_result["positives"],
            nice_to_have_positives,
        )
        why_not_selected = str(enhancement.get("why_not_selected") or "").strip() or _build_fallback_not_selected_reason(
            rule_result["negatives"],
            nice_to_have_negatives,
        )
        nice_to_have_assessments_by_id[property_id] = assessments
        ranked.append(
            RankedProperty(
                property_id_norm=property_id,
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
        "nice_to_have_assessments": nice_to_have_assessments_by_id,
        "llm_reasoning_applied": bool(llm_enhancements),
        "ranking_profile": profile,
    }
