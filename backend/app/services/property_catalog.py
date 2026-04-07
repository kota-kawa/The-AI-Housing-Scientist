from __future__ import annotations

import json
from typing import Any

import httpx

from app.catalog import render_property_detail_html
from app.db import Database
from app.llm.base import LLMAdapter
from app.stages.search_normalize import _split_address_levels


DEFAULT_CATALOG_PROFILE = {
    "area_exact_bonus": 40.0,
    "area_municipality_bonus": 24.0,
    "area_partial_bonus": 12.0,
    "area_miss_penalty": 10.0,
    "budget_match_bonus": 30.0,
    "budget_near_bonus": 10.0,
    "budget_far_penalty": 20.0,
    "layout_match_bonus": 15.0,
    "layout_mismatch_penalty": 5.0,
    "station_match_bonus": 12.0,
    "station_near_bonus": 4.0,
    "station_far_penalty": 8.0,
    "move_in_match_bonus": 4.0,
    "must_condition_bonus": 8.0,
    "must_condition_partial_bonus": 3.0,
    "must_condition_missing_penalty": 4.0,
    "nice_to_have_bonus": 3.0,
    "nice_to_have_partial_bonus": 1.5,
    "liked_feature_bonus": 2.0,
    "excluded_feature_penalty": 3.0,
    "semantic_area_exact_bonus": 8.0,
    "semantic_area_municipality_bonus": 3.0,
    "semantic_must_strong_bonus": 6.0,
    "semantic_must_partial_bonus": 2.0,
    "semantic_nice_strong_bonus": 3.0,
    "semantic_nice_partial_bonus": 1.5,
    "semantic_bonus_cap": 18.0,
    "semantic_penalty_cap": 12.0,
}
CATALOG_LLM_SHORTLIST_MIN = 6
CATALOG_LLM_SHORTLIST_MAX = 10


def _resolve_profile(profile: dict[str, Any] | None = None) -> dict[str, float]:
    resolved = dict(DEFAULT_CATALOG_PROFILE)
    for key, value in (profile or {}).items():
        try:
            resolved[key] = float(value)
        except (TypeError, ValueError):
            continue
    return resolved


def _compact_text(value: Any, *, max_chars: int = 220) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _build_property_haystack(item: dict[str, Any]) -> str:
    return " ".join(
        [
            str(item.get("building_name") or ""),
            str(item.get("address") or ""),
            str(item.get("area_name") or ""),
            str(item.get("nearest_station") or ""),
            str(item.get("layout") or ""),
            str(item.get("available_date") or ""),
            str(item.get("notes") or ""),
            " ".join(str(feature) for feature in item.get("features", []) or []),
        ]
    )


def _collect_condition_list(user_memory: dict[str, Any], key: str) -> list[str]:
    deduped: list[str] = []
    for token in user_memory.get(key, []) or []:
        text = str(token).strip()
        if text and text not in deduped:
            deduped.append(text)
    return deduped


def _score_area_match(
    item: dict[str, Any],
    target_area: str,
    profile: dict[str, float],
) -> tuple[float, str]:
    if not target_area:
        return 0.0, ""

    target_levels = _split_address_levels(target_area)
    address = str(item.get("address") or "")
    area_name = str(item.get("area_name") or "")
    property_levels = _split_address_levels(address)
    locality = property_levels["locality"]

    if (
        target_levels["municipality"]
        and property_levels["municipality"]
        and target_levels["municipality"] == property_levels["municipality"]
    ):
        if target_levels["locality"]:
            if locality == target_levels["locality"] or target_levels["locality"] in address:
                return (
                    profile["area_exact_bonus"],
                    f"希望エリア {target_area} と住所粒度まで一致",
                )
            return (
                profile["area_municipality_bonus"],
                f"希望エリア {target_area} と同じ市区町村内",
            )
        return profile["area_exact_bonus"], f"希望エリア {target_area} と一致"

    if target_area and (target_area in address or target_area == area_name):
        return profile["area_partial_bonus"], f"希望エリア {target_area} に部分一致"

    return -profile["area_miss_penalty"], f"希望エリア {target_area} とは離れる"


def _build_fallback_condition_assessments(
    *,
    conditions: list[str],
    haystack: str,
) -> list[dict[str, str]]:
    assessments: list[dict[str, str]] = []
    for condition in conditions:
        match_level = "strong" if condition in haystack else "none"
        evidence = condition if match_level == "strong" else ""
        assessments.append(
            {
                "condition": condition,
                "match_level": match_level,
                "evidence": evidence,
            }
        )
    return assessments


def _score_condition_assessments(
    *,
    assessments: list[dict[str, str]],
    strong_bonus: float,
    partial_bonus: float,
    missing_penalty: float = 0.0,
) -> float:
    score = 0.0
    for assessment in assessments:
        level = str(assessment.get("match_level") or "none").strip()
        if level == "strong":
            score += strong_bonus
        elif level == "partial":
            score += partial_bonus
        elif missing_penalty > 0:
            score -= missing_penalty
    return score


def _build_llm_catalog_enhancements(
    *,
    candidates: list[dict[str, Any]],
    user_memory: dict[str, Any],
    query: str,
    adapter: LLMAdapter,
) -> dict[str, dict[str, Any]]:
    must_conditions = _collect_condition_list(user_memory, "must_conditions")
    nice_to_have = _collect_condition_list(user_memory, "nice_to_have")
    schema = {
        "type": "object",
        "properties": {
            "assessments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "property_id": {"type": "string"},
                        "area_match_level": {
                            "type": "string",
                            "enum": ["exact", "municipality", "partial", "none"],
                        },
                        "area_evidence": {"type": "string"},
                        "must_condition_assessments": {
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
                        "property_id",
                        "area_match_level",
                        "area_evidence",
                        "must_condition_assessments",
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
        "query": query,
        "user_preferences": {
            "target_area": str(user_memory.get("target_area") or ""),
            "budget_max": int(user_memory.get("budget_max") or 0),
            "station_walk_max": int(user_memory.get("station_walk_max") or 0),
            "layout_preference": str(user_memory.get("layout_preference") or ""),
            "move_in_date": str(user_memory.get("move_in_date") or ""),
            "must_conditions": must_conditions,
            "nice_to_have": nice_to_have,
            "learned_preferences": user_memory.get("learned_preferences", {}) or {},
        },
        "candidates": [
            {
                "property_id": str(item.get("property_id") or ""),
                "building_name": _compact_text(item.get("building_name") or "候補物件"),
                "address": _compact_text(item.get("address") or ""),
                "area_name": _compact_text(item.get("area_name") or ""),
                "nearest_station": _compact_text(item.get("nearest_station") or ""),
                "station_walk_min": int(item.get("station_walk_min") or 0),
                "layout": _compact_text(item.get("layout") or ""),
                "area_m2": float(item.get("area_m2") or 0.0),
                "rent": int(item.get("rent") or 0),
                "available_date": _compact_text(item.get("available_date") or ""),
                "features": [
                    _compact_text(feature, max_chars=80)
                    for feature in (item.get("features", []) or [])[:8]
                    if str(feature).strip()
                ],
                "notes": _compact_text(item.get("notes") or "", max_chars=240),
            }
            for item in candidates
        ],
        "output_rules": [
            "must_condition_assessments と nice_to_have_assessments は与えられた条件を順に評価する",
            "match_level は strong, partial, none のいずれか",
            "evidence には物件情報にある明示的な根拠だけを書く",
            "area_match_level は exact, municipality, partial, none のいずれか",
            "江東区 と 江東区豊洲 は同じではない。町名まで一致する場合だけ exact にする",
            "近い意味の設備は partial または strong で評価してよいが、推測はしない",
        ],
    }
    result = adapter.generate_structured(
        system=(
            "You are evaluating Japanese rental catalog candidates. "
            "Judge semantic alignment between user conditions and property features using only explicit evidence. "
            "Treat district-level and locality-level area matches carefully."
        ),
        user=json.dumps(payload, ensure_ascii=False, indent=2),
        schema=schema,
        temperature=0.1,
    )

    enhancements: dict[str, dict[str, Any]] = {}
    for raw_item in result.get("assessments", []) or []:
        property_id = str(raw_item.get("property_id") or "").strip()
        if not property_id:
            continue
        enhancements[property_id] = {
            "area_match_level": str(raw_item.get("area_match_level") or "none").strip(),
            "area_evidence": str(raw_item.get("area_evidence") or "").strip(),
            "must_condition_assessments": [
                {
                    "condition": str(assessment.get("condition") or "").strip(),
                    "match_level": str(assessment.get("match_level") or "none").strip(),
                    "evidence": str(assessment.get("evidence") or "").strip(),
                }
                for assessment in raw_item.get("must_condition_assessments", []) or []
                if str(assessment.get("condition") or "").strip()
            ],
            "nice_to_have_assessments": [
                {
                    "condition": str(assessment.get("condition") or "").strip(),
                    "match_level": str(assessment.get("match_level") or "none").strip(),
                    "evidence": str(assessment.get("evidence") or "").strip(),
                }
                for assessment in raw_item.get("nice_to_have_assessments", []) or []
                if str(assessment.get("condition") or "").strip()
            ],
        }
    return enhancements


class PropertyCatalogService:
    def __init__(self, db: Database):
        self.db = db

    def search(
        self,
        *,
        query: str,
        user_memory: dict[str, Any],
        count: int = 8,
        adapter: LLMAdapter | None = None,
    ) -> list[dict[str, Any]]:
        profile = _resolve_profile()
        scored: list[dict[str, Any]] = []
        for item in self.db.list_catalog_properties():
            score = self._score_property(item, user_memory, query, profile=profile)
            scored.append({"score": score, "item": item})

        scored.sort(
            key=lambda pair: (
                pair["score"],
                -abs(int(user_memory.get("budget_max") or pair["item"]["rent"]) - pair["item"]["rent"]),
                -pair["item"]["station_walk_min"],
            ),
            reverse=True,
        )

        if adapter is not None and scored:
            shortlist_size = min(
                len(scored),
                max(count * 2, CATALOG_LLM_SHORTLIST_MIN),
                CATALOG_LLM_SHORTLIST_MAX,
            )
            shortlist = [pair["item"] for pair in scored[:shortlist_size]]
            try:
                enhancements = _build_llm_catalog_enhancements(
                    candidates=shortlist,
                    user_memory=user_memory,
                    query=query,
                    adapter=adapter,
                )
            except Exception:
                enhancements = {}

            for pair in scored[:shortlist_size]:
                property_id = str(pair["item"].get("property_id") or "")
                enhancement = enhancements.get(property_id)
                if not enhancement:
                    continue
                semantic_delta = self._score_semantic_enhancement(
                    enhancement,
                    profile=profile,
                    user_memory=user_memory,
                )
                pair["score"] = round(pair["score"] + semantic_delta, 2)

            scored.sort(
                key=lambda pair: (
                    pair["score"],
                    -abs(int(user_memory.get("budget_max") or pair["item"]["rent"]) - pair["item"]["rent"]),
                    -pair["item"]["station_walk_min"],
                ),
                reverse=True,
            )

        normalized: list[dict[str, Any]] = []
        for pair in scored[: max(1, count)]:
            item = pair["item"]
            features = item.get("features", [])
            normalized.append(
                {
                    "title": (
                        f"{item['building_name']} {item['layout']} "
                        f"{item['area_m2']}㎡ 徒歩{item['station_walk_min']}分"
                    ),
                    "url": item["detail_url"],
                    "description": (
                        f"{item['address']} / 賃料{item['rent']:,}円 管理費{item['management_fee']:,}円 / "
                        f"{item['nearest_station']} 徒歩{item['station_walk_min']}分"
                    ),
                    "extra_snippets": features[:3],
                    "source_name": "mock_catalog",
                }
            )
        return normalized

    def fetch_detail_html(self, url: str) -> str | None:
        catalog_item = self.db.get_catalog_property_by_url(url)
        if catalog_item is not None:
            return render_property_detail_html(catalog_item)

        if not url.startswith(("http://", "https://")):
            return None

        try:
            with httpx.Client(
                timeout=6.0,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (compatible; HousingScientistBot/0.1; "
                        "+https://mock-housing.local)"
                    )
                },
                follow_redirects=True,
            ) as client:
                response = client.get(url)
                response.raise_for_status()
        except Exception:
            return None

        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type:
            return None
        return response.text

    def _score_property(
        self,
        item: dict[str, Any],
        user_memory: dict[str, Any],
        query: str,
        *,
        profile: dict[str, float] | None = None,
    ) -> float:
        resolved_profile = profile or _resolve_profile()
        score = 0.0
        haystack = _build_property_haystack(item)

        target_area = str(user_memory.get("target_area") or "").strip()
        if target_area:
            area_score, _ = _score_area_match(item, target_area, resolved_profile)
            score += area_score

        budget_max = int(user_memory.get("budget_max") or 0)
        if budget_max > 0:
            rent = int(item["rent"])
            if rent <= budget_max:
                score += resolved_profile["budget_match_bonus"]
            elif rent <= budget_max + 20000:
                score += resolved_profile["budget_near_bonus"]
            else:
                score -= resolved_profile["budget_far_penalty"]

        layout_preference = str(user_memory.get("layout_preference") or "").strip()
        if layout_preference:
            if layout_preference == item["layout"]:
                score += resolved_profile["layout_match_bonus"]
            else:
                score -= resolved_profile["layout_mismatch_penalty"]

        station_walk_max = int(user_memory.get("station_walk_max") or 0)
        if station_walk_max > 0:
            if int(item["station_walk_min"]) <= station_walk_max:
                score += resolved_profile["station_match_bonus"]
            elif int(item["station_walk_min"]) <= station_walk_max + 3:
                score += resolved_profile["station_near_bonus"]
            else:
                score -= resolved_profile["station_far_penalty"]

        move_in_date = str(user_memory.get("move_in_date") or "").strip()
        if move_in_date and move_in_date in str(item.get("available_date", "")):
            score += resolved_profile["move_in_match_bonus"]

        for token in _collect_condition_list(user_memory, "must_conditions"):
            if token in haystack:
                score += resolved_profile["must_condition_bonus"]

        for token in _collect_condition_list(user_memory, "nice_to_have"):
            if token in haystack:
                score += resolved_profile["nice_to_have_bonus"]

        learned = user_memory.get("learned_preferences", {}) or {}
        frequent_area = str(learned.get("frequent_area") or "").strip()
        if frequent_area and frequent_area in haystack:
            score += resolved_profile["area_partial_bonus"] / 3

        for token in learned.get("liked_features", []) or []:
            text = str(token).strip()
            if text and text in haystack:
                score += resolved_profile["liked_feature_bonus"]

        for token in learned.get("excluded_features", []) or []:
            text = str(token).strip()
            if text and text in haystack:
                score -= resolved_profile["excluded_feature_penalty"]

        return score

    def _score_semantic_enhancement(
        self,
        enhancement: dict[str, Any],
        *,
        profile: dict[str, float],
        user_memory: dict[str, Any],
    ) -> float:
        score = 0.0
        area_match_level = str(enhancement.get("area_match_level") or "none").strip()
        if area_match_level == "exact":
            score += profile["semantic_area_exact_bonus"]
        elif area_match_level == "municipality":
            score += profile["semantic_area_municipality_bonus"]

        must_conditions = _collect_condition_list(user_memory, "must_conditions")
        nice_to_have = _collect_condition_list(user_memory, "nice_to_have")
        must_assessments = enhancement.get("must_condition_assessments") or _build_fallback_condition_assessments(
            conditions=must_conditions,
            haystack="",
        )
        nice_assessments = enhancement.get("nice_to_have_assessments") or _build_fallback_condition_assessments(
            conditions=nice_to_have,
            haystack="",
        )

        score += _score_condition_assessments(
            assessments=must_assessments,
            strong_bonus=profile["semantic_must_strong_bonus"],
            partial_bonus=profile["semantic_must_partial_bonus"],
            missing_penalty=profile["must_condition_missing_penalty"],
        )
        score += _score_condition_assessments(
            assessments=nice_assessments,
            strong_bonus=profile["semantic_nice_strong_bonus"],
            partial_bonus=profile["semantic_nice_partial_bonus"],
        )

        return round(
            max(-profile["semantic_penalty_cap"], min(score, profile["semantic_bonus_cap"])),
            2,
        )
