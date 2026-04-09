from __future__ import annotations

from collections import Counter
import json
from typing import Any

from app.llm.base import LLMAdapter

MAX_SEARCH_HISTORY = 12
MAX_REACTION_HISTORY = 30


def merge_learned_preferences(
    user_memory: dict[str, Any],
    learned_preferences: dict[str, Any],
) -> dict[str, Any]:
    if not learned_preferences:
        return dict(user_memory)
    merged = dict(user_memory)
    merged["learned_preferences"] = learned_preferences
    return merged


def summarize_memory_labels(user_memory: dict[str, Any]) -> list[str]:
    labels: list[str] = []

    area = str(user_memory.get("target_area") or "").strip()
    if area:
        labels.append(area)

    budget_max = int(user_memory.get("budget_max") or 0)
    if budget_max > 0:
        labels.append(f"家賃{int(budget_max / 10000)}万円以下")

    layout = str(user_memory.get("layout_preference") or "").strip()
    if layout:
        labels.append(layout)

    station_walk_max = int(user_memory.get("station_walk_max") or 0)
    if station_walk_max > 0:
        labels.append(f"駅徒歩{station_walk_max}分以内")

    move_in_date = str(user_memory.get("move_in_date") or "").strip()
    if move_in_date:
        labels.append(f"入居時期 {move_in_date}")

    for key in ("must_conditions", "nice_to_have"):
        for token in user_memory.get(key, []) or []:
            text = str(token).strip()
            if text and text not in labels:
                labels.append(text)

    return labels


def build_profile_resume_summary(
    user_memory: dict[str, Any],
    profile_memory: dict[str, Any],
) -> dict[str, Any]:
    learned = profile_memory.get("learned_preferences", {}) or {}
    strategy = profile_memory.get("strategy_memory", {}) or {}
    summary = {
        "last_search_labels": summarize_memory_labels(user_memory),
        "frequent_area": learned.get("frequent_area", ""),
        "stable_preferences": learned.get("stable_preferences", []),
        "liked_features": learned.get("liked_features", []),
        "excluded_features": learned.get("excluded_features", []),
        "preferred_strategy_tags": strategy.get("preferred_strategy_tags", []),
    }
    return summary


def update_profile_memory_with_search(
    profile_memory: dict[str, Any],
    *,
    query: str,
    user_memory: dict[str, Any],
    searched_at: str,
    adapter: LLMAdapter | None = None,
    search_outcome: dict[str, Any] | None = None,
) -> dict[str, Any]:
    updated = dict(profile_memory)
    search_history = list(updated.get("search_history", []) or [])
    search_history.append(
        {
            "searched_at": searched_at,
            "query": query,
            "user_memory": {
                key: value for key, value in user_memory.items() if key != "learned_preferences"
            },
            "search_outcome": search_outcome or {},
        }
    )
    updated["search_history"] = search_history[-MAX_SEARCH_HISTORY:]
    updated["learned_preferences"] = infer_learned_preferences(
        updated["search_history"],
        updated.get("reaction_history", []) or [],
        adapter=adapter,
    )
    updated["strategy_memory"] = infer_strategy_memory(
        updated["search_history"],
        updated.get("reaction_history", []) or [],
    )
    updated["learned_preferences"]["strategy_memory"] = updated["strategy_memory"]
    return updated


def update_profile_memory_with_reaction(
    profile_memory: dict[str, Any],
    *,
    reaction: str,
    property_snapshot: dict[str, Any],
    recorded_at: str,
    adapter: LLMAdapter | None = None,
    strategy_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    updated = dict(profile_memory)
    reaction_history = list(updated.get("reaction_history", []) or [])
    reaction_history.append(
        {
            "recorded_at": recorded_at,
            "reaction": reaction,
            "property_id": property_snapshot.get("property_id_norm"),
            "building_name": property_snapshot.get("building_name"),
            "area_name": property_snapshot.get("area_name"),
            "layout": property_snapshot.get("layout"),
            "features": property_snapshot.get("features", []),
            "strategy_context": strategy_context or {},
        }
    )
    updated["reaction_history"] = reaction_history[-MAX_REACTION_HISTORY:]
    updated["learned_preferences"] = infer_learned_preferences(
        updated.get("search_history", []) or [],
        updated["reaction_history"],
        adapter=adapter,
    )
    updated["strategy_memory"] = infer_strategy_memory(
        updated.get("search_history", []) or [],
        updated["reaction_history"],
    )
    updated["learned_preferences"]["strategy_memory"] = updated["strategy_memory"]
    return updated


def infer_strategy_memory(
    search_history: list[dict[str, Any]],
    reaction_history: list[dict[str, Any]],
) -> dict[str, Any]:
    preferred_counter: Counter[str] = Counter()
    avoided_counter: Counter[str] = Counter()
    issue_counter: Counter[str] = Counter()
    episodes: list[dict[str, Any]] = []
    last_successful_path: list[str] = []

    for entry in search_history:
        outcome = entry.get("search_outcome", {}) or {}
        selected_path = list(outcome.get("selected_path", []) or [])
        path_tags: list[str] = []
        for node in selected_path:
            for tag in node.get("strategy_tags", []) or []:
                text = str(tag).strip()
                if text and text not in path_tags:
                    path_tags.append(text)

        readiness = str(outcome.get("readiness") or "")
        if readiness in {"medium", "high"}:
            preferred_counter.update(path_tags)
            if path_tags:
                last_successful_path = path_tags[:]
        elif readiness == "low":
            avoided_counter.update(path_tags)

        for issue in outcome.get("top_issues", []) or []:
            text = str(issue).strip()
            if text:
                issue_counter[text] += 1

        if outcome:
            episodes.append(
                {
                    "searched_at": entry.get("searched_at", ""),
                    "selected_branch_id": str(outcome.get("selected_branch_id") or ""),
                    "readiness": readiness,
                    "strategy_tags": path_tags,
                    "top_issues": [
                        str(item).strip()
                        for item in outcome.get("top_issues", []) or []
                        if str(item).strip()
                    ][:3],
                }
            )

    for entry in reaction_history:
        strategy_context = entry.get("strategy_context", {}) or {}
        path_tags = [
            str(tag).strip()
            for tag in strategy_context.get("selected_path_tags", []) or []
            if str(tag).strip()
        ]
        if entry.get("reaction") == "favorite":
            preferred_counter.update(path_tags)
        elif entry.get("reaction") == "exclude":
            avoided_counter.update(path_tags)

    preferred_strategy_tags = [tag for tag, count in preferred_counter.most_common(4) if count >= 1]
    avoided_strategy_tags = [
        tag
        for tag, count in avoided_counter.most_common(4)
        if count >= 1 and tag not in preferred_strategy_tags
    ]
    return {
        "episodes": episodes[-8:],
        "preferred_strategy_tags": preferred_strategy_tags,
        "avoided_strategy_tags": avoided_strategy_tags,
        "issue_recurrence": dict(issue_counter.most_common(5)),
        "last_successful_path": last_successful_path[:6],
    }


def _infer_preferences_with_llm(
    *,
    adapter: LLMAdapter,
    search_history: list[dict[str, Any]],
    reaction_history: list[dict[str, Any]],
) -> dict[str, Any]:
    schema = {
        "type": "object",
        "properties": {
            "primary_driver": {"type": "string"},
            "hidden_preferences": {
                "type": "array",
                "items": {"type": "string"},
            },
            "temporal_trend": {"type": "string"},
            "reliability": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["primary_driver", "hidden_preferences", "temporal_trend", "reliability"],
        "additionalProperties": False,
    }
    payload = {
        "search_history": [
            {
                "searched_at": entry.get("searched_at", ""),
                "query": entry.get("query", ""),
                "target_area": str((entry.get("user_memory") or {}).get("target_area") or ""),
                "budget_max": int((entry.get("user_memory") or {}).get("budget_max") or 0),
                "layout_preference": str(
                    (entry.get("user_memory") or {}).get("layout_preference") or ""
                ),
                "station_walk_max": int(
                    (entry.get("user_memory") or {}).get("station_walk_max") or 0
                ),
            }
            for entry in search_history
        ],
        "reaction_history": [
            {
                "recorded_at": entry.get("recorded_at", ""),
                "reaction": entry.get("reaction", ""),
                "building_name": str(entry.get("building_name") or ""),
                "area_name": str(entry.get("area_name") or ""),
                "layout": str(entry.get("layout") or ""),
                "features": (entry.get("features") or [])[:5],
            }
            for entry in reaction_history
        ],
        "output_rules": [
            "primary_driver: 検索・反応履歴から読み取れるユーザーの最重要条件を1文で",
            "hidden_preferences: 表明条件には現れないが行動（お気に入り・除外）から読み取れる隠れた好みのリスト（最大5件）",
            "temporal_trend: 時系列で変化しているパターンの説明。変化が見られなければ空文字",
            "reliability: 履歴が少ない・矛盾が多い場合は低く、一貫性が高い場合は高く（0〜1）",
        ],
    }
    try:
        return adapter.generate_structured(
            system=(
                "You analyze a user's Japanese rental property search and reaction history "
                "to infer latent preferences not captured by explicit filter conditions. "
                "Look for contradictions between stated budget/conditions and actual favorites. "
                "Identify implicit requirements that appear consistently in liked properties."
            ),
            user=json.dumps(payload, ensure_ascii=False, indent=2),
            schema=schema,
            temperature=0.1,
        )
    except Exception:
        return {}


def infer_learned_preferences(
    search_history: list[dict[str, Any]],
    reaction_history: list[dict[str, Any]],
    *,
    adapter: LLMAdapter | None = None,
) -> dict[str, Any]:
    area_counter: Counter[str] = Counter()
    layout_counter: Counter[str] = Counter()
    budget_counter: Counter[str] = Counter()
    walk_counter: Counter[str] = Counter()
    liked_feature_counter: Counter[str] = Counter()
    excluded_feature_counter: Counter[str] = Counter()

    for entry in search_history:
        user_memory = entry.get("user_memory", {}) or {}
        area = str(user_memory.get("target_area") or "").strip()
        layout = str(user_memory.get("layout_preference") or "").strip()
        budget = int(user_memory.get("budget_max") or 0)
        walk = int(user_memory.get("station_walk_max") or 0)
        if area:
            area_counter[area] += 1
        if layout:
            layout_counter[layout] += 1
        if budget > 0:
            budget_counter[f"家賃{int(budget / 10000)}万円以下"] += 1
        if walk > 0:
            walk_counter[f"駅徒歩{walk}分以内"] += 1

    for entry in reaction_history:
        target_counter = (
            liked_feature_counter
            if entry.get("reaction") == "favorite"
            else excluded_feature_counter
        )
        area = str(entry.get("area_name") or "").strip()
        if area:
            target_counter[area] += 1
        layout = str(entry.get("layout") or "").strip()
        if layout:
            target_counter[layout] += 1
        for feature in entry.get("features", []) or []:
            text = str(feature).strip()
            if text:
                target_counter[text] += 1

    stable_preferences: list[str] = []
    for counter in (budget_counter, walk_counter, layout_counter):
        if not counter:
            continue
        label, count = counter.most_common(1)[0]
        if count >= 2:
            stable_preferences.append(label)

    frequent_area = ""
    if area_counter:
        area_label, count = area_counter.most_common(1)[0]
        if count >= 2:
            frequent_area = area_label

    result: dict[str, Any] = {
        "frequent_area": frequent_area,
        "stable_preferences": stable_preferences,
        "liked_features": [
            label for label, count in liked_feature_counter.most_common(3) if count >= 1
        ],
        "excluded_features": [
            label for label, count in excluded_feature_counter.most_common(3) if count >= 1
        ],
    }

    if adapter is not None and (len(search_history) >= 2 or len(reaction_history) >= 2):
        llm_inferred = _infer_preferences_with_llm(
            adapter=adapter,
            search_history=search_history[-10:],
            reaction_history=reaction_history[-10:],
        )
        if llm_inferred:
            result["llm_inferred"] = llm_inferred

    return result
