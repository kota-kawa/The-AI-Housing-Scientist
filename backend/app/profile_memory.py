from __future__ import annotations

from collections import Counter
from typing import Any


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
    summary = {
        "last_search_labels": summarize_memory_labels(user_memory),
        "frequent_area": learned.get("frequent_area", ""),
        "stable_preferences": learned.get("stable_preferences", []),
        "liked_features": learned.get("liked_features", []),
        "excluded_features": learned.get("excluded_features", []),
    }
    return summary


def update_profile_memory_with_search(
    profile_memory: dict[str, Any],
    *,
    query: str,
    user_memory: dict[str, Any],
    searched_at: str,
) -> dict[str, Any]:
    updated = dict(profile_memory)
    search_history = list(updated.get("search_history", []) or [])
    search_history.append(
        {
            "searched_at": searched_at,
            "query": query,
            "user_memory": {
                key: value
                for key, value in user_memory.items()
                if key != "learned_preferences"
            },
        }
    )
    updated["search_history"] = search_history[-MAX_SEARCH_HISTORY:]
    updated["learned_preferences"] = infer_learned_preferences(
        updated["search_history"],
        updated.get("reaction_history", []) or [],
    )
    return updated


def update_profile_memory_with_reaction(
    profile_memory: dict[str, Any],
    *,
    reaction: str,
    property_snapshot: dict[str, Any],
    recorded_at: str,
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
        }
    )
    updated["reaction_history"] = reaction_history[-MAX_REACTION_HISTORY:]
    updated["learned_preferences"] = infer_learned_preferences(
        updated.get("search_history", []) or [],
        updated["reaction_history"],
    )
    return updated


def infer_learned_preferences(
    search_history: list[dict[str, Any]],
    reaction_history: list[dict[str, Any]],
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
            liked_feature_counter if entry.get("reaction") == "favorite" else excluded_feature_counter
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

    return {
        "frequent_area": frequent_area,
        "stable_preferences": stable_preferences,
        "liked_features": [label for label, count in liked_feature_counter.most_common(3) if count >= 1],
        "excluded_features": [
            label for label, count in excluded_feature_counter.most_common(3) if count >= 1
        ],
    }
