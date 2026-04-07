from __future__ import annotations

import re
from typing import Any

from app.llm.base import LLMAdapter

REQUIRED_SLOTS = ["budget_max", "target_area", "station_walk_max"]


def _extract_budget_yen(text: str) -> int | None:
    man_match = re.search(r"(\d+(?:\.\d+)?)\s*万", text)
    if man_match:
        return int(float(man_match.group(1)) * 10000)
    yen_match = re.search(r"(\d{2,3})(?:,|，)?(\d{3})\s*円", text)
    if yen_match:
        return int(f"{yen_match.group(1)}{yen_match.group(2)}")
    return None


def _extract_station_walk(text: str) -> int | None:
    match = re.search(r"徒歩\s*(\d{1,2})\s*分", text)
    if match:
        return int(match.group(1))
    return None


def _extract_target_area(text: str) -> str | None:
    patterns = [
        r"(渋谷|新宿|池袋|品川|目黒|中野|吉祥寺|横浜|梅田|難波|天神|札幌|名古屋)",
        r"(東京都[^\s、。]+区)",
        r"(大阪府[^\s、。]+区)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None


def _extract_move_in_date(text: str) -> str | None:
    match = re.search(r"(\d{4})[/-](\d{1,2})", text)
    if match:
        return f"{match.group(1)}-{int(match.group(2)):02d}"
    if "すぐ" in text or "即" in text:
        return "asap"
    return None


def _rule_based_slots(message: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    budget = _extract_budget_yen(message)
    if budget is not None:
        result["budget_max"] = budget

    station_walk = _extract_station_walk(message)
    if station_walk is not None:
        result["station_walk_max"] = station_walk

    area = _extract_target_area(message)
    if area:
        result["target_area"] = area

    move_in = _extract_move_in_date(message)
    if move_in:
        result["move_in_date"] = move_in

    if "2LDK" in message:
        result["layout_preference"] = "2LDK"
    elif "1LDK" in message:
        result["layout_preference"] = "1LDK"
    elif "1K" in message:
        result["layout_preference"] = "1K"

    return result


def _llm_parse(message: str, adapter: LLMAdapter) -> dict[str, Any]:
    schema = {
        "type": "object",
        "properties": {
            "budget_max": {"type": "integer"},
            "target_area": {"type": "string"},
            "station_walk_max": {"type": "integer"},
            "move_in_date": {"type": "string"},
            "layout_preference": {"type": "string"},
            "must_conditions": {"type": "array", "items": {"type": "string"}},
            "nice_to_have": {"type": "array", "items": {"type": "string"}},
        },
        "required": [],
        "additionalProperties": False,
    }

    return adapter.generate_structured(
        system=(
            "You are a Japanese rental planner. "
            "Extract user constraints into normalized fields."
        ),
        user=f"User message:\n{message}",
        schema=schema,
        temperature=0.0,
    )


def run_planner(
    *,
    message: str,
    user_memory: dict[str, Any],
    adapter: LLMAdapter | None,
) -> dict[str, Any]:
    merged = dict(user_memory)
    merged.update(_rule_based_slots(message))

    if adapter is not None:
        try:
            llm_slots = _llm_parse(message, adapter)
            for key, value in llm_slots.items():
                if value not in (None, "", [], {}):
                    merged[key] = value
        except Exception:
            # Rule-based fallback is the deterministic baseline for PoC.
            pass

    missing_slots = [slot for slot in REQUIRED_SLOTS if slot not in merged]
    if missing_slots:
        next_action = "missing_slots_question"
    else:
        next_action = "search_and_compare"

    plan_summary = {
        "budget_max": merged.get("budget_max"),
        "target_area": merged.get("target_area"),
        "station_walk_max": merged.get("station_walk_max"),
        "move_in_date": merged.get("move_in_date"),
        "layout_preference": merged.get("layout_preference"),
    }

    return {
        "plan": plan_summary,
        "missing_slots": missing_slots,
        "next_action": next_action,
        "user_memory": merged,
    }
