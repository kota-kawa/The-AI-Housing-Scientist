from __future__ import annotations

import re
from typing import Any

from app.llm.base import LLMAdapter

FOLLOW_UP_SLOT_ORDER = ["target_area", "budget_max", "station_walk_max"]
SEARCH_SIGNAL_KEYS = (
    "budget_max",
    "target_area",
    "station_walk_max",
    "move_in_date",
    "layout_preference",
    "must_conditions",
    "nice_to_have",
)
SEARCH_INTENT_KEYWORDS = ("賃貸", "物件", "部屋", "比較", "探", "住まい")
GENERIC_SEARCH_PATTERNS = (
    re.compile(r"^(おすすめ|相談|比較|教えて|探したい|探して|お願いします)$"),
    re.compile(
        r"^(おすすめの)?(賃貸|物件|部屋|家)(を)?"
        r"(探したい|探して|教えて|比較して)(ください|お願いします)?$"
    ),
)
FOLLOW_UP_QUESTIONS: dict[str, dict[str, Any]] = {
    "target_area": {
        "slot": "target_area",
        "label": "希望エリア",
        "question": "どのエリアで探しますか？",
        "examples": ["江東区", "吉祥寺", "横浜駅周辺"],
    },
    "budget_max": {
        "slot": "budget_max",
        "label": "家賃上限",
        "question": "家賃の上限はいくらですか？",
        "examples": ["家賃12万円以内", "家賃15万円まで", "管理費込みで18万円以下"],
    },
    "station_walk_max": {
        "slot": "station_walk_max",
        "label": "駅徒歩",
        "question": "駅から徒歩何分以内を希望しますか？",
        "examples": ["駅徒歩10分以内", "徒歩7分まで", "駅近だとうれしい"],
    },
}
CONDITION_KEYWORDS: dict[str, str] = {
    "ペット": "ペット可",
    "犬": "ペット可",
    "猫": "ペット可",
    "楽器": "楽器可",
    "ピアノ": "楽器可",
    "在宅ワーク": "在宅ワーク向け",
    "テレワーク": "在宅ワーク向け",
    "SOHO": "SOHO相談可",
    "保証人不要": "保証人不要",
    "南向き": "南向き",
    "角部屋": "角部屋",
}


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
        r"((?:東京都|大阪府|神奈川県|埼玉県|千葉県|愛知県|京都府|兵庫県|福岡県|北海道)"
        r"[^\s、。]{1,12}(?:区|市|町|村))",
        r"([^\s、。]{1,12}(?:区|市|町|村))",
        r"([^\s、。]{1,12}駅(?:周辺|近辺)?)",
        r"(渋谷|新宿|池袋|品川|目黒|中野|吉祥寺|横浜|梅田|難波|天神|札幌|名古屋)",
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

    condition_tokens = [label for keyword, label in CONDITION_KEYWORDS.items() if keyword in message]
    if condition_tokens:
        result["must_conditions"] = sorted(set(condition_tokens))

    return result


def _merge_memory(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if key in {"must_conditions", "nice_to_have"}:
            existing = [str(item).strip() for item in merged.get(key, []) or [] if str(item).strip()]
            incoming = [str(item).strip() for item in value or [] if str(item).strip()]
            deduped: list[str] = []
            for item in existing + incoming:
                if item and item not in deduped:
                    deduped.append(item)
            if deduped:
                merged[key] = deduped
            continue
        if value not in (None, "", [], {}):
            merged[key] = value
    return merged


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
            "Extract user constraints into normalized fields. "
            "Classification rules:\n"
            "- must_conditions: 「必須」「絶対」「ないと困る」「必ず」「〜でないとだめ」など強い要件。"
            "複合条件（例:「南向きで角部屋」）は個別に分解してリストに含める。\n"
            "- nice_to_have: 「あったらいい」「できれば」「希望」「なるべく」「好み」「〜だとうれしい」"
            "など優先度の低い要件。判断が難しい場合はこちらに分類する。"
        ),
        user=f"User message:\n{message}",
        schema=schema,
        temperature=0.0,
    )


def _has_structured_search_signal(merged: dict[str, Any]) -> bool:
    return any(merged.get(key) not in (None, "", [], {}) for key in SEARCH_SIGNAL_KEYS)


def _is_generic_search_request(message: str) -> bool:
    normalized = re.sub(r"[\s　、。,.!！?？・]+", "", message)
    return any(pattern.fullmatch(normalized) for pattern in GENERIC_SEARCH_PATTERNS)


def detect_search_signal(message: str) -> bool:
    if _rule_based_slots(message):
        return True
    if _is_generic_search_request(message):
        return True
    return any(keyword in message for keyword in SEARCH_INTENT_KEYWORDS)


def _build_follow_up_questions(
    merged: dict[str, Any],
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    slots = [slot for slot in FOLLOW_UP_SLOT_ORDER if merged.get(slot) in (None, "", [], {})]
    if limit is not None:
        slots = slots[:limit]
    return [dict(FOLLOW_UP_QUESTIONS[slot]) for slot in slots]


def run_planner(
    *,
    message: str,
    user_memory: dict[str, Any],
    adapter: LLMAdapter | None,
) -> dict[str, Any]:
    if adapter is not None:
        try:
            llm_slots = _llm_parse(message, adapter)
            merged = _merge_memory(user_memory, llm_slots)
            # Rule-based fills only slots the LLM left empty (pure fallback).
            rule_slots = _rule_based_slots(message)
            for key, value in rule_slots.items():
                if merged.get(key) in (None, "", [], {}):
                    merged = _merge_memory(merged, {key: value})
        except Exception:
            merged = _merge_memory(user_memory, _rule_based_slots(message))
    else:
        merged = _merge_memory(user_memory, _rule_based_slots(message))

    follow_up_questions = _build_follow_up_questions(merged)

    if not _has_structured_search_signal(merged) and _is_generic_search_request(message):
        missing_slots = [item["slot"] for item in _build_follow_up_questions(merged, limit=3)]
        next_action = "missing_slots_question"
    else:
        missing_slots = []
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
        "follow_up_questions": follow_up_questions,
        "next_action": next_action,
        "user_memory": merged,
    }
