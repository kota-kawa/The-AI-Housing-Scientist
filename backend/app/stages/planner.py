from __future__ import annotations

import json
import re
from typing import Any

from app.llm.base import LLMAdapter

FOLLOW_UP_SLOT_ORDER = [
    "target_area",
    "budget_max",
    "station_walk_max",
    "layout_preference",
    "move_in_date",
]
FOLLOW_UP_OPTIONAL_SLOTS = ["must_conditions", "nice_to_have"]
SEARCH_SIGNAL_KEYS = (
    "budget_max",
    "target_area",
    "station_walk_max",
    "move_in_date",
    "layout_preference",
    "must_conditions",
    "nice_to_have",
)
SEARCH_INTENT_KEYWORDS = ("賃貸", "物件", "部屋", "比較", "探", "住まい", "引っ越", "一人暮らし")
RISK_INTENT_KEYWORDS = (
    "契約",
    "更新料",
    "違約金",
    "短期解約",
    "解約予告",
    "保証会社",
    "敷金",
    "礼金",
    "退去",
    "原状回復",
)
GENERIC_SEARCH_PATTERNS = (
    re.compile(r"^(おすすめ|相談|比較|教えて|探したい|探して|お願いします)$"),
    re.compile(
        r"^(おすすめの)?(賃貸|物件|部屋|家)(を)?"
        r"(探したい|探して|教えて|比較して)(ください|お願いします)?$"
    ),
)
DEFAULT_FOLLOW_UP_QUESTIONS: dict[str, dict[str, Any]] = {
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
    "layout_preference": {
        "slot": "layout_preference",
        "label": "間取り",
        "question": "希望の間取りはありますか？",
        "examples": ["1LDK", "2DK以上", "二人暮らししやすい間取り"],
    },
    "move_in_date": {
        "slot": "move_in_date",
        "label": "入居時期",
        "question": "いつ頃から住み始めたいですか？",
        "examples": ["来月中", "できるだけ早く", "2026-05ごろ"],
    },
    "must_conditions": {
        "slot": "must_conditions",
        "label": "必須条件",
        "question": "絶対に外せない条件があれば教えてください。",
        "examples": ["ペット可は必須", "独立洗面台は外せない", "二人入居可が必要"],
    },
    "nice_to_have": {
        "slot": "nice_to_have",
        "label": "あると良い条件",
        "question": "できれば欲しい条件はありますか？",
        "examples": ["在宅ワークしやすいと嬉しい", "浴室乾燥機があると助かる", "南向きだと理想"],
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
INTENT_VALUES = ("search", "risk_check", "general_question")


def _is_empty(value: Any) -> bool:
    return value in (None, "", [], {})


def _dedupe_texts(values: list[Any], *, limit: int | None = None) -> list[str]:
    deduped: list[str] = []
    for value in values:
        text = " ".join(str(value).split()).strip()
        if text and text not in deduped:
            deduped.append(text)
        if limit is not None and len(deduped) >= limit:
            break
    return deduped


def _coerce_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


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


def _normalize_slots(updates: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key in SEARCH_SIGNAL_KEYS:
        value = updates.get(key)
        if key in {"must_conditions", "nice_to_have"}:
            normalized[key] = _dedupe_texts(list(value or []), limit=6) if isinstance(value, list) else []
            continue
        if key in {"budget_max", "station_walk_max"}:
            coerced = _coerce_int(value)
            if coerced is not None:
                normalized[key] = coerced
            continue
        text = str(value).strip() if value is not None else ""
        if text:
            normalized[key] = text
    return normalized


def _merge_memory(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in _normalize_slots(updates).items():
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
        if not _is_empty(value):
            merged[key] = value
    return merged


def _summarize_profile_history(profile_memory: dict[str, Any] | None) -> dict[str, Any]:
    profile_memory = profile_memory or {}
    search_history = list(profile_memory.get("search_history", []) or [])[-3:]
    recent_searches: list[dict[str, str]] = []
    for entry in search_history:
        user_memory = entry.get("user_memory", {}) or {}
        tokens: list[str] = []
        area = str(user_memory.get("target_area") or "").strip()
        layout = str(user_memory.get("layout_preference") or "").strip()
        budget = _coerce_int(user_memory.get("budget_max")) or 0
        walk = _coerce_int(user_memory.get("station_walk_max")) or 0
        if area:
            tokens.append(area)
        if budget > 0:
            tokens.append(f"家賃{int(budget / 10000)}万円以下")
        if layout:
            tokens.append(layout)
        if walk > 0:
            tokens.append(f"徒歩{walk}分以内")
        recent_searches.append(
            {
                "query": str(entry.get("query") or "").strip(),
                "summary": " / ".join(tokens),
            }
        )
    reaction_history = list(profile_memory.get("reaction_history", []) or [])[-3:]
    recent_reactions = [
        {
            "reaction": str(entry.get("reaction") or "").strip(),
            "building_name": str(entry.get("building_name") or "").strip(),
            "area_name": str(entry.get("area_name") or "").strip(),
            "layout": str(entry.get("layout") or "").strip(),
        }
        for entry in reaction_history
    ]
    return {
        "recent_searches": recent_searches,
        "recent_reactions": recent_reactions,
    }


def _planner_schema() -> dict[str, Any]:
    slot_schema = {
        "type": "object",
        "properties": {
            "budget_max": {"type": ["integer", "null"]},
            "target_area": {"type": ["string", "null"]},
            "station_walk_max": {"type": ["integer", "null"]},
            "move_in_date": {"type": ["string", "null"]},
            "layout_preference": {"type": ["string", "null"]},
            "must_conditions": {"type": "array", "items": {"type": "string"}},
            "nice_to_have": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["must_conditions", "nice_to_have"],
        "additionalProperties": False,
    }
    follow_up_schema = {
        "type": "object",
        "properties": {
            "slot": {"type": "string", "enum": FOLLOW_UP_SLOT_ORDER + FOLLOW_UP_OPTIONAL_SLOTS},
            "question": {"type": "string"},
            "examples": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["slot", "question", "examples"],
        "additionalProperties": False,
    }
    condition_reason_schema = {
        "type": "object",
        "properties": {
            key: {"type": "string"}
            for key in list(SEARCH_SIGNAL_KEYS)
        },
        "required": list(SEARCH_SIGNAL_KEYS),
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "intent": {"type": "string", "enum": list(INTENT_VALUES)},
            "extracted_slots": slot_schema,
            "follow_up_questions": {"type": "array", "items": follow_up_schema},
            "seed_queries": {"type": "array", "items": {"type": "string"}},
            "research_plan": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "goal": {"type": "string"},
                    "strategy": {"type": "array", "items": {"type": "string"}},
                    "rationale": {"type": "string"},
                },
                "required": ["summary", "goal", "strategy", "rationale"],
                "additionalProperties": False,
            },
            "condition_reasons": condition_reason_schema,
        },
        "required": [
            "intent",
            "extracted_slots",
            "follow_up_questions",
            "seed_queries",
            "research_plan",
            "condition_reasons",
        ],
        "additionalProperties": False,
    }


def _llm_parse(
    message: str,
    user_memory: dict[str, Any],
    profile_memory: dict[str, Any] | None,
    adapter: LLMAdapter,
) -> dict[str, Any]:
    prompt_payload = {
        "user_message": message,
        "current_user_memory": user_memory,
        "learned_preferences": user_memory.get("learned_preferences", {}) or {},
        "profile_history_summary": _summarize_profile_history(profile_memory),
        "slot_reference": {
            "target_area": "探したいエリアや駅",
            "budget_max": "家賃上限（円）",
            "station_walk_max": "駅徒歩の上限（分）",
            "layout_preference": "希望間取り",
            "move_in_date": "入居時期。すぐなら asap",
            "must_conditions": "必須条件",
            "nice_to_have": "あると良い条件",
        },
        "output_rules": [
            "intent は search / risk_check / general_question のいずれかにする",
            "search: 部屋探し、比較、引っ越し先探し、住み替え相談",
            "risk_check: 契約条件、更新料、違約金、保証会社、退去費用などの確認",
            "general_question: 上記以外の相談や雑談",
            "『絶対』『必須』『譲れない』『〜じゃないとダメ』は must_conditions",
            "『できれば』『あったらいい』『理想』『〜だとうれしい』は nice_to_have",
            "どちらか不明な条件は nice_to_have に寄せ、既存 user_memory があれば整合させる",
            "follow_up_questions は不足・曖昧な条件だけを自然な日本語で 0〜5 件返す",
            "follow_up_questions はユーザーの語彙や状況を少し反映し、examples も文脈に合わせる",
            "seed_queries は 3〜5 件。短い検索語と自然文を混ぜ、明示条件と既存メモリからのみ作る",
            "research_plan はユーザー条件に即して summary / goal / strategy(3〜5項目) / rationale を返す",
            "condition_reasons は各条件が今回の検索で重要な理由を 1 文ずつ返し、該当しない key は空文字にする",
        ],
    }
    return adapter.generate_structured(
        system=(
            "You are a Japanese rental planner that prepares a personalized search plan. "
            "Understand the user's housing intent from the latest message plus saved memory. "
            "Return only structured data grounded in the provided context. "
            "Do not invent constraints, areas, budgets, or preferences not supported by the message or memory."
        ),
        user=json.dumps(prompt_payload, ensure_ascii=False, indent=2),
        schema=_planner_schema(),
        temperature=0.1,
    )


def _normalize_llm_output(payload: dict[str, Any]) -> dict[str, Any]:
    intent = str(payload.get("intent") or "").strip()
    if intent not in INTENT_VALUES:
        intent = ""

    extracted_slots = payload.get("extracted_slots", {})
    if not isinstance(extracted_slots, dict):
        extracted_slots = {}

    normalized_follow_ups: list[dict[str, Any]] = []
    raw_follow_ups = payload.get("follow_up_questions", [])
    if isinstance(raw_follow_ups, list):
        for item in raw_follow_ups:
            if not isinstance(item, dict):
                continue
            slot = str(item.get("slot") or "").strip()
            if slot not in DEFAULT_FOLLOW_UP_QUESTIONS:
                continue
            question = " ".join(str(item.get("question") or "").split()).strip()
            examples = _dedupe_texts(list(item.get("examples") or []), limit=3)
            if question:
                normalized_follow_ups.append(
                    {
                        "slot": slot,
                        "question": question,
                        "examples": examples,
                    }
                )

    normalized_research_plan = payload.get("research_plan", {})
    if not isinstance(normalized_research_plan, dict):
        normalized_research_plan = {}

    condition_reasons = payload.get("condition_reasons", {})
    if not isinstance(condition_reasons, dict):
        condition_reasons = {}

    return {
        "intent": intent,
        "extracted_slots": _normalize_slots(extracted_slots),
        "follow_up_questions": normalized_follow_ups,
        "seed_queries": _dedupe_texts(list(payload.get("seed_queries") or []), limit=5),
        "research_plan": {
            "summary": " ".join(str(normalized_research_plan.get("summary") or "").split()).strip(),
            "goal": " ".join(str(normalized_research_plan.get("goal") or "").split()).strip(),
            "strategy": _dedupe_texts(list(normalized_research_plan.get("strategy") or []), limit=5),
            "rationale": " ".join(str(normalized_research_plan.get("rationale") or "").split()).strip(),
        },
        "condition_reasons": {
            key: " ".join(str(condition_reasons.get(key) or "").split()).strip()
            for key in SEARCH_SIGNAL_KEYS
        },
    }


def _has_structured_search_signal(merged: dict[str, Any]) -> bool:
    return any(not _is_empty(merged.get(key)) for key in SEARCH_SIGNAL_KEYS)


def _has_search_readiness(merged: dict[str, Any]) -> bool:
    readiness_keys = (
        "target_area",
        "budget_max",
        "station_walk_max",
        "layout_preference",
        "must_conditions",
        "nice_to_have",
    )
    return any(not _is_empty(merged.get(key)) for key in readiness_keys)


def _is_generic_search_request(message: str) -> bool:
    normalized = re.sub(r"[\s　、。,.!！?？・]+", "", message)
    return any(pattern.fullmatch(normalized) for pattern in GENERIC_SEARCH_PATTERNS)


def _fallback_intent(
    message: str,
    message_slots: dict[str, Any] | None = None,
    user_memory: dict[str, Any] | None = None,
) -> str:
    message_slots = message_slots or _rule_based_slots(message)
    user_memory = user_memory or {}
    same_as_previous = any(token in message for token in ("前回と同じ", "前と同じ", "同じ条件"))
    if any(keyword in message for keyword in RISK_INTENT_KEYWORDS) and not _has_structured_search_signal(message_slots):
        return "risk_check"
    if _has_structured_search_signal(message_slots):
        return "search"
    if _is_generic_search_request(message):
        return "search"
    if same_as_previous and _has_structured_search_signal(user_memory):
        return "search"
    if any(keyword in message for keyword in SEARCH_INTENT_KEYWORDS):
        return "search"
    if any(keyword in message for keyword in RISK_INTENT_KEYWORDS):
        return "risk_check"
    return "general_question"


def detect_search_signal(message: str, planner_result: dict[str, Any] | None = None) -> bool:
    if planner_result is not None:
        return str(planner_result.get("intent") or "") == "search"
    return _fallback_intent(message) == "search"


def _build_default_follow_up(slot: str) -> dict[str, Any]:
    template = DEFAULT_FOLLOW_UP_QUESTIONS[slot]
    return {
        "slot": template["slot"],
        "label": template["label"],
        "question": template["question"],
        "examples": list(template["examples"]),
    }


def _build_follow_up_questions(
    merged: dict[str, Any],
    *,
    llm_questions: list[dict[str, Any]] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    llm_questions = llm_questions or []
    items: list[dict[str, Any]] = []
    seen_slots: set[str] = set()

    def should_include(slot: str) -> bool:
        return _is_empty(merged.get(slot))

    for item in llm_questions:
        slot = str(item.get("slot") or "").strip()
        if slot in seen_slots or slot not in DEFAULT_FOLLOW_UP_QUESTIONS or not should_include(slot):
            continue
        block = _build_default_follow_up(slot)
        question = " ".join(str(item.get("question") or "").split()).strip()
        if question:
            block["question"] = question
        examples = _dedupe_texts(list(item.get("examples") or []), limit=3)
        if examples:
            block["examples"] = examples
        items.append(block)
        seen_slots.add(slot)
        if limit is not None and len(items) >= limit:
            return items

    for slot in FOLLOW_UP_SLOT_ORDER + FOLLOW_UP_OPTIONAL_SLOTS:
        if slot in seen_slots or not should_include(slot):
            continue
        items.append(_build_default_follow_up(slot))
        if limit is not None and len(items) >= limit:
            break
    return items


def _build_seed_queries(merged: dict[str, Any], fallback_message: str) -> list[str]:
    area = str(merged.get("target_area") or "").strip()
    layout = str(merged.get("layout_preference") or "").strip()
    budget = _coerce_int(merged.get("budget_max")) or 0
    walk = _coerce_int(merged.get("station_walk_max")) or 0
    move_in = str(merged.get("move_in_date") or "").strip()
    must_conditions = [str(item).strip() for item in merged.get("must_conditions", []) or [] if str(item).strip()]
    nice_to_have = [str(item).strip() for item in merged.get("nice_to_have", []) or [] if str(item).strip()]

    keyword_tokens = [area, "賃貸"]
    if budget > 0:
        keyword_tokens.append(f"{int(budget / 10000)}万円")
    if layout:
        keyword_tokens.append(layout)
    if walk > 0:
        keyword_tokens.append(f"徒歩{walk}分")
    if must_conditions:
        keyword_tokens.extend(must_conditions[:2])

    natural_tokens = [token for token in [area, layout] if token]
    natural_phrase = ""
    if natural_tokens:
        natural_phrase = "".join(natural_tokens)
        if budget > 0:
            natural_phrase += f"で家賃{int(budget / 10000)}万円以内"
        if walk > 0:
            natural_phrase += f"、駅徒歩{walk}分以内"
        if must_conditions:
            natural_phrase += f"、{'・'.join(must_conditions[:2])}"
        natural_phrase += "の賃貸"

    candidates = [
        " ".join(token for token in keyword_tokens if token).strip(),
        natural_phrase,
    ]
    if area and must_conditions:
        candidates.append(f"{area} {' '.join(must_conditions[:2])} 賃貸")
    if area and nice_to_have:
        candidates.append(f"{area} {' '.join(nice_to_have[:2])} 賃貸")
    if area and move_in == "asap":
        candidates.append(f"{area} すぐ入居できる 賃貸")
    elif area and move_in:
        candidates.append(f"{area} {move_in} 入居 賃貸")
    candidates.append(fallback_message)
    return _dedupe_texts(candidates, limit=5)


def _build_plan_summary(merged: dict[str, Any]) -> str:
    summary_tokens: list[str] = []
    if merged.get("target_area"):
        summary_tokens.append(str(merged["target_area"]))
    if merged.get("budget_max"):
        summary_tokens.append(f"家賃{int(int(merged['budget_max']) / 10000)}万円以下")
    if merged.get("layout_preference"):
        summary_tokens.append(str(merged["layout_preference"]))
    if merged.get("station_walk_max"):
        summary_tokens.append(f"徒歩{merged['station_walk_max']}分以内")
    if merged.get("must_conditions"):
        summary_tokens.append(" / ".join(str(item) for item in merged["must_conditions"][:2]))
    return " / ".join(summary_tokens[:4]) if summary_tokens else "条件整理から調査を開始"


def _build_fallback_research_plan(
    merged: dict[str, Any],
    *,
    follow_up_questions: list[dict[str, Any]],
    fallback_message: str,
) -> dict[str, Any]:
    strategy = [
        "希望条件を軸に複数クエリへ展開して候補を広めに収集します。",
        "詳細ページを優先して読み、表記ゆれと重複掲載を整理します。",
        "条件一致度と不足情報を比較し、問い合わせ向きの候補を上位化します。",
    ]
    return {
        "summary": _build_plan_summary(merged),
        "goal": "条件に近い候補を比較し、問い合わせに進める物件を絞り込む",
        "strategy": strategy,
        "rationale": (
            "先に候補集合を作ってから詳細情報と不足情報を見比べると、"
            "問い合わせ前に条件差分を整理しやすいためです。"
        ),
        "seed_queries": _build_seed_queries(merged, fallback_message),
        "open_questions": [
            str(item.get("question") or "").strip()
            for item in follow_up_questions
            if str(item.get("question") or "").strip()
        ],
    }


def _build_default_condition_reasons(merged: dict[str, Any]) -> dict[str, str]:
    reasons = {key: "" for key in SEARCH_SIGNAL_KEYS}
    if merged.get("target_area"):
        reasons["target_area"] = "エリアが候補母集団を大きく左右するため、検索の軸として優先します。"
    if merged.get("budget_max"):
        reasons["budget_max"] = "毎月の支払上限なので、候補比較でも厳格に確認します。"
    if merged.get("station_walk_max"):
        reasons["station_walk_max"] = "通勤や生活動線に直結するため、移動負担の上限として見ます。"
    if merged.get("layout_preference"):
        reasons["layout_preference"] = "暮らし方に合う居室数かを見極めるため、間取りを優先条件に置きます。"
    if merged.get("move_in_date"):
        reasons["move_in_date"] = "入居可能時期が合わない候補を早めに外すためです。"
    if merged.get("must_conditions"):
        reasons["must_conditions"] = "外せない条件なので、候補数より一致度を優先して確認します。"
    if merged.get("nice_to_have"):
        reasons["nice_to_have"] = "候補を並べたあとに差が出やすい加点条件として扱います。"
    return reasons


def run_planner(
    *,
    message: str,
    user_memory: dict[str, Any],
    adapter: LLMAdapter | None,
    profile_memory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    llm_output = {
        "intent": "",
        "extracted_slots": {},
        "follow_up_questions": [],
        "seed_queries": [],
        "research_plan": {"summary": "", "goal": "", "strategy": [], "rationale": ""},
        "condition_reasons": {key: "" for key in SEARCH_SIGNAL_KEYS},
    }
    merged = dict(user_memory)
    message_slots = _rule_based_slots(message)

    if adapter is not None:
        try:
            llm_output = _normalize_llm_output(
                _llm_parse(message, user_memory, profile_memory, adapter)
            )
            merged = _merge_memory(user_memory, llm_output["extracted_slots"])
            for key, value in message_slots.items():
                if _is_empty(merged.get(key)):
                    merged = _merge_memory(merged, {key: value})
        except Exception:
            merged = _merge_memory(user_memory, message_slots)
    else:
        merged = _merge_memory(user_memory, message_slots)

    intent = llm_output["intent"] or _fallback_intent(message, message_slots, user_memory)
    readiness = _has_search_readiness(merged)
    follow_up_questions = _build_follow_up_questions(
        merged,
        llm_questions=llm_output["follow_up_questions"],
        limit=3 if intent == "search" and not readiness else None,
    )

    if intent == "search" and not readiness:
        missing_slots = [item["slot"] for item in follow_up_questions[:3]]
        next_action = "missing_slots_question"
    elif intent == "search":
        missing_slots = []
        next_action = "search_and_compare"
    elif intent == "risk_check":
        missing_slots = []
        next_action = "risk_check"
    else:
        missing_slots = []
        next_action = "guidance"

    plan_summary = {
        "budget_max": merged.get("budget_max"),
        "target_area": merged.get("target_area"),
        "station_walk_max": merged.get("station_walk_max"),
        "move_in_date": merged.get("move_in_date"),
        "layout_preference": merged.get("layout_preference"),
    }

    fallback_plan = _build_fallback_research_plan(
        merged,
        follow_up_questions=follow_up_questions,
        fallback_message=message,
    )
    research_plan = {
        "summary": llm_output["research_plan"]["summary"] or fallback_plan["summary"],
        "goal": llm_output["research_plan"]["goal"] or fallback_plan["goal"],
        "strategy": llm_output["research_plan"]["strategy"] or fallback_plan["strategy"],
        "rationale": llm_output["research_plan"]["rationale"] or fallback_plan["rationale"],
        "open_questions": fallback_plan["open_questions"],
    }
    seed_queries = llm_output["seed_queries"] or fallback_plan["seed_queries"]
    condition_reasons = _build_default_condition_reasons(merged)
    for key, value in llm_output["condition_reasons"].items():
        if value:
            condition_reasons[key] = value

    return {
        "intent": intent,
        "plan": plan_summary,
        "missing_slots": missing_slots,
        "follow_up_questions": follow_up_questions,
        "next_action": next_action,
        "user_memory": merged,
        "seed_queries": seed_queries,
        "research_plan": research_plan,
        "condition_reasons": condition_reasons,
    }
