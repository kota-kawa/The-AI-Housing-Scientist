from __future__ import annotations

import json
import re
from typing import Any

from app.llm.base import LLMAdapter
from app.stages.prompt_examples import PromptExamplesError, sample_prompt_examples

FOLLOW_UP_SLOT_ORDER = [
    "listing_type",
    "target_area",
    "budget_max",
    "layout_preference",
    "station_walk_max",
    "move_in_date",
]
FOLLOW_UP_OPTIONAL_SLOTS = ["must_conditions", "nice_to_have"]
REQUIRED_PLANNING_SLOTS = ("listing_type", "target_area", "budget_max")
SEARCH_SIGNAL_KEYS = (
    "budget_max",
    "target_area",
    "station_walk_max",
    "move_in_date",
    "layout_preference",
    "listing_type",
    "must_conditions",
    "nice_to_have",
)
ALL_SLOT_KEYS = list(FOLLOW_UP_SLOT_ORDER) + list(FOLLOW_UP_OPTIONAL_SLOTS)
INTENT_VALUES = ("search", "risk_check", "general_question")
NEXT_ACTION_VALUES = (
    "missing_slots_question",
    "search_and_compare",
    "risk_check",
    "guidance",
)
TEXT_SLOT_KEYS = {"target_area", "move_in_date", "layout_preference", "listing_type"}
INTEGER_SLOT_KEYS = {"budget_max", "station_walk_max"}
LIST_SLOT_KEYS = {"must_conditions", "nice_to_have"}
QUESTION_SLOT_ORDER = list(REQUIRED_PLANNING_SLOTS) + [
    "layout_preference",
    "station_walk_max",
    "move_in_date",
    "must_conditions",
    "nice_to_have",
]

PLANNING_SLOT_LABELS = {
    "listing_type": "物件種別",
    "target_area": "希望エリア",
    "budget_max": "予算上限",
    "layout_preference": "間取り",
    "station_walk_max": "駅徒歩",
    "move_in_date": "入居時期",
    "must_conditions": "必須条件",
    "nice_to_have": "あると良い条件",
}
PLANNING_SLOT_QUESTIONS = {
    "listing_type": "まずは賃貸か売買かを選んでください。",
    "target_area": "どのエリアや駅を中心に探したいですか？",
    "budget_max": "予算の上限はどれくらいですか？",
    "layout_preference": "希望の間取りがあれば教えてください。未定なら「こだわらない」でも大丈夫です。",
    "station_walk_max": "駅からの距離はどの程度を優先しますか？",
    "move_in_date": "いつ頃から住み始めたいですか？",
    "must_conditions": "外せない条件があれば教えてください。",
    "nice_to_have": "あればうれしい条件があれば教えてください。",
}
PLANNING_SLOT_EXAMPLES = {
    "listing_type": ["賃貸", "売買"],
    "budget_max": ["8万円まで", "10万円まで", "12万円まで", "15万円まで", "20万円まで"],
    "layout_preference": [
        "こだわらない",
        "ワンルーム",
        "1K",
        "1DK",
        "1LDK",
        "2K/2DK",
        "2LDK",
        "3LDK+",
    ],
    "station_walk_max": ["徒歩5分以内", "徒歩7分以内", "徒歩10分以内", "徒歩15分以内"],
    "move_in_date": ["できるだけ早く", "来月中", "2-3ヶ月以内"],
}
PLANNING_SLOT_PLACEHOLDERS = {
    "listing_type": "賃貸 / 売買",
    "target_area": "例: 中野 / 吉祥寺 / 横浜駅周辺",
    "budget_max": "例: 12万円まで / 4000万円まで",
    "layout_preference": "例: こだわらない / 1LDK / 2LDK",
    "station_walk_max": "例: 徒歩10分以内",
    "move_in_date": "例: 来月中 / できるだけ早く",
    "must_conditions": "例: 2階以上 / ペット可 / RC造",
    "nice_to_have": "例: 独立洗面台 / 在宅ワーク向け",
}
PLANNING_SLOT_INPUT_KIND = {
    "listing_type": "single_choice_text",
    "target_area": "single_choice_text",
    "budget_max": "single_choice_text",
    "layout_preference": "single_choice_text",
    "station_walk_max": "single_choice_text",
    "move_in_date": "single_choice_text",
    "must_conditions": "text",
    "nice_to_have": "text",
}
PLANNING_SLOT_KEYBOARD_HINT = {
    "listing_type": "default",
    "target_area": "default",
    "budget_max": "numeric",
    "layout_preference": "default",
    "station_walk_max": "numeric",
    "move_in_date": "default",
    "must_conditions": "default",
    "nice_to_have": "default",
}


# JP: textを正規化する。
# EN: Normalize text.
def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


# JP: dedupe textsを処理する。
# EN: Process dedupe texts.
def _dedupe_texts(values: list[Any], *, limit: int | None = None) -> list[str]:
    deduped: list[str] = []
    for value in values:
        text = _normalize_text(value)
        if text and text not in deduped:
            deduped.append(text)
        if limit is not None and len(deduped) >= limit:
            break
    return deduped


# JP: 複数選択UIから来た区切り文字付きテキストを分割する。
# EN: Split delimited text submitted by multi-select UI chips.
def _split_answer_tokens(value: Any) -> list[str]:
    if isinstance(value, list):
        return _dedupe_texts(value)
    text = _normalize_text(value)
    if not text:
        return []
    return _dedupe_texts(re.split(r"\s*(?:、|,|/|／|\n)\s*", text))


# JP: slot labelを取得する。
# EN: Get slot label.
def _slot_label(slot: str) -> str:
    return PLANNING_SLOT_LABELS.get(slot, slot)


# JP: listing typeを正規化する。
# EN: Normalize listing type.
def _normalize_listing_type(value: Any) -> str | None:
    text = _normalize_text(value)
    if not text:
        return None
    lower = text.lower()
    if any(token in text for token in ["賃貸", "家賃", "借り", "賃借"]) or "rent" in lower:
        return "賃貸"
    if any(token in text for token in ["売買", "購入", "買う", "分譲"]) or "buy" in lower:
        return "売買"
    return text


# JP: budget valueを解析する。
# EN: Parse budget value.
def _parse_budget_value(value: Any) -> int | None:
    text = _normalize_text(value)
    if not text:
        return None
    compact = text.replace(",", "").replace("，", "")
    if compact.isdigit():
        amount = int(compact)
        return amount if amount > 0 else None

    match = None
    if "万" in compact:
        import re

        match = re.search(r"(\d+(?:\.\d+)?)\s*万", compact)
        if match:
            return int(float(match.group(1)) * 10000)
    return None


# JP: station walkを解析する。
# EN: Parse station walk.
def _parse_station_walk_value(value: Any) -> int | None:
    text = _normalize_text(value)
    if not text:
        return None
    compact = text.replace("以内", "").replace("まで", "").replace("駅徒歩", "").replace("徒歩", "")
    compact = compact.replace("分", "").strip()
    if compact.isdigit():
        minutes = int(compact)
        return minutes if minutes > 0 else None
    return None


# JP: slot valueを正規化する。
# EN: Normalize slot value.
def _normalize_slot_value(slot: str, value: Any) -> Any:
    if slot == "listing_type":
        return _normalize_listing_type(value)
    if slot == "layout_preference":
        text = _normalize_text(value)
        if text in {"こだわらない", "指定なし", "未定", "おまかせ"}:
            return None
        return text or None
    if slot == "budget_max":
        return _parse_budget_value(value)
    if slot == "station_walk_max":
        return _parse_station_walk_value(value)
    if slot in {"must_conditions", "nice_to_have"}:
        return _split_answer_tokens(value)[:6]
    text = _normalize_text(value)
    return text or None


# JP: slot valueが設定済みか判定する。
# EN: Check whether slot value is present.
def _has_slot_value(slot: str, user_memory: dict[str, Any]) -> bool:
    value = user_memory.get(slot)
    if slot in LIST_SLOT_KEYS:
        return bool(value)
    return value is not None and _normalize_text(value) != ""


# JP: slot memoryをマージする。
# EN: Merge slot memory.
def _merge_slot_memory(
    base_memory: dict[str, Any], override_memory: dict[str, Any]
) -> dict[str, Any]:
    merged = _sanitize_slot_memory(base_memory)
    override = _sanitize_slot_memory(override_memory)
    for key in TEXT_SLOT_KEYS:
        if override.get(key):
            merged[key] = override.get(key)
    for key in INTEGER_SLOT_KEYS:
        if override.get(key) is not None:
            merged[key] = override.get(key)
    for key in LIST_SLOT_KEYS:
        if override.get(key):
            merged[key] = list(override.get(key) or [])
    return merged


# JP: planner answersを適用する。
# EN: Apply planner answers.
def _apply_planner_answers(
    user_memory: dict[str, Any],
    planner_answers: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    merged = _sanitize_slot_memory(user_memory)
    for item in planner_answers or []:
        if not isinstance(item, dict):
            continue
        slot = _normalize_text(item.get("slot"))
        if slot not in ALL_SLOT_KEYS and slot != "listing_type":
            continue
        normalized = _normalize_slot_value(slot, item.get("value"))
        if slot in LIST_SLOT_KEYS:
            merged[slot] = normalized if isinstance(normalized, list) else []
        elif slot in INTEGER_SLOT_KEYS:
            merged[slot] = normalized if isinstance(normalized, int) else None
        else:
            merged[slot] = normalized
    return merged


# JP: planner answersを要約する。
# EN: Summarize planner answers.
def _summarize_planner_answers(planner_answers: list[dict[str, Any]] | None) -> str:
    tokens: list[str] = []
    for item in planner_answers or []:
        if not isinstance(item, dict):
            continue
        slot = _normalize_text(item.get("slot"))
        value = _normalize_text(item.get("value"))
        if not slot or not value:
            continue
        tokens.append(f"{_slot_label(slot)}は{value}")
    return "、".join(tokens)


# JP: planner messageを構築する。
# EN: Build planner message.
def _build_planner_message(message: str, planner_answers: list[dict[str, Any]] | None) -> str:
    normalized_message = _normalize_text(message)
    answers_summary = _summarize_planner_answers(planner_answers)
    if normalized_message and answers_summary:
        return f"{normalized_message}\n{answers_summary}"
    return normalized_message or answers_summary


# JP: search intentを推定する。
# EN: Infer search intent.
def _infer_intent(
    message: str, user_memory: dict[str, Any], planner_answers: list[dict[str, Any]] | None
) -> str:
    if planner_answers:
        return "search"
    listing_type = _normalize_text(user_memory.get("listing_type"))
    if listing_type in {"賃貸", "売買"}:
        return "search"
    normalized_message = _normalize_text(message)
    if any(
        token in normalized_message
        for token in ["探したい", "部屋", "物件", "賃貸", "売買", "購入", "住みたい"]
    ):
        return "search"
    return "general_question"


# JP: required missing slotsを取得する。
# EN: Get required missing slots.
def _required_missing_slots(user_memory: dict[str, Any]) -> list[str]:
    return [slot for slot in REQUIRED_PLANNING_SLOTS if not _has_slot_value(slot, user_memory)]


# JP: budget tokenを整形する。
# EN: Format budget token.
def _format_budget_token(value: Any) -> str:
    amount = _parse_budget_value(value)
    if amount is None or amount <= 0:
        return ""
    if amount % 10000 == 0:
        return f"{int(amount / 10000)}万円以下"
    return f"{amount:,}円以下"


# JP: base seed queriesを構築する。
# EN: Build base seed queries.
def _build_base_seed_queries(user_memory: dict[str, Any]) -> list[str]:
    area = _normalize_text(user_memory.get("target_area"))
    listing_type = _normalize_text(user_memory.get("listing_type"))
    budget = _format_budget_token(user_memory.get("budget_max"))
    layout = _normalize_text(user_memory.get("layout_preference"))
    must_condition = " ".join(
        _dedupe_texts(list(user_memory.get("must_conditions") or []), limit=1)
    )
    nice_condition = " ".join(_dedupe_texts(list(user_memory.get("nice_to_have") or []), limit=1))

    candidates = _dedupe_texts(
        [
            " ".join(part for part in [area, listing_type, budget, layout, must_condition] if part),
            " ".join(part for part in [area, budget, layout, listing_type] if part),
            " ".join(part for part in [area, layout, listing_type, nice_condition] if part),
            " ".join(part for part in [area, listing_type, layout] if part),
            " ".join(part for part in [area, listing_type, budget] if part),
        ],
        limit=5,
    )
    return [item for item in candidates if item]


# JP: default research planを構築する。
# EN: Build default research plan.
def _default_research_plan(user_memory: dict[str, Any]) -> dict[str, Any]:
    area = _normalize_text(user_memory.get("target_area"))
    layout = _normalize_text(user_memory.get("layout_preference"))
    listing_type = _normalize_text(user_memory.get("listing_type")) or "物件"
    budget = _format_budget_token(user_memory.get("budget_max"))
    summary_tokens = [token for token in [area, budget, layout, listing_type] if token]
    summary = " / ".join(summary_tokens) if summary_tokens else "条件に合う候補を比較します。"
    return {
        "summary": f"{summary}を軸に候補を集めます。"
        if summary_tokens
        else "条件に合う候補を比較します。",
        "goal": "条件に近い候補を比較し、問い合わせや次の確認に進める物件を絞り込みます。",
        "strategy": [
            "必須条件に合う候補を広めに集めて比較の土台を作ります。",
            "詳細ページで不足情報や表記ゆれを確認します。",
            "条件一致度と確認事項を見ながら候補を並べ替えます。",
        ],
        "rationale": "検索を広げすぎず、比較に必要な情報を先に揃えるためです。",
    }


# JP: default condition reasonsを構築する。
# EN: Build default condition reasons.
def _default_condition_reasons(user_memory: dict[str, Any]) -> dict[str, str]:
    reasons = _blank_condition_reasons()
    if _has_slot_value("listing_type", user_memory):
        reasons["listing_type"] = "賃貸か売買かで見るべき候補が大きく変わるためです。"
    if _has_slot_value("target_area", user_memory):
        reasons["target_area"] = "生活圏を固定して候補を絞り込むためです。"
    if _has_slot_value("budget_max", user_memory):
        reasons["budget_max"] = "予算超過の候補を早めに外すためです。"
    if _has_slot_value("layout_preference", user_memory):
        reasons["layout_preference"] = "住み方に合う間取りを優先するためです。"
    if _has_slot_value("station_walk_max", user_memory):
        reasons["station_walk_max"] = "移動負担に直結する条件だからです。"
    if _has_slot_value("move_in_date", user_memory):
        reasons["move_in_date"] = "募集タイミングが合う候補を優先するためです。"
    if _has_slot_value("must_conditions", user_memory):
        reasons["must_conditions"] = "外せない条件なので優先的に確認するためです。"
    if _has_slot_value("nice_to_have", user_memory):
        reasons["nice_to_have"] = "候補の差分を比較する補助軸にするためです。"
    return reasons


# JP: blank slot memoryを処理する。
# EN: Process blank slot memory.
def _blank_slot_memory() -> dict[str, Any]:
    return {
        "budget_max": None,
        "target_area": None,
        "station_walk_max": None,
        "move_in_date": None,
        "layout_preference": None,
        "listing_type": None,
        "must_conditions": [],
        "nice_to_have": [],
    }


# JP: blank research planを処理する。
# EN: Process blank research plan.
def _blank_research_plan() -> dict[str, Any]:
    return {"summary": "", "goal": "", "strategy": [], "rationale": ""}


# JP: blank condition reasonsを処理する。
# EN: Process blank condition reasons.
def _blank_condition_reasons() -> dict[str, str]:
    return dict.fromkeys(SEARCH_SIGNAL_KEYS, "")


# JP: sanitize slot memoryを処理する。
# EN: Process sanitize slot memory.
def _sanitize_slot_memory(raw_memory: Any) -> dict[str, Any]:
    memory = _blank_slot_memory()
    if not isinstance(raw_memory, dict):
        return memory

    for key in TEXT_SLOT_KEYS:
        text = _normalize_text(raw_memory.get(key))
        memory[key] = text or None

    for key in INTEGER_SLOT_KEYS:
        value = raw_memory.get(key)
        memory[key] = value if isinstance(value, int) and not isinstance(value, bool) else None

    for key in LIST_SLOT_KEYS:
        value = raw_memory.get(key)
        memory[key] = _dedupe_texts(list(value), limit=6) if isinstance(value, list) else []

    return memory


# JP: sanitize missing slotsを処理する。
# EN: Process sanitize missing slots.
def _sanitize_missing_slots(raw_missing_slots: Any) -> list[str]:
    return [
        slot
        for slot in _dedupe_texts(list(raw_missing_slots or []), limit=4)
        if slot in ALL_SLOT_KEYS
    ]


# JP: sanitize follow up questionsを処理する。
# EN: Process sanitize follow up questions.
def _sanitize_follow_up_questions(raw_questions: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_questions, list):
        return []

    normalized: list[dict[str, Any]] = []
    seen_slots: set[str] = set()
    for item in raw_questions:
        if not isinstance(item, dict):
            continue
        slot = _normalize_text(item.get("slot"))
        if slot not in ALL_SLOT_KEYS or slot in seen_slots:
            continue
        question = _normalize_text(item.get("question"))
        if not question:
            continue
        normalized.append(
            {
                "slot": slot,
                "label": _normalize_text(item.get("label")),
                "question": question,
                "examples": _dedupe_texts(list(item.get("examples") or []), limit=3),
            }
        )
        seen_slots.add(slot)
    return normalized


# JP: sanitize research planを処理する。
# EN: Process sanitize research plan.
def _sanitize_research_plan(raw_plan: Any) -> dict[str, Any]:
    raw_plan = raw_plan if isinstance(raw_plan, dict) else {}
    return {
        "summary": _normalize_text(raw_plan.get("summary")),
        "goal": _normalize_text(raw_plan.get("goal")),
        "strategy": _dedupe_texts(list(raw_plan.get("strategy") or []), limit=5),
        "rationale": _normalize_text(raw_plan.get("rationale")),
    }


# JP: sanitize condition reasonsを処理する。
# EN: Process sanitize condition reasons.
def _sanitize_condition_reasons(raw_reasons: Any) -> dict[str, str]:
    reasons = _blank_condition_reasons()
    if not isinstance(raw_reasons, dict):
        return reasons
    for key in SEARCH_SIGNAL_KEYS:
        reasons[key] = _normalize_text(raw_reasons.get(key))
    return reasons


# JP: empty planner resultを処理する。
# EN: Process empty planner result.
def _empty_planner_result(user_memory: dict[str, Any]) -> dict[str, Any]:
    merged_memory = _sanitize_slot_memory(user_memory)
    return {
        "intent": "general_question",
        "plan": {
            "budget_max": merged_memory.get("budget_max"),
            "target_area": merged_memory.get("target_area"),
            "station_walk_max": merged_memory.get("station_walk_max"),
            "move_in_date": merged_memory.get("move_in_date"),
            "layout_preference": merged_memory.get("layout_preference"),
            "listing_type": merged_memory.get("listing_type"),
        },
        "missing_slots": [],
        "follow_up_questions": [],
        "next_action": "guidance",
        "user_memory": merged_memory,
        "seed_queries": [],
        "research_plan": _blank_research_plan(),
        "condition_reasons": _blank_condition_reasons(),
    }


# JP: safe intを処理する。
# EN: Process safe int.
def _safe_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


# JP: profile historyを要約する。
# EN: Summarize profile history.
def _summarize_profile_history(profile_memory: dict[str, Any] | None) -> dict[str, Any]:
    profile_memory = profile_memory or {}
    search_history = list(profile_memory.get("search_history", []) or [])[-3:]
    recent_searches: list[dict[str, str]] = []
    for entry in search_history:
        user_memory = entry.get("user_memory", {}) or {}
        tokens: list[str] = []
        area = _normalize_text(user_memory.get("target_area"))
        layout = _normalize_text(user_memory.get("layout_preference"))
        budget = _safe_int(user_memory.get("budget_max"))
        walk = _safe_int(user_memory.get("station_walk_max"))
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
                "query": _normalize_text(entry.get("query")),
                "summary": " / ".join(tokens),
            }
        )
    reaction_history = list(profile_memory.get("reaction_history", []) or [])[-3:]
    recent_reactions = [
        {
            "reaction": _normalize_text(entry.get("reaction")),
            "building_name": _normalize_text(entry.get("building_name")),
            "area_name": _normalize_text(entry.get("area_name")),
            "layout": _normalize_text(entry.get("layout")),
        }
        for entry in reaction_history
    ]
    return {
        "recent_searches": recent_searches,
        "recent_reactions": recent_reactions,
    }


# JP: planner schemaを処理する。
# EN: Process planner schema.
def _planner_schema() -> dict[str, Any]:
    slot_schema = {
        "type": "object",
        "properties": {
            "budget_max": {"type": ["integer", "null"]},
            "target_area": {"type": ["string", "null"]},
            "station_walk_max": {"type": ["integer", "null"]},
            "move_in_date": {"type": ["string", "null"]},
            "layout_preference": {"type": ["string", "null"]},
            "listing_type": {"type": ["string", "null"]},
            "must_conditions": {"type": "array", "items": {"type": "string"}},
            "nice_to_have": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "budget_max",
            "target_area",
            "station_walk_max",
            "move_in_date",
            "layout_preference",
            "listing_type",
            "must_conditions",
            "nice_to_have",
        ],
        "additionalProperties": False,
    }
    follow_up_schema = {
        "type": "object",
        "properties": {
            "slot": {"type": "string", "enum": ALL_SLOT_KEYS},
            "label": {"type": "string"},
            "question": {"type": "string"},
            "examples": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["slot", "label", "question", "examples"],
        "additionalProperties": False,
    }
    condition_reason_schema = {
        "type": "object",
        "properties": {key: {"type": "string"} for key in SEARCH_SIGNAL_KEYS},
        "required": list(SEARCH_SIGNAL_KEYS),
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "intent": {"type": "string", "enum": list(INTENT_VALUES)},
            "user_memory": slot_schema,
            "missing_slots": {
                "type": "array",
                "items": {"type": "string", "enum": ALL_SLOT_KEYS},
            },
            "follow_up_questions": {"type": "array", "items": follow_up_schema},
            "next_action": {"type": "string", "enum": list(NEXT_ACTION_VALUES)},
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
            "user_memory",
            "missing_slots",
            "follow_up_questions",
            "next_action",
            "seed_queries",
            "research_plan",
            "condition_reasons",
        ],
        "additionalProperties": False,
    }


# JP: LLM parseを処理する。
# EN: Process LLM parse.
def _llm_parse(
    message: str,
    user_memory: dict[str, Any],
    profile_memory: dict[str, Any] | None,
    adapter: LLMAdapter,
) -> dict[str, Any]:
    planner_examples = sample_prompt_examples("planner_examples.json", count=2)
    prompt_payload = {
        "user_message": message,
        "current_user_memory": user_memory,
        "learned_preferences": user_memory.get("learned_preferences", {}) or {},
        "profile_history_summary": _summarize_profile_history(profile_memory),
        "slot_reference": {
            "target_area": {"label": "希望エリア", "meaning": "探したいエリアや駅"},
            "budget_max": {"label": "予算上限", "meaning": "家賃または購入予算の上限（円）"},
            "station_walk_max": {"label": "駅徒歩", "meaning": "駅徒歩の上限（分）"},
            "layout_preference": {"label": "間取り", "meaning": "希望間取り"},
            "move_in_date": {"label": "入居時期", "meaning": "すぐなら asap"},
            "must_conditions": {"label": "必須条件", "meaning": "外せない条件"},
            "nice_to_have": {"label": "あると良い条件", "meaning": "できれば欲しい条件"},
            "listing_type": {
                "label": "物件種別",
                "meaning": "賃貸 or 売買（ユーザーの意図から判定、明示なければ null）",
            },
        },
        "decision_rules": [
            "最新メッセージと current_user_memory を統合した user_memory を返す",
            "既存条件は、ユーザーが明確に上書き・否定した場合だけ消す",
            "intent は search / risk_check / general_question のいずれかにする",
            "search では、実際に候補収集を始めるべきなら next_action を search_and_compare にする",
            "search だが曖昧すぎて候補収集の前に確認が必要なら next_action を missing_slots_question にする",
            "missing_slots は 0〜3 件で、候補収集に必要な物件種別・エリア・予算だけにする",
            "layout_preference は任意条件として扱い、未定・こだわらない場合は missing_slots に入れない",
            "next_action が missing_slots_question の follow_up_questions は missing_slots と同じ順序・同じ slot だけを返す",
            "next_action が search_and_compare の follow_up_questions は任意で追加できる条件だけを返す",
            "follow_up_questions の label は slot_reference の日本語ラベルに合わせる",
            "follow_up_questions の question は、例にない回答や自由入力でも答えやすい聞き方にする",
            "follow_up_questions の examples は候補の例示であり、網羅的な選択肢として扱わない",
            "examples は user_message や memory にない特定の地域・予算・条件へ誘導しない",
            "examples は固定候補に見えにくいよう、粒度や表現を少し分散させてよい",
            "next_action が search_and_compare のときは seed_queries を 3〜5 件返す",
            "seed_queries は current_user_memory と今回の user_message を根拠に生成する",
            "seed_queries に profile_history_summary のエリア・条件は含めない（user_message に同じエリアが明示されている場合を除く）",
            "seed_queries は基本クエリだけに留め、近隣エリアや沿線の拡張は入れない",
            "next_action が missing_slots_question のときは seed_queries を空にしてよい",
            "research_plan はユーザー条件に即して summary / goal / strategy / rationale を返す",
            "condition_reasons は各条件が今回の検索で重要な理由を 1 文ずつ返し、該当しない key は空文字にする",
            "『絶対』『必須』『譲れない』『〜じゃないとダメ』は must_conditions",
            "『できれば』『あったらいい』『理想』『〜だとうれしい』は nice_to_have",
            "地名や駅名は『町田』『三軒茶屋』『武蔵小杉』のように接尾辞がなくても target_area に入れる",
            "RC / SRC / 鉄筋コンクリート / 鉄骨 / 木造 など建物構造も条件として扱う",
            "listing_type はユーザーのメッセージから物件種別を判定する。「賃貸」「家賃」「借りる」等は '賃貸'、「購入」「売買」「買う」「分譲」等は '売買' にする。明示がなければ null",
            "与えられたメッセージや memory にない制約・設備は発明しない",
        ],
        "examples_instruction": (
            "examples は input から output を作る完全な few-shot 見本です。"
            "slot 抽出、memory 統合、missing_slots、follow_up_questions、seed_queries、"
            "research_plan、condition_reasons の粒度を合わせてください。"
            "ただし follow_up_questions.examples は非網羅の例示であり、"
            "ユーザーをその候補群に固定してはいけません。"
        ),
        "examples": planner_examples,
    }
    return adapter.generate_structured(
        system=(
            "You are a Japanese property search planner responsible for the entire planning decision. "
            "Infer intent, merge memory, choose the next action, decide which conditions are missing, "
            "write follow-up questions, generate seed queries, and draft the research plan. "
            "Detect the listing type (賃貸/売買) from the user message and set listing_type accordingly. "
            "Treat follow-up examples as non-exhaustive hints, never as exhaustive options. "
            "Return only structured data grounded in the provided message and memory."
        ),
        user=json.dumps(prompt_payload, ensure_ascii=False, indent=2),
        schema=_planner_schema(),
        temperature=0.1,
    )


# JP: planner outputを解析する。
# EN: Parse planner output.
def _parse_planner_output(
    payload: dict[str, Any], *, default_user_memory: dict[str, Any]
) -> dict[str, Any]:
    intent = _normalize_text(payload.get("intent"))
    if intent not in INTENT_VALUES:
        intent = "general_question"

    merged_memory = _merge_slot_memory(default_user_memory, payload.get("user_memory") or {})
    llm_returned_empty = not any(
        [
            merged_memory.get("listing_type"),
            merged_memory.get("budget_max") is not None,
            merged_memory.get("target_area"),
            merged_memory.get("station_walk_max") is not None,
            merged_memory.get("move_in_date"),
            merged_memory.get("layout_preference"),
            merged_memory.get("must_conditions"),
            merged_memory.get("nice_to_have"),
        ]
    )
    # LLM が空の user_memory を返した場合、intent=="search" のときのみ
    # 前回条件を引き継ぐ（「もう一度検索」ユースケース）。
    # general_question 等では引き継がず空のまま返す。
    if llm_returned_empty and intent == "search":
        merged_memory = _sanitize_slot_memory(default_user_memory)

    next_action = _normalize_text(payload.get("next_action"))
    if next_action not in NEXT_ACTION_VALUES:
        next_action = "guidance"

    missing_slots = _sanitize_missing_slots(payload.get("missing_slots"))
    follow_up_questions = _sanitize_follow_up_questions(payload.get("follow_up_questions"))
    seed_queries = _dedupe_texts(list(payload.get("seed_queries") or []), limit=5)
    research_plan = _sanitize_research_plan(payload.get("research_plan"))
    condition_reasons = _sanitize_condition_reasons(payload.get("condition_reasons"))

    return {
        "intent": intent,
        "plan": {
            "budget_max": merged_memory.get("budget_max"),
            "target_area": merged_memory.get("target_area"),
            "station_walk_max": merged_memory.get("station_walk_max"),
            "move_in_date": merged_memory.get("move_in_date"),
            "layout_preference": merged_memory.get("layout_preference"),
            "listing_type": merged_memory.get("listing_type"),
        },
        "missing_slots": missing_slots,
        "follow_up_questions": follow_up_questions,
        "next_action": next_action,
        "user_memory": merged_memory,
        "seed_queries": seed_queries,
        "research_plan": research_plan,
        "condition_reasons": condition_reasons,
    }


# JP: heuristic planner outputを構築する。
# EN: Build heuristic planner output.
def _heuristic_planner_output(
    *,
    message: str,
    user_memory: dict[str, Any],
    planner_answers: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    intent = _infer_intent(message, user_memory, planner_answers)
    return {
        "intent": intent,
        "plan": {
            "budget_max": user_memory.get("budget_max"),
            "target_area": user_memory.get("target_area"),
            "station_walk_max": user_memory.get("station_walk_max"),
            "move_in_date": user_memory.get("move_in_date"),
            "layout_preference": user_memory.get("layout_preference"),
            "listing_type": user_memory.get("listing_type"),
        },
        "missing_slots": [],
        "follow_up_questions": [],
        "required_follow_up_questions": [],
        "next_action": "search_and_compare" if intent == "search" else "guidance",
        "user_memory": _sanitize_slot_memory(user_memory),
        "seed_queries": [],
        "research_plan": _default_research_plan(user_memory)
        if intent == "search"
        else _blank_research_plan(),
        "condition_reasons": _default_condition_reasons(user_memory),
    }


# JP: planner resultを確定する。
# EN: Finalize planner result.
def _finalize_planner_result(result: dict[str, Any]) -> dict[str, Any]:
    intent = _normalize_text(result.get("intent"))
    if intent not in INTENT_VALUES:
        intent = "general_question"

    user_memory = _sanitize_slot_memory(result.get("user_memory"))
    follow_up_questions = _sanitize_follow_up_questions(result.get("follow_up_questions"))
    research_plan = _sanitize_research_plan(result.get("research_plan"))
    condition_reasons = _sanitize_condition_reasons(result.get("condition_reasons"))

    if intent == "risk_check":
        return {
            **result,
            "intent": "risk_check",
            "plan": {
                "budget_max": user_memory.get("budget_max"),
                "target_area": user_memory.get("target_area"),
                "station_walk_max": user_memory.get("station_walk_max"),
                "move_in_date": user_memory.get("move_in_date"),
                "layout_preference": user_memory.get("layout_preference"),
                "listing_type": user_memory.get("listing_type"),
            },
            "missing_slots": [],
            "follow_up_questions": [],
            "required_follow_up_questions": [],
            "next_action": "risk_check",
            "user_memory": user_memory,
            "seed_queries": [],
            "research_plan": _blank_research_plan(),
            "condition_reasons": _blank_condition_reasons(),
        }

    if intent != "search":
        return {
            **result,
            "intent": "general_question",
            "plan": {
                "budget_max": user_memory.get("budget_max"),
                "target_area": user_memory.get("target_area"),
                "station_walk_max": user_memory.get("station_walk_max"),
                "move_in_date": user_memory.get("move_in_date"),
                "layout_preference": user_memory.get("layout_preference"),
                "listing_type": user_memory.get("listing_type"),
            },
            "missing_slots": [],
            "follow_up_questions": [],
            "required_follow_up_questions": [],
            "next_action": "guidance",
            "user_memory": user_memory,
            "seed_queries": [],
            "research_plan": _blank_research_plan(),
            "condition_reasons": _blank_condition_reasons(),
        }

    missing_slots = _required_missing_slots(user_memory)
    required_questions = [
        item for item in follow_up_questions if item["slot"] in set(missing_slots)
    ]
    optional_questions = [
        item for item in follow_up_questions if item["slot"] not in REQUIRED_PLANNING_SLOTS
    ]
    base_queries = _build_base_seed_queries(user_memory) if not missing_slots else []
    merged_research_plan = (
        research_plan if any(research_plan.values()) else _default_research_plan(user_memory)
    )
    merged_condition_reasons = _default_condition_reasons(user_memory)
    for key, value in condition_reasons.items():
        if value:
            merged_condition_reasons[key] = value

    return {
        **result,
        "intent": "search",
        "plan": {
            "budget_max": user_memory.get("budget_max"),
            "target_area": user_memory.get("target_area"),
            "station_walk_max": user_memory.get("station_walk_max"),
            "move_in_date": user_memory.get("move_in_date"),
            "layout_preference": user_memory.get("layout_preference"),
            "listing_type": user_memory.get("listing_type"),
        },
        "missing_slots": missing_slots,
        "follow_up_questions": optional_questions,
        "required_follow_up_questions": required_questions,
        "next_action": "missing_slots_question" if missing_slots else "search_and_compare",
        "user_memory": user_memory,
        "seed_queries": base_queries,
        "research_plan": merged_research_plan,
        "condition_reasons": merged_condition_reasons,
    }


# JP: search signalを検出する。
# EN: Detect search signal.
def detect_search_signal(message: str, planner_result: dict[str, Any] | None = None) -> bool:
    if planner_result is None:
        return False
    return _normalize_text(planner_result.get("intent")) == "search"


# JP: plannerを実行する。
# EN: Run planner.
def run_planner(
    *,
    message: str,
    user_memory: dict[str, Any],
    adapter: LLMAdapter | None,
    profile_memory: dict[str, Any] | None = None,
    planner_answers: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    merged_input_memory = _apply_planner_answers(user_memory, planner_answers)
    planner_message = _build_planner_message(message, planner_answers)

    if adapter is None:
        return _finalize_planner_result(
            _heuristic_planner_output(
                message=planner_message,
                user_memory=merged_input_memory,
                planner_answers=planner_answers,
            )
        )

    try:
        payload = _llm_parse(planner_message, merged_input_memory, profile_memory, adapter)
    except PromptExamplesError:
        raise
    except Exception:
        return _finalize_planner_result(
            _heuristic_planner_output(
                message=planner_message,
                user_memory=merged_input_memory,
                planner_answers=planner_answers,
            )
        )

    if not isinstance(payload, dict):
        return _finalize_planner_result(
            _heuristic_planner_output(
                message=planner_message,
                user_memory=merged_input_memory,
                planner_answers=planner_answers,
            )
        )

    return _finalize_planner_result(
        _parse_planner_output(payload, default_user_memory=merged_input_memory)
    )
