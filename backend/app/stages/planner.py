from __future__ import annotations

import json
from typing import Any

from app.llm.base import LLMAdapter
from app.stages.prompt_examples import PromptExamplesError, sample_prompt_examples

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
ALL_SLOT_KEYS = list(FOLLOW_UP_SLOT_ORDER) + list(FOLLOW_UP_OPTIONAL_SLOTS)
INTENT_VALUES = ("search", "risk_check", "general_question")
NEXT_ACTION_VALUES = (
    "missing_slots_question",
    "search_and_compare",
    "risk_check",
    "guidance",
)
TEXT_SLOT_KEYS = {"target_area", "move_in_date", "layout_preference"}
INTEGER_SLOT_KEYS = {"budget_max", "station_walk_max"}
LIST_SLOT_KEYS = {"must_conditions", "nice_to_have"}


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _dedupe_texts(values: list[Any], *, limit: int | None = None) -> list[str]:
    deduped: list[str] = []
    for value in values:
        text = _normalize_text(value)
        if text and text not in deduped:
            deduped.append(text)
        if limit is not None and len(deduped) >= limit:
            break
    return deduped


def _blank_slot_memory() -> dict[str, Any]:
    return {
        "budget_max": None,
        "target_area": None,
        "station_walk_max": None,
        "move_in_date": None,
        "layout_preference": None,
        "must_conditions": [],
        "nice_to_have": [],
    }


def _blank_research_plan() -> dict[str, Any]:
    return {"summary": "", "goal": "", "strategy": [], "rationale": ""}


def _blank_condition_reasons() -> dict[str, str]:
    return dict.fromkeys(SEARCH_SIGNAL_KEYS, "")


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


def _sanitize_missing_slots(raw_missing_slots: Any) -> list[str]:
    return [
        slot
        for slot in _dedupe_texts(list(raw_missing_slots or []), limit=3)
        if slot in ALL_SLOT_KEYS
    ]


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


def _sanitize_research_plan(raw_plan: Any) -> dict[str, Any]:
    raw_plan = raw_plan if isinstance(raw_plan, dict) else {}
    return {
        "summary": _normalize_text(raw_plan.get("summary")),
        "goal": _normalize_text(raw_plan.get("goal")),
        "strategy": _dedupe_texts(list(raw_plan.get("strategy") or []), limit=5),
        "rationale": _normalize_text(raw_plan.get("rationale")),
    }


def _sanitize_condition_reasons(raw_reasons: Any) -> dict[str, str]:
    reasons = _blank_condition_reasons()
    if not isinstance(raw_reasons, dict):
        return reasons
    for key in SEARCH_SIGNAL_KEYS:
        reasons[key] = _normalize_text(raw_reasons.get(key))
    return reasons


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
        },
        "missing_slots": [],
        "follow_up_questions": [],
        "next_action": "guidance",
        "user_memory": merged_memory,
        "seed_queries": [],
        "research_plan": _blank_research_plan(),
        "condition_reasons": _blank_condition_reasons(),
    }


def _safe_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


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
        "required": [
            "budget_max",
            "target_area",
            "station_walk_max",
            "move_in_date",
            "layout_preference",
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
            "budget_max": {"label": "家賃上限", "meaning": "家賃上限（円）"},
            "station_walk_max": {"label": "駅徒歩", "meaning": "駅徒歩の上限（分）"},
            "layout_preference": {"label": "間取り", "meaning": "希望間取り"},
            "move_in_date": {"label": "入居時期", "meaning": "すぐなら asap"},
            "must_conditions": {"label": "必須条件", "meaning": "外せない条件"},
            "nice_to_have": {"label": "あると良い条件", "meaning": "できれば欲しい条件"},
        },
        "decision_rules": [
            "最新メッセージと current_user_memory を統合した user_memory を返す",
            "既存条件は、ユーザーが明確に上書き・否定した場合だけ消す",
            "intent は search / risk_check / general_question のいずれかにする",
            "search では、実際に候補収集を始めるべきなら next_action を search_and_compare にする",
            "search だが曖昧すぎて候補収集の前に確認が必要なら next_action を missing_slots_question にする",
            "missing_slots は 0〜3 件で、次に聞く価値が高い順に並べる",
            "follow_up_questions は missing_slots と同じ順序・同じ slot だけを返す",
            "follow_up_questions の label は slot_reference の日本語ラベルに合わせる",
            "follow_up_questions の question は、例にない回答や自由入力でも答えやすい聞き方にする",
            "follow_up_questions の examples は候補の例示であり、網羅的な選択肢として扱わない",
            "examples は user_message や memory にない特定の地域・予算・条件へ誘導しない",
            "examples は固定候補に見えにくいよう、粒度や表現を少し分散させてよい",
            "next_action が search_and_compare のときは seed_queries を 5〜8 件返す",
            "seed_queries は current_user_memory.target_area と今回の user_message のみを根拠に生成する",
            "seed_queries に profile_history_summary のエリア・条件は含めない（user_message に同じエリアが明示されている場合を除く）",
            "seed_queries には同一エリアの言い換えを含め、必須条件を外した比較用クエリや予算を少し緩めた確認用クエリを 1 本含めてよい",
            "seed_queries に別エリアへの拡張は入れない。近隣エリアの探索は検索システムが自動的に追加する",
            "next_action が missing_slots_question のときは seed_queries を空にしてよい",
            "research_plan はユーザー条件に即して summary / goal / strategy / rationale を返す",
            "condition_reasons は各条件が今回の検索で重要な理由を 1 文ずつ返し、該当しない key は空文字にする",
            "『絶対』『必須』『譲れない』『〜じゃないとダメ』は must_conditions",
            "『できれば』『あったらいい』『理想』『〜だとうれしい』は nice_to_have",
            "地名や駅名は『町田』『三軒茶屋』『武蔵小杉』のように接尾辞がなくても target_area に入れる",
            "RC / SRC / 鉄筋コンクリート / 鉄骨 / 木造 など建物構造も条件として扱う",
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
            "You are a Japanese rental planner responsible for the entire planning decision. "
            "Infer intent, merge memory, choose the next action, decide which conditions are missing, "
            "write follow-up questions, generate seed queries, and draft the research plan. "
            "Treat follow-up examples as non-exhaustive hints, never as exhaustive options. "
            "Return only structured data grounded in the provided message and memory."
        ),
        user=json.dumps(prompt_payload, ensure_ascii=False, indent=2),
        schema=_planner_schema(),
        temperature=0.1,
    )


def _parse_planner_output(
    payload: dict[str, Any], *, default_user_memory: dict[str, Any]
) -> dict[str, Any]:
    intent = _normalize_text(payload.get("intent"))
    if intent not in INTENT_VALUES:
        intent = "general_question"

    merged_memory = _sanitize_slot_memory(payload.get("user_memory"))
    llm_returned_empty = not any(
        [
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

    missing_slots = _sanitize_missing_slots(payload.get("missing_slots"))
    follow_up_questions = _sanitize_follow_up_questions(payload.get("follow_up_questions"))
    if missing_slots:
        allowed_slots = set(missing_slots)
        follow_up_questions = [
            item for item in follow_up_questions if item["slot"] in allowed_slots
        ]

    next_action = _normalize_text(payload.get("next_action"))
    if next_action not in NEXT_ACTION_VALUES:
        next_action = "guidance"

    seed_queries = _dedupe_texts(list(payload.get("seed_queries") or []), limit=8)
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
        },
        "missing_slots": missing_slots,
        "follow_up_questions": follow_up_questions,
        "next_action": next_action,
        "user_memory": merged_memory,
        "seed_queries": seed_queries,
        "research_plan": research_plan,
        "condition_reasons": condition_reasons,
    }


def detect_search_signal(message: str, planner_result: dict[str, Any] | None = None) -> bool:
    if planner_result is None:
        return False
    return _normalize_text(planner_result.get("intent")) == "search"


def run_planner(
    *,
    message: str,
    user_memory: dict[str, Any],
    adapter: LLMAdapter | None,
    profile_memory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if adapter is None:
        return _empty_planner_result(user_memory)

    try:
        payload = _llm_parse(message, user_memory, profile_memory, adapter)
    except PromptExamplesError:
        raise
    except Exception:
        return _empty_planner_result(user_memory)

    if not isinstance(payload, dict):
        return _empty_planner_result(user_memory)

    return _parse_planner_output(payload, default_user_memory=user_memory)
