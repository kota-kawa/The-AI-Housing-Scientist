from __future__ import annotations

import json
from typing import Any

from app.llm.base import LLMAdapter
from app.models import ChatMessageResponse

RESEARCH_STAGE_ORDER = [
    ("plan_finalize", "計画確認"),
    ("tree_search", "動的探索"),
    ("synthesize", "結果要約"),
]


# JP: generate LLM resume bodyを処理する。
# EN: Process generate LLM resume body.
def _generate_llm_resume_body(profile_summary: dict[str, Any], adapter: LLMAdapter) -> str:
    """LLMで自然なセッション再開メッセージを生成する。失敗時は空文字を返す。"""
    labels = profile_summary.get("last_search_labels") or []
    frequent_area = str(profile_summary.get("frequent_area") or "")
    stable_prefs = list(profile_summary.get("stable_preferences") or [])
    liked_features = list(profile_summary.get("liked_features") or [])
    system = (
        "You are a friendly Japanese rental assistant. "
        "Write a short, warm session resume message in Japanese (2–3 sentences). "
        "Mention the user's previous search conditions naturally and ask whether they'd like to continue. "
        "Do not use bullet points or markdown."
    )
    user_prompt = (
        "前回の検索条件:\n"
        f"- 条件ラベル: {', '.join(labels[:5]) or 'なし'}\n"
        f"- よく使うエリア: {frequent_area or 'なし'}\n"
        f"- 安定した好み: {', '.join(stable_prefs[:3]) or 'なし'}\n"
        f"- 気になる特徴: {', '.join(liked_features[:3]) or 'なし'}\n"
        "この情報をもとに、ユーザーへの自然な「お帰りなさい」メッセージを生成してください。"
    )
    return adapter.generate_text(system=system, user=user_prompt, temperature=0.4).strip()


# JP: display textを正規化する。
# EN: Normalize display text.
def _normalize_display_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


# JP: display textsを正規化する。
# EN: Normalize display texts.
def _normalize_display_texts(values: list[Any], *, limit: int | None = None) -> list[str]:
    items: list[str] = []
    for value in values:
        text = _normalize_display_text(value)
        if text and text not in items:
            items.append(text)
        if limit is not None and len(items) >= limit:
            break
    return items


# JP: generate LLM plan presentationを処理する。
# EN: Process generate LLM plan presentation.
def _generate_llm_plan_presentation(
    *,
    user_message: str,
    conditions: list[dict[str, str]],
    follow_up_questions: list[dict[str, Any]],
    seed_queries: list[str],
    planner_plan: dict[str, Any],
    adapter: LLMAdapter | None,
) -> dict[str, Any] | None:
    if adapter is None:
        return None

    schema = {
        "type": "object",
        "properties": {
            "assistant_message": {"type": "string"},
            "summary": {"type": "string"},
            "goal": {"type": "string"},
            "rationale": {"type": "string"},
            "strategy": {"type": "array", "items": {"type": "string"}},
            "open_questions": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "assistant_message",
            "summary",
            "goal",
            "rationale",
            "strategy",
            "open_questions",
        ],
        "additionalProperties": False,
    }
    prompt_payload = {
        "user_message": user_message,
        "conditions": conditions,
        "follow_up_questions": [
            _normalize_display_text(item.get("question"))
            for item in follow_up_questions
            if _normalize_display_text(item.get("question"))
        ],
        "seed_queries": seed_queries,
        "planner_draft": {
            "summary": _normalize_display_text(planner_plan.get("summary")),
            "goal": _normalize_display_text(planner_plan.get("goal")),
            "strategy": _normalize_display_texts(list(planner_plan.get("strategy") or []), limit=5),
            "rationale": _normalize_display_text(planner_plan.get("rationale")),
        },
        "output_rules": [
            "assistant_message は 1〜2 文の自然な日本語で、計画を作成したことと承認後に調査を始めることを伝える",
            "follow_up_questions がある場合は、分かる条件があれば追加できると軽く触れる",
            "summary はユーザー条件を踏まえた検索方針を 1 文で要約する",
            "goal は比較・絞り込みの到達点を 1 文で書く",
            "rationale はこの進め方が合う理由を 1 文で書く",
            "strategy は 3〜5 項目。条件に即した具体性を優先し、抽象的な定型文を避ける",
            "open_questions は follow_up_questions を短く言い換えるだけにし、新しい条件を発明しない",
            "与えられた条件・質問・下書きにない制約や設備は足さない",
        ],
    }
    try:
        result = adapter.generate_structured(
            system=(
                "You are a Japanese rental planning assistant rewriting a pre-search plan for UI display. "
                "Ground every sentence in the provided user message, extracted conditions, and draft plan. "
                "Do not invent unsupported constraints, areas, budgets, or amenities. "
                "Return only the requested JSON."
            ),
            user=json.dumps(prompt_payload, ensure_ascii=False, indent=2),
            schema=schema,
            temperature=0.3,
        )
    except Exception:
        return None

    presentation = {
        "assistant_message": _normalize_display_text(result.get("assistant_message")),
        "summary": _normalize_display_text(result.get("summary")),
        "goal": _normalize_display_text(result.get("goal")),
        "rationale": _normalize_display_text(result.get("rationale")),
        "strategy": _normalize_display_texts(list(result.get("strategy") or []), limit=5),
        "open_questions": _normalize_display_texts(
            list(result.get("open_questions") or []), limit=3
        ),
    }
    if not any(
        [
            presentation["assistant_message"],
            presentation["summary"],
            presentation["goal"],
            presentation["rationale"],
            presentation["strategy"],
            presentation["open_questions"],
        ]
    ):
        return None
    return presentation


# JP: generate response labelsを処理する。
# EN: Process generate response labels.
def _generate_response_labels(
    *,
    response: ChatMessageResponse,
    adapter: LLMAdapter,
) -> ChatMessageResponse:
    """status_label と各 UIBlock の display_label を LLM で生成して差し込む。"""
    # status_label: ステータスとブロック数を踏まえた一言ラベル
    block_summary = ", ".join(f"{b.type}({b.title})" for b in response.blocks[:4]) or "なし"
    system = (
        "You are a Japanese UI labeling assistant. "
        "Return only the requested JSON with no explanation."
    )
    schema = {
        "type": "object",
        "properties": {
            "status_label": {"type": "string"},
            "block_labels": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["status_label", "block_labels"],
        "additionalProperties": False,
    }
    user_prompt = (
        f"status: {response.status}\n"
        f"assistant_message (冒頭80字): {response.assistant_message[:80]}\n"
        f"blocks: {block_summary}\n"
        "以下を生成してください:\n"
        "- status_label: このレスポンスの状態を表す10字以内の日本語ラベル\n"
        "- block_labels: 各ブロックのタイプラベルを上書きする10字以内の日本語ラベルを順番に配列で"
        f"（{len(response.blocks)}個）"
    )
    result = adapter.generate_structured(
        system=system, user=user_prompt, schema=schema, temperature=0.1
    )
    new_response = response.model_copy(deep=True)
    if result.get("status_label"):
        new_response.status_label = str(result["status_label"]).strip()
    block_labels: list[Any] = result.get("block_labels") or []
    for i, block in enumerate(new_response.blocks):
        if i < len(block_labels) and str(block_labels[i]).strip():
            block.display_label = str(block_labels[i]).strip()
    return new_response


# JP: generate LLM guidance messageを処理する。
# EN: Process generate LLM guidance message.
def _generate_llm_guidance_message(
    *,
    task_memory: dict[str, Any],
    user_message: str,
    adapter: LLMAdapter,
) -> str:
    """タスク状態を踏まえた文脈依存のガイダンスメッセージをLLMで生成する。"""
    status = str(task_memory.get("status") or "")
    has_plan = bool(task_memory.get("draft_research_plan"))
    ranked = list(
        task_memory.get("last_display_ranked_properties")
        or task_memory.get("last_ranked_properties")
        or []
    )
    last_error = str(task_memory.get("last_error") or "")

    context_lines = [f"- 現在の状態: {status or '不明'}"]
    if has_plan:
        context_lines.append("- 調査計画: 作成済み（承認待ち）")
    if ranked:
        context_lines.append(f"- 前回の候補: {len(ranked)}件保持中")
    if last_error:
        context_lines.append(f"- 直前のエラー: {last_error[:80]}")

    system = (
        "You are a friendly Japanese rental assistant. "
        "Write a short, actionable guidance message in Japanese (1–2 sentences) "
        "telling the user what they can do next based on the current session state. "
        "Be specific. Do not use bullet points."
    )
    user_prompt = (
        "セッション状態:\n" + "\n".join(context_lines) + "\n"
        f"ユーザーの最後のメッセージ: {user_message[:120]}\n"
        "次のアクションを案内する短いメッセージを生成してください。"
    )
    return adapter.generate_text(system=system, user=user_prompt, temperature=0.3).strip()
