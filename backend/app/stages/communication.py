from __future__ import annotations

from typing import Any

from app.llm.base import LLMAdapter


# JP: user focus pointsを収集する。
# EN: Collect user focus points.
def _collect_user_focus_points(user_memory: dict[str, Any]) -> list[str]:
    focus_points: list[str] = []

    for key in ("must_conditions", "nice_to_have"):
        for item in user_memory.get(key, []) or []:
            text = str(item).strip()
            if text and text not in focus_points:
                focus_points.append(text)

    learned = user_memory.get("learned_preferences", {}) or {}
    for item in learned.get("liked_features", []) or []:
        text = str(item).strip()
        if text and text not in focus_points:
            focus_points.append(text)

    return focus_points


# JP: LLM confirmation itemsを処理する。
# EN: Process LLM confirmation items.
def _llm_confirmation_items(
    prop: dict[str, Any],
    user_memory: dict[str, Any],
    adapter: LLMAdapter,
) -> list[str]:
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {"type": "string"},
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }
    area_m2 = float(prop.get("area_m2") or 0)
    rent = int(prop.get("rent") or 0)
    management_fee = int(prop.get("management_fee") or 0)
    deposit = int(prop.get("deposit") or 0)
    key_money = int(prop.get("key_money") or 0)
    system = (
        "あなたは日本の賃貸物件アシスタントです。"
        "大家や仲介業者に確認すべき、この物件固有の確認事項を3〜5個提案してください。"
        "築年数、構造、管理方式、間取りの詳細、特約条項など、物件ごとに異なる特徴に焦点を当ててください。"
        "空室状況、初期費用、解約条件、内見日程などの汎用的な質問は標準チェックリストに含まれているため除外してください。"
    )
    user_prompt = (
        "物件情報:\n"
        f"- 建物名: {prop.get('building_name') or '不明'}\n"
        f"- 住所: {prop.get('address') or '不明'}\n"
        f"- 間取り: {prop.get('layout') or '不明'} / {area_m2}㎡\n"
        f"- 家賃: {rent:,}円 + 管理費 {management_fee:,}円\n"
        f"- 敷金: {deposit:,}円 / 礼金: {key_money:,}円\n"
        f"- 最寄り駅: {prop.get('nearest_station') or '不明'} "
        f"徒歩{int(prop.get('station_walk_min') or 0)}分\n"
        f"- 特徴: {', '.join(prop.get('features', []) or []) or 'なし'}\n"
        f"- 備考: {prop.get('notes') or 'なし'}\n"
        "ユーザーの重視点:\n"
        f"- {', '.join(_collect_user_focus_points(user_memory)) or 'なし'}\n"
        "この物件固有の確認項目を3〜5個、簡潔な日本語で提案してください（JSON配列 items に格納）。"
    )
    result = adapter.generate_structured(
        system=system, user=user_prompt, schema=schema, temperature=0.2
    )
    return [str(item).strip() for item in result.get("items", []) if str(item).strip()]


# JP: confirmation itemsを収集する。
# EN: Collect confirmation items.
def _collect_confirmation_items(
    prop: dict[str, Any],
    user_memory: dict[str, Any],
    adapter: LLMAdapter | None = None,
) -> list[str]:
    features = [str(item).strip() for item in prop.get("features", []) or [] if str(item).strip()]
    features_text = " ".join(features + [str(prop.get("notes") or "")])
    focus_points = _collect_user_focus_points(user_memory)

    items = [
        "最新の空室状況と申込スケジュール",
        "初期費用の内訳（仲介手数料・保証会社・鍵交換費を含む）",
        "短期解約違約金・更新料・解約予告期間の条件",
        "内見可能日と申込前に必要な書類",
    ]

    if "ペット" in features_text or any("ペット" in item for item in focus_points):
        items.append("ペット飼育時の条件、追加敷金、種類や頭数の制限")

    if "楽器" in features_text or any("楽器" in item for item in focus_points):
        items.append("楽器演奏の可否、演奏時間帯の制限、防音条件")

    if any(token in features_text for token in ["在宅ワーク", "ワーク", "高速回線"]) or any(
        "在宅ワーク" in item for item in focus_points
    ):
        items.append("回線種別・通信速度の目安、室内でオンライン会議しやすい環境か")

    if int(prop.get("station_walk_min") or 0) <= 5:
        items.append("駅近物件として、騒音や人通りの影響がどの程度あるか")

    if any(token in features_text for token in ["礼金ゼロ", "キャンペーン"]):
        items.append("礼金ゼロやキャンペーン条件が現在も有効か")

    deduped: list[str] = []
    for item in items:
        if item not in deduped:
            deduped.append(item)

    if adapter is not None:
        try:
            llm_items = _llm_confirmation_items(prop, user_memory, adapter)
            for item in llm_items:
                if item and item not in deduped:
                    deduped.append(item)
        except Exception:
            pass

    return deduped


# JP: fallback draftを構築する。
# EN: Build fallback draft.
def _build_fallback_draft(
    prop: dict[str, Any],
    user_memory: dict[str, Any],
    confirmation_items: list[str],
) -> str:
    building_name = str(prop.get("building_name") or "対象物件")
    rent = int(prop.get("rent") or 0)
    station = str(prop.get("nearest_station") or "最寄り駅要確認")
    walk = int(prop.get("station_walk_min") or 0)
    layout = str(prop.get("layout") or "間取り要確認")
    move_in = str(user_memory.get("move_in_date") or "入居時期未定")
    focus_points = _collect_user_focus_points(user_memory)
    features = [str(item).strip() for item in prop.get("features", []) or [] if str(item).strip()]

    intro_lines = [
        "お世話になっております。",
        f"{building_name}について問い合わせさせていただきます。",
        (
            f"現在、家賃帯は{rent:,}円前後、入居希望時期は{move_in}で検討しております。"
            if rent > 0
            else f"入居希望時期は{move_in}で検討しております。"
        ),
        f"物件条件は{layout}、{station} 徒歩{walk or '要確認'}分として認識しています。",
    ]

    if features:
        intro_lines.append(f"募集情報では {', '.join(features[:3])} が魅力だと感じました。")
    if focus_points:
        intro_lines.append(f"特に重視している点は {', '.join(focus_points[:3])} です。")

    numbered_items = "\n".join(
        f"{index}. {item}" for index, item in enumerate(confirmation_items, start=1)
    )

    return (
        "\n".join(intro_lines)
        + "\n以下についてご教示いただけますでしょうか。\n"
        + numbered_items
        + "\n何卒よろしくお願いいたします。"
    )


# JP: LLM draftを構築する。
# EN: Build LLM draft.
def _build_llm_draft(
    *,
    adapter: LLMAdapter,
    prop: dict[str, Any],
    user_memory: dict[str, Any],
    confirmation_items: list[str],
) -> str:
    system = (
        "あなたは日本の賃貸問い合わせアシスタントです。"
        "自然で簡潔な問い合わせメールを日本語で作成してください。"
        "以下の構成に従ってください: 1. 挨拶 2. 問い合わせ対象物件の特定と自己紹介 "
        "3. 確認事項の番号付きリスト 4. 結びの挨拶。"
        "ユーザーの優先事項と物件の特徴を反映し、提供されていない情報を捏造しないでください。"
    )
    user = (
        "物件情報:\n"
        f"- 建物名: {prop.get('building_name') or '対象物件'}\n"
        f"- 家賃: {int(prop.get('rent') or 0)}\n"
        f"- 最寄り駅: {prop.get('nearest_station') or '要確認'}\n"
        f"- 駅徒歩: {int(prop.get('station_walk_min') or 0)}分\n"
        f"- 間取り: {prop.get('layout') or '要確認'}\n"
        f"- 特徴: {', '.join(prop.get('features', []) or []) or '特になし'}\n"
        f"- 備考: {prop.get('notes') or 'なし'}\n"
        "ユーザー条件:\n"
        f"- 入居時期: {user_memory.get('move_in_date') or '未定'}\n"
        f"- 重視点: {', '.join(_collect_user_focus_points(user_memory)) or '特になし'}\n"
        "確認したい事項:\n"
        + "\n".join(f"- {item}" for item in confirmation_items)
        + "\n出力要件:\n"
        "- メール本文のみを出力\n"
        "- 丁寧な日本語\n"
        "- 箇条書きまたは番号付きで確認事項を含める\n"
    )
    return adapter.generate_text(system=system, user=user, temperature=0.3).strip()


# JP: communicationを実行する。
# EN: Run communication.
def run_communication(
    *,
    ranked_properties: list[dict[str, Any]],
    normalized_properties: list[dict[str, Any]],
    user_memory: dict[str, Any],
    selected_property_id: str | None = None,
    adapter: LLMAdapter | None = None,
) -> dict[str, Any]:
    by_id = {item["property_id_norm"]: item for item in normalized_properties}

    if not ranked_properties:
        return {
            "message_draft": "候補物件がないため、問い合わせ文はまだ作成できません。",
            "check_items": [
                "希望エリアを具体化する",
                "予算上限を決める",
                "駅徒歩条件を決める",
            ],
            "pending_action": None,
        }

    top = ranked_properties[0]
    if selected_property_id:
        selected = next(
            (
                item
                for item in ranked_properties
                if item["property_id_norm"] == selected_property_id
            ),
            None,
        )
        if selected is not None:
            top = selected

    prop = by_id.get(top["property_id_norm"], {})
    confirmation_items = _collect_confirmation_items(prop, user_memory, adapter)

    message_draft = _build_fallback_draft(prop, user_memory, confirmation_items)
    if adapter is not None:
        try:
            llm_draft = _build_llm_draft(
                adapter=adapter,
                prop=prop,
                user_memory=user_memory,
                confirmation_items=confirmation_items,
            )
            if llm_draft:
                message_draft = llm_draft
        except Exception:
            pass

    pending_action = {
        "action_type": "send_inquiry",
        "label": "問い合わせ文の送信",
        "target_property_id": top["property_id_norm"],
    }

    return {
        "message_draft": message_draft,
        "check_items": confirmation_items,
        "pending_action": pending_action,
    }
