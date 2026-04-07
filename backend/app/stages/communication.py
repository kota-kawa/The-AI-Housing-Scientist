from __future__ import annotations

from typing import Any


def run_communication(
    *,
    ranked_properties: list[dict[str, Any]],
    normalized_properties: list[dict[str, Any]],
    user_memory: dict[str, Any],
) -> dict[str, Any]:
    by_id = {item["property_id_norm"]: item for item in normalized_properties}

    if not ranked_properties:
        return {
            "message_draft": "候補物件がないため、問い合わせ文はまだ作成できません。",
            "check_items": [
                "希望エリアを具体化する",
                "家賃上限を決める",
                "駅徒歩条件を決める",
            ],
            "pending_action": None,
        }

    top = ranked_properties[0]
    prop = by_id.get(top["property_id_norm"], {})
    building_name = prop.get("building_name_norm", "対象物件")
    rent = int(prop.get("rent") or 0)

    move_in = user_memory.get("move_in_date", "入居時期未定")

    message_draft = (
        "お世話になっております。\n"
        f"{building_name}について問い合わせさせていただきます。\n"
        f"現在、家賃帯は{rent:,}円前後、入居希望時期は{move_in}で検討しております。\n"
        "以下についてご教示いただけますでしょうか。\n"
        "1. 最新の空室状況\n"
        "2. 初期費用の内訳（仲介手数料、保証会社、鍵交換費含む）\n"
        "3. 短期解約違約金・更新料・解約予告期間\n"
        "4. 内見可能日程\n"
        "何卒よろしくお願いいたします。"
    )

    check_items = [
        "空室状況を最終確認",
        "初期費用内訳を確認",
        "違約金・更新料を確認",
        "解約予告期間を確認",
        "内見候補日を2-3案提示",
    ]

    pending_action = {
        "action_type": "send_inquiry",
        "label": "問い合わせ文の送信",
        "target_property_id": top["property_id_norm"],
    }

    return {
        "message_draft": message_draft,
        "check_items": check_items,
        "pending_action": pending_action,
    }
