from __future__ import annotations

import re
from typing import Any

from app.models import RiskItem


RISK_RULES = [
    (
        "renewal_fee",
        re.compile(r"更新料"),
        "medium",
        "更新料の金額と支払い頻度を契約前に確定してください。",
    ),
    (
        "early_termination",
        re.compile(r"短期解約|違約金"),
        "high",
        "短期解約違約金の適用期間と金額を確認してください。",
    ),
    (
        "notice_period",
        re.compile(r"解約予告|\d+か月前"),
        "high",
        "解約予告期限をカレンダー化して管理してください。",
    ),
    (
        "guarantor",
        re.compile(r"保証会社|保証料"),
        "medium",
        "保証会社の更新料・連帯保証人要件を確認してください。",
    ),
]


def run_risk_check(*, source_text: str) -> dict[str, Any]:
    risk_items: list[RiskItem] = []
    must_confirm: list[str] = []

    for risk_type, pattern, severity, recommendation in RISK_RULES:
        if pattern.search(source_text):
            evidence = pattern.search(source_text)
            snippet = evidence.group(0) if evidence else risk_type
            risk_items.append(
                RiskItem(
                    risk_type=risk_type,  # type: ignore[arg-type]
                    severity=severity,  # type: ignore[arg-type]
                    evidence=snippet,
                    recommendation=recommendation,
                )
            )
            must_confirm.append(recommendation)

    if not risk_items:
        risk_items.append(
            RiskItem(
                risk_type="other",
                severity="low",
                evidence="契約条項テキストの明示入力なし",
                recommendation="契約書・初期費用表の原文を貼り付けると精度が上がります。",
            )
        )
        must_confirm.append("更新料・違約金・解約予告・保証会社条件の4点は必ず確認してください。")

    return {
        "risk_items": [item.model_dump() for item in risk_items],
        "must_confirm_list": must_confirm,
    }
