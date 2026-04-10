from __future__ import annotations

import re
from typing import Any

from app.llm.base import LLMAdapter
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

CONTRACT_KEYWORDS = (
    "更新料",
    "短期解約",
    "違約金",
    "解約予告",
    "保証会社",
    "保証料",
    "契約",
    "敷金",
    "礼金",
    "原状回復",
    "退去",
    "特約",
)


# JP: looks like contract textを処理する。
# EN: Process looks like contract text.
def looks_like_contract_text(source_text: str) -> bool:
    keyword_hits = sum(1 for token in CONTRACT_KEYWORDS if token in source_text)
    normalized_length = len(re.sub(r"\s+", "", source_text))
    return keyword_hits >= 2 or (keyword_hits >= 1 and normalized_length >= 80)


# JP: rule based risk resultを構築する。
# EN: Build rule based risk result.
def _build_rule_based_risk_result(source_text: str) -> dict[str, Any]:
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


# JP: LLM risk resultを構築する。
# EN: Build LLM risk result.
def _build_llm_risk_result(source_text: str, adapter: LLMAdapter) -> dict[str, Any]:
    schema = {
        "type": "object",
        "properties": {
            "risk_items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "risk_type": {
                            "type": "string",
                            "enum": [
                                "renewal_fee",
                                "early_termination",
                                "notice_period",
                                "guarantor",
                                "other",
                            ],
                        },
                        "severity": {"type": "string", "enum": ["high", "medium", "low"]},
                        "evidence": {"type": "string"},
                        "recommendation": {"type": "string"},
                    },
                    "required": ["risk_type", "severity", "evidence", "recommendation"],
                    "additionalProperties": False,
                },
            },
            "must_confirm_list": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["risk_items", "must_confirm_list"],
        "additionalProperties": False,
    }
    result = adapter.generate_structured(
        system=(
            "You are a Japanese rental contract risk analyst. "
            "Extract concrete risks from the provided contract text. "
            "Do not invent clauses that do not appear in the source."
        ),
        user=f"契約条項テキスト:\n{source_text}",
        schema=schema,
        temperature=0.0,
    )
    return {
        "risk_items": [RiskItem(**item).model_dump() for item in result.get("risk_items", [])],
        "must_confirm_list": [
            str(item).strip() for item in result.get("must_confirm_list", []) if str(item).strip()
        ],
    }


# JP: risk resultsを結合する。
# EN: Merge risk results.
def _merge_risk_results(primary: dict[str, Any], secondary: dict[str, Any]) -> dict[str, Any]:
    merged_items: list[dict[str, Any]] = []
    seen_item_keys: set[tuple[str, str]] = set()
    for source in [primary, secondary]:
        for raw_item in source.get("risk_items", []) or []:
            item = RiskItem(**raw_item).model_dump()
            dedupe_key = (str(item["risk_type"]), str(item["evidence"]))
            if dedupe_key in seen_item_keys:
                continue
            seen_item_keys.add(dedupe_key)
            merged_items.append(item)

    merged_confirms: list[str] = []
    for source in [primary, secondary]:
        for raw_item in source.get("must_confirm_list", []) or []:
            item = str(raw_item).strip()
            if item and item not in merged_confirms:
                merged_confirms.append(item)

    return {
        "risk_items": merged_items,
        "must_confirm_list": merged_confirms,
    }


# JP: risk checkを実行する。
# EN: Run risk check.
def run_risk_check(
    *,
    source_text: str,
    adapter: LLMAdapter | None = None,
) -> dict[str, Any]:
    rule_based = _build_rule_based_risk_result(source_text)
    if adapter is None:
        return rule_based

    try:
        llm_result = _build_llm_risk_result(source_text, adapter)
    except Exception:
        return rule_based

    return _merge_risk_results(llm_result, rule_based)
