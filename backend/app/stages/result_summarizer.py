from __future__ import annotations

import json
import re
from typing import Any

from app.llm.base import LLMAdapter

PROPERTY_CANDIDATES_KEY = "物件候補リスト"
REJECTION_REASONS_KEY = "却下理由"
COMMON_RISKS_KEY = "共通リスク"
OPEN_QUESTIONS_KEY = "未解決の調査項目"
SUMMARY_KEY = "summary"

MAX_CANDIDATES = 5
MAX_REJECTIONS = 8
MAX_OPEN_QUESTIONS = 6

result_summarizer_sys_msg = """You are an expert Japanese rental research analyst.
You are given outputs from multiple tree-search nodes that belong to the same search branch.
Your task is to compress them into a factual branch summary that is useful for later ranking and decision making.

Important instructions:
- Do NOT hallucinate or invent information that does not appear in the node outputs.
- Merge repeated properties across nodes instead of repeating them.
- Prefer explicit evidence from normalized properties, ranked reasons, search snippets, and detail-page excerpts.
- If something is still unclear, move it to unresolved questions instead of guessing.
"""

branch_result_aggregate_prompt = """You are given:

1) A draft compressed summary of one search branch:
{draft_summary}

2) Compacted outputs from multiple nodes in the same branch:
{branch_nodes}

Your task is to produce an updated branch summary for downstream stages.

Requirements:
1. Preserve useful candidate facts from the draft summary when they are supported by the node outputs.
2. Merge repeated candidates and repeated risks across nodes.
3. Keep only the most decision-relevant candidates and rejection reasons.
4. If evidence is weak or incomplete, keep the candidate concise and list the missing verification in unresolved questions.
5. Return only valid JSON that matches the required schema.
"""


# JP: compact textを処理する。
# EN: Process compact text.
def _compact_text(value: Any, *, max_chars: int = 220) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


# JP: strip htmlを処理する。
# EN: Process strip html.
def _strip_html(value: str) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", " ", value or "", flags=re.IGNORECASE)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# JP: unique stringsを処理する。
# EN: Process unique strings.
def _unique_strings(values: list[Any], *, limit: int) -> list[str]:
    deduped: list[str] = []
    for value in values:
        text = _compact_text(value, max_chars=220).strip()
        if text and text not in deduped:
            deduped.append(text)
        if len(deduped) >= limit:
            break
    return deduped


# JP: property keyを処理する。
# EN: Process property key.
def _property_key(candidate: dict[str, Any]) -> str:
    return (
        str(candidate.get("property_id_norm") or "").strip()
        or str(candidate.get("detail_url") or "").strip()
        or f"{candidate.get('building_name', '')}|{candidate.get('address', '')}"
    )


# JP: candidate completenessを処理する。
# EN: Process candidate completeness.
def _candidate_completeness(candidate: dict[str, Any]) -> int:
    checks = [
        int(candidate.get("rent") or 0) > 0,
        int(candidate.get("station_walk_min") or 0) > 0,
        bool(candidate.get("layout")),
        bool(candidate.get("address")),
        float(candidate.get("area_m2") or 0.0) > 0,
        bool(candidate.get("detail_url")),
        bool(candidate.get("detail_excerpt")),
    ]
    return sum(1 for passed in checks if passed)


# JP: candidate sort keyを処理する。
# EN: Process candidate sort key.
def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (
        float(candidate.get("score") or 0.0),
        _candidate_completeness(candidate),
        int(candidate.get("seen_count") or 0),
        str(candidate.get("building_name") or ""),
    )


# JP: candidate snapshotを構築する。
# EN: Build candidate snapshot.
def _build_candidate_snapshot(
    *,
    prop: dict[str, Any],
    raw_result: dict[str, Any],
    ranked: dict[str, Any],
    detail_html: str,
    node_label: str,
) -> dict[str, Any]:
    search_excerpt = _compact_text(
        raw_result.get("snippet_summary") or raw_result.get("description") or "",
        max_chars=180,
    )
    detail_excerpt = _compact_text(_strip_html(detail_html), max_chars=220) if detail_html else ""
    evidence = _unique_strings(
        [
            ranked.get("why_selected"),
            search_excerpt,
            detail_excerpt,
            prop.get("notes"),
            *list(prop.get("features", []) or [])[:3],
        ],
        limit=4,
    )
    return {
        "property_id_norm": str(prop.get("property_id_norm") or ""),
        "building_name": _compact_text(prop.get("building_name") or "候補物件", max_chars=80),
        "image_url": str(prop.get("image_url") or ""),
        "address": _compact_text(prop.get("address") or "", max_chars=120),
        "rent": int(prop.get("rent") or 0),
        "layout": _compact_text(prop.get("layout") or "", max_chars=24),
        "station_walk_min": int(prop.get("station_walk_min") or 0),
        "area_m2": float(prop.get("area_m2") or 0.0),
        "management_fee": int(prop.get("management_fee") or 0),
        "deposit": int(prop.get("deposit") or 0),
        "key_money": int(prop.get("key_money") or 0),
        "available_date": _compact_text(prop.get("available_date") or "", max_chars=40),
        "agency_name": _compact_text(prop.get("agency_name") or "", max_chars=60),
        "detail_url": str(prop.get("detail_url") or ""),
        "score": float(ranked.get("score") or 0.0),
        "reason": _compact_text(
            ranked.get("why_selected")
            or search_excerpt
            or detail_excerpt
            or "条件に合う候補として残った物件です。",
            max_chars=180,
        ),
        "rejection_reason": _compact_text(ranked.get("why_not_selected") or "", max_chars=180),
        "evidence": evidence,
        "matched_queries": _unique_strings(
            list(raw_result.get("matched_queries", []) or []), limit=3
        ),
        "source_nodes": [node_label],
        "detail_excerpt": detail_excerpt,
        "seen_count": 1,
    }


# JP: node snapshotを構築する。
# EN: Build node snapshot.
def _build_node_snapshot(branch_node: dict[str, Any]) -> dict[str, Any]:
    raw_results = list(branch_node.get("raw_results", []) or [])
    detail_html_map = dict(branch_node.get("detail_html_map", {}) or {})
    normalized_properties = list(branch_node.get("normalized_properties", []) or [])
    ranked_properties = list(branch_node.get("ranked_properties", []) or [])
    duplicate_groups = list(branch_node.get("duplicate_groups", []) or [])
    dropped_properties = list(branch_node.get("dropped_properties", []) or [])
    integrity_reviews = list(branch_node.get("integrity_reviews", []) or [])
    raw_by_url = {
        str(item.get("url") or ""): item
        for item in raw_results
        if str(item.get("url") or "").strip()
    }
    ranked_by_id = {
        str(item.get("property_id_norm") or ""): item
        for item in ranked_properties
        if str(item.get("property_id_norm") or "").strip()
    }

    ordered_ids: list[str] = []
    for item in ranked_properties[:MAX_CANDIDATES]:
        property_id = str(item.get("property_id_norm") or "").strip()
        if property_id and property_id not in ordered_ids:
            ordered_ids.append(property_id)
    for prop in normalized_properties[:MAX_CANDIDATES]:
        property_id = str(prop.get("property_id_norm") or "").strip()
        if property_id and property_id not in ordered_ids:
            ordered_ids.append(property_id)

    candidates: list[dict[str, Any]] = []
    normalized_by_id = {
        str(item.get("property_id_norm") or ""): item
        for item in normalized_properties
        if str(item.get("property_id_norm") or "").strip()
    }
    for property_id in ordered_ids:
        prop = normalized_by_id.get(property_id)
        if not prop:
            continue
        detail_url = str(prop.get("detail_url") or "")
        candidates.append(
            _build_candidate_snapshot(
                prop=prop,
                raw_result=raw_by_url.get(detail_url, {}),
                ranked=ranked_by_id.get(property_id, {}),
                detail_html=str(detail_html_map.get(detail_url) or ""),
                node_label=str(branch_node.get("label") or branch_node.get("branch_id") or "node"),
            )
        )

    search_hits = [
        {
            "title": _compact_text(item.get("title") or "", max_chars=100),
            "url": str(item.get("url") or ""),
            "summary": _compact_text(
                item.get("snippet_summary") or item.get("description") or "",
                max_chars=140,
            ),
            "matched_queries": _unique_strings(
                list(item.get("matched_queries", []) or []), limit=2
            ),
        }
        for item in raw_results[:MAX_CANDIDATES]
        if str(item.get("title") or "").strip() or str(item.get("url") or "").strip()
    ]

    return {
        "branch_id": str(branch_node.get("branch_id") or branch_node.get("node_key") or ""),
        "label": str(branch_node.get("label") or ""),
        "depth": int(branch_node.get("depth") or 0),
        "queries": _unique_strings(list(branch_node.get("queries", []) or []), limit=4),
        "issues": _unique_strings(list(branch_node.get("issues", []) or []), limit=3),
        "search_summary": {
            "raw_result_count": len(raw_results),
            "normalized_count": len(normalized_properties),
            "ranked_count": len(ranked_properties),
            "detail_hit_count": int(
                branch_node.get("search_summary", {}).get("detail_hit_count") or 0
            ),
            "duplicate_group_count": len(duplicate_groups),
        },
        "candidates": candidates,
        "search_hits": search_hits,
        "duplicate_groups": [
            {
                "property_ids": list(group.get("property_ids", []) or [])[:4],
                "confidence": float(group.get("confidence") or 0.0),
                "reason": _compact_text(group.get("reason") or "", max_chars=120),
            }
            for group in duplicate_groups[:4]
        ],
        "dropped_candidates": [
            {
                "target": _compact_text(prop.get("building_name") or "除外候補", max_chars=80),
                "property_id_norm": str(prop.get("property_id_norm") or ""),
                "reason": _compact_text(
                    " / ".join(
                        str(item)
                        for item in (
                            (prop.get("integrity_review") or {}).get("inconsistencies", [])
                        )
                        if str(item).strip()
                    )
                    or "整合性レビューで推薦対象から除外",
                    max_chars=180,
                ),
                "source_node": str(branch_node.get("label") or branch_node.get("branch_id") or "node"),
            }
            for prop in dropped_properties[:MAX_REJECTIONS]
        ],
        "integrity_issue_count": len(
            [
                item
                for item in integrity_reviews
                if item.get("should_drop") or item.get("review_status") == "needs_confirmation"
            ]
        ),
    }


# JP: fallback result summaryを処理する。
# EN: Process fallback result summary.
def _fallback_result_summary(branch_nodes: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, dict[str, Any]] = {}
    rejections: list[dict[str, Any]] = []
    branch_issues: list[dict[str, Any]] = []
    detail_page_hit_count = 0
    fallback_search_hits: list[dict[str, Any]] = []

    for node in branch_nodes:
        search_summary = dict(node.get("search_summary", {}) or {})
        detail_page_hit_count += int(search_summary.get("detail_hit_count") or 0)
        fallback_search_hits.extend(node.get("search_hits", []) or [])

        label = str(node.get("label") or node.get("branch_id") or "node")
        for issue in node.get("issues", []) or []:
            branch_issues.append(
                {
                    "target": label,
                    "property_id_norm": "",
                    "reason": str(issue),
                    "source_node": label,
                }
            )
        for dropped in node.get("dropped_candidates", []) or []:
            rejections.append(
                {
                    "target": str(dropped.get("target") or "除外候補"),
                    "property_id_norm": str(dropped.get("property_id_norm") or ""),
                    "reason": str(dropped.get("reason") or "整合性レビューで推薦対象から除外"),
                    "source_node": str(dropped.get("source_node") or label),
                }
            )

        for candidate in node.get("candidates", []) or []:
            key = _property_key(candidate)
            existing = merged.get(key)
            if existing is None:
                merged[key] = dict(candidate)
            else:
                if _candidate_sort_key(candidate) > _candidate_sort_key(existing):
                    for field in [
                        "building_name",
                        "image_url",
                        "address",
                        "rent",
                        "layout",
                        "station_walk_min",
                        "area_m2",
                        "management_fee",
                        "deposit",
                        "key_money",
                        "available_date",
                        "agency_name",
                        "detail_url",
                        "score",
                        "reason",
                        "detail_excerpt",
                    ]:
                        existing[field] = candidate.get(field)
                existing["evidence"] = _unique_strings(
                    list(existing.get("evidence", []) or [])
                    + list(candidate.get("evidence", []) or []),
                    limit=4,
                )
                existing["matched_queries"] = _unique_strings(
                    list(existing.get("matched_queries", []) or [])
                    + list(candidate.get("matched_queries", []) or []),
                    limit=4,
                )
                existing["source_nodes"] = _unique_strings(
                    list(existing.get("source_nodes", []) or [])
                    + list(candidate.get("source_nodes", []) or []),
                    limit=4,
                )
                existing["seen_count"] = int(existing.get("seen_count") or 0) + 1
                if not str(existing.get("rejection_reason") or "").strip():
                    existing["rejection_reason"] = candidate.get("rejection_reason") or ""

            rejection_reason = str(candidate.get("rejection_reason") or "").strip()
            if rejection_reason:
                rejections.append(
                    {
                        "target": str(candidate.get("building_name") or "候補物件"),
                        "property_id_norm": str(candidate.get("property_id_norm") or ""),
                        "reason": rejection_reason,
                        "source_node": label,
                    }
                )

    merged_candidates = sorted(merged.values(), key=_candidate_sort_key, reverse=True)
    shortlisted = merged_candidates[:MAX_CANDIDATES]
    shortlisted_keys = {_property_key(item) for item in shortlisted}

    filtered_rejections: list[dict[str, Any]] = []
    seen_rejection_keys: set[tuple[str, str]] = set()
    for item in rejections + branch_issues:
        rejection_key = (
            str(item.get("property_id_norm") or item.get("target") or ""),
            str(item.get("reason") or ""),
        )
        if rejection_key in seen_rejection_keys:
            continue
        if item.get("property_id_norm") and str(item.get("property_id_norm")) in shortlisted_keys:
            continue
        seen_rejection_keys.add(rejection_key)
        filtered_rejections.append(item)
        if len(filtered_rejections) >= MAX_REJECTIONS:
            break

    common_risks: list[str] = []
    if not shortlisted and fallback_search_hits:
        common_risks.append("検索ヒットはあるが、整合性レビュー後に推薦可能候補が残っていない")
    if any(not item.get("detail_url") for item in shortlisted):
        common_risks.append("詳細ページ未取得の候補が残っている")
    if any(int(item.get("station_walk_min") or 0) <= 0 for item in shortlisted):
        common_risks.append("駅徒歩分数が未確認の候補が含まれる")
    if any(not str(item.get("layout") or "").strip() for item in shortlisted):
        common_risks.append("間取り未確認の候補が含まれる")
    if any(int(item.get("management_fee") or 0) <= 0 for item in shortlisted):
        common_risks.append("管理費や初期費用の内訳が不足している")
    if any(not str(item.get("available_date") or "").strip() for item in shortlisted):
        common_risks.append("入居可能時期が未確認の候補がある")

    open_questions: list[str] = []
    if not shortlisted and fallback_search_hits:
        open_questions.append("strict条件（対象エリア・間取り・must条件）を固定した再検索")
    for item in shortlisted:
        name = str(item.get("building_name") or "候補物件")
        if int(item.get("station_walk_min") or 0) <= 0:
            open_questions.append(f"{name} の駅徒歩分数")
        if not str(item.get("layout") or "").strip():
            open_questions.append(f"{name} の間取り")
        if float(item.get("area_m2") or 0.0) <= 0:
            open_questions.append(f"{name} の専有面積")
        if not str(item.get("available_date") or "").strip():
            open_questions.append(f"{name} の入居可能時期")
        if int(item.get("management_fee") or 0) <= 0:
            open_questions.append(f"{name} の管理費・初期費用内訳")
        if len(open_questions) >= MAX_OPEN_QUESTIONS:
            break

    property_candidates = [
        {
            "property_id_norm": str(item.get("property_id_norm") or ""),
            "building_name": str(item.get("building_name") or "候補物件"),
            "image_url": str(item.get("image_url") or ""),
            "address": str(item.get("address") or ""),
            "rent": int(item.get("rent") or 0),
            "layout": str(item.get("layout") or ""),
            "station_walk_min": int(item.get("station_walk_min") or 0),
            "area_m2": float(item.get("area_m2") or 0.0),
            "detail_url": str(item.get("detail_url") or ""),
            "score": float(item.get("score") or 0.0),
            "reason": str(item.get("reason") or ""),
            "evidence": _unique_strings(list(item.get("evidence", []) or []), limit=4),
            "matched_queries": _unique_strings(
                list(item.get("matched_queries", []) or []), limit=4
            ),
            "source_nodes": _unique_strings(list(item.get("source_nodes", []) or []), limit=4),
        }
        for item in shortlisted
    ]

    return {
        PROPERTY_CANDIDATES_KEY: property_candidates,
        REJECTION_REASONS_KEY: filtered_rejections,
        COMMON_RISKS_KEY: _unique_strings(common_risks, limit=4),
        OPEN_QUESTIONS_KEY: _unique_strings(open_questions, limit=MAX_OPEN_QUESTIONS),
        SUMMARY_KEY: {
            "branch_node_count": len(branch_nodes),
            "unique_candidate_count": len(merged_candidates),
            "shortlisted_candidate_count": len(property_candidates),
            "rejection_count": len(filtered_rejections),
            "detail_page_hit_count": detail_page_hit_count,
        },
    }


# JP: result summary schemaを処理する。
# EN: Process result summary schema.
def _result_summary_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            PROPERTY_CANDIDATES_KEY: {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "property_id_norm": {"type": "string"},
                        "building_name": {"type": "string"},
                        "image_url": {"type": "string"},
                        "address": {"type": "string"},
                        "rent": {"type": "integer", "minimum": 0},
                        "layout": {"type": "string"},
                        "station_walk_min": {"type": "integer", "minimum": 0},
                        "area_m2": {"type": "number", "minimum": 0},
                        "detail_url": {"type": "string"},
                        "score": {"type": "number"},
                        "reason": {"type": "string"},
                        "evidence": {"type": "array", "items": {"type": "string"}},
                        "matched_queries": {"type": "array", "items": {"type": "string"}},
                        "source_nodes": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [
                        "property_id_norm",
                        "building_name",
                        "image_url",
                        "address",
                        "rent",
                        "layout",
                        "station_walk_min",
                        "area_m2",
                        "detail_url",
                        "score",
                        "reason",
                        "evidence",
                        "matched_queries",
                        "source_nodes",
                    ],
                    "additionalProperties": False,
                },
            },
            REJECTION_REASONS_KEY: {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string"},
                        "property_id_norm": {"type": "string"},
                        "reason": {"type": "string"},
                        "source_node": {"type": "string"},
                    },
                    "required": ["target", "property_id_norm", "reason", "source_node"],
                    "additionalProperties": False,
                },
            },
            COMMON_RISKS_KEY: {"type": "array", "items": {"type": "string"}},
            OPEN_QUESTIONS_KEY: {"type": "array", "items": {"type": "string"}},
            SUMMARY_KEY: {
                "type": "object",
                "properties": {
                    "branch_node_count": {"type": "integer", "minimum": 0},
                    "unique_candidate_count": {"type": "integer", "minimum": 0},
                    "shortlisted_candidate_count": {"type": "integer", "minimum": 0},
                    "rejection_count": {"type": "integer", "minimum": 0},
                    "detail_page_hit_count": {"type": "integer", "minimum": 0},
                },
                "required": [
                    "branch_node_count",
                    "unique_candidate_count",
                    "shortlisted_candidate_count",
                    "rejection_count",
                    "detail_page_hit_count",
                ],
                "additionalProperties": False,
            },
        },
        "required": [
            PROPERTY_CANDIDATES_KEY,
            REJECTION_REASONS_KEY,
            COMMON_RISKS_KEY,
            OPEN_QUESTIONS_KEY,
            SUMMARY_KEY,
        ],
        "additionalProperties": False,
    }


# JP: coerce result summaryを処理する。
# EN: Process coerce result summary.
def _coerce_result_summary(
    result: dict[str, Any],
    *,
    fallback: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(result, dict):
        return fallback
    fallback_candidate_keys = {
        _property_key(candidate)
        for candidate in fallback.get(PROPERTY_CANDIDATES_KEY, []) or []
        if _property_key(candidate)
    }
    result_candidates = [
        item
        for item in result.get(PROPERTY_CANDIDATES_KEY, []) or []
        if fallback_candidate_keys and _property_key(item) in fallback_candidate_keys
    ]
    merged = {
        PROPERTY_CANDIDATES_KEY: result_candidates,
        REJECTION_REASONS_KEY: result.get(REJECTION_REASONS_KEY, fallback[REJECTION_REASONS_KEY]),
        COMMON_RISKS_KEY: result.get(COMMON_RISKS_KEY, fallback[COMMON_RISKS_KEY]),
        OPEN_QUESTIONS_KEY: result.get(OPEN_QUESTIONS_KEY, fallback[OPEN_QUESTIONS_KEY]),
        SUMMARY_KEY: result.get(SUMMARY_KEY, fallback[SUMMARY_KEY]),
    }
    if not merged[PROPERTY_CANDIDATES_KEY] and fallback_candidate_keys:
        merged[PROPERTY_CANDIDATES_KEY] = fallback[PROPERTY_CANDIDATES_KEY]
    if not isinstance(merged[SUMMARY_KEY], dict):
        merged[SUMMARY_KEY] = fallback[SUMMARY_KEY]
    return merged


# JP: result summarizerを実行する。
# EN: Run result summarizer.
def run_result_summarizer(
    *,
    branch_nodes: list[dict[str, Any]],
    adapter: LLMAdapter | None = None,
) -> dict[str, Any]:
    snapshots = [_build_node_snapshot(node) for node in branch_nodes if isinstance(node, dict)]
    fallback = _fallback_result_summary(snapshots)
    if adapter is None or not snapshots:
        return fallback

    payload = {
        "draft_summary": fallback,
        "branch_nodes": snapshots,
        "output_rules": [
            "候補物件は同一物件の重複をまとめる",
            "reason と evidence は明示的な根拠だけを書く",
            "未確定情報は 共通リスク または 未解決の調査項目 へ回す",
            "候補は最大5件、却下理由は最大8件に圧縮する",
        ],
    }
    try:
        result = adapter.generate_structured(
            system=result_summarizer_sys_msg,
            user=branch_result_aggregate_prompt.format(
                draft_summary=json.dumps(fallback, ensure_ascii=False, indent=2),
                branch_nodes=json.dumps(payload["branch_nodes"], ensure_ascii=False, indent=2),
            ),
            schema=_result_summary_schema(),
            temperature=0.1,
        )
    except Exception:
        return fallback
    return _coerce_result_summary(result, fallback=fallback)
