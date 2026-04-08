from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from app.llm.base import LLMAdapter


def _structured_property_ratio(normalized_properties: list[dict[str, Any]]) -> float:
    if not normalized_properties:
        return 0.0

    complete = 0
    for prop in normalized_properties:
        checks = [
            int(prop.get("rent") or 0) > 0,
            int(prop.get("station_walk_min") or 0) > 0,
            bool(prop.get("layout")),
            bool(prop.get("address_norm") or prop.get("address")),
        ]
        if sum(checks) >= 3:
            complete += 1
    return round(complete / len(normalized_properties), 3)


def _score_from_range(value: float, *, min_value: float, max_value: float) -> float:
    if max_value <= min_value:
        return 0.0
    clamped = min(max(value, min_value), max_value)
    return (clamped - min_value) / (max_value - min_value)


def _classify_top_issue(issues: list[str]) -> str:
    joined = " / ".join(issues)
    if "検索結果" in joined:
        return "result_empty"
    if "詳細ページ補完率" in joined:
        return "detail_low"
    if "欠損" in joined:
        return "schema_sparse"
    if "条件一致度" in joined:
        return "match_low"
    if "情報源" in joined:
        return "source_low"
    return "healthy"


def _expand_recommendations_from_issues(issues: list[str], top_score: float) -> list[str]:
    recommendations: list[str] = []
    joined = " / ".join(issues)
    if "詳細ページ補完率" in joined:
        recommendations.extend(["detail_first", "source_diversify"])
    if "欠損" in joined:
        recommendations.extend(["schema_first", "tighten_match"])
    if "条件一致度" in joined:
        recommendations.extend(["tighten_match", "relax_for_coverage"])
    if "情報源" in joined:
        recommendations.extend(["source_diversify", "relax_for_coverage"])
    if not recommendations and top_score >= 60:
        recommendations.extend(["exploit_best", "explore_adjacent"])
    elif not recommendations:
        recommendations.extend(["relax_for_coverage", "source_diversify"])
    deduped: list[str] = []
    for item in recommendations:
        if item not in deduped:
            deduped.append(item)
    return deduped[:2]


def evaluate_branch(
    *,
    branch_id: str,
    label: str,
    queries: list[str],
    raw_results: list[dict[str, Any]],
    normalized_properties: list[dict[str, Any]],
    ranked_properties: list[dict[str, Any]],
    duplicate_groups: list[dict[str, Any]],
    search_summary: dict[str, Any],
    parent_summary: dict[str, Any] | None = None,
    strategy_tags: list[str] | None = None,
    depth: int = 0,
    query_hash: str = "",
) -> dict[str, Any]:
    normalized_count = len(normalized_properties)
    detail_hit_count = int(search_summary.get("detail_hit_count", 0) or 0)
    detail_coverage = round(detail_hit_count / normalized_count, 3) if normalized_count else 0.0
    structured_ratio = _structured_property_ratio(normalized_properties)
    avg_top3_score = 0.0
    top_score = 0.0
    if ranked_properties:
        top_scores = [float(item.get("score") or 0.0) for item in ranked_properties[:3]]
        avg_top3_score = round(sum(top_scores) / len(top_scores), 3)
        top_score = round(top_scores[0], 3)

    source_diversity = len(
        {
            str(item.get("source_name") or "")
            for item in raw_results
            if str(item.get("source_name") or "").strip()
        }
    )
    duplicate_ratio = (
        round(len(duplicate_groups) / normalized_count, 3) if normalized_count else 0.0
    )

    score = 0.0
    score += _score_from_range(normalized_count, min_value=0, max_value=6) * 20.0
    score += detail_coverage * 25.0
    score += structured_ratio * 20.0
    score += _score_from_range(avg_top3_score, min_value=35.0, max_value=95.0) * 25.0
    score += _score_from_range(source_diversity, min_value=1, max_value=3) * 10.0
    score -= duplicate_ratio * 5.0

    issues: list[str] = []
    recommendations: list[str] = []
    if not raw_results:
        issues.append("検索結果が取得できていない")
        recommendations.append("エリアや予算の条件を緩めた分岐を優先する")
    if normalized_count == 0:
        issues.append("正規化後の候補が残っていない")
        recommendations.append("詳細ページが取れないURLを減らし、取得元を増やす")
    if detail_coverage < 0.4:
        issues.append("詳細ページ補完率が低い")
        recommendations.append("詳細ページ取得を優先する検索分岐を残す")
    if structured_ratio < 0.5:
        issues.append("比較に必要な項目の欠損が多い")
        recommendations.append("家賃・駅徒歩・間取りの取得率を優先して評価する")
    if top_score < 60:
        issues.append("上位候補の条件一致度が低い")
        recommendations.append("must条件を緩めた探索と strict 探索を分離して評価する")
    if source_diversity <= 1:
        issues.append("情報源の多様性が低い")
        recommendations.append("Brave と catalog の両系統で候補を確保する")

    summary = (
        f"{label}: score={round(score, 2)}, "
        f"候補{normalized_count}件, 詳細補完率{detail_coverage:.0%}, "
        f"構造化率{structured_ratio:.0%}, 上位平均{avg_top3_score:.1f}"
    )
    issue_class = _classify_top_issue(issues)
    parent_score = float((parent_summary or {}).get("branch_score") or 0.0)
    delta_from_parent = round(score - parent_score, 2) if parent_summary is not None else round(score, 2)
    frontier_score = round(score, 2)
    prune_reasons: list[str] = []

    return {
        "branch_id": branch_id,
        "node_key": branch_id,
        "label": label,
        "status": "completed",
        "depth": depth,
        "query_count": len(queries),
        "queries": queries,
        "query_hash": query_hash,
        "strategy_tags": [str(tag).strip() for tag in strategy_tags or [] if str(tag).strip()],
        "raw_result_count": len(raw_results),
        "normalized_count": normalized_count,
        "detail_hit_count": detail_hit_count,
        "detail_coverage": detail_coverage,
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_ratio": duplicate_ratio,
        "structured_ratio": structured_ratio,
        "top_score": top_score,
        "avg_top3_score": avg_top3_score,
        "source_diversity": source_diversity,
        "branch_score": round(score, 2),
        "frontier_score": frontier_score,
        "delta_from_parent": delta_from_parent,
        "top_issue_class": issue_class,
        "expand_recommendations": _expand_recommendations_from_issues(issues, top_score),
        "prune_reasons": prune_reasons,
        "issues": issues,
        "recommendations": recommendations,
        "summary": summary,
    }


def select_best_branch(branch_summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not branch_summaries:
        return None
    if not any(item.get("status") == "completed" for item in branch_summaries):
        return None
    return sorted(
        branch_summaries,
        key=lambda item: (
            1 if item.get("status") == "completed" else 0,
            float(item.get("branch_score") or 0.0),
            float(item.get("frontier_score") or 0.0),
            float(item.get("detail_coverage") or 0.0),
            float(item.get("avg_top3_score") or 0.0),
            int(item.get("normalized_count") or 0),
            item.get("label", ""),
        ),
        reverse=True,
    )[0]


def summarize_branch_failures(branch_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not branch_summaries:
        return {
            "summary": "branch search の評価対象がありませんでした。",
            "top_issues": ["branch search が開始前に失敗しました。"],
            "recommendations": ["research job の入力条件と検索ソース設定を再確認する"],
        }

    issue_counter: Counter[str] = Counter()
    recommendation_counter: Counter[str] = Counter()
    for item in branch_summaries:
        for issue in item.get("issues", []) or []:
            issue_counter[str(issue)] += 1
        for recommendation in item.get("recommendations", []) or []:
            recommendation_counter[str(recommendation)] += 1

    top_issues = [issue for issue, _ in issue_counter.most_common(3)]
    recommendations = [rec for rec, _ in recommendation_counter.most_common(3)]
    summary = (
        "branch search の結果、十分な候補を安定的に確保できませんでした。"
        if top_issues
        else "branch search の結果はありますが、改善余地のある失敗パターンは見つかりませんでした。"
    )
    return {
        "summary": summary,
        "top_issues": top_issues,
        "recommendations": recommendations,
    }


def evaluate_final_result(
    *,
    selected_branch_summary: dict[str, Any] | None,
    visible_ranked_properties: list[dict[str, Any]],
    search_summary: dict[str, Any],
) -> dict[str, Any]:
    visible_count = len(visible_ranked_properties)
    detail_coverage = float(selected_branch_summary.get("detail_coverage") or 0.0) if selected_branch_summary else 0.0
    structured_ratio = float(selected_branch_summary.get("structured_ratio") or 0.0) if selected_branch_summary else 0.0

    readiness = "low"
    if visible_count >= 3 and detail_coverage >= 0.5 and structured_ratio >= 0.6:
        readiness = "high"
    elif visible_count >= 1 and detail_coverage >= 0.3:
        readiness = "medium"

    recommendations: list[str] = []
    if readiness == "low":
        recommendations.append("条件の優先順位を確認して branch search を再実行する")
    if detail_coverage < 0.5:
        recommendations.append("詳細ページ補完率を上げる取得戦略を優先する")
    if structured_ratio < 0.6:
        recommendations.append("家賃・徒歩・間取りの欠損ペナルティを強める")
    if not recommendations:
        recommendations.append("上位候補の契約条件確認へ進む")

    return {
        "readiness": readiness,
        "visible_candidate_count": visible_count,
        "detail_coverage": detail_coverage,
        "structured_ratio": structured_ratio,
        "search_summary": search_summary,
        "recommendations": recommendations,
    }


def generate_branch_selection_rationale(
    *,
    selected_branch: dict[str, Any],
    other_branches: list[dict[str, Any]],
    adapter: LLMAdapter,
) -> str:
    schema = {
        "type": "object",
        "properties": {
            "why_selected": {"type": "string"},
            "improvement_suggestions": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["why_selected", "improvement_suggestions"],
        "additionalProperties": False,
    }

    def _branch_metrics(branch: dict[str, Any]) -> dict[str, Any]:
        return {
            "branch_id": branch.get("branch_id", ""),
            "label": branch.get("label", ""),
            "branch_score": branch.get("branch_score", 0),
            "normalized_count": branch.get("normalized_count", 0),
            "detail_coverage": branch.get("detail_coverage", 0),
            "structured_ratio": branch.get("structured_ratio", 0),
            "avg_top3_score": branch.get("avg_top3_score", 0),
            "source_diversity": branch.get("source_diversity", 0),
            "duplicate_ratio": branch.get("duplicate_ratio", 0),
            "issues": branch.get("issues", []),
        }

    payload = {
        "selected_branch": _branch_metrics(selected_branch),
        "other_branches": [_branch_metrics(b) for b in other_branches],
        "output_rules": [
            "why_selected: 選ばれた分岐が他より優れていた理由を2〜3文の日本語で説明",
            "improvement_suggestions: 次回の探索で試すべき改善点のリスト（最大3件）",
        ],
    }
    try:
        result = adapter.generate_structured(
            system=(
                "You analyze branch evaluation metrics from a Japanese rental property research system "
                "and explain why the selected branch outperformed alternatives. "
                "Focus on concrete metric differences and actionable next steps."
            ),
            user=json.dumps(payload, ensure_ascii=False, indent=2),
            schema=schema,
            temperature=0.1,
        )
        why = str(result.get("why_selected") or "").strip()
        suggestions = [str(s).strip() for s in (result.get("improvement_suggestions") or []) if str(s).strip()]
        parts = [why] + suggestions
        return " / ".join(p for p in parts if p)
    except Exception:
        return ""


def load_offline_eval_cases(path: str | Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_offline_eval_case(
    case: dict[str, Any],
    *,
    adapter: LLMAdapter | None = None,
) -> dict[str, Any]:
    branch_summaries = [
        evaluate_branch(
            branch_id=str(branch["branch_id"]),
            label=str(branch["label"]),
            queries=list(branch.get("queries", [])),
            raw_results=list(branch.get("raw_results", [])),
            normalized_properties=list(branch.get("normalized_properties", [])),
            ranked_properties=list(branch.get("ranked_properties", [])),
            duplicate_groups=list(branch.get("duplicate_groups", [])),
            search_summary=dict(branch.get("search_summary", {})),
        )
        for branch in case.get("branches", [])
    ]
    selected_branch = select_best_branch(branch_summaries)
    final_result = evaluate_final_result(
        selected_branch_summary=selected_branch,
        visible_ranked_properties=list(case.get("visible_ranked_properties", [])),
        search_summary=dict(case.get("search_summary", {})),
    )
    failure_summary = summarize_branch_failures(branch_summaries)

    analysis: dict[str, Any] = {}
    if adapter is not None and selected_branch is not None and len(branch_summaries) > 1:
        other_branches = [b for b in branch_summaries if b.get("branch_id") != selected_branch.get("branch_id")]
        rationale = generate_branch_selection_rationale(
            selected_branch=selected_branch,
            other_branches=other_branches,
            adapter=adapter,
        )
        if rationale:
            analysis["rationale"] = rationale

    expectations = case.get("expectations", {}) or {}
    checks = {
        "selected_branch_id": selected_branch is not None
        and selected_branch.get("branch_id") == expectations.get("selected_branch_id"),
        "readiness": final_result.get("readiness") == expectations.get("readiness"),
        "top_issue": (
            not expectations.get("top_issue_contains")
            or expectations.get("top_issue_contains") in failure_summary.get("top_issues", [])
        ),
    }
    result: dict[str, Any] = {
        "name": case.get("name", "unnamed"),
        "selected_branch": selected_branch,
        "final_result": final_result,
        "failure_summary": failure_summary,
        "checks": checks,
        "passed": all(checks.values()),
    }
    if analysis:
        result["analysis"] = analysis
    return result


def run_offline_eval_suite(
    cases: list[dict[str, Any]],
    *,
    adapter: LLMAdapter | None = None,
) -> dict[str, Any]:
    case_results = [run_offline_eval_case(case, adapter=adapter) for case in cases]
    passed_count = sum(1 for case in case_results if case["passed"])
    return {
        "case_count": len(case_results),
        "passed_count": passed_count,
        "failed_count": len(case_results) - passed_count,
        "cases": case_results,
    }
