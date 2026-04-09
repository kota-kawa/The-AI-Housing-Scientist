from __future__ import annotations

import json
from typing import Any

from app.llm.base import LLMAdapter
from app.research.journal import ResearchNode
from app.stages.result_summarizer import (
    COMMON_RISKS_KEY,
    OPEN_QUESTIONS_KEY,
    PROPERTY_CANDIDATES_KEY,
)


final_report_system_msg = """You are a Japanese rental research analyst.
You are given a compact research journal for one completed apartment search.
Your task is to produce a clear markdown report that explains:
- how the exploration proceeded,
- why the selected branch was chosen,
- how the shortlisted properties compare,
- what risks remain,
- which property is recommended and why.

Important instructions:
- Use only the provided facts.
- Do not invent candidates, metrics, or conditions.
- Keep the report decision-oriented and easy to reuse as an internal memo or customer-facing explanation.
"""

final_report_prompt = """You are given:

1) Stage-level journal summaries:
{stage_nodes}

2) Nodes on the selected branch:
{selected_branch_nodes}

3) A draft markdown report:
{draft_report}

Write a final markdown report with the following sections in this order:
1. 探索経路
2. 候補比較表
3. リスク
4. 推奨物件と根拠
5. 追加調査の提案

Requirements:
- Keep headings in Japanese.
- The comparison section must include a markdown table.
- If information is missing, state that clearly instead of guessing.
- Reuse the draft when it is already correct, but improve flow and clarity.
"""


def _compact_text(value: Any, *, max_chars: int = 220) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _node_snapshot(node: ResearchNode) -> dict[str, Any]:
    summary = ""
    if isinstance(node.output_payload, dict):
        summary = str(
            node.output_payload.get("summary")
            or node.metrics.get("summary")
            or node.reasoning
            or ""
        ).strip()
    return {
        "id": node.id,
        "stage": node.stage,
        "node_type": node.node_type,
        "status": node.status,
        "intent": node.intent,
        "is_failed": node.is_failed,
        "debug_depth": node.debug_depth,
        "branch_id": node.branch_id,
        "selected": node.selected,
        "summary": _compact_text(summary, max_chars=240),
        "reasoning": _compact_text(node.reasoning, max_chars=220),
        "metrics": {
            key: value
            for key, value in (node.metrics or {}).items()
            if key
            in {
                "branch_id",
                "label",
                "depth",
                "branch_score",
                "frontier_score",
                "detail_coverage",
                "structured_ratio",
                "normalized_count",
                "top_issue_class",
                "strategy_tags",
                "intent",
                "is_failed",
                "debug_depth",
            }
        },
        "output": {
            key: value
            for key, value in (node.output_payload or {}).items()
            if key
            in {
                "summary",
                "selected_branch",
                "selected_path",
                "search_tree_summary",
                "offline_evaluation",
                "failure_summary",
                "research_summary",
            }
        },
    }


def _selected_context(selected_branch_nodes: list[ResearchNode]) -> dict[str, Any]:
    selection_node = next(
        (
            node
            for node in reversed(selected_branch_nodes)
            if node.node_type == "search_selection" and node.selected
        ),
        None,
    )
    selected_branch = {}
    selected_path: list[dict[str, Any]] = []
    branch_result_summary: dict[str, Any] = {}
    search_tree_summary: dict[str, Any] = {}

    if selection_node is not None:
        selected_branch = dict(selection_node.output_payload.get("selected_branch", {}) or {})
        selected_path = list(selection_node.output_payload.get("selected_path", []) or [])
        search_tree_summary = dict(selection_node.output_payload.get("search_tree_summary", {}) or {})
        branch_result_summary = dict(selected_branch.get("branch_result_summary", {}) or {})
        if not branch_result_summary:
            branch_result_summary = dict(selection_node.metrics.get("branch_result_summary", {}) or {})

    return {
        "selected_branch": selected_branch,
        "selected_path": selected_path,
        "branch_result_summary": branch_result_summary,
        "search_tree_summary": search_tree_summary,
    }


def _synthesize_context(
    stage_nodes: list[ResearchNode],
) -> dict[str, Any]:
    synthesize_node = next(
        (node for node in reversed(stage_nodes) if node.stage == "synthesize"),
        None,
    )
    if synthesize_node is None:
        return {
            "offline_evaluation": {},
            "failure_summary": {},
            "research_summary": "",
        }
    return {
        "offline_evaluation": dict(synthesize_node.output_payload.get("offline_evaluation", {}) or {}),
        "failure_summary": dict(synthesize_node.output_payload.get("failure_summary", {}) or {}),
        "research_summary": str(synthesize_node.output_payload.get("research_summary") or "").strip(),
    }


def _comparison_table(candidates: list[dict[str, Any]]) -> str:
    if not candidates:
        return "| 物件 | 家賃 | 間取り | 駅徒歩 | 面積 | 根拠 |\n| --- | --- | --- | --- | --- | --- |\n| 候補なし | 要確認 | 要確認 | 要確認 | 要確認 | 候補比較に必要な情報が不足しています。 |"

    lines = [
        "| 物件 | 家賃 | 間取り | 駅徒歩 | 面積 | 根拠 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in candidates[:5]:
        rent = int(item.get("rent") or 0)
        walk = int(item.get("station_walk_min") or 0)
        area = float(item.get("area_m2") or 0.0)
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item.get("building_name") or "候補物件"),
                    f"{rent:,}円" if rent > 0 else "要確認",
                    str(item.get("layout") or "要確認"),
                    f"{walk}分" if walk > 0 else "要確認",
                    f"{area:.1f}㎡" if area > 0 else "要確認",
                    _compact_text(item.get("reason") or "比較中の候補です。", max_chars=80).replace("|", "/"),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _build_fallback_report(
    *,
    stage_nodes: list[ResearchNode],
    selected_branch_nodes: list[ResearchNode],
) -> tuple[str, dict[str, Any]]:
    selected_context = _selected_context(selected_branch_nodes)
    synthesize_context = _synthesize_context(stage_nodes)
    selected_path = list(selected_context["selected_path"] or [])
    branch_result_summary = dict(selected_context["branch_result_summary"] or {})
    selected_branch = dict(selected_context["selected_branch"] or {})
    search_tree_summary = dict(selected_context["search_tree_summary"] or {})
    offline_evaluation = dict(synthesize_context["offline_evaluation"] or {})
    failure_summary = dict(synthesize_context["failure_summary"] or {})
    research_summary = str(synthesize_context["research_summary"] or "").strip()
    candidates = list(branch_result_summary.get(PROPERTY_CANDIDATES_KEY, []) or [])
    common_risks = [str(item).strip() for item in branch_result_summary.get(COMMON_RISKS_KEY, []) or [] if str(item).strip()]
    open_questions = [str(item).strip() for item in branch_result_summary.get(OPEN_QUESTIONS_KEY, []) or [] if str(item).strip()]

    if not common_risks:
        common_risks = [
            str(item).strip()
            for item in failure_summary.get("top_issues", []) or []
            if str(item).strip()
        ]
    if not open_questions:
        open_questions = [
            str(item).strip()
            for item in offline_evaluation.get("recommendations", []) or []
            if str(item).strip()
        ]

    path_lines: list[str] = []
    for index, item in enumerate(selected_path, start=1):
        label = str(item.get("label") or item.get("branch_id") or f"node-{index}")
        depth = int(item.get("depth") or 0)
        tags = [str(tag).strip() for tag in item.get("strategy_tags", []) or [] if str(tag).strip()]
        score = float(item.get("branch_score") or 0.0)
        path_lines.append(
            f"{index}. {label}（depth {depth}, score {round(score, 2)}, 戦略: {'/'.join(tags[:3]) or 'なし'}）"
        )
    if not path_lines:
        path_lines.append("1. 選択された探索パスの記録が不足しています。")
    if search_tree_summary:
        path_lines.append(
            f"- 探索全体では {search_tree_summary.get('executed_node_count', 0)} ノードを評価し、終了理由は {search_tree_summary.get('termination_reason', 'unknown')} でした。"
        )

    top_candidate = candidates[0] if candidates else {}
    if top_candidate:
        top_name = str(top_candidate.get("building_name") or "候補物件")
        top_reason = str(top_candidate.get("reason") or "条件一致度が高い候補です。")
        recommendation_text = f"{top_name} を推奨します。{top_reason}"
    elif research_summary:
        top_name = str(selected_branch.get("label") or "今回の選定結果")
        recommendation_text = research_summary
    else:
        top_name = str(selected_branch.get("label") or "推奨候補なし")
        recommendation_text = "現時点では問い合わせ推奨まで十分に整理できていません。"

    risk_lines = [f"- {item}" for item in common_risks[:5]] or ["- 重大な共通リスクは明示されていません。"]
    follow_up_lines = [f"- {item}" for item in open_questions[:6]] or ["- 追加調査の提案はまだありません。"]
    report = "\n".join(
        [
            "# 最終レポート",
            "",
            "## 探索経路",
            *path_lines,
            "",
            "## 候補比較表",
            _comparison_table(candidates),
            "",
            "## リスク",
            *risk_lines,
            "",
            "## 推奨物件と根拠",
            recommendation_text,
            "",
            "## 追加調査の提案",
            *follow_up_lines,
        ]
    ).strip()
    return report, {
        "selected_branch_id": str(selected_branch.get("branch_id") or ""),
        "candidate_count": len(candidates),
        "risk_count": len(common_risks),
        "open_question_count": len(open_questions),
    }


def run_final_report(
    *,
    stage_nodes: list[ResearchNode],
    selected_branch_nodes: list[ResearchNode],
    adapter: LLMAdapter | None = None,
) -> dict[str, Any]:
    draft_report, summary = _build_fallback_report(
        stage_nodes=stage_nodes,
        selected_branch_nodes=selected_branch_nodes,
    )
    if adapter is None:
        return {
            "report_markdown": draft_report,
            "summary": summary,
            "llm_applied": False,
        }

    stage_payload = [_node_snapshot(node) for node in stage_nodes]
    branch_payload = [_node_snapshot(node) for node in selected_branch_nodes]
    try:
        report_markdown = adapter.generate_text(
            system=final_report_system_msg,
            user=final_report_prompt.format(
                stage_nodes=_compact_json(stage_payload),
                selected_branch_nodes=_compact_json(branch_payload),
                draft_report=draft_report,
            ),
            temperature=0.2,
        ).strip()
    except Exception:
        report_markdown = ""

    if not report_markdown:
        report_markdown = draft_report

    return {
        "report_markdown": report_markdown,
        "summary": summary,
        "llm_applied": report_markdown != draft_report,
    }
