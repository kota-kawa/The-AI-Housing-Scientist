from __future__ import annotations

import asyncio
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import hashlib
import json
import time
from typing import Any

from app.research.journal import ResearchIntent
from app.research.offline_eval import (
    BRANCH_FAMILY_PRIORITY,
    branch_selection_sort_key,
    evaluate_branch,
    evaluate_final_result,
    is_branch_selection_eligible,
    select_best_branch,
)
from app.stages.result_summarizer import run_result_summarizer

from .agent_manager_types import ResearchExecutionState, SearchNodeArtifacts, SearchNodePlan


class AgentManagerTreeMixin:
    # JP: active user memoryを処理する。
    # EN: Process active user memory.
    def _active_user_memory(self) -> dict[str, Any]:
        return self.approved_plan.get("user_memory_snapshot", self.user_memory)

    # JP: strategy memoryを処理する。
    # EN: Process strategy memory.
    def _strategy_memory(self) -> dict[str, Any]:
        learned = self._active_user_memory().get("learned_preferences", {}) or {}
        strategy = learned.get("strategy_memory", {}) or {}
        if strategy:
            return strategy
        return self.task_memory.get("strategy_memory_snapshot", {}) or {}

    # JP: compose queryを処理する。
    # EN: Process compose query.
    def _compose_query(self, *parts: Any) -> str:
        return " ".join(str(part).strip() for part in parts if str(part).strip()).strip()

    # JP: dedupe queriesを処理する。
    # EN: Process dedupe queries.
    def _dedupe_queries(self, values: list[str], *, limit: int = 5) -> list[str]:
        deduped: list[str] = []
        for value in values:
            text = " ".join(str(value).split()).strip()
            if text and text not in deduped:
                deduped.append(text)
            if len(deduped) >= limit:
                break
        return deduped

    # JP: hash queriesを処理する。
    # EN: Process hash queries.
    def _hash_queries(self, queries: list[str], ranking_profile: dict[str, Any]) -> str:
        payload = {
            "queries": [" ".join(str(item).split()) for item in queries],
            "ranking_profile": ranking_profile,
        }
        return hashlib.sha1(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]

    # JP: next node keyを処理する。
    # EN: Process next node key.
    def _next_node_key(self, state: ResearchExecutionState, operator: str, depth: int) -> str:
        state.node_sequence += 1
        return f"{operator}-d{depth}-n{state.node_sequence}"

    # JP: ranking profileを結合する。
    # EN: Merge ranking profile.
    def _merge_ranking_profile(
        self,
        base_profile: dict[str, Any],
        updates: dict[str, float],
    ) -> dict[str, Any]:
        merged = dict(base_profile)
        for key, value in updates.items():
            merged[key] = float(value)
        return merged

    # JP: LLM query suggestionsを処理する。
    # EN: Process LLM query suggestions.
    def _llm_query_suggestions(
        self,
        *,
        operator: str,
        base_queries: list[str],
        user_memory: dict[str, Any],
    ) -> list[str]:
        if self.research_adapter is None:
            return []
        schema = {
            "type": "object",
            "properties": {
                "query_suggestions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 2,
                }
            },
            "required": ["query_suggestions"],
            "additionalProperties": False,
        }
        try:
            result = self.research_adapter.generate_structured(
                system=(
                    "あなたは日本の賃貸検索クエリの改善を提案するアシスタントです。"
                    "提供された条件に基づいた簡潔なクエリ候補を最大2件返してください。"
                    "サイト名やスクレイピング指示は含めないでください。"
                ),
                user=json.dumps(
                    {
                        "operator": operator,
                        "base_queries": base_queries[:4],
                        "user_memory": {
                            "target_area": str(user_memory.get("target_area") or ""),
                            "budget_max": int(user_memory.get("budget_max") or 0),
                            "station_walk_max": int(user_memory.get("station_walk_max") or 0),
                            "layout_preference": str(user_memory.get("layout_preference") or ""),
                            "must_conditions": user_memory.get("must_conditions", []) or [],
                            "nice_to_have": user_memory.get("nice_to_have", []) or [],
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                schema=schema,
                temperature=0.2,
            )
        except Exception:
            return []
        return self._dedupe_queries(
            [
                str(item).strip()
                for item in result.get("query_suggestions", [])
                if str(item).strip()
            ],
            limit=2,
        )

    # JP: queries for operatorを処理する。
    # EN: Process queries for operator.
    def _queries_for_operator(
        self,
        *,
        base_queries: list[str],
        operator: str,
        user_memory: dict[str, Any],
        area_scope: str,
        constraint_mode: str,
        nearby_hints: list[str] | None = None,
    ) -> list[str]:
        area = str(user_memory.get("target_area") or "").strip()
        layout = str(user_memory.get("layout_preference") or "").strip()
        budget = int(user_memory.get("budget_max") or 0)
        walk = int(user_memory.get("station_walk_max") or 0)
        lt = str(user_memory.get("listing_type") or "").strip()
        must_conditions = [
            str(item).strip()
            for item in user_memory.get("must_conditions", []) or []
            if str(item).strip()
        ]
        nice_to_have = [
            str(item).strip()
            for item in user_memory.get("nice_to_have", []) or []
            if str(item).strip()
        ]
        family_queries = self.build_branch_family_queries(
            user_memory,
            base_queries,
            area_scope=area_scope,
            constraint_mode=constraint_mode,
        )
        nearby_hints = [str(item).strip() for item in nearby_hints or [] if str(item).strip()]
        location_tokens = nearby_hints[:2] if area_scope == "nearby" and nearby_hints else [area]
        location_text = " ".join(token for token in location_tokens if token).strip() or area
        budget_token = f"{int(budget / 10000)}万円" if budget else ""
        walk_token = f"徒歩{walk}分" if walk else ""
        core_must = " ".join(must_conditions[:2]).strip() if constraint_mode == "primary" else ""
        core_nice = " ".join(nice_to_have[:2]).strip()

        if operator in {
            "strict_primary",
            "strict_relaxed",
            "nearby_primary",
            "nearby_relaxed",
        }:
            queries = list(family_queries)
        else:
            queries = list(base_queries or family_queries)
            if operator == "source_diversify":
                lt_info = f"{lt}情報" if lt else "物件情報"
                queries.extend(
                    [
                        self._compose_query(location_text, layout, lt_info),
                        self._compose_query(location_text, layout, "募集", lt),
                        self._compose_query(location_text, "不動産", lt),
                    ]
                )
            elif operator == "detail_first":
                queries.extend(
                    [
                        self._compose_query(location_text, layout, budget_token, "設備", lt),
                        self._compose_query(location_text, layout, "詳細", lt),
                        self._compose_query(location_text, "初期費用", lt),
                    ]
                )
            elif operator == "schema_first":
                queries.extend(
                    [
                        self._compose_query(location_text, layout, "設備", lt),
                        self._compose_query(location_text, layout, "間取り", lt),
                        self._compose_query(location_text, walk_token or "徒歩", lt),
                    ]
                )
            elif operator == "exploit_best":
                queries.extend(
                    [
                        self._compose_query(location_text, layout, budget_token, walk_token, lt),
                        self._compose_query(location_text, layout, core_must, core_nice, lt),
                    ]
                )

        queries.extend(
            self._llm_query_suggestions(
                operator=operator,
                base_queries=(queries or family_queries)[:4],
                user_memory=user_memory,
            )
        )
        return self._dedupe_queries(queries)

    # JP: profile for operatorを処理する。
    # EN: Process profile for operator.
    def _profile_for_operator(
        self,
        *,
        base_profile: dict[str, Any],
        operator: str,
        area_scope: str,
        constraint_mode: str,
    ) -> dict[str, Any]:
        profile = dict(base_profile)
        if area_scope == "nearby":
            profile = self._merge_ranking_profile(
                profile,
                {
                    "area_match_bonus": 12.0,
                    "area_municipality_bonus": 8.0,
                    "area_nearby_bonus": 18.0,
                    "area_partial_bonus": 4.0,
                    "area_miss_penalty": 30.0,
                },
            )
        if constraint_mode == "relaxed":
            profile = self._merge_ranking_profile(
                profile,
                {
                    "budget_near_bonus": 8.0,
                    "budget_far_penalty": 12.0,
                    "station_far_penalty": 6.0,
                },
            )
        if operator == "detail_first":
            return self._merge_ranking_profile(
                profile,
                {
                    "rent_missing_penalty": 24.0,
                    "station_missing_penalty": 12.0,
                    "layout_missing_penalty": 12.0,
                },
            )
        if operator == "schema_first":
            return self._merge_ranking_profile(
                profile,
                {
                    "rent_missing_penalty": 28.0,
                    "station_missing_penalty": 16.0,
                    "layout_missing_penalty": 16.0,
                },
            )
        if operator == "exploit_best":
            return self._merge_ranking_profile(
                profile,
                {
                    "budget_match_bonus": 28.0,
                    "station_match_bonus": 18.0,
                    "layout_match_bonus": 14.0,
                },
            )
        return profile

    # JP: operator labelを処理する。
    # EN: Process operator label.
    def _operator_label(self, operator: str) -> str:
        labels = {
            "strict_primary": "strict条件",
            "strict_relaxed": "条件緩和",
            "nearby_primary": "近隣候補",
            "nearby_relaxed": "近隣+条件緩和",
            "source_diversify": "情報源分散",
            "detail_first": "詳細優先",
            "schema_first": "項目充足優先",
            "exploit_best": "有望条件の深掘り",
        }
        return labels.get(operator, operator)

    # JP: operator descriptionを処理する。
    # EN: Process operator description.
    def _operator_description(self, operator: str) -> str:
        descriptions = {
            "strict_primary": "対象エリア固定・must維持で主候補を探す探索",
            "strict_relaxed": "対象エリア固定・条件緩和で代替候補を探す探索",
            "nearby_primary": "近隣エリアで must 条件を維持した候補探索",
            "nearby_relaxed": "近隣エリアで条件を緩めた候補探索",
            "source_diversify": "異なる検索表現で情報源の多様性を増やす探索",
            "detail_first": "詳細ページ取得率を優先する探索",
            "schema_first": "家賃・徒歩・間取りの取得率を優先する探索",
            "exploit_best": "有望な条件組み合わせを深掘りする探索",
        }
        return descriptions.get(operator, "探索ノード")

    # JP: operator intentを処理する。
    # EN: Process operator intent.
    def _operator_intent(self, operator: str) -> ResearchIntent:
        if operator in {"strict_relaxed", "nearby_primary", "nearby_relaxed", "source_diversify"}:
            return "pivot"
        if operator in {
            "strict_primary",
            "detail_first",
            "schema_first",
            "exploit_best",
        }:
            return "refine"
        return "refine"

    # JP: recovery source summaryかどうかを判定する。
    # EN: Check whether recovery source summary.
    def _is_recovery_source_summary(self, summary: dict[str, Any] | None) -> bool:
        if not summary:
            return False
        if str(summary.get("status") or "").strip() != "completed":
            return True
        return bool(summary.get("prune_reasons"))

    # JP: intent for child planを処理する。
    # EN: Process intent for child plan.
    def _intent_for_child_plan(
        self,
        *,
        operator: str,
        parent_summary: dict[str, Any] | None,
    ) -> ResearchIntent:
        if parent_summary is None:
            return "draft"
        if self._is_recovery_source_summary(parent_summary):
            return "recovery"
        return self._operator_intent(operator)

    # JP: debug depth for child planを処理する。
    # EN: Process debug depth for child plan.
    def _debug_depth_for_child_plan(self, parent_summary: dict[str, Any] | None) -> int:
        if not self._is_recovery_source_summary(parent_summary):
            return 0
        return max(0, int((parent_summary or {}).get("debug_depth") or 0)) + 1

    # JP: available tree operatorsを処理する。
    # EN: Process available tree operators.
    def _available_tree_operators(self) -> list[str]:
        return ["source_diversify", "detail_first", "schema_first", "exploit_best"]

    # JP: operators for issue hintsを処理する。
    # EN: Process operators for issue hints.
    def _operators_for_issue_hints(self, issues: list[str]) -> list[str]:
        joined = " / ".join(str(item).strip() for item in issues if str(item).strip())
        operators: list[str] = []
        if "検索結果" in joined:
            operators.extend(["source_diversify", "detail_first"])
        if "詳細ページ補完率" in joined:
            operators.extend(["detail_first", "schema_first"])
        if "欠損" in joined:
            operators.extend(["schema_first", "detail_first"])
        if "条件一致度" in joined:
            operators.extend(["exploit_best", "source_diversify"])
        if "情報源" in joined:
            operators.extend(["source_diversify", "detail_first"])
        deduped: list[str] = []
        for operator in operators:
            if operator not in deduped:
                deduped.append(operator)
        return deduped

    # JP: nearby hintsをクエリから抽出する。
    # EN: Extract nearby hints from family queries.
    def _nearby_hints_from_queries(
        self,
        queries: list[str],
        *,
        user_memory: dict[str, Any],
    ) -> list[str]:
        target_area = str(user_memory.get("target_area") or "").strip()
        blocked = {
            target_area,
            str(user_memory.get("listing_type") or "").strip(),
            str(user_memory.get("layout_preference") or "").strip(),
            "賃貸",
            "売買",
            "住みやすい",
            "物件情報",
            "情報",
            "募集",
            "不動産",
            "設備",
            "詳細",
            "初期費用",
            "間取り",
            "徒歩",
            "駅近",
            "候補",
        }
        hints: list[str] = []
        for query in queries:
            for token in [str(item).strip() for item in str(query or "").split() if str(item).strip()]:
                if token in blocked:
                    continue
                if "万円" in token or "徒歩" in token or any(char.isdigit() for char in token):
                    continue
                if token not in hints:
                    hints.append(token)
        return hints[:3]

    # JP: initial branch familiesを処理する。
    # EN: Process initial branch families.
    def _initial_branch_families(self) -> list[dict[str, Any]]:
        return [
            {
                "operator": "strict_primary",
                "branch_family": "strict_primary",
                "area_scope": "strict",
                "constraint_mode": "primary",
            },
            {
                "operator": "strict_relaxed",
                "branch_family": "strict_relaxed",
                "area_scope": "strict",
                "constraint_mode": "relaxed",
            },
            {
                "operator": "nearby_primary",
                "branch_family": "nearby_primary",
                "area_scope": "nearby",
                "constraint_mode": "primary",
            },
            {
                "operator": "nearby_relaxed",
                "branch_family": "nearby_relaxed",
                "area_scope": "nearby",
                "constraint_mode": "relaxed",
            },
        ]

    # JP: family labelを処理する。
    # EN: Process family label.
    def _family_label(self, branch_family: str) -> str:
        return self._operator_label(branch_family)

    # JP: candidate node input payloadを処理する。
    # EN: Process candidate node input payload.
    def _candidate_node_input_payload(self, plan: SearchNodePlan) -> dict[str, Any]:
        return {
            "node_key": plan.node_key,
            "label": plan.label,
            "description": plan.description,
            "queries": plan.queries,
            "strategy_tags": plan.strategy_tags,
            "branch_family": plan.branch_family,
            "area_scope": plan.area_scope,
            "constraint_mode": plan.constraint_mode,
            "depth": plan.depth,
            "intent": plan.intent,
            "debug_depth": plan.debug_depth,
        }

    # JP: frontier scoreを推定する。
    # EN: Estimate frontier score.
    def _estimate_frontier_score(
        self,
        *,
        operator: str,
        branch_family: str,
        depth: int,
        parent_summary: dict[str, Any] | None,
    ) -> float:
        base = float((parent_summary or {}).get("branch_score") or 60.0)
        if parent_summary is None:
            base = 60.0

        preferred = set(self._strategy_memory().get("preferred_strategy_tags", []) or [])
        avoided = set(self._strategy_memory().get("avoided_strategy_tags", []) or [])
        retry_issues = set(self._retry_context().get("top_issues", []) or [])

        tag_bonus = {
            "source_diversify": 3.0,
            "detail_first": 4.0,
            "schema_first": 4.0,
            "exploit_best": 5.0,
        }.get(operator, 0.0)
        family_bonus = BRANCH_FAMILY_PRIORITY.get(branch_family, 0) * 2.5
        score = base + tag_bonus + family_bonus
        if operator in preferred:
            score += 4.0
        if operator in avoided:
            score -= 6.0
        if operator == "detail_first" and "詳細ページ補完率が低い" in retry_issues:
            score -= 4.0
        if operator == "schema_first" and "比較に必要な項目の欠損が多い" in retry_issues:
            score -= 4.0
        if depth >= 3:
            score -= (depth - 2) * 5.0
        return round(score, 2)

    # JP: make node planを処理する。
    # EN: Process make node plan.
    def _make_node_plan(
        self,
        state: ResearchExecutionState,
        *,
        operator: str,
        branch_family: str,
        area_scope: str,
        constraint_mode: str,
        base_queries: list[str],
        base_profile: dict[str, Any],
        parent_key: str | None,
        parent_node_id: int | None,
        depth: int,
        extra_tags: list[str] | None = None,
        parent_summary: dict[str, Any] | None = None,
    ) -> SearchNodePlan:
        strategy_tags = [branch_family, operator] + [
            str(tag).strip()
            for tag in extra_tags or []
            if str(tag).strip()
            and str(tag).strip() not in {operator, branch_family}
        ]
        queries = self._queries_for_operator(
            base_queries=base_queries,
            operator=operator,
            user_memory=self._active_user_memory(),
            area_scope=area_scope,
            constraint_mode=constraint_mode,
        )
        nearby_hints = (
            self._nearby_hints_from_queries(queries, user_memory=self._active_user_memory())
            if area_scope == "nearby"
            else []
        )
        if operator == branch_family:
            label = self._family_label(branch_family)
            description = self._operator_description(branch_family)
        else:
            label = f"{self._family_label(branch_family)} / {self._operator_label(operator)}"
            description = (
                f"{self._operator_description(branch_family)} / "
                f"{self._operator_description(operator)}"
            )
        intent = self._intent_for_child_plan(operator=operator, parent_summary=parent_summary)
        return SearchNodePlan(
            node_key=self._next_node_key(state, operator, depth),
            label=label,
            description=description,
            queries=queries,
            ranking_profile=self._profile_for_operator(
                base_profile=base_profile,
                operator=operator,
                area_scope=area_scope,
                constraint_mode=constraint_mode,
            ),
            strategy_tags=self._dedupe_queries(strategy_tags, limit=6),
            depth=depth,
            parent_key=parent_key,
            parent_node_id=parent_node_id,
            intent=intent,
            debug_depth=self._debug_depth_for_child_plan(parent_summary),
            branch_family=branch_family,
            area_scope=area_scope,
            constraint_mode=constraint_mode,
            nearby_hints=nearby_hints,
        )

    # JP: initial node plansを処理する。
    # EN: Process initial node plans.
    def _initial_node_plans(self, state: ResearchExecutionState) -> list[SearchNodePlan]:
        seed_queries = self.seed_queries_for_search(state)
        return [
            self._make_node_plan(
                state,
                operator=family["operator"],
                branch_family=family["branch_family"],
                area_scope=family["area_scope"],
                constraint_mode=family["constraint_mode"],
                base_queries=seed_queries,
                base_profile={},
                parent_key=None,
                parent_node_id=state.root_node.id if state.root_node else None,
                depth=1,
            )
            for family in self._initial_branch_families()
        ]

    # JP: seed queries for searchを処理する。
    # EN: Process seed queries for search.
    def seed_queries_for_search(self, state: ResearchExecutionState) -> list[str]:
        return state.seed_queries or ([state.query] if state.query else [])

    # JP: contextを再試行する。
    # EN: Retry context.
    def _retry_context(self) -> dict[str, Any]:
        return self.approved_plan.get("retry_context", {}) or {}

    # JP: executed tree node countを処理する。
    # EN: Process executed tree node count.
    def _executed_tree_node_count(self, state: ResearchExecutionState) -> int:
        # tree_max_nodes caps executed evaluations; completed plans should not
        # consume the remaining expansion budget just because they stay indexed.
        return len(state.branch_summaries)

    # JP: tree execution budget exhaustedを処理する。
    # EN: Process tree execution budget exhausted.
    def _tree_execution_budget_exhausted(self, state: ResearchExecutionState) -> bool:
        return self._executed_tree_node_count(state) >= self.tree_max_nodes

    # JP: pruned nodeを記録する。
    # EN: Record pruned node.
    def _record_pruned_node(
        self,
        state: ResearchExecutionState,
        *,
        plan: SearchNodePlan,
        parent_node_id: int | None,
        prune_reasons: list[str],
        evaluation: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "node_key": plan.node_key,
            "label": plan.label,
            "branch_family": plan.branch_family,
            "area_scope": plan.area_scope,
            "constraint_mode": plan.constraint_mode,
            "depth": plan.depth,
            "strategy_tags": plan.strategy_tags,
            "intent": plan.intent,
            "is_failed": False,
            "debug_depth": plan.debug_depth,
            "prune_reasons": prune_reasons,
        }
        if evaluation:
            payload["evaluation"] = {
                "branch_score": float(evaluation.get("branch_score") or 0.0),
                "detail_coverage": float(evaluation.get("detail_coverage") or 0.0),
                "top_issue_class": str(evaluation.get("top_issue_class") or ""),
            }
        state.pruned_nodes.append(payload)
        self._record_node(
            stage="tree_search",
            node_type="search_pruned",
            status="completed",
            input_payload={
                "node_key": plan.node_key,
                "queries": plan.queries,
                "strategy_tags": plan.strategy_tags,
            },
            output_payload=payload,
            reasoning="重複・深さ・低品質・失敗再発の条件により探索を剪定する。",
            parent_node_id=parent_node_id,
            branch_id=plan.node_key,
            intent=plan.intent,
            is_failed=False,
            debug_depth=plan.debug_depth,
            metrics={
                **(evaluation or {}),
                "intent": plan.intent,
                "is_failed": False,
                "debug_depth": plan.debug_depth,
            },
        )

    # JP: register frontier nodeを処理する。
    # EN: Process register frontier node.
    def _register_frontier_node(
        self,
        state: ResearchExecutionState,
        *,
        plan: SearchNodePlan,
        parent_summary: dict[str, Any] | None = None,
    ) -> None:
        if self._tree_execution_budget_exhausted(state):
            return
        if plan.depth > self.tree_max_depth:
            self._record_pruned_node(
                state,
                plan=plan,
                parent_node_id=plan.parent_node_id,
                prune_reasons=["depth_limit"],
            )
            return

        query_hash = self._hash_queries(plan.queries, plan.ranking_profile)
        if any(artifact.query_hash == query_hash for artifact in state.node_artifacts.values()):
            self._record_pruned_node(
                state,
                plan=plan,
                parent_node_id=plan.parent_node_id,
                prune_reasons=["duplicate_query_hash"],
            )
            return

        frontier_score = self._estimate_frontier_score(
            operator=plan.strategy_tags[1] if len(plan.strategy_tags) > 1 else plan.branch_family,
            branch_family=plan.branch_family,
            depth=plan.depth,
            parent_summary=parent_summary,
        )
        state.node_plans[plan.node_key] = plan
        state.node_artifacts[plan.node_key] = SearchNodeArtifacts(
            plan=plan,
            query_hash=query_hash,
            frontier_score=frontier_score,
        )
        queued_node = self._record_node(
            stage="tree_search",
            node_type="search_candidate",
            status="queued",
            input_payload=self._candidate_node_input_payload(plan),
            output_payload={
                "summary": f"{plan.label} を depth {plan.depth} の候補として待機しています。",
                "frontier_score": frontier_score,
            },
            reasoning="優先度順に評価するため、探索フロンティアへ候補ノードを登録する。",
            parent_node_id=plan.parent_node_id,
            branch_id=plan.node_key,
            intent=plan.intent,
            is_failed=False,
            debug_depth=plan.debug_depth,
            metrics={
                "branch_id": plan.node_key,
                "label": plan.label,
                "description": plan.description,
                "depth": plan.depth,
                "intent": plan.intent,
                "is_failed": False,
                "debug_depth": plan.debug_depth,
                "strategy_tags": plan.strategy_tags,
                "branch_family": plan.branch_family,
                "area_scope": plan.area_scope,
                "constraint_mode": plan.constraint_mode,
                "query_count": len(plan.queries),
                "queries": plan.queries,
                "frontier_score": frontier_score,
                "status": "queued",
            },
        )
        state.node_artifacts[plan.node_key].journal_node_id = queued_node.id
        state.frontier.append(plan.node_key)

    # JP: frontier nodesを選択する。
    # EN: Select frontier nodes.
    def _select_frontier_nodes(self, state: ResearchExecutionState) -> list[str]:
        batch_limit = min(
            self.tree_batch_size,
            max(0, self.tree_max_nodes - self._executed_tree_node_count(state)),
        )
        if batch_limit <= 0:
            return []
        queued = [
            state.node_artifacts[key]
            for key in state.frontier
            if key in state.node_artifacts and state.node_artifacts[key].status == "queued"
        ]
        queued.sort(
            key=lambda artifact: (
                float(artifact.frontier_score),
                -int(artifact.plan.depth),
                artifact.plan.node_key,
            ),
            reverse=True,
        )
        if batch_limit <= 1 or not queued:
            return [artifact.plan.node_key for artifact in queued[:batch_limit]]

        selected: list[str] = []
        selected_keys: set[str] = set()
        if state.best_node_key:
            deepen_candidate = next(
                (
                    artifact
                    for artifact in queued
                    if self._plan_is_descendant_of(
                        state,
                        node_key=artifact.plan.node_key,
                        ancestor_key=state.best_node_key,
                    )
                ),
                None,
            )
            if deepen_candidate is not None:
                selected.append(deepen_candidate.plan.node_key)
                selected_keys.add(deepen_candidate.plan.node_key)

            explore_candidate = next(
                (
                    artifact
                    for artifact in queued
                    if artifact.plan.node_key not in selected_keys
                    and not self._plan_is_descendant_of(
                        state,
                        node_key=artifact.plan.node_key,
                        ancestor_key=state.best_node_key,
                    )
                ),
                None,
            )
            if explore_candidate is not None and len(selected) < batch_limit:
                selected.append(explore_candidate.plan.node_key)
                selected_keys.add(explore_candidate.plan.node_key)

        for artifact in queued:
            if len(selected) >= batch_limit:
                break
            if artifact.plan.node_key in selected_keys:
                continue
            selected.append(artifact.plan.node_key)
            selected_keys.add(artifact.plan.node_key)
        return selected

    # JP: is descendant ofを計画する。
    # EN: Plan is descendant of.
    def _plan_is_descendant_of(
        self,
        state: ResearchExecutionState,
        *,
        node_key: str,
        ancestor_key: str,
    ) -> bool:
        current_key = node_key
        seen: set[str] = set()
        while current_key and current_key not in seen:
            seen.add(current_key)
            plan = state.node_plans.get(current_key)
            if plan is None or not plan.parent_key:
                return False
            if plan.parent_key == ancestor_key:
                return True
            current_key = plan.parent_key
        return False

    # JP: frontier scoreを計算する。
    # EN: Compute frontier score.
    def _compute_frontier_score(
        self,
        *,
        summary: dict[str, Any],
        parent_summary: dict[str, Any] | None,
        strategy_tags: list[str],
        depth: int,
    ) -> float:
        score = float(summary.get("branch_score") or 0.0)
        if parent_summary is not None:
            if float(summary.get("detail_coverage") or 0.0) > float(
                parent_summary.get("detail_coverage") or 0.0
            ):
                score += 10.0
            if float(summary.get("structured_ratio") or 0.0) > float(
                parent_summary.get("structured_ratio") or 0.0
            ):
                score += 8.0
            if str(summary.get("top_issue_class") or "") == str(
                parent_summary.get("top_issue_class") or ""
            ):
                score -= 8.0
        parent_tags = (
            set(parent_summary.get("strategy_tags", []) or []) if parent_summary else set()
        )
        if any(tag not in parent_tags for tag in strategy_tags):
            score += 5.0

        preferred = set(self._strategy_memory().get("preferred_strategy_tags", []) or [])
        avoided = set(self._strategy_memory().get("avoided_strategy_tags", []) or [])
        score += 4.0 * len(preferred.intersection(strategy_tags))
        score -= 6.0 * len(avoided.intersection(strategy_tags))
        if depth >= 3:
            score -= (depth - 2) * 5.0
        return round(score, 2)

    # JP: prune reasons for summaryを処理する。
    # EN: Process prune reasons for summary.
    def _prune_reasons_for_summary(
        self,
        state: ResearchExecutionState,
        *,
        summary: dict[str, Any],
        parent_summary: dict[str, Any] | None,
    ) -> list[str]:
        reasons: list[str] = []
        if float(summary.get("branch_score") or 0.0) < self.tree_prune_score:
            reasons.append("low_branch_score")
        if (
            int(summary.get("depth") or 0) >= 1
            and float(summary.get("detail_coverage") or 0.0) < 0.2
        ):
            reasons.append("low_detail_coverage")
        if int(summary.get("depth") or 0) > self.tree_max_depth:
            reasons.append("depth_limit")
        if (
            parent_summary is not None
            and str(summary.get("top_issue_class") or "")
            and str(summary.get("top_issue_class") or "")
            == str(parent_summary.get("top_issue_class") or "")
        ):
            artifact = state.node_artifacts.get(str(summary.get("branch_id") or ""))
            if artifact is not None and artifact.issue_streak >= 2:
                reasons.append(f"repeated_issue:{summary.get('top_issue_class')}")
        return reasons

    # JP: failure summaryを構築する。
    # EN: Build failure summary.
    def _build_failure_summary(
        self,
        *,
        plan: SearchNodePlan,
        artifacts: SearchNodeArtifacts,
        error_text: str,
        stage_name: str,
        parent_summary: dict[str, Any] | None,
    ) -> dict[str, Any]:
        failure_stage = self._failure_stage_hint(artifacts)
        search_summary = {}
        if artifacts.retrieve:
            search_summary |= artifacts.retrieve.get("summary", {})
        if artifacts.enrich:
            search_summary |= artifacts.enrich.get("summary", {})
        if artifacts.normalize:
            search_summary |= artifacts.normalize.get("summary", {})

        summary = evaluate_branch(
            branch_id=plan.node_key,
            label=plan.label,
            queries=plan.queries,
            raw_results=artifacts.retrieve.get("raw_results", []),
            normalized_properties=artifacts.normalize.get("normalized_properties", []),
            ranked_properties=artifacts.rank.get("ranked_properties", []),
            duplicate_groups=artifacts.normalize.get("duplicate_groups", []),
            search_summary=search_summary,
            parent_summary=parent_summary,
            strategy_tags=plan.strategy_tags,
            depth=plan.depth,
            query_hash=artifacts.query_hash,
            intent=plan.intent,
            is_failed=True,
            debug_depth=plan.debug_depth,
            branch_family=plan.branch_family,
            area_scope=plan.area_scope,
            constraint_mode=plan.constraint_mode,
        )
        summary["status"] = "failed"
        summary["frontier_score"] = 0.0
        summary["branch_score"] = 0.0
        summary["issues"] = list(summary.get("issues", [])) + [f"{stage_name}: {error_text}"]
        summary["recommendations"] = list(summary.get("recommendations", [])) + [
            "失敗した探索ノードを残しつつ別戦略へ切り替える"
        ]
        summary["summary"] = f"{plan.label}: failed at {stage_name} ({error_text})"
        summary["parent_key"] = plan.parent_key or ""
        summary["strategy_tags"] = plan.strategy_tags
        summary["query_hash"] = artifacts.query_hash
        summary["depth"] = plan.depth
        summary["prune_reasons"] = []
        summary["failure_stage"] = failure_stage
        return summary

    # JP: failure stage hintを処理する。
    # EN: Process failure stage hint.
    def _failure_stage_hint(self, artifacts: SearchNodeArtifacts) -> str:
        if not artifacts.retrieve:
            return "retrieve"
        if not artifacts.enrich:
            return "enrich"
        if not artifacts.normalize:
            return "normalize_dedupe"
        if not artifacts.integrity:
            return "integrity_review"
        if not artifacts.rank:
            return "rank"
        return "tree_search"

    # JP: candidateを実行する。
    # EN: Execute candidate.
    def _execute_candidate(
        self,
        state: ResearchExecutionState,
        *,
        plan: SearchNodePlan,
    ) -> dict[str, Any]:
        artifacts = state.node_artifacts[plan.node_key]
        parent_summary = (
            state.node_artifacts[plan.parent_key].summary
            if plan.parent_key and plan.parent_key in state.node_artifacts
            else None
        )
        artifacts.status = "running"
        self._update_recorded_node(
            artifacts.journal_node_id,
            status="running",
            input_payload=self._candidate_node_input_payload(plan),
            output_payload={
                "summary": f"{plan.label} を depth {plan.depth} で検証しています。",
                "frontier_score": artifacts.frontier_score,
            },
            reasoning="探索フロンティアから次ノードを取り出し、収集と評価を順番に実行する。",
            metrics={
                "branch_id": plan.node_key,
                "label": plan.label,
                "description": plan.description,
                "depth": plan.depth,
                "intent": plan.intent,
                "is_failed": False,
                "debug_depth": plan.debug_depth,
                "strategy_tags": plan.strategy_tags,
                "branch_family": plan.branch_family,
                "area_scope": plan.area_scope,
                "constraint_mode": plan.constraint_mode,
                "query_count": len(plan.queries),
                "queries": plan.queries,
                "frontier_score": artifacts.frontier_score,
                "status": "running",
            },
            intent=plan.intent,
            is_failed=False,
            debug_depth=plan.debug_depth,
        )
        started = time.perf_counter()
        try:
            retrieve_result = self.toolbox.run("retrieve", self.context, branch=plan)
            enrich_result = self.toolbox.run(
                "enrich",
                self.context,
                branch=plan,
                raw_results=retrieve_result.get("raw_results", []),
            )
            artifacts.enrich = enrich_result
            if "expanded_raw_results" in enrich_result:
                effective_raw_results = list(enrich_result.get("expanded_raw_results", []) or [])
            else:
                effective_raw_results = list(retrieve_result.get("raw_results", []) or [])
            retrieve_result = dict(retrieve_result)
            retrieve_result["seed_raw_results"] = list(retrieve_result.get("raw_results", []) or [])
            retrieve_result["raw_results"] = effective_raw_results
            retrieve_summary = dict(retrieve_result.get("summary", {}) or {})
            retrieve_summary["seed_url_count"] = int(
                retrieve_summary.get("unique_url_count") or len(retrieve_result["seed_raw_results"])
            )
            retrieve_summary["unique_url_count"] = len(effective_raw_results)
            retrieve_result["summary"] = retrieve_summary
            artifacts.retrieve = retrieve_result
            self._update_live_progress(
                stage_name="tree_search",
                progress_percent=56,
                current_action="検索結果を構造化中",
                detail=(
                    f"{plan.label} / {len(retrieve_result.get('raw_results', []))}件の候補から"
                    f" 家賃・間取り・駅距離を抽出しています。"
                )
                + (
                    " LLMで不足項目の補完も行っています。"
                    if self.research_adapter is not None
                    else ""
                ),
            )
            normalize_result = self.toolbox.run(
                "normalize_dedupe",
                self.context,
                query=state.query,
                raw_results=retrieve_result.get("raw_results", []),
                detail_html_map=enrich_result.get("detail_html_map", {}),
            )
            artifacts.normalize = normalize_result
            self._update_live_progress(
                stage_name="tree_search",
                progress_percent=64,
                current_action="掲載情報の整合性を確認中",
                detail=(
                    f"{plan.label} / {len(normalize_result.get('normalized_properties', []))}件を"
                    f" 対象に募集終了や条件矛盾を見ています。"
                )
                + (
                    " LLMでも説明文を再確認しています。"
                    if self.research_adapter is not None
                    else ""
                ),
            )
            integrity_result = self.toolbox.run(
                "integrity_review",
                self.context,
                branch=plan,
                normalized_properties=normalize_result.get("normalized_properties", []),
                raw_results=retrieve_result.get("raw_results", []),
                detail_html_map=enrich_result.get("detail_html_map", {}),
            )
            artifacts.integrity = integrity_result
            self._update_live_progress(
                stage_name="tree_search",
                progress_percent=72,
                current_action="候補をランキング中",
                detail=(
                    f"{plan.label} / 残った"
                    f" {len(integrity_result.get('normalized_properties', []))}件を条件一致度で並べています。"
                )
                + (" LLMで選定理由も整えています。" if self.research_adapter is not None else ""),
            )
            ranking_result = self.toolbox.run(
                "rank",
                self.context,
                branch=plan,
                normalized_properties=integrity_result.get("normalized_properties", []),
                ranking_profile=plan.ranking_profile,
            )
            artifacts.rank = ranking_result
            search_summary = (
                retrieve_result.get("summary", {})
                | enrich_result.get("summary", {})
                | normalize_result.get("summary", {})
                | integrity_result.get("summary", {})
            )
            summary = evaluate_branch(
                branch_id=plan.node_key,
                label=plan.label,
                queries=plan.queries,
                raw_results=retrieve_result.get("raw_results", []),
                normalized_properties=integrity_result.get("normalized_properties", []),
                ranked_properties=ranking_result.get("ranked_properties", []),
                duplicate_groups=normalize_result.get("duplicate_groups", []),
                search_summary=search_summary,
                parent_summary=parent_summary,
                strategy_tags=plan.strategy_tags,
                depth=plan.depth,
                query_hash=artifacts.query_hash,
                intent=plan.intent,
                is_failed=False,
                debug_depth=plan.debug_depth,
                branch_family=plan.branch_family,
                area_scope=plan.area_scope,
                constraint_mode=plan.constraint_mode,
            )
            summary["parent_key"] = plan.parent_key or ""
            summary["description"] = plan.description
            summary["frontier_score"] = self._compute_frontier_score(
                summary=summary,
                parent_summary=parent_summary,
                strategy_tags=plan.strategy_tags,
                depth=plan.depth,
            )
            parent_artifacts = state.node_artifacts.get(plan.parent_key or "")
            if parent_summary is not None and summary["top_issue_class"] == parent_summary.get(
                "top_issue_class"
            ):
                artifacts.issue_streak = (
                    parent_artifacts.issue_streak if parent_artifacts else 0
                ) + 1
            elif summary["top_issue_class"] != "healthy":
                artifacts.issue_streak = 1
            else:
                artifacts.issue_streak = 0
            summary["prune_reasons"] = self._prune_reasons_for_summary(
                state,
                summary=summary,
                parent_summary=parent_summary,
            )
            artifacts.summary = summary
            artifacts.frontier_score = float(
                summary.get("frontier_score") or artifacts.frontier_score
            )
            artifacts.status = "completed"
            self._cache_candidate_readiness(
                artifacts=artifacts,
                summary=summary,
                search_summary=search_summary,
            )
            duration_ms = int((time.perf_counter() - started) * 1000)
            self._update_recorded_node(
                artifacts.journal_node_id,
                status="completed",
                input_payload=self._candidate_node_input_payload(plan),
                output_payload={
                    "retrieve_summary": retrieve_result.get("summary", {}),
                    "enrich_summary": enrich_result.get("summary", {}),
                    "normalize_summary": normalize_result.get("summary", {}),
                    "integrity_summary": integrity_result.get("summary", {}),
                    "ranked_property_count": len(ranking_result.get("ranked_properties", [])),
                    "summary": summary.get("summary", ""),
                },
                reasoning="候補探索ノードを実行し、収集・補完・正規化・整合性レビュー・順位付けをまとめて評価する。",
                duration_ms=duration_ms,
                parent_node_id=plan.parent_node_id,
                branch_id=plan.node_key,
                metrics=summary,
                intent=plan.intent,
                is_failed=False,
                debug_depth=plan.debug_depth,
            )
            return summary
        except Exception as exc:
            duration_ms = int((time.perf_counter() - started) * 1000)
            failure_summary = self._build_failure_summary(
                plan=plan,
                artifacts=artifacts,
                error_text=str(exc),
                stage_name="tree_search",
                parent_summary=parent_summary,
            )
            artifacts.summary = failure_summary
            artifacts.status = "failed"
            self._cache_candidate_readiness(
                artifacts=artifacts,
                summary=failure_summary,
                search_summary={},
            )
            self._update_recorded_node(
                artifacts.journal_node_id,
                status="failed",
                input_payload=self._candidate_node_input_payload(plan),
                output_payload={
                    "error": str(exc),
                    "summary": failure_summary.get("summary", ""),
                },
                reasoning="探索ノード単位の失敗を全体失敗に直結させず、別ノード探索を継続する。",
                duration_ms=duration_ms,
                parent_node_id=plan.parent_node_id,
                branch_id=plan.node_key,
                metrics=failure_summary,
                intent=plan.intent,
                is_failed=True,
                debug_depth=plan.debug_depth,
            )
            return failure_summary

    # JP: candidate batch asyncを実行する。
    # EN: Execute candidate batch async.
    async def _execute_candidate_batch_async(
        self,
        state: ResearchExecutionState,
        *,
        plans: list[SearchNodePlan],
    ) -> list[dict[str, Any]]:
        loop = asyncio.get_running_loop()
        worker_count = max(1, min(5, self.tree_batch_size, len(plans)))
        with ThreadPoolExecutor(
            max_workers=worker_count, thread_name_prefix="tree-branch"
        ) as executor:
            tasks = [
                loop.run_in_executor(
                    executor,
                    partial(self._execute_candidate, state, plan=plan),
                )
                for plan in plans
            ]
            return await asyncio.gather(*tasks)

    # JP: upsert branch summaryを処理する。
    # EN: Process upsert branch summary.
    def _upsert_branch_summary(
        self,
        state: ResearchExecutionState,
        *,
        plan: SearchNodePlan,
        summary: dict[str, Any],
    ) -> None:
        branch_id = plan.node_key
        existing_index = next(
            (
                index
                for index, item in enumerate(state.branch_summaries)
                if str(item.get("branch_id") or "").strip() == branch_id
            ),
            None,
        )
        if existing_index is None:
            state.branch_summaries.append(summary)
        else:
            state.branch_summaries[existing_index] = summary

        if summary.get("status") == "failed":
            state.branch_failures[branch_id] = summary
        else:
            state.branch_failures.pop(branch_id, None)

        if summary.get("prune_reasons") and not any(
            str(item.get("node_key") or "").strip() == branch_id for item in state.pruned_nodes
        ):
            artifacts = state.node_artifacts.get(branch_id)
            self._record_pruned_node(
                state,
                plan=plan,
                parent_node_id=artifacts.journal_node_id if artifacts else plan.parent_node_id,
                prune_reasons=list(summary.get("prune_reasons", []) or []),
                evaluation=summary,
            )

    # JP: expand branch batchを処理する。
    # EN: Process expand branch batch.
    def _expand_branch_batch(
        self,
        state: ResearchExecutionState,
        *,
        plans: list[SearchNodePlan],
    ) -> list[dict[str, Any]]:
        if not plans:
            return []

        if len(plans) == 1 or self.tree_batch_size <= 1:
            summaries = [self._execute_candidate(state, plan=plans[0])]
        else:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                summaries = asyncio.run(self._execute_candidate_batch_async(state, plans=plans))
            else:
                summaries = [self._execute_candidate(state, plan=plan) for plan in plans]

        for plan, summary in zip(plans, summaries, strict=False):
            self._upsert_branch_summary(state, plan=plan, summary=summary)
        return summaries

    # JP: expand candidates from summaryを処理する。
    # EN: Process expand candidates from summary.
    def _expand_candidates_from_summary(
        self,
        state: ResearchExecutionState,
        *,
        plan: SearchNodePlan,
        summary: dict[str, Any],
        operators: list[str] | None = None,
    ) -> list[SearchNodePlan]:
        resolved_operators = operators or [
            str(item).strip()
            for item in summary.get("expand_recommendations", []) or []
            if str(item).strip()
        ]
        if not resolved_operators:
            return []

        parent_artifacts = state.node_artifacts.get(plan.node_key)
        base_queries = plan.queries
        base_profile = plan.ranking_profile
        if parent_artifacts and parent_artifacts.rank.get("ranking_profile"):
            base_profile = dict(parent_artifacts.rank.get("ranking_profile", {}))
        children: list[SearchNodePlan] = []
        for operator in resolved_operators[: self._children_budget_for(summary)]:
            children.append(
                self._make_node_plan(
                    state,
                    operator=operator,
                    branch_family=plan.branch_family,
                    area_scope=plan.area_scope,
                    constraint_mode=plan.constraint_mode,
                    base_queries=base_queries,
                    base_profile=base_profile,
                    parent_key=plan.node_key,
                    parent_node_id=parent_artifacts.journal_node_id
                    if parent_artifacts
                    else plan.parent_node_id,
                    depth=plan.depth + 1,
                    extra_tags=plan.strategy_tags,
                    parent_summary=summary,
                )
            )
        return children

    # JP: children budget forを処理する。
    # EN: Process children budget for.
    def _children_budget_for(self, summary: dict[str, Any]) -> int:
        score = float(summary.get("branch_score") or 0.0)
        readiness = str(summary.get("readiness") or "").strip().lower()
        if readiness == "high" and score >= 75:
            return 1
        if score < 50:
            return 3
        return self.tree_children_per_expansion

    # JP: recovery operators for summaryを処理する。
    # EN: Process recovery operators for summary.
    def _recovery_operators_for_summary(
        self,
        *,
        plan: SearchNodePlan,
        summary: dict[str, Any],
    ) -> list[str]:
        prune_reasons = {
            str(item).strip() for item in summary.get("prune_reasons", []) if str(item).strip()
        }
        if "depth_limit" in prune_reasons or any(
            reason.startswith("repeated_issue:") for reason in prune_reasons
        ):
            return []

        joined_issues = " / ".join(str(item) for item in summary.get("issues", []) or [])
        failure_stage = str(summary.get("failure_stage") or "").strip()
        operators: list[str] = []

        if summary.get("status") == "failed":
            if failure_stage == "retrieve" or "検索結果が取得できていない" in joined_issues:
                operators.extend(["source_diversify", "detail_first"])
            if (
                failure_stage in {"enrich", "normalize_dedupe", "integrity_review"}
                or "詳細ページ補完率が低い" in joined_issues
            ):
                operators.extend(["detail_first", "schema_first"])
            if failure_stage == "rank" or "上位候補の条件一致度が低い" in joined_issues:
                operators.extend(["exploit_best", "source_diversify"])

        if "low_detail_coverage" in prune_reasons or "詳細ページ補完率が低い" in joined_issues:
            operators.extend(["detail_first", "schema_first"])
        if "low_branch_score" in prune_reasons or "上位候補の条件一致度が低い" in joined_issues:
            operators.extend(["exploit_best", "source_diversify"])
        if "情報源の多様性が低い" in joined_issues:
            operators.extend(["source_diversify", "detail_first"])
        if not operators:
            operators.extend(
                [
                    str(item).strip()
                    for item in summary.get("expand_recommendations", []) or []
                    if str(item).strip()
                ]
            )
        if not operators and summary.get("status") == "failed":
            operators.extend(["source_diversify", "detail_first"])

        deduped: list[str] = []
        preferred = [
            operator for operator in operators if operator and operator not in plan.strategy_tags
        ] + [operator for operator in operators if operator]
        for operator in preferred:
            if operator not in deduped:
                deduped.append(operator)
        return deduped[: self._children_budget_for(summary)]

    # JP: next candidates after summaryを処理する。
    # EN: Process next candidates after summary.
    def _next_candidates_after_summary(
        self,
        state: ResearchExecutionState,
        *,
        plan: SearchNodePlan,
        summary: dict[str, Any],
    ) -> list[SearchNodePlan]:
        if summary.get("status") == "completed" and not summary.get("prune_reasons"):
            return self._expand_candidates_from_summary(
                state,
                plan=plan,
                summary=summary,
            )
        recovery_operators = self._recovery_operators_for_summary(plan=plan, summary=summary)
        if not recovery_operators:
            return []
        return self._expand_candidates_from_summary(
            state,
            plan=plan,
            summary=summary,
            operators=recovery_operators,
        )

    # JP: default selected branch summaryを処理する。
    # EN: Process default selected branch summary.
    def _default_selected_branch_summary(self) -> dict[str, Any]:
        return {
            "branch_id": "none",
            "node_key": "none",
            "label": "none",
            "branch_family": "strict_primary",
            "area_scope": "strict",
            "constraint_mode": "primary",
            "status": "failed",
            "intent": "draft",
            "is_failed": True,
            "debug_depth": 0,
            "depth": 0,
            "query_count": 0,
            "queries": [],
            "raw_result_count": 0,
            "normalized_count": 0,
            "detail_hit_count": 0,
            "detail_coverage": 0.0,
            "duplicate_group_count": 0,
            "duplicate_ratio": 0.0,
            "structured_ratio": 0.0,
            "top_score": 0.0,
            "avg_top3_score": 0.0,
            "source_diversity": 0,
            "branch_score": 0.0,
            "frontier_score": 0.0,
            "issues": ["tree search の評価対象がありませんでした。"],
            "recommendations": ["検索条件とソース設定を見直して再試行する"],
            "summary": "tree search の結果がありません。",
            "strategy_tags": [],
            "prune_reasons": [],
        }

    # JP: selected artifactsを処理する。
    # EN: Process selected artifacts.
    def _selected_artifacts(self, state: ResearchExecutionState) -> SearchNodeArtifacts | None:
        return state.node_artifacts.get(str(state.selected_branch_summary.get("branch_id") or ""))

    # JP: selected pathを構築する。
    # EN: Build selected path.
    def _build_selected_path(self, state: ResearchExecutionState) -> list[dict[str, Any]]:
        branch_id = str(state.selected_branch_summary.get("branch_id") or "")
        path: list[dict[str, Any]] = []
        current_key = branch_id
        while current_key:
            artifacts = state.node_artifacts.get(current_key)
            if artifacts is None:
                break
            summary = artifacts.summary or {}
            path.append(
                {
                    "branch_id": current_key,
                    "label": artifacts.plan.label,
                    "branch_family": artifacts.plan.branch_family,
                    "area_scope": artifacts.plan.area_scope,
                    "constraint_mode": artifacts.plan.constraint_mode,
                    "depth": artifacts.plan.depth,
                    "intent": str(summary.get("intent") or artifacts.plan.intent),
                    "is_failed": bool(summary.get("is_failed")),
                    "debug_depth": int(summary.get("debug_depth") or artifacts.plan.debug_depth),
                    "strategy_tags": artifacts.plan.strategy_tags,
                    "branch_score": float(summary.get("branch_score") or 0.0),
                    "frontier_score": float(
                        summary.get("frontier_score") or artifacts.frontier_score
                    ),
                    "summary": str(summary.get("summary") or ""),
                }
            )
            current_key = artifacts.plan.parent_key or ""
        path.reverse()
        return path

    # JP: branch path artifactsを処理する。
    # EN: Process branch path artifacts.
    def _branch_path_artifacts(
        self,
        state: ResearchExecutionState,
        *,
        node_key: str,
    ) -> list[SearchNodeArtifacts]:
        path: list[SearchNodeArtifacts] = []
        current_key = node_key
        seen: set[str] = set()
        while current_key and current_key not in seen:
            seen.add(current_key)
            artifacts = state.node_artifacts.get(current_key)
            if artifacts is None:
                break
            path.append(artifacts)
            current_key = artifacts.plan.parent_key or ""
        path.reverse()
        return path

    # JP: branch result nodesを処理する。
    # EN: Process branch result nodes.
    def _branch_result_nodes(
        self,
        state: ResearchExecutionState,
        *,
        node_key: str,
    ) -> list[dict[str, Any]]:
        branch_nodes: list[dict[str, Any]] = []
        for artifacts in self._branch_path_artifacts(state, node_key=node_key):
            summary = artifacts.summary or {}
            if summary.get("status") != "completed":
                continue
            search_summary = {}
            if artifacts.retrieve:
                search_summary |= artifacts.retrieve.get("summary", {})
            if artifacts.enrich:
                search_summary |= artifacts.enrich.get("summary", {})
            if artifacts.normalize:
                search_summary |= artifacts.normalize.get("summary", {})
            if artifacts.integrity:
                search_summary |= artifacts.integrity.get("summary", {})
            if "normalized_properties" in artifacts.integrity:
                node_properties = artifacts.integrity.get("normalized_properties", [])
            else:
                node_properties = artifacts.normalize.get("normalized_properties", [])

            branch_nodes.append(
                {
                    "branch_id": artifacts.plan.node_key,
                    "node_key": artifacts.plan.node_key,
                    "label": artifacts.plan.label,
                    "branch_family": artifacts.plan.branch_family,
                    "area_scope": artifacts.plan.area_scope,
                    "constraint_mode": artifacts.plan.constraint_mode,
                    "depth": artifacts.plan.depth,
                    "queries": artifacts.plan.queries,
                    "strategy_tags": artifacts.plan.strategy_tags,
                    "issues": list(summary.get("issues", []) or []),
                    "search_summary": search_summary,
                    "raw_results": artifacts.retrieve.get("raw_results", []),
                    "detail_html_map": artifacts.enrich.get("detail_html_map", {}),
                    "normalized_properties": node_properties,
                    "dropped_properties": artifacts.integrity.get("dropped_properties", []),
                    "integrity_reviews": artifacts.integrity.get("integrity_reviews", []),
                    "duplicate_groups": artifacts.normalize.get("duplicate_groups", []),
                    "ranked_properties": artifacts.rank.get("ranked_properties", []),
                }
            )
        return branch_nodes

    # JP: attach branch result summariesを処理する。
    # EN: Process attach branch result summaries.
    def _attach_branch_result_summaries(self, state: ResearchExecutionState) -> None:
        cache: dict[tuple[str, ...], dict[str, Any]] = {}
        for artifacts in state.node_artifacts.values():
            if artifacts.summary.get("status") != "completed":
                continue
            path_artifacts = self._branch_path_artifacts(state, node_key=artifacts.plan.node_key)
            path_keys = tuple(
                item.plan.node_key
                for item in path_artifacts
                if item.summary.get("status") == "completed"
            )
            if not path_keys:
                continue
            if path_keys not in cache:
                cache[path_keys] = run_result_summarizer(
                    branch_nodes=self._branch_result_nodes(state, node_key=artifacts.plan.node_key),
                    adapter=self.research_adapter,
                )
            branch_result_summary = self._cache_copy(cache[path_keys])
            artifacts.normalize["branch_result_summary"] = self._cache_copy(branch_result_summary)
            if artifacts.integrity:
                artifacts.integrity["branch_result_summary"] = self._cache_copy(
                    branch_result_summary
                )
            artifacts.summary["branch_result_summary"] = self._cache_copy(branch_result_summary)

    # JP: search tree summaryを構築する。
    # EN: Build search tree summary.
    def _build_search_tree_summary(self, state: ResearchExecutionState) -> dict[str, Any]:
        issue_counter: Counter[str] = Counter()
        max_depth = 0
        for summary in state.branch_summaries:
            issue_class = str(summary.get("top_issue_class") or "").strip()
            if issue_class and issue_class != "healthy":
                issue_counter[issue_class] += 1
            max_depth = max(max_depth, int(summary.get("depth") or 0))
        selected_path_tags: list[str] = []
        for item in state.selected_path:
            for tag in item.get("strategy_tags", []) or []:
                text = str(tag).strip()
                if text and text not in selected_path_tags:
                    selected_path_tags.append(text)
        return {
            "executed_node_count": len(
                [item for item in state.branch_summaries if item.get("status") == "completed"]
            ),
            "failed_node_count": len(
                [item for item in state.branch_summaries if item.get("status") == "failed"]
            ),
            "pruned_node_count": len(state.pruned_nodes),
            "frontier_remaining": len(state.frontier),
            "termination_reason": state.termination_reason or "frontier_exhausted",
            "max_depth_reached": max_depth,
            "selected_branch_id": str(state.selected_branch_summary.get("branch_id") or ""),
            "alternative_branch_ids": list(state.alternative_branch_ids),
            "selected_path_tags": selected_path_tags,
            "retry_context_used": bool(state.retry_context),
            "issue_distribution": dict(issue_counter.most_common(5)),
        }

    # JP: cache candidate readinessを処理する。
    # EN: Process cache candidate readiness.
    def _cache_candidate_readiness(
        self,
        *,
        artifacts: SearchNodeArtifacts,
        summary: dict[str, Any],
        search_summary: dict[str, Any],
    ) -> None:
        artifacts.branch_score = float(summary.get("branch_score") or 0.0)
        if summary.get("status") != "completed":
            artifacts.readiness = "low"
            summary["readiness"] = artifacts.readiness
            return
        result = evaluate_final_result(
            selected_branch_summary=summary,
            visible_ranked_properties=artifacts.rank.get("ranked_properties", []),
            search_summary=search_summary,
        )
        artifacts.readiness = str(result.get("readiness") or "low")
        summary["readiness"] = artifacts.readiness

    # JP: eligible completed artifactsを処理する。
    # EN: Process eligible completed artifacts.
    def _eligible_completed_artifacts(
        self,
        state: ResearchExecutionState,
    ) -> list[SearchNodeArtifacts]:
        completed = [
            artifact
            for artifact in state.node_artifacts.values()
            if artifact.summary.get("status") == "completed"
        ]
        lookup = {artifact.plan.node_key: artifact for artifact in completed}
        eligible = [
            artifact
            for artifact in completed
            if is_branch_selection_eligible(
                artifact.summary,
                parent_summary=(
                    lookup[artifact.plan.parent_key].summary
                    if artifact.plan.parent_key and artifact.plan.parent_key in lookup
                    else None
                ),
            )
        ]
        return eligible or completed

    # JP: familyごとのbest summaryを処理する。
    # EN: Process best summary per family.
    def _best_summaries_by_family(self, state: ResearchExecutionState) -> dict[str, dict[str, Any]]:
        completed = [
            dict(item)
            for item in state.branch_summaries
            if str(item.get("status") or "").strip() == "completed"
        ]
        best_by_family: dict[str, dict[str, Any]] = {}
        for family in BRANCH_FAMILY_PRIORITY:
            family_summaries = [
                item for item in completed if str(item.get("branch_family") or "").strip() == family
            ]
            if not family_summaries:
                continue
            selected = select_best_branch(family_summaries)
            if selected is not None:
                best_by_family[family] = selected
        return best_by_family

    # JP: best score gapを処理する。
    # EN: Process best score gap.
    def _best_score_gap(self, state: ResearchExecutionState) -> float:
        candidates = [
            dict(item)
            for item in state.branch_summaries
            if str(item.get("status") or "").strip() == "completed"
        ]
        if len(candidates) < 2:
            return 0.0
        selected = select_best_branch(candidates)
        if selected is None:
            return 0.0
        family = str(selected.get("branch_family") or "").strip()
        second_candidates = [
            item
            for item in candidates
            if str(item.get("branch_id") or "") != str(selected.get("branch_id") or "")
            and str(item.get("branch_family") or "").strip() == family
        ]
        if not second_candidates:
            second_candidates = [
                item
                for item in candidates
                if str(item.get("branch_id") or "") != str(selected.get("branch_id") or "")
            ]
        if not second_candidates:
            return 0.0
        second = max(second_candidates, key=branch_selection_sort_key)
        best_score = float(selected.get("branch_score") or 0.0)
        second_score = float(second.get("branch_score") or 0.0)
        return round(best_score - second_score, 2)

    # JP: stop for stable bestかどうかを判定する。
    # EN: Check whether stop for stable best.
    def _can_stop_for_stable_best(self, state: ResearchExecutionState) -> bool:
        if self._best_node_readiness(state) != "high":
            return False
        if state.best_node_stability < self.tree_stability_patience:
            return False
        if self._executed_tree_node_count(state) < self.tree_min_nodes_before_stable_stop:
            return False
        return state.best_score_gap >= self.tree_min_best_score_gap

    # JP: refresh best nodeを処理する。
    # EN: Process refresh best node.
    def _refresh_best_node(self, state: ResearchExecutionState, *, candidate_key: str) -> None:
        previous_best_key = state.best_node_key
        completed_summaries = [
            dict(item)
            for item in state.branch_summaries
            if str(item.get("status") or "").strip() == "completed"
        ]
        selected_summary = select_best_branch(completed_summaries)
        if selected_summary is None:
            state.selected_branch_summary = {}
            state.alternative_branch_ids = []
            state.best_node_key = ""
            state.best_node_stability = 0
            state.best_node_readiness = "low"
            state.best_score_gap = 0.0
            return

        best_key = str(selected_summary.get("branch_id") or "")
        best_artifacts = state.node_artifacts.get(best_key)
        state.selected_branch_summary = selected_summary
        state.best_node_readiness = best_artifacts.readiness if best_artifacts is not None else "low"
        best_by_family = self._best_summaries_by_family(state)
        state.alternative_branch_ids = [
            str(item.get("branch_id") or "")
            for family, item in sorted(
                best_by_family.items(),
                key=lambda pair: BRANCH_FAMILY_PRIORITY.get(pair[0], 0),
                reverse=True,
            )
            if str(item.get("branch_id") or "") and str(item.get("branch_id") or "") != best_key
        ]
        if best_key == previous_best_key:
            state.best_node_stability += 1
        else:
            state.best_node_key = best_key
            state.best_node_stability = 1
        state.best_score_gap = self._best_score_gap(state)

    # JP: best node readinessを処理する。
    # EN: Process best node readiness.
    def _best_node_readiness(self, state: ResearchExecutionState) -> str:
        if not state.best_node_key:
            return "low"
        artifacts = state.node_artifacts.get(state.best_node_key)
        if artifacts is None:
            return "low"
        return artifacts.readiness

    # JP: plan finalizeを処理する。
    # EN: Handle plan finalize.
    def _handle_plan_finalize(self, state: ResearchExecutionState) -> str | None:
        _, state.plan_result = self._run_stage(
            stage_name="plan_finalize",
            progress_percent=10,
            latest_summary=(
                "現在: 承認済み計画を確認しています。\n"
                "内容: seed クエリと探索条件を固定しています。"
            ),
            input_payload={"approved_plan": self.approved_plan},
            reasoning="ユーザー承認済みの計画を固定し、以降の調査に使う条件を確定する。",
            runner=lambda: self.toolbox.run("plan_finalize", self.context),
        )
        state.seed_queries = [
            " ".join(str(item).split()).strip()
            for item in state.plan_result.get("seed_queries", []) or []
            if " ".join(str(item).split()).strip()
        ]
        state.query = str(state.plan_result.get("search_query") or "")
        if not state.query and state.seed_queries:
            state.query = state.seed_queries[0]
        state.retry_context = self._retry_context()
        return None

    # JP: tree searchを処理する。
    # EN: Handle tree search.
    def _handle_tree_search(self, state: ResearchExecutionState) -> str | None:
        # JP: runnerを処理する。
        # EN: Process runner.
        def runner() -> dict[str, Any]:
            retry_context = state.retry_context
            state.root_node = self._record_node(
                stage="tree_search",
                node_type="search_root",
                status="completed",
                input_payload={
                    "search_query": state.query,
                    "seed_queries": state.seed_queries,
                    "retry_context": retry_context,
                },
                output_payload={
                    "summary": "承認済み計画から tree search を開始",
                    "strategy_memory": self._strategy_memory(),
                },
                reasoning="承認済み計画、履歴戦略、再試行文脈を束ねた探索根を作る。",
                intent="draft",
                is_failed=False,
                debug_depth=0,
            )

            for plan in self._initial_node_plans(state):
                self._register_frontier_node(state, plan=plan)

            while state.frontier and not self._tree_execution_budget_exhausted(state):
                selected_keys = self._select_frontier_nodes(state)
                if not selected_keys:
                    break

                selected_plans: list[SearchNodePlan] = []
                for node_key in selected_keys:
                    if node_key not in state.frontier:
                        continue
                    state.frontier.remove(node_key)
                    selected_plans.append(state.node_plans[node_key])

                if not selected_plans:
                    continue

                if len(selected_plans) == 1:
                    latest_summary = (
                        f"現在: 探索ノードを検証中\n"
                        f"内容: {selected_plans[0].label} を depth {selected_plans[0].depth} で評価しています。"
                    )
                else:
                    labels = " / ".join(plan.label for plan in selected_plans[:3])
                    latest_summary = (
                        f"現在: 探索ノードを並列検証中\n"
                        f"内容: {len(selected_plans)}件を同時評価しています。{labels}"
                    )
                self._update_job(
                    stage_name="tree_search",
                    progress_percent=24 + min(58, len(state.branch_summaries) * 6),
                    latest_summary=latest_summary,
                )

                should_stop = False
                summaries = self._expand_branch_batch(state, plans=selected_plans)
                for plan, summary in zip(selected_plans, summaries, strict=False):
                    node_key = plan.node_key
                    self._refresh_best_node(state, candidate_key=node_key)

                    if should_stop:
                        continue

                    if self._can_stop_for_stable_best(state):
                        state.termination_reason = "stable_high_readiness"
                        should_stop = True
                    elif self._tree_execution_budget_exhausted(state):
                        state.termination_reason = "node_budget_exhausted"
                        should_stop = True
                    else:
                        for child in self._next_candidates_after_summary(
                            state,
                            plan=plan,
                            summary=summary,
                        ):
                            self._register_frontier_node(
                                state,
                                plan=child,
                                parent_summary=summary,
                            )

                if should_stop:
                    break

            if not state.termination_reason:
                state.termination_reason = (
                    "frontier_exhausted" if not state.frontier else "node_budget_exhausted"
                )

            self._update_live_progress(
                stage_name="tree_search",
                progress_percent=82,
                current_action="探索分岐を要約中",
                detail=("各分岐の候補・リスクを圧縮し、最終候補の比較材料を作っています。"),
            )
            self._attach_branch_result_summaries(state)
            state.selected_branch_summary = (
                state.selected_branch_summary
                or select_best_branch(state.branch_summaries)
                or self._default_selected_branch_summary()
            )
            best_by_family = self._best_summaries_by_family(state)
            state.alternative_branch_ids = [
                str(item.get("branch_id") or "")
                for family, item in sorted(
                    best_by_family.items(),
                    key=lambda pair: BRANCH_FAMILY_PRIORITY.get(pair[0], 0),
                    reverse=True,
                )
                if str(item.get("branch_id") or "")
                and str(item.get("branch_id") or "")
                != str(state.selected_branch_summary.get("branch_id") or "")
            ]
            state.selected_path = self._build_selected_path(state)
            self._update_live_progress(
                stage_name="tree_search",
                progress_percent=88,
                current_action="最良の探索経路を選定中",
                detail=(
                    f"{len(state.branch_summaries)} 分岐から条件一致度と根拠量を見て選んでいます。"
                ),
            )
            selected_artifacts = self._selected_artifacts(state)
            self._record_node(
                stage="tree_search",
                node_type="search_selection",
                status="completed",
                input_payload={"candidate_count": len(state.branch_summaries)},
                output_payload={
                    "selected_branch": state.selected_branch_summary,
                    "selected_path": state.selected_path,
                    "alternative_branch_ids": state.alternative_branch_ids,
                },
                reasoning="tree search の評価結果から最良ノードとその経路を採用する。",
                parent_node_id=(
                    selected_artifacts.journal_node_id
                    if selected_artifacts and selected_artifacts.journal_node_id
                    else state.root_node.id
                    if state.root_node
                    else None
                ),
                branch_id=str(state.selected_branch_summary.get("branch_id") or ""),
                selected=True,
                metrics=state.selected_branch_summary,
                intent=str(state.selected_branch_summary.get("intent") or "draft"),
                is_failed=bool(state.selected_branch_summary.get("is_failed")),
                debug_depth=int(state.selected_branch_summary.get("debug_depth") or 0),
            )
            state.search_tree_summary = self._build_search_tree_summary(state)
            selected_label = str(state.selected_branch_summary.get("label") or "none")
            return {
                "summary": (
                    f"探索ノード{len(state.branch_summaries)}件を評価し、{selected_label} を採用"
                ),
                "selected_branch": state.selected_branch_summary,
                "selected_path": state.selected_path,
                "pruned_node_count": len(state.pruned_nodes),
                "search_tree_summary": state.search_tree_summary,
            }

        self._run_stage(
            stage_name="tree_search",
            progress_percent=22,
            latest_summary="動的 tree search を初期化しています。",
            input_payload={
                "search_query": state.query,
                "seed_queries": state.seed_queries,
                "retry_context": state.retry_context,
            },
            reasoning="固定分岐ではなく、評価に応じて次ノードを選ぶ動的探索を行う。",
            runner=runner,
        )
        return None
