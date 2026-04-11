from __future__ import annotations

from typing import Any

from app.research.offline_eval import evaluate_final_result, summarize_branch_failures
from app.stages.result_summarizer import PROPERTY_CANDIDATES_KEY

from .agent_manager_types import (
    ResearchExecutionResult,
    ResearchExecutionState,
    SearchNodeArtifacts,
)

DISPLAY_CANDIDATE_LIMIT = 6


class AgentManagerExecutionMixin:
    # JP: display candidate keyを処理する。
    # EN: Process display candidate key.
    @staticmethod
    def _display_candidate_key(candidate: dict[str, Any]) -> str:
        return (
            str(candidate.get("property_id_norm") or "").strip()
            or str(candidate.get("detail_url") or "").strip()
        )

    # JP: display value有無を判定する。
    # EN: Check whether display value exists.
    @staticmethod
    def _has_display_value(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, (int, float)):
            return float(value) > 0
        if isinstance(value, (list, tuple, set, dict)):
            return bool(value)
        return True

    # JP: display candidate completenessを処理する。
    # EN: Process display candidate completeness.
    def _display_candidate_completeness(self, candidate: dict[str, Any]) -> int:
        fields = [
            "image_url",
            "detail_url",
            "address",
            "rent",
            "layout",
            "station_walk_min",
            "area_m2",
            "features",
            "notes",
        ]
        return sum(1 for field in fields if self._has_display_value(candidate.get(field)))

    # JP: display snapshotを統合する。
    # EN: Merge display snapshot.
    def _merge_display_snapshot(
        self,
        preferred: dict[str, Any],
        secondary: dict[str, Any],
    ) -> dict[str, Any]:
        merged = dict(preferred)
        for key, value in secondary.items():
            if key == "features":
                existing = [
                    str(item).strip()
                    for item in merged.get("features", []) or []
                    if str(item).strip()
                ]
                for item in value or []:
                    text = str(item).strip()
                    if text and text not in existing:
                        existing.append(text)
                if existing:
                    merged["features"] = existing
                continue
            if not self._has_display_value(merged.get(key)) and self._has_display_value(value):
                merged[key] = value
        return merged

    # JP: display summary fieldsを補完する。
    # EN: Fill display summary fields.
    def _apply_display_summary_fields(
        self,
        normalized_property: dict[str, Any],
        summary_candidate: dict[str, Any],
    ) -> dict[str, Any]:
        if not summary_candidate:
            return dict(normalized_property)
        filled = dict(normalized_property)
        for field in (
            "property_id_norm",
            "building_name",
            "image_url",
            "address",
            "rent",
            "layout",
            "station_walk_min",
            "area_m2",
            "detail_url",
        ):
            value = summary_candidate.get(field)
            if not self._has_display_value(filled.get(field)) and self._has_display_value(value):
                filled[field] = value
        return filled

    # JP: selected pathの完了artifactsを取得する。
    # EN: Get completed artifacts on the selected path.
    def _selected_path_completed_artifacts(
        self,
        state: ResearchExecutionState,
    ) -> list[SearchNodeArtifacts]:
        branch_id = str(state.selected_branch_summary.get("branch_id") or "")
        if not branch_id:
            return []
        return [
            artifacts
            for artifacts in self._branch_path_artifacts(state, node_key=branch_id)
            if artifacts.summary.get("status") == "completed"
        ]

    # JP: display candidate poolを構築する。
    # EN: Build display candidate pool.
    def _build_display_candidate_pool(
        self,
        *,
        state: ResearchExecutionState,
        selected_branch_result_summary: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        entries_by_key: dict[str, dict[str, Any]] = {}
        for artifacts in self._selected_path_completed_artifacts(state):
            if "normalized_properties" in artifacts.integrity:
                normalized_properties = list(artifacts.integrity.get("normalized_properties", []) or [])
            else:
                normalized_properties = list(artifacts.normalize.get("normalized_properties", []) or [])
            ranked_by_id = {
                str(item.get("property_id_norm") or "").strip(): item
                for item in artifacts.rank.get("ranked_properties", []) or []
                if str(item.get("property_id_norm") or "").strip()
            }
            for prop in normalized_properties:
                key = self._display_candidate_key(prop)
                if not key:
                    continue
                ranked = ranked_by_id.get(str(prop.get("property_id_norm") or "").strip(), {})
                incoming_score = float(ranked.get("score") or 0.0)
                incoming_snapshot = dict(prop)
                label = str(artifacts.plan.label or artifacts.plan.node_key or "node")
                existing = entries_by_key.get(key)
                if existing is None:
                    entries_by_key[key] = {
                        "normalized": incoming_snapshot,
                        "best_score": incoming_score,
                        "why_selected": str(ranked.get("why_selected") or "").strip(),
                        "why_not_selected": str(ranked.get("why_not_selected") or "").strip(),
                        "source_nodes": [label] if label else [],
                    }
                    continue

                existing_snapshot = dict(existing.get("normalized") or {})
                existing_score = float(existing.get("best_score") or 0.0)
                preferred_snapshot = existing_snapshot
                secondary_snapshot = incoming_snapshot
                if (
                    self._display_candidate_completeness(incoming_snapshot),
                    incoming_score,
                ) > (
                    self._display_candidate_completeness(existing_snapshot),
                    existing_score,
                ):
                    preferred_snapshot, secondary_snapshot = incoming_snapshot, existing_snapshot
                existing["normalized"] = self._merge_display_snapshot(
                    preferred_snapshot,
                    secondary_snapshot,
                )
                if incoming_score > existing_score:
                    existing["best_score"] = incoming_score
                    existing["why_selected"] = str(ranked.get("why_selected") or "").strip()
                    existing["why_not_selected"] = str(ranked.get("why_not_selected") or "").strip()
                else:
                    if not str(existing.get("why_selected") or "").strip():
                        existing["why_selected"] = str(ranked.get("why_selected") or "").strip()
                    if not str(existing.get("why_not_selected") or "").strip():
                        existing["why_not_selected"] = str(
                            ranked.get("why_not_selected") or ""
                        ).strip()
                if label and label not in existing["source_nodes"]:
                    existing["source_nodes"].append(label)

        summary_candidates = list(
            selected_branch_result_summary.get(PROPERTY_CANDIDATES_KEY, []) or []
        )
        summary_by_key = {
            self._display_candidate_key(item): item
            for item in summary_candidates
            if self._display_candidate_key(item)
        }
        ordered_keys: list[str] = []
        for item in summary_candidates:
            key = self._display_candidate_key(item)
            if key and key in entries_by_key and key not in ordered_keys:
                ordered_keys.append(key)

        remaining_keys = sorted(
            [key for key in entries_by_key if key not in ordered_keys],
            key=lambda key: (
                float(entries_by_key[key].get("best_score") or 0.0),
                self._display_candidate_completeness(
                    entries_by_key[key].get("normalized") or {}
                ),
                str(
                    (entries_by_key[key].get("normalized") or {}).get("building_name") or ""
                ),
            ),
            reverse=True,
        )

        display_ranked_properties: list[dict[str, Any]] = []
        display_normalized_properties: list[dict[str, Any]] = []
        used_property_ids: set[str] = set()
        for key in ordered_keys + remaining_keys:
            entry = entries_by_key[key]
            summary_candidate = summary_by_key.get(key, {})
            normalized_property = self._apply_display_summary_fields(
                entry.get("normalized") or {},
                summary_candidate,
            )
            property_id = str(normalized_property.get("property_id_norm") or "").strip()
            if not property_id or property_id in used_property_ids:
                continue
            used_property_ids.add(property_id)
            display_normalized_properties.append(normalized_property)
            reason = str(summary_candidate.get("reason") or "").strip()
            why_selected = str(entry.get("why_selected") or "").strip() or reason
            display_ranked_properties.append(
                {
                    "property_id_norm": property_id,
                    "score": round(
                        float(entry.get("best_score") or summary_candidate.get("score") or 0.0),
                        2,
                    ),
                    "why_selected": why_selected,
                    "why_not_selected": str(entry.get("why_not_selected") or "").strip(),
                    "reason": reason or why_selected,
                    "building_name": str(
                        normalized_property.get("building_name")
                        or summary_candidate.get("building_name")
                        or "候補物件"
                    ),
                    "detail_url": str(
                        normalized_property.get("detail_url")
                        or summary_candidate.get("detail_url")
                        or ""
                    ),
                    "source_nodes": list(entry.get("source_nodes") or []),
                }
            )
            if len(display_ranked_properties) >= DISPLAY_CANDIDATE_LIMIT:
                break

        return display_ranked_properties, display_normalized_properties

    # JP: synthesizeを処理する。
    # EN: Handle synthesize.
    def _handle_synthesize(self, state: ResearchExecutionState) -> str | None:
        selected_artifacts = self._selected_artifacts(state)
        selected_rank = selected_artifacts.rank if selected_artifacts else {}
        selected_normalize = selected_artifacts.normalize if selected_artifacts else {}
        selected_integrity = selected_artifacts.integrity if selected_artifacts else {}
        selected_retrieve = selected_artifacts.retrieve if selected_artifacts else {}
        selected_enrich = selected_artifacts.enrich if selected_artifacts else {}
        selected_branch_result_summary = (
            selected_integrity.get("branch_result_summary", {})
            or selected_normalize.get("branch_result_summary", {})
            or state.selected_branch_summary.get("branch_result_summary", {})
        )
        selected_properties = (
            selected_integrity["normalized_properties"]
            if "normalized_properties" in selected_integrity
            else selected_normalize.get("normalized_properties", [])
        )
        display_ranked_properties, display_normalized_properties = self._build_display_candidate_pool(
            state=state,
            selected_branch_result_summary=selected_branch_result_summary,
        )
        display_candidate_source = "selected_path_aggregate"
        if not display_ranked_properties:
            display_ranked_properties = list(selected_rank.get("ranked_properties", []))[
                :DISPLAY_CANDIDATE_LIMIT
            ]
            visible_ids = {
                str(item.get("property_id_norm") or "").strip()
                for item in display_ranked_properties
                if str(item.get("property_id_norm") or "").strip()
            }
            display_normalized_properties = [
                item
                for item in selected_properties
                if str(item.get("property_id_norm") or "").strip() in visible_ids
            ]
            display_candidate_source = "selected_leaf_fallback"
        state.display_ranked_properties = display_ranked_properties
        state.display_normalized_properties = display_normalized_properties

        state.source_items = self.collect_source_items(
            ranked_properties=selected_rank.get("ranked_properties", []),
            normalized_properties=selected_properties,
            raw_results=selected_retrieve.get("raw_results", []),
        )
        state.search_summary = (
            selected_normalize.get("summary", {})
            | selected_integrity.get("summary", {})
            | selected_retrieve.get("summary", {})
            | selected_enrich.get("summary", {})
        )
        if "normalized_properties" in selected_integrity:
            state.search_summary["normalized_count"] = len(selected_properties)
        state.search_summary["rankable_candidate_count"] = len(
            selected_rank.get("ranked_properties", [])
        )
        state.search_summary["display_candidate_count"] = len(display_ranked_properties)
        state.search_summary["display_candidate_source"] = display_candidate_source
        if selected_branch_result_summary:
            state.search_summary["branch_result_summary"] = selected_branch_result_summary
        state.failure_summary = summarize_branch_failures(state.branch_summaries)
        state.offline_evaluation = evaluate_final_result(
            selected_branch_summary=state.selected_branch_summary,
            visible_ranked_properties=display_ranked_properties,
            search_summary=state.search_summary,
        )
        state.offline_evaluation["selected_path"] = state.selected_path
        state.offline_evaluation["search_tree_summary"] = state.search_tree_summary
        self._update_live_progress(
            stage_name="synthesize",
            progress_percent=92,
            current_action="最終サマリーを組み立て中",
            detail=(
                f"表示用 {len(display_ranked_properties)}件の候補と"
                f" {len(state.source_items)}件の参照ソースを整理しています。"
            ),
        )
        state.research_summary = self._build_fallback_research_summary(
            ranked_properties=display_ranked_properties,
            normalized_properties=display_normalized_properties,
            search_summary=state.search_summary,
            source_items=state.source_items,
            offline_evaluation=state.offline_evaluation,
            branch_result_summary=selected_branch_result_summary,
        )
        if self.research_adapter is not None:
            try:
                self._update_live_progress(
                    stage_name="synthesize",
                    progress_percent=93,
                    current_action="LLMで最終要約を生成中",
                    detail="探索結果をユーザー向けの日本語サマリーに整えています。",
                )
                llm_summary = self._build_llm_research_summary(
                    ranked_properties=display_ranked_properties,
                    normalized_properties=display_normalized_properties,
                    search_summary=state.search_summary,
                    source_items=state.source_items,
                    offline_evaluation=state.offline_evaluation,
                    branch_result_summary=selected_branch_result_summary,
                    selected_branch_summary=state.selected_branch_summary,
                    branch_summaries=state.branch_summaries,
                    failure_summary=state.failure_summary,
                    selected_path=state.selected_path,
                    search_tree_summary=state.search_tree_summary,
                )
                if llm_summary:
                    state.research_summary = llm_summary
            except Exception:
                pass

        self.db.add_audit_event(
            self.session_id,
            "search_normalize",
            {
                "query": state.query,
                "selected_branch_id": state.selected_branch_summary["branch_id"],
                "branch_summaries": state.branch_summaries,
                "selected_path": state.selected_path,
            },
            selected_normalize,
            "動的 tree search の最良ノードから正規化結果を採用",
        )
        self.db.add_audit_event(
            self.session_id,
            "integrity_review",
            {
                "selected_branch_id": state.selected_branch_summary["branch_id"],
                "selected_path": state.selected_path,
            },
            selected_integrity,
            "古い掲載や矛盾のある候補をランキング前に除外",
        )
        self.db.add_audit_event(
            self.session_id,
            "ranking",
            {
                "selected_branch_id": state.selected_branch_summary["branch_id"],
                "branch_summaries": state.branch_summaries,
                "selected_path": state.selected_path,
            },
            selected_rank,
            "最良探索ノードの順位付け結果を採用",
        )
        self.db.add_audit_event(
            self.session_id,
            "offline_evaluator",
            {
                "selected_branch_id": state.selected_branch_summary["branch_id"],
                "search_tree_summary": state.search_tree_summary,
            },
            state.offline_evaluation,
            "探索品質をオフライン指標で評価し、次の改善候補を提示",
        )

        self._run_stage(
            stage_name="synthesize",
            progress_percent=94,
            latest_summary=(
                "現在: 結果をユーザー向けに整理しています。\n"
                "内容: 推薦理由・注意点・次の確認事項を最終レスポンスへまとめています。"
            ),
            input_payload={"selected_branch_id": state.selected_branch_summary["branch_id"]},
            reasoning="最良ノード、探索経路、失敗要因をまとめて次アクションへ接続する。",
            runner=lambda: {
                "selected_branch_id": state.selected_branch_summary["branch_id"],
                "source_item_count": len(state.source_items),
                "research_summary": state.research_summary,
                "offline_evaluation": state.offline_evaluation,
                "failure_summary": state.failure_summary,
                "selected_path": state.selected_path,
                "search_tree_summary": state.search_tree_summary,
            },
        )
        return None

    # JP: 必要な処理を実行する。
    # EN: Execute the required data.
    def execute(self) -> ResearchExecutionResult:
        state = ResearchExecutionState()
        self.state_machine.run(state, start_stage="plan_finalize")

        selected_artifacts = self._selected_artifacts(state)
        selected_rank = selected_artifacts.rank if selected_artifacts else {}
        selected_normalize = selected_artifacts.normalize if selected_artifacts else {}
        selected_integrity = selected_artifacts.integrity if selected_artifacts else {}
        selected_retrieve = selected_artifacts.retrieve if selected_artifacts else {}
        selected_branch_result_summary = (
            selected_integrity.get("branch_result_summary", {})
            or selected_normalize.get("branch_result_summary", {})
            or state.selected_branch_summary.get("branch_result_summary", {})
        )
        selected_properties = (
            selected_integrity["normalized_properties"]
            if "normalized_properties" in selected_integrity
            else selected_normalize.get("normalized_properties", [])
        )
        display_ranked_properties = list(state.display_ranked_properties or [])
        display_normalized_properties = list(state.display_normalized_properties or [])
        if not display_ranked_properties:
            display_ranked_properties = list(selected_rank.get("ranked_properties", []))[
                :DISPLAY_CANDIDATE_LIMIT
            ]
            display_ids = {
                str(item.get("property_id_norm") or "").strip()
                for item in display_ranked_properties
                if str(item.get("property_id_norm") or "").strip()
            }
            display_normalized_properties = [
                item
                for item in selected_properties
                if str(item.get("property_id_norm") or "").strip() in display_ids
            ]

        return ResearchExecutionResult(
            query=state.query,
            selected_branch_id=str(state.selected_branch_summary.get("branch_id") or "none"),
            branch_summaries=state.branch_summaries,
            branch_result_summary=selected_branch_result_summary,
            final_report_markdown="",
            normalized_properties=selected_properties,
            ranked_properties=selected_rank.get("ranked_properties", []),
            display_normalized_properties=display_normalized_properties,
            display_ranked_properties=display_ranked_properties,
            duplicate_groups=selected_normalize.get("duplicate_groups", []),
            integrity_reviews=selected_integrity.get("integrity_reviews", []),
            dropped_property_ids=selected_integrity.get("dropped_property_ids", []),
            raw_results=selected_retrieve.get("raw_results", []),
            source_items=state.source_items,
            search_summary=state.search_summary,
            offline_evaluation=state.offline_evaluation,
            failure_summary=state.failure_summary,
            research_summary=state.research_summary,
            selected_path=state.selected_path,
            search_tree_summary=state.search_tree_summary,
            pruned_nodes=state.pruned_nodes,
        )
