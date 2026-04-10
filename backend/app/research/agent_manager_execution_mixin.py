from __future__ import annotations

from app.research.offline_eval import evaluate_final_result, summarize_branch_failures

from .agent_manager_types import ResearchExecutionResult, ResearchExecutionState


class AgentManagerExecutionMixin:
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
        selected_properties = selected_integrity.get(
            "normalized_properties", []
        ) or selected_normalize.get("normalized_properties", [])

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
        if selected_branch_result_summary:
            state.search_summary["branch_result_summary"] = selected_branch_result_summary
        state.failure_summary = summarize_branch_failures(state.branch_summaries)
        state.offline_evaluation = evaluate_final_result(
            selected_branch_summary=state.selected_branch_summary,
            visible_ranked_properties=selected_rank.get("ranked_properties", []),
            search_summary=state.search_summary,
        )
        state.offline_evaluation["selected_path"] = state.selected_path
        state.offline_evaluation["search_tree_summary"] = state.search_tree_summary
        state.research_summary = self._build_fallback_research_summary(
            ranked_properties=selected_rank.get("ranked_properties", []),
            normalized_properties=selected_properties,
            search_summary=state.search_summary,
            source_items=state.source_items,
            offline_evaluation=state.offline_evaluation,
            branch_result_summary=selected_branch_result_summary,
        )
        if self.research_adapter is not None:
            try:
                llm_summary = self._build_llm_research_summary(
                    ranked_properties=selected_rank.get("ranked_properties", []),
                    normalized_properties=selected_properties,
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
            latest_summary="結果をユーザー向けに整理しています。",
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
        selected_properties = selected_integrity.get(
            "normalized_properties", []
        ) or selected_normalize.get("normalized_properties", [])

        return ResearchExecutionResult(
            query=state.query,
            selected_branch_id=str(state.selected_branch_summary.get("branch_id") or "none"),
            branch_summaries=state.branch_summaries,
            branch_result_summary=selected_branch_result_summary,
            final_report_markdown="",
            normalized_properties=selected_properties,
            ranked_properties=selected_rank.get("ranked_properties", []),
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
