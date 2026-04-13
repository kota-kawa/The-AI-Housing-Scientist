from __future__ import annotations

import hashlib
from typing import Any

from app.research.offline_eval import (
    BRANCH_FAMILY_PRIORITY,
    evaluate_final_result,
    summarize_branch_failures,
)
from app.stages.search_normalize import is_single_property_search_result
from app.stages.result_summarizer import PROPERTY_CANDIDATES_KEY

from .agent_manager_types import (
    ResearchExecutionResult,
    ResearchExecutionState,
    SearchNodeArtifacts,
)

DISPLAY_CANDIDATE_LIMIT = 6
MIN_DISPLAY_CANDIDATE_COUNT = 3


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

    # JP: branch pathの完了artifactsを取得する。
    # EN: Get completed artifacts on one branch path.
    def _completed_artifacts_for_branch(
        self,
        state: ResearchExecutionState,
        *,
        branch_id: str,
    ) -> list[SearchNodeArtifacts]:
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
        branch_id: str,
        branch_result_summary: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        entries_by_key: dict[str, dict[str, Any]] = {}
        for artifacts in self._completed_artifacts_for_branch(state, branch_id=branch_id):
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

        summary_candidates = list(branch_result_summary.get(PROPERTY_CANDIDATES_KEY, []) or [])
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

    # JP: branch summaryを取得する。
    # EN: Get branch summary by id.
    def _branch_summary_by_id(
        self,
        state: ResearchExecutionState,
        *,
        branch_id: str,
    ) -> dict[str, Any]:
        for item in state.branch_summaries:
            if str(item.get("branch_id") or "") == branch_id:
                return dict(item)
        return {}

    # JP: family failure summaryを構築する。
    # EN: Build failure summary per family.
    def _family_failure_summary(self, state: ResearchExecutionState) -> dict[str, Any]:
        summaries: dict[str, Any] = {}
        for family in [
            "strict_primary",
            "strict_relaxed",
            "nearby_primary",
            "nearby_relaxed",
        ]:
            family_summaries = [
                item
                for item in state.branch_summaries
                if str(item.get("branch_family") or "") == family
            ]
            if not family_summaries:
                continue
            summary = summarize_branch_failures(family_summaries)
            if summary.get("top_issues"):
                summaries[family] = summary
        return summaries

    # JP: alternative display groupsを構築する。
    # EN: Build display groups for alternative families.
    def _build_alternative_display_groups(self, state: ResearchExecutionState) -> list[dict[str, Any]]:
        groups: list[dict[str, Any]] = []
        for branch_id in state.alternative_branch_ids[:3]:
            artifacts = state.node_artifacts.get(branch_id)
            if artifacts is None:
                continue
            branch_summary = self._branch_summary_by_id(state, branch_id=branch_id)
            branch_result_summary = (
                artifacts.integrity.get("branch_result_summary", {})
                or artifacts.normalize.get("branch_result_summary", {})
                or branch_summary.get("branch_result_summary", {})
            )
            ranked_properties, normalized_properties = self._build_display_candidate_pool(
                state=state,
                branch_id=branch_id,
                branch_result_summary=branch_result_summary,
            )
            if not ranked_properties:
                ranked_properties = list(artifacts.rank.get("ranked_properties", []))[:DISPLAY_CANDIDATE_LIMIT]
                visible_ids = {
                    str(item.get("property_id_norm") or "").strip()
                    for item in ranked_properties
                    if str(item.get("property_id_norm") or "").strip()
                }
                normalized_source = (
                    artifacts.integrity.get("normalized_properties", [])
                    if "normalized_properties" in artifacts.integrity
                    else artifacts.normalize.get("normalized_properties", [])
                )
                normalized_properties = [
                    item
                    for item in normalized_source
                    if str(item.get("property_id_norm") or "").strip() in visible_ids
                ]
            if not ranked_properties:
                continue
            groups.append(
                {
                    "branch_id": branch_id,
                    "label": str(branch_summary.get("label") or artifacts.plan.label or "別枠候補"),
                    "branch_family": str(
                        branch_summary.get("branch_family") or artifacts.plan.branch_family or ""
                    ),
                    "ranked_properties": ranked_properties,
                    "normalized_properties": normalized_properties,
                    }
                )
        return groups

    # JP: raw resultから表示用候補を構築する。
    # EN: Build display candidates from raw results.
    def _build_raw_result_display_candidates(
        self,
        state: ResearchExecutionState,
        *,
        excluded_property_ids: set[str],
        excluded_urls: set[str],
        limit: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        from app.area_matching import classify_area_match

        ranked_properties: list[dict[str, Any]] = []
        normalized_properties: list[dict[str, Any]] = []
        seen_urls = {str(url).strip() for url in excluded_urls if str(url).strip()}
        seen_titles: set[str] = set()
        # JP: フォールバック候補もエリアチェックする。
        # EN: Area-check fallback candidates to prevent unrelated properties from appearing.
        target_area = str(self._active_user_memory().get("target_area") or "").strip()
        completed_summaries = sorted(
            [
                item
                for item in state.branch_summaries
                if str(item.get("status") or "").strip() == "completed"
            ],
            key=lambda item: (
                BRANCH_FAMILY_PRIORITY.get(str(item.get("branch_family") or "").strip(), 0),
                float(item.get("branch_score") or 0.0),
                -int(item.get("depth") or 0),
            ),
            reverse=True,
        )
        for summary in completed_summaries:
            branch_id = str(summary.get("branch_id") or "").strip()
            artifacts = state.node_artifacts.get(branch_id)
            if artifacts is None:
                continue
            label = str(summary.get("label") or artifacts.plan.label or "検索候補")
            raw_results = list(artifacts.retrieve.get("raw_results", []) or [])
            detail_html_map = dict(artifacts.enrich.get("detail_html_map", {}) or {})
            for raw in raw_results:
                url = str(raw.get("url") or "").strip()
                title = str(raw.get("title") or "").strip()
                if url and url in seen_urls:
                    continue
                if not url and title and title in seen_titles:
                    continue
                if not url and not title:
                    continue
                detail_html = str(detail_html_map.get(url) or "")
                if not is_single_property_search_result(raw, detail_html):
                    continue
                # JP: エリア不一致の参考候補を除外する。
                # EN: Skip fallback candidates that clearly don't match the target area.
                if target_area:
                    raw_address = str(raw.get("address") or "").strip()
                    raw_area_name = str(raw.get("area_name") or "").strip()
                    raw_description = str(raw.get("description") or "").strip()
                    area_check_text = raw_address or raw_area_name or raw_description or title
                    area_match = classify_area_match(
                        target_area=target_area,
                        address=area_check_text,
                        area_name=raw_area_name,
                        nearby_tokens=artifacts.plan.nearby_hints or [],
                    )
                    if area_match["match_level"] == "none":
                        continue
                property_id = (
                    str(raw.get("property_id_norm") or "").strip()
                    or f"fallback-{hashlib.sha1((url or title).encode('utf-8')).hexdigest()[:12]}"
                )
                if property_id in excluded_property_ids:
                    continue
                description = str(raw.get("description") or "").strip()
                extra_snippets = [
                    str(item).strip()
                    for item in raw.get("extra_snippets", []) or []
                    if str(item).strip()
                ]
                notes_parts = [description] + extra_snippets[:2]
                notes = " / ".join(part for part in notes_parts if part).strip()
                source_name = str(raw.get("source_name") or "source").strip() or "source"
                why_selected = (
                    f"{label} で検索ヒットした参考候補です。"
                    " 条件を柔軟に広げた再表示枠として残しています。"
                )
                why_not_selected = (
                    "家賃・駅徒歩・間取りの一部が未取得のため、"
                    "詳細ページで再確認が必要です。"
                )
                normalized_properties.append(
                    {
                        "property_id_norm": property_id,
                        "building_name": title or "参考候補",
                        "detail_url": url,
                        "image_url": str(raw.get("image_url") or "").strip(),
                        "address": str(raw.get("address") or "").strip(),
                        "area_name": str(raw.get("area_name") or "").strip(),
                        "rent": int(raw.get("rent") or 0),
                        "layout": str(raw.get("layout") or "").strip(),
                        "station_walk_min": int(raw.get("station_walk_min") or 0),
                        "area_m2": float(raw.get("area_m2") or 0.0),
                        "nearest_station": str(raw.get("nearest_station") or "").strip(),
                        "features": [],
                        "notes": notes or f"{source_name} の検索結果から取得した参考候補です。",
                    }
                )
                ranked_properties.append(
                    {
                        "property_id_norm": property_id,
                        "score": round(max(10.0, 38.0 - len(ranked_properties) * 2.5), 2),
                        "why_selected": why_selected,
                        "why_not_selected": why_not_selected,
                        "reason": why_selected,
                        "building_name": title or "参考候補",
                        "detail_url": url,
                        "source_nodes": [label],
                        "reference_only": True,
                    }
                )
                excluded_property_ids.add(property_id)
                if url:
                    seen_urls.add(url)
                if title:
                    seen_titles.add(title)
                if len(ranked_properties) >= limit:
                    return ranked_properties, normalized_properties
        return ranked_properties, normalized_properties

    # JP: 表示候補を minimum 件数まで補完する。
    # EN: Ensure the display candidate list reaches the minimum count.
    def _ensure_minimum_display_candidates(
        self,
        state: ResearchExecutionState,
        *,
        display_ranked_properties: list[dict[str, Any]],
        display_normalized_properties: list[dict[str, Any]],
        alternative_display_groups: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
        ranked = list(display_ranked_properties)
        normalized_by_id = {
            str(item.get("property_id_norm") or "").strip(): dict(item)
            for item in display_normalized_properties
            if str(item.get("property_id_norm") or "").strip()
        }
        normalized = list(normalized_by_id.values())
        existing_property_ids = {
            str(item.get("property_id_norm") or "").strip()
            for item in ranked
            if str(item.get("property_id_norm") or "").strip()
        }
        existing_urls = {
            str(item.get("detail_url") or "").strip()
            for item in normalized
            if str(item.get("detail_url") or "").strip()
        }
        if len(ranked) >= MIN_DISPLAY_CANDIDATE_COUNT:
            return ranked, normalized, ""

        # JP: まず別 family のランク済み候補で補完する。エリア不一致は除外する。
        # EN: Fill from alternative family candidates first. Skip area-mismatched properties.
        from app.area_matching import classify_area_match

        target_area = str(self._active_user_memory().get("target_area") or "").strip()
        for group in alternative_display_groups:
            group_by_id = {
                str(item.get("property_id_norm") or "").strip(): dict(item)
                for item in group.get("normalized_properties", []) or []
                if str(item.get("property_id_norm") or "").strip()
            }
            for item in group.get("ranked_properties", []) or []:
                property_id = str(item.get("property_id_norm") or "").strip()
                if not property_id or property_id in existing_property_ids:
                    continue
                prop = group_by_id.get(property_id)
                if prop is None:
                    continue
                # JP: エリア不一致の候補を別ブランチから補完しない。
                # EN: Don't fill in properties that don't match the target area.
                if target_area:
                    area_match = classify_area_match(
                        target_area=target_area,
                        address=str(prop.get("address") or ""),
                        area_name=str(prop.get("area_name") or ""),
                    )
                    if area_match["match_level"] == "none":
                        continue
                ranked.append(dict(item))
                normalized_by_id[property_id] = prop
                normalized.append(prop)
                existing_property_ids.add(property_id)
                detail_url = str(prop.get("detail_url") or "").strip()
                if detail_url:
                    existing_urls.add(detail_url)
                if len(ranked) >= MIN_DISPLAY_CANDIDATE_COUNT:
                    return ranked, normalized, "alternative_branch_fill"

        # JP: それでも不足する場合は raw result を参考候補として表示する。
        # EN: Fall back to raw search results when ranked candidates are still insufficient.
        missing_count = MIN_DISPLAY_CANDIDATE_COUNT - len(ranked)
        raw_ranked, raw_normalized = self._build_raw_result_display_candidates(
            state,
            excluded_property_ids=existing_property_ids,
            excluded_urls=existing_urls,
            limit=missing_count,
        )
        for prop in raw_normalized:
            property_id = str(prop.get("property_id_norm") or "").strip()
            if property_id and property_id not in normalized_by_id:
                normalized_by_id[property_id] = prop
                normalized.append(prop)
        ranked.extend(raw_ranked)
        if raw_ranked:
            return ranked, normalized, "raw_result_fill"
        return ranked, normalized, ""

    # JP: synthesizeを処理する。
    # EN: Handle synthesize.
    def _handle_synthesize(self, state: ResearchExecutionState) -> str | None:
        selected_artifacts = self._selected_artifacts(state)
        selected_rank = selected_artifacts.rank if selected_artifacts else {}
        selected_normalize = selected_artifacts.normalize if selected_artifacts else {}
        selected_integrity = selected_artifacts.integrity if selected_artifacts else {}
        selected_retrieve = selected_artifacts.retrieve if selected_artifacts else {}
        selected_enrich = selected_artifacts.enrich if selected_artifacts else {}
        selected_result_summary = (
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
            branch_id=str(state.selected_branch_summary.get("branch_id") or ""),
            branch_result_summary=selected_result_summary,
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
        state.alternative_display_groups = self._build_alternative_display_groups(state)
        (
            display_ranked_properties,
            display_normalized_properties,
            minimum_fill_source,
        ) = self._ensure_minimum_display_candidates(
            state,
            display_ranked_properties=display_ranked_properties,
            display_normalized_properties=display_normalized_properties,
            alternative_display_groups=state.alternative_display_groups,
        )
        if minimum_fill_source:
            display_candidate_source = (
                f"{display_candidate_source}+{minimum_fill_source}"
                if display_candidate_source
                else minimum_fill_source
            )
        state.display_ranked_properties = display_ranked_properties
        state.display_normalized_properties = display_normalized_properties

        source_ranked_properties = (
            list(selected_rank.get("ranked_properties", [])) or display_ranked_properties
        )
        source_normalized_properties = selected_properties or display_normalized_properties
        source_raw_results = list(selected_retrieve.get("raw_results", []))
        if not source_raw_results:
            seen_urls: set[str] = set()
            for artifacts in state.node_artifacts.values():
                for item in artifacts.retrieve.get("raw_results", []) or []:
                    url = str(item.get("url") or "").strip()
                    if url and url in seen_urls:
                        continue
                    if url:
                        seen_urls.add(url)
                    source_raw_results.append(item)
        state.source_items = self.collect_source_items(
            ranked_properties=source_ranked_properties,
            normalized_properties=source_normalized_properties,
            raw_results=source_raw_results,
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
        state.search_summary["selected_branch_family"] = str(
            state.selected_branch_summary.get("branch_family") or ""
        )
        state.search_summary["alternative_branch_ids"] = list(state.alternative_branch_ids)
        if selected_result_summary:
            state.search_summary["branch_result_summary"] = selected_result_summary
        state.failure_summary = summarize_branch_failures(state.branch_summaries)
        state.family_failure_summary = self._family_failure_summary(state)
        state.offline_evaluation = evaluate_final_result(
            selected_branch_summary=state.selected_branch_summary,
            visible_ranked_properties=display_ranked_properties,
            search_summary=state.search_summary,
        )
        state.offline_evaluation["selected_path"] = state.selected_path
        state.offline_evaluation["search_tree_summary"] = state.search_tree_summary
        state.offline_evaluation["alternative_branch_ids"] = state.alternative_branch_ids
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
            branch_result_summary=selected_result_summary,
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
                    branch_result_summary=selected_result_summary,
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
        selected_result_summary = (
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
        final_normalized_properties = selected_properties or display_normalized_properties
        final_ranked_properties = (
            list(selected_rank.get("ranked_properties", [])) or display_ranked_properties
        )

        return ResearchExecutionResult(
            query=state.query,
            selected_branch_id=str(state.selected_branch_summary.get("branch_id") or "none"),
            alternative_branch_ids=list(state.alternative_branch_ids),
            branch_summaries=state.branch_summaries,
            branch_result_summary=selected_result_summary,
            final_report_markdown="",
            normalized_properties=final_normalized_properties,
            ranked_properties=final_ranked_properties,
            display_normalized_properties=display_normalized_properties,
            display_ranked_properties=display_ranked_properties,
            alternative_display_groups=state.alternative_display_groups,
            duplicate_groups=selected_normalize.get("duplicate_groups", []),
            integrity_reviews=selected_integrity.get("integrity_reviews", []),
            dropped_property_ids=selected_integrity.get("dropped_property_ids", []),
            raw_results=selected_retrieve.get("raw_results", []),
            source_items=state.source_items,
            search_summary=state.search_summary,
            offline_evaluation=state.offline_evaluation,
            failure_summary=state.failure_summary,
            family_failure_summary=state.family_failure_summary,
            research_summary=state.research_summary,
            selected_path=state.selected_path,
            search_tree_summary=state.search_tree_summary,
            pruned_nodes=state.pruned_nodes,
        )
