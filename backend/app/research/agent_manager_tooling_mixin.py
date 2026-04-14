from __future__ import annotations

from collections.abc import Callable
import html
import json
import re
import threading
import time
from typing import Any
from urllib.parse import urljoin, urlparse

from app.research.journal import ResearchIntent, ResearchNode
from app.research.state_machine import ResearchStageDefinition, ResearchStateMachine
from app.research.tools import CallableResearchTool, Toolbox, ToolContext, ToolSpec
from app.stages.integrity_review import run_integrity_review
from app.stages.ranking import run_ranking
from app.stages.search_normalize import is_single_property_search_result, run_search_and_normalize

from .agent_manager_types import SearchNodePlan

MAX_DETAIL_LINK_CANDIDATES_PER_PAGE = 12
MAX_FOLLOW_DETAIL_LINKS_PER_PAGE = 4
MAX_EXPANDED_RAW_RESULTS = 36
LINK_CONTEXT_WINDOW = 180
DETAIL_LINK_HINT_TOKENS = (
    "詳細",
    "物件詳細",
    "空室",
    "募集",
    "この物件",
    "賃貸マンション",
    "賃貸アパート",
    "間取り",
)
COLLECTION_LINK_HINT_TOKENS = (
    "一覧",
    "検索結果",
    "エリア",
    "沿線",
    "地図",
    "特集",
    "ランキング",
    "条件変更",
)


class AgentManagerToolingMixin:
    @staticmethod
    def _compact_progress_text(value: Any, *, max_chars: int = 120) -> str:
        text = " ".join(str(value or "").split()).strip()
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 1].rstrip() + "…"

    def _display_progress_url(self, url: str, *, max_chars: int = 140) -> str:
        text = self._compact_progress_text(url, max_chars=max_chars)
        if text:
            return text
        return ""

    @staticmethod
    def _clean_html_fragment(value: str) -> str:
        text = re.sub(r"<script[\s\S]*?</script>", " ", value or "", flags=re.IGNORECASE)
        text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        return html.unescape(re.sub(r"\s+", " ", text)).strip()

    @classmethod
    def _extract_html_title(cls, value: str) -> str:
        match = re.search(r"<title[^>]*>([\s\S]*?)</title>", value or "", re.IGNORECASE)
        if not match:
            return ""
        return cls._clean_html_fragment(match.group(1))

    @staticmethod
    def _normalized_host(url: str) -> str:
        host = urlparse(str(url or "")).netloc.lower().strip()
        if ":" in host:
            host = host.split(":", 1)[0]
        if host.startswith("www."):
            host = host[4:]
        return host

    @classmethod
    def _is_same_domain(cls, left_url: str, right_url: str) -> bool:
        left_host = cls._normalized_host(left_url)
        right_host = cls._normalized_host(right_url)
        if not left_host or not right_host:
            return False
        return (
            left_host == right_host
            or left_host.endswith(f".{right_host}")
            or right_host.endswith(f".{left_host}")
        )

    @classmethod
    def _link_property_signal_score(
        cls,
        *,
        url: str,
        anchor_text: str,
        context_text: str,
    ) -> int:
        synthetic_item = {
            "url": url,
            "title": anchor_text,
            "description": context_text,
            "extra_snippets": [],
        }
        score = 0
        if is_single_property_search_result(synthetic_item):
            score += 3

        combined = f"{anchor_text} {context_text}"
        if any(token in combined for token in DETAIL_LINK_HINT_TOKENS):
            score += 1
        if re.search(r"(?:賃料|家賃)[^。\n]{0,18}(?:万|円)", combined):
            score += 1
        if re.search(r"徒歩\s*(?:約|およそ)?\s*\d{1,2}\s*分", combined):
            score += 1
        if re.search(r"\d(?:SLDK|SDK|LDK|DK|K|R)", combined, re.IGNORECASE):
            score += 1
        if re.search(r"\d{1,3}(?:\.\d+)?\s*(?:㎡|m2)", combined, re.IGNORECASE):
            score += 1
        if any(token in combined for token in COLLECTION_LINK_HINT_TOKENS):
            score -= 2
        return score

    def _extract_same_domain_detail_link_candidates(
        self,
        *,
        base_url: str,
        parent_item: dict[str, Any],
        html_text: str,
    ) -> list[dict[str, Any]]:
        candidates_by_url: dict[str, dict[str, Any]] = {}
        for index, match in enumerate(
            re.finditer(
                r"<a\b[^>]*href=(?:\"([^\"]+)\"|'([^']+)')[^>]*>([\s\S]*?)</a>",
                html_text or "",
                re.IGNORECASE,
            )
        ):
            href = html.unescape((match.group(1) or match.group(2) or "").strip())
            if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
                continue
            absolute_url = urljoin(base_url, href)
            parsed = urlparse(absolute_url)
            if parsed.scheme not in {"http", "https"}:
                continue
            absolute_url = parsed._replace(fragment="").geturl()
            if not self._is_same_domain(base_url, absolute_url):
                continue

            anchor_text = self._clean_html_fragment(match.group(3))[:120]
            context_html = html_text[
                max(0, match.start() - LINK_CONTEXT_WINDOW) : min(
                    len(html_text),
                    match.end() + LINK_CONTEXT_WINDOW,
                )
            ]
            context_text = self._clean_html_fragment(context_html)[:220]
            score = self._link_property_signal_score(
                url=absolute_url,
                anchor_text=anchor_text,
                context_text=context_text,
            )
            if score < 2 and self.research_adapter is None:
                continue
            if score < 1:
                continue

            candidate = {
                "url": absolute_url,
                "anchor_text": anchor_text,
                "context_text": context_text,
                "heuristic_score": score,
                "position": index,
                "parent_url": base_url,
                "parent_title": str(parent_item.get("title") or ""),
            }
            existing = candidates_by_url.get(absolute_url)
            if existing is None or (
                candidate["heuristic_score"],
                len(candidate["context_text"]),
            ) > (
                existing["heuristic_score"],
                len(existing["context_text"]),
            ):
                candidates_by_url[absolute_url] = candidate

        candidates = sorted(
            candidates_by_url.values(),
            key=lambda item: (
                int(item.get("heuristic_score") or 0),
                -int(item.get("position") or 0),
            ),
            reverse=True,
        )
        return candidates[:MAX_DETAIL_LINK_CANDIDATES_PER_PAGE]

    def _select_detail_link_candidates_with_llm(
        self,
        *,
        parent_item: dict[str, Any],
        parent_url: str,
        parent_html: str,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        fallback: list[dict[str, Any]] = []
        for threshold in (3, 2):
            for item in candidates:
                if int(item.get("heuristic_score") or 0) < threshold:
                    continue
                if item in fallback:
                    continue
                fallback.append(item)
                if len(fallback) >= MAX_FOLLOW_DETAIL_LINKS_PER_PAGE:
                    break
            if len(fallback) >= MAX_FOLLOW_DETAIL_LINKS_PER_PAGE:
                break
        if self.research_adapter is None or len(candidates) < 2:
            return fallback

        schema = {
            "type": "object",
            "properties": {
                "selected_indexes": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 0},
                }
            },
            "required": ["selected_indexes"],
            "additionalProperties": False,
        }
        payload = {
            "page": {
                "url": parent_url,
                "title": str(parent_item.get("title") or "")
                or self._extract_html_title(parent_html),
                "description": str(parent_item.get("description") or ""),
            },
            "candidate_links": [
                {
                    "candidate_index": index,
                    "url": item.get("url", ""),
                    "anchor_text": item.get("anchor_text", ""),
                    "context_text": item.get("context_text", ""),
                    "heuristic_score": item.get("heuristic_score", 0),
                }
                for index, item in enumerate(candidates[:MAX_DETAIL_LINK_CANDIDATES_PER_PAGE])
            ],
            "selection_rules": [
                "一覧ページ内から、単一の賃貸物件詳細ページに進みそうなリンクだけを選ぶ",
                "検索結果一覧、条件一覧、特集、地図、ランキング、カテゴリーページは選ばない",
                "同じ物件でもっとも具体的な詳細ページを優先し、最大4件まで選ぶ",
                "家賃、間取り、駅徒歩、募集住戸など単一物件の文脈があるリンクを優先する",
            ],
        }
        try:
            result = self.research_adapter.generate_structured(
                system=(
                    "You identify which links on a Japanese rental listing page most likely lead "
                    "to concrete single-property detail pages. Choose only the best candidates."
                ),
                user=json.dumps(payload, ensure_ascii=False, indent=2),
                schema=schema,
                temperature=0.0,
            )
        except Exception:
            return fallback

        selected_indexes: list[int] = []
        for value in result.get("selected_indexes", []) or []:
            try:
                index = int(value)
            except (TypeError, ValueError):
                continue
            if 0 <= index < len(candidates) and index not in selected_indexes:
                selected_indexes.append(index)
            if len(selected_indexes) >= MAX_FOLLOW_DETAIL_LINKS_PER_PAGE:
                break
        if not selected_indexes:
            return fallback
        return [candidates[index] for index in selected_indexes]

    def _build_discovered_detail_result(
        self,
        *,
        parent_item: dict[str, Any],
        child_candidate: dict[str, Any],
        child_html: str,
    ) -> dict[str, Any]:
        inherited_snippets = [
            str(item).strip()
            for item in parent_item.get("extra_snippets", []) or []
            if str(item).strip()
        ]
        context_text = str(child_candidate.get("context_text") or "").strip()
        child_title = self._extract_html_title(child_html)
        anchor_text = str(child_candidate.get("anchor_text") or "").strip()
        if anchor_text in {"見る", "詳細", "詳細を見る", "物件詳細"}:
            anchor_text = ""
        title = (
            anchor_text or child_title or str(parent_item.get("title") or "").strip() or "物件詳細"
        )
        return {
            "title": title,
            "url": str(child_candidate.get("url") or "").strip(),
            "description": context_text or str(parent_item.get("description") or "").strip(),
            "extra_snippets": inherited_snippets[:3],
            "source_name": str(parent_item.get("source_name") or "source"),
            "matched_queries": list(parent_item.get("matched_queries", []) or []),
            "source_names": list(parent_item.get("source_names", []) or []),
            "parent_url": str(parent_item.get("url") or "").strip(),
            "discovered_from": "listing_page_link",
            "discovered_anchor_text": str(child_candidate.get("anchor_text") or "").strip(),
        }

    def _update_live_progress(
        self,
        *,
        stage_name: str,
        progress_percent: int,
        current_action: str,
        detail: str = "",
        url: str = "",
    ) -> None:
        action = self._compact_progress_text(current_action, max_chars=72)
        detail_text = self._compact_progress_text(detail, max_chars=120)
        url_text = self._display_progress_url(url)

        activity_parts = [action]
        if detail_text:
            activity_parts.append(detail_text)
        if url_text:
            activity_parts.append(url_text)
        activity = " | ".join(part for part in activity_parts if part)

        with self._job_lock:
            if (
                self._current_live_activity
                and self._current_live_activity != activity
                and (
                    not self._recent_live_activities
                    or self._recent_live_activities[-1] != self._current_live_activity
                )
            ):
                self._recent_live_activities.append(self._current_live_activity)
                self._recent_live_activities = self._recent_live_activities[-4:]
            self._current_live_activity = activity

            lines = [f"現在: {action}"]
            if detail_text:
                lines.append(f"内容: {detail_text}")
            if url_text:
                lines.append(f"対象: {url_text}")
            if self._recent_live_activities:
                lines.append("直近:")
                lines.extend(f"- {item}" for item in reversed(self._recent_live_activities[-3:]))

        self._update_job(
            stage_name=stage_name,
            progress_percent=progress_percent,
            latest_summary="\n".join(lines),
        )

    # JP: cached singleflight loadを処理する。
    # EN: Process cached singleflight load.
    def _cached_singleflight_load(
        self,
        *,
        cache: dict[str, Any],
        inflight: dict[str, dict[str, Any]],
        key: str,
        loader: Callable[[], Any],
    ) -> tuple[Any, bool]:
        with self._cache_lock:
            if key in cache:
                return self._cache_copy(cache[key]), True
            entry = inflight.get(key)
            if entry is None:
                entry = {
                    "event": threading.Event(),
                    "error": None,
                    "value": None,
                }
                inflight[key] = entry
                owner = True
            else:
                owner = False

        if owner:
            try:
                value = loader()
            except Exception as exc:
                with self._cache_lock:
                    inflight.pop(key, None)
                    entry["error"] = exc
                    entry["event"].set()
                raise

            cached_value = self._cache_copy(value)
            with self._cache_lock:
                cache[key] = cached_value
                inflight.pop(key, None)
                entry["value"] = self._cache_copy(cached_value)
                entry["event"].set()
            return self._cache_copy(cached_value), False

        entry["event"].wait()
        if entry["error"] is not None:
            raise entry["error"]
        return self._cache_copy(entry["value"]), True

    # JP: state machineを構築する。
    # EN: Build state machine.
    def _build_state_machine(self) -> ResearchStateMachine:
        return ResearchStateMachine(
            [
                ResearchStageDefinition(
                    name="plan_finalize",
                    handler=self._handle_plan_finalize,
                    default_next_stage="tree_search",
                ),
                ResearchStageDefinition(
                    name="tree_search",
                    handler=self._handle_tree_search,
                    default_next_stage="synthesize",
                ),
                ResearchStageDefinition(
                    name="synthesize",
                    handler=self._handle_synthesize,
                    default_next_stage=None,
                ),
            ]
        )

    # JP: toolboxを構築する。
    # EN: Build toolbox.
    def _build_toolbox(self) -> Toolbox:
        node_schema = {
            "type": "object",
            "properties": {
                "node_key": {"type": "string"},
                "label": {"type": "string"},
                "description": {"type": "string"},
                "queries": {"type": "array", "items": {"type": "string"}},
                "ranking_profile": {"type": "object"},
                "strategy_tags": {"type": "array", "items": {"type": "string"}},
                "depth": {"type": "integer"},
                "parent_key": {"type": ["string", "null"]},
                "intent": {"type": "string"},
                "debug_depth": {"type": "integer"},
            },
            "required": [
                "node_key",
                "label",
                "description",
                "queries",
                "ranking_profile",
                "strategy_tags",
                "depth",
            ],
            "additionalProperties": True,
        }
        return Toolbox(
            [
                CallableResearchTool(
                    ToolSpec(
                        name="plan_finalize",
                        description="Fix an approved plan as the immutable starting point for tree search.",
                        output_schema={
                            "type": "object",
                            "properties": {
                                "summary": {"type": "string"},
                                "search_query": {"type": "string"},
                                "seed_queries": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["summary", "search_query", "seed_queries"],
                            "additionalProperties": True,
                        },
                    ),
                    self._tool_plan_finalize,
                ),
                CallableResearchTool(
                    ToolSpec(
                        name="retrieve",
                        description="Collect and merge search results for a search candidate.",
                        input_schema={
                            "type": "object",
                            "properties": {"branch": node_schema},
                            "required": ["branch"],
                            "additionalProperties": False,
                        },
                        output_schema={
                            "type": "object",
                            "properties": {
                                "raw_results": {"type": "array"},
                                "summary": {"type": "object"},
                                "per_query": {"type": "array"},
                            },
                            "required": ["raw_results", "summary", "per_query"],
                            "additionalProperties": True,
                        },
                    ),
                    self._tool_retrieve,
                ),
                CallableResearchTool(
                    ToolSpec(
                        name="enrich",
                        description="Fetch property detail pages for a search candidate.",
                        input_schema={
                            "type": "object",
                            "properties": {
                                "branch": node_schema,
                                "raw_results": {"type": "array"},
                            },
                            "required": ["branch", "raw_results"],
                            "additionalProperties": False,
                        },
                        output_schema={
                            "type": "object",
                            "properties": {
                                "detail_html_map": {"type": "object"},
                                "summary": {"type": "object"},
                            },
                            "required": ["detail_html_map", "summary"],
                            "additionalProperties": True,
                        },
                    ),
                    self._tool_enrich,
                ),
                CallableResearchTool(
                    ToolSpec(
                        name="normalize_dedupe",
                        description="Normalize raw results into a shared property schema and group duplicates.",
                        input_schema={
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "raw_results": {"type": "array"},
                                "detail_html_map": {"type": "object"},
                            },
                            "required": ["query", "raw_results", "detail_html_map"],
                            "additionalProperties": False,
                        },
                        output_schema={
                            "type": "object",
                            "properties": {
                                "normalized_properties": {"type": "array"},
                                "duplicate_groups": {"type": "array"},
                                "summary": {"type": "object"},
                            },
                            "required": ["normalized_properties", "duplicate_groups", "summary"],
                            "additionalProperties": True,
                        },
                    ),
                    self._tool_normalize,
                ),
                CallableResearchTool(
                    ToolSpec(
                        name="integrity_review",
                        description="Review listing integrity and drop stale or contradictory candidates before ranking.",
                        input_schema={
                            "type": "object",
                            "properties": {
                                "branch": node_schema,
                                "normalized_properties": {"type": "array"},
                                "raw_results": {"type": "array"},
                                "detail_html_map": {"type": "object"},
                            },
                            "required": [
                                "normalized_properties",
                                "raw_results",
                                "detail_html_map",
                            ],
                            "additionalProperties": False,
                        },
                        output_schema={
                            "type": "object",
                            "properties": {
                                "normalized_properties": {"type": "array"},
                                "integrity_reviews": {"type": "array"},
                                "dropped_property_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "summary": {"type": "object"},
                            },
                            "required": [
                                "normalized_properties",
                                "integrity_reviews",
                                "dropped_property_ids",
                                "summary",
                            ],
                            "additionalProperties": True,
                        },
                    ),
                    self._tool_integrity_review,
                ),
                CallableResearchTool(
                    ToolSpec(
                        name="rank",
                        description="Rank normalized properties under a candidate-specific scoring profile.",
                        input_schema={
                            "type": "object",
                            "properties": {
                                "branch": node_schema,
                                "normalized_properties": {"type": "array"},
                                "ranking_profile": {"type": "object"},
                            },
                            "required": ["normalized_properties", "ranking_profile"],
                            "additionalProperties": False,
                        },
                        output_schema={
                            "type": "object",
                            "properties": {
                                "ranked_properties": {"type": "array"},
                                "why_selected": {"type": "object"},
                                "why_not_selected": {"type": "object"},
                                "ranking_profile": {"type": "object"},
                            },
                            "required": [
                                "ranked_properties",
                                "why_selected",
                                "why_not_selected",
                                "ranking_profile",
                            ],
                            "additionalProperties": True,
                        },
                    ),
                    self._tool_rank,
                ),
            ]
        )

    # JP: jobを更新する。
    # EN: Update job.
    def _update_job(self, *, stage_name: str, progress_percent: int, latest_summary: str) -> None:
        with self._job_lock:
            resolved_progress = max(self._job_progress_percent, progress_percent)
            self._job_progress_percent = resolved_progress
            self.db.update_research_job(
                self.job_id,
                current_stage=stage_name,
                progress_percent=resolved_progress,
                latest_summary=latest_summary,
            )

    # JP: nodeを記録する。
    # EN: Record node.
    def _record_node(
        self,
        *,
        stage: str,
        node_type: str,
        status: str,
        input_payload: dict[str, Any],
        output_payload: dict[str, Any],
        reasoning: str,
        duration_ms: int = 0,
        parent_node_id: int | None = None,
        branch_id: str = "",
        selected: bool = False,
        intent: ResearchIntent = "draft",
        is_failed: bool | None = None,
        debug_depth: int = 0,
        metrics: dict[str, Any] | None = None,
    ) -> ResearchNode:
        resolved_is_failed = status == "failed" if is_failed is None else is_failed
        with self._journal_lock:
            row_id = self.db.add_research_journal_node(
                job_id=self.job_id,
                stage=stage,
                node_type=node_type,
                status=status,
                input_payload=input_payload,
                output_payload=output_payload,
                reasoning=reasoning,
                duration_ms=duration_ms,
                parent_node_id=parent_node_id,
                branch_id=branch_id,
                selected=selected,
                intent=intent,
                is_failed=resolved_is_failed,
                debug_depth=debug_depth,
                metrics_payload=metrics,
            )
            node = ResearchNode(
                id=row_id,
                stage=stage,
                node_type=node_type,
                status=status,
                input_payload=input_payload,
                output_payload=output_payload,
                reasoning=reasoning,
                intent=intent,
                is_failed=resolved_is_failed,
                debug_depth=debug_depth,
                duration_ms=duration_ms,
                parent_node_id=parent_node_id,
                branch_id=branch_id,
                selected=selected,
                metrics=metrics or {},
            )
            self.journal.append(node)
            return node

    # JP: recorded nodeを更新する。
    # EN: Update recorded node.
    def _update_recorded_node(
        self,
        node_id: int | None,
        *,
        status: str | None = None,
        input_payload: dict[str, Any] | None = None,
        output_payload: dict[str, Any] | None = None,
        reasoning: str | None = None,
        duration_ms: int | None = None,
        parent_node_id: int | None = None,
        branch_id: str | None = None,
        selected: bool | None = None,
        intent: ResearchIntent | None = None,
        is_failed: bool | None = None,
        debug_depth: int | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        if node_id is None:
            return

        with self._journal_lock:
            self.db.update_research_journal_node(
                node_id,
                status=status,
                input_payload=input_payload,
                output_payload=output_payload,
                reasoning=reasoning,
                duration_ms=duration_ms,
                parent_node_id=parent_node_id,
                branch_id=branch_id,
                selected=selected,
                intent=intent,
                is_failed=is_failed,
                debug_depth=debug_depth,
                metrics_payload=metrics,
            )

            node = self.journal.get_node(node_id)
            if node is None:
                return
            if status is not None:
                node.status = status
            if input_payload is not None:
                node.input_payload = input_payload
            if output_payload is not None:
                node.output_payload = output_payload
            if reasoning is not None:
                node.reasoning = reasoning
            if duration_ms is not None:
                node.duration_ms = duration_ms
            if parent_node_id is not None:
                node.parent_node_id = parent_node_id
            if branch_id is not None:
                node.branch_id = branch_id
            if selected is not None:
                node.selected = selected
            if intent is not None:
                node.intent = intent
            if is_failed is not None:
                node.is_failed = is_failed
            if debug_depth is not None:
                node.debug_depth = debug_depth
            if metrics is not None:
                node.metrics = metrics

    # JP: stageを実行する。
    # EN: Run stage.
    def _run_stage(
        self,
        *,
        stage_name: str,
        progress_percent: int,
        latest_summary: str,
        input_payload: dict[str, Any],
        reasoning: str,
        runner: Callable[[], dict[str, Any]],
    ) -> tuple[ResearchNode, dict[str, Any]]:
        self._update_job(
            stage_name=stage_name,
            progress_percent=progress_percent,
            latest_summary=latest_summary,
        )
        started = time.perf_counter()
        try:
            output = runner()
        except Exception as exc:  # pragma: no cover - exercised through integration path
            duration_ms = int((time.perf_counter() - started) * 1000)
            self._record_node(
                stage=stage_name,
                node_type="stage",
                status="failed",
                input_payload=input_payload,
                output_payload={"error": str(exc)},
                reasoning=reasoning,
                duration_ms=duration_ms,
            )
            raise RuntimeError(str(exc)) from exc

        duration_ms = int((time.perf_counter() - started) * 1000)
        node = self._record_node(
            stage=stage_name,
            node_type="stage",
            status="completed",
            input_payload=input_payload,
            output_payload=output,
            reasoning=reasoning,
            duration_ms=duration_ms,
        )
        return node, output

    # JP: tool plan finalizeを処理する。
    # EN: Process tool plan finalize.
    def _tool_plan_finalize(self, *, context: ToolContext) -> dict[str, Any]:
        seed_queries = [
            " ".join(str(item).split()).strip()
            for item in context.approved_plan.get("seed_queries", []) or []
            if " ".join(str(item).split()).strip()
        ]
        search_query = str(context.approved_plan.get("search_query") or "").strip()
        if not seed_queries and search_query:
            seed_queries = [search_query]
        return {
            "summary": (
                f"条件 {len(context.approved_plan.get('conditions', []))} 件で調査開始"
                f"（seed {len(seed_queries)}本）"
            ),
            "search_query": search_query or (seed_queries[0] if seed_queries else ""),
            "seed_queries": seed_queries,
        }

    # JP: tool retrieveを処理する。
    # EN: Process tool retrieve.
    def _tool_retrieve(self, *, context: ToolContext, branch: SearchNodePlan) -> dict[str, Any]:
        merged_by_url: dict[str, dict[str, Any]] = {}
        per_query: list[dict[str, Any]] = []
        catalog_total = 0
        brave_total = 0
        brave_errors: list[str] = []
        cache_hit_count = 0

        for index, query in enumerate(branch.queries, start=1):
            self._update_live_progress(
                stage_name="tree_search",
                progress_percent=36,
                current_action="検索結果を収集中",
                detail=f"{branch.label} / クエリ {index}/{len(branch.queries)}: {query}",
            )
            (results, source_summary), cache_hit = self._cached_singleflight_load(
                cache=self.search_result_cache,
                inflight=self._search_result_inflight,
                key=query,
                loader=lambda query=query: self.collect_search_results(
                    query=query,
                    user_memory=self._active_user_memory(),
                    adapter=self.research_adapter,
                ),
            )
            if cache_hit:
                cache_hit_count += 1
            catalog_total += int(source_summary.get("catalog_result_count") or 0)
            brave_total += int(source_summary.get("brave_result_count") or 0)
            if source_summary.get("brave_error"):
                brave_errors.append(str(source_summary["brave_error"]))

            per_query.append(
                {
                    "query": query,
                    "result_count": len(results),
                    "catalog_result_count": int(source_summary.get("catalog_result_count") or 0),
                    "brave_result_count": int(source_summary.get("brave_result_count") or 0),
                }
            )

            for item in results:
                url = str(item.get("url") or "").strip()
                if not url:
                    continue
                source_name = str(item.get("source_name") or "unknown")
                if url not in merged_by_url:
                    merged_by_url[url] = {
                        **item,
                        "matched_queries": [query],
                        "source_names": [source_name],
                        "source_name": source_name,
                    }
                    continue
                existing = merged_by_url[url]
                if query not in existing["matched_queries"]:
                    existing["matched_queries"].append(query)
                if source_name not in existing["source_names"]:
                    existing["source_names"].append(source_name)
                snippets = list(existing.get("extra_snippets", []) or [])
                for snippet in item.get("extra_snippets", []) or []:
                    text = str(snippet).strip()
                    if text and text not in snippets:
                        snippets.append(text)
                existing["extra_snippets"] = snippets[:6]
                if len(existing["source_names"]) > 1:
                    existing["source_name"] = "multi_source"

        raw_results = list(merged_by_url.values())
        return {
            "raw_results": raw_results,
            "summary": {
                "query_count": len(branch.queries),
                "unique_url_count": len(raw_results),
                "catalog_result_count": catalog_total,
                "brave_result_count": brave_total,
                "brave_error_count": len(brave_errors),
                "cache_hit_count": cache_hit_count,
            },
            "per_query": per_query,
        }

    # JP: tool enrichを処理する。
    # EN: Process tool enrich.
    def _tool_enrich(
        self,
        *,
        context: ToolContext,
        branch: SearchNodePlan,
        raw_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        detail_html_map: dict[str, str] = {}
        expanded_raw_results: list[dict[str, Any]] = []
        expanded_by_url: dict[str, dict[str, Any]] = {}
        total = len(raw_results)
        cache_hit_count = 0
        listing_page_expand_count = 0
        discovered_detail_count = 0
        child_fetch_attempt_count = 0
        child_candidate_count = 0
        for index, item in enumerate(raw_results, start=1):
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            detail_html, cache_hit = self._cached_singleflight_load(
                cache=self.detail_html_cache,
                inflight=self._detail_html_inflight,
                key=url,
                loader=lambda url=url: self.fetch_detail_html(url),
            )
            if cache_hit:
                cache_hit_count += 1
            if total:
                self._update_live_progress(
                    stage_name="tree_search",
                    progress_percent=48,
                    current_action="物件詳細ページを取得中",
                    detail=f"{branch.label} / {index}/{total} 件目",
                    url=url,
                )
            if not detail_html:
                continue

            if is_single_property_search_result(item, detail_html):
                detail_html_map[url] = detail_html
                expanded_by_url[url] = dict(item)
                continue

            listing_page_expand_count += 1
            link_candidates = self._extract_same_domain_detail_link_candidates(
                base_url=url,
                parent_item=item,
                html_text=detail_html,
            )
            child_candidate_count += len(link_candidates)
            selected_candidates = self._select_detail_link_candidates_with_llm(
                parent_item=item,
                parent_url=url,
                parent_html=detail_html,
                candidates=link_candidates,
            )
            if selected_candidates:
                self._update_live_progress(
                    stage_name="tree_search",
                    progress_percent=50,
                    current_action="一覧ページから詳細リンクを探索中",
                    detail=(f"{branch.label} / {len(selected_candidates)}件の詳細候補を追跡"),
                    url=url,
                )
            for child_candidate in selected_candidates:
                child_url = str(child_candidate.get("url") or "").strip()
                if not child_url or child_url in detail_html_map or child_url in expanded_by_url:
                    continue
                child_fetch_attempt_count += 1
                child_html, child_cache_hit = self._cached_singleflight_load(
                    cache=self.detail_html_cache,
                    inflight=self._detail_html_inflight,
                    key=child_url,
                    loader=lambda child_url=child_url: self.fetch_detail_html(child_url),
                )
                if child_cache_hit:
                    cache_hit_count += 1
                if not child_html:
                    continue
                discovered_item = self._build_discovered_detail_result(
                    parent_item=item,
                    child_candidate=child_candidate,
                    child_html=child_html,
                )
                if not is_single_property_search_result(discovered_item, child_html):
                    continue
                detail_html_map[child_url] = child_html
                expanded_by_url[child_url] = discovered_item
                discovered_detail_count += 1
                if len(expanded_by_url) >= MAX_EXPANDED_RAW_RESULTS:
                    break
            if len(expanded_by_url) >= MAX_EXPANDED_RAW_RESULTS:
                break

        if expanded_by_url:
            expanded_raw_results = list(expanded_by_url.values())
        else:
            expanded_raw_results = []
        return {
            "detail_html_map": detail_html_map,
            "expanded_raw_results": expanded_raw_results,
            "summary": {
                "detail_attempt_count": total,
                "detail_hit_count": len(detail_html_map),
                "cache_hit_count": cache_hit_count,
                "listing_page_expand_count": listing_page_expand_count,
                "listing_page_child_candidate_count": child_candidate_count,
                "child_fetch_attempt_count": child_fetch_attempt_count,
                "discovered_detail_count": discovered_detail_count,
                "expanded_result_count": len(expanded_raw_results),
                "summary": (
                    f"単一物件ページ {len(detail_html_map)} 件を取得"
                    f"（一覧ページ深掘り {discovered_detail_count} 件）"
                ),
            },
        }

    # JP: tool normalizeを処理する。
    # EN: Process tool normalize.
    def _tool_normalize(
        self,
        *,
        context: ToolContext,
        query: str,
        raw_results: list[dict[str, Any]],
        detail_html_map: dict[str, str],
    ) -> dict[str, Any]:
        return run_search_and_normalize(
            query=query,
            search_results=raw_results,
            detail_fetcher=lambda url: detail_html_map.get(url),
            adapter=self.research_adapter,
            image_resolver=(
                lambda item, prop, detail_html="": self.resolve_property_image(
                    search_result=item,
                    property_data=prop,
                    detail_html=detail_html,
                    adapter=self.research_adapter,
                )
            )
            if self.resolve_property_image is not None
            else None,
        )

    # JP: tool rankを処理する。
    # EN: Process tool rank.
    def _tool_rank(
        self,
        *,
        context: ToolContext,
        branch: SearchNodePlan | None = None,
        normalized_properties: list[dict[str, Any]],
        ranking_profile: dict[str, Any],
    ) -> dict[str, Any]:
        return run_ranking(
            normalized_properties=normalized_properties,
            user_memory=self._active_user_memory(),
            ranking_profile=ranking_profile,
            adapter=self.research_adapter,
            area_scope=branch.area_scope if branch is not None else "strict",
            nearby_hints=branch.nearby_hints if branch is not None else None,
        )

    # JP: tool integrity reviewを処理する。
    # EN: Process tool integrity review.
    def _tool_integrity_review(
        self,
        *,
        context: ToolContext,
        branch: SearchNodePlan | None = None,
        normalized_properties: list[dict[str, Any]],
        raw_results: list[dict[str, Any]],
        detail_html_map: dict[str, str],
    ) -> dict[str, Any]:
        user_mem = self._active_user_memory()
        target_area = str(user_mem.get("target_area") or "").strip()
        listing_type = str(user_mem.get("listing_type") or "").strip()
        layout_preference = str(user_mem.get("layout_preference") or "").strip()
        must_conditions = [
            str(item).strip()
            for item in user_mem.get("must_conditions", []) or []
            if str(item).strip()
        ]
        return run_integrity_review(
            normalized_properties=normalized_properties,
            raw_results=raw_results,
            detail_html_map=detail_html_map,
            adapter=self.research_adapter,
            target_area=target_area,
            listing_type=listing_type,
            layout_preference=layout_preference,
            must_conditions=must_conditions,
            area_scope=branch.area_scope if branch is not None else "strict",
            constraint_mode=branch.constraint_mode if branch is not None else "primary",
            nearby_hints=branch.nearby_hints if branch is not None else None,
        )
