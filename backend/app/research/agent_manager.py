from __future__ import annotations

from collections.abc import Callable
import copy
import threading
from typing import Any

from app.db import Database
from app.llm.base import LLMAdapter
from app.research.journal import ResearchJournal
from app.research.tools import ToolContext

from .agent_manager_execution_mixin import AgentManagerExecutionMixin
from .agent_manager_summary_mixin import AgentManagerSummaryMixin
from .agent_manager_tooling_mixin import AgentManagerToolingMixin
from .agent_manager_tree_mixin import AgentManagerTreeMixin
from .agent_manager_types import (
    ResearchExecutionResult,
    ResearchExecutionState,
    SearchNodeArtifacts,
    SearchNodePlan,
)

SearchNodePlan.__module__ = __name__
SearchNodeArtifacts.__module__ = __name__
ResearchExecutionState.__module__ = __name__
ResearchExecutionResult.__module__ = __name__


class HousingResearchAgentManager(
    AgentManagerExecutionMixin,
    AgentManagerTreeMixin,
    AgentManagerToolingMixin,
    AgentManagerSummaryMixin,
):
    def __init__(
        self,
        *,
        db: Database,
        session_id: str,
        job_id: str,
        approved_plan: dict[str, Any],
        user_memory: dict[str, Any],
        task_memory: dict[str, Any],
        provider: str,
        research_adapter: LLMAdapter | None,
        build_research_queries: Callable[[dict[str, Any], list[str]], list[str]],
        collect_search_results: Callable[..., tuple[list[dict[str, Any]], dict[str, Any]]],
        fetch_detail_html: Callable[[str], str | None],
        collect_source_items: Callable[..., list[dict[str, Any]]],
        resolve_property_image: Callable[..., str] | None = None,
        tree_max_nodes: int = 12,
        tree_max_depth: int = 4,
        tree_batch_size: int = 3,
        tree_children_per_expansion: int = 2,
        tree_prune_score: int = 35,
        tree_stability_patience: int = 2,
        tree_min_nodes_before_stable_stop: int = 8,
        tree_min_best_score_gap: float = 5.0,
    ):
        self.db = db
        self.session_id = session_id
        self.job_id = job_id
        self.approved_plan = approved_plan
        self.user_memory = user_memory
        self.task_memory = task_memory
        self.provider = provider
        self.research_adapter = research_adapter
        self.build_research_queries = build_research_queries
        self.collect_search_results = collect_search_results
        self.fetch_detail_html = fetch_detail_html
        self.collect_source_items = collect_source_items
        self.resolve_property_image = resolve_property_image
        self.tree_max_nodes = max(4, tree_max_nodes)
        self.tree_max_depth = max(1, tree_max_depth)
        self.tree_batch_size = max(1, min(5, tree_batch_size))
        self.tree_children_per_expansion = max(1, tree_children_per_expansion)
        self.tree_prune_score = max(1, tree_prune_score)
        self.tree_stability_patience = max(1, tree_stability_patience)
        self.tree_min_nodes_before_stable_stop = max(1, tree_min_nodes_before_stable_stop)
        self.tree_min_best_score_gap = max(0.0, float(tree_min_best_score_gap))
        self.search_result_cache: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
        self.detail_html_cache: dict[str, str | None] = {}
        self._search_result_inflight: dict[str, dict[str, Any]] = {}
        self._detail_html_inflight: dict[str, dict[str, Any]] = {}
        self._cache_lock = threading.RLock()
        self._job_lock = threading.RLock()
        self._journal_lock = threading.RLock()
        self._job_progress_percent = 0
        self._cache_copy = copy.deepcopy
        self.journal = ResearchJournal()
        self.context = ToolContext(
            session_id=session_id,
            job_id=job_id,
            user_memory=user_memory,
            task_memory=task_memory,
            approved_plan=approved_plan,
            provider=provider,
        )
        self.toolbox = self._build_toolbox()
        self.state_machine = self._build_state_machine()
