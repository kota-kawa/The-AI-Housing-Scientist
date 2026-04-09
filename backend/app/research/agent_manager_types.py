from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.research.journal import ResearchNode


@dataclass(frozen=True)
class SearchNodePlan:
    node_key: str
    label: str
    description: str
    queries: list[str]
    ranking_profile: dict[str, Any]
    strategy_tags: list[str]
    depth: int
    parent_key: str | None = None
    parent_node_id: int | None = None


@dataclass
class SearchNodeArtifacts:
    plan: SearchNodePlan
    query_hash: str
    frontier_score: float
    branch_score: float = 0.0
    readiness: str = "low"
    retrieve: dict[str, Any] = field(default_factory=dict)
    enrich: dict[str, Any] = field(default_factory=dict)
    normalize: dict[str, Any] = field(default_factory=dict)
    integrity: dict[str, Any] = field(default_factory=dict)
    rank: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    status: str = "queued"
    issue_streak: int = 0
    journal_node_id: int | None = None


@dataclass
class ResearchExecutionState:
    plan_result: dict[str, Any] = field(default_factory=dict)
    query: str = ""
    seed_queries: list[str] = field(default_factory=list)
    retry_context: dict[str, Any] = field(default_factory=dict)
    root_node: ResearchNode | None = None
    node_sequence: int = 0
    node_plans: dict[str, SearchNodePlan] = field(default_factory=dict)
    node_artifacts: dict[str, SearchNodeArtifacts] = field(default_factory=dict)
    frontier: list[str] = field(default_factory=list)
    branch_summaries: list[dict[str, Any]] = field(default_factory=list)
    branch_failures: dict[str, dict[str, Any]] = field(default_factory=dict)
    pruned_nodes: list[dict[str, Any]] = field(default_factory=list)
    selected_branch_summary: dict[str, Any] = field(default_factory=dict)
    selected_path: list[dict[str, Any]] = field(default_factory=list)
    best_node_key: str = ""
    best_node_stability: int = 0
    best_node_readiness: str = "low"
    best_score_gap: float = 0.0
    termination_reason: str = ""
    source_items: list[dict[str, Any]] = field(default_factory=list)
    search_summary: dict[str, Any] = field(default_factory=dict)
    offline_evaluation: dict[str, Any] = field(default_factory=dict)
    failure_summary: dict[str, Any] = field(default_factory=dict)
    research_summary: str = ""
    search_tree_summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchExecutionResult:
    query: str
    selected_branch_id: str
    branch_summaries: list[dict[str, Any]]
    branch_result_summary: dict[str, Any]
    final_report_markdown: str
    normalized_properties: list[dict[str, Any]]
    ranked_properties: list[dict[str, Any]]
    duplicate_groups: list[dict[str, Any]]
    integrity_reviews: list[dict[str, Any]]
    dropped_property_ids: list[str]
    raw_results: list[dict[str, Any]]
    source_items: list[dict[str, Any]]
    search_summary: dict[str, Any]
    offline_evaluation: dict[str, Any]
    failure_summary: dict[str, Any]
    research_summary: str
    selected_path: list[dict[str, Any]]
    search_tree_summary: dict[str, Any]
    pruned_nodes: list[dict[str, Any]]
