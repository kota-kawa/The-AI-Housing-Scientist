from __future__ import annotations

from dataclasses import dataclass, field
import threading
from typing import Any, Literal

ResearchIntent = Literal["draft", "refine", "pivot", "recovery"]


@dataclass
class ResearchNode:
    stage: str
    node_type: str
    status: str
    input_payload: dict[str, Any]
    output_payload: dict[str, Any]
    reasoning: str
    intent: ResearchIntent = "draft"
    is_failed: bool = False
    debug_depth: int = 0
    duration_ms: int = 0
    parent_node_id: int | None = None
    branch_id: str = ""
    selected: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)
    id: int | None = None


class ResearchJournal:
    # JP: クラスやインスタンスの初期状態を設定する。
    # EN: Initialize the class or instance state.
    def __init__(self) -> None:
        self.nodes: list[ResearchNode] = []
        self._lock = threading.RLock()

    # JP: appendを処理する。
    # EN: Process append.
    def append(self, node: ResearchNode) -> ResearchNode:
        with self._lock:
            self.nodes.append(node)
            return node

    # JP: stage nodesを処理する。
    # EN: Process stage nodes.
    @property
    def stage_nodes(self) -> list[ResearchNode]:
        with self._lock:
            return [node for node in self.nodes if node.node_type == "stage"]

    # JP: branch nodesを処理する。
    # EN: Process branch nodes.
    @property
    def branch_nodes(self) -> list[ResearchNode]:
        with self._lock:
            return [node for node in self.nodes if node.branch_id]

    # JP: nodeを取得する。
    # EN: Get node.
    def get_node(self, node_id: int | None) -> ResearchNode | None:
        if node_id is None:
            return None
        with self._lock:
            for node in self.nodes:
                if node.id == node_id:
                    return node
        return None

    # JP: children ofを処理する。
    # EN: Process children of.
    def children_of(self, node_id: int | None) -> list[ResearchNode]:
        with self._lock:
            return [node for node in self.nodes if node.parent_node_id == node_id]

    # JP: branch rootを処理する。
    # EN: Process branch root.
    def branch_root(self, branch_id: str) -> ResearchNode | None:
        with self._lock:
            for node in self.nodes:
                if node.branch_id == branch_id and node.node_type == "branch_root":
                    return node
        return None

    # JP: selected branch nodesを処理する。
    # EN: Process selected branch nodes.
    def selected_branch_nodes(self) -> list[ResearchNode]:
        with self._lock:
            selected_nodes = [node for node in self.nodes if node.selected]
            if not selected_nodes:
                return []

            selected_branch_ids: list[str] = []
            for node in selected_nodes:
                branch_id = str(node.branch_id or "").strip()
                if branch_id and branch_id not in selected_branch_ids:
                    selected_branch_ids.append(branch_id)

                selected_branch = node.output_payload.get("selected_branch", {}) or {}
                branch_id = str(selected_branch.get("branch_id") or "").strip()
                if branch_id and branch_id not in selected_branch_ids:
                    selected_branch_ids.append(branch_id)

                for item in node.output_payload.get("selected_path", []) or []:
                    branch_id = str(item.get("branch_id") or "").strip()
                    if branch_id and branch_id not in selected_branch_ids:
                        selected_branch_ids.append(branch_id)

            if not selected_branch_ids:
                return selected_nodes

            return [
                node
                for node in self.nodes
                if node.selected or str(node.branch_id or "").strip() in selected_branch_ids
            ]

    # JP: latest stage nodeを処理する。
    # EN: Process latest stage node.
    def latest_stage_node(self, stage: str) -> ResearchNode | None:
        with self._lock:
            candidates = [
                node for node in self.nodes if node.node_type == "stage" and node.stage == stage
            ]
            if not candidates:
                return None
            return candidates[-1]
