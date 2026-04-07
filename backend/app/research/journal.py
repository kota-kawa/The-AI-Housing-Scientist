from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ResearchNode:
    stage: str
    node_type: str
    status: str
    input_payload: dict[str, Any]
    output_payload: dict[str, Any]
    reasoning: str
    duration_ms: int = 0
    parent_node_id: int | None = None
    branch_id: str = ""
    selected: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)
    id: int | None = None


class ResearchJournal:
    def __init__(self) -> None:
        self.nodes: list[ResearchNode] = []

    def append(self, node: ResearchNode) -> ResearchNode:
        self.nodes.append(node)
        return node

    @property
    def stage_nodes(self) -> list[ResearchNode]:
        return [node for node in self.nodes if node.node_type == "stage"]

    @property
    def branch_nodes(self) -> list[ResearchNode]:
        return [node for node in self.nodes if node.branch_id]

    def get_node(self, node_id: int | None) -> ResearchNode | None:
        if node_id is None:
            return None
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def children_of(self, node_id: int | None) -> list[ResearchNode]:
        return [node for node in self.nodes if node.parent_node_id == node_id]

    def branch_root(self, branch_id: str) -> ResearchNode | None:
        for node in self.nodes:
            if node.branch_id == branch_id and node.node_type == "branch_root":
                return node
        return None

    def selected_branch_nodes(self) -> list[ResearchNode]:
        return [node for node in self.nodes if node.selected]

    def latest_stage_node(self, stage: str) -> ResearchNode | None:
        candidates = [node for node in self.stage_nodes if node.stage == stage]
        if not candidates:
            return None
        return candidates[-1]
