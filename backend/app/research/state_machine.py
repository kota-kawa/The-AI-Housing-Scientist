from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ResearchStageDefinition:
    name: str
    handler: Callable[[Any], str | None]
    default_next_stage: str | None = None


class ResearchStateMachine:
    def __init__(self, stages: list[ResearchStageDefinition]):
        self._stages = {stage.name: stage for stage in stages}

    def run(self, state: Any, *, start_stage: str) -> Any:
        current_stage = start_stage
        guard = 0

        while current_stage is not None:
            guard += 1
            if guard > max(1, len(self._stages) * 3):
                raise RuntimeError("research state machine exceeded transition guard")
            stage = self._stages.get(current_stage)
            if stage is None:
                raise KeyError(f"unknown research stage: {current_stage}")
            next_stage = stage.handler(state)
            current_stage = next_stage if next_stage is not None else stage.default_next_stage

        return state
