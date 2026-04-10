from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class LLMAdapter(ABC):
    # JP: generate textを処理する。
    # EN: Process generate text.
    @abstractmethod
    def generate_text(self, *, system: str, user: str, temperature: float = 0.2) -> str:
        raise NotImplementedError

    # JP: generate structuredを処理する。
    # EN: Process generate structured.
    @abstractmethod
    def generate_structured(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        raise NotImplementedError

    # JP: modelsを一覧化する。
    # EN: List models.
    @abstractmethod
    def list_models(self) -> list[str]:
        raise NotImplementedError

    # JP: last usageを取得する。
    # EN: Get last usage.
    def get_last_usage(self) -> LLMUsage | None:
        return None
