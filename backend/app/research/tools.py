from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

import jsonschema
from jsonschema import ValidationError


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolContext:
    session_id: str
    job_id: str
    user_memory: dict[str, Any]
    task_memory: dict[str, Any]
    approved_plan: dict[str, Any]
    provider: str


class BaseResearchTool(ABC):
    spec: ToolSpec

    # JP: 必要な処理を実行する。
    # EN: Run the required data.
    @abstractmethod
    def run(self, context: ToolContext, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError


class CallableResearchTool(BaseResearchTool):
    # JP: クラスやインスタンスの初期状態を設定する。
    # EN: Initialize the class or instance state.
    def __init__(self, spec: ToolSpec, runner: Callable[..., dict[str, Any]]):
        self.spec = spec
        self._runner = runner

    # JP: 必要な処理を実行する。
    # EN: Run the required data.
    def run(self, context: ToolContext, **kwargs: Any) -> dict[str, Any]:
        return self._runner(context=context, **kwargs)


class Toolbox:
    # JP: クラスやインスタンスの初期状態を設定する。
    # EN: Initialize the class or instance state.
    def __init__(self, tools: list[BaseResearchTool]):
        self._tools = {tool.spec.name: tool for tool in tools}

    # JP: for validationを正規化する。
    # EN: Normalize for validation.
    def _normalize_for_validation(self, value: Any) -> Any:
        if is_dataclass(value):
            return {
                key: self._normalize_for_validation(item) for key, item in asdict(value).items()
            }
        if isinstance(value, dict):
            return {str(key): self._normalize_for_validation(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._normalize_for_validation(item) for item in value]
        return value

    # JP: 必要な処理を検証する。
    # EN: Validate the required data.
    def _validate(self, schema: dict[str, Any], payload: Any, *, label: str) -> None:
        if not schema:
            return
        try:
            jsonschema.validate(self._normalize_for_validation(payload), schema)
        except ValidationError as exc:
            raise ValueError(f"{label} schema validation failed: {exc.message}") from exc

    # JP: 必要な処理を取得する。
    # EN: Get the required data.
    def get(self, name: str) -> BaseResearchTool:
        if name not in self._tools:
            raise KeyError(f"tool not found: {name}")
        return self._tools[name]

    # JP: 必要な処理を実行する。
    # EN: Run the required data.
    def run(self, name: str, context: ToolContext, **kwargs: Any) -> dict[str, Any]:
        tool = self.get(name)
        self._validate(tool.spec.input_schema, kwargs, label=f"{name}.input")
        output = tool.run(context, **kwargs)
        self._validate(tool.spec.output_schema, output, label=f"{name}.output")
        return output
