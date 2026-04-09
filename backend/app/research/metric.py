from __future__ import annotations

from dataclasses import dataclass
from functools import total_ordering
from typing import Any


@dataclass(frozen=True)
@total_ordering
class MetricValue:
    value: float | None
    maximize: bool = True
    name: str | None = None

    def __post_init__(self) -> None:
        if self.value is None:
            return
        object.__setattr__(self, "value", float(self.value))

    @classmethod
    def from_raw(cls, value: Any, *, maximize: bool = True, name: str | None = None) -> MetricValue:
        if value is None or isinstance(value, bool):
            return WorstMetricValue(maximize=maximize, name=name)
        try:
            return cls(float(value), maximize=maximize, name=name)
        except (TypeError, ValueError):
            return WorstMetricValue(maximize=maximize, name=name)

    @property
    def is_worst(self) -> bool:
        return self.value is None

    def as_float(self) -> float | None:
        return self.value

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, MetricValue):
            return NotImplemented
        if self.is_worst:
            return False
        if other.is_worst:
            return True
        if self.maximize != other.maximize:
            raise ValueError("Cannot compare metrics with different optimize directions")
        assert self.value is not None and other.value is not None
        return self.value > other.value if self.maximize else self.value < other.value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MetricValue):
            return NotImplemented
        return (
            self.value == other.value
            and self.maximize == other.maximize
            and self.name == other.name
        )


@dataclass(frozen=True)
class WorstMetricValue(MetricValue):
    value: float | None = None
