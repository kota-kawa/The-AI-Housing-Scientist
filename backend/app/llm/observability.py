from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
import time
from typing import Any

from app.db import Database

from .base import LLMAdapter, LLMUsage


@dataclass(frozen=True)
class TokenPricing:
    prompt_per_1m_tokens_usd: float
    completion_per_1m_tokens_usd: float


def build_cost_estimator(
    pricing_overrides_json: str,
) -> Callable[[str, str, int, int], float | None]:
    try:
        payload = json.loads(pricing_overrides_json or "{}")
    except json.JSONDecodeError:
        payload = {}

    pricing_map: dict[str, dict[str, TokenPricing]] = {}
    if isinstance(payload, dict):
        for provider, provider_config in payload.items():
            if not isinstance(provider_config, dict):
                continue
            pricing_map[str(provider)] = {}
            for model, row in provider_config.items():
                if not isinstance(row, dict):
                    continue
                prompt_cost = row.get("prompt_per_1m_tokens_usd")
                completion_cost = row.get("completion_per_1m_tokens_usd")
                try:
                    pricing_map[str(provider)][str(model)] = TokenPricing(
                        prompt_per_1m_tokens_usd=float(prompt_cost),
                        completion_per_1m_tokens_usd=float(completion_cost),
                    )
                except (TypeError, ValueError):
                    continue

    def estimate(
        provider: str, model: str, prompt_tokens: int, completion_tokens: int
    ) -> float | None:
        provider_prices = pricing_map.get(provider, {})
        price = provider_prices.get(model) or provider_prices.get("*")
        if price is None:
            return None
        estimated = (prompt_tokens / 1_000_000) * price.prompt_per_1m_tokens_usd + (
            completion_tokens / 1_000_000
        ) * price.completion_per_1m_tokens_usd
        return round(estimated, 8)

    return estimate


@dataclass(frozen=True)
class LLMObservationContext:
    session_id: str | None
    job_id: str | None
    operation: str
    provider: str
    model: str
    metadata: dict[str, Any]


class DatabaseLLMObserver:
    def __init__(
        self,
        db: Database,
        *,
        cost_estimator: Callable[[str, str, int, int], float | None] | None = None,
    ):
        self.db = db
        self.cost_estimator = cost_estimator

    def record(
        self,
        *,
        context: LLMObservationContext,
        prompt_chars: int,
        response_chars: int,
        duration_ms: int,
        success: bool,
        error_message: str = "",
        usage: LLMUsage | None = None,
    ) -> None:
        usage = usage or LLMUsage()
        estimated_cost_usd = usage.estimated_cost_usd
        if estimated_cost_usd is None and self.cost_estimator is not None:
            estimated_cost_usd = self.cost_estimator(
                context.provider,
                context.model,
                usage.prompt_tokens,
                usage.completion_tokens,
            )
        self.db.add_llm_call_event(
            session_id=context.session_id,
            job_id=context.job_id,
            provider=context.provider,
            model=context.model,
            operation=context.operation,
            prompt_chars=prompt_chars,
            response_chars=response_chars,
            duration_ms=duration_ms,
            success=success,
            error_message=error_message,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            estimated_cost_usd=estimated_cost_usd,
            metadata=context.metadata,
        )


class ObservedLLMAdapter(LLMAdapter):
    def __init__(
        self,
        *,
        wrapped: LLMAdapter,
        observer: DatabaseLLMObserver,
        context_factory: Callable[[str, dict[str, Any]], LLMObservationContext],
    ):
        self.wrapped = wrapped
        self.observer = observer
        self.context_factory = context_factory

    def _measure_prompt(self, system: str, user: str, extra: dict[str, Any] | None = None) -> int:
        payload = {
            "system": system,
            "user": user,
            "extra": extra or {},
        }
        return len(str(payload))

    def generate_text(self, *, system: str, user: str, temperature: float = 0.2) -> str:
        prompt_chars = self._measure_prompt(system, user, {"temperature": temperature})
        context = self.context_factory("generate_text", {"temperature": temperature})
        started = time.perf_counter()
        try:
            text = self.wrapped.generate_text(system=system, user=user, temperature=temperature)
        except Exception as exc:
            self.observer.record(
                context=context,
                prompt_chars=prompt_chars,
                response_chars=0,
                duration_ms=int((time.perf_counter() - started) * 1000),
                success=False,
                error_message=str(exc),
                usage=self.wrapped.get_last_usage(),
            )
            raise

        self.observer.record(
            context=context,
            prompt_chars=prompt_chars,
            response_chars=len(text),
            duration_ms=int((time.perf_counter() - started) * 1000),
            success=True,
            usage=self.wrapped.get_last_usage(),
        )
        return text

    def generate_structured(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        schema_keys = sorted(schema.get("properties", {}).keys())
        prompt_chars = self._measure_prompt(
            system,
            user,
            {"temperature": temperature, "schema_keys": schema_keys},
        )
        context = self.context_factory(
            "generate_structured",
            {"temperature": temperature, "schema_keys": schema_keys},
        )
        started = time.perf_counter()
        try:
            payload = self.wrapped.generate_structured(
                system=system,
                user=user,
                schema=schema,
                temperature=temperature,
            )
        except Exception as exc:
            self.observer.record(
                context=context,
                prompt_chars=prompt_chars,
                response_chars=0,
                duration_ms=int((time.perf_counter() - started) * 1000),
                success=False,
                error_message=str(exc),
                usage=self.wrapped.get_last_usage(),
            )
            raise

        self.observer.record(
            context=context,
            prompt_chars=prompt_chars,
            response_chars=len(str(payload)),
            duration_ms=int((time.perf_counter() - started) * 1000),
            success=True,
            usage=self.wrapped.get_last_usage(),
        )
        return payload

    def list_models(self) -> list[str]:
        return self.wrapped.list_models()
