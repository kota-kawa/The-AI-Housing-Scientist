from __future__ import annotations

from typing import Any

import httpx
import jsonschema

from .base import LLMAdapter, LLMUsage
from .utils import extract_json_object, with_current_date_context


class AnthropicAdapter(LLMAdapter):
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: int,
        max_retries: int,
    ):
        self.api_key = api_key.strip()
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.base_url = "https://api.anthropic.com/v1"
        self._last_usage: LLMUsage | None = None

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        last_error: Exception | None = None

        with httpx.Client(timeout=self.timeout_seconds) as client:
            for attempt in range(self.max_retries + 1):
                try:
                    response = client.request(method, url, headers=self._headers, json=payload)
                    response.raise_for_status()
                    return response.json()
                except httpx.HTTPError as exc:
                    last_error = exc
                    status = getattr(getattr(exc, "response", None), "status_code", None)
                    non_retryable = status is not None and status not in (408, 409, 429, 500, 502, 503, 504)
                    if non_retryable or attempt == self.max_retries:
                        break

        raise RuntimeError(f"claude request failed: {last_error}")

    def _extract_usage(self, response: dict[str, Any]) -> LLMUsage:
        usage = response.get("usage", {}) or {}
        prompt_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
        return LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            raw=usage if isinstance(usage, dict) else {},
        )

    def _messages_create(self, *, system: str, user: str, temperature: float) -> str:
        response = self._request(
            "POST",
            "/messages",
            {
                "model": self.model,
                "max_tokens": 2048,
                "temperature": temperature,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
        )
        self._last_usage = self._extract_usage(response)
        content = response.get("content", [])
        texts = [part.get("text", "") for part in content if part.get("type") == "text"]
        return "\n".join([t for t in texts if t])

    def generate_text(self, *, system: str, user: str, temperature: float = 0.2) -> str:
        system = with_current_date_context(system)
        return self._messages_create(system=system, user=user, temperature=temperature)

    def generate_structured(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        system = with_current_date_context(system)
        strict_user = (
            f"{user}\n\n"
            "Return only one JSON object that satisfies this JSON Schema:\n"
            f"{schema}"
        )
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_usage_raw: dict[str, Any] = {}

        for attempt in range(self.max_retries + 1):
            text = self._messages_create(system=system, user=strict_user, temperature=temperature)
            usage = self._last_usage or LLMUsage()
            total_prompt_tokens += usage.prompt_tokens
            total_completion_tokens += usage.completion_tokens
            if usage.raw:
                total_usage_raw = usage.raw
            try:
                payload = extract_json_object(text)
                jsonschema.validate(payload, schema)
                self._last_usage = LLMUsage(
                    prompt_tokens=total_prompt_tokens,
                    completion_tokens=total_completion_tokens,
                    total_tokens=total_prompt_tokens + total_completion_tokens,
                    raw=total_usage_raw,
                )
                return payload
            except Exception as exc:
                if attempt == self.max_retries:
                    self._last_usage = LLMUsage(
                        prompt_tokens=total_prompt_tokens,
                        completion_tokens=total_completion_tokens,
                        total_tokens=total_prompt_tokens + total_completion_tokens,
                        raw=total_usage_raw,
                    )
                    raise RuntimeError(f"claude structured response validation failed: {exc}") from exc
                strict_user = (
                    strict_user
                    + "\n\nThe previous response was invalid. Return only valid JSON without markdown fences."
                )

        raise RuntimeError("claude structured response failed unexpectedly")

    def list_models(self) -> list[str]:
        response = self._request("GET", "/models")
        data = response.get("data", [])
        return sorted([item.get("id", "") for item in data if item.get("id")])

    def get_last_usage(self) -> LLMUsage | None:
        return self._last_usage
