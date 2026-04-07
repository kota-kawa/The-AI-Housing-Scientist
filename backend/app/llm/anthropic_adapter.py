from __future__ import annotations

from typing import Any

import httpx
import jsonschema

from .base import LLMAdapter
from .utils import extract_json_object


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
        content = response.get("content", [])
        texts = [part.get("text", "") for part in content if part.get("type") == "text"]
        return "\n".join([t for t in texts if t])

    def generate_text(self, *, system: str, user: str, temperature: float = 0.2) -> str:
        return self._messages_create(system=system, user=user, temperature=temperature)

    def generate_structured(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        strict_user = (
            f"{user}\n\n"
            "Return only one JSON object that satisfies this JSON Schema:\n"
            f"{schema}"
        )

        for attempt in range(self.max_retries + 1):
            text = self._messages_create(system=system, user=strict_user, temperature=temperature)
            try:
                payload = extract_json_object(text)
                jsonschema.validate(payload, schema)
                return payload
            except Exception as exc:
                if attempt == self.max_retries:
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
