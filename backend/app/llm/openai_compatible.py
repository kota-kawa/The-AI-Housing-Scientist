from __future__ import annotations

import json
from typing import Any

import httpx
import jsonschema

from .base import LLMAdapter, LLMUsage
from .utils import extract_json_object, flatten_content, with_current_date_context


class OpenAICompatibleAdapter(LLMAdapter):
    def __init__(
        self,
        *,
        provider_name: str,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: int,
        max_retries: int,
    ):
        self.provider_name = provider_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key.strip()
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self._last_usage: LLMUsage | None = None

    def _should_hide_reasoning(self) -> bool:
        return self.provider_name == "groq" and str(self.model).strip().lower() == "qwen/qwen3-32b"

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        last_error: Exception | None = None

        with httpx.Client(timeout=self.timeout_seconds) as client:
            for attempt in range(self.max_retries + 1):
                try:
                    response = client.request(method, url, headers=self._headers, json=payload)
                    response.raise_for_status()
                    return response.json()
                except (httpx.HTTPError, json.JSONDecodeError) as exc:
                    last_error = exc
                    status = getattr(getattr(exc, "response", None), "status_code", None)
                    non_retryable = status is not None and status not in (
                        408,
                        409,
                        429,
                        500,
                        502,
                        503,
                        504,
                    )
                    if non_retryable or attempt == self.max_retries:
                        break

        raise RuntimeError(f"{self.provider_name} request failed: {last_error}")

    def _extract_usage(self, response: dict[str, Any]) -> LLMUsage:
        usage = response.get("usage", {}) or {}
        prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
        return LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            raw=usage if isinstance(usage, dict) else {},
        )

    def _chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if self._should_hide_reasoning():
            payload["reasoning_format"] = "hidden"
        if response_format is not None:
            payload["response_format"] = response_format

        response = self._request("POST", "/chat/completions", payload)
        self._last_usage = self._extract_usage(response)
        return response

    def generate_text(self, *, system: str, user: str, temperature: float = 0.2) -> str:
        system = with_current_date_context(system)
        response = self._chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
        message = response["choices"][0]["message"]
        return flatten_content(message.get("content", ""))

    def generate_structured(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        system = with_current_date_context(system)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "structured_response",
                "schema": schema,
                "strict": True,
            },
        }
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_usage_raw: dict[str, Any] = {}

        for attempt in range(self.max_retries + 1):
            response = self._chat(
                messages=messages,
                temperature=temperature,
                response_format=response_format,
            )
            usage = self._last_usage or LLMUsage()
            total_prompt_tokens += usage.prompt_tokens
            total_completion_tokens += usage.completion_tokens
            if usage.raw:
                total_usage_raw = usage.raw
            message = response["choices"][0]["message"]
            content = flatten_content(message.get("content", ""))
            try:
                payload = extract_json_object(content)
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
                    raise RuntimeError(
                        f"{self.provider_name} structured response validation failed: {exc}"
                    ) from exc
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The previous response did not satisfy the required JSON schema. "
                            "Return only one valid JSON object matching the schema exactly."
                        ),
                    }
                )

        raise RuntimeError(f"{self.provider_name} structured response failed unexpectedly")

    def list_models(self) -> list[str]:
        response = self._request("GET", "/models")
        data = response.get("data", [])
        return sorted([item.get("id", "") for item in data if item.get("id")])

    def get_last_usage(self) -> LLMUsage | None:
        return self._last_usage
