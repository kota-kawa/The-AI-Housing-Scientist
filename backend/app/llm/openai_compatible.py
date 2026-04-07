from __future__ import annotations

import json
from typing import Any

import httpx
import jsonschema

from .base import LLMAdapter
from .utils import extract_json_object, flatten_content


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

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
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
                except (httpx.HTTPError, json.JSONDecodeError) as exc:
                    last_error = exc
                    status = getattr(getattr(exc, "response", None), "status_code", None)
                    non_retryable = status is not None and status not in (408, 409, 429, 500, 502, 503, 504)
                    if non_retryable or attempt == self.max_retries:
                        break

        raise RuntimeError(f"{self.provider_name} request failed: {last_error}")

    def _chat(self, messages: list[dict[str, Any]], temperature: float, response_format: dict[str, Any] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        return self._request("POST", "/chat/completions", payload)

    def generate_text(self, *, system: str, user: str, temperature: float = 0.2) -> str:
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

        for attempt in range(self.max_retries + 1):
            response = self._chat(
                messages=messages,
                temperature=temperature,
                response_format=response_format,
            )
            message = response["choices"][0]["message"]
            content = flatten_content(message.get("content", ""))
            try:
                payload = extract_json_object(content)
                jsonschema.validate(payload, schema)
                return payload
            except Exception as exc:
                if attempt == self.max_retries:
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
