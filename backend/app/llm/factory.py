from __future__ import annotations

import httpx

from app.config import ProviderName, Settings, get_provider_model

from .anthropic_adapter import AnthropicAdapter
from .base import LLMAdapter
from .openai_compatible import OpenAICompatibleAdapter


class GeminiOpenAIAdapter(OpenAICompatibleAdapter):
    # JP: クラスやインスタンスの初期状態を設定する。
    # EN: Initialize the class or instance state.
    def __init__(self, *, api_key: str, model: str, timeout_seconds: int, max_retries: int):
        super().__init__(
            provider_name="gemini",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=api_key,
            model=model,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )

    # JP: modelsを一覧化する。
    # EN: List models.
    def list_models(self) -> list[str]:
        url = "https://generativelanguage.googleapis.com/v1beta/models"
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.get(url, params={"key": self.api_key})
            response.raise_for_status()
            payload = response.json()
        return sorted(
            [
                item.get("name", "").replace("models/", "")
                for item in payload.get("models", [])
                if item.get("name")
            ]
        )


class GroqOpenAIAdapter(OpenAICompatibleAdapter):
    # JP: クラスやインスタンスの初期状態を設定する。
    # EN: Initialize the class or instance state.
    def __init__(self, *, api_key: str, model: str, timeout_seconds: int, max_retries: int):
        super().__init__(
            provider_name="groq",
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
            model=model,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )


class OpenAIAdapter(OpenAICompatibleAdapter):
    # JP: クラスやインスタンスの初期状態を設定する。
    # EN: Initialize the class or instance state.
    def __init__(self, *, api_key: str, model: str, timeout_seconds: int, max_retries: int):
        super().__init__(
            provider_name="openai",
            base_url="https://api.openai.com/v1",
            api_key=api_key,
            model=model,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )


# JP: adapterを作成する。
# EN: Create adapter.
def create_adapter(
    settings: Settings,
    provider: ProviderName,
    *,
    model: str | None = None,
) -> LLMAdapter:
    resolved_model = model or get_provider_model(settings, provider)

    if provider == "openai":
        return OpenAIAdapter(
            api_key=settings.openai_api_key,
            model=resolved_model,
            timeout_seconds=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )

    if provider == "gemini":
        return GeminiOpenAIAdapter(
            api_key=settings.gemini_api_key,
            model=resolved_model,
            timeout_seconds=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )

    if provider == "groq":
        return GroqOpenAIAdapter(
            api_key=settings.groq_api_key,
            model=resolved_model,
            timeout_seconds=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )

    return AnthropicAdapter(
        api_key=settings.claude_api_key,
        model=resolved_model,
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )
