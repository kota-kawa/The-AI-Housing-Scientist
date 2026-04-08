from __future__ import annotations

from typing import Any

from app.config import ProviderName, Settings, get_provider_api_key, get_provider_model
from app.db import Database
from app.llm import create_adapter
from app.llm_config import (
    build_default_llm_config,
    get_llm_route_definitions,
    normalize_llm_config,
    route_config_for,
)


class LLMConfigManagerMixin:
    settings: Settings
    db: Database
    _model_cache: dict[str, list[str]]

    def _list_models_for_provider(self, provider: ProviderName) -> list[str]:
        if provider not in self._model_cache:
            adapter = create_adapter(self.settings, provider)
            self._model_cache[provider] = adapter.list_models()
        return self._model_cache[provider]

    def _build_model_provider_catalog(
        self,
    ) -> tuple[dict[str, ProviderName], list[dict[str, Any]]]:
        provider_by_model: dict[str, ProviderName] = {}
        model_options: list[dict[str, Any]] = []
        configured_entries: list[tuple[ProviderName, str]] = [
            ("openai", self.settings.openai_model),
            ("gemini", self.settings.gemini_model),
            ("claude", self.settings.claude_model),
            ("groq", self.settings.groq_model_primary),
            ("groq", self.settings.groq_model_secondary),
        ]

        for provider_typed, configured_model in configured_entries:
            model_text = str(configured_model).strip()
            if not model_text:
                continue

            key_present = bool(get_provider_api_key(self.settings, provider_typed))
            available = False
            if key_present:
                try:
                    provider_models = self._list_models_for_provider(provider_typed)
                    available = model_text in provider_models
                except Exception:
                    available = False

            provider_by_model[model_text] = provider_typed
            model_options.append(
                {
                    "model": model_text,
                    "provider": provider_typed,
                    "available": available,
                }
            )
        return provider_by_model, model_options

    def _resolve_provider_for_model(self, model: str) -> ProviderName:
        catalog, _ = self._build_model_provider_catalog()
        model_text = str(model).strip()
        provider = catalog.get(model_text)
        if provider is not None:
            return provider
        raise RuntimeError(f"unknown model: {model_text}")

    def _normalize_llm_config(self, raw_config: Any) -> dict[str, Any]:
        return normalize_llm_config(self.settings, raw_config)

    def _build_default_llm_config(self) -> dict[str, Any]:
        return build_default_llm_config(self.settings)

    def _ensure_session_llm_config(
        self,
        session_id: str,
        *,
        user_memory: dict[str, Any] | None = None,
        task_memory: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        if user_memory is None or task_memory is None:
            user_memory, task_memory = self.db.get_memories(session_id)

        normalized = self._normalize_llm_config(task_memory.get("draft_llm_config"))
        if task_memory.get("draft_llm_config") != normalized:
            task_memory["draft_llm_config"] = normalized
            self.db.update_memories(session_id, user_memory, task_memory)
        return user_memory, task_memory, normalized

    def _active_job_for_session(self, task_memory: dict[str, Any]) -> dict[str, Any] | None:
        active_job_id = str(task_memory.get("active_research_job_id") or "").strip()
        return self.db.get_research_job(active_job_id) if active_job_id else None

    def get_session_llm_config(self, session_id: str) -> dict[str, Any]:
        if not self.db.session_exists(session_id):
            raise RuntimeError("session not found")

        user_memory, task_memory, llm_config = self._ensure_session_llm_config(session_id)
        active_job = self._active_job_for_session(task_memory)
        editable = not (
            active_job is not None and active_job["status"] in {"queued", "running"}
        )
        return {
            "session_id": session_id,
            "preset": str(llm_config.get("preset") or "custom"),
            "routes": llm_config.get("routes", {}),
            "editable": editable,
            "active_job_id": active_job["id"] if active_job and not editable else None,
        }

    def _validate_llm_config(self, llm_config: dict[str, Any]) -> None:
        catalog, _ = self._build_model_provider_catalog()
        for route_definition in get_llm_route_definitions():
            route_key = route_definition["key"]
            route = route_config_for(llm_config, route_key)  # type: ignore[arg-type]
            model = str(route["model"]).strip()
            if not model:
                raise RuntimeError(f"model is required for {route_key}")
            provider = catalog.get(model)
            if provider is None:
                raise RuntimeError(f"unknown model for {route_key}: {model}")
            if not get_provider_api_key(self.settings, provider):
                if self.settings.model_strict_mode:
                    raise RuntimeError(f"API key missing for {provider}")
                continue
            try:
                models = self._list_models_for_provider(provider)
            except Exception as exc:
                if self.settings.model_strict_mode:
                    raise RuntimeError(
                        f"failed to list models for {provider}: {exc}"
                    ) from exc
                continue
            if models and model not in models:
                raise RuntimeError(f"model not available for {provider}: {model}")

    def update_session_llm_config(
        self, session_id: str, config_payload: dict[str, Any]
    ) -> dict[str, Any]:
        if not self.db.session_exists(session_id):
            raise RuntimeError("session not found")

        user_memory, task_memory, _ = self._ensure_session_llm_config(session_id)
        active_job = self._active_job_for_session(task_memory)
        if active_job is not None and active_job["status"] in {"queued", "running"}:
            raise RuntimeError("LLM settings are locked while research is running")

        normalized = self._normalize_llm_config(config_payload)
        self._validate_llm_config(normalized)
        task_memory["draft_llm_config"] = normalized
        self.db.update_memories(session_id, user_memory, task_memory)
        return self.get_session_llm_config(session_id)

    def get_llm_capabilities(self) -> dict[str, Any]:
        providers: dict[str, Any] = {}
        _, model_options = self._build_model_provider_catalog()
        for provider in ["openai", "gemini", "groq", "claude"]:
            default_model = get_provider_model(self.settings, provider)  # type: ignore[arg-type]
            key_present = bool(get_provider_api_key(self.settings, provider))  # type: ignore[arg-type]
            reachable = False
            models: list[str] = []
            details = ""
            if key_present:
                try:
                    models = self._list_models_for_provider(provider)  # type: ignore[arg-type]
                    reachable = True
                    details = f"available={len(models)}"
                except Exception as exc:
                    details = str(exc)
            else:
                details = "API key missing"

            if default_model and default_model not in models:
                models = [default_model, *models]

            providers[provider] = {
                "key_present": key_present,
                "reachable": reachable,
                "default_model": default_model,
                "models": models,
                "details": details,
            }

        return {
            "route_definitions": get_llm_route_definitions(),
            "providers": providers,
            "models": model_options,
            "default_config": self._build_default_llm_config(),
        }
