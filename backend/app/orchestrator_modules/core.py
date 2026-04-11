from __future__ import annotations

from typing import Any

from app.config import ProviderName, Settings, get_provider_model
from app.db import Database
from app.llm import create_adapter
from app.llm.base import LLMAdapter
from app.llm.observability import (
    DatabaseLLMObserver,
    LLMObservationContext,
    ObservedLLMAdapter,
    build_cost_estimator,
)
from app.llm_config import LLMRouteKey, route_config_for
from app.models import ChatMessageResponse
from app.services import PropertyCatalogService, PropertyImageResolver


class OrchestratorCoreMixin:
    settings: Settings
    db: Database

    # JP: クラスやインスタンスの初期状態を設定する。
    # EN: Initialize the class or instance state.
    def __init__(self, settings: Settings, db: Database):
        self.settings = settings
        self.db = db
        self.catalog = PropertyCatalogService(db)
        self.property_images = PropertyImageResolver(
            brave_api_key=settings.brave_search_api_key,
            timeout_seconds=settings.llm_timeout_seconds,
        )
        self._model_cache: dict[str, list[str]] = {}
        self._llm_observer = DatabaseLLMObserver(
            db,
            cost_estimator=build_cost_estimator(settings.llm_pricing_overrides_json),
        )
        self._prime_catalog_notes()

    # JP: prime catalog notesを処理する。
    # EN: Process prime catalog notes.
    def _prime_catalog_notes(self) -> None:
        """起動時にカタログの notes をLLMでリライトする（APIキーがある場合のみ、失敗は無視）。"""
        try:
            default_config = self._build_default_llm_config()
            adapter = self._get_adapter_for_route(
                llm_config=default_config,
                route_key="research_default",
                interaction_type="catalog_rewrite",
            )
            if adapter is not None:
                self.catalog.rewrite_notes_with_llm(adapter)
        except Exception:
            pass

    # JP: profile memory for sessionを取得する。
    # EN: Get profile memory for session.
    def _get_profile_memory_for_session(self, session_id: str) -> dict[str, Any]:
        return {}

    # JP: adapter or noneを取得する。
    # EN: Get adapter or none.
    def _get_adapter_or_none(
        self,
        provider: ProviderName,
        *,
        model: str | None = None,
        session_id: str | None = None,
        job_id: str | None = None,
        interaction_type: str = "general",
    ) -> LLMAdapter | None:
        key_map = {
            "openai": self.settings.openai_api_key,
            "gemini": self.settings.gemini_api_key,
            "groq": self.settings.groq_api_key,
            "claude": self.settings.claude_api_key,
        }
        if not key_map[provider]:
            if self.settings.model_strict_mode:
                raise RuntimeError(f"strict mode: API key missing for {provider}")
            return None

        resolved_model = str(model or get_provider_model(self.settings, provider)).strip()
        adapter = create_adapter(self.settings, provider, model=resolved_model)

        if self.settings.model_strict_mode:
            models = self._list_models_for_provider(provider)
            if resolved_model not in models:
                raise RuntimeError(
                    f"strict mode: model ID not available for {provider}: {resolved_model}"
                )

        if session_id is not None or job_id is not None:
            model_name = getattr(adapter, "model", resolved_model)
            adapter = ObservedLLMAdapter(
                wrapped=adapter,
                observer=self._llm_observer,
                context_factory=lambda operation, metadata: LLMObservationContext(
                    session_id=session_id,
                    job_id=job_id,
                    operation=f"{interaction_type}:{operation}",
                    provider=provider,
                    model=model_name,
                    metadata=metadata,
                ),
            )

        return adapter

    # JP: adapter for routeを取得する。
    # EN: Get adapter for route.
    def _get_adapter_for_route(
        self,
        *,
        llm_config: dict[str, Any],
        route_key: LLMRouteKey,
        session_id: str | None = None,
        job_id: str | None = None,
        interaction_type: str,
    ) -> LLMAdapter | None:
        route = route_config_for(llm_config, route_key)
        model = str(route["model"]).strip()
        provider = self._resolve_provider_for_model(model)
        return self._get_adapter_or_none(
            provider,
            model=model,
            session_id=session_id,
            job_id=job_id,
            interaction_type=interaction_type,
        )

    # JP: moneyを整形する。
    # EN: Format money.
    def _format_money(self, value: Any) -> str:
        amount = int(value or 0)
        return f"{amount:,}円" if amount > 0 else "要確認"

    # JP: areaを整形する。
    # EN: Format area.
    def _format_area(self, value: Any) -> str:
        area = float(value or 0)
        return f"{area:.1f}㎡" if area > 0 else "要確認"

    # JP: walkを整形する。
    # EN: Format walk.
    def _format_walk(self, value: Any) -> str:
        minutes = int(value or 0)
        return f"{minutes}分" if minutes > 0 else "要確認"

    # JP: property reactionsを取得する。
    # EN: Get property reactions.
    def _get_property_reactions(self, task_memory: dict[str, Any]) -> dict[str, str]:
        reactions = task_memory.get("property_reactions", {}) or {}
        return {
            str(property_id): str(reaction)
            for property_id, reaction in reactions.items()
            if str(property_id).strip() and str(reaction).strip()
        }

    # JP: display ranked propertiesを取得する。
    # EN: Get display ranked properties.
    def _display_ranked_properties(self, task_memory: dict[str, Any]) -> list[dict[str, Any]]:
        return list(
            task_memory.get("last_display_ranked_properties")
            or task_memory.get("last_ranked_properties")
            or []
        )

    # JP: display normalized propertiesを取得する。
    # EN: Get display normalized properties.
    def _display_normalized_properties(self, task_memory: dict[str, Any]) -> list[dict[str, Any]]:
        return list(
            task_memory.get("last_display_normalized_properties")
            or task_memory.get("last_normalized_properties")
            or []
        )

    # JP: visible ranked propertiesを処理する。
    # EN: Process visible ranked properties.
    def _visible_ranked_properties(
        self,
        ranked_properties: list[dict[str, Any]],
        task_memory: dict[str, Any],
    ) -> list[dict[str, Any]]:
        excluded_ids = {
            property_id
            for property_id, reaction in self._get_property_reactions(task_memory).items()
            if reaction == "exclude"
        }
        return [item for item in ranked_properties if item["property_id_norm"] not in excluded_ids]

    # JP: sync profile after searchを処理する。
    # EN: Process sync profile after search.
    def _sync_profile_after_search(
        self,
        *,
        profile_id: str,
        user_memory: dict[str, Any],
        query: str,
        adapter: LLMAdapter | None = None,
        search_outcome: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {key: value for key, value in user_memory.items() if key != "learned_preferences"}

    # JP: sync profile after reactionを処理する。
    # EN: Process sync profile after reaction.
    def _sync_profile_after_reaction(
        self,
        *,
        profile_id: str,
        property_snapshot: dict[str, Any],
        reaction: str,
        adapter: LLMAdapter | None = None,
        strategy_context: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        return None

    # JP: session initial responseを構築する。
    # EN: Build session initial response.
    def build_session_initial_response(self, session_id: str) -> ChatMessageResponse | None:
        return None

    # JP: property nameを探索する。
    # EN: Find property name.
    def _find_property_name(self, task_memory: dict[str, Any], property_id: str | None) -> str:
        if not property_id:
            return "選択中の物件"
        for item in self._display_normalized_properties(task_memory):
            if item.get("property_id_norm") == property_id:
                return str(item.get("building_name") or "選択中の物件")
        return "選択中の物件"
