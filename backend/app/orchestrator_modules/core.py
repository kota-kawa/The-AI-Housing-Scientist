from __future__ import annotations

from typing import Any

from app.config import ProviderName, Settings, get_provider_model
from app.db import Database, utc_now_iso
from app.llm import create_adapter
from app.llm.base import LLMAdapter
from app.llm.observability import (
    DatabaseLLMObserver,
    LLMObservationContext,
    ObservedLLMAdapter,
    build_cost_estimator,
)
from app.llm_config import LLMRouteKey, route_config_for
from app.models import ChatMessageResponse, UIBlock
from app.profile_memory import (
    build_profile_resume_summary,
    merge_learned_preferences,
    summarize_memory_labels,
    update_profile_memory_with_reaction,
    update_profile_memory_with_search,
)
from app.services import PropertyCatalogService, PropertyImageResolver

from .shared import _generate_llm_resume_body


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
        session = self.db.get_session(session_id)
        if session is None or not session.get("profile_id"):
            return {}
        profile = self.db.get_profile(session["profile_id"])
        if profile is None:
            return {}
        return profile.get("profile_memory", {}) or {}

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
        profile = self.db.get_profile(profile_id)
        if profile is None:
            return user_memory

        updated_profile_memory = update_profile_memory_with_search(
            profile["profile_memory"],
            query=query,
            user_memory=user_memory,
            searched_at=utc_now_iso(),
            adapter=adapter,
            search_outcome=search_outcome,
        )
        merged_user_memory = merge_learned_preferences(
            {key: value for key, value in user_memory.items() if key != "learned_preferences"},
            updated_profile_memory.get("learned_preferences", {}) or {},
        )
        self.db.update_profile(profile_id, merged_user_memory, updated_profile_memory)
        return merged_user_memory

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
        profile = self.db.get_profile(profile_id)
        if profile is None:
            return None

        updated_profile_memory = update_profile_memory_with_reaction(
            profile["profile_memory"],
            reaction=reaction,
            property_snapshot=property_snapshot,
            recorded_at=utc_now_iso(),
            adapter=adapter,
            strategy_context=strategy_context,
        )
        merged_user_memory = merge_learned_preferences(
            {
                key: value
                for key, value in profile["user_memory"].items()
                if key != "learned_preferences"
            },
            updated_profile_memory.get("learned_preferences", {}) or {},
        )
        self.db.update_profile(profile_id, merged_user_memory, updated_profile_memory)
        return merged_user_memory

    # JP: session initial responseを構築する。
    # EN: Build session initial response.
    def build_session_initial_response(self, session_id: str) -> ChatMessageResponse | None:
        session = self.db.get_session(session_id)
        if session is None:
            return None

        profile = self.db.get_profile(session["profile_id"])
        if profile is None:
            return None

        last_user_memory = {
            key: value
            for key, value in profile["user_memory"].items()
            if key != "learned_preferences"
        }
        if not summarize_memory_labels(last_user_memory):
            return None

        profile_summary = build_profile_resume_summary(last_user_memory, profile["profile_memory"])

        # Fallback body: 構造化テキスト
        summary_lines: list[str] = []
        if profile_summary["last_search_labels"]:
            summary_lines.append(
                f"前回の条件: {' / '.join(profile_summary['last_search_labels'][:5])}"
            )
        if profile_summary["frequent_area"]:
            summary_lines.append(f"よく検索するエリア: {profile_summary['frequent_area']}")
        if profile_summary["stable_preferences"]:
            summary_lines.append(
                f"変えない条件の候補: {' / '.join(profile_summary['stable_preferences'][:3])}"
            )
        if profile_summary["liked_features"]:
            summary_lines.append(
                f"気になる傾向: {' / '.join(profile_summary['liked_features'][:3])}"
            )
        resume_body = "\n".join(summary_lines)

        # LLMで自然な再開メッセージを生成（失敗時はフォールバック）
        try:
            _, _, llm_config = self._ensure_session_llm_config(session_id)
            adapter = self._get_adapter_for_route(
                llm_config=llm_config,
                route_key="planner",
                session_id=session_id,
                interaction_type="profile_resume",
            )
            if adapter is not None:
                llm_body = _generate_llm_resume_body(profile_summary, adapter)
                if llm_body:
                    resume_body = llm_body
        except Exception:
            pass

        _, task_memory = self.db.get_memories(session_id)
        task_memory["profile_resume_pending"] = True
        task_memory["profile_resume_summary"] = profile_summary
        self.db.update_memories(session_id, {}, task_memory)

        return ChatMessageResponse(
            status="awaiting_profile_resume",
            assistant_message="前回の条件を引き継ぎますか？",
            missing_slots=[],
            next_action="resume_or_reset_profile",
            blocks=[
                UIBlock(
                    type="text",
                    title="引き継ぎ候補",
                    content={"body": resume_body},
                ),
                UIBlock(
                    type="actions",
                    title="開始方法",
                    content={
                        "items": [
                            {
                                "label": "前回の条件を引き継ぐ",
                                "action_type": "resume_profile_memory",
                                "payload": {},
                            },
                            {
                                "label": "新しく探し始める",
                                "action_type": "dismiss_profile_resume",
                                "payload": {},
                            },
                        ]
                    },
                ),
            ],
            pending_confirmation=False,
            pending_action=None,
        )

    # JP: property nameを探索する。
    # EN: Find property name.
    def _find_property_name(self, task_memory: dict[str, Any], property_id: str | None) -> str:
        if not property_id:
            return "選択中の物件"
        for item in task_memory.get("last_normalized_properties", []):
            if item.get("property_id_norm") == property_id:
                return str(item.get("building_name") or "選択中の物件")
        return "選択中の物件"
