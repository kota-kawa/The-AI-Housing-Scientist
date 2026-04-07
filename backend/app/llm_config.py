from __future__ import annotations

from typing import Any, Literal

from app.config import ProviderName, Settings, get_provider_api_key, get_provider_model

LLMRouteKey = Literal["planner", "research_default", "communication", "risk_check"]

LLM_ROUTE_DEFINITIONS: list[dict[str, str]] = [
    {
        "key": "planner",
        "label": "条件整理",
        "description": "検索条件の抽出と調査計画の作成に使います。",
    },
    {
        "key": "research_default",
        "label": "調査本体",
        "description": "query_expand などの調査フローで使う既定モデルです。",
    },
    {
        "key": "communication",
        "label": "問い合わせ文",
        "description": "問い合わせ文の下書き生成に使います。",
    },
    {
        "key": "risk_check",
        "label": "契約チェック",
        "description": "契約条項のリスク整理に使います。",
    },
]

_STAGE_PROVIDER_PREFERENCES: dict[LLMRouteKey, list[ProviderName]] = {
    "planner": ["openai", "claude", "gemini", "groq"],
    "research_default": ["openai", "gemini", "claude", "groq"],
    "communication": ["claude", "openai", "gemini", "groq"],
    "risk_check": ["gemini", "openai", "claude", "groq"],
}


def get_llm_route_keys() -> list[LLMRouteKey]:
    return [item["key"] for item in LLM_ROUTE_DEFINITIONS]  # type: ignore[return-value]


def get_llm_route_definitions() -> list[dict[str, str]]:
    return [dict(item) for item in LLM_ROUTE_DEFINITIONS]


def _pick_default_provider(settings: Settings, route_key: LLMRouteKey) -> ProviderName:
    for provider in _STAGE_PROVIDER_PREFERENCES[route_key]:
        if get_provider_api_key(settings, provider):
            return provider
    return settings.llm_default_provider


def build_default_llm_config(settings: Settings) -> dict[str, Any]:
    routes: dict[str, dict[str, str]] = {}
    for route_key in get_llm_route_keys():
        provider = _pick_default_provider(settings, route_key)
        routes[route_key] = {
            "model": get_provider_model(settings, provider),
        }
    return {"preset": "default", "routes": routes}


def normalize_llm_config(settings: Settings, raw_config: Any) -> dict[str, Any]:
    default_config = build_default_llm_config(settings)
    raw_routes = raw_config.get("routes", {}) if isinstance(raw_config, dict) else {}
    routes: dict[str, dict[str, str]] = {}

    for route_key in get_llm_route_keys():
        default_route = dict(default_config["routes"][route_key])
        raw_route = raw_routes.get(route_key, {}) if isinstance(raw_routes, dict) else {}
        model = str(raw_route.get("model") or "").strip() or str(default_route["model"])
        routes[route_key] = {"model": model}

    preset = str(raw_config.get("preset") or "") if isinstance(raw_config, dict) else ""
    if not preset:
        preset = "default" if routes == default_config["routes"] else "custom"

    return {"preset": preset, "routes": routes}


def route_config_for(config: dict[str, Any], route_key: LLMRouteKey) -> dict[str, str]:
    routes = config.get("routes", {}) if isinstance(config, dict) else {}
    route = routes.get(route_key, {}) if isinstance(routes, dict) else {}
    return {
        "model": str(route.get("model") or ""),
    }
