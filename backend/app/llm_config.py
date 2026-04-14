from __future__ import annotations

from typing import Any, Literal

from app.config import Settings, get_provider_model

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

# JP: LLM route keysを取得する。
# EN: Get LLM route keys.
def get_llm_route_keys() -> list[LLMRouteKey]:
    return [item["key"] for item in LLM_ROUTE_DEFINITIONS]  # type: ignore[return-value]


# JP: LLM route definitionsを取得する。
# EN: Get LLM route definitions.
def get_llm_route_definitions() -> list[dict[str, str]]:
    return [dict(item) for item in LLM_ROUTE_DEFINITIONS]


# JP: default LLM configを構築する。
# EN: Build default LLM config.
def build_default_llm_config(settings: Settings) -> dict[str, Any]:
    default_model = str(settings.groq_model_primary).strip() or get_provider_model(settings, "groq")
    routes: dict[str, dict[str, str]] = {}
    for route_key in get_llm_route_keys():
        routes[route_key] = {
            "model": default_model,
        }
    return {"preset": "default", "routes": routes}


# JP: LLM configを正規化する。
# EN: Normalize LLM config.
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


# JP: route config forを処理する。
# EN: Process route config for.
def route_config_for(config: dict[str, Any], route_key: LLMRouteKey) -> dict[str, str]:
    routes = config.get("routes", {}) if isinstance(config, dict) else {}
    route = routes.get(route_key, {}) if isinstance(routes, dict) else {}
    return {
        "model": str(route.get("model") or ""),
    }
