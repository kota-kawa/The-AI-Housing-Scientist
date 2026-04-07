from __future__ import annotations

from typing import Any

from app.config import Settings, get_provider_api_key
from app.llm.factory import create_adapter
from app.models import PreflightProviderReport, PreflightReport
from app.services.brave_search import BraveSearchClient


def _provider_model_valid(settings: Settings, provider: str, models: list[str]) -> tuple[bool, str]:
    if provider == "openai":
        wanted = settings.openai_model
        return wanted in models, f"wanted={wanted}"
    if provider == "gemini":
        wanted = settings.gemini_model
        return wanted in models, f"wanted={wanted}"
    if provider == "claude":
        wanted = settings.claude_model
        return wanted in models, f"wanted={wanted}"

    primary = settings.groq_model_primary
    secondary = settings.groq_model_secondary
    return primary in models, f"wanted={primary}; secondary={secondary}"


def run_preflight(settings: Settings) -> tuple[PreflightReport, bool]:
    providers: dict[str, PreflightProviderReport] = {}

    for provider in ["openai", "gemini", "groq", "claude"]:
        api_key = get_provider_api_key(settings, provider)  # type: ignore[arg-type]
        if not api_key:
            providers[provider] = PreflightProviderReport(
                key_present=False,
                reachable=False,
                model_valid=False,
                details="API key missing",
            )
            continue

        try:
            adapter = create_adapter(settings, provider)  # type: ignore[arg-type]
            models = adapter.list_models()
            model_valid, wanted_msg = _provider_model_valid(settings, provider, models)
            providers[provider] = PreflightProviderReport(
                key_present=True,
                reachable=True,
                model_valid=model_valid,
                details=f"{wanted_msg}; available={len(models)}",
            )
        except Exception as exc:
            providers[provider] = PreflightProviderReport(
                key_present=True,
                reachable=False,
                model_valid=False,
                details=str(exc),
            )

    brave_reachable = False
    if settings.brave_search_api_key:
        try:
            client = BraveSearchClient(settings.brave_search_api_key, timeout_seconds=10)
            results = client.search("賃貸 初期費用", count=1)
            brave_reachable = len(results) > 0
        except Exception:
            brave_reachable = False

    report = PreflightReport(
        strict_mode=settings.model_strict_mode,
        brave_reachable=brave_reachable,
        providers=providers,
    )

    if settings.model_strict_mode:
        ok = report.brave_reachable and all(
            item.key_present and item.reachable and item.model_valid
            for item in report.providers.values()
        )
    else:
        ok = report.brave_reachable and any(
            item.key_present and item.reachable and item.model_valid
            for item in report.providers.values()
        )

    return report, ok
