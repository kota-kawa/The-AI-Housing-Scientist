from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

ProviderName = Literal["openai", "gemini", "groq", "claude"]


TRUE_VALUES = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_env: str
    database_path: str
    llm_default_provider: ProviderName
    model_strict_mode: bool
    run_preflight_on_startup: bool
    preflight_fail_fast: bool
    llm_timeout_seconds: int
    llm_max_retries: int

    openai_api_key: str
    gemini_api_key: str
    groq_api_key: str
    claude_api_key: str
    brave_search_api_key: str

    openai_model: str
    gemini_model: str
    groq_model_primary: str
    groq_model_secondary: str
    claude_model: str


def _clean(value: str | None) -> str:
    return (value or "").strip().rstrip("\r")


@lru_cache(maxsize=1)
def _load_dotenv() -> dict[str, str]:
    candidates = [
        Path.cwd() / ".env",
        Path.cwd().parent / ".env",
    ]

    env_map: dict[str, str] = {}
    for path in candidates:
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env_map[_clean(key)] = _clean(value)
        if env_map:
            break
    return env_map


def _env(key: str, default: str = "", aliases: tuple[str, ...] = ()) -> str:
    val = _clean(os.getenv(key))
    if val:
        return val
    for alias in aliases:
        alias_val = _clean(os.getenv(alias))
        if alias_val:
            return alias_val
    file_map = _load_dotenv()
    file_val = _clean(file_map.get(key))
    if file_val:
        return file_val
    for alias in aliases:
        alias_file_val = _clean(file_map.get(alias))
        if alias_file_val:
            return alias_file_val
    return default


def _env_bool(key: str, default: bool = False) -> bool:
    raw = _clean(os.getenv(key))
    if raw == "":
        return default
    return raw.lower() in TRUE_VALUES


def _env_int(key: str, default: int) -> int:
    raw = _clean(os.getenv(key))
    if raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def load_settings() -> Settings:
    provider_raw = _env("LLM_DEFAULT_PROVIDER", "openai").lower()
    provider: ProviderName = "openai"
    if provider_raw in {"openai", "gemini", "groq", "claude"}:
        provider = provider_raw  # type: ignore[assignment]

    return Settings(
        app_env=_env("APP_ENV", "development"),
        database_path=_env("DATABASE_PATH", str(Path("data/housing_agent.db"))),
        llm_default_provider=provider,
        model_strict_mode=_env_bool("MODEL_STRICT_MODE", True),
        run_preflight_on_startup=_env_bool("RUN_PREFLIGHT_ON_STARTUP", True),
        preflight_fail_fast=_env_bool("PREFLIGHT_FAIL_FAST", False),
        llm_timeout_seconds=_env_int("LLM_TIMEOUT_SECONDS", 45),
        llm_max_retries=_env_int("LLM_MAX_RETRIES", 2),
        openai_api_key=_env("OPENAI_API_KEY"),
        gemini_api_key=_env("GEMINI_API_KEY", aliases=("Gemini_API_KEY",)),
        groq_api_key=_env("GROQ_API_KEY"),
        claude_api_key=_env("CLAUDE_API_KEY", aliases=("ANTHROPIC_API_KEY",)),
        brave_search_api_key=_env("BRAVE_SEARCH_API", aliases=("BRAVE_SEARCH_API_KEY",)),
        openai_model=_env("OPENAI_MODEL", "gpt-5.4-mini"),
        gemini_model=_env("GEMINI_MODEL", "gemini-3-flash"),
        groq_model_primary=_env("GROQ_MODEL_PRIMARY", "openai/gpt-oss-120b"),
        groq_model_secondary=_env("GROQ_MODEL_SECONDARY", "qwen/qwen3-32b"),
        claude_model=_env("CLAUDE_MODEL", "claude-sonnet-4-6"),
    )


def get_provider_model(settings: Settings, provider: ProviderName) -> str:
    if provider == "openai":
        return settings.openai_model
    if provider == "gemini":
        return settings.gemini_model
    if provider == "groq":
        return settings.groq_model_primary
    return settings.claude_model


def get_provider_api_key(settings: Settings, provider: ProviderName) -> str:
    if provider == "openai":
        return settings.openai_api_key
    if provider == "gemini":
        return settings.gemini_api_key
    if provider == "groq":
        return settings.groq_api_key
    return settings.claude_api_key
