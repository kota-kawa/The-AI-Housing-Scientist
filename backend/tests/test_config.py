from app.config import load_settings


def test_load_settings_alias_and_crlf(monkeypatch):
    monkeypatch.setenv("Gemini_API_KEY", "dummy-gemini\r")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key\r")
    monkeypatch.setenv("CLAUDE_API_KEY", "claude-key\r")
    monkeypatch.setenv("GROQ_API_KEY", "groq-key\r")
    monkeypatch.setenv("BRAVE_SEARCH_API", "brave-key\r")

    settings = load_settings()

    assert settings.gemini_api_key == "dummy-gemini"
    assert settings.openai_api_key == "openai-key"
    assert settings.brave_search_api_key == "brave-key"
    assert settings.model_strict_mode is True
    assert settings.research_tree_min_nodes_before_stable_stop == 8
