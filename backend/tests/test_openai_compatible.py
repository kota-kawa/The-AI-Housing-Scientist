import re

from app.llm.openai_compatible import OpenAICompatibleAdapter


def test_generate_text_adds_hidden_reasoning_for_groq_qwen(monkeypatch):
    adapter = OpenAICompatibleAdapter(
        provider_name="groq",
        base_url="https://api.groq.com/openai/v1",
        api_key="test-key",
        model="qwen/qwen3-32b",
        timeout_seconds=10,
        max_retries=0,
    )
    captured: dict[str, object] = {}

    def fake_request(
        method: str, path: str, payload: dict[str, object] | None = None
    ) -> dict[str, object]:
        captured["method"] = method
        captured["path"] = path
        captured["payload"] = payload or {}
        return {
            "choices": [
                {
                    "message": {
                        "content": "ok",
                    }
                }
            ],
            "usage": {},
        }

    monkeypatch.setattr(adapter, "_request", fake_request)

    result = adapter.generate_text(system="system", user="user")

    assert result == "ok"
    assert captured["method"] == "POST"
    assert captured["path"] == "/chat/completions"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "qwen/qwen3-32b"
    assert payload["messages"][1] == {"role": "user", "content": "user"}
    assert payload["temperature"] == 0.2
    assert payload["reasoning_format"] == "hidden"
    system_message = payload["messages"][0]
    assert isinstance(system_message, dict)
    assert system_message["role"] == "system"
    assert system_message["content"].startswith("system\n\n現在の日付は ")
    assert re.search(r"\d{4}年\d{1,2}月\d{1,2}日（[月火水木金土日]）", system_message["content"])


def test_generate_structured_adds_hidden_reasoning_for_groq_qwen(monkeypatch):
    adapter = OpenAICompatibleAdapter(
        provider_name="groq",
        base_url="https://api.groq.com/openai/v1",
        api_key="test-key",
        model="qwen/qwen3-32b",
        timeout_seconds=10,
        max_retries=0,
    )
    captured: dict[str, object] = {}

    def fake_request(
        method: str, path: str, payload: dict[str, object] | None = None
    ) -> dict[str, object]:
        captured["method"] = method
        captured["path"] = path
        captured["payload"] = payload or {}
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"answer":"ok"}',
                    }
                }
            ],
            "usage": {},
        }

    monkeypatch.setattr(adapter, "_request", fake_request)

    result = adapter.generate_structured(
        system="system",
        user="user",
        schema={
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
            },
            "required": ["answer"],
            "additionalProperties": False,
        },
    )

    assert result == {"answer": "ok"}
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["reasoning_format"] == "hidden"
    assert payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "structured_response",
            "schema": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                },
                "required": ["answer"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }


def test_generate_text_does_not_add_hidden_reasoning_for_other_models(monkeypatch):
    adapter = OpenAICompatibleAdapter(
        provider_name="groq",
        base_url="https://api.groq.com/openai/v1",
        api_key="test-key",
        model="openai/gpt-oss-120b",
        timeout_seconds=10,
        max_retries=0,
    )
    captured: dict[str, object] = {}

    def fake_request(
        method: str, path: str, payload: dict[str, object] | None = None
    ) -> dict[str, object]:
        captured["payload"] = payload or {}
        return {
            "choices": [
                {
                    "message": {
                        "content": "ok",
                    }
                }
            ],
            "usage": {},
        }

    monkeypatch.setattr(adapter, "_request", fake_request)

    result = adapter.generate_text(system="system", user="user")

    assert result == "ok"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert "reasoning_format" not in payload
    system_message = payload["messages"][0]
    assert isinstance(system_message, dict)
    assert system_message["role"] == "system"
    assert system_message["content"].startswith("system\n\n現在の日付は ")
    assert re.search(r"\d{4}年\d{1,2}月\d{1,2}日（[月火水木金土日]）", system_message["content"])
