from datetime import UTC, datetime

from app.llm.utils import build_current_date_context, with_current_date_context


def test_build_current_date_context_formats_tokyo_date_and_weekday():
    source = datetime(2026, 4, 9, 3, 15, tzinfo=UTC)

    result = build_current_date_context(source)

    assert result == "現在の日付は 2026年4月9日（木）です。"


def test_with_current_date_context_appends_context_to_system_prompt():
    source = datetime(2026, 4, 9, 12, 0, tzinfo=UTC)

    result = with_current_date_context("system prompt", source)

    assert result == "system prompt\n\n現在の日付は 2026年4月9日（木）です。"
