from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_PROMPT_TIMEZONE = ZoneInfo("Asia/Tokyo")
_WEEKDAYS_JA = ("月", "火", "水", "木", "金", "土", "日")


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = _JSON_OBJECT_RE.search(text)
    if not match:
        raise ValueError("No JSON object found in model response")

    return json.loads(match.group(0))


def flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, dict):
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    chunks.append(part["text"])
                elif isinstance(part.get("content"), str):
                    chunks.append(part["content"])
        return "\n".join(chunks)
    return str(content)


def build_current_date_context(now: datetime | None = None) -> str:
    current = now.astimezone(_PROMPT_TIMEZONE) if now is not None else datetime.now(_PROMPT_TIMEZONE)
    weekday = _WEEKDAYS_JA[current.weekday()]
    return f"現在の日付は {current.year}年{current.month}月{current.day}日（{weekday}）です。"


def with_current_date_context(system: str, now: datetime | None = None) -> str:
    system = system.rstrip()
    current_date_context = build_current_date_context(now)
    if not system:
        return current_date_context
    return f"{system}\n\n{current_date_context}"
