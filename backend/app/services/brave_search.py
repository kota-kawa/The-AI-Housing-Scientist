from __future__ import annotations

import json
from typing import Any

import httpx

from app.llm.base import LLMAdapter


def _rewrite_query_for_brave(adapter: LLMAdapter, query: str) -> str:
    try:
        result = adapter.generate_text(
            system=(
                "Convert the Japanese rental search query into space-separated keywords "
                "optimized for Brave Search. Extract: area, layout (e.g. 1LDK), budget (e.g. 12万円以下), "
                "station walk (e.g. 駅徒歩7分以内), and any must-conditions. "
                "Output only the keyword string, no explanation."
            ),
            user=f"クエリ: {query}",
            temperature=0.0,
        ).strip()
        return result if result else query
    except Exception:
        return query


def _summarize_result_snippets(adapter: LLMAdapter, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items_with_snippets = [
        (i, item)
        for i, item in enumerate(results)
        if item.get("extra_snippets")
    ]
    if not items_with_snippets:
        return results

    schema = {
        "type": "object",
        "properties": {
            "summaries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "result_index": {"type": "integer"},
                        "snippet_summary": {"type": "string"},
                    },
                    "required": ["result_index", "snippet_summary"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["summaries"],
        "additionalProperties": False,
    }
    payload = {
        "results": [
            {
                "result_index": i,
                "title": str(item.get("title") or ""),
                "snippets": [str(s).strip() for s in (item.get("extra_snippets") or [])[:6] if str(s).strip()],
            }
            for i, item in items_with_snippets
        ],
        "output_rules": [
            "snippet_summary: 複数のスニペットから物件の主な特徴を1〜2文の日本語で要約",
            "家賃・間取り・駅徒歩・エリアなど具体的な数値があれば含める",
            "スニペットがない場合は空文字",
        ],
    }
    try:
        result = adapter.generate_structured(
            system=(
                "You summarize Japanese rental property search result snippets into 1-2 sentences "
                "that highlight the most useful facts for apartment hunters."
            ),
            user=json.dumps(payload, ensure_ascii=False, indent=2),
            schema=schema,
            temperature=0.0,
        )
    except Exception:
        return results

    summary_by_index: dict[int, str] = {}
    for item in result.get("summaries", []) or []:
        idx = int(item.get("result_index") or -1)
        summary = str(item.get("snippet_summary") or "").strip()
        if idx >= 0 and summary:
            summary_by_index[idx] = summary

    updated = [dict(item) for item in results]
    for i, item in items_with_snippets:
        if i in summary_by_index:
            updated[i]["snippet_summary"] = summary_by_index[i]
    return updated


class BraveSearchClient:
    def __init__(self, api_key: str, timeout_seconds: int = 20):
        self.api_key = api_key.strip()
        self.timeout_seconds = timeout_seconds
        self.base_url = "https://api.search.brave.com/res/v1/web/search"

    def search(
        self,
        query: str,
        count: int = 20,
        *,
        adapter: LLMAdapter | None = None,
    ) -> list[dict[str, Any]]:
        if not self.api_key:
            raise RuntimeError("BRAVE_SEARCH_API key is missing")

        effective_query = _rewrite_query_for_brave(adapter, query) if adapter is not None else query

        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key,
        }
        params = {
            "q": effective_query,
            "count": max(1, min(count, 20)),
            "safesearch": "moderate",
        }

        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.get(self.base_url, headers=headers, params=params)
            response.raise_for_status()
            payload = response.json()

        results = payload.get("web", {}).get("results", [])
        normalized = []
        for item in results:
            normalized.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "description": item.get("description", ""),
                    "age": item.get("age", ""),
                    "extra_snippets": item.get("extra_snippets", []),
                }
            )

        if adapter is not None:
            normalized = _summarize_result_snippets(adapter, normalized)

        return normalized
