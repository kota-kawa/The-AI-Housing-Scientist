from __future__ import annotations

from typing import Any

import httpx


class BraveSearchClient:
    def __init__(self, api_key: str, timeout_seconds: int = 20):
        self.api_key = api_key.strip()
        self.timeout_seconds = timeout_seconds
        self.base_url = "https://api.search.brave.com/res/v1/web/search"

    def search(self, query: str, count: int = 20) -> list[dict[str, Any]]:
        if not self.api_key:
            raise RuntimeError("BRAVE_SEARCH_API key is missing")

        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key,
        }
        params = {
            "q": query,
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
        return normalized
