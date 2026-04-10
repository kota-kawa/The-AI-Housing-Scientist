from __future__ import annotations

from dataclasses import dataclass, field
import json
import threading
import time
from typing import Any

import httpx

from app.llm.base import LLMAdapter


@dataclass
class _TokenBucket:
    rate_per_second: float
    capacity: int
    tokens: float = field(init=False)
    updated_at: float = field(init=False)
    lock: threading.Lock = field(default_factory=threading.Lock)

    # JP: 初期化直後の補正処理を行う。
    # EN: Run post-initialization adjustments.
    def __post_init__(self) -> None:
        self.rate_per_second = max(0.1, float(self.rate_per_second))
        self.capacity = max(1, int(self.capacity))
        self.tokens = float(self.capacity)
        self.updated_at = time.monotonic()

    # JP: acquireを処理する。
    # EN: Process acquire.
    def acquire(self) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                elapsed = max(0.0, now - self.updated_at)
                if elapsed:
                    self.tokens = min(
                        float(self.capacity),
                        self.tokens + (elapsed * self.rate_per_second),
                    )
                    self.updated_at = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                wait_seconds = max(0.01, (1.0 - self.tokens) / self.rate_per_second)
            time.sleep(wait_seconds)


_BRAVE_BUCKETS: dict[tuple[str, float, int], _TokenBucket] = {}
_BRAVE_BUCKETS_LOCK = threading.Lock()


# JP: brave token bucketを取得する。
# EN: Get brave token bucket.
def _get_brave_token_bucket(
    *,
    api_key: str,
    rate_per_second: float,
    burst_size: int,
) -> _TokenBucket:
    bucket_key = (api_key.strip(), float(rate_per_second), int(burst_size))
    with _BRAVE_BUCKETS_LOCK:
        bucket = _BRAVE_BUCKETS.get(bucket_key)
        if bucket is None:
            bucket = _TokenBucket(
                rate_per_second=rate_per_second,
                capacity=burst_size,
            )
            _BRAVE_BUCKETS[bucket_key] = bucket
        return bucket


# JP: rewrite query for braveを処理する。
# EN: Process rewrite query for brave.
def _rewrite_query_for_brave(adapter: LLMAdapter, query: str) -> str:
    try:
        result = adapter.generate_text(
            system=(
                "Convert the Japanese property search query into space-separated keywords "
                "optimized for Brave Search. Extract: area, listing type (賃貸/売買), "
                "layout (e.g. 1LDK), budget, station walk (e.g. 駅徒歩7分以内), "
                "and any must-conditions. Preserve the listing type keyword as-is. "
                "Output only the keyword string, no explanation."
            ),
            user=f"クエリ: {query}",
            temperature=0.0,
        ).strip()
        return result if result else query
    except Exception:
        return query


# JP: result snippetsを要約する。
# EN: Summarize result snippets.
def _summarize_result_snippets(
    adapter: LLMAdapter, results: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    items_with_snippets = [
        (i, item) for i, item in enumerate(results) if item.get("extra_snippets")
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
                "snippets": [
                    str(s).strip() for s in (item.get("extra_snippets") or [])[:6] if str(s).strip()
                ],
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
    for i, _item in items_with_snippets:
        if i in summary_by_index:
            updated[i]["snippet_summary"] = summary_by_index[i]
    return updated


class BraveSearchClient:
    # JP: クラスやインスタンスの初期状態を設定する。
    # EN: Initialize the class or instance state.
    def __init__(
        self,
        api_key: str,
        timeout_seconds: int = 20,
        *,
        rate_per_second: float = 2.0,
        burst_size: int = 3,
    ):
        self.api_key = api_key.strip()
        self.timeout_seconds = timeout_seconds
        self.rate_per_second = max(0.1, float(rate_per_second))
        self.burst_size = max(1, int(burst_size))
        self.base_url = "https://api.search.brave.com/res/v1/web/search"

    # JP: 必要な処理を検索する。
    # EN: Search the required data.
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

        _get_brave_token_bucket(
            api_key=self.api_key,
            rate_per_second=self.rate_per_second,
            burst_size=self.burst_size,
        ).acquire()

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


class BraveImageSearchClient:
    # JP: クラスやインスタンスの初期状態を設定する。
    # EN: Initialize the class or instance state.
    def __init__(
        self,
        api_key: str,
        timeout_seconds: int = 20,
        *,
        rate_per_second: float = 2.0,
        burst_size: int = 3,
    ):
        self.api_key = api_key.strip()
        self.timeout_seconds = timeout_seconds
        self.rate_per_second = max(0.1, float(rate_per_second))
        self.burst_size = max(1, int(burst_size))
        self.base_url = "https://api.search.brave.com/res/v1/images/search"

    # JP: 必要な処理を検索する。
    # EN: Search the required data.
    def search(
        self,
        query: str,
        count: int = 8,
    ) -> list[dict[str, Any]]:
        if not self.api_key:
            raise RuntimeError("BRAVE_SEARCH_API key is missing")

        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key,
        }
        params = {
            "q": query,
            "count": max(1, min(count, 20)),
            "country": "JP",
            "search_lang": "ja",
            "safesearch": "strict",
            "spellcheck": True,
        }

        _get_brave_token_bucket(
            api_key=self.api_key,
            rate_per_second=self.rate_per_second,
            burst_size=self.burst_size,
        ).acquire()

        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.get(self.base_url, headers=headers, params=params)
            response.raise_for_status()
            payload = response.json()

        normalized: list[dict[str, Any]] = []
        for item in payload.get("results", []) or []:
            thumbnail = item.get("thumbnail", {}) or {}
            properties = item.get("properties", {}) or {}
            normalized.append(
                {
                    "title": str(item.get("title") or ""),
                    "page_url": str(item.get("url") or ""),
                    "source": str(item.get("source") or ""),
                    "image_url": str(properties.get("url") or ""),
                    "thumbnail_url": str(thumbnail.get("src") or ""),
                    "width": int(properties.get("width") or thumbnail.get("width") or 0),
                    "height": int(properties.get("height") or thumbnail.get("height") or 0),
                    "confidence": str(item.get("confidence") or ""),
                }
            )
        return normalized
