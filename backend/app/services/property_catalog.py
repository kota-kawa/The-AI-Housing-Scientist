from __future__ import annotations

from typing import Any

import httpx

from app.catalog import render_property_detail_html
from app.db import Database


class PropertyCatalogService:
    def __init__(self, db: Database):
        self.db = db

    def search(
        self,
        *,
        query: str,
        user_memory: dict[str, Any],
        count: int = 8,
    ) -> list[dict[str, Any]]:
        scored: list[tuple[float, dict[str, Any]]] = []
        for item in self.db.list_catalog_properties():
            score = self._score_property(item, user_memory, query)
            scored.append((score, item))

        scored.sort(
            key=lambda pair: (
                pair[0],
                -abs(int(user_memory.get("budget_max") or pair[1]["rent"]) - pair[1]["rent"]),
                -pair[1]["station_walk_min"],
            ),
            reverse=True,
        )

        normalized: list[dict[str, Any]] = []
        for _, item in scored[: max(1, count)]:
            features = item.get("features", [])
            normalized.append(
                {
                    "title": (
                        f"{item['building_name']} {item['layout']} "
                        f"{item['area_m2']}㎡ 徒歩{item['station_walk_min']}分"
                    ),
                    "url": item["detail_url"],
                    "description": (
                        f"{item['address']} / 賃料{item['rent']:,}円 管理費{item['management_fee']:,}円 / "
                        f"{item['nearest_station']} 徒歩{item['station_walk_min']}分"
                    ),
                    "extra_snippets": features[:3],
                    "source_name": "mock_catalog",
                }
            )
        return normalized

    def fetch_detail_html(self, url: str) -> str | None:
        catalog_item = self.db.get_catalog_property_by_url(url)
        if catalog_item is not None:
            return render_property_detail_html(catalog_item)

        if not url.startswith(("http://", "https://")):
            return None

        try:
            with httpx.Client(
                timeout=6.0,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (compatible; HousingScientistBot/0.1; "
                        "+https://mock-housing.local)"
                    )
                },
                follow_redirects=True,
            ) as client:
                response = client.get(url)
                response.raise_for_status()
        except Exception:
            return None

        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type:
            return None
        return response.text

    def _score_property(
        self,
        item: dict[str, Any],
        user_memory: dict[str, Any],
        query: str,
    ) -> float:
        score = 0.0
        haystack = " ".join(
            [
                str(item.get("building_name", "")),
                str(item.get("address", "")),
                str(item.get("area_name", "")),
                str(item.get("nearest_station", "")),
                str(item.get("layout", "")),
                " ".join(str(feature) for feature in item.get("features", [])),
                query,
            ]
        )

        target_area = str(user_memory.get("target_area") or "").strip()
        if target_area:
            if target_area in item["address"] or target_area in item["area_name"]:
                score += 40
            elif target_area in haystack:
                score += 20
            else:
                score -= 10

        budget_max = int(user_memory.get("budget_max") or 0)
        if budget_max > 0:
            rent = int(item["rent"])
            if rent <= budget_max:
                score += 30
            elif rent <= budget_max + 20000:
                score += 10
            else:
                score -= 20

        layout_preference = str(user_memory.get("layout_preference") or "").strip()
        if layout_preference:
            if layout_preference == item["layout"]:
                score += 15
            else:
                score -= 5

        station_walk_max = int(user_memory.get("station_walk_max") or 0)
        if station_walk_max > 0:
            if int(item["station_walk_min"]) <= station_walk_max:
                score += 12
            elif int(item["station_walk_min"]) <= station_walk_max + 3:
                score += 4
            else:
                score -= 8

        move_in_date = str(user_memory.get("move_in_date") or "").strip()
        if move_in_date and move_in_date in str(item.get("available_date", "")):
            score += 4

        for token in user_memory.get("must_conditions", []) or []:
            if token and token in haystack:
                score += 8

        for token in user_memory.get("nice_to_have", []) or []:
            if token and token in haystack:
                score += 3

        learned = user_memory.get("learned_preferences", {}) or {}
        frequent_area = str(learned.get("frequent_area") or "").strip()
        if frequent_area and frequent_area in haystack:
            score += 4

        for token in learned.get("liked_features", []) or []:
            text = str(token).strip()
            if text and text in haystack:
                score += 2

        for token in learned.get("excluded_features", []) or []:
            text = str(token).strip()
            if text and text in haystack:
                score -= 3

        return score
