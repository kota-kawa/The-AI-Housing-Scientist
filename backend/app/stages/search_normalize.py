from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Any

from app.models import DuplicateGroup, PropertyNormalized


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", "", value).lower()


def _extract_rent(text: str) -> int:
    match_man = re.search(r"(\d+(?:\.\d+)?)\s*万", text)
    if match_man:
        return int(float(match_man.group(1)) * 10000)

    match_yen = re.search(r"(\d{2,3})(?:,|，)?(\d{3})\s*円", text)
    if match_yen:
        return int(f"{match_yen.group(1)}{match_yen.group(2)}")

    return 0


def _extract_layout(text: str) -> str:
    match = re.search(r"(1R|1K|1DK|1LDK|2K|2DK|2LDK|3LDK|4LDK)", text)
    return match.group(1) if match else ""


def _extract_station_walk(text: str) -> int:
    match = re.search(r"徒歩\s*(\d{1,2})\s*分", text)
    return int(match.group(1)) if match else 0


def _extract_area(text: str) -> float:
    match = re.search(r"(\d{1,3}(?:\.\d+)?)\s*(?:m2|㎡)", text)
    return float(match.group(1)) if match else 0.0


def _extract_deposit_or_key_money(text: str, token: str) -> int:
    pattern = rf"{token}\s*(\d+(?:\.\d+)?)\s*万"
    match = re.search(pattern, text)
    if match:
        return int(float(match.group(1)) * 10000)
    return 0


def _build_property(source_id: str, item: dict[str, Any]) -> PropertyNormalized:
    title = item.get("title", "")
    description = item.get("description", "")
    combined = f"{title} {description} {' '.join(item.get('extra_snippets', []))}"
    seed = f"{source_id}:{item.get('url','')}:{title}"
    property_id_norm = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]

    building_name = title.split("|")[0].split(" - ")[0].strip()
    if not building_name:
        building_name = "物件名不明"

    return PropertyNormalized(
        property_id_norm=property_id_norm,
        source_id=source_id,
        building_name_norm=_normalize_text(building_name),
        address_norm="住所要確認",
        layout=_extract_layout(combined),
        area_m2=_extract_area(combined),
        rent=_extract_rent(combined),
        management_fee=0,
        deposit=_extract_deposit_or_key_money(combined, "敷金"),
        key_money=_extract_deposit_or_key_money(combined, "礼金"),
        station_walk_min=_extract_station_walk(combined),
        available_date="要確認",
        agency_name="要確認",
        notes=(description[:200] if description else "検索結果から抽出"),
    )


def run_search_and_normalize(
    *,
    query: str,
    search_results: list[dict[str, Any]],
) -> dict[str, Any]:
    properties: list[PropertyNormalized] = []
    for index, item in enumerate(search_results):
        source_id = f"brave:{index + 1}"
        properties.append(_build_property(source_id, item))

    grouped: dict[str, list[str]] = defaultdict(list)
    for prop in properties:
        key = (
            f"{prop.building_name_norm}|{prop.address_norm}|"
            f"{prop.layout}|{round(prop.area_m2, 1)}"
        )
        grouped[key].append(prop.property_id_norm)

    duplicates = [
        DuplicateGroup(key=key, property_ids=ids)
        for key, ids in grouped.items()
        if len(ids) > 1
    ]

    return {
        "query": query,
        "normalized_properties": [p.model_dump() for p in properties],
        "duplicate_groups": [d.model_dump() for d in duplicates],
    }
