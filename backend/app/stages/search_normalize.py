from __future__ import annotations

import hashlib
import html
import json
import re
from collections import defaultdict
from typing import Any, Callable

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


def _strip_html(value: str) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", " ", value, flags=re.IGNORECASE)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_html_field(value: str, field_name: str) -> str:
    pattern = rf'data-field="{re.escape(field_name)}"[^>]*>(.*?)</'
    match = re.search(pattern, value, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    text = re.sub(r"<[^>]+>", " ", match.group(1))
    return html.unescape(re.sub(r"\s+", " ", text).strip())


def _extract_html_json_field(value: str, field_name: str) -> list[str]:
    raw = _extract_html_field(value, field_name)
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [str(item) for item in payload if str(item).strip()]


def _build_fallback_property(source_id: str, item: dict[str, Any]) -> PropertyNormalized:
    title = item.get("title", "")
    description = item.get("description", "")
    combined = f"{title} {description} {' '.join(item.get('extra_snippets', []))}"
    seed = f"{source_id}:{item.get('url', '')}:{title}"
    property_id_norm = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]

    building_name = title.split("|")[0].split(" - ")[0].strip()
    if not building_name:
        building_name = "物件名不明"

    return PropertyNormalized(
        property_id_norm=property_id_norm,
        source_id=source_id,
        building_name=building_name,
        building_name_norm=_normalize_text(building_name),
        detail_url=item.get("url", ""),
        address="住所要確認",
        address_norm="住所要確認",
        nearest_station="",
        line_name="",
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


def _build_detail_property(
    source_id: str,
    item: dict[str, Any],
    detail_html: str,
) -> PropertyNormalized | None:
    text = _strip_html(detail_html)
    building_name = _extract_html_field(detail_html, "building_name") or item.get("title", "").split("|")[0].strip()
    property_id = _extract_html_field(detail_html, "property_id")
    seed = property_id or f"{source_id}:{item.get('url', '')}:{building_name}"
    property_id_norm = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]

    address = _extract_html_field(detail_html, "address") or "住所要確認"
    layout = _extract_html_field(detail_html, "layout") or _extract_layout(text)
    area_raw = _extract_html_field(detail_html, "area_m2")
    rent_raw = _extract_html_field(detail_html, "rent")
    management_fee_raw = _extract_html_field(detail_html, "management_fee")
    deposit_raw = _extract_html_field(detail_html, "deposit")
    key_money_raw = _extract_html_field(detail_html, "key_money")
    station_walk_raw = _extract_html_field(detail_html, "station_walk_min")

    try:
        area_m2 = float(area_raw) if area_raw else _extract_area(text)
    except ValueError:
        area_m2 = _extract_area(text)

    rent = int(rent_raw) if rent_raw.isdigit() else _extract_rent(text)
    management_fee = int(management_fee_raw) if management_fee_raw.isdigit() else 0
    deposit = int(deposit_raw) if deposit_raw.isdigit() else _extract_deposit_or_key_money(text, "敷金")
    key_money = int(key_money_raw) if key_money_raw.isdigit() else _extract_deposit_or_key_money(text, "礼金")
    station_walk_min = int(station_walk_raw) if station_walk_raw.isdigit() else _extract_station_walk(text)
    nearest_station = _extract_html_field(detail_html, "nearest_station")
    line_name = _extract_html_field(detail_html, "line_name")
    available_date = _extract_html_field(detail_html, "available_date") or "要確認"
    agency_name = _extract_html_field(detail_html, "agency_name") or "要確認"
    notes = _extract_html_field(detail_html, "notes") or item.get("description", "") or "詳細ページから抽出"
    contract_text = _extract_html_field(detail_html, "contract_text")
    features = _extract_html_json_field(detail_html, "features")

    prop = PropertyNormalized(
        property_id_norm=property_id_norm,
        source_id=source_id,
        building_name=building_name or "物件名不明",
        building_name_norm=_normalize_text(building_name or "物件名不明"),
        detail_url=item.get("url", ""),
        address=address,
        address_norm=_normalize_text(address),
        nearest_station=nearest_station,
        line_name=line_name,
        layout=layout,
        area_m2=area_m2,
        rent=rent,
        management_fee=management_fee,
        deposit=deposit,
        key_money=key_money,
        station_walk_min=station_walk_min,
        available_date=available_date,
        agency_name=agency_name,
        notes=" / ".join(part for part in [notes, contract_text, " ".join(features)] if part).strip(),
    )

    return prop if _has_structured_payload(prop) else None


def _has_structured_payload(prop: PropertyNormalized) -> bool:
    return any(
        [
            prop.rent > 0,
            bool(prop.layout),
            prop.area_m2 > 0,
            prop.station_walk_min > 0,
            prop.address != "住所要確認",
        ]
    )


def run_search_and_normalize(
    *,
    query: str,
    search_results: list[dict[str, Any]],
    detail_fetcher: Callable[[str], str | None] | None = None,
) -> dict[str, Any]:
    properties: list[PropertyNormalized] = []
    detail_parsed_count = 0
    fallback_count = 0
    skipped_count = 0

    for index, item in enumerate(search_results):
        source_name = item.get("source_name", "search")
        source_id = f"{source_name}:{index + 1}"

        detail_html = None
        if detail_fetcher is not None and item.get("url"):
            try:
                detail_html = detail_fetcher(str(item["url"]))
            except Exception:
                detail_html = None

        prop = None
        if detail_html:
            prop = _build_detail_property(source_id, item, detail_html)
            if prop is not None:
                detail_parsed_count += 1

        if prop is None:
            prop = _build_fallback_property(source_id, item)
            if _has_structured_payload(prop):
                fallback_count += 1
            else:
                skipped_count += 1
                continue

        properties.append(prop)

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
        "summary": {
            "input_result_count": len(search_results),
            "normalized_count": len(properties),
            "detail_parsed_count": detail_parsed_count,
            "fallback_count": fallback_count,
            "skipped_count": skipped_count,
        },
    }
