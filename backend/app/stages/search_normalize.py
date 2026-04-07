from __future__ import annotations

import hashlib
import html
import json
import re
import unicodedata
from collections import defaultdict
from typing import Any, Callable

from app.models import DuplicateGroup, PropertyNormalized


UNKNOWN_ADDRESS_VALUES = {"", "住所要確認", "要確認"}
BUILDING_TOKEN_ALIASES = {
    "park": "パーク",
    "heights": "ハイツ",
    "court": "コート",
    "terrace": "テラス",
    "residence": "レジデンス",
    "garden": "ガーデン",
    "breeze": "ブリーズ",
    "bay": "ベイ",
    "front": "フロント",
    "loft": "ロフト",
    "suite": "スイート",
    "south": "サウス",
    "north": "ノース",
    "west": "ウエスト",
    "east": "イースト",
    "river": "リバー",
    "works": "ワークス",
}
PREFECTURE_PATTERN = re.compile(r"^(東京都|北海道|(?:京都|大阪)府|.{2,3}県)")
MUNICIPALITY_PATTERN = re.compile(r"^(.+?(?:市|区|町|村))")
ADDRESS_PATTERN = re.compile(
    r"((?:東京都|北海道|(?:京都|大阪)府|.{2,3}県)?"
    r"[^\s|/]{1,24}(?:区|市|町|村)[^\s|/]{0,24}\d[^\s|/]{0,16})"
)


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value or "").lower()
    normalized = re.sub(r"[\s　]+", "", normalized)
    return normalized.strip()


def _normalize_building_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value or "").lower()
    for english, japanese in BUILDING_TOKEN_ALIASES.items():
        normalized = re.sub(rf"\b{re.escape(english)}\b", japanese, normalized)
    normalized = normalized.replace("&", "and")
    normalized = re.sub(r"[()（）【】\[\]\"'`]", "", normalized)
    normalized = re.sub(r"[\s　\-_.／/]+", "", normalized)
    return normalized


def _normalize_address(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value or "").strip()
    if normalized in UNKNOWN_ADDRESS_VALUES:
        return ""
    normalized = normalized.replace("丁目", "-").replace("番地", "-").replace("番", "-").replace("号", "")
    normalized = re.sub(r"[\s　,/]+", "", normalized)
    normalized = re.sub(r"-+", "-", normalized)
    return normalized.lower()


def _split_address_levels(value: str) -> dict[str, str]:
    normalized = _normalize_address(value)
    if not normalized:
        return {"prefecture": "", "municipality": "", "locality": "", "block": ""}

    prefecture = ""
    municipality = ""
    remainder = normalized

    prefecture_match = PREFECTURE_PATTERN.match(normalized)
    if prefecture_match:
        prefecture = prefecture_match.group(1)
        remainder = normalized[len(prefecture) :]

    municipality_match = MUNICIPALITY_PATTERN.match(remainder)
    if municipality_match:
        municipality = municipality_match.group(1)
        remainder = remainder[len(municipality) :]

    digit_match = re.search(r"\d", remainder)
    if digit_match:
        locality = remainder[: digit_match.start()]
        block = remainder[digit_match.start() :]
    else:
        locality = remainder
        block = ""

    return {
        "prefecture": prefecture,
        "municipality": municipality,
        "locality": locality,
        "block": block,
    }


def _levenshtein_ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0

    prev = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        curr = [i]
        for j, right_char in enumerate(right, start=1):
            insert_cost = curr[j - 1] + 1
            delete_cost = prev[j] + 1
            replace_cost = prev[j - 1] + (0 if left_char == right_char else 1)
            curr.append(min(insert_cost, delete_cost, replace_cost))
        prev = curr

    distance = prev[-1]
    return round(1 - (distance / max(len(left), len(right))), 3)


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


def _extract_address(text: str) -> str:
    match = ADDRESS_PATTERN.search(unicodedata.normalize("NFKC", text or ""))
    return match.group(1) if match else ""


def _extract_area_name(address: str) -> str:
    levels = _split_address_levels(address)
    return levels["municipality"] or levels["locality"]


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

    address = _extract_address(combined)
    features = [str(item).strip() for item in item.get("extra_snippets", []) if str(item).strip()]

    return PropertyNormalized(
        property_id_norm=property_id_norm,
        source_id=source_id,
        building_name=building_name,
        building_name_norm=_normalize_building_name(building_name),
        detail_url=item.get("url", ""),
        address=address or "住所要確認",
        address_norm=_normalize_address(address),
        area_name=_extract_area_name(address),
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
        features=features,
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

    address = _extract_html_field(detail_html, "address") or _extract_address(text) or "住所要確認"
    layout = _extract_html_field(detail_html, "layout") or _extract_layout(text)
    area_raw = _extract_html_field(detail_html, "area_m2")
    rent_raw = _extract_html_field(detail_html, "rent")
    management_fee_raw = _extract_html_field(detail_html, "management_fee")
    deposit_raw = _extract_html_field(detail_html, "deposit")
    key_money_raw = _extract_html_field(detail_html, "key_money")
    station_walk_raw = _extract_html_field(detail_html, "station_walk_min")
    area_name = _extract_html_field(detail_html, "area_name") or _extract_area_name(address)

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
        building_name_norm=_normalize_building_name(building_name or "物件名不明"),
        detail_url=item.get("url", ""),
        address=address,
        address_norm=_normalize_address(address),
        area_name=area_name,
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
        features=features,
    )

    return prop if _has_structured_payload(prop) else None


def _has_structured_payload(prop: PropertyNormalized) -> bool:
    return any(
        [
            prop.rent > 0,
            bool(prop.layout),
            prop.area_m2 > 0,
            prop.station_walk_min > 0,
            bool(prop.address_norm),
        ]
    )


def _address_similarity(left: str, right: str) -> float:
    left_levels = _split_address_levels(left)
    right_levels = _split_address_levels(right)
    if not left_levels["municipality"] or not right_levels["municipality"]:
        return 0.0

    score = 0.0
    if left_levels["prefecture"] and left_levels["prefecture"] == right_levels["prefecture"]:
        score += 0.2
    if left_levels["municipality"] == right_levels["municipality"]:
        score += 0.35
    if left_levels["locality"] and left_levels["locality"] == right_levels["locality"]:
        score += 0.25
    if left_levels["block"] and right_levels["block"]:
        if left_levels["block"] == right_levels["block"]:
            score += 0.2
        elif left_levels["block"].split("-")[0] == right_levels["block"].split("-")[0]:
            score += 0.1
    return round(min(score, 1.0), 3)


def _duplicate_match(left: PropertyNormalized, right: PropertyNormalized) -> tuple[float, str] | None:
    name_similarity = _levenshtein_ratio(left.building_name_norm, right.building_name_norm)
    address_similarity = _address_similarity(left.address, right.address)
    same_layout = bool(left.layout and left.layout == right.layout)
    area_gap = abs(float(left.area_m2 or 0) - float(right.area_m2 or 0))
    rent_gap = abs(int(left.rent or 0) - int(right.rent or 0))
    same_station = bool(left.nearest_station and left.nearest_station == right.nearest_station)

    if address_similarity >= 0.8 and same_layout and area_gap <= 1.5 and (name_similarity >= 0.45 or same_station):
        confidence = round(min(1.0, 0.55 + address_similarity * 0.3 + name_similarity * 0.15), 3)
        return confidence, "住所階層が一致し、間取り・面積も近い"

    if address_similarity >= 0.65 and name_similarity >= 0.82 and same_layout and area_gap <= 1.0:
        confidence = round(min(1.0, 0.45 + address_similarity * 0.25 + name_similarity * 0.3), 3)
        return confidence, "建物名の表記ゆれが近く、住所もほぼ一致"

    if (
        not left.address_norm
        and not right.address_norm
        and name_similarity >= 0.96
        and same_layout
        and area_gap <= 0.5
        and rent_gap <= 5000
        and same_station
    ):
        confidence = round(min(1.0, 0.35 + name_similarity * 0.35), 3)
        return confidence, "住所欠落だが建物名・面積・家賃・駅情報が極めて近い"

    return None


def _build_duplicate_groups(properties: list[PropertyNormalized]) -> list[DuplicateGroup]:
    parent = list(range(len(properties)))
    group_reasons: dict[int, list[tuple[float, str]]] = defaultdict(list)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left_index: int, right_index: int, confidence: float, reason: str) -> None:
        left_root = find(left_index)
        right_root = find(right_index)
        if left_root == right_root:
            group_reasons[left_root].append((confidence, reason))
            return
        parent[right_root] = left_root
        group_reasons[left_root].extend(group_reasons.pop(right_root, []))
        group_reasons[left_root].append((confidence, reason))

    for left_index, left_prop in enumerate(properties):
        for right_index in range(left_index + 1, len(properties)):
            right_prop = properties[right_index]
            match = _duplicate_match(left_prop, right_prop)
            if match is None:
                continue
            confidence, reason = match
            union(left_index, right_index, confidence, reason)

    grouped: dict[int, list[PropertyNormalized]] = defaultdict(list)
    for index, prop in enumerate(properties):
        grouped[find(index)].append(prop)

    duplicates: list[DuplicateGroup] = []
    for root, members in grouped.items():
        if len(members) <= 1:
            continue
        reasons = group_reasons.get(root, [])
        best_confidence, best_reason = max(reasons, default=(0.0, ""))
        members_sorted = sorted(members, key=lambda item: item.property_id_norm)
        representative = members_sorted[0]
        duplicates.append(
            DuplicateGroup(
                key=f"{representative.building_name_norm}|{representative.address_norm or 'address-missing'}",
                property_ids=[item.property_id_norm for item in members_sorted],
                confidence=best_confidence,
                reason=best_reason,
            )
        )

    duplicates.sort(key=lambda item: (-item.confidence, item.key))
    return duplicates


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

    duplicates = _build_duplicate_groups(properties)

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
            "duplicate_group_count": len(duplicates),
        },
    }
