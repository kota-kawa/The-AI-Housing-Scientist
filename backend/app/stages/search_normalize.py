from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
import hashlib
import html
import json
import re
from typing import Any
import unicodedata

from app.llm.base import LLMAdapter
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
LAYOUT_PATTERN = re.compile(r"(\d(?:SLDK|SDK|LDK|DK|K|R))", re.IGNORECASE)
RENT_CONTEXT_EXCLUSION_TOKENS = (
    "管理費",
    "共益費",
    "敷金",
    "礼金",
    "保証金",
    "更新料",
    "初期費用",
)
LLM_HTML_MAX_CHARS = 9000
LLM_EXTRACTION_CONFIDENCE_THRESHOLD = 0.35
# JP: 賃貸物件の家賃として妥当な最低金額（円）。これ未満は抽出エラーとみなす。
# EN: Minimum plausible monthly rent (yen). Values below this are treated as extraction errors.
MIN_PLAUSIBLE_RENT = 5000


# JP: textを正規化する。
# EN: Normalize text.
def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value or "").lower()
    normalized = re.sub(r"[\s　]+", "", normalized)
    return normalized.strip()


# JP: building nameを正規化する。
# EN: Normalize building name.
def _normalize_building_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value or "").lower()
    for english, japanese in BUILDING_TOKEN_ALIASES.items():
        normalized = re.sub(rf"\b{re.escape(english)}\b", japanese, normalized)
    normalized = normalized.replace("&", "and")
    normalized = re.sub(r"[()（）【】\[\]\"'`]", "", normalized)
    normalized = re.sub(r"[\s　\-_.／/]+", "", normalized)
    return normalized


# JP: addressを正規化する。
# EN: Normalize address.
def _normalize_address(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value or "").strip()
    if normalized in UNKNOWN_ADDRESS_VALUES:
        return ""
    normalized = (
        normalized.replace("丁目", "-").replace("番地", "-").replace("番", "-").replace("号", "")
    )
    normalized = re.sub(r"[\s　,/]+", "", normalized)
    normalized = re.sub(r"-+", "-", normalized)
    return normalized.lower()


# JP: address levelsを分割する。
# EN: Split address levels.
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


# JP: levenshtein ratioを処理する。
# EN: Process levenshtein ratio.
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


# JP: rentを抽出する。
# EN: Extract rent.
def _extract_rent(text: str) -> int:
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized_no_commas = normalized.replace(",", "")
    rent_label_prefix = r"(?:賃料|家賃)[^0-9]{0,6}"
    match_mixed = re.search(
        rf"{rent_label_prefix}(\d+)\s*万\s*(\d{{1,4}})\s*円?",
        normalized_no_commas,
    )
    if match_mixed:
        return int(match_mixed.group(1)) * 10000 + int(match_mixed.group(2))

    for pattern in [
        rf"{rent_label_prefix}(\d+(?:\.\d+)?)\s*万",
        rf"{rent_label_prefix}(\d{{2,3}}(?:,\d{{3}})?)\s*円",
    ]:
        match = re.search(pattern, normalized)
        if not match:
            continue
        raw = match.group(1).replace(",", "")
        if raw.isdigit():
            return int(raw)
        return int(float(raw) * 10000)

    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*万", normalized):
        window_start = max(0, match.start() - 10)
        window = normalized[window_start : match.start()]
        if any(token in window for token in RENT_CONTEXT_EXCLUSION_TOKENS):
            continue
        return int(float(match.group(1)) * 10000)

    for match in re.finditer(r"(\d{2,3}(?:,\d{3})?)\s*円", normalized):
        window_start = max(0, match.start() - 10)
        window = normalized[window_start : match.start()]
        if any(token in window for token in RENT_CONTEXT_EXCLUSION_TOKENS):
            continue
        return int(match.group(1).replace(",", ""))

    return 0


# JP: layoutを抽出する。
# EN: Extract layout.
def _extract_layout(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "").upper().replace(" ", "")
    normalized = normalized.replace("ワンルーム", "1R")
    match = LAYOUT_PATTERN.search(normalized)
    return match.group(1).upper() if match else ""


# JP: station walkを抽出する。
# EN: Extract station walk.
def _extract_station_walk(text: str) -> int:
    match = re.search(r"徒歩\s*(\d{1,2})\s*分", text)
    return int(match.group(1)) if match else 0


# JP: areaを抽出する。
# EN: Extract area.
def _extract_area(text: str) -> float:
    match = re.search(r"(\d{1,3}(?:\.\d+)?)\s*(?:m2|㎡)", text)
    return float(match.group(1)) if match else 0.0


# JP: deposit or key moneyを抽出する。
# EN: Extract deposit or key money.
def _extract_deposit_or_key_money(text: str, token: str) -> int:
    pattern = rf"{token}\s*(\d+(?:\.\d+)?)\s*万"
    match = re.search(pattern, text)
    if match:
        return int(float(match.group(1)) * 10000)
    return 0


# JP: addressを抽出する。
# EN: Extract address.
def _extract_address(text: str) -> str:
    match = ADDRESS_PATTERN.search(unicodedata.normalize("NFKC", text or ""))
    return match.group(1) if match else ""


# JP: area nameを抽出する。
# EN: Extract area name.
def _extract_area_name(address: str) -> str:
    levels = _split_address_levels(address)
    return levels["municipality"] or levels["locality"]


# JP: strip htmlを処理する。
# EN: Process strip html.
def _strip_html(value: str) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", " ", value, flags=re.IGNORECASE)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# JP: html fieldを抽出する。
# EN: Extract html field.
def _extract_html_field(value: str, field_name: str) -> str:
    pattern = rf'data-field="{re.escape(field_name)}"[^>]*>(.*?)</'
    match = re.search(pattern, value, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    text = re.sub(r"<[^>]+>", " ", match.group(1))
    return html.unescape(re.sub(r"\s+", " ", text).strip())


# JP: html JSON fieldを抽出する。
# EN: Extract html JSON field.
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


# JP: compact LLM textを処理する。
# EN: Process compact LLM text.
def _compact_llm_text(value: str, *, max_chars: int) -> str:
    text = re.sub(r"<!--[\s\S]*?-->", " ", value or "")
    text = re.sub(r"<script[\s\S]*?</script>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
    text = html.unescape(re.sub(r"\s+", " ", text)).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


# JP: coerce positive intを処理する。
# EN: Process coerce positive int.
def _coerce_positive_int(value: Any) -> int:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return 0
    return number if number > 0 else 0


# JP: coerce positive floatを処理する。
# EN: Process coerce positive float.
def _coerce_positive_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if number > 0 else 0.0


# JP: property fields with LLMを抽出する。
# EN: Extract property fields with LLM.
def _extract_property_fields_with_llm(
    *,
    adapter: LLMAdapter | None,
    item: dict[str, Any],
    source_kind: str,
    known_fields: dict[str, Any],
    detail_html: str = "",
    text: str = "",
) -> tuple[dict[str, Any], float]:
    if adapter is None:
        return {}, 0.0

    missing_fields = [
        field_name
        for field_name in ["rent", "layout", "station_walk_min", "area_m2"]
        if not known_fields.get(field_name)
    ]
    if not missing_fields:
        return {}, 0.0

    schema = {
        "type": "object",
        "properties": {
            "rent": {"type": "integer", "minimum": 0},
            "layout": {"type": "string"},
            "station_walk_min": {"type": "integer", "minimum": 0},
            "area_m2": {"type": "number", "minimum": 0},
            "extraction_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": [
            "rent",
            "layout",
            "station_walk_min",
            "area_m2",
            "extraction_confidence",
        ],
        "additionalProperties": False,
    }
    payload = {
        "source_kind": source_kind,
        "missing_fields": missing_fields,
        "known_fields": {
            "rent": _coerce_positive_int(known_fields.get("rent")),
            "layout": str(known_fields.get("layout") or ""),
            "station_walk_min": _coerce_positive_int(known_fields.get("station_walk_min")),
            "area_m2": _coerce_positive_float(known_fields.get("area_m2")),
        },
        "listing": {
            "title": str(item.get("title") or ""),
            "description": str(item.get("description") or ""),
            "extra_snippets": [
                str(snippet).strip()
                for snippet in item.get("extra_snippets", []) or []
                if str(snippet).strip()
            ][:8],
            "url": str(item.get("url") or ""),
            "detail_text_excerpt": _compact_llm_text(text, max_chars=2400),
            "detail_html_excerpt": _compact_llm_text(detail_html, max_chars=LLM_HTML_MAX_CHARS),
        },
        "output_rules": [
            "rent は月額賃料本体のみ。管理費、共益費、敷金、礼金、更新料は含めない",
            "station_walk_min は徒歩分数のみ。バス分数や距離は含めない",
            "area_m2 は専有面積のみ",
            "layout は 1R, 1K, 1DK, 1LDK, 2LDK のような表記で返す。分からなければ空文字",
            "明示されていない値は推測せず 0 または空文字にする",
        ],
    }

    try:
        result = adapter.generate_structured(
            system=(
                "You extract structured rental listing facts from Japanese property pages. "
                "Use only explicit evidence from the provided text or HTML. "
                "Never confuse management fee, deposit, or key money with monthly rent."
            ),
            user=json.dumps(payload, ensure_ascii=False, indent=2),
            schema=schema,
            temperature=0.0,
        )
    except Exception:
        return {}, 0.0

    confidence = float(result.get("extraction_confidence") or 0.0)
    if confidence < LLM_EXTRACTION_CONFIDENCE_THRESHOLD:
        return {}, confidence

    supplements: dict[str, Any] = {}
    rent = _coerce_positive_int(result.get("rent"))
    if not known_fields.get("rent") and rent > 0:
        supplements["rent"] = rent

    layout = _extract_layout(str(result.get("layout") or ""))
    if not known_fields.get("layout") and layout:
        supplements["layout"] = layout

    station_walk_min = _coerce_positive_int(result.get("station_walk_min"))
    if not known_fields.get("station_walk_min") and station_walk_min > 0:
        supplements["station_walk_min"] = station_walk_min

    area_m2 = _coerce_positive_float(result.get("area_m2"))
    if not known_fields.get("area_m2") and area_m2 > 0:
        supplements["area_m2"] = area_m2

    return supplements, confidence


# JP: fallback propertyを構築する。
# EN: Build fallback property.
def _build_fallback_property(
    source_id: str,
    item: dict[str, Any],
    *,
    adapter: LLMAdapter | None = None,
) -> PropertyNormalized:
    title = item.get("title", "")
    description = item.get("description", "")
    snippet_summary = str(item.get("snippet_summary") or "")
    combined = f"{title} {description} {snippet_summary} {' '.join(item.get('extra_snippets', []))}"
    seed = f"{source_id}:{item.get('url', '')}:{title}"
    property_id_norm = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]

    building_name = title.split("|")[0].split(" - ")[0].strip()
    if not building_name:
        building_name = "物件名不明"

    address = _extract_address(combined)
    features = [str(item).strip() for item in item.get("extra_snippets", []) if str(item).strip()]
    layout = _extract_layout(combined)
    area_m2 = _extract_area(combined)
    rent = _extract_rent(combined)
    station_walk_min = _extract_station_walk(combined)

    # JP: 明らかに異常な家賃値は抽出失敗として扱い、LLMに再抽出させる。
    # EN: Treat implausibly low rent as extraction failure so LLM can re-extract.
    if 0 < rent < MIN_PLAUSIBLE_RENT:
        rent = 0

    llm_fields, _ = _extract_property_fields_with_llm(
        adapter=adapter,
        item=item,
        source_kind="search_result_snippet",
        known_fields={
            "rent": rent,
            "layout": layout,
            "station_walk_min": station_walk_min,
            "area_m2": area_m2,
        },
        text=combined,
    )

    llm_rent = int(llm_fields.get("rent") or 0)
    if 0 < llm_rent < MIN_PLAUSIBLE_RENT:
        llm_rent = 0

    return PropertyNormalized(
        property_id_norm=property_id_norm,
        source_id=source_id,
        building_name=building_name,
        building_name_norm=_normalize_building_name(building_name),
        detail_url=item.get("url", ""),
        image_url=str(item.get("image_url") or ""),
        address=address or "住所要確認",
        address_norm=_normalize_address(address),
        area_name=_extract_area_name(address),
        nearest_station="",
        line_name="",
        layout=layout or str(llm_fields.get("layout") or ""),
        area_m2=area_m2 or float(llm_fields.get("area_m2") or 0.0),
        rent=rent or llm_rent,
        management_fee=0,
        deposit=_extract_deposit_or_key_money(combined, "敷金"),
        key_money=_extract_deposit_or_key_money(combined, "礼金"),
        station_walk_min=station_walk_min or int(llm_fields.get("station_walk_min") or 0),
        available_date="要確認",
        agency_name="要確認",
        notes=(description[:200] if description else "検索結果から抽出"),
        features=features,
    )


# JP: detail propertyを構築する。
# EN: Build detail property.
def _build_detail_property(
    source_id: str,
    item: dict[str, Any],
    detail_html: str,
    *,
    adapter: LLMAdapter | None = None,
) -> PropertyNormalized | None:
    text = _strip_html(detail_html)
    building_name = (
        _extract_html_field(detail_html, "building_name")
        or item.get("title", "").split("|")[0].strip()
    )
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
    deposit = (
        int(deposit_raw) if deposit_raw.isdigit() else _extract_deposit_or_key_money(text, "敷金")
    )
    key_money = (
        int(key_money_raw)
        if key_money_raw.isdigit()
        else _extract_deposit_or_key_money(text, "礼金")
    )
    station_walk_min = (
        int(station_walk_raw) if station_walk_raw.isdigit() else _extract_station_walk(text)
    )

    # JP: 明らかに異常な家賃値は抽出失敗として扱い、LLMに再抽出させる。
    # EN: Treat implausibly low rent as extraction failure so LLM can re-extract.
    if 0 < rent < MIN_PLAUSIBLE_RENT:
        rent = 0

    llm_fields, _ = _extract_property_fields_with_llm(
        adapter=adapter,
        item=item,
        source_kind="detail_page_html",
        known_fields={
            "rent": rent,
            "layout": layout,
            "station_walk_min": station_walk_min,
            "area_m2": area_m2,
        },
        detail_html=detail_html,
        text=text,
    )
    llm_rent = int(llm_fields.get("rent") or 0)
    if 0 < llm_rent < MIN_PLAUSIBLE_RENT:
        llm_rent = 0
    layout = layout or str(llm_fields.get("layout") or "")
    area_m2 = area_m2 or float(llm_fields.get("area_m2") or 0.0)
    rent = rent or llm_rent
    station_walk_min = station_walk_min or int(llm_fields.get("station_walk_min") or 0)
    nearest_station = _extract_html_field(detail_html, "nearest_station")
    line_name = _extract_html_field(detail_html, "line_name")
    available_date = _extract_html_field(detail_html, "available_date") or "要確認"
    agency_name = _extract_html_field(detail_html, "agency_name") or "要確認"
    notes = (
        _extract_html_field(detail_html, "notes")
        or item.get("description", "")
        or "詳細ページから抽出"
    )
    contract_text = _extract_html_field(detail_html, "contract_text")
    features = _extract_html_json_field(detail_html, "features")
    image_url = _extract_html_field(detail_html, "image_url")

    prop = PropertyNormalized(
        property_id_norm=property_id_norm,
        source_id=source_id,
        building_name=building_name or "物件名不明",
        building_name_norm=_normalize_building_name(building_name or "物件名不明"),
        detail_url=item.get("url", ""),
        image_url=image_url,
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
        notes=" / ".join(
            part for part in [notes, contract_text, " ".join(features)] if part
        ).strip(),
        features=features,
    )

    return prop if _has_structured_payload(prop) else None


# JP: structured payloadかどうかを判定する。
# EN: Check whether structured payload.
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


# JP: address similarityを処理する。
# EN: Process address similarity.
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


# JP: duplicate matchを処理する。
# EN: Process duplicate match.
def _duplicate_match(
    left: PropertyNormalized, right: PropertyNormalized
) -> tuple[float, str] | None:
    name_similarity = _levenshtein_ratio(left.building_name_norm, right.building_name_norm)
    address_similarity = _address_similarity(left.address, right.address)
    same_layout = bool(left.layout and left.layout == right.layout)
    area_gap = abs(float(left.area_m2 or 0) - float(right.area_m2 or 0))
    rent_gap = abs(int(left.rent or 0) - int(right.rent or 0))
    same_station = bool(left.nearest_station and left.nearest_station == right.nearest_station)

    if (
        address_similarity >= 0.8
        and same_layout
        and area_gap <= 1.5
        and (name_similarity >= 0.45 or same_station)
    ):
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


_LLM_DUPLICATE_CANDIDATE_LIMIT = 20


# JP: LLM verify duplicate pairsを処理する。
# EN: Process LLM verify duplicate pairs.
def _llm_verify_duplicate_pairs(
    *,
    adapter: LLMAdapter,
    candidate_pairs: list[tuple[int, int, PropertyNormalized, PropertyNormalized]],
) -> set[tuple[int, int]]:
    schema = {
        "type": "object",
        "properties": {
            "verdicts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "pair_index": {"type": "integer"},
                        "is_same_building": {"type": "boolean"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["pair_index", "is_same_building", "confidence"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["verdicts"],
        "additionalProperties": False,
    }
    pairs_payload = [
        {
            "pair_index": i,
            "property_a": {
                "building_name": left.building_name,
                "building_name_norm": left.building_name_norm,
                "address": left.address,
                "layout": left.layout,
                "area_m2": left.area_m2,
                "rent": left.rent,
                "nearest_station": left.nearest_station,
            },
            "property_b": {
                "building_name": right.building_name,
                "building_name_norm": right.building_name_norm,
                "address": right.address,
                "layout": right.layout,
                "area_m2": right.area_m2,
                "rent": right.rent,
                "nearest_station": right.nearest_station,
            },
        }
        for i, (_, _, left, right) in enumerate(candidate_pairs)
    ]
    try:
        result = adapter.generate_structured(
            system=(
                "You are a Japanese property deduplication assistant. "
                "Determine whether each pair of rental listings refers to the same physical building. "
                "Account for name variations: English/Japanese transliterations, abbreviations, and "
                "formatting differences (e.g. 'PARK COURT SHIBUYA' = 'パークコート渋谷')."
            ),
            user=json.dumps(
                {
                    "pairs": pairs_payload,
                    "output_rules": [
                        "is_same_building: true なら同一建物と判断",
                        "confidence: 0〜1。名前・住所・物件情報の一致度に基づく確信度",
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            schema=schema,
            temperature=0.0,
        )
    except Exception:
        return set()

    confirmed: set[tuple[int, int]] = set()
    for verdict in result.get("verdicts", []) or []:
        pair_idx = int(verdict.get("pair_index") or -1)
        if pair_idx < 0 or pair_idx >= len(candidate_pairs):
            continue
        if verdict.get("is_same_building") and float(verdict.get("confidence") or 0) >= 0.7:
            left_index, right_index, _, _ = candidate_pairs[pair_idx]
            confirmed.add((left_index, right_index))
    return confirmed


# JP: duplicate groupsを構築する。
# EN: Build duplicate groups.
def _build_duplicate_groups(
    properties: list[PropertyNormalized],
    *,
    adapter: LLMAdapter | None = None,
) -> list[DuplicateGroup]:
    parent = list(range(len(properties)))
    group_reasons: dict[int, list[tuple[float, str]]] = defaultdict(list)

    # JP: 必要な処理を探索する。
    # EN: Find the required data.
    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    # JP: unionを処理する。
    # EN: Process union.
    def union(left_index: int, right_index: int, confidence: float, reason: str) -> None:
        left_root = find(left_index)
        right_root = find(right_index)
        if left_root == right_root:
            group_reasons[left_root].append((confidence, reason))
            return
        parent[right_root] = left_root
        group_reasons[left_root].extend(group_reasons.pop(right_root, []))
        group_reasons[left_root].append((confidence, reason))

    llm_candidates: list[tuple[int, int, PropertyNormalized, PropertyNormalized]] = []

    for left_index, left_prop in enumerate(properties):
        for right_index in range(left_index + 1, len(properties)):
            right_prop = properties[right_index]
            match = _duplicate_match(left_prop, right_prop)
            if match is not None:
                confidence, reason = match
                union(left_index, right_index, confidence, reason)
            elif adapter is not None and len(llm_candidates) < _LLM_DUPLICATE_CANDIDATE_LIMIT:
                name_sim = _levenshtein_ratio(
                    left_prop.building_name_norm, right_prop.building_name_norm
                )
                if name_sim > 0.6:
                    llm_candidates.append((left_index, right_index, left_prop, right_prop))

    if llm_candidates:
        llm_confirmed = _llm_verify_duplicate_pairs(
            adapter=adapter,  # type: ignore[arg-type]
            candidate_pairs=llm_candidates,
        )
        for left_index, right_index in llm_confirmed:
            union(left_index, right_index, 0.72, "LLM判定: 同一建物（表記ゆれ）")

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


# JP: search and normalizeを実行する。
# EN: Run search and normalize.
def run_search_and_normalize(
    *,
    query: str,
    search_results: list[dict[str, Any]],
    detail_fetcher: Callable[[str], str | None] | None = None,
    adapter: LLMAdapter | None = None,
    image_resolver: Callable[[dict[str, Any], dict[str, Any], str], str] | None = None,
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
            prop = _build_detail_property(source_id, item, detail_html, adapter=adapter)
            if prop is not None:
                detail_parsed_count += 1

        if prop is None:
            prop = _build_fallback_property(source_id, item, adapter=adapter)
            if _has_structured_payload(prop):
                fallback_count += 1
            else:
                skipped_count += 1
                continue

        if image_resolver is not None and not str(prop.image_url or "").strip():
            try:
                resolved_image = str(
                    image_resolver(item, prop.model_dump(), detail_html or "")
                ).strip()
            except Exception:
                resolved_image = ""
            if resolved_image:
                prop.image_url = resolved_image

        properties.append(prop)

    duplicates = _build_duplicate_groups(properties, adapter=adapter)

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
