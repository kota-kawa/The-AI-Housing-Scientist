from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
import hashlib
import html
import json
import re
from typing import Any
import unicodedata
from urllib.parse import parse_qs, urlparse

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
# JP: 賃貸物件の家賃として妥当な最高金額（円）。これ超過は売買価格の誤抽出とみなす。
# EN: Maximum plausible monthly rent (yen). Values above this are treated as sale price misextraction.
MAX_PLAUSIBLE_RENT = 1_000_000
DETAIL_URL_POSITIVE_SEGMENTS = {
    "property",
    "properties",
    "detail",
    "details",
    "bukken",
    "room",
    "rooms",
    "heya",
    "bkdtl",
    "estate",
    "unit",
}
DETAIL_URL_NEGATIVE_SEGMENTS = {
    "search",
    "list",
    "lists",
    "feature",
    "features",
    "article",
    "articles",
    "ranking",
    "rank",
    "map",
    "area",
    "city",
    "station",
    "ensen",
    "line",
    "special",
    "theme",
    "new",
    "news",
}
DETAIL_URL_NEGATIVE_QUERY_KEYS = {
    "page",
    "sort",
    "order",
    "q",
    "query",
    "keyword",
    "area",
    "city",
    "line",
    "station",
    "pref",
    "map",
}
COLLECTION_PAGE_TOKENS = (
    "物件一覧",
    "検索結果",
    "該当物件",
    "掲載物件",
    "おすすめ物件",
    "人気物件",
    "周辺の賃貸",
    "この条件の物件",
    "エリアから探す",
    "沿線から探す",
    "こだわり条件",
)


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

    # JP: ラベル付き万パターン（例: 家賃4万 → 40000）
    # EN: Labeled 万 pattern (e.g. 家賃4万 → 40000)
    match_man = re.search(rf"{rent_label_prefix}(\d+(?:\.\d+)?)\s*万", normalized)
    if match_man:
        return int(float(match_man.group(1)) * 10000)

    # JP: ラベル付き円パターン（例: 家賃120,000円 → 120000）
    # EN: Labeled 円 pattern (e.g. 家賃120,000円 → 120000)
    match_yen = re.search(rf"{rent_label_prefix}(\d{{2,3}}(?:,\d{{3}})?)\s*円", normalized)
    if match_yen:
        return int(match_yen.group(1).replace(",", ""))

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
    match = re.search(r"徒歩\s*(?:約|およそ)?\s*(\d{1,2})\s*分", text)
    return int(match.group(1)) if match else 0


# JP: areaを抽出する。
# EN: Extract area.
def _extract_area(text: str) -> float:
    match = re.search(r"(\d{1,3}(?:\.\d+)?)\s*(?:m2|㎡|平方メートル)", text)
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


# JP: HTML断片を表示用テキストへ変換する。
# EN: Convert an HTML fragment to display text.
def _clean_html_fragment(value: str) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", " ", value or "", flags=re.IGNORECASE)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return html.unescape(re.sub(r"\s+", " ", text)).strip()


# JP: label textを正規化する。
# EN: Normalize label text.
def _normalize_label_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", _clean_html_fragment(value))
    return re.sub(r"[\s　:：/／・]+", "", normalized).strip()


# JP: table/dlからラベル付き値を抽出する。
# EN: Extract labeled values from table and definition-list markup.
def _extract_labeled_html_fields(value: str) -> dict[str, list[str]]:
    fields: dict[str, list[str]] = defaultdict(list)

    def add_pair(raw_label: str, raw_value: str) -> None:
        label = _normalize_label_text(raw_label)
        text = _clean_html_fragment(raw_value)
        if label and text and text not in fields[label]:
            fields[label].append(text)

    for row_match in re.finditer(r"<tr[^>]*>([\s\S]*?)</tr>", value or "", re.IGNORECASE):
        cells = [
            match.group(2)
            for match in re.finditer(
                r"<t([hd])[^>]*>([\s\S]*?)</t\1>",
                row_match.group(1),
                re.IGNORECASE,
            )
        ]
        if len(cells) >= 2:
            add_pair(cells[0], " ".join(cells[1:]))

    current_label = ""
    for match in re.finditer(r"<(dt|dd)[^>]*>([\s\S]*?)</\1>", value or "", re.IGNORECASE):
        tag = match.group(1).lower()
        if tag == "dt":
            current_label = match.group(2)
            continue
        if current_label:
            add_pair(current_label, match.group(2))
            current_label = ""

    return fields


# JP: ラベル候補に合う最初の値と根拠を返す。
# EN: Return the first labeled value and evidence matching label candidates.
def _first_labeled_value(
    fields: dict[str, list[str]],
    labels: tuple[str, ...],
) -> tuple[str, str]:
    for label, values in fields.items():
        if any(token in label for token in labels):
            value = values[0] if values else ""
            if value:
                return value, f"{label}: {value}"
    return "", ""


# JP: JSON-LD payloadsを抽出する。
# EN: Extract JSON-LD payloads.
def _extract_json_ld_payloads(value: str) -> list[Any]:
    payloads: list[Any] = []
    for match in re.finditer(
        r"<script[^>]+type=[\"']application/ld\+json[\"'][^>]*>([\s\S]*?)</script>",
        value or "",
        re.IGNORECASE,
    ):
        raw = html.unescape(match.group(1)).strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        payloads.append(payload)
    return payloads


# JP: JSON風構造を平坦に走査する。
# EN: Walk JSON-like structures.
def _walk_json_values(value: Any) -> list[Any]:
    values = [value]
    if isinstance(value, dict):
        for child in value.values():
            values.extend(_walk_json_values(child))
    elif isinstance(value, list):
        for child in value:
            values.extend(_walk_json_values(child))
    return values


# JP: JSON-LDからkeyに合う値を抽出する。
# EN: Extract values matching keys from JSON-LD.
def _json_ld_values(payloads: list[Any], keys: tuple[str, ...]) -> list[Any]:
    matched: list[Any] = []
    normalized_keys = {key.lower() for key in keys}
    for payload in payloads:
        for item in _walk_json_values(payload):
            if not isinstance(item, dict):
                continue
            for key, value in item.items():
                if str(key).lower() in normalized_keys:
                    matched.append(value)
    return matched


# JP: JSON-LDから最初の文字列値を抽出する。
# EN: Extract the first text value from JSON-LD.
def _json_ld_first_text(payloads: list[Any], keys: tuple[str, ...]) -> str:
    for value in _json_ld_values(payloads, keys):
        if isinstance(value, (str, int, float)) and str(value).strip():
            return str(value).strip()
    return ""


# JP: JSON-LD addressをテキスト化する。
# EN: Convert JSON-LD address data to text.
def _json_ld_address(payloads: list[Any]) -> str:
    for value in _json_ld_values(payloads, ("address",)):
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            parts = [
                str(value.get(key) or "").strip()
                for key in [
                    "addressRegion",
                    "addressLocality",
                    "streetAddress",
                    "postalCode",
                ]
            ]
            text = "".join(part for part in parts if part)
            if text:
                return text
    return ""


# JP: 金額表記を円へ変換する。
# EN: Convert a money expression to yen.
def _extract_money_amount(value: str) -> int:
    normalized = unicodedata.normalize("NFKC", str(value or "")).replace(",", "")
    match_man = re.search(r"(\d+(?:\.\d+)?)\s*万\s*(\d{1,4})?\s*円?", normalized)
    if match_man:
        amount = int(float(match_man.group(1)) * 10000)
        if match_man.group(2):
            amount += int(match_man.group(2))
        return amount
    match_yen = re.search(r"(\d{4,7})\s*円?", normalized)
    return int(match_yen.group(1)) if match_yen else 0


# JP: JSON-LDから賃料らしいpriceを抽出する。
# EN: Extract a likely rent price from JSON-LD.
def _json_ld_price(payloads: list[Any]) -> int:
    for value in _json_ld_values(payloads, ("price", "value")):
        amount = _extract_money_amount(str(value))
        if MIN_PLAUSIBLE_RENT <= amount <= MAX_PLAUSIBLE_RENT:
            return amount
    return 0


# JP: 階数情報を抽出する。
# EN: Extract floor-level data.
def _extract_floor_levels(text: str) -> tuple[int, int]:
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    for pattern in [
        r"(\d{1,2})\s*階\s*/\s*(\d{1,2})\s*階(?:建)?",
        r"(\d{1,2})\s*/\s*(\d{1,2})\s*階(?:建)?",
    ]:
        match = re.search(pattern, normalized)
        if match:
            return int(match.group(1)), int(match.group(2))

    floor = 0
    total = 0
    floor_match = re.search(r"(?:所在階|階数|所在)[^0-9]{0,8}(\d{1,2})\s*階", normalized)
    if floor_match:
        floor = int(floor_match.group(1))
    total_match = re.search(r"(\d{1,2})\s*階建", normalized)
    if total_match:
        total = int(total_match.group(1))
    if floor <= 0:
        generic_match = re.search(r"(\d{1,2})\s*階", normalized)
        if generic_match and (not total_match or generic_match.start() != total_match.start()):
            floor = int(generic_match.group(1))
    return floor, total


# JP: オートロック有無を抽出する。
# EN: Extract autolock availability.
def _extract_autolock(text: str) -> bool | None:
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    if "オートロック" not in normalized:
        return None
    return not bool(re.search(r"オートロック\s*(?:なし|無し|無)", normalized))


# JP: 契約条件の根拠文を抽出する。
# EN: Extract contract condition snippets.
def _extract_contract_terms(text: str) -> dict[str, str]:
    source = _clean_html_fragment(text)
    sentences = [item.strip() for item in re.split(r"[。.\n]", source) if item.strip()]
    terms: dict[str, str] = {}
    token_map = {
        "renewal_fee": "更新料",
        "early_termination": "短期解約",
        "notice_period": "解約予告",
        "guarantor": "保証会社",
    }
    for key, token in token_map.items():
        for sentence in sentences:
            if token in sentence:
                terms[key] = sentence[:160]
                break
    return terms


# JP: URLが単一物件ページらしいかを判定する。
# EN: Decide whether a URL looks like a single-property detail page.
def _looks_like_single_property_url(url: str) -> bool:
    parsed = urlparse(str(url or "").strip())
    path = unicodedata.normalize("NFKC", parsed.path or "").lower()
    if not parsed.scheme or not parsed.netloc or not path or path == "/":
        return False

    if parsed.netloc.endswith("mock-housing.local") and path.startswith("/properties/"):
        return True

    segments = [segment for segment in path.split("/") if segment]
    if not segments:
        return False

    segment_set = {segment for segment in segments}
    has_positive_segment = bool(segment_set & DETAIL_URL_POSITIVE_SEGMENTS)
    has_negative_segment = bool(segment_set & DETAIL_URL_NEGATIVE_SEGMENTS)
    query_keys = {key.lower() for key in parse_qs(parsed.query).keys()}
    has_negative_query = bool(query_keys & DETAIL_URL_NEGATIVE_QUERY_KEYS)

    if has_negative_segment and not has_positive_segment:
        return False
    if has_negative_query and not has_positive_segment:
        return False
    if has_positive_segment:
        return True

    last_segment = segments[-1]
    if last_segment in DETAIL_URL_NEGATIVE_SEGMENTS:
        return False
    if re.fullmatch(r"(?:page|list|search)[-_]?\d*", last_segment):
        return False
    if len(last_segment) >= 12 and re.search(r"[a-z]", last_segment):
        return True
    if len(segments) >= 2 and re.search(r"\d", last_segment):
        return True
    return False


# JP: 検索結果本文から具体物件らしさのシグナル数を数える。
# EN: Count property-specific fact signals from a search snippet.
def _search_result_fact_signal_count(item: dict[str, Any]) -> int:
    snippets = [
        str(snippet).strip()
        for snippet in item.get("extra_snippets", []) or []
        if str(snippet).strip()
    ]
    combined = " ".join(
        part
        for part in [
            str(item.get("title") or ""),
            str(item.get("description") or ""),
            *snippets[:6],
        ]
        if part
    )
    checks = [
        _extract_rent(combined) > 0,
        bool(_extract_layout(combined)),
        _extract_station_walk(combined) > 0,
        bool(_extract_address(combined)),
        _extract_area(combined) > 0,
    ]
    return sum(1 for passed in checks if passed)


# JP: 一覧ページらしいシグナル数を数える。
# EN: Count page-collection signals that indicate the page is not a single property detail.
def _collection_signal_count(item: dict[str, Any], detail_html: str) -> int:
    text = _strip_html(detail_html)
    combined = " ".join(
        part
        for part in [
            str(item.get("title") or ""),
            str(item.get("description") or ""),
            text[:5000],
        ]
        if part
    )
    signal_count = sum(1 for token in COLLECTION_PAGE_TOKENS if token in combined)
    if re.search(r"(?:該当|掲載|おすすめ)[^0-9]{0,8}\d+\s*件", combined):
        signal_count += 1
    if len(re.findall(r"(?:賃料|家賃)[^。\n]{0,18}(?:万|円)", text)) >= 3:
        signal_count += 1
    if len(re.findall(r"徒歩\s*(?:約|およそ)?\s*\d{1,2}\s*分", text)) >= 3:
        signal_count += 1
    return signal_count


# JP: HTMLから単一物件ページらしいシグナル数を数える。
# EN: Count structured signals that indicate the HTML is a single property detail page.
def _detail_page_signal_count(item: dict[str, Any], detail_html: str) -> int:
    detail_html_lower = str(detail_html or "").lower()
    if 'data-kind="property-detail"' in detail_html_lower:
        return 99

    text = _strip_html(detail_html)
    json_ld_payloads = _extract_json_ld_payloads(detail_html)
    detail_signals = [
        bool(
            _extract_html_field(detail_html, "building_name")
            or _json_ld_first_text(json_ld_payloads, ("name",))
        ),
        bool(
            _extract_html_field(detail_html, "address")
            or _json_ld_address(json_ld_payloads)
            or _extract_address(text)
        ),
        (
            _extract_money_amount(_extract_html_field(detail_html, "rent"))
            or _json_ld_price(json_ld_payloads)
            or _extract_rent(text)
        )
        > 0,
        bool(_extract_layout(_extract_html_field(detail_html, "layout") or text)),
        (_extract_area(_extract_html_field(detail_html, "area_m2") or text) > 0),
        (
            _extract_station_walk(_extract_html_field(detail_html, "station_walk_min") or text) > 0
        ),
    ]
    return sum(1 for passed in detail_signals if passed)


# JP: 検索結果が単一物件ページかを判定する。
# EN: Decide whether a search result points to one concrete property.
def is_single_property_search_result(
    item: dict[str, Any],
    detail_html: str = "",
) -> bool:
    url = str(item.get("url") or "").strip()
    snippets = [
        str(snippet).strip()
        for snippet in item.get("extra_snippets", []) or []
        if str(snippet).strip()
    ]
    combined = " ".join(
        part
        for part in [
            str(item.get("title") or ""),
            str(item.get("description") or ""),
            *snippets[:6],
        ]
        if part
    )

    if detail_html:
        detail_signal_count = _detail_page_signal_count(item, detail_html)
        collection_signal_count = _collection_signal_count(item, detail_html)
        if detail_signal_count >= 4 and collection_signal_count == 0:
            return True
        if (
            detail_signal_count >= 5
            and collection_signal_count <= 1
            and _search_result_fact_signal_count(item) >= 1
        ):
            return True
        return False

    if not _looks_like_single_property_url(url):
        return False
    if any(token in combined for token in COLLECTION_PAGE_TOKENS):
        return False
    return _search_result_fact_signal_count(item) >= 2


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

    def is_missing_field(field_name: str) -> bool:
        value = known_fields.get(field_name)
        if field_name == "has_autolock":
            return value is None
        if isinstance(value, dict):
            return not value
        return value in {None, "", 0}

    missing_fields = [
        field_name
        for field_name in [
            "rent",
            "management_fee",
            "layout",
            "station_walk_min",
            "area_m2",
            "floor_level",
            "total_floors",
            "has_autolock",
            "contract_terms",
        ]
        if is_missing_field(field_name)
    ]
    if not missing_fields:
        return {}, 0.0

    schema = {
        "type": "object",
        "properties": {
            "rent": {"type": "integer", "minimum": 0},
            "management_fee": {"type": "integer", "minimum": 0},
            "layout": {"type": "string"},
            "station_walk_min": {"type": "integer", "minimum": 0},
            "area_m2": {"type": "number", "minimum": 0},
            "floor_level": {"type": "integer", "minimum": 0},
            "total_floors": {"type": "integer", "minimum": 0},
            "has_autolock": {"type": ["boolean", "null"]},
            "contract_terms": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
            "field_evidence": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
            "extraction_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": [
            "rent",
            "management_fee",
            "layout",
            "station_walk_min",
            "area_m2",
            "floor_level",
            "total_floors",
            "has_autolock",
            "contract_terms",
            "field_evidence",
            "extraction_confidence",
        ],
        "additionalProperties": False,
    }
    payload = {
        "source_kind": source_kind,
        "missing_fields": missing_fields,
        "known_fields": {
            "rent": _coerce_positive_int(known_fields.get("rent")),
            "management_fee": _coerce_positive_int(known_fields.get("management_fee")),
            "layout": str(known_fields.get("layout") or ""),
            "station_walk_min": _coerce_positive_int(known_fields.get("station_walk_min")),
            "area_m2": _coerce_positive_float(known_fields.get("area_m2")),
            "floor_level": _coerce_positive_int(known_fields.get("floor_level")),
            "total_floors": _coerce_positive_int(known_fields.get("total_floors")),
            "has_autolock": known_fields.get("has_autolock"),
            "contract_terms": known_fields.get("contract_terms") or {},
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
            "floor_level は募集住戸の所在階のみ。建物階数と混同しない",
            "has_autolock はオートロック有無が明示されている場合だけ true/false。分からなければ null",
            "contract_terms は renewal_fee, early_termination, notice_period, guarantor など明示根拠がある項目だけ",
            "field_evidence には抽出した値ごとに短い根拠テキストを書く",
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

    management_fee = _coerce_positive_int(result.get("management_fee"))
    if not known_fields.get("management_fee") and management_fee > 0:
        supplements["management_fee"] = management_fee

    layout = _extract_layout(str(result.get("layout") or ""))
    if not known_fields.get("layout") and layout:
        supplements["layout"] = layout

    station_walk_min = _coerce_positive_int(result.get("station_walk_min"))
    if not known_fields.get("station_walk_min") and station_walk_min > 0:
        supplements["station_walk_min"] = station_walk_min

    area_m2 = _coerce_positive_float(result.get("area_m2"))
    if not known_fields.get("area_m2") and area_m2 > 0:
        supplements["area_m2"] = area_m2

    floor_level = _coerce_positive_int(result.get("floor_level"))
    if not known_fields.get("floor_level") and floor_level > 0:
        supplements["floor_level"] = floor_level

    total_floors = _coerce_positive_int(result.get("total_floors"))
    if not known_fields.get("total_floors") and total_floors > 0:
        supplements["total_floors"] = total_floors

    if known_fields.get("has_autolock") is None and result.get("has_autolock") is not None:
        supplements["has_autolock"] = bool(result.get("has_autolock"))

    contract_terms = {
        str(key).strip(): str(value).strip()
        for key, value in (result.get("contract_terms") or {}).items()
        if str(key).strip() and str(value).strip()
    }
    if not known_fields.get("contract_terms") and contract_terms:
        supplements["contract_terms"] = contract_terms

    field_evidence = {
        str(key).strip(): str(value).strip()
        for key, value in (result.get("field_evidence") or {}).items()
        if str(key).strip() and str(value).strip()
    }
    if field_evidence:
        supplements["field_evidence"] = field_evidence

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
    floor_level, total_floors = _extract_floor_levels(combined)
    has_autolock = _extract_autolock(combined)
    contract_terms = _extract_contract_terms(combined)

    # JP: 明らかに異常な家賃値は抽出失敗として扱い、LLMに再抽出させる。
    # EN: Treat implausible rent as extraction failure so LLM can re-extract.
    if rent > 0 and not (MIN_PLAUSIBLE_RENT <= rent <= MAX_PLAUSIBLE_RENT):
        rent = 0

    llm_fields, llm_confidence = _extract_property_fields_with_llm(
        adapter=adapter,
        item=item,
        source_kind="search_result_snippet",
        known_fields={
            "rent": rent,
            "management_fee": 0,
            "layout": layout,
            "station_walk_min": station_walk_min,
            "area_m2": area_m2,
            "floor_level": floor_level,
            "total_floors": total_floors,
            "has_autolock": has_autolock,
            "contract_terms": contract_terms,
        },
        text=combined,
    )

    llm_rent = int(llm_fields.get("rent") or 0)
    if llm_rent > 0 and not (MIN_PLAUSIBLE_RENT <= llm_rent <= MAX_PLAUSIBLE_RENT):
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
        management_fee=int(llm_fields.get("management_fee") or 0),
        deposit=_extract_deposit_or_key_money(combined, "敷金"),
        key_money=_extract_deposit_or_key_money(combined, "礼金"),
        station_walk_min=station_walk_min or int(llm_fields.get("station_walk_min") or 0),
        available_date="要確認",
        agency_name="要確認",
        notes=(description[:200] if description else "検索結果から抽出"),
        features=features,
        floor_level=floor_level or int(llm_fields.get("floor_level") or 0),
        total_floors=total_floors or int(llm_fields.get("total_floors") or 0),
        has_autolock=has_autolock if has_autolock is not None else llm_fields.get("has_autolock"),
        contract_terms=contract_terms or dict(llm_fields.get("contract_terms") or {}),
        field_evidence=dict(llm_fields.get("field_evidence") or {}),
        field_confidence={
            field_name: round(float(llm_confidence), 3)
            for field_name in llm_fields
            if field_name not in {"field_evidence", "contract_terms"}
        },
        extraction_source="search_result_snippet+llm" if llm_fields else "search_result_snippet",
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
    labeled_fields = _extract_labeled_html_fields(detail_html)
    json_ld_payloads = _extract_json_ld_payloads(detail_html)
    field_evidence: dict[str, str] = {}
    field_confidence: dict[str, float] = {}

    # JP: 値の由来を後段で説明できるように保持する。
    # EN: Keep lightweight evidence so downstream reports can explain values.
    def remember(field_name: str, value: Any, evidence: str, confidence: float) -> None:
        if (
            value is None
            or value == ""
            or (not isinstance(value, bool) and value == 0)
            or not str(evidence or "").strip()
        ):
            return
        field_evidence[field_name] = _compact_llm_text(str(evidence), max_chars=180)
        field_confidence[field_name] = round(confidence, 3)

    building_label, building_label_evidence = _first_labeled_value(
        labeled_fields,
        ("物件名", "建物名", "マンション名"),
    )
    building_name = (
        _extract_html_field(detail_html, "building_name")
        or building_label
        or _json_ld_first_text(json_ld_payloads, ("name",))
        or item.get("title", "").split("|")[0].strip()
    )
    property_id = _extract_html_field(detail_html, "property_id")
    seed = property_id or f"{source_id}:{item.get('url', '')}:{building_name}"
    property_id_norm = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]

    address_label, address_evidence = _first_labeled_value(
        labeled_fields,
        ("住所", "所在地", "住所地"),
    )
    layout_label, layout_evidence = _first_labeled_value(
        labeled_fields,
        ("間取り", "間取"),
    )
    area_label, area_evidence = _first_labeled_value(
        labeled_fields,
        ("専有面積", "面積"),
    )
    rent_label, rent_evidence = _first_labeled_value(
        labeled_fields,
        ("賃料", "家賃", "月額賃料"),
    )
    management_fee_label, management_fee_evidence = _first_labeled_value(
        labeled_fields,
        ("管理費", "共益費"),
    )
    deposit_label, deposit_evidence = _first_labeled_value(labeled_fields, ("敷金",))
    key_money_label, key_money_evidence = _first_labeled_value(labeled_fields, ("礼金",))
    station_walk_label, station_walk_evidence = _first_labeled_value(
        labeled_fields,
        ("交通", "アクセス", "最寄駅", "最寄り駅", "駅徒歩"),
    )
    floor_label, floor_evidence = _first_labeled_value(
        labeled_fields,
        ("所在階", "階数", "階"),
    )
    autolock_label, autolock_evidence = _first_labeled_value(
        labeled_fields,
        ("オートロック", "セキュリティ"),
    )

    address_raw = _extract_html_field(detail_html, "address")
    layout_raw = _extract_html_field(detail_html, "layout")
    area_raw = _extract_html_field(detail_html, "area_m2") or area_label
    rent_raw = _extract_html_field(detail_html, "rent") or rent_label
    management_fee_raw = _extract_html_field(detail_html, "management_fee") or management_fee_label
    deposit_raw = _extract_html_field(detail_html, "deposit") or deposit_label
    key_money_raw = _extract_html_field(detail_html, "key_money") or key_money_label
    station_walk_raw = _extract_html_field(detail_html, "station_walk_min") or station_walk_label

    address = (
        address_raw
        or address_label
        or _json_ld_address(json_ld_payloads)
        or _extract_address(text)
        or "住所要確認"
    )
    layout = (
        _extract_layout(layout_raw)
        or _extract_layout(layout_label)
        or _extract_layout(text)
        or layout_raw
        or layout_label
    )
    area_name = _extract_html_field(detail_html, "area_name") or _extract_area_name(address)

    try:
        area_m2 = float(area_raw) if area_raw else _extract_area(text)
    except ValueError:
        area_m2 = _extract_area(area_raw) or _extract_area(text)

    rent = (
        int(rent_raw)
        if str(rent_raw).isdigit()
        else _extract_money_amount(str(rent_raw))
        or _json_ld_price(json_ld_payloads)
        or _extract_rent(text)
    )
    management_fee = (
        int(management_fee_raw)
        if str(management_fee_raw).isdigit()
        else _extract_money_amount(str(management_fee_raw))
    )
    deposit = (
        int(deposit_raw)
        if str(deposit_raw).isdigit()
        else _extract_money_amount(str(deposit_raw)) or _extract_deposit_or_key_money(text, "敷金")
    )
    key_money = (
        int(key_money_raw)
        if str(key_money_raw).isdigit()
        else _extract_money_amount(str(key_money_raw))
        or _extract_deposit_or_key_money(text, "礼金")
    )
    station_walk_min = (
        int(station_walk_raw)
        if str(station_walk_raw).isdigit()
        else _extract_station_walk(station_walk_raw) or _extract_station_walk(text)
    )
    floor_level, total_floors = _extract_floor_levels(floor_label or text)
    if autolock_label:
        has_autolock = not bool(re.search(r"なし|無し|無", autolock_label))
    else:
        has_autolock = _extract_autolock(text)

    # JP: 明らかに異常な家賃値は抽出失敗として扱い、LLMに再抽出させる。
    # EN: Treat implausible rent as extraction failure so LLM can re-extract.
    if rent > 0 and not (MIN_PLAUSIBLE_RENT <= rent <= MAX_PLAUSIBLE_RENT):
        rent = 0

    llm_fields, llm_confidence = _extract_property_fields_with_llm(
        adapter=adapter,
        item=item,
        source_kind="detail_page_html",
        known_fields={
            "rent": rent,
            "management_fee": management_fee,
            "layout": layout,
            "station_walk_min": station_walk_min,
            "area_m2": area_m2,
            "floor_level": floor_level,
            "total_floors": total_floors,
            "has_autolock": has_autolock,
            "contract_terms": _extract_contract_terms(detail_html),
        },
        detail_html=detail_html,
        text=text,
    )
    llm_rent = int(llm_fields.get("rent") or 0)
    if llm_rent > 0 and not (MIN_PLAUSIBLE_RENT <= llm_rent <= MAX_PLAUSIBLE_RENT):
        llm_rent = 0
    layout = layout or str(llm_fields.get("layout") or "")
    area_m2 = area_m2 or float(llm_fields.get("area_m2") or 0.0)
    rent = rent or llm_rent
    management_fee = management_fee or int(llm_fields.get("management_fee") or 0)
    station_walk_min = station_walk_min or int(llm_fields.get("station_walk_min") or 0)
    floor_level = floor_level or int(llm_fields.get("floor_level") or 0)
    total_floors = total_floors or int(llm_fields.get("total_floors") or 0)
    if has_autolock is None and llm_fields.get("has_autolock") is not None:
        has_autolock = bool(llm_fields.get("has_autolock"))
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
    contract_terms = _extract_contract_terms(
        " ".join([contract_text, text, " ".join(features)])
    ) or dict(llm_fields.get("contract_terms") or {})

    remember(
        "building_name", building_name, building_label_evidence or "data-field: building_name", 0.95
    )
    remember("address", address, address_evidence or "data-field/json-ld/detail text: address", 0.9)
    remember("layout", layout, layout_evidence or "data-field/detail text: layout", 0.9)
    remember("area_m2", area_m2, area_evidence or "data-field/detail text: area_m2", 0.9)
    remember("rent", rent, rent_evidence or "data-field/json-ld/detail text: rent", 0.9)
    remember(
        "management_fee",
        management_fee,
        management_fee_evidence or "data-field/table: management_fee",
        0.9,
    )
    remember("deposit", deposit, deposit_evidence or "data-field/table: deposit", 0.85)
    remember("key_money", key_money, key_money_evidence or "data-field/table: key_money", 0.85)
    remember(
        "station_walk_min",
        station_walk_min,
        station_walk_evidence or "data-field/detail text: station_walk_min",
        0.9,
    )
    remember("floor_level", floor_level, floor_evidence or "table/detail text: floor_level", 0.85)
    remember("has_autolock", has_autolock, autolock_evidence or "detail text: オートロック", 0.8)
    for field_name, evidence in dict(llm_fields.get("field_evidence") or {}).items():
        if str(field_name).strip() and str(evidence).strip():
            remember(
                str(field_name),
                llm_fields.get(str(field_name), "llm"),
                str(evidence),
                llm_confidence,
            )

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
        floor_level=floor_level,
        total_floors=total_floors,
        has_autolock=has_autolock,
        contract_terms=contract_terms,
        field_evidence=field_evidence,
        field_confidence=field_confidence,
        extraction_source="detail_page_html+llm" if llm_fields else "detail_page_html",
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

        if not is_single_property_search_result(item, detail_html or ""):
            skipped_count += 1
            continue

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
