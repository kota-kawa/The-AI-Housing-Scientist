from __future__ import annotations

from datetime import date
import json
import re
from typing import Any

from app.llm.base import LLMAdapter

INTEGRITY_DIMENSIONS = (
    "freshness",
    "pricing_consistency",
    "listing_consistency",
    "evidence_completeness",
)

UNAVAILABLE_TOKENS = (
    "成約済",
    "募集終了",
    "掲載終了",
    "申込あり",
    "申込済",
    "満室",
    "空室なし",
)
REFERENCE_TOKENS = (
    "参考写真",
    "同タイプ",
    "別部屋",
    "別号室",
    "反転タイプ",
    "室内写真はイメージ",
)
# JP: 売買・購入物件を示すトークン。賃貸検索では除外対象。
# EN: Tokens indicating sale/purchase listings. Excluded when user searches for rentals.
SALE_LISTING_TOKENS = (
    "購入",
    "分譲",
    "売買",
    "売却",
    "新築一戸建て",
    "中古一戸建て",
    "中古住宅",
    "建売",
    "物件価格",
    "販売価格",
)
# JP: 賃貸物件を示すトークン。売買検索では除外対象。
# EN: Tokens indicating rental listings. Excluded when user searches for purchases.
RENTAL_LISTING_TOKENS = (
    "賃貸",
    "家賃",
    "賃料",
    "敷金",
    "礼金",
    "月額賃料",
)
LAYOUT_PATTERN = re.compile(r"(\d(?:SLDK|SDK|LDK|DK|K|R))", re.IGNORECASE)


# JP: compact textを処理する。
# EN: Process compact text.
def _compact_text(value: Any, *, max_chars: int = 320) -> str:
    text = " ".join(str(value or "").split()).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


# JP: strip htmlを処理する。
# EN: Process strip html.
def _strip_html(value: str) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", " ", value or "", flags=re.IGNORECASE)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# JP: unique stringsを処理する。
# EN: Process unique strings.
def _unique_strings(values: list[Any]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in deduped:
            deduped.append(text)
    return deduped


# JP: score to trustを処理する。
# EN: Process score to trust.
def _score_to_trust(score_map: dict[str, int], issue_count: int = 0) -> int:
    total = sum(max(1, min(5, int(score_map.get(key, 3)))) for key in INTEGRITY_DIMENSIONS)
    trust = int(round((total / (len(INTEGRITY_DIMENSIONS) * 5)) * 100))
    if issue_count > 2:
        trust = max(0, trust - min(20, (issue_count - 2) * 5))
    return trust


# JP: layout valuesを抽出する。
# EN: Extract layout values.
def _extract_layout_values(text: str) -> list[str]:
    normalized = str(text or "").upper().replace("ワンルーム", "1R")
    return _unique_strings(match.group(1).upper() for match in LAYOUT_PATTERN.finditer(normalized))


# JP: walk valuesを抽出する。
# EN: Extract walk values.
def _extract_walk_values(text: str) -> list[int]:
    values = [
        int(match.group(1)) for match in re.finditer(r"徒歩\s*(\d{1,2})\s*分", str(text or ""))
    ]
    return [value for value in values if value > 0]


# JP: yen from matchを処理する。
# EN: Process yen from match.
def _yen_from_match(match: re.Match[str]) -> int:
    man = str(match.group(1) or "").strip()
    sub = str(match.group(2) or "").strip()
    if not man:
        return 0
    if sub:
        return int(float(man)) * 10000 + int(sub)
    if "." in man:
        return int(float(man) * 10000)
    return int(man)


# JP: labeled money valuesを抽出する。
# EN: Extract labeled money values.
def _extract_labeled_money_values(text: str, labels: tuple[str, ...]) -> list[int]:
    normalized = str(text or "").replace(",", "")
    label_pattern = "(?:" + "|".join(re.escape(label) for label in labels) + ")"
    values: list[int] = []

    mixed_pattern = re.compile(
        rf"{label_pattern}[^0-9]{{0,8}}(\d+(?:\.\d+)?)\s*万\s*(\d{{1,4}})?\s*円?",
        re.IGNORECASE,
    )
    yen_pattern = re.compile(
        rf"{label_pattern}[^0-9]{{0,8}}(\d{{2,6}})\s*円",
        re.IGNORECASE,
    )

    for match in mixed_pattern.finditer(normalized):
        amount = _yen_from_match(match)
        if amount > 0:
            values.append(amount)
    for match in yen_pattern.finditer(normalized):
        amount = int(match.group(1))
        if amount > 0:
            values.append(amount)
    return _unique_numeric(values, tolerance=3000)


# JP: unique numericを処理する。
# EN: Process unique numeric.
def _unique_numeric(values: list[int], *, tolerance: int) -> list[int]:
    deduped: list[int] = []
    for value in sorted(value for value in values if value > 0):
        if not deduped or abs(deduped[-1] - value) > tolerance:
            deduped.append(value)
    return deduped


# JP: available dateを解析する。
# EN: Parse available date.
def _parse_available_date(value: str, *, today: date) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None

    match = re.search(r"(\d{4})[/-年]\s*(\d{1,2})(?:[/-月]\s*(\d{1,2}))?", text)
    if not match:
        return None

    year = int(match.group(1))
    month = int(match.group(2))
    day = int(match.group(3) or 1)
    try:
        return date(year, month, day)
    except ValueError:
        return None


# JP: evidence urlsを処理する。
# EN: Process evidence urls.
def _evidence_urls(prop: dict[str, Any], raw_result: dict[str, Any]) -> list[str]:
    return _unique_strings(
        [
            prop.get("detail_url"),
            raw_result.get("url"),
        ]
    )


# JP: evidence completeness scoreを処理する。
# EN: Process evidence completeness score.
def _evidence_completeness_score(
    prop: dict[str, Any],
    raw_result: dict[str, Any],
    detail_html: str,
) -> int:
    checks = [
        int(prop.get("rent") or 0) > 0,
        bool(prop.get("layout")),
        int(prop.get("station_walk_min") or 0) > 0,
        bool(prop.get("address") or prop.get("address_norm")),
        bool(detail_html or raw_result.get("description") or prop.get("notes")),
    ]
    passed = sum(1 for item in checks if item)
    return max(1, min(5, passed))


# JP: rule review for propertyを処理する。
# EN: Process rule review for property.
def _rule_review_for_property(
    *,
    prop: dict[str, Any],
    raw_result: dict[str, Any],
    detail_html: str,
    today: date,
    target_area: str = "",
    listing_type: str = "",
) -> dict[str, Any]:
    source_text = " ".join(
        [
            str(raw_result.get("title") or ""),
            str(raw_result.get("description") or ""),
            str(raw_result.get("snippet_summary") or ""),
            " ".join(str(item) for item in raw_result.get("extra_snippets", []) or []),
            _strip_html(detail_html),
            str(prop.get("notes") or ""),
            " ".join(str(item) for item in prop.get("features", []) or []),
        ]
    )
    inconsistencies: list[str] = []
    hard_drop = False

    # JP: 希望エリアと物件の所在地が一致しない場合は除外する。
    # EN: Drop properties that are clearly outside the target area.
    if target_area:
        area_name = str(prop.get("area_name") or "")
        address = str(prop.get("address") or "")
        area_haystack = f"{area_name} {address} {source_text}"
        if target_area not in area_haystack:
            inconsistencies.append(
                f"希望エリア「{target_area}」と物件の所在地が一致しないため除外"
            )
            hard_drop = True

    scores = {
        "freshness": 5,
        "pricing_consistency": 5,
        "listing_consistency": 5,
        "evidence_completeness": _evidence_completeness_score(prop, raw_result, detail_html),
    }

    # JP: ユーザーの要求種別に合わない物件を除外する。
    # EN: Drop listings that don't match the user's requested listing type.
    if listing_type == "賃貸":
        if any(token in source_text for token in SALE_LISTING_TOKENS):
            scores["listing_consistency"] = 1
            inconsistencies.append(
                "売買・購入物件の情報であり、賃貸物件ではないため除外"
            )
            hard_drop = True
    elif listing_type == "売買" and any(token in source_text for token in RENTAL_LISTING_TOKENS):
        scores["listing_consistency"] = 1
        inconsistencies.append(
            "賃貸物件の情報であり、売買物件ではないため除外"
        )
        hard_drop = True

    if any(token in source_text for token in UNAVAILABLE_TOKENS):
        scores["freshness"] = 1
        inconsistencies.append(
            "募集終了・成約済みを示す表記があり、現時点の募集情報として信頼しづらい"
        )
        hard_drop = True

    available_date = _parse_available_date(
        str(prop.get("available_date") or ""),
        today=today,
    )
    if available_date is not None and (today - available_date).days >= 120:
        scores["freshness"] = min(scores["freshness"], 2)
        inconsistencies.append("入居可能日の表記が古く、掲載が更新されていない可能性がある")

    if any(token in source_text for token in REFERENCE_TOKENS):
        scores["listing_consistency"] = min(scores["listing_consistency"], 3)
        inconsistencies.append(
            "参考写真・別部屋情報の可能性があり、実際の募集住戸と一致するか再確認が必要"
        )

    rent = int(prop.get("rent") or 0)
    rent_values = _extract_labeled_money_values(source_text, ("家賃", "賃料"))
    if rent <= 0:
        scores["pricing_consistency"] = min(scores["pricing_consistency"], 2)
        inconsistencies.append("家賃情報が欠落しており、比較の前提が弱い")
    elif rent_values:
        if all(abs(value - rent) > 3000 for value in rent_values):
            scores["pricing_consistency"] = 1
            inconsistencies.append("抽出済みの家賃と本文中の家賃表記が整合していない")
            hard_drop = True
        elif len(rent_values) >= 2 and max(rent_values) - min(rent_values) >= 10000:
            scores["pricing_consistency"] = min(scores["pricing_consistency"], 2)
            inconsistencies.append("家賃表記が複数あり、どの金額が最新条件か不明")
            hard_drop = True

    management_fee = int(prop.get("management_fee") or 0)
    management_values = _extract_labeled_money_values(source_text, ("管理費", "共益費"))
    if management_fee > 0 and management_values:
        if all(abs(value - management_fee) > 2000 for value in management_values):
            scores["pricing_consistency"] = min(scores["pricing_consistency"], 2)
            inconsistencies.append("管理費・共益費の表記が正規化値と食い違っている")
    elif management_fee == 0 and management_values:
        scores["pricing_consistency"] = min(scores["pricing_consistency"], 3)
        inconsistencies.append("本文には管理費表記がある一方で正規化値が0円のまま")

    layout = str(prop.get("layout") or "").upper().strip()
    layout_values = _extract_layout_values(source_text)
    if not layout:
        scores["listing_consistency"] = min(scores["listing_consistency"], 2)
        inconsistencies.append("間取り情報が欠落している")
    elif layout_values:
        if layout not in layout_values:
            scores["listing_consistency"] = 1
            inconsistencies.append("抽出済みの間取りと本文の表記が一致していない")
            hard_drop = True
        elif len(layout_values) >= 2:
            scores["listing_consistency"] = min(scores["listing_consistency"], 2)
            inconsistencies.append("本文中の間取り表記が複数あり、募集住戸が一意に読めない")

    station_walk = int(prop.get("station_walk_min") or 0)
    walk_values = _unique_numeric(_extract_walk_values(source_text), tolerance=1)
    if (
        station_walk > 0
        and walk_values
        and all(abs(value - station_walk) > 1 for value in walk_values)
        and len(walk_values) >= 1
    ):
        scores["listing_consistency"] = min(scores["listing_consistency"], 2)
        inconsistencies.append("駅徒歩分数の表記が抽出値と食い違っている")

    if scores["evidence_completeness"] <= 2:
        inconsistencies.append("比較に必要な主要項目が不足している")

    inconsistencies = _unique_strings(inconsistencies)
    trust_score = _score_to_trust(scores, len(inconsistencies))
    if hard_drop:
        trust_score = min(trust_score, 45)

    return {
        "property_id_norm": str(prop.get("property_id_norm") or ""),
        "trust_score": trust_score,
        "dimension_scores": scores,
        "inconsistencies": inconsistencies,
        "evidence_urls": _evidence_urls(prop, raw_result),
        "should_drop": hard_drop or trust_score < 55,
    }


# JP: LLM integrity reviewsを構築する。
# EN: Build LLM integrity reviews.
def _build_llm_integrity_reviews(
    *,
    normalized_properties: list[dict[str, Any]],
    raw_by_url: dict[str, dict[str, Any]],
    detail_html_map: dict[str, str],
    adapter: LLMAdapter,
) -> dict[str, dict[str, Any]]:
    schema = {
        "type": "object",
        "properties": {
            "reviews": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "property_id_norm": {"type": "string"},
                        "dimension_scores": {
                            "type": "object",
                            "properties": {
                                "freshness": {"type": "integer", "minimum": 1, "maximum": 5},
                                "pricing_consistency": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 5,
                                },
                                "listing_consistency": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 5,
                                },
                                "evidence_completeness": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 5,
                                },
                            },
                            "required": list(INTEGRITY_DIMENSIONS),
                            "additionalProperties": False,
                        },
                        "inconsistencies": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "should_drop": {"type": "boolean"},
                    },
                    "required": [
                        "property_id_norm",
                        "dimension_scores",
                        "inconsistencies",
                        "should_drop",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["reviews"],
        "additionalProperties": False,
    }

    payload = {
        "review_template": {
            "dimensions": [
                {
                    "key": "freshness",
                    "description": "掲載終了・成約済み・古い入居可能日など、情報鮮度の信頼性を1〜5で評価",
                },
                {
                    "key": "pricing_consistency",
                    "description": "家賃と管理費の表記が本文と矛盾していないかを1〜5で評価",
                },
                {
                    "key": "listing_consistency",
                    "description": "間取り・駅徒歩・参考写真表記など、掲載内容同士の整合性を1〜5で評価",
                },
                {
                    "key": "evidence_completeness",
                    "description": "比較に必要な主要項目が明示されているかを1〜5で評価",
                },
            ],
            "decision_rule": "掲載終了/成約済み、重大な数値矛盾、またはランキングに載せると誤認リスクが高い場合だけ should_drop=true",
        },
        "properties": [],
        "output_rules": [
            "explicit evidence がない内容は inconsistency に書かない",
            "inconsistencies は簡潔な日本語の配列にする",
            "scores は厳しめに採点し、曖昧なら高得点にしない",
            "should_drop は保守的に判定するが、明確な古い募集や重大矛盾は true にする",
        ],
    }

    evidence_url_map: dict[str, list[str]] = {}
    for prop in normalized_properties:
        detail_url = str(prop.get("detail_url") or "")
        raw_result = raw_by_url.get(detail_url, {})
        property_id = str(prop.get("property_id_norm") or "")
        evidence_url_map[property_id] = _evidence_urls(prop, raw_result)
        payload["properties"].append(
            {
                "property_id_norm": property_id,
                "facts": {
                    "building_name": _compact_text(prop.get("building_name"), max_chars=80),
                    "address": _compact_text(prop.get("address"), max_chars=120),
                    "rent": int(prop.get("rent") or 0),
                    "management_fee": int(prop.get("management_fee") or 0),
                    "layout": str(prop.get("layout") or ""),
                    "station_walk_min": int(prop.get("station_walk_min") or 0),
                    "available_date": str(prop.get("available_date") or ""),
                    "notes": _compact_text(prop.get("notes"), max_chars=220),
                    "features": [
                        _compact_text(item, max_chars=60)
                        for item in (prop.get("features", []) or [])[:6]
                    ],
                },
                "source_listing": {
                    "title": _compact_text(raw_result.get("title"), max_chars=120),
                    "description": _compact_text(raw_result.get("description"), max_chars=220),
                    "extra_snippets": [
                        _compact_text(item, max_chars=80)
                        for item in (raw_result.get("extra_snippets", []) or [])[:6]
                    ],
                    "detail_excerpt": _compact_text(
                        _strip_html(detail_html_map.get(detail_url, "")),
                        max_chars=1200,
                    ),
                    "evidence_urls": evidence_url_map[property_id],
                },
            }
        )

    result = adapter.generate_structured(
        system=(
            "You are a strict Japanese property data integrity reviewer. "
            "Review each property like an independent quality-control stage before ranking. "
            "Use only explicit evidence from the provided listing text and detail excerpts. "
            "Do not assume that missing facts are true."
        ),
        user=json.dumps(payload, ensure_ascii=False, indent=2),
        schema=schema,
        temperature=0.1,
    )

    reviews: dict[str, dict[str, Any]] = {}
    for item in result.get("reviews", []) or []:
        property_id = str(item.get("property_id_norm") or "").strip()
        if not property_id:
            continue
        scores = {
            key: max(1, min(5, int((item.get("dimension_scores") or {}).get(key, 3))))
            for key in INTEGRITY_DIMENSIONS
        }
        inconsistencies = _unique_strings(list(item.get("inconsistencies", []) or []))
        reviews[property_id] = {
            "property_id_norm": property_id,
            "trust_score": _score_to_trust(scores, len(inconsistencies)),
            "dimension_scores": scores,
            "inconsistencies": inconsistencies,
            "evidence_urls": evidence_url_map.get(property_id, []),
            "should_drop": bool(item.get("should_drop")),
        }
    return reviews


# JP: reviewsを結合する。
# EN: Merge reviews.
def _merge_reviews(
    rule_review: dict[str, Any],
    llm_review: dict[str, Any] | None,
) -> dict[str, Any]:
    if not llm_review:
        return rule_review

    merged_scores = {
        key: min(
            int(rule_review.get("dimension_scores", {}).get(key, 3)),
            int(llm_review.get("dimension_scores", {}).get(key, 3)),
        )
        for key in INTEGRITY_DIMENSIONS
    }
    inconsistencies = _unique_strings(
        list(rule_review.get("inconsistencies", []) or [])
        + list(llm_review.get("inconsistencies", []) or [])
    )
    trust_score = _score_to_trust(merged_scores, len(inconsistencies))
    if rule_review.get("should_drop") or llm_review.get("should_drop"):
        trust_score = min(
            trust_score,
            int(rule_review.get("trust_score") or 100),
            int(llm_review.get("trust_score") or 100),
            45,
        )

    return {
        "property_id_norm": str(
            rule_review.get("property_id_norm") or llm_review.get("property_id_norm") or ""
        ),
        "trust_score": trust_score,
        "dimension_scores": merged_scores,
        "inconsistencies": inconsistencies,
        "evidence_urls": _unique_strings(
            list(rule_review.get("evidence_urls", []) or [])
            + list(llm_review.get("evidence_urls", []) or [])
        ),
        "should_drop": bool(
            rule_review.get("should_drop") or llm_review.get("should_drop") or trust_score < 55
        ),
    }


# JP: integrity reviewを実行する。
# EN: Run integrity review.
def run_integrity_review(
    *,
    normalized_properties: list[dict[str, Any]],
    raw_results: list[dict[str, Any]] | None = None,
    detail_html_map: dict[str, str] | None = None,
    adapter: LLMAdapter | None = None,
    today: date | None = None,
    target_area: str = "",
    listing_type: str = "",
) -> dict[str, Any]:
    resolved_today = today or date.today()
    raw_by_url = {
        str(item.get("url") or ""): item
        for item in raw_results or []
        if str(item.get("url") or "").strip()
    }
    detail_lookup = dict(detail_html_map or {})

    rule_reviews_by_id: dict[str, dict[str, Any]] = {}
    for prop in normalized_properties:
        detail_url = str(prop.get("detail_url") or "")
        raw_result = raw_by_url.get(detail_url, {})
        rule_review = _rule_review_for_property(
            prop=prop,
            raw_result=raw_result,
            detail_html=str(detail_lookup.get(detail_url) or ""),
            today=resolved_today,
            target_area=target_area,
            listing_type=listing_type,
        )
        rule_reviews_by_id[rule_review["property_id_norm"]] = rule_review

    llm_reviews_by_id: dict[str, dict[str, Any]] = {}
    if adapter is not None and normalized_properties:
        try:
            llm_reviews_by_id = _build_llm_integrity_reviews(
                normalized_properties=normalized_properties,
                raw_by_url=raw_by_url,
                detail_html_map=detail_lookup,
                adapter=adapter,
            )
        except Exception:
            llm_reviews_by_id = {}

    filtered_properties: list[dict[str, Any]] = []
    dropped_properties: list[dict[str, Any]] = []
    merged_reviews: list[dict[str, Any]] = []
    dropped_property_ids: list[str] = []

    for prop in normalized_properties:
        property_id = str(prop.get("property_id_norm") or "")
        merged_review = _merge_reviews(
            rule_reviews_by_id.get(property_id, {}),
            llm_reviews_by_id.get(property_id),
        )
        prop_with_review = {**prop, "integrity_review": merged_review}
        merged_reviews.append(merged_review)
        if merged_review["should_drop"]:
            dropped_property_ids.append(property_id)
            dropped_properties.append(prop_with_review)
        else:
            filtered_properties.append(prop_with_review)

    trust_scores = [int(item.get("trust_score") or 0) for item in merged_reviews]
    avg_trust_score = round(sum(trust_scores) / len(trust_scores), 1) if trust_scores else 0.0

    return {
        "normalized_properties": filtered_properties,
        "dropped_properties": dropped_properties,
        "integrity_reviews": merged_reviews,
        "integrity_reviews_by_id": {
            str(item.get("property_id_norm") or ""): item
            for item in merged_reviews
            if str(item.get("property_id_norm") or "").strip()
        },
        "dropped_property_ids": dropped_property_ids,
        "llm_reasoning_applied": bool(llm_reviews_by_id),
        "summary": {
            "input_count": len(normalized_properties),
            "kept_count": len(filtered_properties),
            "dropped_count": len(dropped_properties),
            "average_trust_score": avg_trust_score,
            "llm_reviewed_count": len(llm_reviews_by_id),
            "integrity_input_count": len(normalized_properties),
            "integrity_kept_count": len(filtered_properties),
            "integrity_dropped_count": len(dropped_properties),
            "integrity_drop_ratio": (
                round(len(dropped_properties) / len(normalized_properties), 3)
                if normalized_properties
                else 0.0
            ),
        },
    }
