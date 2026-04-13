from __future__ import annotations

import re
import unicodedata
from typing import Any

from app.stages.search_normalize import _split_address_levels

AREA_MATCH_LEVELS = ("exact", "municipality", "nearby", "partial", "none")


def _normalize_text(value: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).strip()
    normalized = re.sub(r"[\s　]+", "", normalized)
    return normalized


def _match_text_tokens(text: str, tokens: list[str] | tuple[str, ...]) -> bool:
    normalized_text = _normalize_text(text)
    if not normalized_text:
        return False
    for token in tokens:
        normalized_token = _normalize_text(token)
        if normalized_token and normalized_token in normalized_text:
            return True
    return False


def classify_area_match(
    *,
    target_area: str,
    address: str = "",
    area_name: str = "",
    nearby_tokens: list[str] | tuple[str, ...] | None = None,
) -> dict[str, str]:
    target_text = _normalize_text(target_area)
    if not target_text:
        return {"match_level": "none", "evidence": ""}

    address_text = _normalize_text(address)
    area_text = _normalize_text(area_name)
    combined_text = " ".join(part for part in [address_text, area_text] if part).strip()
    property_levels = _split_address_levels(address or area_name)
    target_levels = _split_address_levels(target_area)

    target_municipality = _normalize_text(target_levels.get("municipality", ""))
    target_locality = _normalize_text(target_levels.get("locality", ""))
    property_municipality = _normalize_text(property_levels.get("municipality", ""))
    property_locality = _normalize_text(property_levels.get("locality", ""))

    if target_municipality and property_municipality and target_municipality == property_municipality:
        if target_locality:
            if property_locality and (
                property_locality == target_locality or target_locality in property_locality
            ):
                return {
                    "match_level": "exact",
                    "evidence": f"住所が希望エリア {target_area} と町名粒度まで一致",
                }
            if _match_text_tokens(combined_text, [target_locality]):
                return {
                    "match_level": "exact",
                    "evidence": f"希望エリア {target_area} の町名表記が候補内にある",
                }
            return {
                "match_level": "municipality",
                "evidence": f"住所が希望エリア {target_area} と同一市区町村内",
            }
        return {
            "match_level": "exact",
            "evidence": f"住所が希望エリア {target_area} と同一市区町村内",
        }

    nearby_values = [str(item).strip() for item in nearby_tokens or [] if str(item).strip()]
    if nearby_values and _match_text_tokens(combined_text, nearby_values):
        token = next(
            (value for value in nearby_values if _normalize_text(value) in combined_text),
            nearby_values[0],
        )
        return {
            "match_level": "nearby",
            "evidence": f"候補の所在地が近隣エリア {token} に一致",
        }

    if _match_text_tokens(combined_text, [target_text]):
        return {
            "match_level": "partial",
            "evidence": f"候補内に希望エリア {target_area} の表記がある",
        }

    if target_locality and _match_text_tokens(combined_text, [target_locality]):
        return {
            "match_level": "partial",
            "evidence": f"候補内に希望エリア {target_locality} の表記がある",
        }

    return {"match_level": "none", "evidence": ""}


def is_match_allowed_for_scope(match_level: str, area_scope: str) -> bool:
    if area_scope == "nearby":
        return match_level in {"exact", "municipality", "nearby"}
    return match_level in {"exact", "municipality"}
