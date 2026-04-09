from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urljoin, urlparse

from app.llm.base import LLMAdapter

from .brave_search import BraveImageSearchClient


IMAGE_BLACKLIST_TOKENS = (
    "logo",
    "icon",
    "sprite",
    "avatar",
    "banner",
    "advert",
    "placeholder",
    "floorplan",
    "floor-plan",
    "plan",
    "map",
    "staticmap",
    "googleapis.com/maps",
    "間取",
    "間取り",
    "図面",
    "地図",
)

IMAGE_POSITIVE_TOKENS = (
    "外観",
    "内観",
    "室内",
    "リビング",
    "room",
    "living",
    "apartment",
    "residence",
    "building",
    "property",
    "マンション",
    "レジデンス",
    "物件",
)


def _compact_text(value: Any, *, max_chars: int = 220) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _strip_html(value: str) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", " ", value or "", flags=re.IGNORECASE)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _parse_tag_attributes(tag: str) -> dict[str, str]:
    attributes: dict[str, str] = {}
    for key, value in re.findall(r'([a-zA-Z0-9:_-]+)\s*=\s*["\']([^"\']+)["\']', tag):
        attributes[key.lower()] = value.strip()
    return attributes


def _normalize_candidate_url(raw_url: str, page_url: str) -> str:
    candidate = str(raw_url or "").strip()
    if not candidate:
        return ""
    if candidate.startswith(("data:", "blob:")):
        return ""
    if candidate.startswith("//"):
        candidate = f"https:{candidate}"
    elif candidate.startswith("/"):
        candidate = urljoin(page_url, candidate) if page_url else ""
    elif not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", candidate):
        candidate = urljoin(page_url, candidate) if page_url else ""
    if not candidate.startswith(("http://", "https://")):
        return ""
    if candidate.lower().endswith(".svg"):
        return ""
    return candidate


def _contains_blacklist_token(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in IMAGE_BLACKLIST_TOKENS)


def _extract_json_ld_image_candidates(detail_html: str, page_url: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for raw_payload in re.findall(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>([\s\S]*?)</script>',
        detail_html or "",
        flags=re.IGNORECASE,
    ):
        try:
            payload = json.loads(raw_payload)
        except json.JSONDecodeError:
            continue
        stack = [payload]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                for key, value in item.items():
                    if key in {"image", "thumbnailUrl", "contentUrl"}:
                        values = value if isinstance(value, list) else [value]
                        for candidate in values:
                            if isinstance(candidate, str):
                                resolved = _normalize_candidate_url(candidate, page_url)
                                if resolved:
                                    candidates.append(
                                        {
                                            "display_url": resolved,
                                            "image_url": resolved,
                                            "page_url": page_url,
                                            "source_kind": "json_ld",
                                            "alt": "",
                                            "title": "",
                                            "context": "",
                                            "width": 0,
                                            "height": 0,
                                        }
                                    )
                    elif isinstance(value, (dict, list)):
                        stack.append(value)
            elif isinstance(item, list):
                stack.extend(item)
    return candidates


def _extract_html_image_candidates(detail_html: str, page_url: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    for field_name in ["image_url", "image", "photo_url"]:
        pattern = rf'data-field=["\']{re.escape(field_name)}["\'][^>]*>(.*?)</'
        for raw_value in re.findall(pattern, detail_html or "", flags=re.IGNORECASE | re.DOTALL):
            candidate_url = _normalize_candidate_url(_strip_html(raw_value), page_url)
            if candidate_url:
                candidates.append(
                    {
                        "display_url": candidate_url,
                        "image_url": candidate_url,
                        "page_url": page_url,
                        "source_kind": "data_field",
                        "alt": "",
                        "title": "",
                        "context": "",
                        "width": 0,
                        "height": 0,
                    }
                )

    for raw_value in re.findall(
        r'<meta[^>]+(?:property|name)=["\'](?:og:image(?::url)?|twitter:image(?::src)?)["\'][^>]+content=["\']([^"\']+)["\']',
        detail_html or "",
        flags=re.IGNORECASE,
    ):
        candidate_url = _normalize_candidate_url(raw_value, page_url)
        if candidate_url:
            candidates.append(
                {
                    "display_url": candidate_url,
                    "image_url": candidate_url,
                    "page_url": page_url,
                    "source_kind": "meta_image",
                    "alt": "",
                    "title": "",
                    "context": "",
                    "width": 0,
                    "height": 0,
                }
            )

    for tag in re.findall(r"<img\b[\s\S]*?>", detail_html or "", flags=re.IGNORECASE):
        attrs = _parse_tag_attributes(tag)
        raw_url = (
            attrs.get("src")
            or attrs.get("data-src")
            or attrs.get("data-original")
            or attrs.get("data-lazy-src")
            or attrs.get("data-url")
            or ""
        )
        if not raw_url and attrs.get("srcset"):
            raw_url = attrs["srcset"].split(",")[0].strip().split(" ")[0]
        candidate_url = _normalize_candidate_url(raw_url, page_url)
        if not candidate_url:
            continue
        context_match = re.search(re.escape(tag), detail_html or "", flags=re.IGNORECASE)
        context = ""
        if context_match:
            start = max(0, context_match.start() - 180)
            end = min(len(detail_html or ""), context_match.end() + 180)
            context = _compact_text(_strip_html((detail_html or "")[start:end]), max_chars=180)
        candidates.append(
            {
                "display_url": candidate_url,
                "image_url": candidate_url,
                "page_url": page_url,
                "source_kind": "html_img",
                "alt": _compact_text(attrs.get("alt") or "", max_chars=120),
                "title": _compact_text(attrs.get("title") or "", max_chars=120),
                "context": context,
                "width": int(attrs.get("width") or 0) if str(attrs.get("width") or "").isdigit() else 0,
                "height": int(attrs.get("height") or 0) if str(attrs.get("height") or "").isdigit() else 0,
            }
        )

    candidates.extend(_extract_json_ld_image_candidates(detail_html, page_url))
    return candidates


def _dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for candidate in candidates:
        display_url = str(candidate.get("display_url") or candidate.get("image_url") or "").strip()
        if not display_url or display_url in seen_urls:
            continue
        combined_text = " ".join(
            [
                display_url,
                str(candidate.get("alt") or ""),
                str(candidate.get("title") or ""),
                str(candidate.get("context") or ""),
            ]
        )
        if _contains_blacklist_token(combined_text):
            continue
        seen_urls.add(display_url)
        deduped.append(candidate)
    return deduped


def _heuristic_score_candidate(
    *,
    candidate: dict[str, Any],
    property_data: dict[str, Any],
    page_url: str,
) -> float:
    score = 0.0
    source_kind = str(candidate.get("source_kind") or "")
    display_url = str(candidate.get("display_url") or candidate.get("image_url") or "")
    alt = str(candidate.get("alt") or "")
    title = str(candidate.get("title") or "")
    context = str(candidate.get("context") or "")
    host = urlparse(display_url).hostname or ""
    page_host = urlparse(page_url).hostname or ""

    if source_kind == "data_field":
        score += 8.0
    elif source_kind == "meta_image":
        score += 6.0
    elif source_kind == "json_ld":
        score += 5.0
    elif source_kind == "html_img":
        score += 4.0
    elif source_kind == "brave_image":
        score += 3.0

    if page_host and host and host == page_host:
        score += 2.0

    width = int(candidate.get("width") or 0)
    height = int(candidate.get("height") or 0)
    if width >= 400 and height >= 300:
        score += 1.5
    elif width >= 200 and height >= 150:
        score += 0.75

    combined = " ".join(
        [
            display_url,
            alt,
            title,
            context,
            str(candidate.get("source") or ""),
        ]
    ).lower()
    if any(token in combined for token in IMAGE_POSITIVE_TOKENS):
        score += 1.2

    for token in [
        str(property_data.get("building_name") or "").strip().lower(),
        str(property_data.get("nearest_station") or "").strip().lower(),
        str(property_data.get("area_name") or "").strip().lower(),
    ]:
        if token and token in combined:
            score += 0.9

    confidence = str(candidate.get("confidence") or "").lower()
    if confidence == "high":
        score += 0.8
    elif confidence == "medium":
        score += 0.4

    if _contains_blacklist_token(combined):
        score -= 6.0

    return round(score, 3)


class PropertyImageResolver:
    def __init__(
        self,
        *,
        brave_api_key: str = "",
        timeout_seconds: int = 20,
        image_search_client: BraveImageSearchClient | None = None,
    ):
        self.image_search_client = image_search_client or (
            BraveImageSearchClient(brave_api_key, timeout_seconds=timeout_seconds)
            if str(brave_api_key).strip()
            else None
        )
        self._resolution_cache: dict[str, str] = {}
        self._image_query_cache: dict[str, list[dict[str, Any]]] = {}

    def _build_resolution_cache_key(
        self,
        *,
        search_result: dict[str, Any],
        property_data: dict[str, Any],
        detail_html: str,
    ) -> str:
        return json.dumps(
            {
                "url": str(search_result.get("url") or property_data.get("detail_url") or ""),
                "building_name": str(property_data.get("building_name") or ""),
                "address": str(property_data.get("address") or ""),
                "detail_html_digest": hash(detail_html or ""),
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    def _build_image_query(
        self,
        *,
        search_result: dict[str, Any],
        property_data: dict[str, Any],
        adapter: LLMAdapter | None,
    ) -> str:
        fallback_query = " ".join(
            token
            for token in [
                str(property_data.get("building_name") or "").strip(),
                str(property_data.get("area_name") or "").strip(),
                str(property_data.get("nearest_station") or "").strip(),
                "賃貸",
                "外観",
            ]
            if token
        ).strip()

        if adapter is None:
            return fallback_query

        try:
            result = adapter.generate_text(
                system=(
                    "You create concise Brave image search queries for Japanese rental listings. "
                    "Prefer actual building or room photos. Avoid maps, logos, floor plans, and banners. "
                    "If the building name alone is too specific or unreliable, generalize to area/station plus 賃貸 外観 or 賃貸 内観. "
                    "Output only one search query."
                ),
                user=json.dumps(
                    {
                        "property": {
                            "building_name": str(property_data.get("building_name") or ""),
                            "address": str(property_data.get("address") or ""),
                            "area_name": str(property_data.get("area_name") or ""),
                            "nearest_station": str(property_data.get("nearest_station") or ""),
                            "layout": str(property_data.get("layout") or ""),
                            "notes": _compact_text(property_data.get("notes") or "", max_chars=120),
                            "features": [
                                str(item).strip()
                                for item in property_data.get("features", [])[:3]
                                if str(item).strip()
                            ],
                        },
                        "search_result": {
                            "title": str(search_result.get("title") or ""),
                            "description": _compact_text(search_result.get("description") or "", max_chars=140),
                            "url": str(search_result.get("url") or ""),
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                temperature=0.0,
            ).strip()
        except Exception:
            return fallback_query

        result = " ".join(result.split())
        return result[:160] if result else fallback_query

    def _search_brave_images(self, query: str) -> list[dict[str, Any]]:
        if not query or self.image_search_client is None:
            return []
        cached = self._image_query_cache.get(query)
        if cached is not None:
            return cached
        try:
            raw_results = self.image_search_client.search(query=query, count=8)
        except Exception:
            raw_results = []
        candidates = [
            {
                "display_url": str(item.get("thumbnail_url") or item.get("image_url") or ""),
                "image_url": str(item.get("image_url") or item.get("thumbnail_url") or ""),
                "page_url": str(item.get("page_url") or ""),
                "source_kind": "brave_image",
                "alt": "",
                "title": _compact_text(item.get("title") or "", max_chars=140),
                "context": "",
                "width": int(item.get("width") or 0),
                "height": int(item.get("height") or 0),
                "confidence": str(item.get("confidence") or ""),
                "source": str(item.get("source") or ""),
            }
            for item in raw_results
            if str(item.get("thumbnail_url") or item.get("image_url") or "").strip()
        ]
        deduped = _dedupe_candidates(candidates)
        self._image_query_cache[query] = deduped
        return deduped

    def _select_with_llm(
        self,
        *,
        adapter: LLMAdapter,
        candidates: list[dict[str, Any]],
        property_data: dict[str, Any],
        page_url: str,
    ) -> dict[str, Any] | None:
        if len(candidates) <= 1:
            return candidates[0] if candidates else None

        schema = {
            "type": "object",
            "properties": {
                "selected_index": {"type": "integer"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "reason": {"type": "string"},
            },
            "required": ["selected_index", "confidence", "reason"],
            "additionalProperties": False,
        }

        try:
            result = adapter.generate_structured(
                system=(
                    "You choose the single best image candidate for a Japanese rental property listing. "
                    "Prefer exterior or interior photos of the actual property. "
                    "Avoid maps, floor plans, logos, icons, tiny thumbnails, banners, and agent branding. "
                    "Use only the provided metadata. If none look appropriate, return selected_index as -1."
                ),
                user=json.dumps(
                    {
                        "property": {
                            "building_name": str(property_data.get("building_name") or ""),
                            "address": str(property_data.get("address") or ""),
                            "nearest_station": str(property_data.get("nearest_station") or ""),
                            "layout": str(property_data.get("layout") or ""),
                            "detail_url": page_url,
                        },
                        "candidates": [
                            {
                                "candidate_index": index,
                                "display_url": str(candidate.get("display_url") or ""),
                                "page_url": str(candidate.get("page_url") or ""),
                                "source_kind": str(candidate.get("source_kind") or ""),
                                "title": str(candidate.get("title") or ""),
                                "alt": str(candidate.get("alt") or ""),
                                "context": str(candidate.get("context") or ""),
                                "width": int(candidate.get("width") or 0),
                                "height": int(candidate.get("height") or 0),
                                "confidence": str(candidate.get("confidence") or ""),
                            }
                            for index, candidate in enumerate(candidates[:8])
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                schema=schema,
                temperature=0.0,
            )
        except Exception:
            return None

        selected_index = int(result.get("selected_index") or -1)
        confidence = float(result.get("confidence") or 0.0)
        if 0 <= selected_index < len(candidates) and confidence >= 0.45:
            return candidates[selected_index]
        return None

    def _select_candidate(
        self,
        *,
        candidates: list[dict[str, Any]],
        property_data: dict[str, Any],
        page_url: str,
        adapter: LLMAdapter | None,
    ) -> dict[str, Any] | None:
        if not candidates:
            return None

        scored = []
        for candidate in _dedupe_candidates(candidates):
            candidate_copy = dict(candidate)
            candidate_copy["heuristic_score"] = _heuristic_score_candidate(
                candidate=candidate_copy,
                property_data=property_data,
                page_url=page_url,
            )
            scored.append(candidate_copy)

        scored.sort(key=lambda item: float(item.get("heuristic_score") or 0.0), reverse=True)
        shortlisted = scored[:8]

        if adapter is not None:
            llm_choice = self._select_with_llm(
                adapter=adapter,
                candidates=shortlisted,
                property_data=property_data,
                page_url=page_url,
            )
            if llm_choice is not None:
                return llm_choice

        return shortlisted[0] if shortlisted else None

    def resolve(
        self,
        *,
        search_result: dict[str, Any],
        property_data: dict[str, Any],
        detail_html: str = "",
        adapter: LLMAdapter | None = None,
    ) -> str:
        explicit = str(property_data.get("image_url") or search_result.get("image_url") or "").strip()
        if explicit:
            return explicit

        cache_key = self._build_resolution_cache_key(
            search_result=search_result,
            property_data=property_data,
            detail_html=detail_html,
        )
        cached = self._resolution_cache.get(cache_key)
        if cached is not None:
            return cached

        page_url = str(search_result.get("url") or property_data.get("detail_url") or "").strip()
        candidates = _extract_html_image_candidates(detail_html, page_url) if detail_html else []
        chosen = self._select_candidate(
            candidates=candidates,
            property_data=property_data,
            page_url=page_url,
            adapter=adapter,
        )

        if chosen is None and self.image_search_client is not None:
            query = self._build_image_query(
                search_result=search_result,
                property_data=property_data,
                adapter=adapter,
            )
            brave_candidates = self._search_brave_images(query)
            chosen = self._select_candidate(
                candidates=brave_candidates,
                property_data=property_data,
                page_url=page_url,
                adapter=adapter,
            )

        resolved = str(chosen.get("display_url") or chosen.get("image_url") or "").strip() if chosen else ""
        self._resolution_cache[cache_key] = resolved
        return resolved
