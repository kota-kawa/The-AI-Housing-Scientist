from datetime import date

from app.llm.base import LLMAdapter
from app.stages.integrity_review import run_integrity_review


class FakeIntegrityAdapter(LLMAdapter):
    def __init__(self, payload: dict):
        self.payload = payload
        self.calls = 0

    def generate_text(self, *, system: str, user: str, temperature: float = 0.2) -> str:
        raise AssertionError("generate_text should not be called in integrity_review")

    def generate_structured(
        self,
        *,
        system: str,
        user: str,
        schema: dict,
        temperature: float = 0.2,
    ) -> dict:
        self.calls += 1
        return self.payload

    def list_models(self) -> list[str]:
        return ["fake-integrity-model"]


def test_integrity_review_rule_based_filters_stale_listing_before_ranking():
    result = run_integrity_review(
        normalized_properties=[
            {
                "property_id_norm": "p1",
                "building_name": "東雲ベイテラス",
                "detail_url": "https://example.com/p1",
                "address": "東京都江東区東雲1-4-8",
                "layout": "1LDK",
                "rent": 118000,
                "management_fee": 8000,
                "station_walk_min": 6,
                "available_date": "2026-05-上旬",
                "notes": "南東向き。2人入居相談可。",
                "features": ["宅配ボックス"],
            },
            {
                "property_id_norm": "p2",
                "building_name": "東雲オールドレジデンス",
                "detail_url": "https://example.com/p2",
                "address": "東京都江東区東雲2-1-2",
                "layout": "1LDK",
                "rent": 118000,
                "management_fee": 8000,
                "station_walk_min": 6,
                "available_date": "2025-11-01",
                "notes": "掲載継続中とあるが更新日は不明。",
                "features": [],
            },
        ],
        raw_results=[
            {
                "url": "https://example.com/p1",
                "title": "東雲ベイテラス 1LDK",
                "description": "家賃118,000円 管理費8,000円 徒歩6分 1LDK",
                "extra_snippets": [],
            },
            {
                "url": "https://example.com/p2",
                "title": "東雲オールドレジデンス 1LDK",
                "description": "成約済みのため募集終了。以前の賃料は118,000円。",
                "extra_snippets": ["参考写真あり"],
            },
        ],
        detail_html_map={
            "https://example.com/p1": "<html><body><p>賃料118,000円</p><p>管理費8,000円</p></body></html>",
            "https://example.com/p2": "<html><body><p>募集終了</p><p>賃料138,000円</p></body></html>",
        },
        today=date(2026, 4, 9),
    )

    assert [item["property_id_norm"] for item in result["normalized_properties"]] == ["p1"]
    assert result["dropped_property_ids"] == ["p2"]
    assert result["summary"]["kept_count"] == 1
    assert result["summary"]["dropped_count"] == 1
    review = result["integrity_reviews_by_id"]["p2"]
    assert review["should_drop"] is True
    assert review["trust_score"] <= 45
    assert any("募集終了" in item or "成約済み" in item for item in review["inconsistencies"])


def test_integrity_review_llm_can_drop_candidate_that_rules_keep():
    adapter = FakeIntegrityAdapter(
        {
            "reviews": [
                {
                    "property_id_norm": "p1",
                    "dimension_scores": {
                        "freshness": 2,
                        "pricing_consistency": 3,
                        "listing_consistency": 2,
                        "evidence_completeness": 3,
                    },
                    "inconsistencies": [
                        "本文中に参考写真・同タイプ住戸の表記があり、募集住戸の実体が曖昧",
                    ],
                    "should_drop": True,
                }
            ]
        }
    )

    result = run_integrity_review(
        normalized_properties=[
            {
                "property_id_norm": "p1",
                "building_name": "月島リファレンスハウス",
                "detail_url": "https://example.com/reference",
                "address": "東京都中央区月島1-2-3",
                "layout": "1LDK",
                "rent": 129000,
                "management_fee": 5000,
                "station_walk_min": 5,
                "available_date": "2026-05-01",
                "notes": "内装写真は同タイプ住戸を掲載。",
                "features": [],
            }
        ],
        raw_results=[
            {
                "url": "https://example.com/reference",
                "title": "月島リファレンスハウス 1LDK",
                "description": "家賃129,000円 管理費5,000円 徒歩5分 1LDK",
                "extra_snippets": [],
            }
        ],
        detail_html_map={
            "https://example.com/reference": "<html><body><p>家賃129,000円</p><p>同タイプ住戸の写真を掲載</p></body></html>"
        },
        adapter=adapter,
        today=date(2026, 4, 9),
    )

    assert adapter.calls == 1
    assert result["llm_reasoning_applied"] is True
    assert result["normalized_properties"] == []
    review = result["integrity_reviews_by_id"]["p1"]
    assert review["should_drop"] is True
    assert review["trust_score"] <= 45
    assert review["evidence_urls"] == ["https://example.com/reference"]


def test_integrity_review_drops_layout_mismatch_before_ranking():
    result = run_integrity_review(
        normalized_properties=[
            {
                "property_id_norm": "p1",
                "building_name": "町田ファミリーレジデンス",
                "detail_url": "https://example.com/p1",
                "address": "東京都町田市原町田6-7-8",
                "layout": "1LDK",
                "rent": 118000,
                "management_fee": 8000,
                "station_walk_min": 7,
                "available_date": "2026-05-01",
                "notes": "1LDK募集",
                "features": [],
            }
        ],
        raw_results=[
            {
                "url": "https://example.com/p1",
                "title": "町田ファミリーレジデンス 1LDK",
                "description": "東京都町田市原町田6-7-8 賃料118,000円 1LDK 徒歩7分",
                "extra_snippets": [],
            }
        ],
        detail_html_map={
            "https://example.com/p1": "<html><body><p>賃料118,000円</p><p>1LDK</p></body></html>"
        },
        layout_preference="ワンルーム",
        today=date(2026, 4, 9),
    )

    assert result["normalized_properties"] == []
    review = result["integrity_reviews_by_id"]["p1"]
    assert review["should_drop"] is True
    assert review["drop_reason_class"] == "layout_mismatch"
    assert result["summary"]["dropped_layout_mismatch_count"] == 1


def test_integrity_review_drops_explicit_must_condition_mismatch():
    result = run_integrity_review(
        normalized_properties=[
            {
                "property_id_norm": "p1",
                "building_name": "町田ローセキュリティ",
                "detail_url": "https://example.com/p1",
                "address": "東京都町田市原町田6-7-8",
                "layout": "1R",
                "rent": 98000,
                "management_fee": 5000,
                "station_walk_min": 8,
                "floor_level": 1,
                "has_autolock": False,
                "available_date": "2026-05-01",
                "notes": "1階、オートロックなし",
                "features": [],
            }
        ],
        raw_results=[
            {
                "url": "https://example.com/p1",
                "title": "町田ローセキュリティ 1R",
                "description": "1階 オートロックなし 賃料98,000円",
                "extra_snippets": [],
            }
        ],
        detail_html_map={"https://example.com/p1": "<html><body>1階 オートロックなし</body></html>"},
        must_conditions=["2階以上", "オートロック"],
        today=date(2026, 4, 9),
    )

    assert result["normalized_properties"] == []
    review = result["integrity_reviews_by_id"]["p1"]
    assert review["drop_reason_class"] == "must_mismatch"
    assert result["summary"]["dropped_must_mismatch_count"] == 1
