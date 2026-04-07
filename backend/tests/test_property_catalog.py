from app.llm.base import LLMAdapter
from app.services.property_catalog import PropertyCatalogService


class FakeCatalogDb:
    def __init__(self, items: list[dict]):
        self.items = items

    def list_catalog_properties(self) -> list[dict]:
        return self.items


class FakeCatalogAdapter(LLMAdapter):
    def __init__(self, payload: dict):
        self.payload = payload
        self.calls = 0

    def generate_text(self, *, system: str, user: str, temperature: float = 0.2) -> str:
        raise AssertionError("generate_text should not be called in property_catalog")

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
        return ["fake-catalog-model"]


def test_catalog_search_distinguishes_municipality_and_locality():
    service = PropertyCatalogService(
        FakeCatalogDb(
            [
                {
                    "property_id": "toyosu",
                    "building_name": "豊洲ブリーズコート",
                    "address": "東京都江東区豊洲4-1-12",
                    "area_name": "江東区",
                    "nearest_station": "豊洲駅",
                    "station_walk_min": 6,
                    "layout": "1LDK",
                    "area_m2": 42.0,
                    "rent": 125000,
                    "management_fee": 8000,
                    "available_date": "2026-05-上旬",
                    "detail_url": "https://example.com/toyosu",
                    "features": ["宅配ボックス"],
                    "notes": "南向き。",
                },
                {
                    "property_id": "shinonome",
                    "building_name": "東雲ベイテラス",
                    "address": "東京都江東区東雲1-4-8",
                    "area_name": "江東区",
                    "nearest_station": "豊洲駅",
                    "station_walk_min": 6,
                    "layout": "1LDK",
                    "area_m2": 42.0,
                    "rent": 125000,
                    "management_fee": 8000,
                    "available_date": "2026-05-上旬",
                    "detail_url": "https://example.com/shinonome",
                    "features": ["宅配ボックス"],
                    "notes": "南向き。",
                },
            ]
        )
    )

    result = service.search(
        query="江東区豊洲 1LDK",
        user_memory={
            "target_area": "江東区豊洲",
            "budget_max": 130000,
            "station_walk_max": 7,
            "layout_preference": "1LDK",
        },
        count=2,
    )

    assert result[0]["title"].startswith("豊洲ブリーズコート")


def test_catalog_search_llm_reranks_semantic_feature_matches():
    adapter = FakeCatalogAdapter(
        {
            "assessments": [
                {
                    "property_id": "remote",
                    "area_match_level": "exact",
                    "area_evidence": "中野区中野3丁目",
                    "must_condition_assessments": [],
                    "nice_to_have_assessments": [
                        {
                            "condition": "在宅ワーク向け",
                            "match_level": "strong",
                            "evidence": "光ファイバ完備とワークカウンターあり",
                        }
                    ],
                },
                {
                    "property_id": "standard",
                    "area_match_level": "exact",
                    "area_evidence": "中野区中野2丁目",
                    "must_condition_assessments": [],
                    "nice_to_have_assessments": [
                        {
                            "condition": "在宅ワーク向け",
                            "match_level": "none",
                            "evidence": "",
                        }
                    ],
                },
            ]
        }
    )
    service = PropertyCatalogService(
        FakeCatalogDb(
            [
                {
                    "property_id": "standard",
                    "building_name": "中野スタンダード",
                    "address": "東京都中野区中野2-10-1",
                    "area_name": "中野区",
                    "nearest_station": "中野駅",
                    "station_walk_min": 5,
                    "layout": "1LDK",
                    "area_m2": 38.0,
                    "rent": 128000,
                    "management_fee": 8000,
                    "available_date": "2026-05-上旬",
                    "detail_url": "https://example.com/standard",
                    "features": ["宅配ボックス"],
                    "notes": "収納が多い物件です。",
                },
                {
                    "property_id": "remote",
                    "building_name": "中野リモートスイート",
                    "address": "東京都中野区中野3-28-11",
                    "area_name": "中野区",
                    "nearest_station": "中野駅",
                    "station_walk_min": 5,
                    "layout": "1LDK",
                    "area_m2": 38.0,
                    "rent": 128000,
                    "management_fee": 8000,
                    "available_date": "2026-05-上旬",
                    "detail_url": "https://example.com/remote",
                    "features": ["光ファイバ完備"],
                    "notes": "ワークカウンターあり。",
                },
            ]
        )
    )

    result = service.search(
        query="中野 1LDK",
        user_memory={
            "target_area": "中野区",
            "budget_max": 130000,
            "station_walk_max": 7,
            "layout_preference": "1LDK",
            "nice_to_have": ["在宅ワーク向け"],
        },
        count=2,
        adapter=adapter,
    )

    assert adapter.calls == 1
    assert result[0]["title"].startswith("中野リモートスイート")
