from app.llm.base import LLMAdapter
from app.services.property_image import PropertyImageResolver


class FakeImageAdapter(LLMAdapter):
    def __init__(self):
        self.text_calls = 0
        self.structured_calls = 0

    def generate_text(self, *, system: str, user: str, temperature: float = 0.2) -> str:
        self.text_calls += 1
        return "中野駅 賃貸 外観"

    def generate_structured(
        self,
        *,
        system: str,
        user: str,
        schema: dict,
        temperature: float = 0.2,
    ) -> dict:
        self.structured_calls += 1
        return {"selected_index": 0, "confidence": 0.92, "reason": "建物写真として最も適切"}

    def list_models(self) -> list[str]:
        return ["fake-image-model"]


class FakeBraveImageSearchClient:
    def search(self, query: str, count: int = 8) -> list[dict]:
        assert "賃貸" in query
        return [
            {
                "title": "マンション入口",
                "page_url": "https://example.com/entrance",
                "source": "example.com",
                "image_url": "https://example.com/images/entrance.jpg",
                "thumbnail_url": "https://example.com/images/entrance-thumb.jpg",
                "width": 1200,
                "height": 800,
                "confidence": "medium",
            },
            {
                "title": "マンション外観",
                "page_url": "https://example.com/building",
                "source": "example.com",
                "image_url": "https://example.com/images/building.jpg",
                "thumbnail_url": "https://example.com/images/building-thumb.jpg",
                "width": 1600,
                "height": 1066,
                "confidence": "high",
            },
        ]


def test_property_image_resolver_extracts_image_from_detail_html():
    resolver = PropertyImageResolver()

    image_url = resolver.resolve(
        search_result={"url": "https://example.com/property"},
        property_data={
            "building_name": "東雲ベイテラス",
            "address": "東京都江東区東雲1-4-8",
        },
        detail_html="""
        <html>
          <head>
            <meta property="og:image" content="https://example.com/images/property-main.jpg" />
          </head>
          <body>
            <img src="/images/property-main.jpg" alt="東雲ベイテラス 外観" width="1200" height="800" />
          </body>
        </html>
        """,
        adapter=None,
    )

    assert image_url == "https://example.com/images/property-main.jpg"


def test_property_image_resolver_uses_llm_and_brave_fallback():
    adapter = FakeImageAdapter()
    resolver = PropertyImageResolver(image_search_client=FakeBraveImageSearchClient())

    image_url = resolver.resolve(
        search_result={
            "title": "中野ワークスイート | 外部掲載",
            "description": "中野駅徒歩6分の1LDK募集",
            "url": "https://example.com/nakano-work-suite",
        },
        property_data={
            "building_name": "中野ワークスイート",
            "address": "東京都中野区中野3-28-11",
            "area_name": "中野区",
            "nearest_station": "中野駅",
            "layout": "1LDK",
            "features": ["在宅ワーク向け"],
        },
        detail_html="<html><body><p>画像なし</p></body></html>",
        adapter=adapter,
    )

    assert image_url == "https://example.com/images/building-thumb.jpg"
    assert adapter.text_calls == 1
    assert adapter.structured_calls >= 1
