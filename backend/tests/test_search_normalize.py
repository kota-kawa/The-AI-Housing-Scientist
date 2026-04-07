from app.llm.base import LLMAdapter
from app.stages.search_normalize import run_search_and_normalize


class FakeNormalizeAdapter(LLMAdapter):
    def __init__(self, payload: dict):
        self.payload = payload
        self.calls = 0

    def generate_text(self, *, system: str, user: str, temperature: float = 0.2) -> str:
        raise AssertionError("generate_text should not be called in search_normalize")

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
        return ["fake-normalize-model"]


def test_search_normalize_dedup_group_with_structured_address():
    items = [
        {
            "title": "サンプルマンション 1LDK 徒歩8分",
            "description": "東京都渋谷区渋谷1-2-3 家賃14.5万 35㎡",
            "url": "https://example.com/1",
            "extra_snippets": [],
        },
        {
            "title": "サンプルマンション 1LDK 徒歩8分",
            "description": "東京都渋谷区渋谷1-2-3 家賃14.8万 35㎡",
            "url": "https://example.com/2",
            "extra_snippets": [],
        },
    ]

    result = run_search_and_normalize(query="渋谷 賃貸", search_results=items)

    assert len(result["normalized_properties"]) == 2
    assert len(result["duplicate_groups"]) == 1


def test_search_normalize_avoids_false_dedup_when_address_missing():
    items = [
        {
            "title": "サンプルマンション 1LDK 徒歩8分",
            "description": "家賃14.5万 35㎡",
            "url": "https://example.com/1",
            "extra_snippets": [],
        },
        {
            "title": "サンプルマンション 1LDK 徒歩8分",
            "description": "家賃15.8万 35㎡",
            "url": "https://example.com/2",
            "extra_snippets": [],
        },
    ]

    result = run_search_and_normalize(query="渋谷 賃貸", search_results=items)

    assert len(result["normalized_properties"]) == 2
    assert len(result["duplicate_groups"]) == 0


def test_search_normalize_prefers_detail_page_payload():
    items = [
        {
            "title": "東雲ベイテラス | Mock Housing",
            "description": "江東区東雲の1LDK",
            "url": "https://mock-housing.local/properties/koto-shinonome-bay",
            "extra_snippets": [],
            "source_name": "mock_catalog",
        }
    ]

    def fetch_detail(url: str) -> str | None:
        if "koto-shinonome-bay" not in url:
            return None
        return """
        <article data-kind="property-detail">
          <h1 data-field="building_name">東雲ベイテラス</h1>
          <p data-field="property_id">koto-shinonome-bay</p>
          <p data-field="address">東京都江東区東雲1-4-8</p>
          <p data-field="nearest_station">豊洲駅</p>
          <p data-field="line_name">東京メトロ有楽町線</p>
          <p data-field="station_walk_min">6</p>
          <p data-field="layout">1LDK</p>
          <p data-field="area_m2">42.1</p>
          <p data-field="rent">118000</p>
          <p data-field="management_fee">8000</p>
          <p data-field="deposit">118000</p>
          <p data-field="key_money">118000</p>
          <p data-field="available_date">2026-05-上旬</p>
          <p data-field="agency_name">Mock Homes 豊洲店</p>
          <section data-field="notes">南東向き。2人入居相談可。</section>
        </article>
        """

    result = run_search_and_normalize(
        query="江東区 賃貸 12万円",
        search_results=items,
        detail_fetcher=fetch_detail,
    )

    prop = result["normalized_properties"][0]

    assert prop["building_name"] == "東雲ベイテラス"
    assert prop["address"] == "東京都江東区東雲1-4-8"
    assert prop["nearest_station"] == "豊洲駅"
    assert prop["rent"] == 118000
    assert result["summary"]["detail_parsed_count"] == 1


def test_search_normalize_extracts_base_rent_without_confusing_management_fee():
    items = [
        {
            "title": "東雲ベイテラス 1LDK",
            "description": "東京都江東区東雲1-4-8 管理費8,000円 家賃118,000円 徒歩6分 42.1㎡",
            "url": "https://example.com/shinonome",
            "extra_snippets": [],
        }
    ]

    result = run_search_and_normalize(
        query="江東区 賃貸 12万円",
        search_results=items,
    )

    prop = result["normalized_properties"][0]
    assert prop["rent"] == 118000


def test_search_normalize_llm_supplements_natural_detail_html():
    adapter = FakeNormalizeAdapter(
        {
            "rent": 118000,
            "layout": "1LDK",
            "station_walk_min": 6,
            "area_m2": 42.1,
            "extraction_confidence": 0.92,
        }
    )
    items = [
        {
            "title": "東雲ベイテラス | External Listing",
            "description": "江東区東雲の募集",
            "url": "https://example.com/external-shinonome",
            "extra_snippets": ["在宅勤務にも使いやすい広さ"],
            "source_name": "external",
        }
    ]

    def fetch_detail(url: str) -> str | None:
        if "external-shinonome" not in url:
            return None
        return """
        <html>
          <body>
            <h1>東雲ベイテラス</h1>
            <p>東京都江東区東雲1-4-8</p>
            <p>月額賃料は11万8000円、共益費8000円です。</p>
            <p>間取りは1LDK。専有面積は42.1平方メートル。</p>
            <p>有楽町線 豊洲駅まで徒歩およそ6分。</p>
          </body>
        </html>
        """

    result = run_search_and_normalize(
        query="江東区 賃貸 12万円",
        search_results=items,
        detail_fetcher=fetch_detail,
        adapter=adapter,
    )

    prop = result["normalized_properties"][0]
    assert adapter.calls == 1
    assert prop["rent"] == 118000
    assert prop["layout"] == "1LDK"
    assert prop["area_m2"] == 42.1
    assert prop["station_walk_min"] == 6
    assert result["summary"]["detail_parsed_count"] == 1


def test_search_normalize_fuzzy_dedup_with_name_variation():
    items = [
        {
            "title": "Park Heights 江東 | Mock Housing",
            "description": "江東区の1LDK",
            "url": "https://example.com/park-heights-1",
            "extra_snippets": [],
            "source_name": "mock_catalog",
        },
        {
            "title": "パークハイツ江東 | Mock Housing",
            "description": "江東区の1LDK",
            "url": "https://example.com/park-heights-2",
            "extra_snippets": [],
            "source_name": "mock_catalog",
        },
    ]

    def fetch_detail(url: str) -> str | None:
        if url.endswith("1"):
            return """
            <article data-kind="property-detail">
              <h1 data-field="building_name">Park Heights 江東</h1>
              <p data-field="property_id">park-heights-1</p>
              <p data-field="address">東京都江東区東雲1-4-8</p>
              <p data-field="nearest_station">豊洲駅</p>
              <p data-field="station_walk_min">6</p>
              <p data-field="layout">1LDK</p>
              <p data-field="area_m2">42.1</p>
              <p data-field="rent">118000</p>
            </article>
            """
        if url.endswith("2"):
            return """
            <article data-kind="property-detail">
              <h1 data-field="building_name">パークハイツ江東</h1>
              <p data-field="property_id">park-heights-2</p>
              <p data-field="address">東京都江東区東雲1-4-8</p>
              <p data-field="nearest_station">豊洲駅</p>
              <p data-field="station_walk_min">6</p>
              <p data-field="layout">1LDK</p>
              <p data-field="area_m2">42.1</p>
              <p data-field="rent">119000</p>
            </article>
            """
        return None

    result = run_search_and_normalize(
        query="江東区 賃貸 12万円",
        search_results=items,
        detail_fetcher=fetch_detail,
    )

    assert len(result["duplicate_groups"]) == 1
    assert result["duplicate_groups"][0]["confidence"] >= 0.8
