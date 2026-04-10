from __future__ import annotations

import html
import json
from typing import Any

CATALOG_URL_PREFIX = "https://mock-housing.local/properties/"
MOCK_PROPERTY_IMAGE_URLS = {
    "koto-shinonome-bay": "https://images.unsplash.com/photo-1460317442991-0ec209397118?auto=format&fit=crop&w=1200&q=80",
    "koto-monzen-river": "https://images.unsplash.com/photo-1494526585095-c41746248156?auto=format&fit=crop&w=1200&q=80",
    "koto-ariake-park": "https://images.unsplash.com/photo-1502672260266-1c1ef2d93688?auto=format&fit=crop&w=1200&q=80",
    "koto-toyosu-breeze": "https://images.unsplash.com/photo-1512917774080-9991f1c4c750?auto=format&fit=crop&w=1200&q=80",
    "sumida-ryogoku-east": "https://images.unsplash.com/photo-1505693416388-ac5ce068fe85?auto=format&fit=crop&w=1200&q=80",
    "shinjuku-west-garden": "https://images.unsplash.com/photo-1464146072230-91cabc968266?auto=format&fit=crop&w=1200&q=80",
    "kichijoji-north-loft": "https://images.unsplash.com/photo-1484154218962-a197022b5858?auto=format&fit=crop&w=1200&q=80",
    "yokohama-bay-front": "https://images.unsplash.com/photo-1493809842364-78817add7ffb?auto=format&fit=crop&w=1200&q=80",
    "nakano-work-suite": "https://images.unsplash.com/photo-1507089947368-19c1da9775ae?auto=format&fit=crop&w=1200&q=80",
    "meguro-pet-garden": "https://images.unsplash.com/photo-1502005097973-6a7082348e28?auto=format&fit=crop&w=1200&q=80",
    "shinagawa-south-court": "https://images.unsplash.com/photo-1505692952047-1a78307da8f2?auto=format&fit=crop&w=1200&q=80",
    "machida-central-studio": "https://images.unsplash.com/photo-1522708323590-d24dbb6b0267?auto=format&fit=crop&w=1200&q=80",
    "machida-south-terrace": "https://images.unsplash.com/photo-1536376072261-38c75010e6c9?auto=format&fit=crop&w=1200&q=80",
}


# JP: catalog detail URLを構築する。
# EN: Build catalog detail URL.
def build_catalog_detail_url(property_id: str) -> str:
    return f"{CATALOG_URL_PREFIX}{property_id}"


# JP: catalog image URLを構築する。
# EN: Build catalog image URL.
def build_catalog_image_url(property_id: str) -> str:
    return MOCK_PROPERTY_IMAGE_URLS.get(str(property_id or "").strip(), "")


CATALOG_SEED: list[dict[str, Any]] = [
    {
        "property_id": "koto-shinonome-bay",
        "building_name": "東雲ベイテラス",
        "address": "東京都江東区東雲1-4-8",
        "area_name": "江東区",
        "nearest_station": "豊洲駅",
        "line_name": "東京メトロ有楽町線",
        "station_walk_min": 6,
        "layout": "1LDK",
        "area_m2": 42.1,
        "rent": 118000,
        "management_fee": 8000,
        "deposit": 118000,
        "key_money": 118000,
        "available_date": "2026-05-上旬",
        "agency_name": "Mock Homes 豊洲店",
        "notes": "南東向き。2人入居相談可。宅配ボックス・浴室乾燥あり。",
        "contract_text": "更新料1ヶ月。短期解約違約金あり（1年未満で1ヶ月分）。解約予告2か月前。保証会社加入必須。",
        "features": ["南東向き", "2人入居相談可", "宅配ボックス", "浴室乾燥"],
    },
    {
        "property_id": "koto-monzen-river",
        "building_name": "門前仲町リバーフロント",
        "address": "東京都江東区牡丹2-7-3",
        "area_name": "江東区",
        "nearest_station": "門前仲町駅",
        "line_name": "東京メトロ東西線",
        "station_walk_min": 7,
        "layout": "1LDK",
        "area_m2": 39.4,
        "rent": 121000,
        "management_fee": 7000,
        "deposit": 121000,
        "key_money": 121000,
        "available_date": "2026-04-下旬",
        "agency_name": "Mock Living 門仲支店",
        "notes": "角部屋。追い焚きあり。スーパー徒歩3分。",
        "contract_text": "更新料1ヶ月。解約予告2か月前。保証料は初回月額総賃料の50%。",
        "features": ["角部屋", "追い焚き", "スーパー近い"],
    },
    {
        "property_id": "koto-ariake-park",
        "building_name": "有明パークレジデンス",
        "address": "東京都江東区有明1-6-19",
        "area_name": "江東区",
        "nearest_station": "有明テニスの森駅",
        "line_name": "ゆりかもめ",
        "station_walk_min": 9,
        "layout": "1DK",
        "area_m2": 34.2,
        "rent": 109000,
        "management_fee": 6000,
        "deposit": 109000,
        "key_money": 0,
        "available_date": "2026-05-中旬",
        "agency_name": "Mock Urban 有明店",
        "notes": "礼金ゼロ。共用ワークラウンジ付き。",
        "contract_text": "短期解約違約金あり（6か月未満で2ヶ月分）。解約予告1か月前。保証会社利用必須。",
        "features": ["礼金ゼロ", "ワークラウンジ", "宅配ボックス"],
    },
    {
        "property_id": "koto-toyosu-breeze",
        "building_name": "豊洲ブリーズコート",
        "address": "東京都江東区豊洲4-1-12",
        "area_name": "江東区",
        "nearest_station": "豊洲駅",
        "line_name": "東京メトロ有楽町線",
        "station_walk_min": 4,
        "layout": "2LDK",
        "area_m2": 58.8,
        "rent": 164000,
        "management_fee": 10000,
        "deposit": 164000,
        "key_money": 164000,
        "available_date": "2026-06-上旬",
        "agency_name": "Mock Homes 豊洲店",
        "notes": "分譲タイプ。床暖房あり。ファミリー向け。",
        "contract_text": "更新料1ヶ月。保証会社加入必須。解約予告2か月前。",
        "features": ["床暖房", "分譲タイプ", "ファミリー向け"],
    },
    {
        "property_id": "sumida-ryogoku-east",
        "building_name": "両国イーストアーク",
        "address": "東京都墨田区亀沢2-15-2",
        "area_name": "墨田区",
        "nearest_station": "両国駅",
        "line_name": "都営大江戸線",
        "station_walk_min": 8,
        "layout": "1LDK",
        "area_m2": 40.3,
        "rent": 115000,
        "management_fee": 5000,
        "deposit": 115000,
        "key_money": 115000,
        "available_date": "2026-05-上旬",
        "agency_name": "Mock Living 両国店",
        "notes": "ペット相談可。SOHO不可。",
        "contract_text": "更新料1ヶ月。ペット飼育時は敷金1ヶ月追加。保証会社加入必須。",
        "features": ["ペット相談可", "バス・トイレ別"],
    },
    {
        "property_id": "shinjuku-west-garden",
        "building_name": "西新宿ガーデンヒルズ",
        "address": "東京都新宿区西新宿4-18-6",
        "area_name": "新宿区",
        "nearest_station": "西新宿五丁目駅",
        "line_name": "都営大江戸線",
        "station_walk_min": 5,
        "layout": "2LDK",
        "area_m2": 56.0,
        "rent": 248000,
        "management_fee": 12000,
        "deposit": 248000,
        "key_money": 248000,
        "available_date": "2026-04-下旬",
        "agency_name": "Mock Prime 新宿店",
        "notes": "新宿駅までバス10分。南向き。",
        "contract_text": "更新料1ヶ月。解約予告2か月前。短期解約違約金あり。",
        "features": ["南向き", "新宿アクセス", "オートロック"],
    },
    {
        "property_id": "kichijoji-north-loft",
        "building_name": "吉祥寺ノースロフト",
        "address": "東京都武蔵野市吉祥寺北町1-9-4",
        "area_name": "武蔵野市",
        "nearest_station": "吉祥寺駅",
        "line_name": "JR中央線",
        "station_walk_min": 11,
        "layout": "1K",
        "area_m2": 26.8,
        "rent": 98000,
        "management_fee": 4000,
        "deposit": 98000,
        "key_money": 0,
        "available_date": "2026-05-中旬",
        "agency_name": "Mock Studio 吉祥寺店",
        "notes": "ロフト付き。礼金ゼロ。自転車置き場あり。",
        "contract_text": "更新料1ヶ月。解約予告1か月前。保証会社加入必須。",
        "features": ["礼金ゼロ", "ロフト", "自転車置き場"],
    },
    {
        "property_id": "yokohama-bay-front",
        "building_name": "横浜ベイフロントテラス",
        "address": "神奈川県横浜市神奈川区栄町12-7",
        "area_name": "横浜市",
        "nearest_station": "横浜駅",
        "line_name": "JR東海道線",
        "station_walk_min": 9,
        "layout": "2LDK",
        "area_m2": 61.2,
        "rent": 172000,
        "management_fee": 10000,
        "deposit": 172000,
        "key_money": 172000,
        "available_date": "2026-05-下旬",
        "agency_name": "Mock Prime 横浜店",
        "notes": "海沿い眺望。食洗機あり。",
        "contract_text": "更新料1ヶ月。解約予告2か月前。保証会社加入必須。",
        "features": ["眺望良好", "食洗機", "宅配ボックス"],
    },
    {
        "property_id": "nakano-work-suite",
        "building_name": "中野ワークスイート",
        "address": "東京都中野区中野3-28-11",
        "area_name": "中野区",
        "nearest_station": "中野駅",
        "line_name": "JR中央線",
        "station_walk_min": 6,
        "layout": "1LDK",
        "area_m2": 45.0,
        "rent": 143000,
        "management_fee": 7000,
        "deposit": 143000,
        "key_money": 143000,
        "available_date": "2026-05-上旬",
        "agency_name": "Mock Living 中野店",
        "notes": "ワークスペース造作あり。高速回線導入済み。",
        "contract_text": "更新料1ヶ月。解約予告2か月前。保証会社加入必須。",
        "features": ["在宅ワーク向け", "高速回線", "2面採光"],
    },
    {
        "property_id": "meguro-pet-garden",
        "building_name": "目黒ペットガーデン",
        "address": "東京都目黒区下目黒2-21-4",
        "area_name": "目黒区",
        "nearest_station": "目黒駅",
        "line_name": "JR山手線",
        "station_walk_min": 10,
        "layout": "1LDK",
        "area_m2": 41.7,
        "rent": 176000,
        "management_fee": 9000,
        "deposit": 176000,
        "key_money": 176000,
        "available_date": "2026-06-上旬",
        "agency_name": "Mock Prime 目黒店",
        "notes": "小型犬1匹まで相談可。専用庭あり。",
        "contract_text": "更新料1ヶ月。ペット飼育時は敷金1ヶ月追加。短期解約違約金あり。",
        "features": ["ペット可", "専用庭", "南向き"],
    },
    {
        "property_id": "shinagawa-south-court",
        "building_name": "品川サウスコート",
        "address": "東京都港区港南3-5-14",
        "area_name": "港区",
        "nearest_station": "品川駅",
        "line_name": "JR山手線",
        "station_walk_min": 12,
        "layout": "2LDK",
        "area_m2": 54.8,
        "rent": 214000,
        "management_fee": 12000,
        "deposit": 214000,
        "key_money": 214000,
        "available_date": "2026-05-下旬",
        "agency_name": "Mock Prime 品川店",
        "notes": "コンシェルジュ付き。法人契約相談可。",
        "contract_text": "更新料1ヶ月。保証会社利用必須。解約予告2か月前。",
        "features": ["法人契約相談可", "コンシェルジュ", "宅配ボックス"],
    },
    {
        "property_id": "machida-central-studio",
        "building_name": "町田セントラルスタジオ",
        "address": "東京都町田市原町田6-7-8",
        "area_name": "町田市",
        "nearest_station": "町田駅",
        "line_name": "小田急線",
        "station_walk_min": 7,
        "layout": "1R",
        "area_m2": 25.4,
        "rent": 112000,
        "management_fee": 8000,
        "deposit": 112000,
        "key_money": 112000,
        "available_date": "2026-05-中旬",
        "agency_name": "Mock Studio 町田店",
        "notes": "3階 / 10階建。オートロックあり。駅前商業施設に近いワンルーム。",
        "contract_text": "更新料1ヶ月。短期解約違約金あり（1年未満で1ヶ月分）。解約予告2か月前。保証会社加入必須。",
        "features": ["オートロック", "2階以上", "宅配ボックス"],
    },
    {
        "property_id": "machida-south-terrace",
        "building_name": "町田サウステラス",
        "address": "東京都町田市森野1-12-5",
        "area_name": "町田市",
        "nearest_station": "町田駅",
        "line_name": "JR横浜線",
        "station_walk_min": 9,
        "layout": "1R",
        "area_m2": 23.8,
        "rent": 105000,
        "management_fee": 7000,
        "deposit": 105000,
        "key_money": 0,
        "available_date": "2026-05-下旬",
        "agency_name": "Mock Living 町田店",
        "notes": "4階 / 8階建。オートロックあり。礼金ゼロのワンルーム。",
        "contract_text": "更新料1ヶ月。解約予告1か月前。保証会社加入必須。",
        "features": ["オートロック", "2階以上", "礼金ゼロ"],
    },
]


# JP: rewrite catalog notesを処理する。
# EN: Process rewrite catalog notes.
def rewrite_catalog_notes(catalog: list[dict[str, Any]], adapter: Any) -> list[dict[str, Any]]:
    """notesフィールドをLLMでリライトして検索品質を向上させる。起動時1回の前処理用。"""
    result: list[dict[str, Any]] = []
    for prop in catalog:
        rewritten = dict(prop)
        try:
            notes = str(prop.get("notes") or "")
            features = [str(f) for f in (prop.get("features") or [])]
            name = str(prop.get("building_name") or "")
            layout = str(prop.get("layout") or "")
            area = str(prop.get("area_name") or "")
            rent = int(prop.get("rent") or 0)
            system = (
                "You are a Japanese rental property copywriter. "
                "Rewrite the property notes to be more informative and search-friendly, "
                "highlighting unique selling points and key details a renter would search for. "
                "Output at most 100 characters in natural Japanese. Output the notes text only, no quotes."
            )
            user_prompt = (
                f"物件: {name} ({area} / {layout} / {rent:,}円)\n"
                f"特徴: {', '.join(features)}\n"
                f"既存の備考: {notes}\n"
                "検索にヒットしやすく魅力的な備考文に書き直してください。"
            )
            new_notes = adapter.generate_text(
                system=system, user=user_prompt, temperature=0.2
            ).strip()
            if new_notes:
                rewritten["notes"] = new_notes
        except Exception:
            pass
        result.append(rewritten)
    return result


# JP: property detail htmlを生成する。
# EN: Render property detail html.
def render_property_detail_html(property_row: dict[str, Any]) -> str:
    image_url = str(
        property_row.get("image_url")
        or build_catalog_image_url(str(property_row.get("property_id") or ""))
    )
    payload = dict(property_row)
    payload["image_url"] = image_url
    features_json = json.dumps(property_row.get("features", []), ensure_ascii=False)
    safe = {key: html.escape(str(value)) for key, value in payload.items()}
    return f"""
    <html lang="ja">
      <head>
        <title>{safe["building_name"]} | Mock Housing</title>
        <meta name="description" content="{safe["building_name"]}の詳細情報" />
        <meta property="og:image" content="{safe["image_url"]}" />
      </head>
      <body>
        <article data-kind="property-detail">
          <h1 data-field="building_name">{safe["building_name"]}</h1>
          <p data-field="property_id">{safe["property_id"]}</p>
          <p data-field="image_url">{safe["image_url"]}</p>
          <p data-field="address">{safe["address"]}</p>
          <p data-field="area_name">{safe["area_name"]}</p>
          <p data-field="nearest_station">{safe["nearest_station"]}</p>
          <p data-field="line_name">{safe["line_name"]}</p>
          <p data-field="station_walk_min">{safe["station_walk_min"]}</p>
          <p data-field="layout">{safe["layout"]}</p>
          <p data-field="area_m2">{safe["area_m2"]}</p>
          <p data-field="rent">{safe["rent"]}</p>
          <p data-field="management_fee">{safe["management_fee"]}</p>
          <p data-field="deposit">{safe["deposit"]}</p>
          <p data-field="key_money">{safe["key_money"]}</p>
          <p data-field="available_date">{safe["available_date"]}</p>
          <p data-field="agency_name">{safe["agency_name"]}</p>
          <img src="{safe["image_url"]}" alt="{safe["building_name"]} の物件写真" />
          <section data-field="notes">{safe["notes"]}</section>
          <section data-field="contract_text">{safe["contract_text"]}</section>
          <script type="application/json" data-field="features">{html.escape(features_json)}</script>
        </article>
      </body>
    </html>
    """.strip()
