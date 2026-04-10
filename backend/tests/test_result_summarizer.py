from app.stages.result_summarizer import (
    COMMON_RISKS_KEY,
    OPEN_QUESTIONS_KEY,
    PROPERTY_CANDIDATES_KEY,
    REJECTION_REASONS_KEY,
    run_result_summarizer,
)


def test_result_summarizer_fallback_merges_candidates_across_branch_nodes():
    branch_nodes = [
        {
            "branch_id": "node-1",
            "label": "initial",
            "depth": 1,
            "queries": ["江東区 賃貸 1LDK"],
            "issues": [],
            "search_summary": {"detail_hit_count": 1},
            "raw_results": [
                {
                    "title": "東雲ベイテラス 1LDK",
                    "url": "https://example.com/p1",
                    "description": "東京都江東区東雲1-4-8 11.8万円 徒歩6分",
                    "matched_queries": ["江東区 賃貸 1LDK"],
                }
            ],
            "detail_html_map": {
                "https://example.com/p1": "<html><body><p>東京都江東区東雲1-4-8</p><p>月額賃料11万8000円</p></body></html>"
            },
            "normalized_properties": [
                {
                    "property_id_norm": "p1",
                    "building_name": "東雲ベイテラス",
                    "address": "東京都江東区東雲1-4-8",
                    "detail_url": "https://example.com/p1",
                    "rent": 118000,
                    "layout": "1LDK",
                    "station_walk_min": 6,
                    "area_m2": 42.1,
                    "management_fee": 0,
                    "available_date": "",
                    "notes": "南東向き",
                    "features": ["在宅勤務しやすい間取り"],
                }
            ],
            "duplicate_groups": [],
            "ranked_properties": [
                {
                    "property_id_norm": "p1",
                    "score": 88.0,
                    "why_selected": "家賃と駅徒歩が条件内で、広さにも余裕があります。",
                    "why_not_selected": "",
                }
            ],
        },
        {
            "branch_id": "node-2",
            "label": "detail_first",
            "depth": 2,
            "queries": ["江東区 東雲 1LDK 詳細"],
            "issues": ["詳細ページ補完率が低い"],
            "search_summary": {"detail_hit_count": 1},
            "raw_results": [
                {
                    "title": "東雲ベイテラス 詳細",
                    "url": "https://example.com/p1",
                    "description": "東雲の1LDK募集",
                    "matched_queries": ["江東区 東雲 1LDK 詳細"],
                }
            ],
            "detail_html_map": {},
            "normalized_properties": [
                {
                    "property_id_norm": "p1",
                    "building_name": "東雲ベイテラス",
                    "address": "東京都江東区東雲1-4-8",
                    "detail_url": "https://example.com/p1",
                    "rent": 118000,
                    "layout": "1LDK",
                    "station_walk_min": 6,
                    "area_m2": 42.1,
                    "management_fee": 0,
                    "available_date": "",
                    "notes": "2人入居相談可",
                    "features": [],
                }
            ],
            "duplicate_groups": [],
            "ranked_properties": [
                {
                    "property_id_norm": "p1",
                    "score": 92.0,
                    "why_selected": "詳細補完後も条件一致度が高く、問い合わせ候補として有力です。",
                    "why_not_selected": "",
                }
            ],
        },
    ]

    result = run_result_summarizer(branch_nodes=branch_nodes, adapter=None)

    assert result["summary"]["branch_node_count"] == 2
    assert len(result[PROPERTY_CANDIDATES_KEY]) == 1
    assert result[PROPERTY_CANDIDATES_KEY][0]["property_id_norm"] == "p1"
    assert result[PROPERTY_CANDIDATES_KEY][0]["score"] == 92.0
    assert "江東区 賃貸 1LDK" in result[PROPERTY_CANDIDATES_KEY][0]["matched_queries"]
    assert result[REJECTION_REASONS_KEY][0]["reason"] == "詳細ページ補完率が低い"
    assert "管理費や初期費用の内訳が不足している" in result[COMMON_RISKS_KEY]
    assert "東雲ベイテラス の入居可能時期" in result[OPEN_QUESTIONS_KEY]


def test_result_summarizer_does_not_promote_dropped_or_search_hit_only_items():
    branch_nodes = [
        {
            "branch_id": "node-1",
            "label": "strict",
            "depth": 1,
            "queries": ["町田 賃貸 ワンルーム"],
            "issues": ["正規化後の候補が残っていない"],
            "search_summary": {"detail_hit_count": 1},
            "raw_results": [
                {
                    "title": "東雲ベイテラス 1LDK",
                    "url": "https://example.com/p1",
                    "description": "東京都江東区東雲1-4-8 1LDK",
                    "matched_queries": ["町田 賃貸 ワンルーム"],
                }
            ],
            "detail_html_map": {},
            "normalized_properties": [],
            "dropped_properties": [
                {
                    "property_id_norm": "p1",
                    "building_name": "東雲ベイテラス",
                    "integrity_review": {
                        "inconsistencies": ["希望エリア「町田」と物件の所在地が一致しないため除外"],
                        "should_drop": True,
                    },
                }
            ],
            "duplicate_groups": [],
            "ranked_properties": [],
        }
    ]

    result = run_result_summarizer(branch_nodes=branch_nodes, adapter=None)

    assert result[PROPERTY_CANDIDATES_KEY] == []
    assert result[REJECTION_REASONS_KEY][0]["target"] == "東雲ベイテラス"
    assert "推薦可能候補が残っていない" in result[COMMON_RISKS_KEY][0]
    assert "strict条件" in result[OPEN_QUESTIONS_KEY][0]
