from app.stages.communication import run_communication


def test_communication_reflects_property_features_and_user_focus():
    normalized_properties = [
        {
            "property_id_norm": "p1",
            "building_name": "目黒ペットガーデン",
            "rent": 176000,
            "nearest_station": "目黒駅",
            "station_walk_min": 4,
            "layout": "1LDK",
            "notes": "高速回線導入済み。小型犬1匹まで相談可。",
            "features": ["ペット可", "高速回線", "礼金ゼロ"],
        }
    ]

    result = run_communication(
        ranked_properties=[
            {
                "property_id_norm": "p1",
                "score": 88.0,
                "why_selected": "",
                "why_not_selected": "",
            }
        ],
        normalized_properties=normalized_properties,
        user_memory={
            "move_in_date": "2026-05",
            "must_conditions": ["ペット可"],
            "nice_to_have": ["在宅ワーク向け"],
        },
        selected_property_id="p1",
        adapter=None,
    )

    assert "ペット" in result["message_draft"]
    assert "回線" in result["message_draft"]
    assert any("ペット" in item for item in result["check_items"])
    assert any("回線" in item or "オンライン会議" in item for item in result["check_items"])
