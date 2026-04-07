from app.stages.planner import run_planner


def test_planner_extracts_required_slots():
    message = "渋谷で家賃15万以内、駅徒歩10分以内の1LDKを探したい"

    result = run_planner(message=message, user_memory={}, adapter=None)

    assert result["missing_slots"] == []
    assert result["plan"]["target_area"] == "渋谷"
    assert result["plan"]["budget_max"] == 150000
    assert result["plan"]["station_walk_max"] == 10


def test_planner_extracts_bare_ward_name_as_target_area():
    message = "江東区で家賃12万円以下、駅徒歩7分以内の1LDKを探しています"

    result = run_planner(message=message, user_memory={}, adapter=None)

    assert result["missing_slots"] == []
    assert result["plan"]["target_area"] == "江東区"


def test_planner_starts_search_even_when_target_area_is_missing():
    message = "家賃15万円以内、駅徒歩10分以内の1LDKを比較したい"

    result = run_planner(message=message, user_memory={}, adapter=None)

    assert result["missing_slots"] == []
    assert result["next_action"] == "search_and_compare"
    assert result["plan"]["target_area"] is None
    assert result["follow_up_questions"][0]["slot"] == "target_area"


def test_planner_asks_follow_up_questions_for_generic_request():
    message = "おすすめの賃貸を探したい"

    result = run_planner(message=message, user_memory={}, adapter=None)

    assert result["next_action"] == "missing_slots_question"
    assert result["missing_slots"] == ["target_area", "budget_max", "station_walk_max"]
    assert [item["slot"] for item in result["follow_up_questions"]] == [
        "target_area",
        "budget_max",
        "station_walk_max",
    ]
