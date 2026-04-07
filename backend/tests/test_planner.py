from app.stages.planner import run_planner


def test_planner_extracts_required_slots():
    message = "渋谷で家賃15万以内、駅徒歩10分以内の1LDKを探したい"

    result = run_planner(message=message, user_memory={}, adapter=None)

    assert result["missing_slots"] == []
    assert result["plan"]["target_area"] == "渋谷"
    assert result["plan"]["budget_max"] == 150000
    assert result["plan"]["station_walk_max"] == 10
