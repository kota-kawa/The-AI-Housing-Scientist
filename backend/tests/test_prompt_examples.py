import pytest

from app.stages import prompt_examples
from app.stages.prompt_examples import (
    PromptExamplesError,
    load_prompt_examples,
    sample_prompt_examples,
    validate_required_prompt_examples,
)


def test_validate_required_prompt_examples_passes():
    validate_required_prompt_examples()


def test_prompt_example_loading_raises_when_required_file_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(prompt_examples, "PROMPTS_DIR", tmp_path)
    prompt_examples._load_prompt_examples.cache_clear()

    with pytest.raises(PromptExamplesError, match="missing"):
        sample_prompt_examples("planner_examples.json", count=2)


def test_planner_prompt_examples_golden_cases_cover_core_slot_patterns():
    examples = {item["case_id"]: item for item in load_prompt_examples("planner_examples.json")}

    assert set(examples) == {
        "generic_search_needs_core_slots",
        "explicit_area_budget_and_structure",
        "memory_merge_with_must_and_nice_conditions",
    }

    generic = examples["generic_search_needs_core_slots"]["output"]
    assert generic["next_action"] == "missing_slots_question"
    assert generic["missing_slots"] == ["target_area", "budget_max", "station_walk_max"]
    assert len(generic["follow_up_questions"]) == 3
    assert "例以外でも大丈夫" in generic["follow_up_questions"][0]["question"]
    assert "ざっくりでも大丈夫" in generic["follow_up_questions"][1]["question"]

    explicit = examples["explicit_area_budget_and_structure"]["output"]
    assert explicit["user_memory"]["target_area"] == "町田"
    assert explicit["user_memory"]["budget_max"] == 100000
    assert explicit["user_memory"]["must_conditions"] == ["RC造"]
    assert len(explicit["seed_queries"]) >= 5
    assert any(any(area in query for area in ["相模原", "南町田"]) for query in explicit["seed_queries"])
    assert any("小田急線" in query for query in explicit["seed_queries"])
    assert any("11万円以下" in query for query in explicit["seed_queries"])

    merged = examples["memory_merge_with_must_and_nice_conditions"]["output"]
    assert merged["user_memory"]["target_area"] == "中野"
    assert merged["user_memory"]["budget_max"] == 140000
    assert merged["user_memory"]["station_walk_max"] == 10
    assert merged["user_memory"]["layout_preference"] == "1LDK"
    assert merged["user_memory"]["must_conditions"] == ["2階以上"]
    assert merged["user_memory"]["nice_to_have"] == ["独立洗面台"]
    assert any("東中野" in query for query in merged["seed_queries"])
    assert any("中央線" in query for query in merged["seed_queries"])
    assert any("15万円" in query for query in merged["seed_queries"])


def test_ranking_prompt_examples_golden_cases_cover_reasoning_levels():
    examples = {item["case_id"]: item for item in load_prompt_examples("ranking_examples.json")}

    assert set(examples) == {
        "strong_remote_work_match",
        "partial_pet_match",
        "none_match_for_independent_sink",
    }

    strong = examples["strong_remote_work_match"]["output"]
    assert strong["nice_to_have_assessments"][0]["match_level"] == "strong"
    assert "高速回線" in strong["nice_to_have_assessments"][0]["evidence"]
    assert "在宅ワーク" in strong["why_selected"]

    partial = examples["partial_pet_match"]["output"]
    assert partial["nice_to_have_assessments"][0]["match_level"] == "partial"
    assert partial["nice_to_have_assessments"][0]["evidence"] == "小型犬相談可"
    assert "条件付き" in partial["why_not_selected"]

    none_case = examples["none_match_for_independent_sink"]["output"]
    assert none_case["nice_to_have_assessments"][0]["match_level"] == "none"
    assert none_case["nice_to_have_assessments"][0]["evidence"] == ""
    assert "記載がなく" in none_case["why_not_selected"]
