from __future__ import annotations

import copy
import json
import logging
import random
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
PLANNER_ALLOWED_SLOTS = {
    "target_area",
    "budget_max",
    "station_walk_max",
    "layout_preference",
    "move_in_date",
    "must_conditions",
    "nice_to_have",
}
PLANNER_ALLOWED_INTENTS = {"search", "risk_check", "general_question"}
PLANNER_ALLOWED_NEXT_ACTIONS = {
    "missing_slots_question",
    "search_and_compare",
    "risk_check",
    "guidance",
}
RANKING_ALLOWED_MATCH_LEVELS = {"strong", "partial", "none"}


class PromptExamplesError(RuntimeError):
    pass


def _fail(filename: str, message: str) -> None:
    full_message = f"invalid prompt examples in {filename}: {message}"
    logger.error(full_message)
    raise PromptExamplesError(full_message)


def _is_int_or_none(value: Any) -> bool:
    return value is None or (isinstance(value, int) and not isinstance(value, bool))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _expect_type(filename: str, value: Any, expected_type: type, path: str) -> None:
    if not isinstance(value, expected_type):
        _fail(filename, f"{path} must be {expected_type.__name__}")


def _expect_string(filename: str, value: Any, path: str) -> None:
    if not isinstance(value, str) or not value.strip():
        _fail(filename, f"{path} must be a non-empty string")


def _expect_string_list(filename: str, value: Any, path: str) -> None:
    if not isinstance(value, list):
        _fail(filename, f"{path} must be a list")
    for index, item in enumerate(value):
        _expect_string(filename, item, f"{path}[{index}]")


def _expect_slot_memory_shape(filename: str, value: Any, path: str) -> None:
    _expect_type(filename, value, dict, path)
    required_keys = {
        "budget_max",
        "target_area",
        "station_walk_max",
        "move_in_date",
        "layout_preference",
        "must_conditions",
        "nice_to_have",
    }
    if set(value.keys()) != required_keys:
        _fail(filename, f"{path} must have keys {sorted(required_keys)}")
    if not _is_int_or_none(value["budget_max"]):
        _fail(filename, f"{path}.budget_max must be integer or null")
    if not _is_int_or_none(value["station_walk_max"]):
        _fail(filename, f"{path}.station_walk_max must be integer or null")
    for key in ["target_area", "move_in_date", "layout_preference"]:
        field_value = value[key]
        if field_value is not None and not isinstance(field_value, str):
            _fail(filename, f"{path}.{key} must be string or null")
    for key in ["must_conditions", "nice_to_have"]:
        _expect_string_list(filename, value[key], f"{path}.{key}")


def _expect_condition_reasons_shape(filename: str, value: Any, path: str) -> None:
    _expect_type(filename, value, dict, path)
    required_keys = {
        "budget_max",
        "target_area",
        "station_walk_max",
        "move_in_date",
        "layout_preference",
        "must_conditions",
        "nice_to_have",
    }
    if set(value.keys()) != required_keys:
        _fail(filename, f"{path} must have keys {sorted(required_keys)}")
    for key in required_keys:
        if not isinstance(value[key], str):
            _fail(filename, f"{path}.{key} must be a string")


def _validate_planner_examples(filename: str, payload: list[Any]) -> None:
    if len(payload) < 2:
        _fail(filename, "at least 2 examples are required")

    for index, item in enumerate(payload):
        path = f"examples[{index}]"
        _expect_type(filename, item, dict, path)
        _expect_string(filename, item.get("case_id"), f"{path}.case_id")
        input_payload = item.get("input")
        output_payload = item.get("output")
        _expect_type(filename, input_payload, dict, f"{path}.input")
        _expect_type(filename, output_payload, dict, f"{path}.output")

        _expect_string(filename, input_payload.get("user_message"), f"{path}.input.user_message")
        current_memory = input_payload.get("current_user_memory")
        _expect_type(filename, current_memory, dict, f"{path}.input.current_user_memory")
        for key in current_memory.keys():
            if key not in PLANNER_ALLOWED_SLOTS:
                _fail(filename, f"{path}.input.current_user_memory contains unsupported key {key}")
        _expect_type(
            filename,
            input_payload.get("learned_preferences"),
            dict,
            f"{path}.input.learned_preferences",
        )
        history = input_payload.get("profile_history_summary")
        _expect_type(filename, history, dict, f"{path}.input.profile_history_summary")
        _expect_type(
            filename,
            history.get("recent_searches"),
            list,
            f"{path}.input.profile_history_summary.recent_searches",
        )
        _expect_type(
            filename,
            history.get("recent_reactions"),
            list,
            f"{path}.input.profile_history_summary.recent_reactions",
        )

        intent = output_payload.get("intent")
        if intent not in PLANNER_ALLOWED_INTENTS:
            _fail(filename, f"{path}.output.intent must be one of {sorted(PLANNER_ALLOWED_INTENTS)}")
        _expect_slot_memory_shape(filename, output_payload.get("user_memory"), f"{path}.output.user_memory")

        missing_slots = output_payload.get("missing_slots")
        _expect_type(filename, missing_slots, list, f"{path}.output.missing_slots")
        for slot_index, slot in enumerate(missing_slots):
            if slot not in PLANNER_ALLOWED_SLOTS:
                _fail(filename, f"{path}.output.missing_slots[{slot_index}] is invalid")

        follow_up_questions = output_payload.get("follow_up_questions")
        _expect_type(filename, follow_up_questions, list, f"{path}.output.follow_up_questions")
        for question_index, question in enumerate(follow_up_questions):
            qpath = f"{path}.output.follow_up_questions[{question_index}]"
            _expect_type(filename, question, dict, qpath)
            if question.get("slot") not in PLANNER_ALLOWED_SLOTS:
                _fail(filename, f"{qpath}.slot is invalid")
            _expect_string(filename, question.get("label"), f"{qpath}.label")
            _expect_string(filename, question.get("question"), f"{qpath}.question")
            _expect_string_list(filename, question.get("examples"), f"{qpath}.examples")

        next_action = output_payload.get("next_action")
        if next_action not in PLANNER_ALLOWED_NEXT_ACTIONS:
            _fail(
                filename,
                f"{path}.output.next_action must be one of {sorted(PLANNER_ALLOWED_NEXT_ACTIONS)}",
            )
        _expect_string_list(filename, output_payload.get("seed_queries"), f"{path}.output.seed_queries")

        research_plan = output_payload.get("research_plan")
        _expect_type(filename, research_plan, dict, f"{path}.output.research_plan")
        _expect_string(filename, research_plan.get("summary"), f"{path}.output.research_plan.summary")
        _expect_string(filename, research_plan.get("goal"), f"{path}.output.research_plan.goal")
        _expect_string_list(filename, research_plan.get("strategy"), f"{path}.output.research_plan.strategy")
        _expect_string(filename, research_plan.get("rationale"), f"{path}.output.research_plan.rationale")
        _expect_condition_reasons_shape(
            filename,
            output_payload.get("condition_reasons"),
            f"{path}.output.condition_reasons",
        )


def _validate_ranking_examples(filename: str, payload: list[Any]) -> None:
    if len(payload) < 2:
        _fail(filename, "at least 2 examples are required")

    for index, item in enumerate(payload):
        path = f"examples[{index}]"
        _expect_type(filename, item, dict, path)
        _expect_string(filename, item.get("case_id"), f"{path}.case_id")
        input_payload = item.get("input")
        output_payload = item.get("output")
        _expect_type(filename, input_payload, dict, f"{path}.input")
        _expect_type(filename, output_payload, dict, f"{path}.output")

        user_preferences = input_payload.get("user_preferences")
        _expect_type(filename, user_preferences, dict, f"{path}.input.user_preferences")
        required_pref_keys = {
            "budget_max",
            "station_walk_max",
            "layout_preference",
            "must_conditions",
            "nice_to_have",
        }
        if set(user_preferences.keys()) != required_pref_keys:
            _fail(filename, f"{path}.input.user_preferences must have keys {sorted(required_pref_keys)}")
        if not isinstance(user_preferences["budget_max"], int) or isinstance(
            user_preferences["budget_max"],
            bool,
        ):
            _fail(filename, f"{path}.input.user_preferences.budget_max must be integer")
        if not isinstance(user_preferences["station_walk_max"], int) or isinstance(
            user_preferences["station_walk_max"],
            bool,
        ):
            _fail(filename, f"{path}.input.user_preferences.station_walk_max must be integer")
        _expect_string(
            filename,
            user_preferences["layout_preference"],
            f"{path}.input.user_preferences.layout_preference",
        )
        _expect_string_list(
            filename,
            user_preferences["must_conditions"],
            f"{path}.input.user_preferences.must_conditions",
        )
        _expect_string_list(
            filename,
            user_preferences["nice_to_have"],
            f"{path}.input.user_preferences.nice_to_have",
        )

        input_property = input_payload.get("property")
        _expect_type(filename, input_property, dict, f"{path}.input.property")
        required_property_keys = {
            "property_id_norm",
            "building_name",
            "address",
            "area_name",
            "nearest_station",
            "station_walk_min",
            "rent",
            "layout",
            "area_m2",
            "features",
            "notes",
            "rule_based_positives",
            "rule_based_negatives",
        }
        if set(input_property.keys()) != required_property_keys:
            _fail(filename, f"{path}.input.property must have keys {sorted(required_property_keys)}")
        for key in ["property_id_norm", "building_name", "address", "area_name", "nearest_station", "layout", "notes"]:
            _expect_string(filename, input_property[key], f"{path}.input.property.{key}")
        if not isinstance(input_property["station_walk_min"], int) or isinstance(
            input_property["station_walk_min"],
            bool,
        ):
            _fail(filename, f"{path}.input.property.station_walk_min must be integer")
        if not isinstance(input_property["rent"], int) or isinstance(input_property["rent"], bool):
            _fail(filename, f"{path}.input.property.rent must be integer")
        if not _is_number(input_property["area_m2"]):
            _fail(filename, f"{path}.input.property.area_m2 must be numeric")
        _expect_string_list(filename, input_property["features"], f"{path}.input.property.features")
        _expect_string_list(
            filename,
            input_property["rule_based_positives"],
            f"{path}.input.property.rule_based_positives",
        )
        _expect_string_list(
            filename,
            input_property["rule_based_negatives"],
            f"{path}.input.property.rule_based_negatives",
        )

        _expect_string(
            filename,
            output_payload.get("property_id_norm"),
            f"{path}.output.property_id_norm",
        )
        if output_payload["property_id_norm"] != input_property["property_id_norm"]:
            _fail(filename, f"{path}.output.property_id_norm must match input.property.property_id_norm")
        _expect_string(filename, output_payload.get("why_selected"), f"{path}.output.why_selected")
        _expect_string(filename, output_payload.get("why_not_selected"), f"{path}.output.why_not_selected")

        assessments = output_payload.get("nice_to_have_assessments")
        _expect_type(filename, assessments, list, f"{path}.output.nice_to_have_assessments")
        for assessment_index, assessment in enumerate(assessments):
            apath = f"{path}.output.nice_to_have_assessments[{assessment_index}]"
            _expect_type(filename, assessment, dict, apath)
            _expect_string(filename, assessment.get("condition"), f"{apath}.condition")
            if assessment.get("match_level") not in RANKING_ALLOWED_MATCH_LEVELS:
                _fail(
                    filename,
                    f"{apath}.match_level must be one of {sorted(RANKING_ALLOWED_MATCH_LEVELS)}",
                )
            if not isinstance(assessment.get("evidence"), str):
                _fail(filename, f"{apath}.evidence must be a string")


VALIDATORS = {
    "planner_examples.json": _validate_planner_examples,
    "ranking_examples.json": _validate_ranking_examples,
}


@lru_cache(maxsize=None)
def _load_prompt_examples(filename: str) -> tuple[dict[str, Any], ...]:
    path = PROMPTS_DIR / filename
    validator = VALIDATORS.get(filename)
    if validator is None:
        _fail(filename, "no validator is registered for this file")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        logger.exception("prompt examples file is missing: %s", path)
        raise PromptExamplesError(f"prompt examples file is missing: {path}") from exc
    except OSError as exc:
        logger.exception("prompt examples file could not be read: %s", path)
        raise PromptExamplesError(f"prompt examples file could not be read: {path}") from exc
    except json.JSONDecodeError as exc:
        logger.exception("prompt examples file is not valid JSON: %s", path)
        raise PromptExamplesError(f"prompt examples file is not valid JSON: {path}") from exc

    if not isinstance(payload, list):
        _fail(filename, "top-level value must be a list")

    validator(filename, payload)
    return tuple(copy.deepcopy(payload))


def load_prompt_examples(filename: str) -> list[dict[str, Any]]:
    return copy.deepcopy(list(_load_prompt_examples(filename)))


def sample_prompt_examples(filename: str, *, count: int = 2) -> list[dict[str, Any]]:
    examples = load_prompt_examples(filename)
    if len(examples) < count:
        _fail(filename, f"at least {count} examples are required to sample")
    return random.sample(examples, count)


def validate_required_prompt_examples() -> None:
    for filename in VALIDATORS:
        _load_prompt_examples(filename)
