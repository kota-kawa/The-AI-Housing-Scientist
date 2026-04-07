import pytest

from app.research.state_machine import ResearchStageDefinition, ResearchStateMachine
from app.research.tools import CallableResearchTool, ToolContext, ToolSpec, Toolbox


def test_research_state_machine_runs_declared_transitions():
    visited: list[str] = []
    state = {"skip_to": "final"}

    def start_handler(current_state: dict[str, str]) -> str | None:
        visited.append("start")
        return "middle"

    def middle_handler(current_state: dict[str, str]) -> str | None:
        visited.append("middle")
        return current_state["skip_to"]

    def final_handler(current_state: dict[str, str]) -> str | None:
        visited.append("final")
        return None

    machine = ResearchStateMachine(
        [
            ResearchStageDefinition(name="start", handler=start_handler, default_next_stage="middle"),
            ResearchStageDefinition(name="middle", handler=middle_handler, default_next_stage="final"),
            ResearchStageDefinition(name="final", handler=final_handler, default_next_stage=None),
        ]
    )

    machine.run(state, start_stage="start")
    assert visited == ["start", "middle", "final"]


def test_toolbox_validates_input_and_output_schema():
    context = ToolContext(
        session_id="session-1",
        job_id="job-1",
        user_memory={},
        task_memory={},
        approved_plan={},
        provider="openai",
    )
    toolbox = Toolbox(
        [
            CallableResearchTool(
                ToolSpec(
                    name="echo",
                    description="echo integer count",
                    input_schema={
                        "type": "object",
                        "properties": {"count": {"type": "integer"}},
                        "required": ["count"],
                        "additionalProperties": False,
                    },
                    output_schema={
                        "type": "object",
                        "properties": {"count": {"type": "integer"}},
                        "required": ["count"],
                        "additionalProperties": False,
                    },
                ),
                lambda *, context, count: {"count": count},
            )
        ]
    )

    assert toolbox.run("echo", context, count=3) == {"count": 3}

    with pytest.raises(Exception):
        toolbox.run("echo", context, count="bad")
