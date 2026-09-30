"""The turn controller decides next steps without running tools."""

from veripatch.studio_domain import StudioAction, StudioDecision, StudioObservation, StudioSession
from veripatch.studio_turn import TurnNext, next_after_tool


def _session() -> StudioSession:
    return StudioSession(
        session_id="turn-policy", repo_root=".", provider="openai",
        model="test-model", reasoning_effort="low",
    )


def _decision(action: StudioAction) -> StudioDecision:
    fields = {"action": action, "rationale": "test", "path": "app.py"}
    if action is StudioAction.EDIT:
        fields.update(old_text="old", new_text="new")
    if action in {StudioAction.RUN_COMMAND, StudioAction.RUN_TESTS}:
        fields["command"] = ["python", "-m", "pytest"]
    return StudioDecision(**fields)


def test_read_and_unverified_command_return_control_to_model() -> None:
    session = _session()
    assert next_after_tool(
        session, _decision(StudioAction.READ),
        verification_command=False, requirements_met=False,
    ) is TurnNext.CONTINUE
    assert next_after_tool(
        session, _decision(StudioAction.RUN_COMMAND),
        verification_command=True, requirements_met=True,
    ) is TurnNext.CONTINUE


def test_successful_mutation_requests_follow_up_but_error_does_not() -> None:
    session = _session()
    edit = _decision(StudioAction.EDIT)
    assert next_after_tool(
        session, edit, verification_command=False, requirements_met=False,
    ) is TurnNext.FOLLOW_UP_MUTATION
    session.observations.append(StudioObservation(kind="tool_error", summary="failed"))
    assert next_after_tool(
        session, edit, verification_command=False, requirements_met=False,
    ) is TurnNext.CONTINUE


def test_verified_work_returns_control_to_model_for_final_review() -> None:
    session = _session()
    session.turn_changed_files = ["app.py"]
    session.verification_passed = True
    assert next_after_tool(
        session, _decision(StudioAction.RUN_TESTS),
        verification_command=True, requirements_met=True,
    ) is TurnNext.CONTINUE
    assert next_after_tool(
        session, _decision(StudioAction.RUN_TESTS),
        verification_command=True, requirements_met=False,
    ) is TurnNext.CONTINUE
