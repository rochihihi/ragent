"""Policy for advancing a Studio turn after a tool result.

This module does not execute tools, approve effects, or format answers.  It
only chooses the next control-flow edge from current-turn evidence.
"""

from __future__ import annotations

from enum import StrEnum

from veripatch.studio_domain import StudioAction, StudioDecision, StudioSession


class TurnNext(StrEnum):
    CONTINUE = "continue"
    FOLLOW_UP_MUTATION = "follow_up_mutation"
    COMPLETE_VERIFIED = "complete_verified"


_MUTATIONS = frozenset({
    StudioAction.EDIT,
    StudioAction.APPLY_PATCH,
    StudioAction.CREATE,
    StudioAction.MOVE_FILE,
    StudioAction.COPY_FILE,
    StudioAction.DELETE_PATH,
    StudioAction.GIT_RESTORE,
})


def next_after_tool(
    session: StudioSession,
    decision: StudioDecision,
    *,
    verification_command: bool,
    requirements_met: bool,
) -> TurnNext:
    """Choose the next turn edge without performing another action."""
    if session.observations and session.observations[-1].kind in {
        "tool_error", "capability_guard",
    }:
        return TurnNext.CONTINUE
    # Successful verification returns evidence to the model, not a finish action.
    if decision.action in _MUTATIONS:
        return TurnNext.FOLLOW_UP_MUTATION
    return TurnNext.CONTINUE
