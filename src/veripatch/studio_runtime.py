"""Runtime boundary between action execution and the Studio agent loop.

The orchestrator decides *what* to do.  This module owns the durable lifecycle
of an action result: mutate-version bookkeeping, evidence persistence, result
normalisation, recovery assessment, and completion follow-up.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

from veripatch import studio_completion as completion
from veripatch.studio_domain import (
    ObservationOutcome,
    StudioAction,
    StudioObservation,
    StudioSession,
)
from veripatch.studio_harness import ToolOutcome, ToolRegistry
from veripatch.studio_store import StudioStore
from veripatch.workspace import SafeWorkspace


class StudioExecutionRuntime:
    """Commit action outcomes without embedding execution policy in the agent."""

    def __init__(
        self,
        store: StudioStore,
        *,
        learn: Callable[[StudioSession, StudioAction, StudioObservation], list[str]],
        mark_finished: Callable[[StudioSession, StudioAction, StudioObservation], None],
        assess: Callable[[StudioSession, StudioAction, StudioObservation], Any],
        refresh: Callable[[StudioSession], None],
        pause: Callable[[StudioSession, str], Any],
        recovery_reason: Callable[[Any], str],
    ) -> None:
        self.store = store
        self.learn = learn
        self.mark_finished = mark_finished
        self.assess = assess
        self.refresh = refresh
        self.pause = pause
        self.recovery_reason = recovery_reason

    def commit(
        self,
        session: StudioSession,
        workspace: SafeWorkspace,
        action: StudioAction,
        observation: StudioObservation,
    ) -> bool:
        """Persist one complete action outcome and decide whether to continue."""
        descriptor = ToolRegistry.for_action(action)
        observation.payload.setdefault("execution_tool", descriptor.name)
        observation.payload.setdefault("execution_surface", descriptor.surface)
        self._record_mutation_evidence(session, workspace, observation)
        session.observations.append(observation)
        memory_changes = self.learn(session, action, observation)
        observation.tool_result = ToolOutcome.normalize(action, observation)
        self.store.save(session, "observation", observation.model_dump(mode="json"))
        if memory_changes:
            self.store.save(
                session,
                "memory_updated",
                {
                    "summary": f"工作记忆已更新：{'；'.join(memory_changes)}。",
                    "memory": session.memory.model_dump(mode="json"),
                },
            )
        self.mark_finished(session, action, observation)
        assessment = self.assess(session, action, observation)
        if assessment.outcome is ObservationOutcome.FAILED_TERMINAL:
            self.pause(session, self.recovery_reason(assessment))
            return True
        self.refresh(session)
        return False

    @staticmethod
    def _record_mutation_evidence(
        session: StudioSession,
        workspace: SafeWorkspace,
        observation: StudioObservation,
    ) -> None:
        if observation.kind not in completion.MUTATIONS:
            return
        session.action_epoch += 1
        session.verification_passed = False
        session.review_completed = False
        session.completion_rejections.clear()
        target = observation.payload.get("destination") or observation.payload.get("path")
        if target and observation.kind != "delete":
            resolved = workspace.resolve(target)
            if resolved.is_file():
                observation.payload["content_sha256"] = hashlib.sha256(
                    resolved.read_bytes()
                ).hexdigest()
        elif observation.kind == "patch":
            observation.payload["content_hashes"] = {
                path: hashlib.sha256(workspace.resolve(path).read_bytes()).hexdigest()
                for path in observation.payload.get("paths", [])
                if workspace.resolve(path).is_file()
            }
