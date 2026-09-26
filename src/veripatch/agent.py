"""Explicit, checkpointed and resumable repair state machine."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from veripatch.config import Settings
from veripatch.domain import (
    ActionKind,
    AgentDecision,
    AgentRunResult,
    AgentRunState,
    IssueSpec,
    Observation,
    RunnerKind,
    RunPhase,
)
from veripatch.indexing import PythonSymbolIndex
from veripatch.models.base import AgentModel, ModelContext
from veripatch.store import SQLiteRunStore
from veripatch.testing import (
    DockerPytestRunner,
    LocalPytestRunner,
    TestRunner,
    baseline_environment_error,
)
from veripatch.workspace import SafeWorkspace

RunnerFactory = Callable[[Path, RunnerKind], TestRunner]


class VeriPatchAgent:
    def __init__(
        self,
        model: AgentModel,
        *,
        settings: Settings | None = None,
        store: SQLiteRunStore | None = None,
        runner_factory: RunnerFactory | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.model = model
        self.store = store or SQLiteRunStore(self.settings.database_path)
        self.runner_factory = runner_factory or self._default_runner_factory

    def _default_runner_factory(self, root: Path, kind: RunnerKind) -> TestRunner:
        if kind is RunnerKind.LOCAL:
            return LocalPytestRunner(root, timeout_seconds=self.settings.test_timeout_seconds)
        return DockerPytestRunner(
            root,
            image=self.settings.docker_image,
            timeout_seconds=self.settings.test_timeout_seconds,
            cpus=self.settings.docker_cpus,
            memory=self.settings.docker_memory,
            pids=self.settings.docker_pids,
        )

    def _observe(self, state: AgentRunState, observation: Observation) -> None:
        state.observations.append(observation)
        self.store.record(state, "observation", observation.model_dump(mode="json"))

    def _fail(self, state: AgentRunState, reason: str) -> None:
        state.phase = RunPhase.FAILED
        state.failure_reason = reason
        self.store.record(state, "failed", {"reason": reason})

    def _sync_workspace(self, state: AgentRunState, workspace: SafeWorkspace) -> None:
        state.original_files = workspace.original_contents
        state.final_diff = workspace.diff()
        state.working_file_hashes = workspace.current_hashes()

    async def run(
        self,
        *,
        repo_root: Path,
        issue: IssueSpec,
        test_command: list[str],
        run_id: str | None = None,
        runner_kind: RunnerKind | str | None = None,
        provider: str = "unknown",
    ) -> AgentRunResult:
        workspace = SafeWorkspace(repo_root, max_file_bytes=self.settings.max_file_bytes)
        resolved_runner = RunnerKind(runner_kind or self.settings.test_runner)
        test_runner = self.runner_factory(workspace.root, resolved_runner)
        index = PythonSymbolIndex(workspace.root).build()
        state = AgentRunState(
            run_id=run_id or uuid4().hex,
            repo_root=str(workspace.root),
            issue=issue,
            test_command=test_command,
            provider=provider,
            runner=resolved_runner,
        )
        self.store.record(
            state,
            "created",
            {
                "issue_id": issue.issue_id,
                "repo_root": str(workspace.root),
                "provider": provider,
                "runner": resolved_runner,
            },
        )
        self._observe(
            state,
            Observation(
                kind="index",
                summary="Built Python AST symbol index.",
                payload={**index.summary(), "parse_errors": index.parse_errors},
            ),
        )

        state.phase = RunPhase.REPRODUCING
        self.store.checkpoint(state)
        try:
            baseline = test_runner.run(test_command)
        except Exception as exc:
            self._fail(state, f"Baseline test failed to execute: {type(exc).__name__}: {exc}")
            return AgentRunResult(state=state, diff="")
        state.last_test = baseline
        environment_error = baseline_environment_error(baseline)
        if environment_error is not None:
            self._observe(
                state,
                Observation(
                    kind="baseline_test",
                    summary="测试环境配置失败。",
                    payload={
                        **baseline.model_dump(mode="json"),
                        "classification": "environment_error",
                    },
                ),
            )
            self._fail(state, f"环境配置失败：{environment_error}")
            return AgentRunResult(state=state, diff="")
        self._observe(
            state,
            Observation(
                kind="baseline_test",
                summary="Baseline tests passed." if baseline.passed else "Issue reproduced.",
                payload=baseline.model_dump(mode="json"),
            ),
        )
        if baseline.passed:
            self._fail(state, "The supplied test command already passes; issue was not reproduced.")
            return AgentRunResult(state=state, diff="")

        state.phase = RunPhase.INVESTIGATING
        self.store.checkpoint(state)
        return await self._continue(state, workspace, index, test_runner)

    async def resume(self, run_id: str) -> AgentRunResult:
        state = self.store.load(run_id)
        if state is None:
            raise KeyError(f"Run not found: {run_id}")
        if state.terminal:
            raise ValueError(f"Terminal run cannot be resumed: {state.phase}")
        workspace = SafeWorkspace(
            Path(state.repo_root), max_file_bytes=self.settings.max_file_bytes
        )
        workspace.load_original_contents(state.original_files)
        if state.pending_edit is not None:
            try:
                changed = workspace.apply_prepared(state.pending_edit)
            except Exception as exc:
                self._fail(state, f"Pending edit recovery failed: {type(exc).__name__}: {exc}")
                return AgentRunResult(state=state, diff=state.final_diff)
            state.changed_files = sorted(set([*state.changed_files, *changed]))
            transaction_id = state.pending_edit.transaction_id
            state.pending_edit = None
            self._sync_workspace(state, workspace)
            state.observations.append(
                Observation(
                    kind="edit",
                    summary="Recovered a durable edit transaction.",
                    payload={"changed_files": changed},
                )
            )
            self.store.record(
                state,
                "edit_recovered",
                {"transaction_id": transaction_id, "changed_files": changed},
            )
        elif state.working_file_hashes:
            try:
                workspace.assert_hashes(state.working_file_hashes)
            except Exception as exc:
                self._fail(state, f"Workspace drift prevents resume: {type(exc).__name__}: {exc}")
                return AgentRunResult(state=state, diff=state.final_diff)
        index = PythonSymbolIndex(workspace.root).build()
        test_runner = self.runner_factory(workspace.root, state.runner)
        baseline_recorded = any(
            observation.kind == "baseline_test" for observation in state.observations
        )
        if baseline_recorded and state.last_test is not None:
            environment_error = baseline_environment_error(state.last_test)
            if environment_error is not None:
                self._fail(state, f"环境配置失败：{environment_error}")
                return AgentRunResult(state=state, diff=state.final_diff)
        if not baseline_recorded:
            state.phase = RunPhase.REPRODUCING
            self.store.append_event(state.run_id, "baseline_restarted", {})
            self.store.checkpoint(state)
            try:
                baseline = test_runner.run(state.test_command)
            except Exception as exc:
                self._fail(
                    state,
                    f"Baseline test failed to execute: {type(exc).__name__}: {exc}",
                )
                return AgentRunResult(state=state, diff=state.final_diff)
            state.last_test = baseline
            environment_error = baseline_environment_error(baseline)
            if environment_error is not None:
                self._observe(
                    state,
                    Observation(
                        kind="baseline_test",
                        summary="测试环境配置失败。",
                        payload={
                            **baseline.model_dump(mode="json"),
                            "classification": "environment_error",
                        },
                    ),
                )
                self._fail(state, f"环境配置失败：{environment_error}")
                return AgentRunResult(state=state, diff=state.final_diff)
            self._observe(
                state,
                Observation(
                    kind="baseline_test",
                    summary=("Baseline tests passed." if baseline.passed else "Issue reproduced."),
                    payload=baseline.model_dump(mode="json"),
                ),
            )
            if baseline.passed:
                self._fail(
                    state,
                    "The supplied test command already passes; issue was not reproduced.",
                )
                return AgentRunResult(state=state, diff=state.final_diff)
        state.phase = RunPhase.INVESTIGATING
        self.store.append_event(
            state.run_id,
            "resumed",
            {"from_step": state.step, "runner": state.runner},
        )
        self.store.checkpoint(state)
        return await self._continue(state, workspace, index, test_runner)

    async def _continue(
        self,
        state: AgentRunState,
        workspace: SafeWorkspace,
        index: PythonSymbolIndex,
        test_runner: TestRunner,
    ) -> AgentRunResult:
        for step in range(state.step + 1, self.settings.max_steps + 1):
            if state.usage.model_calls >= self.settings.max_model_calls:
                self._sync_workspace(state, workspace)
                self._fail(
                    state,
                    f"Model call budget exhausted ({self.settings.max_model_calls}).",
                )
                break
            context = ModelContext(
                issue=state.issue,
                step=step,
                changed_files=state.changed_files,
                index_summary=index.summary(),
                recent_observations=state.observations[-self.settings.max_observations_in_prompt :],
                last_diff=workspace.diff()[-12_000:],
            )
            try:
                reply = await self.model.decide(context)
            except Exception as exc:
                self._sync_workspace(state, workspace)
                self._fail(state, f"Model call failed: {type(exc).__name__}: {exc}")
                break

            state.step = step
            decision = reply.decision
            state.usage.model_calls += 1
            state.usage.input_tokens += reply.input_tokens
            state.usage.cached_input_tokens += reply.cached_input_tokens
            state.usage.output_tokens += reply.output_tokens
            state.usage.reasoning_tokens += reply.reasoning_tokens
            if reply.model:
                state.model = reply.model
            if reply.request_id:
                state.request_ids.append(reply.request_id)
                state.request_ids = state.request_ids[-20:]
            state.decisions.append(decision)
            self.store.append_event(state.run_id, "decision", decision.model_dump(mode="json"))

            if state.usage.input_tokens > self.settings.max_input_tokens:
                self._sync_workspace(state, workspace)
                self._fail(
                    state,
                    f"Input token budget exhausted ({self.settings.max_input_tokens}).",
                )
                break
            if state.usage.output_tokens > self.settings.max_output_tokens:
                self._sync_workspace(state, workspace)
                self._fail(
                    state,
                    f"Output token budget exhausted ({self.settings.max_output_tokens}).",
                )
                break

            fingerprint = hashlib.sha256(
                decision.model_dump_json(exclude={"rationale"}).encode()
            ).hexdigest()
            state.action_fingerprints.append(fingerprint)
            state.action_fingerprints = state.action_fingerprints[-3:]
            self.store.checkpoint(state)
            if len(state.action_fingerprints) == 3 and len(set(state.action_fingerprints)) == 1:
                self._sync_workspace(state, workspace)
                self._fail(state, "Stopped after three identical actions.")
                break

            try:
                observation = self._dispatch(
                    decision=decision,
                    state=state,
                    workspace=workspace,
                    index=index,
                    test_runner=test_runner,
                )
            except Exception as exc:
                state.phase = RunPhase.INVESTIGATING
                observation = Observation(
                    kind="tool_error",
                    summary=f"{decision.action} failed: {type(exc).__name__}: {exc}",
                    payload={"action": decision.action},
                )
            self._sync_workspace(state, workspace)
            self._observe(state, observation)

            if state.terminal:
                break

        if not state.terminal:
            self._sync_workspace(state, workspace)
            self._fail(state, f"Step budget exhausted ({self.settings.max_steps}).")
        return AgentRunResult(state=state, diff=state.final_diff)

    def _dispatch(
        self,
        *,
        decision: AgentDecision,
        state: AgentRunState,
        workspace: SafeWorkspace,
        index: PythonSymbolIndex,
        test_runner: TestRunner,
    ) -> Observation:
        if decision.action is ActionKind.SEARCH:
            results = workspace.search(decision.query or "")
            return Observation(
                kind="search",
                summary=f"Found {len(results)} lexical matches.",
                payload={"query": decision.query, "results": results},
            )
        if decision.action is ActionKind.LOOKUP_SYMBOL:
            records = [record.as_dict() for record in index.lookup(decision.query or "")]
            return Observation(
                kind="lookup_symbol",
                summary=f"Found {len(records)} symbol matches.",
                payload={"query": decision.query, "results": records},
            )
        if decision.action is ActionKind.READ:
            result = workspace.read(
                decision.path or "",
                decision.start_line or 1,
                decision.end_line or 240,
            )
            return Observation(kind="read", summary=f"Read {result['path']}.", payload=result)
        if decision.action is ActionKind.EDIT:
            state.phase = RunPhase.EDITING
            transaction = workspace.prepare_edits(decision.edits)
            state.pending_edit = transaction
            self._sync_workspace(state, workspace)
            self.store.record(
                state,
                "edit_prepared",
                {
                    "transaction_id": transaction.transaction_id,
                    "paths": [file.path for file in transaction.files],
                },
            )
            try:
                changed = workspace.apply_prepared(transaction)
            except Exception:
                state.pending_edit = None
                self._sync_workspace(state, workspace)
                self.store.record(
                    state,
                    "edit_aborted",
                    {"transaction_id": transaction.transaction_id},
                )
                raise
            state.changed_files = sorted(set([*state.changed_files, *changed]))
            state.pending_edit = None
            return Observation(
                kind="edit",
                summary=f"Applied {len(decision.edits)} exact edit(s).",
                payload={"changed_files": changed, "diff": workspace.diff()[-12_000:]},
            )
        if decision.action is ActionKind.RUN_TESTS:
            state.phase = RunPhase.VERIFYING
            outcome = test_runner.run(state.test_command)
            state.last_test = outcome
            if outcome.passed and workspace.diff():
                state.phase = RunPhase.SUCCEEDED
                self.store.append_event(
                    state.run_id,
                    "succeeded",
                    {"changed_files": state.changed_files, "step": state.step},
                )
            elif not outcome.passed:
                state.phase = RunPhase.INVESTIGATING
            return Observation(
                kind="test",
                summary="Verification passed." if outcome.passed else "Verification failed.",
                payload=outcome.model_dump(mode="json"),
            )
        if decision.action is ActionKind.FINISH:
            if not state.last_test or not state.last_test.passed or not workspace.diff():
                raise ValueError("Cannot finish without a non-empty patch and passing verification")
            state.phase = RunPhase.SUCCEEDED
            return Observation(
                kind="finish",
                summary=decision.final_summary or "Repair completed.",
            )
        if decision.action is ActionKind.FAIL:
            self._fail(state, decision.final_summary or decision.rationale)
            return Observation(kind="fail", summary=state.failure_reason or "Agent stopped.")
        raise ValueError(f"Unsupported action: {decision.action}")
