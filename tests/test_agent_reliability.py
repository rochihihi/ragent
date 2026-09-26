import asyncio
import shutil
import sys
from pathlib import Path

import pytest

from veripatch.agent import VeriPatchAgent
from veripatch.config import Settings
from veripatch.domain import (
    ActionKind,
    AgentDecision,
    AgentRunState,
    FileEdit,
    IssueSpec,
    Observation,
    RunnerKind,
    RunPhase,
)
from veripatch.domain import (
    TestOutcome as RunnerOutcome,
)
from veripatch.models.base import ModelContext, ModelReply
from veripatch.models.scripted import DiscountBugDemoModel
from veripatch.store import SQLiteRunStore
from veripatch.workspace import SafeWorkspace


class SimulatedCrash(BaseException):
    pass


class CrashAfterEditModel:
    def __init__(self) -> None:
        self.inner = DiscountBugDemoModel()

    async def decide(self, context: ModelContext) -> ModelReply:
        if any(observation.kind == "edit" for observation in context.recent_observations):
            raise SimulatedCrash("process disappeared")
        return await self.inner.decide(context)


class RepeatingSearchModel:
    async def decide(self, context: ModelContext) -> ModelReply:
        return ModelReply(
            decision=AgentDecision(
                action=ActionKind.SEARCH,
                rationale="Repeat search to exercise the stop policy.",
                query="calculate_discount",
            )
        )


class FailingModel:
    async def decide(self, context: ModelContext) -> ModelReply:
        raise RuntimeError("provider unavailable")


class TokenHeavyModel:
    async def decide(self, context: ModelContext) -> ModelReply:
        return ModelReply(
            decision=AgentDecision(
                action=ActionKind.SEARCH,
                rationale="Search once.",
                query="calculate_discount",
            ),
            input_tokens=101,
            output_tokens=21,
        )


class OutputHeavyModel:
    async def decide(self, context: ModelContext) -> ModelReply:
        return ModelReply(
            decision=AgentDecision(
                action=ActionKind.SEARCH,
                rationale="Search once.",
                query="calculate_discount",
            ),
            output_tokens=21,
            model="test-model",
            request_id="request-private",
        )


class FixedRunner:
    def __init__(self, outcome: RunnerOutcome | Exception) -> None:
        self.outcome = outcome

    def run(self, command: list[str]) -> RunnerOutcome:
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def _repository(tmp_path: Path) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    repository = tmp_path / "repository"
    shutil.copytree(project_root / "examples" / "discount_bug", repository)
    return repository


def _issue() -> IssueSpec:
    return IssueSpec(
        issue_id="discount-percentage",
        title="Incorrect percentage calculation",
        description="A 10 percent discount on 100 should be 90.",
    )


def _settings(tmp_path: Path, max_steps: int = 6) -> Settings:
    return Settings(
        max_steps=max_steps,
        test_timeout_seconds=30,
        database_path=tmp_path / "runs.sqlite3",
        test_runner="local",
    )


def test_resume_continues_from_checkpoint_with_complete_diff(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    settings = _settings(tmp_path)
    store = SQLiteRunStore(settings.database_path)
    run_id = "interrupted-run"
    crashing_agent = VeriPatchAgent(CrashAfterEditModel(), settings=settings, store=store)
    with pytest.raises(SimulatedCrash):
        asyncio.run(
            crashing_agent.run(
                repo_root=repository,
                issue=_issue(),
                test_command=[sys.executable, "-m", "pytest", "-q"],
                run_id=run_id,
                runner_kind=RunnerKind.LOCAL,
                provider="scripted-demo",
            )
        )
    interrupted = store.load(run_id)
    assert interrupted is not None
    assert not interrupted.terminal
    assert interrupted.step == 3
    assert interrupted.original_files

    resumed_agent = VeriPatchAgent(DiscountBugDemoModel(), settings=settings, store=store)
    result = asyncio.run(resumed_agent.resume(run_id))
    assert result.state.phase is RunPhase.SUCCEEDED
    assert "percent / 100" in result.diff
    assert any(event["event_type"] == "resumed" for event in store.events(run_id))


def test_repeated_action_stops_after_three_attempts(tmp_path: Path) -> None:
    settings = _settings(tmp_path, max_steps=5)
    result = asyncio.run(
        VeriPatchAgent(RepeatingSearchModel(), settings=settings).run(
            repo_root=_repository(tmp_path),
            issue=_issue(),
            test_command=[sys.executable, "-m", "pytest", "-q"],
            runner_kind=RunnerKind.LOCAL,
        )
    )
    assert result.state.phase is RunPhase.FAILED
    assert result.state.failure_reason == "Stopped after three identical actions."


def test_step_budget_and_model_failure_are_explicit(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    budget_settings = _settings(tmp_path, max_steps=2)
    budget_result = asyncio.run(
        VeriPatchAgent(RepeatingSearchModel(), settings=budget_settings).run(
            repo_root=repository,
            issue=_issue(),
            test_command=[sys.executable, "-m", "pytest", "-q"],
            runner_kind=RunnerKind.LOCAL,
        )
    )
    assert "Step budget exhausted" in (budget_result.state.failure_reason or "")

    failure_settings = Settings(
        max_steps=2,
        test_timeout_seconds=30,
        database_path=tmp_path / "failure.sqlite3",
        test_runner="local",
    )
    failure_result = asyncio.run(
        VeriPatchAgent(FailingModel(), settings=failure_settings).run(
            repo_root=repository,
            issue=_issue(),
            test_command=[sys.executable, "-m", "pytest", "-q"],
            runner_kind=RunnerKind.LOCAL,
        )
    )
    assert "Model call failed" in (failure_result.state.failure_reason or "")


def test_terminal_run_cannot_resume(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store = SQLiteRunStore(settings.database_path)
    agent = VeriPatchAgent(DiscountBugDemoModel(), settings=settings, store=store)
    result = asyncio.run(
        agent.run(
            repo_root=_repository(tmp_path),
            issue=_issue(),
            test_command=[sys.executable, "-m", "pytest", "-q"],
            runner_kind=RunnerKind.LOCAL,
        )
    )
    with pytest.raises(ValueError, match="Terminal run"):
        asyncio.run(agent.resume(result.state.run_id))


def _pending_state(tmp_path: Path, *, apply: bool = False) -> tuple[AgentRunState, SQLiteRunStore]:
    repository = _repository(tmp_path)
    workspace = SafeWorkspace(repository)
    transaction = workspace.prepare_edits(
        [
            FileEdit(
                path="discount/calc.py",
                old_text="return price * (1 - percent)",
                new_text="return price * (1 - percent / 100)",
            )
        ]
    )
    if apply:
        workspace.apply_prepared(transaction)
    state = AgentRunState(
        run_id="pending-run",
        repo_root=str(repository),
        issue=_issue(),
        test_command=[sys.executable, "-m", "pytest", "-q"],
        provider="scripted-demo",
        runner=RunnerKind.LOCAL,
        phase=RunPhase.EDITING,
        original_files=workspace.original_contents,
        pending_edit=transaction,
        observations=[Observation(kind="baseline_test", summary="Issue reproduced.")],
    )
    store = SQLiteRunStore(tmp_path / "pending.sqlite3")
    store.checkpoint(state)
    return state, store


@pytest.mark.parametrize("already_applied", [False, True])
def test_resume_recovers_prepared_or_already_written_edit(
    tmp_path: Path, already_applied: bool
) -> None:
    state, store = _pending_state(tmp_path, apply=already_applied)
    result = asyncio.run(
        VeriPatchAgent(DiscountBugDemoModel(), settings=_settings(tmp_path), store=store).resume(
            state.run_id
        )
    )
    assert result.state.phase is RunPhase.SUCCEEDED
    assert result.state.pending_edit is None
    assert any(event["event_type"] == "edit_recovered" for event in store.events(state.run_id))


def test_resume_rejects_pending_edit_workspace_drift(tmp_path: Path) -> None:
    state, store = _pending_state(tmp_path)
    (Path(state.repo_root) / "discount" / "calc.py").write_text(
        "unrelated = True\n", encoding="utf-8"
    )
    result = asyncio.run(
        VeriPatchAgent(FailingModel(), settings=_settings(tmp_path), store=store).resume(
            state.run_id
        )
    )
    assert result.state.phase is RunPhase.FAILED
    assert "Pending edit recovery failed" in (result.state.failure_reason or "")


def test_resume_restarts_missing_baseline(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    state = AgentRunState(
        run_id="before-baseline",
        repo_root=str(repository),
        issue=_issue(),
        test_command=[sys.executable, "-m", "pytest", "-q"],
        runner=RunnerKind.LOCAL,
    )
    store = SQLiteRunStore(tmp_path / "baseline.sqlite3")
    store.checkpoint(state)
    result = asyncio.run(
        VeriPatchAgent(FailingModel(), settings=_settings(tmp_path), store=store).resume(
            state.run_id
        )
    )
    assert result.state.phase is RunPhase.FAILED
    event_types = [event["event_type"] for event in store.events(state.run_id)]
    assert "baseline_restarted" in event_types
    assert any(observation.kind == "baseline_test" for observation in result.state.observations)


def test_model_call_and_token_budgets_have_explicit_terminal_states(tmp_path: Path) -> None:
    call_settings = Settings(
        max_steps=4,
        max_model_calls=1,
        database_path=tmp_path / "calls.sqlite3",
        test_timeout_seconds=30,
        test_runner="local",
    )
    call_result = asyncio.run(
        VeriPatchAgent(RepeatingSearchModel(), settings=call_settings).run(
            repo_root=_repository(tmp_path / "calls"),
            issue=_issue(),
            test_command=[sys.executable, "-m", "pytest", "-q"],
            runner_kind=RunnerKind.LOCAL,
        )
    )
    assert "Model call budget exhausted" in (call_result.state.failure_reason or "")

    token_settings = Settings(
        max_steps=3,
        max_input_tokens=100,
        max_output_tokens=20,
        database_path=tmp_path / "tokens.sqlite3",
        test_timeout_seconds=30,
        test_runner="local",
    )
    token_result = asyncio.run(
        VeriPatchAgent(TokenHeavyModel(), settings=token_settings).run(
            repo_root=_repository(tmp_path / "tokens"),
            issue=_issue(),
            test_command=[sys.executable, "-m", "pytest", "-q"],
            runner_kind=RunnerKind.LOCAL,
        )
    )
    assert "Input token budget exhausted" in (token_result.state.failure_reason or "")

    output_result = asyncio.run(
        VeriPatchAgent(OutputHeavyModel(), settings=token_settings).run(
            repo_root=_repository(tmp_path / "outputs"),
            issue=_issue(),
            test_command=[sys.executable, "-m", "pytest", "-q"],
            runner_kind=RunnerKind.LOCAL,
        )
    )
    assert "Output token budget exhausted" in (output_result.state.failure_reason or "")
    assert output_result.state.model == "test-model"
    assert output_result.state.request_ids == ["request-private"]


def test_crash_after_write_recovers_persisted_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repository(tmp_path)
    settings = _settings(tmp_path)
    store = SQLiteRunStore(settings.database_path)
    agent = VeriPatchAgent(DiscountBugDemoModel(), settings=settings, store=store)
    real_observe = agent._observe

    def crash_before_commit(state: AgentRunState, observation: Observation) -> None:
        if observation.kind == "edit":
            raise SimulatedCrash("crash after filesystem write")
        real_observe(state, observation)

    monkeypatch.setattr(agent, "_observe", crash_before_commit)
    with pytest.raises(SimulatedCrash):
        asyncio.run(
            agent.run(
                repo_root=repository,
                issue=_issue(),
                test_command=[sys.executable, "-m", "pytest", "-q"],
                run_id="after-write",
                runner_kind=RunnerKind.LOCAL,
            )
        )
    persisted = store.load("after-write")
    assert persisted is not None and persisted.pending_edit is not None
    assert "percent / 100" in (repository / "discount" / "calc.py").read_text(encoding="utf-8")
    result = asyncio.run(
        VeriPatchAgent(DiscountBugDemoModel(), settings=settings, store=store).resume("after-write")
    )
    assert result.state.phase is RunPhase.SUCCEEDED


@pytest.mark.parametrize("baseline", ["passed", "error"])
def test_baseline_pass_and_execution_error_are_explicit(tmp_path: Path, baseline: str) -> None:
    repository = _repository(tmp_path)
    outcome: RunnerOutcome | Exception
    if baseline == "passed":
        outcome = RunnerOutcome(
            command=["pytest"],
            exit_code=0,
            stdout="1 passed",
            stderr="",
            duration_seconds=0,
        )
    else:
        outcome = RuntimeError("runner unavailable")
    result = asyncio.run(
        VeriPatchAgent(
            FailingModel(),
            settings=_settings(tmp_path),
            runner_factory=lambda _root, _kind: FixedRunner(outcome),
        ).run(
            repo_root=repository,
            issue=_issue(),
            test_command=["pytest", "-q"],
            runner_kind=RunnerKind.LOCAL,
        )
    )
    assert result.state.phase is RunPhase.FAILED
    if baseline == "passed":
        assert "not reproduced" in (result.state.failure_reason or "")
    else:
        assert "failed to execute" in (result.state.failure_reason or "")


def test_baseline_environment_error_stops_before_model_call(tmp_path: Path) -> None:
    outcome = RunnerOutcome(
        command=["python", "-m", "pytest", "-q"],
        exit_code=1,
        stdout="",
        stderr="C:/Python/python.exe: No module named pytest",
        duration_seconds=0,
    )
    result = asyncio.run(
        VeriPatchAgent(
            FailingModel(),
            settings=_settings(tmp_path),
            runner_factory=lambda _root, _kind: FixedRunner(outcome),
        ).run(
            repo_root=_repository(tmp_path),
            issue=_issue(),
            test_command=["python", "-m", "pytest", "-q"],
            runner_kind=RunnerKind.LOCAL,
        )
    )
    assert result.state.phase is RunPhase.FAILED
    assert "环境配置失败" in (result.state.failure_reason or "")
    assert result.state.usage.model_calls == 0
    assert result.state.step == 0
    assert result.state.observations[-1].kind == "baseline_test"
    assert result.state.observations[-1].payload["classification"] == "environment_error"
