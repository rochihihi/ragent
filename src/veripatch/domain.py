"""Typed domain contracts shared by the agent, tools and API."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


class RunPhase(StrEnum):
    QUEUED = "queued"
    CREATED = "created"
    REPRODUCING = "reproducing"
    INVESTIGATING = "investigating"
    EDITING = "editing"
    VERIFYING = "verifying"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ActionKind(StrEnum):
    SEARCH = "search"
    READ = "read"
    LOOKUP_SYMBOL = "lookup_symbol"
    EDIT = "edit"
    RUN_TESTS = "run_tests"
    FINISH = "finish"
    FAIL = "fail"


class RunnerKind(StrEnum):
    LOCAL = "local"
    DOCKER = "docker"


class IssueSpec(BaseModel):
    issue_id: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=500)
    description: str = Field(min_length=1, max_length=20_000)


class FileEdit(BaseModel):
    path: str = Field(min_length=1, max_length=1_000)
    old_text: str = Field(max_length=100_000)
    new_text: str = Field(max_length=100_000)


class AgentDecision(BaseModel):
    """One model-selected action; irrelevant fields must remain unset."""

    action: ActionKind
    rationale: str = Field(min_length=1, max_length=4_000)
    query: str | None = Field(default=None, max_length=1_000)
    path: str | None = Field(default=None, max_length=1_000)
    start_line: int | None = Field(default=None, ge=1)
    end_line: int | None = Field(default=None, ge=1)
    edits: list[FileEdit] = Field(default_factory=list, max_length=8)
    final_summary: str | None = Field(default=None, max_length=4_000)

    @model_validator(mode="after")
    def validate_action_payload(self) -> AgentDecision:
        if self.action in {ActionKind.SEARCH, ActionKind.LOOKUP_SYMBOL} and not self.query:
            raise ValueError(f"{self.action} requires query")
        if self.action is ActionKind.READ and not self.path:
            raise ValueError("read requires path")
        if self.action is ActionKind.EDIT and not self.edits:
            raise ValueError("edit requires at least one edit")
        if self.start_line and self.end_line and self.start_line > self.end_line:
            raise ValueError("start_line cannot exceed end_line")
        return self


class Observation(BaseModel):
    kind: str
    summary: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class TestOutcome(BaseModel):
    command: list[str]
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float = Field(ge=0)
    timed_out: bool = False
    terminal_id: str | None = None
    pid: int | None = None
    launch_state: str | None = None
    window_confirmed: bool | None = None

    @property
    def passed(self) -> bool:
        return not self.timed_out and self.exit_code == 0


class UsageTotals(BaseModel):
    input_tokens: int = Field(default=0, ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    model_calls: int = Field(default=0, ge=0)


class PreparedFileChange(BaseModel):
    path: str
    before_text: str
    after_text: str
    before_sha256: str
    after_sha256: str


class EditTransaction(BaseModel):
    transaction_id: str
    edits: list[FileEdit]
    files: list[PreparedFileChange]


class AgentRunState(BaseModel):
    run_id: str
    repo_root: str
    issue: IssueSpec
    test_command: list[str]
    provider: str = "unknown"
    runner: RunnerKind = RunnerKind.LOCAL
    phase: RunPhase = RunPhase.CREATED
    step: int = 0
    observations: list[Observation] = Field(default_factory=list)
    decisions: list[AgentDecision] = Field(default_factory=list)
    changed_files: list[str] = Field(default_factory=list)
    original_files: dict[str, str] = Field(default_factory=dict)
    working_file_hashes: dict[str, str] = Field(default_factory=dict)
    action_fingerprints: list[str] = Field(default_factory=list)
    pending_edit: EditTransaction | None = None
    last_test: TestOutcome | None = None
    final_diff: str = ""
    failure_reason: str | None = None
    usage: UsageTotals = Field(default_factory=UsageTotals)
    model: str | None = None
    reasoning_effort: str = "medium"
    request_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @property
    def terminal(self) -> bool:
        return self.phase in {RunPhase.SUCCEEDED, RunPhase.FAILED}


class AgentRunResult(BaseModel):
    state: AgentRunState
    diff: str
