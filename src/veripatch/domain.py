"""Typed domain contracts shared by the agent, tools and API."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(UTC)


class FileEdit(BaseModel):
    path: str = Field(min_length=1, max_length=1_000)
    old_text: str = Field(max_length=100_000)
    new_text: str = Field(max_length=100_000)


class TestOutcome(BaseModel):
    command: list[str]
    exit_code: int | None
    stdout: str
    stderr: str
    duration_seconds: float = Field(ge=0)
    timed_out: bool = False
    terminal_id: str | None = None
    pid: int | None = None
    launch_state: str | None = None
    window_confirmed: bool | None = None
    sandboxed: bool = False
    execution_mode: str | None = None

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
