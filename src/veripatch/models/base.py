"""Provider-independent model protocol."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field

from veripatch.domain import AgentDecision, IssueSpec, Observation


class ModelContext(BaseModel):
    issue: IssueSpec
    step: int = Field(ge=0)
    changed_files: list[str]
    index_summary: dict[str, int]
    recent_observations: list[Observation]
    last_diff: str = ""


class ModelReply(BaseModel):
    decision: AgentDecision
    input_tokens: int = Field(default=0, ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    model: str | None = None
    request_id: str | None = None


class AgentModel(Protocol):
    async def decide(self, context: ModelContext) -> ModelReply:
        """Select exactly one typed action for the next agent step."""
