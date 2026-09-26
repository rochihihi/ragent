"""Typed contracts for the interactive RAgent coding agent."""

from __future__ import annotations

import shlex
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from veripatch.domain import UsageTotals, utc_now


class StudioAction(StrEnum):
    BATCH = "batch"
    LIST_FILES = "list_files"
    SEARCH = "search"
    READ = "read"
    EDIT = "edit"
    APPLY_PATCH = "apply_patch"
    CREATE = "create"
    MOVE_FILE = "move_file"
    COPY_FILE = "copy_file"
    DELETE_PATH = "delete_path"
    GIT_STATUS = "git_status"
    GIT_DIFF = "git_diff"
    GIT_LOG = "git_log"
    GIT_BRANCH = "git_branch"
    GIT_COMMIT = "git_commit"
    GIT_RESTORE = "git_restore"
    RUN_TESTS = "run_tests"
    RUN_COMMAND = "run_command"
    START_TERMINAL = "start_terminal"
    POLL_TERMINAL = "poll_terminal"
    WRITE_TERMINAL = "write_terminal"
    STOP_TERMINAL = "stop_terminal"
    INSPECT_PROCESSES = "inspect_processes"
    MCP_CALL = "mcp_call"
    REQUEST_PERMISSION = "request_permission"
    RESPOND = "respond"
    FINISH = "finish"
    FAIL = "fail"


class VerificationMode(StrEnum):
    QUICK = "quick"
    AUTO = "auto"
    STRICT = "strict"


class PermissionMode(StrEnum):
    ASK = "ask"
    IMPORTANT = "important"
    FULL = "full"


class TaskVerificationPolicy(StrEnum):
    REQUIRED_BY_USER = "required_by_user"
    ALLOWED = "allowed"
    SKIPPED_BY_USER = "skipped_by_user"
    NOT_APPLICABLE = "not_applicable"


class ResponseStyle(StrEnum):
    CONCISE = "concise"
    STANDARD = "standard"
    TEACHING = "teaching"


class PlanStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"


class StudioPlanItem(BaseModel):
    key: str
    title: str
    status: PlanStatus = PlanStatus.PENDING
    note: str | None = None


class StudioMemory(BaseModel):
    constraints: list[str] = Field(default_factory=list, max_length=40)
    facts: list[str] = Field(default_factory=list, max_length=80)
    hypotheses: list[str] = Field(default_factory=list, max_length=40)
    relevant_files: list[str] = Field(default_factory=list, max_length=80)
    failures: list[str] = Field(default_factory=list, max_length=40)


class StudioMemoryUpdate(BaseModel):
    facts: list[str] = Field(default_factory=list, max_length=12)
    hypotheses: list[str] = Field(default_factory=list, max_length=12)
    relevant_files: list[str] = Field(default_factory=list, max_length=20)


class StudioRequirement(BaseModel):
    key: str
    description: str
    expected: str | None = None
    satisfied: bool = False
    evidence: str | None = None


class StudioTaskContract(BaseModel):
    objective: str
    intent: str = "change"
    intent_confidence: str = "legacy"
    intent_rationale: str | None = None
    intent_source: str = "deterministic"
    requires_clarification: bool = False
    allowed_actions: list[str] = Field(default_factory=list, max_length=40)
    denied_actions: list[str] = Field(default_factory=list, max_length=40)
    evidence_required: bool = False
    requirements: list[StudioRequirement] = Field(default_factory=list, max_length=20)
    dialogue_act: str = "instruction"
    objectives: list[str] = Field(default_factory=list, max_length=20)
    questions: list[str] = Field(default_factory=list, max_length=20)
    requested_actions: list[str] = Field(default_factory=list, max_length=40)
    prohibited_actions: list[str] = Field(default_factory=list, max_length=40)
    conditions: list[str] = Field(default_factory=list, max_length=20)
    references: list[str] = Field(default_factory=list, max_length=40)


class StudioTaskState(BaseModel):
    """Single durable source of truth for the active task."""

    objective: str = ""
    intent: str = "answer"
    allowed_actions: list[str] = Field(default_factory=list, max_length=40)
    denied_actions: list[str] = Field(default_factory=list, max_length=40)
    current_phase: str = "understand"
    completed_actions: list[str] = Field(default_factory=list, max_length=80)
    skipped_actions: list[str] = Field(default_factory=list, max_length=40)
    verification_policy: TaskVerificationPolicy = TaskVerificationPolicy.NOT_APPLICABLE
    completion_conditions: list[str] = Field(default_factory=list, max_length=40)
    blockers: list[str] = Field(default_factory=list, max_length=40)
    replan_count: int = 0
    current_strategy: str | None = Field(default=None, max_length=2_000)


class ObservationOutcome(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED_RETRYABLE = "failed_retryable"
    FAILED_TERMINAL = "failed_terminal"
    BLOCKED_BY_POLICY = "blocked_by_policy"


class StudioObservationAssessment(BaseModel):
    outcome: ObservationOutcome
    category: str
    retryable: bool = False
    evidence: str
    next_strategy: str | None = None
    next_phase: str
    should_replan: bool = False


class FinalReviewVerdict(StrEnum):
    PASSED = "passed"
    CORRECTED = "corrected"
    BLOCKED = "blocked"


class StudioFinalReview(BaseModel):
    """Controller-owned audit of a proposed final answer."""

    verdict: FinalReviewVerdict
    goal_complete: bool
    policy_compliant: bool
    claims_grounded: bool
    changed_files_disclosed: bool
    verification_claim_grounded: bool
    blockers: list[str] = Field(default_factory=list, max_length=40)
    corrections: list[str] = Field(default_factory=list, max_length=40)
    evidence: list[str] = Field(default_factory=list, max_length=80)
    reviewed_message: str = Field(max_length=30_000)


class StudioPermissionRequest(BaseModel):
    decision: dict[str, Any] | None = None
    request_id: str
    path: str
    reason: str
    access: str = "read"
    command: list[str] = Field(default_factory=list, max_length=30)
    follow_up_command: list[str] = Field(default_factory=list, max_length=30)
    capability: str | None = Field(default=None, max_length=1_000)
    operation: str | None = Field(default=None, max_length=100)
    purpose: str | None = Field(default=None, max_length=2_000)
    impact: str | None = Field(default=None, max_length=2_000)
    scope: str | None = Field(default=None, max_length=2_000)
    recovery: str | None = Field(default=None, max_length=2_000)
    recommendation: str | None = Field(default=None, max_length=2_000)
    destructive: bool = False
    risk: str = Field(default="unknown", pattern="^(low|medium|high|unknown)$")


class StudioClaim(BaseModel):
    """An audit reference, not a user-visible answer segment."""

    kind: str = Field(pattern="^(observation|fact|inference|unknown)$")
    text: str = Field(min_length=1, max_length=4000)
    observation_id: int | None = Field(default=None, ge=0)


class StudioDecision(BaseModel):
    action: StudioAction
    rationale: str = Field(min_length=1, max_length=4_000)
    path: str | None = Field(default=None, max_length=1_000)
    access: str | None = Field(default=None, pattern="^(read|write)$")
    destination: str | None = Field(default=None, max_length=1_000)
    revision: str | None = Field(default=None, max_length=200)
    branch: str | None = Field(default=None, max_length=200)
    commit_message: str | None = Field(default=None, max_length=500)
    query: str | None = Field(default=None, max_length=1_000)
    old_text: str | None = Field(default=None, max_length=100_000)
    new_text: str | None = Field(default=None, max_length=100_000)
    content: str | None = Field(default=None, max_length=100_000)
    patch: str | None = Field(default=None, max_length=300_000)
    command: list[str] = Field(default_factory=list, max_length=30)
    terminal_id: str | None = Field(default=None, max_length=100)
    input: str | None = Field(default=None, max_length=20_000)
    mcp_tool: str | None = Field(default=None, max_length=200)
    mcp_arguments: dict[str, Any] = Field(default_factory=dict)
    message: str | None = Field(default=None, max_length=20_000)
    claims: list[StudioClaim] = Field(default_factory=list, max_length=20)
    memory_update: StudioMemoryUpdate | None = None
    actions: list[StudioDecision] = Field(default_factory=list, max_length=1000)

    @model_validator(mode="before")
    @classmethod
    def normalize_common_model_variants(cls, value: Any) -> Any:
        """Accept harmless provider formatting drift without weakening action checks."""
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        edits = normalized.get("edits")
        if isinstance(edits, list) and edits and isinstance(edits[0], dict):
            # Studio executes one audited exact edit per step. Flatten only the
            # first edit; remaining edits stay visible to the model next step.
            for key, item in edits[0].items():
                normalized.setdefault(key, item)
        elif isinstance(edits, dict):
            for key, item in edits.items():
                normalized.setdefault(key, item)
        action = normalized.get("action")
        if isinstance(action, str):
            action_key = action.strip().casefold().replace("-", "_")
            action_aliases = {
                "list": "list_files",
                "list_directory": "list_files",
                "list_repo_files": "list_files",
                "search_code": "search",
                "grep": "search",
                "read_file": "read",
                "open_file": "read",
                "edit_file": "edit",
                "replace_text": "edit",
                "patch": "apply_patch",
                "apply_diff": "apply_patch",
                "create_file": "create",
                "write_file": "create",
                "move": "move_file",
                "rename": "move_file",
                "copy": "copy_file",
                "delete": "delete_path",
                "remove_file": "delete_path",
                "git_history": "git_log",
                "git_branches": "git_branch",
                "git_rollback": "git_restore",
                "run_test": "run_tests",
                "run_pytest": "run_tests",
                "execute": "run_command",
                "execute_command": "run_command",
                "terminal_start": "start_terminal",
                "terminal_poll": "poll_terminal",
                "terminal_write": "write_terminal",
                "terminal_stop": "stop_terminal",
                "list_processes": "inspect_processes",
                "inspect_windows": "inspect_processes",
                "request_access": "request_permission",
                "ask_permission": "request_permission",
                "respond_user": "respond",
                "answer": "respond",
                "finish_task": "finish",
                "complete": "finish",
                "failure": "fail",
            }
            normalized["action"] = action_aliases.get(action_key, action_key)
        field_aliases = {
            "file_path": "path",
            "filepath": "path",
            "file": "path",
            "filename": "path",
            "target_file": "path",
            "source_path": "path",
            "destination_path": "destination",
            "target_path": "destination",
            "commit": "commit_message",
            "search_query": "query",
            "old": "old_text",
            "original": "old_text",
            "original_text": "old_text",
            "new": "new_text",
            "replacement": "new_text",
            "replacement_text": "new_text",
            "file_content": "content",
            "code": "content",
            "source": "content",
            "source_code": "content",
            "body": "content",
            "reason": "rationale",
            "explanation": "rationale",
            "final_answer": "message",
            "response": "message",
            "summary": "message",
        }
        for source, target in field_aliases.items():
            if target not in normalized and source in normalized:
                normalized[target] = normalized[source]
        if not normalized.get("rationale"):
            normalized["rationale"] = "执行下一项经过审计的操作"
        if normalized.get("action") in {"respond", "finish", "fail"} and not normalized.get(
            "message"
        ):
            normalized["message"] = normalized["rationale"]
        command = normalized.get("command")
        if isinstance(command, str):
            normalized["command"] = shlex.split(command, posix=False)
        elif command is None:
            # Providers commonly emit explicit null for schema fields that do
            # not apply to the selected action. Treat it like the field being
            # omitted; command-bearing actions are still rejected below.
            normalized["command"] = []
        for field in ("old_text", "new_text"):
            text_parts = normalized.get(field)
            if isinstance(text_parts, list) and all(isinstance(part, str) for part in text_parts):
                normalized[field] = "\n".join(text_parts)
        return normalized

    @model_validator(mode="after")
    def validate_action_fields(self) -> StudioDecision:
        if self.action is StudioAction.BATCH:
            if not self.actions:
                raise ValueError("batch requires actions")
            if any(item.action is StudioAction.BATCH for item in self.actions):
                raise ValueError("nested batch actions are not allowed")
            return self
        if self.action in {StudioAction.READ, StudioAction.EDIT} and not self.path:
            raise ValueError(f"{self.action} requires path")
        if self.action is StudioAction.SEARCH and not self.query:
            raise ValueError("search requires query")
        if self.action is StudioAction.INSPECT_PROCESSES and self.query is None:
            raise ValueError("inspect_processes requires query (use an empty string for all)")
        if self.action is StudioAction.MCP_CALL and not self.mcp_tool:
            raise ValueError("mcp_call requires mcp_tool")
        if self.action is StudioAction.EDIT and (self.old_text is None or self.new_text is None):
            raise ValueError("edit requires old_text and new_text")
        if self.action is StudioAction.CREATE and (not self.path or self.content is None):
            raise ValueError("create requires path and content")
        if self.action in {StudioAction.MOVE_FILE, StudioAction.COPY_FILE} and (
            not self.path or not self.destination
        ):
            raise ValueError(f"{self.action} requires path and destination")
        if self.action in {StudioAction.DELETE_PATH, StudioAction.GIT_RESTORE} and not self.path:
            raise ValueError(f"{self.action} requires path")
        if self.action is StudioAction.GIT_COMMIT and not self.commit_message:
            raise ValueError("git_commit requires commit_message")
        if self.action is StudioAction.APPLY_PATCH and not self.patch:
            raise ValueError("apply_patch requires patch")
        if (
            self.action
            in {
                StudioAction.RUN_TESTS,
                StudioAction.RUN_COMMAND,
                StudioAction.START_TERMINAL,
            }
            and not self.command
        ):
            raise ValueError(f"{self.action} requires command")
        if (
            self.action
            in {
                StudioAction.POLL_TERMINAL,
                StudioAction.WRITE_TERMINAL,
                StudioAction.STOP_TERMINAL,
            }
            and not self.terminal_id
        ):
            raise ValueError(f"{self.action} requires terminal_id")
        if self.action is StudioAction.WRITE_TERMINAL and self.input is None:
            raise ValueError("write_terminal requires input")
        if (
            self.action in {StudioAction.RESPOND, StudioAction.FINISH, StudioAction.FAIL}
            and not self.message
        ):
            raise ValueError(f"{self.action} requires message")
        if self.action is StudioAction.REQUEST_PERMISSION:
            if not self.path:
                raise ValueError("request_permission requires path")
            if self.access is None:
                self.access = "read"
        return self


class StudioMessage(BaseModel):
    role: str
    content: str
    created_at: datetime = Field(default_factory=utc_now)


class ToolResultStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"
    NOT_APPLICABLE = "not_applicable"


class StudioToolResult(BaseModel):
    status: ToolResultStatus
    evidence: list[str] = Field(default_factory=list, max_length=40)
    failure_category: str | None = None
    retryable: bool | None = None
    next_strategy: str | None = None
    changed_files: list[str] = Field(default_factory=list, max_length=80)


class StudioObservation(BaseModel):
    kind: str
    summary: str
    payload: dict[str, Any] = Field(default_factory=dict)
    tool_result: StudioToolResult | None = None


class StudioContextSummary(BaseModel):
    objective: str | None = None
    intent: str | None = None
    constraints: list[str] = Field(default_factory=list, max_length=40)
    completed_actions: list[str] = Field(default_factory=list, max_length=40)
    pending_steps: list[str] = Field(default_factory=list, max_length=20)
    relevant_files: list[str] = Field(default_factory=list, max_length=80)
    failures: list[str] = Field(default_factory=list, max_length=40)
    summarized_message_count: int = 0


class StudioSession(BaseModel):
    enabled_skills: list[str] = Field(default_factory=list, max_length=10)
    session_id: str
    repo_root: str
    provider: str
    model: str
    reasoning_effort: str
    response_style: ResponseStyle = ResponseStyle.CONCISE
    verification_mode: VerificationMode = VerificationMode.AUTO
    test_command: list[str] = Field(default_factory=list)
    baseline_completed: bool = False
    baseline_reproduced: bool = False
    verification_passed: bool = False
    title: str = "新编码任务"
    status: str = "idle"
    messages: list[StudioMessage] = Field(default_factory=list)
    observations: list[StudioObservation] = Field(default_factory=list)
    turn_observation_start: int = 0
    changed_files: list[str] = Field(default_factory=list)
    turn_changed_files: list[str] = Field(default_factory=list)
    steer_prior_changed_files: list[str] = Field(default_factory=list, max_length=80)
    steer_notice_delivered: bool = False
    turn_language_change_from: str | None = Field(default=None, max_length=20)
    turn_required_language_suffix: str | None = Field(default=None, max_length=20)
    task_contract: StudioTaskContract | None = None
    task_state: StudioTaskState | None = None
    usage: UsageTotals = Field(default_factory=UsageTotals)
    plan: list[StudioPlanItem] = Field(default_factory=list)
    turn_budget: int = 0
    review_completed: bool = False
    review_summary: str | None = None
    memory: StudioMemory = Field(default_factory=StudioMemory)
    context_estimated_tokens: int = 0
    context_actual_input_tokens: int | None = None
    context_trimmed_items: list[str] = Field(default_factory=list)
    context_summary: StudioContextSummary = Field(default_factory=StudioContextSummary)
    action_attempts: dict[str, int] = Field(default_factory=dict)
    recovery_attempts: dict[str, int] = Field(default_factory=dict)
    action_epoch: int = 0
    completion_rejections: dict[str, int] = Field(default_factory=dict)
    activity: str = "idle"
    pause_reason: str | None = None
    step: int = 0
    failure_reason: str | None = None
    permission_mode: PermissionMode = PermissionMode.IMPORTANT
    action_grants: list[str] = Field(default_factory=list)
    once_grants: list[str] = Field(default_factory=list)
    resume_decision: StudioDecision | None = None
    remaining_actions: list[StudioDecision] = Field(default_factory=list, max_length=8)
    approved_paths: list[str] = Field(default_factory=list, max_length=40)
    approved_write_paths: list[str] = Field(default_factory=list, max_length=40)
    approved_commands: list[list[str]] = Field(default_factory=list, max_length=40)
    approved_capabilities: list[str] = Field(default_factory=list, max_length=40)
    pending_permission: StudioPermissionRequest | None = None
    pending_model_call: dict[str, Any] | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator("reasoning_effort", mode="before")
    @classmethod
    def migrate_disabled_openai_reasoning(cls, value: Any) -> Any:
        """Keep older sessions usable after removing the misleading disabled option."""
        return "low" if value == "none" else value


class StudioReply(BaseModel):
    decision: StudioDecision
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    model: str | None = None
    protocol: str = "unknown"
    fallback_reason: str | None = None
    response_id: str | None = None
    tool_continuation: dict[str, Any] | None = None
    latency_ms: int = 0
    status_code: int | None = None


class SemanticIntentAssessment(BaseModel):
    intent: str = Field(pattern="^(answer|analysis|change|verify|launch_only|install|execute)$")
    confidence: str = Field(pattern="^(low|medium|high)$")
    requires_clarification: bool
    clarification_question: str = Field(default="", max_length=500)
    rationale: str = Field(default="", max_length=500)
    dialogue_act: str = Field(default="instruction", max_length=80)
    objectives: list[str] = Field(default_factory=list, max_length=20)
    questions: list[str] = Field(default_factory=list, max_length=20)
    requested_actions: list[str] = Field(default_factory=list, max_length=40)
    prohibited_actions: list[str] = Field(default_factory=list, max_length=40)
    conditions: list[str] = Field(default_factory=list, max_length=20)
    references: list[str] = Field(default_factory=list, max_length=40)
    corrections: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("clarification_question", "rationale", mode="before")
    @classmethod
    def normalize_absent_question(cls, value: Any) -> Any:
        # Providers commonly encode "no question" as null. This optional display
        # field must not invalidate an otherwise valid semantic decision.
        return "" if value is None else value

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value: Any) -> str:
        if isinstance(value, (int, float)):
            if value >= 0.8:
                return "high"
            if value >= 0.5:
                return "medium"
            return "low"
        return str(value).casefold()
