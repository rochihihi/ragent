"""Reproducible evaluation task execution and metric aggregation."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from veripatch.agent import VeriPatchAgent
from veripatch.config import Settings
from veripatch.domain import IssueSpec, RunnerKind, RunPhase
from veripatch.models.base import AgentModel
from veripatch.store import SQLiteRunStore


class EvaluationTask(BaseModel):
    task_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")
    repository: str
    issue: IssueSpec
    test_command: list[str]
    category: str = "uncategorized"

    @field_validator("task_id")
    @classmethod
    def reject_parent_segments(cls, value: str) -> str:
        if ".." in value:
            raise ValueError("task_id cannot contain '..'")
        return value


class EvaluationRecord(BaseModel):
    task_id: str
    run_id: str
    benchmark_kind: str
    provider: str
    model: str
    reasoning_effort: str
    runner: RunnerKind
    task_hash: str
    docker_image: str
    docker_image_id: str | None
    source_git_sha: str | None
    max_steps: int
    max_model_calls: int
    max_input_tokens: int
    max_output_tokens: int
    category: str
    resolved: bool
    phase: RunPhase
    steps: int = Field(ge=0)
    model_calls: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    cached_input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    reasoning_tokens: int = Field(ge=0)
    duration_seconds: float = Field(ge=0)
    changed_files: list[str]
    failure_reason: str | None = None
    failure_category: str | None = None


class EvaluationSummary(BaseModel):
    tasks: int
    resolved: int
    resolved_rate: float
    average_steps: float
    average_duration_seconds: float
    average_model_calls: float
    total_input_tokens: int
    total_cached_input_tokens: int
    total_output_tokens: int
    total_reasoning_tokens: int
    failure_counts: dict[str, int]


def repository_tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    ignored = {".git", ".pytest_cache", "__pycache__", ".ruff_cache"}
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or any(part in ignored for part in path.relative_to(root).parts):
            continue
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def classify_failure(reason: str | None) -> str | None:
    if reason is None:
        return None
    lowered = reason.casefold()
    if "baseline" in lowered or "not reproduced" in lowered:
        return "baseline"
    if "budget" in lowered or "identical actions" in lowered:
        return "budget"
    if "model" in lowered or "api" in lowered:
        return "provider"
    if "drift" in lowered or "pending edit" in lowered:
        return "workspace"
    if "test" in lowered or "docker" in lowered:
        return "verification"
    return "agent"


def command_identity(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value or None


def git_revision() -> str | None:
    project_root = Path(__file__).resolve().parents[2]
    return command_identity(["git", "-C", str(project_root), "rev-parse", "HEAD"])


def docker_image_identity(image: str) -> str | None:
    return command_identity(["docker", "image", "inspect", "--format", "{{.Id}}", image])


def load_tasks(path: Path) -> list[EvaluationTask]:
    tasks: list[EvaluationTask] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            tasks.append(EvaluationTask.model_validate_json(line))
        except Exception as exc:
            raise ValueError(f"Invalid task at {path}:{line_number}: {exc}") from exc
    return tasks


async def run_evaluation(
    tasks: Iterable[EvaluationTask],
    *,
    model_factory: Callable[[], AgentModel],
    settings: Settings,
    results_path: Path,
    provider: str,
    runner: RunnerKind,
    benchmark_kind: str,
) -> tuple[list[EvaluationRecord], EvaluationSummary]:
    records: list[EvaluationRecord] = []
    results_path.parent.mkdir(parents=True, exist_ok=True)
    source_git_sha = git_revision()
    docker_image_id = (
        docker_image_identity(settings.docker_image) if runner is RunnerKind.DOCKER else None
    )
    with tempfile.TemporaryDirectory(prefix="veripatch-eval-") as temp_directory:
        temp_root = Path(temp_directory)
        for task in tasks:
            source = Path(task.repository).resolve()
            if not source.is_dir():
                raise ValueError(f"Task repository does not exist: {source}")
            target = temp_root / task.task_id
            task_hash = repository_tree_hash(source)
            shutil.copytree(source, target)
            test_command = list(task.test_command)
            if Path(test_command[0]).name.casefold() in {"python", "python.exe"}:
                test_command[0] = sys.executable
            task_store = SQLiteRunStore(temp_root / f"{task.task_id}.sqlite3")
            agent = VeriPatchAgent(model_factory(), settings=settings, store=task_store)
            started = time.perf_counter()
            result = await agent.run(
                repo_root=target,
                issue=task.issue,
                test_command=test_command,
                provider=provider,
                runner_kind=runner,
            )
            duration = time.perf_counter() - started
            state = result.state
            record = EvaluationRecord(
                task_id=task.task_id,
                run_id=state.run_id,
                benchmark_kind=benchmark_kind,
                provider=provider,
                model=state.model
                or (settings.deepseek_model if provider == "deepseek" else settings.model),
                reasoning_effort=settings.reasoning_effort,
                runner=runner,
                task_hash=task_hash,
                docker_image=settings.docker_image,
                docker_image_id=docker_image_id,
                source_git_sha=source_git_sha,
                max_steps=settings.max_steps,
                max_model_calls=settings.max_model_calls,
                max_input_tokens=settings.max_input_tokens,
                max_output_tokens=settings.max_output_tokens,
                category=task.category,
                resolved=state.phase is RunPhase.SUCCEEDED,
                phase=state.phase,
                steps=state.step,
                model_calls=state.usage.model_calls,
                input_tokens=state.usage.input_tokens,
                cached_input_tokens=state.usage.cached_input_tokens,
                output_tokens=state.usage.output_tokens,
                reasoning_tokens=state.usage.reasoning_tokens,
                duration_seconds=duration,
                changed_files=state.changed_files,
                failure_reason=state.failure_reason,
                failure_category=classify_failure(state.failure_reason),
            )
            records.append(record)

    results_path.write_text(
        "".join(record.model_dump_json() + "\n" for record in records), encoding="utf-8"
    )
    resolved = sum(record.resolved for record in records)
    count = len(records)
    summary = EvaluationSummary(
        tasks=count,
        resolved=resolved,
        resolved_rate=resolved / count if count else 0.0,
        average_steps=sum(record.steps for record in records) / count if count else 0.0,
        average_duration_seconds=(
            sum(record.duration_seconds for record in records) / count if count else 0.0
        ),
        average_model_calls=(
            sum(record.model_calls for record in records) / count if count else 0.0
        ),
        total_input_tokens=sum(record.input_tokens for record in records),
        total_cached_input_tokens=sum(record.cached_input_tokens for record in records),
        total_output_tokens=sum(record.output_tokens for record in records),
        total_reasoning_tokens=sum(record.reasoning_tokens for record in records),
        failure_counts=dict(
            Counter(
                record.failure_category for record in records if record.failure_category is not None
            )
        ),
    )
    return records, summary


def write_markdown_report(
    records: list[EvaluationRecord], summary: EvaluationSummary, path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# VeriPatch Benchmark Report",
        "",
        "> This is a curated repository benchmark, not SWE-bench.",
        "",
        f"- Tasks: {summary.tasks}",
        f"- Resolved: {summary.resolved}",
        f"- Resolved rate: {summary.resolved_rate:.1%}",
        f"- Average steps: {summary.average_steps:.2f}",
        f"- Average duration: {summary.average_duration_seconds:.2f}s",
        f"- Input/output/reasoning tokens: {summary.total_input_tokens} / "
        f"{summary.total_output_tokens} / {summary.total_reasoning_tokens}",
        "",
        "| Task | Category | Provider / model | Result | Steps | Duration | Failure |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for record in records:
        lines.append(
            f"| {record.task_id} | {record.category} | {record.provider} / {record.model} | "
            f"{'resolved' if record.resolved else 'failed'} | {record.steps} | "
            f"{record.duration_seconds:.2f}s | {record.failure_category or '-'} |"
        )
    lines.extend(["", "## Reproducibility", ""])
    if records:
        lines.append(f"- Source Git SHA: `{records[0].source_git_sha or 'unavailable'}`")
        lines.append(
            f"- Docker image: `{records[0].docker_image}` "
            f"(`{records[0].docker_image_id or 'unavailable'}`)"
        )
    for record in records:
        lines.append(f"- `{record.task_id}` tree SHA-256: `{record.task_hash}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def task_to_json(task: EvaluationTask) -> str:
    return json.dumps(task.model_dump(mode="json"), ensure_ascii=False)
