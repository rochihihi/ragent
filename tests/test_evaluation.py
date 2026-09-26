import asyncio
import json
from pathlib import Path

import pytest

from veripatch.config import Settings
from veripatch.domain import IssueSpec, RunnerKind
from veripatch.evaluation import (
    EvaluationTask,
    classify_failure,
    command_identity,
    load_tasks,
    repository_tree_hash,
    run_evaluation,
    write_markdown_report,
)
from veripatch.models.scripted import DiscountBugDemoModel


def test_load_tasks_reports_invalid_line(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    path.write_text("not-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match=":1"):
        load_tasks(path)


def test_runtime_evaluation_is_explicitly_labeled(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    task = EvaluationTask(
        task_id="discount",
        repository=str(project_root / "examples" / "discount_bug"),
        issue=IssueSpec(
            issue_id="discount",
            title="Incorrect discount",
            description="A ten percent discount on 100 should equal 90.",
        ),
        test_command=["python", "-m", "pytest", "-q"],
    )
    output = tmp_path / "result.jsonl"
    records, summary = asyncio.run(
        run_evaluation(
            [task],
            model_factory=DiscountBugDemoModel,
            settings=Settings(
                database_path=tmp_path / "unused.sqlite3",
                test_timeout_seconds=30,
                test_runner="local",
            ),
            results_path=output,
            provider="scripted-demo",
            runner=RunnerKind.LOCAL,
            benchmark_kind="runtime_validation",
        )
    )
    assert records[0].resolved
    assert records[0].benchmark_kind == "runtime_validation"
    assert summary.resolved_rate == 1.0
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["provider"] == "scripted-demo"
    assert persisted["docker_image_id"] is None

    report = tmp_path / "result.md"
    write_markdown_report(records, summary, report)
    report_text = report.read_text(encoding="utf-8")
    assert "not SWE-bench" in report_text
    assert "tree SHA-256" in report_text


def test_task_ids_hashes_and_failure_categories_are_reproducible(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        EvaluationTask(
            task_id="../outside",
            repository=str(tmp_path),
            issue=IssueSpec(issue_id="1", title="Bug", description="Broken"),
            test_command=["pytest"],
        )
    source = tmp_path / "repository"
    source.mkdir()
    (source / "app.py").write_text("value = 1\n", encoding="utf-8")
    first = repository_tree_hash(source)
    (source / ".pytest_cache").mkdir()
    (source / ".pytest_cache" / "ignored").write_text("noise", encoding="utf-8")
    assert repository_tree_hash(source) == first
    (source / "app.py").write_text("value = 2\n", encoding="utf-8")
    assert repository_tree_hash(source) != first
    assert classify_failure("Model call budget exhausted") == "budget"
    assert classify_failure("Workspace drift prevents resume") == "workspace"
    assert classify_failure(None) is None
    assert command_identity(["definitely-missing-veripatch-command"]) is None
