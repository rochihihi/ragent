import asyncio
import shutil
import sys
from pathlib import Path

from veripatch.agent import VeriPatchAgent
from veripatch.config import Settings
from veripatch.domain import IssueSpec, RunnerKind, RunPhase
from veripatch.models.scripted import DiscountBugDemoModel
from veripatch.store import SQLiteRunStore


def test_agent_repairs_frozen_discount_bug(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    repository = tmp_path / "repository"
    shutil.copytree(project_root / "examples" / "discount_bug", repository)
    settings = Settings(
        max_steps=6,
        test_timeout_seconds=30,
        database_path=tmp_path / "runs.sqlite3",
    )
    store = SQLiteRunStore(settings.database_path)
    agent = VeriPatchAgent(DiscountBugDemoModel(), settings=settings, store=store)
    result = asyncio.run(
        agent.run(
            repo_root=repository,
            issue=IssueSpec(
                issue_id="discount-percentage",
                title="Incorrect percentage calculation",
                description="A 10 percent discount on 100 should be 90.",
            ),
            test_command=[sys.executable, "-m", "pytest", "-q"],
            runner_kind=RunnerKind.LOCAL,
            provider="scripted-demo",
        )
    )
    assert result.state.phase is RunPhase.SUCCEEDED
    assert result.state.changed_files == ["discount/calc.py"]
    assert "percent / 100" in result.diff
    persisted = store.load(result.state.run_id)
    assert persisted is not None
    assert persisted.phase is RunPhase.SUCCEEDED
