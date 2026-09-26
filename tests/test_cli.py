import argparse
import asyncio
import shutil
from pathlib import Path

import pytest

from veripatch import cli
from veripatch.config import Settings
from veripatch.domain import AgentRunState, IssueSpec
from veripatch.quota import QuotaSnapshot
from veripatch.store import SQLiteRunStore


def test_parser_exposes_all_public_commands() -> None:
    parser = cli.build_parser()
    arguments = {
        "demo": [],
        "run": ["--repo", ".", "--title", "Bug", "--description", "Broken"],
        "resume": ["run-1"],
        "inspect": ["run-1"],
        "eval": ["--tasks", "tasks.jsonl"],
        "serve": [],
        "auth": ["status", "deepseek"],
        "quota": ["status", "deepseek"],
    }
    for command, command_arguments in arguments.items():
        namespace = parser.parse_args([command, *command_arguments])
        assert namespace.command == command


def test_cli_demo_runs_complete_offline_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    fake_root = tmp_path / "project"
    shutil.copytree(project_root / "examples", fake_root / "examples")
    monkeypatch.setattr(cli, "_project_root", lambda: fake_root)
    exit_code = asyncio.run(
        cli._run_demo(
            Settings(
                database_path=tmp_path / "demo.sqlite3",
                test_timeout_seconds=30,
                test_runner="local",
            )
        )
    )
    assert exit_code == 0
    assert "percent / 100" in capsys.readouterr().out


def test_cli_inspect_prints_persisted_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    database = tmp_path / "inspect.sqlite3"
    store = SQLiteRunStore(database)
    store.checkpoint(
        AgentRunState(
            run_id="inspect-me",
            repo_root=str(tmp_path),
            issue=IssueSpec(issue_id="1", title="Bug", description="Broken behavior"),
            test_command=["python", "-m", "pytest"],
        )
    )
    monkeypatch.setenv("VERIPATCH_DATABASE_PATH", str(database))
    monkeypatch.setattr("sys.argv", ["veripatch", "inspect", "inspect-me"])
    cli.main()
    output = capsys.readouterr().out
    assert "inspect-me" in output
    assert "Broken behavior" in output


def test_run_repository_wrapper_uses_requested_runner(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    repository = tmp_path / "repository"
    shutil.copytree(project_root / "examples" / "discount_bug", repository)
    args = argparse.Namespace(
        issue_id="discount",
        title="Incorrect discount",
        description="A ten percent discount on 100 should equal 90.",
        provider="scripted-demo",
        runner="local",
        repo=str(repository),
        pytest_arg=["-q"],
    )
    exit_code = asyncio.run(
        cli._run_repository(
            args,
            Settings(
                database_path=tmp_path / "run.sqlite3",
                test_timeout_seconds=30,
                test_runner="local",
            ),
        )
    )
    assert exit_code == 0
    assert '"runner": "local"' in capsys.readouterr().out


def test_cli_auth_status_and_login_do_not_echo_secret(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    saved: list[tuple[str, str]] = []
    monkeypatch.setattr("sys.argv", ["veripatch", "auth", "login", "deepseek"])
    monkeypatch.setattr(cli.getpass, "getpass", lambda _: "ds-private-value")
    monkeypatch.setattr(cli, "save_api_key", lambda provider, key: saved.append((provider, key)))
    cli.main()
    output = capsys.readouterr().out
    assert saved == [("deepseek", "ds-private-value")]
    assert "ds-private-value" not in output

    monkeypatch.setattr("sys.argv", ["veripatch", "auth", "status", "deepseek"])
    monkeypatch.setattr(
        cli,
        "credential_status",
        lambda: {"deepseek": {"configured": True, "source": "keyring"}},
    )
    cli.main()
    assert '"configured": true' in capsys.readouterr().out


def test_cli_quota_status_prints_safe_snapshot(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr("sys.argv", ["veripatch", "quota", "status", "deepseek"])
    monkeypatch.setattr(
        cli,
        "get_provider_quota",
        lambda provider, settings: QuotaSnapshot(
            provider=provider,
            supported=True,
            configured=True,
            is_available=True,
            low_balance=False,
        ),
    )
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 0
    assert '"is_available": true' in capsys.readouterr().out
