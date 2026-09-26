import shutil
from pathlib import Path
from subprocess import CompletedProcess, TimeoutExpired

import pytest

from veripatch.domain import TestOutcome as RunnerOutcome
from veripatch.testing import (
    DockerPytestRunner,
    InProcessDemoPytestRunner,
    LocalPytestRunner,
    UnsafeTestCommand,
    _local_python,
    _sanitized_environment,
    baseline_environment_error,
    studio_pytest_runner,
    validate_pytest_command,
)


def test_accepts_python_pytest_command() -> None:
    validate_pytest_command(["python", "-m", "pytest", "-q", "tests"])


def test_studio_uses_embedded_pytest_when_frozen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("veripatch.testing.sys.frozen", True, raising=False)

    assert isinstance(studio_pytest_runner(tmp_path), InProcessDemoPytestRunner)


@pytest.mark.parametrize(
    "command",
    [
        ["powershell", "-Command", "pytest"],
        ["python", "-m", "pytest", "../outside"],
        ["pytest", "--basetemp=C:/temp"],
        ["pytest", "--override-ini=python_files=app.py"],
        [str(Path("C:/Python/python.exe")), "-c", "print('unsafe')"],
    ],
)
def test_rejects_unsafe_test_commands(command: list[str]) -> None:
    with pytest.raises(UnsafeTestCommand):
        validate_pytest_command(command)


def test_sanitized_environment_removes_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("SAFE_VALUE", "visible")
    environment = _sanitized_environment()
    assert "OPENAI_API_KEY" not in environment
    assert "DEEPSEEK_API_KEY" not in environment
    assert "AWS_SECRET_ACCESS_KEY" not in environment
    assert environment["SAFE_VALUE"] == "visible"


@pytest.mark.parametrize(
    ("exit_code", "stderr", "expected"),
    [
        (1, "No module named pytest", "缺少 pytest"),
        (127, "docker: error during connect: dockerDesktopLinuxEngine", "Docker Desktop"),
        (4, "pytest: error: file not found", "退出码 4"),
    ],
)
def test_baseline_environment_errors_are_classified(
    exit_code: int, stderr: str, expected: str
) -> None:
    outcome = RunnerOutcome(
        command=["python", "-m", "pytest"],
        exit_code=exit_code,
        stdout="",
        stderr=stderr,
        duration_seconds=0,
    )
    assert expected in (baseline_environment_error(outcome) or "")


def test_real_pytest_failure_is_valid_bug_evidence() -> None:
    outcome = RunnerOutcome(
        command=["pytest"], exit_code=1, stdout="1 failed", stderr="", duration_seconds=0
    )
    assert baseline_environment_error(outcome) is None


def test_docker_command_has_security_boundaries(tmp_path: Path) -> None:
    runner = DockerPytestRunner(tmp_path, image="veripatch-test", cpus="0.5", memory="256m")
    command = runner.build_command(["python", "-m", "pytest", "-q", "tests"])
    joined = " ".join(command)
    assert "--network none" in joined
    assert "--read-only" in command
    assert "--security-opt no-new-privileges" in joined
    assert "--cap-drop ALL" in joined
    assert "--cpus 0.5" in joined
    assert "--memory 256m" in joined
    assert "target=/workspace" in joined
    assert "readonly" in joined
    assert command[-5:] == ["python", "-m", "pytest", "-q", "tests"]


def test_docker_runner_captures_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args, **kwargs):
        return CompletedProcess(args=args[0], returncode=0, stdout="2 passed", stderr="")

    monkeypatch.setattr("veripatch.testing.subprocess.run", fake_run)
    outcome = DockerPytestRunner(tmp_path).run(["python", "-m", "pytest", "-q"])
    assert outcome.passed
    assert outcome.stdout == "2 passed"


def test_docker_runner_reports_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args, **kwargs):
        raise TimeoutExpired(args[0], timeout=1, output=b"partial", stderr=b"slow")

    monkeypatch.setattr("veripatch.testing.subprocess.run", fake_run)
    outcome = DockerPytestRunner(tmp_path, timeout_seconds=1).run(["python", "-m", "pytest"])
    assert outcome.timed_out
    assert outcome.exit_code == 124
    assert outcome.stdout == "partial"


def test_local_runner_reports_timeout_with_text_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(*args, **kwargs):
        raise TimeoutExpired(args[0], timeout=1, output="partial", stderr="slow")

    monkeypatch.setattr("veripatch.testing.subprocess.run", fake_run)
    outcome = LocalPytestRunner(tmp_path, timeout_seconds=1).run(["python", "-m", "pytest", "-q"])
    assert outcome.timed_out
    assert outcome.stdout == "partial"
    assert outcome.stderr == "slow"


def test_frozen_runner_prefers_repository_virtualenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    python = tmp_path / ".venv" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.touch()
    monkeypatch.setattr("veripatch.testing.sys.frozen", True, raising=False)
    assert _local_python(tmp_path) == str(python)


def test_frozen_runner_never_uses_its_own_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("veripatch.testing.sys.frozen", True, raising=False)
    monkeypatch.setattr("veripatch.testing.sys.executable", "C:/VeriPatch.exe")
    monkeypatch.setattr("veripatch.testing.shutil.which", lambda name: "C:/Python/python.exe")
    assert _local_python(tmp_path) == "C:/Python/python.exe"


def test_in_process_demo_runner_executes_bundled_pytest(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "examples" / "discount_bug"
    repository = tmp_path / "demo"
    shutil.copytree(source, repository)
    runner = InProcessDemoPytestRunner(repository)
    failing = runner.run(["python", "-m", "pytest", "-q"])
    assert failing.exit_code == 1
    calculator = repository / "discount" / "calc.py"
    calculator.write_text(
        calculator.read_text(encoding="utf-8").replace(
            "return price * (1 - percent)", "return price * (1 - percent / 100)"
        ),
        encoding="utf-8",
    )
    passing = runner.run(["python", "-m", "pytest", "-q"])
    assert passing.passed


def test_docker_runner_explains_missing_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(*args, **kwargs):
        raise FileNotFoundError("docker")

    monkeypatch.setattr("veripatch.testing.subprocess.run", fake_run)
    with pytest.raises(RuntimeError, match="Docker executable"):
        DockerPytestRunner(tmp_path).run(["pytest", "-q"])
