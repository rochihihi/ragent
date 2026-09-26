"""Deterministic pytest execution with an allowlist and timeout."""

from __future__ import annotations

import importlib
import locale
import os
import shutil
import subprocess
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Protocol

from veripatch.domain import TestOutcome


class UnsafeTestCommand(ValueError):
    pass


def baseline_environment_error(outcome: TestOutcome) -> str | None:
    """Return a user-facing setup error when pytest never produced valid bug evidence."""
    output = f"{outcome.stdout}\n{outcome.stderr}".casefold()
    if outcome.timed_out:
        return "基线测试运行超时，请先在项目目录中手动确认测试命令可以完成。"
    if "no module named pytest" in output:
        return "Python 环境缺少 pytest，请在目标仓库的虚拟环境中安装测试依赖。"
    if any(
        marker in output
        for marker in (
            "dockerdesktoplinuxengine",
            "cannot connect to the docker daemon",
            "is the docker daemon running",
            "error during connect",
        )
    ):
        return "Docker Desktop 未启动或 Docker 引擎不可用。"
    if outcome.exit_code not in {0, 1}:
        return (
            f"测试命令未能正常运行（退出码 {outcome.exit_code}），"
            "请先在项目目录中手动运行并修复测试环境。"
        )
    return None


class TestRunner(Protocol):
    def run(self, command: list[str]) -> TestOutcome:
        """Run an immutable pytest command and return deterministic evidence."""


BLOCKED_ARGUMENT_PREFIXES = (
    "--basetemp",
    "--rootdir",
    "--confcutdir",
    "--override-ini",
)
BLOCKED_ARGUMENTS = {"-c", "--override-ini"}


def validate_pytest_command(command: list[str]) -> None:
    if not command:
        raise UnsafeTestCommand("Test command cannot be empty")
    normalized = [part.casefold() for part in command]
    executable_name = Path(command[0]).name.casefold()
    valid_prefix = executable_name in {"pytest", "pytest.exe"} or (
        len(normalized) >= 3
        and executable_name in {"python", "python.exe", "py", "py.exe"}
        and normalized[1:3] == ["-m", "pytest"]
    )
    if not valid_prefix:
        raise UnsafeTestCommand("Only pytest or python -m pytest commands are allowed")
    for argument in command[1:]:
        if argument in BLOCKED_ARGUMENTS or argument.startswith(BLOCKED_ARGUMENT_PREFIXES):
            raise UnsafeTestCommand(f"Blocked pytest argument: {argument}")
        path_like = argument.split("::", 1)[0]
        if Path(path_like).is_absolute() or ".." in Path(path_like).parts:
            raise UnsafeTestCommand(f"Test target escapes repository: {argument}")


def _sanitized_environment() -> dict[str, str]:
    blocked_fragments = (
        "TOKEN",
        "SECRET",
        "PASSWORD",
        "API_KEY",
        "ACCESS_KEY",
        "PRIVATE_KEY",
        "CREDENTIAL",
        "AUTHORIZATION",
        "COOKIE",
    )
    return {
        key: value
        for key, value in os.environ.items()
        if not any(fragment in key.upper() for fragment in blocked_fragments)
    }


def _local_python(root: Path) -> str:
    """Resolve Python without mistaking a frozen VeriPatch executable for Python."""
    if not getattr(sys, "frozen", False):
        return sys.executable
    candidates = (
        root / ".venv" / "Scripts" / "python.exe",
        root / "venv" / "Scripts" / "python.exe",
        root / ".venv" / "bin" / "python",
        root / "venv" / "bin" / "python",
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    resolved = shutil.which("py") or shutil.which("python") or shutil.which("python3")
    if resolved is None:
        raise RuntimeError(
            "Python was not found. Create a .venv in the repository or install Python."
        )
    return resolved


class LocalPytestRunner:
    def __init__(self, root: Path, *, timeout_seconds: int = 120) -> None:
        self.root = root.resolve()
        self.timeout_seconds = timeout_seconds

    def run(self, command: list[str]) -> TestOutcome:
        validate_pytest_command(command)
        command = list(command)
        if Path(command[0]).name.casefold() in {"python", "python.exe"}:
            command[0] = _local_python(self.root)
        started = time.perf_counter()
        console_encoding = locale.getpreferredencoding(False)
        try:
            completed = subprocess.run(
                command,
                cwd=self.root,
                env=_sanitized_environment(),
                capture_output=True,
                text=True,
                encoding=console_encoding,
                errors="replace",
                timeout=self.timeout_seconds,
                shell=False,
                check=False,
            )
            return TestOutcome(
                command=command,
                exit_code=completed.returncode,
                stdout=completed.stdout[-20_000:],
                stderr=completed.stderr[-20_000:],
                duration_seconds=time.perf_counter() - started,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = (
                exc.stdout.decode(console_encoding, "replace")
                if isinstance(exc.stdout, bytes)
                else (exc.stdout or "")
            )
            stderr = (
                exc.stderr.decode(console_encoding, "replace")
                if isinstance(exc.stderr, bytes)
                else (exc.stderr or "")
            )
            return TestOutcome(
                command=command,
                exit_code=124,
                stdout=stdout[-20_000:],
                stderr=stderr[-20_000:],
                duration_seconds=time.perf_counter() - started,
                timed_out=True,
            )


class InProcessDemoPytestRunner:
    """Run the bundled, trusted demo through pytest inside the frozen desktop process."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def run(self, command: list[str]) -> TestOutcome:
        validate_pytest_command(command)
        import pytest

        arguments = list(command)
        executable = Path(arguments[0]).name.casefold()
        arguments = (
            arguments[3:]
            if executable in {"python", "python.exe", "py", "py.exe"}
            else arguments[1:]
        )
        # PyInstaller's windowed executable exposes StringIO-like standard streams
        # without ``fileno``.  Pytest's faulthandler plugin requires a real file
        # descriptor, so disable only that plugin for this trusted bundled demo.
        arguments = ["-p", "no:faulthandler", *arguments]
        package_names = {child.name for child in self.root.iterdir() if child.is_dir()}
        for name, module in list(sys.modules.items()):
            if name.split(".", 1)[0] in package_names or name.startswith("test_"):
                sys.modules.pop(name, None)
                continue
            module_path = getattr(module, "__file__", None)
            if module_path is None:
                continue
            try:
                Path(module_path).resolve().relative_to(self.root)
            except ValueError:
                continue
            sys.modules.pop(name, None)
        for cache in self.root.rglob("__pycache__"):
            shutil.rmtree(cache, ignore_errors=True)
        importlib.invalidate_caches()
        stdout = StringIO()
        stderr = StringIO()
        started = time.perf_counter()
        previous_directory = Path.cwd()
        sys.path.insert(0, str(self.root))
        try:
            os.chdir(self.root)
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = int(pytest.main(arguments))
        finally:
            os.chdir(previous_directory)
            if sys.path and sys.path[0] == str(self.root):
                sys.path.pop(0)
        return TestOutcome(
            command=command,
            exit_code=exit_code,
            stdout=stdout.getvalue()[-20_000:],
            stderr=stderr.getvalue()[-20_000:],
            duration_seconds=time.perf_counter() - started,
        )


def studio_pytest_runner(root: Path) -> TestRunner:
    """Select a pytest runner that also works inside the frozen desktop app."""
    if getattr(sys, "frozen", False):
        return InProcessDemoPytestRunner(root)
    return LocalPytestRunner(root)


class DockerPytestRunner:
    """Run pytest in a resource-bounded, network-disabled Docker container."""

    def __init__(
        self,
        root: Path,
        *,
        image: str = "veripatch-sandbox:latest",
        timeout_seconds: int = 120,
        cpus: str = "1.0",
        memory: str = "512m",
        pids: int = 128,
        docker_executable: str = "docker",
    ) -> None:
        self.root = root.resolve()
        self.image = image
        self.timeout_seconds = timeout_seconds
        self.cpus = cpus
        self.memory = memory
        self.pids = pids
        self.docker_executable = docker_executable

    def build_command(self, command: list[str]) -> list[str]:
        validate_pytest_command(command)
        executable = Path(command[0]).name.casefold()
        pytest_arguments = command[3:] if executable in {"python", "python.exe"} else command[1:]
        mount = f"type=bind,source={self.root},target=/workspace,readonly"
        return [
            self.docker_executable,
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
            "--cpus",
            self.cpus,
            "--memory",
            self.memory,
            "--pids-limit",
            str(self.pids),
            "--security-opt",
            "no-new-privileges",
            "--cap-drop",
            "ALL",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1",
            "--env",
            "PYTEST_ADDOPTS=-p no:cacheprovider",
            "--mount",
            mount,
            "--workdir",
            "/workspace",
            self.image,
            "python",
            "-m",
            "pytest",
            *pytest_arguments,
        ]

    def run(self, command: list[str]) -> TestOutcome:
        docker_command = self.build_command(command)
        started = time.perf_counter()
        console_encoding = locale.getpreferredencoding(False)
        try:
            completed = subprocess.run(
                docker_command,
                cwd=self.root,
                env=_sanitized_environment(),
                capture_output=True,
                text=True,
                encoding=console_encoding,
                errors="replace",
                timeout=self.timeout_seconds,
                shell=False,
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("Docker executable was not found") from exc
        except subprocess.TimeoutExpired as exc:
            stdout = (
                exc.stdout.decode(console_encoding, "replace")
                if isinstance(exc.stdout, bytes)
                else (exc.stdout or "")
            )
            stderr = (
                exc.stderr.decode(console_encoding, "replace")
                if isinstance(exc.stderr, bytes)
                else (exc.stderr or "")
            )
            return TestOutcome(
                command=command,
                exit_code=124,
                stdout=stdout[-20_000:],
                stderr=stderr[-20_000:],
                duration_seconds=time.perf_counter() - started,
                timed_out=True,
            )
        return TestOutcome(
            command=command,
            exit_code=completed.returncode,
            stdout=completed.stdout[-20_000:],
            stderr=completed.stderr[-20_000:],
            duration_seconds=time.perf_counter() - started,
        )


PytestRunner = LocalPytestRunner
