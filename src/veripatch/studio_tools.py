"""Project detection and constrained command execution for Studio."""

from __future__ import annotations

import locale
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from typing import TextIO, TypedDict
from uuid import uuid4

from veripatch.domain import TestOutcome
from veripatch.testing import _sanitized_environment


class UnsafeStudioCommand(ValueError):
    pass


def resolve_studio_executable(name: str) -> str | None:
    """Resolve tools installed during this run even if the parent's PATH is stale."""
    discovered = shutil.which(name)
    if discovered:
        return discovered
    if name.casefold() in {"go", "go.exe"}:
        installed = Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")) / "Go" / "bin" / "go.exe"
        if installed.is_file():
            return str(installed)
    return None


class CommandPermissionDetails(TypedDict):
    purpose: str
    impact: str
    scope: str
    recovery: str
    recommendation: str
    risk: str
    destructive: bool


def command_permission_details(
    command: list[str], rationale: str | None = None
) -> CommandPermissionDetails:
    """Display the proposed reason, never infer side effects from command text."""
    return {
        "purpose": f"模型提供的执行理由（未独立核实）：{rationale.strip()}"
        if rationale and rationale.strip()
        else "模型未提供执行理由；请核对下方完整命令。",
        "impact": "该命令将以当前用户权限执行；没有进程沙箱，可能读写文件、访问网络或启动程序。",
        "scope": "授权对象是显示的完整命令；工作目录不是文件访问边界，实际影响范围未确认。",
        "recovery": "未确认副作用及其可恢复性；执行前请确认目标并保留必要备份。",
        "recommendation": "请结合完整命令、工作目录和执行理由决定；不确定时拒绝并要求说明或缩小范围。",
        "risk": "unknown",
        "destructive": False,  # Compatibility field, not a finding of non-destructiveness.
    }


def describe_command(command: list[str]) -> tuple[str, str, str]:
    """Compatibility presentation helper; command text is not safety evidence."""
    details = command_permission_details(command)
    return details["purpose"], details["impact"], details["risk"]


def is_detached_launch(command: list[str]) -> bool:
    """Recognize Windows commands whose child must outlive the agent call."""
    lowered = [part.casefold() for part in command]
    if lowered[:3] == ["cmd", "/c", "start"]:
        return True
    executable = Path(lowered[0]).name if lowered else ""
    if executable in {"go", "go.exe"} and len(lowered) >= 3:
        return lowered[1] == "run" and lowered[-1].endswith(".go")
    if executable in {"powershell", "powershell.exe", "pwsh", "pwsh.exe"}:
        script = " ".join(lowered[1:])
        if "start-process" not in script:
            return False
        # A compound build/copy/install script must run to completion so its
        # real exit code and output are captured before launch is reported.
        return not any(
            marker in script
            for marker in (
                "new-item",
                "copy-item",
                "move-item",
                "remove-item",
                "go build",
                "winget ",
                " install ",
            )
        )
    return executable in {"pythonw", "pythonw.exe"} and any(
        part.endswith(".py") for part in lowered[1:]
    )


def command_capability(command: list[str], root: Path) -> str | None:
    """Return a deliberately narrow permission shared by equivalent GUI launch commands."""
    if not is_detached_launch(command):
        return None
    joined = " ".join(command)
    candidates = re.findall(r"[^\s'\"]+\.(?:py|html?|go)", joined, flags=re.IGNORECASE)
    if not candidates:
        return None
    raw_target = candidates[-1].rstrip(",;)")
    target = Path(raw_target)
    try:
        resolved = target.resolve() if target.is_absolute() else (root.resolve() / target).resolve()
        relative = resolved.relative_to(root.resolve()).as_posix().casefold()
    except (OSError, ValueError):
        return None
    return f"launch:{relative}"


STACK_MARKERS = {
    "pyproject.toml": "Python",
    "requirements.txt": "Python",
    "package.json": "Node.js",
    "pnpm-lock.yaml": "Node.js",
    "Cargo.toml": "Rust",
    "go.mod": "Go",
    "pom.xml": "Java",
    "build.gradle": "Java",
    "build.gradle.kts": "Kotlin",
    "*.sln": ".NET",
}


def detect_project(root: Path) -> dict[str, object]:
    stacks: list[str] = []
    markers: list[str] = []
    for pattern, stack in STACK_MARKERS.items():
        matches = list(root.glob(pattern))
        if not matches:
            continue
        markers.extend(path.name for path in matches[:3])
        if stack not in stacks:
            stacks.append(stack)
    return {"stacks": stacks or ["Unknown"], "markers": markers}


def validate_studio_command(command: list[str]) -> None:
    """Validate argv structure only; neither authorize nor infer command safety."""
    if not isinstance(command, list) or not command:
        raise UnsafeStudioCommand("Command must be a nonempty argv list")
    if any(not isinstance(argument, str) for argument in command):
        raise UnsafeStudioCommand("Command arguments must be strings")
    if not command[0].strip():
        raise UnsafeStudioCommand("Command executable cannot be empty")
    if any("\x00" in argument for argument in command):
        raise UnsafeStudioCommand("Command arguments cannot contain NUL")


class ManagedTerminal:
    def __init__(self, terminal_id: str, root: Path, process: subprocess.Popen[str]) -> None:
        self.terminal_id = terminal_id
        self.root = root
        self.process = process
        self.output: Queue[str] = Queue()
        self.history = ""
        self.window_confirmed = False
        self.launch_tracked = False
        for stream in (getattr(process, "stdout", None), getattr(process, "stderr", None)):
            if stream is not None:
                threading.Thread(target=self._read_stream, args=(stream,), daemon=True).start()

    def _read_stream(self, stream: TextIO) -> None:
        for line in stream:
            self.output.put(str(line))

    def poll(self) -> dict[str, object]:
        chunks: list[str] = []
        while True:
            try:
                chunks.append(self.output.get_nowait())
            except Empty:
                break
        text = "".join(chunks)
        self.history = (self.history + text)[-40_000:]
        exit_code = self.process.poll()
        return {
            "terminal_id": self.terminal_id,
            "running": exit_code is None,
            "exit_code": exit_code,
            "output": text[-20_000:],
        }


class TerminalRegistry:
    """In-process registry for auditable long-running command sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, ManagedTerminal] = {}
        self._lock = threading.Lock()

    def start(
        self,
        root: Path,
        command: list[str],
        *,
        approved_commands: list[list[str]] | None = None,
        approved_capabilities: list[str] | None = None,
        expect_window: bool = False,
    ) -> dict[str, object]:
        runner = SafeStudioCommandRunner(
            root,
            approved_commands=approved_commands,
            approved_capabilities=approved_capabilities,
        )
        validate_studio_command(command)
        if not runner._is_approved(command):
            raise UnsafeStudioCommand("Terminal execution requires explicit approval")
        resolved = list(command)
        executable = resolve_studio_executable(resolved[0])
        if executable is not None:
            resolved[0] = executable
        environment = _sanitized_environment()
        environment.update({"NO_COLOR": "1", "PYTHONDONTWRITEBYTECODE": "1"})
        process = subprocess.Popen(
            resolved,
            cwd=root.resolve(),
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding=locale.getpreferredencoding(False),
            errors="replace",
            shell=False,
            creationflags=(getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0),
        )
        terminal_id = uuid4().hex[:12]
        terminal = ManagedTerminal(terminal_id, root.resolve(), process)
        terminal.launch_tracked = expect_window
        with self._lock:
            self._sessions[terminal_id] = terminal
        return {
            "terminal_id": terminal_id,
            "pid": process.pid,
            "running": process.poll() is None,
            "output": "",
            "window_confirmed": None,
        }

    def _get(self, root: Path, terminal_id: str) -> ManagedTerminal:
        with self._lock:
            terminal = self._sessions.get(terminal_id)
        if terminal is None or terminal.root != root.resolve():
            raise ValueError(f"Unknown terminal session: {terminal_id}")
        return terminal

    def register_launch(
        self, root: Path, process: subprocess.Popen[str],
    ) -> str:
        terminal_id = uuid4().hex[:12]
        terminal = ManagedTerminal(terminal_id, root.resolve(), process)
        terminal.launch_tracked = True
        with self._lock:
            self._sessions[terminal_id] = terminal
        return terminal_id

    def inspect_launch(self, root: Path, terminal_id: str) -> dict[str, object]:
        terminal = self._get(root, terminal_id)
        observed = terminal.poll()
        observed["output"] = terminal.history[-20_000:]
        observed.update({
            "pid": terminal.process.pid,
            "window_confirmed": None,
            "launch_state": (
                "running_unconfirmed" if observed["running"] else "exited_unconfirmed"
            ),
        })
        return observed

    def poll(self, root: Path, terminal_id: str) -> dict[str, object]:
        terminal = self._get(root, terminal_id)
        return self.inspect_launch(root, terminal_id) if terminal.launch_tracked else terminal.poll()

    def write(self, root: Path, terminal_id: str, content: str) -> dict[str, object]:
        terminal = self._get(root, terminal_id)
        if terminal.process.poll() is not None or terminal.process.stdin is None:
            raise ValueError(f"Terminal session has exited: {terminal_id}")
        terminal.process.stdin.write(content)
        terminal.process.stdin.flush()
        return {"terminal_id": terminal_id, "written": len(content), "running": True}

    def stop(self, root: Path, terminal_id: str) -> dict[str, object]:
        terminal = self._get(root, terminal_id)
        if terminal.process.poll() is None:
            terminal.process.terminate()
            try:
                terminal.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                terminal.process.kill()
        result = terminal.poll()
        result["stopped"] = True
        return result


TERMINALS = TerminalRegistry()


def inspect_visible_processes(query: str = "") -> list[dict[str, object]]:
    """Return actual visible top-level windows with their owning process IDs."""
    if os.name != "nt":
        return []
    import ctypes

    user32 = ctypes.windll.user32
    matches: list[dict[str, object]] = []
    needle = query.casefold().strip()

    def collect(handle: int, _parameter: int) -> bool:
        if not user32.IsWindowVisible(handle):
            return True
        length = user32.GetWindowTextLengthW(handle)
        if length <= 0:
            return True
        title_buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(handle, title_buffer, length + 1)
        title = title_buffer.value.strip()
        if not title or (needle and needle not in title.casefold()):
            return True
        pid = ctypes.c_ulong()
        user32.GetWindowThreadProcessId(handle, ctypes.byref(pid))
        matches.append({"pid": int(pid.value), "window_handle": handle, "title": title})
        return True

    callback = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)(collect)
    user32.EnumWindows(callback, 0)
    return matches[:100]


class SafeStudioCommandRunner:
    def __init__(
        self,
        root: Path,
        *,
        timeout_seconds: int = 180,
        approved_commands: list[list[str]] | None = None,
        approved_capabilities: list[str] | None = None,
    ) -> None:
        self.root = root.resolve()
        self.timeout_seconds = timeout_seconds
        self.approved_commands = approved_commands or []
        self.approved_capabilities = approved_capabilities or []

    def _is_approved(self, command: list[str]) -> bool:
        capability = command_capability(command, self.root)
        return command in self.approved_commands or (
            capability is not None and capability in self.approved_capabilities
        )

    def run(self, command: list[str]) -> TestOutcome:
        validate_studio_command(command)
        if not self._is_approved(command):
            raise UnsafeStudioCommand("Command execution requires explicit approval")
        executable_name = Path(command[0]).name.casefold()
        if (
            executable_name in {"python", "python.exe", "py"}
            and len(command) > 1
            and not command[1].startswith("-")
            and Path(command[1]).suffix.casefold() == ".py"
            and not (self.root / command[1]).is_file()
        ):
            raise FileNotFoundError(f"Workspace script does not exist: {command[1]}")
        resolved = list(command)
        executable = resolve_studio_executable(resolved[0])
        if executable is not None:
            resolved[0] = executable
        started = time.perf_counter()
        encoding = locale.getpreferredencoding(False)
        environment = _sanitized_environment()
        environment.update({"CI": "1", "NO_COLOR": "1", "PYTHONDONTWRITEBYTECODE": "1"})
        try:
            completed = subprocess.run(
                resolved,
                cwd=self.root,
                env=environment,
                capture_output=True,
                text=True,
                encoding=encoding,
                errors="replace",
                timeout=self.timeout_seconds,
                shell=False,
                check=False,
                creationflags=(
                    getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
                ),
            )
            return TestOutcome(
                command=command,
                exit_code=completed.returncode,
                stdout=completed.stdout[-20_000:],
                stderr=completed.stderr[-20_000:],
                duration_seconds=time.perf_counter() - started,
            )
        except subprocess.TimeoutExpired as exc:
            return TestOutcome(
                command=command,
                exit_code=124,
                stdout=str(exc.stdout or "")[-20_000:],
                stderr=str(exc.stderr or "")[-20_000:],
                duration_seconds=time.perf_counter() - started,
                timed_out=True,
            )

    def launch(self, command: list[str]) -> TestOutcome:
        """Start an explicitly approved GUI command without waiting for it to close."""
        validate_studio_command(command)
        if not self._is_approved(command):
            raise UnsafeStudioCommand("Launching a GUI command requires explicit approval")
        if not is_detached_launch(command):
            raise UnsafeStudioCommand("Command is not a recognized detached launch")
        # An associated document is opened by Windows, often in an existing
        # browser process. The short-lived `cmd /c start` PID cannot prove
        # whether that browser reused a window or tab.
        if os.name == "nt" and [part.casefold() for part in command[:3]] == ["cmd", "/c", "start"]:
            arguments = command[3:]
            if arguments and arguments[0] == "":
                arguments = arguments[1:]
            if len(arguments) == 1 and Path(arguments[0]).suffix.casefold() in {
                ".html", ".htm", ".pdf", ".png", ".jpg", ".jpeg", ".svg", ".txt",
            }:
                target = (self.root / arguments[0]).resolve()
                if not target.is_relative_to(self.root) or not target.is_file():
                    raise UnsafeStudioCommand("Document to open must exist in the workspace")
                started = time.perf_counter()
                os.startfile(str(target))
                return TestOutcome(
                    command=command,
                    exit_code=0,
                    stdout=f"Windows accepted open request for {target.name}",
                    stderr="",
                    duration_seconds=time.perf_counter() - started,
                    launch_state="dispatched",
                    window_confirmed=None,
                )
        resolved = list(command)
        # `cmd /c start` exits before its child has created a window. For a
        # Python GUI, launch the target directly so the observed PID belongs
        # to the application rather than to the short-lived cmd wrapper.
        if [part.casefold() for part in resolved[:3]] == ["cmd", "/c", "start"]:
            child = resolved[3:]
            if child and child[0] == "":
                child = child[1:]
            if (
                len(child) >= 2
                and Path(child[0]).name.casefold() in {"python", "python.exe", "pythonw", "pythonw.exe"}
                and child[-1].casefold().endswith(".py")
            ):
                resolved = child
        executable = resolve_studio_executable(resolved[0])
        if executable is not None:
            resolved[0] = executable
        started = time.perf_counter()
        environment = _sanitized_environment()
        environment.update({"NO_COLOR": "1", "PYTHONDONTWRITEBYTECODE": "1"})
        creationflags = 0
        if os.name == "nt":
            creationflags = (
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                | getattr(subprocess, "DETACHED_PROCESS", 0)
                | getattr(subprocess, "CREATE_NO_WINDOW", 0)
            )
        process = subprocess.Popen(
            resolved,
            cwd=self.root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding=locale.getpreferredencoding(False),
            errors="replace",
            shell=False,
            close_fds=True,
            creationflags=creationflags,
        )
        terminal_id = TERMINALS.register_launch(
            self.root, process,
        )
        # A running process is not proof of a visible window. Keep its handle
        # so an uncertain launch can be inspected instead of started again.
        observed = TERMINALS.inspect_launch(self.root, terminal_id)
        launch_state = str(observed["launch_state"])
        output = str(observed["output"])
        exit_code = observed["exit_code"]
        return TestOutcome(
            command=command,
            # None means still running, not a synthetic successful exit.
            exit_code=exit_code,
            stdout=output,
            stderr="",
            duration_seconds=time.perf_counter() - started,
            terminal_id=terminal_id,
            pid=process.pid,
            launch_state=launch_state,
            window_confirmed=None,
        )


def _visible_window_handles() -> set[int]:
    """Return visible top-level Windows handles without opening a console window."""
    if os.name != "nt":
        return set()
    try:
        import ctypes

        handles: set[int] = set()
        callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
        user32 = ctypes.windll.user32

        @callback_type  # type: ignore[untyped-decorator]
        def collect(hwnd: int, _lparam: int) -> bool:
            if user32.IsWindowVisible(hwnd) and user32.GetWindowTextLengthW(hwnd) > 0:
                handles.add(int(hwnd))
            return True

        user32.EnumWindows(collect, 0)
        return handles
    except (AttributeError, OSError):
        return set()
