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
    """Return deterministic decision support for every command permission request."""
    purpose, impact, risk = describe_command(command)
    rendered = " ".join(command).casefold()
    executable = Path(command[0]).name.casefold() if command else "未知程序"
    python_inline_delete = (
        executable in {"python", "python.exe", "py"}
        and "-c" in rendered
        and any(marker in rendered for marker in ("shutil.rmtree", ".unlink(", ".rmdir("))
    )
    destructive = python_inline_delete or any(
        marker in f" {rendered} "
        for marker in (" remove-item ", " del ", " erase ", " rmdir ", " rd ", " rm ")
    )
    network = any(
        marker in rendered
        for marker in ("curl ", "wget ", "invoke-webrequest", "pip install", "npm install")
    )
    installer = any(
        marker in rendered
        for marker in ("winget install", "pip install", "npm install", "choco install")
    )
    if executable in {"python", "python.exe", "py"} and lowered_module(command) == "py_compile":
        targets = [part for part in command[3:] if part.casefold().endswith(".py")]
        named = "、".join(targets) or "命令中列出的 Python 文件"
        return {
            "purpose": f"对 {named} 做 Python 语法编译检查，确认文件可以被解释器加载。",
            "impact": (
                "不会启动该程序，也不会改写源文件；Python 可能在对应的 __pycache__ "
                "目录生成可删除的 .pyc 编译缓存。"
            ),
            "scope": f"只检查 {named}，不检查或运行其他项目文件。",
            "recovery": "如产生 __pycache__ 或 .pyc，可直接删除；源代码不会因此改变。",
            "recommendation": "目标文件正确时可以允许；这一步只用于语法验证。",
            "risk": "low",
            "destructive": False,
        }
    if destructive:
        if python_inline_delete:
            original = " ".join(command)
            path_match = re.search(
                r"\b(?:root|target|path|folder|directory)\s*=\s*Path\(r?(['\"])(.+?)\1\)",
                original,
                re.IGNORECASE,
            )
            target = path_match.group(2) if path_match else "命令中指定的目录"
            clears_children = "iterdir()" in rendered and "shutil.rmtree" in rendered
            purpose = (
                f"永久清空 {target} 内的全部文件和子文件夹，保留目录本身；"
                "完成后再次列出目录并确认剩余项目数量为 0。"
                if clears_children
                else f"永久删除 {target} 中由这段 Python 代码指定的文件或目录。"
            )
            impact = (
                f"{target} 内的源码、隐藏文件、配置和未提交内容都可能被永久删除；"
                "这不是只读检查，也不保证进入回收站。"
            )
        return {
            "purpose": purpose
            if python_inline_delete
            else "执行下方删除命令，移除命令中指定的文件或目录。",
            "impact": impact
            if python_inline_delete
            else "目标内容会被删除；递归参数可能同时删除全部子目录和隐藏内容。",
            "scope": (
                f"删除范围是 {target} 的直接内容及其子目录；目录 {target} 本身会保留。"
                if python_inline_delete and clears_children
                else "仅限完整命令中明确写出的目标路径，但通配符可能扩大匹配范围。"
            ),
            "recovery": "不保证进入回收站。未提交到 Git 或没有备份的内容可能无法恢复。",
            "recommendation": (
                "只有在目标路径和删除范围完全符合预期时才允许；不确定请拒绝并要求缩小范围。"
            ),
            "risk": "high",
            "destructive": True,
        }
    if installer:
        return {
            "purpose": purpose,
            "impact": impact,
            "scope": f"允许 {executable} 安装命令中列出的软件包及其依赖。",
            "recovery": "通常可以卸载，但可能留下缓存、配置文件或 PATH 变更。",
            "recommendation": "确认软件名称、来源和安装位置后再允许。",
            "risk": "high",
            "destructive": False,
        }
    if network:
        return {
            "purpose": purpose,
            "impact": "会连接外部网络并接收或发送命令中指定的数据。",
            "scope": "仅允许执行下方这一条命令；请核对域名、上传内容和保存位置。",
            "recovery": "网络请求本身无法撤回；下载到本地的文件通常可以删除。",
            "recommendation": "仅在目标域名可信且命令未包含密钥、隐私数据时允许。",
            "risk": "high",
            "destructive": False,
        }
    if risk == "low":
        return {
            "purpose": purpose,
            "impact": impact,
            "scope": "只允许执行下方这一条命令一次，不会自动授权后续命令。",
            "recovery": "只读操作不会产生需要恢复的文件更改。",
            "recommendation": "目标路径无敏感内容时通常可以允许。",
            "risk": risk,
            "destructive": False,
        }
    if risk == "medium":
        return {
            "purpose": purpose,
            "impact": impact,
            "scope": "只允许执行下方这一条完整命令一次，不会自动授权其他命令。",
            "recovery": "关闭启动的程序可以结束进程；文件变化是否可恢复取决于该程序行为。",
            "recommendation": "确认目标程序和文件正确后可以允许；不确定时先要求只读检查。",
            "risk": risk,
            "destructive": False,
        }
    return {
        "purpose": (
            f"为完成当前步骤，RAgent 计划：{rationale.strip()}"
            if rationale and purpose == "执行 Agent 为当前任务提议的命令。"
            else purpose
        ),
        "impact": impact,
        "scope": "RAgent 无法可靠推断这条命令的完整影响范围。",
        "recovery": "无法确认命令产生的变化是否能够恢复。",
        "recommendation": (
            "系统暂时无法自动判断这条命令的全部影响。请核对完整命令和目标路径；"
            "确认与当前任务一致时可本次允许，看不懂或范围不明确时再拒绝并要求调整。"
        ),
        "risk": "unknown",
        "destructive": False,
    }


def lowered_module(command: list[str]) -> str | None:
    """Return the module passed to ``python -m`` without interpreting shell text."""
    lowered = [part.casefold() for part in command]
    try:
        index = lowered.index("-m")
    except ValueError:
        return None
    return lowered[index + 1] if index + 1 < len(lowered) else None


def describe_command(command: list[str]) -> tuple[str, str, str]:
    """Explain an argv command without trusting model-authored risk claims."""
    lowered = [part.casefold() for part in command]
    rendered = " ".join(lowered)
    original = " ".join(command)
    executable = Path(lowered[0]).name if lowered else ""
    python_executable = executable in {"python", "python.exe", "py"}
    module = lowered_module(command) if python_executable else None
    if module == "py_compile":
        targets = [part for part in command[3:] if part.casefold().endswith(".py")]
        named = "、".join(targets) or "命令中列出的 Python 文件"
        return (
            f"编译检查 {named} 的 Python 语法。",
            "不会运行程序或修改源文件；可能生成可删除的 .pyc 缓存。",
            "low",
        )
    if module == "compileall":
        target = command[-1] if len(command) > 3 else "当前项目"
        return (
            f"递归编译检查 {target} 中的 Python 文件语法。",
            "不会运行项目或修改源文件；可能生成可删除的 __pycache__/.pyc 缓存。",
            "low",
        )
    if module == "pytest" or executable in {"pytest", "pytest.exe"}:
        target = next((part for part in command if part.startswith("tests")), "项目测试集")
        return (
            f"运行 {target}，用自动化测试检查当前实现是否符合预期。",
            "会执行测试代码；通常不改源文件，但测试自身可能创建缓存或临时文件。",
            "medium",
        )
    if module in {"ruff", "mypy", "pyright"} or executable in {"ruff", "mypy", "pyright"}:
        tool = module or executable
        return (
            f"运行 {tool} 检查代码质量、格式或类型问题。",
            "默认只报告问题；若完整命令包含 format 或 --fix，可能自动修改项目文件。",
            "medium",
        )
    if module == "pip" and "install" in lowered:
        packages = [
            part for part in command[lowered.index("install") + 1 :] if not part.startswith("-")
        ]
        named = "、".join(packages) or "命令中列出的 Python 包"
        return (
            f"安装 {named} 及其依赖，供当前任务构建或运行使用。",
            "会联网下载并改变当前 Python 环境，可能安装多个间接依赖。",
            "high",
        )
    if executable in {"powershell", "powershell.exe", "pwsh", "pwsh.exe"}:
        mutating = any(
            marker in rendered
            for marker in (
                " set-content ",
                " add-content ",
                " remove-item ",
                " move-item ",
                " copy-item ",
                " new-item ",
                " start-process ",
            )
        )
        if "select-string" in rendered and not mutating:
            path_match = re.search(r"-path\s+['\"]?([^'\";]+)", original, re.IGNORECASE)
            target = path_match.group(1).strip() if path_match else "指定文件"
            if target.startswith("$"):
                variable = re.escape(target[1:])
                assigned = re.search(rf"\${variable}\s*=\s*['\"]([^'\"]+)", original)
                if assigned:
                    target = assigned.group(1)
            return (
                f"在 {target} 中搜索代码结构和关键词，并显示匹配行及行号，"
                "帮助 Agent 定位需要修改的位置。",
                "这是只读代码搜索：不会运行该程序、不会修改或删除文件，也不会更改系统设置。",
                "low",
            )
        read_markers = ("get-content", "get-childitem", "test-path")
        if any(marker in rendered for marker in read_markers) and not mutating:
            return (
                "读取文件、列出目录或检查路径是否存在，以了解项目结构和当前代码。",
                "这是只读检查：不会运行项目、修改文件、安装软件或更改系统设置。",
                "low",
            )
    go_probe = re.fullmatch(
        r"cmd\s+/c\s+if\s+exist\s+([^&|<>]+?)\s+"
        r"\(dir\s+/a\s+([^&|<>]+?)\)\s+else\s+"
        r"\(echo\s+[a-z0-9_-]+\)\s+&{1,2}\s+where\s+go\s+&{1,2}\s+go\s+version",
        rendered,
    )
    if go_probe:
        location = go_probe.group(1).strip()
        return (
            f"检查 {location} 是否存在，并确认系统能否找到 Go 及其版本。",
            "只会读取目录、PATH 和版本信息，不会安装 Go、修改文件或更改系统设置。",
            "low",
        )
    if lowered[:3] == ["cmd", "/c", "where"] and "version" in rendered:
        tool = command[3] if len(command) > 3 else "目标工具"
        tool = "Go" if tool.casefold() == "go" else tool
        return (
            f"检查电脑是否能找到 {tool}，并读取已安装版本。",
            "只读取系统环境信息，不会安装软件或修改文件。",
            "low",
        )
    if lowered and Path(lowered[0]).name in {"winget", "winget.exe"} and "install" in lowered:
        return (
            "使用 Windows 包管理器安装任务所需的软件。",
            "会向电脑安装软件并可能修改 PATH 等系统环境配置。",
            "high",
        )
    if is_detached_launch(command):
        return (
            "在 Windows 中启动该程序或打开该文件。",
            "会创建一个独立进程或窗口；程序本身可能继续访问项目文件。",
            "medium",
        )
    if any(part in {"tasklist", "get-process"} for part in lowered):
        return (
            "查看当前正在运行的程序和进程状态。",
            "只读取进程信息，不会关闭程序或修改文件。",
            "low",
        )
    if python_executable and "-c" in lowered:
        return (
            "运行一段 Python 命令，通常用于检查项目内容或验证结果。",
            "Python 命令具备读写文件的能力；请在下方查看完整命令后再决定。",
            "high",
        )
    return (
        "执行 Agent 为当前任务提议的命令。",
        "无法自动确认它是否会修改文件或系统；请先查看完整命令。",
        "unknown",
    )


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


ALLOWED_COMMANDS: dict[str, set[tuple[str, ...]]] = {
    "python": {("-m", "pytest"), ("-m", "ruff"), ("-m", "mypy"), ("-m", "py_compile")},
    "python.exe": {
        ("-m", "pytest"),
        ("-m", "ruff"),
        ("-m", "mypy"),
        ("-m", "py_compile"),
    },
    "py": {("-m", "pytest"), ("-m", "ruff"), ("-m", "mypy"), ("-m", "py_compile")},
    "pytest": {()},
    "pytest.exe": {()},
    "ruff": {("check",), ("format",)},
    "mypy": {()},
    "npm": {("test",), ("run", "test"), ("run", "build"), ("run", "lint"), ("run", "typecheck")},
    "npm.cmd": {
        ("test",),
        ("run", "test"),
        ("run", "build"),
        ("run", "lint"),
        ("run", "typecheck"),
    },
    "pnpm": {("test",), ("run", "test"), ("run", "build"), ("run", "lint"), ("run", "typecheck")},
    "yarn": {("test",), ("build",), ("lint",)},
    "cargo": {("test",), ("check",), ("clippy",)},
    "go": {("test",), ("build",)},
    "dotnet": {("test",), ("build",)},
    "git": {("status",), ("diff",)},
}


def _validate_command_arguments(command: list[str]) -> None:
    for argument in command[1:]:
        if any(marker in argument for marker in (";", "&", "|", ">", "<", "\n", "\r")):
            raise UnsafeStudioCommand("Shell operators are not allowed")
        target = argument.split("::", 1)[0]
        if Path(target).is_absolute() or ".." in Path(target).parts:
            raise UnsafeStudioCommand(f"Command target escapes workspace: {argument}")


def validate_studio_command(command: list[str]) -> None:
    if not command:
        raise UnsafeStudioCommand("Command cannot be empty")
    executable = Path(command[0]).name.casefold()
    prefixes = ALLOWED_COMMANDS.get(executable)
    if prefixes is None:
        raise UnsafeStudioCommand(f"Command is not allowlisted: {command[0]}")
    arguments = tuple(part.casefold() for part in command[1:])
    is_workspace_python_script = (
        executable in {"python", "python.exe", "py"}
        and bool(arguments)
        and not arguments[0].startswith("-")
        and Path(arguments[0]).suffix == ".py"
    )
    if not is_workspace_python_script and not any(
        arguments[: len(prefix)] == prefix for prefix in prefixes
    ):
        raise UnsafeStudioCommand(f"Command arguments are not allowlisted: {' '.join(command)}")
    _validate_command_arguments(command)


class ManagedTerminal:
    def __init__(self, terminal_id: str, root: Path, process: subprocess.Popen[str]) -> None:
        self.terminal_id = terminal_id
        self.root = root
        self.process = process
        self.output: Queue[str] = Queue()
        self.history = ""
        self.window_confirmed = False
        self.window_baseline: set[int] | None = None
        self.window_requires_pid = True
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
        if not runner._is_approved(command):
            validate_studio_command(command)
        resolved = list(command)
        executable = shutil.which(resolved[0])
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
        window_confirmed = False
        if expect_window and os.name == "nt":
            deadline = time.monotonic() + 4.0
            while time.monotonic() < deadline and process.poll() is None:
                if any(item["pid"] == process.pid for item in inspect_visible_processes()):
                    window_confirmed = True
                    break
                time.sleep(0.2)
        terminal.window_confirmed = window_confirmed
        return {
            "terminal_id": terminal_id,
            "pid": process.pid,
            "running": process.poll() is None,
            "output": "",
            "window_confirmed": window_confirmed,
        }

    def _get(self, root: Path, terminal_id: str) -> ManagedTerminal:
        with self._lock:
            terminal = self._sessions.get(terminal_id)
        if terminal is None or terminal.root != root.resolve():
            raise ValueError(f"Unknown terminal session: {terminal_id}")
        return terminal

    def register_launch(
        self, root: Path, process: subprocess.Popen[str], *,
        window_baseline: set[int], window_requires_pid: bool,
    ) -> str:
        terminal_id = uuid4().hex[:12]
        terminal = ManagedTerminal(terminal_id, root.resolve(), process)
        terminal.launch_tracked = True
        terminal.window_baseline = window_baseline
        terminal.window_requires_pid = window_requires_pid
        with self._lock:
            self._sessions[terminal_id] = terminal
        return terminal_id

    def inspect_launch(self, root: Path, terminal_id: str) -> dict[str, object]:
        terminal = self._get(root, terminal_id)
        observed = terminal.poll()
        observed["output"] = terminal.history[-20_000:]
        if not terminal.window_confirmed:
            if terminal.window_requires_pid:
                terminal.window_confirmed = any(
                    item["pid"] == terminal.process.pid for item in inspect_visible_processes()
                )
            elif terminal.window_baseline is not None:
                terminal.window_confirmed = bool(
                    _visible_window_handles() - terminal.window_baseline
                )
        observed.update({
            "pid": terminal.process.pid,
            "window_confirmed": terminal.window_confirmed,
            "launch_state": (
                "window_confirmed" if terminal.window_confirmed else
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
        if self._is_approved(command):
            if not command:
                raise UnsafeStudioCommand("Command cannot be empty")
            # This exact argv was explicitly approved by the user and is passed
            # to subprocess with shell=False. Shell metacharacters inside one
            # argument (for example Python -c statements separated by ';') are
            # therefore data for the target executable, not shell operators.
        else:
            validate_studio_command(command)
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
        executable = shutil.which(resolved[0])
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
        target_is_python = (
            Path(resolved[0]).name.casefold()
            in {"python", "python.exe", "pythonw", "pythonw.exe"}
            and any(part.casefold().endswith(".py") for part in resolved[1:])
        )
        executable = shutil.which(resolved[0])
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
        windows_before = _visible_window_handles()
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
            self.root, process, window_baseline=windows_before,
            window_requires_pid=target_is_python,
        )
        # A running process is not proof of a visible window. Keep its handle
        # so an uncertain launch can be inspected instead of started again.
        deadline = time.monotonic() + 8.0
        observed = TERMINALS.inspect_launch(self.root, terminal_id)
        while time.monotonic() < deadline:
            if observed["window_confirmed"]:
                break
            if not observed["running"]:
                break
            time.sleep(0.25)
            observed = TERMINALS.inspect_launch(self.root, terminal_id)
        if not observed["running"] and not observed["window_confirmed"]:
            # Allow the output reader to collect a short traceback from a
            # process that exited just before the final inspection.
            time.sleep(0.05)
            observed = TERMINALS.inspect_launch(self.root, terminal_id)
        window_confirmed = bool(observed["window_confirmed"])
        launch_state = str(observed["launch_state"])
        output = str(observed["output"])
        exit_code = observed["exit_code"]
        return TestOutcome(
            command=command,
            # Zero means the launch request started a still-running process;
            # window confirmation is a separate, explicit completion fact.
            exit_code=0 if window_confirmed or observed["running"] else (
                exit_code if exit_code not in (None, 0) else 1
            ),
            stdout=(
                f"Started process {process.pid}; window_confirmed=true"
                if window_confirmed
                else f"Process {process.pid} did not create a visible window; state={launch_state}"
            ),
            stderr="" if window_confirmed else (
                output or ("" if observed["running"] else "未检测到目标程序的可见窗口。")
            ),
            duration_seconds=time.perf_counter() - started,
            terminal_id=terminal_id,
            pid=process.pid,
            launch_state=launch_state,
            window_confirmed=window_confirmed,
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
