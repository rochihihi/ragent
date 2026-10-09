"""Host-owned execution policy with explicitly approved per-call host exceptions.

The Node broker owns the sandbox; the requested argv is only interpreted inside
it. No installation, elevation, or automatic unsandboxed fallback happens at tool time.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SandboxError(RuntimeError):
    """A host-side sandbox failure, not permission to retry outside it."""


class SandboxedResult(subprocess.CompletedProcess[str]):
    sandboxed = True


def _console_text(value: bytes | str, fallback: str) -> str:
    if isinstance(value, str):
        return value
    try:
        return value.decode("utf-8")  # Broker and Node tools emit UTF-8.
    except UnicodeDecodeError:
        return value.decode(fallback, "replace")


class SandboxSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["required", "off"] = "required"
    allow_approved_host_execution: bool = True
    allowed_domains: list[str] = Field(default_factory=list, max_length=100)
    tool_read_paths: list[str] = Field(default_factory=list, max_length=30)

    @field_validator("allowed_domains")
    @classmethod
    def domains(cls, values: list[str]) -> list[str]:
        for value in values:
            if not re.fullmatch(r"(?:\*\.)?[a-zA-Z0-9](?:[a-zA-Z0-9.-]{0,251}[a-zA-Z0-9])?", value):
                raise ValueError("填写域名，不允许 URL、端口或全局 * 通配符")
        return list(dict.fromkeys(value.lower() for value in values))

    @field_validator("tool_read_paths")
    @classmethod
    def paths(cls, values: list[str]) -> list[str]:
        for value in values:
            if not Path(value).is_absolute() or "\x00" in value or any(c in value for c in "*?[]"):
                raise ValueError("工具读取目录必须是无通配符的绝对路径")
        return list(dict.fromkeys(str(Path(value).resolve()) for value in values))


def configuration_path() -> Path:
    return Path(
        os.environ.get("RAGENT_SANDBOX_CONFIG")
        or str(Path(os.environ.get("APPDATA", str(Path.home()))) / "RAgent" / "sandbox.json")
    )


def host_control_roots() -> list[Path]:
    """Protected even when a model has been granted broad native file access."""
    return [
        configuration_path().parent.resolve(),
        (Path(os.environ.get("APPDATA", str(Path.home()))) / "RAgent").resolve(),
    ]


def settings() -> SandboxSettings:
    path = configuration_path()
    if not path.exists():
        return SandboxSettings()
    try:
        return SandboxSettings.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise SandboxError(
            "SandboxError: 沙箱配置无效；请在设置中修正，未降级为普通执行。"
        ) from exc


def save_settings(value: SandboxSettings) -> None:
    path = configuration_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".sandbox-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(value.model_dump_json(indent=2))
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def policy_fingerprint() -> str:
    return hashlib.sha256(settings().model_dump_json().encode()).hexdigest()


def runtime_root() -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))
    return base / "sandbox" / "runtime"


def _node() -> str:
    packaged = runtime_root() / "node.exe"
    node = str(packaged) if packaged.is_file() else shutil.which("node")
    if (
        not node
        or not (
            runtime_root() / "node_modules/@anthropic-ai/sandbox-runtime/dist/index.js"
        ).is_file()
    ):
        raise SandboxError(
            "SandboxError: 沙箱运行库或 Node 未安装，请重新打包或安装锁定依赖；未执行命令。"
        )
    return str(Path(node).resolve())


def _broker_environment() -> dict[str, str]:
    # Never let a workspace-controlled Node loader execute in the host broker.
    keys = {
        "PATH",
        "SYSTEMROOT",
        "WINDIR",
        "TEMP",
        "TMP",
        "HOME",
        "USERPROFILE",
        "APPDATA",
        "LOCALAPPDATA",
        "PROGRAMDATA",
        "PROGRAMFILES",
        "PROGRAMFILES(X86)",
    }
    return {key: value for key, value in os.environ.items() if key.upper() in keys}


def description() -> dict[str, Any]:
    policy = settings()
    return {
        **policy.model_dump(),
        "backend": "anthropic-srt",
        "windows_alpha": os.name == "nt",
        "scope": (
            "命令、终端、外部测试、本地 stdio MCP；原生文件工具另受路径授权控制。"
            "HTTP MCP 的远端副作用不在本机沙箱内。"
        ),
        "failure_policy": "不可用时拒绝执行；模型不能关闭沙箱或自动提权安装。",
        "read_policy": "不是全盘读取隔离；保护用户配置和凭据目录，写入限工作区与显式授权路径。",
        "gui_policy": (
            "桌面窗口、浏览器打开和其他不兼容操作可提出 execution_mode=host；"
            "每次必须单独批准，结果标记无沙箱；不会关闭后续调用的沙箱。"
            if policy.allow_approved_host_execution
            else "严格策略禁止沙箱外执行，桌面操作也不能通过普通审批绕过。"
        ),
    }


def management(action: Literal["probe", "install"]) -> dict[str, Any]:
    if action == "install" and os.name != "nt":
        raise SandboxError("仅 Windows 使用账户/WFP 安装；其他系统请安装原生沙箱依赖。")
    try:
        # A behavioral probe now grants/cleans a runtime-only ACL lease and
        # launches a fixed self-test, so it must share the execution lock.
        with SandboxLaunch([], _broker_environment(), True) as lease:
            _acquire_windows_lock(lease)
            process = subprocess.Popen(
                [_node(), str(runtime_root() / "bridge.mjs"), action],
                cwd=runtime_root(),
                env=lease.env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                start_new_session=os.name != "nt",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0,
            )
            try:
                output, errors = process.communicate(timeout=180 if action == "install" else 90)
            except BaseException:
                stop_process_tree(process)
                raise
        if process.returncode:
            raise SandboxError(f"SandboxError: {errors.decode('utf-8', 'replace')[-2000:]}")
        payload = json.loads(output)
        if not isinstance(payload, dict):
            raise ValueError("Invalid sandbox status")
        return payload
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        raise SandboxError(
            f"SandboxError: 沙箱检查未完成（{type(exc).__name__}），不会普通执行。"
        ) from exc


def stop_process_tree(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        # Fixed system executable + the broker's PID, never a model-supplied shell.
        taskkill = Path(os.environ.get("SYSTEMROOT", r"C:\Windows")) / "System32/taskkill.exe"
        subprocess.run(
            [str(taskkill), "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            timeout=10,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    else:
        import signal

        try:
            getattr(os, "killpg")(process.pid, signal.SIGTERM)  # noqa: B009 (Windows stubs omit POSIX)
        except ProcessLookupError:
            return
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            getattr(os, "killpg")(process.pid, getattr(signal, "SIGKILL"))  # noqa: B009
        else:
            process.kill()
        process.wait(timeout=3)


@dataclass
class SandboxLaunch:
    argv: list[str]
    env: dict[str, str]
    enabled: bool
    request: Path | None = None
    status: Path | None = None
    lock_stream: Any = None
    _closed: bool = False
    _guard: threading.Lock = field(default_factory=threading.Lock)

    def close(self) -> None:
        with self._guard:
            if self._closed:
                return
            self._closed = True
            if self.request:
                self.request.unlink(missing_ok=True)
            if self.status:
                self.status.unlink(missing_ok=True)
            if self.lock_stream:
                self.lock_stream.close()  # Releases the cross-process Windows byte lock.

    def __enter__(self) -> SandboxLaunch:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def follow(self, process: subprocess.Popen[Any]) -> None:
        def reap() -> None:
            try:
                process.wait()
            finally:
                self.close()

        threading.Thread(target=reap, daemon=True).start()

    def check_error(self) -> None:
        if self.status and self.status.is_file():
            result = json.loads(self.status.read_text(encoding="utf-8"))
            if result.get("state") == "error":
                raise SandboxError(f"SandboxError: {result.get('message', '沙箱启动失败')}")

    def wait_ready(self, process: subprocess.Popen[Any], timeout: float = 30) -> None:
        if not self.enabled:
            return
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.status and self.status.is_file():
                try:
                    result = json.loads(self.status.read_text(encoding="utf-8"))
                except ValueError:
                    time.sleep(0.01)  # Broker is finishing the small status record.
                    continue
                self.check_error()
                if result.get("state") == "ready":
                    return
            if process.poll() is not None:
                raise SandboxError("SandboxError: 沙箱 broker 在准备完成前退出，命令未放行。")
            time.sleep(0.02)
        raise SandboxError("SandboxError: 沙箱准备超时，未降级；请检查运行库状态。")


def _acquire_windows_lock(holder: SandboxLaunch) -> None:
    if os.name != "nt":
        return
    import msvcrt

    control = configuration_path().parent
    control.mkdir(parents=True, exist_ok=True)
    holder.lock_stream = (control / "sandbox-execution.lock").open("a+b")
    holder.lock_stream.seek(0)
    try:
        msvcrt.locking(holder.lock_stream.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as exc:
        raise SandboxError(
            "SandboxError: Windows 沙箱正在执行另一项任务；"
            "请等待或停止它，避免账户权限交叉。"
        ) from exc


def prepare(
    root: Path,
    argv: list[str],
    environment: dict[str, str],
    *,
    read_paths: list[Path] | None = None,
    write_paths: list[Path] | None = None,
    explicit_env: dict[str, str] | None = None,
    interactive: bool = False,
    host_execution_approved: bool = False,
) -> SandboxLaunch:
    policy = settings()
    if host_execution_approved:
        ensure_host_execution_allowed()
        return SandboxLaunch(list(argv), environment, False)
    if policy.mode == "off":
        return SandboxLaunch(list(argv), environment, False)
    node = _node()
    protected = [
        configuration_path().parent,
        Path(os.environ.get("APPDATA", str(Path.home()))) / "RAgent",
        Path.home() / ".ssh",
        Path.home() / ".aws",
        Path.home() / ".codex",
        Path.home() / ".config",
        Path.home() / ".npmrc",
        Path.home() / ".gitconfig",
        Path.home() / ".pypirc",
    ]
    if os.environ.get("CODEX_HOME"):
        protected.append(Path(os.environ["CODEX_HOME"]).resolve())
    runtime = runtime_root().resolve()
    tool_reads = [Path(node).parent]
    # Frozen Python prefixes may point at the entire private _MEI tree. The
    # embedded interpreter is NOT a project-test interpreter; don't grant its
    # whole desktop distribution to the sandbox account.
    if not getattr(sys, "frozen", False):
        tool_reads.append(Path(sys.base_prefix).resolve())
    if os.name == "nt":
        # Machine-wide tools already use the sandbox account's baseline read
        # rights. Do not attempt WRITE_DAC changes to Program Files/Windows.
        tool_reads = [path for path in tool_reads if path.is_relative_to(Path.home())]
    if not getattr(sys, "frozen", False) and sys.prefix != sys.base_prefix:
        tool_reads.append(Path(sys.prefix).resolve())
    reads = [
        root.resolve(),
        *tool_reads,
        runtime,
        *(read_paths or []),
        *(write_paths or []),
        *(Path(path) for path in policy.tool_read_paths),
    ]
    writes = [root.resolve(), *(write_paths or [])]
    # Broad grants can override read denies on some platforms. Reject overlap
    # rather than claiming protected credentials are still isolated.
    for path in [
        root.resolve(),
        *(read_paths or []),
        *(write_paths or []),
        *(Path(p) for p in policy.tool_read_paths),
    ]:
        for private in protected:
            private = private.resolve()
            path = path.resolve()
            if path == private or path.is_relative_to(private) or private.is_relative_to(path):
                raise SandboxError(
                    "SandboxError: 工作区/授权路径覆盖受保护的用户配置目录，请缩小范围。"
                )
    holder = SandboxLaunch([], _broker_environment(), True)
    try:
        control = configuration_path().parent
        control.mkdir(parents=True, exist_ok=True)
        _acquire_windows_lock(holder)
        descriptor, request = tempfile.mkstemp(
            prefix="sandbox-request-", suffix=".json", dir=control
        )
        holder.request = Path(request)
        holder.status = Path(request + ".status")
        resolved = list(argv)
        executable = shutil.which(resolved[0])
        if executable:
            resolved[0] = str(Path(executable).resolve())
        if os.name == "nt" and Path(resolved[0]).name.casefold() in {
            "npm.cmd",
            "npx.cmd",
            "npm",
            "npx",
        }:
            # Node cannot spawn a .cmd shim with shell=False. Use npm's actual
            # JavaScript CLI, keeping every user argument as an argv element.
            name = Path(resolved[0]).stem.casefold()
            script = Path(resolved[0]).parent / "node_modules/npm/bin" / f"{name}-cli.js"
            if not script.is_file():
                raise SandboxError(
                    "SandboxError: 找不到 npm/npx 的 JS 入口；请核对 Node 安装目录。"
                )
            resolved = [node, str(script.resolve()), *resolved[1:]]
        payload = {
            "argv": resolved,
            "cwd": str(root.resolve()),
            "interactive": interactive,
            "statusPath": str(holder.status),
            "env": {
                **{
                    key: environment[key]
                    for key in (
                        "CI",
                        "NO_COLOR",
                        "PYTHONDONTWRITEBYTECODE",
                        "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
                    )
                    if key in environment
                },
                **(explicit_env or {}),
            },
            "config": {
                "network": {
                    "allowedDomains": policy.allowed_domains,
                    "deniedDomains": [],
                    "strictAllowlist": True,
                },
                "filesystem": {
                    "allowRead": [str(p.resolve()) for p in reads],
                    "allowWrite": [str(p.resolve()) for p in writes],
                    "denyRead": [str(p.resolve()) for p in protected],
                    # SRT Windows ReadDeny uses FILE_ALL_ACCESS (read AND
                    # write/delete denied). In 0.0.78 a same-holder duplicate
                    # denyWrite can replace it with the weaker WriteDeny mask.
                    # Keep only the stronger denyRead for protected paths;
                    # other platforms still need both independent policies.
                    "denyWrite": (
                        [] if os.name == "nt" else [str(p.resolve()) for p in protected]
                    ) + [str(runtime)],
                },
            },
        }
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False)
        holder.argv = [node, str(runtime / "bridge.mjs"), "execute", str(holder.request)]
        return holder
    except BaseException:
        holder.close()
        raise


def ensure_host_execution_allowed() -> None:
    """Check host-owned policy; this function never grants user authority."""
    policy = settings()
    if policy.mode == "required" and not policy.allow_approved_host_execution:
        raise SandboxError("SandboxError: 用户策略禁止沙箱外执行；审批不能覆盖此限制。")


def run(
    root: Path,
    argv: list[str],
    environment: dict[str, str],
    *,
    timeout: float,
    encoding: str,
    read_paths: list[Path] | None = None,
    write_paths: list[Path] | None = None,
    host_execution_approved: bool = False,
) -> subprocess.CompletedProcess[str]:
    with prepare(
        root, argv, environment, read_paths=read_paths, write_paths=write_paths,
        host_execution_approved=host_execution_approved,
    ) as launch:
        if not launch.enabled and not host_execution_approved:
            return subprocess.run(
                launch.argv,
                cwd=root,
                env=launch.env,
                capture_output=True,
                text=True,
                encoding=encoding,
                errors="replace",
                timeout=timeout,
                shell=False,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0,
            )
        process = subprocess.Popen(
            launch.argv,
            cwd=runtime_root() if launch.enabled else root,
            env=launch.env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            shell=False,
            start_new_session=os.name != "nt",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0,
        )
        try:
            raw_stdout, raw_stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            stop_process_tree(process)
            raw_stdout, raw_stderr = process.communicate(timeout=3)
            stdout = _console_text(raw_stdout, encoding)
            stderr = _console_text(raw_stderr, encoding)
            raise subprocess.TimeoutExpired(argv, timeout, output=stdout, stderr=stderr) from None
        except BaseException:
            stop_process_tree(process)
            raise
        stdout = _console_text(raw_stdout, encoding)
        stderr = _console_text(raw_stderr, encoding)
        launch.check_error()
        if launch.enabled and "RAGENT_SANDBOX_ERROR:" in stderr and process.returncode != 0:
            raise SandboxError(f"SandboxError: {stderr[-2000:]}")
        result_type = SandboxedResult if launch.enabled else subprocess.CompletedProcess
        return result_type(argv, process.returncode, stdout, stderr)
