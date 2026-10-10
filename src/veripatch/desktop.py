"""Native Windows desktop shell for the local VeriPatch application."""

from __future__ import annotations

import os
import socket
import sqlite3
import sys
import tempfile
import threading
import time
import urllib.request
from ctypes import Structure, byref, sizeof, windll
from ctypes.wintypes import DWORD, HWND, RECT
from dataclasses import replace
from pathlib import Path
from typing import Any

from veripatch.api import create_app
from veripatch.config import Settings


class WindowControls:
    """Small JS bridge for the custom frameless title bar."""

    def __init__(self, sandbox_token: str = "") -> None:
        # Keep native objects private. pywebview recursively exposes public
        # js_api attributes and traversing a WinForms window can hang WebView2.
        self._window: Any | None = None
        self._restore_bounds: tuple[int, int, int, int] | None = None
        self._tray: WindowsTray | None = None
        self._quitting = False
        self._sandbox_token = sandbox_token

    def sandbox_token(self) -> str:
        """Private setting capability for this desktop window, not an HTTP API."""
        return self._sandbox_token

    def _bind(self, window: Any) -> None:
        self._window = window

    def _bind_tray(self, tray: WindowsTray) -> None:
        self._tray = tray

    def minimize(self) -> None:
        if self._window is not None:
            self._window.minimize()

    def maximize(self) -> None:
        if self._window is None:
            return
        if self._restore_bounds is not None:
            self.restore()
            return
        native = getattr(self._window, "native", None)
        if os.name != "nt" or native is None:
            self._window.maximize()
            return

        # A borderless WinForms window can cover the taskbar when using the
        # regular Maximized state. Fit it to the active monitor's work area.
        hwnd = int(native.Handle.ToInt64())
        bounds = RECT()
        windll.user32.GetWindowRect(HWND(hwnd), byref(bounds))
        self._restore_bounds = (
            bounds.left,
            bounds.top,
            bounds.right - bounds.left,
            bounds.bottom - bounds.top,
        )
        monitor = windll.user32.MonitorFromWindow(HWND(hwnd), 2)
        info = _MonitorInfo()
        info.cbSize = sizeof(_MonitorInfo)
        windll.user32.GetMonitorInfoW(monitor, byref(info))
        area = info.rcWork
        windll.user32.SetWindowPos(
            HWND(hwnd),
            HWND(0),
            area.left,
            area.top,
            area.right - area.left,
            area.bottom - area.top,
            0x0014,
        )

    def restore(self) -> None:
        if self._window is None:
            return
        native = getattr(self._window, "native", None)
        if os.name != "nt" or native is None or self._restore_bounds is None:
            self._window.restore()
            return

        hwnd = int(native.Handle.ToInt64())
        bounds = self._restore_bounds
        windll.user32.SetWindowPos(HWND(hwnd), HWND(0), *bounds, 0x0014)
        self._restore_bounds = None

    def close(self) -> None:
        if self._window is None:
            return
        if self._tray is not None:
            self._window.hide()
            self._tray.show_background_notice()
        else:
            self._window.destroy()

    def show(self) -> None:
        if self._window is None:
            return
        self._window.show()
        if os.name == "nt":
            _activate_existing_window()

    def quit(self) -> None:
        self._quitting = True
        if self._tray is not None:
            self._tray.dispose()
        if self._window is not None:
            self._window.destroy()

    def on_closing(self, *_args: object) -> bool:
        if self._quitting or self._tray is None:
            return True
        self.close()
        return False


class WindowsTray:
    """Windows notification-area icon with an independent native message loop."""

    def __init__(self, controls: WindowControls, icon_path: Path) -> None:
        self.controls = controls
        self.icon_path = icon_path
        self._notify: Any | None = None
        self._notice_shown = False

    def start(self) -> None:
        if os.name != "nt" or self._notify is not None:
            return
        import pystray  # type: ignore[import-untyped]
        from PIL import Image

        menu = pystray.Menu(
            pystray.MenuItem("打开 RAgent", self._open, default=True),
            pystray.MenuItem("退出", self._exit),
        )
        notify = pystray.Icon("RAgent", Image.open(self.icon_path), "RAgent", menu)
        self._notify = notify
        notify.run_detached()

    def _open(self, _sender: object = None, _event: object = None) -> None:
        self.controls.show()

    def _exit(self, _sender: object = None, _event: object = None) -> None:
        self.controls.quit()

    def show_background_notice(self) -> None:
        if self._notify is None or self._notice_shown:
            return
        self._notify.notify("RAgent 正在后台运行", "RAgent")
        self._notice_shown = True

    def dispose(self) -> None:
        if self._notify is not None:
            self._notify.stop()
        self._notify = None


def _asset_path(name: str) -> Path:
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).parents[2]))
    return root / "assets" / name


class _MonitorInfo(Structure):
    _fields_ = [
        ("cbSize", DWORD),
        ("rcMonitor", RECT),
        ("rcWork", RECT),
        ("dwFlags", DWORD),
    ]


def _available_port(preferred: int = 2002) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", preferred))
        except OSError:
            probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _activate_existing_window(user32: Any | None = None) -> bool:
    """Restore the existing instance instead of making a second launch look inert."""
    if os.name != "nt" and user32 is None:
        return False
    api = user32 or windll.user32
    hwnd = api.FindWindowW(None, "RAgent")
    if not hwnd:
        return False
    api.ShowWindow(HWND(hwnd), 9)  # SW_RESTORE
    api.SetForegroundWindow(HWND(hwnd))
    return True


def _set_taskbar_identity(shell32: Any | None = None) -> None:
    """Give the frameless executable a stable, independent taskbar identity."""
    if os.name != "nt" and shell32 is None:
        return
    api = shell32 or windll.shell32
    api.SetCurrentProcessExplicitAppUserModelID("RAgent.Desktop")


def _claim_single_instance() -> int | None:
    """Keep concurrent desktop launches from racing for the same local server."""
    if os.name != "nt":
        return 1
    handle = int(windll.kernel32.CreateMutexW(None, False, "Local\\RAgentDesktop"))
    if not handle or int(windll.kernel32.GetLastError()) == 183:
        _activate_existing_window()
        if handle:
            windll.kernel32.CloseHandle(handle)
        return None
    return handle


def _desktop_settings() -> Settings:
    settings = Settings.from_env()
    data_root = Path(os.getenv("LOCALAPPDATA", Path.home())) / "VeriPatch"
    candidates = (
        data_root / "veripatch.sqlite3",
        data_root / "desktop.sqlite3",
        Path(tempfile.gettempdir()) / "RAgent" / "desktop.sqlite3",
    )
    for candidate in candidates:
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            # A development server may already hold the shared database. Probe
            # the WAL transition before handing the path to StudioStore; if a
            # profile directory is unavailable, try the next user-local path.
            with sqlite3.connect(candidate, timeout=0.25) as connection:
                connection.execute("PRAGMA journal_mode = WAL").fetchone()
            return replace(settings, database_path=candidate)
        except (OSError, sqlite3.Error):
            continue
    raise RuntimeError("无法创建桌面数据文件，请检查用户目录的写入权限。")


def _wait_until_ready(url: str, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=0.5) as response:  # noqa: S310
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError("VeriPatch local service did not start")


def _packaged_self_check(report_path: Path) -> int:
    """Check the built executable without a window, live credentials, or the desktop mutex."""
    import json

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from veripatch.studio_api import create_studio_router
    from veripatch.studio_domain import StudioSession
    from veripatch.studio_store import StudioStore

    try:
        with tempfile.TemporaryDirectory(prefix="ragent-package-check-") as directory:
            root = Path(directory)
            db = root / "state.db"
            StudioStore(db).save(StudioSession(
                session_id="check", repo_root=str(root), provider="deepseek",
                model="test", reasoning_effort="low",
            ), "created", {})
            app = FastAPI()
            app.include_router(create_studio_router(Settings(database_path=db)))
            with TestClient(app) as client:
                page = client.get("/studio")
                if page.status_code != 200:
                    raise RuntimeError("Packaged Vue page unavailable")
                import re

                asset = re.search(r'src="(/assets/[^" ]+\.js)"', page.text)
                if asset is None or client.get(asset.group(1)).status_code != 200:
                    raise RuntimeError("Packaged Vue JavaScript unavailable")
                url = "/studio-api/sessions/check/skills"
                content = (
                    "---\nname: package-check\ndescription: >-\n"
                    "  Check YAML\n  multiline parsing\n---\nNever execute scripts."
                )
                preview = client.post(url + "/preview", json={"content": content})
                if preview.status_code != 200:
                    raise RuntimeError(f"Packaged YAML preview failed: {preview.text}")
                imported = client.post(url, json={"content": content})
                if imported.status_code != 201:
                    raise RuntimeError(f"Packaged skill import failed: {imported.text}")
                modes = client.put(url, json={"modes": {"package-check": "auto"}})
                if modes.status_code != 200:
                    raise RuntimeError("Packaged skill modes unavailable")
                removed = client.delete(url + "/package-check",
                                        params={"version": imported.json()["version"]})
                if removed.status_code != 200:
                    raise RuntimeError("Packaged skill backup/delete unavailable")
        result = {"ok": True, "checks": ["vue_page", "vue_assets", "yaml_multiline",
                  "skill_preview", "skill_import", "skill_modes", "skill_backup_delete"]}
        code = 0
    except Exception as exc:
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        code = 1
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return code


def _packaged_skill_sandbox_check(report_path: Path) -> int:
    """Opt-in end-to-end check using the installed sandbox and external Python.

    Does not install accounts, change policy, or request a host fallback.
    Projects are disposable; the broker manages and releases its normal ACL leases.
    """
    import json
    import shutil

    from veripatch import skill_runtime, studio_sandbox, studio_skills
    from veripatch.studio_domain import StudioSession

    try:
        if os.name != "nt" or studio_sandbox.settings().mode != "required":
            raise RuntimeError("This check requires Windows with sandbox mode=required")
        python = shutil.which("python")
        if not python:
            raise RuntimeError("External Python is required; the EXE is not a Python interpreter")
        base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))
        example = base / "examples/skills/sales-report-demo"
        files = {path.relative_to(example).as_posix(): path.read_bytes()
                 for path in example.rglob("*") if path.is_file()}
        checks = []
        with tempfile.TemporaryDirectory(prefix="ragent-skill-sandbox-check-") as directory:
            root = Path(directory)
            installed = studio_skills.install_package(str(root), files)
            smoke = skill_runtime.run_smoke(
                str(root), "sales-report-demo", installed["version"], confirmed=True,
            )
            if not smoke["passed"]:
                raise RuntimeError(f"Snapshot smoke failed: {smoke['cases']}")
            skill_runtime.configure(
                str(root), "sales-report-demo", installed["version"], percent=100,
                auto_rollback=False, min_samples=5, failure_rate=0.5, max_latency_ms=0,
            )
            session = StudioSession(
                session_id="sandbox-version-check", repo_root=str(root), provider="deepseek",
                model="self-check", reasoning_effort="low", skill_task_id="self-check",
                skill_modes={"sales-report-demo": "auto"},
            )
            skill_runtime.bind(session, fresh=True)
            bound = skill_runtime.version_detail(str(root), "sales-report-demo",
                                                 installed["version"])
            checks.append("confirmed_snapshot_smoke_in_sandbox")

            def execute(folder: Path, *args: str):
                result = studio_sandbox.run(
                    root, [python, str(folder / "scripts/analyze.py"), *args],
                    dict(os.environ), timeout=60, encoding="utf-8",
                )
                if not getattr(result, "sandboxed", False) or result.returncode != 0:
                    raise RuntimeError(f"Sandbox execution failed: {result.stderr[-2000:]}")
                return result.stdout

            folder = Path(installed["base_directory"])
            if "--input" not in execute(folder, "--help"):
                raise RuntimeError("Script help unavailable")
            checks.append("imported_skill_help_in_sandbox")
            before = {key: (folder / key).read_bytes() for key in files}
            report = json.loads(execute(folder))
            if (report["completed_orders"], report["units"], report["revenue"]) != (4, 7, "269.80"):
                raise RuntimeError("Incorrect sales result")
            checks.append("imported_skill_analysis_in_sandbox")
            if json.loads(execute(Path(bound["base_directory"])))["revenue"] != "269.80":
                raise RuntimeError("Version snapshot execution failed")
            checks.append("version_snapshot_analysis_in_sandbox")
            # Reproduce the previous importer, which used a private mkdtemp.
            legacy = Path(tempfile.mkdtemp(prefix=".legacy-", dir=folder.parent))
            for relative, data in files.items():
                destination = legacy / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
            saved = studio_skills.remove(str(root), "sales-report-demo", installed["version"])
            if skill_runtime.bind(session, fresh=False)[0]["version"] != installed["version"]:
                raise RuntimeError("Paused task lost its version after removal")
            checks.append("task_version_survives_package_removal")
            legacy.rename(folder)
            repaired = studio_skills.repair_permissions(
                str(root), "sales-report-demo", installed["version"]
            )
            if not Path(repaired["backup"]).is_dir() or not Path(saved["backup"]).is_dir():
                raise RuntimeError("Repair failed to preserve backups")
            if json.loads(execute(folder))["revenue"] != "269.80":
                raise RuntimeError("Repaired legacy package failed")
            if before != {key: (folder / key).read_bytes() for key in files}:
                raise RuntimeError("Skill contents changed during execution or repair")
            checks.extend(["legacy_private_package_repaired", "repaired_skill_in_sandbox",
                           "unchanged_contents_and_backups"])
        result = {"ok": True, "checks": checks}
        code = 0
    except Exception as exc:
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        code = 1
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return code


def main() -> None:
    """Start the private API server and host it inside a native desktop window."""
    if len(sys.argv) == 3 and sys.argv[1] == "--self-check":
        raise SystemExit(_packaged_self_check(Path(sys.argv[2]).resolve()))
    if len(sys.argv) == 3 and sys.argv[1] == "--self-check-sandbox":
        raise SystemExit(_packaged_skill_sandbox_check(Path(sys.argv[2]).resolve()))
    instance_mutex = _claim_single_instance()
    if instance_mutex is None:
        return
    _set_taskbar_identity()
    try:
        import uvicorn
        import webview
    except ImportError as exc:
        raise SystemExit(
            "Install VeriPatch with the desktop extra: pip install .[desktop]"
        ) from exc

    port = _available_port()
    url = f"http://127.0.0.1:{port}"
    app = create_app(_desktop_settings())
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    server_thread = threading.Thread(target=server.run, name="veripatch-server", daemon=True)
    server_thread.start()
    _wait_until_ready(url)
    controls = WindowControls(app.state.sandbox_ui_token)
    window = webview.create_window(
        "RAgent",
        f"{url}/studio",
        js_api=controls,
        width=1440,
        height=940,
        min_size=(1080, 700),
        frameless=True,
        easy_drag=False,
        shadow=True,
        background_color="#e7e6e2",
        text_select=True,
    )
    assert window is not None
    controls._bind(window)
    tray = WindowsTray(controls, _asset_path("ragent.ico"))

    def start_tray() -> None:
        # Bind after pywebview has finished exposing the JS bridge. Keeping the
        # native tray object on the bridge during startup can make WebView2 walk
        # the object graph and prevent the main window from appearing.
        controls._bind_tray(tray)
        tray.start()

    try:
        webview.start(start_tray, gui="edgechromium", private_mode=False)
    finally:
        tray.dispose()
        server.should_exit = True
        server_thread.join(timeout=5)
        if os.name == "nt":
            windll.kernel32.CloseHandle(instance_mutex)


if __name__ == "__main__":
    main()
