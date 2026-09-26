"""Native Windows desktop shell for the local VeriPatch application."""

from __future__ import annotations

import os
import socket
import sys
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

    def __init__(self) -> None:
        # Keep native objects private. pywebview recursively exposes public
        # js_api attributes and traversing a WinForms window can hang WebView2.
        self._window: Any | None = None
        self._restore_bounds: tuple[int, int, int, int] | None = None
        self._tray: WindowsTray | None = None
        self._quitting = False

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
    data_root.mkdir(parents=True, exist_ok=True)
    return replace(settings, database_path=data_root / "veripatch.sqlite3")


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


def main() -> None:
    if "--mcp-server" in sys.argv:
        from veripatch.mcp_server import main as mcp_main

        sys.argv.remove("--mcp-server")
        mcp_main()
        return
    """Start the private API server and host it inside a native desktop window."""
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
    config = uvicorn.Config(
        create_app(_desktop_settings()),
        host="127.0.0.1",
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    server_thread = threading.Thread(target=server.run, name="veripatch-server", daemon=True)
    server_thread.start()
    _wait_until_ready(url)
    controls = WindowControls()
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
