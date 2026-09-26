from veripatch.desktop import (
    WindowControls,
    WindowsTray,
    _activate_existing_window,
    _set_taskbar_identity,
)


class FakeWindow:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def minimize(self) -> None:
        self.calls.append("minimize")

    def maximize(self) -> None:
        self.calls.append("maximize")

    def restore(self) -> None:
        self.calls.append("restore")

    def destroy(self) -> None:
        self.calls.append("close")

    def hide(self) -> None:
        self.calls.append("hide")

    def show(self) -> None:
        self.calls.append("show")


def test_window_controls_delegate_to_native_window() -> None:
    controls = WindowControls()
    window = FakeWindow()
    controls._bind(window)

    controls.minimize()
    controls.maximize()
    controls.restore()
    controls.close()

    assert window.calls == ["minimize", "maximize", "restore", "close"]


class FakeUser32:
    def __init__(self, hwnd: int) -> None:
        self.hwnd = hwnd
        self.calls: list[tuple[str, int]] = []

    def FindWindowW(self, _class_name: object, title: str) -> int:
        assert title == "RAgent"
        return self.hwnd

    def ShowWindow(self, hwnd: int, mode: int) -> None:
        self.calls.append(("restore", mode))

    def SetForegroundWindow(self, hwnd: int) -> None:
        self.calls.append(("foreground", int(getattr(hwnd, "value", hwnd))))


def test_existing_instance_is_restored_and_activated() -> None:
    api = FakeUser32(42)

    assert _activate_existing_window(api) is True
    assert api.calls == [("restore", 9), ("foreground", 42)]


def test_missing_existing_instance_is_not_activated() -> None:
    api = FakeUser32(0)

    assert _activate_existing_window(api) is False
    assert api.calls == []


def test_taskbar_identity_is_registered() -> None:
    values: list[str] = []

    class FakeShell32:
        def SetCurrentProcessExplicitAppUserModelID(self, value: str) -> None:
            values.append(value)

    _set_taskbar_identity(FakeShell32())

    assert values == ["RAgent.Desktop"]


def test_close_hides_to_tray_and_explicit_quit_destroys_window(tmp_path) -> None:
    controls = WindowControls()
    window = FakeWindow()
    tray = WindowsTray(controls, tmp_path / "ragent.ico")
    notices: list[bool] = []
    disposed: list[bool] = []
    tray.show_background_notice = lambda: notices.append(True)  # type: ignore[method-assign]
    tray.dispose = lambda: disposed.append(True)  # type: ignore[method-assign]
    controls._bind(window)
    controls._bind_tray(tray)

    controls.close()
    assert controls.on_closing() is False
    controls.quit()

    assert window.calls == ["hide", "hide", "close"]
    assert notices == [True, True]
    assert disposed == [True]


def test_tray_notice_only_appears_once(tmp_path) -> None:
    controls = WindowControls()
    tray = WindowsTray(controls, tmp_path / "ragent.ico")

    class FakeNotify:
        def __init__(self) -> None:
            self.calls = 0

        def notify(self, *_args) -> None:
            self.calls += 1

    notify = FakeNotify()
    tray._notify = notify

    tray.show_background_notice()
    tray.show_background_notice()

    assert notify.calls == 1
