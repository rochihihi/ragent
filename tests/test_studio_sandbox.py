"""Policy/broker integration tests, not proof of kernel isolation.

No test installs accounts, elevates, or changes the machine's firewall.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from veripatch import studio_sandbox as sandbox
from veripatch.mcp_client import call_project_tool
from veripatch.sandbox_management import create_sandbox_router
from veripatch.studio_agent import StudioAgent
from veripatch.studio_domain import StudioDecision, StudioSession
from veripatch.studio_permissions import fingerprint, requires_approval
from veripatch.studio_tools import SafeStudioCommandRunner, TerminalRegistry
from veripatch.testing import LocalPytestRunner
from veripatch.workspace import SafeWorkspace, WorkspaceSecurityError


def required() -> None:
    sandbox.save_settings(sandbox.SandboxSettings())


def test_approved_tool_execution_does_not_keep_preparing_context(tmp_path, monkeypatch):
    required()
    store = Mock()
    agent = StudioAgent(Mock(), store)
    session = StudioSession(
        session_id="resume", repo_root=str(tmp_path), provider="openai_official",
        model="test", reasoning_effort="medium", activity="preparing_context",
    )
    decision = StudioDecision(
        action="run_command", rationale="check", command=["whoami.exe"], call_id="approved",
    )

    def execute_once(current, workspace, action, **kwargs):
        assert current.activity == "executing_tool"
        assert current.tool_call_results["approved"]["status"] == "running"
        event = store.save.call_args
        assert event.args[1] == "tool_call_started"
        assert event.args[2]["summary"] == "正在启动沙箱并执行命令。"
        assert kwargs["permission_checked"] is True
        return False

    monkeypatch.setattr(agent, "_execute_once", execute_once)
    assert not agent._execute(session, SafeWorkspace(tmp_path), decision, permission_checked=True)
    assert session.tool_call_results["approved"]["status"] == "completed"


def test_missing_config_defaults_required() -> None:
    sandbox.configuration_path().unlink()
    assert sandbox.settings().mode == "required"


def test_invalid_config_never_disables_boundary() -> None:
    sandbox.configuration_path().write_text('{"mode":"auto"}', encoding="utf-8")
    with pytest.raises(sandbox.SandboxError, match="未降级"):
        sandbox.settings()


@pytest.mark.parametrize("domains", [["*"], ["https://example.com"], ["x;evil"], ["x:443"]])
def test_domains_are_not_shells_or_global_bypass(domains) -> None:
    with pytest.raises(ValueError):
        sandbox.SandboxSettings(allowed_domains=domains)


@pytest.mark.parametrize("paths", [["relative"], ["C:/users/*"]])
def test_read_paths_are_exact_absolute_paths(paths) -> None:
    with pytest.raises(ValueError):
        sandbox.SandboxSettings(tool_read_paths=paths)


def test_policy_change_invalidates_previous_action_grant() -> None:
    decision = StudioDecision(action="run_command", rationale="check", command=["python", "a.py"])
    old = fingerprint(decision)
    required()
    assert fingerprint(decision) != old
    required_key = fingerprint(decision)
    sandbox.save_settings(sandbox.SandboxSettings(allowed_domains=["example.com"]))
    assert fingerprint(decision) != required_key


def test_required_does_not_reuse_unbound_historical_command_grants(tmp_path) -> None:
    command = ["python", "a.py"]
    session = StudioSession(
        session_id="test",
        repo_root=str(tmp_path),
        provider="openai",
        model="test",
        reasoning_effort="low",
        approved_commands=[command],
    )
    decision = StudioDecision(action="run_command", rationale="check", command=command)
    assert not requires_approval(session, decision)  # Explicit legacy off-mode fixture.
    required()
    assert requires_approval(session, decision)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows npm shim")
def test_npx_shim_becomes_node_argv_not_host_shell(tmp_path, monkeypatch) -> None:
    required()
    monkeypatch.setattr(sandbox, "_node", lambda: sys.executable)
    shim = tmp_path / "tools/npx.cmd"
    script = shim.parent / "node_modules/npm/bin/npx-cli.js"
    script.parent.mkdir(parents=True)
    script.write_text("// fixture", encoding="utf-8")
    shim.touch()
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: str(shim))
    with sandbox.prepare(tmp_path, ["npx", "literal&arg", ""], {}) as launch:
        payload = json.loads(launch.request.read_text(encoding="utf-8"))
        assert payload["argv"] == [sys.executable, str(script), "literal&arg", ""]


def test_missing_runtime_never_spawns_requested_command(tmp_path, monkeypatch) -> None:
    required()
    monkeypatch.setattr(sandbox, "runtime_root", lambda: tmp_path / "missing-runtime")
    popen = Mock()
    monkeypatch.setattr(sandbox.subprocess, "Popen", popen)
    command = [sys.executable, "-c", "print('must not execute')"]
    with pytest.raises(sandbox.SandboxError, match="运行库"):
        SafeStudioCommandRunner(tmp_path, approved_commands=[command]).run(command)
    popen.assert_not_called()


def test_pytest_cannot_fallback_to_main_process(tmp_path, monkeypatch) -> None:
    required()
    monkeypatch.setattr(sandbox, "runtime_root", lambda: tmp_path / "missing-runtime")
    with pytest.raises(sandbox.SandboxError):
        LocalPytestRunner(tmp_path).run(["python", "-m", "pytest"])


def test_terminal_requires_runtime_even_after_approval(tmp_path, monkeypatch) -> None:
    required()
    monkeypatch.setattr(sandbox, "runtime_root", lambda: tmp_path / "missing-runtime")
    command = [sys.executable, "-c", "print('no')"]
    with pytest.raises(sandbox.SandboxError):
        TerminalRegistry().start(tmp_path, command, approved_commands=[command])


def test_mcp_connection_test_uses_same_boundary(tmp_path, monkeypatch) -> None:
    required()
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "external": {
                        "command": sys.executable,
                        "args": ["-c", "print('no')"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("RAGENT_MCP_CONFIG", str(path))
    monkeypatch.setattr(sandbox, "runtime_root", lambda: tmp_path / "missing-runtime")
    assert call_project_tool(tmp_path, "list_servers", {})[0]["name"] == "external"
    with pytest.raises(sandbox.SandboxError):
        call_project_tool(tmp_path, "external::list_tools", {})


def test_gui_never_bypasses_required_boundary(tmp_path) -> None:
    required()
    command = ["cmd", "/c", "start", "", "calculator.html"]
    with pytest.raises(sandbox.SandboxError, match="GUI"):
        SafeStudioCommandRunner(tmp_path, approved_commands=[command]).launch(command)


def test_native_file_grant_cannot_disable_or_delete_host_sandbox_policy(tmp_path) -> None:
    from veripatch.domain import FileEdit

    required()
    host = sandbox.configuration_path()
    original = host.read_text(encoding="utf-8")
    workspace = SafeWorkspace(
        tmp_path, approved_roots=[host.parent], approved_write_roots=[host.parent]
    )
    edit = FileEdit(path=str(host), old_text=original, new_text='{"mode":"off"}')
    with pytest.raises(WorkspaceSecurityError, match="宿主配置"):
        workspace.prepare_edits([edit], protect_tests=False)
    with pytest.raises(WorkspaceSecurityError, match="宿主配置"):
        workspace.delete_path(str(host.parent))
    assert sandbox.settings().mode == "required"


def test_broker_does_not_inherit_secret_or_node_loader(monkeypatch) -> None:
    monkeypatch.setenv("NODE_OPTIONS", "--require malicious.js")
    monkeypatch.setenv("NODE_PATH", "untrusted")
    monkeypatch.setenv("OPENAI_API_KEY", "private")
    environment = sandbox._broker_environment()
    assert not {"NODE_OPTIONS", "NODE_PATH", "OPENAI_API_KEY"} & environment.keys()


def test_request_preserves_argv_roots_and_removes_private_host_env(tmp_path, monkeypatch) -> None:
    required()
    monkeypatch.setattr(sandbox, "_node", lambda: sys.executable)
    read_only = tmp_path / "readonly"
    writable = tmp_path / "approved-write"
    argv = [sys.executable, "literal with spaces", 'quote"; $(danger)', "", "a&b"]
    with sandbox.prepare(
        tmp_path,
        argv,
        {"OPENAI_API_KEY": "private", "CI": "1"},
        read_paths=[read_only],
        write_paths=[writable],
        explicit_env={"MCP_EXPLICIT_TOKEN": "intended"},
    ) as launch:
        request_path = launch.request
        payload = json.loads(request_path.read_text(encoding="utf-8"))
        assert payload["argv"] == argv
        assert payload["env"] == {"CI": "1", "MCP_EXPLICIT_TOKEN": "intended"}
        assert payload["config"]["network"]["allowedDomains"] == []
        fs = payload["config"]["filesystem"]
        protected = str(sandbox.configuration_path().parent.resolve())
        assert protected in fs["denyRead"]
        if sys.platform == "win32":
            assert protected not in fs["denyWrite"]  # ReadDeny already denies ALL access.
        else:
            assert protected in fs["denyWrite"]
        assert str(read_only) in fs["allowRead"] and str(read_only) not in fs["allowWrite"]
        assert str(writable) in fs["allowWrite"]
        assert launch.argv[:3] == [
            sys.executable,
            str(sandbox.runtime_root() / "bridge.mjs"),
            "execute",
        ]
        assert "danger" not in " ".join(launch.argv)
    assert not request_path.exists()


def test_broad_grant_cannot_reallow_private_config(tmp_path, monkeypatch) -> None:
    required()
    monkeypatch.setattr(sandbox, "_node", lambda: sys.executable)
    with pytest.raises(sandbox.SandboxError, match="受保护"):
        sandbox.prepare(sandbox.configuration_path().parent.parent, [sys.executable], {})


@pytest.mark.skipif(sys.platform != "win32", reason="Windows shared-account lease")
def test_windows_execution_lease_is_exclusive(tmp_path, monkeypatch) -> None:
    required()
    monkeypatch.setattr(sandbox, "_node", lambda: sys.executable)
    with (
        sandbox.prepare(tmp_path, [sys.executable], {}),
        pytest.raises(sandbox.SandboxError, match="另一项任务"),
    ):
        sandbox.prepare(tmp_path, [sys.executable], {})
    with sandbox.prepare(tmp_path, [sys.executable], {}) as second:
        assert second.enabled


def test_sdk_failure_has_host_sandbox_diagnostic(tmp_path) -> None:
    status = tmp_path / "status"
    status.write_text('{"state":"error","message":"WFP not installed"}', encoding="utf-8")
    with (
        sandbox.SandboxLaunch([], {}, True, status=status) as launch,
        pytest.raises(sandbox.SandboxError, match="WFP"),
    ):
        launch.check_error()
    assert not status.exists()


def test_failure_strategy_never_suggests_unrestricted_retry() -> None:
    category, strategy, retryable = StudioAgent._classify_tool_failure(
        "tool_error", "SandboxError: dependencies not found"
    )
    assert category == "sandbox_unavailable" and not retryable
    assert "不得关闭沙箱" in strategy


def test_explicit_off_retains_argv_not_shell(tmp_path) -> None:
    command = [sys.executable, "-c", "import sys; print(repr(sys.argv[1:]))", "a&b", ""]
    outcome = SafeStudioCommandRunner(tmp_path, approved_commands=[command]).run(command)
    assert outcome.passed and not outcome.sandboxed
    assert "['a&b', '']" in outcome.stdout


def client() -> TestClient:
    app = FastAPI()
    app.include_router(create_sandbox_router("private-desktop-token"))
    return TestClient(app)


def ui_headers() -> dict[str, str]:
    return {"x-veripatch-ui": "1", "x-ragent-sandbox-key": "private-desktop-token"}


def test_management_requires_native_capability_and_explicit_confirmation() -> None:
    api = client()
    assert "private-desktop-token" not in api.get("/sandbox-api/settings").text
    assert api.put("/sandbox-api/settings", json={"settings": {"mode": "off"}}).status_code == 403
    assert (
        api.put(
            "/sandbox-api/settings",
            headers={"x-veripatch-ui": "1"},
            json={"settings": {"mode": "off"}, "confirm_unrestricted": True},
        ).status_code
        == 403
    )
    assert (
        api.put(
            "/sandbox-api/settings", headers=ui_headers(), json={"settings": {"mode": "off"}}
        ).status_code
        == 409
    )
    assert (
        api.put(
            "/sandbox-api/settings", headers=ui_headers(), json={"settings": {"mode": "required"}}
        ).status_code
        == 200
    )
    assert sandbox.settings().mode == "required"


def test_install_is_never_called_by_get_or_missing_confirmation(monkeypatch) -> None:
    manage = Mock(return_value={"cancelled": True})
    monkeypatch.setattr(sandbox, "management", manage)
    api = client()
    assert api.get("/sandbox-api/settings").status_code == 200
    assert api.post("/sandbox-api/install", headers=ui_headers(), json={}).status_code == 422
    manage.assert_not_called()
    assert api.post(
        "/sandbox-api/install", headers=ui_headers(), json={"confirm_system_changes": True}
    ).json()["cancelled"]
    manage.assert_called_once_with("install")


def test_real_unavailable_backend_cannot_write_target(tmp_path) -> None:
    """Read-only readiness check + refused tool; never invokes system install."""
    try:
        probe = sandbox.management("probe")
    except sandbox.SandboxError:
        pytest.skip("Runtime dependencies not installed; unit tests cover this case")
    if probe["ready"]:
        pytest.skip("This case specifically requires an unprovisioned backend")
    required()
    marker = tmp_path / "must-not-exist.txt"
    command = [
        sys.executable,
        "-c",
        "from pathlib import Path; Path('must-not-exist.txt').write_text('unsafe')",
    ]
    with pytest.raises(sandbox.SandboxError):
        SafeStudioCommandRunner(tmp_path, approved_commands=[command]).run(command)
    assert not marker.exists()


@pytest.mark.skipif(shutil.which("node") is None, reason="Node required for broker argv unit test")
def test_bridge_round_trips_argv_using_stub_not_real_isolation(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    request = tmp_path / "request.json"
    literals = ["", "a b", "a&b", 'quote"; $(danger)', "中文", "back\\slash"]
    request.write_text(
        json.dumps(
            {
                "argv": [
                    shutil.which("node"),
                    "-e",
                    "process.stdout.write(JSON.stringify(process.argv.slice(1)))",
                    *literals,
                ],
                "cwd": str(tmp_path),
                "env": {},
                "config": {},
                "interactive": False,
                "statusPath": str(tmp_path / "status"),
            }
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            shutil.which("node"),
            "--import",
            (root / "tests/fixtures/sandbox_runtime_stub.mjs").as_uri(),
            str(root / "sandbox/runtime/bridge.mjs"),
            "execute",
            str(request),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=15,
        check=True,
    )
    assert json.loads(result.stdout) == literals
    assert not request.exists()


@pytest.mark.skipif(sys.platform != "win32" or shutil.which("node") is None,
                    reason="Windows bootstrap protocol with stub")
@pytest.mark.parametrize("failure", [None, "dependency", "initialize", "grant", "cleanup", "restore", "restore_throw"])
def test_probe_checks_real_startup_protocol_and_runtime_only_lease(tmp_path, failure) -> None:
    """SDK stub proves ordering/cleanup only; it never changes any real ACL."""
    root = Path(__file__).resolve().parents[1]
    log = tmp_path / "acl.jsonl"
    environment = {**os.environ, "RAGENT_TEST_ACL_LOG": str(log)}
    if failure:
        environment[{
            "dependency": "RAGENT_TEST_DEPENDENCY_FAILURE",
            "initialize": "RAGENT_TEST_INIT_FAILURE",
            "grant": "RAGENT_TEST_GRANT_FAILURE",
            "cleanup": "RAGENT_TEST_CLEANUP_FAILURE",
            "restore": "RAGENT_TEST_RESTORE_FAILURE",
            "restore_throw": "RAGENT_TEST_RESTORE_THROW",
        }[failure]] = "1"
    result = subprocess.run(
        [shutil.which("node"), "--import",
         (root / "tests/fixtures/sandbox_runtime_stub.mjs").as_uri(),
         str(root / "sandbox/runtime/bridge.mjs"), "probe"],
        env=environment, capture_output=True, text=True, encoding="utf-8",
        timeout=15, check=True,
    )
    status = json.loads(result.stdout)
    assert status["ready"] is (failure is None)
    assert status["startupVerified"] is (failure in {None, "cleanup", "restore", "restore_throw"})
    if failure == "dependency":
        assert not log.exists()  # No permission changes for an uninstalled sandbox.
        return
    events = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert events[0]["event"] == "grant"
    assert events[0]["read"] == [str(root / "sandbox/runtime")]
    assert events[0]["write"] == []
    assert [event["event"] for event in events][-3:] == ["restore", "revoke", "reset"]
    if failure != "grant":
        assert events[1]["event"] == "initialize"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows shared-account lease")
def test_management_probe_cannot_overlap_execution(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sandbox, "_node", lambda: sys.executable)
    popen = Mock()
    monkeypatch.setattr(sandbox.subprocess, "Popen", popen)
    required()
    with (
        sandbox.prepare(tmp_path, [sys.executable], {}),
        pytest.raises(sandbox.SandboxError, match="另一项任务"),
    ):
        sandbox.management("probe")
    popen.assert_not_called()


def test_management_timeout_stops_known_process_tree(monkeypatch) -> None:
    monkeypatch.setattr(sandbox, "_node", lambda: sys.executable)
    process = Mock()
    process.communicate.side_effect = subprocess.TimeoutExpired("broker", 90)
    monkeypatch.setattr(sandbox.subprocess, "Popen", Mock(return_value=process))
    stop = Mock()
    monkeypatch.setattr(sandbox, "stop_process_tree", stop)
    with pytest.raises(sandbox.SandboxError, match="检查未完成"):
        sandbox.management("probe")
    stop.assert_called_once_with(process)


def test_frozen_prefix_does_not_grant_entire_desktop_distribution(tmp_path, monkeypatch) -> None:
    required()
    private = tmp_path / "private-_MEI"
    runtime = private / "sandbox/runtime"
    monkeypatch.setattr(sandbox, "runtime_root", lambda: runtime)
    monkeypatch.setattr(sandbox, "_node", lambda: str(runtime / "node.exe"))
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "base_prefix", str(private))
    with sandbox.prepare(tmp_path / "workspace", ["python", "-c", "pass"], {}) as launch:
        paths = json.loads(launch.request.read_text())["config"]["filesystem"]["allowRead"]
    assert str(runtime) in paths
    assert str(private) not in paths


@pytest.mark.skipif(sys.platform != "win32" or shutil.which("node") is None,
                    reason="Windows bootstrap protocol with stub")
def test_execution_initialization_failure_releases_bootstrap(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    request = tmp_path / "request.json"
    log = tmp_path / "acl.jsonl"
    request.write_text(json.dumps({
        "argv": ["must-not-execute"], "cwd": str(tmp_path), "env": {}, "config": {},
        "statusPath": str(tmp_path / "status"),
    }), encoding="utf-8")
    result = subprocess.run(
        [shutil.which("node"), "--import",
         (root / "tests/fixtures/sandbox_runtime_stub.mjs").as_uri(),
         str(root / "sandbox/runtime/bridge.mjs"), "execute", str(request)],
        env={**os.environ, "RAGENT_TEST_ACL_LOG": str(log), "RAGENT_TEST_INIT_FAILURE": "1"},
        capture_output=True, text=True, encoding="utf-8", timeout=15,
    )
    assert result.returncode == 125
    assert "WFP verification failure" in result.stderr
    events = [json.loads(line)["event"] for line in log.read_text().splitlines()]
    assert events == ["grant", "initialize", "restore", "revoke", "reset"]
    assert json.loads((tmp_path / "status").read_text())["state"] == "error"
