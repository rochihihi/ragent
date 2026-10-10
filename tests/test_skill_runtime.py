"""Real filesystem release lifecycle, stable task bindings, and sandbox-only smoke gates."""

import json
import threading
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from veripatch import skill_runtime as runtime
from veripatch import studio_skills as skills
from veripatch.config import Settings
from veripatch.domain import TestOutcome as Outcome
from veripatch.studio_api import create_studio_router
from veripatch.studio_domain import StudioDecision, StudioSession
from veripatch.studio_store import StudioStore
from veripatch.workspace import SafeWorkspace, WorkspaceSecurityError


def package(text="old"):
    return {
        "SKILL.md": f"---\nname: demo\ndescription: Demonstration\n---\n{text}".encode(),
        "scripts/demo.py": f"print({text!r})".encode(),
        "references/rules.md": text.encode(),
    }


def state(root, name="task"):
    return StudioSession(
        session_id=name,
        repo_root=str(root),
        provider="deepseek",
        model="test",
        reasoning_effort="low",
        skill_modes={"demo": "auto"},
        skill_task_id=name,
    )


def install(root, text="old"):
    return skills.install_package(str(root), package(text))


def test_full_snapshot_survives_replace_and_delete(tmp_path):
    old = install(tmp_path)
    session = state(tmp_path)
    runtime.bind(session, fresh=True)
    loaded = skills.activate(session, "demo")
    assert loaded["version"] == old["version"]
    assert Path(loaded["base_directory"], "scripts/demo.py").read_bytes() == b"print('old')"
    new = skills.install_package(
        str(tmp_path), package("new"), replace=True, expected_version=old["version"]
    )
    assert runtime.bind(session, fresh=False)[0]["version"] == old["version"]
    next_session = state(tmp_path, "new-task")
    assert runtime.bind(next_session, fresh=True)[0]["version"] == new["version"]
    skills.remove(str(tmp_path), "demo", new["version"])
    assert runtime.bind(session, fresh=False)[0]["version"] == old["version"]
    assert runtime.bind(state(tmp_path), fresh=True) == []


def test_direct_paths_rewritten_before_approval_and_tamper_fails(tmp_path):
    info = install(tmp_path)
    session = state(tmp_path)
    runtime.bind(session, fresh=True)
    decision = StudioDecision(
        action="run_command",
        rationale="test",
        command=["python", ".agents/skills/demo/scripts/demo.py"],
    )
    mapped = runtime.guard_action(session, decision)
    assert "skill-runtime" in mapped.command[1]
    assert decision.command[1] == ".agents/skills/demo/scripts/demo.py"
    Path(mapped.command[1]).write_text("print('tampered')")
    with pytest.raises(ValueError, match="修改"):
        runtime.guard_action(session, mapped)
    assert runtime.releases(str(tmp_path))["demo"]["active"] == info["version"]


def test_reject_shell_original_and_protect_runtime(tmp_path):
    install(tmp_path)
    session = state(tmp_path)
    runtime.bind(session, fresh=True)
    decision = StudioDecision(
        action="run_command",
        rationale="test",
        command=["cmd", "/c", "python .agents/skills/demo/scripts/demo.py"],
    )
    with pytest.raises(ValueError, match="shell"):
        runtime.guard_action(session, decision)
    with pytest.raises(WorkspaceSecurityError):
        SafeWorkspace(tmp_path)._assert_write_allowed(runtime.runtime_root(str(tmp_path)))


def test_watcher_debounce_and_last_good(tmp_path):
    old = install(tmp_path)
    runtime.observe(str(tmp_path), now=0)
    guide = tmp_path / ".agents/skills/demo/SKILL.md"
    guide.write_text("---\nname: demo", encoding="utf-8")
    runtime.observe(str(tmp_path), now=1)
    runtime.observe(str(tmp_path), now=1.5)
    assert runtime.releases(str(tmp_path))["demo"]["active"] == old["version"]
    runtime.observe(str(tmp_path), now=3)
    assert runtime.releases(str(tmp_path))["demo"]["watch_error"]
    guide.write_bytes(package("new")["SKILL.md"])
    runtime.observe(str(tmp_path), now=4)
    runtime.observe(str(tmp_path), now=6)
    release = runtime.releases(str(tmp_path))["demo"]
    assert release["active"] != old["version"]
    assert "watch_error" not in release
    count = len(release["versions"])
    runtime.observe(str(tmp_path), now=7)
    assert len(runtime.releases(str(tmp_path))["demo"]["versions"]) == count


def test_changes_during_capture_never_publish(tmp_path):
    old = install(tmp_path)
    original = skills._files
    calls = 0

    def changed(path):
        nonlocal calls
        data = original(path)
        calls += 1
        if calls == 1:
            (path / "references/rules.md").write_bytes(b"concurrent")
        return data

    with patch.object(skills, "_files", changed), pytest.raises(skills.SkillConflictError):
        runtime.publish(str(tmp_path), "demo")
    assert runtime.releases(str(tmp_path))["demo"]["active"] == old["version"]


def test_bad_python_rejected_without_executing_or_replacing(tmp_path):
    old = install(tmp_path)
    broken = package("new")
    broken["scripts/demo.py"] = b"print("
    with pytest.raises(ValueError, match="静态检查"):
        skills.install_package(str(tmp_path), broken, replace=True, expected_version=old["version"])
    assert skills.detail(str(tmp_path), "demo")["version"] == old["version"]


def test_registry_failure_restores_old_directory(tmp_path):
    old = install(tmp_path)
    with (
        patch.object(runtime, "_save", side_effect=OSError("disk failure")),
        pytest.raises(OSError),
    ):
        skills.install_package(
            str(tmp_path), package("new"), replace=True, expected_version=old["version"]
        )
    assert skills.detail(str(tmp_path), "demo")["version"] == old["version"]
    assert runtime.releases(str(tmp_path))["demo"]["active"] == old["version"]


def test_restore_after_delete_and_version_conflict(tmp_path):
    old = install(tmp_path)
    new = skills.install_package(
        str(tmp_path), package("new"), replace=True, expected_version=old["version"]
    )
    with pytest.raises(skills.SkillConflictError):
        runtime.restore(str(tmp_path), "demo", old["version"], old["version"])
    result = runtime.restore(str(tmp_path), "demo", old["version"], new["version"])
    assert result["version"] == old["version"]
    assert Path(result["backup"]).is_dir()
    skills.remove(str(tmp_path), "demo", old["version"])
    assert runtime.restore(str(tmp_path), "demo", new["version"], None)["version"] == new["version"]


def test_canary_stable_per_task_and_auto_rollback(tmp_path):
    old = install(tmp_path)
    new = skills.install_package(
        str(tmp_path), package("new"), replace=True, expected_version=old["version"]
    )
    runtime.configure(
        str(tmp_path),
        "demo",
        new["version"],
        percent=30,
        auto_rollback=True,
        min_samples=3,
        failure_rate=0.5,
        max_latency_ms=0,
    )
    counts = {old["version"]: 0, new["version"]: 0}
    for index in range(100):
        session = state(tmp_path, str(index))
        first = runtime.bind(session, fresh=True)[0]["version"]
        counts[first] += 1
        assert runtime.bind(session, fresh=False)[0]["version"] == first
    assert all(10 < count < 90 for count in counts.values())
    for _ in range(3):
        runtime.record(str(tmp_path), "demo", new["version"], success=False, latency_ms=10)
    release = runtime.releases(str(tmp_path))["demo"]
    assert release["active"] == old["version"] and release["candidate"] is None
    # Watcher/restart must not immediately republish the rejected working-copy bytes.
    runtime.observe(str(tmp_path), now=0)
    runtime.observe(str(tmp_path), now=2)
    assert runtime.releases(str(tmp_path))["demo"]["active"] == old["version"]


def test_full_rollout_latency_and_no_flip_flop(tmp_path):
    old = install(tmp_path)
    new = skills.install_package(
        str(tmp_path), package("new"), replace=True, expected_version=old["version"]
    )
    runtime.configure(
        str(tmp_path),
        "demo",
        new["version"],
        percent=100,
        auto_rollback=True,
        min_samples=3,
        failure_rate=0.8,
        max_latency_ms=100,
    )
    for _ in range(5):
        runtime.record(str(tmp_path), "demo", new["version"], success=True, latency_ms=200)
    assert runtime.releases(str(tmp_path))["demo"]["active"] == old["version"]
    for _ in range(3):
        runtime.record(str(tmp_path), "demo", old["version"], success=False, latency_ms=500)
    release = runtime.releases(str(tmp_path))["demo"]
    assert release["active"] is None  # Never fall back to the known-bad newer version.
    assert release["versions"][new["version"]]["quarantined"]
    runtime.restore(str(tmp_path), "demo", old["version"], new["version"])
    assert runtime.releases(str(tmp_path))["demo"]["active"] == old["version"]


def test_watcher_skips_unchanged_package_content(tmp_path):
    install(tmp_path)
    runtime.observe(str(tmp_path), now=0)
    runtime.observe(str(tmp_path), now=2)
    with patch.object(runtime, "_capture", side_effect=AssertionError("unneeded read")):
        runtime.observe(str(tmp_path), now=3)


def test_saved_rollout_policy_applies_to_later_imports(tmp_path):
    old = install(tmp_path)
    second = skills.install_package(
        str(tmp_path), package("second"), replace=True, expected_version=old["version"]
    )
    runtime.configure(
        str(tmp_path),
        "demo",
        second["version"],
        percent=20,
        auto_rollback=False,
        min_samples=5,
        failure_rate=0.5,
        max_latency_ms=0,
    )
    third = skills.install_package(
        str(tmp_path), package("third"), replace=True, expected_version=second["version"]
    )
    release = runtime.releases(str(tmp_path))["demo"]
    assert release["active"] == old["version"]
    assert release["candidate"] == third["version"] and release["percent"] == 20


def test_restore_invalid_working_copy_and_old_task_continues(tmp_path):
    old = install(tmp_path)
    (tmp_path / ".agents/skills/demo/SKILL.md").write_text("broken")
    diagnostic = runtime.cached_listing(str(tmp_path))["diagnostics"][0]
    assert runtime.bind(state(tmp_path), fresh=True)[0]["version"] == old["version"]
    restored = runtime.restore(str(tmp_path), "demo", old["version"], diagnostic["version"])
    assert restored["version"] == old["version"]


def test_linked_working_copy_does_not_invalidate_last_good_snapshot(tmp_path):
    old = install(tmp_path)
    original = skills._linked
    with patch.object(
        skills,
        "_linked",
        side_effect=lambda p: (p.name == "demo" and p.parent.name == "skills") or original(p),
    ):
        runtime.observe(str(tmp_path), now=0)
        runtime.observe(str(tmp_path), now=2)
        assert runtime.releases(str(tmp_path))["demo"]["watch_error"]
        assert runtime.bind(state(tmp_path), fresh=True)[0]["version"] == old["version"]


def test_oversized_registry_never_replaces_good_registry(tmp_path):
    old = install(tmp_path)
    registry = runtime._load(str(tmp_path))
    registry["extra"] = "x" * (4 * 1024 * 1024)
    with pytest.raises(ValueError, match="容量上限"):
        runtime._save(str(tmp_path), registry)
    assert runtime.releases(str(tmp_path))["demo"]["active"] == old["version"]


def test_lease_counts_released_on_exception(tmp_path):
    info = install(tmp_path)
    session = state(tmp_path)
    with pytest.raises(RuntimeError), runtime.task_scope():
        runtime.bind(session, fresh=True)
        assert (
            runtime.releases(str(tmp_path))["demo"]["versions"][info["version"]]["references"] == 1
        )
        raise RuntimeError("cancel")
    assert runtime.releases(str(tmp_path))["demo"]["versions"][info["version"]]["references"] == 0


def test_metadata_cache_skips_resource_reads(tmp_path):
    install(tmp_path)
    runtime.cached_listing(str(tmp_path))
    with patch.object(skills, "_files", side_effect=AssertionError("unneeded read")):
        assert runtime.cached_listing(str(tmp_path))["items"][0]["name"] == "demo"


def test_smoke_requires_confirmation_sandbox_and_actual_execution(tmp_path):
    files = package()
    files["tests/smoke.json"] = json.dumps(
        [
            {"script": "scripts/demo.py", "args": [], "contains": "old"},
        ]
    ).encode()
    info = skills.install_package(str(tmp_path), files)
    assert runtime.releases(str(tmp_path))["demo"]["active"] is None
    with pytest.raises(ValueError, match="确认"):
        runtime.run_smoke(str(tmp_path), "demo", info["version"], confirmed=False)
    with patch("veripatch.studio_sandbox.settings") as settings:
        settings.return_value.mode = "off"
        with pytest.raises(ValueError, match="沙箱"):
            runtime.run_smoke(str(tmp_path), "demo", info["version"], confirmed=True)
        settings.return_value.mode = "required"
        with patch("veripatch.studio_tools.SafeStudioCommandRunner.run") as run:
            run.return_value = Outcome(
                command=["python"],
                exit_code=0,
                stdout="old",
                stderr="",
                duration_seconds=0.1,
                sandboxed=True,
            )
            assert runtime.run_smoke(str(tmp_path), "demo", info["version"], confirmed=True)[
                "passed"
            ]
            run.assert_called_once()
    runtime.configure(
        str(tmp_path),
        "demo",
        info["version"],
        percent=100,
        auto_rollback=False,
        min_samples=5,
        failure_rate=0.5,
        max_latency_ms=0,
    )
    assert runtime.releases(str(tmp_path))["demo"]["active"] == info["version"]


def test_first_release_cannot_canary_and_smoke_manifest_is_validated(tmp_path):
    info = install(tmp_path)
    with pytest.raises(ValueError, match="稳定版"):
        runtime.configure(
            str(tmp_path),
            "demo",
            info["version"],
            percent=10,
            auto_rollback=False,
            min_samples=5,
            failure_rate=0.5,
            max_latency_ms=0,
        )
    files = package()
    files["tests/smoke.json"] = b'[{"script":"../bad.py"}]'
    with pytest.raises(ValueError):
        runtime.check_package(files)


def test_history_release_restore_api_and_lifespan(tmp_path):
    info = install(tmp_path)
    store = StudioStore(tmp_path / "state.db")
    store.save(state(tmp_path), "created", {})
    app = FastAPI()
    app.include_router(create_studio_router(Settings(database_path=store.path)))
    url = "/studio-api/sessions/task/skills"
    with TestClient(app) as client:
        data = client.get(url).json()
        assert data["releases"]["demo"]["active"] == info["version"]
        assert client.get(url + "/history").status_code == 200
        assert client.put(url + "/demo/release", json={"version": "0" * 64}).status_code == 409
        assert (
            client.post(url + "/demo/smoke", json={"version": info["version"]}).status_code == 400
        )
        assert (
            client.post(
                url + "/demo/restore",
                json={"version": info["version"], "expected_version": info["version"]},
            ).status_code
            == 200
        )
    assert not any(t.name == "ragent-skill-watch" and t.is_alive() for t in threading.enumerate())


def test_concurrent_update_keeps_bound_old_script(tmp_path):
    old = install(tmp_path)
    session = state(tmp_path)
    runtime.bind(session, fresh=True)
    errors = []

    def update():
        try:
            skills.install_package(
                str(tmp_path), package("new"), replace=True, expected_version=old["version"]
            )
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=update)
    worker.start()
    for _ in range(10):
        item = runtime.version_detail(str(tmp_path), "demo", session.skill_bindings["demo"])
        assert Path(item["base_directory"], "scripts/demo.py").read_bytes() == b"print('old')"
    worker.join(timeout=5)
    assert not worker.is_alive() and not errors
