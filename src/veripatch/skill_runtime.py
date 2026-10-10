"""Versioned skill releases. No imported code is executed by the publisher or watcher."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import threading
import time
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any
from uuid import uuid4

from veripatch import studio_skills as skills

_REFERENCES: Counter[tuple[str, str, str]] = Counter()
_OBSERVED: dict[str, tuple[Any, float]] = {}
_PROCESSED: dict[str, Any] = {}
_CATALOG_CACHE: dict[str, tuple[Any, dict[str, Any]]] = {}
_LEASE_KEYS: ContextVar[list[tuple[str, str, str]] | None] = ContextVar(
    "skill_leases",
    default=None,
)
SETTLE_SECONDS = 1.0


def runtime_root(project: str) -> Path:
    root = Path(project).resolve() / ".agents" / "skill-runtime"
    if skills._linked(root) or not root.resolve().is_relative_to(Path(project).resolve()):
        raise ValueError("技能版本目录不能包含链接或指向项目外部")
    return root


def _path(project: str, name: str, version: str) -> Path:
    skills._name(name)
    if (
        not isinstance(version, str)
        or len(version) != 64
        or any(c not in "0123456789abcdef" for c in version)
    ):
        raise ValueError("无效技能版本")
    root = runtime_root(project)
    path = root / "versions" / name / version
    for parent in [path, path.parent, path.parent.parent]:
        if skills._linked(parent):
            raise ValueError("技能版本不能包含链接")
    return path


def _load(project: str) -> dict[str, Any]:
    path = runtime_root(project) / "registry.json"
    if not path.exists():
        return {"schema": 1, "skills": {}}
    if skills._linked(path) or path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("技能注册表不安全或过大")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema") != 1
        or not isinstance(value.get("skills"), dict)
    ):
        raise ValueError("技能注册表格式无效")
    return value


def _save(project: str, registry: dict[str, Any]) -> None:
    serialized = json.dumps(registry, ensure_ascii=False)
    if len(serialized.encode("utf-8")) > 4 * 1024 * 1024:
        raise ValueError("技能注册表已达容量上限，保留原版本；请备份并归档历史项目")
    root = runtime_root(project)
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / f".registry-{uuid4().hex}.tmp"
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, root / "registry.json")
    finally:
        temporary.unlink(missing_ok=True)


def check_package(files: dict[str, bytes]) -> dict[str, Any]:
    """Static smoke checks only. ast.parse does not import or execute Python files."""
    decoded = skills.decode_package(
        files=[
            {"path": name, "data": skills.base64.b64encode(data).decode()}
            for name, data in files.items()
        ]
    )
    python_files = 0
    for name, data in decoded.items():
        if name.endswith(".py"):
            try:
                ast.parse(data, filename=name)
            except (SyntaxError, ValueError) as exc:
                raise ValueError(f"Python 静态检查失败：{name}: {exc}") from exc
            python_files += 1
    cases = smoke_cases(decoded)
    return {
        "static": "passed",
        "python_files": python_files,
        "smoke_cases": len(cases),
        "executable_smoke": "not_run",
    }


def smoke_cases(files: dict[str, bytes]) -> list[dict[str, Any]]:
    if "tests/smoke.json" not in files:
        return []
    cases = json.loads(files["tests/smoke.json"].decode("utf-8-sig"))
    if not isinstance(cases, list) or not 1 <= len(cases) <= 10:
        raise ValueError("tests/smoke.json 需要包含 1–10 个样例")
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("冒烟样例必须为对象")
        script = skills._relative(case.get("script", ""))
        if script not in files or not script.endswith(".py"):
            raise ValueError("冒烟测试只支持包内 Python 脚本")
        args = case.get("args", [])
        if (
            not isinstance(args, list)
            or len(args) > 30
            or any(not isinstance(a, str) or len(a) > 1000 or "\0" in a for a in args)
        ):
            raise ValueError("冒烟测试参数无效")
        if not isinstance(case.get("contains", ""), str):
            raise ValueError("冒烟期望输出必须是字符串")
    return cases


def _signature(project: str) -> tuple[Any, ...]:
    root = skills.root_for(project)
    entries = []
    if not root.exists():
        return ()
    for current, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in [*dirs, *sorted(names)]:
            path = Path(current) / name
            linked = skills._linked(path)
            if linked and name in dirs:
                dirs.remove(name)
            info = path.lstat()
            entries.append(
                (
                    path.relative_to(root).as_posix(),
                    info.st_size,
                    info.st_mtime_ns,
                    info.st_ctime_ns,
                    linked,
                )
            )
            if len(entries) > 50_000:
                raise ValueError("技能目录条目过多")
    return tuple(entries)


def cached_listing(project: str) -> dict[str, Any]:
    """Avoid rereading binary resources on unchanged management refreshes."""
    with skills._LOCK:
        signature = _signature(project)
        key = str(Path(project).resolve())
        cached = _CATALOG_CACHE.get(key)
        if cached is None or cached[0] != signature:
            cached = (signature, skills.listing(project))
            _CATALOG_CACHE[key] = cached
        return copy.deepcopy(cached[1])


def _capture(project: str, name: str) -> tuple[dict[str, bytes], dict[str, Any]]:
    before = _signature(project)
    files = skills._files(skills._directory(project, name))
    checks = check_package(files)
    # A second content pass detects changes even when writers preserve timestamps.
    if before != _signature(project) or skills._version(files) != skills._version(
        skills._files(skills._directory(project, name))
    ):
        raise skills.SkillConflictError("技能文件仍在变化，本次不发布")
    return files, checks


def _snapshot(project: str, name: str, files: dict[str, bytes]) -> str:
    version = skills._version(files)
    target = _path(project, name, version)
    if target.exists():
        if skills._version(skills._files(target)) != version:
            raise ValueError("已保存的技能版本被修改，拒绝使用")
        return version
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = target.parent / f".stage-{uuid4().hex}"
    stage.mkdir()
    try:
        for relative, data in files.items():
            destination = stage / skills._relative(relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        stage.rename(target)
    finally:
        if stage.exists():
            skills.shutil.rmtree(stage)
    return version


def publish(
    project: str,
    name: str,
    *,
    reason: str = "management",
    percent: int | None = None,
    expected_version: str | None = None,
) -> dict[str, Any]:
    with skills._LOCK:
        files, checks = _capture(project, name)
        info = skills.parse(files["SKILL.md"].decode("utf-8-sig"))
        if info["name"] != name:
            raise ValueError("技能名称与目录不匹配")
        version = _snapshot(project, name, files)
        if expected_version and version != expected_version:
            raise skills.SkillConflictError("技能已变化，请刷新后发布")
        registry = _load(project)
        entry = registry["skills"].setdefault(
            name,
            {
                "active": None,
                "candidate": None,
                "percent": 0,
                "versions": {},
                "policy": {
                    "auto_rollback": False,
                    "min_samples": 5,
                    "failure_rate": 0.5,
                    "max_latency_ms": 0,
                },
            },
        )
        entry["versions"].setdefault(
            version,
            {
                "created_at": time.time(),
                "reason": reason,
                "checks": checks,
                "samples": [],
                "tasks": [],
            },
        )
        if percent is None:
            percent = entry.get("watch_percent", 100)
        if not 0 <= percent <= 100:
            raise ValueError("发布比例超出范围")
        if reason == "file_watch" and entry["versions"][version].get("quarantined"):
            raise ValueError("该版本已自动隔离，需在管理界面确认后重新发布")
        has_smoke = checks["smoke_cases"] > 0
        passed = entry["versions"][version]["checks"].get("executable_smoke") == "passed"
        if not entry["active"] and percent != 100:
            raise ValueError("首次发布必须使用 100%（没有稳定版可用于分流）")
        if percent < 100 and entry["active"] == version:
            if not entry.get("previous"):
                raise ValueError("没有不同的稳定版，不能进行灰度分流")
            entry["active"] = entry["previous"]
        if has_smoke and not passed:
            entry["candidate"], entry["percent"] = version, 0
            entry["notice"] = "新版待执行冒烟测试，尚未发布给新任务"
        elif percent == 100:
            if entry["active"] != version:
                entry["previous"] = entry["active"]
            entry["active"], entry["candidate"], entry["percent"] = version, None, 0
            entry["notice"] = "已发布"
        else:
            entry["candidate"], entry["percent"] = version, percent
            entry["notice"] = f"灰度发布 {percent}%"
        entry["deleted"] = False
        entry["observed"] = version
        entry["rollback_checked_version"] = None
        entry["evaluation_since"] = time.time()
        entry["versions"][version]["quarantined"] = False
        entry.pop("watch_error", None)
        _save(project, registry)
        return entry


def retire(project: str, name: str) -> None:
    with skills._LOCK:
        registry = _load(project)
        if name in registry["skills"]:
            registry["skills"][name]["deleted"] = True
            _save(project, registry)


def observe(project: str, *, now: float | None = None) -> None:
    """Polling watcher: coalesce changes; only publish after a quiet stable capture."""
    with skills._LOCK:
        now = time.monotonic() if now is None else now
        key = str(Path(project).resolve())
        signature = _signature(project)
        previous = _OBSERVED.get(key)
        if previous is None or previous[0] != signature:
            _OBSERVED[key] = (signature, now)
            return
        if now - previous[1] < SETTLE_SECONDS:
            return
        if _PROCESSED.get(key) == signature:
            return
        registry = _load(project)
        current = skills.root_for(project)
        names = (
            {p.name for p in current.iterdir() if p.is_dir() and not p.name.startswith(".")}
            if current.exists()
            else set()
        )
        for name in names:
            try:
                files, _ = _capture(project, name)
                version = skills._version(files)
                entry = registry["skills"].get(name, {})
                if entry.get("observed") == version and not entry.get("deleted"):
                    continue
                publish(project, name, reason="file_watch", percent=entry.get("watch_percent", 100))
                registry = _load(project)
                registry["skills"][name]["observed"] = version
                _save(project, registry)
            except (ValueError, OSError) as exc:
                registry = _load(project)
                if name in registry["skills"]:
                    registry["skills"][name]["watch_error"] = str(exc)[:1000]
                    _save(project, registry)
        for name in registry["skills"].keys() - names:
            retire(project, name)
        _PROCESSED[key] = signature


def releases(project: str) -> dict[str, Any]:
    with skills._LOCK:
        registry = _load(project)
        result = copy.deepcopy(registry["skills"])
        for name, entry in result.items():
            for version, record in entry["versions"].items():
                samples = record.pop("samples", [])
                tasks = record.pop("tasks", [])
                record["metrics"] = {
                    "executions": len(samples),
                    "errors": sum(not s["success"] for s in samples),
                    "mean_latency_ms": round(
                        sum(s["latency_ms"] for s in samples) / max(1, len(samples))
                    ),
                    "task_samples": len(tasks),
                    "task_successes": sum(t["success"] for t in tasks),
                }
                record["references"] = _REFERENCES[(str(Path(project).resolve()), name, version)]
        return result


def version_detail(project: str, name: str, version: str) -> dict[str, Any]:
    path = _path(project, name, version)
    files = skills._files(path)
    if not files or skills._version(files) != version:
        raise ValueError("技能版本丢失或被修改，请恢复可信版本")
    info = skills.parse(files["SKILL.md"].decode("utf-8-sig"))
    if info["name"] != name:
        raise ValueError("技能版本名称不一致")
    return {
        **info,
        "version": version,
        "location": str(path / "SKILL.md"),
        "base_directory": str(path),
        "source_directory": str(skills.root_for(project) / skills._name(name)),
        "files": [{"path": p, "size": len(data)} for p, data in sorted(files.items())],
    }


def bind(session: Any, *, fresh: bool) -> list[dict[str, Any]]:
    with skills._LOCK:
        if fresh:
            session.skill_bindings = {}
            session.active_skill_contents = {}
        registry = _load(session.repo_root)
        # First use migrates legacy installed packages without executing them.
        for item in cached_listing(session.repo_root)["items"]:
            if item["name"] not in registry["skills"]:
                publish(session.repo_root, item["name"], reason="initial_capture")
        registry = _load(session.repo_root)
        if fresh:
            for name, entry in registry["skills"].items():
                if entry.get("deleted") or skills.mode_for(session, name) == "disabled":
                    continue
                version = entry.get("active")
                # Stable assignment for one task; random task id, not session id.
                bucket = (
                    int(
                        hashlib.sha256(f"{session.skill_task_id}:{name}".encode()).hexdigest()[:8],
                        16,
                    )
                    % 100
                )
                if entry.get("candidate") and bucket < entry.get("percent", 0):
                    version = entry["candidate"]
                if version:
                    session.skill_bindings[name] = version
        else:
            # Migrate pre-upgrade paused tasks only if their cached version is available.
            for name, item in session.active_skill_contents.items():
                if name not in session.skill_bindings:
                    version = item.get("version")
                    path = _path(session.repo_root, name, version)
                    if not path.is_dir():
                        raise ValueError("旧任务技能版本无法恢复，请重新发送任务以加载新版")
                    session.skill_bindings[name] = version
        result = []
        for name, version in session.skill_bindings.items():
            item = version_detail(session.repo_root, name, version)
            if name in session.active_skill_contents:
                session.active_skill_contents[name] = item
            result.append(
                {
                    k: item[k]
                    for k in ("name", "description", "location", "compatibility", "version")
                }
            )
            leases = _LEASE_KEYS.get()
            key = (str(Path(session.repo_root).resolve()), name, version)
            if leases is not None and key not in leases:
                _REFERENCES[key] += 1
                leases.append(key)
        session.enabled_skills = [
            n for n in session.skill_bindings if skills.mode_for(session, n) == "pinned"
        ]
        return result


@contextmanager
def task_scope():
    keys: list[tuple[str, str, str]] = []
    token = _LEASE_KEYS.set(keys)
    try:
        yield
    finally:
        with skills._LOCK:
            for key in keys:
                _REFERENCES[key] -= 1
                if not _REFERENCES[key]:
                    del _REFERENCES[key]
        _LEASE_KEYS.reset(token)


def map_path(session: Any, value: str) -> str:
    if ".agents/skills/" not in value.replace("\\", "/").casefold():
        return value
    path = Path(value)
    try:
        resolved = (path if path.is_absolute() else Path(session.repo_root) / path).resolve()
    except (OSError, ValueError):
        return value
    for name, version in session.skill_bindings.items():
        source = skills._directory(session.repo_root, name).resolve()
        if resolved == source or resolved.is_relative_to(source):
            return str(_path(session.repo_root, name, version) / resolved.relative_to(source))
    return value


def guard_action(session: Any, decision: Any) -> Any:
    """Resolve direct package arguments before approval; never rewrite arbitrary shell code."""
    if not session.skill_bindings:
        return decision
    updates: dict[str, Any] = {}
    if decision.path and decision.action.value == "read":
        updates["path"] = map_path(session, decision.path)
    command = []
    for argument in decision.command or []:
        mapped = map_path(session, argument)
        if mapped == argument and "=" in argument:
            prefix, value = argument.split("=", 1)
            mapped = prefix + "=" + map_path(session, value)
        command.append(mapped)
    if command:
        updates["command"] = command
    for name, version in session.skill_bindings.items():
        snapshot = _path(session.repo_root, name, version)
        if any(
            str(snapshot).replace("\\", "/").casefold() in a.replace("\\", "/").casefold()
            for a in [updates.get("path", ""), *command]
        ):
            version_detail(session.repo_root, name, version)
    # Shell source referring to mutable originals cannot be safely token-rewritten.
    if command and any(".agents/skills/" in a.replace("\\", "/").casefold() for a in command):
        raise ValueError(
            "请使用本任务技能版本的绝对脚本路径，不要在 shell 字符串中引用可变技能目录"
        )
    return decision.model_copy(update=updates)


def configure(
    project: str,
    name: str,
    version: str,
    *,
    percent: int,
    auto_rollback: bool,
    min_samples: int,
    failure_rate: float,
    max_latency_ms: int,
) -> dict[str, Any]:
    if not (
        0 <= percent <= 100
        and 3 <= min_samples <= 100
        and 0 < failure_rate <= 1
        and 0 <= max_latency_ms <= 600_000
    ):
        raise ValueError("发布策略超出范围")
    with skills._LOCK:
        entry = publish(project, name, percent=percent, expected_version=version)
        registry = _load(project)
        entry = registry["skills"][name]
        entry["policy"] = {
            "auto_rollback": auto_rollback,
            "min_samples": min_samples,
            "failure_rate": failure_rate,
            "max_latency_ms": max_latency_ms,
        }
        entry["watch_percent"] = percent
        _save(project, registry)
        return entry


def record(
    project: str, name: str, version: str, *, success: bool, latency_ms: int, task: bool = False
) -> None:
    with skills._LOCK:
        registry = _load(project)
        entry = registry["skills"].get(name)
        if not entry or version not in entry["versions"]:
            return
        record = entry["versions"][version]
        field = "tasks" if task else "samples"
        record[field] = [
            *record.get(field, []),
            {"success": success, "latency_ms": latency_ms, "time": time.time()},
        ][-100:]
        samples = [
            s for s in record.get("samples", []) if s["time"] >= entry.get("evaluation_since", 0)
        ]
        policy = entry["policy"]
        if (
            policy["auto_rollback"]
            and len(samples) >= policy["min_samples"]
            and entry.get("rollback_checked_version") != version
        ):
            failures = sum(not s["success"] for s in samples) / len(samples)
            slow = (
                policy["max_latency_ms"] > 0
                and sum(s["latency_ms"] for s in samples) / len(samples) > policy["max_latency_ms"]
            )
            if failures >= policy["failure_rate"] or slow:
                if entry.get("candidate") == version:
                    entry["candidate"], entry["percent"] = None, 0
                    entry["notice"] = "灰度版触发阈值，已停止分流；已有任务仍使用原版本"
                    entry["rollback_checked_version"] = version
                    record["quarantined"] = True
                elif entry.get("active") == version:
                    previous = entry.get("previous")
                    if previous and entry["versions"][previous].get("quarantined"):
                        previous = None
                    entry["active"], entry["previous"] = previous, version
                    entry["notice"] = "执行指标触发阈值，已回退后续新任务；外部操作未撤销"
                    if previous is None:
                        entry["notice"] = "无可用稳定版本，已停止新任务加载；请手动恢复可信版本"
                    entry["rollback_checked_version"] = version
                    record["quarantined"] = True
        _save(project, registry)


def restore(project: str, name: str, version: str, expected_version: str | None) -> dict[str, Any]:
    with skills._LOCK:
        version_detail(project, name, version)
        if version not in _load(project)["skills"].get(name, {}).get("versions", {}):
            raise ValueError("没有此历史版本")
        files = skills._files(_path(project, name, version))
        target = skills._directory(project, name)
        result = skills.install_package(
            project,
            files,
            replace=target.exists(),
            expected_version=expected_version,
            release_percent=100,
        )
        publish(project, name, reason="manual_restore", percent=100, expected_version=version)
        return result


def run_smoke(project: str, name: str, version: str, *, confirmed: bool) -> dict[str, Any]:
    from veripatch import studio_sandbox
    from veripatch.studio_tools import SafeStudioCommandRunner, resolve_studio_executable

    if not confirmed or studio_sandbox.settings().mode != "required":
        raise ValueError("冒烟测试需明确确认，且必须启用默认沙箱；不会自动沙箱外执行")
    with skills._LOCK:
        version_detail(project, name, version)
        files = skills._files(_path(project, name, version))
        cases = smoke_cases(files)
        if not cases:
            raise ValueError("技能没有 tests/smoke.json，只有静态检查，不能标记执行测试通过")
    python = resolve_studio_executable("python")
    if not python:
        raise ValueError("需要外部 Python 解释器；不会自动安装")
    evidence = []
    for case in cases:
        version_detail(project, name, version)
        command = [
            python,
            str(_path(project, name, version) / case["script"]),
            *case.get("args", []),
        ]
        started = time.monotonic()
        try:
            result = SafeStudioCommandRunner(
                Path(project),
                approved_commands=[command],
                timeout_seconds=30,
            ).run(command)
            success = (
                result.sandboxed
                and result.exit_code == 0
                and case.get("contains", "") in result.stdout
            )
            evidence.append(
                {
                    "script": case["script"],
                    "passed": success,
                    "exit_code": result.exit_code,
                    "stdout": result.stdout[:2000],
                    "stderr": result.stderr[:2000],
                }
            )
        except Exception as exc:
            success = False
            evidence.append({"script": case["script"], "passed": False, "error": str(exc)[:1000]})
        record(
            project,
            name,
            version,
            success=success,
            latency_ms=int((time.monotonic() - started) * 1000),
        )
    passed = all(item["passed"] for item in evidence)
    with skills._LOCK:
        registry = _load(project)
        entry = registry["skills"][name]["versions"][version]
        entry["checks"]["executable_smoke"] = "passed" if passed else "failed"
        # Do not persist potentially sensitive script stdout to a registry/model context.
        entry["checks"]["smoke_at"] = time.time()
        _save(project, registry)
    return {
        "passed": passed,
        "cases": evidence,
        "notice": "测试完成；通过后可点击发布。测试产生的外部副作用不会自动撤销。",
    }


class Watcher:
    """One bounded thread per application, explicitly stopped with router lifespan."""

    def __init__(self) -> None:
        self.projects: set[str] = set()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def add(self, project: str) -> None:
        with skills._LOCK:
            self.projects.add(str(Path(project).resolve()))

    def start(self) -> None:
        if self.thread is None:
            self.thread = threading.Thread(
                target=self._loop,
                name="ragent-skill-watch",
                daemon=True,
            )
            self.thread.start()

    def _loop(self) -> None:
        while not self.stop_event.wait(0.5):
            with skills._LOCK:
                projects = list(self.projects)
            for project in projects:
                try:
                    observe(project)
                except (OSError, ValueError):
                    # Management exposes validation errors; never publish invalid bytes.
                    continue

    def close(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=3)
