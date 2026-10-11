"""Identity-based replay, atomic acceptance and uncertain external outcomes."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from veripatch.config import Settings
from veripatch.studio_agent import StudioAgent
from veripatch.studio_api import create_studio_router
from veripatch.studio_domain import PermissionMode, StudioDecision, StudioMessage, StudioSession
from veripatch.studio_store import IdempotencyConflict, StudioStore
from veripatch.workspace import SafeWorkspace


def state(root):
    return StudioSession(
        session_id="s",
        repo_root=str(root),
        provider="openai",
        model="test",
        reasoning_effort="low",
        permission_mode="full",
    )


def app_store(root):
    store = StudioStore(root / "state.db")
    store.save(state(root), "created", {})
    app = FastAPI()
    app.include_router(create_studio_router(Settings(database_path=store.path)))
    return app, store


def test_atomic_receipt_rolls_back_state_and_event(tmp_path):
    store = StudioStore(tmp_path / "state.db")
    session = state(tmp_path)
    store.save(session, "before", {})
    with (
        pytest.raises(RuntimeError),
        store.request_transaction("s", "message", "id", {"content": "x"}) as receipt,
    ):
        session.title = "changed"
        store.save(session, "after", {})
        receipt.response = {"status": "accepted"}
        raise RuntimeError("commit not reached")
    assert store.load("s").title != "changed"
    assert len(store.events("s")) == 1
    assert store.unfinished_requests("s") == []
    with store.request_transaction("s", "message", "id", {"content": "x"}) as receipt:
        assert not receipt.replayed
        receipt.response = {"status": "accepted"}
    with StudioStore(store.path).request_transaction(
        "s",
        "message",
        "id",
        {"content": "x"},
    ) as replay:
        assert replay.replayed and replay.response == {"status": "accepted"}
    with (
        pytest.raises(IdempotencyConflict),
        store.request_transaction("s", "message", "id", {"content": "different"}),
    ):
        pass


def test_concurrent_claim_has_one_winner_and_batch_is_atomic(tmp_path):
    store = StudioStore(tmp_path / "state.db")
    barrier = threading.Barrier(6)

    def claim(_):
        barrier.wait()
        return store.claim_operation("s", "operation", "digest")[0]

    with ThreadPoolExecutor(max_workers=6) as pool:
        assert sum(pool.map(claim, range(6))) == 1
    assert not store.claim_batch("s", [("new", "a"), ("operation", "digest")])
    assert store.operation("s", "new") is None


def test_stale_session_cannot_repeat_completed_operation_or_change_parameters(tmp_path):
    store = StudioStore(tmp_path / "state.db")
    session = state(tmp_path)
    stale = session.model_copy(deep=True)
    agent = StudioAgent(None, store)
    decision = StudioDecision(
        call_id="create", action="create", rationale="x", path="file.txt", content="first"
    )
    agent._execute(session, SafeWorkspace(tmp_path), decision)
    (tmp_path / "file.txt").write_text("external", encoding="utf-8")
    agent._execute(stale, SafeWorkspace(tmp_path), decision)
    assert (tmp_path / "file.txt").read_text() == "external"
    assert stale.tool_call_results["create"]["status"] == "completed"
    stale.tool_call_results["create"] = {"status": "pending"}
    store.save(stale, "stale", {})
    assert store.operation("s", "create")["state"] == "completed"
    agent._execute(stale, SafeWorkspace(tmp_path), decision.model_copy(update={"content": "bad"}))
    assert stale.status == "paused"
    assert "参数" in stale.pause_reason
    assert (tmp_path / "file.txt").read_text() == "external"


def test_crash_after_external_effect_never_replays_same_identity(tmp_path, monkeypatch):
    store = StudioStore(tmp_path / "state.db")
    agent = StudioAgent(None, store)
    session = state(tmp_path)
    decision = StudioDecision(
        call_id="uncertain", action="create", rationale="x", path="file.txt", content="effect"
    )

    def crash(*args, **kwargs):
        (tmp_path / "file.txt").write_text("effect")
        raise KeyboardInterrupt("crash before acknowledgement")

    monkeypatch.setattr(agent, "_execute_once", crash)
    with pytest.raises(KeyboardInterrupt):
        agent._execute(session, SafeWorkspace(tmp_path), decision)
    assert store.operation("s", "uncertain")["state"] == "unknown"
    other = StudioAgent(None, StudioStore(store.path))
    reloaded = state(tmp_path)  # deliberately no in-memory ledger
    assert other._execute(reloaded, SafeWorkspace(tmp_path), decision)
    assert reloaded.status == "paused"
    assert len(reloaded.observations) == 0
    assert (tmp_path / "file.txt").read_text() == "effect"


def test_queued_message_replay_is_not_a_second_steer_and_survives_restart(tmp_path):
    app, store = app_store(tmp_path)
    session = store.load("s")
    session.status = "running"
    store.save(session, "running", {})
    client = TestClient(app)
    endpoint = "/studio-api/sessions/s/messages"
    body = {"content": "same", "message_id": "message-1"}
    first = client.post(endpoint, json=body)
    assert first.status_code == 202 and first.json()["status"] == "queued"
    assert client.post(endpoint, json=body).json() == first.json()
    assert len([e for e in store.events("s") if e["event_type"] == "steer_queued"]) == 1
    assert client.post(endpoint, json={**body, "content": "different"}).status_code == 409
    assert client.post(endpoint, json={**body, "message_id": "message-2"}).status_code == 202
    persisted = StudioStore(store.path)
    assert persisted.next_steer("s") == ("same", "message-1")
    assert persisted.next_steer("s") == ("same", "message-2")
    assert persisted.next_steer("s") is None


def test_duplicate_accepted_message_starts_once_and_new_id_runs_again(tmp_path, monkeypatch):
    calls = []

    async def handle(self, session, content, **options):
        calls.append(content)
        session.messages.append(
            StudioMessage(role="user", content=content, message_id=options.get("submission_id"))
        )
        session.status = "completed"
        self.store.save(session, "done", {})
        return session

    monkeypatch.setattr(StudioAgent, "handle", handle)
    monkeypatch.setattr("veripatch.studio_api.StudioProviderModel", lambda *args: None)
    app, store = app_store(tmp_path)

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as c:
            endpoint = "/studio-api/sessions/s/messages"
            body = {"content": "same", "message_id": "one"}
            assert (await c.post(endpoint, json=body)).json()["status"] == "accepted"
            for _ in range(100):
                if store.load("s").status == "completed":
                    break
                await asyncio.sleep(0.01)
            assert (await c.post(endpoint, json=body)).json()["status"] == "accepted"
            assert calls == ["same"]
            assert (await c.post(endpoint, json={**body, "message_id": "two"})).status_code == 202
            for _ in range(100):
                if len(calls) == 2:
                    break
                await asyncio.sleep(0.01)
            assert calls == ["same", "same"]

    asyncio.run(exercise())


@pytest.mark.parametrize("approved", [False, True])
def test_permission_receipt_replays_and_conflicting_decision_is_rejected(
    tmp_path,
    monkeypatch,
    approved,
):
    calls = []

    async def handle(self, session, content, **kwargs):
        calls.append(content)
        session.status = "completed"
        self.store.save(session, "done", {})
        return session

    monkeypatch.setattr(StudioAgent, "handle", handle)
    monkeypatch.setattr("veripatch.studio_api.StudioProviderModel", lambda *args: None)
    app, store = app_store(tmp_path)
    session = store.load("s")
    session.permission_mode = PermissionMode.ASK
    agent = StudioAgent(None, store)
    decision = StudioDecision(
        call_id="approval", action="create", rationale="x", path="file.txt", content="x"
    )
    agent._execute(session, SafeWorkspace(tmp_path), decision)
    request_id = session.pending_permission.request_id

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as c:
            url = f"/studio-api/sessions/s/permissions/{request_id}"
            body = {"approved": approved}
            first = await c.post(url, json=body)
            assert first.status_code == 200
            assert (await c.post(url, json=body)).json() == first.json()
            assert (await c.post(url, json={"approved": not approved})).status_code == 409
            await asyncio.sleep(0.05)
            assert len(calls) == int(approved)
            expected = "permission_approved" if approved else "permission_denied"
            assert len([e for e in store.events("s") if e["event_type"] == expected]) == 1

    asyncio.run(exercise())


def test_interrupted_accepted_payload_is_resumable_without_resubmission(tmp_path, monkeypatch):
    _, store = app_store(tmp_path)
    session = store.load("s")
    with store.request_transaction("s", "message", "lost", {"content": "original"}) as r:
        session.status = "running"
        store.save(session, "accepted", {})
        r.response = {"session_id": "s", "status": "accepted"}
    app = FastAPI()
    app.include_router(create_studio_router(Settings(database_path=store.path)))
    assert store.load("s").status == "paused"
    calls = []

    async def handle(self, session, content, **options):
        calls.append((content, options))
        session.messages.append(
            StudioMessage(role="user", content=content, message_id=options["submission_id"])
        )
        session.status = "completed"
        self.store.save(session, "done", {})
        return session

    monkeypatch.setattr(StudioAgent, "handle", handle)
    monkeypatch.setattr("veripatch.studio_api.StudioProviderModel", lambda *args: None)

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as c:
            replay = await c.post(
                "/studio-api/sessions/s/messages",
                json={
                    "content": "original",
                    "message_id": "lost",
                },
            )
            assert replay.json()["status"] == "accepted"
            assert calls == []  # replay is not permission to restart a process
            assert (await c.post("/studio-api/sessions/s/resume")).status_code == 202
            await asyncio.sleep(0.05)
            assert calls[0][0] == "original"
            assert calls[0][1]["submission_id"] == "lost"
            assert calls[0][1]["record_user_message"]

    asyncio.run(exercise())


def test_metric_replay_survives_eviction_and_repairs_failed_registry_write(tmp_path, monkeypatch):
    from veripatch import skill_runtime as runtime
    from veripatch import studio_skills as skills

    info = skills.install_package(
        str(tmp_path),
        {
            "SKILL.md": b"---\nname: demo\ndescription: Demo\n---\nDemo",
        },
    )
    args = (str(tmp_path), "demo", info["version"])
    original_save = runtime._save
    monkeypatch.setattr(runtime, "_save", lambda *args: (_ for _ in ()).throw(OSError("crash")))
    with pytest.raises(OSError):
        runtime.record(*args, success=False, latency_ms=12, operation_id="first")
    monkeypatch.setattr(runtime, "_save", original_save)
    runtime.record(*args, success=False, latency_ms=999, operation_id="first")
    samples = runtime._load(str(tmp_path))["skills"]["demo"]["versions"][info["version"]]["samples"]
    assert len(samples) == 1 and samples[0]["latency_ms"] == 12
    for i in range(105):
        runtime.record(*args, success=True, latency_ms=1, operation_id=str(i))
    before = runtime._load(str(tmp_path))["skills"]["demo"]["versions"][info["version"]]["samples"]
    runtime.record(*args, success=False, latency_ms=99, operation_id="first")
    assert (
        runtime._load(str(tmp_path))["skills"]["demo"]["versions"][info["version"]]["samples"]
        == before
    )


def test_metric_duplicate_failures_do_not_trigger_rollback(tmp_path):
    from veripatch import skill_runtime as runtime
    from veripatch import studio_skills as skills

    info = skills.install_package(
        str(tmp_path),
        {
            "SKILL.md": b"---\nname: demo\ndescription: Demo\n---\nDemo",
        },
    )
    project, version = str(tmp_path), info["version"]
    runtime.configure(
        project,
        "demo",
        version,
        percent=100,
        auto_rollback=True,
        min_samples=3,
        failure_rate=0.5,
        max_latency_ms=0,
    )
    for _ in range(5):
        runtime.record(project, "demo", version, success=False, latency_ms=1, operation_id="one")
    assert runtime.releases(project)["demo"]["active"] == version
    for identity in ("two", "three"):
        runtime.record(project, "demo", version, success=False, latency_ms=1, operation_id=identity)
    assert runtime.releases(project)["demo"]["active"] is None


def test_approval_path_mapping_does_not_change_identity(tmp_path, monkeypatch):
    from veripatch import skill_runtime as runtime
    from veripatch import studio_skills as skills
    from veripatch.studio_permissions import fingerprint

    skills.install_package(
        str(tmp_path),
        {
            "SKILL.md": b"---\nname: demo\ndescription: Demo\n---\nDemo",
            "scripts/demo.py": b"print('demo')",
        },
    )
    session = state(tmp_path)
    session.skill_modes = {"demo": "auto"}
    session.skill_task_id = "skill-task"
    session.permission_mode = PermissionMode.ASK
    runtime.bind(session, fresh=True)
    store = StudioStore(tmp_path / "state.db")
    agent = StudioAgent(None, store)
    decision = StudioDecision(
        call_id="skill-command",
        action="run_command",
        rationale="x",
        command=["python", ".agents/skills/demo/scripts/demo.py"],
    )
    agent._execute(session, SafeWorkspace(tmp_path), decision)
    resumed = StudioDecision.model_validate(session.pending_permission.decision)
    assert "skill-runtime" in resumed.command[1]
    assert agent._operation_digest(decision, session) == agent._operation_digest(resumed, session)
    session.pending_permission = None
    session.once_grants.append(fingerprint(resumed))
    calls = []
    monkeypatch.setattr(agent, "_execute_once", lambda *args, **kwargs: calls.append(True) or False)
    agent._execute(session, SafeWorkspace(tmp_path), resumed)
    assert calls == [True]
    assert store.operation("s", "skill-command")["state"] == "completed"


def test_pause_after_effect_is_not_a_pending_replay(tmp_path, monkeypatch):
    from veripatch.studio_domain import StudioObservation

    store = StudioStore(tmp_path / "state.db")
    session = state(tmp_path)
    agent = StudioAgent(None, store)
    decision = StudioDecision(call_id="effect-pause", action="list_files", rationale="x")
    calls = []

    def run(current, *args, **kwargs):
        calls.append(True)
        current.observations.append(StudioObservation(kind="files", summary="effect recorded"))
        agent._pause(current, "paused after committing evidence")
        return True

    monkeypatch.setattr(agent, "_execute_once", run)
    agent._execute(session, SafeWorkspace(tmp_path), decision)
    assert store.operation("s", decision.call_id)["state"] == "completed"
    agent._execute(state(tmp_path), SafeWorkspace(tmp_path), decision)
    assert calls == [True]


def test_mcp_argument_names_are_not_removed_from_operation_fingerprint(tmp_path):
    session = state(tmp_path)
    first = StudioDecision(
        action="mcp_call",
        rationale="reason",
        mcp_tool="server::tool",
        mcp_arguments={"rationale": "business parameter"},
    )
    other = first.model_copy(update={"mcp_arguments": {"rationale": "changed"}})
    assert StudioAgent._operation_digest(first, session) != StudioAgent._operation_digest(
        other, session
    )


def test_message_at_worker_shutdown_boundary_is_not_stranded(tmp_path, monkeypatch):
    completed = threading.Event()
    release = threading.Event()
    calls = []

    async def handle(self, session, content, **options):
        calls.append(content)
        session.messages.append(
            StudioMessage(role="user", content=content, message_id=options["submission_id"])
        )
        session.status = "completed"
        self.store.save(session, "done", {})
        if len(calls) == 1:
            completed.set()
            await asyncio.to_thread(release.wait, 5)
        return session

    monkeypatch.setattr(StudioAgent, "handle", handle)
    monkeypatch.setattr("veripatch.studio_api.StudioProviderModel", lambda *args: None)
    app, store = app_store(tmp_path)

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            url = "/studio-api/sessions/s/messages"
            try:
                await client.post(url, json={"content": "first", "message_id": "first"})
                assert await asyncio.to_thread(completed.wait, 3)
                queued = await client.post(url, json={"content": "tail", "message_id": "tail"})
                assert queued.json()["status"] == "queued"
            finally:
                release.set()
            for _ in range(200):
                if len(calls) == 2 and store.load("s").status == "completed":
                    break
                await asyncio.sleep(0.01)
            assert calls == ["first", "tail"]
            replay = await client.post(url, json={"content": "tail", "message_id": "tail"})
            assert replay.json() == queued.json()
            assert calls == ["first", "tail"]

    asyncio.run(exercise())
