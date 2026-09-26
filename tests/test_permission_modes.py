from pathlib import Path

import pytest

from veripatch.studio_agent import StudioAgent
from veripatch.studio_api import CreateStudioSession, UpdateStudioSession
from veripatch.studio_domain import StudioDecision, StudioSession
from veripatch.studio_permissions import fingerprint, requires_approval, session_rule
from veripatch.studio_store import StudioStore
from veripatch.workspace import SafeWorkspace


def session(root: Path, mode="important"):
    return StudioSession(
        session_id="permissions",
        repo_root=str(root),
        provider="openai",
        model="test",
        reasoning_effort="low",
        permission_mode=mode,
    )


@pytest.mark.parametrize("mode,expected", [("ask", True), ("important", False), ("full", False)])
def test_create_gate(tmp_path, mode, expected):
    s = session(tmp_path, mode)
    agent = StudioAgent(None, StudioStore(tmp_path / "state.sqlite3"))
    decision = StudioDecision(
        action="create", rationale="创建指定文件", path="sample.txt", content="ok"
    )
    agent._execute(s, SafeWorkspace(tmp_path), decision)
    assert (s.pending_permission is not None) == expected
    assert (tmp_path / "sample.txt").exists() != expected


@pytest.mark.parametrize("mode", ["ask", "important", "full"])
def test_read_never_prompts(tmp_path, mode):
    assert not requires_approval(
        session(tmp_path, mode), StudioDecision(action="read", rationale="读取", path="x")
    )


@pytest.mark.parametrize(
    "command,important",
    [
        (["python", "-m", "pytest", "-q"], False),
        (["npm", "run", "build"], False),
        (["pip", "install", "example"], True),
        (["git", "push"], True),
        (["python", "-c", "print('hello')"], True),
    ],
)
def test_command_matrix(tmp_path, command, important):
    d = StudioDecision(action="run_command", rationale="执行", command=command)
    assert requires_approval(session(tmp_path, "ask"), d)
    assert requires_approval(session(tmp_path), d) == important
    assert not requires_approval(session(tmp_path, "full"), d)


def test_once_is_exact_and_consumed(tmp_path):
    s = session(tmp_path, "ask")
    agent = StudioAgent(None, StudioStore(tmp_path / "state.sqlite3"))
    d = StudioDecision(action="create", rationale="创建", path="a.txt", content="a")
    s.once_grants.append(fingerprint(d))
    assert not requires_approval(s, d)
    other = d.model_copy(update={"content": "different"})
    assert requires_approval(s, other)
    agent._execute(s, SafeWorkspace(tmp_path), d)
    assert s.once_grants == []
    assert requires_approval(s, d)


def test_session_grant_survives_serialization(tmp_path):
    s = session(tmp_path, "ask")
    d = StudioDecision(action="create", rationale="创建", path="a", content="a")
    s.action_grants.append(fingerprint(d))
    restored = StudioSession.model_validate_json(s.model_dump_json())
    assert restored.permission_mode == "ask"
    assert not requires_approval(restored, d)
    assert requires_approval(restored, d.model_copy(update={"path": "b"}))


def test_session_grant_matches_safe_file_actions_in_same_directory(tmp_path):
    s = session(tmp_path, "ask")
    d = StudioDecision(action="create", rationale="创建", path="a.txt", content="a")
    s.action_grants.append(session_rule(d))
    assert not requires_approval(s, d.model_copy(update={"path": "b.txt", "content": "b"}))
    assert requires_approval(s, d.model_copy(update={"path": "nested/b.txt"}))


def test_session_grant_matches_allowlisted_command_family(tmp_path):
    s = session(tmp_path, "ask")
    d = StudioDecision(
        action="run_tests", rationale="验证", command=["python", "-m", "pytest", "-q"]
    )
    s.action_grants.append(session_rule(d))
    assert not requires_approval(
        s,
        d.model_copy(update={"command": ["python", "-m", "pytest", "tests/test_app.py", "-q"]}),
    )
    assert requires_approval(
        s,
        d.model_copy(update={"command": ["python", "-m", "py_compile", "app.py"]}),
    )


def test_defaults_and_validation(tmp_path):
    assert session(tmp_path).permission_mode == "important"
    with pytest.raises(ValueError):
        session(tmp_path, "invalid")
    assert (
        CreateStudioSession(repo_root=str(tmp_path), provider="openai", model="x").permission_mode
        == "important"
    )


def test_legacy_exact_command_session_approval(tmp_path):
    s = session(tmp_path, "ask")
    command = ["python", "-c", "print('approved')"]
    s.approved_commands = [command]
    d = StudioDecision(action="run_command", rationale="执行", command=command)
    assert not requires_approval(s, d)
    assert requires_approval(s, d.model_copy(update={"command": ["python", "-c", "other"]}))
    assert (
        UpdateStudioSession(
            provider="openai", model="x", reasoning_effort="low", verification_mode="auto"
        ).permission_mode
        is None
    )


def test_settings_roundtrip_and_revoke_grants(tmp_path):
    from fastapi.testclient import TestClient

    from veripatch.api import create_app
    from veripatch.config import Settings

    settings = Settings(database_path=tmp_path / "settings.sqlite3")
    store = StudioStore(settings.database_path)
    s = session(tmp_path)
    s.action_grants = ["old"]
    s.approved_commands = [["python", "-c", "pass"]]
    store.save(s, "created", {})
    body = dict(
        provider="openai",
        model="gpt-5.6-sol",
        reasoning_effort="low",
        verification_mode="auto",
        test_command=[],
        permission_mode="ask",
    )
    with TestClient(create_app(settings)) as client:
        endpoint = f"/studio-api/sessions/{s.session_id}/settings"
        response = client.patch(endpoint, json=body)
        assert response.status_code == 200, response.text
        assert response.json()["permission_mode"] == "ask"
        body["permission_mode"] = "invalid"
        assert client.patch(endpoint, json=body).status_code == 422
    restored = store.load(s.session_id)
    assert restored.permission_mode == "ask"
    assert restored.action_grants == restored.approved_commands == []


def test_full_does_not_silently_override_task_denial(tmp_path):
    from veripatch.studio_domain import StudioTaskContract

    s = session(tmp_path, "full")
    s.task_contract = StudioTaskContract(
        objective="仅分析", intent="analysis", allowed_actions=["read", "respond"]
    )
    agent = StudioAgent(None, StudioStore(tmp_path / "state.sqlite3"))
    agent._execute(
        s,
        SafeWorkspace(tmp_path),
        StudioDecision(action="create", rationale="不应执行", path="bad", content="bad"),
    )
    assert not (tmp_path / "bad").exists()
    assert s.pending_permission is not None
    assert s.pending_permission.operation == "create"
    assert s.status == "waiting_permission"


@pytest.mark.parametrize("approved,scope", [(True, "once"), (True, "session"), (False, "once")])
def test_api_approval_resumes_exact_decision(tmp_path, monkeypatch, approved, scope):
    from fastapi.testclient import TestClient

    from veripatch.api import create_app
    from veripatch.config import Settings

    settings = Settings(database_path=tmp_path / "api.sqlite3")
    store = StudioStore(settings.database_path)
    s = session(tmp_path, "ask")
    agent = StudioAgent(None, store)
    d = StudioDecision(action="create", rationale="创建", path="approved.txt", content="ok")
    agent._execute(s, SafeWorkspace(tmp_path), d)
    request_id = s.pending_permission.request_id

    async def resume(self, state, content, **kwargs):
        assert kwargs["resume_after_permission"]
        exact = state.resume_decision
        state.resume_decision = None
        assert exact == d
        self._execute(state, SafeWorkspace(tmp_path), exact)
        state.status = "idle"
        self.store.save(state, "resumed", {})
        return state

    monkeypatch.setattr(StudioAgent, "handle", resume)
    monkeypatch.setattr("veripatch.studio_api.StudioProviderModel", lambda *args: None)
    with TestClient(create_app(settings)) as client:
        endpoint = f"/studio-api/sessions/{s.session_id}/permissions/{request_id}"
        response = client.post(endpoint, json={"approved": approved, "scope": scope})
        assert response.status_code == 200
        assert client.post(endpoint, json={"approved": approved}).status_code == 409
    restored = store.load(s.session_id)
    assert (tmp_path / "approved.txt").exists() == approved, restored.failure_reason
    assert restored.once_grants == []
    assert bool(restored.action_grants) == (approved and scope == "session")


def test_batch_stops_before_first_unapproved_write(tmp_path):
    s = session(tmp_path, "ask")
    agent = StudioAgent(None, StudioStore(tmp_path / "batch.sqlite3"))
    actions = [
        StudioDecision(action="create", rationale="创建", path=name, content="ok")
        for name in ["one", "two"]
    ]
    assert agent._execute_batch(s, SafeWorkspace(tmp_path), actions, "创建文件")
    assert not (tmp_path / "one").exists()
    assert not (tmp_path / "two").exists()
    assert s.pending_permission.decision["path"] == "one"


def test_real_loop_resumes_approved_edit_without_reasking(tmp_path):
    import asyncio

    from veripatch.studio_domain import StudioReply

    d = StudioDecision(action="create", rationale="按要求创建", path="sample.txt", content="ok")

    class Model:
        def __init__(self):
            self.decisions = [
                d,
                StudioDecision(
                    action="finish", rationale="完成", message="已创建 sample.txt；未运行命令。"
                ),
            ]

        async def decide(self, context):
            return StudioReply(decision=self.decisions.pop(0))

    model = Model()
    agent = StudioAgent(model, StudioStore(tmp_path / "loop.sqlite3"))
    s = session(tmp_path, "ask")
    message = "创建 sample.txt，内容是 ok，不运行任何命令。"
    asyncio.run(agent.handle(s, message))
    assert s.status == "waiting_permission"
    s.once_grants.append(fingerprint(d))
    s.resume_decision = d
    s.pending_permission = None
    asyncio.run(
        agent.handle(
            s, message, continuation=True, record_user_message=False, resume_after_permission=True
        )
    )
    assert (tmp_path / "sample.txt").read_text() == "ok"
    assert s.pending_permission is None
    assert s.once_grants == []
