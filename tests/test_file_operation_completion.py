import asyncio

import pytest

from veripatch.studio_agent import StudioAgent
from veripatch.studio_domain import (
    PermissionMode,
    StudioDecision,
    StudioReply,
    StudioSession,
    StudioTaskState,
    TaskVerificationPolicy,
)
from veripatch.studio_permissions import fingerprint
from veripatch.studio_store import StudioStore
from veripatch.workspace import SafeWorkspace


def setup_agent(tmp_path, model=None):
    session = StudioSession(
        session_id="delete",
        repo_root=str(tmp_path),
        provider="openai",
        model="test",
        reasoning_effort="low",
        permission_mode="full",
    )
    return (
        StudioAgent(model, StudioStore(tmp_path / "state.sqlite3")),
        session,
        SafeWorkspace(tmp_path),
    )


@pytest.mark.parametrize("kind", ["file", "empty_file", "directory"])
@pytest.mark.parametrize("repeat", [False, True])
def test_approved_delete_finishes_without_tests(tmp_path, kind, repeat):
    target = tmp_path / "probe.txt"
    if kind == "directory":
        target.mkdir()
    else:
        target.write_text("hello" if kind == "file" else "", encoding="utf-8")
    delete = StudioDecision(action="delete_path", path="probe.txt", rationale="删除指定目标")

    class Model:
        def __init__(self):
            self.decisions = (
                [delete]
                + ([delete] if repeat else [])
                + [
                    StudioDecision(
                        action="finish", rationale="删除完成", message="已删除 probe.txt。"
                    )
                ]
            )

        async def decide(self, context):
            return StudioReply(decision=self.decisions.pop(0))

    model = Model()
    agent, session, _ = setup_agent(tmp_path, model)
    session.permission_mode = PermissionMode.IMPORTANT
    message = "删除 probe.txt。"
    asyncio.run(agent.handle(session, message))
    assert session.status == "waiting_permission"
    assert target.exists()
    session.once_grants.append(fingerprint(delete))
    session.resume_decision = delete
    session.pending_permission = None
    asyncio.run(
        agent.handle(
            session,
            message,
            continuation=True,
            record_user_message=False,
            resume_after_permission=True,
        )
    )
    assert session.status == "completed", session.messages[-1].content
    assert not target.exists()
    assert sum(item.kind == "delete" for item in session.observations) == 1
    assert not any(
        item.kind in {"test", "command", "verification_gate"} for item in session.observations
    )
    assert not session.verification_passed
    assert "测试已通过" not in session.messages[-1].content


def test_explicit_tests_and_mixed_code_changes_still_require_tests(tmp_path):
    agent, session, workspace = setup_agent(tmp_path)
    (tmp_path / "probe.txt").write_text("hello", encoding="utf-8")
    agent._execute(
        session, workspace, StudioDecision(action="delete_path", path="probe.txt", rationale="删除")
    )
    assert not agent._requires_command_verification(session)
    assert agent._automatic_verification_decision(session) is None
    session.task_state = StudioTaskState(
        verification_policy=TaskVerificationPolicy.REQUIRED_BY_USER
    )
    assert agent._requires_command_verification(session)
    session.task_state = None
    agent._execute(
        session,
        workspace,
        StudioDecision(action="create", path="code.py", content="x = 1", rationale="新增代码"),
    )
    assert agent._requires_command_verification(session)


def test_stale_deletion_evidence_cannot_finish(tmp_path):
    agent, session, workspace = setup_agent(tmp_path)
    target = tmp_path / "probe.txt"
    target.write_text("hello", encoding="utf-8")
    agent._execute(
        session, workspace, StudioDecision(action="delete_path", path="probe.txt", rationale="删除")
    )
    target.write_text("new user content", encoding="utf-8")
    assert not agent._has_file_operation_evidence(session)
    assert agent._requires_command_verification(session)
    assert target.read_text() == "new user content"
    session.task_state = StudioTaskState(verification_policy=TaskVerificationPolicy.SKIPPED_BY_USER)
    assert agent._validate_task_contract(session)


def test_create_edit_delete_in_same_conversation(tmp_path):
    class Model:
        def __init__(self):
            self.decisions = [
                StudioDecision(
                    action="create", path="probe.txt", content="hello", rationale="创建"
                ),
                StudioDecision(action="finish", message="已创建 probe.txt。", rationale="完成"),
                StudioDecision(
                    action="edit",
                    path="probe.txt",
                    old_text="hello",
                    new_text="world",
                    rationale="修改",
                ),
                StudioDecision(action="finish", message="已修改 probe.txt。", rationale="完成"),
                StudioDecision(action="delete_path", path="probe.txt", rationale="删除"),
                StudioDecision(action="finish", message="已删除 probe.txt。", rationale="完成"),
            ]

        async def decide(self, context):
            return StudioReply(decision=self.decisions.pop(0))

    model = Model()
    agent, session, _ = setup_agent(tmp_path, model)
    session.permission_mode = PermissionMode.IMPORTANT
    for message in [
        "创建 probe.txt，内容为 hello。不运行任何命令。",
        "把 probe.txt 中的 hello 改为 world。不运行任何命令。",
    ]:
        asyncio.run(agent.handle(session, message))
        assert session.status == "completed"
    asyncio.run(agent.handle(session, "删除 probe.txt。"))
    assert session.status == "waiting_permission"
    approved = StudioDecision.model_validate(session.pending_permission.decision)
    session.once_grants.append(fingerprint(approved))
    session.resume_decision = approved
    session.pending_permission = None
    asyncio.run(
        agent.handle(
            session,
            "删除 probe.txt。",
            continuation=True,
            record_user_message=False,
            resume_after_permission=True,
        )
    )
    assert session.status == "completed", session.messages[-1].content
    assert not (tmp_path / "probe.txt").exists()
    assert not model.decisions
    assert not any(
        item.kind in {"test", "command", "verification_gate"} for item in session.observations
    )
