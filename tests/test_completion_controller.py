import asyncio
import os
from unittest.mock import patch

import pytest

from veripatch.studio_agent import StudioAgent
from veripatch.studio_domain import (
    ObservationOutcome,
    StudioAction,
    StudioDecision,
    StudioMessage,
    StudioObservation,
    StudioSession,
    StudioRequirement,
    StudioReply,
    StudioTaskContract,
    StudioTaskState,
    VerificationMode,
)
from veripatch.studio_store import StudioStore
from veripatch.studio_tools import TERMINALS, inspect_visible_processes
from veripatch.workspace import SafeWorkspace
from veripatch import studio_completion as completion


def setup(tmp_path):
    s = StudioSession(
        session_id="completion",
        repo_root=str(tmp_path),
        provider="openai",
        model="test",
        reasoning_effort="low",
        permission_mode="full",
        status="running",
    )
    return StudioAgent(None, StudioStore(tmp_path / "state.sqlite3")), s, SafeWorkspace(tmp_path)


def test_response_gate_requires_effect_after_stale_tool_failure(tmp_path):
    _, session, _ = setup(tmp_path)
    session.task_contract = StudioTaskContract(objective="修改文件", intent="change")
    session.observations.extend([
        StudioObservation(kind="tool_error", summary="old failure", payload={"action": "read"}),
        StudioObservation(kind="create", summary="new file", payload={"path": "new.py"}),
    ])
    check = completion.assess_response(session, ["还需要完成任务效果"])
    assert check.state == "blocked"
    session.observations.append(StudioObservation(
        kind="tool_error", summary="current failure", payload={"action": "run_tests"},
    ))
    assert completion.assess_response(session, ["还需要完成任务效果"]).state == "ready"


def test_text_edit_move_copy_complete_without_commands(tmp_path):
    agent, session, workspace = setup(tmp_path)
    for d in [
        StudioDecision(action="create", path="a.txt", content="hello", rationale="create"),
        StudioDecision(
            action="edit", path="a.txt", old_text="hello", new_text="world", rationale="edit"
        ),
        StudioDecision(action="move_file", path="a.txt", destination="b.txt", rationale="move"),
        StudioDecision(action="copy_file", path="b.txt", destination="c.txt", rationale="copy"),
    ]:
        before = session.action_epoch
        agent._execute(session, workspace, d)
        assert session.action_epoch == before + 1
    assert agent._has_file_operation_evidence(session)
    assert agent._execute(
        session,
        workspace,
        StudioDecision(action="finish", rationale="done", message="已完成文件操作。"),
    )
    assert session.status == "completed"
    assert not session.verification_passed


def test_completion_rejection_stops_without_new_evidence(tmp_path):
    agent, session, workspace = setup(tmp_path)
    agent._execute(
        session,
        workspace,
        StudioDecision(action="create", path="a.py", content="x=1", rationale="create"),
    )
    finish = StudioDecision(action="finish", rationale="done", message="已完成。")
    assert not agent._execute(session, workspace, finish)
    assert agent._execute(session, workspace, finish)
    assert session.status == "paused"
    assert "没有产生新的文件变更" in session.messages[-1].content
    assert "请提出具体的修改要求" not in session.messages[-1].content


def test_rejected_finish_exposes_unmet_goal_and_refreshes_files(tmp_path):
    (tmp_path / "old.txt").write_text("existing", encoding="utf-8")
    agent, session, _ = setup(tmp_path)
    session.status = "idle"

    class Model:
        def __init__(self):
            self.contexts = []

        async def decide(self, context):
            self.contexts.append(context)
            step = len(self.contexts)
            if step == 1:
                return StudioReply(
                    decision=StudioDecision(action="finish", rationale="mistook old file for new", message="完成"),
                )
            if step == 2:
                assert context["completion"]["state"] == "blocked"
                assert context["completion"]["unmet"]
                assert "不要再次 finish" in context["instruction"]
                return StudioReply(
                    decision=StudioDecision(action="create", path="new.txt", content="new", rationale="create new artifact"),
                )
            assert "new.txt" in context["files"]
            assert context["completion"]["state"] == "ready"
            return StudioReply(
                decision=StudioDecision(action="finish", rationale="new file exists", message="已创建 new.txt"),
            )

    model = Model()
    agent.model = model
    result = asyncio.run(agent.handle(session, "新建一个说明文件"))
    assert result.status == "completed"
    assert result.turn_changed_files == ["new.txt"]
    assert len(model.contexts) == 3


def test_repeating_paused_request_keeps_its_file_evidence(tmp_path):
    agent, session, workspace = setup(tmp_path)
    request = "新建一个说明文件"
    agent._execute(
        session,
        workspace,
        StudioDecision(action="create", path="new.txt", content="new", rationale="create"),
    )
    session.task_contract = StudioTaskContract(
        objective=request,
        intent="change",
        requirements=[StudioRequirement(key="workspace_change", description="创建说明文件")],
    )
    session.plan = StudioAgent._build_plan(session.verification_mode, session.task_contract, request)
    session.messages.append(StudioMessage(role="user", content=request))
    session.status = "paused"

    class Model:
        async def decide(self, context):
            assert context["changed_files"] == ["new.txt"]
            return StudioReply(
                decision=StudioDecision(action="finish", rationale="existing task evidence", message="已创建 new.txt"),
            )

    agent.model = Model()
    result = asyncio.run(agent.handle(session, request))
    assert result.status == "completed"
    assert result.turn_changed_files == ["new.txt"]


def test_visible_terminal_launch_completes_without_second_launch(tmp_path):
    agent, session, workspace = setup(tmp_path)
    session.verification_mode = VerificationMode.QUICK
    agent._execute(
        session,
        workspace,
        StudioDecision(action="create", path="calculator.py", content="print('ok')\n", rationale="create"),
    )
    session.task_contract = StudioTaskContract(
        objective="创建计算器并打开",
        intent="change",
        requirements=[
            StudioRequirement(key="workspace_change", description="创建计算器"),
            StudioRequirement(key="launch_after_change", description="打开新产物"),
        ],
    )
    with patch("veripatch.studio_agent.TERMINALS.start", return_value={
        "terminal_id": "calculator-terminal", "pid": 1234, "running": True,
        "output": "", "window_confirmed": True,
    }) as start:
        terminal = agent._execute(
            session,
            workspace,
            StudioDecision(action="start_terminal", command=["python", "calculator.py"], rationale="open"),
        )
    assert terminal is False
    assert session.status == "running"
    assert start.call_args.kwargs["expect_window"] is True
    assert len([item for item in session.observations if item.kind == "terminal"]) == 1
    assert not any(item.kind == "command" for item in session.observations)


def test_running_terminal_blocks_duplicate_detached_launch(tmp_path):
    agent, session, workspace = setup(tmp_path)
    agent._execute(
        session,
        workspace,
        StudioDecision(action="create", path="calculator.py", content="print('ok')\n", rationale="create"),
    )
    session.observations.append(StudioObservation(
        kind="terminal", summary="already started",
        payload={"terminal_id": "existing", "launch_target": "calculator.py", "running": True,
                 "window_confirmed": False, "command": ["python", "calculator.py"]},
    ))
    with patch("veripatch.studio_agent.TERMINALS.poll", return_value={"running": True}), patch(
        "veripatch.studio_agent.SafeStudioCommandRunner.launch"
    ) as launch:
        terminal = agent._execute(
            session,
            workspace,
            StudioDecision(
                action="run_command", command=["cmd", "/c", "start", "", "python", "calculator.py"],
                rationale="open again",
            ),
        )
    assert terminal is False
    assert session.observations[-1].kind == "launch_reused"
    launch.assert_not_called()


def test_changed_python_app_uses_one_detached_launch(tmp_path):
    from veripatch.domain import TestOutcome

    agent, session, workspace = setup(tmp_path)
    agent._execute(
        session, workspace,
        StudioDecision(action="create", path="calculator.py", content="print('ok')\n", rationale="create"),
    )
    session.task_contract = StudioTaskContract(
        objective="创建计算器并打开",
        intent="change",
        requirements=[
            StudioRequirement(key="workspace_change", description="创建计算器"),
            StudioRequirement(key="launch_after_change", description="打开新产物"),
        ],
    )
    with patch("veripatch.studio_agent.SafeStudioCommandRunner.run") as run, patch(
        "veripatch.studio_agent.SafeStudioCommandRunner.launch",
        return_value=TestOutcome(
            command=["cmd", "/c", "start", "", "python", "calculator.py"],
            exit_code=0,
            stdout="Started process 123; window_confirmed=true",
            stderr="",
            duration_seconds=0,
        ),
    ) as launch:
        agent._execute(
            session, workspace,
            StudioDecision(action="run_command", command=["python", "calculator.py"], rationale="open"),
        )
        agent._execute(
            session, workspace,
            StudioDecision(action="run_command", command=["cmd", "/c", "start", "", "python", "calculator.py"], rationale="open again"),
        )
    run.assert_not_called()
    launch.assert_called_once_with(["cmd", "/c", "start", "", "python", "calculator.py"])
    assert session.observations[-1].kind == "launch_reused"
    assert session.verification_passed is False
    assert next(item for item in session.observations if item.kind == "command").payload[
        "execution_role"
    ] == "launch"
    assert not StudioAgent._validate_task_contract(session)


def test_uncertain_launch_is_inspected_instead_of_started_again(tmp_path):
    from veripatch.domain import TestOutcome

    agent, session, workspace = setup(tmp_path)
    agent._execute(
        session, workspace,
        StudioDecision(action="create", path="calculator.py", content="print('ok')\n", rationale="create"),
    )
    session.task_contract = StudioTaskContract(
        objective="创建计算器并打开", intent="change",
        requirements=[StudioRequirement(key="launch_after_change", description="打开新产物")],
    )
    outcome = TestOutcome(
        command=["cmd", "/c", "start", "", "python", "calculator.py"],
        exit_code=0, stdout="Process 123 did not create a visible window",
        stderr="", duration_seconds=0, terminal_id="tracked", pid=123,
        launch_state="running_unconfirmed", window_confirmed=False,
    )
    with patch("veripatch.studio_agent.SafeStudioCommandRunner.launch", return_value=outcome) as launch, patch(
        "veripatch.studio_agent.TERMINALS.inspect_launch",
        return_value={"terminal_id": "tracked", "pid": 123, "running": True,
                      "exit_code": None, "output": "", "window_confirmed": True,
                      "launch_state": "window_confirmed"},
    ):
        decision = StudioDecision(
            action="run_command", command=["python", "calculator.py"], rationale="open",
        )
        agent._execute(session, workspace, decision)
        assert StudioAgent._validate_task_contract(session) == ["打开新产物"]
        agent._execute(session, workspace, decision)

    launch.assert_called_once()
    assert session.observations[-1].kind == "launch_reused"
    assert StudioAgent._validate_task_contract(session) == []


def test_terminal_action_cannot_restart_a_tracked_gui_launch(tmp_path):
    agent, session, workspace = setup(tmp_path)
    agent._execute(
        session, workspace,
        StudioDecision(action="create", path="calculator.py", content="print('ok')\n", rationale="create"),
    )
    session.task_contract = StudioTaskContract(
        objective="创建计算器并打开", intent="change",
        requirements=[StudioRequirement(key="launch_after_change", description="打开新产物")],
    )
    session.observations.append(StudioObservation(
        kind="command", summary="启动中",
        payload={"command": ["python", "calculator.py"], "execution_role": "launch",
                 "launch_target": "calculator.py", "terminal_id": "tracked", "pid": 123,
                 "launch_state": "running_unconfirmed", "window_confirmed": False,
                 "exit_code": 0},
    ))
    with patch("veripatch.studio_agent.TERMINALS.inspect_launch", return_value={
        "terminal_id": "tracked", "pid": 123, "running": True,
        "exit_code": None, "output": "", "window_confirmed": False,
        "launch_state": "running_unconfirmed",
    }), patch("veripatch.studio_agent.TERMINALS.start") as start:
        agent._execute(
            session, workspace,
            StudioDecision(action="start_terminal", command=["python", "calculator.py"], rationale="open"),
        )
    start.assert_not_called()
    assert session.observations[-1].kind == "launch_reused"


def test_polling_existing_terminal_can_confirm_the_window(tmp_path):
    agent, session, workspace = setup(tmp_path)
    agent._execute(
        session, workspace,
        StudioDecision(action="create", path="calculator.py", content="print('ok')\n", rationale="create"),
    )
    session.task_contract = StudioTaskContract(
        objective="创建计算器并打开", intent="change",
        requirements=[StudioRequirement(key="launch_after_change", description="打开新产物")],
    )
    session.observations.append(StudioObservation(
        kind="terminal", summary="启动中",
        payload={"terminal_id": "tracked", "launch_target": "calculator.py",
                 "running": True, "window_confirmed": False},
    ))
    with patch("veripatch.studio_agent.TERMINALS.poll", return_value={
        "terminal_id": "tracked", "running": True, "exit_code": None,
        "output": "", "window_confirmed": True, "launch_state": "window_confirmed",
    }):
        agent._execute(
            session, workspace,
            StudioDecision(action="poll_terminal", terminal_id="tracked", rationale="inspect"),
        )
    assert session.observations[-1].payload["launch_target"] == "calculator.py"
    assert StudioAgent._validate_task_contract(session) == []


def test_process_inspection_refreshes_live_state(tmp_path):
    agent, session, workspace = setup(tmp_path)
    decision = StudioDecision(action="inspect_processes", query="calculator", rationale="check")
    with patch(
        "veripatch.studio_executor.inspect_visible_processes",
        side_effect=[[], [{"title": "calculator"}]],
    ) as inspect:
        agent._execute(session, workspace, decision)
        first = session.observations[-1]
        agent._execute(session, workspace, decision)
        second = session.observations[-1]

    assert inspect.call_count == 2
    assert first.kind == second.kind == "processes"
    assert first.payload["windows"] == []
    assert second.payload["windows"] == [{"title": "calculator"}]


def test_document_dispatch_completes_compound_task_without_window_pid(tmp_path):
    from veripatch.domain import TestOutcome

    agent, session, workspace = setup(tmp_path)
    agent._execute(
        session, workspace,
        StudioDecision(
            action="create", path="calculator.html",
            content="<html><body>Calculator</body></html>", rationale="create",
        ),
    )
    session.task_contract = StudioTaskContract(
        objective="创建计算器并打开", intent="change",
        requirements=[
            StudioRequirement(key="workspace_change", description="创建计算器"),
            StudioRequirement(key="launch_after_change", description="打开新产物"),
        ],
    )
    session.verification_passed = True
    decision = StudioDecision(
        action="run_command", command=["cmd", "/c", "start", "", "calculator.html"],
        rationale="open",
    )
    outcome = TestOutcome(
        command=decision.command, exit_code=0,
        stdout="Windows accepted open request for calculator.html", stderr="",
        duration_seconds=0, launch_state="dispatched", window_confirmed=None,
    )
    with patch("veripatch.studio_agent.SafeStudioCommandRunner.launch", return_value=outcome) as launch:
        agent._execute(session, workspace, decision)
        agent._execute(session, workspace, decision)

    launch.assert_called_once()
    assert StudioAgent._validate_task_contract(session) == []
    assert session.status == "running"
    assert session.observations[-1].kind == "launch_reused"
    assert not session.messages or "系统已接受打开" not in session.messages[-1].content


@pytest.mark.skipif(
    os.name != "nt" or os.environ.get("RAGENT_RUN_GUI_SMOKE") != "1",
    reason="Run explicitly on an interactive Windows desktop",
)
def test_real_windows_gui_launch_is_confirmed_and_not_repeated(tmp_path):
    agent, session, workspace = setup(tmp_path)
    agent._execute(
        session,
        workspace,
        StudioDecision(
            action="create", path="calculator.py",
            content=(
                "import tkinter as tk\n"
                "root = tk.Tk()\n"
                "root.title('RAgent GUI launch smoke')\n"
                "root.after(20000, root.destroy)\n"
                "root.mainloop()\n"
            ),
            rationale="create smoke window",
        ),
    )
    session.task_contract = StudioTaskContract(
        objective="创建并打开临时窗口", intent="change",
        requirements=[
            StudioRequirement(key="workspace_change", description="创建临时程序"),
            StudioRequirement(key="launch_after_change", description="打开临时窗口"),
        ],
    )
    decision = StudioDecision(
        action="run_command", command=["python", "calculator.py"], rationale="open",
    )
    terminal_id = None
    try:
        agent._execute(session, workspace, decision)
        launched = next(item for item in reversed(session.observations) if item.kind == "command")
        terminal_id = launched.payload["terminal_id"]
        assert launched.payload["window_confirmed"] is True
        pid = launched.payload["pid"]
        assert len([
            window for window in inspect_visible_processes("RAgent GUI launch smoke")
            if window["pid"] == pid
        ]) == 1

        agent._execute(session, workspace, decision)
        assert session.observations[-1].kind == "launch_reused"
        assert session.observations[-1].payload["pid"] == pid
        assert len([
            window for window in inspect_visible_processes("RAgent GUI launch smoke")
            if window["pid"] == pid
        ]) == 1

        agent._execute(
            session, workspace,
            StudioDecision(
                action="start_terminal", command=["python", "calculator.py"],
                rationale="open through terminal again",
            ),
        )
        assert session.observations[-1].kind == "launch_reused"
        assert len([
            window for window in inspect_visible_processes("RAgent GUI launch smoke")
            if window["pid"] == pid
        ]) == 1
    finally:
        if terminal_id is not None:
            TERMINALS.stop(workspace.root, terminal_id)


def test_prior_turn_change_cannot_complete_a_new_change_request(tmp_path):
    from veripatch.studio_completion import assess

    agent, session, workspace = setup(tmp_path)
    agent._execute(
        session,
        workspace,
        StudioDecision(action="create", path="old.py", content="x = 1", rationale="old turn"),
    )
    session.turn_changed_files = []
    session.turn_observation_start = len(session.observations)
    session.task_contract = StudioTaskContract(
        objective="新建一个计算器",
        intent="change",
        requirements=[
            StudioRequirement(key="workspace_change", description="产生本轮文件变更")
        ],
    )

    unmet = agent._validate_task_contract(session)
    assert unmet == ["产生本轮文件变更"]
    assert assess(session, unmet).state == "blocked"


def test_explicit_unchanged_completion_needs_current_turn_read(tmp_path):
    from veripatch.studio_domain import StudioMessage

    agent, session, _ = setup(tmp_path)
    session.task_contract = StudioTaskContract(
        objective="已有实现就不要重复修改",
        intent="change",
        requirements=[
            StudioRequirement(key="workspace_change", description="产生本轮文件变更")
        ],
    )
    session.messages.append(StudioMessage(role="user", content="已有实现就不要重复修改"))
    assert agent._validate_task_contract(session) == ["产生本轮文件变更"]
    session.observations.append(
        StudioObservation(kind="read", summary="已核对现有实现", payload={"path": "app.py"})
    )
    assert agent._validate_task_contract(session) == []


def test_failed_verification_pause_reports_the_real_failure(tmp_path):
    agent, session, _ = setup(tmp_path)
    session.observations.append(
        StudioObservation(
            kind="test",
            summary="测试未通过。",
            payload={
                "command": ["python", "-m", "pytest", "-q"],
                "exit_code": 1,
                "stdout": "FAILED test_app.py::test_add - assert -1 == 5",
                "stderr": "",
            },
        )
    )
    agent._pause(session, "完成条件未发生变化：关联测试曾失败")
    message = session.messages[-1].content
    assert "验证未通过" in message
    assert "python -m pytest -q" in message
    assert "FAILED test_app.py::test_add" in message
    assert "是否需要运行验证" not in message


def test_batch_commits_only_successful_epochs(tmp_path):
    agent, session, workspace = setup(tmp_path)
    agent._execute_batch(
        session,
        workspace,
        [
            StudioDecision(action="create", path="a.txt", content="ok", rationale="create"),
            StudioDecision(
                action="create", path="a.txt", content="overwrite", rationale="conflict"
            ),
            StudioDecision(action="create", path="b.txt", content="bad", rationale="must not run"),
        ],
        "创建文件",
    )
    assert session.action_epoch == 1
    assert not (tmp_path / "b.txt").exists()
    assert session.observations[-1].kind == "tool_error"


def test_verified_fallback_never_leaves_running(tmp_path):
    agent, session, workspace = setup(tmp_path)
    agent._finish_from_verified_evidence(session, workspace, "fallback")
    assert session.status != "running"


def test_public_loop_never_returns_orphan_running_state(tmp_path):
    agent, session, _ = setup(tmp_path)

    async def orphan(*args, **kwargs):
        return session

    agent._handle = orphan
    asyncio.run(agent.handle(session, "继续"))
    assert session.status == "paused"


def test_external_edits_invalidate_file_verification(tmp_path):
    agent, session, workspace = setup(tmp_path)
    agent._execute(
        session,
        workspace,
        StudioDecision(action="create", path="a.txt", content="ok", rationale="create"),
    )
    (tmp_path / "a.txt").write_text("user edit", encoding="utf-8")
    assert not agent._has_file_operation_evidence(session)


def test_batch_permission_resume_retains_remaining_actions(tmp_path):
    from veripatch.studio_domain import PermissionMode, StudioReply
    from veripatch.studio_permissions import fingerprint

    actions = [
        StudioDecision(action="create", path=name, content="ok", rationale="create")
        for name in ["a.txt", "b.txt"]
    ]

    class Model:
        async def decide(self, context):
            return StudioReply(
                decision=StudioDecision(
                    action="finish", rationale="done", message="已创建 a.txt 和 b.txt。"
                )
            )

    agent, session, workspace = setup(tmp_path)
    agent.model = Model()
    session.permission_mode = PermissionMode.ASK
    assert agent._execute_batch(session, workspace, actions, "创建 a.txt 和 b.txt")
    assert session.remaining_actions == actions[1:]
    for d in actions:
        session.once_grants.append(fingerprint(d))
        session.resume_decision = d
        session.pending_permission = None
        asyncio.run(
            agent.handle(
                session,
                "创建 a.txt 和 b.txt",
                continuation=True,
                record_user_message=False,
                resume_after_permission=True,
            )
        )
    assert session.status == "completed"
    assert session.action_epoch == 2
    assert not session.remaining_actions


def test_same_test_output_is_not_progress(tmp_path):
    from veripatch.studio_completion import CompletionCheck
    from veripatch.studio_domain import StudioObservation

    agent, session, _ = setup(tmp_path)
    check = CompletionCheck("needs_verification", ("test failed",))
    for duration in [1, 2]:
        session.observations.append(
            StudioObservation(
                kind="test",
                summary="failed",
                payload={
                    "command": ["pytest"],
                    "exit_code": 1,
                    "stderr": "failure",
                    "duration_seconds": duration,
                },
            )
        )
        terminal = agent._reject_completion(session, check)
    assert terminal and session.status == "paused"


def test_failed_test_is_not_cleared_by_unrelated_static_check(tmp_path):
    from veripatch.studio_completion import assess, unresolved_verification_failure
    from veripatch.studio_domain import StudioObservation

    _agent, session, _workspace = setup(tmp_path)
    session.turn_changed_files = ["app.py"]
    session.observations.extend(
        [
            StudioObservation(
                kind="test",
                summary="测试未通过",
                payload={"command": ["python", "-m", "pytest", "-q"], "exit_code": 1},
            ),
            StudioObservation(
                kind="command",
                summary="静态检查通过",
                payload={"command": ["python", "-m", "py_compile", "app.py"], "exit_code": 0},
            ),
        ]
    )

    assert unresolved_verification_failure(session)
    assert assess(session, []).state == "needs_verification"


def test_model_handoff_uses_controller_completion():
    from veripatch.studio_model import StudioProviderModel

    reply = StudioDecision(action="respond", rationale="done", message="已重命名文件。")
    assert not StudioProviderModel._is_unverified_execution_handoff(
        reply,
        {
            "changed_files": ["a.txt", "b.txt"],
            "task_contract": {"intent": "change"},
            "completion": {"requires_commands": False},
        },
    )


def test_document_patch_uses_same_content_verification(tmp_path):
    agent, session, workspace = setup(tmp_path)
    (tmp_path / "a.md").write_text("old\n", encoding="utf-8")
    agent._execute(
        session,
        workspace,
        StudioDecision(
            action="apply_patch",
            rationale="更新文档",
            patch="--- a/a.md\n+++ b/a.md\n@@ -1 +1 @@\n-old\n+new\n",
        ),
    )
    assert agent._has_file_operation_evidence(session)
    assert not agent._requires_command_verification(session)


def test_same_root_failure_escalates_then_stops_within_workspace_epoch(tmp_path):
    agent, session, _ = setup(tmp_path)
    session.task_state = StudioTaskState(intent="change")
    assessments = []
    strategies = []
    for duration in ("0.1s", "0.7s", "1.3s"):
        observation = StudioObservation(
            kind="test",
            summary="关联测试失败",
            payload={
                "command": ["python", "-m", "pytest", "-q"],
                "exit_code": 1,
                "stdout": f"FAILED test_app.py::test_add - AssertionError in {duration}",
                "failure_category": "test_failure",
                "retryable": True,
            },
        )
        assessment = agent._assess_observation(session, StudioAction.RUN_TESTS, observation)
        assessments.append(assessment)
        strategies.append(assessment.next_strategy)

    assert [item.outcome for item in assessments] == [
        ObservationOutcome.FAILED_RETRYABLE,
        ObservationOutcome.FAILED_RETRYABLE,
        ObservationOutcome.FAILED_TERMINAL,
    ]
    assert strategies[0] != strategies[1]
    assert assessments[-1].retryable is False
    assert session.task_state.replan_count == 2


def test_workspace_change_starts_new_recovery_cycle(tmp_path):
    agent, session, _ = setup(tmp_path)
    session.task_state = StudioTaskState(intent="change")

    def failure():
        observation = StudioObservation(
            kind="test",
            summary="关联测试失败",
            payload={
                "exit_code": 1,
                "stdout": "FAILED test_app.py::test_add - AssertionError",
                "failure_category": "test_failure",
                "retryable": True,
            },
        )
        assessment = agent._assess_observation(session, StudioAction.RUN_TESTS, observation)
        return observation, assessment

    first, _ = failure()
    session.action_epoch += 1
    second, assessment = failure()

    assert first.payload["recovery_attempt"] == 1
    assert second.payload["recovery_attempt"] == 1
    assert first.payload["failure_signature"] != second.payload["failure_signature"]
    assert assessment.outcome is ObservationOutcome.FAILED_RETRYABLE


def test_execute_pauses_after_third_same_root_tool_failure(tmp_path):
    agent, session, workspace = setup(tmp_path)
    decision = StudioDecision(action="read", path="missing.py", rationale="inspect")

    assert not agent._execute(session, workspace, decision)
    assert not agent._execute(session, workspace, decision)
    assert agent._execute(session, workspace, decision)

    assert session.status == "paused"
    assert session.observations[-1].payload["recovery_attempt"] == 3
    assert "missing_path" in session.messages[-1].content


def test_recovery_attempt_survives_checkpoint_reload(tmp_path):
    agent, session, _ = setup(tmp_path)
    session.task_state = StudioTaskState(intent="change")

    def assess(target):
        observation = StudioObservation(
            kind="test",
            summary="关联测试失败",
            payload={
                "exit_code": 1,
                "stdout": "FAILED test_app.py::test_add - AssertionError",
                "failure_category": "test_failure",
                "retryable": True,
            },
        )
        agent._assess_observation(target, StudioAction.RUN_TESTS, observation)
        return observation

    assert assess(session).payload["recovery_attempt"] == 1
    restored = agent.store.load(session.session_id)
    assert restored is not None
    assert assess(restored).payload["recovery_attempt"] == 2
