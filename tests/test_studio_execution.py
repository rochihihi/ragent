from veripatch.studio_execution import classify_command, launch_effect_satisfied


def test_changed_python_program_is_a_launch_when_task_requires_open(tmp_path):
    execution = classify_command(
        ["python", "calculator.py"], root=tmp_path,
        changed_files=["calculator.py"], launch_required=True,
    )
    assert execution.role == "launch"
    assert execution.target == "calculator.py"
    assert execution.command == ["cmd", "/c", "start", "", "python", "calculator.py"]


def test_verification_and_launch_have_distinct_roles(tmp_path):
    verify = classify_command(
        ["python", "-m", "py_compile", "calculator.py"], root=tmp_path,
        changed_files=["calculator.py"], launch_required=True,
    )
    launch = classify_command(
        ["cmd", "/c", "start", "", "python", "calculator.py"], root=tmp_path,
        changed_files=["calculator.py"], launch_required=True,
    )
    assert verify.role == "verification"
    assert launch.role == "launch"
    assert verify.command != launch.command


def test_python_program_without_launch_goal_keeps_existing_command_behavior(tmp_path):
    execution = classify_command(
        ["python", "probe.py"], root=tmp_path,
        changed_files=["probe.py"], launch_required=False,
    )
    assert execution.role == "verification"
    assert execution.command == ["python", "probe.py"]


def test_document_dispatch_is_distinct_from_window_confirmation():
    assert launch_effect_satisfied({"launch_state": "dispatched", "exit_code": 0})
    assert not launch_effect_satisfied({"launch_state": "dispatched", "exit_code": 1})
    assert not launch_effect_satisfied({"launch_state": "exited_unconfirmed", "exit_code": 1})
