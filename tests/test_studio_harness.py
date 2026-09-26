from veripatch.studio_domain import StudioAction, StudioDecision, StudioObservation, StudioSession
from veripatch.studio_harness import ActionLedger, TaskEvidence, ToolOutcome, ToolRegistry, ToolRouter


def test_registry_describes_workspace_and_mcp_surfaces() -> None:
    assert ToolRegistry.get("mcp_call").surface == "mcp"
    assert ToolRegistry.get("read").read_only is True


def test_router_preserves_model_choice_without_preference() -> None:
    decision = StudioDecision(action=StudioAction.LIST_FILES, rationale="查看项目")
    assert ToolRouter.route(decision, "查看项目") is decision


def test_router_adapts_equivalent_read_when_mcp_is_requested() -> None:
    decision = StudioDecision(action=StudioAction.LIST_FILES, rationale="查看项目")
    routed = ToolRouter.route(decision, "请用 MCP 查看项目")
    assert routed.action is StudioAction.MCP_CALL
    assert routed.mcp_tool == "list_project_files"


def test_explicit_tool_request_requires_fresh_data_not_mcp_discovery(tmp_path) -> None:
    requirements = TaskEvidence.requested_tools("请用 MCP 查看项目文件，再用 list_files 核对")
    assert [item.expected for item in requirements] == ["mcp_call", "list_files"]
    session = StudioSession(
        session_id="tool-contract", repo_root=str(tmp_path), provider="openai",
        model="test", reasoning_effort="low",
        observations=[StudioObservation(
            kind="mcp_tool", summary="discovery",
            payload={"tool": "list_tools", "arguments": {}, "result": {}},
        )],
    )
    from veripatch.studio_domain import StudioTaskContract
    session.task_contract = StudioTaskContract(objective="inspect", intent="analysis", requirements=requirements)
    assert TaskEvidence.pending_tools(session) == ["mcp_call", "list_files"]
    session.observations.append(StudioObservation(
        kind="mcp_tool", summary="files",
        payload={"tool": "list_project_files", "arguments": {}, "result": {"files": ["a.py"]}},
    ))
    assert TaskEvidence.pending_tools(session) == ["list_files"]
    assert TaskEvidence.required_surface(session) is None
    assert ToolRouter.route(
        StudioDecision(action=StudioAction.LIST_FILES, rationale="核对"),
        "", required_surface=TaskEvidence.required_surface(session),
    ).action is StudioAction.LIST_FILES
    session.turn_observation_start = len(session.observations)
    assert TaskEvidence.pending_tools(session) == ["mcp_call", "list_files"]


def test_tool_mention_without_request_does_not_create_obligation() -> None:
    assert TaskEvidence.requested_tools("MCP 是什么？") == []
    assert ToolRouter.requested_surface("MCP 是什么？") is None
    assert ToolRouter.requested_surface("为什么用 MCP 查看项目？") is None
    assert ToolRouter.requested_surface("用mcp看项目") == "mcp"
    routed = ToolRouter.route(
        StudioDecision(action=StudioAction.LIST_FILES, rationale="继续查看"),
        "继续", required_surface="mcp",
    )
    assert routed.action is StudioAction.MCP_CALL


def test_ledger_matches_success_across_workspace_and_mcp_tools(tmp_path) -> None:
    session = StudioSession(
        session_id="outcomes", repo_root=str(tmp_path), provider="openai", model="test", reasoning_effort="low",
        observations=[
            StudioObservation(kind="files", summary="list", payload={"files": ["a.py"]}),
            StudioObservation(kind="read", summary="read", payload={"path": "a.py", "content": "a"}),
            StudioObservation(kind="mcp_tool", summary="mcp", payload={
                "tool": "search_project", "arguments": {"query": "abc"}, "result": {"matches": []},
            }),
            StudioObservation(kind="command", summary="command", payload={
                "command": ["python", "-V"], "exit_code": 0,
            }),
        ],
    )
    decisions = [
        StudioDecision(action=StudioAction.LIST_FILES, rationale="again"),
        StudioDecision(action=StudioAction.READ, rationale="again", path="a.py"),
        StudioDecision(action=StudioAction.MCP_CALL, rationale="again", mcp_tool="search_project", mcp_arguments={"query": "abc"}),
        StudioDecision(action=StudioAction.RUN_COMMAND, rationale="again", command=["python", "-V"]),
    ]
    assert [ActionLedger.successful_observation(session, item)[0] for item in decisions] == [0, 1, 2, 3]
    assert ActionLedger.successful_observation(
        session, StudioDecision(action=StudioAction.READ, rationale="other", path="b.py")
    ) is None
    assert ActionLedger.successful_observation(
        session, StudioDecision(action=StudioAction.MCP_CALL, rationale="other", mcp_tool="search_project", mcp_arguments={"query": "new"})
    ) is None


def test_ledger_never_reuses_previous_turn_or_pre_edit_evidence(tmp_path) -> None:
    read = StudioDecision(action=StudioAction.READ, rationale="read", path="a.py")
    session = StudioSession(
        session_id="freshness", repo_root=str(tmp_path), provider="openai", model="test", reasoning_effort="low",
        observations=[StudioObservation(kind="read", summary="old", payload={"path": "a.py"})],
        turn_observation_start=1,
    )
    assert ActionLedger.successful_observation(session, read) is None
    session.turn_observation_start = 0
    session.observations.append(StudioObservation(kind="edit", summary="changed", payload={"path": "a.py"}))
    assert ActionLedger.successful_observation(session, read) is None
    session.observations.append(StudioObservation(kind="read", summary="new", payload={"path": "a.py"}))
    assert ActionLedger.successful_observation(session, read)[0] == 2
    assert [item["observation_id"] for item in ActionLedger.current_outcomes(session)] == [0, 1, 2]
    session.turn_observation_start = 2
    assert [item["observation_id"] for item in ActionLedger.current_outcomes(session)] == [2]


def test_ledger_requires_success_and_ignores_model_wording(tmp_path) -> None:
    session = StudioSession(
        session_id="status", repo_root=str(tmp_path), provider="openai", model="test", reasoning_effort="low",
        observations=[StudioObservation(kind="command", summary="failed", payload={
            "command": ["python", "-V"], "exit_code": 1,
        })],
    )
    command = StudioDecision(action=StudioAction.RUN_COMMAND, rationale="one", command=["python", "-V"])
    assert ActionLedger.successful_observation(session, command) is None
    assert ActionLedger.fingerprint(command, 0) == ActionLedger.fingerprint(
        command.model_copy(update={"rationale": "different wording"}), 0
    )
    assert ActionLedger.fingerprint(command, 0) != ActionLedger.fingerprint(command, 1)
    assert ActionLedger.refreshable(StudioAction.POLL_TERMINAL)
    assert ActionLedger.refreshable(StudioAction.INSPECT_PROCESSES)
    assert not ActionLedger.refreshable(StudioAction.START_TERMINAL)


def test_tool_outcome_normalizes_command_failure_and_changed_files() -> None:
    observation = StudioObservation(
        kind="command", summary="command failed",
        payload={"command": ["python", "x.py"], "exit_code": 1, "changed_files": ["x.py"]},
    )
    result = ToolOutcome.normalize(StudioAction.RUN_COMMAND, observation)
    assert result.status == "failed"
    assert result.changed_files == ["x.py"]
    observation.tool_result = result
    assert not ToolOutcome.succeeded(observation)


def test_ledger_invalidates_old_read_after_git_restore(tmp_path) -> None:
    read = StudioDecision(action=StudioAction.READ, rationale="read", path="a.py")
    session = StudioSession(
        session_id="restore", repo_root=str(tmp_path), provider="openai",
        model="test", reasoning_effort="low",
        observations=[
            StudioObservation(kind="read", summary="old", payload={"path": "a.py"}),
            StudioObservation(kind="git_restore", summary="restored", payload={"path": "a.py"}),
        ],
    )
    assert ActionLedger.successful_observation(session, read) is None
