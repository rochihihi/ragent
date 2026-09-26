import pytest

from veripatch.studio_agent import StudioAgent
from veripatch.studio_domain import (
    StudioDecision, StudioObservation, StudioSession, StudioTaskContract,
)
from veripatch.studio_store import StudioStore
from veripatch.workspace import SafeWorkspace


def test_answer_review_does_not_replace_evidence_free_identity(tmp_path):
    session = StudioSession(
        session_id="identity", repo_root=str(tmp_path), provider="openai_official",
        model="gpt-6-luna", reasoning_effort="auto",
        task_contract=StudioTaskContract(
            objective="你是谁", intent="answer", evidence_required=False,
        ),
    )
    decision = StudioDecision(
        action="respond", rationale="从 agent_identity 回答",
        message="我是 RAgent，由 OpenAI 提供，当前模型是 gpt-6-luna。",
        claims=[{"kind": "fact", "text": "名称：RAgent；提供方：openai_official；模型：gpt-6-luna。"}],
    )
    agent = StudioAgent(None, StudioStore(tmp_path / "state.db"))

    assert agent._execute(session, SafeWorkspace(tmp_path), decision)
    assert session.messages[-1].content == decision.message
    review = next(event for event in agent.store.events(session.session_id)
                  if event["event_type"] == "claim_review")
    assert review["payload"]["claims"][0]["effective_kind"] == "unknown"


@pytest.mark.parametrize("action", ["respond", "finish"])
@pytest.mark.parametrize("via_mcp", [False, True])
@pytest.mark.parametrize("listing", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_operation_answers_are_rendered_from_evidence(tmp_path, action, via_mcp, listing, empty):
    result = ([] if empty else ["app.py"]) if listing else {
        "path": "app.py", "content": "" if empty else "1: return a + b",
    }
    session = StudioSession(
        session_id="operation", repo_root=str(tmp_path), provider="deepseek",
        model="test", reasoning_effort="low",
        observations=[StudioObservation(
            kind="mcp_tool" if via_mcp else "files" if listing else "read",
            summary="tool result",
            payload={
                "tool": "list_project_files" if listing else "read_project_file",
                "result": result,
            } if via_mcp else {"files": result} if listing else result,
        )],
    )
    decision = StudioDecision(
        action=action, rationale="answer", message="UNSUPPORTED PROSE",
        claims=[{"kind": "observation", "text": "invented purpose", "observation_id": 0}],
    )
    agent = StudioAgent(None, StudioStore(tmp_path / "state.db"))
    assert agent._execute(session, SafeWorkspace(tmp_path), decision)
    answer = session.messages[-1].content
    assert "尚不能确认" not in answer
    assert "invented purpose" not in answer
    assert "UNSUPPORTED PROSE" not in answer
    assert ("为空" if empty else "app.py") in answer


@pytest.mark.parametrize("payload", [{}, {"tool": "list_tools", "result": []}])
def test_operation_claim_requires_recognized_evidence(tmp_path, payload):
    session = StudioSession(
        session_id="invalid", repo_root=str(tmp_path), provider="deepseek",
        model="test", reasoning_effort="low",
        observations=[StudioObservation(kind="mcp_tool", summary="ok", payload=payload)],
    )
    decision = StudioDecision(
        action="respond", rationale="answer", message="answer",
        claims=[{"kind": "observation", "text": "app.py", "observation_id": 0}],
    )
    answer, audit = StudioAgent._render_claims(session, decision)
    assert answer == "目前没有足够证据回答这个问题。"
    assert audit[0]["source_matched"] is False


@pytest.mark.parametrize("action", ["respond", "finish"])
def test_equivalent_file_observations_render_once_without_losing_audit(tmp_path, action):
    session = StudioSession(
        session_id="duplicate-list",
        repo_root=str(tmp_path),
        provider="deepseek",
        model="test",
        reasoning_effort="low",
        observations=[
            StudioObservation(kind="files", summary="built-in", payload={"files": ["app.py"]}),
            StudioObservation(
                kind="mcp_tool",
                summary="mcp",
                payload={"tool": "list_project_files", "result": ["app.py"]},
            ),
            StudioObservation(
                kind="files", summary="updated", payload={"files": ["app.py", "new.py"]}
            ),
        ],
    )
    decision = StudioDecision(
        action=action,
        rationale="report files",
        message="ignored",
        claims=[
            {"kind": "observation", "text": "files", "observation_id": index}
            for index in (0, 0, 1, 2)
        ],
    )
    agent = StudioAgent(None, StudioStore(tmp_path / "state.db"))

    assert agent._execute(session, SafeWorkspace(tmp_path), decision)
    answer = session.messages[-1].content
    assert answer.count("文件列表：") == 1
    assert answer.count("app.py") == 1
    assert "new.py" in answer
    reviews = [
        event
        for event in agent.store.events(session.session_id)
        if event["event_type"] == "claim_review"
    ]
    assert len(reviews[0]["payload"]["claims"]) == 4


@pytest.mark.parametrize("via_mcp", [False, True])
def test_file_listing_claims_audit_without_rewriting_the_final_answer(tmp_path, via_mcp):
    files = ["app.py", "calc.py", "note.txt"]
    session = StudioSession(
        session_id="listed-files", repo_root=str(tmp_path), provider="deepseek",
        model="test", reasoning_effort="low",
        observations=[StudioObservation(
            kind="mcp_tool" if via_mcp else "files", summary="found files",
            payload={"tool": "list_project_files", "result": files}
            if via_mcp else {"files": files},
        )],
    )
    claims = [{"kind": "observation", "text": "found files", "observation_id": 0}]
    claims.extend(
        {"kind": "fact", "text": path, "observation_id": 0} for path in files
    )
    decision = StudioDecision(
        action="respond", rationale="show files",
        message="项目文件有 app.py、calc.py 和 note.txt。", claims=claims,
    )
    agent = StudioAgent(None, StudioStore(tmp_path / "state.db"))

    assert agent._execute(session, SafeWorkspace(tmp_path), decision)
    assert session.messages[-1].content == decision.message
    review = next(
        event for event in agent.store.events(session.session_id)
        if event["event_type"] == "claim_review"
    )
    assert len(review["payload"]["claims"]) == 4


def test_file_listing_fallback_does_not_repeat_items_as_facts(tmp_path):
    files = ["app.py", "calc.py"]
    session = StudioSession(
        session_id="listed-files-fallback", repo_root=str(tmp_path),
        provider="deepseek", model="test", reasoning_effort="low",
        observations=[StudioObservation(kind="files", summary="found files", payload={"files": files})],
    )
    decision = StudioDecision(
        action="respond", rationale="show files", message="UNSUPPORTED PROSE",
        claims=[
            {"kind": "observation", "text": "found files", "observation_id": 0},
            {"kind": "fact", "text": "app.py", "observation_id": 0},
            {"kind": "fact", "text": "calc.py", "observation_id": 0},
        ],
    )

    answer, audit = StudioAgent._render_claims(session, decision)
    assert answer == "文件列表：\napp.py\ncalc.py"
    assert len(audit) == 3


def test_current_mcp_listing_replaces_historical_builtin_listing(tmp_path):
    session = StudioSession(
        session_id="mcp-current-list",
        repo_root=str(tmp_path),
        provider="deepseek",
        model="test",
        reasoning_effort="low",
        turn_observation_start=1,
        observations=[
            StudioObservation(
                kind="files", summary="old built-in result",
                payload={"files": ["app.py", "__pycache__/app.pyc"]},
            ),
            StudioObservation(
                kind="mcp_tool", summary="current MCP result",
                payload={"tool": "list_project_files", "result": ["app.py"]},
            ),
        ],
    )
    decision = StudioDecision(
        action="respond", rationale="report files", message="ignored",
        claims=[
            {"kind": "observation", "text": "files", "observation_id": index}
            for index in (1, 0)
        ],
    )
    answer, audit = StudioAgent._render_claims(session, decision)
    assert answer == "文件列表：\napp.py"
    assert len(audit) == 2
    assert all(item["source_matched"] for item in audit)


def test_mcp_tool_description_stays_in_audit_not_file_answer(tmp_path):
    description = "List UTF-8 project files while skipping generated directories."
    session = StudioSession(
        session_id="mcp-metadata",
        repo_root=str(tmp_path), provider="deepseek", model="test", reasoning_effort="low",
        observations=[
            StudioObservation(
                kind="mcp_tool", summary="list tools",
                payload={"tool": "list_tools", "result": [{"description": description}]},
            ),
            StudioObservation(
                kind="mcp_tool", summary="list files",
                payload={"tool": "list_project_files", "result": ["app.py"]},
            ),
        ],
    )
    decision = StudioDecision(
        action="respond", rationale="show files", message="ignored",
        claims=[
            {"kind": "observation", "text": "files", "observation_id": 1},
            {"kind": "unknown", "text": f"该工具描述为“{description}”", "observation_id": 0},
        ],
    )
    answer, audit = StudioAgent._render_claims(session, decision)
    assert answer == "文件列表：\napp.py"
    assert len(audit) == 2
    assert audit[1]["text"].endswith(description + "”")


@pytest.mark.parametrize("via_mcp", [False, True])
@pytest.mark.parametrize("listing", [False, True])
def test_file_evidence_is_transport_independent(tmp_path, via_mcp, listing):
    result = ["app.py"] if listing else {"path": "app.py", "content": "1: return a + b"}
    observation = StudioObservation(
        kind="mcp_tool" if via_mcp else "files" if listing else "read",
        summary="tool result",
        payload={
            "tool": "list_project_files" if listing else "read_project_file",
            "result": result,
        } if via_mcp else {"files": result} if listing else result,
    )
    session = StudioSession(
        session_id="transport", repo_root=str(tmp_path), provider="deepseek",
        model="test", reasoning_effort="low", observations=[observation],
    )
    for text, supported in [
        ("app.py" if listing else "return a + b", True),
        ("app" if listing else "用于上下文压缩", False),
    ]:
        decision = StudioDecision(
            action="respond", rationale="answer", message="answer",
            claims=[{"kind": "fact", "text": text, "observation_id": 0}],
        )
        answer, audit = StudioAgent._render_claims(session, decision)
        assert audit[0]["source_matched"] is supported
        assert ("目前没有足够证据" not in answer) is supported


@pytest.mark.parametrize("action", ["respond", "finish"])
@pytest.mark.parametrize(
    "kind,source,text,label",
    [
        ("fact", 0, "probe", "中的原文"),
        ("fact", 0, "用于上下文压缩", "目前没有足够证据"),
        ("fact", 99, "probe", "目前没有足够证据"),
        ("fact", 1, "用于上下文压缩", "目前没有足够证据"),
        ("inference", 0, "可能用于压缩测试", "推测"),
        ("unknown", None, "文件的真实用途", "目前没有足够证据"),
    ],
)
def test_answer_sources_on_both_response_routes(tmp_path, action, kind, source, text, label):
    session = StudioSession(
        session_id="claims",
        repo_root=str(tmp_path),
        provider="deepseek",
        model="test",
        reasoning_effort="low",
        observations=[
            StudioObservation(
                kind="read", summary="read", payload={"path": "probe.txt", "content": "1: probe"}
            ),
            StudioObservation(
                kind="result_review",
                summary="用于上下文压缩",
                payload={"content": "用于上下文压缩"},
            ),
        ],
    )
    store = StudioStore(tmp_path / "state.db")
    agent = StudioAgent(None, store)
    decision = StudioDecision(
        action=action,
        rationale="answer",
        message="UNSUPPORTED PROSE",
        claims=[{"kind": kind, "observation_id": source, "text": text}],
    )
    assert agent._execute(session, SafeWorkspace(tmp_path), decision)
    answer = session.messages[-1].content
    assert label in answer
    assert "UNSUPPORTED PROSE" not in answer
    assert any(e["event_type"] == "claim_review" for e in store.events(session.session_id))
