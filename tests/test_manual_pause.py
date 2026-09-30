import asyncio
import time
from threading import Event

from fastapi.testclient import TestClient

from veripatch.api import create_app
from veripatch.config import Settings
from veripatch.studio_agent import StudioAgent
from veripatch.studio_domain import (
    SemanticIntentAssessment,
    StudioDecision,
    StudioReply,
    StudioSession,
)
from veripatch.studio_store import StudioStore


def state(tmp_path):
    return StudioSession(
        session_id="pause-test",
        repo_root=str(tmp_path),
        provider="openai",
        model="test",
        reasoning_effort="low",
    )


def test_default_loop_can_run_more_than_sixty_model_steps(tmp_path):
    (tmp_path / "a.txt").write_text("evidence", encoding="utf-8")

    class Model:
        calls = 0

        async def decide(self, context):
            self.calls += 1
            assert context["steps_remaining_this_turn"] is None
            return StudioReply(
                decision=StudioDecision(
                    action="read" if self.calls <= 65 else "respond",
                    rationale="inspect",
                    path="a.txt" if self.calls <= 65 else None,
                    message="已读取。",
                )
            )

    model = Model()
    session = state(tmp_path)
    asyncio.run(
        StudioAgent(model, StudioStore(tmp_path / "state.db"), max_context_tokens=1_000_000).handle(
            session, "了解这个项目"
        )
    )
    assert model.calls == 66
    assert session.status == "idle"
    assert session.turn_budget is None


def test_pause_cancels_waiting_model_without_executing_action(tmp_path):
    signal = Event()

    class Model:
        cancelled = False

        async def decide(self, context):
            signal.set()
            try:
                await asyncio.sleep(100)
            finally:
                self.cancelled = True

    model = Model()
    session = state(tmp_path)
    asyncio.run(
        StudioAgent(
            model, StudioStore(tmp_path / "state.db"), pause_requested=signal.is_set
        ).handle(session, "处理项目")
    )
    assert session.status == "paused"
    assert model.cancelled
    assert not session.observations


def test_semantic_understanding_is_not_overridden_by_lexical_scope(tmp_path):
    class Model:
        async def classify_intent(self, messages, message):
            return SemanticIntentAssessment(
                intent="change",
                confidence="high",
                requires_clarification=False,
                requested_actions=["create"],
                prohibited_actions=["run_command", "run_tests"],
                objectives=["创建说明文件"],
            )

        async def decide(self, context):
            assert context["task_contract"]["scope_actions"] == []
            return StudioReply(
                decision=StudioDecision(
                    action="create", rationale="implement", path="note.txt", content="done"
                )
            )

    session = state(tmp_path)
    asyncio.run(
        StudioAgent(Model(), StudioStore(tmp_path / "state.db"), max_steps=1).handle(
            session, "就照上面的第二种方案做"
        )
    )
    assert (tmp_path / "note.txt").read_text(encoding="utf-8") == "done"
    assert session.pending_permission is None
    assert session.task_contract.intent_source == "model_pending"
    assert session.task_contract.objective == "就照上面的第二种方案做"
    assert session.task_contract.denied_actions == []


def test_pause_preserves_remaining_batch_without_repeating_committed_write(tmp_path, monkeypatch):
    signal = Event()
    writes = []

    class Model:
        calls = 0

        async def decide(self, context):
            self.calls += 1
            if self.calls == 1:
                return StudioReply(
                    decision=StudioDecision(
                        action="batch",
                        rationale="create notes",
                        actions=[
                            StudioDecision(
                                action="create", rationale="create", path=path, content="note"
                            )
                            for path in ["a.txt", "b.txt"]
                        ],
                    )
                )
            return StudioReply(
                decision=StudioDecision(
                    action="finish", rationale="done", message="已创建说明文件。"
                )
            )

    original = StudioAgent._execute

    def execute(self, session, workspace, decision, **kwargs):
        result = original(self, session, workspace, decision, **kwargs)
        if decision.action.value == "create":
            writes.append(decision.path)
            signal.set()
        return result

    monkeypatch.setattr(StudioAgent, "_execute", execute)
    session = state(tmp_path)
    model = Model()
    agent = StudioAgent(model, StudioStore(tmp_path / "state.db"), pause_requested=signal.is_set)
    asyncio.run(agent.handle(session, "按方案创建说明文件"))
    assert session.status == "paused"
    assert writes == ["a.txt"]
    assert [a.path for a in session.remaining_actions] == ["b.txt"]
    signal.clear()
    agent.pause_requested = lambda: False
    asyncio.run(
        agent.handle(session, "按方案创建说明文件", continuation=True, record_user_message=False)
    )
    assert session.status == "completed"
    assert writes == ["a.txt", "b.txt"]
    assert session.remaining_actions == []


def test_api_pause_during_tool_preserves_result_and_resume_does_not_repeat(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("evidence", encoding="utf-8")
    entered, release = Event(), Event()
    calls = []

    class Model:
        def __init__(self, *args):
            pass

        async def decide(self, context):
            calls.append(context)
            return StudioReply(
                decision=StudioDecision(
                    action="read" if len(calls) == 1 else "respond",
                    rationale="inspect",
                    path="a.txt" if len(calls) == 1 else None,
                    message="已读取。",
                )
            )

    original = StudioAgent._execute_action

    def slow_tool(self, session, workspace, decision, **kwargs):
        if decision.action.value == "read":
            entered.set()
            assert release.wait(5)
        return original(self, session, workspace, decision, **kwargs)

    monkeypatch.setattr("veripatch.studio_api.StudioProviderModel", Model)
    monkeypatch.setattr(StudioAgent, "_execute_action", slow_tool)
    database = tmp_path / "state.db"
    session = state(tmp_path)
    store = StudioStore(database)
    store.save(session, "created", {})

    def wait_status(client, expected):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            result = client.get("/studio-api/sessions/pause-test").json()
            if result["status"] == expected:
                return result
            time.sleep(0.01)
        raise AssertionError(result)

    with TestClient(create_app(settings=Settings(database_path=database))) as client:
        assert (
            client.post(
                "/studio-api/sessions/pause-test/messages", json={"content": "了解项目"}
            ).status_code
            == 202
        )
        assert entered.wait(5)
        try:
            assert (
                client.post("/studio-api/sessions/pause-test/pause").json()["status"] == "pausing"
            )
            assert client.post("/studio-api/sessions/pause-test/resume").status_code == 409
        finally:
            release.set()
        paused = wait_status(client, "paused")
        assert len([o for o in paused["observations"] if o["kind"] == "read"]) == 1
        assert len(calls) == 1
        assert client.post("/studio-api/sessions/pause-test/resume").status_code == 202
        resumed = wait_status(client, "idle")
        assert len([o for o in resumed["observations"] if o["kind"] == "read"]) == 1
        assert len([m for m in resumed["messages"] if m["role"] == "user"]) == 1
