import asyncio

import pytest

from veripatch.studio_agent import StudioAgent
from veripatch.studio_domain import (
    StudioDecision, StudioMessage, StudioObservation, StudioReply, StudioSession,
    StudioTaskContract,
)
from veripatch.studio_store import StudioStore
from veripatch.workspace import SafeWorkspace


def session_for(tmp_path):
    return StudioSession(session_id="original-request", repo_root=str(tmp_path),
                         provider="deepseek", model="test", reasoning_effort="low")


def test_new_artifact_request_is_not_rewritten_by_preflight_or_old_contract(tmp_path):
    original = "新建一个计算器，然后打开"
    old = tmp_path / "calculator_project" / "app.js"
    old.parent.mkdir()
    old.write_text("original calculator", encoding="utf-8")
    session = session_for(tmp_path)
    session.task_contract = StudioTaskContract(
        objective="在现有计算器项目中实现一个计算器", intent="change",
        objectives=["修改旧计算器"], conditions=["先修改旧文件"],
    )
    session.messages = [StudioMessage(role="user", content="之前创建了一个计算器")]

    class Model:
        classifier_calls = 0

        async def classify_intent(self, messages, message):
            self.classifier_calls += 1
            raise AssertionError("A preflight classifier must not rewrite the request")

        async def decide(self, context):
            assert context["current_request"] == original
            assert context["authority"]["objective"] == original
            assert context["task_contract"]["objective"] == original
            assert context["task_contract"]["conditions"] == []
            assert "advisory" in context["authority"]["rule"]
            return StudioReply(decision=StudioDecision(
                action="respond", rationale="确认独立新产物的要求",
                message="我会另建一个独立计算器，保留已有文件。",
            ))

    model = Model()
    asyncio.run(StudioAgent(model, StudioStore(tmp_path / "state.db")).handle(session, original))
    assert model.classifier_calls == 0
    assert old.read_text(encoding="utf-8") == "original calculator"


@pytest.mark.parametrize("action", ["respond", "finish"])
def test_explanation_of_prior_patch_survives_unmatched_claim_audit(tmp_path, action):
    session = session_for(tmp_path)
    session.task_contract = StudioTaskContract(
        objective="我说的新建一个计算器，你怎么把原来的改了", intent="answer",
        evidence_required=True,
    )
    session.observations = [StudioObservation(kind="patch", summary="修改旧文件",
        payload={"paths": ["calculator_project/app.js"]})]
    session.turn_observation_start = 1
    message = "我误把新建理解成修改旧项目，改了 calculator_project/app.js。这不符合你的要求。"
    decision = StudioDecision(action=action, rationale="承认误解", message=message,
        claims=[{"kind": "observation", "text": "修改了旧文件", "observation_id": 0}])
    store = StudioStore(tmp_path / "state.db")
    assert StudioAgent(None, store)._execute(session, SafeWorkspace(tmp_path), decision)
    assert session.messages[-1].content == message
    audit = next(e for e in store.events(session.session_id) if e["event_type"] == "claim_review")
    assert audit["payload"]["claims"][0]["source_matched"] is False
    assert audit["payload"]["model_message"] == message


def test_default_final_review_records_disagreement_without_rewriting(tmp_path):
    session = session_for(tmp_path)
    message = "测试通过。"
    store = StudioStore(tmp_path / "state.db")
    decision = StudioDecision(action="finish", rationale="完成", message=message)
    assert StudioAgent(None, store)._execute(session, SafeWorkspace(tmp_path), decision)
    assert session.messages[-1].content == message
    review = next(e for e in store.events(session.session_id) if e["event_type"] == "result_review")
    assert review["payload"]["verification_claim_grounded"] is False


def test_create_collision_returns_to_model_without_overwriting(tmp_path):
    target = tmp_path / "calculator.html"
    target.write_text("old calculator", encoding="utf-8")
    session = session_for(tmp_path)
    decisions = [
        StudioDecision(action="create", rationale="新建", path="calculator.html", content="new calculator"),
        StudioDecision(action="create", rationale="保留旧文件，另选新路径",
                       path="new-calculator.html", content="new calculator"),
        StudioDecision(action="finish", rationale="完成独立新产物", message="已新建 new-calculator.html，保留旧文件。"),
    ]

    class Model:
        async def decide(self, context):
            if len(decisions) == 2:
                assert "already exists" in str(context["recent_observations"])
            return StudioReply(decision=decisions.pop(0))

    asyncio.run(StudioAgent(Model(), StudioStore(tmp_path / "state.db")).handle(session, "新建一个计算器"))
    assert session.status == "completed"
    assert target.read_text(encoding="utf-8") == "old calculator"
    assert (tmp_path / "new-calculator.html").read_text(encoding="utf-8") == "new calculator"
