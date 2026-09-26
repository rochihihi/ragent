import asyncio

import pytest
from pydantic import ValidationError

from veripatch.studio_agent import StudioAgent
from veripatch.studio_domain import (
    SemanticIntentAssessment,
    StudioAction,
    StudioDecision,
    StudioReply,
    StudioSession,
)
from veripatch.studio_intent import classify_intent, semantic_policy
from veripatch.studio_store import StudioStore


@pytest.mark.parametrize("extra", [{}, {"rationale": None}, {"rationale": ""}])
def test_optional_explanation_does_not_destroy_decision(extra):
    result = SemanticIntentAssessment.model_validate(
        dict(
            intent="answer",
            confidence="high",
            requires_clarification=False,
            requested_actions=["respond"],
            **extra,
        )
    )
    assert result.intent == "answer"


def test_required_decision_fields_remain_strict():
    with pytest.raises(ValidationError):
        SemanticIntentAssessment.model_validate({"rationale": "hello"})


def test_response_only_execute_label_never_grants_commands():
    assessment = SemanticIntentAssessment(
        intent="execute",
        confidence="high",
        requires_clarification=False,
        requested_actions=["respond"],
    )
    policy = semantic_policy(classify_intent("开始吧"), assessment)
    assert policy.intent == "answer"
    assert not assessment.requires_clarification
    assert StudioAction.RUN_COMMAND not in policy.allowed_actions
    assert StudioAction.EDIT not in policy.allowed_actions


def test_interview_question_survives_storage_and_next_turn(tmp_path):
    question = "面试练习开始。\n\n第一题：Python 的 list 和 tuple 有什么区别？"

    class Model:
        calls = 0

        async def classify_intent(self, messages, message):
            if self.calls:
                assert any(item["content"] == question for item in messages)
            return SemanticIntentAssessment(
                intent="execute",
                confidence="high",
                requires_clarification=False,
                requested_actions=["respond"],
            )

        async def decide(self, context):
            self.calls += 1
            return StudioReply(
                decision=StudioDecision(
                    action=StudioAction.RESPOND,
                    rationale="纯对话",
                    message=question if self.calls == 1 else "list 可变，tuple 不可变。",
                )
            )

    session = StudioSession(
        session_id="interview",
        repo_root=str(tmp_path),
        provider="deepseek",
        model="test",
        reasoning_effort="low",
    )
    store = StudioStore(tmp_path / "state.db")
    agent = StudioAgent(Model(), store)
    asyncio.run(agent.handle(session, "陪我模拟面试"))
    assert store.load(session.session_id).messages[-1].content == question
    asyncio.run(agent.handle(session, "这题我不会"))
    assert "可变" in session.messages[-1].content
