import asyncio

import pytest

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


def test_optional_null_question_preserves_semantic_instruction():
    assessment = SemanticIntentAssessment.model_validate(
        {
            "intent": "change",
            "confidence": "high",
            "requires_clarification": False,
            "clarification_question": None,
            "rationale": "明确要求重命名",
            "requested_actions": ["move_file"],
        }
    )
    policy = semantic_policy(classify_intent("把 a.txt 重命名为 b.txt。"), assessment)
    assert assessment.clarification_question == ""
    assert StudioAction.MOVE_FILE in policy.allowed_actions


@pytest.mark.parametrize(
    "message,action",
    [
        ("把 a.txt 重命名为 b.txt。", "move_file"),
        ("给它留个副本叫 c.txt。", "copy_file"),
        ("将它挪到 docs/a.txt。", "move_file"),
    ],
)
def test_semantic_instruction_does_not_require_keyword_match(message, action):
    assessment = SemanticIntentAssessment(
        intent="change",
        confidence="high",
        requires_clarification=False,
        rationale="明确的文件操作",
        requested_actions=[action],
    )
    policy = semantic_policy(classify_intent(message), assessment)
    assert policy.mutation_requested
    assert StudioAction(action) in policy.allowed_actions
    assert not assessment.requires_clarification


def test_semantic_prohibition_wins_over_requested_action():
    assessment = SemanticIntentAssessment(
        intent="change",
        confidence="high",
        requires_clarification=False,
        rationale="只重命名",
        requested_actions=["move_file"],
        prohibited_actions=["run_command", "run_tests"],
    )
    p = semantic_policy(classify_intent("重命名，不运行任何命令"), assessment)
    assert StudioAction.MOVE_FILE in p.allowed_actions
    assert StudioAction.RUN_COMMAND not in p.allowed_actions


def test_discussion_does_not_inherit_write_capability():
    assessment = SemanticIntentAssessment(
        intent="answer",
        confidence="high",
        requires_clarification=False,
        rationale="询问之前结果",
        questions=["改了什么"],
    )
    p = semantic_policy(classify_intent("你重命名了吗？"), assessment)
    assert not p.mutation_requested
    assert StudioAction.MOVE_FILE not in p.allowed_actions


def test_explicit_file_action_does_not_depend_on_semantic_classifier(tmp_path):
    class UnavailableClassifierModel:
        def __init__(self):
            self.decisions = [
                StudioDecision(
                    action=StudioAction.MOVE_FILE,
                    path="a.txt",
                    destination="b.txt",
                    rationale="执行明确文件操作",
                ),
                StudioDecision(
                    action=StudioAction.FINISH,
                    rationale="文件操作已核对",
                    message="已完成文件操作，未运行命令。",
                ),
            ]

        async def classify_intent(self, messages, message):
            pytest.fail("明确独立任务不应发起额外语义分类")

        async def decide(self, context):
            return StudioReply(
                decision=self.decisions.pop(0), input_tokens=1, output_tokens=1
            )

    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    session = StudioSession(
        session_id="semantic-failure",
        repo_root=str(tmp_path),
        provider="deepseek",
        model="deepseek-v4-flash",
        reasoning_effort="high",
    )
    agent = StudioAgent(UnavailableClassifierModel(), StudioStore(tmp_path / "state.sqlite3"))
    asyncio.run(agent.handle(session, "把 a.txt 重命名为 b.txt。"))
    assert session.status == "completed"
    assert not (tmp_path / "a.txt").exists()
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "hello"


@pytest.mark.parametrize("action,destination", [("move_file", "b.txt"), ("copy_file", "c.txt")])
def test_semantic_file_instruction_reaches_executor_and_finishes(tmp_path, action, destination):
    class Model:
        def __init__(self):
            self.decisions = [
                StudioDecision(
                    action=StudioAction(action),
                    path="a.txt",
                    destination=destination,
                    rationale="执行明确文件操作",
                ),
                StudioDecision(
                    action=StudioAction.FINISH,
                    rationale="文件操作已核对",
                    message="已完成文件操作，未运行命令。",
                ),
            ]

        async def classify_intent(self, messages, message):
            return SemanticIntentAssessment.model_validate(
                {
                    "intent": "change",
                    "confidence": "high",
                    "requires_clarification": False,
                    "clarification_question": None,
                    "rationale": "明确文件操作",
                    "requested_actions": [action],
                    "prohibited_actions": ["run_command", "run_tests"],
                }
            )

        async def decide(self, context):
            return StudioReply(decision=self.decisions.pop(0), input_tokens=1, output_tokens=1)

    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    session = StudioSession(
        session_id="semantic-execute",
        repo_root=str(tmp_path),
        provider="deepseek",
        model="test",
        reasoning_effort="low",
    )
    message = (
        f"把 a.txt 重命名为 {destination}。"
        if action == "move_file"
        else f"给 a.txt 留个副本叫 {destination}。"
    ) + "不要运行命令。"
    asyncio.run(
        StudioAgent(Model(), StudioStore(tmp_path / "state.sqlite3")).handle(session, message)
    )
    assert session.status == "completed"
    assert (tmp_path / destination).read_text() == "hello"
    assert (tmp_path / "a.txt").exists() == (action == "copy_file")
    assert not any(o.kind in {"command", "test", "capability_guard"} for o in session.observations)
