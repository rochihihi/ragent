import asyncio
from unittest.mock import patch

import pytest

from veripatch.studio_agent import StudioAgent
from veripatch.studio_domain import (
    SemanticIntentAssessment,
    StudioAction,
    StudioDecision,
    StudioMessage,
    StudioObservation,
    StudioReply,
    StudioSession,
    StudioTaskContract,
    VerificationMode,
)
from veripatch.studio_skills import install, selected
from veripatch.studio_store import StudioStore


@pytest.mark.parametrize(
    "question",
    [
        "刚才做了什么？有没有运行命令？",
        "刚才改了哪些文件，测试运行了吗？",
        "What changed and did you run any commands?",
    ],
)
def test_history_question_reaches_model_with_operation_evidence(tmp_path, question):
    answer = "已将 a.txt 重命名为 b.txt，内容未变。没有运行命令或测试。"

    class Model:
        calls = 0

        async def classify_intent(self, messages, message):
            return SemanticIntentAssessment(
                intent="answer", confidence="high", requires_clarification=False
            )

        async def decide(self, context):
            self.calls += 1
            assert context["current_request"] == question
            facts = context["audit_facts"]
            assert facts["recorded_command_count"] == 0
            assert facts["recent_operations"][0]["kind"] == "move"
            assert facts["recent_operations"][0]["payload"]["destination"] == "b.txt"
            return StudioReply(
                decision=StudioDecision(
                    action=StudioAction.RESPOND,
                    rationale="依据操作记录回答两个问题",
                    message=answer,
                )
            )

    session = StudioSession(
        session_id="history",
        repo_root=str(tmp_path),
        provider="deepseek",
        model="test",
        reasoning_effort="low",
        observations=[
            StudioObservation(
                kind="move",
                summary="a.txt → b.txt，内容不变",
                payload={"source": "a.txt", "destination": "b.txt"},
            )
        ],
    )
    model = Model()
    asyncio.run(StudioAgent(model, StudioStore(tmp_path / "state.db")).handle(session, question))
    assert model.calls == 1
    assert session.messages[-1].content == answer
    assert len(session.observations) == 1


@pytest.mark.parametrize(
    "intent,message,required,absent",
    [
        ("change", "把 a.txt 重命名为 b.txt", "implement", "launch"),
        ("answer", "讨论如何启动程序", "review", "launch"),
        ("verify", "检查一下", "verify", "implement"),
    ],
)
def test_plan_obeys_contract(intent, message, required, absent):
    contract = StudioTaskContract(objective=message, intent=intent)
    plan = StudioAgent._build_plan(VerificationMode.AUTO, contract, message)
    assert required in {p.key for p in plan}
    assert absent not in {p.key for p in plan}


def test_no_enabled_skills_does_not_scan(tmp_path):
    with patch("veripatch.studio_skills.root_for", side_effect=AssertionError("unexpected IO")):
        assert selected(str(tmp_path), []) == []


def test_pure_answer_skips_redundant_semantic_model_call(tmp_path):
    class Model:
        classify_calls = 0
        decide_calls = 0

        async def classify_intent(self, messages, message):
            self.classify_calls += 1
            raise AssertionError("pure answer should not classify twice")

        async def decide(self, context):
            self.decide_calls += 1
            return StudioReply(
                decision=StudioDecision(
                    action=StudioAction.RESPOND,
                    rationale="answer",
                    message="这是直接回答。",
                )
            )

    model = Model()
    session = StudioSession(
        session_id="fast-answer",
        repo_root=str(tmp_path),
        provider="deepseek",
        model="test",
        reasoning_effort="low",
    )
    asyncio.run(
        StudioAgent(model, StudioStore(tmp_path / "state.db")).handle(session, "什么是 Python？")
    )
    assert model.classify_calls == 0
    assert model.decide_calls == 1


def test_skill_snapshot_shared_across_classification_and_steps(tmp_path):
    content = "---\nname: test-skill\ndescription: test\n---\noriginal instructions"
    install(str(tmp_path), content)
    answer = "开场\n\n问题\n\n参考内容" + "内容" * 400

    class Model:
        calls = 0

        async def classify_intent(self, messages, message):
            assert "original instructions" in messages[0]["content"]
            (tmp_path / ".agents/skills/test-skill/SKILL.md").write_text(
                content.replace("original", "new"), encoding="utf-8"
            )
            return SemanticIntentAssessment(
                intent="answer", confidence="high", requires_clarification=False
            )

        async def decide(self, context):
            assert context["skills"]["selected"][0]["content"] == content
            self.calls += 1
            return StudioReply(
                decision=StudioDecision(
                    action=StudioAction.LIST_FILES if self.calls == 1 else StudioAction.RESPOND,
                    rationale="check",
                    message=answer if self.calls > 1 else None,
                )
            )

    session = StudioSession(
        session_id="snapshot",
        repo_root=str(tmp_path),
        provider="deepseek",
        model="test",
        reasoning_effort="low",
        enabled_skills=["test-skill"],
        messages=[
            StudioMessage(role="user", content="先介绍工作流"),
            StudioMessage(role="assistant", content="可以继续说明。"),
        ],
    )
    store = StudioStore(tmp_path / "state.db")
    with patch("veripatch.studio_skills.selected", wraps=selected) as loader:
        asyncio.run(StudioAgent(Model(), store).handle(session, "介绍一下这个工作流"))
        assert loader.call_count == 1
    assert session.messages[-1].content == answer
    assert "new instructions" in selected(str(tmp_path), ["test-skill"])[0]["content"]
