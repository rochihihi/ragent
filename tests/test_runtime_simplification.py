import asyncio
from unittest.mock import patch

import pytest

from veripatch.studio_agent import StudioAgent
from veripatch.studio_domain import (
    StudioAction,
    StudioDecision,
    StudioMessage,
    StudioObservation,
    StudioReply,
    StudioSession,
    StudioTaskContract,
)
from veripatch.studio_skills import activate, install, selected
from veripatch.studio_store import StudioStore


@pytest.mark.parametrize(
    "kind,payload",
    [
        ("read", {"path": "app.py", "content": "print(1)"}),
        ("command", {"launch_state": "dispatched", "exit_code": 0}),
        ("test", {"exit_code": 1, "stderr": "temporary failure"}),
        ("tool_error", {"error": "temporary failure", "next_strategy": "旧策略建议"}),
        ("verification_gate", {"unmet": ["尚无测试通过证据"]}),
    ],
)
def test_tool_facts_reach_model_without_local_next_action_instruction(tmp_path, kind, payload):
    class Model:
        contexts = []

        async def decide(self, context):
            self.contexts.append(context)
            assert "instruction" not in context
            assert context["latest_tool_result"]["kind"] == kind
            return StudioReply(decision=StudioDecision(
                action=StudioAction.RESPOND, rationale="依据结果决定回答", message="已查看结果。",
            ))

    session = StudioSession(
        session_id="facts-only", repo_root=str(tmp_path), provider="deepseek",
        model="test", reasoning_effort="low",
        observations=[StudioObservation(kind=kind, summary="实际工具结果", payload=payload)],
    )
    model = Model()
    agent = StudioAgent(model, StudioStore(tmp_path / "state.db"))
    asyncio.run(agent.handle(session, "说明结果", continuation=True))
    assert len(model.contexts) == 1
    context = model.contexts[0]
    assert context["latest_tool_result"]["kind"] == kind
    assert session.messages[-1].content == "已查看结果。"
    for packet in (
        agent._recovery_context(session, context),
        agent._compact_retry_context(context),
    ):
        assert "instruction" not in packet
        assert packet["latest_tool_result"]["kind"] == kind
        assert packet["verification"] == context["verification"]


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
            raise AssertionError("Independent intent classification must not run")

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
def test_new_session_does_not_generate_a_local_plan(intent, message, required, absent):
    contract = StudioTaskContract(objective=message, intent=intent)
    session = StudioSession(session_id="plan", repo_root=".", provider="openai",
                            model="test", reasoning_effort="low", task_contract=contract)
    assert session.plan == []
    assert not hasattr(StudioAgent, "_build_plan")


def test_no_enabled_skills_does_not_scan(tmp_path):
    with patch("veripatch.studio_skills.root_for", side_effect=AssertionError("unexpected IO")):
        assert selected(str(tmp_path), []) == []


def test_primary_model_can_answer_when_auxiliary_classifier_is_unavailable(tmp_path):
    class Model:
        classify_calls = 0
        decide_calls = 0

        async def classify_intent(self, messages, message):
            raise AssertionError("Independent intent classification must not run")

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


def test_skill_snapshot_shared_across_execution_steps(tmp_path):
    content = "---\nname: test-skill\ndescription: test\n---\noriginal instructions"
    install(str(tmp_path), content)
    answer = "开场\n\n问题\n\n参考内容" + "内容" * 400

    class Model:
        calls = 0

        async def classify_intent(self, messages, message):
            raise AssertionError("Independent intent classification must not run")

        async def decide(self, context):
            assert context["skills"]["selected"][0]["content"] == content
            self.calls += 1
            if self.calls == 1:
                (tmp_path / ".agents/skills/test-skill/SKILL.md").write_text(
                    content.replace("original", "new"), encoding="utf-8"
                )
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
    with patch("veripatch.studio_skills.activate", wraps=activate) as loader:
        asyncio.run(StudioAgent(Model(), store).handle(session, "介绍一下这个工作流"))
        assert loader.call_count == 1
    assert session.messages[-1].content == answer
    assert "new instructions" in selected(str(tmp_path), ["test-skill"])[0]["content"]
