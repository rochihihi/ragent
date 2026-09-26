import asyncio

from veripatch.studio_agent import StudioAgent
from veripatch.studio_domain import SemanticIntentAssessment, StudioSession
from veripatch.studio_store import StudioStore


def test_single_digit_reaches_model_and_asks_specific_question(tmp_path):
    class Model:
        async def classify_intent(self, messages, message):
            assert message == "1"
            return SemanticIntentAssessment(
                intent="answer", confidence="low", requires_clarification=True,
                clarification_question="你说的 1 指哪一项？", rationale="没有待选项",
            )

        async def decide(self, context):
            raise AssertionError("Must wait for the missing reference")

    session = StudioSession(
        session_id="digit", repo_root=str(tmp_path), provider="deepseek",
        model="deepseek-v4-flash", reasoning_effort="low",
    )
    result = asyncio.run(StudioAgent(Model(), StudioStore(tmp_path / "state.db")).handle(
        session, "1",
    ))
    assert result.activity == "waiting_user"
    assert result.messages[-1].content == "你说的 1 指哪一项？"


def test_edit_permission_does_not_supply_missing_objective(tmp_path):
    class Model:
        async def classify_intent(self, messages, message):
            raise AssertionError("Missing objective must be caught before model inference")

        async def decide(self, context):
            raise AssertionError("Must not invent an edit")

    target = tmp_path / "hello.py"
    target.write_text("VERSION = '1.0'\n", encoding="utf-8")
    session = StudioSession(
        session_id="missing-objective", repo_root=str(tmp_path), provider="deepseek",
        model="deepseek-v4-flash", reasoning_effort="low",
    )
    result = asyncio.run(StudioAgent(Model(), StudioStore(tmp_path / "state.db")).handle(
        session, "修改 hello.py，不要运行命令。",
    ))
    assert result.activity == "waiting_user"
    assert result.messages[-1].content == "hello.py 具体要改什么？"
    assert target.read_text(encoding="utf-8") == "VERSION = '1.0'\n"
