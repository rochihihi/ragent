import asyncio
from types import SimpleNamespace

import pytest

from veripatch.config import Settings
from veripatch.studio_domain import StudioDecision
from veripatch.studio_model import StudioProviderModel


@pytest.mark.parametrize("policy", [
    {"task_contract": {"intent": "answer"}},
    {"task_contract": {"intent": "analysis"}},
    {"task_state": {"verification_policy": "skipped_by_user"}},
    {"task_contract": {"denied_actions": ["run_command", "run_tests"]}},
])
def test_valid_status_reply_is_accepted_without_retries(policy):
    class Replay(StudioProviderModel):
        calls = 0

        async def _request_decision(self, *args):
            self.calls += 1
            return SimpleNamespace(), StudioDecision(
                action="respond", rationale="汇报现有结果",
                message="hello.py 添加了注释，没有运行测试命令。",
            )

    model = Replay("deepseek", Settings(), client=SimpleNamespace())
    reply = asyncio.run(model.decide({
        "changed_files": ["hello.py"],
        "verification": {"verification_passed": False}, **policy,
    }))
    assert reply.decision.action.value == "respond"
    assert model.calls == 1


def test_actual_tool_unavailable_claim_is_still_detected():
    decision = StudioDecision(action="respond", rationale="无法继续",
                              message="没有工作区工具，无法读取。")
    assert StudioProviderModel._is_false_tool_unavailable_failure(decision)
