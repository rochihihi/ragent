import asyncio
from types import SimpleNamespace

import pytest

from veripatch.config import Settings
from veripatch.studio_domain import StudioDecision
from veripatch.studio_model import StudioProviderModel, _response_instructions


@pytest.mark.parametrize("mode", ["concise", "standard", "teaching"])
def test_current_style_reaches_decision_without_rewriting_answer(mode):
    class Replay(StudioProviderModel):
        async def _request_decision(self, prompt, compatible, instructions):
            assert instructions == _response_instructions({"response_style": {"mode": mode}})
            return SimpleNamespace(), StudioDecision(
                action="respond", rationale="回答问题", message="完整的题目与例子\n保留全部内容"
            )

    model = Replay("deepseek", Settings(), client=SimpleNamespace())
    reply = asyncio.run(model.decide({"response_style": {"mode": mode}}))
    assert reply.decision.message == "完整的题目与例子\n保留全部内容"


@pytest.mark.parametrize("provider,model_name", [
    ("deepseek", "deepseek-v4-flash"),
    ("deepseek", "deepseek-v4-pro"),
    ("openai", "gpt-test"),
])
def test_style_reaches_provider_payload(provider, model_name):
    payloads = []

    async def create(**kwargs):
        payloads.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content='{"action":"respond","rationale":"answer","message":"example"}',
            tool_calls=None,
        ))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    model = StudioProviderModel(provider, Settings(model=model_name, deepseek_model=model_name), client=client)
    model.base_url = "https://example.invalid/v1"
    model.api_key = None
    instructions = _response_instructions({"response_style": {"mode": "teaching"}})
    asyncio.run(model._request_decision("context", "context", instructions))
    assert any(instructions in msg["content"] for msg in payloads[0]["messages"])
