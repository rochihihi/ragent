import asyncio
from types import SimpleNamespace

import pytest

from veripatch.config import Settings
from veripatch.domain import ActionKind, AgentDecision, IssueSpec
from veripatch.models.base import ModelContext
from veripatch.models.openai import OpenAIResponsesModel


class FakeResponses:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.kwargs = None

    async def parse(self, **kwargs):
        self.kwargs = kwargs
        if self.fail:
            raise OSError("network down")
        return SimpleNamespace(
            id="resp_openai",
            model="gpt-5.6-terra-202608",
            output_parsed=AgentDecision(
                action=ActionKind.SEARCH,
                rationale="Find the implementation.",
                query="calculate_discount",
            ),
            usage=SimpleNamespace(
                input_tokens=123,
                output_tokens=45,
                input_tokens_details=SimpleNamespace(cached_tokens=23),
                output_tokens_details=SimpleNamespace(reasoning_tokens=7),
            ),
        )


def _context() -> ModelContext:
    return ModelContext(
        issue=IssueSpec(issue_id="1", title="Bug", description="Find and repair it"),
        step=1,
        changed_files=[],
        index_summary={"symbols": 1},
        recent_observations=[],
    )


def test_openai_adapter_uses_structured_responses_and_tracks_usage() -> None:
    responses = FakeResponses()
    client = SimpleNamespace(responses=responses)
    model = OpenAIResponsesModel(Settings(), client=client)
    reply = asyncio.run(model.decide(_context()))
    assert reply.decision.action is ActionKind.SEARCH
    assert reply.input_tokens == 123
    assert reply.output_tokens == 45
    assert reply.cached_input_tokens == 23
    assert reply.reasoning_tokens == 7
    assert reply.model == "gpt-5.6-terra-202608"
    assert reply.request_id == "resp_openai"
    assert responses.kwargs["model"] == "gpt-5.6-terra"
    assert responses.kwargs["reasoning"] == {"effort": "medium"}
    assert responses.kwargs["text_format"] is AgentDecision
    assert "Simplified Chinese" in responses.kwargs["input"][0]["content"]


def test_openai_adapter_wraps_provider_errors() -> None:
    client = SimpleNamespace(responses=FakeResponses(fail=True))
    model = OpenAIResponsesModel(Settings(), client=client)
    with pytest.raises(RuntimeError, match="Responses API failed"):
        asyncio.run(model.decide(_context()))


def test_openai_adapter_rejects_missing_structured_output() -> None:
    responses = FakeResponses()

    async def no_output(**kwargs):
        return SimpleNamespace(output_parsed=None, usage=None)

    responses.parse = no_output
    model = OpenAIResponsesModel(Settings(), client=SimpleNamespace(responses=responses))
    with pytest.raises(RuntimeError, match="no structured"):
        asyncio.run(model.decide(_context()))
